'''
OpenPi Policy Integration for Dual-Arm Robot Control
The policy must be trained in the robot coordinate system.
Author: Wang Zheng
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from policy.base import BasePolicy
import numpy as np
import copy
from typing import Dict, List, Any, Optional, Tuple, Set
try:
    import torch
except ImportError:
    print("torch not found")
    pass

# OpenPI imports
try:
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
except ImportError:
    print("OpenPI not found")
    pass

from utils.logger_config import logger
import scipy.spatial.transform as st
import time
from utils.dsl_pose_utils import pose_to_mat, mat_to_pose, convert_pose_mat_rep
from utils.calibration_utils import load_tcp_umi_transforms
from utils.transform import process_quaternion
from utils.latency_tracker import LatencyTracker
from scipy.spatial.transform import Rotation


class OpenPiAlohaPolicy(BasePolicy):
    def __init__(
        self,
        pi_config_name: str,
        ckpt_dir: str,
        task: str = "",
        device: str = 'cuda',
        obs_horizon: int = 1,
        chunk_length: int = 100,
        downsample: int = 1,
        obs_time_interval: float = 0.04,
        execution_fps: Optional[float] = None,
        enable_rtc: bool = False,
    ) -> None:
        super().__init__()
        self.pi_config_name = pi_config_name
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.device = device
        self.obs_horizon = obs_horizon
        self.chunk_length = chunk_length
        self.downsample = downsample
        self.obs_time_interval = obs_time_interval
        self.execution_fps = execution_fps
        self.enable_rtc = enable_rtc

        self._latency_tracker: LatencyTracker = LatencyTracker(maxlen=100)
        self._policy = self.load_openpi_policy(self.pi_config_name, self.ckpt_dir, self.device)

    def reset(self) -> None:
        """
        Reset the policy's internal state for a new episode.
        
        This clears the cached previous actions and state used by RTC,
        ensuring the policy doesn't use stale data from the previous episode.
        """
        if hasattr(self._policy, '_prev_actions'):
            self._policy._prev_actions = None
            logger.info("OpenPiPolicy: Cleared _prev_actions for RTC reset")
        if hasattr(self._policy, '_prev_state_abs'):
            self._policy._prev_state_abs = None
            logger.info("OpenPiPolicy: Cleared _prev_state_abs for RTC reset")

    def load_openpi_policy(self, pi_config_name: str, ckpt_dir: str, device='cuda'):
        self.pi_config = _config.get_config(pi_config_name)
        print(f"Loading policy from {ckpt_dir}")
        policy = _policy_config.create_trained_policy(self.pi_config, ckpt_dir)
        return policy

    def _convert_obs_to_pi(self, obs_batch_robot: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        state = obs_batch_robot['robot_joint_pos'].astype(np.float32)
        ego_image = obs_batch_robot['camera0_rgb']
        left_wrist_image = obs_batch_robot['camera1_rgb']
        right_wrist_image = obs_batch_robot['camera2_rgb']

        repacked_data = { # Keep the observation horizon dimension if > 1
            "observation/base_image": ego_image.squeeze(0),
            "observation/left_wrist_image": left_wrist_image.squeeze(0),
            "observation/right_wrist_image": right_wrist_image.squeeze(0),
            "observation/state": state.squeeze(0),
            "prompt": self.task,
            # "advantage": np.array([1.0], dtype=np.float32) # For Pi0.6 inference
        }
        repacked_data = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in repacked_data.items()}
        return repacked_data

    def predict(self, observation, action_configs: List[Dict[str, Any]], chunk_length: int = 1, action_index: Optional[int] = None) -> Dict[str, np.ndarray]:
        # 1. Organize observation data for the policy
        obs_batch_robot = self.sensor_data_to_umi_data(observation)
        obs_batch_for_policy = self._convert_obs_to_pi(obs_batch_robot)

        # Prepare rtc_config if RTC is enabled and action_index is provided
        rtc_config: Optional[Dict[str, float | int]] = None
        if self.enable_rtc and action_index is not None:
            # Compute inference_delay dynamically from tracked latencies
            default_delay_steps = 10
            if self.execution_fps is not None and self.execution_fps > 0:
                inference_delay = self._latency_tracker.compute_inference_delay_steps(
                    execution_fps=self.execution_fps,
                    use_percentile=0.95,
                    default_delay_steps=default_delay_steps
                )
            else:
                inference_delay = default_delay_steps
            # logger.info(f"Inference delay: {inference_delay}")
            # Compute prefix_attention_horizon based on inference_delay and action_index
            prefix_attention_horizon = chunk_length - inference_delay - action_index

            rtc_config = {
                "inference_delay": inference_delay,
                "prefix_attention_horizon": prefix_attention_horizon,
                "max_guidance_weight": 10.0
            }

        with torch.no_grad():
            inference_delay = int(rtc_config.get("inference_delay", 0)) if rtc_config is not None else 0
            if self.enable_rtc and action_index is not None and rtc_config is not None and (action_index + inference_delay < chunk_length):
                model_outputs = self._policy.infer(
                    obs_batch_for_policy,
                    time_base=action_index,
                    rtc_config=rtc_config
                )
                robot_actions = model_outputs["actions"]
            else:
                if self.enable_rtc and action_index is not None and rtc_config is not None and (action_index + inference_delay >= chunk_length):
                    # For RTC, we need: action_index + inference_delay < chunk_length.
                    # A practical lower bound for policy fps is:
                    #   fps > execution_fps / (chunk_length - M - inference_delay)
                    # where M is a safety margin in steps.
                    rtc_chunk_length: int = chunk_length
                    recommended_fps_range_str: str = ""
                    if self.execution_fps is not None and self.execution_fps > 0:
                        exec_fps: float = float(self.execution_fps)
                        # Use inference_delay as the minimum safety margin:
                        # For the very first inference, no action chunk is being executed yet,
                        # so the effective "execution consumed during inference" can be smaller.
                        # Using m_min=inference_delay provides a conservative lower-bound.
                        m_min: int = max(int(inference_delay), 0)
                        m_max: int = max(chunk_length-inference_delay-1, m_min)

                        def _min_required_policy_fps(margin_steps: int) -> Optional[float]:
                            denom: int = rtc_chunk_length - int(margin_steps) - int(inference_delay)
                            if denom <= 0:
                                return None
                            return exec_fps / float(denom)

                        fps_lb_m2 = _min_required_policy_fps(m_min)
                        fps_lb_mmax = _min_required_policy_fps(m_max)
                        if fps_lb_m2 is not None and fps_lb_mmax is not None:
                            recommended_fps_range_str = (
                                f", recommended policy.fps lower-bound for M={m_min}..{m_max}: "
                                f"{fps_lb_m2:.3f}..{fps_lb_mmax:.3f} Hz "
                                f"(execution_fps={exec_fps:.1f} Hz, chunk_length={rtc_chunk_length}, inference_delay={inference_delay})"
                            )
                        else:
                            recommended_fps_range_str = (
                                f", cannot compute recommended fps (execution_fps={exec_fps:.1f} Hz, "
                                f"chunk_length={rtc_chunk_length}, inference_delay={inference_delay}, M in [{m_min},{m_max}])"
                            )
                    logger.warning(
                        f"FPS seems too low for RTC, fallback to normal infer(): "
                        f"action_index={action_index}, inference_delay={inference_delay}, "
                        f"action_index+inference_delay={action_index + inference_delay} (>=chunk_length)"
                        f"{recommended_fps_range_str}."
                    )
                model_outputs = self._policy.infer(obs_batch_for_policy)
                robot_actions = model_outputs["actions"] # absolute joint positions

        return {action_configs[0]['device_name']: robot_actions.astype(np.float64)}

    def predict_value(self, observation: Dict[str, np.ndarray], predicted_action: np.ndarray) -> Dict[str, np.ndarray]:
        obs_batch_robot = self.sensor_data_to_umi_data(observation)
        obs_batch_for_policy = self._convert_obs_to_pi(obs_batch_robot)
        processed_action = torch.from_numpy(predicted_action).unsqueeze(0)
        with torch.no_grad():
            outputs = self._policy.infer_value(obs_batch_for_policy, processed_action)
            value = outputs["value"]
        return {"value": value.astype(np.float64)}

    def sensor_data_to_umi_data(self, obs_dict: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # Validate input format based on configured obs_horizon
        use_history = self.obs_horizon > 1

        if use_history:
            # History mode: each value is a list of obs_horizon arrays
            for key, value in obs_dict.items():
                assert isinstance(value, list), f"Expected list for key '{key}', got {type(value)}"
                assert len(value) == self.obs_horizon, f"Expected {self.obs_horizon} items, got {len(value)}"
        else:
            # Single observation mode: each value is a single array (or list with 1 element)
            obs_dict = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in obs_dict.items()}

        batch_data: Dict[str, np.ndarray] = {}

        # Process camera observations
        cameras = [name for name in obs_dict.keys() if 'camera' in name.lower()]
        for idx, camera_name in enumerate(cameras):
            if use_history:
                # Stack all observations in history
                raw_images = obs_dict[camera_name]
                stacked_images = np.stack(raw_images)
                batch_data[f'camera{idx}_rgb'] = (stacked_images / 255.0).transpose(0, 3, 1, 2)  # (obs_horizon, C, H, W)
            else:
                raw_image = obs_dict[camera_name]
                stacked_images = np.stack(raw_image)[None, :]
                batch_data[f'camera{idx}_rgb'] = (stacked_images / 255.0).transpose(0, 3, 1, 2)  # (1, C, H, W)

        # Process robot observation
        robot_device = next(iter([key for key in obs_dict.keys() if 'robot' in key.lower()]), None)
        if robot_device is None:
            raise KeyError("Robot observation data not found in observation dictionary")

        if use_history:
            robot_obs_list = obs_dict[robot_device]
            robot_obs_array = np.stack(robot_obs_list)  # (obs_horizon, obs_dim)
        else:
            robot_obs = obs_dict[robot_device]
            assert isinstance(robot_obs, np.ndarray)
            robot_obs_array = robot_obs if robot_obs.ndim > 1 else robot_obs[None, :]

        # Validate observation dimension
        obs_dim = robot_obs_array.shape[-1]
        expected_obs_dim = 14
        if obs_dim != expected_obs_dim:
            raise ValueError(f"Robot observation dimension {obs_dim} does not match expected "
                             f"{expected_obs_dim} for {self.num_robots} robot(s)")

        batch_data['robot_joint_pos'] = robot_obs_array
        return batch_data

    def warm_up(
        self,
        action_configs: List[Dict[str, Any]],
        num_observations: int = 10,
        sample_observation: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Warm up the model by running inference on observations.
        If a sample observation is provided (from PolicyConnector), use it to
        align warm-up input dimensions with the real inference path.
        First 5 observations use standard inference (no RTC), next 5 use RTC.
        
        The first call of each inference mode (no RTC and with RTC) is excluded
        from the latency tracker to avoid counting JIT compilation time.
        
        Args:
            action_configs: Action configuration list for predict method
            num_observations: Total number of warm-up observations (default: 10)
            sample_observation: Optional observation batch to mirror real data
        """
        logger.info(f"Starting warm-up with {num_observations} observations")

        if sample_observation is not None:
            observation = sample_observation
            logger.info("Warm-up uses provided sample observation from PolicyConnector")
        else:
            observation = {
                "OpenCVCameraDevice_0": [
                    np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
                    for _ in range(self.obs_horizon)
                ],
                "OpenCVCameraDevice_1": [
                    np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                    for _ in range(self.obs_horizon)
                ], 
                "OpenCVCameraDevice_2": [
                    np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                    for _ in range(self.obs_horizon)
                ],
                "I2RTRobot_1": [
                    np.random.rand(14).astype(np.float64)
                    for _ in range(self.obs_horizon)
                ],
            }
        
        # First 5 observations: without RTC (action_index=None)
        # The first call (i=0) is excluded from latency tracking to skip JIT overhead
        logger.info("Warming up with standard inference (no RTC)...")
        for i in range(min(5, num_observations)):
            start_time = time.time()
            try:
                _ = self.predict(
                    observation,
                    action_configs, 
                    chunk_length=self.chunk_length, 
                    action_index=None
                )
                inference_time = time.time() - start_time
                # Skip the first call (i=0) for latency tracking to exclude JIT compilation time
                if i > 0:
                    self._latency_tracker.add(inference_time)
                logger.info(f"Warm-up observation {i+1}/5 (no RTC): {inference_time*1000:.2f} ms"
                           + (" [excluded from latency tracker]" if i == 0 else ""))
            except Exception as e:
                logger.error(f"Warm-up observation {i+1}/5 failed: {e}")
        
        # Next 5 observations: with RTC (action_index=0) if RTC is enabled
        # The first RTC call (i=5) is excluded from latency tracking to skip JIT overhead
        if self.enable_rtc:
            logger.info("Warming up with RTC...")
            first_rtc_call = True
            for i in range(5, min(10, num_observations)):
                start_time = time.time()
                try:
                    # Use a valid action_index (0 is safe as it's the first index)
                    _ = self.predict(
                        observation,
                        action_configs, 
                        chunk_length=self.chunk_length, 
                        action_index=0
                    )
                    inference_time = time.time() - start_time
                    # Skip the first RTC call for latency tracking to exclude JIT compilation time
                    if not first_rtc_call:
                        self._latency_tracker.add(inference_time)
                    logger.info(f"Warm-up observation {i + 1}/5 (RTC): {inference_time * 1000:.2f} ms"
                                + (" [excluded from latency tracker]" if first_rtc_call else ""))
                    first_rtc_call = False
                except Exception as e:
                    logger.error(f"Warm-up observation {i+1}/10 failed: {e}")

        logger.info("Warming up value prediction...")
        action_dim = action_configs[0].get('action_dim')
        if action_dim is None:
            logger.warning("Value warm-up skipped: action_configs[0]['action_dim'] is missing")
        else:
            predicted_action = np.zeros((self.chunk_length, action_dim), dtype=np.float64)
            for i in range(min(5, num_observations)):
                start_time = time.time()
                try:
                    _ = self.predict_value(copy.deepcopy(observation), predicted_action)
                    inference_time = time.time() - start_time
                    logger.info(f"Warm-up value prediction {i+1}/5: {inference_time*1000:.2f} ms")
                except Exception as e:
                    logger.error(f"Warm-up value prediction {i+1}/5 failed: {e}")
        
        # Log latency tracker statistics after warm-up
        if len(self._latency_tracker) > 0:
            logger.info(
                f"Warm-up latency statistics: count={len(self._latency_tracker)}, "
                f"mean={self._latency_tracker.mean()*1000:.2f} ms, "
                f"p95={self._latency_tracker.p95()*1000:.2f} ms, "
                f"max={self._latency_tracker.max()*1000:.2f} ms"
            )
            if self.execution_fps is not None and self.execution_fps > 0:
                computed_delay = self._latency_tracker.compute_inference_delay_steps(
                    execution_fps=self.execution_fps,
                    use_percentile=0.95,
                    default_delay_steps=6
                )
                logger.info(f"Computed inference_delay based on p95 latency: {computed_delay} steps "
                           f"(execution_fps={self.execution_fps:.1f} Hz)")
        
        logger.info("Warm-up completed")
