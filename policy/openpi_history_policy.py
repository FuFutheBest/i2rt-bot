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
from skimage.transform import resize
import scipy.spatial.transform as st
import time
from utils.dsl_pose_utils import pose_to_mat, mat_to_pose, convert_pose_mat_rep
from utils.calibration_utils import load_tcp_umi_transforms
from utils.transform import process_quaternion
from utils.latency_tracker import LatencyTracker
from scipy.spatial.transform import Rotation


def center_crop_square_resize(
    img: np.ndarray,
    crop_min_ratio: float = 0.05,
    crop_max_ratio: float = 0.95,
    tgt: int = 224,
) -> np.ndarray:
    """
    img: (H, W, C)  uint8
    1) center-crop to square based on the shorter edge
    2) crop to [crop_min_ratio, crop_max_ratio] range of the square on both axes
    3) resize → tgt×tgt
    """
    H, W, C = img.shape
    # Center-crop to square based on the shorter edge
    shorter_edge = min(H, W)
    h_off = (H - shorter_edge) // 2
    w_off = (W - shorter_edge) // 2
    img = img[h_off:h_off + shorter_edge, w_off:w_off + shorter_edge]

    # Dynamic crop of the square dimension (e.g., [15:285] when size==300 and ratios are 0.05/0.95)
    margin = int(shorter_edge * crop_min_ratio)
    end = int(shorter_edge * crop_max_ratio)
    if end > margin:
        img = img[margin:end, margin:end]

    img = resize(img, (tgt, tgt), preserve_range=True, anti_aliasing=True).astype(
        np.uint8
    )
    return img

def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    quat: (..., 4)  [x, y, z, w]
    return (..., 3) axis-angle (OpenCV convention, norm = angle)
    """
    rot = st.Rotation.from_quat(quat[..., [0, 1, 2, 3]])  # scipy: [x,y,z,w]
    return rot.as_rotvec()

def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """
    axis_angle: (..., 3)
    return (..., 4) quat
    """
    rot = st.Rotation.from_rotvec(axis_angle)
    return rot.as_quat()

class OpenPiHistoryPolicy(BasePolicy):
    def __init__(
        self,
        pi_config_name: str,
        ckpt_dir: str,
        task: str = "",
        device: str = 'cuda',
        obs_horizon: int = 1,
        downsample: int = 1,
        obs_time_interval: float = 0.04,
        execution_fps: Optional[float] = None,
        enable_rtc: bool = False,
        tcp_umi_json_path: Optional[str] = None,
        num_robots: int = 2,
        chunk_length: int = 50,
    ) -> None:
        super().__init__()
        self.pi_config_name = pi_config_name
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.device = device
        self.obs_horizon = obs_horizon
        self.downsample = downsample
        self.obs_time_interval = obs_time_interval
        self.execution_fps = execution_fps
        self.enable_rtc = enable_rtc
        if num_robots not in (1, 2):
            raise ValueError("num_robots must be 1 or 2")
        self.num_robots = num_robots
        self.chunk_length = chunk_length
        self._latency_tracker: LatencyTracker = LatencyTracker(maxlen=self.chunk_length)
        self._policy = self.load_openpi_policy(self.pi_config_name, self.ckpt_dir, self.device)

        if tcp_umi_json_path is not None:
            self.tcp_umi_T, self.umi_tcp_T = self._load_calibration(tcp_umi_json_path)
        else:
            self.tcp_umi_T, self.umi_tcp_T = None, None

    def _load_calibration(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        tcp_umi, umi_tcp = load_tcp_umi_transforms(json_path)
        logger.info(f"Loaded TCP↔UMI calibration from {json_path}")
        return tcp_umi, umi_tcp

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

    def _convert_obs_to_pi(self, obs_batch_robot: Dict[str, np.ndarray], for_action_policy: bool = True) -> Dict[str, np.ndarray]:
        def squeeze_single_batch(value: np.ndarray) -> np.ndarray:
            return value.squeeze(0) if value.shape[0] == 1 else value

        model_config = getattr(self.pi_config, "model", None)
        action_observation_horizon = int(getattr(model_config, "action_observation_horizon", None))
        value_observation_horizon = int(getattr(model_config, "value_observation_horizon", None))
        use_action_observation_history = for_action_policy and (action_observation_horizon > 1)
        use_value_observation_history = not for_action_policy and (value_observation_horizon > 1)

        left_pos = obs_batch_robot["robot0_eef_pos"]
        left_axis_angle = obs_batch_robot["robot0_eef_rot_axis_angle"]
        left_gripper = obs_batch_robot["robot0_gripper_width"]

        right_pos = obs_batch_robot.get("robot1_eef_pos")
        right_axis_angle = obs_batch_robot.get("robot1_eef_rot_axis_angle")
        right_gripper = obs_batch_robot.get("robot1_gripper_width")

        # For single-arm, pad zeros for right arm since OpenPi expects bimanual input
        if self.num_robots == 1 or right_pos is None or right_axis_angle is None or right_gripper is None:
            zeros_like_pos = np.zeros_like(left_pos)
            zeros_like_rot = np.zeros_like(left_axis_angle)
            zeros_like_gripper = np.zeros_like(left_gripper)
            right_pos = right_pos if right_pos is not None else zeros_like_pos
            right_axis_angle = right_axis_angle if right_axis_angle is not None else zeros_like_rot
            right_gripper = right_gripper if right_gripper is not None else zeros_like_gripper

        # Handle batched data (obs_horizon > 1) vs single observation
        # Both cases: concatenate along axis=-1 works for both (obs_horizon, dim) and (dim,) shapes
        left_state = np.concatenate([left_pos, left_axis_angle, left_gripper], axis=-1)
        right_state = np.concatenate([right_pos, right_axis_angle, right_gripper], axis=-1)
        state = np.concatenate([left_state, right_state], axis=-1).astype(np.float32)

        left_wrist_image = obs_batch_robot['camera0_rgb']
        right_wrist_key = 'camera1_rgb'
        right_wrist_image = obs_batch_robot.get(right_wrist_key)

        repacked_data = { # Keep the observation horizon dimension if > 1
            "observation/left_wrist_image": squeeze_single_batch(left_wrist_image),
            "observation/right_wrist_image": squeeze_single_batch(right_wrist_image),
            "observation/state": squeeze_single_batch(state),
            "prompt": self.task,
            # "advantage": np.array([1.0], dtype=np.float32) # For Pi0.6 inference
        }
        if for_action_policy:
            value_prediction = obs_batch_robot['value_prediction']
            repacked_data['value_prediction'] = squeeze_single_batch(value_prediction)

        if use_action_observation_history or use_value_observation_history:
            # These source keys are internal names produced by sensor_data_to_umi_data().
            # The target keys are the OpenPI data-transform contract used by AsyncSequenceAlohaInputs.
            state_is_pad = obs_batch_robot['state_is_pad']
            left_wrist_image_is_pad = obs_batch_robot['camera0_rgb_is_pad']
            right_wrist_image_is_pad = obs_batch_robot['camera1_rgb_is_pad']
            mask_key_map = {
                "observation/left_wrist_image_is_pad": squeeze_single_batch(left_wrist_image_is_pad),
                "observation/right_wrist_image_is_pad": squeeze_single_batch(right_wrist_image_is_pad),
                "observation/state_is_pad": squeeze_single_batch(state_is_pad),
            }
            if for_action_policy:
                value_prediction_is_pad = obs_batch_robot['value_prediction_is_pad']
                mask_key_map['value_prediction_is_pad'] = squeeze_single_batch(value_prediction_is_pad)

            # Make is_pads: Here all False means the corresponding observation is not 'paded', and won't be masked in the 'input mask' of the OpenPI model
            for target_key, value in mask_key_map.items():
                if value is None:
                    logger.warning(f"we use observation history, but {target_key} is not found in the observation dictionary")
                else:  
                    repacked_data[target_key] = squeeze_single_batch(value)
        repacked_data = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in repacked_data.items()}
        return repacked_data

    def predict(self, observation, action_configs: List[Dict[str, Any]], chunk_length: int = 50, action_index: Optional[int] = None) -> Dict[str, np.ndarray]:
        # 1. Organize observation data for the policy
        obs_batch_robot, robot_state = self.sensor_data_to_umi_data(observation, target_frame="robot", for_action_policy=True) # obs_batch_robot is in the **umi** coordinate system
        obs_batch_for_policy = self._convert_obs_to_pi(obs_batch_robot, for_action_policy=True)

        # Prepare rtc_config if RTC is enabled and action_index is provided
        rtc_config: Optional[Dict[str, float | int]] = None
        if self.enable_rtc and action_index is not None:
            # Compute inference_delay dynamically from tracked latencies
            default_delay_steps = 4
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
                robot_rel_action = model_outputs["actions"]
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
                robot_rel_action = model_outputs["actions"] # action is the absolute pose in the **umi** coordinate system

        robot_rel_action = np.asarray(robot_rel_action)
        if robot_rel_action.ndim == 1:
            robot_rel_action = robot_rel_action[None, :]
        action_dim = robot_rel_action.shape[-1]
        if action_dim < 7:
            raise ValueError(f"OpenPi policy returned action dimension {action_dim}, expected at least 7")

        # Validate action dimension matches configured num_robots
        expected_action_dim = 7 * self.num_robots  # 7 for single, 14 for dual
        if action_dim != expected_action_dim:
            raise ValueError(f"Action dimension {action_dim} does not match expected "
                             f"{expected_action_dim} for {self.num_robots} robot(s)")

        # Process actions based on num_robots
        if self.num_robots == 1:
            # Single robot: process left arm only
            robot_current_pose_left = robot_state[:6]
            robot_current_mat_left = pose_to_mat(robot_current_pose_left)

            robot_rel_action_pos_left, robot_rel_action_rpy_left, robot_rel_action_gripper = robot_rel_action[...,:3], robot_rel_action[...,3:6], robot_rel_action[...,6:7]
            robot_rel_action_axis_angle_left = Rotation.from_euler('xyz', robot_rel_action_rpy_left, degrees=False).as_rotvec()
            robot_rel_action_pose_left = np.concatenate([robot_rel_action_pos_left, robot_rel_action_axis_angle_left], axis=-1)
            robot_rel_action_mat_left = pose_to_mat(robot_rel_action_pose_left)
            robot_abs_action_mat_left = convert_pose_mat_rep(robot_rel_action_mat_left, robot_current_mat_left, pose_rep='relative', backward=True)
            robot_abs_action_pose_left = mat_to_pose(robot_abs_action_mat_left)
            left_abs_action = np.concatenate([robot_abs_action_pose_left, robot_rel_action_gripper], axis=-1)

            single_pose = self.action_to_single_pose(left_abs_action)
            single_pose[:, 7] = np.clip(single_pose[:, 7], 0.0, 0.08)
            return {action_configs[0]['device_name']: single_pose.astype(np.float64)}
        else:
            # Dual robots: process both arms
            robot_current_pose_left = robot_state[:6]
            robot_current_mat_left = pose_to_mat(robot_current_pose_left)

            robot_rel_action_pos_left, robot_rel_action_rpy_left, robot_rel_action_gripper = robot_rel_action[...,:3], robot_rel_action[...,3:6], robot_rel_action[...,6:7]
            robot_rel_action_axis_angle_left = Rotation.from_euler('xyz', robot_rel_action_rpy_left, degrees=False).as_rotvec()
            robot_rel_action_pose_left = np.concatenate([robot_rel_action_pos_left, robot_rel_action_axis_angle_left], axis=-1)
            robot_rel_action_mat_left = pose_to_mat(robot_rel_action_pose_left)
            robot_abs_action_mat_left = convert_pose_mat_rep(robot_rel_action_mat_left, robot_current_mat_left, pose_rep='relative', backward=True)
            robot_abs_action_pose_left = mat_to_pose(robot_abs_action_mat_left)
            left_abs_action = np.concatenate([robot_abs_action_pose_left, robot_rel_action_gripper], axis=-1)

            robot_current_pose_right = robot_state[7:13]
            robot_current_mat_right = pose_to_mat(robot_current_pose_right)

            robot_rel_action_pos_right, robot_rel_action_rpy_right, robot_rel_action_gripper_right = robot_rel_action[...,7:10], robot_rel_action[...,10:13], robot_rel_action[...,13:14]
            robot_rel_action_axis_angle_right = Rotation.from_euler('xyz', robot_rel_action_rpy_right, degrees=False).as_rotvec()
            robot_rel_action_pose_right = np.concatenate([robot_rel_action_pos_right, robot_rel_action_axis_angle_right], axis=-1)
            robot_rel_action_mat_right = pose_to_mat(robot_rel_action_pose_right)
            robot_abs_action_mat_right = convert_pose_mat_rep(robot_rel_action_mat_right, robot_current_mat_right, pose_rep='relative', backward=True)
            robot_abs_action_pose_right = mat_to_pose(robot_abs_action_mat_right)
            right_abs_action = np.concatenate([robot_abs_action_pose_right, robot_rel_action_gripper_right], axis=-1)

            robot_abs_action = np.concatenate([left_abs_action, right_abs_action], axis=-1)
            dual_pose = self.action_to_dual_pose(robot_abs_action)
            dual_pose[:, 7] = np.clip(dual_pose[:, 7], 0.0, 0.08)
            dual_pose[:, 15] = np.clip(dual_pose[:, 15], 0.0, 0.08)
            return {action_configs[0]['device_name']: dual_pose.astype(np.float64)}

    def predict_value(self, observation: Dict[str, np.ndarray], predicted_action: np.ndarray) -> Dict[str, np.ndarray]:
        obs_batch_robot, _ = self.sensor_data_to_umi_data(observation, target_frame="robot", for_action_policy=False) # obs_batch_robot is in the **umi** coordinate system
        obs_batch_for_policy = self._convert_obs_to_pi(obs_batch_robot, for_action_policy=False)
        processed_action = torch.from_numpy(predicted_action).unsqueeze(0)
        with torch.no_grad():
            outputs = self._policy.infer_value(obs_batch_for_policy, processed_action)
            value = outputs["value"]
        return {"value": value.astype(np.float64)}

    def sensor_data_to_umi_data(self, obs_dict: Dict[str, Any], target_frame: str = "umi", for_action_policy: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # Validate input format based on configured obs_horizon
        model_config = getattr(self.pi_config, "model", None)
        action_observation_horizon = int(getattr(model_config, "action_observation_horizon", 1))
        value_observation_horizon = int(getattr(model_config, "value_observation_horizon", 1))
        use_action_observation_history = for_action_policy
        use_value_observation_history = not for_action_policy
        use_history = self.obs_horizon > 1 and use_action_observation_history == 1 and use_value_observation_history == 1

        if use_history:
            # History mode: each value is a list of obs_horizon arrays
            for key, value in obs_dict.items():
                assert isinstance(value, list), f"Expected list for key '{key}', got {type(value)}"
                assert len(value) == self.obs_horizon, f"Expected {self.obs_horizon} items, got {len(value)}"
        elif use_action_observation_history:
            for key, value in obs_dict.items():
                assert isinstance(value, list), f"Expected list for key '{key}', got {type(value)}"
                assert len(value) >= action_observation_horizon, f"Expected {action_observation_horizon} items, got {len(value)}"
                obs_dict[key] = value[-action_observation_horizon:]
        elif use_value_observation_history:
            for key, value in obs_dict.items():
                assert isinstance(value, list), f"Expected list for key '{key}', got {type(value)}"
                assert len(value) >= value_observation_horizon, f"Expected {value_observation_horizon} items, got {len(value)}"
                obs_dict[key] = value[-value_observation_horizon:]
        else:
            raise ValueError(f"Invalid observation history mode: use_history={use_history}, use_action_observation_history={use_action_observation_history}, use_value_observation_history={use_value_observation_history}")
        batch_data: Dict[str, np.ndarray] = {}

        # Process camera observations
        cameras = [name for name in obs_dict.keys() if 'camera' in name.lower() and not name.endswith('_is_pad')]
        for idx, camera_name in enumerate(cameras):
            if use_history:
                # Stack all observations in history
                raw_images = obs_dict[camera_name]
                processed_images = [center_crop_square_resize(img) for img in raw_images]
                stacked_images = np.stack(processed_images)
                batch_data[f'camera{idx}_rgb'] = (stacked_images / 255.0).transpose(0, 3, 1, 2)  # (obs_horizon, C, H, W)
            elif use_action_observation_history or use_value_observation_history:
                # Stack all observations in history
                raw_images = obs_dict[camera_name]
                processed_images = [center_crop_square_resize(img) for img in raw_images]
                stacked_images = np.stack(processed_images)
                batch_data[f'camera{idx}_rgb'] = (stacked_images / 255.0).transpose(0, 3, 1, 2)  # (obs_horizon, C, H, W)
                pad_key = f"{camera_name}_is_pad"
                if pad_key in obs_dict:
                    batch_data[f'camera{idx}_rgb_is_pad'] = np.asarray(obs_dict[pad_key], dtype=bool)
                else:
                    logger.warning(f"camera{idx}_rgb_is_pad not found in obs_dict. ")
            else:
                raw_image = obs_dict[camera_name]
                processed_images = [center_crop_square_resize(raw_image)]
                stacked_images = np.stack(processed_images)
                batch_data[f'camera{idx}_rgb'] = (stacked_images / 255.0).transpose(0, 3, 1, 2)  # (1, C, H, W)

        # Process robot observation
        robot_device = next(iter([key for key in obs_dict.keys() if (key.lower().startswith('robot') or key.lower().startswith('rizon')) and not key.endswith('_is_pad')]), None)
        # logger.info(f"robot_device: {robot_device}")
        if robot_device is None:
            raise KeyError("Robot observation data not found in observation dictionary")

        if use_history:
            robot_obs_list = obs_dict[robot_device]
            robot_obs_array = np.stack(robot_obs_list)  # (obs_horizon, obs_dim)
        elif use_action_observation_history or use_value_observation_history:
            robot_obs_list = obs_dict[robot_device]
            robot_obs_array = np.stack(robot_obs_list)
        else:
            robot_obs = obs_dict[robot_device]
            assert isinstance(robot_obs, np.ndarray)
            robot_obs_array = robot_obs if robot_obs.ndim > 1 else robot_obs[None, :]

        # Validate observation dimension
        obs_dim = robot_obs_array.shape[-1]
        expected_obs_dim = 22 if self.num_robots == 1 else 44
        if obs_dim != expected_obs_dim:
            raise ValueError(f"Robot observation dimension {obs_dim} does not match expected "
                             f"{expected_obs_dim} for {self.num_robots} robot(s)")

        # Extract left arm (indices 14:22) for all timesteps
        left_tcp_pose_w_gripper = robot_obs_array[:, 14:22]  # (obs_horizon, 8)
        left_pos = left_tcp_pose_w_gripper[:, :3]  # (obs_horizon, 3)
        left_quat_xyzw = np.stack([process_quaternion(left_tcp_pose_w_gripper[i, 3:7], "f2l") for i in range(len(left_tcp_pose_w_gripper))])
        left_axis_angle = quaternion_to_axis_angle(left_quat_xyzw)  # (obs_horizon, 3)
        if target_frame == "umi" and self.tcp_umi_T is not None:
            left_pos, left_axis_angle = self._robot_pose_to_umi(left_pos, left_axis_angle)
        left_gripper = left_tcp_pose_w_gripper[:, 7:8]  # (obs_horizon, 1)
        batch_data['robot0_eef_pos'] = left_pos
        batch_data['robot0_eef_rot_axis_angle'] = left_axis_angle
        # TODO: remove this hack later
        batch_data['robot0_gripper_width'] = left_gripper

        if use_action_observation_history or use_value_observation_history:
            robot_pad_key = f"{robot_device}_is_pad"
            if robot_pad_key in obs_dict:
                batch_data['state_is_pad'] = np.asarray(obs_dict[robot_pad_key], dtype=bool)
            else:
                logger.warning(f"state_is_pad not found in obs_dict. ")

        robot_left_pose = np.concatenate([left_pos, left_axis_angle, left_gripper], axis=-1)  # (obs_horizon, 7)
        robot_pose_parts = [robot_left_pose]

        # Extract right arm if dual-arm configuration
        if self.num_robots == 2:
            right_tcp_pose_w_gripper = robot_obs_array[:, 36:44]  # (obs_horizon, 8)
            right_pos = right_tcp_pose_w_gripper[:, :3]  # (obs_horizon, 3)
            right_quat_xyzw = np.stack([process_quaternion(right_tcp_pose_w_gripper[i, 3:7], "f2l") for i in range(len(right_tcp_pose_w_gripper))])
            right_axis_angle = quaternion_to_axis_angle(right_quat_xyzw)  # (obs_horizon, 3)
            if target_frame == "umi" and self.tcp_umi_T is not None:
                right_pos, right_axis_angle = self._robot_pose_to_umi(right_pos, right_axis_angle)
            right_gripper = right_tcp_pose_w_gripper[:, 7:8]  # (obs_horizon, 1)

            batch_data['robot1_eef_pos'] = right_pos
            batch_data['robot1_eef_rot_axis_angle'] = right_axis_angle
            # TODO: remove this hack later
            batch_data['robot1_gripper_width'] = right_gripper

            if use_action_observation_history or use_value_observation_history:
                robot_pad_key = f"{robot_device}_is_pad"
                if robot_pad_key in obs_dict:
                    batch_data['state_is_pad'] = np.asarray(obs_dict[robot_pad_key], dtype=bool)
                else:
                    logger.warning(f"state_is_pad not found in obs_dict. ")

            robot_right_pose = np.concatenate([right_pos, right_axis_angle, right_gripper], axis=-1)  # (obs_horizon, 7)
            robot_pose_parts.append(robot_right_pose)

        # Process value prediction
        value_keys = next(iter([key for key in obs_dict.keys() if 'value_prediction' in key.lower() and not key.endswith('_is_pad')]), None)
        if value_keys is not None:
            value_obs_list = obs_dict[value_keys]
            value_obs_array = np.stack(value_obs_list)
            batch_data[value_keys] = value_obs_array
            value_pad_key = f"{value_keys}_is_pad"
            if value_pad_key in obs_dict:
                batch_data[value_pad_key] = np.asarray(obs_dict[value_pad_key], dtype=bool)
                batch_data[value_pad_key][:] = True
            else:
                logger.warning(f"{value_pad_key} not found in obs_dict. ")

        # Return the latest robot pose and batched data
        robot_pose = copy.deepcopy(np.concatenate(robot_pose_parts, axis=-1)[-1, :])  # (14,) or (7,)
        logger.info(f"robot_pose {'for action policy' if for_action_policy else 'for value policy'}: {np.concatenate(robot_pose_parts, axis=-1)}")
        logger.info(f"latest robot_pose {'for action policy' if for_action_policy else 'for value policy'}: {robot_pose}")
        return batch_data, robot_pose

    def _robot_pose_to_umi(self, pos: np.ndarray, axis_angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.tcp_umi_T is None:
            return pos, axis_angle
        pose_mat = pose_to_mat(np.concatenate([pos, axis_angle], axis=-1))
        transformed = pose_mat @ self.tcp_umi_T
        pose = mat_to_pose(transformed)
        return pose[:, :3], pose[:, 3:]

    def action_to_single_pose(self, umi_action):
        pos, axis_angle, gripper = umi_action[:, :3], umi_action[:, 3:6], umi_action[:, 6:6 + 1]
        quat = axis_angle_to_quaternion(axis_angle)
        single_pose = np.concatenate([pos, quat, gripper], axis=-1)
        return single_pose

    def action_to_dual_pose(self, umi_action):
        left_pos, left_axis_angle, left_gripper = umi_action[:, :3], umi_action[:, 3:6], umi_action[:, 6:6 + 1]
        right_pos, right_axis_angle, right_gripper = umi_action[:, 7:10], umi_action[:, 10:13], umi_action[:, 13:13 + 1]
        left_quat = axis_angle_to_quaternion(left_axis_angle)
        left_pose = np.concatenate([left_pos, left_quat, left_gripper], axis=-1)
        right_quat = axis_angle_to_quaternion(right_axis_angle)
        right_pose = np.concatenate([right_pos, right_quat, right_gripper], axis=-1)
        dual_pose = np.concatenate([left_pose, right_pose], axis=-1)
        return dual_pose

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
            # Create dummy observation (fixed format)
            # - dict_keys(['HikCameraDevice_1', 'HikCameraDevice_0', 'RizonRobot_1'])
            # - each value is a list with 2 elements
            #   - camera: (360, 270, 3) uint8
            #   - robot: (44,) float64
            observation = {
                "HikCameraDevice_1": [
                    np.random.randint(0, 255, (360, 270, 3), dtype=np.uint8)
                    for _ in range(self.obs_horizon)
                ],
                "HikCameraDevice_0": [
                    np.random.randint(0, 255, (360, 270, 3), dtype=np.uint8)
                    for _ in range(self.obs_horizon)
                ], 
                "RizonRobot_1": [
                    np.random.rand(44).astype(np.float64)
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
