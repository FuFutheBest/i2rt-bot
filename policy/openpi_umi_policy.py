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
from typing import Dict, List, Any, Optional, Tuple
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
from utils.openpi_checkpoint import validate_openpi_checkpoint_contract
from skimage.transform import resize
import time
from utils.calibration_utils import (
    load_tcp_umi_transforms,
    resolve_repository_calibration_path,
)
from utils.latency_tracker import LatencyTracker
from scipy.spatial.transform import Rotation as _R

# Dual-arm layout of the robot obs SHM: each arm is 8-dim [pos3, quat_wxyz4, grip1]
_RAW_LEFT = slice(14, 22)
_RAW_RIGHT = slice(36, 44)

# Dual-arm pi0: UmiDualArmInputs/Outputs are hard-wired to two arms, so the model
# always consumes a 16-dim two-arm state and emits a 16-dim action.
ARM_DIM = 8
N_ARMS = 2
STATE_DIM = ARM_DIM * N_ARMS  # 16

# Defaults preserve direct construction compatibility.  The active OpenPi config
# supplies these values explicitly, and SuperOffline binds them into its action
# calibration digest because they change the emitted control action.
GRIPPER_OPEN_MAX = 0.11

# Empirical correction: the left gripper's predicted opening is offset high, so
# subtract this constant (meters) from the left-arm gripper action at inference time.
LEFT_GRIPPER_OFFSET = 0.0

# Empirical correction: the right gripper's predicted opening is offset high, so
# subtract this constant (meters) from the right-arm gripper action at inference time.
RIGHT_GRIPPER_OFFSET = 0.0

# Wrist-image preprocessing to match the training framing.
# Device frame is 640(W) x 360(H) (data_shape=[360, 640, 3], set by the camera, 16:9).
# Crop 185px from the left and right of the 640-wide frame -> 270(W) x 360(H), an exact
# 4:3 portrait (270*4 == 360*3), then resize to 240(W) x 320(H) -- also 4:3, a uniform
# 0.889x scale so there is no aspect distortion. openpi's model transform then
# resize-with-pads this to 224. Camera applies NO rotation (rotation_degrees=0).
_WRIST_CROP_LRTB = (185, 185, 0, 0)   # pixels cropped from (left, right, top, bottom): 185px off L/R of 640 -> 270 wide
_WRIST_RESIZE_HW = (320, 240)         # (H, W) -> 320H x 240W portrait 4:3, matches training wrist_image framing

def _latest_frame(v):
    """obs value may be list(obs_horizon) or ndarray with a horizon axis; take the latest frame."""
    if isinstance(v, list):
        v = v[-1]
    return np.asarray(v)

def _quat_wxyz_to_mat(qwxyz):
    q_xyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=np.float64)
    q_xyzw = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-12)
    return _R.from_quat(q_xyzw).as_matrix()

def _mat_to_quat_wxyz(m):
    x, y, z, w = _R.from_matrix(m).as_quat()
    wxyz = np.array([w, x, y, z], dtype=np.float64)
    return wxyz if wxyz[0] >= 0 else -wxyz   # canonicalize w>=0 to match training

def _mat_to_quat_xyzw(m):
    return _R.from_matrix(m).as_quat()       # [x,y,z,w] for the control SHM

def _pose_mat(pos, quat_wxyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_mat(quat_wxyz)
    T[:3, 3] = pos
    return T

def _as_uint8_hwc(img):
    img = np.asarray(img)
    if img.ndim == 4:                       # (horizon,H,W,C) / (horizon,C,H,W)
        img = img[-1]
    if img.ndim == 3 and img.shape[0] == 3: # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img) if img.max() <= 1.0 + 1e-6 else img
        img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def _crop_resize_wrist(img):
    """Crop then resize a wrist image to match the training framing.
    img: HWC uint8 (already rotated by the camera device). Returns HWC uint8 at _WRIST_RESIZE_HW.
    openpi's model transforms will further resize this to 224 internally."""
    h, w = img.shape[:2]
    left, right, top, bottom = _WRIST_CROP_LRTB
    img = img[top:h - bottom, left:w - right]
    out_h, out_w = _WRIST_RESIZE_HW
    img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)
    return img.astype(np.uint8)


def _finite_gripper_value(value: float, *, name: str, positive: bool = False) -> float:
    """Normalize a configured gripper conversion scalar before model loading."""
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "a positive finite number" if positive else "a finite number"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


# Debug dump of the wrist images fed to the policy. Enable by setting the env var
# OPENPI_DUMP_OBS to a directory (or "1" to use ./logs/openpi_obs_dump). Only the
# first OPENPI_DUMP_OBS_N frames (default 1) per process are written.
_DUMP_STATE = {"count": 0}

def _maybe_dump_obs(images: Dict[str, np.ndarray]) -> None:
    flag = os.environ.get("OPENPI_DUMP_OBS")
    if not flag:
        return
    limit = int(os.environ.get("OPENPI_DUMP_OBS_N", "1"))
    if _DUMP_STATE["count"] >= limit:
        return
    out_dir = "logs/openpi_obs_dump" if flag in ("1", "true", "True") else flag
    os.makedirs(out_dir, exist_ok=True)
    import cv2  # local import; cv2 is always available where cameras run
    idx = _DUMP_STATE["count"]
    for name, img in images.items():
        rgb = _as_uint8_hwc(img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        path = os.path.join(out_dir, f"{name}_{idx:03d}.png")
        cv2.imwrite(path, bgr)
        logger.info(f"[OPENPI_DUMP_OBS] wrote {path} shape={rgb.shape}")
    _DUMP_STATE["count"] += 1


class OpenPiUmiPolicy(BasePolicy):
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
        mask_state: bool = False,
        gripper_open_max: float = GRIPPER_OPEN_MAX,
        left_gripper_offset: float = LEFT_GRIPPER_OFFSET,
        right_gripper_offset: float = RIGHT_GRIPPER_OFFSET,
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
        # When True, the state fed to the model is fully zeroed. This matches
        # checkpoints trained with the proprioceptive state masked out (all zeros),
        # so inference must present a zero state to avoid a train/inference mismatch.
        self.mask_state = mask_state
        self.gripper_open_max = _finite_gripper_value(
            gripper_open_max,
            name="gripper_open_max",
            positive=True,
        )
        self.left_gripper_offset = _finite_gripper_value(
            left_gripper_offset,
            name="left_gripper_offset",
        )
        self.right_gripper_offset = _finite_gripper_value(
            right_gripper_offset,
            name="right_gripper_offset",
        )
        # This policy targets the dual-arm pi0 checkpoint: UmiDualArmInputs/Outputs
        # are hard-wired to two arms and always emit a 16-dim action. Single-arm
        # deployment is not supported -- a dummy second arm would pollute the
        # cross-arm attention and the 8-vs-16 action dims would mismatch.
        if num_robots != N_ARMS:
            raise ValueError(f"OpenPiUmiPolicy only supports dual-arm (num_robots={N_ARMS})")
        self.num_robots = num_robots
        self.chunk_length = chunk_length
        self._latency_tracker: LatencyTracker = LatencyTracker(maxlen=self.chunk_length)
        self._policy = self.load_openpi_policy(self.pi_config_name, self.ckpt_dir, self.device)

        if tcp_umi_json_path is not None:
            self.tcp_umi_T, self.umi_tcp_T = self._load_calibration(tcp_umi_json_path)
        else:
            self.tcp_umi_T, self.umi_tcp_T = None, None

    def _load_calibration(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        calibration_path = resolve_repository_calibration_path(json_path)
        tcp_umi, umi_tcp = load_tcp_umi_transforms(str(calibration_path))
        logger.info(f"Loaded TCP↔UMI calibration from {calibration_path}")
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
        contract = validate_openpi_checkpoint_contract(
            pi_config_name,
            ckpt_dir,
            expected_action_dim=STATE_DIM,
            expected_state_dim=STATE_DIM,
        )
        logger.info(
            "Validated OpenPI checkpoint contract: "
            f"config={contract['config_name']} asset_id={contract['asset_id']} "
            f"state_dim={contract['state_dim']} action_dim={contract['action_dim']}"
        )
        self.pi_config = _config.get_config(pi_config_name)
        print(f"Loading policy from {ckpt_dir}")
        policy = _policy_config.create_trained_policy(self.pi_config, ckpt_dir)
        return policy

    def _arm_to_umi(self, arm8):
        """robot SHM one arm [pos3, quat_wxyz4, grip1] -> (pos3, quat_wxyz4, grip1),
        applying TCP->UMI if calibration is available."""
        pos = arm8[0:3].astype(np.float64)
        quat_wxyz = arm8[3:7].astype(np.float64)   # SHM already stores wxyz
        grip = arm8[7:8].astype(np.float64)
        if getattr(self, "tcp_umi_T", None) is not None:
            T_umi = _pose_mat(pos, quat_wxyz) @ self.tcp_umi_T
            pos = T_umi[:3, 3]
            quat_wxyz = _mat_to_quat_wxyz(T_umi[:3, :3])
        return pos, quat_wxyz, grip

    def _convert_obs_to_pi(self, observation):
        # --- cameras: follow config obs.devices order, [0]=left wrist->wrist_image_1, [1]=right wrist->wrist_image_2
        cam_keys = [k for k in observation.keys() if 'camera' in k.lower()]
        assert len(cam_keys) >= 2, f"need 2 wrist cameras, got: {cam_keys}"
        left_dev = _as_uint8_hwc(_latest_frame(observation[cam_keys[0]]))
        right_dev = _as_uint8_hwc(_latest_frame(observation[cam_keys[1]]))
        left_img = _crop_resize_wrist(left_dev)
        right_img = _crop_resize_wrist(right_dev)
        _maybe_dump_obs({
            "left_device": left_dev,    # device SHM output (rotated + resized to data_shape)
            "right_device": right_dev,
            "left_img": left_img,       # final policy input (after _crop_resize_wrist)
            "right_img": right_img,
        })

        # --- robot obs: 44-dim dual-arm vector
        robot_key = next(k for k in observation.keys()
                         if k.lower().startswith(('rizon', 'robot')))
        robot_obs = _latest_frame(observation[robot_key])
        if robot_obs.ndim > 1:
            robot_obs = robot_obs[-1]

        lpos, lquat, lgrip = self._arm_to_umi(robot_obs[_RAW_LEFT])
        rpos, rquat, rgrip = self._arm_to_umi(robot_obs[_RAW_RIGHT])

        ego = np.zeros(7, dtype=np.float64)   # ego is dropped by UmiDualArmInputs, zeros are fine
        state23 = np.concatenate(
            [lpos, lquat, lgrip, rpos, rquat, rgrip, ego]
        ).astype(np.float32)

        # Checkpoints trained with the state masked out expect a zero state at
        # inference; otherwise the live proprioception would be out-of-distribution.
        if self.mask_state:
            state23 = np.zeros_like(state23)

        # NOTE: create_trained_policy applies only its (empty) repack_transforms
        # argument, NOT UmiDualArmDataConfig.repack_transforms. So we must emit the
        # post-repack keys that UmiDualArmInputs expects directly (state /
        # left_wrist_image / right_wrist_image), not the dataset's observation.* keys.
        return {
            "state": state23,
            "left_wrist_image": left_img,
            "right_wrist_image": right_img,
            "prompt": self.task,
        }

    def predict(self, observation, action_configs, chunk_length=50, action_index=None):
        # 1) Build openpi input (23-dim quat-wxyz state, 2 wrist images, prompt)
        pi_input = self._convert_obs_to_pi(observation)

        # 2) Inference (optionally RTC). RTC guides the new chunk's prefix with the
        #    tail of the currently executing chunk; it engages only when RTC is on,
        #    an action_index is known, and inference latency still leaves room in the
        #    chunk (action_index + inference_delay < chunk_length). Hyper-parameters
        #    follow the OpenPiPolicy convention. Output is already "absolute" (T,16)
        #    [pos3, quat_wxyz4, grip1]x2 (UMI frame).
        rtc_config: Optional[Dict[str, float | int]] = None
        if self.enable_rtc and action_index is not None:
            default_delay_steps = 4
            if self.execution_fps is not None and self.execution_fps > 0:
                inference_delay = self._latency_tracker.compute_inference_delay_steps(
                    execution_fps=self.execution_fps,
                    use_percentile=0.95,
                    default_delay_steps=default_delay_steps,
                )
            else:
                inference_delay = default_delay_steps
            prefix_attention_horizon = chunk_length - inference_delay - action_index
            rtc_config = {
                "inference_delay": inference_delay,
                "prefix_attention_horizon": prefix_attention_horizon,
                "max_guidance_weight": 10.0,
            }

        with torch.no_grad():
            inference_delay = int(rtc_config.get("inference_delay", 0)) if rtc_config is not None else 0
            if (
                self.enable_rtc
                and action_index is not None
                and rtc_config is not None
                and (action_index + inference_delay < chunk_length)
            ):
                out = self._policy.infer(pi_input, time_base=action_index, rtc_config=rtc_config)
            else:
                if (
                    self.enable_rtc
                    and action_index is not None
                    and rtc_config is not None
                    and (action_index + inference_delay >= chunk_length)
                ):
                    logger.warning(
                        "FPS too low for RTC, falling back to normal infer(): "
                        f"action_index={action_index}, inference_delay={inference_delay}, "
                        f"action_index+inference_delay={action_index + inference_delay} "
                        f">= chunk_length={chunk_length}."
                    )
                out = self._policy.infer(pi_input)
        actions = np.asarray(out["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]

        T = actions.shape[0]
        if actions.shape[-1] != STATE_DIM:
            raise ValueError(f"openpi action dim {actions.shape[-1]}, expected {STATE_DIM} (dual-arm)")

        # 3) Per arm: UMI->TCP (tool offset) + wxyz->xyzw + gripper clip
        ctrl = np.zeros((T, STATE_DIM), dtype=np.float64)
        for a in range(N_ARMS):
            blk = actions[:, a*ARM_DIM:(a+1)*ARM_DIM]
            pos, quat_wxyz, grip = blk[:, 0:3], blk[:, 3:7], blk[:, 7:8]
            # Both grippers are biased high; subtract a per-arm offset (only the
            # gripper dim is touched, all pose dims are left unchanged).
            grip_offset = (
                self.left_gripper_offset if a == 0 else self.right_gripper_offset
            )
            for t in range(T):
                T_umi = _pose_mat(pos[t], quat_wxyz[t])
                T_tcp = T_umi @ self.umi_tcp_T if getattr(self, "umi_tcp_T", None) is not None else T_umi
                ctrl[t, a*ARM_DIM:a*ARM_DIM+3] = T_tcp[:3, 3]
                ctrl[t, a*ARM_DIM+3:a*ARM_DIM+7] = _mat_to_quat_xyzw(T_tcp[:3, :3])   # xyzw for the control SHM
                ctrl[t, a*ARM_DIM+7] = np.clip(
                    grip[t, 0] - grip_offset,
                    0.0,
                    self.gripper_open_max,
                )

        return {action_configs[0]['device_name']: ctrl[:chunk_length].astype(np.float64)}

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
