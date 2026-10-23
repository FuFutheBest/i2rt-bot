#!/usr/bin/env python3
"""
NMX VLA policy adapter for SuperInference policy connector.

This module bridges SuperInference observations to local nmx_vla in-process
inference and converts returned action chunks to PolicyConnector output format.

Author: Junbo Wang, Cursor Agent
"""

import sys
import importlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from policy.base import BasePolicy
from utils.dsl_pose_utils import convert_pose_mat_rep, mat_to_pose, pose_to_mat
from utils.logger_config import logger
from utils.transform import process_quaternion

# import debugpy
# debugpy.listen(5678)
# print("🔍 Waiting for VSCode attach on 0.0.0.0:5678 ...")
# debugpy.wait_for_client()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NMX_VLA_ROOT = _REPO_ROOT / "third_party" / "nmx_vla"
if str(_NMX_VLA_ROOT) not in sys.path:
    sys.path.append(str(_NMX_VLA_ROOT))

class NMXVLAPolicy(BasePolicy):
    """Policy wrapper for local nmx_vla inference."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        task: str = "",
        unnorm_key: Optional[str] = None,
        unnormalize_method: Union[str, List[str]] = "quan",
        obs_horizon: int = 1,
        downsample: int = 1,
        obs_time_interval: float = 0.04,
        camera_devices: Optional[List[str]] = None,
        robot_device: Optional[str] = None,
        include_state: bool = False,
        rot_type: str = "XYZ",
    ) -> None:
        super().__init__()
        self.ckpt_path: Optional[str] = ckpt_path
        self.device: str = device
        self.task: str = task
        self.unnorm_key: Optional[str] = unnorm_key
        self.unnormalize_method: Union[str, List[str]] = unnormalize_method
        self.obs_horizon: int = int(obs_horizon)
        self.downsample: int = int(downsample)
        self.obs_time_interval: float = float(obs_time_interval)
        self.camera_devices: List[str] = list(camera_devices) if camera_devices else []
        self.robot_device: Optional[str] = robot_device
        self.include_state: bool = bool(include_state)
        self.rot_type: str = str(rot_type)
        self.min_infer_obs: int = 1
        self._local_model: Optional[Any] = None
        self._action_norm_stats: Optional[Dict[str, Any]] = None
        self._prediction_context_timestamp_ns: Optional[int] = None

    def load(self) -> None:
        """Load local nmx_vla model for in-process inference."""
        if not self.ckpt_path:
            raise ValueError("NMXVLAPolicy requires `ckpt_path` for local inference")
        self._local_model = self._load_local_model(
            ckpt_path=self.ckpt_path,
            device=self.device,
        )
        self._action_norm_stats = self._load_action_norm_stats()
        logger.info(
            f"NMXVLAPolicy loaded local model from {self.ckpt_path} on {self.device}"
        )
        self.initialized = True

    def reset(self) -> None:
        """Reset runtime states for a new episode."""
        self._prediction_context_timestamp_ns = None

    def _load_local_model(self, ckpt_path: str, device: str) -> Any:
        try:
            torch_module = importlib.import_module("torch")
            framework_module = importlib.import_module(
                "nmx_vla.model.framework.base_framework"
            )
            baseframework = getattr(framework_module, "baseframework")
        except Exception as exc:
            raise ImportError(
                "Failed to import nmx_vla local inference dependencies"
            ) from exc

        checkpoint_file = self._resolve_checkpoint_path(ckpt_path)
        self._validate_checkpoint_layout(checkpoint_file)
        model = baseframework.from_pretrained(str(checkpoint_file))
        model = model.to(device).eval()
        return model

    def _resolve_checkpoint_path(self, ckpt_path: str) -> Path:
        """
        Resolve checkpoint input to a concrete .pt file.

        Supported inputs:
        - /.../run_id/checkpoints/steps_xxx_pytorch_model.pt
        - /.../run_id/checkpoints/                (auto pick latest)
        - /.../run_id/                            (auto pick latest from checkpoints/)
        """
        path = Path(ckpt_path).expanduser().resolve()
        if path.is_file():
            if path.suffix != ".pt":
                raise ValueError(f"Checkpoint file must be .pt, got: {path}")
            return path

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

        if path.name == "checkpoints":
            checkpoints_dir = path
        else:
            checkpoints_dir = path / "checkpoints"

        if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find checkpoints directory at: {checkpoints_dir}"
            )

        candidate_files = list(checkpoints_dir.glob("*.pt"))
        if len(candidate_files) == 0:
            raise FileNotFoundError(f"No .pt checkpoint found in: {checkpoints_dir}")

        best_file, best_step = self._select_latest_checkpoint(candidate_files)
        logger.info(f"NMXVLAPolicy resolved checkpoint: {best_file} (step={best_step})")
        return best_file

    def _select_latest_checkpoint(self, candidate_files: List[Path]) -> Tuple[Path, int]:
        step_pattern = re.compile(r"steps_(\d+)_pytorch_model\.pt$")
        best_file: Optional[Path] = None
        best_step = -1
        for file_path in candidate_files:
            match = step_pattern.search(file_path.name)
            if match is None:
                continue
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_file = file_path

        if best_file is not None:
            return best_file, best_step

        # Fallback for non-standard names: choose newest by mtime
        newest = sorted(candidate_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        return newest, -1

    def _validate_checkpoint_layout(self, checkpoint_file: Path) -> None:
        run_dir = checkpoint_file.parents[1]
        config_yaml = run_dir / "config.yaml"
        dataset_statistics = run_dir / "dataset_statistics.json"
        if not config_yaml.exists():
            raise FileNotFoundError(f"Missing config.yaml for run dir: {run_dir}")
        if not dataset_statistics.exists():
            raise FileNotFoundError(
                f"Missing dataset_statistics.json for run dir: {run_dir}"
            )

    def _load_action_norm_stats(self) -> Dict[str, Any]:
        """Load action normalization statistics from local model checkpoint metadata."""
        if self._local_model is None:
            raise RuntimeError("Local model is not loaded yet")

        norm_stats = getattr(self._local_model, "norm_stats", None)
        if not isinstance(norm_stats, dict) or len(norm_stats) == 0:
            raise RuntimeError("Missing norm_stats in local nmx_vla model")

        resolved_key = self.unnorm_key
        if resolved_key is None:
            if len(norm_stats) == 1:
                resolved_key = next(iter(norm_stats.keys()))
            else:
                raise ValueError(
                    "Multiple norm_stats keys detected, please set `unnorm_key` in policy config: "
                    f"{list(norm_stats.keys())}"
                )
        if resolved_key not in norm_stats:
            raise KeyError(
                f"Invalid unnorm_key '{resolved_key}', available keys: {list(norm_stats.keys())}"
            )

        action_stats = norm_stats[resolved_key].get("action")
        if not isinstance(action_stats, dict):
            raise RuntimeError(f"Missing action stats for unnorm_key '{resolved_key}'")

        logger.info(
            "NMXVLAPolicy uses action norm stats key: "
            f"{resolved_key}, method: {self.unnormalize_method}"
        )
        return action_stats

    def _unnormalize_action_chunk(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Convert normalized actions to raw action range using dataset statistics."""
        if self._action_norm_stats is None:
            raise RuntimeError("Action norm stats are not initialized")

        stats = self._action_norm_stats

        action_q01 = np.asarray(stats["q01"], dtype=np.float64)
        action_q99 = np.asarray(stats["q99"], dtype=np.float64)
        action_min = np.asarray(stats["min"], dtype=np.float64)
        action_max = np.asarray(stats["max"], dtype=np.float64)
        if isinstance(self.unnormalize_method, str):
            if self.unnormalize_method == "quan":
                action_low = action_q01
                action_high = action_q99
            elif self.unnormalize_method == "minmax":
                action_low = action_min
                action_high = action_max
            else:
                raise ValueError(f"Invalid unnormalize_method: {self.unnormalize_method}")
        elif isinstance(self.unnormalize_method, list):
            assert len(self.unnormalize_method) == normalized_actions.shape[-1]
            action_low = np.zeros(normalized_actions.shape[-1], dtype=np.float64)
            action_high = np.zeros(normalized_actions.shape[-1], dtype=np.float64)
            for i, method in enumerate(self.unnormalize_method):
                if method == "quan":
                    action_low[i] = action_q01[i]
                    action_high[i] = action_q99[i]
                elif method == "minmax":
                    action_low[i] = action_min[i]
                    action_high[i] = action_max[i]
                else:
                    raise ValueError(f"Invalid unnormalize_method: {method}")
        else:
            raise ValueError(f"Invalid unnormalize_method: {self.unnormalize_method}")

        normalized_actions = np.asarray(normalized_actions, dtype=np.float64)
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)

        return 0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low

    def close(self) -> None:
        """Release local model reference."""
        self._local_model = None
        self._action_norm_stats = None

    def __del__(self) -> None:
        self.close()

    def set_prediction_context(self, timestamp_ns: int) -> None:
        """Receive timestamp from PolicyConnector for optional tracing."""
        self._prediction_context_timestamp_ns = int(timestamp_ns)

    def _extract_latest_frame(self, data: Any) -> np.ndarray:
        """Extract latest observation frame from single frame or history list."""
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Observation history list is empty")
            return np.asarray(data[-1])
        array = np.asarray(data)
        if array.ndim >= 4:
            return np.asarray(array[-1])
        return array

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Center-crop image on the longer spatial dimension, then resize to 256x256.
        NOTE: !!!!!!!! hardcode here
        """
        image_array = np.asarray(image, dtype=np.uint8)
        if image_array.ndim < 2:
            raise ValueError(f"Invalid image shape: {image_array.shape}")

        height, width = image_array.shape[:2]
        crop_size = min(height, width)
        top = max((height - crop_size) // 2, 0)
        left = max((width - crop_size) // 2, 0)
        cropped = image_array[top : top + crop_size, left : left + crop_size]
        target_size = 256
        row_indices = np.linspace(0, crop_size - 1, target_size).astype(np.int64)
        col_indices = np.linspace(0, crop_size - 1, target_size).astype(np.int64)
        resized = cropped[row_indices[:, None], col_indices[None, :]]
        return np.asarray(resized, dtype=np.uint8)

    def _find_robot_device(self, observation: Dict[str, Any]) -> Optional[str]:
        if self.robot_device is not None and self.robot_device in observation:
            return self.robot_device
        for key in observation.keys():
            lowered = key.lower()
            if lowered.startswith("robot") or lowered.startswith("rizon"):
                return key
        return None

    def _collect_images(self, observation: Dict[str, Any]) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        if self.camera_devices:
            for camera_name in self.camera_devices:
                if camera_name not in observation:
                    logger.warning(f"NMXVLAPolicy missing camera observation: {camera_name}")
                    continue
                image = self._extract_latest_frame(observation[camera_name])
                images.append(self._preprocess_image(image))
            return images

        for key, value in observation.items():
            lowered = key.lower()
            if "camera" not in lowered:
                continue
            image = self._extract_latest_frame(value)
            images.append(self._preprocess_image(image))
        return images

    def _collect_state(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        if not self.include_state:
            return None
        robot_key = self._find_robot_device(observation)
        if robot_key is None:
            return None
        state = self._extract_latest_frame(observation[robot_key])
        state = np.asarray(state)
        if state.ndim == 1:
            state = state[None, :]
        return state.astype(np.float32)

    def _build_vla_input(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        images = self._collect_images(observation)
        if len(images) == 0:
            raise ValueError("No camera observations found for nmx_vla inference")

        example: Dict[str, Any] = {
            "image": images,
            "lang": self.task,
        }
        state = self._collect_state(observation)
        if state is not None:
            # example["state"] = state
            # TODO: !!!!!!!!!!!!!! hack state as zeros
            example["state"] = np.zeros((1, 8))

        payload: Dict[str, Any] = {
            "examples": [example],
        }
        if self._prediction_context_timestamp_ns is not None:
            payload["timestamp_ns"] = self._prediction_context_timestamp_ns
        return payload

    def _extract_action_chunk(self, payload_or_response: Dict[str, Any]) -> np.ndarray:
        if "normalized_actions" in payload_or_response:
            data = payload_or_response
        else:
            data = payload_or_response.get("data", payload_or_response)
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected nmx_vla response data type: {type(data)}")
        if "normalized_actions" not in data:
            raise KeyError(
                f"'normalized_actions' missing in response keys: {list(data.keys())}"
            )

        actions = np.asarray(data["normalized_actions"])
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk shape [T, D], got {actions.shape}")
        return actions.astype(np.float64)

    def _fit_action_dim(self, action_chunk: np.ndarray, target_dim: int) -> np.ndarray:
        if target_dim <= 0:
            return action_chunk
        current_dim = int(action_chunk.shape[1])
        if current_dim == target_dim:
            return action_chunk
        if current_dim > target_dim:
            logger.warning(
                f"NMXVLAPolicy truncates action dim from {current_dim} to {target_dim}"
            )
            return action_chunk[:, :target_dim]

        logger.warning(
            f"NMXVLAPolicy pads action dim from {current_dim} to {target_dim} with zeros"
        )
        padding = np.zeros((action_chunk.shape[0], target_dim - current_dim), dtype=action_chunk.dtype)
        return np.concatenate([action_chunk, padding], axis=1)

    def _fit_chunk_length(self, action_chunk: np.ndarray, chunk_length: int) -> np.ndarray:
        if chunk_length <= 0:
            return action_chunk
        if action_chunk.shape[0] >= chunk_length:
            return action_chunk[:chunk_length]
        missing = chunk_length - action_chunk.shape[0]
        tail = np.repeat(action_chunk[-1:, :], missing, axis=0)
        return np.concatenate([action_chunk, tail], axis=0)

    def _extract_robot_state(self, observation: Dict[str, Any]) -> np.ndarray:
        robot_key = self._find_robot_device(observation)
        if robot_key is None:
            raise KeyError("Robot observation data not found in observation dictionary")
        robot_state = self._extract_latest_frame(observation[robot_key])
        robot_state = np.asarray(robot_state, dtype=np.float64).reshape(-1)
        return robot_state

    def _extract_arm_current_pose(self, robot_state: np.ndarray, arm_index: int) -> np.ndarray:
        if arm_index == 0:
            if robot_state.shape[0] < 22:
                raise ValueError(
                    "Unsupported robot observation dimension for left arm: "
                    f"expected at least 22, got {robot_state.shape[0]}"
                )
            tcp_pose_w_gripper = robot_state[14:22]
        else:
            if robot_state.shape[0] < 44:
                raise ValueError(
                    "Unsupported robot observation dimension for right arm: "
                    f"expected at least 44, got {robot_state.shape[0]}"
                )
            tcp_pose_w_gripper = robot_state[36:44]

        pos = tcp_pose_w_gripper[:3]
        quat_xyzw = np.asarray(process_quaternion(tcp_pose_w_gripper[3:7], "f2l"), dtype=np.float64)
        axis_angle = Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, axis_angle], axis=-1)

    def _convert_relative_action_to_absolute(
        self,
        action_chunk: np.ndarray,
        observation: Dict[str, Any],
        target_action_dim: int,
    ) -> np.ndarray:
        robot_state = self._extract_robot_state(observation)
        left_current_pose = self._extract_arm_current_pose(robot_state, arm_index=0)
        left_current_mat = pose_to_mat(left_current_pose)

        action_dim = int(action_chunk.shape[-1])
        left_rel_pos = action_chunk[:, :3]
        if self.rot_type == 'wxyz':
            left_rel_quat_wxyz = action_chunk[:, 3:7]
            left_rel_quat_xyzw = np.asarray(
                [process_quaternion(quat, "f2l") for quat in left_rel_quat_wxyz],
                dtype=np.float64,
            )
        elif self.rot_type == 'xyzw':
            left_rel_quat_xyzw = action_chunk[:, 3:7]
        elif self.rot_type in ["xyz", "zyx", "XYZ", "ZYX"]:
            left_rel_rpy = action_chunk[:, 3:6]
            left_rel_quat_xyzw = Rotation.from_euler(self.rot_type, left_rel_rpy).as_quat()
        else:
            raise ValueError(f"Invalid rotation type: {self.rot_type}")
        left_rel_gripper = action_chunk[:, 7:8]
        left_rel_axis_angle = Rotation.from_quat(left_rel_quat_xyzw).as_rotvec()
        left_rel_pose = np.concatenate([left_rel_pos, left_rel_axis_angle], axis=-1)
        left_rel_mat = pose_to_mat(left_rel_pose)
        left_abs_mat = convert_pose_mat_rep(
            left_rel_mat,
            left_current_mat,
            pose_rep="relative",
            backward=True,
        )
        left_abs_pose = mat_to_pose(left_abs_mat)
        left_abs_quat_xyzw = Rotation.from_rotvec(left_abs_pose[:, 3:6]).as_quat()
        left_abs_action = np.concatenate([left_abs_pose[:, :3], left_abs_quat_xyzw, left_rel_gripper], axis=-1)

        if target_action_dim <= 8:
            left_abs_action[:, 7] = np.clip(left_abs_action[:, 7], 0.0, 0.08)
            return left_abs_action.astype(np.float64)

        right_abs_action: Optional[np.ndarray] = None
        has_right_action = action_dim >= 16 and robot_state.shape[0] >= 44
        if has_right_action:
            right_current_pose = self._extract_arm_current_pose(robot_state, arm_index=1)
            right_current_mat = pose_to_mat(right_current_pose)
            right_rel_pos = action_chunk[:, 8:11]
            right_rel_quat_wxyz = action_chunk[:, 11:15]
            right_rel_gripper = action_chunk[:, 15:16]
            right_rel_quat_xyzw = np.asarray(
                [process_quaternion(quat, "f2l") for quat in right_rel_quat_wxyz],
                dtype=np.float64,
            )
            right_rel_axis_angle = Rotation.from_quat(right_rel_quat_xyzw).as_rotvec()
            right_rel_pose = np.concatenate([right_rel_pos, right_rel_axis_angle], axis=-1)
            right_rel_mat = pose_to_mat(right_rel_pose)
            right_abs_mat = convert_pose_mat_rep(
                right_rel_mat,
                right_current_mat,
                pose_rep="relative",
                backward=True,
            )
            right_abs_pose = mat_to_pose(right_abs_mat)
            right_abs_quat_xyzw = Rotation.from_rotvec(right_abs_pose[:, 3:6]).as_quat()
            right_abs_action = np.concatenate([right_abs_pose[:, :3], right_abs_quat_xyzw, right_rel_gripper], axis=-1)
        else:
            logger.warning(
                "Dual-arm action requested but right arm action/state is missing; padding zeros."
            )
            right_abs_action = np.zeros_like(left_abs_action)

        dual_action = np.concatenate([left_abs_action, right_abs_action], axis=-1)
        dual_action[:, 7] = np.clip(dual_action[:, 7], 0.0, 0.08)
        dual_action[:, 15] = np.clip(dual_action[:, 15], 0.0, 0.08)
        return dual_action.astype(np.float64)

    def predict(
        self,
        observation: Dict[str, Any],
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        action_index: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Run one inference request and return action chunk for the first control target."""
        _ = action_index  # Keep interface compatibility with PolicyConnector.
        if len(action_configs) == 0:
            raise ValueError("action_configs is empty")

        payload = self._build_vla_input(observation)
        if self._local_model is None:
            raise RuntimeError("Local backend is not loaded, call load() first")
        output = self._local_model.predict_action(**payload)
        action_chunk = self._extract_action_chunk(output)

        # Unnormalize
        action_chunk = self._unnormalize_action_chunk(action_chunk)

        # Convert to absolute action
        target_dim = int(action_configs[0].get("action_dim", action_chunk.shape[1]))
        action_chunk = self._convert_relative_action_to_absolute(
            action_chunk=action_chunk,
            observation=observation,
            target_action_dim=target_dim,
        )

        target_device = str(action_configs[0]["device_name"])
        action_chunk = self._fit_action_dim(action_chunk, target_dim=target_dim)
        action_chunk = self._fit_chunk_length(action_chunk, chunk_length=chunk_length)
        return {target_device: action_chunk}
