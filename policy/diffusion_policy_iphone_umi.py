"""
Diffusion Policy - runs a diffusion policy.

Author: Han Xue, Wang Zheng
"""

import sys
import os
import json
import time
from pathlib import Path
from policy.base import BasePolicy
import dill
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import torch
import hydra
import transforms3d as t3d
from copy import deepcopy
from utils.logger_config import logger
from skimage.transform import resize
import scipy.spatial.transform as st
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
UMI_BASE_DIR = ROOT_DIR / "third_party" / "umi_base"
if str(UMI_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(UMI_BASE_DIR))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "third_party/data-scaling-laws"
    )
)
############### Dependency on data-scaling-laws ###############
from umi.common.pose_util import (
    pose_to_mat,
    mat_to_pose10d,
    mat_to_pose,
    pose10d_to_mat,
)
from diffusion_policy.common.space_utils import (
    pose_6d_to_pose_9d,
    ortho6d_to_rotation_matrix,
    pose_3d_9d_to_homo_matrix_batch,
    homo_matrix_to_pose_9d_batch
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.image_utils import center_crop_and_resize_image
from umi.real_world.real_inference_util import (
    get_real_obs_dict,
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
)
from diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.cv2_util import get_image_transform

#################################################
from utils.calibration_utils import (
    load_tcp_umi_transforms,
    load_transform_from_json,
    resolve_repository_calibration_path,
)
from utils.transform import process_quaternion
from utils.rerun_visualization import visualize_action_time_series


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
    img = img[h_off : h_off + shorter_edge, w_off : w_off + shorter_edge]

    # Dynamic crop of the square dimension (e.g., [15:285] when size==300 and ratios are 0.05/0.95)
    margin = int(shorter_edge * crop_min_ratio)
    end = int(shorter_edge * crop_max_ratio)
    if end > margin:
        img = img[margin:end, margin:end]

    img = resize(img, (tgt, tgt), preserve_range=True, anti_aliasing=True).astype(
        np.uint8
    )
    return img


def center_crop_and_resize_image(image, target_size=(224, 224)):
    """
    Center crop the image to square using min(h, w), then resize to target size

    Args:
        image: Input image
        target_size: Target size (height, width)

    Returns:
        Processed image
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)

    # Calculate center crop coordinates
    y_start = (h - min_dim) // 2
    x_start = (w - min_dim) // 2

    # Center crop to square
    cropped_img = image[y_start : y_start + min_dim, x_start : x_start + min_dim]

    # Resize to target size
    resized_img = cv2.resize(cropped_img, target_size)

    return resized_img


def quaternion_to_pose9d(pose: np.ndarray) -> np.ndarray:
    """
    pose: (posi,quat)
    quat: (..., 4) [x,y,z,w]
    return (..., 6) (r11,r21,r31,r12,r22,r32)
    """
    pose = np.asarray(pose)
    assert pose.shape[-1] == 7  # Ensure input_dim must is 7

    pos = pose[..., :3]
    quat = pose[..., 3:]

    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if not np.all(quat_norm):
        raise ValueError("Zero norm encountered in input")
    quat_xyzw = quat / quat_norm
    rot_mats = st.Rotation.from_quat(quat_xyzw).as_matrix()
    rot6 = rot_mats[..., :, :2].reshape(pose.shape[:-1] + (6,), order="F")
    return np.concatenate([pos, rot6], axis=-1)


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


class DiffusionPolicyIPhoneUMI(BasePolicy):
    def __init__(
        self,
        ckpt_dir: str,
        num_inference_steps: int = 16,
        device: str = "cuda",
        obs_horizon: int = 2,
        downsample: int = 3,
        obs_time_interval: float = 0.04,
        left_robot_base_to_world_transform: Optional[str] = None,
        right_robot_base_to_world_transform: Optional[str] = None,
        tcp_umi_json_path: Optional[str] = None,
        enable_visualization: bool = True,
        camera_devices: Optional[List[str]] = None,
        robot_device: Optional[str] = None,
        preprocess_resize_height: int = 224,
        preprocess_resize_width: int = 224,
        use_relative_action: bool = False,
        use_inter_gripper: bool = False,
        center_crop_min_ratio: float = 0.05,
        center_crop_max_ratio: float = 0.95,
    ) -> None:
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.obs_horizon = obs_horizon
        self.obs_time_interval = obs_time_interval
        self.downsample = downsample
        self.enable_visualization = enable_visualization
        self.use_inter_gripper = use_inter_gripper
        self.use_relative_action = use_relative_action
        self.lowdim_keys = list(
            [
                "left_robot_tcp_pose",
                "left_robot_gripper_width",
                "right_robot_tcp_pose",
                "right_robot_gripper_width",
            ]
        )
        assert self.obs_horizon > 0, "obs_horizon must be greater than 0"
        assert self.downsample > 0, "downsample must be greater than 0"
        self.min_infer_obs = (self.obs_horizon - 1) * self.downsample + 1
        self._policy, self.action_representation = (
            self.load_diffusion_policy(
                self.ckpt_dir, self.num_inference_steps, self.device
            )
        )
        if left_robot_base_to_world_transform is not None:
            self.left_robot_base_to_world_transform = self._load_rigid_transform(
                left_robot_base_to_world_transform,
                "left_robot_base_to_world_transform",
            )
        if right_robot_base_to_world_transform is not None:
            self.right_robot_base_to_world_transform = self._load_rigid_transform(
                right_robot_base_to_world_transform,
                "right_robot_base_to_world_transform",
            )
        if tcp_umi_json_path is not None:
            self.tcp_umi_T, self.umi_tcp_T = self._load_calibration(tcp_umi_json_path)
        else:
            self.tcp_umi_T, self.umi_tcp_T = None, None
        self.camera_devices: List[str] = list(camera_devices) if camera_devices else []
        self.robot_device: Optional[str] = robot_device
        self._offset_visualization_failed: bool = False
        self._last_num_robots: int = (
            2  # cache latest robot count for single/dual compatibility
        )
        self._missing_camera_warning_issued: bool = False
        self._camera_fallback_warning_issued: bool = False
        # Preprocessing configuration
        self.preprocess_resize_height: int = preprocess_resize_height
        self.preprocess_resize_width: int = preprocess_resize_width
        self.center_crop_min_ratio: float = center_crop_min_ratio
        self.center_crop_max_ratio: float = center_crop_max_ratio
        self.action_step_interval_s: float = 0.05
        self.action_step_interval_ns: int = int(self.action_step_interval_s * 1e9)

    def _load_calibration(self, json_path: str) -> Tuple[np.ndarray, np.ndarray]:
        calibration_path = resolve_repository_calibration_path(json_path)
        tcp_umi, umi_tcp = load_tcp_umi_transforms(str(calibration_path))
        logger.info(f"Loaded TCP↔UMI calibration from {calibration_path}")
        return tcp_umi, umi_tcp

    @staticmethod
    def _load_rigid_transform(json_path: str, name: str) -> np.ndarray:
        calibration_path = resolve_repository_calibration_path(json_path)
        transform = np.asarray(
            load_transform_from_json(str(calibration_path)), dtype=np.float64
        )
        if not np.isfinite(transform).all():
            raise ValueError(f"{name} must contain finite values")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"{name} must be a homogeneous 4x4 transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError(f"{name} rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError(f"{name} rotation must have determinant +1")
        return transform

    def load_diffusion_policy(self, ckpt_dir, num_inference_steps=16, device="cuda"):
        ckpt_path = os.path.join(ckpt_dir, "checkpoints", "latest.ckpt")
        payload = torch.load(
            open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill
        )
        cfg = payload["cfg"]
        # print(payload["cfg"].keys())
        logger.info(f"model_name: {cfg.policy.obs_encoder.model_name}")
        if "dataset_path" in cfg.task.dataset:
            logger.info("dataset_path:", cfg.task.dataset.dataset_path)
        obs_res = get_real_obs_resolution(cfg.task.shape_meta)
        logger.info(f"obs_res: {obs_res}")
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.num_inference_steps = num_inference_steps
        policy.eval()
        policy.to(device)

        action_representation = (
            cfg.task.dataset.action_representation
            if "action_representation" in cfg.task.dataset
            else "relative"
        )
        action_representation = (
            "only-y-inference"
            if action_representation == "only-y-train"
            else action_representation
        )
        logger.info(f"action_representation: {action_representation}")
        return policy, action_representation


    def _create_colored_overlay(
        self, original: np.ndarray, shifted: np.ndarray
    ) -> np.ndarray:
        original_float = original.astype(np.float32)
        shifted_float = shifted.astype(np.float32)
        overlay = np.zeros_like(original_float)
        overlay[..., 1] = original_float[..., 1]
        overlay[..., 0] = 0.6 * shifted_float[..., 0]
        overlay[..., 2] = 0.6 * shifted_float[..., 2]
        return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        obs_dict = deepcopy(obs_dict)

        absolute_obs_dict = dict()
        for key in list(obs_dict.keys()):
            absolute_obs_dict[key] = obs_dict[key].copy()
        # logger.info(f"use_relative_action: {self.use_relative_action}")
        # convert absolute action to relative action
        if self.use_relative_action:
            for key in self.lowdim_keys:
                if "robot_tcp_pose" in key and key in list(obs_dict.keys()):
                    base_absolute_action = obs_dict[key][-1].copy()
                    obs_dict[key] = absolute_actions_to_relative_actions(
                        obs_dict[key], base_absolute_action=base_absolute_action,action_representation=self.action_representation
                    )

        return obs_dict, absolute_obs_dict

    def _log_raw_model_predictions(
        self, device_name: Optional[str], raw_action: np.ndarray
    ) -> None:
        """Log model outputs before post-processing for rerun visualization."""
        if not self.enable_visualization or device_name is None:
            return
        try:
            raw_action = np.asarray(raw_action)
            if raw_action.ndim != 2 or raw_action.shape[1] not in (10, 20):
                return

            timestamp_ns = time.time_ns()
            if raw_action.shape[1] == 10:
                dummy_gripper_left = raw_action[:, 9]
            else:
                dummy_gripper_left = raw_action[:, 18]
            left_positions = raw_action[:, :3]
            visualize_action_time_series(
                "time_series/action_trends",
                device_name,
                "left_arm_raw",
                left_positions,
                dummy_gripper_left,
                timestamp_ns,
            )

            if raw_action.shape[1] == 20:
                right_positions = raw_action[:, 9:12]
                dummy_gripper_right = raw_action[:, 19]
                visualize_action_time_series(
                    "time_series/action_trends",
                    device_name,
                    "right_arm_raw",
                    right_positions,
                    dummy_gripper_right,
                    timestamp_ns,
                )
        except Exception as exc:
            logger.debug(f"Failed to log raw model predictions: {exc}")

    def _log_base_absolute_action(
        self,
        device_name: Optional[str],
        absolute_obs: Dict[str, np.ndarray],
    ) -> None:
        """Visualize the base absolute action that anchors relative predictions."""
        if not self.enable_visualization or device_name is None:
            return
        if not absolute_obs:
            return

        try:
            timestamp_ns = time.time_ns()
            for arm_prefix in ("left", "right"):
                pose_key = f"{arm_prefix}_robot_tcp_pose"
                pose_seq = absolute_obs.get(pose_key)
                if pose_seq is None:
                    continue
                pose_seq = np.asarray(pose_seq)
                if pose_seq.ndim == 1:
                    pose_seq = pose_seq[None, ...]
                if pose_seq.shape[-1] < 3:
                    continue

                base_pose = pose_seq[-1]
                positions = np.asarray(base_pose[:3], dtype=float).reshape(1, 3)

                gripper_key = f"{arm_prefix}_robot_gripper_width"
                gripper_seq = absolute_obs.get(gripper_key)
                if gripper_seq is not None:
                    gripper_seq = np.asarray(gripper_seq)
                    base_gripper = float(np.reshape(gripper_seq[-1], -1)[0])
                else:
                    base_gripper = 0.0
                grippers = np.array([base_gripper], dtype=float)

                visualize_action_time_series(
                    "time_series/action_trends",
                    device_name,
                    f"{arm_prefix}_arm_base_absolute_action",
                    positions,
                    grippers,
                    timestamp_ns,
                )
        except Exception as exc:
            logger.debug(f"Failed to log base absolute action: {exc}")

    def post_process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Post-process the action before sending to the robot
        """

        if action.shape[-1] == 10 or action.shape[-1] == 20:
            # convert to 6D pose
            left_rot_mat_batch = ortho6d_to_rotation_matrix(
                action[:, 3:9]
            )  # (action_steps, 3, 3)
            left_quat_batch = np.array(
                [t3d.quaternions.mat2quat(rot_mat) for rot_mat in left_rot_mat_batch]
            )  # (action_steps, 4)
            left_quat_batch = np.array(
                [process_quaternion(left_quat, "f2l") for left_quat in left_quat_batch]
            )
            left_trans_batch = action[:, :3]  # (action_steps, 3)
            left_action_7d = np.concatenate(
                [left_trans_batch, left_quat_batch], axis=1
            )  # (action_steps, 7)
            if action.shape[-1] == 20:
                right_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 12:18])
                right_quat_batch = np.array(
                    [
                        t3d.quaternions.mat2quat(rot_mat)
                        for rot_mat in right_rot_mat_batch
                    ]
                )
                right_quat_batch = np.array(
                    [
                        process_quaternion(right_quat, "f2l")
                        for right_quat in right_quat_batch
                    ]
                )
                right_trans_batch = action[:, 9:12]
                right_action_7d = np.concatenate(
                    [right_trans_batch, right_quat_batch], axis=1
                )
            else:
                right_action_7d = None
        else:
            raise NotImplementedError
        # add gripper action
        if action.shape[-1] == 10:
            left_action = np.concatenate(
                [
                    left_action_7d,
                    action[:, 9][:, np.newaxis],
                ],
                axis=1,
            )
            right_action = None
        elif action.shape[-1] == 20:
            left_action = np.concatenate(
                    [
                        left_action_7d,
                        action[:, 18][:, np.newaxis],
                    ],
                    axis=1,
                )

            right_action = np.concatenate(
                [
                    right_action_7d,
                    action[:, 19][:, np.newaxis],
                ],
                axis=1,
            )

        else:
            raise NotImplementedError

        if right_action is None:
            action_all = left_action
        else:
            # print(left_action,right_action)
            action_all = np.concatenate([left_action, right_action], axis=-1)
        return action_all

    def predict(
        self, observation, action_configs: List[Dict[str, Any]], chunk_length: int = 1
    ) -> Dict[str, np.ndarray]:
        obs_batch_robot = self.sensor_data_to_umi_data(
            observation, target_frame="robot"
        )
        # logger.info(f"left_tcp_pose: {obs_batch_robot['left_robot_tcp_pose']}")
        process_batch, absolute_process_batch = self.pre_process_obs(obs_batch_robot)

        obs_dict = dict_apply(
            process_batch,
            lambda x: torch.from_numpy(x)
            .unsqueeze(0)
            .to(device="cuda"),  # add batchsize
        )
        # single-arm
        if 'right_robot_tcp_pose' not in obs_dict:
            logger.info(f"left_gripper_width: {obs_dict['left_robot_gripper_width']}")
            # logger.info(f"relative_left_robot_tcp_pose: {obs_dict['left_robot_tcp_pose']}")
            with torch.no_grad():
                action = self._policy.predict_action(obs_dict)
            logger.info(f"policy output action: {action}")
            transformed_action = action["action"][0].cpu().numpy()
            device_name = action_configs[0].get("device_name") if action_configs else None
            self._log_base_absolute_action(device_name, absolute_process_batch)
            self._log_raw_model_predictions(device_name, transformed_action[:10, :])
            if self.use_relative_action:
                base_absolute_action = obs_batch_robot["left_robot_tcp_pose"][-1]
                print(base_absolute_action)
                action_all_abs = relative_actions_to_absolute_actions(
                    transformed_action, base_absolute_action=base_absolute_action, action_representation=self.action_representation
                )
            else:
                action_all_abs = transformed_action
            single_pose = self.post_process_action(action_all_abs)
            # single_pose[:, 7] = np.clip(single_pose[:, 7], 0.0, 0.08)
            return {action_configs[0]["device_name"]: single_pose.astype(np.float64)}
        # dual-arm
        else:
            with torch.no_grad():
                action = self._policy.predict_action(obs_dict)
            logger.info(f"policy output action: {action}")
            transformed_action = action["action"][0].cpu().numpy()
            device_name = (
                action_configs[0].get("device_name") if action_configs else None
            )
            self._log_base_absolute_action(device_name, absolute_process_batch)
            self._log_raw_model_predictions(device_name, transformed_action[:10, :])
            if self.use_relative_action:
                left_base_absolute_action = obs_batch_robot["left_robot_tcp_pose"][-1]
                right_base_absolute_action = obs_batch_robot["right_robot_tcp_pose"][-1]
                base_absolute_action = np.concatenate([left_base_absolute_action,right_base_absolute_action])
                action_all_abs = relative_actions_to_absolute_actions(
                    transformed_action, base_absolute_action=base_absolute_action,action_representation = self.action_representation
                )
            else:
                action_all_abs = transformed_action
            dual_pose = self.post_process_action(action_all_abs)
            # dual_pose[:, 7] = np.clip(dual_pose[:, 7], 0.0, 0.08)
            # dual_pose[:, 15] = np.clip(dual_pose[:, 15], 0.0, 0.08)
            return {action_configs[0]["device_name"]: dual_pose.astype(np.float64)}


    def sensor_data_to_umi_data(
        self, obs_dict: Dict[str, List[np.ndarray]], target_frame: str = "umi"
    ) -> Dict[str, np.ndarray]:
        batch_data: Dict[str, np.ndarray] = {}
        available_cameras = [
            name for name in obs_dict.keys() if "camera" in name.lower()
        ]
        cameras = self.camera_devices or available_cameras
        if self.camera_devices:
            missing_cameras = [
                name for name in self.camera_devices if name not in obs_dict
            ]
            if missing_cameras:
                if not self._missing_camera_warning_issued:
                    logger.warning(
                        "DiffusionPolicyIPhoneUMI: Missing camera observations for %s. "
                        "Available devices: %s",
                        missing_cameras,
                        list(obs_dict.keys()),
                    )
                    self._missing_camera_warning_issued = True
            cameras = [name for name in self.camera_devices if name in obs_dict]
            if not cameras and available_cameras:
                if not self._camera_fallback_warning_issued:
                    logger.warning(
                        "DiffusionPolicyIPhoneUMI: Falling back to detected camera streams %s",
                        available_cameras,
                    )
                    self._camera_fallback_warning_issued = True
                cameras = available_cameras
        if not cameras:
            raise KeyError(
                f"No camera observation data available. Expected devices {self.camera_devices} "
                f"but observation dictionary only has {list(obs_dict.keys())}"
            )
        for idx, camera_name in enumerate(cameras):
            raw_images = obs_dict[camera_name]
            if isinstance(raw_images, np.ndarray):
                image_sequence = [raw_images]
            else:
                image_sequence = [np.asarray(frame) for frame in raw_images]
            processed_images = []

            for image in image_sequence:
                target_h = self.preprocess_resize_height
                target_w = self.preprocess_resize_width
                hi, wi = image.shape[:2]
                if hi != target_h or wi != target_w:
                    image = center_crop_and_resize_image(image, (target_h, target_w))
                # logger.info(image.shape)
                processed_images.append(image)
                # # Debug visualization
                # logger.info(f'shape: {image.shape}')
                # cv2.imshow('debug',image)
                # cv2.waitKey()
            # for id in range(0,2):
            #     cv2.imwrite(f"img_{id}.png", processed_images[id])
            stacked_images = np.stack(processed_images)
            if idx == 0:
                batch_data["left_wrist_img"] = (stacked_images / 255.0).transpose(
                    0, 3, 1, 2
                )
            else:
                batch_data["right_wrist_img"] = (stacked_images / 255.0).transpose(
                    0, 3, 1, 2
                )

        robot_device = self.robot_device
        if robot_device is None:
            robot_device = next(
                iter(
                    [
                        key
                        for key in obs_dict.keys()
                        if key.lower().startswith("robot")
                        or key.lower().startswith("rizon")
                    ]
                ),
                None,
            )
        if robot_device is None:
            raise KeyError("Robot observation data not found in observation dictionary")

        robot_obs_seq = obs_dict[robot_device]
        if isinstance(robot_obs_seq, np.ndarray):
            robot_obs_array = (
                robot_obs_seq if robot_obs_seq.ndim > 1 else robot_obs_seq[None, :]
            )
        else:
            robot_obs_array = np.asarray([np.asarray(obs) for obs in robot_obs_seq])
        if robot_obs_array.ndim == 1:
            robot_obs_array = robot_obs_array[None, :]

        obs_dim = robot_obs_array.shape[-1]
        if obs_dim < 22:
            raise ValueError(
                f"Unsupported robot observation dimension: expected at least 22, got {obs_dim}"
            )
        is_dual_arm = obs_dim >= 44
        self._last_num_robots = (
            2 if is_dual_arm else 1
        )  # remember whether this batch was single or dual arm
        left_tcp_pose_w_gripper = robot_obs_array[:, 14:22]
        left_pos = left_tcp_pose_w_gripper[:, :3]
        # logger.info(f"left_tcp_pose_w_gripper: {left_tcp_pose_w_gripper}")
        left_quat_xyzw = np.stack(
            [
                process_quaternion(left_tcp_pose_w_gripper[i, 3:7], "f2l")
                for i in range(len(left_tcp_pose_w_gripper))
            ]
        )

        right_tcp_pose_w_gripper = right_pos = right_quat_xyzw = None
        if is_dual_arm:
            right_tcp_pose_w_gripper = robot_obs_array[:, 36:44]
            right_pos = right_tcp_pose_w_gripper[:, :3]
            right_quat_xyzw = np.stack(
                [
                    process_quaternion(right_tcp_pose_w_gripper[i, 3:7], "f2l")
                    for i in range(len(right_tcp_pose_w_gripper))
                ]
            )

        if target_frame == "umi" and self.tcp_umi_T is not None:
            left_axis_angle = quaternion_to_axis_angle(left_quat_xyzw)
            left_pos, left_axis_angle = self._robot_pose_to_umi(
                left_pos, left_axis_angle
            )
            left_quat_xyzw = axis_angle_to_quaternion(left_axis_angle)
            if is_dual_arm and right_pos is not None and right_quat_xyzw is not None:
                right_axis_angle = quaternion_to_axis_angle(right_quat_xyzw)
                right_pos, right_axis_angle = self._robot_pose_to_umi(
                    right_pos, right_axis_angle
                )
                right_quat_xyzw = axis_angle_to_quaternion(right_axis_angle)
        left_tcp_pose = quaternion_to_pose9d(
            np.concatenate([left_pos, left_quat_xyzw], axis=-1)
        )
        left_gripper = left_tcp_pose_w_gripper[:, 7:8]
        batch_data["left_robot_tcp_pose"] = left_tcp_pose
        batch_data["left_robot_gripper_width"] = left_gripper
        # batch_data['robot0_gripper_width'] = (left_gripper / 60.0) * 80.0
        if is_dual_arm and right_tcp_pose_w_gripper is not None:
            right_gripper = right_tcp_pose_w_gripper[:, 7:8]
            right_tcp_pose = quaternion_to_pose9d(
                np.concatenate([right_pos, right_quat_xyzw], axis=-1)
            )
            batch_data["right_robot_tcp_pose"] = right_tcp_pose
            batch_data["right_robot_gripper_width"] = right_gripper
        if self.use_inter_gripper:
            base_absolute_action_in_world = homo_matrix_to_pose_9d_batch(
                self.right_robot_base_to_world_transform
                @ pose_3d_9d_to_homo_matrix_batch(
                    batch_data["right_robot_tcp_pose"][-1:]
                )
            )[0]
            left_robot_tcp_pose_in_world = homo_matrix_to_pose_9d_batch(
                self.left_robot_base_to_world_transform
                @ pose_3d_9d_to_homo_matrix_batch(batch_data["left_robot_tcp_pose"])
            )
            batch_data['left_robot_wrt_right_robot_tcp_pose'] = absolute_actions_to_relative_actions(
                left_robot_tcp_pose_in_world, base_absolute_action=base_absolute_action_in_world,action_representation = self.action_representation)
            base_absolute_action_in_world = homo_matrix_to_pose_9d_batch(
                self.left_robot_base_to_world_transform
                @ pose_3d_9d_to_homo_matrix_batch(
                    batch_data["left_robot_tcp_pose"][-1:]
                )
            )[0]
            right_robot_tcp_pose_in_world = homo_matrix_to_pose_9d_batch(
                self.right_robot_base_to_world_transform
                @ pose_3d_9d_to_homo_matrix_batch(batch_data["right_robot_tcp_pose"])
            )
            batch_data["right_robot_wrt_left_robot_tcp_pose"] = (
                absolute_actions_to_relative_actions(
                    right_robot_tcp_pose_in_world,
                    base_absolute_action=base_absolute_action_in_world,
                    action_representation=self.action_representation,
                )
            )
        return batch_data

    def umi_action_to_robot(self, action: np.ndarray) -> np.ndarray:
        robot_actions = []
        n_robots = int(action.shape[-1] // 10)
        for robot_idx in range(n_robots):
            start = robot_idx * 10
            action_pose10d = action[..., start : start + 9]
            action_grip = action[..., start + 9 : start + 10]
            action_pose_mat = pose10d_to_mat(action_pose10d)
            if self.tcp_umi_T is not None:
                action_pose_mat = self._umi_pose_to_robot_mat(action_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
            robot_actions.append(action_pose)
            robot_actions.append(action_grip)
        return np.concatenate(robot_actions, axis=-1)

    def _robot_pose_to_umi(
        self, pos: np.ndarray, axis_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.tcp_umi_T is None:
            return pos, axis_angle
        pose_mat = pose_to_mat(np.concatenate([pos, axis_angle], axis=-1))
        transformed = pose_mat @ self.tcp_umi_T
        pose = mat_to_pose(transformed)
        return pose[:, :3], pose[:, 3:]

    def _umi_pose_to_robot_mat(self, pose_mat: np.ndarray) -> np.ndarray:
        if self.tcp_umi_T is None:
            return pose_mat
        return self.tcp_umi_T @ pose_mat @ self.umi_tcp_T

    def _convert_obs_to_umi(
        self, robot_batch: Dict[str, np.ndarray], num_robots: int
    ) -> Dict[str, np.ndarray]:
        umi_batch: Dict[str, np.ndarray] = {}
        for key, value in robot_batch.items():
            umi_batch[key] = value.copy() if isinstance(value, np.ndarray) else value
        for robot_idx in range(num_robots):
            pos_key = f"robot{robot_idx}_eef_pos"
            rot_key = f"robot{robot_idx}_eef_rot_axis_angle"
            if pos_key not in umi_batch or rot_key not in umi_batch:
                continue  # skip absent arm entries in single-arm mode
            umi_pos, umi_rot = self._robot_pose_to_umi(
                umi_batch[pos_key], umi_batch[rot_key]
            )
            umi_batch[pos_key] = umi_pos
            umi_batch[rot_key] = umi_rot
        return umi_batch
