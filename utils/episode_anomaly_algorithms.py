#!/usr/bin/env python3
"""Format-neutral episode anomaly detection algorithms.

Author: SuperInference Contributors
"""

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from utils.logger_config import logger


class EpisodeAnomalyAlgorithms:
    """Reusable checks over decoded arrays and timestamp streams."""

    config: Dict[str, Any]

    def _detect_missing_devices(
        self, data_dict: Dict, feat_mapping: Dict, episode_dir: Path
    ) -> List[Dict]:
        """Detect missing devices."""
        anomalies = []

        expected_devices = {
            "pose_unified": "pose sensor",
            "pose_left": "left robot pose",
            "pose_right": "right robot pose",
            "img_left": "left camera",
            "img_right": "right camera",
            "grip_left": "left gripper",
            "grip_right": "right gripper",
        }

        for feat_key, device_name in expected_devices.items():
            if feat_key in feat_mapping:
                device_key = feat_mapping[feat_key]
                if device_key not in data_dict or len(data_dict[device_key]) == 0:
                    anomalies.append(
                        {
                            "severity": "error",
                            "type": "missing_device",
                            "message": f"Device '{device_name}' data missing or empty",
                            "device": device_name,
                            "episode_path": str(episode_dir),
                        }
                    )

        return anomalies

    def _detect_frequency_anomalies(
        self, data_dict: Dict, timestamp_dict: Dict, episode_dir: Path
    ) -> List[Dict]:
        """Detect frequency anomalies."""
        anomalies = []

        expected_fps = self.config.get("expected_fps", {})

        if not expected_fps:
            return anomalies

        # Ensure expected_fps values are proper numeric types
        try:
            for fps_type, fps_range in expected_fps.items():
                if isinstance(fps_range, (list, tuple)) and len(fps_range) == 2:
                    expected_fps[fps_type] = [float(fps_range[0]), float(fps_range[1])]
                else:
                    logger.error(
                        f"Invalid expected_fps format for {fps_type}: {fps_range}, expected [min, max]"
                    )
                    return anomalies
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid expected_fps values: {e}")
            return anomalies

        for device_key, timestamps in timestamp_dict.items():
            if len(timestamps) < 2:
                continue

            duration = timestamps[-1] - timestamps[0]
            if duration > 0:
                actual_fps = (len(timestamps) - 1) / duration

                # Determine expected frequency
                expected_range = None
                if any(cam_key in device_key.lower() for cam_key in ["camera", "cam"]):
                    expected_range = expected_fps.get("image")
                elif any(
                    pose_key in device_key.lower()
                    for pose_key in ["vive", "pose", "robot_left", "robot_right"]
                ):
                    expected_range = expected_fps.get("pose")
                elif "encoder" in device_key.lower() or "grip" in device_key.lower():
                    expected_range = expected_fps.get("grip")

                if expected_range:
                    min_fps = expected_range[0]
                    max_fps = expected_range[1]

                    if actual_fps < min_fps or actual_fps > max_fps:
                        anomalies.append(
                            {
                                "severity": "error",
                                "type": "frequency_anomaly",
                                "message": f"Device '{device_key}' frequency anomaly: {actual_fps:.1f}Hz (expected {expected_range[0]}-{expected_range[1]}Hz)",
                                "device": device_key,
                                "actual_fps": actual_fps,
                                "expected_range": expected_range,
                                "episode_path": str(episode_dir),
                            }
                        )

        return anomalies

    def _detect_length_anomalies(
        self, data_dict: Dict, timestamp_dict: Dict, episode_dir: Path
    ) -> List[Dict]:
        """Detect length anomalies."""
        anomalies = []

        min_frames = self.config.get("min_frames")

        if min_frames is None:
            return anomalies

        # Ensure min_frames is integer
        try:
            min_frames = int(min_frames)
        except (ValueError, TypeError):
            logger.error(f"Invalid min_frames value: {min_frames}, expected integer")
            return anomalies

        # Check minimum recording length
        for device_key, timestamps in timestamp_dict.items():
            if len(timestamps) < min_frames:
                anomalies.append(
                    {
                        "severity": "error",
                        "type": "short_recording",
                        "message": f"Recording too short: {device_key} has only {len(timestamps)} frames (minimum {min_frames} required)",
                        "device": device_key,
                        "frames": len(timestamps),
                        "min_frames": min_frames,
                        "episode_path": str(episode_dir),
                    }
                )

        return anomalies

    def _detect_image_anomalies(
        self,
        data_dict: Dict,
        timestamp_dict: Dict,
        feat_mapping: Dict,
        episode_dir: Path,
    ) -> List[Dict]:
        """Detect image-specific anomalies."""
        anomalies = []

        image_check_ratio = self.config.get("image_check_ratio")

        if image_check_ratio is None:
            return anomalies

        # Ensure image_check_ratio is float
        try:
            image_check_ratio = float(image_check_ratio)
        except (ValueError, TypeError):
            logger.error(
                f"Invalid image_check_ratio value: {image_check_ratio}, expected float"
            )
            return anomalies

        for img_key in ["img_left", "img_right"]:
            if img_key in feat_mapping:
                device_key = feat_mapping[img_key]
                if device_key in data_dict:
                    jpg_data_list = data_dict[device_key]

                    # Empty data should have been caught by _detect_missing_devices
                    if len(jpg_data_list) == 0:
                        logger.error(f"Empty image data for {device_key} ")
                        continue

                    # Sample images to check
                    num_check = max(1, int(len(jpg_data_list) * image_check_ratio))
                    check_indices = np.linspace(
                        0, len(jpg_data_list) - 1, num_check, dtype=int
                    )

                    num_black = 0
                    for idx in check_indices:
                        jpg_bytes = jpg_data_list[idx]

                        # Quick check: very small JPG files are likely black images
                        if (
                            len(jpg_bytes) < 1000
                        ):  # Less than 1KB is likely a black image
                            num_black += 1
                            continue

                        # Decode and check if image is completely black
                        img = cv2.imdecode(jpg_bytes, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Check if image is completely black (all pixels < 5)
                            if np.max(img) < 5:
                                num_black += 1

                    if num_black > 0:
                        black_ratio = num_black / num_check
                        anomalies.append(
                            {
                                "severity": "error",
                                "type": "black_image",
                                "message": f"Camera '{device_key}' detected black images ({num_black}/{num_check} frames, {black_ratio:.1%})",
                                "device": device_key,
                                "num_black": num_black,
                                "num_checked": num_check,
                                "black_ratio": black_ratio,
                                "episode_path": str(episode_dir),
                            }
                        )

        return anomalies

    def _detect_pose_anomalies(
        self,
        data_dict: Dict,
        timestamp_dict: Dict,
        feat_mapping: Dict,
        episode_dir: Path,
    ) -> List[Dict]:
        """Detect pose-specific anomalies."""
        anomalies = []

        zero_data_threshold = self.config.get("zero_data_threshold")
        frozen_data_threshold = self.config.get("frozen_data_threshold")
        frozen_frame_diff_threshold = self.config.get("frozen_frame_diff_threshold")
        max_position_jump = self.config.get("max_position_jump")
        max_velocity = self.config.get("max_velocity")
        max_quaternion_jump = self.config.get("max_quaternion_jump")

        if not all(
            x is not None
            for x in [
                zero_data_threshold,
                frozen_data_threshold,
                frozen_frame_diff_threshold,
                max_position_jump,
            ]
        ):
            return anomalies

        # Ensure all parameters are proper numeric types
        try:
            zero_data_threshold = float(zero_data_threshold)
            frozen_data_threshold = float(frozen_data_threshold)
            frozen_frame_diff_threshold = float(frozen_frame_diff_threshold)
            max_position_jump = float(max_position_jump)
            if max_velocity is not None:
                max_velocity = float(max_velocity)
            if max_quaternion_jump is not None:
                max_quaternion_jump = float(max_quaternion_jump)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid pose anomaly config values: {e}")
            return anomalies

        arm_streams = []
        if "pose_unified" in feat_mapping:
            device_key = feat_mapping["pose_unified"]
            pose_data = data_dict.get(device_key)
            if pose_data is not None:
                if pose_data.ndim != 3 or pose_data.shape[1:] != (2, 7):
                    anomalies.append(
                        {
                            "severity": "error",
                            "type": "invalid_pose_format",
                            "message": f"Invalid pose data format: expected (T, 2, 7), got {pose_data.shape}",
                            "device": device_key,
                            "actual_shape": pose_data.shape,
                            "episode_path": str(episode_dir),
                        }
                    )
                else:
                    pose_timestamps = timestamp_dict.get(device_key)
                    arm_streams.extend(
                        [
                            ("left", pose_data[:, 0, :], pose_timestamps),
                            ("right", pose_data[:, 1, :], pose_timestamps),
                        ]
                    )

        for pose_key, arm_name in [("pose_left", "left"), ("pose_right", "right")]:
            if pose_key not in feat_mapping:
                continue
            device_key = feat_mapping[pose_key]
            pose_data = data_dict.get(device_key)
            if pose_data is None:
                continue
            if pose_data.ndim != 2 or pose_data.shape[1] != 7:
                anomalies.append(
                    {
                        "severity": "error",
                        "type": "invalid_pose_format",
                        "message": f"Invalid {arm_name} pose format: expected (T, 7), got {pose_data.shape}",
                        "device": device_key,
                        "actual_shape": pose_data.shape,
                        "episode_path": str(episode_dir),
                    }
                )
                continue
            arm_streams.append((arm_name, pose_data, timestamp_dict.get(device_key)))

        for arm_name, arm_data, pose_timestamps in arm_streams:
            if len(arm_data) == 0:
                continue
            xyz_data = arm_data[:, :3]
            quat_data = arm_data[:, 3:7]

            xyz_zero_ratio = np.sum(np.abs(xyz_data) < 1e-9) / xyz_data.size
            if xyz_zero_ratio > zero_data_threshold:
                anomalies.append(
                    {
                        "severity": "error",
                        "type": "zero_data",
                        "message": f"Pose {arm_name} arm XYZ data is {xyz_zero_ratio:.2%} zeros",
                        "device": f"pose_{arm_name}",
                        "component": "xyz",
                        "zero_ratio": xyz_zero_ratio,
                        "episode_path": str(episode_dir),
                    }
                )

            quat_zero_ratio = np.sum(np.abs(quat_data) < 1e-9) / quat_data.size
            if quat_zero_ratio > zero_data_threshold:
                anomalies.append(
                    {
                        "severity": "error",
                        "type": "zero_data",
                        "message": f"Pose {arm_name} arm quaternion data is {quat_zero_ratio:.2%} zeros",
                        "device": f"pose_{arm_name}",
                        "component": "quaternion",
                        "zero_ratio": quat_zero_ratio,
                        "episode_path": str(episode_dir),
                    }
                )

            if len(xyz_data) <= 1:
                continue
            position_diffs = np.linalg.norm(xyz_data[1:] - xyz_data[:-1], axis=1)
            jump_indices = np.where(position_diffs > max_position_jump)[0]
            if len(jump_indices) > 0:
                max_jump = float(np.max(position_diffs[jump_indices]))
                anomalies.append(
                    {
                        "severity": "error",
                        "type": "position_jump",
                        "message": f"{arm_name} arm position jumps: {len(jump_indices)} occurrences (max: {max_jump:.4f}m)",
                        "arm": arm_name,
                        "num_jumps": len(jump_indices),
                        "max_jump": max_jump,
                        "episode_path": str(episode_dir),
                    }
                )

            if max_velocity is None or pose_timestamps is None:
                continue
            time_diffs = pose_timestamps[1:] - pose_timestamps[:-1]
            valid_time_mask = time_diffs > 1e-6
            if not np.any(valid_time_mask):
                continue
            velocities = np.zeros_like(position_diffs)
            velocities[valid_time_mask] = (
                position_diffs[valid_time_mask] / time_diffs[valid_time_mask]
            )
            high_velocity_indices = np.where(velocities > max_velocity)[0]
            if len(high_velocity_indices) > 0:
                max_velocity_found = float(np.max(velocities[high_velocity_indices]))
                anomalies.append(
                    {
                        "severity": "error",
                        "type": "high_velocity",
                        "message": f"{arm_name} arm high velocity: {len(high_velocity_indices)} occurrences (max: {max_velocity_found:.2f}m/s, limit: {max_velocity}m/s)",
                        "arm": arm_name,
                        "num_high_velocity": len(high_velocity_indices),
                        "max_velocity_found": max_velocity_found,
                        "velocity_limit": max_velocity,
                        "episode_path": str(episode_dir),
                    }
                )

        return anomalies

    def _detect_grip_anomalies(
        self,
        data_dict: Dict,
        timestamp_dict: Dict,
        feat_mapping: Dict,
        episode_dir: Path,
    ) -> List[Dict]:
        """Detect gripper-specific anomalies."""
        anomalies = []

        zero_data_threshold = self.config.get("zero_data_threshold")
        frozen_data_threshold = self.config.get("frozen_data_threshold")
        frozen_frame_diff_threshold = self.config.get("frozen_frame_diff_threshold")

        if not all(
            x is not None
            for x in [
                zero_data_threshold,
                frozen_data_threshold,
                frozen_frame_diff_threshold,
            ]
        ):
            return anomalies

        # Ensure all parameters are proper numeric types
        try:
            zero_data_threshold = float(zero_data_threshold)
            frozen_data_threshold = float(frozen_data_threshold)
            frozen_frame_diff_threshold = float(frozen_frame_diff_threshold)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid gripper anomaly config values: {e}")
            return anomalies

        for grip_key in ["grip_left", "grip_right"]:
            if grip_key in feat_mapping:
                device_key = feat_mapping[grip_key]
                if device_key in data_dict:
                    grip_data = data_dict[device_key]

                    if len(grip_data) == 0:
                        logger.error(f"Empty gripper data for {device_key}")

                    # Check for zero data
                    zero_ratio = np.sum(np.abs(grip_data) < 1e-9) / grip_data.size
                    if zero_ratio > zero_data_threshold:
                        anomalies.append(
                            {
                                "severity": "error",
                                "type": "zero_data",
                                "message": f"{grip_key} data is {zero_ratio:.2%} zeros",
                                "device": grip_key,
                                "zero_ratio": zero_ratio,
                                "episode_path": str(episode_dir),
                            }
                        )
                        continue

                    # Check for frozen data
                    if len(grip_data) > 1:
                        grip_diffs = np.abs(np.diff(grip_data.flatten()))
                        grip_frozen_frames = grip_diffs < frozen_frame_diff_threshold
                        grip_frozen_ratio = np.sum(grip_frozen_frames) / len(
                            grip_frozen_frames
                        )

                        if grip_frozen_ratio > frozen_data_threshold:
                            mean_val = np.mean(grip_data)
                            anomalies.append(
                                {
                                    "severity": "error",
                                    "type": "gripper_frozen",
                                    "message": f"{grip_key} frozen (no movement detected, {grip_frozen_ratio:.1%} frozen frames)",
                                    "device": grip_key,
                                    "mean_val": float(mean_val),
                                    "frozen_ratio": float(grip_frozen_ratio),
                                    "episode_path": str(episode_dir),
                                }
                            )

        return anomalies
