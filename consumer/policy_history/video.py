#!/usr/bin/env python3
"""
Policy history helper mixins.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from utils.logger_config import logger


class PolicyHistoryVideoMixin:
    """Mixin containing a cohesive subset of policy-history behavior."""

    @staticmethod
    def sanitize_history_video_component(value: str) -> str:
        """Return a filesystem-safe path component for history image output."""
        sanitized = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in value
        )
        return sanitized or "unknown"

    @staticmethod
    def format_optional_history_timestamp(timestamp_ns: Optional[int]) -> str:
        """Format optional timestamps for history image filenames."""
        if timestamp_ns is None:
            return "none"
        try:
            return str(int(timestamp_ns))
        except (TypeError, ValueError):
            return "none"

    @staticmethod
    def convert_history_video_frame(frame: Any) -> Optional[np.ndarray]:
        """Normalize image-like observations to contiguous uint8 RGB frames."""
        frame_array = np.asarray(frame)
        if frame_array.size == 0:
            return None

        if frame_array.ndim == 2:
            frame_array = np.repeat(frame_array[:, :, None], 3, axis=2)
        elif frame_array.ndim == 3:
            if frame_array.shape[2] == 1:
                frame_array = np.repeat(frame_array, 3, axis=2)
            elif frame_array.shape[2] > 3:
                frame_array = frame_array[:, :, :3]
        else:
            return None

        if frame_array.dtype == np.uint8:
            return np.ascontiguousarray(frame_array)

        frame_float = frame_array.astype(np.float32, copy=False)
        max_value = float(np.nanmax(frame_float)) if frame_float.size > 0 else 0.0
        min_value = float(np.nanmin(frame_float)) if frame_float.size > 0 else 0.0
        if max_value <= 1.0 and min_value >= 0.0:
            frame_float = frame_float * 255.0
        frame_uint8 = np.clip(frame_float, 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(frame_uint8)

    def save_history_query_latest_images(
        self,
        *,
        latest_data: Dict[str, Tuple[int, np.ndarray]],
        history_kind: str,
        step: int,
        frame_index: int,
        target_timestamp_ns: int,
        matched_frame_timestamp_ns: Optional[int],
        query_summary_frame_timestamp_ns: int,
        manual_selection: bool,
    ) -> None:
        """Save the latest camera observations used for history query debugging."""
        owner = self.owner
        if not owner.enable_policy_history_video:
            return
        if not latest_data:
            return

        camera_names = list(owner.camera_devices)
        if len(camera_names) == 0:
            camera_names = [
                key for key in latest_data.keys()
                if "camera" in key.lower()
            ]
        if len(camera_names) == 0:
            return

        output_dir = owner.policy_history_video_dir / "query_latest"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(
                "PolicyConnector: Failed to create query latest image directory "
                f"{output_dir}: {exc}"
            )
            return

        mode = "manual" if manual_selection else "auto"
        matched_text = self.format_optional_history_timestamp(matched_frame_timestamp_ns)
        frames: List[np.ndarray] = []
        camera_timestamps: List[int] = []
        for camera_name in camera_names:
            if camera_name not in latest_data:
                continue
            camera_timestamp_ns, image = latest_data[camera_name]
            frame = self.convert_history_video_frame(image)
            if frame is None:
                continue
            frames.append(frame)
            camera_timestamps.append(int(camera_timestamp_ns))

        if not frames:
            return

        max_h = max(int(frame.shape[0]) for frame in frames)
        padded_frames = []
        for frame in frames:
            pad_h = max_h - int(frame.shape[0])
            if pad_h > 0:
                frame = np.pad(frame, ((0, pad_h), (0, 0), (0, 0)), constant_values=0)
            padded_frames.append(frame)

        camera_timestamp_text = (
            str(max(camera_timestamps)) if camera_timestamps else "none"
        )
        query_index = int(owner._policy_history_video_clip_index)
        output_path = (
            output_dir
            / (
                f"{history_kind}_{mode}_query_{query_index:06d}"
                f"_latest_frame_{int(frame_index):03d}"
                f"_step_{int(step):03d}_target_{int(target_timestamp_ns)}"
                f"_matched_{matched_text}"
                f"_latest_{int(query_summary_frame_timestamp_ns)}"
                f"_camera_{camera_timestamp_text}.png"
            )
        )
        try:
            imageio.imwrite(str(output_path), np.hstack(padded_frames))
        except Exception as exc:
            logger.warning(
                "PolicyConnector: Failed to save query latest history image "
                f"{output_path}: {exc}"
            )

    def save_policy_history_keyframe_images(
        self,
        history_spec: Dict[str, Any],
        keyframe_observations: List[Dict[str, np.ndarray]],
    ) -> None:
        """Save policy-history keyframe camera images for debugging."""
        owner = self.owner
        if not owner.enable_policy_history_video:
            return
        if len(keyframe_observations) == 0:
            return

        camera_names = list(owner.camera_devices)
        if len(camera_names) == 0:
            camera_names = [
                key
                for key in keyframe_observations[0].keys()
                if "camera" in key.lower()
            ]
        if len(camera_names) == 0:
            logger.warning(
                "PolicyConnector: History image saving skipped because no camera observation keys were found."
            )
            return

        base_timestamp_ns = int(history_spec.get("base_timestamp_ns", time.time_ns()))
        clip_index = owner._policy_history_video_clip_index
        owner._policy_history_video_clip_index += 1
        step_list = [int(step) for step in history_spec.get("keyframe_after_steps", [])]
        frame_infos = list(history_spec.get("history_image_frame_infos", []))
        first_step = step_list[0] if len(step_list) > 0 else 0
        last_step = step_list[-1] if len(step_list) > 0 else (len(keyframe_observations) - 1)
        clip_dir = (
            owner.policy_history_video_dir
            / f"history_clip_{clip_index:06d}_s{first_step:03d}_e{last_step:03d}_{base_timestamp_ns}"
        )
        clip_dir.mkdir(parents=True, exist_ok=True)

        for camera_name in camera_names:
            safe_camera_name = self.sanitize_history_video_component(camera_name)
            saved_count = 0
            for frame_index, obs in enumerate(keyframe_observations):
                if camera_name not in obs:
                    continue
                frame = self.convert_history_video_frame(obs[camera_name])
                if frame is None:
                    continue
                step = step_list[frame_index] if frame_index < len(step_list) else frame_index
                frame_info = (
                    frame_infos[frame_index] if frame_index < len(frame_infos) else {}
                )
                target_timestamp_ns = frame_info.get("target_timestamp_ns")
                matched_frame_timestamp_ns = frame_info.get(
                    "matched_frame_timestamp_ns"
                )
                device_timestamps = frame_info.get("device_timestamps", {})
                camera_timestamp_ns = None
                if isinstance(device_timestamps, dict):
                    camera_timestamp_ns = device_timestamps.get(camera_name)
                target_text = self.format_optional_history_timestamp(
                    target_timestamp_ns
                )
                matched_text = self.format_optional_history_timestamp(
                    matched_frame_timestamp_ns
                )
                camera_text = self.format_optional_history_timestamp(
                    camera_timestamp_ns
                )
                output_path = (
                    clip_dir
                    / (
                        f"{safe_camera_name}_frame_{frame_index:03d}"
                        f"_step_{step:03d}_target_{target_text}"
                        f"_matched_{matched_text}_camera_{camera_text}.png"
                    )
                )
                try:
                    imageio.imwrite(str(output_path), frame)
                    all_frame_index = owner._policy_history_video_all_counters.get(
                        safe_camera_name,
                        0,
                    )
                    all_output_path = (
                        owner.policy_history_video_all_dir
                        / f"{safe_camera_name}_frame_{all_frame_index:06d}.png"
                    )
                    imageio.imwrite(str(all_output_path), frame)
                    owner._policy_history_video_all_counters[safe_camera_name] = (
                        all_frame_index + 1
                    )
                    saved_count += 1
                except Exception as exc:
                    logger.warning(
                        "PolicyConnector: Failed to save policy history keyframe image "
                        f"{output_path}: {exc}"
                    )

            if saved_count == 0:
                logger.warning(
                    "PolicyConnector: History image saving skipped for camera "
                    f"{camera_name} due to missing valid frames."
                )
                continue
            logger.info(
                "PolicyConnector: Saved policy history keyframe images to "
                f"{clip_dir} (camera={camera_name}, frames={saved_count})"
            )

