#!/usr/bin/env python3
"""
Policy history helper mixins.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.shm_utils import (
    MANAGER_HEADER_SIZE,
    SUMMARY_FRAME_HEADER_SIZE,
    get_summary_frame_start_offset,
    unpack_manager_header,
    unpack_summary_frame_header,
)


class PolicyHistoryFramesMixin:
    """Mixin containing a cohesive subset of policy-history behavior."""

    def resolve_keyframe_interval_ns(
        self,
        history_spec: Dict[str, Any],
    ) -> int:
        """Resolve policy-history image keyframe interval in nanoseconds."""
        keyframe_interval_ns = history_spec.get("keyframe_interval_ns")
        if keyframe_interval_ns is not None:
            keyframe_interval_ns = int(keyframe_interval_ns)
            if keyframe_interval_ns > 0:
                return keyframe_interval_ns

        keyframe_interval_s = history_spec.get("keyframe_interval_s")
        if keyframe_interval_s is not None:
            keyframe_interval_ns = int(round(float(keyframe_interval_s) * 1e9))
            if keyframe_interval_ns > 0:
                return keyframe_interval_ns

        sampled_video_fps = history_spec.get("sampled_video_fps")
        if sampled_video_fps is not None:
            sampled_video_fps = float(sampled_video_fps)
            if sampled_video_fps > 0:
                return int(round(1e9 / sampled_video_fps))

        return self.owner.action_step_interval_ns

    @staticmethod
    def resolve_keyframe_target_timestamp_ns(
        *,
        base_timestamp_ns: int,
        step: int,
        interval_ns: int,
        frame_index: int,
        frame_count: int,
        history_kind: str,
        history_spec: Optional[Dict[str, Any]],
    ) -> int:
        """Resolve a target timestamp for an image or pose history keyframe."""
        target_timestamp_ns = int(base_timestamp_ns) + int(step) * int(interval_ns)
        if history_kind not in ("history_image", "history_pose") or history_spec is None:
            return target_timestamp_ns

        start_offset_ns = history_spec.get("history_keyframe_offset_start_ns")
        end_offset_ns = history_spec.get("history_keyframe_offset_end_ns")
        if start_offset_ns is None or end_offset_ns is None:
            return target_timestamp_ns

        if frame_count <= 1:
            ratio = 0.0
        else:
            ratio = float(frame_index) / float(frame_count - 1)
        extra_offset_ns = int(
            round((int(end_offset_ns) - int(start_offset_ns)) * ratio)
        )
        return target_timestamp_ns + extra_offset_ns

    def manual_target_specs(
        self,
        history_spec: Dict[str, Any],
    ) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
        """Build target timestamp specs for manual image and pose selection."""
        image_steps = [int(step) for step in history_spec.get("keyframe_after_steps", [])]
        image_interval_ns = self.resolve_keyframe_interval_ns(history_spec)
        image_base_ns = int(history_spec["base_timestamp_ns"])
        image_targets = [
            {
                "frame_index": idx,
                "step": step,
                "target_timestamp_ns": (
                    self.resolve_keyframe_target_timestamp_ns(
                        base_timestamp_ns=image_base_ns,
                        step=step,
                        interval_ns=image_interval_ns,
                        frame_index=idx,
                        frame_count=len(image_steps),
                        history_kind="history_image",
                        history_spec=history_spec,
                    )
                ),
            }
            for idx, step in enumerate(image_steps)
        ]

        pose_steps = [int(step) for step in history_spec.get("pose_after_steps", [])]
        pose_base = history_spec.get("pose_base_timestamp_ns")
        pose_interval = history_spec.get("pose_interval_ns")
        pose_targets: List[Dict[str, int]] = []
        if pose_steps and pose_base is not None and pose_interval:
            pose_base_ns = int(pose_base)
            pose_interval_ns = int(pose_interval)
            pose_targets = [
                {
                    "frame_index": idx,
                    "step": step,
                    "target_timestamp_ns": (
                        self.resolve_keyframe_target_timestamp_ns(
                            base_timestamp_ns=pose_base_ns,
                            step=step,
                            interval_ns=pose_interval_ns,
                            frame_index=idx,
                            frame_count=len(pose_steps),
                            history_kind="history_pose",
                            history_spec=history_spec,
                        )
                    ),
                }
                for idx, step in enumerate(pose_steps)
            ]
        return image_targets, pose_targets

    def summary_frame_candidates(
        self,
        target_timestamp_ns: int,
        *,
        window_ms: float,
        max_candidates: int,
    ) -> List[Dict[str, Any]]:
        """Return summary-frame candidates around a target timestamp."""
        owner = self.owner
        window_ns = int(float(window_ms) * 1_000_000)
        target_timestamp_ns = int(target_timestamp_ns)
        with owner.summary_shm_lock:
            if owner.summary_shm is None:
                return []
            buf = owner.summary_shm.buf
            manager_header = unpack_manager_header(buf[:MANAGER_HEADER_SIZE])
            device_count = int(manager_header["device_count"])
            buffer_size = int(manager_header["buffer_size"])
            current_frames_count = int(manager_header["current_frames_count"])
            write_index = int(manager_header["write_index"])
            if current_frames_count <= 0:
                return []
            device_keys = list(owner.devices.keys())
            frame_sizes = [owner.devices[key]["frame_size"] for key in device_keys]
            if current_frames_count < buffer_size:
                virtual_indices = range(current_frames_count)
            else:
                virtual_indices = range(write_index, write_index + buffer_size)

            candidates: List[Dict[str, Any]] = []
            for virtual_index in virtual_indices:
                frame_index = int(virtual_index % buffer_size)
                frame_start_offset = get_summary_frame_start_offset(
                    frame_index,
                    device_count,
                    frame_sizes,
                )
                summary_frame_header = unpack_summary_frame_header(
                    buf[
                        frame_start_offset : frame_start_offset
                        + SUMMARY_FRAME_HEADER_SIZE
                    ]
                )
                timestamp_ns = int(summary_frame_header["frame_timestamp_ns"])
                diff_ns = timestamp_ns - target_timestamp_ns
                if abs(diff_ns) <= window_ns:
                    candidates.append(
                        {
                            "frame_index": frame_index,
                            "frame_timestamp_ns": timestamp_ns,
                            "target_timestamp_ns": target_timestamp_ns,
                            "diff_ns": diff_ns,
                            "diff_ms": diff_ns / 1_000_000,
                        }
                    )

        if not candidates:
            nearest = owner.find_frame_by_timestamp(
                target_timestamp_ns,
                tolerance_ms=max(float(window_ms), 1.0),
            )
            if nearest is None:
                return []
            return [
                {
                    "frame_index": int(nearest["frame_index"]),
                    "frame_timestamp_ns": int(nearest["frame_timestamp_ns"]),
                    "target_timestamp_ns": target_timestamp_ns,
                    "diff_ns": int(nearest["timestamp_diff_ns"]),
                    "diff_ms": float(nearest["timestamp_diff_ms"]),
                }
            ]

        if len(candidates) > max_candidates:
            candidates = sorted(candidates, key=lambda item: abs(item["diff_ns"]))[
                :max_candidates
            ]
        return sorted(candidates, key=lambda item: item["frame_timestamp_ns"])

    def read_summary_frame_by_physical_index(
        self,
        frame_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Read a complete summary frame by physical ring-buffer index."""
        owner = self.owner
        with owner.summary_shm_lock:
            result = owner._read_all_device_data_unlocked(
                frame_index=int(frame_index),
                include_frame_timestamp=True,
                physical_frame_index=True,
            )
        if result is None:
            return None
        frame_timestamp_ns, data = result
        if not data:
            return None
        return {
            "frame_index": int(frame_index),
            "frame_timestamp_ns": int(frame_timestamp_ns),
            "data": data,
        }

    def log_history_query_latest_snapshot(
        self,
        *,
        history_kind: str,
        history_spec: Optional[Dict[str, Any]],
        step: int,
        frame_index: int,
        target_timestamp_ns: int,
        matched_frame_timestamp_ns: Optional[int],
        history_timestamp_diff_ns: Optional[int],
        history_timestamp_diff_ms: Optional[float],
        manual_selection: bool,
        save_latest_images: bool = False,
    ) -> None:
        """Log the latest summary snapshot when querying history frames."""
        owner = self.owner
        read_result = owner.read_all_device_data(
            frame_index=-1,
            include_frame_timestamp=True,
            device_names=list(owner.devices.keys()),
        )
        if read_result is None:
            return
        query_summary_frame_timestamp_ns, latest_data = read_result
        if not latest_data:
            return

        query_wall_timestamp_ns = time.time_ns()
        base_timestamp_ns = None
        pose_base_timestamp_ns = None
        if history_spec is not None:
            base_timestamp_ns = history_spec.get("base_timestamp_ns")
            pose_base_timestamp_ns = history_spec.get("pose_base_timestamp_ns")

        for device_name, (timestamp, data) in latest_data.items():
            if not owner._is_robot_state_device(device_name):
                continue
            owner.action_state_xyz_logger.log_state(
                device_name=device_name,
                state=data,
                timestamp_ns=int(timestamp),
                event_type="query_xyz",
                history_kind=history_kind,
                history_step=int(step),
                history_frame_index=int(frame_index),
                history_target_timestamp_ns=int(target_timestamp_ns),
                history_matched_frame_timestamp_ns=matched_frame_timestamp_ns,
                history_timestamp_diff_ns=history_timestamp_diff_ns,
                history_timestamp_diff_ms=history_timestamp_diff_ms,
                history_base_timestamp_ns=base_timestamp_ns,
                history_pose_base_timestamp_ns=pose_base_timestamp_ns,
                query_wall_timestamp_ns=int(query_wall_timestamp_ns),
                query_summary_frame_timestamp_ns=int(query_summary_frame_timestamp_ns),
                manual_selection=bool(manual_selection),
            )

        if save_latest_images:
            self.save_history_query_latest_images(
                latest_data=latest_data,
                history_kind=history_kind,
                step=step,
                frame_index=frame_index,
                target_timestamp_ns=target_timestamp_ns,
                matched_frame_timestamp_ns=matched_frame_timestamp_ns,
                query_summary_frame_timestamp_ns=int(query_summary_frame_timestamp_ns),
                manual_selection=manual_selection,
            )

    def collect_history_frames_by_indices(
        self,
        frame_indices: List[int],
        *,
        history_kind: str,
        history_spec: Dict[str, Any],
        target_specs: List[Dict[str, int]],
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """Collect selected physical summary frames into policy observations."""
        owner = self.owner
        observations: List[Dict[str, np.ndarray]] = []
        pending_state_logs: List[Dict[str, Any]] = []
        pose_robot_timestamps: List[int] = []
        image_frame_infos: List[Dict[str, Any]] = []
        if history_kind == "history_pose":
            history_spec["history_pose_robot_timestamps_ns"] = []
        elif history_kind == "history_image":
            history_spec["history_image_frame_infos"] = []
        for idx, frame_index in enumerate(frame_indices):
            frame_data = self.read_summary_frame_by_physical_index(frame_index)
            if frame_data is None:
                return None
            target_spec = target_specs[idx] if idx < len(target_specs) else {}
            target_timestamp = int(
                target_spec.get("target_timestamp_ns", frame_data["frame_timestamp_ns"])
            )
            step = int(target_spec.get("step", idx))
            timestamp_diff_ns = int(frame_data["frame_timestamp_ns"]) - target_timestamp
            event_prefix = (
                "history_pose" if history_kind == "history_pose" else "history_image"
            )
            history_timestamp_field = f"{event_prefix}_timestamp_ns"
            device_timestamps = {
                key: int(value[0])
                for key, value in frame_data["data"].items()
            }
            observation = {
                key: value[1] for key, value in frame_data["data"].items()
            }
            owner.timestamp_logger.log_event(
                f"{event_prefix}_manual_read",
                legacy_event_type="history_keyframe_read",
                base_timestamp_ns=history_spec.get("base_timestamp_ns"),
                step=step,
                frame_index=idx,
                history_kind=history_kind,
                target_timestamp_ns=target_timestamp,
                matched_frame_index=int(frame_index),
                matched_frame_timestamp_ns=int(frame_data["frame_timestamp_ns"]),
                history_timeline_timestamp_ns=int(frame_data["frame_timestamp_ns"]),
                **{history_timestamp_field: int(frame_data["frame_timestamp_ns"])},
                timestamp_diff_ns=timestamp_diff_ns,
                timestamp_diff_ms=timestamp_diff_ns / 1_000_000,
                manual_selection=True,
                device_timestamps=device_timestamps,
                device_names=list(frame_data["data"].keys()),
            )
            self.log_history_query_latest_snapshot(
                history_kind=history_kind,
                history_spec=history_spec,
                step=step,
                frame_index=idx,
                target_timestamp_ns=target_timestamp,
                matched_frame_timestamp_ns=int(frame_data["frame_timestamp_ns"]),
                history_timestamp_diff_ns=timestamp_diff_ns,
                history_timestamp_diff_ms=timestamp_diff_ns / 1_000_000,
                manual_selection=True,
                save_latest_images=history_kind == "history_image" and idx == 0,
            )
            if history_kind == "history_pose":
                step_robot_timestamps: List[int] = []
                for device_name, data in observation.items():
                    if owner._is_robot_state_device(device_name):
                        timestamp = int(device_timestamps[device_name])
                        step_robot_timestamps.append(timestamp)
                        pending_state_logs.append(
                            {
                                "device_name": device_name,
                                "state": data,
                                "timestamp_ns": timestamp,
                                "event_type": "history_xyz",
                                "history_step": step,
                                "history_frame_index": idx,
                                "history_target_timestamp_ns": target_timestamp,
                                "history_matched_frame_timestamp_ns": int(
                                    frame_data["frame_timestamp_ns"]
                                ),
                                "history_timestamp_diff_ns": timestamp_diff_ns,
                                "history_timestamp_diff_ms": timestamp_diff_ns / 1_000_000,
                                "manual_selection": True,
                            }
                        )
                if step_robot_timestamps:
                    pose_robot_timestamps.append(int(max(step_robot_timestamps)))
            elif history_kind == "history_image":
                for device_name, data in observation.items():
                    if owner._is_robot_state_device(device_name):
                        timestamp = int(device_timestamps[device_name])
                        pending_state_logs.append(
                            {
                                "device_name": device_name,
                                "state": data,
                                "timestamp_ns": target_timestamp,
                                "event_type": "history_image",
                                "history_step": step,
                                "history_frame_index": idx,
                                "history_target_timestamp_ns": target_timestamp,
                                "history_state_timestamp_ns": timestamp,
                                "history_matched_frame_timestamp_ns": int(
                                    frame_data["frame_timestamp_ns"]
                                ),
                                "history_timestamp_diff_ns": timestamp_diff_ns,
                                "history_timestamp_diff_ms": timestamp_diff_ns / 1_000_000,
                                "manual_selection": True,
                            }
                        )
                image_frame_infos.append(
                    {
                        "frame_index": int(idx),
                        "step": int(step),
                        "target_timestamp_ns": int(target_timestamp),
                        "matched_frame_timestamp_ns": int(
                            frame_data["frame_timestamp_ns"]
                        ),
                        "device_timestamps": device_timestamps,
                        "manual_selection": True,
                    }
                )
            observations.append(observation)
        for log_kwargs in pending_state_logs:
            owner.action_state_xyz_logger.log_state(**log_kwargs)
        if history_kind == "history_pose":
            history_spec["history_pose_robot_timestamps_ns"] = pose_robot_timestamps
        elif history_kind == "history_image":
            history_spec["history_image_frame_infos"] = image_frame_infos
        return observations

    def collect_history_frames_at(
        self,
        base_timestamp_ns: int,
        steps: List[int],
        interval_ns: int,
        tolerance_ms: float,
        block: bool = True,
        history_kind: str = "history",
        history_spec: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """Collect one observation per step at ``base + step * interval`` timestamps."""
        owner = self.owner
        observations: List[Dict[str, np.ndarray]] = []
        pending_state_logs: List[Dict[str, Any]] = []
        pose_robot_timestamps: List[int] = []
        image_frame_infos: List[Dict[str, Any]] = []
        if history_kind == "history_image" and history_spec is not None:
            history_spec["history_image_frame_infos"] = []
        elif history_kind == "history_pose" and history_spec is not None:
            history_spec["history_pose_robot_timestamps_ns"] = []
        for idx, step in enumerate(steps):
            target_timestamp = self.resolve_keyframe_target_timestamp_ns(
                base_timestamp_ns=base_timestamp_ns,
                step=step,
                interval_ns=interval_ns,
                frame_index=idx,
                frame_count=len(steps),
                history_kind=history_kind,
                history_spec=history_spec,
            )
            matched_frame = owner.find_frame_by_timestamp(
                target_timestamp,
                tolerance_ms=tolerance_ms,
            )
            if matched_frame is None or "data" not in matched_frame:
                event_prefix = (
                    "history_pose" if history_kind == "history_pose" else "history_image"
                )
                history_timestamp_field = f"{event_prefix}_timestamp_ns"
                owner.timestamp_logger.log_event(
                    f"{event_prefix}_missed",
                    legacy_event_type="history_keyframe_missed",
                    base_timestamp_ns=base_timestamp_ns,
                    step=step,
                    frame_index=idx,
                    history_kind=history_kind,
                    target_timestamp_ns=target_timestamp,
                    history_timeline_timestamp_ns=target_timestamp,
                    **{history_timestamp_field: target_timestamp},
                    interval_ns=interval_ns,
                    tolerance_ms=tolerance_ms,
                    block=block,
                )
                if block:
                    return None
                # Preserve legacy behavior: a late miss can truncate history.
                if idx < len(steps) / 2:
                    return None
                break
            device_timestamps = {
                key: int(value[0])
                for key, value in matched_frame["data"].items()
            }
            observation = {
                key: value[1] for key, value in matched_frame["data"].items()
            }
            event_prefix = (
                "history_pose" if history_kind == "history_pose" else "history_image"
            )
            history_timestamp_field = f"{event_prefix}_timestamp_ns"
            history_timeline_timestamp_ns = matched_frame.get("frame_timestamp_ns")
            owner.timestamp_logger.log_event(
                f"{event_prefix}_read",
                legacy_event_type="history_keyframe_read",
                base_timestamp_ns=base_timestamp_ns,
                step=step,
                frame_index=idx,
                history_kind=history_kind,
                target_timestamp_ns=target_timestamp,
                matched_frame_index=matched_frame.get("frame_index"),
                matched_frame_timestamp_ns=matched_frame.get("frame_timestamp_ns"),
                history_timeline_timestamp_ns=history_timeline_timestamp_ns,
                **{history_timestamp_field: history_timeline_timestamp_ns},
                timestamp_diff_ns=matched_frame.get("timestamp_diff_ns"),
                timestamp_diff_ms=matched_frame.get("timestamp_diff_ms"),
                interval_ns=interval_ns,
                tolerance_ms=tolerance_ms,
                block=block,
                device_timestamps=device_timestamps,
                device_names=list(matched_frame["data"].keys()),
            )
            self.log_history_query_latest_snapshot(
                history_kind=history_kind,
                history_spec=history_spec,
                step=step,
                frame_index=idx,
                target_timestamp_ns=target_timestamp,
                matched_frame_timestamp_ns=matched_frame.get("frame_timestamp_ns"),
                history_timestamp_diff_ns=matched_frame.get("timestamp_diff_ns"),
                history_timestamp_diff_ms=matched_frame.get("timestamp_diff_ms"),
                manual_selection=False,
                save_latest_images=history_kind == "history_image" and idx == 0,
            )
            if history_kind == "history_pose":
                step_robot_timestamps: List[int] = []
                for device_name, data in observation.items():
                    if owner._is_robot_state_device(device_name):
                        timestamp = int(device_timestamps[device_name])
                        step_robot_timestamps.append(timestamp)
                        pending_state_logs.append(
                            {
                                "device_name": device_name,
                                "state": data,
                                "timestamp_ns": timestamp,
                                "event_type": "history_xyz",
                                "history_step": step,
                                "history_frame_index": idx,
                                "history_target_timestamp_ns": target_timestamp,
                                "history_matched_frame_timestamp_ns": matched_frame.get(
                                    "frame_timestamp_ns"
                                ),
                                "history_timestamp_diff_ns": matched_frame.get(
                                    "timestamp_diff_ns"
                                ),
                                "history_timestamp_diff_ms": matched_frame.get(
                                    "timestamp_diff_ms"
                                ),
                            }
                        )
                if step_robot_timestamps:
                    pose_robot_timestamps.append(int(max(step_robot_timestamps)))
            elif history_kind == "history_image" and history_spec is not None:
                for device_name, data in observation.items():
                    if owner._is_robot_state_device(device_name):
                        timestamp = int(device_timestamps[device_name])
                        pending_state_logs.append(
                            {
                                "device_name": device_name,
                                "state": data,
                                "timestamp_ns": int(target_timestamp),
                                "event_type": "history_image",
                                "history_step": step,
                                "history_frame_index": idx,
                                "history_target_timestamp_ns": target_timestamp,
                                "history_state_timestamp_ns": timestamp,
                                "history_matched_frame_timestamp_ns": matched_frame.get(
                                    "frame_timestamp_ns"
                                ),
                                "history_timestamp_diff_ns": matched_frame.get(
                                    "timestamp_diff_ns"
                                ),
                                "history_timestamp_diff_ms": matched_frame.get(
                                    "timestamp_diff_ms"
                                ),
                            }
                        )
                image_frame_infos.append(
                    {
                        "frame_index": int(idx),
                        "step": int(step),
                        "target_timestamp_ns": int(target_timestamp),
                        "matched_frame_timestamp_ns": int(
                            matched_frame.get("frame_timestamp_ns")
                        ),
                        "device_timestamps": device_timestamps,
                        "manual_selection": False,
                    }
                )
            observations.append(observation)
        for log_kwargs in pending_state_logs:
            owner.action_state_xyz_logger.log_state(**log_kwargs)
        if history_spec is not None:
            if history_kind == "history_pose":
                history_spec["history_pose_robot_timestamps_ns"] = pose_robot_timestamps
            elif history_kind == "history_image":
                history_spec["history_image_frame_infos"] = image_frame_infos
        return observations

    def collect_policy_history_observations(
        self,
        history_spec: Dict[str, Any],
        block: bool = True,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """Collect image observations for policy history update."""
        keyframe_after_steps = [
            int(step) for step in history_spec.get("keyframe_after_steps", [])
        ]
        if not keyframe_after_steps:
            return []
        return self.collect_history_frames_at(
            base_timestamp_ns=int(history_spec["base_timestamp_ns"]),
            steps=keyframe_after_steps,
            interval_ns=self.resolve_keyframe_interval_ns(history_spec),
            tolerance_ms=float(history_spec.get("tolerance_ms", 80.0)),
            block=block,
            history_kind="history_image",
            history_spec=history_spec,
        )

    def collect_policy_history_poses(
        self,
        history_spec: Dict[str, Any],
        block: bool = True,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """Collect robot-state observations on the denser action timeline."""
        pose_after_steps = [
            int(step) for step in history_spec.get("pose_after_steps", [])
        ]
        pose_base = history_spec.get("pose_base_timestamp_ns")
        pose_interval_ns = history_spec.get("pose_interval_ns")
        if not pose_after_steps or pose_base is None or not pose_interval_ns:
            return None
        history_spec["history_pose_robot_timestamps_ns"] = []
        return self.collect_history_frames_at(
            base_timestamp_ns=int(pose_base),
            steps=pose_after_steps,
            interval_ns=int(pose_interval_ns),
            tolerance_ms=float(history_spec.get("tolerance_ms", 80.0)),
            block=block,
            history_kind="history_pose",
            history_spec=history_spec,
        )

