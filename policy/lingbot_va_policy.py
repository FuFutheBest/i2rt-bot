#!/usr/bin/env python3
"""
LingBot-VA websocket policy adapter for SuperInference.

This client keeps LingBot-VA model state on a separate socket server and maps
SuperInference observations/actions to the server protocol.
"""

from __future__ import annotations

import io
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import websockets.sync.client
from scipy.spatial.transform import Rotation

from policy.base import BasePolicy
from utils.dsl_pose_utils import convert_pose_mat_rep, mat_to_pose, pose_to_mat
from utils.logger_config import logger
from utils.latency_tracker import LatencyTracker
from utils.msgpack_numpy import Packer, unpackb
from utils.transform import process_quaternion


class LingBotVAPolicy(BasePolicy):
    """Websocket client wrapper for LingBot-VA inference server.
    
    Args:
        remote_image_encoding: Encoding format for images sent to remote inference server.
            Supported options: "raw" (no compression), "jpeg" (JPEG compression).
            Default is "raw".
        remote_image_jpeg_quality: JPEG compression quality (1-100) when using JPEG encoding.
            Only effective when remote_image_encoding is set to "jpeg".
            Default is 85.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18080,
        api_key: Optional[str] = None,
        connect_timeout: float = 300.0,
        task: str = "",
        camera_devices: Optional[List[str]] = None,
        model_camera_keys: Optional[List[str]] = None,
        robot_device: Optional[str] = None,
        frame_chunk_size: int = 4,
        history_update_tolerance_ms: float = 50.0,
        history_anchor_mode: str = "arrival",
        history_arrival_pos_tolerance_m: float = 0.02,
        history_arrival_rot_tolerance_rad: float = 0.2,
        history_arrival_timeout_s: float = 3.0,
        history_now_base_offset_ns: int = 300_000_000,
        history_now_last_keyframe_offset_ns: Optional[int] = None,
        history_now_continuous_timeline: bool = False,
        history_now_continuity_gap_ns: int = 0,
        history_state_from_robot_pose: bool = True,
        skip_last_obs_latents: int = 0,
        stateless_reset_each_predict: bool = False,
        reset_on_load: bool = True,
        gripper_max_width: float = 0.11,
        normalize_gripper_width: bool = True,
        print_predicted_gripper_action: bool = False,
        block: bool = True,
        execution_fps: Optional[float] = None,
        enable_rtc: bool = False,
        rtc_latency_window: int = 100,
        rtc_latency_stat: str = "mean",
        rtc_latency_percentile: float = 0.95,
        rtc_default_delay_steps: int = 4,
        rtc_max_guidance_weight: float = 10.0,
        rtc_prefix_attention_schedule: str = "exp",
        rtc_guidance_full_steps_override: Optional[int] = None,
        rtc_guidance_decay_steps_override: Optional[int] = None,
        remote_image_encoding: str = "raw",
        remote_image_jpeg_quality: int = 85,
        **_: Any,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = int(port)
        self.api_key = api_key
        self.connect_timeout = float(connect_timeout)
        self.task = task
        self.camera_devices = list(camera_devices) if camera_devices else []
        self.model_camera_keys = list(model_camera_keys) if model_camera_keys else []
        self.robot_device = robot_device
        self.source_action_per_frame = 1
        self.actions_per_latent_frame = 1
        self.source_video_frame_stride = 1
        self.video_frames_per_latent = 1
        self.sampled_video_fps: Optional[float] = None
        self.frame_chunk_size = int(frame_chunk_size)
        self.history_update_tolerance_ms = float(history_update_tolerance_ms)
        self.history_anchor_mode = str(history_anchor_mode).lower()
        self.history_arrival_pos_tolerance_m = float(history_arrival_pos_tolerance_m)
        self.history_arrival_rot_tolerance_rad = float(history_arrival_rot_tolerance_rad)
        self.history_arrival_timeout_s = float(history_arrival_timeout_s)
        self.history_now_base_offset_ns = int(history_now_base_offset_ns)
        if history_now_last_keyframe_offset_ns is None:
            history_now_last_keyframe_offset_ns = self.history_now_base_offset_ns
        self.history_now_last_keyframe_offset_ns = int(
            history_now_last_keyframe_offset_ns
        )
        self.history_now_continuous_timeline = bool(history_now_continuous_timeline)
        self.history_now_continuity_gap_ns = int(history_now_continuity_gap_ns)
        self.history_state_from_robot_pose = bool(history_state_from_robot_pose)
        self.skip_last_obs_latents = int(skip_last_obs_latents)
        if self.skip_last_obs_latents < 0:
            raise ValueError("skip_last_obs_latents must be >= 0")
        self.stateless_reset_each_predict = bool(stateless_reset_each_predict)
        self.reset_on_load = bool(reset_on_load)
        self.relative_pose_frame: Optional[str] = None
        self.action_quaternion_order: Optional[str] = None
        self.action_condition_mode: Optional[str] = None
        self.gripper_max_width = float(gripper_max_width)
        self.normalize_gripper_width = bool(normalize_gripper_width)
        self.print_predicted_gripper_action = bool(print_predicted_gripper_action)
        self.block = bool(block)
        self.execution_fps = (
            float(execution_fps)
            if execution_fps is not None and float(execution_fps) > 0
            else None
        )
        self.enable_rtc = bool(enable_rtc)
        self.rtc_latency_stat = str(rtc_latency_stat).lower()
        self.rtc_latency_percentile = float(rtc_latency_percentile)
        self.rtc_default_delay_steps = max(int(rtc_default_delay_steps), 0)
        self.rtc_max_guidance_weight = float(rtc_max_guidance_weight)
        self.rtc_prefix_attention_schedule = str(rtc_prefix_attention_schedule)
        self.rtc_guidance_full_steps_override = self._parse_optional_nonnegative_int(
            rtc_guidance_full_steps_override,
            "rtc_guidance_full_steps_override",
        )
        self.rtc_guidance_decay_steps_override = self._parse_optional_nonnegative_int(
            rtc_guidance_decay_steps_override,
            "rtc_guidance_decay_steps_override",
        )
        self._latency_tracker = LatencyTracker(maxlen=max(int(rtc_latency_window), 1))
        self.remote_image_encoding = str(remote_image_encoding).lower()
        self.remote_image_jpeg_quality = max(int(remote_image_jpeg_quality), 1)

        self.obs_horizon = 1
        self.min_infer_obs = 1
        self._uri = f"ws://{host}:{self.port}"
        self._packer = Packer()
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: Dict[str, Any] = {}
        self._last_server_action: Optional[np.ndarray] = None
        self._last_history_action: Optional[np.ndarray] = None
        self._last_history_steps: List[int] = []
        self._last_pose_steps: List[int] = []
        self._last_action_target_poses: Optional[List[np.ndarray]] = None
        self._last_reference_robot_state: Optional[np.ndarray] = None
        self._last_history_relative_chunk: Optional[np.ndarray] = None
        self._prediction_context_timestamp_ns: Optional[int] = None
        self._last_history_end_target_ns: Optional[int] = None
        self._last_kv_cache_update_summary: Dict[str, Any] = {}
        self._is_first_prediction = True
        self._initial_robot_state: Optional[np.ndarray] = None
        self._rtc_last_overlap_steps: Optional[int] = None
        if self.enable_rtc:
            logger.info(
                "LingBotVAPolicy RTC enabled: "
                f"execution_fps={self.execution_fps}, "
                f"latency_stat={self.rtc_latency_stat}, "
                f"default_delay_steps={self.rtc_default_delay_steps}, "
                f"schedule={self.rtc_prefix_attention_schedule}, "
                f"guidance_full_steps_override={self.rtc_guidance_full_steps_override}, "
                f"guidance_decay_steps_override={self.rtc_guidance_decay_steps_override}"
            )

    def load(self) -> None:
        """Connect to the LingBot-VA server and optionally reset its state."""
        self._ws, self._server_metadata = self._wait_for_server()
        self._sync_server_timing_metadata()
        logger.info(f"LingBotVAPolicy connected to {self._uri}: {self._server_metadata}")
        if self.reset_on_load:
            self.reset()
        self.initialized = True

    def _sync_server_timing_metadata(self) -> None:
        self.frame_chunk_size = int(self._server_metadata["frame_chunk_size"])
        self.source_action_per_frame = int(self._server_metadata["source_action_per_frame"])
        self.source_video_frame_stride = int(self._server_metadata["source_video_frame_stride"])
        self.actions_per_latent_frame = int(self._server_metadata["actions_per_latent_frame"])
        self.video_frames_per_latent = int(self._server_metadata["video_frames_per_latent"])
        self.sampled_video_fps = float(self._server_metadata["sampled_video_fps"])
        self.action_condition_mode = str(
            self._server_metadata.get("action_condition_mode", "")
        )
        if self.skip_last_obs_latents >= self.frame_chunk_size:
            raise ValueError(
                "skip_last_obs_latents must be smaller than frame_chunk_size: "
                f"{self.skip_last_obs_latents} >= {self.frame_chunk_size}"
            )
        logger.info(
            "LingBotVAPolicy action timing: "
            f"source_action_per_frame={self.source_action_per_frame}, "
            f"source_video_frame_stride={self.source_video_frame_stride}, "
            f"video_frames_per_latent={self.video_frames_per_latent}, "
            f"actions_per_latent_frame={self.actions_per_latent_frame}, "
            f"frame_chunk_size={self.frame_chunk_size}, "
            f"skip_last_obs_latents={self.skip_last_obs_latents}, "
            f"sampled_video_fps={self.sampled_video_fps}"
        )
        self.relative_pose_frame = str(self._server_metadata["relative_pose_frame"])
        logger.info(
            "LingBotVAPolicy relative pose frame from training metadata: "
            f"relative_pose_frame={self.relative_pose_frame}"
        )
        self.action_quaternion_order = str(self._server_metadata["action_quaternion_order"])
        logger.info(
            "LingBotVAPolicy action quaternion order from training metadata: "
            f"action_quaternion_order={self.action_quaternion_order}"
        )

    def close(self) -> None:
        """Close websocket connection."""
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception as exc:
            logger.debug(f"LingBotVAPolicy websocket close ignored: {exc}")
        finally:
            self._ws = None

    def __del__(self) -> None:
        self.close()

    def reset(self) -> None:
        """Reset server-side KV cache and episode state."""
        if self._ws is None:
            return
        self._send_request({"reset": True, "prompt": self.task})
        self._last_server_action = None
        self._last_history_action = None
        self._last_history_steps = []
        self._last_pose_steps = []
        self._last_action_target_poses = None
        self._last_reference_robot_state = None
        self._last_history_relative_chunk = None
        self._last_history_end_target_ns = None
        self._last_kv_cache_update_summary = {}
        self._is_first_prediction = True
        self._initial_robot_state = None
        self._rtc_last_overlap_steps = None
        logger.info("LingBotVAPolicy reset server state")

    def recover_stateless(self) -> None:
        """Fallback hook used by PolicyConnector when history collection fails."""
        self.reset()

    def set_prediction_context(self, timestamp_ns: int) -> None:
        """Save the observation timestamp."""
        self._prediction_context_timestamp_ns = int(timestamp_ns)

    def get_history_update_spec(
        self,
        inference_start_timestamp_ns: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Tell PolicyConnector which keyframes are needed for the KV-cache update.

        In "arrival" anchor mode the keyframes are anchored backward from the moment the
        arm physically reaches the last action of the chunk, so ``base_timestamp_ns`` is
        left for PolicyConnector to resolve once arrival is detected. In "now" mode the
        history targets are anchored at prediction time and sampled forward.
        """
        if self.stateless_reset_each_predict or not self._last_history_steps:
            return None
        # scale_factor = 1.5
        scale_factor = 1.0
        video_interval_s = 1.0 / self.sampled_video_fps * scale_factor
        keyframe_interval_ns = int(round(video_interval_s * 1e9))
        spec = {
            "anchor_mode": self.history_anchor_mode,
            "keyframe_after_steps": list(self._last_history_steps),
            "tolerance_ms": self.history_update_tolerance_ms,
            "sampled_video_fps": self.sampled_video_fps,
            "keyframe_interval_s": video_interval_s,
            "keyframe_interval_ns": int(round(video_interval_s * 1e9)),
        }
        # Robot poses live on the (denser) action timeline: images are sampled at the
        # video frame rate, poses at the action rate. Each latent spans the same wall
        # time (video_frames_per_latent images == actions_per_latent_frame poses), so the
        # pose interval is the latent duration divided by actions_per_latent_frame.
        if self._last_pose_steps and self.history_state_from_robot_pose:
            latent_duration_s = self.video_frames_per_latent * video_interval_s
            pose_interval_s = latent_duration_s / max(int(self.actions_per_latent_frame), 1)
            spec["pose_after_steps"] = list(self._last_pose_steps)
            spec["pose_interval_s"] = pose_interval_s
            spec["pose_interval_ns"] = int(round(pose_interval_s * 1e9))
        if self.history_anchor_mode == "arrival":
            # base_timestamp_ns is resolved by PolicyConnector when the arm arrives.
            spec["arrival_timeout_ns"] = int(round(self.history_arrival_timeout_s * 1e9))
        else:
            # Anchor at prediction time and look forward. Image keyframe targets can
            # stretch linearly from the base offset to the last-keyframe offset.
            spec["base_timestamp_ns"] = time.time_ns() + self.history_now_base_offset_ns
            spec["history_keyframe_offset_start_ns"] = self.history_now_base_offset_ns
            spec["history_keyframe_offset_end_ns"] = (
                self.history_now_last_keyframe_offset_ns
            )
            self._last_history_end_target_ns = self._resolve_history_target_timestamp_ns(
                base_timestamp_ns=int(spec["base_timestamp_ns"]),
                step=int(self._last_history_steps[-1]),
                interval_ns=keyframe_interval_ns,
                frame_index=len(self._last_history_steps) - 1,
                frame_count=len(self._last_history_steps),
            )
        return spec

    def _resolve_default_now_history_base_ns(
        self,
        inference_start_timestamp_ns: Optional[int],
    ) -> int:
        """Resolve the legacy now-mode base timestamp."""
        if self.async_base_timestamp and inference_start_timestamp_ns is not None:
            return int(inference_start_timestamp_ns - 530000000)
        return time.time_ns() + self.history_now_base_offset_ns

    def _resolve_now_history_base_ns(
        self,
        default_base_timestamp_ns: int,
    ) -> int:
        """Choose the now-mode base timestamp.

        In continuous mode, each new history segment starts at the previous segment's
        end target. This keeps history_image/history_pose close across chunks even
        when KV-cache updates and inference add serial delay before the next predict.
        """
        if (
            not self.history_now_continuous_timeline
            or self._last_history_end_target_ns is None
        ):
            return int(default_base_timestamp_ns)
        return int(self._last_history_end_target_ns) + self.history_now_continuity_gap_ns

    def _resolve_history_target_timestamp_ns(
        self,
        *,
        base_timestamp_ns: int,
        step: int,
        interval_ns: int,
        frame_index: int,
        frame_count: int,
    ) -> int:
        """Mirror PolicyConnector's now-mode history target timestamp formula."""
        target_timestamp_ns = int(base_timestamp_ns) + int(step) * int(interval_ns)
        if frame_count <= 1:
            ratio = 0.0
        else:
            ratio = float(frame_index) / float(frame_count - 1)
        extra_offset_ns = int(
            round(
                (
                    self.history_now_last_keyframe_offset_ns
                    - self.history_now_base_offset_ns
                )
                * ratio
            )
        )
        return target_timestamp_ns + extra_offset_ns

    def is_history_anchor_reached(self, observation: Dict[str, Any]) -> bool:
        """Return True when the arm has reached the last action pose of the chunk.

        Used by PolicyConnector to backward-anchor history keyframes. Compares the
        current per-arm pose against the cached final-action pose using position and
        orientation tolerances. Returns False when the target or current state cannot
        be verified, so arrival-mode history is anchored only after a real arrival.
        """
        if self.history_anchor_mode != "arrival":
            return True
        if not self._last_action_target_poses:
            logger.warning("LingBotVAPolicy arrival check skipped: no target pose cached.")
            return False
        try:
            robot_state = self._extract_robot_state(observation)
            for arm_index, target_pose in enumerate(self._last_action_target_poses):
                current_pose = self._extract_arm_current_pose(robot_state, arm_index)
                pos_err = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
                if pos_err > self.history_arrival_pos_tolerance_m:
                    return False
                rot_err = float(
                    (
                        Rotation.from_rotvec(current_pose[3:6]).inv()
                        * Rotation.from_rotvec(target_pose[3:6])
                    ).magnitude()
                )
                if rot_err > self.history_arrival_rot_tolerance_rad:
                    return False
            return True
        except Exception as exc:
            logger.warning(f"LingBotVAPolicy arrival check failed ({exc}); waiting.")
            return False

    def update_history_with_observations(
        self,
        keyframe_observations: List[Dict[str, np.ndarray]],
        pose_observations: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> None:
        """Send executed keyframe observations and the executed state to update KV cache.

        ``keyframe_observations`` carry the camera images (video timeline) while
        ``pose_observations`` carry the robot states sampled on the denser action
        timeline. The relative ``state`` is rebuilt from the actual robot poses; if that
        fails we fall back to the predicted relative action chunk.
        """
        if self.stateless_reset_each_predict:
            return
        if self._last_history_action is None:
            logger.warning("LingBotVAPolicy history update skipped: no previous action")
            return
        if len(keyframe_observations) == 0:
            logger.warning("LingBotVAPolicy history update skipped: no keyframes")
            return

        state = None
        if self.history_state_from_robot_pose and pose_observations:
            video_frames_per_latent = max(int(self.video_frames_per_latent), 1)
            expected_latents = len(keyframe_observations) // video_frames_per_latent
            state = self._build_relative_state_from_poses(
                pose_observations, expected_latents
            )
        if state is None:
            self._last_history_relative_chunk = None
            state = self._last_history_action
            state_source = "predicted_action"
        else:
            state_source = "robot_pose"

        payload = {
            "compute_kv_cache": True,
            "obs": [self._build_server_image_dict(obs) for obs in keyframe_observations],
            "state": state,
        }
        logger.info(
            "LingBotVAPolicy sends history update: "
            f"history_frames={len(keyframe_observations)}, "
            f"pose_frames={len(pose_observations) if pose_observations else 0}, "
            f"state_source={state_source}, "
            f"state_shape={np.asarray(state).shape}"
        )
        kv_cache_trigger_timestamp_ns = time.time_ns()
        self._last_kv_cache_update_summary = {
            "trigger_timestamp_ns": kv_cache_trigger_timestamp_ns,
            "history_frames": len(keyframe_observations),
            "pose_frames": len(pose_observations) if pose_observations else 0,
            "state_source": state_source,
            "state_shape": tuple(np.asarray(state).shape),
            "success": False,
        }
        try:
            self._send_request(payload)
        except Exception:
            kv_cache_end_timestamp_ns = time.time_ns()
            self._last_kv_cache_update_summary.update(
                {
                    "end_timestamp_ns": kv_cache_end_timestamp_ns,
                    "duration_ns": (
                        kv_cache_end_timestamp_ns - kv_cache_trigger_timestamp_ns
                    ),
                    "success": False,
                }
            )
            raise
        kv_cache_end_timestamp_ns = time.time_ns()
        self._last_kv_cache_update_summary.update(
            {
                "end_timestamp_ns": kv_cache_end_timestamp_ns,
                "duration_ns": (
                    kv_cache_end_timestamp_ns - kv_cache_trigger_timestamp_ns
                ),
                "success": True,
            }
        )
        logger.info(
            "LingBotVAPolicy updated server KV cache with "
            f"{len(keyframe_observations)} history frames"
        )

    def _absolute_pose_to_relative_mat(
        self,
        absolute_mat: np.ndarray,
        reference_mat: np.ndarray,
    ) -> np.ndarray:
        """Inverse of :meth:`_relative_pose_to_absolute_mat`."""
        if self.relative_pose_frame == "local_frame":
            return convert_pose_mat_rep(
                absolute_mat,
                reference_mat,
                pose_rep="relative",
                backward=False,
            )

        relative_mat = absolute_mat.copy()
        relative_mat[:, :3, 3] = absolute_mat[:, :3, 3] - reference_mat[:3, 3]
        relative_mat[:, :3, :3] = (
            np.linalg.inv(reference_mat[:3, :3]) @ absolute_mat[:, :3, :3]
        )
        return relative_mat

    def _relative_quat_xyzw_to_action_order(self, quat_xyzw: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_extract_relative_quat_xyzw` (per-arm, shape [T, 4])."""
        order = str(self.action_quaternion_order).lower()
        if order == "xyzw":
            return quat_xyzw
        if order == "wxyz":
            return np.asarray(
                [process_quaternion(q, "l2f") for q in quat_xyzw], dtype=np.float64
            )
        if order in ("xyz", "zyx"):
            euler = Rotation.from_quat(quat_xyzw).as_euler(order)
            pad = np.zeros((euler.shape[0], 1), dtype=np.float64)
            return np.concatenate([euler, pad], axis=-1)
        raise ValueError(f"Invalid LingBot-VA action_quaternion_order: {order!r}")

    def _arm_relative_channels_from_states(
        self,
        robot_states: List[np.ndarray],
        arm_index: int,
        reference_robot_state: np.ndarray,
    ) -> np.ndarray:
        """Build [T, 8] relative action channels for one arm from actual robot poses."""
        reference_pose = self._extract_arm_current_pose(
            reference_robot_state, arm_index
        )
        reference_mat = pose_to_mat(reference_pose)

        absolute_poses = np.stack(
            [self._extract_arm_current_pose(rs, arm_index) for rs in robot_states],
            axis=0,
        )
        absolute_mats = pose_to_mat(absolute_poses)
        relative_mats = self._absolute_pose_to_relative_mat(absolute_mats, reference_mat)
        relative_pose = mat_to_pose(relative_mats)
        rel_pos = relative_pose[:, :3]
        rel_quat_xyzw = Rotation.from_rotvec(relative_pose[:, 3:6]).as_quat()
        # Enforce the qw >= 0 hemisphere to match the training convention (the data
        # pipeline negates quaternions with negative scalar part). q and -q encode the
        # same rotation, so negating when qw < 0 keeps the pose unchanged while aligning
        # the action distribution with what the model was trained on.
        rel_quat_xyzw[rel_quat_xyzw[:, 3] < 0] *= -1.0
        rot_channels = self._relative_quat_xyzw_to_action_order(rel_quat_xyzw)

        gripper_offset = 21 if arm_index == 0 else 43
        gripper = np.asarray(
            [rs[gripper_offset] for rs in robot_states], dtype=np.float64
        ).reshape(-1, 1)
        return np.concatenate([rel_pos, rot_channels, gripper], axis=-1)

    def _build_relative_state_from_poses(
        self,
        pose_observations: List[Dict[str, np.ndarray]],
        expected_latents: int,
    ) -> Optional[np.ndarray]:
        """Construct the raw relative ``state`` ([D, F, H]) from actual robot poses.

        Poses are sampled on the action timeline (``actions_per_latent_frame`` per
        latent). Returns None (and the caller falls back to the predicted action chunk)
        when the state cannot be safely constructed or aligned with the video latents.
        """
        self._last_history_relative_chunk = None
        if self._last_history_action is None:
            return None

        target_dim, target_frames, steps_per_frame = self._last_history_action.shape
        if steps_per_frame <= 0:
            return None

        try:
            robot_states = [self._extract_robot_state(obs) for obs in pose_observations]
            num_poses = (len(robot_states) // steps_per_frame) * steps_per_frame
            if num_poses <= 0:
                return None
            latents = num_poses // steps_per_frame
            # The pose-derived latent count must match the video-derived latent count the
            # server infers from the images (state_latent_count), otherwise the cache
            # frames and action conditioning would be misaligned.
            if latents != expected_latents:
                logger.warning(
                    "LingBotVAPolicy state-from-pose skipped: pose latents "
                    f"({latents}) != image latents ({expected_latents})."
                )
                return None
            robot_states = robot_states[:num_poses]

            n_arms = 2 if target_dim >= 16 else 1
            reference_robot_state = robot_states[0]
            if n_arms == 2 and reference_robot_state.shape[0] < 44:
                return None

            arm_channels = [
                self._arm_relative_channels_from_states(
                    robot_states,
                    arm_index,
                    reference_robot_state,
                )
                for arm_index in range(n_arms)
            ]
            # [T, D_built]
            per_step = np.concatenate(arm_channels, axis=-1)
        except Exception as exc:
            self._last_history_relative_chunk = None
            logger.warning(
                f"LingBotVAPolicy failed to build state from robot poses ({exc}); "
                "falling back to predicted action."
            )
            return None

        # t = f * H + h -> [latents, H, D_built] -> [D_built, latents, H]
        state = per_step.reshape(latents, steps_per_frame, per_step.shape[-1])
        state = np.moveaxis(state, -1, 0)
        # Match the channel dimension expected by the server.
        if state.shape[0] != target_dim:
            state = self._fit_state_dim(state, target_dim)
        if state.shape[1] < target_frames:
            missing_frames = target_frames - state.shape[1]
            history_prefix = np.asarray(
                self._last_history_action[:, :missing_frames, :],
                dtype=state.dtype,
            )
            if history_prefix.shape[0] != state.shape[0]:
                history_prefix = self._fit_state_dim(history_prefix, state.shape[0])
            state = np.concatenate([history_prefix, state], axis=1)
        elif state.shape[1] > target_frames:
            state = state[:, -target_frames:, :]
        self._last_history_relative_chunk = np.moveaxis(state, 0, -1).reshape(
            state.shape[1] * steps_per_frame,
            state.shape[0],
        ).copy()
        return np.ascontiguousarray(state, dtype=np.float64)

    def _fit_state_dim(self, state: np.ndarray, target_dim: int) -> np.ndarray:
        current_dim = state.shape[0]
        if current_dim > target_dim:
            return state[:target_dim]
        pad = np.zeros(
            (target_dim - current_dim,) + state.shape[1:], dtype=state.dtype
        )
        return np.concatenate([state, pad], axis=0)

    def _wait_for_server(
        self,
    ) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        logger.info(f"Waiting for LingBot-VA server at {self._uri}...")
        start_time = time.time()
        for key in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(key, None)

        while True:
            if time.time() - start_time > self.connect_timeout:
                raise TimeoutError(
                    f"Failed to connect LingBot-VA server within {self.connect_timeout}s"
                )
            try:
                headers = (
                    {"Authorization": f"Api-Key {self.api_key}"}
                    if self.api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=20,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except Exception as exc:
                logger.info(f"Still waiting for LingBot-VA server: {exc}")
                time.sleep(2.0)

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        assert self._ws is not None
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"LingBot-VA server error:\n{response}")
        result = unpackb(response)
        if isinstance(result, dict) and result.get("status") == "error":
            raise RuntimeError(f"LingBot-VA server error: {result.get('error')}")
        return result

    def _extract_latest_frame(self, data: Any) -> np.ndarray:
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Observation history list is empty")
            return np.asarray(data[-1])
        array = np.asarray(data)
        if array.ndim >= 4:
            return np.asarray(array[-1])
        return array

    def _resolve_camera_pairs(self, observation: Dict[str, Any]) -> List[Tuple[str, str]]:
        if self.camera_devices:
            camera_devices = self.camera_devices
        else:
            camera_devices = [key for key in observation.keys() if "camera" in key.lower()]

        if len(camera_devices) == 0:
            raise ValueError("No camera observation found for LingBot-VA inference")

        if self.model_camera_keys:
            if len(self.model_camera_keys) != len(camera_devices):
                raise ValueError(
                    "model_camera_keys length must match camera_devices length: "
                    f"{len(self.model_camera_keys)} != {len(camera_devices)}"
                )
            return list(zip(camera_devices, self.model_camera_keys))

        return [(camera_name, camera_name) for camera_name in camera_devices]

    def _normalize_observation_image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[2] > 3:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            if array.size > 0 and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _encode_remote_image_payload(self, image: np.ndarray) -> Union[np.ndarray, Dict[str, Any]]:
        if self.remote_image_encoding in {"", "raw", "none"}:
            return self._normalize_observation_image(image)

        if self.remote_image_encoding != "jpeg":
            raise ValueError(
                "Unsupported remote_image_encoding for LingBotVAPolicy: "
                f"{self.remote_image_encoding!r}"
            )

        normalized = self._normalize_observation_image(image)
        buffer = io.BytesIO()
        imageio.imwrite(
            buffer,
            normalized,
            format="JPEG",
            quality=self.remote_image_jpeg_quality,
        )
        return {
            "__lingbot_image_encoding__": "jpeg",
            "data": buffer.getvalue(),
        }

    def _build_server_image_dict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        image_dict: Dict[str, Any] = {}
        for camera_name, model_key in self._resolve_camera_pairs(observation):
            if camera_name not in observation:
                raise KeyError(f"Missing camera observation: {camera_name}")
            image = self._extract_latest_frame(observation[camera_name])
            image_dict[model_key] = self._encode_remote_image_payload(image)
        return image_dict

    def _build_infer_payload(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "obs": [self._build_server_image_dict(observation)],
            "prompt": self.task,
        }

    def _extract_server_action(self, response: Dict[str, Any]) -> np.ndarray:
        data = response.get("data", response)
        if not isinstance(data, dict) or "action" not in data:
            raise KeyError(f"LingBot-VA response missing action, keys={list(response.keys())}")
        return np.asarray(data["action"], dtype=np.float64)

    def _should_skip_first_action_latent(self, first_prediction: bool) -> bool:
        if not first_prediction:
            return False
        mode = str(self.action_condition_mode or "").lower()
        return mode not in ("", "fastwam")

    def _get_skipped_action_latents(
        self,
        server_action: np.ndarray,
        first_prediction: bool,
    ) -> int:
        if server_action.ndim != 3:
            return 0
        return 1 if self._should_skip_first_action_latent(first_prediction) else 0

    def _slice_action_for_history(
        self,
        server_action: np.ndarray,
        skipped_action_latents: int,
        skip_last_obs_latents: int = 0,
    ) -> np.ndarray:
        if server_action.ndim != 3:
            return server_action.copy()
        start_latent = 0
        mode = str(self.action_condition_mode or "").lower()
        if skipped_action_latents > 0 and mode in ("", "fastwam"):
            start_latent = int(skipped_action_latents)
        end_latent = max(start_latent, server_action.shape[1] - int(skip_last_obs_latents))
        return server_action[:, start_latent:end_latent, :].copy()

    def _server_action_to_chunk(self, server_action: np.ndarray) -> np.ndarray:
        steps_per_latent = self.actions_per_latent_frame
        if server_action.ndim == 3:
            channels, frames, steps_per_frame = server_action.shape
            del channels
            steps_per_latent = int(steps_per_frame)
            if steps_per_latent != self.actions_per_latent_frame:
                logger.warning(
                    "LingBotVAPolicy updates actions_per_latent_frame from "
                    f"{self.actions_per_latent_frame} to {steps_per_latent} "
                    "based on server action shape"
                )
                self.actions_per_latent_frame = steps_per_latent
            chunk = np.moveaxis(server_action, 0, -1).reshape(
                frames * steps_per_frame,
                server_action.shape[0],
            )
        elif server_action.ndim == 2:
            chunk = server_action
        else:
            raise ValueError(
                f"Expected server action shape [C,F,H] or [T,D], got {server_action.shape}"
            )

        return np.asarray(chunk, dtype=np.float64)

    def _find_robot_device(self, observation: Dict[str, Any]) -> Optional[str]:
        if self.robot_device is not None and self.robot_device in observation:
            return self.robot_device
        for key in observation.keys():
            lowered = key.lower()
            if lowered.startswith("robot") or lowered.startswith("rizon"):
                return key
        return None

    def _extract_robot_state(self, observation: Dict[str, Any]) -> np.ndarray:
        robot_key = self._find_robot_device(observation)
        if robot_key is None:
            raise KeyError("Robot observation data not found in observation dictionary")
        robot_state = self._extract_latest_frame(observation[robot_key])
        return np.asarray(robot_state, dtype=np.float64).reshape(-1)

    def _get_relative_pose_reference_robot_state(self, observation: Dict[str, Any]) -> np.ndarray:
        """Return the robot state selected by relative_pose_frame semantics."""
        current_robot_state = self._extract_robot_state(observation)
        if self.relative_pose_frame == "local_frame":
            return current_robot_state

        if self._initial_robot_state is None:
            self._initial_robot_state = current_robot_state.copy()
            logger.info(
                "LingBotVAPolicy cached initial robot state for world_frame relative pose"
            )
        return self._initial_robot_state

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
        quat_xyzw = np.asarray(
            process_quaternion(tcp_pose_w_gripper[3:7], "f2l"),
            dtype=np.float64,
        )
        axis_angle = Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, axis_angle], axis=-1)

    def _extract_action_target_poses(
        self,
        absolute_action_chunk: np.ndarray,
        target_action_dim: int,
    ) -> Optional[List[np.ndarray]]:
        """Extract the final action's per-arm absolute pose ([pos3, rotvec3]).

        The absolute action chunk stores each arm as [pos(3), quat_xyzw(4), gripper(1)].
        """
        if absolute_action_chunk.ndim != 2 or absolute_action_chunk.shape[0] == 0:
            return None
        last = np.asarray(absolute_action_chunk[-1], dtype=np.float64)
        action_dim = int(last.shape[0])
        n_arms = 2 if (target_action_dim > 8 and action_dim >= 16) else 1
        poses: List[np.ndarray] = []
        for arm_index in range(n_arms):
            base = arm_index * 8
            if base + 7 > action_dim:
                break
            pos = last[base:base + 3]
            quat_xyzw = last[base + 3:base + 7]
            rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
            poses.append(np.concatenate([pos, rotvec], axis=-1))
        return poses if poses else None

    def _extract_relative_quat_xyzw(
        self,
        action_chunk: np.ndarray,
        quat_slice: slice,
    ) -> np.ndarray:
        quat = np.asarray(action_chunk[:, quat_slice], dtype=np.float64)
        order = str(self.action_quaternion_order).lower()
        if order == "wxyz":
            return np.asarray([process_quaternion(q, "f2l") for q in quat], dtype=np.float64)
        if order == "xyzw":
            return quat
        if order in ("xyz", "zyx", "XYZ", "ZYX"):
            return Rotation.from_euler(order, quat[:, :3]).as_quat()
        raise ValueError(f"Invalid LingBot-VA action_quaternion_order: {order!r}")

    def _relative_pose_to_absolute_mat(
        self,
        relative_pose: np.ndarray,
        reference_mat: np.ndarray,
    ) -> np.ndarray:
        if self.relative_pose_frame == "local_frame":
            return convert_pose_mat_rep(
                pose_to_mat(relative_pose),
                reference_mat,
                pose_rep="relative",
                backward=True,
            )

        relative_mat = pose_to_mat(relative_pose)
        absolute_mat = relative_mat.copy()
        absolute_mat[:, :3, 3] = relative_mat[:, :3, 3] + reference_mat[:3, 3]
        absolute_mat[:, :3, :3] = reference_mat[:3, :3] @ relative_mat[:, :3, :3]
        return absolute_mat

    def _postprocess_grippers(self, action_chunk: np.ndarray) -> np.ndarray:
        if not self.normalize_gripper_width:
            return action_chunk
        if self.gripper_max_width <= 0:
            raise ValueError(f"gripper_max_width must be positive, got {self.gripper_max_width}")
        action_chunk[:, 7] = np.clip(action_chunk[:, 7], 0.0, self.gripper_max_width) / self.gripper_max_width
        if action_chunk.shape[1] > 15:
            action_chunk[:, 15] = np.clip(action_chunk[:, 15], 0.0, self.gripper_max_width) / self.gripper_max_width
        return action_chunk

    def _build_absolute_hold_action(
        self,
        observation: Dict[str, Any],
        target_action_dim: int,
    ) -> np.ndarray:
        reference_robot_state = self._get_relative_pose_reference_robot_state(observation)
        if reference_robot_state.shape[0] < 22:
            raise ValueError(
                "Unsupported robot observation dimension for hold action: "
                f"expected at least 22, got {reference_robot_state.shape[0]}"
            )
        left_tcp = reference_robot_state[14:22]
        left_action = np.concatenate(
            [
                left_tcp[:3],
                np.asarray(process_quaternion(left_tcp[3:7], "f2l"), dtype=np.float64),
                left_tcp[7:8],
            ],
            axis=-1,
        )
        if target_action_dim <= 8:
            return self._postprocess_grippers(left_action[None, :]).astype(np.float64)[0]

        if reference_robot_state.shape[0] >= 44:
            right_tcp = reference_robot_state[36:44]
            right_action = np.concatenate(
                [
                    right_tcp[:3],
                    np.asarray(process_quaternion(right_tcp[3:7], "f2l"), dtype=np.float64),
                    right_tcp[7:8],
                ],
                axis=-1,
            )
        else:
            logger.warning(
                "Dual-arm hold action requested but right arm state is missing; padding zeros."
            )
            right_action = np.zeros_like(left_action)
        hold_action = np.concatenate([left_action, right_action], axis=-1)
        return self._postprocess_grippers(hold_action[None, :]).astype(np.float64)[0]

    def _convert_relative_action_to_absolute(
        self,
        action_chunk: np.ndarray,
        observation: Dict[str, Any],
        target_action_dim: int,
    ) -> np.ndarray:
        reference_robot_state = self._get_relative_pose_reference_robot_state(observation)
        return self._convert_relative_action_to_absolute_with_reference(
            action_chunk=action_chunk,
            reference_robot_state=reference_robot_state,
            target_action_dim=target_action_dim,
        )

    def _convert_relative_action_to_absolute_with_reference(
        self,
        action_chunk: np.ndarray,
        reference_robot_state: np.ndarray,
        target_action_dim: int,
        postprocess_grippers: bool = True,
    ) -> np.ndarray:
        left_reference_pose = self._extract_arm_current_pose(reference_robot_state, arm_index=0)
        left_reference_mat = pose_to_mat(left_reference_pose)

        action_dim = int(action_chunk.shape[-1])
        left_rel_pos = action_chunk[:, :3]
        left_rel_quat_xyzw = self._extract_relative_quat_xyzw(action_chunk, slice(3, 7))
        left_rel_gripper = action_chunk[:, 7:8]
        left_rel_axis_angle = Rotation.from_quat(left_rel_quat_xyzw).as_rotvec()
        left_rel_pose = np.concatenate([left_rel_pos, left_rel_axis_angle], axis=-1)
        left_abs_mat = self._relative_pose_to_absolute_mat(
            left_rel_pose,
            left_reference_mat,
        )
        left_abs_pose = mat_to_pose(left_abs_mat)
        left_abs_quat_xyzw = Rotation.from_rotvec(left_abs_pose[:, 3:6]).as_quat()
        left_abs_action = np.concatenate(
            [left_abs_pose[:, :3], left_abs_quat_xyzw, left_rel_gripper],
            axis=-1,
        )

        if target_action_dim <= 8:
            if postprocess_grippers:
                left_abs_action = self._postprocess_grippers(left_abs_action)
            return left_abs_action.astype(np.float64)

        has_right_action = action_dim >= 16 and reference_robot_state.shape[0] >= 44
        if has_right_action:
            right_reference_pose = self._extract_arm_current_pose(reference_robot_state, arm_index=1)
            right_reference_mat = pose_to_mat(right_reference_pose)
            right_rel_pos = action_chunk[:, 8:11]
            right_rel_quat_xyzw = self._extract_relative_quat_xyzw(action_chunk, slice(11, 15))
            right_rel_gripper = action_chunk[:, 15:16]
            right_rel_axis_angle = Rotation.from_quat(right_rel_quat_xyzw).as_rotvec()
            right_rel_pose = np.concatenate(
                [right_rel_pos, right_rel_axis_angle],
                axis=-1,
            )
            right_abs_mat = self._relative_pose_to_absolute_mat(
                right_rel_pose,
                right_reference_mat,
            )
            right_abs_pose = mat_to_pose(right_abs_mat)
            right_abs_quat_xyzw = Rotation.from_rotvec(right_abs_pose[:, 3:6]).as_quat()
            right_abs_action = np.concatenate(
                [right_abs_pose[:, :3], right_abs_quat_xyzw, right_rel_gripper],
                axis=-1,
            )
        else:
            logger.warning(
                "Dual-arm action requested but right arm action/state is missing; padding zeros."
            )
            right_abs_action = np.zeros_like(left_abs_action)

        dual_action = np.concatenate([left_abs_action, right_abs_action], axis=-1)
        if postprocess_grippers:
            dual_action = self._postprocess_grippers(dual_action)
        return dual_action.astype(np.float64)

    def _convert_absolute_action_to_relative_with_reference(
        self,
        action_chunk: np.ndarray,
        reference_robot_state: np.ndarray,
        target_action_dim: int,
    ) -> np.ndarray:
        """Convert absolute robot actions back to LingBot-VA relative channels."""
        if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
            raise ValueError(f"Expected non-empty action chunk [T,D], got {action_chunk.shape}")

        left_reference_pose = self._extract_arm_current_pose(reference_robot_state, arm_index=0)
        left_reference_mat = pose_to_mat(left_reference_pose)
        left_abs_pose = np.concatenate(
            [
                action_chunk[:, :3],
                Rotation.from_quat(action_chunk[:, 3:7]).as_rotvec(),
            ],
            axis=-1,
        )
        left_rel_mat = self._absolute_pose_to_relative_mat(
            pose_to_mat(left_abs_pose),
            left_reference_mat,
        )
        left_rel_pose = mat_to_pose(left_rel_mat)
        left_rel_quat_xyzw = Rotation.from_rotvec(left_rel_pose[:, 3:6]).as_quat()
        left_rel_quat_xyzw[left_rel_quat_xyzw[:, 3] < 0] *= -1.0
        left_rel_action = np.concatenate(
            [
                left_rel_pose[:, :3],
                self._relative_quat_xyzw_to_action_order(left_rel_quat_xyzw),
                action_chunk[:, 7:8],
            ],
            axis=-1,
        )

        if target_action_dim <= 8:
            return left_rel_action.astype(np.float64)

        if action_chunk.shape[1] < 16 or reference_robot_state.shape[0] < 44:
            right_rel_action = np.zeros_like(left_rel_action)
        else:
            right_reference_pose = self._extract_arm_current_pose(reference_robot_state, arm_index=1)
            right_reference_mat = pose_to_mat(right_reference_pose)
            right_abs_pose = np.concatenate(
                [
                    action_chunk[:, 8:11],
                    Rotation.from_quat(action_chunk[:, 11:15]).as_rotvec(),
                ],
                axis=-1,
            )
            right_rel_mat = self._absolute_pose_to_relative_mat(
                pose_to_mat(right_abs_pose),
                right_reference_mat,
            )
            right_rel_pose = mat_to_pose(right_rel_mat)
            right_rel_quat_xyzw = Rotation.from_rotvec(right_rel_pose[:, 3:6]).as_quat()
            right_rel_quat_xyzw[right_rel_quat_xyzw[:, 3] < 0] *= -1.0
            right_rel_action = np.concatenate(
                [
                    right_rel_pose[:, :3],
                    self._relative_quat_xyzw_to_action_order(right_rel_quat_xyzw),
                    action_chunk[:, 15:16],
                ],
                axis=-1,
            )

        return np.concatenate([left_rel_action, right_rel_action], axis=-1).astype(np.float64)

    def _fit_action_dim(self, action_chunk: np.ndarray, target_dim: int) -> np.ndarray:
        if target_dim <= 0 or action_chunk.shape[1] == target_dim:
            return action_chunk
        if action_chunk.shape[1] > target_dim:
            logger.warning(
                f"LingBotVAPolicy truncates action dim from {action_chunk.shape[1]} to {target_dim}"
            )
            return action_chunk[:, :target_dim]
        padding = np.zeros(
            (action_chunk.shape[0], target_dim - action_chunk.shape[1]),
            dtype=action_chunk.dtype,
        )
        return np.concatenate([action_chunk, padding], axis=1)

    def _fit_chunk_length(self, action_chunk: np.ndarray, chunk_length: int) -> np.ndarray:
        if chunk_length <= 0 or action_chunk.shape[0] == chunk_length:
            return action_chunk
        if action_chunk.shape[0] > chunk_length:
            return action_chunk[:chunk_length]
        tail = np.repeat(action_chunk[-1:, :], chunk_length - action_chunk.shape[0], axis=0)
        return np.concatenate([action_chunk, tail], axis=0)

    def _log_predicted_gripper_action(self, action_chunk: np.ndarray) -> None:
        if not self.print_predicted_gripper_action:
            return
        if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
            logger.info(
                f"LingBotVAPolicy predicted gripper action unavailable: shape={action_chunk.shape}"
            )
            return
        gripper_indices = [7]
        if action_chunk.shape[1] > 15:
            gripper_indices.append(15)
        if max(gripper_indices) >= action_chunk.shape[1]:
            logger.info(
                "LingBotVAPolicy predicted gripper action unavailable: "
                f"action_dim={action_chunk.shape[1]}, expected indices={gripper_indices}"
            )
            return
        gripper_action = action_chunk[:, gripper_indices]
        logger.info(
            "LingBotVAPolicy predicted gripper action "
            f"indices={gripper_indices}, shape={gripper_action.shape}:\n"
            f"{np.array2string(gripper_action, precision=6, suppress_small=False)}"
        )

    def _split_action_chunk_by_config(
        self,
        action_chunk: np.ndarray,
        action_configs: List[Dict[str, Any]],
        chunk_length: int,
    ) -> Dict[str, np.ndarray]:
        """Split one model action vector into per-device chunks."""
        if not action_configs:
            raise ValueError("action_configs is empty")

        total_target_dim = sum(int(cfg.get("action_dim", 0)) for cfg in action_configs)
        if total_target_dim <= 0:
            raise ValueError(f"Invalid action_configs action dims: {action_configs}")

        fitted_chunk = self._fit_chunk_length(action_chunk, chunk_length=chunk_length)
        fitted_chunk = self._fit_action_dim(fitted_chunk, target_dim=total_target_dim)
        self._log_predicted_gripper_action(fitted_chunk)

        action_chunk_dict: Dict[str, np.ndarray] = {}
        start_dim = 0
        for cfg in action_configs:
            device_name = str(cfg["device_name"])
            action_dim = int(cfg["action_dim"])
            end_dim = start_dim + action_dim
            action_chunk_dict[device_name] = fitted_chunk[:, start_dim:end_dim].astype(
                np.float64
            )
            start_dim = end_dim
        return action_chunk_dict

    def _build_history_steps(
        self,
        skipped_action_latents: int = 0,
        skip_last_obs_latents: int = 0,
    ) -> List[int]:
        steps_per_latent = max(int(getattr(self, "video_frames_per_latent", 1)), 1)
        num_steps = (self.frame_chunk_size - skip_last_obs_latents) * steps_per_latent
        start_step = skipped_action_latents * steps_per_latent + 1
        if start_step > num_steps:
            return []
        return list(range(start_step, num_steps + 1))

    def _build_pose_history_steps(
        self,
        skipped_action_latents: int = 0,
        skip_last_obs_latents: int = 0,
    ) -> List[int]:
        """Pose keyframe steps live on the (denser) action timeline.

        Images are sampled at the video frame rate (``video_frames_per_latent`` per
        latent) while robot poses are sampled at the action rate
        (``actions_per_latent_frame`` per latent), so the two timelines differ in both
        count and interval.
        """
        steps_per_latent = max(int(getattr(self, "actions_per_latent_frame", 1)), 1)
        num_steps = (self.frame_chunk_size - skip_last_obs_latents) * steps_per_latent
        start_step = skipped_action_latents * steps_per_latent + 1
        if start_step > num_steps:
            return []
        return list(range(start_step, num_steps + 1))

    def _compute_rtc_inference_delay_steps(self) -> int:
        if self.execution_fps is None or self.execution_fps <= 0:
            return self.rtc_default_delay_steps
        if len(self._latency_tracker) == 0:
            return self.rtc_default_delay_steps
        if self.rtc_latency_stat in {"mean", "avg", "average"}:
            latency_s = self._latency_tracker.mean()
        elif self.rtc_latency_stat in {"p95", "percentile", "quantile"}:
            latency_s = self._latency_tracker.percentile(self.rtc_latency_percentile)
        elif self.rtc_latency_stat == "max":
            latency_s = self._latency_tracker.max()
        else:
            logger.warning(
                f"Unknown rtc_latency_stat={self.rtc_latency_stat!r}; using mean latency."
            )
            latency_s = self._latency_tracker.mean()
        return max(1, int(math.ceil(float(latency_s) * float(self.execution_fps))))

    def _parse_optional_nonnegative_int(
        self,
        value: Optional[int],
        name: str,
    ) -> Optional[int]:
        if value is None:
            return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{name} must be >= 0 or null, got {value}")
        return parsed

    def _rtc_latency_summary(self) -> Dict[str, float | int | None]:
        if len(self._latency_tracker) == 0:
            return {
                "samples": 0,
                "mean_latency_ms": None,
                "latest_latency_ms": None,
                "mean_occupied_steps": None,
                "mean_occupied_steps_ceil": None,
            }
        mean_latency_s = float(self._latency_tracker.mean())
        latest_latency_s = float(self._latency_tracker.latest() or 0.0)
        mean_steps = (
            mean_latency_s * float(self.execution_fps)
            if self.execution_fps is not None and self.execution_fps > 0
            else None
        )
        return {
            "samples": int(len(self._latency_tracker)),
            "mean_latency_ms": mean_latency_s * 1000.0,
            "latest_latency_ms": latest_latency_s * 1000.0,
            "mean_occupied_steps": mean_steps,
            "mean_occupied_steps_ceil": (
                int(math.ceil(mean_steps)) if mean_steps is not None else None
            ),
        }

    def _build_rtc_prefix_weights(
        self,
        *,
        total_steps: int,
        inference_delay: int,
        prefix_attention_horizon: int,
        schedule: str,
    ) -> np.ndarray:
        total_steps = int(total_steps)
        if total_steps <= 0:
            return np.zeros((0,), dtype=np.float64)

        start = max(int(inference_delay), 0)
        end = max(start + int(prefix_attention_horizon), 0)
        start = min(start, total_steps)
        end = min(end, total_steps)
        start = min(start, end)

        step = np.arange(total_steps, dtype=np.float64)
        schedule = str(schedule).lower()
        if schedule == "ones":
            weights = np.ones(total_steps, dtype=np.float64)
        elif schedule == "zeros":
            weights = (step < start).astype(np.float64)
        elif schedule in {"linear", "exp"}:
            denom = max(end - start + 1, 1)
            weights = np.clip((float(start - 1) - step) / float(denom) + 1.0, 0.0, 1.0)
            if schedule == "exp":
                weights = weights * np.expm1(weights) / (math.e - 1.0)
        else:
            raise ValueError(f"Invalid RTC prefix schedule: {schedule!r}")

        weights = np.where(step >= end, np.zeros_like(weights), weights)
        return weights.astype(np.float64)

    def _format_rtc_weight_distribution(self, weights: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(value):.4f}" for value in weights.tolist()) + "]"

    def _shift_rtc_prefix_chunk(
        self,
        action_chunk: np.ndarray,
        time_base: int,
    ) -> Optional[np.ndarray]:
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected RTC prefix chunk [T,D], got {action_chunk.shape}")
        if action_chunk.shape[0] == 0:
            return None
        time_base = max(int(time_base), 0)
        if time_base >= action_chunk.shape[0]:
            return None
        remainder = action_chunk[time_base + 1:]
        pad_len = action_chunk.shape[0] - remainder.shape[0]
        if pad_len > 0:
            pad = np.zeros((pad_len, action_chunk.shape[1]), dtype=action_chunk.dtype)
            return np.concatenate([remainder, pad], axis=0)
        return remainder

    def _chunk_to_server_action(
        self,
        action_chunk: np.ndarray,
        template_server_action: np.ndarray,
    ) -> np.ndarray:
        if template_server_action.ndim != 3:
            raise ValueError(
                "RTC template server action must have shape [C,F,H], "
                f"got {template_server_action.shape}"
            )
        channels, frames, steps_per_frame = template_server_action.shape
        action_chunk = self._fit_chunk_length(action_chunk, int(frames) * int(steps_per_frame))
        action_chunk = self._fit_action_dim(action_chunk, int(channels))
        return np.moveaxis(
            action_chunk.reshape(int(frames), int(steps_per_frame), int(channels)),
            -1,
            0,
        ).astype(np.float64)

    def _build_rtc_prefix_action(
        self,
        observation: Dict[str, Any],
        action_index: Optional[int],
        target_action_dim: int,
    ) -> Optional[np.ndarray]:
        if action_index is None or self._last_server_action is None:
            return None
        if self._last_reference_robot_state is None:
            return None
        previous_server_action = np.asarray(self._last_server_action, dtype=np.float64)
        if previous_server_action.ndim != 3:
            return None

        previous_relative_chunk = self._server_action_to_chunk(previous_server_action)
        self._rtc_last_overlap_steps = max(
            previous_relative_chunk.shape[0] - int(action_index) - 1,
            0,
        )
        previous_absolute_chunk = self._convert_relative_action_to_absolute_with_reference(
            action_chunk=previous_relative_chunk,
            reference_robot_state=self._last_reference_robot_state,
            target_action_dim=target_action_dim,
            postprocess_grippers=False,
        )
        current_reference_robot_state = self._get_relative_pose_reference_robot_state(
            observation
        )
        current_relative_chunk = self._convert_absolute_action_to_relative_with_reference(
            action_chunk=previous_absolute_chunk,
            reference_robot_state=current_reference_robot_state,
            target_action_dim=target_action_dim,
        )
        shifted_chunk = self._shift_rtc_prefix_chunk(
            current_relative_chunk,
            time_base=int(action_index),
        )
        if shifted_chunk is None:
            return None
        return self._chunk_to_server_action(
            shifted_chunk,
            template_server_action=previous_server_action,
        )

    def _build_rtc_config(
        self,
        *,
        action_index: Optional[int],
        chunk_length: int,
        has_prefix: bool,
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_rtc or action_index is None or not has_prefix:
            return None
        estimated_inference_delay = self._compute_rtc_inference_delay_steps()
        overlap_steps = int(self._rtc_last_overlap_steps or 0)
        if (
            self.rtc_guidance_full_steps_override is None
            and self.rtc_guidance_decay_steps_override is None
            and overlap_steps <= int(estimated_inference_delay)
        ):
            logger.warning(
                "LingBotVAPolicy RTC skipped because inference delay reaches chunk end: "
                f"action_index={action_index}, inference_delay={estimated_inference_delay}, "
                f"chunk_length={chunk_length}, overlap_steps={overlap_steps}"
            )
            return None

        guidance_full_steps = (
            int(self.rtc_guidance_full_steps_override)
            if self.rtc_guidance_full_steps_override is not None
            else int(estimated_inference_delay)
        )
        guidance_full_steps = min(max(guidance_full_steps, 0), overlap_steps)

        remaining_overlap_steps = max(overlap_steps - guidance_full_steps, 0)
        prefix_attention_horizon = (
            int(self.rtc_guidance_decay_steps_override)
            if self.rtc_guidance_decay_steps_override is not None
            else remaining_overlap_steps
        )
        prefix_attention_horizon = min(
            max(prefix_attention_horizon, 0),
            remaining_overlap_steps,
        )
        if guidance_full_steps + prefix_attention_horizon <= 0:
            return None
        latency_summary = self._rtc_latency_summary()
        weights = self._build_rtc_prefix_weights(
            total_steps=int(chunk_length),
            inference_delay=int(guidance_full_steps),
            prefix_attention_horizon=int(prefix_attention_horizon),
            schedule=self.rtc_prefix_attention_schedule,
        )
        return {
            "inference_delay": int(guidance_full_steps),
            "estimated_inference_delay": int(estimated_inference_delay),
            "prefix_attention_horizon": int(prefix_attention_horizon),
            "max_guidance_weight": float(self.rtc_max_guidance_weight),
            "prefix_attention_schedule": self.rtc_prefix_attention_schedule,
            "latency_samples": int(len(self._latency_tracker)),
            "latency_mean_ms": float(self._latency_tracker.mean() * 1000.0),
            "latency_latest_ms": latency_summary["latest_latency_ms"],
            "mean_occupied_steps": latency_summary["mean_occupied_steps"],
            "mean_occupied_steps_ceil": latency_summary["mean_occupied_steps_ceil"],
            "overlap_steps": overlap_steps,
            "overlap_steps_after_delay": prefix_attention_horizon,
            "guidance_full_steps_override": self.rtc_guidance_full_steps_override,
            "guidance_decay_steps_override": self.rtc_guidance_decay_steps_override,
            "weight_nonzero_steps": int(np.count_nonzero(weights > 0.0)),
            "weight_sum": float(np.sum(weights)),
            "weight_min": float(np.min(weights)) if weights.size > 0 else 0.0,
            "weight_max": float(np.max(weights)) if weights.size > 0 else 0.0,
            "weight_distribution": weights.tolist(),
        }

    def predict(
        self,
        observation: Dict[str, Any],
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        action_index: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Request one action chunk from the server."""
        if len(action_configs) == 0:
            raise ValueError("action_configs is empty")

        if self.stateless_reset_each_predict:
            self.reset()

        first_prediction = self._is_first_prediction
        total_target_dim = sum(int(cfg.get("action_dim", 0)) for cfg in action_configs)
        target_dim = total_target_dim if total_target_dim > 0 else 0
        payload = self._build_infer_payload(observation)
        rtc_prefix_action = None
        rtc_config = None
        if self.enable_rtc:
            rtc_prefix_action = self._build_rtc_prefix_action(
                observation=observation,
                action_index=action_index,
                target_action_dim=target_dim if target_dim > 0 else 16,
            )
            rtc_config = self._build_rtc_config(
                action_index=action_index,
                chunk_length=chunk_length,
                has_prefix=rtc_prefix_action is not None,
            )
        if rtc_config is not None and rtc_prefix_action is not None:
            payload["rtc_config"] = rtc_config
            payload["rtc_prefix_action"] = rtc_prefix_action
            logger.info(
                "LingBotVAPolicy sends RTC infer request: "
                f"action_index={action_index}, "
                f"inference_delay={rtc_config['inference_delay']}, "
                f"estimated_inference_delay={rtc_config['estimated_inference_delay']}, "
                f"prefix_attention_horizon={rtc_config['prefix_attention_horizon']}, "
                f"overlap_steps={rtc_config['overlap_steps']}, "
                f"overlap_steps_after_delay={rtc_config['overlap_steps_after_delay']}, "
                f"guidance_full_steps_override={rtc_config['guidance_full_steps_override']}, "
                f"guidance_decay_steps_override={rtc_config['guidance_decay_steps_override']}, "
                f"mean_occupied_steps={rtc_config['mean_occupied_steps']}, "
                f"weight_nonzero_steps={rtc_config['weight_nonzero_steps']}, "
                f"weight_sum={rtc_config['weight_sum']:.4f}, "
                f"weight_min={rtc_config['weight_min']:.4f}, "
                f"weight_max={rtc_config['weight_max']:.4f}, "
                "weight_distribution="
                f"{self._format_rtc_weight_distribution(np.asarray(rtc_config['weight_distribution']))}"
            )

        request_start = time.monotonic()
        response = self._send_request(payload)
        request_latency_s = time.monotonic() - request_start
        if self.enable_rtc:
            self._latency_tracker.add(request_latency_s)
            latency_summary = self._rtc_latency_summary()
            logger.info(
                "LingBotVAPolicy RTC inference latency stats: "
                f"latest={request_latency_s * 1000.0:.2f}ms, "
                f"mean={float(latency_summary['mean_latency_ms']):.2f}ms, "
                f"samples={latency_summary['samples']}, "
                f"mean_occupied_steps={latency_summary['mean_occupied_steps']}, "
                f"mean_occupied_steps_ceil={latency_summary['mean_occupied_steps_ceil']}"
            )
        server_action = self._extract_server_action(response)
        self._last_server_action = server_action.copy()
        skipped_action_latents = self._get_skipped_action_latents(
            server_action,
            first_prediction=first_prediction,
        )
        self._last_history_action = self._slice_action_for_history(
            server_action,
            skipped_action_latents=skipped_action_latents,
            skip_last_obs_latents=self.skip_last_obs_latents,
        )

        action_chunk = self._server_action_to_chunk(server_action)
        target_dim = total_target_dim if total_target_dim > 0 else action_chunk.shape[1]
        action_chunk = self._convert_relative_action_to_absolute(
            action_chunk=action_chunk,
            observation=observation,
            target_action_dim=target_dim,
        )
        # Cache the reference robot state used for relative<->absolute conversion so the
        # history KV-cache state can be rebuilt from actual robot poses later.
        self._last_reference_robot_state = self._get_relative_pose_reference_robot_state(
            observation
        ).copy()

        if skipped_action_latents > 0:
            hold_steps = min(
                skipped_action_latents * self.actions_per_latent_frame,
                action_chunk.shape[0],
            )
            if hold_steps > 0:
                action_chunk[:hold_steps] = self._build_absolute_hold_action(
                    observation=observation,
                    target_action_dim=target_dim,
                )

        # Cache the final action's absolute per-arm pose so PolicyConnector can detect
        # when the arm has physically reached the end of this chunk.
        self._last_action_target_poses = self._extract_action_target_poses(
            action_chunk, target_dim
        )

        self._last_history_steps = self._build_history_steps(
            skipped_action_latents=skipped_action_latents,
            skip_last_obs_latents=self.skip_last_obs_latents,
        )
        self._last_pose_steps = self._build_pose_history_steps(
            skipped_action_latents=skipped_action_latents,
            skip_last_obs_latents=self.skip_last_obs_latents,
        )
        self._is_first_prediction = False
        return self._split_action_chunk_by_config(
            action_chunk=action_chunk,
            action_configs=action_configs,
            chunk_length=chunk_length,
        )
