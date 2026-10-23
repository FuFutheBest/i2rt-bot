"""
Configurable two-stage action chunk smoothing and upsampling.

Author: Cursor Agent
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.logger_config import logger

from .action_manager import BaseChunkManager, DropFirstActions


class SmoothChunkManager(BaseChunkManager):
    """Optionally smooth action chunks, then optionally upsample them for execution."""

    def __init__(
        self,
        smoothing: Mapping[str, Any] | None = None,
        upsampling: Mapping[str, Any] | None = None,
        quaternion_format: str = "xyzw",
        drop_first_actions: DropFirstActions = 0,
        skip_first_chunk: bool = False,
        execution_fps: float = 30.0,
        interpolate_auto_drop: bool = True,
        auto_drop_gripper_mode: str = "linear",
    ) -> None:
        super().__init__(
            drop_first_actions=drop_first_actions,
            skip_first_chunk=skip_first_chunk,
            execution_fps=execution_fps,
            interpolate_auto_drop=interpolate_auto_drop,
            quaternion_format=quaternion_format,
            auto_drop_gripper_mode=auto_drop_gripper_mode,
        )
        self.smoothing_config = dict(smoothing or {})
        self.upsampling_config = dict(upsampling or {})
        self.smoothing_method = self._normalize_optional_method(
            self.smoothing_config.get("method"),
            valid_methods={"savgol", "moving_average"},
            config_name="smoothing.method",
        )
        self.upsampling_method = self._normalize_optional_method(
            self.upsampling_config.get("method"),
            valid_methods={"linear", "cubic"},
            config_name="upsampling.method",
        )
        if self.smoothing_method is None and self.upsampling_method is None:
            raise ValueError(
                "At least one of smoothing.method or upsampling.method must be configured"
            )

        quat_format = str(quaternion_format).lower()
        if quat_format not in {"xyzw", "wxyz"}:
            raise ValueError(
                f"quaternion_format must be 'xyzw' or 'wxyz', got {quaternion_format!r}"
            )
        self.quaternion_format = quat_format

        self.smoothing_ratio = self._positive_int(
            self.smoothing_config.get(
                "upsample_ratio", 2 if self.smoothing_method == "savgol" else 3
            ),
            "smoothing.upsample_ratio",
        )
        self.upsampling_ratio = self._positive_int(
            self.upsampling_config.get("ratio", 1),
            "upsampling.ratio",
        )

        self.filter_window = self._normalize_filter_window(
            self.smoothing_config.get(
                "window", 13 if self.smoothing_method == "savgol" else 5
            )
        )
        self.savgol_polyorder = max(0, int(self.smoothing_config.get("polyorder", 3)))
        self.savgol_mode = str(self.smoothing_config.get("mode", "interp"))
        self.boundary_context_steps = max(
            0, int(self.smoothing_config.get("boundary_context_steps", 6))
        )
        blend_steps = self.smoothing_config.get("boundary_blend_steps")
        if blend_steps is None:
            blend_steps = self.boundary_context_steps
        self.boundary_blend_steps = max(0, int(blend_steps))
        self.smooth_gripper = bool(self.smoothing_config.get("smooth_gripper", True))
        if self.savgol_mode not in {"mirror", "constant", "nearest", "wrap", "interp"}:
            raise ValueError(
                "smoothing.mode must be one of mirror, constant, nearest, wrap, or interp"
            )

        command_fps = self.smoothing_config.get("command_fps", execution_fps)
        self.command_fps = self._positive_float(command_fps, "smoothing.command_fps")
        source_fps = self.smoothing_config.get("source_fps", self.command_fps)
        self.source_fps = self._positive_float(source_fps, "smoothing.source_fps")

    def put(
        self,
        action_chunk_dict: dict[str, np.ndarray],
        observation_timestamp: int | None = None,
    ) -> None:
        if not action_chunk_dict:
            BaseChunkManager.put(self, action_chunk_dict, observation_timestamp)
            return

        context_action_dict = self._get_dispatched_context()
        processed_chunk_dict: dict[str, np.ndarray] = {}
        common_original_offsets: np.ndarray | None = None
        for device_name, source_chunk in action_chunk_dict.items():
            chunk = np.asarray(source_chunk)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            if chunk.size == 0:
                processed_chunk_dict[device_name] = chunk.copy()
                processed_original_offsets = np.empty((0,), dtype=np.float64)
                if common_original_offsets is None:
                    common_original_offsets = processed_original_offsets
                continue

            context = context_action_dict.get(device_name)
            processed, processed_original_offsets = self._apply_smoothing(
                chunk,
                context,
            )
            processed, processed_original_offsets = self._apply_execution_upsampling(
                processed,
                processed_original_offsets,
                context,
            )
            processed_chunk_dict[device_name] = processed
            if common_original_offsets is None:
                common_original_offsets = processed_original_offsets
            elif (
                common_original_offsets.shape != processed_original_offsets.shape
                or not np.allclose(
                    common_original_offsets,
                    processed_original_offsets,
                    rtol=0.0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    "SmoothChunkManager requires all device chunks to share "
                    "the same processed-to-original timeline"
                )

        if common_original_offsets is None:
            common_original_offsets = np.empty((0,), dtype=np.float64)
        BaseChunkManager.put(
            self,
            processed_chunk_dict,
            observation_timestamp,
            original_action_offsets=common_original_offsets,
            alignment_origin_steps=self._timeline_position_for_original_zero(
                common_original_offsets
            ),
            original_step_interval_ns=self._infer_original_step_interval_ns(
                common_original_offsets
            ),
        )

    def _apply_smoothing(
        self,
        chunk: np.ndarray,
        context_actions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        original_offsets = np.arange(chunk.shape[0], dtype=np.float64)
        if self.smoothing_method is None:
            return chunk.copy(), original_offsets
        if self.smoothing_method == "savgol":
            if self.boundary_context_steps <= 0 or context_actions is None:
                savgol_context = None
            else:
                savgol_context = context_actions[-self.boundary_context_steps :]
            return self._smooth_savgol(chunk, savgol_context), original_offsets
        return self._smooth_moving_average(chunk, context_actions)

    def _apply_execution_upsampling(
        self,
        chunk: np.ndarray,
        original_offsets: np.ndarray,
        context_actions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.upsampling_method is None or self.upsampling_ratio <= 1:
            return chunk.copy(), original_offsets.copy()

        quat_slices = self._get_quaternion_slices(chunk.shape[1])
        last_action = self._validate_last_action(context_actions, chunk.shape[1])
        if last_action is None:
            interpolated = self._interpolate_sequence(
                chunk,
                quat_slices,
                ratio=self.upsampling_ratio,
                method=self.upsampling_method,
            )
            interpolated_offsets = self._interpolate_original_offsets(
                original_offsets,
                ratio=self.upsampling_ratio,
            )
        else:
            merged = np.vstack([last_action[None, :], chunk])
            interpolated = self._interpolate_sequence(
                merged,
                quat_slices,
                ratio=self.upsampling_ratio,
                method=self.upsampling_method,
            )[1:]
            source_step = self._infer_offset_step(original_offsets)
            merged_offsets = np.concatenate(
                [
                    np.asarray(
                        [float(original_offsets[0]) - source_step],
                        dtype=np.float64,
                    ),
                    original_offsets,
                ]
            )
            interpolated_offsets = self._interpolate_original_offsets(
                merged_offsets,
                ratio=self.upsampling_ratio,
            )[1:]

        tail_trim = self.upsampling_ratio - 1
        if tail_trim > 0 and interpolated.shape[0] > tail_trim:
            interpolated = interpolated[:-tail_trim]
            interpolated_offsets = interpolated_offsets[:-tail_trim]
        if interpolated.shape[0] != interpolated_offsets.shape[0]:
            raise RuntimeError(
                "SmoothChunkManager action and timeline upsampling lengths differ"
            )
        return (
            interpolated.astype(chunk.dtype, copy=False),
            interpolated_offsets,
        )

    @staticmethod
    def _interpolate_original_offsets(
        original_offsets: np.ndarray,
        ratio: int,
    ) -> np.ndarray:
        """Upsample source-index coordinates with the action interpolation grid."""
        if ratio <= 1 or original_offsets.size == 0:
            return original_offsets.copy()
        source_positions = (
            np.arange(original_offsets.shape[0], dtype=np.float64) * float(ratio)
        )
        output_positions = np.arange(
            ratio * original_offsets.shape[0],
            dtype=np.float64,
        )
        return np.interp(
            output_positions,
            source_positions,
            original_offsets,
        )

    @staticmethod
    def _infer_offset_step(original_offsets: np.ndarray) -> float:
        """Infer one pre-upsampling source step from a monotonic timeline."""
        if original_offsets.size <= 1:
            return 1.0
        positive_differences = np.diff(original_offsets)
        positive_differences = positive_differences[positive_differences > 1e-9]
        if positive_differences.size == 0:
            return 1.0
        return float(np.median(positive_differences))

    @staticmethod
    def _timeline_position_for_original_zero(
        original_offsets: np.ndarray,
    ) -> float:
        """Locate raw policy action zero on the processed execution timeline."""
        if original_offsets.size == 0 or original_offsets[0] >= -1e-9:
            return 0.0

        right_index = int(np.searchsorted(original_offsets, 0.0, side="left"))
        if right_index >= original_offsets.shape[0]:
            return 0.0
        right_offset = float(original_offsets[right_index])
        if abs(right_offset) <= 1e-9:
            return float(right_index)

        left_index = max(0, right_index - 1)
        left_offset = float(original_offsets[left_index])
        offset_span = right_offset - left_offset
        if offset_span <= 1e-12:
            return float(right_index)
        return float(left_index) + (0.0 - left_offset) / offset_span

    def _infer_original_step_interval_ns(
        self,
        original_offsets: np.ndarray,
    ) -> int:
        """Infer raw policy spacing from the processed execution timeline."""
        processed_step_interval_ns = self._default_step_interval_ns()
        if original_offsets.size <= 1:
            return processed_step_interval_ns

        positive_differences = np.diff(original_offsets)
        positive_differences = positive_differences[positive_differences > 1e-9]
        if positive_differences.size == 0:
            return processed_step_interval_ns
        processed_to_original_scale = float(np.median(positive_differences))
        return int(round(processed_step_interval_ns / processed_to_original_scale))

    def _smooth_savgol(
        self,
        chunk: np.ndarray,
        context_actions: np.ndarray | None,
    ) -> np.ndarray:
        context = self._validate_context(context_actions, chunk.shape[1])
        if context is None:
            combined = chunk
            context_len = 0
        else:
            combined = np.vstack([context, chunk])
            context_len = context.shape[0]

        quat_slices = self._get_quaternion_slices(combined.shape[1])
        dense = self._upsample_for_filter(combined, quat_slices)
        filtered_dense = self._savgol_filter_sequence(
            dense,
            quat_slices,
            self._get_gripper_indices(combined.shape[1]),
        )
        sample_indices = (
            np.arange(combined.shape[0], dtype=np.int64) * self.smoothing_ratio
        )
        sample_indices = np.clip(sample_indices, 0, filtered_dense.shape[0] - 1)
        filtered_combined = filtered_dense[sample_indices]
        filtered_chunk = filtered_combined[context_len : context_len + chunk.shape[0]]
        if filtered_chunk.shape[0] != chunk.shape[0]:
            logger.warning(
                "SmoothChunkManager Savitzky-Golay output length %s != input length %s; using input",
                filtered_chunk.shape[0],
                chunk.shape[0],
            )
            return chunk.copy()
        return self._blend_boundary_start(filtered_chunk, context, quat_slices).astype(
            chunk.dtype, copy=False
        )

    def _smooth_moving_average(
        self,
        chunk: np.ndarray,
        context_actions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        quat_slices = self._get_quaternion_slices(chunk.shape[1])
        last_action = self._validate_last_action(context_actions, chunk.shape[1])
        has_boundary = last_action is not None
        actions = np.vstack([last_action[None, :], chunk]) if has_boundary else chunk
        dense = self._upsample_for_filter(actions, quat_slices)
        filtered = self._moving_average_filter(
            filtered_input=dense, quat_slices=quat_slices
        )
        times_s, downsampled = self._downsample_by_fps(filtered, quat_slices)
        original_offsets = times_s * self.source_fps
        if has_boundary:
            boundary_mask = times_s >= (1.0 / self.source_fps) - 1e-9
            downsampled = downsampled[boundary_mask]
            original_offsets = original_offsets[boundary_mask] - 1.0
        if downsampled.shape[0] == 0:
            logger.warning(
                "SmoothChunkManager moving-average output is empty; using input"
            )
            return (
                chunk.copy(),
                np.arange(chunk.shape[0], dtype=np.float64),
            )
        return (
            downsampled.astype(chunk.dtype, copy=False),
            original_offsets.astype(np.float64, copy=False),
        )

    def _get_dispatched_context(self) -> dict[str, np.ndarray]:
        if self.current_action_chunk_dict is None or self.current_step <= 0:
            return {}
        context_steps = (
            self.boundary_context_steps if self.smoothing_method == "savgol" else 1
        )
        context_steps = max(
            context_steps, 1 if self.upsampling_method is not None else 0
        )
        if context_steps <= 0:
            return {}

        result: dict[str, np.ndarray] = {}
        for device_name, source_chunk in self.current_action_chunk_dict.items():
            chunk = (
                source_chunk.reshape(1, -1) if source_chunk.ndim == 1 else source_chunk
            )
            end_index = min(self.current_step, chunk.shape[0])
            start_index = max(0, end_index - context_steps)
            if start_index < end_index:
                result[device_name] = chunk[start_index:end_index].copy()
        return result

    def _upsample_for_filter(
        self, actions: np.ndarray, quat_slices: list[slice]
    ) -> np.ndarray:
        if self.smoothing_ratio <= 1:
            return actions.copy()
        upsampled = self._interpolate_sequence(
            actions,
            quat_slices,
            ratio=self.smoothing_ratio,
            method="linear",
        )
        tail_trim = self.smoothing_ratio - 1
        if upsampled.shape[0] > tail_trim:
            upsampled = upsampled[:-tail_trim]
        return upsampled

    def _interpolate_sequence(
        self,
        actions: np.ndarray,
        quat_slices: list[slice],
        ratio: int,
        method: str,
    ) -> np.ndarray:
        if ratio <= 1 or actions.shape[0] == 0:
            return actions.copy()

        num_steps, action_dim = actions.shape
        interpolated = np.empty((ratio * num_steps, action_dim), dtype=actions.dtype)
        interpolated[::ratio] = actions
        if num_steps == 1:
            interpolated[1:] = actions[-1]
            return interpolated

        non_quat_mask = self._get_non_quaternion_mask(action_dim, quat_slices)
        if method == "cubic" and np.any(non_quat_mask):
            source_times = np.arange(num_steps, dtype=np.float64)
            dense_times = np.arange(
                (num_steps - 1) * ratio + 1, dtype=np.float64
            ) / float(ratio)
            spline = CubicSpline(
                source_times,
                actions[:, non_quat_mask].astype(np.float64, copy=False),
                axis=0,
                bc_type="natural",
            )
            interpolated[: dense_times.size, non_quat_mask] = spline(
                dense_times
            ).astype(
                actions.dtype,
                copy=False,
            )

        ratios = np.arange(1, ratio, dtype=np.float64) / float(ratio)
        for idx in range(num_steps - 1):
            start = actions[idx].astype(np.float64, copy=False)
            end = actions[idx + 1].astype(np.float64, copy=False)
            start_index = idx * ratio + 1
            end_index = start_index + ratio - 1
            segment = start[None, :] + (end - start)[None, :] * ratios[:, None]
            if method == "cubic":
                segment[:, non_quat_mask] = interpolated[
                    start_index:end_index, non_quat_mask
                ]
            for quat_slice in quat_slices:
                slerped = self._slerp_quaternion_segment(
                    start[quat_slice], end[quat_slice], ratios
                )
                if slerped is not None:
                    segment[:, quat_slice] = slerped
                else:
                    segment[:, quat_slice] = self._normalize_quaternions(
                        segment[:, quat_slice]
                    )
            interpolated[start_index:end_index] = segment.astype(
                actions.dtype, copy=False
            )

        interpolated[(num_steps - 1) * ratio + 1 :] = actions[-1]
        return interpolated

    def _savgol_filter_sequence(
        self,
        trajectory: np.ndarray,
        quat_slices: list[slice],
        gripper_indices: list[int],
    ) -> np.ndarray:
        params = self._resolve_savgol_params(trajectory.shape[0])
        if params is None:
            return trajectory.copy()
        window, polyorder = params
        source = trajectory.astype(np.float64, copy=False)
        filtered = source.copy()
        non_quat_mask = self._get_non_quaternion_mask(trajectory.shape[1], quat_slices)
        if gripper_indices:
            non_quat_mask[gripper_indices] = False
        if np.any(non_quat_mask):
            filtered[:, non_quat_mask] = savgol_filter(
                source[:, non_quat_mask],
                window_length=window,
                polyorder=polyorder,
                axis=0,
                mode=self.savgol_mode,
            )
        for quat_slice in quat_slices:
            aligned = self._align_quaternion_signs(source[:, quat_slice])
            quat_filtered = savgol_filter(
                aligned,
                window_length=window,
                polyorder=polyorder,
                axis=0,
                mode=self.savgol_mode,
            )
            normalized = self._normalize_quaternions(quat_filtered)
            invalid_mask = np.linalg.norm(quat_filtered, axis=1) < 1e-8
            normalized[invalid_mask] = aligned[invalid_mask]
            filtered[:, quat_slice] = normalized
        if self.smooth_gripper and gripper_indices:
            filtered[:, gripper_indices] = savgol_filter(
                source[:, gripper_indices],
                window_length=window,
                polyorder=polyorder,
                axis=0,
                mode=self.savgol_mode,
            )
        return filtered.astype(trajectory.dtype, copy=False)

    def _moving_average_filter(
        self,
        filtered_input: np.ndarray,
        quat_slices: list[slice],
    ) -> np.ndarray:
        if self.filter_window < 3 or filtered_input.shape[0] == 0:
            return filtered_input.copy()
        half_window = self.filter_window // 2
        filtered = np.empty_like(filtered_input)
        for idx in range(filtered_input.shape[0]):
            left = max(0, idx - half_window)
            right = min(filtered_input.shape[0], idx + half_window + 1)
            window = filtered_input[left:right].astype(np.float64, copy=False)
            averaged = np.mean(window, axis=0)
            for quat_slice in quat_slices:
                averaged[quat_slice] = self._average_quaternion_window(
                    window[:, quat_slice]
                )
            filtered[idx] = averaged.astype(filtered_input.dtype, copy=False)
        return filtered

    def _downsample_by_fps(
        self,
        trajectory: np.ndarray,
        quat_slices: list[slice],
    ) -> tuple[np.ndarray, np.ndarray]:
        if trajectory.shape[0] == 0:
            return np.empty((0,), dtype=np.float64), trajectory.copy()
        dense_dt = 1.0 / self.source_fps / float(self.smoothing_ratio)
        duration_s = dense_dt * float(trajectory.shape[0] - 1)
        if duration_s <= 0.0:
            return np.array([0.0], dtype=np.float64), trajectory[0:1].copy()

        output_count = int(np.floor(duration_s * self.command_fps + 1e-9)) + 1
        output_times = np.arange(output_count, dtype=np.float64) / self.command_fps
        source_times = np.arange(trajectory.shape[0], dtype=np.float64) * dense_dt
        downsampled = np.empty(
            (output_count, trajectory.shape[1]), dtype=trajectory.dtype
        )
        for output_idx, output_time in enumerate(output_times):
            right = min(
                int(np.searchsorted(source_times, output_time, side="right")),
                trajectory.shape[0] - 1,
            )
            left = max(0, right - 1)
            if left == right:
                downsampled[output_idx] = trajectory[left]
                continue
            start_time = source_times[left]
            end_time = source_times[right]
            alpha = (
                0.0
                if end_time <= start_time
                else (output_time - start_time) / (end_time - start_time)
            )
            start = trajectory[left].astype(np.float64, copy=False)
            end = trajectory[right].astype(np.float64, copy=False)
            sample = start + (end - start) * alpha
            for quat_slice in quat_slices:
                slerped = self._slerp_quaternion_segment(
                    start[quat_slice],
                    end[quat_slice],
                    np.asarray([alpha], dtype=np.float64),
                )
                if slerped is not None:
                    sample[quat_slice] = slerped[0]
            downsampled[output_idx] = sample.astype(trajectory.dtype, copy=False)
        return output_times, downsampled

    def _blend_boundary_start(
        self,
        chunk: np.ndarray,
        context: np.ndarray | None,
        quat_slices: list[slice],
    ) -> np.ndarray:
        if self.boundary_blend_steps <= 0 or context is None or context.shape[0] == 0:
            return chunk
        blend_count = min(self.boundary_blend_steps, chunk.shape[0])
        blended = chunk.astype(np.float64, copy=True)
        context64 = context.astype(np.float64, copy=False)
        anchor = context64[-1]
        velocity = (
            context64[-1] - context64[-2]
            if context64.shape[0] >= 2
            else np.zeros_like(anchor)
        )
        reference = np.asarray(
            [anchor + velocity * float(idx + 1) for idx in range(blend_count)],
            dtype=np.float64,
        )
        for quat_slice in quat_slices:
            reference[:, quat_slice] = anchor[quat_slice]
        weights = np.linspace(1.0 / blend_count, 1.0, blend_count, dtype=np.float64)
        weights = weights * weights * (3.0 - 2.0 * weights)
        blend_mask = np.ones(chunk.shape[1], dtype=bool)
        if not self.smooth_gripper:
            blend_mask[self._get_gripper_indices(chunk.shape[1])] = False
        blended[:blend_count, blend_mask] = (
            (1.0 - weights[:, None]) * reference[:, blend_mask]
            + weights[:, None] * blended[:blend_count, blend_mask]
        )
        for quat_slice in quat_slices:
            for idx, weight in enumerate(weights):
                slerped = self._slerp_quaternion_segment(
                    reference[idx, quat_slice],
                    chunk[idx, quat_slice],
                    np.asarray([weight], dtype=np.float64),
                )
                if slerped is not None:
                    blended[idx, quat_slice] = slerped[0]
        return blended.astype(chunk.dtype, copy=False)

    def _slerp_quaternion_segment(
        self,
        start_quat: np.ndarray,
        end_quat: np.ndarray,
        ratios: np.ndarray,
    ) -> np.ndarray | None:
        start_xyzw = self._normalize_quaternion(self._to_xyzw(start_quat))
        end_xyzw = self._normalize_quaternion(self._to_xyzw(end_quat))
        if start_xyzw is None or end_xyzw is None:
            return None
        try:
            interpolated = Slerp([0.0, 1.0], R.from_quat([start_xyzw, end_xyzw]))(
                ratios
            ).as_quat()
        except ValueError as exc:
            logger.warning(
                "SmoothChunkManager quaternion slerp failed; using normalized lerp: %s",
                exc,
            )
            interpolated = self._normalize_quaternions(
                (1.0 - ratios)[:, None] * start_xyzw[None, :]
                + ratios[:, None] * end_xyzw[None, :]
            )
        if self.quaternion_format == "wxyz":
            interpolated = interpolated[:, [3, 0, 1, 2]]
        return interpolated

    def _average_quaternion_window(self, quats: np.ndarray) -> np.ndarray:
        normalized = [
            quat
            for item in quats
            if (quat := self._normalize_quaternion(item)) is not None
        ]
        if not normalized:
            return quats[0]
        aligned = np.asarray(normalized, dtype=np.float64)
        for idx in range(1, aligned.shape[0]):
            if float(np.dot(aligned[0], aligned[idx])) < 0.0:
                aligned[idx] = -aligned[idx]
        result = self._normalize_quaternion(np.sum(aligned, axis=0))
        return aligned[0] if result is None else result

    def _align_quaternion_signs(self, quats: np.ndarray) -> np.ndarray:
        aligned = quats.astype(np.float64, copy=True)
        last_valid: np.ndarray | None = None
        for idx in range(aligned.shape[0]):
            normalized = self._normalize_quaternion(aligned[idx])
            if normalized is None:
                if last_valid is not None:
                    aligned[idx] = last_valid
                continue
            if last_valid is not None and float(np.dot(last_valid, normalized)) < 0.0:
                normalized = -normalized
            aligned[idx] = normalized
            last_valid = normalized
        return aligned

    def _validate_context(
        self,
        context_actions: np.ndarray | None,
        action_dim: int,
    ) -> np.ndarray | None:
        if context_actions is None:
            return None
        context = np.asarray(context_actions)
        if context.ndim == 1:
            context = context.reshape(1, -1)
        if context.ndim != 2 or context.shape[0] == 0 or context.shape[1] != action_dim:
            logger.warning(
                "SmoothChunkManager received incompatible boundary context; ignoring it"
            )
            return None
        return context

    def _validate_last_action(
        self,
        context_actions: np.ndarray | None,
        action_dim: int,
    ) -> np.ndarray | None:
        context = self._validate_context(context_actions, action_dim)
        return None if context is None else context[-1]

    def _resolve_savgol_params(self, length: int) -> tuple[int, int] | None:
        if length < 3:
            return None
        window = min(self.filter_window, length)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return None
        return window, min(self.savgol_polyorder, window - 1)

    def _get_quaternion_slices(self, action_dim: int) -> list[slice]:
        if action_dim == 8:
            return [slice(3, 7)]
        if action_dim == 16:
            return [slice(3, 7), slice(11, 15)]
        logger.warning(
            "SmoothChunkManager action_dim=%s has no known quaternion layout; treating it as a vector",
            action_dim,
        )
        return []

    @staticmethod
    def _get_gripper_indices(action_dim: int) -> list[int]:
        if action_dim == 8:
            return [7]
        if action_dim == 16:
            return [7, 15]
        return []

    @staticmethod
    def _get_non_quaternion_mask(
        action_dim: int, quat_slices: list[slice]
    ) -> np.ndarray:
        mask = np.ones(action_dim, dtype=bool)
        for quat_slice in quat_slices:
            mask[quat_slice] = False
        return mask

    def _to_xyzw(self, quat: np.ndarray) -> np.ndarray:
        if self.quaternion_format == "xyzw":
            return quat.astype(np.float64, copy=False)
        return quat[[1, 2, 3, 0]].astype(np.float64, copy=False)

    @staticmethod
    def _normalize_quaternion(quat: np.ndarray) -> np.ndarray | None:
        if not np.all(np.isfinite(quat)):
            return None
        norm = np.linalg.norm(quat)
        return None if norm < 1e-8 else quat / norm

    @staticmethod
    def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        return quats / np.where(norms < 1e-8, 1.0, norms)

    @staticmethod
    def _normalize_optional_method(
        value: Any,
        valid_methods: set[str],
        config_name: str,
    ) -> str | None:
        if value is None or str(value).strip().lower() in {"", "none", "null"}:
            return None
        method = str(value).strip().lower()
        if method not in valid_methods:
            raise ValueError(
                f"{config_name} must be one of {sorted(valid_methods)} or null, got {value!r}"
            )
        return method

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        result = int(value)
        if result < 1:
            raise ValueError(f"{name} must be at least 1, got {value!r}")
        return result

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        result = float(value)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return result

    def _normalize_filter_window(self, value: Any) -> int:
        window = self._positive_int(value, "smoothing.window")
        if window > 1 and window % 2 == 0:
            logger.info(
                "SmoothChunkManager smoothing.window %s is even; using %s",
                window,
                window + 1,
            )
            window += 1
        return window
