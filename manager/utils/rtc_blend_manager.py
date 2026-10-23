"""Execution-side RTC chunk blending.

The policy keeps the raw action chunk for the next RTC prefix. This manager
only changes the chunk dispatched to the robot: it aligns the new chunk to the
current wall-clock position, then exponentially transfers the command from the
remaining executed chunk to the new raw prediction.
"""

from __future__ import annotations

import numpy as np

from .action_manager import BaseChunkManager, DropFirstActions


class RtcBlendChunkManager(BaseChunkManager):
    """Blend the active chunk tail into each newly-arriving raw chunk."""

    def __init__(
        self,
        blend_steps: int = 12,
        blend_decay_rate: float = 0.45,
        drop_first_actions: DropFirstActions = "auto",
        skip_first_chunk: bool = False,
        execution_fps: float = 30.0,
        interpolate_auto_drop: bool = True,
        quaternion_format: str = "xyzw",
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
        if isinstance(blend_steps, bool) or int(blend_steps) < 0:
            raise ValueError("blend_steps must be a non-negative integer")
        if float(blend_decay_rate) <= 0:
            raise ValueError("blend_decay_rate must be positive")
        self.blend_steps = int(blend_steps)
        self.blend_decay_rate = float(blend_decay_rate)
        self._last_blend_count = 0

    def put(
        self,
        action_chunk_dict: dict[str, np.ndarray],
        observation_timestamp: int | None = None,
    ) -> None:
        if not action_chunk_dict or not self.current_action_chunk_dict:
            self._put_aligned(action_chunk_dict, observation_timestamp)
            return

        alignment = self._resolve_drop_alignment(observation_timestamp)
        aligned_chunks: dict[str, np.ndarray] = {}
        original_offsets: np.ndarray | None = None
        old_step = int(self.current_step)
        for device_name, source in action_chunk_dict.items():
            raw = np.asarray(source)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            aligned = self._time_align_chunk(raw, alignment.offset_steps)
            old_chunk = np.asarray(self.current_action_chunk_dict.get(device_name, raw))
            old_remaining = old_chunk[old_step:].copy()
            aligned_chunks[device_name] = self._blend(old_remaining, aligned)
            if original_offsets is None:
                original_offsets = self._time_align_original_offsets(
                    np.arange(raw.shape[0], dtype=np.float64),
                    alignment.offset_steps,
                )

        if not aligned_chunks or any(chunk.shape[0] == 0 for chunk in aligned_chunks.values()):
            self._put_aligned(
                {},
                observation_timestamp,
                alignment_offset_steps=alignment.offset_steps,
                aligned_drop_count=alignment.start_index,
            )
            return

        self._put_aligned(
            aligned_chunks,
            observation_timestamp,
            original_action_offsets=original_offsets,
            alignment_offset_steps=alignment.offset_steps,
            aligned_drop_count=alignment.start_index,
        )

    def _put_aligned(
        self,
        action_chunk_dict: dict[str, np.ndarray],
        observation_timestamp: int | None,
        *,
        original_action_offsets: np.ndarray | None = None,
        alignment_offset_steps: float = 0.0,
        aligned_drop_count: int = 0,
    ) -> None:
        configured_drop = self.drop_first_actions
        # The incoming arrays have already been aligned and blended. Prevent
        # BaseChunkManager from applying the latency offset a second time.
        self.drop_first_actions = 0
        try:
            BaseChunkManager.put(
                self,
                action_chunk_dict,
                observation_timestamp,
                original_action_offsets=original_action_offsets,
                original_step_interval_ns=self._default_step_interval_ns(),
            )
        finally:
            self.drop_first_actions = configured_drop
        diagnostics = dict(self.last_switch_diagnostics)
        diagnostics.update(
            {
                "manager_type": type(self).__name__,
                "configured_drop_first_actions": configured_drop,
                "alignment_offset_steps": float(alignment_offset_steps),
                "drop_count": int(aligned_drop_count),
                "rtc_blend_applied": bool(diagnostics.get("chunk_replaced", False)),
                "rtc_blend_steps": int(self.blend_steps),
                "rtc_blend_decay_rate": float(self.blend_decay_rate),
                "rtc_blend_old_weight_first": 1.0,
                "rtc_blend_old_weight_last": 0.0 if self._last_blend_count > 1 else 1.0,
                "rtc_blend_actual_steps": int(self._last_blend_count),
                "rtc_blend_input_is_raw_policy_chunk": True,
            }
        )
        self.last_switch_diagnostics = diagnostics

    def _blend(self, old_remaining: np.ndarray, new_aligned: np.ndarray) -> np.ndarray:
        if new_aligned.shape[0] == 0:
            self._last_blend_count = 0
            return new_aligned.copy()
        if old_remaining.shape[0] == 0 or self.blend_steps <= 0:
            self._last_blend_count = 0
            return new_aligned.copy()
        count = min(self.blend_steps, old_remaining.shape[0], new_aligned.shape[0])
        self._last_blend_count = count
        result = new_aligned.astype(np.float64, copy=True)
        old = old_remaining[:count].astype(np.float64, copy=False)
        new = new_aligned[:count].astype(np.float64, copy=False)
        if count == 1:
            old_weight = np.ones((1,), dtype=np.float64)
        else:
            progress = np.arange(count, dtype=np.float64) / float(count - 1)
            # Normalize the exponential over the complete blend horizon. This
            # preserves the old command exactly at k=0 and reaches the raw new
            # command exactly at k=count-1, instead of exhausting most of the
            # blend in the first few control ticks.
            old_weight = np.expm1(self.blend_decay_rate * (1.0 - progress)) / np.expm1(
                self.blend_decay_rate
            )
        new_weight = 1.0 - old_weight
        result[:count] = old * old_weight[:, None] + new * new_weight[:, None]

        # Pose quaternions need spherical interpolation rather than component
        # averaging. Gripper channels remain linear with the same blend curve.
        for quat_slice in self._get_quaternion_slices(result.shape[1]):
            for row_index in range(count):
                slerped = self._slerp_quaternion_segment(
                    old[row_index, quat_slice],
                    new[row_index, quat_slice],
                    np.asarray([new_weight[row_index]]),
                )
                if slerped is not None:
                    result[row_index, quat_slice] = slerped[0]
        return result.astype(new_aligned.dtype, copy=False)
