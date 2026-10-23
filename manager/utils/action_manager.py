"""
Time-align incoming action chunks for asynchronous policy execution.

Author: SuperInference Contributors
"""

import math
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from utils.latency_tracker import LatencyTracker
from utils.logger_config import logger


DropFirstActions = Union[int, str]
_STEP_EPSILON = 1e-6


@dataclass(frozen=True)
class _DropAlignment:
    """Resolved temporal offset for one incoming action chunk."""

    offset_steps: float
    start_index: int
    latency_seconds: Optional[float]
    source: str


class BaseChunkManager:
    """
    The basic manager always carries out the action step-by-step 
    in the latest action chunk immediately. The older chunks will be directly
    dropped whenever the new chunk arrives.
    """
    def __init__(
        self,
        drop_first_actions: DropFirstActions = 0,
        skip_first_chunk: bool = False,
        execution_fps: float = 30.0,
        interpolate_auto_drop: bool = True,
        quaternion_format: str = "xyzw",
        auto_drop_gripper_mode: str = "linear",
    ) -> None:
        self.drop_first_actions = self._validate_drop_first_actions(drop_first_actions)
        self.skip_first_chunk = bool(skip_first_chunk)
        self.execution_fps = float(execution_fps)
        if self.drop_first_actions == "auto" and self.execution_fps <= 0:
            raise ValueError("execution_fps must be positive when drop_first_actions='auto'")
        self.interpolate_auto_drop = bool(interpolate_auto_drop)
        self.quaternion_format = self._validate_quaternion_format(quaternion_format)
        self.auto_drop_gripper_mode = self._validate_gripper_mode(auto_drop_gripper_mode)
        self._inference_latency_tracker = LatencyTracker()
        self.current_action_chunk_dict = None
        self.current_step = 0  # Reset step counter for new chunk
        self.last_read_time = 0
        self._has_received_chunk = False
        self.current_original_start_index = 0
        self.current_original_start_offset = 0.0
        self.current_original_action_offsets: Optional[np.ndarray] = None
        self.current_original_step_interval_ns: Optional[int] = None
        self._latest_original_action_index: Optional[int] = None
        self._latest_original_action_offset: Optional[float] = None
        # Keep the latest temporal replacement decision available to optional
        # RTC managers and action-switch diagnostics.  Consumers treat this as
        # an opaque mapping, so the base manager can expose it without pulling
        # in any visualizer or runner-specific code.
        self.last_switch_diagnostics: Dict[str, object] = {}

    @staticmethod
    def _validate_drop_first_actions(drop_first_actions: DropFirstActions) -> DropFirstActions:
        """Validate a fixed drop count or the automatic latency-based mode."""
        if isinstance(drop_first_actions, str):
            normalized_value = drop_first_actions.strip().lower()
            if normalized_value != "auto":
                raise ValueError("drop_first_actions must be a non-negative integer or 'auto'")
            return normalized_value
        if isinstance(drop_first_actions, bool) or not isinstance(drop_first_actions, Integral):
            raise TypeError("drop_first_actions must be a non-negative integer or 'auto'")
        if drop_first_actions < 0:
            raise ValueError("drop_first_actions must be non-negative")
        return int(drop_first_actions)

    @staticmethod
    def _validate_quaternion_format(quaternion_format: str) -> str:
        """Validate the quaternion storage order used by EEF action chunks."""
        normalized_value = str(quaternion_format).strip().lower()
        if normalized_value not in {"xyzw", "wxyz"}:
            raise ValueError("quaternion_format must be 'xyzw' or 'wxyz'")
        return normalized_value

    @staticmethod
    def _validate_gripper_mode(gripper_mode: str) -> str:
        """Validate fractional resampling behavior for gripper channels."""
        normalized_value = str(gripper_mode).strip().lower()
        if normalized_value not in {"linear", "hold"}:
            raise ValueError("auto_drop_gripper_mode must be 'linear' or 'hold'")
        return normalized_value

    def _resolve_drop_alignment(
        self,
        observation_timestamp: Optional[int],
    ) -> _DropAlignment:
        """Resolve this chunk's exact temporal offset from its own observation."""
        if self.drop_first_actions != "auto":
            fixed_steps = int(self.drop_first_actions)
            return _DropAlignment(
                offset_steps=float(fixed_steps),
                start_index=fixed_steps,
                latency_seconds=None,
                source="fixed",
            )

        if observation_timestamp is not None:
            latency_seconds = max(0.0, (time.time_ns() - int(observation_timestamp)) / 1e9)
            self._inference_latency_tracker.add(latency_seconds)
            source = "current_chunk"
        elif len(self._inference_latency_tracker) > 0:
            latency_seconds = self._inference_latency_tracker.p95()
            source = "p95_fallback"
        else:
            latency_seconds = 0.0
            source = "no_latency_data"
            logger.warning(
                "BaseChunkManager cannot auto-align a chunk without an observation "
                "timestamp or latency history; using zero offset"
            )

        offset_steps = max(0.0, latency_seconds * self.execution_fps)
        nearest_integer_offset = round(offset_steps)
        if abs(offset_steps - nearest_integer_offset) <= _STEP_EPSILON:
            offset_steps = float(nearest_integer_offset)
        start_index = max(0, int(math.ceil(offset_steps - _STEP_EPSILON)))
        logger.info(
            "BaseChunkManager auto alignment resolved to "
            f"{offset_steps:.3f} steps (start_index={start_index}, source={source}, "
            f"latency_ms={latency_seconds * 1000.0:.2f}, "
            f"samples={len(self._inference_latency_tracker)}, "
            f"p95_latency_ms={self._inference_latency_tracker.p95() * 1000.0:.2f}, "
            f"execution_fps={self.execution_fps:.2f})"
        )
        return _DropAlignment(
            offset_steps=offset_steps,
            start_index=start_index,
            latency_seconds=latency_seconds,
            source=source,
        )

    def _time_align_chunk(
        self,
        chunk: np.ndarray,
        offset_steps: float,
    ) -> np.ndarray:
        """Resample a trajectory at a fractional source-step offset."""
        source = np.asarray(chunk)
        if source.ndim == 0:
            raise ValueError("Action chunks must have a time dimension")

        base_index = max(0, int(math.floor(offset_steps)))
        fraction = max(0.0, offset_steps - float(base_index))
        if fraction <= _STEP_EPSILON:
            return source[base_index:].copy()
        if base_index + 1 >= source.shape[0]:
            return source[:0].copy()

        start = source[base_index:-1].astype(np.float64, copy=False)
        end = source[base_index + 1 :].astype(np.float64, copy=False)
        interpolated = start + (end - start) * fraction

        if source.ndim == 2:
            quaternion_slices = self._get_quaternion_slices(source.shape[1])
            for quaternion_slice in quaternion_slices:
                for row_index in range(interpolated.shape[0]):
                    slerped = self._slerp_quaternion_segment(
                        start[row_index, quaternion_slice],
                        end[row_index, quaternion_slice],
                        np.asarray([fraction], dtype=np.float64),
                    )
                    if slerped is not None:
                        interpolated[row_index, quaternion_slice] = slerped[0]

            if self.auto_drop_gripper_mode == "hold":
                gripper_indices = self._get_gripper_indices(source.shape[1])
                if gripper_indices:
                    interpolated[:, gripper_indices] = start[:, gripper_indices]

        output_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.float64
        return interpolated.astype(output_dtype, copy=False)

    def _slerp_quaternion_segment(
        self,
        start_quaternion: np.ndarray,
        end_quaternion: np.ndarray,
        ratios: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Interpolate one quaternion segment with SciPy's shortest-path SLERP."""
        start_xyzw = self._normalize_quaternion(self._to_xyzw(start_quaternion))
        end_xyzw = self._normalize_quaternion(self._to_xyzw(end_quaternion))
        if start_xyzw is None or end_xyzw is None:
            return None
        try:
            interpolated = Slerp(
                [0.0, 1.0],
                Rotation.from_quat([start_xyzw, end_xyzw]),
            )(ratios).as_quat()
        except ValueError:
            interpolated = self._normalize_quaternions(
                (1.0 - ratios)[:, None] * start_xyzw[None, :]
                + ratios[:, None] * end_xyzw[None, :]
            )
        if self.quaternion_format == "wxyz":
            interpolated = interpolated[:, [3, 0, 1, 2]]
        return interpolated

    @staticmethod
    def _get_quaternion_slices(action_dim: int) -> list[slice]:
        """Return known single- and dual-arm EEF quaternion layouts."""
        if action_dim == 8:
            return [slice(3, 7)]
        if action_dim == 16:
            return [slice(3, 7), slice(11, 15)]
        return []

    @staticmethod
    def _get_gripper_indices(action_dim: int) -> list[int]:
        """Return known single- and dual-arm EEF gripper channels."""
        if action_dim == 8:
            return [7]
        if action_dim == 16:
            return [7, 15]
        return []

    def _to_xyzw(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert the configured quaternion order to SciPy's xyzw order."""
        if self.quaternion_format == "xyzw":
            return quaternion.astype(np.float64, copy=False)
        return quaternion[[1, 2, 3, 0]].astype(np.float64, copy=False)

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> Optional[np.ndarray]:
        """Normalize one quaternion, returning None for invalid input."""
        if not np.all(np.isfinite(quaternion)):
            return None
        norm = float(np.linalg.norm(quaternion))
        return None if norm < 1e-8 else quaternion / norm

    @staticmethod
    def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
        """Normalize a batch of quaternions with a safe zero-norm fallback."""
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        return quaternions / np.where(norms < 1e-8, 1.0, norms)

    def is_empty(self) -> bool:
        """Check if there is no chunk in the cache"""
        return self.current_action_chunk_dict is None

    def is_new_chunk(self, observation_timestamp: Optional[int] = None) -> bool:
        """Return True if the current observation timestamp is newer than the previous one, """
        return observation_timestamp is not None and observation_timestamp != self.last_read_time

    def get_current_action_chunk(self) -> Dict[str, np.ndarray]:
        """Return an isolated snapshot of the post-processed chunk being executed."""
        if not self.current_action_chunk_dict:
            return {}
        return {
            device_name: np.asarray(chunk).copy()
            for device_name, chunk in self.current_action_chunk_dict.items()
        }

    def get_current_original_action_offsets(self) -> np.ndarray:
        """Return each cached action's exact offset in the incoming policy chunk."""
        if self.current_original_action_offsets is not None:
            return self.current_original_action_offsets.copy()
        if not self.current_action_chunk_dict:
            return np.empty((0,), dtype=np.float64)

        chunk_length = min(
            len(np.asarray(chunk)) for chunk in self.current_action_chunk_dict.values()
        )
        return self.current_original_start_offset + np.arange(
            chunk_length,
            dtype=np.float64,
        )

    def get_original_action_step_interval_ns(self) -> int:
        """Return the time represented by one original policy action step."""
        if self.current_original_step_interval_ns is None:
            return self._default_step_interval_ns()
        return int(self.current_original_step_interval_ns)

    def _default_step_interval_ns(self) -> int:
        """Return the execution interval used by untransformed action chunks."""
        if self.execution_fps <= 0:
            return 0
        return int(round(1e9 / self.execution_fps))

    @staticmethod
    def _prepare_original_action_offsets(
        original_action_offsets: Optional[np.ndarray],
        chunk_length: int,
    ) -> np.ndarray:
        """Validate or construct the source-index timeline for a processed chunk."""
        if original_action_offsets is None:
            return np.arange(chunk_length, dtype=np.float64)

        offsets = np.asarray(original_action_offsets, dtype=np.float64)
        if offsets.ndim != 1 or offsets.shape[0] != chunk_length:
            raise ValueError(
                "original_action_offsets must be one-dimensional and match "
                f"the processed chunk length ({chunk_length})"
            )
        if not np.all(np.isfinite(offsets)):
            raise ValueError("original_action_offsets must contain only finite values")
        if offsets.size > 1 and np.any(np.diff(offsets) < -_STEP_EPSILON):
            raise ValueError("original_action_offsets must be non-decreasing")
        return offsets.copy()

    @staticmethod
    def _time_align_original_offsets(
        original_action_offsets: np.ndarray,
        offset_steps: float,
    ) -> np.ndarray:
        """Apply the same fractional slicing used for the processed action data."""
        base_index = max(0, int(math.floor(offset_steps)))
        fraction = max(0.0, offset_steps - float(base_index))
        if fraction <= _STEP_EPSILON:
            return original_action_offsets[base_index:].copy()
        if base_index + 1 >= original_action_offsets.shape[0]:
            return original_action_offsets[:0].copy()
        return (
            original_action_offsets[base_index:-1] * (1.0 - fraction)
            + original_action_offsets[base_index + 1 :] * fraction
        )

    @staticmethod
    def _offset_to_action_index(original_action_offset: float) -> int:
        """Convert a fractional original offset to the integer SHM action index."""
        return max(0, int(math.ceil(original_action_offset - _STEP_EPSILON)))

    def put(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        observation_timestamp: Optional[int] = None,
        *,
        original_action_offsets: Optional[np.ndarray] = None,
        alignment_origin_steps: float = 0.0,
        original_step_interval_ns: Optional[int] = None,
    ) -> None:
        """Put the action chunk dict to local chunk cache"""
        if not action_chunk_dict:
            self.current_action_chunk_dict = {}
            self.current_step = 0
            self.last_read_time = observation_timestamp
            self.current_original_start_index = 0
            self.current_original_start_offset = 0.0
            self.current_original_action_offsets = None
            self.current_original_step_interval_ns = None
            self._latest_original_action_index = None
            self._latest_original_action_offset = None
            self.last_switch_diagnostics = {
                "manager_type": type(self).__name__,
                "chunk_replaced": False,
                "new_processed_chunk_length": 0,
            }
            logger.warning("BaseChunkManager received an empty action chunk")
            return

        if self.skip_first_chunk and not self._has_received_chunk:
            self.current_action_chunk_dict = {}
            self.current_step = 0
            self.last_read_time = observation_timestamp
            self._has_received_chunk = True
            self.current_original_start_index = 0
            self.current_original_start_offset = 0.0
            self.current_original_action_offsets = None
            self.current_original_step_interval_ns = None
            self._latest_original_action_index = None
            self._latest_original_action_offset = None
            self.last_switch_diagnostics = {
                "manager_type": type(self).__name__,
                "chunk_replaced": False,
                "skipped_first_chunk": True,
            }
            logger.warning("BaseChunkManager skipped the first received action chunk")
            return

        had_previous_chunk = bool(self.current_action_chunk_dict)
        previous_current_step = int(self.current_step)
        previous_chunk_length = 0
        if had_previous_chunk:
            previous_chunk_length = min(
                len(np.asarray(chunk))
                for chunk in self.current_action_chunk_dict.values()
            )
        previous_remaining_steps = max(
            0, previous_chunk_length - previous_current_step
        )
        previous_original_current_offset = self._latest_original_action_offset
        previous_observation_timestamp = self.last_read_time

        processed_chunk_length = min(
            len(np.asarray(chunk)) for chunk in action_chunk_dict.values()
        )
        resolved_original_offsets = self._prepare_original_action_offsets(
            original_action_offsets,
            processed_chunk_length,
        )
        resolved_original_step_interval_ns = (
            self._default_step_interval_ns()
            if original_step_interval_ns is None
            else int(original_step_interval_ns)
        )
        if resolved_original_step_interval_ns <= 0:
            raise ValueError("original_step_interval_ns must be positive")
        if not math.isfinite(float(alignment_origin_steps)):
            raise ValueError("alignment_origin_steps must be finite")

        resolved_alignment = self._resolve_drop_alignment(observation_timestamp)
        if not self._has_received_chunk:
            applied_offset_steps = 0.0
            drop_count = 0
        elif self.drop_first_actions == "auto":
            applied_offset_steps = max(
                0.0,
                resolved_alignment.offset_steps + float(alignment_origin_steps),
            )
            drop_count = max(
                0,
                int(math.ceil(applied_offset_steps - _STEP_EPSILON)),
            )
            if abs(float(alignment_origin_steps)) > _STEP_EPSILON:
                logger.info(
                    "BaseChunkManager shifted auto-alignment by "
                    f"{float(alignment_origin_steps):.3f} processed steps "
                    "to account for the original action-zero timeline"
                )
        else:
            applied_offset_steps = resolved_alignment.offset_steps
            drop_count = resolved_alignment.start_index

        original_chunk_length = processed_chunk_length
        if drop_count > 0:
            device_names = list(action_chunk_dict.keys())
            min_chunk_length = original_chunk_length
            if drop_count >= min_chunk_length:
                self.current_action_chunk_dict = {}
                self.current_step = 0
                self.last_read_time = observation_timestamp
                self.current_original_start_index = 0
                self.current_original_start_offset = 0.0
                self.current_original_action_offsets = None
                self.current_original_step_interval_ns = (
                    resolved_original_step_interval_ns
                )
                self._latest_original_action_index = None
                self._latest_original_action_offset = None
                self.last_switch_diagnostics = {
                    "manager_type": type(self).__name__,
                    "chunk_replaced": had_previous_chunk,
                    "previous_current_step": previous_current_step,
                    "previous_chunk_length": previous_chunk_length,
                    "previous_remaining_steps": previous_remaining_steps,
                    "previous_original_current_offset": previous_original_current_offset,
                    "previous_observation_timestamp_ns": previous_observation_timestamp,
                    "alignment_offset_steps": float(applied_offset_steps),
                    "drop_count": int(drop_count),
                    "new_processed_chunk_length": 0,
                }
                logger.warning(
                    "BaseChunkManager dropped entire action chunk because "
                    f"drop_first_actions={drop_count} >= min_chunk_length={min_chunk_length}"
                )
                return
            fractional_offset = (
                abs(applied_offset_steps - round(applied_offset_steps)) > _STEP_EPSILON
            )
            if (
                self.drop_first_actions == "auto"
                and self.interpolate_auto_drop
                and fractional_offset
            ):
                logger.info(
                    "BaseChunkManager time-aligning new chunk at "
                    f"{applied_offset_steps:.3f} steps for devices={device_names}"
                )
                action_chunk_dict = {
                    device_name: self._time_align_chunk(chunk, applied_offset_steps)
                    for device_name, chunk in action_chunk_dict.items()
                }
                resolved_original_offsets = self._time_align_original_offsets(
                    resolved_original_offsets,
                    applied_offset_steps,
                )
            else:
                logger.info(
                    "BaseChunkManager dropping first "
                    f"{drop_count} actions from new chunk for devices={device_names}"
                )
                action_chunk_dict = {
                    device_name: chunk[drop_count:]
                    for device_name, chunk in action_chunk_dict.items()
                }
                resolved_original_offsets = resolved_original_offsets[drop_count:].copy()
                applied_offset_steps = float(drop_count)

        # Deep copy to avoid shared memory reference issues
        self.current_action_chunk_dict = {
            k: v.copy() for k, v in action_chunk_dict.items()
        }
        self.current_step = 0
        self.last_read_time = observation_timestamp
        self._has_received_chunk = True
        self.current_original_action_offsets = resolved_original_offsets.copy()
        self.current_original_step_interval_ns = resolved_original_step_interval_ns
        self.current_original_start_offset = (
            float(resolved_original_offsets[0])
            if resolved_original_offsets.size > 0
            else 0.0
        )
        self.current_original_start_index = self._offset_to_action_index(
            self.current_original_start_offset
        )
        self._latest_original_action_index = None
        self._latest_original_action_offset = None
        self.last_switch_diagnostics = {
            "manager_type": type(self).__name__,
            "execution_fps": float(self.execution_fps),
            "configured_drop_first_actions": self.drop_first_actions,
            "chunk_replaced": had_previous_chunk,
            "previous_current_step": previous_current_step,
            "previous_chunk_length": previous_chunk_length,
            "previous_remaining_steps": previous_remaining_steps,
            "previous_original_current_offset": previous_original_current_offset,
            "previous_observation_timestamp_ns": previous_observation_timestamp,
            # The runner records authoritative write/dispatch timestamps; do
            # not sample the wall clock a second time in the manager.
            "manager_put_timestamp_ns": None,
            "incoming_observation_age_ms": None,
            "alignment_offset_steps": float(applied_offset_steps),
            "drop_count": int(drop_count),
            "new_processed_chunk_length": int(processed_chunk_length),
            "new_original_start_index": int(self.current_original_start_index),
            "new_original_start_offset": float(self.current_original_start_offset),
            "new_original_step_interval_ns": int(resolved_original_step_interval_ns),
        }

    def get(
        self,
        query_timestamp: Optional[int] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get next action dict from cached current chunk dict."""
        try:
            if self.current_action_chunk_dict is None:
                return None
            
            # Get the chunk length from any device (they should all have the same chunk length)
            device_names = list(self.current_action_chunk_dict.keys())
            if not device_names:
                return None
            
            first_chunk = self.current_action_chunk_dict[device_names[0]]
            chunk_length = len(first_chunk)
            
            # Check if we have more actions in current chunk
            if self.current_step < chunk_length:
                # Extract action for current step from all devices
                action_dict = {}
                for device_name, chunk in self.current_action_chunk_dict.items():
                    action_dict[device_name] = chunk[self.current_step]
                
                if (
                    self.current_original_action_offsets is not None
                    and self.current_step
                    < self.current_original_action_offsets.shape[0]
                ):
                    self._latest_original_action_offset = float(
                        self.current_original_action_offsets[self.current_step]
                    )
                else:
                    self._latest_original_action_offset = float(
                        self.current_original_start_offset + self.current_step
                    )
                self._latest_original_action_index = self._offset_to_action_index(
                    self._latest_original_action_offset
                )
                self.current_step += 1
                return action_dict
            
            # No more actions in current chunk
            return None
                
        except Exception as e:
            logger.error(f"Error getting next action from current chunk: {e}")
            return None
    
    def get_original_current_step(self) -> Optional[int]:
        """Return the original chunk index for the most recently dispatched action."""
        return self._latest_original_action_index

    def get_original_current_offset(self) -> Optional[float]:
        """Return the fractional original-chunk offset of the latest action."""
        return self._latest_original_action_offset

    def reset(self) -> None:
        """Reset the chunk manager state - clear all cached actions."""
        self.current_action_chunk_dict = None
        self.current_step = 0
        self.last_read_time = 0
        self._has_received_chunk = False
        self.current_original_start_index = 0
        self.current_original_start_offset = 0.0
        self.current_original_action_offsets = None
        self.current_original_step_interval_ns = None
        self._latest_original_action_index = None
        self._latest_original_action_offset = None
        self.last_switch_diagnostics = {}
        self._inference_latency_tracker.reset()
        logger.info("ChunkManager: Action cache cleared (reset called)")

    def get_last_switch_diagnostics(self) -> Dict[str, object]:
        """Return the latest temporal replacement/alignment decision."""
        return dict(self.last_switch_diagnostics)
        
