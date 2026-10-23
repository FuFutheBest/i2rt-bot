"""Utilities for handling debug metadata columns in action chunks."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Columns 0-7 correspond to the left arm (position, quaternion, gripper)
_DEBUG_INSERT_START = 8  # insert after the left gripper column
_DEBUG_COLUMN_COUNT = 2  # [step_index, iteration]
_SINGLE_ARM_CORE_DIM = 8
_DUAL_ARM_CORE_DIM = 16
_VALID_CORE_DIMS = {_SINGLE_ARM_CORE_DIM, _DUAL_ARM_CORE_DIM}
_VALID_DEBUG_DIMS = {
    _SINGLE_ARM_CORE_DIM + _DEBUG_COLUMN_COUNT,
    _DUAL_ARM_CORE_DIM + _DEBUG_COLUMN_COUNT,
}


def has_debug_metadata(chunk: np.ndarray) -> bool:
    """Return True if chunk includes the debug columns injected for inspection."""

    return chunk.ndim == 2 and chunk.shape[1] in _VALID_DEBUG_DIMS


def split_debug_metadata(chunk: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (core_chunk, metadata) where metadata holds the debug columns.

    The debug metadata is stored between the left and right arm columns:
      [... left data ..., step_index, iteration, ... right data ...]
    """

    if not has_debug_metadata(chunk):
        return chunk, None

    metadata = chunk[:, _DEBUG_INSERT_START : _DEBUG_INSERT_START + _DEBUG_COLUMN_COUNT].copy()
    left = chunk[:, :_DEBUG_INSERT_START]
    right = chunk[:, _DEBUG_INSERT_START + _DEBUG_COLUMN_COUNT :]
    core = np.concatenate([left, right], axis=1)
    return core, metadata


def merge_debug_metadata(core_chunk: np.ndarray, metadata: Optional[np.ndarray]) -> np.ndarray:
    """Re-insert metadata columns after the left-arm block if metadata is provided."""

    if metadata is None:
        return core_chunk

    if core_chunk.shape[0] != metadata.shape[0]:
        raise ValueError(
            "Metadata row count must match chunk row count: "
            f"{metadata.shape[0]} != {core_chunk.shape[0]}"
        )

    if core_chunk.shape[1] not in _VALID_CORE_DIMS:
        raise ValueError(
            "Unsupported core chunk width for debug metadata insertion: "
            f"{core_chunk.shape[1]}"
        )

    left = core_chunk[:, :_DEBUG_INSERT_START]
    right = core_chunk[:, _DEBUG_INSERT_START:]
    return np.concatenate([left, metadata, right], axis=1)


def remove_debug_metadata(chunk: np.ndarray) -> np.ndarray:
    """Convenience helper: return chunk without debug columns."""

    core, _ = split_debug_metadata(chunk)
    return core
