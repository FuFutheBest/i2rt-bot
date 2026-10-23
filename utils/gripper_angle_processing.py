#!/usr/bin/env python3
"""
Gripper angle stabilization utilities.

Provides angle unwrapping and spike rejection for rotary-encoder and
multi-turn motor angle streams.

Author: Zixi Ying
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

ANGLE_PERIOD_DEG: float = 360.0
DEFAULT_MAX_JUMP_DEG: float = 45.0


def unwrap_angle_to_reference(raw_angle: float, reference_angle: float) -> float:
    """Map a raw angle to the 360-deg equivalence class nearest to reference."""
    period = ANGLE_PERIOD_DEG
    offset = round((float(reference_angle) - float(raw_angle)) / period)
    return float(raw_angle) + offset * period


def stabilize_angle_reading(
    raw_angle: float,
    last_angle: Optional[float],
    *,
    max_jump_deg: float = DEFAULT_MAX_JUMP_DEG,
) -> Tuple[float, float]:
    """Unwrap and reject spikes; returns (stable_angle, updated_last_angle)."""
    raw = float(raw_angle)
    if last_angle is None:
        return raw, raw

    reference = float(last_angle)
    candidate = unwrap_angle_to_reference(raw, reference)
    if abs(candidate - reference) > float(max_jump_deg):
        return reference, reference

    return candidate, candidate


def unwrap_angle_sequence(
    raw_angles: List[float],
    *,
    max_jump_deg: float = DEFAULT_MAX_JUMP_DEG,
) -> List[float]:
    """Build a continuous unwrapped angle sequence from raw readings."""
    if not raw_angles:
        return []

    unwrapped: List[float] = [float(raw_angles[0])]
    for raw_angle in raw_angles[1:]:
        stable, updated_last = stabilize_angle_reading(
            float(raw_angle),
            unwrapped[-1],
            max_jump_deg=max_jump_deg,
        )
        if updated_last == unwrapped[-1] and stable == unwrapped[-1]:
            continue
        unwrapped.append(stable)

    if len(unwrapped) < 2:
        return unwrapped

    angles_rad = np.deg2rad(np.asarray(unwrapped, dtype=np.float64))
    return np.rad2deg(np.unwrap(angles_rad)).tolist()


def filter_spikes_by_diff(
    raw_angles: List[float],
    timestamps: List[float],
    *,
    max_jump_deg: float = DEFAULT_MAX_JUMP_DEG,
) -> Tuple[List[float], List[float]]:
    """Drop raw samples whose unwrapped jump exceeds max_jump_deg."""
    if not raw_angles:
        return [], []

    filtered_raw = [float(raw_angles[0])]
    filtered_ts = [float(timestamps[0])]
    last_unwrapped = filtered_raw[0]

    for raw_angle, ts in zip(raw_angles[1:], timestamps[1:]):
        stable, updated_last = stabilize_angle_reading(
            float(raw_angle),
            last_unwrapped,
            max_jump_deg=max_jump_deg,
        )
        if updated_last == last_unwrapped and stable == last_unwrapped:
            continue
        filtered_raw.append(float(raw_angle))
        filtered_ts.append(float(ts))
        last_unwrapped = stable

    return filtered_raw, filtered_ts
