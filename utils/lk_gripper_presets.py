"""Preset parameters and geometry helpers for LKTech grippers."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict


DEFAULT_LK_GRIPPER_TYPE = "soft"

LK_GRIPPER_PRESETS: Dict[str, Dict[str, Any]] = {
    "soft": {
        "min_torque_iq": -40,
        "max_torque_iq": 190,
        "arm_length_m": 0.06543,
        "apex_angle_max_deg": 90.0,
    },
}


def get_lk_gripper_preset(gripper_type: str | None = None) -> Dict[str, Any]:
    """Return a copy of the preset for an LKTech gripper type."""
    resolved_type = (gripper_type or DEFAULT_LK_GRIPPER_TYPE).strip().lower()
    if resolved_type not in LK_GRIPPER_PRESETS:
        available = ", ".join(sorted(LK_GRIPPER_PRESETS))
        raise ValueError(f"Unsupported LK gripper_type '{resolved_type}'. Available: {available}")

    preset = deepcopy(LK_GRIPPER_PRESETS[resolved_type])
    preset["gripper_type"] = resolved_type
    return preset


def calculate_lk_gripper_max_width(
    arm_length_m: float,
    apex_angle_deg: float,
    fallback: float,
) -> float:
    """Calculate physical maximum jaw width as ``2 * L * sin(apex / 2)``.

    Invalid or non-positive geometry falls back to the caller's configured
    width.  Keeping this helper in the shared utility module lets the R1
    adapters remain importable without constructing hardware clients.
    """
    try:
        arm_length = float(arm_length_m)
        apex_radians = math.radians(float(apex_angle_deg))
    except (TypeError, ValueError):
        return float(fallback)

    if (
        not math.isfinite(arm_length)
        or not math.isfinite(apex_radians)
        or arm_length <= 0.0
        or apex_radians <= 0.0
    ):
        return float(fallback)
    return float(2.0 * arm_length * math.sin(apex_radians / 2.0))
