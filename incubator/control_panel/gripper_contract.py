"""Pure configuration helpers shared by gripper-facing workflows.

This module intentionally has no hardware or optional-SDK imports.  The
standalone preflight can therefore use the same arm-shape rules as the Control
Panel without making its ``--help`` path depend on Flask or serial packages.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def rizon_gripper_side_count(
    device: Mapping[str, Any], gripper: Mapping[str, Any] | None = None
) -> int:
    """Return the number of Rizon gripper slots represented by a config.

    ``RizonRobot`` treats a list-valued ``robot_sn`` together with a control
    width of at least 14 as dual-arm.  Configs from older releases sometimes
    omit one of those fields, so the helper falls back to the shape of nested
    gripper values.  A scalar serial or scalar-only gripper is single-arm.
    """

    raw_serials = device.get("robot_sn", device.get("serial_number"))
    control = device.get("control")
    control = control if isinstance(control, Mapping) else {}
    control_format = control.get("format")
    control_format = control_format if isinstance(control_format, Mapping) else {}
    raw_shape = device.get(
        "control_shape", control_format.get("shape")
    )
    if isinstance(raw_shape, (list, tuple)):
        raw_shape = raw_shape[0] if raw_shape else None
    shape: int | None = None
    if isinstance(raw_shape, bool):
        shape = None
    else:
        try:
            numeric_shape = float(raw_shape) if raw_shape is not None else None
            if numeric_shape is not None and math.isfinite(numeric_shape):
                shape = int(numeric_shape)
        except (TypeError, ValueError, OverflowError):
            shape = None

    # Match the runtime constructor when an explicit control shape is present:
    # only a list-valued serial can make a shape >= 14 dual-arm.
    if shape is not None:
        if shape < 14:
            return 1
        return 2 if isinstance(raw_serials, (list, tuple)) else 1

    if isinstance(raw_serials, (list, tuple)):
        if len(raw_serials) >= 2:
            return 2
        if len(raw_serials) == 1:
            return 1
    elif raw_serials is not None and str(raw_serials).strip():
        return 1

    if isinstance(gripper, Mapping):
        # With no serial/shape metadata, preserve an explicitly two-slot
        # contract while allowing scalar and one-item single-arm presets.
        for key in (
            "enabled",
            "port",
            "motor_id",
            "baudrate",
            "speed_dps",
            "calibration_files",
        ):
            value = gripper.get(key)
            if isinstance(value, (list, tuple)):
                if len(value) >= 2:
                    return 2
                if len(value) == 1:
                    return 1

    # The Rizon constructor's defaults are single-arm (string serial and
    # control_shape=(7,)); choose that conservative default for incomplete
    # synthetic mappings too.
    return 1


def rizon_gripper_enabled_sides(
    device: Mapping[str, Any], gripper: Mapping[str, Any]
) -> tuple[bool, ...]:
    """Use state/control gripper primitives for native structured configs.

    Only legacy panel mappings without stream formats use the old enable flag.
    Imports stay local so inspecting the panel never imports device adapters.
    """
    side_count = rizon_gripper_side_count(device, gripper)
    state = device.get("state_feedback")
    control = device.get("control")
    if isinstance(state, Mapping) and isinstance(control, Mapping):
        from devices.robots.rizon.config import (
            validate_control_layouts,
            validate_state_schema,
        )
        from utils.data_format import parse_data_format

        state_layouts = validate_state_schema(
            parse_data_format(state["format"]), side_count
        ).arm_layouts
        control_layouts = validate_control_layouts(
            control["type"], parse_data_format(control["format"]), side_count
        )
        return tuple(
            observed.find("gripper") is not None or commanded.find("gripper") is not None
            for observed, commanded in zip(state_layouts, control_layouts)
        )

    enabled = gripper.get("enabled", False)
    values = list(enabled) if isinstance(enabled, (list, tuple)) else [enabled] * side_count
    return tuple(
        index < len(values)
        and (
            values[index] is True
            or str(values[index]).strip().casefold() in {"1", "true", "yes", "on"}
        )
        for index in range(side_count)
    )


__all__ = ["rizon_gripper_side_count", "rizon_gripper_enabled_sides"]
