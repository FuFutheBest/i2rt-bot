"""Shared, hardware-free inference option validation.

The control panel and standalone policy entry points use the same environment
contract for smoke-test inference.  Keeping the validation here avoids making
core consumers import the UI package.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


MOCK_INFERENCE_ENV = "SUPERINFERENCE_CONTROL_PANEL_TEST_MODE"
POLICY_OVERRIDES_ENV = "SUPERINFERENCE_CONTROL_PANEL_POLICY_OVERRIDES"
MOCK_POLICY_CLASS = "ControlPanelNoMotionPolicy"


def as_bool(value: Any) -> bool:
    """Parse configuration booleans consistently across launch surfaces."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off", "none", "null"}:
        return False
    return bool(value)


def _enabled_devices(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_devices = config.get("devices")
    if isinstance(raw_devices, Mapping):
        raw_devices = raw_devices.get("devices")
    if not isinstance(raw_devices, list):
        return []
    devices: list[Mapping[str, Any]] = []
    for device in raw_devices:
        if not isinstance(device, Mapping):
            continue
        enabled = device.get("enabled", True)
        if not as_bool(enabled) or (
            isinstance(enabled, (list, tuple))
            and not any(as_bool(item) for item in enabled)
        ):
            continue
        devices.append(device)
    return devices


def mock_inference_config_error(config: Mapping[str, Any]) -> Optional[str]:
    """Return a safe-mode contract error, or ``None`` when it is valid."""
    from utils.policy_control_config import ControlTargetConfig

    rizon_devices = [
        device
        for device in _enabled_devices(config)
        if str(device.get("class", "")).casefold() == "rizonrobot"
    ]
    if len(rizon_devices) != 1:
        return "Test inference requires exactly one configured dual-arm Rizon device"
    rizon = rizon_devices[0]
    control = rizon.get("control")
    if not isinstance(control, Mapping) or control.get("type") != "eef_abs":
        return "Test inference is only available for absolute TCP control (eef_abs)"
    control_format = control.get("format")
    control_shape = (
        control_format.get("shape") if isinstance(control_format, Mapping) else None
    )
    if not isinstance(control_shape, (list, tuple)) or list(control_shape) != [16]:
        return "Test inference requires a 16D dual-arm control SHM"
    state_feedback = rizon.get("state_feedback")
    state_format = (
        state_feedback.get("format")
        if isinstance(state_feedback, Mapping)
        else None
    )
    data_shape = state_format.get("shape") if isinstance(state_format, Mapping) else None
    if (
        not isinstance(data_shape, (list, tuple))
        or not data_shape
        or int(data_shape[0]) < 44
    ):
        return "Test inference requires the 44D dual-arm Rizon state layout"
    policy = config.get("policy")
    obs = policy.get("obs") if isinstance(policy, Mapping) else None
    observed = obs.get("devices") if isinstance(obs, Mapping) else None
    robot_alias = f"RizonRobot_{rizon.get('device_class_id')}"
    if not isinstance(observed, list) or robot_alias not in observed:
        return (
            "Test inference requires the dual-arm Rizon state in "
            "policy.obs.devices"
        )
    controls = policy.get("controls") if isinstance(policy, Mapping) else None
    compatible_controls = []
    for index, control in enumerate(controls or []):
        if not isinstance(control, Mapping):
            continue
        try:
            target = ControlTargetConfig.from_mapping(
                control, field=f"policy.controls[{index}]"
            )
        except (TypeError, ValueError):
            continue
        if target.shape == (16,) and target.device_key == robot_alias:
            compatible_controls.append(target)
    if len(compatible_controls) != 1 or len(controls or []) != 1:
        return (
            "Test inference requires exactly one 16D policy command target for "
            "the configured Rizon device"
        )
    return None


def mock_gripper_range(config: Mapping[str, Any]) -> tuple[float, float]:
    """Resolve a conservative gripper range for the no-motion policy."""
    rizon = next(
        (
            device
            for device in _enabled_devices(config)
            if str(device.get("class", "")).casefold() == "rizonrobot"
        ),
        {},
    )
    gripper = rizon.get("gripper") if isinstance(rizon, Mapping) else None
    if not isinstance(gripper, Mapping):
        return 0.0, 0.08
    if str(gripper.get("value_mode", "")).casefold() == "normalized":
        return 0.0, 1.0
    maximum = 0.08
    visual = gripper.get("visual")
    articulation = visual.get("articulation") if isinstance(visual, Mapping) else None
    candidate = (
        articulation.get("max_width_m")
        if isinstance(articulation, Mapping)
        else None
    )
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        numeric = float(candidate)
        if math.isfinite(numeric) and numeric > 0:
            maximum = min(numeric, 0.11)
    return 0.0, maximum


__all__ = [
    "MOCK_INFERENCE_ENV",
    "MOCK_POLICY_CLASS",
    "POLICY_OVERRIDES_ENV",
    "as_bool",
    "mock_gripper_range",
    "mock_inference_config_error",
]
