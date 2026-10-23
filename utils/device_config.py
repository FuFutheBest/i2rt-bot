"""Validation helpers for readable device configuration sections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from utils.device_identity import require_device_class_id

from devices.config import StateFeedbackConfig
from devices.contracts import DeviceKind, parse_device_kind
from devices.robots.config import RobotControlConfig, SourceSelectionConfig
from devices.robots.rizon.config import (
    RizonGripperConfig,
    RizonMotionLimits,
    validate_control_layouts,
    validate_state_schema,
)


class DeviceConfigError(ValueError):
    """Raised when a device configuration contract is invalid."""


REMOVED_DEVICE_CONFIG_FIELDS = frozenset(
    {
        "device_id",
        "device_kind",
        "actuators",
        "enabled_arms",
        "input_shm_names",
        "input_flag_shm_name",
        "dual_source_selection",
    }
)


def reject_removed_device_fields(
    device_config: Mapping[str, Any], *, field: str = "device"
) -> None:
    """Reject removed topology/capability fields instead of dropping them."""
    removed = REMOVED_DEVICE_CONFIG_FIELDS & set(device_config)
    if removed:
        raise DeviceConfigError(
            f"{field} contains removed fields: {', '.join(sorted(removed))}"
        )


@dataclass(frozen=True)
class ParsedDeviceConfig:
    """One validated device config ready to pass to a device constructor."""

    device_class: str
    device_class_id: int
    device_kind: DeviceKind
    state_feedback: StateFeedbackConfig
    control: RobotControlConfig | None
    raw: dict[str, Any]

    @property
    def source_selection(self) -> SourceSelectionConfig | None:
        """Return the typed command-source contract, when the device is writable."""
        return self.control.source_selection if self.control is not None else None

    def constructor_kwargs(self) -> dict[str, Any]:
        """Return constructor values with parsed common contracts injected."""
        kwargs = dict(self.raw)
        kwargs.pop("class", None)
        kwargs["state_feedback"] = self.state_feedback
        if self.control is not None:
            kwargs["control"] = self.control
        return kwargs


def state_feedback_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the required state stream mapping."""
    state_feedback = config.get("state_feedback")
    if not isinstance(state_feedback, Mapping):
        raise DeviceConfigError("device requires a state_feedback mapping")
    return state_feedback


def state_format_mapping(config: Mapping[str, Any]) -> Any:
    """Return the required state stream format mapping."""
    return state_feedback_mapping(config).get("format")


def source_selection_config(
    config: Mapping[str, Any],
) -> SourceSelectionConfig:
    """Parse or retrieve the typed command-source selection contract.

    Device construction normally obtains this object from
    :class:`ParsedDeviceConfig`.  This helper is for callers that receive a
    standalone raw device mapping (for example, a replay or preflight tool).
    Omission intentionally resolves to the contract's single self-command
    default; malformed explicit values fail with a device-config error.
    """
    raw_control = config.get("control")
    if isinstance(raw_control, RobotControlConfig):
        return raw_control.source_selection
    if not isinstance(raw_control, Mapping):
        raise DeviceConfigError("device requires a control mapping")
    try:
        return SourceSelectionConfig.from_mapping(raw_control.get("source_selection"))
    except (TypeError, ValueError) as exc:
        raise DeviceConfigError(str(exc)) from exc


def control_source_selection(config: Mapping[str, Any]) -> SourceSelectionConfig:
    """Explicit alias for :func:`source_selection_config` used by runtimes."""
    return source_selection_config(config)


def _rizon_arm_count(config: Mapping[str, Any]) -> int | None:
    if str(config.get("class", "")).casefold() not in {
        "rizonrobot",
        "rizonrobotlkforcegated",
    }:
        return None
    robot_sn = config.get("robot_sn")
    if isinstance(robot_sn, Sequence) and not isinstance(robot_sn, (str, bytes)):
        return len(robot_sn)
    return 1


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeviceConfigError(f"{field} must be a mapping")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    field: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise DeviceConfigError(
            f"{field} contains unknown fields: {', '.join(sorted(unknown))}"
        )


def _positive_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DeviceConfigError(f"{field} must be a number") from exc
    if number <= 0.0:
        raise DeviceConfigError(f"{field} must be positive")
    return number


def _positive_vector(value: Any, *, field: str) -> list[float]:
    if isinstance(value, Mapping):
        unknown = set(value) - {"left", "right"}
        if unknown:
            raise DeviceConfigError(
                f"{field} supports only left/right arm keys, got {sorted(unknown)}"
            )
        if not value:
            raise DeviceConfigError(f"{field} must not be empty")
        ordered_values: list[Any] = []
        for arm in ("left", "right"):
            if arm not in value:
                continue
            arm_values = value[arm]
            if isinstance(arm_values, (str, bytes)) or not isinstance(
                arm_values, Sequence
            ):
                raise DeviceConfigError(f"{field}.{arm} must be a sequence")
            ordered_values.extend(arm_values)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise DeviceConfigError(f"{field} must be a sequence or left/right mapping")
        ordered_values = list(value)

    if not ordered_values:
        raise DeviceConfigError(f"{field} must not be empty")
    return [_positive_number(item, field=field) for item in ordered_values]


def _validate_motion_limits(config: Mapping[str, Any]) -> None:
    raw_limits = config.get("motion_limits")
    if raw_limits is None:
        return
    limits = _require_mapping(raw_limits, field="motion_limits")
    _reject_unknown_keys(
        limits,
        allowed={
            "velocity",
            "acceleration",
            "max_tcp_contact_wrench",
            "workspace",
        },
        field="motion_limits",
    )
    velocity = _require_mapping(
        limits.get("velocity", {}), field="motion_limits.velocity"
    )
    acceleration = _require_mapping(
        limits.get("acceleration", {}), field="motion_limits.acceleration"
    )
    _reject_unknown_keys(
        velocity,
        allowed={"joints_rad_s", "tcp_linear_m_s", "tcp_angular_rad_s"},
        field="motion_limits.velocity",
    )
    _reject_unknown_keys(
        acceleration,
        allowed={"joints_rad_s2", "tcp_linear_m_s2", "tcp_angular_rad_s2"},
        field="motion_limits.acceleration",
    )

    arm_count = _rizon_arm_count(config)
    for section, nested_key in (
        (velocity, "joints_rad_s"),
        (acceleration, "joints_rad_s2"),
    ):
        if nested_key not in section:
            continue
        vector = _positive_vector(
            section[nested_key],
            field=f"motion_limits.{nested_key}",
        )
        if arm_count is not None and len(vector) != arm_count * 7:
            raise DeviceConfigError(
                f"motion_limits.{nested_key} must contain 7 values per "
                f"Rizon arm; expected {arm_count * 7}, got {len(vector)}"
            )

    for section, nested_key in (
        (velocity, "tcp_linear_m_s"),
        (velocity, "tcp_angular_rad_s"),
        (acceleration, "tcp_linear_m_s2"),
        (acceleration, "tcp_angular_rad_s2"),
    ):
        if nested_key in section:
            _positive_number(
                section[nested_key],
                field=f"motion_limits.{nested_key}",
            )

    if "max_tcp_contact_wrench" in limits:
        wrench = limits["max_tcp_contact_wrench"]
        if wrench is not None and len(
            _positive_vector(wrench, field="motion_limits.max_tcp_contact_wrench")
        ) != 6:
            raise DeviceConfigError(
                "motion_limits.max_tcp_contact_wrench must contain 6 values"
            )
    if "workspace" in limits:
        _require_mapping(limits["workspace"], field="motion_limits.workspace")


def _validate_rizon_sections(
    config: Mapping[str, Any],
    state: StateFeedbackConfig,
    control: RobotControlConfig,
) -> None:
    arm_count = _rizon_arm_count(config)
    if arm_count is None:
        return
    try:
        state_schema = validate_state_schema(state.format, arm_count)
        control_layouts = validate_control_layouts(
            control.type, control.format, arm_count
        )
        gripper_required_by_arm = [
            state_layout.find("gripper") is not None
            or control_layout.find("gripper") is not None
            for state_layout, control_layout in zip(
                state_schema.arm_layouts, control_layouts
            )
        ]
        RizonGripperConfig.from_mapping(
            config.get("gripper"), gripper_required_by_arm=gripper_required_by_arm
        )
        RizonMotionLimits.from_mapping(
            config.get("motion_limits"), arm_count=arm_count
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeviceConfigError(str(exc)) from exc


def parse_device_config(
    device_config: Mapping[str, Any],
    *,
    device_kind: DeviceKind | str,
) -> ParsedDeviceConfig:
    """Parse common stream contracts exactly once for device construction."""
    if not isinstance(device_config, Mapping):
        raise DeviceConfigError("device config must be a mapping")
    reject_removed_device_fields(device_config)
    device_class = str(device_config.get("class", "")).strip()
    if not device_class:
        raise DeviceConfigError("device class is required")
    if "device_class_id" not in device_config:
        raise DeviceConfigError("device_class_id is required")
    try:
        device_class_id = require_device_class_id(device_config["device_class_id"])
    except ValueError as exc:
        raise DeviceConfigError(str(exc)) from exc

    kind = parse_device_kind(device_kind)
    raw_state = device_config.get("state_feedback")
    if not isinstance(raw_state, (Mapping, StateFeedbackConfig)):
        raise DeviceConfigError(f"{device_class} requires a state_feedback mapping")
    try:
        state = StateFeedbackConfig.require(raw_state, device_kind=kind)
    except (TypeError, ValueError) as exc:
        raise DeviceConfigError(str(exc)) from exc

    raw_control = device_config.get("control")
    control: RobotControlConfig | None = None
    if kind.writable:
        if not isinstance(raw_control, (Mapping, RobotControlConfig)):
            raise DeviceConfigError(f"{device_class} requires a control mapping")
        try:
            control = RobotControlConfig.require(raw_control)
        except (TypeError, ValueError) as exc:
            raise DeviceConfigError(str(exc)) from exc
    elif raw_control is not None:
        raise DeviceConfigError(
            f"{device_class} has device_kind={kind.value} and cannot declare control"
        )

    _validate_motion_limits(device_config)
    if _rizon_arm_count(device_config) is not None:
        if control is None:
            raise DeviceConfigError(f"{device_class} requires a control mapping")
        _validate_rizon_sections(device_config, state, control)

    return ParsedDeviceConfig(
        device_class=device_class,
        device_class_id=device_class_id,
        device_kind=kind,
        state_feedback=state,
        control=control,
        raw=dict(device_config),
    )


def normalize_device_config(device_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a structured mapping without deriving legacy fields.

    This name remains for standalone replay callers; construction uses
    :func:`parse_device_config` so the parsed contracts are passed directly.
    """
    from devices import DEVICE_CLASSES

    parsed = parse_device_config(
        device_config,
        device_kind=DEVICE_CLASSES.kind(str(device_config.get("class", ""))),
    )
    return parsed.raw


__all__ = [
    "DeviceConfigError",
    "REMOVED_DEVICE_CONFIG_FIELDS",
    "ParsedDeviceConfig",
    "parse_device_config",
    "normalize_device_config",
    "reject_removed_device_fields",
    "state_feedback_mapping",
    "state_format_mapping",
    "source_selection_config",
    "control_source_selection",
]
