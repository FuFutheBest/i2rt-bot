"""Validated configuration objects for the production Rizon adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from utils.data_format import DataFormat, DataPrimitive
from utils.lk_gripper_presets import (
    DEFAULT_LK_GRIPPER_TYPE,
    calculate_lk_gripper_max_width,
    get_lk_gripper_preset,
)
from utils.safety_bounds import SafetyBounds


@dataclass(frozen=True)
class RizonSegment:
    primitive: DataPrimitive
    start: int
    stop: int


@dataclass(frozen=True)
class RizonArmLayout:
    """One arm's ordered slice of a flattened stream."""

    segments: tuple[RizonSegment, ...]

    def find(self, *names: str) -> RizonSegment | None:
        matches = [segment for segment in self.segments if segment.primitive.name in names]
        if len(matches) > 1:
            raise ValueError(f"format contains duplicate segments for {names}")
        return matches[0] if matches else None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(segment.primitive.name for segment in self.segments)


@dataclass(frozen=True)
class RizonStateSchema:
    """Validated base arm layouts plus an optional ordered R1 extension."""

    arm_layouts: tuple[RizonArmLayout, ...]
    extension: tuple[str, ...] = ()


_R1_GRIPPER_TELEMETRY = (
    "gripper",
    "motor_angle",
    "gripper",
    "motor_angle",
)
_R1_ARM_DETAILS_WXYZ = (
    "xyz",
    "qwqxqyqz",
    "wrench",
    "xyz",
    "qwqxqyqz",
    "wrench",
)
_R1_ARM_DETAILS_XYZW = (
    "xyz",
    "qxqyqzqw",
    "wrench",
    "xyz",
    "qxqyqzqw",
    "wrench",
)
_R1_TCP_VELOCITY = ("twist", "twist")
_R1_COMMAND_GRIPPERS = ("gripper", "gripper")


def split_arm_layouts(
    data_format: DataFormat, arm_count: int, *, context: str
) -> tuple[RizonArmLayout, ...]:
    """Split a flat format into identical per-arm primitive layouts."""
    if arm_count <= 0 or data_format.dimension % arm_count:
        raise ValueError(f"{context} dimension must divide evenly across Rizon arms")
    arm_width = data_format.dimension // arm_count
    layouts: list[RizonArmLayout] = []
    primitive_index = 0
    offset = 0
    for _arm_index in range(arm_count):
        arm_start = offset
        segments: list[RizonSegment] = []
        while primitive_index < len(data_format.primitives) and offset - arm_start < arm_width:
            primitive = data_format.primitives[primitive_index]
            segments.append(
                RizonSegment(primitive=primitive, start=offset, stop=offset + primitive.width)
            )
            offset += primitive.width
            primitive_index += 1
        if offset - arm_start != arm_width:
            raise ValueError(f"{context} primitives cross a Rizon arm boundary")
        layouts.append(RizonArmLayout(tuple(segments)))

    if primitive_index != len(data_format.primitives):
        raise ValueError(f"{context} contains primitives beyond the Rizon arm layouts")
    reference = layouts[0].names
    if any(layout.names != reference for layout in layouts[1:]):
        raise ValueError(f"{context} must repeat the same primitive layout for every arm")
    return tuple(layouts)


def _validate_base_state_layouts(
    layouts: tuple[RizonArmLayout, ...],
) -> tuple[RizonArmLayout, ...]:
    for layout in layouts:
        names = layout.names
        allowed = {"j1-7", "dq1-7", "xyz", "qwqxqyqz", "qxqyqzqw", "gripper"}
        unknown = set(names) - allowed
        if unknown:
            raise ValueError(
                "state_feedback.format.type contains unsupported Rizon primitives: "
                f"{sorted(unknown)}"
            )
        if len(names) != len(set(names)):
            raise ValueError("state_feedback.format.type must not repeat a primitive per arm")
        has_xyz = "xyz" in names
        quaternion_count = int("qwqxqyqz" in names) + int("qxqyqzqw" in names)
        if has_xyz != (quaternion_count == 1):
            raise ValueError(
                "Rizon TCP state feedback requires exactly one xyz and one quaternion "
                "primitive per arm"
            )
    return layouts


def _extended_r1_state_schema(
    data_format: DataFormat, arm_count: int
) -> RizonStateSchema:
    if arm_count != 2:
        raise ValueError("R1 state feedback extensions require exactly two Rizon arms")

    primitives = data_format.primitives
    base_primitive_count = 10
    if len(primitives) < base_primitive_count:
        raise ValueError("state_feedback.format is missing the dual-arm base state")

    base_names = tuple(
        primitive.name for primitive in primitives[:base_primitive_count]
    )
    quaternion_name = base_names[3]
    per_arm_names = (
        "j1-7",
        "dq1-7",
        "xyz",
        quaternion_name,
        "gripper",
    )
    if quaternion_name not in {"qwqxqyqz", "qxqyqzqw"} or base_names != (
        *per_arm_names,
        *per_arm_names,
    ):
        raise ValueError(
            "extended Rizon state_feedback.format must start with identical "
            "[j1-7, dq1-7, xyz, quaternion, gripper] layouts for left and right"
        )

    extension = tuple(
        primitive.name for primitive in primitives[base_primitive_count:]
    )
    arm_details = (
        _R1_ARM_DETAILS_WXYZ
        if quaternion_name == "qwqxqyqz"
        else _R1_ARM_DETAILS_XYZW
    )
    supported_extensions = {
        _R1_GRIPPER_TELEMETRY,
        _R1_GRIPPER_TELEMETRY + arm_details,
        _R1_GRIPPER_TELEMETRY + arm_details + _R1_TCP_VELOCITY,
        _R1_GRIPPER_TELEMETRY
        + arm_details
        + _R1_TCP_VELOCITY
        + _R1_COMMAND_GRIPPERS,
    }
    if extension not in supported_extensions:
        raise ValueError(
            "unsupported R1 state_feedback.format extension; expected ordered "
            "gripper telemetry, optional commanded TCP/wrench, optional TCP "
            "twist, and optional commanded grippers"
        )

    layouts: list[RizonArmLayout] = []
    offset = 0
    for arm_index in range(arm_count):
        segments: list[RizonSegment] = []
        start = arm_index * len(per_arm_names)
        for primitive in primitives[start : start + len(per_arm_names)]:
            segments.append(
                RizonSegment(
                    primitive=primitive,
                    start=offset,
                    stop=offset + primitive.width,
                )
            )
            offset += primitive.width
        layouts.append(RizonArmLayout(tuple(segments)))
    return RizonStateSchema(
        arm_layouts=_validate_base_state_layouts(tuple(layouts)),
        extension=extension,
    )


def validate_state_schema(data_format: DataFormat, arm_count: int) -> RizonStateSchema:
    """Validate a Rizon feedback format without deriving compatibility fields."""
    try:
        layouts = split_arm_layouts(
            data_format, arm_count, context="state_feedback.format"
        )
        return RizonStateSchema(_validate_base_state_layouts(layouts))
    except ValueError as base_error:
        if arm_count != 2 or len(data_format.primitives) <= 10:
            raise
        try:
            return _extended_r1_state_schema(data_format, arm_count)
        except ValueError as extension_error:
            raise ValueError(str(extension_error)) from base_error


def validate_state_layouts(
    data_format: DataFormat, arm_count: int
) -> tuple[RizonArmLayout, ...]:
    """Return per-arm base layouts for callers that do not need extensions."""
    return validate_state_schema(data_format, arm_count).arm_layouts


def validate_control_layouts(
    control_type: str, data_format: DataFormat, arm_count: int
) -> tuple[RizonArmLayout, ...]:
    if control_type not in {"joints", "eef_abs", "eef_rel"}:
        raise ValueError(f"control.type={control_type} is not supported by RizonRobot")
    layouts = split_arm_layouts(data_format, arm_count, context="control.format")
    for layout in layouts:
        names = layout.names
        if control_type == "joints":
            if names != ("j1-7",):
                raise ValueError(
                    "control.type=joints requires control.format.type "
                    "to contain [j1-7] per arm"
                )
            continue
        pose_names = tuple(name for name in names if name != "gripper")
        if (
            len(names) != len(set(names))
            or len(pose_names) != 2
            or set(pose_names) not in (
                {"xyz", "qwqxqyqz"},
                {"xyz", "qxqyqzqw"},
            )
        ):
            raise ValueError(
                f"control.type={control_type} requires one xyz and one quaternion "
                "primitive per arm, with an optional gripper primitive"
            )
    return layouts


def _per_arm(value: Any, arm_count: int, *, field: str) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != arm_count:
            raise ValueError(f"{field} must contain one value per arm")
        return values
    return [value for _ in range(arm_count)]


@dataclass
class RizonGripperArmConfig:
    enabled: bool
    port: str | None
    motor_id: int
    baudrate: int
    speed_dps: float
    home_width: float
    home_ratio: float | None
    timeout: float
    wait_timeout: float
    auto_calibrate: bool
    calibration_directory: str | None
    calibration_file: str | None
    min_torque_iq: int | None
    max_torque_iq: int | None
    calibration_timeout: float
    calibration_settle_time: float
    min_encoder: int | None
    max_encoder: int | None
    min_angle_deg: float | None
    max_angle_deg: float | None
    arm_length_m: float | None
    apex_angle_max_deg: float | None
    gear_ratio: float
    min_vel: float
    max_vel: float
    min_width: float | None
    max_width: float | None
    force_limit: float
    scale_factor: float
    command_offset: float


@dataclass
class RizonGripperConfig:
    backend: str
    name: str
    gripper_type: str
    tool_name: str
    value_mode: str
    command_value_mode: str
    inference_calibration_path: str | None
    arms: tuple[RizonGripperArmConfig, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        gripper_required_by_arm: Sequence[bool],
    ) -> "RizonGripperConfig":
        raw = dict(value or {})
        allowed = {
            "backend",
            "name",
            "gripper_type",
            "type",
            "tool_name",
            "value_mode",
            "command_value_mode",
            "inference_calibration_path",
            "port",
            "motor_id",
            "baudrate",
            "speed_dps",
            "home_width",
            "home_ratio",
            "timeout",
            "wait_timeout",
            "auto_calibrate",
            "calibration_directory",
            "calibration_files",
            "min_torque_iq",
            "max_torque_iq",
            "calibration_timeout",
            "calibration_settle_time",
            "min_encoder",
            "max_encoder",
            "min_angle_deg",
            "max_angle_deg",
            "arm_length_m",
            "apex_angle_max_deg",
            "gear_ratio",
            "min_vel",
            "max_vel",
            "min_width",
            "max_width",
            "force_limit",
            "scale_factor",
            "command_offset",
            "visual",
            "visual_mount_transforms",
        }
        unknown = set(raw) - allowed - {"enabled"}
        if unknown:
            raise ValueError(
                "gripper contains unknown fields: " + ", ".join(sorted(unknown))
            )
        if "enabled" in raw:
            raise ValueError(
                "gripper.enabled is redundant; add or remove gripper primitives in "
                "state_feedback.format.type/control.format.type"
            )
        arm_count = len(gripper_required_by_arm)
        backend = str(raw.get("backend", "flexiv")).strip().casefold()
        if backend not in {"flexiv", "lktech"}:
            raise ValueError("gripper.backend must be flexiv or lktech")
        gripper_type = str(
            raw.get("gripper_type", raw.get("type", DEFAULT_LK_GRIPPER_TYPE))
        ).strip().casefold()
        preset = get_lk_gripper_preset(gripper_type) if backend == "lktech" else {}
        value_mode = str(
            raw.get("value_mode", "normalized" if backend == "lktech" else "width_m")
        ).strip().casefold()
        if value_mode not in {"normalized", "width_m"}:
            raise ValueError("gripper.value_mode must be normalized or width_m")
        command_value_mode = str(
            raw.get("command_value_mode", value_mode)
        ).strip().casefold()
        if command_value_mode not in {"normalized", "width_m"}:
            raise ValueError(
                "gripper.command_value_mode must be normalized or width_m"
            )

        defaults = {
            "port": None,
            "motor_id": 1,
            "baudrate": 115200,
            "speed_dps": 180.0,
            "home_width": None if backend == "lktech" else 0.085,
            "home_ratio": None,
            "timeout": 0.1,
            "wait_timeout": 15.0,
            "auto_calibrate": False,
            "calibration_directory": None,
            "calibration_files": None,
            "min_torque_iq": preset.get("min_torque_iq"),
            "max_torque_iq": preset.get("max_torque_iq"),
            "calibration_timeout": 15.0,
            "calibration_settle_time": 0.24,
            "min_encoder": 13363,
            "max_encoder": 32819,
            "min_angle_deg": None,
            "max_angle_deg": None,
            "arm_length_m": preset.get("arm_length_m"),
            "apex_angle_max_deg": preset.get("apex_angle_max_deg"),
            "gear_ratio": 1.0,
            "min_vel": 0.0,
            "max_vel": 0.1,
            "min_width": 0.0,
            "max_width": None,
            "force_limit": 100.0,
            "scale_factor": 1.0,
            "command_offset": 0.0,
        }
        expanded = {
            key: _per_arm(raw.get(key, default), arm_count, field=f"gripper.{key}")
            for key, default in defaults.items()
        }
        arms: list[RizonGripperArmConfig] = []
        for index, gripper_required in enumerate(gripper_required_by_arm):
            ratio = expanded["home_ratio"][index]
            scale_factor = float(expanded["scale_factor"][index])
            min_width = (
                0.0
                if expanded["min_width"][index] is None
                else float(expanded["min_width"][index])
            )
            max_width = expanded["max_width"][index]
            if max_width is None:
                max_width = (
                    calculate_lk_gripper_max_width(
                        expanded["arm_length_m"][index],
                        expanded["apex_angle_max_deg"][index],
                        0.15,
                    )
                    if backend == "lktech"
                    else max(0.085, float(expanded["home_width"][index] or 0.085))
                )
            home_width = expanded["home_width"][index]
            if home_width is None:
                home_width = max_width
            home_width = float(home_width)
            if not min_width <= home_width <= float(max_width):
                raise ValueError(
                    f"gripper.home_width[{index}] must be within the configured "
                    f"width range [{min_width}, {float(max_width)}]"
                )
            if ratio is not None and not 0.0 <= float(ratio) <= 1.0:
                raise ValueError(f"gripper.home_ratio[{index}] must be within [0, 1]")
            if not 0.0 <= scale_factor <= 1.0:
                raise ValueError(f"gripper.scale_factor[{index}] must be within [0, 1]")
            port = expanded["port"][index]
            if gripper_required and backend == "lktech" and not str(port or "").strip():
                raise ValueError(f"gripper.port[{index}] is required for LKTech")
            arms.append(
                RizonGripperArmConfig(
                    enabled=bool(gripper_required),
                    port=None if port is None else str(port),
                    motor_id=int(expanded["motor_id"][index]),
                    baudrate=int(expanded["baudrate"][index]),
                    speed_dps=float(expanded["speed_dps"][index]),
                    home_width=home_width,
                    home_ratio=None if ratio is None else float(ratio),
                    timeout=float(expanded["timeout"][index]),
                    wait_timeout=float(expanded["wait_timeout"][index]),
                    auto_calibrate=bool(expanded["auto_calibrate"][index]),
                    calibration_directory=(
                        None
                        if expanded["calibration_directory"][index] is None
                        else str(expanded["calibration_directory"][index])
                    ),
                    calibration_file=expanded["calibration_files"][index],
                    min_torque_iq=expanded["min_torque_iq"][index],
                    max_torque_iq=expanded["max_torque_iq"][index],
                    calibration_timeout=float(expanded["calibration_timeout"][index]),
                    calibration_settle_time=float(
                        expanded["calibration_settle_time"][index]
                    ),
                    min_encoder=expanded["min_encoder"][index],
                    max_encoder=expanded["max_encoder"][index],
                    min_angle_deg=expanded["min_angle_deg"][index],
                    max_angle_deg=expanded["max_angle_deg"][index],
                    arm_length_m=expanded["arm_length_m"][index],
                    apex_angle_max_deg=expanded["apex_angle_max_deg"][index],
                    gear_ratio=float(expanded["gear_ratio"][index]),
                    min_vel=float(expanded["min_vel"][index]),
                    max_vel=float(expanded["max_vel"][index]),
                    min_width=min_width,
                    max_width=float(max_width),
                    force_limit=float(expanded["force_limit"][index]),
                    scale_factor=scale_factor,
                    command_offset=float(expanded["command_offset"][index]),
                )
            )
        name = str(raw.get("name", "")).strip()
        if any(arm.enabled for arm in arms) and backend == "flexiv" and not name:
            raise ValueError("gripper.name is required for the Flexiv gripper backend")
        return cls(
            backend=backend,
            name=name,
            gripper_type=gripper_type,
            tool_name=str(raw.get("tool_name", name)),
            value_mode=value_mode,
            command_value_mode=command_value_mode,
            inference_calibration_path=raw.get("inference_calibration_path"),
            arms=tuple(arms),
        )


@dataclass(frozen=True)
class RizonMotionLimits:
    joint_velocity: tuple[float, ...]
    joint_acceleration: tuple[float, ...]
    tcp_linear_velocity: float
    tcp_linear_acceleration: float
    tcp_angular_velocity: float
    tcp_angular_acceleration: float
    max_tcp_contact_wrench: tuple[float, ...] | None
    workspace: SafetyBounds

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None, *, arm_count: int
    ) -> "RizonMotionLimits":
        raw = dict(value or {})
        unknown = set(raw) - {
            "velocity",
            "acceleration",
            "max_tcp_contact_wrench",
            "workspace",
        }
        if unknown:
            raise ValueError(
                "motion_limits contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        raw_velocity = raw.get("velocity", {})
        raw_acceleration = raw.get("acceleration", {})
        if not isinstance(raw_velocity, Mapping):
            raise ValueError("motion_limits.velocity must be a mapping")
        if not isinstance(raw_acceleration, Mapping):
            raise ValueError("motion_limits.acceleration must be a mapping")
        velocity = dict(raw_velocity)
        acceleration = dict(raw_acceleration)
        unknown_velocity = set(velocity) - {
            "joints_rad_s",
            "tcp_linear_m_s",
            "tcp_angular_rad_s",
        }
        if unknown_velocity:
            raise ValueError(
                "motion_limits.velocity contains unknown fields: "
                + ", ".join(sorted(unknown_velocity))
            )
        unknown_acceleration = set(acceleration) - {
            "joints_rad_s2",
            "tcp_linear_m_s2",
            "tcp_angular_rad_s2",
        }
        if unknown_acceleration:
            raise ValueError(
                "motion_limits.acceleration contains unknown fields: "
                + ", ".join(sorted(unknown_acceleration))
            )

        def joint_values(raw_value: Any, default: float, field: str) -> tuple[float, ...]:
            if raw_value is None:
                return (default,) * (arm_count * 7)
            if isinstance(raw_value, Mapping):
                values = [*raw_value.get("left", ()), *raw_value.get("right", ())]
            else:
                values = list(raw_value)
            if len(values) != arm_count * 7:
                raise ValueError(f"{field} must contain 7 values per arm")
            normalized = tuple(float(item) for item in values)
            if any(item <= 0.0 for item in normalized):
                raise ValueError(f"{field} values must be positive")
            return normalized

        wrench = raw.get("max_tcp_contact_wrench")
        normalized_wrench = None if wrench is None else tuple(float(item) for item in wrench)
        if normalized_wrench is not None and (
            len(normalized_wrench) != 6 or any(item <= 0.0 for item in normalized_wrench)
        ):
            raise ValueError(
                "motion_limits.max_tcp_contact_wrench must contain 6 positive values"
            )
        tcp_limit_fields = {
            "tcp_linear_velocity": (
                float(velocity.get("tcp_linear_m_s", 1.0)),
                "motion_limits.velocity.tcp_linear_m_s",
            ),
            "tcp_linear_acceleration": (
                float(acceleration.get("tcp_linear_m_s2", 2.0)),
                "motion_limits.acceleration.tcp_linear_m_s2",
            ),
            "tcp_angular_velocity": (
                float(velocity.get("tcp_angular_rad_s", 2.0)),
                "motion_limits.velocity.tcp_angular_rad_s",
            ),
            "tcp_angular_acceleration": (
                float(acceleration.get("tcp_angular_rad_s2", 5.0)),
                "motion_limits.acceleration.tcp_angular_rad_s2",
            ),
        }
        for number, config_field in tcp_limit_fields.values():
            if number <= 0.0:
                raise ValueError(f"{config_field} must be positive")
        tcp_limits = {
            field: number for field, (number, _config_field) in tcp_limit_fields.items()
        }

        return cls(
            joint_velocity=joint_values(
                velocity.get("joints_rad_s"),
                1.0,
                "motion_limits.velocity.joints_rad_s",
            ),
            joint_acceleration=joint_values(
                acceleration.get("joints_rad_s2"),
                2.0,
                "motion_limits.acceleration.joints_rad_s2",
            ),
            **tcp_limits,
            max_tcp_contact_wrench=normalized_wrench,
            workspace=SafetyBounds.from_config(raw.get("workspace")),
        )


__all__ = [
    "RizonArmLayout",
    "RizonGripperConfig",
    "RizonMotionLimits",
    "RizonStateSchema",
    "validate_control_layouts",
    "validate_state_layouts",
    "validate_state_schema",
]
