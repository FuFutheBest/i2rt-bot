"""Configuration contract for configurable I2RT bimanual teleoperation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SIDES = ("left", "right")
GRIPPER_BACKENDS = {"i2rt_can", "lktech", "none"}
I2RT_GRIPPER_TYPES = {"crank_4310", "linear_3507", "linear_4310"}


@dataclass(frozen=True)
class GripperConfig:
    backend: str
    i2rt_type: str = "linear_4310"
    port: str | None = None
    motor_id: int = 1
    baudrate: int = 115200
    speed_dps: float = 180.0
    command_velocity: float = 0.5
    calibration_file: Path | None = None
    gripper_type: str = "soft"
    min_width_m: float = 0.0
    max_width_m: float | None = None
    startup_width_ratio: float | None = None
    scale_factor: float = 1.0
    command_offset: float = 0.0
    invert: bool = False
    command_hz: float = 20.0
    feedback_hz: float = 20.0
    deadband: float = 0.002


@dataclass(frozen=True)
class EndpointConfig:
    can_channel: str
    rpc_port: int


@dataclass(frozen=True)
class FollowerConfig(EndpointConfig):
    gripper: GripperConfig
    joint_directions: tuple[float, ...] = (1.0,) * 7


@dataclass(frozen=True)
class ArmConfig:
    leader: EndpointConfig
    follower: FollowerConfig


@dataclass(frozen=True)
class ControlConfig:
    frequency_hz: float = 100.0
    alignment_duration_s: float = 3.0
    alignment_max_velocity_rad_s: float = 0.35
    alignment_settle_timeout_s: float = 3.0
    alignment_tolerance: float = 0.08
    alignment_stable_samples: int = 3
    startup_timeout_s: float = 20.0
    rpc_timeout_s: float = 2.0
    cycle_rpc_timeout_s: float = 0.10
    command_frame_max_age_s: float = 0.05
    command_lease_timeout_s: float = 0.15
    feedback_timeout_s: float = 0.10
    watchdog_check_interval_s: float = 0.005
    following_error_fault_threshold: float = 1.0
    following_error_fault_samples: int = 3
    enter_chord_window_s: float = 0.30
    button_debounce_samples: int = 3
    bilateral_kp: float = 0.0
    bilateral_kd: float = 0.0
    leader_feedback: bool = True


@dataclass(frozen=True)
class TeleoperationConfig:
    control: ControlConfig
    arms: Mapping[str, ArmConfig]
    episode_signal_pattern: str = "/tmp/superinference_episode_{signal}"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be within [{minimum}, {maximum}]")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def _resolve_repo_path(value: Any, field: str) -> Path:
    raw_path = Path(_string(value, field)).expanduser()
    return raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path


def _validate_calibration_file(path: Path, field: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid JSON: {path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{field} must contain a JSON object")
    has_angles = document.get("min_angle_deg") is not None and document.get(
        "max_angle_deg"
    ) is not None
    has_encoders = document.get("min_encoder") is not None and document.get(
        "max_encoder"
    ) is not None
    if not (has_angles or has_encoders):
        raise ValueError(
            f"{field} must define min/max angle or min/max encoder calibration"
        )


def _parse_gripper(raw: Any, field: str) -> GripperConfig:
    data = _mapping(raw, field)
    backend = _string(data.get("backend"), f"{field}.backend").lower()
    if backend not in GRIPPER_BACKENDS:
        choices = ", ".join(sorted(GRIPPER_BACKENDS))
        raise ValueError(f"{field}.backend must be one of: {choices}")

    i2rt_type = _string(
        data.get("type", "linear_4310"), f"{field}.type"
    ).lower()
    if backend == "i2rt_can" and i2rt_type not in I2RT_GRIPPER_TYPES:
        choices = ", ".join(sorted(I2RT_GRIPPER_TYPES))
        raise ValueError(f"{field}.type must be one of: {choices}")

    port = None
    calibration_file = None
    if backend == "lktech":
        port = _string(data.get("port"), f"{field}.port")
        calibration_file = _resolve_repo_path(
            data.get("calibration_file"), f"{field}.calibration_file"
        )
        _validate_calibration_file(calibration_file, f"{field}.calibration_file")

    max_width_raw = data.get("max_width_m")
    max_width_m = (
        None
        if max_width_raw is None
        else _number(max_width_raw, f"{field}.max_width_m", minimum=0.0)
    )
    min_width_m = _number(
        data.get("min_width_m", 0.0), f"{field}.min_width_m", minimum=0.0
    )
    if max_width_m is not None and max_width_m <= min_width_m:
        raise ValueError(f"{field}.max_width_m must be greater than min_width_m")

    startup_width_ratio_raw = data.get("startup_width_ratio")
    startup_width_ratio = (
        None
        if startup_width_ratio_raw is None
        else _number(
            startup_width_ratio_raw,
            f"{field}.startup_width_ratio",
            minimum=0.0,
            maximum=1.0,
        )
    )

    return GripperConfig(
        backend=backend,
        i2rt_type=i2rt_type,
        port=port,
        motor_id=_integer(
            data.get("motor_id", 1), f"{field}.motor_id", minimum=1, maximum=32
        ),
        baudrate=_integer(
            data.get("baudrate", 115200),
            f"{field}.baudrate",
            minimum=1,
            maximum=10_000_000,
        ),
        speed_dps=_number(
            data.get("speed_dps", 180.0), f"{field}.speed_dps", minimum=0.001
        ),
        command_velocity=_number(
            data.get("command_velocity", 0.5),
            f"{field}.command_velocity",
            minimum=0.001,
        ),
        calibration_file=calibration_file,
        gripper_type=_string(
            data.get("gripper_type", "soft"), f"{field}.gripper_type"
        ).lower(),
        min_width_m=min_width_m,
        max_width_m=max_width_m,
        startup_width_ratio=startup_width_ratio,
        scale_factor=_number(
            data.get("scale_factor", 1.0),
            f"{field}.scale_factor",
            minimum=0.001,
            maximum=1.0,
        ),
        command_offset=_number(
            data.get("command_offset", 0.0),
            f"{field}.command_offset",
            minimum=0.0,
            maximum=1.0,
        ),
        invert=_boolean(data.get("invert", False), f"{field}.invert"),
        command_hz=_number(
            data.get("command_hz", 20.0), f"{field}.command_hz", minimum=0.1
        ),
        feedback_hz=_number(
            data.get("feedback_hz", 20.0), f"{field}.feedback_hz", minimum=0.1
        ),
        deadband=_number(
            data.get("deadband", 0.002),
            f"{field}.deadband",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _parse_endpoint(raw: Any, field: str) -> EndpointConfig:
    data = _mapping(raw, field)
    return EndpointConfig(
        can_channel=_string(data.get("can_channel"), f"{field}.can_channel"),
        rpc_port=_integer(
            data.get("rpc_port"), f"{field}.rpc_port", minimum=1, maximum=65535
        ),
    )


def _joint_directions(raw: Any, field: str) -> tuple[float, ...]:
    if raw is None:
        return (1.0,) * 7
    if not isinstance(raw, (list, tuple)) or len(raw) != 7:
        raise ValueError(f"{field} must contain exactly seven values")
    result = tuple(_number(value, f"{field}[{index}]") for index, value in enumerate(raw))
    if any(value not in (-1.0, 1.0) for value in result):
        raise ValueError(f"{field} values must be either -1 or 1")
    return result


def _parse_control(raw: Any) -> ControlConfig:
    data = _mapping(raw, "control")
    return ControlConfig(
        frequency_hz=_number(
            data.get("frequency_hz", 100.0),
            "control.frequency_hz",
            minimum=1.0,
        ),
        alignment_duration_s=_number(
            data.get("alignment_duration_s", 3.0),
            "control.alignment_duration_s",
            minimum=0.0,
        ),
        alignment_max_velocity_rad_s=_number(
            data.get("alignment_max_velocity_rad_s", 0.35),
            "control.alignment_max_velocity_rad_s",
            minimum=0.01,
        ),
        alignment_settle_timeout_s=_number(
            data.get("alignment_settle_timeout_s", 3.0),
            "control.alignment_settle_timeout_s",
            minimum=0.0,
        ),
        alignment_tolerance=_number(
            data.get("alignment_tolerance", 0.08),
            "control.alignment_tolerance",
            minimum=0.001,
        ),
        alignment_stable_samples=_integer(
            data.get("alignment_stable_samples", 3),
            "control.alignment_stable_samples",
            minimum=1,
            maximum=1000,
        ),
        startup_timeout_s=_number(
            data.get("startup_timeout_s", 20.0),
            "control.startup_timeout_s",
            minimum=0.1,
        ),
        rpc_timeout_s=_number(
            data.get("rpc_timeout_s", 2.0),
            "control.rpc_timeout_s",
            minimum=0.1,
        ),
        cycle_rpc_timeout_s=_number(
            data.get("cycle_rpc_timeout_s", 0.10),
            "control.cycle_rpc_timeout_s",
            minimum=0.001,
        ),
        command_frame_max_age_s=_number(
            data.get("command_frame_max_age_s", 0.05),
            "control.command_frame_max_age_s",
            minimum=0.001,
        ),
        command_lease_timeout_s=_number(
            data.get("command_lease_timeout_s", 0.15),
            "control.command_lease_timeout_s",
            minimum=0.001,
        ),
        feedback_timeout_s=_number(
            data.get("feedback_timeout_s", 0.10),
            "control.feedback_timeout_s",
            minimum=0.001,
        ),
        watchdog_check_interval_s=_number(
            data.get("watchdog_check_interval_s", 0.005),
            "control.watchdog_check_interval_s",
            minimum=0.001,
        ),
        following_error_fault_threshold=_number(
            data.get("following_error_fault_threshold", 1.0),
            "control.following_error_fault_threshold",
            minimum=0.001,
        ),
        following_error_fault_samples=_integer(
            data.get("following_error_fault_samples", 3),
            "control.following_error_fault_samples",
            minimum=1,
            maximum=1000,
        ),
        enter_chord_window_s=_number(
            data.get("enter_chord_window_s", 0.30),
            "control.enter_chord_window_s",
            minimum=0.01,
        ),
        button_debounce_samples=_integer(
            data.get("button_debounce_samples", 3),
            "control.button_debounce_samples",
            minimum=1,
            maximum=100,
        ),
        bilateral_kp=_number(
            data.get("bilateral_kp", 0.0),
            "control.bilateral_kp",
            minimum=0.0,
        ),
        bilateral_kd=_number(
            data.get("bilateral_kd", 0.0),
            "control.bilateral_kd",
            minimum=0.0,
        ),
        leader_feedback=_boolean(
            data.get("leader_feedback", True), "control.leader_feedback"
        ),
    )


def load_teleoperation_config(path: str | Path) -> TeleoperationConfig:
    """Load and validate a bimanual teleoperation YAML file."""

    config_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"Cannot read teleoperation config: {config_path}") from exc
    root = _mapping(document, "config")
    version = root.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported teleoperation config version: {version}")
    episode_signal_pattern = _string(
        root.get(
            "episode_signal_pattern", "/tmp/superinference_episode_{signal}"
        ),
        "episode_signal_pattern",
    )
    if "{signal}" not in episode_signal_pattern:
        raise ValueError("episode_signal_pattern must contain {signal}")

    raw_arms = _mapping(root.get("arms"), "arms")
    if set(raw_arms) != set(SIDES):
        raise ValueError("arms must contain exactly 'left' and 'right'")

    arms: dict[str, ArmConfig] = {}
    for side in SIDES:
        arm_data = _mapping(raw_arms[side], f"arms.{side}")
        leader = _parse_endpoint(arm_data.get("leader"), f"arms.{side}.leader")
        follower_data = _mapping(
            arm_data.get("follower"), f"arms.{side}.follower"
        )
        follower_endpoint = _parse_endpoint(
            follower_data, f"arms.{side}.follower"
        )
        follower = FollowerConfig(
            can_channel=follower_endpoint.can_channel,
            rpc_port=follower_endpoint.rpc_port,
            gripper=_parse_gripper(
                follower_data.get("gripper"), f"arms.{side}.follower.gripper"
            ),
            joint_directions=_joint_directions(
                follower_data.get("joint_directions"),
                f"arms.{side}.follower.joint_directions",
            ),
        )
        arms[side] = ArmConfig(leader=leader, follower=follower)

    rpc_ports = [
        endpoint.rpc_port
        for arm in arms.values()
        for endpoint in (arm.leader, arm.follower)
    ]
    if len(set(rpc_ports)) != len(rpc_ports):
        raise ValueError("All leader and follower rpc_port values must be unique")

    can_channels = [
        endpoint.can_channel
        for arm in arms.values()
        for endpoint in (arm.leader, arm.follower)
    ]
    if len(set(can_channels)) != len(can_channels):
        raise ValueError("All leader and follower can_channel values must be unique")

    lktech_ports = [
        arm.follower.gripper.port
        for arm in arms.values()
        if arm.follower.gripper.backend == "lktech"
    ]
    if len(set(lktech_ports)) != len(lktech_ports):
        raise ValueError("Each LKTech gripper must use a unique serial port")

    return TeleoperationConfig(
        control=_parse_control(root.get("control", {})),
        arms=arms,
        episode_signal_pattern=episode_signal_pattern,
    )
