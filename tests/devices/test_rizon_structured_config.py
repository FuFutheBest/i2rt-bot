"""Hardware-free contracts for structured Rizon configuration and layouts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import devices.robots.rizon.robot as rizon_robot_module
from devices.robots.base import BaseRobot
from devices.robots.config import RobotControlConfig
from devices.robots.rizon import RizonRobot
from devices.robots.rizon.commands import RizonCommandMixin
from devices.robots.rizon.config import (
    RizonGripperConfig,
    RizonMotionLimits,
    validate_control_layouts,
    validate_state_layouts,
    validate_state_schema,
)
from devices.robots.rizon.gripper import RizonGripperMixin
from utils.data_format import parse_data_format
from utils.device_config import normalize_device_config


def _format(shape: list[int], primitives: list[str]) -> dict[str, object]:
    return {"shape": shape, "type": primitives, "dtype": "float64"}


def test_rizon_constructor_exposes_only_structured_stream_sections() -> None:
    parameters = inspect.signature(RizonRobot.__init__).parameters
    assert {"state_feedback", "control", "gripper", "motion_limits"} <= set(
        parameters
    )
    assert {
        "data_shape",
        "control_shape",
        "using_tcp_pose",
        "mirror_mode",
        "utilize_gripper",
        "gripper_port",
        "max_linear_vel",
        "airexo_type",
    }.isdisjoint(parameters)


def test_base_robot_resolves_grouped_stream_settings() -> None:
    robot = BaseRobot(
        device_class_id=3,
        state_feedback={
            "fps": 120.0,
            "buffer_size": 321,
            "hardware_latency_ms": 4.0,
            "format": _format([14], ["j1-7", "dq1-7"]),
        },
        control={
            "type": "joints",
            "buffer_size": 17,
            "execution": {"mode": "legacy_async", "fps": 40.0},
            "format": _format([7], ["j1-7"]),
        },
    )
    assert robot.data_shape == (14,)
    assert robot.data_dtype == "float64"
    assert robot.fps == 120.0
    assert robot.buffer_size == 321
    assert robot.control_shape == (7,)
    assert robot.control_dtype == "float64"
    assert robot.control_buffer_size == 17
    assert robot.command_fps == 40.0
    assert robot.legacy_async_mode is True


@pytest.mark.parametrize(
    "old_name", ["joint_position", "tcp_pose_absolute", "eef_rel_from_home"]
)
def test_removed_control_types_are_rejected(old_name: str) -> None:
    with pytest.raises(ValueError, match="control.type must be one of"):
        RobotControlConfig.from_mapping(
            {
                "type": old_name,
                "format": _format([7], ["j1-7"]),
            }
        )


def test_rizon_initializes_from_structured_contract_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rizon_robot_module,
        "FlexivAdapter",
        lambda **_kwargs: SimpleNamespace(),
    )
    robot = RizonRobot(
        device_class_id=2,
        robot_sn=["left", "right"],
        state_feedback={
            "fps": 120.0,
            "buffer_size": 100,
            "hardware_latency_ms": 0.0,
            "format": _format(
                [42],
                [
                    "j1-7",
                    "dq1-7",
                    "xyz",
                    "qwqxqyqz",
                    "j1-7",
                    "dq1-7",
                    "xyz",
                    "qwqxqyqz",
                ],
            ),
        },
        control={
            "type": "eef_abs",
            "format": _format(
                [14],
                ["xyz", "qxqyqzqw", "xyz", "qxqyqzqw"],
            ),
        },
        signal_files={},
    )

    assert robot.control_type == "eef_abs"
    assert robot.utilize_gripper is False
    assert robot.data_shape == (42,)
    assert robot.control_shape == (14,)
    assert robot.command_period_s == pytest.approx(1.0 / robot.command_fps)
    assert not hasattr(robot, "delta_time")


def test_rizon_missing_sdk_can_clean_up_partial_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial_robots: list[RizonRobot] = []
    errors: list[str] = []
    original_base_init = BaseRobot.__init__

    def capture_base_init(instance, *args, **kwargs) -> None:
        original_base_init(instance, *args, **kwargs)
        partial_robots.append(instance)

    def missing_adapter(*_args, **_kwargs) -> None:
        raise ImportError("backend=python but flexivrdk is not importable")

    monkeypatch.setattr(BaseRobot, "__init__", capture_base_init)
    monkeypatch.setattr(rizon_robot_module, "FlexivAdapter", missing_adapter)
    monkeypatch.setattr(
        rizon_robot_module,
        "logger",
        SimpleNamespace(error=errors.append, info=lambda *_args, **_kwargs: None),
    )

    with pytest.raises(ImportError, match="flexivrdk is not importable"):
        RizonRobot(
            device_class_id=2,
            robot_sn="left",
            state_feedback={
                "fps": 120.0,
                "buffer_size": 100,
                "hardware_latency_ms": 0.0,
                "format": _format([21], ["j1-7", "dq1-7", "xyz", "qwqxqyqz"]),
            },
            control={
                "type": "joints",
                "format": _format([7], ["j1-7"]),
            },
            signal_files={},
        )

    partial_robots.pop().stop_server()
    assert errors == []


@pytest.mark.parametrize(
    ("control_type", "primitives", "message"),
    [
        ("joints", ["xyz", "qxqyqzqw"], "requires control.format"),
        ("eef_abs", ["j1-7"], "requires one xyz"),
    ],
)
def test_control_type_must_match_format_primitives(
    control_type: str, primitives: list[str], message: str
) -> None:
    data_format = parse_data_format(
        _format(
            [
                sum(
                    7 if item == "j1-7" else 3 if item == "xyz" else 4
                    for item in primitives
                )
            ],
            primitives,
        ),
        require_dtype=True,
    )
    with pytest.raises(ValueError, match=message):
        validate_control_layouts(control_type, data_format, 1)


class _StateHarness(RizonCommandMixin, RizonGripperMixin):
    def _check_operational_status_change(self) -> None:
        return None


def test_state_feedback_order_and_quaternion_order_come_from_format() -> None:
    harness = _StateHarness()
    state_format = parse_data_format(
        _format([21], ["xyz", "qxqyqzqw", "j1-7", "dq1-7"]),
        require_dtype=True,
    )
    harness.state_layouts = validate_state_layouts(state_format, 1)
    harness.data_shape = (21,)
    harness.data_dtype = "float64"
    harness.is_operational = True
    harness.is_dual_arm = False
    harness.gripper = None
    harness.gripper_config = RizonGripperConfig.from_mapping(
        None, gripper_required_by_arm=[False]
    )
    harness.robot = SimpleNamespace(
        states=lambda: SimpleNamespace(
            q=np.arange(7, dtype=np.float64),
            dq=np.arange(10, 17, dtype=np.float64),
            tcp_pose=np.array([1.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3]),
        )
    )

    state = harness.get_robot_state()
    np.testing.assert_allclose(
        state,
        [
            1.0,
            2.0,
            3.0,
            0.1,
            0.2,
            0.3,
            0.5,
            *range(7),
            *range(10, 17),
        ],
    )


def test_rizon_rejects_removed_home_relative_control_type() -> None:
    data_format = parse_data_format(
        _format([7], ["xyz", "qxqyqzqw"]), require_dtype=True
    )

    with pytest.raises(ValueError, match="not supported by RizonRobot"):
        validate_control_layouts("eef_rel_from_home", data_format, 1)


def test_r1_extended_state_is_validated_and_serialized_from_format() -> None:
    primitives = [
        "j1-7",
        "dq1-7",
        "xyz",
        "qwqxqyqz",
        "gripper",
        "j1-7",
        "dq1-7",
        "xyz",
        "qwqxqyqz",
        "gripper",
        "gripper",
        "motor_angle",
        "gripper",
        "motor_angle",
        "xyz",
        "qwqxqyqz",
        "wrench",
        "xyz",
        "qwqxqyqz",
        "wrench",
        "twist",
        "twist",
        "gripper",
        "gripper",
    ]
    state_format = parse_data_format(_format([88], primitives), require_dtype=True)
    harness = _StateHarness()
    harness.state_schema = validate_state_schema(state_format, 2)
    harness.state_layouts = harness.state_schema.arm_layouts
    harness.data_shape = (88,)
    harness.data_dtype = "float64"
    harness.is_operational = True
    harness.is_dual_arm = True
    harness.gripper_calibration_data = None
    harness.gripper_config = RizonGripperConfig.from_mapping(
        {"backend": "flexiv", "name": "test_gripper"},
        gripper_required_by_arm=[True, True],
    )
    harness.last_gripper_value = [0.0, 0.0]
    harness.last_gripper_command_width = [0.03, 0.04]

    def robot(offset: float, width: float):
        states = SimpleNamespace(
            q=np.arange(7, dtype=np.float64) + offset,
            dq=np.arange(7, dtype=np.float64) + offset + 10.0,
            tcp_pose=np.array(
                [offset + 1.0, offset + 2.0, offset + 3.0, 1.0, 0.0, 0.0, 0.0]
            ),
            ext_wrench_in_tcp=np.arange(6, dtype=np.float64) + offset + 20.0,
            tcp_vel=np.arange(6, dtype=np.float64) + offset + 30.0,
        )
        gripper = SimpleNamespace(
            states=lambda: SimpleNamespace(width=width),
        )
        return SimpleNamespace(states=lambda: states), gripper

    harness.robot_left, harness.gripper_left = robot(0.0, 0.01)
    harness.robot_right, harness.gripper_right = robot(100.0, 0.02)
    harness._last_commanded_tcp_by_robot = {
        id(harness.robot_left): [41.0, 42.0, 43.0, 1.0, 0.0, 0.0, 0.0],
        id(harness.robot_right): [51.0, 52.0, 53.0, 1.0, 0.0, 0.0, 0.0],
    }

    state = harness.get_robot_state()

    assert state.shape == (88,)
    np.testing.assert_allclose(state[44:48], [0.01, 0.0, 0.02, 0.0])
    np.testing.assert_allclose(state[48:55], [41.0, 42.0, 43.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(state[68:74], np.arange(6) + 120.0)
    np.testing.assert_allclose(state[74:80], np.arange(6) + 30.0)
    np.testing.assert_allclose(state[86:88], [0.03, 0.04])


def test_control_decoder_honors_wxyz_and_primitive_order() -> None:
    harness = _StateHarness()
    control_format = parse_data_format(
        _format([8], ["gripper", "qwqxqyqz", "xyz"]), require_dtype=True
    )
    harness.control_layouts = validate_control_layouts(
        "eef_abs", control_format, 1
    )
    harness.control_shape = (8,)
    harness.control_dtype = "float64"

    command = harness._decode_control_array(
        np.array([0.08, 0.5, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    )[0]
    assert command["gripper"] == 0.08
    np.testing.assert_allclose(command["quaternion_xyzw"], [0.1, 0.2, 0.3, 0.5])
    np.testing.assert_allclose(command["position"], [1.0, 2.0, 3.0])


def test_rizon_normalization_accepts_tcp_feedback_without_gripper() -> None:
    normalized = normalize_device_config(
        {
            "class": "RizonRobot",
            "device_class_id": 1,
            "robot_sn": "Rizon4-test",
            "state_feedback": {
                "fps": 100.0,
                "buffer_size": 100,
                "hardware_latency_ms": 0.0,
                "format": _format(
                    [21], ["j1-7", "dq1-7", "xyz", "qwqxqyqz"]
                ),
            },
            "control": {
                "type": "eef_abs",
                "format": _format([7], ["xyz", "qxqyqzqw"]),
            },
        }
    )
    assert normalized["state_feedback"]["format"]["shape"] == [21]
    assert "data_shape" not in normalized
    assert "using_tcp_pose" not in normalized


def test_gripper_enable_switch_is_rejected_as_redundant() -> None:
    with pytest.raises(ValueError, match="gripper.enabled is redundant"):
        RizonGripperConfig.from_mapping(
            {"enabled": [True, True]}, gripper_required_by_arm=[True, True]
        )


def test_gripper_feedback_and_command_units_are_independent() -> None:
    config = RizonGripperConfig.from_mapping(
        {
            "backend": "lktech",
            "port": "/dev/test-gripper",
            "value_mode": "width_m",
            "command_value_mode": "normalized",
            "arm_length_m": 0.05,
            "apex_angle_max_deg": 60.0,
        },
        gripper_required_by_arm=[True],
    )

    assert config.value_mode == "width_m"
    assert config.command_value_mode == "normalized"
    assert config.arms[0].max_width == pytest.approx(0.05)
    harness = _StateHarness()
    harness.gripper_config = config
    assert harness._get_home_gripper_feedback_value(0) == pytest.approx(0.05)
    assert harness._get_home_gripper_command_value(0) == pytest.approx(1.0)


def test_nested_robot_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="control.execution contains unknown"):
        BaseRobot(
            state_feedback={
                "fps": 100.0,
                "buffer_size": 100,
                "hardware_latency_ms": 0.0,
                "format": _format([7], ["j1-7"]),
            },
            control={
                "type": "joints",
                "format": _format([7], ["j1-7"]),
                "execution": {"fps_typo": 100.0},
            },
        )

    with pytest.raises(ValueError, match="motion_limits.velocity contains unknown"):
        RizonMotionLimits.from_mapping(
            {"velocity": {"tcp_linear": 0.5}}, arm_count=1
        )
