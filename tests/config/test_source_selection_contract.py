"""Tests for the structured robot control source contract."""

from __future__ import annotations

import time

import pytest

from devices.config import ShmEndpointConfig
from devices.contracts import DeviceKind
from devices.robots.base import BaseRobot
from devices.robots.config import RobotControlConfig, SourceSelectionConfig
from utils.device_config import DeviceConfigError, parse_device_config


def _control(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "type": "joints",
        "format": {
            "shape": [14],
            "dtype": "float64",
            "type": ["j1-7", "j1-7"],
        },
    }
    value.update(overrides)
    return value


def test_omitted_source_selection_defaults_to_self_command() -> None:
    config = RobotControlConfig.from_mapping(_control())

    assert config.source_selection.mode == "single_source"
    assert config.source_selection.source is not None
    assert config.source_selection.source.resolve(
        owner_device="I2RTRobot", owner_device_class_id=1
    ) == "I2RTRobot_1_control"


def test_dual_source_uses_the_fixed_selector_protocol() -> None:
    config = RobotControlConfig.from_mapping(
        _control(
            source_selection={
                "mode": "dual_source",
                "sources": {
                    "policy": {"device": "self", "port": "command"},
                    "teleoperation": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "state",
                    },
                },
                "dual_source_selector": {
                    "endpoint": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "control_flag",
                    },
                },
            }
        )
    )

    selection = config.source_selection
    assert selection.sources is not None
    assert selection.sources["teleoperation"].resolve(
        owner_device="I2RTRobot", owner_device_class_id=1
    ) == "KeyboardTeleoperator_0_data"
    assert selection.dual_source_selector is not None
    assert selection.dual_source_selector.endpoint.resolve_shm_name(
        owner_device="I2RTRobot", owner_device_class_id=1
    ) == "KeyboardTeleoperator_0_control_flag"


def test_control_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="control contains unknown fields: unexpected"):
        RobotControlConfig.from_mapping(_control(unexpected=True))


def test_removed_top_level_device_fields_are_rejected() -> None:
    with pytest.raises(DeviceConfigError, match="actuators"):
        parse_device_config(
            {
                "class": "BaseDevice",
                "device_class_id": 0,
                "actuators": {"arms": {"left": True, "right": True}},
                "state_feedback": {
                    "fps": 10.0,
                    "buffer_size": 2,
                    "hardware_latency_ms": 0.0,
                    "format": {
                        "shape": [2],
                        "dtype": "float64",
                        "type": ["linear:2"],
                    },
                },
            },
            device_kind=DeviceKind.LOW_DIM_R,
        )


def test_external_endpoint_requires_an_id() -> None:
    with pytest.raises(ValueError, match="device_class_id is required"):
        ShmEndpointConfig.from_mapping(
            {"device": "KeyboardTeleoperator", "port": "state"}
        )


def test_self_endpoint_cannot_override_its_owner_id() -> None:
    with pytest.raises(ValueError, match="must be omitted"):
        ShmEndpointConfig.from_mapping(
            {"device": "self", "device_class_id": 2, "port": "command"}
        )
    with pytest.raises(ValueError, match="must be omitted"):
        ShmEndpointConfig(device="self", device_class_id=2, port="command")


def test_dual_source_requires_binary_selector() -> None:
    with pytest.raises(ValueError, match="dual_source_selector is required"):
        SourceSelectionConfig.from_mapping(
            {
                "mode": "dual_source",
                "sources": {
                    "policy": {"device": "self", "port": "command"},
                    "teleoperation": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "state",
                    },
                },
            }
        )


def test_dual_source_selector_rejects_mapping_override() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        SourceSelectionConfig.from_mapping(
            {
                "mode": "dual_source",
                "sources": {
                    "policy": {"device": "self", "port": "command"},
                    "teleoperation": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "state",
                    },
                },
                "dual_source_selector": {
                    "endpoint": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "control_flag",
                    },
                    "mapping": {"0": "teleoperation", "1": "policy"},
                },
            }
        )


def test_robot_resolves_external_single_source_and_keeps_own_command_channel() -> None:
    robot = BaseRobot(
        device_class_id=3,
        state_feedback={
            "fps": 100.0,
            "buffer_size": 4,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [7],
                "dtype": "float64",
                "type": ["j1-7"],
            },
        },
        control=_control(
            source_selection={
                "mode": "single_source",
                "source": {
                    "device": "JointSlider",
                    "device_class_id": 0,
                    "port": "state",
                },
            }
        ),
    )

    assert robot._selected_source_name() == "JointSlider_0_data"
    assert robot.control_shm_names == [
        "JointSlider_0_data",
        "BaseRobot_3_control",
    ]
    assert robot._external_control_shm_names == {"JointSlider_0_data"}
    assert robot._owned_control_shm_names == {"BaseRobot_3_control"}


def test_robot_resolves_dual_source_roles_from_selector_value() -> None:
    robot = BaseRobot(
        device_class_id=1,
        state_feedback={
            "fps": 100.0,
            "buffer_size": 4,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [14],
                "dtype": "float64",
                "type": ["j1-7", "j1-7"],
            },
        },
        control=_control(
            source_selection={
                "mode": "dual_source",
                "sources": {
                    "policy": {"device": "self", "port": "command"},
                    "teleoperation": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "state",
                    },
                },
                "dual_source_selector": {
                    "endpoint": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "control_flag",
                    },
                },
            }
        ),
    )

    assert robot._selected_source_name(0.0) == "BaseRobot_1_control"
    assert robot._selected_source_name(1.0) == "KeyboardTeleoperator_0_data"
    assert robot._selected_source_name(0.6) == "BaseRobot_1_control"
    assert robot._selected_source_name(float("nan")) == "BaseRobot_1_control"
    assert robot._control_flag_shm_name == "KeyboardTeleoperator_0_control_flag"


def test_base_robot_cannot_own_a_dual_source_selector() -> None:
    with pytest.raises(ValueError, match="selector-producing robot"):
        BaseRobot(
            device_class_id=1,
            state_feedback={
                "fps": 100.0,
                "buffer_size": 4,
                "hardware_latency_ms": 0.0,
                "format": {
                    "shape": [14],
                    "dtype": "float64",
                    "type": ["j1-7", "j1-7"],
                },
            },
            control=_control(
                source_selection={
                    "mode": "dual_source",
                    "sources": {
                        "policy": {"device": "self", "port": "command"},
                        "teleoperation": {
                            "device": "KeyboardTeleoperator",
                            "device_class_id": 0,
                            "port": "state",
                        },
                    },
                    "dual_source_selector": {
                        "endpoint": {"device": "self", "port": "control_flag"},
                    },
                }
            ),
        )


def test_source_endpoint_rejects_control_flag_port() -> None:
    with pytest.raises(ValueError, match="state, command"):
        SourceSelectionConfig.from_mapping(
            {
                "mode": "single_source",
                "source": {"device": "self", "port": "control_flag"},
            }
        )


def test_dual_source_endpoints_cannot_resolve_to_one_channel() -> None:
    with pytest.raises(ValueError, match="different shared-memory channels"):
        BaseRobot(
            device_class_id=1,
            state_feedback={
                "fps": 100.0,
                "buffer_size": 4,
                "hardware_latency_ms": 0.0,
                "format": {
                    "shape": [14],
                    "dtype": "float64",
                    "type": ["j1-7", "j1-7"],
                },
            },
            control=_control(
                source_selection={
                    "mode": "dual_source",
                    "sources": {
                        "policy": {"device": "self", "port": "command"},
                        "teleoperation": {
                            "device": "BaseRobot",
                            "device_class_id": 1,
                            "port": "command",
                        },
                    },
                    "dual_source_selector": {
                        "endpoint": {
                            "device": "KeyboardTeleoperator",
                            "device_class_id": 0,
                            "port": "control_flag",
                        },
                    },
                }
            ),
        )


def test_selector_value_is_snapshotted_for_one_control_tick() -> None:
    robot = BaseRobot(
        device_class_id=1,
        state_feedback={
            "fps": 100.0,
            "buffer_size": 4,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [14],
                "dtype": "float64",
                "type": ["j1-7", "j1-7"],
            },
        },
        control=_control(
            source_selection={
                "mode": "dual_source",
                "sources": {
                    "policy": {"device": "self", "port": "command"},
                    "teleoperation": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "state",
                    },
                },
                "dual_source_selector": {
                    "endpoint": {
                        "device": "KeyboardTeleoperator",
                        "device_class_id": 0,
                        "port": "control_flag",
                    },
                },
            }
        ),
    )
    robot._read_source_selector_value = lambda: 1.0
    robot._process_source_selection_transition()
    robot._read_source_selector_value = lambda: 0.0

    assert robot._selected_source_name(robot._selector_value_for_control_tick()) == (
        "KeyboardTeleoperator_0_data"
    )
    assert not robot._frame_is_fresh(time.time_ns() + 1_000_000_000, 100.0)
