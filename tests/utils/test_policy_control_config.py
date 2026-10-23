"""Tests for structured policy command-target configuration."""

from __future__ import annotations

import pytest

from utils.policy_control_config import ControlTargetConfig, parse_control_targets


def _target() -> dict:
    return {
        "endpoint": {
            "device": "RizonRobot",
            "device_class_id": 2,
            "port": "command",
        },
        "shape": [16],
        "dtype": "float64",
    }


def test_control_target_resolves_canonical_command_shm() -> None:
    target = ControlTargetConfig.from_mapping(_target())

    assert target.device_key == "RizonRobot_2"
    assert target.shm_name == "RizonRobot_2_control"
    assert target.runtime_mapping()["shm_name"] == "RizonRobot_2_control"
    assert target.config_mapping()["endpoint"] == _target()["endpoint"]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (
            {"shm": "RizonRobot_2_control", "shape": [16], "dtype": "float64"},
            "shm was removed",
        ),
        (
            {
                "endpoint": {
                    "device": "RizonRobot",
                    "device_class_id": 2,
                    "port": "state",
                },
                "shape": [16],
                "dtype": "float64",
            },
            "command",
        ),
    ],
)
def test_control_target_rejects_legacy_or_non_command_endpoint(
    target: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ControlTargetConfig.from_mapping(target)


def test_parse_control_targets_requires_a_sequence() -> None:
    with pytest.raises(ValueError, match="must be a sequence"):
        parse_control_targets(_target(), field="policy.controls")


def test_policy_runner_rejects_legacy_raw_shm_target() -> None:
    """The action-executor runtime must not bypass the schema migration."""
    from manager.base_policy_runner import BasePolicyRunner

    with pytest.raises(ValueError, match="shm was removed"):
        BasePolicyRunner(
            config={"policy": {"params": {}}},
            robot_control_shms=[
                {"shm": "RizonRobot_2_control", "shape": [16], "dtype": "float64"}
            ],
            signal_files={"policy_runner_reset_command": "/tmp/test-reset"},
        )
