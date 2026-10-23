"""Contract tests for Yam-head tracking command endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.yam_head_tracking import PlannedRobotCenterTracking


def _strategy(config: dict[str, object]) -> PlannedRobotCenterTracking:
    return PlannedRobotCenterTracking(
        config,
        repo_root=Path.cwd(),
        left2world_json_path=None,
        right2world_json_path=None,
        yam2world_json_path=None,
    )


def test_planned_tracking_resolves_a_command_endpoint() -> None:
    strategy = _strategy(
        {
            "control_endpoint": {
                "device": "RizonRobot",
                "device_class_id": 2,
                "port": "command",
            }
        }
    )

    assert strategy.control_shm_name == "RizonRobot_2_control"
    assert strategy.control_endpoint.port == "command"


@pytest.mark.parametrize(
    "config, match",
    [
        ({"control_shm_name": "RizonRobot_2_control"}, "is removed"),
        (
            {
                "control_endpoint": {
                    "device": "RizonRobot",
                    "device_class_id": 2,
                    "port": "command",
                },
                "control_shm_name": "RizonRobot_2_control",
            },
            "is removed",
        ),
        (
            {
                "control_endpoint": {
                    "device": "RizonRobot",
                    "device_class_id": 2,
                    "port": "state",
                }
            },
            "port must be one of: command",
        ),
        (
            {"control_endpoint": {"device": "self", "port": "command"}},
            "must name the robot device",
        ),
    ],
)
def test_planned_tracking_rejects_noncanonical_control_endpoint(
    config: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _strategy(config)
