"""Regression tests for Rizon integer and sampled-JSON Home commands."""

from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from unittest import mock


def _rizon_robot_class():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "devices" or name.startswith("devices.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        return importlib.import_module("devices.robot").RizonRobot
    finally:
        for name in list(sys.modules):
            if name == "devices" or name.startswith("devices."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def _home_target() -> dict:
    return {
        "schema_version": 1,
        "source": "home_pose_candidates",
        "config_name": "fixture",
        "config_hash": "a" * 64,
        "arms": {
            "left": {
                "candidate_id": "left-01",
                "joints": [0.1] * 7,
                "collision_checked": False,
            },
            "right": {
                "candidate_id": "right-01",
                "joints": [-0.1] * 7,
                "collision_checked": False,
            },
        },
    }


def test_rizon_consumes_sampled_json_home_command(tmp_path) -> None:
    RizonRobot = _rizon_robot_class()
    robot = RizonRobot.__new__(RizonRobot)
    command_path = tmp_path / "RizonRobot_2_home"
    command_path.write_text(json.dumps(_home_target()), encoding="utf-8")
    robot._robot_home_command_pattern = str(tmp_path / "{device_name}_{device_class_id}_home")
    robot.device_name = "RizonRobot"
    robot.device_class_id = 2
    robot._clear_control_shm = mock.Mock()
    robot._go_home_with_candidate = mock.Mock(return_value=True)
    robot._notify_policy_runner_reset = mock.Mock()

    robot._check_and_execute_home_command()

    assert not command_path.exists()
    robot._go_home_with_candidate.assert_called_once_with(_home_target())
    robot._notify_policy_runner_reset.assert_called_once_with()
    result = json.loads(
        (tmp_path / "RizonRobot_2_home.result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "applied"
    assert result["candidates"] == {"left": "left-01", "right": "right-01"}


def test_rizon_reports_invalid_sampled_json_without_killing_worker(tmp_path) -> None:
    RizonRobot = _rizon_robot_class()
    robot = RizonRobot.__new__(RizonRobot)
    command_path = tmp_path / "RizonRobot_2_home"
    command_path.write_text('{"schema_version":', encoding="utf-8")
    robot._robot_home_command_pattern = str(tmp_path / "{device_name}_{device_class_id}_home")
    robot.device_name = "RizonRobot"
    robot.device_class_id = 2

    robot._check_and_execute_home_command()

    assert not command_path.exists()
    result = json.loads(
        (tmp_path / "RizonRobot_2_home.result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "failed"
    assert "JSONDecodeError" in result["error"]


def test_rizon_keeps_integer_home_preset_compatibility(tmp_path) -> None:
    RizonRobot = _rizon_robot_class()
    robot = RizonRobot.__new__(RizonRobot)
    command_path = tmp_path / "RizonRobot_2_home"
    command_path.write_text("3", encoding="utf-8")
    robot._robot_home_command_pattern = str(tmp_path / "{device_name}_{device_class_id}_home")
    robot.device_name = "RizonRobot"
    robot.device_class_id = 2
    robot._clear_control_shm = mock.Mock()
    robot._go_home_with_index = mock.Mock(return_value=True)
    robot._notify_policy_runner_reset = mock.Mock()

    robot._check_and_execute_home_command()

    robot._go_home_with_index.assert_called_once_with(3)
    robot._notify_policy_runner_reset.assert_called_once_with()
