"""Source-level checks for readable low-dimensional device configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from devices import DEVICE_CLASSES
from devices.contracts import DeviceKind
from utils.data_format import parse_data_format, parse_image_format
from utils.device_config import normalize_device_config


LOW_DIM_CLASSES = {
    "BaseDevice",
    "I2RTRobot",
    "JointSlider",
    "KeyboardTeleoperator",
    "Quest3Teleoperator",
    "Quest3VuerTeleoperator",
    "ReplayTeleoperator",
    "RizonRobot",
    "RizonRobotLKForceGated",
    "RoboTwinSimRobot",
    "RotaryEncoderDevice",
    "SimRobot",
    "ViveTrackerDevice",
    "X1Robot",
    "X1VuerTeleoperator",
    "YamHeadRobot",
}

CONTROLLABLE_CLASSES = {
    "I2RTRobot",
    "RizonRobot",
    "RizonRobotLKForceGated",
    "RoboTwinSimRobot",
    "SimRobot",
    "X1Robot",
}

LEGACY_STATE_FIELDS = {
    "data_shape",
    "data_dtype",
    "data_format",
    "fps",
    "buffer_size",
    "hardware_latency_ms",
}

LEGACY_CONTROL_FIELDS = {
    "control_shape",
    "control_dtype",
    "control_buffer_size",
    "control_type",
    "control_shm_name",
    "control_flag_shm_name",
    "impedance_parameter",
    "command_fps",
    "legacy_async_mode",
}

LEGACY_RIZON_FIELDS = {
    "max_velocity",
    "max_acceleration",
    "max_linear_vel",
    "max_linear_acc",
    "max_angular_vel",
    "max_angular_acc",
    "max_contact_wrench",
    "safety_bounds",
    "manual_confirm",
    "mirror_mode",
    "using_tcp_pose",
    "utilize_gripper",
    "enable",
}

LEGACY_RIZON_CONSTRUCTOR_FIELDS = {
    "gripper_name",
    "tool_name",
    "inference_gripper_calibration_path",
}

LEGACY_X1_CONTROL_FIELDS = {
    "enable_left",
    "enable_right",
    "enable_head",
    "utilize_gripper",
    "move_vel",
}


def test_low_dim_source_configs_use_grouped_contract(repo_root: Path) -> None:
    checked = 0
    for path in sorted((repo_root / "configs" / "devices").rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for device in document.get("devices", []):
            device_class = str(device.get("class", ""))
            if device_class not in LOW_DIM_CLASSES:
                continue

            checked += 1
            legacy_fields = LEGACY_STATE_FIELDS & set(device)
            if device_class in CONTROLLABLE_CLASSES:
                legacy_fields |= LEGACY_CONTROL_FIELDS & set(device)
            if device_class.startswith("RizonRobot"):
                legacy_fields |= LEGACY_RIZON_FIELDS & set(device)
            if device_class == "RizonRobot":
                legacy_fields |= LEGACY_RIZON_CONSTRUCTOR_FIELDS & set(device)
            if device_class == "X1Robot":
                legacy_fields |= LEGACY_X1_CONTROL_FIELDS & set(device)
            assert not legacy_fields, f"{path.name}: {sorted(legacy_fields)}"
            assert "state_feedback" in device, path.name

            state_format = parse_data_format(
                device["state_feedback"]["format"],
                require_dtype=True,
                context=f"{path.name}.state_feedback.format",
            )
            assert state_format.dimension > 0
            if device_class in CONTROLLABLE_CLASSES:
                assert "control" in device, path.name
                raw_control_format = device["control"]["format"]
                if isinstance(raw_control_format, str):
                    assert raw_control_format.startswith("${")
                else:
                    control_format = parse_data_format(
                        raw_control_format,
                        require_dtype=True,
                        context=f"{path.name}.control.format",
                    )
                    assert control_format.dimension > 0
            if device_class.startswith("RizonRobot") or device_class == "X1Robot":
                assert "motion_limits" in device, path.name
            if device_class == "RizonRobot":
                assert "grippers_enabled" not in device["control"], path.name
                gripper = device.get("gripper", {})
                assert "enabled" not in gripper, path.name

    assert checked >= 32


def test_source_configs_match_registered_device_capabilities(
    repo_root: Path,
) -> None:
    checked = 0
    for path in sorted((repo_root / "configs" / "devices").rglob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = document.get("devices", [])
        if isinstance(entries, dict):
            entries = entries.get("devices", [])
        for device in entries or []:
            if not isinstance(device, dict) or device.get("enabled", True) is False:
                continue
            device_class = device.get("class")
            if device_class not in DEVICE_CLASSES:
                continue
            checked += 1
            kind = DEVICE_CLASSES.kind(device_class)
            if kind is DeviceKind.CAMERA:
                assert "state_feedback" in device, (path.name, device_class)
                image_format = parse_image_format(device["state_feedback"]["format"])
                assert image_format.type == "rgb"
                continue
            if kind.readable:
                assert "state_feedback" in device, (path.name, device_class)
            if kind is DeviceKind.LOW_DIM_RW:
                assert "control" in device, (path.name, device_class)

    assert checked >= 70


def test_standalone_replay_config_uses_same_robot_contract(repo_root: Path) -> None:
    path = repo_root / "configs" / "devices" / "rizon_dual_robot_data_replay.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    robot = document["robot"]
    yam_head = document["yam_head_tracking"]["yam_head"]

    assert not (LEGACY_STATE_FIELDS | LEGACY_CONTROL_FIELDS | LEGACY_RIZON_FIELDS) & set(
        robot
    )
    assert not LEGACY_STATE_FIELDS & set(yam_head)
    normalized_robot = normalize_device_config(robot)
    normalized_yam_head = normalize_device_config(yam_head)
    assert normalized_robot["state_feedback"]["format"]["shape"] == [44]
    assert normalized_robot["control"]["format"]["shape"] == [16]
    assert normalized_yam_head["state_feedback"]["format"]["shape"] == [20]
    assert "data_shape" not in normalized_robot
    assert "control_shape" not in normalized_robot
    assert "enabled" not in robot["gripper"]
