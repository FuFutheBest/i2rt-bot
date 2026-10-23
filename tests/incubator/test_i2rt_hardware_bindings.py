"""Hardware-free tests for backend-owned I2RT system configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from incubator.nedf3.I2RT_DM3.hardware_bindings import (
    HardwareBindingStore,
    HardwareCandidate,
    HardwareInventory,
    build_launch_plan,
)
from incubator.nedf3.I2RT_DM3.system_profiles import SystemProfileRegistry


def candidate(identity: str, backend: str, key: str) -> HardwareCandidate:
    return HardwareCandidate(
        identity=identity,
        backend=backend,
        label=identity,
        device_key=key,
        fingerprint={"identity": key},
    )


def lktech_inventory() -> HardwareInventory:
    return HardwareInventory(
        {
            "v4l2": (
                candidate("/dev/video0", "v4l2", "camera-left"),
                candidate("/dev/video2", "v4l2", "camera-right"),
            ),
            "serial": (
                candidate("/dev/ttyUSB0", "serial", "gripper-left"),
                candidate("/dev/ttyUSB1", "serial", "gripper-right"),
            ),
            "realsense": (),
        }
    )


def test_manifest_exposes_only_web_editable_hardware() -> None:
    registry = SystemProfileRegistry()
    profiles = {item["id"]: item for item in registry.public_profiles()}

    assert set(profiles) == {
        "lktech_fisheye_2cam",
        "yam_d405_2cam",
        "yam_d405_3cam",
    }
    assert profiles["lktech_fisheye_2cam"]["gripper"]["roles"] == [
        "left",
        "right",
    ]
    assert profiles["yam_d405_2cam"]["gripper"]["roles"] == []
    assert set(profiles["yam_d405_3cam"]["cameras"]) == {"left", "right", "head"}
    assert "can_channel" not in str(profiles).casefold()
    assert "can_leader" not in str(profiles).casefold()


def test_binding_store_rejects_duplicate_physical_device(tmp_path: Path) -> None:
    profile = SystemProfileRegistry().get("lktech_fisheye_2cam")
    inventory = lktech_inventory()
    duplicate = HardwareInventory(
        {
            **inventory.candidates,
            "serial": (
                candidate("/dev/ttyUSB0", "serial", "same-gripper"),
                candidate("/dev/ttyUSB1", "serial", "same-gripper"),
            ),
        }
    )
    store = HardwareBindingStore(tmp_path / "bindings.json")

    with pytest.raises(ValueError, match="同一个物理设备"):
        store.save(
            profile,
            cameras={"left": "/dev/video0", "right": "/dev/video2"},
            grippers={"left": "/dev/ttyUSB0", "right": "/dev/ttyUSB1"},
            inventory=duplicate,
        )


def test_binding_revalidates_fingerprint_before_launch(tmp_path: Path) -> None:
    profile = SystemProfileRegistry().get("lktech_fisheye_2cam")
    store = HardwareBindingStore(tmp_path / "bindings.json")
    store.save(
        profile,
        cameras={"left": "/dev/video0", "right": "/dev/video2"},
        grippers={"left": "/dev/ttyUSB0", "right": "/dev/ttyUSB1"},
        inventory=lktech_inventory(),
    )
    changed = HardwareInventory(
        {
            **lktech_inventory().candidates,
            "v4l2": (
                candidate("/dev/video0", "v4l2", "different-camera"),
                candidate("/dev/video2", "v4l2", "camera-right"),
            ),
        }
    )

    with pytest.raises(RuntimeError, match="设备身份已变化"):
        store.validate(profile, changed)


def test_lktech_launch_snapshot_contains_saved_ports(tmp_path: Path) -> None:
    profile = SystemProfileRegistry().get("lktech_fisheye_2cam")
    store = HardwareBindingStore(tmp_path / "bindings.json")
    binding = store.save(
        profile,
        cameras={"left": "/dev/video0", "right": "/dev/video2"},
        grippers={"left": "/dev/ttyUSB0", "right": "/dev/ttyUSB1"},
        inventory=lktech_inventory(),
    )

    plan = build_launch_plan(profile, binding, tmp_path / "launches")
    snapshot = yaml.safe_load(plan.teleop_config.read_text(encoding="utf-8"))

    assert snapshot["arms"]["left"]["follower"]["gripper"]["port"] == "/dev/ttyUSB0"
    assert snapshot["arms"]["right"]["follower"]["gripper"]["port"] == "/dev/ttyUSB1"
    assert plan.environment["SI_I2RT_LEFT_CAMERA"] == "/dev/video0"
    assert plan.environment["SI_I2RT_RIGHT_CAMERA"] == "/dev/video2"
    assert plan.environment["SI_I2RT_TELEOP_CONFIG_PATH"] == str(plan.teleop_config)


def test_yam_profile_rejects_any_web_gripper_binding(tmp_path: Path) -> None:
    profile = SystemProfileRegistry().get("yam_d405_2cam")
    inventory = HardwareInventory(
        {
            "v4l2": (),
            "serial": (candidate("/dev/ttyUSB0", "serial", "usb-can"),),
            "realsense": (
                candidate("d405-left", "realsense", "d405-left"),
                candidate("d405-right", "realsense", "d405-right"),
            ),
        }
    )

    with pytest.raises(ValueError, match="夹爪绑定"):
        HardwareBindingStore(tmp_path / "bindings.json").save(
            profile,
            cameras={"left": "d405-left", "right": "d405-right"},
            grippers={"left": "/dev/ttyUSB0"},
            inventory=inventory,
        )


def test_two_camera_yam_launch_disables_head_and_uses_synthetic_stream(
    tmp_path: Path,
) -> None:
    profile = SystemProfileRegistry().get("yam_d405_2cam")
    inventory = HardwareInventory(
        {
            "v4l2": (),
            "serial": (),
            "realsense": (
                candidate("d405-left", "realsense", "d405-left"),
                candidate("d405-right", "realsense", "d405-right"),
            ),
        }
    )
    store = HardwareBindingStore(tmp_path / "bindings.json")
    binding = store.save(
        profile,
        cameras={"left": "d405-left", "right": "d405-right"},
        grippers={},
        inventory=inventory,
    )

    plan = build_launch_plan(profile, binding, tmp_path / "launches")
    saver = yaml.safe_load(plan.saver_config.read_text(encoding="utf-8"))

    assert plan.environment["SI_I2RT_HEAD_CAMERA_ENABLED"] == "false"
    assert "SI_I2RT_HEAD_RS_SERIAL" not in plan.environment
    assert saver["devices"]["cameras"] == {
        "right": "RealSenseCamera_3_data",
        "left": "RealSenseCamera_2_data",
        "head": None,
    }
    assert saver["synthetic_camera_shapes"] == {"head": [480, 640, 3]}


def test_three_camera_yam_profile_builds_all_camera_bindings(tmp_path: Path) -> None:
    profile = SystemProfileRegistry().get("yam_d405_3cam")
    inventory = HardwareInventory(
        {
            "v4l2": (),
            "serial": (),
            "realsense": (
                candidate("d405-left", "realsense", "d405-left"),
                candidate("d405-right", "realsense", "d405-right"),
                candidate("d405-head", "realsense", "d405-head"),
            ),
        }
    )
    store = HardwareBindingStore(tmp_path / "bindings.json")
    binding = store.save(
        profile,
        cameras={
            "left": "d405-left",
            "right": "d405-right",
            "head": "d405-head",
        },
        grippers={},
        inventory=inventory,
    )

    plan = build_launch_plan(profile, binding, tmp_path / "launches")

    assert plan.environment == {
        "SI_I2RT_LEFT_RS_SERIAL": "d405-left",
        "SI_I2RT_RIGHT_RS_SERIAL": "d405-right",
        "SI_I2RT_HEAD_RS_SERIAL": "d405-head",
        "SI_I2RT_HEAD_CAMERA_ENABLED": "true",
        "SI_I2RT_YAM_GRIPPER_TYPE": "crank_4310",
        "SI_I2RT_TELEOP_CONFIG_PATH": str(profile.teleop_config),
    }
    assert plan.saver_config.name == "i2rt_d405_yam_saver.yaml"
