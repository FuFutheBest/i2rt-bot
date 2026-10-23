"""Configuration contracts for terminal I2RT NEDF3 collection."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from consumer.io.async_data_saver_i2rt_NEDF3 import load_config
from devices.robots.i2rt_teleop.config import load_teleoperation_config
from utils.config_parser import parse_device_configs


def test_i2rt_terminal_configs_compose_and_match_shm_roles(repo_root: Path) -> None:
    saver = load_config(repo_root / "configs" / "i2rt_nedf3_saver.yaml")
    assert saver.leader_shm == "I2RTRobot_0_data"
    assert saver.follower_shm == "I2RTRobot_1_data"
    assert saver.camera_shm == {
        "right": "WebCameraDevice_3_data",
        "left": "WebCameraDevice_2_data",
        "head": None,
    }
    assert saver.synthetic_camera_shapes == {"head": (480, 640, 3)}
    assert saver.sample_hz == 90.0
    assert saver.resolved_gripper_sample_hz == 30.0
    assert saver.resolved_camera_sample_hz == 30.0
    teleoperation = load_teleoperation_config(
        repo_root / "configs" / "i2rt_bimanual_teleop.yaml"
    )
    for arm in teleoperation.arms.values():
        assert arm.follower.gripper.command_hz == 30.0
        assert arm.follower.gripper.feedback_hz == 30.0
    assert teleoperation.control.frequency_hz == 100.0

    config_dir = str((repo_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="i2rt_data_collect_async")
    devices = OmegaConf.to_container(cfg.devices.devices, resolve=True)
    teleop_devices = [
        entry for entry in devices if entry["class"] == "I2RTBimanualTeleop"
    ]
    cameras = [entry for entry in devices if entry["class"] == "WebCameraDevice"]

    assert len(teleop_devices) == 2
    assert teleop_devices[0]["device_class_id"] == 0
    assert "managed_by" not in teleop_devices[0]
    assert teleop_devices[1]["device_class_id"] == 1
    assert teleop_devices[1]["managed_by"] == "I2RTBimanualTeleop_0"
    assert teleop_devices[0]["device_name"] == "I2RTRobot"
    assert teleop_devices[1]["device_name"] == "I2RTRobot"
    assert teleop_devices[0]["enable_visualizer"] is False
    assert all(device["state_feedback"]["fps"] == 100.0 for device in teleop_devices)
    assert [camera["device_class_id"] for camera in cameras] == [2, 3]
    assert all(camera["state_feedback"]["fps"] == 30.0 for camera in cameras)
    assert all(
        camera["state_feedback"]["format"]["shape"] == [360, 640, 3]
        for camera in cameras
    )
    assert cameras[0]["capture_id"].endswith(
        "USB2_openMV_Camera_00.00.01-video-index0"
    )
    assert cameras[1]["capture_id"].endswith(
        "USB3_openMV_Camera_00.00.01-video-index0"
    )
    assert cfg.global_settings.restart_on_failure is False
    assert {
        "i2rt_random_home_request",
        "i2rt_random_home_target",
        "i2rt_random_home_result_pattern",
        "i2rt_takeover_hold_request",
        "i2rt_takeover_hold_ack",
    } <= set(cfg.runtime.signal_files)


def test_d405_yam_configs_match_camera_and_gripper_contract(
    repo_root: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SI_I2RT_LEFT_RS_SERIAL", "left-d405")
    monkeypatch.setenv("SI_I2RT_RIGHT_RS_SERIAL", "right-d405")
    monkeypatch.setenv("SI_I2RT_HEAD_RS_SERIAL", "head-d405")
    config_dir = str((repo_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="i2rt_d405_yam_collect_async")
    devices = OmegaConf.to_container(cfg.devices.devices, resolve=True)
    cameras = [device for device in devices if device["class"] == "RealSense"]
    assert [(camera["device_class_id"], camera["realsense_serial"]) for camera in cameras] == [
        (2, "left-d405"),
        (3, "right-d405"),
        (4, "head-d405"),
    ]
    assert all(camera["device_name"] == "RealSenseCamera" for camera in cameras)
    assert [camera["state_feedback"]["format"]["shape"] for camera in cameras] == [
        [480, 640, 3],
        [480, 640, 3],
        [480, 640, 3],
    ]

    teleoperation = load_teleoperation_config(
        repo_root / "configs" / "i2rt_d405_yam_teleop.yaml"
    )
    assert {arm.follower.gripper.backend for arm in teleoperation.arms.values()} == {
        "i2rt_can"
    }
    assert {arm.follower.gripper.i2rt_type for arm in teleoperation.arms.values()} == {
        "crank_4310"
    }
    assert all(arm.follower.gripper.port is None for arm in teleoperation.arms.values())
    assert {
        side: (arm.leader.can_channel, arm.follower.can_channel)
        for side, arm in teleoperation.arms.items()
    } == {"left": ("can1", "can3"), "right": ("can0", "can2")}
    assert teleoperation.arms["left"].follower.joint_directions == (1.0,) * 7
    assert teleoperation.arms["right"].follower.joint_directions == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
    )

    saver = load_config(repo_root / "configs" / "i2rt_d405_yam_saver.yaml")
    assert saver.camera_shm == {
        "right": "RealSenseCamera_3_data",
        "left": "RealSenseCamera_2_data",
        "head": "RealSenseCamera_4_data",
    }
    assert saver.synthetic_camera_shapes == {}


def test_d405_yam_device_config_can_disable_the_head_camera(
    repo_root: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SI_I2RT_LEFT_RS_SERIAL", "left-d405")
    monkeypatch.setenv("SI_I2RT_RIGHT_RS_SERIAL", "right-d405")
    monkeypatch.delenv("SI_I2RT_HEAD_RS_SERIAL", raising=False)
    monkeypatch.setenv("SI_I2RT_HEAD_CAMERA_ENABLED", "false")

    config_dir = str((repo_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="i2rt_d405_yam_collect_async")
    device_list = OmegaConf.to_container(cfg.devices.devices, resolve=True)
    devices = parse_device_configs({"devices": device_list})
    cameras = [
        device
        for device in devices
        if device["device_class"] == "RealSense"
    ]

    assert [
        (
            camera["device_class_id"],
            camera["config"]["realsense_serial"],
        )
        for camera in cameras
    ] == [(2, "left-d405"), (3, "right-d405")]
