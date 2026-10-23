"""Hardware-free contracts for the I2RT DM3 application."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import incubator.nedf3.I2RT_DM3.controller as controller_module
import incubator.nedf3.I2RT_DM3.process_manager as process_manager_module
from incubator.nedf3.I2RT_DM3.controller import I2RTDM3Controller
from incubator.nedf3.I2RT_DM3.controller import (
    resolve_visualizable_episode,
    visualizable_episode_dirs,
)
from incubator.nedf3.I2RT_DM3.process_manager import I2RTProcessManager
from incubator.nedf3.Vive_DM3.upload_manager import (
    UPLOAD_BACKEND_OSS,
    AutoUploadManager,
    UploadPolicy,
)
from incubator.nedf3.windows_DM3.dm3_api import DM3ApiClient
from utils.runtime_profiles import RuntimeProfileNotFoundError


def test_process_manager_builds_i2rt_only_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = I2RTProcessManager()
    calls: list[tuple[str, list[str], bool]] = []

    def capture(managed, command: list[str], *, stdin: bool) -> None:
        calls.append((managed.name, command, stdin))

    monkeypatch.setattr(manager, "_start_process", capture)
    manager.start_device()
    manager.start_saver(
        output_dir=str(tmp_path),
        task_name="task",
        task_description="description",
        dm3_user_id="user-id",
        dm3_user_name="user-name",
        task_execution_id="execution-id",
    )

    device_command = calls[0][1]
    assert calls[0][0] == "device"
    assert calls[0][2] is False
    assert device_command[-2:] == ["--config-name", "i2rt_data_collect_async"]
    assert device_command[2].endswith("device_starter.py")

    saver_command = calls[1][1]
    assert calls[1][0] == "saver"
    assert calls[1][2] is True
    assert saver_command[2:4] == ["-m", "consumer.io.async_data_saver_i2rt_NEDF3"]
    assert saver_command[saver_command.index("--task-execution-id") + 1] == "execution-id"
    assert "tracker" not in " ".join(device_command + saver_command).lower()


def test_d405_saver_uses_verified_recording_profile_without_device_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = I2RTProcessManager(
        device_config_name="i2rt_d405_yam_collect_async",
        require_recording_profile=True,
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager,
        "_start_process",
        lambda _managed, command, *, stdin: calls.append(command),
    )
    selections: list[dict[str, Any]] = []

    def select(_role: str, **kwargs: Any) -> SimpleNamespace:
        selections.append(kwargs)
        return SimpleNamespace(python=tmp_path / "recording-python", profile=None)

    monkeypatch.setattr(process_manager_module, "select_runtime_python", select)
    saver_args = {
        "output_dir": str(tmp_path),
        "task_name": "task",
        "task_description": "description",
        "dm3_user_id": "user-id",
        "dm3_user_name": "user-name",
        "task_execution_id": "execution-id",
    }
    manager.start_saver(**saver_args)
    assert calls[0][0] == str(tmp_path / "recording-python")
    assert selections[0]["config_name"] == "i2rt_d405_yam_collect_async"
    assert selections[0]["fallback"] is None
    assert selections[0]["require_verified_profile"] is True

    def missing(_role: str, **_kwargs: Any) -> None:
        raise RuntimeProfileNotFoundError("missing recording profile")

    monkeypatch.setattr(process_manager_module, "select_runtime_python", missing)
    with pytest.raises(RuntimeError, match="NEDF3 录制环境不可用"):
        manager.start_saver(**saver_args)
    assert len(calls) == 1


def test_d405_yam_controller_uses_new_configs_and_blocks_lktech_calibration() -> None:
    controller = I2RTDM3Controller(
        device_config_name="i2rt_d405_yam_collect_async",
        saver_config_path="configs/i2rt_d405_yam_saver.yaml",
    )
    assert controller.device_config_name == "i2rt_d405_yam_collect_async"
    assert controller.saver_config_path == "configs/i2rt_d405_yam_saver.yaml"
    with pytest.raises(RuntimeError, match="不使用 LKTech 标定"):
        controller.start_gripper_calibration()


def test_two_camera_profile_launch_uses_two_camera_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = I2RTDM3Controller()
    profile = controller.profile_registry.get("yam_d405_2cam")
    saver_config = tmp_path / "yam_d405_2cam.saver.yaml"
    environment = {
        "SI_I2RT_LEFT_RS_SERIAL": "left-serial",
        "SI_I2RT_RIGHT_RS_SERIAL": "right-serial",
        "SI_I2RT_HEAD_CAMERA_ENABLED": "false",
    }
    plan = SimpleNamespace(
        teleop_config=profile.teleop_config,
        saver_config=saver_config,
        environment=environment,
    )
    monkeypatch.setattr(controller.binding_store, "validate", lambda *_args: {})
    monkeypatch.setattr(controller, "_hardware_discovery", lambda: None)
    monkeypatch.setattr(controller_module, "build_launch_plan", lambda *_args: plan)
    validated_roles: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        controller_module,
        "validate_d405_yam_rig",
        lambda _environment, _teleop, roles: validated_roles.append(roles),
    )
    starts: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        controller.processes,
        "start_device",
        lambda config, *, environment: starts.append((config, environment)),
    )

    controller.start_device(profile.profile_id, profile.revision)

    assert validated_roles == [("left", "right")]
    assert starts == [("i2rt_d405_yam_collect_async", environment)]
    assert controller.saver_config_path == str(saver_config)
    assert controller.active_profile_id == "yam_d405_2cam"


def test_d405_yam_device_start_checks_camera_and_gripper_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = I2RTDM3Controller(device_config_name="i2rt_d405_yam_collect_async")
    calls: list[str] = []
    monkeypatch.setattr(controller.processes, "start_device", calls.append)
    monkeypatch.setenv("SI_I2RT_LEFT_RS_SERIAL", "left-serial")
    monkeypatch.setenv("SI_I2RT_RIGHT_RS_SERIAL", "right-serial")
    monkeypatch.setenv("SI_I2RT_HEAD_RS_SERIAL", "head-serial")
    monkeypatch.setenv("SI_I2RT_YAM_GRIPPER_TYPE", "crank_4310")
    can_serials = {
        "can1": "left-leader",
        "can3": "left-follower",
        "can0": "right-leader",
        "can2": "right-follower",
    }
    for role, side, channel in (
        ("LEADER", "LEFT", "can1"),
        ("FOLLOWER", "LEFT", "can3"),
        ("LEADER", "RIGHT", "can0"),
        ("FOLLOWER", "RIGHT", "can2"),
    ):
        monkeypatch.setenv(f"SI_I2RT_CAN_{role}_{side}_SERIAL", can_serials[channel])
    can_flags = ["UP"]

    def inspect_can(command: list[str], **_kwargs):
        channel = command[-1].removeprefix("/sys/class/net/")
        if command[0] == "udevadm":
            return SimpleNamespace(stdout=f"ID_SERIAL={can_serials[channel]}\n")
        return SimpleNamespace(
            stdout=json.dumps(
                [
                    {
                        "flags": can_flags,
                        "linkinfo": {
                            "info_data": {"bittiming": {"bitrate": 1_000_000}}
                        },
                    }
                ]
            )
        )

    monkeypatch.setattr(controller_module.subprocess, "run", inspect_can)

    def camera(serial: str):
        return SimpleNamespace(get_info=lambda _key: serial)

    monkeypatch.setitem(
        sys.modules,
        "pyrealsense2",
        SimpleNamespace(
            camera_info=SimpleNamespace(serial_number=object()),
            context=lambda: SimpleNamespace(
                query_devices=lambda: [
                    camera("left-serial"),
                    camera("right-serial"),
                    camera("head-serial"),
                ]
            ),
        ),
    )
    controller.start_device()
    assert calls == ["i2rt_d405_yam_collect_async"]

    monkeypatch.delenv("SI_I2RT_HEAD_RS_SERIAL")
    monkeypatch.setitem(
        sys.modules,
        "pyrealsense2",
        SimpleNamespace(
            camera_info=SimpleNamespace(serial_number=object()),
            context=lambda: SimpleNamespace(
                query_devices=lambda: [
                    camera("left-serial"),
                    camera("right-serial"),
                ]
            ),
        ),
    )
    controller_module.validate_d405_yam_rig(camera_roles=("left", "right"))
    monkeypatch.setenv("SI_I2RT_HEAD_RS_SERIAL", "head-serial")

    monkeypatch.setenv("SI_I2RT_RIGHT_RS_SERIAL", "left-serial")
    with pytest.raises(RuntimeError, match="三个不同的"):
        controller.start_device()
    monkeypatch.setenv("SI_I2RT_RIGHT_RS_SERIAL", "missing-serial")
    with pytest.raises(RuntimeError, match="找不到配置的 D405"):
        controller.start_device()
    monkeypatch.setenv("SI_I2RT_RIGHT_RS_SERIAL", "right-serial")
    monkeypatch.setenv("SI_I2RT_HEAD_RS_SERIAL", "right-serial")
    with pytest.raises(RuntimeError, match="三个不同的"):
        controller.start_device()
    monkeypatch.setenv("SI_I2RT_HEAD_RS_SERIAL", "head-serial")
    monkeypatch.setenv("SI_I2RT_YAM_GRIPPER_TYPE", "linear_3507")
    with pytest.raises(RuntimeError, match="夹爪型号"):
        controller.start_device()
    monkeypatch.setenv("SI_I2RT_YAM_GRIPPER_TYPE", "crank_4310")
    monkeypatch.setenv("SI_I2RT_CAN_LEADER_LEFT_SERIAL", "wrong-adapter")
    with pytest.raises(RuntimeError, match="CAN 适配器身份"):
        controller.start_device()
    monkeypatch.setenv("SI_I2RT_CAN_LEADER_LEFT_SERIAL", "left-leader")
    can_flags.clear()
    with pytest.raises(RuntimeError, match="必须为 UP"):
        controller.start_device()
    assert calls == ["i2rt_d405_yam_collect_async"]


def test_process_manager_builds_idle_only_gripper_calibration_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = I2RTProcessManager()
    calls: list[tuple[str, list[str], bool]] = []

    def capture(managed, command: list[str], *, stdin: bool) -> None:
        calls.append((managed.name, command, stdin))

    monkeypatch.setattr(manager, "_start_process", capture)
    manager.start_calibration()

    name, command, stdin = calls[0]
    assert name == "calibration"
    assert stdin is False
    assert command[2].endswith("calibrate_i2rt_grippers.py")
    assert command[-4:] == ["--config", "configs/i2rt_bimanual_teleop.yaml", "--side", "both"]

    manager.device.running = True
    with pytest.raises(RuntimeError, match="停止设备"):
        manager.start_calibration()


def test_process_manager_launches_latest_episode_in_recording_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = I2RTProcessManager(
        device_config_name="i2rt_d405_yam_collect_async",
        require_recording_profile=True,
    )
    older = tmp_path / "episode_0001"
    newer = tmp_path / "episode_0002"
    older.mkdir()
    newer.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    calls: list[tuple[str, list[str], bool]] = []

    monkeypatch.setattr(
        process_manager_module,
        "select_runtime_python",
        lambda _role, **_kwargs: SimpleNamespace(
            python=tmp_path / "recording-python",
            profile=None,
        ),
    )
    monkeypatch.setattr(
        manager,
        "_start_process",
        lambda managed, command, *, stdin: calls.append(
            (managed.name, command, stdin)
        ),
    )

    episode = manager.start_visualizer(newer)

    assert episode == newer
    name, command, stdin = calls[0]
    assert name == "visualizer"
    assert stdin is False
    assert command[0] == str(tmp_path / "recording-python")
    assert command[-2].endswith("scripts/visualization/i2rt_nedf3_rerun.py")
    assert command[-1] == str(newer)


def test_visualizer_episode_selection_stays_inside_data_root(tmp_path: Path) -> None:
    episode = tmp_path / "task" / "episode_0001"
    (episode / "data").mkdir(parents=True)
    (episode / "metadata.json").write_text("{}", encoding="utf-8")
    incomplete = tmp_path / "task" / "episode_0002"
    incomplete.mkdir()

    assert visualizable_episode_dirs(tmp_path) == [episode]
    assert resolve_visualizable_episode(tmp_path, "task/episode_0001") == episode
    with pytest.raises(ValueError, match="不能离开"):
        resolve_visualizable_episode(tmp_path, "../outside/episode_0001")
    with pytest.raises(ValueError, match="不是已完成"):
        resolve_visualizable_episode(tmp_path, "task/episode_0002")


def test_process_manager_blocks_device_start_during_gripper_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = I2RTProcessManager()
    calls: list[str] = []
    monkeypatch.setattr(
        manager,
        "_start_process",
        lambda managed, _command, *, stdin: calls.append(managed.name),
    )
    manager.calibration.running = True

    with pytest.raises(RuntimeError, match="标定进程运行时不能启动设备"):
        manager.start_device()

    assert calls == []


def test_device_readiness_waits_for_device_starter_success_log() -> None:
    manager = I2RTProcessManager()

    manager._update_runtime_from_log(
        manager.device,
        "Started WebCameraDevice device 0",
    )
    assert not manager.device.ready_for_episode

    manager._update_runtime_from_log(
        manager.device,
        "Successfully started 3/4 devices",
    )
    assert manager.device.ready_for_episode


def test_device_logs_track_teleoperation_phase_and_watchdog_fault() -> None:
    manager = I2RTProcessManager()

    manager._update_runtime_from_log(
        manager.device,
        "Teleoperation ready; press both top buttons to enable, one to disable",
    )
    assert manager.device.teleoperation_phase == "safe_hold"

    for line in (
        "Aligning followers over 3.00 seconds (max distance 0.500 rad, "
        "velocity limit 0.700 rad/s) while leaders hold position",
        "Aligning followers to frozen leader targets over 3.00 seconds",
    ):
        manager._update_runtime_from_log(manager.device, line)
        assert manager.device.teleoperation_phase == "aligning"

    manager._update_runtime_from_log(manager.device, "Teleoperation enabled")
    manager.device.ready_for_episode = True
    assert manager.device.teleoperation_phase == "active"

    manager._update_runtime_from_log(
        manager.device,
        "left/follower watchdog fault: command lease expired",
    )
    assert manager.device.teleoperation_phase == "fault"
    assert manager.device.teleoperation_fault == "command lease expired"
    assert not manager.device.ready_for_episode


def test_process_logs_remove_ansi_sequences_before_display() -> None:
    manager = I2RTProcessManager()
    manager.device.append_log(
        "\x1b[32m2026-10-15 19:09:56.947\x1b[0m | "
        "\x1b[1mINFO    \x1b[0m | All devices stopped\n"
    )

    lines, offset = manager.device.read_logs_since(0)

    assert lines == [
        "2026-10-15 19:09:56.947 | INFO     | All devices stopped\n"
    ]
    assert offset == 1


def test_i2rt_controller_uses_i2rt_upload_resource_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeUploadManager:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.configured: list[dict[str, Any]] = []
            self.log_callback = kwargs.get("log_callback")

        def configure(self, **kwargs: Any) -> None:
            self.configured.append(kwargs)

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(controller_module, "AutoUploadManager", FakeUploadManager)
    controller = I2RTDM3Controller()
    requested_types: list[str] = []

    def fetch_oss_sts_token(*, resource_type: str) -> dict[str, str]:
        requested_types.append(resource_type)
        return {"token": "test"}

    monkeypatch.setattr(
        controller.session,
        "get_client",
        lambda: SimpleNamespace(fetch_oss_sts_token=fetch_oss_sts_token),
    )

    manager = controller.create_upload_manager(tmp_path)

    assert controller.resource_type == "i2rt"
    assert manager.configured[-1] == {"resource_type": "i2rt"}
    manager.kwargs["token_fetcher"]("i2rt")
    assert requested_types == ["i2rt"]


def test_upload_manager_preserves_explicit_empty_resource_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = AutoUploadManager.__new__(AutoUploadManager)
    manager._state_lock = threading.RLock()
    manager.data_dir = tmp_path
    manager.data_dirs = [tmp_path]
    manager.resource_type = "previous"
    manager.policy = UploadPolicy()
    manager.poll_interval_sec = 0
    manager.max_retries = 5
    manager.upload_workers = 2
    manager.multipart_threads = 4
    manager.upload_backend = UPLOAD_BACKEND_OSS
    manager.http_sink_url = ""
    manager.http_timeout_sec = 120
    manager._state = {"episodes": {}}
    monkeypatch.setattr(manager, "_save_state_locked", lambda: 0)

    manager.configure(resource_type="")

    assert manager.resource_type == ""
    assert manager._state["resource_type"] == ""


def test_dm3_sts_request_keeps_resource_type_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DM3ApiClient()
    captured: dict[str, Any] = {}

    def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured.update({"method": method, "path": path, **kwargs})
        return {"token": "test"}

    monkeypatch.setattr(client, "_request", request)

    client.fetch_oss_sts_token(resource_type="")

    assert captured["path"] == "/api/v1/oss/sts-token"
    assert captured["params"] == {"resource_type": ""}


def test_i2rt_gui_exposes_only_requested_workflow_tabs(repo_root: Path) -> None:
    source = (
        repo_root / "incubator" / "nedf3" / "I2RT_DM3" / "gui.py"
    ).read_text(encoding="utf-8")

    assert 'text="DM3 登录"' in source
    assert 'text="数据采集"' in source
    assert 'text="上传与性能监控"' in source
    assert "TrackerDiagnosticFrame" not in source
    assert "DiagnosticFrame" not in source
    assert "CalibrationManager" not in source
