"""DeviceStarter shared-memory readiness behavior."""

from __future__ import annotations

from collections import Counter
import os
import signal
from typing import Any

import device_starter as device_starter_module
import pytest
from device_starter import DeviceStarter, _install_child_signal_handlers
from devices.cameras.usb_link import UsbPortRequirementError
from utils.i2rt_random_home import read_random_home_request
from utils import device_readiness
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    pack_buffer_header,
    pack_device_header,
    pack_frame_header,
)


def _starter_with_devices(devices: list[dict[str, Any]]) -> DeviceStarter:
    starter = DeviceStarter.__new__(DeviceStarter)
    starter.devices = devices
    starter.config = {
        "global_settings": {
            "managed_device_ready_timeout": 1.0,
            "managed_device_ready_poll_interval": 0.0,
        }
    }
    return starter


def _usb_camera_device(requirement: str) -> dict[str, Any]:
    return {
        "device_class": "WebCameraOpenPi",
        "device_class_id": 7,
        "config": {
            "device_class_id": 7,
            "capture_id": "/dev/video7",
            "usb_port_requirement": requirement,
        },
    }


def test_child_signal_handlers_interrupt_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[int, Any] = {}
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    _install_child_signal_handlers()

    with pytest.raises(KeyboardInterrupt):
        handlers[signal.SIGINT](signal.SIGINT, None)
    handlers[signal.SIGTERM](signal.SIGTERM, None)


def test_device_can_disable_visualizer_when_global_visualizers_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        pid = 1234

        def start(self) -> None:
            pass

    starter = DeviceStarter.__new__(DeviceStarter)
    starter.enable_visualizers = True
    device = {
        "device_class": "I2RTBimanualTeleop",
        "device_class_id": 0,
        "config": {"enable_visualizer": False},
    }
    monkeypatch.setattr(starter, "_start_generic_device_process", lambda _: _Process())
    monkeypatch.setattr(
        starter,
        "_start_generic_visualizer_process",
        lambda _: pytest.fail("visualizer should not start"),
    )

    assert starter._start_device_process(device)
    assert "visualizer_process" not in device


@pytest.mark.parametrize(
    ("requirement", "speed"),
    [("usb_2.0", 480.0), ("usb_3.0", 5000.0), ("any", None)],
)
def test_usb_preflight_accepts_matching_web_camera_requirement(
    monkeypatch: pytest.MonkeyPatch,
    requirement: str,
    speed: float | None,
) -> None:
    starter = _starter_with_devices([_usb_camera_device(requirement)])
    monkeypatch.setattr(device_starter_module, "usb_speed_mbps", lambda _index: speed)

    starter._preflight_web_camera_usb_requirements()


def test_usb_preflight_rejects_mismatch_before_starting_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [_usb_camera_device("usb_2.0")]
    starter = DeviceStarter.__new__(DeviceStarter)
    starter.running = False
    starter.config = {}
    starter._parse_device_configs = lambda: devices

    monkeypatch.setattr(device_starter_module, "usb_speed_mbps", lambda _index: 5000.0)
    monkeypatch.setattr(
        device_starter_module,
        "cleanup_before_startup",
        lambda *_args, **_kwargs: pytest.fail("cleanup should not run"),
    )
    monkeypatch.setattr(
        starter,
        "_start_device_process",
        lambda _device: pytest.fail("device should not start"),
    )

    assert not starter.start_all()
    with pytest.raises(UsbPortRequirementError):
        starter._preflight_web_camera_usb_requirements()


def test_device_starter_publishes_one_fresh_i2rt_random_home_startup_request(
    tmp_path,
) -> None:
    """The paired I2RT workers must start from one shared, fresh request ID."""
    request_path = tmp_path / "i2rt-request.json"
    starter = DeviceStarter.__new__(DeviceStarter)
    starter.config = {
        "runtime": {
            "signal_files": {
                "i2rt_random_home_request": str(request_path),
                "i2rt_random_home_target": str(tmp_path / "i2rt-target.json"),
                "i2rt_random_home_result_pattern": str(
                    tmp_path / "i2rt-result-{device_id}.json"
                ),
            }
        }
    }
    starter.devices = [
        {
            "device_class": "I2RTRobot",
            "device_class_id": 1,
            "config": {"mode": "follower", "random_init_pose_sync": True},
        },
        {
            "device_class": "I2RTRobot",
            "device_class_id": 0,
            "config": {"mode": "leader", "random_init_pose_sync": True},
        },
    ]

    starter._prepare_i2rt_random_home_startup()

    request_id = read_random_home_request(request_path, reason="startup")
    assert request_id is not None
    assert len(request_id) == 32


def test_device_starter_rejects_partial_i2rt_random_home_pair(tmp_path) -> None:
    starter = DeviceStarter.__new__(DeviceStarter)
    starter.config = {
        "runtime": {
            "signal_files": {
                "i2rt_random_home_request": str(tmp_path / "request.json"),
                "i2rt_random_home_target": str(tmp_path / "target.json"),
                "i2rt_random_home_result_pattern": str(
                    tmp_path / "result-{device_id}.json"
                ),
            }
        }
    }
    starter.devices = [
        {
            "device_class": "I2RTRobot",
            "device_class_id": 1,
            "config": {"mode": "follower", "random_init_pose_sync": True},
        }
    ]

    with pytest.raises(ValueError, match="leader/follower pair"):
        starter._prepare_i2rt_random_home_startup()


def test_wait_for_device_streams_requires_changing_frames_from_every_stream(
    monkeypatch: Any,
) -> None:
    class _LiveProcess:
        def is_alive(self) -> bool:
            return True

    cameras = [
        {
            "device_class": "RoboTwinCamera",
            "device_class_id": device_class_id,
            "config": {"managed_by": "RoboTwinSimRobot_0"},
            "process": None,
        }
        for device_class_id in range(3)
    ]
    owner = {
        "device_class": "RoboTwinSimRobot",
        "device_class_id": 0,
        "config": {},
        "process": _LiveProcess(),
    }
    starter = _starter_with_devices([owner, *cameras])
    attempts: Counter[str] = Counter()
    closed: list[str] = []

    class _Channel:
        def __init__(
            self,
            name: str,
            device_class: str,
            device_class_id: int,
            timestamp_ns: int,
        ) -> None:
            self.name = name
            frame_size = FRAME_HEADER_SIZE + 1
            self.buf = bytearray(DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_size)
            self.buf[:DEVICE_HEADER_SIZE] = pack_device_header(
                device_class,
                device_class_id,
                30.0,
                "uint8",
                (1,),
                frame_size,
            )
            self.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = (
                pack_buffer_header(1, 1, 0)
            )
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            self.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = (
                pack_frame_header(timestamp_ns)
            )

        def close(self) -> None:
            closed.append(self.name)

    def connect(name: str, **kwargs: Any) -> _Channel:
        assert kwargs == {"read_only": True, "log_missing": False}
        attempts[name] += 1
        if attempts[name] == 1:
            raise FileNotFoundError(name)
        device_class, raw_device_class_id = name.removesuffix("_data").rsplit("_", 1)
        device_class_id = int(raw_device_class_id)
        timestamp_ns = 999 if attempts[name] == 2 else 1_001
        return _Channel(name, device_class, device_class_id, timestamp_ns)

    monkeypatch.setattr(device_readiness, "connect_to_shared_memory", connect)
    monkeypatch.setattr(device_readiness.time, "sleep", lambda _seconds: None)

    assert starter._wait_for_device_streams()
    assert attempts == Counter(
        {
            "RoboTwinSimRobot_0_data": 3,
            **{
                f"RoboTwinCamera_{device_class_id}_data": 3
                for device_class_id in range(3)
            },
        }
    )
    assert closed == [
        "RoboTwinSimRobot_0_data",
        *(f"RoboTwinCamera_{device_class_id}_data" for device_class_id in range(3)),
        "RoboTwinSimRobot_0_data",
        *(f"RoboTwinCamera_{device_class_id}_data" for device_class_id in range(3)),
    ]


def test_wait_for_device_streams_fails_when_owner_exits(monkeypatch: Any) -> None:
    class _DeadProcess:
        def is_alive(self) -> bool:
            return False

    starter = _starter_with_devices(
        [
            {
                "device_class": "RoboTwinSimRobot",
                "device_class_id": 0,
                "config": {},
                "process": _DeadProcess(),
            },
            {
                "device_class": "RoboTwinCamera",
                "device_class_id": 0,
                "config": {"managed_by": "RoboTwinSimRobot_0"},
                "process": None,
            },
        ]
    )

    def missing(_name: str, **_kwargs: Any) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(device_readiness, "connect_to_shared_memory", missing)

    assert not starter._wait_for_device_streams()


def test_wait_for_device_streams_rejects_an_unchanged_stale_frame(
    monkeypatch: Any,
) -> None:
    class _LiveProcess:
        def is_alive(self) -> bool:
            return True

    starter = _starter_with_devices(
        [
            {
                "device_class": "BaseDevice",
                "device_class_id": 0,
                "config": {},
                "process": _LiveProcess(),
            }
        ]
    )

    class _Channel:
        def __init__(self) -> None:
            frame_size = FRAME_HEADER_SIZE + 1
            self.buf = bytearray(
                DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_size
            )
            self.buf[:DEVICE_HEADER_SIZE] = pack_device_header(
                "BaseDevice", 0, 10.0, "uint8", (1,), frame_size
            )
            self.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = (
                pack_buffer_header(1, 1, 0)
            )
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            self.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = (
                pack_frame_header(123)
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        device_readiness,
        "connect_to_shared_memory",
        lambda *_args, **_kwargs: _Channel(),
    )
    with pytest.raises(
        device_readiness.DeviceReadinessError,
        match="Timed out waiting for live device stream",
    ):
        device_readiness.wait_for_device_streams(starter.devices, timeout_s=0.0)


def test_synchronized_manager_normalizes_params_master_device_class_id() -> None:
    starter = DeviceStarter.__new__(DeviceStarter)
    starter.config = {
        "device_manager": {
            "type": "synchronized",
            "params": {"buffer_size": 12, "master_device_class_id": 2},
        }
    }

    normalized = starter._normalized_device_manager_config()

    assert normalized["master_device_class_id"] == 2
    assert normalized["params"] == {"buffer_size": 12}
    assert starter.config["device_manager"] == normalized


def test_synchronized_manager_rejects_conflicting_master_device_class_ids() -> None:
    starter = DeviceStarter.__new__(DeviceStarter)
    starter.config = {
        "device_manager": {
            "type": "synchronized",
            "master_device_class_id": 1,
            "params": {"master_device_class_id": 2},
        }
    }

    try:
        starter._normalized_device_manager_config()
    except ValueError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("conflicting synchronized manager master IDs must fail")


def test_monitor_reports_an_unrecoverable_bridge_exit() -> None:
    class _DeadProcess:
        exitcode = 0

        def is_alive(self) -> bool:
            return False

    starter = DeviceStarter.__new__(DeviceStarter)
    starter.running = True
    starter.config = {"global_settings": {"restart_on_failure": False}}
    starter.devices = [
        {
            "device_class": "RoboTwinSimRobot",
            "device_class_id": 0,
            "process": _DeadProcess(),
            "restart_count": 0,
        }
    ]
    starter.policy_runner_process = None
    starter.device_manager_process = None

    assert not starter._monitor_devices()
    assert starter._child_failure_detected
    assert starter._shutdown_exit_code() == 1


def test_monitor_reports_visualizer_exit_as_unrecoverable() -> None:
    class _DeadProcess:
        exitcode = 0

        def is_alive(self) -> bool:
            return False

    starter = DeviceStarter.__new__(DeviceStarter)
    starter.running = True
    starter.config = {"global_settings": {"restart_on_failure": False}}
    starter.devices = [
        {
            "device_class": "RoboTwinSimRobot",
            "device_class_id": 0,
            "process": None,
            "visualizer_process": _DeadProcess(),
            "restart_count": 0,
        }
    ]
    starter.policy_runner_process = None
    starter.device_manager_process = None

    assert not starter._monitor_devices()
    assert starter._child_failure_detected
    assert starter._shutdown_exit_code() == 1


@pytest.mark.parametrize(
    ("child_failure_detected", "expected_exit_code"),
    [(False, 0), (True, 1)],
)
def test_signal_cleanup_preserves_child_failure_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    child_failure_detected: bool,
    expected_exit_code: int,
) -> None:
    starter = DeviceStarter.__new__(DeviceStarter)
    starter._child_failure_detected = child_failure_detected
    starter._shutdown_started = False
    stopped: list[bool] = []
    monkeypatch.setattr(starter, "stop_all", lambda: stopped.append(True))

    with pytest.raises(SystemExit) as exc_info:
        starter._signal_handler(signal.SIGINT, None)

    assert exc_info.value.code == expected_exit_code
    assert stopped == [True]


def test_repeated_signal_does_not_interrupt_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starter = DeviceStarter.__new__(DeviceStarter)
    starter._shutdown_started = True
    monkeypatch.setattr(
        starter,
        "stop_all",
        lambda: pytest.fail("shutdown must not be entered twice"),
    )

    starter._signal_handler(signal.SIGINT, None)


def test_stop_all_requests_shutdown_from_every_device_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            self.alive = False

        def terminate(self) -> None:
            pytest.fail("graceful shutdown should finish before SIGTERM")

        def kill(self) -> None:
            pytest.fail("graceful shutdown should finish before SIGKILL")

    starter = DeviceStarter.__new__(DeviceStarter)
    starter.running = True
    starter._shutdown_started = False
    starter._shutdown_complete = False
    starter.config = {
        "runtime": {
            "signal_files": {"reset_signal": str(tmp_path / "reset-signal")}
        },
        "global_settings": {
            "device_shutdown_timeout": 0.0,
            "managed_device_owner_shutdown_timeout": 0.0,
        },
    }
    starter.devices = [
        {
            "device_class": "I2RTBimanualTeleop",
            "device_class_id": 0,
            "config": {},
            "process": _Process(1001),
        },
        {
            "device_class": "I2RTBimanualTeleop",
            "device_class_id": 1,
            "config": {"managed_by": "I2RTBimanualTeleop_0"},
            "process": None,
        },
        {
            "device_class": "WebCameraDevice",
            "device_class_id": 0,
            "config": {},
            "process": _Process(1002),
        },
    ]
    starter.device_manager_process = None
    starter.policy_runner_process = None
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, signum: sent.append((pid, signum)))

    starter.stop_all()

    assert sent == [(1001, signal.SIGINT), (1002, signal.SIGINT)]
    assert starter._shutdown_complete
