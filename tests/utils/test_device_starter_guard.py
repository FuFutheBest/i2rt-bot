"""Tests for DeviceStarter cross-process and gripper preflight guards."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch

import psutil
import pytest

from utils.device_starter_guard import (
    DeviceStarterAlreadyRunningError,
    DeviceStarterPreflightError,
    DeviceStarterProcessGuard,
    check_gripper_websocket_preflight,
    configured_gripper_endpoints,
    device_starter_resources,
)

Address = namedtuple("Address", "ip port")
Connection = namedtuple("Connection", "laddr raddr status pid")


def _config(*, exclusive_client: bool = True) -> dict:
    return {
        "summary_shm": "device_summary_data",
        "devices": [
            {"class": "WebCameraOpenPi", "device_class_id": 0},
            {
                "class": "RizonRobot",
                "device_class_id": 2,
                "robot_sn": ["left-serial", "right-serial"],
                "gripper": {
                    "enabled": [True, True],
                    "backend": "lktech_websocket",
                    "server": {
                        "host": "127.0.0.1",
                        "port": 8765,
                        "exclusive_client": exclusive_client,
                    },
                },
            },
        ],
    }


def _listener() -> Connection:
    return Connection(Address("127.0.0.1", 8765), (), psutil.CONN_LISTEN, 100)


def _active_client(pid: int = 2725) -> Connection:
    return Connection(
        Address("127.0.0.1", 55032),
        Address("127.0.0.1", 8765),
        psutil.CONN_ESTABLISHED,
        pid,
    )


def test_resources_cover_shm_devices_robots_and_gripper() -> None:
    resources = device_starter_resources(_config())
    assert "shm:device_summary_data" in resources
    assert "device:WebCameraOpenPi:0" in resources
    assert "device:RizonRobot:2" in resources
    assert "robot:left-serial" in resources
    assert "robot:right-serial" in resources
    assert "gripper-websocket:127.0.0.1:8765" in resources


def test_resources_lock_default_summary_shm_name() -> None:
    config = _config()
    config.pop("summary_shm")
    assert "shm:device_summary_data" in device_starter_resources(config)


def test_string_false_values_do_not_enable_devices_or_grippers() -> None:
    config = _config()
    config["devices"][0]["enabled"] = "false"
    config["devices"][1]["gripper"]["enabled"] = ["false", "false"]
    config["devices"][1]["gripper"]["server"]["exclusive_client"] = "false"

    resources = device_starter_resources(config)

    assert "device:WebCameraOpenPi:0" not in resources
    assert "gripper-websocket:127.0.0.1:8765" not in resources


def test_ipv6_gripper_uri_is_bracketed() -> None:
    config = _config()
    config["devices"][1]["gripper"]["server"] = {
        "uri": "ws://[::1]:8765",
        "exclusive_client": True,
    }

    endpoint = configured_gripper_endpoints(config)[0]

    assert endpoint.host == "::1"
    assert endpoint.uri == "ws://[::1]:8765"


def test_overlapping_starter_is_rejected_with_owner_metadata(tmp_path: Path) -> None:
    first = DeviceStarterProcessGuard(
        _config(), lock_dir=tmp_path, argv=["device_starter.py", "first"]
    )
    second = DeviceStarterProcessGuard(
        _config(), lock_dir=tmp_path, argv=["device_starter.py", "second"]
    )
    first.acquire()
    try:
        with pytest.raises(DeviceStarterAlreadyRunningError, match=r"PID \d+"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_preflight_requires_listener() -> None:
    with pytest.raises(DeviceStarterPreflightError, match="not listening"):
        check_gripper_websocket_preflight(_config(), connections=[])


def test_preflight_reports_active_exclusive_client() -> None:
    process = Mock()
    process.cmdline.return_value = ["python", "-m", "control_panel.lingbotva_server"]
    with (
        patch("utils.device_starter_guard.psutil.Process", return_value=process),
        pytest.raises(DeviceStarterPreflightError, match=r"PID 2725.*control_panel"),
    ):
        check_gripper_websocket_preflight(
            _config(),
            connections=[_listener(), _active_client()],
        )


def test_preflight_allows_free_exclusive_server() -> None:
    check_gripper_websocket_preflight(_config(), connections=[_listener()])


def test_preflight_allows_active_client_for_multi_client_server() -> None:
    check_gripper_websocket_preflight(
        _config(exclusive_client=False),
        connections=[_listener(), _active_client()],
    )
