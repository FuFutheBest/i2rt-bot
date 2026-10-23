"""Passive Rizon safety-heartbeat tests; no SDK or hardware is imported."""

from __future__ import annotations

import time
from typing import Any

from incubator.control_panel import lifecycle


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


class _Handle:
    def __init__(self, status: str, *, fault: bool, operational: bool) -> None:
        self._status = status
        self._fault = fault
        self._operational = operational

    def operational_status(self) -> _Status:
        return _Status(self._status)

    def fault(self) -> bool:
        return self._fault

    def is_operational(self) -> bool:
        return self._operational


class _Device:
    is_dual_arm = False

    def __init__(self, handle: Any) -> None:
        self.robot = handle

    def _check_operational_status_change(self) -> None:
        return None


def test_worker_reports_external_estop_fail_closed(monkeypatch: Any) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        lifecycle,
        "_send_ownership_event",
        lambda event, **details: events.append((event, details)),
    )
    device = _Device(
        _Handle("EMERGENCY_STOP", fault=False, operational=False)
    )
    lifecycle._install_rizon_health_reporter(device, "RizonRobot")

    device._check_operational_status_change()

    health = events[-1][1]["health"]
    assert health["known"] is True
    assert health["estop"] is True
    assert health["operational"] is False
    assert health["fail_closed"] is True


def test_unknown_status_is_never_treated_as_healthy(monkeypatch: Any) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        lifecycle,
        "_send_ownership_event",
        lambda event, **details: events.append((event, details)),
    )
    device = _Device(_Handle("UNKNOWN", fault=False, operational=True))
    lifecycle._install_rizon_health_reporter(device, "RizonRobot")

    device._check_operational_status_change()

    health = events[-1][1]["health"]
    assert health["known"] is False
    assert health["fail_closed"] is True


def test_old_worker_generation_cannot_overwrite_new_health() -> None:
    manager = object.__new__(lifecycle.DeviceLifecycleManager)
    device = {
        "device_class": "RizonRobot",
        "device_class_id": 2,
        "ownership_token": "new-generation",
        "device_health": {"known": True, "estop": False},
        "device_health_received_at": time.monotonic(),
        "process": type("Process", (), {"pid": 222})(),
    }

    manager._apply_ownership_event(
        device,
        {
            "event": "device_health",
            "owner_token": "new-generation",
            "generation": "old-generation",
            "pid": 111,
            "health": {"known": True, "estop": True},
        },
    )

    assert device["device_health"] == {"known": True, "estop": False}


def test_progress_is_accepted_only_from_current_worker_generation() -> None:
    manager = object.__new__(lifecycle.DeviceLifecycleManager)
    device = {
        "device_class": "YamHeadRobot",
        "device_class_id": 0,
        "ownership_token": "current-generation",
        "progress": {"phase": "yam_connecting", "updated_at": None},
        "process": type("Process", (), {"pid": 222})(),
    }

    for generation, pid in (
        ("old-generation", 222),
        ("current-generation", 111),
    ):
        manager._apply_ownership_event(
            device,
            {
                "event": "device_progress",
                "owner_token": "current-generation",
                "generation": generation,
                "pid": pid,
                "phase": "yam_moving_init_pose",
            },
        )
        assert device["progress"]["phase"] == "yam_connecting"

    manager._apply_ownership_event(
        device,
        {
            "event": "device_progress",
            "owner_token": "current-generation",
            "generation": "current-generation",
            "pid": 222,
            "phase": "yam_moving_init_pose",
        },
    )

    assert device["progress"]["phase"] == "yam_moving_init_pose"
    assert isinstance(device["progress"]["updated_at"], float)


def test_rizon_initialization_tracks_stream_readiness_until_health_evidence() -> None:
    manager = object.__new__(lifecycle.DeviceLifecycleManager)
    manager.config = {
        "global_settings": {
            "device_health_stale_after": 1.5,
        }
    }

    safety = manager._device_safety_status(
        {
            "device_class": "RizonRobot",
            "device_health": None,
            "device_health_received_at": None,
        },
        alive=True,
        stream_ready=False,
    )

    assert safety["known"] is False
    assert safety["initializing"] is True
    assert safety["fail_closed"] is True

    after_first_frame = manager._device_safety_status(
        {
            "device_class": "RizonRobot",
            "device_health": None,
            "device_health_received_at": None,
        },
        alive=True,
        stream_ready=True,
    )

    assert after_first_frame["initializing"] is False
    assert after_first_frame["reason"] == "Rizon health heartbeat is unavailable"
