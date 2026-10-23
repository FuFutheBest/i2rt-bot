"""Shutdown ordering contracts for ``BaseDeviceManager``."""

from __future__ import annotations

import signal
import threading

from manager.base_device_manager import BaseDeviceManager


class _TrackedSharedMemory:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def close(self) -> None:
        self._events.append(f"{self._name}.close")

    def unlink(self) -> None:
        self._events.append(f"{self._name}.unlink")


def test_signal_defers_shared_memory_cleanup_until_active_update_returns() -> None:
    """A signal must not release a view while the interrupted update still uses it."""
    events: list[str] = []
    manager = object.__new__(BaseDeviceManager)
    manager.running = True
    manager._shutdown_complete = False
    manager._shm_access_lock = threading.RLock()
    manager.devices = [{"shm": _TrackedSharedMemory(events, "device")}]
    manager.summary_shm = _TrackedSharedMemory(events, "summary")

    def update() -> None:
        events.append("update.begin")
        manager._signal_handler(signal.SIGINT, None)
        assert events == ["update.begin"]
        events.append("update.end")

    manager._update_summary_shm = update

    manager.start()

    assert events == [
        "update.begin",
        "update.end",
        "device.close",
        "summary.close",
        "summary.unlink",
    ]
    assert manager._shutdown_complete
    assert manager.devices[0]["shm"] is None
    assert manager.summary_shm is None


def test_stop_is_idempotent_after_signal_requested_shutdown() -> None:
    events: list[str] = []
    manager = object.__new__(BaseDeviceManager)
    manager.running = True
    manager._shutdown_complete = False
    manager._shm_access_lock = threading.RLock()
    manager.devices = [{"shm": _TrackedSharedMemory(events, "device")}]
    manager.summary_shm = _TrackedSharedMemory(events, "summary")

    manager._signal_handler(signal.SIGTERM, None)
    manager.stop()
    manager.stop()

    assert events == ["device.close", "summary.close", "summary.unlink"]
