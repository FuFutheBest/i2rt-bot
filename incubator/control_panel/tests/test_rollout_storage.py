"""Rollout disk budget tests without hardware or real filesystem pressure."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from incubator.control_panel.rollout_storage import (
    GIB,
    MIB,
    RolloutStorageMonitor,
    estimate_configured_bytes_per_second,
)


def _config() -> dict:
    return {
        "devices": [
            {
                "class": "WebCameraOpenPi",
                "enabled": True,
                "data_shape": [360, 640, 3],
                "fps": 30,
            },
            {
                "class": "WebCameraOpenPi",
                "enabled": True,
                "data_shape": [360, 640, 3],
                "fps": 30,
            },
            {
                "class": "WebCameraOpenPi",
                "enabled": True,
                "data_shape": [480, 640, 3],
                "fps": 30,
            },
            {"class": "RizonRobot", "data_shape": [44], "fps": 120},
        ]
    }


def test_config_estimate_counts_image_devices_only() -> None:
    rate = estimate_configured_bytes_per_second(_config())
    assert 10 * MIB < rate < 20 * MIB


def test_disabled_monitor_does_not_touch_disk(tmp_path: Path) -> None:
    def fail_disk_usage(path: Path) -> None:
        raise AssertionError(f"unexpected disk read: {path}")

    monitor = RolloutStorageMonitor(disk_usage=fail_disk_usage)
    assert monitor.snapshot(
        output_root=tmp_path,
        enabled=False,
        config=_config(),
    ) == {"schema_version": 1, "enabled": False, "status": "disabled"}


def test_budget_reports_remaining_time_and_reserve(tmp_path: Path) -> None:
    monitor = RolloutStorageMonitor(
        disk_usage=lambda path: SimpleNamespace(
            total=1024 * GIB,
            used=424 * GIB,
            free=600 * GIB,
        ),
        clock=lambda: 10.0,
    )
    payload = monitor.snapshot(
        output_root=tmp_path,
        enabled=True,
        config=_config(),
    )
    assert payload["status"] == "healthy"
    assert payload["free_bytes"] == 600 * GIB
    assert payload["reserve_bytes"] == int(1024 * GIB * 0.02)
    assert payload["usable_bytes"] == payload["free_bytes"] - payload["reserve_bytes"]
    assert payload["estimated_remaining_seconds"] > 10 * 60 * 60
    assert payload["estimate_source"] == "configuration"


def test_low_space_is_critical_and_visible_in_payload(tmp_path: Path) -> None:
    monitor = RolloutStorageMonitor(
        disk_usage=lambda path: SimpleNamespace(
            total=1024 * GIB,
            used=1018 * GIB,
            free=6 * GIB,
        )
    )
    payload = monitor.snapshot(
        output_root=tmp_path,
        enabled=True,
        config=_config(),
    )
    assert payload["status"] == "critical"
    assert payload["usable_bytes"] == 0
    assert payload["estimated_remaining_seconds"] == 0


def test_live_consumption_raises_conservative_rate(tmp_path: Path) -> None:
    samples = iter(
        [
            600 * GIB,
            600 * GIB - 80 * MIB,
            600 * GIB - 160 * MIB,
            600 * GIB - 240 * MIB,
        ]
    )
    now = [0.0]

    def disk_usage(path: Path) -> SimpleNamespace:
        return SimpleNamespace(total=1024 * GIB, used=0, free=next(samples))

    monitor = RolloutStorageMonitor(
        disk_usage=disk_usage,
        clock=lambda: now[0],
        refresh_interval_s=2.0,
    )
    payload = None
    for step in range(4):
        now[0] = step * 2.0
        payload = monitor.snapshot(
            output_root=tmp_path,
            enabled=True,
            config=_config(),
            recording_state={"active": True},
        )
    assert payload is not None
    assert payload["observed_samples"] == 3
    assert payload["observed_bytes_per_second"] == 40 * MIB
    assert payload["estimated_bytes_per_second"] == 40 * MIB
    assert payload["estimate_source"] == "observed"


def test_disk_error_fails_open_with_unavailable_status(tmp_path: Path) -> None:
    def disk_usage(path: Path) -> None:
        raise OSError("mount disappeared")

    monitor = RolloutStorageMonitor(disk_usage=disk_usage)
    payload = monitor.snapshot(
        output_root=tmp_path,
        enabled=True,
        config=_config(),
    )
    assert payload["status"] == "unavailable"
    assert "mount disappeared" in payload["error"]
