"""Low-overhead rollout storage budgeting for the control panel."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Optional


MIB = 1024**2
GIB = 1024**3
DEFAULT_BYTES_PER_SECOND = 12 * MIB
MIN_BYTES_PER_SECOND = 4 * MIB
NON_CAMERA_BYTES_PER_SECOND = 512 * 1024
ESTIMATED_JPEG_BYTES_PER_PIXEL = 0.5
MIN_CAMERA_BYTES_PER_FRAME = 32 * 1024
MIN_RESERVE_BYTES = 10 * GIB
RESERVE_FRACTION = 0.02
WARNING_FREE_FRACTION = 0.05
WARNING_FREE_BYTES = 50 * GIB
WARNING_REMAINING_SECONDS = 2 * 60 * 60
CRITICAL_REMAINING_SECONDS = 30 * 60
REFRESH_INTERVAL_SECONDS = 2.0
MAX_OBSERVED_BYTES_PER_SECOND = 2 * GIB


def _positive_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _device_entries(config: Optional[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not isinstance(config, Mapping):
        return []
    devices: Any = config.get("devices", [])
    if isinstance(devices, Mapping):
        devices = devices.get("devices", [])
    if not isinstance(devices, list):
        return []
    return [entry for entry in devices if isinstance(entry, Mapping)]


def estimate_configured_bytes_per_second(
    config: Optional[Mapping[str, Any]],
) -> int:
    """Estimate a conservative NEDF camera write rate from the active config."""
    camera_rate = 0.0
    camera_count = 0
    for device in _device_entries(config):
        if device.get("enabled") is False:
            continue
        class_name = str(device.get("class") or "").casefold()
        if "camera" not in class_name and "realsense" not in class_name:
            continue
        camera_count += 1
        state_feedback = device.get("state_feedback")
        state_feedback = state_feedback if isinstance(state_feedback, Mapping) else {}
        state_format = state_feedback.get("format")
        state_format = state_format if isinstance(state_format, Mapping) else {}
        fps = _positive_number(
            state_feedback.get("fps", device.get("fps"))
        ) or 30.0
        shape = state_format.get(
            "shape", device.get("data_shape", device.get("shape"))
        )
        pixels: Optional[float] = None
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            height = _positive_number(shape[0])
            width = _positive_number(shape[1])
            if height is not None and width is not None:
                pixels = height * width
        bytes_per_frame = (
            max(
                float(MIN_CAMERA_BYTES_PER_FRAME),
                pixels * ESTIMATED_JPEG_BYTES_PER_PIXEL,
            )
            if pixels is not None
            else 128 * 1024
        )
        camera_rate += bytes_per_frame * fps

    if camera_count == 0:
        return DEFAULT_BYTES_PER_SECOND
    return int(max(MIN_BYTES_PER_SECOND, camera_rate + NON_CAMERA_BYTES_PER_SECOND))


class RolloutStorageMonitor:
    """Cache disk usage and learn a conservative live rollout write rate."""

    def __init__(
        self,
        *,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        clock: Callable[[], float] = time.monotonic,
        refresh_interval_s: float = REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self._disk_usage = disk_usage
        self._clock = clock
        self._refresh_interval_s = max(0.0, float(refresh_interval_s))
        self._cache_key: Optional[tuple[str, bool]] = None
        self._cached_at = 0.0
        self._cached: Optional[dict[str, Any]] = None
        self._sample_path: Optional[str] = None
        self._sample_time: Optional[float] = None
        self._sample_free: Optional[int] = None
        self._sample_active = False
        self._observed_rate: Optional[float] = None
        self._observed_samples = 0

    def _reset_samples(self, path: Optional[str] = None) -> None:
        self._sample_path = path
        self._sample_time = None
        self._sample_free = None
        self._sample_active = False
        self._observed_rate = None
        self._observed_samples = 0

    def _observe_rate(
        self,
        *,
        path: str,
        now: float,
        free: int,
        active: bool,
    ) -> None:
        if path != self._sample_path:
            self._reset_samples(path)
        if (
            active
            and self._sample_active
            and self._sample_time is not None
            and self._sample_free is not None
        ):
            elapsed = now - self._sample_time
            consumed = self._sample_free - free
            if elapsed >= self._refresh_interval_s and consumed > 0:
                rate = consumed / elapsed
                if MIN_BYTES_PER_SECOND / 16 <= rate <= MAX_OBSERVED_BYTES_PER_SECOND:
                    if self._observed_rate is None:
                        self._observed_rate = rate
                    else:
                        self._observed_rate = self._observed_rate * 0.75 + rate * 0.25
                    self._observed_samples += 1
        self._sample_time = now
        self._sample_free = free
        self._sample_active = active

    def snapshot(
        self,
        *,
        output_root: Optional[Path],
        enabled: bool,
        config: Optional[Mapping[str, Any]],
        recording_state: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if not enabled:
            self._cache_key = None
            self._cached = None
            self._reset_samples()
            return {
                "schema_version": 1,
                "enabled": False,
                "status": "disabled",
            }
        if output_root is None:
            self._reset_samples()
            return {
                "schema_version": 1,
                "enabled": True,
                "status": "unavailable",
                "error": "Rollout save path is unavailable",
            }

        path = str(Path(output_root).expanduser().resolve(strict=False))
        now = self._clock()
        cache_key = (path, True)
        if (
            self._cache_key == cache_key
            and self._cached is not None
            and now - self._cached_at < self._refresh_interval_s
        ):
            return dict(self._cached)

        try:
            usage = self._disk_usage(Path(path))
            total = max(0, int(usage.total))
            free = max(0, int(usage.free))
        except OSError as exc:
            payload = {
                "schema_version": 1,
                "enabled": True,
                "status": "unavailable",
                "path": path,
                "error": f"Unable to read rollout disk usage: {exc}",
            }
            self._cache_key = cache_key
            self._cached_at = now
            self._cached = payload
            return dict(payload)

        active = bool(
            isinstance(recording_state, Mapping)
            and recording_state.get("active") is True
        )
        self._observe_rate(path=path, now=now, free=free, active=active)
        configured_rate = estimate_configured_bytes_per_second(config)
        observed_rate = int(self._observed_rate or 0)
        estimated_rate = max(configured_rate, observed_rate)
        reserve = max(MIN_RESERVE_BYTES, int(total * RESERVE_FRACTION))
        usable = max(0, free - reserve)
        remaining = int(usable / estimated_rate) if estimated_rate > 0 else None
        free_fraction = free / total if total > 0 else 0.0

        if usable <= 0 or (
            remaining is not None and remaining <= CRITICAL_REMAINING_SECONDS
        ):
            status = "critical"
        elif (
            free_fraction <= WARNING_FREE_FRACTION
            or free <= WARNING_FREE_BYTES
            or (remaining is not None and remaining <= WARNING_REMAINING_SECONDS)
        ):
            status = "warning"
        else:
            status = "healthy"

        payload = {
            "schema_version": 1,
            "enabled": True,
            "status": status,
            "path": path,
            "total_bytes": total,
            "free_bytes": free,
            "reserve_bytes": reserve,
            "usable_bytes": usable,
            "free_fraction": free_fraction,
            "estimated_bytes_per_second": estimated_rate,
            "configured_bytes_per_second": configured_rate,
            "observed_bytes_per_second": (
                observed_rate if self._observed_samples > 0 else None
            ),
            "observed_samples": self._observed_samples,
            "estimate_source": (
                "observed" if observed_rate > configured_rate else "configuration"
            ),
            "estimated_remaining_seconds": remaining,
            "recording_active": active,
            "sampled_at_monotonic": now,
        }
        self._cache_key = cache_key
        self._cached_at = now
        self._cached = payload
        return dict(payload)
