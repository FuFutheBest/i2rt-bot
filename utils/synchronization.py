"""Shared configuration contract for device-stream synchronization modes."""

from __future__ import annotations

from typing import Any


BINARY_MATCH = "binary_match"
CAMERA_ALIGN = "camera_align"


class SynchronizationModeError(ValueError):
    """Raised when a device-manager synchronization mode is unsupported."""


_SYNC_MODE_ALIASES = {
    "binary": BINARY_MATCH,
    BINARY_MATCH: BINARY_MATCH,
    "camera": CAMERA_ALIGN,
    "camera_sync": CAMERA_ALIGN,
    CAMERA_ALIGN: CAMERA_ALIGN,
}


def normalize_sync_mode(value: Any = BINARY_MATCH) -> str:
    """Return the canonical synchronization mode for a config value."""
    if not isinstance(value, str):
        raise SynchronizationModeError(
            "sync_mode must be 'binary_match' or 'camera_align'"
        )
    normalized = _SYNC_MODE_ALIASES.get(value.strip().casefold())
    if normalized is None:
        raise SynchronizationModeError(
            "sync_mode must be 'binary_match' or 'camera_align'"
        )
    return normalized
