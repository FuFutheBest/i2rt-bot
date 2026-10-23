#!/usr/bin/env python3
"""Helpers for NEDF3 metadata.extra camera fields."""

from __future__ import annotations

METADATA_CAMERA_COUNT = 3

DEFAULT_DEVICE_TYPE = "robopocket_2.0_pro"
DEFAULT_HARDWARE_STRING = "310"


def expand_triple_csv(value: str, *, default: str = "") -> str:
    """Replicate one value for camera_0..2, or validate three explicit values."""
    raw = (value or default).strip()
    if not raw:
        return ""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) == 1:
        return ",".join([parts[0]] * METADATA_CAMERA_COUNT)
    if len(parts) != METADATA_CAMERA_COUNT:
        raise ValueError(
            f"需要 1 个或 {METADATA_CAMERA_COUNT} 个值，当前为 {len(parts)} 个。"
        )
    return ",".join(parts)

