"""Shared device capability contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any


class DeviceKind(str, Enum):
    """I/O capability relative to SuperInference state/control channels."""

    CAMERA = "camera"
    LOW_DIM_R = "low_dim_r"
    LOW_DIM_RW = "low_dim_rw"
    LOW_DIM_W = "low_dim_w"

    @property
    def readable(self) -> bool:
        return self in {self.CAMERA, self.LOW_DIM_R, self.LOW_DIM_RW}

    @property
    def writable(self) -> bool:
        return self in {self.LOW_DIM_RW, self.LOW_DIM_W}

    @property
    def low_dimensional(self) -> bool:
        return self is not self.CAMERA


def parse_device_kind(value: Any, *, context: str = "device_kind") -> DeviceKind:
    if isinstance(value, DeviceKind):
        return value
    try:
        return DeviceKind(str(value).strip().casefold())
    except ValueError as exc:
        choices = ", ".join(kind.value for kind in DeviceKind)
        raise ValueError(f"{context} must be one of: {choices}") from exc


__all__ = ["DeviceKind", "parse_device_kind"]
