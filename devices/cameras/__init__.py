"""Lazy exports for camera device implementations.

Author: Jun Lv, SuperInference
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "OpenCVCameraDevice": ("devices.cameras.realsense", "OpenCVCameraDevice"),
    "HikCameraDevice": ("devices.cameras.hik", "HikCameraDevice"),
    "WebCameraDevice": ("devices.cameras.webcam", "WebCameraDevice"),
    "WebCameraOpenPi": ("devices.cameras.openpi", "WebCameraOpenPi"),
    "ZedStereoDevice": ("devices.cameras.zed", "ZedStereoDevice"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a camera implementation only when it is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""
    return sorted((*globals(), *_EXPORTS))
