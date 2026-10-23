#!/usr/bin/env python3
"""Compatibility facade for camera device implementations.

Author: Jun Lv, SuperInference
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "OpenCVCameraDevice": ("devices.cameras.realsense", "OpenCVCameraDevice"),
    "HikCameraDevice": ("devices.cameras.hik", "HikCameraDevice"),
    "WebCameraDevice": ("devices.cameras.webcam", "WebCameraDevice"),
    "ZedStereoDevice": ("devices.cameras.zed", "ZedStereoDevice"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve legacy camera exports without eager SDK imports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LAZY_EXPORTS))


def main() -> None:
    """Preserve the legacy ``python -m devices.camera`` entry point."""
    from devices.cameras.realsense import main as implementation_main

    implementation_main()


if __name__ == "__main__":
    main()
