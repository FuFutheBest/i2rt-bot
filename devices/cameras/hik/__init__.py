#!/usr/bin/env python3
"""Hikvision camera adapter package.

Author: OpenAI Assistant
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["HikCameraDevice"]


def __getattr__(name: str) -> Any:
    """Lazily expose HikCameraDevice so importing the package stays SDK-free."""
    if name != "HikCameraDevice":
        raise AttributeError(f"module 'devices.cameras.hik' has no attribute {name!r}")
    module = importlib.import_module("devices.cameras.hik.camera")
    value = getattr(module, name)
    globals()[name] = value
    return value
