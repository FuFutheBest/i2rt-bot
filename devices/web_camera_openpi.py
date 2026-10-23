#!/usr/bin/env python3
"""Compatibility facade for the OpenPI web camera implementation.

Author: SuperInference
"""

# The exported class is provided dynamically by ``__getattr__``.
# ruff: noqa: F822

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["WebCameraOpenPi"]


def __getattr__(name: str) -> Any:
    """Resolve ``WebCameraOpenPi`` from its camera package implementation."""
    if name != "WebCameraOpenPi":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module("devices.cameras.openpi"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
