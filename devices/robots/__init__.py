"""Lazy exports for robot device implementations.

Author: Jun Lv, Zixi Ying, SuperInference
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "BaseRobot": ("devices.robots.base", "BaseRobot"),
    "SimRobot": ("devices.robots.sim", "SimRobot"),
    "DualArmSimRobot": ("devices.robots.sim", "DualArmSimRobot"),
    "RizonRobot": ("devices.robots.rizon", "RizonRobot"),
    "I2RTRobot": ("devices.robots.i2rt", "I2RTRobot"),
    "X1Robot": ("incubator.teleoperation.x1.x1_robot", "X1Robot"),
    "RizonRobotLKForceGated": ("incubator.r1.r1_rizon", "RizonRobotLKForceGated"),
    "YamHeadRobot": ("incubator.r1.r1_rizon", "YamHeadRobot"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a robot implementation only when it is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""
    return sorted((*globals(), *_EXPORTS))
