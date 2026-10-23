#!/usr/bin/env python3
"""Compatibility facade for robot device implementations.

Author: Jun Lv, Zixi Ying, SuperInference
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "BaseRobot": ("devices.robots.base", "BaseRobot"),
    "SimRobot": ("devices.robots.sim", "SimRobot"),
    "DualArmSimRobot": ("devices.robots.sim", "DualArmSimRobot"),
    "RizonRobot": ("devices.robots.rizon", "RizonRobot"),
    "I2RTRobot": ("devices.robots.i2rt", "I2RTRobot"),
    "X1Robot": ("incubator.teleoperation.x1.x1_robot", "X1Robot"),
    "RizonRobotLKForceGated": ("incubator.r1.r1_rizon", "RizonRobotLKForceGated"),
    "YamHeadRobot": ("incubator.r1.r1_rizon", "YamHeadRobot"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve legacy robot exports without eager implementation imports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LAZY_EXPORTS))


def main() -> None:
    """Preserve the legacy ``python -m devices.robot`` entry point."""
    from devices.robots.i2rt import real_main

    real_main()


if __name__ == "__main__":
    main()
