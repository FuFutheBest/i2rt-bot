"""Lazy device registry for hardware-backed implementations.

Importing :mod:`devices` intentionally imports no camera or robot SDK. Device
classes are resolved only when a registry entry or package attribute is used.

Author: Jun Lv, Han Xue, SuperInference
"""

# Lazy module attributes intentionally have no eager assignments.
# ruff: noqa: F822

from __future__ import annotations

import importlib
from collections.abc import Iterator, MutableMapping
from typing import Any

from devices.contracts import DeviceKind, parse_device_kind

DeviceClassSpec = tuple[str, str] | tuple[str, str, DeviceKind]


class LazyDeviceRegistry(MutableMapping[str, Any]):
    """Dictionary-compatible registry backed by import specifications."""

    def __init__(self, specs: dict[str, DeviceClassSpec]) -> None:
        self._specs: dict[str, tuple[str, str]] = {}
        self._kinds: dict[str, DeviceKind] = {}
        for name, spec in specs.items():
            if len(spec) == 2:
                module_name, attribute_name = spec
                kind = DeviceKind.LOW_DIM_R
            else:
                module_name, attribute_name, kind = spec
            self._specs[name] = (module_name, attribute_name)
            self._kinds[name] = parse_device_kind(kind)
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        try:
            module_name, attribute_name = self._specs[key]
        except KeyError:
            raise KeyError(key) from None
        value = getattr(importlib.import_module(module_name), attribute_name)
        registered_kind = self._kinds[key]
        class_kind = parse_device_kind(
            getattr(value, "device_kind", registered_kind),
            context=f"{key}.device_kind",
        )
        if class_kind is not registered_kind:
            raise TypeError(
                f"{key}.device_kind={class_kind.value!r} conflicts with registry "
                f"kind {registered_kind.value!r}"
            )
        self._cache[key] = value
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._specs.setdefault(key, ("", ""))
        self._kinds[key] = parse_device_kind(
            getattr(value, "device_kind", DeviceKind.LOW_DIM_R)
        )

    def __delitem__(self, key: str) -> None:
        if key not in self._specs:
            raise KeyError(key)
        del self._specs[key]
        self._kinds.pop(key, None)
        self._cache.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._specs

    def get(self, key: str, default: Any = None) -> Any:
        """Resolve known keys while retaining normal ``dict.get`` behavior."""
        return self[key] if key in self._specs else default

    def kind(self, key: str) -> DeviceKind:
        """Return registered capability metadata without importing the class."""
        try:
            return self._kinds[key]
        except KeyError:
            raise KeyError(key) from None


DEVICE_CLASSES = {
    "BaseDevice": ("devices.base", "BaseDevice", DeviceKind.LOW_DIM_R),
    "OpenCVCameraDevice": ("devices.cameras.realsense", "OpenCVCameraDevice", DeviceKind.CAMERA),
    "WebCameraDevice": ("devices.cameras.webcam", "WebCameraDevice", DeviceKind.CAMERA),
    "WebCameraOpenPi": ("devices.cameras.openpi", "WebCameraOpenPi", DeviceKind.CAMERA),
    "IPhoneCameraDevice": ("devices.iphone", "IPhoneCameraDevice", DeviceKind.CAMERA),
    "RealSense": ("devices.realsense", "RealSenseCameraDevice", DeviceKind.CAMERA),
    "HikCameraDevice": ("devices.cameras.hik", "HikCameraDevice", DeviceKind.CAMERA),
    "BaseRobot": ("devices.robots.base", "BaseRobot", DeviceKind.LOW_DIM_RW),
    "SimRobot": ("devices.robots.sim", "SimRobot", DeviceKind.LOW_DIM_RW),
    "DualArmSimRobot": ("devices.robots.sim", "DualArmSimRobot", DeviceKind.LOW_DIM_RW),
    "RoboTwinSimRobot": ("devices.robots.robotwin", "RoboTwinSimRobot", DeviceKind.LOW_DIM_RW),
    "RoboTwinCamera": ("devices.robots.robotwin", "RoboTwinCamera", DeviceKind.CAMERA),
    "JointSlider": ("devices.teleoperator", "JointSlider"),
    "DualArmJointSlider": ("devices.teleoperator", "DualArmJointSlider"),
    "Airexoskeleton": ("devices.teleoperator", "Airexoskeleton"),
    "DualAirexoskeleton": ("devices.teleoperator", "DualAirexoskeleton"),
    "KeyboardTeleoperator": ("devices.teleoperator", "KeyboardTeleoperator"),
    "Quest3Teleoperator": ("devices.teleoperator", "Quest3Teleoperator"),
    "ViveTrackerDevice": ("devices.vive_tracker", "ViveTrackerDevice"),
    "RizonRobot": ("devices.robots.rizon", "RizonRobot", DeviceKind.LOW_DIM_RW),
    "RotaryEncoderDevice": ("devices.rotary_encoder", "RotaryEncoderDevice"),
    "I2RTRobot": ("devices.robots.i2rt", "I2RTRobot", DeviceKind.LOW_DIM_RW),
    "I2RTBimanualTeleop": (
        "devices.robots.i2rt_bimanual",
        "I2RTBimanualTeleop",
        DeviceKind.LOW_DIM_R,
    ),
    "ZedStereoDevice": ("devices.cameras.zed", "ZedStereoDevice", DeviceKind.CAMERA),
    "X1Robot": ("incubator.teleoperation.x1.x1_robot", "X1Robot", DeviceKind.LOW_DIM_RW),
    "X1VuerTeleoperator": ("incubator.teleoperation.x1.x1_teleoperator", "X1VuerTeleoperator"),
    "TjCameraDevice": ("incubator.teleoperation.x1.tj_camera", "TjCameraDevice", DeviceKind.CAMERA),
    "Quest3VuerTeleoperator": ("incubator.r1.r1_teleoperator", "Quest3VuerTeleoperator"),
    "ReplayTeleoperator": ("incubator.r1.r1_teleoperator", "ReplayTeleoperator"),
    "RizonRobotLKForceGated": ("incubator.r1.r1_rizon", "RizonRobotLKForceGated", DeviceKind.LOW_DIM_RW),
    "YamHeadRobot": ("incubator.r1.r1_rizon", "YamHeadRobot", DeviceKind.LOW_DIM_R),
}

DEVICE_CLASSES = LazyDeviceRegistry(DEVICE_CLASSES)

__all__ = [
    "BaseDevice",
    "ViveTrackerDevice",
    "OpenCVCameraDevice",
    "WebCameraDevice",
    "WebCameraOpenPi",
    "IPhoneCameraDevice",
    "HikCameraDevice",
    "RealSense",
    "BaseRobot",
    "SimRobot",
    "DualArmSimRobot",
    "RoboTwinSimRobot",
    "RoboTwinCamera",
    "JointSlider",
    "DualArmJointSlider",
    "Airexoskeleton",
    "DualAirexoskeleton",
    "KeyboardTeleoperator",
    "Quest3Teleoperator",
    "RizonRobot",
    "RotaryEncoderDevice",
    "I2RTRobot",
    "I2RTBimanualTeleop",
    "ZedStereoDevice",
    "X1Robot",
    "X1VuerTeleoperator",
    "TjCameraDevice",
    "Quest3VuerTeleoperator",
    "ReplayTeleoperator",
    "RizonRobotLKForceGated",
    "YamHeadRobot",
    "DEVICE_CLASSES",
    "LazyDeviceRegistry",
    "DeviceKind",
]


def __getattr__(name: str) -> Any:
    """Lazily expose registered device classes as module attributes."""
    if name in DEVICE_CLASSES:
        value = DEVICE_CLASSES[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy device names in interactive discovery."""
    return sorted((*globals(), *DEVICE_CLASSES))
