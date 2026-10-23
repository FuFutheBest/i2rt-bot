"""Contracts for the lazy device registry and compatibility facades.

Most checks are hardware-free; optional OpenCV adapter/import checks are
explicitly marked for the hardware profile.

Author: SuperInference
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import devices
import numpy as np
import pytest
import yaml
from devices.contracts import DeviceKind


EXPECTED_DEVICE_CLASSES = {
    "BaseDevice",
    "OpenCVCameraDevice",
    "WebCameraDevice",
    "WebCameraOpenPi",
    "IPhoneCameraDevice",
    "RealSense",
    "HikCameraDevice",
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
    "ViveTrackerDevice",
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
}


def _active_device_classes(repo_root: Path) -> set[str]:
    """Collect enabled device class names from device presets."""
    classes: set[str] = set()
    for path in sorted((repo_root / "configs" / "devices").rglob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            continue
        entries = config.get("devices", [])
        if isinstance(entries, dict):
            entries = entries.get("devices", [])
        for entry in entries or []:
            if (
                isinstance(entry, dict)
                and entry.get("enabled", True) is not False
                and isinstance(entry.get("class"), str)
            ):
                classes.add(entry["class"])
    return classes


def _module_source_paths(repo_root: Path, module_name: str) -> list[Path]:
    """Return source files for a dotted module without importing it."""
    module_path = repo_root.joinpath(*module_name.split("."))
    source_path = module_path.with_suffix(".py")
    if source_path.is_file():
        return [source_path]
    if module_path.is_dir():
        return sorted(module_path.rglob("*.py"))
    return []


def _source_declares_symbol(path: Path, symbol: str) -> bool:
    """Check an adapter source tree for a class/export without loading SDKs."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            aliases = node.names
            if any((alias.asname or alias.name.rsplit(".", maxsplit=1)[-1]) == symbol for alias in aliases):
                return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == symbol for target in targets):
                return True
    return False


def test_registry_shape_does_not_resolve_entries(monkeypatch: Any) -> None:
    """Membership and iteration must not import implementation modules."""
    imported: list[str] = []

    def fail_on_import(module_name: str) -> ModuleType:
        imported.append(module_name)
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(devices.importlib, "import_module", fail_on_import)

    assert set(devices.DEVICE_CLASSES) == EXPECTED_DEVICE_CLASSES
    assert len(devices.DEVICE_CLASSES) == len(EXPECTED_DEVICE_CLASSES)
    assert "RizonRobot" in devices.DEVICE_CLASSES
    assert devices.DEVICE_CLASSES.get("UnknownDevice") is None
    assert imported == []


def test_registry_exposes_device_kinds_without_importing_adapters(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        devices.importlib,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(
            AssertionError(f"unexpected import: {module_name}")
        ),
    )

    assert devices.DEVICE_CLASSES.kind("WebCameraOpenPi") is DeviceKind.CAMERA
    assert devices.DEVICE_CLASSES.kind("RizonRobot") is DeviceKind.LOW_DIM_RW
    assert (
        devices.DEVICE_CLASSES.kind("I2RTBimanualTeleop")
        is DeviceKind.LOW_DIM_R
    )
    assert devices.DEVICE_CLASSES.kind("YamHeadRobot") is DeviceKind.LOW_DIM_R


def test_registry_resolves_and_caches_one_spec(monkeypatch: Any) -> None:
    """A lookup resolves exactly one module and caches the resulting class."""
    module = ModuleType("test_device_adapter")

    class TestDevice:
        pass

    module.TestDevice = TestDevice
    registry = devices.LazyDeviceRegistry(
        {"TestDevice": ("test_device_adapter", "TestDevice")}
    )
    imports: list[str] = []

    def import_module(module_name: str) -> ModuleType:
        imports.append(module_name)
        return module

    monkeypatch.setattr(devices.importlib, "import_module", import_module)

    assert registry["TestDevice"] is TestDevice
    assert registry.get("TestDevice") is TestDevice
    assert imports == ["test_device_adapter"]


def test_legacy_facades_are_lazy(repo_root: Path) -> None:
    """Importing compatibility paths must not load implementations in a clean process."""
    script = """
import importlib
import sys

camera = importlib.import_module("devices.camera")
robot = importlib.import_module("devices.robot")
openpi = importlib.import_module("devices.web_camera_openpi")

assert "OpenCVCameraDevice" in camera.__all__
assert "RizonRobot" in robot.__all__
assert openpi.__all__ == ["WebCameraOpenPi"]
assert "devices.cameras.realsense" not in sys.modules
assert "devices.cameras.openpi" not in sys.modules
assert "devices.robots.rizon" not in sys.modules
assert "devices.robots.i2rt" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_hik_package_import_is_sdk_lazy() -> None:
    """The public Hik package must not load its implementation or vendor SDK."""
    for module_name in (
        "devices.cameras.hik",
        "devices.cameras.hik.camera",
        "devices.cameras.hik.sdk",
    ):
        sys.modules.pop(module_name, None)

    importlib.import_module("devices.cameras.hik")

    assert "devices.cameras.hik.camera" not in sys.modules
    assert "devices.cameras.hik.sdk" not in sys.modules

    sdk = importlib.import_module("devices.cameras.hik.sdk")

    assert sdk._SDK_LOADED is False
    assert "third_party.MvImport.MvCameraControl_class" not in sys.modules


def test_zed_normalizes_numpy_dtype_for_shm_layout() -> None:
    """ZED's SHM byte layout must match frames when dtype is a NumPy scalar."""
    from devices.cameras.zed import ZedStereoDevice

    device = ZedStereoDevice(
        state_feedback={
            "fps": 30.0,
            "buffer_size": 2,
            "hardware_latency_ms": 0.0,
            "format": {"shape": [2, 4, 3], "dtype": "uint8", "type": "rgb"},
        },
    )

    assert device.data_dtype == "uint8"
    assert device._stereo_rgb.dtype == np.dtype("uint8")
    assert device.frame_data_size == 2 * 4 * 3

    with pytest.raises(ValueError, match="Unsupported ZED state_feedback.format.dtype 'int8'"):
        ZedStereoDevice(
            state_feedback={
                "fps": 30.0,
                "buffer_size": 2,
                "hardware_latency_ms": 0.0,
                "format": {"shape": [2, 4, 3], "dtype": "int8", "type": "rgb"},
            },
        )


@pytest.mark.hardware
def test_webcam_lookup_isolated_from_camera_sdks() -> None:
    """Resolving a webcam must not import RealSense or Hikvision adapters."""
    pytest.importorskip(
        "cv2",
        reason="webcam registry resolution requires the optional OpenCV hardware profile",
    )
    devices.DEVICE_CLASSES._cache.pop("WebCameraDevice", None)
    for module_name in (
        "devices.cameras.webcam",
        "devices.cameras.realsense",
        "devices.cameras.hik",
    ):
        sys.modules.pop(module_name, None)

    webcam_class = devices.DEVICE_CLASSES["WebCameraDevice"]

    assert webcam_class.__module__ == "devices.cameras.webcam"
    assert "devices.cameras.webcam" in sys.modules
    assert "devices.cameras.realsense" not in sys.modules
    assert "devices.cameras.hik" not in sys.modules


@pytest.mark.hardware
def test_legacy_exports_match_new_package_classes() -> None:
    """Old import paths and new package paths must expose identical classes."""
    pytest.importorskip(
        "cv2",
        reason="camera facade identity requires the optional OpenCV hardware profile",
    )
    from devices.cameras.openpi import WebCameraOpenPi as NewWebCameraOpenPi
    from devices.cameras.hik.camera import HikCameraDevice as NewHikCameraDevice
    from devices.camera import HikCameraDevice as LegacyHikCameraDevice
    from devices.camera import OpenCVCameraDevice as LegacyOpenCVCameraDevice
    from devices.cameras.realsense import OpenCVCameraDevice as NewOpenCVCameraDevice
    from devices.robot import I2RTRobot as LegacyI2RTRobot
    from devices.robot import RizonRobot as LegacyRizonRobot
    from devices.robots.i2rt import I2RTRobot as NewI2RTRobot
    from devices.robots.rizon import RizonRobot as NewRizonRobot
    from devices.web_camera_openpi import WebCameraOpenPi as LegacyWebCameraOpenPi

    assert LegacyWebCameraOpenPi is NewWebCameraOpenPi
    assert LegacyHikCameraDevice is NewHikCameraDevice
    assert LegacyOpenCVCameraDevice is NewOpenCVCameraDevice
    assert LegacyRizonRobot is NewRizonRobot
    assert LegacyI2RTRobot is NewI2RTRobot


def test_enabled_config_classes_have_registry_specs(repo_root: Path) -> None:
    """Enabled presets point to valid registry specs without importing optional SDKs."""
    specs = devices.DEVICE_CLASSES._specs
    failures: list[str] = []
    for class_name in sorted(_active_device_classes(repo_root)):
        spec = specs.get(class_name)
        if spec is None:
            failures.append(f"{class_name}: missing registry entry")
            continue
        module_name, symbol = spec
        source_paths = _module_source_paths(repo_root, module_name)
        if not source_paths:
            failures.append(f"{class_name}: module source not found ({module_name})")
        elif not any(_source_declares_symbol(path, symbol) for path in source_paths):
            failures.append(f"{class_name}: symbol {symbol} not declared in {module_name}")

    assert failures == []


def test_import_devices_without_vendor_sdks(repo_root: Path) -> None:
    """A clean interpreter can import the registry while all SDK imports fail."""
    script = """
import builtins
import importlib

blocked_roots = {
    "cv2",
    "flexiv_rdk_ext",
    "flexivrdk",
    "pybullet",
    "pyrealsense2",
    "rerun",
}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked_roots or name.startswith("third_party.MvImport"):
        raise RuntimeError(f"vendor SDK imported eagerly: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
devices = importlib.import_module("devices")
importlib.import_module("devices.camera")
importlib.import_module("devices.robot")
importlib.import_module("devices.web_camera_openpi")
assert "RizonRobot" in devices.DEVICE_CLASSES
assert set(devices.DEVICE_CLASSES)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.hardware
def test_optional_adapter_classes_import_without_vendor_sdks(repo_root: Path) -> None:
    """Adapter class definitions remain importable when vendor SDKs are absent."""
    pytest.importorskip(
        "cv2",
        reason="optional adapter import contract requires the hardware OpenCV profile",
    )
    script = """
import builtins
import importlib

blocked = {"flexiv_rdk_ext", "flexivrdk", "pyrealsense2"}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked or name.startswith("third_party.MvImport"):
        raise ImportError(f"blocked optional SDK: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from devices.cameras.hik import HikCameraDevice
from devices.cameras.realsense import OpenCVCameraDevice
from devices.robots.rizon import RizonRobot
assert isinstance(HikCameraDevice, type)
assert isinstance(OpenCVCameraDevice, type)
assert isinstance(RizonRobot, type)
assert importlib.import_module("devices.robot").RizonRobot is RizonRobot
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
