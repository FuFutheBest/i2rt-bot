"""Keep optional visualizer dependencies out of the core manager import path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_base_policy_runner_import_does_not_require_visualizer_stack() -> None:
    """A hardware-free manager import must not require matplotlib/OpenCV/Plotly."""
    script = """
import builtins

blocked = {"cv2", "matplotlib", "plotly", "visualizers"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from manager.base_policy_runner import BasePolicyRunner
runner = BasePolicyRunner(
    config={"policy": {"params": {}}},
    signal_files={"policy_runner_reset_command": "/tmp/superinference-test-reset"},
)
assert BasePolicyRunner.__name__ == "BasePolicyRunner"
assert type(runner.action_visualizer).__name__ == "_DisabledActionVisualizer"
assert runner.action_visualizer.enable is False
runner.action_visualizer.update()
runner.stop()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_device_starter_import_does_not_require_visualizer_stack() -> None:
    """The default device starter path must stay usable without plotting packages."""
    script = """
import builtins

blocked = {"cv2", "matplotlib", "plotly", "visualizers"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import device_starter
assert device_starter.DeviceStarter.__name__ == "DeviceStarter"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_windows_device_starter_import_does_not_require_visualizer_stack() -> None:
    """The Windows spawn entrypoint must not load plotting packages at import time."""
    script = """
import builtins

blocked = {"cv2", "matplotlib", "plotly", "visualizers"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import windows_device_start
assert windows_device_start.DeviceStarter.__name__ == "DeviceStarter"
assert hasattr(windows_device_start, "DEVICE_CLASSES")
assert hasattr(windows_device_start, "MANAGER_CLASSES")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
