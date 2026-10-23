"""Hardware-free characterization tests for split device adapters.

Author: SuperInference Contributors
"""

from __future__ import annotations

import ast
from pathlib import Path


def _class_methods(path: Path, class_name: str) -> set[str]:
    """Return methods declared directly on a class without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _source(path: Path) -> str:
    """Read implementation source for string-level compatibility contracts."""
    return path.read_text(encoding="utf-8")


def test_rizon_split_preserves_public_surface(repo_root: Path) -> None:
    """Rizon's public lifecycle remains present across focused mixins."""
    rizon_root = repo_root / "devices" / "robots" / "rizon"
    methods: set[str] = set()
    for path in rizon_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods.update(
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )

    assert {
        "execute_control_command",
        "get_robot_state",
        "get_status",
        "start_server",
        "stop_server",
        "switch_to",
    } <= methods

    base_source = _source(repo_root / "devices" / "robots" / "base.py")
    home_source = _source(rizon_root / "home.py")
    gripper_source = _source(rizon_root / "gripper.py")
    assert 'f"{self.device_name}_{self.device_class_id}_control"' in base_source
    assert home_source.count("self._clear_control_shm()") >= 2
    assert "np.clip(target_width, min_width, float(max_width))" in gripper_source


def test_rizon_modules_stay_focused(repo_root: Path) -> None:
    """The split must not hide the monolith under a new filename."""
    rizon_root = repo_root / "devices" / "robots" / "rizon"
    implementation_files = sorted(
        path for path in rizon_root.glob("*.py") if path.name != "__init__.py"
    )

    assert {path.stem for path in implementation_files} == {
        "commands",
        "config",
        "connection",
        "gripper",
        "home",
        "planner",
        "robot",
        "safety",
    }
    assert {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in implementation_files
        if len(path.read_text(encoding="utf-8").splitlines()) >= 1000
    } == {}


def test_i2rt_public_surface_is_unchanged(repo_root: Path) -> None:
    """Protect I2RT lifecycle methods while neighboring robot code moves."""
    methods = _class_methods(
        repo_root / "devices" / "robots" / "i2rt.py",
        "I2RTRobot",
    )

    assert {
        "execute_control_command",
        "get_robot_state",
        "get_status",
        "start_server",
        "stop_server",
    } <= methods


def test_openpi_camera_public_surface_is_unchanged(repo_root: Path) -> None:
    """Protect the OpenPI camera adapter while camera internals move."""
    path = repo_root / "devices" / "cameras" / "openpi.py"
    methods = _class_methods(path, "WebCameraOpenPi")
    source = _source(path)

    assert {
        "_apply_rotation",
        "_capture_frame",
        "_initialize_camera",
        "get_device_info",
    } <= methods
    assert "class WebCameraOpenPi(WebCameraDevice):" in source
