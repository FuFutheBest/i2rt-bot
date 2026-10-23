"""AST contracts for device adapter decomposition.

Author: SuperInference
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

def _python_files(root: Path) -> Iterable[Path]:
    """Yield implementation modules in a split adapter package."""
    yield from sorted(path for path in root.rglob("*.py") if path.name != "__init__.py")


def _class_methods(path: Path, class_name: str) -> set[str]:
    """Return methods declared directly on a class without importing hardware code."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} not found in {path}")


def test_split_implementation_modules_stay_focused(repo_root: Path) -> None:
    """Rizon and Hik splits must not hide a renamed monolith."""
    roots = (
        repo_root / "devices" / "robots" / "rizon",
        repo_root / "devices" / "cameras" / "hik",
    )
    for root in roots:
        for path in _python_files(root):
            assert len(path.read_text(encoding="utf-8").splitlines()) < 1000, path


def test_split_device_modules_do_not_import_incubator(repo_root: Path) -> None:
    """Production adapter modules must remain independent of incubator code."""
    roots = (
        repo_root / "devices" / "robots" / "rizon",
        repo_root / "devices" / "cameras" / "hik",
    )
    violations: list[str] = []
    for root in roots:
        for path in _python_files(root):
            module = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == "incubator" or name.startswith("incubator.") for name in names):
                    violations.append(str(path.relative_to(repo_root)))
    assert violations == []


def test_rizon_i2rt_and_openpi_public_method_contracts(repo_root: Path) -> None:
    """Pin public control surfaces while allowing internal decomposition."""
    rizon_root = repo_root / "devices" / "robots" / "rizon"
    rizon_methods: set[str] = set()
    for path, class_name in (
        (rizon_root / "robot.py", "RizonRobot"),
        (rizon_root / "commands.py", "RizonCommandMixin"),
        (rizon_root / "connection.py", "RizonConnectionMixin"),
    ):
        rizon_methods.update(_class_methods(path, class_name))

    assert {
        "start_server",
        "stop_server",
        "get_status",
        "get_robot_state",
        "get_control_array",
        "execute_control_command",
        "switch_to",
    } <= rizon_methods
    assert {
        "start_server",
        "stop_server",
        "get_status",
        "get_robot_state",
        "execute_control_command",
    } <= _class_methods(repo_root / "devices" / "robots" / "i2rt.py", "I2RTRobot")
    assert {
        "_initialize_camera",
        "_capture_frame",
        "get_device_info",
    } <= _class_methods(repo_root / "devices" / "cameras" / "openpi.py", "WebCameraOpenPi")


def test_rizon_shm_reset_and_safety_contracts(repo_root: Path) -> None:
    """Pin critical SHM naming, reset ordering, and safety hooks."""
    rizon_root = repo_root / "devices" / "robots" / "rizon"
    robot_source = (rizon_root / "robot.py").read_text(encoding="utf-8")
    base_source = (repo_root / "devices" / "robots" / "base.py").read_text(
        encoding="utf-8"
    )
    home_source = (rizon_root / "home.py").read_text(encoding="utf-8")
    command_source = (rizon_root / "commands.py").read_text(encoding="utf-8")

    assert 'f"{self.device_name}_{self.device_class_id}_control"' in base_source
    assert "max_contact_wrench" in robot_source
    assert "self.safety_bounds.clip_position" in command_source
    assert home_source.count("self._clear_control_shm()") >= 2
    assert home_source.index("self._clear_control_shm()") < home_source.index(
        "self._notify_policy_runner_reset()"
    )


def test_force_gated_rizon_keeps_fail_closed_enable_contract(repo_root: Path) -> None:
    """The optional force-gated implementation must not use bool(string)."""
    source = (repo_root / "incubator" / "r1" / "r1_rizon.py").read_text(
        encoding="utf-8"
    )
    method_start = source.index("def _normalize_gripper_active_config")
    method_end = source.find("\n    def ", method_start + 1)
    method = source[method_start:] if method_end < 0 else source[method_start:method_end]
    assert "casefold() in {\"1\", \"true\", \"yes\", \"on\"}" in method
    assert "return [bool(item)" not in method
