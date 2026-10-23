"""Architecture tests for incubator isolation.

Author: SuperInference Contributors
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CORE_DIRS = ("consumer", "devices", "manager", "policy", "utils", "visualizers")
pytestmark = pytest.mark.architecture


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_core_does_not_import_incubator() -> None:
    violations: list[str] = []
    for directory in CORE_DIRS:
        for path in (ROOT / directory).rglob("*.py"):
            for module in _imported_modules(path):
                if module == "incubator" or module.startswith("incubator."):
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")
    assert violations == []


def test_incubator_control_panel_keeps_fastwam_in_one_adapter() -> None:
    """FastWAM integration belongs to one lifecycle adapter, not a mirror tree."""
    panel_root = ROOT / "incubator" / "control_panel"
    allowed_source = "incubator/control_panel/fastwam_workflow.py"
    duplicate_sources = [
        str(path.relative_to(ROOT))
        for path in panel_root.rglob("*")
        if path.is_file()
        and "tests" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix not in {".md", ".pyc"}
        and "fastwam" in path.name.casefold()
        and str(path.relative_to(ROOT)) != allowed_source
    ]
    assert duplicate_sources == []
    assert (ROOT / allowed_source).is_file()


def test_incubator_has_no_runtime_config_mirror() -> None:
    panel_configs = ROOT / "incubator" / "control_panel" / "configs"
    mirrored_files = (
        sorted(path.relative_to(ROOT) for path in panel_configs.rglob("*") if path.is_file())
        if panel_configs.exists()
        else []
    )
    assert mirrored_files == []
