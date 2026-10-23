"""Focused checks for RoboTwin dependency preflight reporting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "scripts" / "robotwin" / "preflight.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "robotwin_preflight_test", PREFLIGHT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_check_reports_import_errors_and_sapien_remediation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    def fail_import(name: str) -> None:
        assert name == "sapien"
        raise ModuleNotFoundError("No module named 'pkg_resources'")

    monkeypatch.setattr(module.importlib, "import_module", fail_import)

    assert module._module("sapien") == {
        "kind": "python_module",
        "label": "sapien",
        "value": "sapien",
        "ok": False,
        "error": "ModuleNotFoundError: No module named 'pkg_resources'",
        "hint": (
            "Run: python -m pip install -r requirements/requirements-robotwin.txt"
        ),
    }
