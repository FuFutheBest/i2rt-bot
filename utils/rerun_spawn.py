"""Locate and start the viewer bundled with the Rerun Python package."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any


def resolve_rerun_executable_path() -> str | None:
    """Return an explicit Rerun executable path when one is available."""
    configured = os.environ.get("RERUN_CLI_PATH", "").strip()
    if configured and Path(configured).is_file():
        return configured

    try:
        import rerun_sdk.rerun_cli as rerun_cli
    except ImportError:
        return None

    executable = Path(rerun_cli.__file__).resolve().parent / (
        "rerun.exe" if sys.platform == "win32" else "rerun"
    )
    return str(executable) if executable.is_file() else None


def spawn_rerun_viewer(**kwargs: Any) -> None:
    """Spawn Rerun without requiring its executable to be on ``PATH``."""
    import rerun as rr

    executable = kwargs.pop("executable_path", None) or resolve_rerun_executable_path()
    if executable is not None and "executable_path" in inspect.signature(rr.spawn).parameters:
        kwargs["executable_path"] = executable
    rr.spawn(**kwargs)


__all__ = ["resolve_rerun_executable_path", "spawn_rerun_viewer"]
