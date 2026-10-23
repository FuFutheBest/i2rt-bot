"""Trusted runtime interpreter selection for host and packaged deployments."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from utils.runtime_profiles import RuntimeProfileError, select_runtime_python

DEVICE_PYTHON_ENV = "SUPERINFERENCE_DEVICE_PYTHON"
INFERENCE_PYTHON_ENV = "SUPERINFERENCE_INFERENCE_PYTHON"
RELEASE_ROOT_ENV = "SUPERINFERENCE_RELEASE_ROOT"
DEVICE_PROFILE_NAME_ENV = "SUPERINFERENCE_DEVICE_PROFILE_NAME"
INFERENCE_PROFILE_NAME_ENV = "SUPERINFERENCE_INFERENCE_PROFILE_NAME"

RELEASE_DEVICE_PYTHON = Path("runtime/python312/bin/python3.12")
_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,40}")
_PYTHON_VERSION_RE = re.compile(r"(?<!\d)3[._-]?(1\d)(?!\d)")


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value.strip()).expanduser()


def _validated_active_python(active_python: Path | None = None) -> Path | None:
    candidate = Path(active_python or sys.executable).expanduser()
    try:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    except OSError:
        pass
    return None


def resolve_device_python(
    explicit: Path | None = None,
    *,
    active_python: Path | None = None,
    config_name: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Return the interpreter approved for isolated device workers.

    Source deployments may safely use the interpreter already running the
    control panel. Packaged releases remain pinned to their bundled Python 3.12
    runtime even if the release environment is incomplete or overridden.
    """
    if explicit is not None:
        return Path(explicit)

    configured = _environment_path(DEVICE_PYTHON_ENV)
    release_root = _environment_path(RELEASE_ROOT_ENV)
    if release_root is not None:
        bundled = release_root / RELEASE_DEVICE_PYTHON
        if configured is not None:
            try:
                if configured.resolve(strict=False) == bundled.resolve(strict=False):
                    return configured
            except OSError:
                pass
        return bundled

    if configured is not None:
        return configured
    active = _validated_active_python(active_python)
    try:
        selection = select_runtime_python(
            "device",
            config_name=config_name,
            fallback=active,
            repo_root=repo_root,
            require_verified_profile=True,
        )
    except RuntimeProfileError as exc:
        raise RuntimeError(f"Device Python profile is invalid: {exc}") from exc
    if selection is None:
        raise RuntimeError(
            'Device Python is not configured and the active interpreter is '
            'not executable'
        )
    return selection.python


def resolve_inference_python(
    explicit: Path | None = None,
    *,
    active_python: Path | None = None,
    config_name: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Return the approved OpenPi inference interpreter or wrapper.

    Packaged deployments set ``SUPERINFERENCE_INFERENCE_PYTHON`` to their policy
    wrapper. Source deployments inherit the interpreter used to start the panel,
    which keeps direct inference and Control Panel on the same OpenPI environment.
    """
    if explicit is not None:
        return Path(explicit)
    configured = _environment_path(INFERENCE_PYTHON_ENV)
    if configured is not None:
        return configured
    active = _validated_active_python(active_python)
    try:
        selection = select_runtime_python(
            "policy",
            config_name=config_name,
            fallback=active,
            repo_root=repo_root,
            require_verified_profile=True,
        )
    except RuntimeProfileError as exc:
        raise RuntimeError(f"Inference Python profile is invalid: {exc}") from exc
    if selection is None:
        raise RuntimeError(
            'Inference Python is not configured and the active interpreter is '
            'not executable'
        )
    return selection.python


def resolve_python_profile_name(
    prefix: str,
    python_path: Path,
    *,
    environment_name: str,
) -> str:
    """Return a bounded display label without executing the interpreter."""
    configured = os.environ.get(environment_name, "").strip()
    if _PROFILE_NAME_RE.fullmatch(configured):
        return configured

    candidate = Path(python_path).expanduser()
    for part in reversed(candidate.parts):
        lowered = part.lower()
        if "python" not in lowered and "venv" not in lowered:
            continue
        match = _PYTHON_VERSION_RE.search(lowered)
        if match is not None:
            return f"{prefix}3{match.group(1)}"

    try:
        if candidate.resolve(strict=False) == Path(sys.executable).resolve(
            strict=False
        ):
            return f"{prefix}{sys.version_info.major}{sys.version_info.minor}"
    except OSError:
        pass
    return prefix


__all__ = [
    "DEVICE_PROFILE_NAME_ENV",
    "DEVICE_PYTHON_ENV",
    "INFERENCE_PROFILE_NAME_ENV",
    "INFERENCE_PYTHON_ENV",
    "RELEASE_DEVICE_PYTHON",
    "RELEASE_ROOT_ENV",
    "resolve_device_python",
    "resolve_inference_python",
    "resolve_python_profile_name",
]
