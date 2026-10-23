#!/usr/bin/env python3
"""Compatibility entrypoint for the optional NEDF3 rollout recorder."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from typing import Any

_TARGET_MODULE = "incubator.r1.async_infer_data_saver_NEDF3"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
__all__ = ["InferenceDataSaverConsumer", "main"]  # noqa: F822 - resolved by __getattr__


def _implementation() -> Any:
    return importlib.import_module(_TARGET_MODULE)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_implementation(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))


if __name__ == "__main__":
    root = str(_REPOSITORY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    runpy.run_module(_TARGET_MODULE, run_name="__main__")
