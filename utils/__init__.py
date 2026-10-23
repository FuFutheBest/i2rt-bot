"""Small, dependency-light utility package exports.

The logger is intentionally loaded lazily.  Importing a focused helper such as
``utils.inference_options`` must not create log files or configure process-wide
logging as a side effect (the Control Panel relies on that boundary).
"""

from __future__ import annotations

from typing import Any

__all__ = ["setup_logger", "logger"]


def __getattr__(name: str) -> Any:
    if name in {"setup_logger", "logger"}:
        from .logger_config import logger, setup_logger

        return logger if name == "logger" else setup_logger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
