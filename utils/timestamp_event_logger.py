#!/usr/bin/env python3
"""Lightweight JSONL timestamp event logging for runtime timing analysis."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.logger_config import logger


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _make_run_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_pid{os.getpid()}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        preview_count = min(8, flat.size)
        if np.issubdtype(value.dtype, np.number):
            preview = flat[:preview_count].astype(float, copy=False).tolist()
        else:
            preview = flat[:preview_count].tolist()
        summary: Dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": preview,
        }
        if flat.size > 0 and np.issubdtype(value.dtype, np.number):
            finite = flat[np.isfinite(flat)]
            if finite.size > 0:
                summary["min"] = float(np.min(finite))
                summary["max"] = float(np.max(finite))
        return summary
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class TimestampEventLogger:
    """Append-only JSONL event logger shared by policy/runner processes."""

    def __init__(
        self,
        source: str,
        *,
        log_dir: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.source = source
        self.enabled = _env_flag("SI_TIMESTAMP_LOG_ENABLED", True) if enabled is None else enabled
        self._lock = threading.Lock()
        self._warned = False
        self.path: Optional[Path] = None

        if not self.enabled:
            return

        explicit_file = os.environ.get("SI_TIMESTAMP_LOG_FILE")
        if explicit_file:
            self.path = Path(explicit_file).expanduser()
        else:
            base_dir = Path(
                log_dir or os.environ.get("SI_TIMESTAMP_LOG_DIR", "logs/timestamp_events")
            ).expanduser()
            run_id = os.environ.get("SI_TIMESTAMP_RUN_ID", _make_run_id())
            safe_source = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source)
            self.path = base_dir / f"{safe_source}_{run_id}.jsonl"

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.log_event("logger_started", log_path=str(self.path))
            logger.info(f"Timestamp event logging enabled: {self.path}")
        except Exception as exc:
            self.enabled = False
            logger.warning(f"Timestamp event logging disabled: {exc}")

    @staticmethod
    def now_iso(event_ns: int) -> str:
        return datetime.fromtimestamp(event_ns / 1e9, tz=timezone.utc).isoformat()

    def log_event(self, event_type: str, **fields: Any) -> None:
        if not self.enabled or self.path is None:
            return

        event_ns = time.time_ns()
        event_time_utc = self.now_iso(event_ns)
        record = {
            "event_type": event_type,
            "event_ns": event_ns,
            "event_time_utc": event_time_utc,
            # ActionStateXYZLogger-compatible aliases for cross-trace tooling.
            "timestamp_ns": event_ns,
            "timestamp_time_utc": event_time_utc,
            "source": self.source,
            "pid": os.getpid(),
        }
        record.update(_json_safe(fields))

        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning(f"Failed to write timestamp event log: {exc}")
