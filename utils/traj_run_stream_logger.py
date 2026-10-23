#!/usr/bin/env python3
"""
Append-only JSONL sink for ``traj_run_record`` stream fragments (e.g. sparse waypoints).

Used by the robot device process to log policy commands with ``time.perf_counter()`` timestamps
for later merging into a full ``traj_run_record`` JSON.

Author: Zixi Ying
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from utils.logger_config import logger


class TrajRunJsonlSink:
    """
    Thread-safe append of one JSON object per line (JSONL).

    Each line is a self-contained object; downstream tooling merges lines into
    ``traj_run_record`` fragments (sparse plan, dense command trace, recorded samples, etc.).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False

    def append(self, record: Dict[str, Any]) -> None:
        """Serialize ``record`` as one JSON line and append to the file."""
        if self._closed:
            logger.warning("TrajRunJsonlSink.append called after close: {}", self._path)
            return
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def close(self) -> None:
        """Mark sink closed (file is opened per append; nothing to flush)."""
        self._closed = True


SparseTrajRunJsonlSink = TrajRunJsonlSink
