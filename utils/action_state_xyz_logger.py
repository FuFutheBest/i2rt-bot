#!/usr/bin/env python3
"""Async JSONL logging for action/state xyz traces."""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from utils.logger_config import logger


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _make_run_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False).reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _xyz_payload(
    values: np.ndarray,
    start: int,
    width_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size < start + 3:
        return None
    xyz = flat[start : start + 3].astype(float, copy=False).tolist()
    if not np.all(np.isfinite(xyz)):
        return None
    payload: Dict[str, Any] = {"x": xyz[0], "y": xyz[1], "z": xyz[2], "xyz": xyz}
    if width_index is not None and flat.size > width_index:
        width = float(flat[width_index])
        if np.isfinite(width):
            payload["w"] = width
            payload["width"] = width
            payload["gripper_width"] = width
    return payload


def _action_arm_records(action: np.ndarray) -> Iterable[Dict[str, Any]]:
    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    if flat.size >= 16:
        left = _xyz_payload(flat, 0, 7)
        right = _xyz_payload(flat, 8, 15)
        if left is not None:
            yield {"arm": "left", **left}
        if right is not None:
            yield {"arm": "right", **right}
        return
    payload = _xyz_payload(flat, 0, 7 if flat.size >= 8 else None)
    if payload is not None:
        yield {"arm": "single", **payload}


def _state_arm_records(state: np.ndarray) -> Iterable[Dict[str, Any]]:
    flat = np.asarray(state, dtype=np.float64).reshape(-1)
    if flat.size >= 44:
        left = _xyz_payload(flat, 14, 21)
        right = _xyz_payload(flat, 36, 43)
        if left is not None:
            yield {"arm": "left", **left}
        if right is not None:
            yield {"arm": "right", **right}
        return
    if flat.size >= 22:
        payload = _xyz_payload(flat, 14, 21)
        if payload is not None:
            yield {"arm": "single", **payload}
        return
    payload = _xyz_payload(flat, 0, 7 if flat.size >= 8 else None)
    if payload is not None:
        yield {"arm": "single", **payload}


JOINT14_NAMES = (
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_gripper",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_gripper",
)


def _joint14_payload(values: np.ndarray) -> Optional[Dict[str, Any]]:
    """Return the explicit joint-space payload used by RoboTwin traces."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size != len(JOINT14_NAMES):
        return None
    if not np.all(np.isfinite(flat)):
        return None
    return {
        "vector_layout": "joint14",
        "joint_names": list(JOINT14_NAMES),
        "joint_positions": flat,
    }


class ActionStateXYZLogger:
    """Non-blocking action/state xyz logger.

    The robot control path only enqueues small dictionaries with ``put_nowait``.
    Disk I/O is performed by a daemon writer thread.
    """

    def __init__(
        self,
        source: str,
        *,
        log_dir: Optional[str] = None,
        enabled: Optional[bool] = None,
        queue_size: Optional[int] = None,
    ) -> None:
        self.source = source
        self.enabled = (
            _env_flag("SI_ACTION_STATE_XYZ_LOG_ENABLED", True)
            if enabled is None
            else bool(enabled)
        )
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(
            maxsize=int(queue_size or os.environ.get("SI_ACTION_STATE_XYZ_QUEUE_SIZE", "20000"))
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._drop_count = 0
        self._warned_drop = False
        self.path: Optional[Path] = None

        if not self.enabled:
            return

        explicit_file = os.environ.get("SI_ACTION_STATE_XYZ_LOG_FILE")
        if explicit_file:
            self.path = Path(explicit_file).expanduser()
        else:
            base_dir = Path(
                log_dir or os.environ.get("SI_ACTION_STATE_XYZ_LOG_DIR", "logs/action_state_xyz")
            ).expanduser()
            run_id = os.environ.get(
                "SI_ACTION_STATE_XYZ_RUN_ID",
                os.environ.get("SI_TIMESTAMP_RUN_ID", _make_run_id()),
            )
            safe_source = "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source
            )
            self.path = base_dir / run_id / f"{safe_source}_xyz.jsonl"

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._writer_loop,
                name=f"{self.source}-xyz-jsonl-writer",
                daemon=True,
            )
            self._thread.start()
            self.log_meta("logger_started", log_path=str(self.path))
            atexit.register(self.close)
            logger.info(f"Action/state xyz logging enabled: {self.path}")
        except Exception as exc:
            self.enabled = False
            logger.warning(f"Action/state xyz logging disabled: {exc}")

    @staticmethod
    def now_iso(event_ns: int) -> str:
        return datetime.fromtimestamp(event_ns / 1e9, tz=timezone.utc).isoformat()

    def _enqueue(self, record: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            self._queue.put_nowait(_json_safe(record))
        except queue.Full:
            self._drop_count += 1
            if not self._warned_drop or self._drop_count % 1000 == 0:
                self._warned_drop = True
                logger.warning(
                    "Action/state xyz log queue is full; dropped "
                    f"{self._drop_count} records so far"
                )

    def _base_record(self, event_type: str, timestamp_ns: Optional[int]) -> Dict[str, Any]:
        event_ns = int(timestamp_ns if timestamp_ns is not None else time.time_ns())
        event_time_utc = self.now_iso(event_ns)
        return {
            "event_type": event_type,
            "timestamp_ns": event_ns,
            "timestamp_time_utc": event_time_utc,
            # TimestampEventLogger-compatible aliases for cross-trace tooling.
            "event_ns": event_ns,
            "event_time_utc": event_time_utc,
            "source": self.source,
            "pid": os.getpid(),
        }

    def log_meta(self, event_type: str, **fields: Any) -> None:
        record = self._base_record(event_type, fields.pop("timestamp_ns", None))
        record.update(fields)
        self._enqueue(record)

    def log_action(
        self,
        *,
        device_name: str,
        action: np.ndarray,
        timestamp_ns: int,
        event_type: str = "action_xyz",
        vector_layout: Optional[str] = None,
        **fields: Any,
    ) -> None:
        flat = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector_layout == "joint14":
            payload = _joint14_payload(flat)
            if payload is None:
                return
            record = self._base_record(event_type, timestamp_ns)
            record.update(
                {
                    "device_name": device_name,
                    "vector_dim": int(flat.size),
                    "vector_kind": "action",
                    **payload,
                    **fields,
                }
            )
            self._enqueue(record)
            return
        for arm_record in _action_arm_records(flat):
            record = self._base_record(event_type, timestamp_ns)
            record.update(
                {
                    "device_name": device_name,
                    "vector_dim": int(flat.size),
                    "vector_kind": "action",
                    **arm_record,
                    **fields,
                }
            )
            self._enqueue(record)

    def log_state(
        self,
        *,
        device_name: str,
        state: np.ndarray,
        timestamp_ns: int,
        event_type: str = "state_xyz",
        vector_layout: Optional[str] = None,
        **fields: Any,
    ) -> None:
        flat = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector_layout == "joint14":
            payload = _joint14_payload(flat)
            if payload is None:
                return
            record = self._base_record(event_type, timestamp_ns)
            record.update(
                {
                    "device_name": device_name,
                    "vector_dim": int(flat.size),
                    "vector_kind": "state",
                    **payload,
                    **fields,
                }
            )
            self._enqueue(record)
            return
        for arm_record in _state_arm_records(flat):
            record = self._base_record(event_type, timestamp_ns)
            record.update(
                {
                    "device_name": device_name,
                    "vector_dim": int(flat.size),
                    "vector_kind": "state",
                    **arm_record,
                    **fields,
                }
            )
            self._enqueue(record)

    def _writer_loop(self) -> None:
        assert self.path is not None
        pending: List[Dict[str, Any]] = []
        last_flush = time.monotonic()

        def flush_pending() -> None:
            nonlocal last_flush
            if not pending:
                return
            try:
                with self.path.open("a", encoding="utf-8") as file:
                    for record in pending:
                        file.write(
                            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                pending.clear()
                last_flush = time.monotonic()
            except Exception as exc:
                logger.warning(f"Failed to write action/state xyz log: {exc}")
                pending.clear()

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                item = None

            if item is not None:
                pending.append(item)
                self._queue.task_done()

            now = time.monotonic()
            if pending and (len(pending) >= 128 or now - last_flush >= 0.5 or item is None):
                flush_pending()

            if self._stop_event.is_set() and self._queue.empty():
                break

        flush_pending()

    def close(self) -> None:
        if not self.enabled or self._thread is None:
            return
        self._stop_event.set()
        try:
            self._thread.join(timeout=2.0)
        except RuntimeError:
            pass
        self._thread = None
