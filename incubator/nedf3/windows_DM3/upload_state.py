#!/usr/bin/env python3
"""
Local upload.json helpers for DM3 Windows collection workflow.

Author: yichen Xie
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger_config import logger


UPLOAD_STATE_FILE = "upload.json"
VALID_UPLOAD_STATUSES = {"pending", "uploading", "cancelled", "success", "failed"}


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def resolve_upload_state_path(data_dir: Path) -> Path:
    """Return the upload.json path for a local data directory."""
    return data_dir / UPLOAD_STATE_FILE


def load_upload_state(data_dir: Path) -> Dict[str, Any]:
    """Load upload.json if present, otherwise return an empty state."""
    state_path = resolve_upload_state_path(data_dir)
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_upload_state(data_dir: Path, state: Dict[str, Any]) -> Path:
    """Write upload.json using UTF-8 pretty JSON."""
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = resolve_upload_state_path(data_dir)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info(f"Wrote DM3 upload state: {state_path}")
    return state_path


def update_upload_state(
    data_dir: Path,
    *,
    status: str,
    task_execution: Optional[Dict[str, Any]] = None,
    note: str = "",
    resource_type: str = "raw",
) -> Path:
    """Create or update upload.json for a local collected data directory."""
    if status not in VALID_UPLOAD_STATUSES:
        raise ValueError(f"Invalid upload status: {status}")

    state = load_upload_state(data_dir)
    state.update(
        {
            "status": status,
            "resource_type": resource_type,
            "updated_at": utc_now_iso(),
        }
    )
    if note:
        state["note"] = note
    if task_execution is not None:
        state["task_execution"] = _summarize_task_execution(task_execution)
    state.setdefault("created_at", utc_now_iso())
    return write_upload_state(data_dir, state)


def _summarize_task_execution(task_execution: Dict[str, Any]) -> Dict[str, Any]:
    """Keep upload.json compact while preserving the binding to a DM3 task."""
    task_info = task_execution.get("task")
    if not isinstance(task_info, dict):
        task_info = {}
    return {
        "id": task_execution.get("id"),
        "status": task_execution.get("status") or task_execution.get("execution_status"),
        "name": task_execution.get("name") or task_info.get("name"),
        "task_id": task_info.get("id") or task_execution.get("task_id"),
        "task_name": task_info.get("name"),
        "raw": task_execution,
    }

