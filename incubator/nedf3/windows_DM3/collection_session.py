#!/usr/bin/env python3
"""
Shared DM3 collection session state for data collection and NEDF3 metadata.

Persists login user info and the selected DM3 task execution so
``data_collection_gui.py`` and ``async_data_saver_NEDF3.py`` can read them.

Author: Yichen Xie
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger_config import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_SESSION_FILE = PROJECT_ROOT / "dm3_collection_session.json"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_collection_session() -> Dict[str, Any]:
    """Load the shared DM3 collection session, or return an empty dict."""
    if not COLLECTION_SESSION_FILE.is_file():
        return {}
    try:
        with COLLECTION_SESSION_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to load DM3 collection session: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def write_collection_session(state: Dict[str, Any]) -> Path:
    """Persist the shared DM3 collection session."""
    COLLECTION_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with COLLECTION_SESSION_FILE.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    logger.info(f"Wrote DM3 collection session: {COLLECTION_SESSION_FILE}")
    return COLLECTION_SESSION_FILE


def clear_collection_session() -> None:
    """Remove the shared session file if present."""
    if COLLECTION_SESSION_FILE.is_file():
        COLLECTION_SESSION_FILE.unlink()
        logger.info(f"Cleared DM3 collection session: {COLLECTION_SESSION_FILE}")


def _find_user_payload(login_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Locate the user object inside a DM3 login response."""
    containers: List[Dict[str, Any]] = [login_response]
    nested = login_response.get("data")
    if isinstance(nested, dict):
        containers.append(nested)

    user_keys = ("user", "userInfo", "user_info", "profile", "account")
    for container in containers:
        for key in user_keys:
            value = container.get(key)
            if isinstance(value, dict):
                return value
    return None


def extract_user_from_login(login_response: Dict[str, Any]) -> Tuple[str, str]:
    """Extract ``(user_id, user_name)`` from a DM3 login response."""
    user_id = ""
    user_name = ""

    user_payload = _find_user_payload(login_response)
    if user_payload is not None:
        for key in ("user_id", "id", "uid"):
            value = user_payload.get(key)
            if value is not None and str(value).strip():
                user_id = str(value).strip()
                break
        for key in (
            "user_name",
            "name",
            "username",
            "nickname",
            "real_name",
            "display_name",
            "phone_number",
        ):
            value = user_payload.get(key)
            if value is not None and str(value).strip():
                user_name = str(value).strip()
                break

    containers: List[Dict[str, Any]] = [login_response]
    nested = login_response.get("data")
    if isinstance(nested, dict):
        containers.append(nested)

    if not user_id:
        for container in containers:
            for key in ("user_id", "uid"):
                value = container.get(key)
                if value is not None and str(value).strip():
                    user_id = str(value).strip()
                    break
            if user_id:
                break

    if not user_name:
        for container in containers:
            for key in ("user_name", "username", "nickname", "name"):
                value = container.get(key)
                if value is not None and str(value).strip():
                    user_name = str(value).strip()
                    break
            if user_name:
                break

    return user_id, user_name


def resolve_session_user_fields(session: Dict[str, Any]) -> Tuple[str, str]:
    """Resolve user fields from session, re-parsing login payload when needed."""
    user_id = str(session.get("user_id") or "").strip()
    user_name = str(session.get("user_name") or "").strip()
    if user_id and user_name:
        return user_id, user_name

    login_response = session.get("login_response")
    if isinstance(login_response, dict):
        extracted_id, extracted_name = extract_user_from_login(login_response)
        user_id = user_id or extracted_id
        user_name = user_name or extracted_name

    login_phone = str(session.get("login_phone") or "").strip()
    if not user_id:
        user_id = login_phone
    if not user_name:
        user_name = login_phone
    return user_id, user_name


def update_login_session(
    user_id: str,
    user_name: str,
    *,
    login_phone: str = "",
    login_response: Optional[Dict[str, Any]] = None,
) -> Path:
    """Store DM3 login user fields in the shared session."""
    phone = login_phone.strip()
    resolved_id, resolved_name = user_id.strip(), user_name.strip()
    if login_response is not None:
        extracted_id, extracted_name = extract_user_from_login(login_response)
        resolved_id = resolved_id or extracted_id
        resolved_name = resolved_name or extracted_name
    if not resolved_id and phone:
        resolved_id = phone
    if not resolved_name and phone:
        resolved_name = phone

    state = load_collection_session()
    state.update(
        {
            "user_id": resolved_id,
            "user_name": resolved_name,
            "login_phone": phone,
            "updated_at": utc_now_iso(),
        }
    )
    if login_response is not None:
        state["login_response"] = login_response
    return write_collection_session(state)


def update_selected_task_session(
    task_execution: Dict[str, Any],
    *,
    task_description: str,
) -> Path:
    """Store the selected DM3 task execution and description."""
    task_info = task_execution.get("task") if isinstance(task_execution.get("task"), dict) else {}
    state = load_collection_session()
    state.update(
        {
            "task_execution_id": task_execution.get("id"),
            "task_id": task_info.get("id") or task_execution.get("task_id"),
            "task_name": task_execution.get("name") or task_info.get("name") or "",
            "task_description": normalize_task_description(task_description),
            "task_execution": task_execution,
            "updated_at": utc_now_iso(),
        }
    )
    return write_collection_session(state)


def normalize_task_description(text: str) -> str:
    """Flatten task description to a single line without newline characters."""
    if not text:
        return ""
    normalized = (
        str(text)
        .replace("\\n", " ")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    return " ".join(normalized.split())


METADATA_CAMERA_COUNT = 3


def expand_triple_csv(value: str, *, field_name: str = "value", default: str = "") -> str:
    """Replicate one comma-separated value for camera_0..2, or validate three explicit values."""
    raw = (value or default).strip()
    if not raw:
        raise ValueError(f"请填写 {field_name}。")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) == 1:
        return ",".join([parts[0]] * METADATA_CAMERA_COUNT)
    if len(parts) != METADATA_CAMERA_COUNT:
        raise ValueError(
            f"{field_name} 需要 1 个或 {METADATA_CAMERA_COUNT} 个值，当前为 {len(parts)} 个。"
        )
    return ",".join(parts)


def build_extra_device_entries(identifier: str) -> List[Dict[str, str]]:
    """Build metadata.extra device list from a GUI-entered identifier.

    One value is replicated for camera_0..2. Comma-separated values may supply
    one or three explicit identifiers.
    """
    raw = identifier.strip()
    if not raw:
        return []
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return []
    if len(parts) == 1:
        parts = parts * 3
    elif len(parts) != 3:
        raise ValueError(
            f"device identifier 需要 1 个或 3 个值，当前为 {len(parts)} 个。"
        )
    return [{"identifier": part} for part in parts]


def parse_extra_device_identifier(device_field: Any) -> str:
    """Parse device identifier from metadata.extra device field."""
    if isinstance(device_field, list):
        for item in device_field:
            if not isinstance(item, dict):
                continue
            for key in ("identifier", "indentifier"):
                value = item.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""
    if isinstance(device_field, dict):
        legacy_value = device_field.get("0")
        if legacy_value is not None and str(legacy_value).strip():
            return str(legacy_value).strip()
        for key in ("identifier", "indentifier"):
            value = device_field.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def extract_task_description(task_execution: Dict[str, Any]) -> str:
    """Resolve a human-readable task description from a DM3 task execution."""
    task_info = task_execution.get("task") if isinstance(task_execution.get("task"), dict) else {}
    for container in (task_info, task_execution):
        for key in ("description", "task_description", "desc", "summary"):
            value = container.get(key)
            if value is not None and str(value).strip():
                return normalize_task_description(str(value))
    for container in (task_info, task_execution):
        for key in ("name", "task_name"):
            value = container.get(key)
            if value is not None and str(value).strip():
                return normalize_task_description(str(value))
    return ""


def get_metadata_fields_from_session(
    *,
    extra_device_zero: Optional[str] = None,
) -> Dict[str, Any]:
    """Build NEDF3 metadata fields from the shared session and GUI overrides."""
    session = load_collection_session()
    user_id, user_name = resolve_session_user_fields(session)
    device_zero = (extra_device_zero or "").strip()
    return {
        "codebase_version": "",
        "device": build_extra_device_entries(device_zero),
        "user_id": user_id,
        "user_name": user_name,
        "task_description": normalize_task_description(
            str(session.get("task_description") or "")
        ),
    }

