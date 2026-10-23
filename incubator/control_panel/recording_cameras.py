"""Stable camera identities for NEDF inference recordings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from .recording_contract import (
    CAMERA_ROLES,
    NEDF3_CAMERA_IDS,
    NEDF3_CAMERA_ROLE_LABELS,
    NEDF3_DATA_FORMAT,
    camera_ids_for_format,
    normalize_data_format,
)

LOGICAL_CAMERA_IDS = NEDF3_CAMERA_IDS


def normalize_camera_role_devices(
    value: Optional[Mapping[str, Any]],
) -> dict[str, Optional[str]]:
    """Normalize role or ``*_camera`` keys and reject duplicate devices."""
    if value is None:
        source: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        source = value
    else:
        raise ValueError("camera roles must be a JSON object or null")
    result: dict[str, Optional[str]] = {}
    for role in CAMERA_ROLES:
        raw = source.get(role, source.get(f"{role}_camera"))
        if raw is None:
            result[role] = None
            continue
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{role} camera device must be a non-empty string or null")
        result[role] = raw.strip()

    assigned = [device for device in result.values() if device is not None]
    if len(set(assigned)) != len(assigned):
        raise ValueError("head, left, and right camera roles must use different devices")
    return result


def complete_camera_role_devices(
    value: Optional[Mapping[str, Any]],
    candidates: Iterable[Any],
) -> dict[str, Optional[str]]:
    """Fill legacy configs from their ordered policy camera inputs."""
    result = normalize_camera_role_devices(value)
    ordered: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        device = candidate.strip()
        if device not in ordered:
            ordered.append(device)

    used = {device for device in result.values() if device is not None}
    if used:
        return result

    remaining = [device for device in ordered if device not in used]
    if len(remaining) >= 3:
        result["head"] = remaining.pop(0)
    for role in ("left", "right"):
        if result[role] is None and remaining:
            result[role] = remaining.pop(0)
    return normalize_camera_role_devices(result)


def camera_id_overrides(
    role_devices: Optional[Mapping[str, Any]],
    *,
    data_format: Any = NEDF3_DATA_FORMAT,
) -> dict[str, str]:
    """Return the physical-device to stable NEDF camera-id mapping."""
    normalized = normalize_camera_role_devices(role_devices)
    camera_ids = camera_ids_for_format(data_format)
    return {
        device: camera_ids[role]
        for role, device in normalized.items()
        if device is not None
    }


def resolve_recorded_camera_ids(
    camera_info: Any,
    requested: Optional[Mapping[str, Any]] = None,
    *,
    data_format: Any = NEDF3_DATA_FORMAT,
) -> dict[str, Optional[str]]:
    """Resolve canonical ids first, then legacy physical-device ids."""
    info = camera_info if isinstance(camera_info, Mapping) else {}
    normalized_format = normalize_data_format(data_format)
    logical_camera_ids = camera_ids_for_format(normalized_format)
    keys = {
        str(camera_id).casefold(): str(camera_id)
        for camera_id in info
        if isinstance(camera_id, str) and camera_id
    }
    values: dict[str, list[str]] = {}
    for camera_id, source_device in info.items():
        if not isinstance(camera_id, str) or not isinstance(source_device, str):
            continue
        values.setdefault(source_device.casefold(), []).append(camera_id)

    expected = normalize_camera_role_devices(requested)
    label_roles = {
        label.casefold(): role for role, label in NEDF3_CAMERA_ROLE_LABELS.items()
    }
    ids_by_role: dict[str, list[str]] = {role: [] for role in CAMERA_ROLES}
    if normalized_format == NEDF3_DATA_FORMAT:
        for camera_id, label in info.items():
            if not isinstance(camera_id, str) or not isinstance(label, str):
                continue
            role = label_roles.get(label.strip().casefold())
            if role is not None:
                ids_by_role[role].append(camera_id)

    result: dict[str, Optional[str]] = {}
    for role in CAMERA_ROLES:
        labeled = ids_by_role[role]
        if len(labeled) == 1:
            result[role] = labeled[0]
            continue
        logical = keys.get(logical_camera_ids[role].casefold())
        if logical is not None:
            result[role] = logical
            continue
        device = expected[role]
        direct = keys.get(device.casefold()) if device is not None else None
        if direct is not None:
            result[role] = direct
            continue
        matches = values.get(device.casefold(), []) if device is not None else []
        result[role] = matches[0] if len(matches) == 1 else None
    return result


__all__ = [
    "CAMERA_ROLES",
    "LOGICAL_CAMERA_IDS",
    "camera_id_overrides",
    "complete_camera_role_devices",
    "normalize_camera_role_devices",
    "resolve_recorded_camera_ids",
]
