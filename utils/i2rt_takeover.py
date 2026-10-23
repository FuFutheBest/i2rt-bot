"""Fresh file-based barriers for the I2RT policy-to-human takeover state.

The scalar I2RT control flag selects policy, alignment, or teleoperation, but
it cannot carry a request identity.  This module supplies the identity-bearing
request and follower-hold acknowledgement needed before a leader may move
toward the follower pose during the alignment phase.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


TAKEOVER_HOLD_REQUEST_SCHEMA = "i2rt/takeover_hold_request/v1"
TAKEOVER_HOLD_ACK_SCHEMA = "i2rt/takeover_hold_ack/v2"
_MAX_PAYLOAD_BYTES = 64 * 1024


class TakeoverHoldSyncError(ValueError):
    """A takeover hold request or acknowledgement is missing or unsafe."""


def new_takeover_hold_request_id() -> str:
    """Return a collision-resistant identity for one `.5` hold barrier."""
    return secrets.token_hex(16)


def publish_takeover_hold_request(
    path: str | Path,
    *,
    source_device_id: int,
    target_device_id: int,
    request_id: Optional[str] = None,
) -> str:
    """Publish a fresh leader request before it enters the hold phase."""
    source_id = _normalize_device_id(source_device_id, field="source_device_id")
    target_id = _normalize_device_id(target_device_id, field="target_device_id")
    if source_id == target_id:
        raise TakeoverHoldSyncError("takeover target_device_id must differ from source_device_id")
    normalized_request_id = _normalize_request_id(
        new_takeover_hold_request_id() if request_id is None else request_id
    )
    _atomic_write_json(
        Path(path),
        {
            "schema_version": TAKEOVER_HOLD_REQUEST_SCHEMA,
            "request_id": normalized_request_id,
            "source_device_id": source_id,
            "target_device_id": target_id,
            "issued_at_ns": int(time.time_ns()),
        },
    )
    return normalized_request_id


def read_takeover_hold_request(
    path: str | Path,
    *,
    expected_source_device_id: int,
    expected_target_device_id: int,
) -> Optional[str]:
    """Read a valid leader request addressed to one follower."""
    expected_source = _normalize_device_id(
        expected_source_device_id, field="expected_source_device_id"
    )
    expected_target = _normalize_device_id(
        expected_target_device_id, field="expected_target_device_id"
    )
    payload = _read_json_object(Path(path), label="takeover hold request")
    if payload is None:
        return None
    if payload.get("schema_version") != TAKEOVER_HOLD_REQUEST_SCHEMA:
        raise TakeoverHoldSyncError("takeover hold request schema is invalid")
    _require_matching_device_id(
        payload,
        field="source_device_id",
        expected=expected_source,
    )
    _require_matching_device_id(
        payload,
        field="target_device_id",
        expected=expected_target,
    )
    _require_positive_timestamp(payload, field="issued_at_ns")
    return _normalize_request_id(payload.get("request_id"))


def publish_takeover_hold_ack(
    path: str | Path,
    *,
    request_id: str,
    source_device_id: int,
    target_device_id: int,
    hold_joint_pos: Sequence[float],
    measured_at_ns: int,
) -> None:
    """Publish the follower's measured, physically stable pose for one request."""
    source_id = _normalize_device_id(source_device_id, field="source_device_id")
    target_id = _normalize_device_id(target_device_id, field="target_device_id")
    if source_id == target_id:
        raise TakeoverHoldSyncError("takeover target_device_id must differ from source_device_id")
    measurement_timestamp = _normalize_positive_timestamp(
        measured_at_ns,
        field="measured_at_ns",
    )
    _atomic_write_json(
        Path(path),
        {
            "schema_version": TAKEOVER_HOLD_ACK_SCHEMA,
            "request_id": _normalize_request_id(request_id),
            "source_device_id": source_id,
            "target_device_id": target_id,
            "held_at_ns": int(time.time_ns()),
            "measured_at_ns": measurement_timestamp,
            "hold_joint_pos": _normalize_hold_joint_pos(hold_joint_pos),
        },
    )


def read_takeover_hold_ack(
    path: str | Path,
    *,
    request_id: str,
    expected_source_device_id: int,
    expected_target_device_id: int,
) -> Optional[np.ndarray]:
    """Read a matching follower acknowledgement and its measured hold target."""
    expected_request_id = _normalize_request_id(request_id)
    expected_source = _normalize_device_id(
        expected_source_device_id, field="expected_source_device_id"
    )
    expected_target = _normalize_device_id(
        expected_target_device_id, field="expected_target_device_id"
    )
    payload = _read_json_object(Path(path), label="takeover hold acknowledgement")
    if payload is None:
        return None
    if payload.get("schema_version") != TAKEOVER_HOLD_ACK_SCHEMA:
        raise TakeoverHoldSyncError("takeover hold acknowledgement schema is invalid")
    if _normalize_request_id(payload.get("request_id")) != expected_request_id:
        return None
    _require_matching_device_id(
        payload,
        field="source_device_id",
        expected=expected_source,
    )
    _require_matching_device_id(
        payload,
        field="target_device_id",
        expected=expected_target,
    )
    _require_positive_timestamp(payload, field="held_at_ns")
    _require_positive_timestamp(payload, field="measured_at_ns")
    return np.asarray(
        _normalize_hold_joint_pos(payload.get("hold_joint_pos")), dtype=np.float64
    )


def _normalize_request_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 32:
        raise TakeoverHoldSyncError("takeover request_id must be a 32-character hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise TakeoverHoldSyncError(
            "takeover request_id must be a 32-character hex string"
        ) from exc
    return value.lower()


def _normalize_device_id(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TakeoverHoldSyncError(f"takeover {field} must be an integer")
    return value


def _require_matching_device_id(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: int,
) -> None:
    actual = payload.get(field)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
        raise TakeoverHoldSyncError(f"takeover {field} is invalid")


def _require_positive_timestamp(payload: Mapping[str, Any], *, field: str) -> None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TakeoverHoldSyncError(f"takeover {field} is invalid")


def _normalize_positive_timestamp(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TakeoverHoldSyncError(f"takeover {field} is invalid")
    return value


def _normalize_hold_joint_pos(value: Any) -> list[float]:
    if not isinstance(value, (Sequence, np.ndarray)) or isinstance(value, (str, bytes)):
        raise TakeoverHoldSyncError("takeover hold target must contain a 14D joint sequence")
    try:
        joints = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TakeoverHoldSyncError(
            "takeover hold target must contain numeric joints"
        ) from exc
    if joints.ndim != 1 or joints.size != 14 or not np.all(np.isfinite(joints)):
        raise TakeoverHoldSyncError(
            "takeover hold target must contain 14 finite joints"
        )
    return joints.tolist()


def _read_json_object(path: Path, *, label: str) -> Optional[dict[str, Any]]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_symlink() or not path.is_file() or not 0 < info.st_size <= _MAX_PAYLOAD_BYTES:
        raise TakeoverHoldSyncError(f"{label} is not a safe regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TakeoverHoldSyncError(f"{label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise TakeoverHoldSyncError(f"{label} must be a JSON object")
    return dict(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            stream.write(encoded + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - rollout deployments are Linux.
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "TAKEOVER_HOLD_ACK_SCHEMA",
    "TAKEOVER_HOLD_REQUEST_SCHEMA",
    "TakeoverHoldSyncError",
    "new_takeover_hold_request_id",
    "publish_takeover_hold_ack",
    "publish_takeover_hold_request",
    "read_takeover_hold_ack",
    "read_takeover_hold_request",
]
