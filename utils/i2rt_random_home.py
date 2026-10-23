"""Fresh, local-file coordination for I2RT randomized home targets."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


RANDOM_HOME_REQUEST_SCHEMA = "i2rt/random_home_request/v1"
RANDOM_HOME_TARGET_SCHEMA = "i2rt/random_home_target/v1"
RANDOM_HOME_RESULT_SCHEMA = "i2rt/random_home_result/v1"
_MAX_PAYLOAD_BYTES = 64 * 1024
_VALID_REASONS = {"startup", "episode_reset"}


class RandomHomeSyncError(ValueError):
    """A randomized-home coordination payload is missing or unsafe."""


def new_random_home_request_id() -> str:
    """Return a collision-resistant request ID shared by the leader and follower."""
    return secrets.token_hex(16)


def publish_random_home_request(
    path: str | Path,
    *,
    reason: str,
    request_id: Optional[str] = None,
) -> str:
    """Atomically publish a fresh request before both I2RT devices home."""
    if reason not in _VALID_REASONS:
        raise RandomHomeSyncError(f"unsupported random-home reason: {reason}")
    normalized_request_id = _normalize_request_id(
        new_random_home_request_id() if request_id is None else request_id
    )
    _atomic_write_json(
        Path(path),
        {
            "schema_version": RANDOM_HOME_REQUEST_SCHEMA,
            "request_id": normalized_request_id,
            "reason": reason,
            "issued_at_ns": int(time.time_ns()),
        },
    )
    return normalized_request_id


def read_random_home_request(path: str | Path, *, reason: str) -> Optional[str]:
    """Return the current request ID only when it is for the expected lifecycle."""
    if reason not in _VALID_REASONS:
        raise RandomHomeSyncError(f"unsupported random-home reason: {reason}")
    payload = _read_json_object(Path(path), label="random-home request")
    if payload is None:
        return None
    if payload.get("schema_version") != RANDOM_HOME_REQUEST_SCHEMA:
        raise RandomHomeSyncError("random-home request schema is invalid")
    if payload.get("reason") != reason:
        return None
    issued_at_ns = payload.get("issued_at_ns")
    if isinstance(issued_at_ns, bool) or not isinstance(issued_at_ns, int) or issued_at_ns <= 0:
        raise RandomHomeSyncError("random-home request issued_at_ns is invalid")
    return _normalize_request_id(payload.get("request_id"))


def publish_random_home_target(
    path: str | Path,
    *,
    request_id: str,
    home_joint_pos: Sequence[float],
    source_device_id: int,
) -> None:
    """Atomically publish the follower's applied 14D target for its leader."""
    normalized_request_id = _normalize_request_id(request_id)
    joints = _normalize_home_joint_pos(home_joint_pos)
    if isinstance(source_device_id, bool) or not isinstance(source_device_id, int):
        raise RandomHomeSyncError("source_device_id must be an integer")
    _atomic_write_json(
        Path(path),
        {
            "schema_version": RANDOM_HOME_TARGET_SCHEMA,
            "request_id": normalized_request_id,
            "source_device_id": source_device_id,
            "written_at_ns": int(time.time_ns()),
            "home_joint_pos": joints,
        },
    )


def read_random_home_target(
    path: str | Path,
    *,
    request_id: str,
    expected_source_device_id: int,
) -> Optional[np.ndarray]:
    """Return the follower target only when it belongs to one source/request.

    A target for another request is normally a stale file. It is ignored rather
    than treated as a valid fallback, so the leader waits for its matching
    follower move instead of reusing the previous episode's pose.
    """
    expected_request_id = _normalize_request_id(request_id)
    expected_source_id = _normalize_source_device_id(expected_source_device_id)
    payload = _read_json_object(Path(path), label="random-home target")
    if payload is None:
        return None
    if payload.get("schema_version") != RANDOM_HOME_TARGET_SCHEMA:
        raise RandomHomeSyncError("random-home target schema is invalid")
    written_at_ns = payload.get("written_at_ns")
    if isinstance(written_at_ns, bool) or not isinstance(written_at_ns, int) or written_at_ns <= 0:
        raise RandomHomeSyncError("random-home target written_at_ns is invalid")
    if _normalize_request_id(payload.get("request_id")) != expected_request_id:
        return None
    source_device_id = payload.get("source_device_id")
    if (
        isinstance(source_device_id, bool)
        or not isinstance(source_device_id, int)
        or source_device_id != expected_source_id
    ):
        raise RandomHomeSyncError("random-home target source_device_id is invalid")
    return np.asarray(_normalize_home_joint_pos(payload.get("home_joint_pos")), dtype=np.float64)


def random_home_result_path(
    pattern: str | Path,
    *,
    source_device_id: int,
) -> Path:
    """Resolve one device-specific result path from the runtime pattern."""
    if isinstance(source_device_id, bool) or not isinstance(source_device_id, int):
        raise RandomHomeSyncError("source_device_id must be an integer")
    raw_pattern = str(pattern)
    if not raw_pattern or "{device_id}" not in raw_pattern:
        raise RandomHomeSyncError(
            "random-home result pattern must contain {device_id}"
        )
    try:
        rendered = raw_pattern.format(device_id=source_device_id)
    except (KeyError, ValueError) as exc:
        raise RandomHomeSyncError("random-home result pattern is invalid") from exc
    if not rendered or "{" in rendered or "}" in rendered:
        raise RandomHomeSyncError("random-home result pattern is invalid")
    return Path(rendered)


def publish_random_home_result(
    pattern: str | Path,
    *,
    request_id: str,
    source_device_id: int,
    success: bool,
) -> Path:
    """Atomically publish one device's matching randomized-home result."""
    if not isinstance(success, bool):
        raise RandomHomeSyncError("random-home result success must be a boolean")
    normalized_request_id = _normalize_request_id(request_id)
    result_path = random_home_result_path(
        pattern,
        source_device_id=source_device_id,
    )
    _atomic_write_json(
        result_path,
        {
            "schema_version": RANDOM_HOME_RESULT_SCHEMA,
            "request_id": normalized_request_id,
            "source_device_id": source_device_id,
            "success": success,
            "completed_at_ns": int(time.time_ns()),
        },
    )
    return result_path


def read_random_home_result(
    pattern: str | Path,
    *,
    request_id: str,
    source_device_id: int,
) -> Optional[bool]:
    """Read one matching device result, ignoring a stale request result."""
    expected_request_id = _normalize_request_id(request_id)
    result_path = random_home_result_path(
        pattern,
        source_device_id=source_device_id,
    )
    payload = _read_json_object(result_path, label="random-home result")
    if payload is None:
        return None
    if payload.get("schema_version") != RANDOM_HOME_RESULT_SCHEMA:
        raise RandomHomeSyncError("random-home result schema is invalid")
    if _normalize_request_id(payload.get("request_id")) != expected_request_id:
        return None
    result_device_id = payload.get("source_device_id")
    if (
        isinstance(result_device_id, bool)
        or not isinstance(result_device_id, int)
        or result_device_id != source_device_id
    ):
        raise RandomHomeSyncError("random-home result source_device_id is invalid")
    completed_at_ns = payload.get("completed_at_ns")
    if (
        isinstance(completed_at_ns, bool)
        or not isinstance(completed_at_ns, int)
        or completed_at_ns <= 0
    ):
        raise RandomHomeSyncError("random-home result completed_at_ns is invalid")
    success = payload.get("success")
    if not isinstance(success, bool):
        raise RandomHomeSyncError("random-home result success must be a boolean")
    return success


def _normalize_request_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 32:
        raise RandomHomeSyncError("random-home request_id must be a 32-character hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RandomHomeSyncError(
            "random-home request_id must be a 32-character hex string"
        ) from exc
    return value.lower()


def _normalize_source_device_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RandomHomeSyncError("source_device_id must be an integer")
    return value


def _normalize_home_joint_pos(value: Any) -> list[float]:
    if (
        not isinstance(value, (Sequence, np.ndarray))
        or isinstance(value, (str, bytes))
    ):
        raise RandomHomeSyncError("random-home target must contain a 14D joint sequence")
    try:
        joints_array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RandomHomeSyncError(
            "random-home target must contain numeric joints"
        ) from exc
    if joints_array.ndim != 1 or joints_array.size != 14 or not np.all(
        np.isfinite(joints_array)
    ):
        raise RandomHomeSyncError("random-home target must contain 14 finite joints")
    return joints_array.tolist()


def _read_json_object(path: Path, *, label: str) -> Optional[dict[str, Any]]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_symlink() or not path.is_file() or not 0 < info.st_size <= _MAX_PAYLOAD_BYTES:
        raise RandomHomeSyncError(f"{label} is not a safe regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RandomHomeSyncError(f"{label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise RandomHomeSyncError(f"{label} must be a JSON object")
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
    if os.name == "nt":  # pragma: no cover - production rollout hosts are Linux.
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "RANDOM_HOME_REQUEST_SCHEMA",
    "RANDOM_HOME_RESULT_SCHEMA",
    "RANDOM_HOME_TARGET_SCHEMA",
    "RandomHomeSyncError",
    "new_random_home_request_id",
    "publish_random_home_request",
    "publish_random_home_result",
    "publish_random_home_target",
    "random_home_result_path",
    "read_random_home_request",
    "read_random_home_result",
    "read_random_home_target",
]
