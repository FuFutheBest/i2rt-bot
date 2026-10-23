#!/usr/bin/env python3
"""Private recorder command/status protocol and dense segment journal.

This module deliberately uses only the Python standard library so the web
control process and the NEDF recorder can share the protocol without sharing a
Python environment.  The recorder is the sole writer of acknowledgements,
status, command journal entries, and per-episode dense labels.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import stat
import subprocess
import tempfile
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from utils.runtime_profiles import select_runtime_python

from .home_transition import HomeTransitionError, normalize_home_target

try:  # Linux production path.  Tests on other platforms use the process lock.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore


SCHEMA_VERSION = 1
MAX_COMMAND_BYTES = 64 * 1024
VALID_ACTIONS = frozenset(
    {
        "episode_accept",
        "episode_discard",
        "segment_accept",
        "segment_reject",
        "shutdown",
    }
)
VALID_STATES = frozenset(
    {
        "starting",
        "armed",
        "recording",
        "finalizing",
        "transitioning",
        "incomplete",
        "error",
        "stopped",
    }
)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
RECORDER_PREFLIGHT_MODULES = (
    "dm3_core.nedf",
    "nmx_msg",
    "cv2",
    "yaml",
    "numpy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_recorder_python(configured_path: Optional[str], fallback_path: str) -> str:
    """Resolve the recorder interpreter through the shared runtime registry."""
    selection = select_runtime_python(
        "recording",
        environment_names=(
            "SUPERINFERENCE_RECORDER_PYTHON",
            "SUPERINFERENCE_RECORDING_PYTHON",
        ),
        config_name=os.environ.get("SUPERINFERENCE_CONFIG_NAME"),
        fallback=configured_path or fallback_path,
        require_verified_profile=True,
    )
    if selection is None:
        return str(Path(fallback_path).expanduser())
    return str(selection.python)


def preflight_recorder_python(
    python_path: str,
    *,
    timeout_s: float = 10.0,
) -> Tuple[bool, Optional[str]]:
    """Verify the selected interpreter and NEDF imports without starting hardware."""
    path = Path(python_path).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        return False, f"Recorder Python is missing or not executable: {path}"
    import_code = ";".join(f"import {module}" for module in RECORDER_PREFLIGHT_MODULES)
    try:
        result = subprocess.run(
            [str(path), "-c", import_code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(0.1, float(timeout_s)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Recorder Python preflight failed: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown import failure").strip()
        return False, f"Recorder Python cannot import NEDF dependencies: {detail}"
    return True, None


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - production is Linux
        return
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Recorder control path is not a real directory: {path}")
    os.chmod(path, 0o700)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON by fsyncing a same-directory temp file then replacing."""
    ensure_private_directory(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        data = _canonical_json(payload) + b"\n"
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


_append_lock = threading.Lock()


def append_jsonl_durable(path: Path, payload: Mapping[str, Any], *, crc32: bool = False) -> Dict[str, Any]:
    """Append exactly one fsynced JSON line under an advisory file lock."""
    ensure_private_directory(path.parent)
    record = dict(payload)
    if crc32:
        record["crc32"] = f"{zlib.crc32(_canonical_json(record)) & 0xFFFFFFFF:08x}"
    encoded = _canonical_json(record) + b"\n"

    with _append_lock:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            written = os.write(fd, encoded)
            if written != len(encoded):
                raise OSError(f"Short append to {path}: {written}/{len(encoded)}")
            os.fsync(fd)
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)
    return record


@dataclass
class CommandProblem(Exception):
    code: str
    message: str
    failed: bool = False

    def __str__(self) -> str:
        return self.message


class RecordingControlPlane:
    """Recorder-owned command spool, status document, and dense label state."""

    def __init__(
        self,
        control_dir: Path,
        run_id: str,
        *,
        owner_pid: Optional[int] = None,
        reference_kind: str = "control_shm",
        reference_name: str = "RizonRobot_2_control",
        status_flush_interval_s: float = 0.1,
    ) -> None:
        if not _SAFE_ID_RE.fullmatch(run_id):
            raise ValueError(f"Unsafe SUPERINFERENCE_RUN_ID: {run_id!r}")
        self.control_dir = Path(control_dir).expanduser().resolve()
        self.run_id = run_id
        self.owner_pid = int(owner_pid or os.getpid())
        self.reference_kind = str(reference_kind)
        self.reference_name = str(reference_name)
        self.status_flush_interval_s = max(0.0, float(status_flush_interval_s))

        self.commands_dir = self.control_dir / "commands"
        self.acks_dir = self.control_dir / "acks"
        self.status_dir = self.control_dir / "status"
        self.status_path = self.status_dir / "recorder.json"
        self.journal_path = self.control_dir / "command_journal.jsonl"
        for directory in (self.control_dir, self.commands_dir, self.acks_dir, self.status_dir):
            ensure_private_directory(directory)

        self._lock = threading.RLock()
        self._last_status_flush_monotonic = 0.0
        self._episode_dir: Optional[Path] = None
        self._reference_timestamps: List[int] = []
        self._journal: Dict[str, Dict[str, Any]] = self._load_journal()
        self._status: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "owner_pid": self.owner_pid,
            "status": "starting",
            "ready": False,
            "active": False,
            "error": None,
            "revision": 0,
            "updated_at": utc_now(),
            "episode": None,
            "reference": {"kind": self.reference_kind, "name": self.reference_name},
            "capabilities": {
                "actions": sorted(VALID_ACTIONS),
                "dense_segments": True,
                "legacy_signal_files": True,
                "interval": "[start_frame,end_frame)",
            },
            "pending": [],
            "segments": [],
            "checkpoints": [],
        }
        self._flush_status(force=True)

    @classmethod
    def from_environment(
        cls,
        *,
        reference_kind: str = "control_shm",
        reference_name: Optional[str] = None,
    ) -> Optional["RecordingControlPlane"]:
        control_dir = os.environ.get("SUPERINFERENCE_RECORDING_CONTROL_DIR")
        run_id = os.environ.get("SUPERINFERENCE_RUN_ID")
        if not control_dir and not run_id:
            return None
        if not control_dir or not run_id:
            raise ValueError(
                "SUPERINFERENCE_RECORDING_CONTROL_DIR and SUPERINFERENCE_RUN_ID must be set together"
            )
        return cls(
            Path(control_dir),
            run_id,
            reference_kind=reference_kind,
            reference_name=(
                reference_name
                or os.environ.get("SUPERINFERENCE_RECORDING_REFERENCE_SHM")
                or "RizonRobot_2_control"
            ),
        )

    def _load_journal(self) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        if not self.journal_path.exists():
            return entries
        try:
            if self.journal_path.is_symlink() or not self.journal_path.is_file():
                raise ValueError("Command journal is not a regular file")
            with self.journal_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = item.get("idempotency_key")
                    if isinstance(key, str) and key:
                        entries[key] = item
        except FileNotFoundError:
            pass
        return entries

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._status.get("active"))

    @property
    def episode_id(self) -> Optional[str]:
        with self._lock:
            episode = self._status.get("episode")
            return episode.get("id") if isinstance(episode, dict) else None

    def _flush_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_flush_monotonic < self.status_flush_interval_s:
            return
        self._status["updated_at"] = utc_now()
        atomic_write_json(self.status_path, self._status)
        self._last_status_flush_monotonic = now

    def _bump_revision(self) -> int:
        self._status["revision"] = int(self._status.get("revision", 0)) + 1
        return int(self._status["revision"])

    def set_state(
        self,
        state: str,
        *,
        ready: Optional[bool] = None,
        active: Optional[bool] = None,
        error: Any = None,
        bump_revision: bool = True,
    ) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"Invalid recorder state: {state}")
        with self._lock:
            self._status["status"] = state
            if ready is not None:
                self._status["ready"] = bool(ready)
            if active is not None:
                self._status["active"] = bool(active)
            self._status["error"] = error
            if bump_revision:
                self._bump_revision()
            self._flush_status(force=True)

    def arm(self) -> None:
        self.set_state("armed", ready=True, active=False)

    def begin_episode(
        self,
        *,
        index: int,
        episode_dir: Path,
        episode_id: Optional[str] = None,
    ) -> str:
        with self._lock:
            identifier = episode_id or str(uuid.uuid4())
            self._episode_dir = Path(episode_dir)
            self._reference_timestamps = []
            self._status["episode"] = {
                "id": identifier,
                "index": int(index),
                "frame_count": 0,
                "elapsed_ms": 0.0,
                "first_timestamp_ns": None,
                "latest_timestamp_ns": None,
                "checkpoint_revision": 0,
                "untagged_start_frame": 0,
            }
            self._status["segments"] = []
            self._status["checkpoints"] = [
                {
                    "revision": 0,
                    "frame": 0,
                    "timestamp_ns": None,
                    "created_at": utc_now(),
                }
            ]
            self._status["status"] = "recording"
            self._status["ready"] = True
            self._status["active"] = True
            self._status["error"] = None
            self._bump_revision()
            self._flush_status(force=True)
            return identifier

    def note_reference_commit(self, timestamp_ns: int) -> None:
        with self._lock:
            episode = self._status.get("episode")
            if not self._status.get("active") or not isinstance(episode, dict):
                return
            timestamp = int(timestamp_ns)
            self._reference_timestamps.append(timestamp)
            first = self._reference_timestamps[0]
            episode["frame_count"] = len(self._reference_timestamps)
            episode["first_timestamp_ns"] = first
            episode["latest_timestamp_ns"] = timestamp
            episode["elapsed_ms"] = max(0.0, (timestamp - first) / 1_000_000.0)
            self._flush_status(force=False)

    def force_progress_flush(self) -> None:
        with self._lock:
            self._flush_status(force=True)

    def _require_active_episode(self, command: Mapping[str, Any]) -> Tuple[Dict[str, Any], Path]:
        episode = self._status.get("episode")
        if not self._status.get("active") or not isinstance(episode, dict) or self._episode_dir is None:
            raise CommandProblem("not_recording", "No recording episode is active")
        requested_episode = command.get("episode_id")
        if requested_episode != episode.get("id"):
            raise CommandProblem(
                "episode_conflict",
                f"Command episode {requested_episode!r} does not match active episode {episode.get('id')!r}",
            )
        return episode, self._episode_dir

    def append_segment(
        self,
        label: str,
        command: Mapping[str, Any],
        *,
        allow_empty: bool = False,
        source_action: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if label not in {"accept", "reject"}:
            raise ValueError(f"Invalid dense label: {label}")
        with self._lock:
            episode, episode_dir = self._require_active_episode(command)
            start = int(episode.get("untagged_start_frame", 0))
            end = len(self._reference_timestamps)
            if end <= start:
                if allow_empty:
                    return None
                raise CommandProblem(
                    "no_new_reference_frame",
                    "No newly committed reference frame exists after the previous checkpoint",
                )

            checkpoint_revision = int(episode.get("checkpoint_revision", 0)) + 1
            segment_index = len(self._status["segments"])
            start_timestamp = self._reference_timestamps[start]
            end_timestamp = self._reference_timestamps[end - 1]
            segment = {
                "schema_version": SCHEMA_VERSION,
                "episode_id": episode["id"],
                "segment_index": segment_index,
                "checkpoint_revision": checkpoint_revision,
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_timestamp_ns": start_timestamp,
                "end_timestamp_ns": end_timestamp,
                "interval": "[start_frame,end_frame)",
                "command_id": command.get("command_id"),
                "idempotency_key": command.get("idempotency_key"),
                "source_action": source_action or command.get("action"),
                "created_at": utc_now(),
            }
            durable_segment = append_jsonl_durable(
                episode_dir / "dense_labels.jsonl", segment, crc32=True
            )

            episode["checkpoint_revision"] = checkpoint_revision
            episode["untagged_start_frame"] = end
            self._status["segments"].append(durable_segment)
            self._status["checkpoints"].append(
                {
                    "revision": checkpoint_revision,
                    "frame": end,
                    "timestamp_ns": end_timestamp,
                    "label": label,
                    "created_at": durable_segment["created_at"],
                }
            )
            self._bump_revision()
            self._flush_status(force=True)
            return dict(durable_segment)

    def accept_tail(self, command: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        return self.append_segment(
            "accept", command, allow_empty=True, source_action="episode_accept"
        )

    def reject_tail(self, command: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        return self.append_segment(
            "reject", command, allow_empty=True, source_action="episode_discard"
        )

    def mark_incomplete(self, error: Any = None) -> None:
        self.set_state("incomplete", ready=False, active=False, error=error)

    def mark_error(self, error: Any) -> None:
        self.set_state("error", ready=False, active=False, error=error)

    def _validate_command_file(self, path: Path) -> Dict[str, Any]:
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            raise CommandProblem("command_missing", "Command disappeared before it could be read")
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise CommandProblem("unsafe_command_file", "Command is not a regular non-symlink file", True)
        if file_stat.st_size <= 0 or file_stat.st_size > MAX_COMMAND_BYTES:
            raise CommandProblem("invalid_command_size", f"Invalid command size: {file_stat.st_size}")
        try:
            command = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CommandProblem("invalid_json", f"Invalid command JSON: {exc}") from exc
        if not isinstance(command, dict):
            raise CommandProblem("invalid_command", "Command must be a JSON object")

        required = (
            "schema_version",
            "run_id",
            "command_id",
            "idempotency_key",
            "action",
            "created_at",
        )
        missing = [name for name in required if name not in command]
        if missing:
            raise CommandProblem("missing_fields", f"Missing command fields: {', '.join(missing)}")
        if command.get("schema_version") != SCHEMA_VERSION:
            raise CommandProblem("schema_mismatch", "Unsupported command schema_version")
        if command.get("run_id") != self.run_id:
            raise CommandProblem("run_id_mismatch", "Command run_id does not match recorder run")
        action = command.get("action")
        if action not in VALID_ACTIONS:
            raise CommandProblem("invalid_action", f"Unsupported recorder action: {action!r}")
        for field in ("command_id", "idempotency_key"):
            value = command.get(field)
            if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
                raise CommandProblem("unsafe_identifier", f"Unsafe {field}: {value!r}")
        if path.stem != command["command_id"]:
            raise CommandProblem("command_id_mismatch", "Command filename does not match command_id")
        expected_revision = command.get("expected_revision")
        if expected_revision is not None and not isinstance(expected_revision, int):
            raise CommandProblem("invalid_revision", "expected_revision must be an integer or null")
        try:
            home_target = normalize_home_target(command.get("home_target"))
        except HomeTransitionError as exc:
            raise CommandProblem("invalid_home_target", str(exc)) from exc
        if home_target is not None and action not in {
            "episode_accept",
            "episode_discard",
        }:
            raise CommandProblem(
                "invalid_home_target",
                "home_target is valid only for whole-episode commands",
            )
        if home_target is not None:
            command["home_target"] = home_target
        return command

    @staticmethod
    def _fingerprint(command: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": command.get("run_id"),
            "idempotency_key": command.get("idempotency_key"),
            "action": command.get("action"),
            "episode_id": command.get("episode_id"),
            "home_target": command.get("home_target"),
        }

    def _make_ack(
        self,
        command: Mapping[str, Any],
        status: str,
        *,
        error: Any = None,
    ) -> Dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "owner_pid": self.owner_pid,
            "command_id": command.get("command_id"),
            "idempotency_key": command.get("idempotency_key"),
            "action": command.get("action"),
            "status": status,
            "episode_id": command.get("episode_id"),
            "revision": snapshot["revision"],
            "error": error,
            "recording": bool(snapshot.get("active")),
            "updated_at": utc_now(),
        }

    def _journal_event(self, event: Mapping[str, Any]) -> None:
        durable = append_jsonl_durable(self.journal_path, event, crc32=False)
        key = durable.get("idempotency_key")
        if isinstance(key, str) and key:
            self._journal[key] = durable

    def _write_ack(self, ack: Mapping[str, Any]) -> None:
        command_id = ack.get("command_id")
        if not isinstance(command_id, str) or not _SAFE_ID_RE.fullmatch(command_id):
            return
        atomic_write_json(self.acks_dir / f"{command_id}.json", ack)

    def _pending_command_paths(self) -> List[Path]:
        paths: List[Tuple[int, str, Path]] = []
        for path in self.commands_dir.glob("*.json"):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            paths.append((info.st_mtime_ns, path.name, path))
        paths.sort()
        return [item[2] for item in paths]

    def process_commands(
        self,
        handler: Callable[[Mapping[str, Any]], None],
        *,
        allowed_actions: Optional[Iterable[str]] = None,
    ) -> int:
        """Validate and apply all currently spooled commands in creation order."""
        allowed = set(allowed_actions) if allowed_actions is not None else None
        paths = self._pending_command_paths()
        if not paths:
            with self._lock:
                if self._status.get("pending"):
                    self._status["pending"] = []
                    self._flush_status(force=True)
            return 0
        with self._lock:
            pending_ids = [path.stem for path in paths]
            if self._status.get("pending") != pending_ids:
                self._status["pending"] = pending_ids
                self._flush_status(force=True)

        processed = 0
        for path in paths:
            command: Dict[str, Any] = {
                "command_id": path.stem,
                "idempotency_key": path.stem,
                "action": None,
                "episode_id": None,
            }
            try:
                command = self._validate_command_file(path)
                if allowed is not None and command["action"] not in allowed:
                    continue
                # Another command owner may have appended to the shared journal
                # since this recorder process started.
                self._journal = self._load_journal()
                fingerprint = self._fingerprint(command)
                previous = self._journal.get(command["idempotency_key"])
                if previous is not None:
                    if previous.get("fingerprint") != fingerprint:
                        raise CommandProblem(
                            "idempotency_conflict",
                            "idempotency_key was already used for a different command",
                        )
                    previous_ack = previous.get("ack")
                    if previous.get("phase") == "completed" and isinstance(previous_ack, dict):
                        replay = dict(previous_ack)
                        replay["command_id"] = command["command_id"]
                        replay["owner_pid"] = self.owner_pid
                        replay["replayed"] = True
                        self._write_ack(replay)
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass
                        processed += 1
                        continue
                    raise CommandProblem(
                        "indeterminate_previous_attempt",
                        "A previous attempt began but did not durably record completion",
                        True,
                    )

                expected_revision = command.get("expected_revision")
                current_revision = int(self.snapshot()["revision"])
                if expected_revision is not None and expected_revision != current_revision:
                    raise CommandProblem(
                        "revision_conflict",
                        f"expected_revision={expected_revision}, current_revision={current_revision}",
                    )

                started = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "owner_pid": self.owner_pid,
                    "phase": "started",
                    "idempotency_key": command["idempotency_key"],
                    "command_id": command["command_id"],
                    "fingerprint": fingerprint,
                    "created_at": utc_now(),
                }
                self._journal_event(started)
                handler(command)
                ack = self._make_ack(command, "applied")
            except CommandProblem as exc:
                ack = self._make_ack(
                    command,
                    "failed" if exc.failed else "rejected",
                    error={"code": exc.code, "message": exc.message},
                )
            except Exception as exc:  # Fault isolation: never kill the saver loop.
                ack = self._make_ack(
                    command,
                    "failed",
                    error={"code": "handler_error", "message": f"{type(exc).__name__}: {exc}"},
                )

            completed = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "owner_pid": self.owner_pid,
                "phase": "completed",
                "idempotency_key": command.get("idempotency_key"),
                "command_id": command.get("command_id"),
                "fingerprint": self._fingerprint(command),
                "ack": ack,
                "created_at": utc_now(),
            }
            self._journal_event(completed)
            self._write_ack(ack)
            processed += 1
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        remaining_ids = [path.stem for path in self._pending_command_paths()]
        with self._lock:
            if self._status.get("pending") != remaining_ids:
                self._status["pending"] = remaining_ids
                self._flush_status(force=True)
        return processed


class WholeEpisodeCommandConsumer:
    """Inference-process owner for whole-episode commands.

    The recorder process deliberately ignores these actions.  This consumer
    invokes PolicyConnector's existing halt/reset/home transition, then writes
    an ACK only after the recorder reports the next episode as active.
    """

    ACTIONS = frozenset({"episode_accept", "episode_discard"})

    def __init__(self, control_dir: Path, run_id: str, *, owner_pid: Optional[int] = None) -> None:
        if not _SAFE_ID_RE.fullmatch(run_id):
            raise ValueError(f"Unsafe SUPERINFERENCE_RUN_ID: {run_id!r}")
        self.control_dir = Path(control_dir).expanduser().resolve()
        self.run_id = run_id
        self.owner_pid = int(owner_pid or os.getpid())
        self.commands_dir = self.control_dir / "commands"
        self.acks_dir = self.control_dir / "acks"
        self.status_path = self.control_dir / "status" / "recorder.json"
        self.journal_path = self.control_dir / "command_journal.jsonl"
        for directory in (
            self.control_dir,
            self.commands_dir,
            self.acks_dir,
            self.status_path.parent,
        ):
            ensure_private_directory(directory)

    @classmethod
    def from_environment(cls) -> Optional["WholeEpisodeCommandConsumer"]:
        control_dir = os.environ.get("SUPERINFERENCE_RECORDING_CONTROL_DIR")
        run_id = os.environ.get("SUPERINFERENCE_RUN_ID")
        if not control_dir and not run_id:
            return None
        if not control_dir or not run_id:
            raise ValueError(
                "SUPERINFERENCE_RECORDING_CONTROL_DIR and SUPERINFERENCE_RUN_ID must be set together"
            )
        return cls(Path(control_dir), run_id)

    def read_status(self) -> Dict[str, Any]:
        try:
            info = self.status_path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise CommandProblem(
                    "unsafe_status_file", "Recorder status is not a regular non-symlink file", True
                )
            if info.st_size <= 0 or info.st_size > MAX_COMMAND_BYTES * 4:
                raise CommandProblem("invalid_status_size", "Recorder status size is invalid", True)
            status_payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CommandProblem("recorder_status_missing", "Recorder status is not available", True) from exc
        except json.JSONDecodeError as exc:
            raise CommandProblem("invalid_recorder_status", f"Invalid recorder status JSON: {exc}", True) from exc
        if status_payload.get("schema_version") != SCHEMA_VERSION:
            raise CommandProblem("status_schema_mismatch", "Recorder status schema mismatch", True)
        if status_payload.get("run_id") != self.run_id:
            raise CommandProblem("status_run_id_mismatch", "Recorder status belongs to another run", True)
        return status_payload

    def _pending_paths(self) -> List[Path]:
        sortable: List[Tuple[int, str, Path]] = []
        for path in self.commands_dir.glob("*.json"):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            sortable.append((info.st_mtime_ns, path.name, path))
        sortable.sort()
        return [item[2] for item in sortable]

    def _peek_action(self, path: Path) -> Optional[str]:
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                return None
            if info.st_size <= 0 or info.st_size > MAX_COMMAND_BYTES:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("action") if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _validate(self, path: Path) -> Dict[str, Any]:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise CommandProblem("unsafe_command_file", "Command is not a regular file", True)
        if info.st_size <= 0 or info.st_size > MAX_COMMAND_BYTES:
            raise CommandProblem("invalid_command_size", "Command size is invalid")
        try:
            command = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CommandProblem("invalid_json", f"Invalid command JSON: {exc}") from exc
        required = (
            "schema_version",
            "run_id",
            "command_id",
            "idempotency_key",
            "action",
            "episode_id",
            "created_at",
        )
        if not isinstance(command, dict):
            raise CommandProblem("invalid_command", "Command must be a JSON object")
        missing = [field for field in required if field not in command]
        if missing:
            raise CommandProblem("missing_fields", f"Missing fields: {', '.join(missing)}")
        if command.get("schema_version") != SCHEMA_VERSION:
            raise CommandProblem("schema_mismatch", "Unsupported command schema")
        if command.get("run_id") != self.run_id:
            raise CommandProblem("run_id_mismatch", "Command belongs to another run")
        if command.get("action") not in self.ACTIONS:
            raise CommandProblem("invalid_action", "Not a whole-episode command")
        for field in ("command_id", "idempotency_key"):
            if not isinstance(command.get(field), str) or not _SAFE_ID_RE.fullmatch(command[field]):
                raise CommandProblem("unsafe_identifier", f"Unsafe {field}")
        if command["command_id"] != path.stem:
            raise CommandProblem("command_id_mismatch", "Filename does not match command_id")
        expected_revision = command.get("expected_revision")
        if expected_revision is not None and not isinstance(expected_revision, int):
            raise CommandProblem("invalid_revision", "expected_revision must be int or null")
        try:
            home_target = normalize_home_target(command.get("home_target"))
        except HomeTransitionError as exc:
            raise CommandProblem("invalid_home_target", str(exc)) from exc
        if home_target is not None:
            command["home_target"] = home_target
        return command

    def _load_journal(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        if not self.journal_path.exists():
            return result
        with self.journal_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = item.get("idempotency_key")
                if isinstance(key, str) and key:
                    result[key] = item
        return result

    @staticmethod
    def _fingerprint(command: Mapping[str, Any]) -> Dict[str, Any]:
        return RecordingControlPlane._fingerprint(command)

    def _write_ack(self, ack: Mapping[str, Any]) -> None:
        atomic_write_json(self.acks_dir / f"{ack['command_id']}.json", ack)

    def _make_ack(self, command: Mapping[str, Any], result: str, error: Any = None) -> Dict[str, Any]:
        try:
            recorder_status = self.read_status()
        except CommandProblem:
            recorder_status = {"revision": -1, "active": False}
        ack = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "owner_pid": self.owner_pid,
            "command_id": command.get("command_id"),
            "idempotency_key": command.get("idempotency_key"),
            "action": command.get("action"),
            "status": result,
            "episode_id": command.get("episode_id"),
            "revision": recorder_status.get("revision", -1),
            "error": error,
            "recording": bool(recorder_status.get("active")),
            "updated_at": utc_now(),
        }
        if result == "applied" and command.get("action") in self.ACTIONS:
            receipt = self._read_finalized_receipt(command.get("episode_id"))
            if receipt is not None:
                ack["finalized_source"] = receipt
        return ack

    def _read_finalized_receipt(self, episode_id: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(episode_id, str) or not _SAFE_ID_RE.fullmatch(episode_id):
            return None
        path = self.control_dir / "finalized" / f"{episode_id}.json"
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                return None
            if info.st_size <= 0 or info.st_size > MAX_COMMAND_BYTES * 4:
                return None
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(receipt, dict):
            return None
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("episode_id") != episode_id
            or receipt.get("result") not in {"success", "failure", "incomplete"}
        ):
            return None
        return receipt

    def process_commands(self, handler: Callable[[Mapping[str, Any]], None]) -> int:
        processed = 0
        for path in self._pending_paths():
            if self._peek_action(path) not in self.ACTIONS:
                continue
            command: Dict[str, Any] = {
                "command_id": path.stem,
                "idempotency_key": path.stem,
                "action": self._peek_action(path),
                "episode_id": None,
            }
            try:
                command = self._validate(path)
                fingerprint = self._fingerprint(command)
                previous = self._load_journal().get(command["idempotency_key"])
                if previous is not None:
                    if previous.get("fingerprint") != fingerprint:
                        raise CommandProblem(
                            "idempotency_conflict",
                            "idempotency_key was used for another command",
                        )
                    if previous.get("phase") == "completed" and isinstance(previous.get("ack"), dict):
                        replay = dict(previous["ack"])
                        replay["command_id"] = command["command_id"]
                        replay["owner_pid"] = self.owner_pid
                        replay["replayed"] = True
                        self._write_ack(replay)
                        receipt = replay.get("finalized_source")
                        if replay.get("status") == "applied" and isinstance(receipt, Mapping):
                            write_recording_decision(receipt, command)
                        path.unlink(missing_ok=True)
                        processed += 1
                        continue
                    raise CommandProblem(
                        "indeterminate_previous_attempt",
                        "Previous whole-episode transition did not durably complete",
                        True,
                    )

                recorder_status = self.read_status()
                episode = recorder_status.get("episode")
                if not recorder_status.get("active") or not isinstance(episode, dict):
                    raise CommandProblem("not_recording", "No recording episode is active")
                if command.get("episode_id") != episode.get("id"):
                    raise CommandProblem("episode_conflict", "Command does not target the active episode")
                expected = command.get("expected_revision")
                current = int(recorder_status.get("revision", -1))
                if expected is not None and expected != current:
                    raise CommandProblem(
                        "revision_conflict",
                        f"expected_revision={expected}, current_revision={current}",
                    )

                started = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "owner_pid": self.owner_pid,
                    "phase": "started",
                    "idempotency_key": command["idempotency_key"],
                    "command_id": command["command_id"],
                    "fingerprint": fingerprint,
                    "created_at": utc_now(),
                }
                append_jsonl_durable(self.journal_path, started)
                handler(command)
                ack = self._make_ack(command, "applied")
            except CommandProblem as exc:
                ack = self._make_ack(
                    command,
                    "failed" if exc.failed else "rejected",
                    {"code": exc.code, "message": exc.message},
                )
            except Exception as exc:
                ack = self._make_ack(
                    command,
                    "failed",
                    {"code": "transition_error", "message": f"{type(exc).__name__}: {exc}"},
                )

            completed = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "owner_pid": self.owner_pid,
                "phase": "completed",
                "idempotency_key": command.get("idempotency_key"),
                "command_id": command.get("command_id"),
                "fingerprint": self._fingerprint(command),
                "ack": ack,
                "created_at": utc_now(),
            }
            append_jsonl_durable(self.journal_path, completed)
            self._write_ack(ack)
            receipt = ack.get("finalized_source")
            if ack.get("status") == "applied" and isinstance(receipt, Mapping):
                write_recording_decision(receipt, command)
            path.unlink(missing_ok=True)
            processed += 1
        return processed

    def wait_for_next_episode(
        self,
        previous_episode_id: str,
        *,
        auto_restart: bool,
        timeout_s: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() < deadline:
            status_payload = self.read_status()
            state = status_payload.get("status")
            if state in {"error", "incomplete", "stopped"}:
                raise CommandProblem(
                    "recorder_transition_failed",
                    f"Recorder entered terminal state {state}: {status_payload.get('error')}",
                    True,
                )
            episode = status_payload.get("episode")
            if auto_restart:
                if (
                    state == "recording"
                    and status_payload.get("active")
                    and isinstance(episode, dict)
                    and episode.get("id") != previous_episode_id
                ):
                    return status_payload
            elif state == "armed" and not status_payload.get("active"):
                return status_payload
            time.sleep(0.05)
        raise CommandProblem(
            "recorder_transition_timeout",
            "Timed out waiting for recorder to arm the next episode",
            True,
        )


def write_incomplete_metadata(
    episode_dir: Path,
    *,
    episode_id: Optional[str],
    episode_index: int,
    reason: str,
    error: Any = None,
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode_result": "incomplete",
        "is_success": False,
        "episode_id": episode_id,
        "episode_index": int(episode_index),
        "reason": reason,
        "error": error,
        "timestamp": utc_now(),
    }
    path = Path(episode_dir) / "episode_incomplete.json"
    atomic_write_json(path, payload)
    return path


def quarantine_episode(
    episode_dir: Path,
    output_dir: Path,
    *,
    episode_index: int,
    episode_id: Optional[str],
) -> Tuple[Path, bool]:
    """Rename a temp episode to an explicit non-training incomplete directory.

    On any rename failure the original temp directory is deliberately retained.
    """
    episode_dir = Path(episode_dir)
    output_dir = Path(output_dir)
    suffix = (episode_id or "unknown").replace("-", "")[:12]
    target = output_dir / f"episode_{int(episode_index):04d}_incomplete_{suffix}"
    counter = 1
    while target.exists():
        target = output_dir / f"episode_{int(episode_index):04d}_incomplete_{suffix}_{counter}"
        counter += 1
    try:
        os.replace(episode_dir, target)
        _fsync_directory(output_dir)
        return target, True
    except OSError:
        return episode_dir, False


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_conversion_source_receipt(
    episode_dir: Path,
    *,
    episode_id: str,
    episode_index: int,
    result: str,
    control: Optional[RecordingControlPlane] = None,
    config_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Publish the finalized source identity inside the episode and run spool.

    Large MCAP payloads are represented by stable stat identities so receipt
    generation remains off the recorder's critical path. Small metadata and
    dense-label sidecars are content hashed.
    """
    final_dir = Path(episode_dir).expanduser().resolve(strict=True)
    if result not in {"success", "failure", "incomplete"}:
        raise ValueError(f"Unsupported finalized episode result: {result!r}")
    metadata_path = final_dir / "metadata.json"
    try:
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Finalized episode has unreadable metadata.json") from exc
    raw_data_format = str(metadata_payload.get("data_format") or "").upper()
    data_format = raw_data_format.rsplit(".", 1)[-1]
    if data_format != "NEDF3":
        raise ValueError(f"Unsupported finalized episode format: {raw_data_format!r}")
    files: List[Dict[str, Any]] = []
    for path in sorted(final_dir.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.name in {".conversion_source.json", ".recording_decision.json"}
        ):
            continue
        info = path.stat()
        files.append(
            {
                "relpath": path.relative_to(final_dir).as_posix(),
                "size": int(info.st_size),
                "mtime_ns": int(info.st_mtime_ns),
            }
        )
    identity = {
        "episode_id": str(episode_id),
        "episode_index": int(episode_index),
        "result": result,
        "data_format": data_format,
        "path": str(final_dir),
        "files": files,
        "metadata_sha256": _sha256_file(metadata_path),
        "episode_result_sha256": _sha256_file(final_dir / "episode_result.json"),
        "episode_incomplete_sha256": _sha256_file(final_dir / "episode_incomplete.json"),
        "dense_labels_sha256": _sha256_file(final_dir / "dense_labels.jsonl"),
    }
    fingerprint = hashlib.sha256(_canonical_json(identity)).hexdigest()
    receipt: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "fingerprint": fingerprint,
        "config_name": config_name,
        "decision_required": True,
        "finalized_at": utc_now(),
    }
    atomic_write_json(final_dir / ".conversion_source.json", receipt)
    if control is not None:
        finalized_dir = control.control_dir / "finalized"
        ensure_private_directory(finalized_dir)
        atomic_write_json(finalized_dir / f"{episode_id}.json", receipt)
    return receipt


def write_recording_decision(
    receipt: Mapping[str, Any], command: Mapping[str, Any]
) -> Dict[str, Any]:
    """Publish the user decision only after its whole-operation ACK is durable."""
    action = command.get("action")
    status = {"episode_accept": "accepted", "episode_discard": "rejected"}.get(action)
    if status is None:
        raise ValueError(f"Unsupported recording decision action: {action!r}")
    episode_id = receipt.get("episode_id")
    episode_path = receipt.get("path")
    if not isinstance(episode_id, str) or not isinstance(episode_path, str):
        raise ValueError("Finalized receipt has no episode identity")
    final_dir = Path(episode_path).expanduser().resolve(strict=True)
    marker = json.loads((final_dir / ".conversion_source.json").read_text(encoding="utf-8"))
    if marker.get("fingerprint") != receipt.get("fingerprint") or marker.get("episode_id") != episode_id:
        raise ValueError("Finalized receipt does not match the episode marker")
    decision = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "action": action,
        "status": status,
        "command_id": command.get("command_id"),
        "idempotency_key": command.get("idempotency_key"),
        "decided_at": utc_now(),
    }
    atomic_write_json(final_dir / ".recording_decision.json", decision)
    return decision
