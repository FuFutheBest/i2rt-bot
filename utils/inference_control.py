"""Private command spool for controlling an already-loaded inference process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Dict, Mapping, Optional
import uuid

from .home_transition import HomeTransitionError, normalize_home_target
from .recording_control import (
    MAX_COMMAND_BYTES,
    SCHEMA_VERSION,
    atomic_write_json,
    ensure_private_directory,
    utc_now,
)


DEFAULT_INFERENCE_CONTROL_RUN_ROOT = Path(
    "/tmp/superinference-control-panel/inference-runs"
)
INFERENCE_CONTROL_ACTION = "reset"
INFERENCE_CONTROL_ACTIONS = frozenset({"pause", "reset", "resume"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_ACK_STATUSES = frozenset({"applied", "rejected", "failed"})


class InferenceControlError(RuntimeError):
    """Base error for the inference command channel."""


class InferenceControlConflictError(InferenceControlError):
    """The inference session changed or already has a pending command."""


class InferenceControlRequestError(InferenceControlError):
    """An inference-control request has an invalid public shape."""


class InferenceControlSecurityError(InferenceControlError):
    """A spool artifact failed ownership or schema validation."""


def _safe_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise InferenceControlRequestError(f"{field} is invalid")
    return value


def _read_json_file(path: Path, *, max_bytes: int = MAX_COMMAND_BYTES) -> Dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise InferenceControlSecurityError(f"Cannot inspect {path.name}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise InferenceControlSecurityError(f"{path.name} is not a regular file")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise InferenceControlSecurityError(f"{path.name} has an invalid size")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceControlSecurityError(f"{path.name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise InferenceControlSecurityError(f"{path.name} must contain an object")
    return payload


class InferenceControlCoordinator:
    """Web-process owner of one already-loaded inference command spool."""

    def __init__(self, run_root: Path = DEFAULT_INFERENCE_CONTROL_RUN_ROOT) -> None:
        self.run_root = Path(run_root)
        self.run_id: Optional[str] = None
        self.control_dir: Optional[Path] = None
        self.owner_pid: Optional[int] = None
        self._commands: Dict[str, Dict[str, Any]] = {}
        self._acks: Dict[str, Dict[str, Any]] = {}

    @property
    def commands_dir(self) -> Path:
        return self._required_control_dir() / "commands"

    @property
    def acks_dir(self) -> Path:
        return self._required_control_dir() / "acks"

    def _required_control_dir(self) -> Path:
        if self.control_dir is None:
            raise InferenceControlConflictError("Inference control run is not prepared")
        return self.control_dir

    def prepare_run(self) -> Path:
        if self.control_dir is not None:
            return self.control_dir
        ensure_private_directory(self.run_root)
        run_id = str(uuid.uuid4())
        control_dir = self.run_root / run_id
        ensure_private_directory(control_dir)
        ensure_private_directory(control_dir / "commands")
        ensure_private_directory(control_dir / "acks")
        self.run_id = run_id
        self.control_dir = control_dir
        atomic_write_json(
            control_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "owner_pid": None,
                "created_at": utc_now(),
            },
        )
        return control_dir

    def bind_owner(self, owner_pid: int) -> None:
        if isinstance(owner_pid, bool) or not isinstance(owner_pid, int) or owner_pid <= 0:
            raise InferenceControlSecurityError("Inference process PID is invalid")
        self.owner_pid = int(owner_pid)
        atomic_write_json(
            self._required_control_dir() / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "owner_pid": self.owner_pid,
                "created_at": utc_now(),
            },
        )

    def _validate_ack(self, path: Path, command: Mapping[str, Any]) -> Dict[str, Any]:
        payload = _read_json_file(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise InferenceControlSecurityError("Inference ACK schema mismatch")
        if payload.get("run_id") != self.run_id:
            raise InferenceControlSecurityError("Inference ACK belongs to another run")
        if payload.get("owner_pid") != self.owner_pid:
            raise InferenceControlSecurityError("Inference ACK owner mismatch")
        if payload.get("command_id") != command.get("command_id"):
            raise InferenceControlSecurityError("Inference ACK command mismatch")
        if payload.get("idempotency_key") != command.get("idempotency_key"):
            raise InferenceControlSecurityError("Inference ACK idempotency mismatch")
        if payload.get("action") != command.get("action"):
            raise InferenceControlSecurityError("Inference ACK action mismatch")
        if payload.get("status") not in _ACK_STATUSES:
            raise InferenceControlSecurityError("Inference ACK status is invalid")
        if not isinstance(payload.get("inference_paused"), bool):
            raise InferenceControlSecurityError(
                "Inference ACK paused state is invalid"
            )
        if "reset_pending" in payload and not isinstance(payload["reset_pending"], bool):
            raise InferenceControlSecurityError(
                "Inference ACK reset state is invalid"
            )
        payload.setdefault("reset_pending", False)
        return payload

    def _refresh_acks(self) -> None:
        for command_id, command in tuple(self._commands.items()):
            path = self.acks_dir / f"{command_id}.json"
            if not path.exists():
                continue
            # ACKs are normally immutable, but reset completion updates the
            # paused bit after the robot consumes its Home signal.
            self._acks[command_id] = self._validate_ack(path, command)

    def _public_command(self, command: Mapping[str, Any]) -> Dict[str, Any]:
        command_id = str(command["command_id"])
        ack = self._acks.get(command_id)
        if ack is not None:
            return dict(ack)
        return {
            "command_id": command_id,
            "idempotency_key": command["idempotency_key"],
            "action": command["action"],
            "status": "pending",
            "error": None,
            "created_at": command.get("created_at"),
        }

    def state(self) -> Dict[str, Any]:
        self._refresh_acks()
        commands = list(self._commands.values())
        pending = next(
            (
                self._public_command(command)
                for command in commands
                if str(command["command_id"]) not in self._acks
            ),
            None,
        )
        last = self._public_command(commands[-1]) if commands else None
        paused = False
        reset_pending = False
        for command in commands:
            ack = self._acks.get(str(command["command_id"]))
            if ack is not None and isinstance(ack.get("inference_paused"), bool):
                paused = bool(ack["inference_paused"])
                reset_pending = bool(ack.get("reset_pending", False))
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "available": self.control_dir is not None and self.owner_pid is not None,
            "paused": paused,
            "reset_pending": reset_pending,
            "pending_command": pending,
            "last_command": last,
        }

    def _submit(
        self,
        *,
        action: str,
        idempotency_key: str,
        home_target: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if action not in INFERENCE_CONTROL_ACTIONS:
            raise InferenceControlRequestError("Unsupported inference action")
        safe_idempotency = _safe_id(idempotency_key, field="idempotency_key")
        normalized_home_target = None
        if action == "reset":
            try:
                normalized_home_target = normalize_home_target(home_target)
            except HomeTransitionError as exc:
                raise InferenceControlRequestError(str(exc)) from exc
        elif home_target is not None:
            raise InferenceControlRequestError(
                "home_target is only valid for reset"
            )

        self._refresh_acks()
        for existing in self._commands.values():
            if existing.get("idempotency_key") != safe_idempotency:
                continue
            if (
                existing.get("action") != action
                or existing.get("home_target") != normalized_home_target
            ):
                raise InferenceControlConflictError(
                    "idempotency_key was already used for another inference command"
                )
            return self._public_command(existing)
        if any(command_id not in self._acks for command_id in self._commands):
            raise InferenceControlConflictError(
                "An inference control command is already pending"
            )
        if self.owner_pid is None:
            raise InferenceControlConflictError("Inference control is not ready")

        command = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "command_id": str(uuid.uuid4()),
            "idempotency_key": safe_idempotency,
            "action": action,
            "created_at": utc_now(),
        }
        if normalized_home_target is not None:
            command["home_target"] = normalized_home_target
        atomic_write_json(
            self.commands_dir / f"{command['command_id']}.json",
            command,
        )
        self._commands[str(command["command_id"])] = command
        return self._public_command(command)

    def submit_pause(self, *, idempotency_key: str) -> Dict[str, Any]:
        return self._submit(action="pause", idempotency_key=idempotency_key)

    def submit_reset(
        self,
        *,
        idempotency_key: str,
        home_target: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._submit(
            action="reset",
            idempotency_key=idempotency_key,
            home_target=home_target,
        )

    def submit_resume(self, *, idempotency_key: str) -> Dict[str, Any]:
        return self._submit(action="resume", idempotency_key=idempotency_key)


class InferenceControlConsumer:
    """Inference-process consumer that ACKs only after an action is applied."""

    def __init__(
        self,
        control_dir: Path,
        run_id: str,
        *,
        owner_pid: Optional[int] = None,
    ) -> None:
        self.control_dir = Path(control_dir).expanduser().resolve()
        self.run_id = _safe_id(run_id, field="run_id")
        self.owner_pid = int(owner_pid or os.getpid())
        self.commands_dir = self.control_dir / "commands"
        self.acks_dir = self.control_dir / "acks"
        ensure_private_directory(self.control_dir)
        ensure_private_directory(self.commands_dir)
        ensure_private_directory(self.acks_dir)

    @classmethod
    def from_environment(cls) -> Optional["InferenceControlConsumer"]:
        control_dir = os.environ.get("SUPERINFERENCE_INFERENCE_CONTROL_DIR")
        run_id = os.environ.get("SUPERINFERENCE_INFERENCE_CONTROL_RUN_ID")
        if not control_dir and not run_id:
            return None
        if not control_dir or not run_id:
            raise ValueError(
                "SUPERINFERENCE_INFERENCE_CONTROL_DIR and "
                "SUPERINFERENCE_INFERENCE_CONTROL_RUN_ID must be set together"
            )
        return cls(Path(control_dir), run_id)

    def _pending_paths(self) -> list[Path]:
        paths = list(self.commands_dir.glob("*.json"))
        paths.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
        return paths

    def _validate_command(self, path: Path) -> Dict[str, Any]:
        command = _read_json_file(path)
        if command.get("schema_version") != SCHEMA_VERSION:
            raise InferenceControlRequestError("Unsupported command schema")
        if command.get("run_id") != self.run_id:
            raise InferenceControlRequestError("Command belongs to another run")
        action = command.get("action")
        if action not in INFERENCE_CONTROL_ACTIONS:
            raise InferenceControlRequestError("Unsupported inference action")
        command_id = _safe_id(command.get("command_id"), field="command_id")
        _safe_id(command.get("idempotency_key"), field="idempotency_key")
        if command_id != path.stem:
            raise InferenceControlRequestError("Filename does not match command_id")
        if action == "reset":
            try:
                normalized = normalize_home_target(command.get("home_target"))
            except HomeTransitionError as exc:
                raise InferenceControlRequestError(str(exc)) from exc
            if normalized is not None:
                command["home_target"] = normalized
        elif "home_target" in command:
            raise InferenceControlRequestError(
                "home_target is only valid for reset"
            )
        return command

    def _ack(
        self,
        command: Mapping[str, Any],
        status: str,
        *,
        error: Optional[Mapping[str, str]] = None,
        inference_paused: bool,
        reset_pending: bool = False,
    ) -> None:
        atomic_write_json(
            self.acks_dir / f"{command.get('command_id')}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "owner_pid": self.owner_pid,
                "command_id": command.get("command_id"),
                "idempotency_key": command.get("idempotency_key"),
                "action": command.get("action"),
                "status": status,
                "error": dict(error) if error is not None else None,
                "inference_paused": bool(inference_paused),
                "reset_pending": bool(reset_pending),
                "updated_at": utc_now(),
            },
        )

    def process_commands(
        self,
        handler: Callable[[Dict[str, Any]], None],
        *,
        state_provider: Callable[[], Mapping[str, Any]],
    ) -> int:
        processed = 0
        for path in self._pending_paths():
            command: Dict[str, Any] = {"command_id": path.stem}
            try:
                command = self._validate_command(path)
                ack_path = self.acks_dir / f"{command['command_id']}.json"
                if ack_path.exists():
                    path.unlink(missing_ok=True)
                    processed += 1
                    continue
                handler(command)
                state = state_provider()
                self._ack(
                    command,
                    "applied",
                    inference_paused=bool(state.get("inference_paused")),
                    reset_pending=bool(state.get("reset_pending")),
                )
            except InferenceControlRequestError as exc:
                command.setdefault("action", INFERENCE_CONTROL_ACTION)
                state = state_provider()
                self._ack(
                    command,
                    "rejected",
                    error={"code": "invalid_command", "message": str(exc)},
                    inference_paused=bool(state.get("inference_paused")),
                    reset_pending=bool(state.get("reset_pending")),
                )
            except Exception as exc:
                command.setdefault("action", INFERENCE_CONTROL_ACTION)
                state = state_provider()
                self._ack(
                    command,
                    "failed",
                    error={
                        "code": "handler_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                    inference_paused=bool(state.get("inference_paused")),
                    reset_pending=bool(state.get("reset_pending")),
                )
            path.unlink(missing_ok=True)
            processed += 1
        return processed

    def update_applied_state(
        self,
        command_id: str,
        *,
        inference_paused: bool,
        reset_pending: bool = False,
        status: str = "applied",
        error: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Update an applied command after an asynchronous reset completes."""
        safe_command_id = _safe_id(command_id, field="command_id")
        path = self.acks_dir / f"{safe_command_id}.json"
        payload = _read_json_file(path)
        if payload.get("run_id") != self.run_id or payload.get("owner_pid") != self.owner_pid:
            raise InferenceControlSecurityError("Inference ACK owner mismatch")
        if payload.get("status") != "applied":
            raise InferenceControlConflictError("Inference command was not applied")
        if status not in _ACK_STATUSES:
            raise InferenceControlRequestError("Inference ACK status is invalid")
        payload["inference_paused"] = bool(inference_paused)
        payload["reset_pending"] = bool(reset_pending)
        payload["status"] = status
        payload["error"] = dict(error) if error is not None else None
        payload["updated_at"] = utc_now()
        atomic_write_json(path, payload)


__all__ = [
    "DEFAULT_INFERENCE_CONTROL_RUN_ROOT",
    "INFERENCE_CONTROL_ACTION",
    "INFERENCE_CONTROL_ACTIONS",
    "InferenceControlConflictError",
    "InferenceControlConsumer",
    "InferenceControlCoordinator",
    "InferenceControlError",
    "InferenceControlRequestError",
    "InferenceControlSecurityError",
]
