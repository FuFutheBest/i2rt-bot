"""Authoritative, de-duplicated control-panel state observation.

The browser must never infer that a hardware transition succeeded merely from
the HTTP acknowledgement that accepted the command.  ``StateSyncBroker``
keeps one host-side sampler for all browser tabs, assigns an epoch/sequence to
every observation, and reconciles explicit transition tokens only against the
observed process, SHM, workflow, recorder, and gripper state.

Sampling is deliberately adaptive: transitions are checked quickly, healthy
active state backs off to one second, and an unchanged idle service backs off
to 2.5 seconds.  A blocked lifecycle call does not block ``current()``; callers
receive the last observation marked stale and all controls can fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional


TRANSITION_INTERVAL_S = 0.2
RECOVERY_INTERVAL_S = 0.25
ACTIVE_INTERVAL_S = 1.0
IDLE_INTERVAL_S = 2.5
MIN_STALE_AFTER_S = 1.5
MAX_STALE_AFTER_S = 7.5
COMPLETED_RETENTION_S = 3.0
FAILED_RETENTION_S = 15.0
MAX_TRANSITIONS = 128
MAX_TOKEN_HISTORY = 512
EXPECTED_HOME_TRANSITION_MAX_S = 90.0

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")
_TERMINAL_FAILURES = frozenset(
    {
        "failed",
        "error",
        "degraded",
        "stop_timeout",
        "orphaned",
        "external",
        "conflict",
        "unavailable",
    }
)
_TRANSITION_STATUSES = frozenset(
    {
        "starting",
        "stopping",
        "calibrating",
        "moving",
        "queued",
        "probing",
        "finalizing",
        "transitioning",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateTransitionConflict(RuntimeError):
    """A second tab attempted a conflicting transition for one resource."""


@dataclass
class _Transition:
    token: str
    scope: str
    action: str
    desired: str
    started_monotonic: float
    started_at: str
    timeout_s: float
    status: str = "pending"
    acknowledged: bool = False
    reason: Optional[str] = None
    finished_monotonic: Optional[float] = None
    finished_at: Optional[str] = None

    def public(self, now: float, *, duplicate: bool = False) -> Dict[str, Any]:
        elapsed_ms = max(0, round((now - self.started_monotonic) * 1000))
        deadline_ms = max(0, round(self.timeout_s * 1000))
        return {
            "token": self.token,
            "scope": self.scope,
            "action": self.action,
            "desired": self.desired,
            "status": self.status,
            "acknowledged": self.acknowledged,
            "reason": self.reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": elapsed_ms,
            "timeout_ms": deadline_ms,
            "remaining_ms": max(0, deadline_ms - elapsed_ms),
            "duplicate": duplicate,
        }


class StateSyncBroker:
    """Share one authoritative state sampler across all HTTP/SSE clients."""

    def __init__(
        self,
        source: Callable[[], Mapping[str, Any]],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = _utc_now,
        epoch_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._source = source
        self._monotonic = monotonic
        self._utc_now = utc_now
        self.epoch = str(epoch_factory())
        self._condition = threading.Condition(threading.RLock())
        self._sample_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot: Optional[Dict[str, Any]] = None
        self._sequence = 0
        self._observed_revision = 0
        self._event_revision = 0
        self._semantic_digest: Optional[str] = None
        self._last_sample_monotonic: Optional[float] = None
        self._expected_interval_s = IDLE_INTERVAL_S
        self._stable_samples = 0
        self._last_error: Optional[str] = None
        self._transitions: Dict[str, _Transition] = {}
        self._token_history: Dict[str, _Transition] = {}
        self._last_healthy: Dict[str, str] = {}

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="control-panel-state-sync",
                daemon=True,
            )
            self._thread.start()

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))

    def begin(
        self,
        scope: str,
        action: str,
        desired: str,
        *,
        timeout_ms: int,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_scope = str(scope).strip()
        if not normalized_scope or len(normalized_scope) > 256:
            raise ValueError("transition scope is invalid")
        requested_token = str(token or uuid.uuid4().hex).strip()
        if not _TOKEN_RE.fullmatch(requested_token):
            raise ValueError("transition token is invalid")
        now = self._monotonic()
        with self._condition:
            self._expire_locked(now)
            historical = self._token_history.get(requested_token)
            if historical is not None:
                if (
                    historical.scope == normalized_scope
                    and historical.action == str(action)
                    and historical.desired == str(desired)
                ):
                    return historical.public(now, duplicate=True)
                raise StateTransitionConflict(
                    "Transition token was already used for a different request"
                )
            existing = self._transitions.get(normalized_scope)
            if existing is not None and existing.status in {"pending", "acknowledged"}:
                if str(action) == "stop" or str(desired) == "stopped":
                    self._finish_locked(
                        existing,
                        "superseded",
                        "Superseded by a safety-reducing stop request",
                        now,
                    )
                else:
                    raise StateTransitionConflict(
                        f"Transition already active for {normalized_scope}"
                    )
            active_count = sum(
                item.status in {"pending", "acknowledged"}
                for item in self._transitions.values()
            )
            if (
                active_count >= MAX_TRANSITIONS
                and str(action) != "stop"
                and str(desired) != "stopped"
            ):
                raise StateTransitionConflict(
                    "Too many control transitions are already active"
                )
            transition = _Transition(
                token=requested_token,
                scope=normalized_scope,
                action=str(action),
                desired=str(desired),
                started_monotonic=now,
                started_at=self._utc_now(),
                timeout_s=max(0.1, float(timeout_ms) / 1000.0),
            )
            self._transitions[normalized_scope] = transition
            self._token_history[requested_token] = transition
            self._expected_interval_s = TRANSITION_INTERVAL_S
            self._stable_samples = 0
            self._trim_transitions_locked()
            self._trim_token_history_locked()
            self._event_revision += 1
            self._condition.notify_all()
        self._wake.set()
        return transition.public(now, duplicate=False)

    def acknowledge(self, token: str) -> Optional[Dict[str, Any]]:
        now = self._monotonic()
        with self._condition:
            transition = self._find_token_locked(token)
            if transition is None:
                return None
            if transition.status == "pending":
                transition.status = "acknowledged"
                transition.acknowledged = True
                self._event_revision += 1
                self._condition.notify_all()
            result = transition.public(now)
        self._wake.set()
        return result

    def fail(self, token: str, reason: Any) -> Optional[Dict[str, Any]]:
        now = self._monotonic()
        with self._condition:
            transition = self._find_token_locked(token)
            if transition is None:
                return None
            self._finish_locked(
                transition,
                "failed",
                str(reason or "Command failed"),
                now,
            )
            result = transition.public(now)
        self._wake.set()
        return result

    def current(self) -> Dict[str, Any]:
        if self._snapshot is None:
            self.sample_now()
        now = self._monotonic()
        with self._condition:
            self._expire_locked(now)
            if self._snapshot is None:
                raise RuntimeError(self._last_error or "No state observation available")
            return self._decorate_current_locked(copy.deepcopy(self._snapshot), now)

    def sample_if_due(self) -> Dict[str, Any]:
        """Refresh at most once per adaptive interval across concurrent tabs."""
        now = self._monotonic()
        with self._condition:
            due = bool(
                self._snapshot is None
                or self._last_sample_monotonic is None
                or now - self._last_sample_monotonic >= self._expected_interval_s
            )
        if not due:
            return self.current()
        if not self._sample_lock.acquire(blocking=False):
            # Another request/background sample owns the expensive process/SHM
            # probe. Return the prior observation immediately instead of making
            # this browser wait or duplicating work.
            with self._condition:
                if self._snapshot is None:
                    self._condition.wait_for(
                        lambda: self._snapshot is not None or self._last_error is not None,
                        timeout=max(MIN_STALE_AFTER_S, self._expected_interval_s),
                    )
                    if self._snapshot is None:
                        raise RuntimeError(
                            self._last_error or "Initial state observation timed out"
                        )
                return self._decorate_current_locked(
                    copy.deepcopy(self._snapshot), self._monotonic()
                )
        try:
            now = self._monotonic()
            with self._condition:
                due = bool(
                    self._snapshot is None
                    or self._last_sample_monotonic is None
                    or now - self._last_sample_monotonic
                    >= self._expected_interval_s
                )
            return self._sample_once() if due else self.current()
        finally:
            self._sample_lock.release()

    def wait_for_event(
        self, event_revision: int, timeout: float = 15.0
    ) -> tuple[int, Optional[Dict[str, Any]]]:
        deadline = self._monotonic() + max(0.0, float(timeout))
        with self._condition:
            if (
                int(event_revision) > self._event_revision
                and self._snapshot is not None
            ):
                # EventSource may reconnect after the control service restarted
                # and send a Last-Event-ID from the previous epoch. Never wait
                # for the new broker to count up to that unrelated revision.
                now = self._monotonic()
                return (
                    self._event_revision,
                    self._decorate_current_locked(
                        copy.deepcopy(self._snapshot), now
                    ),
                )
            while self._event_revision <= int(event_revision) and not self._stop.is_set():
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    return self._event_revision, None
                self._condition.wait(timeout=remaining)
                if (
                    int(event_revision) > self._event_revision
                    and self._snapshot is not None
                ):
                    now = self._monotonic()
                    return (
                        self._event_revision,
                        self._decorate_current_locked(
                            copy.deepcopy(self._snapshot), now
                        ),
                    )
            if self._snapshot is None:
                return self._event_revision, None
            now = self._monotonic()
            self._expire_locked(now)
            return (
                self._event_revision,
                self._decorate_current_locked(copy.deepcopy(self._snapshot), now),
            )

    def sample_now(self) -> Dict[str, Any]:
        with self._sample_lock:
            return self._sample_once()

    def _sample_once(self) -> Dict[str, Any]:
        try:
            raw = dict(self._source())
        except BaseException as exc:
            with self._condition:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._event_revision += 1
                self._condition.notify_all()
                if self._snapshot is not None:
                    return self._decorate_current_locked(
                        copy.deepcopy(self._snapshot), self._monotonic()
                    )
            raise

        now = self._monotonic()
        observed_at = self._utc_now()
        with self._condition:
            recovered_from_error = self._last_error is not None
            self._last_error = None
            self._last_sample_monotonic = now
            self._sequence += 1
            transition_changed = self._reconcile_locked(raw, now)
            semantic_digest = self._digest(raw)
            semantic_changed = semantic_digest != self._semantic_digest
            if semantic_changed:
                self._semantic_digest = semantic_digest
                self._observed_revision += 1
                self._stable_samples = 0
            elif recovered_from_error:
                # Transport/source recovery is itself authoritative new
                # information even when the device payload is byte-identical.
                self._observed_revision += 1
                self._stable_samples = 0
            else:
                self._stable_samples += 1
            self._expected_interval_s = self._choose_interval_locked(raw)
            self._snapshot = self._annotate_locked(
                copy.deepcopy(raw), observed_at, now
            )
            if semantic_changed or transition_changed or recovered_from_error:
                self._event_revision += 1
                self._condition.notify_all()
            return self._decorate_current_locked(
                copy.deepcopy(self._snapshot), now
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_now()
            except BaseException:
                pass
            with self._condition:
                delay = self._expected_interval_s
            self._wake.wait(timeout=max(0.05, delay))
            self._wake.clear()

    def _find_token_locked(self, token: str) -> Optional[_Transition]:
        requested = str(token)
        return next(
            (
                transition
                for transition in self._transitions.values()
                if transition.token == requested
            ),
            None,
        )

    def _finish_locked(
        self,
        transition: _Transition,
        status: str,
        reason: Optional[str],
        now: float,
    ) -> None:
        if transition.status == status and transition.finished_monotonic is not None:
            return
        transition.status = status
        transition.reason = reason
        transition.finished_monotonic = now
        transition.finished_at = self._utc_now()
        self._event_revision += 1
        self._condition.notify_all()

    def _expire_locked(self, now: float) -> None:
        remove: list[str] = []
        changed = False
        for scope, transition in self._transitions.items():
            if transition.status in {"pending", "acknowledged"}:
                if now - transition.started_monotonic >= transition.timeout_s:
                    self._finish_locked(
                        transition,
                        "timed_out",
                        "Observed state did not reach the requested target before timeout",
                        now,
                    )
                    changed = True
                continue
            finished = transition.finished_monotonic or transition.started_monotonic
            retention = (
                COMPLETED_RETENTION_S
                if transition.status == "completed"
                else FAILED_RETENTION_S
            )
            if now - finished >= retention:
                remove.append(scope)
        for scope in remove:
            self._transitions.pop(scope, None)
            changed = True
        if changed:
            self._event_revision += 1
            self._condition.notify_all()

    def _trim_transitions_locked(self) -> None:
        if len(self._transitions) <= MAX_TRANSITIONS:
            return
        ordered = sorted(
            self._transitions.items(),
            key=lambda item: item[1].started_monotonic,
        )
        for scope, transition in ordered:
            if len(self._transitions) <= MAX_TRANSITIONS:
                break
            if transition.status not in {"pending", "acknowledged"}:
                self._transitions.pop(scope, None)

    def _trim_token_history_locked(self) -> None:
        if len(self._token_history) <= MAX_TOKEN_HISTORY:
            return
        # Make at most one bounded pass. Live tokens remain indexed for
        # idempotency, so a temporary over-limit history is safer than either
        # replaying an in-flight command or looping forever while holding the
        # broker condition lock.
        for token, transition in list(self._token_history.items()):
            if len(self._token_history) <= MAX_TOKEN_HISTORY:
                break
            if transition.status not in {"pending", "acknowledged"}:
                self._token_history.pop(token, None)

    @staticmethod
    def _device_by_key(raw: Mapping[str, Any], key: str) -> Optional[Mapping[str, Any]]:
        devices = raw.get("devices")
        if not isinstance(devices, list):
            return None
        return next(
            (
                device
                for device in devices
                if isinstance(device, Mapping) and str(device.get("key")) == key
            ),
            None,
        )

    def _transition_result(
        self, raw: Mapping[str, Any], transition: _Transition
    ) -> tuple[Optional[str], Optional[str]]:
        scope = transition.scope
        desired = transition.desired
        if scope.startswith("device:"):
            device = self._device_by_key(raw, scope[len("device:") :])
            if device is None:
                return "failed", "Device disappeared from the selected config"
            status = str(device.get("status") or "unknown").lower()
            running = device.get("running") is True
            shm = device.get("shm") if isinstance(device.get("shm"), Mapping) else {}
            connected = shm.get("connected") is True
            ready = device.get("ready") is True
            if desired == "running":
                safety = (
                    device.get("safety")
                    if isinstance(device.get("safety"), Mapping)
                    else {}
                )
                estop = safety.get("estop")
                estop_active = (
                    estop.get("active") is True
                    if isinstance(estop, Mapping)
                    else estop is True
                )
                if (
                    safety.get("fail_closed") is True
                    and (
                        safety.get("initializing") is not True
                        or estop_active
                        or safety.get("fault") is True
                        or safety.get("operational") is False
                    )
                ):
                    return "failed", str(
                        safety.get("reason")
                        or "Motion-device safety state is fail-closed"
                    )
                if status == "running" and running and ready:
                    return "completed", None
                if status in _TERMINAL_FAILURES:
                    return "failed", self._device_reason(device)
            elif desired == "stopped":
                if status == "stopped" and not running and not connected:
                    return "completed", None
                if status in {"stop_timeout", "orphaned", "external", "failed"}:
                    return "failed", self._device_reason(device)
            return None, None

        workflow = raw.get("workflow")
        if scope.startswith("workflow:") and isinstance(workflow, Mapping):
            name = scope.split(":", 1)[1]
            stage = workflow.get(name)
            if not isinstance(stage, Mapping):
                return "failed", f"Workflow stage {name} disappeared"
            status = str(stage.get("status") or "unknown").lower()
            active = (
                stage.get("active") is True
                if name == "runtime"
                else stage.get("running") is True
            )
            ready = stage.get("ready") is True if name == "runtime" else active
            if desired == "running":
                if status == "running" and active and ready:
                    return "completed", None
                if status in {"failed", "error"}:
                    return "failed", str(stage.get("error") or f"{name} failed")
            elif desired == "stopped":
                if status == "stopped" and not active:
                    return "completed", None
                if status in {"failed", "error"} and active:
                    return "failed", str(stage.get("error") or f"{name} stop failed")
            return None, None

        if scope.startswith("gripper:"):
            grippers = raw.get("grippers")
            sides = grippers.get("sides") if isinstance(grippers, Mapping) else None
            side = scope.split(":", 1)[1]
            item = sides.get(side) if isinstance(sides, Mapping) else None
            if not isinstance(item, Mapping):
                return "failed", f"Gripper side {side} disappeared"
            status = str(item.get("status") or "unavailable").lower()
            if desired == "idle" and status == "idle" and item.get("busy") is not True:
                return "completed", None
            if status in {"error", "unavailable"}:
                return "failed", str(item.get("last_error") or f"{side} gripper failed")
            return None, None

        return None, None

    def _reconcile_locked(self, raw: Mapping[str, Any], now: float) -> bool:
        changed = False
        self._expire_locked(now)
        for transition in self._transitions.values():
            if transition.status not in {"pending", "acknowledged"}:
                continue
            status, reason = self._transition_result(raw, transition)
            if status is not None:
                self._finish_locked(transition, status, reason, now)
                changed = True
        return changed

    @staticmethod
    def _device_reason(device: Mapping[str, Any]) -> str:
        safety = device.get("safety")
        shm = device.get("shm")
        return str(
            device.get("last_error")
            or (safety.get("reason") if isinstance(safety, Mapping) else None)
            or (shm.get("error") if isinstance(shm, Mapping) else None)
            or f"Device observed as {device.get('status', 'unknown')}"
        )

    def _choose_interval_locked(self, raw: Mapping[str, Any]) -> float:
        if any(
            transition.status in {"pending", "acknowledged"}
            for transition in self._transitions.values()
        ) or self._raw_has_status(raw, _TRANSITION_STATUSES):
            return TRANSITION_INTERVAL_S
        if self._raw_has_status(raw, _TERMINAL_FAILURES):
            return min(
                ACTIVE_INTERVAL_S,
                RECOVERY_INTERVAL_S * (2 ** min(2, self._stable_samples // 4)),
            )
        if self._raw_is_active(raw):
            return 0.5 if self._stable_samples < 3 else ACTIVE_INTERVAL_S
        return 1.0 if self._stable_samples < 2 else IDLE_INTERVAL_S

    @staticmethod
    def _raw_has_status(raw: Mapping[str, Any], statuses: frozenset[str]) -> bool:
        devices = raw.get("devices")
        if isinstance(devices, list) and any(
            isinstance(device, Mapping)
            and str(device.get("status") or "").lower() in statuses
            for device in devices
        ):
            return True
        workflow = raw.get("workflow")
        if isinstance(workflow, Mapping) and any(
            isinstance(workflow.get(name), Mapping)
            and str(workflow[name].get("status") or "").lower() in statuses
            for name in ("runtime", "inference")
        ):
            return True
        recording = raw.get("recording")
        if isinstance(recording, Mapping) and str(
            recording.get("status") or ""
        ).lower() in statuses:
            return True
        grippers = raw.get("grippers")
        if isinstance(grippers, Mapping):
            values = [grippers]
            sides = grippers.get("sides")
            if isinstance(sides, Mapping):
                values.extend(item for item in sides.values() if isinstance(item, Mapping))
            if any(str(item.get("status") or "").lower() in statuses for item in values):
                return True
        return False

    @staticmethod
    def _raw_is_active(raw: Mapping[str, Any]) -> bool:
        devices = raw.get("devices")
        if isinstance(devices, list) and any(
            isinstance(device, Mapping)
            and (
                device.get("running") is True
                or device.get("desired_state") == "running"
                or str(device.get("status") or "").lower()
                in {"running", "degraded", "orphaned", "external", "stop_timeout"}
            )
            for device in devices
        ):
            return True
        workflow = raw.get("workflow")
        if isinstance(workflow, Mapping):
            runtime = workflow.get("runtime")
            inference = workflow.get("inference")
            if isinstance(runtime, Mapping) and runtime.get("active") is True:
                return True
            if isinstance(inference, Mapping) and inference.get("running") is True:
                return True
        recording = raw.get("recording")
        if isinstance(recording, Mapping) and recording.get("active") is True:
            return True
        grippers = raw.get("grippers")
        return isinstance(grippers, Mapping) and grippers.get("active") is True

    @staticmethod
    def _semantic_view(raw: Mapping[str, Any]) -> Dict[str, Any]:
        devices = []
        for device in raw.get("devices") or []:
            if not isinstance(device, Mapping):
                continue
            shm = device.get("shm") if isinstance(device.get("shm"), Mapping) else {}
            safety = (
                device.get("safety")
                if isinstance(device.get("safety"), Mapping)
                else {}
            )
            devices.append(
                {
                    "key": device.get("key"),
                    "status": device.get("status"),
                    "running": device.get("running"),
                    "ready": device.get("ready"),
                    "desired_state": device.get("desired_state"),
                    "pid": device.get("pid"),
                    "exit_code": device.get("exit_code"),
                    "restart_count": device.get("restart_count"),
                    "last_error": device.get("last_error"),
                    "shm": {
                        "connected": shm.get("connected"),
                        "fresh": shm.get("fresh"),
                        "error": shm.get("error"),
                    },
                    "safety": {
                        "operational": safety.get("operational"),
                        "fault": safety.get("fault"),
                        "estop": safety.get("estop"),
                        "fail_closed": safety.get("fail_closed"),
                        "reason": safety.get("reason"),
                    },
                }
            )
        return {
            "controller_revision": raw.get("revision"),
            "selected_config": raw.get("selected_config"),
            "active_config": raw.get("active_config"),
            "devices": devices,
            "workflow": raw.get("workflow"),
            "recording": raw.get("recording"),
            "grippers": raw.get("grippers"),
        }

    @classmethod
    def _digest(cls, raw: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            cls._semantic_view(raw),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _scope_transition_locked(
        self, scope: str, now: float
    ) -> Optional[Dict[str, Any]]:
        transition = self._transitions.get(scope)
        return transition.public(now) if transition is not None else None

    @staticmethod
    def _explicit_estop(safety: Mapping[str, Any]) -> Optional[bool]:
        raw = safety.get("estop")
        if isinstance(raw, Mapping):
            raw = raw.get("active")
        if raw is True:
            return True
        if safety.get("stale") is True or safety.get("known") is False:
            return None
        return raw if isinstance(raw, bool) else None

    @staticmethod
    def _expected_home_transition(
        snapshot: Mapping[str, Any], observed_at: str
    ) -> Optional[Dict[str, Any]]:
        workflow = snapshot.get("workflow")
        if not isinstance(workflow, Mapping):
            return None

        pending: Any = None
        inference = workflow.get("inference")
        reset = inference.get("reset") if isinstance(inference, Mapping) else None
        if isinstance(reset, Mapping):
            candidate = reset.get("pending_command")
            if isinstance(candidate, Mapping) and candidate.get("action") == "reset":
                pending = candidate
        if pending is None:
            recording = workflow.get("recording")
            if isinstance(recording, Mapping):
                candidate = recording.get("pending_command")
                if isinstance(candidate, Mapping) and candidate.get("action") in {
                    "episode_accept",
                    "episode_discard",
                }:
                    pending = candidate
        if not isinstance(pending, Mapping):
            return None

        created_at = pending.get("created_at")
        if not isinstance(created_at, str):
            return None
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        if created.tzinfo is None or observed.tzinfo is None:
            return None
        age_s = (observed - created).total_seconds()
        if age_s < 0 or age_s > EXPECTED_HOME_TRANSITION_MAX_S:
            return None
        return {
            "active": True,
            "kind": "home_reset",
            "action": str(pending.get("action")),
            "command_id": pending.get("command_id"),
            "started_at": created_at,
            "age_ms": round(age_s * 1000),
            "max_age_ms": round(EXPECTED_HOME_TRANSITION_MAX_S * 1000),
        }

    def _annotate_device_locked(
        self,
        device: Dict[str, Any],
        observed_at: str,
        now: float,
        expected_home: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = str(device.get("key") or "")
        scope = f"device:{key}"
        status = str(device.get("status") or "unknown").lower()
        shm = device.get("shm") if isinstance(device.get("shm"), Mapping) else {}
        raw_safety = (
            dict(device.get("safety"))
            if isinstance(device.get("safety"), Mapping)
            else {}
        )
        motion = str(device.get("risk") or "").lower() == "motion"
        estop = self._explicit_estop(raw_safety)
        initializing = bool(
            raw_safety.get("initializing") is True or status == "starting"
        )
        health_lost = bool(
            motion
            and device.get("desired_state") == "running"
            and not initializing
            and (
                device.get("ready") is not True
                or status in {"failed", "degraded", "stop_timeout"}
                or (shm.get("connected") is True and shm.get("fresh") is not True)
            )
        )
        raw_reason = str(
            raw_safety.get("reason")
            or device.get("last_error")
            or (shm.get("error") if isinstance(shm, Mapping) else None)
            or ""
        )
        transient_heartbeat_loss = bool(
            health_lost
            and status in {"running", "degraded"}
            and shm.get("connected") is True
            and (
                shm.get("fresh") is not True
                or (
                    "health heartbeat" in raw_reason.casefold()
                    and any(
                        marker in raw_reason.casefold()
                        for marker in ("stale", "fresh", "unavailable", "waiting")
                    )
                )
            )
        )
        expected_home_active = bool(
            expected_home
            and motion
            and device.get("running") is True
            and device.get("desired_state") == "running"
            and transient_heartbeat_loss
            and estop is not True
            and raw_safety.get("fault") is not True
            and raw_safety.get("operational") is not False
            and raw_safety.get("reduced") is not True
        )
        fail_closed = bool(
            raw_safety.get("fail_closed") is True
            or estop is True
            or raw_safety.get("fault") is True
            or raw_safety.get("operational") is False
            or health_lost
        )
        raw_fail_closed = fail_closed
        if expected_home_active:
            fail_closed = False
        reason = str(
            raw_reason
            or (
                "Motion-device health was lost; emergency stop, fault, or disconnect cannot be excluded"
                if health_lost
                else ""
            )
        )
        healthy = bool(
            status == "running"
            and device.get("ready") is True
            and not fail_closed
        )
        if healthy:
            self._last_healthy[scope] = observed_at
        safety = {
            **raw_safety,
            "initializing": initializing,
            "estop": {
                "active": estop,
                "known": estop is not None,
                "source": raw_safety.get("source") or "device_health_or_inference",
            },
            "fail_closed": fail_closed,
            "reason": reason or None,
            "last_seen": raw_safety.get("last_seen") or observed_at,
        }
        if expected_home_active:
            device["expected_transition"] = dict(expected_home or {})
            safety.update(
                {
                    "transient_fail_closed_suppressed": True,
                    "raw_fail_closed": raw_fail_closed,
                    "raw_reason": reason or None,
                }
            )
        device["observed"] = {
            "state": status,
            "running": device.get("running") is True,
            "ready": device.get("ready") is True,
            "source": "owned_process+device_shm",
            "reason": reason or None,
            "last_seen": observed_at,
            "last_healthy_at": self._last_healthy.get(scope),
        }
        device["desired"] = {"state": str(device.get("desired_state") or "stopped")}
        device["transition"] = self._scope_transition_locked(scope, now)
        device["reason"] = reason or None
        device["last_seen"] = observed_at
        device["estop"] = safety["estop"]
        device["safety"] = safety
        return device

    def _annotate_workflow_locked(
        self, snapshot: Dict[str, Any], observed_at: str, now: float
    ) -> None:
        workflow = snapshot.get("workflow")
        if not isinstance(workflow, dict):
            return
        for name in ("runtime", "inference"):
            stage = workflow.get(name)
            if not isinstance(stage, dict):
                continue
            active = (
                stage.get("active") is True
                if name == "runtime"
                else stage.get("running") is True
            )
            desired = str(stage.get("desired_state") or ("running" if active else "stopped"))
            reason = str(stage.get("error") or "") or None
            scope = f"workflow:{name}"
            if str(stage.get("status")) == "running" and not reason:
                self._last_healthy[scope] = observed_at
            stage["observed"] = {
                "state": str(stage.get("status") or "unknown"),
                "active": active,
                "source": "owned_processes+readiness_channels",
                "reason": reason,
                "last_seen": observed_at,
                "last_healthy_at": self._last_healthy.get(scope),
            }
            stage["desired"] = {"state": desired}
            stage["transition"] = self._scope_transition_locked(scope, now)
            stage["reason"] = reason
            stage["last_seen"] = observed_at

    def _annotate_grippers_locked(
        self, snapshot: Dict[str, Any], observed_at: str, now: float
    ) -> None:
        grippers = snapshot.get("grippers")
        if not isinstance(grippers, dict):
            return
        sides = grippers.get("sides")
        if not isinstance(sides, dict):
            return
        for side, item in sides.items():
            if not isinstance(item, dict):
                continue
            scope = f"gripper:{side}"
            status = str(item.get("status") or "unavailable")
            reason = str(item.get("last_error") or "") or None
            if status == "idle" and not reason:
                self._last_healthy[scope] = observed_at
            item["observed"] = {
                "state": status,
                "source": "gripper_worker+serial_probe",
                "reason": reason,
                "last_seen": observed_at,
                "last_healthy_at": self._last_healthy.get(scope),
            }
            item["desired"] = {
                "state": "idle" if item.get("busy") is not True else status
            }
            item["transition"] = self._scope_transition_locked(scope, now)
            item["reason"] = reason
            item["last_seen"] = observed_at

    def _annotate_recording_locked(
        self, snapshot: Dict[str, Any], observed_at: str, now: float
    ) -> None:
        recording = snapshot.get("recording")
        if not isinstance(recording, dict):
            return
        status = str(recording.get("status") or "unavailable")
        reason = str(recording.get("error") or "") or None
        recording["observed"] = {
            "state": status,
            "source": "recorder_control+owner_process",
            "reason": reason,
            "last_seen": observed_at,
        }
        recording["desired"] = {
            "state": "active" if recording.get("active") is True else "stopped"
        }
        recording["transition"] = self._scope_transition_locked("recording", now)
        recording["reason"] = reason
        recording["last_seen"] = observed_at

    def _annotate_locked(
        self, snapshot: Dict[str, Any], observed_at: str, now: float
    ) -> Dict[str, Any]:
        expected_home = self._expected_home_transition(snapshot, observed_at)
        devices = snapshot.get("devices")
        if isinstance(devices, list):
            snapshot["devices"] = [
                self._annotate_device_locked(
                    device,
                    observed_at,
                    now,
                    expected_home,
                )
                if isinstance(device, dict)
                else device
                for device in devices
            ]
        self._annotate_workflow_locked(snapshot, observed_at, now)
        self._annotate_recording_locked(snapshot, observed_at, now)
        self._annotate_grippers_locked(snapshot, observed_at, now)
        motion_device_safety = [
            device.get("safety")
            for device in snapshot.get("devices") or []
            if isinstance(device, Mapping)
            and str(device.get("risk") or "").lower() == "motion"
            and isinstance(device.get("safety"), Mapping)
        ]
        estop_values = [
            item["estop"].get("active")
            for item in motion_device_safety
            if isinstance(item.get("estop"), Mapping)
        ]
        estop_known = bool(estop_values) and all(
            isinstance(value, bool) for value in estop_values
        )
        estop_active: Optional[bool]
        if any(value is True for value in estop_values):
            estop_active = True
            estop_known = True
        elif estop_known:
            estop_active = False
        else:
            estop_active = None
        fail_closed = estop_active is True or any(
            (
                item.get("fail_closed") is True
                and item.get("initializing") is not True
            )
            or item.get("fault") is True
            or item.get("operational") is False
            for item in motion_device_safety
        )
        reasons = [
            str(item.get("reason"))
            for item in motion_device_safety
            if item.get("reason")
        ]
        snapshot["safety"] = {
            "estop": {
                "active": estop_active,
                "known": estop_known,
                "source": "device_health_aggregate",
            },
            "fail_closed": fail_closed,
            "reason": reasons[0] if reasons else None,
            "last_seen": observed_at,
        }
        snapshot["state_sync"] = {
            "epoch": self.epoch,
            "sequence": self._sequence,
            "observed_revision": self._observed_revision,
            "event_revision": self._event_revision,
            "observed_at": observed_at,
            "authoritative": True,
            "stale": False,
            "stale_after_ms": round(
                min(
                    MAX_STALE_AFTER_S,
                    max(MIN_STALE_AFTER_S, self._expected_interval_s * 3.0),
                )
                * 1000
            ),
            "expected_interval_ms": round(self._expected_interval_s * 1000),
            "source": "process+shm+workflow+recorder+gripper",
            "last_error": None,
            "transitions": {
                scope: transition.public(now)
                for scope, transition in self._transitions.items()
            },
        }
        return snapshot

    def _decorate_current_locked(
        self, snapshot: Dict[str, Any], now: float
    ) -> Dict[str, Any]:
        sync = snapshot.setdefault("state_sync", {})
        age_s = (
            float("inf")
            if self._last_sample_monotonic is None
            else max(0.0, now - self._last_sample_monotonic)
        )
        stale_after_s = min(
            MAX_STALE_AFTER_S,
            max(MIN_STALE_AFTER_S, self._expected_interval_s * 3.0),
        )
        sync.update(
            {
                "epoch": self.epoch,
                "sequence": self._sequence,
                "observed_revision": self._observed_revision,
                "event_revision": self._event_revision,
                "age_ms": None if age_s == float("inf") else round(age_s * 1000),
                "stale_after_ms": round(stale_after_s * 1000),
                "expected_interval_ms": round(self._expected_interval_s * 1000),
                "stale": age_s > stale_after_s,
                "authoritative": age_s <= stale_after_s and self._last_error is None,
                "last_error": self._last_error,
                "transitions": {
                    scope: transition.public(now)
                    for scope, transition in self._transitions.items()
                },
            }
        )
        by_scope = sync["transitions"]
        for device in snapshot.get("devices") or []:
            if isinstance(device, dict):
                device["transition"] = by_scope.get(f"device:{device.get('key')}")
                observed = device.get("observed")
                if isinstance(observed, dict):
                    observed["stale"] = sync["stale"]
        workflow = snapshot.get("workflow")
        if isinstance(workflow, dict):
            for name in ("runtime", "inference"):
                stage = workflow.get(name)
                if isinstance(stage, dict):
                    stage["transition"] = by_scope.get(f"workflow:{name}")
                    observed = stage.get("observed")
                    if isinstance(observed, dict):
                        observed["stale"] = sync["stale"]
        grippers = snapshot.get("grippers")
        sides = grippers.get("sides") if isinstance(grippers, dict) else None
        if isinstance(sides, dict):
            for side, item in sides.items():
                if isinstance(item, dict):
                    item["transition"] = by_scope.get(f"gripper:{side}")
                    observed = item.get("observed")
                    if isinstance(observed, dict):
                        observed["stale"] = sync["stale"]
        recording = snapshot.get("recording")
        if isinstance(recording, dict):
            recording["transition"] = by_scope.get("recording")
            observed = recording.get("observed")
            if isinstance(observed, dict):
                observed["stale"] = sync["stale"]
        return snapshot


__all__ = [
    "ACTIVE_INTERVAL_S",
    "IDLE_INTERVAL_S",
    "RECOVERY_INTERVAL_S",
    "StateSyncBroker",
    "StateTransitionConflict",
    "TRANSITION_INTERVAL_S",
]
