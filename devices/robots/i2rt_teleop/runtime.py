"""Worker-local control loop and watchdog for I2RT teleoperation."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import ControlConfig


LOGGER = logging.getLogger(__name__)

SAFE_HOLD = "SAFE_HOLD"
ALIGNING = "ALIGNING"
ACTIVE = "ACTIVE"
FAULT = "FAULT"
STOPPED = "STOPPED"
CONTROL_PHASES = {ALIGNING, ACTIVE}
ARM_DOF = 6

LEADER_DEFAULT_KP = np.array([250.0, 400.0, 400.0, 100.0, 25.0, 25.0])
LEADER_DEFAULT_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])


def rpc_integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    """Decode an integer scalar accepted directly or after portal transport."""
    if isinstance(value, np.ndarray):
        if value.shape != () or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{field} must be an integer")
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _vector(value: Any, size: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"{field} must contain exactly {size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must contain only finite values")
    return result.copy()


@dataclass(frozen=True)
class ControlFrame:
    """Validated command accepted by a worker control session."""

    session_id: str
    sequence: int
    issued_at_ns: int
    phase: str
    joint_position: np.ndarray

    @classmethod
    def from_mapping(cls, value: Any, command_size: int) -> ControlFrame:
        if not isinstance(value, Mapping):
            raise ValueError("control frame must be a mapping")
        session_id = value.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        phase = value.get("phase")
        if phase not in CONTROL_PHASES:
            raise ValueError(f"phase must be one of: {sorted(CONTROL_PHASES)}")
        return cls(
            session_id=session_id,
            sequence=rpc_integer(value.get("sequence"), "sequence", minimum=0),
            issued_at_ns=rpc_integer(
                value.get("issued_at_ns"), "issued_at_ns", minimum=1
            ),
            phase=phase,
            joint_position=_vector(
                value.get("joint_position"), command_size, "joint_position"
            ),
        )


class WorkerControlRuntime:
    """Own all endpoint I/O and enforce the local command/feedback contract."""

    def __init__(
        self,
        endpoint: Any,
        *,
        role: str,
        side: str,
        control: ControlConfig,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if role not in {"leader", "follower"}:
            raise ValueError("role must be 'leader' or 'follower'")
        self.endpoint = endpoint
        self.role = role
        self.side = side
        self.control = control
        self.clock_ns = clock_ns
        self.sleep = sleep
        self.command_size = 6 if role == "leader" else 7

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._safe_state_applied_event = threading.Event()
        # The control loop owns all endpoint calls.  The watchdog intentionally
        # lives in a different thread so a blocked hardware call cannot prevent
        # status/RPC locks from being checked.
        self._io_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        # Keep the old name as an internal compatibility alias for callers
        # which inspected the control thread while developing this runtime.
        self._thread: threading.Thread | None = None
        self._io_inflight = False
        self._io_started_ns: int | None = None
        self._last_io_progress_ns: int | None = None
        self._control_epoch = 0
        self._phase = SAFE_HOLD
        self._session_id: str | None = None
        self._phase_started_ns = self.clock_ns()
        self._last_received_sequence = -1
        self._last_applied_sequence = -1
        self._last_command_applied_ns: int | None = None
        self._last_feedback_ns: int | None = None
        self._sample: np.ndarray | None = None
        self._buttons = np.zeros(1, dtype=np.float64)
        self._pending_frame: ControlFrame | None = None
        self._safe_action_pending = True
        self._leader_feedback_update_pending = False
        self._following_error_count = 0
        self._fault_reason: str | None = None
        self._last_error: str | None = None

    def start(
        self,
        timeout_s: float,
        *,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        if self._io_thread is not None or self._watchdog_thread is not None:
            raise RuntimeError("worker runtime already started")
        if cancel_event is not None and cancel_event.is_set():
            return False
        self._stop_event.clear()
        self._io_thread = threading.Thread(
            target=self._io_loop,
            name=f"i2rt-{self.side}-{self.role}-control",
            daemon=True,
        )
        self._thread = self._io_thread
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"i2rt-{self.side}-{self.role}-watchdog",
            daemon=True,
        )
        self._io_thread.start()
        self._watchdog_thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._ready_event.is_set():
            if cancel_event is not None and cancel_event.is_set():
                self.request_stop()
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.close()
                raise TimeoutError(
                    f"Timed out initializing {self.side}/{self.role} safe state"
                )
            self._ready_event.wait(timeout=min(remaining, 0.05))
        if cancel_event is not None and cancel_event.is_set():
            self.request_stop()
            return False
        with self._lock:
            fault_reason = self._fault_reason if self._phase == FAULT else None
        if fault_reason is not None:
            self.close()
            raise RuntimeError(
                f"Could not initialize {self.side}/{self.role}: "
                f"{fault_reason}"
            )
        return True

    def request_stop(self) -> None:
        """Wake startup and ask runtime threads to stop without blocking."""

        self._stop_event.set()
        self._ready_event.set()

    def close(self) -> bool:
        io_thread = self._io_thread or self._thread
        watchdog_thread = self._watchdog_thread
        if io_thread is None and watchdog_thread is None:
            return True
        with self._lock:
            if self._phase == STOPPED:
                return True
            self._phase = SAFE_HOLD
            self._control_epoch += 1
            self._session_id = None
            self._pending_frame = None
            self._leader_feedback_update_pending = False
            self._safe_action_pending = True
            self._safe_state_applied_event.clear()
        self._safe_state_applied_event.wait(
            timeout=max(0.1, self.control.feedback_timeout_s * 2.0)
        )
        self._stop_event.set()
        join_timeout = max(1.0, self.control.feedback_timeout_s * 2.0)
        threads_alive = False
        for thread in (io_thread, watchdog_thread):
            if thread is None or thread is threading.current_thread():
                continue
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                threads_alive = True
        if threads_alive:
            LOGGER.error(
                "%s/%s control loop did not stop; check hardware I/O timeouts",
                self.side,
                self.role,
            )
            return False
        with self._lock:
            self._phase = STOPPED
        return True

    def ping(self) -> dict[str, str]:
        with self._lock:
            return {"side": self.side, "role": self.role, "phase": self._phase}

    def get_sample(self) -> dict[str, Any]:
        with self._lock:
            if self._sample is None or self._last_feedback_ns is None:
                raise RuntimeError("endpoint has not produced a valid sample")
            return {
                "state": self._sample.copy(),
                "buttons": self._buttons.copy(),
                "sampled_at_ns": self._last_feedback_ns,
                "phase": self._phase,
                "fault_reason": self._fault_reason,
                "last_applied_sequence": self._last_applied_sequence,
            }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def begin_control_session(self, session_id: str) -> dict[str, Any]:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        now_ns = self.clock_ns()
        with self._lock:
            if self._phase == FAULT:
                raise RuntimeError(
                    f"worker is faulted: {self._fault_reason}; reset it first"
                )
            if self._session_id is not None and self._session_id != session_id:
                raise RuntimeError("another control session is already active")
            self._session_id = session_id
            self._phase = ALIGNING
            self._phase_started_ns = now_ns
            self._control_epoch += 1
            self._last_received_sequence = -1
            self._last_applied_sequence = -1
            self._last_command_applied_ns = None
            self._pending_frame = None
            self._following_error_count = 0
            self._fault_reason = None
            self._leader_feedback_update_pending = False
            return self._status_locked()

    def set_phase(self, session_id: str, phase: str) -> dict[str, Any]:
        if phase != ACTIVE:
            raise ValueError("a control session can only transition to ACTIVE")
        now_ns = self.clock_ns()
        with self._lock:
            self._require_session_locked(session_id)
            if self._phase != ALIGNING:
                raise RuntimeError(f"cannot activate worker from {self._phase}")
            if self._last_applied_sequence < 0:
                raise RuntimeError(
                    "cannot activate before an alignment frame is applied"
                )
            self._phase = ACTIVE
            self._phase_started_ns = now_ns
            self._control_epoch += 1
            self._last_command_applied_ns = now_ns
            self._pending_frame = None
            # A leader must always leave the alignment hold here. Depending on
            # configuration this either enables bilateral feedback or restores
            # zero-gain free motion.
            self._leader_feedback_update_pending = self.role == "leader"
            return self._status_locked()

    def submit_control_frame(self, value: Any) -> dict[str, Any]:
        frame = ControlFrame.from_mapping(value, self.command_size)
        now_ns = self.clock_ns()
        age_ns = now_ns - frame.issued_at_ns
        max_age_ns = int(self.control.command_frame_max_age_s * 1e9)
        if age_ns < 0:
            raise ValueError("control frame timestamp is in the future")
        if age_ns > max_age_ns:
            raise ValueError("control frame is stale")
        with self._lock:
            self._require_session_locked(frame.session_id)
            if self._phase != frame.phase:
                raise RuntimeError(
                    f"frame phase {frame.phase} does not match worker phase "
                    f"{self._phase}"
                )
            if frame.sequence <= self._last_received_sequence:
                raise ValueError("control frame sequence must increase")
            self._last_received_sequence = frame.sequence
            self._pending_frame = frame
            return self._status_locked()

    def end_control_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._require_session_locked(session_id)
            self._phase = SAFE_HOLD
            self._control_epoch += 1
            self._session_id = None
            self._pending_frame = None
            self._last_command_applied_ns = None
            self._following_error_count = 0
            self._safe_action_pending = True
            self._safe_state_applied_event.clear()
            self._leader_feedback_update_pending = False
            return self._status_locked()

    def reset_fault(self) -> dict[str, Any]:
        with self._lock:
            if self._phase != FAULT:
                return self._status_locked()
            self._phase = SAFE_HOLD
            self._control_epoch += 1
            self._session_id = None
            self._pending_frame = None
            self._fault_reason = None
            self._last_error = None
            self._last_command_applied_ns = None
            self._following_error_count = 0
            self._safe_action_pending = True
            self._safe_state_applied_event.clear()
            self._leader_feedback_update_pending = False
            return self._status_locked()

    def run_once(self) -> None:
        """Execute one hardware cycle; exposed for deterministic tests."""
        self._run_io_cycle()
        self._run_watchdog_check()
        # A synchronous test cycle should observe the same safe-state action
        # immediately as the threaded runtime does on its next I/O iteration.
        self._apply_pending_safe_action()

    def _io_loop(self) -> None:
        while not self._stop_event.is_set():
            started_ns = self.clock_ns()
            self._run_io_cycle()
            elapsed_s = (self.clock_ns() - started_ns) / 1e9
            remaining = self.control.watchdog_check_interval_s - elapsed_s
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            started_ns = self.clock_ns()
            self._run_watchdog_check()
            elapsed_s = (self.clock_ns() - started_ns) / 1e9
            remaining = self.control.watchdog_check_interval_s - elapsed_s
            if remaining > 0:
                self._stop_event.wait(remaining)

    # Kept as a small alias for code which used the previous private target.
    def _run(self) -> None:
        self._io_loop()

    def _run_io_cycle(self) -> None:
        started_ns = self.clock_ns()
        with self._lock:
            self._io_inflight = True
            self._io_started_ns = started_ns
        try:
            try:
                state, buttons = self._read_endpoint()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"feedback read failed: {exc}"
                return

            with self._lock:
                self._sample = state
                self._buttons = buttons
                self._last_feedback_ns = self.clock_ns()
                self._last_error = None
                phase = self._phase
                frame = self._pending_frame
                self._pending_frame = None
                epoch = self._control_epoch
                update_leader_feedback = (
                    self._leader_feedback_update_pending and phase == ACTIVE
                )
                if update_leader_feedback:
                    self._leader_feedback_update_pending = False

            if update_leader_feedback:
                try:
                    self.endpoint.set_feedback_enabled(
                        self.control.leader_feedback,
                        LEADER_DEFAULT_KP * self.control.bilateral_kp,
                        LEADER_DEFAULT_KD * self.control.bilateral_kd,
                    )
                except Exception as exc:
                    self._trip_fault(f"could not configure leader feedback: {exc}")

            if frame is not None and phase in CONTROL_PHASES:
                with self._lock:
                    should_apply = (
                        self._session_id == frame.session_id
                        and self._phase == frame.phase
                        and self._control_epoch == epoch
                    )
                if should_apply:
                    try:
                        if self.role == "leader" and frame.phase == ALIGNING:
                            self.endpoint.enter_position_hold(frame.joint_position)
                        else:
                            self.endpoint.command_joint_pos(frame.joint_position)
                    except Exception as exc:
                        self._trip_fault(f"command failed: {exc}")
                    else:
                        applied_ns = self.clock_ns()
                        following_fault: str | None = None
                        with self._lock:
                            # A concurrent watchdog fault clears the session;
                            # in that case the command may have completed on the
                            # device, but it must not be reported as current.
                            if (
                                self._session_id == frame.session_id
                                and self._phase == frame.phase
                            ):
                                self._last_applied_sequence = frame.sequence
                                self._last_command_applied_ns = applied_ns
                                if self.role == "follower":
                                    threshold = (
                                        self.control.following_error_fault_threshold
                                    )
                                    joint_errors = np.abs(
                                        state[:ARM_DOF]
                                        - frame.joint_position[:ARM_DOF]
                                    )
                                    worst_joint = int(np.argmax(joint_errors))
                                    error = float(joint_errors[worst_joint])
                                    if error > threshold:
                                        self._following_error_count += 1
                                    else:
                                        self._following_error_count = 0
                                    if (
                                        self._following_error_count
                                        >= self.control.following_error_fault_samples
                                    ):
                                        following_fault = (
                                            f"{self.side} follower following error "
                                            f"{error:.3f} rad at joint {worst_joint + 1} "
                                            f"exceeds {threshold:.3f} rad during "
                                            f"{frame.phase}"
                                        )
                        if following_fault is not None:
                            self._trip_fault(following_fault)
        finally:
            self._apply_pending_safe_action()
            with self._lock:
                self._io_inflight = False
                self._last_io_progress_ns = self.clock_ns()
            self._ready_event.set()

    def _run_watchdog_check(self) -> None:
        now_ns = self.clock_ns()
        feedback_timeout_ns = int(self.control.feedback_timeout_s * 1e9)
        lease_timeout_ns = int(self.control.command_lease_timeout_s * 1e9)
        reason: str | None = None
        with self._lock:
            if self._phase == STOPPED:
                return
            io_stalled = (
                self._io_inflight
                and self._io_started_ns is not None
                and now_ns - self._io_started_ns > feedback_timeout_ns
            )
            if io_stalled and self._phase in CONTROL_PHASES:
                reason = (
                    "hardware I/O stalled for "
                    f"{(now_ns - self._io_started_ns) / 1e9:.3f}s"
                )
            elif self._phase != STOPPED:
                feedback_reference = (
                    self._last_feedback_ns or self._phase_started_ns
                )
                if now_ns - feedback_reference > feedback_timeout_ns:
                    reason = self._last_error or "feedback timeout"
            if reason is None and self._phase in CONTROL_PHASES:
                lease_reference = (
                    self._last_command_applied_ns or self._phase_started_ns
                )
                if now_ns - lease_reference > lease_timeout_ns:
                    reason = "command lease expired"
        if reason is not None:
            self._trip_fault(reason)

    def _read_endpoint(self) -> tuple[np.ndarray, np.ndarray]:
        result = self.endpoint.get_info()
        if self.role == "leader":
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ValueError("leader endpoint must return (state, buttons)")
            state = _vector(result[0], 7, "leader state")
            buttons = np.asarray(result[1], dtype=np.float64).reshape(-1)
            if buttons.size == 0 or not np.all(np.isfinite(buttons)):
                raise ValueError("leader buttons must contain finite values")
            return state, buttons.copy()
        return _vector(result, 7, "follower state"), np.zeros(1, dtype=np.float64)

    def _apply_pending_safe_action(self) -> None:
        with self._lock:
            if not self._safe_action_pending or self._sample is None:
                return
            self._safe_action_pending = False
            state = self._sample.copy()
        try:
            self.endpoint.enter_safe_state(state)
        except Exception as exc:
            reason = f"safe-state command failed: {exc}"
            with self._lock:
                if self._phase != FAULT:
                    self._trip_fault_locked(reason)
                else:
                    self._last_error = reason
                # Do not repeatedly retry a safe-state operation which has
                # already failed; the process-level supervisor will restart it.
                self._safe_action_pending = False
        finally:
            self._safe_state_applied_event.set()
            self._ready_event.set()

    def _trip_fault(self, reason: str) -> None:
        with self._lock:
            self._trip_fault_locked(reason)

    def _trip_fault_locked(self, reason: str) -> None:
        if self._phase == FAULT:
            return
        LOGGER.error("%s/%s watchdog fault: %s", self.side, self.role, reason)
        self._phase = FAULT
        self._fault_reason = reason
        self._last_error = reason
        self._control_epoch += 1
        self._session_id = None
        self._pending_frame = None
        self._leader_feedback_update_pending = False
        self._safe_action_pending = True
        self._safe_state_applied_event.clear()
        self._ready_event.set()

    def _require_session_locked(self, session_id: str) -> None:
        if self._phase == FAULT:
            raise RuntimeError(f"worker is faulted: {self._fault_reason}")
        if self._session_id != session_id:
            raise RuntimeError("control session does not match")

    def _status_locked(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "role": self.role,
            "phase": self._phase,
            "session_id": self._session_id,
            "last_received_sequence": self._last_received_sequence,
            "last_applied_sequence": self._last_applied_sequence,
            "last_feedback_ns": self._last_feedback_ns,
            "last_command_applied_ns": self._last_command_applied_ns,
            "io_inflight": self._io_inflight,
            "last_io_progress_ns": self._last_io_progress_ns,
            "fault_reason": self._fault_reason,
            "last_error": self._last_error,
        }
