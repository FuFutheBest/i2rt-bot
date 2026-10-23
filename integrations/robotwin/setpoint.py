"""Non-blocking target ownership for the RoboTwin physics loop."""

from __future__ import annotations

import importlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np

from .contracts import ControlSetpoint, validate_action

EefResolver = Callable[[np.ndarray, np.ndarray], np.ndarray]


class LatestSetpoint:
    """Accept monotonically newer control frames and hold the most recent target.

    Accepting a command never waits for the simulated robot to reach it.  The
    physics loop applies the held target as a SAPIEN drive target on each tick.
    """

    def __init__(self, action_schema: str) -> None:
        self.action_schema = action_schema
        self._lock = threading.Lock()
        self._latest: Optional[ControlSetpoint] = None
        self._sequence = 0

    @property
    def latest(self) -> Optional[ControlSetpoint]:
        with self._lock:
            return self._latest

    def accept(self, action: Any, control_write_timestamp_ns: int) -> Optional[ControlSetpoint]:
        """Return a new setpoint, or ``None`` for an already-consumed frame."""
        timestamp_ns = int(control_write_timestamp_ns)
        if timestamp_ns <= 0:
            raise ValueError("RoboTwin control frame timestamp must be positive")
        vector = validate_action(action, self.action_schema)
        with self._lock:
            if self._latest is not None and timestamp_ns <= self._latest.control_write_timestamp_ns:
                return None
            self._sequence += 1
            self._latest = ControlSetpoint(
                sequence=self._sequence,
                control_write_timestamp_ns=timestamp_ns,
                action_schema=self.action_schema,
                action=vector,
            )
            return self._latest


def load_eef_resolver(spec: str) -> EefResolver:
    """Load a pure-Numpy eef16-to-joint14 resolver from ``module:callable``."""
    module_name, separator, attribute_path = str(spec).partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("eef_resolver must use the form 'package.module:callable'")
    module = importlib.import_module(module_name)
    target: Any = module
    for attribute in attribute_path.split("."):
        target = getattr(target, attribute)
    if not callable(target):
        raise TypeError(f"eef_resolver {spec!r} is not callable")
    return target


class AsyncEefResolver:
    """Resolve eef16 goals outside the fixed-rate SAPIEN physics thread.

    The resolver is intentionally given only NumPy arrays, not SAPIEN objects.
    This keeps planner/IK work out of the physics loop and avoids concurrent
    access to SAPIEN scene state.
    """

    def __init__(self, resolver: EefResolver) -> None:
        self._resolver = resolver
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robotwin-eef")
        self._future: Optional[Future[tuple[int, np.ndarray]]] = None
        self._pending: Optional[tuple[int, np.ndarray, np.ndarray]] = None
        self._latest_submitted_sequence = 0
        self._lock = threading.Lock()

    def submit(self, setpoint: ControlSetpoint, measured_joint14: Any) -> None:
        if setpoint.action_schema != "eef16":
            raise ValueError("AsyncEefResolver only accepts eef16 setpoints")
        current = validate_action(measured_joint14, "joint14")
        target = setpoint.action.copy()
        with self._lock:
            self._latest_submitted_sequence = setpoint.sequence
            request = (setpoint.sequence, target, current)
            if self._future is None:
                self._future = self._executor.submit(self._resolve, *request)
            else:
                # A non-interruptible resolver may already be running. Keep
                # exactly one replacement request so old setpoints cannot
                # build an arbitrarily long queue behind it.
                self._pending = request

    def poll(self) -> Optional[tuple[int, np.ndarray]]:
        """Return the newest completed joint target, without blocking."""
        with self._lock:
            future = self._future
        if future is None or not future.done():
            return None

        try:
            sequence, target = future.result()
        except Exception:
            self._advance_after_completed_future(future)
            raise

        with self._lock:
            if self._future is not future:
                return None
            self._future = None
            pending = self._pending
            self._pending = None
            latest_sequence = self._latest_submitted_sequence
            if pending is not None:
                self._future = self._executor.submit(self._resolve, *pending)
                return None
        if sequence != latest_sequence:
            return None
        return sequence, target

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _advance_after_completed_future(self, completed: Future[tuple[int, np.ndarray]]) -> None:
        """Drop a failed resolver result and start the newest coalesced request."""
        with self._lock:
            if self._future is not completed:
                return
            self._future = None
            pending = self._pending
            self._pending = None
            if pending is not None:
                self._future = self._executor.submit(self._resolve, *pending)

    def _resolve(self, sequence: int, target_eef16: np.ndarray, current_joint14: np.ndarray) -> tuple[int, np.ndarray]:
        resolved = self._resolver(target_eef16.copy(), current_joint14.copy())
        return sequence, validate_action(resolved, "joint14")
