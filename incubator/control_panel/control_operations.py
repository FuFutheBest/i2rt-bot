"""Bounded background execution for UI lifecycle commands.

The HTTP control routes retain their synchronous behavior for legacy clients.
The browser can opt in to fast command acceptance, after which this dispatcher
executes the operation while ``StateSyncBroker`` remains the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import queue
import threading
from typing import Any, Callable, Dict, Optional, Protocol


class TransitionBroker(Protocol):
    def acknowledge(self, token: str) -> Optional[Dict[str, Any]]: ...

    def fail(self, token: str, reason: Any) -> Optional[Dict[str, Any]]: ...


class ControlOperationQueueFull(RuntimeError):
    """The bounded control queue cannot accept another operation."""


class ControlOperationDispatcherClosed(RuntimeError):
    """The server is shutting down and no longer accepts operations."""


@dataclass(order=True)
class _QueuedOperation:
    priority: int
    sequence: int
    token: str = field(compare=False)
    scope: str = field(compare=False)
    lock_scope: str = field(compare=False)
    action: str = field(compare=False)
    operation: Callable[[], Dict[str, Any]] = field(compare=False, repr=False)
    cancelled: bool = field(default=False, compare=False)
    running: bool = field(default=False, compare=False)
    finished: bool = field(default=False, compare=False)


class ControlOperationDispatcher:
    """Run slow controls off request threads without losing ordering.

    Operations for different resources may run in parallel. Operations for one
    scope are serialized, and a safety-reducing stop cancels older queued work
    for that scope before it is allowed to execute.
    """

    def __init__(
        self,
        broker: TransitionBroker,
        *,
        worker_count: int = 8,
        max_pending: int = 64,
        log_callback: Optional[Callable[[str, str], None]] = None,
        cancel_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._broker = broker
        self._worker_count = int(worker_count)
        self._queue: queue.PriorityQueue[_QueuedOperation] = queue.PriorityQueue(
            maxsize=int(max_pending)
        )
        self._log_callback = log_callback
        self._cancel_callback = cancel_callback
        self._lock = threading.RLock()
        self._sequence = itertools.count()
        self._threads: list[threading.Thread] = []
        self._scope_locks: Dict[str, threading.Lock] = {}
        self._jobs_by_scope: Dict[str, list[_QueuedOperation]] = {}
        self._closed = False

    @staticmethod
    def _is_stop(action: str, desired: str) -> bool:
        return str(action).casefold() == "stop" or str(desired).casefold() == "stopped"

    @staticmethod
    def _lock_scope(scope: str) -> str:
        normalized = str(scope)
        # FastWAM model loading is independent from the device runtime. Keep
        # runtime/inference serialized because they share SHM and lifecycle
        # dependencies, but do not queue the Server behind a slow robot gohome
        # or runtime startup.
        if normalized == "workflow:server":
            return normalized
        if normalized.startswith("workflow:"):
            return "workflow"
        return normalized

    def _ensure_workers_locked(self) -> None:
        if self._threads:
            return
        for index in range(self._worker_count):
            thread = threading.Thread(
                target=self._worker,
                name=f"control-operation-{index + 1}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def submit(
        self,
        *,
        token: str,
        scope: str,
        action: str,
        desired: str,
        operation: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_scope = str(scope)
        lock_scope = self._lock_scope(normalized_scope)
        normalized_action = str(action)
        stop = self._is_stop(normalized_action, desired)
        item = _QueuedOperation(
            priority=0 if stop else 10,
            sequence=next(self._sequence),
            token=str(token),
            scope=normalized_scope,
            lock_scope=lock_scope,
            action=normalized_action,
            operation=operation,
        )

        should_signal_stop = stop
        with self._lock:
            if self._closed:
                raise ControlOperationDispatcherClosed(
                    "Control server is shutting down"
                )
            if stop:
                for existing in self._jobs_by_scope.get(lock_scope, []):
                    if existing.finished or existing.action.casefold() == "stop":
                        continue
                    if existing.running:
                        should_signal_stop = True
                    else:
                        existing.cancelled = True
            self._jobs_by_scope.setdefault(lock_scope, []).append(item)
            self._scope_locks.setdefault(lock_scope, threading.Lock())
            try:
                self._queue.put_nowait(item)
            except queue.Full as exc:
                self._jobs_by_scope[lock_scope].remove(item)
                if not self._jobs_by_scope[lock_scope]:
                    self._jobs_by_scope.pop(lock_scope, None)
                    self._scope_locks.pop(lock_scope, None)
                raise ControlOperationQueueFull(
                    "Too many control operations are waiting"
                ) from exc
            self._ensure_workers_locked()

        if should_signal_stop and self._cancel_callback is not None:
            try:
                self._cancel_callback(normalized_scope)
            except BaseException as exc:
                self._log(
                    "WARNING",
                    f"Unable to signal cancellation for {normalized_scope}: "
                    f"{type(exc).__name__}: {exc}",
                )

        acknowledged = self._broker.acknowledge(item.token)
        return acknowledged or {
            "token": item.token,
            "scope": item.scope,
            "action": item.action,
            "desired": str(desired),
            "status": "acknowledged",
            "acknowledged": True,
        }

    def _log(self, level: str, message: str) -> None:
        if self._log_callback is None:
            return
        try:
            self._log_callback(level, message)
        except BaseException:
            pass

    def _finish(self, item: _QueuedOperation) -> None:
        with self._lock:
            item.running = False
            item.finished = True
            jobs = self._jobs_by_scope.get(item.lock_scope)
            if jobs is not None:
                try:
                    jobs.remove(item)
                except ValueError:
                    pass
                if not jobs:
                    self._jobs_by_scope.pop(item.lock_scope, None)
                    self._scope_locks.pop(item.lock_scope, None)

    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                with self._lock:
                    if self._closed:
                        return
                continue

            try:
                with self._lock:
                    scope_lock = self._scope_locks.get(item.lock_scope)
                    cancelled = item.cancelled
                if cancelled or scope_lock is None:
                    continue
                with scope_lock:
                    with self._lock:
                        if item.cancelled or self._closed:
                            continue
                        item.running = True
                    try:
                        item.operation()
                    except BaseException as exc:
                        self._broker.fail(
                            item.token,
                            f"{type(exc).__name__}: {exc}",
                        )
                        self._log(
                            "ERROR",
                            f"Asynchronous {item.scope} {item.action} failed: "
                            f"{type(exc).__name__}: {exc}",
                        )
                    else:
                        # Wake the broker after the operation changes process or
                        # SHM state. Completion is still observation-driven.
                        self._broker.acknowledge(item.token)
            finally:
                self._finish(item)
                self._queue.task_done()

    def close(self) -> None:
        """Reject new work and cancel operations that have not started."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            queued = [
                item
                for jobs in self._jobs_by_scope.values()
                for item in jobs
                if not item.running and not item.finished
            ]
            for item in queued:
                item.cancelled = True
        for item in queued:
            self._broker.fail(item.token, "Control server is shutting down")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            jobs = [item for values in self._jobs_by_scope.values() for item in values]
            return {
                "closed": self._closed,
                "pending": sum(not item.running and not item.finished for item in jobs),
                "running": sum(item.running and not item.finished for item in jobs),
                "scopes": sorted({item.scope for item in jobs if not item.finished}),
            }


__all__ = [
    "ControlOperationDispatcher",
    "ControlOperationDispatcherClosed",
    "ControlOperationQueueFull",
]
