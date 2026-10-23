from __future__ import annotations

import threading
import time
from typing import Any, Dict

import pytest

from incubator.control_panel.control_operations import (
    ControlOperationDispatcher,
    ControlOperationQueueFull,
)


class RecordingBroker:
    def __init__(self) -> None:
        self.acknowledged: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.failure_event = threading.Event()

    def acknowledge(self, token: str) -> Dict[str, Any]:
        self.acknowledged.append(token)
        return {
            'token': token,
            'status': 'acknowledged',
            'acknowledged': True,
        }

    def fail(self, token: str, reason: Any) -> Dict[str, Any]:
        self.failed.append((token, str(reason)))
        self.failure_event.set()
        return {'token': token, 'status': 'failed', 'reason': str(reason)}


def _wait_idle(dispatcher: ControlOperationDispatcher, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = dispatcher.snapshot()
        if snapshot['pending'] == 0 and snapshot['running'] == 0:
            return
        time.sleep(0.01)
    raise AssertionError(f'dispatcher did not become idle: {dispatcher.snapshot()}')


def test_stop_cancels_older_queued_start_for_the_same_scope() -> None:
    broker = RecordingBroker()
    blocker_entered = threading.Event()
    release_blocker = threading.Event()
    stop_finished = threading.Event()
    calls: list[str] = []
    dispatcher = ControlOperationDispatcher(
        broker,
        worker_count=1,
        max_pending=8,
    )

    def blocker() -> Dict[str, Any]:
        blocker_entered.set()
        assert release_blocker.wait(timeout=2.0)
        return {'ok': True}

    dispatcher.submit(
        token='token-blocker',
        scope='device:blocker',
        action='start',
        desired='running',
        operation=blocker,
    )
    assert blocker_entered.wait(timeout=1.0)

    dispatcher.submit(
        token='token-start',
        scope='device:camera',
        action='start',
        desired='running',
        operation=lambda: calls.append('start') or {'ok': True},
    )
    dispatcher.submit(
        token='token-stop',
        scope='device:camera',
        action='stop',
        desired='stopped',
        operation=lambda: (
            calls.append('stop'),
            stop_finished.set(),
            {'ok': True},
        )[-1],
    )

    release_blocker.set()
    assert stop_finished.wait(timeout=2.0)
    _wait_idle(dispatcher)
    dispatcher.close()

    assert calls == ['stop']


def test_workflow_stop_signals_running_stage_then_runs_after_it() -> None:
    broker = RecordingBroker()
    start_entered = threading.Event()
    cancel_received = threading.Event()
    stop_finished = threading.Event()
    calls: list[str] = []

    def cancel(scope: str) -> None:
        calls.append(f'cancel:{scope}')
        cancel_received.set()

    dispatcher = ControlOperationDispatcher(
        broker,
        worker_count=2,
        max_pending=8,
        cancel_callback=cancel,
    )

    def start() -> Dict[str, Any]:
        calls.append('start')
        start_entered.set()
        assert cancel_received.wait(timeout=2.0)
        calls.append('start-exit')
        return {'ok': True}

    def stop() -> Dict[str, Any]:
        calls.append('stop')
        stop_finished.set()
        return {'ok': True}

    dispatcher.submit(
        token='token-start',
        scope='workflow:inference',
        action='start',
        desired='running',
        operation=start,
    )
    assert start_entered.wait(timeout=1.0)
    dispatcher.submit(
        token='token-stop',
        scope='workflow:runtime',
        action='stop',
        desired='stopped',
        operation=stop,
    )

    assert stop_finished.wait(timeout=2.0)
    _wait_idle(dispatcher)
    dispatcher.close()

    assert calls == [
        'start',
        'cancel:workflow:runtime',
        'start-exit',
        'stop',
    ]


def test_fastwam_server_and_runtime_start_run_in_parallel() -> None:
    broker = RecordingBroker()
    runtime_entered = threading.Event()
    server_entered = threading.Event()
    release = threading.Event()
    dispatcher = ControlOperationDispatcher(broker, worker_count=2, max_pending=8)

    def runtime_start() -> Dict[str, Any]:
        runtime_entered.set()
        assert server_entered.wait(timeout=2.0)
        assert release.wait(timeout=2.0)
        return {'ok': True}

    def server_start() -> Dict[str, Any]:
        server_entered.set()
        assert runtime_entered.wait(timeout=2.0)
        assert release.wait(timeout=2.0)
        return {'ok': True}

    dispatcher.submit(
        token='token-runtime',
        scope='workflow:runtime',
        action='start',
        desired='running',
        operation=runtime_start,
    )
    dispatcher.submit(
        token='token-server',
        scope='workflow:server',
        action='start',
        desired='running',
        operation=server_start,
    )

    assert runtime_entered.wait(timeout=1.0)
    assert server_entered.wait(timeout=1.0)
    release.set()
    _wait_idle(dispatcher)
    dispatcher.close()


def test_device_stop_signals_running_start_before_serialized_cleanup() -> None:
    broker = RecordingBroker()
    start_entered = threading.Event()
    cancel_received = threading.Event()
    stop_finished = threading.Event()
    calls: list[str] = []

    def cancel(scope: str) -> None:
        calls.append(f'cancel:{scope}')
        cancel_received.set()

    dispatcher = ControlOperationDispatcher(
        broker,
        worker_count=2,
        max_pending=8,
        cancel_callback=cancel,
    )

    def start() -> Dict[str, Any]:
        calls.append('start')
        start_entered.set()
        assert cancel_received.wait(timeout=2.0)
        calls.append('start-exit')
        return {'ok': True}

    def stop() -> Dict[str, Any]:
        calls.append('stop')
        stop_finished.set()
        return {'ok': True}

    dispatcher.submit(
        token='token-device-start',
        scope='device:RizonRobot:2',
        action='start',
        desired='running',
        operation=start,
    )
    assert start_entered.wait(timeout=1.0)
    dispatcher.submit(
        token='token-device-stop',
        scope='device:RizonRobot:2',
        action='stop',
        desired='stopped',
        operation=stop,
    )

    assert stop_finished.wait(timeout=2.0)
    _wait_idle(dispatcher)
    dispatcher.close()

    assert calls == [
        'start',
        'cancel:device:RizonRobot:2',
        'start-exit',
        'stop',
    ]


def test_background_failure_is_reported_to_transition_broker() -> None:
    broker = RecordingBroker()
    dispatcher = ControlOperationDispatcher(broker, worker_count=1)

    def fail() -> Dict[str, Any]:
        raise RuntimeError('simulated failure')

    dispatcher.submit(
        token='token-failure',
        scope='device:camera',
        action='start',
        desired='running',
        operation=fail,
    )

    assert broker.failure_event.wait(timeout=2.0)
    _wait_idle(dispatcher)
    dispatcher.close()
    assert broker.failed == [
        ('token-failure', 'RuntimeError: simulated failure')
    ]


def test_bounded_queue_rejects_excess_pending_work() -> None:
    broker = RecordingBroker()
    entered = threading.Event()
    release = threading.Event()
    dispatcher = ControlOperationDispatcher(
        broker,
        worker_count=1,
        max_pending=1,
    )

    def blocker() -> Dict[str, Any]:
        entered.set()
        assert release.wait(timeout=2.0)
        return {'ok': True}

    dispatcher.submit(
        token='token-running',
        scope='device:running',
        action='start',
        desired='running',
        operation=blocker,
    )
    assert entered.wait(timeout=1.0)
    dispatcher.submit(
        token='token-queued',
        scope='device:queued',
        action='start',
        desired='running',
        operation=lambda: {'ok': True},
    )

    with pytest.raises(ControlOperationQueueFull):
        dispatcher.submit(
            token='token-overflow',
            scope='device:overflow',
            action='start',
            desired='running',
            operation=lambda: {'ok': True},
        )

    release.set()
    _wait_idle(dispatcher)
    dispatcher.close()
