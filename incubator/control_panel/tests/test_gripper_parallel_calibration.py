from __future__ import annotations

from collections import deque
import copy
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import pytest

from incubator.control_panel import gripper_control
from incubator.control_panel.gripper_control import GripperProfile


def _profile(side: str, *, calibration_file: str | None = None) -> GripperProfile:
    return GripperProfile(
        side=side,
        port='/dev/ttyUSB1' if side == 'left' else '/dev/ttyUSB0',
        motor_id=1,
        baudrate=115200,
        speed_dps=1000.0,
        speed_source='gripper.speed_dps',
        read_timeout=0.1,
        wait_timeout=4.0,
        calibration_timeout=15.0,
        calibration_settle_time=0.24,
        min_torque_iq=-40,
        max_torque_iq=190,
        min_encoder=None,
        max_encoder=None,
        min_angle_deg=10.0,
        max_angle_deg=1010.0,
        arm_length_m=0.06543,
        apex_angle_max_deg=90.0,
        gear_ratio=10.0,
        force_limit=100.0,
        calibration_file=calibration_file or f'/unused/{side}.json',
        label=f'fake-{side}',
    )


def _payload(side: str) -> dict[str, Any]:
    return {
        'label': f'fake-{side}',
        'min_encoder': None,
        'max_encoder': None,
        'min_angle_deg': 100.0,
        'max_angle_deg': 1100.0,
    }


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.closed = False

    def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(copy.deepcopy(dict(payload)))

    def poll(self, timeout: float = 0.0) -> bool:
        del timeout
        return bool(self.events)

    def recv(self) -> dict[str, Any]:
        return self.events.pop(0)

    def close(self) -> None:
        self.closed = True


class _DeliveredThenRaisesConnection(_FakeConnection):
    """Model the ambiguous pipe case where delivery succeeds before send raises."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def send(self, payload: dict[str, Any]) -> None:
        super().send(payload)
        if payload.get('action') == 'calibrate' and not self.failed:
            self.failed = True
            raise BrokenPipeError('injected second calibration send failure')


class _FakeProcess:
    next_pid = 61000

    def __init__(
        self,
        *,
        target: Any,
        args: tuple[Any, ...],
        name: str,
        daemon: bool,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.pid = type(self).next_pid
        type(self).next_pid += 1
        self.exitcode: int | None = None
        self.alive = False
        self.join_calls = 0

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float = 0.0) -> None:
        del timeout
        self.join_calls += 1

    def terminate(self) -> None:
        self.alive = False
        self.exitcode = -15


class _StubbornProcess(_FakeProcess):
    def terminate(self) -> None:
        self.exitcode = None


class _FakeContext:
    def __init__(
        self,
        *,
        fail_start_index: int | None = None,
        stubborn_side: str | None = None,
    ) -> None:
        self.processes: list[_FakeProcess] = []
        self.parent_connections: list[_FakeConnection] = []
        self.fail_start_index = fail_start_index
        self.stubborn_side = stubborn_side

    def Pipe(self, duplex: bool = True) -> tuple[_FakeConnection, _FakeConnection]:
        assert duplex is True
        parent, child = _FakeConnection(), _FakeConnection()
        self.parent_connections.append(parent)
        return parent, child

    def Process(self, **kwargs: Any) -> _FakeProcess:
        index = len(self.processes)
        side = next(iter(kwargs['args'][1]))
        base = _StubbornProcess if side == self.stubborn_side else _FakeProcess

        if index == self.fail_start_index:
            class _StartFails(base):
                def start(self) -> None:
                    raise RuntimeError('injected second spawn failure')

            process = _StartFails(**kwargs)
        else:
            process = base(**kwargs)
        self.processes.append(process)
        return process


class _SecondCalibrationSendFailsContext(_FakeContext):
    def Pipe(self, duplex: bool = True) -> tuple[_FakeConnection, _FakeConnection]:
        assert duplex is True
        parent = (
            _DeliveredThenRaisesConnection()
            if len(self.parent_connections) == 1
            else _FakeConnection()
        )
        child = _FakeConnection()
        self.parent_connections.append(parent)
        return parent, child


def _manager(
    *,
    context: Any | None = None,
    auto_monitor: bool = False,
    calibration_writer: Callable[[Any], None] | None = None,
) -> tuple[gripper_control.GripperControlManager, Any, list[Any]]:
    context = context or _FakeContext()
    writes: list[Any] = []
    writer = calibration_writer or (
        lambda values: writes.append(copy.deepcopy(list(values)))
    )
    manager = gripper_control.GripperControlManager(
        context=context,
        auto_monitor=auto_monitor,
        sdk_module=Path(gripper_control.__file__),
        device_python=Path(sys.executable),
        port_discovery=lambda: ['/dev/ttyUSB0', '/dev/ttyUSB1'],
        port_validator=lambda port: gripper_control.validate_ttyusb_port(
            port, require_present=False
        ),
        permission_granter=lambda port: str(port),
        permission_policy_check=lambda port: None,
        calibration_writer=writer,
    )
    manager._profiles = {side: _profile(side) for side in ('left', 'right')}
    manager._status = 'idle'
    for side in ('left', 'right'):
        manager._sides[side]['configured'] = True
        manager._sides[side]['status'] = 'idle'
    return manager, context, writes


def _slot(manager: gripper_control.GripperControlManager, side: str) -> Any:
    return manager._workers[f'calibration:{side}']


def _completed_event(
    manager: gripper_control.GripperControlManager,
    side: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slot = _slot(manager, side)
    return {
        'event': 'completed',
        'owner_token': slot.owner_token,
        'command_id': slot.inflight['command_id'],
        'action': 'calibrate',
        'side': side,
        'results': [
            {
                'side': side,
                'target_percent': None,
                'actual_percent': 40.0 if side == 'left' else 60.0,
                'actual_width_m': 0.04 if side == 'left' else 0.06,
                'calibration': payload or _payload(side),
            }
        ],
    }


def _cleanup_event(slot: Any) -> dict[str, Any]:
    return {
        'event': 'cleanup_completed',
        'owner_token': slot.owner_token,
    }


def test_dual_calibration_spawns_every_lane_before_sending_either_command() -> None:
    context = _FakeContext(fail_start_index=1)
    manager, _, _ = _manager(context=context)
    try:
        with pytest.raises(
            gripper_control.GripperControlError,
            match='parallel|second spawn',
        ):
            manager.calibrate('both', 'parallel-partial-spawn-0001')

        assert len(context.processes) == 2
        # The first process existed, but no calibration command was released
        # before the second process start failed. Only containment shutdown is
        # allowed on its pipe.
        assert all(
            item.get('action') != 'calibrate'
            for item in context.parent_connections[0].sent
        )
        assert manager._workers == {}
        assert manager._calibration is None
        assert manager._hardware_unknown is False
        assert manager._side_revisions == {'left': 0, 'right': 0}
    finally:
        manager.close()


def test_second_command_send_failure_contains_both_lanes_and_latches_unknown() -> None:
    context = _SecondCalibrationSendFailsContext()
    manager, _, writes = _manager(context=context)
    try:
        with pytest.raises(
            gripper_control.GripperControlError,
            match='parallel|command|unknown',
        ):
            manager.calibrate('both', 'parallel-partial-send-0001')

        assert writes == []
        assert manager._calibration is None
        assert manager._workers == {}
        assert manager._last_calibration_completion is None
        assert manager._hardware_unknown is True
        assert manager._side_revisions == {'left': 1, 'right': 1}
        assert all(
            any(item.get('action') == 'calibrate' for item in connection.sent)
            for connection in context.parent_connections
        )
        state = manager.state()
        assert state['hardware_unknown'] is True
        assert state['global_busy'] is True
        assert all(state['sides'][side]['hardware_unknown'] for side in ('left', 'right'))
        assert all(not state['sides'][side]['can_position'] for side in ('left', 'right'))
        with pytest.raises(gripper_control.GripperConflictError, match='unknown'):
            manager.position(
                'left',
                30.0,
                'parallel-partial-send-move-0001',
                expected_side_revision=1,
            )
        with pytest.raises(gripper_control.GripperConflictError, match='unknown'):
            manager.calibrate('both', 'parallel-partial-send-retry-0001')
    finally:
        manager.close()


def test_success_waits_for_both_cleanup_acks_and_publishes_once() -> None:
    manager, _, writes = _manager()
    try:
        accepted = manager.calibrate('both', 'parallel-success-marker-0001')
        assert set(manager._workers) == {'calibration:left', 'calibration:right'}
        assert all(
            list(process.args[1]) in (['left'], ['right'])
            for process in [slot.process for slot in manager._workers.values()]
        )

        manager._apply_event_locked(_completed_event(manager, 'left'))
        state = manager.state()
        assert state['calibration']['completed_sides'] == ['left']
        assert state['calibration']['last_completion'] is None
        assert writes == []
        assert manager._sides['left']['last_completed_command_id'] is None
        assert manager._sides['right']['last_completed_command_id'] is None

        manager._apply_event_locked(_completed_event(manager, 'right'))

        assert len(writes) == 1
        assert [profile.side for profile, _ in writes[0]] == ['left', 'right']
        marker = manager.state()['calibration']['last_completion']
        assert marker['command_id'] == accepted['command_id']
        assert marker['sides'] == ['left', 'right']
        for side in ('left', 'right'):
            assert (
                manager._sides[side]['last_completed_command_id']
                == accepted['command_id']
            )
            assert manager._sides[side]['last_completed_action'] == 'calibrate'
        assert manager._side_revisions == {'left': 1, 'right': 1}
    finally:
        manager.close()


def test_one_side_failure_cancels_peer_without_half_commit() -> None:
    manager, _, writes = _manager()
    try:
        manager.calibrate('both', 'parallel-left-failure-0001')
        left = _slot(manager, 'left')
        right = _slot(manager, 'right')
        right.connection.events.append(_cleanup_event(right))

        manager._apply_event_locked(
            {
                'event': 'failed',
                'owner_token': left.owner_token,
                'command_id': left.inflight['command_id'],
                'error': 'injected left calibration failure',
                'hardware_unknown': False,
            }
        )

        assert writes == []
        assert manager._calibration is None
        assert manager._workers == {}
        assert manager._hardware_unknown is False
        assert any(item.get('action') == 'shutdown' for item in right.connection.sent)
        assert manager._sides['left']['status'] == 'error'
        assert manager._sides['right']['status'] == 'error'
        assert manager._last_calibration_completion is None
    finally:
        manager.close()


def test_dual_failure_preserves_each_side_error_and_never_commits() -> None:
    manager, _, writes = _manager()
    try:
        manager.calibrate('both', 'parallel-dual-failure-0001')
        left = _slot(manager, 'left')
        right = _slot(manager, 'right')
        right.connection.events.append(
            {
                'event': 'failed',
                'owner_token': right.owner_token,
                'command_id': right.inflight['command_id'],
                'error': 'injected right calibration failure',
                'hardware_unknown': False,
            }
        )
        manager._apply_event_locked(
            {
                'event': 'failed',
                'owner_token': left.owner_token,
                'command_id': left.inflight['command_id'],
                'error': 'injected left calibration failure',
                'hardware_unknown': False,
            }
        )

        assert writes == []
        assert 'left calibration failure' in manager._sides['left']['last_error']
        assert 'right calibration failure' in manager._sides['right']['last_error']
        assert manager._hardware_unknown is False
        assert manager._last_calibration_completion is None
    finally:
        manager.close()


def test_second_json_verification_failure_rolls_back_both_and_no_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != 'posix':
        pytest.skip('atomic calibration fsync/rollback requires POSIX semantics')
    monkeypatch.setattr(gripper_control, 'REPO_ROOT', tmp_path)
    calibration_root = tmp_path / 'calibration'
    calibration_root.mkdir()
    old_bytes: dict[str, bytes] = {}
    manager, _, _ = _manager(
        calibration_writer=gripper_control._atomic_write_calibration_payloads
    )
    try:
        for side in ('left', 'right'):
            path = calibration_root / f'{side}.json'
            path.write_text(
                '{"min_angle_deg": 1.0, "max_angle_deg": 2.0}\n',
                encoding='utf-8',
            )
            old_bytes[side] = path.read_bytes()
            manager._profiles[side] = _profile(
                side, calibration_file=str(path)
            )

        native_read = gripper_control._read_calibration
        verification_reads = 0

        def corrupt_second_read(path: Path) -> dict[str, Any]:
            nonlocal verification_reads
            verification_reads += 1
            if verification_reads == 2:
                raise gripper_control.GripperControlError(
                    f'injected corrupt second JSON: {path.name}'
                )
            return native_read(path)

        monkeypatch.setattr(
            gripper_control, '_read_calibration', corrupt_second_read
        )
        manager.calibrate('both', 'parallel-second-json-corrupt-0001')
        manager._apply_event_locked(_completed_event(manager, 'left'))
        manager._apply_event_locked(_completed_event(manager, 'right'))

        for side in ('left', 'right'):
            path = calibration_root / f'{side}.json'
            assert path.read_bytes() == old_bytes[side]
            assert manager._sides[side]['last_completed_command_id'] is None
        assert manager._last_calibration_completion is None
        assert manager._calibration is None
        manifests = list(calibration_root.glob('.gripper-calibration-*.json'))
        assert len(manifests) == 1
        assert 'rolled_back' in manifests[0].read_text(encoding='utf-8')
    finally:
        manager.close()


def test_rollback_failure_latches_unknown_and_persists_restart_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != 'posix':
        pytest.skip('atomic calibration fsync/rollback requires POSIX semantics')
    monkeypatch.setattr(gripper_control, 'REPO_ROOT', tmp_path)
    calibration_root = tmp_path / 'calibration'
    calibration_root.mkdir()
    paths = {
        side: calibration_root / f'{side}.json'
        for side in ('left', 'right')
    }
    for path in paths.values():
        path.write_text(
            '{"min_angle_deg": 1.0, "max_angle_deg": 2.0}\n',
            encoding='utf-8',
        )

    native_replace = gripper_control.os.replace
    injected = {'commit': False, 'rollback': False}

    def fail_second_commit_and_first_rollback(source: Any, target: Any) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if (
            target_path == paths['right']
            and source_path.name.startswith('.right.json.')
            and '.rollback.' not in source_path.name
            and not injected['commit']
        ):
            injected['commit'] = True
            raise OSError('injected second calibration commit failure')
        if (
            target_path == paths['left']
            and '.rollback.' in source_path.name
            and not injected['rollback']
        ):
            injected['rollback'] = True
            raise OSError('injected first calibration rollback failure')
        native_replace(source, target)

    monkeypatch.setattr(gripper_control.os, 'replace', fail_second_commit_and_first_rollback)
    manager, _, _ = _manager(
        calibration_writer=gripper_control._atomic_write_calibration_payloads
    )
    try:
        for side, path in paths.items():
            manager._profiles[side] = _profile(side, calibration_file=str(path))
        manager.calibrate('both', 'parallel-rollback-failure-0001')
        manager._apply_event_locked(_completed_event(manager, 'left'))
        manager._apply_event_locked(_completed_event(manager, 'right'))

        assert injected == {'commit': True, 'rollback': True}
        assert manager._hardware_unknown is True
        assert manager._last_calibration_completion is None
        assert manager._calibration is None
        assert manager._profiles['left'].min_angle_deg == 10.0
        assert manager._profiles['right'].min_angle_deg == 10.0
        assert '100.0' in paths['left'].read_text(encoding='utf-8')
        assert '1.0' in paths['right'].read_text(encoding='utf-8')
        manifests = list(calibration_root.glob('.gripper-calibration-*.json'))
        assert len(manifests) == 1
        manifest = manifests[0].read_text(encoding='utf-8')
        assert 'rollback_failed' in manifest
        assert 'injected first calibration rollback failure' in manifest

        state = manager.state()
        assert state['hardware_unknown'] is True
        assert state['global_busy'] is True
        assert state['can_calibrate'] is False
        assert state['can_position'] is False
        assert state['can_probe'] is False
        with pytest.raises(gripper_control.GripperConflictError, match='unknown'):
            manager.position(
                'left',
                20.0,
                'parallel-rollback-failure-move-0001',
                expected_side_revision=1,
            )
        with pytest.raises(gripper_control.GripperConflictError, match='unknown'):
            manager.calibrate('both', 'parallel-rollback-failure-retry-0001')
        with pytest.raises(gripper_control.GripperControlError, match='unresolved'):
            gripper_control._assert_no_unresolved_calibration_transactions(
                manager._profiles
            )
    finally:
        manager.close()


def test_stubborn_peer_latches_global_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gripper_control, 'STOP_TIMEOUT_S', 0.01)
    context = _FakeContext(stubborn_side='right')
    manager, _, writes = _manager(context=context)
    try:
        manager.calibrate('both', 'parallel-stubborn-peer-0001')
        left = _slot(manager, 'left')
        manager._apply_event_locked(
            {
                'event': 'failed',
                'owner_token': left.owner_token,
                'command_id': left.inflight['command_id'],
                'error': 'injected left failure before stubborn cleanup',
                'hardware_unknown': False,
            }
        )

        assert writes == []
        assert manager._hardware_unknown is True
        assert 'calibration:right' in manager._workers
        state = manager.state()
        assert state['global_busy'] is True
        assert state['sides']['left']['can_position'] is False
        assert state['sides']['right']['can_position'] is False
    finally:
        for slot in manager._workers.values():
            slot.process.alive = False
        manager.close()


def test_stop_cancels_and_reaps_both_calibration_lanes_without_commit() -> None:
    manager, _, writes = _manager()
    try:
        manager.calibrate('both', 'parallel-stop-0001')
        slots = [_slot(manager, side) for side in ('left', 'right')]
        slots[0].connection.events.append(_completed_event(manager, 'left'))
        slots[1].connection.events.append(_cleanup_event(slots[1]))

        result = manager.stop('test_stop')

        assert result['ok'] is True
        assert result['stopped'] is True
        assert manager._calibration is None
        assert manager._workers == {}
        assert manager._hardware_unknown is False
        assert writes == []
        assert all(
            any(item.get('action') == 'shutdown' for item in slot.connection.sent)
            for slot in slots
        )
    finally:
        manager.close()


def test_timeout_aborts_transaction_and_reaps_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _, writes = _manager()
    try:
        manager.calibrate('both', 'parallel-timeout-0001')
        slots = [_slot(manager, side) for side in ('left', 'right')]

        def safe_stop(slot: Any, reason: str) -> tuple[bool, bool]:
            del reason
            slot.process.alive = False
            slot.inflight = None
            slot.cleanup_acknowledged = True
            return True, False

        monkeypatch.setattr(manager, '_stop_slot_locked', safe_stop)
        slots[0].inflight['sent_at'] = (
            time.monotonic() - gripper_control.CALIBRATE_ONE_TIMEOUT_S - 1.0
        )
        manager._refresh_locked()

        assert writes == []
        assert manager._calibration is None
        assert manager._workers == {}
        assert manager._hardware_unknown is False
        assert 'timed out' in manager._last_error
    finally:
        manager.close()


class _Endpoint:
    def __init__(self) -> None:
        self.peer: '_Endpoint | None' = None
        self.items: deque[dict[str, Any]] = deque()
        self.condition = threading.Condition()

    def send(self, payload: dict[str, Any]) -> None:
        assert self.peer is not None
        with self.peer.condition:
            self.peer.items.append(copy.deepcopy(dict(payload)))
            self.peer.condition.notify_all()

    def poll(self, timeout: float = 0.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self.condition:
            while not self.items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return True

    def recv(self) -> dict[str, Any]:
        with self.condition:
            if not self.items:
                raise EOFError
            return self.items.popleft()

    def close(self) -> None:
        # A thread-backed test endpoint represents both duplicated process FDs.
        pass


class _ThreadProcess:
    next_pid = 71000

    def __init__(
        self,
        *,
        args: tuple[Any, ...],
        name: str,
        delays: dict[str, float],
        starts: dict[str, float],
        **_: Any,
    ) -> None:
        self.connection = args[0]
        self.side = next(iter(args[1]))
        self.token = args[2]
        self.name = name
        self.delays = delays
        self.starts = starts
        self.pid = type(self).next_pid
        type(self).next_pid += 1
        self.exitcode: int | None = None
        self.alive = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.alive = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _emit(self, event: str, **values: Any) -> None:
        self.connection.send(
            {'event': event, 'owner_token': self.token, **values}
        )

    def _run(self) -> None:
        try:
            self._emit('ready', pid=self.pid, pgid=self.pid)
            while not self.connection.poll(0.02):
                if self.stop_event.is_set():
                    return
            command = self.connection.recv()
            if command.get('action') == 'shutdown':
                self._emit('stopped')
                return
            self.starts[self.side] = time.monotonic()
            self._emit(
                'started',
                command_id=command['command_id'],
                action='calibrate',
                side=self.side,
            )
            if self.stop_event.wait(self.delays[self.side]):
                self._emit('cleanup_completed')
                return
            self._emit(
                'completed',
                command_id=command['command_id'],
                action='calibrate',
                side=self.side,
                results=[
                    {
                        'side': self.side,
                        'target_percent': None,
                        'actual_percent': 50.0,
                        'actual_width_m': 0.05,
                        'calibration': _payload(self.side),
                    }
                ],
            )
        finally:
            self.alive = False
            self.exitcode = 0

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float = 0.0) -> None:
        if self.thread is not None:
            self.thread.join(timeout)

    def terminate(self) -> None:
        self.stop_event.set()
        self.join(0.1)
        self.alive = False
        self.exitcode = -15


class _ThreadContext:
    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays
        self.starts: dict[str, float] = {}
        self.processes: list[_ThreadProcess] = []

    def Pipe(self, duplex: bool = True) -> tuple[_Endpoint, _Endpoint]:
        assert duplex is True
        parent, child = _Endpoint(), _Endpoint()
        parent.peer = child
        child.peer = parent
        return parent, child

    def Process(self, **kwargs: Any) -> _ThreadProcess:
        process = _ThreadProcess(
            **kwargs,
            delays=self.delays,
            starts=self.starts,
        )
        self.processes.append(process)
        return process


def test_parallel_wall_clock_tracks_max_side_duration_not_sum() -> None:
    delays = {'left': 0.35, 'right': 0.35}
    context = _ThreadContext(delays)
    manager, _, writes = _manager(context=context, auto_monitor=True)
    started = time.monotonic()
    try:
        accepted = manager.calibrate('both', 'parallel-wall-clock-0001')
        deadline = time.monotonic() + 2.0
        marker = None
        while time.monotonic() < deadline:
            marker = manager.state()['calibration']['last_completion']
            if marker and marker['command_id'] == accepted['command_id']:
                break
            time.sleep(0.01)
        elapsed = time.monotonic() - started

        assert marker is not None
        assert len(writes) == 1
        assert abs(context.starts['left'] - context.starts['right']) < 0.08
        assert elapsed >= max(delays.values()) * 0.8
        assert elapsed < sum(delays.values()) * 0.8
    finally:
        manager.close()


def test_closed_manager_rejects_parallel_calibration_without_spawning() -> None:
    manager, context, _ = _manager()
    manager.close()

    with pytest.raises(
        gripper_control.GripperConflictError,
        match='closed|shutting down',
    ):
        manager.calibrate('both', 'parallel-after-close-0001')

    assert context.processes == []
    assert manager._workers == {}
    assert manager._calibration is None
