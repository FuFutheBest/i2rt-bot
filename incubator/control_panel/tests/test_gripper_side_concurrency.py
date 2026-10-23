from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import pytest

from incubator.control_panel import gripper_control
from incubator.control_panel.gripper_control import GripperProfile


def _profile(side: str) -> GripperProfile:
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
        calibration_file=f'/unused/{side}.json',
        label=f'fake-{side}',
    )


def _verified_artifact(profile: GripperProfile) -> dict[str, Any]:
    return {
        "name": Path(profile.calibration_file).name,
        "present": True,
        "limits_valid": True,
        "valid": True,
        "status": "valid",
        "identity_status": "verified",
        "expected": {"motor_id": profile.motor_id, "port": profile.port},
        "observed": {"motor_id": profile.motor_id, "port": profile.port},
        "reason": None,
    }


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.closed = False

    def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(dict(payload))

    def poll(self, timeout: float = 0.0) -> bool:
        del timeout
        return bool(self.events)

    def recv(self) -> dict[str, Any]:
        return self.events.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    next_pid = 41000

    def __init__(self, *, target: Any, args: tuple[Any, ...], name: str, daemon: bool) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
        self.exitcode: int | None = None
        self.alive = False

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float = 0.0) -> None:
        del timeout

    def terminate(self) -> None:
        self.alive = False
        self.exitcode = -15


class _FakeContext:
    def __init__(self) -> None:
        self.processes: list[_FakeProcess] = []

    def Pipe(self, duplex: bool = True) -> tuple[_FakeConnection, _FakeConnection]:
        assert duplex is True
        return _FakeConnection(), _FakeConnection()

    def Process(self, **kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(**kwargs)
        self.processes.append(process)
        return process


def _manager() -> tuple[gripper_control.GripperControlManager, _FakeContext]:
    context = _FakeContext()
    manager = gripper_control.GripperControlManager(
        context=context,
        auto_monitor=False,
        sdk_module=Path(gripper_control.__file__),
        device_python=Path(sys.executable),
        port_discovery=lambda: ['/dev/ttyUSB0', '/dev/ttyUSB1'],
        port_validator=lambda port: gripper_control.validate_ttyusb_port(
            port, require_present=False
        ),
        permission_granter=lambda port: str(port),
        permission_policy_check=lambda port: None,
    )
    manager._profiles = {side: _profile(side) for side in ('left', 'right')}
    manager._status = 'idle'
    for side in ('left', 'right'):
        manager._sides[side]['configured'] = True
        manager._sides[side]['status'] = 'idle'
    return manager, context


def test_left_and_right_workers_are_truly_active_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        "_calibration_artifact_state",
        _verified_artifact,
    )
    manager, context = _manager()
    try:
        left = manager.position(
            'left', 25.0, 'left-position-0001', expected_side_revision=0
        )
        manager._dispatch_locked()
        left_only = manager.state()
        assert left_only['busy_sides'] == ['left']
        assert left_only['sides']['left']['can_position'] is False
        assert left_only['sides']['left']['can_queue_position'] is True
        assert left_only['sides']['right']['can_position'] is True

        right = manager.position(
            'right', 75.0, 'right-position-0001', expected_side_revision=0
        )
        manager._dispatch_locked()

        assert left['accepted'] is True
        assert right['accepted'] is True
        assert set(manager._workers) == {'left', 'right'}
        assert all(slot.process.is_alive() for slot in manager._workers.values())
        assert all(slot.inflight is not None for slot in manager._workers.values())
        assert len(context.processes) == 2
        worker_profile_sides = [set(process.args[1]) for process in context.processes]
        assert worker_profile_sides == [{'left'}, {'right'}]
        state = manager.state()
        assert state['global_busy'] is False
        assert state['busy_sides'] == ['left', 'right']
        assert state['sides']['left']['can_queue_position'] is True
        assert state['sides']['right']['can_queue_position'] is True
        assert state['worker']['pids'] == sorted(
            process.pid for process in context.processes
        )
    finally:
        manager.close()


def test_side_revision_rejects_stale_same_side_but_replays_idempotently() -> None:
    manager, _ = _manager()
    try:
        first = manager.position(
            'left', 10.0, 'left-position-cas-1', expected_side_revision=0
        )
        assert first['side_revision'] == 1
        duplicate = manager.position(
            'left', 10.0, 'left-position-cas-1', expected_side_revision=0
        )
        assert duplicate['duplicate'] is True
        assert duplicate['accepted'] is False

        with pytest.raises(gripper_control.GripperConflictError, match='Stale left'):
            manager.position(
                'left', 20.0, 'left-position-cas-2', expected_side_revision=0
            )

        replacement = manager.position(
            'left', 20.0, 'left-position-cas-3', expected_side_revision=1
        )
        assert replacement['accepted'] is True
        assert replacement['side_revision'] == 2
        assert manager._pending_positions['left']['target_percent'] == 20.0
    finally:
        manager.close()


def test_calibration_invalidates_old_position_tokens_and_locks_both_sides() -> None:
    manager, _ = _manager()
    try:
        calibration = manager.calibrate('both', 'calibrate-both-cas-1')
        assert calibration['accepted'] is True
        state = manager.state()
        assert state['global_busy'] is True
        assert state['busy_sides'] == ['left', 'right']
        assert state['sides']['left']['can_position'] is False
        assert state['sides']['right']['can_position'] is False
        with pytest.raises(gripper_control.GripperConflictError, match='Stale left'):
            manager.position(
                'left', 50.0, 'left-after-calibration', expected_side_revision=0
            )
    finally:
        manager.close()


def test_one_side_failure_releases_only_that_side_and_preserves_other_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        "_calibration_artifact_state",
        _verified_artifact,
    )
    manager, _ = _manager()
    try:
        manager.position(
            'left', 30.0, 'left-failure-0001', expected_side_revision=0
        )
        manager.position(
            'right', 70.0, 'right-pending-0001', expected_side_revision=0
        )
        # Keep right queued while left is dispatched, then inject a normal
        # per-side command failure (hardware state is not globally unknown).
        manager._last_sent_at['right'] = time.monotonic()
        manager._dispatch_locked()
        assert set(manager._workers) == {'left'}
        assert set(manager._pending_positions) == {'right'}
        left_slot = manager._workers['left']
        manager._apply_event_locked(
            {
                'event': 'failed',
                'owner_token': left_slot.owner_token,
                'command_id': left_slot.inflight['command_id'],
                'error': 'injected left failure',
                'hardware_unknown': False,
            }
        )

        assert left_slot.inflight is None
        assert set(manager._pending_positions) == {'right'}
        state = manager.state()
        assert state['global_busy'] is False
        assert state['busy_sides'] == ['right']
        assert state['sides']['left']['status'] == 'error'
        assert state['sides']['left']['can_position'] is True
        assert state['sides']['right']['status'] == 'queued'
        assert state['sides']['right']['busy'] is True
    finally:
        manager.close()


def test_global_runtime_blocker_disables_both_sides_while_idle() -> None:
    manager, _ = _manager()
    try:
        state = manager.state(['Stop inference before controlling grippers'])
        assert state['busy_sides'] == []
        assert state['sides']['left']['can_position'] is False
        assert state['sides']['right']['can_position'] is False
        assert 'Stop inference' in state['sides']['left']['blockers'][0]
    finally:
        manager.close()
