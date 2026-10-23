from __future__ import annotations

import os
from pathlib import Path
import sys
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
    next_pid = 51000

    def __init__(self, *, target: Any, args: tuple[Any, ...], name: str, daemon: bool) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
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


class _StubbornProcess(_FakeProcess):
    def terminate(self) -> None:
        self.exitcode = None


class _StubbornContext(_FakeContext):
    def Process(self, **kwargs: Any) -> _StubbornProcess:
        process = _StubbornProcess(**kwargs)
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


def test_abrupt_inflight_exit_latches_unknown_and_blocks_both_sides() -> None:
    manager, _ = _manager()
    try:
        manager.position(
            'left', 25.0, 'left-abrupt-exit-0001', expected_side_revision=0
        )
        manager._dispatch_locked()
        slot = manager._workers['left']
        slot.process.alive = False
        slot.process.exitcode = -9

        manager._refresh_locked()

        state = manager.state()
        assert manager._hardware_unknown is True
        assert state['global_busy'] is True
        assert state['sides']['left']['can_position'] is False
        assert state['sides']['right']['can_position'] is False
        with pytest.raises(gripper_control.GripperControlError, match='unknown'):
            manager.stop('must_not_clear_unknown')
    finally:
        manager.close()


def test_stop_drains_cleanup_failure_before_discarding_worker_pipe() -> None:
    manager, _ = _manager()
    try:
        manager.position(
            'left', 30.0, 'left-cleanup-failure-0001', expected_side_revision=0
        )
        manager._dispatch_locked()
        slot = manager._workers['left']
        slot.connection.events.append(
            {
                'event': 'failed',
                'owner_token': slot.owner_token,
                'command_id': slot.inflight['command_id'],
                'action': 'position',
                'side': 'left',
                'error': 'injected cleanup failure',
                'hardware_unknown': True,
            }
        )
        slot.process.alive = False
        slot.process.exitcode = 1

        with pytest.raises(gripper_control.GripperControlError, match='unknown'):
            manager.stop('observe_cleanup_failure')

        assert manager._hardware_unknown is True
        assert slot.connection.closed is True
        assert slot.process.join_calls >= 1
    finally:
        manager.close()


class _FakeConfig:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _DisableAlwaysFailsGripper:
    instances: list['_DisableAlwaysFailsGripper'] = []

    def __init__(self, robot: Any, *, config: Any, min_vel: float, max_vel: float) -> None:
        del robot, min_vel, max_vel
        self.config = config
        type(self).instances.append(self)

    def Enable(self, label: str) -> None:
        self.label = label

    def Disable(self) -> None:
        raise RuntimeError('injected disable failure')


def test_post_enable_preflight_failure_keeps_handle_for_cleanup_retry() -> None:
    calls = 0

    def preflight(profiles: Any) -> None:
        nonlocal calls
        del profiles
        calls += 1
        if calls == 2:
            raise RuntimeError('injected post-enable race')

    _DisableAlwaysFailsGripper.instances = []
    runtime = gripper_control._WorkerRuntime(
        {'left': _profile('left')},
        sdk_loader=lambda: (_FakeConfig, _DisableAlwaysFailsGripper),
        calibration_writer=lambda values: None,
        preflight=preflight,
    )

    with pytest.raises(RuntimeError, match='post-enable race'):
        runtime._create('left', recalibrate=False)

    assert runtime.grippers.get('left') is _DisableAlwaysFailsGripper.instances[0]
    assert gripper_control._close_worker_runtime(runtime, attempts=2) is not None


def test_calibration_reload_error_is_contained_as_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    manager, _ = _manager()
    try:
        manager.calibrate('both', 'calibration-reload-error-0001')
        manager._calibration_writer = lambda values: (_ for _ in ()).throw(
            gripper_control.GripperControlError('injected unreadable calibration')
        )
        slots = {
            side: manager._workers[f'calibration:{side}']
            for side in ('left', 'right')
        }
        for side, slot in slots.items():
            manager._apply_event_locked(
                {
                    'event': 'completed',
                    'owner_token': slot.owner_token,
                    'command_id': slot.inflight['command_id'],
                    'action': 'calibrate',
                    'side': side,
                    'results': [{
                        'side': side,
                        'target_percent': None,
                        'actual_percent': 50.0,
                        'actual_width_m': 0.04,
                        'calibration': {
                            'min_angle_deg': 100.0,
                            'max_angle_deg': 1100.0,
                        },
                    }],
                }
            )

        assert manager._hardware_unknown is False
        assert manager._sides['left']['status'] == 'error'
        assert manager._sides['right']['status'] == 'error'
        assert manager._sides['left']['last_completed_command_id'] is None
        assert manager._sides['right']['last_completed_command_id'] is None
    finally:
        manager.close()


def test_invalid_calibration_field_is_contained_without_half_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    manager, _ = _manager()
    try:
        manager.calibrate('both', 'calibration-invalid-field-0001')
        for side in ('left', 'right'):
            slot = manager._workers[f'calibration:{side}']
            manager._apply_event_locked(
                {
                    'event': 'completed',
                    'owner_token': slot.owner_token,
                    'command_id': slot.inflight['command_id'],
                    'action': 'calibrate',
                    'side': side,
                    'results': [{
                        'side': side,
                        'target_percent': None,
                        'actual_percent': 50.0,
                        'actual_width_m': 0.04,
                        'calibration': {
                            'min_angle_deg': (
                                100.0 if side == 'left' else 'not-a-number'
                            ),
                            'max_angle_deg': 1100.0,
                        },
                    }],
                }
            )

        assert manager._hardware_unknown is False
        assert manager._sides['left']['status'] == 'error'
        assert manager._sides['right']['status'] == 'error'
        assert manager._sides['left']['last_completed_command_id'] is None
        assert manager._sides['right']['last_completed_command_id'] is None
    finally:
        manager.close()


def test_missing_calibration_limits_never_publish_success_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    manager, _ = _manager()
    try:
        manager.calibrate('both', 'calibration-missing-limits-0001')
        for side in ('left', 'right'):
            slot = manager._workers[f'calibration:{side}']
            manager._apply_event_locked(
                {
                    'event': 'completed',
                    'owner_token': slot.owner_token,
                    'command_id': slot.inflight['command_id'],
                    'action': 'calibrate',
                    'side': side,
                    'results': [{
                        'side': side,
                        'target_percent': None,
                        'actual_percent': 50.0,
                        'actual_width_m': 0.04,
                        'calibration': (
                            {
                                'min_angle_deg': 100.0,
                                'max_angle_deg': 1100.0,
                            }
                            if side == 'left'
                            else {}
                        ),
                    }],
                }
            )

        assert manager._sides['left']['status'] == 'error'
        assert manager._sides['right']['status'] == 'error'
        assert manager._sides['left']['last_completed_command_id'] is None
        assert manager._sides['right']['last_completed_command_id'] is None
        assert manager._sides['left']['last_completed_action'] is None
        assert manager._sides['right']['last_completed_action'] is None
    finally:
        manager.close()


def test_stop_and_reconfigure_invalidate_deadband_cache() -> None:
    manager, _ = _manager()
    try:
        manager._last_targets = {'left': 20.0, 'right': 80.0}
        manager.stop('invalidate_deadband')
        assert manager._last_targets == {}

        manager._profiles = {side: _profile(side) for side in ('left', 'right')}
        manager._last_targets = {'left': 20.0, 'right': 80.0}
        manager.configure('empty', {})
        assert manager._last_targets == {}
    finally:
        manager.close()


def test_same_target_deadband_requires_measured_position_convergence() -> None:
    manager, _ = _manager()
    try:
        manager._last_targets['left'] = 0.0
        manager._sides['left']['actual_percent'] = 0.5
        aligned = manager.position(
            'left', 0.0, 'left-aligned-deadband', expected_side_revision=0
        )
        assert aligned['accepted'] is False
        assert aligned['status'] == 'deadband'
        assert manager._pending_positions == {}

        manager._sides['left']['actual_percent'] = 50.0
        off_target = manager.position(
            'left', 0.0, 'left-off-target-retry', expected_side_revision=0
        )
        assert off_target['accepted'] is True
        assert off_target['status'] == 'queued'
        assert manager._pending_positions['left']['target_percent'] == 0.0
    finally:
        manager.close()


def test_side_failure_invalidates_only_that_side_deadband_cache() -> None:
    manager, _ = _manager()
    try:
        manager._last_targets = {'left': 20.0, 'right': 80.0}
        manager.position(
            'left', 50.0, 'left-normal-failure-0001', expected_side_revision=0
        )
        manager._dispatch_locked()
        slot = manager._workers['left']
        manager._apply_event_locked(
            {
                'event': 'failed',
                'owner_token': slot.owner_token,
                'command_id': slot.inflight['command_id'],
                'action': 'position',
                'side': 'left',
                'error': 'injected side-local failure',
                'hardware_unknown': False,
            }
        )

        assert 'left' not in manager._last_targets
        assert manager._last_targets['right'] == 80.0
        assert manager._hardware_unknown is False
    finally:
        manager.close()


def test_calibration_acceptance_invalidates_both_deadband_caches() -> None:
    manager, _ = _manager()
    try:
        manager._last_targets = {'left': 20.0, 'right': 80.0}
        manager.calibrate('both', 'calibration-cache-reset-0001')
        assert manager._last_targets == {}
    finally:
        manager.close()


def test_completed_worker_processes_are_joined_and_removed() -> None:
    manager, context = _manager()
    try:
        for side, target in (('left', 25.0), ('right', 75.0)):
            manager.position(
                side,
                target,
                f'{side}-cleanup-completed-0001',
                expected_side_revision=0,
            )
        manager._dispatch_locked()

        for side, slot in tuple(manager._workers.items()):
            command_id = slot.inflight['command_id']
            manager._apply_event_locked(
                {
                    'event': 'completed',
                    'owner_token': slot.owner_token,
                    'command_id': command_id,
                    'action': 'position',
                    'side': side,
                    'results': [
                        {
                            'side': side,
                            'target_percent': 25.0 if side == 'left' else 75.0,
                            'actual_percent': 25.0 if side == 'left' else 75.0,
                            'actual_width_m': 0.02 if side == 'left' else 0.06,
                        }
                    ],
                }
            )
            slot.process.alive = False
            slot.process.exitcode = 0

        manager._refresh_locked()

        assert manager._workers == {}
        assert all(process.join_calls >= 1 for process in context.processes)
    finally:
        manager.close()


def test_close_never_swallows_a_worker_that_is_still_alive() -> None:
    manager, _ = _manager()
    manager._context = _StubbornContext()
    manager.position(
        'left', 25.0, 'left-stubborn-worker-0001', expected_side_revision=0
    )
    manager._dispatch_locked()

    with pytest.raises(gripper_control.GripperControlError, match='did not stop|unknown'):
        manager.close()

    assert manager._workers['left'].process.is_alive() is True


def test_closed_manager_rejects_new_position_and_calibration_commands() -> None:
    manager, _ = _manager()
    manager.close()

    with pytest.raises(
        gripper_control.GripperConflictError,
        match='closed|shutting down',
    ):
        manager.position(
            'left', 25.0, 'left-after-close-0001', expected_side_revision=0
        )
    with pytest.raises(
        gripper_control.GripperConflictError,
        match='closed|shutting down',
    ):
        manager.calibrate('both', 'calibration-after-close-0001')

    assert manager._workers == {}
    assert manager._pending_positions == {}


def test_serial_holder_scan_excludes_all_owned_worker_pids(
    tmp_path: Path,
) -> None:
    if os.name != 'posix':
        pytest.skip('Linux /proc character-device semantics are required')
    proc_root = tmp_path / 'proc'
    for pid in (4242, 4243, 4244):
        fd_root = proc_root / str(pid) / 'fd'
        fd_root.mkdir(parents=True)
        try:
            (fd_root / '7').symlink_to('/dev/null')
        except OSError as exc:  # pragma: no cover - non-POSIX test host
            pytest.skip(f'symlink support is required: {exc}')

    assert gripper_control.serial_port_holders(
        '/dev/null',
        exclude_pids=[4242, 4243],
        proc_root=proc_root,
    ) == [4244]
