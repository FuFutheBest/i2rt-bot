from __future__ import annotations

from types import SimpleNamespace
import signal
import sys
import threading

import pytest

from incubator.control_panel import server
from incubator.control_panel.gripper_control import GripperConflictError


class _Probe:
    def __init__(self, calls: list[str], name: str, *, fail: bool = False) -> None:
        self.calls = calls
        self.name = name
        self.fail = fail

    def _run(self) -> None:
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError(f'injected {self.name} failure')

    def close(self) -> None:
        self._run()

    def shutdown(self) -> None:
        self._run()

    def stop_all(self) -> None:
        self._run()


def test_shutdown_continues_other_cleanup_after_gripper_close_failure() -> None:
    calls: list[str] = []
    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._shutdown = False
    grippers = _Probe(calls, 'grippers', fail=True)
    controller.grippers = grippers
    controller._owns_conversion_jobs = True
    controller.conversion_jobs = _Probe(calls, 'conversion')
    controller._close_preview_readers_locked = lambda: calls.append('camera_preview')
    controller._close_robot_state_readers_locked = lambda: calls.append('robot_preview')
    controller.workflow = _Probe(calls, 'workflow')
    controller.starter = _Probe(calls, 'starter')
    controller.logs = SimpleNamespace(append=lambda *args, **kwargs: calls.append('log'))

    with pytest.raises(RuntimeError, match='grippers'):
        controller.shutdown()

    assert calls[:1] == ['grippers']
    assert {
        'conversion',
        'camera_preview',
        'robot_preview',
        'workflow',
        'starter',
    }.issubset(calls)
    assert controller.workflow is None
    assert controller.starter is None

    grippers.fail = False
    controller.shutdown()
    assert calls.count('grippers') == 2


class _StructuredStarter:
    def __init__(self, report: dict) -> None:
        self.report = report
        self.calls = 0

    def stop_all(self) -> dict:
        self.calls += 1
        return self.report


class _ForceStage:
    def __init__(
        self,
        calls: list[str],
        name: str,
        pid: int,
        *,
        fail: bool = False,
    ) -> None:
        self.calls = calls
        self.name = name
        self.pid = pid
        self.fail = fail

    def _run(self) -> dict:
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError(f'injected {self.name} failure')
        return {'ok': True, 'killed_pids': [self.pid]}

    def force_stop_all(self) -> dict:
        return self._run()

    def force_stop(self) -> dict:
        return self._run()


def test_force_stop_all_runs_every_owned_stage_and_preserves_server() -> None:
    calls: list[str] = []
    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._shutdown = False
    controller._force_stop_lock = threading.Lock()
    controller._force_stop_in_progress = False
    controller._workflow_cancel_event = threading.Event()
    controller.revision = 11
    controller.workflow = _ForceStage(calls, 'workflow', 101)
    controller.starter = _ForceStage(calls, 'devices', 102)
    controller.grippers = _ForceStage(calls, 'gripper_commands', 103)
    controller._close_preview_readers_locked = lambda: calls.append('preview')
    controller._close_robot_state_readers_locked = lambda: calls.append(
        'robot_preview'
    )
    controller.logs = SimpleNamespace(
        append=lambda *args, **kwargs: calls.append('log')
    )

    result = controller.force_stop_all()

    assert result['ok'] is True
    assert result['web_server_preserved'] is True
    assert result['killed_pids'] == [101, 102, 103]
    assert calls[:3] == [
        'workflow',
        'devices',
        'gripper_commands',
    ]
    assert controller.revision == 12
    assert controller._force_stop_in_progress is False


def test_force_stop_all_continues_after_one_stage_failure() -> None:
    calls: list[str] = []
    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._shutdown = False
    controller._force_stop_lock = threading.Lock()
    controller._force_stop_in_progress = False
    controller._workflow_cancel_event = threading.Event()
    controller.revision = 0
    controller.workflow = _ForceStage(calls, 'workflow', 101, fail=True)
    controller.starter = _ForceStage(calls, 'devices', 102)
    controller.grippers = _ForceStage(calls, 'gripper_commands', 103)
    controller._close_preview_readers_locked = lambda: None
    controller._close_robot_state_readers_locked = lambda: None
    controller.logs = SimpleNamespace(append=lambda *args, **kwargs: None)

    result = controller.force_stop_all()

    assert result['ok'] is False
    assert 'workflow' in result['errors']
    assert calls == [
        'workflow',
        'devices',
        'gripper_commands',
    ]
    assert result['killed_pids'] == [102, 103]
    assert controller._force_stop_in_progress is False


def _shutdown_controller(starter: object) -> server.DeviceController:
    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._shutdown = False
    controller._shutdown_complete = False
    controller.grippers = _Probe([], 'grippers')
    controller._owns_conversion_jobs = False
    controller._close_preview_readers_locked = lambda: None
    controller._close_robot_state_readers_locked = lambda: None
    controller.workflow = None
    controller.starter = starter
    controller.logs = SimpleNamespace(append=lambda *args, **kwargs: None)
    return controller


def test_shutdown_rejects_unresolved_owned_device_workers() -> None:
    starter = _StructuredStarter(
        {
            'ok': False,
            'incomplete_devices': ['RizonRobot:2', 'WebCameraOpenPi:0'],
            'external_devices': ['WebCameraOpenPi:0'],
        }
    )
    controller = _shutdown_controller(starter)

    with pytest.raises(
        server.DeviceControlError,
        match='Owned device shutdown incomplete: RizonRobot:2',
    ):
        controller.shutdown()

    assert starter.calls == 1
    assert controller.starter is starter
    assert controller._shutdown_complete is False


def test_shutdown_allows_external_only_devices_to_remain_running() -> None:
    starter = _StructuredStarter(
        {
            'ok': False,
            'incomplete_devices': ['RizonRobot:2'],
            'external_devices': ['RizonRobot:2'],
        }
    )
    controller = _shutdown_controller(starter)

    controller.shutdown()

    assert starter.calls == 1
    assert controller.starter is None
    assert controller._shutdown_complete is True


def test_sigterm_cleanup_failure_keeps_server_alive_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    installed_handlers: dict[int, object] = {}

    class FailingController:
        def __init__(self) -> None:
            self.logs = SimpleNamespace(
                append=lambda *args, **kwargs: calls.append('logged')
            )

        def shutdown(self) -> None:
            calls.append('shutdown')
            raise RuntimeError('owned motion worker is still active')

    class StateSync:
        def start(self) -> None:
            calls.append('state_sync_start')

        def close(self) -> None:
            calls.append('state_sync_close')

    controller = FailingController()
    state_sync = StateSync()

    class App:
        config = {
            'DEVICE_CONTROLLER': controller,
            'STATE_SYNC': state_sync,
        }

        @staticmethod
        def run(**kwargs) -> None:
            del kwargs
            installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
            calls.append('server_resumed')

    class Lock:
        def __init__(self, path) -> None:
            del path

        def acquire(self) -> None:
            calls.append('lock_acquired')

        def release(self) -> None:
            calls.append('lock_released')

    monkeypatch.setattr(server, 'create_app', lambda **kwargs: App())
    monkeypatch.setattr(server, 'SingleInstanceLock', Lock)
    monkeypatch.setattr(server.atexit, 'register', lambda callback: None)
    monkeypatch.setattr(
        server.signal,
        'signal',
        lambda signum, handler: installed_handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(sys, 'argv', ['control-panel', '--no-process-cleanup'])

    server.main()

    assert 'server_resumed' in calls
    assert calls.count('shutdown') == 1
    assert 'logged' in calls
    assert 'lock_released' not in calls
    assert callable(installed_handlers[signal.SIGTERM])


class _ControlGrippers:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def replay(self, *args, **kwargs):
        del args, kwargs
        return None

    def stop(self, reason: str):
        self.calls.append(('stop', reason))
        return {'ok': True, 'stopped': True, 'status': 'idle'}

    def calibrate(self, side: str, key: str):
        self.calls.append(('calibrate', side, key))
        return {
            'accepted': True,
            'duplicate': False,
            'command_id': 'calibration-command',
            'status': 'pending',
        }

    def position(
        self,
        side: str,
        target: float,
        key: str,
        *,
        expected_side_revision: int,
    ):
        self.calls.append(
            ('position', side, target, key, expected_side_revision)
        )
        return {
            'accepted': True,
            'duplicate': False,
            'command_id': 'position-command',
            'status': 'queued',
        }


def _control_controller(state: dict) -> tuple[server.DeviceController, _ControlGrippers]:
    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._shutdown = False
    controller.revision = 7
    grippers = _ControlGrippers()
    controller.grippers = grippers
    controller._gripper_state_locked = lambda: state
    controller.logs = SimpleNamespace(append=lambda *args, **kwargs: None)
    return controller, grippers


def test_safety_stop_is_not_rejected_by_stale_ui_revision() -> None:
    state = {
        'can_calibrate': True,
        'blockers': [],
        'sides': {},
    }
    controller, grippers = _control_controller(state)

    result = controller.control_gripper('stop', expected_revision=6)

    assert grippers.calls == [('stop', 'api_stop')]
    assert result['revision'] == 8


def test_single_side_calibration_uses_requested_side_capability() -> None:
    state = {
        'can_calibrate': False,
        'blockers': ['right gripper is unavailable'],
        'sides': {
            'left': {'can_calibrate': True, 'blockers': []},
            'right': {
                'can_calibrate': False,
                'blockers': ['right gripper is unavailable'],
            },
        },
    }
    controller, grippers = _control_controller(state)

    result = controller.control_gripper(
        'calibrate',
        expected_revision=7,
        side='left',
        idempotency_key='left-calibration-0001',
    )

    assert grippers.calls == [
        ('calibrate', 'left', 'left-calibration-0001')
    ]
    assert result['revision'] == 8


class _StateGrippers:
    def configured_ports(self):
        return {'left': '/dev/ttyUSB1', 'right': '/dev/ttyUSB0'}

    def owned_pids(self):
        return [111, 222]

    def state(self, blockers=()):
        assert list(blockers) == []
        sides = {}
        for side in ('left', 'right'):
            sides[side] = {
                'can_calibrate': True,
                'can_position': True,
                'blockers': [],
                'calibration_blockers': [],
                'position_blockers': [],
                'capabilities': {
                    'calibrate': {'can': True, 'blockers': []},
                    'position': {'can': True, 'blockers': []},
                },
            }
        return {
            'available': True,
            'active': False,
            'busy': False,
            'can_calibrate': True,
            'can_position': True,
            'sides': sides,
        }


def test_serial_holder_blocks_only_affected_side_and_excludes_all_owned_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def holders(ports, *, exclude_pids=()):
        calls.append((tuple(ports), tuple(exclude_pids)))
        return {
            port: ([999] if port == '/dev/ttyUSB1' else [])
            for port in ports
        }

    monkeypatch.setattr(server, 'serial_port_holders_many', holders)
    monkeypatch.setattr(
        server,
        'process_command',
        lambda pid: 'python incubator/control_panel/gripper_control.py',
    )
    monkeypatch.setattr(server, 'rizon_shared_memory_names', lambda: [])
    controller = object.__new__(server.DeviceController)
    controller.selected_config = 'selected'
    controller._effective_config = {'devices': []}
    controller.starter = None
    controller.workflow = None
    controller.grippers = _StateGrippers()
    controller._hardware_permission_state_locked = lambda targets: {
        'required': False,
        'affected_ports': [],
    }

    state = controller._gripper_state_locked(devices=[], workflow_state={})

    assert calls == [
        (('/dev/ttyUSB1', '/dev/ttyUSB0'), (111, 222)),
    ]
    assert state['sides']['left']['can_position'] is False
    assert state['sides']['left']['can_calibrate'] is False
    assert 'held by another process' in ' '.join(
        state['sides']['left']['blockers']
    )
    assert (
        'PID 999 (python incubator/control_panel/gripper_control.py)'
        in ' '.join(
            state['sides']['left']['blockers']
        )
    )
    assert state['sides']['right']['can_position'] is True
    assert state['sides']['right']['can_calibrate'] is True
    assert state['can_position'] is True
    assert state['can_calibrate'] is False
    assert state['can_bind_ports'] is False


def test_running_cameras_do_not_block_grippers_but_motion_devices_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, 'rizon_shared_memory_names', lambda: [])
    controller = object.__new__(server.DeviceController)
    controller.selected_config = 'selected'
    controller._effective_config = {'devices': []}
    controller.starter = None
    controller.workflow = None
    camera = {
        'key': 'WebCameraOpenPi:0',
        'class': 'WebCameraOpenPi',
        'status': 'running',
        'running': True,
        'desired_state': 'running',
        'risk': None,
    }
    rizon = {
        'key': 'RizonRobot:2',
        'class': 'RizonRobot',
        'status': 'running',
        'running': True,
        'desired_state': 'running',
        'risk': 'motion',
    }

    assert controller._gripper_blockers_locked(
        devices=[camera], workflow_state={}
    ) == []
    assert controller._gripper_blockers_locked(
        devices=[camera, rizon], workflow_state={}
    ) == [
        'Stop motion devices before controlling grippers: RizonRobot:2'
    ]


def test_shutdown_controller_rejects_new_motion_commands() -> None:
    state = {
        'can_calibrate': True,
        'blockers': [],
        'sides': {
            'left': {'can_position': True, 'blockers': []},
        },
    }
    controller, grippers = _control_controller(state)
    controller._shutdown = True

    with pytest.raises(GripperConflictError, match='shutting down|stopped'):
        controller.control_gripper(
            'position',
            expected_revision=7,
            expected_side_revision=0,
            side='left',
            target_percent=25.0,
            idempotency_key='position-after-shutdown-0001',
        )

    assert grippers.calls == []
