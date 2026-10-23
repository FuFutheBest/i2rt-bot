"""Controller locking and public-log boundary regression tests."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger as global_logger
import pytest

from incubator.control_panel import server
from incubator.control_panel.hardware_discovery import HardwareSnapshot
from incubator.control_panel.lifecycle import DeviceConflictError
from incubator.control_panel.workflow import WorkflowConflictError


def test_panel_yam_disconnect_tolerance_override_is_runtime_only() -> None:
    source = {
        'devices': [
            {
                'class': 'YamHeadRobot',
                'device_class_id': 0,
                'disconnect_zero_joint_count': 6,
                'disconnect_zero_tolerance_rad': 0.005,
            },
            {
                'class': 'YamHeadRobot',
                'device_class_id': 1,
                'disconnect_zero_tolerance_rad': 0.05,
            },
            {'class': 'BaseDevice', 'device_class_id': 7},
        ]
    }

    runtime = server.DeviceController._apply_control_panel_safety_overrides(
        source
    )

    assert source['devices'][0]['disconnect_zero_joint_count'] == 6
    assert source['devices'][0]['disconnect_zero_tolerance_rad'] == 0.005
    assert runtime['devices'][0]['disconnect_zero_joint_count'] == 5
    assert runtime['devices'][0]['disconnect_zero_tolerance_rad'] == 0.02
    assert runtime['devices'][1]['disconnect_zero_joint_count'] == 5
    assert runtime['devices'][1]['disconnect_zero_tolerance_rad'] == 0.02
    assert 'disconnect_zero_tolerance_rad' not in runtime['devices'][2]


def test_fastwam_server_reservation_is_independent_from_runtime() -> None:
    controller = server.DeviceController.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._device_operation_condition = threading.Condition(controller._lock)
    controller._device_operations = {}
    controller._workflow_cancelled_device_keys = set()
    controller._workflow_cancel_event = threading.Event()
    controller._lifecycle_mutation_pending = False
    controller._force_stop_lock = threading.Lock()
    controller._force_stop_in_progress = False

    with controller._lock:
        controller._begin_workflow_operation_locked('runtime:start')
        controller._begin_workflow_operation_locked('server:start')
        assert controller._workflow_operation_active_locked('runtime:start')
        assert controller._workflow_operation_active_locked('server:start')

        controller._end_workflow_operation_locked('server:start')
        assert controller._workflow_operation_active_locked('runtime:start')
        assert not controller._workflow_operation_active_locked('server:start')
        controller._end_workflow_operation_locked('runtime:start')


def test_server_stop_signal_does_not_cancel_runtime_start() -> None:
    controller = server.DeviceController.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._workflow_cancel_event = threading.Event()

    controller.request_async_stop('workflow:server')

    assert not controller._workflow_cancel_event.is_set()


@pytest.mark.parametrize('flat_backend', ['lktech', ''])
def test_runtime_gripper_migration_drops_flat_legacy_server_on_private_copy(
    flat_backend: str,
) -> None:
    source = {
        'devices': [
            {
                'class': 'RizonRobot',
                'device_class_id': 1,
                'gripper_backend': flat_backend,
                'gripper_server': {'host': '127.0.0.1', 'port': 8765},
            }
        ]
    }

    runtime = server.DeviceController._direct_gripper_config(source)

    device = runtime['devices'][0]
    assert device['gripper_backend'] == 'lktech'
    assert 'gripper_server' not in device
    assert source['devices'][0]['gripper_backend'] == flat_backend
    assert 'gripper_server' in source['devices'][0]


def _write_config(config_dir: Path, name: str) -> None:
    config_path = config_dir / f'{name}.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """# @package _global_
devices:
  - class: BaseDevice
    device_class_id: 7
    data_shape: [2]
    data_dtype: float32
    fps: 10
""",
        encoding='utf-8',
    )


def test_import_has_no_root_logger_or_hardware_registry_side_effects(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import json
import sys
from pathlib import Path
import incubator.control_panel.server
blocked = ['utils.logger_config', 'devices', 'manager', 'visualizers']
print(json.dumps({
    'loaded': [name for name in blocked if name in sys.modules],
    'logs_created': (Path.cwd() / 'logs').exists(),
}))
"""
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(repo_root)
    completed = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip())
    assert result == {'loaded': [], 'logs_created': False}


def test_workflow_api_does_not_start_devices_for_invalid_runtime_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'launch')
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'launch'}).status_code == 200
    state = client.get('/api/state').get_json()
    assert state['workflow']['stop_order'] == [
        'recording',
        'inference',
        'action_executor',
        'device_manager',
        'devices',
    ]

    runtime = client.post('/api/workflow/runtime/start')
    inference = client.post(
        '/api/workflow/inference/start',
        json={
            'expected_selected_config': state['selected_config'],
            'expected_revision': state['revision'],
            'expected_generation': state['checkpoint']['generation'],
            'expected_options_generation': state['inference_options']['generation'],
        },
    )
    assert runtime.status_code == 409
    assert '当前配置未启用 DeviceManager' in runtime.get_json()['error']
    assert inference.status_code == 409
    assert '请先正确启动' in inference.get_json()['error']
    after = client.get('/api/state').get_json()
    assert all(device['status'] == 'stopped' for device in after['devices'])
    app.config['DEVICE_CONTROLLER'].shutdown()


class BlockingLifecycle:
    """Hardware-free lifecycle double exposing deterministic action barriers."""

    instances: List['BlockingLifecycle'] = []

    def __init__(self, config: Dict[str, Any], **_: Any) -> None:
        self.config = config
        self.running = False
        self.stopped = False
        self.start_entered = threading.Event()
        self.release_start = threading.Event()
        self.block_start = False
        type(self).instances.append(self)

    def start_monitor(self) -> None:
        return None

    def has_active_devices(self) -> bool:
        return self.running

    def start_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        if self.block_start:
            self.start_entered.set()
            assert self.release_start.wait(timeout=5.0)
        self.running = True
        return self._device(device_class, device_class_id)

    def stop_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        self.running = False
        return self._device(device_class, device_class_id)

    def restart_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return self.start_device(device_class, device_class_id)

    def stop_all(self) -> Dict[str, Any]:
        stopped_devices = ['BaseDevice:7'] if self.running else []
        self.running = False
        self.stopped = True
        return {
            'ok': True,
            'devices': [self._device('BaseDevice', 7)],
            'stopped_devices': stopped_devices,
            'incomplete_devices': [],
            'external_devices': [],
        }

    def get_device_status(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return self._device(device_class, device_class_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            'running': self.running,
            'monitor_running': True,
            'total_devices': 1,
            'enable_visualizers': False,
            'enable_device_manager': False,
            'enable_policy_runner': False,
            'device_manager_running': False,
            'policy_runner_running': False,
            'devices': [self._device('BaseDevice', 7)],
        }

    def _device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return {
            'key': f'{device_class}:{device_class_id}',
            'class': device_class,
            'id': device_class_id,
            'running': self.running,
            'status': 'running' if self.running else 'stopped',
            'desired_state': 'running' if self.running else 'stopped',
            'capabilities': {'preview': False},
        }


class StaticDiscovery:
    """A deterministic passive-discovery double used by controller tests."""

    def __init__(self, snapshot: HardwareSnapshot | None = None) -> None:
        self.snapshot = snapshot or HardwareSnapshot(
            revision='empty-hardware-revision',
            generated_at='2026-07-12T00:00:00+00:00',
            interfaces=[],
            cameras=[],
            robots=[],
            providers=[],
            warnings=[],
        )

    def discover(self) -> HardwareSnapshot:
        return self.snapshot


class RuntimePreparationLifecycle:
    """Hardware-free required-device start/restart orchestration double."""

    def __init__(
        self,
        states: Dict[str, Dict[str, Any]],
        *,
        delayed_ready: set[str] | None = None,
    ) -> None:
        self._states = copy.deepcopy(states)
        self._lock = threading.RLock()
        self.delayed_ready = set(delayed_ready or ())
        self.actions: List[tuple[str, str]] = []
        self.action_entered = threading.Event()

    def _device(self, key: str) -> Dict[str, Any]:
        device_class, device_class_id = key.rsplit(':', 1)
        state = self._states[key]
        return {
            'key': key,
            'class': device_class,
            'id': int(device_class_id) if device_class_id.isdigit() else device_class_id,
            'status': state['status'],
            'running': state.get('running', False),
            'ready': state.get('ready', False),
            'desired_state': state.get('desired_state', 'stopped'),
            'last_error': state.get('last_error'),
            'safety': copy.deepcopy(state.get('safety', {})),
            'capabilities': {'preview': 'Camera' in device_class},
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            devices = [self._device(key) for key in self._states]
            return {'devices': devices, 'running': any(d['running'] for d in devices)}

    def get_device_status(
        self,
        device_class: str,
        device_class_id: Any,
    ) -> Dict[str, Any]:
        with self._lock:
            return self._device(f'{device_class}:{device_class_id}')

    def _activate(self, action: str, key: str) -> Dict[str, Any]:
        with self._lock:
            self.actions.append((action, key))
            self.action_entered.set()
            state = self._states[key]
            state.update(
                {
                    'status': 'starting' if key in self.delayed_ready else 'running',
                    'running': True,
                    'ready': key not in self.delayed_ready,
                    'desired_state': 'running',
                    'last_error': None,
                }
            )
            return self._device(key)

    def start_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return self._activate('start', f'{device_class}:{device_class_id}')

    def restart_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return self._activate('restart', f'{device_class}:{device_class_id}')

    def stop_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        key = f'{device_class}:{device_class_id}'
        with self._lock:
            self.actions.append(('stop', key))
            self._states[key].update(
                {
                    'status': 'stopped',
                    'running': False,
                    'ready': False,
                    'desired_state': 'stopped',
                }
            )
            return self._device(key)

    def mark_ready(self, key: str) -> None:
        with self._lock:
            self._states[key].update(
                {'status': 'running', 'running': True, 'ready': True}
            )

    def stop_all(self) -> Dict[str, Any]:
        with self._lock:
            for state in self._states.values():
                state.update(
                    {
                        'status': 'stopped',
                        'running': False,
                        'ready': False,
                        'desired_state': 'stopped',
                    }
                )
            return {'ok': True, 'devices': self.get_status()['devices']}


class RuntimePreparationWorkflow:
    def __init__(self, required_keys: set[str]) -> None:
        self._required_keys = set(required_keys)
        self.start_calls: List[List[Dict[str, Any]]] = []
        self.active = False
        self.start_entered = threading.Event()
        self.release_start = threading.Event()
        self.block_start = False
        self.stop_calls = 0

    def required_device_keys(self) -> set[str]:
        return set(self._required_keys)

    def runtime_preparation_blockers(
        self,
        devices: List[Dict[str, Any]],
    ) -> List[str]:
        del devices
        return []

    def start_runtime(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_key = {device['key']: device for device in devices}
        assert all(by_key[key]['ready'] is True for key in self._required_keys)
        self.start_calls.append(copy.deepcopy(devices))
        if self.block_start:
            self.start_entered.set()
            assert self.release_start.wait(timeout=5.0)
        self.active = True
        self.start_entered.set()
        return {
            'runtime': {'status': 'running', 'active': True, 'ready': True},
            'inference': {'status': 'stopped', 'running': False},
        }

    def has_active_processes(self) -> bool:
        return self.active

    def stop_all(self) -> None:
        self.stop_calls += 1
        self.active = False


def _runtime_preparation_controller(
    tmp_path: Path,
    starter: RuntimePreparationLifecycle,
    workflow: RuntimePreparationWorkflow,
) -> server.DeviceController:
    controller = server.DeviceController(config_dir=tmp_path)
    controller.selected_config = 'fixture'
    controller.starter = starter  # type: ignore[assignment]
    controller.workflow = workflow  # type: ignore[assignment]
    return controller


def test_runtime_start_prepares_required_devices_and_skips_optional_head(
    tmp_path: Path,
) -> None:
    head = 'WebCameraOpenPi:2'
    left = 'WebCameraOpenPi:0'
    right = 'WebCameraOpenPi:4'
    robot = 'RizonRobot:2'
    starter = RuntimePreparationLifecycle(
        {
            head: {'status': 'stopped'},
            left: {'status': 'stopped'},
            right: {
                'status': 'running',
                'running': True,
                'ready': True,
                'desired_state': 'running',
            },
            robot: {
                'status': 'degraded',
                'running': True,
                'ready': False,
                'desired_state': 'running',
                'last_error': 'health heartbeat stale',
            },
        }
    )
    workflow = RuntimePreparationWorkflow({left, right, robot})
    controller = _runtime_preparation_controller(tmp_path, starter, workflow)
    try:
        result = controller.control_workflow('runtime', 'start')

        assert set(starter.actions) == {('start', left), ('restart', robot)}
        assert all(key != head for _, key in starter.actions)
        assert len(workflow.start_calls) == 1
        assert result['workflow']['runtime']['ready'] is True
        assert {
            (item['action'], item['key']) for item in result['device_actions']
        } == {('start', left), ('restart', robot)}
    finally:
        controller.shutdown()


def test_runtime_device_wait_does_not_hold_controller_state_lock(
    tmp_path: Path,
) -> None:
    robot = 'RizonRobot:2'
    starter = RuntimePreparationLifecycle(
        {robot: {'status': 'stopped'}},
        delayed_ready={robot},
    )
    workflow = RuntimePreparationWorkflow({robot})
    controller = _runtime_preparation_controller(tmp_path, starter, workflow)
    errors: List[BaseException] = []

    def launch() -> None:
        try:
            controller.control_workflow('runtime', 'start')
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=launch)
    thread.start()
    try:
        assert starter.action_entered.wait(timeout=2.0)
        acquired = controller._lock.acquire(timeout=0.5)
        assert acquired is True
        if acquired:
            controller._lock.release()
        starter.mark_ready(robot)
        thread.join(timeout=2.0)
        assert thread.is_alive() is False
        assert errors == []
        assert len(workflow.start_calls) == 1
    finally:
        starter.mark_ready(robot)
        thread.join(timeout=2.0)
        controller.shutdown()


def test_device_stop_cancels_runtime_start_and_rolls_back_started_workers(
    tmp_path: Path,
) -> None:
    robot = 'RizonRobot:2'
    starter = RuntimePreparationLifecycle(
        {
            robot: {
                'status': 'running',
                'running': True,
                'ready': True,
                'desired_state': 'running',
            }
        }
    )
    workflow = RuntimePreparationWorkflow({robot})
    workflow.block_start = True
    controller = _runtime_preparation_controller(tmp_path, starter, workflow)
    errors: List[BaseException] = []

    def launch() -> None:
        try:
            controller.control_workflow('runtime', 'start')
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=launch)
    thread.start()
    try:
        assert workflow.start_entered.wait(timeout=2.0)
        stopped = controller.control_device('stop', 'RizonRobot', 2)
        assert stopped['device']['status'] == 'stopped'
    finally:
        workflow.release_start.set()
        thread.join(timeout=2.0)
        rollback_calls = workflow.stop_calls
        controller.shutdown()

    assert thread.is_alive() is False
    assert len(errors) == 1
    assert isinstance(errors[0], WorkflowConflictError)
    assert 'RizonRobot:2' in str(errors[0])
    assert rollback_calls == 1
    assert workflow.active is False
    assert starter.get_device_status('RizonRobot', 2)['status'] == 'stopped'


@pytest.mark.parametrize(
    'safety',
    [
        {'known': True, 'estop': True, 'fail_closed': True},
        {'known': True, 'operational': False, 'fail_closed': True},
        {'known': True, 'reduced': True, 'fail_closed': True},
    ],
)
def test_runtime_never_auto_restarts_motion_device_under_safety_interlock(
    tmp_path: Path,
    safety: Dict[str, Any],
) -> None:
    robot = 'RizonRobot:2'
    starter = RuntimePreparationLifecycle(
        {
            robot: {
                'status': 'degraded',
                'running': True,
                'ready': False,
                'desired_state': 'running',
                'last_error': 'robot safety state changed',
                'safety': safety,
            }
        }
    )
    workflow = RuntimePreparationWorkflow({robot})
    controller = _runtime_preparation_controller(tmp_path, starter, workflow)
    try:
        with pytest.raises(WorkflowConflictError, match='安全互锁'):
            controller.control_workflow('runtime', 'start')
    finally:
        controller.shutdown()

    assert starter.actions == []
    assert workflow.start_calls == []


def test_runtime_allows_motion_device_initialization_safety_window(
    tmp_path: Path,
) -> None:
    robot = 'RizonRobot:2'
    starter = RuntimePreparationLifecycle(
        {
            robot: {
                'status': 'starting',
                'running': True,
                'ready': False,
                'desired_state': 'running',
                'safety': {
                    'known': False,
                    'fail_closed': True,
                    'initializing': True,
                },
            }
        },
        delayed_ready={robot},
    )
    workflow = RuntimePreparationWorkflow({robot})
    controller = _runtime_preparation_controller(tmp_path, starter, workflow)
    errors: List[BaseException] = []

    def launch() -> None:
        try:
            controller.control_workflow('runtime', 'start')
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=launch)
    thread.start()
    try:
        threading.Event().wait(0.05)
        starter.mark_ready(robot)
        assert workflow.start_entered.wait(timeout=2.0)
        thread.join(timeout=2.0)
    finally:
        starter.mark_ready(robot)
        thread.join(timeout=2.0)
        controller.shutdown()

    assert errors == []
    assert len(workflow.start_calls) == 1


def test_slow_workflow_transition_keeps_state_and_emergency_stop_responsive(
    tmp_path: Path,
) -> None:
    stop_entered = threading.Event()
    release_stop = threading.Event()
    mutation_entered = threading.Event()
    workflow_errors: List[BaseException] = []

    checkpoint = {
        'default_path': None,
        'override_path': None,
        'effective_path': None,
        'override_active': False,
        'exists': False,
        'params_exists': False,
        'valid': False,
        'error': None,
    }
    snapshot = {
        'config_name': 'fixture',
        'runtime': {
            'status': 'running',
            'active': True,
            'ready': True,
            'can_start': False,
            'can_stop': True,
            'blockers': [],
        },
        'inference': {
            'status': 'running',
            'running': True,
            'can_start': False,
            'can_stop': True,
            'blockers': [],
            'checkpoint': checkpoint,
        },
        'profiles': {},
        'recording': {'status': 'armed', 'active': False},
        'stop_order': [],
    }

    class BlockingWorkflowTransition:
        def observe(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
            del devices
            return copy.deepcopy(snapshot)

        def checkpoint_state(self) -> Dict[str, Any]:
            raise AssertionError('controller state must reuse observed checkpoint')

        def stop_inference(self, devices: List[Dict[str, Any]]) -> Dict[str, Any]:
            del devices
            stop_entered.set()
            assert release_stop.wait(timeout=5.0)
            return copy.deepcopy(snapshot)

        def has_active_processes(self) -> bool:
            return True

        def stop_all(self) -> None:
            return None

    class ConversionState:
        def state(self) -> Dict[str, Any]:
            return {'configured': False, 'available': False, 'jobs': []}

    starter = BlockingLifecycle({'devices': []})
    workflow = BlockingWorkflowTransition()
    controller = server.DeviceController(
        config_dir=tmp_path,
        conversion_jobs=ConversionState(),  # type: ignore[arg-type]
    )
    controller.selected_config = 'fixture'
    controller.starter = starter  # type: ignore[assignment]
    controller.workflow = workflow  # type: ignore[assignment]

    def stop_inference() -> None:
        try:
            controller.control_workflow('inference', 'stop')
        except BaseException as exc:
            workflow_errors.append(exc)

    def lifecycle_mutation() -> None:
        with controller._lock, controller._lifecycle_mutation_guard_locked():
            mutation_entered.set()

    workflow_thread = threading.Thread(target=stop_inference)
    mutation_thread = threading.Thread(target=lifecycle_mutation)
    workflow_thread.start()
    try:
        assert stop_entered.wait(timeout=2.0)
        started = time.perf_counter()
        state = controller.state()
        assert time.perf_counter() - started < 0.5
        assert state['selected_config'] == 'fixture'

        with pytest.raises(DeviceConflictError, match='Workflow action'):
            controller.control_device('start', 'BaseDevice', 7)

        started = time.perf_counter()
        stopped = controller.control_device('stop', 'BaseDevice', 7)
        assert time.perf_counter() - started < 0.5
        assert stopped['device']['status'] == 'stopped'

        mutation_thread.start()
        assert mutation_entered.wait(timeout=0.1) is False
    finally:
        release_stop.set()
        workflow_thread.join(timeout=2.0)
        if mutation_thread.ident is not None:
            mutation_thread.join(timeout=2.0)
        controller.shutdown()

    assert workflow_thread.is_alive() is False
    assert mutation_thread.ident is not None
    assert mutation_thread.is_alive() is False
    assert mutation_entered.is_set()
    assert workflow_errors == []


class SwitchLifecycle:
    """Multi-device lifecycle double implementing the stop-all contract."""

    instances: List['SwitchLifecycle'] = []
    incomplete_stop_keys: set[str] = set()

    def __init__(self, config: Dict[str, Any], **_: Any) -> None:
        self.config = config
        self.stopped = False
        self.block_start = False
        self.start_entered = threading.Event()
        self.release_start = threading.Event()
        self._states: Dict[str, str] = {
            f"{device['class']}:{device['device_class_id']}": 'stopped'
            for device in config.get('devices', [])
            if device.get('enabled', True) is not False
        }
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.incomplete_stop_keys = set()

    def start_monitor(self) -> None:
        return None

    def has_active_devices(self) -> bool:
        return any(
            status in {'running', 'stop_timeout', 'external', 'orphaned'}
            for status in self._states.values()
        )

    def start_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        key = f'{device_class}:{device_class_id}'
        if self.block_start:
            self.start_entered.set()
            assert self.release_start.wait(timeout=5.0)
        self._states[key] = 'running'
        return self._device(key)

    def stop_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        key = f'{device_class}:{device_class_id}'
        self._states[key] = 'stopped'
        return self._device(key)

    def restart_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return self.start_device(device_class, device_class_id)

    def mark_status(self, key: str, status: str) -> None:
        assert key in self._states
        self._states[key] = status

    def stop_all(self) -> Dict[str, Any]:
        stopped_devices: List[str] = []
        incomplete_devices: List[str] = []
        external_devices: List[str] = []
        for key, status in list(self._states.items()):
            if status in {'external', 'orphaned'}:
                external_devices.append(key)
            elif status in {'running', 'stop_timeout'}:
                if key in type(self).incomplete_stop_keys:
                    self._states[key] = 'stop_timeout'
                    incomplete_devices.append(key)
                else:
                    self._states[key] = 'stopped'
                    stopped_devices.append(key)
        self.stopped = not incomplete_devices and not external_devices
        return {
            'ok': self.stopped,
            'devices': [self._device(key) for key in self._states],
            'stopped_devices': stopped_devices,
            'incomplete_devices': incomplete_devices,
            'external_devices': external_devices,
        }

    def get_device_status(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        return self._device(f'{device_class}:{device_class_id}')

    def get_status(self) -> Dict[str, Any]:
        devices = [self._device(key) for key in self._states]
        return {
            'running': any(device['running'] for device in devices),
            'monitor_running': True,
            'total_devices': len(devices),
            'enable_visualizers': False,
            'enable_device_manager': False,
            'enable_policy_runner': False,
            'device_manager_running': False,
            'policy_runner_running': False,
            'devices': devices,
        }

    def _device(self, key: str) -> Dict[str, Any]:
        device_class, device_class_id = key.rsplit(':', 1)
        status = self._states[key]
        running = status in {'running', 'stop_timeout'}
        return {
            'key': key,
            'class': device_class,
            'id': int(device_class_id) if device_class_id.isdigit() else device_class_id,
            'running': running,
            'status': status,
            'desired_state': 'running' if status == 'running' else 'stopped',
            'capabilities': {'preview': False},
        }


def _install_switch_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    SwitchLifecycle.reset()
    monkeypatch.setattr(server, 'DeviceLifecycleManager', SwitchLifecycle)
    monkeypatch.setattr(server, 'HardwareDiscovery', StaticDiscovery)


def _write_multi_device_config(config_dir: Path, name: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f'{name}.yaml').write_text(
        """devices:
  - class: BaseDevice
    device_class_id: 7
    data_shape: [2]
    data_dtype: float32
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-000001, Rizon4R-000002]
""",
        encoding='utf-8',
    )


def _write_camera_config(config_dir: Path, name: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f'{name}.yaml').write_text(
        """devices:
  - class: WebCameraOpenPi
    device_class_id: 4
    capture_id: 99
    data_shape: [480, 640, 3]
    data_dtype: uint8
    fps: 30
""",
        encoding='utf-8',
    )


def _write_robot_config(config_dir: Path, name: str) -> None:
    config_path = config_dir / f'{name}.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """# @package _global_
devices:
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-111111, Rizon4R-222222]
    network_interface_whitelist: [192.0.2.99]
    data_shape: [2, 7]
    data_dtype: float64
    fps: 100
""",
        encoding='utf-8',
    )


@pytest.mark.parametrize(
    ('names', 'expected'),
    [
        (
            [
                'aaa_dual_pi05',
                server.DEFAULT_RUNTIME_CONFIG_NAME,
                'config',
            ],
            server.DEFAULT_RUNTIME_CONFIG_NAME,
        ),
        (
            ['z_dual_pi05', 'A_DUAL_PI05', 'config'],
            'config',
        ),
        (['zeta', 'config', 'alpha'], 'config'),
        (['zeta', 'Alpha', 'alpha'], 'Alpha'),
        ([], None),
    ],
)
def test_recommended_runtime_config_has_deterministic_fallbacks(
    names: List[str], expected: str | None
) -> None:
    assert server.recommended_runtime_config(names) == expected


def test_configs_api_marks_the_recommended_bootstrap_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'config')
    _write_config(config_dir, 'other_dual_pi05')
    _write_config(config_dir, server.DEFAULT_RUNTIME_CONFIG_NAME)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()

    response = client.get('/api/configs')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['recommended_config'] == server.DEFAULT_RUNTIME_CONFIG_NAME
    defaults = [item['name'] for item in payload['items'] if item['default']]
    assert defaults == [server.DEFAULT_RUNTIME_CONFIG_NAME]
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_configs_api_hides_workflows_not_supported_by_control_panel(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'device_only')
    (config_dir / 'openpi.yaml').write_text(
        """devices:
  - class: BaseDevice
    device_class_id: 7
policy:
  class: OpenPiPolicy
""",
        encoding='utf-8',
    )
    (config_dir / 'lingbot.yaml').write_text(
        """devices:
  - class: BaseDevice
    device_class_id: 7
policy:
  class: LingBotVAPolicy
""",
        encoding='utf-8',
    )
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
    )
    app.testing = True

    payload = app.test_client().get('/api/configs').get_json()

    assert payload['configs'] == ['device_only', 'openpi']
    openpi = next(item for item in payload['items'] if item['name'] == 'openpi')
    assert openpi['workflow_kind'] == 'openpi'
    assert openpi['policy_class'] == 'OpenPiPolicy'
    assert openpi['hardware_required'] is True
    assert openpi['status'] == 'available'
    assert openpi['control_panel_supported'] is True
    assert all(item['name'] != 'lingbot' for item in payload['items'])
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_auto_apply_uses_recommended_hardware_bindings_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_robot_config(config_dir, server.DEFAULT_RUNTIME_CONFIG_NAME)
    snapshot = HardwareSnapshot(
        revision='robot-hardware-revision',
        generated_at='2026-07-12T00:00:00+00:00',
        interfaces=[
            {
                'id': 'interface_robot_lan',
                'name': 'enp3s0',
                'ipv4': '10.42.0.71',
                'prefixlen': 24,
                'network': '10.42.0.0/24',
                'operstate': 'up',
                'robot_facing_candidate': True,
                'source': 'linux_interfaces',
            }
        ],
        cameras=[],
        robots=[
            {
                'id': 'robot_left_detected',
                'model': 'Rizon4',
                'variant': 'standard',
                'rdk_sn': 'Rizon4-333333',
                'serial_number': 'Rizon4-333333',
                'ip': '10.42.0.10',
                'source': 'elements_cache',
            },
            {
                'id': 'robot_right_detected',
                'model': 'Rizon4R',
                'variant': '4r',
                'rdk_sn': 'Rizon4R-444444',
                'serial_number': 'Rizon4R-444444',
                'ip': '10.42.0.11',
                'source': 'elements_cache',
            },
        ],
        providers=[],
        warnings=[],
    )
    SwitchLifecycle.reset()
    monkeypatch.setattr(server, 'DeviceLifecycleManager', SwitchLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(snapshot),
    )
    app.testing = True
    client = app.test_client()

    state = client.get('/api/state').get_json()
    configs = client.get('/api/configs').get_json()
    target = configs['recommended_config']
    assert client.get(f'/api/configs/{target}').status_code == 200
    discovery = client.get(
        f'/api/hardware/discovery?config_name={target}'
    ).get_json()

    selected = client.post(
        '/api/config/select',
        json={
            'name': target,
            'overrides': discovery['recommended_overrides'],
            'discovery_revision': discovery['revision'],
            'auto_apply': True,
            'expected_revision': state['revision'],
            'expected_selected_config': None,
        },
    )

    assert selected.status_code == 200
    payload = selected.get_json()
    assert payload['selected_config'] == target
    assert all(not device['running'] for device in payload['devices'])
    robot_config = SwitchLifecycle.instances[0].config['devices'][0]
    assert robot_config['network_interface_whitelist'] == ['10.42.0.71']
    assert robot_config['robot_sn'] == [
        'Rizon4-111111',
        'Rizon4R-222222',
    ]
    assert 'robot_left' not in discovery['recommended_overrides']['RizonRobot:2']
    assert 'robot_right' not in discovery['recommended_overrides']['RizonRobot:2']
    assert payload['effective_bindings']['RizonRobot:2'][
        'network_interface'
    ]['ipv4'] == '10.42.0.71'
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_offline_hardware_does_not_block_selection_then_allows_manual_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_robot_config(config_dir, 'offline-robot')
    offline = HardwareSnapshot(
        revision='offline-hardware-revision',
        generated_at='2026-07-16T00:00:00+00:00',
        interfaces=[
            {
                'id': 'interface_unrelated',
                'name': 'eth0',
                'ipv4': '10.128.0.241',
                'prefixlen': 23,
                'network': '10.128.0.0/23',
                'operstate': 'up',
                'robot_facing_candidate': False,
                'source': 'linux_interfaces',
            }
        ],
        cameras=[],
        robots=[],
        providers=[],
        warnings=['robots are unplugged'],
    )
    class DiscoveryProbe(StaticDiscovery):
        def __init__(self, snapshot: HardwareSnapshot) -> None:
            super().__init__(snapshot)
            self.calls = 0

        def discover(self) -> HardwareSnapshot:
            self.calls += 1
            return super().discover()

    discovery = DiscoveryProbe(offline)
    SwitchLifecycle.reset()
    monkeypatch.setattr(server, 'DeviceLifecycleManager', SwitchLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=discovery,
    )
    app.testing = True
    client = app.test_client()

    selected = client.post(
        '/api/config/select',
        json={'name': 'offline-robot'},
    )

    assert selected.status_code == 200
    assert discovery.calls == 1
    selected_state = selected.get_json()
    assert selected_state['selected_config'] == 'offline-robot'
    assert all(not item['running'] for item in selected_state['devices'])
    selected_robot = SwitchLifecycle.instances[0].config['devices'][0]
    assert selected_robot['robot_sn'] == [
        'Rizon4-111111',
        'Rizon4R-222222',
    ]
    assert selected_robot['network_interface_whitelist'] == ['192.0.2.99']
    offline_payload = client.get('/api/hardware/discovery').get_json()
    assert discovery.calls == 2
    assert offline_payload['configured_detection']['RizonRobot:2'][
        'detected'
    ] == [False, False]

    discovery.snapshot = HardwareSnapshot(
        revision='online-hardware-revision',
        generated_at='2026-07-16T00:01:00+00:00',
        interfaces=[
            {
                'id': 'interface_robot_lan',
                'name': 'enp132s0',
                'ipv4': '192.168.2.10',
                'prefixlen': 24,
                'network': '192.168.2.0/24',
                'operstate': 'up',
                'robot_facing_candidate': True,
                'source': 'linux_interfaces',
            }
        ],
        cameras=[],
        robots=[
            {
                'id': 'robot_left_manual',
                'model': 'Rizon4',
                'variant': 'standard',
                'rdk_sn': 'Rizon4-063087',
                'serial_number': 'Rizon4-063087',
                'ip': '192.168.2.102',
                'source': 'elements_cache',
            },
            {
                'id': 'robot_right_manual',
                'model': 'Rizon4R',
                'variant': '4r',
                'rdk_sn': 'Rizon4R-062071',
                'serial_number': 'Rizon4R-062071',
                'ip': '192.168.2.101',
                'source': 'elements_cache',
            },
        ],
        providers=[],
        warnings=[],
    )
    online_payload = client.get('/api/hardware/discovery').get_json()
    rebound = client.post(
        '/api/bindings',
        json={
            'name': 'offline-robot',
            'discovery_revision': online_payload['revision'],
            'overrides': {
                'RizonRobot:2': {
                    'robot_left': 'robot_left_manual',
                    'robot_right': 'robot_right_manual',
                    'network_interface': 'interface_robot_lan',
                }
            },
        },
    )

    assert rebound.status_code == 200
    effective_robot = app.config['DEVICE_CONTROLLER'].starter.config['devices'][0]
    assert effective_robot['robot_sn'] == [
        'Rizon4-063087',
        'Rizon4R-062071',
    ]
    assert effective_robot['network_interface_whitelist'] == ['192.168.2.10']
    assert all(not item['running'] for item in rebound.get_json()['devices'])
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_auto_apply_is_one_shot_and_never_replaces_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, server.DEFAULT_RUNTIME_CONFIG_NAME)
    BlockingLifecycle.instances = []
    monkeypatch.setattr(server, 'DeviceLifecycleManager', BlockingLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()
    controller = app.config['DEVICE_CONTROLLER']
    replace_calls = 0
    original_replace = controller._replace_starter_locked

    def counted_replace(**kwargs: Any) -> Dict[str, Any]:
        nonlocal replace_calls
        replace_calls += 1
        return original_replace(**kwargs)

    monkeypatch.setattr(controller, '_replace_starter_locked', counted_replace)
    request_payload = {
        'name': server.DEFAULT_RUNTIME_CONFIG_NAME,
        'auto_apply': True,
        'expected_revision': 0,
        'expected_selected_config': None,
    }

    first = client.post('/api/config/select', json=request_payload)
    first_starter = controller.starter
    second = client.post('/api/config/select', json=request_payload)

    assert first.status_code == 200
    assert second.status_code == 409
    conflict = second.get_json()
    assert conflict['revision'] == 1
    assert conflict['selected_config'] == server.DEFAULT_RUNTIME_CONFIG_NAME
    assert replace_calls == 1
    assert controller.starter is first_starter
    assert first_starter.stopped is False
    controller.shutdown()


def test_auto_apply_rejects_an_existing_active_selection_without_stopping_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first')
    _write_config(config_dir, server.DEFAULT_RUNTIME_CONFIG_NAME)
    BlockingLifecycle.instances = []
    monkeypatch.setattr(server, 'DeviceLifecycleManager', BlockingLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
    )
    app.testing = True
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'first'}).status_code == 200
    assert client.post('/api/devices/BaseDevice/7/start').status_code == 202
    current = app.config['DEVICE_CONTROLLER'].starter
    current_revision = client.get('/api/state').get_json()['revision']
    rejected = client.post(
        '/api/config/select',
        json={
            'name': server.DEFAULT_RUNTIME_CONFIG_NAME,
            'auto_apply': True,
            'expected_revision': current_revision,
            'expected_selected_config': None,
            'stop_active': True,
        },
    )

    assert rejected.status_code == 409
    assert app.config['DEVICE_CONTROLLER'].starter is current
    assert app.config['DEVICE_CONTROLLER'].selected_config_name() == 'first'
    assert current.running is True
    assert current.stopped is False
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_start_and_config_select_cannot_create_stale_unmanaged_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first')
    _write_config(config_dir, 'second')
    BlockingLifecycle.instances = []
    monkeypatch.setattr(server, 'DeviceLifecycleManager', BlockingLifecycle)
    controller = server.DeviceController(config_dir=config_dir)
    controller.select_config('first')
    current = BlockingLifecycle.instances[0]
    original_resolver = controller._require_starter_locked
    resolver_entered = threading.Event()
    release_resolver = threading.Event()
    select_finished = threading.Event()

    def paused_resolver() -> BlockingLifecycle:
        resolved = original_resolver()
        resolver_entered.set()
        assert release_resolver.wait(timeout=5.0)
        return resolved

    monkeypatch.setattr(controller, '_require_starter_locked', paused_resolver)

    start_errors: List[BaseException] = []
    select_errors: List[BaseException] = []

    def start() -> None:
        try:
            controller.control_device('start', 'BaseDevice', 7)
        except BaseException as exc:
            start_errors.append(exc)

    def select() -> None:
        try:
            controller.select_config('second')
        except BaseException as exc:
            select_errors.append(exc)
        finally:
            select_finished.set()

    start_thread = threading.Thread(target=start)
    start_thread.start()
    assert resolver_entered.wait(timeout=5.0)
    select_thread = threading.Thread(target=select)
    select_thread.start()

    # The second candidate is constructed outside the controller lock, proving
    # its request reached the lock while the first manager's start still owns it.
    deadline = threading.Event()
    for _ in range(100):
        if len(BlockingLifecycle.instances) >= 2:
            break
        deadline.wait(0.01)
    assert len(BlockingLifecycle.instances) == 2
    # In the fixed implementation, resolver execution is already inside the
    # controller lock, so config replacement cannot finish while it is paused.
    # A resolver called before acquiring the lock would let select complete and
    # then start the stale manager after this release.
    assert select_finished.wait(timeout=0.1) is False
    release_resolver.set()
    start_thread.join(timeout=5.0)
    select_thread.join(timeout=5.0)

    assert start_errors == []
    assert len(select_errors) == 1
    assert isinstance(select_errors[0], DeviceConflictError)
    assert controller.starter is current
    assert current.running is True
    assert all(
        lifecycle is current or lifecycle.running is False
        for lifecycle in BlockingLifecycle.instances
    )
    controller.shutdown()


def test_stop_active_switch_stops_non_motion_devices_and_reports_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first')
    _write_config(config_dir, 'second')
    _install_switch_doubles(monkeypatch)
    app = server.create_app(config_dir=config_dir)
    app.testing = True
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'first'}).status_code == 200
    assert client.post('/api/devices/BaseDevice/7/start').status_code == 202

    switched = client.post(
        '/api/config/select',
        json={'name': 'second', 'stop_active': True},
    )

    assert switched.status_code == 200
    payload = switched.get_json()
    assert payload['selected_config'] == 'second'
    assert payload['previous_config'] == 'first'
    assert payload['stopped_previous'] is True
    assert payload['stopped_devices'] == ['BaseDevice:7']
    assert SwitchLifecycle.instances[0].stopped is True
    assert app.config['DEVICE_CONTROLLER'].starter is SwitchLifecycle.instances[1]
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_stop_active_motion_timeout_preserves_selection_and_partial_stop_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_device_config(config_dir, 'first')
    _write_config(config_dir, 'second')
    _install_switch_doubles(monkeypatch)
    SwitchLifecycle.incomplete_stop_keys = {'RizonRobot:2'}
    app = server.create_app(config_dir=config_dir)
    app.testing = True
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'first'}).status_code == 200
    assert client.post('/api/devices/BaseDevice/7/start').status_code == 202
    assert client.post('/api/devices/RizonRobot/2/start').status_code == 202
    old_starter = app.config['DEVICE_CONTROLLER'].starter

    rejected = client.post(
        '/api/config/select',
        json={'name': 'second', 'stop_active': True},
    )

    assert rejected.status_code == 409
    payload = rejected.get_json()
    assert payload['selected_config'] == 'first'
    assert payload['stopped_previous'] is True
    assert payload['stopped_devices'] == ['BaseDevice:7']
    assert payload['incomplete_devices'] == ['RizonRobot:2']
    assert app.config['DEVICE_CONTROLLER'].selected_config_name() == 'first'
    assert app.config['DEVICE_CONTROLLER'].starter is old_starter
    assert old_starter.get_device_status('BaseDevice', 7)['status'] == 'stopped'
    assert old_starter.get_device_status('RizonRobot', 2)['status'] == 'stop_timeout'

    # Make teardown finite without erasing the assertions about the rejected
    # switch's partial stop result.
    SwitchLifecycle.incomplete_stop_keys = set()
    app.config['DEVICE_CONTROLLER'].shutdown()


@pytest.mark.parametrize('shared_memory_status', ['external', 'orphaned'])
def test_stop_active_switch_is_blocked_by_unowned_shared_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_memory_status: str,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first')
    _write_config(config_dir, 'second')
    _install_switch_doubles(monkeypatch)
    app = server.create_app(config_dir=config_dir)
    app.testing = True
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'first'}).status_code == 200
    old_starter = SwitchLifecycle.instances[0]
    old_starter.mark_status('BaseDevice:7', shared_memory_status)

    rejected = client.post(
        '/api/config/select',
        json={'name': 'second', 'stop_active': True},
    )

    assert rejected.status_code == 409
    payload = rejected.get_json()
    assert payload['selected_config'] == 'first'
    assert payload['stopped_previous'] is False
    assert payload['stopped_devices'] == []
    assert payload['external_devices'] == ['BaseDevice:7']
    assert app.config['DEVICE_CONTROLLER'].starter is old_starter
    old_starter.mark_status('BaseDevice:7', 'stopped')
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_start_and_stop_active_switch_are_serialized_by_controller_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first')
    _write_config(config_dir, 'second')
    _install_switch_doubles(monkeypatch)
    controller = server.DeviceController(config_dir=config_dir)
    controller.select_config('first')
    current = SwitchLifecycle.instances[0]
    current.block_start = True
    switch_finished = threading.Event()
    start_errors: List[BaseException] = []
    switch_errors: List[BaseException] = []
    switch_results: List[Dict[str, Any]] = []

    def start() -> None:
        try:
            controller.control_device('start', 'BaseDevice', 7)
        except BaseException as exc:
            start_errors.append(exc)

    def switch() -> None:
        try:
            switch_results.append(
                controller.select_config('second', stop_active=True)
            )
        except BaseException as exc:
            switch_errors.append(exc)
        finally:
            switch_finished.set()

    start_thread = threading.Thread(target=start)
    start_thread.start()
    assert current.start_entered.wait(timeout=5.0)
    switch_thread = threading.Thread(target=switch)
    switch_thread.start()

    # Candidate construction happens before the lock, but replacement and
    # stop-all must wait until the in-flight start releases the controller lock.
    for _ in range(100):
        if len(SwitchLifecycle.instances) >= 2:
            break
        threading.Event().wait(0.01)
    assert len(SwitchLifecycle.instances) == 2
    assert switch_finished.wait(timeout=0.1) is False

    current.release_start.set()
    start_thread.join(timeout=5.0)
    switch_thread.join(timeout=5.0)

    assert start_thread.is_alive() is False
    assert switch_thread.is_alive() is False
    assert start_errors == []
    assert switch_errors == []
    assert switch_results[0]['selected_config'] == 'second'
    assert switch_results[0]['stopped_devices'] == ['BaseDevice:7']
    assert current.stopped is True
    assert controller.starter is SwitchLifecycle.instances[1]
    controller.shutdown()


def test_slow_device_start_keeps_controller_state_and_other_keys_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_device_config(config_dir, 'fixture')
    _install_switch_doubles(monkeypatch)
    controller = server.DeviceController(config_dir=config_dir)
    controller.select_config('fixture')
    gripper_stops: List[str] = []
    monkeypatch.setattr(
        controller.grippers,
        'stop',
        lambda reason: gripper_stops.append(reason) or {'ok': True},
    )
    current = SwitchLifecycle.instances[0]
    current.block_start = True
    errors: List[BaseException] = []

    def start_rizon() -> None:
        try:
            controller.control_device('start', 'RizonRobot', 2)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=start_rizon)
    thread.start()
    try:
        assert current.start_entered.wait(timeout=5.0)
        before = time.monotonic()
        snapshot = controller.state()
        assert time.monotonic() - before < 0.5
        assert snapshot['selected_config'] == 'fixture'

        # The blocked Rizon request has already passed its barrier. A distinct
        # device key must be able to start without waiting for it.
        current.block_start = False
        camera_like = controller.control_device('start', 'BaseDevice', 7)
        assert camera_like['device']['status'] == 'running'
    finally:
        current.release_start.set()
        thread.join(timeout=5.0)
        controller.shutdown()

    assert thread.is_alive() is False
    assert errors == []
    assert gripper_stops == ['device_start']


def test_web_device_start_uses_observed_path_without_waiting_for_sdk_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'fixture')
    BlockingLifecycle.instances = []
    observed_calls: List[tuple[str, Any]] = []

    def observed_start(
        self: BlockingLifecycle,
        device_class: str,
        device_class_id: Any,
    ) -> Dict[str, Any]:
        observed_calls.append((device_class, device_class_id))
        self.running = True
        return self._device(device_class, device_class_id)

    monkeypatch.setattr(
        BlockingLifecycle,
        'start_device_observed',
        observed_start,
        raising=False,
    )
    monkeypatch.setattr(server, 'DeviceLifecycleManager', BlockingLifecycle)
    controller = server.DeviceController(config_dir=config_dir)
    controller.select_config('fixture')
    gripper_stops: List[str] = []
    monkeypatch.setattr(
        controller.grippers,
        'stop',
        lambda reason: gripper_stops.append(reason) or {'ok': True},
    )
    current = BlockingLifecycle.instances[0]
    current.block_start = True

    result = controller.control_device('start', 'BaseDevice', 7)

    assert result['device']['status'] == 'running'
    assert observed_calls == [('BaseDevice', 7)]
    assert current.start_entered.is_set() is False
    assert gripper_stops == []
    controller.shutdown()


def test_discovery_for_candidate_config_does_not_mutate_current_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first')
    _write_camera_config(config_dir, 'second')
    snapshot = HardwareSnapshot(
        revision='camera-hardware-revision',
        generated_at='2026-07-12T00:00:00+00:00',
        interfaces=[],
        cameras=[
            {
                'id': 'camera_opaque_second',
                'devnode': '/dev/video4',
                'stable_path': '/dev/v4l/by-id/camera-second-video-index0',
                'name': 'Camera Second',
                'capture_capable': True,
                'source': 'sysfs_v4l2',
            }
        ],
        robots=[],
        providers=[{'name': 'sysfs_v4l2', 'mode': 'read_only'}],
        warnings=[],
    )
    SwitchLifecycle.reset()
    monkeypatch.setattr(server, 'DeviceLifecycleManager', SwitchLifecycle)
    monkeypatch.setattr(server, 'HardwareDiscovery', lambda: StaticDiscovery(snapshot))
    app = server.create_app(config_dir=config_dir)
    app.testing = True
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'first'}).status_code == 200
    before = client.get('/api/state').get_json()

    discovered = client.get('/api/hardware/discovery?config_name=second')

    assert discovered.status_code == 200
    payload = discovered.get_json()
    assert payload['selected_config'] == 'first'
    assert payload['config_name'] == 'second'
    assert payload['recommended_overrides'] == {
        'WebCameraOpenPi:4': {'capture_device': 'camera_opaque_second'}
    }
    after = client.get('/api/state').get_json()
    assert after['selected_config'] == before['selected_config'] == 'first'
    assert after['revision'] == before['revision']
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_global_logs_are_not_exposed_and_shutdown_rejects_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first')
    BlockingLifecycle.instances = []
    monkeypatch.setattr(server, 'DeviceLifecycleManager', BlockingLifecycle)
    app = server.create_app(config_dir=config_dir)
    app.testing = True
    client = app.test_client()

    secret = 'SECRET_SENTINEL token=abc123 password=hunter2'
    global_logger.info(secret)
    logs = client.get('/api/logs').get_json()
    assert secret not in str(logs)
    assert 'token=abc123' not in str(logs)
    assert 'password=hunter2' not in str(logs)

    controller = app.config['DEVICE_CONTROLLER']
    controller.shutdown()
    select = client.post('/api/config/select', json={'name': 'first'})
    start = client.post('/api/devices/BaseDevice/7/start')
    assert select.status_code == 409
    assert start.status_code == 409
