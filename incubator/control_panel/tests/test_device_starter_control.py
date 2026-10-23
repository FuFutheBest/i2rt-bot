"""Hardware-free lifecycle and REST tests for device control v1."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import threading
import time
import types
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from incubator.control_panel import lifecycle
from incubator.control_panel.server import DeviceController, create_app
from incubator.control_panel.lifecycle import (
    DeviceConflictError,
    DeviceControlError,
    DeviceLifecycleManager,
)
from devices.config import StateFeedbackConfig
from devices.robots.config import RobotControlConfig
from incubator.control_panel.hardware_discovery import CAPTURE_BINDING_STATE_KEY

DeviceStarter = DeviceLifecycleManager


def _config() -> Dict[str, Any]:
    return {
        'devices': [
            {
                'class': 'BaseDevice',
                'device_class_id': 7,
                'data_shape': [2],
                'data_dtype': 'float32',
                'fps': 10.0,
            }
        ],
        'global_settings': {
            'restart_on_failure': True,
            'max_restart_attempts': 2,
            'restart_delay': 0.0,
        },
        'runtime': {'signal_files': {}},
    }


def _two_device_config() -> Dict[str, Any]:
    config = _config()
    config['devices'].append(
        {
            'class': 'WebCameraOpenPi',
            'device_class_id': 4,
            'data_shape': [8, 8, 3],
            'data_dtype': 'uint8',
            'fps': 30.0,
        }
    )
    return config


def _robot_control(**overrides: object) -> Dict[str, Any]:
    control: Dict[str, Any] = {
        'type': 'joints',
        'format': {
            'shape': [2],
            'dtype': 'float64',
            'type': ['linear:2'],
        },
    }
    control.update(overrides)
    return control


def test_lifecycle_validates_control_before_worker_signature_filtering() -> None:
    config = _config()
    config['devices'][0]['control'] = _robot_control()

    manager = DeviceStarter(
        config=config,
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )

    assert manager.devices[0]['config']['control'] == _robot_control()
    manager.stop_all()


def test_lifecycle_parses_full_structured_device_contract() -> None:
    config = _config()
    config['devices'][0].update(
        {
            'class': 'BaseRobot',
            'state_feedback': {
                'fps': 10.0,
                'buffer_size': 2,
                'hardware_latency_ms': 0.0,
                'format': {
                    'shape': [2],
                    'dtype': 'float64',
                    'type': ['linear:2'],
                },
            },
            'control': _robot_control(),
        }
    )

    parsed = lifecycle._parse_device_control_contract(
        config['devices'][0], context='devices[0]', for_constructor=True
    )
    assert isinstance(parsed['state_feedback'], StateFeedbackConfig)
    assert isinstance(parsed['control'], RobotControlConfig)


def test_lifecycle_rejects_removed_control_field_before_worker_spawn() -> None:
    config = _config()
    config['devices'][0]['control'] = _robot_control(enabled_arms=[True, True])

    with pytest.raises(DeviceControlError, match='enabled_arms'):
        DeviceStarter(
            config=config,
            enable_device_manager=False,
            enable_policy_runner=False,
            install_signal_handlers=False,
        )


def test_worker_rechecks_control_contract_before_signature_filtering() -> None:
    with pytest.raises(DeviceControlError, match='enabled_arms'):
        lifecycle._run_device_worker(
            'BaseDevice',
            {
                'class': 'BaseDevice',
                'device_class_id': 7,
                'control': _robot_control(enabled_arms=[True, True]),
            },
            'test-owner',
            None,
        )



def _missing_shm(device_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'name': f"{device_info['device_class']}_{device_info['device_class_id']}_data",
        'connected': False,
        'last_timestamp_ns': None,
        'age_ms': None,
        'fps': device_info['config'].get('fps'),
        'shape': device_info['config'].get('data_shape'),
        'dtype': device_info['config'].get('data_dtype'),
        'fresh': False,
        'frames': 0,
        'error': None,
    }


class FakeProcess:
    _next_pid = 1000

    def __init__(self) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.exitcode = None
        self.alive = False
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_calls = 0
        self.join_timeouts = []

    def start(self) -> None:
        self.started = True
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        self.exitcode = -9

    def join(self, timeout: float = 0.0) -> None:
        self.join_calls += 1
        self.join_timeouts.append(timeout)


class StubbornProcess(FakeProcess):
    """A fake worker that ignores the graceful SIGTERM request."""

    def terminate(self) -> None:
        self.terminated = True


class BlockingStartProcess(FakeProcess):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def start(self) -> None:
        self.entered.set()
        assert self.release.wait(timeout=5.0)
        super().start()


def _spawn_probe_target(
    device_class_name: str,
    device_config: Dict[str, Any],
    owner_token: str,
    ownership_conn: Any,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Pickleable hardware-free target used to exercise real spawn."""
    ownership_conn.send(
        {
            'event': 'worker_started',
            'owner_token': owner_token,
            'start_method': mp.get_start_method(),
            'class': device_class_name,
            'device_class_id': device_config['device_class_id'],
            'runtime_config': runtime_config,
        }
    )
    ownership_conn.close()


@pytest.fixture
def starter(monkeypatch: pytest.MonkeyPatch) -> DeviceStarter:
    instance = DeviceStarter(
        config=_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    monkeypatch.setattr(
        instance, '_start_generic_device_process', lambda device: FakeProcess()
    )
    yield instance
    instance.stop_all()


def test_independent_start_and_stop_cleans_worker(starter: DeviceStarter) -> None:
    started = starter.start_device('BaseDevice', '7')
    process = starter.devices[0]['process']

    assert started['status'] == 'starting'
    assert started['desired_state'] == 'running'
    assert process.started

    stopped = starter.stop_device('BaseDevice', 7, timeout=0.0)

    assert stopped['status'] == 'stopped'
    assert stopped['desired_state'] == 'stopped'
    assert process.terminated
    assert process.join_calls >= 1
    assert starter.devices[0]['process'] is None


def test_observed_start_skips_blocking_ownership_wait(
    starter: DeviceStarter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = []
    monkeypatch.setattr(
        starter,
        '_wait_for_ownership_preflight',
        lambda device: waits.append(device),
    )

    started = starter.start_device_observed('BaseDevice', 7)

    assert started['status'] == 'starting'
    assert started['running'] is True
    assert waits == []


def test_async_stop_during_preflight_prevents_worker_spawn(
    starter: DeviceStarter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    process = FakeProcess()
    results = []
    errors = []

    def slow_preflight(device_info: Dict[str, Any]) -> None:
        del device_info
        entered.set()
        assert release.wait(timeout=5.0)

    monkeypatch.setattr(starter, '_prepare_rizon_rdk', slow_preflight)
    monkeypatch.setattr(
        starter, '_start_generic_device_process', lambda device: process
    )

    def start_device() -> None:
        try:
            results.append(starter.start_device_observed('BaseDevice', 7))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=start_device)
    thread.start()
    try:
        assert entered.wait(timeout=2.0)
        assert starter.request_stop_device('BaseDevice:7') is True
        assert starter.request_stop_device('BaseDevice:7') is False
        stopping = starter.get_device_status('BaseDevice', 7)
        assert stopping['status'] == 'stopping'
        assert stopping['desired_state'] == 'stopped'
    finally:
        release.set()
        thread.join(timeout=5.0)

    assert thread.is_alive() is False
    assert errors == []
    assert results[0]['status'] == 'stopped'
    assert process.started is False


def test_async_stop_during_spawn_terminates_worker_after_spawn(
    starter: DeviceStarter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    process = BlockingStartProcess(entered, release)
    results = []
    errors = []
    monkeypatch.setattr(
        starter, '_start_generic_device_process', lambda device: process
    )

    def start_device() -> None:
        try:
            results.append(starter.start_device_observed('BaseDevice', 7))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=start_device)
    thread.start()
    try:
        assert entered.wait(timeout=2.0)
        assert starter.request_stop_device('BaseDevice:7') is True
    finally:
        release.set()
        thread.join(timeout=5.0)

    assert thread.is_alive() is False
    assert errors == []
    assert process.started is True
    assert process.terminated is True
    assert results[0]['status'] == 'stopping'
    stopped = starter.stop_device('BaseDevice', 7, timeout=0.0)
    assert stopped['status'] == 'stopped'


def test_controller_async_stop_forwards_to_lifecycle_without_waiting() -> None:
    requested = []

    class Starter:
        def request_stop_device(self, device_key: str) -> bool:
            requested.append(device_key)
            return True

    controller = DeviceController.__new__(DeviceController)
    controller.starter = Starter()
    controller._lock = threading.RLock()
    controller._device_operations = {}
    controller._workflow_cancelled_device_keys = set()
    controller._workflow_cancel_event = threading.Event()
    controller._lifecycle_mutation_pending = False

    controller.request_async_stop('device:BaseDevice:7')

    assert requested == ['BaseDevice:7']


def test_slow_spawn_does_not_block_status_or_another_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = DeviceStarter(
        config=_two_device_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    entered = threading.Event()
    release = threading.Event()
    slow_process = BlockingStartProcess(entered, release)
    camera_process = FakeProcess()
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    monkeypatch.setattr(
        instance,
        '_start_generic_device_process',
        lambda device: (
            slow_process
            if device['device_class'] == 'BaseDevice'
            else camera_process
        ),
    )
    errors = []

    def start_slow_device() -> None:
        try:
            instance.start_device_observed('BaseDevice', 7)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=start_slow_device)
    thread.start()
    try:
        assert entered.wait(timeout=2.0)
        before = time.monotonic()
        status = instance.get_status()
        assert time.monotonic() - before < 0.5
        status_by_key = {item['key']: item for item in status['devices']}
        assert status_by_key['BaseDevice:7']['status'] == 'starting'

        camera = instance.start_device_observed('WebCameraOpenPi', 4)
        assert camera['status'] == 'starting'
        assert camera_process.started is True
    finally:
        release.set()
        thread.join(timeout=5.0)
        instance.stop_all()

    assert thread.is_alive() is False
    assert errors == []


def test_manual_desired_stop_prevents_monitor_restart(
    starter: DeviceStarter, monkeypatch: pytest.MonkeyPatch
) -> None:
    starter.start_device('BaseDevice', 7)
    device = starter.devices[0]
    process = device['process']
    process.alive = False
    process.exitcode = 1
    device['desired_state'] = 'stopped'

    restart_calls = []
    monkeypatch.setattr(
        starter,
        '_start_device_process',
        lambda info: restart_calls.append(info) or True,
    )
    starter.start_monitor()

    deadline = time.time() + 1.0
    while device['process'] is not None and time.time() < deadline:
        time.sleep(0.01)
    starter.running = False
    starter._monitor_wakeup.set()

    assert restart_calls == []
    assert device['process'] is None
    assert device['lifecycle_status'] == 'stopped'


def test_automatic_restart_rechecks_desired_state_after_leaving_monitor_lock(
    starter: DeviceStarter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = starter.devices[0]
    dead_process = FakeProcess()
    dead_process.exitcode = 1
    device['process'] = dead_process
    device['desired_state'] = 'running'
    device['last_exit_at'] = None
    restart_calls = []
    released = []

    class StopDuringHandoffLock:
        def acquire(self, blocking: bool = True) -> bool:
            assert blocking is False
            starter.running = False
            device['desired_state'] = 'stopped'
            return True

        def release(self) -> None:
            released.append(True)

    with monkeypatch.context() as patch:
        patch.setattr(
            starter,
            '_device_operation_lock',
            lambda info: StopDuringHandoffLock(),
        )
        patch.setattr(
            starter,
            '_start_device_process',
            lambda info: restart_calls.append(info) or True,
        )
        starter.running = True
        starter._monitor_devices()

    assert restart_calls == []
    assert released == [True]


def test_start_server_false_is_structured_terminal_failure_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FalseStartDevice:
        def __init__(self, device_class_id: int) -> None:
            self.device_class_id = device_class_id

        def start_server(self) -> bool:
            return False

        def stop_server(self) -> None:
            return None

    class CaptureConnection:
        def __init__(self) -> None:
            self.events = []
            self.closed = False

        def send(self, payload: Dict[str, Any]) -> None:
            self.events.append(dict(payload))

        def close(self) -> None:
            self.closed = True

    fake_devices = types.ModuleType('devices')
    fake_devices.DEVICE_CLASSES = {'BaseDevice': FalseStartDevice}
    fake_shm_utils = types.ModuleType('utils.shm_utils')
    fake_shm_utils.create_shared_memory = lambda *args, **kwargs: None
    fake_shm_utils.make_readonly = lambda value: value
    fake_utils = types.ModuleType('utils')
    fake_utils.shm_utils = fake_shm_utils
    monkeypatch.setitem(sys.modules, 'devices', fake_devices)
    monkeypatch.setitem(sys.modules, 'utils', fake_utils)
    monkeypatch.setitem(sys.modules, 'utils.shm_utils', fake_shm_utils)
    monkeypatch.setattr(lifecycle.signal, 'signal', lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle, '_worker_shm_preflight', lambda name: False)

    owner_token = 'terminal-false-start-owner'
    connection = CaptureConnection()
    with pytest.raises(
        lifecycle.DeviceStartupError,
        match='start_server returned False',
    ):
        lifecycle._run_device_worker(
            'BaseDevice',
            dict(_config()['devices'][0]),
            owner_token,
            connection,
        )

    event_names = [event['event'] for event in connection.events]
    assert 'start_failed' in event_names
    assert event_names.index('start_failed') < event_names.index('worker_error')
    start_failed = next(
        event for event in connection.events if event['event'] == 'start_failed'
    )
    assert start_failed['owner_token'] == owner_token
    assert start_failed['code'] == 'start_server_returned_false'
    assert start_failed['recoverable'] is False
    assert start_failed['error'] == 'Device start_server returned False'
    assert connection.closed is True

    instance = DeviceStarter(
        config=_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    device = instance.devices[0]
    device['ownership_token'] = owner_token
    for event in connection.events:
        instance._apply_ownership_event(device, event)

    dead_process = FakeProcess()
    dead_process.exitcode = 1
    device['process'] = dead_process
    restart_calls = []
    monkeypatch.setattr(
        instance,
        '_start_device_process',
        lambda info: restart_calls.append(info) or True,
    )
    instance.start_monitor()
    deadline = time.time() + 1.0
    while device['process'] is not None and time.time() < deadline:
        time.sleep(0.01)
    # Give a zero-delay restart policy one extra monitor wakeup.  The terminal
    # desired state must still prevent any automatic hardware retry.
    instance._monitor_wakeup.set()
    time.sleep(0.05)

    status = instance.get_device_status('BaseDevice', 7)
    assert restart_calls == []
    assert device['process'] is None
    assert device['lifecycle_status'] == 'failed'
    assert status['status'] == 'failed'
    assert status['desired_state'] == 'stopped'
    assert status['failure_code'] == 'start_server_returned_false'
    assert status['failure_recoverable'] is False
    assert status['failure'] == {
        'code': 'start_server_returned_false',
        'recoverable': False,
        'message': 'Device start_server returned False',
    }
    assert status['ownership']['status'] == 'start_failed'
    instance.running = False
    instance._monitor_wakeup.set()
    if instance._monitor_thread is not None:
        instance._monitor_thread.join(timeout=2.0)


def test_worker_sigterm_unwinds_before_running_device_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    installed_handlers: Dict[int, Any] = {}

    class SignalDevice:
        def __init__(self, device_class_id: int) -> None:
            self.device_class_id = device_class_id

        def start_server(self) -> None:
            calls.append('start_entered')
            handler = installed_handlers[lifecycle.signal.SIGTERM]
            handler(lifecycle.signal.SIGTERM, None)
            calls.append('start_continued')

        def stop_server(self) -> None:
            calls.append('cleanup')

    class CaptureConnection:
        def __init__(self) -> None:
            self.events: list[Dict[str, Any]] = []
            self.closed = False

        def send(self, payload: Dict[str, Any]) -> None:
            self.events.append(dict(payload))

        def close(self) -> None:
            self.closed = True

    fake_devices = types.ModuleType('devices')
    fake_devices.DEVICE_CLASSES = {'BaseDevice': SignalDevice}
    fake_shm_utils = types.ModuleType('utils.shm_utils')
    fake_shm_utils.create_shared_memory = lambda *args, **kwargs: None
    fake_shm_utils.make_readonly = lambda value: value
    fake_utils = types.ModuleType('utils')
    fake_utils.shm_utils = fake_shm_utils
    monkeypatch.setitem(sys.modules, 'devices', fake_devices)
    monkeypatch.setitem(sys.modules, 'utils', fake_utils)
    monkeypatch.setitem(sys.modules, 'utils.shm_utils', fake_shm_utils)
    monkeypatch.setattr(lifecycle, '_worker_shm_preflight', lambda name: False)
    monkeypatch.setattr(
        lifecycle.signal,
        'signal',
        lambda signum, handler: installed_handlers.__setitem__(signum, handler),
    )

    connection = CaptureConnection()
    lifecycle._run_device_worker(
        'BaseDevice',
        dict(_config()['devices'][0]),
        'signal-stop-owner',
        connection,
    )

    event_names = [event['event'] for event in connection.events]
    assert calls == ['start_entered', 'cleanup']
    assert 'stop_requested' in event_names
    assert 'worker_stopped' in event_names
    assert 'worker_error' not in event_names
    assert event_names.index('stop_requested') < event_names.index('worker_stopped')
    assert connection.closed is True


def test_allowlisted_device_start_failure_preserves_stage_and_recoverability() -> None:
    class StagedFailure:
        @staticmethod
        def startup_failure() -> Dict[str, Any]:
            return {
                'code': 'rizon_connect_failed',
                'recoverable': True,
                'message': 'Rizon connection or enable failed.',
            }

    assert lifecycle._device_start_failure(StagedFailure()) == {
        'code': 'rizon_connect_failed',
        'recoverable': True,
        'message': 'Rizon connection or enable failed.',
    }


def test_structured_start_failure_is_not_overwritten_by_worker_error(
    starter: DeviceStarter,
) -> None:
    device = starter.devices[0]
    device['ownership_token'] = 'structured-failure-owner'
    starter._apply_ownership_event(
        device,
        {
            'owner_token': 'structured-failure-owner',
            'event': 'start_failed',
            'code': 'rizon_connect_failed',
            'recoverable': True,
            'error': 'Rizon connection or enable failed.',
        },
    )
    starter._apply_ownership_event(
        device,
        {
            'owner_token': 'structured-failure-owner',
            'event': 'worker_error',
            'error_type': 'DeviceStartupError',
            'error': 'generic worker wrapper',
        },
    )

    assert device['ownership_status'] == 'start_failed'
    assert device['lifecycle_status'] == 'failed'
    assert device['failure_code'] == 'rizon_connect_failed'
    assert device['failure_recoverable'] is True
    assert device['last_error'] == 'Rizon connection or enable failed.'


def test_existing_external_shm_blocks_start(
    starter: DeviceStarter, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = _missing_shm(starter.devices[0])
    external['connected'] = True
    external['fresh'] = True
    monkeypatch.setattr(starter, '_probe_shared_memory', lambda info: external)

    with pytest.raises(DeviceConflictError, match='already exists'):
        starter.start_device('BaseDevice', 7)

    status = starter.get_device_status('BaseDevice', 7)
    assert status['status'] == 'external'
    assert status['capabilities']['start'] is False


def test_explicitly_missing_camera_binding_is_visible_but_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _two_device_config()
    config['devices'][1][CAPTURE_BINDING_STATE_KEY] = 'missing'
    starter = DeviceStarter(
        config=config,
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(starter, '_probe_shared_memory', _missing_shm)

    status = starter.get_device_status('WebCameraOpenPi', 4)
    assert status['status'] == 'stopped'
    assert status['binding_status'] == 'missing'
    assert status['capabilities']['start'] is False
    assert 'binding is missing' in status['reason']

    with pytest.raises(DeviceConflictError, match='Camera binding is missing'):
        starter.start_device('WebCameraOpenPi', 4)

    assert starter.get_device_status('WebCameraOpenPi', 4)['desired_state'] == 'stopped'
    starter.stop_all()


def _write_test_config(config_dir: Path, name: str = 'demo') -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f'{name}.yaml').write_text(
        """devices:
  - class: BaseDevice
    device_class_id: 7
    data_shape: [2]
    data_dtype: float32
    fps: 10.0
global_settings:
  restart_on_failure: true
  max_restart_attempts: 2
  restart_delay: 0.0
runtime:
  signal_files: {}
""",
        encoding='utf-8',
    )


def test_rest_select_start_state_and_config_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_test_config(config_dir, 'demo')
    _write_test_config(config_dir, 'other')
    monkeypatch.setattr(
        DeviceStarter,
        '_probe_shared_memory',
        lambda self, device: _missing_shm(device),
    )
    monkeypatch.setattr(
        DeviceStarter,
        '_start_generic_device_process',
        lambda self, device: FakeProcess(),
    )

    app = create_app(config_dir=config_dir)
    app.testing = True
    client = app.test_client()

    listed = client.get('/api/configs')
    assert listed.status_code == 200
    assert listed.get_json()['configs'] == ['demo', 'other']

    selected = client.post('/api/config/select', json={'name': 'demo'})
    assert selected.status_code == 200
    assert selected.get_json()['selected_config'] == 'demo'
    assert selected.get_json()['lifecycle']['enable_device_manager'] is False
    assert selected.get_json()['lifecycle']['enable_policy_runner'] is False

    started = client.post('/api/devices/BaseDevice/7/start')
    assert started.status_code == 202
    assert started.get_json()['device']['key'] == 'BaseDevice:7'

    state = client.get('/api/state').get_json()
    assert state['active_config'] == 'demo'
    assert state['devices'][0]['desired_state'] == 'running'

    conflict = client.post('/api/config/select', json={'name': 'other'})
    assert conflict.status_code == 409

    stopped = client.post('/api/devices/BaseDevice/7/stop')
    assert stopped.status_code == 200
    assert stopped.get_json()['device']['status'] == 'stopped'

    app.config['DEVICE_CONTROLLER'].shutdown()


def test_force_stop_endpoint_requires_exact_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_test_config(config_dir, 'demo')
    app = create_app(config_dir=config_dir)
    app.testing = True
    controller = app.config['DEVICE_CONTROLLER']
    calls: list[str] = []
    monkeypatch.setattr(
        controller,
        'force_stop_all',
        lambda: calls.append('force')
        or {
            'ok': True,
            'killed_pids': [1234],
            'web_server_preserved': True,
        },
    )
    client = app.test_client()

    missing = client.post('/api/processes/force-stop-all', json={})
    wrong = client.post(
        '/api/processes/force-stop-all',
        json={'confirmation': 'force-stop-all'},
    )
    extra = client.post(
        '/api/processes/force-stop-all',
        json={'confirmation': 'FORCE_STOP_ALL', 'extra': True},
    )
    accepted = client.post(
        '/api/processes/force-stop-all',
        json={'confirmation': 'FORCE_STOP_ALL'},
    )

    assert missing.status_code == 400
    assert wrong.status_code == 400
    assert extra.status_code == 400
    assert accepted.status_code == 200
    assert accepted.get_json()['killed_pids'] == [1234]
    assert accepted.headers['Cache-Control'] == 'no-store, max-age=0'
    assert calls == ['force']
    controller.shutdown()


def test_motion_stop_timeout_never_sigkills_and_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config['devices'][0]['class'] = 'RizonRobot'
    instance = DeviceStarter(
        config=config,
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    process = StubbornProcess()
    process.start()
    device = instance.devices[0]
    device['process'] = process
    device['desired_state'] = 'running'

    status = instance.stop_device('RizonRobot', 7, timeout=0.0)

    assert status['status'] == 'stop_timeout'
    assert status['running'] is True
    assert status['risk'] == 'motion'
    assert status['desired_state'] == 'stopped'
    assert process.terminated is True
    assert process.killed is False
    assert max(process.join_timeouts) >= 30.0
    assert 'Safety stop incomplete' in status['last_error']
    assert 'SIGKILL was not sent' in status['last_error']
    assert device['process'] is process

    # Do not leave the intentionally stubborn test double active for cleanup.
    process.alive = False
    device['process'] = None
    instance.stop_all()


def test_non_motion_force_kill_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = DeviceStarter(
        config=_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    process = StubbornProcess()
    process.start()
    device = instance.devices[0]
    device['process'] = process
    device['desired_state'] = 'running'

    status = instance.stop_device('BaseDevice', 7, timeout=0.0)

    assert status['status'] == 'stopped'
    assert status['running'] is False
    assert process.killed is True
    assert 'force-killed' in status['last_error']
    assert device['process'] is None
    instance.stop_all()


def test_normal_stop_all_removes_all_matching_owned_shm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = DeviceStarter(
        config=_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    process = FakeProcess()
    process.start()
    device = instance.devices[0]
    device['process'] = process
    device['desired_state'] = 'running'
    device['owned_shm_names'] = [
        'RizonRobot_2_data',
        'RizonRobot_2_control',
    ]
    monkeypatch.setattr(
        instance,
        '_owned_shm_snapshot',
        lambda _device: {
            'RizonRobot_2_data': (15, 601),
            'RizonRobot_2_control': (15, 602),
        },
    )
    cleanup_calls: list[dict[str, tuple[int, int]]] = []

    def cleanup(snapshot: dict[str, tuple[int, int]]) -> tuple[list[str], list[str]]:
        cleanup_calls.append(snapshot)
        return sorted(snapshot), []

    monkeypatch.setattr(instance, '_unlink_force_killed_shm', cleanup)

    result = instance.stop_all()

    assert result['ok'] is True
    assert result['removed_shm'] == [
        'RizonRobot_2_control',
        'RizonRobot_2_data',
    ]
    assert cleanup_calls == [{
        'RizonRobot_2_data': (15, 601),
        'RizonRobot_2_control': (15, 602),
    }]
    assert device['owned_shm_names'] == []


def test_normal_stop_all_preserves_replaced_owned_shm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = DeviceStarter(
        config=_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    process = FakeProcess()
    process.start()
    device = instance.devices[0]
    device['process'] = process
    device['desired_state'] = 'running'
    device['owned_shm_names'] = ['RizonRobot_2_control']
    monkeypatch.setattr(
        instance,
        '_owned_shm_snapshot',
        lambda _device: {'RizonRobot_2_control': (15, 701)},
    )
    monkeypatch.setattr(
        instance,
        '_unlink_force_killed_shm',
        lambda _snapshot: ([], ['RizonRobot_2_control']),
    )

    result = instance.stop_all()

    assert result['ok'] is False
    assert result['removed_shm'] == []
    assert result['incomplete_devices'] == ['BaseDevice:7']
    assert result['devices'][0]['status'] == 'orphaned'
    assert device['owned_shm_names'] == ['RizonRobot_2_control']


def test_force_stop_all_kills_stubborn_motion_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config['devices'][0]['class'] = 'RizonRobot'
    instance = DeviceStarter(
        config=config,
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    process = StubbornProcess()
    process.start()
    device = instance.devices[0]
    device['process'] = process
    device['desired_state'] = 'running'

    result = instance.force_stop_all()

    assert result['ok'] is True
    assert result['forced_devices'] == ['RizonRobot:7']
    assert result['incomplete_devices'] == []
    assert result['killed_pids'] == [process.pid]
    assert process.terminated is False
    assert process.killed is True
    status = instance.get_device_status('RizonRobot', 7)
    assert status['status'] == 'stopped'
    assert status['running'] is False
    assert status['failure_code'] == 'force_stopped'
    assert status['desired_state'] == 'stopped'


def test_force_stop_all_skips_external_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = DeviceStarter(
        config=_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )

    def external_shm(device_info: Dict[str, Any]) -> Dict[str, Any]:
        status = _missing_shm(device_info)
        status.update({'connected': True, 'fresh': True, 'frames': 1})
        return status

    monkeypatch.setattr(instance, '_probe_shared_memory', external_shm)

    result = instance.force_stop_all()

    assert result['ok'] is True
    assert result['killed_pids'] == []
    assert result['forced_devices'] == []
    assert result['incomplete_devices'] == []
    assert result['external_devices'] == ['BaseDevice:7']
    status = instance.get_device_status('BaseDevice', 7)
    assert status['status'] == 'external'
    assert 'did not signal or unlink' in status['last_error']


def test_yam_stop_timeout_covers_ramp_and_feedback_confirmation() -> None:
    config = _config()
    config['devices'][0].update(
        {
            'class': 'YamHeadRobot',
            'ramp_duration_s': 28.0,
            'disconnect_zero_timeout_s': 4.0,
        }
    )
    instance = DeviceStarter(
        config=config,
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )

    timeout = instance._resolved_stop_timeout(instance.devices[0], 0.0)

    assert timeout >= 37.0
    instance.stop_all()


def test_yam_stop_block_event_is_fail_closed_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config['devices'][0]['class'] = 'YamHeadRobot'
    instance = DeviceStarter(
        config=config,
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    process = StubbornProcess()
    process.start()
    device = instance.devices[0]
    device['process'] = process
    device['desired_state'] = 'stopped'
    token = device['ownership_token']

    instance._apply_ownership_event(
        device,
        {
            'event': 'stop_blocked',
            'owner_token': token,
            'code': 'yam_zero_not_confirmed',
        },
    )
    status = instance._status_for_device(device)

    assert status['status'] == 'stop_timeout'
    assert status['running'] is True
    assert status['failure_code'] == 'yam_zero_not_confirmed'
    assert status['failure_recoverable'] is True
    assert 'connection and zero-position hold remain active' in status['last_error']
    assert status['capabilities']['stop'] is True
    assert status['capabilities']['start'] is False

    process.alive = False
    device['process'] = None
    instance.stop_all()


def test_real_spawn_worker_can_start_from_non_main_thread() -> None:
    instance = DeviceStarter(
        config=_config(),
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    instance._device_process_target = _spawn_probe_target
    device = instance.devices[0]
    result: Dict[str, Any] = {}
    errors = []

    def launch() -> None:
        try:
            process = instance._start_generic_device_process(device)
            assert process is not None
            process.start()
            instance._close_ownership_sender(device)
            process.join(timeout=15.0)
            events = instance._drain_ownership_events(
                device, wait_timeout=2.0
            )
            result['exitcode'] = process.exitcode
            result['events'] = events
        except BaseException as exc:  # make thread failures visible to pytest
            errors.append(exc)

    thread = threading.Thread(target=launch, name='FlaskRequestThreadProbe')
    thread.start()
    thread.join(timeout=20.0)

    assert thread.is_alive() is False
    assert errors == []
    assert result['exitcode'] == 0
    assert result['events'][0]['start_method'] == 'spawn'
    assert result['events'][0]['runtime_config'] == _config()['runtime']
    assert device['ownership_start_method'] == 'spawn'
    instance._close_ownership_channel(device)


def test_worker_ownership_race_refuses_and_preserves_external_shm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    device_class_id = int(uuid.uuid4().int % 1_000_000_000)
    config['devices'][0]['device_class_id'] = device_class_id
    config['global_settings']['ownership_handshake_timeout'] = 10.0
    instance = DeviceStarter(
        config=config,
        enable_device_manager=False,
        enable_policy_runner=False,
        install_signal_handlers=False,
    )
    device = instance.devices[0]
    name = f'BaseDevice_{device_class_id}_data'
    external = shared_memory.SharedMemory(create=True, size=64, name=name)
    marker = b'external-owner'
    external.buf[: len(marker)] = marker

    # Simulate the exact TOCTOU window: the controller's preflight missed the
    # segment, but it exists before the spawned worker's ownership claim.
    monkeypatch.setattr(instance, '_probe_shared_memory', _missing_shm)
    try:
        with pytest.raises(DeviceConflictError, match='ownership conflict'):
            instance.start_device('BaseDevice', device_class_id)

        assert bytes(external.buf[: len(marker)]) == marker
        attached = shared_memory.SharedMemory(name=name)
        try:
            assert bytes(attached.buf[: len(marker)]) == marker
        finally:
            attached.close()
        assert device['ownership_status'] == 'conflict'
        assert device['desired_state'] == 'stopped'
        assert 'existing segment was not removed' in device['last_error']
    finally:
        process = device.get('process')
        if process is not None:
            process.join(timeout=5.0)
        instance._close_ownership_channel(device)
        external.close()
        external.unlink()
