"""Hardware-free tests for the runtime and inference dependency chain."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading
import time
import types
from typing import Any, Dict, List

import pytest

from incubator.control_panel.inference_control import InferenceControlConsumer
from incubator.control_panel.recording import RecordingCoordinator, RecordingError
from incubator.control_panel.workflow import (
    WorkflowConflictError,
    WorkflowError,
    WorkflowLifecycle,
    _ExclusiveSharedMemoryGuard,
    _action_executor_worker,
    _manager_worker,
)


def _config(checkpoint: Path) -> Dict[str, Any]:
    return {
        'devices': [
            {
                'class': 'BaseDevice',
                'device_class_id': 7,
                'enabled': True,
            }
        ],
        'device_manager': {
            'type': 'base',
            'params': {'buffer_size': 10},
        },
        'action_executor': {
            'type': 'base_policy_runner',
            'robot_control_shms': [],
            'signal_files': {
                'policy_runner_reset_command': '/tmp/fake-reset',
            },
        },
        'policy': {
            'class': 'OpenPiUmiPolicy',
            'params': {
                'pi_config_name': 'fixture_pi05',
                'ckpt_dir': str(checkpoint),
            },
            'infer_rollout': {
                'enabled': True,
                'required': True,
                'ready_timeout': 1.0,
                'shutdown_timeout': 1.0,
            },
        },
    }


def _device(*, ready: bool) -> Dict[str, Any]:
    return {
        'key': 'BaseDevice:7',
        'enabled': True,
        'ready': ready,
    }


class FakeConnection:
    def __init__(self, role: str) -> None:
        self.events = [{'event': 'ready', 'role': role}]
        self.closed = False

    def poll(self) -> bool:
        return bool(self.events)

    def recv(self) -> Dict[str, Any]:
        return self.events.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    _next_pid = 8000

    def __init__(self, role: str, events: List[str]) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.role = role
        self.events = events
        self.alive = True
        self.exitcode = None

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.events.append(f'{self.role}:terminate')
        self.alive = False
        self.exitcode = -15

    def join(self, timeout: float = 0.0) -> None:
        self.events.append(f'{self.role}:join')

    def kill(self) -> None:
        self.events.append(f'{self.role}:kill')
        self.alive = False
        self.exitcode = -9


class FakeInferenceProcess:
    # Keep fake process groups outside Linux's configurable PID range so tests
    # can never signal an unrelated host process through os.killpg().
    _next_pid = 1 << 30

    def __init__(self, events: List[str]) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.events = events
        self.returncode = None

    def poll(self) -> Any:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.events.append(f'inference:signal:{sig}')
        self.returncode = -sig

    def wait(self, timeout: float = 0.0) -> int:
        self.events.append('inference:wait')
        return int(self.returncode or 0)


class FakeRecordingCoordinator:
    def __init__(self, root: Path, events: List[str]) -> None:
        self.root = root
        self.events = events
        self.run_id = 'recording-run'
        self.control_dir = root / self.run_id
        self.owner_pid = None
        self.ready = True
        self.healthy = True
        self.status = 'armed'
        self.recorder_ready = True
        self.error = None
        self.pending_command: Dict[str, Any] | None = None

    def prepare_run(self) -> Path:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        return self.control_dir

    def bind_owner(self, owner_pid: int) -> None:
        self.owner_pid = owner_pid

    def wait_until_ready(self, process: Any, timeout: float) -> Any:
        del process, timeout
        return (True, '') if self.ready else (False, 'recorder failed')

    def state(self) -> Dict[str, Any]:
        healthy = self.healthy
        return {
            'configured': True,
            'enabled': True,
            'required': True,
            'status': self.status if healthy else 'error',
            'active': False,
            'recorder_ready': self.recorder_ready if healthy else False,
            'error': self.error if healthy else 'recorder lost',
            'run_id': self.run_id,
            'revision': 0,
            'reference': None,
            'episode': None,
            'segments': [],
            'checkpoints': [],
            'pending_command': self.pending_command,
            'capabilities': {},
        }

    def is_healthy(self, state: Dict[str, Any]) -> bool:
        return RecordingCoordinator.is_healthy(state)

    def request_shutdown(self, timeout: float) -> Any:
        del timeout
        self.events.append('recording:shutdown')
        return True, None

    def submit_action(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        action = str(args[0]) if args else str(kwargs['action'])
        self.pending_command = {
            'command_id': 'fake',
            'action': action,
            'status': 'pending',
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        return {'command_id': 'fake', 'status': 'pending'}


class CaptureConnection:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.closed = False

    def send(self, payload: Dict[str, Any]) -> None:
        self.events.append(dict(payload))

    def close(self) -> None:
        self.closed = True


def _workflow(
    tmp_path: Path,
    *,
    registry_ok: bool = True,
    path_exists: Any = None,
    shm_identity: Any = None,
    shm_unlink: Any = None,
    events: List[str] | None = None,
    config: Dict[str, Any] | None = None,
    test_inference_mode: bool = False,
    recording_enabled_override: bool | None = None,
    registry_checker: Any = None,
) -> WorkflowLifecycle:
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / 'params').mkdir(exist_ok=True)
    repo_root = tmp_path / 'repo'
    (repo_root / 'consumer').mkdir(parents=True)
    (repo_root / 'consumer' / 'openpi_policy_infer.py').touch()
    inference_python = tmp_path / 'openpi-python'
    inference_python.touch()
    recording_python = tmp_path / 'recording-python'
    recording_python.touch()
    recording_events = events if events is not None else []
    fake_recording = FakeRecordingCoordinator(
        tmp_path / 'recording-runs', recording_events
    )
    workflow = WorkflowLifecycle(
        'fixture',
        config or _config(checkpoint),
        test_inference_mode=test_inference_mode,
        recording_enabled_override=recording_enabled_override,
        repo_root=repo_root,
        device_python=Path(sys.executable),
        inference_python=inference_python,
        recording_python=recording_python,
        recording_run_root=tmp_path / 'recording-runs',
        inference_control_run_root=tmp_path / 'inference-control-runs',
        recording_profile_checker=lambda path: (True, ''),
        recording_coordinator_factory=lambda root: fake_recording,
        registry_checker=registry_checker
        or (
            lambda *args: (
                registry_ok,
                '' if registry_ok else 'OpenPI registry does not contain fixture_pi05',
            )
        ),
        path_exists=path_exists or (lambda path: Path(path).exists()),
        shm_identity=shm_identity or (lambda name: None),
        **({'shm_unlink': shm_unlink} if shm_unlink is not None else {}),
    )
    process_events = events if events is not None else []

    def spawn(role: str, process_config: Dict[str, Any]) -> Any:
        del process_config
        return FakeProcess(role, process_events), FakeConnection(role)

    workflow._spawn_runtime_worker = spawn  # type: ignore[method-assign]
    workflow._fake_recording = fake_recording  # type: ignore[attr-defined]
    return workflow


class FakeSharedMemory:
    def __init__(
        self,
        name: str,
        identity: tuple[int, int],
        names: Dict[str, tuple[int, int]],
    ) -> None:
        self.name = name
        self._name = f'/{name}'
        self.size = 64
        self.identity = identity
        self.names = names
        self.buf = memoryview(bytearray(self.size))
        self.closed = False
        self.unlink_count = 0

    def close(self) -> None:
        self.closed = True

    def unlink(self) -> None:
        self.unlink_count += 1
        self.names.pop(self.name, None)


def _fake_shm_guard(
    names: Dict[str, tuple[int, int]],
) -> tuple[_ExclusiveSharedMemoryGuard, List[FakeSharedMemory], List[str]]:
    handles: List[FakeSharedMemory] = []
    detached: List[str] = []

    def factory(*, create: bool, size: int, name: str) -> FakeSharedMemory:
        assert create is True
        assert size > 0
        if name in names:
            raise FileExistsError(name)
        identity = (1, len(handles) + 100)
        names[name] = identity
        handle = FakeSharedMemory(name, identity, names)
        handles.append(handle)
        return handle

    guard = _ExclusiveSharedMemoryGuard(
        CaptureConnection(),
        shm_factory=factory,
        identity=lambda name: names.get(name),
        handle_identity=lambda handle: handle.identity,
        tracker_unregister=detached.append,
    )
    return guard, handles, detached


def test_runtime_start_is_locked_until_every_enabled_device_is_ready(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)

    with pytest.raises(WorkflowConflictError, match='设备未就绪'):
        workflow.start_runtime([_device(ready=False)])

    stopped = workflow.state([_device(ready=False)])
    assert stopped['runtime']['status'] == 'stopped'
    assert stopped['runtime']['can_start'] is True
    assert stopped['runtime']['blockers'] == ['设备未就绪：BaseDevice:7']
    assert workflow.required_device_keys() == {'BaseDevice:7'}


def test_observe_uses_cached_snapshot_while_explicit_transition_holds_lock(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    seeded = workflow.start_runtime([_device(ready=True)])
    lock_entered = threading.Event()
    release_lock = threading.Event()

    def hold_transition_lock() -> None:
        with workflow._lock:
            lock_entered.set()
            assert release_lock.wait(timeout=5.0)

    thread = threading.Thread(target=hold_transition_lock)
    thread.start()
    try:
        assert lock_entered.wait(timeout=2.0)
        started = time.perf_counter()
        observed = workflow.observe([_device(ready=True)])
        elapsed = time.perf_counter() - started

        assert elapsed < 0.25
        assert observed == seeded
        assert observed is not workflow._last_observation
    finally:
        release_lock.set()
        thread.join(timeout=2.0)
        workflow.stop_all()

    assert thread.is_alive() is False


def _three_camera_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config['devices'] = [
        {'class': 'WebCameraOpenPi', 'device_class_id': 0, 'enabled': True},
        {'class': 'WebCameraOpenPi', 'device_class_id': 4, 'enabled': True},
        {'class': 'WebCameraOpenPi', 'device_class_id': 2, 'enabled': True},
        {'class': 'RizonRobot', 'device_class_id': 2, 'enabled': True},
    ]
    config['device_manager']['master_device_class_id'] = 2
    config['policy']['obs'] = {
        'devices': [
            'WebCameraOpenPi_0',
            'WebCameraOpenPi_4',
            'RizonRobot_2',
        ]
    }
    config['policy']['infer_rollout'].update(
        {
            'head_camera': 'WebCameraOpenPi_2',
            'left_camera': 'WebCameraOpenPi_0',
            'right_camera': 'WebCameraOpenPi_4',
        }
    )
    return config


def _camera_devices(*, head_ready: bool) -> List[Dict[str, Any]]:
    return [
        {'key': 'WebCameraOpenPi:0', 'ready': True},
        {'key': 'WebCameraOpenPi:4', 'ready': True},
        {'key': 'WebCameraOpenPi:2', 'ready': head_ready},
        {'key': 'RizonRobot:2', 'ready': True},
    ]


def test_recording_only_head_is_optional_and_not_runtime_master(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    workflow.config = _three_camera_config(workflow.config)
    original_config = copy.deepcopy(workflow.config)

    without_head, without_head_keys = workflow._runtime_process_config(
        _camera_devices(head_ready=False)
    )
    assert workflow._not_ready_device_keys(_camera_devices(head_ready=False)) == []
    assert without_head_keys == {
        'WebCameraOpenPi:0',
        'WebCameraOpenPi:4',
        'RizonRobot:2',
    }
    assert [
        workflow._config_device_key(device) for device in without_head['devices']
    ] == [
        'WebCameraOpenPi:0',
        'WebCameraOpenPi:4',
        'RizonRobot:2',
    ]
    assert without_head['device_manager']['master_device_class_id'] == 2

    workflow._runtime_device_keys = without_head_keys
    assert workflow._recording_conversion_run_config()['head_camera'] is None

    with_head, with_head_keys = workflow._runtime_process_config(
        _camera_devices(head_ready=True)
    )
    assert with_head_keys == {
        'WebCameraOpenPi:0',
        'WebCameraOpenPi:4',
        'WebCameraOpenPi:2',
        'RizonRobot:2',
    }
    assert with_head['device_manager']['master_device_class_id'] == 3
    assert workflow.config == original_config


def test_policy_observed_head_is_a_required_runtime_dependency(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    workflow.config = _three_camera_config(workflow.config)
    workflow.config['policy']['obs']['devices'].insert(0, 'WebCameraOpenPi_2')

    assert workflow._optional_head_device_key() is None
    assert workflow._not_ready_device_keys(_camera_devices(head_ready=False)) == [
        'WebCameraOpenPi:2'
    ]


def test_configured_yam_head_arm_is_required_without_policy_observation(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    workflow.config['devices'].append(
        {'class': 'YamHeadRobot', 'device_class_id': 0, 'enabled': True}
    )
    base = _device(ready=True)
    head_stopped = {'key': 'YamHeadRobot:0', 'ready': False}
    head_ready = {'key': 'YamHeadRobot:0', 'ready': True}

    assert workflow.required_device_keys() == {
        'BaseDevice:7',
        'YamHeadRobot:0',
    }
    assert workflow._not_ready_device_keys([base, head_stopped]) == [
        'YamHeadRobot:0'
    ]
    without_head, without_head_keys = workflow._runtime_process_config(
        [base, head_stopped]
    )
    assert without_head_keys == {'BaseDevice:7', 'YamHeadRobot:0'}
    assert [
        workflow._config_device_key(device) for device in without_head['devices']
    ] == ['BaseDevice:7', 'YamHeadRobot:0']

    with_head, with_head_keys = workflow._runtime_process_config(
        [base, head_ready]
    )
    assert with_head_keys == {'BaseDevice:7', 'YamHeadRobot:0'}
    assert [
        workflow._config_device_key(device) for device in with_head['devices']
    ] == ['BaseDevice:7', 'YamHeadRobot:0']

    workflow.config['policy']['obs'] = {'devices': ['YamHeadRobot_0']}
    assert workflow.required_device_keys() == {
        'BaseDevice:7',
        'YamHeadRobot:0',
    }


def test_runtime_device_keys_clear_after_stop_and_failed_start(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    workflow.start_runtime([_device(ready=True)])
    assert workflow.state([_device(ready=True)])['runtime']['device_keys'] == [
        'BaseDevice:7'
    ]
    workflow.stop_runtime([_device(ready=True)])
    assert workflow._runtime_device_keys is None

    attempts = 0

    def fail_second_spawn(role: str, process_config: Dict[str, Any]) -> Any:
        nonlocal attempts
        del process_config
        attempts += 1
        if attempts == 2:
            raise OSError('executor spawn failed')
        return FakeProcess(role, []), FakeConnection(role)

    workflow._spawn_runtime_worker = fail_second_spawn  # type: ignore[method-assign]
    with pytest.raises(OSError, match='executor spawn failed'):
        workflow.start_runtime([_device(ready=True)])
    assert workflow._runtime_device_keys is None


def test_normal_runtime_stop_cleans_matching_claimed_shm(tmp_path: Path) -> None:
    names: Dict[str, tuple[int, int]] = {}
    unlinked: List[str] = []

    def unlink(name: str) -> None:
        unlinked.append(name)
        names.pop(name)

    workflow = _workflow(
        tmp_path,
        shm_identity=names.get,
        shm_unlink=unlink,
    )
    workflow.start_runtime([_device(ready=True)])
    names['RizonRobot_2_control'] = (15, 501)
    workflow._executor_connection.events.append(
        {
            'event': 'shm_claimed',
            'name': 'RizonRobot_2_control',
            'dev': 15,
            'ino': 501,
        }
    )

    workflow.stop_runtime([_device(ready=True)])

    assert unlinked == ['RizonRobot_2_control']
    assert names == {}
    assert workflow._runtime_shm_claims['action_executor'] == {}


def test_manager_worker_uses_composed_config_not_missing_hydra_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    class FakeManager:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def start(self) -> bool:
            return True

    class FakeAffinity:
        def apply(self, role: str) -> None:
            captured['affinity_role'] = role

    class FakeAffinitySettings:
        @classmethod
        def from_runtime(cls, runtime: Any) -> FakeAffinity:
            assert runtime is None
            return FakeAffinity()

    fake_module = types.ModuleType('manager')
    fake_module.MANAGER_CLASSES = {'base': FakeManager}
    monkeypatch.setitem(sys.modules, 'manager', fake_module)
    monkeypatch.setattr(
        'incubator.control_panel.workflow.CpuAffinitySettings',
        FakeAffinitySettings,
    )
    config = _config(tmp_path / 'checkpoint')
    missing_resolved = tmp_path / 'missing' / 'config_resolved.yaml'
    config['device_manager']['params']['config_path'] = str(missing_resolved)
    connection = CaptureConnection()

    _manager_worker(config, connection)

    assert missing_resolved.exists() is False
    assert 'config_path' not in captured
    assert captured['config'] == config
    assert captured['buffer_size'] == 10
    assert captured['affinity_role'] == 'device_manager'
    assert connection.events == [
        {'event': 'ready', 'role': 'device_manager'}
    ]
    assert connection.closed is True


def test_action_executor_worker_uses_shared_paced_runner_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: List[str] = []
    captured: Dict[str, Any] = {}

    class FakeAffinity:
        def apply(self, role: str) -> None:
            events.append(f'affinity:{role}')

    class FakeAffinitySettings:
        @classmethod
        def from_runtime(cls, runtime: Any) -> FakeAffinity:
            assert runtime is None
            return FakeAffinity()

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def connect(self) -> bool:
            events.append('connect')
            return True

        def run_connected(self) -> bool:
            events.append('run_connected')
            return True

        def stop(self) -> None:
            events.append('stop')

    fake_module = types.ModuleType('manager')
    fake_module.MANAGER_CLASSES = {'base_policy_runner': FakeRunner}
    monkeypatch.setitem(sys.modules, 'manager', fake_module)
    monkeypatch.setattr(
        'incubator.control_panel.workflow.CpuAffinitySettings',
        FakeAffinitySettings,
    )
    config = _config(tmp_path / 'checkpoint')
    config['action_executor']['execution_fps'] = 30.0
    config['action_executor']['execution_clock'] = 'sim_time'
    config['action_executor']['sim_clock_poll_interval_s'] = 0.002
    connection = CaptureConnection()

    _action_executor_worker(config, connection)

    assert events == [
        'affinity:policy_runner',
        'connect',
        'run_connected',
        'stop',
    ]
    assert captured['execution_fps'] == 30.0
    assert captured['execution_clock'] == 'sim_time'
    assert captured['sim_clock_poll_interval_s'] == 0.002
    assert captured['config'] == config
    assert connection.events == [
        {'event': 'ready', 'role': 'action_executor'}
    ]
    assert connection.closed is True


def test_shared_runner_loop_waits_at_execution_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import manager.base_policy_runner as policy_runner_module

    runner = policy_runner_module.BasePolicyRunner.__new__(
        policy_runner_module.BasePolicyRunner
    )
    runner.execution_fps = 30.0
    runner.execution_clock = 'wall_time'
    events: List[Any] = []

    def update_once(*, dispatch_ready: bool) -> None:
        assert dispatch_ready is True
        events.append('update')
        runner.running = False

    def timing(interval: float) -> Any:
        events.append(('interval', interval))
        return lambda: events.append('wait')

    runner._update_execution = update_once  # type: ignore[method-assign]
    monkeypatch.setattr(policy_runner_module, 'precise_loop_timing', timing)

    assert runner.run_connected() is True
    assert events == [('interval', pytest.approx(1.0 / 30.0)), 'update', 'wait']


def test_runtime_shm_guard_never_unlinks_an_existing_external_segment() -> None:
    names = {'policy_data': (9, 9)}
    guard, handles, detached = _fake_shm_guard(names)

    with pytest.raises(FileExistsError):
        guard.create('policy_data', 64)
    guard.close()

    assert names == {'policy_data': (9, 9)}
    assert handles == []
    assert detached == []


def test_runtime_shm_guard_cleans_only_its_unchanged_inode() -> None:
    names: Dict[str, tuple[int, int]] = {}
    guard, handles, detached = _fake_shm_guard(names)

    channel = guard.create('policy_data', 64)
    channel.close()
    guard.close()

    assert names == {}
    assert handles[0].unlink_count == 1
    assert handles[0].closed is True
    assert detached == []


def test_runtime_shm_guard_does_not_unlink_a_replacement_inode() -> None:
    names: Dict[str, tuple[int, int]] = {}
    guard, handles, detached = _fake_shm_guard(names)

    guard.create('policy_data', 64)
    names['policy_data'] = (5, 500)
    guard.close()

    assert names == {'policy_data': (5, 500)}
    assert handles[0].unlink_count == 0
    assert handles[0].closed is True
    assert detached == ['/policy_data']


def test_runtime_preflight_blocks_existing_summary_or_policy_shm(
    tmp_path: Path,
) -> None:
    workflow = _workflow(
        tmp_path,
        shm_identity=lambda name: (7, 70) if name == 'policy_data' else None,
    )

    state = workflow.state([_device(ready=True)])

    assert state['runtime']['can_start'] is False
    assert '共享内存已被占用：policy_data' in state['runtime']['blockers']
    with pytest.raises(WorkflowConflictError, match='policy_data'):
        workflow.start_runtime([_device(ready=True)])


def test_runtime_reports_ready_only_after_both_workers_report_ready(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)

    state = workflow.start_runtime([_device(ready=True)])

    assert state['runtime']['status'] == 'running'
    assert state['runtime']['ready'] is True
    assert state['runtime']['device_manager']['ready'] is True
    assert state['runtime']['action_executor']['ready'] is True
    assert state['inference']['can_start'] is True


def test_inference_preflight_is_fail_closed_for_registry_and_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / 'checkpoint'

    def exists(path: Path) -> bool:
        return Path(path) != checkpoint

    workflow = _workflow(
        tmp_path,
        registry_ok=False,
        path_exists=exists,
    )
    workflow.start_runtime([_device(ready=True)])
    state = workflow.state([_device(ready=True)])

    assert state['inference']['can_start'] is False
    assert any('Checkpoint does not exist' in item for item in state['inference']['blockers'])
    assert any('registry does not contain fixture_pi05' in item for item in state['inference']['blockers'])
    with pytest.raises(WorkflowConflictError):
        workflow.start_inference([_device(ready=True)])


def test_inference_preflight_checks_effective_checkpoint_and_control_dimension(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Any, ...]] = []
    config = _config(tmp_path / 'checkpoint')
    config['policy']['controls'] = [
        {
            'endpoint': {'device': 'LeftRobot', 'device_class_id': 0, 'port': 'command'},
            'shape': [8],
        },
        {
            'endpoint': {'device': 'RightRobot', 'device_class_id': 0, 'port': 'command'},
            'shape': [8],
        },
    ]

    def check(*args: Any) -> tuple[bool, str]:
        calls.append(args)
        return True, ''

    workflow = _workflow(tmp_path, config=config, registry_checker=check)
    workflow.start_runtime([_device(ready=True)])

    assert calls
    assert calls[-1][3:] == (
        'fixture_pi05',
        (tmp_path / 'checkpoint').resolve(),
        16,
    )

    override = tmp_path / 'override'
    (override / 'params').mkdir(parents=True)
    workflow.set_checkpoint_override(override)
    workflow.state([_device(ready=True)])

    assert calls[-1][4:] == (override.resolve(), 16)


def test_checkpoint_free_mode_bypasses_model_preflight_and_injects_mock_env(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / 'missing-checkpoint'
    config = _config(checkpoint)
    config['devices'] = [
        {
            'class': 'RizonRobot',
            'device_class_id': 2,
            'enabled': True,
            'state_feedback': {
                'fps': 120.0,
                'format': {
                    'shape': [44],
                    'dtype': 'float64',
                    'type': ['linear:44'],
                },
            },
            'control': {
                'type': 'eef_abs',
                'format': {
                    'shape': [16],
                    'dtype': 'float64',
                    'type': ['linear:16'],
                },
            },
            'gripper': {'value_mode': 'width_m'},
        }
    ]
    config['policy']['controls'] = [
        {
            'endpoint': {
                'device': 'RizonRobot',
                'device_class_id': 2,
                'port': 'command',
            },
            'shape': [16],
            'dtype': 'float64',
        }
    ]
    config['policy']['obs'] = {'devices': ['RizonRobot_2']}
    workflow = _workflow(
        tmp_path,
        config=config,
        registry_ok=False,
        test_inference_mode=True,
        recording_enabled_override=False,
    )
    device = {'key': 'RizonRobot:2', 'enabled': True, 'ready': True}
    calls: List[Dict[str, Any]] = []
    fake_inference = FakeInferenceProcess([])

    def popen(command: List[str], **kwargs: Any) -> FakeInferenceProcess:
        calls.append({'command': list(command), **kwargs})
        return fake_inference

    workflow._popen_factory = popen
    workflow.start_runtime([device])
    before = workflow.state([device])
    assert before['inference']['can_start'] is True
    assert not any('Checkpoint' in item for item in before['inference']['blockers'])
    assert not any('registry' in item for item in before['inference']['blockers'])

    state = workflow.start_inference([device])
    assert state['inference']['mode'] == 'test'
    assert state['inference']['test_mode'] is True
    assert calls[0]['env']['SUPERINFERENCE_CONTROL_PANEL_TEST_MODE'] == '1'
    assert calls[0]['env']['SUPERINFERENCE_INFERENCE_CONTROL_RUN_ID']
    assert Path(calls[0]['env']['SUPERINFERENCE_INFERENCE_CONTROL_DIR']).parent == (
        tmp_path / 'inference-control-runs'
    )
    rollout = json.loads(
        calls[0]['env']['SUPERINFERENCE_CONTROL_PANEL_POLICY_OVERRIDES']
    )['infer_rollout']
    assert rollout['enabled'] is False
    assert 'SUPERINFERENCE_CHECKPOINT_OVERRIDE' not in calls[0]['env']
    assert not any(item.startswith('policy.params.ckpt_dir=') for item in calls[0]['command'])

    reset_state = state['inference']['reset']
    assert reset_state['can'] is True
    reset_result = workflow.inference_reset(idempotency_key='reset-0001')
    assert reset_result['command']['status'] == 'pending'
    after_reset = workflow.state([device])
    assert after_reset['runtime']['ready'] is True
    assert after_reset['inference']['running'] is True
    assert after_reset['inference']['reset']['can'] is False
    assert after_reset['inference']['reset']['pending_command'] is not None
    assert workflow.inference_process is fake_inference


def test_pause_and_resume_keep_runtime_and_loaded_inference_processes(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path, recording_enabled_override=False)
    fake_inference = FakeInferenceProcess([])
    workflow._popen_factory = lambda *_args, **_kwargs: fake_inference
    device = _device(ready=True)
    workflow.start_runtime([device])
    workflow.start_inference([device])
    manager = workflow.device_manager_process
    executor = workflow.action_executor_process
    coordinator = workflow.inference_control_coordinator
    assert coordinator is not None
    consumer = InferenceControlConsumer(
        coordinator.control_dir,
        str(coordinator.run_id),
        owner_pid=fake_inference.pid,
    )

    pause = workflow.inference_control("pause", idempotency_key="pause-0001")
    assert pause["command"]["status"] == "pending"
    paused = True
    assert consumer.process_commands(
        lambda command: None,
        state_provider=lambda: {"inference_paused": paused},
    ) == 1
    state = workflow.state([device])
    assert state["inference"]["reset"]["paused"] is True
    assert state["inference"]["reset"]["can_resume"] is True
    assert state["inference"]["reset"]["can_pause"] is False

    resume = workflow.inference_control("resume", idempotency_key="resume-0001")
    assert resume["command"]["status"] == "pending"
    paused = False
    assert consumer.process_commands(
        lambda command: None,
        state_provider=lambda: {"inference_paused": paused},
    ) == 1
    state = workflow.state([device])
    assert state["inference"]["reset"]["paused"] is False
    assert state["inference"]["reset"]["can_pause"] is True
    assert state["inference"]["reset"]["can_resume"] is False
    assert workflow.inference_process is fake_inference
    assert workflow.device_manager_process is manager
    assert workflow.action_executor_process is executor
    assert state["runtime"]["ready"] is True
    assert state["inference"]["running"] is True


def test_inference_uses_fixed_profile_and_reverse_dependency_stop_order(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    popen_calls: List[Dict[str, Any]] = []

    def popen(command: List[str], **kwargs: Any) -> FakeInferenceProcess:
        popen_calls.append({'command': command, **kwargs})
        return fake_inference

    workflow._popen_factory = popen
    workflow.start_runtime([_device(ready=True)])
    state = workflow.start_inference([_device(ready=True)])
    assert state['inference']['running'] is True

    call = popen_calls[0]
    assert call['command'] == [
        str(workflow.inference_python),
        str(workflow.repo_root / 'consumer' / 'openpi_policy_infer.py'),
        '--config-name',
        'fixture',
    ]
    assert call['env']['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false'
    assert call['env']['SUPERINFERENCE_INFERENCE_CONTROL_RUN_ID']
    assert Path(call['env']['SUPERINFERENCE_INFERENCE_CONTROL_DIR']).parent == (
        tmp_path / 'inference-control-runs'
    )
    assert call['env']['SUPERINFERENCE_RUN_ID'] == 'recording-run'
    assert Path(call['env']['SUPERINFERENCE_RECORDING_CONTROL_DIR']) == (
        tmp_path / 'recording-runs' / 'recording-run'
    )
    assert call['env']['SUPERINFERENCE_RECORDER_PYTHON'] == str(
        workflow.recording_python
    )
    expected_output = (workflow.repo_root / 'inference_rollout_data').resolve()
    assert call['env']['SUPERINFERENCE_RECORDING_OUTPUT_DIR'] == str(
        expected_output
    )
    rollout_override = json.loads(
        call['env']['SUPERINFERENCE_CONTROL_PANEL_POLICY_OVERRIDES']
    )['infer_rollout']
    assert rollout_override['output_dir'] == str(expected_output)
    assert workflow.config['policy']['infer_rollout'].get('output_dir') is None
    assert call['start_new_session'] is True

    # The fake is not a real process group; exercise the portable signal path.
    workflow._inference_new_session = False
    workflow.stop_all()

    recording_index = events.index('recording:shutdown')
    inference_index = events.index(f'inference:signal:{signal.SIGINT}')
    executor_index = events.index('action_executor:terminate')
    manager_index = events.index('device_manager:terminate')
    assert recording_index < inference_index < executor_index < manager_index
    assert workflow.has_active_processes() is False


def test_checkpoint_override_uses_hydra_env_and_preserves_runtime(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    default_path = workflow.config['policy']['params']['ckpt_dir']
    override = tmp_path / 'override space,[x]=${literal}'
    (override / 'params').mkdir(parents=True)
    fake_inference = FakeInferenceProcess(events)
    popen_calls: List[Dict[str, Any]] = []

    def popen(command: List[str], **kwargs: Any) -> FakeInferenceProcess:
        popen_calls.append({'command': list(command), **kwargs})
        return fake_inference

    workflow._popen_factory = popen
    workflow.start_runtime([_device(ready=True)])
    manager_process = workflow.device_manager_process
    workflow.set_checkpoint_override(override)

    state = workflow.state([_device(ready=True)])
    checkpoint = state['inference']['checkpoint']
    assert state['runtime']['ready'] is True
    assert workflow.device_manager_process is manager_process
    assert checkpoint['default_path'] == str(Path(default_path).resolve())
    assert checkpoint['override_path'] == str(override.resolve())
    assert checkpoint['effective_path'] == str(override.resolve())
    assert checkpoint['valid'] is True
    assert workflow.config['policy']['params']['ckpt_dir'] == default_path

    workflow.start_inference([_device(ready=True)])
    call = popen_calls[0]
    assert call['command'][-1] == (
        'policy.params.ckpt_dir='
        '${oc.env:SUPERINFERENCE_CHECKPOINT_OVERRIDE}'
    )
    assert call['env']['SUPERINFERENCE_CHECKPOINT_OVERRIDE'] == str(
        override.resolve()
    )
    with pytest.raises(WorkflowConflictError, match='Stop inference'):
        workflow.set_checkpoint_override(None)

    workflow._inference_new_session = False
    workflow.stop_all()


def test_inference_start_fails_closed_until_recorder_is_armed(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow._fake_recording.ready = False  # type: ignore[attr-defined]
    workflow.start_runtime([_device(ready=True)])

    with pytest.raises(WorkflowError, match='recorder failed'):
        workflow.start_inference([_device(ready=True)])

    assert f'inference:signal:{signal.SIGINT}' in events
    state = workflow.state([_device(ready=True)])
    assert state['inference']['status'] == 'failed'
    assert state['inference']['running'] is False


def test_running_inference_stops_when_required_recorder_loses_readiness(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False

    workflow._fake_recording.healthy = False  # type: ignore[attr-defined]
    state = workflow.state([_device(ready=True)])

    assert state['inference']['status'] == 'failed'
    assert f'inference:signal:{signal.SIGINT}' in events


@pytest.mark.parametrize('status', ['finalizing', 'transitioning'])
def test_episode_transition_keeps_runtime_and_inference_loaded(
    tmp_path: Path,
    status: str,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    coordinator = workflow._fake_recording  # type: ignore[attr-defined]
    coordinator.status = status
    coordinator.recorder_ready = False

    state = workflow.state([_device(ready=True)])

    assert state['runtime']['ready'] is True
    assert state['inference']['status'] == 'running'
    assert state['inference']['running'] is True
    assert state['inference']['reset']['can'] is False
    assert 'recording actions' in state['inference']['reset']['blockers'][0]
    assert fake_inference.returncode is None
    assert not any(event.startswith('inference:signal:') for event in events)


def test_recording_shutdown_spool_error_does_not_block_inference_stop(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False

    def fail_shutdown(timeout: float) -> Any:
        del timeout
        events.append('recording:shutdown-attempt')
        raise RecordingError('corrupt ACK')

    workflow._fake_recording.request_shutdown = fail_shutdown  # type: ignore[attr-defined]
    workflow.stop_inference([_device(ready=True)])

    assert events.index('recording:shutdown-attempt') < events.index(
        f'inference:signal:{signal.SIGINT}'
    )


def test_ready_device_loss_stops_inference_then_runtime(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False

    state = workflow.state([_device(ready=False)])

    assert state['runtime']['status'] == 'failed'
    assert state['runtime']['active'] is False
    assert state['inference']['running'] is False
    inference_index = events.index(f'inference:signal:{signal.SIGINT}')
    executor_index = events.index('action_executor:terminate')
    manager_index = events.index('device_manager:terminate')
    assert inference_index < executor_index < manager_index


def _stale_motion_device(*, fault: bool = False) -> Dict[str, Any]:
    return {
        'key': 'BaseDevice:7',
        'enabled': True,
        'ready': False,
        'running': True,
        'status': 'running',
        'risk': 'motion',
        'safety': {
            'fault': fault,
            'operational': not fault,
            'estop': {'active': False},
        },
    }


def test_pending_reset_tolerates_transient_motion_device_staleness(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(
        tmp_path,
        events=events,
        recording_enabled_override=False,
    )
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False
    workflow.inference_reset(idempotency_key='reset-stale-motion')

    state = workflow.state([_stale_motion_device()])

    assert state['runtime']['status'] == 'running'
    assert state['runtime']['ready'] is True
    assert state['inference']['running'] is True
    assert state['inference']['reset']['pending_command']['action'] == 'reset'
    assert 'home_target' not in state['inference']['reset']['pending_command']
    assert not any(event.startswith('inference:signal:') for event in events)
    assert 'action_executor:terminate' not in events
    assert 'device_manager:terminate' not in events

    coordinator = workflow.inference_control_coordinator
    assert coordinator is not None
    consumer = InferenceControlConsumer(
        coordinator.control_dir,
        str(coordinator.run_id),
        owner_pid=fake_inference.pid,
    )
    assert consumer.process_commands(
        lambda command: None,
        state_provider=lambda: {
            'inference_paused': True,
            'reset_pending': True,
        },
    ) == 1

    acknowledged = workflow.state([_stale_motion_device()])

    assert acknowledged['runtime']['status'] == 'running'
    assert acknowledged['runtime']['ready'] is True
    assert acknowledged['inference']['running'] is True
    assert acknowledged['inference']['reset']['pending_command'] is None
    assert acknowledged['inference']['reset']['reset_pending'] is True


@pytest.mark.parametrize('action', ['episode_accept', 'episode_discard'])
def test_pending_recording_reset_keeps_runtime_and_inference_loaded(
    tmp_path: Path,
    action: str,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False
    workflow.recording_action(
        action,
        episode_id='episode-fixture',
        expected_checkpoint_revision=0,
        idempotency_key=f'{action}-stale-motion',
    )

    state = workflow.state([_stale_motion_device()])

    assert state['runtime']['status'] == 'running'
    assert state['runtime']['ready'] is True
    assert state['inference']['running'] is True
    assert state['recording']['pending_command']['action'] == action
    assert not any(event.startswith('inference:signal:') for event in events)
    assert 'action_executor:terminate' not in events
    assert 'device_manager:terminate' not in events


def test_pending_reset_does_not_mask_explicit_motion_device_fault(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(
        tmp_path,
        events=events,
        recording_enabled_override=False,
    )
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False
    workflow.inference_reset(idempotency_key='reset-faulted-motion')

    state = workflow.state([_stale_motion_device(fault=True)])

    assert state['runtime']['status'] == 'failed'
    assert state['runtime']['active'] is False
    assert state['inference']['running'] is False
    assert any(event.startswith('inference:signal:') for event in events)


def test_expired_pending_reset_does_not_mask_motion_device_staleness(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(
        tmp_path,
        events=events,
        recording_enabled_override=False,
    )
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False
    workflow.inference_reset(idempotency_key='reset-expired-motion')
    coordinator = workflow.inference_control_coordinator
    assert coordinator is not None
    command = next(iter(coordinator._commands.values()))
    command['created_at'] = '2020-01-01T00:00:00+00:00'

    state = workflow.state([_stale_motion_device()])

    assert state['runtime']['status'] == 'failed'
    assert state['inference']['running'] is False


def test_pending_pause_does_not_mask_motion_device_staleness(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(
        tmp_path,
        events=events,
        recording_enabled_override=False,
    )
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False
    coordinator = workflow.inference_control_coordinator
    assert coordinator is not None
    coordinator.submit_pause(idempotency_key='pause-stale-motion')

    state = workflow.state([_stale_motion_device()])

    assert state['runtime']['status'] == 'failed'
    assert state['inference']['running'] is False


def test_observe_reports_dependency_loss_before_slow_safety_stop_finishes(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False

    stop_entered = threading.Event()
    allow_stop = threading.Event()
    stop_finished = threading.Event()
    original_stop = workflow._stop_runtime_locked

    def slow_stop(*, stop_inference: bool) -> None:
        stop_entered.set()
        if not allow_stop.wait(timeout=2.0):
            raise AssertionError('test did not release asynchronous safety stop')
        try:
            original_stop(stop_inference=stop_inference)
        finally:
            stop_finished.set()

    workflow._stop_runtime_locked = slow_stop  # type: ignore[method-assign]
    try:
        started = time.perf_counter()
        observed = workflow.observe([_device(ready=False)])
        elapsed = time.perf_counter() - started

        assert elapsed < 0.25
        assert observed['runtime']['status'] == 'stopping'
        assert observed['runtime']['desired_state'] == 'stopped'
        assert observed['runtime']['active'] is True
        assert observed['runtime']['ready'] is False
        assert observed['runtime']['can_start'] is False
        assert observed['inference']['status'] == 'stopping'
        assert observed['inference']['desired_state'] == 'stopped'
        assert observed['inference']['running'] is True
        assert 'devices not ready' in observed['runtime']['reason']
        assert stop_entered.wait(timeout=1.0)
    finally:
        allow_stop.set()

    assert stop_finished.wait(timeout=2.0)
    stopped = workflow.observe([_device(ready=False)])
    assert stopped['runtime']['status'] == 'failed'
    assert stopped['runtime']['active'] is False
    assert stopped['inference']['status'] == 'failed'
    assert stopped['inference']['running'] is False
    inference_index = events.index(f'inference:signal:{signal.SIGINT}')
    executor_index = events.index('action_executor:terminate')
    manager_index = events.index('device_manager:terminate')
    assert inference_index < executor_index < manager_index


def test_pending_safety_stop_keeps_manager_controls_responsive_and_skips_recorder_probe(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    fake_inference = FakeInferenceProcess(events)
    workflow._popen_factory = lambda *args, **kwargs: fake_inference
    workflow.start_runtime([_device(ready=True)])
    workflow.start_inference([_device(ready=True)])
    workflow._inference_new_session = False

    stop_entered = threading.Event()
    allow_stop = threading.Event()
    stop_finished = threading.Event()
    original_stop = workflow._stop_runtime_locked

    def slow_stop(*, stop_inference: bool) -> None:
        stop_entered.set()
        if not allow_stop.wait(timeout=2.0):
            raise AssertionError('test did not release asynchronous safety stop')
        try:
            original_stop(stop_inference=stop_inference)
        finally:
            stop_finished.set()

    workflow._stop_runtime_locked = slow_stop  # type: ignore[method-assign]
    observed = workflow.observe([_device(ready=False)])
    assert observed['runtime']['status'] == 'stopping'
    assert stop_entered.wait(timeout=1.0)

    coordinator = workflow.recording_coordinator
    assert coordinator is not None

    def forbidden_recorder_probe() -> Dict[str, Any]:
        raise AssertionError('pending safety-stop observation probed recorder state')

    coordinator.state = forbidden_recorder_probe  # type: ignore[method-assign]
    started = time.perf_counter()
    assert workflow.has_active_processes() is True
    workflow.stop_all()
    state_snapshot = workflow.state([_device(ready=False)])
    observed_again = workflow.observe([_device(ready=False)])
    stop_snapshot = workflow.stop_runtime([_device(ready=False)])
    inference_snapshot = workflow.stop_inference([_device(ready=False)])
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
    for snapshot in (
        state_snapshot,
        observed_again,
        stop_snapshot,
        inference_snapshot,
    ):
        assert snapshot['runtime']['status'] == 'stopping'
        assert snapshot['runtime']['active'] is True
        assert snapshot['recording']['status'] == 'stopping'
        assert snapshot['recording']['capabilities']['episode_accept']['can'] is False

    allow_stop.set()
    assert stop_finished.wait(timeout=2.0)


def test_device_stop_api_reaches_starter_during_slow_workflow_safety_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if 'fcntl' not in sys.modules:
        fake_fcntl = types.ModuleType('fcntl')
        fake_fcntl.LOCK_EX = 1  # type: ignore[attr-defined]
        fake_fcntl.LOCK_NB = 2  # type: ignore[attr-defined]
        fake_fcntl.LOCK_UN = 8  # type: ignore[attr-defined]
        fake_fcntl.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, 'fcntl', fake_fcntl)
    from incubator.control_panel import server

    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    workflow.start_runtime([_device(ready=True)])
    stop_entered = threading.Event()
    allow_stop = threading.Event()
    stop_finished = threading.Event()
    original_stop = workflow._stop_runtime_locked

    def slow_stop(*, stop_inference: bool) -> None:
        stop_entered.set()
        if not allow_stop.wait(timeout=2.0):
            raise AssertionError('test did not release asynchronous safety stop')
        try:
            original_stop(stop_inference=stop_inference)
        finally:
            stop_finished.set()

    workflow._stop_runtime_locked = slow_stop  # type: ignore[method-assign]
    workflow.observe([_device(ready=False)])
    assert stop_entered.wait(timeout=1.0)

    starter_stop_called = threading.Event()

    class Starter:
        def stop_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
            assert device_class == 'BaseDevice'
            assert str(device_class_id) == '7'
            starter_stop_called.set()
            return {
                'key': 'BaseDevice:7',
                'class': 'BaseDevice',
                'id': 7,
                'status': 'stopped',
                'running': False,
                'ready': False,
                'desired_state': 'stopped',
                'shm': {'connected': False, 'fresh': False, 'error': None},
            }

    controller = object.__new__(server.DeviceController)
    controller._lock = threading.RLock()
    controller._shutdown = False
    controller.starter = Starter()
    controller.workflow = workflow
    controller.revision = 0
    controller.logs = server.LogStore()
    controller._effective_config = None
    monkeypatch.setattr(server, 'DeviceController', lambda **kwargs: controller)

    app = server.create_app(
        config_dir=tmp_path,
        recording_replay=object(),
    )
    app.testing = True
    started = time.perf_counter()
    response = app.test_client().post(
        '/api/devices/BaseDevice/7/stop',
        headers={'X-Control-Transition': 'safety-stop-during-background-teardown'},
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.get_json()['device']['status'] == 'stopped'
    assert starter_stop_called.is_set()
    assert elapsed < 0.25

    allow_stop.set()
    assert stop_finished.wait(timeout=2.0)
    app.config['STATE_SYNC'].close()


def test_stop_timeout_keeps_runtime_owned_and_blocks_dependency_teardown(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    workflow.start_runtime([_device(ready=True)])

    class StubbornProcess(FakeProcess):
        def terminate(self) -> None:
            self.events.append(f'{self.role}:terminate')

        def kill(self) -> None:
            self.events.append(f'{self.role}:kill')

    stubborn = StubbornProcess('action_executor', events)
    workflow.action_executor_process = stubborn

    with pytest.raises(WorkflowConflictError, match='ActionExecutor did not exit'):
        workflow.stop_all()

    assert workflow.action_executor_process is stubborn
    assert workflow.has_active_processes() is True
    assert 'device_manager:terminate' not in events


def test_force_stop_all_kills_runtime_and_inference_without_recording_flush(
    tmp_path: Path,
) -> None:
    events: List[str] = []
    workflow = _workflow(tmp_path, events=events)
    workflow.start_runtime([_device(ready=True)])
    inference = FakeInferenceProcess(events)
    workflow.inference_process = inference
    workflow._inference_new_session = False
    workflow._inference_desired = True
    workflow._inference_ready = True
    manager_pid = workflow.device_manager_process.pid
    executor_pid = workflow.action_executor_process.pid

    result = workflow.force_stop_all()

    assert result['ok'] is True
    assert result['recording_flush_bypassed'] is True
    assert set(result['killed_pids']) == {
        inference.pid,
        manager_pid,
        executor_pid,
    }
    assert f'inference:signal:{signal.SIGKILL}' in events
    assert 'action_executor:kill' in events
    assert 'device_manager:kill' in events
    assert 'recording:shutdown' not in events
    assert workflow.inference_process is None
    assert workflow.action_executor_process is None
    assert workflow.device_manager_process is None
    assert workflow.recording_coordinator is None


def test_force_stop_all_cleans_matching_runtime_shm_claims(
    tmp_path: Path,
) -> None:
    names: Dict[str, tuple[int, int]] = {}
    unlinked: List[str] = []
    config = _config(tmp_path / 'checkpoint')
    config['summary_shm'] = 'custom_summary_data'

    def unlink(name: str) -> None:
        unlinked.append(name)
        names.pop(name)

    workflow = _workflow(
        tmp_path,
        config=config,
        shm_identity=names.get,
        shm_unlink=unlink,
    )
    workflow.start_runtime([_device(ready=True)])
    names.update(
        {
            'custom_summary_data': (11, 101),
            'policy_data': (11, 102),
        }
    )
    workflow._manager_connection.events.append(
        {
            'event': 'shm_claimed',
            'name': 'custom_summary_data',
            'dev': 11,
            'ino': 101,
        }
    )
    workflow._executor_connection.events.append(
        {
            'event': 'shm_claimed',
            'name': 'policy_data',
            'dev': 11,
            'ino': 102,
        }
    )

    result = workflow.force_stop_all()

    assert result['ok'] is True
    assert result['removed_shm'] == ['custom_summary_data', 'policy_data']
    assert result['remaining_shm'] == []
    assert result['preserved_replacement_shm'] == []
    assert sorted(unlinked) == ['custom_summary_data', 'policy_data']
    assert names == {}


def test_force_stop_all_preserves_replaced_runtime_shm(
    tmp_path: Path,
) -> None:
    names: Dict[str, tuple[int, int]] = {}
    unlinked: List[str] = []
    workflow = _workflow(
        tmp_path,
        shm_identity=names.get,
        shm_unlink=unlinked.append,
    )
    workflow.start_runtime([_device(ready=True)])
    workflow._executor_connection.events.append(
        {
            'event': 'shm_claimed',
            'name': 'policy_data',
            'dev': 12,
            'ino': 201,
        }
    )
    names['policy_data'] = (12, 999)

    result = workflow.force_stop_all()

    assert result['ok'] is True
    assert result['removed_shm'] == []
    assert result['remaining_shm'] == []
    assert result['preserved_replacement_shm'] == ['policy_data']
    assert unlinked == []
    assert names == {'policy_data': (12, 999)}


def test_force_stop_all_never_unlinks_unclaimed_external_runtime_shm(
    tmp_path: Path,
) -> None:
    names: Dict[str, tuple[int, int]] = {}
    unlinked: List[str] = []
    workflow = _workflow(
        tmp_path,
        shm_identity=names.get,
        shm_unlink=unlinked.append,
    )
    workflow.start_runtime([_device(ready=True)])
    names['policy_data'] = (13, 301)

    result = workflow.force_stop_all()

    assert result['ok'] is True
    assert result['removed_shm'] == []
    assert result['preserved_replacement_shm'] == []
    assert unlinked == []
    assert names == {'policy_data': (13, 301)}


def test_force_stop_all_reports_runtime_shm_cleanup_failure(
    tmp_path: Path,
) -> None:
    names: Dict[str, tuple[int, int]] = {}

    def fail_unlink(name: str) -> None:
        raise PermissionError(name)

    workflow = _workflow(
        tmp_path,
        shm_identity=names.get,
        shm_unlink=fail_unlink,
    )
    workflow.start_runtime([_device(ready=True)])
    names['policy_data'] = (14, 401)
    workflow._executor_connection.events.append(
        {
            'event': 'shm_claimed',
            'name': 'policy_data',
            'dev': 14,
            'ino': 401,
        }
    )

    result = workflow.force_stop_all()

    assert result['ok'] is False
    assert result['removed_shm'] == []
    assert result['remaining_shm'] == ['policy_data']
    assert workflow._runtime_error == 'Force stop incomplete: SHM policy_data'
    assert names == {'policy_data': (14, 401)}
