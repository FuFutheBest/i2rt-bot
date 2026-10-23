"""Hardware-free tests for persistent home-pose candidate orchestration."""

from __future__ import annotations

import copy
import stat
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict

import pytest

sys.modules.setdefault('devices', types.SimpleNamespace(DEVICE_CLASSES={}))
import manager  # noqa: E402, F401
sys.modules.setdefault('visualizers', types.SimpleNamespace(VISUALIZER_CLASSES={}))

from incubator.control_panel import server  # noqa: E402
from incubator.control_panel.home_pose_candidates import (  # noqa: E402
    HomePoseCandidateCoordinator,
    HomePoseCandidateStorageError,
    HomePoseCandidateStore,
    build_generation_context,
)
from incubator.control_panel.lifecycle import DeviceLifecycleManager  # noqa: E402
from incubator.control_panel.runtime_config import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    compose_runtime_config,
)


LEFT_HOME = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RIGHT_HOME = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]


def _urdfs(directory: Path) -> Dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    left = directory / 'left.urdf'
    right = directory / 'right.urdf'
    left.write_text('<robot name="Rizon4"/>', encoding='utf-8')
    right.write_text('<robot name="Rizon4R"/>', encoding='utf-8')
    return {'left': left, 'right': right}


def _effective_config(
    *,
    left: list[float] = LEFT_HOME,
    right: list[float] = RIGHT_HOME,
    serials: tuple[str, str] = ('Rizon4-000001', 'Rizon4R-000002'),
    capture_device: str = '/dev/video0',
    tool_x: float = 0.01,
) -> Dict[str, Any]:
    return {
        'devices': [
            {
                'class': 'RizonRobot',
                'device_class_id': 2,
                'robot_sn': list(serials),
                'home': {
                    'joints_left': list(left),
                    'joints_right': list(right),
                },
                'gripper': {
                    'tool_name': 'umi3_tcp',
                    'tool_transform': {
                        'translation': [tool_x, 0.0, 0.2],
                        'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                    },
                },
            },
            {
                'class': 'WebCameraOpenPi',
                'device_class_id': 4,
                'capture_device': capture_device,
            },
        ]
    }


def _candidate(side: str, index: int) -> Dict[str, Any]:
    offset = 0.01 * (index + 1)
    joints = [value + offset for value in (LEFT_HOME if side == 'left' else RIGHT_HOME)]
    pose = {
        'position_m': [0.4 + offset, 0.1, 0.3],
        'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0],
        'matrix': [
            [1.0, 0.0, 0.0, 0.4 + offset],
            [0.0, 1.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    return {
        'joints': joints,
        'target_frame_pose': pose,
        'requested_pose': pose,
        'position_residual_m': 0.0001,
        'orientation_residual_rad': 0.0002,
        'joint_distance_from_home_rad': 0.1 + offset,
    }


def _fake_sampler(**kwargs: Any) -> Dict[str, Any]:
    count = int(kwargs['parameters']['candidate_count'])
    side = kwargs['side']
    return {
        'side': side,
        'variant': 'Rizon4' if side == 'left' else 'Rizon4R',
        'status': 'complete',
        'target_frame': 'tcp' if kwargs['tool_transform'] is not None else 'flange',
        'tool_status': 'explicit-config-transform',
        'attempts': count,
        'elapsed_seconds': 0.01,
        'failure_counts': {},
        'collision_check': {
            'performed': False,
            'safe': None,
            'method': None,
            'reason': 'kinematic-only',
        },
        'candidates': [_candidate(side, index) for index in range(count)],
    }


def _parameter_factory(preset: str, count: int) -> Dict[str, Any]:
    return {'preset_name': preset, 'candidate_count': count}


def _wait_status(
    coordinator: HomePoseCandidateCoordinator,
    expected: set[str],
    timeout: float = 3.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = coordinator.state()
        if state['status'] in expected:
            return state
        time.sleep(0.01)
    raise AssertionError(f"candidate state did not reach {expected}: {coordinator.state()}")


def _cached_payload(key: str) -> Dict[str, Any]:
    return {
        'schema_version': 1,
        'status': 'ready',
        'key': key,
        'config_name': 'demo',
        'config_hash': 'a' * 64,
        'params': {'preset': 'conservative', 'count': 1, 'seed': 0},
        'generated_at': '2026-07-13T00:00:00+00:00',
        'error': None,
        'arms': {
            side: {
                'model': 'Rizon4' if side == 'left' else 'Rizon4R',
                'status': 'ready',
                'target_frame': 'flange',
                'candidates': [{'id': f'{side}-01', 'joints': [0.0] * 7}],
                'error': None,
            }
            for side in ('left', 'right')
        },
    }


def test_store_is_atomic_private_bounded_and_path_safe(tmp_path: Path) -> None:
    store = HomePoseCandidateStore(tmp_path / 'cache', max_bytes=4096)
    key = 'a' * 64
    baseline = _cached_payload(key)
    path = store.write(key, baseline)

    assert store.load(key) == baseline
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert not list(path.parent.glob('*.tmp'))

    oversized = copy.deepcopy(baseline)
    oversized['padding'] = 'x' * 8192
    with pytest.raises(HomePoseCandidateStorageError):
        store.write(key, oversized)
    assert store.load(key) == baseline

    with pytest.raises(HomePoseCandidateStorageError):
        store.load('../escape')


def test_cache_key_uses_only_kinematic_fields_and_generation_params(
    tmp_path: Path,
) -> None:
    urdfs = _urdfs(tmp_path / 'urdf')
    first = build_generation_context(
        'demo',
        _effective_config(),
        {'preset': 'conservative', 'count': 6, 'seed': 7},
        urdf_paths=urdfs,
        sampler_source_hash='b' * 64,
    )
    rebound = build_generation_context(
        'demo',
        _effective_config(
            serials=('Rizon4-999999', 'Rizon4R-888888'),
            capture_device='/dev/video8',
        ),
        {'preset': 'conservative', 'count': 6, 'seed': 7},
        urdf_paths=urdfs,
        sampler_source_hash='b' * 64,
    )
    assert rebound['config_hash'] == first['config_hash']
    assert rebound['key'] == first['key']

    changed_home = build_generation_context(
        'demo',
        _effective_config(left=[value + 0.01 for value in LEFT_HOME]),
        first['params'],
        urdf_paths=urdfs,
        sampler_source_hash='b' * 64,
    )
    changed_tool = build_generation_context(
        'demo',
        _effective_config(tool_x=0.02),
        first['params'],
        urdf_paths=urdfs,
        sampler_source_hash='b' * 64,
    )
    changed_params = build_generation_context(
        'demo',
        _effective_config(),
        {'preset': 'conservative', 'count': 5, 'seed': 7},
        urdf_paths=urdfs,
        sampler_source_hash='b' * 64,
    )
    assert changed_home['config_hash'] != first['config_hash']
    assert changed_tool['config_hash'] != first['config_hash']
    assert changed_params['config_hash'] == first['config_hash']
    assert changed_params['key'] != first['key']


def test_process_restart_hits_cache_without_invoking_sampler(tmp_path: Path) -> None:
    urdfs = _urdfs(tmp_path / 'urdf')
    store = HomePoseCandidateStore(tmp_path / 'cache')
    calls: list[str] = []

    def sampler(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs['side'])
        return _fake_sampler(**kwargs)

    first = HomePoseCandidateCoordinator(
        store=store,
        sampler=sampler,
        parameter_factory=_parameter_factory,
        urdf_paths=urdfs,
        sampler_source_hash='c' * 64,
    )
    first.schedule('demo', _effective_config())
    ready = _wait_status(first, {'ready'})
    assert calls == ['left', 'right']
    first.shutdown()

    def forbidden_sampler(**_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError('cache hit must not invoke sampler')

    second = HomePoseCandidateCoordinator(
        store=store,
        sampler=forbidden_sampler,
        parameter_factory=_parameter_factory,
        urdf_paths=urdfs,
        sampler_source_hash='c' * 64,
    )
    cached = second.schedule('demo', _effective_config())
    assert cached['status'] == 'ready'
    assert cached['key'] == ready['key']
    second.shutdown()


def test_partial_result_and_generation_token_discard_stale_state(
    tmp_path: Path,
) -> None:
    urdfs = _urdfs(tmp_path / 'urdf')

    def one_side_fails(**kwargs: Any) -> Dict[str, Any]:
        if kwargs['side'] == 'right':
            raise RuntimeError('right sampling failed')
        return _fake_sampler(**kwargs)

    partial_coordinator = HomePoseCandidateCoordinator(
        store=HomePoseCandidateStore(tmp_path / 'partial-cache'),
        sampler=one_side_fails,
        parameter_factory=_parameter_factory,
        urdf_paths=urdfs,
        sampler_source_hash='d' * 64,
    )
    partial_coordinator.schedule('partial', _effective_config())
    partial = _wait_status(partial_coordinator, {'partial'})
    assert partial['arms']['left']['status'] == 'ready'
    assert partial['arms']['right']['status'] == 'error'
    partial_coordinator.shutdown()

    started = threading.Event()
    release = threading.Event()

    def blocking_sampler(**kwargs: Any) -> Dict[str, Any]:
        if kwargs['home_joints'][0] == LEFT_HOME[0] and kwargs['side'] == 'left':
            started.set()
            release.wait(timeout=2.0)
        return _fake_sampler(**kwargs)

    coordinator = HomePoseCandidateCoordinator(
        store=HomePoseCandidateStore(tmp_path / 'race-cache'),
        sampler=blocking_sampler,
        parameter_factory=_parameter_factory,
        urdf_paths=urdfs,
        sampler_source_hash='e' * 64,
    )
    coordinator.schedule('first', _effective_config())
    assert started.wait(timeout=1.0)
    second_config = _effective_config(left=[value + 1.0 for value in LEFT_HOME])
    coordinator.schedule('second', second_config)
    current = _wait_status(coordinator, {'ready'})
    release.set()
    time.sleep(0.05)
    assert current['config_name'] == 'second'
    assert coordinator.state()['config_name'] == 'second'
    coordinator.shutdown()


def _write_config(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'demo.yaml').write_text(
        f"""devices:
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-000001, Rizon4R-000002]
    data_shape: [44]
    data_dtype: float64
    control_shape: [16]
    fps: 120.0
    home:
      joints_left: {LEFT_HOME}
      joints_right: {RIGHT_HOME}
global_settings:
  restart_on_failure: true
  max_restart_attempts: 2
  restart_delay: 0.0
runtime:
  signal_files: {{}}
""",
        encoding='utf-8',
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


def test_candidate_api_is_not_exposed_by_control_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_config(config_dir)
    monkeypatch.setattr(
        DeviceLifecycleManager,
        '_probe_shared_memory',
        lambda self, device_info: _missing_shm(device_info),
    )
    app = server.create_app(config_dir=config_dir)
    app.testing = True
    client = app.test_client()

    assert client.get('/api/home-pose-candidates').status_code == 404
    assert (
        client.post('/api/home-pose-candidates/regenerate', json={}).status_code
        == 404
    )
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_real_0710_config_generates_six_candidates_per_arm(
    tmp_path: Path,
) -> None:
    name = 'config_infer_fisheye_umi3_dual_openpi_pi05_chip_0710_multi'
    if not (DEFAULT_CONFIG_DIR / 'legacy' / f'{name}.yaml').is_file():
        pytest.skip('repository integration config is unavailable')

    coordinator = HomePoseCandidateCoordinator(
        store=HomePoseCandidateStore(tmp_path / 'real-cache'),
    )
    coordinator.schedule(name, compose_runtime_config(name))
    ready = _wait_status(coordinator, {'ready', 'partial', 'error'}, timeout=15.0)
    assert ready['status'] == 'ready', ready
    assert ready['arms']['left']['model'] == 'Rizon4'
    assert ready['arms']['right']['model'] == 'Rizon4R'
    assert ready['arms']['left']['target_frame'] == 'flange'
    assert ready['arms']['right']['target_frame'] == 'flange'
    assert len(ready['arms']['left']['candidates']) == 6
    assert len(ready['arms']['right']['candidates']) == 6
    assert ready['arms']['left']['collision_check']['performed'] is False
    assert ready['arms']['right']['collision_check']['performed'] is False
    coordinator.shutdown()
