"""Hardware-free REST tests for the aggregate Rizon joint-state endpoint."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

# DeviceStarter imports production hardware registries eagerly; the route tests
# need only its process-control shell and must not require vendor SDKs.
sys.modules.setdefault('devices', types.SimpleNamespace(DEVICE_CLASSES={}))
import manager  # noqa: E402, F401
sys.modules.setdefault(
    'visualizers', types.SimpleNamespace(VISUALIZER_CLASSES={})
)

from incubator.control_panel import server as device_control_server  # noqa: E402
from incubator.control_panel.lifecycle import DeviceLifecycleManager  # noqa: E402
from incubator.control_panel.robot_state_preview import RobotStatePreviewResult  # noqa: E402

DeviceStarter = DeviceLifecycleManager


NOW_NS = 1_700_000_000_123_000_000


class FakeRobotReader:
    def __init__(self, result: RobotStatePreviewResult) -> None:
        self.result = result
        self.read_calls = 0
        self.close_calls = 0

    def read(self) -> RobotStatePreviewResult:
        self.read_calls += 1
        return self.result

    def close(self) -> None:
        self.close_calls += 1


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


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / 'configs'
    directory.mkdir()
    robot = """devices:
  - class: RizonRobot
    device_class_id: 2
    data_shape: [44]
    data_dtype: float64
    fps: 120.0
global_settings:
  restart_on_failure: true
  max_restart_attempts: 2
  restart_delay: 0.0
runtime:
  signal_files: {}
"""
    no_robot = robot.replace(
        "class: RizonRobot\n    device_class_id: 2\n    data_shape: [44]\n    data_dtype: float64\n    fps: 120.0",
        "class: BaseDevice\n    device_class_id: 7\n    data_shape: [2]\n    data_dtype: float32\n    fps: 10.0",
    )
    (directory / 'demo.yaml').write_text(robot, encoding='utf-8')
    (directory / 'other.yaml').write_text(robot, encoding='utf-8')
    (directory / 'no_robot.yaml').write_text(no_robot, encoding='utf-8')
    (directory / 'yam.yaml').write_text(
        robot.replace(
            'global_settings:',
            """  - class: YamHeadRobot
    device_class_id: 0
    data_shape: [20]
    data_dtype: float64
    fps: 60.0
    yam_arm: yam
    yam_initial_q: 0,2,0.73,0.82,0,0.02
global_settings:""",
        ),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        DeviceStarter,
        '_probe_shared_memory',
        lambda self, device_info: _missing_shm(device_info),
    )
    return directory


def _dual_result() -> RobotStatePreviewResult:
    return RobotStatePreviewResult(
        joint_positions={
            'left': tuple(float(index) / 10 for index in range(7)),
            'right': tuple(-float(index) / 10 for index in range(7)),
        },
        metadata={
            'status': 'ok',
            'timestamp_ns': NOW_NS,
            'schema': 'dual_tcp_gripper_44',
        },
        tcp_poses={
            'left': (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
            'right': (-0.1, -0.2, 0.4, 0.0, 0.0, 1.0, 0.0),
        },
        gripper_positions={
            'left': 0.031,
            'right': 0.074,
        },
    )


def test_robot_state_uses_only_selected_config_and_reuses_cached_reader(
    config_dir: Path,
) -> None:
    readers = []
    factory_calls = []

    def factory(device_class_id: int) -> FakeRobotReader:
        factory_calls.append(device_class_id)
        reader = FakeRobotReader(_dual_result())
        readers.append(reader)
        return reader

    app = device_control_server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=factory,
    )
    app.testing = True
    client = app.test_client()

    assert client.get('/api/robots/state').get_json()['status'] == 'no_config'
    assert factory_calls == []
    assert client.post('/api/config/select', json={'name': 'demo'}).status_code == 200

    first = client.get('/api/robots/state')
    second = client.get('/api/robots/state')
    payload = first.get_json()
    assert first.status_code == second.status_code == 200
    assert payload['status'] == 'ok'
    assert payload['timestamp'] == NOW_NS
    assert payload['timestamp_ns'] == NOW_NS
    assert payload['left'] == pytest.approx([index / 10 for index in range(7)])
    assert payload['right'] == pytest.approx([-index / 10 for index in range(7)])
    assert payload['tcp_poses']['left'] == pytest.approx(
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    )
    assert payload['tcp_poses']['right'] == pytest.approx(
        [-0.1, -0.2, 0.4, 0.0, 0.0, 1.0, 0.0]
    )
    assert payload['gripper_positions'] == pytest.approx(
        {'left': 0.031, 'right': 0.074}
    )
    assert payload['tool_names'] == {'left': None, 'right': None}
    assert payload['devices'] == [
        {
            'device_class_id': 2,
            'schema': 'dual_tcp_gripper_44',
            'status': 'ok',
            'timestamp_ns': NOW_NS,
        }
    ]
    assert factory_calls == [2]
    assert readers[0].read_calls == 1

    assert client.post('/api/config/select', json={'name': 'other'}).status_code == 200
    assert readers[0].close_calls == 1
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_robot_state_missing_shm_is_pollable_and_single_arm_maps_right(
    config_dir: Path,
) -> None:
    missing = RobotStatePreviewResult(
        joint_positions={},
        metadata={'status': 'not_found', 'timestamp_ns': None, 'schema': None},
    )
    single = RobotStatePreviewResult(
        joint_positions={'single': tuple(range(7))},
        metadata={
            'status': 'ok',
            'timestamp_ns': NOW_NS,
            'schema': 'single_tcp_gripper_22',
        },
        tcp_poses={
            'single': (0.5, 0.1, 0.7, 0.0, 0.0, 0.0, 1.0)
        },
        gripper_positions={'single': 0.057},
    )
    results = [missing, single]
    readers = []

    def factory(device_class_id: int) -> FakeRobotReader:
        del device_class_id
        reader = FakeRobotReader(results[len(readers)])
        readers.append(reader)
        return reader

    app = device_control_server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=factory,
    )
    app.testing = True
    client = app.test_client()
    client.post('/api/config/select', json={'name': 'demo'})

    payload = client.get('/api/robots/state').get_json()
    assert payload['status'] == 'not_found'
    assert payload['left'] is None
    assert payload['right'] is None

    client.post('/api/config/select', json={'name': 'other'})
    payload = client.get('/api/robots/state').get_json()
    assert payload['status'] == 'ok'
    assert payload['left'] is None
    assert payload['right'] == list(range(7))
    assert payload['tcp_poses']['left'] is None
    assert payload['tcp_poses']['right'] == pytest.approx(
        [0.5, 0.1, 0.7, 0.0, 0.0, 0.0, 1.0]
    )
    assert payload['gripper_positions'] == {'left': None, 'right': 0.057}
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_robot_state_exposes_yam_initpos_then_fresh_six_joint_shm(
    config_dir: Path,
) -> None:
    yam_result = RobotStatePreviewResult(
        joint_positions={'yam': (0.1, 2.1, 0.7, 0.8, -0.1, 0.03)},
        metadata={
            'status': 'ok',
            'timestamp_ns': NOW_NS,
            'age_ms': 4.0,
            'schema': 'yam_joint_commanded_measured_20',
            'shm_name': 'YamHeadRobot_0_data',
        },
        tcp_poses={
            'yam': (0.2, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        },
    )
    yam_readers = []

    def yam_factory(device_class_id: int) -> FakeRobotReader:
        assert device_class_id == 0
        reader = FakeRobotReader(yam_result)
        yam_readers.append(reader)
        return reader

    app = device_control_server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=lambda _device_class_id: FakeRobotReader(_dual_result()),
        yam_state_reader_factory=yam_factory,
    )
    app.testing = True
    client = app.test_client()
    assert client.post('/api/config/select', json={'name': 'yam'}).status_code == 200

    payload = client.get('/api/robots/state').get_json()
    assert payload['yam_head'] == {
        'configured': True,
        'device_key': 'YamHeadRobot:0',
        'model': 'yam',
        'status': 'ok',
        'urdf_url': '/static/assets/yam/yam_no_gripper.urdf',
        'base_position_world': [0.0, 0.0, 0.2],
        'joint_count': 6,
        'joint_names': [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ],
        'initial_joints': [0.0, 2.0, 0.73, 0.82, 0.0, 0.02],
        'joints': [0.1, 2.1, 0.7, 0.8, -0.1, 0.03],
        'tcp_pose': [0.2, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0],
        'fresh': True,
        'source': 'live',
        'timestamp_ns': NOW_NS,
        'age_ms': 4.0,
        'schema': 'yam_joint_commanded_measured_20',
        'shm_name': 'YamHeadRobot_0_data',
    }
    assert len(yam_readers) == 1
    assert client.get('/api/robots/state').get_json()['yam_head']['fresh'] is True
    assert len(yam_readers) == 1
    assert yam_readers[0].read_calls == 1

    client.post('/api/config/select', json={'name': 'demo'})
    assert yam_readers[0].close_calls == 1
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_robot_state_coalesces_concurrent_viewers_and_expires(
    config_dir: Path,
) -> None:
    readers = []

    def factory(_device_class_id: int) -> FakeRobotReader:
        reader = FakeRobotReader(_dual_result())
        readers.append(reader)
        return reader

    app = device_control_server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=factory,
    )
    app.testing = True
    client = app.test_client()
    assert (
        client.post('/api/config/select', json={'name': 'demo'}).status_code
        == 200
    )
    controller = app.config['DEVICE_CONTROLLER']

    with ThreadPoolExecutor(max_workers=5) as pool:
        payloads = list(
            pool.map(lambda _index: controller.read_robot_states(), range(5))
        )

    assert [payload['timestamp_ns'] for payload in payloads] == [NOW_NS] * 5
    assert len(readers) == 1
    assert readers[0].read_calls == 1

    controller._robot_state_cache_at -= 1.0
    assert controller.read_robot_states()['timestamp_ns'] == NOW_NS
    assert readers[0].read_calls == 2
    controller.shutdown()


def test_config_without_rizon_never_constructs_a_reader(config_dir: Path) -> None:
    calls = []

    def factory(device_class_id: int) -> FakeRobotReader:
        calls.append(device_class_id)
        return FakeRobotReader(_dual_result())

    app = device_control_server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=factory,
    )
    app.testing = True
    client = app.test_client()
    client.post('/api/config/select', json={'name': 'no_robot'})

    response = client.get('/api/robots/state')
    assert response.status_code == 200
    assert response.get_json() == {
        'configured_tools': {
            'left': {
                'active_element_count': 0,
                'asset': None,
                'name': None,
                'reason': 'rizon-tool-not-configured',
                'resolution': 'none',
                'resolution_warning': None,
                'resolved_tool_name': None,
                'resolved_tool_source': None,
                'source': None,
                'status': 'not-configured',
                'transform': None,
                'transform_source': None,
            },
            'right': {
                'active_element_count': 0,
                'asset': None,
                'name': None,
                'reason': 'rizon-tool-not-configured',
                'resolution': 'none',
                'resolution_warning': None,
                'resolved_tool_name': None,
                'resolved_tool_source': None,
                'source': None,
                'status': 'not-configured',
                'transform': None,
                'transform_source': None,
            },
        },
        'devices': [],
        'home': {'joints_left': None, 'joints_right': None},
        'yam_head': {
            'configured': False,
            'device_key': None,
            'model': None,
            'status': 'not-configured',
            'urdf_url': None,
            'base_position_world': [0.0, 0.0, 0.2],
            'joint_count': 6,
            'joint_names': [
                'joint1',
                'joint2',
                'joint3',
                'joint4',
                'joint5',
                'joint6',
            ],
            'initial_joints': None,
            'joints': None,
            'tcp_pose': None,
            'fresh': False,
            'source': 'unavailable',
            'timestamp_ns': None,
            'age_ms': None,
            'schema': None,
            'shm_name': None,
        },
        'left': None,
        'right': None,
        'gripper_positions': {'left': None, 'right': None},
        'status': 'not_configured',
        'tcp_poses': {'left': None, 'right': None},
        'tool_names': {'left': None, 'right': None},
        'timestamp': None,
        'timestamp_ns': None,
    }
    assert calls == []
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_stale_joint_payload_is_not_exposed_as_live(config_dir: Path) -> None:
    stale = RobotStatePreviewResult(
        joint_positions={
            'left': tuple(range(7)),
            'right': tuple(range(7)),
        },
        metadata={
            'status': 'stale',
            'timestamp_ns': NOW_NS,
            'schema': 'dual_tcp_gripper_44',
        },
        tcp_poses={
            'left': (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
            'right': (0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0),
        },
        gripper_positions={
            'left': 0.02,
            'right': 0.08,
        },
    )

    app = device_control_server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=lambda device_class_id: FakeRobotReader(stale),
    )
    app.testing = True
    client = app.test_client()
    client.post('/api/config/select', json={'name': 'demo'})

    payload = client.get('/api/robots/state').get_json()
    assert payload['status'] == 'stale'
    assert payload['timestamp_ns'] is None
    assert payload['left'] is None
    assert payload['right'] is None
    assert payload['tcp_poses'] == {'left': None, 'right': None}
    assert payload['gripper_positions'] == {'left': None, 'right': None}
    app.config['DEVICE_CONTROLLER'].shutdown()
