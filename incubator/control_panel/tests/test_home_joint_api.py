"""Hardware-free tests for config-derived Rizon home joint visualization."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

sys.modules.setdefault('devices', types.SimpleNamespace(DEVICE_CLASSES={}))
import manager  # noqa: E402, F401
sys.modules.setdefault('visualizers', types.SimpleNamespace(VISUALIZER_CLASSES={}))

from incubator.control_panel import server  # noqa: E402
from incubator.control_panel.lifecycle import DeviceLifecycleManager  # noqa: E402
from incubator.control_panel.robot_state_preview import (  # noqa: E402
    RobotStatePreviewResult,
)
from incubator.control_panel.runtime_config import (  # noqa: E402
    extract_robot_home,
    inspect_runtime_config,
)


LEFT_A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RIGHT_A = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
LEFT_B = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
RIGHT_B = [-1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7]


def _config(home: str) -> str:
    return f"""devices:
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-012345, Rizon4R-012346]
    control_shape: [16]
    data_shape: [44]
    data_dtype: float64
    fps: 120.0
    home:
{home}
global_settings:
  restart_on_failure: true
  max_restart_attempts: 2
  restart_delay: 0.0
runtime:
  signal_files: {{}}
"""


def _write_config(directory: Path, name: str, left: list, right: list) -> None:
    home = f"      joints_left: {left}\n      joints_right: {right}"
    (directory / f'{name}.yaml').write_text(_config(home), encoding='utf-8')


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


class MissingRobotReader:
    def read(self) -> RobotStatePreviewResult:
        return RobotStatePreviewResult(
            joint_positions={},
            metadata={
                'status': 'not_found',
                'timestamp_ns': None,
                'schema': None,
            },
        )

    def close(self) -> None:
        return None


def test_extract_home_resolves_selected_preset_and_sanitizes_vectors() -> None:
    config = {
        'devices': [
            {
                'class': 'RizonRobot',
                'device_class_id': 2,
                'home': {
                    'preset_index': 1,
                    'presets': [
                        {'joints_left': LEFT_A, 'joints_right': RIGHT_A},
                        {'joints_left': LEFT_B, 'joints_right': RIGHT_B},
                    ],
                },
            }
        ]
    }
    assert extract_robot_home(config) == {
        'joints_left': LEFT_B,
        'joints_right': RIGHT_B,
    }

    config['devices'][0]['home'] = {
        'joints_left': [0, 1, 2, 3, 4, 5, math.inf],
        'joints_right': [0, 1, 2, 3, 4, 5, True],
    }
    assert extract_robot_home(config) == {
        'joints_left': None,
        'joints_right': None,
    }


def test_single_arm_home_requires_explicit_config_side() -> None:
    device = {
        'class': 'RizonRobot',
        'device_class_id': 1,
        'home': {'joints': LEFT_A},
    }
    assert extract_robot_home({'devices': [device]}) == {
        'joints_left': None,
        'joints_right': None,
    }
    device['arm_side'] = 'left'
    assert extract_robot_home({'devices': [device]}) == {
        'joints_left': LEFT_A,
        'joints_right': None,
    }


def test_inspect_exposes_only_sanitized_home(tmp_path: Path) -> None:
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    _write_config(config_dir, 'demo', LEFT_A, RIGHT_A)

    inspected = inspect_runtime_config('demo', config_dir)
    assert inspected['home'] == {
        'joints_left': LEFT_A,
        'joints_right': RIGHT_A,
    }
    assert set(inspected['home']) == {'joints_left', 'joints_right'}


def test_robot_state_home_changes_immediately_with_selected_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    _write_config(config_dir, 'first', LEFT_A, RIGHT_A)
    _write_config(config_dir, 'second', LEFT_B, RIGHT_B)
    monkeypatch.setattr(
        DeviceLifecycleManager,
        '_probe_shared_memory',
        lambda self, device_info: _missing_shm(device_info),
    )

    app = server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=lambda _device_class_id: MissingRobotReader(),
    )
    app.testing = True
    client = app.test_client()

    assert client.get('/api/robots/state').get_json()['home'] == {
        'joints_left': None,
        'joints_right': None,
    }
    response = client.post('/api/config/select', json={'name': 'first'})
    assert response.status_code == 200, response.get_json()
    first = client.get('/api/robots/state').get_json()
    assert first['status'] == 'not_found'
    assert first['left'] is None and first['right'] is None
    assert first['home'] == {
        'joints_left': LEFT_A,
        'joints_right': RIGHT_A,
    }

    assert client.post('/api/config/select', json={'name': 'second'}).status_code == 200
    second = client.get('/api/robots/state').get_json()
    assert second['home'] == {
        'joints_left': LEFT_B,
        'joints_right': RIGHT_B,
    }
    state = client.get('/api/state').get_json()
    assert state['effective_config']['home'] == second['home']
    detail = client.get('/api/configs/second').get_json()
    assert detail['home'] == second['home']
    assert detail['base_config']['home'] == second['home']
    assert detail['effective_config']['home'] == second['home']
    app.config['DEVICE_CONTROLLER'].shutdown()
