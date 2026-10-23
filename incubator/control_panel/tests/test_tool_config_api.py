"""Hardware-free tests for config-derived Flexiv tool/TCP metadata."""

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
from incubator.control_panel.robot_state_preview import RobotStatePreviewResult  # noqa: E402
from incubator.control_panel.runtime_config import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    extract_robot_tools,
    inspect_runtime_config,
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


class MissingRobotReader:
    def read(self) -> RobotStatePreviewResult:
        return RobotStatePreviewResult(
            joint_positions={},
            metadata={'status': 'not_found', 'timestamp_ns': None, 'schema': None},
        )

    def close(self) -> None:
        return None


def _dual_device(gripper: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'class': 'RizonRobot',
        'device_class_id': 2,
        'robot_sn': ['Rizon4-012345', 'Rizon4R-012346'],
        'data_shape': [44],
        'control_shape': [16],
        'gripper': gripper,
    }


def test_empty_tool_name_without_elements_is_none_not_zero_offset() -> None:
    tools = extract_robot_tools(
        {'devices': [_dual_device({'name': 'gripper', 'tool_name': ''})]}
    )
    for side in ('left', 'right'):
        assert tools[side] == {
            'name': None,
            'resolved_tool_name': None,
            'status': 'not-configured',
            'transform': None,
            'asset': None,
            'source': None,
            'resolved_tool_source': None,
            'transform_source': None,
            'resolution': 'none',
            'resolution_warning': None,
            'active_element_count': 0,
            'reason': 'rizon-tool-not-configured',
        }


def test_named_tool_without_authoritative_pose_is_explicitly_unknown() -> None:
    tools = extract_robot_tools(
        {'devices': [_dual_device({'tool_name': 'umi3_tcp'})]}
    )
    assert tools['left']['name'] == 'umi3_tcp'
    assert tools['right']['name'] == 'umi3_tcp'
    assert tools['left']['status'] == 'name-known-pose-unknown'
    assert tools['left']['transform'] is None
    assert tools['left']['reason'] == 'tool-transform-not-present-in-config'


def test_explicit_flange_transform_is_finite_and_quaternion_is_normalized() -> None:
    tools = extract_robot_tools(
        {
            'devices': [
                _dual_device(
                    {
                        'tool_name': 'umi3_tcp',
                        'tool_transforms': {
                            'left': {
                                'translation': [0.01, -0.02, 0.31],
                                'quaternion_wxyz': [2.0, 0.0, 0.0, 0.0],
                            },
                            'right': {
                                'pose': [0.02, 0.03, 0.32, 0.0, 0.0, 0.0, 3.0],
                            },
                        },
                    }
                )
            ]
        }
    )
    assert tools['left']['status'] == 'available'
    assert tools['left']['transform'] == {
        'frame': 'flange',
        'translation': [0.01, -0.02, 0.31],
        'quaternion': [1.0, 0.0, 0.0, 0.0],
        'quaternion_order': 'wxyz',
    }
    assert tools['right']['transform']['quaternion'] == [0.0, 0.0, 0.0, 1.0]
    assert tools['right']['source'] == 'gripper.tool_transforms.right'


@pytest.mark.parametrize(
    'transform',
    [
        {'translation': [0.0, 0.0, math.inf], 'quaternion': [1, 0, 0, 0]},
        {'translation': [0.0, 0.0, 0.1], 'quaternion': [0, 0, 0, 0]},
        {'translation': [0.0, 0.0], 'quaternion': [1, 0, 0, 0]},
        {'translation': [0.0, 0.0, 0.1], 'quaternion': [True, 0, 0, 0]},
    ],
)
def test_invalid_transform_never_crosses_api_contract(transform: Dict[str, Any]) -> None:
    tools = extract_robot_tools(
        {
            'devices': [
                _dual_device(
                    {'tool_name': 'umi3_tcp', 'tool_transform': transform}
                )
            ]
        }
    )
    assert tools['left']['status'] == 'invalid-transform'
    assert tools['left']['transform'] is None
    assert tools['left']['reason'] == 'configured-tool-transform-is-invalid'


def _write_config(directory: Path, name: str, tool_yaml: str) -> None:
    (directory / f'{name}.yaml').write_text(
        f"""devices:
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-012345, Rizon4R-012346]
    data_shape: [44]
    data_dtype: float64
    control_shape: [16]
    fps: 120.0
    gripper:
      name: gripper
{tool_yaml}
global_settings:
  restart_on_failure: true
  max_restart_attempts: 2
  restart_delay: 0.0
runtime:
  signal_files: {{}}
""",
        encoding='utf-8',
    )


def test_inspect_and_robot_state_switch_resolved_tool_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    _write_config(
        config_dir,
        'known',
        """      tool_name: umi3_tcp
      tool_transform:
        translation: [0.01, 0.02, 0.30]
        quaternion_wxyz: [0.5, 0.5, 0.5, 0.5]
""",
    )
    _write_config(
        config_dir,
        'controller',
        """      tool_name: ""
      elements:
        - type: tool
          activate: true
          name: elements_tcp
          tcp_location: [0.0, 0.0, 0.24, 1.0, 0.0, 0.0, 0.0]
""",
    )
    monkeypatch.setattr(
        DeviceLifecycleManager,
        '_probe_shared_memory',
        lambda self, device_info: _missing_shm(device_info),
    )

    inspected = inspect_runtime_config('known', config_dir)
    assert inspected['configured_tools']['left']['status'] == 'available'

    app = server.create_app(
        config_dir=config_dir,
        robot_state_reader_factory=lambda _device_class_id: MissingRobotReader(),
    )
    app.testing = True
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'known'}).status_code == 200
    known = client.get('/api/robots/state').get_json()
    assert known['tcp_poses'] == {'left': None, 'right': None}
    assert known['configured_tools']['left']['status'] == 'available'
    detail = client.get('/api/configs/known').get_json()
    assert detail['base_config']['configured_tools'] == known['configured_tools']
    assert detail['effective_config']['configured_tools'] == known['configured_tools']

    assert client.post('/api/config/select', json={'name': 'controller'}).status_code == 200
    controller = client.get('/api/robots/state').get_json()
    assert controller['configured_tools']['left']['status'] == 'available'
    assert controller['configured_tools']['left']['resolved_tool_name'] == 'elements_tcp'
    assert controller['configured_tools']['left']['resolved_tool_source'] == 'gripper.elements[0]'
    assert controller['configured_tools']['left']['resolution'] == 'elements-active-tool'
    assert controller['configured_tools']['left']['transform']['translation'] == [0.0, 0.0, 0.24]
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_real_0710_multi_without_elements_does_not_fabricate_tool_pose() -> None:
    name = 'config_infer_fisheye_umi3_dual_openpi_pi05_chip_0710_multi'
    if not (DEFAULT_CONFIG_DIR / 'legacy' / f'{name}.yaml').is_file():
        pytest.skip('repository integration config is unavailable')
    tools = inspect_runtime_config(name)['configured_tools']
    for side in ('left', 'right'):
        assert tools[side]['status'] == 'not-configured'
        assert tools[side]['name'] is None
        assert tools[side]['transform'] is None
