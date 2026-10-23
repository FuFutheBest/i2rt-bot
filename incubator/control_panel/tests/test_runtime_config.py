"""Tests for shared Hydra runtime configuration composition."""

import json
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig

from incubator.control_panel.runtime_config import (
    DEFAULT_CONFIG_DIR,
    RuntimeConfigError,
    compose_runtime_config,
    extract_camera_slots,
    extract_execution_profile,
    extract_gripper_slots,
    extract_robot_slots,
    extract_yam_visualization,
    inspect_runtime_config,
    list_runtime_configs,
    runtime_config_metadata,
    validate_runtime_config,
)


def _robot_control(**overrides: object) -> dict[str, object]:
    control: dict[str, object] = {
        "type": "joints",
        "format": {
            "shape": [2],
            "dtype": "float64",
            "type": ["linear:2"],
        },
    }
    control.update(overrides)
    return control


def test_runtime_validation_rejects_removed_control_and_raw_shm_fields() -> None:
    config = {
        "devices": [
            {
                "class": "BaseRobot",
                "device_class_id": 0,
                "control": _robot_control(enabled_arms=[True, True]),
            }
        ],
        "policy": {
            "controls": [{"shm": "BaseRobot_0_control", "shape": [2]}]
        },
        "action_executor": {
            "robot_control_shms": [
                {"shm": "BaseRobot_0_control", "shape": [2]}
            ],
        },
    }

    errors = validate_runtime_config(config)

    assert any("enabled_arms" in error for error in errors)
    assert any("policy.controls[0].shm was removed" in error for error in errors)
    assert any(
        "action_executor.robot_control_shms[0].shm was removed" in error
        for error in errors
    )


def test_compose_normalizes_hydra_device_group(tmp_path: Path) -> None:
    config_dir = tmp_path / 'configs'
    (config_dir / 'devices').mkdir(parents=True)
    (config_dir / 'devices' / 'demo.yaml').write_text(
        """devices:
  - class: WebCameraOpenPi
    device_class_id: ${camera.id}
    data_shape: [480, 640, 3]
    data_dtype: uint8
    fps: 30
""",
        encoding='utf-8',
    )
    (config_dir / 'launch.yaml').write_text(
        """defaults:
  - devices: demo
  - _self_
camera:
  id: 2
""",
        encoding='utf-8',
    )

    config = compose_runtime_config('launch', config_dir)
    inspected = inspect_runtime_config('launch.yaml', config_dir)

    assert list_runtime_configs(config_dir) == ['launch']
    assert config['devices'][0]['device_class_id'] == 2
    assert inspected['valid'] is True
    assert inspected['devices'][0]['device_class'] == 'WebCameraOpenPi'
    assert inspected['devices'][0]['capabilities']['preview'] is True
    assert inspected['devices'][0]['shape'] == [480, 640, 3]
    assert inspected['devices'][0]['dtype'] == 'uint8'
    assert 'config' not in inspected
    # The full resolved mapping remains available to the process-start path.
    assert config['camera']['id'] == 2
    assert config['devices'][0]['data_shape'] == [480, 640, 3]


@pytest.mark.parametrize(
    ('policy_class', 'workflow_kind', 'hardware_required', 'supported'),
    [
        (None, 'device', False, True),
        ('ControlPanelNoMotionPolicy', 'device', False, True),
        ('OpenPiUmiPolicy', 'openpi', True, True),
        ('DiffusionPolicy', 'diffusion', True, False),
        ('LingBotVAPolicy', 'lingbotva', True, False),
    ],
)
def test_runtime_config_metadata_derives_control_panel_capabilities(
    policy_class: str | None,
    workflow_kind: str,
    hardware_required: bool,
    supported: bool,
) -> None:
    config = {
        'devices': [{'class': 'BaseDevice', 'device_class_id': 1}],
    }
    if policy_class:
        config['policy'] = {'class': policy_class}

    metadata = runtime_config_metadata('fixture', config)

    assert metadata['workflow_kind'] == workflow_kind
    assert metadata['policy_class'] == policy_class
    assert metadata['hardware_required'] is hardware_required
    assert metadata['status'] == 'available'
    assert metadata['control_panel_supported'] is supported


def test_runtime_config_metadata_allows_explicit_overrides() -> None:
    metadata = runtime_config_metadata(
        'fixture',
        {
            'metadata': {
                'workflow_kind': 'custom',
                'policy_class': 'CustomPolicy',
                'hardware_required': True,
                'status': 'experimental',
                'control_panel_supported': True,
            },
            'devices': [],
        },
    )

    assert metadata == {
        'workflow_kind': 'custom',
        'policy_class': 'CustomPolicy',
        'hardware_required': True,
        'status': 'experimental',
        'control_panel_supported': False,
        'support_reason': 'metadata does not match the composed policy/workflow: custom/CustomPolicy vs device/None',
    }


def test_runtime_config_metadata_respects_device_enabled_string_booleans() -> None:
    disabled = runtime_config_metadata(
        'fixture',
        {
            'devices': [
                {'class': 'RizonRobot', 'device_class_id': 2, 'enabled': 'false'},
            ],
        },
    )
    assert disabled['hardware_required'] is False

    enabled = runtime_config_metadata(
        'fixture',
        {
            'devices': [
                {'class': 'RizonRobot', 'device_class_id': 2, 'enabled': 'true'},
            ],
        },
    )
    assert enabled['hardware_required'] is True


def test_runtime_metadata_parses_runtime_boolean_strings() -> None:
    config = {
        'runtime': {
            'enable_device_manager': 'false',
            'enable_visualizers': 'true',
        },
        'devices': [],
    }

    metadata = extract_execution_profile(config)

    assert metadata['runtime'] == {
        'enable_device_manager': False,
        'enable_visualizers': True,
    }


def test_gripper_slots_reject_boolean_motor_id_aliases() -> None:
    slots = extract_gripper_slots(
        {
            "devices": [
                {
                    "class": "RizonRobot",
                    "device_class_id": 2,
                    "gripper": {
                        "backend": "lktech",
                        "enabled": [True, False],
                        "port": ["/dev/ttyUSB0", "/dev/ttyUSB1"],
                        "motor_id": [True, 2],
                        "calibration_files": [
                            "calibration/left.json",
                            "calibration/right.json",
                        ],
                    },
                }
            ]
        }
    )

    assert slots["left"] == {
        "side": "left",
        "configured": True,
        "port": "/dev/ttyUSB0",
        "motor_id": None,
        "motor_id_valid": False,
        "calibration_file": "left.json",
        "backend": "lktech",
    }
    assert slots["right"]["configured"] is False


def test_gripper_slots_support_single_arm_scalar_and_legacy_flat_fields() -> None:
    slots = extract_gripper_slots(
        {
            "devices": [
                {
                    "class": "RizonRobot",
                    "device_class_id": 2,
                    "robot_sn": "Rizon4-single",
                    "control_shape": [8],
                    "gripper_backend": "",
                    "gripper_server": {"host": "127.0.0.1", "port": 8765},
                    "utilize_gripper": "true",
                    "gripper_port": "/dev/ttyUSB0",
                    "gripper_motor_id": 1,
                    "gripper_calibration_files": "calibration/single.json",
                }
            ]
        }
    )

    assert slots["left"] == {
        "side": "left",
        "configured": True,
        "port": "/dev/ttyUSB0",
        "motor_id": 1,
        "motor_id_valid": True,
        "calibration_file": "single.json",
        "backend": "lktech",
    }
    assert slots["right"]["configured"] is False


def test_camera_slots_distinguish_inference_required_from_recording_only() -> None:
    config = {
        'devices': [
            {'class': 'WebCameraOpenPi', 'device_class_id': 0, 'data_shape': [360, 640, 3]},
            {'class': 'WebCameraOpenPi', 'device_class_id': 4, 'data_shape': [360, 640, 3]},
            {'class': 'WebCameraOpenPi', 'device_class_id': 2, 'data_shape': [480, 640, 3]},
            {'class': 'RizonRobot', 'device_class_id': 2, 'data_shape': [44]},
        ],
        'policy': {
            'obs': {
                'devices': [
                    'WebCameraOpenPi_0',
                    'WebCameraOpenPi_4',
                    'RizonRobot_2',
                ]
            },
            'infer_rollout': {
                'head_camera': 'WebCameraOpenPi_2',
                'left_camera': 'WebCameraOpenPi_0',
                'right_camera': 'WebCameraOpenPi_4',
            },
        },
    }

    slots = extract_camera_slots(config)

    assert list(slots) == ['head', 'left', 'right']
    assert slots['head'] == {
        'role': 'head',
        'device_key': 'WebCameraOpenPi:2',
        'configured': True,
        'required': False,
        'inference_required': False,
        'recording_configured': True,
        'source': 'recording',
    }
    assert slots['left']['device_key'] == 'WebCameraOpenPi:0'
    assert slots['left']['required'] is True
    assert slots['right']['device_key'] == 'WebCameraOpenPi:4'
    assert slots['right']['required'] is True

    config['policy']['obs']['devices'].insert(0, 'WebCameraOpenPi_2')
    assert extract_camera_slots(config)['head']['required'] is True


def test_robot_slots_make_configured_yam_head_required() -> None:
    config = {
        'devices': [
            {'class': 'RizonRobot', 'device_class_id': 2, 'data_shape': [44]},
            {'class': 'YamHeadRobot', 'device_class_id': 0, 'data_shape': [20]},
        ],
        'policy': {
            'obs': {'devices': ['RizonRobot_2']},
            'controls': [
                {
                    'endpoint': {
                        'device': 'RizonRobot',
                        'device_class_id': 2,
                        'port': 'command',
                    },
                    'shape': [16],
                },
            ],
        },
    }

    slots = extract_robot_slots(config)

    assert slots == {
        'rizon': {
            'role': 'rizon',
            'device_key': 'RizonRobot:2',
            'configured': True,
            'required': True,
            'inference_required': True,
            'source': 'policy',
        },
        'i2rt_head': {
            'role': 'i2rt_head',
            'device_key': 'YamHeadRobot:0',
            'configured': True,
            'required': True,
            'inference_required': True,
            'source': 'runtime-default',
        },
    }

    config['policy']['obs']['devices'].append('YamHeadRobot_0')
    required = extract_robot_slots(config)['i2rt_head']
    assert required['required'] is True
    assert required['source'] == 'policy'


def test_yam_visualization_uses_configured_model_and_startup_initpos() -> None:
    config = {
        'devices': [
            {
                'class': 'YamHeadRobot',
                'device_class_id': 0,
                'yam_arm': 'yam',
                'yam_initial_q': '0,2,0.73,0.82,0,0.02',
            }
        ]
    }

    assert extract_yam_visualization(config) == {
        'configured': True,
        'device_key': 'YamHeadRobot:0',
        'model': 'yam',
        'status': 'available',
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
    }

    config['devices'][0]['yam_initial_q'] = 'invalid'
    assert extract_yam_visualization(config)['status'] == 'invalid-initpos'
    config['devices'][0]['yam_arm'] = 'unknown-yam'
    unsupported = extract_yam_visualization(config)
    assert unsupported['configured'] is True
    assert unsupported['status'] == 'unsupported-model'
    assert unsupported['urdf_url'] is None


def test_inspect_is_safe_summary_and_filters_disabled_devices(tmp_path: Path) -> None:
    config_dir = tmp_path / 'configs'
    (config_dir / 'devices').mkdir(parents=True)
    (config_dir / 'devices' / 'mixed.yaml').write_text(
        """devices:
  - class: WebCameraOpenPi
    device_class_id: 0
    data_shape: [480, 640, 3]
    data_dtype: uint8
    fps: 30
    shm_name: camera_preview_data
    secret: super-secret-value
    token: bearer-token-value
    ip: 10.0.0.7
    path: /srv/private/calibration.json
  - class: RizonRobot
    device_class_id: 1
    enabled: false
    data_shape: [44]
    data_dtype: float64
    fps: 120
    secret: disabled-secret-value
    token: disabled-token-value
    ip: 10.0.0.8
    path: /srv/private/robot.json
""",
        encoding='utf-8',
    )
    (config_dir / 'launch.yaml').write_text(
        """defaults:
  - devices: mixed
  - _self_
""",
        encoding='utf-8',
    )

    inspected = inspect_runtime_config('launch', config_dir)
    payload = json.dumps(inspected, sort_keys=True)

    assert inspected['valid'] is True
    assert inspected['device_count'] == 1
    assert inspected['disabled_count'] == 1
    assert [device['device_class_id'] for device in inspected['devices']] == [0]
    assert inspected['devices'][0]['shm_name'] == 'camera_preview_data'
    assert 'config' not in inspected
    assert 'config' not in inspected['devices'][0]
    for forbidden in (
        'super-secret-value',
        'bearer-token-value',
        'disabled-secret-value',
        'disabled-token-value',
        '10.0.0.7',
        '10.0.0.8',
        '/srv/private',
    ):
        assert forbidden not in payload

    # Inspecting is a redacted view; composition still returns the complete
    # resolved values required by the device launcher.
    composed = compose_runtime_config('launch', config_dir)
    assert composed['devices'][0]['secret'] == 'super-secret-value'
    assert composed['devices'][1]['enabled'] is False


def test_compose_rejects_config_group_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeConfigError, match='launch config relative to configs'):
        compose_runtime_config('../secret', tmp_path)


def test_real_giftbox_0703_config_resolves_hydra_runtime_context() -> None:
    name = 'config_infer_fisheye_umi3_dual_openpi_pi05_giftbox_0703'
    if not (DEFAULT_CONFIG_DIR / f'{name}.yaml').is_file():
        pytest.skip('repository integration config is unavailable')

    hydra_state = HydraConfig.instance()
    previous = hydra_state.cfg
    config = compose_runtime_config(name)
    serialized = json.dumps(config, sort_keys=True)

    assert isinstance(config.get('devices'), list)
    assert config['devices']
    assert 'hydra' not in config
    assert '${hydra:' not in serialized
    assert hydra_state.cfg is previous


def test_all_top_level_configs_have_no_missing_hydra_runtime_context() -> None:
    if not DEFAULT_CONFIG_DIR.is_dir():
        pytest.skip('repository configs directory is unavailable')

    names = list_runtime_configs(DEFAULT_CONFIG_DIR)
    successes = 0
    errors = {}
    for name in names:
        try:
            inspect_runtime_config(name, DEFAULT_CONFIG_DIR)
            successes += 1
        except RuntimeConfigError as exc:
            errors[name] = str(exc)

    assert names
    assert successes > 0
    forbidden_fragments = (
        'HydraConfig was not set',
        'Unsupported interpolation type hydra',
        'hydra:runtime',
    )
    assert {
        name: error
        for name, error in errors.items()
        if any(fragment in error for fragment in forbidden_fragments)
    } == {}
