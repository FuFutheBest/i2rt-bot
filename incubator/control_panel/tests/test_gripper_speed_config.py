from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest

from incubator.control_panel import gripper_control
from incubator.control_panel.runtime_config import compose_runtime_config


CONFIG_NAME = 'policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi'


def _config_with_speed(speed: object = None, *, missing: bool = False):
    config = compose_runtime_config(CONFIG_NAME)
    robot = next(
        item for item in config['devices'] if item.get('class') == 'RizonRobot'
    )
    if missing:
        robot['gripper'].pop('speed_dps', None)
    else:
        robot['gripper']['speed_dps'] = speed
    return config


def test_config_switch_refreshes_resolved_gripper_speed_state() -> None:
    manager = gripper_control.GripperControlManager(
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
    try:
        manager.configure('explicit-speed', _config_with_speed([1000.0, 750.0]))
        state = manager.state()
        assert state['config_name'] == 'explicit-speed'
        assert state['sides']['left']['move_speed_dps'] == pytest.approx(1000.0)
        assert state['sides']['right']['move_speed_dps'] == pytest.approx(750.0)
        assert state['sides']['left']['move_speed_source'] == 'gripper.speed_dps'
        assert state['sides']['right']['move_speed_source'] == 'gripper.speed_dps'

        manager.configure('default-speed', _config_with_speed(missing=True))
        state = manager.state()
        assert state['config_name'] == 'default-speed'
        assert state['sides']['left']['move_speed_dps'] == pytest.approx(
            gripper_control.DEFAULT_MOVE_SPEED_DPS
        )
        assert state['sides']['right']['move_speed_dps'] == pytest.approx(
            gripper_control.DEFAULT_MOVE_SPEED_DPS
        )
        assert state['sides']['left']['move_speed_source'] == 'sdk_default'
        assert state['sides']['right']['move_speed_source'] == 'sdk_default'
    finally:
        manager.close()


def test_resolved_speed_maps_to_bundled_sdk_dps_without_opening_hardware() -> None:
    profiles = gripper_control.extract_gripper_profiles(
        compose_runtime_config(CONFIG_NAME)
    )
    try:
        config_class, gripper_class = gripper_control._default_sdk_loader()
    except ModuleNotFoundError:
        pytest.skip("LKTech SDK submodule is not initialized")
    assert list(inspect.signature(gripper_class.Move).parameters) == [
        'self',
        'target_width_m',
        'velocity',
        'force_limit',
    ]

    for profile in profiles.values():
        config = config_class(
            port=profile.port,
            motor_id=profile.motor_id,
            baudrate=profile.baudrate,
            speed_dps=profile.speed_dps,
            home_position=1.0,
            read_timeout=profile.read_timeout,
            wait_timeout=profile.wait_timeout,
            auto_calibrate=False,
            min_torque_iq=profile.min_torque_iq,
            max_torque_iq=profile.max_torque_iq,
            calibration_timeout=profile.calibration_timeout,
            calibration_settle_time=profile.calibration_settle_time,
            calibration_dir=None,
            min_encoder=profile.min_encoder,
            max_encoder=profile.max_encoder,
            min_angle_deg=profile.min_angle_deg,
            max_angle_deg=profile.max_angle_deg,
            arm_length_m=profile.arm_length_m,
            apex_angle_max_deg=profile.apex_angle_max_deg,
            gear_ratio=profile.gear_ratio,
            label=profile.label,
        )
        sdk_gripper = gripper_class(
            None,
            config=config,
            min_vel=0.0,
            max_vel=profile.sdk_move_velocity,
        )
        assert sdk_gripper._motor is None
        assert sdk_gripper._velocity_to_speed_dps(
            profile.sdk_move_velocity
        ) == pytest.approx(profile.speed_dps)
