from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from incubator.control_panel import gripper_control, server
from incubator.control_panel.gripper_control import (
    GripperProfile,
    _WorkerRuntime,
    _assert_worker_hardware_idle,
    _atomic_write_calibrations,
    _calibration_payload,
    _close_worker_runtime,
    extract_gripper_profiles,
)
from incubator.control_panel.hardware_discovery import HardwareSnapshot
from incubator.control_panel.runtime_config import compose_runtime_config


CONFIG_NAME = 'policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi'


def _profile(side: str) -> GripperProfile:
    index = 0 if side == 'left' else 1
    return GripperProfile(
        side=side,
        port=f'/dev/ttyUSB{1 - index}',
        motor_id=1,
        baudrate=115200,
        speed_dps=1000.0,
        speed_source='gripper.speed_dps',
        read_timeout=0.1,
        wait_timeout=4.0,
        calibration_timeout=15.0,
        calibration_settle_time=0.24,
        min_torque_iq=-40,
        max_torque_iq=190,
        min_encoder=None,
        max_encoder=None,
        min_angle_deg=10.0,
        max_angle_deg=1010.0,
        arm_length_m=0.06543,
        apex_angle_max_deg=90.0,
        gear_ratio=10.0,
        force_limit=100.0,
        calibration_file=f'/unused/{side}.json',
        label=f'fake-{side}',
    )


class FakeConfig:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class FakeMotor:
    def __init__(self) -> None:
        self.wait_calls = []

    def wait_for_stop(self, timeout: float) -> bool:
        self.wait_calls.append(timeout)
        return True


class FakeGripper:
    instances = []
    events = []

    def __init__(self, robot: Any, *, config: Any, min_vel: float, max_vel: float) -> None:
        assert robot is None
        self.config = config
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.motor = FakeMotor()
        self.width = 0.0
        self.enabled = False
        self.disabled = False
        self.calibration_calls = 0
        self.moves = []
        self._min_encoder = config.min_encoder
        self._max_encoder = config.max_encoder
        self._min_angle_deg = config.min_angle_deg
        self._max_angle_deg = config.max_angle_deg
        type(self).instances.append(self)

    def Enable(self, label: str) -> None:
        self.label = label
        self.enabled = True
        type(self).events.append(('enable', label))

    def Disable(self) -> None:
        self.disabled = True
        self.enabled = False
        type(self).events.append(('disable', self.label))

    def _calibrate_encoder_range(self) -> None:
        self.calibration_calls += 1
        type(self).events.append(('calibrate', self.label))
        self._min_angle_deg = 100.0
        self._max_angle_deg = 1100.0

    def Move(self, width: float, velocity: float, force: float) -> bool:
        self.moves.append((width, velocity, force))
        self.width = width
        return True

    def states(self) -> Any:
        return SimpleNamespace(width=self.width, is_moving=False)


def _sdk_loader() -> tuple[Any, Any]:
    return FakeConfig, FakeGripper


def _single_arm_profile_config(root: Path, *, enabled: Any = True) -> dict[str, Any]:
    calibration = root / 'calibration' / 'single.json'
    calibration.parent.mkdir(parents=True, exist_ok=True)
    calibration.write_text(
        json.dumps(
            {
                'motor_id': 1,
                'port': '/dev/ttyUSB0',
                'min_angle_deg': 10.0,
                'max_angle_deg': 1010.0,
            }
        ),
        encoding='utf-8',
    )
    return {
        'devices': [
            {
                'class': 'RizonRobot',
                'device_class_id': 1,
                'robot_sn': 'Rizon4-single',
                'control_shape': [8],
                'gripper': {
                    'backend': 'lktech',
                    'enabled': enabled,
                    'port': '/dev/ttyUSB0',
                    'motor_id': 1,
                    'calibration_files': 'calibration/single.json',
                },
            }
        ]
    }


@pytest.mark.parametrize('disabled', ['false', 'FALSE', '0', 'off', 'no', '', 0, False, None])
def test_profile_extraction_fails_closed_for_disabled_string_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disabled: Any,
) -> None:
    monkeypatch.setattr(gripper_control, 'REPO_ROOT', tmp_path)
    config = _single_arm_profile_config(tmp_path, enabled=disabled)

    assert extract_gripper_profiles(config) == {}


def test_profile_extraction_accepts_single_arm_scalar_calibration_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gripper_control, 'REPO_ROOT', tmp_path)
    config = _single_arm_profile_config(tmp_path, enabled='true')

    profiles = extract_gripper_profiles(config)

    assert set(profiles) == {'left'}
    assert profiles['left'].calibration_file == str(tmp_path / 'calibration' / 'single.json')


def test_profile_extraction_accepts_single_arm_one_item_calibration_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gripper_control, 'REPO_ROOT', tmp_path)
    config = _single_arm_profile_config(tmp_path, enabled=[True])
    gripper = config['devices'][0]['gripper']
    gripper['port'] = ['/dev/ttyUSB0']
    gripper['motor_id'] = [1]
    gripper['calibration_files'] = ['calibration/single.json']

    profiles = extract_gripper_profiles(config)

    assert set(profiles) == {'left'}


@pytest.mark.skip(reason="site-specific LKTech calibration artifacts are not bundled")
def test_effective_config_side_binding_does_not_trust_calibration_port() -> None:
    config = compose_runtime_config(CONFIG_NAME)
    profiles = extract_gripper_profiles(config)

    # YAML is authoritative. Historical JSON files contain the opposite ports.
    assert profiles['left'].port == '/dev/ttyUSB1'
    assert profiles['right'].port == '/dev/ttyUSB0'
    assert profiles['left'].max_width_m == pytest.approx(0.0925319933860716)
    assert profiles['left'].speed_dps == pytest.approx(1000.0)
    assert profiles['right'].speed_dps == pytest.approx(1000.0)
    assert profiles['left'].speed_source == 'gripper.speed_dps'
    assert profiles['right'].speed_source == 'gripper.speed_dps'
    assert profiles['left'].sdk_move_velocity * profiles['left'].calibrated_span_deg == pytest.approx(1000.0)
    assert profiles['right'].sdk_move_velocity * profiles['right'].calibrated_span_deg == pytest.approx(1000.0)


def test_move_speed_defaults_to_sdk_value_when_config_field_is_missing() -> None:
    config = compose_runtime_config(CONFIG_NAME)
    robot = next(item for item in config['devices'] if item.get('class') == 'RizonRobot')
    robot['gripper'].pop('speed_dps', None)

    profiles = extract_gripper_profiles(config)

    assert {item.speed_dps for item in profiles.values()} == {
        gripper_control.DEFAULT_MOVE_SPEED_DPS
    }
    assert {item.speed_source for item in profiles.values()} == {'sdk_default'}


def test_structured_gripper_slots_follow_stream_primitives() -> None:
    from incubator.control_panel.runtime_config import extract_gripper_slots

    config = compose_runtime_config(CONFIG_NAME)
    robot = next(item for item in config["devices"] if item["class"] == "RizonRobot")
    assert "enabled" not in robot["gripper"]
    assert set(extract_gripper_profiles(config)) == {"left", "right"}
    assert all(slot["configured"] for slot in extract_gripper_slots(config).values())

    # Keeping a gripper settings block must not enable undeclared channels.
    for stream in (robot["state_feedback"], robot["control"]):
        fmt = stream["format"]
        fmt["type"] = [primitive for primitive in fmt["type"] if primitive != "gripper"]
        fmt["shape"][0] -= 2
    assert extract_gripper_profiles(config) == {}
    assert not any(slot["configured"] for slot in extract_gripper_slots(config).values())


@pytest.mark.skip(reason="site-specific LKTech calibration artifacts are not bundled")
def test_move_speed_resolves_each_side_and_does_not_use_legacy_max_vel() -> None:
    config = compose_runtime_config(CONFIG_NAME)
    robot = next(item for item in config['devices'] if item.get('class') == 'RizonRobot')
    robot['gripper']['speed_dps'] = [1000.0, 750.0]
    robot['gripper']['max_vel'] = [0.001, 999.0]

    profiles = extract_gripper_profiles(config)

    assert profiles['left'].speed_dps == pytest.approx(1000.0)
    assert profiles['right'].speed_dps == pytest.approx(750.0)
    assert profiles['left'].sdk_move_velocity * profiles['left'].calibrated_span_deg == pytest.approx(1000.0)
    assert profiles['right'].sdk_move_velocity * profiles['right'].calibrated_span_deg == pytest.approx(750.0)


@pytest.mark.parametrize(
    'speed',
    [
        True,
        0.0,
        -1.0,
        float('nan'),
        float('inf'),
        gripper_control.MAX_MOVE_SPEED_DPS + 0.01,
    ],
)
def test_move_speed_rejects_invalid_or_unencodable_values(speed: Any) -> None:
    config = compose_runtime_config(CONFIG_NAME)
    robot = next(item for item in config['devices'] if item.get('class') == 'RizonRobot')
    robot['gripper']['speed_dps'] = [speed, 1000.0]

    with pytest.raises(gripper_control.GripperControlError, match='speed_dps.*left'):
        extract_gripper_profiles(config)


@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_valid"),
    [
        (
            {"min_angle_deg": 10.0, "max_angle_deg": 1010.0},
            "unverified",
            False,
        ),
        (
            {
                "min_angle_deg": 10.0,
                "max_angle_deg": 1010.0,
                "motor_id": True,
                "port": "/dev/ttyUSB1",
            },
            "stale",
            False,
        ),
        (
            {
                "min_angle_deg": 10.0,
                "max_angle_deg": 1010.0,
                "motor_id": 1,
                "port": "/dev/ttyUSB1",
            },
            "valid",
            True,
        ),
    ],
)
def test_calibration_artifact_requires_matching_verified_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    expected_status: str,
    expected_valid: bool,
) -> None:
    monkeypatch.setattr(gripper_control, "REPO_ROOT", tmp_path)
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "left.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    profile = replace(_profile("left"), calibration_file="calibration/left.json")

    artifact = gripper_control._calibration_artifact_state(profile)

    assert artifact["status"] == expected_status
    assert artifact["valid"] is expected_valid
    assert artifact["limits_valid"] is True
    if expected_valid:
        assert artifact["identity_status"] == "verified"
    else:
        assert artifact["identity_status"] != "verified"


def test_calibration_artifact_rejects_path_outside_repository_root() -> None:
    artifact = gripper_control._calibration_artifact_state(_profile("left"))

    assert artifact["status"] == "invalid"
    assert artifact["valid"] is False
    assert artifact["limits_valid"] is False


def test_profile_extraction_rejects_boolean_motor_id() -> None:
    config = compose_runtime_config(CONFIG_NAME)
    robot = next(
        item for item in config["devices"] if item.get("class") == "RizonRobot"
    )
    robot["gripper"]["motor_id"] = [True, 1]

    with pytest.raises(
        gripper_control.GripperControlError,
        match="motor_id for left",
    ):
        extract_gripper_profiles(config)


def test_fake_worker_recalibrates_both_without_constructing_robot() -> None:
    FakeGripper.instances = []
    FakeGripper.events = []
    transactions = []
    runtime = _WorkerRuntime(
        {side: _profile(side) for side in ('left', 'right')},
        sdk_loader=_sdk_loader,
        calibration_writer=lambda values: transactions.append(list(values)),
        preflight=lambda profiles: None,
    )
    result = runtime.execute({'action': 'calibrate', 'side': 'both'})
    runtime.close()

    assert [item['side'] for item in result['results']] == ['left', 'right']
    assert len(transactions) == 1
    assert [item[0].side for item in transactions[0]] == ['left', 'right']
    assert all(item.calibration_calls == 1 for item in FakeGripper.instances)
    assert all(item.disabled for item in FakeGripper.instances)
    assert FakeGripper.events == [
        ('enable', 'fake-left'),
        ('calibrate', 'fake-left'),
        ('disable', 'fake-left'),
        ('enable', 'fake-right'),
        ('calibrate', 'fake-right'),
        ('disable', 'fake-right'),
    ]


def test_fake_worker_maps_percent_to_width_waits_and_reads_actual() -> None:
    FakeGripper.instances = []
    profile = _profile('left')
    runtime = _WorkerRuntime(
        {'left': profile},
        sdk_loader=_sdk_loader,
        calibration_writer=lambda values: None,
        preflight=lambda profiles: None,
    )
    result = runtime.execute(
        {'action': 'position', 'side': 'left', 'target_percent': 25.0}
    )
    runtime.close()

    expected_width = profile.max_width_m * 0.25
    assert FakeGripper.instances[0].moves == [
        (expected_width, 1.0, 100.0)
    ]
    assert FakeGripper.instances[0].max_vel == pytest.approx(1.0)
    assert result['results'][0]['move_speed_dps'] == pytest.approx(1000.0)
    assert FakeGripper.instances[0].motor.wait_calls
    assert result['results'][0]['actual_percent'] == pytest.approx(25.0)
    assert FakeGripper.instances[0].disabled


def test_dual_calibration_file_transaction_rolls_back_partial_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gripper_control, 'REPO_ROOT', tmp_path)
    calibration_root = tmp_path / 'calibration'
    calibration_root.mkdir()
    profiles = []
    grippers = []
    old_payloads = {}
    for side in ('left', 'right'):
        path = calibration_root / f'{side}.json'
        payload = {'label': side, 'min_angle_deg': 1.0, 'max_angle_deg': 2.0}
        path.write_text(__import__('json').dumps(payload), encoding='utf-8')
        old_payloads[side] = path.read_bytes()
        profile = _profile(side)
        profiles.append(
            GripperProfile(
                **{**profile.__dict__, 'calibration_file': str(path)}
            )
        )
        grippers.append(
            SimpleNamespace(
                _min_encoder=None,
                _max_encoder=None,
                _min_angle_deg=100.0,
                _max_angle_deg=1100.0,
            )
        )

    native_replace = gripper_control.os.replace
    failed = False

    def fail_second_commit(source: Any, destination: Any) -> None:
        nonlocal failed
        destination_path = Path(destination)
        if destination_path == Path(profiles[1].calibration_file) and not failed:
            failed = True
            raise OSError('injected second-file commit failure')
        native_replace(source, destination)

    monkeypatch.setattr(gripper_control.os, 'replace', fail_second_commit)
    with pytest.raises(OSError, match='injected'):
        _atomic_write_calibrations(list(zip(profiles, grippers)))

    assert Path(profiles[0].calibration_file).read_bytes() == old_payloads['left']
    assert Path(profiles[1].calibration_file).read_bytes() == old_payloads['right']
    manifests = list(calibration_root.glob('.gripper-calibration-*.json'))
    assert len(manifests) == 1
    assert 'rolled_back' in manifests[0].read_text(encoding='utf-8')


def test_profile_extraction_rejects_duplicate_port_and_calibration_file() -> None:
    config = compose_runtime_config(CONFIG_NAME)
    robot = next(item for item in config['devices'] if item.get('class') == 'RizonRobot')
    robot['gripper']['port'] = ['/dev/ttyUSB0', '/dev/ttyUSB0']
    with pytest.raises(gripper_control.GripperControlError, match='same serial port'):
        extract_gripper_profiles(config)

    config = compose_runtime_config(CONFIG_NAME)
    robot = next(item for item in config['devices'] if item.get('class') == 'RizonRobot')
    first = robot['gripper']['calibration_files'][0]
    robot['gripper']['calibration_files'] = [first, first]
    with pytest.raises(gripper_control.GripperControlError, match='same calibration file'):
        extract_gripper_profiles(config)


@pytest.mark.parametrize(
    ('low', 'high', 'message'),
    [
        (1.0, None, 'incomplete angle'),
        (float('nan'), 2.0, 'non-finite angle'),
        (2.0, 2.0, 'span is zero'),
    ],
)
def test_calibration_payload_rejects_invalid_limits(
    low: float, high: float | None, message: str
) -> None:
    gripper = SimpleNamespace(
        _min_encoder=None,
        _max_encoder=None,
        _min_angle_deg=low,
        _max_angle_deg=high,
    )
    with pytest.raises(RuntimeError, match=message):
        _calibration_payload(_profile('left'), gripper)


def test_worker_preflight_rejects_holder_and_rizon_shm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = GripperProfile(
        **{**_profile('left').__dict__, 'port': '/dev/null'}
    )
    monkeypatch.setattr(gripper_control, 'rizon_shared_memory_names', lambda: ['RizonRobot_2_data'])
    with pytest.raises(RuntimeError, match='shared memory'):
        _assert_worker_hardware_idle({'left': profile})

    monkeypatch.setattr(gripper_control, 'rizon_shared_memory_names', lambda: [])
    monkeypatch.setattr(gripper_control, 'serial_port_holders', lambda port: [4321])
    with pytest.raises(RuntimeError, match='held by another process'):
        _assert_worker_hardware_idle({'left': profile})


def test_ttyusb_discovery_empty_and_stably_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_root = tmp_path / 'dev'
    dev_root.mkdir()
    assert gripper_control.discover_ttyusb_ports(dev_root=dev_root) == []

    for name in ('ttyUSB10', 'ttyUSB2', 'ttyUSB0', 'ttyUSBx', 'ttyACM0'):
        (dev_root / name).touch()
    real_is_char_device = Path.is_char_device

    def fake_is_char_device(path: Path) -> bool:
        if path.parent == dev_root:
            return path.name in {'ttyUSB0', 'ttyUSB2', 'ttyUSB10'}
        return real_is_char_device(path)

    monkeypatch.setattr(Path, 'is_char_device', fake_is_char_device)
    assert gripper_control.discover_ttyusb_ports(dev_root=dev_root) == [
        str(dev_root / 'ttyUSB0'),
        str(dev_root / 'ttyUSB2'),
        str(dev_root / 'ttyUSB10'),
    ]


def test_ttyusb_path_validation_rejects_arbitrary_paths() -> None:
    for value in (
        '/tmp/ttyUSB0',
        '/dev/ttyUSB0/../../etc/passwd',
        '/dev/ttyUSB0;touch /tmp/pwned',
        '/dev/serial/by-id/gripper',
        '/dev/ttyACM0',
    ):
        with pytest.raises(gripper_control.GripperRequestError):
            gripper_control.validate_ttyusb_port(value, require_present=False)


def test_chmod_failure_is_clear_and_blocks_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        'validate_ttyusb_port',
        lambda port, **kwargs: str(port),
    )

    def denied_chmod(path: str, mode: int) -> None:
        raise PermissionError('owner is root')

    monkeypatch.setattr(
        gripper_control.os,
        'lstat',
        lambda path: SimpleNamespace(
            st_uid=0, st_gid=20,
            st_mode=gripper_control.stat.S_IFCHR | 0o660,
            st_dev=1, st_ino=2, st_rdev=3,
        ),
    )
    monkeypatch.setattr(gripper_control.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getegid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getgroups', lambda: [1000])
    monkeypatch.setattr(
        gripper_control.os, 'access', lambda path, mode, **kwargs: False
    )
    monkeypatch.setattr(gripper_control.os, 'chmod', denied_chmod)
    monkeypatch.setattr(
        gripper_control.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout='', stderr='sudo denied'
        ),
    )
    with pytest.raises(
        gripper_control.GripperControlError,
        match=r'Cannot chmod.*sudo denied',
    ):
        gripper_control.ensure_ttyusb_permissions(
            '/dev/ttyUSB0', allow_world_rw=True
        )


def test_ttyusb_permission_defaults_to_group_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        'validate_ttyusb_port',
        lambda port, **kwargs: str(port),
    )
    monkeypatch.setattr(
        gripper_control.os,
        'lstat',
        lambda path: SimpleNamespace(
            st_uid=0, st_gid=20,
            st_mode=gripper_control.stat.S_IFCHR | 0o600,
            st_dev=1, st_ino=2, st_rdev=3,
        ),
    )
    monkeypatch.setattr(gripper_control.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getegid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getgroups', lambda: [20, 1000])
    access_checks = iter((False, True))
    monkeypatch.setattr(
        gripper_control.os,
        'access',
        lambda path, mode, **kwargs: next(access_checks),
    )
    applied = []
    monkeypatch.setattr(
        gripper_control.os,
        'chmod',
        lambda path, mode: applied.append((path, mode)),
    )

    assert gripper_control.ensure_ttyusb_permissions('/dev/ttyUSB0') == '/dev/ttyUSB0'
    assert applied == [('/dev/ttyUSB0', 0o660)]


def test_ttyusb_permission_world_fallback_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        'validate_ttyusb_port',
        lambda port, **kwargs: str(port),
    )
    monkeypatch.setattr(
        gripper_control.os,
        'lstat',
        lambda path: SimpleNamespace(
            st_uid=0, st_gid=20,
            st_mode=gripper_control.stat.S_IFCHR | 0o666,
            st_dev=1, st_ino=2, st_rdev=3,
        ),
    )
    monkeypatch.setattr(gripper_control.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getegid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getgroups', lambda: [1000])
    monkeypatch.delenv('SUPERINFERENCE_GRIPPER_ALLOW_WORLD_RW', raising=False)
    with pytest.raises(gripper_control.GripperControlError, match='add it to dialout'):
        gripper_control.ensure_ttyusb_permissions('/dev/ttyUSB0')


def test_existing_world_rw_permission_needs_no_chmod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        'validate_ttyusb_port',
        lambda port, **kwargs: str(port),
    )
    monkeypatch.setattr(
        gripper_control.os,
        'lstat',
        lambda path: SimpleNamespace(
            st_uid=0, st_gid=20,
            st_mode=gripper_control.stat.S_IFCHR | 0o666,
            st_dev=1, st_ino=2, st_rdev=3,
        ),
    )
    monkeypatch.setattr(gripper_control.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getegid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getgroups', lambda: [1000])
    monkeypatch.setattr(
        gripper_control.os, 'access', lambda path, mode, **kwargs: True
    )
    monkeypatch.setattr(
        gripper_control.os,
        'chmod',
        lambda path, mode: pytest.fail('chmod must not run when permission exists'),
    )
    assert gripper_control.ensure_ttyusb_permissions(
        '/dev/ttyUSB0', allow_world_rw=True
    ) == '/dev/ttyUSB0'


def test_permission_probe_fails_if_access_remains_false_after_chmod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        'validate_ttyusb_port',
        lambda port, **kwargs: str(port),
    )
    monkeypatch.setattr(
        gripper_control.os,
        'lstat',
        lambda path: SimpleNamespace(
            st_uid=1000, st_gid=1000,
            st_mode=gripper_control.stat.S_IFCHR | 0o600,
            st_dev=1, st_ino=2, st_rdev=3,
        ),
    )
    monkeypatch.setattr(gripper_control.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getegid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getgroups', lambda: [1000])
    monkeypatch.setattr(
        gripper_control.os, 'access', lambda path, mode, **kwargs: False
    )
    monkeypatch.setattr(gripper_control.os, 'chmod', lambda path, mode: None)
    with pytest.raises(
        gripper_control.GripperControlError,
        match='still unavailable',
    ):
        gripper_control.ensure_ttyusb_permissions('/dev/ttyUSB0')


def test_permission_probe_revalidates_node_before_early_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def changing_validator(port: str, **kwargs: Any) -> str:
        calls.append(port)
        if len(calls) > 1:
            raise gripper_control.GripperRequestError('ttyUSB node was replaced')
        return port

    monkeypatch.setattr(
        gripper_control, 'validate_ttyusb_port', changing_validator
    )
    monkeypatch.setattr(
        gripper_control.os,
        'lstat',
        lambda path: SimpleNamespace(
            st_uid=0, st_gid=20,
            st_mode=gripper_control.stat.S_IFCHR | 0o666,
            st_dev=1, st_ino=2, st_rdev=3,
        ),
    )
    monkeypatch.setattr(gripper_control.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getegid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getgroups', lambda: [1000])
    monkeypatch.setattr(
        gripper_control.os, 'access', lambda path, mode, **kwargs: True
    )
    with pytest.raises(
        gripper_control.GripperRequestError,
        match='node was replaced',
    ):
        gripper_control.ensure_ttyusb_permissions(
            '/dev/ttyUSB0', allow_world_rw=True
        )
    assert calls == ['/dev/ttyUSB0', '/dev/ttyUSB0']


def test_permission_probe_rejects_character_device_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        'validate_ttyusb_port',
        lambda port, **kwargs: str(port),
    )
    metadata = iter(
        (
            SimpleNamespace(
                st_uid=0, st_gid=20,
                st_mode=gripper_control.stat.S_IFCHR | 0o666,
                st_dev=1, st_ino=2, st_rdev=3,
            ),
            SimpleNamespace(
                st_uid=0, st_gid=20,
                st_mode=gripper_control.stat.S_IFCHR | 0o666,
                st_dev=1, st_ino=4, st_rdev=5,
            ),
        )
    )
    monkeypatch.setattr(gripper_control.os, 'lstat', lambda path: next(metadata))
    monkeypatch.setattr(gripper_control.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getegid', lambda: 1000)
    monkeypatch.setattr(gripper_control.os, 'getgroups', lambda: [1000])
    access_calls = []

    def effective_access(path: str, mode: int, **kwargs: Any) -> bool:
        access_calls.append(kwargs)
        return True

    monkeypatch.setattr(gripper_control.os, 'access', effective_access)
    with pytest.raises(
        gripper_control.GripperControlError,
        match='serial device changed',
    ):
        gripper_control.ensure_ttyusb_permissions(
            '/dev/ttyUSB0', allow_world_rw=True
        )
    assert access_calls == [{'effective_ids': True}]


def test_serial_holder_scan_tolerates_lazy_hidepid_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / 'proc'
    denied_fd = proc_root / '4242' / 'fd'
    denied_fd.mkdir(parents=True)
    original_scandir = os.scandir

    def guarded_scandir(path: str | os.PathLike[str]):
        if Path(path) == denied_fd:
            raise PermissionError('hidepid')
        return original_scandir(path)

    monkeypatch.setattr(os, 'scandir', guarded_scandir)
    assert gripper_control.serial_port_holders(
        '/dev/null', proc_root=proc_root
    ) == []


def test_serial_holder_scan_still_detects_readable_character_device(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / 'proc'
    fd_root = proc_root / '4242' / 'fd'
    fd_root.mkdir(parents=True)
    try:
        (fd_root / '7').symlink_to('/dev/null')
    except OSError as exc:  # pragma: no cover - non-POSIX test host
        pytest.skip(f'symlink support is required: {exc}')
    assert gripper_control.serial_port_holders(
        '/dev/null', proc_root=proc_root
    ) == [4242]


def test_serial_holder_bulk_scan_visits_proc_once_and_separates_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != 'posix':
        pytest.skip('Linux /proc character-device semantics are required')
    proc_root = tmp_path / 'proc'
    fd_root = proc_root / '4242' / 'fd'
    fd_root.mkdir(parents=True)
    (fd_root / '7').symlink_to('/dev/null')
    original_scandir = os.scandir
    proc_scans = 0

    def counting_scandir(path: str | os.PathLike[str]):
        nonlocal proc_scans
        if Path(path) == proc_root:
            proc_scans += 1
        return original_scandir(path)

    monkeypatch.setattr(os, 'scandir', counting_scandir)

    assert gripper_control.serial_port_holders_many(
        ['/dev/null', '/dev/zero'], proc_root=proc_root
    ) == {
        '/dev/null': [4242],
        '/dev/zero': [],
    }
    assert proc_scans == 1


def test_process_command_reports_bounded_cmdline_for_holder_diagnostics(
    tmp_path: Path,
) -> None:
    process_root = tmp_path / 'proc' / '4242'
    process_root.mkdir(parents=True)
    (process_root / 'cmdline').write_bytes(
        b'python\0incubator/control_panel/gripper_control.py\0'
        b'--config-name\0demo\0'
    )

    assert gripper_control.process_command(
        4242, proc_root=tmp_path / 'proc'
    ) == 'python incubator/control_panel/gripper_control.py --config-name demo'


def _manager_with_fake_ports(
    discovered: list[str],
    *,
    permission_granter=lambda port: str(port),
) -> gripper_control.GripperControlManager:
    manager = gripper_control.GripperControlManager(
        auto_monitor=False,
        sdk_module=Path(gripper_control.__file__),
        device_python=Path(sys.executable),
        port_discovery=lambda: list(discovered),
        port_validator=lambda port: gripper_control.validate_ttyusb_port(
            port, require_present=False
        ),
        permission_granter=permission_granter,
        permission_policy_check=lambda port: None,
    )
    manager._profiles = {
        'left': _profile('left'),
        'right': _profile('right'),
    }
    manager._status = 'idle'
    for side in ('left', 'right'):
        manager._sides[side]['configured'] = True
        manager._sides[side]['status'] = 'idle'
    return manager


def test_manager_uses_active_python_for_source_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SUPERINFERENCE_RELEASE_ROOT', raising=False)
    monkeypatch.delenv('SUPERINFERENCE_DEVICE_PYTHON', raising=False)

    manager = gripper_control.GripperControlManager(
        auto_monitor=False,
        sdk_module=Path(gripper_control.__file__),
    )

    assert manager.device_python == Path(sys.executable)


def test_disappeared_ttyusb_disables_only_affected_side() -> None:
    discovered = ['/dev/ttyUSB0', '/dev/ttyUSB1']
    manager = _manager_with_fake_ports(discovered)
    assert manager.state()['sides']['left']['port_present'] is True

    discovered.remove('/dev/ttyUSB1')
    state = manager.state()
    assert state['available_ports'] == ['/dev/ttyUSB0']
    assert state['sides']['left']['port_present'] is False
    assert state['sides']['left']['can_position'] is False
    assert state['sides']['right']['port_present'] is True
    manager.close()


def test_port_binding_busy_interlock_and_atomic_swap() -> None:
    discovered = ['/dev/ttyUSB0', '/dev/ttyUSB1']
    manager = _manager_with_fake_ports(discovered)
    assert manager.bind_ports(
        {'left': '/dev/ttyUSB0', 'right': '/dev/ttyUSB1'}
    ) == {'left': '/dev/ttyUSB0', 'right': '/dev/ttyUSB1'}

    manager._pending_positions = {'left': {'action': 'position'}}
    with pytest.raises(gripper_control.GripperConflictError, match='active'):
        manager.bind_ports({'left': '/dev/ttyUSB1', 'right': '/dev/ttyUSB0'})
    manager._pending_positions = {}
    manager.close()


def test_permission_failure_prevents_worker_creation() -> None:
    def permission_failure(port: str) -> str:
        raise gripper_control.GripperControlError(f'chmod failed for {port}')

    manager = _manager_with_fake_ports(
        ['/dev/ttyUSB0', '/dev/ttyUSB1'],
        permission_granter=permission_failure,
    )
    with pytest.raises(gripper_control.GripperControlError, match='chmod failed'):
        manager.position(
            'left',
            50.0,
            'position-permission-failure',
            expected_side_revision=manager._side_revisions['left'],
        )
    assert manager._process is None
    assert manager._pending_positions == {}
    manager.close()


class FlakyDisableGripper(FakeGripper):
    failures_remaining = 1

    def Disable(self) -> None:
        if type(self).failures_remaining:
            type(self).failures_remaining -= 1
            raise RuntimeError('injected disable failure')
        super().Disable()


def test_cleanup_retries_and_does_not_lose_gripper_reference() -> None:
    FlakyDisableGripper.instances = []
    FlakyDisableGripper.events = []
    FlakyDisableGripper.failures_remaining = 1
    runtime = _WorkerRuntime(
        {'left': _profile('left')},
        sdk_loader=lambda: (FakeConfig, FlakyDisableGripper),
        calibration_writer=lambda values: None,
        preflight=lambda profiles: None,
    )
    runtime.execute({'action': 'position', 'side': 'left', 'target_percent': 25.0})

    assert _close_worker_runtime(runtime, attempts=2) is None
    assert runtime.grippers == {}
    assert FlakyDisableGripper.instances[0].disabled


class StaticDiscovery:
    def discover(self) -> HardwareSnapshot:
        return HardwareSnapshot(
            revision='none',
            generated_at='2026-07-13T00:00:00+00:00',
            interfaces=[],
            cameras=[],
            robots=[],
            providers=[],
            warnings=[],
        )


class FakeGripperManager:
    def __init__(self) -> None:
        self.config_name = None
        self.calls = []
        self.bindings = {'left': '/dev/ttyUSB1', 'right': '/dev/ttyUSB0'}
        self.side_revisions = {'left': 0, 'right': 0}
        self.busy_sides: set[str] = set()

    def configure(self, name: str, config: Mapping[str, Any]) -> None:
        self.config_name = name

    def configured_ports(self) -> dict[str, str]:
        return {}

    def owned_pid(self) -> None:
        return None

    def active(self) -> bool:
        return False

    def state(self, blockers=()) -> dict[str, Any]:
        return {
            'config_name': self.config_name,
            'available': True,
            'status': 'idle',
            'active': False,
            'busy': False,
            'can_calibrate': not blockers,
            'can_position': not blockers,
            'blockers': list(blockers),
            'last_error': None,
            'available_ports': ['/dev/ttyUSB0', '/dev/ttyUSB1'],
            'worker': {'pid': None, 'ready': False, 'idle_timeout_s': 10.0},
            'sides': {
                side: {
                    'configured': True,
                    'port': self.bindings[side],
                    'port_present': True,
                    'status': 'moving' if side in self.busy_sides else 'idle',
                    'target_percent': None,
                    'actual_percent': None,
                    'actual_width_m': None,
                    'last_error': None,
                    'last_command_id': None,
                    'max_width_m': 0.09,
                    'calibrated': True,
                    'motor_id': 1,
                    'calibration_artifact': {
                        'present': True,
                        'valid': True,
                        'status': 'valid',
                        'identity_status': 'verified',
                        'expected': {'port': self.bindings[side], 'motor_id': 1},
                        'observed': {'port': self.bindings[side], 'motor_id': 1},
                    },
                    'revision': self.side_revisions[side],
                    'busy': side in self.busy_sides,
                    'can_calibrate': not blockers,
                    'can_position': not blockers and side not in self.busy_sides,
                    'can_queue_position': not blockers,
                    'queue_position_blockers': list(blockers),
                    'capabilities': {
                        'position': {
                            'can': not blockers and side not in self.busy_sides,
                            'blockers': (
                                list(blockers)
                                + ([f'{side} gripper command is active']
                                   if side in self.busy_sides else [])
                            ),
                        },
                        'queue_position': {
                            'can': not blockers,
                            'blockers': list(blockers),
                        },
                    },
                }
                for side in ('left', 'right')
            },
        }

    def bind_ports(self, bindings: Mapping[str, Any]) -> dict[str, str]:
        for side, port in bindings.items():
            port = gripper_control.validate_ttyusb_port(
                port, require_present=False
            )
            self.bindings[str(side)] = port
        if len(set(self.bindings.values())) != len(self.bindings):
            raise gripper_control.GripperRequestError('same serial port')
        self.calls.append(('bind_ports', dict(bindings)))
        return dict(self.bindings)

    def calibrate(self, side: str, key: str) -> dict[str, Any]:
        self.calls.append(('calibrate', side, key))
        return {'accepted': True, 'command_id': 'cal-1', 'status': 'pending'}

    def position(
        self,
        side: str,
        target: float,
        key: str,
        *,
        expected_side_revision: int,
    ) -> dict[str, Any]:
        if expected_side_revision != self.side_revisions[side]:
            raise gripper_control.GripperConflictError(
                f'Stale {side} gripper revision'
            )
        self.side_revisions[side] += 1
        self.calls.append(('position', side, target, key, expected_side_revision))
        return {
            'accepted': True,
            'command_id': f'pos-{side}',
            'status': 'queued',
            'side_revision': self.side_revisions[side],
        }

    def stop(self, reason: str) -> dict[str, Any]:
        self.calls.append(('stop', reason))
        return {'ok': True, 'stopped': False, 'status': 'idle'}

    def close(self) -> None:
        return None


class FakeRecordingWorkflow:
    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self.calls = calls

    def recording_action(self, action: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(('recording', action, kwargs))
        return {'ok': True, 'action': action}

    def stop_all(self) -> None:
        return None


def _write_minimal_config(root: Path) -> None:
    root.mkdir(parents=True)
    (root / 'test.yaml').write_text(
        'devices:\n  - class: BaseDevice\n    device_class_id: 1\n'
        '    data_shape: [1]\n    data_dtype: float64\n    fps: 1\n',
        encoding='utf-8',
    )


def _write_gripper_binding_config(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "test.yaml").write_text(
        """devices:
  - class: RizonRobot
    device_class_id: 2
    data_shape: [44]
    data_dtype: float64
    fps: 120
    gripper:
      backend: lktech
      enabled: [true, true]
      port: [/dev/ttyUSB1, /dev/ttyUSB0]
      motor_id: [1, 1]
      calibration_files: [calibration/lktech_gripper/left.json, calibration/lktech_gripper/right.json]
""",
        encoding="utf-8",
    )


def test_config_switch_applies_panel_gripper_port_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, 'rizon_shared_memory_names', lambda: [])
    config_dir = Path(__file__).resolve().parents[3] / 'configs'
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
        inference_preferences_path=tmp_path / 'preferences.json',
    )
    app.testing = True

    response = app.test_client().post(
        '/api/config/select',
        json={
            'name': CONFIG_NAME,
            'gripper_overrides': {
                'left': '/dev/ttyUSB8',
                'right': '/dev/ttyUSB9',
            },
        },
    )

    assert response.status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    assert controller.grippers.configured_ports() == {
        'left': '/dev/ttyUSB8',
        'right': '/dev/ttyUSB9',
    }
    assert controller._gripper_port_overrides == {
        'left': '/dev/ttyUSB8',
        'right': '/dev/ttyUSB9',
    }


def test_gripper_binding_updates_effective_preflight_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "rizon_shared_memory_names", lambda: [])
    # This API test owns no robot; a host/sibling test's SHM must not affect it.
    monkeypatch.setattr(
        server.DeviceLifecycleManager, "_shared_memory_name",
        lambda *_: f"test_gripper_binding_{tmp_path.name}",
    )
    config_dir = tmp_path / "configs"
    _write_gripper_binding_config(config_dir)
    manager = FakeGripperManager()
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
        gripper_control=manager,
    )
    app.testing = True
    client = app.test_client()
    try:
        selected = client.post("/api/config/select", json={"name": "test"})
        assert selected.status_code == 200
        rebound = client.post(
            "/api/grippers/bindings",
            json={
                "revision": selected.get_json()["revision"],
                "bindings": {
                    "left": "/dev/ttyUSB0",
                    "right": "/dev/ttyUSB1",
                },
            },
        )
        assert rebound.status_code == 200

        controller = app.config["DEVICE_CONTROLLER"]
        robot = next(
            device
            for device in controller._effective_config["devices"]
            if device.get("class") == "RizonRobot"
        )
        assert robot["gripper"]["port"] == ["/dev/ttyUSB0", "/dev/ttyUSB1"]

        report = client.get("/api/preflight?config_name=test")
        assert report.status_code == 200
        gripper_check = next(
            check
            for check in report.get_json()["checks"]
            if check["id"] == "gripper_calibration"
        )
        assert gripper_check["status"] == "ok"
        assert gripper_check["evidence"]["identity_mismatch"] == []
    finally:
        app.config["DEVICE_CONTROLLER"].shutdown()


def test_gripper_api_contract_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, 'rizon_shared_memory_names', lambda: [])
    config_dir = tmp_path / 'configs'
    _write_minimal_config(config_dir)
    manager = FakeGripperManager()
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
        gripper_control=manager,
    )
    app.testing = True
    client = app.test_client()
    selected = client.post('/api/config/select', json={'name': 'test'}).get_json()
    revision = selected['revision']

    response = client.post(
        '/api/grippers/position',
        json={
            'side': 'left',
            'target_percent': 25.0,
            'revision': revision,
            'side_revision': 0,
            'idempotency_key': 'position-0001',
        },
    )
    assert response.status_code == 202
    assert manager.calls[-1] == ('position', 'left', 25.0, 'position-0001', 0)
    assert client.get('/api/grippers').get_json()['grippers']['sides']['left']['configured']

    right = client.post(
        '/api/grippers/position',
        json={
            'side': 'right',
            'target_percent': 75.0,
            'revision': revision,
            'side_revision': 0,
            'idempotency_key': 'position-right-0001',
        },
    )
    assert right.status_code == 202
    assert right.get_json()['revision'] == revision
    assert manager.calls[-1] == (
        'position', 'right', 75.0, 'position-right-0001', 0
    )

    same_side_stale = client.post(
        '/api/grippers/position',
        json={
            'side': 'left',
            'target_percent': 30.0,
            'revision': revision,
            'side_revision': 0,
            'idempotency_key': 'position-left-stale',
        },
    )
    assert same_side_stale.status_code == 409

    stale = client.post(
        '/api/grippers/calibrate',
        json={
            'side': 'both',
            'revision': revision - 1,
            'idempotency_key': 'calibrate-0001',
        },
    )
    assert stale.status_code == 409
    unexpected = client.post(
        '/api/grippers/position',
        json={
            'side': 'left',
            'target_percent': 50,
            'revision': response.get_json()['revision'],
            'side_revision': 0,
            'idempotency_key': 'position-0002',
            'port': '/dev/ttyUSB9',
        },
    )
    assert unexpected.status_code == 422

    ports = client.get('/api/grippers/ports')
    assert ports.status_code == 200
    assert ports.get_json()['available_ports'] == ['/dev/ttyUSB0', '/dev/ttyUSB1']
    current_revision = response.get_json()['revision']
    rebound = client.post(
        '/api/grippers/bindings',
        json={
            'revision': current_revision,
            'bindings': {
                'left': '/dev/ttyUSB0',
                'right': '/dev/ttyUSB1',
            },
        },
    )
    assert rebound.status_code == 200
    assert rebound.get_json()['bindings'] == {
        'left': '/dev/ttyUSB0',
        'right': '/dev/ttyUSB1',
    }
    malicious = client.post(
        '/api/grippers/bindings',
        json={
            'revision': rebound.get_json()['revision'],
            'bindings': {'left': '/dev/ttyUSB0;touch /tmp/pwned'},
        },
    )
    assert malicious.status_code == 422
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_gripper_api_accepts_latest_target_while_same_side_is_moving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, 'rizon_shared_memory_names', lambda: [])
    config_dir = tmp_path / 'configs'
    _write_minimal_config(config_dir)
    manager = FakeGripperManager()
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
        gripper_control=manager,
    )
    app.testing = True
    client = app.test_client()
    revision = client.post(
        '/api/config/select', json={'name': 'test'}
    ).get_json()['revision']

    manager.busy_sides.add('left')
    response = client.post(
        '/api/grippers/position',
        json={
            'side': 'left',
            'target_percent': 0.0,
            'revision': revision,
            'side_revision': 0,
            'idempotency_key': 'position-moving-latest-target',
        },
    )

    assert response.status_code == 202
    assert manager.calls[-1] == (
        'position', 'left', 0.0,
        'position-moving-latest-target', 0,
    )
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_recording_transition_stops_gripper_first(tmp_path: Path) -> None:
    config_dir = tmp_path / 'configs'
    _write_minimal_config(config_dir)
    manager = FakeGripperManager()
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
        gripper_control=manager,
    )
    controller = app.config['DEVICE_CONTROLLER']
    calls: list[tuple[Any, ...]] = manager.calls
    controller.workflow = FakeRecordingWorkflow(calls)

    result = controller.control_recording(
        'accept_segment',
        episode_id='episode-1',
        expected_checkpoint_revision=0,
        idempotency_key='recording-0001',
    )

    assert result['ok'] is True
    assert calls[0] == ('stop', 'recording_accept_segment')
    assert calls[1][0:2] == ('recording', 'accept_segment')
    controller.shutdown()
