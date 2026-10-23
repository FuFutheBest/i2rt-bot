"""Safety and API contracts for read-only gripper position probes."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from incubator.control_panel import gripper_control
from incubator.control_panel.gripper_control import GripperProfile, _WorkerSlot
from incubator.control_panel.hardware_discovery import HardwareSnapshot


def _profile(side: str = 'left') -> GripperProfile:
    return GripperProfile(
        side=side,
        port='/dev/ttyUSB1' if side == 'left' else '/dev/ttyUSB0',
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


def _verified_artifact(profile: GripperProfile) -> dict[str, Any]:
    return {
        "name": Path(profile.calibration_file).name,
        "present": True,
        "limits_valid": True,
        "valid": True,
        "status": "valid",
        "identity_status": "verified",
        "expected": {"motor_id": profile.motor_id, "port": profile.port},
        "observed": {"motor_id": profile.motor_id, "port": profile.port},
        "reason": None,
    }


class ReadOnlyMotor:
    events: list[Any] = []

    def __init__(self, port: str, **kwargs: Any) -> None:
        self.port = port
        self.kwargs = kwargs
        type(self).events.append(('init', port, kwargs))

    def connect(self) -> None:
        type(self).events.append('connect')

    def read_multi_turn_angle(self) -> float:
        type(self).events.append('read_multi_turn_angle')
        return 560.0

    def disconnect(self) -> None:
        type(self).events.append('disconnect')

    def motor_on(self) -> None:  # pragma: no cover - must never be reached
        raise AssertionError('read-only probe must never energize the motor')


def test_probe_reads_calibrated_position_without_enabling_or_moving() -> None:
    ReadOnlyMotor.events = []

    result = gripper_control._probe_gripper_read_only(
        _profile(),
        motor_loader=lambda: ReadOnlyMotor,
        preflight=lambda profiles: None,
    )

    expected = math.sin(math.radians(45.0) / 2.0) / math.sin(
        math.radians(90.0) / 2.0
    ) * 100.0
    assert result['side'] == 'left'
    assert result['target_percent'] is None
    assert result['actual_percent'] == pytest.approx(expected)
    assert ReadOnlyMotor.events == [
        ('init', '/dev/ttyUSB1', {
            'motor_id': 1,
            'baudrate': 115200,
            'timeout': 0.1,
        }),
        'connect',
        'read_multi_turn_angle',
        'disconnect',
    ]
    source = inspect.getsource(gripper_control._probe_gripper_read_only)
    for forbidden in ('Enable(', 'motor_on(', 'Move(', 'calibrat'):
        assert forbidden not in source


def _manager() -> gripper_control.GripperControlManager:
    manager = gripper_control.GripperControlManager(
        auto_monitor=False,
        sdk_module=Path('/unused/sdk.py'),
        device_python=Path('/unused/python'),
        port_discovery=lambda: ['/dev/ttyUSB0', '/dev/ttyUSB1'],
        port_validator=lambda value: str(value),
        permission_granter=lambda value: str(value),
        permission_policy_check=lambda value: None,
    )
    manager._profiles = {side: _profile(side) for side in ('left', 'right')}
    manager._config_name = 'cfg'
    manager._status = 'idle'
    for side in ('left', 'right'):
        manager._sides[side]['configured'] = True
        manager._sides[side]['status'] = 'idle'
    return manager


def test_probe_completion_uses_separate_markers_and_failure_preserves_slider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gripper_control,
        "_calibration_artifact_state",
        _verified_artifact,
    )
    manager = _manager()
    manager._sides['left'].update(
        {
            'actual_percent': 42.0,
            'last_command_id': 'calibration-1',
            'last_completed_command_id': 'calibration-1',
            'last_completed_action': 'calibrate',
            'last_probe_command_id': 'probe-1',
            'status': 'probing',
        }
    )
    slot = _WorkerSlot(
        lane='left',
        process=None,
        connection=None,
        owner_token='owner-probe-1',
        inflight={
            'command_id': 'probe-1',
            'action': 'probe',
            'side': 'left',
        },
    )
    manager._workers = {'left': slot}

    manager._apply_event_locked(
        {
            'event': 'completed',
            'owner_token': 'owner-probe-1',
            'command_id': 'probe-1',
            'results': [
                {
                    'side': 'left',
                    'target_percent': None,
                    'actual_percent': 57.5,
                    'actual_width_m': 0.052,
                }
            ],
        }
    )

    state = manager._sides['left']
    assert state['status'] == 'idle'
    assert state['actual_percent'] == pytest.approx(57.5)
    assert state['last_probe_completed_command_id'] == 'probe-1'
    assert state['last_completed_command_id'] == 'calibration-1'
    assert state['last_completed_action'] == 'calibrate'

    state.update(
        {
            'status': 'probing',
            'last_probe_command_id': 'probe-2',
        }
    )
    slot.inflight = {
        'command_id': 'probe-2',
        'action': 'probe',
        'side': 'left',
    }
    manager._apply_event_locked(
        {
            'event': 'failed',
            'owner_token': 'owner-probe-1',
            'command_id': 'probe-2',
            'error': 'injected read timeout',
            'hardware_unknown': False,
        }
    )

    state = manager._sides['left']
    assert state['status'] == 'unavailable'
    assert state['actual_percent'] == pytest.approx(57.5)
    assert state['last_probe_completed_command_id'] == 'probe-1'
    assert manager._hardware_unknown is False
    manager._workers = {}
    manager._available = lambda: (True, None)
    rendered = manager.state()['sides']['left']
    assert rendered['can_position'] is False
    assert rendered['can_probe'] is True


def test_probe_command_is_side_scoped_and_calibration_interlocked(monkeypatch) -> None:
    manager = _manager()
    sent: list[dict[str, Any]] = []

    def send(command: dict[str, Any]) -> None:
        sent.append(dict(command))
        side = str(command['side'])
        manager._sides[side]['status'] = 'probing'
        manager._sides[side]['last_probe_command_id'] = command['command_id']

    monkeypatch.setattr(manager, '_send_locked', send)
    result = manager.probe(
        'left',
        'probe-key-0001',
        expected_side_revision=0,
    )

    assert result['accepted'] is True
    assert result['action'] == 'probe'
    assert result['side_revision'] == 1
    assert sent[0]['action'] == 'probe'
    assert manager._side_revisions == {'left': 1, 'right': 0}

    manager._calibration = SimpleNamespace(sides=('left', 'right'))
    with pytest.raises(gripper_control.GripperConflictError, match='calibration'):
        manager.probe(
            'right',
            'probe-key-0002',
            expected_side_revision=0,
        )


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


class ProbeApiManager:
    def __init__(self) -> None:
        self.side_revisions = {'left': 0, 'right': 0}
        self.calls: list[Any] = []
        self.config_name: str | None = None

    def configure(self, name: str, config: Mapping[str, Any]) -> None:
        self.config_name = name

    def configured_ports(self) -> dict[str, str]:
        return {}

    def owned_pid(self) -> None:
        return None

    def active(self) -> bool:
        return False

    def state(self, blockers=()) -> dict[str, Any]:
        sides = {}
        for side, port in (('left', '/dev/ttyUSB1'), ('right', '/dev/ttyUSB0')):
            allowed = not blockers
            sides[side] = {
                'configured': True,
                'port': port,
                'port_present': True,
                'status': 'idle',
                'revision': self.side_revisions[side],
                'busy': False,
                'calibrated': True,
                'can_calibrate': allowed,
                'can_position': allowed,
                'can_probe': allowed,
                'blockers': list(blockers),
                'calibration_blockers': list(blockers),
                'position_blockers': list(blockers),
                'probe_blockers': list(blockers),
                'capabilities': {
                    name: {'can': allowed, 'blockers': list(blockers)}
                    for name in ('calibrate', 'position', 'probe')
                },
            }
        return {
            'config_name': self.config_name,
            'available': True,
            'status': 'idle',
            'active': False,
            'busy': False,
            'can_calibrate': not blockers,
            'can_position': not blockers,
            'can_probe': not blockers,
            'blockers': list(blockers),
            'available_ports': ['/dev/ttyUSB0', '/dev/ttyUSB1'],
            'sides': sides,
        }

    def probe(
        self,
        side: str,
        key: str,
        *,
        expected_side_revision: int,
    ) -> dict[str, Any]:
        if expected_side_revision != self.side_revisions[side]:
            raise gripper_control.GripperConflictError('stale probe revision')
        self.side_revisions[side] += 1
        self.calls.append(('probe', side, key, expected_side_revision))
        return {
            'accepted': True,
            'action': 'probe',
            'side': side,
            'command_id': f'probe-{side}',
            'status': 'pending',
            'side_revision': self.side_revisions[side],
        }

    def stop(self, reason: str) -> dict[str, Any]:
        return {'ok': True, 'stopped': False, 'status': 'idle'}

    def close(self) -> None:
        return None


def test_probe_api_keeps_global_revision_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from incubator.control_panel import server

    monkeypatch.setattr(server, 'rizon_shared_memory_names', lambda: [])

    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    (config_dir / 'test.yaml').write_text(
        'devices:\n  - class: BaseDevice\n    device_class_id: 1\n'
        '    data_shape: [1]\n    data_dtype: float64\n    fps: 1\n',
        encoding='utf-8',
    )
    manager = ProbeApiManager()
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
        '/api/grippers/probe',
        json={
            'side': 'left',
            'revision': revision,
            'side_revision': 0,
            'idempotency_key': 'probe-api-0001',
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload['revision'] == revision
    assert payload['command']['action'] == 'probe'
    assert manager.calls == [('probe', 'left', 'probe-api-0001', 0)]
    app.config['DEVICE_CONTROLLER'].shutdown()
