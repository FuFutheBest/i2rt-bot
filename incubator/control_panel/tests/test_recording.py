"""Hardware-free tests for the owned recording-control spool."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from incubator.control_panel import server
from incubator.control_panel.recording import (
    RECORDING_ACTIONS,
    RecordingConflictError,
    RecordingCoordinator,
    RecordingRequestError,
    default_recording_profile_check,
    recording_config_state,
    resolve_recording_python,
)
from incubator.control_panel.runtime_config import compose_runtime_config
from incubator.control_panel.recording_control import (
    RecordingControlPlane,
    WholeEpisodeCommandConsumer,
)
from utils.runtime_profiles import REGISTRY_ENV, RuntimeRegistry


@pytest.fixture(autouse=True)
def _isolated_runtime_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "missing-runtime-profiles.json"))


def _status(
    coordinator: RecordingCoordinator,
    *,
    owner_pid: int = 2001,
    revision: int = 3,
) -> Dict[str, Any]:
    payload = {
        'schema_version': 1,
        'run_id': coordinator.run_id,
        'owner_pid': owner_pid,
        'status': 'recording',
        'ready': True,
        'active': True,
        'error': None,
        'revision': 19,
        'reference': {'kind': 'control_shm', 'name': 'RizonRobot_2_control'},
        'episode': {
            'id': 'episode-abc',
            'index': 7,
            'frame_count': 120,
            'elapsed_ms': 4000.0,
            'first_timestamp_ns': 1000,
            'latest_timestamp_ns': 4000001000,
            'checkpoint_revision': revision,
            'untagged_start_frame': 80,
        },
        'segments': [
            {
                'checkpoint_revision': 1,
                'start_frame': 0,
                'end_frame': 80,
                'label': 'accept',
            }
        ],
        'checkpoints': [
            {
                'revision': 1,
                'frame': 80,
                'timestamp_ns': 2000001000,
            }
        ],
    }
    path = coordinator.status_dir / 'recorder.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return payload


def test_recorder_arm_satisfies_panel_readiness(tmp_path: Path) -> None:
    """The recorder status handshake must unblock required inference startup."""
    coordinator = RecordingCoordinator(
        tmp_path / "runs",
        pid_validator=lambda candidate, ancestor: candidate == ancestor,
    )
    control_dir = coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())

    recorder = RecordingControlPlane(
        control_dir,
        str(coordinator.run_id),
        owner_pid=os.getpid(),
        status_flush_interval_s=0.0,
    )
    recorder.arm()

    process = SimpleNamespace(poll=lambda: None)
    ready, error = coordinator.wait_until_ready(process, timeout=0.1, poll_interval=0.0)

    assert ready is True
    assert error == ""
    assert coordinator.state()["status"] == "armed"


def _coordinator(tmp_path: Path) -> RecordingCoordinator:
    coordinator = RecordingCoordinator(
        tmp_path / 'runs',
        pid_validator=lambda candidate, ancestor: (
            candidate == 2001 and ancestor == 2000
        ),
    )
    coordinator.prepare_run()
    coordinator.bind_owner(2000)
    _status(coordinator)
    return coordinator


def _home_target(candidate_id: str = "left-01") -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "home_pose_candidates",
        "config_name": "demo",
        "config_hash": "a" * 64,
        "arms": {
            "left": {
                "candidate_id": candidate_id,
                "joints": [0.1] * 7,
                "collision_checked": False,
            },
            "right": {
                "candidate_id": "right-01",
                "joints": [-0.1] * 7,
                "collision_checked": False,
            },
        },
    }


@pytest.mark.parametrize('status', ['finalizing', 'transitioning'])
def test_expected_episode_transition_is_healthy_without_ready_flag(
    status: str,
) -> None:
    state = {
        'status': status,
        'recorder_ready': False,
        'error': None,
    }

    assert RecordingCoordinator.is_healthy(state) is True
    assert RecordingCoordinator.is_healthy({**state, 'error': 'flush failed'}) is False
    assert RecordingCoordinator.is_healthy(
        {**state, 'status': 'recording'}
    ) is False


def test_recording_python_resolution_is_controlled_and_prioritized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        'policy': {
            'infer_rollout': {'python_path': '/trusted/policy/python'}
        }
    }
    assert resolve_recording_python(config) == Path('/trusted/policy/python')

    monkeypatch.setenv(
        'SUPERINFERENCE_RECORDING_PYTHON', '/trusted/env/python'
    )
    assert resolve_recording_python(config) == Path('/trusted/env/python')
    assert resolve_recording_python(
        config, explicit=Path('/trusted/cli/python')
    ) == Path('/trusted/cli/python')

    monkeypatch.delenv('SUPERINFERENCE_RECORDING_PYTHON')
    assert resolve_recording_python({}) is None


def test_recording_python_uses_config_specific_runtime_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "runtime_profiles.json"
    registry = RuntimeRegistry.load(repo_root=tmp_path, path=registry_path)
    registry.register("nedf3", python=Path(sys.executable), imports=["json"])
    registry.verify("nedf3")
    registry.bind("recording", "nedf3", config_name="policy_infer/openpi/demo")
    registry.save()
    monkeypatch.setenv(REGISTRY_ENV, str(registry_path))

    assert resolve_recording_python(
        {},
        config_name="policy_infer/openpi/demo",
        repo_root=tmp_path,
    ) == Path(sys.executable)


def test_recording_profile_check_requires_writer_and_reader_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / 'python'
    python.write_text('#!/bin/sh\n', encoding='utf-8')
    python.chmod(0o755)
    calls = []
    results = iter(
        (
            SimpleNamespace(
                returncode=13,
                stdout='',
                stderr=(
                    'missing required NEDF3 recorder API: '
                    'NEDFWriterBase.write_message'
                ),
            ),
            SimpleNamespace(returncode=0, stdout='', stderr=''),
        )
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return next(results)

    monkeypatch.setattr('incubator.control_panel.recording.subprocess.run', run)
    available, error = default_recording_profile_check(python)
    assert available is False
    assert error == (
        'missing required NEDF3 recorder API: NEDFWriterBase.write_message'
    )
    script = calls[0][0][-1]
    for required in (
        'write_message',
        'iter_decoded_messages',
        'get_raw_topics_timestamps',
        'get_writer',
        'get_reader',
        'dm3_core.nedf',
        'dm3_core.mcap',
        'dm3-core',
        '0.2.18',
        '2.2.0',
    ):
        assert required in script
    assert calls[0][1]['timeout'] == 15.0
    assert calls[0][1]['check'] is False

    assert default_recording_profile_check(python) == (True, '')


def test_target_dual_pi05_config_requires_recording_and_action_names_match_ui() -> None:
    config = compose_runtime_config(
        'policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi'
    )
    rollout = config['policy']['infer_rollout']
    assert rollout['enabled'] is True
    assert recording_config_state(config) == (True, True, True)
    assert RECORDING_ACTIONS == frozenset(
        {
            'episode_accept',
            'episode_discard',
            'segment_accept',
            'segment_reject',
        }
    )


def test_run_spool_is_private_and_command_has_no_browser_timeline_fields(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)

    result = coordinator.submit_action(
        'segment_accept',
        episode_id='episode-abc',
        expected_checkpoint_revision=3,
        idempotency_key='request-0001',
    )

    assert result['status'] == 'pending'
    public = coordinator.state()
    assert public['segments'][0]['tag'] == 'accept'
    assert public['checkpoints'][0]['id'] == 'checkpoint-1'
    assert os.stat(coordinator.control_dir).st_mode & 0o077 == 0
    command_path = coordinator.commands_dir / f"{result['command_id']}.json"
    command = json.loads(command_path.read_text(encoding='utf-8'))
    assert command['action'] == 'segment_accept'
    assert command['episode_id'] == 'episode-abc'
    assert command['expected_revision'] == 19
    assert command['expected_checkpoint_revision'] == 3
    assert 'frame' not in command
    assert 'timestamp_ns' not in command
    assert 'path' not in command


def test_recording_commands_are_revision_checked_and_idempotent(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    first = coordinator.submit_action(
        'segment_reject',
        episode_id='episode-abc',
        expected_checkpoint_revision=3,
        idempotency_key='request-0002',
    )
    duplicate = coordinator.submit_action(
        'segment_reject',
        episode_id='episode-abc',
        expected_checkpoint_revision=3,
        idempotency_key='request-0002',
    )
    assert duplicate['command_id'] == first['command_id']

    with pytest.raises(RecordingConflictError, match='another recording action'):
        coordinator.submit_action(
            'episode_accept',
            episode_id='episode-abc',
            expected_checkpoint_revision=3,
            idempotency_key='request-0002',
        )
    with pytest.raises(RecordingConflictError, match='revision changed'):
        coordinator.submit_action(
            'segment_accept',
            episode_id='episode-abc',
            expected_checkpoint_revision=2,
            idempotency_key='request-0003',
        )


def test_whole_episode_home_target_is_durable_and_idempotency_bound(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    target = _home_target()
    first = coordinator.submit_action(
        'episode_accept',
        episode_id='episode-abc',
        expected_checkpoint_revision=3,
        idempotency_key='home-request',
        home_target=target,
    )
    command = json.loads(
        (coordinator.commands_dir / f"{first['command_id']}.json").read_text(
            encoding='utf-8'
        )
    )
    assert command['home_target'] == target

    changed = _home_target(candidate_id='left-02')
    with pytest.raises(RecordingConflictError, match='another recording action'):
        coordinator.submit_action(
            'episode_accept',
            episode_id='episode-abc',
            expected_checkpoint_revision=3,
            idempotency_key='home-request',
            home_target=changed,
        )


def test_segment_command_rejects_sampled_home_target(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    with pytest.raises(RecordingRequestError, match='whole-episode'):
        coordinator.submit_action(
            'segment_accept',
            episode_id='episode-abc',
            expected_checkpoint_revision=3,
            idempotency_key='segment-home-request',
            home_target=_home_target(),
        )


def test_ack_must_match_run_command_and_process_tree(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    command = coordinator.submit_action(
        'segment_accept',
        episode_id='episode-abc',
        expected_checkpoint_revision=3,
        idempotency_key='request-0004',
    )
    ack_path = coordinator.acks_dir / f"{command['command_id']}.json"
    ack_path.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'run_id': coordinator.run_id,
                'owner_pid': 2001,
                'command_id': command['command_id'],
                'idempotency_key': 'request-0004',
                'action': 'segment_accept',
                'status': 'applied',
                'episode_id': 'episode-abc',
                'revision': 4,
            }
        ),
        encoding='utf-8',
    )

    # The saver may exit immediately after writing a shutdown/action ACK.
    # Its PID was already verified from status before the command was sent.
    coordinator._pid_validator = lambda candidate, ancestor: False

    applied = coordinator.submit_action(
        'segment_accept',
        episode_id='episode-abc',
        expected_checkpoint_revision=3,
        idempotency_key='request-0004',
    )
    assert applied['status'] == 'applied'
    assert applied['revision'] == 4
    assert coordinator.state()['last_command']['status'] == 'applied'

    ack_path.write_text(
        ack_path.read_text(encoding='utf-8').replace(
            '"owner_pid": 2001', '"owner_pid": 9999'
        ),
        encoding='utf-8',
    )
    coordinator._acks.clear()
    state = coordinator.state()
    assert state['status'] == 'error'
    assert 'outside the inference tree' in state['error']


def test_shutdown_bypasses_ui_revision_and_waits_for_owned_ack(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    elapsed = [0.0]

    def sleep(duration: float) -> None:
        command_path = next(coordinator.commands_dir.glob('*.json'))
        command = json.loads(command_path.read_text(encoding='utf-8'))
        assert command['action'] == 'shutdown'
        assert command['expected_revision'] is None
        ack = {
            'schema_version': 1,
            'run_id': coordinator.run_id,
            'owner_pid': 2001,
            'command_id': command['command_id'],
            'idempotency_key': command['idempotency_key'],
            'action': 'shutdown',
            'status': 'applied',
            'episode_id': 'episode-abc',
            'revision': 20,
            'error': None,
        }
        (coordinator.acks_dir / command_path.name).write_text(
            json.dumps(ack), encoding='utf-8'
        )
        elapsed[0] += duration

    coordinator._clock = lambda: elapsed[0]
    coordinator._sleep = sleep

    assert coordinator.request_shutdown(1.0) == (True, None)


def test_panel_and_consumer_share_spool_end_to_end(tmp_path: Path) -> None:
    panel = RecordingCoordinator(
        tmp_path / 'runs',
        pid_validator=lambda candidate, ancestor: candidate == ancestor,
    )
    control_dir = panel.prepare_run()
    panel.bind_owner(os.getpid())
    recorder = RecordingControlPlane(
        control_dir,
        str(panel.run_id),
        owner_pid=os.getpid(),
        status_flush_interval_s=0.0,
    )
    recorder.arm()
    assert panel.state()['status'] == 'armed'

    def dense_handler(command: Dict[str, Any]) -> None:
        action = command['action']
        if action == 'segment_accept':
            recorder.append_segment('accept', command)
        elif action == 'segment_reject':
            recorder.append_segment('reject', command)
        elif action == 'shutdown':
            recorder.mark_incomplete({'code': 'panel_shutdown'})

    episode_dir = tmp_path / 'episode_0000_temp'
    episode_dir.mkdir()
    first_episode = recorder.begin_episode(index=0, episode_dir=episode_dir)
    for timestamp in range(1, 6):
        recorder.note_reference_commit(timestamp * 1_000_000)

    first = panel.submit_action(
        'segment_accept',
        episode_id=first_episode,
        expected_checkpoint_revision=0,
        idempotency_key='e2e-segment-accept',
    )
    assert first['status'] == 'pending'
    assert recorder.process_commands(
        dense_handler,
        allowed_actions={'segment_accept', 'segment_reject', 'shutdown'},
    ) == 1
    accepted = panel.state()
    assert accepted['last_command']['status'] == 'applied'
    assert accepted['episode']['checkpoint_revision'] == 1
    assert accepted['segments'][0] == {
        'checkpoint_id': 'checkpoint-1',
        'previous_checkpoint_id': 'checkpoint-0',
        'start_frame': 0,
        'end_frame': 5,
        'tag': 'accept',
    }

    for timestamp in range(6, 10):
        recorder.note_reference_commit(timestamp * 1_000_000)
    panel.submit_action(
        'segment_reject',
        episode_id=first_episode,
        expected_checkpoint_revision=1,
        idempotency_key='e2e-segment-reject',
    )
    recorder.process_commands(
        dense_handler,
        allowed_actions={'segment_accept', 'segment_reject', 'shutdown'},
    )
    rejected = panel.state()
    assert rejected['episode']['checkpoint_revision'] == 2
    assert rejected['segments'][1]['tag'] == 'reject'
    assert rejected['segments'][1]['start_frame'] == 5
    assert rejected['segments'][1]['end_frame'] == 9

    panel.submit_action(
        'episode_accept',
        episode_id=first_episode,
        expected_checkpoint_revision=2,
        idempotency_key='e2e-episode-accept',
    )
    # Saver owns only dense/shutdown actions; whole actions remain for policy.
    assert recorder.process_commands(
        dense_handler,
        allowed_actions={'segment_accept', 'segment_reject', 'shutdown'},
    ) == 0
    whole = WholeEpisodeCommandConsumer(
        control_dir, str(panel.run_id), owner_pid=os.getpid()
    )

    def accept_transition(command: Dict[str, Any]) -> None:
        recorder.accept_tail(command)
        recorder.arm()
        next_dir = tmp_path / 'episode_0001_temp'
        next_dir.mkdir()
        recorder.begin_episode(index=1, episode_dir=next_dir)

    assert whole.process_commands(accept_transition) == 1
    after_accept = panel.state()
    assert after_accept['last_command']['status'] == 'applied'
    assert after_accept['episode']['id'] != first_episode
    assert after_accept['status'] == 'recording'

    second_episode = after_accept['episode']['id']
    recorder.note_reference_commit(20_000_000)
    panel.submit_action(
        'episode_discard',
        episode_id=second_episode,
        expected_checkpoint_revision=0,
        idempotency_key='e2e-episode-discard',
    )

    discarded_tail: Dict[str, Any] = {}

    def discard_transition(command: Dict[str, Any]) -> None:
        tail = recorder.reject_tail(command)
        assert tail is not None
        discarded_tail.update(tail)
        recorder.arm()
        next_dir = tmp_path / 'episode_0002_temp'
        next_dir.mkdir()
        recorder.begin_episode(index=2, episode_dir=next_dir)

    assert whole.process_commands(discard_transition) == 1
    after_discard = panel.state()
    assert after_discard['last_command']['status'] == 'applied'
    assert after_discard['episode']['id'] != second_episode
    assert discarded_tail['label'] == 'reject'

    def shutdown_sleep(duration: float) -> None:
        del duration
        recorder.process_commands(
            dense_handler,
            allowed_actions={'segment_accept', 'segment_reject', 'shutdown'},
        )

    panel._sleep = shutdown_sleep
    assert panel.request_shutdown(1.0) == (True, None)
    assert panel.state()['status'] == 'incomplete'


def test_recording_api_rejects_browser_frame_time_and_path(
    tmp_path: Path,
) -> None:
    app = server.create_app(config_dir=tmp_path)
    controller = app.config['DEVICE_CONTROLLER']

    class FakeWorkflow:
        def state(self, devices: Any) -> Dict[str, Any]:
            del devices
            return {
                'config_name': 'fixture',
                'runtime': {'active': False},
                'inference': {'running': True},
                'profiles': {},
                'recording': {
                    'configured': True,
                    'enabled': True,
                    'required': True,
                    'status': 'recording',
                },
                'stop_order': [
                    'recording',
                    'inference',
                    'action_executor',
                    'device_manager',
                    'devices',
                ],
            }

        def recording_action(self, action: str, **kwargs: Any) -> Dict[str, Any]:
            return {
                'ok': True,
                'command': {
                    'command_id': 'command-1',
                    'action': action,
                    'status': 'pending',
                },
                'recording': {'status': 'recording', **kwargs},
            }

    controller.workflow = FakeWorkflow()
    client = app.test_client()
    invalid = client.post(
        '/api/recording/actions/segment_accept',
        json={
            'episode_id': 'episode-abc',
            'expected_checkpoint_revision': 3,
            'idempotency_key': 'request-0005',
            'frame': 120,
            'timestamp_ns': 99,
            'path': '/tmp/forbidden',
        },
    )
    assert invalid.status_code == 422
    assert 'Unexpected recording fields' in invalid.get_json()['error']

    accepted = client.post(
        '/api/recording/actions/segment_accept',
        json={
            'episode_id': 'episode-abc',
            'expected_checkpoint_revision': 3,
            'idempotency_key': 'request-0005',
        },
    )
    assert accepted.status_code == 202
    assert accepted.get_json()['command']['status'] == 'pending'


def test_standalone_reset_api_accepts_only_current_control_context(
    tmp_path: Path,
) -> None:
    app = server.create_app(config_dir=tmp_path)
    controller = app.config['DEVICE_CONTROLLER']
    calls: list[dict[str, Any]] = []

    def reset(**kwargs: Any) -> Dict[str, Any]:
        calls.append(dict(kwargs))
        return {
            'ok': True,
            'command': {'command_id': 'reset-1', 'status': 'pending'},
        }

    controller.control_inference_reset = reset
    client = app.test_client()
    response = client.post(
        '/api/inference/reset',
        json={
            'idempotency_key': 'reset-request-1',
            'expected_selected_config': 'fixture',
            'expected_revision': 7,
        },
    )
    assert response.status_code == 202
    assert calls == [
        {
            'idempotency_key': 'reset-request-1',
            'expected_selected_config': 'fixture',
            'expected_revision': 7,
        }
    ]

    invalid = client.post(
        '/api/inference/reset',
        json={
            'idempotency_key': 'reset-request-2',
            'expected_selected_config': 'fixture',
            'expected_revision': 7,
            'home_target': '/tmp/not-browser-controlled',
        },
    )
    assert invalid.status_code == 400


def test_pause_and_resume_api_accept_only_current_control_context(
    tmp_path: Path,
) -> None:
    app = server.create_app(config_dir=tmp_path)
    controller = app.config['DEVICE_CONTROLLER']
    calls: list[dict[str, Any]] = []

    def control(action: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append({'action': action, **kwargs})
        return {
            'ok': True,
            'command': {'command_id': f'{action}-1', 'status': 'pending'},
        }

    controller.control_inference_pause_resume = control
    client = app.test_client()
    for action in ('pause', 'resume'):
        response = client.post(
            f'/api/inference/{action}',
            json={
                'idempotency_key': f'{action}-request-1',
                'expected_selected_config': 'fixture',
                'expected_revision': 7,
            },
        )
        assert response.status_code == 202
        assert response.get_json()['command']['status'] == 'pending'

    assert calls == [
        {
            'action': 'pause',
            'idempotency_key': 'pause-request-1',
            'expected_selected_config': 'fixture',
            'expected_revision': 7,
        },
        {
            'action': 'resume',
            'idempotency_key': 'resume-request-1',
            'expected_selected_config': 'fixture',
            'expected_revision': 7,
        },
    ]

    invalid = client.post(
        '/api/inference/pause',
        json={
            'idempotency_key': 'pause-request-2',
            'expected_selected_config': 'fixture',
            'expected_revision': 7,
            'home_target': {},
        },
    )
    assert invalid.status_code == 400
