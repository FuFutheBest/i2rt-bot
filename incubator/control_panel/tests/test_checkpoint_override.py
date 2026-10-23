"""Hardware-free checkpoint browser and session override regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from incubator.control_panel import server
from incubator.control_panel.hardware_discovery import HardwareSnapshot
from incubator.control_panel.workflow import WorkflowLifecycle


class StaticDiscovery:
    def __init__(self) -> None:
        self.snapshot = HardwareSnapshot(
            revision='checkpoint-static-discovery',
            generated_at='2026-07-14T00:00:00+00:00',
            interfaces=[],
            cameras=[],
            robots=[],
            providers=[],
            warnings=[],
        )

    def discover(self) -> HardwareSnapshot:
        return self.snapshot


class FakeRuntimeProcess:
    def __init__(self) -> None:
        self.pid = 73001
        self.alive = True
        self.exitcode = None

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False
        self.exitcode = -15

    def join(self, timeout: float = 0.0) -> None:
        del timeout

    def kill(self) -> None:
        self.alive = False
        self.exitcode = -9


def _checkpoint(root: Path, name: str) -> Path:
    checkpoint = root / name
    (checkpoint / 'params').mkdir(parents=True)
    return checkpoint.resolve()


def _write_config(config_dir: Path, name: str, checkpoint: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    source = config_dir / f'{name}.yaml'
    source.write_text(
        f"""devices:
  - class: BaseDevice
    device_class_id: 7
    enabled: true
device_manager:
  type: base
action_executor:
  type: base_policy_runner
policy:
  class: OpenPiUmiPolicy
  params:
    pi_config_name: fixture_pi05
    ckpt_dir: {json.dumps(str(checkpoint))}
  infer_rollout:
    enabled: false
    required: false
""",
        encoding='utf-8',
    )
    return source


def _workflow_factory(tmp_path: Path) -> Any:
    repo_root = tmp_path / 'repo'
    (repo_root / 'consumer').mkdir(parents=True, exist_ok=True)
    (repo_root / 'consumer' / 'openpi_policy_infer.py').touch()
    inference_python = tmp_path / 'openpi-python'
    inference_python.touch()
    recording_python = tmp_path / 'recording-python'
    recording_python.touch()

    def factory(name: str, config: Any, **kwargs: Any) -> WorkflowLifecycle:
        kwargs.update(
            {
                'repo_root': repo_root,
                'device_python': Path(sys.executable),
                'inference_python': inference_python,
                'recording_python': recording_python,
                'registry_checker': lambda *args: (True, ''),
                'recording_profile_checker': lambda path: (True, ''),
            }
        )
        return WorkflowLifecycle(name, config, **kwargs)

    return factory


def _app(
    tmp_path: Path,
    root: Path,
    config_dir: Path,
    *,
    inference_preferences_path: Path | None = None,
) -> Any:
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(),
        workflow_factory=_workflow_factory(tmp_path),
        checkpoint_browse_roots=[root],
        inference_preferences_path=inference_preferences_path,
    )
    app.testing = True
    return app


def _context(state: dict[str, Any]) -> dict[str, Any]:
    checkpoint = state['checkpoint']
    return {
        'expected_selected_config': state['selected_config'],
        'expected_revision': state['revision'],
        'expected_generation': checkpoint['generation'],
        'expected_options_generation': state['inference_options']['generation'],
    }


def _options_context(state: dict[str, Any]) -> dict[str, Any]:
    options = state['inference_options']
    return {
        'expected_selected_config': state['selected_config'],
        'expected_revision': state['revision'],
        'expected_generation': options['generation'],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hydra_env_override_resolves_special_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    value = '/tmp/override space,[x]=${literal}'
    environment = dict(os.environ)
    environment['SUPERINFERENCE_CHECKPOINT_OVERRIDE'] = value
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / 'consumer' / 'openpi_policy_infer.py'),
            '--config-name',
            'policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi',
            '--cfg',
            'job',
            '--resolve',
                    'policy.params.ckpt_dir='
            '${oc.env:SUPERINFERENCE_CHECKPOINT_OVERRIDE}',
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert value in completed.stdout


def test_override_set_clear_switch_and_source_yaml_immutable(tmp_path: Path) -> None:
    root = tmp_path / 'allowed'
    default_a = _checkpoint(root, 'default-a')
    override = _checkpoint(root, 'override step=20 [safe]')
    default_b = _checkpoint(root, 'default-b')
    config_dir = tmp_path / 'configs'
    source_a = _write_config(config_dir, 'first', default_a)
    _write_config(config_dir, 'second', default_b)
    source_hash = _sha256(source_a)
    app = _app(tmp_path, root, config_dir)
    client = app.test_client()

    selected = client.post('/api/config/select', json={'name': 'first'})
    assert selected.status_code == 200
    initial = selected.get_json()
    initial_checkpoint = initial['checkpoint']
    assert initial_checkpoint['default_path'] == str(default_a)
    assert initial_checkpoint['effective_path'] == str(default_a)
    assert initial_checkpoint['override_path'] is None
    assert initial_checkpoint['valid'] is True

    applied = client.post(
        '/api/checkpoint/override',
        json={**_context(initial), 'path': str(override)},
    )
    assert applied.status_code == 200
    applied_state = applied.get_json()
    assert applied_state['checkpoint']['default_path'] == str(default_a)
    assert applied_state['checkpoint']['override_path'] == str(override)
    assert applied_state['checkpoint']['effective_path'] == str(override)
    assert applied_state['checkpoint']['generation'] > initial_checkpoint['generation']
    assert _sha256(source_a) == source_hash

    stale = client.post(
        '/api/checkpoint/override',
        json={**_context(initial), 'path': str(default_a)},
    )
    assert stale.status_code == 409

    cleared = client.delete(
        '/api/checkpoint/override',
        json=_context(applied_state),
    )
    assert cleared.status_code == 200
    cleared_state = cleared.get_json()
    assert cleared_state['checkpoint']['override_path'] is None
    assert cleared_state['checkpoint']['effective_path'] == str(default_a)

    reapplied = client.post(
        '/api/checkpoint/override',
        json={**_context(cleared_state), 'path': str(override)},
    ).get_json()
    switched = client.post('/api/config/select', json={'name': 'second'})
    assert switched.status_code == 200
    switched_state = switched.get_json()
    assert switched_state['checkpoint']['config_name'] == 'second'
    assert switched_state['checkpoint']['override_path'] is None
    assert switched_state['checkpoint']['effective_path'] == str(default_b)
    assert switched_state['checkpoint']['generation'] > reapplied['checkpoint']['generation']
    assert _sha256(source_a) == source_hash
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_rollout_settings_are_persisted_but_toggles_reset_on_config_switch(
    tmp_path: Path,
) -> None:
    root = tmp_path / 'allowed'
    default_a = _checkpoint(root, 'default-a')
    default_b = _checkpoint(root, 'default-b')
    output = root / 'datasets' / 'rollouts'
    output.mkdir(parents=True)
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', default_a)
    _write_config(config_dir, 'second', default_b)
    preferences = tmp_path / 'panel-data' / 'inference-options.json'
    app = _app(
        tmp_path,
        root,
        config_dir,
        inference_preferences_path=preferences,
    )
    client = app.test_client()
    initial = client.post('/api/config/select', json={'name': 'first'}).get_json()

    browse = client.get('/api/rollout/browse').get_json()
    assert browse['path'] == str(root.resolve())
    enabled = client.post(
        '/api/rollout/settings',
        json={
            **_options_context(initial),
            'enabled': True,
            'path': str(output),
        },
    )
    assert enabled.status_code == 200
    enabled_state = enabled.get_json()
    options = enabled_state['inference_options']
    assert options['rollout_enabled'] is True
    assert options['rollout_output_root'] == str(output.resolve())
    assert options['rollout_path_valid'] is True
    assert Path(options['rollout_effective_output_root']) == output.resolve()
    storage = options['rollout_storage']
    assert storage['enabled'] is True
    assert storage['status'] in {'healthy', 'warning', 'critical'}
    assert storage['path'] == str(output.resolve())
    assert storage['free_bytes'] > 0
    assert storage['estimated_bytes_per_second'] > 0
    assert storage['estimated_remaining_seconds'] >= 0

    stale = client.post(
        '/api/rollout/settings',
        json={**_options_context(initial), 'enabled': False},
    )
    assert stale.status_code == 409

    switched = client.post('/api/config/select', json={'name': 'second'})
    assert switched.status_code == 200
    switched_options = switched.get_json()['inference_options']
    assert switched_options['test_mode'] is False
    assert switched_options['rollout_enabled'] is False
    assert switched_options['rollout_output_root'] == str(output.resolve())
    assert switched_options['generation'] > options['generation']
    app.config['DEVICE_CONTROLLER'].shutdown()

    reloaded = _app(
        tmp_path,
        root,
        config_dir,
        inference_preferences_path=preferences,
    )
    reloaded_client = reloaded.test_client()
    reloaded_state = reloaded_client.post(
        '/api/config/select', json={'name': 'first'}
    ).get_json()
    reloaded_options = reloaded_state['inference_options']
    assert reloaded_options['rollout_enabled'] is False
    assert reloaded_options['rollout_output_root'] == str(output.resolve())
    remembered_browse = reloaded_client.get('/api/rollout/browse').get_json()
    assert remembered_browse['path'] == str(output.parent.resolve())
    reloaded.config['DEVICE_CONTROLLER'].shutdown()


def test_rollout_path_is_replayable_before_inference_without_enabling_recording(
    tmp_path: Path,
) -> None:
    root = tmp_path / 'allowed'
    default = _checkpoint(root, 'default')
    output = root / 'datasets' / 'rollouts'
    (output / 'episode_0001').mkdir(parents=True)
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', default)
    app = _app(
        tmp_path,
        root,
        config_dir,
        inference_preferences_path=tmp_path / 'panel-data' / 'inference-options.json',
    )
    client = app.test_client()
    try:
        initial = client.post('/api/config/select', json={'name': 'first'}).get_json()
        selected = client.post(
            '/api/rollout/settings',
            json={
                **_options_context(initial),
                'enabled': False,
                'path': str(output),
            },
        )
        assert selected.status_code == 200
        state = selected.get_json()
        assert state['inference_options']['rollout_enabled'] is False
        assert state['workflow']['inference']['running'] is False

        recordings = client.get('/api/recordings').get_json()
        assert recordings['root_state'] == 'ready'
        assert recordings['total'] == 1
        assert recordings['items'][0]['id'] == 'episode_0001'
    finally:
        app.config['DEVICE_CONTROLLER'].shutdown()


def test_rollout_browser_rejects_invalid_manual_paths(tmp_path: Path) -> None:
    root = tmp_path / 'allowed'
    default = _checkpoint(root, 'default')
    file_path = root / 'not-a-directory.txt'
    file_path.write_text('not a directory', encoding='utf-8')
    outside = tmp_path / 'outside'
    outside.mkdir()
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', default)
    app = _app(tmp_path, root, config_dir)
    client = app.test_client()
    client.post('/api/config/select', json={'name': 'first'})

    invalid_paths = (
        'relative/path',
        str(root / 'missing'),
        str(file_path),
        str(outside),
        str(root / '..' / 'outside'),
        '/' + ('x' * (server.MAX_CHECKPOINT_PATH_LENGTH + 1)),
    )
    for path in invalid_paths:
        response = client.get(
            '/api/rollout/browse', query_string={'path': path}
        )
        assert response.status_code == 422, path

    app.config['DEVICE_CONTROLLER'].shutdown()


def test_browser_is_root_bounded_sorted_and_response_limited(tmp_path: Path) -> None:
    root = tmp_path / 'allowed'
    default = _checkpoint(root, 'step10')
    step2 = _checkpoint(root, 'step2')
    invalid = root / 'step3-no-params'
    invalid.mkdir()
    artifact = root / 'weights.bin'
    artifact.write_bytes(b'weights')
    outside = tmp_path / 'outside'
    outside.mkdir()
    escape = root / 'escape'
    symlink_params = root / 'symlink-params'
    symlink_params.mkdir()
    outside_params = outside / 'params'
    outside_params.mkdir()
    try:
        escape.symlink_to(outside, target_is_directory=True)
        (symlink_params / 'params').symlink_to(
            outside_params, target_is_directory=True
        )
    except OSError:
        escape = None
        symlink_params = None
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', default)
    app = _app(tmp_path, root, config_dir)
    client = app.test_client()
    client.post('/api/config/select', json={'name': 'first'})

    response = client.get(
        '/api/checkpoints/browse', query_string={'path': str(root)}
    )
    assert response.status_code == 200
    payload = response.get_json()
    names = [item['name'] for item in payload['entries']]
    assert names.index(step2.name) < names.index(default.name)
    entries = {item['name']: item for item in payload['entries']}
    assert entries[step2.name]['selectable'] is True
    assert entries[invalid.name]['navigable'] is True
    assert entries[invalid.name]['selectable'] is False
    assert entries[artifact.name]['kind'] == 'file'
    assert entries[artifact.name]['selectable'] is False
    if symlink_params is not None:
        assert entries[symlink_params.name]['selectable'] is False
    assert payload['current_selectable'] is False

    traversal = client.get(
        '/api/checkpoints/browse',
        query_string={'path': str(root / '..' / 'outside')},
    )
    assert traversal.status_code == 422
    assert client.get(
        '/api/checkpoints/browse',
        query_string={'path': '/' + ('x' * (server.MAX_CHECKPOINT_PATH_LENGTH + 1))},
    ).status_code == 422
    assert client.get(
        '/api/checkpoints/browse', query_string={'path': str(artifact)}
    ).status_code == 422
    if escape is not None:
        assert client.get(
            '/api/checkpoints/browse', query_string={'path': str(escape)}
        ).status_code == 422

    state = client.get('/api/state').get_json()
    assert client.post(
        '/api/checkpoint/override',
        json={**_context(state), 'path': str(artifact)},
    ).status_code == 422
    assert client.post(
        '/api/checkpoint/override',
        json={**_context(state), 'path': str(invalid)},
    ).status_code == 422
    if symlink_params is not None:
        assert client.post(
            '/api/checkpoint/override',
            json={**_context(state), 'path': str(symlink_params)},
        ).status_code == 422

    for index in range(server.MAX_CHECKPOINT_BROWSE_ENTRIES + 5):
        (root / f'bulk-{index}').mkdir()
    limited = client.get(
        '/api/checkpoints/browse', query_string={'path': str(root)}
    ).get_json()
    assert limited['truncated'] is True
    assert len(limited['entries']) == server.MAX_CHECKPOINT_BROWSE_ENTRIES
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_state_reports_invalid_default_checkpoint_path(tmp_path: Path) -> None:
    root = tmp_path / 'allowed'
    invalid = root / 'default-without-params'
    invalid.mkdir(parents=True)
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', invalid)
    app = _app(tmp_path, root, config_dir)
    state = app.test_client().post(
        '/api/config/select', json={'name': 'first'}
    ).get_json()
    checkpoint = state['checkpoint']
    assert checkpoint['default_path'] == str(invalid.resolve())
    assert checkpoint['effective_path'] == str(invalid.resolve())
    assert checkpoint['valid'] is False
    assert 'params directory does not exist' in checkpoint['path_error']
    assert state['workflow']['inference']['can_start'] is False
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_runtime_active_preserved_but_inference_active_blocks_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / 'allowed'
    default = _checkpoint(root, 'default')
    override = _checkpoint(root, 'override')
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', default)
    app = _app(tmp_path, root, config_dir)
    client = app.test_client()
    client.post('/api/config/select', json={'name': 'first'})
    controller = app.config['DEVICE_CONTROLLER']
    workflow = controller.workflow
    assert workflow is not None
    manager = FakeRuntimeProcess()
    executor = FakeRuntimeProcess()
    workflow.device_manager_process = manager
    workflow.action_executor_process = executor
    workflow._runtime_desired = True
    workflow._manager_ready = True
    workflow._executor_ready = True
    assert controller.starter is not None
    controller.starter.get_status = lambda: {
        'running': False,
        'monitor_running': True,
        'total_devices': 1,
        'device_manager_running': False,
        'policy_runner_running': False,
        'devices': [
            {
                'key': 'BaseDevice:7',
                'class': 'BaseDevice',
                'device_class_id': 7,
                'enabled': True,
                'ready': True,
                'running': False,
                'status': 'stopped',
                'desired_state': 'stopped',
            }
        ],
    }

    current = client.get('/api/state').get_json()
    assert current['checkpoint']['can_edit'] is True
    applied = client.post(
        '/api/checkpoint/override',
        json={**_context(current), 'path': str(override)},
    )
    assert applied.status_code == 200
    assert workflow.device_manager_process is manager
    assert workflow.action_executor_process is executor

    workflow._inference_desired = True
    blocked_state = controller.state()
    assert blocked_state['checkpoint']['can_edit'] is False
    blocked = client.delete(
        '/api/checkpoint/override', json=_context(blocked_state)
    )
    assert blocked.status_code == 409
    workflow._inference_desired = False
    controller.shutdown()


def test_inference_start_requires_matching_checkpoint_context(tmp_path: Path) -> None:
    root = tmp_path / 'allowed'
    default = _checkpoint(root, 'default')
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', default)
    app = _app(tmp_path, root, config_dir)
    client = app.test_client()
    selected = client.post('/api/config/select', json={'name': 'first'}).get_json()

    assert client.post('/api/workflow/inference/start').status_code == 422
    stale = _context(selected)
    stale['expected_generation'] += 1
    assert client.post(
        '/api/workflow/inference/start', json=stale
    ).status_code == 409
    current = client.get('/api/state').get_json()
    blocked = client.post(
        '/api/workflow/inference/start', json=_context(current)
    )
    assert blocked.status_code == 409
    assert 'DeviceManager' in blocked.get_json()['error']
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_binding_workflow_rebuild_preserves_override(tmp_path: Path) -> None:
    root = tmp_path / 'allowed'
    default = _checkpoint(root, 'default')
    override = _checkpoint(root, 'override')
    config_dir = tmp_path / 'configs'
    _write_config(config_dir, 'first', default)
    app = _app(tmp_path, root, config_dir)
    client = app.test_client()
    state = client.post('/api/config/select', json={'name': 'first'}).get_json()
    applied = client.post(
        '/api/checkpoint/override',
        json={**_context(state), 'path': str(override)},
    ).get_json()
    controller = app.config['DEVICE_CONTROLLER']
    generation = applied['checkpoint']['generation']
    controller._effective_from_snapshot = lambda base, overrides, snapshot: (
        copy.deepcopy(base),
        {},
    )
    updated = controller.update_bindings(
        overrides={'BaseDevice:7': {'serial': 'opaque-test'}},
        discovery_revision=controller.hardware_discovery.snapshot.revision,
    )
    assert updated['checkpoint']['override_path'] == str(override)
    assert updated['checkpoint']['generation'] == generation
    assert controller.workflow is not None
    assert controller.workflow.checkpoint_state()['effective_path'] == str(override)
    controller.shutdown()
