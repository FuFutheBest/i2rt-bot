from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from incubator.control_panel.conversion_jobs import ConversionJobError, ConversionJobManager
from incubator.control_panel import server
from incubator.control_panel.recording_control import (
    RecordingControlPlane,
    write_conversion_source_receipt,
)


class FakePreflight:
    def __init__(self) -> None:
        self.activated = False
        self.failure = None

    def status(self):
        if self.failure:
            return {'available': False, 'can_probe': False, 'state': 'activation_probe_failed', 'blocker': self.failure}
        if self.activated:
            return {'available': True, 'can_probe': False, 'state': 'ready', 'blocker': ''}
        return {'available': False, 'can_probe': True, 'state': 'awaiting_first_finalized_episode', 'blocker': 'awaiting_first_finalized_episode'}

    def available(self):
        status = self.status()
        return status['available'], status['blocker']

    def invalidate(self):
        return None

    def record_probe_failure(self, job_id, error):
        self.failure = str(error)


def _source(
    tmp_path: Path,
    *,
    episode_id: str = 'episode-abc',
    data_format: str = 'NEDF3',
    result: str = 'success',
) -> tuple[Path, dict]:
    root = tmp_path / 'rollouts'
    episode = root / 'episode_0007'
    episode.mkdir(parents=True)
    (episode / 'metadata.json').write_text(
        json.dumps({'data_format': data_format}) + '\n', encoding='utf-8'
    )
    outcome_name = 'episode_incomplete.json' if result == 'incomplete' else 'episode_result.json'
    (episode / outcome_name).write_text(
        json.dumps({'episode_result': result, 'is_success': result == 'success'}) + '\n',
        encoding='utf-8',
    )
    (episode / 'dense_labels.jsonl').write_text('{"label":"accept"}\n', encoding='utf-8')
    (episode / 'data.mcap').write_bytes(b'mcap fixture')
    receipt = write_conversion_source_receipt(
        episode,
        episode_id=episode_id,
        episode_index=7,
        result=result,
    )
    return root, receipt


@pytest.mark.parametrize('result', ['success', 'failure', 'incomplete'])
def test_queue_accepts_every_finalized_episode_result(
    tmp_path: Path, result: str
) -> None:
    root, receipt = _source(tmp_path, result=result)
    manager = _manager(tmp_path)
    job = manager.enqueue(receipt, source_root=root)
    assert job['source']['result'] == result
    assert job['state'] == 'queued'


def test_nedf3_queue_recovers_canonical_wrist_mapping_when_config_is_missing(
    tmp_path: Path,
) -> None:
    root, receipt = _source(tmp_path, data_format='NEDF3')
    manager = _manager(tmp_path)
    job = manager.enqueue(
        receipt,
        source_root=root,
        run_config={'head_camera': None, 'left_camera': None, 'right_camera': None},
    )
    assert job['run_config']['head_camera'] is None
    assert job['run_config']['left_camera'] == 'camera_1'
    assert job['run_config']['right_camera'] == 'camera_0'


def _manager(tmp_path: Path, **kwargs) -> ConversionJobManager:
    python = tmp_path / 'python'
    python.write_text('#!/bin/sh\n', encoding='utf-8')
    python.chmod(0o755)
    script = tmp_path / 'convert.py'
    script.write_text('# fixture\n', encoding='utf-8')
    return ConversionJobManager(
        tmp_path / 'queue',
        converter_python=python,
        converter_script=script,
        auto_start=False,
        **kwargs,
    )


def test_default_queue_root_uses_portable_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / 'python'
    python.write_text('#!/bin/sh\n', encoding='utf-8')
    script = tmp_path / 'convert.py'
    script.write_text('# fixture\n', encoding='utf-8')
    queue_root = tmp_path / 'fresh-home' / 'conversion-jobs'
    monkeypatch.setenv('SUPERINFERENCE_CONVERSION_OUTPUT_ROOT', str(queue_root))

    manager = ConversionJobManager(
        converter_python=python,
        converter_script=script,
        profile_available=lambda: (False, 'fixture'),
        auto_start=False,
    )

    assert manager.queue_root == queue_root.resolve()
    assert manager.jobs_dir.is_dir()
    assert manager.staging_dir.is_dir()
    assert manager.published_dir.is_dir()


def test_converter_runtime_preserves_venv_python_entry(tmp_path: Path) -> None:
    target = tmp_path / "runtime" / "python3.10"
    target.parent.mkdir()
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    entry = tmp_path / "converter-venv" / "bin" / "python"
    entry.parent.mkdir(parents=True)
    try:
        entry.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    script = tmp_path / "convert.py"
    script.write_text("# fixture\n", encoding="utf-8")

    manager = ConversionJobManager(
        tmp_path / "queue",
        converter_python=entry,
        converter_script=script,
        profile_available=lambda: (False, "fixture"),
        auto_start=False,
    )

    assert manager.converter_python == entry.absolute()
    assert manager.converter_python != entry.resolve()


def test_unavailable_profile_durably_queues_and_is_idempotent(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(
        tmp_path,
        profile_available=lambda: (False, 'AV1 runtime unavailable'),
    )
    first = manager.enqueue(receipt, source_root=root, run_config={'config_name': 'dual-pi05'})
    second = manager.enqueue(receipt, source_root=root, run_config={'config_name': 'ignored'})
    assert first['job_id'] == second['job_id']
    assert first['state'] == 'queued'
    assert len(list(manager.jobs_dir.glob('*.json'))) == 1
    state = manager.state()
    assert state['configured'] is True
    assert state['available'] is False
    assert state['blockers'] == ['AV1 runtime unavailable']
    assert state['counts']['queued'] == 1


def test_source_must_be_contained_and_unchanged(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(tmp_path, profile_available=lambda: (False, 'blocked'))
    another_root = tmp_path / 'another-root'
    another_root.mkdir()
    with pytest.raises(ConversionJobError, match='outside'):
        manager.enqueue(receipt, source_root=another_root)
    (root / 'episode_0007' / 'data.mcap').write_bytes(b'changed')
    with pytest.raises(ConversionJobError, match='changed'):
        manager.enqueue(receipt, source_root=root)


def test_source_receipt_rejects_symlink_even_when_target_identity_matches(
    tmp_path: Path,
) -> None:
    root, receipt = _source(tmp_path)
    source = root / 'episode_0007' / 'data.mcap'
    original = source.stat()
    target = source.with_name('same-data.mcap')
    target.write_bytes(source.read_bytes())
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    source.unlink()
    try:
        source.symlink_to(target.name)
    except (NotImplementedError, OSError):
        pytest.skip('symlink creation is unavailable')

    manager = _manager(tmp_path, profile_available=lambda: (False, 'blocked'))
    with pytest.raises(ConversionJobError, match='symlink'):
        manager.enqueue(receipt, source_root=root)


def test_restart_recovers_running_job_as_retry(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(tmp_path, profile_available=lambda: (False, 'blocked'))
    job = manager.enqueue(receipt, source_root=root)
    path = manager.jobs_dir / f"{job['job_id']}.json"
    payload = json.loads(path.read_text())
    payload['state'] = 'running'
    path.write_text(json.dumps(payload), encoding='utf-8')
    recovered = _manager(tmp_path, profile_available=lambda: (False, 'blocked'))
    result = recovered.state()['jobs'][0]
    assert result['state'] == 'retry'
    assert 'interrupted' in result['error']


def test_discard_cancels_only_unstarted_jobs(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(tmp_path, profile_available=lambda: (False, 'blocked'))
    queued = manager.enqueue(receipt, source_root=root)
    assert manager.cancel_episode('episode-abc') == 1
    assert manager.state()['jobs'][0]['state'] == 'cancelled'
    assert manager.cancel_episode('episode-abc') == 0
    path = manager.jobs_dir / f"{queued['job_id']}.json"
    payload = json.loads(path.read_text())
    payload['state'] = 'ready'
    path.write_text(json.dumps(payload), encoding='utf-8')
    assert manager.cancel_episode('episode-abc') == 0


def test_fake_worker_uses_fixed_low_priority_command_and_publishes(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    commands = []

    def runner(command, **kwargs):
        commands.append((command, kwargs))
        job = json.loads(Path(command[-3]).read_text(encoding='utf-8'))
        staging = Path(command[-1]) / 'shard' / 'meta'
        staging.mkdir(parents=True)
        (staging / 'shard_manifest.json').write_text(
            json.dumps(
                {
                    'job_id': job['job_id'],
                    'profile_id': job['profile']['id'],
                    'source_fingerprint': job['source']['fingerprint'],
                    'merge_command': ['python', 'merge_shards.py'],
                }
            ),
            encoding='utf-8',
        )
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    manager = _manager(
        tmp_path,
        profile_available=lambda: (True, ''),
        process_runner=runner,
    )
    job = manager.enqueue(receipt, source_root=root)
    path = manager.jobs_dir / f"{job['job_id']}.json"
    manager._execute(path, job)
    result = json.loads(path.read_text())
    assert result['state'] == 'ready'
    assert result['attempts'] == 1
    command, kwargs = commands[0]
    assert command[:10] == [
        '/usr/bin/systemd-run', '--user', '--scope', '--quiet',
        '-p', 'CPUQuota=200%', '-p', 'MemoryMax=8G', '-p', 'TasksMax=512',
    ]
    assert command[10:16] == ['/usr/bin/ionice', '-c', '3', '/usr/bin/nice', '-n', '19']
    assert command[16:18] == [str(manager.converter_python), str(manager.converter_script)]
    assert command[18] == '--job-spec'
    assert command[20] == '--staging-dir'
    assert kwargs['start_new_session'] is True
    manifest = manager.published_dir / job['job_id'] / 'meta' / 'shard_manifest.json'
    assert manifest.is_file()
    assert manager.published_shard_for_episode(
        root / 'episode_0007'
    ) == manifest.parents[1].resolve()


def test_published_shard_lookup_fails_closed_on_job_or_manifest_mismatch(
    tmp_path: Path,
) -> None:
    root, receipt = _source(tmp_path)

    def runner(command, **kwargs):
        job = json.loads(Path(command[-3]).read_text(encoding='utf-8'))
        staging = Path(command[-1]) / 'shard' / 'meta'
        staging.mkdir(parents=True)
        (staging / 'shard_manifest.json').write_text(
            json.dumps(
                {
                    'job_id': job['job_id'],
                    'profile_id': job['profile']['id'],
                    'source_fingerprint': job['source']['fingerprint'],
                }
            ),
            encoding='utf-8',
        )
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    manager = _manager(
        tmp_path,
        profile_available=lambda: (True, ''),
        process_runner=runner,
    )
    job = manager.enqueue(receipt, source_root=root)
    job_path = manager.jobs_dir / f"{job['job_id']}.json"
    manager._execute(job_path, job)
    episode = root / 'episode_0007'
    shard = manager.published_dir / job['job_id']
    assert manager.published_shard_for_episode(episode) == shard.resolve()

    payload = json.loads(job_path.read_text(encoding='utf-8'))
    payload['identity']['profile_id'] = 'wrong-profile'
    job_path.write_text(json.dumps(payload), encoding='utf-8')
    assert manager.published_shard_for_episode(episode) is None

    payload['identity']['profile_id'] = manager.profile_id
    job_path.write_text(json.dumps(payload), encoding='utf-8')
    manifest_path = shard / 'meta' / 'shard_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['source_fingerprint'] = '0' * 64
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    assert manager.published_shard_for_episode(episode) is None


def test_busy_gate_preserves_queue_and_fake_failure_does_not_change_source(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(
        tmp_path,
        profile_available=lambda: (True, ''),
        busy=lambda: True,
        process_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('must not run')),
    )
    job = manager.enqueue(receipt, source_root=root)
    assert manager._eligible_job() is None
    assert (root / 'episode_0007' / 'data.mcap').is_file()
    path = manager.jobs_dir / f"{job['job_id']}.json"
    failed = _manager(
        tmp_path,
        profile_available=lambda: (True, ''),
        process_runner=lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout='', stderr='bad'),
    )
    failed._execute(path, json.loads(path.read_text()))
    result = json.loads(path.read_text())
    assert result['state'] == 'error'
    assert (root / 'episode_0007' / 'data.mcap').read_bytes() == b'mcap fixture'


def test_corrupt_spool_entry_does_not_starve_valid_job(tmp_path: Path) -> None:
    first_root, first_receipt = _source(tmp_path / 'first', episode_id='first')
    second_root, second_receipt = _source(tmp_path / 'second', episode_id='second')
    manager = _manager(tmp_path, profile_available=lambda: (True, ''))
    first = manager.enqueue(first_receipt, source_root=first_root)
    second = manager.enqueue(second_receipt, source_root=second_root)
    first_path = manager.jobs_dir / f"{first['job_id']}.json"
    first_path.write_text('{not-json', encoding='utf-8')
    os.utime(first_path, ns=(1, 1))

    selected = manager._eligible_job()

    assert selected is not None
    assert selected[1]['job_id'] == second['job_id']


def test_unsafe_staging_path_fails_job_instead_of_leaving_it_running(
    tmp_path: Path,
) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(
        tmp_path,
        profile_available=lambda: (True, ''),
        process_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('unsafe staging must fail before converter launch')
        ),
    )
    job = manager.enqueue(receipt, source_root=root)
    path = manager.jobs_dir / f"{job['job_id']}.json"
    staging = manager.staging_dir / f".tmp-{job['job_id']}"
    staging.write_text('not a directory', encoding='utf-8')

    manager._execute(path, job)

    failed = json.loads(path.read_text(encoding='utf-8'))
    assert failed['state'] == 'error'
    assert 'Unsafe conversion staging path' in failed['error']


def test_reconcile_recovers_episode_marker_after_panel_restart(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(tmp_path, profile_available=lambda: (False, 'blocked'))
    assert manager.reconcile_source_root(root, run_config={'config_name': 'dual'}) == 1
    assert manager.reconcile_source_root(root, run_config={'config_name': 'dual'}) == 0
    assert manager.state()['counts']['queued'] == 1


def test_recorder_writes_durable_episode_and_run_receipts(tmp_path: Path) -> None:
    control_dir = tmp_path / 'control'
    control = RecordingControlPlane(control_dir, 'run-1', owner_pid=os.getpid())
    root = tmp_path / 'rollouts'
    episode = root / 'episode_0001'
    episode.mkdir(parents=True)
    (episode / 'metadata.json').write_text('{"data_format":"NEDF3"}')
    (episode / 'episode_result.json').write_text('{"episode_result":"success"}')
    receipt = write_conversion_source_receipt(
        episode,
        episode_id='episode-one',
        episode_index=1,
        result='success',
        control=control,
        config_name='dual-pi05',
    )
    assert json.loads((episode / '.conversion_source.json').read_text())['fingerprint'] == receipt['fingerprint']
    assert json.loads((control_dir / 'finalized' / 'episode-one.json').read_text())['fingerprint'] == receipt['fingerprint']


def test_read_only_api_exposes_blocked_queue_and_retry_accepts_only_job_id(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    manager = _manager(tmp_path, profile_available=lambda: (False, 'AV1 unavailable'))
    job = manager.enqueue(receipt, source_root=root)
    app = server.create_app(config_dir=tmp_path, conversion_jobs=manager)
    client = app.test_client()
    response = client.get('/api/conversion-jobs')
    assert response.status_code == 200
    assert response.get_json()['counts']['queued'] == 1
    assert response.get_json()['available'] is False
    invalid = client.post('/api/conversion-jobs/not-a-hash/retry')
    assert invalid.status_code == 409
    assert 'Unsafe conversion job id' in invalid.get_json()['error']
    path = manager.jobs_dir / f"{job['job_id']}.json"
    payload = json.loads(path.read_text())
    payload['state'] = 'error'
    path.write_text(json.dumps(payload), encoding='utf-8')
    retried = client.post(f"/api/conversion-jobs/{job['job_id']}/retry")
    assert retried.status_code == 200
    assert retried.get_json()['job']['state'] == 'retry'


def test_server_background_conversion_does_not_wait_for_active_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        server.DeviceController,
        '_has_active_workflow_locked',
        lambda _self: True,
    )
    monkeypatch.setattr(
        server.DeviceController,
        '_has_active_devices_locked',
        lambda _self: True,
    )
    app = server.create_app(config_dir=tmp_path / 'configs')
    controller = app.config['DEVICE_CONTROLLER']
    try:
        assert controller.conversion_jobs._busy() is False
        assert controller.recording_exports._busy() is False
    finally:
        controller.shutdown()


def test_finalized_receipts_enqueue_once_independently_of_decision_ack(tmp_path: Path) -> None:
    enqueued = []
    cancelled = []
    finalized = []
    from incubator.control_panel.recording import RecordingCoordinator

    coordinator = RecordingCoordinator(
        tmp_path / 'runs',
        pid_validator=lambda candidate, ancestor: True,
        conversion_enqueue=lambda receipt: enqueued.append(dict(receipt)),
        conversion_cancel=lambda episode_id: cancelled.append(episode_id),
        finalized_enqueue=lambda receipt: finalized.append(dict(receipt)),
    )
    coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    finalized_dir = coordinator.control_dir / 'finalized'
    finalized_dir.mkdir(mode=0o700)
    receipt = {
        'schema_version': 1,
        'episode_id': 'episode-accepted',
        'result': 'success',
        'fingerprint': 'a' * 64,
    }
    coordinator._atomic_write_json(finalized_dir / 'episode-accepted.json', receipt)
    coordinator.state()
    coordinator.state()
    assert enqueued == [receipt]
    assert finalized == [receipt]
    rejected = {
        'schema_version': 1,
        'episode_id': 'episode-rejected',
        'result': 'failure',
        'fingerprint': 'b' * 64,
    }
    coordinator._atomic_write_json(finalized_dir / 'episode-rejected.json', rejected)
    coordinator.state()
    assert enqueued == [receipt, rejected]
    assert finalized == [receipt, rejected]
    assert cancelled == []


def test_replay_prefetch_failure_does_not_block_conversion(tmp_path: Path) -> None:
    enqueued = []
    from incubator.control_panel.recording import RecordingCoordinator

    def fail_prefetch(_receipt):
        raise RuntimeError('replay cache unavailable')

    coordinator = RecordingCoordinator(
        tmp_path / 'runs',
        pid_validator=lambda candidate, ancestor: True,
        conversion_enqueue=lambda receipt: enqueued.append(dict(receipt)),
        finalized_enqueue=fail_prefetch,
    )
    coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    receipt = {
        'schema_version': 1,
        'episode_id': 'episode-accepted',
        'result': 'success',
        'path': str(tmp_path / 'episode_0001'),
        'fingerprint': 'c' * 64,
    }
    finalized_dir = coordinator.control_dir / 'finalized'
    finalized_dir.mkdir(mode=0o700)
    coordinator._atomic_write_json(finalized_dir / 'episode-accepted.json', receipt)
    coordinator.state()
    assert enqueued == [receipt]


def test_all_finalized_results_trigger_replay_prefetch(tmp_path: Path) -> None:
    finalized = []
    from incubator.control_panel.recording import RecordingCoordinator

    coordinator = RecordingCoordinator(
        tmp_path / 'runs',
        pid_validator=lambda candidate, ancestor: True,
        finalized_enqueue=lambda receipt: finalized.append(dict(receipt)),
    )
    coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    finalized_dir = coordinator.control_dir / 'finalized'
    finalized_dir.mkdir(mode=0o700)
    receipts = []
    for index, result in enumerate(('success', 'failure', 'incomplete')):
        receipt = {
            'schema_version': 1,
            'episode_id': f'episode-{result}',
            'result': result,
            'path': str(tmp_path / f'episode_{result}'),
            'fingerprint': str(index + 1) * 64,
        }
        receipts.append(receipt)
        coordinator._atomic_write_json(finalized_dir / f'{index}-{result}.json', receipt)
    coordinator.state()
    assert finalized == receipts


def test_first_real_job_can_activate_profile_and_publish(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path, data_format='NEDF3')
    preflight = FakePreflight()

    def runner(command, **kwargs):
        job = json.loads(Path(command[-3]).read_text(encoding='utf-8'))
        assert job['profile']['activation_probe'] is True
        staging = Path(command[-1]) / 'shard' / 'meta'
        staging.mkdir(parents=True)
        (staging / 'shard_manifest.json').write_text(
            json.dumps(
                {
                    'job_id': job['job_id'],
                    'profile_id': job['profile']['id'],
                    'source_fingerprint': job['source']['fingerprint'],
                }
            ),
            encoding='utf-8',
        )
        preflight.activated = True
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    manager = _manager(tmp_path, profile_preflight=preflight, process_runner=runner)
    job = manager.enqueue(receipt, source_root=root)
    assert job['profile']['can_probe'] is True
    path = manager.jobs_dir / f"{job['job_id']}.json"
    manager._execute(path, job)
    completed = json.loads(path.read_text())
    assert completed['state'] == 'ready'
    assert completed['profile']['activation_probe'] is True
    assert manager.state()['profile']['state'] == 'ready'


def test_enqueue_and_state_reads_do_not_trigger_activation_probe(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    preflight = FakePreflight()
    process_calls = []

    def runner(*args, **kwargs):
        process_calls.append((args, kwargs))
        raise AssertionError('enqueue/state must not run the converter')

    manager = _manager(
        tmp_path,
        profile_preflight=preflight,
        process_runner=runner,
    )
    job = manager.enqueue(
        receipt,
        source_root=root,
        run_config={
            'config_name': 'dual-pi05',
            'head_camera': 'WebCameraOpenPi_2',
            'left_camera': 'WebCameraOpenPi_0',
            'right_camera': 'WebCameraOpenPi_4',
        },
    )
    first_state = manager.state()
    second_state = manager.state()

    assert job['state'] == 'queued'
    assert job['profile']['activation_probe'] is False
    assert job['profile']['can_probe'] is True
    assert first_state['counts']['queued'] == 1
    assert second_state['counts']['queued'] == 1
    assert preflight.activated is False
    assert process_calls == []


def test_failed_activation_probe_blocks_other_queued_jobs(tmp_path: Path) -> None:
    root, receipt = _source(
        tmp_path / 'first', episode_id='first', data_format='NEDF3'
    )
    preflight = FakePreflight()
    manager = _manager(
        tmp_path,
        profile_preflight=preflight,
        process_runner=lambda *args, **kwargs: SimpleNamespace(
            returncode=9, stdout='', stderr='decode failed'
        ),
    )
    job = manager.enqueue(receipt, source_root=root)
    path = manager.jobs_dir / f"{job['job_id']}.json"
    manager._execute(path, job)
    failed = json.loads(path.read_text())
    assert failed['state'] == 'error'
    assert failed['profile']['state'] == 'activation_probe_failed'

    second_root, second_receipt = _source(tmp_path / 'second', episode_id='second')
    second = manager.enqueue(second_receipt, source_root=second_root)
    assert second['state'] == 'queued'
    assert manager._eligible_job() is None


def test_close_fails_closed_until_active_converter_finishes(tmp_path: Path) -> None:
    root, receipt = _source(tmp_path)
    runner_entered = threading.Event()
    release_runner = threading.Event()

    def runner(*args, **kwargs):
        del args, kwargs
        runner_entered.set()
        assert release_runner.wait(timeout=5.0)
        return SimpleNamespace(returncode=7, stdout='', stderr='stopped fixture')

    manager = _manager(
        tmp_path,
        profile_available=lambda: (True, ''),
        process_runner=runner,
        poll_interval_s=0.05,
    )
    manager.enqueue(receipt, source_root=root)
    manager.start()
    try:
        assert runner_entered.wait(timeout=2.0)
        with pytest.raises(ConversionJobError, match='still active'):
            manager.close(timeout_s=0.01)
        assert manager._thread is not None
        assert manager._thread.is_alive()
        with pytest.raises(ConversionJobError, match='closed'):
            manager.start()
    finally:
        release_runner.set()
        manager.close(timeout_s=2.0)

    assert manager._thread is None
    assert manager.state()['counts']['error'] == 1
