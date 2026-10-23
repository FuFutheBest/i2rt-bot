"""Durable, low-impact conversion queue for finalized NEDF episodes.

The queue consumes recorder-authored source receipts. Browser requests never
provide a path, interpreter, script, or converter argument. Jobs are addressed
by the source/profile fingerprint, making enqueue and publication idempotent.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .converter_profile import ConverterProfilePreflight, default_profile
from .recording_contract import NEDF3_CAMERA_IDS


SCHEMA_VERSION = 1
VALID_STATES = frozenset({'queued', 'running', 'ready', 'error', 'retry', 'cancelled'})
DEFAULT_QUEUE_ROOT = (
    Path.home() / '.local' / 'share' / 'superinference-control-panel' / 'conversion_jobs'
)
DEFAULT_PROFILE = default_profile()
DEFAULT_CONVERTER_PYTHON = DEFAULT_PROFILE.python
DEFAULT_CONVERTER_SCRIPT = DEFAULT_PROFILE.wrapper_script
DEFAULT_PROFILE_ID = DEFAULT_PROFILE.profile_id
MAX_JOB_BYTES = 2_000_000


class ConversionJobError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def _fsync_directory(path: Path) -> None:
    if os.name != 'posix':
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ConversionJobError(f'Conversion spool is not a real directory: {path}')
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    encoded = _canonical_json(payload) + b'\n'
    if len(encoded) > MAX_JOB_BYTES:
        raise ConversionJobError('Conversion job exceeds size limit')
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'wb', closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Dict[str, Any]:
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_size <= 0 or info.st_size > MAX_JOB_BYTES:
        raise ConversionJobError(f'Unsafe conversion artifact: {path.name}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ConversionJobError(f'Conversion artifact is not an object: {path.name}')
    return value


def _contained_path(path: Path, root: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        root_resolved = root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ConversionJobError('Configured rollout root or finalized episode is missing') from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ConversionJobError('Finalized episode is outside the configured rollout root') from exc
    return resolved


def _safe_source(receipt: Mapping[str, Any], source_root: Path) -> Dict[str, Any]:
    if receipt.get('schema_version') != SCHEMA_VERSION:
        raise ConversionJobError('Unsupported conversion source receipt')
    result = receipt.get('result')
    if result not in {'success', 'failure', 'incomplete'}:
        raise ConversionJobError('Only finalized episodes may be converted')
    episode_id = receipt.get('episode_id')
    fingerprint = receipt.get('fingerprint')
    data_format = receipt.get('data_format')
    raw_path = receipt.get('path')
    if not isinstance(episode_id, str) or not episode_id:
        raise ConversionJobError('Source receipt has no episode_id')
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ConversionJobError('Source receipt has no valid fingerprint')
    if not isinstance(raw_path, str) or not raw_path:
        raise ConversionJobError('Source receipt has no path')
    if data_format != 'NEDF3':
        raise ConversionJobError('Source receipt has no supported data_format')
    source_path = _contained_path(Path(raw_path), source_root)
    marker = source_path / '.conversion_source.json'
    marker_payload = _read_json(marker)
    if (
        marker_payload.get('fingerprint') != fingerprint
        or marker_payload.get('episode_id') != episode_id
        or marker_payload.get('data_format') != data_format
        or marker_payload.get('result') != result
    ):
        raise ConversionJobError('Source marker does not match handoff receipt')
    source_files = receipt.get('files') if isinstance(receipt.get('files'), list) else []
    for item in source_files:
        if not isinstance(item, Mapping) or not isinstance(item.get('relpath'), str):
            raise ConversionJobError('Source receipt contains an invalid file identity')
        relative = Path(item['relpath'])
        if not relative.parts or relative.is_absolute() or '..' in relative.parts:
            raise ConversionJobError('Source receipt contains an unsafe file path')
        candidate = source_path
        candidate_info = None
        for index, part in enumerate(relative.parts):
            candidate = candidate / part
            candidate_info = candidate.lstat()
            if stat.S_ISLNK(candidate_info.st_mode):
                raise ConversionJobError('Source receipt references a symlink')
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(
                candidate_info.st_mode
            ):
                raise ConversionJobError('Source receipt path has a non-directory parent')
        resolved_candidate = candidate.resolve(strict=True)
        try:
            resolved_candidate.relative_to(source_path)
        except ValueError as exc:
            raise ConversionJobError('Source receipt file escapes the episode') from exc
        if candidate_info is None or not stat.S_ISREG(candidate_info.st_mode):
            raise ConversionJobError('Source receipt references an unsafe file')
        if (
            int(item.get('size', -1)) != candidate_info.st_size
            or int(item.get('mtime_ns', -1)) != candidate_info.st_mtime_ns
        ):
            raise ConversionJobError('Finalized source changed after receipt publication')
    return {
        'path': str(source_path),
        'root': str(source_root.resolve()),
        'episode_id': episode_id,
        'episode_index': receipt.get('episode_index'),
        'result': result,
        'data_format': data_format,
        'fingerprint': fingerprint,
        'metadata_sha256': receipt.get('metadata_sha256'),
        'episode_result_sha256': receipt.get('episode_result_sha256'),
        'episode_incomplete_sha256': receipt.get('episode_incomplete_sha256'),
        'dense_labels_sha256': receipt.get('dense_labels_sha256'),
        'files': source_files,
    }


class ConversionJobManager:
    """One-worker persistent conversion scheduler."""

    def __init__(
        self,
        queue_root: Optional[Path] = None,
        *,
        converter_python: Path = DEFAULT_CONVERTER_PYTHON,
        converter_script: Path = DEFAULT_CONVERTER_SCRIPT,
        profile_id: str = DEFAULT_PROFILE_ID,
        profile_available: Optional[Callable[[], tuple[bool, str]]] = None,
        profile_preflight: Optional[ConverterProfilePreflight] = None,
        busy: Callable[[], bool] = lambda: False,
        process_runner: Callable[..., Any] = subprocess.run,
        poll_interval_s: float = 1.0,
        retry_delay_s: float = 30.0,
        auto_start: bool = True,
    ) -> None:
        if queue_root is None:
            configured_root = os.environ.get('SUPERINFERENCE_CONVERSION_OUTPUT_ROOT')
            if isinstance(configured_root, str) and configured_root.strip():
                queue_root = Path(configured_root.strip())
            else:
                data_root = os.environ.get('SUPERINFERENCE_DATA_ROOT')
                queue_root = (
                    Path(data_root.strip()) / 'conversion_jobs'
                    if isinstance(data_root, str) and data_root.strip()
                    else DEFAULT_QUEUE_ROOT
                )
        self.queue_root = Path(queue_root).expanduser().resolve()
        self.jobs_dir = self.queue_root / 'jobs'
        self.staging_dir = self.queue_root / 'staging'
        self.published_dir = self.queue_root / 'published'
        for directory in (self.queue_root, self.jobs_dir, self.staging_dir, self.published_dir):
            _ensure_private_directory(directory)
        # Keep the venv entry path: resolving bin/python follows its symlink to
        # the host interpreter and silently drops the venv's site-packages.
        self.converter_python = Path(converter_python).expanduser().absolute()
        self.converter_script = Path(converter_script).expanduser().resolve()
        self.profile_id = str(profile_id)
        self._profile_preflight = profile_preflight
        if self._profile_preflight is None and profile_available is None:
            if (
                self.converter_python.resolve() == DEFAULT_PROFILE.python.resolve()
                and self.converter_script == DEFAULT_PROFILE.wrapper_script.resolve()
                and self.profile_id == DEFAULT_PROFILE.profile_id
            ):
                self._profile_preflight = ConverterProfilePreflight(DEFAULT_PROFILE)
        self._profile_available = profile_available or self._default_profile_available
        self._busy = busy
        self._process_runner = process_runner
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.retry_delay_s = max(0.0, float(retry_delay_s))
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._recover_interrupted_jobs()
        if auto_start:
            self.start()

    def _default_profile_available(self) -> tuple[bool, str]:
        if self._profile_preflight is None:
            return False, 'custom converter runtime has no trusted profile preflight'
        return self._profile_preflight.available()

    def _profile_status(self) -> Dict[str, Any]:
        if self._profile_preflight is not None:
            return self._profile_preflight.status()
        available, blocker = self._profile_available()
        return {
            'available': bool(available),
            'can_probe': False,
            'state': 'ready' if available else 'blocked',
            'blocker': blocker or '',
        }

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise ConversionJobError('Conversion job manager is closed')
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name='conversion-jobs', daemon=True)
            self._thread.start()

    def close(self, timeout_s: float = 3.0) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_s))
            if thread.is_alive():
                # Never report a successful shutdown while a converter can
                # still write the staging directory. The server then retains
                # its single-instance lock, so a replacement worker cannot
                # recover and execute the same durable job concurrently.
                raise ConversionJobError(
                    'Conversion worker is still active after shutdown timeout'
                )
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _job_path(self, job_id: str) -> Path:
        if len(job_id) != 64 or any(ch not in '0123456789abcdef' for ch in job_id):
            raise ConversionJobError('Unsafe conversion job id')
        return self.jobs_dir / f'{job_id}.json'

    def enqueue(
        self,
        receipt: Mapping[str, Any],
        *,
        source_root: Path,
        run_config: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        source = _safe_source(receipt, source_root)
        effective_run_config = dict(run_config or {})
        if source['data_format'] == 'NEDF3':
            # Camera IDs are part of the NEDF3 data contract. Recovery must
            # not depend on whether the current UI rollout toggle happens to
            # have injected converter fields into the selected config.
            effective_run_config['left_camera'] = (
                effective_run_config.get('left_camera')
                or NEDF3_CAMERA_IDS['left']
            )
            effective_run_config['right_camera'] = (
                effective_run_config.get('right_camera')
                or NEDF3_CAMERA_IDS['right']
            )
        identity = {'source_fingerprint': source['fingerprint'], 'profile_id': self.profile_id}
        job_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
        job_path = self._job_path(job_id)
        with self._lock:
            if job_path.exists():
                existing = _read_json(job_path)
                if existing.get('identity') != identity:
                    raise ConversionJobError('Conversion job id collision')
                return existing
            profile_status = self._profile_status()
            now = _utc_now()
            job = {
                'schema_version': SCHEMA_VERSION,
                'job_id': job_id,
                'identity': identity,
                'state': 'queued',
                'source': source,
                'profile': {
                    'id': self.profile_id,
                    'available': bool(profile_status['available']),
                    'can_probe': bool(profile_status.get('can_probe')),
                    'state': profile_status.get('state'),
                    'activation_probe': False,
                    'blocker': profile_status.get('blocker') or None,
                },
                'run_config': effective_run_config,
                'output': {
                    'shard_root': str(self.published_dir / job_id),
                    'manifest_path': str(self.published_dir / job_id / 'meta' / 'shard_manifest.json'),
                },
                'attempts': 0,
                'error': None,
                'created_at': now,
                'updated_at': now,
                'next_attempt_at': None,
            }
            _atomic_write_json(job_path, job)
        self._wake.set()
        return job

    def cancel_episode(self, episode_id: str) -> int:
        cancelled = 0
        with self._lock:
            for path in self.jobs_dir.glob('*.json'):
                job = _read_json(path)
                if job.get('source', {}).get('episode_id') != episode_id:
                    continue
                if job.get('state') not in {'queued', 'retry'}:
                    continue
                job['state'] = 'cancelled'
                job['updated_at'] = _utc_now()
                _atomic_write_json(path, job)
                cancelled += 1
        return cancelled

    def retry(self, job_id: str) -> Dict[str, Any]:
        path = self._job_path(job_id)
        with self._lock:
            job = _read_json(path)
            if job.get('state') not in {'error', 'retry'}:
                raise ConversionJobError('Only failed conversion jobs can be retried')
            job['state'] = 'retry'
            job['error'] = None
            job['next_attempt_at'] = time.time()
            job['updated_at'] = _utc_now()
            _atomic_write_json(path, job)
        self._wake.set()
        return job

    def reconcile_source_root(
        self,
        source_root: Path,
        *,
        run_config: Optional[Mapping[str, Any]] = None,
    ) -> int:
        root = Path(source_root).expanduser().resolve()
        if not root.is_dir():
            return 0
        added = 0
        for marker in sorted(root.glob('episode_*/.conversion_source.json')):
            try:
                receipt = _read_json(marker)
                before = len(list(self.jobs_dir.glob('*.json')))
                self.enqueue(receipt, source_root=root, run_config=run_config)
                added += int(len(list(self.jobs_dir.glob('*.json'))) > before)
            except (ConversionJobError, OSError, ValueError, json.JSONDecodeError):
                continue
        return added

    def published_shard_for_episode(self, episode_path: Path) -> Optional[Path]:
        """Return the immutable published shard for one trusted source episode."""
        try:
            source_path = Path(episode_path).expanduser().resolve(strict=True)
            receipt = _read_json(source_path / '.conversion_source.json')
            fingerprint = receipt.get('fingerprint')
            if (
                receipt.get('schema_version') != SCHEMA_VERSION
                or receipt.get('result') not in {'success', 'failure', 'incomplete'}
                or not isinstance(fingerprint, str)
                or len(fingerprint) != 64
            ):
                return None
            identity = {
                'source_fingerprint': fingerprint,
                'profile_id': self.profile_id,
            }
            job_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
            with self._lock:
                job = _read_json(self._job_path(job_id))
            if job.get('state') != 'ready' or job.get('identity') != identity:
                return None
            job_source = job.get('source')
            if not isinstance(job_source, Mapping):
                return None
            if Path(str(job_source.get('path') or '')).resolve(strict=True) != source_path:
                return None
            published_root = self.published_dir.resolve(strict=True)
            shard = (published_root / job_id).resolve(strict=True)
            shard.relative_to(published_root)
            manifest = _read_json(shard / 'meta' / 'shard_manifest.json')
            if (
                manifest.get('job_id') != job_id
                or manifest.get('profile_id') != self.profile_id
                or manifest.get('source_fingerprint') != fingerprint
            ):
                return None
            return shard
        except (
            ConversionJobError,
            FileNotFoundError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    def state(self, *, limit: int = 100) -> Dict[str, Any]:
        jobs = []
        with self._lock:
            for path in self._ordered_job_paths(newest_first=True):
                try:
                    jobs.append(_read_json(path))
                except (ConversionJobError, OSError, json.JSONDecodeError):
                    continue
                if len(jobs) >= max(1, min(int(limit), 1000)):
                    break
        counts = {state: 0 for state in sorted(VALID_STATES)}
        for job in jobs:
            if job.get('state') in counts:
                counts[job['state']] += 1
        profile_status = self._profile_status()
        available = bool(profile_status['available'])
        blocker = str(profile_status.get('blocker') or '')
        return {
            'schema_version': SCHEMA_VERSION,
            'configured': True,
            'available': bool(available),
            'blockers': [blocker] if blocker else [],
            'can_probe': bool(profile_status.get('can_probe')),
            'profile': {
                'id': self.profile_id,
                'available': available,
                'can_probe': bool(profile_status.get('can_probe')),
                'state': profile_status.get('state'),
                'blocker': blocker or None,
            },
            'counts': counts,
            'jobs': jobs,
        }

    def _recover_interrupted_jobs(self) -> None:
        for path in self.jobs_dir.glob('*.json'):
            try:
                job = _read_json(path)
            except (ConversionJobError, OSError, json.JSONDecodeError):
                continue
            if job.get('state') == 'running':
                job['state'] = 'retry'
                job['error'] = 'worker interrupted before durable publication'
                job['next_attempt_at'] = time.time()
                job['updated_at'] = _utc_now()
                _atomic_write_json(path, job)

    def _ordered_job_paths(self, *, newest_first: bool = False) -> list[Path]:
        ordered: list[tuple[int, str, Path]] = []
        for path in self.jobs_dir.glob('*.json'):
            try:
                ordered.append((path.stat().st_mtime_ns, path.name, path))
            except OSError:
                continue
        ordered.sort(reverse=newest_first)
        return [item[2] for item in ordered]

    def _eligible_job(self) -> Optional[tuple[Path, Dict[str, Any]]]:
        if self._busy():
            return None
        profile_status = self._profile_status()
        if not profile_status['available'] and not profile_status.get('can_probe'):
            return None
        activation_probe = bool(
            not profile_status['available'] and profile_status.get('can_probe')
        )
        now = time.time()
        for path in self._ordered_job_paths():
            try:
                job = _read_json(path)
            except (ConversionJobError, OSError, json.JSONDecodeError):
                # One damaged private spool entry must not starve every valid
                # conversion behind it. It remains on disk for operator audit.
                continue
            eligible_state = job.get('state') == 'queued' or (
                job.get('state') == 'retry'
                and float(job.get('next_attempt_at') or 0) <= now
            )
            if not eligible_state:
                continue
            if activation_probe and (
                not isinstance(job.get('source'), Mapping)
                or job['source'].get('data_format') != 'NEDF3'
            ):
                # Only the canonical NEDF3 path can activate a new converter
                # profile. Jobs from a different format remain queued and are handled
                # after a real NEDF3 probe publishes the activation receipt.
                continue
            return path, job
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                selected = self._eligible_job()
                if selected is not None:
                    self._execute(*selected)
                    continue
            except Exception:
                pass
            self._wake.wait(self.poll_interval_s)
            self._wake.clear()

    def _execute(self, path: Path, job: Dict[str, Any]) -> None:
        with self._lock:
            current = _read_json(path)
            if current.get('state') not in {'queued', 'retry'}:
                return
            profile_status = self._profile_status()
            activation_probe = bool(
                not profile_status['available'] and profile_status.get('can_probe')
            )
            if not profile_status['available'] and not activation_probe:
                return
            if activation_probe and (
                not isinstance(current.get('source'), Mapping)
                or current['source'].get('data_format') != 'NEDF3'
            ):
                return
            current['state'] = 'running'
            current['attempts'] = int(current.get('attempts') or 0) + 1
            current['updated_at'] = _utc_now()
            current['error'] = None
            current['profile'] = {
                'id': self.profile_id,
                'available': bool(profile_status['available']),
                'can_probe': bool(profile_status.get('can_probe')),
                'state': profile_status.get('state'),
                'activation_probe': activation_probe,
                'blocker': profile_status.get('blocker') or None,
            }
            _atomic_write_json(path, current)
            job = current

        job_id = str(job['job_id'])
        staging = self.staging_dir / f'.tmp-{job_id}'
        command: Sequence[str] = (
            '/usr/bin/systemd-run', '--user', '--scope', '--quiet',
            '-p', 'CPUQuota=200%', '-p', 'MemoryMax=8G', '-p', 'TasksMax=512',
            '/usr/bin/ionice', '-c', '3',
            '/usr/bin/nice', '-n', '19',
            str(self.converter_python), str(self.converter_script),
            '--job-spec', str(path.resolve()),
            '--staging-dir', str(staging.resolve()),
        )
        try:
            if staging.is_symlink() or (
                staging.exists() and not staging.is_dir()
            ):
                raise ConversionJobError('Unsafe conversion staging path')
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(mode=0o700)
            completed = self._process_runner(
                list(command),
                cwd=str(self.converter_script.parent.parent.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                start_new_session=True,
            )
            if int(completed.returncode) != 0:
                detail = (completed.stderr or completed.stdout or 'converter failed').strip()[-1000:]
                raise ConversionJobError(detail)
            if activation_probe:
                if self._profile_preflight is None:
                    raise ConversionJobError('activation probe has no trusted profile preflight')
                self._profile_preflight.invalidate()
                activated = self._profile_status()
                if not activated['available']:
                    raise ConversionJobError(
                        str(activated.get('blocker') or 'activation receipt was not published')
                    )
            shard = staging / 'shard'
            self._validate_real_tree(shard)
            manifest_path = shard / 'meta' / 'shard_manifest.json'
            manifest = _read_json(manifest_path)
            if (
                manifest.get('job_id') != job_id
                or manifest.get('profile_id') != self.profile_id
                or manifest.get('source_fingerprint') != job['source']['fingerprint']
            ):
                raise ConversionJobError('Converter shard manifest does not match the job')
            target = self.published_dir / job_id
            if target.exists():
                self._validate_real_tree(target)
                existing = _read_json(target / 'meta' / 'shard_manifest.json')
                if existing != manifest:
                    raise ConversionJobError('Published shard conflicts with this job')
                shutil.rmtree(staging, ignore_errors=True)
            else:
                os.replace(shard, target)
                _fsync_directory(self.published_dir)
                shutil.rmtree(staging, ignore_errors=True)
            job['state'] = 'ready'
            job['error'] = None
            job['next_attempt_at'] = None
            job['profile'] = {
                'id': self.profile_id,
                'available': True,
                'can_probe': False,
                'state': 'ready',
                'activation_probe': activation_probe,
                'blocker': None,
            }
        except Exception as exc:
            job['state'] = 'error'
            job['error'] = f'{type(exc).__name__}: {exc}'[:1200]
            job['next_attempt_at'] = None
            shutil.rmtree(staging, ignore_errors=True)
            if activation_probe and self._profile_preflight is not None:
                self._profile_preflight.record_probe_failure(job_id, job['error'])
                failed_status = self._profile_status()
                job['profile'] = {
                    'id': self.profile_id,
                    'available': False,
                    'can_probe': False,
                    'state': failed_status.get('state'),
                    'activation_probe': True,
                    'blocker': failed_status.get('blocker') or job['error'],
                }
        job['updated_at'] = _utc_now()
        with self._lock:
            _atomic_write_json(path, job)

    @staticmethod
    def _validate_real_tree(root: Path) -> None:
        if root.is_symlink() or not root.is_dir():
            raise ConversionJobError('Converter output is not a real directory')
        for current_root, directories, files in os.walk(root, followlinks=False):
            current = Path(current_root)
            for name in (*directories, *files):
                if (current / name).is_symlink():
                    raise ConversionJobError('Converter output contains a symlink')


__all__ = ['ConversionJobError', 'ConversionJobManager', 'DEFAULT_QUEUE_ROOT']
