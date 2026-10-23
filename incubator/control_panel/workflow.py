#!/usr/bin/env python3
"""Controlled DeviceManager, ActionExecutor, and inference lifecycle.

The browser never supplies an executable path or command line.  This module
owns the two approved Python profiles and enforces the process dependency order
used by the control panel:

    devices -> DeviceManager -> ActionExecutor -> policy inference

Shutdown always runs in the reverse order.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from utils.cpu_affinity import CpuAffinitySettings
from utils.runtime_profiles import RuntimeProfile, resolve_runtime_profile

from .inference_control import (
    DEFAULT_INFERENCE_CONTROL_RUN_ROOT,
    InferenceControlConflictError,
    InferenceControlCoordinator,
    InferenceControlError,
)
from .recording import (
    DEFAULT_RECORDING_RUN_ROOT,
    RecordingConflictError,
    RecordingCoordinator,
    RecordingError,
    default_recording_profile_check,
    empty_recording_state,
    normalize_recording_camera_mapping,
    recording_config_state,
    recording_timeout,
    resolve_recording_python,
    validate_recording_camera_mapping,
)
from .recording_paths import configured_recording_root
from .inference_options import (
    MOCK_INFERENCE_ENV,
    POLICY_OVERRIDES_ENV,
    build_control_panel_recording_config,
    mock_inference_config_error,
)
from .runtime_config import extract_camera_slots
from .runtime_paths import (
    DEVICE_PROFILE_NAME_ENV,
    INFERENCE_PROFILE_NAME_ENV,
    resolve_device_python,
    resolve_inference_python,
    resolve_python_profile_name,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESET_READINESS_EXEMPTION_SECONDS = 90.0


class WorkflowError(RuntimeError):
    """Base error for controlled runtime and inference operations."""


class WorkflowConflictError(WorkflowError):
    """The requested transition violates a workflow dependency."""


_SAFE_SHM_NAME = re.compile(r'^[A-Za-z0-9_.-]{1,200}$')


def _normalize_shm_name(name: Any) -> str:
    normalized = str(name).lstrip('/')
    if not _SAFE_SHM_NAME.fullmatch(normalized):
        raise ValueError(f'Invalid shared-memory name: {name!r}')
    return normalized


def _runtime_shm_identity(name: str) -> Optional[Tuple[int, int]]:
    """Return a POSIX SHM object's identity without attaching to it."""
    if os.name != 'posix':
        return None
    try:
        result = (Path('/dev/shm') / _normalize_shm_name(name)).stat()
    except (FileNotFoundError, OSError, PermissionError, ValueError):
        return None
    return int(result.st_dev), int(result.st_ino)


def _unlink_runtime_shm(name: str) -> None:
    """Unlink one normalized POSIX SHM name without attaching to its payload."""
    (Path('/dev/shm') / _normalize_shm_name(name)).unlink()


def _runtime_shm_handle_identity(handle: Any) -> Optional[Tuple[int, int]]:
    """Return the identity of the object opened by SharedMemory itself."""
    descriptor = getattr(handle, '_fd', None)
    if os.name != 'posix' or not isinstance(descriptor, int):
        return None
    try:
        result = os.fstat(descriptor)
    except OSError:
        return None
    return int(result.st_dev), int(result.st_ino)


class _OwnedRuntimeSharedMemory:
    """Small DataChannel-compatible wrapper with guarded unlink authority."""

    def __init__(
        self,
        raw_handle: Any,
        claim: Dict[str, Any],
        guard: '_ExclusiveSharedMemoryGuard',
        *,
        read_only: bool,
    ) -> None:
        self._shm = raw_handle
        self._claim = claim
        self._guard = guard
        self._read_only = bool(read_only)

    @property
    def buf(self) -> memoryview:
        buffer = self._shm.buf
        return buffer.toreadonly() if self._read_only else buffer

    @property
    def name(self) -> str:
        return str(self._shm.name)

    @property
    def size(self) -> int:
        return int(self._shm.size)

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self._claim.get('unlinked'):
            raise FileNotFoundError(self.name)
        expected = (int(self._claim['dev']), int(self._claim['ino']))
        if self._guard.identity(self._claim['name']) != expected:
            self._guard.detach_tracker(self._claim)
            return
        self._shm.unlink()
        self._claim['unlinked'] = True


class _ExclusiveSharedMemoryGuard:
    """Patch one spawned runtime worker to atomically claim its producer SHM."""

    def __init__(
        self,
        connection: Any,
        *,
        shm_factory: Optional[Callable[..., Any]] = None,
        identity: Callable[[str], Optional[Tuple[int, int]]] = _runtime_shm_identity,
        handle_identity: Callable[
            [Any], Optional[Tuple[int, int]]
        ] = _runtime_shm_handle_identity,
        tracker_unregister: Optional[Callable[[str], None]] = None,
    ) -> None:
        if shm_factory is None:
            from multiprocessing import shared_memory

            shm_factory = shared_memory.SharedMemory
        self._connection = connection
        self._shm_factory = shm_factory
        self.identity = identity
        self._handle_identity = handle_identity
        self._tracker_unregister = tracker_unregister
        self._claims: list[Dict[str, Any]] = []
        self._module: Any = None
        self._original_create: Any = None
        self._replacement: Any = None

    def install(self) -> None:
        # Install before importing manager: its create_summary/create_policy
        # helpers resolve this module global when they eventually create SHM.
        from utils import shm_utils

        self._module = shm_utils
        self._original_create = shm_utils.create_shared_memory
        self._replacement = self.create
        shm_utils.create_shared_memory = self._replacement

    def create(
        self,
        name: str,
        size: int,
        readonly_after_creation: bool = False,
    ) -> _OwnedRuntimeSharedMemory:
        normalized = _normalize_shm_name(name)
        try:
            raw_handle = self._shm_factory(
                create=True,
                size=int(size),
                name=normalized,
            )
        except FileExistsError:
            _send_worker_event(
                self._connection,
                'shm_conflict',
                name=normalized,
                error=f'Shared memory {normalized} already exists',
            )
            raise

        object_identity = self._handle_identity(raw_handle)
        current_identity = self.identity(normalized)
        if object_identity is None or current_identity != object_identity:
            provisional = {
                'name': normalized,
                'raw_handle': raw_handle,
                'unlinked': False,
                'tracker_detached': False,
            }
            self.detach_tracker(provisional)
            raw_handle.close()
            raise RuntimeError(
                f'Cannot verify shared-memory identity for {normalized}'
            )
        claim = {
            'name': normalized,
            'raw_handle': raw_handle,
            'dev': object_identity[0],
            'ino': object_identity[1],
            'unlinked': False,
            'tracker_detached': False,
        }
        self._claims.append(claim)
        _send_worker_event(
            self._connection,
            'shm_claimed',
            name=normalized,
            dev=object_identity[0],
            ino=object_identity[1],
        )
        return _OwnedRuntimeSharedMemory(
            raw_handle,
            claim,
            self,
            read_only=readonly_after_creation,
        )

    def detach_tracker(self, claim: Dict[str, Any]) -> None:
        if claim.get('unlinked') or claim.get('tracker_detached'):
            return
        tracked_name = getattr(claim['raw_handle'], '_name', None)
        if not tracked_name:
            return
        try:
            if self._tracker_unregister is not None:
                self._tracker_unregister(str(tracked_name))
            else:
                from multiprocessing import resource_tracker

                resource_tracker.unregister(tracked_name, 'shared_memory')
            claim['tracker_detached'] = True
        except (KeyError, OSError, RuntimeError, ValueError):
            pass

    def close(self) -> None:
        if (
            self._module is not None
            and self._module.create_shared_memory is self._replacement
        ):
            self._module.create_shared_memory = self._original_create
        while self._claims:
            claim = self._claims.pop()
            expected = (int(claim['dev']), int(claim['ino']))
            current = self.identity(claim['name'])
            if not claim.get('unlinked') and current == expected:
                try:
                    claim['raw_handle'].unlink()
                    claim['unlinked'] = True
                except (FileNotFoundError, OSError, PermissionError):
                    self.detach_tracker(claim)
            elif not claim.get('unlinked'):
                # Never unlink by name after replacement. Detach our tracker so
                # it cannot later unlink another producer that reused the name.
                self.detach_tracker(claim)
            try:
                claim['raw_handle'].close()
            except (BufferError, FileNotFoundError, OSError):
                pass


def _send_worker_event(connection: Any, event: str, **fields: Any) -> None:
    try:
        connection.send({'event': event, **fields})
    except (AttributeError, BrokenPipeError, EOFError, OSError):
        pass


def _manager_worker(config: Dict[str, Any], connection: Any) -> None:
    """Run the configured DeviceManager in the device Python environment."""
    guard = _ExclusiveSharedMemoryGuard(connection)
    try:
        guard.install()
        CpuAffinitySettings.from_runtime(config.get('runtime')).apply(
            'device_manager'
        )
        from manager import MANAGER_CLASSES

        section = config.get('device_manager') or {}
        manager_type = str(section.get('type', 'base'))
        manager_class = MANAGER_CLASSES.get(manager_type)
        if manager_class is None:
            raise ValueError(f'Unknown DeviceManager type: {manager_type}')

        params = copy.deepcopy(section.get('params') or {})
        # A panel-composed config has no Hydra output artifact.  Pass the exact
        # effective in-memory config used by the device workers instead.
        params.pop('config_path', None)
        params['config'] = config
        manager = manager_class(**params)
        _send_worker_event(connection, 'ready', role='device_manager')
        result = manager.start()
        if result is False:
            raise RuntimeError('DeviceManager start returned False')
    except BaseException as exc:
        _send_worker_event(
            connection,
            'failed',
            role='device_manager',
            error=f'{type(exc).__name__}: {exc}'[:500],
        )
        raise
    finally:
        guard.close()
        try:
            connection.close()
        except (AttributeError, OSError):
            pass


def _action_executor_worker(config: Dict[str, Any], connection: Any) -> None:
    """Run ActionExecutor and report ready only after control SHMs connect."""
    runner: Any = None
    guard = _ExclusiveSharedMemoryGuard(connection)
    try:
        guard.install()
        CpuAffinitySettings.from_runtime(config.get('runtime')).apply(
            'policy_runner'
        )
        from manager import MANAGER_CLASSES

        section = config.get('action_executor') or {}
        runner_type = str(section.get('type', 'base_policy_runner'))
        runner_class = MANAGER_CLASSES.get(runner_type)
        if runner_class is None:
            raise ValueError(f'Unknown ActionExecutor type: {runner_type}')

        valid_params = {
            'robot_control_shms',
            'execution_fps',
            'execution_clock',
            'sim_clock_poll_interval_s',
            'chunk_length',
            'chunk_manager',
            'enable_action_visualizer',
            'action_visualizer_config',
            'device_delays',
            'device_dimension_delays',
            'signal_files',
        }
        params = {
            key: copy.deepcopy(value)
            for key, value in section.items()
            if key in valid_params
        }
        params['config'] = config
        runner = runner_class(**params)
        if runner.connect() is not True:
            raise RuntimeError('ActionExecutor could not connect control SHMs')

        _send_worker_event(connection, 'ready', role='action_executor')
        if runner.run_connected() is False:
            raise RuntimeError('ActionExecutor execution loop returned False')
    except BaseException as exc:
        _send_worker_event(
            connection,
            'failed',
            role='action_executor',
            error=f'{type(exc).__name__}: {exc}'[:500],
        )
        raise
    finally:
        if runner is not None:
            try:
                runner.stop()
            except Exception:
                pass
        guard.close()
        try:
            connection.close()
        except (AttributeError, OSError):
            pass


def _process_alive(process: Any) -> bool:
    if process is None:
        return False
    try:
        return bool(process.is_alive())
    except (AssertionError, OSError, ValueError):
        return False


def _process_exitcode(process: Any) -> Optional[int]:
    if process is None:
        return None
    try:
        return process.exitcode
    except (AssertionError, OSError, ValueError):
        return None


def _default_registry_check(
    python_path: Path,
    repo_root: Path,
    openpi_source: Optional[str],
    config_name: str,
    checkpoint: Path,
    expected_action_dim: Optional[int],
) -> Tuple[bool, str]:
    """Check the selected OpenPI registry and checkpoint without loading model weights."""
    probe = (
        'import sys\n'
        'from pathlib import Path\n'
        'from utils.openpi_source import activate_openpi_source\n'
        'from utils.openpi_checkpoint import validate_openpi_checkpoint_contract\n'
        'source = sys.argv[2] or None\n'
        'activate_openpi_source(source, repo_root=Path(sys.argv[1]))\n'
        'expected_dim = int(sys.argv[5]) if sys.argv[5] else None\n'
        'validate_openpi_checkpoint_contract(\n'
        '    sys.argv[3], Path(sys.argv[4]), expected_action_dim=expected_dim\n'
        ')\n'
    )
    try:
        completed = subprocess.run(
            [
                str(python_path),
                '-c',
                probe,
                str(repo_root),
                openpi_source or '',
                config_name,
                str(checkpoint),
                str(expected_action_dim) if expected_action_dim is not None else '',
            ],
            cwd=str(repo_root),
            env={**os.environ, 'JAX_PLATFORMS': 'cpu'},
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f'OpenPI registry check failed: {type(exc).__name__}'
    if completed.returncode == 0:
        return True, ''
    output = completed.stderr.strip() or completed.stdout.strip()
    detail = output.splitlines()[-1] if output else f'unknown error (code {completed.returncode})'
    return False, f'OpenPI checkpoint contract failed: {detail}'


class WorkflowLifecycle:
    """Own the non-device runtime stages for one selected effective config."""

    workflow_kind = 'openpi'
    inference_display_name = 'OpenPI'
    checkpoint_owner_stage = 'inference'
    checkpoint_config_key = 'ckpt_dir'
    checkpoint_requires_params = True
    # Some model servers consume a concrete weight file while the historical
    # OpenPI workflow consumes a checkpoint directory with ``params/``.  Keep
    # directory-only behavior as the default and let those workflows opt in
    # explicitly.
    checkpoint_accepts_file = False

    def __init__(
        self,
        config_name: str,
        config: Mapping[str, Any],
        *,
        checkpoint_override: Optional[Path] = None,
        test_inference_mode: bool = False,
        recording_enabled_override: Optional[bool] = None,
        recording_output_override: Optional[Path] = None,
        repo_root: Path = REPO_ROOT,
        device_python: Optional[Path] = None,
        inference_python: Optional[Path] = None,
        recording_python: Optional[Path] = None,
        recording_run_root: Path = DEFAULT_RECORDING_RUN_ROOT,
        inference_control_run_root: Path = DEFAULT_INFERENCE_CONTROL_RUN_ROOT,
        mp_context: Any = None,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        registry_checker: Callable[
            [Path, Path, Optional[str], str, Path, Optional[int]], Tuple[bool, str]
        ] = _default_registry_check,
        recording_profile_checker: Callable[
            [Path], Tuple[bool, str]
        ] = default_recording_profile_check,
        recording_coordinator_factory: Optional[Callable[[Path], Any]] = None,
        inference_control_coordinator_factory: Optional[
            Callable[[Path], Any]
        ] = None,
        conversion_enqueue: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Any]] = None,
        conversion_cancel: Optional[Callable[[str], Any]] = None,
        recording_finalized_enqueue: Optional[
            Callable[[Mapping[str, Any]], Any]
        ] = None,
        path_exists: Callable[[Path], bool] = Path.exists,
        shm_identity: Callable[
            [str], Optional[Tuple[int, int]]
        ] = _runtime_shm_identity,
        shm_unlink: Callable[[str], None] = _unlink_runtime_shm,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.config_name = str(config_name)
        self.config = copy.deepcopy(dict(config))
        self.repo_root = repo_root.resolve()
        self._checkpoint_override = (
            Path(checkpoint_override).expanduser().resolve(strict=False)
            if checkpoint_override is not None
            else None
        )
        self._test_inference_mode = bool(test_inference_mode)
        self._recording_enabled_override = (
            bool(recording_enabled_override)
            if recording_enabled_override is not None
            else None
        )
        self._recording_output_override = (
            Path(recording_output_override).expanduser().resolve(strict=False)
            if recording_output_override is not None
            else None
        )
        self.device_python = resolve_device_python(
            device_python,
            config_name=self.config_name,
            repo_root=self.repo_root,
        )
        self.inference_python = resolve_inference_python(
            inference_python,
            config_name=self.config_name,
            repo_root=self.repo_root,
        )
        self._inference_runtime_profile: Optional[RuntimeProfile] = None
        if inference_python is None and not os.environ.get('SUPERINFERENCE_INFERENCE_PYTHON'):
            candidate_profile = resolve_runtime_profile(
                'policy',
                config_name=self.config_name,
                repo_root=self.repo_root,
            )
            if candidate_profile is not None:
                if not candidate_profile.verified:
                    raise WorkflowError(
                        f'Inference Python profile {candidate_profile.name!r} '
                        'is unverified or stale'
                    )
                if candidate_profile.python.absolute() == self.inference_python.absolute():
                    self._inference_runtime_profile = candidate_profile
        self.runtime_profile_name = resolve_python_profile_name(
            'device',
            self.device_python,
            environment_name=DEVICE_PROFILE_NAME_ENV,
        )
        self.inference_profile_name = resolve_python_profile_name(
            'openpi',
            self.inference_python,
            environment_name=INFERENCE_PROFILE_NAME_ENV,
        )
        self.recording_python = resolve_recording_python(
            self.config,
            explicit=recording_python,
            config_name=self.config_name,
            repo_root=self.repo_root,
        )
        self.recording_run_root = Path(recording_run_root)
        self.inference_control_run_root = Path(inference_control_run_root)
        self._mp_context = mp_context or mp.get_context('spawn')
        self._popen_factory = popen_factory
        self._registry_checker = registry_checker
        self._recording_profile_checker = recording_profile_checker
        self._recording_coordinator_factory = recording_coordinator_factory
        self._inference_control_coordinator_factory = (
            inference_control_coordinator_factory
        )
        self._conversion_enqueue = conversion_enqueue
        self._conversion_cancel = conversion_cancel
        self._recording_finalized_enqueue = recording_finalized_enqueue
        self._path_exists = path_exists
        self._shm_identity = shm_identity
        self._shm_unlink = shm_unlink
        self._log_callback = log_callback
        self._lock = threading.RLock()
        # Immutable-by-convention snapshot replaced atomically after every
        # successful state sample. observe() can copy it without waiting behind
        # recorder/process start and stop timeouts that hold the workflow lock.
        self._last_observation: Optional[Dict[str, Any]] = None

        self.device_manager_process: Any = None
        self.action_executor_process: Any = None
        self.inference_process: Any = None
        self._manager_connection: Any = None
        self._executor_connection: Any = None
        self._runtime_shm_claims: Dict[
            str, Dict[str, Tuple[int, int]]
        ] = {
            'device_manager': {},
            'action_executor': {},
        }
        self._manager_ready = False
        self._executor_ready = False
        self._runtime_desired = False
        self._runtime_device_keys: Optional[set[str]] = None
        self._inference_desired = False
        self._runtime_error: Optional[str] = None
        self._inference_error: Optional[str] = None
        self._runtime_started_at: Optional[float] = None
        self._inference_started_at: Optional[float] = None
        self._inference_exit_code: Optional[int] = None
        self._inference_new_session = False
        self._inference_ready = False
        self._safety_stop_pending = False
        self._safety_stop_reason: Optional[str] = None
        self._safety_stop_thread: Optional[threading.Thread] = None
        self.recording_coordinator: Optional[Any] = None
        self.inference_control_coordinator: Optional[Any] = None
        self._registry_cache: Dict[
            Tuple[str, str, str, str, Optional[int]], Tuple[bool, str]
        ] = {}
        self._recording_profile_cache: Optional[Tuple[bool, str]] = None

    def _log(self, level: str, message: str) -> None:
        if self._log_callback is not None:
            self._log_callback(level, message)

    @staticmethod
    def _close_connection(connection: Any) -> None:
        if connection is None:
            return
        try:
            connection.close()
        except (AttributeError, OSError):
            pass

    def _record_runtime_shm_claim(
        self,
        role: str,
        event: Mapping[str, Any],
    ) -> None:
        try:
            raw_name = event.get('name')
            if not isinstance(raw_name, str):
                raise ValueError('missing SHM name')
            name = _normalize_shm_name(raw_name)
            raw_dev = event.get('dev')
            raw_ino = event.get('ino')
            if isinstance(raw_dev, bool) or isinstance(raw_ino, bool):
                raise ValueError('boolean SHM identity')
            identity = (int(raw_dev), int(raw_ino))
            if identity[0] < 0 or identity[1] <= 0:
                raise ValueError('invalid SHM identity')
        except (TypeError, ValueError):
            self._log('WARNING', f'Ignored invalid {role} SHM ownership event')
            return
        self._runtime_shm_claims.setdefault(role, {})[name] = identity

    def _cleanup_force_killed_runtime_shm(
        self,
        role: str,
    ) -> Tuple[list[str], list[str], list[str]]:
        """Remove only names still backed by the killed worker's claimed inode."""
        removed: list[str] = []
        remaining: list[str] = []
        preserved: list[str] = []
        claims = self._runtime_shm_claims.setdefault(role, {})
        for name, expected in list(claims.items()):
            current = self._shm_identity(name)
            if current is None:
                claims.pop(name, None)
                continue
            if current != expected:
                # A different producer reused the deterministic name. Never
                # unlink that replacement merely because the old worker died.
                preserved.append(name)
                claims.pop(name, None)
                continue
            try:
                self._shm_unlink(name)
            except FileNotFoundError:
                claims.pop(name, None)
            except OSError:
                remaining.append(name)
            else:
                removed.append(name)
                claims.pop(name, None)
        return removed, remaining, preserved

    def _drain_connection(self, role: str, connection: Any) -> None:
        if connection is None:
            return
        while True:
            try:
                if not connection.poll():
                    break
                event = connection.recv()
            except (EOFError, OSError):
                break
            if not isinstance(event, dict):
                continue
            event_type = event.get('event')
            if event_type == 'shm_claimed':
                self._record_runtime_shm_claim(role, event)
            elif event_type == 'ready':
                if role == 'device_manager':
                    self._manager_ready = True
                else:
                    self._executor_ready = True
                self._log('INFO', f'{role} reported ready')
            elif event.get('event') == 'failed':
                error = str(event.get('error') or f'{role} failed')
                self._runtime_error = error
                self._log('ERROR', error)

    def _refresh_locked(self, *, defer_safety_stop: bool = False) -> None:
        # The background safety worker deliberately performs bounded process
        # and recorder teardown without holding ``self._lock``.  While that
        # work is pending, observation/control callers must not race it by
        # probing the recorder or draining process connections.
        if self._safety_stop_pending:
            return
        self._drain_connection('device_manager', self._manager_connection)
        self._drain_connection('action_executor', self._executor_connection)

        manager_alive = _process_alive(self.device_manager_process)
        executor_alive = _process_alive(self.action_executor_process)
        if self._runtime_desired and self.device_manager_process is not None and not manager_alive:
            code = _process_exitcode(self.device_manager_process)
            self._runtime_error = self._runtime_error or (
                f'DeviceManager exited before stop (code {code})'
            )
        if self._runtime_desired and self.action_executor_process is not None and not executor_alive:
            code = _process_exitcode(self.action_executor_process)
            self._runtime_error = self._runtime_error or (
                f'ActionExecutor exited before stop (code {code})'
            )

        if self.inference_process is not None:
            try:
                code = self.inference_process.poll()
            except (AttributeError, OSError):
                code = None
            if code is not None:
                self._inference_exit_code = int(code)
                self._inference_ready = False
                if self._inference_desired:
                    self._inference_error = (
                        f'{self.inference_display_name} inference exited before '
                        f'stop (code {code})'
                    )
                    self._log('ERROR', self._inference_error)
                self._inference_desired = False
            elif self._inference_desired and self._inference_ready:
                _, _, recording_required = recording_config_state(
                    self._recording_config()
                )
                coordinator = self.recording_coordinator
                if recording_required and coordinator is not None:
                    recording_state = coordinator.state()
                    if not coordinator.is_healthy(recording_state):
                        self._inference_error = (
                            str(recording_state.get('error'))
                            if recording_state.get('error')
                            else 'Required recorder lost readiness during inference'
                        )
                        self._log('ERROR', self._inference_error)
                        if defer_safety_stop:
                            self._schedule_safety_stop_locked(self._inference_error)
                        else:
                            self._stop_inference_locked(
                                request_recording_shutdown=False,
                                preserve_error=True,
                            )

    def _schedule_safety_stop_locked(self, reason: str) -> None:
        """Stop inference/runtime off the observation request thread.

        Device/recorder loss must be visible to the browser immediately. The
        bounded but potentially long recorder/process teardown therefore runs
        in one daemon thread while ``observe()`` returns a fail-closed
        ``stopping`` snapshot.
        """
        self._runtime_desired = False
        self._inference_desired = False
        self._runtime_error = str(reason)
        self._inference_error = str(reason)
        if self._safety_stop_pending:
            self._safety_stop_reason = str(reason)
            return
        self._safety_stop_pending = True
        self._safety_stop_reason = str(reason)

        def stop_worker() -> None:
            stop_error: Optional[str] = None
            try:
                # Never hold the workflow lock across recorder shutdown or
                # process wait/kill timeouts.  DeviceController holds its own
                # lock while asking about workflow activity, so monopolising
                # this lock here would also make a manual device Stop wait for
                # a potentially long recorder flush.
                self._stop_runtime_locked(stop_inference=True)
            except BaseException as exc:
                stop_error = f'{type(exc).__name__}: {exc}'
                self._log('ERROR', f'Asynchronous safety stop failed: {stop_error}')
            finally:
                with self._lock:
                    reason_text = self._safety_stop_reason or str(reason)
                    self._runtime_error = stop_error or reason_text
                    self._inference_error = stop_error or reason_text
                    self._safety_stop_pending = False
                    self._safety_stop_thread = None

        self._safety_stop_thread = threading.Thread(
            target=stop_worker,
            name='workflow-safety-stop',
            daemon=True,
        )
        self._safety_stop_thread.start()

    @staticmethod
    def _config_device_key(device: Mapping[str, Any]) -> Optional[str]:
        device_class = device.get('class')
        device_class_id = device.get('device_class_id')
        if not isinstance(device_class, str) or device_class_id is None:
            return None
        return f'{device_class}:{device_class_id}'

    def _optional_head_device_key(self) -> Optional[str]:
        head = extract_camera_slots(self.config).get('head', {})
        key = head.get('device_key')
        if not isinstance(key, str) or not key or head.get('required') is True:
            return None
        return key

    def _head_device_key(self) -> Optional[str]:
        key = extract_camera_slots(self.config).get('head', {}).get('device_key')
        return key if isinstance(key, str) and key else None

    def _optional_device_keys(self) -> set[str]:
        keys: set[str] = set()
        optional_head = self._optional_head_device_key()
        if optional_head:
            keys.add(optional_head)
        return keys

    def _required_device_keys(self) -> set[str]:
        optional = self._optional_device_keys()
        keys: set[str] = set()
        devices = self.config.get('devices')
        if not isinstance(devices, list):
            return keys
        for device in devices:
            if not isinstance(device, Mapping) or device.get('enabled', True) is False:
                continue
            key = self._config_device_key(device)
            if key and key not in optional:
                keys.add(key)
        return keys

    def required_device_keys(self) -> set[str]:
        """Return devices that must be ready before the runtime can start."""
        with self._lock:
            return set(self._required_device_keys())

    def _dependency_device_keys(self) -> set[str]:
        if self._runtime_device_keys is not None:
            return set(self._runtime_device_keys)
        return self._required_device_keys()

    def _not_ready_device_keys(
        self,
        devices: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        status_by_key = {
            str(device.get('key')): device
            for device in devices
            if isinstance(device, Mapping) and device.get('key')
        }
        return sorted(
            key
            for key in self._dependency_device_keys()
            if status_by_key.get(key, {}).get('ready') is not True
        )

    def _pending_reset_readiness_exemptions_locked(
        self,
        devices: Sequence[Mapping[str, Any]],
        not_ready: Sequence[str],
    ) -> set[str]:
        """Allow a healthy owned robot to stop publishing while Home is running.

        Rizon executes Home synchronously in its device process, so its passive
        health heartbeat can briefly become stale. Treating that expected gap as
        a lost runtime dependency tears down inference in the middle of reset.
        Explicit robot fault, E-stop, or non-operational state always keeps the
        original fail-closed behavior.
        """
        if self._safety_stop_pending or not not_ready:
            return set()

        pending: Any = None
        coordinator = self.inference_control_coordinator
        if coordinator is not None:
            try:
                control_state = coordinator.state()
            except InferenceControlError:
                control_state = {}
            candidate = control_state.get('pending_command')
            if isinstance(candidate, Mapping) and candidate.get('action') == 'reset':
                pending = candidate
            elif control_state.get('reset_pending') is True:
                candidate = control_state.get('last_command')
                if isinstance(candidate, Mapping) and candidate.get('action') == 'reset':
                    pending = dict(candidate)
                    pending['created_at'] = candidate.get('updated_at')
        if pending is None and self.recording_coordinator is not None:
            try:
                candidate = self.recording_coordinator.state().get('pending_command')
            except RecordingError:
                candidate = None
            if isinstance(candidate, Mapping) and candidate.get('action') in {
                'episode_accept',
                'episode_discard',
            }:
                pending = candidate
        if not isinstance(pending, Mapping):
            return set()
        created_at = pending.get('created_at')
        if not isinstance(created_at, str):
            return set()
        try:
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except ValueError:
            return set()
        if created.tzinfo is None:
            return set()
        age_seconds = (datetime.now(timezone.utc) - created).total_seconds()
        if age_seconds < 0 or age_seconds > RESET_READINESS_EXEMPTION_SECONDS:
            return set()

        status_by_key = {
            str(device.get('key')): device
            for device in devices
            if isinstance(device, Mapping) and device.get('key')
        }
        exempt: set[str] = set()
        for key in not_ready:
            device = status_by_key.get(key)
            if not isinstance(device, Mapping):
                continue
            if str(device.get('risk') or '').lower() != 'motion':
                continue
            if device.get('running') is not True and device.get('status') != 'running':
                continue
            safety = device.get('safety')
            safety = safety if isinstance(safety, Mapping) else {}
            estop = safety.get('estop')
            estop_active = (
                estop.get('active') is True
                if isinstance(estop, Mapping)
                else estop is True
            )
            if (
                safety.get('fault') is True
                or safety.get('operational') is False
                or estop_active
            ):
                continue
            exempt.add(key)
        return exempt

    def _runtime_process_config(
        self,
        devices: Sequence[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Any], set[str]]:
        """Build the exact config consumed by DeviceManager/ActionExecutor.

        A recording-only head camera joins the runtime only when already ready.
        Motion devices such as ``YamHeadRobot`` are always required when they
        are present in the selected config, even if OpenPI does not observe
        them directly.
        """
        process_config = copy.deepcopy(self.config)
        raw_devices = process_config.get('devices')
        configured = (
            [
                device
                for device in raw_devices
                if isinstance(device, Mapping)
                and device.get('enabled', True) is not False
            ]
            if isinstance(raw_devices, list)
            else []
        )
        required = self._required_device_keys()
        optional = self._optional_device_keys()
        head_device_key = self._head_device_key()
        ready_keys = {
            str(device.get('key'))
            for device in devices
            if isinstance(device, Mapping) and device.get('ready') is True
        }
        included = set(required)
        included.update(optional & ready_keys)

        filtered = [
            copy.deepcopy(dict(device))
            for device in configured
            if self._config_device_key(device) in included
        ]
        process_config['devices'] = filtered

        manager = process_config.get('device_manager')
        if isinstance(manager, dict) and filtered:
            original_master_key: Optional[str] = None
            raw_master = manager.get('master_device_class_id')
            if isinstance(raw_master, int) and 0 <= raw_master < len(configured):
                original_master_key = self._config_device_key(configured[raw_master])
            if (
                original_master_key not in included
                or original_master_key == head_device_key
            ):
                preferred = next(
                    (
                        self._config_device_key(device)
                        for device in filtered
                        if 'robot' in str(device.get('class') or '').casefold()
                    ),
                    None,
                )
                original_master_key = preferred or self._config_device_key(filtered[0])
            for index, device in enumerate(filtered):
                if self._config_device_key(device) == original_master_key:
                    manager['master_device_class_id'] = index
                    break
        return process_config, included

    def _runtime_blockers(
        self,
        devices: Sequence[Mapping[str, Any]],
        *,
        include_device_readiness: bool = True,
    ) -> list[str]:
        blockers: list[str] = []
        if not devices:
            blockers.append('当前配置没有可启动的设备')
        if include_device_readiness:
            not_ready = self._not_ready_device_keys(devices)
            if not_ready:
                blockers.append('设备未就绪：' + '、'.join(not_ready))

        manager = self.config.get('device_manager')
        manager_type = manager.get('type') if isinstance(manager, Mapping) else None
        if manager_type in (None, 'null', 'disabled'):
            blockers.append('当前配置未启用 DeviceManager')
        executor = self.config.get('action_executor')
        executor_type = executor.get('type') if isinstance(executor, Mapping) else None
        if executor_type in (None, 'null', 'disabled'):
            blockers.append('当前配置未启用 ActionExecutor')

        summary_name = self.config.get('summary_shm', 'device_summary_data')
        try:
            producer_names = [_normalize_shm_name(summary_name), 'policy_data']
        except ValueError:
            blockers.append('当前配置包含非法 summary_shm 名称')
            producer_names = ['policy_data']
        for name in producer_names:
            if self._shm_identity(name) is not None:
                blockers.append(f'共享内存已被占用：{name}')

        if not self._path_exists(self.device_python):
            blockers.append(f'{self.runtime_profile_name} Python profile 不可用')
        else:
            try:
                expected = self.device_python.resolve()
                current = Path(sys.executable).resolve()
            except OSError:
                expected = self.device_python
                current = Path(sys.executable)
            if expected != current:
                blockers.append(
                    f'控制服务未运行在 {self.runtime_profile_name} Python profile'
                )
        return blockers

    def runtime_preparation_blockers(
        self,
        devices: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        """Return blockers that cannot be fixed by starting/restarting devices."""
        with self._lock:
            if self._safety_stop_pending:
                return ['Asynchronous safety stop is in progress']
            self._refresh_locked()
            return self._runtime_blockers(
                devices,
                include_device_readiness=False,
            )

    def _default_checkpoint_path(self) -> Optional[Path]:
        policy = self.config.get('policy')
        if not isinstance(policy, Mapping):
            return None
        params = policy.get('params')
        if not isinstance(params, Mapping):
            return None
        checkpoint = params.get(self.checkpoint_config_key)
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            return None
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.repo_root / checkpoint_path
        return checkpoint_path.resolve(strict=False)

    def _effective_checkpoint_path(self) -> Optional[Path]:
        return self._checkpoint_override or self._default_checkpoint_path()

    def checkpoint_state(self) -> Dict[str, Any]:
        with self._lock:
            default_path = self._default_checkpoint_path()
            effective_path = self._effective_checkpoint_path()
            exists = bool(
                effective_path is not None and self._path_exists(effective_path)
            )
            params_path = (
                effective_path / 'params'
                if effective_path is not None and self.checkpoint_requires_params
                else None
            )
            if self.checkpoint_requires_params:
                params_exists = bool(
                    params_path is not None
                    and self._path_exists(params_path)
                    and not params_path.is_symlink()
                    and params_path.is_dir()
                    and os.access(params_path, os.R_OK | os.X_OK)
                )
            else:
                params_exists = None
            if effective_path is None:
                error = (
                    'Current config is missing '
                    f'policy.params.{self.checkpoint_config_key}'
                )
            elif not exists:
                error = f'Checkpoint does not exist: {effective_path}'
            elif self.checkpoint_requires_params and not params_exists:
                error = f'Checkpoint params directory does not exist: {params_path}'
            else:
                error = None
            return {
                'default_path': (
                    str(default_path) if default_path is not None else None
                ),
                'override_path': (
                    str(self._checkpoint_override)
                    if self._checkpoint_override is not None
                    else None
                ),
                'effective_path': (
                    str(effective_path) if effective_path is not None else None
                ),
                'override_active': self._checkpoint_override is not None,
                'exists': exists,
                'params_exists': params_exists,
                'params_required': self.checkpoint_requires_params,
                'owner_stage': self.checkpoint_owner_stage,
                'config_key': self.checkpoint_config_key,
                'valid': error is None,
                'error': error,
            }

    def _checkpoint_override_blockers_locked(self) -> list[str]:
        blockers: list[str] = []
        if self._safety_stop_pending:
            blockers.append('Inference safety stop is in progress')
        inference_alive = False
        if self.inference_process is not None:
            try:
                inference_alive = self.inference_process.poll() is None
            except (AttributeError, OSError):
                inference_alive = False
        if inference_alive or self._inference_desired:
            blockers.append('Stop inference before editing checkpoint')
        return blockers

    def checkpoint_override_blockers(self) -> list[str]:
        with self._lock:
            self._refresh_locked()
            return self._checkpoint_override_blockers_locked()

    def set_checkpoint_override(self, checkpoint: Optional[Path]) -> None:
        with self._lock:
            self._refresh_locked()
            blockers = self._checkpoint_override_blockers_locked()
            if blockers:
                raise WorkflowConflictError('; '.join(blockers))
            self._checkpoint_override = (
                Path(checkpoint).expanduser().resolve(strict=False)
                if checkpoint is not None
                else None
            )

    def _inference_option_blockers_locked(self) -> list[str]:
        blockers: list[str] = []
        if self._safety_stop_pending:
            blockers.append('Inference safety stop is in progress')
        inference_alive = False
        if self.inference_process is not None:
            try:
                inference_alive = self.inference_process.poll() is None
            except (AttributeError, OSError):
                inference_alive = False
        if inference_alive or self._inference_desired:
            blockers.append('Stop inference before editing inference options')
        return blockers

    def inference_option_blockers(self) -> list[str]:
        with self._lock:
            self._refresh_locked()
            return self._inference_option_blockers_locked()

    def set_inference_options(
        self,
        *,
        test_mode: bool,
        recording_enabled: bool,
        recording_output: Optional[Path],
    ) -> None:
        with self._lock:
            self._refresh_locked()
            blockers = self._inference_option_blockers_locked()
            if blockers:
                raise WorkflowConflictError('; '.join(blockers))
            if test_mode:
                error = mock_inference_config_error(self.config)
                if error:
                    raise WorkflowConflictError(error)
            if recording_enabled and recording_output is None:
                raise WorkflowConflictError(
                    'Select a rollout save directory before enabling recording'
                )
            self._test_inference_mode = bool(test_mode)
            self._recording_enabled_override = bool(recording_enabled)
            self._recording_output_override = (
                Path(recording_output).expanduser().resolve(strict=False)
                if recording_output is not None
                else None
            )

    def update_runtime_config(self, config: Mapping[str, Any]) -> None:
        with self._lock:
            blockers = self._inference_option_blockers_locked()
            if blockers:
                raise WorkflowConflictError('; '.join(blockers))
            self.config = copy.deepcopy(dict(config))

    def _recording_config(self) -> Mapping[str, Any]:
        if self._recording_enabled_override is None:
            return self.config
        return build_control_panel_recording_config(
            self.config,
            enabled=self._recording_enabled_override,
            output_root=self._recording_output_override,
        )

    def inference_options_state(self) -> Dict[str, Any]:
        with self._lock:
            mock_error = mock_inference_config_error(self.config)
            return {
                'test_mode': self._test_inference_mode,
                'mock_supported': mock_error is None,
                'mock_error': mock_error,
                'recording_enabled': bool(self._recording_enabled_override),
                'recording_output_root': (
                    str(self._recording_output_override)
                    if self._recording_output_override is not None
                    else None
                ),
                'blockers': self._inference_option_blockers_locked(),
            }

    def _policy_parameters(self) -> Tuple[Optional[str], Optional[Path], Optional[str]]:
        policy = self.config.get('policy')
        if not isinstance(policy, Mapping):
            return None, None, None
        params = policy.get('params')
        if not isinstance(params, Mapping):
            return None, None, None
        config_name = params.get('pi_config_name')
        source = policy.get('openpi_src')
        return (
            str(config_name) if isinstance(config_name, str) and config_name else None,
            self._effective_checkpoint_path(),
            str(source) if isinstance(source, str) and source else None,
        )

    def _policy_action_dimension(self) -> Optional[int]:
        policy = self.config.get('policy')
        if not isinstance(policy, Mapping):
            return None
        controls = policy.get('controls')
        if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)):
            return None
        total = 0
        for control in controls:
            if not isinstance(control, Mapping):
                return None
            shape = control.get('shape')
            if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape:
                return None
            size = 1
            for dimension in shape:
                if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                    return None
                size *= dimension
            total += size
        return total or None

    def _recording_profile_result(self) -> Tuple[bool, str]:
        if self.recording_python is None:
            return False, 'recording Python profile is not configured'
        if self._recording_profile_cache is None:
            self._recording_profile_cache = self._recording_profile_checker(
                self.recording_python
            )
        return self._recording_profile_cache

    def _recording_blockers(self) -> list[str]:
        recording_config = self._recording_config()
        configured, enabled, required = recording_config_state(recording_config)
        if not required:
            return []
        blockers: list[str] = []
        if not configured:
            blockers.append('当前配置未配置 policy.infer_rollout 录制')
        elif not enabled:
            blockers.append('当前配置未启用 policy.infer_rollout 录制')
        profile_ok, profile_error = self._recording_profile_result()
        if not profile_ok:
            blockers.append(profile_error or 'recording Python profile 不可用')
        return blockers

    def _recording_conversion_run_config(self) -> Dict[str, Any]:
        recording_config = self._recording_config()
        run_config = {
            'config_name': self.config_name,
            **normalize_recording_camera_mapping(recording_config),
        }
        optional_head = self._optional_head_device_key()
        if (
            optional_head
            and self._runtime_device_keys is not None
            and optional_head not in self._runtime_device_keys
        ):
            # The selected config can expose a head slot purely for recording.
            # If the user did not start it before runtime, preserve inference
            # and explicitly request the converter's black head placeholder.
            run_config['head_camera'] = None
        return run_config

    def _recording_output_root(self) -> Path:
        recording_config = self._recording_config()
        configured_root = configured_recording_root(
            recording_config,
            repo_root=self.repo_root,
        )
        if configured_root is None:
            configured_root = (self.repo_root / 'inference_rollout_data').resolve()
        return configured_root

    def _new_recording_coordinator(
        self,
        source_root: Optional[Path] = None,
    ) -> Any:
        configured, enabled, required = recording_config_state(
            self._recording_config()
        )
        if self._recording_coordinator_factory is not None:
            return self._recording_coordinator_factory(self.recording_run_root)
        source_root = source_root or self._recording_output_root()
        run_config = self._recording_conversion_run_config()

        def enqueue(receipt: Mapping[str, Any]) -> Any:
            if self._conversion_enqueue is None:
                return None
            if not run_config['left_camera'] or not run_config['right_camera']:
                self._log(
                    'ERROR',
                    'LeRobot conversion skipped: wrist camera mapping is missing',
                )
                return None
            return self._conversion_enqueue(
                receipt,
                source_root,
                run_config,
            )

        return RecordingCoordinator(
            self.recording_run_root,
            configured=configured,
            enabled=enabled,
            required=required,
            conversion_enqueue=enqueue,
            conversion_cancel=self._conversion_cancel,
            finalized_enqueue=self._recording_finalized_enqueue,
        )

    def _new_inference_control_coordinator(self) -> Any:
        if self._inference_control_coordinator_factory is not None:
            return self._inference_control_coordinator_factory(
                self.inference_control_run_root
            )
        return InferenceControlCoordinator(self.inference_control_run_root)

    @staticmethod
    def _inference_control_error(command: Any) -> Optional[str]:
        if not isinstance(command, Mapping):
            return None
        error = command.get('error')
        if isinstance(error, Mapping):
            code = str(error.get('code') or '').strip()
            message = str(error.get('message') or '').strip()
            return ': '.join(part for part in (code, message) if part) or None
        return str(error).strip() if error else None

    def _inference_reset_state_locked(
        self,
        *,
        inference_running: bool,
        recording_enabled: bool,
    ) -> Dict[str, Any]:
        coordinator = self.inference_control_coordinator
        control = (
            coordinator.state()
            if coordinator is not None
            else {
                'schema_version': 1,
                'run_id': None,
                'available': False,
                'paused': False,
                'reset_pending': False,
                'pending_command': None,
                'last_command': None,
            }
        )
        blockers: list[str] = []
        if self._safety_stop_pending:
            blockers.append(
                self._safety_stop_reason
                or 'Asynchronous safety stop is in progress'
            )
        if recording_enabled:
            blockers.append('Use the recording actions while rollout recording is enabled')
        if not inference_running:
            blockers.append('Start inference before resetting the robot')
        if control.get('available') is not True:
            blockers.append('Inference control is not ready')
        if control.get('pending_command'):
            blockers.append('An inference control command is already in progress')
        if control.get('reset_pending') is True:
            blockers.append('Inference reset is still moving the robot Home')
        last_command = control.get('last_command')
        paused = control.get('paused') is True
        return {
            **control,
            'can': not blockers,
            'can_pause': not blockers and not paused,
            'can_resume': not blockers and paused,
            'blockers': blockers,
            'error': self._inference_control_error(last_command),
        }

    def _recording_state_locked(self) -> Dict[str, Any]:
        configured, enabled, required = recording_config_state(
            self._recording_config()
        )
        coordinator = self.recording_coordinator
        if self._safety_stop_pending:
            # ``coordinator.state()`` may share I/O/locks with
            # ``request_shutdown``.  Publishing a conservative temporary
            # snapshot keeps every observation responsive while teardown is
            # underway and never implies that recording commands are safe.
            reason = self._safety_stop_reason or 'Asynchronous safety stop is in progress'
            state = empty_recording_state(
                configured=configured,
                enabled=enabled,
                required=required,
                error=reason,
                blockers=[reason],
            )
            state['status'] = 'stopping'
            state['active'] = coordinator is not None
            return state
        if coordinator is not None:
            state = coordinator.state()
            if self._inference_error and not state.get('error'):
                state['error'] = self._inference_error
                state['capabilities'] = {
                    action: {
                        'can': False,
                        'blockers': [self._inference_error],
                    }
                    for action in (
                        'episode_accept',
                        'episode_discard',
                        'segment_accept',
                        'segment_reject',
                    )
                }
            return state
        return empty_recording_state(
            configured=configured,
            enabled=enabled,
            required=required,
            blockers=self._recording_blockers() if required else None,
        )

    def _inference_script(self) -> Path:
        return self.repo_root / 'consumer' / 'openpi_policy_infer.py'

    def _inference_command(self) -> list[str]:
        return [
            str(self.inference_python),
            str(self._inference_script()),
            '--config-name',
            self.config_name,
        ]

    def _apply_inference_checkpoint_override(
        self,
        command: list[str],
        environment: Dict[str, str],
    ) -> None:
        if self._checkpoint_override is None:
            return
        environment['SUPERINFERENCE_CHECKPOINT_OVERRIDE'] = str(
            self._checkpoint_override
        )
        command.append(
            'policy.params.ckpt_dir=${oc.env:SUPERINFERENCE_CHECKPOINT_OVERRIDE}'
        )

    def _inference_blockers(self, runtime_ready: bool) -> list[str]:
        blockers: list[str] = []
        if not runtime_ready:
            blockers.append('请先正确启动 DeviceManager 与 ActionExecutor')
        if not self._path_exists(self.inference_python):
            blockers.append(f'{self.inference_profile_name} Python profile 不可用')
        script = self._inference_script()
        if not self._path_exists(script):
            blockers.append(
                f'{self.inference_display_name} inference 入口脚本不存在'
            )

        if self._test_inference_mode:
            mock_error = mock_inference_config_error(self.config)
            if mock_error:
                blockers.append(mock_error)
        else:
            pi_config_name, checkpoint, source = self._policy_parameters()
            if pi_config_name is None:
                blockers.append(
                    'Current config is missing policy.params.pi_config_name'
                )
            checkpoint_state = self.checkpoint_state()
            if checkpoint_state['error']:
                blockers.append(str(checkpoint_state['error']))

            if (
                pi_config_name is not None
                and checkpoint is not None
                and self._path_exists(self.inference_python)
            ):
                expected_action_dim = self._policy_action_dimension()
                cache_key = (
                    str(self.inference_python),
                    source or '',
                    pi_config_name,
                    str(checkpoint),
                    expected_action_dim,
                )
                checked = self._registry_cache.get(cache_key)
                if checked is None:
                    checked = self._registry_checker(
                        self.inference_python,
                        self.repo_root,
                        source,
                        pi_config_name,
                        checkpoint,
                        expected_action_dim,
                    )
                    self._registry_cache[cache_key] = checked
                if not checked[0]:
                    blockers.append(
                        checked[1]
                        or f'OpenPI registry 缺少 {pi_config_name}'
                    )
        blockers.extend(self._recording_blockers())
        return blockers

    def _spawn_runtime_worker(
        self,
        role: str,
        process_config: Mapping[str, Any],
    ) -> Tuple[Any, Any]:
        parent, child = self._mp_context.Pipe(duplex=False)
        target = _manager_worker if role == 'device_manager' else _action_executor_worker
        process = self._mp_context.Process(
            target=target,
            args=(copy.deepcopy(dict(process_config)), child),
            name='DeviceManager' if role == 'device_manager' else 'ActionExecutor',
        )
        try:
            process.start()
        finally:
            self._close_connection(child)
        return process, parent

    def start_runtime(self, devices: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._safety_stop_pending:
                raise WorkflowConflictError('Asynchronous safety stop is in progress')
            self._refresh_locked()
            current = self._state_locked(devices)
            if current['runtime']['ready']:
                return current
            blockers = self._runtime_blockers(devices)
            if blockers:
                raise WorkflowConflictError('; '.join(blockers))
            if current['runtime']['active']:
                raise WorkflowConflictError('运行时正在启动或处于异常状态，请先停止')

            process_config, runtime_device_keys = self._runtime_process_config(devices)
            self._runtime_desired = True
            self._runtime_device_keys = runtime_device_keys
            self._runtime_error = None
            self._manager_ready = False
            self._executor_ready = False
            self._runtime_shm_claims['device_manager'].clear()
            self._runtime_shm_claims['action_executor'].clear()
            try:
                (
                    self.device_manager_process,
                    self._manager_connection,
                ) = self._spawn_runtime_worker('device_manager', process_config)
                (
                    self.action_executor_process,
                    self._executor_connection,
                ) = self._spawn_runtime_worker('action_executor', process_config)
            except BaseException:
                self._stop_runtime_locked(stop_inference=True)
                raise
            self._runtime_started_at = time.time()
            self._log('INFO', 'Runtime start requested: DeviceManager -> ActionExecutor')
            self._refresh_locked()
            return self._state_locked(devices)

    @staticmethod
    def _stop_mp_process(process: Any, timeout: float = 10.0) -> bool:
        if process is None:
            return True
        if _process_alive(process):
            try:
                process.terminate()
            except (AttributeError, OSError, ValueError):
                pass
        try:
            process.join(timeout=timeout)
        except (AssertionError, AttributeError, OSError, ValueError):
            pass
        if _process_alive(process):
            try:
                process.kill()
            except (AttributeError, OSError, ValueError):
                pass
            try:
                process.join(timeout=2.0)
            except (AssertionError, AttributeError, OSError, ValueError):
                pass
        return not _process_alive(process)

    @staticmethod
    def _force_kill_mp_process(process: Any, timeout: float = 2.0) -> bool:
        if process is None:
            return True
        if _process_alive(process):
            try:
                process.kill()
            except (AttributeError, OSError, ValueError):
                try:
                    process.terminate()
                except (AttributeError, OSError, ValueError):
                    pass
        try:
            process.join(timeout=max(0.0, timeout))
        except (AssertionError, AttributeError, OSError, ValueError):
            pass
        return not _process_alive(process)

    def _signal_inference(self, sig: int) -> None:
        process = self.inference_process
        if process is None:
            return
        if self._inference_new_session and getattr(process, 'pid', None):
            try:
                os.killpg(os.getpgid(process.pid), sig)
                return
            except (OSError, ProcessLookupError):
                pass
        try:
            process.send_signal(sig)
        except (AttributeError, OSError, ValueError):
            pass

    def _stop_inference_locked(
        self,
        *,
        request_recording_shutdown: bool = True,
        preserve_error: bool = False,
    ) -> None:
        process = self.inference_process
        self._inference_desired = False
        if process is not None and request_recording_shutdown:
            coordinator = self.recording_coordinator
            if coordinator is not None:
                timeout = recording_timeout(
                    self._recording_config(), 'shutdown_timeout', 30.0
                )
                try:
                    shutdown_ok, shutdown_error = coordinator.request_shutdown(
                        timeout
                    )
                except (OSError, RecordingError) as exc:
                    shutdown_ok = False
                    shutdown_error = f'Recording shutdown control failed: {exc}'
                if shutdown_ok:
                    self._log('INFO', 'Recording flush/quarantine completed')
                else:
                    self._log(
                        'ERROR',
                        shutdown_error or 'Recording flush/quarantine failed',
                    )
        if process is not None:
            try:
                running = process.poll() is None
            except (AttributeError, OSError):
                running = False
            if running:
                self._signal_inference(signal.SIGINT)
                try:
                    process.wait(timeout=10.0)
                except (AttributeError, subprocess.TimeoutExpired):
                    self._signal_inference(signal.SIGTERM)
                    try:
                        process.wait(timeout=5.0)
                    except (AttributeError, subprocess.TimeoutExpired):
                        self._signal_inference(signal.SIGKILL)
                        try:
                            process.wait(timeout=2.0)
                        except (AttributeError, subprocess.TimeoutExpired):
                            pass
            try:
                code = process.poll()
            except (AttributeError, OSError):
                code = None
            if code is None:
                self._inference_error = (
                    f'{self.inference_display_name} inference did not exit after '
                    'SIGINT/SIGTERM/SIGKILL'
                )
                self._log('ERROR', self._inference_error)
                raise WorkflowConflictError(self._inference_error)
            self._inference_exit_code = code
        self.inference_process = None
        self.inference_control_coordinator = None
        self.recording_coordinator = None
        self._inference_new_session = False
        self._inference_ready = False
        self._inference_started_at = None
        if not preserve_error:
            self._inference_error = None
        self._log('INFO', f'{self.inference_display_name} inference stopped')

    def stop_inference(self, devices: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._safety_stop_pending:
                return self._state_locked(devices)
            self._stop_inference_locked()
            return self._state_locked(devices)

    def _stop_runtime_locked(self, *, stop_inference: bool) -> None:
        if stop_inference:
            self._stop_inference_locked()
        self._runtime_desired = False
        remaining_shm: list[str] = []
        removed_shm: list[str] = []
        preserved_shm: list[str] = []
        # Dependency order is deliberate: executor consumes manager/device SHM.
        had_executor = self.action_executor_process is not None
        self._drain_connection('action_executor', self._executor_connection)
        if not self._stop_mp_process(self.action_executor_process):
            self._runtime_error = 'ActionExecutor did not exit after terminate/kill'
            self._log('ERROR', self._runtime_error)
            raise WorkflowConflictError(self._runtime_error)
        self._drain_connection('action_executor', self._executor_connection)
        if had_executor:
            removed, remaining, preserved = self._cleanup_force_killed_runtime_shm(
                'action_executor'
            )
            removed_shm.extend(removed)
            remaining_shm.extend(remaining)
            preserved_shm.extend(preserved)
        self.action_executor_process = None
        self._close_connection(self._executor_connection)
        self._executor_connection = None
        self._executor_ready = False

        had_manager = self.device_manager_process is not None
        self._drain_connection('device_manager', self._manager_connection)
        if not self._stop_mp_process(self.device_manager_process):
            self._runtime_error = 'DeviceManager did not exit after terminate/kill'
            self._log('ERROR', self._runtime_error)
            raise WorkflowConflictError(self._runtime_error)
        self._drain_connection('device_manager', self._manager_connection)
        if had_manager:
            removed, remaining, preserved = self._cleanup_force_killed_runtime_shm(
                'device_manager'
            )
            removed_shm.extend(removed)
            remaining_shm.extend(remaining)
            preserved_shm.extend(preserved)
        self.device_manager_process = None
        self._close_connection(self._manager_connection)
        self._manager_connection = None
        self._manager_ready = False
        self._runtime_started_at = None
        self._runtime_device_keys = None
        if removed_shm:
            self._log(
                'INFO',
                'Removed stopped runtime SHM: ' + ', '.join(sorted(removed_shm)),
            )
        if preserved_shm:
            self._log(
                'WARNING',
                'Preserved replaced runtime SHM: '
                + ', '.join(sorted(preserved_shm)),
            )
        if remaining_shm:
            self._runtime_error = (
                'Runtime stop incomplete: '
                + ', '.join(f'SHM {name}' for name in sorted(remaining_shm))
            )
            self._log('ERROR', self._runtime_error)
            raise WorkflowConflictError(self._runtime_error)
        self._runtime_error = None
        self._log('INFO', 'Runtime stopped: ActionExecutor -> DeviceManager')

    def stop_runtime(self, devices: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._safety_stop_pending:
                return self._state_locked(devices)
            self._stop_runtime_locked(stop_inference=True)
            return self._state_locked(devices)

    def start_inference(self, devices: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._safety_stop_pending:
                raise WorkflowConflictError('Asynchronous safety stop is in progress')
            self._refresh_locked()
            current = self._state_locked(devices)
            if current['inference']['running']:
                return current
            blockers = self._inference_blockers(current['runtime']['ready'])
            if blockers:
                raise WorkflowConflictError('; '.join(blockers))
            inference_config = self._recording_config()
            configured, enabled, required = recording_config_state(inference_config)
            camera_mapping_errors = (
                validate_recording_camera_mapping(inference_config) if enabled else []
            )
            if camera_mapping_errors:
                raise WorkflowConflictError(
                    'Invalid recording camera mapping: '
                    + '; '.join(camera_mapping_errors)
                )

            coordinator: Optional[Any] = None
            recording_output_root: Optional[Path] = None
            recording_control_dir: Optional[Path] = None
            if enabled:
                recording_output_root = self._recording_output_root()
                coordinator = self._new_recording_coordinator(recording_output_root)
                try:
                    recording_control_dir = coordinator.prepare_run()
                except (OSError, RecordingError) as exc:
                    self._inference_error = f'Unable to prepare recording run: {exc}'
                    raise WorkflowError(self._inference_error) from exc
                self.recording_coordinator = coordinator
            elif required:
                raise WorkflowConflictError('Required inference recording is not enabled')

            inference_control = self._new_inference_control_coordinator()
            try:
                inference_control_dir = inference_control.prepare_run()
            except (OSError, InferenceControlError) as exc:
                self.recording_coordinator = None
                self._inference_error = (
                    f'Unable to prepare inference reset control: {exc}'
                )
                raise WorkflowError(self._inference_error) from exc
            self.inference_control_coordinator = inference_control

            command = self._inference_command()
            environment = (
                self._inference_runtime_profile.process_environment(
                    os.environ,
                    config_name=self.config_name,
                )
                if self._inference_runtime_profile is not None
                else dict(os.environ)
            )
            environment.update({
                'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                'SUPERINFERENCE_INFERENCE_CONTROL_RUN_ID': str(
                    inference_control.run_id
                ),
                'SUPERINFERENCE_INFERENCE_CONTROL_DIR': str(
                    inference_control_dir
                ),
            })
            policy_config = inference_config.get('policy')
            rollout_config = copy.deepcopy(
                policy_config.get('infer_rollout', {})
                if isinstance(policy_config, Mapping)
                else {}
            )
            if (
                coordinator is not None
                and recording_output_root is not None
                and recording_control_dir is not None
            ):
                if self.recording_python is None:
                    raise WorkflowError(
                        'recording Python profile is not configured'
                    )
                rollout_config['output_dir'] = str(recording_output_root)
                environment.update(
                    {
                        'SUPERINFERENCE_RUN_ID': str(coordinator.run_id),
                        'SUPERINFERENCE_RECORDING_CONTROL_DIR': str(
                            recording_control_dir
                        ),
                        'SUPERINFERENCE_RECORDER_PYTHON': str(self.recording_python),
                        'SUPERINFERENCE_CONFIG_NAME': self.config_name,
                        'SUPERINFERENCE_RECORDING_OUTPUT_DIR': str(
                            recording_output_root
                        ),
                    }
                )
            if (
                self._recording_enabled_override is not None
                or coordinator is not None
            ):
                environment[POLICY_OVERRIDES_ENV] = json.dumps(
                    {'infer_rollout': rollout_config},
                    separators=(',', ':'),
                )
            if self._test_inference_mode:
                environment[MOCK_INFERENCE_ENV] = '1'
            else:
                self._apply_inference_checkpoint_override(command, environment)
            try:
                self.inference_process = self._popen_factory(
                    command,
                    cwd=str(self.repo_root),
                    env=environment,
                    start_new_session=True,
                )
            except OSError as exc:
                self.inference_control_coordinator = None
                self.recording_coordinator = None
                self._inference_error = (
                    f'Unable to start {self.inference_display_name} '
                    f'inference: {exc}'
                )
                raise WorkflowError(self._inference_error) from exc
            try:
                inference_control.bind_owner(int(self.inference_process.pid))
                if coordinator is not None:
                    coordinator.bind_owner(int(self.inference_process.pid))
            except (
                AttributeError,
                InferenceControlError,
                RecordingError,
                TypeError,
                ValueError,
            ) as exc:
                self._inference_error = f'Unable to bind inference owner: {exc}'
                self._stop_inference_locked(
                    request_recording_shutdown=False,
                    preserve_error=True,
                )
                raise WorkflowError(self._inference_error) from exc
            self._inference_new_session = True
            self._inference_desired = True
            self._inference_ready = not required
            self._inference_error = None
            self._inference_exit_code = None
            self._inference_started_at = time.time()
            inference_mode = (
                'checkpoint-free test' if self._test_inference_mode else 'model'
            )
            self._log(
                'INFO',
                f'{inference_mode} inference start requested for '
                f'{self.config_name}',
            )

            if required and coordinator is not None:
                ready_timeout = recording_timeout(
                    inference_config, 'ready_timeout', 15.0
                )
                ready, error = coordinator.wait_until_ready(
                    self.inference_process, ready_timeout
                )
                if not ready:
                    self._inference_error = error or 'Required recorder failed to start'
                    self._log('ERROR', self._inference_error)
                    self._stop_inference_locked(
                        request_recording_shutdown=False,
                        preserve_error=True,
                    )
                    raise WorkflowError(self._inference_error)
                self._inference_ready = True
                self._log('INFO', 'Required recorder reported armed/recording ready')
            return self._state_locked(devices)

    def recording_action(
        self,
        action: str,
        *,
        episode_id: str,
        expected_checkpoint_revision: int,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._safety_stop_pending:
                raise RecordingConflictError('Asynchronous safety stop is in progress')
            self._refresh_locked()
            coordinator = self.recording_coordinator
            if coordinator is None or not self._inference_ready:
                raise RecordingConflictError('Inference recorder is not ready')
            command = coordinator.submit_action(
                action,
                episode_id=episode_id,
                expected_checkpoint_revision=expected_checkpoint_revision,
                idempotency_key=idempotency_key,
            )
            return {
                'ok': True,
                'command': command,
                'recording': coordinator.state(),
            }

    def inference_reset(
        self,
        *,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        """Queue Home/reset inside the already-loaded inference process."""
        with self._lock:
            if self._safety_stop_pending:
                raise WorkflowConflictError(
                    'Asynchronous safety stop is in progress'
                )
            self._refresh_locked()
            _, recording_enabled, _ = recording_config_state(
                self._recording_config()
            )
            if recording_enabled:
                raise WorkflowConflictError(
                    'Use the recording actions while rollout recording is enabled'
                )
            process = self.inference_process
            try:
                process_alive = process is not None and process.poll() is None
            except (AttributeError, OSError):
                process_alive = False
            if not process_alive or not self._inference_ready:
                raise WorkflowConflictError(
                    'Inference must be running and ready before reset'
                )
            coordinator = self.inference_control_coordinator
            if coordinator is None:
                raise WorkflowConflictError('Inference reset control is not ready')
            if coordinator.state().get('reset_pending') is True:
                raise WorkflowConflictError(
                    'Inference reset is still moving the robot Home'
                )
            try:
                command = coordinator.submit_reset(
                    idempotency_key=idempotency_key,
                )
            except InferenceControlConflictError as exc:
                raise WorkflowConflictError(str(exc)) from exc
            except InferenceControlError as exc:
                raise WorkflowError(str(exc)) from exc
            return {
                'ok': True,
                'command': command,
                'reset': coordinator.state(),
            }

    def inference_control(
        self,
        action: str,
        *,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        """Pause or resume predictions inside the loaded inference process."""
        if action not in {'pause', 'resume'}:
            raise WorkflowConflictError('Unsupported inference control action')
        with self._lock:
            if self._safety_stop_pending:
                raise WorkflowConflictError(
                    'Asynchronous safety stop is in progress'
                )
            self._refresh_locked()
            _, recording_enabled, _ = recording_config_state(
                self._recording_config()
            )
            if recording_enabled:
                raise WorkflowConflictError(
                    'Use the recording actions while rollout recording is enabled'
                )
            process = self.inference_process
            try:
                process_alive = process is not None and process.poll() is None
            except (AttributeError, OSError):
                process_alive = False
            if not process_alive or not self._inference_ready:
                raise WorkflowConflictError(
                    'Inference must be running and ready before controlling it'
                )
            coordinator = self.inference_control_coordinator
            if coordinator is None:
                raise WorkflowConflictError('Inference control is not ready')
            control = coordinator.state()
            if control.get('reset_pending') is True:
                raise WorkflowConflictError(
                    'Inference reset is still moving the robot Home'
                )
            paused = control.get('paused') is True
            if action == 'pause' and paused:
                raise WorkflowConflictError('Inference is already paused')
            if action == 'resume' and not paused:
                raise WorkflowConflictError('Inference is not paused')
            try:
                submit = (
                    coordinator.submit_pause
                    if action == 'pause'
                    else coordinator.submit_resume
                )
                command = submit(idempotency_key=idempotency_key)
            except InferenceControlConflictError as exc:
                raise WorkflowConflictError(str(exc)) from exc
            except InferenceControlError as exc:
                raise WorkflowError(str(exc)) from exc
            return {
                'ok': True,
                'command': command,
                'control': coordinator.state(),
            }

    def has_active_runtime_processes(self) -> bool:
        """Return whether runtime or inference work is still active.

        Workflow subclasses may own auxiliary model services that do not
        consume device bindings. Binding updates
        need to distinguish those services from DeviceManager,
        ActionExecutor, and inference.
        """
        with self._lock:
            if self._safety_stop_pending:
                return True
            self._refresh_locked()
            inference_running = False
            if self.inference_process is not None:
                try:
                    inference_running = self.inference_process.poll() is None
                except (AttributeError, OSError):
                    inference_running = False
            return (
                _process_alive(self.action_executor_process)
                or _process_alive(self.device_manager_process)
                or inference_running
            )

    def has_active_processes(self) -> bool:
        return self.has_active_runtime_processes()

    def stop_all(self) -> None:
        with self._lock:
            if self._safety_stop_pending:
                return
            self._stop_runtime_locked(stop_inference=True)

    def force_stop_all(self) -> Dict[str, Any]:
        """Immediately kill owned inference/runtime children without flushing."""
        with self._lock:
            self._runtime_desired = False
            self._inference_desired = False
            self._safety_stop_reason = 'Force-stopped by operator'
            killed_pids: list[int] = []
            remaining: list[str] = []
            removed_shm: list[str] = []
            remaining_shm: list[str] = []
            preserved_shm: list[str] = []

            inference = self.inference_process
            if inference is not None:
                try:
                    inference_pid = int(inference.pid)
                except (AttributeError, TypeError, ValueError):
                    inference_pid = None
                try:
                    running = inference.poll() is None
                except (AttributeError, OSError):
                    running = False
                if running:
                    self._signal_inference(signal.SIGKILL)
                    try:
                        inference.wait(timeout=2.0)
                    except (AttributeError, subprocess.TimeoutExpired):
                        pass
                try:
                    inference_running = inference.poll() is None
                except (AttributeError, OSError):
                    inference_running = False
                if inference_running:
                    remaining.append('inference')
                else:
                    if inference_pid is not None:
                        killed_pids.append(inference_pid)
                    try:
                        self._inference_exit_code = inference.poll()
                    except (AttributeError, OSError):
                        self._inference_exit_code = None
                    self.inference_process = None

            for role, attribute, connection_attribute in (
                (
                    'action_executor',
                    'action_executor_process',
                    '_executor_connection',
                ),
                (
                    'device_manager',
                    'device_manager_process',
                    '_manager_connection',
                ),
            ):
                process = getattr(self, attribute)
                connection = getattr(self, connection_attribute)
                self._drain_connection(role, connection)
                had_owned_process = process is not None
                try:
                    pid = int(process.pid) if process is not None else None
                except (AttributeError, TypeError, ValueError):
                    pid = None
                if not self._force_kill_mp_process(process):
                    remaining.append(role)
                    continue
                # SIGKILL bypasses the worker's finally/guard.close(). Drain
                # ownership events already buffered in its pipe, then perform
                # the same inode-checked cleanup from the surviving parent.
                self._drain_connection(role, connection)
                if had_owned_process:
                    removed, stale, preserved = (
                        self._cleanup_force_killed_runtime_shm(role)
                    )
                    removed_shm.extend(removed)
                    remaining_shm.extend(stale)
                    preserved_shm.extend(preserved)
                setattr(self, attribute, None)
                if pid is not None:
                    killed_pids.append(pid)

            self._close_connection(self._executor_connection)
            self._executor_connection = None
            self._close_connection(self._manager_connection)
            self._manager_connection = None
            self._manager_ready = False
            self._executor_ready = False
            self._runtime_device_keys = None
            self._runtime_started_at = None
            self._inference_started_at = None
            self._inference_new_session = False
            self._inference_ready = False
            self.recording_coordinator = None
            self.inference_control_coordinator = None
            self._runtime_error = (
                'Force stop incomplete: '
                + ', '.join(
                    [*remaining, *[f'SHM {name}' for name in remaining_shm]]
                )
                if remaining or remaining_shm
                else None
            )
            self._inference_error = self._runtime_error
            self._last_observation = None
            if removed_shm:
                self._log(
                    'WARNING',
                    'Removed force-killed runtime SHM: '
                    + ', '.join(sorted(set(removed_shm))),
                )
            if remaining_shm:
                self._log(
                    'ERROR',
                    'Force-killed runtime SHM cleanup failed: '
                    + ', '.join(sorted(set(remaining_shm))),
                )
            if preserved_shm:
                self._log(
                    'WARNING',
                    'Preserved replacement runtime SHM with a different inode: '
                    + ', '.join(sorted(set(preserved_shm))),
                )
            self._log(
                'WARNING',
                'Force-stopped inference/runtime; recording flush was bypassed',
            )
            return {
                'ok': not remaining and not remaining_shm,
                'killed_pids': sorted(set(killed_pids)),
                'remaining': remaining,
                'removed_shm': sorted(set(removed_shm)),
                'remaining_shm': sorted(set(remaining_shm)),
                'preserved_replacement_shm': sorted(set(preserved_shm)),
                'recording_flush_bypassed': True,
            }

    def _state_locked(
        self,
        devices: Sequence[Mapping[str, Any]],
        *,
        defer_safety_stop: bool = False,
    ) -> Dict[str, Any]:
        self._refresh_locked(defer_safety_stop=defer_safety_stop)
        manager_alive = _process_alive(self.device_manager_process)
        executor_alive = _process_alive(self.action_executor_process)
        not_ready = self._not_ready_device_keys(devices)
        reset_exempt = self._pending_reset_readiness_exemptions_locked(
            devices, not_ready
        )
        not_ready = [key for key in not_ready if key not in reset_exempt]
        runtime_was_ready = self._manager_ready and self._executor_ready
        runtime_process_failed = bool(
            self._runtime_desired
            and self._runtime_error
            and (not manager_alive or not executor_alive)
        )
        if self._runtime_desired and (
            bool(not_ready) or (runtime_was_ready and not manager_alive)
            or (runtime_was_ready and not executor_alive)
            or runtime_process_failed
        ):
            if not_ready:
                reason = 'Runtime dependency lost; devices not ready: ' + ', '.join(not_ready)
            else:
                reason = 'Runtime dependency process exited unexpectedly'
            if defer_safety_stop:
                self._schedule_safety_stop_locked(reason)
            else:
                self._stop_runtime_locked(stop_inference=True)
                self._runtime_error = reason
                manager_alive = False
                executor_alive = False
        runtime_ready = (
            self._runtime_desired
            and manager_alive
            and executor_alive
            and self._manager_ready
            and self._executor_ready
        )
        runtime_active = manager_alive or executor_alive or self._safety_stop_pending
        if self._safety_stop_pending:
            runtime_status = 'stopping'
        elif runtime_ready:
            runtime_status = 'running'
        elif runtime_active:
            runtime_status = 'starting' if self._runtime_error is None else 'failed'
        elif self._runtime_error:
            runtime_status = 'failed'
        else:
            runtime_status = 'stopped'

        inference_process_alive = False
        inference_pid = None
        if self.inference_process is not None:
            try:
                inference_process_alive = self.inference_process.poll() is None
            except (AttributeError, OSError):
                inference_process_alive = False
            inference_pid = (
                getattr(self.inference_process, 'pid', None)
                if inference_process_alive
                else None
            )
        _, recording_enabled, recording_required = recording_config_state(
            self._recording_config()
        )
        inference_running = bool(
            inference_process_alive
            and (self._inference_ready or not recording_required)
        )
        if self._safety_stop_pending:
            inference_status = 'stopping'
        elif inference_running:
            inference_status = 'running'
        elif inference_process_alive:
            inference_status = 'starting'
        elif self._inference_error:
            inference_status = 'failed'
        else:
            inference_status = 'stopped'

        if self._safety_stop_pending:
            pending_reason = (
                self._safety_stop_reason or 'Asynchronous safety stop is in progress'
            )
            runtime_blockers = [pending_reason]
            runtime_preparation_blockers = [pending_reason]
            inference_blockers = [pending_reason]
        else:
            runtime_blockers = self._runtime_blockers(devices)
            runtime_preparation_blockers = self._runtime_blockers(
                devices,
                include_device_readiness=False,
            )
            inference_blockers = self._inference_blockers(runtime_ready)
        recording_state = self._recording_state_locked()
        if self._safety_stop_pending:
            recording_profile_ok = bool(
                self._recording_profile_cache and self._recording_profile_cache[0]
            )
        else:
            recording_profile_ok, _ = self._recording_profile_result()
        snapshot = {
            'config_name': self.config_name,
            'kind': self.workflow_kind,
            'inference_label': self.inference_display_name,
            'checkpoint_owner_stage': self.checkpoint_owner_stage,
            'runtime': {
                'status': runtime_status,
                'desired_state': 'running' if self._runtime_desired else 'stopped',
                'active': runtime_active,
                'ready': runtime_ready,
                'can_start': (
                    not self._safety_stop_pending
                    and not runtime_active
                    and not runtime_preparation_blockers
                ),
                'can_stop': runtime_active or self._safety_stop_pending,
                'blockers': runtime_blockers,
                # Keep startup blockers separate from transient device readiness.
                # The Control Panel may prepare stopped devices during runtime start.
                'preparation_blockers': runtime_preparation_blockers,
                'readiness_blockers': [
                    blocker
                    for blocker in runtime_blockers
                    if blocker not in runtime_preparation_blockers
                ],
                'error': self._runtime_error,
                'reason': self._runtime_error,
                'started_at': self._runtime_started_at,
                'device_keys': sorted(self._runtime_device_keys or ()),
                'device_manager': {
                    'running': manager_alive,
                    'ready': self._manager_ready,
                    'pid': getattr(self.device_manager_process, 'pid', None) if manager_alive else None,
                },
                'action_executor': {
                    'running': executor_alive,
                    'ready': self._executor_ready,
                    'pid': getattr(self.action_executor_process, 'pid', None) if executor_alive else None,
                },
            },
            'inference': {
                'mode': 'test' if self._test_inference_mode else 'model',
                'test_mode': self._test_inference_mode,
                'status': inference_status,
                'desired_state': 'running' if self._inference_desired else 'stopped',
                'running': inference_running,
                'can_start': (
                    runtime_ready
                    and not inference_process_alive
                    and not inference_blockers
                ),
                'can_stop': inference_process_alive or self._safety_stop_pending,
                'blockers': inference_blockers,
                'error': self._inference_error,
                'reason': self._inference_error,
                'pid': inference_pid,
                'started_at': self._inference_started_at,
                'exit_code': self._inference_exit_code,
                'checkpoint': self.checkpoint_state(),
                'reset': self._inference_reset_state_locked(
                    inference_running=inference_running,
                    recording_enabled=recording_enabled,
                ),
            },
            'inference_options': self.inference_options_state(),
            'profiles': {
                'runtime': {
                    'name': self.runtime_profile_name,
                    'available': self._path_exists(self.device_python),
                },
                'inference': {
                    'name': self.inference_profile_name,
                    'available': self._path_exists(self.inference_python),
                },
                'recording': {
                    'name': 'nedf-recording',
                    'available': recording_profile_ok,
                },
            },
            'recording': recording_state,
            'stop_order': [
                'recording',
                'inference',
                'action_executor',
                'device_manager',
                'devices',
            ],
        }
        self._last_observation = copy.deepcopy(snapshot)
        return snapshot

    def state(self, devices: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            return self._state_locked(devices)

    def observe(self, devices: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Return an observation without waiting for dependency-loss teardown.

        Explicit lifecycle operations retain their synchronous behavior through
        ``state()``.  The shared UI sampler uses this path so an external
        emergency stop, device fault, disconnect, or recorder failure is
        visible immediately while the ordered safety stop continues in the
        background.
        """
        if not self._lock.acquire(blocking=False):
            cached = self._last_observation
            if cached is not None:
                return copy.deepcopy(cached)
            # Normal controller construction samples state before any action,
            # but direct/test callers may race the very first transition. In
            # that exceptional case retain the old blocking fallback rather
            # than returning a partial schema.
            with self._lock:
                return self._state_locked(devices, defer_safety_stop=True)
        try:
            return self._state_locked(devices, defer_safety_stop=True)
        finally:
            self._lock.release()


__all__ = [
    'WorkflowConflictError',
    'WorkflowError',
    'WorkflowLifecycle',
]
