"""FastWAM workflow lifecycle with an independently owned model server."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib import request as urllib_request

from utils.runtime_profiles import RuntimeProfile, RuntimeProfileError, resolve_runtime_profile

from .inference_options import mock_inference_config_error
from .workflow import (
    WorkflowConflictError,
    WorkflowError,
    WorkflowLifecycle,
)


DEFAULT_SERVER_HOST = '127.0.0.1'
DEFAULT_SERVER_PORT = 18090
DEFAULT_SERVER_PORT_RANGE = (18090, 18100)
SERVER_READY_PROBE_TIMEOUT_S = 0.35
FASTWAM_SERVER_PYTHON_ENV = 'SUPERINFERENCE_FASTWAM_SERVER_PYTHON'
FASTWAM_SERVER_PYTHON_EXTERNAL_ENV = 'SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL'
RELEASE_ROOT_ENV = 'SUPERINFERENCE_RELEASE_ROOT'
FASTWAM_SERVER_PORT_ENV = 'SUPERINFERENCE_FASTWAM_SERVER_PORT'
FASTWAM_READY_FILE_ENV = 'SUPERINFERENCE_FASTWAM_READY_FILE'
FASTWAM_PROGRESS_FILE_ENV = 'SUPERINFERENCE_FASTWAM_PROGRESS_FILE'
FASTWAM_RUN_ID_ENV = 'SUPERINFERENCE_FASTWAM_RUN_ID'
FASTWAM_CONFIG_NAME_ENV = 'SUPERINFERENCE_FASTWAM_CONFIG_NAME'
FASTWAM_SERVER_PROGRESS_PHASES = frozenset(
    {
        'spawning_process',
        'validating_config',
        'resolving_checkpoint',
        'validating_contract',
        'building_model',
        'loading_checkpoint',
        'loading_text_encoder',
        'finalizing_model',
        'compiling_model',
        'warming_model',
        'binding_server',
        'ready',
        'stopping',
        'stopped',
        'failed',
    }
)


def _normalize_bind_host(host: Any) -> str:
    """Return a host accepted by socket/server bind APIs.

    IPv6 literals are commonly written with brackets in URLs, but Python's
    socket and websocket server bind calls expect the raw literal.  Brackets
    are added back only when constructing a client URI.
    """
    normalized = str(host).strip()
    if normalized.startswith('[') and normalized.endswith(']'):
        return normalized[1:-1]
    return normalized


def _popen_alive(process: Any) -> bool:
    if process is None:
        return False
    try:
        return process.poll() is None
    except (AttributeError, OSError):
        return False


class FastWAMWorkflowLifecycle(WorkflowLifecycle):
    """Own runtime, FastWAM Server, and the lightweight inference client."""

    workflow_kind = 'fastwam'
    inference_display_name = 'FastWAM'
    checkpoint_owner_stage = 'server'
    checkpoint_config_key = 'ckpt_path'
    checkpoint_requires_params = False
    checkpoint_accepts_file = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        explicit_server_python = kwargs.pop('server_python', None)
        self._server_health_probe: Callable[
            [str, int, Mapping[str, Any]], bool
        ] = kwargs.pop('server_health_probe', self._default_server_health_probe)
        server_run_root = kwargs.pop('server_run_root', None)
        super().__init__(*args, **kwargs)

        configured_python = explicit_server_python or os.environ.get(
            FASTWAM_SERVER_PYTHON_ENV
        )
        self._server_runtime_profile: Optional[RuntimeProfile] = None
        if configured_python is None and not os.environ.get(RELEASE_ROOT_ENV, '').strip():
            try:
                candidate_profile = resolve_runtime_profile(
                    'fastwam-server',
                    config_name=self.config_name,
                    repo_root=self.repo_root,
                )
            except RuntimeProfileError as exc:
                raise WorkflowError(f'FastWAM Server Python profile is invalid: {exc}') from exc
            if candidate_profile is not None:
                if not candidate_profile.verified:
                    raise WorkflowError(
                        f'FastWAM Server Python profile {candidate_profile.name!r} '
                        'is unverified or stale'
                    )
                configured_python = str(candidate_profile.python)
                self._server_runtime_profile = candidate_profile
        self.server_python = Path(
            configured_python or self.inference_python
        ).expanduser().absolute()
        self.server_profile_name = 'fastwam-server'
        self._server_port: Optional[int] = None
        self.server_run_root = Path(
            server_run_root or tempfile.gettempdir()
        ).expanduser().resolve(strict=False)

        self.server_process: Any = None
        self._server_desired = False
        self._server_ready = False
        self._server_error: Optional[str] = None
        self._server_started_at: Optional[float] = None
        self._server_exit_code: Optional[int] = None
        self._server_new_session = False
        self._server_run_id: Optional[str] = None
        self._server_run_dir: Optional[Path] = None
        self._server_ready_file: Optional[Path] = None
        self._server_progress_file: Optional[Path] = None
        self._server_progress: Dict[str, Any] = {
            'phase': 'stopped',
            'updated_at': None,
            'events': [],
        }
        self._server_failure_stop_pending = False
        self._server_failure_stop_thread: Optional[threading.Thread] = None
        self.inference_profile_name = 'fastwam-client'

    def _config_yaml_path(self) -> Path:
        config_root = self.repo_root / 'configs'
        standard = config_root / f'{self.config_name}.yaml'
        if standard.is_file():
            return standard
        return config_root / 'legacy' / f'{self.config_name}.yaml'

    def _server_script(self) -> Path:
        return self.repo_root / 'consumer' / 'fastwam_server.py'

    def _inference_script(self) -> Path:
        return self.repo_root / 'consumer' / 'fastwam_policy_infer.py'

    def _inference_command(self) -> list[str]:
        command = [
            str(self.inference_python),
            str(self._inference_script()),
            '--config-name',
            self.config_name,
        ]
        if self._server_port is not None:
            host, _port = self._server_endpoint()
            command.extend(
                [
                    '--server-host',
                    host,
                    '--server-port',
                    str(self._server_port),
                ]
            )
        return command

    def _apply_inference_checkpoint_override(
        self,
        command: list[str],
        environment: Dict[str, str],
    ) -> None:
        # The heavy model is already loaded by the Server. The client must not
        # depend on checkpoint identity or receive a second checkpoint override.
        del command, environment

    def _policy_params(self) -> Mapping[str, Any]:
        policy = self.config.get('policy')
        if not isinstance(policy, Mapping):
            return {}
        params = policy.get('params')
        return params if isinstance(params, Mapping) else {}

    @staticmethod
    def _coerce_port(value: Any) -> Optional[int]:
        try:
            port = int(value)
        except (TypeError, ValueError):
            return None
        return port if 1 <= port <= 65535 else None

    def _server_port_candidates(self) -> tuple[str, list[int]]:
        params = self._policy_params()
        host = _normalize_bind_host(params.get('host') or DEFAULT_SERVER_HOST)
        preferred = self._coerce_port(params.get('port')) or DEFAULT_SERVER_PORT
        raw_range = params.get('port_range', params.get('server_port_range'))
        values: list[int] = []
        if isinstance(raw_range, Mapping):
            start = self._coerce_port(raw_range.get('start'))
            end = self._coerce_port(raw_range.get('end'))
            if start is not None and end is not None and start <= end:
                values = list(range(start, end + 1))
        elif isinstance(raw_range, str):
            text = raw_range.strip()
            if '-' in text:
                left, right = (part.strip() for part in text.split('-', 1))
                start = self._coerce_port(left)
                end = self._coerce_port(right)
                if start is not None and end is not None and start <= end:
                    values = list(range(start, end + 1))
            else:
                values = [
                    port for item in text.split(',')
                    if (port := self._coerce_port(item.strip())) is not None
                ]
        elif isinstance(raw_range, Sequence) and not isinstance(
            raw_range, (bytes, bytearray, str)
        ):
            parsed = [
                port for item in raw_range
                if (port := self._coerce_port(item)) is not None
            ]
            if len(parsed) == 2 and parsed[0] <= parsed[1]:
                values = list(range(parsed[0], parsed[1] + 1))
            else:
                values = parsed
        range_was_configured = raw_range is not None
        if not values:
            values = list(
                range(DEFAULT_SERVER_PORT_RANGE[0], DEFAULT_SERVER_PORT_RANGE[1] + 1)
            )
        # An explicit range is authoritative.  The legacy ``port`` remains
        # the preferred first choice only when it belongs to that range; this
        # prevents a stale fixed port from escaping the configured pool.
        candidates = (
            [preferred, *values]
            if not range_was_configured or preferred in values
            else values
        )
        return host, list(dict.fromkeys(candidates))

    def _server_endpoint(self) -> tuple[str, int]:
        host, candidates = self._server_port_candidates()
        return host, self._server_port or candidates[0]

    def _port_is_available(self, host: str, port: int) -> bool:
        if not host or self._coerce_port(port) is None:
            return False
        normalized_host = str(host).strip()
        if (
            normalized_host.startswith('[')
            and normalized_host.endswith(']')
        ):
            normalized_host = normalized_host[1:-1]
        bind_host = '127.0.0.1' if normalized_host == 'localhost' else normalized_host
        if normalized_host == '::':
            bind_host = '::'
        family = socket.AF_INET6 if ':' in bind_host else socket.AF_INET
        # Do not set SO_REUSEADDR here.  On some platforms doing so allows a
        # second probe socket to bind over an existing listener, producing a
        # false "available" result immediately before Server startup.
        binder = socket.socket(family, socket.SOCK_STREAM)
        try:
            binder.bind((bind_host, port))
            return True
        except OSError:
            return False
        finally:
            binder.close()

    def _select_server_port(self) -> int:
        host, candidates = self._server_port_candidates()
        for port in candidates:
            if self._port_is_available(host, port):
                return port
        start, end = min(candidates), max(candidates)
        raise WorkflowConflictError(
            f'FastWAM Server 可用端口范围 {start}-{end} 均已被占用'
        )

    def _checkpoint_override_blockers_locked(self) -> list[str]:
        blockers: list[str] = []
        if self._safety_stop_pending:
            blockers.append('Inference safety stop is in progress')
        if _popen_alive(self.server_process) or self._server_desired:
            blockers.append('Stop FastWAM Server before editing checkpoint')
        return blockers

    def _inference_blockers(self, runtime_ready: bool) -> list[str]:
        blockers: list[str] = []
        if not runtime_ready:
            blockers.append('请先正确启动 DeviceManager 与 ActionExecutor')
        if not self._test_inference_mode and not self._server_ready:
            blockers.append('请先启动 FastWAM 模型 Server')
        if not self._path_exists(self.inference_python):
            blockers.append(f'{self.inference_profile_name} Python profile 不可用')
        if not self._path_exists(self._inference_script()):
            blockers.append(
                'FastWAM inference 入口脚本不存在 (fastwam_policy_infer.py)'
            )
        if not self._path_exists(self._config_yaml_path()):
            blockers.append(f'配置文件不存在: configs/{self.config_name}.yaml')
        if self._test_inference_mode:
            mock_error = mock_inference_config_error(self.config)
            if mock_error:
                blockers.append(mock_error)
        blockers.extend(self._recording_blockers())
        return blockers

    def _endpoint_is_occupied(self) -> bool:
        host, port = self._server_endpoint()
        return not self._port_is_available(host, port)

    def _checkpoint_artifact_blockers(self) -> list[str]:
        """Validate the artifact layout required by ``FastWAMRuntime``.

        ``checkpoint_state`` intentionally keeps the generic panel contract
        (for FastWAM, selecting an existing path is enough to edit ownership),
        while the model server needs the complete training export before it can
        load.  Keep this stricter check at the server boundary so a bad path is
        reported before spawning a process that will immediately exit.
        """
        if self._test_inference_mode:
            # Mock inference never loads a checkpoint or starts the model
            # server, so requiring training artifacts would make the software
            # test mode unusable.
            return []

        checkpoint = self._effective_checkpoint_path()
        if checkpoint is None:
            return [
                'Current config is missing '
                f'policy.params.{self.checkpoint_config_key}'
            ]
        if not self._path_exists(checkpoint):
            return [f'Checkpoint does not exist: {checkpoint}']

        try:
            if checkpoint.is_file():
                if checkpoint.suffix.casefold() != '.pt':
                    return [
                        'FastWAM checkpoint file must have a .pt suffix: '
                        f'{checkpoint}'
                    ]
                if (
                    checkpoint.parent.name == 'weights'
                    and checkpoint.parent.parent.name == 'checkpoints'
                ):
                    checkpoint_root = checkpoint.parent.parent.parent
                else:
                    checkpoint_root = checkpoint.parent
            elif checkpoint.is_dir():
                checkpoint_root = checkpoint
                weights_dir = checkpoint_root / 'checkpoints' / 'weights'
                if not weights_dir.is_dir():
                    return [
                        'FastWAM checkpoint is missing '
                        f'checkpoints/weights directory: {weights_dir}'
                    ]
                if not any(
                    path.is_file() and path.suffix.casefold() == '.pt'
                    for path in weights_dir.iterdir()
                ):
                    return [
                        'FastWAM checkpoint has no .pt weights in '
                        f'{weights_dir}'
                    ]
            else:
                return [f'FastWAM checkpoint path is not a file or directory: {checkpoint}']

            missing = [
                name
                for name in ('config.yaml', 'dataset_statistics.json')
                if not (checkpoint_root / name).is_file()
            ]
            if missing:
                return [
                    'FastWAM checkpoint is missing required artifact(s): '
                    + ', '.join(missing)
                    + f' in {checkpoint_root}'
                ]
        except OSError as exc:
            return [f'Unable to inspect FastWAM checkpoint {checkpoint}: {exc}']
        return []

    def _server_blockers(self) -> list[str]:
        if self._test_inference_mode:
            return [
                'FastWAM Server is disabled while mock inference mode is active'
            ]
        blockers: list[str] = []
        release_root = os.environ.get(RELEASE_ROOT_ENV, '').strip()
        external_enabled = os.environ.get(
            FASTWAM_SERVER_PYTHON_EXTERNAL_ENV, ''
        ).strip().casefold() in {'1', 'true', 'yes', 'on'}
        if release_root and not external_enabled:
            bundled_python = (
                Path(release_root).expanduser().resolve(strict=False)
                / 'bin'
                / 'python-policy'
            )
            if self.server_python == bundled_python:
                blockers.append(
                    'FastWAM release requires a separately provisioned NMX model '
                    'environment. Set SUPERINFERENCE_FASTWAM_SERVER_PYTHON and '
                    'SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL=1 before launch.'
                )
        if not self._path_exists(self.server_python):
            blockers.append(
                f'{self.server_profile_name} Python profile 不可用：'
                f'{self.server_python}'
            )
        if not self._path_exists(self._server_script()):
            blockers.append('FastWAM server 入口脚本不存在 (fastwam_server.py)')
        if not self._path_exists(self._config_yaml_path()):
            blockers.append(f'配置文件不存在: configs/{self.config_name}.yaml')
        checkpoint = self.checkpoint_state()
        if checkpoint.get('error'):
            blockers.append(str(checkpoint['error']))
        blockers.extend(self._checkpoint_artifact_blockers())
        host, candidates = self._server_port_candidates()
        if not host or not candidates:
            blockers.append(f'FastWAM Server endpoint 非法: {host}')
        elif not _popen_alive(self.server_process) and not any(
            self._port_is_available(host, candidate) for candidate in candidates
        ):
            blockers.append(
                f'FastWAM Server 可用端口范围 {min(candidates)}-{max(candidates)} '
                '均已被占用'
            )
        return blockers

    def _prepare_server_run(self) -> tuple[str, Path, Path, Path]:
        self.server_run_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(
            tempfile.mkdtemp(
                prefix='control-panel-fastwam-',
                dir=str(self.server_run_root),
            )
        ).resolve()
        os.chmod(run_dir, 0o700)
        return (
            secrets.token_hex(16),
            run_dir,
            run_dir / 'ready.json',
            run_dir / 'progress.json',
        )

    def _cleanup_server_run(self) -> None:
        run_dir = self._server_run_dir
        self._server_ready_file = None
        self._server_progress_file = None
        self._server_run_dir = None
        self._server_run_id = None
        if run_dir is None:
            return
        try:
            parent = run_dir.resolve(strict=False).parent
            expected_parent = self.server_run_root.resolve(strict=False)
            if parent != expected_parent or not run_dir.name.startswith(
                'control-panel-fastwam-'
            ):
                self._log('ERROR', f'Refused to remove unexpected Server run dir: {run_dir}')
                return
            shutil.rmtree(run_dir, ignore_errors=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._log('WARNING', f'Unable to remove FastWAM Server run dir: {exc}')

    def start_server(
        self,
        devices: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        with self._lock:
            if self._safety_stop_pending:
                raise WorkflowConflictError('Asynchronous safety stop is in progress')
            self._refresh_locked()
            if _popen_alive(self.server_process):
                return self._state_locked(devices)
            blockers = self._server_blockers()
            if blockers:
                raise WorkflowConflictError('; '.join(blockers))

            checkpoint = self._effective_checkpoint_path()
            if checkpoint is None:
                raise WorkflowConflictError('FastWAM checkpoint is not configured')
            host, _candidates = self._server_port_candidates()
            selected_port = self._select_server_port()
            self._server_port = selected_port
            try:
                run_id, run_dir, ready_file, progress_file = self._prepare_server_run()
            except OSError as exc:
                self._server_port = None
                self._server_error = f'Unable to prepare FastWAM Server run directory: {exc}'
                raise WorkflowError(self._server_error) from exc
            command = [
                str(self.server_python),
                str(self._server_script()),
                '--config-name',
                self.config_name,
                '--ckpt-path',
                str(checkpoint),
                '--host',
                host,
                '--port',
                str(selected_port),
            ]
            environment = (
                self._server_runtime_profile.process_environment(
                    os.environ,
                    config_name=self.config_name,
                )
                if self._server_runtime_profile is not None
                else dict(os.environ)
            )
            environment.update({
                'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                FASTWAM_READY_FILE_ENV: str(ready_file),
                FASTWAM_PROGRESS_FILE_ENV: str(progress_file),
                FASTWAM_RUN_ID_ENV: run_id,
                FASTWAM_CONFIG_NAME_ENV: self.config_name,
                FASTWAM_SERVER_PORT_ENV: str(selected_port),
            })
            self._server_run_id = run_id
            self._server_run_dir = run_dir
            self._server_ready_file = ready_file
            self._server_progress_file = progress_file
            start_time = time.time()
            self._server_progress = {
                'phase': 'spawning_process',
                'updated_at': start_time,
                'events': [
                    {
                        'phase': 'spawning_process',
                        'updated_at': start_time,
                        'elapsed_ms': 0,
                    }
                ],
            }
            try:
                self.server_process = self._popen_factory(
                    command,
                    cwd=str(self.repo_root),
                    env=environment,
                    start_new_session=True,
                )
            except OSError as exc:
                self._cleanup_server_run()
                self._server_port = None
                self._server_error = f'Unable to start FastWAM Server: {exc}'
                raise WorkflowError(self._server_error) from exc
            self._server_new_session = True
            self._server_desired = True
            self._server_ready = False
            self._server_error = None
            self._server_exit_code = None
            self._server_started_at = start_time
            self._log('INFO', f'FastWAM Server start requested for {self.config_name}')
            return self._state_locked(devices)

    def _signal_server(self, sig: int) -> None:
        process = self.server_process
        if process is None:
            return
        if self._server_new_session and getattr(process, 'pid', None):
            try:
                os.killpg(os.getpgid(process.pid), sig)
                return
            except (OSError, ProcessLookupError):
                pass
        try:
            process.send_signal(sig)
        except (AttributeError, OSError, ValueError):
            pass

    def request_async_server_stop(self) -> None:
        """Interrupt model-server startup without waiting for lifecycle lock.

        The control dispatcher calls this hook when a stop/force-stop request
        arrives while the server is still loading a checkpoint. The normal
        ``stop_server`` path remains responsible for escalation, waiting, and
        private run-directory cleanup.
        """
        self._server_desired = False
        self._signal_server(signal.SIGINT)

    def _stop_server_locked(self, *, preserve_error: bool = False) -> None:
        process = self.server_process
        self._server_desired = False
        if process is not None and _popen_alive(process):
            self._signal_server(signal.SIGINT)
            try:
                process.wait(timeout=10.0)
            except (AttributeError, subprocess.TimeoutExpired):
                self._signal_server(signal.SIGTERM)
                try:
                    process.wait(timeout=5.0)
                except (AttributeError, subprocess.TimeoutExpired):
                    self._signal_server(signal.SIGKILL)
                    try:
                        process.wait(timeout=2.0)
                    except (AttributeError, subprocess.TimeoutExpired):
                        pass
        if process is not None:
            try:
                code = process.poll()
            except (AttributeError, OSError):
                code = None
            if code is None:
                self._server_error = (
                    'FastWAM Server did not exit after SIGINT/SIGTERM/SIGKILL'
                )
                self._log('ERROR', self._server_error)
                raise WorkflowConflictError(self._server_error)
            self._server_exit_code = code
        self.server_process = None
        self._server_new_session = False
        self._server_ready = False
        self._server_port = None
        self._server_started_at = None
        self._server_progress = {
            'phase': 'stopped',
            'updated_at': time.time(),
            'events': [],
        }
        self._cleanup_server_run()
        if not preserve_error:
            self._server_error = None
        self._log('INFO', 'FastWAM Server stopped')

    def stop_server(
        self,
        devices: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        with self._lock:
            if self._safety_stop_pending:
                return self._state_locked(devices)
            self._stop_inference_locked()
            self._stop_server_locked()
            return self._state_locked(devices)

    @staticmethod
    def _identity_matches(
        payload: Any,
        expected: Mapping[str, Any],
        *,
        include_port: bool,
    ) -> bool:
        if not isinstance(payload, Mapping):
            return False
        try:
            pid_matches = int(payload.get('pid')) == int(expected.get('pid'))
            port_matches = (
                not include_port
                or int(payload.get('port')) == int(expected.get('port'))
            )
        except (TypeError, ValueError):
            return False
        return bool(
            payload.get('model_loaded') is True
            and str(payload.get('run_id') or '')
            == str(expected.get('run_id') or '')
            and pid_matches
            and str(payload.get('config_name') or '')
            == str(expected.get('config_name') or '')
            and port_matches
        )

    @staticmethod
    def _default_server_health_probe(
        host: str,
        port: int,
        expected: Mapping[str, Any],
    ) -> bool:
        connect_host = '127.0.0.1' if host in {'0.0.0.0', '::'} else host
        if ':' in connect_host and not connect_host.startswith('['):
            connect_host = f'[{connect_host}]'
        opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
        try:
            with opener.open(
                f'http://{connect_host}:{port}/healthz',
                timeout=SERVER_READY_PROBE_TIMEOUT_S,
            ) as response:
                if getattr(response, 'status', 200) != 200:
                    return False
                payload = json.loads(response.read(65536).decode('utf-8'))
        except (OSError, ValueError):
            return False
        return bool(
            isinstance(payload, Mapping)
            and payload.get('ok') is True
            and FastWAMWorkflowLifecycle._identity_matches(
                payload,
                expected,
                include_port=False,
            )
        )

    def _owned_ready_file_matches(self, expected: Mapping[str, Any]) -> bool:
        ready_file = self._server_ready_file
        if ready_file is None:
            return False
        try:
            info = ready_file.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                return False
            if hasattr(os, 'getuid') and info.st_uid != os.getuid():
                return False
            if info.st_size <= 0 or info.st_size > 65536:
                return False
            payload = json.loads(ready_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        return self._identity_matches(
            payload,
            expected,
            include_port=True,
        )

    def _read_owned_progress_file(
        self,
        expected: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        progress_file = self._server_progress_file
        if progress_file is None:
            return None
        try:
            info = progress_file.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                return None
            if hasattr(os, 'getuid') and info.st_uid != os.getuid():
                return None
            if info.st_size <= 0 or info.st_size > 65536:
                return None
            payload = json.loads(progress_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            pid_matches = int(payload.get('pid')) == int(expected.get('pid'))
        except (TypeError, ValueError):
            return None
        if not (
            str(payload.get('run_id') or '') == str(expected.get('run_id') or '')
            and pid_matches
            and str(payload.get('config_name') or '')
            == str(expected.get('config_name') or '')
        ):
            return None
        phase = str(payload.get('phase') or '').strip().casefold()
        if phase not in FASTWAM_SERVER_PROGRESS_PHASES:
            return None
        events = []
        raw_events = payload.get('events')
        if isinstance(raw_events, Sequence) and not isinstance(
            raw_events, (str, bytes, bytearray)
        ):
            for raw_event in raw_events[-32:]:
                if not isinstance(raw_event, Mapping):
                    continue
                event_phase = str(raw_event.get('phase') or '').strip().casefold()
                if event_phase not in FASTWAM_SERVER_PROGRESS_PHASES:
                    continue
                try:
                    elapsed_ms = max(0, int(raw_event.get('elapsed_ms') or 0))
                    updated_at = float(raw_event.get('updated_at'))
                except (TypeError, ValueError):
                    continue
                events.append(
                    {
                        'phase': event_phase,
                        'elapsed_ms': elapsed_ms,
                        'updated_at': updated_at,
                    }
                )
        try:
            updated_at_value = float(payload.get('updated_at'))
        except (TypeError, ValueError):
            updated_at_value = None
        return {
            'phase': phase,
            'updated_at': updated_at_value,
            'events': events,
        }

    def _server_progress_state(
        self,
        *,
        server_alive: bool,
        server_running: bool,
        server_status: str,
    ) -> Dict[str, Any]:
        if server_alive:
            expected = {
                'run_id': self._server_run_id,
                'pid': getattr(self.server_process, 'pid', None),
                'config_name': self.config_name,
            }
            observed = self._read_owned_progress_file(expected)
            if observed is not None:
                self._server_progress = observed
        progress = copy.deepcopy(self._server_progress)
        if server_running:
            progress['phase'] = 'ready'
        elif server_status == 'stopping':
            progress['phase'] = 'stopping'
        elif server_status == 'failed':
            progress['phase'] = 'failed'
        elif not server_alive:
            progress['phase'] = 'stopped'
        started_at = self._server_started_at
        if server_alive and started_at is not None:
            progress['elapsed_ms'] = max(
                0, int((time.time() - started_at) * 1000.0)
            )
        elif progress.get('elapsed_ms') is None:
            progress['elapsed_ms'] = None
        return progress

    def _probe_server_ready(self) -> bool:
        process = self.server_process
        if process is None or not _popen_alive(process):
            return False
        host, port = self._server_endpoint()
        expected = {
            'run_id': self._server_run_id,
            'pid': getattr(process, 'pid', None),
            'config_name': self.config_name,
            'port': port,
        }
        return bool(
            self._owned_ready_file_matches(expected)
            and self._server_health_probe(host, port, expected)
        )

    def _schedule_server_failure_stop_locked(self, reason: str) -> None:
        if self._server_failure_stop_pending or not _popen_alive(
            self.inference_process
        ):
            return
        self._server_failure_stop_pending = True
        self._inference_desired = False
        self._inference_ready = False
        self._inference_error = reason

        def stop_worker() -> None:
            try:
                # The worker is launched from an observation request, but runs
                # after that request releases the lock.  Reacquire it before
                # touching the shared process/coordinator state.
                with self._lock:
                    self._stop_inference_locked(preserve_error=True)
            except BaseException as exc:
                self._log(
                    'ERROR',
                    'FastWAM inference stop after Server failure failed: '
                    f'{type(exc).__name__}: {exc}',
                )
            finally:
                with self._lock:
                    self._server_failure_stop_pending = False
                    self._server_failure_stop_thread = None

        self._server_failure_stop_thread = threading.Thread(
            target=stop_worker,
            name='fastwam-server-failure-stop',
            daemon=True,
        )
        self._server_failure_stop_thread.start()

    def _refresh_locked(self, *, defer_safety_stop: bool = False) -> None:
        super()._refresh_locked(defer_safety_stop=defer_safety_stop)
        process = self.server_process
        if process is None:
            self._server_ready = False
            return
        if not _popen_alive(process):
            expected = {
                'run_id': self._server_run_id,
                'pid': getattr(process, 'pid', None),
                'config_name': self.config_name,
            }
            observed_progress = self._read_owned_progress_file(expected)
            if observed_progress is not None:
                self._server_progress = observed_progress
            try:
                code = process.poll()
            except (AttributeError, OSError):
                code = None
            if self._server_desired and self._server_error is None:
                self._server_error = (
                    f'FastWAM Server exited unexpectedly (code={code})'
                )
                self._log('ERROR', self._server_error)
            reason = self._server_error or 'FastWAM Server is unavailable'
            self._server_exit_code = code
            self._server_desired = False
            self._server_ready = False
            failed_at = time.time()
            failed_elapsed_ms = (
                max(0, int((failed_at - self._server_started_at) * 1000.0))
                if self._server_started_at is not None
                else None
            )
            failed_events = copy.deepcopy(self._server_progress.get('events', []))
            if not failed_events or failed_events[-1].get('phase') != 'failed':
                failed_events.append(
                    {
                        'phase': 'failed',
                        'updated_at': failed_at,
                        'elapsed_ms': failed_elapsed_ms or 0,
                    }
                )
            self._server_progress = {
                'phase': 'failed',
                'updated_at': failed_at,
                'elapsed_ms': failed_elapsed_ms,
                'events': failed_events[-32:],
            }
            self.server_process = None
            self._server_new_session = False
            self._server_port = None
            self._cleanup_server_run()
            self._schedule_server_failure_stop_locked(reason)
            return
        if not self._server_ready:
            self._server_ready = self._probe_server_ready()

    def has_active_processes(self) -> bool:
        with self._lock:
            if self._server_failure_stop_pending:
                return True
            self._refresh_locked()
            return _popen_alive(self.server_process) or super().has_active_processes()

    def stop_all(self) -> None:
        with self._lock:
            if self._safety_stop_pending:
                return
            self._stop_runtime_locked(stop_inference=True)
            self._stop_server_locked()

    def force_stop_all(self) -> Dict[str, Any]:
        with self._lock:
            result = super().force_stop_all()
            process = self.server_process
            server_pid: Optional[int] = None
            if process is not None:
                try:
                    server_pid = int(process.pid)
                except (AttributeError, TypeError, ValueError):
                    server_pid = None
                if _popen_alive(process):
                    self._signal_server(signal.SIGKILL)
                    try:
                        process.wait(timeout=2.0)
                    except (AttributeError, subprocess.TimeoutExpired):
                        pass
            server_remaining = _popen_alive(process)
            if not server_remaining:
                if process is not None:
                    try:
                        self._server_exit_code = process.poll()
                    except (AttributeError, OSError):
                        self._server_exit_code = None
                self.server_process = None
                if server_pid is not None:
                    result['killed_pids'] = sorted(
                        set(result.get('killed_pids', [])) | {server_pid}
                    )
            else:
                result['remaining'] = sorted(
                    set(result.get('remaining', [])) | {'server'}
                )
            self._server_desired = False
            self._server_ready = False
            self._server_new_session = False
            self._server_port = None
            self._server_started_at = None
            self._server_progress = {
                'phase': 'stopped',
                'updated_at': time.time(),
                'events': [],
            }
            self._cleanup_server_run()
            result['ok'] = not result.get('remaining') and not result.get(
                'remaining_shm'
            )
            return result

    def _state_locked(
        self,
        devices: Sequence[Mapping[str, Any]],
        *,
        defer_safety_stop: bool = False,
    ) -> Dict[str, Any]:
        snapshot = super()._state_locked(
            devices,
            defer_safety_stop=defer_safety_stop,
        )
        server_alive = _popen_alive(self.server_process)
        server_running = bool(server_alive and self._server_ready)
        if self._safety_stop_pending:
            server_status = 'stopping'
            server_blockers = [
                self._safety_stop_reason or 'Asynchronous safety stop is in progress'
            ]
        elif server_running:
            server_status = 'running'
            server_blockers = []
        elif server_alive:
            server_status = 'starting'
            server_blockers = []
        elif self._server_error:
            server_status = 'failed'
            server_blockers = self._server_blockers()
        else:
            server_status = 'stopped'
            server_blockers = self._server_blockers()
        host, port = self._server_endpoint()
        _configured_host, configured_ports = self._server_port_candidates()
        server_progress = self._server_progress_state(
            server_alive=server_alive,
            server_running=server_running,
            server_status=server_status,
        )
        snapshot['server'] = {
            'status': server_status,
            'desired_state': 'running' if self._server_desired else 'stopped',
            'running': server_running,
            'active': server_alive or self._safety_stop_pending,
            'ready': server_running,
            'can_start': (
                not self._safety_stop_pending
                and not server_alive
                and not server_blockers
            ),
            'can_stop': server_alive or self._safety_stop_pending,
            'blockers': server_blockers,
            'error': self._server_error,
            'reason': self._server_error,
            'pid': getattr(self.server_process, 'pid', None) if server_alive else None,
            'started_at': self._server_started_at,
            'exit_code': self._server_exit_code,
            'endpoint': f'{host}:{port}',
            'port': port,
            'port_range': configured_ports,
            'checkpoint': self.checkpoint_state(),
            'progress': server_progress,
        }
        if self._server_failure_stop_pending:
            snapshot['inference']['status'] = 'stopping'
            snapshot['inference']['desired_state'] = 'stopped'
            snapshot['inference']['running'] = False
            snapshot['inference']['can_start'] = False
            snapshot['inference']['can_stop'] = False
        snapshot['profiles']['server'] = {
            'name': self.server_profile_name,
            'available': self._path_exists(self.server_python),
        }
        snapshot['stop_order'] = [
            'recording',
            'inference',
            'action_executor',
            'device_manager',
            'server',
            'devices',
        ]
        self._last_observation = copy.deepcopy(snapshot)
        return snapshot


__all__ = ['FastWAMWorkflowLifecycle']
