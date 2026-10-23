"""Owned recording-control spool for the web control panel.

The browser can request only a small action enum. It never supplies process
paths, output paths, frame numbers, or timestamps. The recorder remains the
sole owner of authoritative progress and episode sidecars.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
import uuid

from utils.runtime_profiles import RuntimeProfileError, select_runtime_python

from .home_transition import HomeTransitionError, normalize_home_target


_RECORDING_PROFILE_CHECK_SCRIPT = """
import sys
try:
    from importlib.metadata import version
    from dm3_core.mcap import McapFileType, ReaderFactory, WriterFactory
    from dm3_core.nedf import NEDFFactory, NEDFReaderBase, NEDFType, NEDFWriterBase
    from nmx_msg.Arm_pb2 import ArmEndForce, ArmState, ArmTcpCmd, ArmTcpState
    from nmx_msg.Hand_pb2 import HandCmd, HandState
    from nmx_msg.Image_pb2 import RGBD
    from nmx_msg.Metadata_pb2 import Metadata
except Exception:
    print("required NEDF3 recorder imports unavailable", file=sys.stderr)
    raise SystemExit(11)
expected_versions = {
    "dm3-core": "0.2.18",
    "nmx_msg": "2.2.0",
}
actual_versions = {name: version(name) for name in expected_versions}
if actual_versions != expected_versions:
    print(
        "unexpected NEDF3 SDK versions: "
        + repr(actual_versions)
        + " expected "
        + repr(expected_versions),
        file=sys.stderr,
    )
    raise SystemExit(12)
required = (
    (NEDFWriterBase, "write_message"),
    (NEDFReaderBase, "iter_decoded_messages"),
    (NEDFReaderBase, "get_raw_topics_timestamps"),
    (NEDFFactory, "get_writer"),
    (NEDFFactory, "get_reader"),
)
missing = [f"{owner.__name__}.{name}" for owner, name in required
           if not callable(getattr(owner, name, None))]
if not ReaderFactory or not WriterFactory or not McapFileType:
    missing.append("dm3_core.mcap factories")
if not hasattr(NEDFType, "NEDF3"):
    missing.append("NEDFType.NEDF3")
if missing:
    print("missing required NEDF3 recorder API: " + ", ".join(missing),
          file=sys.stderr)
    raise SystemExit(13)
for message_type in (
    RGBD, ArmTcpCmd, ArmTcpState, ArmState, ArmEndForce,
    HandCmd, HandState, Metadata,
):
    if not hasattr(message_type, "SerializeToString"):
        print("missing required NEDF3 protobuf API", file=sys.stderr)
        raise SystemExit(14)
"""

DEFAULT_RECORDING_RUN_ROOT = Path(
    '/tmp/superinference-control-panel/runs'
)
RECORDING_ACTIONS = frozenset(
    {
        'episode_accept',
        'episode_discard',
        'segment_accept',
        'segment_reject',
    }
)
RECORDING_CAMERA_FIELDS = (
    'head_camera',
    'left_camera',
    'right_camera',
)
_INTERNAL_ACTIONS = frozenset({'shutdown'})
_STATUS_VALUES = frozenset(
    {
        'starting',
        'armed',
        'recording',
        'finalizing',
        'transitioning',
        'incomplete',
        'error',
        'stopped',
    }
)
_READY_START_STATUSES = frozenset({'armed', 'recording'})
_HEALTHY_STATUSES = frozenset(
    {'armed', 'recording', 'finalizing', 'transitioning'}
)
_ACK_STATUS_VALUES = frozenset({'applied', 'rejected', 'failed'})
_SAFE_TOKEN = re.compile(r'^[A-Za-z0-9_.:-]{1,160}$')
_MAX_JSON_BYTES = 1_000_000
_MAX_TIMELINE_ITEMS = 5000


class RecordingError(RuntimeError):
    """Base error for the controlled recording channel."""


class RecordingConflictError(RecordingError):
    """The recording state changed or another action is pending."""


class RecordingRequestError(RecordingError):
    """A recording request has an invalid public shape."""


class RecordingSecurityError(RecordingError):
    """A spool artifact failed ownership or schema validation."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any, *, limit: int = 500) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value[:limit]


def _safe_error(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        code = _safe_text(value.get('code'), limit=80)
        message = _safe_text(value.get('message'), limit=400)
        if code and message:
            return f'{code}: {message}'
        return message or code
    return _safe_text(value)


def _safe_token(value: Any) -> Optional[str]:
    if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value):
        return value
    return None


def _safe_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _safe_nonnegative_number(value: Any) -> Optional[float | int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return value


def resolve_recording_python(
    config: Mapping[str, Any],
    *,
    explicit: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    config_name: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve explicit, environment, local-profile, then legacy config Python."""
    environment = os.environ if environ is None else environ
    configured_path: Optional[Path] = None
    policy = config.get('policy')
    if isinstance(policy, Mapping):
        rollout = policy.get('infer_rollout')
        if isinstance(rollout, Mapping):
            configured = rollout.get('python_path')
            if isinstance(configured, str) and configured.strip():
                configured_path = Path(configured).expanduser()
    try:
        selection = select_runtime_python(
            'recording',
            explicit=explicit,
            environment_names=(
                'SUPERINFERENCE_RECORDING_PYTHON',
                'SUPERINFERENCE_RECORDER_PYTHON',
            ),
            config_name=config_name,
            fallback=configured_path,
            repo_root=repo_root,
            environ=environment,
            require_verified_profile=True,
        )
    except RuntimeProfileError as exc:
        raise RecordingError(f'recording Python profile is invalid: {exc}') from exc
    return selection.python if selection is not None else None


def default_recording_profile_check(python_path: Path) -> Tuple[bool, str]:
    """Verify the approved interpreter and exact NEDF3 SDK without hardware."""
    path = Path(python_path)
    if not path.is_file() or not os.access(path, os.X_OK):
        return False, f'recording Python profile is not executable: {path}'
    try:
        completed = subprocess.run(
            [
                str(path),
                '-c',
                _RECORDING_PROFILE_CHECK_SCRIPT,
            ],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f'recording profile check failed: {type(exc).__name__}'
    if completed.returncode == 0:
        return True, ''
    detail = (completed.stderr or '').strip().splitlines()
    if detail and detail[-1].startswith(
        (
            'missing required NEDF3 recorder API:',
            'required NEDF3 recorder imports',
            'unexpected NEDF3 SDK versions:',
            'missing required NEDF3 protobuf API',
        )
    ):
        return False, detail[-1]
    return False, 'recording profile lacks required NEDF3 recorder API'


def recording_config_state(
    config: Mapping[str, Any],
) -> Tuple[bool, bool, bool]:
    """Return configured, enabled, required for inference rollout recording."""
    policy = config.get('policy')
    rollout: Any = None
    if isinstance(policy, Mapping):
        rollout = policy.get('infer_rollout')
    configured = isinstance(rollout, Mapping)
    enabled = configured and rollout.get('enabled', False) is True
    required = True if not configured else rollout.get('required', True) is not False
    return configured, enabled, required


def normalize_recording_camera_mapping(
    config: Mapping[str, Any],
) -> Dict[str, Optional[str]]:
    """Return normalized NEDF camera IDs used by the LeRobot converter."""
    policy = config.get('policy')
    rollout: Any = None
    if isinstance(policy, Mapping):
        rollout = policy.get('infer_rollout')

    normalized: Dict[str, Optional[str]] = {}
    for field in RECORDING_CAMERA_FIELDS:
        value = rollout.get(field) if isinstance(rollout, Mapping) else None
        if value is None:
            normalized[field] = None
            continue
        if not isinstance(value, str):
            raise ValueError(
                f'policy.infer_rollout.{field} must be a string or null'
            )
        normalized[field] = value.strip() or None
    return normalized


def validate_recording_camera_mapping(
    config: Mapping[str, Any],
) -> list[str]:
    """Validate optional converter camera roles without touching any device."""
    try:
        mapping = normalize_recording_camera_mapping(config)
    except ValueError as exc:
        return [str(exc)]
    if not any(mapping.values()):
        return []
    errors: list[str] = []
    if mapping['left_camera'] is None:
        errors.append(
            'policy.infer_rollout.left_camera is required when camera roles are configured'
        )
    if mapping['right_camera'] is None:
        errors.append(
            'policy.infer_rollout.right_camera is required when camera roles are configured'
        )

    camera_keys: set[str] = set()
    devices = config.get('devices')
    if isinstance(devices, list):
        for device in devices:
            if not isinstance(device, Mapping):
                continue
            enabled = device.get('enabled', True)
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() not in {
                    'false',
                    '0',
                    'no',
                    'off',
                }
            if enabled is False:
                continue
            device_class = device.get('class')
            if (
                not isinstance(device_class, str)
                or 'camera' not in device_class.casefold()
                or 'device_class_id' not in device
            ):
                continue
            camera_keys.add(f"{device_class}_{device['device_class_id']}")

    for field, camera_key in mapping.items():
        if camera_key is not None and camera_key not in camera_keys:
            errors.append(
                f'policy.infer_rollout.{field} references an unconfigured '
                f'camera device: {camera_key}'
            )

    left = mapping['left_camera']
    right = mapping['right_camera']
    head = mapping['head_camera']
    if left is not None and right is not None and left == right:
        errors.append(
            'policy.infer_rollout.left_camera and right_camera must reference '
            'different devices'
        )
    if head is not None and head in {left, right}:
        errors.append(
            'policy.infer_rollout.head_camera must differ from both wrist cameras'
        )
    return errors


def recording_timeout(
    config: Mapping[str, Any], key: str, default: float
) -> float:
    policy = config.get('policy')
    rollout: Any = None
    if isinstance(policy, Mapping):
        rollout = policy.get('infer_rollout')
    value = rollout.get(key, default) if isinstance(rollout, Mapping) else default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    if not math.isfinite(numeric):
        return default
    return max(0.1, min(numeric, 120.0))


def empty_recording_state(
    *,
    configured: bool,
    enabled: bool,
    required: bool,
    error: Optional[str] = None,
    blockers: Optional[list[str]] = None,
) -> Dict[str, Any]:
    capability_blockers = list(blockers or [])
    if error and error not in capability_blockers:
        capability_blockers.append(error)
    if not configured:
        capability_blockers.append('当前配置未配置推理录制')
    elif not enabled:
        capability_blockers.append('当前配置未启用推理录制')
    capability_blockers = list(dict.fromkeys(capability_blockers))
    capabilities = {
        action: {'can': False, 'blockers': capability_blockers}
        for action in sorted(RECORDING_ACTIONS)
    }
    return {
        'configured': bool(configured),
        'enabled': bool(enabled),
        'required': bool(required),
        'status': 'error' if error else 'stopped',
        'active': False,
        'recorder_ready': False,
        'error': error,
        'run_id': None,
        'revision': 0,
        'reference': None,
        'episode': None,
        'segments': [],
        'checkpoints': [],
        'pending_command': None,
        'last_command': None,
        'capabilities': capabilities,
    }


def _pid_is_descendant(candidate_pid: int, ancestor_pid: int) -> bool:
    """Return true when candidate is ancestor itself or one of its descendants."""
    current = int(candidate_pid)
    ancestor = int(ancestor_pid)
    if current <= 0 or ancestor <= 0:
        return False
    seen: set[int] = set()
    for _ in range(128):
        if current == ancestor:
            return True
        if current <= 1 or current in seen:
            return False
        seen.add(current)
        try:
            raw = (Path('/proc') / str(current) / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2 :].split()
            current = int(fields[1])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            return False
    return False


class RecordingCoordinator:
    """Own one private command/status spool for an inference process tree."""

    def __init__(
        self,
        run_root: Path = DEFAULT_RECORDING_RUN_ROOT,
        *,
        configured: bool = True,
        enabled: bool = True,
        required: bool = True,
        pid_validator: Callable[[int, int], bool] = _pid_is_descendant,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        conversion_enqueue: Optional[Callable[[Mapping[str, Any]], Any]] = None,
        conversion_cancel: Optional[Callable[[str], Any]] = None,
        finalized_enqueue: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.configured = bool(configured)
        self.enabled = bool(enabled)
        self.required = bool(required)
        self._pid_validator = pid_validator
        self._clock = clock
        self._sleep = sleep
        self.run_id: Optional[str] = None
        self.control_dir: Optional[Path] = None
        self.owner_pid: Optional[int] = None
        self._trusted_owner_pids: set[int] = set()
        self._commands: Dict[str, Dict[str, Any]] = {}
        self._acks: Dict[str, Dict[str, Any]] = {}
        self._conversion_enqueue = conversion_enqueue
        self._conversion_cancel = conversion_cancel
        self._conversion_notified: set[str] = set()
        self._finalized_enqueue = finalized_enqueue
        self._finalized_notified: set[str] = set()
        self._finalized_receipt_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def commands_dir(self) -> Path:
        return self._required_control_dir() / 'commands'

    @property
    def acks_dir(self) -> Path:
        return self._required_control_dir() / 'acks'

    @property
    def status_dir(self) -> Path:
        return self._required_control_dir() / 'status'

    def _required_control_dir(self) -> Path:
        if self.control_dir is None:
            raise RecordingConflictError('Recording run has not been prepared')
        return self.control_dir

    @staticmethod
    def _current_uid() -> Optional[int]:
        getuid = getattr(os, 'getuid', None)
        return int(getuid()) if callable(getuid) else None

    def _verify_owned_directory(self, path: Path) -> None:
        try:
            result = path.lstat()
        except OSError as exc:
            raise RecordingSecurityError(f'Cannot inspect recording directory: {path.name}') from exc
        if not stat.S_ISDIR(result.st_mode) or stat.S_ISLNK(result.st_mode):
            raise RecordingSecurityError('Recording spool path is not a real directory')
        uid = self._current_uid()
        if uid is not None and result.st_uid != uid:
            raise RecordingSecurityError('Recording spool directory has a foreign owner')
        if stat.S_IMODE(result.st_mode) & 0o077:
            raise RecordingSecurityError('Recording spool directory is not private')

    def prepare_run(self) -> Path:
        if self.control_dir is not None:
            return self.control_dir
        self.run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.run_root, 0o700)
        except OSError:
            pass
        self._verify_owned_directory(self.run_root)

        run_id = str(uuid.uuid4())
        control_dir = self.run_root / run_id
        control_dir.mkdir(mode=0o700)
        for name in ('commands', 'acks', 'status'):
            (control_dir / name).mkdir(mode=0o700)
        self._verify_owned_directory(control_dir)
        self.run_id = run_id
        self.control_dir = control_dir
        self._atomic_write_json(
            control_dir / 'manifest.json',
            {
                'schema_version': 1,
                'run_id': run_id,
                'owner_pid': None,
                'created_at': _utc_timestamp(),
            },
        )
        return control_dir

    def bind_owner(self, owner_pid: int) -> None:
        if isinstance(owner_pid, bool) or not isinstance(owner_pid, int) or owner_pid <= 0:
            raise RecordingSecurityError('Inference process did not provide a valid PID')
        self.owner_pid = int(owner_pid)
        self._trusted_owner_pids = {self.owner_pid}
        self._atomic_write_json(
            self._required_control_dir() / 'manifest.json',
            {
                'schema_version': 1,
                'run_id': self.run_id,
                'owner_pid': self.owner_pid,
                'created_at': _utc_timestamp(),
            },
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        descriptor = os.open(str(path), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        parent = path.parent
        self._verify_owned_directory(parent)
        encoded = (
            json.dumps(payload, ensure_ascii=True, separators=(',', ':'), sort_keys=True)
            + '\n'
        ).encode('utf-8')
        if len(encoded) > _MAX_JSON_BYTES:
            raise RecordingRequestError('Recording command exceeds the size limit')
        temporary = parent / f'.{path.name}.{uuid.uuid4().hex}.tmp'
        descriptor = os.open(
            str(temporary),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        self._fsync_directory(parent)

    def _secure_read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
            descriptor = os.open(str(path), flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RecordingSecurityError(f'Cannot open recording artifact: {path.name}') from exc
        try:
            result = os.fstat(descriptor)
            uid = self._current_uid()
            if not stat.S_ISREG(result.st_mode):
                raise RecordingSecurityError('Recording artifact is not a regular file')
            if uid is not None and result.st_uid != uid:
                raise RecordingSecurityError('Recording artifact has a foreign owner')
            if result.st_size < 0 or result.st_size > _MAX_JSON_BYTES:
                raise RecordingSecurityError('Recording artifact exceeds the size limit')
            data = b''
            while len(data) <= _MAX_JSON_BYTES:
                chunk = os.read(descriptor, min(65536, _MAX_JSON_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data += chunk
        finally:
            os.close(descriptor)
        try:
            decoded = json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordingSecurityError('Recording artifact is not valid JSON') from exc
        if not isinstance(decoded, dict):
            raise RecordingSecurityError('Recording artifact must be a JSON object')
        return decoded

    def _validate_owner(self, raw: Mapping[str, Any]) -> int:
        if raw.get('run_id') != self.run_id:
            raise RecordingSecurityError('Recording artifact belongs to another run')
        candidate = raw.get('owner_pid')
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
            raise RecordingSecurityError('Recording artifact has no valid owner PID')
        if candidate not in self._trusted_owner_pids:
            if self.owner_pid is None or not self._pid_validator(candidate, self.owner_pid):
                raise RecordingSecurityError('Recording artifact owner is outside the inference tree')
            # A recorder may exit immediately after writing its shutdown ACK.
            # Cache the PID once it has been verified while alive so that ACK
            # remains attributable after /proc removes the exited child.
            self._trusted_owner_pids.add(int(candidate))
        return int(candidate)

    @staticmethod
    def _sanitize_reference(value: Any) -> Optional[Dict[str, str]]:
        if not isinstance(value, Mapping):
            return None
        kind = _safe_token(value.get('kind'))
        name = _safe_token(value.get('name'))
        if kind is None or name is None:
            return None
        return {'kind': kind, 'name': name}

    @staticmethod
    def _sanitize_episode(value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, Mapping):
            return None
        episode_id = _safe_token(value.get('id'))
        index = _safe_nonnegative_int(value.get('index'))
        if episode_id is None or index is None:
            return None
        return {
            'id': episode_id,
            'index': index,
            'frame_count': _safe_nonnegative_int(value.get('frame_count')) or 0,
            'elapsed_ms': _safe_nonnegative_number(value.get('elapsed_ms')) or 0,
            'first_timestamp_ns': _safe_nonnegative_int(value.get('first_timestamp_ns')),
            'latest_timestamp_ns': _safe_nonnegative_int(value.get('latest_timestamp_ns')),
            'checkpoint_revision': _safe_nonnegative_int(
                value.get('checkpoint_revision')
            )
            or 0,
            'untagged_start_frame': _safe_nonnegative_int(
                value.get('untagged_start_frame')
            )
            or 0,
        }

    @staticmethod
    def _sanitize_segments(value: Any) -> list[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result: list[Dict[str, Any]] = []
        for item in value[:_MAX_TIMELINE_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            tag = item.get('tag', item.get('label'))
            start = _safe_nonnegative_int(item.get('start_frame'))
            end = _safe_nonnegative_int(item.get('end_frame'))
            checkpoint_id = _safe_token(item.get('checkpoint_id'))
            previous_id = _safe_token(item.get('previous_checkpoint_id'))
            if tag not in {'accept', 'reject'} or start is None or end is None or end < start:
                continue
            revision = _safe_nonnegative_int(item.get('checkpoint_revision'))
            if checkpoint_id is None and revision is not None:
                checkpoint_id = f'checkpoint-{revision}'
            if previous_id is None and revision is not None and revision > 0:
                previous_id = f'checkpoint-{revision - 1}'
            result.append(
                {
                    'checkpoint_id': checkpoint_id,
                    'previous_checkpoint_id': previous_id,
                    'start_frame': start,
                    'end_frame': end,
                    'tag': tag,
                }
            )
        return result

    @staticmethod
    def _sanitize_checkpoints(value: Any) -> list[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result: list[Dict[str, Any]] = []
        for item in value[:_MAX_TIMELINE_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            revision = _safe_nonnegative_int(item.get('revision'))
            checkpoint_id = _safe_token(item.get('id'))
            if checkpoint_id is None and revision is not None:
                checkpoint_id = f'checkpoint-{revision}'
            index = _safe_nonnegative_int(item.get('index'))
            if index is None:
                index = revision
            frame = _safe_nonnegative_int(item.get('frame'))
            if checkpoint_id is None or index is None or frame is None:
                continue
            result.append(
                {
                    'id': checkpoint_id,
                    'index': index,
                    'frame': frame,
                    'timestamp_ns': _safe_nonnegative_int(item.get('timestamp_ns')),
                    'elapsed_ms': _safe_nonnegative_number(item.get('elapsed_ms')) or 0,
                }
            )
        return result

    def _read_status(self) -> Optional[Dict[str, Any]]:
        raw = self._secure_read_json(self.status_dir / 'recorder.json')
        if raw is None:
            return None
        self._validate_owner(raw)
        if raw.get('schema_version') != 1:
            raise RecordingSecurityError('Unsupported recording status schema')
        status = raw.get('status')
        if status not in _STATUS_VALUES:
            raise RecordingSecurityError('Unsupported recording status value')
        revision = _safe_nonnegative_int(raw.get('revision')) or 0
        ready = raw.get('ready') is True
        active = raw.get('active') is True
        return {
            'configured': self.configured,
            'enabled': self.enabled,
            'required': self.required,
            'status': status,
            'active': active,
            'recorder_ready': ready,
            'error': _safe_error(raw.get('error')),
            'run_id': self.run_id,
            'revision': revision,
            'reference': self._sanitize_reference(raw.get('reference')),
            'episode': self._sanitize_episode(raw.get('episode')),
            'segments': self._sanitize_segments(raw.get('segments')),
            'checkpoints': self._sanitize_checkpoints(raw.get('checkpoints')),
        }

    def _read_ack(self, command: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        command_id = str(command['command_id'])
        raw = self._secure_read_json(self.acks_dir / f'{command_id}.json')
        if raw is None:
            return None
        self._validate_owner(raw)
        if raw.get('schema_version') != 1:
            raise RecordingSecurityError('Unsupported recording ACK schema')
        for field in ('command_id', 'idempotency_key', 'action'):
            if raw.get(field) != command.get(field):
                raise RecordingSecurityError(f'Recording ACK {field} does not match')
        ack_status = raw.get('status')
        if ack_status not in _ACK_STATUS_VALUES:
            raise RecordingSecurityError('Unsupported recording ACK status')
        ack = {
            'command_id': command_id,
            'idempotency_key': str(command['idempotency_key']),
            'action': str(command['action']),
            'status': ack_status,
            'episode_id': _safe_token(raw.get('episode_id')),
            'revision': _safe_nonnegative_int(raw.get('revision')),
            'error': _safe_error(raw.get('error')),
        }
        finalized_source = raw.get('finalized_source')
        if isinstance(finalized_source, Mapping):
            ack['finalized_source'] = dict(finalized_source)
        return ack

    def _refresh_acks(self) -> None:
        for command_id, command in list(self._commands.items()):
            if command_id in self._acks:
                continue
            ack = self._read_ack(command)
            if ack is not None:
                self._acks[command_id] = ack
                self._notify_conversion(command, ack)

    def _refresh_finalized_receipts(self) -> None:
        """Discover finalized recordings independently of user-decision ACKs."""
        finalized_dir = self._required_control_dir() / 'finalized'
        if not finalized_dir.is_dir() or finalized_dir.is_symlink():
            return
        for path in sorted(finalized_dir.glob('*.json')):
            receipt = self._finalized_receipt_cache.get(path.name)
            if receipt is None:
                try:
                    receipt = self._secure_read_json(path)
                except RecordingSecurityError:
                    continue
                if isinstance(receipt, dict):
                    self._finalized_receipt_cache[path.name] = receipt
            if not isinstance(receipt, Mapping):
                continue
            fingerprint = receipt.get('fingerprint')
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                continue
            if fingerprint not in self._conversion_notified and self._conversion_enqueue is not None:
                try:
                    self._conversion_enqueue(receipt)
                except Exception:
                    # Leave it undiscovered so the next state sample retries.
                    pass
                else:
                    self._conversion_notified.add(fingerprint)
            if fingerprint not in self._finalized_notified and self._finalized_enqueue is not None:
                try:
                    self._finalized_enqueue(receipt)
                except Exception:
                    continue
                self._finalized_notified.add(fingerprint)

    def _notify_conversion(
        self,
        command: Mapping[str, Any],
        ack: Mapping[str, Any],
    ) -> None:
        # Conversion scheduling is driven by finalized receipts, not by the
        # later accept/reject/home-transition ACK. Keep this hook for callers
        # that refresh ACK state explicitly, then scan the durable handoff.
        self._refresh_finalized_receipts()

    def _pending_command(self) -> Optional[Dict[str, Any]]:
        for command_id, command in self._commands.items():
            if command_id not in self._acks:
                return {
                    'command_id': command_id,
                    'action': command['action'],
                    'status': 'pending',
                    'created_at': command.get('created_at'),
                }
        return None

    def _last_command(self) -> Optional[Dict[str, Any]]:
        if not self._commands:
            return None
        command = next(reversed(self._commands.values()))
        return self._command_public(command)

    def _capabilities(self, state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        common: list[str] = []
        if not state.get('recorder_ready'):
            common.append('录制器尚未就绪')
        if state.get('status') != 'recording' or not state.get('active'):
            common.append('当前没有正在录制的 episode')
        episode = state.get('episode')
        if not isinstance(episode, Mapping):
            common.append('录制器尚未发布 episode 标识')
        pending = state.get('pending_command')
        if pending:
            common.append('已有录制操作正在处理')
        if state.get('error'):
            common.append(str(state['error']))
        common = list(dict.fromkeys(common))

        capabilities: Dict[str, Dict[str, Any]] = {}
        for action in sorted(RECORDING_ACTIONS):
            blockers = list(common)
            if action.startswith('segment_') and isinstance(episode, Mapping):
                frame_count = int(episode.get('frame_count') or 0)
                untagged = int(episode.get('untagged_start_frame') or 0)
                if frame_count <= untagged:
                    blockers.append('上一个 checkpoint 之后还没有新帧')
                if state.get('reference') is None:
                    blockers.append('录制器未发布权威 reference stream')
            capabilities[action] = {
                'can': not blockers,
                'blockers': list(dict.fromkeys(blockers)),
            }
        return capabilities

    def state(self) -> Dict[str, Any]:
        if self.control_dir is None:
            return empty_recording_state(
                configured=self.configured,
                enabled=self.enabled,
                required=self.required,
            )
        try:
            self._refresh_finalized_receipts()
            self._refresh_acks()
            state = self._read_status()
        except RecordingSecurityError as exc:
            state = empty_recording_state(
                configured=self.configured,
                enabled=self.enabled,
                required=self.required,
                error=str(exc),
            )
            state['run_id'] = self.run_id
        if state is None:
            state = empty_recording_state(
                configured=self.configured,
                enabled=self.enabled,
                required=self.required,
            )
            state.update({'status': 'starting', 'run_id': self.run_id})
        state['pending_command'] = self._pending_command()
        state['last_command'] = self._last_command()
        state['capabilities'] = self._capabilities(state)
        return state

    @staticmethod
    def is_start_ready(state: Mapping[str, Any]) -> bool:
        return bool(
            state.get('recorder_ready') is True
            and state.get('status') in _READY_START_STATUSES
        )

    @staticmethod
    def is_healthy(state: Mapping[str, Any]) -> bool:
        status = state.get('status')
        return bool(
            status in _HEALTHY_STATUSES
            and not state.get('error')
            and (
                status in {'finalizing', 'transitioning'}
                or state.get('recorder_ready') is True
            )
        )

    def wait_until_ready(
        self, process: Any, timeout: float, poll_interval: float = 0.05
    ) -> Tuple[bool, str]:
        deadline = self._clock() + max(0.0, float(timeout))
        last_error = '录制器未在超时前进入 armed/recording 状态'
        while self._clock() < deadline:
            try:
                code = process.poll()
            except (AttributeError, OSError):
                code = None
            if code is not None:
                return False, f'推理进程在录制器就绪前退出 (code {code})'
            state = self.state()
            if self.is_start_ready(state):
                return True, ''
            if state.get('error'):
                return False, str(state['error'])
            if state.get('status') in {'error', 'incomplete', 'stopped'}:
                return False, f"录制器启动失败：{state.get('status')}"
            self._sleep(poll_interval)
        return False, last_error

    def _find_idempotent(
        self, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        for command in self._commands.values():
            if command.get('idempotency_key') == idempotency_key:
                return command
        return None

    def _command_public(self, command: Mapping[str, Any]) -> Dict[str, Any]:
        command_id = str(command['command_id'])
        ack = self._acks.get(command_id)
        if ack is not None:
            return dict(ack)
        return {
            'command_id': command_id,
            'idempotency_key': command['idempotency_key'],
            'action': command['action'],
            'status': 'pending',
            'episode_id': command.get('episode_id'),
            'revision': command.get('expected_revision'),
            'expected_checkpoint_revision': command.get(
                'expected_checkpoint_revision'
            ),
            'error': None,
        }

    def _write_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        command_id = str(command['command_id'])
        self._atomic_write_json(
            self.commands_dir / f'{command_id}.json', command
        )
        self._commands[command_id] = command
        return self._command_public(command)

    def submit_action(
        self,
        action: str,
        *,
        episode_id: str,
        expected_checkpoint_revision: int,
        idempotency_key: str,
        home_target: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if action not in RECORDING_ACTIONS:
            raise RecordingRequestError(f'Unknown recording action: {action}')
        safe_episode = _safe_token(episode_id)
        safe_idempotency = _safe_token(idempotency_key)
        if safe_episode is None:
            raise RecordingRequestError('episode_id is invalid')
        if safe_idempotency is None:
            raise RecordingRequestError('idempotency_key is invalid')
        if (
            isinstance(expected_checkpoint_revision, bool)
            or not isinstance(expected_checkpoint_revision, int)
            or expected_checkpoint_revision < 0
        ):
            raise RecordingRequestError(
                'expected_checkpoint_revision must be a non-negative integer'
            )

        try:
            normalized_home_target = normalize_home_target(home_target)
        except HomeTransitionError as exc:
            raise RecordingRequestError(str(exc)) from exc
        if normalized_home_target is not None and action not in {
            'episode_accept',
            'episode_discard',
        }:
            raise RecordingRequestError(
                'home_target is valid only for whole-episode actions'
            )

        existing = self._find_idempotent(safe_idempotency)
        if existing is not None:
            same = (
                existing.get('action') == action
                and existing.get('episode_id') == safe_episode
                and existing.get('expected_checkpoint_revision')
                == expected_checkpoint_revision
                and existing.get('home_target') == normalized_home_target
            )
            if not same:
                raise RecordingConflictError(
                    'idempotency_key was already used for another recording action'
                )
            self._refresh_acks()
            return self._command_public(existing)

        state = self.state()
        episode = state.get('episode')
        if not isinstance(episode, Mapping) or episode.get('id') != safe_episode:
            raise RecordingConflictError('Recording episode changed; refresh state')
        current_revision = int(episode.get('checkpoint_revision') or 0)
        if current_revision != expected_checkpoint_revision:
            raise RecordingConflictError(
                'Recording checkpoint revision changed; refresh state'
            )
        capability = state['capabilities'][action]
        if not capability['can']:
            raise RecordingConflictError('; '.join(capability['blockers']))

        command = {
            'schema_version': 1,
            'run_id': self.run_id,
            'command_id': str(uuid.uuid4()),
            'idempotency_key': safe_idempotency,
            'action': action,
            'episode_id': safe_episode,
            'expected_revision': int(state.get('revision') or 0),
            'expected_checkpoint_revision': expected_checkpoint_revision,
            'created_at': _utc_timestamp(),
        }
        if normalized_home_target is not None:
            command['home_target'] = normalized_home_target
        return self._write_command(command)

    def request_shutdown(self, timeout: float) -> Tuple[bool, Optional[str]]:
        if self.control_dir is None:
            return True, None
        state = self.state()
        if state.get('status') in {'stopped', 'incomplete'} and not state.get('active'):
            return True, None
        idempotency_key = f'shutdown:{self.run_id}'
        existing = self._find_idempotent(idempotency_key)
        if existing is None:
            episode = state.get('episode')
            command = {
                'schema_version': 1,
                'run_id': self.run_id,
                'command_id': str(uuid.uuid4()),
                'idempotency_key': idempotency_key,
                'action': 'shutdown',
                'episode_id': episode.get('id') if isinstance(episode, Mapping) else None,
                # Shutdown is an internal safety transition. It must not be
                # rejected merely because an earlier queued label bumped the
                # optimistic UI revision before this command is consumed.
                'expected_revision': None,
                'created_at': _utc_timestamp(),
            }
            self._write_command(command)
        else:
            command = existing

        deadline = self._clock() + max(0.0, float(timeout))
        while self._clock() < deadline:
            self._refresh_acks()
            ack = self._acks.get(str(command['command_id']))
            if ack is not None:
                if ack['status'] == 'applied':
                    return True, None
                return False, ack.get('error') or f"recording shutdown {ack['status']}"
            current = self.state()
            if current.get('status') in {'stopped', 'incomplete'} and not current.get('active'):
                return True, None
            self._sleep(0.05)
        return False, '录制器未在超时前完成 flush/quarantine'


__all__ = [
    'DEFAULT_RECORDING_RUN_ROOT',
    'RECORDING_ACTIONS',
    'RecordingConflictError',
    'RecordingCoordinator',
    'RecordingError',
    'RecordingRequestError',
    'RecordingSecurityError',
    'default_recording_profile_check',
    'empty_recording_state',
    'recording_config_state',
    'recording_timeout',
    'resolve_recording_python',
]
