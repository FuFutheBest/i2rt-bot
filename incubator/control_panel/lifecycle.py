"""Isolated, per-device process lifecycle for the web control panel.

This module deliberately does not extend or modify the repository's legacy
``device_starter.py``.  Each device is launched with the ``spawn`` start method
and must atomically claim every shared-memory segment it creates.
"""

import os
import sys
import time
import signal
import copy
import inspect
import logging
import multiprocessing as mp
import threading
import uuid
from typing import Dict, List, Any, Mapping, Optional, Tuple
from pathlib import Path

from utils.cpu_affinity import CpuAffinitySettings

from .shm_protocol import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    unpack_buffer_header,
    unpack_device_header,
)
from .runtime_config import normalize_runtime_config
from .flexiv_rdk import (
    WORKER_RDK_SITE_KEY,
    WORKER_RDK_VERSION_KEY,
    FlexivRDKError,
    FlexivRDKProbeError,
    FlexivRDKResolver,
    activate_flexiv_rdk_worker,
)
from .hardware_discovery import capture_binding_is_missing

logger = logging.getLogger(__name__)


def _parse_device_configs_with_fields(
    config: Dict[str, Any], additional_fields: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Parse enabled devices without logging resolved config values."""
    section = config.get('devices', [])
    if isinstance(section, dict):
        section = section.get('devices', [])
    if not isinstance(section, list):
        raise DeviceControlError('devices must resolve to a list')

    devices: List[Dict[str, Any]] = []
    for index, raw in enumerate(section):
        if not isinstance(raw, dict):
            raise DeviceControlError('each device config must be a mapping')
        if raw.get('enabled', True) is False:
            continue
        parsed_config = _parse_device_control_contract(
            raw, context=f'devices[{index}]'
        )
        device = {
            'device_class': parsed_config['class'],
            'device_class_id': parsed_config['device_class_id'],
            'config': parsed_config,
        }
        device.update(copy.deepcopy(additional_fields))
        devices.append(device)
    return devices


class DeviceControlError(RuntimeError):
    """Base exception for an invalid per-device lifecycle operation."""


class DeviceNotFoundError(DeviceControlError):
    """Raised when a configured device key does not exist."""


class DeviceConflictError(DeviceControlError):
    """Raised when a lifecycle operation would conflict with current state."""


class DeviceStartupError(DeviceControlError):
    """Raised when a device explicitly reports an unrecoverable start failure."""


def _parse_device_control_contract(
    device_config: Mapping[str, Any],
    *,
    context: str,
    for_constructor: bool = False,
) -> Dict[str, Any]:
    """Validate a device contract and prepare typed constructor values on demand.

    The control-panel lifecycle keeps a small compatibility surface for
    device-only inspection fixtures, so it does not require every legacy
    display field to be present here.  Once a device declares ``control``,
    however, it must satisfy the runtime's complete structured control
    contract.  The spawned worker requests ``for_constructor`` before it
    filters kwargs, so constructor-visible state/control values are typed while
    lifecycle status continues to retain the original inspectable mapping.
    """
    from devices.robots.config import RobotControlConfig
    from utils.device_config import (
        DeviceConfigError,
        parse_device_config,
        reject_removed_device_fields,
    )

    parsed = copy.deepcopy(dict(device_config))
    try:
        reject_removed_device_fields(parsed, field=context)
    except DeviceConfigError as exc:
        raise DeviceControlError(str(exc)) from exc
    if "state_feedback" in parsed:
        try:
            from devices import DEVICE_CLASSES

            typed = parse_device_config(
                parsed,
                device_kind=DEVICE_CLASSES.kind(str(parsed.get("class", ""))),
            )
        except (DeviceConfigError, KeyError, ValueError) as exc:
            raise DeviceControlError(f"{context}: {exc}") from exc
        if not for_constructor:
            return parsed
        structured = typed.constructor_kwargs()
        structured["class"] = typed.device_class
        return structured

    if "control" not in parsed:
        return parsed
    try:
        control = RobotControlConfig.require(parsed["control"])
    except (TypeError, ValueError) as exc:
        raise DeviceControlError(f"{context}.control: {exc}") from exc
    if for_constructor:
        parsed["control"] = control
    return parsed


class _DeviceWorkerStopRequested(BaseException):
    """Unwind a worker after SIGINT/SIGTERM without re-entering device locks."""


# These globals only carry state inside a spawned device worker.  The web
# controller itself never replaces the shared-memory helpers.
_WORKER_OWNERSHIP_CONN: Any = None
_WORKER_OWNER_TOKEN: Optional[str] = None
_WORKER_OWNED_SHARED_MEMORY: List[Dict[str, Any]] = []
_WORKER_ORIGINAL_SHARED_MEMORY: Any = None
_WORKER_CLAIM_CONFLICT: Optional[str] = None
_WORKER_STOP_REQUESTED = False


def _send_ownership_event(event: str, **details: Any) -> None:
    """Best-effort lifecycle event from a spawned worker to its owner."""
    connection = _WORKER_OWNERSHIP_CONN
    if connection is None:
        return
    payload = {
        'event': event,
        'owner_token': _WORKER_OWNER_TOKEN,
        'timestamp': time.time(),
    }
    payload.update(details)
    try:
        connection.send(payload)
    except (BrokenPipeError, EOFError, OSError):
        # The worker must remain independently stoppable if the controller dies.
        pass


def _device_start_failure(device: Any) -> Dict[str, Any]:
    """Return an allowlisted public startup diagnostic from a device."""
    default = {
        'code': 'start_server_returned_false',
        'recoverable': False,
        'message': 'Device start_server returned False',
    }
    provider = getattr(device, 'startup_failure', None)
    if not callable(provider):
        return default
    try:
        failure = provider()
    except Exception:
        return default
    if not isinstance(failure, dict):
        return default
    code = str(failure.get('code') or default['code']).strip().lower()
    if not code or any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for char in code):
        code = default['code']
    message = ' '.join(
        str(failure.get('message') or default['message']).split()
    )[:512]
    return {
        'code': code,
        'recoverable': failure.get('recoverable') is True,
        'message': message or default['message'],
    }


def _shared_memory_identity(name: str) -> Optional[Tuple[int, int]]:
    """Return the POSIX object identity without attaching as a consumer."""
    if os.name != 'posix':
        return None
    try:
        stat_result = (Path('/dev/shm') / str(name).lstrip('/')).stat()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return int(stat_result.st_dev), int(stat_result.st_ino)


def _detach_resource_tracker(claim: Dict[str, Any]) -> None:
    """Detach an unlinked/orphaned claim so tracker cannot unlink by name."""
    if claim.get('unlinked') or claim.get('tracker_detached'):
        return
    try:
        from multiprocessing import resource_tracker

        tracked_name = getattr(claim['raw_handle'], '_name', None)
        if tracked_name:
            resource_tracker.unregister(tracked_name, 'shared_memory')
            claim['tracker_detached'] = True
    except (AttributeError, KeyError, OSError, RuntimeError, ValueError):
        pass


class _OwnedSharedMemoryChannel:
    """Device-compatible channel that preserves creator unlink authority."""

    def __init__(
        self,
        raw_handle: Any,
        claim: Dict[str, Any],
        *,
        read_only: bool = False,
    ) -> None:
        self._shm = raw_handle
        self._claim = claim
        self._read_only = bool(read_only)

    @property
    def buf(self) -> memoryview:
        buffer = self._shm.buf
        return buffer.toreadonly() if self._read_only else buffer

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def size(self) -> int:
        return self._shm.size

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    def as_read_only(self) -> '_OwnedSharedMemoryChannel':
        self._read_only = True
        return self

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        """Normal device cleanup, guarded against an already-replaced name."""
        if self._claim.get('unlinked'):
            raise FileNotFoundError(self.name)
        claimed = (int(self._claim['dev']), int(self._claim['ino']))
        current = _shared_memory_identity(self.name)
        if current != claimed:
            _detach_resource_tracker(self._claim)
            self._claim['cleanup_required'] = current == claimed
            _send_ownership_event(
                'owner_unlink_skipped',
                name=self.name,
                reason='missing' if current is None else 'replaced',
            )
            return
        self._shm.unlink()
        self._claim['unlinked'] = True
        _send_ownership_event('cleaned', name=self.name)


class _DirectSharedMemoryConflict(BaseException):
    """Escape legacy `except Exception` blocks after an external attach."""


class _ProtectedAttachedSharedMemory:
    """Direct-create preflight handle that cannot unlink another producer."""

    def __init__(self, raw_handle: Any) -> None:
        self._shm = raw_handle

    @property
    def buf(self) -> memoryview:
        return self._shm.buf

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def size(self) -> int:
        return self._shm.size

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        global _WORKER_CLAIM_CONFLICT

        _WORKER_CLAIM_CONFLICT = self.name
        message = f'Control-panel worker cannot unlink attached SHM {self.name}'
        _send_ownership_event('conflict', name=self.name, error=message)
        raise _DirectSharedMemoryConflict(message)


def _guarded_direct_shared_memory(
    name: Optional[str] = None,
    create: bool = False,
    size: int = 0,
    **kwargs: Any,
) -> Any:
    """Guard device modules that call `SharedMemory` directly.

    `create=True` joins the normal ownership registry. `create=False` remains a
    readable attachment, but its unlink operation is blocked and its consumer
    registration is detached from the worker resource tracker.
    """
    del kwargs
    native = _WORKER_ORIGINAL_SHARED_MEMORY
    if native is None:
        from multiprocessing import shared_memory

        native = shared_memory.SharedMemory
    if create:
        if name is None:
            raise ValueError('named shared memory is required')
        try:
            return _exclusive_worker_create_shared_memory(
                name,
                size,
                rollback_on_conflict=False,
            )
        except FileExistsError as exc:
            raise _DirectSharedMemoryConflict(str(exc)) from exc

    raw_handle = native(name=name, create=False, size=int(size))
    normalized = str(raw_handle.name)
    already_owned = any(
        str(claim['name']) == normalized and not claim.get('unlinked')
        for claim in _WORKER_OWNED_SHARED_MEMORY
    )
    if not already_owned:
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(raw_handle._name, 'shared_memory')
        except (AttributeError, KeyError, OSError, RuntimeError, ValueError):
            raw_handle.close()
            raise
    return _ProtectedAttachedSharedMemory(raw_handle)


def _worker_make_readonly(channel: Any) -> Any:
    """Keep ownership when Robot converts its control channel read-only."""
    if isinstance(channel, _OwnedSharedMemoryChannel):
        return channel.as_read_only()
    from utils import shm_utils

    original = getattr(shm_utils, '_control_panel_original_make_readonly', None)
    if original is None:
        raise TypeError('unsupported shared-memory channel')
    return original(channel)


def _cleanup_worker_owned_shared_memory() -> None:
    """Close registry handles but never perform a fallback name unlink.

    Healthy devices call their owner channel's ``close`` + ``unlink``. If that
    cleanup did not happen, safety wins over convenience: detach Python's
    resource tracker, report the segment as cleanup-required, and leave its
    deterministic name in place so status blocks a restart.
    """
    while _WORKER_OWNED_SHARED_MEMORY:
        claim = _WORKER_OWNED_SHARED_MEMORY.pop()
        name = str(claim['name'])
        raw_handle = claim['raw_handle']
        claimed_identity = (int(claim['dev']), int(claim['ino']))
        current_identity = _shared_memory_identity(name)
        if not claim.get('unlinked'):
            _detach_resource_tracker(claim)
            event = (
                'cleanup_required'
                if current_identity == claimed_identity
                else 'cleanup_skipped_replaced'
            )
            _send_ownership_event(event, name=name)
        try:
            raw_handle.close()
        except (BufferError, FileNotFoundError, OSError):
            pass


def _cleanup_initialization_claims() -> None:
    """Best-effort rollback while a device constructor is still unwinding."""
    while _WORKER_OWNED_SHARED_MEMORY:
        claim = _WORKER_OWNED_SHARED_MEMORY.pop()
        raw_handle = claim['raw_handle']
        claimed = (int(claim['dev']), int(claim['ino']))
        current = _shared_memory_identity(str(claim['name']))
        if not claim.get('unlinked') and current == claimed:
            try:
                raw_handle.unlink()
                claim['unlinked'] = True
                _send_ownership_event(
                    'initialization_claim_rolled_back', name=claim['name']
                )
            except FileNotFoundError:
                claim['unlinked'] = True
            except (OSError, PermissionError):
                _detach_resource_tracker(claim)
        elif not claim.get('unlinked'):
            _detach_resource_tracker(claim)
        try:
            raw_handle.close()
        except (BufferError, FileNotFoundError, OSError):
            pass


def _exclusive_worker_create_shared_memory(
    name: str,
    size: int,
    readonly_after_creation: bool = False,
    *,
    rollback_on_conflict: bool = True,
) -> Any:
    """Atomically create worker-owned SHM without deleting another producer's.

    ``utils.shm_utils.create_shared_memory`` historically unlinks an existing
    segment before creating its own.  That is useful for the original CLI but
    unsafe after a web request's preflight probe because another producer can
    win the race.  A spawned control-panel worker temporarily installs this
    implementation in its private interpreter.  ``create=True`` is the actual
    ownership claim: an existing name raises ``FileExistsError`` and is never
    unlinked.
    """
    global _WORKER_CLAIM_CONFLICT

    from multiprocessing import shared_memory

    normalized_name = str(name)
    _send_ownership_event('claiming', name=normalized_name, size=int(size))
    try:
        native = _WORKER_ORIGINAL_SHARED_MEMORY or shared_memory.SharedMemory
        shm_obj = native(
            create=True,
            size=int(size),
            name=normalized_name,
        )
    except FileExistsError:
        _WORKER_CLAIM_CONFLICT = normalized_name
        _send_ownership_event(
            'conflict',
            name=normalized_name,
            error=f"Shared memory {normalized_name} already exists",
        )
        # If this was a secondary/control claim, roll back any primary SHM
        # created moments earlier by the same still-initializing worker.
        if rollback_on_conflict:
            _cleanup_initialization_claims()
        raise
    except Exception as exc:
        _send_ownership_event(
            'claim_error',
            name=normalized_name,
            error=str(exc),
        )
        raise

    identity = _shared_memory_identity(normalized_name)
    if identity is None:
        # Atomic creation succeeded, but without an identity we cannot safely
        # perform name-based fallback cleanup later.
        try:
            shm_obj.close()
        finally:
            shm_obj.unlink()
        error = f"Cannot verify shared-memory identity for {normalized_name}"
        _send_ownership_event('claim_error', name=normalized_name, error=error)
        raise RuntimeError(error)

    claim = {
        'name': normalized_name,
        'raw_handle': shm_obj,
        'dev': identity[0],
        'ino': identity[1],
        'unlinked': False,
        'tracker_detached': False,
    }
    _WORKER_OWNED_SHARED_MEMORY.append(claim)

    channel = _OwnedSharedMemoryChannel(
        shm_obj,
        claim,
        read_only=bool(readonly_after_creation),
    )

    _send_ownership_event(
        'claimed',
        name=normalized_name,
        size=int(size),
        dev=identity[0],
        ino=identity[1],
    )
    return channel


def _worker_primary_shm_name(
    device_class_name: str, device_config: Dict[str, Any]
) -> str:
    device_name = device_config.get('device_name') or device_class_name
    return f"{device_name}_{device_config.get('device_class_id')}_data"


def _worker_shm_preflight(name: str) -> bool:
    """Non-owning Linux preflight; atomic exclusion happens during creation."""
    if os.name != 'posix':
        return False
    return (Path('/dev/shm') / str(name).lstrip('/')).exists()


def _install_rizon_health_reporter(device: Any, device_class_name: str) -> None:
    """Report passive Flexiv safety state from the existing device thread.

    RDK handles are intentionally queried from the worker thread that already
    owns them, rather than from Flask or a second SDK connection.  The existing
    per-frame diagnostic hook is throttled to 5 Hz and emits only state changes
    plus a one-second heartbeat, so the probe is cheaper than its former
    unthrottled ``operational_status`` call.
    """
    if str(device_class_name) != 'RizonRobot':
        return
    original = getattr(device, '_check_operational_status_change', None)
    if not callable(original):
        return

    last_probe = 0.0
    last_emit = 0.0
    last_signature: Any = None

    def report() -> None:
        nonlocal last_emit, last_probe, last_signature
        now = time.monotonic()
        if now - last_probe < 0.2:
            return
        last_probe = now
        try:
            original()
        except BaseException:
            # Health reporting must never break the producer's state loop.
            pass

        if getattr(device, 'is_dual_arm', False):
            handles = (
                ('left', getattr(device, 'robot_left', None)),
                ('right', getattr(device, 'robot_right', None)),
            )
        else:
            handles = (('single', getattr(device, 'robot', None)),)

        arms: Dict[str, Dict[str, Any]] = {}
        for side, handle in handles:
            if handle is None:
                arms[side] = {
                    'known': False,
                    'operational': None,
                    'fault': None,
                    'estop': None,
                    'operational_status': None,
                    'reason': 'Robot handle is not available',
                }
                continue
            try:
                status_method = getattr(handle, 'operational_status', None)
                status_value = status_method() if callable(status_method) else None
                status_name = str(
                    getattr(status_value, 'name', status_value)
                    if status_value is not None
                    else 'UNKNOWN'
                ).upper()
                fault_method = getattr(handle, 'fault', None)
                fault_raw = fault_method() if callable(fault_method) else None
                fault = bool(fault_raw) if fault_raw is not None else None
                operational_method = getattr(handle, 'is_operational', None)
                if not callable(operational_method):
                    operational_method = getattr(handle, 'operational', None)
                operational_raw = (
                    operational_method() if callable(operational_method) else None
                )
                operational = (
                    bool(operational_raw) if operational_raw is not None else None
                )
                estop = any(
                    token in status_name
                    for token in ('EMERGENCY_STOP', 'EMERGENCY', 'E_STOP', 'ESTOP')
                )
                reduced = 'REDUCED' in status_name
                status_known = bool(
                    status_name
                    and status_name not in {'UNKNOWN', 'NONE', 'NULL'}
                    and 'UNRECOGNIZED' not in status_name
                )
                reasons = []
                if estop:
                    reasons.append('Emergency stop is active')
                if fault is True:
                    reasons.append('Robot fault is active')
                if operational is False:
                    reasons.append('Robot is not operational')
                if reduced:
                    reasons.append('Robot entered reduced safety state')
                complete = bool(
                    callable(status_method)
                    and status_value is not None
                    and status_known
                    and fault_raw is not None
                    and operational_raw is not None
                )
                arms[side] = {
                    'known': complete,
                    'operational': operational,
                    'fault': fault,
                    'estop': estop if complete else None,
                    'operational_status': status_name,
                    'reduced': reduced,
                    'reason': (
                        '; '.join(reasons)
                        or (None if complete else 'Robot health evidence is incomplete')
                    ),
                }
            except BaseException as exc:
                arms[side] = {
                    'known': False,
                    'operational': None,
                    'fault': None,
                    'estop': None,
                    'operational_status': None,
                    'reason': f'Robot health probe failed: {type(exc).__name__}',
                }

        known = bool(arms) and all(item['known'] for item in arms.values())
        estop_values = [item['estop'] for item in arms.values()]
        fault_values = [item['fault'] for item in arms.values()]
        operational_values = [item['operational'] for item in arms.values()]
        estop = True if any(value is True for value in estop_values) else (
            False if known else None
        )
        fault = True if any(value is True for value in fault_values) else (
            False if known else None
        )
        operational = (
            all(value is True for value in operational_values) if known else None
        )
        reduced = any(item.get('reduced') is True for item in arms.values())
        fail_closed = bool(
            not known
            or estop is True
            or fault is True
            or operational is False
            or reduced
        )
        reasons = [str(item['reason']) for item in arms.values() if item.get('reason')]
        payload = {
            'source': 'flexiv_worker_passive_probe',
            'known': known,
            'operational': operational,
            'fault': fault,
            'estop': estop,
            'reduced': reduced,
            'fail_closed': fail_closed,
            'reason': '; '.join(reasons) or None,
            'arms': arms,
        }
        signature = repr(payload)
        if signature == last_signature and now - last_emit < 1.0:
            return
        last_signature = signature
        last_emit = now
        _send_ownership_event(
            'device_health',
            pid=os.getpid(),
            generation=_WORKER_OWNER_TOKEN,
            health=payload,
        )

    # The device calls this hook from its own state-read loop.  Assigning a
    # zero-argument closure on the instance avoids cross-thread RDK access.
    try:
        device._check_operational_status_change = report
    except (AttributeError, TypeError):
        logger.warning('Unable to install passive Rizon health reporter')


def _run_device_worker(
    device_class_name: str,
    device_config: Dict[str, Any],
    owner_token: str,
    ownership_conn: Any,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Spawn-safe, module-level device entrypoint.

    Only primitive configuration data and a one-way Pipe are passed into this
    function.  In particular, no lifecycle-manager instance, class registry,
    lock, Flask request thread, or loguru object needs to be pickled.
    """
    device_config = _parse_device_control_contract(
        device_config,
        context=f"{device_class_name}:{device_config.get('device_class_id')}",
        for_constructor=True,
    )

    global _WORKER_CLAIM_CONFLICT, _WORKER_STOP_REQUESTED
    global _WORKER_ORIGINAL_SHARED_MEMORY
    global _WORKER_OWNERSHIP_CONN, _WORKER_OWNER_TOKEN

    _WORKER_OWNERSHIP_CONN = ownership_conn
    _WORKER_OWNER_TOKEN = owner_token
    _WORKER_OWNED_SHARED_MEMORY.clear()
    _WORKER_CLAIM_CONFLICT = None
    device = None
    _WORKER_STOP_REQUESTED = False
    original_create_shared_memory = None
    original_make_readonly = None
    original_direct_shared_memory = None
    primary_shm_name = _worker_primary_shm_name(
        device_class_name, device_config
    )

    def report_progress(phase: str) -> None:
        normalized = str(phase).strip().casefold()
        if (
            not normalized
            or len(normalized) > 64
            or not normalized.replace('_', '').isalnum()
        ):
            return
        _send_ownership_event(
            'device_progress',
            pid=os.getpid(),
            generation=owner_token,
            phase=normalized,
        )

    try:
        if runtime_config is not None:
            CpuAffinitySettings.from_runtime(runtime_config).apply(
                'device',
                device_class=device_class_name,
                device_class_id=int(device_config['device_class_id']),
            )
        report_progress('worker_bootstrap')
        _send_ownership_event(
            'worker_started',
            pid=os.getpid(),
            start_method=mp.get_start_method(),
        )
        _send_ownership_event('preflight', name=primary_shm_name)
        if _worker_shm_preflight(primary_shm_name):
            message = f"Shared memory {primary_shm_name} already exists"
            _send_ownership_event(
                'conflict', name=primary_shm_name, error=message
            )
            raise FileExistsError(message)
        _send_ownership_event('preflight_clear', name=primary_shm_name)

        # Patch only this spawn worker.  Convenience functions such as
        # create_device_shm/create_control_shm resolve this module global when
        # called, including aliases imported by BaseDevice and Robot classes.
        from utils import shm_utils
        from multiprocessing import shared_memory as direct_shared_memory

        original_create_shared_memory = shm_utils.create_shared_memory
        original_make_readonly = shm_utils.make_readonly
        original_direct_shared_memory = direct_shared_memory.SharedMemory
        _WORKER_ORIGINAL_SHARED_MEMORY = original_direct_shared_memory
        shm_utils._control_panel_original_make_readonly = original_make_readonly
        shm_utils.create_shared_memory = _exclusive_worker_create_shared_memory
        shm_utils.make_readonly = _worker_make_readonly

        if str(device_class_name) == 'RizonRobot':
            activate_flexiv_rdk_worker(device_config)

        from devices import DEVICE_CLASSES as worker_device_classes

        # Some shared-control implementations bypass shm_utils and call
        # multiprocessing.shared_memory.SharedMemory directly. Patch the module
        # object only in this spawned worker after lazy device import.
        direct_shared_memory.SharedMemory = _guarded_direct_shared_memory

        device_class = worker_device_classes.get(device_class_name)
        if not device_class:
            raise ValueError(f"Unknown device class: {device_class_name}")

        valid_params = list(inspect.signature(device_class.__init__).parameters)
        expanded_config = DeviceLifecycleManager._expand_device_config_sections(
            device_config, valid_params
        )
        expanded_config.pop('class', None)
        filtered_config = {
            key: value
            for key, value in expanded_config.items()
            if key in valid_params
        }
        report_progress('constructing_device')
        device = device_class(**filtered_config)
        device._control_panel_progress_reporter = report_progress
        _install_rizon_health_reporter(device, device_class_name)
        if _WORKER_CLAIM_CONFLICT:
            raise FileExistsError(
                f'Shared memory {_WORKER_CLAIM_CONFLICT} already exists'
            )

        def _stop_device(signum: int, frame: Any) -> None:
            global _WORKER_STOP_REQUESTED
            del frame
            if _WORKER_STOP_REQUESTED:
                return
            _WORKER_STOP_REQUESTED = True
            # Avoid re-entering hardware cleanup if a process-group SIGINT is
            # immediately followed by the controller's SIGTERM.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            _send_ownership_event('stop_requested', signal=signum)
            report_progress('stop_requested')
            logger.info(
                f"Stopping {device_class_name} "
                f"{device_config.get('device_class_id')} on signal {signum}"
            )
            # Never run device cleanup from the Python signal handler. The
            # control loop may currently hold a WebSocket/RDK lock, and
            # re-entering stop_server() from that same thread can deadlock the
            # worker forever. Raising a BaseException bypasses broad
            # ``except Exception`` startup handlers, unwinds their locks and
            # context managers, and reaches the single cleanup call in the
            # outer ``finally`` block below.
            raise _DeviceWorkerStopRequested()

        signal.signal(signal.SIGINT, _stop_device)
        signal.signal(signal.SIGTERM, _stop_device)

        logger.info(
            f"Starting {device_class_name} {device_config.get('device_class_id')}"
        )
        report_progress('initializing_device')
        start_result = device.start_server()
        if start_result is False:
            # Some device implementations report an initialization rejection by
            # returning False instead of raising.  Treat that as an explicit,
            # terminal failure so the monitor cannot blindly retry hardware.
            failure = _device_start_failure(device)
            report_progress('start_failed')
            _send_ownership_event(
                'start_failed',
                code=failure['code'],
                recoverable=failure['recoverable'],
                error=failure['message'],
            )
            raise DeviceStartupError(failure['message'])
    except _DeviceWorkerStopRequested:
        logger.info(
            f"{device_class_name} {device_config.get('device_class_id')} stop requested"
        )
    except BaseException as exc:
        # FileExistsError is also reported by the exclusive creation function;
        # the duplicate worker_error preserves diagnostics if failure occurred
        # during the earlier non-owning preflight instead.
        _send_ownership_event(
            'worker_error',
            error=str(exc),
            error_type=type(exc).__name__,
        )
        report_progress('worker_failed')
        logger.error(
            f"{device_class_name} {device_config.get('device_class_id')} error: {exc}"
        )
        raise
    finally:
        if device is not None:
            try:
                device.stop_server()
            except Exception as cleanup_error:
                logger.warning(
                    f"{device_class_name} cleanup error: {cleanup_error}"
                )
        if (
            device_class_name == 'YamHeadRobot'
            and device is not None
            and getattr(device, '_yam_robot', None) is not None
        ):
            _send_ownership_event(
                'stop_blocked',
                code='yam_zero_not_confirmed',
            )
            logger.error(
                'YamHeadRobot cleanup cannot exit before measured joint-zero '
                'is confirmed; retaining the i2rt connection'
            )
            _WORKER_STOP_REQUESTED = False
            signal.signal(signal.SIGINT, _stop_device)
            signal.signal(signal.SIGTERM, _stop_device)
            while getattr(device, '_yam_robot', None) is not None:
                time.sleep(0.1)
        if original_create_shared_memory is not None:
            try:
                from utils import shm_utils

                shm_utils.create_shared_memory = original_create_shared_memory
                if original_make_readonly is not None:
                    shm_utils.make_readonly = original_make_readonly
                if hasattr(shm_utils, '_control_panel_original_make_readonly'):
                    delattr(shm_utils, '_control_panel_original_make_readonly')
                if original_direct_shared_memory is not None:
                    from multiprocessing import shared_memory as direct_shared_memory

                    direct_shared_memory.SharedMemory = original_direct_shared_memory
            except Exception:
                pass
        # Device cleanup runs first. Registry finalization never unlinks by
        # name; any residue remains visible and blocks an unsafe restart.
        _cleanup_worker_owned_shared_memory()
        report_progress('stopped')
        _send_ownership_event('worker_stopped', pid=os.getpid())
        try:
            ownership_conn.close()
        except (AttributeError, OSError):
            pass
        _WORKER_OWNERSHIP_CONN = None
        _WORKER_OWNER_TOKEN = None
        _WORKER_ORIGINAL_SHARED_MEMORY = None
        _WORKER_CLAIM_CONFLICT = None
        _WORKER_STOP_REQUESTED = False


class DeviceLifecycleManager:
    """Own independently spawned device workers for one selected config."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        config_path: Optional[str] = None,
        enable_visualizers: bool = False,
        enable_device_manager: bool = False,
        enable_policy_runner: bool = False,
        install_signal_handlers: bool = False,
        flexiv_rdk_resolver: Optional[FlexivRDKResolver] = None,
    ) -> None:
        del config_path, install_signal_handlers
        if enable_visualizers or enable_device_manager or enable_policy_runner:
            raise DeviceControlError(
                'Control-panel v1 manages device producers only; legacy '
                'visualizers, DeviceManager, and policy runner are unsupported'
            )
        self.config = normalize_runtime_config(config)
        self.devices: List[Dict[str, Any]] = []
        self.running = False
        # Stable status fields retained for the browser.  Control-panel v1
        # never starts the legacy visualizer, DeviceManager, or policy runner.
        self.enable_visualizers = False
        self.enable_device_manager = False
        self.enable_policy_runner = False
        self.device_manager_process = None
        self.policy_runner_process = None
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_wakeup = threading.Event()
        # Web lifecycle calls run in Flask/monitor threads.  A dedicated spawn
        # context avoids forking a multi-threaded interpreter and the locks held
        # by imported camera/robot libraries.
        self._mp_context = mp.get_context('spawn')
        self._device_process_target = _run_device_worker
        self._flexiv_rdk_resolver = flexiv_rdk_resolver or FlexivRDKResolver(
            python=Path(
                os.environ.get('SUPERINFERENCE_DEVICE_PYTHON') or sys.executable
            )
        )

        # Parse once so per-device controls always operate on stable records.
        self.devices = self._parse_device_configs()
        keys = [self._device_key(info) for info in self.devices]
        if len(keys) != len(set(keys)):
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            raise ValueError(f"Duplicate device keys in configuration: {duplicates}")
        self._device_operation_locks = {
            key: threading.RLock() for key in keys
        }

        # Load configuration
        logger.info("Control-panel device lifecycle initialized")


    def _parse_device_configs(self) -> List[Dict[str, Any]]:
        """Parse device configurations and create device dictionaries."""
        # Use generic parser with additional fields for device starter
        additional_fields = {
            'process': None,
            'restart_count': 0,
            'last_restart': None,
            'last_exit_at': None,
            'last_error': None,
            'desired_state': 'stopped',
            'lifecycle_status': 'stopped',
            'started_at': None,
            'ownership_token': None,
            'ownership_status': 'unclaimed',
            'owned_shm_names': [],
            'ownership_conn': None,
            'ownership_child_conn': None,
            'failure_code': None,
            'failure_recoverable': None,
            'device_health': None,
            'device_health_received_at': None,
            'progress': {'phase': 'stopped', 'updated_at': None},
            'spawn_in_progress': False,
            'ownership_waiting': False,
            'flexiv_rdk_version': None,
            'flexiv_rdk_site': None,
            'flexiv_rdk_probe': None,
        }
        return _parse_device_configs_with_fields(self.config, additional_fields)

    @staticmethod
    def _device_key(device_info: Dict[str, Any]) -> str:
        return f"{device_info['device_class']}:{device_info['device_class_id']}"

    def _find_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        requested = (str(device_class), str(device_class_id))
        for device_info in self.devices:
            candidate = (
                str(device_info['device_class']),
                str(device_info['device_class_id']),
            )
            if candidate == requested:
                return device_info
        raise DeviceNotFoundError(f"Unknown device: {device_class}:{device_class_id}")

    def _device_operation_lock(
        self, device_info: Dict[str, Any]
    ) -> threading.RLock:
        return self._device_operation_locks[self._device_key(device_info)]

    def _prepare_rizon_rdk(self, device_info: Dict[str, Any]) -> None:
        """Resolve firmware compatibility before the Rizon worker imports RDK."""
        if str(device_info.get('device_class')) != 'RizonRobot':
            return
        with self._lock:
            device_info['flexiv_rdk_version'] = None
            device_info['flexiv_rdk_site'] = None
            device_info['progress'] = {
                'phase': 'rizon_selecting_rdk',
                'updated_at': time.time(),
            }
            device_info['flexiv_rdk_probe'] = {
                'status': 'probing',
                'started_at': time.time(),
                'attempts': [],
            }

        try:
            selection = self._flexiv_rdk_resolver.select(device_info['config'])
        except FlexivRDKError as exc:
            attempts = (
                [attempt.public_dict() for attempt in exc.attempts]
                if isinstance(exc, FlexivRDKProbeError)
                else []
            )
            with self._lock:
                device_info['config'].pop(WORKER_RDK_VERSION_KEY, None)
                device_info['config'].pop(WORKER_RDK_SITE_KEY, None)
                device_info['desired_state'] = 'stopped'
                device_info['lifecycle_status'] = 'failed'
                device_info['last_error'] = str(exc)
                device_info['failure_code'] = 'flexiv_rdk_probe_failed'
                device_info['failure_recoverable'] = True
                device_info['flexiv_rdk_probe'] = {
                    'status': 'failed',
                    'finished_at': time.time(),
                    'attempts': attempts,
                    'error': str(exc),
                }
            raise DeviceStartupError(str(exc)) from exc

        site = selection.candidate.site_packages
        with self._lock:
            device_info['config'][WORKER_RDK_VERSION_KEY] = (
                selection.candidate.version
            )
            device_info['config'][WORKER_RDK_SITE_KEY] = (
                str(site) if site is not None else None
            )
            device_info['flexiv_rdk_version'] = selection.candidate.version
            device_info['flexiv_rdk_site'] = str(site) if site is not None else None
            device_info['flexiv_rdk_probe'] = {
                'status': 'selected',
                'finished_at': time.time(),
                **selection.public_dict(),
            }

    @staticmethod
    def _process_is_alive(process: Any) -> bool:
        if process is None:
            return False
        try:
            return bool(process.is_alive())
        except (AssertionError, ValueError, OSError):
            return False

    @staticmethod
    def _expand_device_config_sections(device_config: Dict[str, Any], valid_params: List[str]) -> Dict[str, Any]:
        """Merge optional device sub-config sections into constructor kwargs."""
        config = device_config.copy()
        if 'tracking' in valid_params:
            return config

        tracking_config = config.pop('tracking', None)

        if tracking_config is None:
            return config
        if not isinstance(tracking_config, dict):
            logger.warning(f"Ignoring non-dict tracking config for device {config.get('device_class_id')}")
            return config

        merged_config = tracking_config.copy()
        merged_config.update(config)
        return merged_config

    def _start_generic_device_process(self, device_info: Dict[str, Any]) -> Optional[mp.Process]:
        """Build a spawn worker without pickling the lifecycle manager."""
        parent_conn = None
        child_conn = None
        try:
            device_class_name = str(device_info['device_class'])
            parent_conn, child_conn = self._mp_context.Pipe(duplex=False)
            owner_token = uuid.uuid4().hex
            device_config = copy.deepcopy(dict(device_info['config']))
            raw_runtime_config = self.config.get('runtime')
            runtime_config = copy.deepcopy(
                raw_runtime_config if isinstance(raw_runtime_config, dict) else {}
            )
            device_info['ownership_token'] = owner_token
            device_info['ownership_status'] = 'pending'
            device_info['owned_shm_names'] = []
            device_info['ownership_conn'] = parent_conn
            device_info['ownership_child_conn'] = child_conn

            process = self._mp_context.Process(
                target=self._device_process_target,
                args=(
                    device_class_name,
                    device_config,
                    owner_token,
                    child_conn,
                    runtime_config,
                ),
                name=f"{device_class_name}-{device_info['device_class_id']}",
            )
            return process

        except Exception as e:
            for connection in (parent_conn, child_conn):
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass
            device_info['ownership_conn'] = None
            device_info['ownership_child_conn'] = None
            logger.error(f"Failed to create device process for {device_info['device_class']}: {e}")
            return None

    def _start_generic_visualizer_process(self, device_info: Dict[str, Any]) -> Optional[mp.Process]:
        """Start a visualizer process using dynamic import based on device class."""
        raise DeviceControlError('Legacy visualizers are unsupported in control-panel v1')

    def _start_device_manager_process(self) -> Optional[mp.Process]:
        """Start the Device Manager process for data aggregation."""
        raise DeviceControlError('DeviceManager is unsupported in control-panel v1')

    def _start_policy_runner_process(self) -> Optional[mp.Process]:
        """Start the Policy Runner process for action dispatching."""
        raise DeviceControlError('Policy runner is unsupported in control-panel v1')

    @staticmethod
    def _shared_memory_name(device_info: Dict[str, Any]) -> str:
        device_name = (
            device_info['config'].get('device_name')
            or device_info['device_class']
        )
        return f"{device_name}_{device_info['device_class_id']}_data"

    @staticmethod
    def _close_connection(connection: Any) -> None:
        if connection is None:
            return
        try:
            connection.close()
        except (AttributeError, OSError):
            pass

    def _close_ownership_sender(self, device_info: Dict[str, Any]) -> None:
        """Close the parent's duplicate of the worker's send-only Pipe end."""
        self._close_connection(device_info.get('ownership_child_conn'))
        device_info['ownership_child_conn'] = None

    def _close_ownership_channel(self, device_info: Dict[str, Any]) -> None:
        self._close_ownership_sender(device_info)
        self._close_connection(device_info.get('ownership_conn'))
        device_info['ownership_conn'] = None

    def _apply_ownership_event(
        self, device_info: Dict[str, Any], event: Dict[str, Any]
    ) -> None:
        expected_token = device_info.get('ownership_token')
        event_token = event.get('owner_token')
        if expected_token and event_token != expected_token:
            logger.warning(
                f"Ignoring ownership event with the wrong token for "
                f"{self._device_key(device_info)}"
            )
            return

        event_name = str(event.get('event', 'unknown'))
        device_info['ownership_last_event'] = event_name
        if event.get('start_method'):
            device_info['ownership_start_method'] = event['start_method']

        if event_name == 'worker_started':
            device_info['ownership_status'] = 'worker_started'
        elif event_name == 'preflight':
            device_info['ownership_status'] = 'preflight'
        elif event_name == 'preflight_clear':
            device_info['ownership_status'] = 'preflight_clear'
        elif event_name == 'claiming':
            device_info['ownership_status'] = 'claiming'
        elif event_name == 'claimed':
            name = str(event.get('name'))
            owned_names = device_info.setdefault('owned_shm_names', [])
            if name and name not in owned_names:
                owned_names.append(name)
            device_info['ownership_status'] = 'owned'
        elif event_name == 'conflict':
            name = event.get('name') or self._shared_memory_name(device_info)
            error = event.get('error') or f"Shared memory {name} already exists"
            device_info['ownership_status'] = 'conflict'
            device_info['lifecycle_status'] = 'ownership_conflict'
            # An ownership collision is not a crash to auto-restart.  Retrying
            # could repeatedly initialize motion hardware while another owner
            # remains active.
            device_info['desired_state'] = 'stopped'
            device_info['last_error'] = (
                f"Shared-memory ownership conflict: {error}; "
                "the existing segment was not removed"
            )
        elif event_name == 'claim_error':
            device_info['ownership_status'] = 'claim_error'
            device_info['lifecycle_status'] = 'failed'
            device_info['last_error'] = (
                f"Shared-memory ownership claim failed: {event.get('error')}"
            )
        elif event_name == 'start_failed':
            device_info['ownership_status'] = 'start_failed'
            device_info['lifecycle_status'] = 'failed'
            device_info['desired_state'] = 'stopped'
            device_info['failure_code'] = str(
                event.get('code') or 'device_start_failed'
            )
            device_info['failure_recoverable'] = event.get('recoverable') is True
            # The worker accepts this value only from the allowlisted
            # startup_failure protocol and bounds it before sending.
            device_info['last_error'] = (
                ' '.join(
                    str(
                        event.get('error')
                        or 'Device start_server returned False'
                    ).split()
                )[:512]
            )
        elif event_name == 'worker_error':
            if (
                device_info.get('ownership_status')
                not in {'conflict', 'start_failed'}
            ):
                error_type = event.get('error_type', 'Error')
                device_info['last_error'] = (
                    f"Device worker {error_type}: {event.get('error')}"
                )
                device_info['lifecycle_status'] = 'failed'
        elif event_name == 'stop_blocked':
            device_info['ownership_status'] = 'stop_blocked'
            device_info['lifecycle_status'] = 'stop_timeout'
            device_info['failure_code'] = str(
                event.get('code') or 'motion_stop_not_confirmed'
            )
            device_info['failure_recoverable'] = True
            if device_info['failure_code'] == 'yam_zero_not_confirmed':
                device_info['last_error'] = (
                    'Yam disconnect blocked: measured joint-zero was not confirmed; '
                    'the connection and zero-position hold remain active'
                )
            else:
                device_info['last_error'] = (
                    'Motion-device disconnect was blocked by its safety check'
                )
        elif event_name == 'device_health':
            health = event.get('health')
            process = device_info.get('process')
            expected_pid = getattr(process, 'pid', None) if process is not None else None
            event_pid = event.get('pid')
            generation = event.get('generation')
            if generation != expected_token:
                logger.warning(
                    f"Ignoring health event from an old worker generation for "
                    f"{self._device_key(device_info)}"
                )
            elif expected_pid is not None and event_pid != expected_pid:
                logger.warning(
                    f"Ignoring health event from PID {event_pid} for "
                    f"{self._device_key(device_info)}; expected {expected_pid}"
                )
            elif isinstance(health, dict):
                # The worker payload contains only booleans, enum names, and
                # allowlisted diagnostics.  Reception time is authoritative for
                # staleness because it uses the owner's monotonic clock.
                device_info['device_health'] = copy.deepcopy(health)
                device_info['device_health_received_at'] = time.monotonic()
        elif event_name == 'device_progress':
            process = device_info.get('process')
            expected_pid = (
                getattr(process, 'pid', None) if process is not None else None
            )
            phase = str(event.get('phase') or '').strip().casefold()
            if (
                event.get('generation') == expected_token
                and (expected_pid is None or event.get('pid') == expected_pid)
                and phase
                and len(phase) <= 64
                and phase.replace('_', '').isalnum()
            ):
                device_info['progress'] = {
                    'phase': phase,
                    'updated_at': time.time(),
                }

    def _drain_ownership_events(
        self,
        device_info: Dict[str, Any],
        *,
        wait_timeout: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Consume available child ownership events, optionally waiting once."""
        connection = device_info.get('ownership_conn')
        if connection is None:
            return []

        events: List[Dict[str, Any]] = []
        deadline = time.monotonic() + max(0.0, float(wait_timeout))
        first_poll = True
        while True:
            timeout = 0.0
            if first_poll and wait_timeout > 0:
                timeout = max(0.0, deadline - time.monotonic())
            try:
                if not connection.poll(timeout):
                    break
                event = connection.recv()
            except (EOFError, OSError):
                self._close_connection(connection)
                device_info['ownership_conn'] = None
                break
            first_poll = False
            if isinstance(event, dict):
                events.append(event)
                self._apply_ownership_event(device_info, event)
        return events

    def _wait_for_ownership_preflight(
        self, device_info: Dict[str, Any]
    ) -> None:
        """Wait briefly for the spawn worker's pre-device ownership handshake."""
        if device_info.get('ownership_conn') is None:
            return
        configured_timeout = self.config.get('global_settings', {}).get(
            'ownership_handshake_timeout', 2.0
        )
        deadline = time.monotonic() + max(0.0, float(configured_timeout))
        terminal_events = {
            'preflight_clear',
            'claiming',
            'claimed',
            'conflict',
            'claim_error',
            'worker_error',
        }
        while time.monotonic() < deadline:
            events = self._drain_ownership_events(
                device_info,
                wait_timeout=max(0.0, deadline - time.monotonic()),
            )
            if not events:
                return
            if any(event.get('event') in terminal_events for event in events):
                return

    def _probe_shared_memory(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect a device SHM without registering it with resource_tracker.

        The production controller runs on Linux, where POSIX shared-memory
        objects are exposed in /dev/shm.  Reading the headers as an ordinary
        file is intentionally used instead of ``SharedMemory(create=False)``:
        Python's resource tracker may otherwise unlink a segment owned by an
        independently started device when the controller exits.
        """
        name = self._shared_memory_name(device_info)
        config = device_info['config']
        state_feedback = config.get('state_feedback')
        state_feedback = state_feedback if isinstance(state_feedback, dict) else {}
        state_format = state_feedback.get('format')
        state_format = state_format if isinstance(state_format, dict) else {}
        result: Dict[str, Any] = {
            'name': name,
            'connected': False,
            'last_timestamp_ns': None,
            'age_ms': None,
            'fps': state_feedback.get('fps', config.get('fps')),
            'shape': state_format.get('shape', config.get('data_shape')),
            'dtype': state_format.get('dtype', config.get('data_dtype')),
            'fresh': False,
            'frames': 0,
            'error': None,
        }
        shm_path = Path('/dev/shm') / name.lstrip('/')
        if not shm_path.exists():
            return result

        try:
            with shm_path.open('rb', buffering=0) as handle:
                device_header_bytes = handle.read(DEVICE_HEADER_SIZE)
                buffer_header_bytes = handle.read(BUFFER_HEADER_SIZE)
                device_header = unpack_device_header(device_header_bytes)
                buffer_header = unpack_buffer_header(buffer_header_bytes)

                result.update(
                    {
                        'connected': True,
                        'fps': device_header['fps'],
                        'shape': list(device_header['shape']),
                        'dtype': device_header['data_dtype'],
                        'frames': buffer_header['current_frames_count'],
                    }
                )
                if (
                    buffer_header['current_frames_count'] > 0
                    and buffer_header['buffer_size'] > 0
                ):
                    latest_index = (
                        buffer_header['write_index'] - 1
                    ) % buffer_header['buffer_size']
                    frame_offset = (
                        DEVICE_HEADER_SIZE
                        + BUFFER_HEADER_SIZE
                        + latest_index * device_header['frame_size']
                    )
                    handle.seek(frame_offset)
                    timestamp_bytes = handle.read(FRAME_HEADER_SIZE)
                    if len(timestamp_bytes) == FRAME_HEADER_SIZE:
                        timestamp_ns = int.from_bytes(
                            timestamp_bytes, byteorder='little', signed=False
                        )
                        if timestamp_ns > 0:
                            age_ms = max(0.0, (time.time_ns() - timestamp_ns) / 1e6)
                            fps = float(device_header['fps'] or 0.0)
                            stale_after_s = max(2.0, 5.0 / fps) if fps > 0 else 2.0
                            result.update(
                                {
                                    'last_timestamp_ns': timestamp_ns,
                                    'age_ms': round(age_ms, 3),
                                    'fresh': age_ms <= stale_after_s * 1000.0,
                                }
                            )
        except (FileNotFoundError, PermissionError):
            # The producer may unlink between exists() and open().
            return result
        except Exception as exc:
            result['connected'] = True
            result['error'] = str(exc)
        return result

    @staticmethod
    def _is_preview_device(device_info: Dict[str, Any]) -> bool:
        name = str(device_info['device_class']).lower()
        config = device_info['config']
        state_feedback = config.get('state_feedback')
        state_feedback = state_feedback if isinstance(state_feedback, dict) else {}
        state_format = state_feedback.get('format')
        state_format = state_format if isinstance(state_format, dict) else {}
        shape = state_format.get('shape', config.get('data_shape'))
        return (
            'camera' in name
            or 'iphone' in name
            or (
                isinstance(shape, (list, tuple))
                and len(shape) == 3
                and shape[-1] in (1, 3, 4)
            )
        )

    def _is_motion_device(self, device_info: Dict[str, Any]) -> bool:
        """Conservatively identify devices whose shutdown can leave motion."""
        config = device_info['config']
        configured_risk = str(
            config.get('risk')
            or config.get('safety_class')
            or config.get('device_risk')
            or ''
        ).lower()
        if configured_risk in {'motion', 'robot', 'actuator', 'high'}:
            return True
        if configured_risk in {'camera', 'sensor', 'none', 'low'}:
            return False
        if self._is_preview_device(device_info):
            return False
        class_name = str(device_info['device_class']).lower()
        motion_tokens = (
            'robot',
            'teleop',
            'motion',
            'actuator',
            'manipulator',
            'motor',
            'arm',
        )
        return any(token in class_name for token in motion_tokens)

    def _resolved_stop_timeout(
        self,
        device_info: Dict[str, Any],
        requested_timeout: Optional[float],
    ) -> float:
        settings = self.config.get('global_settings', {})
        if self._is_motion_device(device_info):
            config = device_info.get('config', {})
            device_safety_window = 0.0
            if str(device_info.get('device_class')) == 'YamHeadRobot':
                device_safety_window = max(
                    0.0,
                    float(config.get('ramp_duration_s', 8.0)),
                ) + max(
                    0.0,
                    float(config.get('disconnect_zero_timeout_s', 3.0)),
                ) + 5.0
            safe_default = max(
                30.0,
                float(settings.get('motion_stop_timeout', 30.0)),
                device_safety_window,
            )
            if requested_timeout is None:
                return safe_default
            # A generic/API caller cannot silently shorten the motion safety
            # window below the configured minimum.
            return max(safe_default, float(requested_timeout))
        if requested_timeout is not None:
            return max(0.0, float(requested_timeout))
        return max(0.0, float(settings.get('device_stop_timeout', 8.0)))

    def _device_safety_status(
        self,
        device_info: Dict[str, Any],
        *,
        alive: bool,
        stream_ready: bool,
    ) -> Dict[str, Any]:
        """Return the freshest passive device-health observation.

        Only Rizon workers currently expose a structured SDK safety heartbeat.
        Other devices remain ``not_applicable`` and continue to use process +
        SHM readiness. A running Rizon without a heartbeat after its stream is
        ready is fail-closed instead of being shown as healthy.
        """
        is_rizon = str(device_info.get('device_class')) == 'RizonRobot'
        if not is_rizon:
            return {
                'source': 'not_applicable',
                'known': None,
                'operational': None,
                'fault': None,
                'estop': None,
                'reduced': None,
                'fail_closed': False,
                'stale': False,
                'age_ms': None,
                'reason': None,
                'last_seen': None,
                'arms': {},
            }

        health = device_info.get('device_health')
        received_at = device_info.get('device_health_received_at')
        age_s = (
            max(0.0, time.monotonic() - float(received_at))
            if isinstance(received_at, (int, float))
            else None
        )
        configured_stale_after = self.config.get('global_settings', {}).get(
            'device_health_stale_after', 1.5
        )
        try:
            stale_after_s = max(0.5, float(configured_stale_after))
        except (TypeError, ValueError):
            stale_after_s = 1.5
        stale = bool(age_s is None or age_s > stale_after_s)
        initializing = bool(alive and not stream_ready)

        if not isinstance(health, dict):
            reason = (
                None
                if not alive
                else (
                    'Rizon health heartbeat is unavailable'
                    if stream_ready
                    else 'Waiting for the first Rizon health heartbeat'
                )
            )
            return {
                'source': 'flexiv_worker_passive_probe',
                'known': False,
                'operational': None,
                'fault': None,
                'estop': None,
                'reduced': None,
                'fail_closed': bool(alive),
                'initializing': initializing,
                'stale': True,
                'age_ms': None,
                'reason': reason,
                'last_seen': None,
                'arms': {},
            }

        result = copy.deepcopy(health)
        result['stale'] = stale
        result['initializing'] = bool(initializing and stale)
        result['age_ms'] = round(age_s * 1000.0, 3) if age_s is not None else None
        result['last_seen'] = (
            time.time() - age_s if age_s is not None else None
        )
        if stale:
            result['known'] = False
            result['operational'] = None
            result['fault'] = None
            result['estop'] = None
            result['reduced'] = None
            if alive:
                result['fail_closed'] = True
                result['reason'] = (
                    'Waiting for a fresh Rizon health heartbeat'
                    if initializing
                    else (
                        'Rizon health heartbeat is stale; emergency stop, fault, or '
                        'disconnect cannot be excluded'
                    )
                )
            else:
                result['fail_closed'] = False
                result['reason'] = None
        else:
            result['fail_closed'] = bool(result.get('fail_closed'))
        return result

    def _status_for_device(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        # A synchronous starter owns the Pipe while it waits for the preflight
        # event. Other status callers must not consume that event first.
        if not device_info.get('ownership_waiting'):
            self._drain_ownership_events(device_info)
        process = device_info.get('process')
        alive = self._process_is_alive(process)
        shm_status = self._probe_shared_memory(device_info)
        desired_state = device_info.get('desired_state', 'stopped')
        stream_ready = bool(
            shm_status['connected'] and int(shm_status.get('frames') or 0) > 0
        )
        safety = self._device_safety_status(
            device_info,
            alive=alive,
            stream_ready=stream_ready,
        )
        started_at = device_info.get('started_at')

        lifecycle_status = device_info.get('lifecycle_status')
        if lifecycle_status == 'stop_timeout' and alive:
            status = 'stop_timeout'
        elif lifecycle_status == 'ownership_conflict':
            status = 'conflict'
        elif lifecycle_status == 'orphaned':
            status = 'orphaned'
        elif lifecycle_status == 'stopping':
            status = 'stopping'
        elif lifecycle_status == 'failed':
            # A worker may still be inside its cleanup finally block after it
            # has reported a terminal startup error. Surface that error now,
            # rather than mislabeling the process as degraded until it exits.
            status = 'failed'
        elif alive:
            if (
                shm_status['connected']
                and shm_status['fresh']
                and not safety.get('fail_closed')
            ):
                status = 'running'
            elif safety.get('fail_closed') and safety.get('initializing'):
                status = 'starting'
            elif safety.get('fail_closed'):
                status = 'degraded'
            elif not stream_ready:
                status = 'starting'
            else:
                status = 'degraded'
        elif device_info.get('spawn_in_progress') or (
            lifecycle_status == 'starting' and desired_state == 'running'
        ):
            status = 'starting'
        elif shm_status['connected']:
            status = 'orphaned' if process is not None else 'external'
        elif desired_state == 'running':
            status = 'failed'
        elif lifecycle_status == 'stopping':
            status = 'stopping'
        else:
            status = 'stopped'

        exit_code = None
        if process is not None:
            try:
                exit_code = process.exitcode
            except (AssertionError, ValueError):
                exit_code = None

        device_class = str(device_info['device_class'])
        device_class_id = device_info['device_class_id']
        config = device_info['config']
        state_feedback = config.get('state_feedback')
        state_feedback = state_feedback if isinstance(state_feedback, dict) else {}
        state_format = state_feedback.get('format')
        state_format = state_format if isinstance(state_format, dict) else {}
        configured_fps = state_feedback.get('fps', config.get('fps'))
        configured_shape = state_format.get('shape', config.get('data_shape'))
        configured_dtype = state_format.get('dtype', config.get('data_dtype'))
        capture_binding_missing = capture_binding_is_missing(device_info['config'])
        ready = bool(
            alive
            and desired_state == 'running'
            and lifecycle_status != 'stopping'
            and shm_status['connected']
            and shm_status['fresh']
            and not safety.get('fail_closed')
        )
        observed_error = device_info.get('last_error') or safety.get('reason')
        progress = copy.deepcopy(device_info.get('progress') or {})
        if ready:
            progress = {
                'phase': 'ready',
                'updated_at': progress.get('updated_at'),
            }
        elif status == 'stopped':
            progress = {
                'phase': 'stopped',
                'updated_at': progress.get('updated_at'),
            }
        elif status in {'failed', 'conflict', 'orphaned'}:
            progress = {
                'phase': 'failed',
                'updated_at': progress.get('updated_at'),
            }
        return {
            'key': self._device_key(device_info),
            'class': device_class,
            'type': device_class,  # Backward-compatible status field.
            'id': device_class_id,
            'label': f"{device_class} {device_class_id}",
            'configured': True,
            'enabled': device_info['config'].get('enabled', True) is not False,
            'desired_state': desired_state,
            'status': status,
            'running': alive,
            'ready': ready,
            'pid': getattr(process, 'pid', None) if process is not None else None,
            'exit_code': exit_code,
            'started_at': started_at,
            'restart_count': device_info.get('restart_count', 0),
            'last_error': observed_error,
            'reason': (
                'camera binding is missing; select a detected camera before starting'
                if capture_binding_missing
                else None
            ),
            'binding_status': (
                'missing' if capture_binding_missing else None
            ),
            'failure_code': device_info.get('failure_code'),
            'failure_recoverable': device_info.get('failure_recoverable'),
            'progress': progress,
            'failure': (
                {
                    'code': device_info.get('failure_code'),
                    'recoverable': device_info.get('failure_recoverable'),
                    'message': device_info.get('last_error'),
                }
                if device_info.get('failure_code')
                else None
            ),
            'fps': configured_fps,
            'shape': configured_shape,
            'data_shape': configured_shape,
            'dtype': configured_dtype,
            'data_dtype': configured_dtype,
            'shm': shm_status,
            'safety': safety,
            'risk': 'motion' if self._is_motion_device(device_info) else None,
            'flexiv_rdk': (
                {
                    'version': device_info.get('flexiv_rdk_version'),
                    'probe': copy.deepcopy(device_info.get('flexiv_rdk_probe')),
                }
                if device_class == 'RizonRobot'
                else None
            ),
            'ownership': {
                'status': device_info.get('ownership_status', 'unclaimed'),
                'shared_memory': list(device_info.get('owned_shm_names', [])),
                'start_method': device_info.get('ownership_start_method'),
                'last_event': device_info.get('ownership_last_event'),
            },
            'capabilities': {
                'preview': self._is_preview_device(device_info),
                'start': status not in {
                    'running',
                    'starting',
                    'external',
                    'orphaned',
                    'stop_timeout',
                } and not capture_binding_missing,
                'stop': process is not None and alive,
                'restart': process is not None and alive,
            },
        }

    def get_device_status(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        """Return the current process and shared-memory state for one device."""
        with self._lock:
            return self._status_for_device(
                self._find_device(device_class, device_class_id)
            )

    def request_stop_device(self, device_key: str) -> bool:
        """Signal a start in progress without waiting for its operation lock.

        The regular ``stop_device`` path still owns joining, timeout handling,
        and final cleanup. This method only changes the desired state and sends
        the same graceful termination signal used by that path, allowing a
        safety-reducing stop to supersede RDK probing or worker initialization.
        """

        normalized_key = str(device_key)
        with self._lock:
            device_info = next(
                (
                    item
                    for item in self.devices
                    if self._device_key(item) == normalized_key
                ),
                None,
            )
            if device_info is None:
                return False
            if (
                device_info.get('desired_state') == 'stopped'
                and device_info.get('lifecycle_status') in {
                    'stopped',
                    'stopping',
                    'stop_timeout',
                }
            ):
                return False

            device_info['desired_state'] = 'stopped'
            device_info['lifecycle_status'] = 'stopping'
            device_info['progress'] = {
                'phase': 'stop_requested',
                'updated_at': time.time(),
            }
            process = device_info.get('process')
            self._monitor_wakeup.set()

        if self._process_is_alive(process):
            try:
                process.terminate()
            except (AssertionError, ValueError, OSError):
                pass
        return True

    @staticmethod
    def _join_worker(process: Any, timeout: float) -> None:
        try:
            process.join(timeout=max(0.0, timeout))
        except (AssertionError, ValueError, OSError):
            pass

    def _stop_worker(
        self,
        process: Any,
        *,
        timeout: float = 8.0,
        allow_kill: bool = True,
    ) -> Tuple[bool, bool]:
        """Request graceful cleanup and return ``(stopped, force_killed)``."""
        if process is None:
            return True, False
        if self._process_is_alive(process):
            try:
                process.terminate()
            except (AssertionError, ValueError, OSError):
                pass
        self._join_worker(process, timeout)
        forced = False
        if self._process_is_alive(process) and allow_kill:
            forced = True
            try:
                process.kill()
            except (AttributeError, AssertionError, ValueError, OSError):
                pass
            self._join_worker(process, 2.0)
        return not self._process_is_alive(process), forced

    def _force_kill_worker(self, process: Any) -> Tuple[bool, Optional[int]]:
        """Immediately SIGKILL one panel-owned worker and reap it."""
        if process is None:
            return True, None
        try:
            pid = int(process.pid)
        except (AttributeError, TypeError, ValueError):
            pid = None
        if self._process_is_alive(process):
            try:
                process.kill()
            except (AttributeError, AssertionError, OSError, ValueError):
                if pid is not None and os.name == 'posix':
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
        self._join_worker(process, 2.0)
        return not self._process_is_alive(process), pid

    @staticmethod
    def _owned_shm_snapshot(
        device_info: Dict[str, Any],
    ) -> Dict[str, Tuple[int, int]]:
        """Snapshot inode identities for SHM claimed by this worker generation."""
        snapshot: Dict[str, Tuple[int, int]] = {}
        for raw_name in device_info.get('owned_shm_names') or ():
            name = str(raw_name).lstrip('/')
            if not name or '/' in name or '\\' in name:
                continue
            path = Path('/dev/shm') / name
            try:
                result = path.stat()
            except (FileNotFoundError, OSError):
                continue
            snapshot[name] = (int(result.st_dev), int(result.st_ino))
        return snapshot

    @staticmethod
    def _unlink_force_killed_shm(
        snapshot: Dict[str, Tuple[int, int]],
    ) -> Tuple[List[str], List[str]]:
        """Unlink only the exact SHM inodes observed before the owned PID died."""
        removed: List[str] = []
        remaining: List[str] = []
        for name, identity in snapshot.items():
            path = Path('/dev/shm') / name
            try:
                result = path.stat()
            except FileNotFoundError:
                continue
            except OSError:
                remaining.append(name)
                continue
            if (int(result.st_dev), int(result.st_ino)) != identity:
                # Another producer recreated this name after the owned worker
                # died. Never unlink a replacement generation.
                remaining.append(name)
                continue
            try:
                path.unlink()
                removed.append(name)
            except FileNotFoundError:
                continue
            except OSError:
                remaining.append(name)
        return removed, remaining

    def _start_device(
        self,
        device_class: str,
        device_class_id: Any,
        *,
        wait_for_ownership: bool,
    ) -> Dict[str, Any]:
        with self._lock:
            device_info = self._find_device(device_class, device_class_id)
            if capture_binding_is_missing(device_info['config']):
                raise DeviceConflictError(
                    f"Camera binding is missing for {self._device_key(device_info)}; "
                    "select a detected camera before starting it"
                )
            operation_lock = self._device_operation_lock(device_info)

        if not operation_lock.acquire(blocking=False):
            raise DeviceConflictError(
                f"Device action already in progress: {self._device_key(device_info)}"
            )
        try:
            with self._lock:
                if self._process_is_alive(device_info.get('process')):
                    raise DeviceConflictError(
                        f"Device is already running: {self._device_key(device_info)}"
                    )

                shm_status = self._probe_shared_memory(device_info)
                if shm_status['connected']:
                    raise DeviceConflictError(
                        f"Shared memory {shm_status['name']} already exists; "
                        "an external or orphaned device may be running"
                    )

                stale_process = device_info.get('process')
                if stale_process is not None:
                    self._join_worker(stale_process, 0.0)
                    device_info['process'] = None
                    self._close_ownership_channel(device_info)

                device_info['desired_state'] = 'running'
                device_info['lifecycle_status'] = 'starting'
                device_info['last_error'] = None
                device_info['failure_code'] = None
                device_info['failure_recoverable'] = None
                device_info['device_health'] = None
                device_info['device_health_received_at'] = None
                device_info['progress'] = {
                    'phase': 'queued',
                    'updated_at': time.time(),
                }
                self.running = True

            self._prepare_rizon_rdk(device_info)
            with self._lock:
                if device_info.get('desired_state') != 'running':
                    device_info['lifecycle_status'] = 'stopped'
                    device_info['progress'] = {
                        'phase': 'stopped',
                        'updated_at': time.time(),
                    }
                    self._monitor_wakeup.set()
                    return self._status_for_device(device_info)
            if not self._start_device_process(
                device_info,
                wait_for_ownership=wait_for_ownership,
            ):
                with self._lock:
                    device_info['lifecycle_status'] = 'failed'
                    device_info['last_error'] = 'Failed to start device process'
                raise DeviceControlError(device_info['last_error'])
            with self._lock:
                self._monitor_wakeup.set()
                return self._status_for_device(device_info)
        finally:
            operation_lock.release()

    def start_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        """Start one device and synchronously verify the ownership preflight."""
        return self._start_device(
            device_class,
            device_class_id,
            wait_for_ownership=True,
        )

    def start_device_observed(
        self,
        device_class: str,
        device_class_id: Any,
    ) -> Dict[str, Any]:
        """Spawn one device while readiness/ownership completes via observation."""
        return self._start_device(
            device_class,
            device_class_id,
            wait_for_ownership=False,
        )

    def stop_device(
        self,
        device_class: str,
        device_class_id: Any,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Stop exactly one process and prevent the monitor from restarting it."""
        with self._lock:
            device_info = self._find_device(device_class, device_class_id)
            operation_lock = self._device_operation_lock(device_info)
        if not operation_lock.acquire(blocking=False):
            raise DeviceConflictError(
                f"Device action already in progress: {self._device_key(device_info)}"
            )
        try:
            with self._lock:
                process = device_info.get('process')
                if process is None:
                    if self._probe_shared_memory(device_info)['connected']:
                        raise DeviceConflictError(
                            f"Cannot stop external device without an owned PID: "
                            f"{self._device_key(device_info)}"
                        )
                    device_info['desired_state'] = 'stopped'
                    device_info['lifecycle_status'] = 'stopped'
                    device_info['progress'] = {
                        'phase': 'stopped',
                        'updated_at': time.time(),
                    }
                    self._close_ownership_channel(device_info)
                    return self._status_for_device(device_info)

                device_info['desired_state'] = 'stopped'
                device_info['lifecycle_status'] = 'stopping'
                device_info['progress'] = {
                    'phase': 'stop_requested',
                    'updated_at': time.time(),
                }
                self._drain_ownership_events(device_info)
                shm_snapshot = self._owned_shm_snapshot(device_info)
                motion_device = self._is_motion_device(device_info)
                resolved_timeout = self._resolved_stop_timeout(
                    device_info, timeout
                )

            stopped, force_killed = self._stop_worker(
                process,
                timeout=resolved_timeout,
                allow_kill=not motion_device,
            )

            with self._lock:
                device_info['last_exit_at'] = time.time()
                self._drain_ownership_events(device_info)
                if not stopped:
                    if motion_device:
                        device_info['last_error'] = (
                            "Safety stop incomplete: motion device process is still "
                            f"alive after {resolved_timeout:.1f}s; SIGKILL was not sent"
                        )
                        device_info['lifecycle_status'] = 'stop_timeout'
                    else:
                        device_info['last_error'] = (
                            "Device process is still alive after graceful stop and "
                            "force-kill attempts"
                        )
                        device_info['lifecycle_status'] = 'failed'
                else:
                    device_info['process'] = None
                    removed, remaining = self._unlink_force_killed_shm(
                        shm_snapshot
                    )
                    if removed:
                        device_info['owned_shm_names'] = [
                            name
                            for name in device_info.get('owned_shm_names') or []
                            if str(name).lstrip('/') not in removed
                        ]
                    if remaining:
                        device_info['lifecycle_status'] = 'orphaned'
                        device_info['last_error'] = (
                            'Owned process stopped, but claimed shared memory '
                            'could not be removed: ' + ', '.join(remaining)
                        )
                    else:
                        device_info['lifecycle_status'] = 'stopped'
                        device_info['progress'] = {
                            'phase': 'stopped',
                            'updated_at': time.time(),
                        }
                        if force_killed:
                            device_info['last_error'] = (
                                f"Graceful stop timed out after {resolved_timeout:.1f}s; "
                                "the non-motion process was force-killed"
                            )
                        else:
                            device_info['last_error'] = None
                    self._close_ownership_channel(device_info)
                self._monitor_wakeup.set()
                return self._status_for_device(device_info)
        finally:
            operation_lock.release()

    def restart_device(
        self,
        device_class: str,
        device_class_id: Any,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Stop then start one device, preserving its automatic restart count."""
        stopped = self.stop_device(device_class, device_class_id, timeout=timeout)
        if stopped['running']:
            raise DeviceControlError(
                stopped.get('last_error') or 'Device did not stop safely'
            )
        return self.start_device(device_class, device_class_id)

    def restart_device_observed(
        self,
        device_class: str,
        device_class_id: Any,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Stop one device, then use observed ownership/readiness for its spawn."""
        stopped = self.stop_device(device_class, device_class_id, timeout=timeout)
        if stopped['running']:
            raise DeviceControlError(
                stopped.get('last_error') or 'Device did not stop safely'
            )
        return self.start_device_observed(device_class, device_class_id)

    def has_active_devices(self) -> bool:
        """Return true for owned workers or matching external/orphaned SHMs."""
        with self._lock:
            for device_info in self.devices:
                if self._process_is_alive(device_info.get('process')):
                    return True
                if self._probe_shared_memory(device_info)['connected']:
                    return True
            return False

    def start_monitor(self) -> None:
        """Run the crash monitor in a daemon thread for API-driven control."""
        with self._lock:
            if self._monitor_thread and self._monitor_thread.is_alive():
                return
            self.running = True
            self._monitor_wakeup.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_devices,
                name='ControlPanelDeviceMonitor',
                daemon=True,
            )
            self._monitor_thread.start()

    def _start_device_process(
        self,
        device_info: Dict[str, Any],
        *,
        wait_for_ownership: bool = True,
    ) -> bool:
        """Start a single device process and optionally its visualizer."""
        try:
            with self._lock:
                if device_info.get('desired_state') != 'running':
                    device_info['lifecycle_status'] = 'stopped'
                    device_info['progress'] = {
                        'phase': 'stopped',
                        'updated_at': time.time(),
                    }
                    return True
                process = self._start_generic_device_process(device_info)
                if not process:
                    return False
                # Publish an explicit starting state before multiprocessing
                # spawn. Status readers can now observe this device without
                # waiting for spawn/import work to complete.
                device_info['process'] = process
                device_info['spawn_in_progress'] = True
                device_info['ownership_waiting'] = bool(wait_for_ownership)
                device_info['progress'] = {
                    'phase': 'spawning_worker',
                    'updated_at': time.time(),
                }

            process.start()
            with self._lock:
                device_info['spawn_in_progress'] = False
                self._close_ownership_sender(device_info)
                device_info['last_restart'] = time.time()
                device_info['started_at'] = device_info['last_restart']
                device_info['last_exit_at'] = None
                stop_requested = device_info.get('desired_state') != 'running'
                device_info['lifecycle_status'] = (
                    'stopping' if stop_requested else 'starting'
                )
                device_info['last_error'] = None

            if stop_requested and self._process_is_alive(process):
                try:
                    process.terminate()
                except (AssertionError, ValueError, OSError):
                    pass

            logger.info(f"Started {device_info['device_class']} device {device_info['device_class_id']} (PID: {process.pid})")

            # Direct lifecycle callers retain synchronous TOCTOU detection.
            # Web controls use observed start so Rizon and camera workers can
            # be spawned back to back while the state broker reconciles the
            # same authoritative ownership/readiness events asynchronously.
            if wait_for_ownership:
                self._wait_for_ownership_preflight(device_info)
                with self._lock:
                    device_info['ownership_waiting'] = False
                    ownership_conflict = (
                        device_info.get('ownership_status') == 'conflict'
                    )
                if ownership_conflict:
                    self._join_worker(process, 2.0)
                    with self._lock:
                        if not self._process_is_alive(process):
                            device_info['process'] = None
                            self._close_ownership_channel(device_info)
                    raise DeviceConflictError(
                        device_info.get('last_error')
                        or 'Shared-memory ownership conflict'
                    )
            else:
                with self._lock:
                    device_info['ownership_waiting'] = False

            # Browser previews read the producer SHM directly; legacy desktop
            # visualizer processes are intentionally outside control-panel v1.
            if False:  # pragma: no cover - retained legacy body is unreachable
                try:
                    # Wait a bit for device to initialize
                    time.sleep(1.0)

                    visualizer_process = self._start_generic_visualizer_process(device_info)
                    if visualizer_process:
                        device_info['visualizer_process'] = visualizer_process
                        logger.info(f"Started {device_info['device_class']} visualizer for device {device_info['device_class_id']} (PID: {visualizer_process.pid})")

                except Exception as e:
                    logger.warning(f"Failed to start visualizer for {device_info['device_class']} device {device_info['device_class_id']}: {e}")

            return True

        except DeviceConflictError:
            raise
        except Exception as e:
            with self._lock:
                device_info['spawn_in_progress'] = False
                device_info['ownership_waiting'] = False
                self._close_ownership_sender(device_info)
                if not self._process_is_alive(device_info.get('process')):
                    device_info['process'] = None
                    self._close_ownership_channel(device_info)
                device_info['last_error'] = str(e)
                device_info['lifecycle_status'] = 'failed'
            logger.error(f"Failed to start {device_info['device_class']} device {device_info['device_class_id']}: {e}")
            return False

    def _monitor_devices(self) -> None:
        """Monitor device processes and restart failed ones if configured."""
        global_settings = self.config.get('global_settings', {})
        restart_on_failure = global_settings.get('restart_on_failure', True)
        max_restart_attempts = global_settings.get('max_restart_attempts', 3)
        restart_delay = global_settings.get('restart_delay', 2.0)

        logger.info("Starting device monitoring...")

        try:
            while self.running:
                restart_candidates: List[
                    Tuple[Dict[str, Any], threading.RLock]
                ] = []
                with self._lock:
                    now = time.time()
                    for device_info in self.devices:
                        if (
                            device_info.get('spawn_in_progress')
                            or device_info.get('ownership_waiting')
                        ):
                            continue
                        self._drain_ownership_events(device_info)
                        process = device_info.get('process')
                        if process is None or self._process_is_alive(process):
                            continue

                        if device_info.get('last_exit_at') is None:
                            device_info['last_exit_at'] = now
                            try:
                                exit_code = process.exitcode
                            except (AssertionError, ValueError):
                                exit_code = None
                            if not device_info.get('last_error'):
                                device_info['last_error'] = (
                                    f"Device process exited with code {exit_code}"
                                )
                            if device_info.get('lifecycle_status') not in {
                                'ownership_conflict',
                                'stop_timeout',
                            }:
                                device_info['lifecycle_status'] = 'failed'
                            logger.warning(
                                f"Device {self._device_key(device_info)} process died "
                                f"(exit code {exit_code})"
                            )

                        # A user-requested stop is terminal.  This desired-state
                        # check is what prevents the monitor from undoing a
                        # manual stop from the control UI.
                        if device_info.get('desired_state') != 'running':
                            self._join_worker(process, 0.0)
                            device_info['process'] = None
                            if (
                                device_info.get('lifecycle_status')
                                != 'ownership_conflict'
                                and device_info.get('ownership_status')
                                != 'start_failed'
                            ):
                                device_info['lifecycle_status'] = 'stopped'
                            self._close_ownership_channel(device_info)
                            continue

                        if now - device_info['last_exit_at'] < float(restart_delay):
                            continue

                        if not restart_on_failure or device_info['restart_count'] >= max_restart_attempts:
                            if device_info.get('lifecycle_status') != 'failed':
                                logger.error(
                                    f"Device {self._device_key(device_info)} exceeded "
                                    "restart attempts or restart is disabled"
                                )
                            device_info['lifecycle_status'] = 'failed'
                            continue

                        shm_status = self._probe_shared_memory(device_info)
                        if shm_status['connected']:
                            device_info['last_error'] = (
                                f"Refusing automatic restart while SHM "
                                f"{shm_status['name']} still exists"
                            )
                            continue

                        attempt = device_info['restart_count'] + 1
                        logger.info(
                            f"Restarting device {self._device_key(device_info)} "
                            f"(attempt {attempt}/{max_restart_attempts})"
                        )
                        operation_lock = self._device_operation_lock(device_info)
                        if not operation_lock.acquire(blocking=False):
                            continue
                        self._join_worker(process, 0.0)
                        device_info['process'] = None
                        self._close_ownership_channel(device_info)
                        device_info['restart_count'] = attempt
                        device_info['last_exit_at'] = None
                        device_info['lifecycle_status'] = 'starting'
                        restart_candidates.append((device_info, operation_lock))

                for device_info, operation_lock in restart_candidates:
                    try:
                        with self._lock:
                            should_restart = bool(
                                self.running
                                and device_info.get('desired_state') == 'running'
                            )
                        if not should_restart:
                            continue
                        try:
                            self._start_device_process(device_info)
                        except DeviceConflictError as exc:
                            logger.error(
                                f"Automatic restart refused for "
                                f"{self._device_key(device_info)}: {exc}"
                            )
                    finally:
                        operation_lock.release()

                self._monitor_wakeup.wait(timeout=1.0)
                self._monitor_wakeup.clear()
        except KeyboardInterrupt:
            logger.info("Device monitoring interrupted")
            self.running = False

    def start_all(self) -> None:
        """Start all enabled devices and the Device Manager."""
        raise DeviceControlError(
            'Bulk start is intentionally unavailable; start each device explicitly'
        )
        if any(self._process_is_alive(info.get('process')) for info in self.devices):
            logger.warning("Device starter is already running")
            return

        logger.info("Starting Device Starter")

        if not self.devices:
            logger.warning("No enabled devices found in configuration")
            return False

        self.running = True

        # Start all devices first
        started_count = 0
        for device_info in self.devices:
            device_info['desired_state'] = 'running'
            try:
                if self._start_device_process(device_info):
                    started_count += 1
            except DeviceConflictError as exc:
                logger.error(
                    f"Refusing to start {self._device_key(device_info)}: {exc}"
                )

        if started_count == 0:
            logger.error("Failed to start any devices")
            self.running = False
            return False

        action_executor_config = self.config.get('action_executor', {})
        should_start_policy = (
            self.enable_policy_runner
            and action_executor_config
            and action_executor_config.get('type') not in [None, 'null', 'disabled']
        )

        # Start Device Manager after devices are ready
        if self.enable_device_manager:
            logger.info("Starting Device Manager for data aggregation...")
            self.device_manager_process = self._start_device_manager_process()
            if self.device_manager_process:
                self.device_manager_process.start()
                logger.info(f"Started Device Manager (PID: {self.device_manager_process.pid})")
            else:
                logger.error("Failed to start Device Manager")

        # Start policy runner if configured and not disabled
        if should_start_policy:
            logger.info("Starting Policy Runner for action dispatching...")
            self.policy_runner_process = self._start_policy_runner_process()
            if self.policy_runner_process:
                self.policy_runner_process.start()
                logger.info(f"Started Policy Runner (PID: {self.policy_runner_process.pid})")
            else:
                logger.error("Failed to start Policy Runner")

        # Write robot home index to shared file for PolicyConnector
        self._write_robot_home_index()

        logger.info(f"Successfully started {started_count}/{len(self.devices)} devices")
        return True

    def stop_all(self) -> Dict[str, Any]:
        """Stop owned workers and report anything that remains active.

        The report is intentionally structured because config replacement must
        never assume cleanup succeeded.  Matching SHM without an owned live PID
        is external/orphaned state and is reported, never unlinked or killed.
        """
        logger.info("Stopping all configured device workers")
        with self._lock:
            self.running = False
            for device_info in self.devices:
                device_info['desired_state'] = 'stopped'
        self._monitor_wakeup.set()

        # Do not wait for an in-flight spawn while holding the manager lock:
        # that operation needs the same lock to publish its result. Once all
        # per-device locks are owned, no manual or automatic restart can race
        # the teardown below.
        operation_locks = [
            self._device_operation_lock(device_info)
            for device_info in self.devices
        ]
        for operation_lock in operation_locks:
            operation_lock.acquire()

        statuses: List[Dict[str, Any]] = []
        stopped_devices: List[str] = []
        incomplete_devices: List[str] = []
        external_devices: List[str] = []
        removed_shm: List[str] = []

        try:
            with self._lock:
                self.running = False
                for device_info in self.devices:
                    device_info['desired_state'] = 'stopped'

                for device_info in self.devices:
                    key = self._device_key(device_info)
                    self._stop_worker(
                        device_info.get('visualizer_process'), timeout=2.0
                    )
                    device_info['visualizer_process'] = None
                    process = device_info.get('process')

                    if process is None:
                        shm_status = self._probe_shared_memory(device_info)
                        if shm_status['connected']:
                            device_info['lifecycle_status'] = 'external'
                            device_info['last_error'] = (
                                "Cannot stop matching shared memory without an "
                                "owned live process"
                            )
                            external_devices.append(key)
                            incomplete_devices.append(key)
                        else:
                            device_info['lifecycle_status'] = 'stopped'
                            self._close_ownership_channel(device_info)
                        statuses.append(self._status_for_device(device_info))
                        continue

                    device_info['lifecycle_status'] = 'stopping'
                    self._drain_ownership_events(device_info)
                    shm_snapshot = self._owned_shm_snapshot(device_info)
                    motion_device = self._is_motion_device(device_info)
                    stop_timeout = self._resolved_stop_timeout(device_info, None)
                    stopped, force_killed = self._stop_worker(
                        process,
                        timeout=stop_timeout,
                        allow_kill=not motion_device,
                    )
                    device_info['last_exit_at'] = time.time()
                    self._drain_ownership_events(device_info)

                    if not stopped:
                        if motion_device:
                            device_info['lifecycle_status'] = 'stop_timeout'
                            device_info['last_error'] = (
                                "Safety stop incomplete: motion device process is "
                                f"still alive after {stop_timeout:.1f}s; SIGKILL "
                                "was not sent"
                            )
                        else:
                            device_info['lifecycle_status'] = 'failed'
                            device_info['last_error'] = (
                                "Device process is still alive after graceful stop "
                                "and force-kill attempts"
                            )
                        incomplete_devices.append(key)
                        statuses.append(self._status_for_device(device_info))
                        continue

                    removed, remaining = self._unlink_force_killed_shm(
                        shm_snapshot
                    )
                    removed_shm.extend(removed)
                    if removed:
                        device_info['owned_shm_names'] = [
                            name
                            for name in device_info.get('owned_shm_names') or []
                            if str(name).lstrip('/') not in removed
                        ]
                    # A dead owned worker is not sufficient: a leaked segment could
                    # still collide with the candidate config and must block swap.
                    shm_status = self._probe_shared_memory(device_info)
                    if remaining or shm_status['connected']:
                        device_info['lifecycle_status'] = 'orphaned'
                        device_info['last_error'] = (
                            "Owned process stopped but claimed shared memory "
                            "remains; the segment was not removed"
                        )
                        incomplete_devices.append(key)
                        statuses.append(self._status_for_device(device_info))
                        continue

                    device_info['process'] = None
                    device_info['lifecycle_status'] = 'stopped'
                    if force_killed:
                        device_info['last_error'] = (
                            f"Graceful stop timed out after {stop_timeout:.1f}s; "
                            "the non-motion process was force-killed"
                        )
                    else:
                        device_info['last_error'] = None
                    self._close_ownership_channel(device_info)
                    stopped_devices.append(key)
                    statuses.append(self._status_for_device(device_info))
        finally:
            for operation_lock in reversed(operation_locks):
                operation_lock.release()

        monitor_thread = self._monitor_thread
        if (
            monitor_thread
            and monitor_thread is not threading.current_thread()
            and monitor_thread.is_alive()
        ):
            monitor_thread.join(timeout=2.0)

        incomplete_devices = sorted(set(incomplete_devices))
        external_devices = sorted(set(external_devices))
        stopped_devices = sorted(set(stopped_devices))
        report = {
            'ok': not incomplete_devices,
            'devices': statuses,
            'stopped_devices': stopped_devices,
            'incomplete_devices': incomplete_devices,
            'external_devices': external_devices,
            'removed_shm': sorted(set(removed_shm)),
        }
        if incomplete_devices:
            # Device keys are the only runtime identifiers included in this
            # message.  Config values, provider paths, IPs, and exceptions stay
            # out of the public/controller log boundary.
            logger.error(
                "Device shutdown incomplete for keys: "
                + ", ".join(incomplete_devices)
            )
        else:
            logger.info("All configured device workers stopped")
        return report

    def force_stop_all(self) -> Dict[str, Any]:
        """SIGKILL every panel-owned device child without touching externals.

        This is an explicit operator recovery path. It deliberately bypasses
        robot disable, Yam zero confirmation, recorder flush, and ordinary
        per-device operation locks. Shared memory is removed only when its
        inode still matches a name claimed by the killed worker generation.
        """
        logger.warning(
            "Force-stopping all panel-owned device workers; graceful hardware "
            "shutdown is being bypassed"
        )
        with self._lock:
            for device_info in self.devices:
                self._drain_ownership_events(device_info)
                device_info['desired_state'] = 'stopped'
                if self._process_is_alive(device_info.get('process')):
                    device_info['lifecycle_status'] = 'stopping'
                device_info['progress'] = {
                    'phase': 'force_stop_requested',
                    'updated_at': time.time(),
                }
        self._monitor_wakeup.set()

        killed_pids: List[int] = []
        handled_pids: set[int] = set()
        force_target_keys: set[str] = set()
        snapshots: Dict[str, Dict[str, Tuple[int, int]]] = {
            self._device_key(item): {} for item in self.devices
        }

        # Repeat briefly so a worker whose multiprocessing spawn completes
        # concurrently with this request is also contained.
        for _ in range(3):
            targets: List[Tuple[Dict[str, Any], Any]] = []
            visualizers: List[Any] = []
            with self._lock:
                for device_info in self.devices:
                    self._drain_ownership_events(device_info)
                    key = self._device_key(device_info)
                    process = device_info.get('process')
                    try:
                        process_pid = int(process.pid) if process is not None else None
                    except (AttributeError, TypeError, ValueError):
                        process_pid = None
                    if (
                        self._process_is_alive(process)
                        and (process_pid is None or process_pid not in handled_pids)
                    ):
                        snapshots[key].update(
                            self._owned_shm_snapshot(device_info)
                        )
                        targets.append((device_info, process))
                        force_target_keys.add(key)
                        if process_pid is not None:
                            handled_pids.add(process_pid)
                    visualizer = device_info.get('visualizer_process')
                    if self._process_is_alive(visualizer):
                        visualizers.append(visualizer)

            for visualizer in visualizers:
                stopped, pid = self._force_kill_worker(visualizer)
                if stopped and pid is not None:
                    killed_pids.append(pid)
            for _device_info, process in targets:
                stopped, pid = self._force_kill_worker(process)
                if stopped and pid is not None:
                    killed_pids.append(pid)
            if not targets and not visualizers:
                break
            time.sleep(0.05)

        # Legacy web-disabled manager/policy children are still owned by this
        # lifecycle object and belong in the global recovery action.
        for attribute in ('policy_runner_process', 'device_manager_process'):
            process = getattr(self, attribute, None)
            stopped, pid = self._force_kill_worker(process)
            if stopped:
                setattr(self, attribute, None)
                if pid is not None:
                    killed_pids.append(pid)

        statuses: List[Dict[str, Any]] = []
        forced_devices: List[str] = []
        incomplete_devices: List[str] = []
        external_devices: List[str] = []
        removed_shm: List[str] = []
        with self._lock:
            for device_info in self.devices:
                key = self._device_key(device_info)
                device_info['visualizer_process'] = None
                process = device_info.get('process')
                if self._process_is_alive(process):
                    device_info['lifecycle_status'] = 'stop_timeout'
                    device_info['last_error'] = (
                        'Force stop failed: panel-owned device process is still alive'
                    )
                    incomplete_devices.append(key)
                    statuses.append(self._status_for_device(device_info))
                    continue

                had_owned_process = key in force_target_keys
                removed, remaining = self._unlink_force_killed_shm(
                    snapshots.get(key, {}) if had_owned_process else {}
                )
                removed_shm.extend(removed)
                if removed:
                    owned_names = device_info.get('owned_shm_names') or []
                    device_info['owned_shm_names'] = [
                        name for name in owned_names if str(name).lstrip('/') not in removed
                    ]

                shm_status = self._probe_shared_memory(device_info)
                if not had_owned_process and shm_status['connected']:
                    # A global recovery action must never claim that an external
                    # producer was killed merely because its SHM matches this
                    # config. Keep the resource visible and report it as skipped.
                    device_info['process'] = None
                    device_info['lifecycle_status'] = 'external'
                    device_info['ownership_status'] = 'external'
                    device_info['last_error'] = (
                        'Matching shared memory is externally owned; force stop '
                        'did not signal or unlink it'
                    )
                    device_info['progress'] = {
                        'phase': 'external',
                        'updated_at': time.time(),
                    }
                    self._close_ownership_channel(device_info)
                    external_devices.append(key)
                    statuses.append(self._status_for_device(device_info))
                    continue
                if remaining or shm_status['connected']:
                    device_info['process'] = None
                    device_info['lifecycle_status'] = 'orphaned'
                    device_info['last_error'] = (
                        'Force-killed owned process, but matching shared memory remains'
                    )
                    incomplete_devices.append(key)
                    statuses.append(self._status_for_device(device_info))
                    continue

                if not had_owned_process:
                    # With neither an owned live PID nor matching SHM, this
                    # device was already idle (or held only a reaped process
                    # object). Do not misreport it as force-killed.
                    device_info['last_error'] = None
                    device_info['failure_code'] = None
                    device_info['failure_recoverable'] = None
                else:
                    forced_devices.append(key)
                    device_info['last_error'] = (
                        'Force-stopped by operator; graceful hardware shutdown was bypassed'
                    )
                    device_info['failure_code'] = 'force_stopped'
                    device_info['failure_recoverable'] = True
                device_info['process'] = None
                device_info['lifecycle_status'] = 'stopped'
                device_info['ownership_status'] = (
                    'force_killed' if had_owned_process else 'unclaimed'
                )
                device_info['progress'] = {
                    'phase': 'stopped',
                    'updated_at': time.time(),
                }
                self._close_ownership_channel(device_info)
                statuses.append(self._status_for_device(device_info))

            self._monitor_wakeup.set()

        return {
            'ok': not incomplete_devices,
            'devices': statuses,
            'forced_devices': sorted(set(forced_devices)),
            'incomplete_devices': sorted(set(incomplete_devices)),
            'external_devices': sorted(set(external_devices)),
            'killed_pids': sorted(set(killed_pids)),
            'removed_shm': sorted(set(removed_shm)),
        }

    def _create_reset_signal(self) -> None:
        """Create reset signal file to notify PolicyConnector to reconnect SHMs."""
        try:
            signal_files = self.config.get('runtime', {}).get('signal_files', {})
            reset_signal_path = signal_files.get('reset_signal')
            assert reset_signal_path is not None, "reset_signal_path is required in signal_files"
            reset_signal_file = Path(reset_signal_path)
            reset_signal_file.parent.mkdir(parents=True, exist_ok=True)
            reset_signal_file.touch()  # Update modification time
            logger.info(f"Created reset signal file at {reset_signal_path}")
        except Exception as e:
            logger.warning(f"Failed to create reset signal file: {e}")

    def _write_robot_home_index(self) -> None:
        """Write robot home index to shared file for PolicyConnector to read."""
        try:
            robot_cfg = self.config.get('robot', {})
            home_index = robot_cfg.get('home', 0) if isinstance(robot_cfg, dict) else 0

            signal_files = self.config.get('runtime', {}).get('signal_files', {})
            home_index_path = signal_files.get('robot_home_index')
            assert home_index_path is not None, "home_index_path is required in signal_files"
            home_index_file = Path(home_index_path)
            home_index_file.parent.mkdir(parents=True, exist_ok=True)
            home_index_file.write_text(str(home_index))
            logger.info(f"Wrote robot home index {home_index} to {home_index_path}")
        except Exception as e:
            logger.warning(f"Failed to write robot home index: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all managed devices, visualizers, and Device Manager."""
        with self._lock:
            devices = [self._status_for_device(info) for info in self.devices]
            manager_running = self._process_is_alive(self.device_manager_process)
            policy_running = self._process_is_alive(self.policy_runner_process)
            monitor_running = bool(
                self._monitor_thread and self._monitor_thread.is_alive()
            )
            return {
                'running': any(device['running'] for device in devices),
                'monitor_running': monitor_running,
                'total_devices': len(devices),
                'enable_visualizers': self.enable_visualizers,
                'enable_device_manager': self.enable_device_manager,
                'enable_policy_runner': self.enable_policy_runner,
                'device_manager_running': manager_running,
                'device_manager_pid': (
                    getattr(self.device_manager_process, 'pid', None)
                    if manager_running
                    else None
                ),
                'policy_runner_running': policy_running,
                'policy_runner_pid': (
                    getattr(self.policy_runner_process, 'pid', None)
                    if policy_running
                    else None
                ),
                'devices': devices,
            }


__all__ = [
    'DeviceConflictError',
    'DeviceControlError',
    'DeviceLifecycleManager',
    'DeviceNotFoundError',
    'DeviceStartupError',
]
