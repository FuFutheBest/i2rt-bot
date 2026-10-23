#!/usr/bin/env python3
"""Web control plane for independently managed SuperInference devices.

The server deliberately starts no hardware, manager, or policy process on
boot. A selected runtime can prepare its required devices on operator request.
"""

from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
import re
import secrets
import signal
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from flask import Flask, Response, jsonify, render_template, request, send_file
from utils.hydra_config import DEFAULT_HYDRA_CONFIG_NAME

from .lifecycle import (
    DeviceConflictError,
    DeviceControlError,
    DeviceLifecycleManager,
    DeviceNotFoundError,
)
from .hardware_discovery import (
    CAPTURE_BINDING_STATE_KEY,
    HardwareDiscovery,
    HardwareSnapshot,
    apply_discovery_overrides,
    capture_binding_is_missing,
    recommend_overrides,
)
from .camera_preview import CameraPreviewReader, CameraPreviewResult
from .robot_state_preview import RobotStatePreviewReader, YamStatePreviewReader
from .runtime_config import (
    DEFAULT_CONFIG_DIR,
    RuntimeConfigError,
    compose_runtime_config,
    extract_camera_slots,
    extract_robot_home,
    extract_robot_tools,
    extract_yam_visualization,
    inspect_runtime_config,
    list_runtime_configs,
    normalize_config_name,
    runtime_config_metadata,
    summarize_runtime_config,
    validate_runtime_config,
)
from .workflow import (
    WorkflowConflictError,
    WorkflowError,
    WorkflowLifecycle,
)
from .fastwam_workflow import FastWAMWorkflowLifecycle
from .recording import (
    RECORDING_ACTIONS,
    RecordingConflictError,
    RecordingError,
    RecordingRequestError,
    RecordingSecurityError,
    empty_recording_state,
    normalize_recording_camera_mapping,
    resolve_recording_python,
)
from .conversion_jobs import ConversionJobError, ConversionJobManager
from .recording_exports import RecordingExportError, RecordingExportManager
from .recording_replay import (
    RecordingReplayBuildFailed,
    RecordingReplayBuilding,
    RecordingReplayNotFound,
    RecordingReplayRequestError,
    RecordingReplayService,
    RecordingReplayUnavailable,
)
from .recording_cameras import LOGICAL_CAMERA_IDS
from .recording_paths import (
    configured_recording_root,
    iter_recording_episode_roots,
    latest_sibling_recording_root,
)
from .inference_options import (
    build_control_panel_recording_config,
    default_preferences_path,
    load_rollout_output_root,
    mock_inference_config_error,
    save_rollout_output_root,
)
from .rollout_storage import RolloutStorageMonitor
from .gripper_control import (
    GripperConflictError,
    GripperControlError,
    GripperControlManager,
    GripperRequestError,
    process_command,
    rizon_shared_memory_names,
    serial_port_holders_many,
)
from .gripper_permissions import (
    GripperPermissionError,
    GripperPermissionRequiredError,
    MAX_PASSWORD_BYTES,
    TtyUsbPermissionAuthorizer,
)
from .can_permissions import (
    CAN_INTERFACE_PATTERN,
    CanInterfacePermissionAuthorizer,
    CanPermissionRequiredError,
)
from .state_sync import StateSyncBroker, StateTransitionConflict
from .control_operations import (
    ControlOperationDispatcher,
    ControlOperationDispatcherClosed,
    ControlOperationQueueFull,
)
from .runtime_paths import resolve_device_python
from .process_cleanup import (
    ProcessCleanupError,
    cleanup_before_startup,
    force_cleanup_requested,
)
from .diagnostics import (
    build_preflight_report,
    build_runtime_metrics,
    build_runtime_tuning_schema,
)
from .generated_config_export import (
    GeneratedConfigExportError,
    export_generated_config as write_generated_config,
    persist_source_config as write_source_config,
)


# Keep the no-argument choice identical to device_starter's Hydra default. The
# panel does not register a separate config or a Control Panel-only alias.
DEFAULT_RUNTIME_CONFIG_NAME = DEFAULT_HYDRA_CONFIG_NAME
GLOBAL_MAX_REQUEST_BYTES = 1024 * 1024
MAX_GRIPPER_PASSWORD_BODY_BYTES = MAX_PASSWORD_BYTES * 4
DEFAULT_CHECKPOINT_BROWSE_ROOTS = (
    Path('/home'),
    Path('/mnt'),
    Path('/data'),
    Path('/models'),
)
MAX_CHECKPOINT_PATH_LENGTH = 4096
MAX_CHECKPOINT_BROWSE_ENTRIES = 500
MAX_CHECKPOINT_BROWSE_DEPTH = 32
RECORDING_REPLAY_ROOTS_ENV = 'SUPERINFERENCE_RECORDING_REPLAY_ROOTS'
EVENT_STREAM_CLIENT_COOKIE = 'superinference_event_client'
EVENT_STREAM_CLIENT_PATTERN = re.compile(r'^[A-Za-z0-9_-]{16,128}$')
EVENT_STREAM_COOKIE_MAX_AGE_S = 30 * 24 * 60 * 60
PREVIEW_BROWSER_CLIENT_VERSION = '20260722-short-preview5'
CONTROL_PANEL_YAM_DISCONNECT_ZERO_JOINT_COUNT = 5
CONTROL_PANEL_YAM_DISCONNECT_ZERO_TOLERANCE_RAD = 0.02
RUNTIME_DEVICE_READY_TIMEOUT_S = 90.0
RUNTIME_DEVICE_POLL_INTERVAL_S = 0.1
RUNTIME_START_TRANSITION_TIMEOUT_MS = 120_000
SERVER_START_TRANSITION_TIMEOUT_MS = 600_000
WORKFLOW_OPERATION_KEY = '__workflow__'
SERVER_WORKFLOW_OPERATION_KEY = '__workflow_server__'
FORCE_STOP_ALL_CONFIRMATION = 'FORCE_STOP_ALL'

# Config metadata is requested during every panel refresh, while composing a
# Hydra config can be comparatively expensive. Keep only the config-independent
# picker payload here; request-specific selection state is applied on return.
_CONFIG_ITEMS_CACHE_LOCK = threading.RLock()
_CONFIG_ITEMS_CACHE: Dict[
    str,
    Tuple[
        Tuple[Tuple[str, Any, Any, Any], ...],
        list[Dict[str, Any]],
    ],
] = {}


@dataclass
class _PreviewReaderSlot:
    """One cached SHM reader and its per-device serialization lock."""

    reader: Any
    lock: threading.Lock


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConfigSwitchConflictError(DeviceConflictError):
    """A config switch stopped some state but could not safely commit."""

    def __init__(
        self,
        message: str,
        *,
        selected_config: Optional[str],
        stopped_devices: list[str],
        incomplete_devices: list[str],
        external_devices: Optional[list[str]] = None,
    ) -> None:
        super().__init__(message)
        self.selected_config = selected_config
        self.stopped_devices = sorted(set(stopped_devices))
        self.incomplete_devices = sorted(set(incomplete_devices))
        self.external_devices = sorted(set(external_devices or []))
        self.stopped_previous = bool(self.stopped_devices)


class BootstrapConfigConflictError(DeviceConflictError):
    """A one-shot bootstrap selection lost its compare-and-swap race."""

    def __init__(
        self,
        message: str,
        *,
        expected_revision: int,
        current_revision: int,
        expected_selected_config: Optional[str],
        current_selected_config: Optional[str],
    ) -> None:
        super().__init__(message)
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        self.expected_selected_config = expected_selected_config
        self.current_selected_config = current_selected_config


def recommended_runtime_config(names: list[str]) -> Optional[str]:
    """Choose the deterministic cold-start config without composing files."""
    ordered = sorted(set(names), key=lambda name: (name.casefold(), name))
    if not ordered:
        return None
    if DEFAULT_RUNTIME_CONFIG_NAME in ordered:
        return DEFAULT_RUNTIME_CONFIG_NAME

    dual_pi05 = [
        name
        for name in ordered
        if 'dual' in name.casefold() and 'pi05' in name.casefold()
    ]
    if dual_pi05:
        return dual_pi05[0]

    return ordered[0]


def _config_tree_fingerprint(
    config_dir: Path,
) -> Tuple[Tuple[str, Any, Any, Any], ...]:
    """Return a cheap fingerprint for YAML inputs used by Hydra composition."""
    root = config_dir.resolve()
    entries: list[Tuple[str, Any, Any, Any]] = []
    try:
        root_stat = root.stat()
        entries.append((".", root_stat.st_mtime_ns, root_stat.st_size, root_stat.st_ino))
    except OSError:
        return ((".", None, None, None),)

    try:
        paths = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            ),
            key=lambda path: str(path.relative_to(root)),
        )
    except OSError as exc:
        return (("<scan-error>", type(exc).__name__, None, None),)

    for path in paths:
        relative = str(path.relative_to(root))
        try:
            stat_result = path.stat()
        except OSError as exc:
            entries.append((relative, type(exc).__name__, None, None))
            continue
        entries.append(
            (relative, stat_result.st_mtime_ns, stat_result.st_size, stat_result.st_ino)
        )
    return tuple(entries)


def control_panel_config_items(
    config_dir: Path,
    *,
    selected: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Build the picker payload and omit workflows this panel cannot run."""
    root = Path(config_dir).resolve()
    if not root.is_dir():
        # Preserve the existing public error for a missing config directory.
        list_runtime_configs(root)
    fingerprint = _config_tree_fingerprint(root)
    cache_key = str(root)

    with _CONFIG_ITEMS_CACHE_LOCK:
        cached = _CONFIG_ITEMS_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            items = copy.deepcopy(cached[1])
        else:
            items: list[Dict[str, Any]] = []
            for name in list_runtime_configs(root):
                errors: list[str] = []
                config: Optional[Mapping[str, Any]] = None
                try:
                    config = compose_runtime_config(name, config_dir=root)
                    errors = validate_runtime_config(config)
                except RuntimeConfigError as exc:
                    errors = [str(exc)]
                metadata = runtime_config_metadata(name, config, errors=errors)
                if config is None:
                    # A failed Hydra composition cannot be safely classified from an
                    # absent policy node. Hide it from the actionable picker rather
                    # than treating it as a device workflow by default; direct
                    # inspect still exposes the composition error for repair.
                    metadata["control_panel_supported"] = False
                    metadata["support_reason"] = (
                        "configuration could not be composed: "
                        + (errors[0] if errors else "unknown composition error")
                    )
                if not metadata["control_panel_supported"]:
                    continue
                items.append(
                    {
                        "name": name,
                        "label": metadata.get("label", name),
                        "valid": not errors,
                        "error": "; ".join(errors),
                        **metadata,
                    }
                )

            post_fingerprint = _config_tree_fingerprint(root)
            scan_failed = bool(fingerprint and fingerprint[0][0] == "<scan-error>")
            if post_fingerprint == fingerprint and not scan_failed:
                _CONFIG_ITEMS_CACHE[cache_key] = (
                    fingerprint,
                    copy.deepcopy(items),
                )
            else:
                _CONFIG_ITEMS_CACHE.pop(cache_key, None)

    for item in items:
        item["selected"] = item["name"] == selected
    return items


class LogStore:
    """Small cursor-addressable in-memory log used by the control UI."""

    def __init__(self, max_items: int = 2000) -> None:
        self._items: Deque[Dict[str, Any]] = deque(maxlen=max_items)
        self._sequence = 0
        self._lock = threading.Lock()

    def append(
        self, level: str, message: str, *, device_key: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._lock:
            self._sequence += 1
            item = {
                'seq': self._sequence,
                'ts': _utc_timestamp(),
                'level': str(level).upper(),
                'message': str(message),
                'device_key': device_key,
            }
            self._items.append(item)
            return item

    def query(
        self,
        *,
        cursor: int = 0,
        limit: int = 250,
        device_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            items = [item for item in self._items if item['seq'] > cursor]
            if device_key:
                items = [
                    item
                    for item in items
                    if item.get('device_key') in (None, device_key)
                    and (
                        item.get('device_key') == device_key
                        or device_key in item['message']
                    )
                ]
            items = items[: max(1, min(limit, 1000))]
            next_cursor = items[-1]['seq'] if items else max(cursor, self._sequence)
            return {'items': items, 'next_cursor': next_cursor}


class DeviceController:
    """Own the one lifecycle manager associated with the selected config."""

    ROBOT_STATE_COALESCE_S = 0.05

    def _recording_repo_root(self) -> Path:
        """Resolve relative rollout paths beside the configured ``configs`` dir."""
        config_dir = getattr(self, 'config_dir', None)
        if config_dir is None:
            # A few embedding tests construct a controller with ``__new__``;
            # use the actual repository root for those minimal objects.
            return Path(__file__).resolve().parents[2]
        return Path(config_dir).parent.resolve()

    def __init__(
        self,
        config_dir: Path = DEFAULT_CONFIG_DIR,
        *,
        preview_reader_factory: Optional[Callable[[str, int], Any]] = None,
        robot_state_reader_factory: Optional[Callable[[int], Any]] = None,
        yam_state_reader_factory: Optional[Callable[[int], Any]] = None,
        hardware_discovery: Optional[HardwareDiscovery] = None,
        workflow_factory: Optional[Callable[..., WorkflowLifecycle]] = None,
        device_python: Optional[Path] = None,
        recording_python: Optional[Path] = None,
        conversion_jobs: Optional[ConversionJobManager] = None,
        recording_exports: Optional[RecordingExportManager] = None,
        gripper_control: Optional[GripperControlManager] = None,
        gripper_permissions: Optional[TtyUsbPermissionAuthorizer] = None,
        can_permissions: Optional[CanInterfacePermissionAuthorizer] = None,
        checkpoint_browse_roots: Optional[Sequence[Path]] = None,
        inference_preferences_path: Optional[Path] = None,
        rollout_storage: Optional[RolloutStorageMonitor] = None,
    ) -> None:
        self.config_dir = config_dir.resolve()
        self.selected_config: Optional[str] = None
        self.starter: Optional[DeviceLifecycleManager] = None
        self.workflow: Optional[WorkflowLifecycle] = None
        self.revision = 0
        self.started_at = _utc_timestamp()
        self.logs = LogStore()
        self._lock = threading.RLock()
        self._device_operation_condition = threading.Condition(self._lock)
        self._device_operations: Dict[str, str] = {}
        self._workflow_cancelled_device_keys: set[str] = set()
        self._workflow_cancel_event = threading.Event()
        self._lifecycle_mutation_pending = False
        self._force_stop_lock = threading.Lock()
        self._force_stop_in_progress = False
        # Serializes config/binding mutation against a privileged ttyUSB/CAN
        # permission reservation. State reads and safety-reducing stop paths
        # remain responsive during a sudo timeout.
        self._gripper_permission_context_lock = threading.Lock()
        self._preview_reader_factory = (
            preview_reader_factory or CameraPreviewReader
        )
        self._preview_readers: Dict[Tuple[str, str], _PreviewReaderSlot] = {}
        self._robot_state_reader_factory = (
            robot_state_reader_factory or RobotStatePreviewReader
        )
        self._robot_state_readers: Dict[str, _PreviewReaderSlot] = {}
        self._yam_state_reader_factory = (
            yam_state_reader_factory or YamStatePreviewReader
        )
        self._yam_state_readers: Dict[str, _PreviewReaderSlot] = {}
        self._robot_state_cache_lock = threading.Lock()
        self._robot_state_cache: Optional[Dict[str, Any]] = None
        self._robot_state_cache_key: Optional[Tuple[int, Optional[str]]] = None
        self._robot_state_cache_at = 0.0
        self.hardware_discovery = hardware_discovery or HardwareDiscovery()
        # Explicit factories are test/embedding injection points. The default
        # factory is selected per config in _make_workflow().
        self._workflow_factory = workflow_factory
        self.recording_python = (
            Path(recording_python) if recording_python is not None else None
        )
        self.device_python = resolve_device_python(device_python)
        self.gripper_permissions = (
            gripper_permissions or TtyUsbPermissionAuthorizer()
        )
        self.can_permissions = (
            can_permissions or CanInterfacePermissionAuthorizer()
        )
        revision_binding = getattr(
            self.gripper_permissions, 'set_revision_provider', None
        )
        if callable(revision_binding):
            revision_binding(lambda: self.revision)
        self.grippers = gripper_control or GripperControlManager(
            log_callback=lambda level, message: self.logs.append(level, message),
            device_python=self.device_python,
        )
        permission_binding = getattr(
            self.grippers, 'set_permission_authorization', None
        )
        if callable(permission_binding):
            lease_check = getattr(
                self.gripper_permissions, 'lease_allows', None
            )
            lease_revoke = getattr(
                self.gripper_permissions, 'revoke_leases', None
            )
            lease_verify = getattr(
                self.gripper_permissions,
                'verify_lease_for_connection',
                None,
            )
            permission_binding(
                lease_check=lease_check if callable(lease_check) else None,
                lease_revoke=(
                    lease_revoke if callable(lease_revoke) else None
                ),
                lease_verify=(
                    lease_verify if callable(lease_verify) else None
                ),
            )
        self._owns_conversion_jobs = conversion_jobs is None
        # Conversion is intentionally a constrained background worker (nice,
        # ionice and cgroup limits are applied by ConversionJobManager).  Do
        # not gate it on an active runtime/inference session: rollout normally
        # keeps those processes alive between episodes, so such a gate prevents
        # per-episode shards from ever being prepared during a recording run.
        self.conversion_jobs = conversion_jobs or ConversionJobManager()
        self._recording_finalized_enqueue: Optional[
            Callable[[Mapping[str, Any]], Any]
        ] = None
        self._base_config: Optional[Dict[str, Any]] = None
        self._effective_config: Optional[Dict[str, Any]] = None
        self._fallback_recording_config: Optional[Dict[str, Any]] = None
        self._fallback_recording_config_loaded = False
        self._effective_bindings: Dict[str, Dict[str, Any]] = {}
        self._gripper_port_overrides: Dict[str, str] = {}
        self._checkpoint_override: Optional[Path] = None
        self._checkpoint_generation = 0
        self._checkpoint_browse_roots = self._resolve_checkpoint_roots(
            checkpoint_browse_roots
        )
        self._inference_test_mode = False
        self._rollout_enabled = False
        self._rollout_settings_dirty = False
        self._rollout_output_root_user_set = False
        self._inference_options_generation = 0
        self._inference_preferences_path = Path(
            inference_preferences_path or default_preferences_path()
        ).expanduser().resolve(strict=False)
        self._rollout_output_root = load_rollout_output_root(
            self._inference_preferences_path
        )
        self.rollout_storage = rollout_storage or RolloutStorageMonitor()
        self._binding_discovery_revision: Optional[str] = None
        self._discovery_snapshot_cache: Dict[
            str, Tuple[float, HardwareSnapshot]
        ] = {}
        self._shutdown = False
        self._shutdown_complete = False
        self._owns_recording_exports = recording_exports is None
        self.recording_exports = recording_exports or RecordingExportManager(
            self.recording_source_root,
            self.conversion_jobs,
        )

    def _bump_revision(self) -> int:
        revoke = getattr(
            getattr(self, 'gripper_permissions', None),
            'revoke_leases',
            None,
        )
        if callable(revoke):
            try:
                revoke()
            except Exception as exc:
                self.logs.append(
                    'ERROR',
                    'Failed to invalidate ttyUSB permission authorization: '
                    f'{type(exc).__name__}: {exc}',
                )
        self.revision += 1
        return self.revision

    @contextmanager
    def _lifecycle_mutation_guard_locked(self) -> Iterator[None]:
        """Serialize manager replacement against in-flight device actions."""
        self._ensure_device_operation_state_locked()
        while self._lifecycle_mutation_pending:
            self._device_operation_condition.wait()
        self._lifecycle_mutation_pending = True
        try:
            while self._device_operations:
                # The condition releases the controller lock while waiting, so
                # the authoritative state broker remains responsive.
                self._device_operation_condition.wait()
            yield
        finally:
            self._lifecycle_mutation_pending = False
            self._device_operation_condition.notify_all()

    def _ensure_device_operation_state_locked(self) -> None:
        """Initialize concurrency state for legacy/test-constructed controllers."""
        if not hasattr(self, '_device_operation_condition'):
            self._device_operation_condition = threading.Condition(self._lock)
        if not hasattr(self, '_device_operations'):
            self._device_operations = {}
        if not hasattr(self, '_workflow_cancelled_device_keys'):
            self._workflow_cancelled_device_keys = set()
        if not hasattr(self, '_workflow_cancel_event'):
            self._workflow_cancel_event = threading.Event()
        if not hasattr(self, '_lifecycle_mutation_pending'):
            self._lifecycle_mutation_pending = False
        if not hasattr(self, '_force_stop_lock'):
            self._force_stop_lock = threading.Lock()
        if not hasattr(self, '_force_stop_in_progress'):
            self._force_stop_in_progress = False

    @staticmethod
    def _workflow_operation_key(action: str = '') -> str:
        return (
            SERVER_WORKFLOW_OPERATION_KEY
            if str(action).startswith('server:')
            else WORKFLOW_OPERATION_KEY
        )

    def _workflow_operation_active_locked(self, action: str = '') -> bool:
        self._ensure_device_operation_state_locked()
        return self._workflow_operation_key(action) in self._device_operations

    def _begin_workflow_operation_locked(self, action: str) -> None:
        """Reserve one workflow transition without monopolizing state reads."""

        self._ensure_device_operation_state_locked()
        if self._force_stop_in_progress:
            raise DeviceConflictError('Force stop is in progress')
        if self._lifecycle_mutation_pending:
            raise DeviceConflictError(
                'Config or lifecycle mutation is in progress; retry the workflow action'
            )
        operation_key = self._workflow_operation_key(action)
        # A FastWAM model Server owns only model memory and its TCP endpoint;
        # it may load in parallel with the motion runtime. Runtime/inference,
        # device, and recording operations continue to share the main workflow
        # reservation because they touch common SHM and lifecycle state.
        blocking_operations = {
            key: value
            for key, value in self._device_operations.items()
            if (
                key == operation_key
                or operation_key != SERVER_WORKFLOW_OPERATION_KEY
                and key != SERVER_WORKFLOW_OPERATION_KEY
            )
        }
        if blocking_operations:
            active = ', '.join(
                f'{key} ({value})'
                for key, value in sorted(blocking_operations.items())
            )
            raise DeviceConflictError(
                'Another device or workflow action is in progress: ' + active
            )
        if operation_key == WORKFLOW_OPERATION_KEY:
            if str(action).endswith(':stop'):
                self._workflow_cancel_event.clear()
            self._workflow_cancelled_device_keys.clear()
        self._device_operations[operation_key] = action

    def _end_workflow_operation_locked(self, action: str = '') -> None:
        operation_key = self._workflow_operation_key(action)
        if operation_key == WORKFLOW_OPERATION_KEY:
            self._workflow_cancelled_device_keys.clear()
            self._workflow_cancel_event.clear()
        self._end_device_operation_locked(operation_key)

    def request_async_stop(self, scope: str) -> None:
        """Signal an in-flight start without waiting for the controller lock."""

        normalized = str(scope)
        if normalized.startswith('workflow:'):
            if normalized == 'workflow:server':
                # Server startup has its own reservation and cancellation hook;
                # do not cancel a concurrently loading motion runtime.
                workflow = getattr(self, 'workflow', None)
                request_stop = getattr(
                    workflow, 'request_async_server_stop', None
                )
                if callable(request_stop):
                    request_stop()
                return
            self._workflow_cancel_event.set()
            return
        if normalized.startswith('device:'):
            device_key = normalized[len('device:') :]
            starter = getattr(self, 'starter', None)
            request_stop = getattr(starter, 'request_stop_device', None)
            if device_key and callable(request_stop):
                request_stop(device_key)
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._ensure_device_operation_state_locked()
            if normalized.startswith('device:') and self._workflow_operation_active_locked():
                key = normalized[len('device:') :]
                if key:
                    self._workflow_cancelled_device_keys.add(key)
        finally:
            self._lock.release()

    def _raise_if_workflow_cancelled_locked(self) -> None:
        self._ensure_device_operation_state_locked()
        if self._workflow_cancel_event.is_set():
            raise WorkflowConflictError(
                'Workflow start was superseded by a stop request'
            )

    def _begin_device_operation_locked(
        self,
        key: str,
        action: str,
        *,
        allow_workflow: bool = False,
    ) -> None:
        self._ensure_device_operation_state_locked()
        if self._force_stop_in_progress:
            raise DeviceConflictError('Force stop is in progress')
        if self._lifecycle_mutation_pending:
            raise DeviceConflictError(
                'Config or lifecycle mutation is in progress; retry the device action'
            )
        workflow_action = self._device_operations.get(WORKFLOW_OPERATION_KEY)
        if workflow_action is not None and not allow_workflow and action != 'stop':
            raise DeviceConflictError(
                'Workflow action is in progress; only an emergency device stop '
                f'is allowed until it finishes: {workflow_action}'
            )
        if key in self._device_operations:
            raise DeviceConflictError(f'Device action already in progress for {key}')
        self._device_operations[key] = action

    def _end_device_operation_locked(self, key: str) -> None:
        self._ensure_device_operation_state_locked()
        self._device_operations.pop(key, None)
        self._device_operation_condition.notify_all()

    @contextmanager
    def _gripper_permission_mutation_guard(self) -> Iterator[None]:
        """Fail fast instead of waiting behind a privileged reservation."""
        if not self._gripper_permission_context_lock.acquire(blocking=False):
            raise DeviceConflictError(
                '设备权限授权正在进行；请稍后重试配置或绑定变更'
            )
        try:
            yield
        finally:
            self._gripper_permission_context_lock.release()

    def selected_config_name(self) -> Optional[str]:
        with self._lock:
            return self.selected_config

    def _resolve_checkpoint_roots(
        self, roots: Optional[Sequence[Path]]
    ) -> tuple[Path, ...]:
        candidates = roots if roots is not None else DEFAULT_CHECKPOINT_BROWSE_ROOTS
        resolved: list[Path] = []
        for candidate in candidates:
            try:
                root = Path(candidate).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if root.is_dir() and root not in resolved:
                resolved.append(root)
        if not resolved:
            fallback = self.config_dir.parent.resolve(strict=False)
            if fallback.is_dir():
                resolved.append(fallback)
        return tuple(resolved)

    @staticmethod
    def _path_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _checkpoint_root_for(self, path: Path) -> Optional[Path]:
        return next(
            (
                root
                for root in self._checkpoint_browse_roots
                if self._path_within(path, root)
            ),
            None,
        )

    def _checkpoint_requires_params_locked(self) -> bool:
        workflow = self.workflow
        return bool(
            getattr(workflow, 'checkpoint_requires_params', True)
            if workflow is not None
            else True
        )

    def _checkpoint_accepts_file_locked(self) -> bool:
        workflow = self.workflow
        return bool(
            getattr(workflow, 'checkpoint_accepts_file', False)
            if workflow is not None
            else False
        )

    def _checkpoint_owner_stage_locked(self) -> str:
        workflow = self.workflow
        owner = getattr(workflow, 'checkpoint_owner_stage', 'inference')
        return str(owner or 'inference')

    @staticmethod
    def _natural_path_key(value: str) -> tuple[Any, ...]:
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.split(r'(\d+)', value)
            if part
        )

    def _canonical_checkpoint_path(
        self,
        raw_path: Any,
        *,
        require_directory: bool,
        allow_file: bool = False,
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeConfigError('checkpoint path is required')
        value = raw_path.strip()
        if (
            len(value) > MAX_CHECKPOINT_PATH_LENGTH
            or any(ord(character) < 32 for character in value)
        ):
            raise RuntimeConfigError('checkpoint path is invalid')
        try:
            lexical = Path(value).expanduser()
        except RuntimeError as exc:
            raise RuntimeConfigError('checkpoint path is invalid') from exc
        if '..' in lexical.parts:
            raise RuntimeConfigError('checkpoint path must not contain ..')
        if not lexical.is_absolute():
            raise RuntimeConfigError('checkpoint path must be absolute')
        try:
            canonical = lexical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeConfigError(
                f'checkpoint path does not exist: {value}'
            ) from exc
        root = self._checkpoint_root_for(canonical)
        if root is None:
            raise RuntimeConfigError(
                'checkpoint path is outside the allowed browse roots'
            )
        if len(canonical.relative_to(root).parts) > MAX_CHECKPOINT_BROWSE_DEPTH:
            raise RuntimeConfigError('checkpoint path is too deeply nested')
        if canonical.is_file():
            if not allow_file:
                raise RuntimeConfigError('checkpoint path must be a directory')
            if canonical.suffix.casefold() != '.pt':
                raise RuntimeConfigError(
                    'checkpoint file must have a .pt suffix'
                )
            if not os.access(canonical, os.R_OK):
                raise RuntimeConfigError('checkpoint file is not readable')
            return canonical
        if not canonical.is_dir():
            raise RuntimeConfigError('checkpoint path must be a directory')
        if not os.access(canonical, os.R_OK | os.X_OK):
            raise RuntimeConfigError('checkpoint path is not readable')
        if not require_directory:
            params_path = canonical / 'params'
            if (
                params_path.is_symlink()
                or not params_path.is_dir()
                or not os.access(params_path, os.R_OK | os.X_OK)
            ):
                raise RuntimeConfigError(
                    'checkpoint directory must contain a readable params directory'
                )
        return canonical

    def _checkpoint_mutation_blockers_locked(self) -> list[str]:
        blockers: list[str] = []
        if (
            self.selected_config is None
            or self._effective_config is None
            or self.workflow is None
        ):
            blockers.append('Select a config before editing checkpoint')
        else:
            workflow = self.workflow
            assert workflow is not None
            blockers.extend(workflow.checkpoint_override_blockers())
        return blockers

    def _validate_checkpoint_context_locked(
        self,
        *,
        expected_selected_config: str,
        expected_revision: int,
        expected_generation: int,
        require_editable: bool,
    ) -> None:
        if (
            self.selected_config != expected_selected_config
            or self.revision != expected_revision
            or self._checkpoint_generation != expected_generation
        ):
            raise DeviceConflictError(
                'Checkpoint context changed; refresh state and retry'
            )
        if require_editable:
            blockers = self._checkpoint_mutation_blockers_locked()
            if blockers:
                raise DeviceConflictError('; '.join(blockers))

    def _checkpoint_state_locked(
        self, workflow_state: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        checkpoint: Optional[Dict[str, Any]] = None
        if isinstance(workflow_state, Mapping):
            inference_state = workflow_state.get('inference')
            if isinstance(inference_state, Mapping):
                observed_checkpoint = inference_state.get('checkpoint')
                if isinstance(observed_checkpoint, Mapping):
                    # Workflow observations carry the same checkpoint payload.
                    # Reuse it so /api/state never waits behind a long workflow
                    # start/stop operation merely to reacquire the workflow lock.
                    checkpoint = copy.deepcopy(dict(observed_checkpoint))
        if self.workflow is not None:
            if checkpoint is None:
                checkpoint = self.workflow.checkpoint_state()
            override_path = checkpoint.get('override_path')
            if isinstance(override_path, str) and override_path:
                try:
                    canonical_override = self._canonical_checkpoint_path(
                        override_path,
                        require_directory=not self._checkpoint_requires_params_locked(),
                        allow_file=self._checkpoint_accepts_file_locked(),
                    )
                    if canonical_override != Path(override_path):
                        raise DeviceConflictError(
                            'Checkpoint override identity changed; select it again'
                        )
                except (RuntimeConfigError, DeviceConflictError) as exc:
                    checkpoint['valid'] = False
                    checkpoint['error'] = str(exc)
                    checkpoint['path_error'] = str(exc)
        else:
            checkpoint = {
                'default_path': None,
                'override_path': None,
                'effective_path': None,
                'override_active': False,
                'exists': False,
                'params_exists': False,
                'valid': False,
                'error': 'Select a config before editing checkpoint',
            }
        blockers: list[str] = []
        if self.selected_config is None or self.workflow is None:
            blockers.append('Select a config before editing checkpoint')
        owner_stage = self._checkpoint_owner_stage_locked()
        if isinstance(workflow_state, Mapping):
            owner = workflow_state.get(owner_stage)
            if isinstance(owner, Mapping) and (
                owner.get('running') is True
                or owner.get('desired_state') == 'running'
                or owner.get('status') in {'starting', 'running', 'stopping'}
            ):
                blockers.append(
                    f'Stop {owner_stage} before editing checkpoint'
                )
        checkpoint.update(
            {
                'config_name': self.selected_config,
                'generation': self._checkpoint_generation,
                'can_edit': not blockers,
                'editable': not blockers,
                'blockers': blockers,
                'path_error': checkpoint.get('error'),
                'owner_stage': owner_stage,
            }
        )
        return checkpoint

    def browse_checkpoints(self, raw_path: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            config_name = self.selected_config
            generation = self._checkpoint_generation
            revision = self.revision
            checkpoint = (
                self.workflow.checkpoint_state()
                if self.workflow is not None
                else {}
            )
            roots = list(self._checkpoint_browse_roots)
            requires_params = self._checkpoint_requires_params_locked()
        if not roots:
            raise RuntimeConfigError('no checkpoint browse roots are available')
        if raw_path is None or not str(raw_path).strip():
            candidate = checkpoint.get('effective_path')
            try:
                target = Path(str(candidate)).expanduser().resolve(strict=True)
                if target.is_file():
                    target = target.parent
                elif target.is_dir() and self._checkpoint_root_for(target.parent) is not None:
                    target = target.parent
                if not target.is_dir() or self._checkpoint_root_for(target) is None:
                    raise OSError
            except (OSError, RuntimeError, TypeError, ValueError):
                target = roots[0]
        else:
            requested = self._canonical_checkpoint_path(
                raw_path,
                require_directory=True,
                allow_file=self._checkpoint_accepts_file_locked(),
            )
            # Browsing is directory-based.  If a FastWAM user pastes a
            # concrete weight file, show its parent so the file appears as a
            # selectable entry in the same response.
            target = requested.parent if requested.is_file() else requested
        target = self._canonical_checkpoint_path(
            str(target), require_directory=True,
        )
        root = self._checkpoint_root_for(target)
        if root is None:
            raise RuntimeConfigError(
                'checkpoint browse path is outside the allowed roots'
            )
        entries: list[Dict[str, Any]] = []
        try:
            iterator = os.scandir(target)
            with iterator:
                for entry in iterator:
                    try:
                        canonical = Path(entry.path).resolve(strict=True)
                    except (OSError, RuntimeError):
                        continue
                    if self._checkpoint_root_for(canonical) is None:
                        continue
                    is_dir = canonical.is_dir()
                    is_file = canonical.is_file()
                    if not (is_dir or is_file):
                        continue
                    mode = os.R_OK | (os.X_OK if is_dir else 0)
                    if not os.access(canonical, mode):
                        continue
                    selectable = bool(
                        (
                            is_file
                            and self._checkpoint_accepts_file_locked()
                            and canonical.suffix.casefold() == '.pt'
                        )
                        or (
                            is_dir
                            and (
                                not requires_params
                                or (
                                    not (canonical / 'params').is_symlink()
                                    and (canonical / 'params').is_dir()
                                    and os.access(
                                        canonical / 'params', os.R_OK | os.X_OK
                                    )
                                )
                            )
                        )
                    )
                    entries.append(
                        {
                            'name': entry.name,
                            'path': str(canonical),
                            'kind': 'directory' if is_dir else 'file',
                            'is_symlink': entry.is_symlink(),
                            'navigable': is_dir,
                            'selectable': selectable,
                        }
                    )
        except OSError as exc:
            raise RuntimeConfigError(
                f'unable to browse checkpoint directory: {target}'
            ) from exc
        entries.sort(
            key=lambda item: (
                0 if item['kind'] == 'directory' else 1,
                self._natural_path_key(str(item['name'])),
                str(item['name']),
            )
        )
        truncated = len(entries) > MAX_CHECKPOINT_BROWSE_ENTRIES
        entries = entries[:MAX_CHECKPOINT_BROWSE_ENTRIES]
        parent = target.parent
        parent_path = (
            str(parent)
            if target != root and self._checkpoint_root_for(parent) is not None
            else None
        )
        return {
            'config_name': config_name,
            'revision': revision,
            'generation': generation,
            'path': str(target),
            'root': str(root),
            'roots': [str(item) for item in roots],
            'parent': parent_path,
            'current_selectable': bool(
                not requires_params
                or (
                    not (target / 'params').is_symlink()
                    and (target / 'params').is_dir()
                    and os.access(target / 'params', os.R_OK | os.X_OK)
                )
            ),
            'requires_params': requires_params,
            'accepts_files': self._checkpoint_accepts_file_locked(),
            'entries': entries,
            'truncated': truncated,
        }

    def set_checkpoint_override(
        self,
        raw_path: Any,
        *,
        expected_selected_config: str,
        expected_revision: int,
        expected_generation: int,
    ) -> Dict[str, Any]:
        with self._lock:
            self._validate_checkpoint_context_locked(
                expected_selected_config=expected_selected_config,
                expected_revision=expected_revision,
                expected_generation=expected_generation,
                require_editable=True,
            )
            canonical = self._canonical_checkpoint_path(
                raw_path,
                require_directory=not self._checkpoint_requires_params_locked(),
                allow_file=self._checkpoint_accepts_file_locked(),
            )
            if self._checkpoint_override == canonical:
                return self.state()
            workflow = self.workflow
            assert workflow is not None
            workflow.set_checkpoint_override(canonical)
            self._checkpoint_override = canonical
            self._checkpoint_generation += 1
            self._bump_revision()
            self.logs.append('INFO', 'Checkpoint override selected for this session')
            return self.state()

    def clear_checkpoint_override(
        self,
        *,
        expected_selected_config: str,
        expected_revision: int,
        expected_generation: int,
    ) -> Dict[str, Any]:
        with self._lock:
            self._validate_checkpoint_context_locked(
                expected_selected_config=expected_selected_config,
                expected_revision=expected_revision,
                expected_generation=expected_generation,
                require_editable=True,
            )
            if self._checkpoint_override is None:
                return self.state()
            workflow = self.workflow
            assert workflow is not None
            workflow.set_checkpoint_override(None)
            self._checkpoint_override = None
            self._checkpoint_generation += 1
            self._bump_revision()
            self.logs.append('INFO', 'Checkpoint override cleared for this session')
            return self.state()

    def _canonical_rollout_path(
        self,
        raw_path: Any,
        *,
        require_writable: bool,
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeConfigError('rollout save path is required')
        value = raw_path.strip()
        if (
            len(value) > MAX_CHECKPOINT_PATH_LENGTH
            or any(ord(character) < 32 for character in value)
        ):
            raise RuntimeConfigError('rollout save path is invalid')
        try:
            lexical = Path(value).expanduser()
        except RuntimeError as exc:
            raise RuntimeConfigError('rollout save path is invalid') from exc
        if '..' in lexical.parts:
            raise RuntimeConfigError('rollout save path must not contain ..')
        if not lexical.is_absolute():
            raise RuntimeConfigError('rollout save path must be absolute')
        try:
            canonical = lexical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeConfigError(
                f'rollout save path does not exist: {value}'
            ) from exc
        root = self._checkpoint_root_for(canonical)
        if root is None:
            raise RuntimeConfigError(
                'rollout save path is outside the allowed browse roots'
            )
        if len(canonical.relative_to(root).parts) > MAX_CHECKPOINT_BROWSE_DEPTH:
            raise RuntimeConfigError('rollout save path is too deeply nested')
        if not canonical.is_dir():
            raise RuntimeConfigError('rollout save path must be a directory')
        mode = os.R_OK | os.X_OK | (os.W_OK if require_writable else 0)
        if not os.access(canonical, mode):
            requirement = 'readable and writable' if require_writable else 'readable'
            raise RuntimeConfigError(
                f'rollout save path must be {requirement}'
            )
        return canonical

    def _inference_options_mutation_blockers_locked(self) -> list[str]:
        if (
            self.selected_config is None
            or self._effective_config is None
            or self.workflow is None
        ):
            return ['Select a config before editing inference options']
        return list(self.workflow.inference_option_blockers())

    def _validate_inference_options_context_locked(
        self,
        *,
        expected_selected_config: str,
        expected_revision: int,
        expected_generation: int,
        require_editable: bool,
    ) -> None:
        if (
            self.selected_config != expected_selected_config
            or self.revision != expected_revision
            or self._inference_options_generation != expected_generation
        ):
            raise DeviceConflictError(
                'Inference options changed; refresh state and retry'
            )
        if require_editable:
            blockers = self._inference_options_mutation_blockers_locked()
            if blockers:
                raise DeviceConflictError('; '.join(blockers))

    def _inference_options_state_locked(
        self,
        workflow_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        blockers: list[str] = []
        if self.selected_config is None or self.workflow is None:
            blockers.append('Select a config before editing inference options')
        elif isinstance(workflow_state, Mapping):
            inference = workflow_state.get('inference')
            if isinstance(inference, Mapping) and (
                inference.get('running') is True
                or inference.get('desired_state') == 'running'
                or inference.get('status') in {'starting', 'running', 'stopping'}
            ):
                blockers.append('Stop inference before editing inference options')
        else:
            blockers.extend(self.workflow.inference_option_blockers())

        config = self._effective_config
        mock_error = (
            mock_inference_config_error(config)
            if isinstance(config, Mapping)
            else 'Select a config before enabling test inference'
        )
        output_root = getattr(self, '_rollout_output_root', None)
        if output_root is None and isinstance(self._effective_config, Mapping):
            output_root = configured_recording_root(
                self._effective_config,
                repo_root=self._recording_repo_root(),
            )
        path_error: Optional[str] = None
        canonical_output: Optional[Path] = None
        if output_root is not None:
            try:
                canonical_output = self._canonical_rollout_path(
                    str(output_root), require_writable=True
                )
            except RuntimeConfigError as exc:
                path_error = str(exc)
        effective_output = canonical_output
        rollout_enabled = bool(getattr(self, '_rollout_enabled', False))
        recording_state = (
            workflow_state.get('recording')
            if isinstance(workflow_state, Mapping)
            and isinstance(workflow_state.get('recording'), Mapping)
            else None
        )
        rollout_storage = getattr(self, 'rollout_storage', None)
        if rollout_storage is None:
            rollout_storage = RolloutStorageMonitor()
            self.rollout_storage = rollout_storage
        storage = rollout_storage.snapshot(
            output_root=canonical_output,
            enabled=rollout_enabled,
            config=(
                self._effective_config
                if isinstance(self._effective_config, Mapping)
                else None
            ),
            recording_state=recording_state,
        )
        return {
            'config_name': self.selected_config,
            'generation': int(getattr(self, '_inference_options_generation', 0)),
            'test_mode': bool(getattr(self, '_inference_test_mode', False)),
            'mock_supported': mock_error is None,
            'mock_error': mock_error,
            'rollout_enabled': rollout_enabled,
            'rollout_output_root': (
                str(output_root) if output_root is not None else None
            ),
            'rollout_effective_output_root': (
                str(effective_output) if effective_output is not None else None
            ),
            'rollout_path_valid': canonical_output is not None,
            'rollout_path_error': path_error,
            'rollout_storage': storage,
            'can_edit': not blockers,
            'editable': not blockers,
            'blockers': blockers,
        }

    def browse_rollout_paths(self, raw_path: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            config_name = self.selected_config
            generation = self._inference_options_generation
            revision = self.revision
            remembered = self._rollout_output_root
            if remembered is None and isinstance(self._effective_config, Mapping):
                remembered = configured_recording_root(
                    self._effective_config,
                    repo_root=self._recording_repo_root(),
                )
            roots = list(self._checkpoint_browse_roots)
        if not roots:
            raise RuntimeConfigError('no rollout browse roots are available')
        if raw_path is None or not str(raw_path).strip():
            try:
                selected = self._canonical_rollout_path(
                    str(remembered), require_writable=False
                )
                selected_root = self._checkpoint_root_for(selected)
                if selected_root is None:
                    raise OSError
                target = (
                    selected.parent
                    if selected != selected_root
                    and self._checkpoint_root_for(selected.parent) is not None
                    else selected
                )
            except (OSError, RuntimeError, RuntimeConfigError, TypeError):
                target = roots[0]
        else:
            target = self._canonical_rollout_path(
                raw_path, require_writable=False
            )
        target = self._canonical_rollout_path(
            str(target), require_writable=False
        )
        root = self._checkpoint_root_for(target)
        if root is None:
            raise RuntimeConfigError(
                'rollout browse path is outside the allowed roots'
            )
        entries: list[Dict[str, Any]] = []
        try:
            with os.scandir(target) as iterator:
                for entry in iterator:
                    try:
                        canonical = Path(entry.path).resolve(strict=True)
                        if (
                            self._checkpoint_root_for(canonical) is None
                            or not canonical.is_dir()
                            or not os.access(canonical, os.R_OK | os.X_OK)
                        ):
                            continue
                    except (OSError, RuntimeError):
                        continue
                    entries.append(
                        {
                            'name': entry.name,
                            'path': str(canonical),
                            'kind': 'directory',
                            'is_symlink': entry.is_symlink(),
                            'navigable': True,
                            'selectable': os.access(
                                canonical, os.R_OK | os.W_OK | os.X_OK
                            ),
                        }
                    )
        except OSError as exc:
            raise RuntimeConfigError(
                f'unable to browse rollout directory: {target}'
            ) from exc
        entries.sort(
            key=lambda item: (
                self._natural_path_key(str(item['name'])),
                str(item['name']),
            )
        )
        truncated = len(entries) > MAX_CHECKPOINT_BROWSE_ENTRIES
        entries = entries[:MAX_CHECKPOINT_BROWSE_ENTRIES]
        parent = target.parent
        return {
            'config_name': config_name,
            'revision': revision,
            'generation': generation,
            'path': str(target),
            'root': str(root),
            'roots': [str(item) for item in roots],
            'parent': (
                str(parent)
                if target != root and self._checkpoint_root_for(parent) is not None
                else None
            ),
            'current_selectable': os.access(
                target, os.R_OK | os.W_OK | os.X_OK
            ),
            'entries': entries,
            'truncated': truncated,
        }

    def set_test_inference_mode(
        self,
        enabled: Any,
        *,
        expected_selected_config: str,
        expected_revision: int,
        expected_generation: int,
    ) -> Dict[str, Any]:
        if not isinstance(enabled, bool):
            raise RuntimeConfigError('enabled must be a boolean')
        with self._lock:
            self._validate_inference_options_context_locked(
                expected_selected_config=expected_selected_config,
                expected_revision=expected_revision,
                expected_generation=expected_generation,
                require_editable=True,
            )
            if enabled and isinstance(self._effective_config, Mapping):
                error = mock_inference_config_error(self._effective_config)
                if error:
                    raise RuntimeConfigError(error)
            if self._inference_test_mode == enabled:
                return self.state()
            workflow = self.workflow
            assert workflow is not None
            workflow.set_inference_options(
                test_mode=enabled,
                recording_enabled=self._rollout_enabled,
                recording_output=self._rollout_output_root,
            )
            self._inference_test_mode = enabled
            self._inference_options_generation += 1
            self._bump_revision()
            self.logs.append(
                'INFO',
                'Checkpoint-free test inference '
                + ('enabled' if enabled else 'disabled'),
            )
            return self.state()

    def set_rollout_settings(
        self,
        *,
        enabled: Any,
        raw_path: Any = None,
        expected_selected_config: str,
        expected_revision: int,
        expected_generation: int,
    ) -> Dict[str, Any]:
        if not isinstance(enabled, bool):
            raise RuntimeConfigError('enabled must be a boolean')
        with self._lock:
            self._validate_inference_options_context_locked(
                expected_selected_config=expected_selected_config,
                expected_revision=expected_revision,
                expected_generation=expected_generation,
                require_editable=True,
            )
            output_root = self._rollout_output_root
            if output_root is None and enabled and isinstance(
                self._effective_config, Mapping
            ):
                output_root = configured_recording_root(
                    self._effective_config,
                    repo_root=self._recording_repo_root(),
                )
            if raw_path is not None:
                output_root = self._canonical_rollout_path(
                    raw_path, require_writable=True
                )
            elif enabled:
                if output_root is None:
                    raise RuntimeConfigError(
                        'Select a rollout save directory before enabling recording'
                    )
                output_root = self._canonical_rollout_path(
                    str(output_root), require_writable=True
                )
            changed = (
                enabled != self._rollout_enabled
                or output_root != self._rollout_output_root
            )
            if not changed:
                return self.state()
            workflow = self.workflow
            assert workflow is not None
            workflow.set_inference_options(
                test_mode=self._inference_test_mode,
                recording_enabled=enabled,
                recording_output=output_root,
            )
            if output_root is not None and output_root != self._rollout_output_root:
                try:
                    save_rollout_output_root(
                        self._inference_preferences_path, output_root
                    )
                except OSError as exc:
                    workflow.set_inference_options(
                        test_mode=self._inference_test_mode,
                        recording_enabled=self._rollout_enabled,
                        recording_output=self._rollout_output_root,
                    )
                    raise RuntimeConfigError(
                        f'Unable to remember rollout save directory: {exc}'
                    ) from exc
            self._rollout_output_root = output_root
            self._rollout_enabled = enabled
            self._rollout_settings_dirty = True
            if raw_path is not None:
                self._rollout_output_root_user_set = True
            self._inference_options_generation += 1
            self._bump_revision()
            self.logs.append(
                'INFO',
                'Inference rollout recording '
                + ('enabled' if enabled else 'disabled'),
            )
            return self.state()

    def _recording_config_locked(self) -> Optional[Mapping[str, Any]]:
        rollout_output_root = getattr(self, '_rollout_output_root', None)
        rollout_enabled = bool(getattr(self, '_rollout_enabled', False))
        if isinstance(self._effective_config, Mapping):
            configured_root = configured_recording_root(
                self._effective_config,
                repo_root=self._recording_repo_root(),
            )
            output_root = configured_root
            if rollout_output_root is not None and (
                configured_root is None
                or bool(getattr(self, '_rollout_output_root_user_set', False))
            ):
                output_root = rollout_output_root
            return build_control_panel_recording_config(
                self._effective_config,
                enabled=rollout_enabled,
                output_root=output_root,
            )
        if not getattr(self, '_fallback_recording_config_loaded', False):
            fallback: Optional[Dict[str, Any]] = None
            try:
                name = recommended_runtime_config(list_runtime_configs(self.config_dir))
                if name is not None:
                    fallback = compose_runtime_config(name, self.config_dir)
            except RuntimeConfigError:
                fallback = None
            self._fallback_recording_config = fallback
            self._fallback_recording_config_loaded = True
        fallback = self._fallback_recording_config
        if isinstance(fallback, Mapping) and rollout_output_root is not None:
            return build_control_panel_recording_config(
                fallback,
                enabled=False,
                output_root=rollout_output_root,
            )
        return fallback

    def recording_source_root(self) -> Optional[Path]:
        """Return a replay root without mutating the selected lifecycle."""
        with self._lock:
            config = self._recording_config_locked()
            if not isinstance(config, Mapping):
                return None
            source_root = configured_recording_root(
                config,
                repo_root=self._recording_repo_root(),
            )
            if source_root is None:
                return None
            configured_catalogs: list[Path] = []
            raw_catalogs = os.environ.get(RECORDING_REPLAY_ROOTS_ENV, '')
            for raw_catalog in raw_catalogs.split(os.pathsep):
                if raw_catalog.strip():
                    configured_catalogs.append(Path(raw_catalog.strip()))
            return latest_sibling_recording_root(
                source_root,
                sibling_parents=configured_catalogs,
            )

    def recording_camera_roles_config(self) -> Dict[str, Optional[str]]:
        """Return the selected config's normalized semantic camera roles."""
        with self._lock:
            config = self._recording_config_locked()
            if not isinstance(config, Mapping):
                return {"head": None, "left": None, "right": None}
            mapping = normalize_recording_camera_mapping(config)
            return {
                role: mapping[f"{role}_camera"]
                for role in ("head", "left", "right")
            }

    def recording_runtime_python(self) -> Optional[Path]:
        """Return the recorder interpreter resolved for the selected config."""
        with self._lock:
            config = copy.deepcopy(self._effective_config or {})
            explicit = self.recording_python
            config_name = self.selected_config
        return resolve_recording_python(
            config,
            explicit=explicit,
            config_name=config_name,
            repo_root=self._recording_repo_root(),
        )

    def _recording_camera_roles_locked(
        self,
        devices: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        config = self._effective_config
        expected = self.recording_camera_roles_config()
        configured_keys: set[str] = set()
        if isinstance(config, Mapping) and isinstance(config.get("devices"), list):
            for item in config["devices"]:
                if not isinstance(item, Mapping):
                    continue
                device_class = item.get("class")
                if isinstance(device_class, str) and "device_class_id" in item:
                    configured_keys.add(f"{device_class}_{item['device_class_id']}")

        live: Dict[str, Mapping[str, Any]] = {}
        for item in devices:
            device_class = item.get("class", item.get("device_class"))
            device_class_id = item.get("id", item.get("device_class_id"))
            if isinstance(device_class, str) and device_class_id is not None:
                live[f"{device_class}_{device_class_id}"] = item

        roles: Dict[str, Dict[str, Any]] = {}
        for role, device_key in expected.items():
            device = live.get(device_key) if device_key is not None else None
            if device_key is None:
                status, reason = "unknown", "camera mapping missing; LeRobot conversion unavailable"
            elif device_key not in configured_keys:
                status, reason = "missing", "camera device is not configured"
            elif device is not None and device.get("ready") is True:
                status, reason = "healthy", None
            else:
                current = str(device.get("status") if device is not None else "stopped")
                status, reason = "configured", f"camera device status: {current}"
            roles[role] = {
                "role": role,
                "camera_id": LOGICAL_CAMERA_IDS[role],
                "device_key": device_key,
                "topic": (
                    "/camera/color/compressed/jpg/"
                    f"{LOGICAL_CAMERA_IDS[role]}"
                ),
                "shm_name": f"{device_key}_data" if device_key else None,
                "status": status,
                "reason": reason,
            }
        return roles

    def _has_active_devices_locked(self) -> bool:
        return bool(self.starter and self.starter.has_active_devices())

    def _has_active_workflow_locked(self) -> bool:
        return bool(self.workflow and self.workflow.has_active_processes())

    def _ensure_mutable_locked(self) -> None:
        if self._shutdown:
            raise DeviceConflictError("Device control server is shutting down")
        if self._force_stop_in_progress:
            raise DeviceConflictError('Force stop is in progress')
        if (
            self._has_active_devices_locked()
            or self._has_active_workflow_locked()
            or self.grippers.active()
        ):
            raise DeviceConflictError(
                "Stop inference, runtime, and all running/external devices before "
                "changing config bindings"
            )

    def _ensure_source_config_saveable_locked(self) -> None:
        """Allow persistence beside running devices without changing live state."""
        if self._shutdown:
            raise DeviceConflictError("Device control server is shutting down")
        if self._force_stop_in_progress:
            raise DeviceConflictError('Force stop is in progress')
        self._ensure_device_operation_state_locked()
        if self._device_operations:
            raise DeviceConflictError(
                'Wait for the current device or workflow action before saving config'
            )
        if self._has_active_workflow_locked() or self.grippers.active():
            raise DeviceConflictError(
                'Stop inference, runtime, and standalone gripper control before '
                'saving config'
            )

    @staticmethod
    def _opaque_binding_overrides(
        bindings: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Recover provider IDs from the controller's sanitized binding state."""
        opaque: Dict[str, Dict[str, Any]] = {}
        for raw_key, raw_fields in bindings.items():
            if not isinstance(raw_fields, Mapping):
                continue
            fields: Dict[str, Any] = {}
            for raw_field, raw_value in raw_fields.items():
                if not isinstance(raw_value, Mapping):
                    continue
                provider_id = raw_value.get('id')
                if isinstance(provider_id, str) and provider_id:
                    fields[str(raw_field)] = provider_id
                elif (
                    str(raw_field) == 'capture_device'
                    and provider_id is None
                    and raw_value.get('status') == 'missing'
                ):
                    fields[str(raw_field)] = None
            if fields:
                opaque[str(raw_key)] = fields
        return opaque

    @staticmethod
    def _complete_camera_binding_swaps(
        current: Mapping[str, Mapping[str, str]],
        requested: Mapping[str, Any],
        camera_keys: set[str],
    ) -> Dict[str, Any]:
        """Turn a sparse camera selection into one atomic physical swap.

        Older tabs can submit only the clicked slot after already rendering the
        displaced slot with its new value.  If the selected physical camera is
        still owned by another logical slot, exchange that slot with the
        clicked slot's previous camera before validating the composed config.
        """

        completed = {
            str(key): copy.deepcopy(value)
            for key, value in requested.items()
        }
        requested_camera_keys = [
            str(key) for key in requested if str(key) in camera_keys
        ]
        assignments = {
            str(key): str(fields.get('capture_device'))
            for key, fields in current.items()
            if str(key) in camera_keys
            and isinstance(fields, Mapping)
            and isinstance(fields.get('capture_device'), str)
            and fields.get('capture_device')
        }

        for key in requested_camera_keys:
            raw_binding = completed.get(key)
            if not isinstance(raw_binding, Mapping):
                continue
            selected = raw_binding.get('capture_device')
            if not isinstance(selected, str) or not selected:
                assignments.pop(key, None)
                continue

            previous = assignments.get(key)
            owner = next(
                (
                    other_key
                    for other_key, camera_id in assignments.items()
                    if other_key != key and camera_id == selected
                ),
                None,
            )
            if owner is not None:
                if previous and previous != selected:
                    owner_binding = copy.deepcopy(dict(current.get(owner, {})))
                    owner_binding['capture_device'] = previous
                    completed[owner] = owner_binding
                    assignments[owner] = previous
                else:
                    completed[owner] = {}
                    assignments.pop(owner, None)
            assignments[key] = selected

        return completed

    @classmethod
    def _camera_bindings_from_config(
        cls,
        config: Mapping[str, Any],
        snapshot: HardwareSnapshot,
    ) -> Dict[str, Dict[str, str]]:
        """Resolve config-default capture paths into opaque discovery IDs."""

        cameras = [
            camera
            for camera in snapshot.cameras
            if isinstance(camera, Mapping)
            and isinstance(camera.get('id'), str)
            and camera.get('id')
        ]
        resolved: Dict[str, Dict[str, str]] = {}
        for key, device in cls._device_configs_by_key(config).items():
            if str(device.get('class')) not in {
                'WebCameraDevice',
                'WebCameraOpenPi',
            }:
                continue
            if capture_binding_is_missing(device):
                continue
            capture = device.get('capture_id')
            capture_text = str(capture)
            try:
                capture_number: Optional[int] = int(capture)
            except (TypeError, ValueError):
                match = re.search(r'/dev/video(\d+)$', capture_text)
                capture_number = int(match.group(1)) if match else None

            camera = next(
                (
                    item
                    for item in cameras
                    if capture_text
                    in {
                        str(item.get('id') or ''),
                        str(item.get('stable_path') or ''),
                        str(item.get('devnode') or ''),
                    }
                    or (
                        capture_number is not None
                        and str(item.get('devnode') or '').endswith(
                            f'/video{capture_number}'
                        )
                    )
                ),
                None,
            )
            if camera is not None:
                resolved[key] = {'capture_device': str(camera['id'])}
        return resolved

    @staticmethod
    def _video_device_number(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if normalized.isdecimal():
            return int(normalized)
        match = re.fullmatch(r'/dev/video(\d+)', normalized)
        return int(match.group(1)) if match else None

    @classmethod
    def _camera_bindings_as_capture_ids(
        cls,
        config: Mapping[str, Any],
        bindings: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Persist physical capture bindings without changing logical identity."""
        from utils.device_identity import validate_device_class_ids

        result = copy.deepcopy(dict(config))
        raw_devices = result.get("devices")
        devices = raw_devices.get("devices") if isinstance(raw_devices, Mapping) else raw_devices
        if not isinstance(devices, list):
            return result, copy.deepcopy(dict(bindings))
        validate_device_class_ids(devices)
        by_key = {
            f"{device['class']}:{device['device_class_id']}": device
            for device in devices
        }
        slots = extract_camera_slots(result)
        camera = result.setdefault("camera", {})
        for role in ("head", "left", "right"):
            slot = slots.get(role)
            if not isinstance(slot, Mapping):
                continue
            key = slot.get("device_key")
            device = by_key.get(key)
            if device is None:
                continue
            capture = device.get("capture_id")
            if capture is not None:
                camera[f"{role}_capture_id"] = capture
            device.pop(CAPTURE_BINDING_STATE_KEY, None)
        return result, copy.deepcopy(dict(bindings))

    @staticmethod
    def _device_configs_by_key(
        config: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        devices = config.get('devices', [])
        if not isinstance(devices, list):
            return {}
        return {
            f"{device.get('class')}:{device.get('device_class_id')}": device
            for device in devices
            if isinstance(device, dict)
            and device.get('enabled', True) is not False
        }

    def _yam_can_interfaces_for_device_locked(
        self, device_class: str, device_class_id: Any
    ) -> list[str]:
        if device_class != 'YamHeadRobot':
            return []
        config = self._effective_config
        if not isinstance(config, Mapping):
            return []
        key = f'{device_class}:{device_class_id}'
        device = self._device_configs_by_key(config).get(key)
        if not isinstance(device, Mapping):
            return []
        if device.get('yam_sim') is True:
            return []
        return [str(device.get('yam_channel') or 'can0')]

    def _ensure_can_ready_for_device_locked(
        self, device_class: str, device_class_id: Any
    ) -> None:
        interfaces = self._yam_can_interfaces_for_device_locked(
            device_class, device_class_id
        )
        if interfaces:
            self.can_permissions.ensure_ready(interfaces)

    @staticmethod
    def _binding_status_has_owned_worker(device: Mapping[str, Any]) -> bool:
        status = str(device.get('status') or '')
        # External/orphaned SHM prevents this controller from starting the
        # logical device, but it does not own a worker whose live config could
        # be mutated.  Allow the UI to stage a new binding for the next start;
        # lifecycle start guards continue to block until the external owner or
        # orphaned SHM is gone.
        if status in {'external', 'orphaned'}:
            return False
        return bool(
            device.get('running') is True
            or device.get('desired_state') == 'running'
            or status
            in {
                'starting',
                'running',
                'degraded',
                'stopping',
                'stop_timeout',
            }
        )

    def _apply_binding_config_locked(
        self,
        starter: DeviceLifecycleManager,
        effective_config: Dict[str, Any],
        *,
        physical_noop_keys: Optional[set[str]] = None,
    ) -> set[str]:
        """Update device configs in place without replacing active workers."""

        def apply() -> set[str]:
            noop_keys = physical_noop_keys or set()
            statuses = starter.get_status().get('devices', [])
            status_by_key = {
                str(device.get('key')): device
                for device in statuses
                if isinstance(device, Mapping) and device.get('key')
            }
            current_by_key = self._device_configs_by_key(starter.config)
            candidate_by_key = self._device_configs_by_key(effective_config)
            if set(current_by_key) != set(candidate_by_key):
                raise DeviceConflictError(
                    "Runtime binding update cannot change configured device keys"
                )

            changed_config_keys = {
                key
                for key in current_by_key
                if current_by_key.get(key) != candidate_by_key.get(key)
            }
            changed_binding_keys = changed_config_keys - noop_keys
            changed_active = sorted(
                key
                for key in changed_binding_keys
                if self._binding_status_has_owned_worker(
                    status_by_key.get(key, {})
                )
            )
            if changed_active:
                raise DeviceConflictError(
                    "Runtime binding update would change an active device: "
                    + ", ".join(changed_active)
                )

            device_records = getattr(starter, 'devices', None)
            if isinstance(device_records, list):
                for record in device_records:
                    if not isinstance(record, dict):
                        continue
                    key = f"{record.get('device_class')}:{record.get('device_class_id')}"
                    candidate = candidate_by_key.get(key)
                    if key in changed_config_keys and candidate is not None:
                        record['config'] = copy.deepcopy(candidate)
            starter.config = copy.deepcopy(effective_config)
            return changed_binding_keys

        starter_lock = getattr(starter, '_lock', None)
        if starter_lock is None:
            return apply()
        with starter_lock:
            return apply()

    @staticmethod
    def _direct_gripper_config(config: Mapping[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(dict(config))
        devices = result.get('devices', [])
        if isinstance(devices, Mapping):
            devices = devices.get('devices', [])
        if not isinstance(devices, list):
            return result
        for device in devices:
            if not isinstance(device, dict):
                continue
            if str(device.get('class', '')).casefold() != 'rizonrobot':
                continue
            gripper = device.get('gripper')
            if not isinstance(gripper, dict):
                flat_backend = str(device.get('gripper_backend') or '').strip().casefold()
                if flat_backend == 'lktech_websocket':
                    device['gripper_backend'] = 'lktech'
                    device.pop('gripper_server', None)
                elif flat_backend in {'', 'lktech'} and 'gripper_server' in device:
                    # ``gripper_server`` belonged to the removed daemon. Keep
                    # this migration private to the runtime copy and make the
                    # historical empty-backend default explicit.
                    if not flat_backend:
                        device['gripper_backend'] = 'lktech'
                    device.pop('gripper_server', None)
                continue
            # Migrate only the removed service-era transport in memory. Keep
            # the legitimate Flexiv gripper backend untouched; the source YAML
            # is never rewritten as part of selection.
            backend = str(gripper.get('backend', '')).strip().casefold()
            if backend == 'lktech_websocket':
                gripper['backend'] = 'lktech'
                gripper.pop('server', None)
            elif backend in {'', 'lktech'}:
                if not backend:
                    # Historical configs used the direct LKTech default but
                    # retained an obsolete server block.
                    gripper['backend'] = 'lktech'
                gripper.pop('server', None)
        return result

    @staticmethod
    def _apply_gripper_port_overrides(
        config: Mapping[str, Any],
        overrides: Mapping[str, str],
    ) -> Dict[str, Any]:
        result = copy.deepcopy(dict(config))
        if not overrides:
            return result
        devices = result.get('devices', [])
        if isinstance(devices, Mapping):
            devices = devices.get('devices', [])
        if not isinstance(devices, list):
            return result
        for device in devices:
            if not isinstance(device, dict):
                continue
            if str(device.get('class', '')).casefold() != 'rizonrobot':
                continue
            gripper = device.get('gripper')
            if not isinstance(gripper, dict):
                continue
            raw_ports = gripper.get('port', ['', ''])
            ports = list(raw_ports) if isinstance(raw_ports, (list, tuple)) else [raw_ports, raw_ports]
            if len(ports) != 2:
                continue
            for index, side in enumerate(('left', 'right')):
                if side in overrides:
                    ports[index] = str(overrides[side])
            gripper['port'] = ports
        return result

    @staticmethod
    def _normalize_gripper_port_overrides(overrides: Any) -> Dict[str, str]:
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise RuntimeConfigError('gripper_overrides must be an object')
        unknown = sorted(set(map(str, overrides)) - {'left', 'right'})
        if unknown:
            raise RuntimeConfigError(
                'Unknown gripper override side(s): ' + ', '.join(unknown)
            )
        normalized: Dict[str, str] = {}
        for side, raw_port in overrides.items():
            port = str(raw_port).strip()
            if not re.fullmatch(r'/dev/ttyUSB[0-9]+', port):
                raise RuntimeConfigError(
                    f'Invalid {side} gripper serial port: {port or "empty"}'
                )
            normalized[str(side)] = port
        if len(set(normalized.values())) != len(normalized):
            raise RuntimeConfigError(
                'left and right grippers cannot use the same serial port'
            )
        return normalized

    @staticmethod
    def _apply_control_panel_safety_overrides(
        config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Adjust panel-owned runtime safety thresholds without editing YAML."""
        result = copy.deepcopy(dict(config))
        devices = result.get('devices', [])
        if isinstance(devices, Mapping):
            devices = devices.get('devices', [])
        if not isinstance(devices, list):
            return result
        for device in devices:
            if not isinstance(device, dict):
                continue
            if str(device.get('class', '')).casefold() != 'yamheadrobot':
                continue
            device['disconnect_zero_joint_count'] = (
                CONTROL_PANEL_YAM_DISCONNECT_ZERO_JOINT_COUNT
            )
            device['disconnect_zero_tolerance_rad'] = (
                CONTROL_PANEL_YAM_DISCONNECT_ZERO_TOLERANCE_RAD
            )
        return result

    def _runtime_config_for_gripper(
        self,
        config: Mapping[str, Any],
        *,
        port_overrides: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        return self._apply_gripper_port_overrides(
            self._direct_gripper_config(config),
            (
                getattr(self, '_gripper_port_overrides', {})
                if port_overrides is None
                else port_overrides
            ),
        )

    def _apply_gripper_transport_config_locked(
        self,
        runtime_config: Dict[str, Any],
    ) -> None:
        starter = self._require_starter_locked()
        self._apply_binding_config_locked(
            starter,
            runtime_config,
        )
        workflow = self.workflow
        if workflow is not None:
            update_config = getattr(workflow, 'update_runtime_config', None)
            if callable(update_config):
                update_config(runtime_config)
            else:
                workflow.config = copy.deepcopy(runtime_config)
        self.grippers.configure(self.selected_config or '', runtime_config)

    def _make_workflow(
        self,
        name: str,
        effective_config: Dict[str, Any],
        *,
        checkpoint_override: Optional[Path] = None,
        preserve_inference_options: bool = True,
    ) -> WorkflowLifecycle:
        factory = self._workflow_factory
        if factory is None:
            metadata = runtime_config_metadata(name, effective_config)
            policy = effective_config.get('policy')
            policy_class = (
                policy.get('class') if isinstance(policy, Mapping) else None
            )
            factory = (
                FastWAMWorkflowLifecycle
                if (
                    metadata.get('workflow_kind') == 'fastwam'
                    and policy_class == 'FastWAMPolicy'
                )
                else WorkflowLifecycle
            )
        return factory(
            name,
            effective_config,
            checkpoint_override=checkpoint_override,
            test_inference_mode=(
                self._inference_test_mode if preserve_inference_options else False
            ),
            recording_enabled_override=(
                self._rollout_enabled if preserve_inference_options else False
            ),
            recording_output_override=self._rollout_output_root,
            recording_python=self.recording_python,
            conversion_enqueue=lambda receipt, source_root, run_config: self.conversion_jobs.enqueue(
                receipt, source_root=source_root, run_config=run_config
            ),
            conversion_cancel=self.conversion_jobs.cancel_episode,
            recording_finalized_enqueue=self._recording_finalized_enqueue,
            log_callback=lambda level, message: self.logs.append(level, message),
        )

    def set_recording_finalized_enqueue(
        self,
        callback: Optional[Callable[[Mapping[str, Any]], Any]],
    ) -> None:
        """Bind advisory replay prefetch without changing recording authority."""
        with self._lock:
            self._recording_finalized_enqueue = callback
            workflow = self.workflow
            if workflow is None:
                return
            setattr(workflow, '_recording_finalized_enqueue', callback)
            coordinator = getattr(workflow, 'recording_coordinator', None)
            if coordinator is not None:
                setattr(coordinator, '_finalized_enqueue', callback)

    def _reconcile_conversion_sources(self, config: Mapping[str, Any]) -> int:
        source_root = getattr(self, '_rollout_output_root', None) or configured_recording_root(
            config, repo_root=self._recording_repo_root()
        )
        if source_root is None:
            return 0
        run_config = {
            'config_name': self.selected_config,
            **normalize_recording_camera_mapping(config),
        }
        if not run_config['left_camera'] or not run_config['right_camera']:
            self.logs.append(
                'WARNING',
                'LeRobot conversion reconciliation is using the finalized data contract because the current config has no wrist camera mapping',
            )
        recovered = 0
        for episode_root in iter_recording_episode_roots(source_root):
            recovered += self.conversion_jobs.reconcile_source_root(
                episode_root,
                run_config=run_config,
            )
        return recovered

    @staticmethod
    def _validate_discovery_revision(
        snapshot: HardwareSnapshot,
        discovery_revision: Optional[str],
        *,
        required: bool,
    ) -> None:
        if required and not discovery_revision:
            raise RuntimeConfigError(
                "discovery_revision is required when applying overrides"
            )
        if (
            required
            and discovery_revision
            and discovery_revision != snapshot.revision
        ):
            raise DeviceConflictError(
                "Hardware discovery changed; refresh discovery before applying bindings"
            )

    def _remember_discovery_snapshot(self, snapshot: HardwareSnapshot) -> None:
        """Retain the exact short-lived inventory returned to a binding editor."""

        now = time.monotonic()
        with self._lock:
            self._discovery_snapshot_cache = {
                revision: entry
                for revision, entry in self._discovery_snapshot_cache.items()
                if now - entry[0] <= 30.0
            }
            self._discovery_snapshot_cache[snapshot.revision] = (now, snapshot)
            if len(self._discovery_snapshot_cache) > 8:
                oldest = min(
                    self._discovery_snapshot_cache,
                    key=lambda revision: self._discovery_snapshot_cache[revision][0],
                )
                self._discovery_snapshot_cache.pop(oldest, None)

    def _cached_discovery_snapshot(
        self, discovery_revision: Optional[str]
    ) -> Optional[HardwareSnapshot]:
        if not discovery_revision:
            return None
        entry = self._discovery_snapshot_cache.get(discovery_revision)
        if entry is None:
            return None
        if time.monotonic() - entry[0] > 30.0:
            self._discovery_snapshot_cache.pop(discovery_revision, None)
            return None
        return entry[1]

    @staticmethod
    def _preserve_unchanged_camera_bindings(
        snapshot: HardwareSnapshot,
        current: Mapping[str, Mapping[str, str]],
        requested_keys: set[str],
    ) -> HardwareSnapshot:
        """Keep a running peer camera valid when its frame probe is busy."""

        preserved_ids = {
            str(fields.get('capture_device'))
            for key, fields in current.items()
            if key not in requested_keys
            and isinstance(fields, Mapping)
            and isinstance(fields.get('capture_device'), str)
            and fields.get('capture_device')
        }
        if not preserved_ids:
            return snapshot
        cameras = []
        changed = False
        for camera in snapshot.cameras:
            item = copy.deepcopy(dict(camera))
            if (
                str(item.get('id')) in preserved_ids
                and item.get('capture_capable') is False
            ):
                item['capture_capable'] = True
                item['binding_preserved'] = True
                changed = True
            cameras.append(item)
        return replace(snapshot, cameras=cameras) if changed else snapshot

    @staticmethod
    def _overrides_require_discovery_snapshot(overrides: Any) -> bool:
        """Return whether an override contains a hardware provider identifier."""

        if not isinstance(overrides, Mapping):
            return bool(overrides)
        for raw_binding in overrides.values():
            if not isinstance(raw_binding, Mapping):
                return True
            if any(value is not None for value in raw_binding.values()):
                return True
        return False

    @classmethod
    def _complete_config_camera_bindings(
        cls,
        base_config: Mapping[str, Any],
        requested: Any,
        snapshot: HardwareSnapshot,
    ) -> Any:
        """Make config selection a complete, conflict-free camera assignment."""

        if requested is not None and not isinstance(requested, Mapping):
            return requested
        completed: Dict[str, Any] = {
            str(key): copy.deepcopy(value)
            for key, value in (requested or {}).items()
        }
        camera_keys = [
            key
            for key, device in cls._device_configs_by_key(base_config).items()
            if str(device.get('class')) in {
                'WebCameraDevice',
                'WebCameraOpenPi',
            }
        ]
        config_defaults = cls._camera_bindings_from_config(base_config, snapshot)
        explicit_keys: set[str] = set()
        used_camera_ids: Dict[str, str] = {}

        # Explicit selections win over config defaults. Duplicate explicit
        # selections are deterministic: the first logical config slot keeps
        # the camera and the later slot is parked.
        for key in camera_keys:
            raw_binding = completed.get(key)
            if (
                not isinstance(raw_binding, Mapping)
                or 'capture_device' not in raw_binding
            ):
                continue
            explicit_keys.add(key)
            selected = raw_binding.get('capture_device')
            if selected is None:
                continue
            if not isinstance(selected, str):
                continue
            match = snapshot.lookup(selected)
            if match is None or match[0] != 'camera':
                continue
            camera_id = str(match[1].get('id'))
            owner = used_camera_ids.get(camera_id)
            if owner is not None and owner != key:
                parked = copy.deepcopy(dict(raw_binding))
                parked['capture_device'] = None
                completed[key] = parked
                continue
            used_camera_ids[camera_id] = key

        # Every remaining camera gets either its matching config device or an
        # explicit parking slot. This prevents a sparse request from silently
        # reactivating a conflicting /dev/videoN default.
        for key in camera_keys:
            if key in explicit_keys:
                continue
            selected = config_defaults.get(key, {}).get('capture_device')
            if not isinstance(selected, str) or not selected:
                selected = None
            elif selected in used_camera_ids:
                selected = None
            else:
                used_camera_ids[selected] = key
            completed[key] = {'capture_device': selected}
        return completed

    @staticmethod
    def _effective_from_snapshot(
        base_config: Dict[str, Any],
        overrides: Any,
        snapshot: HardwareSnapshot,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        try:
            return apply_discovery_overrides(base_config, overrides, snapshot)
        except ValueError as exc:
            raise RuntimeConfigError(str(exc)) from exc

    @staticmethod
    def _active_keys(starter: DeviceLifecycleManager) -> list[str]:
        try:
            devices = starter.get_status().get('devices', [])
        except (AttributeError, DeviceControlError, RuntimeError, ValueError):
            return []
        return sorted(
            {
                str(device.get('key'))
                for device in devices
                if device.get('key')
                and (
                    device.get('running') is True
                    or device.get('desired_state') == 'running'
                    or device.get('status')
                    in {'external', 'orphaned', 'stop_timeout'}
                )
            }
        )

    @staticmethod
    def _normalize_stop_report(
        starter: DeviceLifecycleManager,
        raw_report: Any,
        active_before: list[str],
    ) -> Dict[str, Any]:
        report = raw_report if isinstance(raw_report, dict) else {}
        incomplete = {
            str(key) for key in report.get('incomplete_devices', []) if key
        }
        external = {
            str(key) for key in report.get('external_devices', []) if key
        }
        still_active = set(DeviceController._active_keys(starter))
        incomplete.update(still_active)
        stopped = {
            str(key) for key in report.get('stopped_devices', []) if key
        }
        # Lifecycle doubles and older managers may return no report.  Derive a
        # safe key-only result from the before/after status without assuming
        # that an attempted stop succeeded.
        stopped.update(set(active_before) - incomplete)
        return {
            'ok': bool(report.get('ok', not incomplete)) and not incomplete,
            'devices': report.get('devices', []),
            'stopped_devices': sorted(stopped),
            'incomplete_devices': sorted(incomplete),
            'external_devices': sorted(external),
        }

    def _replace_starter_locked(
        self,
        *,
        name: str,
        base_config: Dict[str, Any],
        effective_config: Dict[str, Any],
        bindings: Dict[str, Dict[str, Any]],
        discovery_revision: Optional[str],
        candidate: DeviceLifecycleManager,
        candidate_workflow: WorkflowLifecycle,
        stop_active: bool,
        gripper_port_overrides: Mapping[str, str],
    ) -> Dict[str, Any]:
        if self._shutdown:
            raise DeviceConflictError("Device control server is shutting down")

        previous_config = self.selected_config
        previous_starter = self.starter
        previous_workflow = self.workflow
        active_before: list[str] = []
        stop_report: Dict[str, Any] = {
            'ok': True,
            'devices': [],
            'stopped_devices': [],
            'incomplete_devices': [],
            'external_devices': [],
        }
        if previous_starter is not None or previous_workflow is not None:
            has_active = bool(
                (previous_starter and previous_starter.has_active_devices())
                or (
                    previous_workflow
                    and previous_workflow.has_active_processes()
                )
                or self.grippers.active()
            )
            if has_active and not stop_active:
                raise DeviceConflictError(
                    "Active inference, runtime, or devices must be stopped before "
                    "changing config; "
                    "retry with stop_active=true to stop owned devices"
                )
            # Teardown order is a safety contract: consumers first, then the
            # shared-memory producers they depend on.
            self.grippers.stop('config_change')
            if previous_workflow is not None:
                previous_workflow.stop_all()
            if previous_starter is not None:
                active_before = self._active_keys(previous_starter)
                raw_report = previous_starter.stop_all()
                stop_report = self._normalize_stop_report(
                    previous_starter, raw_report, active_before
                )
                if not stop_report['ok']:
                    incomplete = stop_report['incomplete_devices']
                    raise ConfigSwitchConflictError(
                        "Config switch aborted because previous device keys remain "
                        "active: " + ", ".join(incomplete),
                        selected_config=previous_config,
                        stopped_devices=stop_report['stopped_devices'],
                        incomplete_devices=incomplete,
                        external_devices=stop_report['external_devices'],
                    )

        # Readers are closed only after the old lifecycle is confirmed stopped;
        # until this point the selected config and its inspection state remain
        # unchanged even if cleanup failed.
        self._close_preview_readers_locked()
        self._close_robot_state_readers_locked()
        candidate.start_monitor()
        self.starter = candidate
        self.workflow = candidate_workflow
        self.selected_config = name
        self._base_config = copy.deepcopy(base_config)
        self._effective_config = copy.deepcopy(effective_config)
        self._effective_bindings = copy.deepcopy(bindings)
        self._gripper_port_overrides = dict(gripper_port_overrides)
        if configured_recording_root(
            effective_config,
            repo_root=self._recording_repo_root(),
        ) is not None:
            # A YAML-owned path is the default for this selection. Do not carry
            # a browser-selected path from another config across the switch.
            self._rollout_output_root = None
            self._rollout_output_root_user_set = False
        self._checkpoint_override = None
        self._checkpoint_generation += 1
        self._inference_test_mode = False
        self._rollout_enabled = False
        self._rollout_settings_dirty = False
        self._inference_options_generation += 1
        self._binding_discovery_revision = (
            discovery_revision if bindings else None
        )
        self.grippers.configure(
            name,
            getattr(candidate, 'config', effective_config),
        )
        self._bump_revision()
        return {
            'previous_config': previous_config,
            'stopped_previous': bool(stop_report['stopped_devices']),
            'stopped_devices': stop_report['stopped_devices'],
        }

    def select_config(
        self,
        config_name: str,
        *,
        overrides: Any = None,
        discovery_revision: Optional[str] = None,
        stop_active: bool = False,
        auto_apply: bool = False,
        expected_revision: Optional[int] = None,
        expected_selected_config: Optional[str] = None,
        gripper_overrides: Any = None,
    ) -> Dict[str, Any]:
        name = normalize_config_name(config_name)
        base_config = compose_runtime_config(name, config_dir=self.config_dir)
        errors = validate_runtime_config(base_config)
        if errors:
            raise RuntimeConfigError('; '.join(errors))
        metadata = runtime_config_metadata(name, base_config, errors=errors)
        if not metadata['control_panel_supported']:
            raise RuntimeConfigError(
                f"config '{name}' is not supported by the control panel: "
                f"{metadata['support_reason']}"
            )

        requires_discovery_snapshot = self._overrides_require_discovery_snapshot(
            overrides
        )
        binding_revision: Optional[str] = None
        try:
            snapshot = (
                self._cached_discovery_snapshot(discovery_revision)
                if requires_discovery_snapshot
                else None
            )
            if snapshot is None:
                snapshot = self.hardware_discovery.discover()
        except Exception as exc:
            if requires_discovery_snapshot:
                raise RuntimeConfigError(
                    f'Hardware discovery is unavailable for explicit overrides: {exc}'
                ) from exc
            # Discovery is advisory for a binding-free config switch. Keep the
            # config selectable and park its camera slots until a later refresh
            # succeeds, at which point the operator can choose real devices.
            snapshot = HardwareSnapshot(
                revision='',
                generated_at='',
                interfaces=[],
                cameras=[],
                robots=[],
                providers=[],
                warnings=[str(exc)],
            )
        selection_overrides = self._complete_config_camera_bindings(
            base_config,
            overrides,
            snapshot,
        )
        has_overrides = bool(selection_overrides)
        if has_overrides:
            self._validate_discovery_revision(
                snapshot,
                discovery_revision,
                required=requires_discovery_snapshot,
            )
            effective_config, bindings = self._effective_from_snapshot(
                base_config, selection_overrides, snapshot
            )
            binding_revision = (
                snapshot.revision
                if snapshot.revision
                and self._overrides_require_discovery_snapshot(selection_overrides)
                else None
            )
        else:
            # The current inventory is diagnostic only when there is no
            # explicit override. Missing config hardware remains visible as an
            # unmatched selector instead of becoming a config-selection gate.
            effective_config, bindings = self._effective_from_snapshot(
                base_config, {}, snapshot
            )
        gripper_port_overrides = self._normalize_gripper_port_overrides(
            gripper_overrides
        )
        effective_config = self._apply_gripper_port_overrides(
            effective_config, gripper_port_overrides
        )
        effective_config = self._apply_control_panel_safety_overrides(
            effective_config
        )
        effective_errors = validate_runtime_config(effective_config)
        if effective_errors:
            raise RuntimeConfigError('; '.join(effective_errors))

        # Construction parses plain config only and starts no process.  Doing
        # it before the controller lock avoids leaving the old selection half
        # torn down if the candidate is invalid.
        runtime_config = self._runtime_config_for_gripper(
            effective_config,
            port_overrides={},
        )
        candidate = DeviceLifecycleManager(config=runtime_config)
        candidate_workflow = self._make_workflow(
            name,
            runtime_config,
            preserve_inference_options=False,
        )

        # Compose first, then take the lifecycle lock.  A slow Hydra compose
        # should not delay an unrelated status request.
        with (
            self._gripper_permission_mutation_guard(),
            self._lock,
            self._lifecycle_mutation_guard_locked(),
        ):
            if auto_apply:
                if expected_revision is None:
                    raise RuntimeConfigError(
                        'expected_revision is required for auto_apply'
                    )
                if (
                    self.revision != expected_revision
                    or self.selected_config != expected_selected_config
                    or self.selected_config is not None
                    or self.starter is not None
                    or self.workflow is not None
                    or self._has_active_devices_locked()
                    or self._has_active_workflow_locked()
                    or self.grippers.active()
                ):
                    raise BootstrapConfigConflictError(
                        'Automatic config selection was superseded by current state',
                        expected_revision=expected_revision,
                        current_revision=self.revision,
                        expected_selected_config=expected_selected_config,
                        current_selected_config=self.selected_config,
                    )
            switch_result = self._replace_starter_locked(
                name=name,
                base_config=base_config,
                effective_config=effective_config,
                bindings=bindings,
                discovery_revision=binding_revision,
                candidate=candidate,
                candidate_workflow=candidate_workflow,
                stop_active=stop_active,
                gripper_port_overrides=gripper_port_overrides,
            )
            recovered_jobs = self._reconcile_conversion_sources(effective_config)
            if recovered_jobs:
                self.logs.append(
                    'INFO',
                    f'Recovered {recovered_jobs} accepted episode conversion job(s)',
                )
            self.logs.append('INFO', f"Selected config {name}")
            result = self.state()
            result.update(switch_result)
            return result

    def update_bindings(
        self,
        *,
        overrides: Any,
        discovery_revision: Optional[str],
    ) -> Dict[str, Any]:
        reserved_keys: list[str] = []
        with self._lock:
            if (
                self.selected_config is None
                or self._base_config is None
                or self._effective_config is None
            ):
                raise DeviceConflictError(
                    "Select a config before applying runtime bindings"
                )
            starter = self._require_starter_locked()
            if self.workflow and self.workflow.has_active_runtime_processes():
                raise DeviceConflictError(
                    "Stop inference and runtime before changing device bindings"
                )
            if not isinstance(overrides, Mapping) or not overrides:
                raise RuntimeConfigError(
                    "overrides must contain at least one device binding"
                )
            requested_keys = {str(key) for key in overrides}
            base_devices = self._device_configs_by_key(self._base_config)
            camera_keys = {
                key
                for key, device in base_devices.items()
                if str(device.get('class')) in {
                    'WebCameraDevice',
                    'WebCameraOpenPi',
                }
            }
            # Reuse the exact active snapshot returned to the dropdown. A
            # second frame probe here can change reachability while handling
            # the same user action and manufacture a revision conflict.
            snapshot = self._cached_discovery_snapshot(discovery_revision)
            if snapshot is None:
                active_discover = getattr(
                    self.hardware_discovery, 'discover_with_camera_probe', None
                )
                snapshot = (
                    active_discover()
                    if requested_keys & camera_keys and callable(active_discover)
                    else self.hardware_discovery.discover()
                )
            self._validate_discovery_revision(
                snapshot,
                discovery_revision,
                required=True,
            )
            merged_overrides: Dict[str, Any] = self._opaque_binding_overrides(
                self._effective_bindings
            )
            current_camera_bindings = self._camera_bindings_from_config(
                self._effective_config,
                snapshot,
            )
            for key, binding in merged_overrides.items():
                if key in camera_keys:
                    current_camera_bindings[key] = copy.deepcopy(binding)
            completed_overrides = self._complete_camera_binding_swaps(
                current_camera_bindings,
                overrides,
                camera_keys,
            )
            requested_keys = {str(key) for key in completed_overrides}
            for raw_key, raw_binding in completed_overrides.items():
                key = str(raw_key)
                if not isinstance(raw_binding, Mapping):
                    raise RuntimeConfigError(
                        f"override for {key} must be an object"
                    )
                if raw_binding:
                    merged_overrides[key] = copy.deepcopy(dict(raw_binding))
                else:
                    merged_overrides.pop(key, None)
            snapshot = self._preserve_unchanged_camera_bindings(
                snapshot,
                current_camera_bindings,
                requested_keys,
            )
            effective_config, bindings = self._effective_from_snapshot(
                self._base_config, merged_overrides, snapshot
            )
            effective_config = self._apply_control_panel_safety_overrides(
                effective_config
            )
            effective_errors = validate_runtime_config(effective_config)
            if effective_errors:
                raise RuntimeConfigError('; '.join(effective_errors))
            runtime_config = self._runtime_config_for_gripper(
                effective_config
            )
            desired_camera_bindings = self._camera_bindings_from_config(
                effective_config,
                snapshot,
            )
            camera_physical_noops = {
                key
                for key in camera_keys
                if current_camera_bindings.get(key, {}).get('capture_device')
                == desired_camera_bindings.get(key, {}).get('capture_device')
            }
            workflow = self.workflow
            if workflow is None:
                raise DeviceConflictError(
                    "Select a config before applying runtime bindings"
                )
            current_by_key = self._device_configs_by_key(starter.config)
            candidate_by_key = self._device_configs_by_key(runtime_config)
            changed_keys = sorted(
                key
                for key in current_by_key
                if current_by_key.get(key) != candidate_by_key.get(key)
            )
            operation_keys = [
                key for key in changed_keys if key not in camera_physical_noops
            ]
            applied_keys: set[str] = set()
            reported_keys: list[str] = []
            try:
                for key in operation_keys:
                    self._begin_device_operation_locked(key, 'binding')
                    reserved_keys.append(key)
                applied_keys = self._apply_binding_config_locked(
                    starter,
                    runtime_config,
                    physical_noop_keys=camera_physical_noops,
                )
                workflow.update_runtime_config(runtime_config)
                self._effective_config = copy.deepcopy(effective_config)
                self._effective_bindings = copy.deepcopy(bindings)
                self._binding_discovery_revision = (
                    snapshot.revision if bindings else None
                )
                self._bump_revision()
                reported_keys = sorted(applied_keys) or sorted(requested_keys)
                self.logs.append(
                    'INFO',
                    "Updated runtime bindings in place for "
                    + ", ".join(reported_keys),
                )
            finally:
                for key in reserved_keys:
                    self._end_device_operation_locked(key)
            return self.state()

    def discovery(
        self,
        config_name: Optional[str] = None,
        *,
        probe_cameras: bool = False,
        probe_robots: bool = False,
        refresh_robots: bool = False,
    ) -> Dict[str, Any]:
        target_name: Optional[str]
        base_config: Optional[Dict[str, Any]]
        if config_name:
            target_name = normalize_config_name(config_name)
            base_config = compose_runtime_config(
                target_name, config_dir=self.config_dir
            )
        else:
            with self._lock:
                target_name = self.selected_config
                base_config = copy.deepcopy(self._base_config)
        if probe_robots and hasattr(
            self.hardware_discovery, 'discover_with_robot_probe'
        ):
            snapshot = self.hardware_discovery.discover_with_robot_probe(
                probe_camera_frames=probe_cameras,
                force=refresh_robots,
            )
        elif probe_cameras and hasattr(
            self.hardware_discovery, 'discover_with_camera_probe'
        ):
            snapshot = self.hardware_discovery.discover_with_camera_probe()
        else:
            snapshot = self.hardware_discovery.discover()
        self._remember_discovery_snapshot(snapshot)
        payload = snapshot.public_payload()
        payload['selected_config'] = self.selected_config_name()
        payload['config_name'] = target_name
        payload['candidate_config'] = target_name
        if target_name and base_config is not None:
            recommended, configured = recommend_overrides(base_config, snapshot)
        else:
            recommended, configured = {}, {}
        payload['recommended_overrides'] = recommended
        payload['configured_detection'] = configured
        return payload

    def inspect_config(self, config_name: str) -> Dict[str, Any]:
        name = normalize_config_name(config_name)
        base = inspect_runtime_config(name, self.config_dir)
        payload = dict(base)
        payload['base_config'] = {
            'name': base['name'],
            'devices': copy.deepcopy(base['devices']),
            'home': copy.deepcopy(base['home']),
            'configured_tools': copy.deepcopy(base['configured_tools']),
        }
        with self._lock:
            if name == self.selected_config and self._effective_config is not None:
                payload['effective_config'] = summarize_runtime_config(
                    name, self._effective_config
                )
                payload['effective_bindings'] = copy.deepcopy(
                    self._effective_bindings
                )
                payload['discovery_revision'] = self._binding_discovery_revision
            else:
                payload['effective_config'] = None
                payload['effective_bindings'] = {}
                payload['discovery_revision'] = None
        return payload

    def preflight(
        self,
        config_name: Optional[str] = None,
        *,
        probe_cameras: bool = False,
        probe_robots: bool = False,
        refresh_robots: bool = False,
    ) -> Dict[str, Any]:
        """Return a unified, read-only startup checklist.

        Discovery is intentionally passive by default. Camera frame and robot
        reachability probes are opt-in because they can open hardware handles;
        this method still never starts a device or policy process.
        """
        target_name: Optional[str]
        if config_name:
            target_name = normalize_config_name(config_name)
        else:
            with self._lock:
                target_name = self.selected_config

        config_summary: Optional[Dict[str, Any]] = None
        if target_name:
            try:
                composed = compose_runtime_config(
                    target_name, config_dir=self.config_dir
                )
                config_summary = summarize_runtime_config(target_name, composed)
            except RuntimeConfigError as exc:
                config_summary = {
                    "name": target_name,
                    "valid": False,
                    "errors": [str(exc)],
                    "metadata": runtime_config_metadata(
                        target_name, None, errors=[str(exc)]
                    ),
                }

        selected_name = self.selected_config_name()
        state: Optional[Dict[str, Any]] = None
        if target_name and target_name == selected_name:
            state = self.state()
            effective = state.get("effective_config")
            if isinstance(effective, Mapping):
                config_summary = dict(effective)

        discovery_payload: Optional[Dict[str, Any]] = None
        discovery_error: Optional[str] = None
        if target_name or config_summary is not None:
            try:
                discovery_payload = self.discovery(
                    target_name,
                    probe_cameras=probe_cameras,
                    probe_robots=probe_robots,
                    refresh_robots=refresh_robots,
                )
            except Exception as exc:
                discovery_error = f"{type(exc).__name__}: {exc}"

        report = build_preflight_report(
            config_summary,
            discovery=discovery_payload,
            grippers=(state or {}).get("grippers") if state else None,
            workflow=(state or {}).get("workflow") if state else None,
            devices=(state or {}).get("devices") if state else None,
            discovery_error=discovery_error,
        )
        report["selected_config"] = selected_name
        report["discovery_revision"] = (
            discovery_payload.get("revision") if discovery_payload else None
        )
        if config_summary:
            report["config"] = {
                "name": config_summary.get("name"),
                "metadata": copy.deepcopy(
                    config_summary.get("metadata")
                    or runtime_config_metadata(
                        str(config_summary.get("name") or target_name or ""),
                        config_summary,
                    )
                ),
                "execution_profile": copy.deepcopy(
                    config_summary.get("execution_profile")
                ),
            }
        else:
            report["config"] = None
        return report

    def runtime_metrics(self) -> Dict[str, Any]:
        """Return the current allowlisted runtime telemetry projection."""
        state = self.state()
        summary = state.get("effective_config") or state.get("base_config")
        return build_runtime_metrics(
            state,
            summary if isinstance(summary, Mapping) else None,
        )

    def runtime_tuning_schema(self) -> Dict[str, Any]:
        """Return restart scopes and safe tuning fields for the debug UI."""
        state = self.state()
        summary = state.get("effective_config") or state.get("base_config")
        payload = build_runtime_tuning_schema(
            summary if isinstance(summary, Mapping) else None
        )
        # A tuning schema is valid only for the observed selected config and
        # revision. The frontend rejects stale schemas after another client
        # changes bindings or switches presets.
        payload["selected_config"] = state.get("selected_config")
        payload["revision"] = state.get("revision")
        return payload


    @staticmethod
    def _generated_config_calibration_artifacts(
        grippers: Mapping[str, Any],
    ) -> Dict[str, Any]:
        # Keep only identity-bearing calibration provenance for exported YAML.
        sides = grippers.get('sides')
        if not isinstance(sides, Mapping):
            return {}
        result: Dict[str, Any] = {}
        for raw_side, raw_state in sides.items():
            side = str(raw_side)
            if side not in {'left', 'right'} or not isinstance(raw_state, Mapping):
                continue
            artifact = raw_state.get('calibration_artifact')
            if not isinstance(artifact, Mapping):
                continue
            item: Dict[str, Any] = {
                key: artifact[key]
                for key in (
                    'name',
                    'present',
                    'limits_valid',
                    'valid',
                    'status',
                    'identity_status',
                    'sha256',
                    'mtime_ns',
                )
                if key in artifact
            }
            for identity_key in ('expected', 'observed'):
                identity = artifact.get(identity_key)
                if not isinstance(identity, Mapping):
                    continue
                item[identity_key] = {
                    key: identity[key]
                    for key in ('motor_id', 'port')
                    if key in identity
                }
            if item:
                result[side] = item
        return result

    def export_generated_config(
        self,
        *,
        generated_id: str,
        expected_selected_config: str,
        expected_revision: int,
        tuning_overrides: Mapping[str, Any],
        overwrite: bool,
    ) -> Dict[str, Any]:
        # Persist a stopped session as a new, explicit native Hydra preset.
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
        ):
            raise RuntimeConfigError('expected_revision must be an integer')
        if not isinstance(expected_selected_config, str) or not expected_selected_config:
            raise RuntimeConfigError(
                'expected_selected_config must be a non-empty string'
            )
        if not isinstance(tuning_overrides, Mapping):
            raise RuntimeConfigError('tuning_overrides must be an object')
        if not isinstance(overwrite, bool):
            raise RuntimeConfigError('overwrite must be a boolean')

        with self._gripper_permission_mutation_guard(), self._lock:
            self._ensure_mutable_locked()
            if (
                self.selected_config != expected_selected_config
                or self.revision != expected_revision
            ):
                raise DeviceConflictError(
                    'Config changed before generated configuration export'
                )
            if self._effective_config is None:
                raise DeviceConflictError(
                    'Select a config before exporting a generated configuration'
                )

            effective_config = self._apply_gripper_port_overrides(
                self._direct_gripper_config(self._effective_config),
                self._gripper_port_overrides,
            )
            gripper_state = self.grippers.state()
            bindings = copy.deepcopy(self._effective_bindings)
            effective_config, bindings = self._camera_bindings_as_capture_ids(
                effective_config,
                bindings,
            )
            if self._gripper_port_overrides:
                bindings['gripper_ports'] = copy.deepcopy(
                    self._gripper_port_overrides
                )
            try:
                export = write_generated_config(
                    effective_config,
                    config_dir=self.config_dir,
                    source_config_name=self.selected_config,
                    revision=self.revision,
                    generated_id=generated_id,
                    bindings=bindings,
                    calibration_artifacts=self._generated_config_calibration_artifacts(
                        gripper_state
                    ),
                    tuning_overrides=tuning_overrides,
                    overwrite=overwrite,
                )
            except GeneratedConfigExportError as exc:
                raise RuntimeConfigError(str(exc)) from exc

            payload = export.as_dict()
            payload.update(
                {
                    'ok': True,
                    'selected_config': self.selected_config,
                    'revision': self.revision,
                }
            )
            self.logs.append(
                'INFO',
                'Exported generated config '
                + f"{payload['config_name']} from {self.selected_config}",
            )
            return payload

    def save_current_config(
        self,
        *,
        expected_selected_config: str,
        expected_revision: int,
        tuning_overrides: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Persist current settings without mutating an active device lifecycle."""
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise RuntimeConfigError('expected_revision must be an integer')
        if not isinstance(expected_selected_config, str) or not expected_selected_config:
            raise RuntimeConfigError('expected_selected_config must be a non-empty string')
        if not isinstance(tuning_overrides, Mapping):
            raise RuntimeConfigError('tuning_overrides must be an object')

        with self._gripper_permission_mutation_guard(), self._lock:
            self._ensure_source_config_saveable_locked()
            if (
                self.selected_config != expected_selected_config
                or self.revision != expected_revision
            ):
                raise DeviceConflictError('Config changed before source configuration save')
            if not isinstance(self._effective_config, Mapping):
                raise DeviceConflictError('Select a config before saving its source configuration')
            reload_deferred = self._has_active_devices_locked()

            persisted = self._apply_gripper_port_overrides(
                self._direct_gripper_config(self._effective_config),
                self._gripper_port_overrides,
            )
            persisted, persisted_bindings = self._camera_bindings_as_capture_ids(
                persisted,
                self._effective_bindings,
            )
            rollout_dirty = bool(getattr(self, '_rollout_settings_dirty', False))
            output_root = getattr(self, '_rollout_output_root', None)
            if output_root is None and rollout_dirty:
                output_root = configured_recording_root(
                    persisted,
                    repo_root=self._recording_repo_root(),
                )
            if output_root is not None or rollout_dirty:
                policy = persisted.get('policy')
                if not isinstance(policy, dict):
                    policy = {}
                    persisted['policy'] = policy
                rollout = policy.get('infer_rollout')
                if not isinstance(rollout, dict):
                    rollout = {}
                    policy['infer_rollout'] = rollout
                if output_root is not None:
                    rollout['output_dir'] = str(output_root)
                if rollout_dirty:
                    rollout['enabled'] = bool(self._rollout_enabled)
                    rollout['required'] = bool(self._rollout_enabled)

            try:
                export = write_source_config(
                    persisted,
                    config_dir=self.config_dir,
                    source_config_name=self.selected_config,
                    revision=self.revision,
                    tuning_overrides=tuning_overrides,
                )
            except GeneratedConfigExportError as exc:
                raise RuntimeConfigError(str(exc)) from exc

            self._rollout_settings_dirty = False
            if reload_deferred:
                # Replacing the lifecycle would stop or detach running devices.
                # Keep the live session untouched; a later config selection or
                # Control Panel restart will load the source file just written.
                self._bump_revision()
            else:
                # Camera IDs may change lifecycle keys. Recompose and replace
                # stopped managers so the next start uses the saved source.
                reloaded_base = compose_runtime_config(
                    self.selected_config,
                    config_dir=self.config_dir,
                )
                errors = validate_runtime_config(reloaded_base)
                if errors:
                    raise RuntimeConfigError('; '.join(errors))
                reloaded_effective = self._apply_control_panel_safety_overrides(
                    reloaded_base
                )
                runtime_config = self._runtime_config_for_gripper(
                    reloaded_effective,
                    port_overrides={},
                )
                candidate = DeviceLifecycleManager(config=runtime_config)
                candidate_workflow = self._make_workflow(
                    self.selected_config,
                    runtime_config,
                    preserve_inference_options=False,
                )
                self._replace_starter_locked(
                    name=self.selected_config,
                    base_config=reloaded_base,
                    effective_config=reloaded_effective,
                    bindings=persisted_bindings,
                    discovery_revision=(
                        self._binding_discovery_revision
                        if persisted_bindings
                        else None
                    ),
                    candidate=candidate,
                    candidate_workflow=candidate_workflow,
                    stop_active=False,
                    gripper_port_overrides={},
                )
            payload = export.as_dict()
            payload.update(
                {
                    'ok': True,
                    'selected_config': self.selected_config,
                    'revision': self.revision,
                    'tuning_overrides': copy.deepcopy(dict(tuning_overrides)),
                    'reload_deferred': reload_deferred,
                }
            )
            self.logs.append(
                'INFO',
                f'Saved current panel state to source config {self.selected_config}'
                + ('; lifecycle reload deferred' if reload_deferred else ''),
            )
            return payload

    def _assert_start_preflight_locked(self, stage: str) -> None:
        """Reject unsafe starts after config selection, before any worker opens."""
        if stage not in {'runtime', 'server', 'inference'}:
            return
        # A few embedding tests install lifecycle doubles directly. They do not
        # have a composed config, so retain their explicit low-level contract.
        if self._base_config is None or self.selected_config is None:
            return
        if self.starter is None or self.workflow is None:
            raise DeviceConflictError('Select a config before starting workflow')

        try:
            devices = self.starter.get_status().get('devices', [])
        except Exception as exc:
            raise WorkflowConflictError(
                f'启动前无法读取设备状态：{type(exc).__name__}: {exc}'
            ) from exc

        workflow_state: Dict[str, Any] = {}
        observe = getattr(self.workflow, 'observe', None)
        state_method = getattr(self.workflow, 'state', None)
        try:
            if callable(observe):
                workflow_state = dict(observe(devices))
            elif callable(state_method):
                workflow_state = dict(state_method(devices))
        except Exception as exc:
            raise WorkflowConflictError(
                f'启动前无法读取 workflow 状态：{type(exc).__name__}: {exc}'
            ) from exc

        if not workflow_state:
            preparation = getattr(
                self.workflow, 'runtime_preparation_blockers', None
            )
            blockers = list(preparation(devices)) if callable(preparation) else []
            workflow_state = {
                'runtime': {
                    'status': 'stopped',
                    'active': False,
                    'blockers': blockers,
                    'preparation_blockers': blockers,
                },
                'inference': {
                    'status': 'stopped',
                    'running': False,
                    'blockers': [],
                },
            }

        summary = summarize_runtime_config(
            self.selected_config,
            self._effective_config or self._base_config,
        )
        # FastWAM's model Server is deliberately independent from physical
        # devices. Avoid probing serial permissions or scanning gripper
        # holders while that process is loading; the server-stage filter below
        # still keeps gripper diagnostics available for runtime/inference.
        gripper_state = (
            None
            if stage == 'server'
            else self._gripper_state_locked(devices, workflow_state)
        )
        # Discovery is deliberately omitted here: lifecycle startup owns
        # starting configured devices, while the read-only preflight endpoint
        # separately verifies physical identity. Do not turn an unavailable
        # passive scan into a race with worker startup.
        report = build_preflight_report(
            summary,
            discovery=None,
            grippers=gripper_state,
            workflow=workflow_state,
            devices=devices,
        )
        blockers: list[str] = []
        for check in report.get('checks', []):
            if not isinstance(check, Mapping):
                continue
            check_id = str(check.get('id') or '')
            status = str(check.get('status') or '')
            # Camera/robot discovery and the aggregate workflow row are
            # intentionally stage-specific. Runtime owns opening logical
            # devices; inference owns checkpoint/consumer readiness.
            if check_id in {
                'camera_slots',
                'robot_slots',
                'hardware_discovery',
                'workflow_idle',
            } or (
                # Loading a FastWAM model is independent from physical
                # devices and gripper calibration. Those checks remain part
                # of runtime/inference preflight but must not gate server
                # startup.
                stage == 'server'
                and check_id in {'execution_profile', 'gripper_calibration'}
            ):
                continue
            if status not in {'blocking', 'unknown'}:
                continue
            evidence = check.get('evidence')
            evidence_map = evidence if isinstance(evidence, Mapping) else {}
            details: list[str] = []
            for field in (
                'errors',
                'issues',
                'blockers',
                'missing',
                'invalid_config_identity',
                'identity_mismatch',
                'missing_ports',
                'invalid_artifacts',
            ):
                value = evidence_map.get(field)
                if isinstance(value, (list, tuple)):
                    details.extend(str(item) for item in value if item)
                elif value:
                    details.append(str(value))
            if details:
                blockers.extend(details[:8])
            else:
                remediation = check.get('remediation')
                if remediation:
                    blockers.append(str(remediation))

        workflow_map = (
            workflow_state if isinstance(workflow_state, Mapping) else {}
        )
        if stage == 'server':
            # FastWAM loads the heavyweight model in an independent Server
            # process. Starting it is intentionally allowed before the device
            # runtime; only the server's own artifact/endpoint checks apply.
            if str(workflow_map.get('kind') or '') != 'fastwam':
                raise WorkflowConflictError(
                    'The selected config does not use a model Server stage'
                )
            server_state = (
                workflow_map.get('server')
                if isinstance(workflow_map.get('server'), Mapping)
                else {}
            )
            server_blockers = server_state.get('blockers')
            if isinstance(server_blockers, (list, tuple)):
                blockers.extend(str(item) for item in server_blockers if item)
            if not self._inference_test_mode:
                checkpoint = self._checkpoint_state_locked(workflow_state)
                if checkpoint.get('valid') is not True:
                    blockers.append(
                        str(
                            checkpoint.get('error')
                            or '当前 checkpoint 无效'
                        )
                    )
            blockers = list(dict.fromkeys(item[:512] for item in blockers if item))
            if blockers:
                raise WorkflowConflictError(
                    '启动前预检未通过：' + '；'.join(blockers[:8])
                )
            return
        runtime_state = (
            workflow_map.get('runtime')
            if isinstance(workflow_map.get('runtime'), Mapping)
            else {}
        )
        inference_state = (
            workflow_map.get('inference')
            if isinstance(workflow_map.get('inference'), Mapping)
            else {}
        )
        active_statuses = {'starting', 'running', 'stopping'}
        if stage == 'runtime':
            if (
                runtime_state.get('active')
                or inference_state.get('running')
                or runtime_state.get('status') in active_statuses
                or inference_state.get('status') in active_statuses
            ):
                blockers.append('当前 runtime/inference 仍在运行，请先停止')
            preparation = runtime_state.get('preparation_blockers')
            if isinstance(preparation, (list, tuple)):
                blockers.extend(str(item) for item in preparation if item)
            if runtime_state.get('error'):
                blockers.append(str(runtime_state['error']))
        else:
            if runtime_state.get('ready') is not True:
                blockers.append('请先正确启动 DeviceManager 与 ActionExecutor')
            inference_blockers = inference_state.get('blockers')
            if isinstance(inference_blockers, (list, tuple)):
                blockers.extend(str(item) for item in inference_blockers if item)
            if inference_state.get('error'):
                blockers.append(str(inference_state['error']))
            if inference_state.get('running') or inference_state.get('status') in active_statuses:
                blockers.append('Inference 已在运行，请先停止')
            if not self._inference_test_mode:
                checkpoint = self._checkpoint_state_locked(workflow_state)
                if checkpoint.get('valid') is not True:
                    blockers.append(
                        str(
                            checkpoint.get('error')
                            or '当前 checkpoint 无效'
                        )
                    )
        blockers = list(dict.fromkeys(item[:512] for item in blockers if item))
        if blockers:
            raise WorkflowConflictError(
                '启动前预检未通过：' + '；'.join(blockers[:8])
            )

    def _require_starter_locked(self) -> DeviceLifecycleManager:
        if self._shutdown:
            raise DeviceConflictError("Device control server is shutting down")
        starter = self.starter
        if starter is None:
            raise DeviceConflictError("Select a config before controlling devices")
        return starter

    def state(self) -> Dict[str, Any]:
        with self._lock:
            if self.starter:
                lifecycle = self.starter.get_status()
                devices = lifecycle['devices']
            else:
                lifecycle = {
                    'running': False,
                    'monitor_running': False,
                    'total_devices': 0,
                    'device_manager_running': False,
                    'policy_runner_running': False,
                    'devices': [],
                }
                devices = []

            active = any(
                device['running']
                or device['status'] in {'external', 'orphaned'}
                or device['desired_state'] == 'running'
                for device in devices
            )
            if self.workflow:
                observe = getattr(self.workflow, 'observe', None)
                workflow = (
                    observe(devices)
                    if callable(observe)
                    else self.workflow.state(devices)
                )
            else:
                workflow = {
                    'config_name': None,
                    'kind': None,
                    'inference_label': None,
                    'checkpoint_owner_stage': 'inference',
                    'runtime': {
                        'status': 'stopped',
                        'active': False,
                        'ready': False,
                        'can_start': False,
                        'can_stop': False,
                        'blockers': ['请先选择运行配置'],
                    },
                    'inference': {
                        'status': 'stopped',
                        'running': False,
                        'can_start': False,
                        'can_stop': False,
                        'blockers': ['请先选择运行配置'],
                        'reset': {
                            'available': False,
                            'can': False,
                            'pending_command': None,
                            'last_command': None,
                            'blockers': ['请先选择运行配置'],
                            'error': None,
                        },
                    },
                    'profiles': {},
                    'recording': empty_recording_state(
                        configured=False,
                        enabled=False,
                        required=True,
                        blockers=['请先选择运行配置'],
                    ),
                    'stop_order': [
                        'recording',
                        'inference',
                        'action_executor',
                        'device_manager',
                        'devices',
                    ],
                }
            active = active or bool(
                workflow['runtime'].get('active')
                or workflow['inference'].get('running')
                or (
                    isinstance(workflow.get('server'), Mapping)
                    and workflow['server'].get('active')
                )
            )
            grippers = self._gripper_state_locked(devices, workflow)
            active = active or bool(grippers.get('active'))
            checkpoint = self._checkpoint_state_locked(workflow)
            inference_options = self._inference_options_state_locked(workflow)
            base_summary = (
                summarize_runtime_config(self.selected_config, self._base_config)
                if self.selected_config and self._base_config is not None
                else None
            )
            effective_summary = (
                summarize_runtime_config(
                    self.selected_config, self._effective_config
                )
                if self.selected_config and self._effective_config is not None
                else None
            )
            return {
                'revision': self.revision,
                'selected_config': self.selected_config,
                'active_config': self.selected_config if active else None,
                'base_config': base_summary,
                'effective_config': effective_summary,
                'effective_bindings': copy.deepcopy(self._effective_bindings),
                'discovery_revision': self._binding_discovery_revision,
                'devices': devices,
                'lifecycle': lifecycle,
                'workflow': workflow,
                'checkpoint': checkpoint,
                'inference_options': inference_options,
                'recording': {
                    **copy.deepcopy(workflow['recording']),
                    'camera_roles': self._recording_camera_roles_locked(devices),
                },
                'conversion': self.conversion_jobs.state(),
                'grippers': grippers,
                'server': {
                    'pid': os.getpid(),
                    'hostname': socket.gethostname(),
                    'started_at': self.started_at,
                    'force_stop_in_progress': self._force_stop_in_progress,
                    'manager_auto_start': False,
                    'policy_auto_start': False,
                    'packaged_release': bool(
                        os.environ.get('SUPERINFERENCE_RELEASE_ROOT')
                    ),
                    'elements_version': os.environ.get(
                        'SUPERINFERENCE_ELEMENTS_VERSION'
                    ),
                    'flexivrdk_override': os.environ.get(
                        'SUPERINFERENCE_FLEXIV_RDK_VERSION'
                    ),
                    'flexivrdk_version': os.environ.get(
                        'SUPERINFERENCE_FLEXIV_RDK_VERSION'
                    ),
                    'flexivrdk_versions_available': [
                        item
                        for item in os.environ.get(
                            'SUPERINFERENCE_FLEXIV_RDK_VERSIONS', ''
                        ).split(',')
                        if item
                    ],
                },
            }

    def _assert_runtime_start_context_locked(
        self,
        starter: Optional[DeviceLifecycleManager],
        workflow: WorkflowLifecycle,
        selected_config: Optional[str],
        *,
        check_cancellation: bool = True,
    ) -> None:
        if self._shutdown:
            raise DeviceConflictError(
                'Device control server is shutting down'
            )
        if (
            self.starter is not starter
            or self.workflow is not workflow
            or self.selected_config != selected_config
        ):
            raise DeviceConflictError(
                '运行配置在启动过程中发生变化，请基于最新状态重试'
            )

        if check_cancellation:
            self._raise_if_workflow_cancelled_locked()

    @staticmethod
    def _runtime_device_reason(device: Mapping[str, Any]) -> str:
        safety = (
            device.get('safety')
            if isinstance(device.get('safety'), Mapping)
            else {}
        )
        return str(
            device.get('last_error')
            or safety.get('reason')
            or device.get('status')
            or 'unknown state'
        )

    @staticmethod
    def _runtime_device_recovery_action(
        key: str,
        device: Mapping[str, Any],
        started: set[str],
        restarted: set[str],
    ) -> Optional[str]:
        if device.get('ready') is True:
            return None
        status = str(device.get('status') or 'unknown').lower()
        running = device.get('running') is True
        safety = (
            device.get('safety')
            if isinstance(device.get('safety'), Mapping)
            else {}
        )
        if (
            (
                safety.get('fail_closed') is True
                and safety.get('initializing') is not True
            )
            or safety.get('estop') is True
            or safety.get('operational') is False
            or safety.get('reduced') is True
        ):
            raise WorkflowConflictError(
                f'必需设备 {key} 处于安全互锁状态，禁止自动重启：'
                + DeviceController._runtime_device_reason(device)
            )
        if status in {'external', 'orphaned', 'conflict', 'stop_timeout'}:
            raise WorkflowConflictError(
                f'必需设备 {key} 无法自动恢复：'
                + DeviceController._runtime_device_reason(device)
            )
        if status in {'starting', 'stopping'}:
            return None
        if status == 'stopped':
            if key not in started:
                return 'start'
            if key not in restarted:
                return 'restart'
        elif status in {'failed', 'degraded'} or running:
            if key not in restarted:
                return 'restart'
        else:
            return None
        raise WorkflowConflictError(
            f'必需设备 {key} 自动恢复后仍不可用：'
            + DeviceController._runtime_device_reason(device)
        )

    def _start_runtime_with_required_devices(
        self,
        starter: DeviceLifecycleManager,
        workflow: WorkflowLifecycle,
        selected_config: Optional[str],
    ) -> tuple[Dict[str, Any], list[Dict[str, Any]], int]:
        with self._lock:
            self._assert_runtime_start_context_locked(
                starter,
                workflow,
                selected_config,
            )

        devices = starter.get_status().get('devices', [])
        preparation_blockers = workflow.runtime_preparation_blockers(devices)
        if preparation_blockers:
            raise WorkflowConflictError('; '.join(preparation_blockers))
        required_keys = set(workflow.required_device_keys())
        deadline = time.monotonic() + RUNTIME_DEVICE_READY_TIMEOUT_S
        started: set[str] = set()
        restarted: set[str] = set()
        device_actions: list[Dict[str, Any]] = []

        while True:
            with self._lock:
                self._assert_runtime_start_context_locked(
                    starter,
                    workflow,
                    selected_config,
                )
                cancelled = sorted(
                    required_keys & self._workflow_cancelled_device_keys
                )
                if cancelled:
                    raise WorkflowConflictError(
                        '运行时启动已被设备停止操作取消：'
                        + '、'.join(cancelled)
                    )
            devices = starter.get_status().get('devices', [])
            status_by_key = {
                str(device.get('key')): device
                for device in devices
                if isinstance(device, Mapping) and device.get('key')
            }
            missing = sorted(required_keys - set(status_by_key))
            if missing:
                raise WorkflowConflictError(
                    '运行配置中的必需设备不存在：' + '、'.join(missing)
                )

            if all(
                status_by_key[key].get('ready') is True
                for key in required_keys
            ):
                with self._lock:
                    self._assert_runtime_start_context_locked(
                        starter,
                        workflow,
                        selected_config,
                    )
                    latest_devices = starter.get_status().get('devices', [])
                    latest_by_key = {
                        str(device.get('key')): device
                        for device in latest_devices
                        if isinstance(device, Mapping) and device.get('key')
                    }
                    if not all(
                        latest_by_key.get(key, {}).get('ready') is True
                        for key in required_keys
                    ):
                        continue
                # Process creation may be slow. The workflow reservation keeps
                # config mutation and competing start actions out while the
                # controller lock remains available to authoritative sampling.
                workflow_state = workflow.start_runtime(latest_devices)
                with self._lock:
                    self._assert_runtime_start_context_locked(
                        starter,
                        workflow,
                        selected_config,
                    )
                    cancelled = sorted(
                        required_keys & self._workflow_cancelled_device_keys
                    )
                    if cancelled:
                        cancellation_error = WorkflowConflictError(
                            '运行时启动已被设备停止操作取消：'
                            + '、'.join(cancelled)
                        )
                    else:
                        cancellation_error = None
                        revision = self._bump_revision()
                        self.logs.append('INFO', 'runtime start requested')
                        self._end_workflow_operation_locked()
                        return workflow_state, device_actions, revision
                try:
                    workflow.stop_all()
                except BaseException as exc:
                    raise WorkflowConflictError(
                        f'{cancellation_error}; runtime rollback failed: '
                        f'{type(exc).__name__}: {exc}'
                    ) from exc
                raise cancellation_error

            actions: list[tuple[str, str]] = []
            for key in sorted(required_keys):
                action = self._runtime_device_recovery_action(
                    key,
                    status_by_key[key],
                    started,
                    restarted,
                )
                if action is not None:
                    actions.append((key, action))

            for key, requested_action in actions:
                device_class, device_class_id = key.rsplit(':', 1)
                with self._lock:
                    self._assert_runtime_start_context_locked(
                        starter,
                        workflow,
                        selected_config,
                    )
                    if key in self._workflow_cancelled_device_keys:
                        raise WorkflowConflictError(
                            f'运行时启动已被设备停止操作取消：{key}'
                        )
                    current = starter.get_device_status(
                        device_class,
                        device_class_id,
                    )
                    action = self._runtime_device_recovery_action(
                        key,
                        current,
                        started,
                        restarted,
                    )
                    if action is None:
                        continue
                    if action != requested_action:
                        requested_action = action
                    self._ensure_can_ready_for_device_locked(
                        device_class, device_class_id
                    )
                    self._begin_device_operation_locked(
                        key,
                        requested_action,
                        allow_workflow=True,
                    )

                try:
                    if requested_action == 'start':
                        observed_start = getattr(
                            starter,
                            'start_device_observed',
                            starter.start_device,
                        )
                        result = observed_start(device_class, device_class_id)
                        started.add(key)
                    else:
                        observed_restart = getattr(
                            starter,
                            'restart_device_observed',
                            starter.restart_device,
                        )
                        result = observed_restart(device_class, device_class_id)
                        restarted.add(key)
                except BaseException:
                    with self._lock:
                        self._end_device_operation_locked(key)
                    raise

                with self._lock:
                    try:
                        self._assert_runtime_start_context_locked(
                            starter,
                            workflow,
                            selected_config,
                        )
                        self._bump_revision()
                        self.logs.append(
                            'INFO',
                            f'runtime auto-{requested_action} requested for {key}',
                            device_key=key,
                        )
                        device_actions.append(
                            {
                                'key': key,
                                'action': requested_action,
                                'status': result.get('status'),
                            }
                        )
                    finally:
                        self._end_device_operation_locked(key)

            if actions:
                continue
            if time.monotonic() >= deadline:
                details = [
                    (
                        f"{key}={status_by_key[key].get('status', 'unknown')}"
                        f" ({self._runtime_device_reason(status_by_key[key])})"
                    )
                    for key in sorted(required_keys)
                    if status_by_key[key].get('ready') is not True
                ]
                raise WorkflowConflictError(
                    '等待必需设备就绪超时：' + '；'.join(details)
                )
            time.sleep(RUNTIME_DEVICE_POLL_INTERVAL_S)

    def workflow_start_requirements(self, stage: str) -> Dict[str, bool]:
        with self._lock:
            workflow = self.workflow
            if workflow is None:
                raise DeviceConflictError(
                    'Select a config before controlling the runtime workflow'
                )
            if stage == 'server' and getattr(
                workflow, 'workflow_kind', 'openpi'
            ) != 'fastwam':
                raise DeviceControlError(
                    'The selected config does not use a model Server stage'
                )
            return {
                'checkpoint': (
                    stage
                    == getattr(
                        workflow,
                        'checkpoint_owner_stage',
                        'inference',
                    )
                ),
                'options': stage == 'inference',
            }

    def control_workflow(
        self,
        stage: str,
        action: str,
        *,
        expected_selected_config: Optional[str] = None,
        expected_revision: Optional[int] = None,
        expected_generation: Optional[int] = None,
        expected_options_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        if (stage, action) not in {
            ('runtime', 'start'),
            ('runtime', 'stop'),
            ('server', 'start'),
            ('server', 'stop'),
            ('inference', 'start'),
            ('inference', 'stop'),
        }:
            raise DeviceControlError(
                f"Unknown workflow action: {stage}/{action}"
            )
        with self._lock:
            starter = self._require_starter_locked()
            workflow = self.workflow
            if workflow is None:
                raise DeviceConflictError(
                    "Select a config before controlling the runtime workflow"
                )
            if stage == 'server' and getattr(
                workflow, 'workflow_kind', 'openpi'
            ) != 'fastwam':
                raise DeviceControlError(
                    'The selected config does not use a model Server stage'
                )
            if action == 'start':
                checkpoint_required = stage == getattr(
                    workflow,
                    'checkpoint_owner_stage',
                    'inference',
                )
                options_required = stage == 'inference'
                if checkpoint_required:
                    if (
                        expected_selected_config is None
                        or expected_revision is None
                        or expected_generation is None
                    ):
                        raise RuntimeConfigError(
                            f'{stage} start requires checkpoint context fields'
                        )
                    self._validate_checkpoint_context_locked(
                        expected_selected_config=expected_selected_config,
                        expected_revision=expected_revision,
                        expected_generation=expected_generation,
                        require_editable=False,
                    )
                    if (
                        not self._inference_test_mode
                        and self._checkpoint_override is not None
                    ):
                        canonical_override = self._canonical_checkpoint_path(
                            str(self._checkpoint_override),
                            require_directory=(
                                not self._checkpoint_requires_params_locked()
                            ),
                            allow_file=self._checkpoint_accepts_file_locked(),
                        )
                        if canonical_override != self._checkpoint_override:
                            raise DeviceConflictError(
                                'Checkpoint override identity changed; select it again'
                            )
                if options_required:
                    if (
                        expected_selected_config is None
                        or expected_revision is None
                        or expected_options_generation is None
                    ):
                        raise RuntimeConfigError(
                            'Inference start requires options context fields'
                        )
                    self._validate_inference_options_context_locked(
                        expected_selected_config=expected_selected_config,
                        expected_revision=expected_revision,
                        expected_generation=expected_options_generation,
                        require_editable=False,
                    )
                self._assert_start_preflight_locked(stage)
            selected_config = self.selected_config
            workflow_action = f'{stage}:{action}'
            self._begin_workflow_operation_locked(workflow_action)

        try:
            if action == 'start' and stage != 'server':
                self.grippers.stop(f'{stage}_start')

            if stage == 'runtime' and action == 'start':
                workflow_state, device_actions, revision = (
                    self._start_runtime_with_required_devices(
                        starter,
                        workflow,
                        selected_config,
                    )
                )
                return {
                    'ok': True,
                    'stage': stage,
                    'action': action,
                    'revision': revision,
                    'device_actions': device_actions,
                    'workflow': workflow_state,
                }

            devices = starter.get_status().get('devices', [])
            if stage == 'runtime':
                workflow_state = workflow.stop_runtime(devices)
            elif stage == 'server':
                if action == 'start':
                    workflow_state = workflow.start_server(devices)
                else:
                    workflow_state = workflow.stop_server(devices)
            elif action == 'start':
                workflow_state = workflow.start_inference(devices)
            else:
                workflow_state = workflow.stop_inference(devices)
            with self._lock:
                self._assert_runtime_start_context_locked(
                    starter,
                    workflow,
                    selected_config,
                    check_cancellation=stage != 'server',
                )
                revision = self._bump_revision()
                self.logs.append('INFO', f'{stage} {action} requested')
                return {
                    'ok': True,
                    'stage': stage,
                    'action': action,
                    'revision': revision,
                    'workflow': workflow_state,
                }
        finally:
            with self._lock:
                if self._workflow_operation_active_locked(workflow_action):
                    self._end_workflow_operation_locked(workflow_action)

    @staticmethod
    def _device_has_motion_risk(device: Mapping[str, Any]) -> bool:
        risk = str(device.get('risk') or '').strip().casefold()
        if risk in {'camera', 'sensor', 'none', 'low'}:
            return False
        capabilities = device.get('capabilities')
        motion_capability = bool(
            isinstance(capabilities, Mapping)
            and capabilities.get('motion') is True
        )
        class_name = str(
            device.get('class') or device.get('type') or ''
        ).casefold()
        motion_class = any(
            token in class_name
            for token in (
                'robot',
                'teleop',
                'motion',
                'actuator',
                'manipulator',
                'motor',
                'arm',
            )
        )
        if (
            risk not in {'motion', 'robot', 'actuator', 'high'}
            and not motion_capability
            and not motion_class
        ):
            return False
        return True

    @classmethod
    def _device_is_active_for_gripper(cls, device: Mapping[str, Any]) -> bool:
        if not cls._device_has_motion_risk(device):
            return False
        return bool(
            device.get('running')
            or device.get('desired_state') == 'running'
            or device.get('status')
            in {
                'starting',
                'running',
                'degraded',
                'stopping',
                'stop_timeout',
                'external',
                'orphaned',
            }
        )

    def _gripper_blockers_locked(
        self,
        devices: Optional[Sequence[Mapping[str, Any]]] = None,
        workflow_state: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        blockers: list[str] = []
        if self.selected_config is None or self._effective_config is None:
            blockers.append('Select a config before controlling grippers')
        if devices is None:
            devices = (
                self.starter.get_status().get('devices', [])
                if self.starter is not None
                else []
            )
        active_devices = sorted(
            str(device.get('key'))
            for device in devices
            if self._device_is_active_for_gripper(device)
        )
        if active_devices:
            blockers.append(
                'Stop motion devices before controlling grippers: '
                + ', '.join(active_devices)
            )
        if workflow_state is None and self.workflow is not None:
            observe = getattr(self.workflow, 'observe', None)
            workflow_state = (
                observe(devices)
                if callable(observe)
                else self.workflow.state(devices)
            )
        if isinstance(workflow_state, Mapping):
            runtime = workflow_state.get('runtime')
            inference = workflow_state.get('inference')
            recording = workflow_state.get('recording')
            if isinstance(runtime, Mapping) and runtime.get('active'):
                blockers.append('Stop runtime before controlling grippers')
            if isinstance(inference, Mapping) and (
                inference.get('running') or inference.get('status') == 'starting'
            ):
                blockers.append('Stop inference before controlling grippers')
            if isinstance(recording, Mapping) and recording.get('status') not in {
                None,
                'disabled',
                'idle',
                'stopped',
                'unavailable',
            }:
                blockers.append('Stop recording before controlling grippers')
        shm_names = rizon_shared_memory_names()
        if shm_names:
            blockers.append('Rizon shared memory exists; stop its owner first')
        return blockers

    def _gripper_serial_blockers_locked(self) -> Dict[str, list[str]]:
        blockers: Dict[str, list[str]] = {}
        owned_pids_method = getattr(self.grippers, 'owned_pids', None)
        if callable(owned_pids_method):
            owned_pids = list(owned_pids_method())
        else:
            owned_pid = self.grippers.owned_pid()
            owned_pids = [owned_pid] if owned_pid else []
        configured_ports = self.grippers.configured_ports()
        holders_by_port = serial_port_holders_many(
            list(configured_ports.values()),
            exclude_pids=owned_pids,
        )
        for side, port in configured_ports.items():
            holders = holders_by_port.get(port, [])
            if holders:
                holder_labels = []
                for pid in holders:
                    command = process_command(pid)
                    holder_labels.append(
                        f'PID {pid}' + (f' ({command})' if command else '')
                    )
                blockers.setdefault(side, []).append(
                    f'{side} gripper serial port {port} is held by another '
                    'process: ' + ', '.join(holder_labels)
                )
        return blockers

    def _yam_can_interfaces_locked(self) -> list[str]:
        config = self._effective_config
        if not isinstance(config, Mapping):
            return []
        devices = config.get('devices')
        if not isinstance(devices, list):
            return []
        interfaces: set[str] = set()
        for device in devices:
            if not isinstance(device, Mapping):
                continue
            if device.get('class') != 'YamHeadRobot':
                continue
            if (
                device.get('enabled', True) is False
                or device.get('yam_sim') is True
            ):
                continue
            interfaces.add(str(device.get('yam_channel') or 'can0'))
        return sorted(interfaces)

    def _hardware_permission_state_locked(
        self,
        permission_targets: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if permission_targets is None:
            configured_ports_method = getattr(
                self.grippers, 'configured_ports', None
            )
            configured_ports = (
                configured_ports_method()
                if callable(configured_ports_method)
                else {}
            )
            permission_targets = sorted({
                str(port)
                for port in (
                    configured_ports.values()
                    if isinstance(configured_ports, Mapping)
                    else []
                )
                if port
            })
        target_ports = sorted(set(map(str, permission_targets)))
        target_can_interfaces = self._yam_can_interfaces_locked()
        inspection_errors: list[str] = []

        try:
            tty_permission = self.gripper_permissions.status(target_ports)
        except GripperPermissionError as error:
            inspection_errors.append(error.code)
            tty_permission = {
                'required': False,
                'target_ports': [],
                'affected_ports': [],
                'context_id': '',
            }
        try:
            can_permission = self.can_permissions.status(
                target_can_interfaces
            )
        except GripperPermissionError as error:
            inspection_errors.append(error.code)
            can_permission = {
                'required': False,
                'target_interfaces': target_can_interfaces,
                'affected_interfaces': [],
                'unavailable_interfaces': [],
                'expected_bitrate': getattr(
                    self.can_permissions, 'bitrate', 1_000_000
                ),
                'context_id': '',
                'interfaces': {},
            }

        authorizable_ports = sorted(set(map(
            str, tty_permission.get('target_ports') or []
        )))
        affected_ports = sorted(set(map(
            str, tty_permission.get('affected_ports') or []
        )))
        affected_can_interfaces = sorted(set(map(
            str, can_permission.get('affected_interfaces') or []
        )))
        context_payload = {
            'tty': str(tty_permission.get('context_id') or ''),
            'can': str(can_permission.get('context_id') or ''),
            'target_ports': authorizable_ports,
            'target_can_interfaces': target_can_interfaces,
        }
        context_id = hashlib.sha256(
            json.dumps(
                context_payload,
                sort_keys=True,
                separators=(',', ':'),
            ).encode('utf-8')
        ).hexdigest()
        return {
            'required': bool(affected_ports or affected_can_interfaces),
            'tty_required': bool(affected_ports),
            'can_required': bool(affected_can_interfaces),
            'target_ports': authorizable_ports,
            'affected_ports': affected_ports,
            'target_can_interfaces': list(
                can_permission.get('target_interfaces')
                or target_can_interfaces
            ),
            'affected_can_interfaces': affected_can_interfaces,
            'unavailable_can_interfaces': list(
                can_permission.get('unavailable_interfaces') or []
            ),
            'expected_can_bitrate': int(
                can_permission.get('expected_bitrate') or 1_000_000
            ),
            'can_interfaces': copy.deepcopy(
                can_permission.get('interfaces') or {}
            ),
            'context_id': context_id,
            'inspection_error': ';'.join(inspection_errors),
        }

    def _gripper_state_locked(
        self,
        devices: Optional[Sequence[Mapping[str, Any]]] = None,
        workflow_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        global_blockers = self._gripper_blockers_locked(devices, workflow_state)
        serial_blockers = self._gripper_serial_blockers_locked()
        state = self.grippers.state(global_blockers)
        sides = state.get('sides', {})
        configured_ports_method = getattr(self.grippers, 'configured_ports', None)
        configured_ports = (
            configured_ports_method()
            if callable(configured_ports_method)
            else {}
        )
        permission_targets = sorted({
            str(port)
            for port in (
                configured_ports.values()
                if isinstance(configured_ports, Mapping)
                else []
            )
            if port
        })
        permission = self._hardware_permission_state_locked(permission_targets)
        state['permission'] = permission
        permission_ports = set(permission.get('affected_ports') or [])
        for item in sides.values():
            if not isinstance(item, dict):
                continue
            item['permission_required'] = item.get('port') in permission_ports
            if not item['permission_required']:
                continue
            permission_blocker = (
                '夹爪串口缺少读写权限，需要先完成安全授权'
            )
            for key in (
                'blockers',
                'calibration_blockers',
                'position_blockers',
                'queue_position_blockers',
                'probe_blockers',
            ):
                current = list(item.get(key) or item.get('blockers') or [])
                item[key] = list(dict.fromkeys(current + [permission_blocker]))
            item['can_calibrate'] = False
            item['can_position'] = False
            item['can_queue_position'] = False
            item['can_probe'] = False
            capabilities = item.get('capabilities')
            if isinstance(capabilities, dict):
                for action, blocker_key in (
                    ('calibrate', 'calibration_blockers'),
                    ('position', 'position_blockers'),
                    ('queue_position', 'queue_position_blockers'),
                    ('probe', 'probe_blockers'),
                ):
                    capability = capabilities.get(action)
                    if isinstance(capability, dict):
                        capability['can'] = False
                        capability['blockers'] = list(item[blocker_key])
        for side, extra_blockers in serial_blockers.items():
            item = sides.get(side)
            if not isinstance(item, dict):
                continue
            for key in (
                'blockers',
                'calibration_blockers',
                'position_blockers',
                'queue_position_blockers',
                'probe_blockers',
            ):
                current = item.get(key)
                if not isinstance(current, list):
                    current = list(item.get('blockers') or [])
                item[key] = list(dict.fromkeys(current + extra_blockers))
            item['can_calibrate'] = False
            item['can_position'] = False
            item['can_queue_position'] = False
            item['can_probe'] = False
            capabilities = item.get('capabilities')
            if isinstance(capabilities, dict):
                for action, blocker_key in (
                    ('calibrate', 'calibration_blockers'),
                    ('position', 'position_blockers'),
                    ('queue_position', 'queue_position_blockers'),
                    ('probe', 'probe_blockers'),
                ):
                    capability = capabilities.get(action)
                    if isinstance(capability, dict):
                        capability['can'] = False
                        capability['blockers'] = list(item[blocker_key])

        state['can_calibrate'] = bool(
            state.get('available')
            and not global_blockers
            and sides
            and all(item.get('can_calibrate') for item in sides.values())
        )
        state['can_position'] = bool(
            state.get('available')
            and not global_blockers
            and any(item.get('can_position') for item in sides.values())
        )
        state['can_queue_position'] = bool(
            state.get('available')
            and not global_blockers
            and any(item.get('can_queue_position') for item in sides.values())
        )
        state['can_probe'] = bool(
            state.get('available')
            and not global_blockers
            and any(item.get('can_probe') for item in sides.values())
        )
        state['calibration_blockers'] = list(dict.fromkeys(
            blocker
            for item in sides.values()
            for blocker in (
                item.get('calibration_blockers')
                or item.get('blockers')
                or []
            )
        ))
        binding_blockers = list(dict.fromkeys(
            global_blockers
            + [
                blocker
                for blockers in serial_blockers.values()
                for blocker in blockers
            ]
        ))
        state['can_bind_ports'] = bool(
            self.selected_config is not None
            and not binding_blockers
            and not state.get('active')
            and not state.get('busy')
        )
        state['binding_blockers'] = list(binding_blockers)
        return state

    def gripper_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'revision': self.revision,
                'grippers': self._gripper_state_locked(),
            }

    def gripper_permission_ports(self) -> list[str]:
        """Snapshot only the currently configured left/right gripper ports."""
        with self._lock:
            return self._gripper_permission_ports_locked()

    def _gripper_permission_ports_locked(self) -> list[str]:
        configured_ports_method = getattr(
            self.grippers, 'configured_ports', None
        )
        configured_ports = (
            configured_ports_method()
            if callable(configured_ports_method)
            else {}
        )
        if not isinstance(configured_ports, Mapping):
            return []
        return sorted({
            str(port) for port in configured_ports.values() if port
        })

    def _validate_gripper_permission_context_locked(
        self,
        *,
        expected_revision: int,
        expected_ports: Sequence[str],
    ) -> list[str]:
        current_ports = self._gripper_permission_ports_locked()
        submitted_ports = sorted(set(map(str, expected_ports)))
        if (
            getattr(self, '_shutdown', False)
            or expected_revision != self.revision
            or submitted_ports != current_ports
        ):
            raise GripperPermissionError(
                'permission_context_changed',
                '夹爪配置或串口绑定已变化，请关闭并重新打开授权窗口。',
                status=409,
            )
        return current_ports

    def _validate_hardware_permission_context_locked(
        self,
        *,
        expected_revision: int,
        expected_ports: Sequence[str],
        expected_can_interfaces: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        permission = self._hardware_permission_state_locked()
        current_ports = sorted(set(map(
            str, permission.get('target_ports') or []
        )))
        current_can_interfaces = sorted(set(map(
            str, permission.get('target_can_interfaces') or []
        )))
        submitted_ports = sorted(set(map(str, expected_ports)))
        submitted_can_interfaces = sorted(set(map(
            str, expected_can_interfaces
        )))
        if (
            getattr(self, '_shutdown', False)
            or expected_revision != self.revision
            or submitted_ports != current_ports
            or submitted_can_interfaces != current_can_interfaces
        ):
            raise GripperPermissionError(
                'permission_context_changed',
                'config、夹爪绑定或 Yam CAN 目标已变化，请关闭并重新打开授权窗口。',
                status=409,
            )
        return current_ports, current_can_interfaces

    @staticmethod
    def _zero_permission_secret(secret: Optional[bytearray]) -> None:
        if secret is None:
            return
        for index in range(len(secret)):
            secret[index] = 0

    def _repair_permissions(
        self,
        *,
        expected_revision: int,
        expected_ports: Sequence[str],
        expected_can_interfaces: Optional[Sequence[str]],
        client_key: str,
        secret: bytearray,
    ) -> Dict[str, Any]:
        if not self._gripper_permission_context_lock.acquire(blocking=False):
            self._zero_permission_secret(secret)
            raise GripperPermissionError(
                'permission_repair_busy',
                '已有设备权限授权或配置变更正在进行，请稍后重试。',
                status=429,
                retry_after=1,
            )
        tty_secret: Optional[bytearray] = None
        can_secret: Optional[bytearray] = None
        lease_token: Optional[str] = None
        ports: list[str] = []
        can_interfaces: list[str] = []
        try:
            with self._lock:
                if expected_can_interfaces is None:
                    ports = self._validate_gripper_permission_context_locked(
                        expected_revision=expected_revision,
                        expected_ports=expected_ports,
                    )
                else:
                    ports, can_interfaces = (
                        self._validate_hardware_permission_context_locked(
                            expected_revision=expected_revision,
                            expected_ports=expected_ports,
                            expected_can_interfaces=expected_can_interfaces,
                        )
                    )

            can_result: Optional[Dict[str, Any]] = None
            if can_interfaces:
                can_secret = bytearray(secret)
                can_result = self.can_permissions.authorize_with_password(
                    can_secret, client_key, can_interfaces
                )

            tty_result: Optional[Dict[str, Any]] = None
            if ports:
                tty_secret = bytearray(secret)
                tty_result = self.gripper_permissions.authorize_with_password(
                    tty_secret, client_key, ports
                )
                lease_token = tty_result.pop('_lease_token', None)

            try:
                with self._lock:
                    if expected_can_interfaces is None:
                        self._validate_gripper_permission_context_locked(
                            expected_revision=expected_revision,
                            expected_ports=ports,
                        )
                    else:
                        self._validate_hardware_permission_context_locked(
                            expected_revision=expected_revision,
                            expected_ports=ports,
                            expected_can_interfaces=can_interfaces,
                        )
                    if lease_token:
                        commit = getattr(
                            self.gripper_permissions,
                            'commit_authorization',
                            None,
                        )
                        if not callable(commit):
                            raise GripperPermissionError(
                                'permission_repair_internal_error',
                                'Verified ttyUSB authorization cannot be committed.',
                                status=500,
                            )
                        assert tty_result is not None
                        tty_result['permission'] = commit(
                            str(lease_token),
                            ports,
                            controller_revision=self.revision,
                        )

                    if expected_can_interfaces is None:
                        return tty_result or {
                            'ok': True,
                            'authorized': True,
                            'method': 'already_accessible',
                            'ports': ports,
                        }
                    return {
                        'ok': True,
                        'authorized': True,
                        'method': 'sudo_hardware_permissions',
                        'ports': ports,
                        'can_interfaces': can_interfaces,
                        'ttyusb': tty_result,
                        'can': can_result,
                        'permission': self._hardware_permission_state_locked(
                            ports
                        ),
                    }
            except Exception:
                discard = getattr(
                    self.gripper_permissions,
                    'discard_authorization',
                    None,
                )
                if callable(discard):
                    discard(str(lease_token) if lease_token else None)
                if lease_token:
                    revoke = getattr(
                        self.gripper_permissions, 'revoke_leases', None
                    )
                    if callable(revoke):
                        revoke(ports)
                raise
        finally:
            self._zero_permission_secret(secret)
            self._zero_permission_secret(tty_secret)
            self._zero_permission_secret(can_secret)
            self._gripper_permission_context_lock.release()

    def repair_gripper_permissions(
        self,
        *,
        expected_revision: int,
        expected_ports: Sequence[str],
        client_key: str,
        secret: bytearray,
    ) -> Dict[str, Any]:
        """Freeze binding/config context across the complete chmod operation."""
        return self._repair_permissions(
            expected_revision=expected_revision,
            expected_ports=expected_ports,
            expected_can_interfaces=None,
            client_key=client_key,
            secret=secret,
        )

    def repair_hardware_permissions(
        self,
        *,
        expected_revision: int,
        expected_ports: Sequence[str],
        expected_can_interfaces: Sequence[str],
        client_key: str,
        secret: bytearray,
    ) -> Dict[str, Any]:
        """Repair the frozen ttyUSB and Yam CAN targets from one prompt."""
        return self._repair_permissions(
            expected_revision=expected_revision,
            expected_ports=expected_ports,
            expected_can_interfaces=expected_can_interfaces,
            client_key=client_key,
            secret=secret,
        )

    def update_gripper_ports(
        self,
        bindings: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> Dict[str, Any]:
        operation_key = 'grippers:binding'
        with self._gripper_permission_mutation_guard():
            with self._lock:
                if expected_revision != self.revision:
                    raise GripperConflictError(
                        f'Stale revision {expected_revision}; current revision is {self.revision}'
                    )
                blockers = self._gripper_blockers_locked()
                serial_blockers = self._gripper_serial_blockers_locked()
                blockers = list(dict.fromkeys(
                    blockers
                    + [
                        blocker
                        for values in serial_blockers.values()
                        for blocker in values
                    ]
                ))
                if blockers or self.grippers.active():
                    raise GripperConflictError(
                        '; '.join(blockers or ['Gripper control is busy'])
                    )
                manager_state = self.grippers.state()
                current_ports = self.grippers.configured_ports()
                if not current_ports:
                    current_ports = {
                        side: str(item.get('port'))
                        for side, item in (
                            manager_state.get('sides', {}).items()
                            if isinstance(manager_state, Mapping)
                            else []
                        )
                        if isinstance(item, Mapping) and item.get('port')
                    }
                available_ports_method = getattr(
                    self.grippers, 'available_ports', None
                )
                available_ports = set(
                    available_ports_method()
                    if callable(available_ports_method)
                    else manager_state.get('available_ports', [])
                )
                candidate_ports = dict(current_ports)
                for raw_side, raw_port in bindings.items():
                    side = str(raw_side)
                    port = str(raw_port)
                    if side not in candidate_ports:
                        raise GripperRequestError(
                            f'{side} LKTech gripper is not configured'
                        )
                    if not re.fullmatch(r'/dev/ttyUSB[0-9]+', port):
                        raise GripperRequestError(
                            'gripper port must match /dev/ttyUSB<number>'
                        )
                    if port not in available_ports:
                        raise GripperRequestError(
                            f'gripper port is no longer available: {port}'
                        )
                    candidate_ports[side] = port
                if len(set(candidate_ports.values())) != len(candidate_ports):
                    raise GripperRequestError(
                        'left and right grippers cannot use the same serial port'
                    )
                selected_config = self.selected_config
                starter = self._require_starter_locked()
                self._begin_device_operation_locked(operation_key, 'binding')
                self.grippers.stop('gripper_binding_change')

            try:
                with self._lock:
                    if (
                        expected_revision != self.revision
                        or self.selected_config != selected_config
                        or self.starter is not starter
                        or self._effective_config is None
                    ):
                        raise GripperConflictError(
                            'Config changed during gripper binding update'
                        )
                    bind_ports = getattr(self.grippers, 'bind_ports', None)
                    if not callable(bind_ports):
                        raise GripperControlError(
                            'Gripper serial binding is unavailable'
                        )
                    selected = bind_ports(bindings)
                    self._gripper_port_overrides = dict(selected)
                    # Keep the session's effective config aligned with the
                    # newly observed ttyUSB identities. Start preflight,
                    # diagnostics, and config export all consume this direct
                    # desired config rather than a stale source preset.
                    self._effective_config = self._apply_gripper_port_overrides(
                        self._effective_config,
                        self._gripper_port_overrides,
                    )
                    runtime_base = self._apply_gripper_port_overrides(
                        self._direct_gripper_config(self._effective_config),
                        self._gripper_port_overrides,
                    )
                    self._apply_gripper_transport_config_locked(runtime_base)
                    self._bump_revision()
                with self._lock:
                    self.logs.append(
                        'INFO',
                        'Updated gripper ttyUSB bindings: '
                        + ', '.join(
                            f'{side}={port}'
                            for side, port in sorted(selected.items())
                        ),
                    )
                    return {
                        'ok': True,
                        'revision': self.revision,
                        'bindings': selected,
                        'grippers': self._gripper_state_locked(),
                    }
            finally:
                with self._lock:
                    self._end_device_operation_locked(operation_key)

    def control_gripper(
        self,
        action: str,
        *,
        expected_revision: int,
        expected_side_revision: Optional[int] = None,
        side: Optional[str] = None,
        target_percent: Optional[float] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if getattr(self, '_shutdown', False) and action != 'stop':
                raise GripperConflictError(
                    'Device control server is shutting down'
                )
            replay = getattr(self.grippers, 'replay', None)
            duplicate = (
                replay(
                    action,
                    str(side),
                    str(idempotency_key),
                    target_percent,
                )
                if callable(replay) and action != 'stop'
                else None
            )
            if duplicate is not None:
                return {
                    'ok': True,
                    'revision': self.revision,
                    'command': duplicate,
                    'grippers': self._gripper_state_locked(),
                }
            if action == 'stop':
                # A safety stop is idempotent and must remain available even if
                # the UI snapshot is stale. It cannot start hardware or broaden
                # authority; it only tears down manager-owned gripper workers.
                command = self.grippers.stop('api_stop')
                self._bump_revision()
            else:
                if expected_revision != self.revision:
                    raise GripperConflictError(
                        f'Stale revision {expected_revision}; '
                        f'current revision is {self.revision}'
                    )
                state = self._gripper_state_locked()
                affected_ports = set(
                    state.get('permission', {}).get('affected_ports') or []
                )
                if action == 'calibrate' and str(side) not in {'left', 'right'}:
                    relevant_ports = [
                        str(item.get('port'))
                        for item in state.get('sides', {}).values()
                        if isinstance(item, Mapping) and item.get('port')
                    ]
                else:
                    side_item = state.get('sides', {}).get(str(side), {})
                    relevant_ports = [str(side_item.get('port'))] if side_item.get('port') else []
                required_ports = sorted(
                    port for port in relevant_ports if port in affected_ports
                )
                if required_ports:
                    raise GripperPermissionRequiredError(required_ports)
                if action == 'position':
                    side_state = state.get('sides', {}).get(str(side), {})
                    # Slider control is a latest-target stream.  A side-local
                    # worker may still be finishing the previous target while
                    # the manager safely coalesces the next one.  Calibration,
                    # permission, topology and unknown-hardware interlocks are
                    # still represented by can_queue_position and fail closed.
                    capability = side_state.get(
                        'can_queue_position', side_state.get('can_position')
                    )
                    capability_blockers = side_state.get(
                        'queue_position_blockers', side_state.get('blockers', [])
                    )
                elif action == 'probe':
                    side_state = state.get('sides', {}).get(str(side), {})
                    capability = side_state.get('can_probe')
                    capability_blockers = side_state.get(
                        'probe_blockers', side_state.get('blockers', [])
                    )
                elif action == 'calibrate' and str(side) in {'left', 'right'}:
                    side_state = state.get('sides', {}).get(str(side), {})
                    capability = side_state.get('can_calibrate')
                    capability_blockers = side_state.get(
                        'calibration_blockers', side_state.get('blockers', [])
                    )
                elif action == 'calibrate':
                    capability = state.get('can_calibrate')
                    capability_blockers = state.get(
                        'calibration_blockers', state.get('blockers', [])
                    )
                else:
                    raise GripperRequestError(f'Unknown gripper action: {action}')
                if not capability:
                    raise GripperConflictError(
                        '; '.join(capability_blockers or ['Gripper control is busy'])
                    )
                if action == 'calibrate':
                    command = self.grippers.calibrate(
                        str(side), str(idempotency_key)
                    )
                elif action == 'position':
                    command = self.grippers.position(
                        str(side),
                        target_percent,
                        str(idempotency_key),
                        expected_side_revision=expected_side_revision,
                    )
                elif action == 'probe':
                    command = self.grippers.probe(
                        str(side),
                        str(idempotency_key),
                        expected_side_revision=expected_side_revision,
                    )
                # Position commands are side-scoped and may be submitted from
                # the same UI revision concurrently. They do not change config
                # or safety topology, so keep the global controller revision
                # stable; bindings/calibration/stop still invalidate stale UIs.
                if command.get('accepted') and action not in {'position', 'probe'}:
                    self._bump_revision()
            self.logs.append('INFO', f'gripper {action} requested')
            return {
                'ok': True,
                'revision': self.revision,
                'command': command,
                'grippers': self._gripper_state_locked(),
            }

    def control_recording(
        self,
        action: str,
        *,
        episode_id: str,
        expected_checkpoint_revision: int,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._shutdown:
                raise RecordingConflictError(
                    'Device control server is shutting down'
                )
            starter = self.starter
            workflow = self.workflow
            if workflow is None:
                raise RecordingConflictError(
                    'Select a config before controlling recording'
                )
            selected_config = self.selected_config
            self._begin_workflow_operation_locked(f'recording:{action}')

        try:
            # Recording transitions must never overlap a manual serial session.
            # stop() is synchronous and fails closed if the worker cannot clean up,
            # but it must not monopolize the controller state lock while waiting.
            self.grippers.stop(f'recording_{action}')
            result = workflow.recording_action(
                action,
                episode_id=episode_id,
                expected_checkpoint_revision=expected_checkpoint_revision,
                idempotency_key=idempotency_key,
            )
            with self._lock:
                self._assert_runtime_start_context_locked(
                    starter,
                    workflow,
                    selected_config,
                )
                result['revision'] = self._bump_revision()
                self.logs.append('INFO', f'recording {action} requested')
                return result
        finally:
            with self._lock:
                self._end_workflow_operation_locked()

    def control_inference_reset(
        self,
        *,
        idempotency_key: str,
        expected_selected_config: str,
        expected_revision: int,
    ) -> Dict[str, Any]:
        """Queue a Home/reset without stopping runtime or loaded inference."""
        with self._lock:
            if self._shutdown:
                raise WorkflowConflictError(
                    'Device control server is shutting down'
                )
            starter = self.starter
            workflow = self.workflow
            if workflow is None:
                raise WorkflowConflictError(
                    'Select a config before resetting inference'
                )
            selected_config = self.selected_config
            if (
                selected_config != expected_selected_config
                or self.revision != expected_revision
            ):
                raise WorkflowConflictError(
                    'Config or control state changed; refresh before resetting'
                )
            self._begin_workflow_operation_locked('inference:reset')

        try:
            # A manual gripper serial session must not overlap robot Home motion.
            self.grippers.stop('inference_reset')
            result = workflow.inference_reset(
                idempotency_key=idempotency_key,
            )
            with self._lock:
                self._assert_runtime_start_context_locked(
                    starter,
                    workflow,
                    selected_config,
                )
                result['revision'] = self._bump_revision()
                self.logs.append('INFO', 'standalone inference reset requested')
                return result
        finally:
            with self._lock:
                self._end_workflow_operation_locked()

    def control_inference_pause_resume(
        self,
        action: str,
        *,
        idempotency_key: str,
        expected_selected_config: str,
        expected_revision: int,
    ) -> Dict[str, Any]:
        """Pause or resume an already-loaded inference process."""
        if action not in {'pause', 'resume'}:
            raise WorkflowConflictError('Unsupported inference control action')
        with self._lock:
            if self._shutdown:
                raise WorkflowConflictError(
                    'Device control server is shutting down'
                )
            starter = self.starter
            workflow = self.workflow
            if workflow is None:
                raise WorkflowConflictError(
                    'Select a config before controlling inference'
                )
            selected_config = self.selected_config
            if (
                selected_config != expected_selected_config
                or self.revision != expected_revision
            ):
                raise WorkflowConflictError(
                    'Config or control state changed; refresh before controlling inference'
                )
            self._begin_workflow_operation_locked(f'inference:{action}')

        try:
            result = workflow.inference_control(
                action,
                idempotency_key=idempotency_key,
            )
            with self._lock:
                self._assert_runtime_start_context_locked(
                    starter,
                    workflow,
                    selected_config,
                )
                result['revision'] = self._bump_revision()
                self.logs.append('INFO', f'inference {action} requested')
                return result
        finally:
            with self._lock:
                self._end_workflow_operation_locked()

    def device_status(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        with self._lock:
            return self._require_starter_locked().get_device_status(
                device_class, device_class_id
            )

    @staticmethod
    def _preview_metadata(
        result: CameraPreviewResult, device: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add stable UI fields even while camera SHM is not available."""

        metadata = dict(result.metadata)
        timestamp_ns = metadata.get('timestamp_ns')
        metadata.setdefault('timestamp', timestamp_ns)
        metadata.setdefault('device_class', device['class'])
        metadata.setdefault('device_class_id', device['id'])
        if metadata.get('fps') is None:
            metadata['fps'] = device.get('fps')
        if metadata.get('shape') is None:
            metadata['shape'] = device.get('shape')
        if metadata.get('dtype') is None:
            metadata['dtype'] = device.get('dtype')
        metadata['available'] = result.jpeg_bytes is not None
        metadata['ok'] = metadata.get('status') == 'ok'
        return metadata

    def _preview_slot_locked(
        self, device_class: str, device_class_id: Any
    ) -> Tuple[_PreviewReaderSlot, Dict[str, Any]]:
        # get_device_status is the authorization boundary: only a device in the
        # currently selected config may influence a shared-memory lookup.
        device = self.device_status(device_class, device_class_id)
        capabilities = device.get('capabilities') or {}
        if not (
            capabilities.get('preview') is True
            or capabilities.get('camera') is True
        ):
            raise DeviceControlError(
                f"Device does not support camera preview: {device_class}:{device_class_id}"
            )

        canonical_class = str(device['class'])
        canonical_id = str(device['id'])
        key = (canonical_class, canonical_id)
        slot = self._preview_readers.get(key)
        if slot is None:
            try:
                numeric_id = int(device['id'])
            except (TypeError, ValueError) as exc:
                raise DeviceControlError(
                    f"Camera preview requires an integer device ID: {device['id']}"
                ) from exc
            slot = _PreviewReaderSlot(
                reader=self._preview_reader_factory(canonical_class, numeric_id),
                lock=threading.Lock(),
            )
            self._preview_readers[key] = slot
        return slot, device

    def read_preview(
        self, device_class: str, device_class_id: Any
    ) -> Tuple[CameraPreviewResult, Dict[str, Any]]:
        """Read one latest frame after validating the selected config device."""

        # Hold the lifecycle lock through the bounded SHM copy. This prevents a
        # successful config switch from closing a reader between validation and
        # use, while the per-device lock prevents concurrent reads on one handle.
        with self._lock:
            slot, device = self._preview_slot_locked(device_class, device_class_id)
            with slot.lock:
                result = slot.reader.read()
            return result, self._preview_metadata(result, device)

    def _close_preview_readers_locked(self) -> None:
        slots = list(self._preview_readers.values())
        self._preview_readers.clear()
        for slot in slots:
            with slot.lock:
                try:
                    slot.reader.close()
                except (BufferError, OSError, RuntimeError, ValueError):
                    # Closing a consumer handle is best effort. Never attempt to
                    # unlink producer-owned shared memory from this control plane.
                    self.logs.append('WARNING', 'Preview reader close failed')

    def _robot_state_slot_locked(
        self, device: Dict[str, Any]
    ) -> _PreviewReaderSlot:
        """Return a cached reader for an authorized Rizon config device."""

        canonical = self.device_status('RizonRobot', device['id'])
        if str(canonical.get('class')) != 'RizonRobot':
            raise DeviceControlError(
                f"Device is not a RizonRobot: {canonical.get('class')}:{canonical.get('id')}"
            )
        key = str(canonical['id'])
        slot = self._robot_state_readers.get(key)
        if slot is None:
            try:
                numeric_id = int(canonical['id'])
            except (TypeError, ValueError) as exc:
                raise DeviceControlError(
                    f"Robot state preview requires an integer device ID: {canonical['id']}"
                ) from exc
            slot = _PreviewReaderSlot(
                reader=self._robot_state_reader_factory(numeric_id),
                lock=threading.Lock(),
            )
            self._robot_state_readers[key] = slot
        return slot

    def _yam_state_slot_locked(
        self, device: Dict[str, Any]
    ) -> _PreviewReaderSlot:
        """Return a cached read-only reader for the configured YAM device."""

        canonical = self.device_status('YamHeadRobot', device['id'])
        if str(canonical.get('class')) != 'YamHeadRobot':
            raise DeviceControlError(
                f"Device is not a YamHeadRobot: {canonical.get('class')}:{canonical.get('id')}"
            )
        key = str(canonical['id'])
        slot = self._yam_state_readers.get(key)
        if slot is None:
            try:
                numeric_id = int(canonical['id'])
            except (TypeError, ValueError) as exc:
                raise DeviceControlError(
                    f"YAM state preview requires an integer device ID: {canonical['id']}"
                ) from exc
            slot = _PreviewReaderSlot(
                reader=self._yam_state_reader_factory(numeric_id),
                lock=threading.Lock(),
            )
            self._yam_state_readers[key] = slot
        return slot

    def _read_yam_state_locked(
        self,
        devices: Sequence[Mapping[str, Any]],
        visualization: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            **copy.deepcopy(dict(visualization)),
            'joints': None,
            'tcp_pose': None,
            'fresh': False,
            'source': (
                'initpos'
                if visualization.get('initial_joints') is not None
                else 'unavailable'
            ),
            'timestamp_ns': None,
            'age_ms': None,
            'schema': None,
            'shm_name': None,
        }
        if visualization.get('configured') is not True:
            return payload
        if visualization.get('status') != 'available':
            return payload

        device_key = str(visualization.get('device_key') or '')
        device = next(
            (
                item
                for item in devices
                if isinstance(item, Mapping)
                and str(item.get('class')) == 'YamHeadRobot'
                and str(item.get('key') or f"YamHeadRobot:{item.get('id')}")
                == device_key
            ),
            None,
        )
        if device is None:
            payload['status'] = 'not-found'
            return payload

        slot = self._yam_state_slot_locked(dict(device))
        with slot.lock:
            result = slot.reader.read()
        metadata = result.metadata
        joints = (
            result.joint_positions.get('yam') if result.status == 'ok' else None
        )
        tcp_pose = result.tcp_poses.get('yam') if result.status == 'ok' else None
        payload.update(
            {
                'status': result.status,
                'joints': list(joints) if joints is not None else None,
                'tcp_pose': list(tcp_pose) if tcp_pose is not None else None,
                'fresh': joints is not None,
                'source': 'live' if joints is not None else payload['source'],
                'timestamp_ns': metadata.get('timestamp_ns'),
                'age_ms': metadata.get('age_ms'),
                'schema': metadata.get('schema'),
                'shm_name': metadata.get('shm_name'),
            }
        )
        return payload

    @staticmethod
    def _robot_side_hint(device: Dict[str, Any]) -> Optional[str]:
        config = device.get('config') or {}
        for key in ('arm_side', 'robot_side', 'side'):
            value = str(config.get(key, '')).lower()
            if value in {'left', 'right'}:
                return value
        return None

    def _current_robot_state_cache_key(self) -> Tuple[int, Optional[str]]:
        with self._lock:
            return self.revision, self.selected_config

    @staticmethod
    def _cached_robot_state_payload(
        snapshot: Mapping[str, Any], cache_age_s: float
    ) -> Dict[str, Any]:
        payload = copy.deepcopy(dict(snapshot))
        yam_head = payload.get('yam_head')
        if isinstance(yam_head, dict):
            age_ms = yam_head.get('age_ms')
            if isinstance(age_ms, (int, float)) and not isinstance(age_ms, bool):
                yam_head['age_ms'] = float(age_ms) + max(
                    0.0, cache_age_s
                ) * 1000.0
        return payload

    def read_robot_states(self) -> Dict[str, Any]:
        """Coalesce concurrent passive SHM reads into one near-live snapshot."""
        requested_key = self._current_robot_state_cache_key()
        with self._robot_state_cache_lock:
            now = time.monotonic()
            if (
                self._robot_state_cache is not None
                and self._robot_state_cache_key == requested_key
                and now - self._robot_state_cache_at
                < self.ROBOT_STATE_COALESCE_S
            ):
                return self._cached_robot_state_payload(
                    self._robot_state_cache,
                    now - self._robot_state_cache_at,
                )

            snapshot = self._read_robot_states_uncached()
            completed_at = time.monotonic()
            if self._current_robot_state_cache_key() == requested_key:
                self._robot_state_cache = copy.deepcopy(snapshot)
                self._robot_state_cache_key = requested_key
                self._robot_state_cache_at = completed_at
            return snapshot

    def _read_robot_states_uncached(self) -> Dict[str, Any]:
        """Read configured Rizon and YAM state without owning hardware."""

        with self._lock:
            home = extract_robot_home(self._effective_config or {})
            configured_tools = extract_robot_tools(self._effective_config or {})
            yam_visualization = extract_yam_visualization(
                self._effective_config or {}
            )
            if self.starter is None:
                return {
                    'status': 'no_config',
                    'timestamp': None,
                    'timestamp_ns': None,
                    'left': None,
                    'right': None,
                    'tcp_poses': {'left': None, 'right': None},
                    'gripper_positions': {'left': None, 'right': None},
                    'tool_names': {'left': None, 'right': None},
                    'configured_tools': configured_tools,
                    'home': home,
                    'yam_head': self._read_yam_state_locked(
                        [], yam_visualization
                    ),
                    'devices': [],
                }

            all_devices = self.starter.get_status().get('devices', [])
            yam_head = self._read_yam_state_locked(
                all_devices, yam_visualization
            )
            devices = [
                device
                for device in all_devices
                if str(device.get('class')) == 'RizonRobot'
            ]
            if not devices:
                return {
                    'status': 'not_configured',
                    'timestamp': None,
                    'timestamp_ns': None,
                    'left': None,
                    'right': None,
                    'tcp_poses': {'left': None, 'right': None},
                    'gripper_positions': {'left': None, 'right': None},
                    'tool_names': {'left': None, 'right': None},
                    'configured_tools': configured_tools,
                    'home': home,
                    'yam_head': yam_head,
                    'devices': [],
                }

            joints: Dict[str, Optional[list]] = {'left': None, 'right': None}
            tcp_poses: Dict[str, Optional[list]] = {
                'left': None,
                'right': None,
            }
            gripper_positions: Dict[str, Optional[float]] = {
                'left': None,
                'right': None,
            }
            single_states = []
            timestamps = []
            summaries = []

            for device in devices:
                slot = self._robot_state_slot_locked(device)
                with slot.lock:
                    result = slot.reader.read()
                # A stale reader may still expose its last decoded frame. Do
                # not present that frame as live motion to the web viewer.
                positions = (
                    result.joint_positions if result.status == 'ok' else {}
                )
                poses = result.tcp_poses if result.status == 'ok' else {}
                grippers = (
                    result.gripper_positions if result.status == 'ok' else {}
                )
                metadata = result.metadata
                timestamp_ns = metadata.get('timestamp_ns')
                if result.status == 'ok' and isinstance(
                    timestamp_ns, (int, float)
                ):
                    timestamps.append(int(timestamp_ns))
                summaries.append(
                    {
                        'device_class_id': device['id'],
                        'status': result.status,
                        'timestamp_ns': timestamp_ns,
                        'schema': metadata.get('schema'),
                    }
                )
                for side in ('left', 'right'):
                    values = positions.get(side)
                    if values is not None:
                        joints[side] = list(values)
                        pose = poses.get(side)
                        if pose is not None:
                            tcp_poses[side] = list(pose)
                        gripper_position = grippers.get(side)
                        if gripper_position is not None:
                            gripper_positions[side] = float(gripper_position)
                values = positions.get('single')
                if values is not None:
                    single_states.append(
                        (
                            self._robot_side_hint(device),
                            list(values),
                            (
                                list(poses['single'])
                                if poses.get('single') is not None
                                else None
                            ),
                            (
                                float(grippers['single'])
                                if grippers.get('single') is not None
                                else None
                            ),
                        )
                    )

            for hint, values, pose, gripper_position in single_states:
                if hint and joints[hint] is None:
                    joints[hint] = values
                    tcp_poses[hint] = pose
                    gripper_positions[hint] = gripper_position
                    continue
                # Existing single-arm inference configs operate the right arm.
                # With multiple unlabelled readers, fill both sides deterministically.
                side = 'right' if joints['right'] is None else 'left'
                if joints[side] is None:
                    joints[side] = values
                    tcp_poses[side] = pose
                    gripper_positions[side] = gripper_position

            available = joints['left'] is not None or joints['right'] is not None
            statuses = {summary['status'] for summary in summaries}
            if available:
                status = 'ok'
            elif len(statuses) == 1:
                status = next(iter(statuses))
            else:
                status = 'unavailable'
            timestamp_ns = max(timestamps) if timestamps else None
            return {
                'status': status,
                'timestamp': timestamp_ns,
                'timestamp_ns': timestamp_ns,
                'left': joints['left'],
                'right': joints['right'],
                'tcp_poses': tcp_poses,
                'gripper_positions': gripper_positions,
                # Tool identity is not present in SHM. Keep it explicitly null
                # instead of connecting to RDK from this passive read endpoint.
                'tool_names': {'left': None, 'right': None},
                'configured_tools': configured_tools,
                'home': home,
                'yam_head': yam_head,
                'devices': summaries,
            }

    def _close_robot_state_readers_locked(self) -> None:
        slots = [
            *self._robot_state_readers.values(),
            *self._yam_state_readers.values(),
        ]
        self._robot_state_readers.clear()
        self._yam_state_readers.clear()
        for slot in slots:
            with slot.lock:
                try:
                    slot.reader.close()
                except (BufferError, OSError, RuntimeError, ValueError):
                    self.logs.append('WARNING', 'Robot state reader close failed')

    def control_device(
        self, action: str, device_class: str, device_class_id: Any
    ) -> Dict[str, Any]:
        if action not in {'start', 'stop', 'restart'}:
            raise DeviceControlError(f"Unknown action: {action}")
        requested_key = f'{device_class}:{device_class_id}'
        with self._lock:
            starter = self._require_starter_locked()
            status_reader = getattr(starter, 'get_device_status', None)
            device_before = (
                status_reader(device_class, device_class_id)
                if callable(status_reader)
                else {'class': device_class}
            )
            motion_device = self._device_has_motion_risk(device_before)
            workflow = self.workflow
            workflow_transition = self._workflow_operation_active_locked()
            # A physical stop is always a safety-reducing operation. During a
            # long workflow transition it must bypass the busy workflow lock
            # and reach the device immediately; starts/restarts remain blocked.
            workflow_active = bool(
                workflow
                and not workflow_transition
                and workflow.has_active_runtime_processes()
            )
            if workflow_active and action == 'start':
                raise DeviceConflictError(
                    "Stop inference and runtime before starting another device"
                )
            if action in {'start', 'restart'}:
                self._ensure_can_ready_for_device_locked(
                    device_class, device_class_id
                )
            self._begin_device_operation_locked(requested_key, action)
            if workflow_transition and action == 'stop':
                self._workflow_cancelled_device_keys.add(requested_key)

        try:
            # Spawn, hardware initialization, and workflow teardown can be
            # slow. The operation registration prevents config replacement
            # while leaving state reads and other device keys responsive.
            if action in {'start', 'restart'} and motion_device:
                self.grippers.stop('device_start')
            if workflow_active and workflow is not None:
                workflow.stop_all()
            if action == 'start':
                observed_start = getattr(
                    starter,
                    'start_device_observed',
                    starter.start_device,
                )
                device = observed_start(device_class, device_class_id)
            elif action == 'stop':
                device = starter.stop_device(device_class, device_class_id)
            elif action == 'restart':
                observed_restart = getattr(
                    starter,
                    'restart_device_observed',
                    starter.restart_device,
                )
                device = observed_restart(device_class, device_class_id)
        except BaseException:
            with self._lock:
                self._end_device_operation_locked(requested_key)
            raise

        with self._lock:
            try:
                if self.starter is not starter:
                    raise DeviceConflictError(
                        'Selected config changed during the device action'
                    )
                key = str(device['key'])
                self._bump_revision()
                cancelled = bool(
                    action in {'start', 'restart'}
                    and device.get('desired_state') == 'stopped'
                )
                self.logs.append(
                    'INFO',
                    (
                        f"{action} superseded by stop for {key}"
                        if cancelled
                        else f"{action} requested for {key}"
                    ),
                    device_key=key,
                )
                return {
                    'ok': True,
                    'action': action,
                    'cancelled': cancelled,
                    'revision': self.revision,
                    'device': device,
                }
            finally:
                self._end_device_operation_locked(requested_key)

    def force_stop_all(self) -> Dict[str, Any]:
        """Force-kill panel-owned children while keeping the web server alive."""
        with self._lock:
            self._ensure_device_operation_state_locked()
            if self._shutdown:
                raise DeviceConflictError(
                    'Device control server is shutting down'
                )
            force_lock = self._force_stop_lock
        if not force_lock.acquire(blocking=False):
            raise DeviceConflictError('Force stop is already in progress')

        try:
            with self._lock:
                self._force_stop_in_progress = True
                self._workflow_cancel_event.set()
                workflow = self.workflow
                starter = self.starter
                grippers = self.grippers

            results: Dict[str, Any] = {}
            errors: Dict[str, str] = {}

            # Interrupt a potentially long model-server load before entering
            # the definitive force-stop stage. The workflow still owns the
            # SIGKILL, wait, and private-run cleanup below.
            request_server_stop = getattr(
                workflow, 'request_async_server_stop', None
            )
            if callable(request_server_stop):
                try:
                    request_server_stop()
                    results['server_interrupt'] = {
                        'ok': True,
                        'killed_pids': [],
                    }
                except BaseException as exc:
                    errors['server_interrupt'] = (
                        f'{type(exc).__name__}: {exc}'
                    )

            def force_stage(
                name: str,
                target: Any,
                method_name: str = 'force_stop_all',
            ) -> None:
                if target is None:
                    results[name] = {'ok': True, 'killed_pids': []}
                    return
                method = getattr(target, method_name, None)
                if not callable(method):
                    errors[name] = f'{type(target).__name__} has no {method_name}()'
                    return
                try:
                    result = method()
                except BaseException as exc:
                    errors[name] = f'{type(exc).__name__}: {exc}'
                    return
                results[name] = (
                    dict(result) if isinstance(result, Mapping) else {'ok': True}
                )

            # Dependency order remains inference/runtime -> devices -> gripper
            # command workers. No stage prevents later stages from attempting
            # containment.
            force_stage('workflow', workflow)
            force_stage('devices', starter)
            force_stage('gripper_commands', grippers, 'force_stop')

            killed_pids = sorted(
                {
                    int(pid)
                    for result in results.values()
                    for pid in (result.get('killed_pids') or [])
                    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
                }
            )
            incomplete = sorted(
                name
                for name, result in results.items()
                if result.get('ok') is False
            )
            with self._lock:
                self._close_preview_readers_locked()
                self._close_robot_state_readers_locked()
                revision = self._bump_revision()
                self.logs.append(
                    'WARNING',
                    'Operator force-stopped all panel-owned child processes; '
                    'graceful robot disable, Yam zero confirmation, gripper hold, '
                    'and recording flush were bypassed',
                )
            return {
                'ok': not errors and not incomplete,
                'action': 'force_stop_all',
                'revision': revision,
                'killed_pids': killed_pids,
                'incomplete_stages': incomplete,
                'errors': errors,
                'results': results,
                'web_server_preserved': True,
                'warning': (
                    'Forced termination bypassed graceful hardware shutdown and '
                    'recording finalization'
                ),
            }
        finally:
            with self._lock:
                self._force_stop_in_progress = False
            force_lock.release()

    def shutdown(self) -> None:
        with self._lock, self._lifecycle_mutation_guard_locked():
            if getattr(self, '_shutdown_complete', False):
                return
            self._shutdown = True
            errors: list[tuple[str, BaseException]] = []

            def cleanup_stage(name: str, callback: Callable[[], None]) -> bool:
                try:
                    callback()
                except BaseException as exc:
                    errors.append((name, exc))
                    return False
                return True

            cleanup_stage('grippers', self.grippers.close)
            recording_exports = getattr(self, 'recording_exports', None)
            if (
                getattr(self, '_owns_recording_exports', False)
                and recording_exports is not None
            ):
                cleanup_stage('recording_exports', recording_exports.close)
            if self._owns_conversion_jobs:
                cleanup_stage('conversion_jobs', self.conversion_jobs.close)
            cleanup_stage('camera_preview', self._close_preview_readers_locked)
            cleanup_stage('robot_preview', self._close_robot_state_readers_locked)

            workflow = self.workflow
            if workflow is not None and cleanup_stage('workflow', workflow.stop_all):
                self.workflow = None
            starter = self.starter
            if starter is not None and cleanup_stage(
                'starter', lambda: self._stop_starter_for_shutdown(starter)
            ):
                self.starter = None
            if errors:
                try:
                    self.logs.append(
                        'ERROR',
                        'Device control shutdown errors: '
                        + '; '.join(
                            f'{name}: {type(exc).__name__}: {exc}'
                            for name, exc in errors
                        ),
                    )
                except BaseException:
                    pass
                # Keep _shutdown_complete false so a subsequent cleanup call can
                # retry failed stages. Re-raise the first original exception to
                # preserve its type for callers while all other stages ran.
                raise errors[0][1]

            self._shutdown_complete = True
            self.logs.append('INFO', 'Device control server stopped')

    @staticmethod
    def _stop_starter_for_shutdown(starter: Any) -> None:
        """Require every panel-owned device worker to stop before exit."""
        report = starter.stop_all()
        if not isinstance(report, Mapping):
            return

        incomplete = {
            str(key) for key in (report.get('incomplete_devices') or ()) if key
        }
        external = {
            str(key) for key in (report.get('external_devices') or ()) if key
        }
        owned_incomplete = sorted(incomplete - external)
        if owned_incomplete:
            raise DeviceControlError(
                'Owned device shutdown incomplete: '
                + ', '.join(owned_incomplete)
            )
        if report.get('ok') is False and not incomplete:
            raise DeviceControlError(
                'Owned device shutdown failed without a complete status report'
            )


class SingleInstanceLock:
    """Advisory process lock preventing two control servers on one host."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open('a+', encoding='utf-8')
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError) as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"Another device control server holds {self.path}"
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()))
        self._handle.flush()

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _parse_device_key(device_key: str) -> Tuple[str, str]:
    if ':' not in device_key:
        raise DeviceNotFoundError(
            "device key must use the '<class>:<id>' format"
        )
    return tuple(device_key.rsplit(':', 1))  # type: ignore[return-value]


def _origin_endpoint(value: str) -> Optional[Tuple[str, str, int]]:
    try:
        parsed = urlsplit(value)
        if parsed.path not in {'', '/'} or parsed.query or parsed.fragment:
            return None
        scheme = parsed.scheme.casefold()
        hostname = (parsed.hostname or '').casefold()
        if scheme not in {'http', 'https'} or not hostname:
            return None
        port = parsed.port or (443 if scheme == 'https' else 80)
        return scheme, hostname, int(port)
    except (TypeError, ValueError):
        return None


def _trusted_gripper_password_transport() -> bool:
    """Allow the operator-requested password flow on every same-origin transport."""
    return True


def create_app(
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    initial_config: Optional[str] = None,
    preview_reader_factory: Optional[Callable[[str, int], Any]] = None,
    robot_state_reader_factory: Optional[Callable[[int], Any]] = None,
    yam_state_reader_factory: Optional[Callable[[int], Any]] = None,
    hardware_discovery: Optional[HardwareDiscovery] = None,
    workflow_factory: Optional[Callable[..., WorkflowLifecycle]] = None,
    device_python: Optional[Path] = None,
    recording_python: Optional[Path] = None,
    conversion_jobs: Optional[ConversionJobManager] = None,
    recording_replay: Optional[RecordingReplayService] = None,
    recording_exports: Optional[RecordingExportManager] = None,
    gripper_control: Optional[GripperControlManager] = None,
    state_sync_factory: Optional[
        Callable[[Callable[[], Mapping[str, Any]]], StateSyncBroker]
    ] = None,
    gripper_permissions: Optional[TtyUsbPermissionAuthorizer] = None,
    can_permissions: Optional[CanInterfacePermissionAuthorizer] = None,
    checkpoint_browse_roots: Optional[Sequence[Path]] = None,
    inference_preferences_path: Optional[Path] = None,
) -> Flask:
    """Application factory used by production and hardware-free tests."""
    package_root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(package_root / 'templates'),
        static_folder=str(package_root / 'static'),
        static_url_path='/static',
    )
    controller = DeviceController(
        config_dir=config_dir,
        preview_reader_factory=preview_reader_factory,
        robot_state_reader_factory=robot_state_reader_factory,
        yam_state_reader_factory=yam_state_reader_factory,
        hardware_discovery=hardware_discovery,
        workflow_factory=workflow_factory,
        device_python=device_python,
        recording_python=recording_python,
        conversion_jobs=conversion_jobs,
        recording_exports=recording_exports,
        gripper_control=gripper_control,
        gripper_permissions=gripper_permissions,
        can_permissions=can_permissions,
        checkpoint_browse_roots=checkpoint_browse_roots,
        inference_preferences_path=inference_preferences_path,
    )
    app.config['DEVICE_CONTROLLER'] = controller
    state_sync = (state_sync_factory or StateSyncBroker)(controller.state)
    app.config['STATE_SYNC'] = state_sync
    control_dispatcher = ControlOperationDispatcher(
        state_sync,
        log_callback=controller.logs.append,
        cancel_callback=controller.request_async_stop,
    )
    app.config['CONTROL_OPERATION_DISPATCHER'] = control_dispatcher
    replay = recording_replay or RecordingReplayService(
        controller.recording_source_root,
        camera_roles_provider=controller.recording_camera_roles_config,
        published_shard_provider=(
            controller.conversion_jobs.published_shard_for_episode
        ),
        # The replay inspector must use the same validated NEDF runtime as the
        # recorder.  Otherwise a CLI-provided ``--recording-python`` would only
        # affect episode writing while inspection silently fell back to the
        # retired converter profile.
        inspector_python=controller.recording_python,
        inspector_python_provider=controller.recording_runtime_python,
    )
    app.config['RECORDING_REPLAY'] = replay
    export_manager = getattr(controller, 'recording_exports', None)
    if export_manager is None:
        class _UnavailableRecordingExports:
            @staticmethod
            def state(*, limit: int = 20) -> Dict[str, Any]:
                del limit
                return {
                    'schema_version': 1,
                    'configured': False,
                    'available': False,
                    'blockers': ['Recording export manager is unavailable'],
                    'counts': {},
                    'jobs': [],
                }

            @staticmethod
            def enqueue(
                episode_ids: Sequence[str],
                *,
                dataset_name: Any = None,
            ) -> Dict[str, Any]:
                del episode_ids, dataset_name
                raise RecordingExportError(
                    'Recording export manager is unavailable'
                )

        export_manager = _UnavailableRecordingExports()
        setattr(controller, 'recording_exports', export_manager)
    app.config['RECORDING_EXPORTS'] = export_manager
    finalized_enqueue = getattr(replay, 'enqueue_finalized', None)
    controller.set_recording_finalized_enqueue(
        finalized_enqueue if callable(finalized_enqueue) else None
    )
    app.config.setdefault('PREVIEW_STREAM_FPS', 5.0)
    if app.config.get('MAX_CONTENT_LENGTH') is None:
        app.config['MAX_CONTENT_LENGTH'] = GLOBAL_MAX_REQUEST_BYTES
    app.config.setdefault(
        'GRIPPER_PERMISSION_CSRF_TOKEN', secrets.token_urlsafe(32)
    )
    event_stream_lock = threading.Lock()
    event_stream_clients: set[str] = set()

    def _event_stream_client_id() -> str:
        cookie = str(
            request.cookies.get(EVENT_STREAM_CLIENT_COOKIE, '')
        ).strip()
        if EVENT_STREAM_CLIENT_PATTERN.fullmatch(cookie):
            return cookie
        # Pages opened before this release have no cookie until reloaded. Group
        # their reconnects by browser fingerprint so old tabs cannot consume
        # every HTTP/1.1 connection while a current tab sends a control action.
        legacy = '\0'.join(
            (
                str(request.remote_addr or ''),
                str(request.headers.get('User-Agent', '')),
            )
        )
        digest = hashlib.sha256(legacy.encode('utf-8')).hexdigest()[:32]
        return f'legacy-{digest}'

    def _run_observed_transition(
        scope: str,
        action: str,
        desired: str,
        timeout_ms: int,
        operation: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Register a cross-tab token and resolve it only from observations."""
        requested_token = request.headers.get('X-Control-Transition')
        try:
            transition = state_sync.begin(
                scope,
                action,
                desired,
                timeout_ms=timeout_ms,
                token=requested_token,
            )
        except ValueError as exc:
            raise DeviceControlError(str(exc)) from exc
        prefer_tokens = {
            token.strip().casefold()
            for value in request.headers.getlist('Prefer')
            for token in value.split(',')
        }
        respond_async = 'respond-async' in prefer_tokens
        if transition.get('duplicate') is True:
            return {
                'ok': True,
                'duplicate': True,
                'accepted': transition.get('status') in {
                    'pending',
                    'acknowledged',
                },
                'asynchronous': respond_async,
                'transition': transition,
            }
        if respond_async:
            try:
                acknowledged = control_dispatcher.submit(
                    token=transition['token'],
                    scope=scope,
                    action=action,
                    desired=desired,
                    operation=operation,
                )
            except (
                ControlOperationDispatcherClosed,
                ControlOperationQueueFull,
            ) as exc:
                state_sync.fail(transition['token'], exc)
                raise DeviceConflictError(str(exc)) from exc
            return {
                'ok': True,
                'accepted': True,
                'asynchronous': True,
                'transition': acknowledged,
            }
        try:
            result = operation()
        except BaseException as exc:
            state_sync.fail(transition['token'], exc)
            raise
        acknowledged = state_sync.acknowledge(transition['token'])
        result['transition'] = acknowledged or transition
        return result

    @app.get('/')
    def index() -> Response:
        template = Path(app.root_path) / 'templates' / 'device_control.html'
        if template.is_file():
            response = app.make_response(
                render_template(
                    'device_control.html',
                    gripper_permission_csrf=app.config[
                        'GRIPPER_PERMISSION_CSRF_TOKEN'
                    ],
                    gripper_permission_password_allowed=(
                        _trusted_gripper_password_transport()
                    ),
                )
            )
            response.headers['Cache-Control'] = 'no-store, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Referrer-Policy'] = 'no-referrer'
            cookie = str(
                request.cookies.get(EVENT_STREAM_CLIENT_COOKIE, '')
            ).strip()
            if not EVENT_STREAM_CLIENT_PATTERN.fullmatch(cookie):
                response.set_cookie(
                    EVENT_STREAM_CLIENT_COOKIE,
                    secrets.token_urlsafe(24),
                    max_age=EVENT_STREAM_COOKIE_MAX_AGE_S,
                    httponly=True,
                    samesite='Lax',
                )
            return response
        return jsonify(
            {
                'service': 'SuperInference Device Control',
                'api': '/api/state',
            }
        )

    @app.get('/recordings')
    def recordings_page() -> Response:
        template = Path(app.root_path) / 'templates' / 'recordings.html'
        if template.is_file():
            return render_template('recordings.html')
        return jsonify(
            {
                'service': 'SuperInference Recording Inspection',
                'api': '/api/recordings',
            }
        )

    @app.get('/api/health')
    def health() -> Response:
        return jsonify({'ok': True, 'pid': os.getpid()})

    @app.post('/api/processes/force-stop-all')
    def force_stop_all_processes() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {'confirmation'}:
            raise DeviceControlError(
                'JSON body must contain only the confirmation field'
            )
        if payload.get('confirmation') != FORCE_STOP_ALL_CONFIRMATION:
            raise DeviceControlError(
                'Explicit FORCE_STOP_ALL confirmation is required'
            )
        response = jsonify(controller.force_stop_all())
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/api/state')
    def state() -> Response:
        response = jsonify(state_sync.sample_if_due())
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Vary'] = 'Accept'
        return response

    @app.get('/api/recording')
    def recording_state() -> Response:
        snapshot = state_sync.sample_if_due()
        return jsonify(
            {
                'revision': snapshot['revision'],
                'recording': snapshot['recording'],
            }
        )

    def _query_int(name: str, default: Optional[int]) -> Optional[int]:
        raw = request.args.get(name)
        if raw is None or raw == '':
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise RecordingReplayRequestError(f'{name} must be an integer') from exc

    @app.get('/api/recordings')
    def recordings_list() -> Response:
        page = _query_int('page', 1)
        page_size = _query_int('page_size', 50)
        return jsonify(
            replay.list(
                page=1 if page is None else page,
                page_size=50 if page_size is None else page_size,
                status=request.args.get('status', 'all'),
                result=request.args.get('result', 'all'),
                date=request.args.get('date', 'all'),
                query=request.args.get('query', ''),
            )
        )

    @app.delete('/api/recordings')
    def recordings_delete() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RecordingReplayRequestError('JSON body must be an object')
        if payload.get('confirm') is not True:
            raise RecordingReplayRequestError(
                'Explicit confirmation is required to delete recordings'
            )
        return jsonify(replay.delete(payload.get('episode_ids')))

    @app.get('/api/recording-exports')
    def recording_exports_state() -> Response:
        limit = _query_int('limit', 20)
        return jsonify(
            controller.recording_exports.state(
                limit=20 if limit is None else limit,
            )
        )

    @app.post('/api/recording-exports')
    def create_recording_export() -> Tuple[Response, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RecordingExportError('JSON body must be an object')
        unexpected = set(payload) - {'episode_ids', 'dataset_name'}
        if unexpected:
            raise RecordingExportError(
                'Unexpected recording export fields: '
                + ', '.join(sorted(unexpected))
            )
        episode_ids = payload.get('episode_ids')
        if not isinstance(episode_ids, list):
            raise RecordingExportError('episode_ids must be an array')
        job = controller.recording_exports.enqueue(
            episode_ids,
            dataset_name=payload.get('dataset_name'),
        )
        return jsonify(job), 202

    def _recording_summary_payload(episode_id: str) -> Dict[str, Any]:
        summary = replay.summary(episode_id)
        config_name = summary.get('config_name')
        if not isinstance(config_name, str) or not config_name:
            return summary
        try:
            config = controller.inspect_config(config_name)
        except (RuntimeConfigError, OSError, ValueError):
            summary['visualization'] = {
                'config_name': config_name,
                'source': 'recorded-config-unavailable',
                'configured_tools': {},
            }
            return summary

        configured_tools = config.get('configured_tools')
        if not isinstance(configured_tools, Mapping):
            configured_tools = {}
        summary['visualization'] = {
            'config_name': config_name,
            'source': 'recorded-config',
            'configured_tools': copy.deepcopy(dict(configured_tools)),
        }

        camera_slots = config.get('camera_slots')
        camera_roles = summary.get('camera_roles')
        videos = summary.get('videos')
        if not isinstance(camera_slots, Mapping) or not isinstance(camera_roles, dict):
            return summary
        for role in ('head', 'left', 'right'):
            slot = camera_slots.get(role)
            camera = camera_roles.get(role)
            video = videos.get(role) if isinstance(videos, Mapping) else None
            if (
                not isinstance(slot, Mapping)
                or slot.get('configured') is not False
                or not isinstance(camera, Mapping)
                or bool(camera.get('count'))
                or (isinstance(video, Mapping) and video.get('available') is True)
            ):
                continue
            camera_roles[role] = {
                **dict(camera),
                'status': 'not_configured',
                'reason': None,
            }
        return summary

    @app.get('/api/recordings/<episode_id>/summary')
    def recording_summary(episode_id: str) -> Response:
        return jsonify(_recording_summary_payload(episode_id))

    @app.get('/api/recordings/<episode_id>/joint-trajectory')
    def recording_joint_trajectory(episode_id: str) -> Response:
        start = _query_int('start', 0)
        stride = _query_int('stride', 1)
        max_points = _query_int('max_points', 5000)
        return jsonify(
            replay.trajectory(
                episode_id,
                start=0 if start is None else start,
                end=_query_int('end', None),
                stride=1 if stride is None else stride,
                max_points=5000 if max_points is None else max_points,
            )
        )

    @app.get('/api/recordings/<episode_id>/playback-trajectory')
    def recording_playback_trajectory(episode_id: str) -> Response:
        start = _query_int('start', 0)
        stride = _query_int('stride', 1)
        max_points = _query_int('max_points', 5000)
        return jsonify(
            replay.playback_trajectory(
                episode_id,
                start=0 if start is None else start,
                end=_query_int('end', None),
                stride=1 if stride is None else stride,
                max_points=5000 if max_points is None else max_points,
            )
        )

    @app.get('/api/recordings/<episode_id>/videos/<role>')
    def recording_video(episode_id: str, role: str) -> Response:
        response = send_file(
            replay.video_path(episode_id, role),
            mimetype='video/mp4',
            conditional=True,
            etag=True,
            max_age=3600,
        )
        response.headers['Accept-Ranges'] = 'bytes'
        return response

    @app.get('/api/conversion-jobs')
    def conversion_jobs_state() -> Response:
        return jsonify(controller.conversion_jobs.state())

    @app.get('/api/grippers')
    def grippers_state() -> Response:
        snapshot = state_sync.sample_if_due()
        return jsonify(
            {
                'revision': snapshot['revision'],
                'state_sync': snapshot['state_sync'],
                'grippers': snapshot['grippers'],
            }
        )

    @app.get('/api/grippers/ports')
    def gripper_ports() -> Response:
        payload = state_sync.sample_if_due()
        grippers = payload['grippers']
        return jsonify(
            {
                'revision': payload['revision'],
                'available_ports': grippers.get('available_ports', []),
                'bindings': {
                    side: item.get('port')
                    for side, item in grippers.get('sides', {}).items()
                    if isinstance(item, Mapping) and item.get('port')
                },
                'can_bind_ports': grippers.get('can_bind_ports', False),
                'blockers': grippers.get('binding_blockers', []),
            }
        )

    def _permission_response(
        payload: Mapping[str, Any],
        status: int = 200,
        *,
        retry_after: Optional[int] = None,
    ) -> Response:
        response = jsonify(dict(payload))
        response.status_code = status
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        if retry_after is not None:
            response.headers['Retry-After'] = str(max(1, int(retry_after)))
        return response

    def _permission_error_response(error: GripperPermissionError) -> Response:
        payload: Dict[str, Any] = {
            'ok': False,
            'code': error.code,
            'error': str(error),
        }
        payload.update(error.details)
        return _permission_response(
            payload,
            error.status,
            retry_after=error.retry_after,
        )

    def _require_same_origin_permission_request() -> None:
        origin = _origin_endpoint(str(request.headers.get('Origin') or ''))
        expected = _origin_endpoint(request.host_url)
        fetch_site = str(request.headers.get('Sec-Fetch-Site') or '').casefold()
        if (
            origin is None
            or expected is None
            or origin != expected
            or fetch_site not in {'', 'none', 'same-origin'}
        ):
            raise GripperPermissionError(
                'permission_origin_rejected',
                '仅允许控制面板同源提交串口授权请求。',
                status=403,
            )
        supplied = str(request.headers.get('X-CSRF-Token') or '')
        expected_token = str(app.config['GRIPPER_PERMISSION_CSRF_TOKEN'])
        if not supplied or not secrets.compare_digest(supplied, expected_token):
            raise GripperPermissionError(
                'permission_csrf_rejected',
                '串口授权请求未通过 CSRF 校验。',
                status=403,
            )

    def _permission_client_key() -> str:
        return str(request.remote_addr or 'unknown')[:128]

    def _bounded_permission_json(max_bytes: int) -> Dict[str, Any]:
        if request.content_length is not None and request.content_length > max_bytes:
            raise GripperPermissionError(
                'invalid_permission_request',
                '串口权限授权请求无效。',
                status=422,
            )
        if request.mimetype != 'application/json':
            raise GripperPermissionError(
                'invalid_permission_request',
                '串口权限授权请求无效。',
                status=422,
            )
        raw_body = bytearray(request.stream.read(max_bytes + 1))
        try:
            if len(raw_body) > max_bytes:
                raise GripperPermissionError(
                    'invalid_permission_request',
                    '串口权限授权请求无效。',
                    status=422,
                )
            try:
                payload = json.loads(raw_body)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                raise GripperPermissionError(
                    'invalid_permission_request',
                    '串口权限授权请求无效。',
                    status=422,
                ) from None
            if not isinstance(payload, dict):
                raise GripperPermissionError(
                    'invalid_permission_request',
                    '串口权限授权请求无效。',
                    status=422,
                )
            return payload
        finally:
            for index in range(len(raw_body)):
                raw_body[index] = 0

    def _permission_context(
        payload: Mapping[str, Any],
    ) -> Tuple[int, list[str]]:
        expected_fields = {'revision', 'ports', 'password'}
        if set(payload) != expected_fields:
            raise GripperPermissionError(
                'invalid_permission_request',
                '串口权限授权请求无效。',
                status=422,
            )
        revision = payload.get('revision')
        ports = payload.get('ports')
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or not isinstance(ports, list)
            or not 1 <= len(ports) <= 2
            or any(
                not isinstance(port, str)
                or re.fullmatch(r'/dev/ttyUSB[0-9]+', port) is None
                for port in ports
            )
            or len(set(ports)) != len(ports)
        ):
            raise GripperPermissionError(
                'invalid_permission_request',
                '串口权限授权请求无效。',
                status=422,
            )
        return revision, sorted(ports)

    def _hardware_permission_context(
        payload: Mapping[str, Any],
    ) -> Tuple[int, list[str], list[str]]:
        expected_fields = {
            'revision',
            'ports',
            'can_interfaces',
            'password',
        }
        if set(payload) != expected_fields:
            raise GripperPermissionError(
                'invalid_permission_request',
                '设备权限授权请求无效。',
                status=422,
            )
        revision = payload.get('revision')
        ports = payload.get('ports')
        can_interfaces = payload.get('can_interfaces')
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or not isinstance(ports, list)
            or len(ports) > 2
            or any(
                not isinstance(port, str)
                or re.fullmatch(r'/dev/ttyUSB[0-9]+', port) is None
                for port in ports
            )
            or len(set(ports)) != len(ports)
            or not isinstance(can_interfaces, list)
            or len(can_interfaces) > 4
            or any(
                not isinstance(interface, str)
                or CAN_INTERFACE_PATTERN.fullmatch(interface) is None
                for interface in can_interfaces
            )
            or len(set(can_interfaces)) != len(can_interfaces)
            or not (ports or can_interfaces)
        ):
            raise GripperPermissionError(
                'invalid_permission_request',
                '设备权限授权请求无效。',
                status=422,
            )
        return revision, sorted(ports), sorted(can_interfaces)

    @app.post('/api/grippers/permissions/sudo')
    def authorize_gripper_permissions_with_sudo() -> Response:
        secret: Optional[bytearray] = None
        try:
            _require_same_origin_permission_request()
            payload = _bounded_permission_json(
                MAX_GRIPPER_PASSWORD_BODY_BYTES
            )
            revision, ports = _permission_context(payload)
            password = payload.get('password')
            if not isinstance(password, str):
                payload.clear()
                raise GripperPermissionError(
                    'invalid_sudo_password',
                    'sudo 密码格式无效。',
                    status=422,
                )
            secret = bytearray(password.encode('utf-8'))
            password = ''
            payload.clear()
            result = controller.repair_gripper_permissions(
                expected_revision=revision,
                expected_ports=ports,
                client_key=_permission_client_key(),
                secret=secret,
            )
            return _permission_response(result)
        except UnicodeEncodeError:
            return _permission_response(
                {
                    'ok': False,
                    'code': 'invalid_sudo_password',
                    'error': 'sudo 密码格式无效。',
                },
                422,
            )
        except GripperPermissionError as error:
            return _permission_error_response(error)
        except Exception:
            return _permission_response(
                {
                    'ok': False,
                    'code': 'permission_repair_failed',
                    'error': '串口权限授权失败。',
                },
                500,
            )
        finally:
            if secret is not None:
                for index in range(len(secret)):
                    secret[index] = 0

    @app.post('/api/hardware/permissions/sudo')
    def authorize_hardware_permissions_with_sudo() -> Response:
        secret: Optional[bytearray] = None
        try:
            _require_same_origin_permission_request()
            payload = _bounded_permission_json(
                MAX_GRIPPER_PASSWORD_BODY_BYTES
            )
            revision, ports, can_interfaces = (
                _hardware_permission_context(payload)
            )
            password = payload.get('password')
            if not isinstance(password, str):
                payload.clear()
                raise GripperPermissionError(
                    'invalid_sudo_password',
                    'sudo 密码格式无效。',
                    status=422,
                )
            secret = bytearray(password.encode('utf-8'))
            password = ''
            payload.clear()
            result = controller.repair_hardware_permissions(
                expected_revision=revision,
                expected_ports=ports,
                expected_can_interfaces=can_interfaces,
                client_key=_permission_client_key(),
                secret=secret,
            )
            return _permission_response(result)
        except UnicodeEncodeError:
            return _permission_response(
                {
                    'ok': False,
                    'code': 'invalid_sudo_password',
                    'error': 'sudo 密码格式无效。',
                },
                422,
            )
        except GripperPermissionError as error:
            return _permission_error_response(error)
        except Exception:
            return _permission_response(
                {
                    'ok': False,
                    'code': 'permission_repair_failed',
                    'error': '设备权限授权失败。',
                },
                500,
            )
        finally:
            if secret is not None:
                for index in range(len(secret)):
                    secret[index] = 0

    def _gripper_payload(expected_fields: set[str]) -> Dict[str, Any]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise GripperRequestError('A JSON object is required')
        unexpected = sorted(set(payload) - expected_fields)
        if unexpected:
            raise GripperRequestError(
                'Unexpected gripper fields: ' + ', '.join(unexpected)
            )
        missing = sorted(expected_fields - set(payload))
        if missing:
            raise GripperRequestError(
                'Missing gripper fields: ' + ', '.join(missing)
            )
        revision = payload.get('revision')
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise GripperRequestError('revision must be an integer')
        return payload

    @app.post('/api/grippers/bindings')
    def bind_gripper_ports() -> Response:
        payload = _gripper_payload({'bindings', 'revision'})
        bindings = payload['bindings']
        if not isinstance(bindings, Mapping):
            raise GripperRequestError('bindings must be an object')
        return jsonify(
            controller.update_gripper_ports(
                bindings,
                expected_revision=payload['revision'],
            )
        )

    @app.post('/api/grippers/calibrate')
    def calibrate_grippers() -> Tuple[Response, int]:
        payload = _gripper_payload({'side', 'revision', 'idempotency_key'})
        if not isinstance(payload['side'], str):
            raise GripperRequestError('side must be a string')
        if not isinstance(payload['idempotency_key'], str):
            raise GripperRequestError('idempotency_key must be a string')
        result = controller.control_gripper(
            'calibrate',
            side=payload['side'],
            expected_revision=payload['revision'],
            idempotency_key=payload['idempotency_key'],
        )
        return jsonify(result), 202 if result['command'].get('accepted') else 200

    @app.post('/api/grippers/position')
    def position_gripper() -> Tuple[Response, int]:
        payload = _gripper_payload(
            {
                'side',
                'target_percent',
                'revision',
                'side_revision',
                'idempotency_key',
            }
        )
        if not isinstance(payload['side'], str):
            raise GripperRequestError('side must be a string')
        if isinstance(payload['target_percent'], bool) or not isinstance(
            payload['target_percent'], (int, float)
        ):
            raise GripperRequestError('target_percent must be a number')
        if not isinstance(payload['idempotency_key'], str):
            raise GripperRequestError('idempotency_key must be a string')
        if isinstance(payload['side_revision'], bool) or not isinstance(
            payload['side_revision'], int
        ):
            raise GripperRequestError('side_revision must be an integer')
        result = controller.control_gripper(
            'position',
            side=payload['side'],
            target_percent=payload['target_percent'],
            expected_revision=payload['revision'],
            expected_side_revision=payload['side_revision'],
            idempotency_key=payload['idempotency_key'],
        )
        return jsonify(result), 202 if result['command'].get('accepted') else 200

    @app.post('/api/grippers/probe')
    def probe_gripper() -> Tuple[Response, int]:
        payload = _gripper_payload(
            {'side', 'revision', 'side_revision', 'idempotency_key'}
        )
        if not isinstance(payload['side'], str):
            raise GripperRequestError('side must be a string')
        if not isinstance(payload['idempotency_key'], str):
            raise GripperRequestError('idempotency_key must be a string')
        if isinstance(payload['side_revision'], bool) or not isinstance(
            payload['side_revision'], int
        ):
            raise GripperRequestError('side_revision must be an integer')
        result = controller.control_gripper(
            'probe',
            side=payload['side'],
            expected_revision=payload['revision'],
            expected_side_revision=payload['side_revision'],
            idempotency_key=payload['idempotency_key'],
        )
        return jsonify(result), 202 if result['command'].get('accepted') else 200

    @app.post('/api/grippers/stop')
    def stop_grippers() -> Response:
        payload = _gripper_payload({'revision'})
        return jsonify(
            controller.control_gripper(
                'stop', expected_revision=payload['revision']
            )
        )

    @app.post('/api/conversion-jobs/<job_id>/retry')
    def retry_conversion_job(job_id: str) -> Response:
        try:
            return jsonify({'ok': True, 'job': controller.conversion_jobs.retry(job_id)})
        except (ConversionJobError, FileNotFoundError, json.JSONDecodeError) as exc:
            return jsonify({'error': str(exc)}), 409

    @app.get('/api/configs')
    def configs() -> Response:
        selected = controller.selected_config_name()
        items = control_panel_config_items(
            controller.config_dir,
            selected=selected,
        )
        names = [item['name'] for item in items]
        recommended = recommended_runtime_config(names)
        for item in items:
            item['default'] = item['name'] == recommended
        return jsonify(
            {
                'items': items,
                'configs': names,
                'selected_config': selected,
                'recommended_config': recommended,
            }
        )

    @app.get('/api/configs/inspect')
    def inspect_config_query() -> Response:
        name = request.args.get('name', '')
        return jsonify(controller.inspect_config(name))

    @app.get('/api/configs/<path:config_name>')
    def inspect_config(config_name: str) -> Response:
        return jsonify(controller.inspect_config(config_name))

    @app.get('/api/preflight')
    def preflight() -> Response:
        def query_bool(name: str) -> bool:
            return request.args.get(name, '').strip().lower() in {
                '1', 'true', 'yes', 'on'
            }

        payload = controller.preflight(
            request.args.get('config_name') or request.args.get('name'),
            probe_cameras=query_bool('probe_cameras'),
            probe_robots=query_bool('probe_robots'),
            refresh_robots=query_bool('refresh_robots'),
        )
        response = jsonify(payload)
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/api/runtime/metrics')
    def runtime_metrics() -> Response:
        response = jsonify(controller.runtime_metrics())
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/api/runtime/tuning')
    def runtime_tuning() -> Response:
        response = jsonify(controller.runtime_tuning_schema())
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/api/hardware/discovery')
    def hardware_discovery() -> Response:
        # A candidate name is composed only for recommendations.  It does not
        # mutate the selected lifecycle or the controller revision.
        candidate = request.args.get('config_name')
        probe_cameras = request.args.get('probe_cameras', '').strip().lower()
        probe_robots = request.args.get('probe_robots', '').strip().lower()
        refresh_robots = request.args.get('refresh_robots', '').strip().lower()
        return jsonify(
            controller.discovery(
                candidate,
                probe_cameras=probe_cameras in {'1', 'true', 'yes', 'on'},
                probe_robots=probe_robots in {'1', 'true', 'yes', 'on'},
                refresh_robots=refresh_robots in {'1', 'true', 'yes', 'on'},
            )
        )

    @app.post('/api/config/select')
    def select_config() -> Response:
        payload = request.get_json(silent=True) or {}
        name = payload.get('name') or payload.get('config_name')
        if not name:
            raise RuntimeConfigError("JSON field 'name' is required")
        stop_active = payload.get('stop_active', False)
        if not isinstance(stop_active, bool):
            raise RuntimeConfigError("JSON field 'stop_active' must be a boolean")
        auto_apply = payload.get('auto_apply', False)
        if not isinstance(auto_apply, bool):
            raise RuntimeConfigError("JSON field 'auto_apply' must be a boolean")
        expected_revision = payload.get('expected_revision')
        if auto_apply:
            if not isinstance(expected_revision, int) or isinstance(
                expected_revision, bool
            ):
                raise RuntimeConfigError(
                    "JSON field 'expected_revision' must be an integer for auto_apply"
                )
            if 'expected_selected_config' not in payload:
                raise RuntimeConfigError(
                    "JSON field 'expected_selected_config' is required for auto_apply"
                )
            if payload.get('expected_selected_config') is not None:
                raise RuntimeConfigError(
                    "JSON field 'expected_selected_config' must be null for auto_apply"
                )
        return jsonify(
            controller.select_config(
                name,
                overrides=payload.get('overrides'),
                discovery_revision=payload.get('discovery_revision'),
                stop_active=stop_active,
                auto_apply=auto_apply,
                expected_revision=expected_revision,
                expected_selected_config=payload.get('expected_selected_config'),
                gripper_overrides=payload.get('gripper_overrides'),
            )
        )

    @app.post('/api/configs/generated')
    def export_generated_config() -> Tuple[Response, int]:
        payload = request.get_json(silent=True)
        required_fields = {
            'generated_id',
            'expected_selected_config',
            'expected_revision',
            'tuning_overrides',
            'overwrite',
        }
        if not isinstance(payload, Mapping) or set(payload) != required_fields:
            raise RuntimeConfigError(
                'JSON body must contain only generated_id, '
                'expected_selected_config, expected_revision, '
                'tuning_overrides, and overwrite'
            )
        if not isinstance(payload['generated_id'], str):
            raise RuntimeConfigError('generated_id must be a string')
        if not isinstance(payload['expected_selected_config'], str):
            raise RuntimeConfigError(
                'expected_selected_config must be a string'
            )
        if isinstance(payload['expected_revision'], bool) or not isinstance(
            payload['expected_revision'], int
        ):
            raise RuntimeConfigError('expected_revision must be an integer')
        if not isinstance(payload['tuning_overrides'], Mapping):
            raise RuntimeConfigError('tuning_overrides must be an object')
        if not isinstance(payload['overwrite'], bool):
            raise RuntimeConfigError('overwrite must be a boolean')
        result = controller.export_generated_config(
            generated_id=payload['generated_id'],
            expected_selected_config=payload['expected_selected_config'],
            expected_revision=payload['expected_revision'],
            tuning_overrides=payload['tuning_overrides'],
            overwrite=payload['overwrite'],
        )
        return jsonify(result), 201

    @app.post('/api/config/save')
    def save_current_config() -> Response:
        payload = request.get_json(silent=True)
        required_fields = {
            'expected_selected_config',
            'expected_revision',
            'tuning_overrides',
        }
        if not isinstance(payload, Mapping) or set(payload) != required_fields:
            raise RuntimeConfigError(
                'JSON body must contain only expected_selected_config, '
                'expected_revision, and tuning_overrides'
            )
        if not isinstance(payload['expected_selected_config'], str):
            raise RuntimeConfigError('expected_selected_config must be a string')
        if isinstance(payload['expected_revision'], bool) or not isinstance(
            payload['expected_revision'], int
        ):
            raise RuntimeConfigError('expected_revision must be an integer')
        if not isinstance(payload['tuning_overrides'], Mapping):
            raise RuntimeConfigError('tuning_overrides must be an object')
        result = controller.save_current_config(
            expected_selected_config=payload['expected_selected_config'],
            expected_revision=payload['expected_revision'],
            tuning_overrides=payload['tuning_overrides'],
        )
        return jsonify(result)

    @app.post('/api/bindings')
    def update_bindings() -> Response:
        payload = request.get_json(silent=True) or {}
        if 'overrides' not in payload:
            raise RuntimeConfigError("JSON field 'overrides' is required")
        requested_name = payload.get('name') or payload.get('config_name')
        if requested_name:
            requested_name = normalize_config_name(requested_name)
            if requested_name != controller.selected_config_name():
                raise DeviceConflictError(
                    "Binding config does not match the selected config"
                )
        return jsonify(
            controller.update_bindings(
                overrides=payload.get('overrides'),
                discovery_revision=payload.get('discovery_revision'),
            )
        )

    @app.get('/api/devices')
    def devices() -> Response:
        state_payload = state_sync.sample_if_due()
        return jsonify(
            {
                'revision': state_payload['revision'],
                'items': state_payload['devices'],
                'devices': state_payload['devices'],
            }
        )

    def _checkpoint_context_payload(
        *, require_path: bool = False
    ) -> Dict[str, Any]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RuntimeConfigError('JSON object body is required')
        expected_config = payload.get('expected_selected_config')
        expected_revision = payload.get('expected_revision')
        expected_generation = payload.get('expected_generation')
        if not isinstance(expected_config, str) or not expected_config:
            raise RuntimeConfigError(
                'expected_selected_config must be a non-empty string'
            )
        for key, value in (
            ('expected_revision', expected_revision),
            ('expected_generation', expected_generation),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise RuntimeConfigError(f'{key} must be an integer')
        if require_path and (
            not isinstance(payload.get('path'), str)
            or not payload.get('path', '').strip()
        ):
            raise RuntimeConfigError('path must be a non-empty string')
        return payload

    def _inference_options_context_payload() -> Dict[str, Any]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RuntimeConfigError('JSON object body is required')
        expected_config = payload.get('expected_selected_config')
        expected_revision = payload.get('expected_revision')
        expected_generation = payload.get('expected_generation')
        if not isinstance(expected_config, str) or not expected_config:
            raise RuntimeConfigError(
                'expected_selected_config must be a non-empty string'
            )
        for key, value in (
            ('expected_revision', expected_revision),
            ('expected_generation', expected_generation),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise RuntimeConfigError(f'{key} must be an integer')
        if not isinstance(payload.get('enabled'), bool):
            raise RuntimeConfigError('enabled must be a boolean')
        return payload

    def _workflow_start_context_payload(
        requirements: Mapping[str, bool],
    ) -> Dict[str, Any]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RuntimeConfigError('JSON object body is required')
        expected_config = payload.get('expected_selected_config')
        expected_revision = payload.get('expected_revision')
        if not isinstance(expected_config, str) or not expected_config:
            raise RuntimeConfigError(
                'expected_selected_config must be a non-empty string'
            )
        if not isinstance(expected_revision, int) or isinstance(
            expected_revision, bool
        ):
            raise RuntimeConfigError('expected_revision must be an integer')
        if requirements.get('checkpoint'):
            expected_generation = payload.get('expected_generation')
            if not isinstance(expected_generation, int) or isinstance(
                expected_generation, bool
            ):
                raise RuntimeConfigError(
                    'expected_generation must be an integer'
                )
        if requirements.get('options'):
            options_generation = payload.get('expected_options_generation')
            if not isinstance(options_generation, int) or isinstance(
                options_generation, bool
            ):
                raise RuntimeConfigError(
                    'expected_options_generation must be an integer'
                )
        return payload

    @app.get('/api/checkpoints/browse')
    def browse_checkpoints() -> Response:
        return jsonify(controller.browse_checkpoints(request.args.get('path')))

    @app.post('/api/checkpoint/override')
    def set_checkpoint_override() -> Response:
        payload = _checkpoint_context_payload(require_path=True)
        return jsonify(
            controller.set_checkpoint_override(
                payload['path'],
                expected_selected_config=payload['expected_selected_config'],
                expected_revision=payload['expected_revision'],
                expected_generation=payload['expected_generation'],
            )
        )

    @app.delete('/api/checkpoint/override')
    def clear_checkpoint_override() -> Response:
        payload = _checkpoint_context_payload()
        return jsonify(
            controller.clear_checkpoint_override(
                expected_selected_config=payload['expected_selected_config'],
                expected_revision=payload['expected_revision'],
                expected_generation=payload['expected_generation'],
            )
        )

    @app.get('/api/rollout/browse')
    def browse_rollout_paths() -> Response:
        return jsonify(controller.browse_rollout_paths(request.args.get('path')))

    @app.post('/api/inference/test-mode')
    def set_test_inference_mode() -> Response:
        payload = _inference_options_context_payload()
        return jsonify(
            controller.set_test_inference_mode(
                payload['enabled'],
                expected_selected_config=payload['expected_selected_config'],
                expected_revision=payload['expected_revision'],
                expected_generation=payload['expected_generation'],
            )
        )

    @app.post('/api/rollout/settings')
    def set_rollout_settings() -> Response:
        payload = _inference_options_context_payload()
        return jsonify(
            controller.set_rollout_settings(
                enabled=payload['enabled'],
                raw_path=payload.get('path'),
                expected_selected_config=payload['expected_selected_config'],
                expected_revision=payload['expected_revision'],
                expected_generation=payload['expected_generation'],
            )
        )

    @app.post('/api/workflow/<stage>/<action>')
    def control_workflow(stage: str, action: str) -> Response:
        if stage not in {'runtime', 'server', 'inference'} or action not in {'start', 'stop'}:
            raise DeviceControlError(
                f"Unknown workflow action: {stage}/{action}"
            )
        payload: Dict[str, Any] = {}
        if action == 'start' and stage in {'server', 'inference'}:
            requirements = controller.workflow_start_requirements(stage)
            payload = _workflow_start_context_payload(requirements)
        result = _run_observed_transition(
            f'workflow:{stage}',
            action,
            'running' if action == 'start' else 'stopped',
            (
                RUNTIME_START_TRANSITION_TIMEOUT_MS
                if stage == 'runtime' and action == 'start'
                else SERVER_START_TRANSITION_TIMEOUT_MS
                if stage == 'server' and action == 'start'
                else 30_000 if action == 'start' else 75_000
            ),
            lambda: controller.control_workflow(
                stage,
                action,
                expected_selected_config=payload.get(
                    'expected_selected_config'
                ),
                expected_revision=payload.get('expected_revision'),
                expected_generation=payload.get('expected_generation'),
                expected_options_generation=payload.get(
                    'expected_options_generation'
                ),
            ),
        )
        response = jsonify(result)
        if result.get('asynchronous') is True:
            response.headers['Preference-Applied'] = 'respond-async'
        status_code = 202 if action == 'start' or result.get('accepted') else 200
        return response, status_code

    @app.post('/api/recording/actions/<action>')
    def control_recording(action: str) -> Tuple[Response, int]:
        if action not in RECORDING_ACTIONS:
            raise RecordingRequestError(
                f'Unknown recording action: {action}'
            )
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RecordingRequestError('A JSON object is required')
        allowed_fields = {
            'episode_id',
            'expected_checkpoint_revision',
            'idempotency_key',
        }
        unexpected = sorted(set(payload) - allowed_fields)
        if unexpected:
            raise RecordingRequestError(
                'Unexpected recording fields: ' + ', '.join(unexpected)
            )
        missing = sorted(allowed_fields - set(payload))
        if missing:
            raise RecordingRequestError(
                'Missing recording fields: ' + ', '.join(missing)
            )
        if not isinstance(payload['episode_id'], str):
            raise RecordingRequestError('episode_id must be a string')
        if not isinstance(payload['idempotency_key'], str):
            raise RecordingRequestError('idempotency_key must be a string')
        expected_checkpoint_revision = payload['expected_checkpoint_revision']
        if isinstance(expected_checkpoint_revision, bool) or not isinstance(
            expected_checkpoint_revision, int
        ):
            raise RecordingRequestError(
                'expected_checkpoint_revision must be an integer'
            )
        result = controller.control_recording(
            action,
            episode_id=payload['episode_id'],
            expected_checkpoint_revision=expected_checkpoint_revision,
            idempotency_key=payload['idempotency_key'],
        )
        command_status = result['command']['status']
        status_code = 202 if command_status == 'pending' else 200
        if command_status in {'rejected', 'failed'}:
            status_code = 409
        return jsonify(result), status_code

    @app.post('/api/inference/reset')
    def reset_inference() -> Tuple[Response, int]:
        payload = request.get_json(silent=True)
        required_fields = {
            'idempotency_key',
            'expected_selected_config',
            'expected_revision',
        }
        if not isinstance(payload, dict) or set(payload) != required_fields:
            raise DeviceControlError(
                'JSON body must contain only idempotency_key, '
                'expected_selected_config, and expected_revision'
            )
        if not isinstance(payload['idempotency_key'], str):
            raise DeviceControlError('idempotency_key must be a string')
        if not isinstance(payload['expected_selected_config'], str):
            raise DeviceControlError(
                'expected_selected_config must be a string'
            )
        expected_revision = payload['expected_revision']
        if isinstance(expected_revision, bool) or not isinstance(
            expected_revision, int
        ):
            raise DeviceControlError('expected_revision must be an integer')
        result = controller.control_inference_reset(
            idempotency_key=payload['idempotency_key'],
            expected_selected_config=payload['expected_selected_config'],
            expected_revision=expected_revision,
        )
        status = result['command'].get('status')
        return jsonify(result), 202 if status == 'pending' else 200

    @app.post('/api/inference/<action>')
    def control_loaded_inference(action: str) -> Tuple[Response, int]:
        if action not in {'pause', 'resume'}:
            raise DeviceControlError('Unsupported inference control action')
        payload = request.get_json(silent=True)
        required_fields = {
            'idempotency_key',
            'expected_selected_config',
            'expected_revision',
        }
        if not isinstance(payload, dict) or set(payload) != required_fields:
            raise DeviceControlError(
                'JSON body must contain only idempotency_key, '
                'expected_selected_config, and expected_revision'
            )
        if not isinstance(payload['idempotency_key'], str):
            raise DeviceControlError('idempotency_key must be a string')
        if not isinstance(payload['expected_selected_config'], str):
            raise DeviceControlError(
                'expected_selected_config must be a string'
            )
        expected_revision = payload['expected_revision']
        if isinstance(expected_revision, bool) or not isinstance(
            expected_revision, int
        ):
            raise DeviceControlError('expected_revision must be an integer')
        result = controller.control_inference_pause_resume(
            action,
            idempotency_key=payload['idempotency_key'],
            expected_selected_config=payload['expected_selected_config'],
            expected_revision=expected_revision,
        )
        status = result['command'].get('status')
        return jsonify(result), 202 if status == 'pending' else 200

    @app.get('/api/robots/state')
    def robot_states() -> Response:
        # This aggregate endpoint accepts no caller-supplied device ID. The
        # controller derives and authorizes readers exclusively from the
        # currently selected config, so it cannot probe arbitrary SHM names.
        return jsonify(controller.read_robot_states())

    @app.get('/api/devices/<device_class>/<device_class_id>')
    @app.get('/api/devices/<device_class>/<device_class_id>/state')
    def device_state(device_class: str, device_class_id: str) -> Response:
        return jsonify(controller.device_status(device_class, device_class_id))

    def _preview_error_code(metadata: Dict[str, Any]) -> int:
        status = metadata.get('status')
        if status == 'not_found':
            return 404
        if status in {
            'invalid_header',
            'invalid_shape',
            'unsupported_dtype',
        }:
            return 422
        return 503

    def _jpeg_response(
        result: CameraPreviewResult, metadata: Dict[str, Any]
    ) -> Response:
        if result.jpeg_bytes is None:
            response = jsonify(metadata)
            response.status_code = _preview_error_code(metadata)
            return response
        response = Response(result.jpeg_bytes, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['X-Preview-Status'] = str(metadata.get('status', 'unknown'))
        timestamp_ns = metadata.get('timestamp_ns')
        if timestamp_ns is not None:
            response.headers['X-Preview-Timestamp-Ns'] = str(timestamp_ns)
        return response

    @app.get('/api/devices/<device_class>/<device_class_id>/preview/status')
    def preview_status(device_class: str, device_class_id: str) -> Response:
        _, metadata = controller.read_preview(device_class, device_class_id)
        # Polling clients need a normal JSON status even when SHM has not been
        # created yet; lifecycle/config authorization errors still use 4xx.
        return jsonify(metadata)

    @app.get('/api/devices/<device_class>/<device_class_id>/preview.jpg')
    def preview_jpeg(device_class: str, device_class_id: str) -> Response:
        browser_cookie = str(
            request.cookies.get(EVENT_STREAM_CLIENT_COOKIE, '')
        ).strip()
        if (
            EVENT_STREAM_CLIENT_PATTERN.fullmatch(browser_cookie)
            and request.args.get('client_version')
            != PREVIEW_BROWSER_CLIENT_VERSION
        ):
            response = jsonify(
                {
                    'ok': False,
                    'code': 'stale_preview_client',
                    'error': 'Refresh this control-panel tab before previewing',
                }
            )
            response.status_code = 409
            response.headers['Cache-Control'] = 'no-store, max-age=0'
            return response
        result, metadata = controller.read_preview(device_class, device_class_id)
        return _jpeg_response(result, metadata)

    @app.get('/api/devices/<device_class>/<device_class_id>/preview.mjpg')
    def preview_mjpeg(device_class: str, device_class_id: str) -> Response:
        first = controller.read_preview(device_class, device_class_id)
        if first[0].jpeg_bytes is None:
            return jsonify(first[1]), _preview_error_code(first[1])

        try:
            configured_fps = float(app.config['PREVIEW_STREAM_FPS'])
        except (TypeError, ValueError):
            configured_fps = 5.0
        interval = 1.0 / max(0.2, min(configured_fps, 10.0))

        def multipart_stream() -> Iterator[bytes]:
            pending: Optional[Tuple[CameraPreviewResult, Dict[str, Any]]] = first
            try:
                while True:
                    if pending is None:
                        pending = controller.read_preview(device_class, device_class_id)
                    result, metadata = pending
                    pending = None
                    if result.jpeg_bytes is not None:
                        timestamp_ns = metadata.get('timestamp_ns')
                        timestamp_header = (
                            f"X-Preview-Timestamp-Ns: {timestamp_ns}\r\n"
                            if timestamp_ns is not None
                            else ''
                        )
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            + f"Content-Length: {len(result.jpeg_bytes)}\r\n".encode('ascii')
                            + f"X-Preview-Status: {metadata.get('status', 'unknown')}\r\n".encode('ascii')
                            + timestamp_header.encode('ascii')
                            + b'\r\n'
                            + result.jpeg_bytes
                            + b'\r\n'
                        )
                    # Limit both successful streams and transient-error retries.
                    # time.sleep releases the GIL and GeneratorExit is handled
                    # below when Werkzeug closes a disconnected client.
                    time.sleep(interval if result.jpeg_bytes is not None else max(interval, 0.5))
            except (GeneratorExit, BrokenPipeError, ConnectionResetError):
                return
            except (DeviceControlError, OSError, RuntimeError, ValueError):
                # Config switches and server shutdown intentionally end streams.
                return

        response = Response(
            multipart_stream(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    def _control_response(action: str, device_class: str, device_class_id: str) -> Response:
        device_key = f'{device_class}:{device_class_id}'
        result = _run_observed_transition(
            f'device:{device_key}',
            action,
            'stopped' if action == 'stop' else 'running',
            45_000 if action in {'start', 'restart'} else 75_000,
            lambda: controller.control_device(action, device_class, device_class_id),
        )
        response = jsonify(result)
        if result.get('asynchronous') is True:
            response.headers['Preference-Applied'] = 'respond-async'
        status_code = (
            202
            if action in {'start', 'restart'} or result.get('accepted')
            else 200
        )
        return response, status_code

    @app.post('/api/devices/<device_class>/<device_class_id>/<action>')
    def control_device(
        device_class: str, device_class_id: str, action: str
    ) -> Response:
        if action not in {'start', 'stop', 'restart'}:
            raise DeviceControlError(f"Unknown action: {action}")
        return _control_response(action, device_class, device_class_id)

    @app.post('/api/devices/<device_key>/<action>')
    def control_device_by_key(device_key: str, action: str) -> Response:
        if action not in {'start', 'stop', 'restart'}:
            raise DeviceControlError(f"Unknown action: {action}")
        device_class, device_class_id = _parse_device_key(device_key)
        return _control_response(action, device_class, device_class_id)

    @app.get('/api/logs')
    def logs() -> Response:
        try:
            cursor = int(request.args.get('cursor', '0'))
            limit = int(request.args.get('limit', '250'))
        except ValueError as exc:
            raise DeviceControlError('cursor and limit must be integers') from exc
        return jsonify(
            controller.logs.query(
                cursor=max(0, cursor),
                limit=limit,
                device_key=request.args.get('device_key'),
            )
        )

    @app.get('/api/events')
    def events() -> Response:
        """Broadcast de-duplicated observations without one probe per tab."""
        last_event_header = request.headers.get('Last-Event-ID', '-1')
        client_id = _event_stream_client_id()
        with event_stream_lock:
            if client_id in event_stream_clients:
                response = Response(status=204)
                response.headers['Cache-Control'] = 'no-store, max-age=0'
                response.headers['Pragma'] = 'no-cache'
                return response
            event_stream_clients.add(client_id)

        def release_client() -> None:
            with event_stream_lock:
                event_stream_clients.discard(client_id)

        def stream() -> Any:
            try:
                try:
                    last_revision = int(last_event_header)
                except ValueError:
                    last_revision = -1
                state_sync.start()
                while True:
                    event_revision, snapshot = state_sync.wait_for_event(
                        last_revision, timeout=5.0
                    )
                    if snapshot is None:
                        heartbeat = state_sync.current()
                        yield (
                            "event: heartbeat\n"
                            f"data: {json.dumps(heartbeat)}\n\n"
                        )
                        continue
                    last_revision = event_revision
                    yield (
                        f"id: {event_revision}\n"
                        f"event: state\n"
                        f"data: {json.dumps(snapshot)}\n\n"
                    )
            finally:
                release_client()

        response = Response(stream(), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['X-Accel-Buffering'] = 'no'
        response.call_on_close(release_client)
        return response

    @app.errorhandler(DeviceNotFoundError)
    def device_not_found(error: DeviceNotFoundError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 404

    @app.errorhandler(DeviceConflictError)
    def device_conflict(error: DeviceConflictError) -> Tuple[Response, int]:
        payload: Dict[str, Any] = {'ok': False, 'error': str(error)}
        if isinstance(error, ConfigSwitchConflictError):
            payload.update(
                {
                    'selected_config': error.selected_config,
                    'stopped_previous': error.stopped_previous,
                    'stopped_devices': error.stopped_devices,
                    'incomplete_devices': error.incomplete_devices,
                    'external_devices': error.external_devices,
                }
            )
        elif isinstance(error, BootstrapConfigConflictError):
            payload.update(
                {
                    'expected_revision': error.expected_revision,
                    'revision': error.current_revision,
                    'expected_selected_config': error.expected_selected_config,
                    'selected_config': error.current_selected_config,
                }
            )
        return jsonify(payload), 409

    @app.errorhandler(StateTransitionConflict)
    def state_transition_conflict(
        error: StateTransitionConflict,
    ) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 409

    @app.errorhandler(RuntimeConfigError)
    def bad_config(error: RuntimeConfigError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 422

    @app.errorhandler(DeviceControlError)
    def bad_control(error: DeviceControlError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 400

    @app.errorhandler(WorkflowConflictError)
    def workflow_conflict(error: WorkflowConflictError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 409

    @app.errorhandler(WorkflowError)
    def bad_workflow(error: WorkflowError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 400

    @app.errorhandler(GripperConflictError)
    def gripper_conflict(error: GripperConflictError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 409

    @app.errorhandler(GripperPermissionRequiredError)
    def gripper_permission_required(
        error: GripperPermissionRequiredError,
    ) -> Response:
        return _permission_error_response(error)

    @app.errorhandler(CanPermissionRequiredError)
    def can_permission_required(
        error: CanPermissionRequiredError,
    ) -> Response:
        return _permission_error_response(error)

    @app.errorhandler(GripperPermissionError)
    def hardware_permission_error(
        error: GripperPermissionError,
    ) -> Response:
        return _permission_error_response(error)

    @app.errorhandler(GripperRequestError)
    def bad_gripper_request(error: GripperRequestError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 422

    @app.errorhandler(GripperControlError)
    def bad_gripper_control(error: GripperControlError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 400

    @app.errorhandler(RecordingConflictError)
    def recording_conflict(
        error: RecordingConflictError,
    ) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 409

    @app.errorhandler(RecordingRequestError)
    def bad_recording_request(
        error: RecordingRequestError,
    ) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 422

    @app.errorhandler(RecordingSecurityError)
    def recording_security_error(
        error: RecordingSecurityError,
    ) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 409

    @app.errorhandler(RecordingError)
    def bad_recording(error: RecordingError) -> Tuple[Response, int]:
        return jsonify({'ok': False, 'error': str(error)}), 400

    @app.errorhandler(RecordingReplayNotFound)
    def recording_replay_not_found(
        error: RecordingReplayNotFound,
    ) -> Tuple[Response, int]:
        return jsonify(
            {'ok': False, 'code': 'recording_not_found', 'error': str(error)}
        ), 404

    @app.errorhandler(RecordingReplayRequestError)
    def bad_recording_replay_request(
        error: RecordingReplayRequestError,
    ) -> Tuple[Response, int]:
        return jsonify(
            {'ok': False, 'code': 'invalid_recording_request', 'error': str(error)}
        ), 422

    @app.errorhandler(RecordingReplayBuilding)
    def recording_replay_building(error: RecordingReplayBuilding) -> Response:
        response = jsonify(
            {
                'cache_state': 'building',
                'retry_after_ms': error.retry_after_ms,
                'episode_id': error.episode_id,
            }
        )
        response.status_code = 202
        response.headers['Retry-After'] = str(
            max(1, math.ceil(error.retry_after_ms / 1000))
        )
        return response

    @app.errorhandler(RecordingReplayBuildFailed)
    def recording_replay_build_failed(
        error: RecordingReplayBuildFailed,
    ) -> Response:
        response = jsonify(
            {
                'cache_state': 'failed',
                'retry_after_ms': error.retry_after_ms,
                'error': str(error),
            }
        )
        response.status_code = 503
        response.headers['Retry-After'] = str(
            max(1, math.ceil(error.retry_after_ms / 1000))
        )
        return response

    @app.errorhandler(RecordingReplayUnavailable)
    def recording_replay_unavailable(
        error: RecordingReplayUnavailable,
    ) -> Tuple[Response, int]:
        return jsonify(
            {
                'ok': False,
                'code': error.code,
                'error': str(error),
                'availability': error.availability,
            }
        ), 409

    @app.errorhandler(RecordingExportError)
    def bad_recording_export(
        error: RecordingExportError,
    ) -> Tuple[Response, int]:
        return jsonify(
            {
                'ok': False,
                'code': 'invalid_recording_export',
                'error': str(error),
            }
        ), 422

    if initial_config:
        controller.select_config(initial_config)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18088)
    parser.add_argument('--config-name', default=None)
    parser.add_argument('--config-dir', type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument('--device-python', type=Path, default=None)
    parser.add_argument('--recording-python', type=Path, default=None)
    parser.add_argument(
        '--lock-file',
        type=Path,
        default=Path('/tmp/superinference_device_control.lock'),
    )
    parser.add_argument(
        '--no-process-cleanup',
        action='store_true',
        help='Do not stop stale SuperInference launcher processes before startup.',
    )
    parser.add_argument(
        '--force-process-cleanup',
        action='store_true',
        help='Force-kill related launcher processes after graceful cleanup fails.',
    )
    args = parser.parse_args()

    if not args.no_process_cleanup:
        try:
            force_cleanup = args.force_process_cleanup or force_cleanup_requested()
            if force_cleanup:
                cleanup_report = cleanup_before_startup('control_panel', force=True)
            else:
                cleanup_report = cleanup_before_startup('control_panel')
        except ProcessCleanupError as exc:
            raise RuntimeError(f'Cannot start control panel: {exc}') from exc
        if cleanup_report.candidates:
            print(f'control_panel: process cleanup: {cleanup_report.summary()}')

    instance_lock = SingleInstanceLock(args.lock_file)
    instance_lock.acquire()
    app = create_app(
        config_dir=args.config_dir,
        initial_config=args.config_name,
        device_python=args.device_python,
        recording_python=args.recording_python,
    )
    controller: DeviceController = app.config['DEVICE_CONTROLLER']
    state_sync: Optional[StateSyncBroker] = app.config.get('STATE_SYNC')
    control_dispatcher: Optional[ControlOperationDispatcher] = app.config.get(
        'CONTROL_OPERATION_DISPATCHER'
    )
    if state_sync is not None:
        state_sync.start()
    cleanup_lock = threading.Lock()
    cleaned_up = False
    cleanup_in_progress = False

    def cleanup() -> None:
        nonlocal cleaned_up, cleanup_in_progress
        with cleanup_lock:
            if cleaned_up or cleanup_in_progress:
                return
            cleanup_in_progress = True
        succeeded = False
        try:
            if control_dispatcher is not None:
                control_dispatcher.close()
            if state_sync is not None:
                state_sync.close()
            controller.shutdown()
            succeeded = True
        finally:
            with cleanup_lock:
                cleanup_in_progress = False
                if succeeded:
                    cleaned_up = True
        if succeeded:
            instance_lock.release()

    def handle_signal(signum: int, frame: Any) -> None:
        del frame
        # Ignore a second Ctrl-C/SIGTERM while bounded child cleanup runs.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            cleanup()
        except BaseException as exc:
            # Cleanup can be retried (for example after a stubborn worker
            # finally exits). Keep the controller process and its instance lock
            # alive so a second server cannot start beside an unresolved owned
            # motion worker.
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
            try:
                controller.logs.append(
                    'ERROR',
                    'Shutdown signal cleanup failed; control server remains '
                    f'fail-closed for retry: {type(exc).__name__}: {exc}',
                )
            except BaseException:
                pass
            return
        raise SystemExit(128 + signum)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    controller.logs.append(
        'INFO', f"Device control server listening on {args.host}:{args.port}"
    )
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == '__main__':
    main()
