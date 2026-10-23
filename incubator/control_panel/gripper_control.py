"""Isolated LKTech gripper control for the web control panel.

This module never imports or constructs a Flexiv RDK robot.  A spawned worker
loads only the bundled LKTech wrapper and always creates ``Gripper(robot=None)``.
Browser requests contain a side and a percentage; serial ports, motor IDs,
calibration paths, and SDK units are resolved from the selected effective
configuration in the server process.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .runtime_paths import resolve_device_python
from .gripper_contract import rizon_gripper_enabled_sides


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_MODULE = REPO_ROOT / 'third_party/lktech_motor_sdk/example/lktech_gripper.py'
SIDES = ('left', 'right')
POSITION_MIN_INTERVAL_S = 0.2
POSITION_DEADBAND_PERCENT = 0.25
# A repeated command may be suppressed only after measured hardware feedback
# confirms that the jaws are already at the requested target.  Comparing only
# against the previous command caused a partially moved gripper to get stuck:
# the browser retried the same target, but the manager treated it as deadband.
POSITION_SETTLE_TOLERANCE_PERCENT = 1.0
WORKER_IDLE_TIMEOUT_S = 10.0
POSITION_TIMEOUT_S = 20.0
PROBE_TIMEOUT_S = 8.0
CALIBRATE_ONE_TIMEOUT_S = 85.0
STOP_TIMEOUT_S = 5.0
DEFAULT_MOVE_SPEED_DPS = 180.0
# LKTech position commands encode a non-negative uint32 speed at 0.01 dps/LSB.
MIN_MOVE_SPEED_DPS = 0.01
MAX_MOVE_SPEED_DPS = ((1 << 32) - 1) / 100.0


class GripperControlError(RuntimeError):
    """Base error for a controlled gripper request."""


class GripperConflictError(GripperControlError):
    """The requested operation conflicts with active system state."""


class GripperRequestError(GripperControlError):
    """The request is malformed or outside the public command contract."""


class GripperCalibrationStateUnknownError(GripperControlError):
    """A calibration commit failed and the prior files could not be restored."""


TTYUSB_PATH_PATTERN = re.compile(r'/dev/ttyUSB[0-9]+\Z')


def validate_ttyusb_port(port: Any, *, require_present: bool = True) -> str:
    """Return one canonical, non-symlink ``/dev/ttyUSBN`` character device."""
    if not isinstance(port, str) or not TTYUSB_PATH_PATTERN.fullmatch(port):
        raise GripperRequestError('gripper port must match /dev/ttyUSB<number>')
    path = Path(port)
    if path.parent != Path('/dev') or path.name == '':
        raise GripperRequestError('gripper port must be a direct /dev/ttyUSB device')
    if not require_present:
        return port
    try:
        if (
            path.is_symlink()
            or path.resolve(strict=True) != path
            or not path.is_char_device()
        ):
            raise GripperRequestError(
                f'gripper port is not an available character device: {port}'
            )
    except OSError as exc:
        raise GripperRequestError(f'cannot inspect gripper port: {port}') from exc
    return port


def discover_ttyusb_ports(*, dev_root: Path = Path('/dev')) -> list[str]:
    """Passively enumerate safe ttyUSB character devices in stable numeric order."""
    try:
        candidates = list(Path(dev_root).glob('ttyUSB*'))
    except OSError:
        return []
    ports: set[str] = set()
    for candidate in candidates:
        name_match = re.fullmatch(r'ttyUSB([0-9]+)', candidate.name)
        if name_match is None:
            continue
        try:
            if candidate.is_symlink() or not candidate.is_char_device():
                continue
        except OSError:
            continue
        ports.add(str(candidate))
    return sorted(
        ports,
        key=lambda value: (int(Path(value).name.removeprefix('ttyUSB')), value),
    )


def _chmod_ttyusb(path: str, mode: int) -> None:
    try:
        os.chmod(path, mode)
        return
    except (OSError, PermissionError) as direct_error:
        try:
            result = subprocess.run(
                ['/usr/bin/sudo', '-n', '/bin/chmod', f'{mode:04o}', '--', path],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError) as sudo_exc:
            raise GripperControlError(
                f'Cannot chmod {path} to {mode:04o}: {sudo_exc}'
            ) from sudo_exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or str(direct_error)).strip()
            raise GripperControlError(
                f'Cannot chmod {path} to {mode:04o}: '
                f'exit {result.returncode}{f": {detail}" if detail else ""}'
            )


def ensure_ttyusb_permissions(
    port: Any,
    *,
    allow_world_rw: Optional[bool] = None,
) -> str:
    """Grant the owning worker process read/write access before an SDK connection."""
    safe_port = validate_ttyusb_port(port)
    try:
        metadata = os.lstat(safe_port)
        if not stat.S_ISCHR(metadata.st_mode):
            raise GripperControlError(
                f'gripper port changed to a non-character device: {safe_port}'
            )
        initial_identity = (metadata.st_dev, metadata.st_ino, metadata.st_rdev)
        process_has_owner_or_group = bool(
            metadata.st_uid == os.geteuid()
            or metadata.st_gid in set(os.getgroups())
            or metadata.st_gid == os.getegid()
        )
    except OSError as exc:
        raise GripperControlError(f'Cannot inspect permissions on {safe_port}') from exc
    if allow_world_rw is None:
        allow_world_rw = os.environ.get(
            'SUPERINFERENCE_GRIPPER_ALLOW_WORLD_RW', ''
        ).strip().casefold() in {'1', 'true', 'yes', 'on'}

    current_mode = stat.S_IMODE(metadata.st_mode)
    already_accessible = bool(
        os.access(safe_port, os.R_OK | os.W_OK, effective_ids=True)
        and (
            process_has_owner_or_group
            or (allow_world_rw and current_mode & 0o006 == 0o006)
        )
    )
    if already_accessible:
        # Revalidate after the permission probe so a ttyUSB unplug/replacement
        # cannot turn the approved path into a different node before connect.
        validate_ttyusb_port(safe_port)
        current = os.lstat(safe_port)
        if (
            not stat.S_ISCHR(current.st_mode)
            or (current.st_dev, current.st_ino, current.st_rdev) != initial_identity
        ):
            raise GripperControlError(
                f'gripper serial device changed while checking {safe_port}'
            )
        if not os.access(safe_port, os.R_OK | os.W_OK, effective_ids=True):
            raise GripperControlError(
                f'Read/write permission changed while checking {safe_port}'
            )
        return safe_port
    if process_has_owner_or_group:
        _chmod_ttyusb(safe_port, 0o660)
    elif allow_world_rw:
        _chmod_ttyusb(safe_port, 0o666)
    else:
        raise GripperControlError(
            f'Worker process is not the owner or device group for {safe_port}; '
            'add it to dialout or explicitly set '
            'SUPERINFERENCE_GRIPPER_ALLOW_WORLD_RW=1 to permit chmod 0666'
        )
    try:
        validate_ttyusb_port(safe_port)
        current = os.lstat(safe_port)
        if (
            not stat.S_ISCHR(current.st_mode)
            or (current.st_dev, current.st_ino, current.st_rdev) != initial_identity
        ):
            raise GripperControlError(
                f'gripper serial device changed while granting permission on {safe_port}'
            )
        accessible = os.access(
            safe_port, os.R_OK | os.W_OK, effective_ids=True
        )
    except OSError as exc:
        raise GripperControlError(
            f'Cannot verify read/write permission on {safe_port}'
        ) from exc
    if not accessible:
        raise GripperControlError(
            f'Read/write permission is still unavailable on {safe_port}'
        )
    return safe_port


def ttyusb_permission_policy_error(
    port: Any,
    *,
    allow_world_rw: Optional[bool] = None,
) -> Optional[str]:
    """Report a fail-closed permission-policy blocker without changing the device."""
    safe_port = validate_ttyusb_port(port)
    try:
        metadata = os.stat(safe_port)
        has_owner_or_group = bool(
            metadata.st_uid == os.geteuid()
            or metadata.st_gid in set(os.getgroups())
            or metadata.st_gid == os.getegid()
        )
    except OSError:
        return f'Cannot inspect permissions on {safe_port}'
    if has_owner_or_group:
        return None
    if allow_world_rw is None:
        allow_world_rw = os.environ.get(
            'SUPERINFERENCE_GRIPPER_ALLOW_WORLD_RW', ''
        ).strip().casefold() in {'1', 'true', 'yes', 'on'}
    if allow_world_rw:
        return None
    return (
        f'Worker process is not in the device group for {safe_port}; '
        'add it to dialout before connecting the gripper'
    )


@dataclass(frozen=True)
class GripperProfile:
    side: str
    port: str
    motor_id: int
    baudrate: int
    speed_dps: float
    speed_source: str
    read_timeout: float
    wait_timeout: float
    calibration_timeout: float
    calibration_settle_time: float
    min_torque_iq: int
    max_torque_iq: int
    min_encoder: Optional[int]
    max_encoder: Optional[int]
    min_angle_deg: Optional[float]
    max_angle_deg: Optional[float]
    arm_length_m: float
    apex_angle_max_deg: float
    gear_ratio: float
    force_limit: float
    calibration_file: str
    label: str

    @property
    def max_width_m(self) -> float:
        return 2.0 * self.arm_length_m * math.sin(
            math.radians(self.apex_angle_max_deg) / 2.0
        )

    @property
    def calibrated_span_deg(self) -> Optional[float]:
        if self.min_angle_deg is None or self.max_angle_deg is None:
            return None
        span = abs(float(self.max_angle_deg) - float(self.min_angle_deg))
        return span if math.isfinite(span) and span >= 1e-6 else None

    @property
    def sdk_move_velocity(self) -> float:
        """External ``Move.velocity`` that resolves to ``speed_dps`` in the SDK.

        With angle calibration the bundled wrapper converts velocity back to dps
        as ``calibrated_span_deg * velocity``.  Without an angle span it scales
        against the constructor's ``max_vel``; using the same value for both is
        therefore the wrapper-supported way to request the configured base dps.
        """
        span = self.calibrated_span_deg
        return self.speed_dps / span if span is not None else 1.0


def _side_value(value: Any, index: int, default: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else default
    return default if value is None else value


def _gripper_enabled(value: Any) -> bool:
    """Parse an enable flag conservatively at the motion-control boundary.

    YAML/JSON strings are common when values arrive through environment or web
    overrides. Unknown strings are disabled so a typo cannot start a motor.
    """
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value)) and value != 0
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _normalize_motor_id(value: Any, *, field: str = "motor_id") -> int:
    """Normalize a finite integer motor ID while rejecting boolean aliases."""
    if isinstance(value, bool) or value is None:
        raise GripperControlError(f"Invalid {field}: expected an integer motor ID")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GripperControlError(
            f"Invalid {field}: expected an integer motor ID"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise GripperControlError(f"Invalid {field}: expected an integer motor ID")
    motor_id = int(numeric)
    if not 1 <= motor_id <= 32:
        raise GripperControlError(
            f"Invalid {field}: expected a value from 1 to 32 (LKTech protocol)"
        )
    return motor_id


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _resolve_move_speed_dps(
    raw: Any,
    index: int,
    side: str,
) -> tuple[float, str]:
    value = _side_value(raw, index, None)
    if value is None:
        return DEFAULT_MOVE_SPEED_DPS, 'sdk_default'
    if isinstance(value, bool):
        raise GripperControlError(f'Invalid gripper.speed_dps for {side}: boolean')
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise GripperControlError(
            f'Invalid gripper.speed_dps for {side}: expected a number'
        ) from exc
    if (
        not math.isfinite(speed)
        or speed < MIN_MOVE_SPEED_DPS
        or speed > MAX_MOVE_SPEED_DPS
    ):
        raise GripperControlError(
            f'Invalid gripper.speed_dps for {side}: expected '
            f'{MIN_MOVE_SPEED_DPS}..{MAX_MOVE_SPEED_DPS} dps'
        )
    return speed, 'gripper.speed_dps'


def _safe_calibration_path(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise GripperControlError('LKTech gripper calibration file is required')
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    calibration_root = (REPO_ROOT / 'calibration').resolve(strict=False)
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(calibration_root)
    except ValueError as exc:
        raise GripperControlError(
            'LKTech calibration file must remain under repository calibration/'
        ) from exc
    current = calibration_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise GripperControlError(
                'LKTech calibration path must not contain symbolic links'
            )
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(calibration_root)
    except ValueError as exc:
        raise GripperControlError(
            'LKTech calibration file must remain under repository calibration/'
        ) from exc
    if resolved.exists() and not resolved.is_file():
        raise GripperControlError('LKTech calibration path is not a regular file')
    return resolved


def _read_calibration(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GripperControlError(
            f'Cannot read LKTech calibration file for {path.name}'
        ) from exc
    if not isinstance(value, dict):
        raise GripperControlError('LKTech calibration must be a JSON object')
    return value


def _assert_no_unresolved_calibration_transactions(
    profiles: Mapping[str, GripperProfile],
) -> None:
    """Fail closed when a prior process left calibration files indeterminate."""
    targets = {
        Path(profile.calibration_file).resolve(strict=False)
        for profile in profiles.values()
    }
    if not targets:
        return
    checked: set[Path] = set()
    for directory in sorted({path.parent for path in targets}, key=str):
        try:
            manifests = sorted(directory.glob('.gripper-calibration-*.json'))
        except OSError as exc:
            raise GripperControlError(
                f'Cannot inspect LKTech calibration transactions in {directory}'
            ) from exc
        for manifest_path in manifests:
            resolved_manifest = manifest_path.resolve(strict=False)
            if resolved_manifest in checked:
                continue
            checked.add(resolved_manifest)
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise GripperControlError(
                    f'Cannot verify LKTech calibration transaction {manifest_path.name}'
                ) from exc
            if not isinstance(manifest, Mapping):
                raise GripperControlError(
                    f'Invalid LKTech calibration transaction {manifest_path.name}'
                )
            state = str(manifest.get('state', '')).strip().casefold()
            if state in {'committed', 'rolled_back'}:
                continue
            raw_files = manifest.get('files')
            manifest_targets = {
                Path(str(item.get('path'))).resolve(strict=False)
                for item in raw_files
                if isinstance(raw_files, list)
                and isinstance(item, Mapping)
                and item.get('path')
            } if isinstance(raw_files, list) else set()
            if manifest_targets and targets.isdisjoint(manifest_targets):
                continue
            raise GripperControlError(
                'LKTech calibration storage is unresolved after transaction '
                f'{manifest_path.name} ({state or "unknown"}); inspect the calibration '
                'files and transaction backups before restarting gripper control'
            )


def extract_gripper_profiles(config: Mapping[str, Any]) -> Dict[str, GripperProfile]:
    """Resolve enabled nested LKTech grippers from a composed config."""
    raw_devices = config.get('devices', [])
    if isinstance(raw_devices, Mapping):
        raw_devices = raw_devices.get('devices', [])
    if not isinstance(raw_devices, list):
        return {}

    profiles: Dict[str, GripperProfile] = {}
    for device in raw_devices:
        if not isinstance(device, Mapping) or not _gripper_enabled(
            device.get('enabled', True)
        ):
            continue
        if str(device.get('class', '')).casefold() != 'rizonrobot'.casefold():
            continue
        gripper = device.get('gripper')
        if not isinstance(gripper, Mapping):
            flat_backend = str(device.get('gripper_backend') or '').strip().casefold()
            if flat_backend not in {'lktech', ''}:
                continue
            if flat_backend == '' and 'gripper_server' not in device:
                continue
            gripper = {
                'backend': 'lktech',
                'enabled': device.get('utilize_gripper', False),
                'port': device.get('gripper_port'),
                'motor_id': device.get('gripper_motor_id', 1),
                'baudrate': device.get('gripper_baudrate', 115200),
                'speed_dps': device.get('gripper_speed_dps', DEFAULT_MOVE_SPEED_DPS),
                'timeout': device.get('gripper_timeout', 0.1),
                'wait_timeout': device.get('gripper_wait_timeout', 15.0),
                'calibration_files': device.get('gripper_calibration_files'),
            }
        backend = str(gripper.get('backend', 'lktech')).strip().casefold()
        if backend != 'lktech':
            continue

        gripper_type = str(gripper.get('gripper_type', 'soft')).strip().casefold()
        if gripper_type != 'soft':
            raise GripperControlError(
                f'Unsupported LKTech gripper type: {gripper_type}'
            )
        defaults = {
            'min_torque_iq': -40,
            'max_torque_iq': 190,
            'arm_length_m': 0.06543,
            'apex_angle_max_deg': 90.0,
        }
        try:
            enabled_sides = rizon_gripper_enabled_sides(device, gripper)
        except (KeyError, TypeError, ValueError) as exc:
            raise GripperControlError(f"Invalid Rizon gripper stream contract: {exc}") from exc
        for index, side in enumerate(SIDES[:len(enabled_sides)]):
            if not enabled_sides[index]:
                continue
            if side in profiles:
                raise GripperControlError(
                    f'Multiple enabled LKTech grippers are configured for {side}'
                )
            port = str(_side_value(gripper.get('port'), index, '')).strip()
            if not port.startswith('/dev/') or Path(port).name != port.removeprefix('/dev/'):
                raise GripperControlError(f'Invalid configured LKTech port for {side}')
            calibration_path = _safe_calibration_path(
                _side_value(gripper.get('calibration_files'), index, None)
            )
            calibration = _read_calibration(calibration_path)
            # Port and motor binding always come from the effective YAML.  Old
            # calibration JSON files may contain a stale/reversed port field.
            label = str(calibration.get('label') or f'LKTech-Gripper-{index + 1}/2')
            speed_dps, speed_source = _resolve_move_speed_dps(
                gripper.get('speed_dps'), index, side
            )
            profile = GripperProfile(
                side=side,
                port=port,
                motor_id=_normalize_motor_id(
                    _side_value(gripper.get('motor_id'), index, 1),
                    field=f'gripper.motor_id for {side}',
                ),
                baudrate=int(_side_value(gripper.get('baudrate'), index, 115200)),
                speed_dps=speed_dps,
                speed_source=speed_source,
                read_timeout=float(_side_value(gripper.get('timeout'), index, 0.1)),
                wait_timeout=float(_side_value(gripper.get('wait_timeout'), index, 15.0)),
                calibration_timeout=float(
                    _side_value(gripper.get('calibration_timeout'), index, 15.0)
                ),
                calibration_settle_time=float(
                    _side_value(gripper.get('calibration_settle_time'), index, 0.24)
                ),
                min_torque_iq=int(
                    calibration.get(
                        'min_torque_iq',
                        _side_value(
                            gripper.get('min_torque_iq'),
                            index,
                            defaults['min_torque_iq'],
                        ),
                    )
                ),
                max_torque_iq=int(
                    calibration.get(
                        'max_torque_iq',
                        _side_value(
                            gripper.get('max_torque_iq'),
                            index,
                            defaults['max_torque_iq'],
                        ),
                    )
                ),
                min_encoder=_optional_int(calibration.get('min_encoder')),
                max_encoder=_optional_int(calibration.get('max_encoder')),
                min_angle_deg=_optional_float(calibration.get('min_angle_deg')),
                max_angle_deg=_optional_float(calibration.get('max_angle_deg')),
                arm_length_m=float(
                    calibration.get(
                        'arm_length_m',
                        _side_value(
                            gripper.get('arm_length_m'),
                            index,
                            defaults['arm_length_m'],
                        ),
                    )
                ),
                apex_angle_max_deg=float(
                    calibration.get(
                        'apex_angle_max_deg',
                        _side_value(
                            gripper.get('apex_angle_max_deg'),
                            index,
                            defaults['apex_angle_max_deg'],
                        ),
                    )
                ),
                gear_ratio=float(
                    calibration.get(
                        'gear_ratio',
                        _side_value(gripper.get('gear_ratio'), index, 1.0),
                    )
                ),
                force_limit=float(
                    _side_value(gripper.get('force_limit'), index, 100.0)
                ),
                calibration_file=str(calibration_path),
                label=label,
            )
            if profile.arm_length_m <= 0.0:
                raise GripperControlError(f'Invalid arm_length_m for {side} gripper')
            if not 0.0 < profile.apex_angle_max_deg <= 180.0:
                raise GripperControlError(
                    f'Invalid apex_angle_max_deg for {side} gripper'
                )
            profiles[side] = profile
    real_ports: Dict[str, str] = {}
    calibration_paths: Dict[str, str] = {}
    for side, profile in profiles.items():
        real_port = str(Path(profile.port).resolve(strict=False))
        if real_port in real_ports:
            raise GripperControlError(
                f'{side} and {real_ports[real_port]} grippers resolve to the same serial port'
            )
        real_ports[real_port] = side
        calibration_path = str(_safe_calibration_path(profile.calibration_file))
        if calibration_path in calibration_paths:
            raise GripperControlError(
                f'{side} and {calibration_paths[calibration_path]} grippers use the same calibration file'
            )
        calibration_paths[calibration_path] = side
    _assert_no_unresolved_calibration_transactions(profiles)
    return profiles


def serial_port_holders_many(
    ports: Sequence[str],
    *,
    exclude_pids: Sequence[int] = (),
    proc_root: Path = Path('/proc'),
) -> Dict[str, list[int]]:
    """Find holders for several character devices with one ``/proc`` scan."""
    unique_ports = list(dict.fromkeys(ports))
    holders_by_port: Dict[str, list[int]] = {
        port: [] for port in unique_ports
    }
    ports_by_device: Dict[int, list[str]] = {}
    for port in unique_ports:
        try:
            target = os.stat(port)
        except OSError:
            continue
        if not stat.S_ISCHR(target.st_mode):
            continue
        ports_by_device.setdefault(target.st_rdev, []).append(port)
    if not ports_by_device:
        return holders_by_port

    excluded = {os.getpid(), *[int(pid) for pid in exclude_pids if pid]}
    holders_by_device: Dict[int, set[int]] = {
        device: set() for device in ports_by_device
    }
    try:
        with os.scandir(proc_root) as processes:
            for proc in processes:
                if not proc.name.isdigit():
                    continue
                pid = int(proc.name)
                if pid in excluded:
                    continue
                matched_devices: set[int] = set()
                try:
                    with os.scandir(Path(proc.path) / 'fd') as descriptors:
                        for descriptor in descriptors:
                            try:
                                info = descriptor.stat()
                            except (FileNotFoundError, PermissionError, OSError):
                                continue
                            device = info.st_rdev
                            if (
                                stat.S_ISCHR(info.st_mode)
                                and device in holders_by_device
                            ):
                                holders_by_device[device].add(pid)
                                matched_devices.add(device)
                                if len(matched_devices) == len(ports_by_device):
                                    break
                except (FileNotFoundError, PermissionError, OSError):
                    continue
    except (FileNotFoundError, PermissionError, OSError):
        pass

    for device, device_ports in ports_by_device.items():
        holders = sorted(holders_by_device[device])
        for port in device_ports:
            holders_by_port[port] = holders.copy()
    return holders_by_port


def serial_port_holders(
    port: str,
    *,
    exclude_pids: Sequence[int] = (),
    proc_root: Path = Path('/proc'),
) -> list[int]:
    """Find processes with one configured character device open."""
    return serial_port_holders_many(
        [port],
        exclude_pids=exclude_pids,
        proc_root=proc_root,
    ).get(port, [])


def process_command(
    pid: int,
    *,
    proc_root: Path = Path('/proc'),
    max_chars: int = 300,
) -> Optional[str]:
    """Return one bounded, display-safe command for a visible process."""
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return None
    if numeric_pid <= 0 or max_chars <= 0:
        return None

    process_root = proc_root / str(numeric_pid)
    try:
        with (process_root / 'cmdline').open('rb') as handle:
            payload = handle.read(max_chars * 8 + 1)
    except (FileNotFoundError, PermissionError, OSError):
        payload = b''

    arguments = []
    for raw in payload.split(b'\0'):
        if not raw:
            continue
        value = raw.decode(errors='replace')
        value = re.sub(r'[\x00-\x1f\x7f]+', ' ', value).strip()
        if value:
            arguments.append(value)
    command = shlex.join(arguments) if arguments else ''

    if not command:
        try:
            command = (process_root / 'comm').read_text(
                encoding='utf-8', errors='replace'
            )
        except (FileNotFoundError, PermissionError, OSError):
            return None
        command = re.sub(r'[\x00-\x1f\x7f]+', ' ', command).strip()
    if not command:
        return None
    if len(command) > max_chars:
        return command[: max(1, max_chars - 3)] + '...'
    return command


def rizon_shared_memory_names() -> list[str]:
    root = Path('/dev/shm')
    if not root.is_dir():
        return []
    return sorted(
        item.name
        for item in root.iterdir()
        if item.name.casefold().startswith('rizonrobot_')
    )


def gripper_lock_path(port: str) -> Path:
    digest = hashlib.sha256(str(Path(port).resolve(strict=False)).encode()).hexdigest()
    return Path('/tmp') / f'superinference-gripper-{digest[:24]}.lock'


def _lock_path(port: str) -> Path:
    """Backward-compatible alias for older tests and helper code."""
    return gripper_lock_path(port)


def _calibration_payload_from_limits(
    profile: GripperProfile,
    limits: Mapping[str, Any],
) -> Dict[str, Any]:
    path = Path(profile.calibration_file)
    prior = _read_calibration(path)
    payload = dict(prior)
    payload.update(
        {
            'label': profile.label,
            'motor_id': profile.motor_id,
            'port': profile.port,
            'saved_at_epoch': time.time(),
            'min_encoder': limits.get('min_encoder'),
            'max_encoder': limits.get('max_encoder'),
            'min_angle_deg': limits.get('min_angle_deg'),
            'max_angle_deg': limits.get('max_angle_deg'),
            'gripper_type': prior.get('gripper_type', 'soft'),
            'min_torque_iq': profile.min_torque_iq,
            'max_torque_iq': profile.max_torque_iq,
            'arm_length_m': profile.arm_length_m,
            'apex_angle_max_deg': profile.apex_angle_max_deg,
            'max_width': profile.max_width_m,
            'gear_ratio': profile.gear_ratio,
        }
    )
    min_angle = payload.get('min_angle_deg')
    max_angle = payload.get('max_angle_deg')
    min_encoder = payload.get('min_encoder')
    max_encoder = payload.get('max_encoder')
    has_any_angle = min_angle is not None or max_angle is not None
    has_any_encoder = min_encoder is not None or max_encoder is not None
    if has_any_angle and (min_angle is None or max_angle is None):
        raise RuntimeError('LKTech calibration produced an incomplete angle range')
    if has_any_encoder and (min_encoder is None or max_encoder is None):
        raise RuntimeError('LKTech calibration produced an incomplete encoder range')
    if has_any_angle and has_any_encoder:
        raise RuntimeError('LKTech calibration produced ambiguous angle and encoder ranges')
    if not has_any_angle and not has_any_encoder:
        raise RuntimeError('LKTech calibration produced no position limits')
    if has_any_angle:
        low = float(min_angle)
        high = float(max_angle)
        if not math.isfinite(low) or not math.isfinite(high):
            raise RuntimeError('LKTech calibration produced non-finite angle limits')
        if abs(high - low) < 1e-6:
            raise RuntimeError('LKTech calibration angle span is zero')
    if has_any_encoder and int(min_encoder) == int(max_encoder):
        raise RuntimeError('LKTech calibration encoder span is zero')
    return payload


def _calibration_payload(profile: GripperProfile, gripper: Any) -> Dict[str, Any]:
    return _calibration_payload_from_limits(
        profile,
        {
            'min_encoder': getattr(gripper, '_min_encoder', None),
            'max_encoder': getattr(gripper, '_max_encoder', None),
            'min_angle_deg': getattr(gripper, '_min_angle_deg', None),
            'max_angle_deg': getattr(gripper, '_max_angle_deg', None),
        },
    )


def _profile_with_calibration(
    profile: GripperProfile,
    calibration: Mapping[str, Any],
) -> GripperProfile:
    """Validate one complete calibration payload and return a staged profile."""
    if not isinstance(calibration, Mapping):
        raise GripperControlError(f'Calibration for {profile.side} must be an object')

    def optional_encoder(name: str) -> Optional[int]:
        value = calibration.get(name)
        if value is None:
            return None
        if isinstance(value, bool):
            raise GripperControlError(
                f'Calibration for {profile.side} has invalid {name}'
            )
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GripperControlError(
                f'Calibration for {profile.side} has invalid {name}'
            ) from exc
        if not math.isfinite(number) or not number.is_integer():
            raise GripperControlError(
                f'Calibration for {profile.side} has invalid {name}'
            )
        return int(number)

    def optional_angle(name: str) -> Optional[float]:
        value = calibration.get(name)
        if value is None:
            return None
        if isinstance(value, bool):
            raise GripperControlError(
                f'Calibration for {profile.side} has invalid {name}'
            )
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GripperControlError(
                f'Calibration for {profile.side} has invalid {name}'
            ) from exc
        if not math.isfinite(number):
            raise GripperControlError(
                f'Calibration for {profile.side} has invalid {name}'
            )
        return number

    min_encoder = optional_encoder('min_encoder')
    max_encoder = optional_encoder('max_encoder')
    min_angle = optional_angle('min_angle_deg')
    max_angle = optional_angle('max_angle_deg')
    has_any_encoder = min_encoder is not None or max_encoder is not None
    has_any_angle = min_angle is not None or max_angle is not None
    if has_any_encoder and (min_encoder is None or max_encoder is None):
        raise GripperControlError(
            f'Calibration for {profile.side} has an incomplete encoder range'
        )
    if has_any_angle and (min_angle is None or max_angle is None):
        raise GripperControlError(
            f'Calibration for {profile.side} has an incomplete angle range'
        )
    if has_any_encoder and has_any_angle:
        raise GripperControlError(
            f'Calibration for {profile.side} has ambiguous position ranges'
        )
    if not has_any_encoder and not has_any_angle:
        raise GripperControlError(
            f'Calibration for {profile.side} has no usable limits'
        )
    if has_any_encoder and min_encoder == max_encoder:
        raise GripperControlError(
            f'Calibration for {profile.side} has a zero encoder span'
        )
    if has_any_angle and abs(float(max_angle) - float(min_angle)) < 1e-6:
        raise GripperControlError(
            f'Calibration for {profile.side} has a zero angle span'
        )
    return replace(
        profile,
        min_encoder=min_encoder,
        max_encoder=max_encoder,
        min_angle_deg=min_angle,
        max_angle_deg=max_angle,
    )


def _calibration_artifact_state(profile: GripperProfile) -> Dict[str, Any]:
    """Return bounded artifact status; valid means safe for motion.

    A usable limit range alone is insufficient. The persisted artifact must also
    prove that it was measured for this config's motor ID and serial port.
    """
    expected_identity = {
        'motor_id': profile.motor_id,
        'port': profile.port,
    }
    try:
        path = _safe_calibration_path(profile.calibration_file)
    except GripperControlError as exc:
        # A path outside the repository calibration root is never a motion-safe
        # artifact, even if an embedding caller supplied cached limits.
        return {
            'name': Path(str(profile.calibration_file)).name,
            'present': False,
            'limits_valid': False,
            'valid': False,
            'status': 'invalid',
            'identity_status': 'invalid',
            'expected': expected_identity,
            'observed': {},
            'reason': str(exc)[:256],
        }

    artifact: Dict[str, Any] = {
        'name': path.name,
        'present': False,
        'limits_valid': False,
        'valid': False,
        'status': 'missing',
        'identity_status': 'unverified',
        'expected': expected_identity,
        'observed': {},
        'reason': None,
    }
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return artifact
    except OSError as exc:
        artifact.update({'status': 'invalid', 'identity_status': 'invalid', 'reason': str(exc)[:256]})
        return artifact
    if not path.is_file():
        artifact.update({
            'status': 'invalid',
            'identity_status': 'invalid',
            'reason': 'calibration artifact is not a regular file',
        })
        return artifact
    if stat_result.st_size > 1_000_000:
        artifact.update({
            'status': 'invalid',
            'identity_status': 'invalid',
            'reason': 'calibration artifact exceeds the 1 MiB safety limit',
        })
        return artifact
    try:
        raw_bytes = path.read_bytes()
        calibration = _read_calibration(path)
        staged = _profile_with_calibration(profile, calibration)
    except (OSError, UnicodeError, GripperControlError, ValueError, TypeError) as exc:
        artifact.update({
            'present': True,
            'status': 'invalid',
            'identity_status': 'invalid',
            'reason': str(exc)[:256],
        })
        return artifact

    limits_valid = bool(
        (
            staged.min_angle_deg is not None
            and staged.max_angle_deg is not None
        )
        or (
            staged.min_encoder is not None
            and staged.max_encoder is not None
        )
    )
    observed: Dict[str, Any] = {}
    mismatches: list[str] = []
    raw_motor = calibration.get('motor_id')
    if raw_motor is not None:
        try:
            observed['motor_id'] = _normalize_motor_id(
                raw_motor,
                field='calibration.motor_id',
            )
            if observed['motor_id'] != profile.motor_id:
                mismatches.append('motor_id')
        except GripperControlError:
            mismatches.append('motor_id')
    raw_port = calibration.get('port')
    if raw_port is not None:
        if isinstance(raw_port, str) and raw_port:
            observed['port'] = raw_port
        else:
            mismatches.append('port')
        if observed.get('port') != profile.port:
            mismatches.append('port')
    mismatches = list(dict.fromkeys(mismatches))
    identity_verified = (
        limits_valid
        and not mismatches
        and observed.get('motor_id') == profile.motor_id
        and observed.get('port') == profile.port
    )
    if not limits_valid:
        status = 'invalid'
        identity_status = 'invalid'
        reason = 'calibration artifact has no usable limits'
    elif mismatches:
        status = 'stale'
        identity_status = 'mismatch'
        reason = (
            'calibration identity does not match the selected config: '
            + ', '.join(mismatches)
        )
    elif identity_verified:
        status = 'valid'
        identity_status = 'verified'
        reason = None
    else:
        status = 'unverified'
        identity_status = 'unverified'
        reason = 'calibration artifact must include matching motor_id and port'

    artifact.update({
        'present': True,
        'limits_valid': limits_valid,
        'valid': bool(limits_valid and identity_verified),
        'status': status,
        'identity_status': identity_status,
        'observed': observed,
        'reason': reason,
        'size_bytes': int(stat_result.st_size),
        'mtime_ns': int(stat_result.st_mtime_ns),
        'sha256': hashlib.sha256(raw_bytes).hexdigest()[:16],
    })
    return artifact


def _write_json_temp(path: Path, payload: Mapping[str, Any]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8', closefd=True) as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_calibration_payloads(
    calibrated: Sequence[tuple[GripperProfile, Mapping[str, Any]]],
) -> None:
    """Commit validated calibration payloads with backups and rollback."""
    if not calibrated:
        return
    normalized: list[tuple[GripperProfile, Dict[str, Any]]] = []
    targets: set[str] = set()
    for profile, raw_payload in calibrated:
        safe_path = _safe_calibration_path(profile.calibration_file)
        target = str(safe_path)
        if target in targets:
            raise RuntimeError('Duplicate LKTech calibration transaction target')
        targets.add(target)
        normalized_profile = replace(profile, calibration_file=target)
        payload = copy.deepcopy(dict(raw_payload))
        _profile_with_calibration(normalized_profile, payload)
        normalized.append((normalized_profile, payload))
    transaction_id = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prepared = []
    try:
        for profile, payload in normalized:
            path = Path(profile.calibration_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_bytes() if path.exists() else None
            temporary = _write_json_temp(path, payload)
            backup = None
            if original is not None:
                backup = path.with_suffix(path.suffix + f'.{stamp}.{transaction_id[:8]}.bak')
                backup.write_bytes(original)
                with backup.open('rb') as stream:
                    os.fsync(stream.fileno())
            prepared.append(
                {
                    'profile': profile,
                    'path': path,
                    'temporary': temporary,
                    'backup': backup,
                    'original': original,
                    'payload': payload,
                }
            )
    except BaseException:
        for item in prepared:
            item['temporary'].unlink(missing_ok=True)
            if item['backup'] is not None:
                item['backup'].unlink(missing_ok=True)
        raise

    manifest_dir = prepared[0]['path'].parent
    manifest_path = manifest_dir / f'.gripper-calibration-{transaction_id}.json'
    manifest = {
        'transaction_id': transaction_id,
        'state': 'prepared',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'files': [
            {
                'side': item['profile'].side,
                'path': str(item['path']),
                'backup': str(item['backup']) if item['backup'] else None,
            }
            for item in prepared
        ],
    }
    manifest_temp = _write_json_temp(manifest_path, manifest)
    os.replace(manifest_temp, manifest_path)
    _fsync_directory(manifest_dir)
    committed = []
    try:
        for item in prepared:
            os.replace(item['temporary'], item['path'])
            committed.append(item)
            _fsync_directory(item['path'].parent)
            reloaded = _read_calibration(item['path'])
            _profile_with_calibration(item['profile'], reloaded)
            if reloaded != item['payload']:
                raise RuntimeError(
                    f"Calibration verification mismatch for {item['profile'].side}"
                )
        manifest['state'] = 'committed'
        manifest['committed_at'] = datetime.now(timezone.utc).isoformat()
        replacement = _write_json_temp(manifest_path, manifest)
        os.replace(replacement, manifest_path)
        _fsync_directory(manifest_dir)
    except BaseException as commit_error:
        rollback_errors = []
        for item in reversed(committed):
            try:
                if item['original'] is None:
                    item['path'].unlink(missing_ok=True)
                else:
                    descriptor, restore_name = tempfile.mkstemp(
                        prefix=f".{item['path'].name}.rollback.",
                        dir=str(item['path'].parent),
                    )
                    restore = Path(restore_name)
                    with os.fdopen(descriptor, 'wb', closefd=True) as stream:
                        stream.write(item['original'])
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(restore, item['path'])
                _fsync_directory(item['path'].parent)
            except BaseException as exc:
                rollback_errors.append(str(exc))
        manifest['state'] = 'rollback_failed' if rollback_errors else 'rolled_back'
        manifest['rollback_errors'] = rollback_errors
        try:
            replacement = _write_json_temp(manifest_path, manifest)
            os.replace(replacement, manifest_path)
            _fsync_directory(manifest_dir)
        except BaseException as manifest_error:
            rollback_errors.append(
                'cannot persist transaction outcome: '
                f'{type(manifest_error).__name__}: {manifest_error}'
            )
        if manifest['state'] == 'rollback_failed':
            detail = '; '.join(rollback_errors)[:1000]
            raise GripperCalibrationStateUnknownError(
                'LKTech calibration commit failed and rollback was incomplete; '
                f'calibration storage is unknown: {detail}'
            ) from commit_error
        raise
    finally:
        for item in prepared:
            item['temporary'].unlink(missing_ok=True)


def _atomic_write_calibrations(
    calibrated: Sequence[tuple[GripperProfile, Any]],
) -> None:
    """Compatibility wrapper for SDK objects used by direct runtime tests."""
    _atomic_write_calibration_payloads(
        [(profile, _calibration_payload(profile, gripper)) for profile, gripper in calibrated]
    )


def _default_sdk_loader() -> tuple[Any, Any]:
    from third_party.lktech_motor_sdk.example.lktech_gripper import (
        Gripper,
        LKGripperConfig,
    )

    return LKGripperConfig, Gripper


def _default_motor_loader() -> Any:
    from third_party.lktech_motor_sdk.example.lktech_gripper import LKMotor

    return LKMotor


def _probe_effective_range(profile: GripperProfile) -> tuple[float, float, str]:
    if profile.min_angle_deg is not None and profile.max_angle_deg is not None:
        open_position = float(profile.min_angle_deg)
        closed_position = float(profile.max_angle_deg)
        mechanical_span = abs(open_position - closed_position)
        motor_limit = float(profile.apex_angle_max_deg) * float(profile.gear_ratio)
        clipped_span = min(mechanical_span, motor_limit)
        direction = 1.0 if open_position >= closed_position else -1.0
        return closed_position + direction * clipped_span, closed_position, 'angle'
    if profile.min_encoder is not None and profile.max_encoder is not None:
        return float(profile.min_encoder), float(profile.max_encoder), 'encoder'
    raise RuntimeError(f'{profile.side} gripper has no usable position calibration')


def _align_probe_angle(profile: GripperProfile, raw_angle: float) -> float:
    open_position = float(profile.min_angle_deg)
    closed_position = float(profile.max_angle_deg)
    low = min(open_position, closed_position)
    high = max(open_position, closed_position)
    midpoint = 0.5 * (open_position + closed_position)
    period = 3600.0
    base = int(round((midpoint - float(raw_angle)) / period))
    candidates = [float(raw_angle) + offset * period for offset in range(base - 1, base + 2)]

    def score(value: float) -> tuple[float, float]:
        if value < low:
            distance = low - value
        elif value > high:
            distance = value - high
        else:
            distance = 0.0
        return distance, abs(value - midpoint)

    return min(candidates, key=score)


def _read_probe_position(
    motor: Any,
    profile: GripperProfile,
    *,
    retries: int = 5,
    retry_interval: float = 0.05,
) -> float:
    attempts = max(1, int(retries))
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            if profile.min_angle_deg is not None and profile.max_angle_deg is not None:
                return _align_probe_angle(profile, float(motor.read_multi_turn_angle()))
            try:
                return float(motor.read_status2().encoder)
            except BaseException as status_error:
                try:
                    return float(motor.read_encoder().encoder)
                except BaseException as encoder_error:
                    raise RuntimeError(
                        f'status2 failed: {status_error}; read_encoder failed: {encoder_error}'
                    ) from encoder_error
        except BaseException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(retry_interval)
    raise RuntimeError(
        f'{profile.side} read-only gripper probe failed after {attempts} attempts: '
        f'{last_error}'
    ) from last_error


def _probe_result(profile: GripperProfile, current_position: float) -> Dict[str, Any]:
    open_position, closed_position, _ = _probe_effective_range(profile)
    span = closed_position - open_position
    if not math.isfinite(span) or abs(span) < 1e-6:
        raise RuntimeError(f'{profile.side} gripper calibration span is invalid')
    normalized_open = 1.0 - (float(current_position) - open_position) / span
    normalized_open = min(1.0, max(0.0, normalized_open))
    apex_radians = math.radians(float(profile.apex_angle_max_deg))
    max_width = profile.max_width_m
    actual_width = 2.0 * float(profile.arm_length_m) * math.sin(
        normalized_open * apex_radians / 2.0
    )
    actual_percent = min(100.0, max(0.0, actual_width / max_width * 100.0))
    return {
        'side': profile.side,
        'target_percent': None,
        'actual_width_m': actual_width,
        'actual_percent': actual_percent,
    }


def _probe_gripper_read_only(
    profile: GripperProfile,
    *,
    motor_loader: Callable[[], Any] = _default_motor_loader,
    preflight: Optional[Callable[[Mapping[str, GripperProfile]], None]] = None,
) -> Dict[str, Any]:
    if preflight is None:
        preflight = _assert_worker_hardware_idle
    preflight({profile.side: profile})
    motor_class = motor_loader()
    motor = motor_class(
        profile.port,
        motor_id=profile.motor_id,
        baudrate=profile.baudrate,
        timeout=profile.read_timeout,
    )
    read_error: Optional[BaseException] = None
    result: Optional[Dict[str, Any]] = None
    try:
        motor.connect()
        preflight({profile.side: profile})
        result = _probe_result(profile, _read_probe_position(motor, profile))
    except BaseException as exc:
        read_error = exc
    try:
        motor.disconnect()
    except BaseException as exc:
        if read_error is None:
            read_error = RuntimeError(
                f'{profile.side} read-only gripper probe could not close serial: '
                f'{type(exc).__name__}: {exc}'
            )
    if read_error is not None:
        raise read_error
    if result is None:
        raise RuntimeError(f'{profile.side} read-only gripper probe returned no result')
    return result


def _assert_worker_hardware_idle(
    profiles: Mapping[str, GripperProfile],
) -> None:
    """Recheck passive safety gates immediately before opening a serial port."""
    shm_names = rizon_shared_memory_names()
    if shm_names:
        raise RuntimeError('Rizon shared memory exists; gripper worker refused to start')
    for side, profile in profiles.items():
        try:
            info = os.stat(profile.port)
        except OSError as exc:
            raise RuntimeError(f'{side} gripper serial port is unavailable') from exc
        if not stat.S_ISCHR(info.st_mode):
            raise RuntimeError(f'{side} gripper serial port is not a character device')
        if serial_port_holders(profile.port):
            raise RuntimeError(
                f'{side} gripper serial port is held by another process'
            )


def _close_worker_runtime(runtime: '_WorkerRuntime', attempts: int = 2) -> Optional[BaseException]:
    last_error: Optional[BaseException] = None
    for _ in range(max(1, int(attempts))):
        try:
            runtime.close()
            return None
        except BaseException as exc:
            last_error = exc
    return last_error


class _WorkerRuntime:
    """SDK adapter kept separate so tests can use a fake gripper implementation."""

    def __init__(
        self,
        profiles: Mapping[str, GripperProfile],
        *,
        sdk_loader: Callable[[], tuple[Any, Any]] = _default_sdk_loader,
        calibration_writer: Callable[
            [Sequence[tuple[GripperProfile, Any]]], None
        ] = _atomic_write_calibrations,
        preflight: Callable[[Mapping[str, GripperProfile]], None] = _assert_worker_hardware_idle,
    ) -> None:
        self.profiles = dict(profiles)
        self.config_class, self.gripper_class = sdk_loader()
        self.calibration_writer = calibration_writer
        self.preflight = preflight
        self.grippers: Dict[str, Any] = {}

    def _create(self, side: str, *, recalibrate: bool) -> Any:
        profile = self.profiles[side]
        self.preflight({side: profile})
        config = self.config_class(
            port=profile.port,
            motor_id=profile.motor_id,
            baudrate=profile.baudrate,
            speed_dps=profile.speed_dps,
            home_position=1.0,
            read_timeout=profile.read_timeout,
            wait_timeout=profile.wait_timeout,
            auto_calibrate=False,
            min_torque_iq=profile.min_torque_iq,
            max_torque_iq=profile.max_torque_iq,
            calibration_timeout=profile.calibration_timeout,
            calibration_settle_time=profile.calibration_settle_time,
            calibration_dir=None,
            min_encoder=None if recalibrate else profile.min_encoder,
            max_encoder=None if recalibrate else profile.max_encoder,
            min_angle_deg=None if recalibrate else profile.min_angle_deg,
            max_angle_deg=None if recalibrate else profile.max_angle_deg,
            arm_length_m=profile.arm_length_m,
            apex_angle_max_deg=profile.apex_angle_max_deg,
            gear_ratio=profile.gear_ratio,
            label=profile.label,
        )
        gripper = self.gripper_class(
            None,
            config=config,
            min_vel=0.0,
            max_vel=profile.sdk_move_velocity,
        )
        # Retain ownership before Enable. If the post-open race check fails and
        # Disable also fails, worker-level cleanup must still be able to retry
        # the exact object instead of losing the only hardware handle.
        self.grippers[side] = gripper
        try:
            gripper.Enable(profile.label)
            # Existing device workers do not participate in our advisory lock.
            # Recheck after the SDK opens the port and fail closed on a race.
            self.preflight({side: profile})
        except BaseException:
            try:
                self._close_side(side)
            except BaseException:
                pass
            raise
        return gripper

    def _close_side(self, side: str) -> None:
        gripper = self.grippers.get(side)
        if gripper is not None:
            gripper.Disable()
            self.grippers.pop(side, None)

    def execute(self, command: Mapping[str, Any]) -> Dict[str, Any]:
        action = str(command['action'])
        requested_side = str(command['side'])
        sides = list(SIDES) if requested_side == 'both' else [requested_side]
        results = []
        calibrated = []
        for side in sides:
            if side not in self.profiles:
                raise RuntimeError(f'{side} LKTech gripper is not configured')
            profile = self.profiles[side]
            if action == 'probe':
                results.append(
                    _probe_gripper_read_only(profile, preflight=self.preflight)
                )
                continue
            if action == 'calibrate':
                self._close_side(side)
                gripper = self._create(side, recalibrate=True)
                self.grippers[side] = gripper
                try:
                    calibrate = getattr(gripper, '_calibrate_encoder_range', None)
                    if not callable(calibrate):
                        raise RuntimeError('LKTech wrapper has no explicit recalibration API')
                    calibrate()
                    calibration_payload = _calibration_payload(profile, gripper)
                    state = gripper.states()
                    calibrated.append((profile, gripper))
                finally:
                    # Strictly finish and power off one side before touching the next.
                    self._close_side(side)
            elif action == 'position':
                gripper = self.grippers.get(side)
                if gripper is None:
                    gripper = self._create(side, recalibrate=False)
                    self.grippers[side] = gripper
                target_percent = float(command['target_percent'])
                target_width = profile.max_width_m * target_percent / 100.0
                if not gripper.Move(
                    target_width,
                    profile.sdk_move_velocity,
                    profile.force_limit,
                ):
                    raise RuntimeError(f'{side} gripper rejected the position command')
                motor = getattr(gripper, 'motor', None)
                wait_for_stop = getattr(motor, 'wait_for_stop', None)
                if callable(wait_for_stop) and not wait_for_stop(
                    timeout=min(profile.wait_timeout, POSITION_TIMEOUT_S - 1.0)
                ):
                    raise RuntimeError(f'{side} gripper did not settle before timeout')
                state = self.grippers[side].states()
            else:
                raise RuntimeError(f'Unknown worker action: {action}')
            actual_width = float(state.width)
            result = {
                'side': side,
                'target_percent': command.get('target_percent'),
                'actual_width_m': actual_width,
                'actual_percent': max(
                    0.0,
                    min(100.0, actual_width / profile.max_width_m * 100.0),
                ),
                'move_speed_dps': profile.speed_dps,
            }
            if action == 'calibrate' and command.get('defer_commit'):
                result['calibration'] = calibration_payload
            results.append(result)
        if calibrated and not command.get('defer_commit'):
            self.calibration_writer(calibrated)
        return {'results': results}

    def close(self) -> None:
        errors = []
        for side in list(self.grippers):
            try:
                self._close_side(side)
            except BaseException as exc:
                errors.append(f'{side}: {type(exc).__name__}: {exc}')
        if errors:
            raise RuntimeError('LKTech gripper cleanup failed: ' + '; '.join(errors))


def _gripper_worker(
    connection: Any,
    profiles: Mapping[str, GripperProfile],
    owner_token: str,
    idle_timeout_s: float,
) -> None:
    locks = []
    runtime: Optional[_WorkerRuntime] = None

    def emit(event: str, **fields: Any) -> None:
        try:
            connection.send({'event': event, 'owner_token': owner_token, **fields})
        except (BrokenPipeError, EOFError, OSError):
            pass

    def stop_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise SystemExit(0)

    try:
        if os.name == 'posix':
            os.setsid()
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        for profile in profiles.values():
            lock_path = _lock_path(profile.port)
            handle = lock_path.open('a+', encoding='utf-8')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(
                    f'{profile.side} gripper serial lock is held by another owner'
                ) from exc
            locks.append(handle)
        _assert_worker_hardware_idle(profiles)
        runtime = _WorkerRuntime(profiles)
        emit('ready', pid=os.getpid(), pgid=os.getpgrp())
        last_activity = time.monotonic()
        while True:
            remaining = max(0.05, idle_timeout_s - (time.monotonic() - last_activity))
            if not connection.poll(min(0.25, remaining)):
                if time.monotonic() - last_activity >= idle_timeout_s:
                    emit('idle_timeout')
                    return
                continue
            command = connection.recv()
            if not isinstance(command, Mapping):
                raise RuntimeError('Worker command must be an object')
            if command.get('action') == 'shutdown':
                emit('stopped', reason=str(command.get('reason', 'requested')))
                return
            command_id = str(command.get('command_id', ''))
            emit(
                'started',
                command_id=command_id,
                action=command.get('action'),
                side=command.get('side'),
            )
            try:
                result = runtime.execute(command)
            except BaseException as command_exc:
                cleanup_error = _close_worker_runtime(runtime)
                if cleanup_error is None:
                    runtime = None
                emit(
                    'failed',
                    command_id=command_id,
                    action=command.get('action'),
                    side=command.get('side'),
                    error=f'{type(command_exc).__name__}: {command_exc}'[:500],
                    hardware_unknown=cleanup_error is not None,
                )
                return
            cleanup_error = _close_worker_runtime(runtime)
            if cleanup_error is not None:
                emit(
                    'failed',
                    command_id=command_id,
                    action=command.get('action'),
                    side=command.get('side'),
                    error=(
                        'LKTech cleanup failed: '
                        f'{type(cleanup_error).__name__}: {cleanup_error}'
                    )[:500],
                    hardware_unknown=True,
                )
                return
            runtime = None
            emit(
                'completed',
                command_id=command_id,
                action=command.get('action'),
                side=command.get('side'),
                **result,
            )
            # Until the UI implements a trusted heartbeat, every command is a
            # short motor session. finally calls Disable before the worker exits.
            return
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            emit('worker_failed', error=f'{type(exc).__name__}: {exc}'[:500])
    finally:
        if runtime is not None:
            cleanup_error = _close_worker_runtime(runtime)
            if cleanup_error is not None:
                emit(
                    'worker_failed',
                    error=(
                        'LKTech final cleanup failed: '
                        f'{type(cleanup_error).__name__}: {cleanup_error}'
                    )[:500],
                    hardware_unknown=True,
                )
            else:
                emit('cleanup_completed')
        for handle in reversed(locks):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            except OSError:
                pass
        try:
            connection.close()
        except (AttributeError, OSError):
            pass


@dataclass
class _WorkerSlot:
    """One isolated worker lane.

    Position lanes are keyed by ``left``/``right`` and receive only that
    side's profile. Calibration lanes are keyed by ``calibration:<side>``;
    their hardware sessions run concurrently while the manager coordinates
    one all-or-nothing calibration transaction.
    """

    lane: str
    process: Any
    connection: Any
    owner_token: str
    ready: bool = False
    pid: Optional[int] = None
    pgid: Optional[int] = None
    inflight: Optional[Dict[str, Any]] = None
    cleanup_acknowledged: bool = False


@dataclass
class _CalibrationTransaction:
    command_id: str
    requested_side: str
    sides: tuple[str, ...]
    started_at: float
    started_at_epoch: float
    results: Dict[str, Dict[str, Any]]
    progress: Dict[str, str]
    errors: Dict[str, str]
    phase: str = 'starting'


class GripperControlManager:
    """Own isolated, rate-limited gripper workers scoped by side."""

    def __init__(
        self,
        *,
        log_callback: Optional[Callable[[str, str], None]] = None,
        context: Any = None,
        sdk_module: Path = SDK_MODULE,
        device_python: Optional[Path] = None,
        idle_timeout_s: float = WORKER_IDLE_TIMEOUT_S,
        auto_monitor: bool = True,
        port_discovery: Callable[[], Sequence[str]] = discover_ttyusb_ports,
        port_validator: Callable[[Any], str] = validate_ttyusb_port,
        permission_granter: Callable[[Any], str] = ensure_ttyusb_permissions,
        permission_policy_check: Callable[[Any], Optional[str]] = (
            ttyusb_permission_policy_error
        ),
        permission_lease_check: Optional[Callable[..., bool]] = None,
        permission_lease_revoke: Optional[Callable[[], None]] = None,
        permission_lease_verify: Optional[Callable[[Any], str]] = None,
        leased_permission_policy_check: Optional[
            Callable[[Any], Optional[str]]
        ] = None,
        calibration_writer: Callable[
            [Sequence[tuple[GripperProfile, Mapping[str, Any]]]], None
        ] = _atomic_write_calibration_payloads,
    ) -> None:
        self._log_callback = log_callback or (lambda level, message: None)
        self._context = context or mp.get_context('spawn')
        self.sdk_module = Path(sdk_module)
        self.device_python = resolve_device_python(device_python)
        self.idle_timeout_s = max(1.0, float(idle_timeout_s))
        self._port_discovery = port_discovery
        self._port_validator = port_validator
        self._permission_granter = permission_granter
        self._permission_policy_check = permission_policy_check
        self._permission_lease_check = permission_lease_check
        self._permission_lease_revoke = permission_lease_revoke
        self._permission_lease_verify = permission_lease_verify
        self._leased_permission_policy_check = (
            leased_permission_policy_check
            or (lambda port: ttyusb_permission_policy_error(
                port, allow_world_rw=True
            ))
        )
        self._calibration_writer = calibration_writer
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._monitor_stop = threading.Event()
        self._closed = False
        self._monitor: Optional[threading.Thread] = None
        self._config_name: Optional[str] = None
        self._profiles: Dict[str, GripperProfile] = {}
        self._profile_error: Optional[str] = None
        self._workers: Dict[str, _WorkerSlot] = {}
        self._calibration: Optional[_CalibrationTransaction] = None
        self._last_calibration_completion: Optional[Dict[str, Any]] = None
        self._pending_positions: Dict[str, Dict[str, Any]] = {}
        self._last_sent_at: Dict[str, float] = {side: 0.0 for side in SIDES}
        self._side_revisions: Dict[str, int] = {side: 0 for side in SIDES}
        self._last_targets: Dict[str, float] = {}
        self._last_error: Optional[str] = None
        self._hardware_unknown = False
        self._status = 'unavailable'
        self._idempotency: Dict[str, tuple[tuple[Any, ...], Dict[str, Any]]] = {}
        self._sides: Dict[str, Dict[str, Any]] = {
            side: {
                'configured': False,
                'status': 'unavailable',
                'target_percent': None,
                'actual_percent': None,
                'actual_width_m': None,
                'last_error': None,
                'last_command_id': None,
                'last_completed_command_id': None,
                'last_completed_action': None,
                'last_probe_command_id': None,
                'last_probe_completed_command_id': None,
            }
            for side in SIDES
        }
        if auto_monitor:
            self._monitor = threading.Thread(
                target=self._monitor_loop,
                name='gripper-control-monitor',
                daemon=True,
            )
            self._monitor.start()

    def set_permission_authorization(
        self,
        *,
        lease_check: Optional[Callable[..., bool]],
        lease_revoke: Optional[Callable[[], None]],
        lease_verify: Optional[Callable[[Any], str]],
    ) -> None:
        """Attach the explicit-grant lease owned by the permission authorizer."""
        with self._lock:
            self._permission_lease_check = lease_check
            self._permission_lease_revoke = lease_revoke
            self._permission_lease_verify = lease_verify

    def _permission_lease_allows_locked(
        self, port: str, *, renew: bool
    ) -> bool:
        checker = self._permission_lease_check
        if checker is None:
            return False
        try:
            return bool(checker(port, renew=renew))
        except Exception as exc:
            self._log(
                'ERROR',
                'ttyUSB permission lease validation failed: '
                f'{type(exc).__name__}: {exc}',
            )
            return False

    def _revoke_permission_leases_locked(self) -> None:
        revoke = self._permission_lease_revoke
        if revoke is None:
            return
        try:
            revoke()
        except Exception as exc:
            self._log(
                'ERROR',
                'ttyUSB permission lease invalidation failed: '
                f'{type(exc).__name__}: {exc}',
            )

    def _log(self, level: str, message: str) -> None:
        self._log_callback(level, message)

    def _assert_open_locked(self) -> None:
        if self._closed or self._monitor_stop.is_set():
            raise GripperConflictError(
                'Gripper control manager is closed or shutting down'
            )

    def configure(self, config_name: str, config: Mapping[str, Any]) -> None:
        with self._lock:
            self._assert_open_locked()
            self._stop_locked('config_change')
            self._last_targets.clear()
            self._config_name = str(config_name)
            self._profile_error = None
            try:
                self._profiles = extract_gripper_profiles(config)
            except GripperControlError as exc:
                self._profiles = {}
                self._profile_error = str(exc)
            for side in SIDES:
                self._side_revisions[side] += 1
                configured = side in self._profiles
                self._sides[side].update(
                    {
                        'configured': configured,
                        'status': 'idle' if configured else 'unavailable',
                        'target_percent': None,
                        'actual_percent': None,
                        'actual_width_m': None,
                        'last_error': None,
                        'last_command_id': None,
                        'last_completed_command_id': None,
                        'last_completed_action': None,
                        'last_probe_command_id': None,
                        'last_probe_completed_command_id': None,
                    }
                )
            self._last_error = self._profile_error
            self._status = 'idle' if self._profiles else 'unavailable'
            self._idempotency.clear()
            self._last_calibration_completion = None

    def configured_ports(self) -> Dict[str, str]:
        with self._lock:
            return {side: profile.port for side, profile in self._profiles.items()}

    def available_ports(self) -> list[str]:
        with self._lock:
            return self._available_ports_locked()

    def _available_ports_locked(self) -> list[str]:
        try:
            raw_ports = self._port_discovery()
        except BaseException as exc:
            self._log('ERROR', f'ttyUSB discovery failed: {type(exc).__name__}: {exc}')
            return []
        ports: set[str] = set()
        for raw_port in raw_ports:
            try:
                ports.add(self._port_validator(raw_port))
            except GripperRequestError:
                continue
        return sorted(
            ports,
            key=lambda value: (int(Path(value).name.removeprefix('ttyUSB')), value),
        )

    def bind_ports(self, bindings: Mapping[str, Any]) -> Dict[str, str]:
        """Atomically apply session-only side-to-ttyUSB bindings while idle."""
        if not isinstance(bindings, Mapping) or not bindings:
            raise GripperRequestError('bindings must contain at least one gripper side')
        unexpected = sorted(str(side) for side in bindings if str(side) not in SIDES)
        if unexpected:
            raise GripperRequestError(
                'Unexpected gripper binding sides: ' + ', '.join(unexpected)
            )
        with self._lock:
            self._assert_open_locked()
            self._refresh_locked()
            if self.active():
                raise GripperConflictError(
                    'Stop the active gripper command before changing serial ports'
                )
            discovered = set(self._available_ports_locked())
            candidate = dict(self._profiles)
            for raw_side, raw_port in bindings.items():
                side = str(raw_side)
                if side not in candidate:
                    raise GripperConflictError(
                        f'{side} LKTech gripper is not configured'
                    )
                port = self._port_validator(raw_port)
                if port not in discovered:
                    raise GripperRequestError(
                        f'gripper port is no longer available: {port}'
                    )
                candidate[side] = replace(candidate[side], port=port)
            used: Dict[str, str] = {}
            for side, profile in candidate.items():
                prior = used.get(profile.port)
                if prior is not None:
                    raise GripperRequestError(
                        f'{side} and {prior} grippers cannot use the same serial port'
                    )
                used[profile.port] = side
            self._revoke_permission_leases_locked()
            self._profiles = candidate
            for side in bindings:
                self._side_revisions[str(side)] += 1
            self._last_targets.clear()
            self._idempotency.clear()
            self._last_calibration_completion = None
            self._last_error = None
            for side in SIDES:
                self._sides[side]['last_error'] = None
            for side in bindings:
                self._sides[str(side)].update(
                    {
                        'status': 'idle',
                        'target_percent': None,
                        'actual_percent': None,
                        'actual_width_m': None,
                        'last_command_id': None,
                        'last_completed_command_id': None,
                        'last_completed_action': None,
                        'last_probe_command_id': None,
                        'last_probe_completed_command_id': None,
                    }
                )
            return {side: profile.port for side, profile in self._profiles.items()}

    @staticmethod
    def _alive(process: Any) -> bool:
        if process is None:
            return False
        try:
            return bool(process.is_alive())
        except (AttributeError, AssertionError, OSError, ValueError):
            return False

    @property
    def _process(self) -> Any:
        """Compatibility view used by older diagnostics/tests."""
        for slot in self._workers.values():
            if self._alive(slot.process):
                return slot.process
        return None

    def owned_pids(self) -> list[int]:
        with self._lock:
            return sorted({
                int(slot.pid)
                for slot in self._workers.values()
                if slot.pid and self._alive(slot.process)
            })

    def owned_pid(self) -> Optional[int]:
        """Backward-compatible first owned PID; prefer :meth:`owned_pids`."""
        pids = self.owned_pids()
        return pids[0] if pids else None

    def _inflight_commands_locked(self) -> list[Dict[str, Any]]:
        return [
            slot.inflight
            for slot in self._workers.values()
            if slot.inflight is not None
        ]

    def _global_busy_locked(self) -> bool:
        return bool(
            self._hardware_unknown
            or self._calibration is not None
            or any(
                slot.lane.startswith('calibration:') and self._alive(slot.process)
                for slot in self._workers.values()
            )
            or any(
                command.get('action') == 'calibrate'
                for command in self._inflight_commands_locked()
            )
        )

    def _busy_sides_locked(self) -> set[str]:
        busy_sides = set(self._pending_positions)
        if self._calibration is not None:
            busy_sides.update(self._calibration.sides)
        for command in self._inflight_commands_locked():
            requested = command.get('side')
            busy_sides.update(SIDES if requested == 'both' else (str(requested),))
        return {side for side in busy_sides if side in SIDES}

    def active(self) -> bool:
        with self._lock:
            return bool(
                self._calibration is not None
                or any(self._alive(slot.process) for slot in self._workers.values())
                or self._inflight_commands_locked()
                or self._pending_positions
            )

    def _available(self) -> tuple[bool, Optional[str]]:
        if self._profile_error:
            return False, self._profile_error
        if not self._profiles:
            return False, 'Selected config has no enabled LKTech grippers'
        if not self.sdk_module.is_file():
            return False, 'LKTech SDK submodule is not initialized'
        if not self.device_python.is_file():
            return False, (
                'Configured gripper Python is unavailable: '
                f'{self.device_python}'
            )
        try:
            active_python = Path(sys.executable).resolve(strict=True)
            approved_python = self.device_python.resolve(strict=True)
        except OSError:
            return False, 'Cannot verify the active gripper Python profile'
        if active_python != approved_python:
            return False, (
                'Control server Python does not match configured gripper Python: '
                f'active={active_python}, configured={approved_python}'
            )
        return True, None

    def _prepare_ports_locked(self, requested: Sequence[str]) -> None:
        discovered = set(self._available_ports_locked())
        for side in requested:
            profile = self._profiles.get(side)
            if profile is None:
                raise GripperConflictError(f'{side} LKTech gripper is not configured')
            port = self._port_validator(profile.port)
            if port not in discovered:
                raise GripperConflictError(
                    f'{side} gripper serial port is no longer available: {port}'
                )
            try:
                if self._permission_lease_allows_locked(port, renew=False):
                    verifier = self._permission_lease_verify
                    if verifier is None:
                        raise GripperControlError(
                            'Explicit ttyUSB permission verification is unavailable'
                        )
                    verifier(port)
                else:
                    self._permission_granter(port)
            except GripperControlError:
                raise
            except BaseException as exc:
                raise GripperControlError(
                    f'Cannot grant read/write permission on {port}: '
                    f'{type(exc).__name__}: {exc}'
                ) from exc

    def _ensure_worker_locked(
        self,
        lane: str,
        requested: Sequence[str],
    ) -> _WorkerSlot:
        existing = self._workers.get(lane)
        if existing is not None and self._alive(existing.process):
            raise GripperConflictError(f'{lane} gripper worker is already active')
        self._prepare_ports_locked(requested)
        parent, child = self._context.Pipe(duplex=True)
        token = uuid.uuid4().hex
        worker_profiles = {
            side: copy.deepcopy(self._profiles[side])
            for side in requested
        }
        process = self._context.Process(
            target=_gripper_worker,
            args=(
                child,
                worker_profiles,
                token,
                self.idle_timeout_s,
            ),
            name=f'ControlPanelLKTechGripper-{lane}',
            daemon=False,
        )
        try:
            process.start()
        except BaseException as exc:
            if self._alive(process):
                try:
                    process.terminate()
                except (AttributeError, OSError, ValueError):
                    pass
                try:
                    process.join(timeout=STOP_TIMEOUT_S)
                except (AssertionError, OSError, ValueError):
                    pass
            if self._alive(process):
                self._hardware_unknown = True
                try:
                    child.close()
                except (AttributeError, OSError):
                    pass
                self._workers[lane] = _WorkerSlot(
                    lane=lane,
                    process=process,
                    connection=parent,
                    owner_token=token,
                    pid=int(process.pid) if process.pid else None,
                )
                raise GripperControlError(
                    f'Cannot contain partially started {lane} gripper worker; '
                    'hardware state is unknown'
                ) from exc
            try:
                parent.close()
                child.close()
            except (AttributeError, OSError):
                pass
            raise GripperControlError(
                f'Cannot start isolated LKTech gripper worker for {lane}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        try:
            child.close()
        except (AttributeError, OSError):
            pass
        slot = _WorkerSlot(
            lane=lane,
            process=process,
            connection=parent,
            owner_token=token,
            pid=int(process.pid) if process.pid else None,
        )
        self._workers[lane] = slot
        self._status = 'starting'
        self._log('INFO', f'Started isolated LKTech gripper worker for {lane}')
        return slot

    def _send_locked(self, command: Dict[str, Any]) -> None:
        if command.get('action') not in {'position', 'probe'} or command.get('side') not in SIDES:
            raise GripperControlError(
                'Only side-scoped position/probe commands use the standard worker path'
        )
        side = str(command['side'])
        slot = self._ensure_worker_locked(side, (side,))
        # A pipe send may deliver bytes and still raise. Mark the command as
        # potentially active before sending so containment never assumes that a
        # position command could not have reached hardware.
        slot.inflight = {
            **command,
            'sent_at': time.monotonic(),
            'timeout_s': (
                PROBE_TIMEOUT_S
                if command['action'] == 'probe'
                else POSITION_TIMEOUT_S
            ),
        }
        try:
            slot.connection.send(command)
        except (AttributeError, BrokenPipeError, EOFError, OSError) as exc:
            message = f'{side} LKTech gripper worker command channel failed'
            self._last_error = message
            stopped, cleanup_unknown = self._stop_slot_locked(
                slot, 'command_channel_failed'
            )
            if stopped:
                self._workers.pop(side, None)
            if not stopped or (cleanup_unknown and command['action'] != 'probe'):
                self._hardware_unknown = True
            raise GripperControlError(message) from exc
        sent_at = time.monotonic()
        self._last_sent_at[side] = sent_at
        if command['action'] == 'probe':
            self._sides[side]['status'] = 'probing'
            self._sides[side]['last_probe_command_id'] = command['command_id']
        else:
            self._sides[side]['status'] = 'moving'
            self._sides[side]['last_command_id'] = command['command_id']
        self._recompute_status_locked()

    def _contain_calibration_workers_locked(self, reason: str) -> tuple[list[str], list[str]]:
        failed_lanes: list[str] = []
        unknown_lanes: list[str] = []
        for lane, slot in tuple(self._workers.items()):
            if not lane.startswith('calibration:'):
                continue
            stopped, cleanup_unknown = self._stop_slot_locked(slot, reason)
            if stopped:
                self._workers.pop(lane, None)
            else:
                failed_lanes.append(lane)
            if cleanup_unknown:
                unknown_lanes.append(lane)
        return failed_lanes, unknown_lanes

    def _start_calibration_locked(
        self,
        command: Mapping[str, Any],
        requested: Sequence[str],
    ) -> None:
        """Spawn all lanes first, then release both calibration commands."""
        self._prepare_ports_locked(requested)
        transaction = _CalibrationTransaction(
            command_id=str(command['command_id']),
            requested_side=str(command['side']),
            sides=tuple(str(side) for side in requested),
            started_at=time.monotonic(),
            started_at_epoch=time.time(),
            results={},
            progress={str(side): 'starting' for side in requested},
            errors={},
        )
        self._calibration = transaction
        slots: Dict[str, _WorkerSlot] = {}
        try:
            # Do not send either hardware command until both process starts have
            # succeeded. This keeps a partial spawn from moving one gripper.
            for side in requested:
                lane = f'calibration:{side}'
                slots[str(side)] = self._ensure_worker_locked(lane, (str(side),))

            # From this point a child command may reach hardware. Invalidate
            # both compare-and-swap tokens exactly once for the global
            # transaction, never once per child lane.
            for side in SIDES:
                self._side_revisions[side] += 1
            sent_at = time.monotonic()
            for side in requested:
                side = str(side)
                slot = slots[side]
                child_command = {
                    'command_id': transaction.command_id,
                    'action': 'calibrate',
                    'side': side,
                    'transaction_side': transaction.requested_side,
                    'defer_commit': True,
                }
                # Treat send as ambiguous until worker cleanup proves otherwise:
                # multiprocessing pipes may deliver and then raise locally.
                slot.inflight = {
                    **child_command,
                    'sent_at': sent_at,
                    'timeout_s': CALIBRATE_ONE_TIMEOUT_S,
                }
                transaction.progress[side] = 'dispatching'
                slot.connection.send(child_command)
                transaction.progress[side] = 'dispatched'
                self._last_sent_at[side] = sent_at
            transaction.phase = 'running'
        except BaseException as exc:
            failed_lanes, unknown_lanes = self._contain_calibration_workers_locked(
                'calibration_start_failed'
            )
            self._calibration = None
            unknown = bool(
                self._hardware_unknown or failed_lanes or unknown_lanes
            )
            self._hardware_unknown = self._hardware_unknown or unknown
            message = (
                'Cannot start parallel LKTech gripper calibration: '
                f'{type(exc).__name__}: {exc}'
            )[:500]
            if unknown:
                message += '; hardware state is unknown'
            self._last_error = message
            for side in requested:
                self._sides[str(side)]['status'] = 'error'
                self._sides[str(side)]['last_error'] = message
            self._recompute_status_locked()
            if isinstance(exc, GripperControlError) and not unknown:
                raise
            raise GripperControlError(message) from exc

        self._status = 'calibrating'
        for side in requested:
            side = str(side)
            self._sides[side]['status'] = 'calibrating'
            self._sides[side]['last_error'] = None
            self._sides[side]['last_command_id'] = transaction.command_id
        self._last_error = None

    def _idempotent_locked(
        self,
        key: str,
        signature: tuple[Any, ...],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(key, str) or not (8 <= len(key) <= 128):
            raise GripperRequestError('idempotency_key must contain 8..128 characters')
        cached = self._idempotency.get(key)
        if cached is None:
            return None
        if cached[0] != signature:
            raise GripperConflictError(
                'idempotency_key was already used for a different gripper command'
            )
        result = copy.deepcopy(cached[1])
        result['duplicate'] = True
        result['accepted'] = False
        return result

    def _remember_idempotency_locked(
        self,
        key: str,
        signature: tuple[Any, ...],
        result: Mapping[str, Any],
    ) -> None:
        self._idempotency[key] = (signature, copy.deepcopy(dict(result)))
        while len(self._idempotency) > 512:
            self._idempotency.pop(next(iter(self._idempotency)))

    def replay(
        self,
        action: str,
        side: str,
        idempotency_key: str,
        target_percent: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return an existing command before controller revision validation."""
        if action == 'calibrate':
            signature: tuple[Any, ...] = ('calibrate', side)
        elif action == 'probe':
            signature = ('probe', side)
        elif action == 'position':
            try:
                target = float(target_percent)
            except (TypeError, ValueError) as exc:
                raise GripperRequestError('target_percent must be a number') from exc
            signature = ('position', side, round(target, 6))
        else:
            return None
        with self._lock:
            return self._idempotent_locked(idempotency_key, signature)

    def calibrate(self, side: str, idempotency_key: str) -> Dict[str, Any]:
        if side not in {'left', 'right', 'both'}:
            raise GripperRequestError('side must be left, right, or both')
        with self._lock:
            self._assert_open_locked()
            self._refresh_locked()
            signature = ('calibrate', side)
            duplicate = self._idempotent_locked(idempotency_key, signature)
            if duplicate is not None:
                return duplicate
            requested = SIDES if side == 'both' else (side,)
            missing = [item for item in requested if item not in self._profiles]
            if missing:
                raise GripperConflictError(
                    'Configured LKTech gripper is unavailable: ' + ', '.join(missing)
                )
            if self._global_busy_locked():
                raise GripperConflictError(
                    'Gripper calibration is blocked by calibration activity or '
                    'unknown hardware state'
                )
            if self.active():
                raise GripperConflictError('Another gripper command is active')
            # Calibration moves to mechanical limits and invalidates every
            # cached commanded target before any future deadband decision.
            self._last_targets.clear()
            command = {
                'command_id': uuid.uuid4().hex,
                'action': 'calibrate',
                'side': side,
            }
            self._start_calibration_locked(command, requested)
            result = {
                'accepted': True,
                'duplicate': False,
                'command_id': command['command_id'],
                'action': 'calibrate',
                'side': side,
                'status': 'pending',
                'side_revisions': copy.deepcopy(self._side_revisions),
            }
            self._remember_idempotency_locked(
                idempotency_key, signature, result
            )
            self._wake.set()
            return result

    def position(
        self,
        side: str,
        target_percent: float,
        idempotency_key: str,
        *,
        expected_side_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        if side not in SIDES:
            raise GripperRequestError('side must be left or right')
        try:
            target = float(target_percent)
        except (TypeError, ValueError) as exc:
            raise GripperRequestError('target_percent must be a number') from exc
        if not math.isfinite(target) or not 0.0 <= target <= 100.0:
            raise GripperRequestError('target_percent must be within [0, 100]')
        with self._lock:
            self._assert_open_locked()
            self._refresh_locked()
            signature = ('position', side, round(target, 6))
            duplicate = self._idempotent_locked(idempotency_key, signature)
            if duplicate is not None:
                return duplicate
            if side not in self._profiles:
                raise GripperConflictError(f'{side} LKTech gripper is not configured')
            if (
                not isinstance(expected_side_revision, int)
                or isinstance(expected_side_revision, bool)
                or expected_side_revision != self._side_revisions[side]
            ):
                raise GripperConflictError(
                    f'Stale {side} gripper revision {expected_side_revision}; '
                    f'current revision is {self._side_revisions[side]}'
                )
            if self._global_busy_locked():
                raise GripperConflictError(
                    'Gripper hardware state is unknown'
                    if self._hardware_unknown
                    else 'Gripper calibration is active'
                )
            self._prepare_ports_locked((side,))
            previous_target = self._last_targets.get(side)
            previous_actual = self._sides[side].get('actual_percent')
            actual_at_target = bool(
                isinstance(previous_actual, (int, float))
                and not isinstance(previous_actual, bool)
                and math.isfinite(float(previous_actual))
                and abs(float(previous_actual) - target)
                <= POSITION_SETTLE_TOLERANCE_PERCENT
            )
            command = {
                'command_id': uuid.uuid4().hex,
                'action': 'position',
                'side': side,
                'target_percent': target,
            }
            suppressed = bool(
                previous_target is not None
                and abs(previous_target - target) < POSITION_DEADBAND_PERCENT
                and actual_at_target
                and side not in self._pending_positions
                and side not in self._busy_sides_locked()
            )
            if not suppressed:
                self._side_revisions[side] += 1
                replaced = self._pending_positions.get(side)
                if replaced is not None:
                    self._log('DEBUG', f'Coalesced pending {side} gripper slider command')
                self._pending_positions[side] = command
                self._sides[side]['target_percent'] = target
                self._sides[side]['status'] = 'queued'
                self._sides[side]['last_error'] = None
                self._sides[side]['last_command_id'] = command['command_id']
                status = 'queued'
            else:
                status = 'deadband'
            result = {
                'accepted': not suppressed,
                'duplicate': False,
                'command_id': command['command_id'],
                'action': 'position',
                'side': side,
                'target_percent': target,
                'status': status,
                'side_revision': self._side_revisions[side],
            }
            self._remember_idempotency_locked(
                idempotency_key, signature, result
            )
            self._wake.set()
            return result

    def probe(
        self,
        side: str,
        idempotency_key: str,
        *,
        expected_side_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read one side's calibrated position without enabling or moving it."""
        if side not in SIDES:
            raise GripperRequestError('side must be left or right')
        with self._lock:
            self._assert_open_locked()
            self._refresh_locked()
            signature: tuple[Any, ...] = ('probe', side)
            duplicate = self._idempotent_locked(idempotency_key, signature)
            if duplicate is not None:
                return duplicate
            if side not in self._profiles:
                raise GripperConflictError(f'{side} LKTech gripper is not configured')
            if (
                not isinstance(expected_side_revision, int)
                or isinstance(expected_side_revision, bool)
                or expected_side_revision != self._side_revisions[side]
            ):
                raise GripperConflictError(
                    f'Stale {side} gripper revision {expected_side_revision}; '
                    f'current revision is {self._side_revisions[side]}'
                )
            if self._global_busy_locked():
                raise GripperConflictError(
                    'Gripper calibration or unknown hardware state blocks probing'
                )
            if side in self._busy_sides_locked():
                raise GripperConflictError(f'{side} gripper command is active')
            profile = self._profiles[side]
            _probe_effective_range(profile)
            command = {
                'command_id': uuid.uuid4().hex,
                'action': 'probe',
                'side': side,
            }
            self._send_locked(command)
            self._side_revisions[side] += 1
            self._sides[side]['last_error'] = None
            result = {
                'accepted': True,
                'duplicate': False,
                'command_id': command['command_id'],
                'action': 'probe',
                'side': side,
                'status': 'pending',
                'side_revision': self._side_revisions[side],
            }
            self._remember_idempotency_locked(
                idempotency_key, signature, result
            )
            self._wake.set()
            return result

    @staticmethod
    def _command_sides(command: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
        if not command:
            return ()
        requested = str(command.get('side', ''))
        if requested == 'both':
            return SIDES
        return (requested,) if requested in SIDES else ()

    def _recompute_status_locked(self) -> None:
        commands = self._inflight_commands_locked()
        if self._calibration is not None or any(
            command.get('action') == 'calibrate' for command in commands
        ):
            self._status = 'calibrating'
        elif self._pending_positions or any(
            command.get('action') == 'position' for command in commands
        ):
            self._status = 'moving'
        elif any(command.get('action') == 'probe' for command in commands):
            self._status = 'probing'
        elif self._hardware_unknown or self._last_error or any(
            item.get('status') == 'error' for item in self._sides.values()
        ):
            self._status = 'error'
        else:
            self._status = 'idle' if self._profiles else 'unavailable'

    def _slot_for_event_locked(self, event: Mapping[str, Any]) -> Optional[_WorkerSlot]:
        token = event.get('owner_token')
        return next(
            (slot for slot in self._workers.values() if slot.owner_token == token),
            None,
        )

    def _mark_command_failed_locked(
        self,
        slot: _WorkerSlot,
        error: str,
        *,
        hardware_unknown: bool,
    ) -> None:
        action = str((slot.inflight or {}).get('action', ''))
        requested_sides = self._command_sides(slot.inflight)
        if hardware_unknown:
            self._hardware_unknown = True
            self._last_error = error
            affected_sides = tuple(side for side in SIDES if side in self._profiles)
            self._pending_positions.clear()
            self._last_targets.clear()
        else:
            affected_sides = requested_sides
            for side in requested_sides:
                self._pending_positions.pop(side, None)
                if action != 'probe':
                    self._last_targets.pop(side, None)
            if not requested_sides:
                self._last_error = error
        for side in affected_sides:
            self._sides[side]['status'] = (
                'unavailable'
                if action == 'probe' and not hardware_unknown
                else 'error'
            )
            self._sides[side]['last_error'] = error
        slot.inflight = None
        slot.ready = False
        slot.cleanup_acknowledged = not hardware_unknown
        self._recompute_status_locked()
        self._wake.set()

    @staticmethod
    def _calibration_slot_side(slot: _WorkerSlot) -> Optional[str]:
        prefix, separator, side = slot.lane.partition(':')
        if prefix == 'calibration' and separator and side in SIDES:
            return side
        return None

    def _fail_calibration_transaction_locked(
        self,
        error: str,
        *,
        failed_side: Optional[str] = None,
        hardware_unknown: bool = False,
        reason: str = 'calibration_failed',
    ) -> None:
        transaction = self._calibration
        if transaction is None:
            if hardware_unknown:
                self._hardware_unknown = True
                self._last_error = error[:500]
            return
        message = str(error or 'LKTech gripper calibration failed')[:500]
        if failed_side in transaction.sides:
            transaction.errors[str(failed_side)] = message
            transaction.progress[str(failed_side)] = 'failed'
        transaction.phase = 'aborting'
        # Events drained while stopping a peer may add its independent cleanup
        # or failure detail, but an aborting coordinator may never finalize.
        failed_lanes, unknown_lanes = self._contain_calibration_workers_locked(reason)
        self._calibration = None
        hardware_unknown = bool(
            hardware_unknown
            or self._hardware_unknown
            or failed_lanes
            or unknown_lanes
        )
        if hardware_unknown:
            self._hardware_unknown = True
            lanes = sorted(set(failed_lanes + unknown_lanes))
            if lanes:
                message = (
                    message
                    + '; hardware state is unknown after cleanup: '
                    + ', '.join(lanes)
                )[:500]
        self._last_error = message
        self._pending_positions.clear()
        self._last_targets.clear()
        for side in transaction.sides:
            side_error = transaction.errors.get(side)
            if side_error is None and failed_side is not None:
                side_error = (
                    f'Calibration transaction aborted because {failed_side} failed: '
                    f'{message}'
                )[:500]
            self._sides[side]['status'] = 'error'
            self._sides[side]['last_error'] = side_error or message
        self._recompute_status_locked()
        self._log('ERROR', message)
        self._wake.set()

    def _complete_calibration_side_locked(
        self,
        slot: _WorkerSlot,
        event: Mapping[str, Any],
    ) -> None:
        transaction = self._calibration
        side = self._calibration_slot_side(slot)
        if transaction is None or side not in transaction.sides:
            return
        command_id = str(event.get('command_id', ''))
        if (
            slot.inflight is None
            or slot.inflight.get('command_id') != command_id
            or transaction.command_id != command_id
        ):
            return
        try:
            raw_results = event.get('results', [])
            if not isinstance(raw_results, Sequence) or isinstance(
                raw_results, (str, bytes, bytearray)
            ):
                raise GripperControlError(
                    f'Calibration result for {side} must be a list'
                )
            results = [
                result
                for result in raw_results
                if isinstance(result, Mapping) and result.get('side') == side
            ]
            if len(results) != 1 or len(raw_results) != 1:
                raise GripperControlError(
                    f'Calibration worker for {side} returned an incomplete result'
                )
            result = copy.deepcopy(dict(results[0]))
            payload = result.get('calibration')
            if not isinstance(payload, Mapping):
                raise GripperControlError(
                    f'Calibration worker for {side} returned no calibration payload'
                )
            profile = self._profiles.get(side)
            if profile is None:
                raise GripperControlError(
                    f'{side} LKTech gripper disappeared during calibration'
                )
            _profile_with_calibration(profile, payload)
            actual_width = float(result.get('actual_width_m'))
            actual_percent = float(result.get('actual_percent'))
            if (
                not math.isfinite(actual_width)
                or not math.isfinite(actual_percent)
                or actual_width < 0.0
                or not 0.0 <= actual_percent <= 100.0
            ):
                raise GripperControlError(
                    f'Calibration worker for {side} returned an invalid position'
                )
        except (GripperControlError, TypeError, ValueError, OverflowError) as exc:
            slot.inflight = None
            slot.ready = False
            slot.cleanup_acknowledged = True
            self._fail_calibration_transaction_locked(
                str(exc),
                failed_side=side,
                hardware_unknown=False,
                reason='calibration_result_invalid',
            )
            return

        slot.inflight = None
        slot.ready = False
        slot.cleanup_acknowledged = True
        transaction.results[side] = result
        transaction.progress[side] = 'awaiting_peer'
        self._sides[side].update(
            {
                'status': 'calibrating',
                'target_percent': result.get('target_percent'),
                'actual_percent': actual_percent,
                'actual_width_m': actual_width,
            }
        )
        if set(transaction.results) != set(transaction.sides):
            self._recompute_status_locked()
            self._wake.set()
            return

        transaction.phase = 'committing'
        try:
            staged_profiles = {
                item: _profile_with_calibration(
                    self._profiles[item],
                    transaction.results[item]['calibration'],
                )
                for item in transaction.sides
            }
            self._calibration_writer(
                [
                    (
                        self._profiles[item],
                        transaction.results[item]['calibration'],
                    )
                    for item in transaction.sides
                ]
            )
        except BaseException as exc:
            self._fail_calibration_transaction_locked(
                f'Calibration commit failed: {type(exc).__name__}: {exc}',
                hardware_unknown=isinstance(
                    exc, GripperCalibrationStateUnknownError
                ),
                reason='calibration_commit_failed',
            )
            return

        # One locked publication point: profiles, both side completion markers,
        # and the transaction-level marker become visible together.
        self._profiles.update(staged_profiles)
        completed_at = time.time()
        marker = {
            'command_id': transaction.command_id,
            'completed_at_epoch': completed_at,
            'sides': list(transaction.sides),
        }
        for item in transaction.sides:
            result = transaction.results[item]
            transaction.progress[item] = 'completed'
            self._sides[item].update(
                {
                    'status': 'idle',
                    'target_percent': result.get('target_percent'),
                    'actual_percent': result.get('actual_percent'),
                    'actual_width_m': result.get('actual_width_m'),
                    'last_error': None,
                    'last_completed_command_id': transaction.command_id,
                    'last_completed_action': 'calibrate',
                }
            )
        transaction.phase = 'completed'
        self._last_calibration_completion = marker
        self._calibration = None
        self._last_error = None
        self._log(
            'INFO',
            'LKTech gripper calibration transaction completed ('
            + ', '.join(transaction.sides)
            + ')',
        )
        self._recompute_status_locked()
        self._wake.set()

    def _apply_calibration_event_locked(
        self,
        slot: _WorkerSlot,
        event: Mapping[str, Any],
    ) -> None:
        name = str(event.get('event', ''))
        side = self._calibration_slot_side(slot)
        transaction = self._calibration
        aborting = transaction is not None and transaction.phase == 'aborting'
        if name == 'ready':
            slot.ready = True
            slot.pid = int(event['pid'])
            slot.pgid = int(event['pgid'])
            if transaction is not None and side in transaction.sides:
                transaction.progress[str(side)] = 'ready'
            self._recompute_status_locked()
            return
        if name == 'started':
            if transaction is not None and side in transaction.sides and not aborting:
                transaction.progress[str(side)] = 'running'
            return
        if name == 'completed':
            if transaction is None:
                command_id = str(event.get('command_id', ''))
                if (
                    slot.inflight is not None
                    and slot.inflight.get('command_id') == command_id
                ):
                    # A stop/config/shutdown invalidated the coordinator while
                    # the child was finishing. Completed still proves that this
                    # side's runtime cleanup succeeded, but must not commit or
                    # publish the now-cancelled calibration result.
                    slot.inflight = None
                    slot.ready = False
                    slot.cleanup_acknowledged = True
                return
            if aborting:
                slot.inflight = None
                slot.ready = False
                slot.cleanup_acknowledged = True
                if side in transaction.sides:
                    transaction.progress[str(side)] = 'cancelled_after_cleanup'
                return
            self._complete_calibration_side_locked(slot, event)
            return
        if name in {'failed', 'worker_failed'}:
            error = str(event.get('error') or 'LKTech gripper worker failed')[:500]
            hardware_unknown = bool(event.get('hardware_unknown'))
            slot.inflight = None
            slot.ready = False
            slot.cleanup_acknowledged = not hardware_unknown
            if aborting and side in transaction.sides:
                transaction.errors[str(side)] = error
                transaction.progress[str(side)] = 'failed'
                if hardware_unknown:
                    self._hardware_unknown = True
                    self._last_error = error
            elif transaction is not None and side in transaction.sides:
                self._fail_calibration_transaction_locked(
                    error,
                    failed_side=side,
                    hardware_unknown=hardware_unknown,
                )
            elif hardware_unknown:
                self._hardware_unknown = True
                self._last_error = error
            return
        if name == 'cleanup_completed':
            slot.cleanup_acknowledged = True
            slot.inflight = None
            slot.ready = False
            if transaction is not None and side in transaction.sides:
                transaction.progress[str(side)] = 'cleanup_completed'
            return
        if name in {'idle_timeout', 'stopped'}:
            slot.cleanup_acknowledged = True
            slot.ready = False
            if transaction is not None and side in transaction.sides:
                transaction.progress[str(side)] = 'stopped'
            return

    def _apply_event_locked(self, event: Mapping[str, Any]) -> None:
        slot = self._slot_for_event_locked(event)
        if slot is None:
            return
        if self._calibration_slot_side(slot) is not None:
            self._apply_calibration_event_locked(slot, event)
            return
        name = str(event.get('event', ''))
        if name == 'ready':
            slot.ready = True
            slot.pid = int(event['pid'])
            slot.pgid = int(event['pgid'])
            self._recompute_status_locked()
            return
        if name == 'started':
            return
        if name == 'completed':
            command_id = str(event.get('command_id', ''))
            if slot.inflight and slot.inflight.get('command_id') == command_id:
                action = str(slot.inflight['action'])
                results = [
                    result
                    for result in event.get('results', [])
                    if isinstance(result, Mapping)
                    and str(result.get('side')) in self._sides
                ]
                if action == 'probe':
                    side = str(slot.inflight.get('side', ''))
                    matching = [
                        result for result in results
                        if str(result.get('side')) == side
                    ]
                    try:
                        if len(matching) != 1:
                            raise GripperControlError(
                                f'Read-only probe for {side} returned '
                                f'{len(matching)} matching results'
                            )
                        result = matching[0]
                        actual_percent = float(result.get('actual_percent'))
                        actual_width = float(result.get('actual_width_m'))
                        if (
                            not math.isfinite(actual_percent)
                            or not 0.0 <= actual_percent <= 100.0
                            or not math.isfinite(actual_width)
                            or actual_width < 0.0
                        ):
                            raise GripperControlError(
                                f'Read-only probe for {side} returned invalid position data'
                            )
                    except (TypeError, ValueError, GripperControlError) as exc:
                        error = str(exc)[:500]
                        self._mark_command_failed_locked(
                            slot,
                            error,
                            hardware_unknown=False,
                        )
                        self._log('ERROR', f'{slot.lane}: {error}')
                        return
                    self._sides[side].update(
                        {
                            'status': 'idle',
                            'actual_percent': actual_percent,
                            'actual_width_m': actual_width,
                            'last_error': None,
                            'last_probe_completed_command_id': command_id,
                        }
                    )
                    slot.inflight = None
                    slot.ready = False
                    slot.cleanup_acknowledged = True
                    self._log('DEBUG', f'LKTech gripper probe completed ({side})')
                    self._recompute_status_locked()
                    self._wake.set()
                    return
                staged_profiles: Dict[str, GripperProfile] = {}
                if action == 'calibrate':
                    try:
                        for result in results:
                            side = str(result.get('side'))
                            profile = self._profiles.get(side)
                            if profile is None:
                                continue
                            calibration = _read_calibration(
                                Path(profile.calibration_file)
                            )
                            updated_profile = replace(
                                profile,
                                min_encoder=_optional_int(
                                    calibration.get('min_encoder')
                                ),
                                max_encoder=_optional_int(
                                    calibration.get('max_encoder')
                                ),
                                min_angle_deg=_optional_float(
                                    calibration.get('min_angle_deg')
                                ),
                                max_angle_deg=_optional_float(
                                    calibration.get('max_angle_deg')
                                ),
                            )
                            if not (
                                (
                                    updated_profile.min_encoder is not None
                                    and updated_profile.max_encoder is not None
                                )
                                or (
                                    updated_profile.min_angle_deg is not None
                                    and updated_profile.max_angle_deg is not None
                                )
                            ):
                                raise GripperControlError(
                                    f'Calibration for {side} has no usable limits'
                                )
                            staged_profiles[side] = updated_profile
                    except (
                        GripperControlError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ) as exc:
                        # Hardware cleanup already succeeded (the worker emitted
                        # completed), but unusable calibration metadata must be
                        # surfaced as a contained command failure. Never let it
                        # terminate the monitor thread or publish success markers.
                        for result in results:
                            side = str(result.get('side'))
                            self._sides[side].update(
                                {
                                    'target_percent': result.get('target_percent'),
                                    'actual_percent': result.get('actual_percent'),
                                    'actual_width_m': result.get('actual_width_m'),
                                }
                            )
                        error = str(exc)[:500]
                        self._mark_command_failed_locked(
                            slot,
                            error,
                            hardware_unknown=False,
                        )
                        self._log('ERROR', f'{slot.lane}: {error}')
                        return
                    self._profiles.update(staged_profiles)

                for result in results:
                    if not isinstance(result, Mapping):
                        continue
                    side = str(result.get('side'))
                    self._sides[side].update(
                        {
                            'status': 'queued' if side in self._pending_positions else 'idle',
                            'target_percent': result.get('target_percent'),
                            'actual_percent': result.get('actual_percent'),
                            'actual_width_m': result.get('actual_width_m'),
                            'last_error': None,
                            'last_completed_command_id': command_id,
                            'last_completed_action': action,
                        }
                    )
                    if result.get('target_percent') is not None:
                        self._last_targets[side] = float(result['target_percent'])
                slot.inflight = None
                slot.ready = False
                slot.cleanup_acknowledged = True
                self._log('INFO', f'LKTech gripper {action} completed ({slot.lane})')
                self._recompute_status_locked()
                self._wake.set()
            return
        if name in {'failed', 'worker_failed'}:
            error = str(event.get('error') or 'LKTech gripper worker failed')[:500]
            self._mark_command_failed_locked(
                slot,
                error,
                hardware_unknown=bool(event.get('hardware_unknown')),
            )
            self._log('ERROR', f'{slot.lane}: {error}')
            return
        if name == 'cleanup_completed':
            slot.cleanup_acknowledged = True
            slot.inflight = None
            slot.ready = False
            self._recompute_status_locked()
            return
        if name in {'idle_timeout', 'stopped'}:
            slot.cleanup_acknowledged = True
            slot.ready = False
            self._recompute_status_locked()

    def _drain_slot_locked(self, slot: _WorkerSlot) -> None:
        while True:
            try:
                if not slot.connection.poll(0):
                    break
                event = slot.connection.recv()
            except (AttributeError, BrokenPipeError, EOFError, OSError):
                break
            if isinstance(event, Mapping):
                self._apply_event_locked(event)

    def _drain_locked(self) -> None:
        for slot in tuple(self._workers.values()):
            self._drain_slot_locked(slot)

    def _close_slot_connection_locked(self, slot: _WorkerSlot) -> None:
        try:
            slot.connection.close()
        except (AttributeError, OSError):
            pass

    def _refresh_locked(self) -> None:
        self._drain_locked()
        now = time.monotonic()
        timed_out = next(
            (
                slot
                for slot in self._workers.values()
                if slot.inflight
                and now - float(slot.inflight['sent_at'])
                > float(slot.inflight['timeout_s'])
            ),
            None,
        )
        if timed_out is not None:
            action = str(timed_out.inflight['action'])
            error = f'{timed_out.lane} LKTech gripper {action} timed out'
            calibration_side = self._calibration_slot_side(timed_out)
            if calibration_side is not None and self._calibration is not None:
                self._fail_calibration_transaction_locked(
                    error,
                    failed_side=calibration_side,
                    hardware_unknown=False,
                    reason='calibration_timeout',
                )
                return
            if action == 'probe':
                inflight = copy.deepcopy(timed_out.inflight)
                side = str(inflight.get('side', ''))
                command_id = str(inflight.get('command_id', ''))
                stopped, cleanup_unknown = self._stop_slot_locked(
                    timed_out, 'probe_timeout'
                )
                completed = bool(
                    side in self._sides
                    and self._sides[side].get('last_probe_completed_command_id')
                    == command_id
                )
                if stopped:
                    self._workers.pop(timed_out.lane, None)
                if not completed:
                    timed_out.inflight = inflight
                    self._mark_command_failed_locked(
                        timed_out,
                        error,
                        hardware_unknown=not stopped or cleanup_unknown,
                    )
                    self._log('ERROR', f'{timed_out.lane}: {error}')
                return
            self._mark_command_failed_locked(
                timed_out,
                error,
                hardware_unknown=True,
            )
            self._stop_locked(
                'command_timeout',
                preserve_error=True,
                allow_unknown=True,
            )
            return

        for lane, slot in tuple(self._workers.items()):
            if self._alive(slot.process):
                continue
            try:
                slot.process.join(timeout=0)
            except (AssertionError, OSError, ValueError):
                pass
            if slot.inflight:
                exit_code = getattr(slot.process, 'exitcode', None)
                error = (
                    f'{lane} LKTech gripper worker exited before ACK '
                    f'(exit {exit_code})'
                )
                calibration_side = self._calibration_slot_side(slot)
                if calibration_side is not None and self._calibration is not None:
                    self._fail_calibration_transaction_locked(
                        error,
                        failed_side=calibration_side,
                        hardware_unknown=True,
                        reason='calibration_worker_exit',
                    )
                    return
                self._mark_command_failed_locked(
                    slot,
                    error,
                    hardware_unknown=str(slot.inflight.get('action')) != 'probe',
                )
                self._log('ERROR', error)
            self._close_slot_connection_locked(slot)
            if self._workers.get(lane) is slot:
                self._workers.pop(lane, None)
        self._recompute_status_locked()

    def _dispatch_locked(self) -> None:
        if self._global_busy_locked() or not self._pending_positions:
            return
        now = time.monotonic()
        for side in SIDES:
            command = self._pending_positions.get(side)
            if command is None:
                continue
            slot = self._workers.get(side)
            if slot is not None and (slot.inflight or self._alive(slot.process)):
                continue
            if now - self._last_sent_at[side] < POSITION_MIN_INTERVAL_S:
                continue
            self._pending_positions.pop(side, None)
            try:
                self._send_locked(command)
            except GripperControlError as exc:
                error = str(exc)
                self._last_targets.pop(side, None)
                self._sides[side]['status'] = 'error'
                self._sides[side]['last_error'] = error
                self._log('ERROR', f'{side}: {error}')
        self._recompute_status_locked()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.is_set():
            with self._lock:
                try:
                    self._refresh_locked()
                    self._dispatch_locked()
                except Exception as exc:
                    error = (
                        'LKTech gripper monitor failed closed: '
                        f'{type(exc).__name__}: {exc}'
                    )[:500]
                    self._last_error = error
                    self._hardware_unknown = True
                    self._log('ERROR', error)
                    try:
                        self._stop_locked(
                            'monitor_failure',
                            preserve_error=True,
                            allow_unknown=True,
                        )
                    except GripperControlError as stop_exc:
                        self._log('ERROR', str(stop_exc))
            self._wake.wait(0.05)
            self._wake.clear()

    def _stop_slot_locked(
        self,
        slot: _WorkerSlot,
        reason: str,
    ) -> tuple[bool, bool]:
        process = slot.process
        had_inflight = slot.inflight is not None
        unknown_before = self._hardware_unknown
        if self._alive(process):
            try:
                slot.connection.send({'action': 'shutdown', 'reason': reason})
            except (AttributeError, BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            try:
                process.join(timeout=STOP_TIMEOUT_S)
            except (AssertionError, OSError, ValueError):
                pass
        if self._alive(process):
            try:
                if slot.ready and slot.pid and slot.pgid == slot.pid:
                    os.killpg(slot.pgid, signal.SIGTERM)
                else:
                    process.terminate()
            except (AttributeError, ProcessLookupError, OSError, ValueError):
                pass
            try:
                process.join(timeout=STOP_TIMEOUT_S)
            except (AssertionError, OSError, ValueError):
                pass
        stopped = not self._alive(process)
        cleanup_unknown = False
        if stopped:
            # A cleanup-failed/worker-failed event may arrive immediately before
            # process exit. Consume the pipe before discarding it, otherwise a
            # dead PID could be mistaken for a safely disabled motor.
            self._drain_slot_locked(slot)
            cleanup_unknown = bool(
                (self._hardware_unknown and not unknown_before)
                or (
                    had_inflight
                    and not getattr(slot, 'cleanup_acknowledged', False)
                )
            )
            self._close_slot_connection_locked(slot)
        return stopped, cleanup_unknown

    def _stop_locked(
        self,
        reason: str,
        *,
        preserve_error: bool = False,
        allow_unknown: bool = False,
    ) -> None:
        self._revoke_permission_leases_locked()
        unknown_before = self._hardware_unknown
        # Invalidate the coordinator before draining shutdown events so a stop,
        # config switch, or server shutdown can never publish a late commit.
        self._calibration = None
        self._pending_positions.clear()
        self._last_targets.clear()
        failed_lanes = []
        unknown_lanes = []
        for lane, slot in tuple(self._workers.items()):
            stopped, cleanup_unknown = self._stop_slot_locked(slot, reason)
            if stopped:
                self._workers.pop(lane, None)
            else:
                failed_lanes.append(lane)
            if cleanup_unknown:
                unknown_lanes.append(lane)
        if failed_lanes or unknown_lanes or unknown_before or self._hardware_unknown:
            self._hardware_unknown = True
            lanes = sorted(set(failed_lanes + unknown_lanes))
            prior_error = self._last_error
            self._last_error = (
                'LKTech gripper hardware state is unknown after stop'
                + (': ' + ', '.join(lanes) if lanes else '')
                + (f'; {prior_error}' if prior_error else '')
            )
            self._recompute_status_locked()
            if not allow_unknown:
                raise GripperControlError(self._last_error)
            return
        for side in SIDES:
            if self._sides[side]['configured']:
                self._sides[side]['status'] = (
                    'error'
                    if preserve_error and self._sides[side].get('last_error')
                    else 'idle'
                )
        if not preserve_error:
            self._last_error = None
            self._hardware_unknown = False
            for side in SIDES:
                self._sides[side]['last_error'] = None
        self._recompute_status_locked()

    def stop(self, reason: str = 'requested') -> Dict[str, Any]:
        with self._lock:
            was_active = self.active()
            self._stop_locked(reason)
            return {'ok': True, 'stopped': was_active, 'status': self._status}

    def force_stop(self, reason: str = 'force_stop_all') -> Dict[str, Any]:
        """SIGKILL panel-owned short-command workers without cleanup ACKs."""
        with self._lock:
            self._revoke_permission_leases_locked()
            self._calibration = None
            self._pending_positions.clear()
            self._last_targets.clear()
            killed_pids: list[int] = []
            remaining: list[str] = []
            had_workers = bool(self._workers)
            for lane, slot in tuple(self._workers.items()):
                process = slot.process
                pid = slot.pid
                if self._alive(process):
                    try:
                        if slot.ready and pid and slot.pgid == pid:
                            os.killpg(slot.pgid, signal.SIGKILL)
                        else:
                            process.kill()
                    except (
                        AttributeError,
                        ProcessLookupError,
                        OSError,
                        ValueError,
                    ):
                        pass
                    try:
                        process.join(timeout=2.0)
                    except (AssertionError, OSError, ValueError):
                        pass
                if self._alive(process):
                    remaining.append(lane)
                    continue
                if pid is not None:
                    killed_pids.append(int(pid))
                self._close_slot_connection_locked(slot)
                self._workers.pop(lane, None)

            if had_workers:
                self._hardware_unknown = True
                self._last_error = (
                    'Force-stopped gripper command workers; hardware state is unknown'
                    + (': ' + ', '.join(sorted(remaining)) if remaining else '')
                )
                for side in SIDES:
                    if self._sides[side]['configured']:
                        self._sides[side]['status'] = 'error'
                        self._sides[side]['last_error'] = self._last_error
                self._log('WARNING', f'{reason}: {self._last_error}')
            self._recompute_status_locked()
            self._wake.set()
            return {
                'ok': not remaining,
                'killed_pids': sorted(set(killed_pids)),
                'remaining': sorted(remaining),
                'hardware_unknown': self._hardware_unknown,
                'status': self._status,
            }

    def state(self, blockers: Sequence[str] = ()) -> Dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            available_ports = self._available_ports_locked()
            available_port_set = set(available_ports)
            available, availability_error = self._available()
            all_blockers = list(dict.fromkeys(
                [str(item) for item in blockers if item]
                + ([availability_error] if availability_error else [])
                + (
                    ['LKTech hardware state is unknown; stop and inspect before retrying']
                    if self._hardware_unknown
                    else []
                )
            ))
            active = self.active()
            busy_sides = self._busy_sides_locked()
            global_busy = self._global_busy_locked()
            busy = bool(global_busy or busy_sides)
            sides = {}
            for side in SIDES:
                item = copy.deepcopy(self._sides[side])
                profile = self._profiles.get(side)
                side_busy = side in busy_sides
                item['port'] = profile.port if profile else None
                item['port_present'] = bool(
                    profile and profile.port in available_port_set
                )
                item['max_width_m'] = profile.max_width_m if profile else None
                item['motor_id'] = profile.motor_id if profile else None
                artifact = (
                    _calibration_artifact_state(profile)
                    if profile is not None
                    else None
                )
                item['calibration_artifact'] = artifact
                item['calibrated'] = bool(artifact and artifact.get('valid'))
                item['move_speed_dps'] = profile.speed_dps if profile else None
                item['move_speed_source'] = profile.speed_source if profile else None
                item['hardware_unknown'] = self._hardware_unknown
                item['revision'] = self._side_revisions[side]
                item['busy'] = side_busy
                base_side_blockers = list(all_blockers)
                if profile is None:
                    base_side_blockers.append(
                        f'{side} LKTech gripper is not configured'
                    )
                elif profile.port not in available_port_set:
                    base_side_blockers.append(
                        f'{side} gripper serial port is not connected: {profile.port}'
                    )
                else:
                    try:
                        if self._permission_lease_allows_locked(
                            profile.port, renew=False
                        ):
                            permission_error = (
                                self._leased_permission_policy_check(
                                    profile.port
                                )
                            )
                        else:
                            permission_error = self._permission_policy_check(
                                profile.port
                            )
                    except GripperControlError as exc:
                        permission_error = str(exc)
                    if permission_error:
                        base_side_blockers.append(
                            f'{side} gripper: {permission_error}'
                        )

                calibration_blockers = list(base_side_blockers)
                if active:
                    calibration_blockers.append(
                        'Stop active gripper commands before calibration'
                    )

                position_blockers = list(base_side_blockers)
                if profile is not None and not item['calibrated']:
                    artifact_status = str(
                        (artifact or {}).get('status') or 'invalid'
                    )
                    if artifact_status == 'missing':
                        position_blockers.append(
                            f'{side} LKTech calibration artifact is missing'
                        )
                    elif artifact_status == 'stale':
                        position_blockers.append(
                            f'{side} LKTech calibration identity is stale'
                        )
                    elif artifact_status == 'invalid':
                        position_blockers.append(
                            f'{side} LKTech calibration artifact is invalid'
                        )
                    elif artifact_status == 'unverified':
                        position_blockers.append(
                            f'{side} LKTech calibration identity is unverified'
                        )
                    else:
                        position_blockers.append(
                            f'{side} LKTech gripper requires calibration'
                        )
                if item.get('status') == 'unavailable' and item.get('last_error'):
                    position_blockers.append(
                        f'{side} gripper position readback is unavailable; '
                        'refresh the gripper state before moving'
                    )
                if global_busy:
                    position_blockers.append(
                        'Gripper calibration or global safety interlock is active'
                    )
                elif side_busy:
                    position_blockers.append(f'{side} gripper command is active')
                # Slider updates are latest-target commands.  The manager can
                # safely coalesce one new target per side while an earlier
                # position/probe worker is finishing, so expose a separate
                # capability that ignores only the side-local busy blocker.
                # Global calibration/unknown-state interlocks still fail closed.
                queue_position_blockers = list(base_side_blockers)
                if profile is not None and not item['calibrated']:
                    queue_position_blockers.extend(
                        position_blockers[len(base_side_blockers):]
                    )
                if item.get('status') == 'unavailable' and item.get('last_error'):
                    queue_position_blockers.append(
                        f'{side} gripper position readback is unavailable; '
                        'refresh the gripper state before moving'
                    )
                if global_busy:
                    queue_position_blockers.append(
                        'Gripper calibration or global safety interlock is active'
                    )
                probe_blockers = list(base_side_blockers)
                if profile is not None and not item['calibrated']:
                    probe_blockers.extend(
                        position_blockers[len(base_side_blockers):]
                    )
                if global_busy:
                    probe_blockers.append(
                        'Gripper calibration or global safety interlock is active'
                    )
                elif side_busy:
                    probe_blockers.append(f'{side} gripper command is active')
                calibration_blockers = list(dict.fromkeys(calibration_blockers))
                position_blockers = list(dict.fromkeys(position_blockers))
                queue_position_blockers = list(dict.fromkeys(
                    queue_position_blockers
                ))
                probe_blockers = list(dict.fromkeys(probe_blockers))
                item['blockers'] = position_blockers
                item['calibration_blockers'] = calibration_blockers
                item['position_blockers'] = position_blockers
                item['queue_position_blockers'] = queue_position_blockers
                item['probe_blockers'] = probe_blockers
                item['can_calibrate'] = (
                    available
                    and not calibration_blockers
                    and profile is not None
                )
                item['can_position'] = (
                    available
                    and not position_blockers
                    and not global_busy
                    and not side_busy
                )
                item['can_queue_position'] = (
                    available
                    and not queue_position_blockers
                    and not global_busy
                )
                item['can_probe'] = (
                    available
                    and not probe_blockers
                    and not global_busy
                    and not side_busy
                )
                item['capabilities'] = {
                    'calibrate': {
                        'can': item['can_calibrate'],
                        'blockers': calibration_blockers,
                    },
                    'position': {
                        'can': item['can_position'],
                        'blockers': position_blockers,
                    },
                    'queue_position': {
                        'can': item['can_queue_position'],
                        'blockers': queue_position_blockers,
                    },
                    'probe': {
                        'can': item['can_probe'],
                        'blockers': probe_blockers,
                    },
                }
                sides[side] = item
            workers = {
                lane: {
                    'pid': slot.pid if self._alive(slot.process) else None,
                    'ready': slot.ready,
                    'side': slot.inflight.get('side') if slot.inflight else None,
                    'action': slot.inflight.get('action') if slot.inflight else None,
                }
                for lane, slot in sorted(self._workers.items())
            }
            worker_pids = sorted(
                item['pid'] for item in workers.values() if item['pid'] is not None
            )
            transaction = self._calibration
            calibration_state = {
                'active': transaction is not None,
                'command_id': transaction.command_id if transaction else None,
                'requested_side': (
                    transaction.requested_side if transaction else None
                ),
                'phase': transaction.phase if transaction else 'idle',
                'started_at_epoch': (
                    transaction.started_at_epoch if transaction else None
                ),
                'progress': (
                    copy.deepcopy(transaction.progress) if transaction else {}
                ),
                'errors': copy.deepcopy(transaction.errors) if transaction else {},
                'completed_sides': (
                    sorted(transaction.results) if transaction else []
                ),
                'last_completion': copy.deepcopy(
                    self._last_calibration_completion
                ),
            }
            calibration_blockers = list(dict.fromkeys(
                blocker
                for item in sides.values()
                for blocker in item['calibration_blockers']
            ))
            return {
                'config_name': self._config_name,
                'available': available,
                'status': self._status,
                'active': active,
                'busy': busy,
                'global_busy': global_busy,
                'hardware_unknown': self._hardware_unknown,
                'busy_sides': sorted(busy_sides),
                'pending_sides': sorted(self._pending_positions),
                'side_revisions': copy.deepcopy(self._side_revisions),
                'can_calibrate': available and not all_blockers and not active and all(
                    sides[side]['can_calibrate'] for side in SIDES
                ),
                'can_position': available and not all_blockers and any(
                    item['can_position'] for item in sides.values()
                ),
                'can_queue_position': available and not all_blockers and any(
                    item['can_queue_position'] for item in sides.values()
                ),
                'can_probe': available and not all_blockers and any(
                    item['can_probe'] for item in sides.values()
                ),
                'blockers': all_blockers,
                'calibration_blockers': calibration_blockers,
                'last_error': self._last_error,
                'calibration': calibration_state,
                'worker': {
                    'pid': worker_pids[0] if len(worker_pids) == 1 else None,
                    'pids': worker_pids,
                    'ready': bool(workers) and all(
                        item['ready'] for item in workers.values()
                    ),
                    'lanes': workers,
                    'idle_timeout_s': self.idle_timeout_s,
                    'session_mode': 'short_command_per_side_parallel_calibration',
                },
                'transport': {
                    'mode': 'direct_serial',
                    'uri': None,
                },
                'sides': sides,
                'rate_limit_hz': 1.0 / POSITION_MIN_INTERVAL_S,
                'deadband_percent': POSITION_DEADBAND_PERCENT,
                'settle_tolerance_percent': POSITION_SETTLE_TOLERANCE_PERCENT,
                'available_ports': available_ports,
                'percent_semantics': {'closed': 0.0, 'open': 100.0},
            }

    def close(self) -> None:
        self._monitor_stop.set()
        self._wake.set()
        close_error: Optional[GripperControlError] = None
        with self._lock:
            self._closed = True
            self._stop_locked(
                'server_shutdown',
                preserve_error=True,
                allow_unknown=True,
            )
            live_lanes = sorted(
                lane
                for lane, slot in self._workers.items()
                if self._alive(slot.process)
            )
            if live_lanes:
                self._hardware_unknown = True
                self._last_error = (
                    'LKTech gripper worker did not stop; hardware state is '
                    'unknown: ' + ', '.join(live_lanes)
                )
                self._recompute_status_locked()
                close_error = GripperControlError(self._last_error)
        if self._monitor is not None and self._monitor is not threading.current_thread():
            self._monitor.join(timeout=1.0)
        if close_error is not None:
            raise close_error


__all__ = [
    'GripperCalibrationStateUnknownError',
    'GripperConflictError',
    'GripperControlError',
    'GripperControlManager',
    'GripperProfile',
    'GripperRequestError',
    'discover_ttyusb_ports',
    'ensure_ttyusb_permissions',
    'extract_gripper_profiles',
    'gripper_lock_path',
    'rizon_shared_memory_names',
    'serial_port_holders',
    'serial_port_holders_many',
    'ttyusb_permission_policy_error',
    'validate_ttyusb_port',
]
