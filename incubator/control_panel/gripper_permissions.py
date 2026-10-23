"""Strict, short-lived ttyUSB permission repair for gripper actions.

The control panel normally relies on a udev rule or membership in the serial
device group. This module adds one bounded recovery path when those normal
permissions are missing:

* a password-fed ``sudo chmod 0666`` operation over already-opened ttyUSB
  descriptors; the web layer decides which transports may expose it.

Passwords are accepted only as mutable byte buffers, written to stdin, and
zeroed in ``finally`` blocks.  They are never placed in argv, environment
variables, exception messages, responses, or persistent state.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Sequence
import uuid


TTYUSB_NAME_PATTERN = re.compile(r'ttyUSB([0-9]+)\Z')
SUDO_PATH = '/usr/bin/sudo'
CHMOD_PATH = '/bin/chmod'
DEFAULT_TIMEOUT_S = 4.0
DEFAULT_RATE_WINDOW_S = 60.0
DEFAULT_RATE_ATTEMPTS = 3
DEFAULT_GLOBAL_RATE_ATTEMPTS = 6
DEFAULT_LEASE_TTL_S = 120.0
DEFAULT_PENDING_LEASE_TTL_S = 10.0
MAX_RATE_CLIENTS = 64
MAX_PASSWORD_BYTES = 512


class GripperPermissionError(RuntimeError):
    """A sanitized permission-repair failure suitable for an API response."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int,
        retry_after: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.status = int(status)
        self.retry_after = retry_after
        self.details = dict(details or {})


class GripperPermissionRequiredError(GripperPermissionError):
    """Raised before a hardware action when its ttyUSB node is inaccessible."""

    def __init__(self, ports: Sequence[str]) -> None:
        super().__init__(
            'ttyusb_permission_required',
            '夹爪串口缺少读写权限，请先完成安全授权。',
            status=409,
            details={'affected_ports': sorted(set(map(str, ports)))},
        )


@dataclass(frozen=True)
class PortIdentity:
    path: str
    st_dev: int
    st_ino: int
    st_rdev: int


@dataclass
class OpenPortSnapshot:
    identity: PortIdentity
    descriptor: int

    @property
    def proc_path(self) -> str:
        return f'/proc/{os.getpid()}/fd/{self.descriptor}'


@dataclass(frozen=True)
class _PermissionLease:
    identity: PortIdentity
    expires_at: float
    controller_revision: int
    context_ports: tuple[str, ...]


@dataclass(frozen=True)
class _PendingPermissionLease:
    identities: tuple[PortIdentity, ...]
    expires_at: float


def _identity(path: str, metadata: os.stat_result) -> PortIdentity:
    return PortIdentity(
        path=path,
        st_dev=int(metadata.st_dev),
        st_ino=int(metadata.st_ino),
        st_rdev=int(metadata.st_rdev),
    )


def _same_identity(metadata: os.stat_result, identity: PortIdentity) -> bool:
    return bool(
        stat.S_ISCHR(metadata.st_mode)
        and int(metadata.st_dev) == identity.st_dev
        and int(metadata.st_ino) == identity.st_ino
        and int(metadata.st_rdev) == identity.st_rdev
    )


def discover_strict_ttyusb_ports(dev_root: Path = Path('/dev')) -> list[str]:
    """List direct, non-symlink ttyUSB character devices in numeric order."""
    root = Path(dev_root)
    try:
        candidates = list(root.iterdir())
    except OSError:
        return []
    ports: list[tuple[int, str]] = []
    for candidate in candidates:
        match = TTYUSB_NAME_PATTERN.fullmatch(candidate.name)
        if match is None or candidate.parent != root:
            continue
        try:
            metadata = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISCHR(metadata.st_mode):
            continue
        ports.append((int(match.group(1)), str(candidate)))
    return [path for _index, path in sorted(ports, key=lambda item: item)]


def _read_write_access(path: str) -> bool:
    try:
        return bool(
            os.access(path, os.R_OK | os.W_OK, effective_ids=True)
        )
    except TypeError:  # pragma: no cover - only for non-POSIX test hosts
        return bool(os.access(path, os.R_OK | os.W_OK))


def _service_has_owner_or_group(metadata: os.stat_result) -> bool:
    """Return whether the service principal may rely on owner/group mode bits."""
    try:
        effective_uid = int(os.geteuid())
        effective_gid = int(os.getegid())
        groups = {int(group) for group in os.getgroups()}
    except (AttributeError, OSError):  # pragma: no cover - non-POSIX host
        return False
    return bool(
        int(metadata.st_uid) == effective_uid
        or int(metadata.st_gid) == effective_gid
        or int(metadata.st_gid) in groups
    )


def _world_rw_policy_enabled() -> bool:
    return os.environ.get(
        'SUPERINFERENCE_GRIPPER_ALLOW_WORLD_RW', ''
    ).strip().casefold() in {'1', 'true', 'yes', 'on'}


def _zero(buffer: Optional[bytearray]) -> None:
    if buffer is None:
        return
    for index in range(len(buffer)):
        buffer[index] = 0


class TtyUsbPermissionAuthorizer:
    """Serialize, rate-limit, and verify privileged ttyUSB chmod operations."""

    def __init__(
        self,
        *,
        port_discovery: Callable[[], Sequence[str]] = discover_strict_ttyusb_ports,
        access_check: Callable[[str], bool] = _read_write_access,
        process_factory: Callable[..., Any] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        ownership_check: Callable[[os.stat_result], bool] = (
            _service_has_owner_or_group
        ),
        revision_provider: Optional[Callable[[], int]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        rate_window_s: float = DEFAULT_RATE_WINDOW_S,
        rate_attempts: int = DEFAULT_RATE_ATTEMPTS,
        global_rate_attempts: int = DEFAULT_GLOBAL_RATE_ATTEMPTS,
        lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
        pending_lease_ttl_s: float = DEFAULT_PENDING_LEASE_TTL_S,
    ) -> None:
        self._port_discovery = port_discovery
        self._access_check = access_check
        self._process_factory = process_factory
        self._monotonic = monotonic
        self._ownership_check = ownership_check
        self._revision_provider = revision_provider
        self._timeout_s = max(0.1, float(timeout_s))
        self._rate_window_s = max(1.0, float(rate_window_s))
        self._rate_attempts = max(1, int(rate_attempts))
        self._global_rate_attempts = max(1, int(global_rate_attempts))
        self._lease_ttl_s = max(1.0, float(lease_ttl_s))
        self._pending_lease_ttl_s = max(
            1.0, float(pending_lease_ttl_s)
        )
        self._operation_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._lease_lock = threading.Lock()
        self._attempts: OrderedDict[str, Deque[float]] = OrderedDict()
        self._global_attempts: Deque[float] = deque()
        self._leases: Dict[str, _PermissionLease] = {}
        self._pending_leases: Dict[str, _PendingPermissionLease] = {}
        # Persists for this authorizer lifetime even when active leases are
        # revoked. A replacement inode must be explicitly committed before it
        # can replace an already-trusted identity.
        self._trusted_identities: Dict[str, PortIdentity] = {}
        # A chmod result is not trusted until the controller completes the
        # second-phase commit. Failed/discarded contexts remain tainted for
        # this process lifetime and require a later explicit successful commit.
        self._uncommitted_identities: Dict[str, PortIdentity] = {}

    def set_revision_provider(
        self, provider: Optional[Callable[[], int]]
    ) -> None:
        with self._lease_lock:
            self._revision_provider = provider
            self._leases.clear()
            self._pending_leases.clear()

    def _purge_expired_leases_locked(self, now: float) -> None:
        for port, lease in list(self._leases.items()):
            if lease.expires_at <= now:
                self._leases.pop(port, None)
        for token, pending in list(self._pending_leases.items()):
            if pending.expires_at <= now:
                self._pending_leases.pop(token, None)

    def _controller_revision_locked(
        self, controller_revision: Optional[int]
    ) -> tuple[bool, int]:
        if controller_revision is not None:
            try:
                return True, int(controller_revision)
            except (TypeError, ValueError):
                return False, 0
        if self._revision_provider is None:
            return True, 0
        try:
            return True, int(self._revision_provider())
        except Exception:
            return False, 0

    def _path_has_pending_authorization_locked(self, path: str) -> bool:
        return any(
            identity.path == path
            for pending in self._pending_leases.values()
            for identity in pending.identities
        )

    def _adopt_world_rw_lease_locked(
        self,
        port: str,
        metadata: os.stat_result,
        *,
        now: float,
        controller_revision: int,
        context_ports: Sequence[str] = (),
    ) -> bool:
        """Adopt an already-safe direct character node without privilege use."""
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISCHR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o006 != 0o006
            or not self._access_check(port)
        ):
            return False
        identity = _identity(port, metadata)
        if port in self._uncommitted_identities:
            return False
        trusted = self._trusted_identities.get(port)
        if trusted is not None and trusted != identity:
            return False
        if trusted is None:
            self._trusted_identities[port] = identity
        self._leases[port] = _PermissionLease(
            identity=identity,
            expires_at=now + self._lease_ttl_s,
            controller_revision=int(controller_revision),
            context_ports=tuple(context_ports) or (port,),
        )
        return True

    @staticmethod
    def _context_id(identities: Sequence[PortIdentity]) -> Optional[str]:
        if not identities:
            return None
        digest = hashlib.sha256()
        digest.update(b'ttyusb-permission-context-v1\0')
        for identity in sorted(
            identities,
            key=lambda item: (
                item.path, item.st_dev, item.st_ino, item.st_rdev
            ),
        ):
            for value in (
                identity.path,
                identity.st_dev,
                identity.st_ino,
                identity.st_rdev,
            ):
                encoded = str(value).encode('utf-8')
                digest.update(len(encoded).to_bytes(4, 'big'))
                digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _safe_direct_port(port: Any) -> Optional[str]:
        value = str(port)
        path = Path(value)
        if (
            TTYUSB_NAME_PATTERN.fullmatch(path.name) is None
            or path.parent != Path('/dev')
            or value != f'/dev/{path.name}'
        ):
            return None
        return value

    def lease_allows(
        self,
        port: Any,
        *,
        renew: bool = False,
        controller_revision: Optional[int] = None,
    ) -> bool:
        """Validate or safely adopt a world-RW direct ttyUSB character node."""
        safe_port = self._safe_direct_port(port)
        if safe_port is None:
            return False
        now = self._monotonic()
        with self._lease_lock:
            self._purge_expired_leases_locked(now)
            revision_ok, current_revision = self._controller_revision_locked(
                controller_revision
            )
            if not revision_ok:
                self._leases.pop(safe_port, None)
                return False
            try:
                metadata = os.lstat(safe_port)
            except OSError:
                self._leases.pop(safe_port, None)
                return False
            lease = self._leases.get(safe_port)
            if lease is not None and (
                current_revision != lease.controller_revision
                or stat.S_ISLNK(metadata.st_mode)
                or not _same_identity(metadata, lease.identity)
                or stat.S_IMODE(metadata.st_mode) & 0o006 != 0o006
                or not self._access_check(safe_port)
            ):
                self._leases.pop(safe_port, None)
                lease = None
            if lease is not None:
                if renew:
                    self._leases[safe_port] = _PermissionLease(
                        identity=lease.identity,
                        expires_at=now + self._lease_ttl_s,
                        controller_revision=lease.controller_revision,
                        context_ports=lease.context_ports,
                    )
                return True

            # A password flow stages chmod results before the controller
            # commits its binding revision. Do not auto-adopt that path in
            # this window: pending lookup and adoption share the lease lock.
            if self._path_has_pending_authorization_locked(safe_port):
                return False
            return self._adopt_world_rw_lease_locked(
                safe_port,
                metadata,
                now=now,
                controller_revision=current_revision,
            )

    def verify_lease_for_connection(self, port: Any) -> str:
        """Read-only pre-connect revalidation; it never chmods or invokes sudo."""
        safe_port = self._safe_direct_port(port)
        if safe_port is None or not self.lease_allows(safe_port, renew=True):
            raise GripperPermissionError(
                'ttyusb_permission_required',
                'The explicit ttyUSB permission authorization is no longer valid.',
                status=409,
                details={'affected_ports': [str(port)]},
            )
        return safe_port

    def permission_required(self, port: Any) -> bool:
        """Apply the legacy policy plus an exact explicit-authorization lease."""
        safe_port = self._safe_direct_port(port)
        if safe_port is None:
            return True
        try:
            metadata = os.lstat(safe_port)
        except OSError:
            return True
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISCHR(metadata.st_mode)
            or not self._access_check(safe_port)
        ):
            return True
        if self._ownership_check(metadata) or _world_rw_policy_enabled():
            return False
        return not self.lease_allows(safe_port)

    def _stage_authorization(
        self, identities: Sequence[PortIdentity]
    ) -> str:
        token = uuid.uuid4().hex
        now = self._monotonic()
        with self._lease_lock:
            self._purge_expired_leases_locked(now)
            staged = tuple(identities)
            self._pending_leases[token] = _PendingPermissionLease(
                identities=staged,
                expires_at=now + self._pending_lease_ttl_s,
            )
            for identity in staged:
                self._uncommitted_identities[identity.path] = identity
        return token

    def commit_authorization(
        self,
        token: str,
        expected_ports: Sequence[str],
        *,
        controller_revision: int,
    ) -> Dict[str, Any]:
        """Commit a verified grant only after the controller context is stable."""
        now = self._monotonic()
        with self._lease_lock:
            self._purge_expired_leases_locked(now)
            pending = self._pending_leases.pop(str(token), None)
            if pending is None:
                raise GripperPermissionError(
                    'permission_context_changed',
                    'The verified ttyUSB authorization context has expired; retry.',
                    status=409,
                )
            submitted = sorted(set(map(str, expected_ports)))
            staged = [identity.path for identity in pending.identities]
            if submitted != staged:
                raise GripperPermissionError(
                    'permission_context_changed',
                    'The configured ttyUSB authorization context changed; retry.',
                    status=409,
                )
            if self._revision_provider is not None:
                try:
                    current_revision = int(self._revision_provider())
                except Exception:
                    current_revision = -1
                if current_revision != int(controller_revision):
                    raise GripperPermissionError(
                        'permission_context_changed',
                        'The controller revision changed before authorization commit.',
                        status=409,
                    )

            explicit: list[PortIdentity] = []
            for identity in pending.identities:
                try:
                    metadata = os.lstat(identity.path)
                except OSError:
                    raise GripperPermissionError(
                        'ttyusb_disappeared',
                        'The ttyUSB device disappeared before authorization commit.',
                        status=409,
                    ) from None
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not _same_identity(metadata, identity)
                ):
                    raise GripperPermissionError(
                        'ttyusb_changed',
                        'The ttyUSB device changed before authorization commit.',
                        status=409,
                    )
                if not self._access_check(identity.path):
                    raise GripperPermissionError(
                        'ttyusb_permission_not_applied',
                        'The ttyUSB read/write permission is not available.',
                        status=409,
                    )
                if self._ownership_check(metadata) or _world_rw_policy_enabled():
                    continue
                if stat.S_IMODE(metadata.st_mode) & 0o006 != 0o006:
                    raise GripperPermissionError(
                        'ttyusb_permission_not_applied',
                        'The explicit ttyUSB permission grant is not present.',
                        status=409,
                    )
                explicit.append(identity)

            for identity in pending.identities:
                self._leases.pop(identity.path, None)
                self._uncommitted_identities.pop(identity.path, None)
                self._trusted_identities[identity.path] = identity
            for identity in explicit:
                self._leases[identity.path] = _PermissionLease(
                    identity=identity,
                    expires_at=now + self._lease_ttl_s,
                    controller_revision=int(controller_revision),
                    context_ports=tuple(submitted),
                )
        return self.status(expected_ports)

    def discard_authorization(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lease_lock:
            self._pending_leases.pop(str(token), None)

    def revoke_leases(self, ports: Sequence[str] = ()) -> None:
        """Invalidate grants on revision/config/binding/stop transitions."""
        targets = {str(port) for port in ports if port}
        with self._lease_lock:
            if targets:
                for port in targets:
                    self._leases.pop(port, None)
                for token, pending in list(self._pending_leases.items()):
                    if any(
                        identity.path in targets
                        for identity in pending.identities
                    ):
                        self._pending_leases.pop(token, None)
            else:
                self._leases.clear()
                self._pending_leases.clear()

    def status(self, allowed_ports: Sequence[str] = ()) -> Dict[str, Any]:
        ports = self._validated_targets(allowed_ports, allow_empty=True)
        for _attempt in range(2):
            identities = self._current_identities(ports)
            affected = [
                port for port in ports if self.permission_required(port)
            ]
            if self._current_identities(ports) == identities:
                return {
                    'required': bool(affected),
                    'target_ports': ports,
                    'affected_ports': affected,
                    'context_id': self._context_id(identities),
                }
        raise GripperPermissionError(
            'ttyusb_changed',
            'The ttyUSB identity changed while reading permission status.',
            status=409,
        )

    def required_for(self, ports: Sequence[str]) -> list[str]:
        status = self.status(ports)
        affected = set(status['affected_ports'])
        return sorted({str(port) for port in ports if str(port) in affected})

    def authorize_with_password(
        self,
        secret: bytearray,
        client_key: str,
        allowed_ports: Sequence[str],
    ) -> Dict[str, Any]:
        """Use sudo stdin once to chmod the validated open ttyUSB descriptors."""
        if not isinstance(secret, bytearray):
            raise TypeError('secret must be a mutable bytearray')
        if (
            not secret
            or len(secret) > MAX_PASSWORD_BYTES
            or 0 in secret
            or 10 in secret
            or 13 in secret
        ):
            _zero(secret)
            raise GripperPermissionError(
                'invalid_sudo_password',
                'sudo 密码格式无效。',
                status=422,
            )
        if not self._operation_lock.acquire(blocking=False):
            _zero(secret)
            raise GripperPermissionError(
                'permission_repair_busy',
                '已有串口权限授权正在进行，请稍后重试。',
                status=429,
                retry_after=1,
            )
        snapshots: list[OpenPortSnapshot] = []
        try:
            self._consume_rate_limit(client_key)
            ports = self._validated_targets(allowed_ports)
            affected = [port for port in ports if self.permission_required(port)]
            if not affected:
                return self._success_payload(ports, method='already_accessible')
            inaccessible = [
                port for port in affected if not self._access_check(port)
            ]
            if not inaccessible:
                token = self._stage_authorization(
                    self._current_identities(ports)
                )
                return self._success_payload(
                    ports,
                    method='existing_explicit_permission',
                    lease_token=token,
                )
            snapshots = self._open_snapshots(ports)
            # The proc-fd paths refer to the already-opened O_PATH handles, so
            # a ttyUSB unplug/replacement cannot redirect chmod to a new
            # filesystem object between validation and privilege elevation.
            # The only privileged path is the operator-requested fixed chmod.
            argv = [
                SUDO_PATH,
                '-S',
                '-k',
                '-p',
                '',
                '--',
                CHMOD_PATH,
                '0666',
                '--',
                *[snapshot.proc_path for snapshot in snapshots],
            ]
            method = 'sudo_chmod_open_descriptors'
            self._run_process(
                argv,
                secret=secret,
            )
            self._verify_snapshots(snapshots)
            token = self._stage_authorization(
                [snapshot.identity for snapshot in snapshots]
            )
            return self._success_payload(
                ports, method=method, lease_token=token
            )
        finally:
            _zero(secret)
            self._close_snapshots(snapshots)
            self._operation_lock.release()

    def _validated_discovery(self, *, allow_empty: bool = False) -> list[str]:
        try:
            raw_ports = list(self._port_discovery())
        except Exception:
            raise GripperPermissionError(
                'ttyusb_discovery_failed',
                '无法安全发现 ttyUSB 串口。',
                status=503,
            ) from None
        ports: list[tuple[int, str]] = []
        seen: set[str] = set()
        for raw_port in raw_ports:
            port = str(raw_port)
            path = Path(port)
            match = TTYUSB_NAME_PATTERN.fullmatch(path.name)
            if (
                match is None
                or path.parent != Path('/dev')
                or port in seen
            ):
                raise GripperPermissionError(
                    'unsafe_ttyusb_discovery',
                    '串口发现结果未通过安全校验。',
                    status=409,
                )
            try:
                metadata = os.lstat(port)
            except OSError:
                raise GripperPermissionError(
                    'ttyusb_disappeared',
                    'ttyUSB 串口在授权前已消失，请重新插拔后重试。',
                    status=409,
                ) from None
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISCHR(metadata.st_mode):
                raise GripperPermissionError(
                    'unsafe_ttyusb_device',
                    'ttyUSB 路径不是可授权的直接字符设备。',
                    status=409,
                )
            seen.add(port)
            ports.append((int(match.group(1)), port))
        if not ports and not allow_empty:
            raise GripperPermissionError(
                'ttyusb_not_found',
                '未发现可授权的 ttyUSB 字符设备。',
                status=409,
            )
        return [port for _index, port in sorted(ports, key=lambda item: item)]

    def _validated_targets(
        self,
        allowed_ports: Sequence[str],
        *,
        allow_empty: bool = False,
    ) -> list[str]:
        discovered = set(self._validated_discovery(allow_empty=True))
        targets: list[tuple[int, str]] = []
        seen: set[str] = set()
        for raw_port in allowed_ports:
            port = str(raw_port)
            path = Path(port)
            match = TTYUSB_NAME_PATTERN.fullmatch(path.name)
            if (
                match is None
                or path.parent != Path('/dev')
                or port in seen
            ):
                raise GripperPermissionError(
                    'unsafe_ttyusb_target',
                    '当前夹爪串口目标未通过安全校验。',
                    status=409,
                )
            try:
                metadata = os.lstat(port)
            except OSError:
                raise GripperPermissionError(
                    'ttyusb_disappeared',
                    '当前配置的夹爪 ttyUSB 串口已消失，请重新选择。',
                    status=409,
                ) from None
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISCHR(metadata.st_mode):
                raise GripperPermissionError(
                    'unsafe_ttyusb_device',
                    '夹爪 ttyUSB 路径不是可授权的直接字符设备。',
                    status=409,
                )
            if port not in discovered:
                raise GripperPermissionError(
                    'ttyusb_disappeared',
                    '当前配置的夹爪 ttyUSB 串口已消失，请重新选择。',
                    status=409,
                )
            seen.add(port)
            targets.append((int(match.group(1)), port))
        if not targets and not allow_empty:
            raise GripperPermissionError(
                'ttyusb_target_not_configured',
                '当前配置没有可授权的夹爪 ttyUSB 串口。',
                status=409,
            )
        return [port for _index, port in sorted(targets, key=lambda item: item)]

    @staticmethod
    def _current_identities(ports: Sequence[str]) -> list[PortIdentity]:
        identities: list[PortIdentity] = []
        for port in ports:
            try:
                metadata = os.lstat(port)
            except OSError:
                raise GripperPermissionError(
                    'ttyusb_disappeared',
                    'The ttyUSB device disappeared before authorization.',
                    status=409,
                ) from None
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISCHR(metadata.st_mode):
                raise GripperPermissionError(
                    'unsafe_ttyusb_device',
                    'The ttyUSB path is not a direct character device.',
                    status=409,
                )
            identities.append(_identity(str(port), metadata))
        return identities

    def _open_snapshots(self, ports: Sequence[str]) -> list[OpenPortSnapshot]:
        if not hasattr(os, 'O_PATH'):
            raise GripperPermissionError(
                'safe_permission_repair_unavailable',
                '当前系统不支持安全的串口权限授权。',
                status=503,
            )
        snapshots: list[OpenPortSnapshot] = []
        flags = int(os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            for port in ports:
                before = os.lstat(port)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISCHR(before.st_mode):
                    raise GripperPermissionError(
                        'unsafe_ttyusb_device',
                        'ttyUSB 路径不是可授权的直接字符设备。',
                        status=409,
                    )
                descriptor = os.open(port, flags)
                try:
                    opened = os.fstat(descriptor)
                    identity = _identity(port, before)
                    if not _same_identity(opened, identity):
                        raise GripperPermissionError(
                            'ttyusb_changed',
                            'ttyUSB 串口在授权前发生变化，请重试。',
                            status=409,
                        )
                    snapshots.append(OpenPortSnapshot(identity, descriptor))
                except Exception:
                    os.close(descriptor)
                    raise
            return snapshots
        except GripperPermissionError:
            self._close_snapshots(snapshots)
            raise
        except OSError:
            self._close_snapshots(snapshots)
            raise GripperPermissionError(
                'ttyusb_disappeared',
                'ttyUSB 串口在授权前已消失，请重试。',
                status=409,
            ) from None

    def _verify_snapshots(self, snapshots: Sequence[OpenPortSnapshot]) -> None:
        expected_paths = [snapshot.identity.path for snapshot in snapshots]
        current_paths = self._validated_targets(expected_paths)
        if current_paths != expected_paths:
            raise GripperPermissionError(
                'ttyusb_changed',
                'ttyUSB 设备集合在授权期间发生变化，请重新探测。',
                status=409,
            )
        for snapshot in snapshots:
            try:
                opened = os.fstat(snapshot.descriptor)
                current = os.lstat(snapshot.identity.path)
            except OSError:
                raise GripperPermissionError(
                    'ttyusb_disappeared',
                    'ttyUSB 串口在授权期间消失，请重新插拔后重试。',
                    status=409,
                ) from None
            if (
                not _same_identity(opened, snapshot.identity)
                or not _same_identity(current, snapshot.identity)
            ):
                raise GripperPermissionError(
                    'ttyusb_changed',
                    'ttyUSB 串口在授权期间被替换，请重新探测。',
                    status=409,
                )
            if stat.S_IMODE(opened.st_mode) & 0o666 != 0o666:
                raise GripperPermissionError(
                    'ttyusb_permission_not_applied',
                    'sudo 已返回，但串口读写权限仍未生效。',
                    status=409,
                )
            if not self._access_check(snapshot.identity.path):
                raise GripperPermissionError(
                    'ttyusb_permission_not_applied',
                    '串口读写权限仍不可用。',
                    status=409,
                )

    @staticmethod
    def _close_snapshots(snapshots: Sequence[OpenPortSnapshot]) -> None:
        for snapshot in snapshots:
            try:
                os.close(snapshot.descriptor)
            except OSError:
                pass

    def _consume_rate_limit(self, client_key: str) -> None:
        now = self._monotonic()
        key = str(client_key or 'unknown')[:128]
        with self._rate_lock:
            cutoff = now - self._rate_window_s
            while self._global_attempts and self._global_attempts[0] <= cutoff:
                self._global_attempts.popleft()
            for stale_key, attempts in list(self._attempts.items()):
                while attempts and attempts[0] <= cutoff:
                    attempts.popleft()
                if not attempts:
                    self._attempts.pop(stale_key, None)
            # Check the global cap before creating a per-client bucket so an
            # attacker cannot grow the mapping with spoofed/new source IPs.
            if len(self._global_attempts) >= self._global_rate_attempts:
                oldest = self._global_attempts[0]
                retry_after = max(
                    1,
                    int(math.ceil(oldest + self._rate_window_s - now)),
                )
                raise GripperPermissionError(
                    'permission_repair_rate_limited',
                    'sudo 授权尝试过于频繁，请稍后重试。',
                    status=429,
                    retry_after=retry_after,
                )
            client_attempts = self._attempts.get(key)
            if client_attempts is None:
                while len(self._attempts) >= MAX_RATE_CLIENTS:
                    self._attempts.popitem(last=False)
                client_attempts = deque()
                self._attempts[key] = client_attempts
            else:
                self._attempts.move_to_end(key)
            if len(client_attempts) >= self._rate_attempts:
                retry_after = max(
                    1,
                    int(math.ceil(
                        client_attempts[0] + self._rate_window_s - now
                    )),
                )
                raise GripperPermissionError(
                    'permission_repair_rate_limited',
                    'sudo 授权尝试过于频繁，请稍后重试。',
                    status=429,
                    retry_after=retry_after,
                )
            client_attempts.append(now)
            self._global_attempts.append(now)

    def _run_process(
        self,
        argv: Sequence[str],
        *,
        secret: bytearray,
    ) -> None:
        expected_prefix = [SUDO_PATH]
        if list(argv[:1]) != expected_prefix:
            raise GripperPermissionError(
                'permission_repair_internal_error',
                '串口权限授权命令未通过内部校验。',
                status=500,
            )
        process = None
        try:
            process = self._process_factory(
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
                bufsize=0,
                env={
                    'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
                    'LANG': 'C',
                    'LC_ALL': 'C',
                },
            )
            if process.stdin is None:
                raise GripperPermissionError(
                    'authorization_failed',
                    '授权失败；无法确认是密码还是 sudo 策略导致。',
                    status=403,
                )
            try:
                process.stdin.write(secret)
                process.stdin.write(b'\n')
                process.stdin.flush()
            finally:
                process.stdin.close()
                process.stdin = None
                _zero(secret)
            try:
                process.communicate(
                    timeout=self._timeout_s
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise GripperPermissionError(
                    'sudo_timeout',
                    'sudo 授权超时，未继续等待。',
                    status=504,
                ) from None
            returncode = int(process.returncode or 0)
            if returncode == 0:
                return
            if returncode == 66:
                raise GripperPermissionError(
                    'ttyusb_disappeared',
                    'ttyUSB 串口在授权期间消失，请重新插拔后重试。',
                    status=409,
                )
            if returncode in {73, 75}:
                raise GripperPermissionError(
                    'ttyusb_changed',
                    'ttyUSB 串口在授权期间发生变化，请重新探测。',
                    status=409,
                )
            if returncode == 77:
                raise GripperPermissionError(
                    'ttyusb_permission_not_applied',
                    '串口读写权限未能安全应用。',
                    status=409,
                )
            raise GripperPermissionError(
                'authorization_failed',
                '授权失败；无法确认是密码还是 sudo 策略导致。',
                status=403,
            )
        except GripperPermissionError:
            raise
        except (OSError, subprocess.SubprocessError):
            raise GripperPermissionError(
                'authorization_failed',
                '授权失败；无法确认是密码还是 sudo 策略导致。',
                status=403,
            ) from None
        finally:
            _zero(secret)
            self._invalidate_sudo_ticket()
            if process is not None and getattr(process, 'stdin', None) is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

    def _invalidate_sudo_ticket(self) -> None:
        """Best-effort invalidation of any timestamp created by password sudo."""
        process = None
        try:
            process = self._process_factory(
                [SUDO_PATH, '-k'],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
                bufsize=0,
                env={
                    'PATH': '/usr/sbin:/usr/bin:/sbin:/bin',
                    'LANG': 'C',
                    'LC_ALL': 'C',
                },
            )
            try:
                process.communicate(timeout=min(1.0, self._timeout_s))
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        except (OSError, subprocess.SubprocessError):
            pass

    def _success_payload(
        self,
        ports: Sequence[str],
        *,
        method: str,
        lease_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            'ok': True,
            'authorized': True,
            'method': method,
            'ports': list(ports),
            'permission': self.status(ports),
        }
        if lease_token:
            # Internal two-phase token. The HTTP/controller layer must consume
            # and remove it before serializing the response.
            payload['_lease_token'] = lease_token
        return payload


__all__ = [
    'GripperPermissionError',
    'GripperPermissionRequiredError',
    'MAX_PASSWORD_BYTES',
    'TtyUsbPermissionAuthorizer',
    'discover_strict_ttyusb_ports',
]
