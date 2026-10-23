"""Bounded sudo recovery for configured SocketCAN interfaces.

Yam/i2rt can open an already-configured SocketCAN interface as the service
user, but bringing that interface up at the required bitrate needs
``CAP_NET_ADMIN``.  This module provides a narrowly scoped password flow for
that one operation without installing a helper or persisting credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .gripper_permissions import GripperPermissionError, MAX_PASSWORD_BYTES


CAN_INTERFACE_PATTERN = re.compile(r"can([0-9]+)\Z")
SUDO_PATH = "/usr/bin/sudo"
IP_PATH = "/usr/sbin/ip"
DEFAULT_BITRATE = 1_000_000
DEFAULT_TIMEOUT_S = 4.0
DEFAULT_CACHE_TTL_S = 0.75


def _zero(buffer: Optional[bytearray]) -> None:
    if buffer is None:
        return
    for index in range(len(buffer)):
        buffer[index] = 0


def _environment() -> Dict[str, str]:
    return {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
    }


def _validate_interface_name(value: Any) -> str:
    interface = str(value)
    if CAN_INTERFACE_PATTERN.fullmatch(interface) is None:
        raise GripperPermissionError(
            "unsafe_can_interface",
            "Yam CAN 接口名称未通过安全校验。",
            status=409,
        )
    return interface


@dataclass(frozen=True)
class CanLinkSnapshot:
    interface: str
    present: bool
    identity: str = ""
    ifindex: Optional[int] = None
    up: bool = False
    bitrate: Optional[int] = None
    state: str = ""

    def ready(self, expected_bitrate: int) -> bool:
        return bool(self.present and self.up and self.bitrate == int(expected_bitrate))

    def payload(self, expected_bitrate: int) -> Dict[str, Any]:
        return {
            "present": self.present,
            "up": self.up,
            "bitrate": self.bitrate,
            "expected_bitrate": int(expected_bitrate),
            "state": self.state,
            "ready": self.ready(expected_bitrate),
        }


def read_can_link_snapshot(
    interface: str,
    *,
    ip_path: str = IP_PATH,
    sys_class_net: Path = Path("/sys/class/net"),
) -> CanLinkSnapshot:
    """Read one CAN link with structured ``ip -json`` output."""
    safe_interface = _validate_interface_name(interface)
    sysfs_entry = Path(sys_class_net) / safe_interface
    if not sysfs_entry.exists():
        return CanLinkSnapshot(interface=safe_interface, present=False)
    try:
        resolved = str(sysfs_entry.resolve(strict=True))
    except OSError:
        return CanLinkSnapshot(interface=safe_interface, present=False)
    try:
        result = subprocess.run(
            [
                str(ip_path),
                "-details",
                "-json",
                "link",
                "show",
                "dev",
                safe_interface,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
            env=_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        raise GripperPermissionError(
            "can_inspection_unavailable",
            "无法读取 Yam CAN 接口状态。",
            status=503,
        ) from None
    if result.returncode != 0:
        if not sysfs_entry.exists():
            return CanLinkSnapshot(interface=safe_interface, present=False)
        raise GripperPermissionError(
            "can_inspection_unavailable",
            "无法读取 Yam CAN 接口状态。",
            status=503,
        )
    try:
        items = json.loads(result.stdout)
        item = items[0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        raise GripperPermissionError(
            "can_inspection_unavailable",
            "Yam CAN 接口状态格式无效。",
            status=503,
        ) from None
    if not isinstance(item, Mapping) or item.get("ifname") != safe_interface:
        raise GripperPermissionError(
            "can_inspection_unavailable",
            "Yam CAN 接口状态与请求目标不一致。",
            status=503,
        )
    linkinfo = item.get("linkinfo")
    linkinfo = linkinfo if isinstance(linkinfo, Mapping) else {}
    kind = str(item.get("link_type") or linkinfo.get("info_kind") or "")
    if kind != "can" or str(linkinfo.get("info_kind") or "") != "can":
        raise GripperPermissionError(
            "unsafe_can_interface",
            "配置目标不是 SocketCAN 接口。",
            status=409,
        )
    info_data = linkinfo.get("info_data")
    info_data = info_data if isinstance(info_data, Mapping) else {}
    bittiming = info_data.get("bittiming")
    bittiming = bittiming if isinstance(bittiming, Mapping) else {}
    raw_bitrate = bittiming.get("bitrate")
    try:
        bitrate = int(raw_bitrate) if raw_bitrate is not None else None
    except (TypeError, ValueError):
        bitrate = None
    try:
        ifindex = int(item.get("ifindex"))
    except (TypeError, ValueError):
        ifindex = None
    flags = item.get("flags")
    flags = flags if isinstance(flags, list) else []
    identity = ":".join(
        (
            str(ifindex if ifindex is not None else ""),
            resolved,
            str(item.get("parentbus") or ""),
            str(item.get("parentdev") or ""),
        )
    )
    return CanLinkSnapshot(
        interface=safe_interface,
        present=True,
        identity=identity,
        ifindex=ifindex,
        up="UP" in {str(flag) for flag in flags},
        bitrate=bitrate,
        state=str(info_data.get("state") or item.get("operstate") or ""),
    )


class CanPermissionRequiredError(GripperPermissionError):
    """Raised before Yam start when its SocketCAN link needs sudo setup."""

    def __init__(
        self,
        interfaces: Sequence[str],
        *,
        expected_bitrate: int = DEFAULT_BITRATE,
    ) -> None:
        affected = sorted(set(map(str, interfaces)))
        super().__init__(
            "can_interface_permission_required",
            "Yam CAN 接口尚未以 1 Mbps 启用，请先完成 sudo 授权。",
            status=409,
            details={
                "affected_can_interfaces": affected,
                "expected_can_bitrate": int(expected_bitrate),
            },
        )


class CanInterfacePermissionAuthorizer:
    """Inspect and configure only explicitly selected ``canN`` links."""

    def __init__(
        self,
        *,
        link_reader: Optional[Callable[[str], CanLinkSnapshot]] = None,
        process_factory: Callable[..., Any] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        ip_path: str = IP_PATH,
        bitrate: int = DEFAULT_BITRATE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
    ) -> None:
        self._ip_path = str(ip_path)
        self._link_reader = link_reader or (
            lambda interface: read_can_link_snapshot(interface, ip_path=self._ip_path)
        )
        self._process_factory = process_factory
        self._monotonic = monotonic
        self._bitrate = int(bitrate)
        self._timeout_s = max(0.1, float(timeout_s))
        self._cache_ttl_s = max(0.0, float(cache_ttl_s))
        self._operation_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: Dict[str, tuple[float, CanLinkSnapshot]] = {}

    @property
    def bitrate(self) -> int:
        return self._bitrate

    @staticmethod
    def _validated_targets(
        interfaces: Sequence[str],
        *,
        allow_empty: bool = False,
    ) -> list[str]:
        targets: list[tuple[int, str]] = []
        seen: set[str] = set()
        for raw_interface in interfaces:
            interface = _validate_interface_name(raw_interface)
            if interface in seen:
                raise GripperPermissionError(
                    "unsafe_can_interface",
                    "Yam CAN 接口列表包含重复目标。",
                    status=409,
                )
            seen.add(interface)
            match = CAN_INTERFACE_PATTERN.fullmatch(interface)
            assert match is not None
            targets.append((int(match.group(1)), interface))
        if not targets and not allow_empty:
            raise GripperPermissionError(
                "can_interface_not_configured",
                "当前 config 没有可授权的 Yam CAN 接口。",
                status=409,
            )
        return [name for _index, name in sorted(targets)]

    def invalidate(self, interfaces: Sequence[str] = ()) -> None:
        with self._cache_lock:
            if interfaces:
                for interface in interfaces:
                    self._cache.pop(str(interface), None)
            else:
                self._cache.clear()

    def _snapshot(self, interface: str, *, refresh: bool = False) -> CanLinkSnapshot:
        now = self._monotonic()
        if not refresh:
            with self._cache_lock:
                cached = self._cache.get(interface)
                if cached is not None and now - cached[0] <= self._cache_ttl_s:
                    return cached[1]
        snapshot = self._link_reader(interface)
        if snapshot.interface != interface:
            raise GripperPermissionError(
                "can_inspection_unavailable",
                "Yam CAN 接口状态与请求目标不一致。",
                status=503,
            )
        with self._cache_lock:
            self._cache[interface] = (now, snapshot)
        return snapshot

    @staticmethod
    def _context_id(snapshots: Sequence[CanLinkSnapshot], bitrate: int) -> str:
        digest = hashlib.sha256()
        digest.update(b"can-permission-context-v1\0")
        digest.update(str(int(bitrate)).encode("ascii"))
        for snapshot in snapshots:
            payload = (
                snapshot.interface,
                snapshot.present,
                snapshot.identity,
                snapshot.ifindex,
                snapshot.up,
                snapshot.bitrate,
                snapshot.state,
            )
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def status(
        self,
        allowed_interfaces: Sequence[str] = (),
        *,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        interfaces = self._validated_targets(allowed_interfaces, allow_empty=True)
        snapshots = [
            self._snapshot(interface, refresh=refresh) for interface in interfaces
        ]
        affected = [
            item.interface
            for item in snapshots
            if item.present and not item.ready(self._bitrate)
        ]
        unavailable = [item.interface for item in snapshots if not item.present]
        return {
            "required": bool(affected),
            "target_interfaces": interfaces,
            "affected_interfaces": affected,
            "unavailable_interfaces": unavailable,
            "expected_bitrate": self._bitrate,
            "context_id": self._context_id(snapshots, self._bitrate),
            "interfaces": {
                item.interface: item.payload(self._bitrate) for item in snapshots
            },
        }

    def ensure_ready(self, interfaces: Sequence[str]) -> Dict[str, Any]:
        status = self.status(interfaces, refresh=True)
        unavailable = list(status["unavailable_interfaces"])
        if unavailable:
            raise GripperPermissionError(
                "can_interface_unavailable",
                "未发现 config 指定的 Yam CAN 接口：" + "、".join(unavailable),
                status=409,
                details={"unavailable_can_interfaces": unavailable},
            )
        affected = list(status["affected_interfaces"])
        if affected:
            raise CanPermissionRequiredError(affected, expected_bitrate=self._bitrate)
        return status

    def authorize_with_password(
        self,
        secret: bytearray,
        client_key: str,
        allowed_interfaces: Sequence[str],
    ) -> Dict[str, Any]:
        del client_key
        if not isinstance(secret, bytearray):
            raise TypeError("secret must be a mutable bytearray")
        if (
            not secret
            or len(secret) > MAX_PASSWORD_BYTES
            or 0 in secret
            or 10 in secret
            or 13 in secret
        ):
            _zero(secret)
            raise GripperPermissionError(
                "invalid_sudo_password",
                "sudo 密码格式无效。",
                status=422,
            )
        if not self._operation_lock.acquire(blocking=False):
            _zero(secret)
            raise GripperPermissionError(
                "permission_repair_busy",
                "已有 CAN 权限授权正在进行，请稍后重试。",
                status=429,
                retry_after=1,
            )
        try:
            interfaces = self._validated_targets(allowed_interfaces)
            before_status = self.status(interfaces, refresh=True)
            unavailable = list(before_status["unavailable_interfaces"])
            if unavailable:
                raise GripperPermissionError(
                    "can_interface_unavailable",
                    "Yam CAN 接口在授权前已消失。",
                    status=409,
                    details={"unavailable_can_interfaces": unavailable},
                )
            affected = list(before_status["affected_interfaces"])
            if not affected:
                return {
                    "ok": True,
                    "authorized": True,
                    "method": "already_ready",
                    "interfaces": interfaces,
                    "permission": before_status,
                }
            identities = {
                interface: self._snapshot(interface, refresh=True).identity
                for interface in affected
            }
            for interface in affected:
                self._run_sudo(
                    self._down_command(interface),
                    secret=bytearray(secret),
                )
                self.invalidate([interface])
                down = self._snapshot(interface, refresh=True)
                if not down.present or down.identity != identities[interface]:
                    raise GripperPermissionError(
                        "can_interface_changed",
                        "Yam CAN 接口在授权期间发生变化，请重新探测。",
                        status=409,
                    )
                self._run_sudo(
                    self._up_command(interface),
                    secret=bytearray(secret),
                )
                self.invalidate([interface])
                ready = self._snapshot(interface, refresh=True)
                if not ready.present or ready.identity != identities[interface]:
                    raise GripperPermissionError(
                        "can_interface_changed",
                        "Yam CAN 接口在授权期间发生变化，请重新探测。",
                        status=409,
                    )
                if not ready.ready(self._bitrate):
                    raise GripperPermissionError(
                        "can_permission_not_applied",
                        "sudo 已返回，但 Yam CAN 接口仍未以 1 Mbps 启用。",
                        status=409,
                    )
            permission = self.status(interfaces, refresh=True)
            return {
                "ok": True,
                "authorized": True,
                "method": "sudo_ip_link",
                "interfaces": interfaces,
                "permission": permission,
            }
        finally:
            _zero(secret)
            self._invalidate_sudo_ticket()
            self._operation_lock.release()

    def _down_command(self, interface: str) -> list[str]:
        return [
            SUDO_PATH,
            "-S",
            "-k",
            "-p",
            "",
            "--",
            self._ip_path,
            "link",
            "set",
            interface,
            "down",
        ]

    def _up_command(self, interface: str) -> list[str]:
        return [
            SUDO_PATH,
            "-S",
            "-k",
            "-p",
            "",
            "--",
            self._ip_path,
            "link",
            "set",
            interface,
            "up",
            "type",
            "can",
            "bitrate",
            str(self._bitrate),
        ]

    def _valid_command(self, argv: Sequence[str]) -> bool:
        if len(argv) < 11:
            return False
        prefix = [
            SUDO_PATH,
            "-S",
            "-k",
            "-p",
            "",
            "--",
            self._ip_path,
            "link",
            "set",
        ]
        if list(argv[:9]) != prefix:
            return False
        interface = str(argv[9])
        if CAN_INTERFACE_PATTERN.fullmatch(interface) is None:
            return False
        tail = list(argv[10:])
        return tail in (
            ["down"],
            ["up", "type", "can", "bitrate", str(self._bitrate)],
        )

    def _run_sudo(self, argv: Sequence[str], *, secret: bytearray) -> None:
        if not self._valid_command(argv):
            _zero(secret)
            raise GripperPermissionError(
                "permission_repair_internal_error",
                "CAN 权限授权命令未通过内部校验。",
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
                env=_environment(),
            )
            if process.stdin is None:
                raise GripperPermissionError(
                    "authorization_failed",
                    "授权失败；无法确认是密码还是 sudo 策略导致。",
                    status=403,
                )
            try:
                process.stdin.write(secret)
                process.stdin.write(b"\n")
                process.stdin.flush()
            finally:
                process.stdin.close()
                process.stdin = None
                _zero(secret)
            try:
                process.communicate(timeout=self._timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise GripperPermissionError(
                    "sudo_timeout",
                    "sudo 授权超时，未继续等待。",
                    status=504,
                ) from None
            if int(process.returncode or 0) != 0:
                raise GripperPermissionError(
                    "authorization_failed",
                    "授权失败；无法确认是密码还是 sudo 策略导致。",
                    status=403,
                )
        except GripperPermissionError:
            raise
        except (OSError, subprocess.SubprocessError):
            raise GripperPermissionError(
                "authorization_failed",
                "授权失败；无法确认是密码还是 sudo 策略导致。",
                status=403,
            ) from None
        finally:
            _zero(secret)
            if process is not None and getattr(process, "stdin", None) is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

    def _invalidate_sudo_ticket(self) -> None:
        process = None
        try:
            process = self._process_factory(
                [SUDO_PATH, "-k"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                close_fds=True,
                bufsize=0,
                env=_environment(),
            )
            try:
                process.communicate(timeout=min(1.0, self._timeout_s))
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        except (OSError, subprocess.SubprocessError):
            pass


__all__ = [
    "CAN_INTERFACE_PATTERN",
    "CanInterfacePermissionAuthorizer",
    "CanLinkSnapshot",
    "CanPermissionRequiredError",
    "DEFAULT_BITRATE",
    "read_can_link_snapshot",
]
