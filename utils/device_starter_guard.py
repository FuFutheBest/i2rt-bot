"""Cross-process guards and hardware preflight checks for DeviceStarter."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from utils.inference_options import as_bool

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal source installs
    psutil = None  # type: ignore[assignment]


class DeviceStarterGuardError(RuntimeError):
    """Base class for DeviceStarter startup guard failures."""


class DeviceStarterAlreadyRunningError(DeviceStarterGuardError):
    """Raised when another DeviceStarter owns an overlapping runtime resource."""


class DeviceStarterPreflightError(DeviceStarterGuardError):
    """Raised when a required hardware service is not available."""


@dataclass(frozen=True)
class GripperWebSocketEndpoint:
    """LKTech WebSocket endpoint required by a configured robot."""

    host: str
    port: int
    exclusive_client: bool

    @property
    def uri(self) -> str:
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"ws://{host}:{self.port}"


def _normalize_host(host: Any) -> str:
    normalized = str(host).strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        return normalized[1:-1]
    return normalized


def _configured_devices(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    devices: Any = config.get("devices", [])
    if isinstance(devices, Mapping):
        devices = devices.get("devices", [])
    if not isinstance(devices, list):
        return []
    enabled_devices = []
    for device in devices:
        if not isinstance(device, Mapping):
            continue
        enabled = device.get("enabled", True)
        if not as_bool(enabled):
            continue
        if isinstance(enabled, (list, tuple)) and not any(
            as_bool(item) for item in enabled
        ):
            continue
        enabled_devices.append(device)
    return enabled_devices


def _any_enabled(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(as_bool(item) for item in value)
    return as_bool(value)


def configured_gripper_endpoints(config: Mapping[str, Any]) -> tuple[GripperWebSocketEndpoint, ...]:
    """Return distinct LKTech WebSocket endpoints required by enabled robots."""
    endpoints = set()
    for device in _configured_devices(config):
        if device.get("class") != "RizonRobot":
            continue
        gripper = device.get("gripper", {})
        if not isinstance(gripper, Mapping):
            continue
        if str(gripper.get("backend", "")).strip().lower() != "lktech_websocket":
            continue
        if not _any_enabled(gripper.get("enabled", True)):
            continue

        server = gripper.get("server", {})
        if not isinstance(server, Mapping):
            server = {}
        uri = server.get("uri")
        if uri:
            parsed = urlparse(str(uri))
            host = _normalize_host(parsed.hostname or "127.0.0.1")
            port = parsed.port or 8765
        else:
            host = _normalize_host(server.get("host", "127.0.0.1"))
            port = int(server.get("port", 8765))
        endpoints.add(
            GripperWebSocketEndpoint(
                host=host,
                port=port,
                # This repository's standalone server is intentionally single-client.
                # Set false only when using the Control Panel multi-client server.
                exclusive_client=as_bool(server.get("exclusive_client", True)),
            )
        )
    return tuple(sorted(endpoints, key=lambda endpoint: (endpoint.host, endpoint.port)))


def device_starter_resources(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Build stable resource names that must not be owned by two starters."""
    resources = set()
    summary_shm = config.get("summary_shm")
    shared = config.get("shared", {})
    if not summary_shm and isinstance(shared, Mapping):
        summary_shm = shared.get("summary_shm_name")
    # BaseDeviceManager falls back to this name when no explicit summary SHM
    # is configured; lock that implicit resource too so legacy configs cannot
    # silently start a second writer.
    resources.add(f"shm:{summary_shm or 'device_summary_data'}")

    for device in _configured_devices(config):
        device_class = device.get("class")
        device_class_id = device.get("device_class_id")
        if device_class is not None and device_class_id is not None:
            resources.add(f"device:{device_class}:{device_class_id}")
        robot_serials = device.get("robot_sn", [])
        if isinstance(robot_serials, str):
            robot_serials = [robot_serials]
        if isinstance(robot_serials, (list, tuple)):
            resources.update(f"robot:{serial}" for serial in robot_serials if serial)

    for endpoint in configured_gripper_endpoints(config):
        resources.add(f"gripper-websocket:{endpoint.host}:{endpoint.port}")
    return tuple(sorted(resources))


def _lock_filename(resource: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", resource).strip("-")[:48]
    digest = hashlib.sha256(resource.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}.lock"


class DeviceStarterProcessGuard:
    """Hold advisory locks for every hardware/SHM resource used by a starter."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        lock_dir: Path | None = None,
        argv: Sequence[str] | None = None,
    ) -> None:
        self.resources = device_starter_resources(config)
        runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
        self.lock_dir = lock_dir or (
            runtime_root / f"superinference-device-starter-{os.getuid()}"
        )
        self.argv = list(argv or [])
        self._locks: list[tuple[str, TextIO]] = []

    def acquire(self) -> None:
        """Acquire every resource lock without waiting."""
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "cwd": os.getcwd(),
            "argv": self.argv,
        }
        try:
            for resource in self.resources:
                path = self.lock_dir / _lock_filename(resource)
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    owner = self._read_owner(handle)
                    handle.close()
                    raise DeviceStarterAlreadyRunningError(
                        self._format_lock_error(resource, owner)
                    ) from exc
                handle.seek(0)
                handle.truncate()
                json.dump(metadata, handle, ensure_ascii=True)
                handle.flush()
                self._locks.append((resource, handle))
        except Exception:
            self.release()
            raise

    @staticmethod
    def _read_owner(handle: TextIO) -> dict[str, Any]:
        try:
            handle.seek(0)
            value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _format_lock_error(resource: str, owner: Mapping[str, Any]) -> str:
        pid = owner.get("pid", "unknown")
        cwd = owner.get("cwd", "unknown")
        argv = owner.get("argv", [])
        command = " ".join(str(part) for part in argv) if isinstance(argv, list) else str(argv)
        return (
            f"Another DeviceStarter already owns {resource!r}: PID {pid}, cwd={cwd}, command={command!r}. "
            "Stop the existing DeviceStarter cleanly before starting another one."
        )

    def release(self) -> None:
        """Release all acquired resource locks."""
        while self._locks:
            _, handle = self._locks.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _address_host_port(address: Any) -> tuple[str | None, int | None]:
    if not address:
        return None, None
    if hasattr(address, "ip") and hasattr(address, "port"):
        return str(address.ip), int(address.port)
    if isinstance(address, (list, tuple)) and len(address) >= 2:
        return str(address[0]), int(address[1])
    return None, None


def _is_local_host(host: str) -> bool:
    return _normalize_host(host).lower() in {"127.0.0.1", "localhost", "::1"}


def _service_unit_for_pid(pid: int) -> str | None:
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    matches = re.findall(r"/([^/]+\.service)(?:/|$)", cgroup)
    return matches[-1] if matches else None


def _describe_process(pid: int | None) -> str:
    if pid is None:
        return "PID unknown"
    if psutil is None:
        return f"PID {pid}, command='unavailable'"
    try:
        command = " ".join(psutil.Process(pid).cmdline())
    except (psutil.Error, OSError):
        command = "unavailable"
    unit = _service_unit_for_pid(pid)
    unit_text = f", systemd unit={unit}" if unit else ""
    return f"PID {pid}, command={command!r}{unit_text}"


def _endpoint_connections(
    endpoint: GripperWebSocketEndpoint,
    connections: Iterable[Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    listeners: list[Any] = []
    clients: list[Any] = []
    server_side_established: list[Any] = []
    for connection in connections:
        _, local_port = _address_host_port(getattr(connection, "laddr", None))
        remote_host, remote_port = _address_host_port(getattr(connection, "raddr", None))
        status = getattr(connection, "status", None)
        if status == psutil.CONN_LISTEN and local_port == endpoint.port:
            listeners.append(connection)
        elif status == psutil.CONN_ESTABLISHED and remote_port == endpoint.port:
            if _is_local_host(endpoint.host) or _normalize_host(remote_host or "") == endpoint.host:
                clients.append(connection)
        elif status == psutil.CONN_ESTABLISHED and local_port == endpoint.port:
            server_side_established.append(connection)
    return listeners, clients, server_side_established


def _describe_remote_client(connection: Any) -> str:
    host, port = _address_host_port(getattr(connection, "raddr", None))
    address = f"{host}:{port}" if host and port else "unknown address"
    return f"remote client {address} (client PID unavailable)"


def check_gripper_websocket_preflight(
    config: Mapping[str, Any],
    *,
    connections: Iterable[Any] | None = None,
) -> tuple[GripperWebSocketEndpoint, ...]:
    """Fail before hardware startup when an exclusive gripper endpoint is unavailable."""
    endpoints = configured_gripper_endpoints(config)
    if not endpoints:
        return endpoints
    if psutil is None:
        raise DeviceStarterPreflightError(
            "LK gripper WebSocket preflight requires the optional 'psutil' package."
        )
    try:
        source = psutil.net_connections(kind="tcp") if connections is None else connections
        all_connections = list(source)
    except (psutil.Error, OSError) as exc:
        raise DeviceStarterPreflightError(
            f"Unable to inspect LK gripper WebSocket ownership: {exc}"
        ) from exc

    for endpoint in endpoints:
        if not _is_local_host(endpoint.host):
            continue
        listeners, active_clients, server_connections = _endpoint_connections(
            endpoint, all_connections
        )
        if not listeners:
            raise DeviceStarterPreflightError(
                f"LK gripper WebSocket server is not listening at {endpoint.uri}. "
                "Start the configured LKTech gripper service before DeviceStarter."
            )
        if endpoint.exclusive_client and active_clients:
            owners = "; ".join(
                _describe_process(getattr(connection, "pid", None))
                for connection in active_clients
            )
            raise DeviceStarterPreflightError(
                f"Exclusive LK gripper WebSocket server {endpoint.uri} already has an active client: {owners}. "
                "Stop the owning client or service before DeviceStarter; do not stop the 8765 listener."
            )
        if endpoint.exclusive_client and server_connections:
            owners = "; ".join(_describe_remote_client(connection) for connection in server_connections)
            raise DeviceStarterPreflightError(
                f"Exclusive LK gripper WebSocket server {endpoint.uri} already has an active client: {owners}. "
                "Stop the owning client before DeviceStarter; do not stop the 8765 listener."
            )
    return endpoints
