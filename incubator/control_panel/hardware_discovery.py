"""Passive hardware discovery and in-memory runtime binding helpers.

The control panel never opens a camera and never constructs a Flexiv RDK
``Robot`` from this module.  Camera inventory comes from sysfs/udev links and
robot inventory comes from cached Flexiv Elements Studio logs plus passive
socket observations.
"""

from __future__ import annotations

import copy
from collections import deque
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .camera_reachability import CameraFrameProbe


_ELEMENTS_ROBOT_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} [^\]]+)\].*?"
    r"(?:connect to|disconnect from) ip\s*=\s*(?P<ip>\d+\.\d+\.\d+\.\d+)"
    r".*?sn(?:\.toStdString\(\))?\s*=\s*(?P<serial>[A-Za-z0-9-]+)",
    re.IGNORECASE,
)
_RIZON_SERIAL_RE = re.compile(r"^Rizon\s*4(?P<right>R?)-(?P<number>\d{6})$")
_ELEMENTS_CACHE_SERIAL_RE = re.compile(
    r"\bA02L(?:MR)?(?:-[A-Za-z0-9]+)+-\d{6}\b",
    re.IGNORECASE,
)
_ELEMENTS_DIRECTORY_KEYWORDS = ("flexiv", "elements", "fsp")
_ELEMENTS_CACHE_FILENAMES = ("lastConnectionTimeInfo.xml",)
_ELEMENTS_CACHE_PATTERNS = (
    "UIApp_*.log",
    "lastConnectionTimeInfo.xml",
    "generated_robot_*.json",
)
_ELEMENTS_SEARCH_ROOTS_ENV = "SUPERINFERENCE_ELEMENTS_CACHE_ROOTS"
_WEBCAM_CLASSES = {
    "WebCameraDevice",
    "WebCameraOpenPi",
}
CAPTURE_BINDING_STATE_KEY = "_control_panel_capture_binding"
CAPTURE_BINDING_MISSING = "missing"
_ROBOT_PROBE_WORKERS = 4
_ROBOT_PING_TIMEOUT_SECONDS = 1.0


def capture_binding_is_missing(device: Mapping[str, Any]) -> bool:
    """Return whether a camera was explicitly parked without physical hardware."""

    return device.get(CAPTURE_BINDING_STATE_KEY) == CAPTURE_BINDING_MISSING


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _opaque_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, OSError, PermissionError):
        return None


def _run_text(command: List[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def _default_robot_reachability_probe(
    robot: Mapping[str, Any],
) -> Dict[str, Any]:
    """Ping one cached robot IP without opening an RDK control connection."""

    started = time.monotonic()
    try:
        ip = str(ipaddress.ip_address(str(robot.get("ip") or "")))
    except ValueError:
        return {
            "reachable": False,
            "status": "missing_ip",
            "duration_seconds": 0.0,
        }
    try:
        completed = subprocess.run(
            [
                "ping",
                "-n",
                "-c",
                "1",
                "-W",
                str(max(1, int(_ROBOT_PING_TIMEOUT_SECONDS))),
                ip,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_ROBOT_PING_TIMEOUT_SECONDS + 1.0,
        )
        reachable = completed.returncode == 0
        status = "reachable" if reachable else "unreachable"
    except subprocess.TimeoutExpired:
        reachable = False
        status = "timeout"
    except OSError:
        reachable = False
        status = "probe_unavailable"
    return {
        "reachable": reachable,
        "status": status,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def _robot_probe_key(robot: Mapping[str, Any]) -> str:
    return "\x1f".join(
        (
            str(robot.get("rdk_sn") or robot.get("serial_number") or ""),
            str(robot.get("ip") or ""),
        )
    )


class RobotReachabilityCoordinator:
    """Keep slow robot reachability checks off the HTTP request thread."""

    def __init__(
        self,
        probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        max_workers: int = _ROBOT_PROBE_WORKERS,
    ) -> None:
        self._probe = probe
        self._max_workers = max(1, min(int(max_workers), 8))
        self._lock = threading.RLock()
        self._generation = 0
        self._fingerprint = ""
        self._state = "idle"
        self._total = 0
        self._completed = 0
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._results: Dict[str, Dict[str, Any]] = {}
        self._thread: Optional[threading.Thread] = None

    def start(
        self,
        robots: Iterable[Mapping[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        candidates = [copy.deepcopy(dict(robot)) for robot in robots]
        fingerprint = hashlib.sha256(
            json.dumps(
                sorted(_robot_probe_key(robot) for robot in candidates),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self._lock:
            if self._state == "running":
                return
            if not force and self._state == "complete" and fingerprint == self._fingerprint:
                return
            self._generation += 1
            generation = self._generation
            self._fingerprint = fingerprint
            self._state = "running"
            self._total = len(candidates)
            self._completed = 0
            self._started_at = _utc_now()
            self._finished_at = None
            valid_keys = {_robot_probe_key(robot) for robot in candidates}
            self._results = {
                key: value for key, value in self._results.items() if key in valid_keys
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(generation, candidates),
                name="robot-reachability-discovery",
                daemon=True,
            )
            self._thread.start()

    def _run(
        self,
        generation: int,
        candidates: List[Dict[str, Any]],
    ) -> None:
        work_lock = threading.Lock()
        next_index = 0

        def worker() -> None:
            nonlocal next_index
            while True:
                with work_lock:
                    if next_index >= len(candidates):
                        return
                    robot = candidates[next_index]
                    next_index += 1
                key = _robot_probe_key(robot)
                started = time.monotonic()
                try:
                    raw_result = dict(self._probe(robot))
                    reachable = raw_result.get("reachable") is True
                    status = str(
                        raw_result.get("status")
                        or ("reachable" if reachable else "unreachable")
                    )
                    duration = float(
                        raw_result.get(
                            "duration_seconds", time.monotonic() - started
                        )
                    )
                except Exception as exc:
                    reachable = False
                    status = f"probe_error:{type(exc).__name__}"
                    duration = time.monotonic() - started
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._results[key] = {
                        "reachable": reachable,
                        "status": status,
                        "duration_seconds": round(max(0.0, duration), 3),
                        "checked_at": _utc_now(),
                    }
                    self._completed += 1

        workers = [
            threading.Thread(
                target=worker,
                name=f"robot-reachability-worker-{index}",
                daemon=True,
            )
            for index in range(min(self._max_workers, max(1, len(candidates))))
        ]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()
        with self._lock:
            if generation == self._generation:
                self._state = "complete"
                self._completed = self._total
                self._finished_at = _utc_now()

    def project(
        self, robots: Iterable[Mapping[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidates = [copy.deepcopy(dict(robot)) for robot in robots]
        with self._lock:
            results = copy.deepcopy(self._results)
            state = self._state
            progress = {
                "state": state,
                "total": self._total,
                "completed": self._completed,
                "reachable": sum(
                    result.get("reachable") is True for result in results.values()
                ),
                "started_at": self._started_at,
                "finished_at": self._finished_at,
            }
        if state == "idle":
            return candidates, progress
        reachable: List[Dict[str, Any]] = []
        for robot in candidates:
            result = results.get(_robot_probe_key(robot))
            if not result or result.get("reachable") is not True:
                continue
            item = dict(robot)
            item.update(
                {
                    "detected": True,
                    "reachable": True,
                    "probe_status": result.get("status"),
                    "probe_checked_at": result.get("checked_at"),
                    "confidence": "ping_probe",
                }
            )
            reachable.append(item)
        progress["reachable"] = len(reachable)
        return reachable, progress

    def wait(self, timeout: float = 5.0) -> bool:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))
        with self._lock:
            return self._state != "running"


def _default_interfaces() -> List[Dict[str, Any]]:
    raw = _run_text(["ip", "-j", "-4", "addr", "show"])
    try:
        records = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    interfaces: List[Dict[str, Any]] = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, Mapping):
            continue
        ifname = str(record.get("ifname", ""))
        if not ifname or ifname == "lo":
            continue
        for address in record.get("addr_info", []):
            if not isinstance(address, Mapping) or address.get("family") != "inet":
                continue
            local = str(address.get("local", ""))
            prefixlen = address.get("prefixlen")
            try:
                parsed = ipaddress.ip_interface(f"{local}/{int(prefixlen)}")
            except (TypeError, ValueError):
                continue
            if parsed.ip.is_loopback:
                continue
            interfaces.append(
                {
                    "name": ifname,
                    "ipv4": str(parsed.ip),
                    "prefixlen": parsed.network.prefixlen,
                    "network": str(parsed.network),
                    "operstate": str(record.get("operstate", "unknown")).lower(),
                }
            )
    return interfaces


def _default_socket_text() -> str:
    return _run_text(["ss", "-H", "-tnp"])


def default_elements_cache_paths(home: Optional[Path] = None) -> Iterable[Path]:
    """Find passive Elements cache files without assuming an install layout.

    Flexiv RDK requires a serial number before constructing ``Robot`` and has
    no public enumeration API.  Elements Studio retains serials in its logs and
    settings, but installer layouts vary.  Search to a bounded depth for cache
    filenames and recursively inspect only Flexiv/Elements/FSP-named trees.
    Extra roots can be supplied through ``SUPERINFERENCE_ELEMENTS_CACHE_ROOTS``.
    """

    search_roots: List[Path] = [Path(home) if home is not None else Path.home()]
    configured = os.environ.get(_ELEMENTS_SEARCH_ROOTS_ENV, "")
    for value in configured.split(os.pathsep):
        if value.strip():
            search_roots.append(Path(value.strip()).expanduser())

    cache_paths: set[Path] = set()
    candidate_roots: set[Path] = set()
    visited: set[Path] = set()
    queue = deque((root, 0) for root in search_roots)
    skipped_names = {
        ".cache",
        ".git",
        ".npm",
        ".rustup",
        ".cargo",
        "node_modules",
        "miniconda3",
        "anaconda3",
    }

    while queue and len(visited) < 2048:
        current, depth = queue.popleft()
        try:
            resolved = current.resolve()
        except OSError:
            continue
        if resolved in visited or not resolved.is_dir() or resolved.is_symlink():
            continue
        visited.add(resolved)
        name = resolved.name.lower()
        if depth > 0 and any(keyword in name for keyword in _ELEMENTS_DIRECTORY_KEYWORDS):
            candidate_roots.add(resolved)
            continue
        if depth >= 5:
            continue
        try:
            children = list(resolved.iterdir())
        except (OSError, PermissionError):
            continue
        for child in children:
            try:
                if child.is_symlink():
                    continue
                if child.is_file():
                    if child.name in _ELEMENTS_CACHE_FILENAMES or (
                        child.name.startswith("UIApp_") and child.suffix == ".log"
                    ):
                        cache_paths.add(child.resolve())
                    continue
                if not child.is_dir() or child.name in skipped_names:
                    continue
            except (OSError, PermissionError):
                continue
            if depth == 0 and child.name.startswith(".") and child.name != ".local":
                continue
            queue.append((child, depth + 1))

    for root in candidate_roots:
        for pattern in _ELEMENTS_CACHE_PATTERNS:
            try:
                for path in root.rglob(pattern):
                    if path.is_file() and not path.is_symlink():
                        cache_paths.add(path.resolve())
            except (OSError, PermissionError):
                continue

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    logs = sorted(
        (path for path in cache_paths if path.name.startswith("UIApp_")),
        key=modified,
        reverse=True,
    )[:16]
    metadata = sorted(
        (path for path in cache_paths if not path.name.startswith("UIApp_")),
        key=modified,
        reverse=True,
    )[:16]
    return sorted({*logs, *metadata}, key=modified, reverse=True)


# Backward-compatible private name retained for existing provider injections.
_default_elements_logs = default_elements_cache_paths


def _tail_text(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError, PermissionError):
        return ""


def _elements_serial(raw_serial: str) -> Optional[Tuple[str, str]]:
    """Convert Elements' hardware serial to the RDK model serial convention."""
    raw = str(raw_serial).strip()
    match = re.search(r"-(\d{6})$", raw)
    if not match:
        return None
    head = raw[: match.start()]
    if "LMR" in head.upper():
        model = "Rizon4R"
    elif "L" in head.upper():
        model = "Rizon4"
    else:
        return None
    return model, f"{model}-{match.group(1)}"


@dataclass(frozen=True)
class HardwareSnapshot:
    revision: str
    generated_at: str
    interfaces: List[Dict[str, Any]]
    cameras: List[Dict[str, Any]]
    robots: List[Dict[str, Any]]
    providers: List[Dict[str, Any]]
    warnings: List[str]
    robot_probe: Optional[Dict[str, Any]] = None

    def public_payload(self) -> Dict[str, Any]:
        payload = {
            "revision": self.revision,
            "generated_at": self.generated_at,
            "interfaces": copy.deepcopy(self.interfaces),
            "cameras": copy.deepcopy(self.cameras),
            "robots": copy.deepcopy(self.robots),
            "providers": copy.deepcopy(self.providers),
            "warnings": list(self.warnings),
        }
        if self.robot_probe is not None:
            payload["robot_probe"] = copy.deepcopy(self.robot_probe)
        return payload

    def lookup(self, provider_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        for kind, items in (
            ("interface", self.interfaces),
            ("camera", self.cameras),
            ("robot", self.robots),
        ):
            for item in items:
                if item.get("id") == provider_id:
                    return kind, item
        return None


class HardwareDiscovery:
    """Discover hardware with passive inventory and isolated reachability probes."""

    def __init__(
        self,
        *,
        sys_video_root: Path = Path("/sys/class/video4linux"),
        dev_root: Path = Path("/dev"),
        by_id_root: Path = Path("/dev/v4l/by-id"),
        by_path_root: Path | None = None,
        interface_provider: Callable[[], List[Dict[str, Any]]] = _default_interfaces,
        socket_provider: Callable[[], str] = _default_socket_text,
        elements_log_provider: Callable[
            [], Iterable[Path]
        ] = default_elements_cache_paths,
        camera_frame_probe: Optional[
            Callable[[Iterable[str]], Mapping[str, Mapping[str, object]]]
        ] = None,
        robot_reachability_probe: Optional[
            Callable[[Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
        robot_probe_workers: int = _ROBOT_PROBE_WORKERS,
    ) -> None:
        self.sys_video_root = Path(sys_video_root)
        self.dev_root = Path(dev_root)
        self.by_id_root = Path(by_id_root)
        self.by_path_root = (
            Path(by_path_root)
            if by_path_root is not None
            else self.dev_root / "v4l" / "by-path"
        )
        self.interface_provider = interface_provider
        self.socket_provider = socket_provider
        self.elements_log_provider = elements_log_provider
        self.camera_frame_probe = camera_frame_probe or CameraFrameProbe().probe
        self.robot_reachability = RobotReachabilityCoordinator(
            robot_reachability_probe or _default_robot_reachability_probe,
            max_workers=robot_probe_workers,
        )

    def discover(self) -> HardwareSnapshot:
        return self._discover(probe_camera_frames=False)

    def discover_with_camera_probe(self) -> HardwareSnapshot:
        """Discover hardware and fail closed unless a camera yields one frame."""

        return self._project_robot_reachability(
            self._discover(probe_camera_frames=True)
        )

    def discover_with_robot_probe(
        self,
        *,
        probe_camera_frames: bool = False,
        force: bool = False,
    ) -> HardwareSnapshot:
        """Start/reuse a background robot probe and return its latest snapshot."""

        snapshot = self._discover(probe_camera_frames=probe_camera_frames)
        self.robot_reachability.start(snapshot.robots, force=force)
        return self._project_robot_reachability(snapshot)

    def _project_robot_reachability(
        self, snapshot: HardwareSnapshot
    ) -> HardwareSnapshot:
        robots, progress = self.robot_reachability.project(snapshot.robots)
        if progress.get("state") == "idle":
            return snapshot
        providers = [*snapshot.providers]
        providers.append(
            {
                "name": "icmp_robot_probe",
                "mode": "active_read_only",
                "state": progress.get("state"),
            }
        )
        canonical = json.dumps(
            {
                "interfaces": snapshot.interfaces,
                "cameras": snapshot.cameras,
                "robots": robots,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return HardwareSnapshot(
            revision=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            generated_at=snapshot.generated_at,
            interfaces=snapshot.interfaces,
            cameras=snapshot.cameras,
            robots=robots,
            providers=providers,
            warnings=snapshot.warnings,
            robot_probe=progress,
        )

    def _discover(self, *, probe_camera_frames: bool) -> HardwareSnapshot:
        warnings: List[str] = []
        robots = self._discover_robots(warnings)
        interfaces = self._discover_interfaces(robots, warnings)
        cameras = self._discover_cameras(warnings)
        providers = [
            {"name": "linux_interfaces", "mode": "read_only"},
            {"name": "sysfs_v4l2", "mode": "read_only"},
            {
                "name": "elements_cache",
                "mode": "passive_cache",
                "confidence": "cached",
            },
        ]
        if probe_camera_frames:
            cameras = self._probe_camera_frames(cameras, warnings)
            providers.append(
                {
                    "name": "v4l2_frame_probe",
                    "mode": "active_isolated",
                }
            )
        canonical = json.dumps(
            {
                "interfaces": interfaces,
                "cameras": cameras,
                "robots": robots,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return HardwareSnapshot(
            revision=revision,
            generated_at=_utc_now(),
            interfaces=interfaces,
            cameras=cameras,
            robots=robots,
            providers=providers,
            warnings=warnings,
        )

    def _probe_camera_frames(
        self,
        cameras: List[Dict[str, Any]],
        warnings: List[str],
    ) -> List[Dict[str, Any]]:
        probe_paths = [
            str(camera.get("stable_path") or camera["devnode"])
            for camera in cameras
        ]
        try:
            raw_results = self.camera_frame_probe(probe_paths)
        except Exception as exc:
            raw_results = {}
            warnings.append(
                "Active camera frame probe failed: "
                f"{type(exc).__name__}: {exc}"
            )
        probed: List[Dict[str, Any]] = []
        for camera, path in zip(cameras, probe_paths):
            result = raw_results.get(path, {})
            reachable = result.get("reachable") is True
            status = str(result.get("status") or "probe_unavailable")
            item = dict(camera)
            item["capture_capable"] = reachable
            item["frame_reachable"] = reachable
            item["probe_status"] = status
            item["confidence"] = "frame_probe" if reachable else "unreachable"
            probed.append(item)
            if not reachable:
                warnings.append(
                    f"Camera {camera['devnode']} did not produce a frame ({status})"
                )
        return probed

    def _discover_interfaces(
        self, robots: List[Dict[str, Any]], warnings: List[str]
    ) -> List[Dict[str, Any]]:
        robot_ips = []
        for robot in robots:
            try:
                robot_ips.append(ipaddress.ip_address(str(robot["ip"])))
            except (KeyError, ValueError):
                continue
        interfaces: List[Dict[str, Any]] = []
        try:
            raw_interfaces = self.interface_provider()
        except Exception:
            raw_interfaces = []
        for raw in raw_interfaces:
            try:
                interface = ipaddress.ip_interface(
                    f"{raw['ipv4']}/{int(raw['prefixlen'])}"
                )
            except (KeyError, TypeError, ValueError):
                continue
            item = {
                "id": _opaque_id("interface", raw.get("name"), interface),
                "name": str(raw.get("name", "unknown")),
                "ipv4": str(interface.ip),
                "prefixlen": interface.network.prefixlen,
                "network": str(interface.network),
                "operstate": str(raw.get("operstate", "unknown")),
                "robot_facing_candidate": any(
                    robot_ip in interface.network for robot_ip in robot_ips
                ),
                "source": "linux_interfaces",
            }
            interfaces.append(item)
        interfaces.sort(key=lambda item: (not item["robot_facing_candidate"], item["name"], item["ipv4"]))
        if not interfaces:
            warnings.append("No non-loopback IPv4 interfaces were discovered")
        elif robot_ips and not any(item["robot_facing_candidate"] for item in interfaces):
            warnings.append("No local IPv4 interface shares a subnet with cached robots")
        return interfaces

    def _stable_links(self) -> Dict[str, List[str]]:
        links: Dict[str, List[str]] = {}
        for root in (self.by_id_root, self.by_path_root):
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir(), key=lambda value: value.name):
                try:
                    resolved = str(path.resolve(strict=True))
                except (FileNotFoundError, OSError):
                    continue
                links.setdefault(resolved, []).append(str(path))
        for values in links.values():
            values.sort()
        return links

    def _usb_attributes(self, sysfs_path: Path) -> Dict[str, Optional[str]]:
        attributes = {
            "vendor_id": None,
            "product_id": None,
            "serial": None,
            "manufacturer": None,
            "product": None,
        }
        filenames = {
            "vendor_id": "idVendor",
            "product_id": "idProduct",
            "serial": "serial",
            "manufacturer": "manufacturer",
            "product": "product",
        }
        for parent in (sysfs_path, *sysfs_path.parents):
            for key, filename in filenames.items():
                if attributes[key] is None:
                    attributes[key] = _read_text(parent / filename)
            if attributes["vendor_id"] and attributes["product_id"]:
                break
            if parent == Path("/sys"):
                break
        return attributes

    def _discover_cameras(self, warnings: List[str]) -> List[Dict[str, Any]]:
        cameras: List[Dict[str, Any]] = []
        stable_links = self._stable_links()
        if not self.sys_video_root.is_dir():
            warnings.append("/sys/class/video4linux is unavailable")
            return cameras
        for entry in sorted(self.sys_video_root.glob("video*"), key=lambda path: path.name):
            index_text = _read_text(entry / "index")
            # V4L2 USB cameras commonly expose metadata as index1. The capture
            # endpoint is index0; never open either endpoint during discovery.
            if index_text != "0":
                continue
            devnode = self.dev_root / entry.name
            resolved_devnode = str(devnode.resolve())
            links = stable_links.get(resolved_devnode, [])
            index0_links = [link for link in links if "video-index0" in link]
            stable_path = (index0_links or links or [None])[0]
            try:
                sysfs_realpath = entry.resolve(strict=True)
            except (FileNotFoundError, OSError):
                sysfs_realpath = entry.resolve()
            usb = self._usb_attributes(sysfs_realpath)
            identity_parts = [
                str(usb.get(key) or "")
                for key in ("vendor_id", "product_id", "serial", "product")
            ]
            usb_identity = "|".join(identity_parts) if any(identity_parts) else ""
            identity = stable_path or usb_identity or resolved_devnode
            item = {
                "id": _opaque_id("camera", identity),
                "devnode": str(devnode),
                "realpath": resolved_devnode,
                "sysfs_realpath": str(sysfs_realpath),
                "name": _read_text(entry / "name") or entry.name,
                "v4l2_index": 0,
                "capture_capable": True,
                "stable_path": stable_path,
                "usb": usb,
                "source": "sysfs_v4l2",
                "confidence": "sysfs",
            }
            cameras.append(item)
        cameras.sort(key=lambda item: _video_number(item["devnode"]))
        if not cameras:
            warnings.append("No capture-capable V4L2 index0 cameras were discovered")
        for camera in cameras:
            if not camera.get("stable_path"):
                warnings.append(
                    f"Camera {camera['devnode']} has no stable /dev/v4l/by-id link"
                )
        return cameras

    def _discover_robots(self, warnings: List[str]) -> List[Dict[str, Any]]:
        socket_text = ""
        try:
            socket_text = self.socket_provider() or ""
        except Exception:
            pass
        active_peers = set(
            re.findall(r"(?:\[::ffff:)?(\d+\.\d+\.\d+\.\d+)\]?:\d+", socket_text)
        )
        latest: Dict[str, Dict[str, Any]] = {}
        cache_count = 0
        try:
            paths = list(self.elements_log_provider())
        except Exception:
            paths = []
        for path in paths:
            text = _tail_text(Path(path))
            if not text:
                continue
            cache_count += 1
            for match in _ELEMENTS_ROBOT_RE.finditer(text):
                converted = _elements_serial(match.group("serial"))
                if converted is None:
                    continue
                model, serial = converted
                try:
                    ip = str(ipaddress.ip_address(match.group("ip")))
                except ValueError:
                    continue
                timestamp = match.group("timestamp")
                previous = latest.get(serial)
                if (
                    previous
                    and previous.get("ip")
                    and str(previous.get("last_seen", "")) > timestamp
                ):
                    continue
                latest[serial] = {
                    "id": _opaque_id("robot", serial),
                    "model": model,
                    "serial_number": serial,
                    "rdk_sn": serial,
                    "variant": "4r" if model == "Rizon4R" else "standard",
                    "ip": ip,
                    "source": "elements_cache",
                    "confidence": "cached",
                    "last_seen": timestamp,
                    "peer_active": ip in active_peers,
                }
            for raw_serial in _ELEMENTS_CACHE_SERIAL_RE.findall(text):
                converted = _elements_serial(raw_serial)
                if converted is None:
                    continue
                model, serial = converted
                if serial in latest:
                    continue
                try:
                    last_seen = datetime.fromtimestamp(
                        Path(path).stat().st_mtime, timezone.utc
                    ).isoformat()
                except OSError:
                    last_seen = ""
                latest[serial] = {
                    "id": _opaque_id("robot", serial),
                    "model": model,
                    "serial_number": serial,
                    "rdk_sn": serial,
                    "variant": "4r" if model == "Rizon4R" else "standard",
                    "ip": None,
                    "source": "elements_cache",
                    "confidence": "cached",
                    "last_seen": last_seen,
                    "peer_active": False,
                }
        robots = sorted(latest.values(), key=lambda item: (item["model"], item["serial_number"]))
        if not paths:
            warnings.append("Flexiv Elements cache was not found; robots were not enumerated")
        elif not robots:
            warnings.append("Flexiv Elements cache contained no supported Rizon4 records")
        else:
            warnings.append(
                "Robot inventory is cached/passive and does not prove current RDK availability"
            )
        if cache_count == 0 and paths:
            warnings.append("Flexiv Elements cache files could not be read")
        return robots


def _video_number(devnode: str) -> int:
    match = re.search(r"video(\d+)$", str(devnode))
    return int(match.group(1)) if match else 1_000_000


def canonical_device_key(device: Mapping[str, Any]) -> str:
    return f"{device.get('class')}:{device.get('device_class_id')}"


def _normalized_rizon_serial(value: Any) -> Optional[Tuple[str, str]]:
    text = re.sub(r"\s+", "", str(value or ""))
    match = _RIZON_SERIAL_RE.fullmatch(text)
    if not match:
        return None
    model = "Rizon4R" if match.group("right") else "Rizon4"
    return model, f"{model}-{match.group('number')}"


def _devices(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = config.get("devices", [])
    if not isinstance(raw, list):
        return []
    return [device for device in raw if isinstance(device, dict)]


def recommend_overrides(
    config: Mapping[str, Any], snapshot: HardwareSnapshot
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    """Return opaque recommendations plus configured detection diagnostics."""
    recommendations: Dict[str, Dict[str, str]] = {}
    configured: Dict[str, Dict[str, Any]] = {}
    robots_by_serial = {
        str(robot.get("rdk_sn", robot.get("serial_number"))): robot
        for robot in snapshot.robots
        if robot.get("rdk_sn", robot.get("serial_number"))
    }
    robot_interfaces = [
        interface
        for interface in snapshot.interfaces
        if interface.get("robot_facing_candidate")
    ]

    webcam_devices = [
        device
        for device in _devices(config)
        if str(device.get("class")) in _WEBCAM_CLASSES
        and device.get("enabled", True) is not False
    ]
    camera_by_number = {
        _video_number(str(camera.get("devnode"))): camera
        for camera in snapshot.cameras
        if camera.get("capture_capable") is not False
    }
    used_camera_ids: set[str] = set()
    assigned: Dict[str, Dict[str, Any]] = {}
    for device in webcam_devices:
        key = canonical_device_key(device)
        try:
            number = int(device.get("capture_id"))
        except (TypeError, ValueError):
            continue
        camera = camera_by_number.get(number)
        if (
            camera
            and str(camera["id"]) not in used_camera_ids
        ):
            assigned[key] = camera
            used_camera_ids.add(str(camera["id"]))
    available_cameras = [
        camera
        for camera in snapshot.cameras
        if camera.get("capture_capable") is not False
        and (camera.get("stable_path") or camera.get("devnode"))
        and str(camera["id"]) not in used_camera_ids
    ]
    for device in webcam_devices:
        key = canonical_device_key(device)
        camera = assigned.get(key)
        configured[key] = {
            "kind": "camera",
            "configured_device_class_id": device.get("device_class_id"),
            "detected": camera is not None,
        }
        if camera is None and available_cameras:
            camera = available_cameras.pop(0)
            used_camera_ids.add(str(camera["id"]))
        if camera is not None:
            recommendations[key] = {"capture_device": str(camera["id"])}

    for device in _devices(config):
        if str(device.get("class")) != "RizonRobot" or device.get("enabled", True) is False:
            continue
        key = canonical_device_key(device)
        raw_serials = device.get("robot_sn")
        serials = list(raw_serials) if isinstance(raw_serials, list) else [raw_serials]
        normalized = [_normalized_rizon_serial(value) for value in serials]
        detected_flags = [
            bool(item and item[1] in robots_by_serial) for item in normalized
        ]
        whitelist = device.get("network_interface_whitelist") or []
        if isinstance(whitelist, str):
            whitelist = [whitelist]
        local_ips = {str(item.get("ipv4")) for item in snapshot.interfaces}
        configured[key] = {
            "kind": "robot",
            "configured_serials": [item[1] if item else None for item in normalized],
            "detected": detected_flags,
            "configured_interfaces_local": [
                str(value) in local_ips for value in whitelist
            ],
        }
        binding: Dict[str, str] = {}
        configured_sides = (
            ("robot_left", normalized[0] if normalized else None),
            ("robot_right", normalized[1] if len(normalized) > 1 else None),
        )
        for field, configured_robot in configured_sides:
            if configured_robot is None:
                continue
            robot = robots_by_serial.get(configured_robot[1])
            if robot is not None:
                binding[field] = str(robot["id"])
        if robot_interfaces:
            binding["network_interface"] = str(robot_interfaces[0]["id"])
        if binding:
            recommendations[key] = binding
    return recommendations, configured


def _binding_public(kind: str, item: Mapping[str, Any]) -> Dict[str, Any]:
    if kind == "robot":
        return {
            "id": item.get("id"),
            "model": item.get("model"),
            "serial_number": item.get("serial_number"),
            "rdk_sn": item.get("rdk_sn", item.get("serial_number")),
            "variant": item.get("variant") or (
                "4r" if item.get("model") == "Rizon4R" else "standard"
            ),
            "ip": item.get("ip"),
            "source": item.get("source"),
            "confidence": item.get("confidence"),
        }
    if kind == "interface":
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "ipv4": item.get("ipv4"),
            "network": item.get("network"),
            "source": item.get("source"),
        }
    return {
        "id": item.get("id"),
        "devnode": item.get("devnode"),
        "stable_path": item.get("stable_path"),
        "name": item.get("name"),
        "source": item.get("source"),
    }


def apply_discovery_overrides(
    config: Mapping[str, Any],
    overrides: Any,
    snapshot: HardwareSnapshot,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Resolve opaque bindings into a deep-copied effective runtime config."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be an object keyed by canonical device key")
    effective = copy.deepcopy(dict(config))
    devices = _devices(effective)
    by_key = {canonical_device_key(device): device for device in devices}
    bindings: Dict[str, Dict[str, Any]] = {}
    camera_assignments: Dict[str, str] = {}

    for key, raw_binding in overrides.items():
        if key not in by_key:
            raise ValueError(f"unknown override device key: {key}")
        if not isinstance(raw_binding, Mapping):
            raise ValueError(f"override for {key} must be an object")
        device = by_key[key]
        device_class = str(device.get("class"))
        if device_class == "RizonRobot":
            allowed = {"robot_left", "robot_right", "network_interface"}
        elif device_class in _WEBCAM_CLASSES:
            allowed = {"capture_device"}
        else:
            raise ValueError(f"device does not support runtime bindings: {key}")
        unexpected = set(raw_binding) - allowed
        if unexpected:
            raise ValueError(
                f"unsupported override fields for {key}: {sorted(unexpected)}"
            )
        resolved: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for field, provider_id in raw_binding.items():
            if provider_id is None and field in {"robot_left", "robot_right"}:
                # A missing recommendation is represented as null by the UI.
                # Preserve the composed serial for that side rather than
                # treating null as an arbitrary hardware identifier.
                continue
            if (
                provider_id is None
                and field == "capture_device"
                and device_class in _WEBCAM_CLASSES
            ):
                resolved[field] = (CAPTURE_BINDING_MISSING, {})
                continue
            if not isinstance(provider_id, str):
                raise ValueError(f"override {key}.{field} must be an opaque id")
            match = snapshot.lookup(provider_id)
            if match is None:
                raise ValueError(f"unknown or stale opaque id for {key}.{field}")
            resolved[field] = match

        public: Dict[str, Any] = {}
        if device_class == "RizonRobot":
            serials = device.get("robot_sn")
            serial_list = list(serials) if isinstance(serials, list) else [serials]
            has_robot_override = any(
                field in resolved for field in ("robot_left", "robot_right")
            )
            if has_robot_override and len(serial_list) != 2:
                raise ValueError(f"runtime robot binding requires dual-arm robot_sn: {key}")
            for field, index, expected_model in (
                ("robot_left", 0, "Rizon4"),
                ("robot_right", 1, "Rizon4R"),
            ):
                if field not in resolved:
                    continue
                kind, item = resolved[field]
                expected_variant = "4r" if expected_model == "Rizon4R" else "standard"
                if kind != "robot" or not (
                    item.get("variant") == expected_variant
                    or item.get("model") == expected_model
                ):
                    raise ValueError(f"{key}.{field} must reference {expected_model}")
                rdk_sn = item.get("rdk_sn", item.get("serial_number"))
                if not isinstance(rdk_sn, str) or not rdk_sn:
                    raise ValueError(f"{key}.{field} has no RDK serial")
                serial_list[index] = rdk_sn
                public[field] = _binding_public(kind, item)
            if "network_interface" in resolved:
                kind, item = resolved["network_interface"]
                if kind != "interface":
                    raise ValueError(f"{key}.network_interface must reference an interface")
                local_ips = {str(value.get("ipv4")) for value in snapshot.interfaces}
                if str(item.get("ipv4")) not in local_ips:
                    raise ValueError(f"{key}.network_interface is not a local IPv4")
                device["network_interface_whitelist"] = [str(item["ipv4"])]
                public["network_interface"] = _binding_public(kind, item)
            if has_robot_override:
                device["robot_sn"] = serial_list
        else:
            if "capture_device" not in resolved:
                continue
            kind, item = resolved["capture_device"]
            if kind == CAPTURE_BINDING_MISSING:
                device[CAPTURE_BINDING_STATE_KEY] = CAPTURE_BINDING_MISSING
                public["capture_device"] = {
                    "id": None,
                    "status": CAPTURE_BINDING_MISSING,
                    "detected": False,
                }
                bindings[str(key)] = public
                continue
            if kind != "camera" or not item.get("capture_capable"):
                raise ValueError(f"{key}.capture_device must reference a capture camera")
            stable_path = item.get("stable_path")
            devnode = item.get("devnode")
            capture_path = stable_path or devnode
            if not isinstance(capture_path, str) or not capture_path:
                raise ValueError(f"{key}.capture_device has no usable device path")
            camera_id = str(item["id"])
            previous_key = camera_assignments.get(camera_id)
            if previous_key and previous_key != key:
                raise ValueError(
                    f"camera binding is duplicated by {previous_key} and {key}"
                )
            camera_assignments[camera_id] = key
            # Duplicate USB serials do not always receive distinct by-id
            # symlinks. The opaque discovery id and revision still bind this
            # session to the exact snapshot, so the discovered devnode is a
            # valid fallback until hardware is re-enumerated.
            device["capture_id"] = capture_path
            device.pop(CAPTURE_BINDING_STATE_KEY, None)
            public["capture_device"] = _binding_public(kind, item)
        if public:
            bindings[str(key)] = public

    # Reject an override that collides with a non-overridden camera's current
    # devnode. Logical device_class_id remains unchanged because it defines SHM keys.
    camera_by_number = {
        _video_number(str(camera.get("devnode"))): str(camera.get("id"))
        for camera in snapshot.cameras
    }
    used: Dict[str, str] = dict(camera_assignments)
    for device in devices:
        key = canonical_device_key(device)
        if str(device.get("class")) not in _WEBCAM_CLASSES or key in bindings:
            continue
        capture = device.get("capture_id")
        try:
            camera_id = camera_by_number.get(int(capture))
        except (TypeError, ValueError):
            camera_id = next(
                (
                    str(camera["id"])
                    for camera in snapshot.cameras
                    if capture in {camera.get("stable_path"), camera.get("devnode")}
                ),
                None,
            )
        if camera_id:
            previous = used.get(camera_id)
            if previous and previous != key:
                raise ValueError(f"camera binding is duplicated by {previous} and {key}")
            used[camera_id] = key

    validate_effective_hardware_config(effective, snapshot)
    return effective, bindings


def validate_effective_hardware_config(
    config: Mapping[str, Any], snapshot: HardwareSnapshot
) -> None:
    """Validate binding structure without requiring hardware to be present.

    Selecting a config creates no device process, so a serial, camera, or
    network interface that is absent on the current host must not prevent the
    selection. Explicit discovery overrides are checked against ``snapshot``
    while they are resolved above; device startup remains the point where the
    composed host settings must actually work.
    """
    del snapshot
    for device in _devices(config):
        if str(device.get("class")) != "RizonRobot" or device.get("enabled", True) is False:
            continue
        # Generic/hardware-free Rizon test configs may omit robot_sn entirely.
        # Validate side/model ordering whenever a serial is actually configured
        # or supplied by an override, without inventing a hardware requirement.
        if device.get("robot_sn") is not None:
            raw = device.get("robot_sn")
            serials = list(raw) if isinstance(raw, list) else [raw]
            if len(serials) == 2:
                left = _normalized_rizon_serial(serials[0])
                right = _normalized_rizon_serial(serials[1])
                if not left or left[0] != "Rizon4":
                    raise ValueError("left robot_sn must be a valid Rizon4 serial")
                if not right or right[0] != "Rizon4R":
                    raise ValueError("right robot_sn must be a valid Rizon4R serial")
                device["robot_sn"] = [left[1], right[1]]
            elif len(serials) == 1:
                normalized = _normalized_rizon_serial(serials[0])
                if not normalized:
                    raise ValueError("robot_sn must be a valid Rizon4/Rizon4R serial")
                device["robot_sn"] = normalized[1]
            else:
                raise ValueError("robot_sn must contain one or two serials")


__all__ = [
    "CAPTURE_BINDING_MISSING",
    "CAPTURE_BINDING_STATE_KEY",
    "HardwareDiscovery",
    "HardwareSnapshot",
    "apply_discovery_overrides",
    "canonical_device_key",
    "capture_binding_is_missing",
    "default_elements_cache_paths",
    "recommend_overrides",
    "validate_effective_hardware_config",
]
