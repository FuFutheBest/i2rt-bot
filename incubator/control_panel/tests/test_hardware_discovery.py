"""Hardware-free passive discovery and runtime-binding contract tests."""

from __future__ import annotations

import builtins
import copy
from dataclasses import replace
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from incubator.control_panel import server
from incubator.control_panel.hardware_discovery import (
    CAPTURE_BINDING_STATE_KEY,
    HardwareDiscovery,
    HardwareSnapshot,
    RobotReachabilityCoordinator,
    _default_elements_logs,
    apply_discovery_overrides,
    recommend_overrides,
)
from incubator.control_panel.runtime_config import compose_runtime_config


class IdleLifecycle:
    """Process-free lifecycle double that retains the effective config."""

    instances: List['IdleLifecycle'] = []

    def __init__(self, config: Dict[str, Any], **_: Any) -> None:
        self.config = copy.deepcopy(config)
        self.active_keys: set[str] = set()
        self.external_keys: set[str] = set()
        self.monitor_token = object()
        type(self).instances.append(self)

    def start_monitor(self) -> None:
        return None

    def has_active_devices(self) -> bool:
        return bool(self.active_keys or self.external_keys)

    def get_device_status(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        key = f'{device_class}:{device_class_id}'
        return next(
            device
            for device in self.get_status()['devices']
            if device['key'] == key
        )

    def start_device(self, device_class: str, device_class_id: Any) -> Dict[str, Any]:
        key = f'{device_class}:{device_class_id}'
        self.active_keys.add(key)
        return self.get_device_status(device_class, device_class_id)

    def stop_all(self) -> Dict[str, Any]:
        stopped = sorted(self.active_keys)
        self.active_keys.clear()
        return {
            'ok': True,
            'devices': self.get_status()['devices'],
            'stopped_devices': stopped,
            'incomplete_devices': [],
            'external_devices': sorted(self.external_keys),
        }

    def get_status(self) -> Dict[str, Any]:
        devices = [
            {
                'key': f"{device['class']}:{device['device_class_id']}",
                'class': device['class'],
                'id': device['device_class_id'],
                'running': f"{device['class']}:{device['device_class_id']}" in self.active_keys,
                'ready': f"{device['class']}:{device['device_class_id']}" in self.active_keys,
                'status': (
                    'running'
                    if f"{device['class']}:{device['device_class_id']}" in self.active_keys
                    else (
                        'external'
                        if f"{device['class']}:{device['device_class_id']}"
                        in self.external_keys
                        else 'stopped'
                    )
                ),
                'desired_state': (
                    'running'
                    if f"{device['class']}:{device['device_class_id']}" in self.active_keys
                    else 'stopped'
                ),
                'pid': (
                    44000 + int(device['device_class_id'])
                    if f"{device['class']}:{device['device_class_id']}" in self.active_keys
                    else None
                ),
                'shm': {
                    'name': f"{device['class']}_{device['device_class_id']}_data",
                    'connected': (
                        f"{device['class']}:{device['device_class_id']}" in self.active_keys
                        or f"{device['class']}:{device['device_class_id']}"
                        in self.external_keys
                    ),
                    'fresh': (
                        f"{device['class']}:{device['device_class_id']}" in self.active_keys
                        or f"{device['class']}:{device['device_class_id']}"
                        in self.external_keys
                    ),
                },
                'capabilities': {'preview': 'Camera' in str(device['class'])},
            }
            for device in self.config.get('devices', [])
            if device.get('enabled', True) is not False
        ]
        return {
            'running': bool(self.active_keys),
            'monitor_running': True,
            'total_devices': len(devices),
            'enable_visualizers': False,
            'enable_device_manager': False,
            'enable_policy_runner': False,
            'device_manager_running': False,
            'policy_runner_running': False,
            'devices': devices,
        }


def test_elements_cache_search_is_keyword_driven_and_layout_independent(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / 'vendor-suite' / 'custom-elements-install'
    log = cache_root / 'deep' / 'logs' / 'UIApp_recent.log'
    log.parent.mkdir(parents=True)
    log.write_text(
        '[2026-07-12 15:15:15.640054][debug] will connect to '
        'ip = 192.168.2.102, sn.toStdString() = A02L-10-M4-063087\n',
        encoding='utf-8',
    )
    metadata = (
        cache_root
        / 'different'
        / 'settings'
        / 'generated_robot_custom.json'
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        '{"manipulator":{"armSerialNumber":"A02LMR-06-M1-062071"}}\n',
        encoding='utf-8',
    )

    paths = list(_default_elements_logs(tmp_path))

    assert log.resolve() in paths
    assert metadata.resolve() in paths
    discovery = HardwareDiscovery(
        sys_video_root=tmp_path / 'missing-sysfs',
        dev_root=tmp_path / 'missing-dev',
        by_id_root=tmp_path / 'missing-by-id',
        interface_provider=lambda: [
            {
                'name': 'enp6s0',
                'ipv4': '192.168.2.10',
                'prefixlen': 24,
                'operstate': 'up',
            }
        ],
        socket_provider=lambda: '',
        elements_log_provider=lambda: paths,
    )
    robots = discovery.discover().robots
    by_serial = {robot['rdk_sn']: robot for robot in robots}
    assert by_serial['Rizon4-063087']['ip'] == '192.168.2.102'
    assert by_serial['Rizon4R-062071']['ip'] is None
    assert all(robot['confidence'] == 'cached' for robot in robots)


class StaticDiscovery:
    def __init__(self, snapshot: HardwareSnapshot) -> None:
        self.snapshot = snapshot

    def discover(self) -> HardwareSnapshot:
        return self.snapshot

    def discover_with_camera_probe(self) -> HardwareSnapshot:
        return self.snapshot


class ChangingCameraDiscovery(StaticDiscovery):
    """Return a different active revision if a binding probes a second time."""

    def __init__(
        self,
        passive: HardwareSnapshot,
        active_snapshots: List[HardwareSnapshot],
    ) -> None:
        super().__init__(passive)
        self.active_snapshots = active_snapshots
        self.active_calls = 0

    def discover_with_camera_probe(self) -> HardwareSnapshot:
        index = min(self.active_calls, len(self.active_snapshots) - 1)
        self.active_calls += 1
        return self.active_snapshots[index]


class RobotProbeDiscovery(StaticDiscovery):
    def __init__(
        self,
        passive: HardwareSnapshot,
        active: HardwareSnapshot,
    ) -> None:
        super().__init__(passive)
        self.active = active
        self.calls: List[tuple[bool, bool]] = []

    def discover_with_robot_probe(
        self,
        *,
        probe_camera_frames: bool = False,
        force: bool = False,
    ) -> HardwareSnapshot:
        self.calls.append((probe_camera_frames, force))
        return self.active


def _robot_candidate(serial: str, ip: str) -> Dict[str, Any]:
    model = "Rizon4R" if serial.startswith("Rizon4R-") else "Rizon4"
    return {
        "id": f"robot-{serial}",
        "model": model,
        "serial_number": serial,
        "rdk_sn": serial,
        "variant": "4r" if model == "Rizon4R" else "standard",
        "ip": ip,
        "source": "elements_cache",
        "confidence": "cached",
    }


def test_robot_probe_runs_in_background_with_bounded_parallelism() -> None:
    release = threading.Event()
    two_workers_entered = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def probe(robot: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_workers_entered.set()
        release.wait(2.0)
        with lock:
            active -= 1
        return {
            "reachable": robot["rdk_sn"] != "Rizon4-000003",
            "status": "reachable",
        }

    coordinator = RobotReachabilityCoordinator(probe, max_workers=2)
    robots = [
        _robot_candidate("Rizon4-000001", "192.168.2.101"),
        _robot_candidate("Rizon4R-000002", "192.168.2.102"),
        _robot_candidate("Rizon4-000003", "192.168.2.103"),
    ]

    started = time.monotonic()
    coordinator.start(robots)
    assert time.monotonic() - started < 0.1
    assert two_workers_entered.wait(1.0)
    projected, progress = coordinator.project(robots)
    assert projected == []
    assert progress["state"] == "running"
    assert progress["total"] == 3
    assert maximum_active == 2

    release.set()
    assert coordinator.wait(2.0)
    projected, progress = coordinator.project(robots)
    assert progress["state"] == "complete"
    assert progress["total"] == 3
    assert progress["completed"] == 3
    assert progress["reachable"] == 2
    assert {robot["rdk_sn"] for robot in projected} == {
        "Rizon4-000001",
        "Rizon4R-000002",
    }
    assert all(robot["confidence"] == "ping_probe" for robot in projected)


def test_robot_probe_preserves_last_successes_during_refresh() -> None:
    refresh_release = threading.Event()
    phase = {"refresh": False}

    def probe(robot: Dict[str, Any]) -> Dict[str, Any]:
        if phase["refresh"]:
            refresh_release.wait(2.0)
            return {
                "reachable": robot["rdk_sn"] == "Rizon4-000001",
                "status": "reachable",
            }
        return {"reachable": True, "status": "reachable"}

    coordinator = RobotReachabilityCoordinator(probe, max_workers=2)
    robots = [
        _robot_candidate("Rizon4-000001", "192.168.2.101"),
        _robot_candidate("Rizon4R-000002", "192.168.2.102"),
    ]
    coordinator.start(robots)
    assert coordinator.wait(2.0)
    assert len(coordinator.project(robots)[0]) == 2

    phase["refresh"] = True
    coordinator.start(robots, force=True)
    projected, progress = coordinator.project(robots)
    assert progress["state"] == "running"
    assert len(projected) == 2

    refresh_release.set()
    assert coordinator.wait(2.0)
    projected, progress = coordinator.project(robots)
    assert progress["state"] == "complete"
    assert [robot["rdk_sn"] for robot in projected] == ["Rizon4-000001"]


def test_robot_probe_route_is_async_capable_and_cached_for_config_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "robot.yaml").write_text(
        """devices:
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-000001, Rizon4R-000002]
    network_interface_whitelist: [192.168.2.10]
    data_shape: [44]
    data_dtype: float64
""",
        encoding="utf-8",
    )
    robots = [
        _robot_candidate("Rizon4-000001", "192.168.2.101"),
        _robot_candidate("Rizon4R-000002", "192.168.2.102"),
    ]
    passive = HardwareSnapshot(
        revision="passive-revision",
        generated_at="2026-08-06T00:00:00+00:00",
        interfaces=[],
        cameras=[],
        robots=robots,
        providers=[],
        warnings=[],
    )
    active = replace(
        passive,
        revision="reachable-revision",
        robot_probe={
            "state": "complete",
            "total": 2,
            "completed": 2,
            "reachable": 2,
        },
    )
    discovery = RobotProbeDiscovery(passive, active)
    IdleLifecycle.instances = []
    monkeypatch.setattr(server, "DeviceLifecycleManager", IdleLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=discovery,
    )
    app.testing = True
    client = app.test_client()

    dropdown = client.get(
        "/api/hardware/discovery?config_name=robot&probe_cameras=1&"
        "probe_robots=1&refresh_robots=1"
    )
    assert dropdown.status_code == 200
    payload = dropdown.get_json()
    assert payload["revision"] == "reachable-revision"
    assert payload["robot_probe"]["reachable"] == 2
    assert discovery.calls == [(True, True)]

    selected = client.post(
        "/api/config/select",
        json={
            "name": "robot",
            "discovery_revision": payload["revision"],
            "overrides": {
                "RizonRobot:2": {
                    "robot_left": robots[0]["id"],
                    "robot_right": robots[1]["id"],
                }
            },
        },
    )
    assert selected.status_code == 200
    assert selected.get_json()["discovery_revision"] == "reachable-revision"


def _write_video_endpoint(
    sys_video_root: Path,
    dev_root: Path,
    name: str,
    *,
    index: int,
    product: str,
    serial: str,
) -> Path:
    endpoint = sys_video_root / name
    endpoint.mkdir(parents=True, exist_ok=True)
    (endpoint / 'index').write_text(str(index), encoding='utf-8')
    (endpoint / 'name').write_text(product, encoding='utf-8')
    (endpoint / 'idVendor').write_text('1234', encoding='utf-8')
    (endpoint / 'idProduct').write_text('5678', encoding='utf-8')
    (endpoint / 'serial').write_text(serial, encoding='utf-8')
    (endpoint / 'manufacturer').write_text('Noematrix Test', encoding='utf-8')
    (endpoint / 'product').write_text(product, encoding='utf-8')
    devnode = dev_root / name
    devnode.parent.mkdir(parents=True, exist_ok=True)
    devnode.touch()
    return devnode


def _link_by_id(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - Windows without symlink rights
        pytest.skip(f'symlink support is required for by-id discovery: {exc}')


def test_active_camera_discovery_requires_a_real_frame_and_passive_does_not_probe(
    tmp_path: Path,
) -> None:
    sys_video_root = tmp_path / 'sys' / 'class' / 'video4linux'
    dev_root = tmp_path / 'dev'
    by_id_root = dev_root / 'v4l' / 'by-id'
    by_path_root = dev_root / 'v4l' / 'by-path'
    _write_video_endpoint(
        sys_video_root,
        dev_root,
        'video0',
        index=0,
        product='Reachable Camera',
        serial='CAM-OK',
    )
    _write_video_endpoint(
        sys_video_root,
        dev_root,
        'video4',
        index=0,
        product='Broken Camera',
        serial='CAM-BROKEN',
    )
    _link_by_id(by_path_root / 'usb-camera-video-index0', dev_root / 'video0')
    calls: List[List[str]] = []

    def probe(paths: Any) -> Dict[str, Dict[str, object]]:
        requested = list(paths)
        calls.append(requested)
        return {
            path: {
                'reachable': Path(path).resolve().name == 'video0',
                'status': 'ok' if Path(path).resolve().name == 'video0' else 'timeout',
            }
            for path in requested
        }

    discovery = HardwareDiscovery(
        sys_video_root=sys_video_root,
        dev_root=dev_root,
        by_id_root=by_id_root,
        interface_provider=lambda: [],
        socket_provider=lambda: '',
        elements_log_provider=lambda: [],
        camera_frame_probe=probe,
    )

    passive = discovery.discover()
    assert calls == []
    assert all(camera['capture_capable'] for camera in passive.cameras)
    assert passive.cameras[0]['stable_path'] == str(
        by_path_root / 'usb-camera-video-index0'
    )

    active = discovery.discover_with_camera_probe()
    assert len(calls) == 1
    by_node = {Path(camera['devnode']).name: camera for camera in active.cameras}
    assert by_node['video0']['frame_reachable'] is True
    assert by_node['video0']['capture_capable'] is True
    assert by_node['video4']['frame_reachable'] is False
    assert by_node['video4']['capture_capable'] is False
    assert by_node['video4']['probe_status'] == 'timeout'

    config = {
        'devices': [
            {'class': 'WebCameraOpenPi', 'device_class_id': 0, 'enabled': True},
            {'class': 'WebCameraOpenPi', 'device_class_id': 4, 'enabled': True},
        ]
    }
    recommended, _ = recommend_overrides(config, active)
    recommended_ids = {
        binding['capture_device'] for binding in recommended.values()
    }
    assert recommended_ids == {by_node['video0']['id']}


def _camera_snapshot(revision: str = 'camera-revision-1') -> HardwareSnapshot:
    return HardwareSnapshot(
        revision=revision,
        generated_at='2026-07-12T00:00:00+00:00',
        interfaces=[],
        cameras=[
            {
                'id': 'camera_opaque_0',
                'devnode': '/dev/video0',
                'realpath': '/dev/video0',
                'stable_path': '/dev/v4l/by-id/test-camera-video-index0',
                'name': 'Test Camera',
                'capture_capable': True,
                'source': 'sysfs_v4l2',
                'confidence': 'sysfs',
                'usb': {
                    'vendor_id': '1234',
                    'product_id': '5678',
                    'serial': 'CAMERA-0',
                    'manufacturer': 'Noematrix Test',
                    'product': 'Test Camera',
                },
            }
        ],
        robots=[],
        providers=[{'name': 'sysfs_v4l2', 'mode': 'read_only'}],
        warnings=[],
    )


def _multi_binding_snapshot() -> HardwareSnapshot:
    cameras = [
        {
            'id': f'camera_opaque_{index}',
            'devnode': f'/dev/video{index}',
            'realpath': f'/dev/video{index}',
            'stable_path': f'/dev/v4l/by-id/test-camera-{index}-video-index0',
            'name': f'Test Camera {index}',
            'capture_capable': True,
            'source': 'sysfs_v4l2',
            'confidence': 'sysfs',
            'usb': {'serial': f'CAMERA-{index}'},
        }
        for index in (0, 4, 8)
    ]
    return HardwareSnapshot(
        revision='multi-binding-revision-1',
        generated_at='2026-07-12T00:00:00+00:00',
        interfaces=[
            {
                'id': 'interface_opaque_robot',
                'name': 'enp6s0',
                'ipv4': '192.168.2.67',
                'prefixlen': 24,
                'network': '192.168.2.0/24',
                'robot_facing_candidate': True,
                'source': 'linux_interfaces',
            }
        ],
        cameras=cameras,
        robots=[
            {
                'id': 'robot_opaque_left',
                'model': 'Rizon4',
                'serial_number': 'Rizon4-062717',
                'rdk_sn': 'Rizon4-062717',
                'variant': 'standard',
                'ip': '192.168.2.100',
                'source': 'elements_cache',
                'confidence': 'cached',
            },
            {
                'id': 'robot_opaque_right',
                'model': 'Rizon4R',
                'serial_number': 'Rizon4R-062069',
                'rdk_sn': 'Rizon4R-062069',
                'variant': '4r',
                'ip': '192.168.2.177',
                'source': 'elements_cache',
                'confidence': 'cached',
            },
        ],
        providers=[],
        warnings=[],
    )


def _write_multi_binding_config(config_dir: Path, name: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f'{name}.yaml'
    path.write_text(
        """devices:
  - class: WebCameraOpenPi
    device_class_id: 0
    capture_id: 0
    data_shape: [480, 640, 3]
    data_dtype: uint8
  - class: WebCameraOpenPi
    device_class_id: 4
    capture_id: 4
    data_shape: [480, 640, 3]
    data_dtype: uint8
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-063087, Rizon4R-062071]
    network_interface_whitelist: [192.168.2.67]
    data_shape: [44]
    data_dtype: float64
""",
        encoding='utf-8',
    )
    return path


def _write_three_camera_config(config_dir: Path, name: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f'{name}.yaml'
    path.write_text(
        """devices:
  - class: WebCameraOpenPi
    device_class_id: 0
    capture_id: 0
    data_shape: [480, 640, 3]
    data_dtype: uint8
  - class: WebCameraOpenPi
    device_class_id: 4
    capture_id: 4
    data_shape: [480, 640, 3]
    data_dtype: uint8
  - class: WebCameraOpenPi
    device_class_id: 2
    capture_id: 2
    data_shape: [480, 640, 3]
    data_dtype: uint8
""",
        encoding='utf-8',
    )
    return path


def _write_camera_config(config_dir: Path, name: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f'{name}.yaml'
    path.write_text(
        """devices:
  - class: WebCameraOpenPi
    device_class_id: 4
    capture_id: 0
    data_shape: [480, 640, 3]
    data_dtype: uint8
    fps: 30
    api_token: TOP_SECRET_DISCOVERY_SENTINEL
""",
        encoding='utf-8',
    )
    return path


def _write_base_config(config_dir: Path, name: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f'{name}.yaml'
    path.write_text(
        """devices:
  - class: BaseDevice
    device_class_id: 7
    data_shape: [2]
    data_dtype: float32
""",
        encoding='utf-8',
    )
    return path


def _app_with_snapshot(
    config_dir: Path,
    snapshot: HardwareSnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    IdleLifecycle.instances = []
    monkeypatch.setattr(server, 'DeviceLifecycleManager', IdleLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=StaticDiscovery(snapshot),
    )
    app.testing = True
    return app


def test_passive_inventory_converts_elements_serials_and_deduplicates_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys_video_root = tmp_path / 'sys' / 'class' / 'video4linux'
    dev_root = tmp_path / 'dev'
    by_id_root = dev_root / 'v4l' / 'by-id'
    video0 = _write_video_endpoint(
        sys_video_root,
        dev_root,
        'video0',
        index=0,
        product='Left Camera',
        serial='CAM-A',
    )
    video1 = _write_video_endpoint(
        sys_video_root,
        dev_root,
        'video1',
        index=1,
        product='Left Camera Metadata',
        serial='CAM-A',
    )
    video4 = _write_video_endpoint(
        sys_video_root,
        dev_root,
        'video4',
        index=0,
        product='Right Camera',
        serial='CAM-B',
    )
    video5 = _write_video_endpoint(
        sys_video_root,
        dev_root,
        'video5',
        index=1,
        product='Right Camera Metadata',
        serial='CAM-B',
    )
    _link_by_id(by_id_root / 'camera-a-alias', video0)
    _link_by_id(by_id_root / 'camera-a-video-index0', video0)
    _link_by_id(by_id_root / 'camera-a-video-index1', video1)
    _link_by_id(by_id_root / 'camera-b-video-index0', video4)
    _link_by_id(by_id_root / 'camera-b-video-index1', video5)

    elements_log = tmp_path / 'UIApp_test.log'
    elements_log.write_text(
        """[2026-07-11 15:45:24.931139][debug] will connect to ip = 192.168.2.100, sn.toStdString() = A02L-09-M4-062717
[2026-07-11 15:45:25.931139][debug] will connect to ip = 192.168.2.177, sn.toStdString() = A02LMR-06-M1-062069
""",
        encoding='utf-8',
    )
    provider_calls = {'interfaces': 0, 'sockets': 0, 'logs': 0}

    def interfaces() -> List[Dict[str, Any]]:
        provider_calls['interfaces'] += 1
        return [
            {
                'name': 'enp6s0',
                'ipv4': '192.168.2.67',
                'prefixlen': 24,
                'operstate': 'up',
            }
        ]

    def sockets() -> str:
        provider_calls['sockets'] += 1
        return 'ESTAB 0 0 192.168.2.67:40123 192.168.2.177:5000'

    def logs() -> List[Path]:
        provider_calls['logs'] += 1
        return [elements_log]

    discovery = HardwareDiscovery(
        sys_video_root=sys_video_root,
        dev_root=dev_root,
        by_id_root=by_id_root,
        interface_provider=interfaces,
        socket_provider=sockets,
        elements_log_provider=logs,
    )
    forbidden_before = {
        name: sys.modules.get(name) for name in ('flexivrdk', 'cv2')
    }
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name.split('.', 1)[0] in {'flexivrdk', 'cv2'}:
            pytest.fail(f'active hardware library imported during discovery: {name}')
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, '__import__', guarded_import)
        snapshot = discovery.discover()

    assert provider_calls == {'interfaces': 1, 'sockets': 1, 'logs': 1}
    assert {
        name: sys.modules.get(name) for name in ('flexivrdk', 'cv2')
    } == forbidden_before
    assert [robot['serial_number'] for robot in snapshot.robots] == [
        'Rizon4-062717',
        'Rizon4R-062069',
    ]
    assert [robot['model'] for robot in snapshot.robots] == [
        'Rizon4',
        'Rizon4R',
    ]
    assert [robot['rdk_sn'] for robot in snapshot.robots] == [
        'Rizon4-062717',
        'Rizon4R-062069',
    ]
    assert [robot['variant'] for robot in snapshot.robots] == [
        'standard',
        '4r',
    ]
    assert all(robot['source'] == 'elements_cache' for robot in snapshot.robots)
    assert all(robot['confidence'] == 'cached' for robot in snapshot.robots)
    assert snapshot.robots[0]['peer_active'] is False
    assert snapshot.robots[1]['peer_active'] is True
    assert snapshot.interfaces[0]['name'] == 'enp6s0'
    assert snapshot.interfaces[0]['ipv4'] == '192.168.2.67'
    assert snapshot.interfaces[0]['robot_facing_candidate'] is True
    assert [Path(camera['devnode']).name for camera in snapshot.cameras] == [
        'video0',
        'video4',
    ]
    assert len({camera['id'] for camera in snapshot.cameras}) == 2
    assert snapshot.cameras[0]['stable_path'].endswith(
        'camera-a-video-index0'
    )
    assert snapshot.cameras[1]['stable_path'].endswith(
        'camera-b-video-index0'
    )
    assert {provider['name'] for provider in snapshot.providers} == {
        'linux_interfaces',
        'sysfs_v4l2',
        'elements_cache',
    }

    robot_config = {
        'devices': [
            {
                'class': 'RizonRobot',
                'device_class_id': 2,
                'robot_sn': ['Rizon4-063087', 'Rizon4R-062071'],
                'network_interface_whitelist': ['192.168.2.10'],
            }
        ]
    }
    recommended, _ = recommend_overrides(robot_config, snapshot)
    assert recommended['RizonRobot:2'] == {
        'network_interface': snapshot.interfaces[0]['id'],
    }

    matching_config = {
        'devices': [
            {
                'class': 'RizonRobot',
                'device_class_id': 2,
                'robot_sn': ['Rizon4-062717', 'Rizon4R-062069'],
                'network_interface_whitelist': ['192.168.2.10'],
            }
        ]
    }
    matching_recommended, _ = recommend_overrides(matching_config, snapshot)
    left = next(robot for robot in snapshot.robots if robot['model'] == 'Rizon4')
    right = next(robot for robot in snapshot.robots if robot['model'] == 'Rizon4R')
    assert matching_recommended['RizonRobot:2'] == {
        'robot_left': left['id'],
        'robot_right': right['id'],
        'network_interface': snapshot.interfaces[0]['id'],
    }

    camera_config = {
        'devices': [
            {'class': 'WebCameraOpenPi', 'device_class_id': 10, 'capture_id': 0},
            {'class': 'WebCameraOpenPi', 'device_class_id': 11, 'capture_id': 99},
        ]
    }
    camera_recommended, _ = recommend_overrides(camera_config, snapshot)
    assert camera_recommended['WebCameraOpenPi:10']['capture_device'] != (
        camera_recommended['WebCameraOpenPi:11']['capture_device']
    )


def test_robot_recommendation_matches_configured_serial_not_inventory_order() -> None:
    snapshot = HardwareSnapshot(
        revision='robot-order-revision',
        generated_at='2026-08-06T00:00:00+00:00',
        interfaces=[],
        cameras=[],
        robots=[
            {
                'id': 'stale-left',
                'model': 'Rizon4',
                'variant': 'standard',
                'rdk_sn': 'Rizon4-063079',
                'serial_number': 'Rizon4-063079',
                'peer_active': False,
            },
            {
                'id': 'configured-left',
                'model': 'Rizon4',
                'variant': 'standard',
                'rdk_sn': 'Rizon4-063114',
                'serial_number': 'Rizon4-063114',
                'peer_active': True,
            },
            {
                'id': 'configured-right',
                'model': 'Rizon4R',
                'variant': '4r',
                'rdk_sn': 'Rizon4R-062067',
                'serial_number': 'Rizon4R-062067',
                'peer_active': True,
            },
        ],
        providers=[],
        warnings=[],
    )
    config = {
        'devices': [
            {
                'class': 'RizonRobot',
                'device_class_id': 2,
                'robot_sn': ['Rizon4-063114', 'Rizon4R-062067'],
            }
        ]
    }

    recommended, configured = recommend_overrides(config, snapshot)

    assert recommended['RizonRobot:2'] == {
        'robot_left': 'configured-left',
        'robot_right': 'configured-right',
    }
    assert configured['RizonRobot:2']['detected'] == [True, True]


def test_null_robot_side_overrides_preserve_config_and_unknown_ids_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    (config_dir / 'robot.yaml').write_text(
        """devices:
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [Rizon4-063087, Rizon4R-062071]
    network_interface_whitelist: [192.168.2.67]
    data_shape: [44]
    data_dtype: float64
""",
        encoding='utf-8',
    )
    snapshot = HardwareSnapshot(
        revision='robot-revision-1',
        generated_at='2026-07-12T00:00:00+00:00',
        interfaces=[
            {
                'id': 'interface_opaque_robot',
                'name': 'enp6s0',
                'ipv4': '192.168.2.67',
                'prefixlen': 24,
                'network': '192.168.2.0/24',
                'robot_facing_candidate': True,
                'source': 'linux_interfaces',
            }
        ],
        cameras=[],
        robots=[
            {
                'id': 'robot_opaque_left',
                'model': 'Rizon4',
                'serial_number': 'Rizon4-062717',
                'rdk_sn': 'Rizon4-062717',
                'variant': 'standard',
                'ip': '192.168.2.100',
                'source': 'elements_cache',
                'confidence': 'cached',
            },
            {
                'id': 'robot_opaque_right',
                'model': 'Rizon4R',
                'serial_number': 'Rizon4R-062069',
                'rdk_sn': 'Rizon4R-062069',
                'variant': '4r',
                'ip': '192.168.2.177',
                'source': 'elements_cache',
                'confidence': 'cached',
            },
        ],
        providers=[],
        warnings=[],
    )
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    preserved = client.post(
        '/api/config/select',
        json={
            'name': 'robot',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'RizonRobot:2': {'robot_left': None, 'robot_right': None}
            },
        },
    )
    assert preserved.status_code == 200
    assert app.config['DEVICE_CONTROLLER'].starter.config['devices'][0][
        'robot_sn'
    ] == ['Rizon4-063087', 'Rizon4R-062071']
    assert preserved.get_json()['effective_bindings'] == {}

    rejected = client.post(
        '/api/bindings',
        json={
            'name': 'robot',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'RizonRobot:2': {'robot_right': 'not_a_discovery_id'}
            },
        },
    )
    assert rejected.status_code == 422
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_v4l2_override_rejects_camera_classes_without_capture_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    (config_dir / 'opencv.yaml').write_text(
        """devices:
  - class: OpenCVCameraDevice
    device_class_id: 4
    data_shape: [480, 640, 3]
    data_dtype: uint8
""",
        encoding='utf-8',
    )
    snapshot = _camera_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    response = app.test_client().post(
        '/api/config/select',
        json={
            'name': 'opencv',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'OpenCVCameraDevice:4': {
                    'capture_device': 'camera_opaque_0'
                }
            },
        },
    )
    assert response.status_code == 422
    assert 'does not support runtime bindings' in response.get_json()['error']
    app.config['DEVICE_CONTROLLER'].shutdown()


@pytest.mark.parametrize(
    'arbitrary_value',
    ['/dev/video0', '192.168.2.177', 'Rizon4-062717'],
)
def test_select_rejects_non_opaque_runtime_binding_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arbitrary_value: str,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_camera_config(config_dir, 'camera')
    app = _app_with_snapshot(config_dir, _camera_snapshot(), monkeypatch)
    response = app.test_client().post(
        '/api/config/select',
        json={
            'name': 'camera',
            'discovery_revision': 'camera-revision-1',
            'overrides': {
                'WebCameraOpenPi:4': {'capture_device': arbitrary_value}
            },
        },
    )

    assert response.status_code == 422
    assert response.get_json()['ok'] is False
    assert arbitrary_value not in response.get_json()['error']
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_select_rejects_stale_discovery_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_camera_config(config_dir, 'camera')
    app = _app_with_snapshot(config_dir, _camera_snapshot(), monkeypatch)
    response = app.test_client().post(
        '/api/config/select',
        json={
            'name': 'camera',
            'discovery_revision': 'stale-revision',
            'overrides': {
                'WebCameraOpenPi:4': {'capture_device': 'camera_opaque_0'}
            },
        },
    )

    assert response.status_code == 409
    assert response.get_json()['ok'] is False
    assert app.config['DEVICE_CONTROLLER'].selected_config_name() is None
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_camera_dropdown_discovery_binding_and_start_use_one_reachable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirror pointerdown discovery, selection, binding, then start."""

    config_dir = tmp_path / 'configs'
    _write_three_camera_config(config_dir, 'camera')
    passive = _multi_binding_snapshot()
    dropdown_snapshot = replace(
        passive,
        revision='dropdown-reachable-revision',
        cameras=[
            {
                **camera,
                'capture_capable': camera['id'] != 'camera_opaque_0',
                'frame_reachable': camera['id'] != 'camera_opaque_0',
                'probe_status': (
                    'busy' if camera['id'] == 'camera_opaque_0' else 'ok'
                ),
            }
            for camera in passive.cameras
        ],
    )
    later_snapshot = replace(passive, revision='probe-raced-revision')
    discovery = ChangingCameraDiscovery(
        passive, [dropdown_snapshot, later_snapshot]
    )
    IdleLifecycle.instances = []
    monkeypatch.setattr(server, 'DeviceLifecycleManager', IdleLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=discovery,
    )
    app.testing = True
    client = app.test_client()

    selected = client.post('/api/config/select', json={'name': 'camera'})
    assert selected.status_code == 200

    # A camera already owned by the panel cannot be opened by the isolated
    # discovery probe, but that unrelated running slot must remain valid.
    first_started = client.post('/api/devices/WebCameraOpenPi/0/start')
    assert first_started.status_code == 202
    deadline = time.monotonic() + 2.0
    while (
        'WebCameraOpenPi:0' not in IdleLifecycle.instances[-1].active_keys
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert 'WebCameraOpenPi:0' in IdleLifecycle.instances[-1].active_keys

    dropdown = client.get(
        '/api/hardware/discovery?config_name=camera&probe_cameras=1'
    )
    assert dropdown.status_code == 200
    dropdown_payload = dropdown.get_json()
    assert dropdown_payload['revision'] == dropdown_snapshot.revision
    reachable = next(
        camera
        for camera in dropdown_payload['cameras']
        if camera['id'] == 'camera_opaque_8'
    )
    assert reachable['capture_capable'] is True

    rebound = client.post(
        '/api/bindings',
        json={
            'name': 'camera',
            'discovery_revision': dropdown_payload['revision'],
            'overrides': {
                'WebCameraOpenPi:4': {'capture_device': reachable['id']}
            },
        },
    )
    assert rebound.status_code == 200
    assert discovery.active_calls == 1
    assert IdleLifecycle.instances[-1].config['devices'][0][
        'capture_id'
    ].endswith('test-camera-0-video-index0')
    assert IdleLifecycle.instances[-1].config['devices'][1][
        'capture_id'
    ].endswith('test-camera-8-video-index0')

    started = client.post('/api/devices/WebCameraOpenPi/4/start')
    assert started.status_code == 202
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = client.get('/api/state').get_json()
        camera = next(
            device
            for device in state['devices']
            if device['key'] == 'WebCameraOpenPi:4'
        )
        if camera['status'] == 'running':
            break
        time.sleep(0.01)
    assert camera['status'] == 'running'
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_three_camera_config_parks_missing_slot_without_duplicate_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    yaml_path = _write_three_camera_config(config_dir, 'three-camera')
    yaml_before = yaml_path.read_bytes()
    snapshot = _camera_snapshot()
    snapshot.cameras.append(
        {
            'id': 'camera_opaque_2',
            'devnode': '/dev/video2',
            'realpath': '/dev/video2',
            'stable_path': '/dev/v4l/by-id/test-camera-2-video-index0',
            'name': 'Test Camera 2',
            'capture_capable': True,
            'source': 'sysfs_v4l2',
            'confidence': 'sysfs',
            'usb': {'serial': 'CAMERA-2'},
        }
    )
    config = compose_runtime_config('three-camera', config_dir)

    with pytest.raises(
        ValueError,
        match='camera binding is duplicated by WebCameraOpenPi:4 and WebCameraOpenPi:2',
    ):
        apply_discovery_overrides(
            config,
            {
                'WebCameraOpenPi:0': {'capture_device': 'camera_opaque_0'},
                'WebCameraOpenPi:4': {'capture_device': 'camera_opaque_2'},
            },
            snapshot,
        )

    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    response = app.test_client().post(
        '/api/config/select',
        json={
            'name': 'three-camera',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:0': {'capture_device': 'camera_opaque_0'},
                'WebCameraOpenPi:4': {'capture_device': 'camera_opaque_2'},
            },
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['selected_config'] == 'three-camera'
    assert payload['effective_bindings']['WebCameraOpenPi:2']['capture_device'] == {
        'id': None,
        'status': 'missing',
        'detected': False,
    }
    devices = {
        f"{device['class']}:{device['device_class_id']}": device
        for device in app.config['DEVICE_CONTROLLER'].starter.config['devices']
    }
    assert devices['WebCameraOpenPi:4']['capture_id'].endswith(
        'test-camera-2-video-index0'
    )
    assert devices['WebCameraOpenPi:2'][CAPTURE_BINDING_STATE_KEY] == 'missing'
    assert yaml_path.read_bytes() == yaml_before

    rebound = app.test_client().post(
        '/api/bindings',
        json={
            'name': 'three-camera',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:0': {'capture_device': 'camera_opaque_0'},
            },
        },
    )
    assert rebound.status_code == 200
    assert rebound.get_json()['effective_bindings']['WebCameraOpenPi:2'][
        'capture_device'
    ]['status'] == 'missing'
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_all_missing_camera_slots_allow_config_selection_without_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_three_camera_config(config_dir, 'offline-cameras')

    class UnavailableDiscovery:
        def discover(self) -> HardwareSnapshot:
            raise OSError('sysfs unavailable')

    IdleLifecycle.instances = []
    monkeypatch.setattr(server, 'DeviceLifecycleManager', IdleLifecycle)
    app = server.create_app(
        config_dir=config_dir,
        hardware_discovery=UnavailableDiscovery(),
    )
    app.testing = True
    response = app.test_client().post(
        '/api/config/select',
        json={
            'name': 'offline-cameras',
            'discovery_revision': 'stale-client-revision',
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['selected_config'] == 'offline-cameras'
    assert payload['discovery_revision'] is None
    assert all(
        binding['capture_device']['status'] == 'missing'
        for binding in payload['effective_bindings'].values()
    )
    app.config['DEVICE_CONTROLLER'].shutdown()


def test_runtime_camera_override_is_memory_only_and_public_state_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    yaml_path = _write_camera_config(config_dir, 'camera')
    _write_base_config(config_dir, 'other')
    yaml_before = yaml_path.read_bytes()
    snapshot = _camera_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'camera'}).status_code == 200
    bound = client.post(
        '/api/bindings',
        json={
            'name': 'camera',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:4': {
                    'capture_device': 'camera_opaque_0'
                }
            },
        },
    )

    assert bound.status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    effective_device = controller.starter.config['devices'][0]
    assert effective_device['device_class_id'] == 4
    assert effective_device['capture_id'] == (
        '/dev/v4l/by-id/test-camera-video-index0'
    )
    assert yaml_path.read_bytes() == yaml_before

    state = client.get('/api/state').get_json()
    binding = state['effective_bindings']['WebCameraOpenPi:4']['capture_device']
    assert binding == {
        'id': 'camera_opaque_0',
        'devnode': '/dev/video0',
        'stable_path': '/dev/v4l/by-id/test-camera-video-index0',
        'name': 'Test Camera',
        'source': 'sysfs_v4l2',
    }
    assert state['discovery_revision'] == snapshot.revision
    assert state['base_config']['devices'][0]['device_class_id'] == 4
    assert state['effective_config']['devices'][0]['device_class_id'] == 4
    assert 'api_token' not in state['base_config']['devices'][0]
    assert 'capture_id' not in state['base_config']['devices'][0]
    assert 'api_token' not in state['effective_config']['devices'][0]
    assert 'capture_id' not in state['effective_config']['devices'][0]
    assert 'TOP_SECRET_DISCOVERY_SENTINEL' not in str(state)

    inspected = client.get('/api/configs/camera').get_json()
    assert inspected['base_config']['name'] == 'camera'
    assert inspected['base_config']['devices'][0]['device_class_id'] == 4
    assert inspected['effective_bindings'] == state['effective_bindings']
    assert inspected['effective_config']['devices'][0]['device_class_id'] == 4
    assert 'TOP_SECRET_DISCOVERY_SENTINEL' not in str(inspected)

    # Returning without client overrides rebuilds a complete assignment from
    # the current discovery snapshot rather than resurrecting stale session state.
    assert client.post('/api/config/select', json={'name': 'other'}).status_code == 200
    unbound = client.post('/api/config/select', json={'name': 'camera'})
    assert unbound.status_code == 200
    unbound_state = unbound.get_json()
    assert unbound_state['effective_bindings']['WebCameraOpenPi:4'][
        'capture_device'
    ]['id'] == 'camera_opaque_0'
    assert unbound_state['discovery_revision'] == snapshot.revision
    assert controller.starter.config['devices'][0]['device_class_id'] == 4
    assert controller.starter.config['devices'][0]['capture_id'].endswith(
        'test-camera-video-index0'
    )
    assert yaml_path.read_bytes() == yaml_before
    unbound_inspect = client.get('/api/configs/camera').get_json()
    assert unbound_inspect['effective_bindings'] == unbound_state[
        'effective_bindings'
    ]
    assert unbound_inspect['discovery_revision'] == snapshot.revision
    assert 'TOP_SECRET_DISCOVERY_SENTINEL' not in str(unbound_inspect)
    controller.shutdown()


def test_runtime_camera_override_uses_discovered_devnode_without_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / 'configs'
    _write_camera_config(config_dir, 'camera')
    snapshot = _camera_snapshot()
    snapshot.cameras[0]['stable_path'] = None
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'camera'}).status_code == 200
    bound = client.post(
        '/api/bindings',
        json={
            'name': 'camera',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:4': {
                    'capture_device': 'camera_opaque_0'
                }
            },
        },
    )

    assert bound.status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    assert controller.starter.config['devices'][0]['capture_id'] == '/dev/video0'
    public = bound.get_json()['effective_bindings'][
        'WebCameraOpenPi:4'
    ]['capture_device']
    assert public['devnode'] == '/dev/video0'
    assert public['stable_path'] is None
    controller.shutdown()


def test_sparse_binding_update_preserves_unrelated_active_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    yaml_path = _write_multi_binding_config(config_dir, 'multi')
    yaml_before = yaml_path.read_bytes()
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    starter = controller.starter
    assert isinstance(starter, IdleLifecycle)
    starter.active_keys.add('WebCameraOpenPi:0')
    monitor_token = starter.monitor_token
    active_before = next(
        device
        for device in starter.get_status()['devices']
        if device['key'] == 'WebCameraOpenPi:0'
    )

    camera_update = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:4': {
                    'capture_device': 'camera_opaque_8'
                }
            },
        },
    )
    assert camera_update.status_code == 200

    robot_update = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'RizonRobot:2': {
                    'robot_left': 'robot_opaque_left',
                    'robot_right': 'robot_opaque_right',
                    'network_interface': 'interface_opaque_robot',
                }
            },
        },
    )
    assert robot_update.status_code == 200

    assert controller.starter is starter
    assert starter.monitor_token is monitor_token
    active_after = next(
        device
        for device in starter.get_status()['devices']
        if device['key'] == 'WebCameraOpenPi:0'
    )
    assert active_after == active_before
    devices = {
        f"{device['class']}:{device['device_class_id']}": device
        for device in starter.config['devices']
    }
    assert devices['WebCameraOpenPi:4']['capture_id'] == (
        '/dev/v4l/by-id/test-camera-8-video-index0'
    )
    assert devices['RizonRobot:2']['robot_sn'] == [
        'Rizon4-062717',
        'Rizon4R-062069',
    ]
    assert yaml_path.read_bytes() == yaml_before

    active_rejected = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:0': {
                    'capture_device': 'camera_opaque_4'
                }
            },
        },
    )
    assert active_rejected.status_code == 409
    assert 'WebCameraOpenPi:0' in active_rejected.get_json()['error']
    assert controller.starter is starter
    assert next(
        device
        for device in starter.get_status()['devices']
        if device['key'] == 'WebCameraOpenPi:0'
    ) == active_before
    controller.shutdown()


def test_camera_binding_update_is_not_blocked_by_auxiliary_model_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_binding_config(config_dir, 'multi')
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    assert controller.workflow is not None
    workflow = controller.workflow
    monkeypatch.setattr(workflow, 'has_active_processes', lambda: True)
    monkeypatch.setattr(
        workflow,
        'has_active_runtime_processes',
        lambda: False,
    )

    updated = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:4': {
                    'capture_device': 'camera_opaque_8'
                }
            },
        },
    )

    assert updated.status_code == 200
    assert updated.get_json()['effective_bindings']['WebCameraOpenPi:4'][
        'capture_device'
    ]['id'] == 'camera_opaque_8'
    assert controller.workflow is workflow
    assert controller.workflow.has_active_processes() is True

    controller.control_device('start', 'WebCameraOpenPi', 4)
    assert 'WebCameraOpenPi:4' in controller.starter.active_keys
    controller.shutdown()


def test_full_camera_draft_ignores_unchanged_running_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_binding_config(config_dir, 'multi')
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    starter = controller.starter
    assert isinstance(starter, IdleLifecycle)
    running_key = 'WebCameraOpenPi:0'
    changed_key = 'WebCameraOpenPi:4'
    starter.active_keys.add(running_key)

    calls: list[tuple[str, str]] = []
    controller.grippers.stop = lambda reason: calls.append(('stop', reason))
    controller.grippers.configure = lambda name, config: calls.append(
        ('configure', name)
    )
    updated = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            # Reproduce an older tab sending every camera slot even though the
            # running slot is unchanged.
            'overrides': {
                changed_key: {'capture_device': 'camera_opaque_8'},
                running_key: {'capture_device': 'camera_opaque_0'},
            },
        },
    )

    assert updated.status_code == 200
    devices = {
        f"{device['class']}:{device['device_class_id']}": device
        for device in starter.config['devices']
    }
    assert devices[running_key]['capture_id'].endswith(
        'test-camera-0-video-index0'
    )
    assert devices[changed_key]['capture_id'].endswith(
        'test-camera-8-video-index0'
    )
    assert running_key in starter.active_keys
    assert calls == []
    controller.shutdown()


def test_camera_binding_is_independent_of_other_camera_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_binding_config(config_dir, 'multi')
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    left_key = 'WebCameraOpenPi:0'
    right_key = 'WebCameraOpenPi:4'
    controller._device_operations[left_key] = 'start'

    right_update = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                right_key: {'capture_device': 'camera_opaque_8'},
            },
        },
    )

    assert right_update.status_code == 200
    assert controller._device_operations == {left_key: 'start'}

    left_update = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                left_key: {'capture_device': 'camera_opaque_4'},
            },
        },
    )

    assert left_update.status_code == 409
    assert left_key in left_update.get_json()['error']
    assert controller._device_operations == {left_key: 'start'}
    controller._device_operations.clear()
    controller.shutdown()


def test_two_camera_bindings_can_be_swapped_in_one_atomic_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_binding_config(config_dir, 'multi')
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    swapped = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:0': {
                    'capture_device': 'camera_opaque_4'
                },
                'WebCameraOpenPi:4': {
                    'capture_device': 'camera_opaque_0'
                },
            },
        },
    )

    assert swapped.status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    devices = {
        f"{device['class']}:{device['device_class_id']}": device
        for device in controller.starter.config['devices']
    }
    assert devices['WebCameraOpenPi:0']['capture_id'].endswith(
        'test-camera-4-video-index0'
    )
    assert devices['WebCameraOpenPi:4']['capture_id'].endswith(
        'test-camera-0-video-index0'
    )
    assert set(swapped.get_json()['effective_bindings']) == {
        'WebCameraOpenPi:0', 'WebCameraOpenPi:4'
    }
    controller.shutdown()


def test_sparse_camera_binding_automatically_swaps_current_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_binding_config(config_dir, 'multi')
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    left_key = 'WebCameraOpenPi:0'
    right_key = 'WebCameraOpenPi:4'
    before = {
        left_key: 'camera_opaque_0',
        right_key: 'camera_opaque_4',
    }

    swapped = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                right_key: {'capture_device': before[left_key]},
            },
        },
    )

    assert swapped.status_code == 200
    bindings = swapped.get_json()['effective_bindings']
    assert bindings[right_key]['capture_device']['id'] == before[left_key]
    assert bindings[left_key]['capture_device']['id'] == before[right_key]
    controller.shutdown()


def test_duplicate_full_camera_draft_preserves_primary_swap_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_binding_config(config_dir, 'multi')
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    left_key = 'WebCameraOpenPi:0'
    right_key = 'WebCameraOpenPi:4'
    before = {
        left_key: 'camera_opaque_0',
        right_key: 'camera_opaque_4',
    }

    swapped = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            # The clicked slot is inserted first. Reproduce a stale UI draft
            # that incorrectly repeats its selected camera for the old owner.
            'overrides': {
                right_key: {'capture_device': before[left_key]},
                left_key: {'capture_device': before[left_key]},
            },
        },
    )

    assert swapped.status_code == 200
    bindings = swapped.get_json()['effective_bindings']
    assert bindings[right_key]['capture_device']['id'] == before[left_key]
    assert bindings[left_key]['capture_device']['id'] == before[right_key]
    controller.shutdown()


def test_external_camera_bindings_are_staged_for_the_next_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / 'configs'
    _write_multi_binding_config(config_dir, 'multi')
    snapshot = _multi_binding_snapshot()
    app = _app_with_snapshot(config_dir, snapshot, monkeypatch)
    client = app.test_client()

    assert client.post('/api/config/select', json={'name': 'multi'}).status_code == 200
    controller = app.config['DEVICE_CONTROLLER']
    starter = controller.starter
    assert isinstance(starter, IdleLifecycle)
    starter.external_keys.update(
        {'WebCameraOpenPi:0', 'WebCameraOpenPi:4'}
    )
    monitor_token = starter.monitor_token

    swapped = client.post(
        '/api/bindings',
        json={
            'name': 'multi',
            'discovery_revision': snapshot.revision,
            'overrides': {
                'WebCameraOpenPi:0': {
                    'capture_device': 'camera_opaque_4'
                },
                'WebCameraOpenPi:4': {
                    'capture_device': 'camera_opaque_0'
                },
            },
        },
    )

    assert swapped.status_code == 200
    assert controller.starter is starter
    assert starter.monitor_token is monitor_token
    devices = {
        f"{device['class']}:{device['device_class_id']}": device
        for device in starter.config['devices']
    }
    assert devices['WebCameraOpenPi:0']['capture_id'].endswith(
        'test-camera-4-video-index0'
    )
    assert devices['WebCameraOpenPi:4']['capture_id'].endswith(
        'test-camera-0-video-index0'
    )
    external = {
        device['key']: device
        for device in swapped.get_json()['devices']
        if device['key'] in starter.external_keys
    }
    assert set(external) == starter.external_keys
    assert all(device['status'] == 'external' for device in external.values())
    assert all(device['running'] is False for device in external.values())
    assert all(device['desired_state'] == 'stopped' for device in external.values())
    assert all(device['shm']['connected'] is True for device in external.values())

    starter.external_keys.clear()
    controller.shutdown()
