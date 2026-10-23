"""Hardware-free REST tests for camera preview and server cleanup."""

from __future__ import annotations

import signal
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pytest

# The REST layer is intentionally hardware-free. DeviceStarter imports the
# production registries eagerly, so replace only those registries before import
# instead of requiring RealSense/robot vendor SDKs in this unit test process.
sys.modules.setdefault('devices', types.SimpleNamespace(DEVICE_CLASSES={}))
import manager  # noqa: E402, F401
sys.modules.setdefault(
    'visualizers', types.SimpleNamespace(VISUALIZER_CLASSES={})
)

from incubator.control_panel import server as device_control_server  # noqa: E402
from incubator.control_panel.lifecycle import DeviceLifecycleManager  # noqa: E402
from incubator.control_panel.camera_preview import CameraPreviewReader  # noqa: E402

DeviceStarter = DeviceLifecycleManager
from incubator.control_panel.shm_protocol import (  # noqa: E402
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    calculate_device_shm_size,
    pack_buffer_header,
    pack_device_header,
    pack_frame_header,
)


NOW_NS = 1_000_000_000_000


class FakeChannel:
    def __init__(self, data: bytearray) -> None:
        self.buf = data
        self.is_read_only = True
        self.closed = False
        self.unlink_calls = 0

    def close(self) -> None:
        self.closed = True

    def unlink(self) -> None:
        self.unlink_calls += 1
        raise AssertionError("The preview API must never unlink producer SHM")


class FakeConnector:
    def __init__(self, items: Iterable[Any]) -> None:
        self.items = list(items)
        self.calls = []

    def __call__(
        self, device_class: str, device_class_id: int, *, read_only: bool
    ) -> FakeChannel:
        self.calls.append((device_class, device_class_id, read_only))
        item = self.items[min(len(self.calls) - 1, len(self.items) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


def _camera_segment() -> bytearray:
    device_class = 'WebCameraOpenPi'
    device_class_id = 2
    shape = (8, 10, 3)
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :, 1] = 220
    frame_data_size = frame.nbytes
    frame_size = FRAME_HEADER_SIZE + frame_data_size
    data = bytearray(calculate_device_shm_size(1, frame_data_size))
    data[:DEVICE_HEADER_SIZE] = pack_device_header(
        device_class,
        device_class_id,
        30.0,
        'uint8',
        shape,
        frame_size,
        4.5,
    )
    header_end = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
    data[DEVICE_HEADER_SIZE:header_end] = pack_buffer_header(1, 1, 0)
    data[header_end:header_end + FRAME_HEADER_SIZE] = pack_frame_header(
        NOW_NS - 10_000_000
    )
    data[header_end + FRAME_HEADER_SIZE:] = frame.tobytes()
    return data


def _missing_shm(device_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'name': f"{device_info['device_class']}_{device_info['device_class_id']}_data",
        'connected': False,
        'last_timestamp_ns': None,
        'age_ms': None,
        'fps': device_info['config'].get('fps'),
        'shape': device_info['config'].get('data_shape'),
        'dtype': device_info['config'].get('data_dtype'),
        'fresh': False,
        'frames': 0,
        'error': None,
    }


def _write_configs(config_dir: Path) -> None:
    config_dir.mkdir(parents=True)
    body = """devices:
  - class: WebCameraOpenPi
    device_class_id: 2
    data_shape: [8, 10, 3]
    data_dtype: uint8
    fps: 30.0
  - class: BaseDevice
    device_class_id: 7
    data_shape: [2]
    data_dtype: float32
    fps: 10.0
global_settings:
  restart_on_failure: true
  max_restart_attempts: 2
  restart_delay: 0.0
runtime:
  signal_files: {}
"""
    (config_dir / 'demo.yaml').write_text(body, encoding='utf-8')
    (config_dir / 'other.yaml').write_text(body, encoding='utf-8')


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / 'configs'
    _write_configs(directory)
    monkeypatch.setattr(
        DeviceStarter,
        '_probe_shared_memory',
        lambda self, device_info: _missing_shm(device_info),
    )
    return directory


def _app_with_connector(
    config_dir: Path, connector: FakeConnector
) -> tuple:
    factory_calls = []

    def factory(device_class: str, device_class_id: int) -> CameraPreviewReader:
        factory_calls.append((device_class, device_class_id))
        return CameraPreviewReader(
            device_class,
            device_class_id,
            connector=connector,
            clock_ns=lambda: NOW_NS,
        )

    app = device_control_server.create_app(
        config_dir=config_dir,
        preview_reader_factory=factory,
    )
    app.testing = True
    return app, app.test_client(), factory_calls


def test_preview_authorizes_selected_config_device_and_capability(
    config_dir: Path,
) -> None:
    connector = FakeConnector([FileNotFoundError('missing')])
    app, client, factory_calls = _app_with_connector(config_dir, connector)

    assert client.get(
        '/api/devices/WebCameraOpenPi/2/preview/status'
    ).status_code == 409
    assert client.post('/api/config/select', json={'name': 'demo'}).status_code == 200
    assert client.get(
        '/api/devices/WebCameraOpenPi/999/preview/status'
    ).status_code == 404
    noncamera = client.get('/api/devices/BaseDevice/7/preview/status')
    assert noncamera.status_code == 400
    assert 'does not support camera preview' in noncamera.get_json()['error']
    assert factory_calls == []
    assert connector.calls == []

    app.config['DEVICE_CONTROLLER'].shutdown()


def test_not_found_status_is_pollable_and_images_use_404(config_dir: Path) -> None:
    connector = FakeConnector([FileNotFoundError('missing')])
    app, client, factory_calls = _app_with_connector(config_dir, connector)
    client.post('/api/config/select', json={'name': 'demo'})

    response = client.get('/api/devices/WebCameraOpenPi/2/preview/status')
    payload = response.get_json()
    assert response.status_code == 200
    assert payload['status'] == 'not_found'
    assert payload['timestamp'] is None
    assert payload['age_ms'] is None
    assert payload['fps'] == 30.0
    assert payload['shape'] == [8, 10, 3]
    assert payload['dtype'] == 'uint8'
    assert payload['available'] is False

    jpeg = client.get('/api/devices/WebCameraOpenPi/2/preview.jpg')
    assert jpeg.status_code == 404
    assert jpeg.get_json()['status'] == 'not_found'
    mjpeg = client.get('/api/devices/WebCameraOpenPi/2/preview.mjpg')
    assert mjpeg.status_code == 404
    assert mjpeg.get_json()['status'] == 'not_found'
    assert factory_calls == [('WebCameraOpenPi', 2)]

    app.config['DEVICE_CONTROLLER'].shutdown()


def test_status_jpeg_and_mjpeg_reuse_reader_then_config_switch_closes_it(
    config_dir: Path,
) -> None:
    channel = FakeChannel(_camera_segment())
    connector = FakeConnector([channel])
    app, client, factory_calls = _app_with_connector(config_dir, connector)
    client.post('/api/config/select', json={'name': 'demo'})

    status = client.get('/api/devices/WebCameraOpenPi/2/preview/status')
    payload = status.get_json()
    assert payload['status'] == 'ok'
    assert payload['timestamp'] == NOW_NS - 10_000_000
    assert payload['timestamp_ns'] == NOW_NS - 10_000_000
    assert payload['age_ms'] == 10.0
    assert payload['fps'] == 30.0
    assert payload['shape'] == [8, 10, 3]
    assert payload['dtype'] == 'uint8'

    jpeg = client.get('/api/devices/WebCameraOpenPi/2/preview.jpg')
    assert jpeg.status_code == 200
    assert jpeg.content_type == 'image/jpeg'
    assert jpeg.data.startswith(b'\xff\xd8')
    assert jpeg.headers['Cache-Control'].startswith('no-store')

    stream = client.get(
        '/api/devices/WebCameraOpenPi/2/preview.mjpg', buffered=False
    )
    assert stream.status_code == 200
    assert stream.content_type.startswith('multipart/x-mixed-replace')
    first_part = next(stream.response)
    assert first_part.startswith(b'--frame\r\nContent-Type: image/jpeg')
    assert b'X-Preview-Status: ok' in first_part
    assert b'\xff\xd8' in first_part
    stream.close()

    assert factory_calls == [('WebCameraOpenPi', 2)]
    assert connector.calls == [('WebCameraOpenPi', 2, True)]
    assert channel.closed is False
    assert client.post('/api/config/select', json={'name': 'other'}).status_code == 200
    assert channel.closed is True
    assert channel.unlink_calls == 0

    # shutdown is idempotent and must not turn consumer close into unlink.
    controller = app.config['DEVICE_CONTROLLER']
    controller.shutdown()
    controller.shutdown()
    assert channel.unlink_calls == 0


def test_cookie_browser_requires_current_short_preview_protocol(
    config_dir: Path,
) -> None:
    channel = FakeChannel(_camera_segment())
    connector = FakeConnector([channel])
    app, client, _ = _app_with_connector(config_dir, connector)
    client.post('/api/config/select', json={'name': 'demo'})

    # API clients without a browser cookie remain backwards compatible.
    assert client.get(
        '/api/devices/WebCameraOpenPi/2/preview.jpg'
    ).status_code == 200
    client.get('/')
    stale = client.get('/api/devices/WebCameraOpenPi/2/preview.jpg')
    assert stale.status_code == 409
    assert stale.get_json()['code'] == 'stale_preview_client'

    current = client.get(
        '/api/devices/WebCameraOpenPi/2/preview.jpg',
        query_string={
            'client_version': (
                device_control_server.PREVIEW_BROWSER_CLIENT_VERSION
            )
        },
    )
    assert current.status_code == 200
    assert current.content_type == 'image/jpeg'
    app.config['DEVICE_CONTROLLER'].shutdown()
    assert channel.unlink_calls == 0


def test_main_uses_loopback_defaults_and_signal_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = {}
    callbacks = []
    run_args = {}

    class FakeLogs:
        def append(self, *args: Any, **kwargs: Any) -> None:
            return None

    class FakeController:
        def __init__(self) -> None:
            self.logs = FakeLogs()
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    controller = FakeController()

    class FakeApp:
        config = {'DEVICE_CONTROLLER': controller}

        def run(self, **kwargs: Any) -> None:
            run_args.update(kwargs)
            handler = handlers[signal.SIGTERM]
            handler(signal.SIGTERM, None)

    class FakeLock:
        release_calls = 0

        def __init__(self, path: Path) -> None:
            self.path = path

        def acquire(self) -> None:
            return None

        def release(self) -> None:
            self.release_calls += 1

    fake_lock = FakeLock(Path('/tmp/fake'))
    monkeypatch.setattr(
        device_control_server, 'SingleInstanceLock', lambda path: fake_lock
    )
    monkeypatch.setattr(device_control_server, 'create_app', lambda **kwargs: FakeApp())
    monkeypatch.setattr(
        device_control_server.atexit, 'register', lambda callback: callbacks.append(callback)
    )
    monkeypatch.setattr(
        device_control_server.signal,
        'signal',
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(sys, 'argv', ['device_control_server.py'])

    with pytest.raises(SystemExit) as exited:
        device_control_server.main()

    assert exited.value.code == 128 + signal.SIGTERM
    assert run_args['host'] == '127.0.0.1'
    assert run_args['port'] == 18088
    assert run_args['threaded'] is True
    assert controller.shutdown_calls == 1
    assert fake_lock.release_calls == 1
    callbacks[0]()
    assert controller.shutdown_calls == 1
    assert fake_lock.release_calls == 1
