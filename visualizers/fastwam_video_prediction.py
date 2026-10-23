#!/usr/bin/env python3
"""Six-panel FastWAM camera/input/prediction video dashboard.

The dashboard is intentionally dependency-light: a standard-library HTTP server
serves six MJPEG streams, while OpenCV handles frame conversion and encoding.
The inference process can therefore be viewed locally or from another machine
without adding a web framework to the robot environment.
"""

from __future__ import annotations

import html
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from utils.logger_config import logger
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    DataChannel,
    connect_to_device_shm,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
)


_PANEL_NAMES = (
    "left_camera",
    "left_input",
    "left_prediction",
    "right_camera",
    "right_input",
    "right_prediction",
)
_PANEL_TITLES = {
    "left_camera": "Left hand | camera",
    "left_input": "Left hand | input observation",
    "left_prediction": "Left hand | prediction",
    "right_camera": "Right hand | camera",
    "right_input": "Right hand | input observation",
    "right_prediction": "Right hand | prediction",
}


def _normalise_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Return an HWC RGB uint8 frame for either CHW or HWC input."""
    array = np.asarray(frame)
    if array.size == 0:
        raise ValueError("Video dashboard frame is empty")
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an image with shape HWC/CHW, got {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.min()) >= -1.0 and float(finite.max()) <= 1.0:
            array = (array + 1.0) * 127.5 if float(finite.min()) < 0.0 else array * 255.0
    return np.ascontiguousarray(np.clip(array, 0.0, 255.0).astype(np.uint8))


class _CameraSHMReader:
    """Read the latest RGB frame from one device shared-memory ring buffer."""

    def __init__(self, shared_memory_name: str) -> None:
        self.shared_memory_name = str(shared_memory_name)
        self.channel: Optional[DataChannel] = None

    def connect(self) -> bool:
        if self.channel is not None:
            return True
        if not self.shared_memory_name.endswith("_data"):
            raise ValueError(
                f"Camera SHM name must end with '_data': {self.shared_memory_name}"
            )
        base_name = self.shared_memory_name[:-5]
        device_name, separator, device_class_id_text = base_name.rpartition("_")
        if not separator or not device_name:
            raise ValueError(f"Invalid camera SHM name: {self.shared_memory_name}")
        try:
            self.channel = connect_to_device_shm(
                device_name,
                int(device_class_id_text),
                read_only=True,
            )
            logger.info(
                f"FastWAM dashboard connected to camera SHM: {self.shared_memory_name}"
            )
            return True
        except FileNotFoundError:
            return False

    def close(self) -> None:
        channel = self.channel
        self.channel = None
        if channel is not None:
            channel.close()

    def read_latest(self) -> Optional[np.ndarray]:
        channel = self.channel
        if channel is None:
            return None
        device_header = unpack_device_header(channel.buf[:DEVICE_HEADER_SIZE])
        buffer_start = DEVICE_HEADER_SIZE
        buffer_header = unpack_buffer_header(
            channel.buf[buffer_start : buffer_start + BUFFER_HEADER_SIZE]
        )
        if int(buffer_header["current_frames_count"]) <= 0:
            return None

        buffer_size = int(buffer_header["buffer_size"])
        write_index = int(buffer_header["write_index"])
        frame_size = int(device_header["frame_size"])
        latest_index = (write_index - 1) % buffer_size
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
        data_start = frame_offset + FRAME_HEADER_SIZE
        data_end = frame_offset + frame_size
        dtype = get_dtype(str(device_header["data_dtype"]))
        frame = np.frombuffer(
            channel.buf[data_start:data_end],
            dtype=dtype,
        ).reshape(tuple(device_header["shape"]))
        return np.array(frame, copy=True)


class FastWAMVideoPredictionVisualizer:
    """Serve live camera, model input, and prediction views for both wrists."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        display_fps: float = 8.0,
        panel_size: Tuple[int, int] = (640, 360),
        jpeg_quality: int = 85,
        auto_open_browser: bool = False,
        camera_shm_names: Optional[Sequence[str]] = None,
        live_camera_fps: float = 30.0,
        max_prediction_frames: int = 16,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.display_fps = max(float(display_fps), 1.0)
        self.panel_width = max(int(panel_size[0]), 64)
        self.panel_height = max(int(panel_size[1]), 64)
        self.jpeg_quality = max(20, min(int(jpeg_quality), 95))
        self.auto_open_browser = bool(auto_open_browser)
        self.live_camera_fps = max(float(live_camera_fps), 1.0)
        self.max_prediction_frames = max(int(max_prediction_frames), 0)
        shm_names = list(camera_shm_names or [])
        if shm_names and len(shm_names) != 2:
            raise ValueError(
                "FastWAM dashboard camera_shm_names must contain left and right SHMs"
            )
        self._camera_readers = [_CameraSHMReader(name) for name in shm_names]

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._camera_thread: Optional[threading.Thread] = None
        self._closed = False
        self._observation_index = 0
        self._prediction_chunk_index = 0
        self._prediction_index = 0
        self._prediction_frame_count = 0
        self._prediction_sequences: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
        self._frames: Dict[str, bytes] = {}
        self._versions: Dict[str, int] = {name: 0 for name in _PANEL_NAMES}
        self._placeholder_jpeg = self._make_placeholder()

    @property
    def url(self) -> Optional[str]:
        if self._server is None:
            return None
        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self._server.server_port}/"

    def start(self) -> Optional[str]:
        """Start the dashboard and return its browser URL."""
        if self._server is not None:
            return self.url
        handler_cls = self._make_handler()
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), handler_cls)
            self._server.daemon_threads = True
        except Exception as exc:
            logger.warning(f"FastWAM video dashboard could not start: {exc}")
            self._server = None
            return None

        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="fastwam-video-dashboard-http",
            daemon=True,
        )
        self._server_thread.start()
        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="fastwam-video-dashboard-render",
            daemon=True,
        )
        self._render_thread.start()
        if self._camera_readers:
            self._camera_thread = threading.Thread(
                target=self._camera_loop,
                name="fastwam-video-dashboard-camera",
                daemon=True,
            )
            self._camera_thread.start()
        dashboard_url = self.url
        logger.info(f"FastWAM video dashboard available at {dashboard_url}")
        if self.auto_open_browser and dashboard_url:
            try:
                webbrowser.open(dashboard_url)
            except Exception as exc:
                logger.debug(f"Could not open FastWAM dashboard browser: {exc}")
        return dashboard_url

    def update(
        self,
        input_frames: Sequence[np.ndarray],
        prediction_frames: Optional[Sequence[Sequence[np.ndarray]]] = None,
        *,
        observation_index: Optional[int] = None,
    ) -> None:
        """Publish the latest two input views and optional prediction sequences."""
        if self._server is None or self._closed:
            return
        try:
            inputs = list(input_frames)
            if len(inputs) != 2:
                logger.warning(
                    "FastWAM dashboard requires exactly two input views, got "
                    f"{len(inputs)}"
                )
                return
            input_jpegs = [self._encode_frame(inputs[index]) for index in range(2)]
            sequences: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
            if prediction_frames is not None:
                predicted = list(prediction_frames)
                for side, index in (("left", 0), ("right", 1)):
                    if index < len(predicted):
                        sequences[side] = [
                            _normalise_rgb_frame(frame)
                            for frame in predicted[index]
                            if frame is not None
                        ]
            with self._condition:
                if observation_index is not None:
                    self._observation_index = max(int(observation_index), 0)
                self._set_frame_locked("left_input", input_jpegs[0])
                self._set_frame_locked("right_input", input_jpegs[1])
                if prediction_frames is not None:
                    self._prediction_chunk_index += 1
                    self._prediction_sequences = sequences
                    self._prediction_index = 0
                    self._prediction_frame_count = max(
                        (len(sequence) for sequence in sequences.values()),
                        default=0,
                    )
                    self._publish_prediction_frames_locked()
                self._condition.notify_all()
        except Exception as exc:
            logger.warning(f"FastWAM video dashboard frame update failed: {exc}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as exc:
                logger.debug(f"FastWAM dashboard server shutdown failed: {exc}")
        for thread in (self._camera_thread, self._render_thread, self._server_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        for reader in self._camera_readers:
            reader.close()
        self._camera_thread = None
        self._render_thread = None
        self._server_thread = None
        self._server = None

    def _camera_loop(self) -> None:
        interval = 1.0 / self.live_camera_fps
        last_connect_attempt = 0.0
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            if any(reader.channel is None for reader in self._camera_readers):
                if loop_start - last_connect_attempt >= 1.0:
                    last_connect_attempt = loop_start
                    for reader in self._camera_readers:
                        if reader.channel is None:
                            try:
                                reader.connect()
                            except Exception as exc:
                                logger.warning(
                                    "FastWAM dashboard camera SHM connection failed "
                                    f"for {reader.shared_memory_name}: {exc}"
                                )
            for panel_name, reader in zip(
                ("left_camera", "right_camera"),
                self._camera_readers,
            ):
                try:
                    frame = reader.read_latest()
                    if frame is not None:
                        jpeg = self._encode_frame(frame)
                        with self._condition:
                            self._set_frame_locked(panel_name, jpeg)
                            self._condition.notify_all()
                except Exception as exc:
                    logger.warning(
                        "FastWAM dashboard camera frame read failed for "
                        f"{reader.shared_memory_name}: {exc}"
                    )
                    reader.close()
            elapsed = time.monotonic() - loop_start
            self._stop_event.wait(max(interval - elapsed, 0.0))

    def _render_loop(self) -> None:
        interval = 1.0 / self.display_fps
        while not self._stop_event.wait(interval):
            with self._condition:
                if any(self._prediction_sequences.values()):
                    self._prediction_index += 1
                    self._publish_prediction_frames_locked()
                    self._condition.notify_all()

    def _publish_prediction_frames_locked(self) -> None:
        for side, panel_name in (("left", "left_prediction"), ("right", "right_prediction")):
            sequence = self._prediction_sequences.get(side, [])
            if sequence:
                index = self._prediction_index % len(sequence)
                self._set_frame_locked(panel_name, self._encode_frame(sequence[index]))
            else:
                self._set_frame_locked(panel_name, self._placeholder_jpeg)

    def _set_frame_locked(self, panel_name: str, jpeg: bytes) -> None:
        self._frames[panel_name] = jpeg
        self._versions[panel_name] += 1

    def _get_frame(self, panel_name: str, version: int, timeout: float) -> Tuple[int, bytes]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._closed and self._versions.get(panel_name, 0) <= version:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return (
                self._versions.get(panel_name, 0),
                self._frames.get(panel_name, self._placeholder_jpeg),
            )

    def _get_status(self) -> Dict[str, float | int]:
        with self._condition:
            current_frame = (
                self._prediction_index % self._prediction_frame_count + 1
                if self._prediction_frame_count > 0
                else 0
            )
            return {
                "observation_index": self._observation_index,
                "prediction_chunk_index": self._prediction_chunk_index,
                "prediction_frame_index": current_frame,
                "prediction_frame_count": self._prediction_frame_count,
                "display_fps": self.display_fps,
                "live_camera_fps": self.live_camera_fps,
                "max_prediction_frames": self.max_prediction_frames,
            }

    def _get_snapshot(self, panel_name: str) -> bytes:
        with self._condition:
            return self._frames.get(panel_name, self._placeholder_jpeg)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        rgb = _normalise_rgb_frame(frame)
        scale = min(self.panel_width / rgb.shape[1], self.panel_height / rgb.shape[0])
        target_width = max(1, min(self.panel_width, int(round(rgb.shape[1] * scale))))
        target_height = max(1, min(self.panel_height, int(round(rgb.shape[0] * scale))))
        resized = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
        top = (self.panel_height - target_height) // 2
        left = (self.panel_width - target_width) // 2
        canvas[top : top + target_height, left : left + target_width] = resized
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode a dashboard frame")
        return encoded.tobytes()

    def _make_placeholder(self) -> bytes:
        canvas = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "waiting for frames",
            (32, self.panel_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", canvas)
        return encoded.tobytes() if ok else b""

    def _make_handler(self):
        visualizer = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/" or path == "/index.html":
                    self._send_html()
                    return
                if path == "/healthz":
                    self._send_bytes(b'{"ok":true}', "application/json")
                    return
                if path == "/status":
                    self._send_bytes(
                        json.dumps(visualizer._get_status()).encode("utf-8"),
                        "application/json",
                    )
                    return
                snapshot_prefix = "/snapshot/"
                if path.startswith(snapshot_prefix):
                    panel_name = path[len(snapshot_prefix) :]
                    if panel_name in _PANEL_NAMES:
                        self._send_bytes(
                            visualizer._get_snapshot(panel_name),
                            "image/jpeg",
                        )
                        return
                prefix = "/stream/"
                if path.startswith(prefix):
                    panel_name = path[len(prefix) :]
                    if panel_name in _PANEL_NAMES:
                        self._stream(panel_name)
                        return
                self.send_error(HTTPStatus.NOT_FOUND)

            def _send_html(self) -> None:
                panels = "".join(
                    f'<figure class="{html.escape(name)}">'
                    f'<figcaption><span class="dot"></span>'
                    f'{html.escape(_PANEL_TITLES[name])}</figcaption>'
                    f'<img id="{html.escape(name)}" '
                    f'src="/{"snapshot" if name.endswith("_input") else "stream"}/{name}" '
                    f'alt="{html.escape(_PANEL_TITLES[name])}"></figure>'
                    for name in _PANEL_NAMES
                )
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                    "<title>FastWAM video prediction</title>"
                    "<style>"
                    ":root{color-scheme:dark;--bg:#080b12;--card:#111827;"
                    "--border:#263248;--text:#edf4ff;--muted:#8fa4bf;--cyan:#36d7ff;"
                    "--violet:#a78bfa;--green:#4ade80}*{box-sizing:border-box}"
                    "body{margin:0;background:radial-gradient(circle at 50% -20%,#172554 0,"
                    "var(--bg) 42%);color:var(--text);font-family:Inter,ui-sans-serif,"
                    "system-ui,sans-serif;overflow:hidden}header{height:82px;display:flex;"
                    "align-items:center;justify-content:space-between;padding:14px 18px 12px;"
                    "border-bottom:1px solid rgba(143,164,191,.18);"
                    "background:rgba(8,11,18,.82);backdrop-filter:blur(16px)}"
                    ".brand{display:flex;align-items:center;gap:12px}.logo{width:38px;height:38px;"
                    "display:grid;place-items:center;border-radius:11px;background:linear-gradient("
                    "135deg,var(--cyan),var(--violet));color:#071018;font-weight:900;"
                    "box-shadow:0 0 28px rgba(54,215,255,.25)}h1{font-size:18px;margin:0;"
                    "letter-spacing:.02em}.subtitle{margin-top:3px;color:var(--muted);"
                    "font-size:12px}.metrics{display:flex;gap:8px;flex-wrap:wrap;"
                    "justify-content:flex-end}.metric{min-width:106px;padding:7px 10px;"
                    "border:1px solid var(--border);border-radius:10px;background:rgba(17,24,39,.8)}"
                    ".metric-label{display:block;color:var(--muted);font-size:10px;"
                    "letter-spacing:.08em;text-transform:uppercase}.metric-value{display:block;"
                    "margin-top:2px;font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}"
                    "main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));"
                    "grid-template-rows:repeat(2,minmax(0,1fr));gap:10px;padding:10px;"
                    "height:calc(100vh - 82px)}figure{margin:0;position:relative;"
                    "background:linear-gradient(145deg,#111827,#090d15);min-width:0;min-height:0;"
                    "overflow:hidden;border:1px solid var(--border);border-radius:13px;"
                    "box-shadow:0 10px 30px rgba(0,0,0,.24)}figure::after{content:'';"
                    "position:absolute;inset:0;pointer-events:none;"
                    "box-shadow:inset 0 0 35px rgba(0,0,0,.28)}figcaption{position:absolute;"
                    "z-index:2;left:10px;top:10px;display:flex;align-items:center;gap:7px;"
                    "padding:6px 10px;border:1px solid rgba(255,255,255,.11);border-radius:8px;"
                    "background:rgba(5,8,14,.72);backdrop-filter:blur(10px);font-size:12px;"
                    "font-weight:650;letter-spacing:.02em}.dot{width:7px;height:7px;border-radius:50%;"
                    "background:var(--cyan);box-shadow:0 0 10px var(--cyan)}"
                    ".left_prediction .dot,.right_prediction .dot{background:var(--violet);"
                    "box-shadow:0 0 10px var(--violet)}.left_camera .dot,.right_camera .dot{"
                    "background:var(--green);box-shadow:0 0 10px var(--green)}"
                    "img{display:block;width:100%;height:100%;object-fit:contain}"
                    "@media(max-width:900px){header{height:106px;align-items:flex-start}"
                    ".metrics{max-width:62%}.metric{min-width:92px;padding:5px 8px}"
                    "main{height:calc(100vh - 106px);gap:6px;padding:6px}}"
                    "</style></head><body><header><div class='brand'><div class='logo'>W</div>"
                    "<div><h1>FastWAM Prediction Monitor</h1>"
                    "<div class='subtitle'>Live camera · model observation · world prediction</div>"
                    "</div></div><div class='metrics'>"
                    "<div class='metric'><span class='metric-label'>Observation</span>"
                    "<span class='metric-value' id='obs'>#0</span></div>"
                    "<div class='metric'><span class='metric-label'>Prediction chunk</span>"
                    "<span class='metric-value' id='chunk'>#0</span></div>"
                    "<div class='metric'><span class='metric-label'>Prediction frame</span>"
                    "<span class='metric-value' id='frame'>0 / 0</span></div>"
                    "<div class='metric'><span class='metric-label'>Display FPS</span>"
                    "<span class='metric-value' id='fps'>—</span></div>"
                    "<div class='metric'><span class='metric-label'>Max frames</span>"
                    "<span class='metric-value' id='maxframes'>—</span></div>"
                    "</div></header><main>"
                    + panels
                    + "</main><script>"
                    "const set=(id,value)=>document.getElementById(id).textContent=value;"
                    "let lastObservation=-1;"
                    "async function refresh(){try{const r=await fetch('/status',{cache:'no-store'});"
                    "const s=await r.json();set('obs','#'+s.observation_index);"
                    "set('chunk','#'+s.prediction_chunk_index);"
                    "set('frame',s.prediction_frame_index+' / '+s.prediction_frame_count);"
                    "set('fps',Number(s.display_fps).toFixed(1));"
                    "set('maxframes',s.max_prediction_frames);"
                    "if(s.observation_index!==lastObservation){lastObservation=s.observation_index;"
                    "for(const id of ['left_input','right_input']){"
                    "document.getElementById(id).src='/snapshot/'+id+'?obs='+s.observation_index;}}"
                    "}catch(_){}}"
                    "refresh();setInterval(refresh,250);"
                    "</script></body></html>"
                )
                self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")

            def _send_bytes(self, payload: bytes, content_type: str) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _stream(self, panel_name: str) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                version = -1
                try:
                    while not visualizer._closed:
                        version, jpeg = visualizer._get_frame(panel_name, version, timeout=1.0)
                        if not jpeg:
                            continue
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def log_message(self, format: str, *args: object) -> None:
                return

        return DashboardHandler


__all__ = ["FastWAMVideoPredictionVisualizer"]
