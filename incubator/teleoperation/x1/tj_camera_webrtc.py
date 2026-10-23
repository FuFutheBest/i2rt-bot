#!/usr/bin/env python3
"""Low-latency WebRTC video server for the TJ head camera (nmxrdk CAMID_0).

改写自 quest3_yam_teleop/test_webrtc_zed_to_quest3.py，把视频源从 ZED 换成
TJ 机器人相机（nmxrdk robot.camera(CAMID_0).get_rgbd）。给 tj_controller_teleop
的 WebRTCStereoVideoPlane 拉流，替代原来的 JPEG-over-WebSocket，显著降低画面延迟。

TJ 相机是单目，这里把同一帧左右复制成 side-by-side 立体画面（与原 hstack 一致）。

在 robot_test 容器里运行（venv，需先 pip install aiortc aiohttp_cors av）：
    source /root/tvenv/bin/activate
    cd /root/incubator/teleoperation/x1
    python3 tj_camera_webrtc.py --https           # 用 certs/ 下自签证书
    # 浏览器自测： https://<主机IP>:8080/  点 Start WebRTC

然后 tj_controller_teleop.py 用:
    python3 tj_controller_teleop.py --arm-side both --https \
        --webrtc-src https://<主机IP>:8080/offer
"""

import argparse
import asyncio
import json
import logging
import ssl
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

# Make the SuperInference project root importable (for utils.shm_utils when
# running in --shm-name mode). This file lives under <root>/incubator/teleoperation/x1/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import aiohttp_cors
import aiortc.codecs.vpx as vpx_codec
import cv2
import numpy as np
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame

import nmxrdk
import nmxrt

from utils.shm_utils import (
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_shared_memory,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
)


DEFAULT_ROBOT_ID = "1e783be2dd870ef5"
DEFAULT_CAM_ID = "CAMID_0"

INDEX_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>TJ Camera WebRTC Test</title>
  <style>
    body { margin: 0; background: #111; color: white; font-family: sans-serif; }
    #bar { position: fixed; top: 0; left: 0; right: 0; padding: 12px; background: rgba(0,0,0,.65); z-index: 2; }
    #stats { position: fixed; top: 56px; left: 0; padding: 8px; background: rgba(0,0,0,.65); z-index: 2; font-size: 13px; white-space: pre-wrap; }
    video { width: 100vw; height: 100vh; object-fit: contain; background: #000; }
    button { font-size: 18px; padding: 8px 16px; }
  </style>
</head>
<body>
  <div id="bar">
    <button id="start" onclick="start()">Start WebRTC</button>
    <span id="status">idle</span>
  </div>
  <pre id="stats">stats: idle</pre>
  <video id="video" autoplay playsinline muted></video>
  <script>
    let pc = null, statsTimer = null;
    const iceServers = __ICE_SERVERS__;
    async function start() {
      document.getElementById("status").textContent = "connecting";
      pc = new RTCPeerConnection({ sdpSemantics: "unified-plan", iceServers });
      pc.addEventListener("connectionstatechange", () => {
        document.getElementById("status").textContent = pc.connectionState;
      });
      pc.addEventListener("track", (evt) => {
        if (evt.track.kind === "video") {
          document.getElementById("video").srcObject = evt.streams[0];
        }
      });
      pc.addTransceiver("video", { direction: "recvonly" });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") resolve();
        else {
          const cs = () => {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", cs); resolve();
            }
          };
          pc.addEventListener("icegatheringstatechange", cs);
        }
      });
      const response = await fetch("/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pc.localDescription)
      });
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
      statsTimer = setInterval(updateStats, 1000);
    }
    async function updateStats() {
      if (!pc) return;
      const report = await pc.getStats();
      let inbound = null, cp = null;
      report.forEach((s) => {
        if (s.type === "inbound-rtp" && s.kind === "video") inbound = s;
        if (s.type === "candidate-pair" && s.state === "succeeded" && s.nominated) cp = s;
      });
      const lines = [];
      if (inbound) {
        lines.push(`recv fps: ${inbound.framesPerSecond ?? "n/a"}`);
        if (inbound.totalDecodeTime && inbound.framesDecoded)
          lines.push(`avg decode: ${(inbound.totalDecodeTime / inbound.framesDecoded * 1000).toFixed(1)} ms`);
      }
      if (cp && cp.currentRoundTripTime)
        lines.push(`RTT: ${(cp.currentRoundTripTime * 1000).toFixed(1)} ms`);
      document.getElementById("stats").textContent = lines.length ? lines.join("\\n") : "stats: waiting";
    }
  </script>
</body>
</html>
"""


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Low-latency WebRTC stream for TJ head camera.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--cam-id", default=DEFAULT_CAM_ID)
    parser.add_argument("--shm-name", default="",
                        help="若指定(如 TjCameraDevice_2_data),从该 SHM 读帧而非直接抓相机"
                             "(数据采集模式:相机由 TjCameraDevice 抓取,避免重复 grab)。")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280, help="输出立体画面总宽(左右拼接)。")
    parser.add_argument("--height", type=int, default=480, help="输出画面高。")
    parser.add_argument("--codec", default="auto", choices=["auto", "video/VP8"])
    parser.add_argument("--video-bitrate-mbps", type=float, default=8.0)
    parser.add_argument("--vpx-threads", type=int, default=8,
                        help="libvpx 编码线程数。")
    parser.add_argument("--https", action="store_true",
                        help="用 certs/ 下自签证书跑 HTTPS（WebRTC 在 https 页面里需要）。")
    parser.add_argument("--cert-file", default=None)
    parser.add_argument("--key-file", default=None)
    parser.add_argument("--ice-server", action="append",
                        help="ICE server URL，例如 stun:stun.l.google.com:19302。可重复。")
    parser.add_argument("--profile", action="store_true", help="打印每阶段耗时。")
    args = parser.parse_args()
    if args.https and not (args.cert_file and args.key_file):
        args.cert_file = args.cert_file or str(here / "certs" / "cert.pem")
        args.key_file = args.key_file or str(here / "certs" / "key.pem")
    return args


def tune_vp8_bitrate(args):
    bitrate = int(args.video_bitrate_mbps * 1_000_000)
    vpx_codec.DEFAULT_BITRATE = bitrate
    vpx_codec.MAX_BITRATE = max(vpx_codec.MAX_BITRATE, bitrate)
    vpx_codec.MAX_FRAME_RATE = args.fps
    forced_threads = max(1, int(args.vpx_threads))
    vpx_codec.number_of_threads = lambda pixels, cpus: forced_threads
    print(f"VP8 tuning: target/max bitrate={args.video_bitrate_mbps:.1f} Mbps, "
          f"fps={args.fps}, threads={forced_threads}")


class TJCaptureWorker:
    """后台线程自由抓取 TJ 相机帧并转成 side-by-side 立体 RGB。

    把阻塞的 nmxrdk get_rgbd 移出 asyncio 事件循环，编码任务可并行按需取最新帧。
    """

    def __init__(self, robot_id, cam_id, output_size, fps):
        self.output_size = output_size            # (W, H) 立体总尺寸
        self.robot = nmxrdk.RobotClient(robot_id)
        self.cam = self.robot.camera(cam_id)

        self._stereo_rgb = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        self._latest_frame = None
        self._frame_index = 0
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()

        self.capture_fps = 0.0
        self._fps_window_count = 0
        self._fps_window_start = time.time()

        self._thread = threading.Thread(target=self._run, name="tj-capture", daemon=True)
        self._thread.start()

    def _run(self):
        half_w = self.output_size[0] // 2
        h = self.output_size[1]
        while not self._stop.is_set():
            try:
                rgb_frame, _ = self.cam.get_rgbd(decode=True)
            except Exception:
                time.sleep(0.005)
                continue
            if isinstance(rgb_frame, nmxrt.Error):
                time.sleep(0.005)
                continue
            data = rgb_frame.get("data") if hasattr(rgb_frame, "get") else None
            if data is None:
                time.sleep(0.005)
                continue

            eye = cv2.resize(data, (half_w, h))
            if eye.ndim == 3 and eye.shape[2] == 3:
                eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
            # 单目复制成左右眼
            self._stereo_rgb[:, :half_w] = eye
            self._stereo_rgb[:, half_w:] = eye

            with self._lock:
                self._latest_frame = self._stereo_rgb.copy()
                self._frame_index += 1
            self._frame_ready.set()

            self._fps_window_count += 1
            now = time.time()
            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                self.capture_fps = self._fps_window_count / elapsed
                self._fps_window_count = 0
                self._fps_window_start = now

    def get_latest(self, last_index, timeout=1.0):
        if not self._frame_ready.wait(timeout=timeout):
            return None, last_index
        with self._lock:
            if self._frame_index <= last_index:
                self._frame_ready.clear()
                return None, last_index
            frame = self._latest_frame
            index = self._frame_index
        return frame, index

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)


class ShmCaptureWorker:
    """读取 TjCameraDevice 发布的单目 BGR SHM 帧，转成 side-by-side 立体 RGB。

    与 TJCaptureWorker 暴露相同的 get_latest / _latest_frame / capture_fps /
    close 接口，因此 TJStereoVideoTrack 无需改动即可使用。本进程不抓相机，相机
    由 TjCameraDevice 独占 grab，避免同一相机被两个进程重复读取。
    """

    def __init__(self, shm_name, output_size, fps):
        self.shm_name = shm_name
        self.output_size = output_size            # (W, H) 立体总尺寸
        self.fps = max(1, int(fps))

        self._shm = None
        self._stereo_rgb = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        self._latest_frame = None
        self._frame_index = 0
        self._last_shm_write_index = -1
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()

        self.capture_fps = 0.0
        self._fps_window_count = 0
        self._fps_window_start = time.time()

        self._thread = threading.Thread(target=self._run, name="shm-capture", daemon=True)
        self._thread.start()

    def _connect(self):
        if self._shm is not None:
            return True
        try:
            self._shm = connect_to_shared_memory(self.shm_name, read_only=True, log_missing=False)
            print(f"[shm] connected to {self.shm_name}")
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            print(f"[shm] connect failed: {exc!r}")
            return False

    def _read_latest_mono(self):
        if not self._connect():
            return None
        buf = self._shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )
        if buffer_header["current_frames_count"] == 0:
            return None
        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        if write_index == self._last_shm_write_index:
            return None
        self._last_shm_write_index = write_index

        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        dtype = get_dtype(device_header["data_dtype"], np.uint8)
        arr = np.frombuffer(buf[data_start : data_start + data_size], dtype=dtype)
        shape = device_header["shape"]
        if len(shape) > 0:
            arr = arr.reshape(shape)
        return np.ascontiguousarray(arr)

    def _run(self):
        half_w = self.output_size[0] // 2
        h = self.output_size[1]
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                mono = self._read_latest_mono()
            except Exception as exc:
                print(f"[shm] read failed: {exc!r}")
                mono = None

            if mono is not None:
                eye = cv2.resize(mono, (half_w, h))
                if eye.ndim == 3 and eye.shape[2] == 3:
                    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
                self._stereo_rgb[:, :half_w] = eye
                self._stereo_rgb[:, half_w:] = eye
                with self._lock:
                    self._latest_frame = self._stereo_rgb.copy()
                    self._frame_index += 1
                self._frame_ready.set()

                self._fps_window_count += 1
                now = time.time()
                elapsed = now - self._fps_window_start
                if elapsed >= 1.0:
                    self.capture_fps = self._fps_window_count / elapsed
                    self._fps_window_count = 0
                    self._fps_window_start = now

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            time.sleep(sleep_for if sleep_for > 0 else 0)
            if sleep_for <= 0:
                next_tick = time.monotonic()

    def get_latest(self, last_index, timeout=1.0):
        if not self._frame_ready.wait(timeout=timeout):
            return None, last_index
        with self._lock:
            if self._frame_index <= last_index:
                self._frame_ready.clear()
                return None, last_index
            frame = self._latest_frame
            index = self._frame_index
        return frame, index

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None


class TJStereoVideoTrack(VideoStreamTrack):
    def __init__(self, robot_id, cam_id, fps, output_size, profile=False, shm_name=""):
        super().__init__()
        self.fps = fps
        self.output_size = output_size
        self.profile = profile
        self.frame_count = 0
        self.last_report = time.time()

        self._clock_rate = 90000
        self._timestamp = 0
        self._timestamp_step = int(self._clock_rate / fps)
        self._start_time = time.time()
        self._time_base = Fraction(1, self._clock_rate)
        self._last_capture_id = 0

        if shm_name:
            print(f"[track] SHM mode: reading frames from '{shm_name}'")
            self.capture = ShmCaptureWorker(shm_name, output_size, fps)
        else:
            print(f"[track] direct-grab mode: cam_id={cam_id}")
            self.capture = TJCaptureWorker(robot_id, cam_id, output_size, fps)

    async def recv(self):
        target_time = self._start_time + (self._timestamp / self._clock_rate)
        wait_time = target_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        pts = self._timestamp
        self._timestamp += self._timestamp_step

        loop = asyncio.get_event_loop()
        frame_rgb, self._last_capture_id = await loop.run_in_executor(
            None, self.capture.get_latest, self._last_capture_id, 1.0
        )
        if frame_rgb is None:
            frame_rgb = self.capture._latest_frame
            if frame_rgb is None:
                frame_rgb = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)

        self.frame_count += 1
        now = time.time()
        if now - self.last_report >= 1.0:
            elapsed = now - self.last_report
            print(f"webrtc send fps: {self.frame_count / elapsed:.1f} | "
                  f"tj capture fps: {self.capture.capture_fps:.1f}")
            self.frame_count = 0
            self.last_report = now

        frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = self._time_base
        return frame

    def close(self):
        self.capture.close()


def force_codec(pc, sender, forced_codec):
    if forced_codec == "auto":
        return
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    codecs = RTCRtpSender.getCapabilities("video").codecs
    preferred = [c for c in codecs if c.mimeType == forced_codec]
    if preferred:
        transceiver.setCodecPreferences(preferred)


async def wait_for_ice_gathering_complete(pc):
    if pc.iceGatheringState == "complete":
        return
    complete = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def check_state():
        if pc.iceGatheringState == "complete":
            complete.set()

    await complete.wait()


def build_ice_servers(args):
    urls = args.ice_server or ["stun:stun.l.google.com:19302"]
    return [{"urls": urls, "username": "", "credential": ""}]


async def index(request):
    ice_servers = build_ice_servers(request.app["args"])
    html = INDEX_HTML_TEMPLATE.replace("__ICE_SERVERS__", json.dumps(ice_servers))
    return web.Response(content_type="text/html", text=html)


async def offer(request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    ice_servers = [
        RTCIceServer(urls=s["urls"], username=s["username"], credential=s["credential"])
        for s in build_ice_servers(request.app["args"])
    ]
    pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
    request.app["pcs"].add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"WebRTC connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            request.app["pcs"].discard(pc)

    args = request.app["args"]
    track = TJStereoVideoTrack(
        args.robot_id, args.cam_id, args.fps,
        (args.width, args.height), profile=args.profile,
        shm_name=args.shm_name,
    )
    request.app["tracks"].add(track)

    sender = pc.addTrack(track)
    force_codec(pc, sender, args.codec)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await wait_for_ice_gathering_complete(pc)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in app["pcs"]])
    for track in app["tracks"]:
        track.close()
    app["pcs"].clear()
    app["tracks"].clear()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    tune_vp8_bitrate(args)

    app = web.Application()
    app["args"] = args
    app["pcs"] = set()
    app["tracks"] = set()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*",
            allow_headers="*", allow_methods="*"),
    })
    for route in list(app.router.routes()):
        cors.add(route)

    ssl_context = None
    scheme = "http"
    if args.cert_file and args.key_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
        scheme = "https"

    print(f"Serving TJ camera WebRTC on {scheme}://{args.host}:{args.port}  "
          f"(offer 端点: {scheme}://<主机IP>:{args.port}/offer)")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
