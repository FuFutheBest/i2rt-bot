"""Stereo SHM -> WebRTC server for Quest 3 (SuperInference port of R1's
``zed_webrtc_server.py``).

Difference from R1: this process does NOT open the ZED camera. Instead it
READS the side-by-side stereo RGB frames published by ``ZedStereoDevice``
(the single ``zed.grab()`` owner) from its shared memory
(``<device>_<id>_data``) and re-encodes them as a WebRTC VP8 stream behind a
``/offer`` endpoint. The Quest browser / Vuer ``WebRTCStereoVideoPlane``
consume that endpoint via ``Quest3VuerTeleoperator``'s ``webrtc_src``.

So the data path is:

    ZedStereoDevice (grab -> SHM)  ->  zed_webrtc_server (SHM -> WebRTC)  ->  Quest

Quick start (HTTPS, which the Quest browser requires):

    python scripts/zed_webrtc_server.py \\
        --port 8080 \\
        --shm-name ZedStereoDevice_0_data \\
        --width 2560 --height 720 \\
        --cert-file certs/cert.pem --key-file certs/key.pem

Then point the teleoperator at it (config ``webrtc_src``):

    https://<HOST_IP>:8080/offer
"""

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
from fractions import Fraction

# This script lives in scripts/; add the repo root so ``from utils...`` resolves
# whether it is launched directly or auto-started by device_starter.py.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import aiohttp_cors
import aiortc.codecs.vpx as vpx_codec
import numpy as np
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.codecs.vpx import Vp8Encoder
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame

from incubator.r1.shm_capture_worker import (
    LatestFrameShmCaptureWorker as ShmStereoCaptureWorker,
)


INDEX_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Stereo SHM WebRTC</title>
  <style>
    body { margin: 0; background: #111; color: white; font-family: sans-serif; }
    #bar { position: fixed; top: 0; left: 0; right: 0; padding: 12px; background: rgba(0,0,0,.65); z-index: 2; }
    #stats { position: fixed; top: 56px; left: 0; padding: 8px; background: rgba(0,0,0,.65); z-index: 2; font-size: 13px; white-space: pre-wrap; }
    #log { position: fixed; left: 0; right: 0; bottom: 0; max-height: 35vh; overflow: auto; padding: 8px; background: rgba(0,0,0,.65); font-size: 13px; white-space: pre-wrap; }
    video { width: 100vw; height: 100vh; object-fit: contain; background: #000; }
    button { font-size: 18px; padding: 8px 16px; }
  </style>
</head>
<body>
  <div id="bar">
    <button id="start" onclick="start()">Start WebRTC</button>
    <button id="stop" onclick="stop()" style="display:none">Stop</button>
    <span id="status">idle</span>
  </div>
  <pre id="stats">stats: idle</pre>
  <video id="video" autoplay playsinline muted></video>
  <pre id="log"></pre>
  <script>
    let pc = null;
    let statsTimer = null;
    const iceServers = __ICE_SERVERS__;

    function log(message) {
      const line = `${new Date().toLocaleTimeString()} ${message}`;
      console.log(line);
      document.getElementById("log").textContent += line + "\\n";
    }

    async function start() {
      document.getElementById("status").textContent = "connecting";
      document.getElementById("start").style.display = "none";
      document.getElementById("stop").style.display = "inline-block";

      pc = new RTCPeerConnection({
        sdpSemantics: "unified-plan",
        iceServers: iceServers
      });

      pc.addEventListener("connectionstatechange", () => {
        document.getElementById("status").textContent = pc.connectionState;
        log(`connectionState=${pc.connectionState}`);
      });

      pc.addEventListener("iceconnectionstatechange", () => {
        log(`iceConnectionState=${pc.iceConnectionState}`);
      });

      pc.addEventListener("track", (evt) => {
        if (evt.track.kind === "video") {
          log("video track received");
          document.getElementById("video").srcObject = evt.streams[0];
        }
      });

      pc.addTransceiver("video", { direction: "recvonly" });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          const checkState = () => {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", checkState);
              resolve();
            }
          };
          pc.addEventListener("icegatheringstatechange", checkState);
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
      if (!pc) {
        return;
      }
      const report = await pc.getStats();
      let inbound = null;
      let candidatePair = null;
      report.forEach((stat) => {
        if (stat.type === "inbound-rtp" && stat.kind === "video") {
          inbound = stat;
        }
        if (stat.type === "candidate-pair" && stat.state === "succeeded" && stat.nominated) {
          candidatePair = stat;
        }
      });

      const lines = [];
      if (inbound) {
        lines.push(`recv fps: ${inbound.framesPerSecond ?? "n/a"}`);
        lines.push(`frames decoded: ${inbound.framesDecoded ?? "n/a"}`);
      }
      if (candidatePair && candidatePair.currentRoundTripTime) {
        lines.push(`RTT: ${(candidatePair.currentRoundTripTime * 1000).toFixed(1)} ms`);
      }
      document.getElementById("stats").textContent = lines.length ? lines.join("\\n") : "stats: waiting";
    }

    function stop() {
      document.getElementById("stop").style.display = "none";
      document.getElementById("start").style.display = "inline-block";
      if (pc) {
        pc.close();
        pc = null;
      }
      if (statsTimer) {
        clearInterval(statsTimer);
        statsTimer = null;
      }
      document.getElementById("status").textContent = "stopped";
    }
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-latency stereo WebRTC stream for Quest 3, sourced from "
        "ZedStereoDevice's shared memory (no direct camera access)."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--shm-name", default="ZedStereoDevice_0_data",
                        help="Shared memory name written by ZedStereoDevice.")
    parser.add_argument("--fps", type=int, default=60,
                        help="WebRTC output frame rate (and SHM poll rate).")
    parser.add_argument("--width", type=int, default=2560, help="Stereo frame width (must match the device).")
    parser.add_argument("--height", type=int, default=720, help="Stereo frame height (must match the device).")
    parser.add_argument("--codec", default="auto", choices=["auto", "video/VP8"])
    parser.add_argument("--video-bitrate-mbps", type=float, default=8.0)
    parser.add_argument("--vpx-threads", type=int, default=8,
                        help="libvpx encoder threads (raise to use more CPU cores).")
    parser.add_argument("--cert-file", help="TLS certificate file (Quest browser requires HTTPS).")
    parser.add_argument("--key-file", help="TLS key file.")
    parser.add_argument("--ice-server", action="append",
                        help="ICE server URL, e.g. stun:stun.l.google.com:19302. May be repeated.")
    parser.add_argument("--enable-stun", action="store_true",
                        help="Include public STUN servers in ICE (default OFF for LAN).")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-stage timing to locate FPS bottlenecks.")
    parser.add_argument("--keyframe-interval", type=int, default=60,
                        help="Force a VP8 keyframe every N frames (default 60 = 1s @ 60fps).")
    parser.add_argument(
        "--first-frame-log",
        action="store_true",
        default=False,
        help="Print latency from /offer received -> first frame emitted.",
    )
    return parser.parse_args()


def tune_vp8_bitrate(args):
    bitrate = int(args.video_bitrate_mbps * 1_000_000)
    vpx_codec.DEFAULT_BITRATE = bitrate
    vpx_codec.MAX_BITRATE = max(vpx_codec.MAX_BITRATE, bitrate)
    vpx_codec.MAX_FRAME_RATE = args.fps

    forced_threads = max(1, int(args.vpx_threads))
    vpx_codec.number_of_threads = lambda pixels, cpus: forced_threads

    _kf_dist = max(1, int(args.keyframe_interval))
    _orig_encode = Vp8Encoder.encode

    def _patched_encode(self, frame, force_keyframe=False):
        had_codec = self.codec is not None
        result = _orig_encode(self, frame, force_keyframe)
        if not had_codec and self.codec is not None:
            try:
                self.cfg.kf_max_dist = _kf_dist
                vpx_codec.lib.vpx_codec_enc_config_set(self.codec, self.cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[vp8] failed to apply kf_max_dist={_kf_dist}: {exc!r}")
        return result

    Vp8Encoder.encode = _patched_encode

    print(
        f"VP8 tuning: target/max bitrate={args.video_bitrate_mbps:.1f} Mbps, "
        f"fps={args.fps}, threads={forced_threads}, "
        f"keyframe every {_kf_dist} frames (~{_kf_dist / max(1, args.fps):.2f}s)"
    )


class ZedStereoVideoTrack(VideoStreamTrack):
    def __init__(self, capture, fps, output_size, profile=False,
                 first_frame_log=True, created_at=None, client_tag=""):
        super().__init__()
        self.fps = max(1, int(fps))
        self.output_size = output_size
        self.profile = profile
        self.first_frame_log = first_frame_log
        self.created_at = created_at if created_at is not None else time.time()
        self.client_tag = client_tag
        self._first_frame_sent = False
        self.frame_count = 0
        self.last_report = time.time()

        self._clock_rate = 90000
        self._timestamp = 0
        self._timestamp_step = int(self._clock_rate / self.fps)
        self._start_time = time.time()
        self._time_base = Fraction(1, self._clock_rate)
        self._last_capture_id = 0

        # Shared singleton owned by main(); we do NOT close it here.
        self.capture = capture

    async def recv(self):
        target_time = self._start_time + (self._timestamp / self._clock_rate)
        wait_time = target_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        pts = self._timestamp
        self._timestamp += self._timestamp_step

        # Non-blocking fetch: this is a LIVE stream, so we pace ourselves to
        # `fps` via the sleep above and always send the most-recent frame. The
        # old blocking wait-for-a-newer-index (timeout=1.0) turned any gap in
        # SHM publishing into a hard 1 fps freeze (each recv() blocked the full
        # 1.0s timeout). With timeout=0.0, get_latest returns immediately; if no
        # newer frame is ready we just resend the last one, so the send rate is
        # bounded only by the encoder, never by upstream frame-arrival jitter.
        loop = asyncio.get_event_loop()
        frame_rgb, self._last_capture_id = await loop.run_in_executor(
            None, self.capture.get_latest, self._last_capture_id, 0.0
        )
        if frame_rgb is None:
            frame_rgb = self.capture._latest_frame
            if frame_rgb is None:
                frame_rgb = np.zeros(
                    (self.output_size[1], self.output_size[0], 3), dtype=np.uint8
                )

        self.frame_count += 1
        now = time.time()
        if now - self.last_report >= 1.0:
            elapsed = now - self.last_report
            print(
                f"webrtc recv fps: {self.frame_count / elapsed:.1f} | "
                f"shm capture fps: {self.capture.capture_fps:.1f}"
            )
            self.frame_count = 0
            self.last_report = now

        frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = self._time_base

        if not self._first_frame_sent:
            self._first_frame_sent = True
            if self.first_frame_log:
                latency = time.time() - self.created_at
                print(
                    f"[track{(' ' + self.client_tag) if self.client_tag else ''}] "
                    f"first frame ready {latency * 1000:.0f}ms after track creation"
                )
        return frame

    def close(self):
        pass


def force_codec(pc, sender, forced_codec):
    if forced_codec == "auto":
        print("Codec preference: auto (Quest/Vuer offer order)")
        return
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    codecs = RTCRtpSender.getCapabilities("video").codecs
    preferred_codecs = [codec for codec in codecs if codec.mimeType == forced_codec]
    if preferred_codecs:
        transceiver.setCodecPreferences(preferred_codecs)


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
    if args.ice_server:
        urls = args.ice_server
    elif args.enable_stun:
        urls = ["stun:stun.l.google.com:19302"]
    else:
        return []
    return [{"urls": urls, "username": "", "credential": ""}]


async def index(request):
    ice_servers = build_ice_servers(request.app["args"])
    html = INDEX_HTML_TEMPLATE.replace("__ICE_SERVERS__", json.dumps(ice_servers))
    return web.Response(content_type="text/html", text=html)


async def offer(request):
    t_offer_received = time.time()
    try:
        params = await request.json()
        sdp = params.get("sdp")
        typ = params.get("type")
        if not isinstance(sdp, str) or not isinstance(typ, str):
            raise ValueError("missing sdp/type")
    except Exception:
        return web.json_response({"error": "Invalid offer payload"}, status=400)

    offer_desc = RTCSessionDescription(sdp=sdp, type=typ)
    args = request.app["args"]
    client_tag = f"{request.remote}"
    ice_servers = [
        RTCIceServer(urls=server["urls"], username=server["username"], credential=server["credential"])
        for server in build_ice_servers(args)
    ]
    pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
    request.app["pcs"].add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"WebRTC connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            request.app["pcs"].discard(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"Server ICE connection state: {pc.iceConnectionState}")

    capture = request.app["capture"]
    track = ZedStereoVideoTrack(
        capture,
        args.fps,
        (args.width, args.height),
        profile=args.profile,
        first_frame_log=args.first_frame_log,
        created_at=t_offer_received,
        client_tag=client_tag,
    )
    request.app["tracks"].add(track)

    sender = pc.addTrack(track)
    force_codec(pc, sender, request.app["args"].codec)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await wait_for_ice_gathering_complete(pc)

    print(
        f"[offer {client_tag}] negotiated in "
        f"{(time.time() - t_offer_received) * 1000:.0f}ms "
        f"(STUN={'on' if args.enable_stun or args.ice_server else 'off'})"
    )

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
    capture = app.get("capture")
    if capture is not None:
        capture.close()
        app["capture"] = None


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    tune_vp8_bitrate(args)

    app = web.Application()
    app["args"] = args
    app["pcs"] = set()
    app["tracks"] = set()

    print(f"[shm] starting capture worker on '{args.shm_name}' "
          f"(waits for ZedStereoDevice to come up)")
    app["capture"] = ShmStereoCaptureWorker(
        args.shm_name, args.fps, (args.width, args.height)
    )

    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=False,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
            )
        },
    )
    for route in list(app.router.routes()):
        cors.add(route)

    ssl_context = None
    scheme = "http"
    if args.cert_file and args.key_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
        scheme = "https"

    print(f"Serving stereo SHM WebRTC on {scheme}://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
