"""Stereo SHM -> WebRTC server for Quest 3, GStreamer ``webrtcbin`` backend.

This is a drop-in alternative to ``scripts/zed_webrtc_server.py`` (the aiortc
backend). It exists because the aiortc send path (RTP packetize + SRTP + sendto)
runs entirely in ONE asyncio loop in pure Python: the GIL serializes it onto a
single core, so with a multi-threaded VP8 encoder the send loop is starved and
fps collapses to ~6-13 even on a fast wired LAN.

``webrtcbin`` moves the whole encode -> packetize -> SRTP -> DTLS/ICE -> send
chain into native, multi-threaded C, so the send side is no longer GIL-bound.
The codec stays VP8 and the HTTP ``/offer`` signaling contract is identical, so
the Quest / Vuer ``WebRTCStereoVideoPlane`` client needs NO changes.

Data path (unchanged):

    ZedStereoDevice (grab -> SHM)  ->  this server (SHM -> appsrc -> webrtcbin)  ->  Quest

Pipeline:

    appsrc(RGB) -> videoconvert -> vp8enc(deadline=1, threads=N) -> rtpvp8pay
                -> application/x-rtp,encoding-name=VP8 -> webrtcbin

The browser is the offerer (POSTs SDP to ``/offer``); webrtcbin answers. We
gather ICE non-trickle (wait for COMPLETE) and return the full answer SDP, which
matches the existing aiortc server's behavior.

Author: SuperInference R1 teleoperation team.
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import ssl
import sys
import threading
import time
from typing import Optional, Tuple


def _ensure_system_gi_on_path() -> None:
    """PyGObject (``gi``) ships as a SYSTEM dist-package, not a venv wheel. The
    venv interpreter (CPython 3.10) is ABI-compatible with the system
    ``_gi.cpython-310*.so``, so we just append the system ``dist-packages`` dir
    to ``sys.path`` (APPEND, so venv packages keep priority and only the missing
    ``gi`` falls through to the system).
    """
    try:
        import gi  # noqa: F401  -- already importable (e.g. --system-site-packages)
        return
    except ModuleNotFoundError:
        pass
    candidates = [
        "/usr/lib/python3/dist-packages",
        f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages",
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "gi")) and path not in sys.path:
            sys.path.append(path)


_ensure_system_gi_on_path()

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import GLib, Gst, GstSdp, GstWebRTC  # noqa: E402

import aiohttp_cors  # noqa: E402
from aiohttp import web  # noqa: E402

# scripts/ -> repo root so ``from utils...`` resolves.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from incubator.r1.shm_capture_worker import (  # noqa: E402
    LatestFrameShmCaptureWorker as ShmStereoCaptureWorker,
)

logger = logging.getLogger("zed_webrtc_gst")


# Minimal desktop test page (browser = offerer, recvonly video). The real Quest
# client talks to /offer directly via Vuer, so this is only for local testing.
INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"/><title>GST WebRTC</title>
<style>body{margin:0;background:#111;color:#fff;font-family:sans-serif}
video{width:100vw;height:100vh;object-fit:contain;background:#000}
#bar{position:fixed;top:0;left:0;right:0;padding:10px;background:rgba(0,0,0,.6);z-index:2}</style>
</head><body>
<div id="bar"><button onclick="start()">Start</button> <span id="s">idle</span></div>
<video id="v" autoplay playsinline muted></video>
<script>
let pc=null;
async function start(){
  document.getElementById('s').textContent='connecting';
  pc=new RTCPeerConnection({iceServers:[]});
  pc.addTransceiver('video',{direction:'recvonly'});
  pc.ontrack=(e)=>{document.getElementById('v').srcObject=e.streams[0];};
  pc.onconnectionstatechange=()=>{document.getElementById('s').textContent=pc.connectionState;};
  const offer=await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise(r=>{if(pc.iceGatheringState==='complete')r();
    else pc.addEventListener('icegatheringstatechange',()=>{if(pc.iceGatheringState==='complete')r();});});
  const resp=await fetch('/offer',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sdp:pc.localDescription.sdp,type:pc.localDescription.type})});
  const ans=await resp.json();
  await pc.setRemoteDescription(ans);
}
</script></body></html>
"""


class GstWebRTCSession:
    """One WebRTC peer == one GStreamer pipeline (own encoder + webrtcbin).

    A background feeder thread pulls the latest stereo RGB frame from the shared
    ``ShmStereoCaptureWorker`` and pushes it into this session's ``appsrc`` at the
    target fps (resending the last frame if no newer one is ready, so the encoder
    is always fed and the send rate is bounded only by the encoder/transport).
    """

    def __init__(
        self,
        capture: ShmStereoCaptureWorker,
        fps: int,
        out_size: Tuple[int, int],
        bitrate_bps: int,
        threads: int,
        keyframe_dist: int,
        cpu_used: int,
        client_tag: str,
        profile: bool = False,
    ) -> None:
        self.capture = capture
        self.fps = max(1, int(fps))
        self.out_w, self.out_h = out_size
        self.bitrate_bps = int(bitrate_bps)
        self.threads = max(1, int(threads))
        self.keyframe_dist = max(1, int(keyframe_dist))
        self.cpu_used = int(cpu_used)
        self.client_tag = client_tag
        self.profile = profile

        self._stop = threading.Event()
        self._feeder: Optional[threading.Thread] = None
        self._created_at = time.time()
        self._first_buffer_pushed = False

        self.pipeline = self._build_pipeline()
        self.appsrc = self.pipeline.get_by_name("src")
        self.webrtc = self.pipeline.get_by_name("sendrecv")

    def _build_pipeline(self) -> Gst.Pipeline:
        # error-resilient + cbr + deadline=1 (realtime) keep latency low; threads
        # parallelize the native encode without touching the Python GIL.
        desc = (
            f"appsrc name=src is-live=true do-timestamp=true format=time "
            f"caps=video/x-raw,format=RGB,width={self.out_w},height={self.out_h},"
            f"framerate={self.fps}/1 "
            f"! queue max-size-buffers=2 leaky=downstream "
            f"! videoconvert n-threads={self.threads} "
            f"! vp8enc name=enc deadline=1 cpu-used={self.cpu_used} threads={self.threads} "
            f"target-bitrate={self.bitrate_bps} keyframe-max-dist={self.keyframe_dist} "
            f"error-resilient=default end-usage=cbr "
            f"! rtpvp8pay name=pay pt=96 picture-id-mode=2 "
            f"! application/x-rtp,media=video,encoding-name=VP8,payload=96 "
            f"! webrtcbin name=sendrecv bundle-policy=max-bundle latency=0"
        )
        logger.info("[gst %s] pipeline: %s", self.client_tag, desc)
        return Gst.parse_launch(desc)

    def _on_negotiation_needed(self, _element) -> None:
        # We answer the browser's offer, so we do NOT initiate negotiation here.
        pass

    def start_media(self) -> None:
        self.pipeline.set_state(Gst.State.PLAYING)
        self._feeder = threading.Thread(
            target=self._feed_loop, name=f"gst-feed-{self.client_tag}", daemon=True
        )
        self._feeder.start()

    def _feed_loop(self) -> None:
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        last_index = 0
        last_frame = None
        n_pushed = 0
        last_report = time.time()
        while not self._stop.is_set():
            frame, last_index = self.capture.get_latest(last_index, timeout=0.0)
            if frame is None:
                frame = last_frame if last_frame is not None else self.capture._latest_frame
            else:
                last_frame = frame
            if frame is not None:
                self._push_frame(frame)
                n_pushed += 1
            now = time.time()
            if now - last_report >= 1.0:
                logger.info(
                    "[gst %s] push fps: %.1f | shm capture fps: %.1f",
                    self.client_tag, n_pushed / (now - last_report), self.capture.capture_fps,
                )
                n_pushed = 0
                last_report = now
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def _push_frame(self, frame) -> None:
        try:
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            self.appsrc.emit("push-buffer", buf)
            if self.profile and not self._first_buffer_pushed:
                self._first_buffer_pushed = True
                logger.info(
                    "[gst %s] first frame pushed %.0fms after session start",
                    self.client_tag,
                    (time.time() - self._created_at) * 1000,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[gst %s] push-buffer failed: %r", self.client_tag, exc)

    def handle_offer(self, offer_sdp: str, done: "concurrent.futures.Future") -> None:
        """Drive set-remote -> create-answer -> set-local -> gather ICE -> resolve.

        Runs in the GLib main-loop thread (scheduled via GLib.idle_add).
        """
        try:
            self.start_media()

            res, sdpmsg = GstSdp.SDPMessage.new()
            GstSdp.sdp_message_parse_buffer(offer_sdp.encode("utf-8"), sdpmsg)
            offer = GstWebRTC.WebRTCSessionDescription.new(
                GstWebRTC.WebRTCSDPType.OFFER, sdpmsg
            )

            def on_answer_created(promise, _data):
                reply = promise.get_reply()
                answer = reply.get_value("answer")
                self.webrtc.emit("set-local-description", answer, Gst.Promise.new())
                self._await_ice_then_resolve(done)

            def on_remote_set(promise, _data):
                promise.wait()
                ans_promise = Gst.Promise.new_with_change_func(on_answer_created, None)
                self.webrtc.emit("create-answer", None, ans_promise)

            remote_promise = Gst.Promise.new_with_change_func(on_remote_set, None)
            self.webrtc.emit("set-remote-description", offer, remote_promise)
        except Exception as exc:  # noqa: BLE001
            if not done.done():
                done.set_exception(exc)

    def _await_ice_then_resolve(self, done: "concurrent.futures.Future") -> None:
        def resolve_if_complete() -> None:
            state = self.webrtc.get_property("ice-gathering-state")
            if state == GstWebRTC.WebRTCICEGatheringState.COMPLETE:
                local = self.webrtc.get_property("local-description")
                if local is not None and not done.done():
                    done.set_result(local.sdp.as_text())
                    return True
            return False

        if resolve_if_complete():
            return

        def on_notify(_obj, _pspec):
            resolve_if_complete()

        self.webrtc.connect("notify::ice-gathering-state", on_notify)
        # Safety timeout: return whatever we have after 3s (host candidates on a
        # LAN gather almost instantly).
        def fallback() -> bool:
            if not done.done():
                local = self.webrtc.get_property("local-description")
                if local is not None:
                    done.set_result(local.sdp.as_text())
                else:
                    done.set_exception(RuntimeError("ICE gathering timed out"))
            return False

        GLib.timeout_add(3000, fallback)

    def close(self) -> None:
        self._stop.set()
        if self._feeder is not None:
            self._feeder.join(timeout=2)
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)


# ----------------------------------------------------------------------------
# aiohttp signaling (browser = offerer). Bridges to the GLib loop thread.
# ----------------------------------------------------------------------------
async def index(_request: web.Request) -> web.Response:
    return web.Response(content_type="text/html", text=INDEX_HTML)


async def offer(request: web.Request) -> web.Response:
    t0 = time.time()
    try:
        params = await request.json()
        offer_sdp = params.get("sdp")
        if not isinstance(offer_sdp, str) or not offer_sdp:
            raise ValueError("missing 'sdp'")
    except Exception:
        return web.json_response({"error": "Invalid offer payload"}, status=400)

    app = request.app
    args = app["args"]
    client_tag = request.remote or "client"

    session = GstWebRTCSession(
        capture=app["capture"],
        fps=args.fps,
        out_size=app["out_size"],
        bitrate_bps=int(args.video_bitrate_mbps * 1_000_000),
        threads=args.vpx_threads,
        keyframe_dist=args.keyframe_interval,
        cpu_used=args.cpu_used,
        client_tag=client_tag,
        profile=args.profile,
    )
    app["sessions"].add(session)

    loop = asyncio.get_event_loop()
    fut: concurrent.futures.Future = concurrent.futures.Future()
    GLib.idle_add(session.handle_offer, offer_sdp, fut)
    answer_sdp = await asyncio.wrap_future(fut, loop=loop)

    logger.info(
        "[offer %s] answered in %.0fms (codec=VP8, backend=webrtcbin)",
        client_tag,
        (time.time() - t0) * 1000,
    )
    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": answer_sdp, "type": "answer"}),
    )


async def on_shutdown(app: web.Application) -> None:
    for session in list(app["sessions"]):
        session.close()
    app["sessions"].clear()
    capture = app.get("capture")
    if capture is not None:
        capture.close()
        app["capture"] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GStreamer webrtcbin stereo WebRTC server (SHM source)."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--shm-name", default="ZedStereoDevice_0_data")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--video-bitrate-mbps", type=float, default=3.0)
    parser.add_argument("--vpx-threads", type=int, default=4,
                        help="vp8enc/videoconvert threads (native C, not GIL-bound).")
    parser.add_argument("--cpu-used", type=int, default=5,
                        help="vp8enc speed/quality knob (0..16, higher=faster).")
    parser.add_argument("--keyframe-interval", type=int, default=60)
    parser.add_argument("--cert-file")
    parser.add_argument("--key-file")
    parser.add_argument("--profile", action="store_true")
    # Accepted for CLI compatibility with the aiortc server (ignored here).
    parser.add_argument("--codec", default="auto")
    parser.add_argument("--ice-server", action="append")
    parser.add_argument("--enable-stun", action="store_true")
    parser.add_argument(
        "--first-frame-log",
        action="store_true",
        default=False,
        help="Print latency from /offer received -> first frame emitted.",
    )
    return parser.parse_args()


def _start_glib_loop() -> GLib.MainLoop:
    Gst.init(None)
    loop = GLib.MainLoop()
    threading.Thread(target=loop.run, name="glib-main", daemon=True).start()
    return loop


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    _start_glib_loop()

    scale = max(0.05, float(args.scale))
    out_w = int(round(args.width * scale))
    out_h = int(round(args.height * scale))
    out_w -= out_w % 2
    out_h -= out_h % 2
    if (out_w, out_h) != (args.width, args.height):
        logger.info(
            "[encode] downscaling %dx%d -> %dx%d (scale=%.3f) before VP8 encode",
            args.width, args.height, out_w, out_h, scale,
        )

    logger.info(
        "VP8(webrtcbin): bitrate=%.1f Mbps, fps=%d, threads=%d, cpu-used=%d, "
        "keyframe every %d frames",
        args.video_bitrate_mbps, args.fps, args.vpx_threads, args.cpu_used,
        args.keyframe_interval,
    )

    app = web.Application()
    app["args"] = args
    app["sessions"] = set()
    app["out_size"] = (out_w, out_h)
    app["capture"] = ShmStereoCaptureWorker(args.shm_name, args.fps, (out_w, out_h))

    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=False, expose_headers="*",
                allow_headers="*", allow_methods="*",
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

    logger.info("Serving stereo SHM WebRTC (webrtcbin) on %s://%s:%d",
                scheme, args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
