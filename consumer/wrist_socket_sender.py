#!/usr/bin/env python3
"""Wrist Socket Sender - stream stacked dual-wrist frames to a remote receiver.

This consumer reads the two wrist cameras from the device manager's summary
shared memory, applies the same crop + resize the LingBot-VA inference server
uses (crop_before_resize -> resize to per-view HxW), stacks the two views
horizontally (left = wrist_image_1, right = wrist_image_2) and ships each
stacked frame to a remote receiver over a raw TCP socket.

The wire protocol matches ``run_socket_stream_infer.py`` on the receiving side
(video_pred_vis/lingbotva_min):

    magic   : 4 bytes  == b"LVAF"
    width   : uint16    (stacked frame width  == 2 * view_width)
    height  : uint16    (stacked frame height == view_height)
    length  : uint32    (payload byte count   == width * height * 3)
    payload : ``length`` bytes of raw rgb24, shape (height, width, 3)

Because the receiver's minimal engine only supports resize/pad/original (no
crop), the crop is performed here so the receiver gets ready-to-encode frames.

Prerequisites:
- ``device_starter`` must be running and publishing the two WebCameraOpenPi
  feeds into the summary SHM (default name ``device_summary_data``).
- The receiver (run_socket_stream_infer.py) must be listening first.

Example:
    python consumer/wrist_socket_sender.py \\
        --host 127.0.0.1 --port 9999 \\
        --left-cam-id 2 --right-cam-id 0 \\
        --crop 185 185 0 0 --resize 320 240 --fps 10
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consumer.core.base import BaseConsumer
from utils.logger_config import logger

HEADER_MAGIC = b"LVAF"
HEADER_FORMAT = ">4sHHI"


class WristSocketSender(BaseConsumer):
    """Read two wrist cameras from summary SHM and TCP-stream stacked frames."""

    def __init__(
        self,
        *,
        summary_shm_name: str,
        left_device: str,
        right_device: str,
        crop_lrtb: Tuple[int, int, int, int],
        resize_hw: Tuple[int, int],
        host: str,
        port: int,
        fps: float,
    ) -> None:
        super().__init__(summary_shm_name=summary_shm_name)
        self.left_device = left_device
        self.right_device = right_device
        self.crop_lrtb = crop_lrtb
        self.resize_hw = resize_hw
        self.host = host
        self.port = port
        self.fps = fps
        self.obs_devices: List[str] = [left_device, right_device]
        self._sock: Optional[socket.socket] = None

    def _connect_receiver(self) -> None:
        """(Re)connect the TCP client socket to the receiver, retrying forever."""
        self._close_receiver()
        while self.running:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(None)
                self._sock = sock
                logger.info(f"Connected to receiver {self.host}:{self.port}")
                return
            except OSError as exc:
                logger.warning(
                    f"Receiver {self.host}:{self.port} not reachable ({exc}); "
                    "retrying in 1s"
                )
                time.sleep(1.0)

    def _close_receiver(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _preprocess_view(self, frame: np.ndarray) -> np.ndarray:
        """Crop then resize a single RGB (H, W, 3) uint8 view to resize_hw."""
        height, width = frame.shape[:2]
        left, right, top, bottom = self.crop_lrtb
        crop_h = height - top - bottom
        crop_w = width - left - right
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(
                f"crop {self.crop_lrtb} removes the whole {height}x{width} frame"
            )
        cropped = frame[top : top + crop_h, left : left + crop_w]
        out_h, out_w = self.resize_hw
        if (crop_h, crop_w) == (out_h, out_w):
            resized = cropped
        else:
            # cv2.resize takes (width, height).
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if resized.dtype != np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(resized)

    def _read_stacked_frame(self) -> Optional[np.ndarray]:
        """Read the latest frame for both cameras and return the stacked view."""
        data = self.read_all_device_data(device_names=self.obs_devices)
        if not data:
            return None
        if self.left_device not in data or self.right_device not in data:
            return None

        _, left_raw = data[self.left_device]
        _, right_raw = data[self.right_device]
        left_view = self._preprocess_view(np.asarray(left_raw))
        right_view = self._preprocess_view(np.asarray(right_raw))
        if left_view.shape != right_view.shape:
            raise ValueError(
                f"view shape mismatch: left={left_view.shape}, right={right_view.shape}"
            )
        # Horizontal stack: left half = wrist_image_1, right half = wrist_image_2.
        return np.ascontiguousarray(np.concatenate([left_view, right_view], axis=1))

    def _send_frame(self, stacked: np.ndarray) -> bool:
        """Frame + send one stacked RGB image. Returns False if the peer dropped."""
        height, width = stacked.shape[:2]
        payload = stacked.tobytes()
        header = struct.pack(HEADER_FORMAT, HEADER_MAGIC, width, height, len(payload))
        try:
            assert self._sock is not None
            self._sock.sendall(header + payload)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.warning(f"Send failed ({exc}); will reconnect to receiver")
            return False

    def run(self) -> None:
        if not self.connect():
            logger.error("WristSocketSender failed to connect to summary SHM")
            return

        for name in (self.left_device, self.right_device):
            if name not in self.devices:
                logger.error(
                    f"Camera device '{name}' not found in summary SHM. "
                    f"Available: {sorted(self.devices)}"
                )
                return

        self.running = True
        self._connect_receiver()

        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        next_deadline = time.perf_counter()
        sent = 0
        logged_size = False
        logger.info(
            f"Streaming {self.left_device} + {self.right_device} to "
            f"{self.host}:{self.port} at {self.fps} Hz. Press Ctrl+C to stop."
        )

        try:
            while self.running:
                try:
                    stacked = self._read_stacked_frame()
                except ValueError as exc:
                    logger.error(f"Preprocess error: {exc}")
                    break

                if stacked is None:
                    # Cameras not ready yet; wait briefly and retry.
                    time.sleep(0.05)
                    continue

                if not logged_size:
                    logger.info(
                        f"Stacked frame: {stacked.shape[1]}x{stacked.shape[0]} "
                        f"(WxH), dtype={stacked.dtype}"
                    )
                    logged_size = True

                if self._sock is None:
                    self._connect_receiver()

                if not self._send_frame(stacked):
                    self._connect_receiver()
                    continue

                sent += 1
                if sent % max(1, int(self.fps)) == 0:
                    logger.info(f"sent {sent} frames")

                if interval > 0:
                    next_deadline += interval
                    sleep_s = next_deadline - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        # Fell behind; reset the cadence to avoid burst catch-up.
                        next_deadline = time.perf_counter()
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down.")
        finally:
            self._close_receiver()
            self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read two wrist cameras from summary SHM, crop+resize+stack, and "
            "TCP-stream the stacked frames to a LingBot-VA socket receiver."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Receiver host")
    parser.add_argument("--port", type=int, default=9999, help="Receiver port")
    parser.add_argument(
        "--summary-shm", type=str, default="device_summary_data",
        help="Summary shared memory name published by device_starter.",
    )
    parser.add_argument(
        "--device-type", type=str, default="WebCameraOpenPi",
        help="Camera device type prefix used to build the SHM device key.",
    )
    parser.add_argument("--left-cam-id", type=int, default=2, help="Left wrist camera id")
    parser.add_argument("--right-cam-id", type=int, default=0, help="Right wrist camera id")
    parser.add_argument(
        "--left-device", type=str, default=None,
        help="Full left device key (overrides --device-type/--left-cam-id).",
    )
    parser.add_argument(
        "--right-device", type=str, default=None,
        help="Full right device key (overrides --device-type/--right-cam-id).",
    )
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        default=(185, 185, 0, 0),
        help="Pixels to crop before resize (mirrors crop_before_resize).",
    )
    parser.add_argument(
        "--resize", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
        default=(320, 240),
        help="Per-view output size as HEIGHT WIDTH (mirrors crop_resize_size).",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Send frame rate (Hz)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_device = args.left_device or f"{args.device_type}_{args.left_cam_id}"
    right_device = args.right_device or f"{args.device_type}_{args.right_cam_id}"

    sender = WristSocketSender(
        summary_shm_name=args.summary_shm,
        left_device=left_device,
        right_device=right_device,
        crop_lrtb=tuple(args.crop),
        resize_hw=tuple(args.resize),
        host=args.host,
        port=args.port,
        fps=args.fps,
    )
    sender.run()


if __name__ == "__main__":
    main()
