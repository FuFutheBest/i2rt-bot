#!/usr/bin/env python3
"""
iPhone camera device using the Record3D stream.

Author: Letian
"""

import time
import threading
import queue
from typing import Optional, Dict, Any, Mapping

import numpy as np
import cv2

from .base import BaseDevice
from devices.config import StateFeedbackConfig
from devices.contracts import DeviceKind
from utils.logger_config import logger
from utils.time_control import precise_loop_timing

try:
    from record3d import Record3DStream
except ImportError:  # pragma: no cover - graceful degradation when dependency missing
    Record3DStream = None


class IPhoneCameraDevice(BaseDevice):
    """
    Capture RGB frames from an iPhone via the Record3D SDK and publish them to shared memory.
    """

    device_kind = DeviceKind.CAMERA

    DEVICE_TYPE_TRUEDEPTH = 0
    DEVICE_TYPE_LIDAR = 1

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        capture_id: int = 0,
        device_udid: Optional[str] = None,
        flip_truedepth: bool = True,
        rotation_degree: int = 0,
        debug: bool = False,
    ) -> None:
        if Record3DStream is None:
            raise ImportError(
                "record3d package is required for IPhoneCameraDevice. "
                "Install it with `pip install record3d`."
            )

        rotation_degree = self._validate_rotation_degree(rotation_degree)

        super().__init__(
            device_class_id=device_class_id,
            device_name="IPhoneCameraDevice",
            state_feedback=state_feedback,
        )
        np_dtype = np.dtype(self.data_dtype)

        self.capture_id = int(capture_id)
        self.requested_udid = device_udid
        self.flip_truedepth = flip_truedepth
        self.rotation_degree = rotation_degree
        if self.rotation_degree:
            logger.info(
                f"iPhone camera frames will be rotated {self.rotation_degree} degrees clockwise before sharing."
            )

        self.session: Optional[Record3DStream] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_running = False
        self._frame_event = threading.Event()
        self._frame_queue: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue(maxsize=1)

        self._connected_udid: Optional[str] = None
        self._connected_product_id: Optional[str] = None
        self._resize_warned = False
        self._output_dtype = np_dtype
        self.debug = debug
        self._debug_window_name = f"iPhoneCameraDevice_{self.device_class_id}"

    # ------------------------------------------------------------------
    # Record3D session management
    def _on_new_frame(self) -> None:
        self._frame_event.set()

    def _on_stream_stopped(self) -> None:
        logger.info("iPhone stream stopped by device.")
        self._frame_event.set()

    def _select_device(self) -> Any:
        devices = Record3DStream.get_connected_devices()
        if not devices:
            raise RuntimeError("No Record3D devices detected. Please connect an iPhone and reopen the app.")

        if self.requested_udid:
            for dev in devices:
                if getattr(dev, "udid", None) == self.requested_udid:
                    return dev
            raise RuntimeError(f"Record3D device with UDID {self.requested_udid} not found.")

        if self.capture_id >= len(devices):
            raise RuntimeError(
                f"Record3D device index {self.capture_id} out of range (found {len(devices)} devices)."
            )
        return devices[self.capture_id]

    def _connect(self) -> None:
        dev = self._select_device()
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stream_stopped
        self.session.connect(dev)

        self._connected_udid = getattr(dev, "udid", None)
        self._connected_product_id = getattr(dev, "product_id", None)
        logger.info(f"Connected to Record3D device: product_id={self._connected_product_id} udid={self._connected_udid}")

    def _stream_worker(self) -> None:
        while self._stream_running:
            if not self._frame_event.wait(timeout=1.0):
                continue
            self._frame_event.clear()

            if not self.session:
                continue

            try:
                frame = self.session.get_rgb_frame()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Failed to retrieve frame from iPhone: {exc}")
                continue

            if frame is None:
                continue

            # Flip TrueDepth streams for consistency with training data
            if (
                self.flip_truedepth
                and hasattr(self.session, "get_device_type")
                and self.session.get_device_type() == self.DEVICE_TYPE_TRUEDEPTH
            ):
                frame = cv2.flip(frame, 1)

            timestamp_ns = time.time_ns()
            # Keep only the latest frame if producer is faster than consumer
            try:
                while True:
                    self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            if frame is not None:
                self._frame_queue.put((timestamp_ns, frame.copy()))
            else:
                logger.warning("Received None frame from Record3D, skipping...")
                continue


    def _start_stream_thread(self) -> None:
        if self._stream_thread and self._stream_thread.is_alive():
            return
        self._stream_running = True
        self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._stream_thread.start()

    def _stop_stream_thread(self) -> None:
        self._stream_running = False
        self._frame_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

    def _cleanup_session(self) -> None:
        if self.session:
            try:
                self.session.stop_stream()
            except Exception:  # pragma: no cover - ensure cleanup continues
                pass
            self.session = None

    # ------------------------------------------------------------------
    # Frame processing helpers
    @staticmethod
    def _validate_rotation_degree(rotation_degree: int) -> int:
        """Validate supported right-angle image rotations."""
        rotation_degree = int(rotation_degree)
        valid_degrees = {0, 90, 180, 270}
        if rotation_degree not in valid_degrees:
            raise ValueError(
                f"Unsupported rotation_degree={rotation_degree}. "
                f"Supported values: {sorted(valid_degrees)}"
            )
        return rotation_degree

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Rotate image according to configured rotation_degree."""
        if self.rotation_degree == 0:
            return frame

        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        try:
            return cv2.rotate(frame, rotate_map[self.rotation_degree])
        except Exception as exc:
            logger.error(
                f"Failed to rotate iPhone frame by {self.rotation_degree} degrees: {exc}"
            )
            return frame

    def _ensure_color(self, frame: np.ndarray) -> np.ndarray:
        frame = self._apply_rotation(frame)

        if len(self.data_shape) == 3:
            target_height, target_width, _ = self.data_shape
        else:
            target_height, target_width = self.data_shape

        # Resize if needed
        if frame.shape[0] != target_height or frame.shape[1] != target_width:
            if not self._resize_warned:
                logger.warning(
                    "Resizing iPhone stream from %s to (%d, %d). Consider matching the Record3D resolution.",
                    frame.shape[:2],
                    target_height,
                    target_width,
                )
                self._resize_warned = True
            frame = cv2.resize(frame, (target_width, target_height))

        if len(self.data_shape) == 3:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
        else:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        return frame.astype(self._output_dtype, copy=False)

    # ------------------------------------------------------------------
    # BaseDevice overrides
    def _display_debug_frame(self, frame: np.ndarray) -> None:
        if not self.debug:
            return
        if frame.ndim == 3 and frame.shape[2] == 3:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            display = frame
        cv2.imshow(self._debug_window_name, display)
        cv2.waitKey(1)

    def _close_debug_window(self) -> None:
        if getattr(self, "debug", False):
            try:
                cv2.destroyWindow(self._debug_window_name)
            except cv2.error:
                pass

    def start_server(self) -> None:
        if self.running:
            logger.info(f"IPhone camera {self.device_class_id} is already running")
            return

        logger.info(f"Starting iPhone camera device (id={self.device_class_id})...")
        self.running = True

        try:
            self._connect()
        except Exception as exc:
            self.running = False
            raise RuntimeError(f"Failed to connect to iPhone camera: {exc}") from exc

        self._start_stream_thread()
        self._frame_queue = queue.Queue(maxsize=1)
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory for iPhone camera")

        logger.info(f"iPhone camera server started. Shared memory: {self.shared_memory_name}")
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        try:
            while self.running:
                try:
                    timestamp_ns, frame = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    wait_for_next_iteration()
                    continue

                processed = self._ensure_color(frame)
                self._write_array_to_shm_with_timestamp(processed, timestamp_ns)
                self._display_debug_frame(processed)
                wait_for_next_iteration()
        finally:
            self.stop_server()

    def stop_server(self) -> None:
        if not self.running:
            return

        logger.info(f"Stopping iPhone camera device (id={self.device_class_id})...")
        self.running = False
        self._stop_stream_thread()
        self._cleanup_shared_memory()
        self._cleanup_session()
        self._close_debug_window()
        logger.info("iPhone camera device stopped")

    def close(self) -> None:
        self.stop_server()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        finally:
            self._close_debug_window()

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update(
            {
                "udid": self._connected_udid,
                "product_id": self._connected_product_id,
                "flip_truedepth": self.flip_truedepth,
                "rotation_degree": self.rotation_degree,
            }
        )
        return info


def main() -> None:
    """CLI entry point for manual testing."""
    import argparse
    from utils.shm_utils import get_dtype

    parser = argparse.ArgumentParser(description="iPhone Camera Device Server")
    parser.add_argument("--device-class-id", "-d", type=int, default=0, help="Internal device id")
    parser.add_argument("--fps", "-f", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--dtype", default="uint8", help="Output dtype")
    parser.add_argument(
        "--shape",
        "-s",
        type=str,
        default="480,640,3",
        help="Output frame shape, e.g. '480,640,3'",
    )
    parser.add_argument("--buffer-size", "-b", type=int, default=1, help="Shared memory buffer size")
    parser.add_argument("--device-class-id", type=int, default=0, help="Logical shared-memory ID")
    parser.add_argument("--capture-id", type=int, default=0, help="Index into detected Record3D devices")
    parser.add_argument("--device-udid", type=str, default=None, help="Explicit Record3D device UDID")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal flip for TrueDepth cameras")
    parser.add_argument(
        "--rotation-degree",
        type=int,
        default=0,
        choices=(0, 90, 180, 270),
        help="Rotate output frames clockwise before resizing/sharing",
    )
    parser.add_argument("--debug", action="store_true", help="Display incoming frames in a debug window")

    args = parser.parse_args()

    try:
        data_shape = tuple(int(x.strip()) for x in args.shape.split(","))
    except ValueError as exc:
        raise SystemExit(f"Invalid shape string '{args.shape}': {exc}")

    dtype = get_dtype(args.dtype)

    device = IPhoneCameraDevice(
        device_class_id=args.device_class_id,
        state_feedback={
            "fps": args.fps,
            "buffer_size": args.buffer_size,
            "hardware_latency_ms": 0.0,
            "format": {"shape": list(data_shape), "dtype": args.dtype, "type": "rgb"},
        },
        capture_id=args.capture_id,
        device_udid=args.device_udid,
        flip_truedepth=not args.no_flip,
        rotation_degree=args.rotation_degree,
        debug=args.debug,
    )

    logger.info(f"Launching iPhone camera device with configuration: {device.get_device_info()}")

    try:
        device.start_server()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        device.close()


if __name__ == "__main__":
    main()
