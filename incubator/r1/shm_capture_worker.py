"""Background worker for polling the latest frame from a device SHM ring."""

import threading
import time
from typing import Optional, Tuple

import numpy as np

from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_shared_memory,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
)


class LatestFrameShmCaptureWorker:
    """Poll a device SHM ring and expose only its most recent frame."""

    def __init__(
        self,
        shm_name: str,
        fps: int,
        output_size: Tuple[int, int],
        *,
        close_timeout_s: float = 2.0,
        thread_name: str = "shm-capture",
    ) -> None:
        self.shm_name = shm_name
        self.fps = max(1, int(fps))
        self.output_size = output_size
        self.close_timeout_s = float(close_timeout_s)
        if self.close_timeout_s < 0.0:
            raise ValueError("close_timeout_s must be >= 0")

        self._shm = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_index = 0
        self._last_shm_write_index = -1
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()

        self.capture_fps = 0.0
        self._fps_window_count = 0
        self._fps_window_start = time.time()

        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    def _connect(self) -> bool:
        if self._shm is not None:
            return True
        try:
            self._shm = connect_to_shared_memory(
                self.shm_name,
                read_only=True,
                log_missing=False,
            )
            print(f"[shm] connected to {self.shm_name}")
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001
            print(f"[shm] connect failed: {exc!r}")
            return False

    def _read_latest_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame, or ``None`` if no new frame is available."""
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
        frame_offset = (
            DEVICE_HEADER_SIZE
            + BUFFER_HEADER_SIZE
            + latest_index * frame_size
        )
        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        dtype = get_dtype(device_header["data_dtype"], np.uint8)
        frame = np.frombuffer(
            buf[data_start : data_start + data_size],
            dtype=dtype,
        )
        shape = device_header["shape"]
        if shape:
            frame = frame.reshape(shape)
        return np.ascontiguousarray(frame)

    def _run(self) -> None:
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                frame = self._read_latest_frame()
            except Exception as exc:  # noqa: BLE001
                print(f"[shm] read failed: {exc!r}")
                frame = None

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame.copy()
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
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def get_latest(
        self,
        last_index: int,
        timeout: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Return a frame newer than ``last_index``, if one becomes available."""
        if not self._frame_ready.wait(timeout=timeout):
            return None, last_index
        with self._lock:
            if self._frame_index <= last_index:
                self._frame_ready.clear()
                return None, last_index
            frame = self._latest_frame
            index = self._frame_index
        return frame, index

    def close(self) -> None:
        """Stop polling and close the read-only SHM handle."""
        self._stop.set()
        self._thread.join(timeout=self.close_timeout_s)
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:  # noqa: BLE001
                pass
            self._shm = None
