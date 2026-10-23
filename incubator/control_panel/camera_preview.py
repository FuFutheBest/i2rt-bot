#!/usr/bin/env python3
"""Read camera preview frames from a device shared-memory ring buffer.

The preview path is deliberately a consumer only: it connects to an existing
device SHM through :mod:`utils.shm_utils`, copies a complete frame out, and
never creates or unlinks shared memory.  This also lets the web control plane
show camera output before the optional device-summary manager is running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .shm_protocol import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    DTYPE_MAP,
    FRAME_HEADER_SIZE,
    connect_to_device_shm,
    unpack_buffer_header,
    unpack_device_header,
    unpack_frame_header,
)


@dataclass(frozen=True)
class CameraPreviewResult:
    """A JPEG preview and JSON-friendly metadata about the source frame."""

    jpeg_bytes: Optional[bytes]
    metadata: Dict[str, Any]

    @property
    def ok(self) -> bool:
        """Whether a fresh JPEG was produced."""

        return self.jpeg_bytes is not None and self.metadata.get("status") == "ok"


@dataclass(frozen=True)
class _RawFrame:
    payload: bytes
    timestamp_ns: int
    shape: Tuple[int, ...]
    dtype: np.dtype
    device_type: str
    fps: float
    hardware_latency_ms: float
    buffer_size: int
    current_frames_count: int
    write_index: int
    read_attempts: int


class _PreviewReadError(RuntimeError):
    def __init__(
        self,
        status: str,
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        reconnect: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.metadata = metadata or {}
        self.reconnect = reconnect


class CameraPreviewReader:
    """Read and JPEG-encode the newest frame from one camera device SHM.

    Args:
        device_class: Device class/SHM prefix, for example ``WebCameraOpenPi``.
        device_class_id: Logical device ID from the selected config.
        stale_after_s: Maximum frame age before status becomes ``stale``.
        jpeg_quality: OpenCV JPEG quality in the inclusive range 1..100.
        max_read_retries: Bounded retries when a producer advances while a
            frame is being copied.
        connector: Injection point for tests. It must follow
            ``connect_to_device_shm`` and honor ``read_only=True``.
        clock_ns: Injection point for deterministic freshness tests.

    ``read`` performs a lightweight seqlock-style check around the payload
    copy: both the ring-buffer header and the selected frame timestamp must be
    identical before and after copying. A stale cached handle is closed and
    reconnected once, which handles a device process unlinking/recreating its
    SHM after restart.
    """

    def __init__(
        self,
        device_class: str,
        device_class_id: int,
        *,
        stale_after_s: float = 2.0,
        jpeg_quality: int = 80,
        max_read_retries: int = 3,
        connector: Callable[..., Any] = connect_to_device_shm,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not isinstance(device_class, str) or not device_class.strip():
            raise ValueError("device_class must be a non-empty string")
        if stale_after_s < 0:
            raise ValueError("stale_after_s must be non-negative")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if max_read_retries < 1:
            raise ValueError("max_read_retries must be at least 1")

        self.device_class = device_class.strip()
        self.device_class_id = int(device_class_id)
        self.stale_after_s = float(stale_after_s)
        self.jpeg_quality = int(jpeg_quality)
        self.max_read_retries = int(max_read_retries)
        self._connector = connector
        self._clock_ns = clock_ns
        self._channel: Optional[Any] = None

    @property
    def shm_name(self) -> str:
        return f"{self.device_class}_{self.device_class_id}_data"

    @property
    def connected(self) -> bool:
        return self._channel is not None

    def __enter__(self) -> "CameraPreviewReader":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close only this reader's handle; never unlink producer-owned SHM."""

        channel = self._channel
        self._channel = None
        if channel is None:
            return
        try:
            channel.close()
        except (BufferError, OSError, ValueError):
            # A broken/stale handle is already unusable. In particular, do not
            # escalate to unlinking it: lifecycle ownership stays with device.
            pass

    def read(self) -> CameraPreviewResult:
        """Return the latest preview JPEG plus timestamp/age/shape/status.

        Status values currently emitted are ``ok``, ``stale``, ``not_found``,
        ``connection_error``, ``empty``, ``invalid_header``, ``invalid_shape``,
        ``unsupported_dtype``, ``inconsistent``, ``read_error``, and
        ``encode_error``. Error results have ``jpeg_bytes=None``. A stale result
        still carries the last successfully copied JPEG so the UI can display
        it with a stale badge.
        """

        connected_at_start = self.connected
        reconnected = False

        # At most one handle-level reconnect per call. Consistency retries are
        # separately bounded inside _read_consistent_frame.
        for handle_attempt in range(2):
            try:
                channel = self._ensure_connected()
            except FileNotFoundError:
                return self._error_result(
                    "not_found", f"Shared memory {self.shm_name!r} does not exist"
                )
            except Exception as exc:
                self.close()
                return self._error_result("connection_error", str(exc))

            try:
                raw = self._read_consistent_frame(channel)
            except _PreviewReadError as exc:
                if exc.reconnect:
                    self.close()
                    if handle_attempt == 0:
                        reconnected = True
                        continue
                return self._error_result(
                    exc.status,
                    str(exc),
                    metadata={**exc.metadata, "reconnected": reconnected},
                )
            except (BufferError, OSError, TypeError, ValueError) as exc:
                self.close()
                if handle_attempt == 0:
                    reconnected = True
                    continue
                return self._error_result(
                    "read_error", str(exc), metadata={"reconnected": reconnected}
                )

            now_ns = int(self._clock_ns())
            age_ns = max(0, now_ns - raw.timestamp_ns)
            age_ms = age_ns / 1_000_000.0
            is_fresh = age_ns <= int(self.stale_after_s * 1_000_000_000)

            # If an already-open handle became stale, its producer may have
            # unlinked/recreated the same SHM name. Reopen once before reporting.
            if not is_fresh and connected_at_start and handle_attempt == 0:
                self.close()
                reconnected = True
                continue

            metadata = self._metadata_for_raw(
                raw,
                status="ok" if is_fresh else "stale",
                age_ms=age_ms,
                fresh=is_fresh,
                reconnected=reconnected,
            )
            try:
                jpeg_bytes = self._encode_jpeg(raw)
            except (cv2.error, ValueError) as exc:
                return self._error_result(
                    "encode_error", str(exc), metadata=metadata
                )

            if not is_fresh:
                # Force name lookup on the next request; an attached Linux SHM
                # handle can otherwise outlive an unlinked producer segment.
                self.close()
            return CameraPreviewResult(jpeg_bytes=jpeg_bytes, metadata=metadata)

        return self._error_result("read_error", "Unable to read camera preview")

    def _ensure_connected(self) -> Any:
        if self._channel is None:
            channel = self._connector(
                self.device_class, self.device_class_id, read_only=True
            )
            if getattr(channel, "is_read_only", True) is False:
                try:
                    channel.close()
                finally:
                    raise PermissionError(
                        f"Connector returned writable handle for {self.shm_name}"
                    )
            self._channel = channel
        return self._channel

    def _read_consistent_frame(self, channel: Any) -> _RawFrame:
        try:
            buf = channel.buf
            segment_size = len(buf)
        except (BufferError, OSError, TypeError, ValueError) as exc:
            raise _PreviewReadError(
                "read_error", str(exc), reconnect=True
            ) from exc

        for attempt in range(1, self.max_read_retries + 1):
            try:
                device_header_bytes = bytes(buf[:DEVICE_HEADER_SIZE])
                device_header = unpack_device_header(device_header_bytes)
                shape, dtype, frame_size = self._validate_device_header(
                    device_header, segment_size
                )

                header_start = DEVICE_HEADER_SIZE
                header_end = header_start + BUFFER_HEADER_SIZE
                buffer_before = unpack_buffer_header(bytes(buf[header_start:header_end]))
                self._validate_buffer_header(buffer_before, frame_size, segment_size)

                if buffer_before["current_frames_count"] == 0:
                    raise _PreviewReadError(
                        "empty",
                        "Camera SHM is connected but does not contain a frame yet",
                        metadata={
                            "shape": shape,
                            "dtype": dtype.name,
                            "read_attempts": attempt,
                        },
                    )

                frame_index = (
                    buffer_before["write_index"] - 1
                ) % buffer_before["buffer_size"]
                frame_offset = (
                    DEVICE_HEADER_SIZE
                    + BUFFER_HEADER_SIZE
                    + frame_index * frame_size
                )
                timestamp_start = frame_offset
                timestamp_end = timestamp_start + FRAME_HEADER_SIZE
                data_start = timestamp_end
                data_end = frame_offset + frame_size

                timestamp_before = unpack_frame_header(
                    bytes(buf[timestamp_start:timestamp_end])
                )["timestamp_ns"]
                # bytes(...) is the intentional, immediate copy that severs all
                # ndarray/JPEG work from the producer-owned memoryview.
                payload = bytes(buf[data_start:data_end])
                timestamp_after = unpack_frame_header(
                    bytes(buf[timestamp_start:timestamp_end])
                )["timestamp_ns"]
                buffer_after = unpack_buffer_header(
                    bytes(buf[header_start:header_end])
                )
            except _PreviewReadError:
                raise
            except (BufferError, OSError, TypeError, ValueError) as exc:
                raise _PreviewReadError(
                    "invalid_header", str(exc), reconnect=True
                ) from exc

            stable_header = (
                buffer_before["buffer_size"] == buffer_after["buffer_size"]
                and buffer_before["current_frames_count"]
                == buffer_after["current_frames_count"]
                and buffer_before["write_index"] == buffer_after["write_index"]
            )
            if (
                stable_header
                and timestamp_before == timestamp_after
                and timestamp_before > 0
            ):
                return _RawFrame(
                    payload=payload,
                    timestamp_ns=timestamp_before,
                    shape=shape,
                    dtype=dtype,
                    device_type=device_header["device_type"],
                    fps=float(device_header["fps"]),
                    hardware_latency_ms=float(
                        device_header["hardware_latency_ms"]
                    ),
                    buffer_size=buffer_before["buffer_size"],
                    current_frames_count=buffer_before["current_frames_count"],
                    write_index=buffer_before["write_index"],
                    read_attempts=attempt,
                )

        raise _PreviewReadError(
            "inconsistent",
            "Camera producer changed write_index or timestamp during frame copy",
            metadata={"read_attempts": self.max_read_retries},
            reconnect=True,
        )

    def _validate_device_header(
        self, device_header: Dict[str, Any], segment_size: int
    ) -> Tuple[Tuple[int, ...], np.dtype, int]:
        if device_header["device_type"] != self.device_class[:20]:
            raise _PreviewReadError(
                "invalid_header",
                "Device type in SHM header does not match requested device",
                reconnect=True,
            )
        if int(device_header["device_class_id"]) != self.device_class_id:
            raise _PreviewReadError(
                "invalid_header",
                "Device ID in SHM header does not match requested device",
                reconnect=True,
            )

        dtype_name = str(device_header["data_dtype"])
        dtype_type = DTYPE_MAP.get(dtype_name)
        if dtype_type is None or np.dtype(dtype_type) != np.dtype(np.uint8):
            raise _PreviewReadError(
                "unsupported_dtype",
                f"Camera preview requires uint8 SHM data, got {dtype_name!r}",
                metadata={"dtype": dtype_name},
            )
        dtype = np.dtype(dtype_type)

        shape = tuple(int(dimension) for dimension in device_header["shape"])
        valid_shape = (
            len(shape) == 2
            or (len(shape) == 3 and shape[2] in (1, 3, 4))
        ) and all(dimension > 0 for dimension in shape)
        if not valid_shape:
            raise _PreviewReadError(
                "invalid_shape",
                f"Expected HxW, HxWx1, HxWx3, or HxWx4 camera data, got {shape}",
                metadata={"shape": shape, "dtype": dtype.name},
            )

        expected_data_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        frame_size = int(device_header["frame_size"])
        if frame_size != FRAME_HEADER_SIZE + expected_data_size:
            raise _PreviewReadError(
                "invalid_header",
                "Frame size does not match camera shape and dtype",
                metadata={"shape": shape, "dtype": dtype.name},
                reconnect=True,
            )
        if segment_size < DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_size:
            raise _PreviewReadError(
                "invalid_header",
                "Shared-memory segment is smaller than one declared frame",
                metadata={"shape": shape, "dtype": dtype.name},
                reconnect=True,
            )
        return shape, dtype, frame_size

    @staticmethod
    def _validate_buffer_header(
        buffer_header: Dict[str, int], frame_size: int, segment_size: int
    ) -> None:
        buffer_size = int(buffer_header["buffer_size"])
        frame_count = int(buffer_header["current_frames_count"])
        write_index = int(buffer_header["write_index"])
        if buffer_size <= 0:
            raise _PreviewReadError(
                "invalid_header", "Camera SHM buffer_size must be positive", reconnect=True
            )
        if not 0 <= frame_count <= buffer_size:
            raise _PreviewReadError(
                "invalid_header", "Camera SHM frame count is out of range", reconnect=True
            )
        if not 0 <= write_index < buffer_size:
            raise _PreviewReadError(
                "invalid_header", "Camera SHM write index is out of range", reconnect=True
            )
        required_size = (
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + buffer_size * frame_size
        )
        if required_size > segment_size:
            raise _PreviewReadError(
                "invalid_header",
                "Camera SHM ring layout exceeds the attached segment",
                reconnect=True,
            )

    def _encode_jpeg(self, raw: _RawFrame) -> bytes:
        image = np.frombuffer(raw.payload, dtype=raw.dtype).reshape(raw.shape)
        if image.ndim == 3 and image.shape[2] == 3:
            # Camera devices publish RGB; OpenCV's JPEG encoder consumes BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif image.ndim == 3:
            image = image[:, :, 0]

        encoded, jpeg = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not encoded:
            raise ValueError("OpenCV failed to encode the camera frame as JPEG")
        return jpeg.tobytes()

    def _metadata_for_raw(
        self,
        raw: _RawFrame,
        *,
        status: str,
        age_ms: float,
        fresh: bool,
        reconnected: bool,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "source": "device_shm",
            "shm_name": self.shm_name,
            "device_type": raw.device_type,
            "device_class_id": self.device_class_id,
            "timestamp_ns": raw.timestamp_ns,
            "age_ms": age_ms,
            "shape": raw.shape,
            "dtype": raw.dtype.name,
            "fresh": fresh,
            "fps": raw.fps,
            "hardware_latency_ms": raw.hardware_latency_ms,
            "buffer_size": raw.buffer_size,
            "current_frames_count": raw.current_frames_count,
            "write_index": raw.write_index,
            "read_attempts": raw.read_attempts,
            "reconnected": reconnected,
        }

    def _error_result(
        self,
        status: str,
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CameraPreviewResult:
        details: Dict[str, Any] = {
            "status": status,
            "source": "device_shm",
            "shm_name": self.shm_name,
            "device_type": self.device_class,
            "device_class_id": self.device_class_id,
            "timestamp_ns": None,
            "age_ms": None,
            "shape": None,
            "dtype": None,
            "fresh": False,
            "message": message,
        }
        if metadata:
            details.update(metadata)
            details["status"] = status
            details["message"] = message
        return CameraPreviewResult(jpeg_bytes=None, metadata=details)


def read_camera_preview(
    device_class: str,
    device_class_id: int,
    **reader_options: Any,
) -> CameraPreviewResult:
    """One-shot convenience wrapper; streaming callers should reuse a reader."""

    with CameraPreviewReader(device_class, device_class_id, **reader_options) as reader:
        return reader.read()


__all__ = [
    "CameraPreviewReader",
    "CameraPreviewResult",
    "read_camera_preview",
]
