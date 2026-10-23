#!/usr/bin/env python3
"""Read Rizon and YAM joint positions from existing device SHM ring buffers.

This module is intentionally a preview-only consumer. It attaches to
``RizonRobot_<device_class_id>_data`` with a read-only handle, copies the latest
complete frame, and never creates, unlinks, or writes shared memory.

The payload emitted by :class:`devices.robot.RizonRobot` is not a bare 7-axis
vector. The supported layouts are:

* 14D single arm: ``q(7), dq(7)``
* 22D single arm: ``q(7), dq(7), tcp_pose(7), gripper(1)``
* 44D dual arm: ``left(q7,dq7,tcp7,gripper1), right(...)``

The 7-axis joint-position slices and, where present, the 7D TCP poses and
scalar gripper positions are exposed from the same copied frame. Protocol-side
arm names (``single``, or ``left``/``right``) are preserved so the web layer
can map them consistently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

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


RIZON_DEVICE_TYPE = "RizonRobot"
YAM_DEVICE_TYPE = "YamHeadRobot"
JOINTS_PER_ARM = 7
YAM_JOINT_COUNT = 6


@dataclass(frozen=True)
class _RobotStateSchema:
    name: str
    joint_slices: Tuple[Tuple[str, int, int], ...]
    tcp_slices: Tuple[Tuple[str, int, int], ...]
    gripper_slices: Tuple[Tuple[str, int, int], ...]
    payload_layout: str


_RIZON_SCHEMAS = {
    (14,): _RobotStateSchema(
        name="single_joint_14",
        joint_slices=(("single", 0, 7),),
        tcp_slices=(),
        gripper_slices=(),
        payload_layout="q7,dq7",
    ),
    (22,): _RobotStateSchema(
        name="single_tcp_gripper_22",
        joint_slices=(("single", 0, 7),),
        tcp_slices=(("single", 14, 21),),
        gripper_slices=(("single", 21, 22),),
        payload_layout="q7,dq7,tcp7,gripper1",
    ),
    (44,): _RobotStateSchema(
        name="dual_tcp_gripper_44",
        joint_slices=(("left", 0, 7), ("right", 22, 29)),
        tcp_slices=(("left", 14, 21), ("right", 36, 43)),
        gripper_slices=(("left", 21, 22), ("right", 43, 44)),
        payload_layout=(
            "left(q7,dq7,tcp7,gripper1),right(q7,dq7,tcp7,gripper1)"
        ),
    ),
}

_YAM_SCHEMAS = {
    (6,): _RobotStateSchema(
        name="yam_joint_6",
        joint_slices=(("yam", 0, 6),),
        tcp_slices=(),
        gripper_slices=(),
        payload_layout="q6",
    ),
    (20,): _RobotStateSchema(
        name="yam_joint_commanded_measured_20",
        joint_slices=(("yam", 0, 6),),
        tcp_slices=(("yam", 13, 20),),
        gripper_slices=(),
        payload_layout="q6,commanded_tcp7,measured_tcp7",
    ),
}


@dataclass(frozen=True)
class RobotStatePreviewResult:
    """JSON-friendly joint positions plus metadata about the source frame."""

    joint_positions: Dict[str, Tuple[float, ...]]
    metadata: Dict[str, Any]
    tcp_poses: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    gripper_positions: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether a fresh, schema-valid state frame was read."""

        return bool(self.joint_positions) and self.metadata.get("status") == "ok"

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "read_error"))

    def to_dict(self) -> Dict[str, Any]:
        """Return a Flask/jsonify-ready payload without NumPy values."""

        return {
            **self.metadata,
            "joint_positions": {
                arm: list(positions)
                for arm, positions in self.joint_positions.items()
            },
            "tcp_poses": {
                arm: list(pose) for arm, pose in self.tcp_poses.items()
            },
            "gripper_positions": dict(self.gripper_positions),
        }


@dataclass(frozen=True)
class _RawRobotFrame:
    payload: bytes
    timestamp_ns: int
    shape: Tuple[int, ...]
    dtype: np.dtype
    schema: _RobotStateSchema
    device_type: str
    fps: float
    hardware_latency_ms: float
    buffer_size: int
    current_frames_count: int
    write_index: int
    read_attempts: int


class _RobotStateReadError(RuntimeError):
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


class RobotStatePreviewReader:
    """Read the newest RizonRobot state frame without owning its lifecycle.

    A lightweight seqlock-style check surrounds the payload copy: both the ring
    header and selected frame timestamp must remain unchanged. A cached handle
    that becomes stale is closed and reopened once, which handles the producer
    unlinking and recreating the same SHM name during a device restart.
    """

    def __init__(
        self,
        device_class_id: int,
        *,
        stale_after_s: float = 1.0,
        max_read_retries: int = 3,
        connector: Callable[..., Any] = connect_to_device_shm,
        clock_ns: Callable[[], int] = time.time_ns,
        device_type: str = RIZON_DEVICE_TYPE,
        schemas: Optional[Dict[Tuple[int, ...], _RobotStateSchema]] = None,
        joints_per_arm: int = JOINTS_PER_ARM,
        label: str = "Rizon",
    ) -> None:
        if stale_after_s < 0:
            raise ValueError("stale_after_s must be non-negative")
        if max_read_retries < 1:
            raise ValueError("max_read_retries must be at least 1")

        self.device_class_id = int(device_class_id)
        self.stale_after_s = float(stale_after_s)
        self.max_read_retries = int(max_read_retries)
        self._connector = connector
        self._clock_ns = clock_ns
        self.device_type = str(device_type)
        self._schemas = schemas or _RIZON_SCHEMAS
        self.joints_per_arm = int(joints_per_arm)
        self.label = str(label)
        self._channel: Optional[Any] = None

    @property
    def shm_name(self) -> str:
        return f"{self.device_type}_{self.device_class_id}_data"

    @property
    def connected(self) -> bool:
        return self._channel is not None

    def __enter__(self) -> "RobotStatePreviewReader":
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
            pass

    def read(self) -> RobotStatePreviewResult:
        """Return latest 7-axis joint positions and frame health metadata."""

        connected_at_start = self.connected
        reconnected = False

        for handle_attempt in range(2):
            try:
                channel = self._ensure_connected()
            except FileNotFoundError:
                return self._error_result(
                    "not_found", f"Shared memory {self.shm_name!r} does not exist"
                )
            except Exception as exc:
                self.close()
                return self._error_result(
                    "connection_error",
                    str(exc),
                    metadata={"reconnected": reconnected},
                )

            try:
                raw = self._read_consistent_frame(channel)
            except _RobotStateReadError as exc:
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
            fresh = age_ns <= int(self.stale_after_s * 1_000_000_000)

            if not fresh and connected_at_start and handle_attempt == 0:
                self.close()
                reconnected = True
                continue

            metadata = self._metadata_for_raw(
                raw,
                status="ok" if fresh else "stale",
                age_ms=age_ms,
                fresh=fresh,
                reconnected=reconnected,
            )
            joint_positions = self._decode_joint_positions(raw)
            if not all(
                np.isfinite(position)
                for positions in joint_positions.values()
                for position in positions
            ):
                return self._error_result(
                    "invalid_data",
                    f"{self.label} joint positions contain NaN or infinity",
                    metadata=metadata,
                )

            tcp_poses = self._decode_tcp_poses(raw)
            for arm, pose in tcp_poses.items():
                if not all(np.isfinite(value) for value in pose):
                    return self._error_result(
                        "invalid_data",
                        f"{self.label} {arm} TCP pose contains NaN or infinity",
                        metadata=metadata,
                    )
                quaternion = np.asarray(pose[3:7], dtype=np.float64)
                if not np.isfinite(quaternion).all() or float(
                    np.linalg.norm(quaternion)
                ) <= np.finfo(np.float64).eps:
                    return self._error_result(
                        "invalid_data",
                        f"{self.label} {arm} TCP quaternion has zero norm",
                        metadata=metadata,
                    )

            gripper_positions = self._decode_gripper_positions(raw)
            for arm, position in gripper_positions.items():
                if not np.isfinite(position):
                    return self._error_result(
                        "invalid_data",
                        f"{self.label} {arm} gripper position contains NaN or infinity",
                        metadata=metadata,
                    )

            if not fresh:
                # A Linux handle can outlive the producer's unlinked name. Force
                # a fresh name lookup on the next request after reporting stale.
                self.close()
            return RobotStatePreviewResult(
                joint_positions=joint_positions,
                metadata=metadata,
                tcp_poses=tcp_poses,
                gripper_positions=gripper_positions,
            )

        return self._error_result("read_error", "Unable to read robot state")

    def _ensure_connected(self) -> Any:
        if self._channel is None:
            channel = self._connector(
                self.device_type, self.device_class_id, read_only=True
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

    def _read_consistent_frame(self, channel: Any) -> _RawRobotFrame:
        try:
            buf = channel.buf
            segment_size = len(buf)
        except (BufferError, OSError, TypeError, ValueError) as exc:
            raise _RobotStateReadError(
                "read_error", str(exc), reconnect=True
            ) from exc

        for attempt in range(1, self.max_read_retries + 1):
            try:
                device_header = unpack_device_header(
                    bytes(buf[:DEVICE_HEADER_SIZE])
                )
                shape, dtype, frame_size, schema = self._validate_device_header(
                    device_header, segment_size
                )

                header_start = DEVICE_HEADER_SIZE
                header_end = header_start + BUFFER_HEADER_SIZE
                buffer_before = unpack_buffer_header(
                    bytes(buf[header_start:header_end])
                )
                self._validate_buffer_header(
                    buffer_before, frame_size, segment_size
                )

                if buffer_before["current_frames_count"] == 0:
                    raise _RobotStateReadError(
                        "empty",
                        f"{self.label} SHM is connected but has no state frame yet",
                        metadata={
                            "shape": shape,
                            "dtype": dtype.name,
                            "schema": schema.name,
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
                timestamp_end = frame_offset + FRAME_HEADER_SIZE
                data_end = frame_offset + frame_size

                timestamp_before = unpack_frame_header(
                    bytes(buf[frame_offset:timestamp_end])
                )["timestamp_ns"]
                payload = bytes(buf[timestamp_end:data_end])
                timestamp_after = unpack_frame_header(
                    bytes(buf[frame_offset:timestamp_end])
                )["timestamp_ns"]
                buffer_after = unpack_buffer_header(
                    bytes(buf[header_start:header_end])
                )
            except _RobotStateReadError:
                raise
            except (BufferError, OSError, TypeError, ValueError) as exc:
                raise _RobotStateReadError(
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
                return _RawRobotFrame(
                    payload=payload,
                    timestamp_ns=timestamp_before,
                    shape=shape,
                    dtype=dtype,
                    schema=schema,
                    device_type=device_header["device_type"],
                    fps=float(device_header["fps"]),
                    hardware_latency_ms=float(
                        device_header["hardware_latency_ms"]
                    ),
                    buffer_size=int(buffer_before["buffer_size"]),
                    current_frames_count=int(
                        buffer_before["current_frames_count"]
                    ),
                    write_index=int(buffer_before["write_index"]),
                    read_attempts=attempt,
                )

        raise _RobotStateReadError(
            "inconsistent",
            f"{self.label} producer changed write_index or timestamp during frame copy",
            metadata={"read_attempts": self.max_read_retries},
            reconnect=True,
        )

    def _validate_device_header(
        self, device_header: Dict[str, Any], segment_size: int
    ) -> Tuple[Tuple[int, ...], np.dtype, int, _RobotStateSchema]:
        if device_header["device_type"] != self.device_type:
            raise _RobotStateReadError(
                "invalid_header",
                f"Device type in SHM header is not {self.device_type}",
                reconnect=True,
            )
        if int(device_header["device_class_id"]) != self.device_class_id:
            raise _RobotStateReadError(
                "invalid_header",
                f"Device ID in SHM header does not match requested {self.device_type}",
                reconnect=True,
            )

        dtype_name = str(device_header["data_dtype"])
        dtype_type = DTYPE_MAP.get(dtype_name)
        if dtype_type is None or not np.issubdtype(dtype_type, np.floating):
            raise _RobotStateReadError(
                "unsupported_dtype",
                f"{self.label} state requires a floating dtype, got {dtype_name!r}",
                metadata={"dtype": dtype_name},
            )
        dtype = np.dtype(dtype_type)

        shape = tuple(int(dimension) for dimension in device_header["shape"])
        schema = self._schemas.get(shape)
        if schema is None:
            raise _RobotStateReadError(
                "invalid_shape",
                f"Unsupported {self.label} state shape: {shape}",
                metadata={"shape": shape, "dtype": dtype.name},
            )

        expected_data_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        frame_size = int(device_header["frame_size"])
        if frame_size != FRAME_HEADER_SIZE + expected_data_size:
            raise _RobotStateReadError(
                "invalid_header",
                f"Frame size does not match {self.label} shape and dtype",
                metadata={"shape": shape, "dtype": dtype.name},
                reconnect=True,
            )
        if segment_size < DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_size:
            raise _RobotStateReadError(
                "invalid_header",
                "Shared-memory segment is smaller than one declared frame",
                metadata={"shape": shape, "dtype": dtype.name},
                reconnect=True,
            )
        return shape, dtype, frame_size, schema

    def _validate_buffer_header(
        self,
        buffer_header: Dict[str, int],
        frame_size: int,
        segment_size: int,
    ) -> None:
        buffer_size = int(buffer_header["buffer_size"])
        frame_count = int(buffer_header["current_frames_count"])
        write_index = int(buffer_header["write_index"])
        if buffer_size <= 0:
            raise _RobotStateReadError(
                "invalid_header",
                f"{self.label} SHM buffer_size must be positive",
                reconnect=True,
            )
        if not 0 <= frame_count <= buffer_size:
            raise _RobotStateReadError(
                "invalid_header",
                f"{self.label} SHM frame count is out of range",
                reconnect=True,
            )
        if not 0 <= write_index < buffer_size:
            raise _RobotStateReadError(
                "invalid_header",
                f"{self.label} SHM write index is out of range",
                reconnect=True,
            )
        required_size = (
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + buffer_size * frame_size
        )
        if required_size > segment_size:
            raise _RobotStateReadError(
                "invalid_header",
                f"{self.label} SHM ring layout exceeds the attached segment",
                reconnect=True,
            )

    @staticmethod
    def _decode_joint_positions(
        raw: _RawRobotFrame,
    ) -> Dict[str, Tuple[float, ...]]:
        state = np.frombuffer(raw.payload, dtype=raw.dtype).reshape(raw.shape)
        flat_state = state.reshape(-1)
        return {
            arm: tuple(float(value) for value in flat_state[start:end])
            for arm, start, end in raw.schema.joint_slices
        }

    @staticmethod
    def _decode_tcp_poses(
        raw: _RawRobotFrame,
    ) -> Dict[str, Tuple[float, ...]]:
        state = np.frombuffer(raw.payload, dtype=raw.dtype).reshape(raw.shape)
        flat_state = state.reshape(-1)
        return {
            arm: tuple(float(value) for value in flat_state[start:end])
            for arm, start, end in raw.schema.tcp_slices
        }

    @staticmethod
    def _decode_gripper_positions(
        raw: _RawRobotFrame,
    ) -> Dict[str, float]:
        state = np.frombuffer(raw.payload, dtype=raw.dtype).reshape(raw.shape)
        flat_state = state.reshape(-1)
        return {
            arm: float(flat_state[start])
            for arm, start, end in raw.schema.gripper_slices
            if end - start == 1
        }

    def _metadata_for_raw(
        self,
        raw: _RawRobotFrame,
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
            "schema": raw.schema.name,
            "payload_layout": raw.schema.payload_layout,
            "arm_order": tuple(arm for arm, _, _ in raw.schema.joint_slices),
            "tcp_arm_order": tuple(
                arm for arm, _, _ in raw.schema.tcp_slices
            ),
            "gripper_arm_order": tuple(
                arm for arm, _, _ in raw.schema.gripper_slices
            ),
            "joints_per_arm": self.joints_per_arm,
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
    ) -> RobotStatePreviewResult:
        details: Dict[str, Any] = {
            "status": status,
            "source": "device_shm",
            "shm_name": self.shm_name,
            "device_type": self.device_type,
            "device_class_id": self.device_class_id,
            "timestamp_ns": None,
            "age_ms": None,
            "shape": None,
            "dtype": None,
            "schema": None,
            "arm_order": (),
            "tcp_arm_order": (),
            "gripper_arm_order": (),
            "joints_per_arm": self.joints_per_arm,
            "fresh": False,
            "reconnected": False,
            "message": message,
        }
        if metadata:
            details.update(metadata)
            details["status"] = status
            details["message"] = message
        return RobotStatePreviewResult(
            joint_positions={},
            metadata=details,
            tcp_poses={},
            gripper_positions={},
        )


def read_robot_state_preview(
    device_class_id: int, **reader_options: Any
) -> RobotStatePreviewResult:
    """One-shot convenience wrapper; polling callers should reuse a reader."""

    with RobotStatePreviewReader(device_class_id, **reader_options) as reader:
        return reader.read()


class YamStatePreviewReader(RobotStatePreviewReader):
    """Read the six commanded YAM joints from ``YamHeadRobot_<id>_data``."""

    def __init__(self, device_class_id: int, **reader_options: Any) -> None:
        super().__init__(
            device_class_id,
            device_type=YAM_DEVICE_TYPE,
            schemas=_YAM_SCHEMAS,
            joints_per_arm=YAM_JOINT_COUNT,
            label="YAM",
            **reader_options,
        )


__all__ = [
    "JOINTS_PER_ARM",
    "RIZON_DEVICE_TYPE",
    "YAM_DEVICE_TYPE",
    "YAM_JOINT_COUNT",
    "RobotStatePreviewReader",
    "RobotStatePreviewResult",
    "YamStatePreviewReader",
    "read_robot_state_preview",
]
