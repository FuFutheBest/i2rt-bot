#!/usr/bin/env python3
"""Terminal-first asynchronous NEDF3 recorder for I2RT dual-arm teleoperation.
    Author: Yichen Xie
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import select
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

import numpy as np
import yaml

from consumer.io.nedf3_i2rt_contract import (
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_CONTRACT_VERSION,
    NEDF3_I2RT_HAND_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_METADATA_TOPIC,
    NEDF3_RECORD_TOPICS,
    NEDF3EpisodeMetadata,
    NEDF3Writer,
    create_vendor_writer,
)
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_shared_memory,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
    unpack_frame_header,
)
from utils.logger_config import logger

WriterFactory = Callable[[Path, NEDF3EpisodeMetadata], NEDF3Writer]


def select_main_camera(synthetic_image_roles: Collection[str]) -> str:
    """Select a real camera as the episode thumbnail and main view."""
    return "camera_0" if "head" in synthetic_image_roles else "camera_2"

I2RT_CAMERA_SOURCE_SHAPES = {
    "right": ((360, 640, 3), (480, 640, 3)),
    "left": ((360, 640, 3), (480, 640, 3)),
    "head": ((480, 640, 3),),
}


@dataclass(frozen=True)
class DeviceFrame:
    timestamp_ns: int
    data: np.ndarray


@dataclass(frozen=True)
class I2RTState:
    left_joints: np.ndarray
    left_gripper: float
    right_joints: np.ndarray
    right_gripper: float


@dataclass(frozen=True)
class SaverConfig:
    output_dir: Path
    sample_hz: float
    queue_size: int
    jpeg_quality: int
    leader_shm: str
    follower_shm: str
    camera_shm: Mapping[str, str | None]
    task: str
    task_description: str
    gripper_sample_hz: float | None = None
    camera_sample_hz: float | None = None
    synthetic_camera_shapes: Mapping[str, tuple[int, int, int]] = field(
        default_factory=dict
    )
    episode_signal_pattern: str = "/tmp/superinference_episode_{signal}"
    dm3_user_id: str = ""
    dm3_user_name: str = ""
    task_execution_id: str = ""

    @property
    def resolved_gripper_sample_hz(self) -> float:
        return (
            self.sample_hz
            if self.gripper_sample_hz is None
            else self.gripper_sample_hz
        )

    @property
    def resolved_camera_sample_hz(self) -> float:
        return (
            self.sample_hz
            if self.camera_sample_hz is None
            else self.camera_sample_hz
        )

    @property
    def polling_hz(self) -> float:
        return 2.0 * max(
            self.sample_hz,
            self.resolved_gripper_sample_hz,
            self.resolved_camera_sample_hz,
        )

    @property
    def camera_polling_hz(self) -> float:
        return max(self.sample_hz, 2.0 * self.resolved_camera_sample_hz)


@dataclass(frozen=True)
class WriteRecord:
    topic: str
    message: Any
    timestamp_ns: int


@dataclass(frozen=True)
class CameraWriteRecord:
    topic: str
    role: str
    image: np.ndarray
    timestamp_ns: int
    jpeg_quality: int


@dataclass(frozen=True)
class WriteBatch:
    records: tuple[WriteRecord | CameraWriteRecord, ...]


class EpisodeSignalWatcher:
    """Consume edge-triggered episode commands emitted by teleoperation."""

    SIGNALS = ("toggle", "discard")

    def __init__(self, pattern: str) -> None:
        if "{signal}" not in pattern:
            raise ValueError("episode signal pattern must contain {signal}")
        self.pattern = pattern

    def clear(self) -> None:
        for signal in self.SIGNALS:
            try:
                self._path(signal).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Could not clear stale episode {signal} signal: {exc}")

    def drain(self) -> list[str]:
        pending: list[tuple[int, str, Path]] = []
        for signal in self.SIGNALS:
            path = self._path(signal)
            try:
                raw = path.read_text(encoding="ascii").strip()
                timestamp_ns = int(raw)
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                try:
                    timestamp_ns = path.stat().st_mtime_ns
                except FileNotFoundError:
                    continue
            pending.append((timestamp_ns, signal, path))

        actions: list[str] = []
        for _timestamp_ns, signal, path in sorted(pending):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(f"Could not consume episode {signal} signal: {exc}")
                continue
            actions.append(signal)
        return actions

    def _path(self, signal: str) -> Path:
        return Path(self.pattern.format(signal=signal))


def split_i2rt_state(values: Any) -> I2RTState:
    """Split ``[left six, left hand, right six, right hand]``."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size != 14:
        raise ValueError(f"I2RT state must contain 14 values, got {flat.size}")
    if not np.isfinite(flat).all():
        raise ValueError("I2RT state must contain only finite values")
    return I2RTState(
        left_joints=flat[:6].copy(),
        left_gripper=float(flat[6]),
        right_joints=flat[7:13].copy(),
        right_gripper=float(flat[13]),
    )


def arm_state_message(values: Sequence[float]) -> Any:
    """Encode exactly six I2RT joint positions as ``ArmState``."""
    from nmx_msg.Arm_pb2 import ArmState

    joints = np.asarray(values, dtype=np.float64).reshape(-1)
    if joints.size != 6 or not np.isfinite(joints).all():
        raise ValueError("I2RT ArmState requires exactly six finite joint positions")
    message = ArmState()
    for index, value in enumerate(joints):
        joint = message.joints.add()
        joint.name = f"joint{index}"
        joint.pos = float(value)
    return message


def hand_state_message(value: float) -> Any:
    """Encode one finite I2RT gripper state as ``HandState``."""
    from nmx_msg.Hand_pb2 import HandState

    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError("I2RT gripper state must be finite")
    message = HandState()
    joint = message.joints.add()
    joint.name = "gripper"
    joint.pos = resolved
    return message


def prepare_camera_image(role: str, image: np.ndarray) -> np.ndarray:
    """Validate an I2RT camera frame and preserve its captured geometry."""
    if role not in I2RT_CAMERA_SOURCE_SHAPES:
        raise ValueError(f"unknown I2RT camera role: {role!r}")
    frame = np.asarray(image)
    allowed_shapes = I2RT_CAMERA_SOURCE_SHAPES[role]
    if frame.shape not in allowed_shapes:
        raise ValueError(
            f"{role} camera frame must have one of {allowed_shapes}, got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"{role} camera frame must use uint8, got {frame.dtype}")
    return np.ascontiguousarray(frame)


def rgbd_message(image: np.ndarray, *, jpeg_quality: int = 90) -> Any:
    """Encode an HWC uint8 image into an RGBD message with empty depth."""
    import cv2
    from nmx_msg.Image_pb2 import ImageDataType, RGBD

    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"camera frame must have shape (height, width, 3), got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"camera frame must use uint8, got {frame.dtype}")
    quality = int(jpeg_quality)
    if not 1 <= quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode camera frame")

    message = RGBD()
    message.rgb.rows = int(frame.shape[0])
    message.rgb.cols = int(frame.shape[1])
    message.rgb.channels = 3
    message.rgb.type = ImageDataType.ENCODED
    message.rgb.encoded_format = "jpeg"
    message.rgb.data = encoded.tobytes()
    message.depth.rows = 0
    message.depth.cols = 0
    message.depth.channels = 0
    message.depth.type = ImageDataType.ENCODED
    message.depth.encoded_format = ""
    message.depth.data = b""
    message.depth_scale = 1.0
    return message


def metadata_message(metadata: NEDF3EpisodeMetadata) -> Any:
    from nmx_msg.Metadata_pb2 import Metadata

    message = Metadata()
    message.version = NEDF3_I2RT_CONTRACT_VERSION
    message.metadata = json.dumps(
        metadata.as_dict(), allow_nan=False, ensure_ascii=True, sort_keys=True
    ).encode("utf-8")
    return message


class DeviceShmReader:
    """Read stable snapshots directly from one device ring buffer."""

    def __init__(self, shm_name: str) -> None:
        self.shm_name = shm_name
        self.channel: Any = None
        self.header: Optional[dict[str, Any]] = None

    def connect(self) -> None:
        if self.channel is not None:
            return
        self.channel = connect_to_shared_memory(self.shm_name, read_only=True)
        self.header = unpack_device_header(self.channel.buf[:DEVICE_HEADER_SIZE])

    @property
    def shape(self) -> tuple[int, ...]:
        if self.header is None:
            raise RuntimeError(f"reader {self.shm_name} is not connected")
        return tuple(int(value) for value in self.header["shape"])

    @property
    def dtype(self) -> np.dtype[Any]:
        if self.header is None:
            raise RuntimeError(f"reader {self.shm_name} is not connected")
        return np.dtype(get_dtype(self.header["data_dtype"]))

    def read_latest(self) -> Optional[DeviceFrame]:
        if self.channel is None or self.header is None:
            raise RuntimeError(f"reader {self.shm_name} is not connected")
        frame_size = int(self.header["frame_size"])
        data_size = frame_size - FRAME_HEADER_SIZE
        expected_size = int(np.prod(self.shape)) * self.dtype.itemsize
        if data_size != expected_size:
            raise RuntimeError(
                f"{self.shm_name} frame bytes {data_size} do not match "
                f"shape {self.shape} and dtype {self.dtype}"
            )

        for _attempt in range(3):
            before = unpack_buffer_header(
                self.channel.buf[
                    DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
                ]
            )
            if before["current_frames_count"] == 0:
                return None
            frame_index = (before["write_index"] - 1) % before["buffer_size"]
            offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_index * frame_size
            timestamp_before = unpack_frame_header(
                self.channel.buf[offset : offset + FRAME_HEADER_SIZE]
            )["timestamp_ns"]
            raw = bytes(
                self.channel.buf[
                    offset + FRAME_HEADER_SIZE : offset + FRAME_HEADER_SIZE + data_size
                ]
            )
            timestamp_after = unpack_frame_header(
                self.channel.buf[offset : offset + FRAME_HEADER_SIZE]
            )["timestamp_ns"]
            after = unpack_buffer_header(
                self.channel.buf[
                    DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
                ]
            )
            if (
                timestamp_before > 0
                and timestamp_before == timestamp_after
                and before["write_index"] == after["write_index"]
            ):
                array = np.frombuffer(raw, dtype=self.dtype).reshape(self.shape).copy()
                return DeviceFrame(timestamp_ns=timestamp_before, data=array)
        raise RuntimeError(f"could not obtain a stable frame from {self.shm_name}")

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.header = None


class I2RTNEDF3Saver:
    """Record I2RT and camera SHMs into independently finalized episodes."""

    def __init__(
        self,
        config: SaverConfig,
        *,
        writer_factory: Optional[WriterFactory] = None,
    ) -> None:
        self.config = config
        self.writer_factory = writer_factory or (
            lambda path, metadata: create_vendor_writer(
                output_path=path, metadata=metadata
            )
        )
        self.readers: dict[str, DeviceShmReader] = {
            "leader": DeviceShmReader(config.leader_shm),
            "follower": DeviceShmReader(config.follower_shm),
            **{
                f"camera:{role}": DeviceShmReader(shm_name)
                for role, shm_name in config.camera_shm.items()
                if shm_name is not None
            },
        }
        self._synthetic_images = {
            role: np.zeros(shape, dtype=np.uint8)
            for role, shape in config.synthetic_camera_shapes.items()
        }
        self._queue: queue.Queue[WriteBatch | None] = queue.Queue(config.queue_size)
        self._writer: Optional[NEDF3Writer] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_error: Optional[BaseException] = None
        self._episode_dir: Optional[Path] = None
        self._episode_started_ns: Optional[int] = None
        self._active = False
        self._last_timestamps: dict[str, int] = {}
        self._next_sample_ns: dict[str, int] = {}
        self._topic_counts = {topic: 0 for topic in NEDF3_RECORD_TOPICS}
        self._dropped_batches = 0
        self._last_drop_warning = 0.0
        self._state_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._active

    def connect(self) -> None:
        connected: list[DeviceShmReader] = []
        try:
            for reader in self.readers.values():
                reader.connect()
                connected.append(reader)
            for role in ("leader", "follower"):
                reader = self.readers[role]
                if reader.shape != (14,) or not np.issubdtype(reader.dtype, np.floating):
                    raise ValueError(
                        f"{reader.shm_name} must be a floating-point (14,) stream; "
                        f"got {reader.shape} {reader.dtype}"
                    )
            for role in NEDF3_CAMERA_TOPICS:
                key = f"camera:{role}"
                if key not in self.readers:
                    if role not in self._synthetic_images:
                        raise ValueError(
                            f"camera role {role!r} has neither SHM nor a synthetic image"
                        )
                    prepare_camera_image(role, self._synthetic_images[role])
                    continue
                reader = self.readers[key]
                allowed_shapes = I2RT_CAMERA_SOURCE_SHAPES[role]
                if reader.shape not in allowed_shapes or reader.dtype != np.uint8:
                    raise ValueError(
                        f"{reader.shm_name} must be an HWC uint8 RGB stream with "
                        f"one of the shapes {allowed_shapes}; got "
                        f"{reader.shape} {reader.dtype}"
                    )
        except Exception:
            for reader in connected:
                reader.close()
            raise

    def close(self) -> None:
        if self.active:
            self.stop_episode()
        for reader in self.readers.values():
            reader.close()

    def start_episode(self) -> Path:
        with self._state_lock:
            if self._active:
                raise RuntimeError("an I2RT NEDF3 episode is already active")
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self._recover_incomplete_episodes()
            index = self._next_episode_index()
            episode_id = f"episode_{index:04d}"
            # dm3-core 0.2.x derives the chunk directory with a string split on
            # the first dot, so keep every parent component dot-free.
            episode_dir = self.config.output_dir / f"{episode_id}_inprogress"
            episode_dir.mkdir(parents=False, exist_ok=False)
            metadata_extra: dict[str, Any] = {
                "state_layout": (
                    "left_joint0..5,left_gripper,"
                    "right_joint0..5,right_gripper"
                ),
                "sample_hz": {
                    "arm": self.config.sample_hz,
                    "gripper": self.config.resolved_gripper_sample_hz,
                    "camera": self.config.resolved_camera_sample_hz,
                },
                "synthetic_black_cameras": sorted(self._synthetic_images),
                "camera_color_space": {
                    "shared_memory": "rgb",
                    "jpeg": "rgb",
                    "opencv_encoder_input": "bgr",
                },
                "camera_processing": {
                    "head": {"transform": "none"},
                    "wrist": {"transform": "none"},
                },
            }
            if self.config.dm3_user_id:
                metadata_extra["user_id"] = self.config.dm3_user_id
            if self.config.dm3_user_name:
                metadata_extra["user_name"] = self.config.dm3_user_name
            if self.config.task_execution_id:
                metadata_extra["task_execution_id"] = self.config.task_execution_id
            metadata = NEDF3EpisodeMetadata(
                episode_id=episode_id,
                task=self.config.task,
                task_description=self.config.task_description,
                device_type=("I2RTLeader", "I2RTFollower", "RGBDCamera"),
                hardware_string=(
                    self.config.leader_shm,
                    self.config.follower_shm,
                    *(
                        shm_name
                        for shm_name in self.config.camera_shm.values()
                        if shm_name is not None
                    ),
                ),
                main_camera=select_main_camera(self._synthetic_images),
                extra=metadata_extra,
            )
            try:
                writer = self.writer_factory(episode_dir, metadata)
            except Exception:
                shutil.rmtree(episode_dir, ignore_errors=True)
                raise

            self._queue = queue.Queue(self.config.queue_size)
            self._writer = writer
            self._writer_error = None
            self._episode_dir = episode_dir
            self._episode_started_ns = time.time_ns()
            self._last_timestamps.clear()
            self._next_sample_ns.clear()
            self._topic_counts = {topic: 0 for topic in NEDF3_RECORD_TOPICS}
            self._dropped_batches = 0
            self._last_drop_warning = 0.0
            self._active = True
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="i2rt-nedf3-writer",
                daemon=True,
            )
            self._writer_thread.start()
            self._enqueue(
                WriteBatch(
                    (
                        WriteRecord(
                            NEDF3_METADATA_TOPIC,
                            metadata_message(metadata),
                            self._episode_started_ns,
                        ),
                    )
                )
            )
            return episode_dir

    def collect_once(self, *, now_ns: int | None = None) -> int:
        if not self._active:
            return 0
        if self._writer_error is not None:
            raise RuntimeError("NEDF3 writer failed") from self._writer_error

        sample_time_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        camera_due = self._sample_due(
            "camera", self.config.resolved_camera_sample_hz, sample_time_ns
        )
        camera_poll_due = self._sample_due(
            "camera_poll", self.config.camera_polling_hz, sample_time_ns
        )
        batches = 0
        for role in ("leader", "follower"):
            arm_schedule = f"{role}:arm_schedule"
            gripper_schedule = f"{role}:gripper_schedule"
            arm_due = self._sample_due(
                arm_schedule, self.config.sample_hz, sample_time_ns
            )
            gripper_due = self._sample_due(
                gripper_schedule,
                self.config.resolved_gripper_sample_hz,
                sample_time_ns,
            )
            if arm_due or gripper_due:
                frame = self.readers[role].read_latest()
                if frame is None:
                    continue
                arm_key = f"{role}:arm"
                gripper_key = f"{role}:gripper"
                write_arm = arm_due and frame.timestamp_ns > self._last_timestamps.get(
                    arm_key, 0
                )
                write_gripper = (
                    gripper_due
                    and frame.timestamp_ns > self._last_timestamps.get(gripper_key, 0)
                )
                if not write_arm and not write_gripper:
                    continue
                state = split_i2rt_state(frame.data)
                records = []
                for side in ("left", "right"):
                    joints = state.left_joints if side == "left" else state.right_joints
                    gripper = (
                        state.left_gripper if side == "left" else state.right_gripper
                    )
                    if write_arm:
                        records.append(
                            WriteRecord(
                                NEDF3_I2RT_JOINT_TOPICS[role][side],
                                arm_state_message(joints),
                                frame.timestamp_ns,
                            )
                        )
                    if write_gripper:
                        records.append(
                            WriteRecord(
                                NEDF3_I2RT_HAND_TOPICS[role][side],
                                hand_state_message(gripper),
                                frame.timestamp_ns,
                            )
                        )
                if self._enqueue(WriteBatch(tuple(records))):
                    if write_arm:
                        self._last_timestamps[arm_key] = frame.timestamp_ns
                        self._mark_sampled(
                            arm_schedule, self.config.sample_hz, sample_time_ns
                        )
                    if write_gripper:
                        self._last_timestamps[gripper_key] = frame.timestamp_ns
                        self._mark_sampled(
                            gripper_schedule,
                            self.config.resolved_gripper_sample_hz,
                            sample_time_ns,
                        )
                    batches += 1

        wrote_synthetic_camera = False
        if camera_poll_due:
            for role, topic in NEDF3_CAMERA_TOPICS.items():
                key = f"camera:{role}"
                if role in self._synthetic_images:
                    if not camera_due:
                        continue
                    frame = DeviceFrame(
                        timestamp_ns=time.time_ns(),
                        data=self._synthetic_images[role],
                    )
                else:
                    frame = self.readers[key].read_latest()
                if frame is None or frame.timestamp_ns <= self._last_timestamps.get(
                    key, 0
                ):
                    continue
                batch = WriteBatch(
                    (
                        CameraWriteRecord(
                            topic,
                            role,
                            frame.data,
                            frame.timestamp_ns,
                            self.config.jpeg_quality,
                        ),
                    )
                )
                if self._enqueue(batch):
                    self._last_timestamps[key] = frame.timestamp_ns
                    wrote_synthetic_camera = wrote_synthetic_camera or (
                        role in self._synthetic_images
                    )
                    batches += 1
            self._mark_sampled(
                "camera_poll", self.config.camera_polling_hz, sample_time_ns
            )
            if camera_due and wrote_synthetic_camera:
                self._mark_sampled(
                    "camera", self.config.resolved_camera_sample_hz, sample_time_ns
                )
        return batches

    def _sample_due(self, stream: str, frequency_hz: float, now_ns: int) -> bool:
        deadline_ns = self._next_sample_ns.get(stream)
        return deadline_ns is None or now_ns >= deadline_ns

    def _mark_sampled(self, stream: str, frequency_hz: float, now_ns: int) -> None:
        period_ns = max(1, int(round(1e9 / frequency_hz)))
        deadline_ns = self._next_sample_ns.get(stream)
        self._next_sample_ns[stream] = (
            now_ns + period_ns if deadline_ns is None else deadline_ns + period_ns
        )

    def stop_episode(self, *, discard: bool = False) -> Path:
        with self._state_lock:
            if not self._active or self._episode_dir is None:
                raise RuntimeError("no I2RT NEDF3 episode is active")
            self._active = False
            episode_dir = self._episode_dir

        self._queue.join()
        self._queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)
            if self._writer_thread.is_alive():
                self._writer_error = RuntimeError("NEDF3 writer thread did not stop")
        try:
            if self._writer is not None:
                self._writer.close()
        except BaseException as exc:
            self._writer_error = self._writer_error or exc

        ended_ns = time.time_ns()
        stats = {
            "contract_version": NEDF3_I2RT_CONTRACT_VERSION,
            "started_at_ns": self._episode_started_ns,
            "ended_at_ns": ended_ns,
            "duration_s": (
                0.0
                if self._episode_started_ns is None
                else (ended_ns - self._episode_started_ns) / 1e9
            ),
            "topic_counts": self._topic_counts,
            "dropped_batches": self._dropped_batches,
            "writer_error": None if self._writer_error is None else repr(self._writer_error),
        }
        (episode_dir / "episode_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        self._writer = None
        self._writer_thread = None
        self._episode_dir = None
        self._episode_started_ns = None

        if discard:
            shutil.rmtree(episode_dir, ignore_errors=True)
            return episode_dir

        missing = [topic for topic, count in self._topic_counts.items() if count == 0]
        episode_name = episode_dir.name.removesuffix("_inprogress")
        if self._writer_error is not None or missing:
            quarantine = self.config.output_dir / "quarantine"
            quarantine.mkdir(exist_ok=True)
            destination = quarantine / episode_name
            if destination.exists():
                destination = quarantine / f"{episode_name}_{uuid.uuid4().hex[:8]}"
            episode_dir.rename(destination)
            reason = {
                "missing_topics": missing,
                "writer_error": stats["writer_error"],
            }
            (destination / "quarantine_reason.json").write_text(
                json.dumps(reason, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return destination

        destination = self.config.output_dir / episode_name
        episode_dir.rename(destination)
        return destination

    def _writer_loop(self) -> None:
        while True:
            batch = self._queue.get()
            try:
                if batch is None:
                    return
                if self._writer_error is not None or self._writer is None:
                    continue
                for record in batch.records:
                    message = (
                        rgbd_message(
                            prepare_camera_image(record.role, record.image),
                            jpeg_quality=record.jpeg_quality,
                        )
                        if isinstance(record, CameraWriteRecord)
                        else record.message
                    )
                    self._writer.write_message(
                        record.topic, message, record.timestamp_ns
                    )
                    self._topic_counts[record.topic] += 1
            except BaseException as exc:
                self._writer_error = exc
            finally:
                self._queue.task_done()

    def _enqueue(self, batch: WriteBatch) -> bool:
        try:
            self._queue.put_nowait(batch)
            return True
        except queue.Full:
            self._dropped_batches += 1
            now = time.monotonic()
            if now - self._last_drop_warning >= 1.0:
                logger.warning(
                    "I2RT NEDF3 writer queue is full; "
                    f"dropped {self._dropped_batches} batches"
                )
                self._last_drop_warning = now
            return False

    def _next_episode_index(self) -> int:
        indices: list[int] = []
        candidates = list(self.config.output_dir.iterdir())
        quarantine = self.config.output_dir / "quarantine"
        if quarantine.is_dir():
            candidates.extend(quarantine.iterdir())
        for path in candidates:
            match = re.match(r"episode_(\d+)", path.name)
            if match is not None:
                indices.append(int(match.group(1)))
        return max(indices, default=-1) + 1

    def _recover_incomplete_episodes(self) -> None:
        for episode_dir in self.config.output_dir.glob("episode_*_inprogress"):
            if not episode_dir.is_dir():
                continue
            episode_name = episode_dir.name.removesuffix("_inprogress")
            quarantine = self.config.output_dir / "quarantine"
            quarantine.mkdir(exist_ok=True)
            destination = quarantine / f"{episode_name}_interrupted"
            if destination.exists():
                destination = quarantine / (
                    f"{episode_name}_interrupted_{uuid.uuid4().hex[:8]}"
                )
            episode_dir.rename(destination)
            (destination / "quarantine_reason.json").write_text(
                json.dumps(
                    {"reason": "recorder exited before episode finalization"},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


def load_config(path: str | Path) -> SaverConfig:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    devices = document.get("devices", {})
    cameras = devices.get("cameras", {}) if isinstance(devices, Mapping) else {}
    synthetic_camera_shapes_raw = document.get("synthetic_camera_shapes", {})
    required_roles = set(NEDF3_CAMERA_TOPICS)
    if set(cameras) != required_roles:
        raise ValueError(
            f"devices.cameras must define exactly {sorted(required_roles)}, got {sorted(cameras)}"
        )
    if not isinstance(synthetic_camera_shapes_raw, Mapping):
        raise ValueError("synthetic_camera_shapes must be a mapping")
    synthetic_camera_shapes: dict[str, tuple[int, int, int]] = {}
    for role, raw_shape in synthetic_camera_shapes_raw.items():
        if role not in required_roles:
            raise ValueError(f"unknown synthetic camera role: {role}")
        if (
            not isinstance(raw_shape, (list, tuple))
            or len(raw_shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_shape
            )
        ):
            raise ValueError(
                f"synthetic_camera_shapes.{role} must be [height, width, 3]"
            )
        shape = tuple(raw_shape)
        if shape[0] <= 0 or shape[1] <= 0 or shape[2] != 3:
            raise ValueError(
                f"synthetic_camera_shapes.{role} must be [height, width, 3]"
            )
        synthetic_camera_shapes[str(role)] = shape
    synthetic_roles = {str(role) for role, name in cameras.items() if name is None}
    if set(synthetic_camera_shapes) != synthetic_roles:
        raise ValueError(
            "synthetic_camera_shapes must define exactly the camera roles whose "
            f"SHM value is null; expected {sorted(synthetic_roles)}"
        )
    signal_pattern = str(
        document.get(
            "episode_signal_pattern", "/tmp/superinference_episode_{signal}"
        )
    )
    if "{signal}" not in signal_pattern:
        raise ValueError("episode_signal_pattern must contain {signal}")
    sample_hz = float(document.get("sample_hz", 25.0))
    gripper_sample_hz = float(document.get("gripper_sample_hz", sample_hz))
    camera_sample_hz = float(document.get("camera_sample_hz", sample_hz))
    queue_size = int(document.get("queue_size", 5000))
    jpeg_quality = int(document.get("jpeg_quality", 90))
    for field_name, frequency_hz in (
        ("sample_hz", sample_hz),
        ("gripper_sample_hz", gripper_sample_hz),
        ("camera_sample_hz", camera_sample_hz),
    ):
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError(f"{field_name} must be finite and positive")
    if queue_size <= 0:
        raise ValueError("queue_size must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    return SaverConfig(
        output_dir=Path(document.get("output_dir", "data/i2rt_nedf3")),
        sample_hz=sample_hz,
        queue_size=queue_size,
        jpeg_quality=jpeg_quality,
        leader_shm=str(devices.get("leader", "I2RTRobot_0_data")),
        follower_shm=str(devices.get("follower", "I2RTRobot_1_data")),
        camera_shm={
            str(role): None if name is None else str(name)
            for role, name in cameras.items()
        },
        task=str(document.get("task", "data_collection")),
        task_description=str(document.get("task_description", "")),
        gripper_sample_hz=gripper_sample_hz,
        camera_sample_hz=camera_sample_hz,
        synthetic_camera_shapes=synthetic_camera_shapes,
        episode_signal_pattern=signal_pattern,
        dm3_user_id=str(document.get("dm3_user_id", "")),
        dm3_user_name=str(document.get("dm3_user_name", "")),
        task_execution_id=str(document.get("task_execution_id", "")),
    )


def apply_runtime_overrides(
    config: SaverConfig,
    *,
    output_dir: str | None = None,
    task_name: str | None = None,
    task_description: str | None = None,
    dm3_user_id: str | None = None,
    dm3_user_name: str | None = None,
    task_execution_id: str | None = None,
) -> SaverConfig:
    """Apply GUI-provided episode metadata without mutating the YAML config."""
    values: dict[str, Any] = {}
    if output_dir is not None:
        values["output_dir"] = Path(output_dir).expanduser()
    if task_name is not None:
        values["task"] = task_name
    if task_description is not None:
        values["task_description"] = task_description
    if dm3_user_id is not None:
        values["dm3_user_id"] = dm3_user_id
    if dm3_user_name is not None:
        values["dm3_user_name"] = dm3_user_name
    if task_execution_id is not None:
        values["task_execution_id"] = task_execution_id
    return replace(config, **values)


def _advance_sample_deadline(deadline: float, interval: float, now: float) -> float:
    elapsed_intervals = max(1, int((now - deadline) // interval) + 1)
    return deadline + elapsed_intervals * interval


def run_terminal(
    saver: I2RTNEDF3Saver,
    *,
    auto_start: bool,
    duration: float,
    clock: Callable[[], float] = time.monotonic,
    wait_for_input: Optional[Callable[[float], bool]] = None,
) -> int:
    saver.connect()
    signal_watcher = EpisodeSignalWatcher(saver.config.episode_signal_pattern)
    signal_watcher.clear()
    started_at = clock()
    if auto_start:
        print(f"Recording to {saver.start_episode()}")
    print(
        "Ready: right second button=start/finish, left second button=discard. "
        "Keyboard: Enter=start/stop, d=discard, q=quit"
    )
    interval = 1.0 / saver.config.polling_hz
    next_sample = clock()
    if wait_for_input is None:
        wait_for_input = lambda timeout: bool(
            select.select([sys.stdin], [], [], timeout)[0]
        )
    try:
        while True:
            now = clock()
            if saver.active and now >= next_sample:
                saver.collect_once()
                now = clock()
                next_sample = _advance_sample_deadline(next_sample, interval, now)
            if duration > 0 and auto_start and now - started_at >= duration:
                destination = saver.stop_episode()
                print(f"Finalized {destination}")
                return 0
            for action in signal_watcher.drain():
                if action == "discard":
                    if saver.active:
                        discarded = saver.stop_episode(discard=True)
                        print(f"Discarded {discarded}")
                elif saver.active:
                    print(f"Finalized {saver.stop_episode()}")
                else:
                    print(f"Recording to {saver.start_episode()}")
            wait_timeout = min(interval, 0.05)
            if saver.active:
                wait_timeout = min(max(0.0, next_sample - clock()), 0.05)
            if not wait_for_input(wait_timeout):
                continue
            line = sys.stdin.readline()
            if line == "":
                if auto_start and duration > 0:
                    remaining = max(0.0, next_sample - clock())
                    if remaining > 0.0:
                        time.sleep(min(remaining, 0.05))
                    continue
                if saver.active:
                    print(f"Finalized {saver.stop_episode()}")
                return 0
            command = line.strip().lower()
            if command == "q":
                if saver.active:
                    print(f"Finalized {saver.stop_episode()}")
                return 0
            if command == "d":
                if saver.active:
                    discarded = saver.stop_episode(discard=True)
                    print(f"Discarded {discarded}")
                continue
            if command == "":
                if saver.active:
                    print(f"Finalized {saver.stop_episode()}")
                else:
                    print(f"Recording to {saver.start_episode()}")
    except KeyboardInterrupt:
        if saver.active:
            print(f"\nFinalized {saver.stop_episode()}")
        return 0
    finally:
        saver.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/i2rt_nedf3_saver.yaml", help="saver YAML path"
    )
    parser.add_argument("--output-dir", help="override the episode output directory")
    parser.add_argument("--task-name", help="override metadata task name")
    parser.add_argument("--task-description", help="override task description")
    parser.add_argument("--dm3-user-id", help="metadata.extra DM3 user ID")
    parser.add_argument("--dm3-user-name", help="metadata.extra DM3 user name")
    parser.add_argument(
        "--task-execution-id", help="metadata.extra DM3 task execution ID"
    )
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to record before stopping; requires --auto-start",
    )
    args = parser.parse_args(argv)
    if args.duration < 0 or (args.duration > 0 and not args.auto_start):
        parser.error("--duration must be non-negative and requires --auto-start")
    config = apply_runtime_overrides(
        load_config(args.config),
        output_dir=args.output_dir,
        task_name=args.task_name,
        task_description=args.task_description,
        dm3_user_id=args.dm3_user_id,
        dm3_user_name=args.dm3_user_name,
        task_execution_id=args.task_execution_id,
    )
    return run_terminal(
        I2RTNEDF3Saver(config),
        auto_start=args.auto_start,
        duration=args.duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DeviceFrame",
    "DeviceShmReader",
    "EpisodeSignalWatcher",
    "I2RTNEDF3Saver",
    "I2RTState",
    "SaverConfig",
    "arm_state_message",
    "apply_runtime_overrides",
    "hand_state_message",
    "load_config",
    "main",
    "metadata_message",
    "prepare_camera_image",
    "rgbd_message",
    "split_i2rt_state",
]
