#!/usr/bin/env python3
"""Convert SuperInference NEDF3 rollouts to LeRobot v2.1.

The inference saver and converter use the canonical R1 NEDF3 HAL layout:
``nmx/hal/arm``, ``hand``, ``head`` and ``camera`` protobuf topics.

The output follows LeRobot v2.1's chunked parquet/video layout. It does not
require the ``lerobot`` Python package; this avoids pulling a second Torch stack
into the hardware-facing recording environment. PyArrow and ffmpeg are
sufficient.

Example:

    python incubator/control_panel/tools/nedf3_rollout_to_lerobot.py \
        --data-dir /path/to/nedf3_rollouts \
        --dataset-name pi05_chip_0712 \
        --output-dir /path/to/lerobot \
        --head-camera WebCameraOpenPi_2 \
        --left-camera WebCameraOpenPi_0 \
        --right-camera WebCameraOpenPi_4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import multiprocessing
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from skimage.transform import resize as skimage_resize

from incubator.control_panel.recording_cameras import resolve_recorded_camera_ids
from incubator.control_panel.nedf3_io import (
    extract_gripper_width,
    extract_joint_state_positions,
    extract_pose_wxyz,
    extract_rgb_jpeg,
    message_timestamp_ns,
)
from incubator.control_panel.recording_contract import (
    LEROBOT_CORE_PARQUET_COLUMNS,
    LEROBOT_DEVICE_TYPE,
    LEROBOT_DUAL_ARM_FEATURE_NAMES,
    LEROBOT_EGO_FEATURE_NAMES,
    LEROBOT_EXTRA_FIELDS,
    LEROBOT_FEATURE_NAMES,
    LEROBOT_FPS,
    LEROBOT_INFO_FIELDS,
    LEROBOT_PARQUET_COLUMNS,
    LEROBOT_PARQUET_SCHEMA_METADATA,
    LEROBOT_TARGET_FPS,
    LEROBOT_VIDEO_FEATURES,
    NEDF3_ARM_TOPICS,
    NEDF3_DATA_FORMAT,
    NEDF3_HAND_TOPICS,
    NEDF3_HEAD_TOPICS,
    camera_topic_for_role,
    normalize_data_format,
)


LOGGER = logging.getLogger("nedf_rollout_to_lerobot")

CODEBASE_VERSION = "v2.1"
NMX_DATA_VERSION = "0.2.30"
TARGET_FPS = LEROBOT_TARGET_FPS
CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

LEFT_POSE_TOPIC = "/lowdim/robot_left"
LEFT_GRIPPER_TOPIC = "/lowdim/gripper_left"
RIGHT_POSE_TOPIC = "/lowdim/robot_right"
RIGHT_GRIPPER_TOPIC = "/lowdim/gripper_right"
DEFAULT_ACTION_TOPIC = "/lowdim/action/rizonrobot_2_control"

HEAD_CAMERA_FEATURE = LEROBOT_VIDEO_FEATURES["head"]
LEFT_CAMERA_FEATURE = LEROBOT_VIDEO_FEATURES["left"]
RIGHT_CAMERA_FEATURE = LEROBOT_VIDEO_FEATURES["right"]
CAMERA_FEATURES = (HEAD_CAMERA_FEATURE, LEFT_CAMERA_FEATURE, RIGHT_CAMERA_FEATURE)
DUAL_ARM_FEATURE_NAMES = LEROBOT_DUAL_ARM_FEATURE_NAMES
EGO_FEATURE_NAMES = LEROBOT_EGO_FEATURE_NAMES
FEATURE_NAMES = LEROBOT_FEATURE_NAMES
SOURCE_DIM = len(DUAL_ARM_FEATURE_NAMES)
OUTPUT_DIM = len(FEATURE_NAMES)
EGO_IDENTITY_POSE = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
QUATERNION_ORDER = "wxyz"
SOURCE_STATE_QUATERNION_ORDER = "wxyz"
SOURCE_ACTION_QUATERNION_ORDER = "xyzw"

TCP_POSE_MACROS: dict[str, Any] = {
    "r1_world": {
        "world_z_rotation_degrees": 180.0,
        "quaternion_order": "wxyz",
        "orientation_composition": "R_world_tcp = Rz_world(pi) @ R_base_tcp",
        "left": {"y_translation_after_rotation": 0.2},
        "right": {"y_translation_after_rotation": -0.2},
    }
}

EPISODE_PATTERN = re.compile(r"^episode_(\d+)$")
REQUIRED_TOPICS = (
    LEFT_POSE_TOPIC,
    LEFT_GRIPPER_TOPIC,
    RIGHT_POSE_TOPIC,
    RIGHT_GRIPPER_TOPIC,
)


class ConversionError(RuntimeError):
    """An episode is structurally invalid for conversion."""


@dataclass(frozen=True)
class TimedArray:
    timestamps_ns: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class EpisodePlan:
    source_dir: Path
    source_index: int
    result: str
    is_success: bool
    task: str
    head_camera: str | None
    left_camera: str
    right_camera: str
    action_topic: str
    data_format: str = NEDF3_DATA_FORMAT
    main_camera: str | None = None
    ego_required: bool = False
    source_episode_id: str | None = None


@dataclass
class AlignedEpisode:
    source_timestamps_ns: np.ndarray
    base_states: np.ndarray
    base_actions: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    camera_indices: dict[str, np.ndarray]
    alignment_error_ns: dict[str, np.ndarray]
    raw_hand_states: dict[str, np.ndarray] = field(default_factory=dict)
    aligned_timestamps_ns: dict[str, np.ndarray] = field(default_factory=dict)
    joint_positions: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EpisodeArtifact:
    episode_index: int
    episode_uuid: str
    source_index: int
    source_name: str
    result: str
    is_success: bool
    task: str
    length: int
    stats: dict[str, Any]
    alignment_summary: dict[str, Any]


@dataclass(frozen=True)
class VideoEncoderConfig:
    ffmpeg_bin: str
    ffprobe_bin: str
    encoder: str
    quality: int
    gop: int
    svt_preset: int
    nvenc_preset: str
    gpu_ids: tuple[int, ...]
    extra_output_args: tuple[str, ...]

    def gpu_for(self, job_ordinal: int, camera_ordinal: int) -> int | None:
        if self.encoder != "av1_nvenc" or not self.gpu_ids:
            return None
        slot = job_ordinal * len(CAMERA_FEATURES) + camera_ordinal
        return self.gpu_ids[slot % len(self.gpu_ids)]


def _stable_episode_uuid(plan: EpisodePlan, aligned: AlignedEpisode) -> str:
    if plan.source_episode_id:
        try:
            return str(uuid.UUID(plan.source_episode_id))
        except ValueError:
            pass
    timestamps = np.asarray(aligned.source_timestamps_ns, dtype=np.int64)
    if timestamps.size == 0:
        raise ConversionError(f"Cannot assign UUID to empty episode: {plan.source_dir}")
    identity_parts = [f"superinference-{plan.data_format.lower()}"]
    if plan.source_episode_id:
        identity_parts.append(plan.source_episode_id)
    identity_parts.extend(
        (
            plan.source_dir.name,
            str(plan.source_index),
            str(int(timestamps[0])),
            str(int(timestamps[-1])),
            str(int(timestamps.size)),
        )
    )
    identity = ":".join(identity_parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


@dataclass(frozen=True)
class EpisodeJob:
    ordinal: int
    bundle_root: Path
    plan: EpisodePlan
    wrist_image_size: tuple[int, int]
    head_image_size: tuple[int, int]
    crop_lrtb: tuple[int, int, int, int]
    tcp_pose_macro: str
    fps: int
    max_alignment_error_ns: int
    camera_workers: int
    encoder: VideoEncoderConfig


class ImageStats:
    """Streaming RGB statistics with the same shape convention as LeRobot."""

    def __init__(self) -> None:
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)
        self.total = np.zeros(3, dtype=np.float64)
        self.total_sq = np.zeros(3, dtype=np.float64)
        self.pixel_count = 0
        self.frame_count = 0

    def update(self, image_rgb: np.ndarray) -> None:
        sample = image_rgb
        max_hw = max(sample.shape[:2])
        if max_hw >= 300:
            stride = max(1, int(max_hw / 150))
            sample = sample[::stride, ::stride]
        values = sample.reshape(-1, 3).astype(np.float64) / 255.0
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.pixel_count += len(values)
        self.frame_count += 1

    def finalize(self) -> dict[str, Any]:
        if self.pixel_count == 0:
            raise ConversionError("No image samples were collected for statistics")
        mean = self.total / self.pixel_count
        variance = np.maximum(self.total_sq / self.pixel_count - np.square(mean), 0.0)

        def channel_shape(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(value)]] for value in values]

        return {
            "min": channel_shape(self.minimum),
            "max": channel_shape(self.maximum),
            "mean": channel_shape(mean),
            "std": channel_shape(np.sqrt(variance)),
            "count": [self.frame_count],
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConversionError(f"Cannot read JSONL {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversionError(
                f"Cannot read JSONL {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ConversionError(f"Expected an object in {path}:{line_number}")
        records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _discover_episode_dirs(data_dir: Path) -> list[tuple[int, Path]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"NEDF data directory does not exist: {data_dir}")
    if (data_dir / "metadata.json").is_file():
        match = EPISODE_PATTERN.match(data_dir.name)
        index = int(match.group(1)) if match else 0
        return [(index, data_dir)]

    episodes: list[tuple[int, Path]] = []
    for path in data_dir.iterdir():
        if not path.is_dir():
            continue
        match = EPISODE_PATTERN.match(path.name)
        if match and (path / "metadata.json").is_file():
            episodes.append((int(match.group(1)), path))
    return sorted(episodes)


def _message_timestamp_ns(message: Any) -> int:
    record = getattr(message, "message", message)
    for name in ("publish_time", "log_time", "timestamp_ns", "timestamp"):
        value = getattr(record, name, None)
        if value is not None:
            return int(value)
    raise ConversionError(f"NEDF message has no timestamp: {type(message)!r}")


def _decoded_lowdim_values(message: Any) -> np.ndarray:
    decoded = getattr(message, "decoded_message", message)
    values = getattr(decoded, "data", None)
    if values is None:
        raise ConversionError(f"NEDF lowdim message has no data field: {type(message)!r}")
    return np.asarray(values, dtype=np.float64)


def _iter_lowdim_messages(reader: Any, topic: str) -> Iterator[Any]:
    legacy_iterator = getattr(reader, "get_lowdim", None)
    if callable(legacy_iterator):
        yield from legacy_iterator(topic)
        return
    yield from reader.iter_decoded_messages(topic)


def _load_lowdim(reader: Any, topic: str, expected_dim: int) -> TimedArray:
    timestamps: list[int] = []
    values: list[np.ndarray] = []
    for message in _iter_lowdim_messages(reader, topic):
        value = _decoded_lowdim_values(message)
        if value.shape != (expected_dim,):
            raise ConversionError(
                f"Topic {topic} has shape {value.shape}; expected ({expected_dim},)"
            )
        timestamps.append(_message_timestamp_ns(message))
        values.append(value)
    if not values:
        raise ConversionError(f"Required topic has no messages: {topic}")
    timestamps_array = np.asarray(timestamps, dtype=np.int64)
    if np.any(np.diff(timestamps_array) < 0):
        order = np.argsort(timestamps_array, kind="stable")
        timestamps_array = timestamps_array[order]
        values = [values[index] for index in order]
    return TimedArray(timestamps_array, np.asarray(values, dtype=np.float64))


def _timed_array(
    timestamps: Sequence[int],
    values: Sequence[np.ndarray | Sequence[float] | float],
    *,
    topic: str,
) -> TimedArray:
    if not values:
        raise ConversionError(f"Required topic has no messages: {topic}")
    timestamps_array = np.asarray(timestamps, dtype=np.int64)
    values_array = np.asarray(values, dtype=np.float64)
    if np.any(np.diff(timestamps_array) < 0):
        order = np.argsort(timestamps_array, kind="stable")
        timestamps_array = timestamps_array[order]
        values_array = values_array[order]
    return TimedArray(timestamps_array, values_array)


def _reader_for_format(source_dir: Path, data_format: str) -> Any:
    from dm3_core.nedf import NEDFFactory, NEDFReaderConfig

    config = NEDFReaderConfig(
        metadata_file_path=str(source_dir / "metadata.json")
    )
    unified_reader = getattr(NEDFFactory, "get_reader", None)
    if callable(unified_reader):
        return unified_reader(config)
    raise ConversionError(f"Installed NEDF SDK cannot read {data_format}")


def _camera_topic(camera_id: str, data_format: str) -> str:
    normalize_data_format(data_format)
    return f"nmx/hal/camera/{camera_id}/rgbd"


def _camera_timestamps(reader: Any, camera_id: str, data_format: str) -> np.ndarray:
    values = reader.get_raw_topics_timestamps().get(
        _camera_topic(camera_id, data_format), ()
    )
    return np.asarray(values, dtype=np.int64)


def _iter_camera_messages(
    reader: Any, camera_id: str, data_format: str
) -> Iterator[Any]:
    yield from reader.iter_decoded_messages(_camera_topic(camera_id, data_format))


def _load_nedf3_pose(reader: Any, side: str, kind: str = "state") -> TimedArray:
    topic = NEDF3_ARM_TOPICS[side][f"tcp/{kind}"]
    timestamps: list[int] = []
    values: list[np.ndarray] = []
    for record in reader.iter_decoded_messages(topic):
        timestamps.append(message_timestamp_ns(record))
        values.append(extract_pose_wxyz(record))
    return _timed_array(timestamps, values, topic=topic)


def _load_nedf3_hand_state(
    reader: Any, side: str, kind: str = "state"
) -> TimedArray:
    topic = NEDF3_HAND_TOPICS[side][kind]
    timestamps: list[int] = []
    values: list[list[float]] = []
    for record in reader.iter_decoded_messages(topic):
        width = extract_gripper_width(record)
        if width is None:
            continue
        timestamps.append(message_timestamp_ns(record))
        values.append([width])
    return _timed_array(timestamps, values, topic=topic)


def _load_nedf3_optional_joints(
    reader: Any, side: str, expected_dimension: int
) -> TimedArray | None:
    topic = (
        NEDF3_HEAD_TOPICS["joint/state"]
        if side == "ego"
        else NEDF3_ARM_TOPICS[side]["joint/state"]
    )
    if not reader.get_raw_topics_timestamps().get(topic):
        return None
    timestamps: list[int] = []
    values: list[np.ndarray] = []
    for record in reader.iter_decoded_messages(topic):
        joints = extract_joint_state_positions(
            record, expected_dimension=expected_dimension
        )
        timestamps.append(message_timestamp_ns(record))
        values.append(joints)
    if not values:
        return None
    return _timed_array(timestamps, values, topic=topic)


def _load_nedf3_head_pose(reader: Any, kind: str = "state") -> TimedArray:
    topic = NEDF3_HEAD_TOPICS[kind]
    timestamps: list[int] = []
    values: list[np.ndarray] = []
    for record in reader.iter_decoded_messages(topic):
        timestamps.append(message_timestamp_ns(record))
        values.append(extract_pose_wxyz(record))
    return _timed_array(timestamps, values, topic=topic)


def _nearest_indices(
    source_timestamps_ns: np.ndarray, target_timestamps_ns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if source_timestamps_ns.size == 0:
        raise ConversionError("Cannot align against an empty source timeline")
    insertion = np.searchsorted(source_timestamps_ns, target_timestamps_ns)
    right = np.clip(insertion, 0, len(source_timestamps_ns) - 1)
    left = np.clip(insertion - 1, 0, len(source_timestamps_ns) - 1)
    right_error = np.abs(source_timestamps_ns[right] - target_timestamps_ns)
    left_error = np.abs(source_timestamps_ns[left] - target_timestamps_ns)
    choose_right = right_error < left_error
    indices = np.where(choose_right, right, left)
    errors = np.abs(source_timestamps_ns[indices] - target_timestamps_ns)
    return indices.astype(np.int64), errors.astype(np.int64)


def _resolve_action_topic(reader: Any, requested_topic: str) -> str:
    topic_timestamps = reader.get_raw_topics_timestamps()
    if topic_timestamps.get(requested_topic):
        return requested_topic
    candidates = sorted(
        topic
        for topic, timestamps in topic_timestamps.items()
        if topic.startswith("/lowdim/action/")
        and "chunk" not in topic.lower()
        and timestamps
    )
    for candidate in candidates:
        try:
            first = next(_iter_lowdim_messages(reader, candidate))
        except StopIteration:
            continue
        if _decoded_lowdim_values(first).shape == (SOURCE_DIM,):
            LOGGER.warning(
                "Action topic %s is absent; inferred %s", requested_topic, candidate
            )
            return candidate
    raise ConversionError(
        f"Executed 16D action topic not found (requested {requested_topic!r})"
    )


def _resolve_cameras(
    metadata: dict[str, Any],
    head_camera: str | None,
    left_camera: str | None,
    right_camera: str | None,
    data_format: str = NEDF3_DATA_FORMAT,
) -> tuple[str | None, str, str]:
    camera_info = metadata.get("camera_info") or {}
    if not isinstance(camera_info, dict):
        raise ConversionError("metadata.camera_info must be an object")
    camera_ids = list(camera_info)
    requested = {
        "head": head_camera,
        "left": left_camera,
        "right": right_camera,
    }
    resolved = resolve_recorded_camera_ids(
        camera_info,
        requested,
        data_format=data_format,
    )
    resolved_head = None if head_camera is None else resolved["head"]
    resolved_left = resolved["left"]
    resolved_right = resolved["right"]
    if not resolved_left:
        raise ConversionError(
            "Left wrist camera is absent from metadata.camera_info"
        )
    if not resolved_right:
        raise ConversionError(
            "Right wrist camera is absent from metadata.camera_info"
        )
    if isinstance(head_camera, str) and not resolved_head:
        raise ConversionError(
            "Head camera must be a non-empty id or None for a black placeholder"
        )
    resolved_ids = [
        camera_id
        for camera_id in (resolved_head, resolved_left, resolved_right)
        if camera_id is not None
    ]
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ConversionError(f"Head/left/right camera ids must be unique: {resolved_ids}")
    LOGGER.info(
        "Resolved NEDF cameras head=%s left=%s right=%s from %s",
        resolved_head,
        resolved_left,
        resolved_right,
        camera_ids,
    )
    return resolved_head, resolved_left, resolved_right


def _build_plan(
    source_index: int,
    source_dir: Path,
    allowed_results: set[str],
    head_camera: str | None,
    left_camera: str | None,
    right_camera: str | None,
    requested_action_topic: str,
) -> EpisodePlan:
    metadata = _read_json(source_dir / "metadata.json")
    try:
        data_format = normalize_data_format(metadata.get("data_format"))
    except ValueError as exc:
        raise ConversionError(str(exc)) from exc
    result_path = source_dir / "episode_result.json"
    if not result_path.is_file():
        result_path = source_dir / "episode_incomplete.json"
    if not result_path.is_file():
        raise ConversionError("episode result metadata is missing")
    result_metadata = _read_json(result_path)
    result = str(result_metadata.get("episode_result", "")).lower()
    if result not in allowed_results:
        raise ConversionError(f"episode result {result!r} is excluded")

    reader = _reader_for_format(source_dir, data_format)
    topics = reader.get_raw_topics_timestamps()
    if data_format == NEDF3_DATA_FORMAT:
        required_topics = (
            NEDF3_ARM_TOPICS["left"]["tcp/state"],
            NEDF3_ARM_TOPICS["left"]["tcp/cmd"],
            NEDF3_ARM_TOPICS["left"]["joint/state"],
            NEDF3_HAND_TOPICS["left"]["state"],
            NEDF3_HAND_TOPICS["left"]["cmd"],
            NEDF3_ARM_TOPICS["right"]["tcp/state"],
            NEDF3_ARM_TOPICS["right"]["tcp/cmd"],
            NEDF3_ARM_TOPICS["right"]["joint/state"],
            NEDF3_HAND_TOPICS["right"]["state"],
            NEDF3_HAND_TOPICS["right"]["cmd"],
        )
        missing_topics = [topic for topic in required_topics if not topics.get(topic)]
        resolved_action_topic = "nmx/hal/arm/{side}/tcp/cmd"
    else:
        missing_topics = [topic for topic in REQUIRED_TOPICS if not topics.get(topic)]
        resolved_action_topic = _resolve_action_topic(reader, requested_action_topic)
    if missing_topics:
        raise ConversionError(f"required topics are missing: {missing_topics}")
    resolved_head, resolved_left, resolved_right = _resolve_cameras(
        metadata,
        head_camera,
        left_camera,
        right_camera,
        data_format,
    )
    for camera_id in (resolved_head, resolved_left, resolved_right):
        if camera_id is None:
            continue
        if _camera_timestamps(reader, camera_id, data_format).size == 0:
            raise ConversionError(f"camera has no images: {camera_id}")
    extra = metadata.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    ego_required = bool(extra.get("ego_pose_required"))
    if ego_required and not topics.get(NEDF3_HEAD_TOPICS["state"]):
        raise ConversionError(
            "metadata requires a measured head state but the NEDF3 head/state topic is empty"
        )
    main_camera_raw = metadata.get("main_camera")
    main_camera = (
        str(main_camera_raw).strip() if main_camera_raw is not None else ""
    ) or None
    resolved_ids = {resolved_head, resolved_left, resolved_right}
    if data_format == NEDF3_DATA_FORMAT and (
        main_camera is None or main_camera not in resolved_ids
    ):
        raise ConversionError(
            f"metadata.main_camera={main_camera!r} is not an exported camera"
        )
    task = str(metadata.get("task_name") or "inference rollout").strip()
    source_episode_id = str(result_metadata.get("episode_id") or "").strip() or None
    return EpisodePlan(
        source_dir=source_dir,
        source_index=source_index,
        result=result,
        is_success=bool(result_metadata.get("is_success", result == "success")),
        task=task,
        head_camera=resolved_head,
        left_camera=resolved_left,
        right_camera=resolved_right,
        action_topic=resolved_action_topic,
        data_format=data_format,
        main_camera=main_camera,
        ego_required=ego_required,
        source_episode_id=source_episode_id,
    )


def _align_nedf3_episode(
    reader: Any,
    plan: EpisodePlan,
    max_alignment_error_ns: int,
    tcp_pose_macro: str,
) -> AlignedEpisode:
    streams: dict[str, TimedArray] = {
        "left_pose": _load_nedf3_pose(reader, "left"),
        "left_gripper": _load_nedf3_hand_state(reader, "left"),
        "right_pose": _load_nedf3_pose(reader, "right"),
        "right_gripper": _load_nedf3_hand_state(reader, "right"),
        "left_action_pose": _load_nedf3_pose(reader, "left", "cmd"),
        "left_action_gripper": _load_nedf3_hand_state(reader, "left", "cmd"),
        "right_action_pose": _load_nedf3_pose(reader, "right", "cmd"),
        "right_action_gripper": _load_nedf3_hand_state(reader, "right", "cmd"),
    }
    raw_topics = reader.get_raw_topics_timestamps()
    ego_topic = NEDF3_HEAD_TOPICS["state"]
    ego_available = bool(raw_topics.get(ego_topic))
    if ego_available:
        streams["ego_pose"] = _load_nedf3_head_pose(reader, "state")
        if raw_topics.get(NEDF3_HEAD_TOPICS["cmd"]):
            streams["ego_action_pose"] = _load_nedf3_head_pose(reader, "cmd")
    elif plan.ego_required:
        raise ConversionError("required NEDF3 ego pose stream is empty")

    camera_ids = {
        HEAD_CAMERA_FEATURE: plan.head_camera,
        LEFT_CAMERA_FEATURE: plan.left_camera,
        RIGHT_CAMERA_FEATURE: plan.right_camera,
    }
    camera_timestamps = {
        feature: _camera_timestamps(reader, camera_id, plan.data_format)
        for feature, camera_id in camera_ids.items()
        if camera_id is not None
    }
    target = streams["left_action_pose"].timestamps_ns.copy()
    all_timelines = [stream.timestamps_ns for stream in streams.values()]
    all_timelines.extend(camera_timestamps.values())
    overlap_start = max(int(timeline[0]) for timeline in all_timelines)
    overlap_end = min(int(timeline[-1]) for timeline in all_timelines)
    target = target[(target >= overlap_start) & (target <= overlap_end)]
    if len(target) < 2:
        raise ConversionError("fewer than two NEDF3 main-camera frames remain in overlap")

    indices: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    for name, stream in streams.items():
        indices[name], errors[name] = _nearest_indices(stream.timestamps_ns, target)
    camera_indices: dict[str, np.ndarray] = {}
    for feature, timestamps in camera_timestamps.items():
        camera_indices[feature], errors[feature] = _nearest_indices(timestamps, target)

    valid = np.ones(len(target), dtype=bool)
    for stream_errors in errors.values():
        valid &= stream_errors <= max_alignment_error_ns
    if not np.all(valid):
        dropped = int((~valid).sum())
        LOGGER.warning(
            "%s: dropping %d/%d NEDF3 frames above %.1f ms alignment error",
            plan.source_dir.name,
            dropped,
            len(valid),
            max_alignment_error_ns / 1e6,
        )
        target = target[valid]
        indices = {name: value[valid] for name, value in indices.items()}
        camera_indices = {
            name: value[valid] for name, value in camera_indices.items()
        }
        errors = {name: value[valid] for name, value in errors.items()}
    if len(target) < 2:
        raise ConversionError("fewer than two valid NEDF3 aligned frames remain")

    left_pose = streams["left_pose"].values[indices["left_pose"]]
    left_gripper = streams["left_gripper"].values[indices["left_gripper"]]
    right_pose = streams["right_pose"].values[indices["right_pose"]]
    right_gripper = streams["right_gripper"].values[indices["right_gripper"]]
    if ego_available:
        ego_pose = streams["ego_pose"].values[indices["ego_pose"]]
    else:
        ego_pose = np.broadcast_to(EGO_IDENTITY_POSE, (len(target), 7))
    raw_states = np.concatenate(
        [left_pose, left_gripper, right_pose, right_gripper, ego_pose], axis=1
    ).astype(np.float32)
    world_dual = _apply_tcp_pose_macro(raw_states[:, :SOURCE_DIM], tcp_pose_macro)
    world_states = np.concatenate([world_dual, raw_states[:, SOURCE_DIM:]], axis=1)
    _validate_pose_vectors(world_states, "NEDF3 aligned state")

    raw_dual_actions = np.concatenate(
        [
            streams["left_action_pose"].values[indices["left_action_pose"]],
            streams["left_action_gripper"].values[indices["left_action_gripper"]],
            streams["right_action_pose"].values[indices["right_action_pose"]],
            streams["right_action_gripper"].values[indices["right_action_gripper"]],
        ],
        axis=1,
    ).astype(np.float32)
    if "ego_action_pose" in streams:
        ego_action = streams["ego_action_pose"].values[indices["ego_action_pose"]]
    else:
        ego_action = np.broadcast_to(EGO_IDENTITY_POSE, (len(target), 7))
    raw_actions = np.concatenate([raw_dual_actions, ego_action], axis=1)
    world_action_dual = _apply_tcp_pose_macro(raw_dual_actions, tcp_pose_macro)
    world_actions = np.concatenate([world_action_dual, ego_action], axis=1)
    _validate_pose_vectors(world_actions, "NEDF3 aligned action")

    aligned_timestamps_ns = {
        "left": streams["left_pose"].timestamps_ns[indices["left_pose"]],
        "right": streams["right_pose"].timestamps_ns[indices["right_pose"]],
        "ego": (
            streams["ego_pose"].timestamps_ns[indices["ego_pose"]]
            if ego_available
            else target
        ),
    }
    joint_positions: dict[str, np.ndarray] = {}
    for side, dimension in (("left", 7), ("right", 7), ("ego", 6)):
        stream = _load_nedf3_optional_joints(reader, side, dimension)
        if stream is None:
            continue
        stream_indices, _ = _nearest_indices(stream.timestamps_ns, target)
        joint_positions[side] = np.asarray(
            stream.values[stream_indices], dtype=np.float32
        )

    return AlignedEpisode(
        source_timestamps_ns=target,
        base_states=raw_states,
        base_actions=raw_actions,
        states=world_states.astype(np.float32),
        actions=world_actions.astype(np.float32),
        camera_indices=camera_indices,
        alignment_error_ns=errors,
        aligned_timestamps_ns=aligned_timestamps_ns,
        joint_positions=joint_positions,
    )


def _align_episode(
    reader: Any,
    plan: EpisodePlan,
    max_alignment_error_ns: int,
    tcp_pose_macro: str,
) -> AlignedEpisode:
    if plan.data_format == NEDF3_DATA_FORMAT:
        return _align_nedf3_episode(
            reader,
            plan,
            max_alignment_error_ns,
            tcp_pose_macro,
        )
    streams = {
        "left_pose": _load_lowdim(reader, LEFT_POSE_TOPIC, 7),
        "left_gripper": _load_lowdim(reader, LEFT_GRIPPER_TOPIC, 1),
        "right_pose": _load_lowdim(reader, RIGHT_POSE_TOPIC, 7),
        "right_gripper": _load_lowdim(reader, RIGHT_GRIPPER_TOPIC, 1),
        "action": _load_lowdim(reader, plan.action_topic, SOURCE_DIM),
    }
    camera_timestamps = {
        LEFT_CAMERA_FEATURE: _camera_timestamps(
            reader, plan.left_camera, plan.data_format
        ),
        RIGHT_CAMERA_FEATURE: _camera_timestamps(
            reader, plan.right_camera, plan.data_format
        ),
    }
    if plan.head_camera is not None:
        camera_timestamps[HEAD_CAMERA_FEATURE] = _camera_timestamps(
            reader, plan.head_camera, plan.data_format
        )
    target = streams["action"].timestamps_ns
    all_timelines = [stream.timestamps_ns for stream in streams.values()]
    all_timelines.extend(camera_timestamps.values())
    overlap_start = max(int(timeline[0]) for timeline in all_timelines)
    overlap_end = min(int(timeline[-1]) for timeline in all_timelines)
    target = target[(target >= overlap_start) & (target <= overlap_end)]
    if len(target) < 2:
        raise ConversionError("fewer than two executed actions remain in common overlap")

    indices: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    for name, stream in streams.items():
        indices[name], errors[name] = _nearest_indices(stream.timestamps_ns, target)
    camera_indices: dict[str, np.ndarray] = {}
    for feature, timestamps in camera_timestamps.items():
        camera_indices[feature], errors[feature] = _nearest_indices(timestamps, target)

    valid = np.ones(len(target), dtype=bool)
    for stream_errors in errors.values():
        valid &= stream_errors <= max_alignment_error_ns
    if not np.all(valid):
        dropped = int((~valid).sum())
        LOGGER.warning(
            "%s: dropping %d/%d frames above %.1f ms alignment error",
            plan.source_dir.name,
            dropped,
            len(valid),
            max_alignment_error_ns / 1e6,
        )
        target = target[valid]
        for name in indices:
            indices[name] = indices[name][valid]
        for feature in camera_indices:
            camera_indices[feature] = camera_indices[feature][valid]
        for name in errors:
            errors[name] = errors[name][valid]
    if len(target) < 2:
        raise ConversionError("fewer than two valid aligned frames remain")

    source_states = np.concatenate(
        [
            streams["left_pose"].values[indices["left_pose"]],
            streams["left_gripper"].values[indices["left_gripper"]],
            streams["right_pose"].values[indices["right_pose"]],
            streams["right_gripper"].values[indices["right_gripper"]],
        ],
        axis=1,
    )
    base_states = source_states
    source_actions = streams["action"].values[indices["action"]]
    base_actions = _convert_action_quaternion_order(source_actions)
    world_states = _apply_tcp_pose_macro(base_states, tcp_pose_macro)
    world_actions = _apply_tcp_pose_macro(base_actions, tcp_pose_macro)
    states = _append_ego_identity(world_states.astype(np.float32))
    actions = _append_ego_identity(world_actions.astype(np.float32))
    _validate_pose_vectors(states, "state")
    _validate_pose_vectors(actions, "action")
    return AlignedEpisode(
        target,
        base_states.astype(np.float32),
        base_actions.astype(np.float32),
        states,
        actions,
        camera_indices,
        errors,
    )


def _convert_action_quaternion_order(values: np.ndarray) -> np.ndarray:
    """Convert control-SHM action quaternions from xyzw to dataset wxyz."""
    transformed = np.asarray(values).copy()
    for block_start in (0, 8):
        quat = np.asarray(values)[:, block_start + 3 : block_start + 7]
        transformed[:, block_start + 3] = quat[:, 3]
        transformed[:, block_start + 4] = quat[:, 0]
        transformed[:, block_start + 5] = quat[:, 1]
        transformed[:, block_start + 6] = quat[:, 2]
    return transformed


def _apply_tcp_pose_macro(values: np.ndarray, macro_name: str) -> np.ndarray:
    """Transform dual-arm TCP poses into a shared world frame."""
    if macro_name == "none":
        return np.asarray(values).copy()
    macro = TCP_POSE_MACROS.get(macro_name)
    if macro is None:
        raise ConversionError(f"Unknown TCP pose macro: {macro_name}")
    if macro.get("world_z_rotation_degrees") != 180.0:
        raise ConversionError(
            f"Unsupported world z rotation in TCP macro {macro_name}: "
            f"{macro.get('world_z_rotation_degrees')}"
        )
    transformed = np.asarray(values).copy()
    for arm_name, block_start in (("left", 0), ("right", 8)):
        config = macro[arm_name]
        original = np.asarray(values)[:, block_start : block_start + 8]

        # Rotate position by Rz(pi), then translate along the world y axis.
        transformed[:, block_start] = -original[:, 0]
        transformed[:, block_start + 1] = (
            -original[:, 1] + config["y_translation_after_rotation"]
        )

        # Left-multiply orientation by q_z180=[0,0,0,1] in wxyz order:
        # q_world_tcp = q_z180 * q_base_tcp.
        quat = original[:, 3:7]
        transformed[:, block_start + 3] = -quat[:, 3]
        transformed[:, block_start + 4] = -quat[:, 2]
        transformed[:, block_start + 5] = quat[:, 1]
        transformed[:, block_start + 6] = quat[:, 0]
    return transformed


def _validate_pose_vectors(values: np.ndarray, name: str) -> None:
    if values.ndim != 2 or values.shape[1] != OUTPUT_DIM:
        raise ConversionError(f"{name} has invalid shape {values.shape}")
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0].tolist()
        raise ConversionError(f"{name} contains a non-finite value at {bad}")
    for start in (3, 11, 19):
        norms = np.linalg.norm(values[:, start : start + 4], axis=1)
        if np.any((norms < 0.5) | (norms > 1.5)):
            bad_index = int(np.flatnonzero((norms < 0.5) | (norms > 1.5))[0])
            raise ConversionError(
                f"{name} quaternion norm {norms[bad_index]:.4f} at frame {bad_index}"
            )


def _append_ego_identity(values: np.ndarray) -> np.ndarray:
    """Pad a 16D dual-arm vector to the raw 23D OpenPi dataset layout.

    Some inference configurations have no ego/head pose. OpenPi's existing UMI dataset transform
    expects the R1-style 23D on-disk schema and unconditionally drops dimensions
    16:23, so an identity pose is an explicit compatibility placeholder.
    """
    if values.ndim != 2 or values.shape[1] != SOURCE_DIM:
        raise ConversionError(f"Cannot pad source array with shape {values.shape}")
    padding = np.broadcast_to(EGO_IDENTITY_POSE, (len(values), len(EGO_IDENTITY_POSE)))
    return np.concatenate([values, padding], axis=1).astype(np.float32, copy=False)


def _sample_indices(length: int) -> set[int]:
    sample_count = min(max(min(100, length), min(int(length**0.75), 10_000)), length)
    return set(np.round(np.linspace(0, length - 1, sample_count)).astype(int).tolist())


def _iter_selected_images(
    reader: Any,
    camera_id: str,
    selected_source_indices: np.ndarray,
    data_format: str = NEDF3_DATA_FORMAT,
) -> Iterator[np.ndarray]:
    if np.any(np.diff(selected_source_indices) < 0):
        raise ConversionError("Selected camera indices are not monotonic")
    output_index = 0
    selected_count = len(selected_source_indices)
    for source_index, message in enumerate(
        _iter_camera_messages(reader, camera_id, data_format)
    ):
        if output_index >= selected_count:
            break
        selected_index = int(selected_source_indices[output_index])
        if source_index < selected_index:
            continue
        if source_index > selected_index:
            raise ConversionError(
                f"Camera iterator skipped source frame {selected_index} for {camera_id}"
            )
        decoded = getattr(message, "decoded_message", message)
        if data_format == NEDF3_DATA_FORMAT:
            encoded = extract_rgb_jpeg(decoded)
        else:
            encoded = getattr(decoded, "data", None)
        if not encoded:
            raise ConversionError(f"Empty JPEG payload for camera {camera_id}")
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ConversionError(f"Failed to decode JPEG for camera {camera_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        while (
            output_index < selected_count
            and int(selected_source_indices[output_index]) == source_index
        ):
            yield rgb
            output_index += 1
    if output_index != selected_count:
        raise ConversionError(
            f"Camera {camera_id} produced {output_index}/{selected_count} aligned frames"
        )


def _encoder_output_args(
    encoder: VideoEncoderConfig, gpu_id: int | None
) -> list[str]:
    if encoder.encoder == "libsvtav1":
        output_args = [
            "-c:v",
            "libsvtav1",
            "-preset",
            str(encoder.svt_preset),
            "-g",
            str(encoder.gop),
            "-crf",
            str(encoder.quality),
            "-pix_fmt",
            "yuv420p",
        ]
    elif encoder.encoder == "av1_nvenc":
        output_args = [
            "-c:v",
            "av1_nvenc",
            "-preset",
            encoder.nvenc_preset,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(encoder.quality),
            "-b:v",
            "0",
            "-g",
            str(encoder.gop),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
        if gpu_id is not None:
            output_args.extend(["-gpu", str(gpu_id)])
    else:
        raise ConversionError(f"Unsupported video encoder: {encoder.encoder}")
    output_args.extend(encoder.extra_output_args)
    return output_args


def _ffmpeg_command(
    video_path: Path,
    width: int,
    height: int,
    fps: int,
    encoder: VideoEncoderConfig,
    gpu_id: int | None,
) -> list[str]:
    command = [
        encoder.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
    ]
    command.extend(_encoder_output_args(encoder, gpu_id))
    command.append(str(video_path))
    return command


def _available_ffmpeg_encoders(ffmpeg_bin: str) -> set[str]:
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConversionError(f"Cannot inspect ffmpeg encoders via {ffmpeg_bin!r}: {exc}") from exc
    return {
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.match(r"^\s*[A-Z.]{6}\s+(\S+)", line))
    }


def _smoke_test_encoder(
    encoder: VideoEncoderConfig, gpu_id: int | None
) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="nedf-encoder-smoke-") as directory:
        video_path = Path(directory) / "smoke.mp4"
        command = [
            encoder.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=30",
            "-frames:v",
            "1",
            "-an",
        ]
        command.extend(_encoder_output_args(encoder, gpu_id))
        command.append(str(video_path))
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                return False, result.stderr.strip()
            _validate_video(
                video_path,
                expected_length=1,
                image_size=(240, 320),
                fps=30,
                ffprobe_bin=encoder.ffprobe_bin,
            )
        except (OSError, subprocess.SubprocessError, ConversionError) as exc:
            return False, str(exc)
    return True, ""


def _resolve_video_encoder(args: argparse.Namespace) -> VideoEncoderConfig:
    try:
        subprocess.run(
            [args.ffprobe_bin, "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConversionError(f"Cannot execute ffprobe via {args.ffprobe_bin!r}: {exc}") from exc

    available = _available_ffmpeg_encoders(args.ffmpeg_bin)
    requested = args.video_encoder
    candidates = ("av1_nvenc", "libsvtav1") if requested == "auto" else (requested,)
    failures: list[str] = []
    for candidate in candidates:
        if candidate not in available:
            failures.append(f"{candidate}: encoder is absent from {args.ffmpeg_bin}")
            continue
        encoder = VideoEncoderConfig(
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            encoder=candidate,
            quality=args.video_quality,
            gop=args.video_gop,
            svt_preset=args.svt_preset,
            nvenc_preset=args.nvenc_preset,
            gpu_ids=args.gpu_ids if candidate == "av1_nvenc" else (),
            extra_output_args=args.ffmpeg_extra_args,
        )
        devices: tuple[int | None, ...]
        if candidate == "av1_nvenc" and encoder.gpu_ids:
            devices = tuple(encoder.gpu_ids)
        else:
            devices = (None,)
        candidate_failures = []
        if len(devices) == 1:
            probe_results = [(devices[0], *_smoke_test_encoder(encoder, devices[0]))]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(devices), 8),
                thread_name_prefix="encoder-probe",
            ) as executor:
                results = executor.map(
                    lambda gpu_id: _smoke_test_encoder(encoder, gpu_id), devices
                )
                probe_results = [
                    (gpu_id, ok, detail)
                    for gpu_id, (ok, detail) in zip(devices, results)
                ]
        for gpu_id, ok, detail in probe_results:
            if not ok:
                label = "default GPU" if gpu_id is None else f"GPU {gpu_id}"
                detail_lines = [line.strip() for line in detail.splitlines() if line.strip()]
                if not detail_lines:
                    detail_summary = "initialization failed"
                elif len(detail_lines) == 1:
                    detail_summary = detail_lines[0]
                else:
                    detail_summary = f"{detail_lines[0]} | {detail_lines[-1]}"
                candidate_failures.append(f"{label}: {detail_summary}")
        if not candidate_failures:
            if requested == "auto":
                LOGGER.info("Auto-selected ffmpeg video encoder: %s", candidate)
            return encoder
        failures.append(f"{candidate}: {'; '.join(candidate_failures)}")
        if requested == "auto":
            LOGGER.warning("Cannot use %s; trying fallback: %s", candidate, candidate_failures)
    raise ConversionError(
        f"No usable AV1 encoder for --video-encoder {requested!r}: "
        + " | ".join(failures)
    )


def _write_video(
    reader: Any,
    camera_id: str,
    selected_source_indices: np.ndarray,
    video_path: Path,
    image_size: tuple[int, int],
    crop_lrtb: tuple[int, int, int, int],
    fps: int,
    encoder: VideoEncoderConfig,
    gpu_id: int | None,
    data_format: str = NEDF3_DATA_FORMAT,
) -> dict[str, Any]:
    height, width = image_size
    video_path.parent.mkdir(parents=True, exist_ok=True)
    stats = ImageStats()
    stats_indices = _sample_indices(len(selected_source_indices))
    process = subprocess.Popen(
        _ffmpeg_command(video_path, width, height, fps, encoder, gpu_id),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    frame_count = 0
    try:
        for frame_index, image in enumerate(
            _iter_selected_images(
                reader,
                camera_id,
                selected_source_indices,
                data_format,
            )
        ):
            resized = _crop_resize_image(image, crop_lrtb, image_size)
            if frame_index in stats_indices:
                stats.update(resized)
            process.stdin.write(np.ascontiguousarray(resized).tobytes())
            frame_count += 1
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise ConversionError(
            f"ffmpeg failed for {video_path}: {stderr.decode(errors='replace').strip()}"
        )
    if frame_count != len(selected_source_indices):
        raise ConversionError(
            f"Video frame count mismatch before encode: {frame_count} != "
            f"{len(selected_source_indices)}"
        )
    return stats.finalize()


def _write_black_video(
    video_path: Path,
    frame_count: int,
    image_size: tuple[int, int],
    fps: int,
    encoder: VideoEncoderConfig,
    gpu_id: int | None,
) -> dict[str, Any]:
    """Encode a deterministic black placeholder for a missing camera stream."""
    height, width = image_size
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        encoder.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={fps}",
        "-frames:v",
        str(frame_count),
        "-an",
    ]
    command.extend(_encoder_output_args(encoder, gpu_id))
    command.append(str(video_path))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConversionError(
            f"ffmpeg failed for black video {video_path}: {result.stderr.strip()}"
        )
    sample_count = len(_sample_indices(frame_count))
    zeros = [[[0.0]], [[0.0]], [[0.0]]]
    return {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": zeros,
        "count": [sample_count],
    }


def _crop_resize_image(
    image_rgb: np.ndarray,
    crop_lrtb: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Apply the same crop/resize geometry as OpenPiUmiPolicy."""
    left, right, top, bottom = crop_lrtb
    height, width = image_rgb.shape[:2]
    if min(crop_lrtb) < 0 or left + right >= width or top + bottom >= height:
        raise ConversionError(
            f"Invalid crop {crop_lrtb} for source image shape {image_rgb.shape}"
        )
    cropped = image_rgb[top : height - bottom, left : width - right]
    output_height, output_width = image_size
    resized = skimage_resize(
        cropped,
        (output_height, output_width),
        preserve_range=True,
        anti_aliasing=True,
    )
    return resized.astype(np.uint8)


def _numeric_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).astype(np.float64).tolist(),
        "max": array.max(axis=0).astype(np.float64).tolist(),
        "mean": array.mean(axis=0).astype(np.float64).tolist(),
        "std": array.std(axis=0).astype(np.float64).tolist(),
        "count": [len(array)],
    }


def _fixed_float_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _extra_parquet_columns(aligned: AlignedEpisode) -> dict[str, pa.Array]:
    """Build the exact NMX LeRobot v2.1 extra-column contract."""
    length = len(aligned.source_timestamps_ns)
    columns: dict[str, pa.Array] = {}
    for side, block_start in (("left", 0), ("right", 8), ("ego", 16)):
        prefix = f"observation.extra.{side}"
        raw_source = (
            aligned.base_states
            if aligned.base_states.ndim == 2
            and aligned.base_states.shape[1] >= block_start + 7
            else aligned.states
        )
        columns[f"{prefix}.raw_pose"] = _fixed_float_list(
            raw_source[:, block_start : block_start + 7], 7
        )
        if side == "ego":
            columns[f"{prefix}.hand_state"] = pa.nulls(length, type=pa.float32())
        else:
            columns[f"{prefix}.hand_state"] = pa.array(
                aligned.states[:, block_start + 7], type=pa.float32()
            )
        raw_hand_state = aligned.raw_hand_states.get(side)
        if raw_hand_state is None:
            columns[f"{prefix}.raw_hand_state"] = pa.nulls(
                length, type=pa.float32()
            )
        else:
            raw_values = np.asarray(raw_hand_state, dtype=np.float32).reshape(-1)
            if raw_values.shape != (length,):
                raise ConversionError(
                    f"{side} raw hand state has shape {raw_values.shape}; "
                    f"expected ({length},)"
                )
            columns[f"{prefix}.raw_hand_state"] = pa.array(
                raw_values,
                mask=~np.isfinite(raw_values),
                type=pa.float32(),
            )
        columns[f"{prefix}.roi_marker_angle"] = pa.nulls(length, type=pa.float32())
        aligned_timestamp = aligned.aligned_timestamps_ns.get(side)
        if aligned_timestamp is None:
            aligned_timestamp = aligned.source_timestamps_ns
        aligned_timestamp = np.asarray(aligned_timestamp, dtype=np.float64)
        if aligned_timestamp.shape != (length,):
            raise ConversionError(
                f"{side} aligned timestamps have shape {aligned_timestamp.shape}; "
                f"expected ({length},)"
            )
        columns[f"{prefix}.aligned_timestamp_ns"] = pa.array(
            aligned_timestamp, type=pa.float64()
        )
    if tuple(columns) != LEROBOT_PARQUET_COLUMNS[len(LEROBOT_CORE_PARQUET_COLUMNS) :]:
        raise ConversionError("LeRobot extra columns drifted from the canonical contract")
    return columns


def _write_parquet(
    path: Path,
    aligned: AlignedEpisode,
    episode_index: int,
    fps: int,
) -> dict[str, Any]:
    length = len(aligned.source_timestamps_ns)
    timestamp = np.arange(length, dtype=np.float32) / np.float32(fps)
    frame_index = np.arange(length, dtype=np.int64)
    episode_indices = np.full(length, episode_index, dtype=np.int64)
    episode_local_indices = np.arange(length, dtype=np.int64)
    task_indices = np.zeros(length, dtype=np.int64)
    arrays: list[pa.Array] = [
        _fixed_float_list(aligned.states, OUTPUT_DIM),
        _fixed_float_list(aligned.actions, OUTPUT_DIM),
        pa.array(timestamp, type=pa.float32()),
        pa.array(frame_index, type=pa.int64()),
        pa.array(episode_indices, type=pa.int64()),
        pa.array(episode_local_indices, type=pa.int64()),
        pa.array(task_indices, type=pa.int64()),
    ]
    names = list(LEROBOT_CORE_PARQUET_COLUMNS)
    extra_columns = _extra_parquet_columns(aligned)
    arrays.extend(extra_columns.values())
    names.extend(extra_columns)
    table = pa.Table.from_arrays(arrays, names=names)
    table = table.replace_schema_metadata(LEROBOT_PARQUET_SCHEMA_METADATA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return {
        "observation.state": _numeric_stats(aligned.states),
        "action": _numeric_stats(aligned.actions),
        "timestamp": _numeric_stats(timestamp),
        "frame_index": _numeric_stats(frame_index),
        "episode_index": _numeric_stats(episode_indices),
        "index": _numeric_stats(episode_local_indices),
        "task_index": _numeric_stats(task_indices),
    }


def _probe_video(path: Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise ConversionError(f"Expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def _validate_video(
    path: Path,
    expected_length: int,
    image_size: tuple[int, int],
    fps: int,
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    stream = _probe_video(path, ffprobe_bin)
    height, width = image_size
    expected = {
        "codec_name": "av1",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
        "nb_frames": expected_length,
    }
    for key, value in expected.items():
        actual: Any = stream.get(key)
        if key in {"width", "height", "nb_frames"}:
            actual = int(actual)
        if actual != value:
            raise ConversionError(
                f"Video {path} has {key}={actual!r}; expected {value!r}"
            )
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    actual_fps = float(numerator) / float(denominator)
    if not math.isclose(actual_fps, fps, rel_tol=0.0, abs_tol=1e-6):
        raise ConversionError(f"Video {path} has fps={actual_fps}; expected {fps}")
    return stream


def _alignment_summary(aligned: AlignedEpisode) -> dict[str, Any]:
    return {
        name: {
            "max_ms": float(errors.max() / 1e6),
            "mean_ms": float(errors.mean() / 1e6),
            "p99_ms": float(np.percentile(errors, 99) / 1e6),
        }
        for name, errors in aligned.alignment_error_ns.items()
    }


def _new_reader(source_dir: Path, data_format: str | None = None) -> Any:
    if data_format is None:
        metadata = _read_json(source_dir / "metadata.json")
        data_format = normalize_data_format(metadata.get("data_format"))
    return _reader_for_format(source_dir, data_format)


def _convert_episode_camera(
    root: Path,
    plan: EpisodePlan,
    aligned: AlignedEpisode,
    episode_index: int,
    feature: str,
    camera_id: str | None,
    image_size: tuple[int, int],
    camera_crop: tuple[int, int, int, int],
    fps: int,
    encoder: VideoEncoderConfig,
    gpu_id: int | None,
    reader: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    episode_chunk = episode_index // CHUNK_SIZE
    video_path = root / VIDEO_PATH.format(
        episode_chunk=episode_chunk,
        video_key=feature,
        episode_index=episode_index,
    )
    if camera_id is None:
        stats = _write_black_video(
            video_path,
            len(aligned.states),
            image_size,
            fps,
            encoder,
            gpu_id,
        )
    else:
        camera_reader = (
            reader
            if reader is not None
            else _new_reader(plan.source_dir)
        )
        stats = _write_video(
            camera_reader,
            camera_id,
            aligned.camera_indices[feature],
            video_path,
            image_size,
            camera_crop,
            fps,
            encoder,
            gpu_id,
            plan.data_format,
        )
    _validate_video(
        video_path,
        len(aligned.states),
        image_size,
        fps,
        encoder.ffprobe_bin,
    )
    return feature, stats


def _convert_episode(
    root: Path,
    plan: EpisodePlan,
    episode_index: int,
    wrist_image_size: tuple[int, int],
    head_image_size: tuple[int, int],
    crop_lrtb: tuple[int, int, int, int],
    tcp_pose_macro: str,
    fps: int,
    max_alignment_error_ns: int,
    camera_workers: int,
    encoder: VideoEncoderConfig,
    job_ordinal: int,
) -> EpisodeArtifact:
    reader = _new_reader(plan.source_dir)
    aligned = _align_episode(
        reader, plan, max_alignment_error_ns, tcp_pose_macro
    )
    episode_chunk = episode_index // CHUNK_SIZE
    parquet_path = root / DATA_PATH.format(
        episode_chunk=episode_chunk, episode_index=episode_index
    )
    stats = _write_parquet(parquet_path, aligned, episode_index, fps)
    _write_joint_sidecar(root, aligned, episode_index)
    camera_configs = (
        (HEAD_CAMERA_FEATURE, plan.head_camera, head_image_size, (0, 0, 0, 0)),
        (LEFT_CAMERA_FEATURE, plan.left_camera, wrist_image_size, crop_lrtb),
        (RIGHT_CAMERA_FEATURE, plan.right_camera, wrist_image_size, crop_lrtb),
    )
    camera_jobs = [
        (
            feature,
            camera_id,
            image_size,
            camera_crop,
            encoder.gpu_for(job_ordinal, camera_ordinal),
        )
        for camera_ordinal, (feature, camera_id, image_size, camera_crop) in enumerate(
            camera_configs
        )
    ]
    if camera_workers == 1:
        for feature, camera_id, image_size, camera_crop, gpu_id in camera_jobs:
            result_feature, feature_stats = _convert_episode_camera(
                root,
                plan,
                aligned,
                episode_index,
                feature,
                camera_id,
                image_size,
                camera_crop,
                fps,
                encoder,
                gpu_id,
                reader,
            )
            stats[result_feature] = feature_stats
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(camera_workers, len(camera_jobs)),
            thread_name_prefix=f"camera-{plan.source_index}",
        ) as executor:
            futures = [
                executor.submit(
                    _convert_episode_camera,
                    root,
                    plan,
                    aligned,
                    episode_index,
                    feature,
                    camera_id,
                    image_size,
                    camera_crop,
                    fps,
                    encoder,
                    gpu_id,
                    None,
                )
                for feature, camera_id, image_size, camera_crop, gpu_id in camera_jobs
            ]
            for future in futures:
                result_feature, feature_stats = future.result()
                stats[result_feature] = feature_stats
    parquet_table = pq.read_table(parquet_path)
    if parquet_table.num_rows != len(aligned.states):
        raise ConversionError(
            f"Parquet row count {parquet_table.num_rows} != {len(aligned.states)}"
        )
    return EpisodeArtifact(
        episode_index=episode_index,
        episode_uuid=_stable_episode_uuid(plan, aligned),
        source_index=plan.source_index,
        source_name=plan.source_dir.name,
        result=plan.result,
        is_success=plan.is_success,
        task=plan.task,
        length=len(aligned.states),
        stats=stats,
        alignment_summary=_alignment_summary(aligned),
    )


def _convert_episode_job(job: EpisodeJob) -> EpisodeArtifact:
    """Build one self-contained provisional episode bundle.

    Provisional scalar indexes are deliberately local. The coordinator compacts
    successful bundles in source order and rewrites only dataset-level episode
    and task identities during commit, so a failed parallel job can never leave
    an episode-index gap.
    """
    job.bundle_root.mkdir(parents=True, exist_ok=False)
    return _convert_episode(
        job.bundle_root,
        job.plan,
        0,
        job.wrist_image_size,
        job.head_image_size,
        job.crop_lrtb,
        job.tcp_pose_macro,
        job.fps,
        job.max_alignment_error_ns,
        job.camera_workers,
        job.encoder,
        job.ordinal,
    )


def _commit_episode_bundle(
    staging_root: Path,
    bundle_root: Path,
    artifact: EpisodeArtifact,
    episode_index: int,
) -> EpisodeArtifact:
    """Atomically compact one successful provisional bundle into the dataset."""
    provisional_parquet = bundle_root / DATA_PATH.format(
        episode_chunk=0, episode_index=0
    )
    table = pq.read_table(provisional_parquet)
    if table.num_rows != artifact.length:
        raise ConversionError(
            f"Bundle row count {table.num_rows} != artifact length {artifact.length} "
            f"for {artifact.source_name}"
        )
    episode_indices = np.full(artifact.length, episode_index, dtype=np.int64)
    episode_local_indices = np.arange(artifact.length, dtype=np.int64)
    episode_column = table.schema.get_field_index("episode_index")
    index_column = table.schema.get_field_index("index")
    if episode_column < 0 or index_column < 0:
        raise ConversionError(f"Bundle parquet is missing scalar index columns: {provisional_parquet}")
    table = table.set_column(
        episode_column,
        table.schema.field(episode_column),
        pa.array(episode_indices, type=pa.int64()),
    )
    table = table.set_column(
        index_column,
        table.schema.field(index_column),
        pa.array(episode_local_indices, type=pa.int64()),
    )

    episode_chunk = episode_index // CHUNK_SIZE
    final_parquet = staging_root / DATA_PATH.format(
        episode_chunk=episode_chunk, episode_index=episode_index
    )
    final_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, final_parquet, compression="snappy")
    for feature in CAMERA_FEATURES:
        provisional_video = bundle_root / VIDEO_PATH.format(
            episode_chunk=0,
            video_key=feature,
            episode_index=0,
        )
        final_video = staging_root / VIDEO_PATH.format(
            episode_chunk=episode_chunk,
            video_key=feature,
            episode_index=episode_index,
        )
        final_video.parent.mkdir(parents=True, exist_ok=True)
        provisional_video.replace(final_video)

    provisional_sidecars = (
        bundle_root / "meta" / "source_sidecars" / "episode_000000"
    )
    if provisional_sidecars.is_dir():
        final_sidecars = (
            staging_root
            / "meta"
            / "source_sidecars"
            / f"episode_{episode_index:06d}"
        )
        final_sidecars.parent.mkdir(parents=True, exist_ok=True)
        if final_sidecars.exists():
            raise ConversionError(
                f"Joint sidecar destination already exists: {final_sidecars}"
            )
        provisional_sidecars.replace(final_sidecars)

    artifact.episode_index = episode_index
    artifact.stats["episode_index"] = _numeric_stats(episode_indices)
    artifact.stats["index"] = _numeric_stats(episode_local_indices)
    shutil.rmtree(bundle_root)
    return artifact


def _video_feature_spec(image_size: tuple[int, int], fps: int) -> dict[str, Any]:
    height, width = image_size
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def _feature_spec(
    wrist_image_size: tuple[int, int],
    head_image_size: tuple[int, int],
    fps: int,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [OUTPUT_DIM],
            "names": {"motors": list(FEATURE_NAMES)},
        },
        "action": {
            "dtype": "float32",
            "shape": [OUTPUT_DIM],
            "names": {"motors": list(FEATURE_NAMES)},
        },
        HEAD_CAMERA_FEATURE: _video_feature_spec(head_image_size, fps),
        LEFT_CAMERA_FEATURE: _video_feature_spec(wrist_image_size, fps),
        RIGHT_CAMERA_FEATURE: _video_feature_spec(wrist_image_size, fps),
    }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def _finalize_metadata(
    root: Path,
    device_type: str,
    artifacts: Sequence[EpisodeArtifact],
    wrist_image_size: tuple[int, int],
    head_image_size: tuple[int, int],
    camera_sources: dict[str, str | None],
    fps: int,
) -> None:
    total_frames = sum(artifact.length for artifact in artifacts)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "nmx_data_version": NMX_DATA_VERSION,
        "robot_type": "dual_arm",
        "total_episodes": len(artifacts),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(artifacts) * len(CAMERA_FEATURES),
        "total_chunks": math.ceil(len(artifacts) / CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "target_fps": TARGET_FPS,
        "splits": {"train": f"0:{len(artifacts)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": _feature_spec(wrist_image_size, head_image_size, fps),
    }
    _write_json(root / "meta" / "info.json", info)
    task = artifacts[0].task
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    episode_records = []
    episode_stats_records = []
    for artifact in artifacts:
        episode_records.append(
            {
                "episode_index": artifact.episode_index,
                "episode_uuid": artifact.episode_uuid,
                "tasks": [artifact.task],
                "task_description": artifact.task,
                "task_annotation": artifact.task,
                "instruct_ch": "",
                "hardware_string": [
                    str(camera_sources[feature] or "")
                    for feature in (
                        LEFT_CAMERA_FEATURE,
                        RIGHT_CAMERA_FEATURE,
                        HEAD_CAMERA_FEATURE,
                    )
                ],
                "rp_type": [],
                "device_type": device_type,
                "collection_mode": "inference_rollout",
                "slam_validity_all": True,
                "width_detection": False,
                "width_calibration": False,
                "length": artifact.length,
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": artifact.length,
                        "action_text": artifact.task,
                    }
                ],
            }
        )
        episode_stats_records.append(
            {
                "episode_index": artifact.episode_index,
                "stats": artifact.stats,
            }
        )
    _write_jsonl(root / "meta" / "episodes.jsonl", episode_records)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", episode_stats_records)
    modality = {
        "state": {
            name: {"start": index, "end": index + 1}
            for index, name in enumerate(FEATURE_NAMES)
        },
        "action": {
            name: {"start": index, "end": index + 1}
            for index, name in enumerate(FEATURE_NAMES)
        },
        "video": {
            feature.rsplit(".", maxsplit=1)[-1]: {"original_key": feature}
            for feature in CAMERA_FEATURES
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }
    _write_json(root / "meta" / "modality.json", modality)


def _write_joint_sidecar(
    root: Path,
    aligned: AlignedEpisode,
    episode_index: int,
) -> list[str]:
    if not aligned.joint_positions:
        return []
    destination = (
        root
        / "meta"
        / "source_sidecars"
        / f"episode_{episode_index:06d}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "timestamps_ns": np.asarray(
            aligned.source_timestamps_ns, dtype=np.int64
        )
    }
    dimensions: dict[str, int] = {}
    for side in ("left", "right", "ego"):
        values = aligned.joint_positions.get(side)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(aligned.states):
            raise ConversionError(
                f"{side} joint sidecar has invalid shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ConversionError(f"{side} joint sidecar contains non-finite values")
        arrays[side] = values
        dimensions[side] = int(values.shape[1])
    if not dimensions:
        return []
    npz_path = destination / "joint_trajectory.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest_path = destination / "joint_trajectory.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "source_data_format": NEDF3_DATA_FORMAT,
            "frames": len(aligned.states),
            "timestamp_key": "timestamps_ns",
            "position_unit": "rad",
            "dimensions": dimensions,
            "archive": npz_path.name,
        },
    )
    return [
        npz_path.relative_to(root).as_posix(),
        manifest_path.relative_to(root).as_posix(),
    ]


def _parquet_signature(path: Path) -> dict[str, tuple[str, int | None]]:
    schema = pq.read_schema(path)
    signature: dict[str, tuple[str, int | None]] = {}
    if tuple(schema.names) != LEROBOT_PARQUET_COLUMNS:
        raise ConversionError(
            f"Unexpected parquet columns in {path}: {schema.names}; "
            f"expected {LEROBOT_PARQUET_COLUMNS}"
        )
    for name in LEROBOT_PARQUET_COLUMNS:
        field = schema.field(name)
        width = field.type.list_size if pa.types.is_fixed_size_list(field.type) else None
        value_type = field.type.value_type if width is not None else field.type
        signature[name] = (str(value_type), width)
    return signature


def _fixed_list_numpy(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.float32)


def _validate_reference_extra_fields(table: pa.Table, path: Path) -> None:
    states = _fixed_list_numpy(table, "observation.state")
    for side, block_start in (("left", 0), ("right", 8), ("ego", 16)):
        prefix = f"observation.extra.{side}"
        raw_pose = _fixed_list_numpy(table, f"{prefix}.raw_pose")
        if raw_pose.shape != (table.num_rows, 7) or not np.isfinite(raw_pose).all():
            raise ConversionError(f"{path}: invalid {side} raw_pose")
        quat_norm = np.linalg.norm(raw_pose[:, 3:7], axis=1)
        if not np.allclose(quat_norm, 1.0, rtol=0.0, atol=1e-4):
            raise ConversionError(f"{path}: {side} raw_pose quaternion is not normalized")
        hand_state = table[f"{prefix}.hand_state"]
        if side == "ego":
            if hand_state.null_count != table.num_rows:
                raise ConversionError(f"{path}: ego hand_state must be null")
        elif not np.array_equal(
            hand_state.to_numpy(), states[:, block_start + 7]
        ):
            raise ConversionError(
                f"{path}: {side} hand_state differs from observation.state"
            )
        aligned_ns = table[f"{prefix}.aligned_timestamp_ns"].to_numpy()
        if not np.isfinite(aligned_ns).all() or np.any(np.diff(aligned_ns) < 0):
            raise ConversionError(f"{path}: invalid {side} aligned_timestamp_ns")


def _validate_state_action_contract(
    table: pa.Table, path: Path, *, require_next_state: bool
) -> None:
    states = _fixed_list_numpy(table, "observation.state")
    actions = _fixed_list_numpy(table, "action")
    expected_shape = (table.num_rows, OUTPUT_DIM)
    if states.shape != expected_shape or actions.shape != expected_shape:
        raise ConversionError(
            f"{path}: state/action shape mismatch: {states.shape}/{actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ConversionError(f"{path}: state/action contains non-finite values")
    for label, values in (("state", states), ("action", actions)):
        for side, start in (("left", 3), ("right", 11), ("ego", 19)):
            norms = np.linalg.norm(values[:, start : start + 4], axis=1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
                raise ConversionError(
                    f"{path}: {label} {side} wxyz quaternion is not normalized"
                )
    if (
        require_next_state
        and table.num_rows > 1
        and not np.array_equal(actions[:-1], states[1:])
    ):
        raise ConversionError(
            f"{path}: action[t] does not exactly equal observation.state[t+1]"
        )


def validate_dataset(
    root: Path,
    ffprobe_bin: str = "ffprobe",
    *,
    require_next_state: bool = True,
) -> dict[str, Any]:
    info = _read_json(root / "meta" / "info.json")
    required_meta = (
        "info.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "modality.json",
    )
    missing_meta = [name for name in required_meta if not (root / "meta" / name).is_file()]
    if missing_meta:
        raise ConversionError(f"Missing LeRobot metadata files: {missing_meta}")
    if set(info) != set(LEROBOT_INFO_FIELDS):
        raise ConversionError(
            f"Unexpected info.json fields: {sorted(info)}; "
            f"expected {sorted(LEROBOT_INFO_FIELDS)}"
        )
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise ConversionError(
            f"Unexpected codebase_version: {info.get('codebase_version')!r}"
        )
    if info.get("nmx_data_version") != NMX_DATA_VERSION:
        raise ConversionError(
            f"Unexpected nmx_data_version: {info.get('nmx_data_version')!r}"
        )
    if info.get("fps") != LEROBOT_FPS:
        raise ConversionError(
            f"LeRobot fps must be {LEROBOT_FPS}, got {info.get('fps')!r}"
        )
    if info.get("target_fps") != LEROBOT_TARGET_FPS:
        raise ConversionError(
            "LeRobot target_fps must be "
            f"{LEROBOT_TARGET_FPS}, got {info.get('target_fps')!r}"
        )
    if info.get("data_path") != DATA_PATH or info.get("video_path") != VIDEO_PATH:
        raise ConversionError("info.json uses non-canonical data/video paths")
    if (root / "meta" / "stats.json").exists():
        raise ConversionError("meta/stats.json must be absent for this contract")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ConversionError("info.json features must be an object")
    expected_features = {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        *CAMERA_FEATURES,
    }
    if set(features) != expected_features:
        raise ConversionError(
            f"Unexpected feature keys: {sorted(features)}; "
            f"expected {sorted(expected_features)}"
        )
    video_features = {
        name
        for name, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    }
    if video_features != set(CAMERA_FEATURES):
        raise ConversionError(
            f"Unexpected video feature keys: {sorted(video_features)}"
        )

    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    episode_stats = _read_jsonl(root / "meta" / "episodes_stats.jsonl")
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    total_episodes = int(info["total_episodes"])
    if len(episodes) != total_episodes or len(episode_stats) != total_episodes:
        raise ConversionError("episode metadata counts do not match total_episodes")
    task_indexes = {
        record.get("task_index")
        for record in tasks
        if isinstance(record.get("task_index"), int)
        and not isinstance(record.get("task_index"), bool)
        and isinstance(record.get("task"), str)
    }
    if len(task_indexes) != len(tasks) or task_indexes != set(range(len(tasks))):
        raise ConversionError("tasks.jsonl does not contain contiguous unique tasks")
    if len(tasks) != int(info["total_tasks"]):
        raise ConversionError("tasks.jsonl count does not match total_tasks")
    episode_uuids: set[str] = set()
    for episode_index, episode in enumerate(episodes):
        if episode.get("episode_index") != episode_index:
            raise ConversionError("episodes.jsonl has non-contiguous episode_index")
        try:
            episode_uuid = str(uuid.UUID(str(episode.get("episode_uuid"))))
        except ValueError as exc:
            raise ConversionError(
                f"episode {episode_index} has an invalid episode_uuid"
            ) from exc
        if episode_uuid in episode_uuids:
            raise ConversionError(f"duplicate episode_uuid: {episode_uuid}")
        episode_uuids.add(episode_uuid)
        if episode_stats[episode_index].get("episode_index") != episode_index:
            raise ConversionError("episodes_stats.jsonl has incorrect episode_index")

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    video_files = sorted((root / "videos").rglob("*.mp4"))
    if len(parquet_files) != int(info["total_episodes"]):
        raise ConversionError(
            f"Parquet count {len(parquet_files)} != episodes {info['total_episodes']}"
        )
    if len(video_files) != int(info["total_videos"]):
        raise ConversionError(
            f"Video count {len(video_files)} != total_videos {info['total_videos']}"
        )
    if len(video_files) != total_episodes * len(CAMERA_FEATURES):
        raise ConversionError("Each episode must contain exactly three videos")
    expected_signature = {
        "observation.state": ("float", OUTPUT_DIM),
        "action": ("float", OUTPUT_DIM),
        "timestamp": ("float", None),
        "frame_index": ("int64", None),
        "episode_index": ("int64", None),
        "index": ("int64", None),
        "task_index": ("int64", None),
    }
    for side in ("left", "right", "ego"):
        prefix = f"observation.extra.{side}"
        expected_signature[f"{prefix}.raw_pose"] = ("float", 7)
        for field in LEROBOT_EXTRA_FIELDS[1:4]:
            expected_signature[f"{prefix}.{field}"] = ("float", None)
        expected_signature[f"{prefix}.aligned_timestamp_ns"] = ("double", None)
    total_rows = 0
    camera_image_sizes = {
        feature: tuple(info["features"][feature]["shape"][:2])
        for feature in CAMERA_FEATURES
    }
    for episode_index, parquet_path in enumerate(parquet_files):
        expected_parquet_path = root / DATA_PATH.format(
            episode_chunk=episode_index // CHUNK_SIZE,
            episode_index=episode_index,
        )
        if parquet_path != expected_parquet_path:
            raise ConversionError(
                f"Non-canonical parquet path {parquet_path}; expected {expected_parquet_path}"
            )
        table = pq.read_table(parquet_path)
        total_rows += table.num_rows
        signature = _parquet_signature(parquet_path)
        if signature != expected_signature:
            raise ConversionError(
                f"Unexpected parquet schema in {parquet_path}: {signature}"
            )
        metadata = table.schema.metadata or {}
        if metadata != LEROBOT_PARQUET_SCHEMA_METADATA:
            raise ConversionError(
                f"Unexpected Parquet metadata in {parquet_path}: {metadata}"
            )
        _validate_state_action_contract(
            table, parquet_path, require_next_state=require_next_state
        )
        _validate_reference_extra_fields(table, parquet_path)
        frame_indices = table["frame_index"].to_numpy()
        timestamps = table["timestamp"].to_numpy()
        episode_indices = table["episode_index"].to_numpy()
        episode_local_indices = table["index"].to_numpy()
        if not np.array_equal(frame_indices, np.arange(table.num_rows)):
            raise ConversionError(f"Non-contiguous frame_index in {parquet_path}")
        expected_timestamps = np.arange(table.num_rows, dtype=np.float32) / np.float32(
            info["fps"]
        )
        if not np.array_equal(timestamps, expected_timestamps):
            raise ConversionError(f"Non-uniform timestamp timeline in {parquet_path}")
        if not np.array_equal(
            episode_indices,
            np.full(table.num_rows, episode_index, dtype=np.int64),
        ):
            raise ConversionError(f"Incorrect episode_index column in {parquet_path}")
        expected_indices = np.arange(table.num_rows)
        if not np.array_equal(episode_local_indices, expected_indices):
            raise ConversionError(f"Non-contiguous episode-local index in {parquet_path}")
        if episodes[episode_index].get("length") != table.num_rows:
            raise ConversionError(
                f"episodes.jsonl length mismatch in episode {episode_index}"
            )
        table_task_indexes = set(int(value) for value in table["task_index"].to_numpy())
        if not table_task_indexes or not table_task_indexes.issubset(task_indexes):
            raise ConversionError(f"Unknown task_index in {parquet_path}")
        episode_videos = [
            root
            / VIDEO_PATH.format(
                episode_chunk=episode_index // CHUNK_SIZE,
                video_key=feature,
                episode_index=episode_index,
            )
            for feature in CAMERA_FEATURES
        ]
        for video_path in episode_videos:
            feature = video_path.parent.name
            _validate_video(
                video_path,
                table.num_rows,
                camera_image_sizes[feature],
                int(info["fps"]),
                ffprobe_bin,
            )
    if total_rows != int(info["total_frames"]):
        raise ConversionError(f"Parquet rows {total_rows} != total_frames {info['total_frames']}")

    report: dict[str, Any] = {
        "status": "passed",
        "dataset_root": str(root),
        "codebase_version": info["codebase_version"],
        "nmx_data_version": info["nmx_data_version"],
        "episodes": len(parquet_files),
        "frames": total_rows,
        "videos": len(video_files),
        "fps": info["fps"],
        "state_dim": OUTPUT_DIM,
        "action_dim": OUTPUT_DIM,
        "camera_features": list(CAMERA_FEATURES),
        "camera_image_sizes_hw": {
            feature: list(image_size)
            for feature, image_size in camera_image_sizes.items()
        },
        "episode_uuids": sorted(episode_uuids),
        "action_is_next_state": require_next_state,
        "info_fields": list(LEROBOT_INFO_FIELDS),
    }
    return report


def convert(args: argparse.Namespace) -> Path:
    if args.fps != LEROBOT_FPS:
        raise ConversionError(
            f"This LeRobot contract requires --fps {LEROBOT_FPS}, got {args.fps}"
        )
    data_dir = args.data_dir.resolve()
    output_parent = args.output_dir.resolve()
    final_root = output_parent / args.dataset_name
    encoder = _resolve_video_encoder(args)
    if final_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {final_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(final_root)
    output_parent.mkdir(parents=True, exist_ok=True)
    allowed_results = (
        {"success"} if args.success_only else {"success", "failure", "incomplete"}
    )
    requested_episodes = set(args.episodes or [])
    discovered = _discover_episode_dirs(data_dir)
    if requested_episodes:
        discovered = [item for item in discovered if item[0] in requested_episodes]
        missing = requested_episodes - {index for index, _ in discovered}
        if missing:
            raise ConversionError(f"Requested episode indices were not found: {sorted(missing)}")
    if args.num_demos > 0:
        discovered = discovered[: args.num_demos]
    if not discovered:
        raise ConversionError(f"No completed NEDF episode directories found in {data_dir}")

    plans: list[EpisodePlan] = []
    skipped: list[dict[str, Any]] = [
        {
            "source_episode": path.name,
            "original_episode_index": int(path.name.split("_")[1]),
            "reason": "temporary episode directory is incomplete",
        }
        for path in sorted(data_dir.glob("episode_[0-9][0-9][0-9][0-9]_temp"))
        if path.is_dir()
    ]
    for source_index, source_dir in discovered:
        try:
            plans.append(
                _build_plan(
                    source_index,
                    source_dir,
                    allowed_results,
                    args.head_camera,
                    args.left_camera,
                    args.right_camera,
                    args.action_topic,
                )
            )
        except Exception as exc:
            skipped.append(
                {
                    "source_episode": source_dir.name,
                    "original_episode_index": source_index,
                    "reason": str(exc),
                }
            )
            LOGGER.warning("Skipping %s: %s", source_dir.name, exc)
    if not plans:
        raise ConversionError("No source episodes passed NEDF structural validation")
    tasks = {plan.task for plan in plans}
    if args.task:
        plans = [replace(plan, task=args.task) for plan in plans]
    elif len(tasks) != 1:
        raise ConversionError(
            f"LeRobot output is configured as single-task, but found tasks: {sorted(tasks)}. "
            "Use --task to provide one canonical task."
        )

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{args.dataset_name}.tmp-", dir=output_parent)
    )
    artifacts: list[EpisodeArtifact] = []
    conversion_failures: list[dict[str, Any]] = []
    try:
        jobs_root = staging_root / ".episode_jobs"
        jobs_root.mkdir()
        jobs = [
            EpisodeJob(
                ordinal=ordinal,
                bundle_root=jobs_root / f"job_{ordinal:06d}",
                plan=plan,
                wrist_image_size=tuple(args.wrist_image_size),
                head_image_size=tuple(args.head_image_size),
                crop_lrtb=tuple(args.crop_lrtb),
                tcp_pose_macro=args.tcp_pose_macro,
                fps=args.fps,
                max_alignment_error_ns=int(args.max_alignment_error_ms * 1e6),
                camera_workers=args.camera_workers,
                encoder=encoder,
            )
            for ordinal, plan in enumerate(plans)
        ]

        def record_failed_job(job: EpisodeJob, exc: BaseException) -> None:
            failure = {
                "source_episode": job.plan.source_dir.name,
                "original_episode_index": job.plan.source_index,
                "reason": str(exc),
            }
            conversion_failures.append(failure)
            skipped.append(failure)
            shutil.rmtree(job.bundle_root, ignore_errors=True)

        def commit_success(job: EpisodeJob, artifact: EpisodeArtifact) -> None:
            episode_index = len(artifacts)
            committed = _commit_episode_bundle(
                staging_root,
                job.bundle_root,
                artifact,
                episode_index,
            )
            artifacts.append(committed)
            LOGGER.info(
                "Committed %s -> episode_%06d (%d frames)",
                job.plan.source_dir.name,
                episode_index,
                committed.length,
            )

        if args.workers == 1:
            for job in jobs:
                LOGGER.info(
                    "Converting %s (%s)", job.plan.source_dir.name, job.plan.result
                )
                try:
                    artifact = _convert_episode_job(job)
                except Exception as exc:
                    record_failed_job(job, exc)
                    if args.fail_on_conversion_error:
                        raise ConversionError(
                            f"Conversion failed for {job.plan.source_dir.name}: {exc}"
                        ) from exc
                    LOGGER.exception(
                        "Skipping %s after conversion error", job.plan.source_dir.name
                    )
                    continue
                commit_success(job, artifact)
        else:
            LOGGER.info(
                "Converting %d episodes with %d episode workers x up to %d camera workers",
                len(jobs),
                min(args.workers, len(jobs)),
                args.camera_workers,
            )
            spawn_context = multiprocessing.get_context("spawn")
            completed: dict[int, tuple[EpisodeArtifact | None, BaseException | None]] = {}
            next_to_commit = 0
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.workers, len(jobs)),
                mp_context=spawn_context,
            ) as executor:
                future_to_job = {
                    executor.submit(_convert_episode_job, job): job for job in jobs
                }
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        completed[job.ordinal] = (future.result(), None)
                    except Exception as exc:
                        completed[job.ordinal] = (None, exc)
                    while next_to_commit in completed:
                        artifact, exc = completed.pop(next_to_commit)
                        ordered_job = jobs[next_to_commit]
                        if exc is not None:
                            record_failed_job(ordered_job, exc)
                            if args.fail_on_conversion_error:
                                for pending_future in future_to_job:
                                    pending_future.cancel()
                                raise ConversionError(
                                    "Conversion failed for "
                                    f"{ordered_job.plan.source_dir.name}: {exc}"
                                ) from exc
                            LOGGER.error(
                                "Skipping %s after parallel conversion error: %s",
                                ordered_job.plan.source_dir.name,
                                exc,
                            )
                        else:
                            assert artifact is not None
                            commit_success(ordered_job, artifact)
                        next_to_commit += 1
        shutil.rmtree(jobs_root, ignore_errors=True)
        if not artifacts:
            raise ConversionError("Every planned episode failed during conversion")
        successful_plan = next(
            plan for plan in plans if plan.source_index == artifacts[0].source_index
        )
        camera_sources = {
            HEAD_CAMERA_FEATURE: successful_plan.head_camera,
            LEFT_CAMERA_FEATURE: successful_plan.left_camera,
            RIGHT_CAMERA_FEATURE: successful_plan.right_camera,
        }
        _finalize_metadata(
            staging_root,
            args.device_type,
            artifacts,
            tuple(args.wrist_image_size),
            tuple(args.head_image_size),
            camera_sources,
            args.fps,
        )
        successful_source_indexes = {artifact.source_index for artifact in artifacts}
        source_data_formats = sorted(
            {
                plan.data_format
                for plan in plans
                if plan.source_index in successful_source_indexes
            }
        )
        report = {
            "status": "completed_with_skips" if skipped else "completed",
            "counts": {
                "discovered": len(discovered),
                "planned": len(plans),
                "converted": len(artifacts),
                "skipped": len(skipped),
                "conversion_failures": len(conversion_failures),
            },
            "source_root": str(data_dir),
            "source_data_formats": source_data_formats,
            "output_root": str(final_root),
            "tcp_pose_macro": args.tcp_pose_macro,
            "camera_sources": camera_sources,
            "parallelism": {
                "episode_workers": args.workers,
                "camera_workers": args.camera_workers,
            },
            "video_encoding": {
                "requested_encoder": args.video_encoder,
                "selected_encoder": encoder.encoder,
                "quality": encoder.quality,
                "gop": encoder.gop,
                "svt_preset": encoder.svt_preset,
                "nvenc_preset": encoder.nvenc_preset,
                "gpu_ids": list(encoder.gpu_ids),
                "ffmpeg_bin": encoder.ffmpeg_bin,
                "ffprobe_bin": encoder.ffprobe_bin,
                "extra_output_args": list(encoder.extra_output_args),
            },
            "converted": [
                {
                    "episode_index": artifact.episode_index,
                    "source_episode": artifact.source_name,
                    "original_episode_index": artifact.source_index,
                    "result": artifact.result,
                    "length": artifact.length,
                    "alignment": artifact.alignment_summary,
                }
                for artifact in artifacts
            ],
            "skipped": skipped,
        }
        _write_json(staging_root / "meta" / "conversion_report.json", report)
        validation = validate_dataset(
            staging_root,
            encoder.ffprobe_bin,
            require_next_state=False,
        )
        _write_json(staging_root / "meta" / "validation_report.json", validation)
        staging_root.replace(final_root)
        validation["dataset_root"] = str(final_root)
        _write_json(final_root / "meta" / "validation_report.json", validation)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    LOGGER.info(
        "Conversion complete: %d episodes, %d frames -> %s",
        len(artifacts),
        sum(artifact.length for artifact in artifacts),
        final_root,
    )
    return final_root


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _optional_camera(value: str) -> str | None:
    value = value.strip()
    return None if value.lower() == "none" else value


def _gpu_ids(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value or value.lower() in {"none", "default"}:
        return ()
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list of GPU indexes") from exc
    if any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("GPU indexes cannot be negative")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("GPU indexes must be unique")
    return parsed


def _ffmpeg_extra_args(value: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(value, posix=True))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot parse ffmpeg arguments: {exc}") from exc


def _resolve_tcp_pose_macro(requested: str, device_type: str) -> str:
    if requested == "auto":
        normalized = device_type.strip().casefold()
        is_r1_rollout = normalized == LEROBOT_DEVICE_TYPE or "r1" in normalized
        return "r1_world" if is_r1_rollout else "none"
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert SuperInference NEDF3 rollouts to LeRobot v2.1."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--head-camera",
        type=_optional_camera,
        default=None,
        help="Head camera id, or None to emit a black observation.images.image video.",
    )
    parser.add_argument("--left-camera", type=_optional_camera, default=None)
    parser.add_argument("--right-camera", type=_optional_camera, default=None)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument(
        "--device-type",
        default=LEROBOT_DEVICE_TYPE,
        help=(
            "Device type stored in LeRobot metadata "
            f"(default: {LEROBOT_DEVICE_TYPE})."
        ),
    )
    parser.add_argument(
        "--task-description",
        "--task",
        dest="task",
        default=None,
        help="Override metadata task_name with one canonical language instruction.",
    )
    parser.add_argument("--fps", type=_positive_int, default=LEROBOT_FPS)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Parallel episode processes (default: 1, preserving legacy behavior).",
    )
    parser.add_argument(
        "--camera-workers",
        type=_positive_int,
        default=1,
        help="Parallel camera encodes within each episode (default: 1; maximum useful value is 3).",
    )
    parser.add_argument(
        "--video-encoder",
        choices=("auto", "libsvtav1", "av1_nvenc"),
        default="libsvtav1",
        help=(
            "AV1 encoder. Default libsvtav1 preserves legacy behavior; auto tries "
            "av1_nvenc then falls back to libsvtav1 after a real encoder smoke test."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        type=_gpu_ids,
        default=(),
        metavar="0,1,...",
        help="NVENC GPU indexes used round-robin across episode/camera jobs.",
    )
    parser.add_argument(
        "--video-quality",
        type=int,
        default=30,
        help="libsvtav1 CRF or av1_nvenc constant-quality value (default: 30).",
    )
    parser.add_argument(
        "--video-gop",
        type=_positive_int,
        default=2,
        help="Video GOP length (default: 2, preserving legacy output).",
    )
    parser.add_argument(
        "--svt-preset",
        type=int,
        default=8,
        help="libsvtav1 preset 0-13 (default: 8).",
    )
    parser.add_argument(
        "--nvenc-preset",
        choices=tuple(f"p{index}" for index in range(1, 8)),
        default="p4",
        help="av1_nvenc preset p1-p7 (default: p4).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable (supports an isolated custom build).",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable paired with --ffmpeg-bin.",
    )
    parser.add_argument(
        "--ffmpeg-extra-args",
        type=_ffmpeg_extra_args,
        default=(),
        metavar="'ARGS'",
        help="Extra output arguments appended after encoder defaults (advanced).",
    )
    parser.add_argument(
        "--wrist-image-size",
        "--image-size",
        dest="wrist_image_size",
        type=_positive_int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(320, 240),
        help="Output wrist image size (default: 320 240, matching OpenPi policy input).",
    )
    parser.add_argument(
        "--head-image-size",
        type=_positive_int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(240, 320),
        help="Output head image size (default: 240 320, matching LeRobot view 0).",
    )
    parser.add_argument(
        "--crop-lrtb",
        type=int,
        nargs=4,
        metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        default=(185, 185, 0, 0),
        help="Crop source wrist frames before resizing (default: 185 185 0 0).",
    )
    parser.add_argument(
        "--tcp-pose-macro",
        choices=("auto", "none", *TCP_POSE_MACROS),
        default="auto",
        help="TCP coordinate macro. auto applies r1_world to rollout/R1 datasets.",
    )
    parser.add_argument(
        "--max-alignment-error-ms",
        type=float,
        default=50.0,
        help="Drop frames whose nearest source sample is farther away (default: 50).",
    )
    parser.add_argument("--num-demos", type=int, default=0)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Exclude completed failure rollouts. Discard/incomplete episodes are always excluded.",
    )
    parser.add_argument(
        "--fail-on-conversion-error",
        action="store_true",
        help=(
            "Abort without publishing a dataset if any planned episode fails during "
            "conversion. Structural exclusions are still reported as skipped."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    if args.max_alignment_error_ms <= 0:
        raise ValueError("--max-alignment-error-ms must be positive")
    if args.num_demos < 0:
        raise ValueError("--num-demos cannot be negative")
    if not args.dataset_name.strip() or Path(args.dataset_name).name != args.dataset_name:
        raise ValueError("--dataset-name must be a single non-empty directory name")
    if not 0 <= args.video_quality <= 63:
        raise ValueError("--video-quality must be between 0 and 63")
    if not 0 <= args.svt_preset <= 13:
        raise ValueError("--svt-preset must be between 0 and 13")
    args.ffmpeg_bin = args.ffmpeg_bin.strip()
    args.ffprobe_bin = args.ffprobe_bin.strip()
    if not args.ffmpeg_bin or not args.ffprobe_bin:
        raise ValueError("--ffmpeg-bin and --ffprobe-bin cannot be empty")
    args.device_type = args.device_type.strip()
    if not args.device_type:
        raise ValueError("--device-type cannot be empty")
    if args.task is not None:
        args.task = args.task.strip()
        if not args.task:
            raise ValueError("--task-description cannot be empty")
    args.tcp_pose_macro = _resolve_tcp_pose_macro(
        args.tcp_pose_macro, args.device_type
    )
    root = convert(args)
    validation = _read_json(root / "meta" / "validation_report.json")
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConversionError, FileExistsError, FileNotFoundError, subprocess.SubprocessError) as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)
