#!/usr/bin/env python3
"""Trusted NEDF3 SDK subprocess for read-only episode inspection.

Snapshot mode opens the SDK reader once, emits a compact JSON summary on
stdout, and writes the complete usable low-dimensional timelines to a caller-
provided temporary NPZ.  The parent process atomically publishes that file in
an external cache; this script never writes inside the episode directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

# ``tools`` is nested below ``control_panel/incubator``.  The previous
# ``parents[2]`` value pointed at ``incubator`` itself, so an absolute script
# invocation could not import the ``incubator`` package from a clean child
# process.  Keep the subprocess self-contained regardless of its cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from incubator.control_panel.recording_cameras import resolve_recorded_camera_ids
from incubator.control_panel.nedf3_io import (
    extract_gripper_width,
    extract_joint_state_positions,
    extract_pose_wxyz,
    extract_rgb_jpeg,
    message_timestamp_ns,
)
from incubator.control_panel.recording_contract import (
    NEDF3_ARM_TOPICS,
    NEDF3_DATA_FORMAT,
    NEDF3_HAND_TOPICS,
    normalize_data_format,
)


JOINT_CANDIDATES = {
    "left": (
        NEDF3_ARM_TOPICS["left"]["joint/state"],
        "/lowdim/joint_left",
        "/lowdim/robot_joint_left",
        "/lowdim/joints_left",
    ),
    "right": (
        NEDF3_ARM_TOPICS["right"]["joint/state"],
        "/lowdim/joint_right",
        "/lowdim/robot_joint_right",
        "/lowdim/joints_right",
    ),
}
TCP_CANDIDATES = {
    "left": (
        NEDF3_ARM_TOPICS["left"]["tcp/state"],
        "/lowdim/robot_left",
        "/lowdim/tcp_left",
        "/lowdim/robot_tcp_left",
    ),
    "right": (
        NEDF3_ARM_TOPICS["right"]["tcp/state"],
        "/lowdim/robot_right",
        "/lowdim/tcp_right",
        "/lowdim/robot_tcp_right",
    ),
}
GRIPPER_CANDIDATES = {
    "left": (NEDF3_HAND_TOPICS["left"]["state"], "/lowdim/gripper_left"),
    "right": (NEDF3_HAND_TOPICS["right"]["state"], "/lowdim/gripper_right"),
}
MAX_LOWDIM_FRAMES = 1_000_000
MAX_LOWDIM_TOPICS = 128
MAX_LOWDIM_DIMENSION = 65_536
MAX_TIMELINE_BYTES = 256 * 1024 * 1024
MAX_GRIPPER_CURVE_POINTS = 1000
MAX_CAMERA_FRAMES = 100_000
MAX_JPEG_BYTES = 32 * 1024 * 1024
CAMERA_ROLES = ("head", "left", "right")
CAMERA_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def _timestamp(message: Any) -> int:
    record = getattr(message, "message", message)
    for name in ("publish_time", "log_time", "timestamp_ns", "timestamp"):
        value = getattr(record, name, None)
        if value is not None:
            return int(value)
    raise ValueError("message_timestamp_missing")


def _values(message: Any) -> list[float]:
    decoded = getattr(message, "decoded_message", message)
    values = getattr(decoded, "data", None)
    if values is None:
        raise ValueError("message_data_missing")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("nonfinite_lowdim_value")
    return result


def _issue(code: str, message: str, severity: str = "warning") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def _reader_and_topics(episode: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from dm3_core.nedf import NEDFFactory, NEDFReaderConfig

    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data_format = normalize_data_format(metadata.get("data_format"))
    config = NEDFReaderConfig(metadata_file_path=str(metadata_path))
    unified_reader = getattr(NEDFFactory, "get_reader", None)
    if callable(unified_reader):
        reader = unified_reader(config)
    else:
        raise ValueError(f"SDK cannot read {data_format}")
    return reader, metadata, reader.get_raw_topics_timestamps()


def _metadata_data_format(metadata: dict[str, Any]) -> str:
    return normalize_data_format(metadata.get("data_format"))


def _camera_topic(camera_id: str, data_format: str) -> str:
    normalize_data_format(data_format)
    return f"nmx/hal/camera/{camera_id}/rgbd"


def _camera_timestamps(
    reader: Any, raw_topics: dict[str, Any], camera_id: str, data_format: str
) -> np.ndarray:
    values = raw_topics.get(_camera_topic(camera_id, data_format))
    if values is None:
        values = []
    return np.asarray(values, dtype=np.int64)


def _iter_camera_messages(reader: Any, camera_id: str, data_format: str):
    yield from reader.iter_decoded_messages(_camera_topic(camera_id, data_format))


def _iter_lowdim_messages(reader: Any, topic: str):
    legacy_iterator = getattr(reader, "get_lowdim", None)
    if callable(legacy_iterator):
        yield from legacy_iterator(topic)
        return
    yield from reader.iter_decoded_messages(topic)


def _semantic(topic: str) -> str:
    if topic in JOINT_CANDIDATES["left"] or topic in JOINT_CANDIDATES["right"]:
        return "joint_position_rad"
    if any(topic in candidates for candidates in TCP_CANDIDATES.values()):
        return "tcp_pose_xyz_qw_qx_qy_qz"
    if any(topic in candidates for candidates in GRIPPER_CANDIDATES.values()):
        return "gripper_width"
    if topic.startswith("/lowdim/action"):
        return "action"
    return "lowdim"


def _base_inspection(
    reader: Any,
    metadata: dict[str, Any],
    raw_topics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    data_format = _metadata_data_format(metadata)
    topic_rows: list[dict[str, Any]] = []
    global_timestamps: list[int] = []
    for name in sorted(raw_topics):
        timestamps = [int(value) for value in (raw_topics.get(name) or [])]
        topic_rows.append({"name": str(name)[:300], "count": len(timestamps)})
        if timestamps:
            global_timestamps.extend((timestamps[0], timestamps[-1]))

    cameras: list[dict[str, Any]] = []
    camera_info = metadata.get("camera_info") if isinstance(metadata, dict) else {}
    if isinstance(camera_info, dict):
        for camera_id in sorted(camera_info):
            timestamps = _camera_timestamps(
                reader, raw_topics, camera_id, data_format
            ).tolist()
            source_device = camera_info.get(camera_id)
            cameras.append(
                {
                    "id": str(camera_id)[:300],
                    "source_device": (
                        str(source_device)[:300]
                        if isinstance(source_device, str)
                        else None
                    ),
                    "count": len(timestamps),
                }
            )
            if timestamps:
                global_timestamps.extend((timestamps[0], timestamps[-1]))
    return topic_rows, cameras, global_timestamps


def _nedf3_stream_values(
    reader: Any,
    topic: str,
    semantic: str,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[np.ndarray] = []
    for record in reader.iter_decoded_messages(topic):
        value: np.ndarray | None
        if semantic == "tcp_pose_xyz_qw_qx_qy_qz":
            value = np.asarray(extract_pose_wxyz(record), dtype=np.float32)
        elif semantic == "gripper_width":
            width = extract_gripper_width(record)
            value = (
                np.asarray([width], dtype=np.float32)
                if width is not None
                else None
            )
        elif semantic == "joint_position_rad":
            value = extract_joint_state_positions(record, expected_dimension=7)
        else:
            raise ValueError("unsupported_nedf3_replay_semantic")
        if value is None:
            continue
        timestamps.append(message_timestamp_ns(record))
        values.append(np.asarray(value, dtype=np.float32))
    if not values:
        return np.asarray([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    timestamps_array = np.asarray(timestamps, dtype=np.int64)
    values_array = np.asarray(values, dtype=np.float32)
    if np.any(np.diff(timestamps_array) < 0):
        order = np.argsort(timestamps_array, kind="stable")
        timestamps_array = timestamps_array[order]
        values_array = values_array[order]
    return timestamps_array, values_array


def _collect_nedf3(
    reader: Any,
    raw_topics: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[dict[str, str]]]:
    arrays: dict[str, np.ndarray] = {
        "cache_schema_version": np.asarray([1], dtype=np.int16),
    }
    capabilities: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    topics: list[tuple[str, str]] = []
    for side in ("left", "right"):
        topics.extend(
            (
                (NEDF3_ARM_TOPICS[side]["joint/state"], "joint_position_rad"),
                (
                    NEDF3_ARM_TOPICS[side]["tcp/state"],
                    "tcp_pose_xyz_qw_qx_qy_qz",
                ),
                (NEDF3_HAND_TOPICS[side]["state"], "gripper_width"),
            )
        )
    total_bytes = int(arrays["cache_schema_version"].nbytes)
    for index, (topic, semantic) in enumerate(topics):
        capability: dict[str, Any] = {
            "topic": topic,
            "semantic": semantic,
            "available": False,
            "count": 0,
            "dimension": None,
            "timestamp_key": None,
            "value_key": None,
            "reason": None,
        }
        capabilities[topic] = capability
        if not raw_topics.get(topic):
            capability["reason"] = "empty_topic"
            continue
        try:
            timestamps, values = _nedf3_stream_values(
                reader, topic, semantic
            )
        except (TypeError, ValueError, MemoryError):
            capability["reason"] = "decode_failed"
            issues.append(
                _issue(
                    "unavailable_nedf3_topic",
                    f"Recorded topic {topic[:120]} cannot be cached for replay",
                )
            )
            continue
        count = int(timestamps.size)
        dimension = int(values.shape[1]) if values.ndim == 2 and count else None
        capability["count"] = count
        capability["dimension"] = dimension
        if count == 0 or dimension is None:
            capability["reason"] = "no_semantic_values"
            continue
        if count > MAX_LOWDIM_FRAMES:
            capability["reason"] = "frame_limit_exceeded"
            issues.append(
                _issue(
                    "lowdim_frame_limit",
                    f"Recorded topic {topic[:120]} exceeds the replay frame limit",
                    "error",
                )
            )
            continue
        projected = total_bytes + int(timestamps.nbytes) + int(values.nbytes)
        if projected > MAX_TIMELINE_BYTES:
            capability["reason"] = "timeline_size_limit_exceeded"
            issues.append(
                _issue(
                    "timeline_size_limit",
                    "NEDF3 replay timeline exceeds the cache limit",
                    "error",
                )
            )
            continue
        timestamp_key = f"topic_{index:03d}_timestamps_ns"
        value_key = f"topic_{index:03d}_values"
        arrays[timestamp_key] = timestamps
        arrays[value_key] = values
        total_bytes = projected
        capability.update(
            {
                "available": True,
                "timestamp_key": timestamp_key,
                "value_key": value_key,
                "reason": None,
            }
        )
    return arrays, capabilities, issues


def _collect_lowdim(
    reader: Any,
    raw_topics: dict[str, Any],
    data_format: str = NEDF3_DATA_FORMAT,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[dict[str, str]]]:
    if data_format == NEDF3_DATA_FORMAT:
        return _collect_nedf3(reader, raw_topics)
    arrays: dict[str, np.ndarray] = {
        "cache_schema_version": np.asarray([1], dtype=np.int16),
    }
    capabilities: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    topics = [str(name) for name in sorted(raw_topics) if str(name).startswith("/lowdim/")]
    if len(topics) > MAX_LOWDIM_TOPICS:
        issues.append(_issue("lowdim_topic_limit", "Low-dimensional topic count exceeds the replay cache limit", "error"))
        topics = topics[:MAX_LOWDIM_TOPICS]

    total_bytes = int(arrays["cache_schema_version"].nbytes)
    for index, topic in enumerate(topics):
        timestamps = np.asarray([int(value) for value in (raw_topics.get(topic) or [])], dtype=np.int64)
        capability: dict[str, Any] = {
            "topic": topic[:300],
            "semantic": _semantic(topic),
            "available": False,
            "count": int(timestamps.size),
            "dimension": None,
            "timestamp_key": None,
            "value_key": None,
            "reason": None,
        }
        capabilities[topic] = capability
        if timestamps.size == 0:
            capability["reason"] = "empty_topic"
            continue
        if timestamps.size > MAX_LOWDIM_FRAMES:
            capability["reason"] = "frame_limit_exceeded"
            issues.append(_issue("lowdim_frame_limit", f"Recorded topic {topic[:120]} exceeds the replay frame limit", "error"))
            continue
        monotonic = bool(np.all(timestamps[1:] >= timestamps[:-1]))
        if not monotonic:
            capability["reason"] = "nonmonotonic_timestamps"
            issues.append(_issue("nonmonotonic_lowdim", f"Recorded topic {topic[:120]} has nonmonotonic timestamps", "error"))
            continue

        dimension: int | None = None
        invalid_reason: str | None = None
        values_array: np.ndarray | None = None
        try:
            iterator = iter(_iter_lowdim_messages(reader, topic))
            first_values = _values(next(iterator))
            dimension = len(first_values)
            if not 1 <= dimension <= MAX_LOWDIM_DIMENSION:
                invalid_reason = "dimension_limit_exceeded"
            estimated = int(timestamps.nbytes) + int(timestamps.size) * int(dimension) * 4
            if invalid_reason is None and total_bytes + estimated > MAX_TIMELINE_BYTES:
                invalid_reason = "timeline_size_limit_exceeded"
            if invalid_reason is None:
                values_array = np.empty((int(timestamps.size), int(dimension)), dtype=np.float32)
                values_array[0] = first_values
                for row_index in range(1, int(timestamps.size)):
                    values = _values(next(iterator))
                    if len(values) != dimension:
                        invalid_reason = "inconsistent_dimension"
                        break
                    values_array[row_index] = values
                if invalid_reason is None:
                    sentinel = object()
                    if next(iterator, sentinel) is not sentinel:
                        invalid_reason = "message_count_mismatch"
        except StopIteration:
            invalid_reason = "message_count_mismatch"
        except (TypeError, ValueError, MemoryError):
            invalid_reason = "decode_failed"

        if invalid_reason is not None or dimension is None or values_array is None:
            capability["reason"] = invalid_reason or "empty_topic"
            issues.append(_issue("unavailable_lowdim_topic", f"Recorded topic {topic[:120]} cannot be cached for replay"))
            continue

        projected = total_bytes + int(timestamps.nbytes) + int(values_array.nbytes)
        if projected > MAX_TIMELINE_BYTES:
            capability["reason"] = "timeline_size_limit_exceeded"
            issues.append(_issue("timeline_size_limit", "Low-dimensional replay timeline exceeds the cache limit", "error"))
            continue
        timestamp_key = f"topic_{index:03d}_timestamps_ns"
        value_key = f"topic_{index:03d}_values"
        arrays[timestamp_key] = timestamps
        arrays[value_key] = values_array
        total_bytes = projected
        capability.update(
            {
                "available": True,
                "dimension": int(dimension),
                "timestamp_key": timestamp_key,
                "value_key": value_key,
                "reason": None,
            }
        )
    return arrays, capabilities, issues


def _joint_info(
    capabilities: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    result: dict[str, dict[str, Any]] = {}
    for side, candidates in JOINT_CANDIDATES.items():
        topic = next((name for name in candidates if name in capabilities and capabilities[name]["count"]), None)
        capability = capabilities.get(topic) if topic else None
        dimension = capability.get("dimension") if capability else None
        available = bool(capability and capability.get("available") and dimension == 7)
        result[side] = {
            "available": available,
            "topic": topic,
            "count": int(capability.get("count") or 0) if capability else 0,
            "dimension": dimension,
        }
        if topic is None:
            issues.append(_issue(f"missing_joint_{side}", f"Recorded {side} 7D joint topic is missing"))
        elif dimension != 7:
            issues.append(_issue(f"invalid_joint_dimension_{side}", f"Recorded {side} joint topic is not consistently 7D", "error"))
        elif not available:
            issues.append(_issue(f"unavailable_joint_{side}", f"Recorded {side} joint topic cannot be replayed", "error"))

    if result["left"]["available"] and result["right"]["available"]:
        left_capability = capabilities[result["left"]["topic"]]
        right_capability = capabilities[result["right"]["topic"]]
        left_timestamps = arrays[left_capability["timestamp_key"]]
        right_timestamps = arrays[right_capability["timestamp_key"]]
        if not np.array_equal(left_timestamps, right_timestamps):
            result["left"]["available"] = False
            result["right"]["available"] = False
            issues.append(_issue("joint_timeline_mismatch", "Left and right joint timestamps are not aligned", "error"))
    return result, issues


def _tcp_info(
    capabilities: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    result: dict[str, dict[str, Any]] = {}
    for side, candidates in TCP_CANDIDATES.items():
        topic = next(
            (
                name
                for name in candidates
                if name in capabilities and capabilities[name]["count"]
            ),
            None,
        )
        capability = capabilities.get(topic) if topic else None
        dimension = capability.get("dimension") if capability else None
        available = bool(
            capability and capability.get("available") and dimension == 7
        )
        result[side] = {
            "available": available,
            "topic": topic,
            "count": int(capability.get("count") or 0) if capability else 0,
            "dimension": dimension,
        }
        if topic is None:
            issues.append(
                _issue(
                    f"missing_tcp_{side}",
                    f"Recorded {side} 7D TCP pose topic is missing",
                )
            )
        elif dimension != 7:
            issues.append(
                _issue(
                    f"invalid_tcp_dimension_{side}",
                    f"Recorded {side} TCP topic is not consistently 7D",
                    "error",
                )
            )
        elif not available:
            issues.append(
                _issue(
                    f"unavailable_tcp_{side}",
                    f"Recorded {side} TCP topic cannot be replayed",
                    "error",
                )
            )
    return result, issues


def _gripper_info(
    capabilities: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    result: dict[str, dict[str, Any]] = {}
    for side, candidates in GRIPPER_CANDIDATES.items():
        topic = next(
            (
                name
                for name in candidates
                if name in capabilities and capabilities[name]["count"]
            ),
            None,
        )
        capability = capabilities.get(topic) if topic else None
        dimension = capability.get("dimension") if capability else None
        available = bool(
            capability and capability.get("available") and dimension == 1
        )
        result[side] = {
            "available": available,
            "topic": topic,
            "count": int(capability.get("count") or 0) if capability else 0,
            "dimension": dimension,
        }
        if topic is None:
            issues.append(
                _issue(
                    f"missing_gripper_{side}",
                    f"Recorded {side} gripper width topic is missing",
                    "info",
                )
            )
        elif dimension != 1:
            issues.append(
                _issue(
                    f"invalid_gripper_dimension_{side}",
                    f"Recorded {side} gripper topic is not consistently 1D",
                )
            )
    return result, issues


def _gripper_curve(
    grippers: dict[str, dict[str, Any]],
    capabilities: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": 1}
    for side in ("left", "right"):
        info = grippers[side]
        if not info.get("available"):
            result[side] = {
                "available": False,
                "topic": info.get("topic"),
                "timestamps_ns": [],
                "values": [],
            }
            continue
        capability = capabilities[str(info["topic"])]
        timestamps = np.asarray(
            arrays[str(capability["timestamp_key"])], dtype=np.int64
        )
        values = np.asarray(
            arrays[str(capability["value_key"])], dtype=np.float32
        )[:, 0]
        if timestamps.size > MAX_GRIPPER_CURVE_POINTS:
            indexes = np.linspace(
                0,
                timestamps.size - 1,
                MAX_GRIPPER_CURVE_POINTS,
                dtype=np.int64,
            )
            timestamps = timestamps[indexes]
            values = values[indexes]
        result[side] = {
            "available": True,
            "topic": info.get("topic"),
            "timestamps_ns": timestamps.tolist(),
            "values": values.tolist(),
        }
    return result


def _camera_fps(timestamps_ns: np.ndarray) -> int:
    if timestamps_ns.size < 2:
        return 30
    differences = np.diff(timestamps_ns)
    differences = differences[differences > 0]
    if differences.size == 0:
        return 30
    estimate = int(round(1_000_000_000 / float(np.median(differences))))
    return max(1, min(60, estimate))


def _ffmpeg_encoder(ffmpeg_bin: Path) -> tuple[str, list[str]]:
    try:
        listed = subprocess.run(
            [str(ffmpeg_bin), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("replay_ffmpeg_unavailable") from exc
    if listed.returncode != 0:
        raise ValueError("replay_ffmpeg_encoder_list_failed")
    available = {
        match.group(1)
        for line in listed.stdout.splitlines()
        if (match := re.match(r"^\s*[A-Z.]{6}\s+(\S+)", line))
    }
    if "h264_nvenc" in available:
        smoke = subprocess.run(
            [
                str(ffmpeg_bin),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        if smoke.returncode == 0:
            return "h264_nvenc", [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "24",
                "-b:v",
                "0",
            ]
    if "libx264" in available:
        return "libx264", [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "24",
            "-threads",
            "1",
        ]
    raise ValueError("replay_h264_encoder_unavailable")


def _write_camera_video(
    reader: Any,
    camera_id: str,
    timestamps_ns: np.ndarray,
    output: Path,
    ffmpeg_bin: Path,
    encoder_args: list[str],
    data_format: str = NEDF3_DATA_FORMAT,
) -> int:
    if timestamps_ns.size == 0 or timestamps_ns.size > MAX_CAMERA_FRAMES:
        raise ValueError("replay_camera_frame_limit")
    fps = _camera_fps(timestamps_ns)
    command = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-vcodec",
        "mjpeg",
        "-i",
        "pipe:0",
        "-an",
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    frames = 0
    try:
        for message in _iter_camera_messages(reader, camera_id, data_format):
            if data_format == NEDF3_DATA_FORMAT:
                payload = extract_rgb_jpeg(message)
            else:
                decoded = getattr(message, "decoded_message", message)
                encoded = getattr(decoded, "data", None)
                if encoded is None:
                    raise ValueError("replay_camera_payload_missing")
                payload = bytes(encoded)
            if not payload or len(payload) > MAX_JPEG_BYTES:
                raise ValueError("replay_camera_payload_invalid")
            process.stdin.write(payload)
            frames += 1
            if frames > MAX_CAMERA_FRAMES:
                raise ValueError("replay_camera_frame_limit")
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise
    if return_code != 0:
        output.unlink(missing_ok=True)
        raise ValueError(
            "replay_video_encode_failed:"
            + stderr.decode("utf-8", errors="replace")[-300:]
        )
    if frames != int(timestamps_ns.size):
        output.unlink(missing_ok=True)
        raise ValueError("replay_camera_count_mismatch")
    try:
        info = output.lstat()
    except OSError as exc:
        raise ValueError("replay_video_output_missing") from exc
    if output.is_symlink() or not output.is_file() or info.st_size <= 0:
        output.unlink(missing_ok=True)
        raise ValueError("replay_video_output_invalid")
    return fps


def _encode_replay_videos(
    reader: Any,
    metadata: dict[str, Any],
    raw_topics: dict[str, Any],
    camera_roles: dict[str, str],
    output_dir: Path,
    ffmpeg_bin: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    data_format = _metadata_data_format(metadata)
    output_dir = output_dir.resolve(strict=True)
    output_info = output_dir.lstat()
    if output_dir.is_symlink() or not output_dir.is_dir() or output_info.st_nlink < 1:
        raise ValueError("unsafe_video_output_directory")
    camera_info = metadata.get("camera_info")
    camera_ids = set(camera_info) if isinstance(camera_info, dict) else set()
    resolved_roles = resolve_recorded_camera_ids(
        camera_info,
        camera_roles,
        data_format=data_format,
    )
    videos: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    try:
        resolved_ffmpeg = ffmpeg_bin.resolve(strict=True)
        encoder, encoder_args = _ffmpeg_encoder(resolved_ffmpeg)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        issues.append(
            _issue(
                "replay_video_encoder_unavailable",
                f"{data_format} video cache cannot be encoded: {exc}",
            )
        )
        return videos, issues
    for role in CAMERA_ROLES:
        camera_id = resolved_roles.get(role)
        if not camera_id or camera_id not in camera_ids:
            continue
        timestamps = _camera_timestamps(
            reader,
            raw_topics,
            camera_id,
            data_format,
        )
        if timestamps.size == 0:
            continue
        output = output_dir / f"video_{role}.mp4"
        try:
            fps = _write_camera_video(
                reader,
                camera_id,
                timestamps,
                output,
                resolved_ffmpeg,
                encoder_args,
                data_format,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            output.unlink(missing_ok=True)
            issues.append(
                _issue(
                    f"replay_video_{role}_failed",
                    f"Recorded {role} video cache could not be generated: {exc}",
                )
            )
            continue
        videos[role] = {
            "available": True,
            "role": role,
            "camera_id": camera_id,
            "filename": output.name,
            "frame_count": int(timestamps.size),
            "fps": fps,
            "start_timestamp_ns": int(timestamps[0]),
            "end_timestamp_ns": int(timestamps[-1]),
            "codec": "h264",
            "encoder": encoder,
        }
    return videos, issues


def inspect_snapshot(
    episode: Path,
    timeline_output: Path,
    *,
    video_output_dir: Path | None = None,
    ffmpeg_bin: Path | None = None,
    camera_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    reader, metadata, raw_topics = _reader_and_topics(episode)
    data_format = _metadata_data_format(metadata)
    topic_rows, cameras, global_timestamps = _base_inspection(reader, metadata, raw_topics)
    arrays, capabilities, issues = _collect_lowdim(
        reader,
        raw_topics,
        data_format=data_format,
    )
    joints, joint_issues = _joint_info(capabilities, arrays)
    tcp, tcp_issues = _tcp_info(capabilities)
    grippers, gripper_issues = _gripper_info(capabilities)
    issues.extend(joint_issues)
    issues.extend(tcp_issues)
    issues.extend(gripper_issues)

    replay_videos: dict[str, Any] = {}
    if video_output_dir is not None and ffmpeg_bin is not None:
        replay_videos, video_issues = _encode_replay_videos(
            reader,
            metadata,
            raw_topics,
            dict(camera_roles or {}),
            video_output_dir,
            ffmpeg_bin,
        )
        issues.extend(video_issues)

    try:
        output_info = timeline_output.lstat()
        if timeline_output.is_symlink() or not timeline_output.is_file() or output_info.st_nlink != 1:
            raise ValueError("unsafe_timeline_output")
    except OSError as exc:
        raise ValueError("timeline_output_unavailable") from exc
    with timeline_output.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())

    return {
        "schema_version": 1,
        "data_format": data_format,
        "time_range_ns": {
            "start": min(global_timestamps) if global_timestamps else None,
            "end": max(global_timestamps) if global_timestamps else None,
        },
        "topics": topic_rows,
        "cameras": cameras,
        "joints": joints,
        "tcp": tcp,
        "grippers": grippers,
        "gripper_curve": _gripper_curve(grippers, capabilities, arrays),
        "replay_videos": replay_videos,
        "capabilities": {
            "schema_version": 1,
            "topics": capabilities,
            "urdf_replay": bool(joints["left"]["available"] and joints["right"]["available"]),
            "tcp_replay": bool(tcp["left"]["available"] or tcp["right"]["available"]),
        },
        "timeline": {
            "schema_version": 1,
            "array_count": len(arrays),
            "uncompressed_bytes": sum(int(value.nbytes) for value in arrays.values()),
        },
        "issues": issues,
    }


def inspect_episode(episode: Path) -> dict[str, Any]:
    """Legacy summary-only entry point retained for operators and tests."""
    reader, metadata, raw_topics = _reader_and_topics(episode)
    data_format = _metadata_data_format(metadata)
    topic_rows, cameras, global_timestamps = _base_inspection(reader, metadata, raw_topics)
    arrays, capabilities, issues = _collect_lowdim(
        reader,
        raw_topics,
        data_format=data_format,
    )
    joint_info, joint_issues = _joint_info(capabilities, arrays)
    tcp_info, tcp_issues = _tcp_info(capabilities)
    gripper_info, gripper_issues = _gripper_info(capabilities)
    issues.extend(joint_issues)
    issues.extend(tcp_issues)
    issues.extend(gripper_issues)
    return {
        "schema_version": 1,
        "data_format": data_format,
        "time_range_ns": {"start": min(global_timestamps) if global_timestamps else None, "end": max(global_timestamps) if global_timestamps else None},
        "topics": topic_rows,
        "cameras": cameras,
        "joints": joint_info,
        "tcp": tcp_info,
        "grippers": gripper_info,
        "gripper_curve": _gripper_curve(gripper_info, capabilities, arrays),
        "issues": issues,
    }


def inspect_trajectory(
    episode: Path, *, start: int, end: int | None, stride: int, max_points: int
) -> dict[str, Any]:
    """Legacy bounded trajectory entry point; the web service uses snapshot mode."""
    reader, metadata, raw_topics = _reader_and_topics(episode)
    data_format = _metadata_data_format(metadata)
    arrays, capabilities, _issues = _collect_lowdim(
        reader,
        raw_topics,
        data_format=data_format,
    )
    joints, _joint_issues = _joint_info(capabilities, arrays)
    if not joints["left"]["available"] or not joints["right"]["available"]:
        raise ValueError("joint_trajectory_unavailable")
    selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in ("left", "right"):
        capability = capabilities[str(joints[side]["topic"])]
        selected[side] = (
            np.asarray(arrays[str(capability["timestamp_key"])], dtype=np.int64),
            np.asarray(arrays[str(capability["value_key"])], dtype=np.float32),
        )
    left_timestamps, left_values = selected["left"]
    right_timestamps, right_values = selected["right"]
    if not np.array_equal(left_timestamps, right_timestamps):
        raise ValueError("joint_timeline_mismatch")
    total = int(left_timestamps.size)
    stop = total if end is None else min(end, total)
    indices = np.arange(min(start, total), stop, stride, dtype=np.int64)
    if int(indices.size) > max_points:
        indices = indices[:: math.ceil(int(indices.size) / max_points)]
    return {
        "schema_version": 1,
        "data_format": data_format,
        "start": start,
        "end": stop,
        "stride": stride,
        "total_frames": total,
        "returned_frames": int(indices.size),
        "timestamps_ns": left_timestamps[indices].tolist(),
        "left": left_values[indices].tolist(),
        "right": right_values[indices].tolist(),
        "issues": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--mode", choices=("summary", "trajectory", "snapshot"), default="summary")
    parser.add_argument("--timeline-output", type=Path)
    parser.add_argument("--video-output-dir", type=Path)
    parser.add_argument("--ffmpeg-bin", type=Path)
    parser.add_argument("--camera-role", action="append", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=5000)
    args = parser.parse_args()
    episode = args.episode.resolve(strict=True)
    if not episode.is_dir() or not (episode / "metadata.json").is_file():
        raise SystemExit(2)
    try:
        if args.mode == "snapshot":
            if args.timeline_output is None:
                raise ValueError("timeline_output_required")
            camera_roles: dict[str, str] = {}
            for raw in args.camera_role:
                role, separator, camera_id = str(raw).partition("=")
                if (
                    not separator
                    or role not in CAMERA_ROLES
                    or CAMERA_ID.fullmatch(camera_id) is None
                    or role in camera_roles
                ):
                    raise ValueError("invalid_camera_role")
                camera_roles[role] = camera_id
            if (args.video_output_dir is None) != (args.ffmpeg_bin is None):
                raise ValueError("video_output_and_ffmpeg_must_be_paired")
            result = inspect_snapshot(
                episode,
                args.timeline_output,
                video_output_dir=args.video_output_dir,
                ffmpeg_bin=args.ffmpeg_bin,
                camera_roles=camera_roles,
            )
        elif args.mode == "trajectory":
            if args.start < 0 or args.stride < 1 or not 1 <= args.max_points <= 5000:
                raise ValueError("invalid_trajectory_range")
            if args.end is not None and args.end < args.start:
                raise ValueError("invalid_trajectory_range")
            result = inspect_trajectory(episode, start=args.start, end=args.end, stride=args.stride, max_points=args.max_points)
        else:
            result = inspect_episode(episode)
    except Exception:
        # The parent deliberately maps all SDK failures to a sanitized error.
        raise SystemExit(3)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
