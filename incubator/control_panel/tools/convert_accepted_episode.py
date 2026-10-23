#!/usr/bin/env python3
"""Convert one finalized NEDF3 episode into a LeRobot shard.

The control-panel scheduler owns the private job spool and invokes this module
with trusted paths.  This process is intentionally separate from the recorder:
no image decoding, video encoding, or source hashing happens on the recorder's
hot path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from incubator.control_panel.converter_profile import default_profile
from incubator.control_panel.recording_contract import (
    LEROBOT_DEVICE_TYPE,
    LEROBOT_FPS,
    NEDF3_CAMERA_IDS,
    NEDF3_CAMERA_ROLE_LABELS,
    NEDF3_CAMERA_TOPICS,
    NEDF3_DATA_FORMAT,
    NEDF3_RECORD_TOPIC_LIST,
    normalize_data_format,
)


SCHEMA_VERSION = 1
CONVERTER_SHA256 = "f9bd73384fae71938668d39af9b36d4883880b50cf7fe7b902f3c2c8162cbc11"
CONVERTER_SCRIPT = Path(__file__).with_name("nedf3_rollout_to_lerobot.py")
MERGER_SHA256 = "966057a762817f596be9fd8a1d813631f1faf4bbe1a5b1f4c9b30dc7fcee5c28"
MERGER_SCRIPT = Path(__file__).with_name("merge_lerobot_shards.py")
PROFILE = default_profile()
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
MAX_JOB_BYTES = 4 * 1024 * 1024


class JobSpecError(RuntimeError):
    """The durable conversion job is unsafe, stale, or internally inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, max_bytes: int = MAX_JOB_BYTES) -> dict[str, Any]:
    info = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise JobSpecError(f"JSON input is not a regular file: {path}")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise JobSpecError(f"JSON input has invalid size: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobSpecError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise JobSpecError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, max_bytes: int = MAX_JOB_BYTES) -> list[dict[str, Any]]:
    info = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise JobSpecError(f"JSONL input is not a regular file: {path}")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise JobSpecError(f"JSONL input has invalid size: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JobSpecError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise JobSpecError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _require_safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise JobSpecError(f"Invalid {field}")
    return value


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JobSpecError(f"{field} must be an object")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JobSpecError(f"{field} must be a non-negative integer")
    return value


def _verified_source(job: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    if job.get("schema_version") != SCHEMA_VERSION:
        raise JobSpecError("Unsupported conversion job schema_version")
    identity = _require_mapping(job.get("identity"), "identity")
    source = _require_mapping(job.get("source"), "source")
    profile = _require_mapping(job.get("profile"), "profile")
    job_id = _require_safe_id(job.get("job_id"), "job_id")
    if len(job_id) != 64 or any(char not in "0123456789abcdef" for char in job_id):
        raise JobSpecError("job_id must be a lowercase SHA-256 hex digest")
    profile_id = _require_safe_id(profile.get("id"), "profile.id")
    if identity.get("profile_id") != profile_id:
        raise JobSpecError("identity.profile_id does not match profile.id")
    if profile_id != PROFILE.profile_id:
        raise JobSpecError("job does not use the installed trusted converter profile")
    activation_probe = profile.get("activation_probe") is True
    if profile.get("available") is not True and not activation_probe:
        blocker = profile.get("blocker") or "conversion profile is unavailable"
        raise JobSpecError(str(blocker))
    if activation_probe and profile.get("can_probe") is not True:
        raise JobSpecError("activation probe is not authorized by the trusted scheduler")
    source_fingerprint = _require_safe_id(
        identity.get("source_fingerprint"), "identity.source_fingerprint"
    )
    if source.get("fingerprint") != source_fingerprint:
        raise JobSpecError("source fingerprint does not match identity")
    source_result = source.get("result")
    if source_result not in {"success", "failure", "incomplete"}:
        raise JobSpecError("Only finalized episodes may be converted")
    _require_safe_id(source.get("episode_id"), "source.episode_id")
    _require_nonnegative_int(source.get("episode_index"), "source.episode_index")

    source_path_raw = source.get("path")
    source_root_raw = source.get("root")
    if not isinstance(source_path_raw, str) or not isinstance(source_root_raw, str):
        raise JobSpecError("source.path and source.root must be strings")
    source_path = Path(source_path_raw).expanduser()
    source_root = Path(source_root_raw).expanduser()
    if not source_path.is_absolute() or not source_root.is_absolute():
        raise JobSpecError("source.path and source.root must be absolute")
    resolved_root = source_root.resolve(strict=True)
    resolved_source = source_path.resolve(strict=True)
    try:
        resolved_source.relative_to(resolved_root)
    except ValueError as exc:
        raise JobSpecError("source.path escapes source.root") from exc
    if not resolved_source.is_dir() or resolved_source.name.endswith("_temp"):
        raise JobSpecError("source episode is not a finalized directory")

    metadata_path = resolved_source / "metadata.json"
    result_path = resolved_source / (
        "episode_incomplete.json" if source_result == "incomplete" else "episode_result.json"
    )
    metadata = _read_json(metadata_path)
    result = _read_json(result_path)
    try:
        data_format = normalize_data_format(metadata.get("data_format"))
    except ValueError as exc:
        raise JobSpecError(str(exc)) from exc
    if data_format != NEDF3_DATA_FORMAT:
        raise JobSpecError("source metadata is not a supported NEDF format")
    if activation_probe and data_format != NEDF3_DATA_FORMAT:
        raise JobSpecError("converter activation probe requires a canonical NEDF3 source")
    if data_format == NEDF3_DATA_FORMAT:
        try:
            uuid.UUID(str(source.get("episode_id")))
        except ValueError as exc:
            raise JobSpecError("canonical NEDF3 source episode_id must be a UUID") from exc
    declared_format = source.get("data_format")
    if declared_format is not None and declared_format != data_format:
        raise JobSpecError("source data_format changed after enqueue")
    if result.get("episode_result") != source_result:
        raise JobSpecError("source result metadata changed after enqueue")
    if bool(result.get("is_success")) is not (source_result == "success"):
        raise JobSpecError("source success flag does not match its finalized result")
    if result.get("episode_id") != source.get("episode_id"):
        raise JobSpecError("source episode_id changed after enqueue")

    expected_hashes = {
        metadata_path: source.get("metadata_sha256"),
        result_path: source.get(
            "episode_incomplete_sha256"
            if source_result == "incomplete"
            else "episode_result_sha256"
        ),
    }
    dense_path = resolved_source / "dense_labels.jsonl"
    if dense_path.is_file():
        expected_hashes[dense_path] = source.get("dense_labels_sha256")
    elif source.get("dense_labels_sha256") not in (None, ""):
        raise JobSpecError("dense_labels.jsonl disappeared after enqueue")
    for path, expected in expected_hashes.items():
        if expected and _sha256_file(path) != expected:
            raise JobSpecError(f"Source fingerprint component changed: {path.name}")
    verified_source = dict(source)
    verified_source["data_format"] = data_format
    return resolved_source, verified_source


def _positive_int(value: Any, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobSpecError(f"{field} must be a positive integer")
    return value


def _integer_tuple(
    value: Any, field: str, length: int, default: Sequence[int], *, positive: bool
) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise JobSpecError(f"{field} must contain {length} integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise JobSpecError(f"{field} must contain integers")
        if positive and item <= 0:
            raise JobSpecError(f"{field} values must be positive")
        if not positive and item < 0:
            raise JobSpecError(f"{field} values cannot be negative")
        result.append(item)
    return tuple(result)


def _optional_camera(value: Any, field: str) -> str:
    if value is None:
        return "None"
    if not isinstance(value, str):
        raise JobSpecError(f"{field} must be null or a non-empty camera id")
    normalized = value.strip()
    if not normalized:
        raise JobSpecError(
            f"{field} must be null or a non-empty camera id; blank strings are invalid"
        )
    if normalized.lower() == "none":
        raise JobSpecError(f"{field} must use JSON null, not the string {value!r}")
    return normalized


def _required_camera(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise JobSpecError(f"{field} must be a non-empty camera id")
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        raise JobSpecError(f"{field} must be a non-empty camera id")
    return normalized


def _converter_command(
    job: Mapping[str, Any], source_path: Path, staging_dir: Path
) -> list[str]:
    run = _require_mapping(job.get("run_config"), "run_config")
    fps = _positive_int(run.get("fps"), "run_config.fps", LEROBOT_FPS)
    if fps != LEROBOT_FPS:
        raise JobSpecError(
            f"run_config.fps must be {LEROBOT_FPS} for the LeRobot contract"
        )
    command = [
        sys.executable,
        str(CONVERTER_SCRIPT),
        "--data-dir",
        str(source_path),
        "--dataset-name",
        "shard",
        "--output-dir",
        str(staging_dir),
        "--head-camera",
        _optional_camera(run.get("head_camera"), "run_config.head_camera"),
        "--left-camera",
        _required_camera(run.get("left_camera"), "run_config.left_camera"),
        "--right-camera",
        _required_camera(run.get("right_camera"), "run_config.right_camera"),
        "--action-topic",
        str(run.get("action_topic") or "/lowdim/action/rizonrobot_2_control"),
        "--device-type",
        str(run.get("device_type") or LEROBOT_DEVICE_TYPE),
        "--fps",
        str(fps),
        "--workers",
        "1",
        "--camera-workers",
        str(
            min(
                _positive_int(
                    run.get("camera_workers"), "run_config.camera_workers", 1
                ),
                3,
            )
        ),
        "--wrist-image-size",
        *map(
            str,
            _integer_tuple(
                run.get("wrist_image_size"),
                "run_config.wrist_image_size",
                2,
                (320, 240),
                positive=True,
            ),
        ),
        "--head-image-size",
        *map(
            str,
            _integer_tuple(
                run.get("head_image_size"),
                "run_config.head_image_size",
                2,
                (240, 320),
                positive=True,
            ),
        ),
        "--crop-lrtb",
        *map(
            str,
            _integer_tuple(
                run.get("crop_lrtb"),
                "run_config.crop_lrtb",
                4,
                (185, 185, 0, 0),
                positive=False,
            ),
        ),
        "--tcp-pose-macro",
        str(run.get("tcp_pose_macro") or "auto"),
        "--max-alignment-error-ms",
        str(float(run.get("max_alignment_error_ms", 50.0))),
        "--video-encoder",
        PROFILE.video_encoder,
        "--video-quality",
        str(PROFILE.video_quality),
        "--video-gop",
        str(PROFILE.video_gop),
        "--svt-preset",
        str(PROFILE.svt_preset),
        "--nvenc-preset",
        PROFILE.nvenc_preset,
        "--ffmpeg-bin",
        str(PROFILE.ffmpeg),
        "--ffprobe-bin",
        str(PROFILE.ffprobe),
        "--fail-on-conversion-error",
    ]
    if _require_mapping(job.get("source"), "source").get("result") == "success":
        command.append("--success-only")
    task = run.get("task")
    if task is not None:
        if not isinstance(task, str) or not task.strip():
            raise JobSpecError("run_config.task must be a non-empty string")
        command.extend(["--task", task.strip()])
    command.extend(["--gpu-ids", ",".join(map(str, PROFILE.gpu_ids))])
    return command


def _verify_vendor_script() -> None:
    for label, path, expected in (
        ("converter", CONVERTER_SCRIPT, CONVERTER_SHA256),
        ("merger", MERGER_SCRIPT, MERGER_SHA256),
    ):
        if not path.is_file() or path.is_symlink():
            raise JobSpecError(f"Vendored {label} is missing or unsafe: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise JobSpecError(
                f"Vendored {label} checksum mismatch: {actual} != {expected}"
            )


def _copy_source_sidecars(source_path: Path, shard_root: Path) -> list[str]:
    destination = shard_root / "meta" / "source_sidecars"
    for name in (
        "metadata.json",
        "episode_result.json",
        "episode_incomplete.json",
        "dense_labels.jsonl",
    ):
        source = source_path / name
        if not source.is_file():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / name
        shutil.copy2(source, target)
    if not destination.is_dir():
        return []
    return sorted(
        path.relative_to(shard_root).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    )


def _shard_artifact_files(shard_root: Path) -> list[Path]:
    manifest_path = shard_root / "meta" / "shard_manifest.json"
    files: list[Path] = []
    for path in sorted(shard_root.rglob("*")):
        if path.is_symlink():
            raise JobSpecError(f"shard artifact cannot be a symbolic link: {path}")
        if path.is_file() and path != manifest_path:
            files.append(path)
    return files


def _artifact_records(shard_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(shard_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _shard_artifact_files(shard_root)
    ]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        digest.update(_sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


@contextmanager
def _isolated_sdk_source(source: Path, staging: Path):
    """Expose a disposable source view for SDKs that write read caches."""
    shadow = staging / "source" / source.name
    if shadow.exists() or shadow.is_symlink():
        raise JobSpecError(f"SDK source staging already exists: {shadow}")
    shadow.mkdir(parents=True, mode=0o700)
    try:
        for candidate in sorted(source.rglob("*")):
            if candidate.is_symlink():
                raise JobSpecError(f"Source episode contains a symlink: {candidate}")
            relative = candidate.relative_to(source)
            destination = shadow / relative
            if candidate.is_dir():
                destination.mkdir(mode=0o700)
                continue
            if not candidate.is_file():
                raise JobSpecError(f"Source episode contains a non-file: {candidate}")
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if candidate.suffix.casefold() == ".mcap":
                try:
                    os.link(candidate, destination, follow_symlinks=False)
                except OSError as exc:
                    raise JobSpecError(
                        "NEDF conversion staging must share a filesystem with the "
                        "recording root so MCAP can be hardlinked without copying"
                    ) from exc
            else:
                shutil.copyfile(candidate, destination, follow_symlinks=False)
        yield shadow
    finally:
        shutil.rmtree(staging / "source", ignore_errors=True)


def _validate_nedf3_source_contract(source_path: Path) -> dict[str, Any]:
    metadata = _read_json(source_path / "metadata.json")
    try:
        data_format = normalize_data_format(metadata.get("data_format"))
    except ValueError as exc:
        raise JobSpecError(str(exc)) from exc
    if data_format != NEDF3_DATA_FORMAT:
        raise JobSpecError("activation source is not NEDF3")
    declared_topics = tuple(metadata.get("record_topic_list") or ())
    if declared_topics != NEDF3_RECORD_TOPIC_LIST:
        raise JobSpecError(
            "activation NEDF3 source does not declare the exact R1 HAL contract"
        )
    expected_camera_info = {
        NEDF3_CAMERA_IDS[role]: NEDF3_CAMERA_ROLE_LABELS[role]
        for role in ("right", "left", "head")
    }
    camera_info = metadata.get("camera_info")
    expected_wrist_camera_info = {
        NEDF3_CAMERA_IDS[role]: NEDF3_CAMERA_ROLE_LABELS[role]
        for role in ("right", "left")
    }
    if camera_info not in (expected_wrist_camera_info, expected_camera_info):
        raise JobSpecError(
            "activation NEDF3 source must use camera_0=right and "
            "camera_1=left, with optional camera_2=head"
        )
    if metadata.get("main_camera") != NEDF3_CAMERA_IDS["right"]:
        raise JobSpecError("activation NEDF3 main_camera must be camera_0")

    try:
        from dm3_core.nedf import NEDFFactory, NEDFReaderConfig

        reader = NEDFFactory.get_reader(
            NEDFReaderConfig(
                metadata_file_path=str(source_path / "metadata.json")
            )
        )
        raw_topics = reader.get_raw_topics_timestamps()
    except Exception as exc:
        raise JobSpecError(f"cannot reopen activation NEDF3 source: {exc}") from exc

    declared_data_topics = set(NEDF3_RECORD_TOPIC_LIST)
    required_data_topics = {
        topic for topic in declared_data_topics if not topic.endswith("/tcp/vel")
    }
    # The R1 HAL declaration is stable across configs, while the head camera
    # and Yam streams are optional at runtime.  Wrist, arm, and hand streams
    # remain mandatory for a dual-arm rollout activation probe.
    optional_head_topics = {
        topic
        for topic in declared_data_topics
        if topic.startswith("nmx/hal/head/")
    }
    if NEDF3_CAMERA_IDS["head"] not in camera_info:
        optional_head_topics.add(NEDF3_CAMERA_TOPICS["head"])
    required_data_topics -= optional_head_topics
    if not required_data_topics.issubset(raw_topics) or not set(raw_topics).issubset(
        declared_data_topics
    ):
        missing = sorted(required_data_topics - set(raw_topics))
        unexpected = sorted(set(raw_topics) - declared_data_topics)
        raise JobSpecError(
            "activation NEDF3 data topics do not match the R1 HAL contract "
            f"contract; missing={missing}, unexpected={unexpected}"
        )
    stream_evidence: dict[str, dict[str, int]] = {}
    for topic in sorted(raw_topics):
        try:
            timestamps = [int(value) for value in raw_topics[topic]]
        except (TypeError, ValueError) as exc:
            raise JobSpecError(f"invalid timestamps for NEDF3 topic {topic}") from exc
        if not timestamps:
            raise JobSpecError(f"activation NEDF3 topic is empty: {topic}")
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            raise JobSpecError(f"activation NEDF3 timestamps regress: {topic}")
        try:
            decoded_count = sum(1 for _ in reader.iter_decoded_messages(topic))
        except Exception as exc:
            raise JobSpecError(f"cannot decode activation NEDF3 topic {topic}: {exc}") from exc
        if decoded_count != len(timestamps):
            raise JobSpecError(
                f"activation NEDF3 decoded count mismatch for {topic}: "
                f"{decoded_count} != {len(timestamps)}"
            )
        stream_evidence[topic] = {
            "count": decoded_count,
            "first_timestamp_ns": timestamps[0],
            "last_timestamp_ns": timestamps[-1],
        }
    return {
        "data_format": NEDF3_DATA_FORMAT,
        "declared_topic_count": len(declared_topics),
        "actual_stream_count": len(stream_evidence),
        "camera_mapping": camera_info,
        "streams": stream_evidence,
    }


def _validate_dataset_media(root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    from incubator.control_panel.tools.nedf3_rollout_to_lerobot import (
        validate_dataset,
    )

    try:
        # Rollout NEDF3 stores the commanded action stream.  It is not a
        # synthetic next-observation target, so validation must preserve the
        # same contract used when the shard is produced.
        strict_contract = validate_dataset(
            root,
            str(PROFILE.ffprobe),
            require_next_state=False,
        )
    except Exception as exc:
        raise JobSpecError(f"strict LeRobot contract validation failed: {exc}") from exc

    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise JobSpecError("activation output has no Parquet files")
    rows = 0
    parquet_hashes = {}
    for path in parquet_files:
        table = pq.read_table(path)
        if table.num_rows <= 0:
            raise JobSpecError(f"activation Parquet is empty: {path}")
        rows += table.num_rows
        parquet_hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    video_files = sorted(root.rglob("*.mp4"))
    if not video_files:
        raise JobSpecError("activation output has no videos")
    video_hashes = {}
    for path in video_files:
        probe = subprocess.run(
            [
                str(PROFILE.ffprobe), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt,nb_frames,width,height",
                "-of", "json", str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        if probe.returncode != 0:
            raise JobSpecError(f"activation ffprobe failed: {probe.stderr[-500:]}")
        try:
            streams = json.loads(probe.stdout).get("streams", [])
        except json.JSONDecodeError as exc:
            raise JobSpecError("activation ffprobe returned invalid JSON") from exc
        if len(streams) != 1 or streams[0].get("codec_name") != "av1":
            raise JobSpecError(f"activation video is not one AV1 stream: {path}")
        decoded = subprocess.run(
            [
                str(PROFILE.full_decode_ffmpeg), "-hide_banner", "-loglevel", "error",
                "-i", str(path), "-map", "0:v:0", "-f", "null", "-",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
        if decoded.returncode != 0:
            raise JobSpecError(f"activation full decode failed: {decoded.stderr[-500:]}")
        video_hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    return {
        "parquet_count": len(parquet_files),
        "rows": rows,
        "video_count": len(video_files),
        "parquet_hashes": parquet_hashes,
        "video_hashes": video_hashes,
        "strict_contract": strict_contract,
    }


def _verify_manifest_artifacts(shard_root: Path, manifest: Mapping[str, Any]) -> None:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise JobSpecError("activation shard manifest has no artifact hashes")
    expected = {
        path.relative_to(shard_root).as_posix()
        for path in _shard_artifact_files(shard_root)
    }
    actual: set[str] = set()
    for item in records:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise JobSpecError("activation shard manifest has an invalid artifact")
        relative = Path(item["path"])
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise JobSpecError("activation shard manifest has an unsafe artifact path")
        relative_name = relative.as_posix()
        if relative_name in actual:
            raise JobSpecError(f"activation shard manifest duplicates {relative_name}")
        path = shard_root / relative
        if path.is_symlink() or not path.is_file():
            raise JobSpecError(f"activation artifact is not a regular file: {path}")
        path = path.resolve(strict=True)
        try:
            path.relative_to(shard_root.resolve())
        except ValueError as exc:
            raise JobSpecError("activation artifact escapes shard") from exc
        if path.stat().st_size != int(item.get("size", -1)):
            raise JobSpecError(f"activation artifact size changed: {path}")
        if _sha256_file(path) != item.get("sha256"):
            raise JobSpecError(f"activation artifact hash changed: {path}")
        actual.add(relative_name)
    if actual != expected:
        raise JobSpecError("activation artifact manifest does not cover the shard")


def _activate_profile(
    shard_root: Path,
    source_path: Path,
    source_tree_before: str,
    manifest: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> None:
    from incubator.control_panel.tools.merge_lerobot_shards import merge_shards

    conversion = _validate_dataset_media(shard_root)
    _verify_manifest_artifacts(shard_root, manifest)
    source_tree_after = _tree_digest(source_path)
    if source_tree_after != source_tree_before:
        raise JobSpecError("activation probe changed the accepted NEDF source")
    merge_parent = Path(tempfile.mkdtemp(prefix=".activation-merge-", dir=shard_root.parent))
    try:
        merged = merge_shards(
            [shard_root, shard_root], merge_parent, "two-shard-activation", link_mode="hardlink"
        )
        merged_validation = _validate_dataset_media(merged)
        source_video_hashes = list(conversion["video_hashes"].values())
        merged_video_hashes = list(merged_validation["video_hashes"].values())
        if sorted(merged_video_hashes) != sorted(source_video_hashes * 2):
            raise JobSpecError("activation merge re-encoded or changed videos")
        merge_manifest = _read_json(merged / "meta" / "merge_manifest.json")
        if len(merge_manifest.get("shards", [])) != 2:
            raise JobSpecError("activation merge manifest does not contain two shards")
    finally:
        shutil.rmtree(merge_parent, ignore_errors=True)
    _atomic_write_json(
        PROFILE.activation_receipt,
        {
            "schema_version": 1,
            "status": "passed",
            "profile_fingerprint": PROFILE.fingerprint,
            "completed_at": _utc_now(),
            "source_tree_sha256": source_tree_after,
            "conversion": {
                "all_parquet_read": True,
                "all_videos_ffprobe": True,
                "all_videos_full_decode": True,
                "manifest_hashes_verified": True,
                "source_unchanged": True,
            },
            "merge": {
                "two_shards": True,
                "no_video_reencode": True,
                "all_parquet_read": True,
                "all_videos_ffprobe": True,
                "all_videos_full_decode": True,
                "manifest_hashes_verified": True,
            },
            "evidence": {
                "source_contract": dict(source_contract),
                "conversion": conversion,
                "merge": merged_validation,
                "job_id": manifest.get("job_id"),
                "source_fingerprint": manifest.get("source_fingerprint"),
            },
        },
    )


def _default_runner(command: Sequence[str]) -> None:
    completed = subprocess.run(list(command), check=False)
    if completed.returncode != 0:
        raise JobSpecError(f"Vendored converter exited with code {completed.returncode}")


def convert_job(
    job_spec: Path,
    staging_dir: Path,
    *,
    runner: Callable[[Sequence[str]], None] | None = None,
) -> Path:
    job_path = Path(job_spec).resolve(strict=True)
    job = _read_json(job_path)
    source_path, source = _verified_source(job)
    profile = _require_mapping(job.get("profile"), "profile")
    activation_probe = profile.get("activation_probe") is True
    source_tree_before = _tree_digest(source_path) if activation_probe else ""
    _verify_vendor_script()
    staging = Path(staging_dir).expanduser()
    if not staging.is_absolute():
        raise JobSpecError("--staging-dir must be absolute")
    staging.mkdir(parents=True, exist_ok=True)
    shard_root = staging / "shard"
    if shard_root.exists():
        raise JobSpecError(f"Staging shard already exists: {shard_root}")

    with _isolated_sdk_source(source_path, staging) as sdk_source:
        source_contract = (
            _validate_nedf3_source_contract(sdk_source)
            if activation_probe
            else {}
        )
        command = _converter_command(job, sdk_source, staging)
        (runner or _default_runner)(command)
    info_path = shard_root / "meta" / "info.json"
    validation_path = shard_root / "meta" / "validation_report.json"
    info = _read_json(info_path)
    validation = _read_json(validation_path)
    if int(info.get("total_episodes", -1)) != 1:
        raise JobSpecError("Per-episode converter did not produce exactly one episode")
    if validation.get("status") != "passed" or int(validation.get("episodes", -1)) != 1:
        raise JobSpecError("Per-episode shard validation did not pass")
    episode_records = _read_jsonl(shard_root / "meta" / "episodes.jsonl")
    if len(episode_records) != 1:
        raise JobSpecError("Per-episode shard does not contain one episode record")
    try:
        episode_uuid = str(uuid.UUID(str(episode_records[0].get("episode_uuid"))))
    except ValueError as exc:
        raise JobSpecError("Per-episode shard has an invalid episode_uuid") from exc
    if source["data_format"] == NEDF3_DATA_FORMAT and episode_uuid != str(
        uuid.UUID(str(source["episode_id"]))
    ):
        raise JobSpecError("NEDF3 source episode_id does not match LeRobot episode_uuid")

    sidecars = _copy_source_sidecars(source_path, shard_root)
    identity = _require_mapping(job.get("identity"), "identity")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "profile_id": identity["profile_id"],
        "source_fingerprint": identity["source_fingerprint"],
        "created_at": _utc_now(),
        "source": {
            "path": str(source_path),
            "episode_id": source["episode_id"],
            "episode_index": source["episode_index"],
            "result": source["result"],
            "data_format": source["data_format"],
            "metadata_sha256": source.get("metadata_sha256"),
            "episode_result_sha256": source.get("episode_result_sha256"),
            "episode_incomplete_sha256": source.get("episode_incomplete_sha256"),
            "dense_labels_sha256": source.get("dense_labels_sha256"),
            "files": source.get("files", []),
        },
        "converter": {
            "script": CONVERTER_SCRIPT.name,
            "sha256": CONVERTER_SHA256,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "ffmpeg": str(PROFILE.ffmpeg),
            "ffprobe": str(PROFILE.ffprobe),
            "profile_fingerprint": PROFILE.fingerprint,
            "profile": dict(profile),
            "run_config_sha256": _json_digest(job.get("run_config", {})),
        },
        "merger": {
            "script": MERGER_SCRIPT.name,
            "sha256": MERGER_SHA256,
        },
        "dataset": {
            "codebase_version": info.get("codebase_version"),
            "frames": info.get("total_frames"),
            "videos": info.get("total_videos"),
            "fps": info.get("fps"),
            "episode_uuid": episode_uuid,
            "features_sha256": _json_digest(info.get("features")),
        },
        "dense_labels": {
            "preserved_as_sidecar": "meta/source_sidecars/dense_labels.jsonl" in sidecars,
            "source_interval": "[start_frame,end_frame)",
            "frame_reference": "source executed-action commit order",
            "parquet_projection": "not_materialized",
        },
        "sidecars": sidecars,
        "artifacts": _artifact_records(shard_root),
    }
    manifest_path = shard_root / "meta" / "shard_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    if activation_probe:
        _activate_profile(
            shard_root,
            source_path,
            source_tree_before,
            manifest,
            source_contract,
        )
    return shard_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one accepted NEDF3 episode into a LeRobot shard."
    )
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        shard = convert_job(args.job_spec, args.staging_dir)
    except (JobSpecError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 1
    print(shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
