#!/usr/bin/env python3
"""Merge validated one-episode LeRobot v2.1 shards without re-encoding video."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from incubator.control_panel.recording_contract import (
    LEROBOT_CORE_PARQUET_COLUMNS,
    LEROBOT_PARQUET_COLUMNS,
    LEROBOT_PARQUET_SCHEMA_METADATA,
)


CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CORE_PARQUET_COLUMNS = LEROBOT_CORE_PARQUET_COLUMNS
NORMALIZED_SCHEMA_VERSION = "nmx_lerobot_v21_0_2_30"
UNION_SCHEMA_VERSION = NORMALIZED_SCHEMA_VERSION
LEGACY_XYZW_ACTION_CONVERTER_SHA256S = {
    "6f48b845aadd00b0e2bc8ed0e360709751e6dad1be024eea4018ad0837ba447d",
}


def _canonical_extra_types() -> dict[str, pa.DataType]:
    types: dict[str, pa.DataType] = {}
    for side in ("left", "right", "ego"):
        prefix = f"observation.extra.{side}"
        types[f"{prefix}.raw_pose"] = pa.list_(pa.float32(), 7)
        for field in ("hand_state", "raw_hand_state", "roi_marker_angle"):
            types[f"{prefix}.{field}"] = pa.float32()
        types[f"{prefix}.aligned_timestamp_ns"] = pa.float64()
    return types


CANONICAL_EXTRA_TYPES = _canonical_extra_types()
CANONICAL_EXTRA_COLUMNS = tuple(CANONICAL_EXTRA_TYPES)
if (*CORE_PARQUET_COLUMNS, *CANONICAL_EXTRA_COLUMNS) != LEROBOT_PARQUET_COLUMNS:
    raise RuntimeError("LeRobot merge schema drifted from the canonical contract")


class MergeError(RuntimeError):
    """Input shards are invalid or incompatible."""


@dataclass(frozen=True)
class Shard:
    root: Path
    manifest: dict[str, Any]
    info: dict[str, Any]
    episode: dict[str, Any]
    episode_stats: dict[str, Any]
    tasks: dict[int, str]
    modality: dict[str, Any]


@dataclass(frozen=True)
class ParquetUnionPlan:
    columns: tuple[str, ...]
    types: dict[str, pa.DataType]
    canonical_enabled: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MergeError(f"Cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MergeError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise MergeError(f"Expected object at {path}:{line_number}")
        result.append(value)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shard_artifact_files(root: Path) -> list[Path]:
    manifest_path = root / "meta" / "shard_manifest.json"
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MergeError(f"Shard artifact cannot be a symbolic link: {path}")
        if path.is_file() and path != manifest_path:
            files.append(path)
    return files


def _verify_shard_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise MergeError(f"Shard manifest has no artifact hashes: {root}")
    expected = {
        path.relative_to(root).as_posix() for path in _shard_artifact_files(root)
    }
    actual: set[str] = set()
    for item in records:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise MergeError(f"Shard manifest contains an invalid artifact: {root}")
        relative = Path(item["path"])
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise MergeError(f"Shard manifest contains an unsafe artifact path: {root}")
        relative_name = relative.as_posix()
        if relative_name in actual:
            raise MergeError(f"Shard manifest duplicates artifact {relative_name}: {root}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise MergeError(f"Shard artifact is not a regular file: {path}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise MergeError(f"Shard artifact escapes its root: {path}") from exc
        try:
            expected_size = int(item.get("size", -1))
        except (TypeError, ValueError) as exc:
            raise MergeError(f"Shard artifact has an invalid size: {path}") from exc
        if resolved.stat().st_size != expected_size:
            raise MergeError(f"Shard artifact size mismatch: {path}")
        if _sha256_file(resolved) != item.get("sha256"):
            raise MergeError(f"Shard artifact hash mismatch: {path}")
        actual.add(relative_name)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise MergeError(
            f"Shard manifest does not cover the complete shard: {root}; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _load_shard(root: Path) -> Shard:
    resolved = Path(root).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise MergeError(f"Shard root is not a directory: {resolved}")
    manifest = _read_json(resolved / "meta" / "shard_manifest.json")
    for field in ("job_id", "profile_id", "source_fingerprint"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise MergeError(f"Shard manifest lacks {field}: {resolved}")
    _verify_shard_artifacts(resolved, manifest)
    info = _read_json(resolved / "meta" / "info.json")
    episodes = _read_jsonl(resolved / "meta" / "episodes.jsonl")
    episode_stats = _read_jsonl(resolved / "meta" / "episodes_stats.jsonl")
    task_records = _read_jsonl(resolved / "meta" / "tasks.jsonl")
    modality = _read_json(resolved / "meta" / "modality.json")
    if int(info.get("total_episodes", -1)) != 1 or len(episodes) != 1 or len(episode_stats) != 1:
        raise MergeError(f"Expected a one-episode shard: {resolved}")
    source = manifest.get("source")
    if isinstance(source, Mapping) and source.get("data_format") == "NEDF3":
        dataset = manifest.get("dataset")
        if not isinstance(dataset, Mapping):
            raise MergeError(f"NEDF3 shard manifest lacks dataset identity: {resolved}")
        try:
            manifest_uuid = str(uuid.UUID(str(dataset.get("episode_uuid"))))
            episode_uuid = str(uuid.UUID(str(episodes[0].get("episode_uuid"))))
        except ValueError as exc:
            raise MergeError(f"NEDF3 shard has an invalid episode UUID: {resolved}") from exc
        if manifest_uuid != episode_uuid:
            raise MergeError(f"NEDF3 shard episode UUID provenance mismatch: {resolved}")
    tasks: dict[int, str] = {}
    for record in task_records:
        index = record.get("task_index")
        task = record.get("task")
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(task, str):
            raise MergeError(f"Invalid tasks.jsonl record in {resolved}")
        tasks[index] = task
    if not tasks:
        raise MergeError(f"Shard has no task records: {resolved}")
    return Shard(
        root=resolved,
        manifest=manifest,
        info=info,
        episode=episodes[0],
        episode_stats=episode_stats[0],
        tasks=tasks,
        modality=modality,
    )


def _episode_uuid_for_shard(shard: Shard) -> str:
    existing = shard.episode.get("episode_uuid")
    if isinstance(existing, str):
        try:
            return str(uuid.UUID(existing))
        except ValueError:
            pass
    identity = ":".join(
        (
            "superinference-lerobot-shard",
            shard.manifest["source_fingerprint"],
            str(shard.episode.get("original_episode_index", 0)),
            str(shard.episode.get("source_episode", "episode_0000")),
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _unique_output_episode_uuid(
    shard: Shard, output_episode_index: int, used: set[str]
) -> str:
    candidate = _episode_uuid_for_shard(shard)
    if candidate not in used:
        return candidate
    identity = ":".join(
        (
            "superinference-lerobot-merged-duplicate",
            candidate,
            shard.manifest["job_id"],
            str(output_episode_index),
        )
    )
    candidate = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
    if candidate in used:
        raise MergeError(
            f"Cannot derive a unique episode UUID for duplicate shard {shard.root}"
        )
    return candidate


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _tcp_pose_semantics(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    config = value.get("config")
    if not isinstance(config, Mapping):
        return value
    return {
        "name": value.get("name"),
        "world_z_rotation_degrees": config.get("world_z_rotation_degrees"),
        "quaternion_order": config.get("quaternion_order"),
        "left_y_translation": (
            config.get("left", {}).get("y_translation_after_rotation")
            if isinstance(config.get("left"), Mapping)
            else None
        ),
        "right_y_translation": (
            config.get("right", {}).get("y_translation_after_rotation")
            if isinstance(config.get("right"), Mapping)
            else None
        ),
    }


def _require_compatible(shards: Sequence[Shard]) -> None:
    first = shards[0]
    fields = (
        "codebase_version",
        "nmx_data_version",
        "robot_type",
        "fps",
        "target_fps",
        "data_path",
        "video_path",
        "features",
    )
    for shard in shards[1:]:
        for field in fields:
            default = {
                "nmx_data_version": "0.2.30",
                "target_fps": 10.0,
            }.get(field)
            first_value = first.info.get(field, default)
            shard_value = shard.info.get(field, default)
            if _canonical(shard_value) != _canonical(first_value):
                raise MergeError(
                    f"Incompatible shard field {field}: {first.root} != {shard.root}"
                )
        first_macro = first.info.get("tcp_pose_macro")
        shard_macro = shard.info.get("tcp_pose_macro")
        if first_macro is not None and shard_macro is not None:
            if _canonical(_tcp_pose_semantics(shard_macro)) != _canonical(
                _tcp_pose_semantics(first_macro)
            ):
                raise MergeError(
                    "Incompatible shard field tcp_pose_macro: "
                    f"{first.root} != {shard.root}"
                )
        if _canonical(shard.modality) != _canonical(first.modality):
            raise MergeError(f"Incompatible modality.json: {shard.root}")


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


def _set_int64_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise MergeError(f"Parquet is missing required column {name}")
    return table.set_column(
        index,
        table.schema.field(index),
        pa.array(np.asarray(values, dtype=np.int64), type=pa.int64()),
    )


def _schema_kind(schema: pa.Schema) -> str:
    names = set(schema.names)
    if tuple(schema.names) == LEROBOT_PARQUET_COLUMNS:
        return "nmx_lerobot_v21_v0230"
    if any(name.endswith(".tcp_state_base") for name in names):
        return "legacy_r1_extra_v1"
    if any(name.endswith(".raw_pose") for name in names):
        return "legacy_raw_pose_v1"
    if tuple(schema.names) == CORE_PARQUET_COLUMNS:
        return "unsupported_core_v0"
    return "union_passthrough"


def _merge_field_type(
    types: dict[str, pa.DataType], name: str, field_type: pa.DataType, source: Path
) -> None:
    existing = types.get(name)
    if existing is not None and not existing.equals(field_type):
        raise MergeError(
            f"Incompatible Parquet type for {name}: {existing} != {field_type} in {source}"
        )
    types[name] = field_type


def _build_union_plan(shards: Sequence[Shard]) -> ParquetUnionPlan:
    schemas = [(shard, pq.read_schema(_source_parquet(shard))) for shard in shards]
    for shard, schema in schemas:
        missing = [name for name in CORE_PARQUET_COLUMNS if name not in schema.names]
        if missing:
            raise MergeError(f"Shard Parquet lacks core columns {missing}: {shard.root}")
    columns = list(LEROBOT_PARQUET_COLUMNS)
    types: dict[str, pa.DataType] = {}
    for shard, schema in schemas:
        for name in CORE_PARQUET_COLUMNS:
            _merge_field_type(types, name, schema.field(name).type, shard.root)
    for name, field_type in CANONICAL_EXTRA_TYPES.items():
        _merge_field_type(types, name, field_type, Path("<canonical-schema>"))
    return ParquetUnionPlan(tuple(columns), types, True)


def _fixed_float_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _null_array(length: int, field_type: pa.DataType) -> pa.Array:
    if pa.types.is_fixed_size_list(field_type):
        flat = pa.nulls(length * field_type.list_size, type=field_type.value_type)
        return pa.FixedSizeListArray.from_arrays(flat, field_type.list_size)
    return pa.nulls(length, type=field_type)


def _fixed_list_numpy(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.float32)


def _tcp_macro_name(shard: Shard) -> str:
    macro = shard.info.get("tcp_pose_macro")
    name = macro.get("name") if isinstance(macro, Mapping) else None
    if name not in {"none", "r1_world"}:
        raise MergeError(f"Cannot normalize unsupported TCP macro {name!r}: {shard.root}")
    return str(name)


def _inverse_tcp_macro(values: np.ndarray, macro_name: str) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float32).copy()
    if macro_name == "none":
        return transformed
    for block_start, y_translation in ((0, 0.2), (8, -0.2)):
        world = np.asarray(values)[:, block_start : block_start + 8]
        transformed[:, block_start] = -world[:, 0]
        transformed[:, block_start + 1] = -world[:, 1] + y_translation
        quat = world[:, 3:7]
        transformed[:, block_start + 3] = quat[:, 3]
        transformed[:, block_start + 4] = quat[:, 2]
        transformed[:, block_start + 5] = -quat[:, 1]
        transformed[:, block_start + 6] = -quat[:, 0]
    return transformed


def _action_xyzw_to_wxyz(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float32).copy()
    for block_start in (0, 8):
        quat = np.asarray(values)[:, block_start + 3 : block_start + 7]
        transformed[:, block_start + 3] = quat[:, 3]
        transformed[:, block_start + 4] = quat[:, 0]
        transformed[:, block_start + 5] = quat[:, 1]
        transformed[:, block_start + 6] = quat[:, 2]
    return transformed


def _correct_legacy_world_action(values: np.ndarray, macro_name: str) -> np.ndarray:
    if macro_name == "none":
        return _action_xyzw_to_wxyz(values)
    transformed = np.asarray(values, dtype=np.float32).copy()
    for block_start in (0, 8):
        legacy_world_quat = np.asarray(values)[:, block_start + 3 : block_start + 7]
        transformed[:, block_start + 3] = legacy_world_quat[:, 1]
        transformed[:, block_start + 4] = -legacy_world_quat[:, 2]
        transformed[:, block_start + 5] = legacy_world_quat[:, 3]
        transformed[:, block_start + 6] = -legacy_world_quat[:, 0]
    return transformed


def _converter_sha256(shard: Shard) -> str | None:
    converter = shard.manifest.get("converter")
    value = converter.get("sha256") if isinstance(converter, Mapping) else None
    return str(value).lower() if isinstance(value, str) else None


def _legacy_nedf_payload(
    table: pa.Table, shard: Shard
) -> tuple[pa.Table, dict[str, pa.Array], bool]:
    converter_sha256 = _converter_sha256(shard)
    if converter_sha256 not in LEGACY_XYZW_ACTION_CONVERTER_SHA256S:
        raise MergeError(
            "Cannot normalize a legacy 7-column shard with an unknown converter hash: "
            f"{shard.root} ({converter_sha256!r})"
        )
    macro_name = _tcp_macro_name(shard)
    world_states = _fixed_list_numpy(table, "observation.state")
    legacy_world_actions = _fixed_list_numpy(table, "action")
    base_states = _inverse_tcp_macro(world_states, macro_name)
    world_actions = _correct_legacy_world_action(legacy_world_actions, macro_name)
    action_index = table.schema.get_field_index("action")
    table = table.set_column(
        action_index,
        table.schema.field(action_index),
        _fixed_float_list(world_actions, world_actions.shape[1]),
    )

    length = table.num_rows
    synthesized: dict[str, pa.Array] = {}
    for side, block_start in (("left", 0), ("right", 8), ("ego", 16)):
        prefix = f"observation.extra.{side}"
        if base_states.shape[1] >= block_start + 7:
            raw_pose = base_states[:, block_start : block_start + 7]
        elif side == "ego":
            raw_pose = np.broadcast_to(
                np.asarray(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
                ),
                (length, 7),
            )
        else:
            raise MergeError(
                f"Legacy state is too narrow for {side} raw_pose: {shard.root}"
            )
        synthesized[f"{prefix}.raw_pose"] = _fixed_float_list(raw_pose, 7)
        if side == "ego":
            synthesized[f"{prefix}.hand_state"] = pa.nulls(
                length, type=pa.float32()
            )
        elif base_states.shape[1] > block_start + 7:
            synthesized[f"{prefix}.hand_state"] = pa.array(
                base_states[:, block_start + 7], type=pa.float32()
            )
        else:
            raise MergeError(
                f"Legacy state is missing {side} hand_state: {shard.root}"
            )
        synthesized[f"{prefix}.raw_hand_state"] = pa.nulls(length, type=pa.float32())
        synthesized[f"{prefix}.roi_marker_angle"] = pa.nulls(length, type=pa.float32())
        synthesized[f"{prefix}.aligned_timestamp_ns"] = pa.nulls(
            length, type=pa.float64()
        )
    return table, synthesized, True


def _first_existing_column(
    table: pa.Table, names: Sequence[str]
) -> pa.Array | pa.ChunkedArray | None:
    for name in names:
        index = table.schema.get_field_index(name)
        if index >= 0:
            return table.column(index)
    return None


def _canonical_extra_payload(
    table: pa.Table, shard: Shard
) -> dict[str, pa.Array | pa.ChunkedArray]:
    """Map historical extra schemas into the exact NMX 0.2.30 columns."""
    states = _fixed_list_numpy(table, "observation.state")
    length = table.num_rows
    synthesized: dict[str, pa.Array | pa.ChunkedArray] = {}
    for side, block_start in (("left", 0), ("right", 8), ("ego", 16)):
        target = f"observation.extra.{side}"
        legacy_side = "head" if side == "ego" else side
        source = f"observation.extra.{legacy_side}"

        raw_pose = _first_existing_column(
            table,
            (
                f"{target}.raw_pose",
                f"{source}.tcp_state_base",
                f"{source}.tcp_state_world",
            ),
        )
        if raw_pose is None:
            if states.shape[1] >= block_start + 7:
                raw_pose = _fixed_float_list(
                    states[:, block_start : block_start + 7], 7
                )
            elif side == "ego":
                identity = np.broadcast_to(
                    np.asarray(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    (length, 7),
                )
                raw_pose = _fixed_float_list(identity, 7)
            else:
                raise MergeError(
                    f"Cannot recover {side} raw_pose from shard {shard.root}"
                )
        synthesized[f"{target}.raw_pose"] = raw_pose

        hand_state = _first_existing_column(
            table, (f"{target}.hand_state", f"{source}.hand_state")
        )
        if hand_state is None and side != "ego" and states.shape[1] > block_start + 7:
            hand_state = pa.array(states[:, block_start + 7], type=pa.float32())
        synthesized[f"{target}.hand_state"] = (
            hand_state
            if hand_state is not None
            else pa.nulls(length, type=pa.float32())
        )

        for field in ("raw_hand_state", "roi_marker_angle", "aligned_timestamp_ns"):
            value = _first_existing_column(
                table, (f"{target}.{field}", f"{source}.{field}")
            )
            synthesized[f"{target}.{field}"] = (
                value
                if value is not None
                else _null_array(
                    length, CANONICAL_EXTRA_TYPES[f"{target}.{field}"]
                )
            )
    return synthesized


def _normalize_parquet(
    table: pa.Table, shard: Shard, plan: ParquetUnionPlan
) -> tuple[pa.Table, dict[str, Any]]:
    original_names = tuple(table.schema.names)
    source_kind = _schema_kind(table.schema)
    synthesized: dict[str, pa.Array | pa.ChunkedArray] = {}
    action_converted = False
    if source_kind == "unsupported_core_v0":
        raise MergeError(f"Legacy 7-column shards are unsupported: {shard.root}")
    else:
        synthesized = _canonical_extra_payload(table, shard)

    arrays: list[pa.Array | pa.ChunkedArray] = []
    for name in plan.columns:
        index = table.schema.get_field_index(name)
        if index >= 0:
            value: pa.Array | pa.ChunkedArray = table.column(index)
        elif name in synthesized:
            value = synthesized[name]
        elif name in CANONICAL_EXTRA_TYPES:
            value = synthesized[name]
        else:
            value = _null_array(table.num_rows, plan.types[name])
        value_type = value.type
        expected_type = plan.types[name]
        if not value_type.equals(expected_type):
            raise MergeError(
                f"Cannot normalize {name} from {value_type} to {expected_type}: {shard.root}"
            )
        arrays.append(value)

    normalized = pa.Table.from_arrays(arrays, names=plan.columns)
    normalized = normalized.replace_schema_metadata(LEROBOT_PARQUET_SCHEMA_METADATA)
    return normalized, {
        "source_schema": source_kind,
        "source_columns": list(original_names),
        "source_schema_metadata": {
            key.decode("utf-8", errors="replace"): value.decode(
                "utf-8", errors="replace"
            )
            for key, value in (table.schema.metadata or {}).items()
        },
        "output_columns": len(plan.columns),
        "action_quaternion_converted": action_converted,
        "missing_columns_synthesized": [
            name for name in plan.columns if name not in original_names
        ],
    }


def _materialize_file(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    if mode in {"auto", "reflink"}:
        completed = subprocess.run(
            ["cp", "--reflink=always", "--preserve=mode,timestamps", str(source), str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            return "reflink"
        if mode == "reflink":
            raise MergeError(f"Cannot reflink {source} -> {target}")
    shutil.copy2(source, target)
    return "copy"


def _source_parquet(shard: Shard) -> Path:
    episode_index = int(shard.episode.get("episode_index", 0))
    template = str(shard.info.get("data_path") or DATA_PATH)
    return shard.root / template.format(
        episode_chunk=episode_index // int(shard.info.get("chunks_size", CHUNK_SIZE)),
        episode_index=episode_index,
    )


def _video_features(info: Mapping[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise MergeError("info.json features must be an object")
    return sorted(
        name
        for name, spec in features.items()
        if isinstance(spec, Mapping) and spec.get("dtype") == "video"
    )


def _is_episode_sidecar_directory(name: str) -> bool:
    prefix = "episode_"
    suffix = name[len(prefix) :] if name.startswith(prefix) else ""
    return len(suffix) == 6 and suffix.isdigit()


def _copy_source_sidecars(
    shard: Shard, staging: Path, output_episode_index: int
) -> None:
    source_root = shard.root / "meta" / "source_sidecars"
    if not source_root.is_dir():
        return

    source_episode_index = int(shard.episode.get("episode_index", 0))
    source_episode_name = f"episode_{source_episode_index:06d}"
    unexpected_episode_directories: list[str] = []
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise MergeError(f"Source sidecar cannot be a symbolic link: {path}")
        if not path.is_dir() or not path.name.startswith("episode_"):
            continue
        relative = path.relative_to(source_root)
        if (
            relative.parts != (source_episode_name,)
            or not _is_episode_sidecar_directory(path.name)
        ):
            unexpected_episode_directories.append(relative.as_posix())
    if unexpected_episode_directories:
        raise MergeError(
            "One-episode shard contains ambiguous episode sidecar directories: "
            f"{shard.root} ({sorted(unexpected_episode_directories)})"
        )

    destination_root = (
        staging
        / "meta"
        / "source_sidecars"
        / f"episode_{output_episode_index:06d}"
    )
    destinations: dict[Path, Path] = {}
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        if relative.parts[0] == source_episode_name:
            relative = Path(*relative.parts[1:])
        destination = destination_root / relative
        previous = destinations.get(destination)
        if previous is not None:
            raise MergeError(
                "Source sidecars collide after episode normalization: "
                f"{previous} and {source}"
            )
        destinations[destination] = source

    for destination, source in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _validate_structure(root: Path) -> dict[str, Any]:
    info = _read_json(root / "meta" / "info.json")
    total_frames = 0
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    reference_schema: pa.Schema | None = None
    for episode_index, path in enumerate(parquet_files):
        schema = pq.read_schema(path)
        if tuple(schema.names) != LEROBOT_PARQUET_COLUMNS:
            raise MergeError(f"Non-uniform normalized Parquet columns in {path}")
        comparable_schema = schema.remove_metadata()
        if reference_schema is None:
            reference_schema = comparable_schema
        elif not comparable_schema.equals(reference_schema):
            raise MergeError(f"Non-uniform normalized Parquet schema in {path}")
        metadata = schema.metadata or {}
        if metadata != LEROBOT_PARQUET_SCHEMA_METADATA:
            raise MergeError(f"Unexpected normalized Parquet metadata in {path}")
        table = pq.read_table(path, columns=["frame_index", "episode_index", "index", "task_index"])
        rows = table.num_rows
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(rows)):
            raise MergeError(f"Non-contiguous frame_index in {path}")
        if not np.array_equal(
            table["episode_index"].to_numpy(), np.full(rows, episode_index)
        ):
            raise MergeError(f"Incorrect episode_index in {path}")
        if not np.array_equal(table["index"].to_numpy(), np.arange(rows)):
            raise MergeError(f"Incorrect per-episode index in {path}")
        total_frames += rows
    videos = list((root / "videos").rglob("*.mp4"))
    if len(parquet_files) != int(info["total_episodes"]):
        raise MergeError("Merged parquet count does not match info.json")
    if total_frames != int(info["total_frames"]):
        raise MergeError("Merged frame count does not match info.json")
    if len(videos) != int(info["total_videos"]):
        raise MergeError("Merged video count does not match info.json")
    return {
        "status": "passed",
        "validation": "structural",
        "episodes": len(parquet_files),
        "frames": total_frames,
        "videos": len(videos),
        "parquet_columns": len(reference_schema.names) if reference_schema is not None else 0,
        "parquet_schema_version": NORMALIZED_SCHEMA_VERSION,
    }


def merge_shards(
    shard_roots: Sequence[Path],
    output_parent: Path,
    dataset_name: str,
    *,
    link_mode: str = "auto",
) -> Path:
    if not shard_roots:
        raise MergeError("At least one shard is required")
    if link_mode not in {"auto", "hardlink", "reflink", "copy"}:
        raise MergeError(f"Unsupported link mode: {link_mode}")
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise MergeError("dataset_name must be one safe path component")
    shards = [_load_shard(root) for root in shard_roots]
    _require_compatible(shards)
    union_plan = _build_union_plan(shards)

    output_dir = Path(output_parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_root = output_dir / dataset_name
    if final_root.exists():
        raise FileExistsError(f"Output dataset already exists: {final_root}")
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_name}.tmp-", dir=output_dir))
    episode_records: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    task_names: list[str] = []
    task_indexes: dict[str, int] = {}
    total_frames = 0
    video_count = 0
    materialization_counts = {"hardlink": 0, "reflink": 0, "copy": 0}
    schema_records: list[dict[str, Any]] = []
    episode_uuids: set[str] = set()
    try:
        for shard in shards:
            for task in shard.tasks.values():
                if task not in task_indexes:
                    task_indexes[task] = len(task_names)
                    task_names.append(task)

        for episode_index, shard in enumerate(shards):
            source_table = pq.read_table(_source_parquet(shard))
            source_table, schema_record = _normalize_parquet(
                source_table, shard, union_plan
            )
            schema_record["output_episode_index"] = episode_index
            schema_record["source_shard_job_id"] = shard.manifest["job_id"]
            schema_records.append(schema_record)
            length = source_table.num_rows
            old_task_values = source_table["task_index"].to_numpy()
            try:
                new_task_values = np.asarray(
                    [task_indexes[shard.tasks[int(value)]] for value in old_task_values],
                    dtype=np.int64,
                )
            except (KeyError, ValueError) as exc:
                raise MergeError(f"Unknown task_index in shard {shard.root}") from exc
            episode_values = np.full(length, episode_index, dtype=np.int64)
            index_values = np.arange(length, dtype=np.int64)
            table = _set_int64_column(source_table, "episode_index", episode_values)
            table = _set_int64_column(table, "index", index_values)
            table = _set_int64_column(table, "task_index", new_task_values)
            destination_parquet = staging / DATA_PATH.format(
                episode_chunk=episode_index // CHUNK_SIZE,
                episode_index=episode_index,
            )
            destination_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination_parquet, compression="snappy")

            stats = dict(shard.episode_stats.get("stats") or {})
            if schema_record["action_quaternion_converted"]:
                stats["action"] = _numeric_stats(
                    _fixed_list_numpy(table, "action")
                )
            stats["episode_index"] = _numeric_stats(episode_values)
            stats["index"] = _numeric_stats(index_values)
            stats["task_index"] = _numeric_stats(new_task_values)
            stats_record = dict(shard.episode_stats)
            stats_record["episode_index"] = episode_index
            stats_record["stats"] = stats
            stats_records.append(stats_record)

            task_values = [
                shard.tasks[int(value)] for value in np.unique(old_task_values)
            ]
            output_episode_uuid = _unique_output_episode_uuid(
                shard, episode_index, episode_uuids
            )
            episode_uuids.add(output_episode_uuid)
            episode_record = {
                "episode_index": episode_index,
                "episode_uuid": output_episode_uuid,
                "tasks": task_values,
                "task_description": str(
                    shard.episode.get("task_description") or task_values[0]
                ),
                "task_annotation": str(
                    shard.episode.get("task_annotation") or task_values[0]
                ),
                "instruct_ch": str(shard.episode.get("instruct_ch") or ""),
                "hardware_string": list(
                    shard.episode.get("hardware_string") or []
                ),
                "rp_type": list(shard.episode.get("rp_type") or []),
                "device_type": str(
                    shard.episode.get("device_type")
                    or shard.info.get("device_type")
                    or "inference_rollout"
                ),
                "collection_mode": str(
                    shard.episode.get("collection_mode") or "inference_rollout"
                ),
                "slam_validity_all": bool(
                    shard.episode.get("slam_validity_all", True)
                ),
                "width_detection": bool(
                    shard.episode.get("width_detection", False)
                ),
                "width_calibration": bool(
                    shard.episode.get("width_calibration", False)
                ),
                "length": length,
                "action_config": list(
                    shard.episode.get("action_config")
                    or [
                        {
                            "start_frame": 0,
                            "end_frame": length,
                            "action_text": task_values[0],
                        }
                    ]
                ),
            }
            episode_records.append(episode_record)

            source_episode_index = int(shard.episode.get("episode_index", 0))
            source_chunk = source_episode_index // int(shard.info.get("chunks_size", CHUNK_SIZE))
            video_template = str(shard.info.get("video_path") or VIDEO_PATH)
            for feature in _video_features(shard.info):
                source_video = shard.root / video_template.format(
                    episode_chunk=source_chunk,
                    video_key=feature,
                    episode_index=source_episode_index,
                )
                if not source_video.is_file():
                    raise MergeError(f"Missing shard video: {source_video}")
                destination_video = staging / VIDEO_PATH.format(
                    episode_chunk=episode_index // CHUNK_SIZE,
                    video_key=feature,
                    episode_index=episode_index,
                )
                used = _materialize_file(source_video, destination_video, link_mode)
                materialization_counts[used] += 1
                video_count += 1

            _copy_source_sidecars(shard, staging, episode_index)
            total_frames += length

        first_info = shards[0].info
        info = {
            "codebase_version": str(first_info.get("codebase_version") or "v2.1"),
            "nmx_data_version": str(
                first_info.get("nmx_data_version") or "0.2.30"
            ),
            "robot_type": str(first_info.get("robot_type") or "dual_arm"),
            "total_episodes": len(shards),
            "total_frames": total_frames,
            "total_tasks": len(task_names),
            "total_videos": video_count,
            "total_chunks": math.ceil(len(shards) / CHUNK_SIZE),
            "chunks_size": CHUNK_SIZE,
            "fps": int(first_info["fps"]),
            "target_fps": float(first_info.get("target_fps", 10.0)),
            "splits": {"train": f"0:{len(shards)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": first_info["features"],
        }
        _write_json(staging / "meta" / "info.json", info)
        _write_jsonl(
            staging / "meta" / "tasks.jsonl",
            [
                {"task_index": task_index, "task": task}
                for task_index, task in enumerate(task_names)
            ],
        )
        _write_jsonl(staging / "meta" / "episodes.jsonl", episode_records)
        _write_jsonl(staging / "meta" / "episodes_stats.jsonl", stats_records)
        stat_keys = set(stats_records[0]["stats"])
        for record in stats_records[1:]:
            if set(record["stats"]) != stat_keys:
                raise MergeError("Shard episode stats feature sets differ")
        _write_json(staging / "meta" / "modality.json", shards[0].modality)
        merge_manifest = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "dataset_name": dataset_name,
            "video_materialization": materialization_counts,
            "parquet_schema": {
                "version": UNION_SCHEMA_VERSION,
                "columns": list(union_plan.columns),
                "episodes": schema_records,
            },
            "shards": [
                {
                    "root": str(shard.root),
                    "job_id": shard.manifest["job_id"],
                    "profile_id": shard.manifest["profile_id"],
                    "source_fingerprint": shard.manifest["source_fingerprint"],
                    "output_episode_index": index,
                    "source_info": shard.info,
                    "source_episode": shard.episode,
                }
                for index, shard in enumerate(shards)
            ],
        }
        _write_json(staging / "meta" / "merge_manifest.json", merge_manifest)
        validation = _validate_structure(staging)
        _write_json(staging / "meta" / "validation_report.json", validation)
        staging.replace(final_root)
        return final_root
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge one-episode LeRobot v2.1 shards without video re-encoding."
    )
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "reflink", "copy"), default="auto"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        root = merge_shards(
            args.shard,
            args.output_dir,
            args.dataset_name,
            link_mode=args.link_mode,
        )
    except (MergeError, FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"merge failed: {exc}", file=sys.stderr)
        return 1
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
