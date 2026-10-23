from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from incubator.control_panel.tools import merge_lerobot_shards as merger
from incubator.control_panel.recording_contract import (
    LEROBOT_PARQUET_COLUMNS,
    LEROBOT_PARQUET_SCHEMA_METADATA,
)


VIDEO_FEATURE = "observation.images.wrist_image_1"
STATE_DIM = 23
LEGACY_CONVERTER_SHA256 = (
    "6f48b845aadd00b0e2bc8ed0e360709751e6dad1be024eea4018ad0837ba447d"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _stats(values: np.ndarray) -> dict:
    return merger._numeric_stats(np.asarray(values))


def _fixed(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _refresh_manifest(root: Path) -> None:
    manifest_path = root / "meta" / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": merger._sha256_file(path),
        }
        for path in merger._shard_artifact_files(root)
    ]
    _write_json(manifest_path, manifest)


def _make_shard(root: Path, marker: str, task: str, length: int, *, fps: int = 30) -> Path:
    info = {
        "codebase_version": "v2.1",
        "source_data_format": "NEDF3",
        "robot_type": "dual_arm",
        "device_type": "inference_r1",
        "total_episodes": 1,
        "total_frames": length,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": merger.DATA_PATH,
        "video_path": merger.VIDEO_PATH,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [STATE_DIM]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            VIDEO_FEATURE: {
                "dtype": "video",
                "shape": [2, 2, 3],
                "info": {"video.codec": "av1", "video.fps": fps},
            },
        },
        "ego_pose_padding": None,
        "wrist_image_processing": {"resize_hw": [2, 2]},
        "head_image_processing": None,
        "camera_sources": {VIDEO_FEATURE: "camera"},
        "tcp_pose_macro": {"name": "none", "config": None},
    }
    _write_json(root / "meta" / "info.json", info)
    _write_json(root / "meta" / "modality.json", {"video": {"wrist": VIDEO_FEATURE}})
    _write_json(
        root / "meta" / "shard_manifest.json",
        {
            "schema_version": 1,
            "job_id": marker * 64,
            "profile_id": "test-profile",
            "source_fingerprint": marker * 64,
            "converter": {"sha256": LEGACY_CONVERTER_SHA256},
        },
    )
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": [task], "length": length}],
    )
    frame = np.arange(length, dtype=np.int64)
    timestamp = frame.astype(np.float32) / np.float32(fps)
    state = np.zeros((length, STATE_DIM), dtype=np.float32)
    action = np.zeros_like(state)
    for start in (3, 11, 19):
        state[:, start] = 1.0
        action[:, start + 3] = 1.0
    table = pa.Table.from_arrays(
        [
            _fixed(state),
            _fixed(action),
            pa.array(timestamp, type=pa.float32()),
            pa.array(frame, type=pa.int64()),
            pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
            pa.array(frame, type=pa.int64()),
            pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
        ],
        names=list(merger.CORE_PARQUET_COLUMNS),
    )
    parquet = root / merger.DATA_PATH.format(episode_chunk=0, episode_index=0)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet, compression="snappy")
    episode_stats = {
        "observation.state": _stats(state),
        "action": _stats(action),
        "timestamp": _stats(timestamp),
        "frame_index": _stats(frame),
        "episode_index": _stats(np.zeros(length, dtype=np.int64)),
        "index": _stats(frame),
        "task_index": _stats(np.zeros(length, dtype=np.int64)),
    }
    _write_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        [{"episode_index": 0, "stats": episode_stats}],
    )
    _write_json(root / "meta" / "stats.json", episode_stats)
    video = root / merger.VIDEO_PATH.format(
        episode_chunk=0, video_key=VIDEO_FEATURE, episode_index=0
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(("video-" + marker).encode())
    sidecar = root / "meta" / "source_sidecars" / "dense_labels.jsonl"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('{"label":"accept","start_frame":0,"end_frame":1}\n')
    joint_sidecars = root / "meta" / "source_sidecars" / "episode_000000"
    joint_sidecars.mkdir()
    _write_json(
        joint_sidecars / "joint_trajectory.json",
        {"schema_version": 1, "marker": marker},
    )
    (joint_sidecars / "joint_trajectory.npz").write_bytes(
        ("joint-" + marker).encode()
    )
    _refresh_manifest(root)
    return root


def _parquet_path(root: Path) -> Path:
    return root / merger.DATA_PATH.format(episode_chunk=0, episode_index=0)


def _replace_action_stats(root: Path, action: np.ndarray) -> None:
    records = [
        json.loads(line)
        for line in (root / "meta/episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records[0]["stats"]["action"] = _stats(action)
    _write_jsonl(root / "meta/episodes_stats.jsonl", records)
    aggregate = json.loads((root / "meta/stats.json").read_text())
    aggregate["action"] = _stats(action)
    _write_json(root / "meta/stats.json", aggregate)


def _upgrade_to_canonical(root: Path) -> Path:
    shard = merger._load_shard(root)
    table = pq.read_table(_parquet_path(root))
    table, synthesized, converted = merger._legacy_nedf_payload(table, shard)
    assert converted is True
    arrays: list[pa.Array | pa.ChunkedArray] = [table[name] for name in merger.CORE_PARQUET_COLUMNS]
    for name in merger.CANONICAL_EXTRA_COLUMNS:
        arrays.append(synthesized[name])
    canonical = pa.Table.from_arrays(
        arrays,
        names=[*merger.CORE_PARQUET_COLUMNS, *merger.CANONICAL_EXTRA_COLUMNS],
    )
    canonical = canonical.replace_schema_metadata(LEROBOT_PARQUET_SCHEMA_METADATA)
    pq.write_table(canonical, _parquet_path(root), compression="snappy")
    action = np.asarray(canonical["action"].combine_chunks().to_pylist(), dtype=np.float32)
    _replace_action_stats(root, action)
    manifest_path = root / "meta/shard_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["converter"]["sha256"] = "3" * 64
    _write_json(manifest_path, manifest)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["quaternion_order"] = "wxyz"
    info["source_quaternion_orders"] = {"state": "wxyz", "action": "xyzw"}
    _write_json(info_path, info)
    _refresh_manifest(root)
    return root


def _upgrade_to_r1_extra38(root: Path) -> Path:
    """Create the historical 38-column converter schema."""
    shard = merger._load_shard(root)
    table = pq.read_table(_parquet_path(root))
    table, canonical, converted = merger._legacy_nedf_payload(table, shard)
    assert converted is True
    state = np.asarray(
        table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32
    )
    action = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
    arrays: list[pa.Array | pa.ChunkedArray] = [
        table[name] for name in merger.CORE_PARQUET_COLUMNS
    ]
    names = list(merger.CORE_PARQUET_COLUMNS)
    for side, start in (("left", 0), ("right", 8)):
        prefix = f"observation.extra.{side}"
        for field, value in (
            ("hand_state", canonical[f"{prefix}.hand_state"]),
            ("hand_cmd", pa.array(action[:, start + 7], type=pa.float32())),
            ("raw_hand_state", pa.array(np.arange(table.num_rows), type=pa.float32())),
            ("roi_marker_angle", pa.nulls(table.num_rows, type=pa.float32())),
            ("aligned_timestamp_ns", pa.array(np.arange(table.num_rows), type=pa.float64())),
            ("tcp_state_base", _fixed(state[:, start : start + 7])),
            ("tcp_cmd_base", _fixed(action[:, start : start + 7])),
            ("tcp_state_world", _fixed(state[:, start : start + 7])),
            ("tcp_cmd_world", _fixed(action[:, start : start + 7])),
        ):
            names.append(f"{prefix}.{field}")
            arrays.append(value)
    head = "observation.extra.head"
    identity = np.broadcast_to(
        np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        (table.num_rows, 7),
    )
    for field, value in (
        ("hand_state", pa.nulls(table.num_rows, type=pa.float32())),
        ("raw_hand_state", pa.nulls(table.num_rows, type=pa.float32())),
        ("roi_marker_angle", pa.nulls(table.num_rows, type=pa.float32())),
        ("aligned_timestamp_ns", pa.array(np.arange(table.num_rows), type=pa.float64())),
        ("tcp_state_base", _fixed(identity)),
        ("tcp_cmd_base", _fixed(identity)),
        ("tcp_state_world", _fixed(identity)),
        ("tcp_cmd_world", _fixed(identity)),
    ):
        names.append(f"{head}.{field}")
        arrays.append(value)
    for side in ("right", "left", "ego"):
        names.append(f"slam_diagnostics_valid_mask.{side}")
        arrays.append(pa.array(np.ones(table.num_rows, dtype=np.int8), type=pa.int8()))
    for side in ("right", "left"):
        names.append(f"width_valid_mask.{side}")
        arrays.append(pa.array(np.ones(table.num_rows, dtype=np.int8), type=pa.int8()))
    historical = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(historical, _parquet_path(root), compression="snappy")
    _replace_action_stats(root, action)
    _refresh_manifest(root)
    return root


def _upgrade_to_legacy22(root: Path) -> Path:
    table = pq.read_table(_parquet_path(root))
    action = merger._action_xyzw_to_wxyz(
        np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
    )
    action_index = table.schema.get_field_index("action")
    table = table.set_column(action_index, table.schema.field(action_index), _fixed(action))
    state = np.asarray(
        table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32
    )
    arrays: list[pa.Array | pa.ChunkedArray] = [table[name] for name in merger.CORE_PARQUET_COLUMNS]
    names = list(merger.CORE_PARQUET_COLUMNS)
    for side, start in (("left", 0), ("right", 8), ("ego", 16)):
        prefix = f"observation.extra.{side}"
        arrays.append(_fixed(state[:, start : start + 7]))
        names.append(f"{prefix}.raw_pose")
        arrays.append(
            pa.nulls(table.num_rows, type=pa.float32())
            if side == "ego"
            else pa.array(state[:, start + 7], type=pa.float32())
        )
        names.append(f"{prefix}.hand_state")
        for field in ("raw_hand_state", "roi_marker_angle"):
            arrays.append(pa.nulls(table.num_rows, type=pa.float32()))
            names.append(f"{prefix}.{field}")
        arrays.append(pa.array(np.arange(table.num_rows), type=pa.float64()))
        names.append(f"{prefix}.aligned_timestamp_ns")
    legacy = pa.Table.from_arrays(arrays, names=names)
    legacy = legacy.replace_schema_metadata(
        {b"slam_anomaly_status": b"missing", b"width_anomaly_status": b"missing"}
    )
    pq.write_table(legacy, _parquet_path(root), compression="snappy")
    _replace_action_stats(root, action)
    _refresh_manifest(root)
    return root


def test_merge_rewrites_indices_and_hardlinks_videos(tmp_path: Path) -> None:
    first = _upgrade_to_canonical(
        _make_shard(tmp_path / "first", "a", "task-a", 2)
    )
    second = _upgrade_to_canonical(
        _make_shard(tmp_path / "second", "b", "task-b", 3)
    )
    merged = merger.merge_shards(
        [first, second], tmp_path / "datasets", "combined", link_mode="hardlink"
    )

    info = json.loads((merged / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5
    assert info["total_tasks"] == 2
    assert info["total_videos"] == 2

    episodes = [
        json.loads(line)
        for line in (merged / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    episode_uuids = [str(uuid.UUID(record["episode_uuid"])) for record in episodes]
    assert len(set(episode_uuids)) == 2

    first_table = pq.read_table(
        merged / merger.DATA_PATH.format(episode_chunk=0, episode_index=0)
    )
    second_table = pq.read_table(
        merged / merger.DATA_PATH.format(episode_chunk=0, episode_index=1)
    )
    assert first_table["episode_index"].to_pylist() == [0, 0]
    assert first_table["index"].to_pylist() == [0, 1]
    assert first_table["task_index"].to_pylist() == [0, 0]
    assert second_table["episode_index"].to_pylist() == [1, 1, 1]
    assert second_table["index"].to_pylist() == [0, 1, 2]
    assert second_table["task_index"].to_pylist() == [1, 1, 1]

    source_video = second / merger.VIDEO_PATH.format(
        episode_chunk=0, video_key=VIDEO_FEATURE, episode_index=0
    )
    merged_video = merged / merger.VIDEO_PATH.format(
        episode_chunk=0, video_key=VIDEO_FEATURE, episode_index=1
    )
    assert os.stat(source_video).st_ino == os.stat(merged_video).st_ino
    assert (
        merged / "meta" / "source_sidecars" / "episode_000001" / "dense_labels.jsonl"
    ).is_file()
    merged_joint_sidecars = (
        merged / "meta" / "source_sidecars" / "episode_000001"
    )
    assert json.loads(
        (merged_joint_sidecars / "joint_trajectory.json").read_text()
    )["marker"] == "b"
    assert (merged_joint_sidecars / "joint_trajectory.npz").read_bytes() == b"joint-b"
    assert not (merged_joint_sidecars / "episode_000000").exists()
    validation = json.loads((merged / "meta" / "validation_report.json").read_text())
    assert validation == {
        "episodes": 2,
        "frames": 5,
        "parquet_columns": 22,
        "parquet_schema_version": merger.UNION_SCHEMA_VERSION,
        "status": "passed",
        "validation": "structural",
        "videos": 2,
    }
    assert tuple(first_table.schema.names) == LEROBOT_PARQUET_COLUMNS
    assert first_table.schema.metadata == LEROBOT_PARQUET_SCHEMA_METADATA
    assert not (merged / "meta/stats.json").exists()
    assert set(info) == {
        "chunks_size",
        "codebase_version",
        "data_path",
        "features",
        "fps",
        "nmx_data_version",
        "robot_type",
        "splits",
        "target_fps",
        "total_chunks",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "total_videos",
        "video_path",
    }


def test_merge_rejects_tampered_shard_artifact_before_reading_it(
    tmp_path: Path,
) -> None:
    shard = _make_shard(tmp_path / "shard", "a", "task", 2)
    video = shard / merger.VIDEO_PATH.format(
        episode_chunk=0, video_key=VIDEO_FEATURE, episode_index=0
    )
    video.write_bytes(b"tamper!")

    with pytest.raises(merger.MergeError, match="artifact hash mismatch"):
        merger.merge_shards(
            [shard], tmp_path / "datasets", "combined", link_mode="copy"
        )


def test_duplicate_shard_inputs_receive_distinct_stable_episode_uuids(
    tmp_path: Path,
) -> None:
    shard = _upgrade_to_canonical(
        _make_shard(tmp_path / "shard", "a", "task", 2)
    )
    merged = merger.merge_shards(
        [shard, shard], tmp_path / "datasets", "combined", link_mode="hardlink"
    )
    episodes = [
        json.loads(line)
        for line in (merged / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert episodes[0]["episode_uuid"] == merger._episode_uuid_for_shard(
        merger._load_shard(shard)
    )
    assert episodes[1]["episode_uuid"] != episodes[0]["episode_uuid"]
    assert str(uuid.UUID(episodes[1]["episode_uuid"])) == episodes[1]["episode_uuid"]


def test_merge_rejects_manifest_that_omits_core_metadata(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path / "shard", "a", "task", 2)
    manifest_path = shard / "meta" / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        item
        for item in manifest["artifacts"]
        if item["path"] != "meta/info.json"
    ]
    _write_json(manifest_path, manifest)

    with pytest.raises(merger.MergeError, match="complete shard"):
        merger.merge_shards(
            [shard], tmp_path / "datasets", "combined", link_mode="copy"
        )


def test_merge_rejects_nested_episode_sidecar_directory(tmp_path: Path) -> None:
    shard = _upgrade_to_canonical(
        _make_shard(tmp_path / "shard", "a", "task", 2)
    )
    nested = (
        shard
        / "meta"
        / "source_sidecars"
        / "archive"
        / "episode_000000"
    )
    nested.mkdir(parents=True)
    (nested / "unexpected.json").write_text("{}\n", encoding="utf-8")
    _refresh_manifest(shard)

    with pytest.raises(merger.MergeError, match="ambiguous episode sidecar"):
        merger.merge_shards(
            [shard], tmp_path / "datasets", "combined", link_mode="copy"
        )


def test_merge_rejects_sidecars_that_collide_when_flattened(tmp_path: Path) -> None:
    shard = _upgrade_to_canonical(
        _make_shard(tmp_path / "shard", "a", "task", 2)
    )
    flat = shard / "meta" / "source_sidecars" / "joint_trajectory.json"
    flat.write_text("{}\n", encoding="utf-8")
    _refresh_manifest(shard)

    with pytest.raises(merger.MergeError, match="collide after episode normalization"):
        merger.merge_shards(
            [shard], tmp_path / "datasets", "combined", link_mode="copy"
        )


def test_mixed_core_and_canonical_schemas_reject_legacy_core(tmp_path: Path) -> None:
    legacy = _make_shard(tmp_path / "legacy", "a", "task", 3)
    native = _upgrade_to_canonical(
        _make_shard(tmp_path / "native", "b", "task", 3)
    )
    with pytest.raises(merger.MergeError, match="Legacy 7-column shards are unsupported"):
        merger.merge_shards(
            [legacy, native], tmp_path / "datasets", "mixed", link_mode="hardlink"
        )


def test_nmx22_shards_remain_exactly_nmx22(tmp_path: Path) -> None:
    legacy = _upgrade_to_legacy22(
        _make_shard(tmp_path / "legacy", "a", "task", 2)
    )
    native = _upgrade_to_canonical(
        _make_shard(tmp_path / "native", "b", "task", 2)
    )
    merged = merger.merge_shards(
        [legacy, native], tmp_path / "datasets", "mixed", link_mode="hardlink"
    )
    tables = [
        pq.read_table(path)
        for path in sorted((merged / "data").rglob("*.parquet"))
    ]
    assert len(tables[0].schema.names) == 22
    assert tables[0].schema.remove_metadata().equals(tables[1].schema.remove_metadata())
    assert tables[0]["observation.extra.left.raw_pose"].combine_chunks().values.null_count == 0
    assert tables[1]["observation.extra.left.raw_pose"].combine_chunks().values.null_count == 0
    assert tables[0].schema.metadata == LEROBOT_PARQUET_SCHEMA_METADATA


def test_historical_r1_extra_schema_maps_head_and_tcp_to_nmx_contract(
    tmp_path: Path,
) -> None:
    historical = _upgrade_to_r1_extra38(
        _make_shard(tmp_path / "historical", "a", "task", 2)
    )
    merged = merger.merge_shards(
        [historical], tmp_path / "datasets", "normalized", link_mode="hardlink"
    )
    table = pq.read_table(next((merged / "data").rglob("*.parquet")))
    assert tuple(table.schema.names) == LEROBOT_PARQUET_COLUMNS
    assert table.schema.metadata == LEROBOT_PARQUET_SCHEMA_METADATA
    assert table["observation.extra.left.raw_hand_state"].to_pylist() == [0.0, 1.0]
    assert table["observation.extra.ego.raw_pose"].to_pylist() == [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    manifest = json.loads((merged / "meta/merge_manifest.json").read_text())
    assert manifest["parquet_schema"]["episodes"][0]["source_schema"] == (
        "legacy_r1_extra_v1"
    )


def test_unknown_legacy_converter_hash_fails_closed(tmp_path: Path) -> None:
    legacy = _make_shard(tmp_path / "legacy", "a", "task", 2)
    manifest_path = legacy / "meta/shard_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["converter"]["sha256"] = "f" * 64
    _write_json(manifest_path, manifest)
    shard = merger._load_shard(legacy)
    table = pq.read_table(_parquet_path(legacy))
    with pytest.raises(merger.MergeError, match="unknown converter hash"):
        merger._legacy_nedf_payload(table, shard)


def test_incompatible_shards_are_not_published(tmp_path: Path) -> None:
    first = _make_shard(tmp_path / "first", "a", "task", 2, fps=30)
    second = _make_shard(tmp_path / "second", "b", "task", 2, fps=20)
    output = tmp_path / "datasets"
    with pytest.raises(merger.MergeError, match="Incompatible shard field fps"):
        merger.merge_shards([first, second], output, "combined")
    assert not (output / "combined").exists()
