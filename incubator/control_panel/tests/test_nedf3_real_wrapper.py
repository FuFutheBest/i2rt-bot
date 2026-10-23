from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
import pytest

from incubator.control_panel.conversion_jobs import ConversionJobManager
from incubator.control_panel.converter_profile import default_profile
from incubator.control_panel.recording_contract import (
    LEROBOT_PARQUET_COLUMNS,
    LEROBOT_PARQUET_SCHEMA_METADATA,
    LEROBOT_VIDEO_FEATURES,
)


INFO_FIELDS = {
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


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _video_paths(shard: Path) -> list[Path]:
    return sorted((shard / "videos").rglob("*.mp4"))


def test_real_accepted_job_wrapper_converts_nedf3_without_source_mutation() -> None:
    source_value = os.environ.get("NEDF3_WRAPPER_PROBE_SOURCE", "").strip()
    output_value = os.environ.get("NEDF3_WRAPPER_PROBE_ROOT", "").strip()
    if not source_value or not output_value:
        pytest.skip("real NEDF3 wrapper probe paths were not requested")

    source = Path(source_value).expanduser().resolve(strict=True)
    probe_root = Path(output_value).expanduser().resolve()
    assert not probe_root.exists(), f"probe output already exists: {probe_root}"
    activation_probe = (
        os.environ.get("NEDF3_WRAPPER_ACTIVATION_PROBE", "").strip() == "1"
    )
    activation_receipt_value = os.environ.get(
        "SUPERINFERENCE_CONVERTER_ACTIVATION_RECEIPT", ""
    ).strip()
    activation_receipt = (
        Path(activation_receipt_value).expanduser().resolve()
        if activation_receipt_value
        else None
    )
    if activation_probe:
        assert activation_receipt is not None
        assert not activation_receipt.exists()
    receipt = json.loads(
        (source / ".conversion_source.json").read_text(encoding="utf-8")
    )
    source_hash_before = _tree_hash(source)
    profile = default_profile()
    manager = ConversionJobManager(
        queue_root=probe_root / "queue",
        converter_python=profile.python,
        converter_script=profile.wrapper_script,
        profile_id=profile.profile_id,
        profile_available=lambda: (True, ""),
        auto_start=False,
    )
    job = manager.enqueue(
        receipt,
        source_root=source.parent,
        run_config={
            "head_camera": "camera_2",
            "left_camera": "camera_1",
            "right_camera": "camera_0",
            "fps": 30,
            "camera_workers": 1,
            "wrist_image_size": [320, 240],
            "head_image_size": [240, 320],
            "crop_lrtb": [0, 0, 0, 0],
            "tcp_pose_macro": "none",
            "max_alignment_error_ms": 20.0,
            "task": "nedf3 recorder contract probe",
        },
    )
    job_path = manager.jobs_dir / f"{job['job_id']}.json"
    if activation_probe:
        job["profile"].update(
            {
                "available": False,
                "can_probe": True,
                "state": "awaiting_first_finalized_episode",
                "activation_probe": True,
                "blocker": "awaiting_first_finalized_episode",
            }
        )
        job_path.write_text(
            json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    staging = probe_root / "wrapper-staging"
    completed = subprocess.run(
        [
            str(profile.python),
            str(profile.wrapper_script),
            "--job-spec",
            str(job_path),
            "--staging-dir",
            str(staging),
        ],
        cwd=profile.wrapper_script.parents[2],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    shard = staging / "shard"
    assert _tree_hash(source) == source_hash_before

    info = json.loads((shard / "meta" / "info.json").read_text(encoding="utf-8"))
    assert set(info) == INFO_FIELDS
    assert info["codebase_version"] == "v2.1"
    assert info["nmx_data_version"] == "0.2.30"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 7
    assert info["total_videos"] == 3
    assert info["fps"] == 30
    assert info["target_fps"] == 10.0
    assert not (shard / "meta" / "stats.json").exists()

    parquet = next((shard / "data").rglob("*.parquet"))
    table = pq.read_table(parquet)
    assert tuple(table.schema.names) == LEROBOT_PARQUET_COLUMNS
    assert table.schema.metadata == LEROBOT_PARQUET_SCHEMA_METADATA
    states = np.asarray(
        table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32
    )
    actions = np.asarray(
        table["action"].combine_chunks().to_pylist(), dtype=np.float32
    )
    assert states.shape == (7, 23)
    assert actions.shape == (7, 23)
    np.testing.assert_array_equal(actions[:-1], states[1:])
    np.testing.assert_array_equal(
        table["timestamp"].to_numpy(),
        np.arange(7, dtype=np.float32) / np.float32(30),
    )

    expected_video_keys = set(LEROBOT_VIDEO_FEATURES.values())
    assert {
        path.parent.name for path in _video_paths(shard)
    } == expected_video_keys
    for video in _video_paths(shard):
        probe = subprocess.run(
            [
                str(profile.ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,nb_frames,avg_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        assert probe.returncode == 0, probe.stderr
        streams = json.loads(probe.stdout)["streams"]
        assert streams == [
            {"codec_name": "av1", "avg_frame_rate": "30/1", "nb_frames": "7"}
        ]
        decoded = subprocess.run(
            [
                str(profile.full_decode_ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60,
        )
        assert decoded.returncode == 0, decoded.stderr

    joint_sidecars = shard / "meta" / "source_sidecars" / "episode_000000"
    assert (joint_sidecars / "joint_trajectory.json").is_file()
    with np.load(joint_sidecars / "joint_trajectory.npz", allow_pickle=False) as joints:
        assert set(joints.files) == {"ego", "left", "right", "timestamps_ns"}
        assert joints["left"].shape == (7, 7)
        assert joints["right"].shape == (7, 7)
        assert joints["ego"].shape == (7, 6)
    for name in ("metadata.json", "episode_result.json", "dense_labels.jsonl"):
        assert (shard / "meta" / "source_sidecars" / name).is_file()

    manifest = json.loads(
        (shard / "meta" / "shard_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["job_id"] == job["job_id"]
    assert manifest["source_fingerprint"] == receipt["fingerprint"]
    assert manifest["dataset"]["episode_uuid"] == receipt["episode_id"]
    assert manifest["dataset"]["episode_uuid"] == json.loads(
        (shard / "meta" / "episodes.jsonl").read_text(encoding="utf-8")
    )["episode_uuid"]
    artifact_paths = {item["path"] for item in manifest["artifacts"]}
    assert artifact_paths == {
        path.relative_to(shard).as_posix()
        for path in shard.rglob("*")
        if path.is_file()
        and path != shard / "meta" / "shard_manifest.json"
    }
    for item in manifest["artifacts"]:
        artifact = shard / item["path"]
        assert artifact.stat().st_size == item["size"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == item["sha256"]
    if activation_probe:
        assert activation_receipt is not None and activation_receipt.is_file()
        activation = json.loads(activation_receipt.read_text(encoding="utf-8"))
        assert activation["status"] == "passed"
        assert activation["evidence"]["source_contract"]["declared_topic_count"] == 19
        assert activation["evidence"]["source_contract"]["actual_stream_count"] == 18
        assert activation["conversion"]["manifest_hashes_verified"] is True
        assert activation["merge"]["no_video_reencode"] is True
    print(
        json.dumps(
            {
                "job_id": job["job_id"],
                "shard": str(shard),
                "frames": info["total_frames"],
                "videos": info["total_videos"],
                "source_sha256": source_hash_before,
            },
            sort_keys=True,
        )
    )
