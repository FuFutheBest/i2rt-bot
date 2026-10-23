from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

import numpy as np
import pytest

from incubator.control_panel.tools import convert_accepted_episode as converter
from incubator.control_panel.tools import nedf3_rollout_to_lerobot as rollout_converter
from incubator.control_panel.recording_contract import (
    NEDF3_CAMERA_IDS,
    NEDF3_CAMERA_ROLE_LABELS,
    NEDF3_CAMERA_TOPICS,
    NEDF3_HEAD_TOPICS,
    NEDF3_RECORD_TOPIC_LIST,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _job(
    tmp_path: Path,
    *,
    available: bool = True,
    data_format: str = "NEDF3",
    result_kind: str = "success",
) -> tuple[Path, Path]:
    episode_id = "12345678-1234-5678-9234-567812345678"
    source_root = tmp_path / "source"
    episode = source_root / "episode_0007"
    episode.mkdir(parents=True)
    metadata = episode / "metadata.json"
    result = episode / (
        "episode_incomplete.json" if result_kind == "incomplete" else "episode_result.json"
    )
    dense = episode / "dense_labels.jsonl"
    _write_json(metadata, {"data_format": data_format, "task_name": "test"})
    _write_json(
        result,
        {
            "episode_result": result_kind,
            "is_success": result_kind == "success",
            "episode_id": episode_id,
            "episode_index": 7,
        },
    )
    dense.write_text(
        json.dumps(
            {
                "label": "accept",
                "start_frame": 0,
                "end_frame": 4,
                "interval": "[start_frame,end_frame)",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fingerprint = "a" * 64
    job = {
        "schema_version": 1,
        "job_id": "b" * 64,
        "identity": {
            "source_fingerprint": fingerprint,
            "profile_id": converter.PROFILE.profile_id,
        },
        "state": "running",
        "source": {
            "path": str(episode),
            "root": str(source_root),
            "episode_id": episode_id,
            "episode_index": 7,
            "result": result_kind,
            "data_format": data_format,
            "fingerprint": fingerprint,
            "metadata_sha256": _sha256(metadata),
            "episode_result_sha256": (
                _sha256(result) if result_kind != "incomplete" else None
            ),
            "episode_incomplete_sha256": (
                _sha256(result) if result_kind == "incomplete" else None
            ),
            "dense_labels_sha256": _sha256(dense),
            "files": [],
        },
        "profile": {
            "id": converter.PROFILE.profile_id,
            "available": available,
            "blocker": None if available else "SDK unavailable",
        },
        "run_config": {
            "head_camera": None,
            "left_camera": "left-camera",
            "right_camera": "right-camera",
            "video_encoder": "libsvtav1",
            "gpu_ids": [7],
            "nvenc_preset": "p7",
            "svt_preset": 1,
            # Executable paths in the job are deliberately ignored.
            "ffmpeg_bin": "/untrusted/ffmpeg",
            "converter_script": "/untrusted/converter.py",
        },
        "output": {
            "shard_root": str(tmp_path / "published" / fingerprint),
            "manifest_path": "meta/shard_manifest.json",
        },
        "attempts": [],
        "error": None,
    }
    path = tmp_path / "job.json"
    _write_json(path, job)
    return path, episode


def _fake_runner(commands: list[list[str]], staging: Path):
    def run(command: list[str]) -> None:
        commands.append(list(command))
        root = staging / "shard"
        _write_json(
            root / "meta" / "info.json",
            {
                "codebase_version": "v2.1",
                "total_episodes": 1,
                "total_frames": 4,
                "total_videos": 0,
                "fps": 30,
                "features": {"observation.state": {"dtype": "float32"}},
            },
        )
        _write_json(
            root / "meta" / "validation_report.json",
            {"status": "passed", "episodes": 1},
        )
        (root / "meta" / "episodes.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "episode_uuid": "12345678-1234-5678-9234-567812345678",
                    "length": 4,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        data = root / "data" / "chunk-000" / "episode_000000.parquet"
        data.parent.mkdir(parents=True)
        data.write_bytes(b"fake parquet")

    return run


def test_vendor_checksum_matches_handoff() -> None:
    assert (
        converter._sha256_file(Path(converter.__file__))
        == converter.PROFILE.wrapper_script_sha256
    )
    assert converter._sha256_file(converter.CONVERTER_SCRIPT) == converter.CONVERTER_SHA256
    assert converter.CONVERTER_SHA256 == converter.PROFILE.converter_source_sha256
    assert converter._sha256_file(converter.MERGER_SCRIPT) == converter.MERGER_SHA256


@pytest.mark.parametrize(
    ("path_name", "expected_name", "label"),
    (
        ("CONVERTER_SCRIPT", "CONVERTER_SHA256", "converter"),
        ("MERGER_SCRIPT", "MERGER_SHA256", "merger"),
    ),
)
def test_vendor_checksum_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_name: str,
    expected_name: str,
    label: str,
) -> None:
    tampered = tmp_path / f"{label}.py"
    tampered.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(converter, path_name, tampered)
    monkeypatch.setattr(converter, expected_name, "0" * 64)
    with pytest.raises(converter.JobSpecError, match=f"{label} checksum mismatch"):
        converter._verify_vendor_script()


def test_activation_manifest_covers_the_complete_shard_except_itself(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard"
    for relative, content in (
        ("data/chunk-000/episode_000000.parquet", b"parquet"),
        (
            "videos/chunk-000/observation.images.image/episode_000000.mp4",
            b"video",
        ),
        (
            "meta/source_sidecars/episode_000000/joint_trajectory.npz",
            b"joints",
        ),
        ("meta/info.json", b"{}\n"),
    ):
        path = shard / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    manifest = {"artifacts": converter._artifact_records(shard)}
    assert {item["path"] for item in manifest["artifacts"]} == {
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.image/episode_000000.mp4",
        "meta/source_sidecars/episode_000000/joint_trajectory.npz",
        "meta/info.json",
    }
    converter._verify_manifest_artifacts(shard, manifest)

    sidecar = (
        shard
        / "meta"
        / "source_sidecars"
        / "episode_000000"
        / "joint_trajectory.npz"
    )
    sidecar.write_bytes(b"changed")
    with pytest.raises(converter.JobSpecError, match="artifact size changed"):
        converter._verify_manifest_artifacts(shard, manifest)


def test_activation_source_requires_exact_nedf3_topics_and_camera_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "episode_0000"
    source.mkdir()
    _write_json(
        source / "metadata.json",
        {
            "data_format": "NEDF3",
            "record_topic_list": list(NEDF3_RECORD_TOPIC_LIST),
            "main_camera": NEDF3_CAMERA_IDS["right"],
            "camera_info": {
                NEDF3_CAMERA_IDS[role]: NEDF3_CAMERA_ROLE_LABELS[role]
                for role in ("right", "left", "head")
            },
        },
    )
    raw_topics = {
        topic: [1_000_000_000, 1_033_333_333]
        for topic in NEDF3_RECORD_TOPIC_LIST
        if not topic.endswith("/tcp/vel")
    }

    class Reader:
        def get_raw_topics_timestamps(self):
            return raw_topics

        def iter_decoded_messages(self, topic: str):
            return iter((SimpleNamespace(), SimpleNamespace()))

    class ReaderConfig:
        def __init__(self, *, metadata_file_path: str) -> None:
            self.metadata_file_path = metadata_file_path

    fake_sdk = SimpleNamespace(
        NEDFFactory=SimpleNamespace(get_reader=lambda _config: Reader()),
        NEDFReaderConfig=ReaderConfig,
    )
    monkeypatch.setitem(sys.modules, "dm3_core.nedf", fake_sdk)

    evidence = converter._validate_nedf3_source_contract(source)

    assert evidence["declared_topic_count"] == 27
    assert evidence["actual_stream_count"] == 25
    assert set(evidence["streams"]) == set(raw_topics)

    raw_topics.pop("nmx/hal/arm/left/joint/state")
    with pytest.raises(converter.JobSpecError, match="R1 HAL contract"):
        converter._validate_nedf3_source_contract(source)


def test_activation_source_allows_optional_head_streams_to_be_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "episode_0000"
    source.mkdir()
    wrist_camera_info = {
        NEDF3_CAMERA_IDS[role]: NEDF3_CAMERA_ROLE_LABELS[role]
        for role in ("right", "left")
    }
    _write_json(
        source / "metadata.json",
        {
            "data_format": "NEDF3",
            "record_topic_list": list(NEDF3_RECORD_TOPIC_LIST),
            "main_camera": NEDF3_CAMERA_IDS["right"],
            "camera_info": wrist_camera_info,
        },
    )
    optional_topics = {
        NEDF3_CAMERA_TOPICS["head"],
        *NEDF3_HEAD_TOPICS.values(),
    }
    raw_topics = {
        topic: [1_000_000_000, 1_033_333_333]
        for topic in NEDF3_RECORD_TOPIC_LIST
        if not topic.endswith("/tcp/vel") and topic not in optional_topics
    }

    class Reader:
        def get_raw_topics_timestamps(self):
            return raw_topics

        def iter_decoded_messages(self, topic: str):
            return iter((SimpleNamespace(), SimpleNamespace()))

    class ReaderConfig:
        def __init__(self, *, metadata_file_path: str) -> None:
            self.metadata_file_path = metadata_file_path

    monkeypatch.setitem(
        sys.modules,
        "dm3_core.nedf",
        SimpleNamespace(
            NEDFFactory=SimpleNamespace(get_reader=lambda _config: Reader()),
            NEDFReaderConfig=ReaderConfig,
        ),
    )

    evidence = converter._validate_nedf3_source_contract(source)

    assert evidence["camera_mapping"] == wrist_camera_info
    assert set(evidence["streams"]) == set(raw_topics)


def test_job_builds_fixed_command_and_writes_manifest(tmp_path: Path) -> None:
    job, source = _job(tmp_path)
    source_hashes = _tree_hashes(source)
    staging = (tmp_path / "staging").resolve()
    commands: list[list[str]] = []
    shard = converter.convert_job(
        job, staging, runner=_fake_runner(commands, staging)
    )

    assert shard == staging / "shard"
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == converter.sys.executable
    assert command[1] == str(converter.CONVERTER_SCRIPT)
    assert command[command.index("--head-camera") + 1] == "None"
    assert command[command.index("--left-camera") + 1] == "left-camera"
    assert command[command.index("--right-camera") + 1] == "right-camera"
    assert "/untrusted/ffmpeg" not in command
    assert "/untrusted/converter.py" not in command
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--video-encoder") + 1] == converter.PROFILE.video_encoder
    assert command[command.index("--ffmpeg-bin") + 1] == str(converter.PROFILE.ffmpeg)
    assert command[command.index("--ffprobe-bin") + 1] == str(converter.PROFILE.ffprobe)
    assert command[command.index("--gpu-ids") + 1] == "0"
    assert command[command.index("--nvenc-preset") + 1] == "p4"
    assert command[command.index("--svt-preset") + 1] == "12"
    assert "--success-only" in command
    assert "--fail-on-conversion-error" in command

    manifest = json.loads(
        (shard / "meta" / "shard_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["job_id"] == "b" * 64
    assert manifest["profile_id"] == converter.PROFILE.profile_id
    assert manifest["source_fingerprint"] == "a" * 64
    assert manifest["converter"]["sha256"] == converter.CONVERTER_SHA256
    assert manifest["dataset"]["episode_uuid"] == (
        "12345678-1234-5678-9234-567812345678"
    )
    assert manifest["dense_labels"]["preserved_as_sidecar"] is True
    assert (shard / "meta" / "source_sidecars" / "dense_labels.jsonl").read_bytes() == (
        source / "dense_labels.jsonl"
    ).read_bytes()
    assert _tree_hashes(source) == source_hashes


def test_nedf3_job_preserves_format_manifest_sidecars_and_source(
    tmp_path: Path,
) -> None:
    job, source = _job(tmp_path, data_format="NEDF3")
    source_hashes = _tree_hashes(source)
    staging = (tmp_path / "staging").resolve()

    shard = converter.convert_job(
        job,
        staging,
        runner=_fake_runner([], staging),
    )

    manifest = json.loads(
        (shard / "meta" / "shard_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["data_format"] == "NEDF3"
    preserved_metadata = json.loads(
        (shard / "meta" / "source_sidecars" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert preserved_metadata["data_format"] == "NEDF3"
    assert _tree_hashes(source) == source_hashes


def test_sdk_generated_timestamp_cache_stays_outside_source(tmp_path: Path) -> None:
    job, source = _job(tmp_path, data_format="NEDF3")
    source_hashes = _tree_hashes(source)
    staging = (tmp_path / "staging").resolve()
    commands: list[list[str]] = []
    fake_runner = _fake_runner(commands, staging)

    def cache_writing_runner(command: list[str]) -> None:
        sdk_source = Path(command[command.index("--data-dir") + 1])
        assert sdk_source != source
        assert sdk_source.name == source.name
        (sdk_source / "timestamps.json").write_text(
            '{"generated":true}\n', encoding="utf-8"
        )
        fake_runner(command)

    converter.convert_job(job, staging, runner=cache_writing_runner)

    assert _tree_hashes(source) == source_hashes
    assert not (source / "timestamps.json").exists()
    assert not (staging / "source").exists()


def test_activation_sdk_validation_uses_isolated_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_path, source = _job(tmp_path, available=False, data_format="NEDF3")
    (source / "data.mcap").write_bytes(b"mcap")
    source_hashes = _tree_hashes(source)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["profile"].update({"activation_probe": True, "can_probe": True})
    _write_json(job_path, job)
    staging = (tmp_path / "staging").resolve()

    def validate_sdk_source(sdk_source: Path) -> dict[str, object]:
        assert sdk_source != source
        (sdk_source / "timestamps.json").write_text(
            '{"generated":true}\n', encoding="utf-8"
        )
        return {"validated": True}

    monkeypatch.setattr(
        converter, "_validate_nedf3_source_contract", validate_sdk_source
    )
    monkeypatch.setattr(converter, "_activate_profile", lambda *_args: None)

    converter.convert_job(
        job_path,
        staging,
        runner=_fake_runner([], staging),
    )

    assert _tree_hashes(source) == source_hashes
    assert not (source / "timestamps.json").exists()
    assert not (staging / "source").exists()


def test_incomplete_job_uses_incomplete_sidecar_without_success_filter(
    tmp_path: Path,
) -> None:
    job, source = _job(tmp_path, result_kind="incomplete")
    staging = (tmp_path / "staging").resolve()
    commands: list[list[str]] = []
    shard = converter.convert_job(
        job, staging, runner=_fake_runner(commands, staging)
    )
    assert "--success-only" not in commands[0]
    assert (
        shard / "meta" / "source_sidecars" / "episode_incomplete.json"
    ).is_file()
    manifest = json.loads(
        (shard / "meta" / "shard_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["result"] == "incomplete"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("head_camera", "", "blank strings are invalid"),
        ("head_camera", "None", "must use JSON null"),
        ("left_camera", None, "left_camera must be a non-empty camera id"),
        ("left_camera", "  ", "left_camera must be a non-empty camera id"),
        ("right_camera", None, "right_camera must be a non-empty camera id"),
        ("right_camera", "None", "right_camera must be a non-empty camera id"),
    ),
)
def test_job_spec_rejects_ambiguous_camera_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    job_path, source = _job(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["run_config"][field] = value
    with pytest.raises(converter.JobSpecError, match=message):
        converter._converter_command(job, source, tmp_path / "staging")


def test_job_spec_rejects_noncanonical_lerobot_fps(tmp_path: Path) -> None:
    job_path, source = _job(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["run_config"]["fps"] = 10

    with pytest.raises(converter.JobSpecError, match="fps must be 30"):
        converter._converter_command(job, source, tmp_path / "staging")


def test_job_spec_defaults_rollout_device_type(tmp_path: Path) -> None:
    job_path, source = _job(tmp_path, data_format="NEDF3")
    job = json.loads(job_path.read_text(encoding="utf-8"))

    command = converter._converter_command(job, source, tmp_path / "staging")

    device_type_index = command.index("--device-type") + 1
    tcp_macro_index = command.index("--tcp-pose-macro") + 1
    assert command[device_type_index] == "rollout"
    assert command[tcp_macro_index] == "auto"
    assert (
        rollout_converter._resolve_tcp_pose_macro("auto", "rollout")
        == "r1_world"
    )


def test_explicit_camera_mapping_ignores_metadata_order_and_main_camera() -> None:
    metadata = {
        "camera_info": {
            "WebCameraOpenPi_4": "right wrist",
            "WebCameraOpenPi_2": "head",
            "WebCameraOpenPi_0": "left wrist",
        },
        "main_camera": "WebCameraOpenPi_4",
    }
    assert rollout_converter._resolve_cameras(
        metadata,
        "WebCameraOpenPi_2",
        "WebCameraOpenPi_0",
        "WebCameraOpenPi_4",
    ) == ("WebCameraOpenPi_2", "WebCameraOpenPi_0", "WebCameraOpenPi_4")
    assert rollout_converter._resolve_cameras(
        metadata,
        None,
        "WebCameraOpenPi_0",
        "WebCameraOpenPi_4",
    ) == (None, "WebCameraOpenPi_0", "WebCameraOpenPi_4")


def test_logical_camera_ids_override_physical_run_mapping() -> None:
    metadata = {
        "camera_info": {
            "camera_2": "WebCameraOpenPi_2",
            "camera_0": "WebCameraOpenPi_4",
            "camera_1": "WebCameraOpenPi_0",
        },
        "main_camera": "camera_0",
    }
    assert rollout_converter._resolve_cameras(
        metadata,
        "WebCameraOpenPi_2",
        "WebCameraOpenPi_0",
        "WebCameraOpenPi_4",
    ) == ("camera_2", "camera_1", "camera_0")


def test_converter_schedules_all_three_explicit_camera_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = rollout_converter.EpisodePlan(
        source_dir=tmp_path / "episode_0000",
        source_index=0,
        result="success",
        is_success=True,
        task="test",
        head_camera="WebCameraOpenPi_2",
        left_camera="WebCameraOpenPi_0",
        right_camera="WebCameraOpenPi_4",
        action_topic=rollout_converter.DEFAULT_ACTION_TOPIC,
    )
    aligned = rollout_converter.AlignedEpisode(
        source_timestamps_ns=np.asarray([1], dtype=np.int64),
        base_states=np.zeros((1, rollout_converter.SOURCE_DIM), dtype=np.float32),
        base_actions=np.zeros((1, rollout_converter.SOURCE_DIM), dtype=np.float32),
        states=np.zeros((1, rollout_converter.OUTPUT_DIM), dtype=np.float32),
        actions=np.zeros((1, rollout_converter.OUTPUT_DIM), dtype=np.float32),
        camera_indices={},
        alignment_error_ns={},
    )
    scheduled: list[tuple[str, str | None]] = []

    def convert_camera(*args, **kwargs):
        del kwargs
        feature = args[4]
        camera_id = args[5]
        scheduled.append((feature, camera_id))
        return feature, {}

    monkeypatch.setattr(rollout_converter, "_new_reader", lambda source: object())
    monkeypatch.setattr(
        rollout_converter, "_align_episode", lambda *args, **kwargs: aligned
    )
    monkeypatch.setattr(rollout_converter, "_write_parquet", lambda *args: {})
    monkeypatch.setattr(
        rollout_converter.pq,
        "read_table",
        lambda path: type("Table", (), {"num_rows": 1})(),
    )
    monkeypatch.setattr(rollout_converter, "_convert_episode_camera", convert_camera)
    encoder = rollout_converter.VideoEncoderConfig(
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        encoder="libsvtav1",
        quality=30,
        gop=30,
        svt_preset=12,
        nvenc_preset="p4",
        gpu_ids=(),
        extra_output_args=(),
    )
    artifact = rollout_converter._convert_episode(
        tmp_path / "output",
        plan,
        0,
        (320, 240),
        (240, 320),
        (185, 185, 0, 0),
        "none",
        30,
        50_000_000,
        1,
        encoder,
        0,
    )
    assert artifact.episode_uuid == str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "superinference-nedf3:episode_0000:0:1:1:1",
        )
    )
    assert scheduled == [
        (rollout_converter.HEAD_CAMERA_FEATURE, "WebCameraOpenPi_2"),
        (rollout_converter.LEFT_CAMERA_FEATURE, "WebCameraOpenPi_0"),
        (rollout_converter.RIGHT_CAMERA_FEATURE, "WebCameraOpenPi_4"),
    ]


def test_null_head_camera_is_the_only_black_placeholder_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aligned = rollout_converter.AlignedEpisode(
        source_timestamps_ns=np.asarray([1, 2], dtype=np.int64),
        base_states=np.zeros((2, rollout_converter.SOURCE_DIM), dtype=np.float32),
        base_actions=np.zeros((2, rollout_converter.SOURCE_DIM), dtype=np.float32),
        states=np.zeros((2, rollout_converter.OUTPUT_DIM), dtype=np.float32),
        actions=np.zeros((2, rollout_converter.OUTPUT_DIM), dtype=np.float32),
        camera_indices={},
        alignment_error_ns={},
    )
    plan = rollout_converter.EpisodePlan(
        source_dir=tmp_path / "episode_0000",
        source_index=0,
        result="success",
        is_success=True,
        task="test",
        head_camera=None,
        left_camera="WebCameraOpenPi_0",
        right_camera="WebCameraOpenPi_4",
        action_topic=rollout_converter.DEFAULT_ACTION_TOPIC,
    )
    black_calls = []

    def black_video(*args):
        black_calls.append(args)
        return {"frames": 2}

    monkeypatch.setattr(rollout_converter, "_write_black_video", black_video)
    monkeypatch.setattr(
        rollout_converter,
        "_write_video",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("null head must not read a camera stream")
        ),
    )
    monkeypatch.setattr(rollout_converter, "_validate_video", lambda *args: None)
    encoder = rollout_converter.VideoEncoderConfig(
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        encoder="libsvtav1",
        quality=30,
        gop=30,
        svt_preset=12,
        nvenc_preset="p4",
        gpu_ids=(),
        extra_output_args=(),
    )

    feature, stats = rollout_converter._convert_episode_camera(
        tmp_path,
        plan,
        aligned,
        0,
        rollout_converter.HEAD_CAMERA_FEATURE,
        None,
        (240, 320),
        (0, 0, 0, 0),
        30,
        encoder,
        None,
    )

    assert feature == rollout_converter.HEAD_CAMERA_FEATURE
    assert stats == {"frames": 2}
    assert len(black_calls) == 1


def test_parquet_matches_h200_nmx_contract(tmp_path: Path) -> None:
    length = 2
    base_states = np.zeros((length, rollout_converter.SOURCE_DIM), dtype=np.float32)
    base_actions = np.zeros_like(base_states)
    states = np.zeros((length, rollout_converter.OUTPUT_DIM), dtype=np.float32)
    actions = np.zeros_like(states)
    for values in (base_states, base_actions, states, actions):
        values[:, 3] = 1.0
        values[:, 11] = 1.0
    states[:, 19] = 1.0
    actions[:, 19] = 1.0
    states[:, 7] = [0.01, 0.02]
    states[:, 15] = [0.03, 0.04]
    aligned = rollout_converter.AlignedEpisode(
        source_timestamps_ns=np.asarray([100, 200], dtype=np.int64),
        base_states=base_states,
        base_actions=base_actions,
        states=states,
        actions=actions,
        camera_indices={},
        alignment_error_ns={},
    )
    path = tmp_path / "episode.parquet"

    rollout_converter._write_parquet(path, aligned, 7, 30)

    table = rollout_converter.pq.read_table(path)
    assert tuple(table.schema.names) == rollout_converter.LEROBOT_PARQUET_COLUMNS
    assert table.schema.metadata == rollout_converter.LEROBOT_PARQUET_SCHEMA_METADATA
    np.testing.assert_allclose(
        table["observation.extra.left.hand_state"].to_numpy(), [0.01, 0.02]
    )
    np.testing.assert_allclose(
        table["observation.extra.right.hand_state"].to_numpy(), [0.03, 0.04]
    )
    assert table["observation.extra.ego.hand_state"].null_count == length
    assert table["observation.extra.ego.raw_pose"].to_pylist() == [
        rollout_converter.EGO_IDENTITY_POSE.tolist(),
        rollout_converter.EGO_IDENTITY_POSE.tolist(),
    ]
    assert table["frame_index"].to_pylist() == [0, 1]
    assert table["episode_index"].to_pylist() == [7, 7]
    assert table["index"].to_pylist() == [0, 1]


def test_strict_validator_rejects_action_that_is_not_the_next_state(
    tmp_path: Path,
) -> None:
    length = 3
    states = np.zeros((length, rollout_converter.OUTPUT_DIM), dtype=np.float32)
    for start in (3, 11, 19):
        states[:, start] = 1.0
    states[:, 0] = [0.0, 1.0, 2.0]
    actions = states.copy()
    actions[:-1] = states[1:]
    aligned = rollout_converter.AlignedEpisode(
        source_timestamps_ns=np.asarray([100, 200, 300], dtype=np.int64),
        base_states=states[:, : rollout_converter.SOURCE_DIM],
        base_actions=actions[:, : rollout_converter.SOURCE_DIM],
        states=states,
        actions=actions,
        camera_indices={},
        alignment_error_ns={},
    )
    path = tmp_path / "episode.parquet"
    rollout_converter._write_parquet(path, aligned, 0, 30)
    table = rollout_converter.pq.read_table(path)

    rollout_converter._validate_state_action_contract(
        table, path, require_next_state=True
    )

    broken_actions = actions.copy()
    broken_actions[0, 0] += 0.5
    broken = table.set_column(
        table.schema.get_field_index("action"),
        "action",
        rollout_converter._fixed_float_list(
            broken_actions, rollout_converter.OUTPUT_DIM
        ),
    )
    with pytest.raises(rollout_converter.ConversionError, match=r"action\[t\]"):
        rollout_converter._validate_state_action_contract(
            broken, path, require_next_state=True
        )


def test_converter_rejects_noncanonical_output_fps_before_reading_source() -> None:
    with pytest.raises(rollout_converter.ConversionError, match="requires --fps 30"):
        rollout_converter.convert(SimpleNamespace(fps=10))


def test_metadata_matches_h200_nmx_contract(tmp_path: Path) -> None:
    artifact = rollout_converter.EpisodeArtifact(
        episode_index=0,
        episode_uuid=str(uuid.uuid4()),
        source_index=7,
        source_name="episode_0007",
        result="success",
        is_success=True,
        task="test task",
        length=2,
        stats={},
        alignment_summary={},
    )
    cameras = {
        rollout_converter.HEAD_CAMERA_FEATURE: "camera_0",
        rollout_converter.LEFT_CAMERA_FEATURE: "camera_1",
        rollout_converter.RIGHT_CAMERA_FEATURE: "camera_2",
    }

    rollout_converter._finalize_metadata(
        tmp_path,
        "rollout",
        [artifact],
        (320, 240),
        (240, 320),
        cameras,
        30,
    )

    info = json.loads((tmp_path / "meta/info.json").read_text(encoding="utf-8"))
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
    assert info["features"][rollout_converter.LEFT_CAMERA_FEATURE]["shape"] == [
        320,
        240,
        3,
    ]
    assert info["features"][rollout_converter.HEAD_CAMERA_FEATURE]["shape"] == [
        240,
        320,
        3,
    ]
    episode = json.loads(
        (tmp_path / "meta/episodes.jsonl").read_text(encoding="utf-8").strip()
    )
    assert set(episode) == {
        "action_config",
        "collection_mode",
        "device_type",
        "episode_index",
        "episode_uuid",
        "hardware_string",
        "instruct_ch",
        "length",
        "rp_type",
        "slam_validity_all",
        "task_annotation",
        "task_description",
        "tasks",
        "width_calibration",
        "width_detection",
    }
    assert episode["device_type"] == "rollout"
    episode_stats = json.loads(
        (tmp_path / "meta/episodes_stats.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert episode_stats == {"episode_index": 0, "stats": {}}
    assert not (tmp_path / "meta/stats.json").exists()


def test_unavailable_profile_never_runs_converter(tmp_path: Path) -> None:
    job, _ = _job(tmp_path, available=False)
    called = False

    def runner(_command: list[str]) -> None:
        nonlocal called
        called = True

    with pytest.raises(converter.JobSpecError, match="SDK unavailable"):
        converter.convert_job(job, (tmp_path / "staging").resolve(), runner=runner)
    assert called is False



def test_source_must_remain_inside_declared_root(tmp_path: Path) -> None:
    job_path, _ = _job(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["source"]["root"] = str((tmp_path / "somewhere-else").resolve())
    Path(job["source"]["root"]).mkdir()
    _write_json(job_path, job)
    with pytest.raises(converter.JobSpecError, match="escapes source.root"):
        converter.convert_job(job_path, (tmp_path / "staging").resolve())
