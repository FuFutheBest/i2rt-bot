import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import pytest

from incubator.control_panel.recording_replay import (
    RecordingReplayBuildFailed,
    RecordingReplayBuilding,
    RecordingReplayNotFound,
    RecordingReplayRequestError,
    RecordingReplayService,
    RecordingReplayUnavailable,
)
from incubator.control_panel.recording_replay_cache import CACHE_SCHEMA_VERSION
from incubator.control_panel import server


def _episode(root: Path, name: str = "episode_0001") -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "data_format": "NEDF3",
                "task_name": "pick item",
                "camera_info": {"camera_left": "left"},
            }
        ),
        encoding="utf-8",
    )
    (episode / "episode_result.json").write_text(
        json.dumps(
            {
                "episode_result": "success",
                "episode_index": 1,
                "episode_id": "opaque-episode-id",
            }
        ),
        encoding="utf-8",
    )
    (episode / "dense_labels.jsonl").write_text(
        json.dumps(
            {
                "label": "accept",
                "start_frame": 0,
                "end_frame": 7,
                "start_timestamp_ns": 100,
                "end_timestamp_ns": 700,
                "checkpoint_revision": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return episode


def test_new_finalized_episode_requires_applied_decision_for_acceptance(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    (episode / ".conversion_source.json").write_text(
        json.dumps({"decision_required": True}), encoding="utf-8"
    )
    metadata = RecordingReplayService._episode_metadata(episode)
    assert metadata["status"] == "incomplete"

    (episode / ".recording_decision.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "action": "episode_accept",
                "episode_id": "opaque-episode-id",
            }
        ),
        encoding="utf-8",
    )
    assert RecordingReplayService._episode_metadata(episode)["status"] == "accepted"


def test_applied_reject_decision_overrides_success_finalize_result(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    (episode / ".conversion_source.json").write_text(
        json.dumps({"decision_required": True}), encoding="utf-8"
    )
    (episode / ".recording_decision.json").write_text(
        json.dumps({"status": "rejected", "action": "episode_discard"}),
        encoding="utf-8",
    )
    assert RecordingReplayService._episode_metadata(episode)["status"] == "rejected"


def _summary(
    *, joints: bool = True, tcp: bool = False, grippers: bool = False
) -> dict:
    topic_capabilities = {}
    if joints:
        topic_capabilities.update({
            "/lowdim/joint_left": {
                "topic": "/lowdim/joint_left",
                "semantic": "joint_position_rad",
                "available": True,
                "count": 7,
                "dimension": 7,
                "timestamp_key": "topic_000_timestamps_ns",
                "value_key": "topic_000_values",
                "reason": None,
            },
            "/lowdim/joint_right": {
                "topic": "/lowdim/joint_right",
                "semantic": "joint_position_rad",
                "available": True,
                "count": 7,
                "dimension": 7,
                "timestamp_key": "topic_001_timestamps_ns",
                "value_key": "topic_001_values",
                "reason": None,
            },
        })
    if tcp:
        topic_capabilities.update({
            "/lowdim/robot_left": {
                "topic": "/lowdim/robot_left",
                "semantic": "tcp_pose_xyz_qw_qx_qy_qz",
                "available": True,
                "count": 7,
                "dimension": 7,
                "timestamp_key": "topic_002_timestamps_ns",
                "value_key": "topic_002_values",
                "reason": None,
            },
            "/lowdim/robot_right": {
                "topic": "/lowdim/robot_right",
                "semantic": "tcp_pose_xyz_qw_qx_qy_qz",
                "available": True,
                "count": 7,
                "dimension": 7,
                "timestamp_key": "topic_003_timestamps_ns",
                "value_key": "topic_003_values",
                "reason": None,
            },
        })
    if grippers:
        topic_capabilities.update({
            "/lowdim/gripper_left": {
                "topic": "/lowdim/gripper_left",
                "semantic": "gripper_width",
                "available": True,
                "count": 7,
                "dimension": 1,
                "timestamp_key": "topic_004_timestamps_ns",
                "value_key": "topic_004_values",
                "reason": None,
            },
            "/lowdim/gripper_right": {
                "topic": "/lowdim/gripper_right",
                "semantic": "gripper_width",
                "available": True,
                "count": 7,
                "dimension": 1,
                "timestamp_key": "topic_005_timestamps_ns",
                "value_key": "topic_005_values",
                "reason": None,
            },
        })
    return {
        "schema_version": 1,
        "time_range_ns": {"start": 100, "end": 700},
        "topics": [{"name": "/lowdim/joint_left", "count": 7}],
        "cameras": [{"id": "camera_left", "count": 7}],
        "joints": {
            "left": {
                "available": joints,
                "topic": "/lowdim/joint_left" if joints else None,
                "count": 7 if joints else 0,
                "dimension": 7 if joints else None,
            },
            "right": {
                "available": joints,
                "topic": "/lowdim/joint_right" if joints else None,
                "count": 7 if joints else 0,
                "dimension": 7 if joints else None,
            },
        },
        "tcp": {
            "left": {
                "available": tcp,
                "topic": "/lowdim/robot_left" if tcp else None,
                "count": 7 if tcp else 0,
                "dimension": 7 if tcp else None,
            },
            "right": {
                "available": tcp,
                "topic": "/lowdim/robot_right" if tcp else None,
                "count": 7 if tcp else 0,
                "dimension": 7 if tcp else None,
            },
        },
        "grippers": {
            "left": {
                "available": grippers,
                "topic": "/lowdim/gripper_left" if grippers else None,
                "count": 7 if grippers else 0,
                "dimension": 1 if grippers else None,
            },
            "right": {
                "available": grippers,
                "topic": "/lowdim/gripper_right" if grippers else None,
                "count": 7 if grippers else 0,
                "dimension": 1 if grippers else None,
            },
        },
        "capabilities": {
            "schema_version": 1,
            "topics": topic_capabilities,
            "urdf_replay": joints,
            "tcp_replay": tcp,
        },
        "timeline": {"schema_version": 1, "array_count": 5, "uncompressed_bytes": 504},
        "issues": [],
    }


def _runner_for(payload, *, timestamps_by_topic=None, values_by_topic=None):
    calls = []
    timestamps_by_topic = dict(timestamps_by_topic or {})
    values_by_topic = dict(values_by_topic or {})

    def run(command, **kwargs):
        calls.append((command, kwargs))
        value = payload(command) if callable(payload) else payload
        if "--timeline-output" in command:
            output = Path(command[command.index("--timeline-output") + 1])
            arrays = {"cache_schema_version": np.asarray([1], dtype=np.int16)}
            topics = value.get("capabilities", {}).get("topics", {})
            for capability in topics.values():
                if not capability.get("available"):
                    continue
                count = int(capability["count"])
                dimension = int(capability["dimension"])
                topic = capability["topic"]
                timestamps = timestamps_by_topic.get(topic)
                if timestamps is None:
                    timestamps = np.arange(count, dtype=np.int64) * 100 + 100
                timestamps = np.asarray(timestamps, dtype=np.int64)
                assert timestamps.shape == (count,)
                values = values_by_topic.get(topic)
                if values is None:
                    fill = 1.0 if "right" in topic else 0.0
                    if "robot_" in topic:
                        fill += 2.0
                    values = np.full((count, dimension), fill, dtype=np.float32)
                values = np.asarray(values, dtype=np.float32)
                assert values.shape == (count, dimension)
                arrays[capability["timestamp_key"]] = timestamps
                arrays[capability["value_key"]] = values
            with output.open("wb") as stream:
                np.savez(stream, **arrays)
        if "--video-output-dir" in command:
            output_dir = Path(command[command.index("--video-output-dir") + 1])
            for video in value.get("replay_videos", {}).values():
                filename = video.get("filename") if isinstance(video, dict) else None
                if isinstance(filename, str):
                    (output_dir / filename).write_bytes(b"cached-h264-video")
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")

    run.calls = calls
    return run


def _ready(call, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return call()
        except RecordingReplayBuilding:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def test_list_missing_configured_root_is_empty(tmp_path):
    service = RecordingReplayService(
        lambda: tmp_path / "does-not-exist",
        cache_root=tmp_path / "cache",
    )
    result = service.list()
    assert result["items"] == []
    assert result["root_configured"] is True
    assert result["root_available"] is False
    assert result["root_state"] == "missing"


def test_release_converter_environment_is_resolved_at_service_creation(
    tmp_path, monkeypatch
):
    converter_python = tmp_path / "release" / "bin" / "python-converter"
    full_decode = tmp_path / "release" / "full-decode" / "ffmpeg"
    av1_only = tmp_path / "release" / "av1" / "ffmpeg"
    monkeypatch.setenv("SUPERINFERENCE_CONVERTER_PYTHON", str(converter_python))
    monkeypatch.setenv(
        "SUPERINFERENCE_CONVERTER_FULL_DECODE_FFMPEG", str(full_decode)
    )
    monkeypatch.setenv("SUPERINFERENCE_CONVERTER_FFMPEG", str(av1_only))

    service = RecordingReplayService(
        lambda: tmp_path / "rollouts",
        cache_root=tmp_path / "cache",
    )

    assert service.inspector_python == converter_python.absolute()
    assert service.replay_ffmpeg == full_decode.resolve()


def test_replay_runtime_paths_are_part_of_cache_profile(tmp_path):
    first = RecordingReplayService(
        lambda: tmp_path / "rollouts",
        inspector_python=Path("/runtime/one/python"),
        replay_ffmpeg=Path("/runtime/one/ffmpeg"),
        cache_root=tmp_path / "cache",
    )
    second = RecordingReplayService(
        lambda: tmp_path / "rollouts",
        inspector_python=Path("/runtime/two/python"),
        replay_ffmpeg=Path("/runtime/two/ffmpeg"),
        cache_root=tmp_path / "cache",
    )
    assert first._profile_fingerprint({}) != second._profile_fingerprint({})


def test_replay_runtime_python_tracks_selected_config(tmp_path):
    selected = [Path("/runtime/recording-one/python")]
    service = RecordingReplayService(
        lambda: tmp_path / "rollouts",
        inspector_python=Path("/runtime/fallback/python"),
        inspector_python_provider=lambda: selected[0],
        replay_ffmpeg=None,
        cache_root=tmp_path / "cache",
    )

    first = service._profile_fingerprint({})
    assert service._effective_inspector_python() == selected[0]

    selected[0] = Path("/runtime/recording-two/python")

    assert service._effective_inspector_python() == selected[0]
    assert service._profile_fingerprint({}) != first


def test_list_is_paginated_and_sanitized(tmp_path):
    root = tmp_path / "rollouts"
    first = _episode(root, "episode_0001")
    _episode(root, "episode_0002")
    (root / "not-an-episode").mkdir()
    (root / "episode_escape").symlink_to(tmp_path)
    result = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
    ).list(page=1, page_size=1, status="accepted")
    assert result["total"] == 2
    assert result["total_pages"] == 2
    assert len(result["items"]) == 1
    assert result["root_state"] == "ready"
    assert "path" not in result["items"][0]
    assert first.is_dir()


def test_list_and_summary_support_legacy_and_dated_episode_roots(tmp_path):
    root = tmp_path / "rollouts"
    direct = _episode(root, "episode_0001")
    dated = _episode(root / "20260715", "episode_0001")
    runner = _runner_for(_summary())
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )

    listing = service.list()
    assert {item["id"] for item in listing["items"]} == {
        "episode_0001",
        "20260715:episode_0001",
    }
    assert service._episode("episode_0001") == direct.resolve()
    assert service._episode("20260715:episode_0001") == dated.resolve()
    summary = _ready(lambda: service.summary("20260715:episode_0001"))
    assert summary["id"] == "20260715:episode_0001"
    assert summary["name"] == "20260715 / episode_0001"


def test_list_filters_by_date_status_and_search(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root, "episode_0001")
    dated = _episode(root / "20260715", "episode_0002")
    metadata = json.loads((dated / "metadata.json").read_text(encoding="utf-8"))
    metadata["task_name"] = "find the blue cup"
    (dated / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    successful_incomplete = _episode(root / "20260715", "episode_0003")
    (successful_incomplete / ".conversion_source.json").write_text(
        json.dumps({"decision_required": True}), encoding="utf-8"
    )
    failed = _episode(root / "20260715", "episode_0004")
    (failed / "episode_result.json").write_text(
        json.dumps({"episode_result": "failure"}), encoding="utf-8"
    )
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
    )

    all_items = service.list()
    assert {row["id"] for row in all_items["dates"]} == {"legacy", "20260715"}
    assert all_items["dates"][0]["id"] == "20260715"

    filtered = service.list(date="20260715", status="accepted", query="blue cup")
    assert filtered["total"] == 1
    assert filtered["items"][0]["id"] == "20260715:episode_0002"
    assert filtered["items"][0]["date"] == "20260715"
    assert filtered["filters"] == {
        "date": "20260715",
        "status": "accepted",
        "result": "all",
        "query": "blue cup",
    }

    successful = service.list(date="20260715", result="success")
    assert {item["id"] for item in successful["items"]} == {
        "20260715:episode_0002",
        "20260715:episode_0003",
    }
    assert {item["status"] for item in successful["items"]} == {
        "accepted",
        "incomplete",
    }

    assert service.list(date="legacy")["items"][0]["id"] == "episode_0001"
    with pytest.raises(RecordingReplayRequestError):
        service.list(date="../20260715")
    with pytest.raises(RecordingReplayRequestError):
        service.list(query="x" * 121)
    with pytest.raises(RecordingReplayRequestError):
        service.list(result="not-a-result")


@pytest.mark.parametrize(
    "episode_id",
    ["2026071:episode_0001", "20260715:../episode_0001", "bad:episode_0001"],
)
def test_dated_episode_ids_reject_unsafe_paths(tmp_path, episode_id):
    service = RecordingReplayService(
        lambda: tmp_path / "rollouts",
        cache_root=tmp_path / "cache",
    )
    with pytest.raises(RecordingReplayRequestError):
        service.summary(episode_id)


@pytest.mark.parametrize("episode_id", ["../episode_0001", "episode_0001/x", "", "."])
def test_episode_path_traversal_is_rejected(tmp_path, episode_id):
    root = tmp_path / "rollouts"
    _episode(root)
    service = RecordingReplayService(lambda: root, cache_root=tmp_path / "cache")
    with pytest.raises(RecordingReplayRequestError):
        service.summary(episode_id)


def test_episode_symlink_is_not_followed(tmp_path):
    root = tmp_path / "rollouts"
    root.mkdir()
    outside = _episode(tmp_path / "outside")
    (root / "episode_link").symlink_to(outside, target_is_directory=True)
    service = RecordingReplayService(lambda: root, cache_root=tmp_path / "cache")
    with pytest.raises(RecordingReplayNotFound):
        service.summary("episode_link")


def test_summary_uses_external_snapshot_cache_and_returns_dense_checkpoints(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    runner = _runner_for(_summary())
    service = RecordingReplayService(
        lambda: root,
        inspector_python=Path("/trusted/python"),
        inspector_script=Path("/trusted/inspect.py"),
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )
    first = _ready(lambda: service.summary("episode_0001"))
    second = _ready(lambda: service.summary("episode_0001"))
    assert first == second
    assert len(runner.calls) == 1
    assert first["joints"]["left"]["available"] is True
    assert first["dense"]["segments"][0]["label"] == "accept"
    assert first["dense"]["checkpoints"][0]["frame"] == 7
    command, kwargs = runner.calls[0]
    assert command[:2] == ["/usr/bin/prlimit", command[1]]
    assert "/usr/bin/nice" in command and "/usr/bin/ionice" in command
    assert "--mode" in command and "snapshot" in command
    assert Path(command[command.index("--episode") + 1]) != root / "episode_0001"
    assert kwargs["env"] == {
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    assert not any(argument.startswith("--nproc=") for argument in command)


def test_real_limited_subprocess_imports_numpy_with_single_thread_env(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    inspector = tmp_path / "inspector.py"
    inspector.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import os
            import numpy as np

            parser = argparse.ArgumentParser()
            parser.add_argument("--episode")
            parser.add_argument("--mode")
            parser.add_argument("--timeline-output")
            parser.add_argument("--video-output-dir")
            parser.add_argument("--ffmpeg-bin")
            parser.add_argument("--camera-role", action="append", default=[])
            args = parser.parse_args()
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                assert os.environ.get(name) == "1", (name, os.environ.get(name))
            with open(args.timeline_output, "wb") as stream:
                np.savez(stream, cache_schema_version=np.asarray([1], dtype=np.int16))
            print(json.dumps({
                "schema_version": 1,
                "time_range_ns": {"start": None, "end": None},
                "topics": [],
                "cameras": [],
                "joints": {
                    "left": {"available": False, "topic": None, "count": 0, "dimension": None},
                    "right": {"available": False, "topic": None, "count": 0, "dimension": None},
                },
                "capabilities": {"schema_version": 1, "topics": {}, "urdf_replay": False},
                "issues": [],
            }))
            """
        ),
        encoding="utf-8",
    )
    service = RecordingReplayService(
        lambda: root,
        inspector_python=Path(sys.executable),
        inspector_script=inspector,
        cache_root=tmp_path / "cache",
        process_runner=subprocess.run,
        build_wait_s=0.5,
    )
    assert _ready(lambda: service.summary("episode_0001"))["cache_state"] == "ready"


def test_inspector_timeout_enters_persisted_failed_backoff(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=timeout,
        timeout_s=1,
    )
    with pytest.raises(RecordingReplayBuildFailed, match="timed out"):
        _ready(lambda: service.summary("episode_0001"))


def test_missing_recorded_joints_returns_unavailable_without_tcp_ik(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    runner = _runner_for(_summary(joints=False))
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )
    with pytest.raises(RecordingReplayUnavailable, match="TCP poses are not converted"):
        _ready(lambda: service.trajectory("episode_0001"))
    assert len(runner.calls) == 1


def test_playback_trajectory_supports_tcp_only_historical_episode(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    runner = _runner_for(_summary(joints=False, tcp=True))
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )

    value = _ready(
        lambda: service.playback_trajectory(
            "episode_0001", start=1, end=6, stride=2, max_points=3
        )
    )
    assert value["schema_version"] == 3
    assert value["timestamps_ns"] == [200, 400, 600]
    assert value["joints"]["available"] is False
    assert value["tcp"]["left_available"] is True
    assert value["tcp"]["right_available"] is True
    assert value["tcp"]["left"][0] == [2.0] * 7
    assert value["tcp"]["right"][0] == [3.0] * 7
    with pytest.raises(RecordingReplayUnavailable) as unavailable:
        service.trajectory("episode_0001")
    assert unavailable.value.code == "joint_trajectory_unavailable"


def test_playback_trajectory_returns_joint_tcp_and_grippers_without_ik(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=_runner_for(
            _summary(joints=True, tcp=True, grippers=True)
        ),
    )

    value = _ready(lambda: service.playback_trajectory("episode_0001"))
    assert value["total_frames"] == 7
    assert value["returned_frames"] == 7
    assert value["joints"]["left_available"] is True
    assert value["joints"]["right_available"] is True
    assert value["tcp"]["left_available"] is True
    assert value["tcp"]["right_available"] is True
    assert value["grippers"]["left_available"] is True
    assert value["grippers"]["right_available"] is True
    assert value["grippers"]["left"][0] == [0.0]
    assert value["grippers"]["right"][0] == [1.0]


def test_playback_trajectory_resamples_independent_stream_timestamps(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    summary = _summary(joints=True, tcp=True)
    for capability in summary["capabilities"]["topics"].values():
        capability["count"] = 3
    for group in ("joints", "tcp"):
        for side in ("left", "right"):
            summary[group][side]["count"] = 3

    row_values = {
        topic: np.repeat(
            (np.arange(3, dtype=np.float32) + offset)[:, None], 7, axis=1
        )
        for topic, offset in {
            "/lowdim/joint_left": 10.0,
            "/lowdim/joint_right": 20.0,
            "/lowdim/robot_left": 30.0,
            "/lowdim/robot_right": 40.0,
        }.items()
    }
    runner = _runner_for(
        summary,
        timestamps_by_topic={
            "/lowdim/joint_left": [50, 250, 450],
            "/lowdim/joint_right": [80, 280, 480],
            "/lowdim/robot_left": [100, 300, 500],
            "/lowdim/robot_right": [90, 310, 490],
        },
        values_by_topic=row_values,
    )
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )

    value = _ready(lambda: service.playback_trajectory("episode_0001"))
    assert value["timestamps_ns"] == [100, 300, 500]
    assert [row[0] for row in value["joints"]["left"]] == [10.0, 11.0, 12.0]
    assert [row[0] for row in value["joints"]["right"]] == [20.0, 21.0, 22.0]
    assert [row[0] for row in value["tcp"]["right"]] == [40.0, 41.0, 42.0]


def test_playback_trajectory_reports_explicit_unavailable_code(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=_runner_for(_summary(joints=False, tcp=False)),
    )

    with pytest.raises(RecordingReplayUnavailable) as unavailable:
        _ready(lambda: service.playback_trajectory("episode_0001"))
    assert unavailable.value.code == "recording_trajectory_unavailable"
    assert unavailable.value.availability["tcp"]["left"]["available"] is False


def test_trajectory_is_loaded_by_bounded_inspector_request(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)

    runner = _runner_for(_summary())
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )
    value = _ready(
        lambda: service.trajectory(
            "episode_0001", start=1, end=6, stride=2, max_points=3
        )
    )
    assert value["id"] == "episode_0001"
    assert value["returned_frames"] == 3
    assert value["timestamps_ns"] == [200, 400, 600]
    assert len(runner.calls) == 1
    with pytest.raises(RecordingReplayRequestError):
        service.trajectory("episode_0001", stride=0)


def test_malformed_inspector_output_is_sanitized(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)

    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not json", stderr="secret path")

    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
    )
    with pytest.raises(RecordingReplayBuildFailed, match="malformed data"):
        _ready(lambda: service.summary("episode_0001"))


def test_sdk_generated_timestamps_stay_in_external_shadow(tmp_path):
    root = tmp_path / "rollouts"
    episode = _episode(root)
    source_before = {
        path.relative_to(episode).as_posix(): path.read_bytes()
        for path in episode.rglob("*")
        if path.is_file()
    }
    base_runner = _runner_for(_summary())

    def runner(command, **kwargs):
        shadow = Path(command[command.index("--episode") + 1])
        assert shadow != episode
        (shadow / "timestamps.json").write_text('{"generated":true}\n', encoding="utf-8")
        return base_runner(command, **kwargs)

    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "external-cache",
        process_runner=runner,
    )
    _ready(lambda: service.summary("episode_0001"))
    _ready(lambda: service.summary("episode_0001"))
    source_after = {
        path.relative_to(episode).as_posix(): path.read_bytes()
        for path in episode.rglob("*")
        if path.is_file()
    }
    assert source_after == source_before
    assert "timestamps.json" not in source_after
    assert len(base_runner.calls) == 1
    assert list((tmp_path / "external-cache").rglob("timeline.npz"))
    assert not list((tmp_path / "external-cache").rglob("timestamps.json"))


def test_episode_index_reuses_unchanged_metadata_and_refreshes_sidecar(tmp_path):
    root = tmp_path / "rollouts"
    episode = _episode(root)
    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        list_cache_ttl_s=0,
    )
    calls = []

    def loader(path):
        calls.append(path.name)
        return RecordingReplayService._episode_metadata(path)

    service._episode_metadata = loader
    assert service.list()["total"] == 1
    assert calls == ["episode_0001"]
    assert service.list()["total"] == 1
    assert calls == ["episode_0001"]
    result_path = episode / "episode_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["episode_result"] = "failure"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    refreshed = service.list(status="rejected")
    assert refreshed["total"] == 1
    assert calls == ["episode_0001", "episode_0001"]


def test_concurrent_recording_lists_share_one_index_scan(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    service = RecordingReplayService(lambda: root, cache_root=tmp_path / "cache")
    original = service._cache.episode_index
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_index(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        release.wait(timeout=2)
        return original(*args, **kwargs)

    service._cache.episode_index = blocked_index
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(service.list()["total"]))
        for _ in range(5)
    ]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=1)
    release.set()
    for thread in threads:
        thread.join(timeout=1)

    assert results == [1] * 5
    assert calls == 1


def test_existing_empty_root_reports_explicit_state(tmp_path):
    root = tmp_path / "rollouts"
    root.mkdir()
    value = RecordingReplayService(lambda: root, cache_root=tmp_path / "cache").list()
    assert value["root_available"] is True
    assert value["root_state"] == "empty"


def test_concurrent_cache_misses_enqueue_only_one_build(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    release = threading.Event()
    started = threading.Event()
    base_runner = _runner_for(_summary())

    def runner(command, **kwargs):
        started.set()
        release.wait(timeout=2)
        return base_runner(command, **kwargs)

    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
        build_wait_s=0,
    )
    results = []

    def request_summary():
        try:
            service.summary("episode_0001")
        except RecordingReplayBuilding:
            results.append("building")

    threads = [threading.Thread(target=request_summary) for _ in range(8)]
    for thread in threads:
        thread.start()
    assert started.wait(timeout=1)
    for thread in threads:
        thread.join(timeout=1)
    assert results == ["building"] * 8
    assert len(base_runner.calls) == 0
    release.set()
    value = _ready(lambda: service.summary("episode_0001"))
    assert value["cache_state"] == "ready"
    assert len(base_runner.calls) == 1


def test_failed_build_backoff_prevents_refresh_retries(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=3, stdout="", stderr="secret")

    service = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
        build_wait_s=0.5,
    )
    with pytest.raises(RecordingReplayBuildFailed):
        service.summary("episode_0001")
    with pytest.raises(RecordingReplayBuildFailed):
        service.summary("episode_0001")
    assert len(calls) == 1


def test_finalize_hint_enqueue_returns_without_waiting_for_filesystem(tmp_path):
    root = tmp_path / "rollouts"
    episode = _episode(root)
    entered = threading.Event()
    release = threading.Event()

    def blocked_root_provider():
        entered.set()
        release.wait(timeout=2)
        return root

    service = RecordingReplayService(
        blocked_root_provider,
        cache_root=tmp_path / "cache",
        process_runner=_runner_for(_summary()),
    )
    started = time.monotonic()
    with service._build_guard:
        assert service.enqueue_finalized(
            {
                "schema_version": 1,
                "result": "success",
                "path": str(episode),
            }
        ) is True
    elapsed = time.monotonic() - started
    assert elapsed < 0.05
    assert entered.wait(timeout=1)
    release.set()


def test_finalize_hint_rejects_failure_discard_and_incomplete(tmp_path):
    service = RecordingReplayService(
        lambda: tmp_path / "rollouts",
        cache_root=tmp_path / "cache",
    )
    absolute = str((tmp_path / "rollouts" / "episode_0001").resolve())
    assert service.enqueue_finalized({"result": "failure", "path": absolute}) is False
    assert service.enqueue_finalized({"result": "discard", "path": absolute}) is False
    assert service.enqueue_finalized({"result": "incomplete", "path": absolute}) is False


def test_corrupt_generation_is_quarantined_and_rebuilt_once(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    runner = _runner_for(_summary())
    cache_root = tmp_path / "cache"
    first_service = RecordingReplayService(
        lambda: root,
        cache_root=cache_root,
        process_runner=runner,
        build_wait_s=0.5,
    )
    _ready(lambda: first_service.summary("episode_0001"))
    timeline = next(cache_root.rglob("timeline.npz"))
    timeline.write_bytes(b"corrupt")
    second_service = RecordingReplayService(
        lambda: root,
        cache_root=cache_root,
        process_runner=runner,
        build_wait_s=0.5,
    )
    rebuilt = _ready(lambda: second_service.summary("episode_0001"))
    assert rebuilt["cache_state"] == "ready"
    assert len(runner.calls) == 2
    assert list(
        (cache_root / f"v{CACHE_SCHEMA_VERSION}" / "quarantine").iterdir()
    )


def test_trajectory_npz_failure_quarantines_and_reenqueues(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    runner = _runner_for(_summary())
    cache_root = tmp_path / "cache"
    service = RecordingReplayService(
        lambda: root,
        cache_root=cache_root,
        process_runner=runner,
        build_wait_s=0.5,
    )
    _ready(lambda: service.summary("episode_0001"))
    timeline = next(cache_root.rglob("timeline.npz"))
    manifest = json.loads((timeline.parent / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["files"]["timeline.npz"]
    timeline.write_bytes(b"x" * expected["size"])
    os.utime(timeline, ns=(expected["mtime_ns"], expected["mtime_ns"]))
    service.build_wait_s = 0
    with pytest.raises(RecordingReplayBuilding):
        service.trajectory("episode_0001")
    service.build_wait_s = 0.5
    recovered = _ready(lambda: service.trajectory("episode_0001"))
    assert recovered["cache_state"] == "ready"
    assert len(runner.calls) == 2
    assert list(
        (cache_root / f"v{CACHE_SCHEMA_VERSION}" / "quarantine").iterdir()
    )


def test_flock_loser_does_not_overwrite_ready_state_with_failed(tmp_path, monkeypatch):
    import incubator.control_panel.recording_replay as replay_module

    root = tmp_path / "rollouts"
    episode = _episode(root)
    runner = _runner_for(_summary())
    cache_root = tmp_path / "cache"
    owner = RecordingReplayService(
        lambda: root,
        cache_root=cache_root,
        process_runner=runner,
        build_wait_s=0.5,
    )
    _ready(lambda: owner.summary("episode_0001"))
    loser = RecordingReplayService(
        lambda: root,
        cache_root=cache_root,
        process_runner=runner,
        build_wait_s=0,
    )
    identity = loser._cache.source_identity(episode)
    entry = loser._cache.entry(
        episode,
        identity.fingerprint,
        loser._profile_fingerprint(loser._expected_camera_roles()),
    )
    monkeypatch.setattr(replay_module, "DEFAULT_BUILD_LOCK_WAIT_S", 0.05)
    with owner._cache.entry_lock(entry, timeout_s=None) as acquired:
        assert acquired
        with pytest.raises(RecordingReplayBuilding):
            loser.summary("episode_0001")
        deadline = time.monotonic() + 1
        while loser._build_events and time.monotonic() < deadline:
            time.sleep(0.01)
        assert loser._cache.read_build_state(entry)["status"] == "ready"


def test_recording_routes_expose_async_cache_contract(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    release = threading.Event()
    started = threading.Event()
    base_runner = _runner_for(_summary())

    def runner(command, **kwargs):
        started.set()
        release.wait(timeout=2)
        return base_runner(command, **kwargs)

    replay = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=runner,
        build_wait_s=0.01,
    )
    app = server.create_app(config_dir=tmp_path / "configs", recording_replay=replay)
    app.testing = True
    client = app.test_client()
    building = client.get("/api/recordings/episode_0001/summary")
    assert building.status_code == 202
    assert building.get_json() == {
        "cache_state": "building",
        "retry_after_ms": 250,
        "episode_id": "episode_0001",
    }
    assert started.wait(timeout=1)
    release.set()
    deadline = time.monotonic() + 2
    while True:
        ready = client.get("/api/recordings/episode_0001/summary")
        if ready.status_code == 200 or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    assert ready.status_code == 200
    assert ready.get_json()["cache_state"] == "ready"
    app.config["DEVICE_CONTROLLER"].shutdown()


def test_recording_routes_map_status_and_reject_bad_queries(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    replay = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=_runner_for(_summary(joints=False)),
    )
    app = server.create_app(
        config_dir=tmp_path / "configs",
        recording_replay=replay,
    )
    app.testing = True
    client = app.test_client()

    listing = client.get("/api/recordings?page=1&page_size=20")
    assert listing.status_code == 200
    assert listing.get_json()["total"] == 1
    assert client.get("/api/recordings?result=success").get_json()["total"] == 1
    assert client.get("/api/recordings?result=failure").get_json()["total"] == 0
    assert client.get("/recordings").status_code == 200
    assert client.get("/api/recordings?page=oops").status_code == 422
    assert client.get("/api/recordings?result=invalid").status_code == 422
    assert client.get("/api/recordings/../episode_0001/summary").status_code in {404, 422}
    _ready(lambda: replay.summary("episode_0001"))
    unavailable = client.get("/api/recordings/episode_0001/joint-trajectory")
    assert unavailable.status_code == 409
    assert unavailable.get_json()["code"] == "joint_trajectory_unavailable"

    app.config["DEVICE_CONTROLLER"].shutdown()


def test_recording_delete_requires_confirmation_and_removes_source_and_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rollouts"
    episode = _episode(root)
    (episode / "data.mcap").write_bytes(b"recording-bytes")
    replay = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        replay_ffmpeg=None,
    )
    identity = replay._cache.source_identity(episode)
    cache_entry = replay._cache.entry(episode, identity.fingerprint, "profile")
    cache_entry.directory.mkdir(parents=True)
    (cache_entry.directory / "video.mp4").write_bytes(b"cached-video")
    app = server.create_app(
        config_dir=tmp_path / "configs",
        recording_replay=replay,
    )
    app.testing = True
    client = app.test_client()

    unconfirmed = client.delete(
        "/api/recordings",
        json={"episode_ids": ["episode_0001"]},
    )
    assert unconfirmed.status_code == 422
    assert episode.is_dir()

    deleted = client.delete(
        "/api/recordings",
        json={"episode_ids": ["episode_0001"], "confirm": True},
    )
    assert deleted.status_code == 200
    assert deleted.get_json()["deleted"] == ["episode_0001"]
    assert deleted.get_json()["freed_bytes"] >= len(b"recording-bytes")
    assert not episode.exists()
    assert not cache_entry.directory.exists()
    assert replay.list()["total"] == 0
    app.config["DEVICE_CONTROLLER"].shutdown()


def test_recording_delete_rejects_active_episode(tmp_path: Path) -> None:
    root = tmp_path / "rollouts"
    active = _episode(root, "episode_0002_temp")
    replay = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        replay_ffmpeg=None,
    )

    with pytest.raises(RecordingReplayRequestError, match="Active recording"):
        replay.delete(["episode_0002_temp"])

    assert active.is_dir()


def test_recording_summary_uses_recorded_config_for_replay_visuals(tmp_path):
    root = tmp_path / "rollouts"
    episode = _episode(root)
    (episode / ".conversion_source.json").write_text(
        json.dumps({"config_name": "recorded_config"}),
        encoding="utf-8",
    )
    replay = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=_runner_for(_summary(joints=True, tcp=True)),
    )
    app = server.create_app(
        config_dir=tmp_path / "configs",
        recording_replay=replay,
    )
    app.testing = True
    controller = app.config["DEVICE_CONTROLLER"]
    controller.inspect_config = lambda name: {
        "name": name,
        "configured_tools": {
            "left": {"status": "not-configured", "visual": {"model": "umi3"}},
            "right": {"status": "not-configured", "visual": {"model": "umi3"}},
        },
        "camera_slots": {
            "head": {"configured": False},
            "left": {"configured": True},
            "right": {"configured": True},
        },
    }
    _ready(lambda: replay.summary("episode_0001"))

    response = app.test_client().get(
        "/api/recordings/episode_0001/summary"
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["config_name"] == "recorded_config"
    assert payload["visualization"]["source"] == "recorded-config"
    assert payload["visualization"]["configured_tools"]["left"]["visual"]["model"] == "umi3"
    assert payload["camera_roles"]["head"]["status"] == "not_configured"
    assert payload["camera_roles"]["head"]["reason"] is None
    controller.shutdown()


def test_recording_routes_fall_back_to_persistent_nedf3_video_cache(tmp_path):
    root = tmp_path / "rollouts"
    _episode(root)
    payload = _summary(joints=False, tcp=True)
    payload["replay_videos"] = {
        "left": {
            "available": True,
            "role": "left",
            "camera_id": "camera_left",
            "filename": "video_left.mp4",
            "frame_count": 7,
            "fps": 30,
            "codec": "h264",
        }
    }
    replay = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=_runner_for(payload),
    )
    app = server.create_app(
        config_dir=tmp_path / "configs",
        recording_replay=replay,
    )
    app.testing = True
    client = app.test_client()

    summary = _ready(lambda: replay.summary("episode_0001"))
    assert summary["videos"]["left"]["available"] is True
    assert summary["videos"]["left"]["source"] == "nedf3_cache"
    assert summary["videos"]["left"]["feature"] == "camera_left"
    assert summary["videos"]["head"]["available"] is False
    response = client.get("/api/recordings/episode_0001/videos/left")
    assert response.status_code == 200
    assert response.data == b"cached-h264-video"
    app.config["DEVICE_CONTROLLER"].shutdown()


def test_recording_routes_expose_synchronized_converted_videos_with_ranges(tmp_path):
    root = tmp_path / "rollouts"
    episode = _episode(root)
    shard = tmp_path / "published-shard"
    features = {
        "head": "observation.images.image",
        "left": "observation.images.wrist_image_1",
        "right": "observation.images.wrist_image_2",
    }
    for index, feature in enumerate(features.values()):
        video = shard / "videos" / "chunk-000" / feature / "episode_000000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(bytes(range(index * 16, index * 16 + 16)))

    replay = RecordingReplayService(
        lambda: root,
        cache_root=tmp_path / "cache",
        process_runner=_runner_for(_summary(joints=False, tcp=True)),
        published_shard_provider=lambda path: shard if path == episode.resolve() else None,
    )
    app = server.create_app(
        config_dir=tmp_path / "configs",
        recording_replay=replay,
    )
    app.testing = True
    client = app.test_client()

    _ready(lambda: replay.summary("episode_0001"))
    summary = client.get("/api/recordings/episode_0001/summary")
    assert summary.status_code == 200
    videos = summary.get_json()["videos"]
    assert all(videos[role]["available"] for role in features)
    assert videos["head"]["url"].endswith("/videos/head")
    assert videos["left"]["feature"] == "observation.images.wrist_image_1"
    assert videos["right"]["feature"] == "observation.images.wrist_image_2"

    ranged = client.get(
        "/api/recordings/episode_0001/videos/head",
        headers={"Range": "bytes=2-5"},
    )
    assert ranged.status_code == 206
    assert ranged.data == bytes(range(2, 6))
    assert ranged.headers["Accept-Ranges"] == "bytes"
    assert ranged.headers["Content-Range"].startswith("bytes 2-5/")
    assert client.get(
        "/api/recordings/episode_0001/videos/invalid"
    ).status_code == 422

    app.config["DEVICE_CONTROLLER"].shutdown()
