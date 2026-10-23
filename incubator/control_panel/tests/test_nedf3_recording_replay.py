from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from incubator.control_panel.recording_contract import (
    NEDF3_ARM_TOPICS,
    NEDF3_CAMERA_TOPICS,
    NEDF3_HAND_TOPICS,
)
from incubator.control_panel.tools import inspect_nedf3_episode as inspector


def _record(timestamp_ns: int, decoded_message: object) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(log_time=timestamp_ns),
        decoded_message=decoded_message,
    )


def _pose(timestamp_ns: int, offset: float) -> SimpleNamespace:
    return _record(
        timestamp_ns,
        SimpleNamespace(
            pose=SimpleNamespace(
                pos=SimpleNamespace(x=offset, y=offset + 1.0, z=offset + 2.0),
                quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
            )
        ),
    )


def _hand(timestamp_ns: int, width: float) -> SimpleNamespace:
    return _record(
        timestamp_ns,
        SimpleNamespace(
            joints=[SimpleNamespace(name="gripper.width", pos=width)]
        ),
    )


def _joint_state(timestamp_ns: int, joints: list[float]) -> SimpleNamespace:
    return _record(
        timestamp_ns,
        SimpleNamespace(joints=[SimpleNamespace(name=f"joint{i}", pos=value) for i, value in enumerate(joints)]),
    )


class _Reader:
    def __init__(self, streams: dict[str, list[SimpleNamespace]]) -> None:
        self.streams = streams

    def iter_decoded_messages(self, topic: str):
        return iter(self.streams.get(topic, ()))


def _fixture() -> tuple[_Reader, dict[str, object], dict[str, list[int]]]:
    timestamps = [1_000_000_000, 1_033_333_333, 1_066_666_666]
    streams: dict[str, list[SimpleNamespace]] = {}
    for side, offset in (("left", 0.1), ("right", 1.1)):
        streams[NEDF3_ARM_TOPICS[side]["tcp/state"]] = [
            _pose(timestamp, offset + index) for index, timestamp in enumerate(timestamps)
        ]
        streams[NEDF3_HAND_TOPICS[side]["state"]] = [
            _hand(timestamp, offset + index / 100.0)
            for index, timestamp in enumerate(timestamps)
        ]
        streams[NEDF3_ARM_TOPICS[side]["joint/state"]] = [
            _joint_state(timestamp, [offset + index] * 7)
            for index, timestamp in enumerate(timestamps)
        ]
    raw_topics = {topic: list(timestamps) for topic in streams}
    for topic in NEDF3_CAMERA_TOPICS.values():
        raw_topics[topic] = list(timestamps)
    metadata: dict[str, object] = {
        "data_format": "NEDF3",
        "camera_info": {
            "camera_0": "right_wrist",
            "camera_1": "left_wrist",
            "camera_2": "head",
        },
    }
    return _Reader(streams), metadata, raw_topics


def test_nedf3_snapshot_builds_replay_timelines(monkeypatch, tmp_path) -> None:
    reader, metadata, raw_topics = _fixture()
    monkeypatch.setattr(
        inspector,
        "_reader_and_topics",
        lambda _episode: (reader, metadata, raw_topics),
    )
    timeline = tmp_path / "timeline.npz"
    timeline.touch()

    result = inspector.inspect_snapshot(tmp_path, timeline)

    assert result["data_format"] == "NEDF3"
    assert result["joints"]["left"] == {
        "available": True,
        "topic": NEDF3_ARM_TOPICS["left"]["joint/state"],
        "count": 3,
        "dimension": 7,
    }
    assert result["tcp"]["right"]["available"] is True
    assert result["grippers"]["left"]["available"] is True
    assert result["capabilities"]["urdf_replay"] is True
    assert result["capabilities"]["tcp_replay"] is True
    assert result["issues"] == []
    with np.load(timeline, allow_pickle=False) as arrays:
        assert arrays["topic_000_values"].shape == (3, 7)
        assert arrays["topic_002_values"].shape == (3, 1)


def test_nedf3_trajectory_uses_arm_joint_state(
    monkeypatch, tmp_path
) -> None:
    reader, metadata, raw_topics = _fixture()
    monkeypatch.setattr(
        inspector,
        "_reader_and_topics",
        lambda _episode: (reader, metadata, raw_topics),
    )

    result = inspector.inspect_trajectory(
        tmp_path,
        start=1,
        end=None,
        stride=1,
        max_points=10,
    )

    assert result["data_format"] == "NEDF3"
    assert result["total_frames"] == 3
    assert result["returned_frames"] == 2
    assert result["timestamps_ns"] == [1_033_333_333, 1_066_666_666]
    assert result["left"][0] == [1.100000023841858] * 7
    assert result["right"][0] == [2.0999999046325684] * 7
