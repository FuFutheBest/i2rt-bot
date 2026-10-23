from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from incubator.control_panel.recording_contract import (
    NEDF3_ARM_TOPICS,
    NEDF3_CAMERA_TOPICS,
    NEDF3_HAND_TOPICS,
)
from incubator.control_panel import episode_anomaly_detector as anomaly


def _record(timestamp_ns: int, decoded_message: object) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(log_time=timestamp_ns),
        decoded_message=decoded_message,
    )


class _UnifiedReader:
    def __init__(self, streams: dict[str, list[SimpleNamespace]]) -> None:
        self.streams = streams

    def iter_decoded_messages(self, topic: str):
        return iter(self.streams.get(topic, ()))


def _install_reader(monkeypatch, reader: _UnifiedReader) -> None:
    monkeypatch.setattr(
        anomaly,
        "NEDFFactory",
        SimpleNamespace(get_reader=lambda _config: reader),
    )
    monkeypatch.setattr(
        anomaly,
        "NEDFReaderConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )


def _pose(x: float) -> SimpleNamespace:
    return SimpleNamespace(
        pose=SimpleNamespace(
            pos=SimpleNamespace(x=x, y=x + 1.0, z=x + 2.0),
            quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )


def _gripper(width: float) -> SimpleNamespace:
    return SimpleNamespace(
        joints=[SimpleNamespace(name="gripper.width", pos=width)]
    )


def test_anomaly_loader_reads_canonical_nedf3_streams(monkeypatch, tmp_path) -> None:
    timestamps = [1_000_000_000, 2_000_000_000]
    streams: dict[str, list[SimpleNamespace]] = {}
    for role, payload in (("left", b"left-jpeg"), ("right", b"right-jpeg")):
        streams[NEDF3_CAMERA_TOPICS[role]] = [
            _record(
                timestamp,
                SimpleNamespace(
                    rgb=SimpleNamespace(data=payload, encoded_format="jpeg")
                ),
            )
            for timestamp in timestamps
        ]
        streams[NEDF3_ARM_TOPICS[role]["tcp/state"]] = [
            _record(timestamp, _pose(float(index)))
            for index, timestamp in enumerate(timestamps)
        ]
        streams[NEDF3_HAND_TOPICS[role]["state"]] = [
            _record(timestamp, _gripper(0.01 + index * 0.01))
            for index, timestamp in enumerate(timestamps)
        ]
    _install_reader(monkeypatch, _UnifiedReader(streams))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "data_format": "NEDF3",
                "camera_info": {
                    "camera_0": "right_wrist",
                    "camera_1": "left_wrist",
                },
            }
        ),
        encoding="utf-8",
    )
    mapping = {
        "img_left": "left-camera",
        "img_right": "right-camera",
        "pose_left": "robot_left",
        "pose_right": "robot_right",
        "grip_left": "gripper_left",
        "grip_right": "gripper_right",
    }

    data, relative_timestamps = anomaly.EpisodeAnomalyDetector({})._load_episode_data(
        tmp_path, mapping
    )

    assert bytes(data["left-camera"][0]) == b"left-jpeg"
    assert bytes(data["right-camera"][0]) == b"right-jpeg"
    assert data["robot_left"].shape == (2, 7)
    assert data["robot_right"].shape == (2, 7)
    np.testing.assert_allclose(data["gripper_left"].ravel(), [0.01, 0.02])
    np.testing.assert_allclose(relative_timestamps["robot_left"], [0.0, 1.0])
