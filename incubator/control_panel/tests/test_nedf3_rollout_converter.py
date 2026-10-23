from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from incubator.control_panel.recording_contract import (
    NEDF3_ARM_TOPICS,
    NEDF3_CAMERA_IDS,
    NEDF3_DATA_FORMAT,
    NEDF3_HAND_TOPICS,
    NEDF3_HEAD_TOPICS,
)
from incubator.control_panel.tools import nedf3_rollout_to_lerobot as converter


def _record(timestamp_ns: int, decoded: object) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(log_time=timestamp_ns, publish_time=timestamp_ns),
        decoded_message=decoded,
    )


def _pose(timestamp_ns: int, x: float) -> SimpleNamespace:
    return _record(
        timestamp_ns,
        SimpleNamespace(
            pose=SimpleNamespace(
                pos=SimpleNamespace(x=x, y=0.1, z=0.2),
                quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
            )
        ),
    )


def _state(timestamp_ns: int, name: str, value: float) -> SimpleNamespace:
    return _record(
        timestamp_ns,
        SimpleNamespace(joints=[SimpleNamespace(name=name, pos=value)]),
    )


def _joints(timestamp_ns: int, joints: list[float]) -> SimpleNamespace:
    return _record(
        timestamp_ns,
        SimpleNamespace(
            joints=[
                SimpleNamespace(name=f"joint{index}", pos=value)
                for index, value in enumerate(joints)
            ]
        ),
    )


class _Reader:
    def __init__(self, records: dict[str, list[SimpleNamespace]]) -> None:
        self.records = records

    def get_raw_topics_timestamps(self) -> dict[str, list[int]]:
        return {
            topic: [record.message.log_time for record in values]
            for topic, values in self.records.items()
        }

    def iter_decoded_messages(self, topic: str):
        yield from self.records.get(topic, ())


def _reader(*, include_ego: bool = True) -> _Reader:
    timestamps = (100, 200, 300, 400)
    records: dict[str, list[SimpleNamespace]] = {}
    for side, offset in (("left", 1.0), ("right", 10.0)):
        records[NEDF3_ARM_TOPICS[side]["tcp/state"]] = [
            _pose(timestamp, offset + index)
            for index, timestamp in enumerate(timestamps)
        ]
        records[NEDF3_ARM_TOPICS[side]["tcp/cmd"]] = [
            _pose(timestamp, offset + 1000 + index)
            for index, timestamp in enumerate(timestamps)
        ]
        records[NEDF3_HAND_TOPICS[side]["state"]] = [
            _state(timestamp, "gripper", offset / 100 + index / 1000)
            for index, timestamp in enumerate(timestamps)
        ]
        records[NEDF3_HAND_TOPICS[side]["cmd"]] = [
            _state(timestamp, "gripper", offset / 10 + index / 100)
            for index, timestamp in enumerate(timestamps)
        ]
        records[NEDF3_ARM_TOPICS[side]["joint/state"]] = [
            _joints(timestamp, [offset + index] * 7)
            for index, timestamp in enumerate(timestamps)
        ]
    if include_ego:
        records[NEDF3_HEAD_TOPICS["state"]] = [
            _pose(timestamp, 100.0 + index)
            for index, timestamp in enumerate(timestamps)
        ]
        records[NEDF3_HEAD_TOPICS["cmd"]] = [
            _pose(timestamp, 200.0 + index)
            for index, timestamp in enumerate(timestamps)
        ]
        records[NEDF3_HEAD_TOPICS["joint/state"]] = [
            _joints(timestamp, [100.0 + index] * 6)
            for index, timestamp in enumerate(timestamps)
        ]
    for camera_id in NEDF3_CAMERA_IDS.values():
        topic = f"nmx/hal/camera/{camera_id}/rgbd"
        records[topic] = [
            _record(timestamp, SimpleNamespace()) for timestamp in timestamps
        ]
    return _Reader(records)


def _plan(*, ego_required: bool = True) -> converter.EpisodePlan:
    return converter.EpisodePlan(
        source_dir=Path("episode_0000"),
        source_index=0,
        result="success",
        is_success=True,
        task="test",
        head_camera=NEDF3_CAMERA_IDS["head"],
        left_camera=NEDF3_CAMERA_IDS["left"],
        right_camera=NEDF3_CAMERA_IDS["right"],
        action_topic="",
        data_format=NEDF3_DATA_FORMAT,
        main_camera=NEDF3_CAMERA_IDS["head"],
        ego_required=ego_required,
    )


def test_nedf3_alignment_uses_recorded_commands_and_preserves_ego() -> None:
    aligned = converter._align_nedf3_episode(
        _reader(), _plan(), max_alignment_error_ns=0, tcp_pose_macro="none"
    )

    assert aligned.states.shape == (4, 23)
    np.testing.assert_allclose(aligned.base_actions[:, 0], [1001, 1002, 1003, 1004])
    assert not np.array_equal(aligned.actions[:-1], aligned.states[1:])
    np.testing.assert_allclose(aligned.base_states[:, 16], [100, 101, 102, 103])
    np.testing.assert_allclose(aligned.base_actions[:, 16], [200, 201, 202, 203])
    assert aligned.joint_positions["left"].shape == (4, 7)
    assert aligned.joint_positions["right"].shape == (4, 7)
    assert aligned.joint_positions["ego"].shape == (4, 6)
    np.testing.assert_array_equal(aligned.source_timestamps_ns, [100, 200, 300, 400])


def test_lerobot_episode_uuid_preserves_source_episode_uuid() -> None:
    source_uuid = "12345678-1234-5678-9234-567812345678"
    plan = replace(_plan(), source_episode_id=source_uuid)
    aligned = converter._align_nedf3_episode(
        _reader(), plan, max_alignment_error_ns=0, tcp_pose_macro="none"
    )

    assert converter._stable_episode_uuid(plan, aligned) == source_uuid


def test_nedf3_required_ego_cannot_be_fabricated() -> None:
    with pytest.raises(converter.ConversionError, match="required NEDF3 ego"):
        converter._align_nedf3_episode(
            _reader(include_ego=False),
            _plan(ego_required=True),
            max_alignment_error_ns=0,
            tcp_pose_macro="none",
        )


def test_nedf3_without_yam_uses_explicit_identity_ego() -> None:
    aligned = converter._align_nedf3_episode(
        _reader(include_ego=False),
        _plan(ego_required=False),
        max_alignment_error_ns=0,
        tcp_pose_macro="none",
    )
    expected = np.broadcast_to(converter.EGO_IDENTITY_POSE, (4, 7))
    np.testing.assert_array_equal(aligned.base_states[:, 16:23], expected)


def test_nedf3_camera_iterator_decodes_rgbd_jpeg() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    encoded, payload = cv2.imencode(".jpg", image)
    assert encoded
    topic = f"nmx/hal/camera/{NEDF3_CAMERA_IDS['right']}/rgbd"
    reader = _Reader(
        {
            topic: [
                _record(
                    100,
                    SimpleNamespace(
                        rgb=SimpleNamespace(
                            data=payload.tobytes(), encoded_format="jpeg"
                        )
                    ),
                )
            ]
        }
    )

    frames = list(
        converter._iter_selected_images(
            reader,
            NEDF3_CAMERA_IDS["right"],
            np.asarray([0], dtype=np.int64),
            NEDF3_DATA_FORMAT,
        )
    )

    assert len(frames) == 1
    assert frames[0].shape == (4, 6, 3)
    assert frames[0][0, 0, 0] > 240


def test_nedf3_joint_sidecar_is_self_describing(tmp_path: Path) -> None:
    aligned = converter._align_nedf3_episode(
        _reader(), _plan(), max_alignment_error_ns=0, tcp_pose_macro="none"
    )
    paths = converter._write_joint_sidecar(tmp_path, aligned, 0)

    assert paths == [
        "meta/source_sidecars/episode_000000/joint_trajectory.npz",
        "meta/source_sidecars/episode_000000/joint_trajectory.json",
    ]
    archive = np.load(tmp_path / paths[0], allow_pickle=False)
    np.testing.assert_array_equal(archive["timestamps_ns"], [100, 200, 300, 400])
    assert archive["left"].shape == (4, 7)
    assert archive["right"].shape == (4, 7)
    assert archive["ego"].shape == (4, 6)
    manifest = json.loads((tmp_path / paths[1]).read_text(encoding="utf-8"))
    assert manifest["source_data_format"] == NEDF3_DATA_FORMAT
    assert manifest["dimensions"] == {"left": 7, "right": 7, "ego": 6}
