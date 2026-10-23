"""Recording-profile tests for the I2RT NEDF3 Rerun visualizer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("nmx_msg")
pytest.importorskip("rerun")

from consumer.io.async_data_saver_i2rt_NEDF3 import (  # noqa: E402
    arm_state_message,
    hand_state_message,
    metadata_message,
    rgbd_message,
)
from consumer.io.i2rt_nedf3_rerun import (  # noqa: E402
    decode_legacy_v1_camera_rgb,
    inspect_episode,
    iter_i2rt_records,
    latest_episode,
    visualize_episode,
)
from consumer.io.nedf3_i2rt_contract import (  # noqa: E402
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_CONTRACT_VERSION,
    NEDF3_I2RT_HAND_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_METADATA_TOPIC,
    NEDF3_RECORD_TOPICS,
    NEDF3EpisodeMetadata,
    create_vendor_writer,
)


def _write_episode(path: Path, *, timestamp_ns: int) -> Path:
    metadata = NEDF3EpisodeMetadata(episode_id=path.name, task="rerun_test")
    writer = create_vendor_writer(output_path=path, metadata=metadata)
    writer.write_message(
        NEDF3_METADATA_TOPIC, metadata_message(metadata), timestamp_ns
    )
    for index, topic in enumerate(NEDF3_CAMERA_TOPICS.values()):
        image = np.full((8, 12, 3), 30 + index, dtype=np.uint8)
        writer.write_message(
            topic,
            rgbd_message(image, jpeg_quality=85),
            timestamp_ns + index + 1,
        )
    for role_index, (role, side_topics) in enumerate(
        NEDF3_I2RT_JOINT_TOPICS.items()
    ):
        for side_index, (side, topic) in enumerate(side_topics.items()):
            offset = role_index * 10 + side_index
            writer.write_message(
                topic,
                arm_state_message(np.arange(6, dtype=np.float64) + offset),
                timestamp_ns + 10 + offset,
            )
            writer.write_message(
                NEDF3_I2RT_HAND_TOPICS[role][side],
                hand_state_message(offset / 10),
                timestamp_ns + 30 + offset,
            )
    writer.close()
    (path / "episode_stats.json").write_text(
        json.dumps({"ended_at_ns": timestamp_ns + 100}), encoding="utf-8"
    )
    return path


def test_visualizer_decodes_canonical_i2rt_episode(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path / "episode_0000", timestamp_ns=time.time_ns())

    summary = inspect_episode(episode)
    records = list(iter_i2rt_records(episode))

    assert set(summary.topic_counts) == set(NEDF3_RECORD_TOPICS)
    assert summary.contract_version == NEDF3_I2RT_CONTRACT_VERSION
    assert len(summary.shards) == 1
    assert [record.kind for record in records].count("camera") == 3
    assert [record.kind for record in records].count("joints") == 4
    assert [record.kind for record in records].count("gripper") == 4
    leader_right = next(
        record
        for record in records
        if record.kind == "joints"
        and record.role == "leader"
        and record.side == "right"
    )
    assert leader_right.values == pytest.approx(
        tuple(float(index) for index in range(6))
    )
    assert all(
        record.image.startswith(b"\xff\xd8")
        for record in records
        if record.kind == "camera"
    )


def test_visualizer_exports_rrd_and_selects_latest_episode(tmp_path: Path) -> None:
    older = _write_episode(tmp_path / "task" / "episode_0001", timestamp_ns=1_000)
    newer = _write_episode(tmp_path / "task" / "episode_0002", timestamp_ns=2_000)
    output = tmp_path / "episode_0002.rrd"

    assert latest_episode(tmp_path) == newer
    summary = visualize_episode(newer, output_rrd=output)

    assert summary.episode_dir == newer.resolve()
    assert output.is_file()
    assert output.stat().st_size > 0
    assert older.is_dir()


def test_visualizer_repairs_v1_rgb_bgr_jpeg_order() -> None:
    source_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    source_rgb[:, :] = (240, 30, 10)
    ok, legacy_jpeg = cv2.imencode(
        ".jpg",
        source_rgb,
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    assert ok

    repaired_rgb = decode_legacy_v1_camera_rgb(legacy_jpeg.tobytes())
    mean_rgb = repaired_rgb.reshape(-1, 3).mean(axis=0)

    assert mean_rgb[0] > 220
    assert mean_rgb[2] < 30
