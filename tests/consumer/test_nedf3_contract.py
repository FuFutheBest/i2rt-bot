from __future__ import annotations

import pytest

from consumer.io.nedf3_contract import (
    HAL_CAMERA_TOPICS,
    NEDF3_RECORD_TOPICS,
    NEDF3EpisodeMetadata,
    validate_topics,
)


def test_nedf3_topic_contract_contains_three_camera_streams() -> None:
    assert len(NEDF3_RECORD_TOPICS) == 19
    assert len(HAL_CAMERA_TOPICS) == 3
    assert all(topic.endswith("/rgbd") for topic in HAL_CAMERA_TOPICS)


def test_nedf3_metadata_is_json_shaped() -> None:
    metadata = NEDF3EpisodeMetadata("episode-1", task="pick", device_type=("x1",))
    assert metadata.as_dict()["device_type"] == ["x1"]


def test_nedf3_topic_validation_rejects_incomplete_lists() -> None:
    with pytest.raises(ValueError, match="missing canonical topics"):
        validate_topics((NEDF3_RECORD_TOPICS[0],))
