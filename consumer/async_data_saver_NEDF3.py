"""Compatibility facade for the dependency-light NEDF3 recorder."""

from consumer.io.nedf3_contract import (
    HAL_CAMERA_TOPICS,
    NEDF3_RECORD_TOPIC_LIST,
    NEDF3_RECORD_TOPICS,
    NEDF3EpisodeMetadata,
)
from consumer.io.nedf3_recorder import NEDF3Recorder

__all__ = [
    "HAL_CAMERA_TOPICS",
    "NEDF3_RECORD_TOPIC_LIST",
    "NEDF3_RECORD_TOPICS",
    "NEDF3EpisodeMetadata",
    "NEDF3Recorder",
]
