"""Dependency-light NEDF3 recording contract.

The core runtime owns topic names and episode metadata shape.  The optional
``nmx_nedf_api`` writer is resolved only when a recorder is started, so CI can
validate the contract without installing the vendor SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

NEDF3_RECORD_TOPICS: tuple[str, ...] = (
    "nmx/nedf/metadata",
    "nmx/hal/camera/camera_0/rgbd",
    "nmx/hal/umi/right/pose",
    "nmx/hal/umi/right/hand_state",
    "nmx/hal/umi/right/raw_hand_state",
    "nmx/hal/umi/right/slam",
    "nmx/hal/umi/right/extra",
    "nmx/hal/camera/camera_1/rgbd",
    "nmx/hal/umi/left/pose",
    "nmx/hal/umi/left/hand_state",
    "nmx/hal/umi/left/raw_hand_state",
    "nmx/hal/umi/left/slam",
    "nmx/hal/umi/left/extra",
    "nmx/hal/camera/camera_2/rgbd",
    "nmx/hal/umi/ego/pose",
    "nmx/hal/umi/ego/hand_state",
    "nmx/hal/umi/ego/raw_hand_state",
    "nmx/hal/umi/ego/slam",
    "nmx/hal/umi/ego/extra",
)
# Historical name used by the branch recorder; keep it as a read-only alias.
NEDF3_RECORD_TOPIC_LIST = NEDF3_RECORD_TOPICS
HAL_CAMERA_TOPICS: tuple[str, ...] = tuple(
    topic for topic in NEDF3_RECORD_TOPICS if "/camera/" in topic
)


class NEDF3Writer(Protocol):
    def write_message(self, topic: str, message: Any, timestamp_ns: int) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class NEDF3EpisodeMetadata:
    """Stable metadata fields shared by live recording and offline tooling."""

    episode_id: str
    task: str = ""
    device_type: tuple[str, ...] = ()
    hardware_string: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "device_type": list(self.device_type),
            "hardware_string": list(self.hardware_string),
            "extra": dict(self.extra),
        }


def validate_topics(topics: Sequence[str]) -> tuple[str, ...]:
    """Validate a configured topic list and return it as an immutable tuple."""
    resolved = tuple(str(topic) for topic in topics)
    missing = [topic for topic in NEDF3_RECORD_TOPICS if topic not in resolved]
    if missing:
        raise ValueError(f"NEDF3 topic list is missing canonical topics: {missing}")
    return resolved


def utc_timestamp_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


def create_vendor_writer(*, output_path: str, topics: Sequence[str] = NEDF3_RECORD_TOPICS) -> NEDF3Writer:
    """Construct the vendor writer lazily with an actionable dependency error."""
    validate_topics(topics)
    try:
        from nmx_nedf_api import NEDFFactory, NEDFWriterConfig  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "NEDF3 recording requires the optional nmx_nedf_api vendor package; "
            "install the hardware/recording profile on the target workstation."
        ) from exc
    return NEDFFactory.create_writer(NEDFWriterConfig(output_path=output_path))
