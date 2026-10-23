"""Canonical NEDF3 contract for I2RT dual-arm teleoperation.

The historical ``/joint`` topic spelling is part of the dataset contract. Its
payload is an ``ArmState`` containing six ordered ``JointState`` entries. The
separate ``/hand_state`` topic contains one gripper ``JointState``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

NEDF3_I2RT_LEGACY_BGR_JPEG_VERSION = "i2rt-nedf3-v1"
NEDF3_I2RT_CONTRACT_VERSION = "i2rt-nedf3-v2"
NEDF3_METADATA_TOPIC = "nmx/nedf/metadata"

NEDF3_CAMERA_TOPICS: Mapping[str, str] = {
    "right": "nmx/hal/camera/camera_0/rgbd",
    "left": "nmx/hal/camera/camera_1/rgbd",
    "head": "nmx/hal/camera/camera_2/rgbd",
}
NEDF3_I2RT_JOINT_TOPICS: Mapping[str, Mapping[str, str]] = {
    "leader": {
        "right": "nmx/hal/i2rt/leader/right/joint",
        "left": "nmx/hal/i2rt/leader/left/joint",
    },
    "follower": {
        "right": "nmx/hal/i2rt/follower/right/joint",
        "left": "nmx/hal/i2rt/follower/left/joint",
    },
}
NEDF3_I2RT_HAND_TOPICS: Mapping[str, Mapping[str, str]] = {
    "leader": {
        "right": "nmx/hal/i2rt/leader/right/hand_state",
        "left": "nmx/hal/i2rt/leader/left/hand_state",
    },
    "follower": {
        "right": "nmx/hal/i2rt/follower/right/hand_state",
        "left": "nmx/hal/i2rt/follower/left/hand_state",
    },
}

NEDF3_RECORD_TOPICS: tuple[str, ...] = (
    NEDF3_METADATA_TOPIC,
    NEDF3_CAMERA_TOPICS["right"],
    NEDF3_I2RT_JOINT_TOPICS["leader"]["right"],
    NEDF3_I2RT_HAND_TOPICS["leader"]["right"],
    NEDF3_I2RT_JOINT_TOPICS["follower"]["right"],
    NEDF3_I2RT_HAND_TOPICS["follower"]["right"],
    NEDF3_CAMERA_TOPICS["left"],
    NEDF3_I2RT_JOINT_TOPICS["leader"]["left"],
    NEDF3_I2RT_HAND_TOPICS["leader"]["left"],
    NEDF3_I2RT_JOINT_TOPICS["follower"]["left"],
    NEDF3_I2RT_HAND_TOPICS["follower"]["left"],
    NEDF3_CAMERA_TOPICS["head"],
)
NEDF3_RECORD_TOPIC_LIST = NEDF3_RECORD_TOPICS
HAL_CAMERA_TOPICS = tuple(NEDF3_CAMERA_TOPICS.values())


class NEDF3Writer(Protocol):
    """Writer surface used by the I2RT saver and its dependency-free tests."""

    def write_message(self, topic: str, message: Any, timestamp_ns: int) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class NEDF3EpisodeMetadata:
    """Metadata supplied when an I2RT episode starts."""

    episode_id: str
    task: str = "data_collection"
    task_description: str = ""
    device_type: tuple[str, ...] = ()
    hardware_string: tuple[str, ...] = ()
    main_camera: str = "camera_2"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "task_description": self.task_description,
            "device_type": list(self.device_type),
            "hardware_string": list(self.hardware_string),
            "main_camera": self.main_camera,
            "contract_version": NEDF3_I2RT_CONTRACT_VERSION,
            "extra": dict(self.extra),
        }


def validate_topics(topics: Sequence[str]) -> tuple[str, ...]:
    """Require the exact canonical topic set while preserving declared order."""
    resolved = tuple(str(topic) for topic in topics)
    duplicates = sorted({topic for topic in resolved if resolved.count(topic) > 1})
    if duplicates:
        raise ValueError(f"NEDF3 topic list contains duplicates: {duplicates}")
    missing = [topic for topic in NEDF3_RECORD_TOPICS if topic not in resolved]
    unexpected = [topic for topic in resolved if topic not in NEDF3_RECORD_TOPICS]
    if missing or unexpected:
        raise ValueError(
            "NEDF3 I2RT topic list differs from the canonical contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return resolved


def utc_timestamp_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


class _ManagedVendorWriter:
    """Adapt the dm3-core context manager to the saver writer protocol."""

    def __init__(self, context: Any) -> None:
        self._context = context
        self._writer = context.__enter__()
        self._closed = False

    def write_message(self, topic: str, message: Any, timestamp_ns: int) -> None:
        if self._closed:
            raise RuntimeError("NEDF3 writer is closed")
        self._writer.write_message(topic, message, timestamp_ns, timestamp_ns)

    def close(self) -> None:
        if not self._closed:
            self._context.__exit__(None, None, None)
            self._closed = True


def create_vendor_writer(
    *,
    output_path: str | Path,
    metadata: NEDF3EpisodeMetadata,
    topics: Sequence[str] = NEDF3_RECORD_TOPICS,
) -> NEDF3Writer:
    """Create a dm3-core NEDF3 writer for one I2RT episode directory."""
    resolved_topics = validate_topics(topics)
    try:
        from dm3_core.nedf import (
            NEDFFactory,
            NEDFMetadata,
            NEDFTopicsConfig,
            NEDFType,
            NEDFWriterConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "I2RT NEDF3 recording requires dm3-core and nmx-msg in the active environment"
        ) from exc

    episode_dir = Path(output_path)
    episode_dir.mkdir(parents=True, exist_ok=True)
    extra = metadata.as_dict()
    extra.pop("extra")
    extra.update(dict(metadata.extra))
    vendor_metadata = NEDFMetadata(
        hardware_config_name="i2rt_dual_bilateral",
        task_name=metadata.task,
        task_description=metadata.task_description,
        collection_mode="teleoperation",
        robot_vendor_model="I2RT_YAM",
        end_effector_vendor_model="I2RT_gripper",
        camera_info={
            "camera_0": "right_wrist",
            "camera_1": "left_wrist",
            "camera_2": "head",
        },
        data_format="NEDF3",
        mcap_file_name="data.mcap",
        mcap_file_type="CHUNK_FILE",
        main_camera=metadata.main_camera,
        robot_type="dual_arm",
        end_effector="gripper",
        extra_keys=[],
        extra=extra,
        record_topic_list=list(resolved_topics),
    )
    context = NEDFFactory.get_writer(
        NEDFType.NEDF3,
        NEDFWriterConfig(
            metadata_file_path=episode_dir / "metadata.json",
            metadata=vendor_metadata,
            topics_config=NEDFTopicsConfig(),
            max_file_size_mb=100,
            mcap_chunk_size_kb=256,
        ),
    )
    return _ManagedVendorWriter(context)


__all__ = [
    "HAL_CAMERA_TOPICS",
    "NEDF3_CAMERA_TOPICS",
    "NEDF3_I2RT_CONTRACT_VERSION",
    "NEDF3_I2RT_LEGACY_BGR_JPEG_VERSION",
    "NEDF3_I2RT_HAND_TOPICS",
    "NEDF3_I2RT_JOINT_TOPICS",
    "NEDF3_METADATA_TOPIC",
    "NEDF3_RECORD_TOPIC_LIST",
    "NEDF3_RECORD_TOPICS",
    "NEDF3EpisodeMetadata",
    "NEDF3Writer",
    "create_vendor_writer",
    "utc_timestamp_ns",
    "validate_topics",
]
