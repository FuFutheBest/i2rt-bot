"""Canonical NEDF3 and LeRobot contracts for inference rollouts.

New recordings follow the R1 HAL contract from
``origin/chenyiming/R1_teleoperation``. The obsolete experimental UMI-NEDF3
layout is intentionally unsupported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


NEDF3_DATA_FORMAT = "NEDF3"

CAMERA_ROLES = ("head", "left", "right")

# Canonical layout: camera_0=right wrist, camera_1=left wrist, camera_2=head.
NEDF3_CAMERA_IDS = {
    "head": "camera_2",
    "left": "camera_1",
    "right": "camera_0",
}

NEDF3_CAMERA_ROLE_LABELS = {
    "head": "head",
    "left": "inhand_left",
    "right": "inhand_right",
}

NEDF3_CAMERA_TOPICS = {
    role: f"nmx/hal/camera/{camera_id}/rgbd"
    for role, camera_id in NEDF3_CAMERA_IDS.items()
}

NEDF3_ARM_KINDS = (
    "tcp/cmd",
    "tcp/state",
    "tcp/vel",
    "joint/state",
    "joint/vel",
    "eef_wrench",
    "extra",
)
NEDF3_HAND_KINDS = ("cmd", "state", "extra")
NEDF3_ARM_TOPICS = {
    side: {kind: f"nmx/hal/arm/{side}/{kind}" for kind in NEDF3_ARM_KINDS}
    for side in ("right", "left")
}
NEDF3_HAND_TOPICS = {
    side: {kind: f"nmx/hal/hand/{side}/{kind}" for kind in NEDF3_HAND_KINDS}
    for side in ("right", "left")
}
NEDF3_HEAD_TOPICS = {
    kind: f"nmx/hal/head/{kind}"
    for kind in ("cmd", "state", "joint/state", "extra")
}

NEDF3_RECORD_TOPIC_LIST = (
    NEDF3_CAMERA_TOPICS["right"],
    *(NEDF3_ARM_TOPICS["right"][kind] for kind in NEDF3_ARM_KINDS),
    *(NEDF3_HAND_TOPICS["right"][kind] for kind in NEDF3_HAND_KINDS),
    NEDF3_CAMERA_TOPICS["left"],
    *(NEDF3_ARM_TOPICS["left"][kind] for kind in NEDF3_ARM_KINDS),
    *(NEDF3_HAND_TOPICS["left"][kind] for kind in NEDF3_HAND_KINDS),
    NEDF3_CAMERA_TOPICS["head"],
    *(NEDF3_HEAD_TOPICS[kind] for kind in ("cmd", "state", "joint/state", "extra")),
)

NEDF3_EXTRA_KEYS: tuple[str, ...] = ()

LEROBOT_VIDEO_FEATURES = {
    "head": "observation.images.image",
    "left": "observation.images.wrist_image_1",
    "right": "observation.images.wrist_image_2",
}

LEROBOT_INFO_FIELDS = (
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
)

LEROBOT_FPS = 30
LEROBOT_TARGET_FPS = 10.0
LEROBOT_DEVICE_TYPE = "rollout"

LEROBOT_DUAL_ARM_FEATURE_NAMES = tuple(
    f"{side}_{suffix}"
    for side in ("left", "right")
    for suffix in (
        "x",
        "y",
        "z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "gripper",
    )
)

LEROBOT_EGO_FEATURE_NAMES = (
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_quat_w",
    "ego_quat_x",
    "ego_quat_y",
    "ego_quat_z",
)

LEROBOT_FEATURE_NAMES = (
    LEROBOT_DUAL_ARM_FEATURE_NAMES + LEROBOT_EGO_FEATURE_NAMES
)

LEROBOT_EXTRA_FIELDS = (
    "raw_pose",
    "hand_state",
    "raw_hand_state",
    "roi_marker_angle",
    "aligned_timestamp_ns",
)

LEROBOT_CORE_PARQUET_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)

LEROBOT_PARQUET_COLUMNS = LEROBOT_CORE_PARQUET_COLUMNS + tuple(
    f"observation.extra.{side}.{field}"
    for side in ("left", "right", "ego")
    for field in LEROBOT_EXTRA_FIELDS
)

LEROBOT_PARQUET_SCHEMA_METADATA = {
    b"slam_anomaly_status": b"missing",
    b"width_anomaly_status": b"missing",
}


def normalize_data_format(value: Any, *, default: str = NEDF3_DATA_FORMAT) -> str:
    """Validate that an SDK enum or metadata value denotes ``NEDF3``."""
    if value is None:
        return default
    normalized = str(value).strip().upper()
    if normalized.endswith(".NEDF3"):
        normalized = NEDF3_DATA_FORMAT
    if normalized != NEDF3_DATA_FORMAT:
        raise ValueError(f"unsupported NEDF data format (NEDF3 required): {value!r}")
    return normalized


def camera_ids_for_format(data_format: Any) -> Mapping[str, str]:
    normalize_data_format(data_format)
    return NEDF3_CAMERA_IDS


def camera_topic_for_role(role: str, data_format: Any) -> str:
    if role not in CAMERA_ROLES:
        raise ValueError(f"unknown camera role: {role!r}")
    camera_id = camera_ids_for_format(data_format)[role]
    return f"nmx/hal/camera/{camera_id}/rgbd"


__all__ = [
    "CAMERA_ROLES",
    "LEROBOT_DEVICE_TYPE",
    "LEROBOT_DUAL_ARM_FEATURE_NAMES",
    "LEROBOT_EGO_FEATURE_NAMES",
    "LEROBOT_EXTRA_FIELDS",
    "LEROBOT_FEATURE_NAMES",
    "LEROBOT_FPS",
    "LEROBOT_INFO_FIELDS",
    "LEROBOT_CORE_PARQUET_COLUMNS",
    "LEROBOT_PARQUET_COLUMNS",
    "LEROBOT_PARQUET_SCHEMA_METADATA",
    "LEROBOT_TARGET_FPS",
    "LEROBOT_VIDEO_FEATURES",
    "NEDF3_CAMERA_IDS",
    "NEDF3_CAMERA_ROLE_LABELS",
    "NEDF3_CAMERA_TOPICS",
    "NEDF3_DATA_FORMAT",
    "NEDF3_ARM_KINDS",
    "NEDF3_ARM_TOPICS",
    "NEDF3_HAND_KINDS",
    "NEDF3_HAND_TOPICS",
    "NEDF3_HEAD_TOPICS",
    "NEDF3_EXTRA_KEYS",
    "NEDF3_RECORD_TOPIC_LIST",
    "camera_ids_for_format",
    "camera_topic_for_role",
    "normalize_data_format",
]
