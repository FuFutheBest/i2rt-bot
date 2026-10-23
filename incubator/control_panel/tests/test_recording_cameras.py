import pytest

from incubator.control_panel.recording_cameras import (
    camera_id_overrides,
    complete_camera_role_devices,
    normalize_camera_role_devices,
    resolve_recorded_camera_ids,
)
from incubator.control_panel.recording_contract import (
    LEROBOT_FEATURE_NAMES,
    LEROBOT_PARQUET_COLUMNS,
    LEROBOT_PARQUET_SCHEMA_METADATA,
    LEROBOT_VIDEO_FEATURES,
    NEDF3_DATA_FORMAT,
    NEDF3_EXTRA_KEYS,
    NEDF3_RECORD_TOPIC_LIST,
    camera_topic_for_role,
)


def test_default_roles_use_stable_nedf_camera_ids() -> None:
    roles = {
        "head": "WebCameraOpenPi_2",
        "left": "WebCameraOpenPi_0",
        "right": "WebCameraOpenPi_4",
    }
    assert camera_id_overrides(roles) == {
        "WebCameraOpenPi_2": "camera_2",
        "WebCameraOpenPi_0": "camera_1",
        "WebCameraOpenPi_4": "camera_0",
    }


def test_nedf3_camera_ids_follow_r1_contract() -> None:
    roles = {
        "head": "WebCameraOpenPi_2",
        "left": "WebCameraOpenPi_0",
        "right": "WebCameraOpenPi_4",
    }
    assert camera_id_overrides(roles, data_format=NEDF3_DATA_FORMAT) == {
        "WebCameraOpenPi_2": "camera_2",
        "WebCameraOpenPi_0": "camera_1",
        "WebCameraOpenPi_4": "camera_0",
    }
    assert camera_topic_for_role("head", NEDF3_DATA_FORMAT) == (
        "nmx/hal/camera/camera_2/rgbd"
    )
    assert camera_topic_for_role("left", NEDF3_DATA_FORMAT) == (
        "nmx/hal/camera/camera_1/rgbd"
    )
    assert camera_topic_for_role("right", NEDF3_DATA_FORMAT) == (
        "nmx/hal/camera/camera_0/rgbd"
    )


def test_nedf3_record_and_lerobot_contracts_are_complete() -> None:
    assert len(NEDF3_RECORD_TOPIC_LIST) == 27
    assert len(set(NEDF3_RECORD_TOPIC_LIST)) == 27
    assert len(NEDF3_EXTRA_KEYS) == 0
    assert len(LEROBOT_FEATURE_NAMES) == 23
    assert len(LEROBOT_PARQUET_COLUMNS) == 22
    assert LEROBOT_PARQUET_COLUMNS[-5:] == (
        "observation.extra.ego.raw_pose",
        "observation.extra.ego.hand_state",
        "observation.extra.ego.raw_hand_state",
        "observation.extra.ego.roi_marker_angle",
        "observation.extra.ego.aligned_timestamp_ns",
    )
    assert LEROBOT_PARQUET_SCHEMA_METADATA == {
        b"slam_anomaly_status": b"missing",
        b"width_anomaly_status": b"missing",
    }
    assert LEROBOT_VIDEO_FEATURES == {
        "head": "observation.images.image",
        "left": "observation.images.wrist_image_1",
        "right": "observation.images.wrist_image_2",
    }


def test_legacy_two_camera_policy_order_fills_left_then_right() -> None:
    assert complete_camera_role_devices(
        {},
        ["WebCameraOpenPi_0", "WebCameraOpenPi_4", "WebCameraOpenPi_0"],
    ) == {
        "head": None,
        "left": "WebCameraOpenPi_0",
        "right": "WebCameraOpenPi_4",
    }


def test_legacy_three_camera_policy_order_fills_head_left_right() -> None:
    assert complete_camera_role_devices(
        {},
        ["head-device", "left-device", "right-device"],
    ) == {
        "head": "head-device",
        "left": "left-device",
        "right": "right-device",
    }


def test_partial_explicit_mapping_does_not_guess_missing_roles() -> None:
    assert complete_camera_role_devices(
        {"left_camera": "left-device"},
        ["head-device", "left-device", "right-device"],
    ) == {
        "head": None,
        "left": "left-device",
        "right": None,
    }


def test_duplicate_physical_camera_roles_are_rejected() -> None:
    with pytest.raises(ValueError, match="different devices"):
        normalize_camera_role_devices(
            {"left": "WebCameraOpenPi_0", "right": "WebCameraOpenPi_0"}
        )


def test_non_object_camera_roles_are_rejected() -> None:
    with pytest.raises(ValueError, match="JSON object or null"):
        normalize_camera_role_devices(["left-device"])


def test_logical_recorded_ids_override_legacy_device_requests() -> None:
    camera_info = {
        "camera_0": "WebCameraOpenPi_2",
        "camera_1": "WebCameraOpenPi_0",
        "camera_2": "WebCameraOpenPi_4",
    }
    assert resolve_recorded_camera_ids(
        camera_info,
        {
            "head": "old-head",
            "left": "old-left",
            "right": "old-right",
        },
    ) == {
        "head": "camera_2",
        "left": "camera_1",
        "right": "camera_0",
    }


def test_nedf3_role_labels_resolve_independently_of_metadata_order() -> None:
    camera_info = {
        "camera_1": "left_wrist",
        "camera_2": "head",
        "camera_0": "right_wrist",
    }
    assert resolve_recorded_camera_ids(
        camera_info,
        {
            "head": "old-head",
            "left": "old-left",
            "right": "old-right",
        },
        data_format=NEDF3_DATA_FORMAT,
    ) == {
        "head": "camera_2",
        "left": "camera_1",
        "right": "camera_0",
    }


def test_legacy_recorded_ids_remain_resolvable() -> None:
    camera_info = {
        "WebCameraOpenPi_2": "head",
        "WebCameraOpenPi_0": "left",
        "WebCameraOpenPi_4": "right",
    }
    assert resolve_recorded_camera_ids(
        camera_info,
        {
            "head_camera": "WebCameraOpenPi_2",
            "left_camera": "WebCameraOpenPi_0",
            "right_camera": "WebCameraOpenPi_4",
        },
    ) == {
        "head": "WebCameraOpenPi_2",
        "left": "WebCameraOpenPi_0",
        "right": "WebCameraOpenPi_4",
    }
