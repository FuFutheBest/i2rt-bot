from __future__ import annotations

import json
from importlib import metadata

import numpy as np
import pytest


pytest.importorskip("nmx_msg")

from incubator.control_panel.nedf3_io import (
    NEDF3ContractError,
    arm_state_message,
    arm_tcp_cmd_xyzw_message,
    arm_tcp_state_wxyz_message,
    arm_wrench_message,
    extra_message,
    extract_extra_payload,
    extract_gripper_width,
    extract_joint_state_positions,
    extract_pose_wxyz,
    hand_cmd_message,
    hand_state_message,
    rgbd_jpeg_message,
)


def test_installed_sdk_exposes_pinned_nedf3_apis() -> None:
    pytest.importorskip("dm3_core.nedf")
    from dm3_core.nedf import NEDFFactory, NEDFReaderBase, NEDFType, NEDFWriterBase

    assert metadata.version("dm3-core") == "0.2.18"
    assert metadata.version("nmx-msg") == "2.2.0"
    assert callable(NEDFFactory.get_reader)
    assert callable(NEDFFactory.get_writer)
    assert callable(NEDFReaderBase.iter_decoded_messages)
    assert callable(NEDFWriterBase.write_message)
    assert hasattr(NEDFType, "NEDF3")


def test_cp312_nedf3_writer_reader_round_trip_in_dotted_parent(tmp_path) -> None:
    pytest.importorskip("dm3_core.nedf")
    from dm3_core.nedf import (
        NEDFFactory,
        NEDFMetadata,
        NEDFReaderConfig,
        NEDFTopicsConfig,
        NEDFType,
        NEDFWriterConfig,
    )

    topic = "nmx/hal/arm/left/joint/state"
    episode_root = tmp_path / "root.with.multiple.dots" / "episode.001"
    episode_root.mkdir(parents=True)
    metadata_path = episode_root / "metadata.json"
    metadata = NEDFMetadata(
        hardware_config_name="cp312_probe",
        task_name="cp312_probe",
        task_description="cp312 NEDF3 round trip",
        collection_mode="inference_rollout",
        robot_vendor_model="Flexiv Rizon",
        end_effector_vendor_model="LKTech UMI3",
        camera_info={},
        data_format="NEDF3",
        mcap_file_name="data.mcap",
        mcap_file_type="CHUNK_FILE",
        main_camera="",
        robot_type="dual_arm",
        record_topic_list=[topic],
    )
    context = NEDFFactory.get_writer(
        NEDFType.NEDF3,
        NEDFWriterConfig(
            metadata_file_path=metadata_path,
            metadata=metadata,
            topics_config=NEDFTopicsConfig(),
            max_file_size_mb=10,
            mcap_chunk_size_kb=64,
        ),
    )
    writer = context.__enter__()
    timestamp_ns = 123_456_789
    writer.write_message(
        topic,
        arm_state_message(positions=[0.1, -0.2, 0.3]),
        timestamp_ns,
        timestamp_ns,
    )
    context.__exit__(None, None, None)

    reader = NEDFFactory.get_reader(
        NEDFReaderConfig(metadata_file_path=metadata_path)
    )
    messages = list(reader.iter_decoded_messages(topic))
    index_path = episode_root / "data" / "mcap_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert reader.check_valid()
    assert len(messages) == 1
    assert index and all(
        (index_path.parent / item["filename"]).is_file() for item in index
    )
    np.testing.assert_allclose(
        extract_joint_state_positions(messages[0].decoded_message),
        [0.1, -0.2, 0.3],
    )


def test_r1_arm_pose_messages_normalize_and_preserve_declared_orders() -> None:
    wxyz = arm_tcp_state_wxyz_message([1, 2, 3, 2, 0, 0, 0])
    xyzw = arm_tcp_cmd_xyzw_message([1, 2, 3, 0, 0, 0, 2])
    expected = np.asarray([1, 2, 3, 2, 0, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(extract_pose_wxyz(wxyz), expected)
    np.testing.assert_array_equal(extract_pose_wxyz(xyzw), expected)


@pytest.mark.parametrize(
    "values",
    (
        [1, 2, 3],
        [1, 2, 3, 0, 0, 0, 0],
        [1, 2, 3, float("nan"), 0, 0, 1],
    ),
)
def test_invalid_pose_fails_closed(values: list[float]) -> None:
    with pytest.raises(NEDF3ContractError):
        arm_tcp_state_wxyz_message(values)


def test_r1_joint_gripper_and_extra_messages_round_trip() -> None:
    state = hand_state_message(0.042)
    assert extract_gripper_width(state) == pytest.approx(0.042)
    assert hand_cmd_message(0.031).DESCRIPTOR.full_name == "nmx.msg.HandCmd"
    joints = arm_state_message(positions=[0.1, -0.2], velocities=[0.3, -0.4])
    np.testing.assert_allclose(extract_joint_state_positions(joints), [0.1, -0.2])
    assert [joint.vel for joint in joints.joints] == pytest.approx([0.3, -0.4])
    payload = {
        "stream_type": "robot_state",
        "joint_position_rad": [0.1, -0.2],
    }
    message = extra_message(payload)
    decoded = extract_extra_payload(message)
    assert decoded["control_panel"] == payload
    assert set(decoded["processed"]) == {
        "aligned_timestamp_ns",
        "alignment_mode",
        "downsample_kept",
        "aligned_pose",
        "aligned_hand_state",
        "aligned_raw_hand_state",
        "aligned_slam",
        "aligned_pose_tcp",
        "aligned_corrected_roi_marker_angle",
        "aligned_corrected_gripper_width",
        "aligned_arkit_diagnostics",
    }
    json.loads(bytes(message.metadata).decode("utf-8"))


def test_rgbd_and_wrench_payloads_match_nedf3_message_types() -> None:
    message = rgbd_jpeg_message(b"jpeg", 320, 240, 3)
    assert message.DESCRIPTOR.full_name == "nmx.msg.RGBD"
    assert message.rgb.data == b"jpeg"
    assert message.rgb.encoded_format == "jpeg"
    assert message.depth.rows == 0
    wrench = arm_wrench_message([1, 2, 3, 4, 5, 6])
    assert wrench.DESCRIPTOR.full_name == "nmx.msg.ArmEndForce"
    assert (wrench.force.x, wrench.force.y, wrench.force.z) == (1, 2, 3)
