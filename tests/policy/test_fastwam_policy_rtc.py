"""Hardware-free RTC contracts for the FastWAM websocket policy adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from incubator.control_panel.runtime_config import compose_runtime_config
from policy.fastwam_policy import FastWAMPolicy
from scripts.analysis.fastwam_offline_utils import (
    relative_action_to_absolute,
    tcp_state16_to_umi_state,
)


def _robot_state() -> np.ndarray:
    state = np.zeros(44, dtype=np.float64)
    # Rizon SHM stores TCP quaternions in wxyz order.
    state[17] = 1.0
    state[39] = 1.0
    return state


def _absolute_action_chunk() -> np.ndarray:
    action = np.zeros((4, 16), dtype=np.float64)
    action[:, 0] = np.arange(4, dtype=np.float64)
    action[:, 8] = 10.0 + np.arange(4, dtype=np.float64)
    action[:, 6] = 1.0
    action[:, 14] = 1.0
    action[:, 7] = 0.03
    action[:, 15] = 0.04
    return action


def _robot_state_with_tcp_anchors() -> np.ndarray:
    state = _robot_state()
    left_quat_xyzw = Rotation.from_euler("xyz", [15.0, -10.0, 35.0], degrees=True).as_quat()
    right_quat_xyzw = Rotation.from_euler("xyz", [-20.0, 25.0, -15.0], degrees=True).as_quat()
    state[14:17] = [0.31, -0.14, 0.42]
    state[17:21] = [left_quat_xyzw[3], *left_quat_xyzw[:3]]
    state[21] = 0.03
    state[36:39] = [-0.27, 0.19, 0.51]
    state[39:43] = [right_quat_xyzw[3], *right_quat_xyzw[:3]]
    state[43] = 0.04
    return state


def _relative_umi_action() -> np.ndarray:
    action = np.zeros((2, 16), dtype=np.float64)
    action[:, 0:3] = [[0.04, -0.03, 0.02], [-0.01, 0.05, 0.03]]
    action[:, 3:7] = Rotation.from_euler(
        "xyz", [[5.0, 8.0, -12.0], [-9.0, 4.0, 18.0]], degrees=True
    ).as_quat()
    action[:, 7] = [0.035, 0.036]
    action[:, 8:11] = [[-0.02, 0.01, 0.05], [0.03, -0.04, 0.01]]
    action[:, 11:15] = Rotation.from_euler(
        "xyz", [[-7.0, 11.0, 6.0], [13.0, -5.0, -16.0]], degrees=True
    ).as_quat()
    action[:, 15] = [0.045, 0.044]
    return action


def _write_nonidentity_tcp_umi_calibration(tmp_path: Path) -> Path:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", [20.0, -15.0, 40.0], degrees=True
    ).as_matrix()
    transform[:3, 3] = [0.08, -0.05, 0.12]
    path = tmp_path / "tcp_to_umi.json"
    path.write_text(json.dumps(transform.tolist()), encoding="utf-8")
    return path


def test_rtc_payload_uses_p95_delay_and_inference_start_anchor() -> None:
    policy = FastWAMPolicy(enable_rtc=True, execution_fps=10.0)
    policy._server_metadata = {"action_horizon": 4, "action_dim": 16}
    policy._previous_absolute_action = _absolute_action_chunk()
    policy._latency_tracker.add(0.09)

    payload = policy._build_rtc_payload({"RizonRobot_2": _robot_state()}, action_index=0)

    assert payload["rtc_config"] == {
        "inference_delay": 1,
        "prefix_attention_horizon": 2,
        "max_guidance_weight": 10.0,
        "rtc_use_vjp": True,
        "action_index": 0,
    }
    np.testing.assert_allclose(payload["prefix_action"][:2], _absolute_action_chunk()[1:3])
    np.testing.assert_allclose(payload["prefix_action"][2:, [6, 14]], np.ones((2, 2)))


def test_rtc_overlap_diagnostics_measure_alignment_and_trend() -> None:
    previous = np.zeros((6, 16), dtype=np.float64)
    previous[:, 0] = np.arange(6, dtype=np.float64)
    previous[:, 8] = np.arange(6, dtype=np.float64)
    current = previous[2:].copy()

    diagnostics = FastWAMPolicy._compute_rtc_overlap_diagnostics(
        previous_action=previous,
        new_action=current,
        action_index=1,
        guidance_horizon=2,
    )

    assert diagnostics["rtc_overlap_available"] is True
    assert diagnostics["rtc_overlap_old_start_index"] == 2
    assert diagnostics["rtc_overlap_length"] == 4
    assert diagnostics["rtc_overlap_guided_length"] == 2
    assert diagnostics["rtc_overlap_left_xyz_mean_error_mm"] == pytest.approx(0.0)
    assert diagnostics["rtc_overlap_left_velocity_cosine_mean"] == pytest.approx(1.0)
    assert diagnostics["rtc_overlap_left_velocity_negative_fraction"] == pytest.approx(0.0)
    assert diagnostics["rtc_overlap_left_mask_edge_jump_mm"] == pytest.approx(1_000.0)


def test_rtc_overlap_diagnostics_exposes_reverse_trend() -> None:
    previous = np.zeros((5, 16), dtype=np.float64)
    previous[:, 0] = np.arange(5, dtype=np.float64)
    current = previous[1:].copy()
    current[:, 0] = current[::-1, 0]

    diagnostics = FastWAMPolicy._compute_rtc_overlap_diagnostics(
        previous_action=previous,
        new_action=current,
        action_index=0,
        guidance_horizon=4,
    )

    assert diagnostics["rtc_overlap_left_velocity_cosine_mean"] == pytest.approx(-1.0)
    assert diagnostics["rtc_overlap_left_velocity_negative_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("action_index", [2])
def test_rtc_payload_falls_back_when_delay_consumes_overlap(action_index: int) -> None:
    policy = FastWAMPolicy(enable_rtc=True, execution_fps=10.0)
    policy._server_metadata = {"action_horizon": 4, "action_dim": 16}
    policy._previous_absolute_action = _absolute_action_chunk()
    policy._latency_tracker.add(0.09)

    assert policy._build_rtc_payload({"RizonRobot_2": _robot_state()}, action_index) == {}


def test_rtc_payload_falls_back_when_latency_outlasts_chunk() -> None:
    policy = FastWAMPolicy(enable_rtc=True, execution_fps=10.0)
    policy._server_metadata = {"action_horizon": 4, "action_dim": 16}
    policy._previous_absolute_action = _absolute_action_chunk()
    policy._latency_tracker.add(0.4)

    assert policy._build_rtc_payload({"RizonRobot_2": _robot_state()}, 0) == {}


def test_rtc_payload_uses_dynamic_p95_delay() -> None:
    action = np.tile(_absolute_action_chunk(), (2, 1))
    policy = FastWAMPolicy(
        enable_rtc=True,
        execution_fps=8.0,
    )
    policy._server_metadata = {"action_horizon": 8, "action_dim": 16}
    policy._previous_absolute_action = action
    policy._latency_tracker.add(0.125)

    payload = policy._build_rtc_payload({"RizonRobot_2": _robot_state()}, 1)

    assert payload["rtc_config"]["inference_delay"] == 1
    assert payload["rtc_config"]["prefix_attention_horizon"] == 5
    assert payload["rtc_config"]["max_guidance_weight"] == 10.0
    diagnostics = policy.get_last_inference_diagnostics()
    assert diagnostics["guide_source_start"] == 2
    assert diagnostics["guide_source_end_exclusive"] == 8
    assert diagnostics["weighted_command_start_offset"] == 0
    assert diagnostics["weighted_command_end_exclusive"] == 5
    assert diagnostics["weighted_model_start"] == 0
def test_policy_preserves_server_action_time_order() -> None:
    policy = FastWAMPolicy(enable_rtc=False)
    server_action = np.arange(12, dtype=np.float64).reshape(3, 4)

    extracted = policy._extract_server_action({"data": {"action": server_action}})

    np.testing.assert_array_equal(extracted, server_action)


@pytest.mark.parametrize("relative_pose_frame", ("local_frame", "world_frame"))
def test_tcp_umi_calibration_keeps_live_and_offline_actions_identical(
    tmp_path: Path,
    relative_pose_frame: str,
) -> None:
    calibration_path = _write_nonidentity_tcp_umi_calibration(tmp_path)
    robot_state = _robot_state_with_tcp_anchors()
    tcp_state16 = np.concatenate([robot_state[14:22], robot_state[36:44]])
    relative_action = _relative_umi_action()
    policy = FastWAMPolicy(tcp_umi_json_path=str(calibration_path))
    policy.relative_pose_frame = relative_pose_frame
    policy.action_quaternion_order = "xyzw"

    live_umi_state = policy._extract_training_state({"RizonRobot_2": robot_state})
    offline_umi_state = tcp_state16_to_umi_state(
        tcp_state16,
        tcp_umi_json_path=calibration_path,
    )
    live_absolute = policy._convert_relative_action_to_absolute(
        relative_action,
        {"RizonRobot_2": robot_state},
        target_action_dim=16,
    )
    rtc_relative = policy._convert_absolute_action_to_relative(
        live_absolute,
        {"RizonRobot_2": robot_state},
    )
    offline_absolute = relative_action_to_absolute(
        relative_action,
        tcp_state16,
        relative_pose_frame=relative_pose_frame,
        action_quaternion_order="xyzw",
        gripper_max_width=policy.gripper_max_width,
        tcp_umi_json_path=calibration_path,
    )
    identity_absolute = relative_action_to_absolute(
        relative_action,
        tcp_state16,
        relative_pose_frame=relative_pose_frame,
        action_quaternion_order="xyzw",
        gripper_max_width=policy.gripper_max_width,
    )

    np.testing.assert_allclose(live_umi_state, offline_umi_state, atol=1e-10)
    np.testing.assert_allclose(live_absolute, offline_absolute, atol=1e-10)
    np.testing.assert_allclose(rtc_relative, relative_action, atol=1e-10)
    assert not np.allclose(live_absolute[:, :15], identity_absolute[:, :15])


def test_memory_policy_sends_oldest_to_latest_image_history() -> None:
    policy = FastWAMPolicy(
        model_variant="fastwam-memory",
        camera_devices=["Camera_Left", "Camera_Right"],
        robot_device="RizonRobot_2",
        condition_history_frames=4,
        obs_horizon=4,
    )
    observation = {
        "Camera_Left": [
            np.full((2, 2, 3), index, dtype=np.uint8) for index in range(4)
        ],
        "Camera_Right": [
            np.full((2, 2, 3), 10 + index, dtype=np.uint8)
            for index in range(4)
        ],
        "RizonRobot_2": [_robot_state() for _ in range(4)],
        "_history_frame_timestamps_ns": np.array([100, 200, 300, 400]),
    }

    payload = policy._build_infer_payload(observation)

    assert "images" not in payload
    assert payload["history_frame_timestamps_ns"] == [100, 200, 300, 400]
    assert [int(frame[0][0, 0, 0]) for frame in payload["image_history"]] == [
        0,
        1,
        2,
        3,
    ]
    assert [int(frame[1][0, 0, 0]) for frame in payload["image_history"]] == [
        10,
        11,
        12,
        13,
    ]


def test_memory_policy_adopts_checkpoint_history_stride() -> None:
    policy = FastWAMPolicy(
        model_variant="fastwam-memory",
        condition_history_frames=4,
        obs_horizon=4,
    )
    policy._server_metadata = {
        "condition_history_frames": 4,
        "condition_history_stride": 4,
        "action_infer_mode": "observed_frame_cache",
        "action_video_attention_mode": "first_frame",
    }

    policy._validate_server_contract()

    assert policy.condition_history_stride == 4
    assert policy.downsample == 4
    assert policy.min_infer_obs == 13


def test_memory_policy_rejects_server_history_stride_mismatch() -> None:
    policy = FastWAMPolicy(
        model_variant="fastwam-memory",
        condition_history_frames=4,
        obs_horizon=4,
        condition_history_stride=1,
    )
    policy._server_metadata = {
        "condition_history_frames": 4,
        "condition_history_stride": 4,
        "action_infer_mode": "observed_frame_cache",
        "action_video_attention_mode": "first_frame",
    }

    with pytest.raises(RuntimeError, match="stride mismatch"):
        policy._validate_server_contract()


def test_reconnect_revalidates_server_contract() -> None:
    class _Connection:
        def __init__(self) -> None:
            self.closed = False
            self.sent = False

        def close(self) -> None:
            self.closed = True

        def send(self, _payload: object) -> None:
            self.sent = True

    policy = FastWAMPolicy(
        model_variant="fastwam-memory",
        condition_history_frames=4,
        obs_horizon=4,
    )
    connection = _Connection()
    policy._wait_for_server = lambda: (
        connection,
        {
            "condition_history_frames": 1,
            "condition_history_stride": 1,
            "action_infer_mode": "observed_frame_cache",
            "action_video_attention_mode": "first_frame",
        },
    )

    with pytest.raises(RuntimeError, match="visual-history mismatch"):
        policy._send_request({"prompt": "test"})

    assert connection.closed is True
    assert connection.sent is False


@pytest.mark.config
def test_current_fastwam_rtc_alias_keeps_chronological_contract() -> None:
    sync = compose_runtime_config(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    )
    rtc = compose_runtime_config(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc"
    )

    assert len(sync["devices"]) == 3
    assert sync["policy"]["params"]["reverse_temporal_chunks"] is False
    assert sync["policy"]["params"]["enable_rtc"] is True
    assert sync["policy"]["params"]["model_variant"] == "fastwam"
    rtc_params = rtc["policy"]["params"]
    assert rtc["policy"]["fps"] == 1.5
    assert rtc_params["enable_rtc"] is True
    assert rtc_params["rtc_use_vjp"] is False
    assert rtc_params["rtc_max_guidance_weight"] == 15.0
    assert rtc_params["rtc_delay_margin_steps"] == 1
    assert rtc_params["rtc_guidance_full_steps_override"] is None
    for forbidden in (
        "rtc_inference_fps",
        "rtc_reverse_temporal_guidance",
        "rtc_inference_delay_steps",
        "rtc_guidance_horizon_steps",
        "rtc_guidance_schedule",
        "rtc_guidance_temporal_weight",
    ):
        assert forbidden not in rtc_params
