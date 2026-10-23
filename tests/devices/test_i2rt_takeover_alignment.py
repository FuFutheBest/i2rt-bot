"""Hardware-free coverage for I2RT's policy-to-human handover protocol."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from devices.robots.i2rt import I2RTRobot
from utils.i2rt_takeover import (
    publish_takeover_hold_request,
    read_takeover_hold_ack,
)


def _make_leader(*, alignment_enabled: bool) -> I2RTRobot:
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 0
    robot.running = False
    robot._i2rt_stopped = True
    robot.data_shape = (14,)
    robot.data_dtype = np.float64
    robot.command_fps = 100.0
    robot.dual_control_mode = True
    robot.is_leader_role = True
    robot.is_follower_role = False
    robot.takeover_alignment_enabled = alignment_enabled
    robot.takeover_alignment_duration_s = 0.0
    robot.takeover_alignment_settle_timeout_s = 0.0
    robot.takeover_hold_ack_timeout_s = 1.0
    robot._control_mode = robot._CONTROL_MODE_AUTONOMOUS
    robot._last_button_pressed = False
    robot._last_button_left = 0.0
    robot._last_button_right = 0.0
    robot._button_down_latched = False
    robot._button_release_samples = 0
    robot._button_release_debounce_samples = 3
    robot._dagger_mode_transition_lock = threading.Lock()
    robot._left_client = object()
    robot._right_client = object()
    robot.shared_memory = None
    robot._apply_bilateral_gains_for_mode = lambda: None
    robot.flags = []
    robot.commands = []
    robot._write_leader_control_flag = lambda value, _timestamp_ns: robot.flags.append(
        float(value)
    )
    robot._command_joint_pos = lambda _client, joints, source: robot.commands.append(
        (source, np.asarray(joints, dtype=np.float64).copy())
    )
    return robot


def _make_follower(
    *,
    request_path: Path,
    acknowledgement_path: Path,
    live_hold_pose: np.ndarray,
) -> I2RTRobot:
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 1
    robot.running = False
    robot._i2rt_stopped = True
    robot.data_shape = (14,)
    robot.data_dtype = np.float64
    robot.dual_control_mode = True
    robot.is_leader_role = False
    robot.is_follower_role = True
    robot._control_mode = robot._CONTROL_MODE_AUTONOMOUS
    robot._takeover_hold_joint_pos = None
    robot._takeover_hold_command_issued_at_monotonic = None
    robot._takeover_hold_feedback_samples = 0
    robot._takeover_hold_last_feedback_monotonic = None
    robot._takeover_hold_feedback_request_id = None
    robot._takeover_hold_faulted = False
    robot._last_takeover_hold_ack_request_id = None
    robot.takeover_hold_feedback_tolerance_rad = 0.01
    robot.takeover_hold_feedback_min_settle_s = 0.0
    robot.takeover_hold_feedback_stable_samples = 2
    robot._takeover_hold_request_path = str(request_path)
    robot._takeover_hold_ack_path = str(acknowledgement_path)
    robot._left_client = object()
    robot._right_client = object()
    robot._apply_bilateral_gains_for_mode = lambda: None
    robot.commands = []
    robot._command_joint_pos = lambda _client, joints, source: robot.commands.append(
        (source, np.asarray(joints, dtype=np.float64).copy())
    )
    robot._read_follower_joint_state_for_takeover_hold = lambda: np.asarray(
        live_hold_pose, dtype=np.float64
    ).copy()
    return robot


def test_alignment_configuration_requires_leader_dual_control() -> None:
    with pytest.raises(ValueError, match="leader"):
        I2RTRobot._validate_takeover_alignment_mode(
            enabled=True,
            is_leader_role=False,
            dual_control_mode=True,
        )
    with pytest.raises(ValueError, match="dual-source"):
        I2RTRobot._validate_takeover_alignment_mode(
            enabled=True,
            is_leader_role=True,
            dual_control_mode=False,
        )

    I2RTRobot._validate_takeover_alignment_mode(
        enabled=True,
        is_leader_role=True,
        dual_control_mode=True,
    )


def test_alignment_disabled_preserves_immediate_binary_handover() -> None:
    robot = _make_leader(alignment_enabled=False)
    current = np.arange(14, dtype=np.float64) / 10.0

    def leader_get_info(_client: object, source: str) -> tuple[np.ndarray, np.ndarray]:
        if "left" in source:
            return current[:7].copy(), np.array([1.0])
        return current[7:].copy(), np.array([0.0])

    robot._leader_get_info = leader_get_info
    robot._align_leader_for_takeover = lambda _state: (_ for _ in ()).throw(
        AssertionError("disabled alignment must not run")
    )

    state = I2RTRobot.get_robot_state.__wrapped__(robot)

    assert robot._control_mode == robot._CONTROL_MODE_TELEOP
    assert robot.flags == [1.0]
    np.testing.assert_allclose(state, current)


def test_alignment_publishes_half_flag_before_binary_handover() -> None:
    robot = _make_leader(alignment_enabled=True)
    current = np.zeros(14, dtype=np.float64)
    current[[6, 13]] = 0.25
    follower = np.ones(14, dtype=np.float64)
    follower[[6, 13]] = 0.75
    aligned = current.copy()
    aligned[:6] = follower[:6]
    aligned[7:13] = follower[7:13]

    def leader_get_info(_client: object, source: str) -> tuple[np.ndarray, np.ndarray]:
        state = aligned if "takeover_alignment" in source else current
        if "left" in source:
            return state[:7].copy(), np.array([1.0])
        return state[7:].copy(), np.array([0.0])

    robot._leader_get_info = leader_get_info
    robot._publish_takeover_hold_request = lambda: "a" * 32
    robot._wait_for_follower_takeover_hold = lambda *, request_id: (
        follower.copy() if request_id == "a" * 32 else None
    )

    with patch("devices.robots.i2rt.time.sleep", lambda _seconds: None):
        state = I2RTRobot.get_robot_state.__wrapped__(robot)

    assert robot._control_mode == robot._CONTROL_MODE_TELEOP
    assert robot.flags
    assert all(flag == 0.5 for flag in robot.flags[:-1])
    assert robot.flags[-1] == 1.0
    np.testing.assert_allclose(state, aligned)

    left_alignment_commands = [
        command
        for source, command in robot.commands
        if source == "command_joint_pos_left_takeover_alignment"
    ]
    right_alignment_commands = [
        command
        for source, command in robot.commands
        if source == "command_joint_pos_right_takeover_alignment"
    ]
    np.testing.assert_array_equal(left_alignment_commands[-1], follower[:6])
    np.testing.assert_array_equal(right_alignment_commands[-1], follower[7:13])


def test_alignment_failure_keeps_policy_control() -> None:
    robot = _make_leader(alignment_enabled=True)
    robot._publish_takeover_hold_request = lambda: "a" * 32
    robot._wait_for_follower_takeover_hold = lambda *, request_id: None

    with patch("devices.robots.i2rt.time.sleep", lambda _seconds: None):
        state = robot._handle_mode_transition(True, current_leader_state=np.zeros(14))

    assert state is None
    assert robot._control_mode == robot._CONTROL_MODE_AUTONOMOUS
    assert robot.flags[-1] == 0.0


def test_concurrent_state_reader_cannot_overlap_takeover_transition() -> None:
    robot = _make_leader(alignment_enabled=True)
    current = np.zeros(14, dtype=np.float64)
    alignment_started = threading.Event()
    finish_alignment = threading.Event()
    second_call_finished = threading.Event()
    alignment_calls = 0

    def leader_get_info(_client: object, _source: str) -> tuple[np.ndarray, np.ndarray]:
        return current[:7].copy(), np.array([1.0])

    def align(_state: np.ndarray) -> np.ndarray:
        nonlocal alignment_calls
        alignment_calls += 1
        alignment_started.set()
        assert finish_alignment.wait(timeout=1.0)
        return current.copy()

    robot._leader_get_info = leader_get_info
    robot._align_leader_for_takeover = align

    first = threading.Thread(
        target=I2RTRobot.get_robot_state.__wrapped__, args=(robot,)
    )
    second = threading.Thread(
        target=lambda: (
            I2RTRobot.get_robot_state.__wrapped__(robot),
            second_call_finished.set(),
        )
    )
    first.start()
    assert alignment_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.05)

    assert not second_call_finished.is_set()
    finish_alignment.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert alignment_calls == 1


def test_half_flag_holds_follower_acknowledges_then_ignores_policy_command(
    tmp_path: Path,
) -> None:
    hold_pose = np.arange(14, dtype=np.float64)
    request_path = tmp_path / "takeover-request.json"
    acknowledgement_path = tmp_path / "takeover-ack.json"
    robot = _make_follower(
        request_path=request_path,
        acknowledgement_path=acknowledgement_path,
        live_hold_pose=hold_pose,
    )
    request_id = publish_takeover_hold_request(
        request_path,
        source_device_id=0,
        target_device_id=1,
        request_id="a" * 32,
    )

    robot._set_follower_control_mode_from_flag(0.5)
    issued_at = robot._takeover_hold_command_issued_at_monotonic
    assert issued_at is not None
    robot._observe_follower_takeover_hold_feedback(
        hold_pose + 0.1,
        measured_at_monotonic=issued_at + 0.01,
        measured_at_ns=1,
    )
    assert (
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=request_id,
            expected_source_device_id=1,
            expected_target_device_id=0,
        )
        is None
    )
    robot._observe_follower_takeover_hold_feedback(
        hold_pose,
        measured_at_monotonic=issued_at + 0.02,
        measured_at_ns=2,
    )
    assert (
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=request_id,
            expected_source_device_id=1,
            expected_target_device_id=0,
        )
        is None
    )
    robot._observe_follower_takeover_hold_feedback(
        hold_pose,
        measured_at_monotonic=issued_at + 0.03,
        measured_at_ns=3,
    )
    acknowledgement = read_takeover_hold_ack(
        acknowledgement_path,
        request_id=request_id,
        expected_source_device_id=1,
        expected_target_device_id=0,
    )
    robot.commands.clear()
    policy_command = np.full(14, -5.0, dtype=np.float64)
    I2RTRobot.execute_control_command.__wrapped__(
        robot, {"I2RTRobot_1_control": policy_command}
    )

    assert robot._control_mode == robot._CONTROL_MODE_ALIGNMENT
    np.testing.assert_array_equal(acknowledgement, hold_pose)
    np.testing.assert_array_equal(robot.commands[0][1], np.arange(7))
    np.testing.assert_array_equal(robot.commands[1][1], np.arange(7, 14))

    next_request_id = publish_takeover_hold_request(
        request_path,
        source_device_id=0,
        target_device_id=1,
        request_id="b" * 32,
    )
    robot._observe_follower_takeover_hold_feedback(
        hold_pose,
        measured_at_monotonic=issued_at + 0.04,
        measured_at_ns=4,
    )
    robot._observe_follower_takeover_hold_feedback(
        hold_pose,
        measured_at_monotonic=issued_at + 0.05,
        measured_at_ns=5,
    )
    assert (
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=next_request_id,
            expected_source_device_id=1,
            expected_target_device_id=0,
        )
        is None
    )
    robot._observe_follower_takeover_hold_feedback(
        hold_pose,
        measured_at_monotonic=issued_at + 0.06,
        measured_at_ns=6,
    )
    np.testing.assert_array_equal(
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=next_request_id,
            expected_source_device_id=1,
            expected_target_device_id=0,
        ),
        hold_pose,
    )


def test_follower_feedback_failure_enters_hold_without_executing_policy(
    tmp_path: Path,
) -> None:
    robot = _make_follower(
        request_path=tmp_path / "takeover-request.json",
        acknowledgement_path=tmp_path / "takeover-ack.json",
        live_hold_pose=np.arange(14, dtype=np.float64),
    )
    robot._read_follower_joint_state_for_takeover_hold = lambda: None

    robot._set_follower_control_mode_from_flag(0.5)
    I2RTRobot.execute_control_command.__wrapped__(
        robot, {"I2RTRobot_1_control": np.full(14, -5.0, dtype=np.float64)}
    )

    assert robot._control_mode == robot._CONTROL_MODE_ALIGNMENT
    assert robot._takeover_hold_joint_pos is None
    assert robot.commands == []


def test_partial_hold_command_failure_stays_fail_closed_without_ack(
    tmp_path: Path,
) -> None:
    hold_pose = np.arange(14, dtype=np.float64)
    request_path = tmp_path / "takeover-request.json"
    acknowledgement_path = tmp_path / "takeover-ack.json"
    robot = _make_follower(
        request_path=request_path,
        acknowledgement_path=acknowledgement_path,
        live_hold_pose=hold_pose,
    )
    request_id = publish_takeover_hold_request(
        request_path,
        source_device_id=0,
        target_device_id=1,
        request_id="c" * 32,
    )
    commands: list[str] = []

    def command(_client: object, _joints: np.ndarray, source: str) -> None:
        commands.append(source)
        if "right" in source:
            raise RuntimeError("right hold RPC failed")

    robot._command_joint_pos = command

    robot._set_follower_control_mode_from_flag(0.5)
    I2RTRobot.execute_control_command.__wrapped__(
        robot, {"I2RTRobot_1_control": np.full(14, -5.0, dtype=np.float64)}
    )
    robot._observe_follower_takeover_hold_feedback(
        hold_pose,
        measured_at_monotonic=time.monotonic() + 1.0,
        measured_at_ns=1,
    )

    assert commands == [
        "command_joint_pos_left_takeover_alignment_hold",
        "command_joint_pos_right_takeover_alignment_hold",
    ]
    assert robot._control_mode == robot._CONTROL_MODE_ALIGNMENT
    assert robot._takeover_hold_faulted is True
    assert robot._takeover_hold_joint_pos is None
    assert (
        read_takeover_hold_ack(
            acknowledgement_path,
            request_id=request_id,
            expected_source_device_id=1,
            expected_target_device_id=0,
        )
        is None
    )


def test_alignment_abort_is_a_policy_reentry() -> None:
    assert I2RTRobot._enters_autonomous_control_mode(0.5, 0.0)
    assert I2RTRobot._enters_autonomous_control_mode(1.0, 0.0)
    assert not I2RTRobot._enters_autonomous_control_mode(0.0, 0.5)


def test_alignment_waits_full_settle_period_even_at_target() -> None:
    robot = _make_leader(alignment_enabled=True)
    robot.takeover_alignment_settle_timeout_s = 0.03
    target = np.ones(14, dtype=np.float64)
    robot._publish_takeover_hold_request = lambda: "a" * 32
    robot._wait_for_follower_takeover_hold = lambda *, request_id: target.copy()
    clock = [0.0]
    reads = []

    def measured_state() -> np.ndarray:
        reads.append(clock[0])
        return target.copy()

    robot._read_leader_joint_state_for_takeover = measured_state
    with patch("devices.robots.i2rt.time.monotonic", lambda: clock[0]), patch(
        "devices.robots.i2rt.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    ):
        state = robot._handle_mode_transition(True, current_leader_state=target)

    np.testing.assert_allclose(state, target)
    assert reads[-1] - reads[0] >= robot.takeover_alignment_settle_timeout_s
    assert robot._control_mode == robot._CONTROL_MODE_TELEOP
    assert all(flag == 0.5 for flag in robot.flags)


def test_leader_does_not_command_alignment_before_stable_follower_feedback(
    tmp_path: Path,
) -> None:
    robot = _make_leader(alignment_enabled=True)
    current = np.zeros(14, dtype=np.float64)
    follower = np.ones(14, dtype=np.float64)
    request_path = tmp_path / "takeover-request.json"
    acknowledgement_path = tmp_path / "takeover-ack.json"
    alignment_flag_written = threading.Event()
    result: list[np.ndarray | None] = []
    robot._takeover_hold_request_path = str(request_path)
    robot._takeover_hold_ack_path = str(acknowledgement_path)
    robot.takeover_hold_ack_timeout_s = 1.0

    def write_flag(value: float, _timestamp_ns: int) -> None:
        robot.flags.append(float(value))
        if value == 0.5:
            alignment_flag_written.set()

    robot._write_leader_control_flag = write_flag
    robot._read_leader_joint_state_for_takeover = lambda: follower.copy()
    follower_robot = _make_follower(
        request_path=request_path,
        acknowledgement_path=acknowledgement_path,
        live_hold_pose=follower,
    )
    worker = threading.Thread(
        target=lambda: result.append(robot._align_leader_for_takeover(current))
    )

    worker.start()
    assert alignment_flag_written.wait(timeout=1.0)
    follower_robot._set_follower_control_mode_from_flag(0.5)
    issued_at = follower_robot._takeover_hold_command_issued_at_monotonic
    assert issued_at is not None

    assert robot.flags == [0.5]
    assert robot.commands == []

    follower_robot._observe_follower_takeover_hold_feedback(
        follower + 0.1,
        measured_at_monotonic=issued_at + 0.01,
        measured_at_ns=1,
    )
    assert robot.commands == []
    follower_robot._observe_follower_takeover_hold_feedback(
        follower,
        measured_at_monotonic=issued_at + 0.02,
        measured_at_ns=2,
    )
    assert robot.commands == []
    follower_robot._observe_follower_takeover_hold_feedback(
        follower,
        measured_at_monotonic=issued_at + 0.03,
        measured_at_ns=3,
    )
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    np.testing.assert_allclose(result[0], follower)
    assert robot.commands
