"""Hardware-free coverage for YAM's mandatory joint-zero disconnect gate."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from utils.yam_disconnect_safety import (
    close_yam_robot_without_can_race,
    confirm_measured_joint_zero,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Robot:
    def __init__(self, feedback: list[object]) -> None:
        self.feedback = deque(feedback)
        self.last_feedback = feedback[-1]
        self.commands: list[np.ndarray] = []

    def command_joint_pos(self, value: np.ndarray) -> None:
        self.commands.append(np.asarray(value, dtype=float).copy())

    def get_joint_pos(self) -> np.ndarray:
        value = self.feedback.popleft() if self.feedback else self.last_feedback
        if isinstance(value, BaseException):
            raise value
        return np.asarray(value, dtype=float)


def _confirm(robot: _Robot, clock: _Clock, **overrides):
    options = {
        "arm_joint_count": 5,
        "tolerance_rad": 0.01,
        "timeout_s": 0.5,
        "consecutive_samples": 3,
        "poll_interval_s": 0.05,
        "monotonic": clock.monotonic,
        "sleep": clock.sleep,
    }
    options.update(overrides)
    return confirm_measured_joint_zero(robot, np.zeros(6), **options)


def test_disconnect_gate_requires_consecutive_measured_zero_samples() -> None:
    clock = _Clock()
    robot = _Robot(
        [
            [0.009] * 5 + [-1.25],
            [0.008] * 5 + [-1.25],
            [0.02] * 5 + [-1.25],
            [0.009] * 5 + [-1.25],
            [0.008] * 5 + [-1.25],
            [0.01] * 5 + [-1.25],
        ]
    )

    result = _confirm(robot, clock)

    assert result.confirmed is True
    assert result.consecutive_samples == 3
    assert result.max_error_rad == 0.01
    assert len(robot.commands) == 6
    assert all(np.array_equal(command, np.zeros(6)) for command in robot.commands)


def test_disconnect_gate_fails_closed_when_feedback_never_reaches_zero() -> None:
    clock = _Clock()
    robot = _Robot([[0.2] * 5 + [-1.25]])

    result = _confirm(robot, clock, timeout_s=0.16)

    assert result.confirmed is False
    assert result.max_error_rad == 0.2
    assert result.feedback is not None
    assert len(robot.commands) >= 3


def test_disconnect_gate_fails_closed_on_missing_or_invalid_feedback() -> None:
    clock = _Clock()
    robot = _Robot(
        [
            RuntimeError("CAN read failed"),
            [0.0, 0.0],
            [0.0, 0.0, 0.0, np.nan, 0.0, -1.25],
        ]
    )

    result = _confirm(robot, clock, timeout_s=0.16)

    assert result.confirmed is False
    assert result.feedback is None
    assert result.error is not None


def test_disconnect_gate_can_succeed_on_a_later_retry() -> None:
    first_clock = _Clock()
    robot = _Robot([[0.15] * 5 + [-1.25]])
    assert _confirm(robot, first_clock, timeout_s=0.11).confirmed is False

    robot.feedback.extend(
        [[0.009] * 5 + [-1.25], [0.008] * 5 + [-1.25], [0.0] * 5 + [-1.25]]
    )
    robot.last_feedback = [0.0] * 5 + [-1.25]
    retry_clock = _Clock()

    assert _confirm(robot, retry_clock).confirmed is True


def test_disconnect_gate_ignores_sixth_joint() -> None:
    clock = _Clock()
    robot = _Robot([[0.01, -0.01, 0.0, 0.009, -0.009, -1.259]])

    result = _confirm(
        robot,
        clock,
        consecutive_samples=1,
        timeout_s=0.1,
    )

    assert result.confirmed is True
    assert result.feedback is not None
    assert result.feedback.tolist() == [0.01, -0.01, 0.0, 0.009, -0.009]


def test_disconnect_gate_rejects_first_five_outside_threshold() -> None:
    clock = _Clock()
    robot = _Robot([[0.0101, 0.0, 0.0, 0.0, 0.0, -1.259]])

    result = _confirm(
        robot,
        clock,
        consecutive_samples=1,
        timeout_s=0.06,
    )

    assert result.confirmed is False
    assert result.max_error_rad == 0.0101


def test_i2rt_control_threads_stop_before_can_socket_closes() -> None:
    events: list[str] = []
    motor_started = threading.Event()

    class MotorChain:
        running = True
        start_thread_flag = True

        def _set_torques_and_update_state(self) -> None:
            motor_started.set()
            while self.running:
                time.sleep(0.001)
            events.append("motor_thread_stopped")

        def close(self) -> None:
            assert "motor_thread_stopped" in events
            events.append("can_socket_closed")

    class Robot:
        def __init__(self) -> None:
            self.motor_chain = MotorChain()
            self._stop_event = threading.Event()
            self._server_thread = threading.Thread(target=self._serve)
            self._server_thread.start()

        def _serve(self) -> None:
            self._stop_event.wait()
            events.append("robot_server_stopped")

        def close(self) -> None:
            self._stop_event.set()
            self._server_thread.join()
            self.motor_chain.close()
            events.append("robot_closed")

    robot = Robot()
    motor_thread = threading.Thread(
        target=robot.motor_chain._set_torques_and_update_state
    )
    motor_thread.start()
    assert motor_started.wait(timeout=1.0)

    close_yam_robot_without_can_race(robot)

    assert not robot._server_thread.is_alive()
    assert not motor_thread.is_alive()
    assert events.index("robot_server_stopped") < events.index("can_socket_closed")
    assert events.index("motor_thread_stopped") < events.index("can_socket_closed")


def test_non_i2rt_robot_uses_normal_close() -> None:
    class Robot:
        closed = False

        def close(self) -> None:
            self.closed = True

    robot = Robot()

    close_yam_robot_without_can_race(robot)

    assert robot.closed is True
