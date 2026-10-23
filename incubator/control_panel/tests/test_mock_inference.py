from __future__ import annotations

import numpy as np
import pytest

from incubator.control_panel.mock_inference import ControlPanelNoMotionPolicy


class Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def robot_state(left_grip: float = 0.02, right_grip: float = 0.02) -> np.ndarray:
    state = np.arange(44, dtype=np.float64) / 10.0
    state[14:22] = [0.31, -0.22, 0.48, 0.71, 0.12, -0.34, 0.59, left_grip]
    state[36:44] = [-0.27, 0.18, 0.52, 0.66, -0.11, 0.42, -0.60, right_grip]
    return state


def test_mock_policy_holds_live_tcp_and_synchronizes_grippers() -> None:
    clock = Clock()
    policy = ControlPanelNoMotionPolicy(
        gripper_min=0.0,
        gripper_max=0.08,
        gripper_travel_seconds=2.0,
        clock=clock,
    )
    state = robot_state()
    result = policy.predict(
        {"RizonRobot_2": state},
        [{"device_name": "RizonRobot_2", "action_dim": 16}],
        chunk_length=3,
    )["RizonRobot_2"]

    assert result.shape == (3, 16)
    expected_left = np.array(
        [state[14], state[15], state[16], state[18], state[19], state[20], state[17]]
    )
    expected_right = np.array(
        [state[36], state[37], state[38], state[40], state[41], state[42], state[39]]
    )
    np.testing.assert_array_equal(result[0, :7], expected_left)
    np.testing.assert_array_equal(result[0, 8:15], expected_right)
    assert result[0, 7] == pytest.approx(0.02)
    assert result[0, 15] == pytest.approx(0.02)
    np.testing.assert_array_equal(result, np.repeat(result[:1], 3, axis=0))

    clock.now += 0.5
    moved = policy.predict(
        {"RizonRobot_2": state},
        [{"device_name": "RizonRobot_2", "action_dim": 16}],
        chunk_length=1,
    )["RizonRobot_2"][0]
    assert moved[7] == pytest.approx(0.04)
    assert moved[15] == pytest.approx(moved[7])
    np.testing.assert_array_equal(moved[:7], expected_left)
    np.testing.assert_array_equal(moved[8:15], expected_right)


def test_mock_gripper_ramp_reflects_at_limits() -> None:
    clock = Clock()
    policy = ControlPanelNoMotionPolicy(
        gripper_min=0.0,
        gripper_max=0.08,
        gripper_travel_seconds=1.0,
        clock=clock,
    )
    state = robot_state(0.07, 0.07)
    action_config = [{"device_name": "RizonRobot_2", "action_dim": 16}]
    policy.predict({"RizonRobot_2": state}, action_config, chunk_length=1)
    clock.now += 0.5
    reflected = policy.predict(
        {"RizonRobot_2": state}, action_config, chunk_length=1
    )["RizonRobot_2"][0]
    assert reflected[7] == pytest.approx(0.05)
    assert reflected[15] == pytest.approx(0.05)
    clock.now += 0.5
    descending = policy.predict(
        {"RizonRobot_2": state}, action_config, chunk_length=1
    )["RizonRobot_2"][0]
    assert descending[7] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "observation,action_configs",
    [
        ({}, [{"device_name": "RizonRobot_2", "action_dim": 16}]),
        ({"RizonRobot_2": np.zeros(43)}, [{"device_name": "RizonRobot_2", "action_dim": 16}]),
        ({"RizonRobot_2": robot_state()}, [{"device_name": "RizonRobot_2", "action_dim": 14}]),
        ({"RizonRobot_2": robot_state()}, []),
    ],
)
def test_mock_policy_rejects_unsupported_inputs(
    observation: dict[str, np.ndarray],
    action_configs: list[dict[str, object]],
) -> None:
    policy = ControlPanelNoMotionPolicy()
    with pytest.raises(ValueError):
        policy.predict(observation, action_configs)
