"""Hardware-free coverage for I2RT randomized follower homes."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from devices.robots.i2rt import I2RTRobot
from utils.i2rt_random_home import (
    publish_random_home_request,
    publish_random_home_target,
    read_random_home_result,
    read_random_home_target,
)


def _random_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "position_radius_m": 0.02,
        "rotation_euler_range_deg": np.array([3.0, 4.0, 5.0]),
        "independent_per_arm": True,
        "max_ik_attempts": 3,
        "ik_max_iters": 20,
        "ik_position_tolerance_m": 1e-4,
        "ik_orientation_tolerance_deg": 0.2,
        "sync_wait_timeout_s": 0.1,
        "fallback_to_nominal_home": True,
        "randomize_on_startup": True,
        "randomize_on_episode_reset": True,
        "randomize_on_recovery": False,
    }
    params.update(overrides)
    return params


def _make_random_init_robot(seed: int = 7) -> I2RTRobot:
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 1
    robot.running = False
    robot._i2rt_stopped = True
    robot.is_follower_role = True
    robot.is_leader_role = False
    robot.random_init_pose = True
    robot.random_init_pose_sync = True
    robot.random_init_pose_params = _random_params()
    robot._random_init_pose_rng = np.random.default_rng(seed)
    robot._i2rt_kinematics = None
    robot._i2rt_kinematics_backend = None
    return robot


def _make_go_home_robot(
    *,
    follower: bool,
    request_path: Path,
    target_path: Path,
) -> I2RTRobot:
    """Build the narrow I2RT surface needed to exercise Home without hardware."""
    robot = _make_random_init_robot()
    robot.is_follower_role = follower
    robot.is_leader_role = not follower
    robot.device_id = 1 if follower else 0
    robot.random_init_pose = follower
    robot.home_joint_pos = np.array(
        [0.0, 1.15, 1.0, -0.45, 0.0, 0.0, 0.21]
        + [0.0, 1.15, 1.0, -0.45, 0.0, 0.0, 0.34],
        dtype=np.float64,
    )
    robot._random_home_request_path = str(request_path)
    robot._random_home_target_path = str(target_path)
    robot._random_home_result_pattern = str(
        target_path.parent / "random-home-result-{device_id}.json"
    )
    robot._left_client = object()
    robot._right_client = object()
    robot._last_trusted_get_info_state_left = None
    robot._last_trusted_get_info_state_right = None
    robot._trusted_state_lock = threading.Lock()
    robot.dual_control_mode = True
    robot._last_button_left = 0.0
    robot._last_button_right = 0.0
    robot._last_control_flag = 0.0
    robot.moves: list[dict[str, np.ndarray]] = []
    robot._smooth_joint_move = lambda **kwargs: robot.moves.append(kwargs)
    robot._follower_get_info = lambda _client, _source: np.zeros(7, dtype=np.float64)
    robot._leader_get_info = lambda _client, _source: (
        np.zeros(7, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )
    return robot


def test_random_pose_offsets_stay_inside_configured_bounds() -> None:
    robot = _make_random_init_robot()
    radius_m = float(robot.random_init_pose_params["position_radius_m"])
    rotation_range_deg = np.asarray(
        robot.random_init_pose_params["rotation_euler_range_deg"], dtype=np.float64
    )

    for _ in range(500):
        position_delta, euler_delta_deg = robot._sample_random_init_pose_offset()

        assert np.linalg.norm(position_delta) <= radius_m
        assert np.all(np.abs(euler_delta_deg) <= rotation_range_deg)


def test_randomized_home_replaces_arm_joints_but_preserves_grippers() -> None:
    robot = _make_random_init_robot()
    nominal = np.array([0.0, 1.15, 1.0, -0.45, 0.0, 0.0, 0.99] * 2)

    def solve(
        nominal_q: np.ndarray,
        _position_delta: np.ndarray,
        _euler_delta_deg: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(nominal_q) + 0.01

    robot._solve_randomized_home_arm = solve
    randomized = robot._generate_randomized_home(nominal)

    assert randomized is not None
    np.testing.assert_allclose(randomized[:6], nominal[:6] + 0.01)
    np.testing.assert_allclose(randomized[7:13], nominal[7:13] + 0.01)
    assert randomized[6] == nominal[6]
    assert randomized[13] == nominal[13]


def test_shared_arm_offset_is_reused_when_independent_sampling_is_disabled() -> None:
    robot = _make_random_init_robot()
    robot.random_init_pose_params["independent_per_arm"] = False
    position_delta = np.array([0.01, -0.01, 0.005])
    euler_delta_deg = np.array([1.0, -2.0, 3.0])
    robot._sample_random_init_pose_offset = lambda: (
        position_delta.copy(), euler_delta_deg.copy()
    )
    solve_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def solve(
        nominal_q: np.ndarray,
        current_position_delta: np.ndarray,
        current_euler_delta_deg: np.ndarray,
    ) -> np.ndarray:
        solve_calls.append(
            (current_position_delta.copy(), current_euler_delta_deg.copy())
        )
        return np.asarray(nominal_q)

    robot._solve_randomized_home_arm = solve
    randomized = robot._generate_randomized_home(np.arange(14, dtype=np.float64))

    assert randomized is not None
    assert len(solve_calls) == 2
    np.testing.assert_array_equal(solve_calls[0][0], solve_calls[1][0])
    np.testing.assert_array_equal(solve_calls[0][1], solve_calls[1][1])


def test_ik_failures_exhaust_attempts_without_mutating_nominal_home() -> None:
    robot = _make_random_init_robot()
    nominal = np.arange(14, dtype=np.float64)
    solve_calls = 0

    def fail_solve(
        _nominal_q: np.ndarray,
        _position_delta: np.ndarray,
        _euler_delta_deg: np.ndarray,
    ) -> None:
        nonlocal solve_calls
        solve_calls += 1
        return None

    robot._solve_randomized_home_arm = fail_solve

    assert robot._generate_randomized_home(nominal) is None
    assert solve_calls == 2 * int(robot.random_init_pose_params["max_ik_attempts"])
    np.testing.assert_array_equal(nominal, np.arange(14, dtype=np.float64))


def test_randomization_reason_and_sync_contract_excludes_recovery() -> None:
    robot = _make_random_init_robot()

    assert robot._should_randomize_init_pose("startup")
    assert robot._should_randomize_init_pose("episode_reset")
    assert not robot._should_randomize_init_pose("recovery")
    assert robot._home_requires_random_sync("startup")
    assert robot._home_requires_random_sync("episode_reset")
    assert not robot._home_requires_random_sync("recovery")


def test_smooth_joint_move_commands_the_exact_final_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = object.__new__(I2RTRobot)
    robot.device_name = "I2RTRobot"
    robot.device_id = 1
    robot.running = False
    robot._i2rt_stopped = True
    robot._left_client = object()
    robot._right_client = object()
    commands: list[tuple[object, np.ndarray, str]] = []
    robot._command_joint_pos = lambda client, target, source: commands.append(
        (client, np.asarray(target, dtype=np.float64).copy(), source)
    )
    monkeypatch.setattr("devices.robots.i2rt.time.sleep", lambda _seconds: None)

    left_target = np.array([1.0, 2.0])
    right_target = np.array([3.0, 4.0])
    robot._smooth_joint_move(
        left_target=left_target,
        right_target=right_target,
        left_current=np.zeros(2),
        right_current=np.zeros(2),
        steps=3,
        dt=0.01,
    )

    assert len(commands) == 6
    np.testing.assert_allclose(commands[-2][1], left_target)
    np.testing.assert_allclose(commands[-1][1], right_target)


def test_leader_home_mapping_keeps_its_grippers_for_both_supported_layouts() -> None:
    follower_home = np.arange(14, dtype=np.float64)
    leader_14 = np.full(14, -1.0)
    leader_14[[6, 13]] = [0.3, 0.4]
    leader_12 = np.full(12, -1.0)

    mapped_14 = I2RTRobot._apply_follower_home_to_leader(leader_14, follower_home)
    mapped_12 = I2RTRobot._apply_follower_home_to_leader(leader_12, follower_home)

    np.testing.assert_array_equal(mapped_14[:6], follower_home[:6])
    np.testing.assert_array_equal(mapped_14[7:13], follower_home[7:13])
    assert mapped_14[6] == 0.3
    assert mapped_14[13] == 0.4
    np.testing.assert_array_equal(mapped_12[:6], follower_home[:6])
    np.testing.assert_array_equal(mapped_12[6:12], follower_home[7:13])


def test_ik_solution_is_rejected_when_outside_vendor_joint_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _make_random_init_robot()

    class FakeKinematics:
        def fk(self, _joint_pos: np.ndarray, _site_name: str) -> np.ndarray:
            return np.eye(4)

        def ik(self, *_: object, **__: object) -> tuple[bool, np.ndarray]:
            return True, np.full(6, 2.0)

    robot._get_i2rt_kinematics = lambda: FakeKinematics()
    monkeypatch.setattr(
        "utils.i2rt_kinematics.arm_joint_limits",
        lambda _kinematics: (np.full(6, -1.0), np.full(6, 1.0)),
    )

    assert robot._solve_randomized_home_arm(
        np.zeros(6), np.zeros(3), np.zeros(3)
    ) is None


def test_robot_waiter_ignores_a_stale_target_for_another_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _make_random_init_robot()
    robot.is_follower_role = False
    robot.is_leader_role = True
    robot.device_id = 0
    robot._random_home_target_path = str(tmp_path / "target.json")
    robot.random_init_pose_params["sync_wait_timeout_s"] = 0.1
    publish_random_home_target(
        robot._random_home_target_path,
        request_id="a" * 32,
        home_joint_pos=np.arange(14, dtype=np.float64),
        source_device_id=1,
    )
    clock = [0.0]
    monkeypatch.setattr("devices.robots.i2rt.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "devices.robots.i2rt.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    assert robot._wait_for_random_home_target(request_id="b" * 32) is None


def test_go_home_publishes_applied_follower_target_then_leader_uses_it(
    tmp_path: Path,
) -> None:
    """Leader can move only to the target the follower applied for this request."""
    request_path = tmp_path / "request.json"
    target_path = tmp_path / "target.json"
    request_id = publish_random_home_request(
        request_path,
        reason="episode_reset",
        request_id="b" * 32,
    )
    follower = _make_go_home_robot(
        follower=True,
        request_path=request_path,
        target_path=target_path,
    )
    randomized = follower.home_joint_pos.copy()
    randomized[:6] += 0.02
    randomized[7:13] -= 0.03
    follower._generate_randomized_home = lambda _nominal: randomized.copy()

    assert follower._go_home(
        reason="episode_reset", random_home_request_id=request_id
    )
    assert len(follower.moves) == 1
    published = read_random_home_target(
        target_path,
        request_id=request_id,
        expected_source_device_id=1,
    )
    np.testing.assert_allclose(published, randomized)

    leader = _make_go_home_robot(
        follower=False,
        request_path=request_path,
        target_path=target_path,
    )
    assert leader._go_home(
        reason="episode_reset", random_home_request_id=request_id
    )
    assert len(leader.moves) == 1
    np.testing.assert_allclose(leader.moves[0]["left_target"], randomized[:6])
    np.testing.assert_allclose(leader.moves[0]["right_target"], randomized[7:13])


def test_go_home_rejects_command_with_a_stale_request_id(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_id = publish_random_home_request(
        request_path,
        reason="episode_reset",
        request_id="b" * 32,
    )
    follower = _make_go_home_robot(
        follower=True,
        request_path=request_path,
        target_path=tmp_path / "target.json",
    )

    assert request_id != "a" * 32
    assert not follower._go_home(
        reason="episode_reset", random_home_request_id="a" * 32
    )
    assert follower.moves == []


def test_i2rt_home_command_publishes_matching_failure_result(tmp_path: Path) -> None:
    """A failed device Home must be observable by the episode reset barrier."""
    request_path = tmp_path / "request.json"
    target_path = tmp_path / "target.json"
    request_id = "c" * 32
    follower = _make_go_home_robot(
        follower=True,
        request_path=request_path,
        target_path=target_path,
    )
    follower._robot_home_command_pattern = str(
        tmp_path / "{device_name}_{device_id}_home"
    )
    follower._clear_control_shm = lambda: None
    follower._apply_bilateral_gains_for_mode = lambda: None
    follower._go_home = lambda **_kwargs: False
    follower._notify_policy_runner_reset = lambda: pytest.fail(
        "failed Home must not reset policy execution"
    )
    command_path = tmp_path / "I2RTRobot_1_home"
    command_path.write_text(
        '{"schema_version":"i2rt/home_command/v1","home_index":0,'
        f'"random_home_request_id":"{request_id}"}}',
        encoding="utf-8",
    )

    follower._check_and_execute_home_command()

    assert read_random_home_result(
        follower._random_home_result_pattern,
        request_id=request_id,
        source_device_id=1,
    ) is False
