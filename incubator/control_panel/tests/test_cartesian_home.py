"""Hardware-free checks for straight-line Cartesian Home planning."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from incubator.control_panel.cartesian_home import (
    CartesianHomeError,
    interpolate_flexiv_poses,
    matrix_to_flexiv_pose,
    plan_joint_home_trajectory,
    pose_error,
    required_joint_waypoint_count,
    required_waypoint_count,
    resolve_joint_target_tcp,
    resolve_tcp_target_joints,
)


def _pose(x: float, y: float, z: float, quaternion=(1.0, 0.0, 0.0, 0.0)):
    return [x, y, z, *quaternion]


def test_waypoints_follow_one_translation_line_and_shortest_slerp() -> None:
    start = _pose(0.0, 0.0, 0.0)
    target = _pose(1.0, 2.0, 3.0, (0.0, 0.0, 0.0, 1.0))

    waypoints = interpolate_flexiv_poses(start, target, waypoint_count=4)

    assert len(waypoints) == 4
    for index, waypoint in enumerate(waypoints, start=1):
        assert waypoint[:3] == pytest.approx(np.asarray(target[:3]) * (index / 4.0))
        assert np.linalg.norm(waypoint[3:]) == pytest.approx(1.0)
    assert waypoints[-1] == pytest.approx(target)
    angular_errors = [pose_error(waypoint, target)[1] for waypoint in waypoints]
    assert angular_errors == sorted(angular_errors, reverse=True)


def test_antipodal_quaternion_does_not_create_a_full_rotation() -> None:
    start = _pose(0.0, 0.0, 0.0)
    same_orientation = _pose(0.1, 0.0, 0.0, (-1.0, 0.0, 0.0, 0.0))

    for waypoint in interpolate_flexiv_poses(start, same_orientation, waypoint_count=5):
        assert pose_error(waypoint, start)[1] == pytest.approx(0.0)


def test_waypoint_count_is_bounded_by_linear_and_angular_steps() -> None:
    eleven_degrees = math.radians(11.0)
    target = _pose(
        0.101,
        0.0,
        0.0,
        (
            math.cos(eleven_degrees / 2.0),
            0.0,
            0.0,
            math.sin(eleven_degrees / 2.0),
        ),
    )
    assert (
        required_waypoint_count(
            _pose(0.0, 0.0, 0.0),
            target,
            max_linear_step_m=0.025,
            max_angular_step_rad=math.radians(5.0),
        )
        == 5
    )

    with pytest.raises(CartesianHomeError, match="above limit"):
        required_waypoint_count(
            _pose(0.0, 0.0, 0.0),
            _pose(3.0, 0.0, 0.0),
            max_linear_step_m=0.01,
            max_angular_step_rad=math.radians(5.0),
            max_waypoints=200,
        )


class _LinearChain:
    def forward(self, joints):
        transform = np.eye(4, dtype=float)
        transform[0, 3] = float(np.asarray(joints)[0])
        return transform


class _YawChain:
    def forward(self, joints):
        angle = float(np.asarray(joints)[0])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
        return transform


class _LinearIKChain:
    lower_limits = (-2.0,) * 7
    upper_limits = (2.0,) * 7

    def __init__(self):
        self.calls = []

    def forward(self, joints):
        transform = np.eye(4, dtype=float)
        transform[0, 3] = float(np.asarray(joints)[0])
        return transform

    def solve_inverse(
        self,
        target,
        *,
        reference_joints,
        initial_joints,
        parameters,
        tool_transform,
        deadline,
    ):
        del parameters, tool_transform, deadline
        reference = np.asarray(reference_joints, dtype=float)
        initial = np.asarray(initial_joints, dtype=float)
        solved = reference.copy()
        solved[0] = float(np.asarray(target)[0, 3])
        self.calls.append(
            {
                "reference": reference.copy(),
                "initial": initial.copy(),
                "solved": solved.copy(),
            }
        )
        return SimpleNamespace(
            success=True,
            joints=tuple(solved),
            optimizer_nfev=3,
            optimizer_message="ok",
        )


class _CurvedJointChain(_LinearIKChain):
    def forward(self, joints):
        first = float(np.asarray(joints)[0])
        transform = np.eye(4, dtype=float)
        transform[0, 3] = first
        transform[2, 3] = 0.2 * first * (1.0 - first)
        return transform


class _DeltaJumpChain(_LinearIKChain):
    def solve_inverse(self, *args, **kwargs):
        outcome = super().solve_inverse(*args, **kwargs)
        solved = np.asarray(outcome.joints, dtype=float)
        solved[1] = 0.04 if len(self.calls) == 1 else -0.04
        outcome.joints = tuple(solved)
        self.calls[-1]["solved"] = solved.copy()
        return outcome


def _ik_parameters():
    return SimpleNamespace(joint_limit_margin_rad=0.01)


def test_joint_target_tcp_uses_measured_world_anchor_and_active_tool() -> None:
    target = resolve_joint_target_tcp(
        _LinearChain(),
        current_joints=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_joints=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        current_flange_pose=_pose(11.0, 3.0, 4.0),
        current_tcp_pose=_pose(11.0, 3.0, 4.2),
    )

    assert target == pytest.approx(_pose(12.0, 3.0, 4.2))


def test_joint_target_tcp_rotates_active_tool_offset_with_target_flange() -> None:
    target = resolve_joint_target_tcp(
        _YawChain(),
        current_joints=[0.0] * 7,
        target_joints=[math.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        current_flange_pose=_pose(0.0, 0.0, 0.0),
        current_tcp_pose=_pose(0.1, 0.0, 0.0),
    )
    expected = _pose(
        0.0,
        0.1,
        0.0,
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
    )

    linear_error, angular_error = pose_error(target, expected)
    assert linear_error == pytest.approx(0.0, abs=1e-12)
    assert angular_error == pytest.approx(0.0, abs=1e-12)


def test_joint_target_tcp_rejects_non_finite_measured_state() -> None:
    with pytest.raises(CartesianHomeError, match="current measured TCP pose"):
        resolve_joint_target_tcp(
            _LinearChain(),
            current_joints=[0.0] * 7,
            target_joints=[0.1] * 7,
            current_flange_pose=_pose(0.0, 0.0, 0.0),
            current_tcp_pose=_pose(float("nan"), 0.0, 0.2),
        )


def test_joint_waypoint_count_bounds_the_largest_joint_delta() -> None:
    assert (
        required_joint_waypoint_count(
            [0.0] * 7,
            [0.31, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
            max_joint_step_rad=0.1,
        )
        == 4
    )


def test_tcp_target_ik_uses_current_as_initial_and_requested_reference() -> None:
    chain = _LinearIKChain()
    solved = resolve_tcp_target_joints(
        chain,
        current_joints=[0.0] * 7,
        current_flange_pose=_pose(0.0, 0.0, 0.0),
        current_tcp_pose=_pose(0.0, 0.0, 0.0),
        target_tcp_pose=_pose(0.2, 0.0, 0.0),
        reference_joints=[0.1] * 7,
        parameters=_ik_parameters(),
    )

    assert solved[0] == pytest.approx(0.2)
    assert chain.calls[0]["initial"] == pytest.approx([0.0] * 7)
    assert chain.calls[0]["reference"] == pytest.approx([0.1] * 7)


def test_joint_home_plan_warm_starts_each_ik_and_ends_at_exact_home() -> None:
    chain = _LinearIKChain()
    target_joints = [0.2, 0.12, -0.08, 0.04, 0.0, 0.0, 0.0]
    plan = plan_joint_home_trajectory(
        chain,
        current_joints=[0.0] * 7,
        target_joints=target_joints,
        current_flange_pose=_pose(0.0, 0.0, 0.0),
        current_tcp_pose=_pose(0.0, 0.0, 0.0),
        target_tcp_pose=_pose(0.2, 0.0, 0.0),
        waypoint_count=4,
        max_joint_step_rad=0.08,
        max_joint_delta_change_rad=0.06,
        max_path_position_deviation_m=0.001,
        max_path_orientation_deviation_rad=math.radians(0.1),
        path_validation_samples=4,
        parameters=_ik_parameters(),
    )

    assert len(plan.joint_waypoints) == 4
    assert plan.joint_waypoints[-1] == pytest.approx(target_joints)
    assert len(chain.calls) == 3
    assert chain.calls[0]["initial"] == pytest.approx([0.0] * 7)
    assert chain.calls[1]["initial"] == pytest.approx(plan.joint_waypoints[0])
    assert chain.calls[0]["reference"] == pytest.approx(
        np.asarray(target_joints) * 0.25
    )
    assert chain.calls[2]["reference"] == pytest.approx(
        np.asarray(target_joints) * 0.75
    )
    assert plan.optimizer_evaluations == 9


def test_real_rizon_chain_plans_continuously_back_to_exact_home() -> None:
    pytest.importorskip("scipy")
    from incubator.control_panel.home_pose_sampler import (
        FlexivKinematicChain,
        flexiv_urdf_for_side,
        sampling_preset,
    )

    home = np.asarray(
        (
            1.36721670627594,
            1.1971391439437866,
            0.8350791096687317,
            1.7742716073989868,
            0.47819629311561584,
            1.3499644994735718,
            0.22787520289421082,
        )
    )
    current = home + np.asarray((0.12, -0.08, 0.07, 0.09, -0.06, 0.06, -0.04))
    chain = FlexivKinematicChain.from_urdf(
        flexiv_urdf_for_side("left"), expected_variant="Rizon4"
    )
    current_tcp = matrix_to_flexiv_pose(chain.forward(current))
    target_tcp = matrix_to_flexiv_pose(chain.forward(home))
    max_joint_step = math.radians(7.5)
    count = max(
        required_waypoint_count(
            current_tcp,
            target_tcp,
            max_linear_step_m=0.025,
            max_angular_step_rad=math.radians(5.0),
        ),
        required_joint_waypoint_count(
            current,
            home,
            max_joint_step_rad=max_joint_step,
        ),
    )
    params = replace(
        sampling_preset("conservative"),
        reference_weight=0.02,
        max_optimizer_evaluations=220,
    )

    plan = plan_joint_home_trajectory(
        chain,
        current_joints=current,
        target_joints=home,
        current_flange_pose=current_tcp,
        current_tcp_pose=current_tcp,
        target_tcp_pose=target_tcp,
        waypoint_count=count,
        max_joint_step_rad=max_joint_step,
        max_joint_delta_change_rad=math.radians(5.0),
        max_path_position_deviation_m=0.008,
        max_path_orientation_deviation_rad=math.radians(3.0),
        path_validation_samples=4,
        parameters=params,
    )

    assert plan.joint_waypoints[-1] == pytest.approx(home)
    assert plan.optimizer_evaluations > 0
    assert plan.max_position_deviation_m <= 0.008
    assert plan.max_orientation_deviation_rad <= math.radians(3.0)


def test_joint_home_plan_rejects_joint_interpolation_that_bows_off_tcp_line() -> None:
    chain = _CurvedJointChain()
    with pytest.raises(CartesianHomeError, match="leaves the validated TCP corridor"):
        plan_joint_home_trajectory(
            chain,
            current_joints=[0.0] * 7,
            target_joints=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            current_flange_pose=_pose(0.0, 0.0, 0.0),
            current_tcp_pose=_pose(0.0, 0.0, 0.0),
            target_tcp_pose=_pose(1.0, 0.0, 0.0),
            waypoint_count=1,
            max_joint_step_rad=1.1,
            max_joint_delta_change_rad=1.1,
            max_path_position_deviation_m=0.01,
            max_path_orientation_deviation_rad=math.radians(0.1),
            path_validation_samples=4,
            parameters=_ik_parameters(),
        )


def test_tcp_target_ik_rejects_solution_outside_safe_joint_limits() -> None:
    with pytest.raises(CartesianHomeError, match="joint-limit margin"):
        resolve_tcp_target_joints(
            _LinearIKChain(),
            current_joints=[0.0] * 7,
            current_flange_pose=_pose(0.0, 0.0, 0.0),
            current_tcp_pose=_pose(0.0, 0.0, 0.0),
            target_tcp_pose=_pose(2.1, 0.0, 0.0),
            reference_joints=[0.0] * 7,
            parameters=_ik_parameters(),
        )


def test_joint_home_plan_rejects_abrupt_joint_delta_change() -> None:
    with pytest.raises(CartesianHomeError, match="changes joint delta"):
        plan_joint_home_trajectory(
            _DeltaJumpChain(),
            current_joints=[0.0] * 7,
            target_joints=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            current_flange_pose=_pose(0.0, 0.0, 0.0),
            current_tcp_pose=_pose(0.0, 0.0, 0.0),
            target_tcp_pose=_pose(0.2, 0.0, 0.0),
            waypoint_count=4,
            max_joint_step_rad=0.2,
            max_joint_delta_change_rad=0.05,
            max_path_position_deviation_m=0.1,
            max_path_orientation_deviation_rad=math.radians(1.0),
            path_validation_samples=2,
            parameters=_ik_parameters(),
        )


@pytest.mark.parametrize(
    "bad_pose",
    (
        [0.0] * 7,
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, float("nan")],
        [0.0] * 6,
    ),
)
def test_invalid_pose_fails_closed(bad_pose) -> None:
    with pytest.raises(CartesianHomeError):
        interpolate_flexiv_poses(
            bad_pose,
            _pose(0.0, 0.0, 0.0),
            waypoint_count=1,
        )
