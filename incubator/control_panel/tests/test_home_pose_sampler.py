"""Hardware-free coverage for deterministic Flexiv FK/IK home-pose sampling."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from incubator.control_panel.home_pose_sampler import (
    FlexivKinematicChain,
    InvalidSamplingInput,
    KinematicsModelError,
    SamplingCancelled,
    flexiv_urdf_for_side,
    sample_home_poses,
    sampling_parameters_from_mapping,
    sampling_preset,
)


LEFT_HOME = (
    1.36721670627594,
    1.1971391439437866,
    0.8350791096687317,
    1.7742716073989868,
    0.47819629311561584,
    1.3499644994735718,
    0.22787520289421082,
)
RIGHT_HOME = (
    1.2663822174072266,
    1.3955061435699463,
    0.6105675101280212,
    1.724402904510498,
    1.0657531023025513,
    1.0845757722854614,
    0.028192929923534393,
)


def _synthetic_urdf(path: Path) -> Path:
    joints = []
    for index in range(1, 8):
        parent = 'base_link' if index == 1 else f'link{index - 1}'
        child = f'link{index}'
        xyz = '1 0 0' if index == 1 else '0 0 0'
        joints.append(
            f'''<joint name="joint{index}" type="revolute">
  <parent link="{parent}"/><child link="{child}"/>
  <origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="0 0 1"/>
  <limit lower="-3" upper="3" effort="1" velocity="1"/>
</joint>'''
        )
    joints.append(
        '''<joint name="flange_fixed" type="fixed">
  <parent link="link7"/><child link="flange"/>
  <origin xyz="1 0 0" rpy="0 0 0"/>
</joint>'''
    )
    links = '<link name="base_link"/><link name="flange"/>' + ''.join(
        f'<link name="link{index}"/>' for index in range(1, 8)
    )
    path.write_text(
        '<robot name="Rizon4">' + links + ''.join(joints) + '</robot>',
        encoding='utf-8',
    )
    return path


def test_fk_uses_standard_urdf_origin_then_joint_motion(tmp_path: Path) -> None:
    chain = FlexivKinematicChain.from_urdf(_synthetic_urdf(tmp_path / 'simple.urdf'))
    transform = chain.forward((math.pi / 2, 0, 0, 0, 0, 0, 0))
    np.testing.assert_allclose(transform[:3, 3], (1.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(
        transform[:3, :3],
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        atol=1e-12,
    )


def test_checked_in_side_models_are_distinct_and_home_fk_is_finite() -> None:
    left_path = flexiv_urdf_for_side('left')
    right_path = flexiv_urdf_for_side('right')
    assert left_path.name == 'flexiv_Rizon4_kinematics.urdf'
    assert right_path.name == 'flexiv_Rizon4R_kinematics.urdf'
    left = FlexivKinematicChain.from_urdf(left_path, expected_variant='Rizon4')
    right = FlexivKinematicChain.from_urdf(right_path, expected_variant='Rizon4R')
    for chain, home in ((left, LEFT_HOME), (right, RIGHT_HOME)):
        pose = chain.forward(home)
        assert np.all(np.isfinite(pose))
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-10)
        assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(
        left.forward(LEFT_HOME)[:3, 3],
        (-0.29978659810489505, -0.11115632980933415, 0.8686232341210115),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        right.forward(RIGHT_HOME)[:3, 3],
        (-0.2512752218379522, 0.1173545949767776, 0.8614299411210088),
        atol=1e-12,
    )


def test_wrong_side_urdf_is_rejected() -> None:
    with pytest.raises(KinematicsModelError, match='variant mismatch'):
        sample_home_poses(
            side='left',
            urdf_path=flexiv_urdf_for_side('right'),
            home_joints=LEFT_HOME,
        )


def test_ik_roundtrip_has_small_measured_fk_residual() -> None:
    pytest.importorskip('scipy')
    chain = FlexivKinematicChain.from_urdf(
        flexiv_urdf_for_side('left'), expected_variant='Rizon4'
    )
    target_joints = np.asarray(LEFT_HOME) + np.asarray((0.03, -0.02, 0.025, 0.03, -0.02, 0.02, -0.015))
    target = chain.forward(target_joints)
    params = dataclasses.replace(
        sampling_preset('conservative'),
        position_residual_m=2e-4,
        orientation_residual_rad=5e-4,
        reference_weight=1e-6,
        max_optimizer_evaluations=220,
    )
    solved = chain.solve_inverse(target, reference_joints=LEFT_HOME, parameters=params)
    assert solved.success, solved.optimizer_message
    assert solved.joints is not None
    assert solved.position_residual_m is not None and solved.position_residual_m <= 2e-4
    assert solved.orientation_residual_rad is not None and solved.orientation_residual_rad <= 5e-4


def test_ik_accepts_a_continuous_initial_guess_separate_from_home_reference() -> None:
    pytest.importorskip('scipy')
    chain = FlexivKinematicChain.from_urdf(
        flexiv_urdf_for_side('left'), expected_variant='Rizon4'
    )
    target_joints = np.asarray(LEFT_HOME) + np.asarray(
        (0.02, -0.015, 0.02, 0.015, -0.01, 0.01, -0.01)
    )
    target = chain.forward(target_joints)
    params = dataclasses.replace(
        sampling_preset('conservative'),
        position_residual_m=2e-4,
        orientation_residual_rad=5e-4,
        reference_weight=1e-6,
        max_optimizer_evaluations=220,
    )

    solved = chain.solve_inverse(
        target,
        reference_joints=LEFT_HOME,
        initial_joints=target_joints,
        parameters=params,
    )

    assert solved.success, solved.optimizer_message
    assert solved.joints is not None


def test_ik_rejects_an_initial_guess_outside_joint_limits() -> None:
    pytest.importorskip('scipy')
    chain = FlexivKinematicChain.from_urdf(
        flexiv_urdf_for_side('left'), expected_variant='Rizon4'
    )
    with pytest.raises(InvalidSamplingInput, match='initial_joints violate'):
        chain.solve_inverse(
            chain.forward(LEFT_HOME),
            reference_joints=LEFT_HOME,
            initial_joints=[100.0] * 7,
        )


def test_sampling_is_deterministic_finite_and_respects_margin() -> None:
    pytest.importorskip('scipy')
    params = dataclasses.replace(
        sampling_preset('conservative'),
        candidate_count=2,
        attempts=48,
        min_joint_distance_rad=0.025,
        min_candidate_distance_rad=0.025,
        timeout_seconds=10.0,
    )
    kwargs = {
        'side': 'left',
        'urdf_path': flexiv_urdf_for_side('left'),
        'home_joints': LEFT_HOME,
        'parameters': params,
        'seed': 713,
    }
    first = sample_home_poses(**kwargs)
    second = sample_home_poses(**kwargs)
    assert first.status == second.status == 'complete'
    assert [candidate.joints for candidate in first.candidates] == [
        candidate.joints for candidate in second.candidates
    ]
    assert len(first.candidates) == 2
    chain = FlexivKinematicChain.from_urdf(flexiv_urdf_for_side('left'))
    lower = np.asarray(chain.lower_limits) + params.joint_limit_margin_rad
    upper = np.asarray(chain.upper_limits) - params.joint_limit_margin_rad
    for candidate in first.candidates:
        joints = np.asarray(candidate.joints)
        assert joints.shape == (7,) and np.all(np.isfinite(joints))
        assert np.all(joints >= lower) and np.all(joints <= upper)
        assert candidate.position_residual_m <= params.position_residual_m
        assert candidate.orientation_residual_rad <= params.orientation_residual_rad
    assert not first.collision_check.performed
    assert first.collision_check.safe is None
    assert 'not collision-safe' in first.collision_check.reason


def test_exact_home_targets_fail_cleanly_when_minimum_distance_is_required() -> None:
    pytest.importorskip('scipy')
    params = dataclasses.replace(
        sampling_preset('conservative'),
        candidate_count=1,
        attempts=3,
        translation_bounds_m=(0.0, 0.0, 0.0),
        translation_radius_m=None,
        orientation_angle_bound_rad=0.0,
    )
    result = sample_home_poses(
        side='right',
        urdf_path=flexiv_urdf_for_side('right'),
        home_joints=RIGHT_HOME,
        parameters=params,
        seed=1,
    )
    assert result.status == 'failed'
    assert result.candidates == ()
    assert dict(result.failure_counts)['too_close_to_home'] == 3


def test_frame_metadata_only_calls_pose_tcp_with_explicit_tool_transform() -> None:
    pytest.importorskip('scipy')
    params = dataclasses.replace(
        sampling_preset('conservative'),
        candidate_count=1,
        attempts=1,
        translation_bounds_m=(0.0, 0.0, 0.0),
        translation_radius_m=None,
        orientation_angle_bound_rad=0.0,
    )
    common = {
        'side': 'left',
        'urdf_path': flexiv_urdf_for_side('left'),
        'home_joints': LEFT_HOME,
        'parameters': params,
    }
    flange = sample_home_poses(**common)
    assert flange.target_frame == 'flange'
    assert flange.tool_status == 'controller-active-unknown'
    tool = np.eye(4)
    tool[2, 3] = 0.12
    tcp = sample_home_poses(**common, tool_transform=tool)
    assert tcp.target_frame == 'tcp'
    assert tcp.tool_status == 'explicit-config-transform'
    assert tcp.home_target_pose.position_m != tcp.home_flange_pose.position_m


def test_parameters_are_whitelisted_and_resource_bounded() -> None:
    assert sampling_preset('wide').timeout_seconds <= 10.0
    updated = sampling_parameters_from_mapping({'count': 3, 'attempts': 12})
    assert updated.candidate_count == 3 and updated.attempts == 12
    with pytest.raises(InvalidSamplingInput, match='unknown sampling parameter'):
        sampling_parameters_from_mapping({'shell_command': 'rm -rf /'})
    with pytest.raises(InvalidSamplingInput, match='candidate_count'):
        dataclasses.replace(sampling_preset(), candidate_count=0)


def test_cancellation_is_checked_before_optimizer_work() -> None:
    pytest.importorskip('scipy')
    with pytest.raises(SamplingCancelled):
        sample_home_poses(
            side='left',
            urdf_path=flexiv_urdf_for_side('left'),
            home_joints=LEFT_HOME,
            should_cancel=lambda: True,
        )
