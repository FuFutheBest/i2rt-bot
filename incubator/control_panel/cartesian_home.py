"""Pure Cartesian planning helpers for bounded Rizon reset motion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


POSE_SIZE = 7
JOINT_COUNT = 7
MAX_WAYPOINTS = 200


class CartesianHomeError(ValueError):
    """A reset pose or Cartesian path is invalid or exceeds a safety bound."""


@dataclass(frozen=True)
class JointHomePlan:
    """A fully validated joint trajectory that approximates one TCP line."""

    start_joints: tuple[float, ...]
    target_joints: tuple[float, ...]
    joint_waypoints: tuple[tuple[float, ...], ...]
    tcp_waypoints: tuple[tuple[float, ...], ...]
    max_position_deviation_m: float
    max_orientation_deviation_rad: float
    optimizer_evaluations: int
    joint_lower_bounds: tuple[float, ...]
    joint_upper_bounds: tuple[float, ...]
    max_command_step_rad: float
    max_delta_change_rad: float
    terminal_requires_tcp: bool


@dataclass(frozen=True)
class _MeasuredKinematicFrame:
    world_from_nominal_base: np.ndarray
    nominal_base_from_world: np.ndarray
    flange_to_tcp: np.ndarray


def normalized_flexiv_pose(value: Any, *, name: str = "pose") -> np.ndarray:
    """Return finite ``[x, y, z, qw, qx, qy, qz]`` with a unit quaternion."""
    try:
        pose = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CartesianHomeError(f"{name} must be a finite 7D Flexiv pose") from exc
    if pose.shape != (POSE_SIZE,) or not np.all(np.isfinite(pose)):
        raise CartesianHomeError(f"{name} must be a finite 7D Flexiv pose")
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    if quaternion_norm <= 1e-12:
        raise CartesianHomeError(f"{name} quaternion must be non-zero")
    normalized = pose.copy()
    normalized[3:] /= quaternion_norm
    return normalized


def finite_joints(value: Any, *, name: str = "joints") -> np.ndarray:
    """Return one finite seven-joint vector."""
    try:
        joints = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CartesianHomeError(f"{name} must contain 7 finite values") from exc
    if joints.shape != (JOINT_COUNT,) or not np.all(np.isfinite(joints)):
        raise CartesianHomeError(f"{name} must contain 7 finite values")
    return joints


def _quaternion_to_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise CartesianHomeError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = (
                math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
                * 2.0
            )
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif diagonal_index == 1:
            scale = (
                math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
                * 2.0
            )
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = (
                math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
                * 2.0
            )
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise CartesianHomeError("rotation produced an invalid quaternion")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def flexiv_pose_to_matrix(value: Any, *, name: str = "pose") -> np.ndarray:
    pose = normalized_flexiv_pose(value, name=name)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _quaternion_to_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_flexiv_pose(value: Any, *, name: str = "transform") -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CartesianHomeError(
            f"{name} must be a finite rigid 4x4 transform"
        ) from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise CartesianHomeError(f"{name} must be a finite rigid 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise CartesianHomeError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise CartesianHomeError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise CartesianHomeError(f"{name} rotation is not proper")
    return np.concatenate((transform[:3, 3], _matrix_to_quaternion_wxyz(rotation)))


def quaternion_angular_distance(first: Any, second: Any) -> float:
    left = normalized_flexiv_pose(first, name="first pose")[3:]
    right = normalized_flexiv_pose(second, name="second pose")[3:]
    dot = min(1.0, max(-1.0, abs(float(np.dot(left, right)))))
    return 2.0 * math.acos(dot)


def pose_error(actual: Any, target: Any) -> tuple[float, float]:
    actual_pose = normalized_flexiv_pose(actual, name="actual pose")
    target_pose = normalized_flexiv_pose(target, name="target pose")
    linear = float(np.linalg.norm(actual_pose[:3] - target_pose[:3]))
    angular = quaternion_angular_distance(actual_pose, target_pose)
    return linear, angular


def required_waypoint_count(
    start: Any,
    target: Any,
    *,
    max_linear_step_m: float,
    max_angular_step_rad: float,
    max_waypoints: int = MAX_WAYPOINTS,
) -> int:
    """Return a bounded segment count that respects both Cartesian step limits."""
    if not math.isfinite(max_linear_step_m) or max_linear_step_m <= 0.0:
        raise CartesianHomeError("max_linear_step_m must be positive")
    if not math.isfinite(max_angular_step_rad) or max_angular_step_rad <= 0.0:
        raise CartesianHomeError("max_angular_step_rad must be positive")
    if (
        isinstance(max_waypoints, bool)
        or not isinstance(max_waypoints, int)
        or not 1 <= max_waypoints <= MAX_WAYPOINTS
    ):
        raise CartesianHomeError(f"max_waypoints must be in [1, {MAX_WAYPOINTS}]")
    start_pose = normalized_flexiv_pose(start, name="start pose")
    target_pose = normalized_flexiv_pose(target, name="target pose")
    linear_distance = float(np.linalg.norm(target_pose[:3] - start_pose[:3]))
    angular_distance = quaternion_angular_distance(start_pose, target_pose)
    count = max(
        1,
        int(math.ceil(linear_distance / max_linear_step_m)),
        int(math.ceil(angular_distance / max_angular_step_rad)),
    )
    if count > max_waypoints:
        raise CartesianHomeError(
            f"Cartesian Home requires {count} waypoints, above limit {max_waypoints}"
        )
    return count


def required_joint_waypoint_count(
    start_joints: Any,
    target_joints: Any,
    *,
    max_joint_step_rad: float,
    max_waypoints: int = MAX_WAYPOINTS,
) -> int:
    """Return a bounded segment count for the largest per-joint displacement."""
    if not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
        raise CartesianHomeError("max_joint_step_rad must be positive")
    if (
        isinstance(max_waypoints, bool)
        or not isinstance(max_waypoints, int)
        or not 1 <= max_waypoints <= MAX_WAYPOINTS
    ):
        raise CartesianHomeError(f"max_waypoints must be in [1, {MAX_WAYPOINTS}]")
    start = finite_joints(start_joints, name="start joints")
    target = finite_joints(target_joints, name="target joints")
    count = max(
        1,
        int(
            math.ceil(
                float(np.max(np.abs(target - start))) / max_joint_step_rad
            )
        ),
    )
    if count > max_waypoints:
        raise CartesianHomeError(
            f"Joint Home requires {count} waypoints, above limit {max_waypoints}"
        )
    return count


def _slerp_wxyz(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    first = np.asarray(start, dtype=float).copy()
    second = np.asarray(target, dtype=float).copy()
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second *= -1.0
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        result /= np.linalg.norm(result)
        return result
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / denominator * first
        + math.sin(fraction * angle) / denominator * second
    )


def interpolate_flexiv_poses(
    start: Any,
    target: Any,
    *,
    waypoint_count: int,
) -> tuple[np.ndarray, ...]:
    """Interpolate translation linearly and orientation by shortest-path SLERP."""
    if (
        isinstance(waypoint_count, bool)
        or not isinstance(waypoint_count, int)
        or not 1 <= waypoint_count <= MAX_WAYPOINTS
    ):
        raise CartesianHomeError(f"waypoint_count must be in [1, {MAX_WAYPOINTS}]")
    start_pose = normalized_flexiv_pose(start, name="start pose")
    target_pose = normalized_flexiv_pose(target, name="target pose")
    waypoints = []
    for index in range(1, waypoint_count + 1):
        fraction = index / waypoint_count
        translation = start_pose[:3] + fraction * (target_pose[:3] - start_pose[:3])
        quaternion = _slerp_wxyz(start_pose[3:], target_pose[3:], fraction)
        waypoints.append(np.concatenate((translation, quaternion)))
    return tuple(waypoints)


def _measured_kinematic_frame(
    chain: Any,
    *,
    current_joints: Any,
    current_flange_pose: Any,
    current_tcp_pose: Any,
) -> _MeasuredKinematicFrame:
    current_q = finite_joints(current_joints, name="current joints")
    try:
        nominal_current_flange = np.asarray(chain.forward(current_q), dtype=float)
    except Exception as exc:
        raise CartesianHomeError(
            f"Flexiv FK failed: {type(exc).__name__}: {exc}"
        ) from exc
    matrix_to_flexiv_pose(nominal_current_flange, name="current nominal flange")
    measured_world_flange = flexiv_pose_to_matrix(
        current_flange_pose, name="current measured flange pose"
    )
    measured_world_tcp = flexiv_pose_to_matrix(
        current_tcp_pose, name="current measured TCP pose"
    )
    try:
        nominal_base_from_world = np.linalg.inv(
            measured_world_flange @ np.linalg.inv(nominal_current_flange)
        )
        world_from_nominal_base = np.linalg.inv(nominal_base_from_world)
        flange_to_tcp = np.linalg.inv(measured_world_flange) @ measured_world_tcp
    except np.linalg.LinAlgError as exc:
        raise CartesianHomeError("Measured Flexiv kinematic frame is singular") from exc
    matrix_to_flexiv_pose(world_from_nominal_base, name="measured world anchor")
    matrix_to_flexiv_pose(flange_to_tcp, name="active flange-to-TCP transform")
    return _MeasuredKinematicFrame(
        world_from_nominal_base=world_from_nominal_base,
        nominal_base_from_world=nominal_base_from_world,
        flange_to_tcp=flange_to_tcp,
    )


def _world_tcp_for_joints(
    chain: Any,
    frame: _MeasuredKinematicFrame,
    joints: Any,
    *,
    name: str,
) -> np.ndarray:
    normalized_joints = finite_joints(joints, name=f"{name} joints")
    try:
        nominal_flange = np.asarray(chain.forward(normalized_joints), dtype=float)
    except Exception as exc:
        raise CartesianHomeError(
            f"Flexiv FK failed for {name}: {type(exc).__name__}: {exc}"
        ) from exc
    matrix_to_flexiv_pose(nominal_flange, name=f"{name} nominal flange")
    return frame.world_from_nominal_base @ nominal_flange @ frame.flange_to_tcp


def _joint_bounds(
    chain: Any,
    *,
    margin_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        lower = np.asarray(chain.lower_limits, dtype=float) + margin_rad
        upper = np.asarray(chain.upper_limits, dtype=float) - margin_rad
    except Exception as exc:
        raise CartesianHomeError("IK chain does not expose finite joint limits") from exc
    if lower.shape != (JOINT_COUNT,) or upper.shape != (JOINT_COUNT,):
        raise CartesianHomeError("IK chain must expose seven joint limits")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise CartesianHomeError("IK chain joint limits must be finite")
    if np.any(lower >= upper):
        raise CartesianHomeError("IK joint-limit margin removes the feasible range")
    return lower, upper


def _assert_joint_margin(
    chain: Any,
    joints: np.ndarray,
    *,
    name: str,
    margin_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _joint_bounds(chain, margin_rad=margin_rad)
    if np.any(joints < lower) or np.any(joints > upper):
        raise CartesianHomeError(f"{name} falls inside the IK joint-limit margin")
    return lower, upper


def resolve_tcp_target_joints(
    chain: Any,
    *,
    current_joints: Any,
    current_flange_pose: Any,
    current_tcp_pose: Any,
    target_tcp_pose: Any,
    reference_joints: Any,
    initial_joints: Optional[Any] = None,
    parameters: Optional[Any] = None,
    deadline: Optional[float] = None,
) -> np.ndarray:
    """Resolve one measured-world TCP target to a bounded joint solution."""
    frame = _measured_kinematic_frame(
        chain,
        current_joints=current_joints,
        current_flange_pose=current_flange_pose,
        current_tcp_pose=current_tcp_pose,
    )
    target_world_tcp = flexiv_pose_to_matrix(target_tcp_pose, name="target TCP pose")
    nominal_target_tcp = frame.nominal_base_from_world @ target_world_tcp
    try:
        outcome = chain.solve_inverse(
            nominal_target_tcp,
            reference_joints=finite_joints(
                reference_joints, name="target reference joints"
            ),
            initial_joints=finite_joints(
                current_joints if initial_joints is None else initial_joints,
                name="target initial joints",
            ),
            parameters=parameters,
            tool_transform=frame.flange_to_tcp,
            deadline=deadline,
        )
    except Exception as exc:
        raise CartesianHomeError(
            f"TCP target IK failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not outcome.success or outcome.joints is None:
        raise CartesianHomeError(
            "TCP target IK did not converge: "
            f"{getattr(outcome, 'optimizer_message', 'unknown failure')}"
        )
    solved = finite_joints(outcome.joints, name="target IK joints")
    margin_rad = float(getattr(parameters, "joint_limit_margin_rad", 0.0))
    if not math.isfinite(margin_rad) or margin_rad < 0.0:
        raise CartesianHomeError("IK joint-limit margin must be finite and non-negative")
    _assert_joint_margin(
        chain,
        solved,
        name="target IK joints",
        margin_rad=margin_rad,
    )
    return solved


def plan_joint_home_trajectory(
    chain: Any,
    *,
    current_joints: Any,
    target_joints: Any,
    current_flange_pose: Any,
    current_tcp_pose: Any,
    target_tcp_pose: Any,
    waypoint_count: int,
    max_joint_step_rad: float,
    max_joint_delta_change_rad: float,
    max_path_position_deviation_m: float,
    max_path_orientation_deviation_rad: float,
    path_validation_samples: int,
    parameters: Any,
    terminal_requires_tcp: bool = True,
    deadline: Optional[float] = None,
) -> JointHomePlan:
    """Pre-compute a continuous IK path before any joint command is sent.

    The previous IK result is the next optimizer initial guess. A linear
    current-to-Home joint reference biases the redundant arm posture without
    allowing the solver to jump directly to a distant Home branch. The final
    waypoint is the configured Home joint vector exactly.
    """
    if (
        isinstance(waypoint_count, bool)
        or not isinstance(waypoint_count, int)
        or not 1 <= waypoint_count <= MAX_WAYPOINTS
    ):
        raise CartesianHomeError(f"waypoint_count must be in [1, {MAX_WAYPOINTS}]")
    if not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
        raise CartesianHomeError("max_joint_step_rad must be positive")
    if (
        not math.isfinite(max_joint_delta_change_rad)
        or max_joint_delta_change_rad <= 0.0
    ):
        raise CartesianHomeError("max_joint_delta_change_rad must be positive")
    if not isinstance(terminal_requires_tcp, bool):
        raise CartesianHomeError("terminal_requires_tcp must be a boolean")
    if (
        not math.isfinite(max_path_position_deviation_m)
        or max_path_position_deviation_m <= 0.0
    ):
        raise CartesianHomeError("max_path_position_deviation_m must be positive")
    if (
        not math.isfinite(max_path_orientation_deviation_rad)
        or max_path_orientation_deviation_rad <= 0.0
    ):
        raise CartesianHomeError(
            "max_path_orientation_deviation_rad must be positive"
        )
    if (
        isinstance(path_validation_samples, bool)
        or not isinstance(path_validation_samples, int)
        or not 1 <= path_validation_samples <= 32
    ):
        raise CartesianHomeError("path_validation_samples must be in [1, 32]")

    start_q = finite_joints(current_joints, name="current joints")
    target_q = finite_joints(target_joints, name="target joints")
    start_tcp = normalized_flexiv_pose(current_tcp_pose, name="current TCP pose")
    target_tcp = normalized_flexiv_pose(target_tcp_pose, name="target TCP pose")
    margin_rad = float(getattr(parameters, "joint_limit_margin_rad", 0.0))
    if not math.isfinite(margin_rad) or margin_rad < 0.0:
        raise CartesianHomeError("IK joint-limit margin must be finite and non-negative")
    lower_bounds, upper_bounds = _assert_joint_margin(
        chain, start_q, name="current joints", margin_rad=margin_rad
    )
    _assert_joint_margin(chain, target_q, name="target joints", margin_rad=margin_rad)
    frame = _measured_kinematic_frame(
        chain,
        current_joints=start_q,
        current_flange_pose=current_flange_pose,
        current_tcp_pose=start_tcp,
    )
    target_from_joints = matrix_to_flexiv_pose(
        _world_tcp_for_joints(chain, frame, target_q, name="target"),
        name="target TCP from joints",
    )
    target_position_error, target_orientation_error = pose_error(
        target_from_joints, target_tcp
    )
    if terminal_requires_tcp and (
        target_position_error > max_path_position_deviation_m
        or target_orientation_error > max_path_orientation_deviation_rad
    ):
        raise CartesianHomeError(
            "Target joints do not reproduce the requested TCP: "
            f"error=({target_position_error:.4f}m,"
            f"{target_orientation_error:.4f}rad)"
        )

    tcp_waypoints = interpolate_flexiv_poses(
        start_tcp, target_tcp, waypoint_count=waypoint_count
    )
    joint_waypoints: list[tuple[float, ...]] = []
    previous = start_q.copy()
    previous_delta: Optional[np.ndarray] = None
    optimizer_evaluations = 0
    for index, tcp_waypoint in enumerate(tcp_waypoints, start=1):
        fraction = index / waypoint_count
        if index == waypoint_count:
            solved = target_q.copy()
        else:
            reference = start_q + fraction * (target_q - start_q)
            nominal_target_tcp = (
                frame.nominal_base_from_world
                @ flexiv_pose_to_matrix(tcp_waypoint, name=f"TCP waypoint {index}")
            )
            try:
                outcome = chain.solve_inverse(
                    nominal_target_tcp,
                    reference_joints=reference,
                    initial_joints=previous,
                    parameters=parameters,
                    tool_transform=frame.flange_to_tcp,
                    deadline=deadline,
                )
            except Exception as exc:
                raise CartesianHomeError(
                    f"IK failed at waypoint {index}/{waypoint_count}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not outcome.success or outcome.joints is None:
                raise CartesianHomeError(
                    f"IK did not converge at waypoint {index}/{waypoint_count}: "
                    f"{getattr(outcome, 'optimizer_message', 'unknown failure')}"
                )
            solved = finite_joints(
                outcome.joints, name=f"IK waypoint {index} joints"
            )
            optimizer_evaluations += int(getattr(outcome, "optimizer_nfev", 0))
        _assert_joint_margin(
            chain,
            solved,
            name=f"IK waypoint {index} joints",
            margin_rad=margin_rad,
        )
        joint_step = float(np.max(np.abs(solved - previous)))
        if joint_step > max_joint_step_rad + 1e-9:
            raise CartesianHomeError(
                f"IK waypoint {index}/{waypoint_count} jumps {joint_step:.4f}rad, "
                f"above {max_joint_step_rad:.4f}rad"
            )
        joint_delta = solved - previous
        if previous_delta is not None:
            delta_change = float(np.max(np.abs(joint_delta - previous_delta)))
            if delta_change > max_joint_delta_change_rad + 1e-9:
                raise CartesianHomeError(
                    f"IK waypoint {index}/{waypoint_count} changes joint delta by "
                    f"{delta_change:.4f}rad, above "
                    f"{max_joint_delta_change_rad:.4f}rad"
                )
        joint_waypoints.append(tuple(float(value) for value in solved))
        previous_delta = joint_delta
        previous = solved

    max_position_deviation = 0.0
    max_orientation_deviation = 0.0
    segment_start_q = start_q
    segment_start_tcp = start_tcp
    for segment_index, (segment_target_q_raw, segment_target_tcp_raw) in enumerate(
        zip(joint_waypoints, tcp_waypoints),
        start=1,
    ):
        segment_target_q = np.asarray(segment_target_q_raw, dtype=float)
        expected_poses = interpolate_flexiv_poses(
            segment_start_tcp,
            segment_target_tcp_raw,
            waypoint_count=path_validation_samples,
        )
        for sample_index, expected_pose in enumerate(expected_poses, start=1):
            fraction = sample_index / path_validation_samples
            sample_q = segment_start_q + fraction * (
                segment_target_q - segment_start_q
            )
            actual_pose = matrix_to_flexiv_pose(
                _world_tcp_for_joints(
                    chain,
                    frame,
                    sample_q,
                    name=f"segment {segment_index} sample {sample_index}",
                ),
                name=f"segment {segment_index} sampled TCP",
            )
            position_error, orientation_error = pose_error(actual_pose, expected_pose)
            max_position_deviation = max(max_position_deviation, position_error)
            max_orientation_deviation = max(
                max_orientation_deviation, orientation_error
            )
            if (
                position_error > max_path_position_deviation_m
                or orientation_error > max_path_orientation_deviation_rad
            ):
                raise CartesianHomeError(
                    "Joint interpolation leaves the validated TCP corridor at "
                    f"segment {segment_index}/{waypoint_count}, "
                    f"sample {sample_index}/{path_validation_samples}: "
                    f"error=({position_error:.4f}m,{orientation_error:.4f}rad)"
                )
        segment_start_q = segment_target_q
        segment_start_tcp = np.asarray(segment_target_tcp_raw, dtype=float)

    return JointHomePlan(
        start_joints=tuple(float(value) for value in start_q),
        target_joints=tuple(float(value) for value in target_q),
        joint_waypoints=tuple(joint_waypoints),
        tcp_waypoints=tuple(
            tuple(float(value) for value in waypoint) for waypoint in tcp_waypoints
        ),
        max_position_deviation_m=max_position_deviation,
        max_orientation_deviation_rad=max_orientation_deviation,
        optimizer_evaluations=optimizer_evaluations,
        joint_lower_bounds=tuple(float(value) for value in lower_bounds),
        joint_upper_bounds=tuple(float(value) for value in upper_bounds),
        max_command_step_rad=float(max_joint_step_rad),
        max_delta_change_rad=float(max_joint_delta_change_rad),
        terminal_requires_tcp=terminal_requires_tcp,
    )


def resolve_joint_target_tcp(
    chain: Any,
    *,
    current_joints: Any,
    target_joints: Any,
    current_flange_pose: Any,
    current_tcp_pose: Any,
) -> np.ndarray:
    """Resolve the measured-world TCP pose corresponding to target joints.

    The checked-in URDF supplies relative flange kinematics. Current RDK
    ``flange_pose`` anchors that nominal chain in the measured world frame, and
    current RDK ``tcp_pose`` supplies the active flange-to-TCP transform. This
    remains valid when ``tool_name`` is empty and avoids guessing tool geometry.
    """
    target_q = finite_joints(target_joints, name="target joints")
    frame = _measured_kinematic_frame(
        chain,
        current_joints=current_joints,
        current_flange_pose=current_flange_pose,
        current_tcp_pose=current_tcp_pose,
    )
    return matrix_to_flexiv_pose(
        _world_tcp_for_joints(chain, frame, target_q, name="target"),
        name="target measured-world TCP",
    )


__all__ = [
    "CartesianHomeError",
    "JointHomePlan",
    "JOINT_COUNT",
    "MAX_WAYPOINTS",
    "finite_joints",
    "flexiv_pose_to_matrix",
    "interpolate_flexiv_poses",
    "matrix_to_flexiv_pose",
    "normalized_flexiv_pose",
    "plan_joint_home_trajectory",
    "pose_error",
    "quaternion_angular_distance",
    "required_joint_waypoint_count",
    "required_waypoint_count",
    "resolve_joint_target_tcp",
    "resolve_tcp_target_joints",
]
