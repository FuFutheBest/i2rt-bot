"""Hardware-free FK/IK sampling around a configured Flexiv home pose.

This module deliberately has no dependency on Flask, Hydra, Flexiv RDK, or
shared memory.  It parses the small serial chain in the checked-in Flexiv URDF
and delegates nonlinear inverse kinematics to SciPy's mature bounded least
squares solver.  The public sampling function is side-effect free and safe to
run in a worker thread.

Collision checking is intentionally *not* claimed: neither production Python
environment currently contains a mature collision library.  Every result
therefore carries explicit collision metadata saying that it is kinematic-only.
"""

from __future__ import annotations

import dataclasses
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np


JOINT_COUNT = 7
_EXPECTED_VARIANT = {'left': 'Rizon4', 'right': 'Rizon4R'}
_ASSET_DIR = Path(__file__).resolve().parent / 'static' / 'assets' / 'flexiv'


class HomePoseSamplingError(RuntimeError):
    """Base class for deterministic home-pose sampling failures."""


class InvalidSamplingInput(HomePoseSamplingError, ValueError):
    """Raised when a caller supplies an unsafe or malformed parameter."""


class KinematicsModelError(HomePoseSamplingError):
    """Raised when a URDF is malformed or is for the wrong arm side."""


class MissingSolverDependency(HomePoseSamplingError):
    """Raised when IK is requested outside a SciPy-enabled runtime."""


class SamplingCancelled(HomePoseSamplingError):
    """Raised when the caller's cancellation callback requests a stop."""


class _SamplingTimedOut(Exception):
    """Internal signal used to return deterministic partial timeout results."""


@dataclass(frozen=True)
class PoseRecord:
    """JSON-friendly rigid pose expressed in the robot base frame."""

    position_m: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    matrix: Tuple[Tuple[float, float, float, float], ...]

    @classmethod
    def from_matrix(cls, transform: np.ndarray) -> 'PoseRecord':
        matrix = _validated_transform(transform, name='pose')
        return cls(
            position_m=tuple(float(value) for value in matrix[:3, 3]),
            quaternion_xyzw=_matrix_to_quaternion_xyzw(matrix[:3, :3]),
            matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'position_m': list(self.position_m),
            'quaternion_xyzw': list(self.quaternion_xyzw),
            'matrix': [list(row) for row in self.matrix],
        }


@dataclass(frozen=True)
class CollisionCheck:
    """Truthful collision-check capability metadata."""

    performed: bool = False
    safe: Optional[bool] = None
    method: Optional[str] = None
    reason: str = (
        'No collision-capable mature library is available in the configured '
        'runtime; candidates are kinematic-only and are not collision-safe.'
    )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SamplingParameters:
    """Bounded, whitelisted controls for one offline sampling run."""

    preset_name: str = 'conservative'
    candidate_count: int = 6
    attempts: int = 96
    translation_bounds_m: Tuple[float, float, float] = (0.04, 0.04, 0.04)
    translation_radius_m: Optional[float] = 0.05
    orientation_angle_bound_rad: float = math.radians(8.0)
    min_joint_distance_rad: float = 0.08
    max_joint_distance_rad: float = 0.8
    min_candidate_distance_rad: float = 0.08
    joint_limit_margin_rad: float = 0.04
    position_residual_m: float = 0.0015
    orientation_residual_rad: float = math.radians(0.75)
    reference_weight: float = 0.005
    max_optimizer_evaluations: int = 120
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        bounds = tuple(float(value) for value in self.translation_bounds_m)
        object.__setattr__(self, 'translation_bounds_m', bounds)
        if len(bounds) != 3 or not all(math.isfinite(v) and v >= 0.0 for v in bounds):
            raise InvalidSamplingInput('translation_bounds_m must contain three finite non-negative values')
        if not isinstance(self.candidate_count, int) or not 1 <= self.candidate_count <= 12:
            raise InvalidSamplingInput('candidate_count must be an integer in [1, 12]')
        if not isinstance(self.attempts, int) or not self.candidate_count <= self.attempts <= 2048:
            raise InvalidSamplingInput('attempts must be an integer in [candidate_count, 2048]')
        radius = self.translation_radius_m
        if radius is not None and (not math.isfinite(radius) or radius <= 0.0):
            raise InvalidSamplingInput('translation_radius_m must be positive or null')
        _positive_finite(self.orientation_angle_bound_rad, 'orientation_angle_bound_rad', allow_zero=True)
        _positive_finite(self.min_joint_distance_rad, 'min_joint_distance_rad', allow_zero=True)
        _positive_finite(self.max_joint_distance_rad, 'max_joint_distance_rad')
        _positive_finite(self.min_candidate_distance_rad, 'min_candidate_distance_rad', allow_zero=True)
        if self.min_joint_distance_rad > self.max_joint_distance_rad:
            raise InvalidSamplingInput('min_joint_distance_rad cannot exceed max_joint_distance_rad')
        _positive_finite(self.joint_limit_margin_rad, 'joint_limit_margin_rad', allow_zero=True)
        _positive_finite(self.position_residual_m, 'position_residual_m')
        _positive_finite(self.orientation_residual_rad, 'orientation_residual_rad')
        _positive_finite(self.reference_weight, 'reference_weight', allow_zero=True)
        if not isinstance(self.max_optimizer_evaluations, int) or not 8 <= self.max_optimizer_evaluations <= 400:
            raise InvalidSamplingInput('max_optimizer_evaluations must be an integer in [8, 400]')
        if not math.isfinite(self.timeout_seconds) or not 0.05 <= self.timeout_seconds <= 30.0:
            raise InvalidSamplingInput('timeout_seconds must be in [0.05, 30.0]')

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload['translation_bounds_m'] = list(self.translation_bounds_m)
        return payload


_PARAMETER_FIELDS = frozenset(field.name for field in dataclasses.fields(SamplingParameters))


def sampling_preset(name: str = 'conservative') -> SamplingParameters:
    """Return one of three reviewed, resource-bounded sampling presets."""

    normalized = str(name).strip().lower()
    if normalized == 'conservative':
        return SamplingParameters()
    if normalized == 'default':
        return SamplingParameters(
            preset_name='default',
            candidate_count=10,
            attempts=160,
            translation_bounds_m=(0.06, 0.06, 0.06),
            translation_radius_m=0.08,
            orientation_angle_bound_rad=math.radians(12.0),
            max_joint_distance_rad=1.1,
            min_candidate_distance_rad=0.07,
            max_optimizer_evaluations=150,
            timeout_seconds=7.0,
        )
    if normalized == 'wide':
        return SamplingParameters(
            preset_name='wide',
            candidate_count=12,
            attempts=300,
            translation_bounds_m=(0.10, 0.10, 0.10),
            translation_radius_m=0.12,
            orientation_angle_bound_rad=math.radians(20.0),
            max_joint_distance_rad=1.5,
            min_candidate_distance_rad=0.06,
            position_residual_m=0.002,
            orientation_residual_rad=math.radians(1.0),
            max_optimizer_evaluations=190,
            timeout_seconds=10.0,
        )
    raise InvalidSamplingInput(f'unknown sampling preset: {name!r}')


def sampling_parameters_from_mapping(
    values: Optional[Mapping[str, Any]],
    *,
    preset: str = 'conservative',
) -> SamplingParameters:
    """Build parameters from a strict whitelist suitable for an API boundary."""

    base = sampling_preset(preset)
    if values is None:
        return base
    if not isinstance(values, Mapping):
        raise InvalidSamplingInput('sampling parameters must be a mapping')
    unknown = set(values) - (_PARAMETER_FIELDS | {'count'})
    if unknown:
        raise InvalidSamplingInput(f'unknown sampling parameter(s): {sorted(unknown)}')
    overrides = dict(values)
    if 'count' in overrides:
        if 'candidate_count' in overrides:
            raise InvalidSamplingInput('use only one of count or candidate_count')
        overrides['candidate_count'] = overrides.pop('count')
    overrides.setdefault('preset_name', base.preset_name)
    try:
        return dataclasses.replace(base, **overrides)
    except TypeError as exc:
        raise InvalidSamplingInput(str(exc)) from exc


@dataclass(frozen=True)
class IKResult:
    """One bounded IK solve, including independently measured FK residuals."""

    success: bool
    joints: Optional[Tuple[float, ...]]
    flange_pose: Optional[PoseRecord]
    target_frame_pose: Optional[PoseRecord]
    requested_pose: PoseRecord
    position_residual_m: Optional[float]
    orientation_residual_rad: Optional[float]
    optimizer_success: bool
    optimizer_nfev: int
    optimizer_message: str


@dataclass(frozen=True)
class HomePoseCandidate:
    """A finite, limit-checked IK candidate near the configured home pose."""

    joints: Tuple[float, ...]
    flange_pose: PoseRecord
    target_frame_pose: PoseRecord
    requested_pose: PoseRecord
    position_residual_m: float
    orientation_residual_rad: float
    joint_distance_from_home_rad: float
    nearest_candidate_distance_rad: Optional[float]
    optimizer_nfev: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'joints': list(self.joints),
            'flange_pose': self.flange_pose.to_dict(),
            'target_frame_pose': self.target_frame_pose.to_dict(),
            'requested_pose': self.requested_pose.to_dict(),
            'position_residual_m': self.position_residual_m,
            'orientation_residual_rad': self.orientation_residual_rad,
            'joint_distance_from_home_rad': self.joint_distance_from_home_rad,
            'nearest_candidate_distance_rad': self.nearest_candidate_distance_rad,
            'optimizer_nfev': self.optimizer_nfev,
        }


@dataclass(frozen=True)
class SamplingResult:
    """Complete or partial deterministic sampling output."""

    schema_version: int
    side: str
    variant: str
    urdf_file: str
    target_frame: str
    tool_status: str
    seed: int
    status: str
    parameters: SamplingParameters
    home_joints: Tuple[float, ...]
    home_flange_pose: PoseRecord
    home_target_pose: PoseRecord
    candidates: Tuple[HomePoseCandidate, ...]
    attempts: int
    failure_counts: Tuple[Tuple[str, int], ...]
    elapsed_seconds: float
    timed_out: bool
    solver: str
    collision_check: CollisionCheck

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'side': self.side,
            'variant': self.variant,
            'urdf_file': self.urdf_file,
            'target_frame': self.target_frame,
            'tool_status': self.tool_status,
            'seed': self.seed,
            'status': self.status,
            'parameters': self.parameters.to_dict(),
            'home_joints': list(self.home_joints),
            'home_flange_pose': self.home_flange_pose.to_dict(),
            'home_target_pose': self.home_target_pose.to_dict(),
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'attempts': self.attempts,
            'failure_counts': dict(self.failure_counts),
            'elapsed_seconds': self.elapsed_seconds,
            'timed_out': self.timed_out,
            'solver': self.solver,
            'collision_check': self.collision_check.to_dict(),
        }


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: Optional[float]
    upper: Optional[float]


@dataclass(frozen=True)
class FlexivKinematicChain:
    """The base_link-to-flange serial chain parsed from a Flexiv URDF."""

    variant: str
    joints: Tuple[_Joint, ...]
    active_names: Tuple[str, ...]
    lower_limits: Tuple[float, ...]
    upper_limits: Tuple[float, ...]
    source_path: Path

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Path | str,
        *,
        expected_variant: Optional[str] = None,
        base_link: str = 'base_link',
        tip_link: str = 'flange',
    ) -> 'FlexivKinematicChain':
        path = Path(urdf_path).expanduser().resolve()
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise KinematicsModelError(f'cannot parse URDF {path}: {exc}') from exc
        variant = str(root.attrib.get('name') or '').strip()
        if expected_variant and variant != expected_variant:
            raise KinematicsModelError(
                f'URDF variant mismatch: expected {expected_variant}, found {variant or "unnamed"}'
            )

        outgoing: dict[str, list[_Joint]] = defaultdict(list)
        for element in root.findall('joint'):
            parent_node = element.find('parent')
            child_node = element.find('child')
            if parent_node is None or child_node is None:
                continue
            parent = parent_node.attrib.get('link')
            child = child_node.attrib.get('link')
            if not parent or not child:
                continue
            kind = str(element.attrib.get('type') or 'fixed').lower()
            if kind not in {'fixed', 'revolute', 'continuous'}:
                raise KinematicsModelError(f'unsupported joint type {kind!r} in {path.name}')
            origin_node = element.find('origin')
            xyz = _parse_vector(origin_node, 'xyz', (0.0, 0.0, 0.0))
            rpy = _parse_vector(origin_node, 'rpy', (0.0, 0.0, 0.0))
            origin = np.eye(4, dtype=float)
            origin[:3, :3] = _rpy_matrix(rpy)
            origin[:3, 3] = xyz
            axis = _parse_vector(element.find('axis'), 'xyz', (0.0, 0.0, 1.0))
            axis_norm = float(np.linalg.norm(axis))
            if kind != 'fixed' and axis_norm <= 1e-12:
                raise KinematicsModelError(f'zero axis for joint {element.attrib.get("name")}')
            if axis_norm > 0.0:
                axis = axis / axis_norm
            limit_node = element.find('limit')
            lower = _optional_float(limit_node, 'lower')
            upper = _optional_float(limit_node, 'upper')
            if kind == 'continuous':
                lower, upper = -math.pi, math.pi
            if kind == 'revolute' and (lower is None or upper is None or lower >= upper):
                raise KinematicsModelError(f'invalid limits for joint {element.attrib.get("name")}')
            outgoing[parent].append(
                _Joint(
                    name=str(element.attrib.get('name') or child),
                    kind=kind,
                    parent=parent,
                    child=child,
                    origin=origin,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                )
            )

        chain = _find_chain(outgoing, base_link, tip_link)
        if chain is None:
            raise KinematicsModelError(f'no serial chain from {base_link} to {tip_link} in {path.name}')
        active = tuple(joint for joint in chain if joint.kind != 'fixed')
        if len(active) != JOINT_COUNT:
            raise KinematicsModelError(
                f'expected {JOINT_COUNT} active joints from {base_link} to {tip_link}, found {len(active)}'
            )
        return cls(
            variant=variant,
            joints=tuple(chain),
            active_names=tuple(joint.name for joint in active),
            lower_limits=tuple(float(joint.lower) for joint in active),
            upper_limits=tuple(float(joint.upper) for joint in active),
            source_path=path,
        )

    def forward(
        self,
        joint_positions: Sequence[float],
        *,
        tool_transform: Optional[Sequence[Sequence[float]]] = None,
    ) -> np.ndarray:
        """Return base_T_flange or base_T_tool using standard URDF semantics."""

        values = _validated_joints(joint_positions, name='joint_positions')
        transform = np.eye(4, dtype=float)
        active_index = 0
        for joint in self.joints:
            transform = transform @ joint.origin
            if joint.kind != 'fixed':
                motion = np.eye(4, dtype=float)
                motion[:3, :3] = _axis_angle_matrix(joint.axis, values[active_index])
                transform = transform @ motion
                active_index += 1
        if tool_transform is not None:
            transform = transform @ _validated_transform(tool_transform, name='tool_transform')
        return transform

    def solve_inverse(
        self,
        target_transform: Sequence[Sequence[float]],
        *,
        reference_joints: Sequence[float],
        initial_joints: Optional[Sequence[float]] = None,
        parameters: Optional[SamplingParameters] = None,
        tool_transform: Optional[Sequence[Sequence[float]]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        deadline: Optional[float] = None,
    ) -> IKResult:
        """Solve one target with bounded SciPy least-squares IK."""

        params = parameters or sampling_preset('conservative')
        target = _validated_transform(target_transform, name='target_transform')
        reference = _validated_joints(reference_joints, name='reference_joints')
        initial = (
            reference.copy()
            if initial_joints is None
            else _validated_joints(initial_joints, name='initial_joints')
        )
        tool = None if tool_transform is None else _validated_transform(tool_transform, name='tool_transform')
        least_squares, Rotation = _require_scipy()
        control = _RunControl(should_cancel=should_cancel, deadline=deadline)
        lower = np.asarray(self.lower_limits, dtype=float) + params.joint_limit_margin_rad
        upper = np.asarray(self.upper_limits, dtype=float) - params.joint_limit_margin_rad
        if np.any(lower >= upper):
            raise InvalidSamplingInput('joint_limit_margin_rad removes the feasible joint range')
        hard_lower = np.asarray(self.lower_limits, dtype=float)
        hard_upper = np.asarray(self.upper_limits, dtype=float)
        if np.any(reference < hard_lower) or np.any(reference > hard_upper):
            raise InvalidSamplingInput('reference_joints violate URDF joint limits')
        if np.any(reference < lower) or np.any(reference > upper):
            raise InvalidSamplingInput('reference_joints fall inside the requested joint-limit margin')
        if np.any(initial < hard_lower) or np.any(initial > hard_upper):
            raise InvalidSamplingInput('initial_joints violate URDF joint limits')
        if np.any(initial < lower) or np.any(initial > upper):
            raise InvalidSamplingInput('initial_joints fall inside the requested joint-limit margin')
        ranges = upper - lower

        def residual(values: np.ndarray) -> np.ndarray:
            control.check()
            current = self.forward(values, tool_transform=tool)
            position_error = (current[:3, 3] - target[:3, 3]) / params.position_residual_m
            rotation_error = Rotation.from_matrix(target[:3, :3].T @ current[:3, :3]).as_rotvec()
            rotation_error = rotation_error / params.orientation_residual_rad
            reference_error = math.sqrt(params.reference_weight) * (values - reference) / ranges
            return np.concatenate((position_error, rotation_error, reference_error))

        try:
            control.check()
            optimized = least_squares(
                residual,
                initial.copy(),
                bounds=(lower, upper),
                method='trf',
                jac='3-point',
                x_scale='jac',
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=params.max_optimizer_evaluations,
            )
            control.check()
        except SamplingCancelled:
            raise
        except _SamplingTimedOut:
            raise
        except Exception as exc:
            return IKResult(
                success=False,
                joints=None,
                flange_pose=None,
                target_frame_pose=None,
                requested_pose=PoseRecord.from_matrix(target),
                position_residual_m=None,
                orientation_residual_rad=None,
                optimizer_success=False,
                optimizer_nfev=0,
                optimizer_message=f'{type(exc).__name__}: {exc}',
            )

        joints = np.asarray(optimized.x, dtype=float)
        if joints.shape != (JOINT_COUNT,) or not np.all(np.isfinite(joints)):
            return IKResult(
                success=False,
                joints=None,
                flange_pose=None,
                target_frame_pose=None,
                requested_pose=PoseRecord.from_matrix(target),
                position_residual_m=None,
                orientation_residual_rad=None,
                optimizer_success=bool(optimized.success),
                optimizer_nfev=int(optimized.nfev),
                optimizer_message='optimizer returned non-finite joints',
            )
        flange_pose = self.forward(joints)
        target_pose = self.forward(joints, tool_transform=tool)
        position_residual = float(np.linalg.norm(target_pose[:3, 3] - target[:3, 3]))
        orientation_residual = float(
            np.linalg.norm(Rotation.from_matrix(target[:3, :3].T @ target_pose[:3, :3]).as_rotvec())
        )
        inside_limits = bool(np.all(joints >= lower - 1e-10) and np.all(joints <= upper + 1e-10))
        success = bool(
            inside_limits
            and position_residual <= params.position_residual_m
            and orientation_residual <= params.orientation_residual_rad
        )
        return IKResult(
            success=success,
            joints=tuple(float(value) for value in joints),
            flange_pose=PoseRecord.from_matrix(flange_pose),
            target_frame_pose=PoseRecord.from_matrix(target_pose),
            requested_pose=PoseRecord.from_matrix(target),
            position_residual_m=position_residual,
            orientation_residual_rad=orientation_residual,
            optimizer_success=bool(optimized.success),
            optimizer_nfev=int(optimized.nfev),
            optimizer_message=str(optimized.message),
        )


def flexiv_urdf_for_side(side: str, *, asset_dir: Optional[Path | str] = None) -> Path:
    """Resolve the checked-in official Rizon4/Rizon4R URDF for one side."""

    normalized = _normalized_side(side)
    directory = Path(asset_dir).expanduser().resolve() if asset_dir is not None else _ASSET_DIR
    variant = _EXPECTED_VARIANT[normalized]
    path = directory / f'flexiv_{variant}_kinematics.urdf'
    if not path.is_file():
        raise KinematicsModelError(f'{variant} URDF is missing: {path}')
    return path


def sample_home_poses(
    *,
    side: str,
    urdf_path: Path | str,
    home_joints: Sequence[float],
    parameters: Optional[SamplingParameters] = None,
    seed: int = 0,
    tool_transform: Optional[Sequence[Sequence[float]]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> SamplingResult:
    """Generate deterministic nearby IK candidates without touching hardware.

    ``tool_transform`` must be an explicit ``flange_T_tool`` matrix.  If it is
    absent, sampling is correctly described as flange sampling because the
    controller's active tool is unknown to the config.
    """

    started = time.monotonic()
    normalized_side = _normalized_side(side)
    expected_variant = _EXPECTED_VARIANT[normalized_side]
    params = parameters or sampling_preset('conservative')
    if not isinstance(seed, int) or isinstance(seed, bool) or not -(2**63) <= seed < 2**63:
        raise InvalidSamplingInput('seed must be a signed 64-bit integer')
    chain = FlexivKinematicChain.from_urdf(urdf_path, expected_variant=expected_variant)
    home = _validated_joints(home_joints, name='home_joints')
    tool = None if tool_transform is None else _validated_transform(tool_transform, name='tool_transform')
    target_frame = 'flange' if tool is None else 'tcp'
    tool_status = 'controller-active-unknown' if tool is None else 'explicit-config-transform'
    home_flange = chain.forward(home)
    home_target = chain.forward(home, tool_transform=tool)
    deadline = started + params.timeout_seconds
    control = _RunControl(should_cancel=should_cancel, deadline=deadline)
    rng = np.random.default_rng(seed)
    _, Rotation = _require_scipy()
    candidates: list[HomePoseCandidate] = []
    failures: Counter[str] = Counter()
    attempts_used = 0
    timed_out = False

    try:
        for _ in range(params.attempts):
            control.check()
            if len(candidates) >= params.candidate_count:
                break
            attempts_used += 1
            translation = rng.uniform(-1.0, 1.0, size=3) * np.asarray(params.translation_bounds_m)
            if params.translation_radius_m is not None and np.linalg.norm(translation) > params.translation_radius_m:
                failures['translation_radius'] += 1
                continue
            axis = rng.normal(size=3)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-12:
                failures['orientation_sample'] += 1
                continue
            angle = float(rng.uniform(-params.orientation_angle_bound_rad, params.orientation_angle_bound_rad))
            rotation_vector = (axis / axis_norm) * angle
            requested = home_target.copy()
            requested[:3, 3] += translation
            requested[:3, :3] = home_target[:3, :3] @ Rotation.from_rotvec(rotation_vector).as_matrix()
            outcome = chain.solve_inverse(
                requested,
                reference_joints=home,
                parameters=params,
                tool_transform=tool,
                should_cancel=should_cancel,
                deadline=deadline,
            )
            if not outcome.success or outcome.joints is None:
                failures['ik_or_residual'] += 1
                continue
            joints = np.asarray(outcome.joints, dtype=float)
            home_distance = float(np.linalg.norm(joints - home))
            if home_distance < params.min_joint_distance_rad:
                failures['too_close_to_home'] += 1
                continue
            if home_distance > params.max_joint_distance_rad:
                failures['too_far_from_home'] += 1
                continue
            nearest = None
            if candidates:
                nearest = min(
                    float(np.linalg.norm(joints - np.asarray(candidate.joints)))
                    for candidate in candidates
                )
                if nearest < params.min_candidate_distance_rad:
                    failures['duplicate'] += 1
                    continue
            assert outcome.flange_pose is not None
            assert outcome.target_frame_pose is not None
            assert outcome.position_residual_m is not None
            assert outcome.orientation_residual_rad is not None
            candidates.append(
                HomePoseCandidate(
                    joints=outcome.joints,
                    flange_pose=outcome.flange_pose,
                    target_frame_pose=outcome.target_frame_pose,
                    requested_pose=outcome.requested_pose,
                    position_residual_m=outcome.position_residual_m,
                    orientation_residual_rad=outcome.orientation_residual_rad,
                    joint_distance_from_home_rad=home_distance,
                    nearest_candidate_distance_rad=nearest,
                    optimizer_nfev=outcome.optimizer_nfev,
                )
            )
    except SamplingCancelled:
        raise
    except _SamplingTimedOut:
        timed_out = True
        failures['timeout'] += 1

    elapsed = time.monotonic() - started
    if len(candidates) >= params.candidate_count:
        status = 'complete'
    elif timed_out:
        status = 'timeout'
    elif candidates:
        status = 'partial'
    else:
        status = 'failed'
    return SamplingResult(
        schema_version=1,
        side=normalized_side,
        variant=chain.variant,
        urdf_file=chain.source_path.name,
        target_frame=target_frame,
        tool_status=tool_status,
        seed=seed,
        status=status,
        parameters=params,
        home_joints=tuple(float(value) for value in home),
        home_flange_pose=PoseRecord.from_matrix(home_flange),
        home_target_pose=PoseRecord.from_matrix(home_target),
        candidates=tuple(candidates),
        attempts=attempts_used,
        failure_counts=tuple(sorted(failures.items())),
        elapsed_seconds=float(elapsed),
        timed_out=timed_out,
        solver='scipy.optimize.least_squares',
        collision_check=CollisionCheck(),
    )


@dataclass(frozen=True)
class _RunControl:
    should_cancel: Optional[Callable[[], bool]]
    deadline: Optional[float]

    def check(self) -> None:
        if self.should_cancel is not None and bool(self.should_cancel()):
            raise SamplingCancelled('home-pose sampling was cancelled')
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _SamplingTimedOut()


def _require_scipy() -> tuple[Any, Any]:
    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise MissingSolverDependency(
            'SciPy is required for IK sampling; run this worker with the configured '
            'device or OpenPI Python environment'
        ) from exc
    return least_squares, Rotation


def _normalized_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in _EXPECTED_VARIANT:
        raise InvalidSamplingInput("side must be 'left' or 'right'")
    return normalized


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise InvalidSamplingInput(f'{name} must be finite')
    if value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = 'non-negative' if allow_zero else 'positive'
        raise InvalidSamplingInput(f'{name} must be {qualifier}')


def _parse_vector(node: Optional[ET.Element], attribute: str, default: Sequence[float]) -> np.ndarray:
    raw = None if node is None else node.attrib.get(attribute)
    if raw is None:
        values = np.asarray(default, dtype=float)
    else:
        try:
            values = np.asarray([float(part) for part in raw.split()], dtype=float)
        except ValueError as exc:
            raise KinematicsModelError(f'invalid {attribute} vector: {raw!r}') from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise KinematicsModelError(f'{attribute} must contain three finite values')
    return values


def _optional_float(node: Optional[ET.Element], attribute: str) -> Optional[float]:
    if node is None or attribute not in node.attrib:
        return None
    try:
        value = float(node.attrib[attribute])
    except ValueError as exc:
        raise KinematicsModelError(f'invalid {attribute}: {node.attrib[attribute]!r}') from exc
    if not math.isfinite(value):
        raise KinematicsModelError(f'{attribute} must be finite')
    return value


def _find_chain(
    outgoing: Mapping[str, Sequence[_Joint]],
    current: str,
    target: str,
    visited: Optional[frozenset[str]] = None,
) -> Optional[list[_Joint]]:
    if current == target:
        return []
    seen = visited or frozenset()
    if current in seen:
        return None
    next_seen = seen | {current}
    for joint in outgoing.get(current, ()):
        tail = _find_chain(outgoing, joint.child, target, next_seen)
        if tail is not None:
            return [joint, *tail]
    return None


def _validated_joints(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidSamplingInput(f'{name} must contain {JOINT_COUNT} numbers') from exc
    if array.shape != (JOINT_COUNT,) or not np.all(np.isfinite(array)):
        raise InvalidSamplingInput(f'{name} must contain exactly {JOINT_COUNT} finite numbers')
    return array.copy()


def _validated_transform(transform: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(transform, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidSamplingInput(f'{name} must be a numeric 4x4 transform') from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise InvalidSamplingInput(f'{name} must be a finite 4x4 transform')
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise InvalidSamplingInput(f'{name} must have homogeneous last row [0, 0, 0, 1]')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=2e-6
    ):
        raise InvalidSamplingInput(f'{name} rotation must be orthonormal with determinant +1')
    return matrix.copy()


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    x, y, z = (float(value) for value in axis)
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ],
        dtype=float,
    )


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a proper rotation matrix to a normalized xyzw quaternion."""

    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


__all__ = [
    'CollisionCheck',
    'FlexivKinematicChain',
    'HomePoseCandidate',
    'HomePoseSamplingError',
    'IKResult',
    'InvalidSamplingInput',
    'KinematicsModelError',
    'MissingSolverDependency',
    'PoseRecord',
    'SamplingCancelled',
    'SamplingParameters',
    'SamplingResult',
    'flexiv_urdf_for_side',
    'sample_home_poses',
    'sampling_parameters_from_mapping',
    'sampling_preset',
]
