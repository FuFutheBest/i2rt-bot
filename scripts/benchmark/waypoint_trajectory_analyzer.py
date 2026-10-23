#!/usr/bin/env python3
"""
Analyze trajectory run records: time alignment, delay scan, and plots.

Reads the JSON produced by ``scripts/benchmark/test_trajectory_planner.py`` (fixed schema).
All timestamps are absolute ``time.perf_counter()`` values from recording.

Delay lag scans use ``--to-compare`` as the reference only. Figures overlay
``recorded_samples`` with ``sparse_waypoint_plan`` and ``dense_command_traj`` when
each has enough points (Cartesian dense is shown as FK TCP poses). Cartesian TCP
quaternions are normalized to Flexiv ``[qw, qx, qy, qz]`` with ``qw >= 0`` before
plotting and delay scans. The figure time axis uses the intersection of those overlays' time spans so
leading/trailing gaps where only one stream exists are omitted; if that
intersection is empty, the delay scan overlap is used instead. The plotted span is
then capped by ``--max-plot-duration`` (default 1000 s) from the window start for
easier inspection of long recordings.

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make project root importable for direct execution: python scripts/xxx.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.analysis.pinocchio_kinematics_solver import PinocchioKinematicsSolver  # noqa: E402
from utils.logger_config import logger  # noqa: E402

TRAJ_RUN_RECORD_KEYS: Tuple[str, ...] = (
    "traj_space",
    "sparse_waypoint_plan",
    "dense_command_traj",
    "recorded_samples",
    "urdf_path",
    "ee_frame",
    "world_frame_transform",
    "tcp_frame_transform",
)

TO_COMPARE_SPARSE = "sparse_waypoint_plan"
TO_COMPARE_DENSE = "dense_command_traj"
TO_COMPARE_DENSE_JOINT = "dense_joint"
TO_COMPARE_CHOICES: Tuple[str, ...] = (TO_COMPARE_SPARSE, TO_COMPARE_DENSE, TO_COMPARE_DENSE_JOINT)

# Overlay plot: iterate (role, matplotlib_style). Same layout for joint vs TCP; legend text differs.
_PLOT_ROLES: Tuple[Tuple[str, str], ...] = (
    ("sample", "line"),
    ("sparse", "sparse"),
    ("dense", "line"),
)
_PLOT_COLORS = {"sample": "#4C72B0", "sparse": "#DD8452", "dense": "#C44E52"}
_PLOT_LABELS_BY_SPACE: Dict[str, Dict[str, str]] = {
    "joint": {
        "sample": "recorded_samples",
        "sparse": "sparse_waypoint_plan",
        "dense": "dense_command_traj",
    },
    "cartesian": {
        "sample": "recorded_samples",
        "sparse": "sparse_waypoint_plan",
        "dense": "dense_command_traj (FK)",
    },
}


TimedSeries7 = Tuple[np.ndarray, np.ndarray]  # (t_abs, values shape (N, 7))


@dataclass(frozen=True)
class JointTrajectoryBundle:
    """Joint-space trajectories: ``sample`` always set; ``sparse`` / ``dense`` optional."""

    sample: TimedSeries7
    sparse: Optional[TimedSeries7]
    dense: Optional[TimedSeries7]

    def delay_reference(self, to_compare: str) -> TimedSeries7:
        """Pick ``(t, q)`` for delay estimation from ``to_compare``."""
        if to_compare == TO_COMPARE_SPARSE:
            if self.sparse is None:
                raise ValueError("sparse_waypoint_plan.joints is empty for joint-space record")
            return self.sparse
        if self.dense is None:
            raise ValueError("dense_command_traj is empty; cannot use to_compare=dense_command_traj")
        return self.dense


@dataclass(frozen=True)
class CartesianTrajectoryBundle:
    """TCP trajectories: ``sample`` and ``sparse`` required for valid records; ``dense`` optional (FK)."""

    sample: TimedSeries7
    sparse: TimedSeries7
    dense: Optional[TimedSeries7]

    def delay_reference(self, to_compare: str) -> TimedSeries7:
        """Pick ``(t, pose)`` for delay estimation from ``to_compare``."""
        if to_compare == TO_COMPARE_SPARSE:
            return self.sparse
        if to_compare == TO_COMPARE_DENSE_JOINT:
            raise ValueError("dense_joint must use joint delay branch, not Cartesian pose reference")
        if self.dense is None:
            raise ValueError("dense_command_traj is empty; cannot use to_compare=dense_command_traj")
        return self.dense


WaypointSeries = List[List[Any]]


def _resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against project root when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_traj_run_record(path: Path) -> Dict[str, Any]:
    """Load and validate the trajectory run record JSON."""
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError("Record root must be a JSON object")
    missing = [key for key in TRAJ_RUN_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f"Record missing keys: {missing}")
    extra = [key for key in record if key not in TRAJ_RUN_RECORD_KEYS]
    if extra:
        raise ValueError(f"Record has unknown keys: {extra}")
    return record


def _try_series_to_arrays(
    series: Any,
    value_dim: int,
    name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return parsed arrays or ``None`` when the series has fewer than two samples."""
    if not isinstance(series, list) or len(series) < 2:
        return None
    return _series_to_arrays(series, value_dim, name)


def _series_to_arrays(series: WaypointSeries, value_dim: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ``[[t, values], ...]`` to sorted numpy arrays."""
    if not isinstance(series, list) or len(series) < 2:
        raise ValueError(f"{name} must be a list with at least two samples")
    times: List[float] = []
    values: List[np.ndarray] = []
    prev_t = -math.inf
    for idx, item in enumerate(series):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name}[{idx}] must be [time, values]")
        t = float(item[0])
        v = np.asarray(item[1], dtype=np.float64).reshape(-1)
        if int(v.size) != value_dim:
            raise ValueError(f"{name}[{idx}] expected length {value_dim}, got {int(v.size)}")
        if t <= prev_t:
            raise ValueError(f"{name} times must be strictly increasing (idx={idx})")
        prev_t = t
        times.append(t)
        values.append(v)
    t_arr = np.asarray(times, dtype=np.float64)
    v_arr = np.stack(values, axis=0)
    order = np.argsort(t_arr)
    return t_arr[order], v_arr[order]


def _common_time_window(ref_t: np.ndarray, sample_t: np.ndarray) -> Tuple[float, float]:
    """
    Return the intersection of the reference and sample time domains.

    Assumes both series are sorted by time. Only timestamps where *both* streams exist
    can be used for lag estimation.
    """
    if ref_t.size < 2 or sample_t.size < 2:
        raise ValueError("Reference and samples need at least two timestamps")
    t_lo = max(float(ref_t[0]), float(sample_t[0]))
    t_hi = min(float(ref_t[-1]), float(sample_t[-1]))
    if t_hi <= t_lo + 1e-9:
        raise ValueError(
            f"No overlapping time window between reference [{ref_t[0]}, {ref_t[-1]}] "
            f"and samples [{sample_t[0]}, {sample_t[-1]}]"
        )
    return t_lo, t_hi


def _prepare_delay_scan_window(
    ref_t: np.ndarray,
    ref_v: np.ndarray,
    sample_t: np.ndarray,
    sample_v: np.ndarray,
    max_lag_s: float,
    step_s: float,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Clip both trajectories to their overlap and build the lag grid.

    Returns clipped arrays plus ``ref_start``/``ref_end`` so each lag trial can test
    whether ``sample_time - lag`` still lands inside the (clipped) reference timeline.
    """
    if max_lag_s <= 0.0 or step_s <= 0.0:
        raise ValueError("max_lag_s and step_s must be positive")
    t_lo, t_hi = _common_time_window(ref_t, sample_t)
    ref_t_w, ref_v_w = _clip_series_to_window(
    ref_t,
    ref_v,
    t_lo,
    t_hi,
)

    sample_t_w, sample_v_w = _clip_series_to_window(
        sample_t,
        sample_v,
        t_lo,
        t_hi,
    )
    lag_values = np.arange(-max_lag_s, max_lag_s + 0.5 * step_s, step_s, dtype=np.float64)
    ref_start = float(ref_t_w[0])
    ref_end = float(ref_t_w[-1])
    return t_lo, t_hi, ref_t_w, ref_v_w, sample_t_w, sample_v_w, lag_values, ref_start, ref_end


def _clip_series_to_window(
    t: np.ndarray,
    v: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip a trajectory to [t_lo, t_hi], inserting interpolated
    boundary samples so every clipped series has identical
    start/end timestamps.
    """
    if t_hi <= t_lo:
        raise ValueError("invalid clip window")

    if t_lo < t[0] or t_hi > t[-1]:
        raise ValueError(
            f"clip window [{t_lo}, {t_hi}] "
            f"outside [{t[0]}, {t[-1]}]"
        )

    mask = (t >= t_lo) & (t <= t_hi)

    t_mid = t[mask]
    v_mid = v[mask]

    return (
        np.asarray(t_mid),
        np.stack(v_mid),
    )


def _interp_joint_array_at_times(
    query_t: np.ndarray,
    ref_t: np.ndarray,
    ref_q: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of joint rows at ``query_t``."""
    query = np.asarray(query_t, dtype=np.float64).reshape(-1)
    if query.size == 0:
        return np.empty((0, ref_q.shape[1]), dtype=np.float64)
    out = np.empty((query.size, ref_q.shape[1]), dtype=np.float64)
    for joint_idx in range(ref_q.shape[1]):
        out[:, joint_idx] = np.interp(query, ref_t, ref_q[:, joint_idx])
    return out


def _canonicalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion(s) in Flexiv [qw, qx, qy, qz] order and enforce qw >= 0.

    ``q`` and ``-q`` represent the same rotation; plotting uses a single hemisphere
    so qw channels do not flip sign between streams or samples.
    """
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim == 1:
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            raise ValueError("quaternion norm must be positive")
        out = q / norm
        if out[0] < 0.0:
            out = -out
        return out
    if q.ndim != 2 or int(q.shape[1]) != 4:
        raise ValueError(f"quaternion array must be (4,) or (N, 4), got shape {q.shape}")
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("quaternion norm must be positive for every row")
    out = q / norms
    flip = out[:, 0] < 0.0
    out[flip] *= -1.0
    return out


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Alias for :func:`_canonicalize_quat_wxyz` (used by orientation error metrics)."""
    return _canonicalize_quat_wxyz(quat)


def _canonicalize_tcp_pose_rows(poses: np.ndarray) -> np.ndarray:
    """Normalize TCP pose rows ``[x, y, z, qw, qx, qy, qz]`` and enforce qw >= 0."""
    out = np.asarray(poses, dtype=np.float64).copy()
    if out.ndim != 2 or int(out.shape[1]) != 7:
        raise ValueError(f"TCP pose rows must have shape (N, 7), got {out.shape}")
    out[:, 3:7] = _canonicalize_quat_wxyz(out[:, 3:7])
    return out


def _canonicalize_timed_tcp_series(series: TimedSeries7) -> TimedSeries7:
    """Apply :func:`_canonicalize_tcp_pose_rows` to a ``(t, tcp_pose)`` series."""
    t_arr, pose_arr = series
    return t_arr, _canonicalize_tcp_pose_rows(pose_arr)


def _orientation_error_deg(target_quat_wxyz: np.ndarray, actual_quat_wxyz: np.ndarray) -> np.ndarray:
    """Shortest-angle orientation error (deg) between quaternion rows."""
    target_q = _normalize_quat_wxyz(target_quat_wxyz)
    actual_q = _normalize_quat_wxyz(actual_quat_wxyz)
    dot = np.clip(np.abs(np.sum(target_q * actual_q, axis=1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def _interp_pose_rows_at_times(
    query_t: np.ndarray,
    ref_t: np.ndarray,
    ref_pose: np.ndarray,
) -> np.ndarray:
    """
    Interpolate Flexiv TCP poses [x,y,z,qw,qx,qy,qz] at each ``query_t``.

    Translation: linear. Orientation: ``Slerp`` on scipy rotations built from quaternions
    in xyzw order (columns 4:7 of ref_pose are qx,qy,qz,qw → scipy convention).
    Output quaternions are written back as qw,qx,qy,qz to match ``ref_pose`` layout.
    """
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R

    query = np.asarray(query_t, dtype=np.float64).reshape(-1)
    if query.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    out = np.empty((query.size, 7), dtype=np.float64)
    for i in range(3):
        out[:, i] = np.interp(query, ref_t, ref_pose[:, i])
    # ref_pose stores qw,qx,qy,qz; scipy Rotation.from_quat expects x,y,z,w per row.
    rotations = R.from_quat(ref_pose[:, [4, 5, 6, 3]])
    slerp = Slerp(ref_t, rotations)
    interp_rot = slerp(query)
    quat_xyzw = interp_rot.as_quat()
    out[:, 3] = quat_xyzw[:, 3]
    out[:, 4] = quat_xyzw[:, 0]
    out[:, 5] = quat_xyzw[:, 1]
    out[:, 6] = quat_xyzw[:, 2]
    return _canonicalize_tcp_pose_rows(out)


def estimate_delay_joint(
    ref_t: np.ndarray,
    ref_q: np.ndarray,
    sample_t: np.ndarray,
    sample_q: np.ndarray,
    max_lag_s: float,
    step_s: float,
) -> Dict[str, Any]:
    """
    Scan lag so that recorded(t) matches reference(t - lag).

    For each candidate ``lag_s``, we compare recorded joint angles at ``sample_t`` to
    reference angles interpolated at ``sample_t - lag``. Positive ``lag`` means the
    physical robot appears delayed relative to the reference command timeline.
    """
    if int(ref_q.shape[1]) != int(sample_q.shape[1]):
        raise ValueError("Joint dimension mismatch between reference and samples")

    t_lo, t_hi, ref_t_w, ref_q_w, sample_t_w, sample_q_w, lag_values, ref_start, ref_end = (
        _prepare_delay_scan_window(ref_t, ref_q, sample_t, sample_q, max_lag_s, step_s)
    )
    best: Optional[Dict[str, Any]] = None
    scans: List[Dict[str, Any]] = []

    for lag_s in lag_values:
        # Times on the reference axis that correspond to each recorded sample if ref leads by lag_s.
        planned_query_t = sample_t_w - float(lag_s)
        overlap_mask = (planned_query_t >= ref_start) & (planned_query_t <= ref_end)
        if int(np.sum(overlap_mask)) < 2:
            continue
        planned_q = _interp_joint_array_at_times(
            planned_query_t[overlap_mask],
            ref_t_w,
            ref_q_w,
        )
        err_rad = sample_q_w[overlap_mask] - planned_q
        err_deg = np.rad2deg(err_rad)
        l2_deg = np.linalg.norm(err_deg, axis=1)
        max_abs_deg = np.max(np.abs(err_deg), axis=1)
        item = {
            "lag_s": float(lag_s),
            "sample_count": int(err_deg.shape[0]),
            "mean_l2_deg": float(np.mean(l2_deg)),
            "mean_max_abs_deg": float(np.mean(max_abs_deg)),
            "max_l2_deg": float(np.max(l2_deg)),
            "max_abs_deg": float(np.max(max_abs_deg)),
        }
        scans.append(item)
        if best is None or item["mean_l2_deg"] < best["mean_l2_deg"]:
            best = item

    if best is None:
        raise ValueError("No overlapping samples for joint delay estimation")

    return {
        "definition": "positive lag means recorded(t) best matches reference(t - lag)",
        "max_lag_s": float(max_lag_s),
        "step_s": float(step_s),
        "time_window": {"t_start": t_lo, "t_end": t_hi},
        "best": best,
        "scan_count": int(len(scans)),
    }


def estimate_delay_pose(
    ref_t: np.ndarray,
    ref_pose: np.ndarray,
    sample_t: np.ndarray,
    sample_pose: np.ndarray,
    max_lag_s: float,
    step_s: float,
) -> Dict[str, Any]:
    """
    Delay scan using a scalar score: mean position L2 (mm) + 0.1 * mean orientation error (deg).

    Same lag logic as ``estimate_delay_joint``, but pose interpolation uses ``Slerp`` on quaternions.
    """
    t_lo, t_hi, ref_t_w, ref_pose_w, sample_t_w, sample_pose_w, lag_values, ref_start, ref_end = (
        _prepare_delay_scan_window(ref_t, ref_pose, sample_t, sample_pose, max_lag_s, step_s)
    )
    best: Optional[Dict[str, Any]] = None
    scans: List[Dict[str, Any]] = []

    for lag_s in lag_values:
        planned_query_t = sample_t_w - float(lag_s)
        overlap_mask = (planned_query_t >= ref_start) & (planned_query_t <= ref_end)
        if int(np.sum(overlap_mask)) < 2:
            continue
        planned_pose = _interp_pose_rows_at_times(
            planned_query_t[overlap_mask],
            ref_t_w,
            ref_pose_w,
        )
        actual = sample_pose_w[overlap_mask]
        pos_err = actual[:, :3] - planned_pose[:, :3]
        pos_l2_mm = np.linalg.norm(pos_err, axis=1) * 1000.0
        ori_deg = _orientation_error_deg(planned_pose[:, 3:7], actual[:, 3:7])
        cost = pos_l2_mm + 0.1 * ori_deg
        item = {
            "lag_s": float(lag_s),
            "sample_count": int(pos_l2_mm.shape[0]),
            "mean_position_l2_mm": float(np.mean(pos_l2_mm)),
            "mean_orientation_err_deg": float(np.mean(ori_deg)),
            "mean_score": float(np.mean(cost)),
            "max_position_l2_mm": float(np.max(pos_l2_mm)),
            "max_orientation_err_deg": float(np.max(ori_deg)),
        }
        scans.append(item)
        if best is None or item["mean_score"] < best["mean_score"]:
            best = item

    if best is None:
        raise ValueError("No overlapping samples for pose delay estimation")

    return {
        "definition": "positive lag means recorded(t) best matches reference(t - lag)",
        "max_lag_s": float(max_lag_s),
        "step_s": float(step_s),
        "time_window": {"t_start": t_lo, "t_end": t_hi},
        "best": best,
        "scan_count": int(len(scans)),
    }


def _make_fk_solver(record: Dict[str, Any]) -> PinocchioKinematicsSolver:
    """Build FK solver from fields stored in the run record."""
    urdf = _resolve_project_path(str(record["urdf_path"]))
    tcp_tf = record.get("tcp_frame_transform")
    return PinocchioKinematicsSolver(
        urdf_path=urdf,
        dt=0.01,
        max_iters=100,
        eps=1.0,
        ee_frame=str(record["ee_frame"]),
        world_frame_transform=record.get("world_frame_transform"),
        tcp_frame_transform=tcp_tf,
        damping=1e-6,
    )


def preprocess_cartesian_sparse(
    record: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load sparse TCP pose reference for Cartesian analysis.

    For ``traj_space == "cartesian"`` records, ``sparse_waypoint_plan.joints`` must be empty
    and ``sparse_waypoint_plan.tcp_pose`` must hold at least two ``[t, pose]`` samples.
    """
    sparse = record["sparse_waypoint_plan"]
    joints = sparse.get("joints") or []
    tcp = sparse.get("tcp_pose") or []
    if len(joints) > 0:
        raise ValueError(
            "Cartesian sparse_waypoint_plan.joints must be empty; use tcp_pose only "
            f"(got {len(joints)} joint samples)"
        )
    if len(tcp) < 2:
        raise ValueError(
            f"Cartesian sparse_waypoint_plan.tcp_pose must contain at least two samples, got {len(tcp)}"
        )
    return _series_to_arrays(list(tcp), 7, "sparse_waypoint_plan.tcp_pose")


def preprocess_cartesian_dense(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """FK dense joint commands to TCP poses."""
    dense = record.get("dense_command_traj") or []
    if len(dense) < 2:
        raise ValueError("dense_command_traj is empty")
    return _dense_joint_arrays_to_tcp_pose(record, list(dense))


def _dense_joint_arrays_to_tcp_pose(record: Dict[str, Any], dense: WaypointSeries) -> Tuple[np.ndarray, np.ndarray]:
    """FK ``dense_command_traj`` joint rows to aligned TCP poses."""
    t_arr, q_arr = _series_to_arrays(dense, 7, "dense_command_traj")
    solver = _make_fk_solver(record)
    poses = [np.asarray(solver.compute_fk(q_arr[i]), dtype=np.float64).reshape(7) for i in range(q_arr.shape[0])]
    return t_arr, _canonicalize_tcp_pose_rows(np.stack(poses, axis=0))


def _optional_dense_tcp_from_record(record: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Dense command as TCP poses, or ``None`` if fewer than two dense samples."""
    dense = record.get("dense_command_traj") or []
    if _try_series_to_arrays(list(dense), 7, "dense_command_traj") is None:
        return None
    return preprocess_cartesian_dense(record)


def _optional_sparse_joint_from_record(record: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Sparse plan in joint space for plotting, or ``None`` if not enough points."""
    joints = record["sparse_waypoint_plan"].get("joints") or []
    return _try_series_to_arrays(list(joints), 7, "sparse_waypoint_plan.joints")


def _optional_dense_joint_from_record(record: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Dense joint command trace for plotting, or ``None`` if not enough points."""
    dense = record.get("dense_command_traj") or []
    return _try_series_to_arrays(list(dense), 7, "dense_command_traj")


def _required_recorded_joint_from_record(record: Dict[str, Any]) -> TimedSeries7:
    """Return recorded joint samples for delay estimation."""
    samples = record["recorded_samples"]
    return _series_to_arrays(list(samples["joints"]), 7, "recorded_samples.joints")


def _required_dense_joint_from_record(record: Dict[str, Any]) -> TimedSeries7:
    """Return dense command trace in joint space or raise when unavailable."""
    dense = record.get("dense_command_traj") or []
    if len(dense) < 2:
        raise ValueError("dense_command_traj is empty; cannot use to_compare=dense_joint")
    return _series_to_arrays(list(dense), 7, "dense_command_traj")


def load_joint_trajectory_bundle(record: Dict[str, Any]) -> JointTrajectoryBundle:
    """Load recorded joints plus optional sparse / dense command traces."""
    samples = record["recorded_samples"]
    sample = _series_to_arrays(list(samples["joints"]), 7, "recorded_samples.joints")
    sparse = _optional_sparse_joint_from_record(record)
    dense = _optional_dense_joint_from_record(record)
    return JointTrajectoryBundle(sample=sample, sparse=sparse, dense=dense)


def load_cartesian_trajectory_bundle(record: Dict[str, Any]) -> CartesianTrajectoryBundle:
    """Load recorded TCP, sparse TCP plan, and optional FK of dense joint commands."""
    samples = record["recorded_samples"]
    sample = _canonicalize_timed_tcp_series(
        _series_to_arrays(list(samples["tcp_pose"]), 7, "recorded_samples.tcp_pose")
    )
    sparse = _canonicalize_timed_tcp_series(preprocess_cartesian_sparse(record))
    dense_raw = _optional_dense_tcp_from_record(record)
    dense = _canonicalize_timed_tcp_series(dense_raw) if dense_raw is not None else None
    return CartesianTrajectoryBundle(sample=sample, sparse=sparse, dense=dense)


def preprocess_joint_reference(
    record: Dict[str, Any],
    to_compare: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return reference joint trajectory for joint-space delay estimation."""
    return load_joint_trajectory_bundle(record).delay_reference(to_compare)


def preprocess_cartesian_reference(record: Dict[str, Any], to_compare: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return reference TCP trajectory for Cartesian delay estimation."""
    return load_cartesian_trajectory_bundle(record).delay_reference(to_compare)


def preprocess_joint(record: Dict[str, Any], to_compare: str) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for joint reference (no FK)."""
    return preprocess_joint_reference(record, to_compare)


def _bundle_role_series(bundle: Any, role: str) -> Optional[TimedSeries7]:
    """Read ``sample`` / ``sparse`` / ``dense`` from either trajectory bundle dataclass."""
    if role not in ("sample", "sparse", "dense"):
        raise ValueError(f"unknown plot role: {role!r}")
    return getattr(bundle, role)


def _bundle_overlay_time_intersection(bundle: Any) -> Optional[Tuple[float, float]]:
    """
    Time-axis intersection of all overlay series that exist with at least two samples.

    For each of ``sample`` / ``sparse`` / ``dense``, take the span ``[t[0], t[-1]]``;
    return ``[max(starts), min(ends)]``. Missing or too-short series are skipped.
    """
    intervals: List[Tuple[float, float]] = []
    for role in ("sample", "sparse", "dense"):
        series = _bundle_role_series(bundle, role)
        if series is None:
            continue
        t_arr = series[0]
        if int(t_arr.size) < 2:
            continue
        intervals.append((float(t_arr[0]), float(t_arr[-1])))
    if len(intervals) == 0:
        return None
    t_lo = max(lo for lo, _ in intervals)
    t_hi = min(hi for _, hi in intervals)
    if t_hi <= t_lo + 1e-9:
        return None
    return t_lo, t_hi


def _cap_plot_time_window(
    t_lo: float,
    t_hi: float,
    max_duration_s: Optional[float],
) -> Tuple[float, float, bool]:
    """
    Limit the plot window to ``[t_lo, t_lo + max_duration_s]``.

    Returns ``(t_lo, t_hi, was_capped)``. Non-positive ``max_duration_s`` disables capping.
    """
    if max_duration_s is None or float(max_duration_s) <= 0.0:
        return t_lo, t_hi, False
    span = float(t_hi) - float(t_lo)
    cap = float(max_duration_s)
    if span <= cap + 1e-9:
        return t_lo, t_hi, False
    return t_lo, t_lo + cap, True


def _plot_time_window_overlay(
    bundle: Any,
    delay_t_lo: float,
    delay_t_hi: float,
) -> Tuple[float, float]:
    """
    Figure time window: overlay series time intersection only.

    Falls back to ``[delay_t_lo, delay_t_hi]`` (reference vs sample overlap from the
    delay scan) when the overlay intersection is missing or degenerate.
    """
    overlay = _bundle_overlay_time_intersection(bundle)
    if overlay is None:
        return delay_t_lo, delay_t_hi
    return overlay


def _plot_trajectory_bundle_overlay(
    bundle: Any,
    traj_space: str,
    t_lo: float,
    t_hi: float,
    output_path: Path,
    delay_ref_label: str,
) -> None:
    """
    Single overlay path for joint (rad→deg) or TCP pose (raw) bundles.

    ``t_lo``/``t_hi`` define the overlay time window; all subplots use the same x-axis
    span ``[0, t_hi - t_lo]`` (relative to ``t_lo``) so sparse/sample/dense are aligned
    even when individual streams have no samples at the window edges.
    """
    if traj_space not in ("joint", "cartesian"):
        raise ValueError(f"traj_space must be joint or cartesian, got {traj_space!r}")
    legend_for = _PLOT_LABELS_BY_SPACE[traj_space]
    t0 = t_lo
    _, sample_vals = bundle.sample
    n_dim = int(sample_vals.shape[1])
    y_labels = (
        [f"q{i} [deg]" for i in range(n_dim)]
        if traj_space == "joint"
        else ["x [m]", "y [m]", "z [m]", "qw", "qx", "qy", "qz"]
    )
    fig_height = max(3.0, 2.0 * n_dim) if traj_space == "joint" else 14.0
    fig, axes = plt.subplots(n_dim, 1, figsize=(12, fig_height), sharex=True)
    if n_dim == 1:
        axes = [axes]
    lw_sample = 1.2 if traj_space == "joint" else 1.1

    for channel_idx in range(n_dim):
        ax = axes[channel_idx]
        for role, plot_kind in _PLOT_ROLES:
            series = _bundle_role_series(bundle, role)
            if series is None:
                continue
            tw, vw = _clip_series_to_window(
            series[0],
            series[1],
            t_lo,
            t_hi,
        )
            y_plot = np.rad2deg(vw[:, channel_idx]) if traj_space == "joint" else vw[:, channel_idx]
            color = _PLOT_COLORS[role]
            label = legend_for[role]
            if plot_kind == "sparse":
                ax.plot(
                    tw - t0,
                    y_plot,
                    "o-",
                    color=color,
                    linewidth=lw_sample,
                    markersize=4,
                    label=label,
                )
            else:
                lw = lw_sample if role == "sample" else 1.0
                alpha = 1.0 if role == "sample" else 0.9
                ax.plot(tw - t0, y_plot, color=color, linewidth=lw, alpha=alpha, label=label)
        ax.set_ylabel(y_labels[channel_idx])
        ax.grid(True, alpha=0.3)
        if channel_idx == 0:
            ax.legend(loc="best")
    span = float(t_hi) - float(t_lo)
    if span > 1e-12:
        for ax in axes:
            ax.set_xlim(0.0, span)
    axes[-1].set_xlabel("t - t_window_start [s]")
    title = "Joint trajectories" if traj_space == "joint" else "TCP pose trajectories"
    fig.suptitle(f"{title} (delay ref: {delay_ref_label})", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _analysis_payload(
    record_path: Path,
    traj_space: str,
    to_compare: str,
    delay: Dict[str, Any],
    output_png: Path,
    plot_time_window: Dict[str, Any],
) -> Dict[str, Any]:
    """JSON-serializable summary written next to the figure."""
    return {
        "record_path": str(record_path),
        "traj_space": traj_space,
        "to_compare": to_compare,
        "delay": delay,
        "plot_time_window": plot_time_window,
        "figure": str(output_png),
    }


def _finalize_plot_time_window(
    bundle: Any,
    delay_t_lo: float,
    delay_t_hi: float,
    max_plot_duration_s: Optional[float],
) -> Tuple[float, float, Dict[str, Any]]:
    """Overlay intersection, optional duration cap, and metadata for JSON output."""
    overlay_lo, overlay_hi = _plot_time_window_overlay(bundle, delay_t_lo, delay_t_hi)
    t_lo, t_hi, capped = _cap_plot_time_window(overlay_lo, overlay_hi, max_plot_duration_s)
    meta: Dict[str, Any] = {
        "t_lo": t_lo,
        "t_hi": t_hi,
        "duration_s": float(t_hi - t_lo),
        "overlay_intersection": {"t_lo": overlay_lo, "t_hi": overlay_hi},
        "max_plot_duration_s": max_plot_duration_s,
        "duration_capped": capped,
    }
    if capped:
        logger.info(
            "Plot window capped to {:.1f} s from t={:.3f} (overlay span was {:.1f} s)",
            float(max_plot_duration_s),
            t_lo,
            overlay_hi - overlay_lo,
        )
    else:
        logger.info(
            "Plot window [{:.3f}, {:.3f}] s (duration {:.1f} s)",
            t_lo,
            t_hi,
            t_hi - t_lo,
        )
    return t_lo, t_hi, meta


def run_analysis(
    record_path: Path,
    output_json: Path,
    output_png: Path,
    delay_max_s: float,
    delay_step_s: float,
    to_compare: str,
    max_plot_duration_s: Optional[float] = 1000.0,
) -> None:
    """Load record, preprocess, align time, estimate delay, plot, and write JSON."""
    if to_compare not in TO_COMPARE_CHOICES:
        raise ValueError(f"to_compare must be one of {TO_COMPARE_CHOICES}")

    record = load_traj_run_record(record_path)
    traj_space = str(record["traj_space"]).strip().lower()
    if traj_space not in {"joint", "cartesian"}:
        raise ValueError(f"Unsupported traj_space: {traj_space!r}")
    if traj_space == "joint" and to_compare == TO_COMPARE_DENSE_JOINT:
        raise ValueError("to_compare=dense_joint is only valid when traj_space='cartesian'")

    if traj_space == "joint":
        bundle = load_joint_trajectory_bundle(record)
        ref_t, ref_q = bundle.delay_reference(to_compare)
        sample_t, sample_q = bundle.sample
        delay = estimate_delay_joint(
            ref_t=ref_t,
            ref_q=ref_q,
            sample_t=sample_t,
            sample_q=sample_q,
            max_lag_s=delay_max_s,
            step_s=delay_step_s,
        )
        delay_t_lo = float(delay["time_window"]["t_start"])
        delay_t_hi = float(delay["time_window"]["t_end"])
        t_lo, t_hi, plot_tw = _finalize_plot_time_window(
            bundle, delay_t_lo, delay_t_hi, max_plot_duration_s
        )
        _plot_trajectory_bundle_overlay(
            bundle=bundle,
            traj_space="joint",
            t_lo=t_lo,
            t_hi=t_hi,
            output_path=output_png,
            delay_ref_label=to_compare,
        )
        payload = _analysis_payload(
            record_path, traj_space, to_compare, delay, output_png, plot_tw
        )
    else:
        bundle = load_cartesian_trajectory_bundle(record)
        if to_compare == TO_COMPARE_DENSE_JOINT:
            # Third compare mode: compute lag in joint space even for Cartesian runs.
            ref_t, ref_q = _required_dense_joint_from_record(record)
            sample_t, sample_q = _required_recorded_joint_from_record(record)
            delay = estimate_delay_joint(
                ref_t=ref_t,
                ref_q=ref_q,
                sample_t=sample_t,
                sample_q=sample_q,
                max_lag_s=delay_max_s,
                step_s=delay_step_s,
            )
        else:
            ref_t, ref_pose = bundle.delay_reference(to_compare)
            sample_t, sample_pose = bundle.sample
            delay = estimate_delay_pose(
                ref_t=ref_t,
                ref_pose=ref_pose,
                sample_t=sample_t,
                sample_pose=sample_pose,
                max_lag_s=delay_max_s,
                step_s=delay_step_s,
            )
        delay_t_lo = float(delay["time_window"]["t_start"])
        delay_t_hi = float(delay["time_window"]["t_end"])
        t_lo, t_hi, plot_tw = _finalize_plot_time_window(
            bundle, delay_t_lo, delay_t_hi, max_plot_duration_s
        )
        _plot_trajectory_bundle_overlay(
            bundle=bundle,
            traj_space="cartesian",
            t_lo=t_lo,
            t_hi=t_hi,
            output_path=output_png,
            delay_ref_label=to_compare,
        )
        payload = _analysis_payload(
            record_path, traj_space, to_compare, delay, output_png, plot_tw
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    logger.info("Saved analysis JSON to {}", output_json)
    logger.info("Saved figure to {}", output_png)
    best = delay["best"]
    logger.info("Best lag: {:.6f} s, metric keys: {}", best["lag_s"], list(best.keys()))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    default_record = Path("logs/traj/trajectory_run_record.json")
    default_out_json = Path("logs/traj/waypoint_trajectory_analysis.json")
    default_out_png = Path("logs/traj/waypoint_trajectory_analysis.png")
    parser = argparse.ArgumentParser(description="Analyze trajectory run record (delay + plot)")
    parser.add_argument(
        "--record",
        type=Path,
        default=default_record,
        help=f"Path to traj_run_record JSON (default: {default_record})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_out_json,
        help=f"Output analysis JSON path (default: {default_out_json})",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=default_out_png,
        help=f"Output figure path (default: {default_out_png})",
    )
    parser.add_argument("--delay-max", type=float, default=2.0, help="Max absolute lag scan [s]")
    parser.add_argument("--delay-step", type=float, default=0.001, help="Lag scan step [s]")
     # "sparse_waypoint_plan" or "dense_command_traj" or "dense_joint"
    parser.add_argument(
        "--to-compare",
        type=str,
        default=TO_COMPARE_DENSE_JOINT,
        choices=list(TO_COMPARE_CHOICES),
        help=(
            "Trajectory used only as the delay-estimation reference "
            "(plots show samples + sparse + dense when available; "
            f"default: {TO_COMPARE_DENSE_JOINT})"
        ),
    )
    parser.add_argument(
        "--max-plot-duration",
        type=float,
        default=1000.0,
        help=(
            "Maximum plot time span [s] from overlay window start (default: 1000). "
            "Set <= 0 to plot the full overlay intersection, while latency analysis is unaffected."
        ),
    )
    return parser.parse_args()

def main() -> None:
    """CLI entry point."""
    args = parse_args()
    max_plot_duration = float(args.max_plot_duration)
    if max_plot_duration <= 0.0:
        max_plot_duration = None

    run_analysis(
        record_path=args.record,
        output_json=args.output_json,
        output_png=args.output_png,
        delay_max_s=float(args.delay_max),
        delay_step_s=float(args.delay_step),
        to_compare=str(args.to_compare),
        max_plot_duration_s=max_plot_duration,
    )


if __name__ == "__main__":
    main()
