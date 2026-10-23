#!/usr/bin/env python3
"""
Run Flexiv waypoint trajectory tests and save a compact ``traj_run_record`` JSON.

The record schema is fixed (see ``TRAJ_RUN_RECORD_KEYS``): only kinematics metadata,
sparse waypoints, dense planner joint commands, and timestamped samples. All sample
times use ``time.perf_counter()`` absolute timestamps.

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
import yaml

# Make project root importable for direct execution: python scripts/xxx.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger_config import logger  # noqa: E402
from scripts.analysis.pinocchio_kinematics_solver import PinocchioKinematicsSolver  # noqa: E402
from utils.time_control import precise_loop_timing, precise_sleep  # noqa: E402

WaypointPlan = List[List[Any]]

# Allowed top-level keys in the saved JSON (no additional keys).
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


def _log_loaded_waypoints(trajectory_space: str, execution_mode: str, waypoint_plan: WaypointPlan) -> None:
    """Emit a single summary line after ``_select_waypoint_plan`` (dry-run or robot path)."""
    final_time_s = float(waypoint_plan[-1][0])
    logger.info(
        "Loaded {} waypoints: trajectory_space={}, execution_mode={}, final_time_s={}",
        len(waypoint_plan),
        trajectory_space,
        execution_mode,
        final_time_s,
    )


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load trajectory YAML config."""
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return cfg


def _resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_flexiv_adapter_class() -> Any:
    """Load FlexivAdapter directly without importing devices package side effects."""
    module_path = PROJECT_ROOT / "devices" / "flexiv_adapter.py"
    spec = importlib.util.spec_from_file_location("flexiv_adapter_standalone", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "FlexivAdapter")


def _normalize_waypoint_plan(raw_plan: Any, value_size: int, name: str) -> WaypointPlan:
    """Validate and normalize [time, value] waypoints."""
    if not isinstance(raw_plan, list) or len(raw_plan) < 2:
        raise ValueError(f"{name} must contain at least two [time, value] waypoints")

    normalized: WaypointPlan = []
    prev_time = -1.0
    for idx, item in enumerate(raw_plan):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name}[{idx}] must be [time, value]")
        time_s = float(item[0])
        value = np.asarray(item[1], dtype=np.float64).reshape(-1)
        if int(value.size) != value_size:
            raise ValueError(f"{name}[{idx}] value length must be {value_size}, got {int(value.size)}")
        if time_s <= prev_time:
            raise ValueError(f"{name} timestamps must be strictly increasing, got {time_s} at idx={idx}")
        prev_time = time_s
        normalized.append([time_s, value.tolist()])

    if abs(float(normalized[0][0])) > 1e-9:
        raise ValueError(f"{name} must start at relative time 0")
    return normalized


def _flexiv_pose_wxyz_to_matrix(pose: Sequence[float]) -> List[List[float]]:
    """
    Convert Flexiv pose [x,y,z,qw,qx,qy,qz] to a row-major 4x4 transform.

    ``scipy.spatial.transform.Rotation.from_quat`` expects (x, y, z, w); Flexiv stores w last in
    the pose vector, hence the reorder ``[qx, qy, qz, qw]`` when calling ``from_quat``.
    """
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if int(pose_arr.size) != 7:
        raise ValueError(f"Flexiv pose must have 7 values, got {int(pose_arr.size)}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _flexiv_wxyz_pose_to_rotation(pose_arr).as_matrix()
    matrix[:3, 3] = pose_arr[:3]
    return matrix.tolist()


def _flexiv_wxyz_pose_to_rotation(pose_wxyz: np.ndarray) -> SciRotation:
    """Build ``SciRotation`` from Flexiv pose ``[x,y,z,qw,qx,qy,qz]`` (translation ignored)."""
    p = np.asarray(pose_wxyz, dtype=np.float64).reshape(-1)
    if int(p.size) != 7:
        raise ValueError(f"Flexiv pose must have 7 values, got {int(p.size)}")
    return SciRotation.from_quat([p[4], p[5], p[6], p[3]])


def _direct_waypoint_segment_dt_s(waypoint_plan: WaypointPlan, index: int) -> float:
    """Time step (s) associated with waypoint ``index`` for discrete feedforward scaling."""
    if index <= 0:
        return max(1.0e-6, float(waypoint_plan[1][0]) - float(waypoint_plan[0][0]))
    return max(1.0e-6, float(waypoint_plan[index][0]) - float(waypoint_plan[index - 1][0]))


def _cartesian_velocity_feedforward_wxyz(
    pose_target: Sequence[float],
    pose_current: Sequence[float],
    dt_s: float,
) -> List[float]:
    """
    TCP velocity in world frame for ``SendCartesianMotionForce``: linear (target - current) / dt
    and angular part from the relative rotation (scipy ``as_rotvec``) / dt.
    """
    tgt = np.asarray(pose_target, dtype=np.float64).reshape(-1)
    cur = np.asarray(pose_current, dtype=np.float64).reshape(-1)
    if int(tgt.size) != 7 or int(cur.size) != 7:
        raise ValueError("pose_target and pose_current must each have 7 values")
    inv_dt = 1.0 / float(dt_s)
    v_lin = (tgt[:3] - cur[:3]) * inv_dt
    r_rel = _flexiv_wxyz_pose_to_rotation(tgt) * _flexiv_wxyz_pose_to_rotation(cur).inv()
    w_ang = np.asarray(r_rel.as_rotvec(), dtype=np.float64).reshape(3) * inv_dt
    out = np.concatenate([v_lin, w_ang], axis=0)
    return [float(v) for v in out]


def _periodic_wave_local_times(freq: float, period_count: int, n_waypoints_per_period: int) -> List[float]:
    """Return local waypoint times for one sinusoidal segment."""
    if n_waypoints_per_period == -1:
        return [
            float((2 * k + 1) / (4.0 * float(freq)))
            for k in range(2 * period_count)
        ]
    if n_waypoints_per_period <= 0:
        raise ValueError("waypoint_plan.periodic_wave.n_waypoints_per_period must be -1 or positive")

    period_s = 1.0 / float(freq)
    local_times: List[float] = []
    for period_idx in range(period_count):
        period_start = float(period_idx) * period_s
        for sample_idx in range(1, n_waypoints_per_period + 1):
            local_times.append(period_start + period_s * float(sample_idx) / float(n_waypoints_per_period))
    return local_times


def _periodic_wave_frequency_amplitude_blocks(
    wave_cfg: Dict[str, Any],
    *,
    joint_space: bool,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Parse shared periodic_wave fields; joint_space selects amplitude units check."""
    freq_arr = np.asarray(wave_cfg.get("frequency", []), dtype=np.float64).reshape(-1)
    amp_arr = np.asarray(wave_cfg.get("amplitude", []), dtype=np.float64).reshape(-1)
    period_count = int(wave_cfg.get("period_count", 1))
    n_waypoints_per_period = int(wave_cfg.get("n_waypoints_per_period", -1))
    if int(freq_arr.size) == 0 or int(amp_arr.size) == 0:
        raise ValueError("waypoint_plan.periodic_wave.frequency/amplitude must be non-empty")
    if np.any(freq_arr <= 0.0):
        raise ValueError("waypoint_plan.periodic_wave.frequency must be positive")
    if np.any(amp_arr < 0.0):
        raise ValueError("waypoint_plan.periodic_wave.amplitude must be non-negative")
    if period_count <= 0:
        raise ValueError("waypoint_plan.periodic_wave.period_count must be positive")
    _ = joint_space  # reserved for future stricter validation
    return freq_arr, amp_arr, period_count, n_waypoints_per_period


def _build_periodic_wave_waypoint_plan(
    trajectory_space: str,
    wave_cfg: Dict[str, Any],
    cartesian_solver: Optional[PinocchioKinematicsSolver],
) -> WaypointPlan:
    """Generate joint- or Cartesian-space sinusoidal waypoints from periodic_wave config."""
    q0 = wave_cfg.get("q0")
    if not isinstance(q0, list):
        raise ValueError("waypoint_plan.periodic_wave.q0 must be provided as a list")
    first_q = np.asarray(q0, dtype=np.float64).reshape(-1)

    freq_arr, amp_arr, period_count, n_waypoints_per_period = _periodic_wave_frequency_amplitude_blocks(
        wave_cfg,
        joint_space=trajectory_space == "joint",
    )

    if trajectory_space == "joint":
        if int(first_q.size) == 0:
            raise ValueError("waypoint_plan.periodic_wave.q0 must be non-empty")
        axes = wave_cfg.get("axes", "joints")
        axes_is_joint = (
            str(axes).strip().lower() in {"joint", "joints"}
            if isinstance(axes, str)
            else isinstance(axes, list)
            and len(axes) == 1
            and str(axes[0]).strip().lower() in {"joint", "joints"}
        )
        if not axes_is_joint:
            raise ValueError(f"joint periodic_wave requires axes='joints', got axes={axes!r}")
        amp_use = np.deg2rad(amp_arr)

        def append_segment_values(sine_val: float, amplitude: float) -> np.ndarray:
            return first_q + float(amplitude) * sine_val * np.ones_like(first_q)

        center_traj = first_q
    elif trajectory_space == "cartesian":
        if int(first_q.size) != 7:
            raise ValueError(f"cartesian periodic_wave.q0 must have 7 joint values, got {int(first_q.size)}")
        if cartesian_solver is None:
            raise RuntimeError("cartesian_solver is required for cartesian periodic_wave")
        axes = wave_cfg.get("axes", ["x", "y", "z"])
        if not isinstance(axes, list):
            raise ValueError("cartesian periodic_wave.axes must be a list")
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        axis_indices: List[int] = []
        for axis in axes:
            axis_key = str(axis).strip().lower()
            if axis_key not in axis_to_index:
                raise ValueError(f"Unsupported cartesian periodic_wave axis: {axis}")
            axis_indices.append(axis_to_index[axis_key])
        center_traj = np.asarray(cartesian_solver.compute_fk(first_q), dtype=np.float64).reshape(-1)
        if int(center_traj.size) != 7:
            raise ValueError("FK center pose must have 7 values")
        amp_use = amp_arr

        def append_segment_values(sine_val: float, amplitude: float) -> np.ndarray:
            pose_val = center_traj.copy()
            for axis_index in axis_indices:
                pose_val[axis_index] = center_traj[axis_index] + float(amplitude) * sine_val
            return pose_val

    else:
        raise ValueError("trajectory_space must be either 'joint' or 'cartesian'")

    waypoints: WaypointPlan = [[0.0, center_traj.tolist()]]
    time_offset = 0.0
    for freq in freq_arr.tolist():
        for amplitude in amp_use.tolist():
            for local_t in _periodic_wave_local_times(float(freq), period_count, n_waypoints_per_period):
                sine_val = float(np.sin(2.0 * np.pi * float(freq) * local_t))
                waypoints.append([time_offset + local_t, append_segment_values(sine_val, float(amplitude)).tolist()])
            # Next (freq, amplitude) block continues after full duration of this frequency segment.
            time_offset += float(period_count) / float(freq)
    return waypoints


def _select_waypoint_plan(
    cfg: Dict[str, Any],
    cartesian_solver: Optional[PinocchioKinematicsSolver],
) -> Tuple[str, WaypointPlan]:
    """Select joint or Cartesian waypoint list from the shared YAML config."""
    trajectory_space = str(cfg.get("trajectory_space", "joint")).strip().lower()
    waypoint_cfg = cfg.get("waypoint_plan")
    if not isinstance(waypoint_cfg, dict):
        raise ValueError("Config key waypoint_plan must be a mapping")

    periodic_cfg = waypoint_cfg.get("periodic_wave", {})
    if not isinstance(periodic_cfg, dict):
        raise ValueError("waypoint_plan.periodic_wave must be a mapping")
    periodic_enabled = bool(periodic_cfg.get("enabled", False))

    if trajectory_space == "joint":
        if periodic_enabled:
            return trajectory_space, _build_periodic_wave_waypoint_plan("joint", periodic_cfg, None)
        return trajectory_space, _normalize_waypoint_plan(
            waypoint_cfg.get("manual_joints"),
            value_size=7,
            name="waypoint_plan.manual_joints",
        )
    if trajectory_space == "cartesian":
        if periodic_enabled:
            return trajectory_space, _build_periodic_wave_waypoint_plan(
                "cartesian",
                periodic_cfg,
                cartesian_solver,
            )
        return trajectory_space, _normalize_waypoint_plan(
            waypoint_cfg.get("manual_poses_wxyz"),
            value_size=7,
            name="waypoint_plan.manual_poses_wxyz",
        )
    raise ValueError("trajectory_space must be either 'joint' or 'cartesian'")


def _make_robot(cfg: Dict[str, Any], backend: str) -> Any:
    """Create, clear fault, enable, and wait for a single Flexiv robot."""
    robot_cfg = dict(cfg.get("robot", {}))
    robot_sn = robot_cfg.get("robot_sn")
    if robot_sn is None:
        raise ValueError("Config missing robot.robot_sn")
    if isinstance(robot_sn, list):
        raise NotImplementedError("test_trajectory_planner.py currently supports single-arm robot_sn only")

    adapter_cls = _load_flexiv_adapter_class()
    adapter = adapter_cls(backend=backend)
    robot = adapter.create_robot(str(robot_sn))
    if robot.fault():
        logger.warning("Robot has faults, attempting to clear...")
        if not robot.clear_fault():
            raise RuntimeError("Failed to clear robot faults")
    robot.enable()

    start = time.time()
    timeout_s = 30.0
    while not robot.is_operational():
        if time.time() - start > timeout_s:
            raise RuntimeError(f"Timeout waiting robot operational after {timeout_s} seconds")
        precise_sleep(0.5, time_func=time.perf_counter)
    return adapter, robot


def _as_float_list(values: Optional[Sequence[float]]) -> Optional[List[float]]:
    """Return None or a flattened float list."""
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64).reshape(-1).tolist()


def _get_motion_limits(cfg: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """Read optional per-joint Ruckig limits from config."""
    return (
        _as_float_list(cfg.get("max_velocity")),
        _as_float_list(cfg.get("max_acceleration")),
        _as_float_list(cfg.get("max_jerk")),
    )


def _read_tool_tcp_frame_transform(tool: Any) -> List[List[float]]:
    """Read current tool TCP transform from robot and convert it to a 4x4 matrix."""
    return _flexiv_pose_wxyz_to_matrix(tool.params().tcp_location)


def _cartesian_ik_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of cartesian_ik settings."""
    return dict(cfg.get("cartesian_ik", {}))


def _require_urdf_ee_frame(ik_cfg: Dict[str, Any]) -> str:
    """Read required URDF end-effector frame name."""
    frame = str(ik_cfg.get("urdf_ee_frame", "")).strip()
    if not frame:
        raise ValueError("cartesian_ik.urdf_ee_frame must be set in the YAML config")
    return frame


def _create_kinematics_solver(ik_cfg: Dict[str, Any]) -> PinocchioKinematicsSolver:
    """Create PinocchioKinematicsSolver for Cartesian IK / FK."""
    urdf_path = _resolve_project_path(str(ik_cfg.get("urdf_path", "Rizon4R-062070.urdf")))
    return PinocchioKinematicsSolver(
        urdf_path=urdf_path,
        dt=float(ik_cfg.get("dt", 0.1)),
        max_iters=int(ik_cfg.get("max_iters", 1000)),
        eps=float(ik_cfg.get("eps", 1e-5)),
        ee_frame=_require_urdf_ee_frame(ik_cfg),
        world_frame_transform=ik_cfg.get("world_frame_transform"),
        tcp_frame_transform=ik_cfg.get("tcp_frame_transform"),
        damping=float(ik_cfg.get("damping", 1e-6)),
    )


def _kinematics_record_fields(cfg: Dict[str, Any], tcp_runtime: Optional[List[List[float]]]) -> Dict[str, Any]:
    """Build URDF path, ee frame, and transforms for the run record."""
    ik_cfg = _cartesian_ik_dict(cfg)
    urdf_resolved = _resolve_project_path(str(ik_cfg.get("urdf_path", "Rizon4R-062070.urdf")))
    ee_frame = _require_urdf_ee_frame(ik_cfg)
    world_tf = ik_cfg.get("world_frame_transform")
    tcp_tf = tcp_runtime if tcp_runtime is not None else ik_cfg.get("tcp_frame_transform")
    return {
        "urdf_path": str(urdf_resolved),
        "ee_frame": ee_frame,
        "world_frame_transform": world_tf,
        "tcp_frame_transform": tcp_tf,
    }


def _first_joint_for_plan(trajectory_space: str, waypoint_plan: WaypointPlan, cfg: Dict[str, Any]) -> List[float]:
    """Return the joint pose used to move to the first waypoint with NRT command."""
    if trajectory_space == "joint":
        return np.asarray(waypoint_plan[0][1], dtype=np.float64).reshape(-1).tolist()

    ik_cfg = _cartesian_ik_dict(cfg)
    seed = np.asarray(ik_cfg.get("seed_joints"), dtype=np.float64).reshape(-1)
    if int(seed.size) == 0:
        raise ValueError("cartesian_ik.seed_joints is required for Cartesian first-pose preparation")
    solver = _create_kinematics_solver(ik_cfg)
    first_joint = solver.solve_ik(q0=seed, tcp_pose=np.asarray(waypoint_plan[0][1], dtype=np.float64))
    if not solver.last_success:
        raise RuntimeError(f"IK failed for first Cartesian waypoint, error_norm={solver.last_error_norm}")
    return first_joint.reshape(-1).tolist()


def _move_to_first_joint(
    adapter: Any, robot: Any, first_joint: Sequence[float], cfg: Dict[str, Any]
) -> None:
    """Move robot to first waypoint in NRT joint mode before starting the planner worker."""
    max_velocity, max_acceleration, _ = _get_motion_limits(cfg)
    joints = np.asarray(first_joint, dtype=np.float64).reshape(-1).tolist()
    dof = len(joints)
    logger.info("Switch to NRT joint position mode for first-pose move.")
    robot.switch_mode(adapter.mode_joint_position)
    precise_sleep(
        float(cfg.get("first_pose_mode_settle_seconds", 1.0)),
        time_func=time.perf_counter,
    )
    logger.info("Move robot to first waypoint with NRT joint command.")
    robot.send_joint_position(
        positions=joints,
        velocities=[0.0] * dof,
        max_vel=max_velocity or [1.0] * dof,
        max_acc=max_acceleration or [2.0] * dof,
    )
    precise_sleep(float(cfg.get("first_pose_hold_seconds", 1.0)), time_func=time.perf_counter)


def _state_recorder_worker(
    robot: Any,
    stop_event: threading.Event,
    samples: List[Dict[str, Any]],
    sample_dt: float,
) -> None:
    """Record robot joint and TCP poses with ``time.perf_counter()`` timestamps."""
    wait_for_next_iteration = precise_loop_timing(sample_dt)
    while not stop_event.is_set():
        now = time.perf_counter()
        states = robot.states()
        samples.append(
            {
                "t_perf": float(now),
                "joints": np.asarray(states.q, dtype=np.float64).reshape(-1).tolist(),
                "tcp_pose": np.asarray(states.tcp_pose, dtype=np.float64).reshape(-1).tolist(),
            }
        )
        wait_for_next_iteration()


def _waypoints_to_absolute_time(rel_plan: WaypointPlan, t0_perf: float) -> List[List[Any]]:
    """
    Map planner-relative times (0 at first waypoint) to absolute ``perf_counter`` timestamps.

    ``t0_perf`` must match how the robot was commanded: planner mode uses the instant before
    ``start_planner``; ``direct_waypoint`` uses ``start_monotonic`` (the timeline anchor for sleeps).
    """
    out: List[List[Any]] = []
    for rel_s, value in rel_plan:
        out.append([float(t0_perf) + float(rel_s), value])
    return out


def _dense_trace_to_list(trace: Any) -> WaypointPlan:
    """Normalize C++ command trace to Python ``[[t, joints], ...]``."""
    if trace is None:
        return []
    rows: WaypointPlan = []
    for item in trace:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            rows.append([float(item[0]), list(item[1])])
    return rows


def _build_recorded_samples(samples: List[Dict[str, Any]]) -> Dict[str, List[List[Any]]]:
    """Split internal recorder rows into parallel joints/tcp_pose series."""
    joints_out: List[List[Any]] = []
    tcp_out: List[List[Any]] = []
    for row in samples:
        t = float(row["t_perf"])
        joints_out.append([t, list(row["joints"])])
        tcp_out.append([t, list(row["tcp_pose"])])
    return {"joints": joints_out, "tcp_pose": tcp_out}


def _build_sparse_waypoint_dict(traj_space: str, rel_plan: WaypointPlan, t0_perf: float) -> Dict[str, List[List[Any]]]:
    """One of ``joints`` / ``tcp_pose`` is populated; the other is an empty list."""
    abs_plan = _waypoints_to_absolute_time(rel_plan, t0_perf)
    if traj_space == "joint":
        return {"joints": abs_plan, "tcp_pose": []}
    if traj_space == "cartesian":
        return {"joints": [], "tcp_pose": abs_plan}
    raise ValueError("traj_space must be 'joint' or 'cartesian'")


def _save_traj_run_record(
    output_path: Path,
    traj_space: str,
    rel_sparse_plan: WaypointPlan,
    dense_rel: WaypointPlan,
    samples: List[Dict[str, Any]],
    kinematics_meta: Dict[str, Any],
    abs_origin_perf: float,
) -> None:
    """
    Write JSON containing only ``TRAJ_RUN_RECORD_KEYS``.

    Time semantics for trajectory inputs:
    - ``rel_sparse_plan``: planner-relative timeline in seconds. First waypoint time must be 0.0,
      and each ``t`` means offset from the segment origin (not wall clock). It is converted to
      absolute ``perf_counter`` time via ``abs_origin_perf + t`` before saving.
    - ``dense_rel``: planner command-trace timeline in seconds, also relative to the same segment
      origin as ``rel_sparse_plan``. It is converted with the same ``abs_origin_perf``.
    - ``samples``: already recorded in absolute ``time.perf_counter()`` timestamps at acquisition
      time (field ``t_perf`` inside each sample row); no extra offset is applied.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "traj_space": traj_space,
        "sparse_waypoint_plan": _build_sparse_waypoint_dict(traj_space, rel_sparse_plan, abs_origin_perf),
        "dense_command_traj": _waypoints_to_absolute_time(dense_rel, abs_origin_perf),
        "recorded_samples": _build_recorded_samples(samples),
        "urdf_path": kinematics_meta["urdf_path"],
        "ee_frame": kinematics_meta["ee_frame"],
        "world_frame_transform": kinematics_meta["world_frame_transform"],
        "tcp_frame_transform": kinematics_meta["tcp_frame_transform"],
    }
    extra = set(payload.keys()) - set(TRAJ_RUN_RECORD_KEYS)
    if extra:
        raise RuntimeError(f"Unexpected record keys: {sorted(extra)}")
    missing = set(TRAJ_RUN_RECORD_KEYS) - set(payload.keys())
    if missing:
        raise RuntimeError(f"Missing record keys: {sorted(missing)}")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    logger.info("Saved traj_run_record to {}", output_path)


# ==================== Core C++ Trajectory Planner Usage ====================
# 1. start_planner(...) starts the C++ planner worker in joint or Cartesian mode.
# 2. add_plans(...) streams timed waypoints into that worker. The first waypoint
#    time must be 0.0, and later timestamps are relative to planner start.


def start_planner(
    robot: Any,
    trajectory_space: str,
    cfg: Dict[str, Any],
    first_joint: Sequence[float],
    tool: Optional[Any] = None,
) -> float:
    """Start the C++ planner worker and return Python-side absolute planner start timestamp."""
    control_dt = float(cfg.get("ruckig_control_dt", 0.003))
    use_rt = bool(cfg.get("rt", True))
    max_velocity, max_acceleration, max_jerk = _get_motion_limits(cfg)

    if trajectory_space == "joint":
        planner_start_perf = time.perf_counter()
        robot.start_joint_position_planner(
            control_dt=control_dt,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
            use_rt=use_rt,
        )
        return planner_start_perf

    ik_cfg = _cartesian_ik_dict(cfg)
    urdf_path = _resolve_project_path(str(ik_cfg.get("urdf_path", "Rizon4R-062070.urdf")))
    planner_start_perf = time.perf_counter()
    robot.start_cartesian_position_planner(
        seed_positions=np.asarray(first_joint, dtype=np.float64).reshape(-1).tolist(),
        control_dt=control_dt,
        urdf_path=str(urdf_path),
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
        free_orientation=bool(ik_cfg.get("free_orientation", False)),
        use_rt=use_rt,
        ik_dt=float(ik_cfg.get("dt", 0.1)),
        ik_max_iters=int(ik_cfg.get("max_iters", 1000)),
        ik_eps=float(ik_cfg.get("eps", 1e-5)),
        ik_damping=float(ik_cfg.get("damping", 1e-6)),
        world_frame_transform=ik_cfg.get("world_frame_transform"),
        tool=tool,
    )
    return planner_start_perf


def add_plans(robot: Any, trajectory_space: str, waypoint_plan: WaypointPlan) -> None:
    """Append all timed waypoints to an already-started C++ planner worker."""
    if trajectory_space == "joint":
        for time_s, joints in waypoint_plan:
            robot.add_joint_position_plan(float(time_s), joints)
        return

    for time_s, pose in waypoint_plan:
        robot.add_cartesian_position_plan(float(time_s), pose)


def run_direct_waypoints(
    adapter: Any,
    robot: Any,
    trajectory_space: str,
    waypoint_plan: WaypointPlan,
    cfg: Dict[str, Any],
    start_monotonic: float,
) -> None:
    """Send waypoint commands directly at their timestamps without C++ trajectory planning."""
    if bool(cfg.get("rt", True)):
        logger.warning("execution_mode=direct_waypoint uses NRT send APIs; cfg.rt is ignored.")
    max_velocity, max_acceleration, _ = _get_motion_limits(cfg)
    use_vel_ff = bool(cfg.get("direct_waypoint_velocity_feedforward", False))
    if use_vel_ff:
        logger.info(
            "direct_waypoint_velocity_feedforward=true: using (target-current)/dt as "
            "joint velocities or Cartesian TCP velocity for NRT sends."
        )

    if trajectory_space == "joint":
        for idx, (time_s, joints) in enumerate(waypoint_plan):
            target_time = start_monotonic + float(time_s)
            precise_sleep(max(0.0, target_time - time.perf_counter()), time_func=time.perf_counter)
            joints_list = np.asarray(joints, dtype=np.float64).reshape(-1).tolist()
            dof = len(joints_list)
            if use_vel_ff:
                dt_s = _direct_waypoint_segment_dt_s(waypoint_plan, idx)
                states = robot.states()
                q_cur = np.asarray(states.q, dtype=np.float64).reshape(-1)
                q_tgt = np.asarray(joints_list, dtype=np.float64).reshape(-1)
                if int(q_cur.size) != dof:
                    raise ValueError(
                        f"states.q length {int(q_cur.size)} does not match waypoint dof {dof}"
                    )
                velocities = (q_tgt - q_cur) / dt_s
                velocities_list = [float(v) for v in velocities.reshape(-1)]
            else:
                velocities_list = [0.0] * dof
            robot.send_joint_position(
                positions=joints_list,
                velocities=velocities_list,
                max_vel=max_velocity or [1.0] * dof,
                max_acc=max_acceleration or [2.0] * dof,
            )
        return
     # SendCartesianMotionForce requires NRT_CARTESIAN_MOTION_FORCE; first-pose prep leaves NRT_JOINT_POSITION.
    settle_cartesian_s = float(
        cfg.get(
            "direct_cartesian_mode_settle_seconds",
            cfg.get("first_pose_mode_settle_seconds", 1.0),
        )
    )
    logger.info(
        "Switch to NRT Cartesian motion-force mode for direct Cartesian waypoints "
        "(settle {:.3f}s).",
        settle_cartesian_s,
    )
    robot.switch_mode(adapter.mode_cartesian_motion_force)
    precise_sleep(settle_cartesian_s, time_func=time.perf_counter)

    for idx, (time_s, pose) in enumerate(waypoint_plan):
        target_time = start_monotonic + float(time_s)
        precise_sleep(max(0.0, target_time - time.perf_counter()), time_func=time.perf_counter)
        pose_list = np.asarray(pose, dtype=np.float64).reshape(-1).tolist()
        if use_vel_ff:
            dt_s = _direct_waypoint_segment_dt_s(waypoint_plan, idx)
            states = robot.states()
            tcp = np.asarray(states.tcp_pose, dtype=np.float64).reshape(-1)
            vel6 = _cartesian_velocity_feedforward_wxyz(pose_list, tcp, dt_s)
        else:
            vel6 = None
        robot.send_cartesian_motion_force(
            pose=pose_list,
            velocity=vel6,
            max_linear_vel=float(cfg.get("max_linear_vel", 0.5)),
            max_angular_vel=float(cfg.get("max_angular_vel", 1.0)),
            max_linear_acc=float(cfg.get("max_linear_acc", 2.0)),
            max_angular_acc=float(cfg.get("max_angular_acc", 5.0)),
        )


def run(config_path: Path, backend: str, dry_run: bool) -> None:
    """Load config, prepare robot, run planner or direct waypoints, and save traj_run_record."""
    cfg = _load_yaml_config(config_path)
    execution_mode = str(cfg.get("execution_mode", "planner")).strip().lower()
    if execution_mode not in {"planner", "direct_waypoint"}:
        raise ValueError("execution_mode must be either 'planner' or 'direct_waypoint'")
    trajectory_space = str(cfg.get("trajectory_space", "joint")).strip().lower()

    if dry_run:
        cartesian_solver: Optional[PinocchioKinematicsSolver] = None
        if trajectory_space == "cartesian":
            cartesian_solver = _create_kinematics_solver(_cartesian_ik_dict(cfg))
        trajectory_space, waypoint_plan = _select_waypoint_plan(cfg, cartesian_solver)
        _log_loaded_waypoints(trajectory_space, execution_mode, waypoint_plan)
        logger.info("Dry-run enabled. Waypoints validated; skip robot connection.")
        return

    adapter, robot = _make_robot(cfg, backend=backend)
    tool_tcp_frame_transform: Optional[List[List[float]]] = None
    tool: Optional[Any] = None
    cartesian_solver: Optional[PinocchioKinematicsSolver] = None
    if trajectory_space == "cartesian":
        tool = adapter.create_tool(robot)
        tool_tcp_frame_transform = _read_tool_tcp_frame_transform(tool)
        cfg = dict(cfg)
        cartesian_ik_cfg = _cartesian_ik_dict(cfg)
        cartesian_ik_cfg["tcp_frame_transform"] = tool_tcp_frame_transform
        cfg["cartesian_ik"] = cartesian_ik_cfg
        cartesian_solver = _create_kinematics_solver(cartesian_ik_cfg)
        logger.info("Merged active tool TCP into cartesian_ik for planning and record.")

    trajectory_space, waypoint_plan = _select_waypoint_plan(cfg, cartesian_solver)
    _log_loaded_waypoints(trajectory_space, execution_mode, waypoint_plan)

    kinematics_meta = _kinematics_record_fields(cfg, tool_tcp_frame_transform)
    first_joint = _first_joint_for_plan(trajectory_space, waypoint_plan, cfg)
    _move_to_first_joint(adapter, robot, first_joint, cfg)

    recording_cfg = dict(cfg.get("recording", {}))
    recording_enabled = bool(recording_cfg.get("enabled", True))
    samples: List[Dict[str, Any]] = []
    stop_recording = threading.Event()
    recorder_thread: Optional[threading.Thread] = None
    command_trajectory_rel: WaypointPlan = []

    # Relative end time of the sparse plan (seconds from segment t=0); used for planner post-hold.
    final_time_s = float(waypoint_plan[-1][0])

    run_start_delay = max(0.0, float(cfg.get("run_start_delay", 0.0)))
    logger.info("Waiting run_start_delay={}s before starting execution.", run_start_delay)
    precise_sleep(run_start_delay, time_func=time.perf_counter)
    start_monotonic = time.perf_counter()

    if recording_enabled:
        recorder_thread = threading.Thread(
            target=_state_recorder_worker,
            args=(
                robot,
                stop_recording,
                samples,
                max(1e-4, float(recording_cfg.get("sample_dt", cfg.get("ruckig_control_dt", 0.003)))),
            ),
            daemon=True,
        )
        recorder_thread.start()

    planner_start_perf: float = 0.0
    try:
        if execution_mode == "planner":
            # Anchor for converting planner-relative trace/waypoint times to absolute perf_counter in JSON.
            planner_start_perf = start_planner(
                robot=robot,
                trajectory_space=trajectory_space,
                cfg=cfg,
                first_joint=first_joint,
                tool=tool,
            )
            add_plans(robot=robot, trajectory_space=trajectory_space, waypoint_plan=waypoint_plan)
            logger.info("Planner started and all waypoints enqueued.")
        else:
            logger.info("Direct waypoint execution started.")
            # Must match ``start_monotonic`` used inside ``run_direct_waypoints`` for sleep scheduling.
            planner_start_perf = time.perf_counter()
            run_direct_waypoints(
                adapter=adapter,
                robot=robot,
                trajectory_space=trajectory_space,
                waypoint_plan=waypoint_plan,
                cfg=cfg,
                start_monotonic=start_monotonic,
            )
            logger.info("Direct waypoint execution finished.")
        final_hold_seconds = float(cfg.get("final_hold_seconds", 0.0))
        sleep_after_start = final_time_s + final_hold_seconds if execution_mode == "planner" else final_hold_seconds
        precise_sleep(sleep_after_start, time_func=time.perf_counter)
        if execution_mode == "planner":
            logger.info("Planner status: {}", robot.position_planner_status())
    finally:
        if recording_enabled:
            stop_recording.set()
            if recorder_thread is not None:
                recorder_thread.join(timeout=5.0)
        if execution_mode == "planner":
            try:
                command_trajectory_rel = _dense_trace_to_list(robot.position_planner_command_trace())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read planner command trace: {}", exc)
                command_trajectory_rel = []
            try:
                robot.stop_position_planner()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to stop planner cleanly: {}", exc)

    if recording_enabled:
        output_path = _resolve_project_path(str(recording_cfg.get("output_path", "logs/traj/trajectory_run_record.json")))
        # Align JSON absolute times with the timeline used for motion: planner t=0 vs direct NRT sleeps.
        abs_origin_perf = planner_start_perf if execution_mode == "planner" else start_monotonic
        _save_traj_run_record(
            output_path=output_path,
            traj_space=trajectory_space,
            rel_sparse_plan=waypoint_plan,
            dense_rel=command_trajectory_rel,
            samples=samples,
            kinematics_meta=kinematics_meta,
            abs_origin_perf=abs_origin_perf,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run waypoint plan with the C++ Flexiv planner worker")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "trajectory" / "waypoint_plan.yaml",
        help="Path to waypoint planning YAML config",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pybind",
        help="FlexivAdapter backend. C++ planner requires pybind backend.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without connecting to robot.",
    )
    args = parser.parse_args()
    run(config_path=args.config, backend=args.backend, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
