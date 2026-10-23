#!/usr/bin/env python3
"""
Rizon dual-arm Cartesian planner configuration helpers.

Parses ``cartesian_planner`` YAML blocks and builds kwargs for
``start_cartesian_position_planner``. Trajectory JSONL recording lives in
``utils.rizon_traj_run_recording``.

Author: Zixi Ying
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_DUAL_ARM_PLANNER_URDF = "assets/robot/rizon/flexiv_Rizon4R_kinematics.urdf"


def _subsection(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = cfg.get(name)
    return dict(section) if isinstance(section, dict) else {}


def _dict_value_for_arm(mapping: Dict[str, Any], arm: str) -> Any:
    """Return ``mapping[arm]`` with case-insensitive keys; then ``both``; else ``None``."""
    arm_key = str(arm).lower()
    if arm_key in mapping:
        return mapping[arm_key]
    for key, val in mapping.items():
        if str(key).lower() == arm_key:
            return val
    return mapping.get("both")


def planner_utilized(planner_cfg: Optional[Dict[str, Any]]) -> bool:
    """Whether ``cartesian_planner.utilized`` is enabled."""
    if not planner_cfg:
        return False
    return bool(planner_cfg.get("utilized", False))


def resolve_urdf_for_arm(planner_cfg: Dict[str, Any], arm: str) -> str:
    """Resolve per-arm URDF from ``cartesian_planner.urdf_path`` (string or left/right/both dict)."""
    value: Any = planner_cfg.get("urdf_path")
    if isinstance(value, dict):
        path = _dict_value_for_arm(value, arm)
        if isinstance(path, str) and path.strip():
            return path.strip()
        other = "right" if str(arm).lower() == "left" else "left"
        fallback = _dict_value_for_arm(value, other)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return DEFAULT_DUAL_ARM_PLANNER_URDF
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_DUAL_ARM_PLANNER_URDF


def resolve_world_tf_for_arm(planner_cfg: Dict[str, Any], arm: str) -> Any:
    """Resolve ``world_frame_transform`` for one arm."""
    value: Any = planner_cfg.get("world_frame_transform")
    if isinstance(value, dict):
        resolved = _dict_value_for_arm(value, arm)
        if resolved is not None:
            return resolved
        other = "right" if str(arm).lower() == "left" else "left"
        return _dict_value_for_arm(value, other)
    return value


def pinocchio_ik_settings(planner_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Settings from ``cartesian_planner.pinocchio_ik`` (flat ``ik_*`` keys still accepted)."""
    ik = _subsection(planner_cfg, "pinocchio_ik")

    def _float(key: str, default: float) -> float:
        if key in ik:
            return float(ik[key])
        return float(planner_cfg.get(key, default))

    return {
        "ik_dt": _float("ik_dt", 0.1),
        "ik_max_iters": int(ik.get("ik_max_iters", planner_cfg.get("ik_max_iters", 1000))),
        "ik_eps": _float("ik_eps", 1e-5),
        "ik_damping": _float("ik_damping", 1e-6),
    }


def traj_smooth_settings(planner_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Trajectory post-processing from ``cartesian_planner.traj_smooth``."""
    smooth = _subsection(planner_cfg, "traj_smooth")
    return {
        "enabled": bool(smooth.get("enabled", False)),
        "upsample_ratio": int(smooth.get("upsample_ratio", 4)),
        "filter_window": int(smooth.get("filter_window", 7)),
    }


def joint_impedance_settings(planner_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Settings from ``cartesian_planner.joint_impedance``."""
    ji = _subsection(planner_cfg, "joint_impedance")
    k_q = ji.get("k_q", ji.get("kq"))
    z_q = ji.get("z_q", ji.get("zq", ji.get("zq_list")))
    return {
        "utilized": bool(ji.get("utilized", False)),
        "k_q": list(k_q) if k_q is not None else None,
        "z_q": list(z_q) if z_q is not None else None,
    }


def build_planner_start_kwargs(
    planner_cfg: Dict[str, Any],
    *,
    arm: str,
    seed_positions: Sequence[float],
    tool: Any,
) -> Dict[str, Any]:
    """Keyword arguments for ``FlexivRobotHandle.start_cartesian_position_planner``."""
    smooth = traj_smooth_settings(planner_cfg)
    ji = joint_impedance_settings(planner_cfg)
    dof = 7
    return {
        "seed_positions": np.asarray(seed_positions, dtype=np.float64).reshape(-1).tolist(),
        "control_dt": float(planner_cfg.get("control_dt", 0.001)),
        "urdf_path": resolve_urdf_for_arm(planner_cfg, arm),
        "max_velocity": planner_cfg.get("max_velocity", [1.0] * dof),
        "max_acceleration": planner_cfg.get("max_acceleration", [2.0] * dof),
        "max_jerk": planner_cfg.get("max_jerk", [3.0] * dof),
        "free_orientation": bool(planner_cfg.get("free_orientation", False)),
        "use_rt": bool(planner_cfg.get("use_rt", True)),
        "world_frame_transform": resolve_world_tf_for_arm(planner_cfg, arm),
        "tool": tool,
        "use_traj_smooth": bool(smooth["enabled"]),
        "traj_smooth_upsample_ratio": int(smooth["upsample_ratio"]),
        "traj_smooth_filter_window": int(smooth["filter_window"]),
        "nrt_vel_ff": bool(planner_cfg.get("nrt_vel_ff", False)),
        "nrt_direct_waypoint": bool(planner_cfg.get("nrt_direct_waypoint", False)),
        "use_joint_impedance": bool(ji["utilized"]),
        "joint_impedance_k_q": ji["k_q"] or [],
        "joint_impedance_z_q": ji["z_q"] or [],
        **pinocchio_ik_settings(planner_cfg),
    }
