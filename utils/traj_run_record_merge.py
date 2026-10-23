#!/usr/bin/env python3
"""
Build a ``traj_run_record`` JSON from JSONL fragments written during a robot run.

Merges ``sparse_waypoint_plan``, ``dense_command_traj``, and ``recorded_samples`` streams
into the same top-level schema as ``scripts/benchmark/test_trajectory_planner.py`` / the analyzer.

Author: Zixi Ying
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger_config import logger

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


def _read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load JSON objects from one JSONL file."""
    if path is None or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _dedupe_strictly_increasing(
    points: List[Tuple[float, List[float]]],
) -> List[List[Any]]:
    """Sort by time and nudge timestamps so the series is strictly increasing."""
    if not points:
        return []
    points_sorted = sorted(points, key=lambda x: x[0])
    out: List[List[Any]] = []
    prev_t = -float("inf")
    for t0, values in points_sorted:
        t = float(t0)
        if t <= prev_t:
            t = prev_t + 1e-9
        out.append([t, values])
        prev_t = t
    return out


def _select_arm_row(row: Dict[str, Any], primary_arm: str) -> bool:
    """Return True if this row should be kept for the merged single-arm view."""
    arm = row.get("arm")
    if arm is None:
        return True
    return str(arm).lower() == str(primary_arm).lower()


def _slice_joint_command(values: List[float], primary_arm: str) -> List[float]:
    """If ``values`` is 14-D (dual), keep the primary arm's 7-D slice."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(arr.size) == 14:
        if str(primary_arm).lower() == "right":
            return arr[7:14].tolist()
        return arr[:7].tolist()
    return arr.tolist()


def merge_traj_run_record_from_jsonl(
    *,
    sparse_jsonl: Optional[Path],
    dense_jsonl: Optional[Path],
    samples_jsonl: Optional[Path],
    traj_space: str,
    urdf_path: str,
    ee_frame: str,
    world_frame_transform: Any,
    tcp_frame_transform: Any,
    primary_arm: str = "left",
) -> Dict[str, Any]:
    """
    Merge three optional JSONL logs into one ``traj_run_record``-shaped dict.

    Dual-arm logs may include an ``arm`` field per row; only ``primary_arm`` rows are kept
    for dense command trace and recorded samples. Sparse joint rows without ``arm`` that
    carry 14-D vectors are split using ``primary_arm``.
    """
    if traj_space not in ("joint", "cartesian"):
        raise ValueError("traj_space must be 'joint' or 'cartesian'")

    sparse_rows = _read_jsonl(sparse_jsonl)
    dense_rows = _read_jsonl(dense_jsonl)
    sample_rows = _read_jsonl(samples_jsonl)

    sparse_pts: List[Tuple[float, List[float]]] = []
    for row in sparse_rows:
        if row.get("record_stream") != "sparse_waypoint_plan":
            continue
        if str(row.get("traj_space", traj_space)) != traj_space:
            continue
        t_perf = float(row["t_perf"])
        if traj_space == "cartesian":
            tcp = row.get("tcp_pose")
            if tcp is None:
                continue
            if not _select_arm_row(row, primary_arm):
                continue
            sparse_pts.append((t_perf, [float(x) for x in np.asarray(tcp, dtype=np.float64).reshape(-1).tolist()]))
        else:
            joints = row.get("joints")
            if joints is None:
                continue
            if not _select_arm_row(row, primary_arm):
                continue
            jlist = _slice_joint_command(list(joints), primary_arm)
            sparse_pts.append((t_perf, jlist))

    if traj_space == "cartesian":
        sparse_dict: Dict[str, Any] = {"joints": [], "tcp_pose": _dedupe_strictly_increasing(sparse_pts)}
    else:
        sparse_dict = {"joints": _dedupe_strictly_increasing(sparse_pts), "tcp_pose": []}

    dense_pts: List[Tuple[float, List[float]]] = []
    for row in dense_rows:
        if row.get("record_stream") != "dense_command_traj":
            continue
        if not _select_arm_row(row, primary_arm):
            continue
        joints = row.get("joints")
        if joints is None:
            continue
        t_perf = float(row.get("t_perf", row.get("t", 0.0)))
        dense_pts.append((t_perf, [float(x) for x in np.asarray(joints, dtype=np.float64).reshape(-1).tolist()]))
    dense_command_traj: List[List[Any]] = _dedupe_strictly_increasing(dense_pts)

    joint_samples: List[Tuple[float, List[float]]] = []
    tcp_samples: List[Tuple[float, List[float]]] = []
    for row in sample_rows:
        if row.get("record_stream") != "recorded_samples":
            continue
        if not _select_arm_row(row, primary_arm):
            continue
        t_perf = float(row["t_perf"])
        jq = row.get("joints")
        tp = row.get("tcp_pose")
        if jq is not None:
            joint_samples.append(
                (t_perf, _slice_joint_command(list(jq), primary_arm))
            )
        if tp is not None:
            tcp_samples.append(
                (t_perf, [float(x) for x in np.asarray(tp, dtype=np.float64).reshape(-1).tolist()])
            )

    recorded_samples = {
        "joints": _dedupe_strictly_increasing(joint_samples),
        "tcp_pose": _dedupe_strictly_increasing(tcp_samples),
    }

    payload: Dict[str, Any] = {
        "traj_space": traj_space,
        "sparse_waypoint_plan": sparse_dict,
        "dense_command_traj": dense_command_traj,
        "recorded_samples": recorded_samples,
        "urdf_path": str(urdf_path),
        "ee_frame": str(ee_frame),
        "world_frame_transform": world_frame_transform,
        "tcp_frame_transform": tcp_frame_transform,
    }
    extra = set(payload.keys()) - set(TRAJ_RUN_RECORD_KEYS)
    if extra:
        raise RuntimeError(f"Unexpected record keys: {sorted(extra)}")
    missing = set(TRAJ_RUN_RECORD_KEYS) - set(payload.keys())
    if missing:
        raise RuntimeError(f"Missing record keys: {sorted(missing)}")
    return payload


def write_traj_run_record_json(record: Dict[str, Any], output_path: Path) -> None:
    """Write ``record`` to ``output_path`` with stable JSON formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Wrote traj_run_record to {}", output_path)


def merge_jsonl_files_to_traj_run_record(
    *,
    sparse_jsonl: Optional[Path],
    dense_jsonl: Optional[Path],
    samples_jsonl: Optional[Path],
    output_json: Path,
    traj_space: str,
    urdf_path: str,
    ee_frame: str,
    world_frame_transform: Any = None,
    tcp_frame_transform: Any = None,
    primary_arm: str = "left",
) -> None:
    """Merge JSONL inputs and write a single ``traj_run_record`` JSON file."""
    record = merge_traj_run_record_from_jsonl(
        sparse_jsonl=sparse_jsonl,
        dense_jsonl=dense_jsonl,
        samples_jsonl=samples_jsonl,
        traj_space=traj_space,
        urdf_path=urdf_path,
        ee_frame=ee_frame,
        world_frame_transform=world_frame_transform,
        tcp_frame_transform=tcp_frame_transform,
        primary_arm=primary_arm,
    )
    write_traj_run_record_json(record, output_json)
