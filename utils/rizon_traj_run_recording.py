#!/usr/bin/env python3
"""
Rizon ``traj_run`` JSONL recording, dense planner dump, and merged JSON helpers.

Keeps ``devices/robot.py`` smaller by owning sinks, merge paths, and background sample loop wiring.

Sparse waypoints and periodic ``recorded_samples`` are logged whenever ``traj_run_recording`` is
enabled and the corresponding JSONL paths are set, independent of the C++ Cartesian planner.
Dense ``dense_command_traj`` is only populated when the dual-arm Cartesian planner runs; otherwise
shutdown writes an empty dense JSONL so merge paths stay consistent. For analysis without dense,
use ``--to-compare sparse_waypoint_plan`` in ``scripts/benchmark/waypoint_trajectory_analyzer.py``.

Author: Zixi Ying
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from utils.flexiv_utils import flexiv_pose_wxyz_to_matrix, flexiv_pose_wxyz_to_xyzw_list
from utils.logger_config import logger
from utils.rizon_traj_planner_utils import (
    _dict_value_for_arm,
    resolve_urdf_for_arm,
    resolve_world_tf_for_arm,
)
from utils.traj_run_stream_logger import TrajRunJsonlSink


def _resolve_traj_path(raw: Any, cwd: Path) -> Optional[Path]:
    if not raw:
        return None
    p = Path(str(raw))
    return p if p.is_absolute() else cwd / p


def resolve_traj_merge_urdf_path(
    traj_run_recording: Dict[str, Any],
    cartesian_planner_cfg: Dict[str, Any],
    primary_arm: str,
) -> str:
    """Merge output URDF: optional override in ``traj_run_recording``, else per-arm ``cartesian_planner``."""
    v: Any = traj_run_recording.get("urdf_path")
    if isinstance(v, str) and v.strip():
        return v.strip()
    if isinstance(v, dict):
        s = _dict_value_for_arm(v, primary_arm)
        if isinstance(s, str) and s.strip():
            return s.strip()
        other = "right" if str(primary_arm).lower() == "left" else "left"
        s2 = _dict_value_for_arm(v, other)
        if isinstance(s2, str) and s2.strip():
            return s2.strip()
    return resolve_urdf_for_arm(cartesian_planner_cfg, primary_arm)


def run_traj_samples_loop(
    sink: TrajRunJsonlSink,
    stop_event: threading.Event,
    interval_s: float,
    poll_rows: Callable[[], Sequence[Dict[str, Any]]],
) -> None:
    """Background loop: call ``poll_rows()`` and append each dict to ``sink`` until ``stop_event``."""
    interval = max(0.05, float(interval_s))
    while True:
        if stop_event.wait(timeout=interval):
            break
        try:
            for row in poll_rows():
                sink.append(row)
        except Exception as exc:
            logger.warning("Trajectory samples logging failed: {}", exc)


class RizonTrajRunRecorder:
    """JSONL sinks + merge metadata for RizonRobot ``traj_run_recording`` config."""

    def __init__(
        self,
        *,
        cartesian_planner_cfg: Dict[str, Any],
        merge_output_path: Optional[Path],
        merge_sparse_path: Optional[Path],
        merge_dense_path: Optional[Path],
        merge_samples_path: Optional[Path],
        merge_traj_space: str,
        merge_urdf_path: Optional[str],
        merge_ee_frame: str,
        merge_primary_arm: str,
        sparse_sink: Optional[TrajRunJsonlSink],
        samples_sink: Optional[TrajRunJsonlSink],
        sample_interval_s: float,
    ) -> None:
        self.cartesian_planner_cfg = cartesian_planner_cfg
        self.merge_output_path = merge_output_path
        self.merge_sparse_path = merge_sparse_path
        self.merge_dense_path = merge_dense_path
        self.merge_samples_path = merge_samples_path
        self.merge_traj_space = merge_traj_space
        self.merge_urdf_path = merge_urdf_path
        self.merge_ee_frame = merge_ee_frame
        self.merge_primary_arm = merge_primary_arm
        self.sparse_sink = sparse_sink
        self.samples_sink = samples_sink
        self.sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._samples_thread: Optional[threading.Thread] = None

    @classmethod
    def maybe_create(
        cls,
        traj_run_recording: Optional[Dict[str, Any]],
        cartesian_planner_cfg: Dict[str, Any],
        control_type: str,
    ) -> Optional["RizonTrajRunRecorder"]:
        if traj_run_recording is None or not isinstance(traj_run_recording, dict):
            return None
        cwd = Path(os.getcwd())

        merge_sparse_path = _resolve_traj_path(
            traj_run_recording.get("sparse_jsonl") or traj_run_recording.get("sparse_jsonl_path"),
            cwd,
        )
        merge_dense_path = _resolve_traj_path(
            traj_run_recording.get("dense_jsonl") or traj_run_recording.get("dense_jsonl_path"),
            cwd,
        )
        merge_samples_path = _resolve_traj_path(
            traj_run_recording.get("samples_jsonl") or traj_run_recording.get("samples_jsonl_path"),
            cwd,
        )
        merge_out = traj_run_recording.get("traj_run_record_json") or traj_run_recording.get(
            "output_traj_run_record_json"
        )
        merge_output_path = _resolve_traj_path(merge_out, cwd) if merge_out else None

        merge_traj_space = (
            "cartesian"
            if control_type in ["eef_abs", "eef_rel"]
            else "joint"
        )
        ts = traj_run_recording.get("traj_space")
        if isinstance(ts, str) and ts in ("joint", "cartesian"):
            merge_traj_space = ts

        merge_primary_arm = "left"
        pa = traj_run_recording.get("primary_arm")
        if isinstance(pa, str) and pa.strip():
            merge_primary_arm = pa.strip().lower()

        merge_urdf_path = resolve_traj_merge_urdf_path(
            traj_run_recording,
            cartesian_planner_cfg,
            merge_primary_arm,
        )

        merge_ee_frame = "flange"
        ee_m = traj_run_recording.get("ee_frame")
        if isinstance(ee_m, str) and ee_m.strip():
            merge_ee_frame = ee_m.strip()

        sample_interval_s = float(traj_run_recording.get("sample_interval_s", 0.15))

        sparse_sink: Optional[TrajRunJsonlSink] = None
        samples_sink: Optional[TrajRunJsonlSink] = None
        rec_enabled = bool(traj_run_recording.get("enabled", True))
        if rec_enabled:
            if merge_sparse_path is not None:
                sparse_sink = TrajRunJsonlSink(merge_sparse_path)
                logger.info("RizonRobot sparse traj_run JSONL enabled: {}", merge_sparse_path)
            if merge_dense_path is not None:
                logger.info(
                    "RizonRobot dense command trace will be written once at shutdown: {}",
                    merge_dense_path,
                )
            if merge_samples_path is not None:
                samples_sink = TrajRunJsonlSink(merge_samples_path)
                logger.info("RizonRobot recorded samples JSONL enabled: {}", merge_samples_path)

        return cls(
            cartesian_planner_cfg=cartesian_planner_cfg,
            merge_output_path=merge_output_path,
            merge_sparse_path=merge_sparse_path,
            merge_dense_path=merge_dense_path,
            merge_samples_path=merge_samples_path,
            merge_traj_space=merge_traj_space,
            merge_urdf_path=merge_urdf_path,
            merge_ee_frame=merge_ee_frame,
            merge_primary_arm=merge_primary_arm,
            sparse_sink=sparse_sink,
            samples_sink=samples_sink,
            sample_interval_s=sample_interval_s,
        )

    def resolve_merge_transforms(
        self,
        *,
        is_dual_arm: bool,
        tool_left: Any,
        tool_right: Any,
        tool_single: Any,
    ) -> Tuple[Any, Any]:
        world_tf: Any = resolve_world_tf_for_arm(
            self.cartesian_planner_cfg,
            self.merge_primary_arm,
        )
        tcp_tf: Any = None
        tool_obj: Any = None
        if is_dual_arm:
            if str(self.merge_primary_arm).lower() == "right":
                tool_obj = tool_right
            else:
                tool_obj = tool_left
        else:
            tool_obj = tool_single
        if tool_obj is not None:
            raw_tool = tool_obj.raw if hasattr(tool_obj, "raw") else tool_obj
            tcp_tf = flexiv_pose_wxyz_to_matrix(raw_tool.params().tcp_location)
        return world_tf, tcp_tf

    def merge_json_if_configured(
        self,
        *,
        is_dual_arm: bool,
        tool_left: Any,
        tool_right: Any,
        tool_single: Any,
    ) -> None:
        if self.merge_output_path is None:
            return
        if not self.merge_urdf_path:
            logger.warning(
                "traj_run_record_json is set to {} but urdf_path is missing; skip auto merge",
                self.merge_output_path,
            )
            return
        try:
            from utils.traj_run_record_merge import merge_jsonl_files_to_traj_run_record

            world_tf, tcp_tf = self.resolve_merge_transforms(
                is_dual_arm=is_dual_arm,
                tool_left=tool_left,
                tool_right=tool_right,
                tool_single=tool_single,
            )
            merge_jsonl_files_to_traj_run_record(
                sparse_jsonl=self.merge_sparse_path,
                dense_jsonl=self.merge_dense_path,
                samples_jsonl=self.merge_samples_path,
                output_json=self.merge_output_path,
                traj_space=self.merge_traj_space,
                urdf_path=str(self.merge_urdf_path),
                ee_frame=str(self.merge_ee_frame),
                world_frame_transform=world_tf,
                tcp_frame_transform=tcp_tf,
                primary_arm=self.merge_primary_arm,
            )
        except Exception as exc:
            logger.error("Failed to write merged traj_run_record to {}: {}", self.merge_output_path, exc)

    def close_sinks(self) -> None:
        for sink in (self.sparse_sink, self.samples_sink):
            if sink is not None:
                sink.close()
        self.sparse_sink = None
        self.samples_sink = None

    def append_sparse_joint(self, joints: List[float]) -> None:
        if self.sparse_sink is None:
            return
        self.sparse_sink.append(
            {
                "record_stream": "sparse_waypoint_plan",
                "traj_space": "joint",
                "t_perf": float(time.perf_counter()),
                "joints": [float(x) for x in joints],
                "tcp_pose": None,
            }
        )

    def append_sparse_tcp(
        self,
        flexiv_pose: Union[List[float], np.ndarray],
        *,
        t_perf: float,
        arm: Optional[str] = None,
    ) -> None:
        if self.sparse_sink is None:
            return
        row: Dict[str, Any] = {
            "record_stream": "sparse_waypoint_plan",
            "traj_space": "cartesian",
            "t_perf": float(t_perf),
            "joints": None,
            "tcp_pose": flexiv_pose,
        }
        if arm is not None:
            row["arm"] = arm
        self.sparse_sink.append(row)

    def _write_empty_dense_jsonl_placeholder(self, path: Path) -> None:
        """Create or truncate ``path`` to an empty file (valid JSONL with zero lines)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def write_dense_dual_planner_trace(
        self,
        *,
        use_cartesian_motion_planner: bool,
        is_dual_arm: bool,
        control_type: str,
        robot_left: Any,
        robot_right: Any,
        planner_perf_origin: Optional[float],
    ) -> None:
        path = self.merge_dense_path
        if path is None:
            return
        wants_dump = (
            bool(use_cartesian_motion_planner)
            and bool(is_dual_arm)
            and control_type in ["eef_abs", "eef_rel"]
            and robot_left is not None
            and robot_right is not None
            and planner_perf_origin is not None
        )
        if not wants_dump:
            try:
                self._write_empty_dense_jsonl_placeholder(path)
                logger.info(
                    "Dense command trace JSONL left empty at {} (planner off, non-dual EEF, or no trace origin)",
                    path,
                )
            except Exception as exc:
                logger.error("Failed to write empty dense JSONL at {}: {}", path, exc)
            return
        try:
            rows_out: List[Dict[str, Any]] = []
            origin = float(planner_perf_origin)
            for arm, robot in (("left", robot_left), ("right", robot_right)):
                trace = robot.position_planner_command_trace()
                for item in trace:
                    rel_t = float(item[0])
                    joints_raw = item[1]
                    joints = [
                        float(x)
                        for x in np.asarray(joints_raw, dtype=np.float64).reshape(-1).tolist()
                    ]
                    rows_out.append(
                        {
                            "record_stream": "dense_command_traj",
                            "arm": arm,
                            "t_rel": rel_t,
                            "t_perf": float(origin + rel_t),
                            "joints": joints,
                        }
                    )
            with path.open("w", encoding="utf-8") as handle:
                for row in rows_out:
                    handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            logger.info("Wrote dense planner command trace ({} rows) to {}", len(rows_out), path)
        except Exception as exc:
            logger.error("Failed to dump dense command trace to {}: {}", path, exc)

    def stop_samples_thread(self, join_timeout_s: float) -> None:
        self._stop_event.set()
        th = self._samples_thread
        if th is not None and th.is_alive() and join_timeout_s > 0:
            th.join(timeout=join_timeout_s)
        self._samples_thread = None

    def start_samples_thread_if_configured(
        self,
        poll_rows: Callable[[], Sequence[Dict[str, Any]]],
    ) -> None:
        self.stop_samples_thread(1.0)
        self._stop_event.clear()
        if self.samples_sink is None:
            return
        self._samples_thread = threading.Thread(
            target=run_traj_samples_loop,
            args=(self.samples_sink, self._stop_event, self.sample_interval_s, poll_rows),
            name="rizon_traj_samples",
            daemon=True,
        )
        self._samples_thread.start()
        logger.info("Started recorded-samples trajectory thread")
