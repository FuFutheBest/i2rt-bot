#!/usr/bin/env python3
"""Rizon robot helper mixin.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.logger_config import logger
from utils.rizon_traj_planner_utils import build_planner_start_kwargs, traj_smooth_settings


class RizonPlannerMixin:
    """Planner and trajectory-recording helpers for RizonRobot."""

    def _reset_planner_command_timeline(self) -> None:
        """Reset planner relative time axis (first waypoint must use time_s=0)."""
        self.current_point_time = 0.0
        self._planner_waypoint_index = 0
        self._planner_add_origin = None

    def _next_planner_time_s(self) -> float:
        """Next strictly spaced planner timestamp: index * (1/command_fps)."""
        time_s = float(self._planner_waypoint_index) * self.command_period_s
        self._planner_waypoint_index += 1
        self.current_point_time = (
            float(self._planner_waypoint_index) * self.command_period_s
        )
        return time_s

    def _on_motion_planner_loop_tick(self) -> None:
        """Log dual-arm Cartesian planner status periodically for backlog diagnosis."""
        if not (
            self.use_cartesian_motion_planner
            and self.is_dual_arm
            and self.robot_left is not None
            and self.robot_right is not None
        ):
            return
        now = time.perf_counter()
        if now - self._planner_diag_last_perf < self._planner_diag_interval_s:
            return
        self._planner_diag_last_perf = now
        try:
            sl = self.robot_left.position_planner_status()
            sr = self.robot_right.position_planner_status()
        except Exception as exc:
            logger.warning("Planner diagnostic: position_planner_status failed: {}", exc)
            return

        def _stat(obj: Any, key: str, default: float = float("nan")) -> float:
            try:
                return float(obj[key])
            except Exception:
                return default

        ql = int(_stat(sl, "queue_size"))
        qr = int(_stat(sr, "queue_size"))
        el = int(_stat(sl, "executed_count"))
        er = int(_stat(sr, "executed_count"))
        cl = int(_stat(sl, "consumed_count", float(el)))
        cr = int(_stat(sr, "consumed_count", float(er)))
        tl = int(_stat(sl, "trajectory_count"))
        tr = int(_stat(sr, "trajectory_count"))
        growth_note = ""
        if self._planner_diag_prev_queue_left is not None and self._planner_diag_prev_queue_right is not None:
            if ql > self._planner_diag_prev_queue_left or qr > self._planner_diag_prev_queue_right:
                growth_note = " | queue_size grew vs last sample"
        self._planner_diag_prev_queue_left = ql
        self._planner_diag_prev_queue_right = qr
        logger.info(
            "Planner diag: L queue={} consumed={} exec={} traj={} | R queue={} consumed={} exec={} traj={}{}",
            ql,
            cl,
            el,
            tl,
            qr,
            cr,
            er,
            tr,
            growth_note,
        )

    def _maybe_write_traj_run_record_json(self) -> None:
        """On shutdown, merge JSONL fragments into one ``traj_run_record`` JSON when configured."""
        if self._traj_rec is None:
            return
        self._traj_rec.merge_json_if_configured(
            is_dual_arm=self.is_dual_arm,
            tool_left=self.tool_left,
            tool_right=self.tool_right,
            tool_single=self.tool,
        )

    def _close_traj_jsonl_sinks(self) -> None:
        if self._traj_rec is not None:
            self._traj_rec.close_sinks()

    def _maybe_log_sparse_joint_vector(self, joints: List[float]) -> None:
        if self._traj_rec is None or self._traj_rec.sparse_sink is None:
            return
        self._traj_rec.append_sparse_joint(joints)

    def _maybe_log_sparse_tcp_flexiv(
        self,
        flexiv_pose: Union[List[float], np.ndarray],
        *,
        t_perf: float,
        arm: Optional[str] = None,
    ) -> None:
        if self._traj_rec is None or self._traj_rec.sparse_sink is None:
            return
        self._traj_rec.append_sparse_tcp(flexiv_pose, t_perf=t_perf, arm=arm)

    def _write_dense_planner_trace_at_shutdown(self) -> None:
        if self._traj_rec is None:
            return
        origin = self._planner_trace_perf_origin
        self._traj_rec.write_dense_dual_planner_trace(
            use_cartesian_motion_planner=self.use_cartesian_motion_planner,
            is_dual_arm=self.is_dual_arm,
            control_type=self.control_type,
            robot_left=self.robot_left,
            robot_right=self.robot_right,
            planner_perf_origin=float(origin) if origin is not None else None,
        )

    def _poll_traj_sample_rows(self) -> List[Dict[str, Any]]:
        """One samples tick for trajectory JSONL (dual or single arm)."""
        rows: List[Dict[str, Any]] = []
        t0 = float(time.perf_counter())
        if self.is_dual_arm and self.robot_left and self.robot_right:
            sl = self.robot_left.states()
            sr = self.robot_right.states()
            rows.append(
                {
                    "record_stream": "recorded_samples",
                    "t_perf": t0,
                    "arm": "left",
                    "joints": [float(x) for x in np.asarray(sl.q, dtype=np.float64).reshape(-1).tolist()],
                    "tcp_pose": [float(x) for x in np.asarray(sl.tcp_pose, dtype=np.float64).reshape(-1).tolist()],
                }
            )
            rows.append(
                {
                    "record_stream": "recorded_samples",
                    "t_perf": t0 + 1e-9,
                    "arm": "right",
                    "joints": [float(x) for x in np.asarray(sr.q, dtype=np.float64).reshape(-1).tolist()],
                    "tcp_pose": [float(x) for x in np.asarray(sr.tcp_pose, dtype=np.float64).reshape(-1).tolist()],
                }
            )
        elif self.robot:
            st = self.robot.states()
            rows.append(
                {
                    "record_stream": "recorded_samples",
                    "t_perf": t0,
                    "joints": [float(x) for x in np.asarray(st.q, dtype=np.float64).reshape(-1).tolist()],
                    "tcp_pose": [float(x) for x in np.asarray(st.tcp_pose, dtype=np.float64).reshape(-1).tolist()],
                }
            )
        return rows

    def _stop_traj_recording_threads(self, join_timeout_s: float = 2.0) -> None:
        if self._traj_rec is not None:
            self._traj_rec.stop_samples_thread(join_timeout_s)

    def _start_traj_recording_threads_if_configured(self) -> None:
        if self._traj_rec is None:
            return
        self._traj_rec.start_samples_thread_if_configured(self._poll_traj_sample_rows)

    def _planner_home_joint_seeds(self) -> Optional[Tuple[List[float], List[float]]]:
        """Home preset joints as 7-DoF lists per arm, or None if invalid."""
        if self.home_joints_left is None or self.home_joints_right is None:
            logger.error(
                "Dual-arm Cartesian planner requires home.joints_left / joints_right (from presets)"
            )
            return None
        left = np.asarray(self.home_joints_left, dtype=np.float64).reshape(-1).tolist()
        right = np.asarray(self.home_joints_right, dtype=np.float64).reshape(-1).tolist()
        if len(left) != 7 or len(right) != 7:
            logger.error(
                "home joints for Cartesian planner must have length 7 per arm; "
                f"got left={len(left)} right={len(right)}"
            )
            return None
        return left, right

    def _start_dual_cartesian_planners_if_configured(self) -> bool:
        """Start C++ Cartesian planners for dual-arm EEF when enabled; set dense-trace time origin."""
        if self.control_type not in ("eef_abs", "eef_rel"):
            self._planner_trace_perf_origin = None
            return True
        if not self.use_cartesian_motion_planner:
            self._planner_trace_perf_origin = None
            logger.info("Dual-arm EEF: Cartesian motion planner disabled; using send_cartesian_motion_force.")
            return True
        seeds = self._planner_home_joint_seeds()
        if seeds is None:
            return False
        try:
            cp = self._cartesian_planner_cfg
            self._planner_trace_perf_origin = float(time.perf_counter())
            for arm, seed, tool, robot in (
                ("left", seeds[0], self.tool_left, self.robot_left),
                ("right", seeds[1], self.tool_right, self.robot_right),
            ):
                robot.start_cartesian_position_planner(
                    **build_planner_start_kwargs(cp, arm=arm, seed_positions=seed, tool=tool)
                )
            smooth = traj_smooth_settings(cp)
            if smooth["enabled"]:
                control_dt = float(cp.get("control_dt", 0.001))
                logger.info(
                    "Cartesian planner: traj_smooth post-process on Ruckig segments "
                    "(upsample={}, filter_window={}, command_hz={:.0f})",
                    smooth["upsample_ratio"],
                    smooth["filter_window"],
                    1.0 / control_dt,
                )
            logger.info("Dual-arm Cartesian position planner started (dense trace origin set).")
            return True
        except Exception as exc:
            logger.error(f"Failed to start dual Cartesian planners: {exc}")
            return False

    def _stop_dual_cartesian_planners_if_configured(self) -> None:
        """Stop C++ Cartesian planner workers before robot Disable/shutdown."""
        dual_eef = self.control_type in ["eef_abs", "eef_rel"]
        if not dual_eef or not self.use_cartesian_motion_planner:
            return
        if not self.is_dual_arm or self.robot_left is None or self.robot_right is None:
            return
        for arm_name, robot in (("left", self.robot_left), ("right", self.robot_right)):
            try:
                robot.stop_position_planner()
                logger.info("{} arm Cartesian position planner stopped", arm_name)
            except Exception as exc:
                logger.warning("Failed to stop {} arm Cartesian planner cleanly: {}", arm_name, exc)
