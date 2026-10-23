"""Trace events that make simulated action application joinable with hardware logs."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from utils.action_state_xyz_logger import ActionStateXYZLogger

from .contracts import ControlSetpoint


class RoboTwinTrace:
    """Write full-vector bridge events through SuperInference's async JSONL logger."""

    def __init__(self, source: str, *, log_dir: Optional[str] = None) -> None:
        self.logger = ActionStateXYZLogger(source, log_dir=log_dir)

    @property
    def path(self) -> Optional[str]:
        return str(self.logger.path) if self.logger.path is not None else None

    def control_accepted(
        self,
        setpoint: ControlSetpoint,
        *,
        timestamp_ns: int,
        sim_step: int,
        sim_time_ns: int,
        control_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        fields: dict[str, Any] = {
            "timestamp_ns": int(timestamp_ns),
            "control_write_timestamp_ns": setpoint.control_write_timestamp_ns,
            "dispatch_timestamp_ns": setpoint.control_write_timestamp_ns,
            "action_timestamp_ns": setpoint.control_write_timestamp_ns,
            "action_schema": setpoint.action_schema,
            "action_sequence": setpoint.sequence,
            "target_action": np.asarray(setpoint.action, dtype=np.float64),
            "sim_step": int(sim_step),
            "sim_time_ns": int(sim_time_ns),
        }
        if control_metrics:
            fields.update(control_metrics)
        self.logger.log_meta(
            "sim_control_accepted",
            **fields,
        )

    def action_applied(
        self,
        setpoint: ControlSetpoint,
        *,
        target_joint14: np.ndarray,
        timestamp_ns: int,
        sim_step: int,
        sim_time_ns: int,
    ) -> None:
        self.logger.log_meta(
            "sim_action_applied",
            timestamp_ns=int(timestamp_ns),
            control_write_timestamp_ns=setpoint.control_write_timestamp_ns,
            dispatch_timestamp_ns=setpoint.control_write_timestamp_ns,
            action_timestamp_ns=setpoint.control_write_timestamp_ns,
            executed_action_timestamp_ns=int(timestamp_ns),
            action_stage="sim_applied",
            action_schema=setpoint.action_schema,
            action_sequence=setpoint.sequence,
            target_action=np.asarray(setpoint.action, dtype=np.float64),
            target_qpos=np.asarray(target_joint14, dtype=np.float64),
            sim_step=int(sim_step),
            sim_time_ns=int(sim_time_ns),
        )

    def measured_state(
        self,
        *,
        measured_qpos: np.ndarray,
        measured_qvel: np.ndarray,
        target_qpos: np.ndarray,
        last_control_write_timestamp_ns: Optional[int],
        last_sim_apply_timestamp_ns: Optional[int],
        timestamp_ns: int,
        sim_step: int,
        sim_time_ns: int,
    ) -> None:
        self.logger.log_meta(
            "sim_measured_state",
            timestamp_ns=int(timestamp_ns),
            measured_timestamp_ns=int(timestamp_ns),
            measured_qpos=np.asarray(measured_qpos, dtype=np.float64),
            measured_qvel=np.asarray(measured_qvel, dtype=np.float64),
            target_qpos=np.asarray(target_qpos, dtype=np.float64),
            last_control_write_timestamp_ns=last_control_write_timestamp_ns,
            last_sim_apply_timestamp_ns=last_sim_apply_timestamp_ns,
            sim_step=int(sim_step),
            sim_time_ns=int(sim_time_ns),
        )

    def close(self) -> None:
        self.logger.close()
