#!/usr/bin/env python3
"""
Thin visualization adapter for PolicyConnector debug and rerun side effects.

Author: OpenAI Assistant
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np

from utils.logger_config import logger
from utils.time_control import precise_loop_timing


class PolicyVisualization:
    """Provide a stable component boundary for visualization side effects."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def maybe_log_robot_state_xyz(self, **kwargs: Any) -> None:
        """Log robot-state observations as xyz state events."""
        device_name = str(kwargs["device_name"])
        if not self.is_robot_state_device(device_name):
            return
        self.owner.action_state_xyz_logger.log_state(
            device_name=device_name,
            state=kwargs["state"],
            timestamp_ns=kwargs["timestamp_ns"],
            event_type="observation_xyz",
            summary_frame_timestamp_ns=kwargs["summary_frame_timestamp_ns"],
            **self._robotwin_joint14_layout(device_name, kwargs["state"]),
        )

    def is_robot_state_device(self, device_name: str) -> bool:
        """Return True if the device name should be treated as a robot state source."""
        lowered = device_name.lower()
        # Virtual camera names such as RoboTwinCamera_0 contain "robot", but
        # their RGB tensors must never be sent to a robot-state visualizer.
        if "camera" in lowered:
            return False
        return (
            device_name in self.owner.robot_state_devices
            or lowered.startswith("robot")
            or lowered.startswith("rizon")
            or "robot" in lowered
        )

    @staticmethod
    def _robotwin_joint14_layout(device_name: str, values: Any) -> Dict[str, str]:
        """Identify RoboTwin's joint target/state vector for the generic trace."""
        if (
            str(device_name).lower().startswith("robotwinsimrobot")
            and np.asarray(values).size == 14
        ):
            return {"vector_layout": "joint14"}
        return {}

    def continuous_state_worker(self) -> None:
        """Continuously sample all robot states from summary SHM for visualization."""
        owner = self.owner
        interval = 1.0 / max(owner.continuous_state_fps, 1e-6)
        wait_for_next_sample = precise_loop_timing(interval)
        logger.info(
            "PolicyConnector: Started continuous robot state xyz sampler at "
            f"{owner.continuous_state_fps:.2f} Hz"
        )
        while owner.continuous_state_running:
            try:
                robot_devices = [
                    name for name in owner.devices.keys() if self.is_robot_state_device(name)
                ]
                if not robot_devices:
                    wait_for_next_sample()
                    continue

                read_result = owner.read_all_device_data(
                    include_frame_timestamp=True,
                    device_names=robot_devices,
                )
                if read_result is None:
                    wait_for_next_sample()
                    continue
                frame_timestamp_ns, all_data = read_result
                if all_data is None:
                    wait_for_next_sample()
                    continue

                for device_name, (timestamp, data) in all_data.items():
                    timestamp = int(timestamp)
                    if owner._last_continuous_state_timestamps.get(device_name) == timestamp:
                        continue
                    owner._last_continuous_state_timestamps[device_name] = timestamp
                    owner.action_state_xyz_logger.log_state(
                        device_name=device_name,
                        state=data,
                        timestamp_ns=timestamp,
                        event_type="state_xyz",
                        summary_frame_timestamp_ns=int(frame_timestamp_ns),
                        **self._robotwin_joint14_layout(device_name, data),
                    )
            except Exception as exc:
                logger.warning(f"Continuous robot state xyz sampling failed: {exc}")
            wait_for_next_sample()

    def start_continuous_state_logging(self) -> None:
        """Start the continuous robot-state logging thread."""
        owner = self.owner
        if owner.continuous_state_fps <= 0 or owner.continuous_state_thread is not None:
            return
        owner.continuous_state_running = True
        owner.continuous_state_thread = threading.Thread(
            target=self.continuous_state_worker,
            daemon=True,
            name="ContinuousStateXYZThread",
        )
        owner.continuous_state_thread.start()

    def stop_continuous_state_logging(self) -> None:
        """Stop the continuous robot-state logging thread."""
        owner = self.owner
        owner.continuous_state_running = False
        if owner.continuous_state_thread is None:
            return
        try:
            owner.continuous_state_thread.join(timeout=2.0)
            if owner.continuous_state_thread.is_alive():
                logger.warning("Continuous robot state xyz thread did not stop gracefully")
        except Exception as exc:
            logger.warning(f"Error stopping continuous robot state xyz thread: {exc}")
        finally:
            owner.continuous_state_thread = None

    def _log_inference_visualizer_marker(
        self,
        *,
        inference_event_type: str,
        timestamp_ns: int,
        observation_timestamp_ns: int,
        action_index: Optional[int],
        inference_start_timestamp_ns: Optional[int] = None,
        success: Optional[bool] = None,
        exception_type: Optional[str] = None,
        exception_message: Optional[str] = None,
        **timing_fields: Any,
    ) -> None:
        """Log inference timing markers with dedicated event_type names for the HTML visualizer."""
        event_timestamp_ns = int(timestamp_ns)
        marker_configs = self.action_configs or [
            {"device_name": "policy_inference", "action_dim": 8}
        ]

        if inference_event_type == "inference_start":
            for cfg in marker_configs:
                device_name = str(cfg.get("device_name", "policy_inference"))
                self.action_state_xyz_logger.log_meta(
                    "inference_start",
                    timestamp_ns=event_timestamp_ns,
                    device_name=f"policy_inference_{device_name}",
                    observation_timestamp_ns=int(observation_timestamp_ns),
                    action_index=action_index,
                    chunk_length=self.chunk_length,
                    policy_class=self.policy_class,
                    inference_phase="start",
                    success=success,
                    **timing_fields,
                )
            return

        if inference_start_timestamp_ns is None:
            return

        inference_duration_ns = event_timestamp_ns - inference_start_timestamp_ns
        interval_ns = max(self.action_step_interval_ns, 5_000_000)
        active_timestamp_ns = inference_start_timestamp_ns + interval_ns
        while active_timestamp_ns < event_timestamp_ns:
            for cfg in marker_configs:
                device_name = str(cfg.get("device_name", "policy_inference"))
                self.action_state_xyz_logger.log_meta(
                    "inference_active",
                    timestamp_ns=int(active_timestamp_ns),
                    device_name=f"policy_inference_{device_name}",
                    observation_timestamp_ns=int(observation_timestamp_ns),
                    action_index=action_index,
                    inference_start=inference_start_timestamp_ns,
                    inference_phase="active",
                    value=0.2,
                    success=success,
                )
            active_timestamp_ns += interval_ns

        for cfg in marker_configs:
            device_name = str(cfg.get("device_name", "policy_inference"))
            self.action_state_xyz_logger.log_meta(
                "inference_end",
                timestamp_ns=event_timestamp_ns,
                device_name=f"policy_inference_{device_name}",
                observation_timestamp_ns=int(observation_timestamp_ns),
                action_index=action_index,
                inference_start=inference_start_timestamp_ns,
                inference_end=event_timestamp_ns,
                inference_duration_ns=inference_duration_ns,
                inference_phase="end",
                success=success,
                exception_type=exception_type,
                exception_message=exception_message,
                chunk_length=self.chunk_length,
                policy_class=self.policy_class,
                **timing_fields,
            )

    def _maybe_log_policy_reference_xyz(self) -> None:
        reference_state = getattr(self.policy, "_last_reference_robot_state", None)
        if reference_state is None:
            return
        timestamp_ns = getattr(self.policy, "_prediction_context_timestamp_ns", None)
        if timestamp_ns is None:
            return

        device_name = None
        configured_robot_device = self.policy_params.get("robot_device")
        if configured_robot_device:
            device_name = str(configured_robot_device)
        elif self.robot_state_devices:
            device_name = sorted(self.robot_state_devices)[0]
        elif self.controls_config:
            device_name = str(self.controls_config[0].get("device_key", ""))
        if not device_name:
            device_name = "reference_robot"

        self.action_state_xyz_logger.log_state(
            device_name=device_name,
            state=np.asarray(reference_state),
            timestamp_ns=int(timestamp_ns),
            event_type="reference_xyz",
            **self._robotwin_joint14_layout(device_name, reference_state),
        )

    def _maybe_log_policy_relative_xyz(self) -> None:
        server_action = getattr(self.policy, "_last_server_action", None)
        if server_action is None or not hasattr(self.policy, "_server_action_to_chunk"):
            return
        timestamp_ns = getattr(self.policy, "_prediction_context_timestamp_ns", None)
        if timestamp_ns is None:
            return

        try:
            relative_chunk = self.policy._server_action_to_chunk(server_action)
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to log relative xyz ({exc})")
            return

        total_target_dim = sum(int(cfg.get("action_dim", 0)) for cfg in self.action_configs)
        if total_target_dim > 0 and relative_chunk.shape[1] != total_target_dim:
            if hasattr(self.policy, "_fit_action_dim"):
                relative_chunk = self.policy._fit_action_dim(
                    relative_chunk,
                    target_dim=total_target_dim,
                )
            elif relative_chunk.shape[1] > total_target_dim:
                relative_chunk = relative_chunk[:, :total_target_dim]
            else:
                padding = np.zeros(
                    (relative_chunk.shape[0], total_target_dim - relative_chunk.shape[1]),
                    dtype=relative_chunk.dtype,
                )
                relative_chunk = np.concatenate([relative_chunk, padding], axis=1)

        start_dim = 0
        for cfg in self.action_configs:
            device_name = str(cfg.get("device_name", "relative_action"))
            action_dim = int(cfg.get("action_dim", 0))
            if action_dim <= 0:
                continue
            device_chunk = relative_chunk[:, start_dim : start_dim + action_dim]
            start_dim += action_dim
            for step_index, action in enumerate(device_chunk):
                self.action_state_xyz_logger.log_action(
                    device_name=device_name,
                    action=action,
                    timestamp_ns=int(timestamp_ns)
                    + int(step_index) * self.action_step_interval_ns,
                    event_type="relative_xyz",
                    prediction_timestamp_ns=int(timestamp_ns),
                    action_step_index=int(step_index),
                    **self._robotwin_joint14_layout(device_name, action),
                )

    def _maybe_log_policy_history_relative_xyz(
        self,
        history_spec: Dict[str, Any],
    ) -> None:
        history_relative_chunk = getattr(self.policy, "_last_history_relative_chunk", None)
        if history_relative_chunk is None:
            return

        pose_timestamps = list(history_spec.get("history_pose_robot_timestamps_ns", []))
        if not pose_timestamps:
            return

        try:
            relative_chunk = np.asarray(history_relative_chunk, dtype=np.float64)
            if relative_chunk.ndim == 3:
                relative_chunk = np.moveaxis(relative_chunk, 0, -1).reshape(
                    relative_chunk.shape[1] * relative_chunk.shape[2],
                    relative_chunk.shape[0],
                )
            elif relative_chunk.ndim != 2:
                return
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to log history relative xyz ({exc})")
            return

        total_target_dim = sum(int(cfg.get("action_dim", 0)) for cfg in self.action_configs)
        if total_target_dim > 0 and relative_chunk.shape[1] != total_target_dim:
            if hasattr(self.policy, "_fit_action_dim"):
                relative_chunk = self.policy._fit_action_dim(
                    relative_chunk,
                    target_dim=total_target_dim,
                )
            elif relative_chunk.shape[1] > total_target_dim:
                relative_chunk = relative_chunk[:, :total_target_dim]
            else:
                padding = np.zeros(
                    (relative_chunk.shape[0], total_target_dim - relative_chunk.shape[1]),
                    dtype=relative_chunk.dtype,
                )
                relative_chunk = np.concatenate([relative_chunk, padding], axis=1)

        step_count = min(relative_chunk.shape[0], len(pose_timestamps))
        start_dim = 0
        for cfg in self.action_configs:
            device_name = str(cfg.get("device_name", "history_relative_action"))
            action_dim = int(cfg.get("action_dim", 0))
            if action_dim <= 0:
                continue
            device_chunk = relative_chunk[:step_count, start_dim : start_dim + action_dim]
            start_dim += action_dim
            for step_index, action in enumerate(device_chunk):
                self.action_state_xyz_logger.log_action(
                    device_name=device_name,
                    action=action,
                    timestamp_ns=int(pose_timestamps[step_index]),
                    event_type="history_relative_xyz",
                    history_step_index=int(step_index),
                    history_pose_timestamp_ns=int(pose_timestamps[step_index]),
                    **self._robotwin_joint14_layout(device_name, action),
                )
