#!/usr/bin/env python3
"""Rizon robot helper mixin.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.logger_config import logger


class RizonCommandMixin:
    """Focused RizonRobot helper methods from commands.py."""

    def get_robot_state(self) -> np.ndarray:
        """Get current robot states."""
        try:
            if not self.is_operational:
                return np.zeros(self.data_shape, dtype=self.data_dtype)

            # Surface contact-wrench safety events via edge detection on the
            # Flexiv operational_status enum.
            self._check_operational_status_change()

            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return np.zeros(self.data_shape, dtype=self.data_dtype)
                sources = (
                    (
                        self.robot_left,
                        self.robot_left.states(),
                        self.gripper_left,
                        "left",
                    ),
                    (
                        self.robot_right,
                        self.robot_right.states(),
                        self.gripper_right,
                        "right",
                    ),
                )
            else:
                if not self.robot:
                    return np.zeros(self.data_shape, dtype=self.data_dtype)
                sources = ((self.robot, self.robot.states(), self.gripper, "single"),)

            parts = [
                self._format_arm_state(states, gripper, label, index)
                for index, (_robot, states, gripper, label) in enumerate(sources)
            ]
            state_schema = getattr(self, "state_schema", None)
            if state_schema is not None and state_schema.extension:
                parts.extend(self._format_state_extension(sources))
            combined_states = np.concatenate(parts).astype(
                self.data_dtype, copy=False
            )
            if combined_states.shape != self.data_shape:
                raise ValueError(
                    "state_feedback.format produced shape "
                    f"{combined_states.shape}, expected {self.data_shape}"
                )
            return combined_states

        except Exception as e:
            logger.error(f"Failed to get robot states: {e}")
            return np.zeros(self.data_shape, dtype=self.data_dtype)

    def _format_arm_state(self, states, gripper, label: str, arm_index: int) -> np.ndarray:
        """Serialize one SDK state object according to state_feedback.format.type."""
        tcp_pose = np.asarray(states.tcp_pose, dtype=self.data_dtype)
        values = {
            "j1-7": np.asarray(states.q, dtype=self.data_dtype),
            "dq1-7": np.asarray(states.dq, dtype=self.data_dtype),
            "xyz": tcp_pose[:3],
            "qwqxqyqz": tcp_pose[3:7],
            "qxqyqzqw": tcp_pose[[4, 5, 6, 3]],
        }
        parts: list[np.ndarray] = []
        for segment in self.state_layouts[arm_index].segments:
            name = segment.primitive.name
            if name == "gripper":
                gripper_value = (
                    self._read_gripper_value_by_backend(
                        gripper, arm_index, label
                    )
                    if self.gripper_config.arms[arm_index].enabled
                    and gripper is not None
                    else 0.0
                )
                parts.append(np.asarray([gripper_value], dtype=self.data_dtype))
            else:
                parts.append(values[name])
        return np.concatenate(parts)

    def _format_state_extension(self, sources) -> list[np.ndarray]:
        """Serialize the validated dual-arm R1 feedback extension."""
        extension = self.state_schema.extension
        parts: list[np.ndarray] = []

        # The first stage publishes measured physical width and motor angle for
        # left then right. The base state read already refreshed last_gripper_value.
        for index, (_robot, _states, gripper, _label) in enumerate(sources):
            if self.gripper_config.arms[index].enabled and gripper is not None:
                width_m, motor_angle = self._gripper_width_and_motor_angle(
                    gripper, self.last_gripper_value[index], index
                )
            else:
                width_m, motor_angle = 0.0, 0.0
            parts.append(
                np.asarray([width_m, motor_angle], dtype=self.data_dtype)
            )

        if len(extension) >= 10:
            quaternion_name = extension[5]
            for robot, states, _gripper, _label in sources:
                command_pose = np.asarray(
                    self._last_commanded_tcp_by_robot.get(
                        id(robot), states.tcp_pose
                    ),
                    dtype=self.data_dtype,
                )
                if quaternion_name == "qxqyqzqw":
                    command_pose = command_pose[[0, 1, 2, 4, 5, 6, 3]]
                wrench = self._state_vector(states, "ext_wrench_in_tcp", 6)
                parts.extend((command_pose, wrench))

        if len(extension) >= 12:
            parts.extend(
                self._state_vector(states, "tcp_vel", 6)
                for _robot, states, _gripper, _label in sources
            )

        if len(extension) == 14:
            parts.append(
                np.asarray(
                    self.last_gripper_command_width,
                    dtype=self.data_dtype,
                )
            )
        return parts

    def _state_vector(self, states, attribute: str, width: int) -> np.ndarray:
        """Read one optional SDK state vector, padding unavailable fields with zero."""
        result = np.zeros(width, dtype=self.data_dtype)
        raw = getattr(states, attribute, None)
        if raw is None:
            return result
        values = np.asarray(raw, dtype=self.data_dtype).reshape(-1)
        result[: min(width, values.size)] = values[:width]
        return result

    def _decode_control_array(self, control_array: np.ndarray) -> list[dict[str, Any]]:
        """Decode a flattened command from control.format.type."""
        flat = np.asarray(control_array, dtype=self.control_dtype).reshape(-1)
        if flat.size != int(np.prod(self.control_shape)):
            raise ValueError(
                f"control command has {flat.size} values, expected {np.prod(self.control_shape)}"
            )
        commands: list[dict[str, Any]] = []
        for layout in self.control_layouts:
            command: dict[str, Any] = {}
            for segment in layout.segments:
                value = flat[segment.start : segment.stop]
                name = segment.primitive.name
                if name == "j1-7":
                    command["joints"] = value
                elif name == "xyz":
                    command["position"] = value
                elif name == "qxqyqzqw":
                    command["quaternion_xyzw"] = value
                elif name == "qwqxqyqz":
                    command["quaternion_xyzw"] = value[[1, 2, 3, 0]]
                elif name == "gripper":
                    command["gripper"] = float(value[0])
            commands.append(command)
        return commands

    def get_control_array(self, control_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Return only the frame selected by ``control.source_selection``."""
        selected_name = self._selected_source_name(
            self._selector_value_for_control_tick()
        )
        return control_dict.get(selected_name) if selected_name is not None else None

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """Execute control command based on control type."""
        try:
            # Source arbitration is resolved by the typed control contract.
            # The command executor receives all connected channels but consumes
            # only the selected endpoint for this tick.
            control_array = self.get_control_array(control_dict)

            if control_array is None:
                return
            if not self.is_operational:
                return

            if self.control_type == "joints":
                self._execute_joint_command(control_array)
            elif self.control_type in ["eef_abs", "eef_rel"]:
                self._execute_eef_command(control_array)
            else:
                logger.error(f"Unknown control type: {self.control_type}")

        except Exception as e:
            logger.error(f"Failed to execute control command: {e}")

    def _execute_joint_command(self, control_array: np.ndarray, is_nrt: bool = True) -> None:
        """Execute joint position command in NRT/RT mode."""
        try:
            commands = self._decode_control_array(control_array)
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return
                left_pos = commands[0]["joints"].tolist()
                right_pos = commands[1]["joints"].tolist()
                self._send_joint_position_command(
                    self.robot_left, left_pos, self.max_velocity[:7], self.max_acceleration[:7], is_nrt=is_nrt
                )
                self._send_joint_position_command(
                    self.robot_right, right_pos, self.max_velocity[7:14], self.max_acceleration[7:14], is_nrt=is_nrt
                )
                self._maybe_log_sparse_joint_vector(left_pos + right_pos)
            else:
                if not self.robot:
                    return
                target_pos = commands[0]["joints"].tolist()
                self._send_joint_position_command(
                    self.robot, target_pos, self.max_velocity, self.max_acceleration, is_nrt=is_nrt
                )
                self._maybe_log_sparse_joint_vector(target_pos)

        except Exception as e:
            logger.error(f"Failed to execute joint command: {e}")

    def _execute_eef_command(self, control_array: np.ndarray, is_nrt: bool = True) -> None:
        """Execute end-effector command in NRT/RT mode."""
        try:
            commands = self._decode_control_array(control_array)
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return
                left_target = None
                right_target = None
                left_target = self._process_eef_command(
                    np.concatenate(
                        [commands[0]["position"], commands[0]["quaternion_xyzw"]]
                    ),
                    self.home_pose_left,
                    "left",
                )
                if left_target is None:
                    return
                right_target = self._process_eef_command(
                    np.concatenate(
                        [commands[1]["position"], commands[1]["quaternion_xyzw"]]
                    ),
                    self.home_pose_right,
                    "right",
                )
                if right_target is None:
                    return

                if self.use_cartesian_motion_planner and left_target is not None and right_target is not None:
                    if self.current_point_time == 0.0:
                        self._planner_add_origin = time.perf_counter()
                    t_add_0 = time.perf_counter()
                    self.robot_left.add_cartesian_position_plan(self.current_point_time, left_target)
                    t_add_mid = time.perf_counter()
                    self.robot_right.add_cartesian_position_plan(self.current_point_time, right_target)
                    t_add_1 = time.perf_counter()
                    dt_left_ms = (t_add_mid - t_add_0) * 1000.0
                    dt_right_ms = (t_add_1 - t_add_mid) * 1000.0
                    dt_total_s = t_add_1 - t_add_0
                    new_tt = time.perf_counter()
                    if self.num_clock_ticks % 10 == 0:
                        logger.info(f"period: {new_tt - self.real_time}, dt_total_s:{dt_total_s})")
                    self.real_time = new_tt
                    self.num_clock_ticks += 1
                    self._maybe_log_sparse_tcp_flexiv(left_target, t_perf=self._planner_add_origin + self.current_point_time, arm="left")
                    self._maybe_log_sparse_tcp_flexiv(right_target, t_perf=self._planner_add_origin + self.current_point_time + 1e-9, arm="right")
                    self._next_planner_time_s()
                    if dt_total_s > self.command_period_s:
                        plan_t = float(self.current_point_time)
                        logger.warning(
                            "add_cartesian_position_plan slow: left={:.2f}ms right={:.2f}ms total={:.2f}ms "
                            "(>{:.2f}ms = 1x command period) plan_time_s={:.4f}",
                            dt_left_ms,
                            dt_right_ms,
                            dt_total_s * 1000.0,
                            self.command_period_s * 1000.0,
                            plan_t,
                        )
                else:
                    t_sparse = time.perf_counter()
                    if left_target is not None:
                        self._maybe_log_sparse_tcp_flexiv(left_target, t_perf=t_sparse, arm="left")
                        self._send_cartesian_motion_command(self.robot_left, left_target, is_nrt=is_nrt)
                    if right_target is not None:
                        self._maybe_log_sparse_tcp_flexiv(right_target, t_perf=t_sparse + 1e-9, arm="right")
                        self._send_cartesian_motion_command(self.robot_right, right_target, is_nrt=is_nrt)

                gripper_specs = (
                    (0, "left", self.gripper_left),
                    (1, "right", self.gripper_right),
                )
                for arm_index, label, gripper in gripper_specs:
                    gripper_value = commands[arm_index].get("gripper")
                    if (
                        gripper_value is not None
                        and gripper is not None
                    ):
                        scaled_value = self._scale_realtime_gripper_command(
                            gripper_value, arm_index
                        )
                        self._move_gripper_value_by_backend(
                            gripper, scaled_value, arm_index, label
                        )
            else:
                if not self.robot:
                    return
                target = self._process_eef_command(
                    np.concatenate(
                        [commands[0]["position"], commands[0]["quaternion_xyzw"]]
                    ),
                    self.home_pose,
                    "single",
                )

                if target is not None:
                    self._maybe_log_sparse_tcp_flexiv(target, t_perf=time.perf_counter())
                    self._send_cartesian_motion_command(self.robot, target, is_nrt=is_nrt)
                    gripper_value = commands[0].get("gripper")
                    if gripper_value is not None and self.gripper is not None:
                        scaled_value = self._scale_realtime_gripper_command(
                            gripper_value, 0
                        )
                        self._move_gripper_value_by_backend(
                            self.gripper, scaled_value, 0, "single"
                        )

        except Exception as e:
            logger.error(f"Failed to execute EEF command: {e}")

    def _process_eef_command(self, command: np.ndarray, home_pose, arm_name: str):
        """Process end-effector command based on control type."""
        try:
            if home_pose is None:
                logger.error(f"Home pose not set for {arm_name} arm")
                return None

            # Extract position and quaternion from command [x, y, z, qx, qy, qz, qw]
            cmd_pos = command[:3]
            cmd_quat = command[3:7]
            if self.control_type == "eef_abs":
                # Absolute positioning
                target_pos = cmd_pos
                target_quat = cmd_quat
                threshold = 0.50
                if arm_name == "left":
                    crt_pos = self.robot_left.states().tcp_pose[:3]
                    # exit if np.linalg.norm(crt_left_pos - target_pos) > 0.01 or np.linalg.norm(crt_left_quat - target_quat) > 0.01
                elif arm_name == "right":
                    crt_pos = self.robot_right.states().tcp_pose[:3]
                else:
                    crt_pos = self.robot.states().tcp_pose[:3]
                delta_move = np.linalg.norm(crt_pos - target_pos)
                if delta_move > threshold:
                    logger.error(f"{arm_name} arm's move is too large {delta_move}")
                    return None

            elif self.control_type == "eef_rel":
                cmd_rot = R.from_quat(cmd_quat)
                robot = (
                    self.robot_left
                    if arm_name == "left"
                    else self.robot_right
                    if arm_name == "right"
                    else self.robot
                )
                crt_pose = robot.states().tcp_pose
                crt_pos = crt_pose[:3]
                crt_quat = crt_pose[3:7]
                crt_rot = R.from_quat(crt_quat)
                    # exit if np.linalg.norm(crt_left_pos - target_pos) > 0.01 or np.linalg.norm(crt_left_quat - target_quat) > 0.01
                target_pos = crt_pos + crt_rot.apply(cmd_pos.copy())
                target_rot = crt_rot * cmd_rot
                target_quat = target_rot.as_quat()

            else:
                logger.error(f"Unknown EEF control type: {self.control_type}")
                return None

            # Apply axis-aligned XYZ safety bounding box on the resolved
            # absolute target position (robot base frame). Quaternion and
            # gripper components are not affected.
            target_pos_arr = np.asarray(target_pos, dtype=float).reshape(-1)
            target_pos_arr, was_clipped = self.safety_bounds.clip_position(
                target_pos_arr, arm_name=arm_name
            )
            if was_clipped:
                self._record_safety_clip_event(arm_name)
            target_pos = target_pos_arr

            return self._xyzw_pose_to_flexiv_pose(
                [target_pos[0], target_pos[1], target_pos[2], target_quat[0], target_quat[1], target_quat[2], target_quat[3]]
            )

        except Exception as e:
            logger.error(f"Failed to process EEF command for {arm_name}: {e}")
            return None
