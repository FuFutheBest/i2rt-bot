#!/usr/bin/env python3
"""Rizon robot helper mixin.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time
from typing import Any, List, Union

import numpy as np

from utils.logger_config import logger


class RizonConnectionMixin:
    """Focused RizonRobot helper methods from connection.py."""

    def _connect_robot(self) -> bool:
        """Connect to Rizon robot(s)."""
        try:
            if self.is_dual_arm:
                return self._connect_dual_arm_robots()
            else:
                return self._connect_single_robot()
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            return False

    def _connect_single_robot(self) -> bool:
        """Connect to single Rizon robot."""
        try:
            logger.info(f"Connecting to Rizon robot: {self.robot_sn}")

            # Create robot interface
            logger.info("Creating robot interface...")
            self.robot = self.flexiv_adapter.create_robot(
                self.robot_sn, self.network_interface_whitelist
            )
            logger.info("Robot interface created successfully")

            # Check for faults and enable
            if not self._check_and_enable_robot(self.robot, "Robot"):
                return False

            self.is_enabled = True

            # Initialize gripper if enabled
            if self.utilize_gripper:
                if not self._initialize_gripper():
                    logger.error("Failed to initialize gripper")
                    return False

            self.is_operational = True
            logger.info("Robot connected and operational")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to single robot: {e}")
            return False

    def _connect_dual_arm_robots(self) -> bool:
        """Connect to dual arm Rizon robots."""
        try:
            logger.info(f"Connecting to dual-arm Rizon robots: {self.left_sn}, {self.right_sn}")

            # Create left robot interface
            logger.info("Creating left robot interface...")
            self.robot_left = self.flexiv_adapter.create_robot(
                self.left_sn, self.network_interface_whitelist
            )
            logger.info("Left robot interface created successfully")

            # Create right robot interface
            logger.info("Creating right robot interface...")
            self.robot_right = self.flexiv_adapter.create_robot(
                self.right_sn, self.network_interface_whitelist
            )
            logger.info("Right robot interface created successfully")

            # Check for faults and enable both robots
            if not self._check_and_enable_robot(self.robot_left, "Left robot"):
                return False
            if not self._check_and_enable_robot(self.robot_right, "Right robot"):
                return False

            self.is_enabled = True

            # Initialize grippers if enabled
            if self.utilize_gripper:
                if not self._initialize_dual_grippers():
                    logger.error("Failed to initialize grippers")
                    return False

            self.is_operational = True
            logger.info("Dual-arm robots connected and operational")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to dual-arm robots: {e}")
            return False

    def _check_and_enable_robot(self, robot, name: str) -> bool:
        """Check for faults and enable a robot."""
        try:
            # Check for faults
            logger.info(f"Checking {name} status...")
            if robot.fault():
                logger.warning(f"{name} has faults, attempting to clear...")
                if not robot.clear_fault():
                    logger.error(f"Failed to clear {name} faults")
                    return False
                logger.info(f"{name} faults cleared successfully")
            else:
                logger.info(f"{name} status: No faults detected")

            # Enable robot
            logger.info(f"Enabling {name}...")
            robot.enable()

            # Wait for operational status
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            logger.info(f"Waiting for {name} to become operational...")

            while not robot.is_operational():
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.error(f"Timeout waiting for {name} to become operational after {timeout} seconds")
                    return False

                # Show progress every 5 seconds
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    remaining = timeout - elapsed
                    logger.info(f"Still waiting for {name}... {remaining:.1f}s remaining")

                time.sleep(1.0)

            logger.info(f"{name} is now operational")
            return True

        except Exception as e:
            logger.error(f"Failed to enable {name}: {e}")
            return False

    def _initialize_gripper(self) -> bool:
        """Initialize gripper for single arm robot."""
        try:
            config = self.gripper_config
            arm_config = config.arms[0]
            if config.backend == "flexiv":
                logger.info(f"Initializing flexiv gripper [{config.name}]")
                self.gripper = self.flexiv_adapter.create_gripper(self.robot)
                self.tool = self.flexiv_adapter.create_tool(self.robot)
                logger.info(f"Enabling gripper [{config.name}]")
                self.gripper.enable(config.name)

                logger.info("Gripper parameters:")
                logger.info(f"  name: {self.gripper.params().name}")
                logger.info(f"  min_width: {round(self.gripper.params().min_width, 3)}")
                logger.info(f"  max_width: {round(self.gripper.params().max_width, 3)}")
                logger.info(f"  min_force: {round(self.gripper.params().min_force, 3)}")
                logger.info(f"  max_force: {round(self.gripper.params().max_force, 3)}")
                logger.info(f"  min_vel: {round(self.gripper.params().min_vel, 3)}")
                logger.info(f"  max_vel: {round(self.gripper.params().max_vel, 3)}")

                self._switch_tool_if_needed(
                    self.robot, self.tool, config.tool_name, "robot"
                )
                self._move_flexiv_gripper_home(self.gripper, 0, "single")
                logger.info("Gripper initialized successfully")
                return True

            logger.info("Initializing LK gripper for single arm")

            self.gripper = self._create_lk_gripper(
                self.robot, 0, config.name or "gripper"
            )
            self.tool = self.flexiv_adapter.create_tool(self.robot)

            logger.info(
                f"LK gripper config: port={arm_config.port}, id={arm_config.motor_id}, "
                f"speed_limit={arm_config.speed_dps:.2f} deg/s, "
                f"home_width={self._get_home_gripper_width(0):.4f} m, "
                f"home_ratio={arm_config.home_ratio}"
            )
            self.gripper.Enable(config.name or "gripper")
            if not self.gripper.Init():
                self._shutdown_grippers()
                return False

            # Print gripper parameters
            logger.info("Gripper parameters:")
            logger.info(f"  name: {self.gripper.params().name}")
            logger.info(f"  min_width: {round(self.gripper.params().min_width, 3)}")
            logger.info(f"  max_width: {round(self.gripper.params().max_width, 3)}")
            logger.info(f"  min_force: {round(self.gripper.params().min_force, 3)}")
            logger.info(f"  max_force: {round(self.gripper.params().max_force, 3)}")
            logger.info(f"  min_vel: {round(self.gripper.params().min_vel, 3)}")
            logger.info(f"  max_vel: {round(self.gripper.params().max_vel, 3)}")

            self._switch_tool_if_needed(
                self.robot, self.tool, config.tool_name, "robot"
            )
            if not self._move_gripper_home_by_backend(self.gripper, 0, "single", wait=True):
                self._shutdown_grippers()
                return False
            logger.info("Gripper initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize gripper: {e}")
            self._shutdown_grippers()
            return False

    def _initialize_dual_grippers(self) -> bool:
        """Initialize grippers for dual arm robots."""
        try:
            config = self.gripper_config
            if config.backend == "flexiv":
                logger.info(f"Initializing flexiv grippers [{config.name}] for dual arms")
                self.gripper_left = self.flexiv_adapter.create_gripper(self.robot_left)
                self.gripper_right = self.flexiv_adapter.create_gripper(self.robot_right)
                self.tool_left = self.flexiv_adapter.create_tool(self.robot_left)
                self.tool_right = self.flexiv_adapter.create_tool(self.robot_right)

                logger.info(f"Enabling left gripper [{config.name}]")
                self.gripper_left.enable(config.name)
                logger.info(f"Enabling right gripper [{config.name}]")
                self.gripper_right.enable(config.name)

                logger.info("Left gripper parameters:")
                logger.info(f"  name: {self.gripper_left.params().name}")
                logger.info(f"  min_width: {round(self.gripper_left.params().min_width, 3)}")
                logger.info(f"  max_width: {round(self.gripper_left.params().max_width, 3)}")

                self._switch_tool_if_needed(
                    self.robot_left, self.tool_left, config.tool_name, "left robot"
                )
                self._switch_tool_if_needed(
                    self.robot_right, self.tool_right, config.tool_name, "right robot"
                )
                self._move_flexiv_gripper_home(self.gripper_left, 0, "left")
                self._move_flexiv_gripper_home(self.gripper_right, 1, "right")

                logger.info("Dual grippers initialized successfully")
                return True

            logger.info("Initializing LK grippers for dual arms")
            arm_specs = [
                ("left", 0, self.robot_left, "gripper_left", "tool_left"),
                ("right", 1, self.robot_right, "gripper_right", "tool_right"),
            ]
            for arm_label, index, robot, gripper_attr, tool_attr in arm_specs:
                arm_config = config.arms[index]
                if not arm_config.enabled:
                    logger.info(f"{arm_label.capitalize()} gripper disabled by config")
                    setattr(self, gripper_attr, None)
                    setattr(self, tool_attr, None)
                    continue

                gripper = self._create_lk_gripper(robot, index, f"{arm_label} gripper")
                tool = self.flexiv_adapter.create_tool(robot)
                setattr(self, gripper_attr, gripper)
                setattr(self, tool_attr, tool)

                logger.info(
                    f"{arm_label.capitalize()} LK gripper config: "
                    f"port={arm_config.port}, id={arm_config.motor_id}, "
                    f"speed_limit={arm_config.speed_dps:.2f} deg/s, "
                    f"home_width={self._get_home_gripper_width(index):.4f} m, "
                    f"home_ratio={arm_config.home_ratio}"
                )

                gripper.Enable(f"{arm_label} gripper")
                if not gripper.Init():
                    self._shutdown_grippers()
                    return False

                logger.info(f"{arm_label.capitalize()} gripper parameters:")
                logger.info(f"  name: {gripper.params().name}")
                logger.info(f"  min_width: {round(gripper.params().min_width, 3)}")
                logger.info(f"  max_width: {round(gripper.params().max_width, 3)}")

                self._switch_tool_if_needed(
                    robot, tool, config.tool_name, f"{arm_label} robot"
                )
                if not self._move_gripper_home_by_backend(gripper, index, arm_label, wait=True):
                    self._shutdown_grippers()
                    return False

            logger.info("Dual grippers initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize dual grippers: {e}")
            self._shutdown_grippers()
            return False

    def _switch_to_joint_control(self, is_nrt: bool = True) -> bool:
        """Switch robot(s) to joint position control mode (NRT or RT)."""
        try:
            target_mode = (
                self.flexiv_adapter.mode_joint_position
                if is_nrt
                else self.flexiv_adapter.mode_rt_joint_position
            )
            mode_name = "NRT_JOINT_POSITION" if is_nrt else "RT_JOINT_POSITION"
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False
                # Switch both robots to joint control
                self.robot_left.switch_mode(target_mode)
                self.robot_right.switch_mode(target_mode)
                logger.info(f"Switched dual-arm robots to {mode_name} mode")
            else:
                if not self.robot:
                    return False
                # Switch single robot to joint control
                self.robot.switch_mode(target_mode)
                logger.info(f"Switched robot to {mode_name} mode")

            # Warning: Cartesian impedance parameters are not used in joint control mode
            logger.warning("Joint control mode active - Impedance parameters are not applied")

            self.current_mode = mode_name
            return True

        except Exception as e:
            logger.error(f"Failed to switch to joint control mode: {e}")
            return False

    def _switch_to_eef_control(self, is_nrt: bool = True) -> bool:
        """Switch robot(s) to end-effector control mode (NRT or RT)."""
        try:
            target_mode = (
                self.flexiv_adapter.mode_cartesian_motion_force
                if is_nrt
                else self.flexiv_adapter.mode_rt_cartesian_motion_force
            )
            mode_name = "NRT_CARTESIAN_MOTION_FORCE" if is_nrt else "RT_CARTESIAN_MOTION_FORCE"
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False
                # Switch both robots to Cartesian control
                self.robot_left.switch_mode(target_mode)
                self.robot_right.switch_mode(target_mode)
                self.robot_left.set_force_control_axis([False, False, False, False, False, False])
                self.robot_right.set_force_control_axis([False, False, False, False, False, False])
                self.robot_left.set_cartesian_impedance(self.impedance_parameter)
                self.robot_right.set_cartesian_impedance(self.impedance_parameter)
                if self.max_contact_wrench is not None:
                    self.robot_left.set_max_contact_wrench(self.max_contact_wrench)
                    self.robot_right.set_max_contact_wrench(self.max_contact_wrench)
                    logger.info(
                        f"Max contact wrench applied to dual-arm: {self.max_contact_wrench}"
                    )
                logger.info(f"Switched dual-arm robots to {mode_name} mode")
            else:
                if not self.robot:
                    return False
                # Switch single robot to Cartesian control
                self.robot.switch_mode(target_mode)
                self.robot.set_force_control_axis([False, False, False, False, False, False])
                self.robot.set_cartesian_impedance(self.impedance_parameter)
                if self.max_contact_wrench is not None:
                    self.robot.set_max_contact_wrench(self.max_contact_wrench)
                    logger.info(
                        f"Max contact wrench applied: {self.max_contact_wrench}"
                    )
                logger.info(f"Switched robot to {mode_name} mode")

            self.current_mode = mode_name
            return True

        except Exception as e:
            logger.error(f"Failed to switch to end-effector control mode: {e}")
            return False

    def switch_to(self, target: str, is_nrt: bool = True) -> bool:
        """
        Public switch interface for runtime control mode switching.

        Args:
            target: "joint" or "eef".
            is_nrt: True for NRT modes, False for RT modes.
        """
        normalized_target = target.strip().lower()
        if normalized_target == "joint":
            return self._switch_to_joint_control(is_nrt=is_nrt)
        if normalized_target == "eef":
            return self._switch_to_eef_control(is_nrt=is_nrt)
        logger.error(f"Unsupported switch target: {target}")
        return False

    def _is_joint_mode_active(self) -> bool:
        """Return True when robot is in either NRT or RT joint position mode."""
        return self.current_mode in {"NRT_JOINT_POSITION", "RT_JOINT_POSITION"}

    def _is_eef_mode_active(self) -> bool:
        """Return True when robot is in either NRT or RT Cartesian motion-force mode."""
        return self.current_mode in {"NRT_CARTESIAN_MOTION_FORCE", "RT_CARTESIAN_MOTION_FORCE"}

    def _send_joint_position_command(
        self,
        robot: Any,
        positions: Union[List[float], np.ndarray],
        max_vel: List[float],
        max_acc: List[float],
        is_nrt: bool = True,
    ) -> None:
        """Send joint command through NRT send API or RT stream API."""
        positions_list = np.asarray(positions, dtype=float).reshape(-1).tolist()
        dof = len(positions_list)
        if is_nrt:
            robot.send_joint_position(
                positions=positions_list,
                velocities=[0.0] * dof,
                max_vel=max_vel,
                max_acc=max_acc,
            )
            return
        robot.stream_joint_position(
            positions=positions_list,
            velocities=[0.0] * dof,
            accelerations=[0.0] * dof,
        )

    def _send_cartesian_motion_command(
        self,
        robot: Any,
        pose: Union[List[float], np.ndarray],
        is_nrt: bool = True
    ) -> None:
        """Send Cartesian command through NRT send API or RT stream API."""
        pose_list = np.asarray(pose, dtype=float).reshape(-1).tolist()
        self._last_commanded_tcp_by_robot[id(robot)] = pose_list
        if is_nrt:
            robot.send_cartesian_motion_force(
                pose_list,
                max_linear_vel=self.max_linear_vel,
                max_linear_acc=self.max_linear_acc,
                max_angular_vel=self.max_angular_vel,
                max_angular_acc=self.max_angular_acc,
            )
            return
        robot.stream_cartesian_motion_force(
            pose=pose_list,
            wrench=[0.0] * 6,
            velocity=[0.0] * 6,
            acceleration=[0.0] * 6,
        )
