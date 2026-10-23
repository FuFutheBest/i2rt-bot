#!/usr/bin/env python3
"""Robot adapter implementation.

Author: Jun Lv, Zixi Ying, OpenAI Assistant
"""

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Union

import numpy as np

from utils.logger_config import logger

from devices.flexiv_adapter import FlexivAdapter
from utils.rizon_traj_planner_utils import (
    planner_utilized,
)
from utils.rizon_traj_run_recording import RizonTrajRunRecorder
from devices.robots.base import BaseRobot
from devices.robots.rizon.config import (
    RizonGripperConfig,
    RizonMotionLimits,
    validate_control_layouts,
    validate_state_schema,
)
from devices.robots.rizon.gripper import RizonGripperMixin
from devices.robots.rizon.planner import RizonPlannerMixin
from devices.robots.rizon.home import RizonHomeMixin
from devices.robots.rizon.connection import RizonConnectionMixin
from devices.robots.rizon.commands import RizonCommandMixin
from devices.robots.rizon.safety import RizonSafetyMixin

class RizonRobot(
    RizonPlannerMixin,
    RizonGripperMixin,
    RizonHomeMixin,
    RizonConnectionMixin,
    RizonCommandMixin,
    RizonSafetyMixin,
    BaseRobot,
):
    """
    Rizon Robot Device - Flexiv Rizon robot interface for SuperInference.

    Inherits from BaseRobot and provides real-time robot state monitoring and joint position control.
    """

    def __init__(
        self,
        device_class_id: int,
        robot_sn: str | Sequence[str],
        state_feedback: Mapping[str, Any],
        control: Mapping[str, Any],
        *,
        gripper: Optional[Mapping[str, Any]] = None,
        motion_limits: Optional[Mapping[str, Any]] = None,
        home: Optional[Mapping[str, Any]] = None,
        network_interface_whitelist: Optional[Sequence[str]] = None,
        signal_files: Optional[Dict[str, str]] = None,
        traj_run_recording: Optional[Dict[str, Any]] = None,
        cartesian_planner: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a Rizon from structured state, control, and hardware sections."""
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            control=control,
        )
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._signal_files = signal_files
        self._robot_home_command_pattern = signal_files.get('robot_home_command_pattern')
        self._policy_runner_reset_command = signal_files.get('policy_runner_reset_command')
        self._reset_signal = signal_files.get('reset_signal')

        serial_numbers = (
            tuple(str(item) for item in robot_sn)
            if isinstance(robot_sn, Sequence) and not isinstance(robot_sn, (str, bytes))
            else (str(robot_sn),)
        )
        if len(serial_numbers) not in {1, 2}:
            raise ValueError("robot_sn must identify one or two Rizon arms")
        self.robot_sn = list(serial_numbers) if len(serial_numbers) == 2 else serial_numbers[0]
        self.is_dual_arm = len(serial_numbers) == 2
        arm_count = len(serial_numbers)
        self.network_interface_whitelist = list(network_interface_whitelist or [])
        if self.state_feedback_config is None or self.control_config is None:
            raise ValueError("RizonRobot requires state_feedback and control mappings")
        self.state_schema = validate_state_schema(
            self.state_feedback_config.format, arm_count
        )
        self.state_layouts = self.state_schema.arm_layouts
        self.control_type = self.control_config.type
        self.control_layouts = validate_control_layouts(
            self.control_config.type, self.control_config.format, arm_count
        )
        # Policy control timestamp - used to filter stale actions after control mode switch
        self._policy_control_start_timestamp = 0
        # Grace period (in nanoseconds) to wait for fresh policy actions after control switch
        # This prevents robot from executing old actions while waiting for new policy actions
        self._policy_control_grace_period_ns = int(0.5 * 1e9)  # 0.5 seconds

        gripper_required_by_arm = [
            state_layout.find("gripper") is not None
            or control_layout.find("gripper") is not None
            for state_layout, control_layout in zip(
                self.state_layouts, self.control_layouts
            )
        ]
        self.gripper_config = RizonGripperConfig.from_mapping(
            gripper, gripper_required_by_arm=gripper_required_by_arm
        )
        self.utilize_gripper = any(
            arm_config.enabled for arm_config in self.gripper_config.arms
        )
        self.gripper_calibration_data = None
        if self.gripper_config.inference_calibration_path is not None:
            self._load_gripper_calibration_data()

        self.impedance_parameter = self.control_config.cartesian_impedance
        if self.control_type in ["eef_abs", "eef_rel"]:
            if self.impedance_parameter is not None:
                logger.info(f"Cartesian impedance configured: {self.impedance_parameter}")
            else:
                # Default values: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
                self.impedance_parameter = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]
                logger.info(f"Using default cartesian : {self.impedance_parameter}")
        else:
            # Joint control mode - impedance parameters not used
            self.impedance_parameter = None
            logger.info("Joint control mode - Impedance parameters not applicable")

        self.motion_limits = RizonMotionLimits.from_mapping(
            motion_limits, arm_count=arm_count
        )
        self.max_velocity = list(self.motion_limits.joint_velocity)
        self.max_acceleration = list(self.motion_limits.joint_acceleration)
        if self.control_type in ["eef_abs", "eef_rel"]:
            self.max_linear_vel = self.motion_limits.tcp_linear_velocity
            self.max_linear_acc = self.motion_limits.tcp_linear_acceleration
            self.max_angular_vel = self.motion_limits.tcp_angular_velocity
            self.max_angular_acc = self.motion_limits.tcp_angular_acceleration
            logger.info(f"Motion limits configured - Linear: vel={self.max_linear_vel} m/s, acc={self.max_linear_acc} m/s²")
            logger.info(f"Motion limits configured - Angular: vel={self.max_angular_vel} rad/s, acc={self.max_angular_acc} rad/s²")
        else:
            # Joint control mode - motion limits not used
            self.max_linear_vel = None
            self.max_linear_acc = None
            self.max_angular_vel = None
            self.max_angular_acc = None
            logger.info("Joint control mode - Motion limits not applicable")

        self.safety_bounds = self.motion_limits.workspace
        self._safety_clip_last_log_time: Dict[str, float] = {}
        self.max_contact_wrench = (
            list(self.motion_limits.max_tcp_contact_wrench)
            if self.control_type in ["eef_abs", "eef_rel"]
            and self.motion_limits.max_tcp_contact_wrench is not None
            else None
        )
        self._last_operational_status: Any = None
        self._apply_lk_gripper_calibration_files()
        self.last_gripper_value = [
            self._get_home_gripper_feedback_value(index)
            for index in range(len(self.gripper_config.arms))
        ]
        self.last_gripper_command_width = [
            self._get_home_gripper_width(index)
            for index in range(len(self.gripper_config.arms))
        ]
        self._last_commanded_tcp_by_robot: Dict[int, List[float]] = {}

        # Robot interfaces
        if self.is_dual_arm:
            self.robot_left = None
            self.robot_right = None
            self.left_sn, self.right_sn = serial_numbers
            logger.info(f"Initializing dual-arm Rizon robots: {self.left_sn}, {self.right_sn}")
        else:
            self.robot = None
            logger.info(f"Initializing single-arm Rizon robot: {serial_numbers[0]}")

        # Gripper interfaces
        self.gripper = None
        self.gripper_left = None
        self.gripper_right = None
        self.tool = None
        self.tool_left = None
        self.tool_right = None

        # Robot status
        self.is_operational = False
        self.is_enabled = False
        self.current_mode = None

        # Parse home configuration
        self.home_joints = None
        self.home_joints_left = None
        self.home_joints_right = None
        self.home_presets = None
        self.home_preset_index = None
        self.home_pose = None
        self.home_pose_left = None
        self.home_pose_right = None

        self.current_point_time = 0.0
        self._planner_waypoint_index = 0
        self.real_time = 0.0
        self.num_clock_ticks = 0
        self.command_period_s = 1.0 / float(self.command_fps)

        # Process home configuration
        if home is not None and isinstance(home, dict):
            self._configure_home(home)

        # Cartesian motion planner and trajectory recording are legacy-async-only features.
        # Both are force-disabled unless legacy_async_mode is enabled, even if their config
        # blocks are present.
        self._cartesian_planner_cfg: Dict[str, Any] = dict(cartesian_planner or {})
        self.use_cartesian_motion_planner = (
            self.legacy_async_mode and planner_utilized(self._cartesian_planner_cfg)
        )

        # BaseRobot.__del__ dispatches to stop_server when construction fails, so
        # establish its recorder state before loading the optional Flexiv SDK.
        self._planner_trace_perf_origin: Optional[float] = None
        self._traj_rec: Optional[RizonTrajRunRecorder] = None

        # Flexiv SDK adapter (keeps direct flexivrdk dependency out of this module)
        self.flexiv_adapter = (
            FlexivAdapter(backend="pybind") if self.use_cartesian_motion_planner else FlexivAdapter()
        )

        # Optional sparse / samples JSONL + merge (``utils/rizon_traj_run_recording``).
        if self.legacy_async_mode:
            self._traj_rec = RizonTrajRunRecorder.maybe_create(
                traj_run_recording,
                self._cartesian_planner_cfg,
                self.control_type,
            )
        # Throttled Cartesian planner diagnostics (queue / executed / trajectory counts).
        self._planner_diag_interval_s: float = 0.5
        self._planner_diag_last_perf: float = 0.0
        self._planner_diag_prev_queue_left: Optional[int] = None
        self._planner_diag_prev_queue_right: Optional[int] = None

    @staticmethod
    def _flexiv_pose_to_xyzw_pose(pose: Union[List[float], np.ndarray]) -> List[float]:
        """Convert Flexiv pose [x,y,z,qw,qx,qy,qz] to [x,y,z,qx,qy,qz,qw]."""
        p = np.asarray(pose, dtype=float).reshape(-1)
        return [float(p[0]), float(p[1]), float(p[2]), float(p[4]), float(p[5]), float(p[6]), float(p[3])]

    @staticmethod
    def _xyzw_pose_to_flexiv_pose(pose: Union[List[float], np.ndarray]) -> List[float]:
        """Convert [x,y,z,qx,qy,qz,qw] to Flexiv pose [x,y,z,qw,qx,qy,qz]."""
        p = np.asarray(pose, dtype=float).reshape(-1)
        return [float(p[0]), float(p[1]), float(p[2]), float(p[6]), float(p[3]), float(p[4]), float(p[5])]

    def start_server(self) -> bool:
        """Start the Rizon robot server."""
        try:
            # Connect to robot
            if not self._connect_robot():
                logger.error("Failed to connect to robot")
                return False

            # Switch to appropriate control mode
            if self.control_type == "joints":
                if not self._switch_to_joint_control():
                    logger.error("Failed to switch to joint control mode")
                    return False
            elif self.control_type in ["eef_abs", "eef_rel"]:
                if not self._switch_to_eef_control():
                    logger.error("Failed to switch to end-effector control mode")
                    return False

                # Set home pose for EEF control (from config or current pose)
                if not self._set_home_pose():
                    logger.error("Failed to set home pose")
                    return False

                # Always go to home position for EEF control
                logger.info("Moving to home position for EEF control...")
                if not self._go_home():
                    logger.error("Failed to go to home position")
                    return False

            # Start base robot server
            if not self._start_dual_cartesian_planners_if_configured():
                return False
            self._start_traj_recording_threads_if_configured()
            super().start_server()
            logger.info(f"Rizon robot device {self.device_class_id} started with {self.control_type} control")
            return True

        except Exception as e:
            logger.error(f"Failed to start Rizon robot device: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the Rizon robot server."""
        try:
            self._stop_traj_recording_threads()
            self._write_dense_planner_trace_at_shutdown()
            self._stop_dual_cartesian_planners_if_configured()
            if self.utilize_gripper:
                self._shutdown_grippers()

            # Disable robot(s) if connected
            if self.is_dual_arm:
                if self.robot_left and self.is_enabled:
                    try:
                        self.robot_left.Disable()
                        logger.info("Left robot disabled")
                    except Exception:
                        pass
                if self.robot_right and self.is_enabled:
                    try:
                        self.robot_right.Disable()
                        logger.info("Right robot disabled")
                    except Exception:
                        pass
            else:
                if self.robot and self.is_enabled:
                    try:
                        self.robot.Disable()
                        logger.info("Robot disabled")
                    except Exception:
                        pass

            # Stop base robot server
            super().stop_server()
            self._close_traj_jsonl_sinks()
            self._maybe_write_traj_run_record_json()
            logger.info(f"Rizon robot device {self.device_class_id} stopped")

        except Exception as e:
            logger.error(f"Error stopping Rizon robot device: {e}")

    def get_status(self) -> dict:
        """Get device status information."""
        status = super().get_status()
        status.update({
            'robot_sn': self.robot_sn,
            'control_type': self.control_type,
            'is_dual_arm': self.is_dual_arm,
            'is_operational': self.is_operational,
            'is_enabled': self.is_enabled,
            'current_mode': self.current_mode,
            'gripper_backend': self.gripper_config.backend if self.utilize_gripper else None,
            'gripper_value_mode': self.gripper_config.value_mode if self.utilize_gripper else None,
            'gripper_ports': [arm.port for arm in self.gripper_config.arms] if self.utilize_gripper else None,
            'gripper_motor_ids': [arm.motor_id for arm in self.gripper_config.arms] if self.utilize_gripper else None,
            'gripper_home_ratios': [arm.home_ratio for arm in self.gripper_config.arms] if self.utilize_gripper else None,
            'gripper_home_widths': [arm.home_width for arm in self.gripper_config.arms] if self.utilize_gripper else None,
            'gripper_scale_factors': [arm.scale_factor for arm in self.gripper_config.arms] if self.utilize_gripper else None,
            'control_shm_connected': all(shm is not None for shm in self.control_shared_memory.values()) if self.control_shared_memory else False,
            'home_pose_set': (self.home_pose is not None) if not self.is_dual_arm else
                           (self.home_pose_left is not None and self.home_pose_right is not None),
            'impedance_parameter': self.impedance_parameter if self.control_type in ["eef_abs", "eef_rel"] else "Not applicable (joint mode)",
            'motion_limits': {
                'max_linear_vel': self.max_linear_vel,
                'max_linear_acc': self.max_linear_acc,
                'max_angular_vel': self.max_angular_vel,
                'max_angular_acc': self.max_angular_acc
            } if self.control_type in ["eef_abs", "eef_rel"] else "Not applicable (joint mode)",
            'legacy_async_mode': self.legacy_async_mode,
            'use_cartesian_motion_planner': self.use_cartesian_motion_planner,
        })
        return status
