#!/usr/bin/env python3
"""
Robot Device - Base robot class and PyBullet-based simulated robot.

Author: Jun Lv,Zixi Ying
"""

import time
import threading
import subprocess
import ctypes
import json
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import NoneType
import numpy as np
import multiprocessing.shared_memory as shm
from typing import Optional, Dict, Any, Union, List, Literal, Mapping, Tuple
import rerun as rr
from devices.base import BaseDevice
from devices.flexiv_adapter import FlexivAdapter
from utils.shm_utils import (
    pack_device_header, pack_buffer_header, pack_frame_header,
    calculate_device_shm_size, DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    get_control_frame_offset, get_control_frame_offsets,
    get_latest_control_frame_index, get_latest_control_frame_offset,
    get_dtype, create_control_shm, connect_to_control_shm, connect_to_device_shm,
    connect_to_shared_memory, unpack_device_header, unpack_buffer_header
)
from utils.logger_config import logger
from utils.time_control import precise_loop_timing
from utils.rizon_traj_planner_utils import (
    build_planner_start_kwargs,
    planner_utilized,
    traj_smooth_settings,
)
from utils.rizon_traj_run_recording import RizonTrajRunRecorder
from utils.lk_gripper_presets import DEFAULT_LK_GRIPPER_TYPE, get_lk_gripper_preset
from utils.safety_bounds import SafetyBounds
from utils.transform import compute_deviation, similarity_transform, axis_similarity_transform
from scipy.spatial.transform import Rotation as R
from utils.transform import scalar_transform

Gripper = None
LKGripperConfig = None
LK_GRIPPER_IMPORT_ERROR = None
REPO_ROOT = Path(__file__).resolve().parents[2]


def track_i2rt_inflight_call(source: str):
    """Decorate I2RT methods whose blocking duration should be watchdog-visible."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self._track_inflight_call(source):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator



from devices.robots.base import BaseRobot


def _validate_sim_stream_formats(robot: BaseRobot, arm_count: int) -> None:
    state = robot.state_feedback_config.format.primitives
    if len(state) != arm_count * 2:
        raise ValueError(
            "SimRobot state_feedback.format.type requires one joint-position and "
            "one joint-velocity primitive per arm"
        )
    joint_widths: list[int] = []
    for arm_index in range(arm_count):
        joint, velocity = state[arm_index * 2 : arm_index * 2 + 2]
        if (
            not joint.name.startswith("j")
            or not velocity.name.startswith("dq")
            or joint.width != velocity.width
        ):
            raise ValueError(
                "SimRobot state_feedback.format.type must pair jN-M with dqN-M "
                "for every arm"
            )
        joint_widths.append(joint.width)

    control = robot.control_config.format.primitives
    if robot.control_type == "joints":
        if len(control) != arm_count or any(
            not primitive.name.startswith("j")
            or primitive.width != joint_widths[index]
            for index, primitive in enumerate(control)
        ):
            raise ValueError(
                "SimRobot joint control requires one matching joint primitive per arm"
            )
        return

    if len(control) != arm_count * 2:
        raise ValueError(
            "SimRobot TCP control requires xyz plus one quaternion per arm"
        )
    for arm_index in range(arm_count):
        names = {
            primitive.name
            for primitive in control[arm_index * 2 : arm_index * 2 + 2]
        }
        if names not in ({"xyz", "qwqxqyqz"}, {"xyz", "qxqyqzqw"}):
            raise ValueError(
                "SimRobot TCP control requires xyz plus one quaternion per arm"
            )

class SimRobot(BaseRobot):
    """
    Simulated robot using PyBullet for physics simulation.
    Controls a robotic arm and provides joint state feedback.
    """

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: Mapping[str, Any],
        control: Mapping[str, Any],
        urdf_path=None,
        joint_indices=None,
        enable_gui: bool = False,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            control=control,
        )

        # Robot-specific attributes
        self.device_name = "SimRobot"
        self.urdf_path = urdf_path
        self.joint_indices = joint_indices
        self.enable_gui = enable_gui

        self.control_type = self.control_config.type
        if self.control_type not in {"joints", "eef_rel", "eef_abs"}:
            raise ValueError(
                f"SimRobot does not support control.type={self.control_config.type}"
            )
        _validate_sim_stream_formats(self, arm_count=1)

        # IK method selection (only PyBullet supported now)
        self.ik_method = "pybullet"

        # Recalculate SHM names with correct device_name
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"
        self.default_control_shm_name = f"{self.device_name}_{self.device_class_id}_control"

        # PyBullet simulation
        self.physics_client = None
        self.robot_id = None
        self.num_joints = 0
        self.joint_names = []
        self.joint_limits = []

        # Initialize PyBullet
        self._initialize_pybullet()

    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation."""
        try:
            import pybullet as p
            import pybullet_data

            # Connect to PyBullet
            if self.enable_gui:
                self.physics_client = p.connect(p.GUI)  # Use GUI for visualization
                logger.info("PyBullet GUI enabled for visualization")
            else:
                self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless simulation
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Set physics parameters
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1.0 / self.fps)

            # Load robot URDF
            if self.urdf_path is None:
                # Use default KUKA iiwa robot
                self.urdf_path = "kuka_iiwa/model.urdf"

            self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0])
            logger.info(f"Loaded robot URDF: {self.urdf_path}")

            # Get joint information
            self.num_joints = p.getNumJoints(self.robot_id)
            logger.info(f"Robot has {self.num_joints} joints")

            # Get joint names and limits
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                self.joint_names.append(joint_info[1].decode('utf-8'))

                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))

            # Set joint indices to control (default: all revolute joints)
            if self.joint_indices is None:
                self.joint_indices = []
                for i in range(self.num_joints):
                    joint_info = p.getJointInfo(self.robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.joint_indices.append(i)

            logger.info(f"Controlling joints: {self.joint_indices}")
            logger.info(f"Joint names: {[self.joint_names[i] for i in self.joint_indices]}")

            # Set initial joint positions to zero
            initial_positions = [0.0] * len(self.joint_indices)
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0)



            logger.info("PyBullet simulation initialized successfully")
            link_state = p.getLinkState(self.robot_id, 6, computeForwardKinematics=True)
            self.init_pose = np.concatenate([np.array(link_state[4]), np.array(link_state[5])]) # in world frame

        except Exception as e:
            logger.error(f"Error initializing PyBullet: {e}")
            raise

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state (joint positions and velocities).

        Returns:
            numpy.ndarray: Current robot state [joint_positions, joint_velocities]
        """
        try:
            import pybullet as p

            if self.robot_id is None:
                return None

            # Get joint states
            joint_states = p.getJointStates(self.robot_id, self.joint_indices)

            # Extract positions and velocities
            positions = [state[0] for state in joint_states]
            velocities = [state[1] for state in joint_states]

            # Combine into state array
            numpy_data_dtype = get_dtype(self.data_dtype)
            state_array = np.array(positions + velocities, dtype=numpy_data_dtype)

            # Reshape if needed
            if len(self.data_shape) > 0:
                state_array = state_array.reshape(self.data_shape)

            return state_array

        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return None

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command (set target joint positions).

        Args:
            control_array: Target joint positions
        """
        try:
            import pybullet as p

            if self.robot_id is None:
                return

            # BaseRobot exposes every configured endpoint in the dictionary
            # (including the robot-owned policy channel).  Consume only the
            # endpoint selected by the typed source-selection contract.
            selected_name = self._selected_source_name(
                self._selector_value_for_control_tick()
            )
            control_array = (
                control_dict.get(selected_name)
                if selected_name is not None
                else None
            )
            if control_array is None:
                p.stepSimulation()
                return
            # Ensure control array has correct shape
            if len(control_array.shape) > 0:
                control_data = control_array.flatten()
            else:
                control_data = [control_array]

            # If end-effector control is enabled, use inverse kinematics
            if self.control_type == "eef_abs":
                target_positions = self._compute_ik_iterative_single_arm(control_data, abs=True)
            elif self.control_type == "eef_rel":
                target_positions = self._compute_ik_iterative_single_arm(control_data, abs=False)
            else:
                target_positions = control_data

            # Limit the number of positions to the number of controlled joints
            num_controlled_joints = len(self.joint_indices)
            if len(target_positions) > num_controlled_joints:
                target_positions = target_positions[:num_controlled_joints]
            elif len(target_positions) < num_controlled_joints:
                # Pad with current positions if not enough targets provided
                current_states = p.getJointStates(self.robot_id, self.joint_indices)
                current_positions = [state[0] for state in current_states]
                target_positions.extend(current_positions[len(target_positions):])

            # Apply joint limits
            for i, (joint_idx, target_pos) in enumerate(zip(self.joint_indices, target_positions)):
                lower_limit, upper_limit = self.joint_limits[joint_idx]
                target_positions[i] = np.clip(target_pos, lower_limit, upper_limit)

            # Reset joint states
            for joint_idx, target_pos in zip(self.joint_indices, target_positions):
                p.resetJointState(self.robot_id, joint_idx, target_pos, 0.0)

            # Step simulation
            p.stepSimulation()

        except Exception as e:
            logger.error(f"Error executing control command: {e}")

    def _compute_ik_iterative_single_arm(self, control_data,abs=True):
        """
        Compute inverse kinematics iteratively using PyBullet's calculateInverseKinematics for single arm.

        Args:
            control_data: End-effector pose [pos, orn] where pos is [x, y, z] and orn is [x, y, z, w] (quaternion)

        Returns:
            target_positions: Joint positions for the arm
        """
        try:
            import pybullet as p

            # Assume control_data contains [pos(3), orn(4)] = 7 elements minimum
            if len(control_data) < 7:  # 3+4 = 7 minimum
                logger.warning("Insufficient control data for end-effector control")
                return control_data

            # Extract target pose
            target_pos = control_data[0:3]
            target_orn = control_data[3:7]
            target_pos,target_orn = compute_deviation([0,0,0],[0,0,0,0],target_pos, target_orn) if abs else compute_deviation(self.init_pose[0:3],self.init_pose[3:7],target_pos, target_orn)

            # Get end-effector link index (assume last link)
            end_effector_link = self.num_joints - 1

            # Iterative IK parameters - relaxed tolerances for better convergence
            max_iterations = 20  # Reduced outer iteration count
            position_tolerance = 1e-3  # Relaxed position tolerance to 1mm
            orientation_tolerance = 1e-2  # Relaxed orientation tolerance

            # Get current joint positions as initial guess
            current_joint_states = p.getJointStates(self.robot_id, self.joint_indices)
            q0 = [state[0] for state in current_joint_states]

            # Iterative IK with PyBullet
            for iteration in range(max_iterations):
                # Calculate IK with PyBullet
                joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    end_effector_link,
                    target_pos,
                    target_orn,
                    lowerLimits=[self.joint_limits[i][0] for i in self.joint_indices],
                    upperLimits=[self.joint_limits[i][1] for i in self.joint_indices],
                    jointRanges=[self.joint_limits[i][1] - self.joint_limits[i][0] for i in self.joint_indices],
                    restPoses=q0,
                    maxNumIterations=10,
                    residualThreshold=1e-5
                )

                # Convert to list and apply joint limits
                joint_positions = list(joint_positions[:len(self.joint_indices)])
                for i, joint_idx in enumerate(self.joint_indices):
                    if joint_idx < len(self.joint_limits):
                        lower_limit, upper_limit = self.joint_limits[joint_idx]
                        joint_positions[i] = np.clip(joint_positions[i], lower_limit, upper_limit)

                # Check error
                link_state = p.getLinkState(self.robot_id, end_effector_link, computeForwardKinematics=True)
                current_pos = link_state[4]
                current_orn = link_state[5]

                pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                dot_product = np.clip(np.dot(current_orn, target_orn), -1.0, 1.0)
                orn_error = 2 * np.sqrt(max(0.0, 1 - dot_product ** 2))

                # Check convergence - using relaxed tolerances
                if pos_error < position_tolerance:  # Focus mainly on position convergence
                    break

            # Print final orientation error
            if joint_positions is not None:
                # Calculate final error after IK iterations
                final_link_state = p.getLinkState(self.robot_id, end_effector_link, computeForwardKinematics=True)
                final_current_pos = final_link_state[4]
                final_current_orn = final_link_state[5]

                final_pos_error = np.linalg.norm(np.array(final_current_pos) - np.array(target_pos))
                final_dot_product = np.clip(np.dot(final_current_orn, target_orn), -1.0, 1.0)
                final_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_dot_product ** 2))
                final_orn_error_deg = np.degrees(final_orn_error_rad)

                # Take only the controlled joints
                controlled_positions = joint_positions[:len(self.joint_indices)]
                return list(controlled_positions)
            else:
                logger.warning("IK computation failed, using original control data")
                return control_data

        except Exception as e:
            logger.error(f"Error in iterative IK computation: {e}")
            return control_data

    def stop_server(self) -> None:
        """Stop the robot server and clean up PyBullet resources."""
        if self.physics_client is not None:
            import pybullet as p
            p.disconnect(self.physics_client)
            self.physics_client = None
            self.robot_id = None
            logger.info("PyBullet simulation disconnected")

        super().stop_server()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop_server()


class DualArmSimRobot(BaseRobot):
    """
    Dual-arm simulated robot using PyBullet for physics simulation.
    Controls two robotic arms and provides joint state feedback for both arms.
    """

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: Mapping[str, Any],
        control: Mapping[str, Any],
        urdf_path=None,
        joint_indices=None,
        enable_gui: bool = False,
        left_arm_position=(0.2, 0.0, 1.2),
        right_arm_position=(-0.2, 0.0, 1.2),
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            control=control,
        )

        # Robot-specific attributes
        self.device_name = "DualArmSimRobot"
        self.urdf_path = urdf_path
        self.joint_indices = joint_indices
        self.enable_gui = enable_gui
        self.left_arm_position = left_arm_position
        self.right_arm_position = right_arm_position

        self.control_type = self.control_config.type
        if self.control_type not in {"joints", "eef_rel", "eef_abs"}:
            raise ValueError(
                "DualArmSimRobot does not support "
                f"control.type={self.control_config.type}"
            )
        _validate_sim_stream_formats(self, arm_count=2)

        # IK method selection (only PyBullet supported now)
        self.ik_method = "pybullet"

        # RTB robot models removed - using PyBullet IK only

        # Recalculate SHM names with correct device_name
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"
        self.default_control_shm_name = f"{self.device_name}_{self.device_class_id}_control"

        # PyBullet simulation
        self.physics_client = None
        self.left_robot_id = None
        self.right_robot_id = None
        self.left_num_joints = 0
        self.right_num_joints = 0
        self.left_joint_names = []
        self.right_joint_names = []
        self.left_joint_limits = []
        self.right_joint_limits = []
        self.left_joint_indices = []
        self.right_joint_indices = []

        # Initialize PyBullet
        self._initialize_pybullet()
        # time.sleep(3)

    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation for dual arms."""
        try:
            import pybullet as p
            import pybullet_data

            # Connect to PyBullet
            if self.enable_gui:
                self.physics_client = p.connect(p.GUI)  # Use GUI for visualization
                logger.info("PyBullet GUI enabled for dual-arm visualization")
            else:
                self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless simulation
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Set physics parameters
            p.setGravity(0, 0, 0)
            p.setTimeStep(1.0 / self.fps)

            # Load robot URDF for both arms
            if self.urdf_path is None:
                # Use default Flexiv Rizon4 robot
                self.urdf_path = "./assets/robot/rizon/flexiv_Rizon4_kinematics.urdf"

            # Load left arm
            self.left_orientation = p.getQuaternionFromEuler([np.pi/2, -np.pi/2, 0])
            self.left_robot_id = p.loadURDF(self.urdf_path, self.left_arm_position, self.left_orientation)
            logger.info(f"Loaded left arm URDF: {self.urdf_path} at {self.left_arm_position}")

            # Load right arm
            self.right_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
            self.right_robot_id = p.loadURDF(self.urdf_path, self.right_arm_position, self.right_orientation)
            logger.info(f"Loaded right arm URDF: {self.urdf_path} at {self.right_arm_position}")

            # Get joint information for left arm
            self.left_num_joints = p.getNumJoints(self.left_robot_id)
            logger.info(f"Left arm has {self.left_num_joints} joints")

            # Get joint information for right arm
            self.right_num_joints = p.getNumJoints(self.right_robot_id)
            logger.info(f"Right arm has {self.right_num_joints} joints")

            # Get joint names and limits for left arm
            for i in range(self.left_num_joints):
                joint_info = p.getJointInfo(self.left_robot_id, i)
                self.left_joint_names.append(joint_info[1].decode('utf-8'))

                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.left_joint_limits.append((lower_limit, upper_limit))

            # Get joint names and limits for right arm
            for i in range(self.right_num_joints):
                joint_info = p.getJointInfo(self.right_robot_id, i)
                self.right_joint_names.append(joint_info[1].decode('utf-8'))

                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.right_joint_limits.append((lower_limit, upper_limit))

            # Set joint indices to control (default: all revolute joints)
            if self.joint_indices is None:
                # Left arm joint indices
                self.left_joint_indices = []
                for i in range(self.left_num_joints):
                    joint_info = p.getJointInfo(self.left_robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.left_joint_indices.append(i)

                # Right arm joint indices
                self.right_joint_indices = []
                for i in range(self.right_num_joints):
                    joint_info = p.getJointInfo(self.right_robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.right_joint_indices.append(i)
            else:
                # Use provided joint indices (assuming they're split between arms)
                mid_point = len(self.joint_indices) // 2
                self.left_joint_indices = self.joint_indices[:mid_point]
                self.right_joint_indices = self.joint_indices[mid_point:]

            logger.info(f"Controlling left arm joints: {self.left_joint_indices}")
            logger.info(f"Left arm joint names: {[self.left_joint_names[i] for i in self.left_joint_indices]}")
            logger.info(f"Controlling right arm joints: {self.right_joint_indices}")
            logger.info(f"Right arm joint names: {[self.right_joint_names[i] for i in self.right_joint_indices]}")

            # Set initial joint positions to zero for both arms
            left_initial_positions = [1] * len(self.left_joint_indices)
            right_initial_positions = [1] * len(self.right_joint_indices)

            for i, joint_idx in enumerate(self.left_joint_indices):
                p.resetJointState(self.left_robot_id, joint_idx, left_initial_positions[i], 0.0)

            for i, joint_idx in enumerate(self.right_joint_indices):
                p.resetJointState(self.right_robot_id, joint_idx,  right_initial_positions[i], 0.0)

            start_time = time.time()

            while time.time() < start_time+1:
                left_link_state = p.getLinkState(self.left_robot_id, 6, computeForwardKinematics=True)
                self.init_pose_left = np.concatenate(
                    [np.array(left_link_state[4]), np.array(left_link_state[5])])  # in world frame

                right_link_state = p.getLinkState(self.right_robot_id, 6, computeForwardKinematics=True)
                self.init_pose_right = np.concatenate(
                    [np.array(right_link_state[4]), np.array(right_link_state[5])])  # in world frame

            left_link_state = p.getLinkState(self.left_robot_id, 6, computeForwardKinematics=True)
            self.init_pose_left = np.concatenate([np.array(left_link_state[4]), np.array(left_link_state[5])]) # in world frame

            right_link_state = p.getLinkState(self.right_robot_id, 6, computeForwardKinematics=True)
            self.init_pose_right = np.concatenate([np.array(right_link_state[4]), np.array(right_link_state[5])]) # in world frame

            logger.info("Dual-arm PyBullet simulation initialized successfully")


        except Exception as e:
            logger.error(f"Error initializing dual-arm PyBullet: {e}")
            raise

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state (joint positions and velocities for both arms).

        Returns:
            numpy.ndarray: Current robot state [left_joint_positions, left_joint_velocities,
                          right_joint_positions, right_joint_velocities]
        """
        try:
            import pybullet as p

            if self.left_robot_id is None or self.right_robot_id is None:
                return None

            # Get joint states for left arm
            left_joint_states = p.getJointStates(self.left_robot_id, self.left_joint_indices)

            # Get joint states for right arm
            right_joint_states = p.getJointStates(self.right_robot_id, self.right_joint_indices)

            # Extract positions and velocities for left arm
            left_positions = [state[0] for state in left_joint_states]
            left_velocities = [state[1] for state in left_joint_states]

            # Extract positions and velocities for right arm
            right_positions = [state[0] for state in right_joint_states]
            right_velocities = [state[1] for state in right_joint_states]

            # Combine into state array: [left_pos, left_vel, right_pos, right_vel]
            numpy_data_dtype = get_dtype(self.data_dtype)
            state_array = np.array(left_positions + left_velocities + right_positions + right_velocities,
                                 dtype=numpy_data_dtype)

            # Reshape if needed
            if len(self.data_shape) > 0:
                state_array = state_array.reshape(self.data_shape)

            return state_array

        except Exception as e:
            logger.error(f"Error getting dual-arm robot state: {e}")
            return None

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command (set target joint positions for both arms).

        Args:
            control_array: Target joint positions for both arms
        """
        try:
            import pybullet as p

            if self.left_robot_id is None or self.right_robot_id is None:
                return

            selected_name = self._selected_source_name(
                self._selector_value_for_control_tick()
            )
            control_array = (
                control_dict.get(selected_name)
                if selected_name is not None
                else None
            )
            if control_array is None:
                p.stepSimulation()
                return

            # Ensure control array has correct shape
            if len(control_array.shape) > 0:
                control_data = control_array.flatten()
            else:
                control_data = [control_array]

            # If end-effector control is enabled, use inverse kinematics
            if self.control_type == "eef_abs":
                target_positions = self._compute_ik_iterative(control_data, abs=True)
            elif self.control_type == "eef_rel":
                target_positions = self._compute_ik_iterative(control_data, abs=False)
            else:
                target_positions = control_data

            # Split into left and right arm targets
            left_joint_count = len(self.left_joint_indices)
            left_target_positions = target_positions[:left_joint_count]
            right_target_positions = target_positions[left_joint_count:]

            # Reset joint states for left arm
            for joint_idx, target_pos in zip(self.left_joint_indices, left_target_positions):
                p.resetJointState(self.left_robot_id, joint_idx, target_pos, 0.0)

            # Reset joint states for right arm
            for joint_idx, target_pos in zip(self.right_joint_indices, right_target_positions):
                p.resetJointState(self.right_robot_id, joint_idx, target_pos, 0.0)

            # Step simulation
            p.stepSimulation()

        except Exception as e:
            logger.error(f"Error executing dual-arm control command: {e}")

    def _compute_ik_iterative(self, control_data,abs):
        """
        Compute inverse kinematics iteratively using PyBullet's calculateInverseKinematics.

        Args:
            control_data: End-effector poses for both arms [left_pos, left_orn, right_pos, right_orn]
                         where pos is [x, y, z] and orn is [x, y, z, w] (quaternion)

        Returns:
            target_positions: Joint positions for both arms
        """
        try:
            import pybullet as p
            # Assume control_data contains [left_pos(3), left_orn(4), right_pos(3), right_orn(4)]
            if len(control_data) < 14:  # 3+4+3+4 = 14 minimum
                logger.warning("Insufficient control data for end-effector control")
                return control_data

            left_target_pos = control_data[0:3]
            left_target_orn = control_data[3:7]
            left_target_pos, left_target_orn = compute_deviation(self.left_arm_position, self.left_orientation,
                                                                 left_target_pos, left_target_orn)  if abs else compute_deviation(self.init_pose_left[0:3],self.init_pose_left[3:7],left_target_pos, left_target_orn)

            right_target_pos = control_data[7:10]
            right_target_orn = control_data[10:14]
            right_target_pos, right_target_orn = compute_deviation(self.right_arm_position, self.right_orientation,
                                                                 right_target_pos, right_target_orn) if abs else compute_deviation(self.init_pose_right[0:3],self.init_pose_right[3:7],right_target_pos, right_target_orn)
            left_end_effector_link = 6
            right_end_effector_link = 6

            # Iterative IK parameters
            max_iterations = 20
            position_tolerance = 1e-3
            orientation_tolerance = 1e-2

            # Get current joint positions as initial guess
            left_current_joint_states = p.getJointStates(self.left_robot_id, self.left_joint_indices)
            left_q0 = [state[0] for state in left_current_joint_states]
            right_current_joint_states = p.getJointStates(self.right_robot_id, self.right_joint_indices)
            right_q0 = [state[0] for state in right_current_joint_states]

            # Compute IK for left arm with iteration
            for iteration in range(max_iterations):
                left_joint_positions = p.calculateInverseKinematics(
                    self.left_robot_id,
                    left_end_effector_link,
                    left_target_pos,
                    left_target_orn,
                    lowerLimits=[self.left_joint_limits[i][0] for i in self.left_joint_indices],
                    upperLimits=[self.left_joint_limits[i][1] for i in self.left_joint_indices],
                    jointRanges=[self.left_joint_limits[i][1] - self.left_joint_limits[i][0] for i in self.left_joint_indices],
                    restPoses=left_q0,  # Use current state as rest pose
                    maxNumIterations=10,  # Increase internal iterations per step
                    residualThreshold=1e-5  # Relaxed residual threshold
                )

                # Convert to list and apply joint limits
                left_joint_positions = list(left_joint_positions[:len(self.left_joint_indices)])
                for i, joint_idx in enumerate(self.left_joint_indices):
                    if joint_idx < len(self.left_joint_limits):
                        lower_limit, upper_limit = self.left_joint_limits[joint_idx]
                        left_joint_positions[i] = np.clip(left_joint_positions[i], lower_limit, upper_limit)

                # Check error
                left_link_state = p.getLinkState(self.left_robot_id, left_end_effector_link, computeForwardKinematics=True)
                left_current_pos = left_link_state[4]
                left_current_orn = left_link_state[5]

                left_pos_error = np.linalg.norm(np.array(left_current_pos) - np.array(left_target_pos))
                left_dot_product = np.clip(np.dot(left_current_orn, left_target_orn), -1.0, 1.0)
                left_orn_error = 2 * np.sqrt(max(0.0, 1 - left_dot_product ** 2))

                # Update initial guess
                left_q0 = left_joint_positions[:len(self.left_joint_indices)]

                # Check convergence
                if left_pos_error < position_tolerance:
                    break

            # Compute IK for right arm with iteration
            for iteration in range(max_iterations):
                right_joint_positions = p.calculateInverseKinematics(
                    self.right_robot_id,
                    right_end_effector_link,
                    right_target_pos,
                    right_target_orn,
                    lowerLimits=[self.right_joint_limits[i][0] for i in self.right_joint_indices],
                    upperLimits=[self.right_joint_limits[i][1] for i in self.right_joint_indices],
                    jointRanges=[self.right_joint_limits[i][1] - self.right_joint_limits[i][0] for i in self.right_joint_indices],
                    restPoses=right_q0,
                    maxNumIterations=10,
                    residualThreshold=1e-5
                )

                # Convert to list and apply joint limits
                right_joint_positions = list(right_joint_positions[:len(self.right_joint_indices)])
                for i, joint_idx in enumerate(self.right_joint_indices):
                    if joint_idx < len(self.right_joint_limits):
                        lower_limit, upper_limit = self.right_joint_limits[joint_idx]
                        right_joint_positions[i] = np.clip(right_joint_positions[i], lower_limit, upper_limit)

                # Check error
                right_link_state = p.getLinkState(self.right_robot_id, right_end_effector_link, computeForwardKinematics=True)
                right_current_pos = right_link_state[4]
                right_current_orn = right_link_state[5]
                right_pos_error = np.linalg.norm(np.array(right_current_pos) - np.array(right_target_pos))
                right_dot_product = np.clip(np.dot(right_current_orn, right_target_orn), -1.0, 1.0)
                right_orn_error = 2 * np.sqrt(max(0.0, 1 - right_dot_product ** 2))

                # Update initial guess
                right_q0 = right_joint_positions[:len(self.right_joint_indices)]

                # Check convergence
                if right_pos_error < position_tolerance:
                    logger.debug(f"Right arm IK converged at iteration {iteration+1}, pos_error: {right_pos_error:.6f}")
                    break

            # Print final orientation errors for both arms
            if left_joint_positions is not None and right_joint_positions is not None:
                # Calculate final errors for left arm
                final_left_link_state = p.getLinkState(self.left_robot_id, left_end_effector_link, computeForwardKinematics=True)
                final_left_current_orn = final_left_link_state[5]
                final_left_dot_product = np.clip(np.dot(final_left_current_orn, left_target_orn), -1.0, 1.0)
                final_left_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_left_dot_product ** 2))
                final_left_orn_error_deg = np.degrees(final_left_orn_error_rad)

                # Calculate final errors for right arm
                final_right_link_state = p.getLinkState(self.right_robot_id, right_end_effector_link, computeForwardKinematics=True)
                final_right_current_orn = final_right_link_state[5]
                final_right_dot_product = np.clip(np.dot(final_right_current_orn, right_target_orn), -1.0, 1.0)
                final_right_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_right_dot_product ** 2))
                final_right_orn_error_deg = np.degrees(final_right_orn_error_rad)
                # Take only the controlled joints
                left_controlled = left_joint_positions[:len(self.left_joint_indices)]
                right_controlled = right_joint_positions[:len(self.right_joint_indices)]
                return list(left_controlled) + list(right_controlled)
            else:
                logger.warning("IK computation failed, using original control data")
                return control_data

        except Exception as e:
            logger.error(f"Error in iterative IK computation: {e}")
            return control_data



    def stop_server(self) -> None:
        """Stop the dual-arm robot server and clean up resources."""
        self.running = False

        # Clean up PyBullet
        if self.physics_client is not None:
            try:
                import pybullet as p
                p.disconnect()
                logger.info("Disconnected from PyBullet")
            except:
                pass
            self.physics_client = None

        # Clean up shared memory
        self._cleanup_shared_memory()
        self._cleanup_control_shared_memory()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.stop_server()


def sim_main() -> None:
    """Main function for testing SimRobot."""
    import argparse

    parser = argparse.ArgumentParser(description='Simulated Robot Device')
    parser.add_argument('--device_class_id', type=int, default=0, help='Device ID')
    parser.add_argument('--urdf_path', type=str, default=None, help='Path to robot URDF file')
    parser.add_argument('--fps', type=float, default=100.0, help='Control frequency (Hz)')
    parser.add_argument('--buffer_size', type=int, default=100, help='State buffer size')
    parser.add_argument('--control_buffer_size', type=int, default=50, help='Control buffer size')
    parser.add_argument('--enable_gui', action='store_true', help='Enable PyBullet GUI for visualization')

    args = parser.parse_args()

    # Create and start simulated robot
    robot = SimRobot(
        device_class_id=args.device_class_id,
        state_feedback={
            "fps": args.fps,
            "buffer_size": args.buffer_size,
            "format": {
                "shape": [14],
                "dtype": "float64",
                "type": ["j1-7", "dq1-7"],
            },
        },
        control={
            "type": "joints",
            "buffer_size": args.control_buffer_size,
            "execution": {"fps": args.fps},
            "format": {
                "shape": [7],
                "dtype": "float64",
                "type": ["j1-7"],
            },
        },
        urdf_path=args.urdf_path,
        enable_gui=args.enable_gui,
    )

    try:
        robot.start_server()
    except KeyboardInterrupt:
        logger.info("Stopping robot...")
    finally:
        robot.stop_server()
