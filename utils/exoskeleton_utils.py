#!/usr/bin/env python3
"""
Robot Utilities - joint angle processing and kinematics for robot control.

This module combines joint angle processing and forward kinematics functionality:
1. Joint angle processing from encoders (bias subtraction, normalization, unit conversion)
2. Forward kinematics computation for dual-arm air exoskeleton system
3. URDF model loading and validation
4. End-effector position and orientation calculation from joint angles

The module supports both individual arm control and dual-arm configuration.

Author: Zixi Ying
"""

import numpy as np
import os
from typing import Union, List, Optional, Tuple
from utils.shm_utils import get_dtype
from scipy.spatial.transform import Rotation as R

# Optional robotics toolbox import with fallback
try:
    import roboticstoolbox as rtb
    from scipy.spatial.transform import Rotation as RR
    ROBOTICS_AVAILABLE = True
except ImportError:
    ROBOTICS_AVAILABLE = False
    print("Warning: roboticstoolbox not available. Kinematics functionality will be disabled.")


class JointAngleProcessor:
    """
    A utility class for processing joint angles from encoders.

    This class handles:
    - Bias subtraction
    - Angle normalization (0-360 degrees)
    - Unit conversion (degrees to radians)
    - Angle range mapping (0-2π to -π to π)
    - Data type transformation
    """

    def __init__(self, bias: Union[np.ndarray, List[float]],
                 num_joints: int = 7,
                 target_dtype: Union[str, np.dtype] = np.float64):
        """
        Initialize the joint angle processor.

        Args:
            bias: Bias values to subtract from raw encoder readings
            num_joints: Number of joints to process
            target_dtype: Target data type for output arrays
        """
        self.bias = np.array(bias, dtype=np.float64)
        self.num_joints = num_joints
        self.target_dtype = target_dtype

        # Validate bias array
        if len(self.bias) < num_joints:
            raise ValueError(f"Bias array length ({len(self.bias)}) must be >= num_joints ({num_joints})")

    def process_raw_angles(self, raw_angles: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Process raw joint angles from encoder.

        This method performs the complete processing pipeline:
        1. Subtract bias
        2. Normalize angles to 0-360 range
        3. Convert to radians
        4. Map to -π to π range
        5. Convert to target data type

        Args:
            raw_angles: Raw joint angles from encoder (in degrees)

        Returns:
            Processed joint angles in radians with target data type
        """
        if raw_angles is None:
            return np.zeros(self.num_joints, dtype=get_dtype(self.target_dtype))

        # Convert to numpy array if needed
        if not isinstance(raw_angles, np.ndarray):
            raw_angles = np.array(raw_angles, dtype=np.float64)

        # Ensure we have enough joints
        if len(raw_angles) < self.num_joints:
            return np.zeros(self.num_joints, dtype=get_dtype(self.target_dtype))

        # Extract the required number of joints
        joint_pose = raw_angles[:self.num_joints].copy()

        # Step 1: Subtract bias
        joint_pose = joint_pose - self.bias[:self.num_joints]

        # Step 2: Normalize angles to 0-360 range
        joint_pose = np.where(joint_pose < 0, joint_pose + 360, joint_pose)

        # Step 3: Convert degrees to radians
        joint_pose_rad = joint_pose * np.pi / 180.0

        # Step 4: Map from [0, 2π] to [-π, π]
        joint_pose_rad = np.where(joint_pose_rad > np.pi, joint_pose_rad - 2*np.pi, joint_pose_rad)

        # Step 5: Convert to target data type
        result = joint_pose_rad.astype(get_dtype(self.target_dtype))

        return result

    def process_raw_angles_for_home(self, raw_angles: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Process raw joint angles specifically for home position initialization.

        This method is similar to process_raw_angles but optimized for home position
        where we might want to preserve the original precision.

        Args:
            raw_angles: Raw joint angles from encoder (in degrees)

        Returns:
            Processed joint angles in radians with target data type
        """
        return self.process_raw_angles(raw_angles)

    def get_zero_array(self) -> np.ndarray:
        """
        Get a zero array with the correct shape and data type.

        Returns:
            Zero array with shape (num_joints,) and target data type
        """
        return np.zeros(self.num_joints, dtype=get_dtype(self.target_dtype))

    def update_bias(self, new_bias: Union[np.ndarray, List[float]]) -> None:
        """
        Update the bias values.

        Args:
            new_bias: New bias values
        """
        self.bias = np.array(new_bias, dtype=np.float64)
        if len(self.bias) < self.num_joints:
            raise ValueError(f"New bias array length ({len(self.bias)}) must be >= num_joints ({self.num_joints})")

    def get_bias(self) -> np.ndarray:
        """
        Get current bias values.

        Returns:
            Current bias array
        """
        return self.bias.copy()


class ExoskeletonKinematics:
    """
    Exoskeleton kinematics class for dual-arm air exoskeleton system.

    This class provides forward kinematics computation for 7-DOF arms using URDF models.
    Supports individual arm control or dual-arm configuration.
    """

    def __init__(self, rbt_type="all", airexo_type="ver1"):
        print(f"Initializing ExoskeletonKinematics with rbt_type: {rbt_type} and airexo_type: {airexo_type}")
        """
        Initialize the exoskeleton kinematics.

        Args:
            rbt_type: Robot type ("left", "right", or "all")
            airexo_type: Version of airexo ("ver1" or "ver2")
        """
        if not ROBOTICS_AVAILABLE:
            raise ImportError("roboticstoolbox is required for kinematics functionality. "
                            "Install with: pip install roboticstoolbox-python")

        # Get project root directory path
        # Go up two levels from current file location to project root
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))

        # Build URDF file paths based on version
        if airexo_type == "ver1":
            left_urdf_path = os.path.join(project_root, "assets", "airexo_left.urdf")
            right_urdf_path = os.path.join(project_root, "assets", "airexo_right.urdf")
        elif airexo_type == "ver2":
            left_urdf_path = os.path.join(project_root, "assets", "airexo_left_0.7.urdf")
            right_urdf_path = os.path.join(project_root, "assets", "airexo_right_0.7.urdf")
        else:
            raise ValueError(f"Invalid airexo_type: {airexo_type}. Must be 'ver1' or 'ver2'")

        # Verify file existence
        if not os.path.exists(left_urdf_path):
            raise FileNotFoundError(f"Left arm URDF file not found: {left_urdf_path}")
        if not os.path.exists(right_urdf_path):
            raise FileNotFoundError(f"Right arm URDF file not found: {right_urdf_path}")

        # Load URDF model
        assert rbt_type in ["left", "right", "all"]
        if rbt_type == "left":
            self.robot_left = rtb.Robot.URDF(left_urdf_path)
            self.robot_right = None
        elif rbt_type == "right":
            self.robot_left = None
            self.robot_right = rtb.Robot.URDF(right_urdf_path)
        else:
            self.robot_left = rtb.Robot.URDF(left_urdf_path)
            self.robot_right = rtb.Robot.URDF(right_urdf_path)

        # Set end-effector link names based on version
        if airexo_type == "ver1":
            self.left_ee = "l_link7"  # Left arm end-effector link name
            self.right_ee = "r_link7"  # Right arm end-effector link name
        else:  # ver2
            self.left_ee = "l_end_effector"  # Left arm end-effector link name
            self.right_ee = "r_end_effector"  # Right arm end-effector link name

        # Set world-to-base rotation matrices based on version
        if airexo_type == "ver1":
            self.wb_R_left = R.from_euler('z', -np.pi/2) * R.from_euler('xyz', np.array([0.0, -45.0, -180.0]) * np.pi / 180.0)
            self.wb_R_right = R.from_euler('z', np.pi/2) * R.from_euler('xyz', np.array([0.0, 45.0, 180.0]) * np.pi / 180.0)
        else:  # ver2
            # self.wb_R_left = R.from_euler('z', -np.pi/2) * R.from_euler('xyz', np.array([-np.pi/2, 0, 0]))
            self.wb_R_left = R.from_euler('xyz', np.array([-np.pi/2, 0, 0]))
            # self.wb_R_right = R.from_euler('z', np.pi/2) * R.from_euler('xyz', np.array([np.pi/2, 0, 0]))
            self.wb_R_right = R.from_euler('xyz', np.array([np.pi/2, 0, 0]))

    def forward_kinematics_left(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Left arm exoskeleton forward kinematics.

        Args:
            joint_angles: 7 joint angles in radians

        Returns:
            Tuple of (position array [x,y,z], 3x3 rotation matrix)
        """
        if self.robot_left is None:
            raise RuntimeError("Left arm robot not initialized")

        if len(joint_angles) != 7:
            raise ValueError("Requires 7 joint angles")

        # Fill joint angles for left arm (7 joints)
        q = np.zeros(7)
        q[0:7] = joint_angles

        # Calculate forward kinematics
        pose = self.robot_left.fkine(q, start="l_base_link1", end=self.left_ee)

        return pose.t, pose.R  # position, rotation matrix

    def forward_kinematics_right(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Right arm exoskeleton forward kinematics.

        Args:
            joint_angles: 7 joint angles in radians

        Returns:
            Tuple of (position array [x,y,z], 3x3 rotation matrix)
        """
        if self.robot_right is None:
            raise RuntimeError("Right arm robot not initialized")

        if len(joint_angles) != 7:
            raise ValueError("Requires 7 joint angles")

        # Fill joint angles for right arm (7 joints)
        q = np.zeros(7)
        q[0:7] = joint_angles

        # Calculate forward kinematics
        pose = self.robot_right.fkine(q, start="r_base_link1", end=self.right_ee)

        return pose.t, pose.R  # position, rotation matrix



def create_processor_for_arm(arm_type: str, num_joints: int = 7,
                           target_dtype: Union[str, np.dtype] = np.float64,
                           airexo_type: str = "ver1") -> JointAngleProcessor:
    """
    Factory function to create a JointAngleProcessor for a specific arm type.

    Args:
        arm_type: Type of arm ("left", "right", or "dual")
        num_joints: Number of joints per arm
        target_dtype: Target data type for output arrays
        airexo_type: Version of airexo ("ver1" or "ver2")

    Returns:
        Configured JointAngleProcessor instance
    """
    if airexo_type == "ver1":
        # Original bias values
        if arm_type == "left":
            bias = np.array([126.03, 251.19, 323.26, 281.4, 314.38, 114.15, 61.52])
        elif arm_type == "right":
            bias = np.array([352.52, 152.57, 72.15, 352.4, 277.20, 358.7, 272.8])
        elif arm_type == "dual":
            # For dual arm, use the extended bias array
            bias = np.array([126.03, 251.19, 323.26, 281.4, 314.38, 114.15, 61.52,
                            352.52, 152.57, 72.15, 352.4, 277.20, 358.7, 272.8])
            num_joints = 14  # Override for dual arm
        else:
            raise ValueError(f"Invalid arm_type: {arm_type}. Must be 'left', 'right', or 'dual'")
    elif airexo_type == "ver2":
        # New bias values for version 2
        if arm_type == "left":
            bias = np.array([355.9, 1, 1.05, 0.81, 359, 356, 358])
        elif arm_type == "right":
            bias = np.array([3, 1.31, 359, 357, 3.49, 0.8, 358.5])
        elif arm_type == "dual":
            # For dual arm, use the extended bias array
            bias = np.array([355.9, 1, 1.05, 0.81, 359, 356, 358,
                            3, 1.31, 359, 357, 3.49, 0.8, 358.5])
            num_joints = 14  # Override for dual arm
        else:
            raise ValueError(f"Invalid arm_type: {arm_type}. Must be 'left', 'right', or 'dual'")
    else:
        raise ValueError(f"Invalid airexo_type: {airexo_type}. Must be 'ver1' or 'ver2'")

    return JointAngleProcessor(bias, num_joints, target_dtype)


# Usage example and backward compatibility
def create_exoskeleton_kinematics(rbt_type="all", airexo_type="ver1"):
    """
    Backward compatibility function for creating ExoskeletonKinematics.

    Args:
        rbt_type: Robot type ("left", "right", or "all")
        airexo_type: Version of airexo ("ver1" or "ver2")

    Returns:
        ExoskeletonKinematics instance
    """
    return ExoskeletonKinematics(rbt_type, airexo_type)

def process_gripper(raw_value: Union[float, int], arm_type: str, airexo_type: str = "ver1") -> float:
    """
    Process raw gripper values from encoder to normalized gripper width.

    Args:
        raw_value: Raw gripper value from encoder
        arm_type: Type of arm ("left" or "right")
        airexo_type: Version of airexo ("ver1" or "ver2")

    Returns:
        Normalized gripper width in meters (0.0 to 0.085)
    """
    if airexo_type == "ver1":
        # Original gripper settings
        if arm_type == "left":
            # Left arm: max 204 (fully open) -> 0.0, min 272 (fully closed) -> 0.085
            # opening process: 272 ------>  360 -> 0 ----------> 204
            max_raw = 204
            min_raw = 272
            max_width = 0.085
            min_width = 0.0
        elif arm_type == "right":
            # Right arm: max 22 (fully open) -> 0.0, min 60 (fully closed) -> 0.085
            # opening process: 60 ------>  360 -> 0 ----------> 22
            max_raw = 22
            min_raw = 60
            max_width = 0.085
            min_width = 0.0
        else:
            raise ValueError(f"Invalid arm_type: {arm_type}. Must be 'left' or 'right'")

        # Linear mapping from raw value to gripper width
        min_raw_norm = min_raw - 360

        if raw_value > min_raw:
            raw_value = raw_value - 360

    elif airexo_type == "ver2":
        # New gripper settings for version 2
        if arm_type == "left":
            # Left arm: min_raw is 152.22, max_raw is 24.43
            # min_raw_norm = min_raw - 360 = 152.22 - 360 = -207.78
            max_raw = 24.43
            min_raw = 150.22
            max_width = 0.085
            min_width = 0.0
        elif arm_type == "right":
            # Right arm: min_raw is 27.59, max_raw is 263.58
            # min_raw_norm = min_raw = 27.59
            max_raw = 263.58
            min_raw = 27.59
            max_width = 0.085
            min_width = 0.0
        else:
            raise ValueError(f"Invalid arm_type: {arm_type}. Must be 'left' or 'right'")

        # Set min_raw_norm based on arm type
        if arm_type == "left":
            min_raw_norm = min_raw - 360  # -207.78
            if raw_value > min_raw:
                raw_value = raw_value - 360
        else:  # right arm
            min_raw_norm = min_raw  # 27.59
            # raw_value stays constant for right arm
    else:
        raise ValueError(f"Invalid airexo_type: {airexo_type}. Must be 'ver1' or 'ver2'")

    # Calculate normalized width
    normalized_width = max_width + (raw_value - max_raw) * (min_width - max_width) / (min_raw_norm - max_raw)

    # Clip to valid range [0.0, 0.085]
    clipped_width = np.clip(normalized_width, 0.0, 0.085)

    return clipped_width


if __name__ == "__main__":
    print("Robot Utils Module - Testing functionality")
    print("=" * 50)

    # Test joint angle processor
    print("Testing Joint Angle Processor:")
    processor = create_processor_for_arm("left")
    raw_angles = [130, 250, 320, 280, 310, 110, 60]
    processed = processor.process_raw_angles(raw_angles)
    print(f"Raw angles: {raw_angles}")
    print(f"Processed angles: {processed}")
    print()

    # Test kinematics (if available)
    if ROBOTICS_AVAILABLE:
        print("Testing Exoskeleton Kinematics:")
        try:
            kin = ExoskeletonKinematics()

            # Left arm test
            left_pos, left_rot = kin.forward_kinematics_left([0.1, 0.2, 0, 0.5, 0, 0, 0])
            print(f"Left arm end-effector position: {left_pos}")
            print(f"Left arm rotation matrix shape: {left_rot.shape}")

            # Right arm test
            right_pos, right_rot = kin.forward_kinematics_right([-0.1, -0.2, 0, -0.5, 0, 0, 0])
            print(f"Right arm end-effector position: {right_pos}")
            print(f"Right arm rotation matrix shape: {right_rot.shape}")

        except Exception as e:
            print(f"Kinematics test failed: {e}")
    else:
        print("Kinematics testing skipped (roboticstoolbox not available)")
