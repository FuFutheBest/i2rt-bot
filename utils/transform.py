#!/usr/bin/env python3
"""
Transform Utilities - coordinate transformation and pose computation functions.

This module:
1. Computes deviation between world-base and base-end-effector transformations
2. Provides similarity transformation with axis flipping
3. Handles position and quaternion conversions between coordinate frames
4. Supports both list and numpy array input formats

Used for robot pose calculations and coordinate frame transformations.

Author: Zixi Ying
"""
import numpy as np
import transforms3d as t3d
from typing import Union, Tuple, List, Optional

def process_quaternion(quat: Union[List[float], np.ndarray], 
                       conversion_type: str) -> Union[List[float], np.ndarray]:
    """Processes the quaternion to the desired format.

    Args:
        quat: The quaternion to process
        conversion_type: Format conversion type of the quaternion.
            Options:
            - "f2l": Scalar-first (wxyz) to scalar-last (xyzw)
            - "l2f": Scalar-last (xyzw) to scalar-first (wxyz)

    Returns:
        The processed quaternion in the desired format.
        Returns the same type as input (list or numpy array).

    Raises:
        ValueError: If invalid conversion type is provided
    """
    if conversion_type not in ['f2l', 'l2f']:
        raise ValueError("Invalid quaternion conversion type. Must be 'f2l' or 'l2f'")
    
    if conversion_type == 'l2f':
        return [quat[3], quat[0], quat[1], quat[2]]
    else:  # conversion_type == 'f2l'
        return [quat[1], quat[2], quat[3], quat[0]]


def _ensure_numpy_array(data: Union[List[float], np.ndarray, None], 
                        default_shape: tuple) -> np.ndarray:
    """Ensures input is a numpy array with proper shape.
    
    Args:
        data: Input data to convert
        default_shape: Shape to use if data is None
        
    Returns:
        Numpy array with proper shape
    """
    if data is None:
        return np.zeros(default_shape)
    elif isinstance(data, list):
        return np.array(data)
    return data


def _ensure_quaternion_format(quat: Union[List[float], np.ndarray], 
                             scalar_first: bool) -> np.ndarray:
    """Ensures quaternion is in scalar-last format for transforms3d.
    
    Args:
        quat: Input quaternion
        scalar_first: Whether input is in scalar-first format
        
    Returns:
        Quaternion in scalar-last format as numpy array
    """
    quat_array = _ensure_numpy_array(quat, (4,))
    if not scalar_first:
        quat_array = np.array(process_quaternion(quat_array, 'l2f'))
    return quat_array


def compute_inverse(pos: Union[List[float], np.ndarray], 
                   quat: Union[List[float], np.ndarray], 
                   scalar_first_quat: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the inverse transformation for a given pose (position and quaternion).

    Args:
        pos: The 3D position vector
        quat: The quaternion representing the rotation
        scalar_first_quat: If True, the quaternion is in scalar-first format
        
    Returns:
        Tuple containing:
        - The 3D position vector after the inverse transformation
        - The quaternion after the inverse transformation
    """
    pos_array = _ensure_numpy_array(pos, (3,))
    quat_array = _ensure_quaternion_format(quat, scalar_first_quat)
    
    # Compute inverse rotation
    inv_rot_mat = t3d.quaternions.quat2mat(quat_array).T
    inv_quat = t3d.quaternions.mat2quat(inv_rot_mat)

    if not scalar_first_quat:
        inv_quat = process_quaternion(inv_quat, 'f2l')
    
    # Compute inverse position
    inv_pos = -np.dot(inv_rot_mat, pos_array)
    
    return inv_pos, inv_quat


def _create_transform_matrix(pos: Union[List[float], np.ndarray, None],
                           quat: Union[List[float], np.ndarray, None],
                           scalar_first_quat: bool) -> np.ndarray:
    """Creates a 4x4 homogeneous transformation matrix from position and quaternion.
    
    Args:
        pos: 3D position vector
        quat: Quaternion orientation
        scalar_first_quat: Whether quaternion is in scalar-first format
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    pos_array = _ensure_numpy_array(pos, (3,))
    
    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, 3] = pos_array
    
    if quat is not None:
        quat_array = _ensure_quaternion_format(quat, scalar_first_quat)
        rot_mat = t3d.quaternions.quat2mat(quat_array)
        transform[:3, :3] = rot_mat
    
    return transform


def compute_deviation(
    a_pos: Union[List[float], np.ndarray, None],
    a_quat: Union[List[float], np.ndarray, None],
    b_pos: Union[List[float], np.ndarray, None],
    b_quat: Union[List[float], np.ndarray, None],
    first_inv: bool = False,
    last_inv: bool = False,
    scalar_first_quat: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the compound transformation between two poses.

    Args:
        a_pos: 3D position of frame A relative to reference frame
        a_quat: Quaternion orientation of frame A
        b_pos: 3D position of frame B relative to frame A
        b_quat: Quaternion orientation of frame B
        first_inv: If True, inverts the first transformation matrix (A)
        last_inv: If True, inverts the second transformation matrix (B)
        scalar_first_quat: If True, the quaternion is in scalar-first format
        
    Returns:
        Tuple containing:
        - The 3D position vector of the compound transformation
        - The quaternion of the compound transformation
    """
    # Create transformation matrices
    a_T = _create_transform_matrix(a_pos, a_quat, scalar_first_quat)
    b_T = _create_transform_matrix(b_pos, b_quat, scalar_first_quat)
    
    # Apply inversions if requested
    if first_inv:
        a_T = np.linalg.inv(a_T)
    if last_inv:
        b_T = np.linalg.inv(b_T)
    
    # Compute compound transformation
    compound_T = a_T @ b_T
    
    # Extract position and orientation
    position = compound_T[:3, 3]
    quaternion = t3d.quaternions.mat2quat(compound_T[:3, :3])

    if not scalar_first_quat:
        quaternion = process_quaternion(quaternion, 'f2l')
    
    return position, quaternion


def compute_deviation_3(
    wh_pos: Union[List[float], np.ndarray, None],
    wh_quat: Union[List[float], np.ndarray, None],
    wd_pos: Union[List[float], np.ndarray, None],
    wd_quat: Union[List[float], np.ndarray, None],
    wr_pos: Union[List[float], np.ndarray, None],
    wr_quat: Union[List[float], np.ndarray, None],
    wc_pos: Union[List[float], np.ndarray, None],
    wc_quat: Union[List[float], np.ndarray, None],
    scalar_first_quat: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the compound transformation for robot pause operation.

    This function computes the transformation required to determine the robot's pose
    after a commanded pause, using the following formula:
    
    T = inv(T_wh) @ T_wd @ inv(T_wr) @ T_wc
    
    The resulting transformation represents the offset needed to properly resume
    operation after a pause.

    Args:
        wh_pos: 3D position of the home pose in world coordinates
        wh_quat: Quaternion of the home pose orientation
        wd_pos: 3D position of the detachment pose in world coordinates
        wd_quat: Quaternion of the detachment pose orientation
        wr_pos: 3D position of the resume pose in world coordinates
        wr_quat: Quaternion of the resume pose orientation
        wc_pos: 3D position of the current pose in world coordinates
        wc_quat: Quaternion of the current pose orientation
        scalar_first_quat: If True, the quaternion is in scalar-first format
        
    Returns:
        Tuple containing:
        - 3D position vector of the compound transformation
        - Quaternion of the compound transformation
    """
    # Create transformation matrices
    wh_T = _create_transform_matrix(wh_pos, wh_quat, scalar_first_quat)
    wd_T = _create_transform_matrix(wd_pos, wd_quat, scalar_first_quat)
    wr_T = _create_transform_matrix(wr_pos, wr_quat, scalar_first_quat)
    wc_T = _create_transform_matrix(wc_pos, wc_quat, scalar_first_quat)
    
    # Compute compound transformation
    T = np.linalg.inv(wh_T) @ wd_T @ np.linalg.inv(wr_T) @ wc_T
    
    # Extract position and orientation
    position = T[:3, 3]
    quaternion = t3d.quaternions.mat2quat(T[:3, :3])
    
    if not scalar_first_quat:
        quaternion = process_quaternion(quaternion, 'f2l')
    
    return position, quaternion

def axis_similarity_transform(pos: Union[list, np.ndarray, None], quat: Union[list, np.ndarray, None], axis: str = "xyz",scalar_first_quat:bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a similarity transformation with axis flipping.

    This function achieves axis flipping by applying a special transformation matrix `dT`,
    where the diagonal elements of `dT` are `(-1, -1, 1, 1)`. The transformation formula is:
    `T_out = dT @ T @ dT`. This is typically used to mirror a coordinate system (e.g., a
    right-hand coordinate system) to another (e.g., a left-hand coordinate system).

    Args:
        pos (np.ndarray): The 3D position vector.
        quat (np.ndarray): The quaternion representing the rotation.
        axis: The axis to flip.
        scalar_first_quat: If True, the quaternion is in scalar-first format.
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - np.ndarray: The 3D position vector after the similarity transformation.
            - np.ndarray: The quaternion after the similarity transformation.
    """
    dT = np.eye(4)
    if "x" in axis:
        dT[0, 0] = -1.
    if "y" in axis:
        dT[1, 1] = -1.
    if "z" in axis:
        dT[2, 2] = -1.
        
    T = _create_transform_matrix(pos, quat, scalar_first_quat)
    T_out = dT @ T @ dT

    position = T_out[:3, 3]
    quaternion = t3d.quaternions.mat2quat(T_out[:3, :3]) 

    if not scalar_first_quat:
        quaternion = process_quaternion(quaternion, 'f2l')
        
    return position, quaternion


def similarity_transform(pos: Union[List[float], np.ndarray, None], 
                        quat: Union[List[float], np.ndarray, None], 
                        scalar_first_quat: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a similarity transformation with axis flipping.

    This function achieves axis flipping by applying a special transformation matrix,
    typically used to mirror a coordinate system.

    Args:
        pos: The 3D position vector
        quat: The quaternion representing the rotation
        scalar_first_quat: If True, the quaternion is in scalar-first format
        
    Returns:
        Tuple containing:
        - The 3D position vector after the similarity transformation
        - The quaternion after the similarity transformation
    """
    # Create transformation matrix
    T = _create_transform_matrix(pos, quat, scalar_first_quat)
    
    # Create axis flipping matrix
    dT = np.eye(4)
    dT[0, 0] = -1
    dT[1, 1] = -1
    
    # Apply similarity transformation
    T_out = dT @ T @ dT
    
    # Extract position and orientation
    position = T_out[:3, 3]
    quaternion = t3d.quaternions.mat2quat(T_out[:3, :3])
    
    if not scalar_first_quat:
        quaternion = process_quaternion(quaternion, 'f2l')
    
    return position, quaternion


def scalar_transform(pos: Union[List[float], np.ndarray], 
                    quat: Union[List[float], np.ndarray], 
                    pos_scaler: float, 
                    quat_scaler: float, 
                    scalar_first_quat: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a scalar transformation to the position and rotation.

    Args:
        pos: The 3D position vector
        quat: The quaternion representing the rotation
        pos_scaler: The scaling factor for the position vector
        quat_scaler: The scaling factor for the rotation angle
        scalar_first_quat: If True, the quaternion is in scalar-first format
        
    Returns:
        Tuple containing:
        - The scaled 3D position vector
        - The scaled quaternion
    """
    pos_array = _ensure_numpy_array(pos, (3,))
    quat_array = _ensure_quaternion_format(quat, scalar_first_quat)
    
    # Scale position
    new_pos = pos_array * pos_scaler
    
    # Scale rotation
    axis, angle = t3d.quaternions.quat2axangle(quat_array)
    print(f"axis: {axis}, angle: {angle}")
    rot_vec = axis * angle
    scaled_rot_vec = rot_vec * quat_scaler
    if np.linalg.norm(scaled_rot_vec) < 1e-6:
        scaled_rot_vec = [1, 0 ,0]
        new_quat = t3d.quaternions.axangle2quat(scaled_rot_vec, 0)
    else:
        new_quat = t3d.quaternions.axangle2quat(scaled_rot_vec, np.linalg.norm(scaled_rot_vec))
    
    if not scalar_first_quat:
        new_quat = process_quaternion(new_quat, 'f2l')
    
    return new_pos, new_quat


def calc_position_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 3D positions.
    
    Args:
        pos1: First position [x, y, z]
        pos2: Second position [x, y, z]
    
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(pos1 - pos2)


def calc_angle_diff(quat1: np.ndarray, quat2: np.ndarray, scalar_last: bool = True) -> float:
    """
    Calculate the rotation angle difference between two quaternions.
    
    Args:
        quat1: First quaternion. If scalar_last=True: [qx, qy, qz, qw], else [qw, qx, qy, qz]
        quat2: Second quaternion. If scalar_last=True: [qx, qy, qz, qw], else [qw, qx, qy, qz]
        scalar_last: If True, quaternions are in [qx, qy, qz, qw] format, 
                     else in [qw, qx, qy, qz] format
    
    Returns:
        The angle difference in radians
    """
    # Convert to scalar-first format [qw, qx, qy, qz] for transforms3d
    if scalar_last:
        q1 = np.array([quat1[3], quat1[0], quat1[1], quat1[2]])
        q2 = np.array([quat2[3], quat2[0], quat2[1], quat2[2]])
    else:
        q1 = np.array(quat1)
        q2 = np.array(quat2)
    
    # Convert to rotation matrices
    R1 = t3d.quaternions.quat2mat(q1)
    R2 = t3d.quaternions.quat2mat(q2)
    
    # Calculate angle difference
    cos_angle_dist = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_angle_dist)


def calc_ee_diff(ee_pose1: Union[float, np.ndarray], ee_pose2: Union[float, np.ndarray]) -> float:
    """
    Calculate the difference of end-effector pose between two poses.
    
    Args:
        ee_pose1: First end-effector pose (e.g., width for gripper, can be scalar or array)
        ee_pose2: Second end-effector pose (e.g., width for gripper, can be scalar or array)
    
    Returns:
        The max absolute difference for array inputs, or absolute difference for scalar inputs
    """
    diff = np.abs(ee_pose1 - ee_pose2)
    # If it's an array, return max; if scalar, return as is
    if isinstance(diff, np.ndarray) and diff.size > 1:
        return float(np.max(diff))
    else:
        return float(diff)


def calc_action_distance(
    action1: np.ndarray, 
    action2: np.ndarray,
    pos_weight: float = 0.4,
    rot_weight: float = 0.4,
    ee_weight: float = 0.2,
    pos_scale: float = 1.0,
    rot_scale: float = 1.0,
    ee_scale: float = 1.0
) -> float:
    """
    Calculate weighted distance between two robot actions.
    
    Supports two formats:
    - 7D: [x, y, z, qx, qy, qz, qw] (position + quaternion, no gripper)
    - 8D+: [x, y, z, qx, qy, qz, qw, gripper_width, ...] (position + quaternion + gripper)
    
    Args:
        action1: First action, 7D or 8D+ array
        action2: Second action, 7D or 8D+ array
        pos_weight: Weight for position distance
        rot_weight: Weight for rotation distance
        ee_weight: Weight for end-effector distance (ignored if action is 7D)
        pos_scale: Scale factor for position distance
        rot_scale: Scale factor for rotation distance
        ee_scale: Scale factor for end-effector distance
    
    Returns:
        Weighted distance
    """
    action_dim = len(action1)
    
    # Extract position [x, y, z]
    pos1 = action1[:3]
    pos2 = action2[:3]
    pos_dist = calc_position_distance(pos1, pos2) * pos_scale
    
    # Extract quaternion [qx, qy, qz, qw]
    quat1 = action1[3:7]
    quat2 = action2[3:7]
    rot_dist = calc_angle_diff(quat1, quat2, scalar_last=True) * rot_scale
    
    # Check if end-effector data is available (8D or more)
    if action_dim >= 8:
        # Extract end-effector (gripper, etc.)
        ee1 = action1[7:]
        ee2 = action2[7:]
        ee_dist = calc_ee_diff(ee1, ee2) * ee_scale
        
        # Weighted sum with ee
        total_weight = pos_weight + rot_weight + ee_weight
        weighted_dist = (pos_weight * pos_dist + rot_weight * rot_dist + ee_weight * ee_dist) / total_weight
    else:
        # No end-effector data (7D), only use position and rotation
        total_weight = pos_weight + rot_weight
        weighted_dist = (pos_weight * pos_dist + rot_weight * rot_dist) / total_weight
    
    return weighted_dist


def orthogonalization(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    a1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2 = a2 - np.sum(a1 * a2, axis=1, keepdims=True) * a1
    a2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
    a3 = np.cross(a1, a2)
    return a3

def pose_6d_to_pose_7d(pose: np.ndarray) -> np.ndarray:
    # convert 6D pose (x, y, z, r, p, y) to 7D pose (x, y, z, qw, qx, qy, qz)
    quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
    return np.concatenate([pose[:3], quat])

def pose_7d_to_pose_6d(pose: np.ndarray) -> np.ndarray:
    # convert 7D pose (x, y, z, qw, qx, qy, qz) to 6D pose (x, y, z, r, p, y)
    quat = pose[3:]
    euler = t3d.euler.quat2euler(quat)
    return np.concatenate([pose[:3], euler])

def pose_7d_to_4x4matrix(pose: np.ndarray) -> np.ndarray:
    # convert 7D pose (x, y, z, qw, qx, qy, qz) to 4x4 transformation matrix
    mat = np.eye(4)
    mat[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
    mat[:3, 3] = pose[:3]
    return mat

def pose_6d_to_4x4matrix(pose: np.ndarray) -> np.ndarray:
    # convert 6D pose (x, y, z, r, p, y) to 4x4 transformation matrix
    mat = np.eye(4)
    quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
    mat[:3, :3] = t3d.quaternions.quat2mat(quat)
    mat[:3, 3] = pose[:3]
    return mat

def matrix4x4_to_pose_6d(mat: np.ndarray) -> np.ndarray:
    # convert 4x4 transformation matrix to 6D pose (x, y, z, r, p, y)
    quat = t3d.quaternions.mat2quat(mat[:3, :3])
    euler = t3d.euler.quat2euler(quat)
    trans = mat[:3, 3]
    return np.concatenate([trans, euler])

def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix from ortho6d representation
    """
    x_raw = ortho6d[:, 0:3]  # batch * 3
    y_raw = ortho6d[:, 3:6]  # batch * 3
    x = normalize_vector(x_raw)  # batch * 3
    z = np.cross(x, y_raw)  # batch * 3
    z = normalize_vector(z)  # batch * 3
    y = np.cross(z, x)  # batch * 3

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    z = z[:, :, np.newaxis]

    matrix = np.concatenate((x, y, z), axis=2)  # batch * 3 * 3
    return matrix

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector (batch * 3)
    """
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)  # batch * 1
    v_mag = np.maximum(v_mag, 1e-8)
    v = v / v_mag
    return v

def transform_point_cloud(pcd: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a point cloud with 4x4 transform_matrix
    Parameters
    ----------
    pcd: (N, 6) or (N, 3) ndarray
    transform_matrix: (4, 4) ndarray
    """
    if pcd.shape[1] == 3: # (x, y, z)
        transformed_xyz = np.matmul(transform_matrix[:3, :3], pcd.T).T + transform_matrix[:3, 3]
        return transformed_xyz
    elif pcd.shape[1] == 6:  # (x, y, z, r, p, y)
        transformed_xyz = np.matmul(transform_matrix[:3, :3], pcd[:, :3].T).T + transform_matrix[:3, 3]
        return np.concatenate([transformed_xyz, pcd[:, 3:]], axis=1)
    else:
        raise NotImplementedError

def pose_6d_to_pose_9d(pose: np.ndarray) -> np.ndarray:
    """
    Convert 6D state to 9D state
    :param pose: np.ndarray (6,), (x, y, z, rx, ry, rz)
    :return: np.ndarray (9,), (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
    """
    rot_6d = pose_6d_to_4x4matrix(pose)[:3, :2].T.flatten()
    return np.concatenate((pose[:3], rot_6d), axis=0)

def pose_3d_9d_to_homo_matrix_batch(pose: np.ndarray) -> np.ndarray:
    """
    Convert 3D / 9D states to 4x4 matrix
    :param pose: np.ndarray (N, 9) or (N, 3)
    :return: np.ndarray (N, 4, 4)
    """
    assert pose.shape[1] in [3, 9], "pose should be (N, 3) or (N, 9)"
    mat = np.eye(4)[None, :, :].repeat(pose.shape[0], axis=0)
    mat[:, :3, 3] = pose[:, :3]
    if pose.shape[1] == 9:
        mat[:, :3, :3] = ortho6d_to_rotation_matrix(pose[:, 3:9])
    return mat

def homo_matrix_to_pose_9d_batch(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 matrix to 9D state
    :param mat: np.ndarray (N, 4, 4)
    :return: np.ndarray (N, 9)
    """
    assert mat.shape[1:] == (4, 4), "mat should be (N, 4, 4)"
    pose = np.zeros((mat.shape[0], 9))
    pose[:, :3] = mat[:, :3, 3]
    pose[:, 3:9] = mat[:, :3, :2].swapaxes(1, 2).reshape(mat.shape[0], -1)
    return pose


try:
    from geometry_msgs.msg import Pose

    def ros_pose_to_4x4matrix(pose: Pose) -> np.ndarray:
        # Convert ROS Pose message to 4x4 transformation matrix
        mat = np.eye(4)
        quat = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        rot_mat = t3d.quaternions.quat2mat(quat)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
        return mat

    def ros_pose_to_6d_pose(pose: Pose) -> np.ndarray:
        # convert ROS Pose message to 6D pose (x, y, z, r, p, y)
        quat = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        euler = t3d.euler.quat2euler(quat)
        trans = np.array([pose.position.x, pose.position.y, pose.position.z])
        return np.concatenate([trans, euler])
except ImportError:
    pass
