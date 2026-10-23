#!/usr/bin/env python3
"""
Compute the offline YamHead focus-on-robot-center trajectory from a LeRobot episode.

Author: Yichen Xie
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import yaml

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logger_config import logger  # noqa: E402

REPLAY_COLUMNS = ("action", "observation.state", "observation/state")


Pose7 = List[float]


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _load_transform(path_value: str, key: str) -> np.ndarray:
    path = _resolve_repo_path(path_value)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    node = data.get(key, data)
    matrix = np.asarray(node["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{path} must contain a 4x4 matrix, got {matrix.shape}")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"{path} must have homogeneous last row [0, 0, 0, 1]")
    return matrix


def _as_vector(values: Any, *, normalize: bool, name: str) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    if vec.shape[0] != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vec.shape[0]}")
    if not normalize:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return vec / norm


def _transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    return (np.asarray(transform, dtype=np.float64).reshape(4, 4) @ point_h)[:3]


def _project_perpendicular(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    projected = vec - axis * float(np.dot(vec, axis))
    norm = float(np.linalg.norm(projected))
    if norm > 1e-9:
        return projected / norm
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(fallback, axis))) > 0.9:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    projected = fallback - axis * float(np.dot(fallback, axis))
    return projected / max(float(np.linalg.norm(projected)), 1e-9)


def _look_at_rotation(
    origin: np.ndarray,
    target: np.ndarray,
    *,
    local_forward: np.ndarray,
    local_up: np.ndarray,
    world_up: np.ndarray,
) -> Optional[np.ndarray]:
    direction = np.asarray(target, dtype=np.float64).reshape(3) - np.asarray(origin, dtype=np.float64).reshape(3)
    distance = float(np.linalg.norm(direction))
    if distance < 1e-6:
        return None
    desired_forward = direction / distance
    desired_up = _project_perpendicular(world_up, desired_forward)
    local_up_projected = _project_perpendicular(local_up, local_forward)
    rotation, _ = R.align_vectors(
        np.vstack([desired_forward, desired_up]),
        np.vstack([local_forward, local_up_projected]),
    )
    return rotation.as_matrix()


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    rotation_orthonormal = u @ vt
    if np.linalg.det(rotation_orthonormal) < 0:
        u[:, -1] *= -1.0
        rotation_orthonormal = u @ vt
    return rotation_orthonormal


def _apply_focus_world_y_pitch(rotation_world: np.ndarray, pitch_deg: float) -> np.ndarray:
    if abs(pitch_deg) < 1e-9:
        return rotation_world
    pitch_world = R.from_rotvec(
        np.deg2rad(pitch_deg) * np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ).as_matrix()
    return _orthonormalize_rotation(pitch_world @ rotation_world)


def _pose7_wxyz_to_matrix(pose7: Pose7) -> np.ndarray:
    """Convert Flexiv/data pose [x, y, z, qw, qx, qy, qz] to a homogeneous matrix."""
    values = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if values.size < 7:
        raise ValueError(f"Expected pose7 [x,y,z,qw,qx,qy,qz], got {values.size} values")
    quat_wxyz = values[3:7] / max(float(np.linalg.norm(values[3:7])), 1e-12)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    return matrix


def _matrix_to_pose7_wxyz(matrix: np.ndarray) -> Pose7:
    """Convert a homogeneous matrix to [x, y, z, qw, qx, qy, qz]."""
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
    return [
        float(mat[0, 3]),
        float(mat[1, 3]),
        float(mat[2, 3]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def _pose7_xyzw_to_wxyz(pose7: List[float]) -> Pose7:
    """Convert [x, y, z, qx, qy, qz, qw] to [x, y, z, qw, qx, qy, qz]."""
    values = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if values.size < 7:
        raise ValueError(f"Expected pose7 [x,y,z,qx,qy,qz,qw], got {values.size} values")
    return [
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[6]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    ]


def _split_dual_arm_row(row: np.ndarray) -> Tuple[Pose7, Pose7, Optional[Pose7]]:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size == 14:
        return values[0:7].tolist(), values[7:14].tolist(), None
    if values.size == 16:
        return values[0:7].tolist(), values[8:15].tolist(), None
    if values.size == 23:
        return values[0:7].tolist(), values[8:15].tolist(), values[16:23].tolist()
    raise ValueError(f"Unsupported replay row dimension: {values.size}; expected 14, 16, or 23")


def _detect_replay_column(columns: List[str], dataset_cfg: Dict[str, Any]) -> str:
    configured = dataset_cfg.get("replay_column")
    if configured is not None:
        name = str(configured)
        if name not in columns:
            raise ValueError(f"dataset.replay_column {name!r} not in parquet columns: {columns}")
        return name
    for name in REPLAY_COLUMNS:
        if name in columns:
            return name
    raise ValueError(f"No replay column found. Tried {REPLAY_COLUMNS}; available columns: {columns}")


def _load_episode_rows(config: Dict[str, Any], replay_column: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    dataset_cfg = config.get("dataset", config)
    replay_cfg = config.get("replay", {})
    data_path = _resolve_repo_path(str(dataset_cfg["data_file"]))
    table = pd.read_parquet(data_path)

    start_index = int(replay_cfg.get("start_index", 0))
    end_index = replay_cfg.get("end_index")
    end_index = len(table) if end_index is None else int(end_index)
    table = table.iloc[start_index:end_index].reset_index(drop=True)
    if len(table) == 0:
        raise ValueError("No frames selected for trajectory computation")

    column = replay_column or _detect_replay_column(list(table.columns), dataset_cfg)
    array = np.asarray(table[column].to_list(), dtype=object)
    if array.ndim == 1 and array.dtype == object:
        array = np.stack([np.asarray(row, dtype=np.float64) for row in array], axis=0)
    else:
        array = np.asarray(array, dtype=np.float64)

    timestamp_column = dataset_cfg.get("timestamp_column")
    if timestamp_column and timestamp_column in table.columns:
        timestamps = table[timestamp_column].to_numpy(dtype=np.float64)
        timestamps = timestamps - timestamps[0] if timestamps.size > 1 else timestamps
    elif "timestamp" in table.columns:
        timestamps = table["timestamp"].to_numpy(dtype=np.float64)
        timestamps = timestamps - timestamps[0] if timestamps.size > 1 else timestamps
    else:
        frequency = float(replay_cfg.get("frequency", 20.0))
        timestamps = np.arange(len(table), dtype=np.float64) / frequency

    logger.info(f"Loaded {len(table)} frames from {data_path} column={column!r}, dim={array.shape[1]}")
    return array, timestamps


def _smooth_transform(previous: Optional[np.ndarray], target: np.ndarray, alpha: float) -> np.ndarray:
    if previous is None:
        return target.copy()
    smoothed = np.eye(4, dtype=np.float64)
    smoothed[:3, 3] = (1.0 - alpha) * previous[:3, 3] + alpha * target[:3, 3]
    rotation_delta = target[:3, :3] @ previous[:3, :3].T
    rotvec = R.from_matrix(rotation_delta).as_rotvec()
    smoothed[:3, :3] = R.from_rotvec(rotvec * alpha).as_matrix() @ previous[:3, :3]
    return smoothed


def _find_deepest_body(element: ET.Element) -> ET.Element:
    current = element
    while True:
        child_bodies = [child for child in current if child.tag == "body"]
        if not child_bodies:
            return current
        current = child_bodies[0]


def _resolve_i2rt_root(path_value: Optional[str]) -> Path:
    candidates: List[Path] = []
    if path_value:
        candidates.append(Path(path_value).expanduser())
    candidates.extend(
        [
            project_root / "third_party" / "i2rt",
            Path("/home/noematrix/code/SuperInference.r1_teleopration/third_party/i2rt"),
        ]
    )
    for candidate in candidates:
        if (candidate / "i2rt" / "robot_models").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate i2rt root. Pass --i2rt-root pointing to the directory that contains i2rt/robot_models."
    )


def _combine_yam_no_gripper_xml(i2rt_root: Path, arm_name: str, gripper_name: str) -> Path:
    if gripper_name != "no_gripper":
        raise ValueError("Offline IK currently supports yam_gripper='no_gripper' only")
    arm_path = i2rt_root / "i2rt" / "robot_models" / "arm" / arm_name / f"{arm_name}.xml"
    gripper_path = i2rt_root / "i2rt" / "robot_models" / "gripper" / gripper_name / f"{gripper_name}.xml"
    mount_config_path = i2rt_root / "i2rt" / "robots" / "config" / f"{gripper_name}.yml"
    if not arm_path.exists():
        raise FileNotFoundError(f"YAM arm XML not found: {arm_path}")
    if not gripper_path.exists():
        raise FileNotFoundError(f"YAM gripper XML not found: {gripper_path}")

    arm_tree = ET.parse(arm_path)
    arm_root = arm_tree.getroot()
    with mount_config_path.open("r", encoding="utf-8") as fp:
        mount_config = yaml.safe_load(fp)
    mount = mount_config["last_joint_mount"][arm_name]

    worldbody = arm_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Arm XML missing worldbody: {arm_path}")
    last_link = _find_deepest_body(worldbody)
    last_link.set("pos", str(mount["pos"]))
    last_link.set("quat", str(mount["quat"]))
    last_joint = last_link.find("joint")
    if last_joint is not None:
        last_joint.set("axis", str(mount["axis"]))

    arm_dir = arm_path.parent.resolve()
    arm_compiler = arm_root.find("compiler")
    arm_meshdir = arm_compiler.get("meshdir", "") if arm_compiler is not None else ""
    arm_asset = arm_root.find("asset")
    if arm_asset is not None:
        for child in arm_asset:
            file_attr = child.get("file")
            if file_attr and not Path(file_attr).is_absolute():
                child.set("file", str((arm_dir / arm_meshdir / file_attr).resolve()))
    if arm_compiler is not None and arm_compiler.get("meshdir"):
        del arm_compiler.attrib["meshdir"]

    grip_tree = ET.parse(gripper_path)
    grip_root = grip_tree.getroot()
    grip_body = grip_root.find(".//body[@name='gripper']")
    if grip_body is None:
        raise ValueError(f"Gripper XML missing body named 'gripper': {gripper_path}")
    existing_gripper = arm_root.find(".//body[@name='gripper']")
    if existing_gripper is not None:
        for parent in arm_root.iter():
            children = list(parent)
            for index, child in enumerate(children):
                if child.tag == "body" and child.get("name") == "gripper":
                    parent.remove(child)
                    parent.insert(index, deepcopy(grip_body))
                    break
            else:
                continue
            break
    else:
        _find_deepest_body(worldbody).append(deepcopy(grip_body))

    output = tempfile.NamedTemporaryFile(
        suffix=".xml",
        prefix=f"superinference_{arm_name}_{gripper_name}_",
        delete=False,
        dir="/tmp",
    )
    output.close()
    arm_tree.write(output.name)
    return Path(output.name)


def _parse_initial_q(spec: str, expected_size: int) -> np.ndarray:
    values = [float(token) for token in str(spec).replace(";", ",").split(",") if token.strip()]
    if len(values) != expected_size:
        raise ValueError(f"yam_initial_q must have {expected_size} values, got {len(values)}: {spec!r}")
    return np.asarray(values, dtype=np.float64)


def _robot_q_to_model_qpos(model: Any, robot_q: np.ndarray) -> np.ndarray:
    import mujoco

    qpos = np.zeros(model.nq, dtype=np.float64)
    n = min(len(robot_q), model.nq)
    qpos[:n] = np.asarray(robot_q, dtype=np.float64)[:n]
    for joint_index in range(model.njnt):
        qpos_address = model.jnt_qposadr[joint_index]
        if qpos_address >= n:
            continue
        if model.jnt_type[joint_index] == mujoco.mjtJoint.mjJNT_SLIDE:
            low, high = model.jnt_range[joint_index]
            qpos[qpos_address] = low + qpos[qpos_address] * (high - low)
    for equality_index in range(model.neq):
        if model.eq_type[equality_index] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        address_1 = model.jnt_qposadr[model.eq_obj1id[equality_index]]
        address_2 = model.jnt_qposadr[model.eq_obj2id[equality_index]]
        coeffs = model.eq_data[equality_index, :5]
        qpos[address_2] = np.polyval(coeffs[::-1], qpos[address_1])
    return qpos


def _orientation_error_rad(achieved: np.ndarray, target: np.ndarray) -> float:
    rotation_error = achieved[:3, :3] @ target[:3, :3].T
    return float(np.arccos(np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0)))


def _solve_mink_ik(
    *,
    configuration: Any,
    target_pose: np.ndarray,
    site_name: str,
    init_qpos: np.ndarray,
    solver: str,
    max_iters: int,
    pos_threshold: float,
    ori_threshold: float,
) -> Tuple[bool, np.ndarray, float, float]:
    import mink

    configuration.update(init_qpos)
    task = mink.FrameTask(
        frame_name=site_name,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    task.set_target(mink.SE3.from_matrix(target_pose))
    success = False
    for _ in range(max_iters):
        velocity = mink.solve_ik(configuration, [task], 0.01, solver, damping=1e-4)
        configuration.integrate_inplace(velocity, 0.01)
        error = task.compute_error(configuration)
        if np.linalg.norm(error[:3]) <= pos_threshold and np.linalg.norm(error[3:]) <= ori_threshold:
            success = True
            break
    achieved = configuration.get_transform_frame_to_world(site_name, "site").as_matrix()
    pos_error = float(np.linalg.norm(achieved[:3, 3] - target_pose[:3, 3]))
    ori_error = _orientation_error_rad(achieved, target_pose)
    return success, np.asarray(configuration.q, dtype=np.float64).copy(), pos_error, ori_error


def compute_trajectory(
    config: Dict[str, Any],
    *,
    replay_column: Optional[str],
    left_init_pose_wxyz: Pose7,
    right_init_pose_wxyz: Pose7,
    focus_origin_yam: np.ndarray,
    filter_tau: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    yam_cfg = dict(config.get("yam_head_tracking", {}).get("yam_head", {}))
    left2world = _load_transform(str(yam_cfg["left2world_json_path"]), "left2world")
    right2world = _load_transform(str(yam_cfg["right2world_json_path"]), "right2world")
    yam2world = _load_transform(str(yam_cfg["yam2world_json_path"]), "yam2world")
    world2yam = np.linalg.inv(yam2world)

    local_forward = _as_vector(yam_cfg.get("yam_focus_local_forward", [1.0, 0.0, 0.0]), normalize=True, name="yam_focus_local_forward")
    local_up = _as_vector(yam_cfg.get("yam_focus_local_up", [0.0, 0.0, 1.0]), normalize=True, name="yam_focus_local_up")
    world_up = _as_vector(yam_cfg.get("yam_focus_world_up", [0.0, 0.0, 1.0]), normalize=True, name="yam_focus_world_up")
    world_y_pitch_deg = float(yam_cfg.get("yam_focus_world_y_pitch_deg", 0.0))
    center_offset = _as_vector(yam_cfg.get("robot_center_offset_xyz", [0.0, 0.0, 0.0]), normalize=False, name="robot_center_offset_xyz")

    rows, timestamps = _load_episode_rows(config, replay_column)
    focus_origin_world = _transform_point(yam2world, focus_origin_yam)
    left_abs_mats: List[np.ndarray] = []
    right_abs_mats: List[np.ndarray] = []
    head_poses: List[Optional[Pose7]] = []
    for row in rows:
        left_abs_pose, right_abs_pose, head_pose = _split_dual_arm_row(row)
        left_abs_mats.append(_pose7_wxyz_to_matrix(left_abs_pose))
        right_abs_mats.append(_pose7_wxyz_to_matrix(right_abs_pose))
        head_poses.append(head_pose)
    left_initial_inv = np.linalg.inv(left_abs_mats[0])
    right_initial_inv = np.linalg.inv(right_abs_mats[0])
    left_rel_mats = [left_initial_inv @ mat for mat in left_abs_mats]
    right_rel_mats = [right_initial_inv @ mat for mat in right_abs_mats]
    left_replay_init_mat = _pose7_wxyz_to_matrix(left_init_pose_wxyz)
    right_replay_init_mat = _pose7_wxyz_to_matrix(right_init_pose_wxyz)

    records: List[Dict[str, float]] = []
    world_targets: List[np.ndarray] = []
    yam_targets: List[np.ndarray] = []
    previous_yam_target: Optional[np.ndarray] = None
    previous_timestamp = float(timestamps[0])

    for frame_index, row in enumerate(rows):
        left_target_mat = left_replay_init_mat @ left_rel_mats[frame_index]
        right_target_mat = right_replay_init_mat @ right_rel_mats[frame_index]
        left_pose = _matrix_to_pose7_wxyz(left_target_mat)
        right_pose = _matrix_to_pose7_wxyz(right_target_mat)
        head_pose = head_poses[frame_index]
        left_world = _transform_point(left2world, np.asarray(left_pose[:3], dtype=np.float64))
        right_world = _transform_point(right2world, np.asarray(right_pose[:3], dtype=np.float64))
        center_world = 0.5 * (left_world + right_world) + center_offset
        rotation_world = _look_at_rotation(
            focus_origin_world,
            center_world,
            local_forward=local_forward,
            local_up=local_up,
            world_up=world_up,
        )
        if rotation_world is None:
            continue
        rotation_world = _apply_focus_world_y_pitch(rotation_world, world_y_pitch_deg)

        world_target = np.eye(4, dtype=np.float64)
        world_target[:3, :3] = rotation_world
        world_target[:3, 3] = focus_origin_world
        yam_target = world2yam @ world_target
        timestamp = float(timestamps[frame_index])
        if filter_tau > 0.0:
            dt = max(timestamp - previous_timestamp, 1e-6)
            alpha = dt / (dt + filter_tau)
            yam_target = _smooth_transform(previous_yam_target, yam_target, alpha)
            previous_yam_target = yam_target.copy()
            previous_timestamp = timestamp
            world_target = yam2world @ yam_target

        quat_world_xyzw = R.from_matrix(world_target[:3, :3]).as_quat()
        quat_yam_xyzw = R.from_matrix(yam_target[:3, :3]).as_quat()
        forward_world = world_target[:3, :3] @ local_forward
        distance = float(np.linalg.norm(center_world - focus_origin_world))

        record: Dict[str, float] = {
            "frame_index": float(frame_index),
            "timestamp": timestamp,
            "left_world_x": float(left_world[0]),
            "left_world_y": float(left_world[1]),
            "left_world_z": float(left_world[2]),
            "left_replay_x": float(left_pose[0]),
            "left_replay_y": float(left_pose[1]),
            "left_replay_z": float(left_pose[2]),
            "left_replay_qw": float(left_pose[3]),
            "left_replay_qx": float(left_pose[4]),
            "left_replay_qy": float(left_pose[5]),
            "left_replay_qz": float(left_pose[6]),
            "right_world_x": float(right_world[0]),
            "right_world_y": float(right_world[1]),
            "right_world_z": float(right_world[2]),
            "right_replay_x": float(right_pose[0]),
            "right_replay_y": float(right_pose[1]),
            "right_replay_z": float(right_pose[2]),
            "right_replay_qw": float(right_pose[3]),
            "right_replay_qx": float(right_pose[4]),
            "right_replay_qy": float(right_pose[5]),
            "right_replay_qz": float(right_pose[6]),
            "center_world_x": float(center_world[0]),
            "center_world_y": float(center_world[1]),
            "center_world_z": float(center_world[2]),
            "focus_origin_world_x": float(focus_origin_world[0]),
            "focus_origin_world_y": float(focus_origin_world[1]),
            "focus_origin_world_z": float(focus_origin_world[2]),
            "distance_m": distance,
            "focus_world_y_pitch_deg": world_y_pitch_deg,
            "forward_world_x": float(forward_world[0]),
            "forward_world_y": float(forward_world[1]),
            "forward_world_z": float(forward_world[2]),
            "target_world_x": float(world_target[0, 3]),
            "target_world_y": float(world_target[1, 3]),
            "target_world_z": float(world_target[2, 3]),
            "target_world_qx": float(quat_world_xyzw[0]),
            "target_world_qy": float(quat_world_xyzw[1]),
            "target_world_qz": float(quat_world_xyzw[2]),
            "target_world_qw": float(quat_world_xyzw[3]),
            "target_yam_x": float(yam_target[0, 3]),
            "target_yam_y": float(yam_target[1, 3]),
            "target_yam_z": float(yam_target[2, 3]),
            "target_yam_qx": float(quat_yam_xyzw[0]),
            "target_yam_qy": float(quat_yam_xyzw[1]),
            "target_yam_qz": float(quat_yam_xyzw[2]),
            "target_yam_qw": float(quat_yam_xyzw[3]),
        }
        if head_pose is not None:
            record.update(
                {
                    "dataset_head_x": float(head_pose[0]),
                    "dataset_head_y": float(head_pose[1]),
                    "dataset_head_z": float(head_pose[2]),
                    "dataset_head_qw": float(head_pose[3]),
                    "dataset_head_qx": float(head_pose[4]),
                    "dataset_head_qy": float(head_pose[5]),
                    "dataset_head_qz": float(head_pose[6]),
                }
            )
        records.append(record)
        world_targets.append(world_target)
        yam_targets.append(yam_target)

    if not records:
        raise ValueError("No valid YamHead focus poses were computed")
    return pd.DataFrame.from_records(records), np.stack(world_targets), np.stack(yam_targets)


def compute_yamhead_ik_trajectory(
    *,
    config: Dict[str, Any],
    center_trajectory: pd.DataFrame,
    i2rt_root: Path,
    solver: str,
    focus_origin_yam: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    import mink
    import mujoco
    import qpsolvers

    yam_cfg = dict(config.get("yam_head_tracking", {}).get("yam_head", {}))
    arm_name = str(yam_cfg.get("yam_arm", "yam"))
    gripper_name = str(yam_cfg.get("yam_gripper", "no_gripper"))
    site_name = str(yam_cfg.get("yam_ee_site", "grasp_site"))
    xml_path = _combine_yam_no_gripper_xml(i2rt_root, arm_name, gripper_name)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    configuration = mink.Configuration(model)

    available_solvers = list(qpsolvers.available_solvers)
    if solver == "auto":
        if "quadprog" in available_solvers:
            solver = "quadprog"
        elif available_solvers:
            solver = available_solvers[0]
        else:
            raise RuntimeError("No qpsolvers backend is available for mink IK")
    if solver not in available_solvers:
        raise RuntimeError(f"IK solver {solver!r} is not available. Available solvers: {available_solvers}")

    world2yam = np.linalg.inv(_load_transform(str(yam_cfg["yam2world_json_path"]), "yam2world"))
    yam2world = np.linalg.inv(world2yam)
    local_forward = _as_vector(
        yam_cfg.get("yam_focus_local_forward", [1.0, 0.0, 0.0]),
        normalize=True,
        name="yam_focus_local_forward",
    )
    local_up = _as_vector(
        yam_cfg.get("yam_focus_local_up", [0.0, 0.0, 1.0]),
        normalize=True,
        name="yam_focus_local_up",
    )
    world_up = _as_vector(
        yam_cfg.get("yam_focus_world_up", [0.0, 0.0, 1.0]),
        normalize=True,
        name="yam_focus_world_up",
    )
    world_y_pitch_deg = float(yam_cfg.get("yam_focus_world_y_pitch_deg", 0.0))
    configured_focus_origin = (
        focus_origin_yam
        if focus_origin_yam is not None
        else (
            None
            if yam_cfg.get("yam_focus_position_xyz") is None
            else _as_vector(yam_cfg["yam_focus_position_xyz"], normalize=False, name="yam_focus_position_xyz")
        )
    )

    n_arm = 6
    q_target = _parse_initial_q(str(yam_cfg.get("yam_initial_q", "0,2.4,0.95,1.15,0,-0.12")), n_arm)
    yam_max_step = float(yam_cfg.get("yam_max_step", 0.03))
    max_iters = int(yam_cfg.get("yam_ik_max_iters", 30))
    soft_pos_tol = float(yam_cfg.get("yam_ik_soft_pos_tol", 0.02))
    soft_ori_tol = float(yam_cfg.get("yam_ik_soft_ori_tol", 0.10))
    filter_tau = float(yam_cfg.get("yam_head_filter_tau", 0.0))
    previous_filtered_target: Optional[np.ndarray] = None
    previous_timestamp = float(center_trajectory["timestamp"].iloc[0])
    configuration.update(_robot_q_to_model_qpos(model, q_target))

    records: List[Dict[str, float]] = []
    target_poses: List[np.ndarray] = []
    achieved_poses: List[np.ndarray] = []

    for _, center_row in center_trajectory.iterrows():
        timestamp = float(center_row["timestamp"])
        center_world = np.array(
            [
                float(center_row["center_world_x"]),
                float(center_row["center_world_y"]),
                float(center_row["center_world_z"]),
            ],
            dtype=np.float64,
        )
        T_yam_now = configuration.get_transform_frame_to_world(site_name, "site").as_matrix()
        if records:
            T_yam_now = achieved_poses[-1]
        focus_origin_yam = (
            configured_focus_origin
            if configured_focus_origin is not None
            else np.asarray(T_yam_now[:3, 3], dtype=np.float64)
        )
        focus_origin_world = _transform_point(yam2world, focus_origin_yam)
        rotation_world = _look_at_rotation(
            focus_origin_world,
            center_world,
            local_forward=local_forward,
            local_up=local_up,
            world_up=world_up,
        )
        if rotation_world is None:
            continue
        rotation_world = _apply_focus_world_y_pitch(rotation_world, world_y_pitch_deg)

        world_target = np.eye(4, dtype=np.float64)
        world_target[:3, :3] = rotation_world
        world_target[:3, 3] = focus_origin_world
        yam_target = world2yam @ world_target
        if filter_tau > 0.0:
            dt = max(timestamp - previous_timestamp, 1e-6)
            alpha = dt / (dt + filter_tau)
            yam_target = _smooth_transform(previous_filtered_target, yam_target, alpha)
            previous_filtered_target = yam_target.copy()
            previous_timestamp = timestamp

        init_qpos = _robot_q_to_model_qpos(model, q_target)
        ik_success, q_ik, pos_error, ori_error = _solve_mink_ik(
            configuration=configuration,
            target_pose=yam_target,
            site_name=site_name,
            init_qpos=init_qpos,
            solver=solver,
            max_iters=max_iters,
            pos_threshold=1e-3,
            ori_threshold=1e-2,
        )
        accepted = bool(ik_success)
        if not ik_success and soft_pos_tol > 0.0 and pos_error <= soft_pos_tol and ori_error <= soft_ori_tol:
            accepted = True

        held = not accepted
        if accepted:
            q_candidate = q_target.copy()
            q_candidate[:n_arm] = q_ik[:n_arm]
            delta_q = np.clip(q_candidate[:n_arm] - q_target[:n_arm], -yam_max_step, yam_max_step)
            q_target = q_target.copy()
            q_target[:n_arm] = q_target[:n_arm] + delta_q

        achieved_yam = configuration.get_transform_frame_to_world(site_name, "site").as_matrix()
        configuration.update(_robot_q_to_model_qpos(model, q_target))
        achieved_yam = configuration.get_transform_frame_to_world(site_name, "site").as_matrix()
        achieved_world = yam2world @ achieved_yam
        target_world = yam2world @ yam_target
        achieved_quat_xyzw = R.from_matrix(achieved_yam[:3, :3]).as_quat()
        achieved_world_quat_xyzw = R.from_matrix(achieved_world[:3, :3]).as_quat()
        target_quat_xyzw = R.from_matrix(yam_target[:3, :3]).as_quat()
        target_world_quat_xyzw = R.from_matrix(target_world[:3, :3]).as_quat()

        record: Dict[str, float] = {
            "frame_index": float(center_row["frame_index"]),
            "timestamp": timestamp,
            "ik_success": float(bool(ik_success)),
            "ik_accepted": float(accepted),
            "held": float(held),
            "ik_pos_error_m": float(pos_error),
            "ik_ori_error_rad": float(ori_error),
            "target_yam_x": float(yam_target[0, 3]),
            "target_yam_y": float(yam_target[1, 3]),
            "target_yam_z": float(yam_target[2, 3]),
            "target_yam_qx": float(target_quat_xyzw[0]),
            "target_yam_qy": float(target_quat_xyzw[1]),
            "target_yam_qz": float(target_quat_xyzw[2]),
            "target_yam_qw": float(target_quat_xyzw[3]),
            "target_world_x": float(target_world[0, 3]),
            "target_world_y": float(target_world[1, 3]),
            "target_world_z": float(target_world[2, 3]),
            "target_world_qx": float(target_world_quat_xyzw[0]),
            "target_world_qy": float(target_world_quat_xyzw[1]),
            "target_world_qz": float(target_world_quat_xyzw[2]),
            "target_world_qw": float(target_world_quat_xyzw[3]),
            "ee_yam_x": float(achieved_yam[0, 3]),
            "ee_yam_y": float(achieved_yam[1, 3]),
            "ee_yam_z": float(achieved_yam[2, 3]),
            "ee_yam_qx": float(achieved_quat_xyzw[0]),
            "ee_yam_qy": float(achieved_quat_xyzw[1]),
            "ee_yam_qz": float(achieved_quat_xyzw[2]),
            "ee_yam_qw": float(achieved_quat_xyzw[3]),
            "ee_world_x": float(achieved_world[0, 3]),
            "ee_world_y": float(achieved_world[1, 3]),
            "ee_world_z": float(achieved_world[2, 3]),
            "ee_world_qx": float(achieved_world_quat_xyzw[0]),
            "ee_world_qy": float(achieved_world_quat_xyzw[1]),
            "ee_world_qz": float(achieved_world_quat_xyzw[2]),
            "ee_world_qw": float(achieved_world_quat_xyzw[3]),
            "focus_world_y_pitch_deg": world_y_pitch_deg,
            "center_world_x": float(center_world[0]),
            "center_world_y": float(center_world[1]),
            "center_world_z": float(center_world[2]),
        }
        for joint_index, joint_value in enumerate(q_target[:n_arm], start=1):
            record[f"q{joint_index}"] = float(joint_value)
        records.append(record)
        target_poses.append(yam_target.copy())
        achieved_poses.append(achieved_yam.copy())

    if not records:
        raise ValueError("No valid YamHead IK frames were computed")
    logger.info(f"Offline YAM IK used xml={xml_path}, solver={solver}")
    return pd.DataFrame.from_records(records), np.stack(target_poses), np.stack(achieved_poses)


def _selected_home_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    home_cfg = config.get("robot", {}).get("home", {})
    if not isinstance(home_cfg, dict):
        return {}
    presets = home_cfg.get("presets")
    if isinstance(presets, list) and presets:
        preset_index = int(home_cfg.get("preset_index", 0))
        if 0 <= preset_index < len(presets) and isinstance(presets[preset_index], dict):
            return presets[preset_index]
    return home_cfg


def _resolve_init_pose_wxyz(
    *,
    config: Dict[str, Any],
    arm: str,
    pose_wxyz: Optional[List[float]],
    pose_xyzw: Optional[List[float]],
) -> Pose7:
    if pose_wxyz is not None and pose_xyzw is not None:
        raise ValueError(f"Pass only one of --{arm}-init-pose-wxyz or --{arm}-init-pose-xyzw")
    if pose_wxyz is not None:
        return [float(value) for value in pose_wxyz]
    if pose_xyzw is not None:
        return _pose7_xyzw_to_wxyz(pose_xyzw)

    home_preset = _selected_home_preset(config)
    pose_key = f"pose_{arm}"
    if pose_key in home_preset:
        logger.info(f"Using robot.home {pose_key} as {arm} replay init pose (xyzw format)")
        return _pose7_xyzw_to_wxyz([float(value) for value in home_preset[pose_key]])

    raise ValueError(
        f"{arm} replay init TCP pose is required for correct offline YamHead trajectory. "
        f"Pass --{arm}-init-pose-wxyz X Y Z QW QX QY QZ using "
        "RobotRecordReplayer's captured TCP after homing, or add robot.home.pose_left/right "
        "in [x,y,z,qx,qy,qz,qw] format."
    )


def _positions_from_columns(dataframe: pd.DataFrame, columns: Tuple[str, str, str]) -> np.ndarray:
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing trajectory columns: {missing_columns}")
    return dataframe[list(columns)].to_numpy(dtype=np.float64)


def _sample_indices(length: int, max_count: int) -> List[int]:
    if length <= 0 or max_count <= 0:
        return []
    if length <= max_count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, max_count, dtype=np.int64).tolist()))


def _format_world_xyz_label(
    name: str,
    point_xyz: np.ndarray,
    *,
    frame_index: Optional[int] = None,
    timestamp: Optional[float] = None,
) -> str:
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    frame_text = "" if frame_index is None else f"[{frame_index}]"
    time_text = "" if timestamp is None else f" t={timestamp:.3f}s"
    return f"{name}{frame_text}{time_text} xyz=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m"


def _trajectory_extent(position_sets: List[np.ndarray]) -> float:
    valid_sets = [positions for positions in position_sets if positions.size > 0]
    if not valid_sets:
        return 1.0
    stacked = np.concatenate(valid_sets, axis=0)
    extent = float(np.max(np.ptp(stacked, axis=0)))
    max_distance_from_origin = float(np.max(np.linalg.norm(stacked, axis=1)))
    return max(extent, 0.25 * max_distance_from_origin, 1.0)


def _log_world_axes(rr: Any, axis_scale: float) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/origin/axes",
        rr.Arrows3D(
            vectors=[
                [axis_scale, 0.0, 0.0],
                [0.0, axis_scale, 0.0],
                [0.0, 0.0, axis_scale],
            ],
            origins=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            colors=[
                [255, 0, 0],
                [0, 255, 0],
                [0, 80, 255],
            ],
            labels=["world +X", "world +Y", "world +Z"],
            radii=[0.004, 0.004, 0.004],
        ),
        static=True,
    )
    rr.log(
        "world/origin/label",
        rr.Points3D(
            [[0.0, 0.0, 0.0]],
            colors=[[255, 255, 255]],
            radii=[0.015],
            labels=["world origin xyz=(0.000, 0.000, 0.000) m"],
        ),
        static=True,
    )


def _log_trajectory_curve(
    rr: Any,
    path: str,
    positions: np.ndarray,
    color: List[int],
    radius: float,
) -> None:
    if len(positions) > 1:
        rr.log(path, rr.LineStrips3D([positions.tolist()], colors=[color], radii=[radius]), static=True)
    elif len(positions) == 1:
        rr.log(path, rr.Points3D(positions.tolist(), colors=[color], radii=[radius]), static=True)


def _log_sampled_world_labels(
    rr: Any,
    path: str,
    positions: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    label_prefix: str,
    color: List[int],
    max_labels: int,
) -> None:
    sampled_indices = _sample_indices(len(positions), max_labels)
    if not sampled_indices:
        return
    labels = [
        _format_world_xyz_label(
            label_prefix,
            positions[index],
            frame_index=int(frame_indices[index]),
            timestamp=float(timestamps[index]),
        )
        for index in sampled_indices
    ]
    rr.log(
        path,
        rr.Points3D(
            positions[sampled_indices].tolist(),
            colors=[color for _ in sampled_indices],
            radii=[0.01 for _ in sampled_indices],
            labels=labels,
        ),
        static=True,
    )


def _log_i2rt_keyframe_axes(
    rr: Any,
    path: str,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    axis_scale: float,
) -> None:
    if len(positions) == 0:
        return
    keyframes: List[Tuple[str, int]] = [("first", 0)]
    if len(positions) > 1:
        keyframes.append(("last", len(positions) - 1))

    for label, index in keyframes:
        quat = np.asarray(quaternions_xyzw[index], dtype=np.float64).reshape(4)
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-9:
            continue
        quat = quat / quat_norm
        position = np.asarray(positions[index], dtype=np.float64).reshape(3)
        frame_index = int(frame_indices[index])
        timestamp = float(timestamps[index])
        frame_path = f"{path}/{label}_frame"
        rr.log(
            frame_path,
            rr.Transform3D(
                translation=position.tolist(),
                quaternion=quat.tolist(),
            ),
            static=True,
        )
        rr.log(
            f"{frame_path}/axes",
            rr.Arrows3D(
                vectors=[
                    [axis_scale, 0.0, 0.0],
                    [0.0, axis_scale, 0.0],
                    [0.0, 0.0, axis_scale],
                ],
                origins=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                colors=[
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 80, 255],
                ],
                labels=[
                    f"i2rt {label} +X",
                    f"i2rt {label} +Y",
                    f"i2rt {label} +Z",
                ],
                radii=[0.004, 0.004, 0.004],
            ),
            static=True,
        )
        rr.log(
            f"{frame_path}/label",
            rr.Points3D(
                [[0.0, 0.0, 0.0]],
                colors=[[255, 255, 255]],
                radii=[0.01],
                labels=[
                    _format_world_xyz_label(
                        f"i2rt_{label}",
                        position,
                        frame_index=frame_index,
                        timestamp=timestamp,
                    )
                ],
            ),
            static=True,
        )


def _format_range_summary(name: str, positions: np.ndarray) -> str:
    return (
        f"{name}: "
        f"x=[{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}], "
        f"y=[{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}], "
        f"z=[{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] m"
    )


def visualize_focus_and_tcp_trajectories(
    *,
    trajectory: pd.DataFrame,
    ik_trajectory: pd.DataFrame,
    rerun_mode: str,
    rerun_save_path: Optional[Path],
    rerun_connect_uri: Optional[str],
    rerun_serve_web_viewer: bool,
    coord_label_count: int,
) -> None:
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rerun visualization requires rerun-sdk. Install it in the active Python environment "
            "or run without --visualize."
        ) from exc

    from utils.rerun_visualization import RerunVisualizer, log_text_summary

    if rerun_mode == "save" and rerun_save_path is not None:
        rerun_save_path.parent.mkdir(parents=True, exist_ok=True)

    visualizer = RerunVisualizer(
        app_name="yamhead_focus_tcp_trajectory",
        spawn=rerun_mode == "spawn",
        operating_mode=rerun_mode,
        connect_uri=rerun_connect_uri,
        save_path=None if rerun_save_path is None else str(rerun_save_path),
        serve_web_viewer=rerun_serve_web_viewer,
    )
    if not visualizer.is_initialized:
        raise RuntimeError("Failed to initialize rerun visualization")

    focus_positions = _positions_from_columns(
        trajectory,
        ("center_world_x", "center_world_y", "center_world_z"),
    )
    tcp_positions = _positions_from_columns(
        ik_trajectory,
        ("ee_world_x", "ee_world_y", "ee_world_z"),
    )
    tcp_quaternions = ik_trajectory[
        ["ee_world_qx", "ee_world_qy", "ee_world_qz", "ee_world_qw"]
    ].to_numpy(dtype=np.float64)
    focus_timestamps = trajectory["timestamp"].to_numpy(dtype=np.float64)
    focus_frame_indices = trajectory["frame_index"].to_numpy(dtype=np.int64)
    tcp_timestamps = ik_trajectory["timestamp"].to_numpy(dtype=np.float64)
    tcp_frame_indices = ik_trajectory["frame_index"].to_numpy(dtype=np.int64)

    axis_scale = 0.25 * _trajectory_extent([focus_positions, tcp_positions])
    i2rt_axis_scale = max(0.05, 0.12 * _trajectory_extent([tcp_positions]))
    _log_world_axes(rr, axis_scale)

    focus_color = [255, 140, 0]
    tcp_color = [0, 200, 255]
    _log_trajectory_curve(rr, "world/focus_center/curve", focus_positions, focus_color, 0.005)
    _log_trajectory_curve(rr, "world/yamhead_tcp/curve", tcp_positions, tcp_color, 0.006)
    _log_sampled_world_labels(
        rr,
        "world/focus_center/world_xyz_labels",
        focus_positions,
        focus_timestamps,
        focus_frame_indices,
        "focus_center",
        focus_color,
        coord_label_count,
    )
    _log_sampled_world_labels(
        rr,
        "world/yamhead_tcp/world_xyz_labels",
        tcp_positions,
        tcp_timestamps,
        tcp_frame_indices,
        "yamhead_tcp",
        tcp_color,
        coord_label_count,
    )
    _log_i2rt_keyframe_axes(
        rr,
        "world/yamhead_tcp/i2rt_keyframe_axes",
        tcp_positions,
        tcp_quaternions,
        tcp_timestamps,
        tcp_frame_indices,
        i2rt_axis_scale,
    )

    first_x_axis = R.from_quat(tcp_quaternions[0] / np.linalg.norm(tcp_quaternions[0])).as_matrix()[:, 0]
    last_x_axis = R.from_quat(tcp_quaternions[-1] / np.linalg.norm(tcp_quaternions[-1])).as_matrix()[:, 0]

    summary_lines = [
        "YamHead focus/TCP trajectory in world frame",
        f"focus frames={len(focus_positions)}, tcp frames={len(tcp_positions)}",
        _format_range_summary("focus_center", focus_positions),
        _format_range_summary("yamhead_tcp", tcp_positions),
        f"i2rt first +X world=({first_x_axis[0]:.3f}, {first_x_axis[1]:.3f}, {first_x_axis[2]:.3f})",
        f"i2rt last +X world=({last_x_axis[0]:.3f}, {last_x_axis[1]:.3f}, {last_x_axis[2]:.3f})",
    ]
    log_text_summary("world/summary", "\n".join(summary_lines))

    for index, position in enumerate(focus_positions):
        rr.set_time("frame", sequence=int(focus_frame_indices[index]))
        rr.set_time("timestamp", timestamp=float(focus_timestamps[index]))
        rr.log(
            "world/focus_center/current",
            rr.Points3D(
                [position.tolist()],
                colors=[focus_color],
                radii=[0.018],
                labels=[
                    _format_world_xyz_label(
                        "focus_center",
                        position,
                        frame_index=int(focus_frame_indices[index]),
                        timestamp=float(focus_timestamps[index]),
                    )
                ],
            ),
        )

    for index, position in enumerate(tcp_positions):
        rr.set_time("frame", sequence=int(tcp_frame_indices[index]))
        rr.set_time("timestamp", timestamp=float(tcp_timestamps[index]))
        rr.log(
            "world/yamhead_tcp/current",
            rr.Points3D(
                [position.tolist()],
                colors=[tcp_color],
                radii=[0.018],
                labels=[
                    _format_world_xyz_label(
                        "yamhead_tcp",
                        position,
                        frame_index=int(tcp_frame_indices[index]),
                        timestamp=float(tcp_timestamps[index]),
                    )
                ],
            ),
        )

    logger.info("Wrote rerun visualization for focus center and YamHead TCP trajectories")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline compute YamHead focus-on-center target trajectory from an episode"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/devices/rizon_dual_robot_data_replay.yaml",
        help="Replay config YAML",
    )
    parser.add_argument("--replay-column", default=None, help="Override episode replay column")
    parser.add_argument(
        "--left-init-pose-wxyz",
        nargs=7,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Left replay init TCP in Flexiv/data format [x,y,z,qw,qx,qy,qz]",
    )
    parser.add_argument(
        "--right-init-pose-wxyz",
        nargs=7,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Right replay init TCP in Flexiv/data format [x,y,z,qw,qx,qy,qz]",
    )
    parser.add_argument(
        "--left-init-pose-xyzw",
        nargs=7,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="Left replay init TCP in RizonRobot home_pose format [x,y,z,qx,qy,qz,qw]",
    )
    parser.add_argument(
        "--right-init-pose-xyzw",
        nargs=7,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="Right replay init TCP in RizonRobot home_pose format [x,y,z,qx,qy,qz,qw]",
    )
    parser.add_argument(
        "--focus-origin-yam",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed focus origin in Yam frame. Defaults to yam_focus_position_xyz or [0,0,0].",
    )
    parser.add_argument(
        "--filter-tau",
        type=float,
        default=None,
        help="Optional EMA tau in seconds. Defaults to yam_head_filter_tau from config.",
    )
    parser.add_argument(
        "--output-csv",
        default="logs/yamhead_focus_trajectory.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-npz",
        default="logs/yamhead_focus_trajectory.npz",
        help="Output NPZ path with world_targets and yam_targets",
    )
    parser.add_argument(
        "--compute-ik",
        action="store_true",
        help="Also run offline YamHead FK/IK and output joint/end-effector trajectory ranges",
    )
    parser.add_argument(
        "--i2rt-root",
        default=None,
        help="Path to local i2rt root containing i2rt/robot_models. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--ik-solver",
        default="auto",
        help="qpsolvers backend for mink IK. Use 'auto' to pick an installed solver.",
    )
    parser.add_argument(
        "--output-ik-csv",
        default="logs/yamhead_ik_trajectory.csv",
        help="Output CSV path for offline IK trajectory",
    )
    parser.add_argument(
        "--output-ik-npz",
        default="logs/yamhead_ik_trajectory.npz",
        help="Output NPZ path for offline IK target and achieved poses",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize focus center and YamHead TCP world-frame trajectories with rerun",
    )
    parser.add_argument(
        "--rerun-mode",
        choices=("spawn", "connect_grpc", "serve_grpc", "save", "stdout", "none"),
        default="spawn",
        help="Rerun operating mode used by --visualize",
    )
    parser.add_argument(
        "--rerun-save-path",
        default="logs/yamhead_focus_tcp_trajectory.rrd",
        help="Output .rrd path when --rerun-mode=save",
    )
    parser.add_argument(
        "--rerun-connect-uri",
        default=None,
        help="Rerun gRPC URI when --rerun-mode=connect_grpc",
    )
    parser.add_argument(
        "--rerun-serve-web-viewer",
        action="store_true",
        help="Serve a web viewer when --rerun-mode=serve_grpc",
    )
    parser.add_argument(
        "--coord-label-count",
        type=int,
        default=12,
        help="Maximum sampled world-coordinate labels per trajectory in rerun",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_yaml(_resolve_repo_path(args.config))
    yam_cfg = dict(config.get("yam_head_tracking", {}).get("yam_head", {}))
    origin_value = (
        args.focus_origin_yam
        if args.focus_origin_yam is not None
        else yam_cfg.get("yam_focus_position_xyz", [0.0, 0.0, 0.0])
    )
    focus_origin_yam = _as_vector(origin_value, normalize=False, name="focus_origin_yam")
    filter_tau = float(args.filter_tau if args.filter_tau is not None else yam_cfg.get("yam_head_filter_tau", 0.0))
    left_init_pose_wxyz = _resolve_init_pose_wxyz(
        config=config,
        arm="left",
        pose_wxyz=args.left_init_pose_wxyz,
        pose_xyzw=args.left_init_pose_xyzw,
    )
    right_init_pose_wxyz = _resolve_init_pose_wxyz(
        config=config,
        arm="right",
        pose_wxyz=args.right_init_pose_wxyz,
        pose_xyzw=args.right_init_pose_xyzw,
    )

    trajectory, world_targets, yam_targets = compute_trajectory(
        config,
        replay_column=args.replay_column,
        left_init_pose_wxyz=left_init_pose_wxyz,
        right_init_pose_wxyz=right_init_pose_wxyz,
        focus_origin_yam=focus_origin_yam,
        filter_tau=filter_tau,
    )

    output_csv = _resolve_repo_path(args.output_csv)
    output_npz = _resolve_repo_path(args.output_npz)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output_csv, index=False)
    np.savez_compressed(
        output_npz,
        world_targets=world_targets,
        yam_targets=yam_targets,
        timestamps=trajectory["timestamp"].to_numpy(dtype=np.float64),
    )

    logger.info(f"Wrote YamHead focus trajectory CSV: {output_csv}")
    logger.info(f"Wrote YamHead focus trajectory NPZ: {output_npz}")
    logger.info(
        "Trajectory summary: "
        f"frames={len(trajectory)}, "
        f"distance_m=[{trajectory['distance_m'].min():.4f}, {trajectory['distance_m'].max():.4f}]"
    )

    run_ik = bool(args.compute_ik or args.visualize)
    if args.visualize and not args.compute_ik:
        logger.info("Enabling offline IK because --visualize includes the YamHead TCP trajectory")

    ik_trajectory: Optional[pd.DataFrame] = None
    if run_ik:
        ik_csv = _resolve_repo_path(args.output_ik_csv)
        ik_npz = _resolve_repo_path(args.output_ik_npz)
        ik_csv.parent.mkdir(parents=True, exist_ok=True)
        ik_npz.parent.mkdir(parents=True, exist_ok=True)
        ik_trajectory, ik_targets, ik_achieved = compute_yamhead_ik_trajectory(
            config=config,
            center_trajectory=trajectory,
            i2rt_root=_resolve_i2rt_root(args.i2rt_root),
            solver=str(args.ik_solver),
            focus_origin_yam=focus_origin_yam,
        )
        ik_trajectory.to_csv(ik_csv, index=False)
        np.savez_compressed(
            ik_npz,
            target_poses_yam=ik_targets,
            achieved_poses_yam=ik_achieved,
            timestamps=ik_trajectory["timestamp"].to_numpy(dtype=np.float64),
            joints=ik_trajectory[[f"q{i}" for i in range(1, 7)]].to_numpy(dtype=np.float64),
        )
        logger.info(f"Wrote YamHead IK trajectory CSV: {ik_csv}")
        logger.info(f"Wrote YamHead IK trajectory NPZ: {ik_npz}")
        logger.info(
            "IK summary: "
            f"frames={len(ik_trajectory)}, "
            f"success={int(ik_trajectory['ik_success'].sum())}, "
            f"accepted={int(ik_trajectory['ik_accepted'].sum())}, "
            f"held={int(ik_trajectory['held'].sum())}"
        )
        logger.info(
            "IK EE world range: "
            f"x=[{ik_trajectory['ee_world_x'].min():.4f}, {ik_trajectory['ee_world_x'].max():.4f}], "
            f"y=[{ik_trajectory['ee_world_y'].min():.4f}, {ik_trajectory['ee_world_y'].max():.4f}], "
            f"z=[{ik_trajectory['ee_world_z'].min():.4f}, {ik_trajectory['ee_world_z'].max():.4f}]"
        )
        joint_summary = ", ".join(
            f"q{i}=[{ik_trajectory[f'q{i}'].min():.4f}, {ik_trajectory[f'q{i}'].max():.4f}]"
            for i in range(1, 7)
        )
        logger.info(f"IK joint range: {joint_summary}")

    if args.visualize:
        if ik_trajectory is None:
            raise RuntimeError("YamHead TCP visualization requires offline IK trajectory")
        visualize_focus_and_tcp_trajectories(
            trajectory=trajectory,
            ik_trajectory=ik_trajectory,
            rerun_mode=str(args.rerun_mode),
            rerun_save_path=_resolve_repo_path(args.rerun_save_path),
            rerun_connect_uri=args.rerun_connect_uri,
            rerun_serve_web_viewer=bool(args.rerun_serve_web_viewer),
            coord_label_count=int(args.coord_label_count),
        )


if __name__ == "__main__":
    main()
