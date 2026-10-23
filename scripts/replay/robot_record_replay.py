#!/usr/bin/env python3
"""
Replay LeRobot episode TCP trajectories on a Flexiv Rizon dual-arm setup.

Expects parquet columns ``action`` or ``observation.state`` with per-arm layout
``[x, y, z, w, x, y, z]`` (position + quaternion wxyz). Supported dual-arm rows:
  - 14D: ``[left_7d, right_7d]`` (no gripper)
  - 16D: ``[left_7d, left_grip, right_7d, right_grip]``
  - 23D: ``[left_7d, left_grip, right_7d, right_grip, ego_7d]`` (UMI dual-arm; the
    trailing 7 ego dims are dropped, matching ``umi_dual_arm_policy``).

Episode poses are converted to relative transforms (first frame = identity):
``T_rel(t) = inv(T_abs(0)) @ T_abs(t)``. At replay time the robot command is
``T_target(t) = T_robot_init @ T_rel(t)``, where ``T_robot_init`` is the TCP
pose after connect/home.

Gripper values (indices 7 / 15) are replayed *absolutely* (no relativization),
sent through ``RizonRobot._move_gripper_value_by_backend`` using whatever
``robot.gripper.value_mode`` the config specifies. Gripper replay is gated by the
gripper being enabled in the robot config; if disabled, only TCP poses replay.

Uses the same robot connect / EEF-mode / home workflow as the other replay tools.

Author: Yichen Xie
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial.transform as st
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from devices.robot import RizonRobot
from utils.device_config import normalize_device_config
from utils.logger_config import logger
from utils.time_control import precise_sleep, precise_wait

LEROBOT_REPLAY_COLUMNS = ("action", "observation.state", "observation/state")


def _load_default_signal_files() -> Dict[str, str]:
    """Load runtime signal_files defaults (standalone scripts do not use Hydra)."""
    runtime_path = project_root / "configs" / "runtime" / "default.yaml"
    with open(runtime_path, "r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)
    signal_files = runtime_cfg.get("signal_files", {})
    if not isinstance(signal_files, dict) or not signal_files:
        raise ValueError(f"No signal_files found in {runtime_path}")
    return {str(k): str(v) for k, v in signal_files.items()}


def _resolve_signal_files(robot_config: Dict[str, Any]) -> Dict[str, str]:
    """Resolve signal_files for RizonRobot when YAML is loaded without Hydra."""
    raw = robot_config.get("signal_files")
    if isinstance(raw, dict) and raw:
        if any(str(v).startswith("${") for v in raw.values()):
            logger.warning(
                "robot.signal_files contains unresolved Hydra references; "
                "using configs/runtime/default.yaml"
            )
            return _load_default_signal_files()
        return {str(k): str(v) for k, v in raw.items()}
    logger.info("robot.signal_files not set; using configs/runtime/default.yaml")
    return _load_default_signal_files()


Pose7 = List[float]  # Flexiv Cartesian target: [x, y, z, qw, qx, qy, qz]
PoseMat = np.ndarray  # 4x4 homogeneous TCP transform


def _pose7_from_xyz_quat_wxyz(row: np.ndarray) -> Pose7:
    """Convert [x, y, z, w, x, y, z] to Flexiv tcp target."""
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size < 7:
        raise ValueError(f"Expected 7 pose values (xyz + wxyz), got {values.size}")
    pos = values[:3]
    quat_wxyz = values[3:7] / max(np.linalg.norm(values[3:7]), 1e-12)
    return [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    ]


def _split_lerobot_dual_arm_row(
    row: np.ndarray,
) -> Tuple[Pose7, Pose7, Optional[float], Optional[float]]:
    """
    Split a LeRobot state/action row into left/right Flexiv TCP targets + grippers.

    Quaternion per arm is wxyz: [x, y, z, w, x, y, z]. Gripper value (when present)
    is returned as-is (absolute); ego dims (23D case) are dropped.
      - 14D: [left_7d, right_7d]                              -> grippers None
      - 16D: [left_7d, left_grip, right_7d, right_grip]
      - 23D: [left_7d, left_grip, right_7d, right_grip, ego_7d]
    """
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size == 14:
        return (
            _pose7_from_xyz_quat_wxyz(values[0:7]),
            _pose7_from_xyz_quat_wxyz(values[7:14]),
            None,
            None,
        )
    if values.size == 16 or values.size == 23:
        return (
            _pose7_from_xyz_quat_wxyz(values[0:7]),
            _pose7_from_xyz_quat_wxyz(values[8:15]),
            float(values[7]),
            float(values[15]),
        )
    raise ValueError(
        "LeRobot dual-arm row must be 14D (no gripper), 16D (with gripper), "
        f"or 23D (with gripper + ego), got {values.size}"
    )


def _pose7_to_matrix(pose7: Pose7) -> PoseMat:
    """Convert Flexiv TCP pose to a 4x4 homogeneous matrix."""
    values = np.asarray(pose7, dtype=np.float64).reshape(-1)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = values[:3]
    quat_wxyz = values[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    mat[:3, :3] = st.Rotation.from_quat(quat_xyzw).as_matrix()
    return mat


def _matrix_to_pose7(mat: PoseMat) -> Pose7:
    """Convert a 4x4 homogeneous matrix to Flexiv TCP pose."""
    pos = mat[:3, 3]
    quat_xyzw = st.Rotation.from_matrix(mat[:3, :3]).as_quat()
    return [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def _tcp_array_to_pose7(tcp_pose: np.ndarray) -> Pose7:
    """Convert Flexiv ``states().tcp_pose`` array to Pose7."""
    values = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
    if values.size < 7:
        raise ValueError(f"Flexiv tcp_pose must have 7 elements, got {values.size}")
    return [float(v) for v in values[:7]]


def _absolute_poses_to_relative_mats(poses: List[Pose7]) -> List[PoseMat]:
    """
    Express poses relative to the first frame (first frame becomes identity).

    T_rel(t) = inv(T_abs(0)) @ T_abs(t)
    """
    mats = [_pose7_to_matrix(pose) for pose in poses]
    init_inv = np.linalg.inv(mats[0])
    return [init_inv @ mat for mat in mats]


def _detect_replay_column(columns: List[str], dataset_cfg: Dict[str, Any]) -> str:
    """Return the LeRobot parquet column used for replay."""
    replay_column = dataset_cfg.get("replay_column")
    if replay_column is not None:
        name = str(replay_column)
        if name not in columns:
            raise ValueError(f"dataset.replay_column '{name}' not in parquet columns: {columns}")
        return name
    for name in LEROBOT_REPLAY_COLUMNS:
        if name in columns:
            return name
    raise ValueError(
        f"No LeRobot replay column found (tried {LEROBOT_REPLAY_COLUMNS}). "
        f"Available columns: {columns}. Set dataset.replay_column in config."
    )


def _load_episode_trajectory(
    dataset_cfg: Dict[str, Any],
    replay_cfg: Dict[str, Any],
) -> Tuple[List[PoseMat], List[PoseMat], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Load left/right relative TCP transforms, absolute grippers, and timestamps.

    Grippers are ``None`` when the episode has no gripper channel (14D rows).
    """
    data_path = Path(str(dataset_cfg["data_file"]))
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read parquet episodes") from exc

    logger.info(f"Loading LeRobot episode from {data_path}")
    if data_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet episodes are supported, got: {data_path.suffix}")
    table = pd.read_parquet(data_path)

    start_index = int(replay_cfg.get("start_index", 0))
    end_index = replay_cfg.get("end_index")
    if end_index is None:
        end_index = len(table)
    else:
        end_index = int(end_index)
    table = table.iloc[start_index:end_index].reset_index(drop=True)
    num_frames = len(table)
    if num_frames == 0:
        raise ValueError("No frames selected for replay")

    replay_col = _detect_replay_column(list(table.columns), dataset_cfg)
    logger.info(f"Replay column: {replay_col}, frames: {num_frames}")

    array = np.asarray(table[replay_col].to_list(), dtype=object)
    if array.ndim == 1 and array.dtype == object:
        array = np.stack([np.asarray(row, dtype=np.float64) for row in array], axis=0)
    else:
        array = np.asarray(array, dtype=np.float64)

    left_abs_poses: List[Pose7] = []
    right_abs_poses: List[Pose7] = []
    left_grips_list: List[Optional[float]] = []
    right_grips_list: List[Optional[float]] = []
    for index in range(num_frames):
        left_pose, right_pose, left_grip, right_grip = _split_lerobot_dual_arm_row(array[index])
        left_abs_poses.append(left_pose)
        right_abs_poses.append(right_pose)
        left_grips_list.append(left_grip)
        right_grips_list.append(right_grip)

    has_gripper = all(g is not None for g in left_grips_list) and all(
        g is not None for g in right_grips_list
    )
    if has_gripper:
        left_grips: Optional[np.ndarray] = np.asarray(left_grips_list, dtype=np.float64)
        right_grips: Optional[np.ndarray] = np.asarray(right_grips_list, dtype=np.float64)
        logger.info(
            "Episode has gripper channel: "
            f"left range=[{left_grips.min():.4f}, {left_grips.max():.4f}], "
            f"right range=[{right_grips.min():.4f}, {right_grips.max():.4f}]"
        )
    else:
        left_grips = None
        right_grips = None
        logger.info("Episode has no gripper channel (14D rows); gripper replay unavailable")

    left_rel_mats = _absolute_poses_to_relative_mats(left_abs_poses)
    right_rel_mats = _absolute_poses_to_relative_mats(right_abs_poses)
    logger.info(
        "Converted episode poses to relative transforms (T_rel = inv(T_0) @ T_t, first frame = I)"
    )

    timestamp_col = dataset_cfg.get("timestamp_column")
    if timestamp_col and timestamp_col in table.columns:
        timestamps = table[timestamp_col].to_numpy(dtype=np.float64)
        if timestamps.size > 1:
            timestamps = timestamps - timestamps[0]
        else:
            timestamps = np.arange(num_frames, dtype=np.float64) / float(
                replay_cfg.get("frequency", 40.0)
            )
    else:
        frequency = float(replay_cfg.get("frequency", dataset_cfg.get("frequency", 40.0)))
        timestamps = np.arange(num_frames, dtype=np.float64) / frequency

    return left_rel_mats, right_rel_mats, left_grips, right_grips, timestamps


class RobotRecordReplayer:
    """Replay recorded TCP poses on a connected Rizon robot."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()
        self.robot: Optional[RizonRobot] = None
        self.replay_frequency: float = float(
            self.config.get("replay", {}).get(
                "frequency",
                self.config.get("test", {}).get("frequency", 40.0),
            )
        )
        self.loop: bool = bool(self.config.get("replay", {}).get("loop", False))
        self.record_actual_tcp: bool = bool(
            self.config.get("replay", {}).get("record_actual_tcp", False)
        )
        self.robot_init_left_mat: Optional[PoseMat] = None
        self.robot_init_right_mat: Optional[PoseMat] = None
        self.replay_gripper: bool = False

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError("Configuration root must be a mapping")
        robot_config = config.get("robot")
        if not isinstance(robot_config, dict):
            raise TypeError("robot configuration must be a mapping")
        config["robot"] = normalize_device_config(robot_config)
        logger.info(f"Configuration loaded from {self.config_path}")
        return config

    def initialize_robot(self) -> bool:
        try:
            robot_config: Dict[str, Any] = self.config["robot"]
            signal_files = _resolve_signal_files(robot_config)
            self.robot = RizonRobot(
                device_class_id=int(robot_config["device_class_id"]),
                robot_sn=robot_config["robot_sn"],
                state_feedback=robot_config["state_feedback"],
                control=robot_config["control"],
                home=robot_config.get("home"),
                gripper=robot_config.get("gripper"),
                motion_limits=robot_config.get("motion_limits"),
                network_interface_whitelist=robot_config.get(
                    "network_interface_whitelist"
                ),
                signal_files=signal_files,
            )

            logger.info("Connecting to robot...")
            if not self.robot._connect_robot():
                raise RuntimeError("Failed to connect to robot")

            logger.info("Switching to EEF control mode...")
            if not self.robot._switch_to_eef_control():
                raise RuntimeError("Failed to switch to EEF control mode")

            logger.info("Setting home pose...")
            if not self.robot._set_home_pose():
                raise RuntimeError("Failed to set home pose")

            self.replay_gripper = bool(getattr(self.robot, "utilize_gripper", False))
            logger.info(
                f"Gripper replay {'enabled' if self.replay_gripper else 'disabled'} "
                f"(value_mode={self.robot.gripper_config.value_mode})"
            )

            self._capture_robot_init_poses()
            logger.info("Robot initialized for replay")
            return True
        except Exception as exc:
            logger.error(f"Failed to initialize robot: {exc}")
            return False

    def _capture_robot_init_poses(self) -> None:
        """Read current robot TCP poses used as T_robot_init in T_target = T_init @ T_rel."""
        if self.robot is None or self.robot.robot_left is None or self.robot.robot_right is None:
            raise RuntimeError("Robot is not connected")
        left_tcp = self.robot.robot_left.states().tcp_pose
        right_tcp = self.robot.robot_right.states().tcp_pose
        self.robot_init_left_mat = _pose7_to_matrix(_tcp_array_to_pose7(left_tcp))
        self.robot_init_right_mat = _pose7_to_matrix(_tcp_array_to_pose7(right_tcp))
        logger.info(
            "Captured robot init TCP: "
            f"left=[{self.robot_init_left_mat[0, 3]:.4f}, {self.robot_init_left_mat[1, 3]:.4f}, "
            f"{self.robot_init_left_mat[2, 3]:.4f}], "
            f"right=[{self.robot_init_right_mat[0, 3]:.4f}, {self.robot_init_right_mat[1, 3]:.4f}, "
            f"{self.robot_init_right_mat[2, 3]:.4f}]"
        )

    def _compose_robot_target(self, rel_mat: PoseMat, init_mat: PoseMat) -> Pose7:
        """T_target = T_robot_init @ T_rel."""
        return _matrix_to_pose7(init_mat @ rel_mat)

    def _send_dual(self, left_target: Pose7, right_target: Pose7) -> None:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized")
        if not self.robot.is_dual_arm:
            raise RuntimeError("Dual-arm replay requires a dual-arm robot config")
        if self.robot.robot_left is None or self.robot.robot_right is None:
            raise RuntimeError("Dual-arm robot interfaces are not available")
        self.robot.robot_left.send_cartesian_motion_force(
            left_target,
            max_linear_vel=self.robot.max_linear_vel,
            max_linear_acc=self.robot.max_linear_acc,
            max_angular_vel=self.robot.max_angular_vel,
            max_angular_acc=self.robot.max_angular_acc,
        )
        self.robot.robot_right.send_cartesian_motion_force(
            right_target,
            max_linear_vel=self.robot.max_linear_vel,
            max_linear_acc=self.robot.max_linear_acc,
            max_angular_vel=self.robot.max_angular_vel,
            max_angular_acc=self.robot.max_angular_acc,
        )

    def _send_gripper(self, left_grip: float, right_grip: float) -> None:
        """Command absolute gripper values via the robot's configured backend/value_mode."""
        if self.robot is None:
            return
        gripper_active_by_arm = [arm.enabled for arm in self.robot.gripper_config.arms]
        if getattr(self.robot, "gripper_left", None) is not None and gripper_active_by_arm[0]:
            self.robot._move_gripper_value_by_backend(
                self.robot.gripper_left, float(left_grip), 0, "left"
            )
        if getattr(self.robot, "gripper_right", None) is not None and gripper_active_by_arm[1]:
            self.robot._move_gripper_value_by_backend(
                self.robot.gripper_right, float(right_grip), 1, "right"
            )

    def _read_actual_tcp_dual(self) -> Tuple[float, float, float]:
        if self.robot is None or self.robot.robot_left is None or self.robot.robot_right is None:
            raise RuntimeError("Robot is not available for TCP readback")
        left_tcp = self.robot.robot_left.states().tcp_pose
        right_tcp = self.robot.robot_right.states().tcp_pose
        return float(left_tcp[0]), float(right_tcp[0]), float(time.time())

    def run_replay(self) -> bool:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized")

        if self.robot_init_left_mat is None or self.robot_init_right_mat is None:
            raise RuntimeError("Robot init poses not captured; call initialize_robot() first")

        dataset_cfg = self.config.get("dataset", self.config)
        replay_cfg = self.config.get("replay", {})
        left_rel_mats, right_rel_mats, left_grips, right_grips, timestamps = _load_episode_trajectory(
            dataset_cfg, replay_cfg
        )

        if not self.robot.is_dual_arm:
            raise ValueError("LeRobot dual-arm replay requires a dual-arm robot config")
        if len(left_rel_mats) != len(right_rel_mats):
            raise ValueError("Left/right pose trajectories must have the same length")

        send_gripper = self.replay_gripper and left_grips is not None and right_grips is not None
        if self.replay_gripper and left_grips is None:
            logger.warning(
                "Gripper enabled on robot but episode has no gripper channel; "
                "replaying TCP poses only"
            )

        num_frames = len(left_rel_mats)
        logger.info(
            f"Starting replay: frames={num_frames}, rate={self.replay_frequency} Hz, "
            f"loop={self.loop}, gripper={'on' if send_gripper else 'off'}"
        )

        recorded_actual: List[Tuple[float, float, float]] = []
        start_wall = time.time()
        frame_index = 0

        try:
            while True:
                if frame_index >= num_frames:
                    if not self.loop:
                        break
                    frame_index = 0
                    start_wall = time.time()

                precise_wait(start_wall + float(timestamps[frame_index]), time_func=time.time)
                left_target = self._compose_robot_target(
                    left_rel_mats[frame_index], self.robot_init_left_mat
                )
                right_target = self._compose_robot_target(
                    right_rel_mats[frame_index], self.robot_init_right_mat
                )
                self._send_dual(left_target, right_target)

                if send_gripper:
                    self._send_gripper(
                        float(left_grips[frame_index]), float(right_grips[frame_index])
                    )

                if self.record_actual_tcp:
                    left_x, right_x, sample_time = self._read_actual_tcp_dual()
                    recorded_actual.append((sample_time, left_x, right_x))

                frame_index += 1
        except KeyboardInterrupt:
            logger.info("Replay interrupted by user")
            return False

        logger.info(f"Replay finished ({num_frames} frames)")
        if recorded_actual:
            logger.info(f"Recorded {len(recorded_actual)} actual TCP samples during replay")
        return True

    def cleanup(self) -> None:
        try:
            if self.robot is not None and getattr(self.robot, "is_dual_arm", None) is not None:
                try:
                    if hasattr(self.robot, "_go_home"):
                        self.robot._go_home()
                    precise_sleep(2.0)
                except Exception as exc:
                    logger.warning(f"Failed to move robot home: {exc}")
                try:
                    self.robot.stop_server()
                except Exception as exc:
                    logger.warning(f"Failed to stop robot server: {exc}")
        except Exception as exc:
            logger.error(f"Cleanup error: {exc}")

    def run(self) -> bool:
        try:
            if not self.initialize_robot():
                return False
            return self.run_replay()
        finally:
            self.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay LeRobot parquet TCP poses on Flexiv Rizon (eef_abs, wxyz)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/devices/rizon_dual_robot_data_replay.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--frequency", type=float, default=None, help="Override replay rate [Hz]")
    parser.add_argument("--data-file", type=str, default=None, help="Override dataset parquet path")
    parser.add_argument(
        "--replay-column",
        type=str,
        default=None,
        help="Override LeRobot column (action or observation.state)",
    )
    parser.add_argument("--loop", action="store_true", help="Loop trajectory after it ends")
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print parquet columns and exit (no robot connection)",
    )

    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    replayer = RobotRecordReplayer(str(config_path))
    if args.frequency is not None:
        replayer.replay_frequency = float(args.frequency)
        replayer.config.setdefault("replay", {})["frequency"] = float(args.frequency)
    if args.data_file is not None:
        replayer.config.setdefault("dataset", {})["data_file"] = args.data_file
    if args.replay_column is not None:
        replayer.config.setdefault("dataset", {})["replay_column"] = args.replay_column
    if args.loop:
        replayer.loop = True
        replayer.config.setdefault("replay", {})["loop"] = True

    if args.list_columns:
        dataset_cfg = replayer.config.get("dataset", replayer.config)
        data_path = Path(str(dataset_cfg["data_file"]))
        if not data_path.is_absolute():
            data_path = project_root / data_path
        import pandas as pd

        table = pd.read_parquet(data_path)
        logger.info(f"Columns in {data_path}:")
        for name in table.columns:
            logger.info(f"  - {name} ({table[name].dtype})")
        try:
            replay_col = _detect_replay_column(list(table.columns), dataset_cfg)
            sample = np.asarray(table[replay_col].iloc[0], dtype=np.float64).reshape(-1)
            logger.info(f"Replay column: {replay_col}, first row dim: {sample.size}")
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)
        sys.exit(0)

    try:
        success = replayer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        replayer.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
