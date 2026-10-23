"""YamHead runtime tracking strategies."""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from devices.config import ShmEndpointConfig
from utils.logger_config import logger
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_device_shm,
    connect_to_shared_memory,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
)


PathLike = Optional[Union[str, Path]]
Vector3Like = Union[np.ndarray, Tuple[float, float, float], List[float]]


class _YamPoseEMA:
    """Low-pass filter for Cartesian YAM targets."""

    def __init__(self, tau: float):
        self.tau = float(tau)
        self._T: Optional[np.ndarray] = None
        self._t_last: Optional[float] = None

    def reset(self) -> None:
        self._T = None
        self._t_last = None

    def update(self, T_now: np.ndarray, t_now: Optional[float] = None) -> np.ndarray:
        if self.tau <= 0.0:
            return T_now
        if t_now is None:
            t_now = time.monotonic()
        if self._T is None:
            self._T = T_now.copy()
            self._t_last = t_now
            return T_now.copy()
        dt = max(t_now - (self._t_last or t_now), 1e-6)
        self._t_last = t_now
        alpha = dt / (dt + self.tau)
        p_new = (1.0 - alpha) * self._T[:3, 3] + alpha * T_now[:3, 3]
        R_prev = self._T[:3, :3]
        R_now = T_now[:3, :3]
        R_delta = R_now @ R_prev.T
        rotvec = R.from_matrix(R_delta).as_rotvec()
        T_smoothed = np.eye(4)
        T_smoothed[:3, :3] = R.from_rotvec(rotvec * alpha).as_matrix() @ R_prev
        T_smoothed[:3, 3] = p_new
        self._T = T_smoothed
        return T_smoothed.copy()


class YamHeadTrackingStrategy:
    """Base interface for YamHead runtime target generators."""

    def wait_until_ready(self, is_running: Callable[[], bool]) -> None:
        raise NotImplementedError

    def step(
        self,
        *,
        q_target: np.ndarray,
        yam_kin: Any,
        robot_q_to_model_qpos: Callable[[np.ndarray], np.ndarray],
    ) -> Optional[np.ndarray]:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class RobotCenterDecayTracking(YamHeadTrackingStrategy):
    """Track the midpoint of a dual-arm robot TCP pair with smoothed noisy focus."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        repo_root: Path,
        left2world_json_path: PathLike,
        right2world_json_path: PathLike,
        yam2world_json_path: PathLike,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.robot_state_device_name = str(
            config.get("robot_state_device_name", "RizonRobot")
        )
        self.robot_state_device_class_id = int(config.get("robot_state_device_class_id", 2))
        self.robot_center_offset_xyz = self._as_unit_or_vector(
            config.get("robot_center_offset_xyz", (0.0, 0.0, 0.0)),
            normalize=False,
            name="robot_center_offset_xyz",
        )
        focus_position = config.get("yam_focus_position_xyz")
        self.yam_focus_position_xyz = (
            None
            if focus_position is None
            else self._as_unit_or_vector(
                focus_position, normalize=False, name="yam_focus_position_xyz"
            )
        )
        self.yam_focus_local_forward = self._as_unit_or_vector(
            config.get("yam_focus_local_forward", (0.0, 0.0, 1.0)),
            normalize=True,
            name="yam_focus_local_forward",
        )
        self.yam_focus_local_up = self._as_unit_or_vector(
            config.get("yam_focus_local_up", (-1.0, 0.0, 0.0)),
            normalize=True,
            name="yam_focus_local_up",
        )
        self.yam_focus_world_up = self._as_unit_or_vector(
            config.get("yam_focus_world_up", (0.0, 0.0, 1.0)),
            normalize=True,
            name="yam_focus_world_up",
        )
        self.yam_focus_target_noise_std_xyz = self._as_unit_or_vector(
            config.get("yam_focus_target_noise_std_xyz", (0.0, 0.0, 0.0)),
            normalize=False,
            name="yam_focus_target_noise_std_xyz",
        )
        if np.any(self.yam_focus_target_noise_std_xyz < 0.0):
            raise ValueError(
                "yam_focus_target_noise_std_xyz values must be non-negative"
            )
        self.yam_focus_target_noise_tau_s = float(
            config.get("yam_focus_target_noise_tau_s", 1.0)
        )
        self._focus_noise_rng = np.random.default_rng(
            config.get("yam_focus_target_noise_seed")
        )
        self._focus_noise_xyz = np.zeros(3, dtype=np.float64)
        self._focus_noise_last_time: Optional[float] = None
        self.yam_focus_world_y_pitch_deg = float(
            config.get("yam_focus_world_y_pitch_deg", 0.0)
        )
        self._head_filter = _YamPoseEMA(float(config.get("yam_head_filter_tau", 0.06)))

        world_transform_paths = (
            left2world_json_path,
            right2world_json_path,
            yam2world_json_path,
        )
        if any(path is not None for path in world_transform_paths) and not all(
            path is not None for path in world_transform_paths
        ):
            raise ValueError(
                "left2world_json_path, right2world_json_path, and yam2world_json_path "
                "must be configured together"
            )
        self.left2world_T = self._load_world_transform(
            left2world_json_path, "left2world"
        )
        self.right2world_T = self._load_world_transform(
            right2world_json_path, "right2world"
        )
        self.yam2world_T = self._load_world_transform(yam2world_json_path, "yam2world")
        self.world2yam_T = np.linalg.inv(self.yam2world_T)
        self._robot_state_shm = None

    @staticmethod
    def _orthonormalize(R3: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(R3)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1.0
            Rn = U @ Vt
        return Rn

    @staticmethod
    def _as_unit_or_vector(
        values: Vector3Like, *, normalize: bool, name: str
    ) -> np.ndarray:
        vec = np.asarray(values, dtype=np.float64).reshape(-1)
        if vec.shape[0] != 3:
            raise ValueError(
                f"{name} must contain exactly 3 values, got {vec.shape[0]}"
            )
        if not normalize:
            return vec
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            raise ValueError(f"{name} must be non-zero")
        return vec / norm

    @staticmethod
    def _transform_point(T: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        return (np.asarray(T, dtype=np.float64).reshape(4, 4) @ point_h)[:3]

    def _resolve_optional_repo_path(self, path_value: PathLike) -> Optional[Path]:
        if path_value is None or str(path_value).strip() == "":
            return None
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path

    def _load_world_transform(self, path_value: PathLike, key: str) -> np.ndarray:
        path = self._resolve_optional_repo_path(path_value)
        if path is None:
            return np.eye(4, dtype=np.float64)
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        node = data.get(key, data)
        if "matrix" not in node:
            raise ValueError(f"World transform file {path} is missing '{key}.matrix'")
        matrix = np.asarray(node["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"World transform file {path} must contain a 4x4 matrix, got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"World transform file {path} contains non-finite values")
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            raise ValueError(
                f"World transform file {path} must have homogeneous last row [0, 0, 0, 1]"
            )
        logger.info(f"[yam-focus] loaded {key} from {path}")
        return matrix

    @staticmethod
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
        self, origin: np.ndarray, target: np.ndarray
    ) -> Optional[np.ndarray]:
        direction = np.asarray(target, dtype=np.float64).reshape(3) - np.asarray(
            origin, dtype=np.float64
        ).reshape(3)
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            return None
        desired_forward = direction / distance
        desired_up = self._project_perpendicular(
            self.yam_focus_world_up, desired_forward
        )
        local_up = self._project_perpendicular(
            self.yam_focus_local_up, self.yam_focus_local_forward
        )
        rotation, _ = R.align_vectors(
            np.vstack([desired_forward, desired_up]),
            np.vstack([self.yam_focus_local_forward, local_up]),
        )
        return rotation.as_matrix()

    def _apply_focus_world_y_pitch(self, rotation_world: np.ndarray) -> np.ndarray:
        if abs(self.yam_focus_world_y_pitch_deg) < 1e-9:
            return rotation_world
        pitch_world = R.from_rotvec(
            np.deg2rad(self.yam_focus_world_y_pitch_deg)
            * np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ).as_matrix()
        return self._orthonormalize(pitch_world @ rotation_world)

    def _connect_robot_state_shm(self) -> None:
        if self._robot_state_shm is not None:
            return
        try:
            self._robot_state_shm = connect_to_device_shm(
                device_name=self.robot_state_device_name,
                device_class_id=self.robot_state_device_class_id,
                read_only=True,
            )
            logger.info(
                f"[yam-focus] connected robot state SHM: "
                f"{self.robot_state_device_name}_{self.robot_state_device_class_id}_data"
            )
        except FileNotFoundError:
            logger.warning(
                f"[yam-focus] robot state SHM not found: "
                f"{self.robot_state_device_name}_{self.robot_state_device_class_id}_data"
            )
        except Exception as exc:
            logger.warning(f"[yam-focus] failed to connect robot state SHM: {exc}")

    def wait_until_ready(self, is_running: Callable[[], bool]) -> None:
        logger.disable("utils.shm_utils")
        try:
            while is_running() and self._robot_state_shm is None:
                self._connect_robot_state_shm()
                if self._robot_state_shm is None:
                    time.sleep(0.1)
        finally:
            logger.enable("utils.shm_utils")
        logger.info(
            f"[yam-focus] tracking center of "
            f"{self.robot_state_device_name}_{self.robot_state_device_class_id}"
        )

    def _read_latest_robot_tcp_pair(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self._connect_robot_state_shm()
        if self._robot_state_shm is None:
            return None, None
        try:
            buf = self._robot_state_shm.buf
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            buffer_header = unpack_buffer_header(
                buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
            if buffer_header["current_frames_count"] == 0:
                return None, None
            latest_index = (buffer_header["write_index"] - 1) % buffer_header[
                "buffer_size"
            ]
            frame_size = device_header["frame_size"]
            frame_offset = (
                DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
            )
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_size = frame_size - FRAME_HEADER_SIZE
            dtype = get_dtype(device_header["data_dtype"], np.float64)
            flat = (
                np.frombuffer(
                    buf[data_start : data_start + data_size],
                    dtype=dtype,
                )
                .reshape(-1)
                .copy()
            )
            if flat.shape[0] < 44:
                logger.warning(
                    f"[yam-focus] robot state too small for dual TCPs: {flat.shape[0]}"
                )
                return None, None
            return flat[14:21], flat[36:43]
        except Exception as exc:
            logger.warning(f"[yam-focus] failed to read robot state: {exc}")
            return None, None

    def _robot_center_target_world(self) -> Optional[np.ndarray]:
        left_tcp, right_tcp = self._read_latest_robot_tcp_pair()
        if left_tcp is None or right_tcp is None:
            return None
        left_world = self._transform_point(self.left2world_T, left_tcp[:3])
        right_world = self._transform_point(self.right2world_T, right_tcp[:3])
        return 0.5 * (left_world + right_world) + self.robot_center_offset_xyz

    def _target_world(self) -> Optional[np.ndarray]:
        return self._robot_center_target_world()

    def _focus_target_noise_world(self, now: float) -> np.ndarray:
        """Generate smooth zero-mean noise for the focus target in world coordinates."""
        std = self.yam_focus_target_noise_std_xyz
        if self._focus_noise_last_time is None:
            self._focus_noise_last_time = now
            return self._focus_noise_xyz.copy()

        dt = max(now - self._focus_noise_last_time, 1e-6)
        self._focus_noise_last_time = now
        if self.yam_focus_target_noise_tau_s <= 0.0:
            self._focus_noise_xyz = self._focus_noise_rng.normal(0.0, std)
        else:
            decay = float(np.exp(-dt / self.yam_focus_target_noise_tau_s))
            innovation_scale = np.sqrt(max(1.0 - decay * decay, 0.0))
            self._focus_noise_xyz = (
                decay * self._focus_noise_xyz
                + innovation_scale * self._focus_noise_rng.normal(0.0, std)
            )

        limit = 3.0 * std
        return np.clip(self._focus_noise_xyz, -limit, limit)

    def _focus_origin_world(
        self,
        *,
        target_world: np.ndarray,
        q_target: np.ndarray,
        yam_kin: Any,
        robot_q_to_model_qpos: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        del target_world
        T_yam_now = yam_kin.fk(robot_q_to_model_qpos(q_target))
        focus_origin_yam = (
            np.asarray(self.yam_focus_position_xyz, dtype=np.float64)
            if self.yam_focus_position_xyz is not None
            else np.asarray(T_yam_now[:3, 3], dtype=np.float64)
        )
        return self._transform_point(self.yam2world_T, focus_origin_yam)

    def step(
        self,
        *,
        q_target: np.ndarray,
        yam_kin: Any,
        robot_q_to_model_qpos: Callable[[np.ndarray], np.ndarray],
    ) -> Optional[np.ndarray]:
        target_world = self._target_world()
        if target_world is None:
            return None
        now = time.monotonic()
        target_world = target_world + self._focus_target_noise_world(now)
        focus_origin_world = self._focus_origin_world(
            target_world=target_world,
            q_target=q_target,
            yam_kin=yam_kin,
            robot_q_to_model_qpos=robot_q_to_model_qpos,
        )
        R_focus_world = self._look_at_rotation(focus_origin_world, target_world)
        if R_focus_world is None:
            return None
        R_focus_world = self._apply_focus_world_y_pitch(R_focus_world)
        T_world_target = np.eye(4)
        T_world_target[:3, :3] = R_focus_world
        T_world_target[:3, 3] = focus_origin_world
        T_yam_target = self.world2yam_T @ T_world_target
        return self._head_filter.update(T_yam_target, t_now=now)

    def reset(self) -> None:
        self._head_filter.reset()
        self._focus_noise_xyz = np.zeros(3, dtype=np.float64)
        self._focus_noise_last_time = None

    def close(self) -> None:
        if self._robot_state_shm is None:
            return
        try:
            self._robot_state_shm.close()
        except Exception as exc:
            logger.warning(f"Failed to close YamHead tracking SHM: {exc}")
        finally:
            self._robot_state_shm = None


class PlannedRobotCenterTracking(RobotCenterDecayTracking):
    """Track the midpoint of the latest robot control command, with actual-state fallback."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        repo_root: Path,
        left2world_json_path: PathLike,
        right2world_json_path: PathLike,
        yam2world_json_path: PathLike,
    ) -> None:
        super().__init__(
            config,
            repo_root=repo_root,
            left2world_json_path=left2world_json_path,
            right2world_json_path=right2world_json_path,
            yam2world_json_path=yam2world_json_path,
        )
        if "control_shm_name" in config:
            raise ValueError(
                "yam_head_tracking.control_shm_name is removed; "
                "use control_endpoint"
            )
        if "control_endpoint" not in config:
            raise ValueError(
                "planned YamHead tracking requires control_endpoint"
            )
        self.control_endpoint = ShmEndpointConfig.from_mapping(
            config["control_endpoint"],
            field="yam_head_tracking.control_endpoint",
            allowed_ports=("command",),
        )
        if self.control_endpoint.device == "self":
            raise ValueError(
                "yam_head_tracking.control_endpoint.device must name the robot device"
            )
        self.control_shm_name = self.control_endpoint.resolve_shm_name(
            owner_device="YamHeadTracking",
            owner_device_class_id=0,
        )
        self.control_dtype = config.get("control_dtype")
        self.min_control_timestamp_ns = int(
            config.get("min_control_timestamp_ns", 0) or 0
        )
        self._control_shm = None

    def _connect_control_shm(self) -> None:
        if self._control_shm is not None:
            return
        try:
            self._control_shm = connect_to_shared_memory(
                self.control_shm_name, read_only=True
            )
            logger.info(
                f"[yam-focus] connected planned control SHM: {self.control_shm_name}"
            )
        except FileNotFoundError:
            logger.warning(
                f"[yam-focus] planned control SHM not found: {self.control_shm_name}"
            )
        except Exception as exc:
            logger.warning(f"[yam-focus] failed to connect planned control SHM: {exc}")

    def wait_until_ready(self, is_running: Callable[[], bool]) -> None:
        logger.disable("utils.shm_utils")
        try:
            while is_running() and self._control_shm is None:
                self._connect_control_shm()
                if self._control_shm is None:
                    time.sleep(0.1)
        finally:
            logger.enable("utils.shm_utils")
        logger.info(f"[yam-focus] tracking planned center from {self.control_shm_name}")

    def _read_latest_control_array(self) -> Optional[np.ndarray]:
        self._connect_control_shm()
        if self._control_shm is None:
            return None
        try:
            buf = self._control_shm.buf
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            buffer_header = unpack_buffer_header(
                buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
            if buffer_header["current_frames_count"] == 0:
                return None

            latest_index = (buffer_header["write_index"] - 1) % buffer_header[
                "buffer_size"
            ]
            frame_size = device_header["frame_size"]
            frame_offset = (
                DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
            )
            timestamp_ns = int.from_bytes(
                buf[frame_offset : frame_offset + FRAME_HEADER_SIZE], byteorder="little"
            )
            if timestamp_ns < self.min_control_timestamp_ns:
                return None

            data_start = frame_offset + FRAME_HEADER_SIZE
            data_size = frame_size - FRAME_HEADER_SIZE
            dtype = get_dtype(
                self.control_dtype or device_header["data_dtype"], np.float64
            )
            return (
                np.frombuffer(
                    buf[data_start : data_start + data_size],
                    dtype=dtype,
                )
                .reshape(-1)
                .copy()
            )
        except Exception as exc:
            logger.warning(f"[yam-focus] failed to read planned control SHM: {exc}")
            return None

    def _planned_dual_arm_center_world(
        self, control_array: np.ndarray
    ) -> Optional[np.ndarray]:
        flat = np.asarray(control_array, dtype=np.float64).reshape(-1)
        if flat.shape[0] >= 16:
            left_pos = flat[:3]
            right_pos = flat[8:11]
        elif flat.shape[0] >= 14:
            left_pos = flat[:3]
            right_pos = flat[7:10]
        else:
            logger.warning(
                f"[yam-focus] planned control too small for dual TCPs: {flat.shape[0]}"
            )
            return None

        if not np.all(np.isfinite(left_pos)) or not np.all(np.isfinite(right_pos)):
            logger.warning(
                "[yam-focus] planned control contains non-finite TCP positions"
            )
            return None

        left_world = self._transform_point(self.left2world_T, left_pos)
        right_world = self._transform_point(self.right2world_T, right_pos)
        return 0.5 * (left_world + right_world) + self.robot_center_offset_xyz

    def _target_world(self) -> Optional[np.ndarray]:
        control_array = self._read_latest_control_array()
        if control_array is not None:
            planned_center = self._planned_dual_arm_center_world(control_array)
            if planned_center is not None:
                return planned_center
        return None

    def close(self) -> None:
        if self._control_shm is not None:
            try:
                self._control_shm.close()
            except Exception as exc:
                logger.warning(f"Failed to close YamHead planned control SHM: {exc}")
            finally:
                self._control_shm = None
        super().close()


class PlannedRobotCenterScaledMotionTracking(PlannedRobotCenterTracking):
    """Preview planned center with small scaled head translation before look-at rotation."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        repo_root: Path,
        left2world_json_path: PathLike,
        right2world_json_path: PathLike,
        yam2world_json_path: PathLike,
    ) -> None:
        super().__init__(
            config,
            repo_root=repo_root,
            left2world_json_path=left2world_json_path,
            right2world_json_path=right2world_json_path,
            yam2world_json_path=yam2world_json_path,
        )
        self.translation_scale = float(config.get("translation_scale", 0.12))
        self.max_translation_m = float(config.get("max_translation_m", 0.08))
        if self.translation_scale < 0.0:
            raise ValueError("translation_scale must be non-negative")
        if self.max_translation_m < 0.0:
            raise ValueError("max_translation_m must be non-negative")

    def _focus_origin_world(
        self,
        *,
        target_world: np.ndarray,
        q_target: np.ndarray,
        yam_kin: Any,
        robot_q_to_model_qpos: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        base_origin_world = super()._focus_origin_world(
            target_world=target_world,
            q_target=q_target,
            yam_kin=yam_kin,
            robot_q_to_model_qpos=robot_q_to_model_qpos,
        )
        delta = (
            np.asarray(target_world, dtype=np.float64).reshape(3) - base_origin_world
        )
        translation = self.translation_scale * delta
        distance = float(np.linalg.norm(translation))
        if self.max_translation_m > 0.0 and distance > self.max_translation_m:
            translation = translation * (self.max_translation_m / distance)
        return base_origin_world + translation


def build_yam_head_tracking_strategy(
    config: Optional[Dict[str, Any]],
    *,
    repo_root: Path,
    left2world_json_path: PathLike,
    right2world_json_path: PathLike,
    yam2world_json_path: PathLike,
) -> Optional[YamHeadTrackingStrategy]:
    if not config:
        return None

    strategy_type = str(config.get("type", "")).strip().lower()
    if not strategy_type and bool(config.get("track_robot_center", False)):
        strategy_type = "robot_center_decay"
    if strategy_type in {"", "disabled", "none", "null"}:
        return None
    if strategy_type == "robot_center_decay":
        return RobotCenterDecayTracking(
            config,
            repo_root=repo_root,
            left2world_json_path=left2world_json_path,
            right2world_json_path=right2world_json_path,
            yam2world_json_path=yam2world_json_path,
        )
    if strategy_type == "planned_robot_center":
        return PlannedRobotCenterTracking(
            config,
            repo_root=repo_root,
            left2world_json_path=left2world_json_path,
            right2world_json_path=right2world_json_path,
            yam2world_json_path=yam2world_json_path,
        )
    if strategy_type == "planned_robot_center_scaled_motion":
        return PlannedRobotCenterScaledMotionTracking(
            config,
            repo_root=repo_root,
            left2world_json_path=left2world_json_path,
            right2world_json_path=right2world_json_path,
            yam2world_json_path=yam2world_json_path,
        )
    raise ValueError(f"Unknown YamHead tracking strategy type: {strategy_type!r}")
