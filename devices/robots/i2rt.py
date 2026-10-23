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
from devices.robots.config import RobotControlConfig
from utils.shm_utils import (
    pack_device_header, pack_buffer_header, pack_frame_header,
    calculate_device_shm_size, DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    get_control_frame_offset,
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

class I2RTRobot(BaseRobot):
    """
    Inherits from BaseRobot and provides I2RT robot interface for SuperInference.
    """
    _TAKEOVER_ALIGNMENT_FLAG = 0.5
    _CONTROL_MODE_AUTONOMOUS = "follower_autonomous"
    _CONTROL_MODE_ALIGNMENT = "takeover_alignment"
    _CONTROL_MODE_TELEOP = "leader_to_follower"
    source_selector_owner_supported = True

    def _can_own_source_selector(self) -> bool:
        return self.is_leader_role

    @property
    def device_id(self) -> int:
        """Compatibility alias for in-memory callers; configs use device_class_id."""
        return int(getattr(self, "device_class_id", 0))

    @device_id.setter
    def device_id(self, value: int) -> None:
        self.device_class_id = int(value)

    def _is_dual_source_mode(self) -> bool:
        """Use the structured source contract, with a test-only legacy fallback."""
        try:
            return super()._is_dual_source_mode()
        except AttributeError:
            return bool(getattr(self, "dual_control_mode", True))

    def __init__(
        self,
        device_class_id=0,
        *,
        state_feedback: Mapping[str, Any],
        control: RobotControlConfig | Mapping[str, Any],
        signal_files: Optional[Dict[str, str]] = None,
        # I2RT-specific arguments
        gripper_type: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = "yam_teaching_handle",
        mode: Literal["follower", "leader", "leader_server", "follower_server", "visualizer_local", "visualizer_remote"] = "follower",
        server_host: str = "localhost",
        server_port: int = 11333,
        can_channel: str = "can0",
        bilateral_kp: float = 0.0,
        bilateral_kd: float = 0.0,
        home_joint_pos: Optional[Union[np.ndarray, List[float]]] = None,
        left_server_host: Optional[str] = None,
        right_server_host: Optional[str] = None,
        left_server_port: Optional[int] = None,
        right_server_port: Optional[int] = None,
        left_can_channel: Optional[str] = None,
        right_can_channel: Optional[str] = None,
        server_startup_wait_s: float = 3.0,
        server_launch_python: str = "python",
        minimum_gello_script: Optional[str] = None,
        disconnection_recovery: Optional[Dict[str, Any]] = None,
        takeover_alignment: Optional[Mapping[str, Any]] = None,
        random_init_pose: bool = False,
        random_init_pose_params: Optional[Mapping[str, Any]] = None,
        random_init_pose_sync: bool = False,
    ):
        self.mode = mode
        self.is_leader_role = "leader" in self.mode
        self.is_follower_role = not self.is_leader_role
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            control=control,
        )
        state_primitives = tuple(
            primitive.name for primitive in self.state_feedback_config.format.primitives
        )
        control_primitives = tuple(
            primitive.name for primitive in self.control_config.format.primitives
        )
        if state_primitives != ("j1-7", "j1-7"):
            raise ValueError(
                "I2RTRobot state_feedback.format.type must be [j1-7, j1-7]"
            )
        if self.control_config.type != "joints" or (
            control_primitives != ("j1-7", "j1-7")
        ):
            raise ValueError(
                "I2RTRobot joint control requires control.type=joints "
                "and control.format.type=[j1-7, j1-7]"
            )

        self.gripper_type = gripper_type
        self.server_host = server_host
        self.server_port = server_port
        self.can_channel = can_channel
        self.bilateral_kp = bilateral_kp
        self.bilateral_kd = bilateral_kd

        self.home_joint_pos = None if home_joint_pos is None else np.asarray(home_joint_pos, dtype=np.float64).flatten()
        self.is_leader_role = "leader" in self.mode
        self.is_follower_role = not self.is_leader_role
        if takeover_alignment is None:
            self.takeover_alignment_config: Dict[str, Any] = {}
        elif isinstance(takeover_alignment, Mapping):
            self.takeover_alignment_config = dict(takeover_alignment)
        else:
            raise ValueError("takeover_alignment must be a mapping when provided")
        takeover_alignment_enabled = self.takeover_alignment_config.get(
            "enabled", False
        )
        if not isinstance(takeover_alignment_enabled, bool):
            raise ValueError("takeover_alignment.enabled must be a boolean")
        self.takeover_alignment_enabled = takeover_alignment_enabled
        self.takeover_alignment_duration_s = float(
            self.takeover_alignment_config.get("duration_s", 0.2)
        )
        self.takeover_alignment_settle_timeout_s = float(
            self.takeover_alignment_config.get("settle_timeout_s", 0.15)
        )
        self.takeover_hold_ack_timeout_s = float(
            self.takeover_alignment_config.get("hold_ack_timeout_s", 1.0)
        )
        self.takeover_hold_feedback_tolerance_rad = float(
            self.takeover_alignment_config.get(
                "hold_feedback_tolerance_rad", 0.05
            )
        )
        self.takeover_hold_feedback_min_settle_s = float(
            self.takeover_alignment_config.get(
                "hold_feedback_min_settle_s", 0.05
            )
        )
        hold_feedback_stable_samples = self.takeover_alignment_config.get(
            "hold_feedback_stable_samples", 2
        )
        if (
            isinstance(hold_feedback_stable_samples, bool)
            or not isinstance(hold_feedback_stable_samples, int)
            or hold_feedback_stable_samples < 2
        ):
            raise ValueError(
                "takeover_alignment.hold_feedback_stable_samples must be an integer of at least 2"
            )
        self.takeover_hold_feedback_stable_samples = hold_feedback_stable_samples
        if (
            not np.isfinite(self.takeover_alignment_duration_s)
            or self.takeover_alignment_duration_s < 0.0
        ):
            raise ValueError("takeover_alignment.duration_s must be finite and non-negative")
        if (
            not np.isfinite(self.takeover_alignment_settle_timeout_s)
            or self.takeover_alignment_settle_timeout_s < 0.0
        ):
            raise ValueError(
                "takeover_alignment.settle_timeout_s must be finite and non-negative"
            )
        if (
            not np.isfinite(self.takeover_hold_ack_timeout_s)
            or self.takeover_hold_ack_timeout_s <= 0.0
        ):
            raise ValueError(
                "takeover_alignment.hold_ack_timeout_s must be finite and positive"
            )
        if (
            not np.isfinite(self.takeover_hold_feedback_tolerance_rad)
            or self.takeover_hold_feedback_tolerance_rad <= 0.0
        ):
            raise ValueError(
                "takeover_alignment.hold_feedback_tolerance_rad must be finite and positive"
            )
        if (
            not np.isfinite(self.takeover_hold_feedback_min_settle_s)
            or self.takeover_hold_feedback_min_settle_s < 0.0
        ):
            raise ValueError(
                "takeover_alignment.hold_feedback_min_settle_s must be finite and non-negative"
            )
        self._validate_takeover_alignment_mode(
            enabled=self.takeover_alignment_enabled,
            is_leader_role=self.is_leader_role,
            dual_control_mode=self._is_dual_source_mode(),
        )

        if not isinstance(random_init_pose, bool):
            raise ValueError("random_init_pose must be a boolean")
        if not isinstance(random_init_pose_sync, bool):
            raise ValueError("random_init_pose_sync must be a boolean")
        if random_init_pose_params is None:
            self.random_init_pose_params: Dict[str, Any] = {}
        elif isinstance(random_init_pose_params, Mapping):
            self.random_init_pose_params = dict(random_init_pose_params)
        else:
            raise ValueError("random_init_pose_params must be a mapping when provided")
        self.random_init_pose = random_init_pose
        self.random_init_pose_sync = random_init_pose_sync
        if self.random_init_pose and not self.is_follower_role:
            raise ValueError("random_init_pose may only be enabled on the follower robot")
        if self.random_init_pose and not self.random_init_pose_sync:
            raise ValueError(
                "random_init_pose requires random_init_pose_sync so the leader follows it"
            )
        self._validate_random_init_pose_config()
        self._random_init_pose_rng = np.random.default_rng(
            self.random_init_pose_params.get("seed")
        )
        self._i2rt_kinematics = None
        self._i2rt_kinematics_backend: Optional[str] = None
        self.left_server_host = left_server_host or server_host
        self.right_server_host = right_server_host or server_host
        self.left_server_port = left_server_port if left_server_port is not None else server_port
        self.right_server_port = right_server_port if right_server_port is not None else server_port + 1
        self.left_can_channel = left_can_channel or can_channel
        self.right_can_channel = right_can_channel or can_channel
        self.server_startup_wait_s = server_startup_wait_s
        self.server_launch_python = server_launch_python
        if minimum_gello_script is None:
            self.minimum_gello_script = str(
                REPO_ROOT / "third_party" / "i2rt" / "scripts" / "minimum_gello.py"
            )
        else:
            self.minimum_gello_script = minimum_gello_script

        self._left_client = None
        self._right_client = None
        self._server_processes: List[subprocess.Popen] = []
        self._last_button_left = 0.0
        self._last_button_right = 0.0
        self._leader_flag_write_index = 0
        self._leader_flag_frame_count = 0
        self._leader_flag_buffer_size = max(100, int(self.control_buffer_size))
        self._leader_flag_owner = False
        self._last_button_pressed = False
        self._button_down_latched = False
        self._button_release_samples = 0
        self._button_release_debounce_samples = 3
        self._control_mode = self._CONTROL_MODE_AUTONOMOUS
        self._takeover_hold_joint_pos: Optional[np.ndarray] = None
        self._last_takeover_hold_ack_request_id: Optional[str] = None
        self._takeover_hold_command_issued_at_monotonic: Optional[float] = None
        self._takeover_hold_feedback_samples = 0
        self._takeover_hold_last_feedback_monotonic: Optional[float] = None
        self._takeover_hold_feedback_request_id: Optional[str] = None
        self._takeover_hold_faulted = False
        self._latest_joint_state: Optional[np.ndarray] = None
        self._last_control_flag = 0.0 if self._is_dual_source_mode() else 1.0
        self._dagger_mode_transition_lock = threading.Lock()
        self._load_default_bilateral_gains()

        # Disconnection recovery configuration
        self._disconnection_recovery_config = disconnection_recovery or {}
        logger.info(f"Disconnection recovery config: {self._disconnection_recovery_config}")
        self._disconnection_recovery_enabled = self._disconnection_recovery_config.get('enabled', False)
        # self._pose_diff_threshold = self._disconnection_recovery_config.get('pose_diff_threshold', 0.3)
        self._stationary_time_threshold = self._disconnection_recovery_config.get('stationary_time_threshold', 2.0)
        self._trusted_state_check_interval = self._disconnection_recovery_config.get('trusted_state_check_interval', 0.01)
        self._stuck_call_check_interval = self._disconnection_recovery_config.get('stuck_call_check_interval', 0.5)
        self._stuck_call_timeout_s = float(self._disconnection_recovery_config.get("stuck_call_timeout_s", 3.0))
        # self._motion_threshold = float(self._disconnection_recovery_config.get("motion_threshold", 0.01))
        self._home_command_active: bool = False
        self._home_command_grace_period_ns: int = int(
            float(self._disconnection_recovery_config.get("home_command_grace_period_s", 0.5)) * 1e9
        )

        # Disconnection monitoring state
        self._last_self_pose: Optional[np.ndarray] = None
        self._self_stationary_start_time: Optional[float] = None
        self._trusted_state_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._recovery_active: bool = False
        self._monitor_state_lock = threading.Lock()
        self._monitored_call_sources = (
            "get_robot_state", "execute_control_command",

            "get_info_left_leader_get_robot_state", "get_info_right_leader_get_robot_state", "get_info_left_follower_get_robot_state",
            "get_info_right_follower_get_robot_state", "get_info_left_leader_go_home", "get_info_right_leader_go_home", "get_info_left_follower_go_home",
            "get_info_right_follower_go_home", "get_info_left_leader_go_rest", "get_info_right_leader_go_rest", "get_info_left_follower_go_rest",
            "get_info_right_follower_go_rest", "get_info_left_leader_trusted_state", "get_info_right_leader_trusted_state", "get_info_left_follower_trusted_state", "get_info_right_follower_trusted_state",

            "command_joint_pos_left_get_robot_state_leader", "command_joint_pos_right_get_robot_state_leader",
            "command_joint_pos_left_smooth_joint_move", "command_joint_pos_right_smooth_joint_move",

            "command_joint_pos_left_execute_control_follower", "command_joint_pos_right_execute_control_follower",
            "command_joint_pos_left_execute_control_leader", "command_joint_pos_right_execute_control_leader",

            "command_joint_pos_left_check_and_execute_teleop_start_leader", "command_joint_pos_right_check_and_execute_teleop_start_leader",
            "command_joint_pos_left_check_and_execute_teleop_start_follower", "command_joint_pos_right_check_and_execute_teleop_start_follower",

            "command_joint_pos_left_start_server_leader", "command_joint_pos_right_start_server_leader",
            "command_joint_pos_left_start_server_follower", "command_joint_pos_right_start_server_follower",

        )
        self._monitored_call_log_interval_s = float(
            self._disconnection_recovery_config.get("monitored_call_log_interval_s", 5.0)
        )
        self._monitored_call_states: Dict[str, Dict[str, Any]] = {
            source: self._new_monitored_call_state()
            for source in self._monitored_call_sources
        }
        self._last_motion_monotonic: float = time.monotonic()

        self._monitor_disconnection_evt = threading.Event()
        self._main_control_loop_thread: Optional[threading.Thread] = None
        self._main_control_loop_generation: int = 0
        self._main_control_loop_lock = threading.Lock()

        # Episode signal tracking for button-controlled data collection
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._episode_signal_pattern = signal_files.get('episode_signal_pattern')
        self._robot_home_command_pattern = signal_files.get('robot_home_command_pattern')
        self._robot_teleop_start_command_pattern = signal_files.get('robot_teleop_start_command_pattern')
        self._robot_teleop_discard_command_pattern = signal_files.get('robot_teleop_discard_command_pattern')
        self._policy_runner_reset_command = signal_files.get('policy_runner_reset_command')
        self._reset_signal = signal_files.get('reset_signal')
        self._robot_reconnect_signal = signal_files.get('robot_reconnect')
        self._robot_halt_command_pattern = signal_files.get('robot_halt_command_pattern')
        self._robot_halt_reply_pattern = signal_files.get('robot_halt_reply_pattern')
        self._takeover_hold_request_path = signal_files.get(
            'i2rt_takeover_hold_request'
        )
        self._takeover_hold_ack_path = signal_files.get('i2rt_takeover_hold_ack')
        self._random_home_request_path = signal_files.get('i2rt_random_home_request')
        self._random_home_target_path = signal_files.get('i2rt_random_home_target')
        self._random_home_result_pattern = signal_files.get(
            'i2rt_random_home_result_pattern'
        )
        if self.random_init_pose_sync and (
            not self._random_home_request_path
            or not self._random_home_target_path
            or not self._random_home_result_pattern
        ):
            raise ValueError(
                "I2RT randomized home sync requires runtime.signal_files."
                "i2rt_random_home_request, i2rt_random_home_target, and "
                "i2rt_random_home_result_pattern"
            )
        if self.takeover_alignment_enabled and (
            not self._takeover_hold_request_path or not self._takeover_hold_ack_path
        ):
            raise ValueError(
                "I2RT takeover alignment requires runtime.signal_files."
                "i2rt_takeover_hold_request and i2rt_takeover_hold_ack"
            )
        self._is_recording_episode = False

        # self._last_get_info_state_left = None
        # self._last_get_info_state_right = None
        self._last_trusted_get_info_state_left = None
        self._last_trusted_get_info_state_right = None
        self._trusted_state_lock = threading.Lock()
        self.state_outlier_threshold = self._disconnection_recovery_config.get('state_outlier_threshold', 0.2)
        self._robot_halt_command_active = False

        self._ensure_rpc_servers_running()
        self._connect_rpc_clients()
        logger.info(
            f"Initialized I2RTRobot_{self.device_class_id} role="
            f"{'leader' if self.is_leader_role else 'follower'} "
            f"left={self.left_server_host}:{self.left_server_port} "
            f"right={self.right_server_host}:{self.right_server_port}"
        )

    def _connect_rpc_clients(self) -> None:
        """Connect to left/right I2RT RPC servers."""
        try:
            import importlib

            portal_module = importlib.import_module("portal")
            self._left_client = portal_module.Client(f"{self.left_server_host}:{self.left_server_port}")
            self._right_client = portal_module.Client(f"{self.right_server_host}:{self.right_server_port}")
            # minimum_gello leader/follower server exposes get_info()
            self._left_client.get_info().result(timeout=2.0)
            self._right_client.get_info().result(timeout=2.0)
        except Exception as e:
            logger.error(f"Failed to connect I2RT RPC clients: {e}")
            self._left_client = None
            self._right_client = None

    def _load_default_bilateral_gains(self) -> None:
        """Load default KP/KD from bilateral_control.py (with local fallback)."""
        leader_default_kp = np.array([250.0, 400.0, 400.0, 100.0, 25.0, 25.0])
        leader_default_kd = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
        follower_default_kp = np.array(
            [250.0, 400.0, 400.0, 100.0, 25.0, 25.0, 25.0]
        )
        follower_default_kd = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 1.5])
        try:
            import importlib.util

            bilateral_path = (
                REPO_ROOT / "third_party" / "i2rt" / "examples" / "bilateral_control.py"
            )
            spec = importlib.util.spec_from_file_location("i2rt_bilateral_control_defaults", bilateral_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                leader_default_kp = np.asarray(module.LEADER_DEFAULT_KP, dtype=np.float64)
                leader_default_kd = np.asarray(module.LEADER_DEFAULT_KD, dtype=np.float64)
                follower_default_kp = np.asarray(module.FOLLOWER_DEFAULT_KP, dtype=np.float64)
                follower_default_kd = np.asarray(module.FOLLOWER_DEFAULT_KD, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Failed to import bilateral defaults, using fallback: {e}")

        self._leader_default_kp = leader_default_kp
        self._leader_default_kd = leader_default_kd
        self._leader_zero_kp = np.zeros_like(leader_default_kp)
        self._leader_zero_kd = np.zeros_like(leader_default_kd)
        self._follower_default_kp = follower_default_kp
        self._follower_default_kd = follower_default_kd
        self._follower_zero_kp = np.zeros_like(follower_default_kp)
        self._follower_zero_kd = np.zeros_like(follower_default_kd)

    def _apply_bilateral_gains_for_mode(self) -> None:
        """Apply KP/KD according to current toggled control mode."""
        if self._left_client is None or self._right_client is None:
            return
        try:
            if self.is_leader_role:
                if self._control_mode == self._CONTROL_MODE_TELEOP:
                    kp = self._leader_zero_kp
                    kd = self._leader_zero_kd
                else:
                    kp = self._leader_default_kp * self.bilateral_kp
                    kd = self._leader_default_kd * self.bilateral_kd
            else:
                kp = self._follower_default_kp * self.bilateral_kp
                kd = self._follower_default_kd * self.bilateral_kd

            self._left_client.update_kp_kd(kp, kd)
            self._right_client.update_kp_kd(kp, kd)


        except Exception as e:
            logger.warning(f"Failed to update KP/KD gains in mode {self._control_mode}: {e}")

    @classmethod
    def _control_mode_from_flag(cls, control_flag: float) -> str:
        """Map the I2RT takeover protocol onto a follower control mode."""
        if abs(float(control_flag) - cls._TAKEOVER_ALIGNMENT_FLAG) <= 1e-6:
            return cls._CONTROL_MODE_ALIGNMENT
        if float(control_flag) > cls._TAKEOVER_ALIGNMENT_FLAG:
            return cls._CONTROL_MODE_TELEOP
        return cls._CONTROL_MODE_AUTONOMOUS

    @classmethod
    def _enters_autonomous_control_mode(
        cls,
        previous_control_flag: float,
        current_control_flag: float,
    ) -> bool:
        """Return whether a flag change must discard stale policy actions."""
        return (
            cls._control_mode_from_flag(previous_control_flag)
            != cls._CONTROL_MODE_AUTONOMOUS
            and cls._control_mode_from_flag(current_control_flag)
            == cls._CONTROL_MODE_AUTONOMOUS
        )

    @staticmethod
    def _validate_takeover_alignment_mode(
        *,
        enabled: bool,
        is_leader_role: bool,
        dual_control_mode: bool,
    ) -> None:
        """Reject alignment settings that cannot enter the three-state protocol."""
        if not enabled:
            return
        if not is_leader_role:
            raise ValueError("takeover_alignment may only be enabled on the leader robot")
        if not dual_control_mode:
            raise ValueError(
                "takeover_alignment.enabled requires dual-source control selection"
            )

    @staticmethod
    def _valid_takeover_joint_state(state: Optional[np.ndarray]) -> bool:
        if state is None:
            return False
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        return state_array.size >= 14 and bool(np.all(np.isfinite(state_array[:14])))

    def _takeover_peer_device_id(self) -> int:
        """Return the fixed peer ID used by the bilateral I2RT pair contract."""
        device_id = int(self.device_class_id)
        if device_id not in {0, 1}:
            raise ValueError(
                "I2RT takeover alignment requires bilateral device IDs 0 and 1"
            )
        return 1 - device_id

    def _publish_takeover_hold_request(self) -> Optional[str]:
        """Create the leader-owned request that makes a `.5` hold fresh."""
        from utils.i2rt_takeover import (
            TakeoverHoldSyncError,
            publish_takeover_hold_request,
        )

        try:
            request_path = getattr(self, "_takeover_hold_request_path", None)
            if not request_path:
                raise TakeoverHoldSyncError("takeover hold request path is unavailable")
            return publish_takeover_hold_request(
                request_path,
                source_device_id=int(self.device_class_id),
                target_device_id=self._takeover_peer_device_id(),
            )
        except (OSError, TakeoverHoldSyncError, ValueError) as exc:
            logger.error(f"Cannot publish I2RT takeover hold request: {exc}")
            return None

    def _read_follower_joint_state_for_takeover_hold(self) -> Optional[np.ndarray]:
        """Read a live follower feedback sample before commanding a hold."""
        try:
            left_pos = self._follower_get_info(
                self._left_client,
                "get_info_left_follower_takeover_alignment_hold",
            )
            right_pos = self._follower_get_info(
                self._right_client,
                "get_info_right_follower_takeover_alignment_hold",
            )
            state = np.concatenate([left_pos, right_pos], axis=0)
            if not self._valid_takeover_joint_state(state):
                return None
            return self._fit_state_shape(state)
        except Exception as exc:
            logger.warning(
                f"Failed to read live I2RT follower hold feedback: {exc}"
            )
            return None

    def _begin_follower_takeover_hold(self) -> bool:
        """Command a live follower pose and reset its feedback acknowledgement."""
        hold_state = self._read_follower_joint_state_for_takeover_hold()
        if not self._valid_takeover_joint_state(hold_state):
            self._takeover_hold_joint_pos = None
            self._takeover_hold_command_issued_at_monotonic = None
            self._takeover_hold_feedback_samples = 0
            self._takeover_hold_last_feedback_monotonic = None
            self._takeover_hold_feedback_request_id = None
            self._takeover_hold_faulted = True
            logger.error(
                "Cannot enter I2RT takeover alignment without live follower feedback"
            )
            return False

        hold_target = np.asarray(
            hold_state, dtype=np.float64
        ).reshape(-1)[:14].copy()
        self._takeover_hold_joint_pos = None
        self._takeover_hold_feedback_samples = 0
        self._takeover_hold_last_feedback_monotonic = None
        self._takeover_hold_feedback_request_id = None
        self._takeover_hold_faulted = False
        try:
            self._command_joint_pos(
                self._left_client,
                hold_target[:7],
                "command_joint_pos_left_takeover_alignment_hold",
            )
            self._command_joint_pos(
                self._right_client,
                hold_target[7:14],
                "command_joint_pos_right_takeover_alignment_hold",
            )
        except Exception as exc:
            self._takeover_hold_joint_pos = None
            self._takeover_hold_command_issued_at_monotonic = None
            self._takeover_hold_feedback_samples = 0
            self._takeover_hold_last_feedback_monotonic = None
            self._takeover_hold_feedback_request_id = None
            self._takeover_hold_faulted = True
            logger.error(f"Failed to command I2RT follower takeover hold: {exc}")
            return False
        self._takeover_hold_joint_pos = hold_target
        self._takeover_hold_command_issued_at_monotonic = time.monotonic()
        logger.info("I2RT follower holding pose during leader takeover alignment")
        return True

    def _acknowledge_follower_takeover_hold(
        self,
        measured_joint_pos: Optional[np.ndarray],
        *,
        measured_at_monotonic: float,
        measured_at_ns: int,
    ) -> None:
        """ACK only fresh, stable feedback that proves the follower is holding."""
        from utils.i2rt_takeover import (
            TakeoverHoldSyncError,
            publish_takeover_hold_ack,
            read_takeover_hold_request,
        )

        if not self.is_follower_role or not self._valid_takeover_joint_state(
            getattr(self, "_takeover_hold_joint_pos", None)
        ) or not self._valid_takeover_joint_state(measured_joint_pos):
            return
        command_issued_at = getattr(
            self, "_takeover_hold_command_issued_at_monotonic", None
        )
        if (
            not isinstance(command_issued_at, (float, int))
            or not np.isfinite(command_issued_at)
            or isinstance(measured_at_monotonic, bool)
            or not isinstance(measured_at_monotonic, (float, int))
            or not np.isfinite(measured_at_monotonic)
            or measured_at_monotonic <= command_issued_at
        ):
            return
        try:
            request_path = getattr(self, "_takeover_hold_request_path", None)
            if not request_path:
                raise TakeoverHoldSyncError("takeover hold request path is unavailable")
            request_id = read_takeover_hold_request(
                request_path,
                expected_source_device_id=self._takeover_peer_device_id(),
                expected_target_device_id=int(self.device_class_id),
            )
        except (OSError, TakeoverHoldSyncError, ValueError) as exc:
            logger.warning(f"Cannot acknowledge I2RT takeover hold: {exc}")
            return
        if request_id is None or request_id == getattr(
            self, "_last_takeover_hold_ack_request_id", None
        ):
            return
        if request_id != getattr(self, "_takeover_hold_feedback_request_id", None):
            self._takeover_hold_feedback_request_id = request_id
            self._takeover_hold_feedback_samples = 0
            self._takeover_hold_last_feedback_monotonic = None
            # The request could have appeared after the RPC feedback response.
            # Start counting only with the next feedback sample.
            return
        previous_feedback_at = getattr(
            self, "_takeover_hold_last_feedback_monotonic", None
        )
        if (
            isinstance(previous_feedback_at, (float, int))
            and measured_at_monotonic <= previous_feedback_at
        ):
            return
        self._takeover_hold_last_feedback_monotonic = measured_at_monotonic
        if (
            measured_at_monotonic - command_issued_at
            < self.takeover_hold_feedback_min_settle_s
        ):
            return
        target = np.asarray(self._takeover_hold_joint_pos, dtype=np.float64).reshape(-1)
        measured = np.asarray(measured_joint_pos, dtype=np.float64).reshape(-1)
        if np.any(
            np.abs(measured[:14] - target[:14])
            > self.takeover_hold_feedback_tolerance_rad
        ):
            self._takeover_hold_feedback_samples = 0
            return
        self._takeover_hold_feedback_samples += 1
        if (
            self._takeover_hold_feedback_samples
            < self.takeover_hold_feedback_stable_samples
        ):
            return
        try:
            acknowledgement_path = getattr(self, "_takeover_hold_ack_path", None)
            if not acknowledgement_path:
                raise TakeoverHoldSyncError(
                    "takeover hold acknowledgement path is unavailable"
                )
            publish_takeover_hold_ack(
                acknowledgement_path,
                request_id=request_id,
                source_device_id=int(self.device_class_id),
                target_device_id=self._takeover_peer_device_id(),
                hold_joint_pos=measured[:14],
                measured_at_ns=measured_at_ns,
            )
            self._last_takeover_hold_ack_request_id = request_id
        except (OSError, TakeoverHoldSyncError, ValueError) as exc:
            logger.warning(f"Cannot acknowledge I2RT takeover hold: {exc}")

    def _observe_follower_takeover_hold_feedback(
        self,
        measured_joint_pos: Optional[np.ndarray],
        *,
        measured_at_monotonic: float,
        measured_at_ns: int,
    ) -> None:
        """Use only a post-command RPC feedback sample for the hold ACK."""
        if self._control_mode != self._CONTROL_MODE_ALIGNMENT:
            return
        self._acknowledge_follower_takeover_hold(
            measured_joint_pos,
            measured_at_monotonic=measured_at_monotonic,
            measured_at_ns=measured_at_ns,
        )

    def _wait_for_follower_takeover_hold(
        self, *, request_id: str
    ) -> Optional[np.ndarray]:
        """Wait for the follower to hold and attest its exact pose before moving."""
        from utils.i2rt_takeover import TakeoverHoldSyncError, read_takeover_hold_ack

        acknowledgement_path = getattr(self, "_takeover_hold_ack_path", None)
        if not acknowledgement_path:
            logger.error("I2RT takeover hold acknowledgement path is unavailable")
            return None
        timeout_s = float(getattr(self, "takeover_hold_ack_timeout_s", 1.0))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() <= deadline:
            try:
                hold_target = read_takeover_hold_ack(
                    acknowledgement_path,
                    request_id=request_id,
                    expected_source_device_id=self._takeover_peer_device_id(),
                    expected_target_device_id=int(self.device_class_id),
                )
            except (OSError, TakeoverHoldSyncError, ValueError) as exc:
                logger.error(f"I2RT takeover hold acknowledgement is invalid: {exc}")
                return None
            if hold_target is not None:
                return hold_target
            time.sleep(min(0.05, 1.0 / max(float(self.command_fps), 1.0)))
        logger.error("Timed out waiting for follower I2RT takeover hold acknowledgement")
        return None

    def _set_follower_control_mode_from_flag(self, control_flag: float) -> None:
        """Enter the requested follower mode, holding safely during alignment."""
        desired_mode = self._control_mode_from_flag(control_flag)
        if desired_mode == self._control_mode:
            if (
                desired_mode == self._CONTROL_MODE_ALIGNMENT
                and not self._valid_takeover_joint_state(
                    self._takeover_hold_joint_pos
                )
            ):
                self._begin_follower_takeover_hold()
            return

        if desired_mode == self._CONTROL_MODE_ALIGNMENT:
            # Block policy execution before either hold RPC. A partial arm
            # command failure must stay fail-closed rather than leave the
            # follower autonomous while the leader waits for its ACK.
            self._control_mode = self._CONTROL_MODE_ALIGNMENT
            self._takeover_hold_joint_pos = None
            self._takeover_hold_command_issued_at_monotonic = None
            self._takeover_hold_feedback_samples = 0
            self._takeover_hold_last_feedback_monotonic = None
            self._takeover_hold_feedback_request_id = None
            self._takeover_hold_faulted = False
            self._apply_bilateral_gains_for_mode()
            self._begin_follower_takeover_hold()
            logger.info(f"I2RT follower control mode set to: {self._control_mode}")
            return
        else:
            self._takeover_hold_joint_pos = None
            self._takeover_hold_command_issued_at_monotonic = None
            self._takeover_hold_feedback_samples = 0
            self._takeover_hold_last_feedback_monotonic = None
            self._takeover_hold_feedback_request_id = None
            self._takeover_hold_faulted = False

        self._control_mode = desired_mode
        self._apply_bilateral_gains_for_mode()
        logger.info(f"I2RT follower control mode set to: {self._control_mode}")

    def _read_leader_joint_state_for_takeover(self) -> Optional[np.ndarray]:
        """Read leader joints without processing another button transition."""
        try:
            left_pos, _ = self._leader_get_info(
                self._left_client, "get_info_left_leader_takeover_alignment"
            )
            right_pos, _ = self._leader_get_info(
                self._right_client, "get_info_right_leader_takeover_alignment"
            )
            state = np.concatenate([left_pos, right_pos], axis=0)
            if not self._valid_takeover_joint_state(state):
                return None
            return self._fit_state_shape(state)
        except Exception as exc:
            logger.warning(f"Failed to read leader state during takeover alignment: {exc}")
            return None

    def _align_leader_for_takeover(
        self, current_leader_state: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Hold the follower, align the leader, then permit teleoperation."""
        if not self._valid_takeover_joint_state(current_leader_state):
            logger.error("Cannot align I2RT leader: current leader state is invalid")
            return None

        request_id = self._publish_takeover_hold_request()
        if request_id is None:
            return None
        self._write_leader_control_flag(
            self._TAKEOVER_ALIGNMENT_FLAG, time.time_ns()
        )
        follower_target = self._wait_for_follower_takeover_hold(request_id=request_id)
        if not self._valid_takeover_joint_state(follower_target):
            logger.error("Cannot align I2RT leader: follower did not acknowledge hold")
            self._write_leader_control_flag(0.0, time.time_ns())
            return None

        control_period_s = 1.0 / max(float(self.command_fps), 1.0)

        current = np.asarray(current_leader_state, dtype=np.float64).reshape(-1)
        target = np.asarray(follower_target, dtype=np.float64).reshape(-1)
        left_current, right_current = current[:6], current[7:13]
        left_target, right_target = target[:6], target[7:13]

        interpolation_steps = max(
            1,
            int(
                np.ceil(
                    self.takeover_alignment_duration_s
                    * max(float(self.command_fps), 1.0)
                )
            ),
        )
        interpolation_start = time.monotonic()
        for step in range(1, interpolation_steps + 1):
            alpha = step / interpolation_steps
            self._command_joint_pos(
                self._left_client,
                left_current * (1.0 - alpha) + left_target * alpha,
                "command_joint_pos_left_takeover_alignment",
            )
            self._command_joint_pos(
                self._right_client,
                right_current * (1.0 - alpha) + right_target * alpha,
                "command_joint_pos_right_takeover_alignment",
            )
            self._write_leader_control_flag(
                self._TAKEOVER_ALIGNMENT_FLAG, time.time_ns()
            )
            deadline = interpolation_start + self.takeover_alignment_duration_s * (
                step / interpolation_steps
            )
            remaining_s = deadline - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)

        settle_deadline = time.monotonic() + self.takeover_alignment_settle_timeout_s
        aligned_state: Optional[np.ndarray] = None
        while True:
            measured_state = self._read_leader_joint_state_for_takeover()
            if measured_state is not None:
                aligned_state = measured_state
            if time.monotonic() >= settle_deadline:
                break
            self._command_joint_pos(
                self._left_client,
                left_target,
                "command_joint_pos_left_takeover_alignment_settle",
            )
            self._command_joint_pos(
                self._right_client,
                right_target,
                "command_joint_pos_right_takeover_alignment_settle",
            )
            self._write_leader_control_flag(
                self._TAKEOVER_ALIGNMENT_FLAG, time.time_ns()
            )
            time.sleep(control_period_s)

        if aligned_state is None:
            logger.error("Leader takeover alignment produced no valid measured state")
            self._write_leader_control_flag(0.0, time.time_ns())
            return None

        if getattr(self, "shared_memory", None) is not None:
            self._write_array_to_shm_with_timestamp(aligned_state, time.time_ns())
        return aligned_state

    def _handle_mode_transition(
        self,
        button_pressed: bool,
        *,
        current_leader_state: Optional[np.ndarray] = None,
        button_down: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """Handle a leader button edge and return an aligned state when used."""
        aligned_state = None
        physical_down = bool(button_pressed if button_down is None else button_down)
        if button_pressed and not self._button_down_latched:
            self._button_down_latched = True
            self._button_release_samples = 0
            if not self._is_dual_source_mode():
                # Data collection mode: toggle episode recording on button press
                self._toggle_episode_recording()
            else:
                entering_teleop = self._control_mode == self._CONTROL_MODE_AUTONOMOUS
                if (
                    entering_teleop
                    and self.is_leader_role
                    and self.takeover_alignment_enabled
                ):
                    aligned_state = self._align_leader_for_takeover(current_leader_state)
                    if aligned_state is not None:
                        self._toggle_control_mode()
                else:
                    self._toggle_control_mode()
        elif not physical_down:
            self._button_release_samples += 1
            if self._button_release_samples >= self._button_release_debounce_samples:
                self._button_down_latched = False
                self._button_release_samples = 0

        self._last_button_pressed = self._button_down_latched
        return aligned_state

    def _toggle_episode_recording(self) -> None:
        """Toggle episode recording state and write appropriate signal file."""
        if not self._is_recording_episode:
            # Start episode
            self._is_recording_episode = True
            self._write_episode_signal("start")
            logger.info("Episode started via button press")
        else:
            # Stop episode
            self._is_recording_episode = False
            self._write_episode_signal("end")
            logger.info("Episode stopped via button press")

    def _write_episode_signal(self, signal_type: str) -> None:
        """Write episode signal file for data saver to detect."""
        signal_path = self._episode_signal_pattern.format(signal=signal_type)
        try:
            with open(signal_path, 'w') as f:
                f.write(f"{time.time():.6f}")
            logger.info(f"Wrote episode {signal_type} signal to {signal_path}")
        except Exception as e:
            logger.error(f"Failed to write episode signal {signal_type}: {e}")

    def _toggle_control_mode(self) -> None:
        """Toggle between leader_to_follower and follower_autonomous modes."""
        self._control_mode = (
            self._CONTROL_MODE_AUTONOMOUS
            if self._control_mode == self._CONTROL_MODE_TELEOP
            else self._CONTROL_MODE_TELEOP
        )
        self._apply_bilateral_gains_for_mode()
        logger.info(f"I2RT control mode toggled to: {self._control_mode}")

    def _rpc_server_mode(self) -> str:
        return "leader_server" if self.is_leader_role else "follower_server"

    def _rpc_endpoints_ready(self) -> bool:
        """Check whether both RPC endpoints are reachable and serving."""
        try:
            import importlib

            portal_module = importlib.import_module("portal")
            left_client = portal_module.Client(f"{self.left_server_host}:{self.left_server_port}")
            right_client = portal_module.Client(f"{self.right_server_host}:{self.right_server_port}")
            left_client.get_info().result(timeout=1.0)
            right_client.get_info().result(timeout=1.0)
            return True
        except Exception:
            return False

    def _launch_rpc_server_process(self, can_channel: str, server_port: int) -> Optional[subprocess.Popen]:
        """Launch a minimum_gello RPC server process for one arm.

        Uses start_new_session=True so the server process does not receive
        SIGINT from the terminal when the user presses Ctrl+C.  This ensures
        the servers stay alive long enough for _go_rest() to send the final
        joint commands during shutdown.
        """
        cmd = [
            self.server_launch_python,
            self.minimum_gello_script,
            "--can_channel", can_channel,
            "--gripper", self.gripper_type,
            "--mode", self._rpc_server_mode(),
            "--server_port", str(server_port),
        ]
        logger.info(f"Launching I2RT RPC server: {' '.join(cmd)}")
        try:
            return subprocess.Popen(cmd, start_new_session=True)
        except Exception as e:
            logger.error(f"Failed to launch minimum_gello process ({can_channel}, {server_port}): {e}")
            return None

    def _ensure_rpc_servers_running(self) -> None:
        """
        Ensure minimum_gello server processes are running before client init.
        If endpoints are already up, skip launching.
        """
        if len(self._server_processes) == 2:
            return

        left_proc = self._launch_rpc_server_process(self.left_can_channel, self.left_server_port)
        right_proc = self._launch_rpc_server_process(self.right_can_channel, self.right_server_port)
        self._server_processes = [proc for proc in [left_proc, right_proc] if proc is not None]
        if len(self._server_processes) != 2:
            logger.error("Failed to launch both minimum_gello RPC server processes")
            return

        logger.info(f"Waiting {self.server_startup_wait_s:.1f}s for minimum_gello RPC servers")
        time.sleep(self.server_startup_wait_s)
        for _ in range(10):
            if self._rpc_endpoints_ready():
                logger.info("minimum_gello RPC servers are ready")
                return
            time.sleep(0.2)
        logger.error("minimum_gello RPC servers are not reachable after launch")

    def _fit_state_shape(self, state_array: np.ndarray) -> np.ndarray:
        """Pad/crop state to configured shape for robust SHM writes."""
        expected_size = int(np.prod(self.data_shape))
        flattened = state_array.astype(get_dtype(self.data_dtype), copy=False).reshape(-1)
        if flattened.size == expected_size:
            return flattened.reshape(self.data_shape)
        shaped = np.zeros(expected_size, dtype=get_dtype(self.data_dtype))
        copy_size = min(expected_size, flattened.size)
        shaped[:copy_size] = flattened[:copy_size]
        return shaped.reshape(self.data_shape)

    def _leader_get_info(self, client, source: str) -> tuple[np.ndarray, np.ndarray]:
        """Read leader info from RPC and normalize return format."""
        with self._track_inflight_call(source):
            result = client.get_info().result()

        if isinstance(result, (list, tuple)) and len(result) == 2:
            joint_pos = np.asarray(result[0], dtype=np.float64).reshape(-1)
            button_state = np.asarray(result[1], dtype=np.float64).reshape(-1)
            return joint_pos, button_state
        joint_pos = np.asarray(result, dtype=np.float64).reshape(-1)
        return joint_pos, np.zeros(1, dtype=np.float64)

    def _command_joint_pos(self, client, joint_pos: np.ndarray, source: str) -> None:
        """Send joint command while exposing blocking time to the stuck-call watchdog."""
        with self._track_inflight_call(source):
            if "smooth_joint_move" in source and self.is_follower_role:
                pass
            client.command_joint_pos(joint_pos)

    def _follower_get_info(self, client, source: str) -> np.ndarray:
        """Read follower joint+gripper state from RPC."""
        with self._track_inflight_call(source):
            result = client.get_info().result()

        return np.asarray(result, dtype=np.float64).reshape(-1)

    def _smooth_joint_move(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        left_current: np.ndarray,
        right_current: np.ndarray,
        steps: int = 100,
        dt: float = 0.03,
    ) -> None:
        """Linearly interpolate both arms from current to target joint positions."""
        left_target = np.asarray(left_target, dtype=np.float64).reshape(-1)
        right_target = np.asarray(right_target, dtype=np.float64).reshape(-1)
        left_current = np.asarray(left_current, dtype=np.float64).reshape(-1)
        right_current = np.asarray(right_current, dtype=np.float64).reshape(-1)
        if steps <= 0:
            self._command_joint_pos(self._left_client, left_target, "command_joint_pos_left_smooth_joint_move")
            self._command_joint_pos(self._right_client, right_target, "command_joint_pos_right_smooth_joint_move")
            return
        for step in range(1, steps + 1):
            alpha = step / steps
            left_interp = left_current * (1.0 - alpha) + left_target * alpha
            right_interp = right_current * (1.0 - alpha) + right_target * alpha
            self._command_joint_pos(self._left_client, left_interp, "command_joint_pos_left_smooth_joint_move")
            self._command_joint_pos(self._right_client, right_interp, "command_joint_pos_right_smooth_joint_move")
            time.sleep(dt)

    def _create_leader_control_flag_shm(self) -> None:
        """Create a leader-owned control flag SHM for shared-control takeover."""
        if not self.is_leader_role or not self._is_dual_source_mode():
            return
        if not self._source_selector_owned_by_self:
            return
        if self._control_flag_shm is not None:
            return
        if self._control_flag_shm_name is None:
            logger.warning("Dual-source leader has no selector endpoint")
            return
        try:
            control_flag_shape = (1,)
            selector_dtype = "float64"
            numpy_dtype = get_dtype(selector_dtype)
            bytes_per_element = np.dtype(numpy_dtype).itemsize
            data_size = np.prod(control_flag_shape) * bytes_per_element
            frame_size = FRAME_HEADER_SIZE + data_size
            total_size = calculate_device_shm_size(self._leader_flag_buffer_size, data_size)

            try:
                existing = shm.SharedMemory(name=self._control_flag_shm_name, create=False)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"Failed to remove old leader control flag SHM: {e}")

            self._control_flag_shm = shm.SharedMemory(
                name=self._control_flag_shm_name,
                create=True,
                size=total_size,
            )
            self._leader_flag_owner = True
            self._leader_flag_write_index = 0
            self._leader_flag_frame_count = 0

            device_header = pack_device_header(
                self.device_name,
                self.device_class_id,
                self.fps,
                selector_dtype,
                control_flag_shape,
                frame_size,
                self.hardware_latency_ms,
            )
            self._control_flag_shm.buf[:DEVICE_HEADER_SIZE] = device_header
            buffer_header = pack_buffer_header(self._leader_flag_buffer_size, 0, 0)
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header
            logger.info(f"Created leader control flag SHM: {self._control_flag_shm_name}")
        except Exception as e:
            logger.error(f"Failed to create leader control flag SHM: {e}")
            self._control_flag_shm = None
            self._leader_flag_owner = False

    def _write_leader_control_flag(self, flag_value: float, timestamp_ns: int) -> None:
        """Write leader button ownership flag to SHM."""
        if not self.is_leader_role or not self._is_dual_source_mode():
            return
        if self._control_flag_shm is None:
            self._create_leader_control_flag_shm()
        if self._control_flag_shm is None:
            return
        try:
            device_header_data = bytes(self._control_flag_shm.buf[:DEVICE_HEADER_SIZE])
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info["frame_size"]
            frame_offset = get_control_frame_offset(self._leader_flag_write_index, frame_size)
            frame_header = pack_frame_header(timestamp_ns)
            self._control_flag_shm.buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = frame_header

            flag_bytes = np.asarray([flag_value], dtype=np.float64).tobytes()
            data_start = frame_offset + FRAME_HEADER_SIZE
            self._control_flag_shm.buf[data_start:data_start + len(flag_bytes)] = flag_bytes

            self._leader_flag_frame_count = min(
                self._leader_flag_frame_count + 1,
                self._leader_flag_buffer_size,
            )
            self._leader_flag_write_index = (
                self._leader_flag_write_index + 1
            ) % self._leader_flag_buffer_size
            buffer_header = pack_buffer_header(
                self._leader_flag_buffer_size,
                self._leader_flag_frame_count,
                self._leader_flag_write_index,
            )
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header
        except Exception as e:
            logger.warning(f"Failed to write leader control flag: {e}")

    def _calculate_pose_difference(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """
        Calculate L2 norm difference between two joint poses.

        Args:
            pose1: First joint pose array
            pose2: Second joint pose array

        Returns:
            L2 norm of the difference
        """
        try:
            # Handle potential shape mismatches (14D with gripper vs 12D without)
            min_len = min(len(pose1), len(pose2))
            if min_len == 0:
                return float('inf')

            # Compare only joint positions (exclude gripper if present)
            # For I2RT: compare first 6 joints of each arm (12 total)
            compare_indices = list(range(6)) + list(range(7, 13))  # Skip gripper dims at [6] and [13]

            p1_joints = pose1[compare_indices] if len(pose1) >= 14 else pose1[:12]
            p2_joints = pose2[compare_indices] if len(pose2) >= 14 else pose2[:12]

            return float(np.linalg.norm(p1_joints - p2_joints))
        except Exception as e:
            logger.warning(f"Failed to calculate pose difference: {e}")
            return 0.0


    def _reconnect_rpc_clients(self) -> bool:
        """
        Attempt to reconnect RPC clients to minimum_gello servers.

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info("Attempting to reconnect I2RT RPC clients...")

        try:
            # Close existing clients
            if self._left_client is not None:
                try:
                    del self._left_client
                except Exception:
                    pass
            if self._right_client is not None:
                try:
                    del self._right_client
                except Exception:
                    pass

            self._left_client = None
            self._right_client = None

            # Wait brief delay for cleanup
            time.sleep(0.5)

            # Attempt reconnection with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    import importlib
                    portal_module = importlib.import_module("portal")

                    self._left_client = portal_module.Client(
                        f"{self.left_server_host}:{self.left_server_port}"
                    )
                    self._right_client = portal_module.Client(
                        f"{self.right_server_host}:{self.right_server_port}"
                    )

                    # Test connections
                    if self._left_client is not None:
                        self._left_client.get_info().result(timeout=2.0)
                    else:
                        logger.warning(f"Left client is None")
                    if self._right_client is not None:
                        self._right_client.get_info().result(timeout=2.0)
                    else:
                        logger.warning(f"Right client is None")
                    logger.info(f"Successfully reconnected RPC clients on attempt {attempt + 1}")
                    return True

                except Exception as e:
                    logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                    else:
                        self._left_client = None
                        self._right_client = None

            return False

        except Exception as e:
            logger.error(f"Failed to reconnect RPC clients: {e}")
            return False

    def _new_monitored_call_state(self) -> Dict[str, Any]:
        now = time.monotonic()
        return {
            "inflight_count": 0,
            "enter_monotonic": None,
            "last_exit_monotonic": now,
            "last_enter_log_monotonic": 0.0,
            "last_exit_log_monotonic": 0.0,
            "stuck_reported": False,
        }

    @contextmanager
    def _track_inflight_call(self, source: str):
        """Track a potentially blocking call for stuck-call watchdog reporting."""
        self._mark_monitored_call_enter(source)
        try:
            yield
        finally:
            self._mark_monitored_call_exit(source)

    def _mark_monitored_call_enter(self, source: str) -> None:
        """Mark a monitored call as entered."""
        now = time.monotonic()
        log_count = None
        with self._monitor_state_lock:
            state = self._monitored_call_states.setdefault(source, self._new_monitored_call_state())
            if state["inflight_count"] <= 0:
                state["enter_monotonic"] = now
            state["inflight_count"] += 1
            if now - state["last_enter_log_monotonic"] >= self._monitored_call_log_interval_s:
                state["last_enter_log_monotonic"] = now
                log_count = state["inflight_count"]
        # if log_count is not None:
        #     logger.info(
        #         f"I2RT monitored_call_enter source={source} "
        #         f"inflight_count={log_count} role={'leader' if self.is_leader_role else 'follower'}"
        #     )

    def _mark_monitored_call_exit(self, source: str) -> None:
        """Mark a monitored call as exited."""
        now = time.monotonic()
        log_count = None
        log_duration = 0.0
        with self._monitor_state_lock:
            state = self._monitored_call_states.setdefault(source, self._new_monitored_call_state())
            enter_time = state["enter_monotonic"]
            state["inflight_count"] = max(0, state["inflight_count"] - 1)
            if enter_time is not None:
                log_duration = now - enter_time
            if state["inflight_count"] == 0:
                state["enter_monotonic"] = None
                state["last_exit_monotonic"] = now
                state["stuck_reported"] = False
            if now - state["last_exit_log_monotonic"] >= self._monitored_call_log_interval_s:
                state["last_exit_log_monotonic"] = now
                log_count = state["inflight_count"]

    def _mark_state_call_enter(self) -> None:
        """Compatibility wrapper for older get_robot_state() instrumentation."""
        self._mark_monitored_call_enter("get_robot_state")

    def _mark_state_call_exit(self) -> None:
        """Compatibility wrapper for older get_robot_state() instrumentation."""
        self._mark_monitored_call_exit("get_robot_state")

    def _mark_rpc_get_info_enter(self) -> None:
        """Compatibility wrapper for older leader get_info() instrumentation."""
        self._mark_monitored_call_enter("left_leader_get_info_rpc")

    def _mark_rpc_get_info_exit(self) -> None:
        """Compatibility wrapper for older leader get_info() instrumentation."""
        self._mark_monitored_call_exit("left_leader_get_info_rpc")


    def _get_stuck_call_status(self) -> tuple[bool, str, float, float]:
        """Return current stuck-call status and durations."""
        now = time.monotonic()
        with self._monitor_state_lock:
            monitored_calls = [
                (
                    source,
                    int(state["inflight_count"]),
                    state["enter_monotonic"],
                    bool(state["stuck_reported"]),
                )
                for source, state in self._monitored_call_states.items()
            ]
            # last_motion_time = self._last_motion_monotonic

        stuck_source = ""
        inflight_duration = 0.0
        stuck_sources = []
        newly_stuck_calls = []
        for source, inflight_count, enter_time, stuck_reported in monitored_calls:
            if inflight_count <= 0 or enter_time is None:
                continue
            current_duration = now - enter_time
            inflight_duration = max(inflight_duration, current_duration)
            if current_duration >= self._stuck_call_timeout_s:
                stuck_sources.append(f"{source}({current_duration:.2f}s)")
                if not stuck_reported:
                    newly_stuck_calls.append((source, inflight_count, current_duration))

        if stuck_sources:
            stuck_source = ", ".join(stuck_sources)
        if newly_stuck_calls:
            with self._monitor_state_lock:
                for source, _inflight_count, _current_duration in newly_stuck_calls:
                    state = self._monitored_call_states.get(source)
                    if state is not None:
                        state["stuck_reported"] = True
            for source, inflight_count, current_duration in newly_stuck_calls:
                logger.warning(
                    f"I2RT monitored_call_stuck source={source} "
                    f"inflight_count={inflight_count} duration={current_duration:.3f}s "
                    f"timeout={self._stuck_call_timeout_s:.3f}s "
                    f"role={'leader' if self.is_leader_role else 'follower'}"
                )

        # stationary_duration = max(0.0, now - last_motion_time)
        # return bool(stuck_source), stuck_source, inflight_duration, stationary_duration
        return bool(stuck_source), stuck_source, inflight_duration

    def _reset_disconnection_monitor_state(self) -> None:
        """Reset monitoring state to avoid stale disconnection signals."""
        self._last_self_pose = None
        self._self_stationary_start_time = None
        now = time.monotonic()
        with self._monitor_state_lock:
            for state in self._monitored_call_states.values():
                state["inflight_count"] = 0
                state["enter_monotonic"] = None
                state["last_exit_monotonic"] = now
                state["last_enter_log_monotonic"] = 0.0
                state["last_exit_log_monotonic"] = 0.0
                state["stuck_reported"] = False
            self._last_motion_monotonic = now

    def _run_main_control_loop(self, loop_generation: int) -> None:
        """Run one generation of the main control loop."""
        logger.info(
            f"[I2RT main-loop g{loop_generation}] started "
            f"role={'leader' if self.is_leader_role else 'follower'}"
        )
        try:
            assert self.fps >= self.command_fps
            iteration = 1
            command_iteration = int(self.fps // self.command_fps)
            while self.running and loop_generation == self._main_control_loop_generation:
                start_time = time.time()

                if hasattr(self, '_check_and_consume_and_reply_robot_halt_command'):
                    if hasattr(self, '_last_robot_halt_check_time'):
                        if time.time() - self._last_robot_halt_check_time > 0.1:  # Check every 100ms
                            self._check_and_consume_and_reply_robot_halt_command()
                            self._last_robot_halt_check_time = time.time()
                    else:
                        self._last_robot_halt_check_time = time.time()
                        self._check_and_consume_and_reply_robot_halt_command()

                # Check for external home command (from GUI reset)
                # Only check periodically to avoid performance impact
                if hasattr(self, '_last_home_check_time'):
                    if time.time() - self._last_home_check_time > 0.1:  # Check every 100ms
                        self._check_and_execute_home_command()
                        self._last_home_check_time = time.time()
                else:
                    self._last_home_check_time = time.time()
                    self._check_and_execute_home_command()

                if hasattr(self, '_check_and_execute_teleop_start_command'):
                    if hasattr(self, '_last_teleop_start_check_time'):
                        if time.time() - self._last_teleop_start_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_start_command()
                            self._last_teleop_start_check_time = time.time()
                    else:
                        self._last_teleop_start_check_time = time.time()
                        self._check_and_execute_teleop_start_command()

                if hasattr(self, '_check_and_execute_teleop_discard_command'):
                    if hasattr(self, '_last_teleop_discard_check_time'):
                        if time.time() - self._last_teleop_discard_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_discard_command()
                            self._last_teleop_discard_check_time = time.time()
                    else:
                        self._last_teleop_discard_check_time = time.time()
                        self._check_and_execute_teleop_discard_command()
                # In legacy async mode the command thread owns source selection
                # and execution.  Keep this state loop free of hardware commands.
                if not self.legacy_async_mode:
                    self._process_source_selection_transition()
                    current_control_flag = self._selector_value_for_control_tick()
                    if current_control_flag is None:
                        current_control_flag = 1.0
                    # Detect transition from human to policy control
                    if not hasattr(self, '_last_control_flag'):
                        self._last_control_flag = 0.0
                    if self._enters_autonomous_control_mode(
                        self._last_control_flag,
                        current_control_flag,
                    ):
                        # Clear control SHM to prevent stale actions from being executed
                        self._clear_control_shm()
                        # Record when policy control started - used to filter stale actions
                        self._policy_control_start_timestamp = time.time_ns()
                        logger.info(f"Control mode switch detected in main loop: cleared control SHM, timestamp={self._policy_control_start_timestamp}")

                    if self.is_follower_role:
                        alignment_transition = (
                            abs(
                                current_control_flag - self._TAKEOVER_ALIGNMENT_FLAG
                            )
                            <= 1e-6
                            or abs(
                                self._last_control_flag
                                - self._TAKEOVER_ALIGNMENT_FLAG
                            )
                            <= 1e-6
                        )
                        if alignment_transition:
                            self._set_follower_control_mode_from_flag(
                                current_control_flag
                            )
                        elif (current_control_flag > self._TAKEOVER_ALIGNMENT_FLAG) != (
                            self._last_control_flag > self._TAKEOVER_ALIGNMENT_FLAG
                        ):
                            # Preserve the existing binary transition path unless
                            # the explicitly configured alignment state is active.
                            self._toggle_control_mode()

                    self._last_control_flag = current_control_flag

                control_dict = {}
                for shm_name in self.control_shm_names:
                    # Read latest control command
                    # Pass minimum timestamp to filter stale actions after control mode switch
                    # Add grace period to allow fresh policy actions to arrive
                    policy_start_ts = getattr(self, '_policy_control_start_timestamp', 0)
                    grace_period_ns = max(
                        getattr(self, '_policy_control_grace_period_ns', int(0.5 * 1e9)),
                        self._home_command_grace_period_ns,
                    )
                    last_home_command_timestamp = getattr(self, '_last_home_command_timestamp', 0)
                    min_timestamp_ns = policy_start_ts + grace_period_ns if policy_start_ts > 0 else 0
                    min_timestamp_ns = max(min_timestamp_ns, last_home_command_timestamp)
                    timestamp_ns, control_array = self.read_latest_control_command(shm_name, min_timestamp_ns)
                    control_dict[shm_name] = control_array

                if any([value is not None for value in control_dict.values()]): # Any non-None values in control_dict
                    if iteration % command_iteration == 0:
                        # Skip command execution while recovery is in progress.
                        if not self._recovery_active and not self._home_command_active and not self._robot_halt_command_active:
                            self.execute_control_command(control_dict)
                iteration = iteration % command_iteration + 1
                # Get current robot state
                state_array = self.get_robot_state()
                # Write state to shared memory
                if state_array is not None:
                    self._write_array_to_shm_with_timestamp(state_array, time.time_ns())

                # Maintain control frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info(f"Stopping {self.device_name}_{self.device_class_id} server...")
        except Exception as e:
            logger.error(f"Error in robot control loop: {e}")
        finally:
            is_latest_generation = (loop_generation == self._main_control_loop_generation)
            logger.info(
                f"[I2RT main-loop g{loop_generation}] exiting "
                f"is_latest={is_latest_generation} running={self.running}"
            )


    @track_i2rt_inflight_call("get_robot_state")
    def get_robot_state(self) -> np.ndarray:
        try:
            if self._left_client is None or self._right_client is None:
                return np.zeros(self.data_shape, dtype=get_dtype(self.data_dtype))

            timestamp_ns = time.time_ns()
            if self.is_leader_role:
                transition_lock = (
                    self._dagger_mode_transition_lock
                    if self._is_dual_source_mode() and self.takeover_alignment_enabled
                    else None
                )
                if transition_lock is not None:
                    transition_lock.acquire()
                try:
                    left_pos, btn_left = self._leader_get_info(self._left_client, "get_info_left_leader_get_robot_state")
                    right_pos, btn_right = self._leader_get_info(self._right_client, "get_info_right_leader_get_robot_state")
                    state = np.concatenate([left_pos, right_pos], axis=0)
                    button_down = bool(btn_left[0] > 0.5 or btn_right[0] > 0.5)
                    button_pressed = button_down and not self._button_down_latched
                    aligned_state = self._handle_mode_transition(
                        button_pressed,
                        current_leader_state=state,
                        button_down=button_down,
                    )
                    if aligned_state is not None:
                        state = aligned_state
                        left_pos = state[:7]
                        right_pos = state[7:14]
                        timestamp_ns = time.time_ns()
                    self._last_button_left = btn_left[0]
                    self._last_button_right = btn_right[0]
                    if button_pressed:
                        self._command_joint_pos(self._left_client, left_pos[:6], "command_joint_pos_left_get_robot_state_leader")
                        self._command_joint_pos(self._right_client, right_pos[:6], "command_joint_pos_right_get_robot_state_leader")
                    self._write_leader_control_flag(
                        1.0
                        if self._control_mode == self._CONTROL_MODE_TELEOP
                        else 0.0,
                        timestamp_ns,
                    )
                    return self._fit_state_shape(state)
                finally:
                    if transition_lock is not None:
                        transition_lock.release()

            left_pos = self._follower_get_info(self._left_client, "get_info_left_follower_get_robot_state")
            right_pos = self._follower_get_info(self._right_client, "get_info_right_follower_get_robot_state")
            state = np.concatenate([left_pos, right_pos], axis=0)
            fitted_state = self._fit_state_shape(state)
            self._latest_joint_state = fitted_state.copy()
            self._observe_follower_takeover_hold_feedback(
                fitted_state,
                measured_at_monotonic=time.monotonic(),
                measured_at_ns=time.time_ns(),
            )
            return fitted_state
        except Exception as e:
            logger.error(f"Failed to get I2RT robot state: {e}")
            return np.zeros(self.data_shape, dtype=get_dtype(self.data_dtype))

    def _check_and_consume_and_reply_robot_halt_command(self) -> None:
        """Check if the robot halt command is present and consume it."""
        from pathlib import Path
        robot_halt_command_file = Path(self._robot_halt_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id, device_id=self.device_class_id))
        if robot_halt_command_file.exists():
            logger.info(f"ROBOT HALT COMMAND FILE FOUND FOR {self.device_name}_{self.device_class_id}, SETTING ROBOT HALT COMMAND ACTIVE")
            self._robot_halt_command_active = True
            robot_halt_command_file.unlink()
            logger.info(f"Consumed robot halt command file for {self.device_name}_{self.device_class_id}")
            # reply
            robot_halt_reply_file = Path(self._robot_halt_reply_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id, device_id=self.device_class_id))
            robot_halt_reply_file.parent.mkdir(parents=True, exist_ok=True)
            robot_halt_reply_file.touch()
            logger.info(f"Replied robot halt command for {self.device_name}_{self.device_class_id}, file: {robot_halt_reply_file}")
        else:
            pass

    def _update_trusted_get_info_state(self, state_left: np.ndarray, state_right: np.ndarray) -> bool:
        def _format_float_list(values: np.ndarray) -> str:
            return "[" + ", ".join(f"{float(value):.3f}" for value in values.tolist()) + "]"

        all_none = (self._last_trusted_get_info_state_left is None and self._last_trusted_get_info_state_right is None )
        all_not_none = (self._last_trusted_get_info_state_left is not None and self._last_trusted_get_info_state_right is not None)
        if not (all_none or all_not_none):
            logger.info("both last_get_info_state_left and last_trusted_get_info_state_left and last_get_info_state_right and last_trusted_get_info_state_right must be None or not None")
            return False

        if all_none:
            with self._trusted_state_lock:
                self._last_trusted_get_info_state_left = state_left
                self._last_trusted_get_info_state_right = state_right
                return True
        else:

            left_state_diff = np.abs(state_left - self._last_trusted_get_info_state_left)
            right_state_diff = np.abs(state_right - self._last_trusted_get_info_state_right)

            # leave out the gripper state
            if len(left_state_diff) == 7:
                left_state_diff = left_state_diff[:6]
            if len(right_state_diff) == 7:
                right_state_diff = right_state_diff[:6]

            left_is_trusted = np.all(left_state_diff < self.state_outlier_threshold) # every joint and gripper is within the outlier threshold
            right_is_trusted = np.all(right_state_diff < self.state_outlier_threshold) # every joint and gripper is within the outlier threshold

            # left judgement
            if left_is_trusted:
                with self._trusted_state_lock:
                    self._last_trusted_get_info_state_left = state_left
            else:
                logger.warning(f"state_outlying! role: {'leader' if self.is_leader_role else 'follower'}, state_left: {_format_float_list(state_left)}, trusted_state_left: {_format_float_list(self._last_trusted_get_info_state_left)}, outlier_index: {np.where(left_state_diff > self.state_outlier_threshold)[0].tolist()}")

            # right judgement
            if right_is_trusted:
                with self._trusted_state_lock:
                    self._last_trusted_get_info_state_right = state_right
            else:
                logger.warning(f"state_outlying! role: {'leader' if self.is_leader_role else 'follower'}, state_right: {_format_float_list(state_right)}, trusted_state_right: {_format_float_list(self._last_trusted_get_info_state_right)}, outlier_index: {np.where(right_state_diff > self.state_outlier_threshold)[0].tolist()}")

            return left_is_trusted and right_is_trusted


    def _monitor_trusted_state(self) -> None:
        """
        Background thread that keeps robot trusted state updated.
        """
        logger.info("Trusted state monitoring thread started")

        while self._monitor_disconnection_evt.is_set():
            try:

                if self._left_client is None or self._right_client is None:
                    logger.error(f"Trusted state monitor sees invalid clients: left={self._left_client is None}, right={self._right_client is None}")
                    time.sleep(self._trusted_state_check_interval)
                    continue

                if self.is_leader_role:
                    left_pos, btn_left = self._leader_get_info(self._left_client, "get_info_left_leader_trusted_state")
                    right_pos, btn_right = self._leader_get_info(self._right_client, "get_info_right_leader_trusted_state")
                else:
                    left_pos = self._follower_get_info(self._left_client, "get_info_left_follower_trusted_state")
                    right_pos = self._follower_get_info(self._right_client, "get_info_right_follower_trusted_state")

                if not self._robot_halt_command_active:
                    self._update_trusted_get_info_state(left_pos, right_pos)
                else:
                    logger.warning(f"robot halt command is active, skipping trusted state update for {self.device_name}_{self.device_class_id}")

                time.sleep(self._trusted_state_check_interval)

            except Exception as e:
                logger.error(f"Error in disconnection monitoring: {e}")
                time.sleep(self._trusted_state_check_interval)

        logger.info("Disconnection monitoring thread stopped")

    def _monitor_stuck_calls(self) -> None:
        """Watchdog that triggers recovery when state/RPC calls are stuck."""
        logger.info("Disconnection stuck-call watchdog thread started")
        while self._monitor_disconnection_evt.is_set():
            try:
                if not self._disconnection_recovery_enabled:
                    time.sleep(self._stuck_call_check_interval)
                    continue

                if self._recovery_active:
                    time.sleep(self._stuck_call_check_interval)
                    continue

                # is_stuck, stuck_source, inflight_duration, stationary_duration = self._get_stuck_call_status()
                is_stuck, stuck_source, inflight_duration = self._get_stuck_call_status()

                if not is_stuck:
                    time.sleep(self._stuck_call_check_interval)
                    continue

                logger.error(
                    f"Disconnection confirmed by stuck-call watchdog: source={stuck_source}, "
                    f"inflight_duration={inflight_duration:.2f}s, "
                    # f"stationary_duration={stationary_duration:.2f}s, "
                    f"role={'leader' if self.is_leader_role else 'follower'}, mode={self._control_mode}"
                )
                self._execute_disconnection_recovery()
                self._reset_disconnection_monitor_state()
            except Exception as e:
                logger.error(f"Error in stuck-call watchdog: {e}")
            time.sleep(self._stuck_call_check_interval)
        logger.info("Disconnection stuck-call watchdog thread stopped")

    def _execute_disconnection_recovery(self) -> None:
        """Execute recovery sequence when disconnection is detected."""
        if self._recovery_active:
            logger.warning("Recovery already in progress, skipping duplicate recovery")
            return

        self._recovery_active = True
        logger.warning("="*60)
        logger.warning("EXECUTING DISCONNECTION RECOVERY")
        logger.warning("="*60)

        try:
            # Step 1: Discard current episode and notify all watchers
            logger.info("Step 1/5: Discarding current episode and informing Leader Robot...")
            # self._write_episode_signal("discard")
            self._signal_connector_robot_reconnect()

            # Step 2: Reconnect RPC clients
            logger.info("Step 2/5: Reconnecting RPC clients...")
            reconnect_success = self._reconnect_rpc_clients()
            if not reconnect_success:
                logger.error("Failed to reconnect RPC clients, recovery incomplete")
                return

            # The Robot should not clear shared memory during disconnection recovery, we should let the server_thread to handle this

            # Step 3: Reset control mode
            logger.info("Step 3/5: Resetting control mode...")
            if self._is_dual_source_mode():
                self._control_mode = self._CONTROL_MODE_AUTONOMOUS
            else:
                self._control_mode = self._CONTROL_MODE_TELEOP
            self._apply_bilateral_gains_for_mode()

            # Step 4: Go home position
            logger.info("Step 4/5: Moving to home position...")
            home_success = self._go_home(reason="recovery")
            if not home_success:
                logger.error("Failed to move to home position")
            else:
                logger.info("Successfully moved to home position")

            # Step 5: Restart main control loop to bypass stuck execute_control_command.
            logger.info("Step 5/5: Restarting main control loop thread...")
            self._restart_main_control_loop(reason="disconnection_recovery")

            # Notify policy runner to reset
            self._notify_policy_runner_reset()

            logger.warning("="*60)
            logger.warning("DISCONNECTION RECOVERY COMPLETED")
            logger.warning("="*60)

        except Exception as e:
            logger.error(f"Exception during disconnection recovery: {e}")
        finally:
            self._recovery_active = False
            self._reset_disconnection_monitor_state()

    def _signal_connector_robot_reconnect(self) -> None:
        """Notify policy connector to treat current episode as discard."""
        if not self._robot_reconnect_signal:
            logger.warning("robot_reconnect signal file path is missing, skip notifying connector")
            return
        try:
            from pathlib import Path

            reconnect_signal_file = Path(self._robot_reconnect_signal)
            reconnect_signal_file.parent.mkdir(parents=True, exist_ok=True)
            reconnect_signal_file.touch()
            logger.info(
                f"Notified connector with robot reconnect signal: {reconnect_signal_file}"
            )
        except Exception as e:
            logger.warning(f"Failed to notify connector robot reconnect signal: {e}")

    @track_i2rt_inflight_call("execute_control_command")
    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        try:
            if self._left_client is None or self._right_client is None:
                return
            if self.is_follower_role:
                if self._control_mode == self._CONTROL_MODE_ALIGNMENT:
                    hold_state = self._takeover_hold_joint_pos
                    if not self._valid_takeover_joint_state(hold_state):
                        logger.warning("I2RT takeover alignment has no valid hold pose")
                        return
                    control_array = np.asarray(hold_state, dtype=np.float64).reshape(-1)
                else:
                    selected_array = self._get_control_array(control_dict)
                    if selected_array is None:
                        return
                    control_array = np.asarray(selected_array, dtype=np.float64).reshape(-1)

                if control_array.size != 14:
                    logger.warning(f"I2RT follower expects 14D command, got {control_array.size}")
                    return

                self._command_joint_pos(self._left_client, control_array[:7], "command_joint_pos_left_execute_control_follower")
                self._command_joint_pos(self._right_client, control_array[7:14], "command_joint_pos_right_execute_control_follower")
                return

            if self._is_dual_source_mode() and self._control_mode != self._CONTROL_MODE_AUTONOMOUS:
                return
            follower_state = self._get_control_array(control_dict)
            if follower_state is None:
                return

            follow_array = np.asarray(follower_state, dtype=np.float64).reshape(-1)
            if follow_array.size < 14:
                logger.warning(f"I2RT leader follow expects at least 14D source, got {follow_array.size}")
                return
            # Leader arm accepts only 6-DoF command per arm, so gripper dimension is dropped.
            self._command_joint_pos(self._left_client, follow_array[:6], "command_joint_pos_left_execute_control_leader")
            self._command_joint_pos(self._right_client, follow_array[7:13], "command_joint_pos_right_execute_control_leader")

        except Exception as e:
            logger.error(f"Failed to execute I2RT control command: {e}")

    def _cleanup_control_shared_memory(self) -> None:
        """Cleanup SHM resources without unlinking externally-owned SHMs."""
        super()._cleanup_control_shared_memory()

        if self._leader_flag_owner and self._control_flag_shm is not None:
            try:
                self._control_flag_shm.close()
                self._control_flag_shm.unlink()
                logger.info(f"Cleaned up leader control flag SHM: {self._control_flag_shm_name}")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Failed to cleanup leader control flag SHM: {e}")
            finally:
                self._control_flag_shm = None
                self._leader_flag_owner = False

    def _check_and_execute_home_command(self) -> None:
        """Check for external home/teleop command files and execute if present."""
        try:
            from pathlib import Path
            home_command_file = Path(self._robot_home_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id, device_id=self.device_class_id))
            has_home_command = home_command_file.exists()
            if not has_home_command:
                return
            logger.info(f"Received home command for {self.device_name}_{self.device_class_id}")
            random_home_request_id: Optional[str] = None
            try:
                command_content = home_command_file.read_text(encoding="utf-8").strip()
                if command_content.startswith("{"):
                    command_payload = json.loads(command_content)
                    if not isinstance(command_payload, Mapping):
                        raise ValueError("I2RT home command must be a JSON object")
                    if command_payload.get("schema_version") != "i2rt/home_command/v1":
                        raise ValueError("I2RT home command has an unsupported schema")
                    home_index = command_payload.get("home_index", 0)
                    if isinstance(home_index, bool) or not isinstance(home_index, int):
                        raise ValueError("I2RT home command home_index must be an integer")
                    request_id = command_payload.get("random_home_request_id")
                    if request_id is not None and not isinstance(request_id, str):
                        raise ValueError(
                            "I2RT home command random_home_request_id must be a string"
                        )
                    random_home_request_id = request_id
                if (
                    self._home_requires_random_sync("episode_reset")
                    and random_home_request_id is None
                ):
                    raise ValueError(
                        "I2RT synchronized randomized-home command requires a fresh request ID"
                    )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                logger.error(f"Ignoring invalid I2RT home command: {exc}")
                try:
                    home_command_file.unlink()
                except OSError:
                    pass
                return
            try:
                home_command_file.unlink()
            except Exception:
                pass

            logger.info(f"unlinking home command file for {self.device_name}_{self.device_class_id}")
            self._home_command_active = True
            self._clear_control_shm()
            self._control_mode = self._CONTROL_MODE_AUTONOMOUS
            self._apply_bilateral_gains_for_mode()
            success = self._go_home(
                reason="episode_reset",
                random_home_request_id=random_home_request_id,
            )
            if self._home_requires_random_sync("episode_reset"):
                try:
                    assert random_home_request_id is not None
                    self._publish_random_home_result(
                        request_id=random_home_request_id,
                        success=success,
                    )
                except Exception as exc:
                    success = False
                    logger.error(
                        f"Failed to publish I2RT randomized-home result: {exc}"
                    )
            home_end_timestamp_ns = time.time_ns()
            self._clear_control_shm()
            self._last_home_command_timestamp = max(
                getattr(self, '_last_home_command_timestamp', 0),
                home_end_timestamp_ns,
            )
            self._policy_control_start_timestamp = max(
                getattr(self, '_policy_control_start_timestamp', 0),
                home_end_timestamp_ns,
            )
            if success:
                self._notify_policy_runner_reset()
                logger.info(
                    "I2RT home command completed successfully; "
                    f"control commands before {home_end_timestamp_ns} are ignored, "
                    f"grace={self._home_command_grace_period_ns / 1e9:.3f}s"
                )
            else:
                logger.error("I2RT home command failed")

            logger.info(f"FINISH CHECK_AND_EXECUTE_HOME_COMMAND, HOME COMMAND ACTIVE FOR {self.device_name}_{self.device_class_id} IS SET TO FALSE")
            self._robot_halt_command_active = False

        except Exception as e:
            logger.debug(f"I2RT home command check failed: {e}")
        finally:
            self._home_command_active = False


    def _check_and_execute_teleop_start_command(self) -> None:
        """Consume teleop-start command file and switch to teleoperation mode."""
        from pathlib import Path
        teleop_start_command_file = Path(self._robot_teleop_start_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id, device_id=self.device_class_id))
        has_teleop_start_command = teleop_start_command_file.exists()
        if not has_teleop_start_command:
            return
        try:
            try:
                teleop_start_command_file.unlink()
            except Exception:
                pass

            self._control_mode = self._CONTROL_MODE_TELEOP
            self._apply_bilateral_gains_for_mode()
            curr_state = self.get_robot_state()
            if curr_state is not None:
                if self.is_leader_role:
                    self._command_joint_pos(self._left_client, curr_state[:6], "command_joint_pos_left_check_and_execute_teleop_start_leader")
                    self._command_joint_pos(self._right_client, curr_state[7:13], "command_joint_pos_right_check_and_execute_teleop_start_leader")
                else:
                    self._command_joint_pos(self._left_client, curr_state[:7], "command_joint_pos_left_check_and_execute_teleop_start_follower")
                    self._command_joint_pos(self._right_client, curr_state[7:14], "command_joint_pos_right_check_and_execute_teleop_start_follower")
                logger.info(f"Set joint pos after bilateral gains: {curr_state}")
            else:
                logger.error(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")
                raise RuntimeError(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")

            logger.info(f"Received teleop-start command for {self.device_name}_{self.device_class_id}")
        except Exception as e:
            logger.warning(f"Failed to process teleop-start command: {e}")

    def _check_and_execute_teleop_discard_command(self) -> None:
        """Consume teleop-discard command file and switch to autonomous mode."""
        from pathlib import Path
        teleop_discard_command_file = Path(self._robot_teleop_discard_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id, device_id=self.device_class_id))
        has_teleop_discard_command = teleop_discard_command_file.exists()
        if not has_teleop_discard_command:
            return
        logger.info(f"Received teleop-discard command for {self.device_name}_{self.device_class_id}")
        try:
            self._is_recording_episode = not self._is_recording_episode
            teleop_discard_command_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to process teleop-discard command: {e}")

    def _clear_control_shm(self) -> None:
        """Clear robot-owned control SHM to drop stale commands."""
        try:
            if self.control_shared_memory is None or self.default_control_shm_name not in self.control_shared_memory:
                return
            control_shm = self.control_shared_memory.get(self.default_control_shm_name)
            if control_shm is None:
                return

            try:
                control_shm.buf[0] = control_shm.buf[0]
                writable_shm = control_shm
            except (PermissionError, AttributeError, TypeError, ValueError):
                writable_shm = connect_to_shared_memory(self.default_control_shm_name, read_only=False)
                self.control_shared_memory[self.default_control_shm_name] = writable_shm

            buf = writable_shm.buf
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            frame_size = device_header["frame_size"]
            buffer_size = buffer_header["buffer_size"]
            zero_ts = pack_frame_header(0)
            for i in range(buffer_size):
                frame_offset = get_control_frame_offset(i, frame_size)
                buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = zero_ts
                data_start = frame_offset + FRAME_HEADER_SIZE
                buf[data_start:frame_offset + frame_size] = b"\x00" * (frame_size - FRAME_HEADER_SIZE)
            reset_header = pack_buffer_header(buffer_size, 0, 0)
            buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = reset_header
            logger.info(f"Cleared I2RT control SHM: {self.default_control_shm_name}")
        except Exception as e:
            logger.error(f"Failed to clear I2RT control SHM: {e}")

    def _validate_random_init_pose_config(self) -> None:
        """Normalize bounded randomized-home settings before touching hardware."""
        config = self.random_init_pose_params
        try:
            position_radius_m = float(config.get("position_radius_m", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("random_init_pose_params.position_radius_m must be numeric") from exc
        if not np.isfinite(position_radius_m) or not 0.0 <= position_radius_m <= 0.15:
            raise ValueError(
                "random_init_pose_params.position_radius_m must be finite and in [0, 0.15]"
            )
        config["position_radius_m"] = position_radius_m

        try:
            rotation_range = np.asarray(
                config.get("rotation_euler_range_deg", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "random_init_pose_params.rotation_euler_range_deg must be numeric"
            ) from exc
        if rotation_range.size == 1:
            rotation_range = np.repeat(rotation_range, 3)
        if (
            rotation_range.size != 3
            or not np.all(np.isfinite(rotation_range))
            or np.any(rotation_range < 0.0)
            or np.any(rotation_range > 30.0)
        ):
            raise ValueError(
                "random_init_pose_params.rotation_euler_range_deg must contain "
                "three finite values in [0, 30]"
            )
        config["rotation_euler_range_deg"] = rotation_range

        bool_defaults = {
            "independent_per_arm": True,
            "fallback_to_nominal_home": True,
            "randomize_on_startup": True,
            "randomize_on_episode_reset": True,
            "randomize_on_recovery": False,
        }
        for key, default in bool_defaults.items():
            value = config.get(key, default)
            if not isinstance(value, bool):
                raise ValueError(f"random_init_pose_params.{key} must be a boolean")
            config[key] = value

        for key, default, minimum, maximum in (
            ("max_ik_attempts", 20, 1, 100),
            ("ik_max_iters", 200, 1, 1000),
        ):
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(
                    f"random_init_pose_params.{key} must be an integer in [{minimum}, {maximum}]"
                )
            config[key] = value

        for key, default, minimum, maximum in (
            ("ik_position_tolerance_m", 1e-4, 1e-6, 0.01),
            ("ik_orientation_tolerance_deg", 0.2, 1e-4, 5.0),
            ("sync_wait_timeout_s", 15.0, 0.1, 60.0),
        ):
            try:
                value = float(config.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"random_init_pose_params.{key} must be numeric") from exc
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"random_init_pose_params.{key} must be finite and in [{minimum}, {maximum}]"
                )
            config[key] = value

        seed = config.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("random_init_pose_params.seed must be an integer or null")

    def _should_randomize_init_pose(self, reason: str) -> bool:
        """Return whether the current home transition should sample a new pose.

        Recovery always returns to the configured nominal position.  A partially
        recovered leader/follower pair must never independently sample a new
        target while the other process is reconnecting.
        """
        if not self.random_init_pose:
            return False
        if reason == "recovery":
            return False
        key = {
            "startup": "randomize_on_startup",
            "episode_reset": "randomize_on_episode_reset",
        }.get(reason, "randomize_on_episode_reset")
        return bool(self.random_init_pose_params[key])

    def _home_requires_random_sync(self, reason: str) -> bool:
        """Recovery remains nominal and independent by design."""
        return self.random_init_pose_sync and reason in {"startup", "episode_reset"}

    def _get_i2rt_kinematics(self) -> Any:
        """Lazily construct the current pinned-submodule FK/IK adapter."""
        if self._i2rt_kinematics is None:
            from utils.i2rt_kinematics import create_i2rt_kinematics

            self._i2rt_kinematics, self._i2rt_kinematics_backend = create_i2rt_kinematics(
                site_name="grasp_site"
            )
            logger.info(
                f"I2RT randomized-home kinematics backend: "
                f"{self._i2rt_kinematics_backend}"
            )
        return self._i2rt_kinematics

    def _sample_random_init_pose_offset(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a uniform position in a ball and independent Euler offsets."""
        radius_limit = float(self.random_init_pose_params["position_radius_m"])
        if radius_limit == 0.0:
            position_delta = np.zeros(3, dtype=np.float64)
        else:
            direction = self._random_init_pose_rng.normal(size=3)
            while np.linalg.norm(direction) < 1e-12:
                direction = self._random_init_pose_rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            radius = radius_limit * float(self._random_init_pose_rng.random()) ** (1.0 / 3.0)
            position_delta = direction * radius
        rotation_range = np.asarray(
            self.random_init_pose_params["rotation_euler_range_deg"],
            dtype=np.float64,
        )
        return position_delta, self._random_init_pose_rng.uniform(
            low=-rotation_range,
            high=rotation_range,
        )

    def _solve_randomized_home_arm(
        self,
        nominal_q: np.ndarray,
        position_delta: np.ndarray,
        euler_delta_deg: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Offset one follower arm EEF pose and solve it against vendor limits."""
        from utils.i2rt_kinematics import arm_joint_limits

        kinematics = self._get_i2rt_kinematics()
        nominal_q = np.asarray(nominal_q, dtype=np.float64).reshape(6)
        nominal_pose = np.asarray(kinematics.fk(nominal_q, "grasp_site"), dtype=np.float64)
        if nominal_pose.shape != (4, 4) or not np.all(np.isfinite(nominal_pose)):
            raise RuntimeError("I2RT FK returned an invalid nominal pose")
        target_pose = nominal_pose.copy()
        target_pose[:3, 3] += np.asarray(position_delta, dtype=np.float64).reshape(3)
        target_pose[:3, :3] = nominal_pose[:3, :3] @ R.from_euler(
            "xyz", euler_delta_deg, degrees=True
        ).as_matrix()
        success, solution = kinematics.ik(
            target_pose,
            "grasp_site",
            init_q=nominal_q,
            max_iters=self.random_init_pose_params["ik_max_iters"],
            pos_threshold=self.random_init_pose_params["ik_position_tolerance_m"],
            ori_threshold=np.deg2rad(
                self.random_init_pose_params["ik_orientation_tolerance_deg"]
            ),
            verbose=False,
        )
        solution = np.asarray(solution, dtype=np.float64).reshape(-1)
        if not success or solution.size < 6 or not np.all(np.isfinite(solution[:6])):
            return None
        solution = solution[:6]
        lower, upper = arm_joint_limits(kinematics)
        if np.any(solution < lower) or np.any(solution > upper):
            return None
        return solution

    def _generate_randomized_home(self, nominal_home: np.ndarray) -> Optional[np.ndarray]:
        """Generate a valid 14D follower target while preserving both grippers."""
        nominal_home = np.asarray(nominal_home, dtype=np.float64).reshape(-1)
        if nominal_home.size != 14 or not np.all(np.isfinite(nominal_home)):
            logger.error("Randomized I2RT follower home requires 14 finite joints")
            return None
        left_nominal_q = nominal_home[:6]
        right_nominal_q = nominal_home[7:13]
        for attempt in range(1, self.random_init_pose_params["max_ik_attempts"] + 1):
            left_position_delta, left_euler_delta = self._sample_random_init_pose_offset()
            if self.random_init_pose_params["independent_per_arm"]:
                right_position_delta, right_euler_delta = self._sample_random_init_pose_offset()
            else:
                right_position_delta = left_position_delta.copy()
                right_euler_delta = left_euler_delta.copy()
            try:
                left_solution = self._solve_randomized_home_arm(
                    left_nominal_q,
                    left_position_delta,
                    left_euler_delta,
                )
                right_solution = self._solve_randomized_home_arm(
                    right_nominal_q,
                    right_position_delta,
                    right_euler_delta,
                )
            except Exception as exc:
                logger.warning(
                    f"I2RT randomized-home IK attempt {attempt}/"
                    f"{self.random_init_pose_params['max_ik_attempts']} failed: {exc}"
                )
                continue
            if left_solution is None or right_solution is None:
                continue
            randomized_home = nominal_home.copy()
            randomized_home[:6] = left_solution
            randomized_home[7:13] = right_solution
            return randomized_home
        logger.error(
            "I2RT randomized-home IK failed after "
            f"{self.random_init_pose_params['max_ik_attempts']} attempts"
        )
        return None

    def _wait_for_random_home_request(self, *, reason: str) -> Optional[str]:
        """Wait briefly for the orchestrator's fresh startup request."""
        from utils.i2rt_random_home import RandomHomeSyncError, read_random_home_request

        deadline = time.monotonic() + self.random_init_pose_params["sync_wait_timeout_s"]
        while time.monotonic() <= deadline:
            try:
                request_id = read_random_home_request(
                    self._random_home_request_path,
                    reason=reason,
                )
            except RandomHomeSyncError as exc:
                logger.error(f"I2RT randomized-home request is invalid: {exc}")
                return None
            if request_id is not None:
                return request_id
            time.sleep(0.05)
        logger.error(f"Timed out waiting for I2RT randomized-home request ({reason})")
        return None

    def _resolve_random_home_request_id(
        self,
        *,
        reason: str,
        requested_id: Optional[str],
    ) -> Optional[str]:
        """Verify that a command refers to the currently published request.

        Startup has no command file, so the device may wait for the starter's
        request.  Episode resets must carry the exact ID that the controller
        just published; otherwise a late legacy command could consume a stale
        request and make the pair choose unrelated homes.
        """
        if requested_id is None:
            if reason != "startup":
                logger.error(
                    "I2RT synchronized randomized-home reset is missing a request ID"
                )
                return None
            return self._wait_for_random_home_request(reason=reason)

        from utils.i2rt_random_home import RandomHomeSyncError, read_random_home_request

        try:
            current_id = read_random_home_request(
                self._random_home_request_path,
                reason=reason,
            )
        except RandomHomeSyncError as exc:
            logger.error(f"I2RT randomized-home request is invalid: {exc}")
            return None
        if current_id is None:
            logger.error(
                f"I2RT randomized-home request for {reason} is missing or has a different reason"
            )
            return None
        if current_id != requested_id:
            logger.error(
                "I2RT randomized-home command request ID does not match the current request"
            )
            return None
        return current_id

    def _publish_random_home_target(self, home: np.ndarray, *, request_id: str) -> None:
        from utils.i2rt_random_home import publish_random_home_target

        publish_random_home_target(
            self._random_home_target_path,
            request_id=request_id,
            home_joint_pos=home,
            source_device_id=int(self.device_class_id),
        )

    def _publish_random_home_result(self, *, request_id: str, success: bool) -> None:
        """Publish this device's completed I2RT home result for one request."""
        from utils.i2rt_random_home import publish_random_home_result

        publish_random_home_result(
            self._random_home_result_pattern,
            request_id=request_id,
            source_device_id=int(self.device_class_id),
            success=success,
        )

    def _wait_for_random_home_target(self, *, request_id: str) -> Optional[np.ndarray]:
        from utils.i2rt_random_home import RandomHomeSyncError, read_random_home_target

        deadline = time.monotonic() + self.random_init_pose_params["sync_wait_timeout_s"]
        while time.monotonic() <= deadline:
            try:
                target = read_random_home_target(
                    self._random_home_target_path,
                    request_id=request_id,
                    expected_source_device_id=self._takeover_peer_device_id(),
                )
            except RandomHomeSyncError as exc:
                logger.warning(f"I2RT randomized-home target is invalid: {exc}")
                target = None
            if target is not None:
                return target
            time.sleep(0.05)
        logger.error("Timed out waiting for matching I2RT randomized-home target")
        return None

    @staticmethod
    def _apply_follower_home_to_leader(
        leader_home: np.ndarray,
        follower_home: np.ndarray,
    ) -> np.ndarray:
        """Mirror follower arm joints while retaining the leader's gripper values."""
        leader = np.asarray(leader_home, dtype=np.float64).reshape(-1).copy()
        follower = np.asarray(follower_home, dtype=np.float64).reshape(-1)
        if follower.size != 14 or not np.all(np.isfinite(follower)):
            raise ValueError("synchronized follower home must contain 14 finite joints")
        if leader.size == 14:
            leader[:6] = follower[:6]
            leader[7:13] = follower[7:13]
        elif leader.size == 12:
            leader[:6] = follower[:6]
            leader[6:12] = follower[7:13]
        else:
            raise ValueError("leader home must contain 12 or 14 joints")
        return leader

    def _go_home(
        self,
        *,
        reason: str = "episode_reset",
        random_home_request_id: Optional[str] = None,
    ) -> bool:
        """Move both I2RT arms to configured home joint positions."""
        logger.info(f"======== GOING HOME!!! for {self.device_name}_{self.device_class_id} =========")
        if reason not in {"startup", "episode_reset", "recovery"}:
            logger.error(f"Unsupported I2RT home reason: {reason}")
            return False
        requires_random_sync = self._home_requires_random_sync(reason)
        if self.home_joint_pos is None:
            if requires_random_sync:
                logger.error(
                    "I2RT synchronized randomized-home requires home_joint_pos on both devices"
                )
                return False
            logger.warning("home_joint_pos is not configured, skipping go-home")
            return True
        if self._left_client is None or self._right_client is None:
            logger.error("I2RT clients are not operational for go-home")
            return False
        try:
            home = np.asarray(self.home_joint_pos, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(home)):
                logger.error("I2RT go-home requires finite joint positions")
                return False
            request_id: Optional[str] = None
            if requires_random_sync:
                request_id = self._resolve_random_home_request_id(
                    reason=reason,
                    requested_id=random_home_request_id,
                )
                if request_id is None:
                    return False
            if self.is_follower_role:
                if home.size != 14:
                    logger.error(f"Follower go-home expects 14 values, got {home.size}")
                    return False
                if self._should_randomize_init_pose(reason):
                    randomized_home = self._generate_randomized_home(home)
                    if randomized_home is None:
                        if not self.random_init_pose_params["fallback_to_nominal_home"]:
                            logger.error(
                                "I2RT randomized-home failed and nominal fallback is disabled"
                            )
                            return False
                        logger.warning(
                            "I2RT randomized-home failed; falling back to nominal home"
                        )
                    else:
                        home = randomized_home
                # left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_home")
                # right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_home")

                if self._last_trusted_get_info_state_left is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    left_current = self._last_trusted_get_info_state_left
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_home")

                if self._last_trusted_get_info_state_right is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    right_current = self._last_trusted_get_info_state_right
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_home")

                left_target = home[:7]
                right_target = home[7:14]
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current,
                    right_current=right_current,
                )
                if requires_random_sync:
                    self._publish_random_home_target(home, request_id=request_id)
            else:
                if requires_random_sync:
                    follower_home = self._wait_for_random_home_target(
                        request_id=request_id
                    )
                    if follower_home is None:
                        return False
                    home = self._apply_follower_home_to_leader(home, follower_home)
                if home.size == 14:
                    left_target = home[:6]
                    right_target = home[7:13]
                elif home.size == 12:
                    left_target = home[:6]
                    right_target = home[6:12]
                else:
                    logger.error(f"Leader go-home expects 12 or 14 values, got {home.size}")
                    return False

                # left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_home")
                # right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_home")

                if self._last_trusted_get_info_state_left is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    left_current = self._last_trusted_get_info_state_left
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_home")

                if self._last_trusted_get_info_state_right is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    right_current = self._last_trusted_get_info_state_right
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_home")

                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current[:6],
                    right_current=right_current[:6],
                )

            logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, left_current: {left_current.tolist()}, right_current: {right_current.tolist()}")
            self._last_button_left = 0.0
            self._last_button_right = 0.0
            self._last_control_flag = 0.0 if self._is_dual_source_mode() else 1.0

            with self._trusted_state_lock:
                self._last_trusted_get_info_state_left = None
                self._last_trusted_get_info_state_right = None
                logger.info(f"reset trusted state for {self.device_name}_{self.device_class_id}")

            return True
        except Exception as e:
            logger.error(f"Failed to execute I2RT go-home: {e}")
            return False

    def _go_rest(self) -> bool:
        """Move both I2RT arms to configured rest joint positions."""
        logger.info(f"GOING REST!!! for {self.device_name}_{self.device_class_id}")
        if self._left_client is None or self._right_client is None:
            logger.error("I2RT clients are not operational for go-rest")
            return False
        try:
            rest_joint_pos = np.zeros_like(self.home_joint_pos)
            rest_joint_pos[6] = 0.99
            rest_joint_pos[13] = 0.99
            if self.is_follower_role:
                left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_rest")
                right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_rest")
                left_target = rest_joint_pos[:7]
                right_target = rest_joint_pos[7:14]
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current,
                    right_current=right_current,
                )
            else:
                left_target = rest_joint_pos[:6]
                right_target = rest_joint_pos[7:13]
                left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_rest")
                right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_rest")
                self._left_client.update_kp_kd(self._leader_default_kp, self._leader_default_kd)
                self._right_client.update_kp_kd(self._leader_default_kp, self._leader_default_kd)
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current[:6],
                    right_current=right_current[:6],
                )
            return True
        except Exception as e:
            logger.error(f"Failed to execute I2RT go-rest: {e}")
            return False

    def _read_control_flag(self) -> float:
        if self.is_leader_role:
            return float(self._control_mode == self._CONTROL_MODE_TELEOP)
        return super()._read_control_flag()

    def _get_control_array(self, control_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Return only the endpoint selected by ``control.source_selection``."""
        selected_name = self._selected_source_name(
            self._selector_value_for_control_tick()
        )
        return control_dict.get(selected_name) if selected_name is not None else None

    def _poll_and_execute_control_once(self) -> None:
        """Keep the legacy command thread inactive during safety operations."""
        if (
            self._recovery_active
            or self._home_command_active
            or self._robot_halt_command_active
        ):
            return
        super()._poll_and_execute_control_once()

    def _notify_policy_runner_reset(self) -> None:
        """Notify policy runner and policy connector to reset when robot is resetting."""
        try:
            from pathlib import Path

            # Create reset command file for policy runner (ActionExecutor)
            reset_command_file = Path(self._policy_runner_reset_command)
            reset_command_file.parent.mkdir(parents=True, exist_ok=True)
            reset_command_file.touch()

            # Also create reset signal file for policy connector
            # This notifies the policy connector to reset its RTC state
            reset_signal_file = Path(self._reset_signal)
            reset_signal_file.touch()

            logger.info("Notified policy runner and policy connector to reset due to robot reset")
        except Exception as e:
            logger.warning(f"Failed to notify policy runner/connector of reset: {e}")

    def _get_status(self) -> dict:
        return {
            "mode": self.mode,
            "is_leader_role": self.is_leader_role,
            "control_shm_names": self.control_shm_names,
            "source_selector_shm_name": self._control_flag_shm_name,
            "left_endpoint": f"{self.left_server_host}:{self.left_server_port}",
            "right_endpoint": f"{self.right_server_host}:{self.right_server_port}",
        }

    def get_status(self) -> dict:
        status = super().get_status()
        status.update(self._get_status())
        return status

    def _prepare_main_control_loop_resources(self) -> None:
        """Prepare SHM resources before entering main control loop."""
        if self.running:
            logger.info(f"Robot {self.device_name}_{self.device_class_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_class_id} server...")
        self.running = True

        # Create state shared memory
        self._create_shared_memory()
        self._prepare_control_loop_resources()

        if not self.shared_memory or not any(self.control_shared_memory.values()):
            self.running = False
            raise RuntimeError("Failed to create/connect to shared memories")

        logger.info(f"Robot {self.device_name}_{self.device_class_id} server started successfully")
        if self.legacy_async_mode:
            self._start_command_loop_thread()
            logger.info(
                "Started precise command thread at {:.3f} Hz (state loop {:.3f} Hz)",
                self.command_fps,
                self.fps,
            )

    def _force_kill_thread(self, thread: threading.Thread, reason: str) -> bool:
        """Force-stop a thread by injecting SystemExit (unsafe fallback)."""
        if not thread.is_alive():
            return True

        thread_id = thread.ident
        if thread_id is None:
            logger.error(f"Failed to force-kill thread ({reason}): thread ident is None")
            return False

        logger.warning(
            f"Force-killing thread ({reason}) using PyThreadState_SetAsyncExc. "
            f"thread={thread.name}, ident={thread_id}"
        )
        try:
            set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
            set_async_exc.argtypes = [ctypes.c_ulong, ctypes.py_object]
            set_async_exc.restype = ctypes.c_int
            affected = set_async_exc(ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit))
            if affected == 0:
                logger.error(f"Force-kill failed: no thread state found for ident={thread_id}")
                return False
            if affected > 1:
                set_async_exc(ctypes.c_ulong(thread_id), None)
                logger.error(
                    f"Force-kill aborted: multiple thread states affected ({affected}) "
                    f"for ident={thread_id}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Force-kill thread failed ({reason}): {e}")
            return False

    def _restart_main_control_loop(self, reason: str = "unspecified") -> None:
        """Start a new main control loop generation and retire old one."""
        if not self.running:
            logger.warning("Skip restarting main control loop because server is not running")
            return

        with self._main_control_loop_lock:
            old_thread = self._main_control_loop_thread
            self._main_control_loop_generation += 1
            generation = self._main_control_loop_generation
            logger.info(
                f"[I2RT main-loop] restart requested reason={reason}, "
                f"new_generation={generation}"
            )
            new_thread = threading.Thread(
                target=self._run_main_control_loop,
                args=(generation,),
                daemon=True,
                name=f"I2RTRobot_{self.device_class_id}_main_control_loop_g{generation}",
            )
            self._main_control_loop_thread = new_thread
            new_thread.start()
            logger.info(
                f"[I2RT main-loop] new thread started: name={new_thread.name}, ident={new_thread.ident}"
            )

        if old_thread is not None and old_thread.is_alive():
            logger.warning(
                "Previous main control loop thread is still alive after restart; "
                "attempting to stop it explicitly"
            )
            old_thread.join(timeout=0.2)
            if old_thread.is_alive():
                killed = self._force_kill_thread(old_thread, reason=f"restart:{reason}")
                old_thread.join(timeout=0.5)
                if old_thread.is_alive():
                    logger.error(
                        f"[I2RT main-loop] old thread still alive after force-kill "
                        f"name={old_thread.name}, ident={old_thread.ident}, killed={killed}"
                    )
                else:
                    logger.info(
                        f"[I2RT main-loop] old thread stopped after force-kill "
                        f"name={old_thread.name}, ident={old_thread.ident}, killed={killed}"
                    )
            else:
                logger.info(
                    f"[I2RT main-loop] old thread exited during graceful wait "
                    f"name={old_thread.name}, ident={old_thread.ident}"
                )

    def start_server(self) -> None:
        try:
            self._i2rt_stopped = False
            logger.info(f" === starting start_server for {self.device_name}_{self.device_class_id} ===")
            logger.info(f"{self.device_name}_{self.device_class_id} Going to home position...")
            if not self._go_home(reason="startup"):
                logger.error(f"Failed to go to home position for {self.device_name}_{self.device_class_id}")
                raise RuntimeError(f"Failed to go to home position for {self.device_name}_{self.device_class_id}")
            self._apply_bilateral_gains_for_mode()
            curr_state = self.get_robot_state()
            if curr_state is not None:
                if self.is_leader_role:
                    self._command_joint_pos(self._left_client, curr_state[:6], "command_joint_pos_left_start_server_leader")
                    self._command_joint_pos(self._right_client, curr_state[7:13], "command_joint_pos_right_start_server_leader")
                else:
                    self._command_joint_pos(self._left_client, curr_state[:7], "command_joint_pos_left_start_server_follower")
                    self._command_joint_pos(self._right_client, curr_state[7:14], "command_joint_pos_right_start_server_follower")
                logger.info(f"Set joint pos after bilateral gains: {curr_state}")
            else:
                logger.error(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")
                raise RuntimeError(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")

            # Start disconnection recovery monitoring thread (follower only)
            logger.info(f"self.is_follower_role:{self.is_follower_role} and self._disconnection_recovery_enabled:{self._disconnection_recovery_enabled} ===")
            self._reset_disconnection_monitor_state()
            self._monitor_disconnection_evt.set()
            logger.info("=== Starting disconnection recovery monitoring thread... ===")
            self._trusted_state_thread = threading.Thread(
                target=self._monitor_trusted_state,
                daemon=True,
                name=f"I2RTRobot_{self.device_class_id}_disconnection_monitor"
            )
            self._trusted_state_thread.start()
            self._watchdog_thread = threading.Thread(
                target=self._monitor_stuck_calls,
                daemon=True,
                name=f"I2RTRobot_{self.device_class_id}_stuck_watchdog"
            )
            self._watchdog_thread.start()
            logger.info("Disconnection recovery monitoring enabled")

            self._prepare_main_control_loop_resources()
            self._restart_main_control_loop(reason="start_server")

            while self.running:
                try:
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break

        except Exception as e:
            logger.error(f"Failed to start {self.device_name}_{self.device_class_id} server: {e}")
            raise e

        finally:
            # Keep shutdown semantics aligned with BaseRobot.start_server:
            # always run stop_server() when start_server exits (including Ctrl+C
            # or external shutdown that flips self.running to False).
            try:
                # IMPORTANT: We should run super().stop_server() to ensure the shared memory is cleaned up before setting the robot home in self.stop_server()
                super().stop_server()
                self.stop_server()
            except Exception as e:
                logger.error(f"Error while stopping {self.device_name}_{self.device_class_id} in start_server finally: {e}")
            logger.info(f"=== Exiting start_server for {self.device_name}_{self.device_class_id} ===")


    def stop_server(self) -> None:
        """Stop robot and terminate locally launched minimum_gello servers.

        Order: _go_rest (needs live RPC servers) -> super cleanup -> terminate servers.
        """
        logger.info(f"=== Stopping {self.device_name}_{self.device_class_id} server... in I2RTRobot self.running={self.running} ===")
        if hasattr(self, '_i2rt_stopped') and self._i2rt_stopped:
            return
        self._monitor_disconnection_evt.clear()
        # Stop disconnection recovery monitoring thread
        if self._trusted_state_thread is not None and self._trusted_state_thread.is_alive():
            logger.info("Stopping disconnection recovery monitoring thread...")
            self._trusted_state_thread.join(timeout=5.0)
            if self._trusted_state_thread.is_alive():
                logger.warning("Disconnection monitoring thread did not stop gracefully")
            self._trusted_state_thread = None
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            logger.info("Stopping disconnection stuck-call watchdog thread...")
            self._watchdog_thread.join(timeout=5.0)
            if self._watchdog_thread.is_alive():
                logger.warning("Disconnection stuck-call watchdog thread did not stop gracefully")
            self._watchdog_thread = None

        self._i2rt_stopped = True

        # stopping main control loop manually
        with self._main_control_loop_lock:
            self._main_control_loop_generation += 1
            control_loop_thread = self._main_control_loop_thread
            self._main_control_loop_thread = None
        if (
            control_loop_thread is not None
            and control_loop_thread.is_alive()
            and control_loop_thread is not threading.current_thread()
        ):
            logger.info("Stopping main control loop thread...")
            control_loop_thread.join(timeout=1.0)
            if control_loop_thread.is_alive():
                logger.warning("Main control loop thread did not stop gracefully")

        try:
            if not self._go_rest():
                logger.error(f"Failed to go to rest position for {self.device_name}_{self.device_class_id}")
        except Exception as e:
            logger.error(f"Exception during go-rest for {self.device_name}_{self.device_class_id}: {e}")
        super().stop_server()
        for proc in self._server_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception as e:
                logger.warning(f"Failed to terminate minimum_gello process: {e}")
        self._server_processes = []

def real_main() -> None:
    """Main function for RizonRobot gripper width calibration."""
    import argparse
    import json
    from datetime import datetime

    from devices.robots.rizon import RizonRobot

    parser = argparse.ArgumentParser(description='Rizon Robot Gripper Calibration')
    parser.add_argument('--device_class_id', type=int, default=0, help='Device ID')
    parser.add_argument('--robot_sn', type=str, default="Rizon4-062931", help='Robot serial number')
    parser.add_argument('--gripper_name', type=str, default="gripper", help='Display name for the gripper')
    parser.add_argument('--tool_name', type=str, default=None, help='Optional Flexiv tool name to switch before calibration')
    parser.add_argument('--gripper_port', type=str, default="/dev/ttyUSB0", help='LK gripper serial port')
    parser.add_argument('--gripper_motor_id', type=int, default=1, help='LK gripper motor ID')
    parser.add_argument('--gripper_baudrate', type=int, default=115200, help='LK gripper baudrate')
    parser.add_argument('--gripper_speed_dps', type=float, default=180.0, help='LK gripper speed limit in deg/s')
    parser.add_argument('--gripper_home_width', type=float, default=0.085, help='Home gripper width in meters')
    parser.add_argument('--gripper_home_ratio', type=float, default=None, help='Optional home opening ratio in [0, 1], overriding gripper_home_width')
    parser.add_argument('--gripper_auto_calibrate', action='store_true', help='Enable auto calibration during init')
    parser.add_argument('--gripper_min_encoder', type=int, default=None, help='Optional minimum encoder value')
    parser.add_argument('--gripper_max_encoder', type=int, default=None, help='Optional maximum encoder value')
    parser.add_argument('--output_path', type=str, default="inference_gripper_width_calibration.json", help='Output JSON file path')

    args = parser.parse_args()

    # Create RizonRobot instance with gripper enabled
    robot = RizonRobot(
        device_class_id=args.device_class_id,
        robot_sn=args.robot_sn,
        state_feedback={
            "fps": 100.0,
            "buffer_size": 100,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [22],
                "type": ["j1-7", "dq1-7", "xyz", "qwqxqyqz", "gripper"],
                "dtype": "float64",
            },
        },
        control={
            "type": "eef_abs",
            "format": {
                "shape": [8],
                "type": ["xyz", "qxqyqzqw", "gripper"],
                "dtype": "float64",
            },
        },
        gripper={
            "backend": "lktech",
            "name": args.gripper_name,
            "tool_name": args.tool_name,
            "port": args.gripper_port,
            "motor_id": args.gripper_motor_id,
            "baudrate": args.gripper_baudrate,
            "speed_dps": args.gripper_speed_dps,
            "home_width": args.gripper_home_width,
            "home_ratio": args.gripper_home_ratio,
            "auto_calibrate": args.gripper_auto_calibrate,
            "min_encoder": args.gripper_min_encoder,
            "max_encoder": args.gripper_max_encoder,
            "value_mode": "width_m",
        },
        signal_files={
            "robot_home_command_pattern": "/tmp/{device_name}_{device_class_id}_home_command",
            "policy_runner_reset_command": "/tmp/policy_runner_reset_command",
            "reset_signal": "/tmp/policy_connector_reset_signal",
        },
    )
    assert not robot.is_dual_arm, "Dual arm robots are not supported for gripper calibration"

    calibration_data = []

    try:
        # Connect to robot and initialize gripper
        if not robot._connect_robot():
            logger.error("Failed to connect to robot")
            return

        logger.info("=" * 50)
        logger.info("Gripper Width Calibration Tool")
        logger.info("=" * 50)
        logger.info("Commands:")
        logger.info("  - Enter a width value (in meters, e.g., 0.05) to move gripper")
        logger.info("  - After gripper moves, enter the measured width")
        logger.info("  - Enter 'q' or 'exit' to quit and save results")
        logger.info("=" * 50)

        while True:
            # Get target width from user
            user_input = input("\nEnter target gripper width (m) or 'q' to quit: ").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                logger.info("Exiting calibration...")
                break

            try:
                target_width = float(user_input)
            except ValueError:
                logger.warning(f"Invalid input: {user_input}. Please enter a number.")
                continue

            # Validate width range
            if target_width < 0 or target_width > 0.1:
                logger.warning(f"Width {target_width} out of range [0, 0.1]. Please try again.")
                continue

            # Move gripper to target width
            logger.info(f"Moving gripper to width: {target_width:.4f} m")
            try:
                robot._move_gripper_value_by_backend(robot.gripper, target_width, 0, "single")
                time.sleep(3.0)  # Wait for gripper to reach position

                # Read current gripper state
                current_width = robot._read_gripper_value_by_backend(robot.gripper, 0, "single")
                logger.info(f"Gripper reported width: {current_width:.4f} m")
            except Exception as e:
                logger.error(f"Failed to move gripper: {e}")
                continue

            # Get measured width from user
            measured_input = input("Enter measured width (m) or 'skip' to skip this point: ").strip()

            if measured_input.lower() == 'skip':
                logger.info("Skipping this data point.")
                continue

            try:
                measured_width = float(measured_input)
            except ValueError:
                logger.warning(f"Invalid measured width: {measured_input}. Skipping this point.")
                continue

            # Record calibration data
            data_point = {
                "target_width": target_width,
                "reported_width": current_width,
                "measured_width": measured_width,
            }
            calibration_data.append(data_point)
            logger.info(f"Recorded: target={target_width:.4f}, reported={current_width:.4f}, measured={measured_width:.4f}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during calibration: {e}")
    finally:
        # Save calibration data to JSON file
        if calibration_data:
            result = {
                "gripper_name": args.gripper_name,
                "calibration_date": datetime.now().isoformat(),
                "data_points": calibration_data
            }

            try:
                with open(args.output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Calibration data saved to: {args.output_path}")
                logger.info(f"Total data points: {len(calibration_data)}")
            except Exception as e:
                logger.error(f"Failed to save calibration data: {e}")

            # Plot calibration curve
            try:
                import matplotlib.pyplot as plt

                target_widths = [d["target_width"] for d in calibration_data]
                measured_widths = [d["measured_width"] for d in calibration_data]

                # Sort data by target width for line plot
                sorted_indices = np.argsort(target_widths)
                target_sorted = np.array(target_widths)[sorted_indices]
                measured_sorted = np.array(measured_widths)[sorted_indices]

                plt.figure(figsize=(10, 6))

                # Plot data points
                plt.scatter(target_widths, measured_widths, c='blue', s=80, label='Data Points', zorder=3)

                # Plot connecting line
                plt.plot(target_sorted, measured_sorted, 'b-', alpha=0.5, linewidth=1.5, label='Measured')

                # Plot ideal line (y = x)
                min_val = min(min(target_widths), min(measured_widths))
                max_val = max(max(target_widths), max(measured_widths))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal (y=x)')

                # Linear fit
                if len(calibration_data) >= 2:
                    coeffs = np.polyfit(target_widths, measured_widths, 1)
                    fit_line = np.poly1d(coeffs)
                    x_fit = np.linspace(min(target_widths), max(target_widths), 100)
                    plt.plot(x_fit, fit_line(x_fit), 'g-', linewidth=2,
                            label=f'Linear Fit: y={coeffs[0]:.4f}x + {coeffs[1]:.4f}')

                plt.xlabel('Target Width (m)', fontsize=12)
                plt.ylabel('Measured Width (m)', fontsize=12)
                plt.title(f'Inference Gripper Calibration: {args.gripper_name}', fontsize=14)
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')

                # Save plot
                plot_path = args.output_path.replace('.json', '_plot.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Calibration plot saved to: {plot_path}")

                # Show plot
                plt.show()

            except Exception as e:
                logger.error(f"Failed to plot calibration data: {e}")
        else:
            logger.info("No calibration data to save.")

        # Cleanup
        try:
            robot.stop_server()
        except:
            pass

        logger.info("Calibration complete.")


if __name__ == "__main__":
    real_main() # calibrate the gripper width
    # sim_main()
