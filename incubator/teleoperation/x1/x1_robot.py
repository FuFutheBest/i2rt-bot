"""X1 dual-arm robot device implementation. Vendor SDK is loaded lazily."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from devices.robots.base import BaseRobot
from utils.logger_config import logger
from utils.shm_utils import get_dtype


class X1Robot(BaseRobot):
    """X1 (TJ / nmxrdk) dual-arm + 2-DoF head robot device.

    Consumes an 18D ``eef_abs`` control vector written by ``X1VuerTeleoperator``:
        [L_xyz(3), L_quat_xyzw(4), L_grip(1),
         R_xyz(3), R_quat_xyzw(4), R_grip(1),
         head_pitch(1), head_yaw(1)]
    (grip in [0, 1]: 0=open, 1=closed; head angles are absolute targets in rad)
    and drives the arms with nmxrdk ``move_line``, the grippers with
    ``set_width`` and the head with ``move_joint``.

    Publishes a 30D state for the teleoperator clutch anchor and downstream
    data collection:
        [L_q(7), L_tcp7_wxyz(7), R_q(7), R_tcp7_wxyz(7), head(2)]
    TCP poses are nmxrdk-native ``[x, y, z, qw, qx, qy, qz]`` (wxyz).

    Hardware control goes through ``incubator/teleoperation/x1/nmx_adapter.py`` so this class never
    touches the raw nmxrdk objects directly.

    Author: Zixi Ying
    """

    # Default init joints / head ported from the incubator teleop_home tool.
    DEFAULT_INIT_LEFT_JOINT = [
        -0.782691, -0.602215, -0.674036, -1.747366, 0.299272, -0.156720, -0.062556
    ]
    DEFAULT_INIT_RIGHT_JOINT = [
        0.868282, -0.262868, 0.663391, -1.677535, -0.119538, -0.355314, -0.103250
    ]
    DEFAULT_INIT_HEAD = [0.60, -0.06]

    def __init__(
        self,
        device_class_id: int = 1,
        *,
        state_feedback: Mapping[str, Any],
        control: Mapping[str, Any],
        robot_id: str = "1e783be2dd870ef5",
        motion_limits: Optional[Mapping[str, Any]] = None,
        gripper_open_m: float = 0.08,
        gripper_close_m: float = 0.0,
        gripper_min_delta_m: float = 0.003,
        head_pitch_range: Tuple[float, float] = (-0.8, 0.95),
        head_yaw_range: Tuple[float, float] = (-1.5, 1.5),
        head_vel: Tuple[float, float] = (1.0, 1.0),
        head_acc: Tuple[float, float] = (2.0, 2.0),
        init_left_joint: Optional[List[float]] = None,
        init_right_joint: Optional[List[float]] = None,
        init_head: Optional[List[float]] = None,
        home_on_start: bool = True,
        home_joint_vel_ratio: float = 10.0,
        home_joint_acc_ratio: float = 10.0,
        home_joint_timeout_ms: int = 15000,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            control=control,
        )

        self.control_type = self.control_config.type
        if self.control_type != "eef_abs":
            raise ValueError("X1Robot supports only control.type=eef_abs")
        self._validate_state_format()
        self._control_layout, self._head_control_slice = self._parse_control_format()
        self.enable_head = self._head_control_slice is not None

        self.robot_id = str(robot_id)
        self.gripper_enabled = tuple(
            layout["gripper"] is not None for layout in self._control_layout
        )
        self.utilize_gripper = any(self.gripper_enabled)
        raw_limits = dict(motion_limits or {})
        unknown_limits = set(raw_limits) - {"velocity"}
        if unknown_limits:
            raise ValueError(
                "X1Robot motion_limits contains unknown fields: "
                + ", ".join(sorted(unknown_limits))
            )
        velocity = raw_limits.get("velocity", {})
        if not isinstance(velocity, Mapping):
            raise ValueError("motion_limits.velocity must be a mapping")
        unknown_velocity = set(velocity) - {"tcp_linear_m_s"}
        if unknown_velocity:
            raise ValueError(
                "X1Robot motion_limits.velocity contains unknown fields: "
                + ", ".join(sorted(unknown_velocity))
            )
        self.move_vel = float(velocity.get("tcp_linear_m_s", 0.3))
        if self.move_vel <= 0.0:
            raise ValueError("motion_limits.velocity.tcp_linear_m_s must be positive")
        self.gripper_open_m = float(gripper_open_m)
        self.gripper_close_m = float(gripper_close_m)
        self.gripper_min_delta_m = float(gripper_min_delta_m)
        self.head_pitch_range = tuple(head_pitch_range)
        self.head_yaw_range = tuple(head_yaw_range)
        self.head_vel = [float(v) for v in head_vel]
        self.head_acc = [float(v) for v in head_acc]
        self.init_left_joint = list(init_left_joint) if init_left_joint else list(self.DEFAULT_INIT_LEFT_JOINT)
        self.init_right_joint = list(init_right_joint) if init_right_joint else list(self.DEFAULT_INIT_RIGHT_JOINT)
        self.init_head = list(init_head) if init_head else list(self.DEFAULT_INIT_HEAD)
        self.home_on_start = bool(home_on_start)
        self.home_joint_vel_ratio = float(home_joint_vel_ratio)
        self.home_joint_acc_ratio = float(home_joint_acc_ratio)
        self.home_joint_timeout_ms = int(home_joint_timeout_ms)

        self._operational = False
        self.adapter: Optional[Any] = None
        self.arm_left: Optional[Any] = None
        self.arm_right: Optional[Any] = None
        self.grip_left: Optional[Any] = None
        self.grip_right: Optional[Any] = None
        self.head_handle: Optional[Any] = None
        self._last_grip = {"left": None, "right": None}

        self._connect_hardware()

        logger.info(
            f"X1Robot initialized: robot_id={self.robot_id}, "
            f"left=enabled, right=enabled, head={self.enable_head}, "
            f"gripper={self.utilize_gripper}, state SHM={self.shared_memory_name}"
        )

    def _validate_state_format(self) -> None:
        primitive_names = tuple(
            primitive.name
            for primitive in self.state_feedback_config.format.primitives
        )
        expected = (
            "j1-7",
            "xyz",
            "qwqxqyqz",
            "j1-7",
            "xyz",
            "qwqxqyqz",
            "j1-2",
        )
        if primitive_names != expected:
            raise ValueError(
                "X1Robot state_feedback.format.type must be "
                "[j1-7, xyz, qwqxqyqz, j1-7, xyz, qwqxqyqz, j1-2]"
            )

    def _parse_control_format(
        self,
    ) -> tuple[tuple[dict[str, Any], ...], slice | None]:
        primitives = self.control_config.format.primitives
        primitive_index = 0
        value_offset = 0
        arm_layouts: list[dict[str, Any]] = []

        def consume(expected_names: set[str], description: str):
            nonlocal primitive_index, value_offset
            if primitive_index >= len(primitives):
                raise ValueError(
                    f"X1Robot control.format.type is missing {description}"
                )
            primitive = primitives[primitive_index]
            if primitive.name not in expected_names:
                raise ValueError(
                    f"X1Robot control.format.type expected {description}, "
                    f"got {primitive.name}"
                )
            primitive_slice = slice(value_offset, value_offset + primitive.width)
            primitive_index += 1
            value_offset += primitive.width
            return primitive, primitive_slice

        for side in ("left", "right"):
            _, position_slice = consume({"xyz"}, f"{side} xyz")
            quaternion, quaternion_slice = consume(
                {"qwqxqyqz", "qxqyqzqw"}, f"{side} quaternion"
            )
            gripper_slice = None
            if (
                primitive_index < len(primitives)
                and primitives[primitive_index].name == "gripper"
            ):
                _, gripper_slice = consume({"gripper"}, f"{side} gripper")
            arm_layouts.append(
                {
                    "position": position_slice,
                    "quaternion": quaternion_slice,
                    "quaternion_order": quaternion.quaternion_order,
                    "gripper": gripper_slice,
                }
            )

        head_slice = None
        if primitive_index < len(primitives):
            _, head_slice = consume({"j1-2"}, "head j1-2")
        if primitive_index != len(primitives):
            remaining = [
                primitive.name for primitive in primitives[primitive_index:]
            ]
            raise ValueError(
                "X1Robot control.format.type contains unsupported trailing "
                f"primitives: {remaining}"
            )
        if value_offset != self.control_config.format.dimension:
            raise ValueError("X1Robot control format layout is internally inconsistent")
        return tuple(arm_layouts), head_slice

    # ------------------------------------------------------------------
    # Hardware connection / homing
    # ------------------------------------------------------------------
    def _connect_hardware(self) -> None:
        """Connect to the nmxrdk RobotClient and enable both arm actuators."""
        from incubator.teleoperation.x1.nmx_adapter import NmxAdapter

        self.adapter = NmxAdapter(self.robot_id)
        self.arm_left = self.adapter.arm("left")
        self.arm_left.enable()
        if self.gripper_enabled[0]:
            self.grip_left = self.adapter.gripper("left")
        self.arm_right = self.adapter.arm("right")
        self.arm_right.enable()
        if self.gripper_enabled[1]:
            self.grip_right = self.adapter.gripper("right")
        if self.enable_head:
            self.head_handle = self.adapter.head()
            self.head_handle.enable()

    def _home(self) -> None:
        """Move arms (joint space) and head to the configured init pose."""
        logger.info("X1Robot: homing to init pose ...")
        if self.arm_left is not None:
            self.arm_left.move_joint(
                self.init_left_joint, self.home_joint_vel_ratio,
                self.home_joint_acc_ratio, True, self.home_joint_timeout_ms,
            )
        if self.arm_right is not None:
            self.arm_right.move_joint(
                self.init_right_joint, self.home_joint_vel_ratio,
                self.home_joint_acc_ratio, True, self.home_joint_timeout_ms,
            )
        if self.enable_head and self.head_handle is not None:
            self.head_handle.move_joint(
                self.init_head, self.head_vel, self.head_acc, True, 5000,
            )
        logger.info("X1Robot: homing done")

    @property
    def is_operational(self) -> bool:
        return self._operational

    def start_server(self) -> None:
        if self.home_on_start:
            try:
                self._home()
            except Exception as exc:
                logger.error(f"X1Robot homing failed: {exc}")
        self._operational = True
        super().start_server()

    # ------------------------------------------------------------------
    # State (published to state SHM)
    # ------------------------------------------------------------------
    def get_robot_state(self) -> np.ndarray:
        """Return 30D state [L_q, L_tcp7, R_q, R_tcp7, head(2)]."""
        state = np.zeros(self.data_shape, dtype=get_dtype(self.data_dtype)).reshape(-1)
        try:
            if self.arm_left is not None:
                state[0:7] = self.arm_left.get_joint_pos()
                state[7:14] = self.arm_left.get_tcp_pose7()
            if self.arm_right is not None:
                state[14:21] = self.arm_right.get_joint_pos()
                state[21:28] = self.arm_right.get_tcp_pose7()
            if self.head_handle is not None:
                head_j = self.head_handle.get_joint_pos()
                state[28:30] = head_j[:2]
        except Exception as exc:
            logger.error(f"X1Robot get_robot_state error: {exc}")
        return state.reshape(self.data_shape)

    # ------------------------------------------------------------------
    # Control (consumed from teleoperator data SHM)
    # ------------------------------------------------------------------
    @staticmethod
    def _pose_to_wxyz(
        values: np.ndarray, layout: Mapping[str, Any]
    ) -> List[float]:
        position = values[layout["position"]]
        quaternion = values[layout["quaternion"]]
        if layout["quaternion_order"] == "xyzw":
            quaternion = quaternion[[3, 0, 1, 2]]
        return [*position.tolist(), *quaternion.tolist()]

    def _decode_control_array(
        self, control_array: np.ndarray
    ) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        values = np.asarray(control_array, dtype=float).reshape(-1)
        expected_size = self.control_config.format.dimension
        if values.size != expected_size:
            raise ValueError(
                f"X1Robot expected {expected_size} control values from "
                f"control.format.shape, got {values.size}"
            )
        arm_commands: list[dict[str, Any]] = []
        for layout in self._control_layout:
            gripper_slice = layout["gripper"]
            arm_commands.append(
                {
                    "pose_wxyz": self._pose_to_wxyz(values, layout),
                    "gripper": (
                        None
                        if gripper_slice is None
                        else float(values[gripper_slice][0])
                    ),
                }
            )
        head = (
            None
            if self._head_control_slice is None
            else values[self._head_control_slice]
        )
        return arm_commands, head

    def _select_control_array(self, control_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Return the endpoint selected by the typed source contract."""
        selected_name = self._selected_source_name()
        if selected_name is None:
            return None
        return control_dict.get(selected_name)

    def _command_gripper(self, gripper: Any, grip_norm: float, side: str) -> None:
        """grip_norm in [0,1]: 0=open, 1=closed -> width in meters (rate-limited)."""
        grip_norm = float(np.clip(grip_norm, 0.0, 1.0))
        width = self.gripper_open_m - grip_norm * (self.gripper_open_m - self.gripper_close_m)
        last = self._last_grip[side]
        if last is None or abs(width - last) > self.gripper_min_delta_m:
            gripper.set_width(width)
            self._last_grip[side] = width

    def _command_head(self, pitch: float, yaw: float) -> None:
        """Drive head to absolute [pitch, yaw] (rad), clamped to safety ranges."""
        j0 = float(np.clip(pitch, self.head_pitch_range[0], self.head_pitch_range[1]))
        j1 = float(np.clip(yaw, self.head_yaw_range[0], self.head_yaw_range[1]))
        self.head_handle.move_joint([j0, j1], self.head_vel, self.head_acc, False, 1000)

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """Execute one command described by ``control.format``."""
        try:
            if not self.is_operational:
                return
            control_array = self._select_control_array(control_dict)
            if control_array is None:
                return
            arm_commands, head_command = self._decode_control_array(control_array)

            if self.arm_left is not None:
                self.arm_left.move_line(
                    arm_commands[0]["pose_wxyz"], self.move_vel, False
                )
                if self.grip_left is not None and arm_commands[0]["gripper"] is not None:
                    self._command_gripper(
                        self.grip_left, arm_commands[0]["gripper"], "left"
                    )
            if self.arm_right is not None:
                self.arm_right.move_line(
                    arm_commands[1]["pose_wxyz"], self.move_vel, False
                )
                if self.grip_right is not None and arm_commands[1]["gripper"] is not None:
                    self._command_gripper(
                        self.grip_right, arm_commands[1]["gripper"], "right"
                    )
            if (
                self.enable_head
                and self.head_handle is not None
                and head_command is not None
            ):
                self._command_head(head_command[0], head_command[1])
        except Exception as exc:
            logger.error(f"X1Robot execute_control_command error: {exc}")

    def stop_server(self) -> None:
        self._operational = False
        if self.enable_head and self.head_handle is not None:
            try:
                self.head_handle.stop()
                self.head_handle.disable()
            except Exception as exc:
                logger.warning(f"X1Robot head stop/disable failed: {exc}")
        super().stop_server()
