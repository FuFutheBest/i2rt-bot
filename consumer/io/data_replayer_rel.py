"""
Data Replayer - Replays data from a file to the policy SHM.

Author: Han Xue, Zheng Wang, Jun Lv
"""

import traceback
import h5py
import time
import yaml
import argparse
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Union, Deque
from collections import deque
import scipy.spatial.transform as st
from consumer.inference.common.policy_connector import PolicyConnector
from utils.shm_utils import (
    DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    pack_buffer_header, pack_frame_header, get_dtype, POLICY_HEADER_SIZE,
    unpack_policy_header, get_device_info_offset, unpack_device_header,
    pack_policy_header, get_policy_data_offset, connect_to_policy_shm
)
from utils.logger_config import logger
ActionType = Union[np.ndarray, Dict[str, np.ndarray]]
from utils.rerun_visualization import (
    RerunVisualizer, visualize_trajectory_with_rotation, gripper_width_to_color,
    visualize_action_time_series, log_text_summary, set_time_context
)
import threading
import tkinter as tk
from tkinter import ttk
import rerun as rr

def pos_quat_to_mat(pos, quat):
    """pos: np.ndarray(3,)
    quat: np.ndarray(4,), xyzw
    """
    res = np.eye(4)
    # quat is expected as [x, y, z, w]
    res[:3, :3] = st.Rotation.from_quat(quat).as_matrix()
    res[:3, 3] = pos
    return res

def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    quat: (..., 4)  [x, y, z, w]
    return (..., 3) axis-angle (OpenCV convention, norm = angle)
    """
    rot = st.Rotation.from_quat(quat[..., [0, 1, 2, 3]])  # scipy: [x,y,z,w]
    return rot.as_rotvec()

def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """
    axis_angle: (..., 3)
    return (..., 4) quat
    """
    rot = st.Rotation.from_rotvec(axis_angle)
    return rot.as_quat()

def load_transform_from_json(json_path: str) -> np.ndarray:
    """
    Load transformation matrix from JSON file.
    
    Args:
        json_path: Path to the JSON file containing the 4x4 transformation matrix
        
    Returns:
        4x4 numpy array representing the transformation matrix
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the JSON file doesn't contain a valid 4x4 matrix
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Transform JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert to numpy array
        transform_matrix = np.array(data, dtype=np.float64)
        
        # Validate shape
        if transform_matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {transform_matrix.shape}")
        
        logger.info(f"Successfully loaded transform matrix from {json_path}")
        return transform_matrix
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading transform matrix from {json_path}: {e}")

"""
Calibration transform between Vive tracker and TCP:
- vive_tcp_T: T_V_T, transform from Vive tracker frame to TCP frame
- tcp_vive_T: T_T_V, inverse of vive_tcp_T

A calibration GUI below allows interactive adjustment of vive_tcp_T using
translation (x,y,z) and extrinsic roll-pitch-yaw (sxyz, global XYZ) angles. The replay
pipeline (extract_traj) consumes vive_tcp_T to convert T_W_V -> T_W_T.

The transformation matrices are loaded from JSON file at startup.
"""
# Default transformation matrix (will be loaded from JSON file)
tcp_vive_T = np.array([ [  0.0000  , -0.9397 ,  0.3420,  -0.0317],
                        [1.0000 ,  0.0000  , 0.0000  , 0.0000],
                        [0.0000  , 0.3420 ,  0.9397 , -0.272],
                        [0.0000  , 0.0000  , 0.0000  , 1.0000]])
vive_tcp_T = np.linalg.inv(tcp_vive_T)


def set_vive_tcp_T(vive_tcp: np.ndarray) -> None:
    """Set global vive_tcp_T and its inverse tcp_vive_T.

    Args:
        vive_tcp: 4x4 homogeneous transform from Vive to TCP.
    """
    global vive_tcp_T, tcp_vive_T
    vive_tcp_T = vive_tcp.copy()
    tcp_vive_T = np.linalg.inv(vive_tcp_T)

def initialize_transform_from_json(json_path: str) -> None:
    """Initialize global transformation matrices from JSON file.

    Args:
        json_path: Path to the JSON file containing the 4x4 transformation matrix.
                  This should be tcp_vive_T (transform from TCP to Vive).
    """
    global tcp_vive_T, vive_tcp_T
    try:
        tcp_vive_T = load_transform_from_json(json_path)
        vive_tcp_T = np.linalg.inv(tcp_vive_T)
        logger.info(f"Initialized transformation matrices from {json_path}")
        logger.info(f"tcp_vive_T:\n{tcp_vive_T}")
        logger.info(f"vive_tcp_T:\n{vive_tcp_T}")
    except Exception as e:
        logger.error(f"Failed to initialize transform from {json_path}: {e}")
        logger.warning("Using default transformation matrices")
        # Keep the default matrices already set


class CalibrationGUI:
    """Interactive 6DoF calibration GUI to edit vive_tcp_T in real time."""

    def __init__(self) -> None:
        # Parameters (meters, radians)
        # Initialize from current global vive_tcp_T matrix
        global vive_tcp_T
        current_matrix = vive_tcp_T.copy()
        
        # Extract position from current matrix
        self.x: float = current_matrix[0, 3]
        self.y: float = current_matrix[1, 3] 
        self.z: float = current_matrix[2, 3]
        
        # Extract rotation from current matrix (quaternion, xyzw) for arcball control
        try:
            rot = st.Rotation.from_matrix(current_matrix[:3, :3])
            self.quat: np.ndarray = rot.as_quat()  # [x,y,z,w]
        except Exception:
            self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        # Also keep euler values for display only (sxyz extrinsic)
        from transforms3d.euler import mat2euler
        try:
            euler_angles = mat2euler(current_matrix[:3, :3], axes='sxyz')
            self.roll: float = float(euler_angles[0])
            self.pitch: float = float(euler_angles[1])
            self.yaw: float = float(euler_angles[2])
        except Exception:
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0

        self.position_range = (-0.20, 0.20)
        self.rotation_range_deg = (-180.0, 180.0)

        self._root: Optional[tk.Tk] = None  # type: ignore[name-defined]
        self._dirty: bool = True
        self._lock = threading.Lock()

        # Tk variables
        self._x_var = None
        self._y_var = None
        self._z_var = None
        # Arcball state
        self._arc_canvas: Optional[tk.Canvas] = None  # type: ignore[name-defined]
        self._arc_center: Optional[tuple] = None
        self._arc_radius: Optional[float] = None
        self._dragging: bool = False
        self._v0: Optional[np.ndarray] = None

        # Value label variables
        self._x_val_var = None
        self._y_val_var = None
        self._z_val_var = None
        self._roll_val_var = None
        self._pitch_val_var = None
        self._yaw_val_var = None

    def _build_matrix(self) -> np.ndarray:
        """Build 4x4 T_V_T from current params using quaternion (no gimbal lock)."""
        with self._lock:
            R = st.Rotation.from_quat(self.quat).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array([self.x, self.y, self.z])
        return T

    def get_current_matrix(self) -> np.ndarray:
        return self._build_matrix()

    def mark_clean(self) -> None:
        with self._lock:
            self._dirty = False

    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def _on_change(self, *_args: Any) -> None:  # type: ignore[name-defined]
        try:
            with self._lock:
                self.x = float(self._x_var.get())
                self.y = float(self._y_var.get())
                self.z = float(self._z_var.get())
                self._dirty = True

            # Update globals immediately for downstream usage
            set_vive_tcp_T(self._build_matrix())

            # Update on-screen value labels
            self._update_value_labels()

            # Log current matrix for visibility
            T = self.get_current_matrix()
            logger.info(f"GUI Updated vive_tcp_T (V->T):\n{T}")
            logger.debug(f"GUI parameters: x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, "
                        f"roll={np.degrees(self.roll):.2f}°, pitch={np.degrees(self.pitch):.2f}°, yaw={np.degrees(self.yaw):.2f}°")
        except Exception as e:
            logger.error(f"CalibrationGUI on_change error: {e}")

    def _setup_gui(self) -> None:
        self._root = tk.Tk()
        self._root.title("Vive->TCP Calibration (x,y,z,r,p,y)")
        self._root.geometry("520x420")

        frame = ttk.Frame(self._root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Let the scale column expand
        frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(frame, text="Position (m)", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        self._x_var = tk.DoubleVar(value=self.x)
        ttk.Label(frame, text="x").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(frame, from_=self.position_range[0], to=self.position_range[1], variable=self._x_var, orient=tk.HORIZONTAL, length=250, command=self._on_change).grid(row=row, column=1, sticky=(tk.W, tk.E))
        self._x_val_var = tk.StringVar(value=f"{self.x:.3f} m")
        ttk.Label(frame, textvariable=self._x_val_var, width=12).grid(row=row, column=2, sticky=tk.W, padx=(10,0))
        row += 1

        self._y_var = tk.DoubleVar(value=self.y)
        ttk.Label(frame, text="y").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(frame, from_=self.position_range[0], to=self.position_range[1], variable=self._y_var, orient=tk.HORIZONTAL, length=250, command=self._on_change).grid(row=row, column=1, sticky=(tk.W, tk.E))
        self._y_val_var = tk.StringVar(value=f"{self.y:.3f} m")
        ttk.Label(frame, textvariable=self._y_val_var, width=12).grid(row=row, column=2, sticky=tk.W, padx=(10,0))
        row += 1

        self._z_var = tk.DoubleVar(value=self.z)
        ttk.Label(frame, text="z").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(frame, from_=self.position_range[0], to=self.position_range[1], variable=self._z_var, orient=tk.HORIZONTAL, length=250, command=self._on_change).grid(row=row, column=1, sticky=(tk.W, tk.E))
        self._z_val_var = tk.StringVar(value=f"{self.z:.3f} m")
        ttk.Label(frame, textvariable=self._z_val_var, width=12).grid(row=row, column=2, sticky=tk.W, padx=(10,0))
        row += 1

        ttk.Label(frame, text="Rotation (arcball)", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,0))
        row += 1
        # Arcball canvas
        self._arc_canvas = tk.Canvas(frame, width=240, height=240, background="#111111", highlightthickness=0)
        self._arc_canvas.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        # Compute center and radius
        self._arc_center = (120, 120)
        self._arc_radius = 100.0
        # Draw circle outline
        self._draw_arcball()
        # Bind mouse events
        self._arc_canvas.bind("<ButtonPress-1>", self._on_arc_down)
        self._arc_canvas.bind("<B1-Motion>", self._on_arc_move)
        self._arc_canvas.bind("<ButtonRelease-1>", self._on_arc_up)
        row += 1

        # Rotation value labels (display only)
        self._roll_val_var = tk.StringVar(value=f"{np.degrees(self.roll):.1f} °")
        self._pitch_val_var = tk.StringVar(value=f"{np.degrees(self.pitch):.1f} °")
        self._yaw_val_var = tk.StringVar(value=f"{np.degrees(self.yaw):.1f} °")
        vals = ttk.Frame(frame)
        vals.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(4,0))
        ttk.Label(vals, text="roll (x):").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(vals, textvariable=self._roll_val_var, width=10).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(vals, text="pitch (y):").grid(row=0, column=2, sticky=tk.W)
        ttk.Label(vals, textvariable=self._pitch_val_var, width=10).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(vals, text="yaw (z):").grid(row=0, column=4, sticky=tk.W)
        ttk.Label(vals, textvariable=self._yaw_val_var, width=10).grid(row=0, column=5, sticky=tk.W)
        row += 1

        # Buttons
        btns = ttk.Frame(frame)
        btns.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btns, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Copy Matrix", command=self._copy_matrix).pack(side=tk.LEFT, padx=5)

        # Initialize globals on launch
        self._on_change()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_value_labels(self) -> None:
        try:
            if self._x_val_var is not None:
                self._x_val_var.set(f"{self.x:.3f} m")
            if self._y_val_var is not None:
                self._y_val_var.set(f"{self.y:.3f} m")
            if self._z_val_var is not None:
                self._z_val_var.set(f"{self.z:.3f} m")
            # Update rpy from current quaternion for display
            try:
                Rm = st.Rotation.from_quat(self.quat).as_matrix()
                from transforms3d.euler import mat2euler
                rpy = mat2euler(Rm, axes='sxyz')
                self.roll, self.pitch, self.yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
            except Exception:
                pass
            if self._roll_val_var is not None:
                self._roll_val_var.set(f"{np.degrees(self.roll):.1f} °")
            if self._pitch_val_var is not None:
                self._pitch_val_var.set(f"{np.degrees(self.pitch):.1f} °")
            if self._yaw_val_var is not None:
                self._yaw_val_var.set(f"{np.degrees(self.yaw):.1f} °")
        except Exception as e:
            logger.error(f"CalibrationGUI value label update error: {e}")

    def _reset(self) -> None:
        try:
            self._x_var.set(0.0)
            self._y_var.set(0.0)
            self._z_var.set(0.0)
            with self._lock:
                self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
                self._dirty = True
            self._on_change()
            # Redraw arcball for cleanliness
            try:
                self._draw_arcball()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"CalibrationGUI reset error: {e}")

    # ---------- Arcball helpers ----------
    def _draw_arcball(self) -> None:
        try:
            if self._arc_canvas is None or self._arc_center is None or self._arc_radius is None:
                return
            self._arc_canvas.delete("all")
            cx, cy = self._arc_center
            r = self._arc_radius
            self._arc_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#666666", width=2)
        except Exception as e:
            logger.error(f"Arcball draw error: {e}")

    def _map_to_sphere(self, x: float, y: float) -> np.ndarray:
        if self._arc_center is None or self._arc_radius is None:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        cx, cy = self._arc_center
        r = self._arc_radius
        # Map to [-1, 1] with y up
        nx = (x - cx) / r
        ny = (cy - y) / r
        d = nx * nx + ny * ny
        if d > 1.0:
            inv_len = 1.0 / np.sqrt(d)
            return np.array([nx * inv_len, ny * inv_len, 0.0], dtype=float)
        return np.array([nx, ny, np.sqrt(1.0 - d)], dtype=float)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product (xyzw)."""
        x1, y1, z1, w1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
        x2, y2, z2, w2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return np.array([x, y, z, w], dtype=float)

    def _on_arc_down(self, event: Any) -> None:  # type: ignore[name-defined]
        try:
            self._dragging = True
            self._v0 = self._map_to_sphere(float(event.x), float(event.y))
        except Exception as e:
            logger.error(f"Arcball down error: {e}")

    def _on_arc_move(self, event: Any) -> None:  # type: ignore[name-defined]
        try:
            if not self._dragging or self._v0 is None:
                return
            v1 = self._map_to_sphere(float(event.x), float(event.y))
            v0 = self._v0
            dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
            if dot >= 0.999999:
                return
            axis = np.cross(v0, v1)
            axis_norm = np.linalg.norm(axis) + 1e-12
            axis_unit = axis / axis_norm
            angle = np.arccos(dot)
            half = 0.5 * angle
            s = np.sin(half)
            q_inc = np.array([axis_unit[0] * s, axis_unit[1] * s, axis_unit[2] * s, np.cos(half)], dtype=float)
            with self._lock:
                # World-space arcball: pre-multiply incremental rotation
                self.quat = self._quat_multiply(q_inc, self.quat)
                self.quat = self.quat / (np.linalg.norm(self.quat) + 1e-12)
                self._dirty = True
            set_vive_tcp_T(self._build_matrix())
            self._update_value_labels()
            self._v0 = v1
        except Exception as e:
            logger.error(f"Arcball move error: {e}")

    def _on_arc_up(self, _event: Any) -> None:  # type: ignore[name-defined]
        self._dragging = False
        self._v0 = None

    def _copy_matrix(self) -> None:
        try:
            T = self.get_current_matrix()
            matrix_str = "np.array([\n" + \
                "\n".join(["    [" + ", ".join([f"{T[i, j]:8.4f}" for j in range(4)]) + "]" + ("," if i < 3 else "") for i in range(4)]) + \
                "\n])"
            if self._root is not None:
                self._root.clipboard_clear()
                self._root.clipboard_append(matrix_str)
                logger.info("vive_tcp_T copied to clipboard")
        except Exception as e:
            logger.error(f"CalibrationGUI copy error: {e}")

    def _on_close(self) -> None:
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        try:
            self._setup_gui()
            if self._root is not None:
                self._root.mainloop()
        except Exception as e:
            logger.error(f"CalibrationGUI run error: {e}")


def action_to_dual_pose(umi_action):
    left_pos, left_axis_angle, left_gripper = umi_action[:, :3], umi_action[:, 3:6], umi_action[:, 6:6 + 1]
    right_pos, right_axis_angle, right_gripper = umi_action[:, 7:10], umi_action[:, 10:13], umi_action[:, 13:13 + 1]
    left_quat = axis_angle_to_quaternion(left_axis_angle)
    left_pose = np.concatenate([left_pos, left_quat, left_gripper], axis=-1)
    right_quat = axis_angle_to_quaternion(right_axis_angle)
    right_pose = np.concatenate([right_pos, right_quat, right_gripper], axis=-1)
    dual_pose = np.concatenate([left_pose, right_pose], axis=-1)
    return dual_pose

def extract_traj_relative(file_path: str, initial_robot_obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Computes the target absolute TCP poses for the robot to replay a recorded
    trajectory from a Vive tracker. The replay is based on matching the
    relative motion of the TCP.

    Args:
        file_path: Path to the HDF5 file containing the recorded Vive tracker data.
        initial_robot_obs: A dictionary containing the initial observation from
                           the robot, including the TCP poses.

    Returns:
        A numpy array of shape (N, 16) containing the target absolute poses
        for a dual-arm robot, where N is the number of frames in the trajectory.
        Each pose is [left_pos, left_quat, left_gripper, right_pos, right_quat, right_gripper].
    """
    # Declare global variables to access updated calibration values
    global vive_tcp_T, tcp_vive_T
    with h5py.File(file_path, "r") as f:
        # Step 1: Load Vive Tracker poses for both arms and convert to 4x4 matrices.
        # T_Wv_V(t): Pose of Vive tracker {V} in Vive world {Wv} at time t.
        vive_data = f['ViveTrackerDevice_0'][()]
        # Expect shape (N, 2, 7) with [pos(3), quat_xyzw(4)] for [left, right]
        if vive_data.ndim != 3 or vive_data.shape[1] < 2 or vive_data.shape[2] != 7:
            raise ValueError(f"Unexpected ViveTrackerDevice_0 shape: {vive_data.shape}, expect (N, 2, 7)")

        left_vive_pose_data = vive_data[:, 0, :]
        right_vive_pose_data = vive_data[:, 1, :]

        left_vive_pose_mats = np.array([pos_quat_to_mat(p[:3], p[3:]) for p in left_vive_pose_data])
        right_vive_pose_mats = np.array([pos_quat_to_mat(p[:3], p[3:]) for p in right_vive_pose_data])

        # Visualize raw Vive tracker poses (left/right) with different colors
        try:
            rr.log("trajectory/world/vive_poses", rr.Clear(recursive=True))

            # Left Vive poses
            left_positions_vive = left_vive_pose_mats[:, :3, 3]
            left_quats_xyzw_vive = np.array([st.Rotation.from_matrix(m[:3, :3]).as_quat() for m in left_vive_pose_mats])
            left_colors_vive = [[255, 180, 50] for _ in range(len(left_positions_vive))]  # orange
            visualize_trajectory_with_rotation(
                "trajectory/world/vive_poses/left_tracker",
                left_positions_vive,
                left_quats_xyzw_vive,
                left_colors_vive,
                arrow_scale=0.02,
                rotation_scale=0.03,
                show_every_n=6,
            )

            # Right Vive poses
            right_positions_vive = right_vive_pose_mats[:, :3, 3]
            right_quats_xyzw_vive = np.array([st.Rotation.from_matrix(m[:3, :3]).as_quat() for m in right_vive_pose_mats])
            right_colors_vive = [[100, 220, 120] for _ in range(len(right_positions_vive))]  # green
            visualize_trajectory_with_rotation(
                "trajectory/world/vive_poses/right_tracker",
                right_positions_vive,
                right_quats_xyzw_vive,
                right_colors_vive,
                arrow_scale=0.02,
                rotation_scale=0.03,
                show_every_n=6,
            )

            log_text_summary(
                "trajectory/world/vive_poses/info",
                "Vive tracker poses visualized (left/right)",
            )
        except Exception as e:
            logger.error(f"Failed to visualize Vive poses: {e}")

        # Step 2: Calculate the absolute TCP trajectory during recording for each arm.
        # T_Wv_TCP_rec(t) = T_Wv_V(t) @ T_V_TCP_rec
        # vive_tcp_T is T_V_TCP_rec (from vive tracker frame to TCP frame)
        left_tcp_pose_mats_record = left_vive_pose_mats @ vive_tcp_T
        right_tcp_pose_mats_record = right_vive_pose_mats @ vive_tcp_T

        # Visualize recorded TCP trajectories for left and right arms
        try:
            # Clear previous recorded visualization
            rr.log("trajectory/world/recorded_tcp", rr.Clear(recursive=True))

            # Left arm recorded trajectory
            left_positions_rec = left_tcp_pose_mats_record[:, :3, 3]
            left_quats_xyzw_rec = np.array([st.Rotation.from_matrix(m[:3, :3]).as_quat() for m in left_tcp_pose_mats_record])
            left_colors_rec = [[60, 180, 255] for _ in range(len(left_positions_rec))]  # cyan-ish
            visualize_trajectory_with_rotation(
                "trajectory/world/recorded_tcp/left_arm",
                left_positions_rec,
                left_quats_xyzw_rec,
                left_colors_rec,
                arrow_scale=0.02,
                rotation_scale=0.03,
                show_every_n=6,
            )

            # Right arm recorded trajectory
            right_positions_rec = right_tcp_pose_mats_record[:, :3, 3]
            right_quats_xyzw_rec = np.array([st.Rotation.from_matrix(m[:3, :3]).as_quat() for m in right_tcp_pose_mats_record])
            right_colors_rec = [[255, 100, 160] for _ in range(len(right_positions_rec))]  # magenta-ish
            visualize_trajectory_with_rotation(
                "trajectory/world/recorded_tcp/right_arm",
                right_positions_rec,
                right_quats_xyzw_rec,
                right_colors_rec,
                arrow_scale=0.02,
                rotation_scale=0.03,
                show_every_n=6,
            )

            log_text_summary(
                "trajectory/world/recorded_tcp/info",
                "Recorded TCP trajectories visualized (left/right)",
            )
        except Exception as e:
            logger.error(f"Failed to visualize recorded TCP trajectories: {e}")

        # Step 3: Calculate the relative TCP motion during recording for each arm.
        # T_rel_rec(t) = inv(T_Wv_TCP_rec(0)) @ T_Wv_TCP_rec(t)
        left_initial_inv = np.linalg.inv(left_tcp_pose_mats_record[0])
        right_initial_inv = np.linalg.inv(right_tcp_pose_mats_record[0])
        left_relative_tcp_pose_mats = left_initial_inv @ left_tcp_pose_mats_record
        right_relative_tcp_pose_mats = right_initial_inv @ right_tcp_pose_mats_record

        # Step 4: Get initial robot TCP pose from observation for both arms.
        initial_robot_poses_mat = get_initial_robot_poses(initial_robot_obs)
        initial_left_tcp_mat = initial_robot_poses_mat['left']
        initial_right_tcp_mat = initial_robot_poses_mat['right']

        # Step 5: Compute target absolute TCP poses for the robot, each following its own relative motion.
        # T_Wr_TCP_robot(t) = T_Wr_TCP_robot(0) @ T_rel_rec(t)
        # Align lengths if needed
        num_frames = min(len(left_relative_tcp_pose_mats), len(right_relative_tcp_pose_mats))
        left_relative_tcp_pose_mats = left_relative_tcp_pose_mats[:num_frames]
        right_relative_tcp_pose_mats = right_relative_tcp_pose_mats[:num_frames]

        target_left_tcp_mats = initial_left_tcp_mat @ left_relative_tcp_pose_mats
        target_right_tcp_mats = initial_right_tcp_mat @ right_relative_tcp_pose_mats

        # Step 6: Convert matrices to the required [pos, quat, gripper] format.
        # Gripper data from recording for left and right arms separately.
        def _prepare_gripper_widths(arr: np.ndarray, frames: int) -> np.ndarray:
            vals = np.array(arr, dtype=float) / 1000.0
            vals = np.clip(vals, 0.0, 0.085)
            if vals.ndim == 1:
                vals = vals.reshape(-1, 1)
            elif vals.ndim >= 2 and vals.shape[1] != 1:
                vals = vals[:, :1]
            if vals.shape[0] < frames:
                vals = np.pad(vals, ((0, frames - vals.shape[0]), (0, 0)), mode='edge')
            elif vals.shape[0] > frames:
                vals = vals[:frames]
            return vals

        try:
            left_gripper_widths = _prepare_gripper_widths(f['RotaryEncoderDevice_0'][()], num_frames)
        except Exception:
            logger.warning("Left gripper encoder 'RotaryEncoderDevice_0' not found; using zeros")
            left_gripper_widths = np.zeros((num_frames, 1), dtype=float)

        try:
            right_gripper_widths = _prepare_gripper_widths(f['RotaryEncoderDevice_1'][()], num_frames)
        except Exception:
            logger.warning("Right gripper encoder 'RotaryEncoderDevice_1' not found; falling back to left widths")
            right_gripper_widths = left_gripper_widths.copy()

        # Convert left arm poses
        left_positions = target_left_tcp_mats[:, :3, 3]
        left_quats_xyzw = np.array([st.Rotation.from_matrix(m[:3, :3]).as_quat() for m in target_left_tcp_mats])
        left_poses = np.concatenate([left_positions, left_quats_xyzw, left_gripper_widths], axis=1)

        # Convert right arm poses
        right_positions = target_right_tcp_mats[:, :3, 3]
        right_quats_xyzw = np.array([st.Rotation.from_matrix(m[:3, :3]).as_quat() for m in target_right_tcp_mats])
        right_poses = np.concatenate([right_positions, right_quats_xyzw, right_gripper_widths], axis=1)

        # Combine into a single dual-arm action array.
        dual_poses = np.concatenate([left_poses, right_poses], axis=1)

    return dual_poses

def get_initial_robot_poses(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extracts the initial 4x4 pose matrices for the left and right robot TCPs
    from the observation dictionary.

    Args:
        obs_dict: The observation dictionary from the robot.

    Returns:
        A dictionary containing the initial 4x4 pose matrices for 'left' and 'right' arms.
    """
    # RizonRobot_1 provides a flat array of state data.
    # Left arm data starts at index 14: [pos(3), quat_wxyz(4), gripper(1)]
    # Right arm data starts at index 36.
    robot_state = obs_dict['RizonRobot_1']

    # Left arm
    left_pos = robot_state[14:17]
    left_quat_wxyz = robot_state[17:21] # Rizon is w,x,y,z
    left_quat_xyzw = left_quat_wxyz[[1, 2, 3, 0]]
    left_mat = st.Rotation.from_quat(left_quat_xyzw).as_matrix()
    initial_left_pose_mat = np.eye(4)
    initial_left_pose_mat[:3, :3] = left_mat
    initial_left_pose_mat[:3, 3] = left_pos

    # Right arm
    right_pos = robot_state[36:39]
    right_quat_wxyz = robot_state[39:43] # Rizon is w,x,y,z
    right_quat_xyzw = right_quat_wxyz[[1, 2, 3, 0]]
    right_mat = st.Rotation.from_quat(right_quat_xyzw).as_matrix()
    initial_right_pose_mat = np.eye(4)
    initial_right_pose_mat[:3, :3] = right_mat
    initial_right_pose_mat[:3, 3] = right_pos

    return {'left': initial_left_pose_mat, 'right': initial_right_pose_mat}


class DataReplayerRelative(PolicyConnector):
    """
    Connector that:
      - Reads observations from summary SHM (via BaseConsumer)
      - Runs a policy to compute action chunking
      - Writes action chunking into policy SHM for ActionExecutor to consume
    
    Policy SHM is connect-only, created by ActionExecutor.
    """

    def __init__(self,
            summary_shm_name: str = "device_summary_data",
            policy_class: str = "TinyMLPPolicy",
            policy_params: Optional[Dict[str, Any]] = None,
            obs_devices: Optional[List[str]] = None,
            controls: Optional[List[Dict[str, Any]]] = None,
            fps: float = 50.0,
            chunk_length: int = 10,
            retry_connect_secs: float = 0.5,
            master_device: Optional[str] = None,
            enable_visualization: bool = True,
            max_trajectory_points: int = 200,
            enable_calibration: bool = False
        ):
        super().__init__(summary_shm_name, policy_class, policy_params, obs_devices, controls, fps, chunk_length, retry_connect_secs, master_device)

        # Visualization parameters
        self.enable_visualization = enable_visualization
        self.max_trajectory_points = max_trajectory_points
        
        # Chunk trajectory storage for different prediction times
        self.chunk_trajectories: Dict[str, List] = {}  # Store all chunk trajectories over time
        self.chunk_counter: int = 0  # Counter for chunk predictions
        
        # Initialize rerun visualization if enabled
        if self.enable_visualization:
            self.visualizer = RerunVisualizer("data_replayer_visualizer", spawn=True)
            self.visualizer.setup_3d_world("trajectory/world", coordinate_system="z_up")

        # Launch calibration GUI in background to edit vive_tcp_T interactively
        if enable_calibration:
            try:
                self.calib_gui: Optional[CalibrationGUI] = CalibrationGUI()
                self._calib_thread = threading.Thread(target=self.calib_gui.run, daemon=True)
                self._calib_thread.start()
                logger.info("Calibration GUI started (Vive->TCP)")
            except Exception as e:
                self.calib_gui = None
                logger.error(f"Failed to start Calibration GUI: {e}")
        else:
            self.calib_gui = None

        # Cache last valid observation to allow recalculation even if current obs is missing
        self._last_obs_dict: Optional[Dict[str, np.ndarray]] = None
        self._last_obs_time: Optional[int] = None

        # Remember calibration mode
        self.enable_calibration: bool = enable_calibration


    def _parse_dual_arm_action(self, action_chunk: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Parse dual arm action chunk into left and right arm data.
        
        Args:
            action_chunk: Shape (chunk_length, 16) for dual arm
                         [left_pos(3), left_quat(4), left_gripper(1), 
                          right_pos(3), right_quat(4), right_gripper(1)]
        
        Returns:
            Dict with 'left_arm' and 'right_arm' keys, each containing pos, quat, gripper
        """
        if action_chunk.shape[1] != 16:
            logger.warning(f"Expected action_chunk with 16 dimensions, got {action_chunk.shape[1]}")
            return {}
        
        parsed_actions = {}
        
        # Left arm: indices 0-7 (pos: 0-2, quat: 3-6, gripper: 7)
        parsed_actions['left_arm'] = {
            'position': action_chunk[:, 0:3],      # [chunk_length, 3]
            'quaternion': action_chunk[:, 3:7],    # [chunk_length, 4] - [x,y,z,w]
            'gripper': action_chunk[:, 7:8]        # [chunk_length, 1]
        }
        
        # Right arm: indices 8-15 (pos: 8-10, quat: 11-14, gripper: 15)
        parsed_actions['right_arm'] = {
            'position': action_chunk[:, 8:11],     # [chunk_length, 3]
            'quaternion': action_chunk[:, 11:15],  # [chunk_length, 4] - [x,y,z,w]
            'gripper': action_chunk[:, 15:16]      # [chunk_length, 1]
        }
        
        return parsed_actions

    def _visualize_predicted_trajectory(self, device_name: str, action_chunk_dict: Dict[str, np.ndarray], timestamp_ns: int) -> None:
        """
        Visualize predicted 3D trajectory with rotation for a device.
        
        Args:
            device_name: Name of the device
            action_chunk_dict: Dictionary containing action chunks for each device
            timestamp_ns: Current timestamp in nanoseconds
        """
        if not self.enable_visualization or device_name not in action_chunk_dict:
            return
            
        try:
            action_chunk = action_chunk_dict[device_name]
            
            # Parse dual arm actions
            parsed_actions = self._parse_dual_arm_action(action_chunk)
            
            if not parsed_actions:
                return
            
            # Store this chunk trajectory
            chunk_key = f"{device_name}_chunk_{self.chunk_counter}"
            if chunk_key not in self.chunk_trajectories:
                self.chunk_trajectories[chunk_key] = {
                    'timestamp': timestamp_ns,
                    'arms': parsed_actions
                }
            
            # Visualize action values for this chunk
            self._visualize_action_values(device_name, parsed_actions, timestamp_ns)
            
            # Visualize all chunk trajectories
            self._visualize_chunk_trajectories(device_name)
            
            # Increment chunk counter
            self.chunk_counter += 1
                    
        except Exception as e:
            logger.error(f"Error visualizing trajectory for {device_name}: {e}")

    def _visualize_action_values(self, device_name: str, parsed_actions: Dict[str, Dict[str, np.ndarray]], 
                                timestamp_ns: int) -> None:
        """
        Visualize action values using rerun utilities.
        
        Args:
            device_name: Name of the device
            parsed_actions: Parsed action data for both arms
            timestamp_ns: Current timestamp in nanoseconds
        """
        try:
            # Set time context
            set_time_context(timestamp_ns, "timestamp")
            
            # Log action time series for each arm
            for arm_name, arm_data in parsed_actions.items():
                positions = arm_data['position']
                grippers = arm_data['gripper']
                
                # Use utility function for time series visualization
                visualize_action_time_series(
                    "action_trends", device_name, arm_name, 
                    positions, grippers, timestamp_ns
                )
            
            # Reset time to main timestamp
            set_time_context(timestamp_ns, "timestamp")
            
        except Exception as e:
            logger.error(f"Error visualizing action values: {e}")

    def _visualize_chunk_trajectories(self, device_name: str) -> None:
        """
        Visualize all chunk trajectories with time direction indicators.
        
        Args:
            device_name: Name of the device
        """
        try:
            # Keep only recent chunks to avoid clutter (last 5 chunks)
            max_chunks_to_show = 5
            
            # Get recent chunk keys for this device and sort by numeric chunk index
            device_chunks = [key for key in self.chunk_trajectories.keys() if device_name in key]
            def _chunk_idx(k: str) -> int:
                try:
                    # expect pattern: "{device}_chunk_{idx}..."
                    part = k.split("_chunk_")[-1]
                    num_str = ''.join(ch for ch in part if ch.isdigit())
                    return int(num_str) if num_str else -1
                except Exception:
                    return -1
            recent_chunks = sorted(device_chunks, key=_chunk_idx)[-max_chunks_to_show:]
            
            for chunk_idx, chunk_key in enumerate(recent_chunks):
                chunk_data = self.chunk_trajectories[chunk_key]
                chunk_timestamp = chunk_data['timestamp']
                chunk_arms = chunk_data['arms']
                
                # Calculate opacity based on age (newer chunks more opaque)
                opacity = 0.3 + 0.7 * (chunk_idx / max(1, len(recent_chunks) - 1))
                
                for arm_name, arm_data in chunk_arms.items():
                    positions = arm_data['position']
                    quaternions = arm_data['quaternion']
                    grippers = arm_data['gripper']
                    
                    if len(positions) < 2:
                        continue
                    
                    # Create chunk-specific path
                    chunk_arm_path = f"trajectory/world/chunk_trajectories/{chunk_key}_{arm_name}"
                    
                    # Create colors for trajectory based on gripper state and opacity
                    segment_colors = []
                    for i in range(len(positions)):
                        gripper_width = grippers[i][0]
                        # Use utility function for color conversion
                        base_color = gripper_width_to_color(gripper_width)
                        
                        # Apply opacity based on chunk age
                        r = int(base_color[0] * opacity)
                        g = int(base_color[1] * opacity)
                        b = int(base_color[2] * opacity)
                        
                        segment_colors.append([r, g, b])
                    
                    # Use utility function for trajectory with rotation visualization
                    visualize_trajectory_with_rotation(
                        chunk_arm_path, positions, quaternions, segment_colors, 
                        arrow_scale=0.02, rotation_scale=0.03, show_every_n=4
                    )
                    
                    # Add chunk info using utility function
                    log_text_summary(
                        f"{chunk_arm_path}/info",
                        f"Chunk {chunk_key} | {arm_name} | Time: {chunk_timestamp/1e9:.3f}s"
                    )
            
            # Clean up old chunks to prevent memory buildup
            if len(self.chunk_trajectories) > max_chunks_to_show * 2:
                # Remove oldest chunks
                all_chunks = sorted(self.chunk_trajectories.keys())
                chunks_to_remove = all_chunks[:-max_chunks_to_show * 2]
                for chunk_key in chunks_to_remove:
                    del self.chunk_trajectories[chunk_key]
                    
        except Exception as e:
            logger.error(f"Error visualizing chunk trajectories: {e}")

    def _visualize_full_trajectory(self, device_name: str, replay_actions: np.ndarray) -> None:
        """Visualize the full recalculated trajectory as a preview for instant feedback.

        This draws entire left/right arm trajectories so calibration changes are immediately visible,
        without waiting for chunk playback.
        """
        try:
            if not self.enable_visualization:
                return

            if replay_actions.ndim != 2 or replay_actions.shape[1] != 16:
                logger.warning(f"Full preview expects (N,16), got {replay_actions.shape}")
                return

            # Clear previous preview tree
            try:
                rr.log("trajectory/world/full_preview", rr.Clear(recursive=True))
            except Exception:
                pass

            left_positions = replay_actions[:, 0:3]
            left_quaternions = replay_actions[:, 3:7]
            left_grippers = replay_actions[:, 7:8]

            right_positions = replay_actions[:, 8:11]
            right_quaternions = replay_actions[:, 11:15]
            right_grippers = replay_actions[:, 15:16]

            # Colors per point based on gripper width
            left_colors = [gripper_width_to_color(float(g[0])) for g in left_grippers]
            right_colors = [gripper_width_to_color(float(g[0])) for g in right_grippers]

            visualize_trajectory_with_rotation(
                f"trajectory/world/full_preview/{device_name}/left_arm",
                left_positions,
                left_quaternions,
                left_colors,
                arrow_scale=0.02,
                rotation_scale=0.03,
                show_every_n=4,
            )

            visualize_trajectory_with_rotation(
                f"trajectory/world/full_preview/{device_name}/right_arm",
                right_positions,
                right_quaternions,
                right_colors,
                arrow_scale=0.02,
                rotation_scale=0.03,
                show_every_n=4,
            )

            log_text_summary(
                f"trajectory/world/full_preview/{device_name}/info",
                f"Full trajectory preview updated. Num points: {replay_actions.shape[0]}"
            )

        except Exception as e:
            logger.error(f"Error visualizing full preview trajectory: {e}")

    def _connect_policy_shm(self) -> None:
        """Connect to policy SHM created by ActionExecutor."""
        try:
            # Connect to existing policy SHM using unified interface (read-write for data replayer)
            self.policy_shm = connect_to_policy_shm(read_only=False)
            
            # Read manager header to get device count and cache device info
            buf = self.policy_shm.buf
            manager_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
            device_count = manager_header['device_count']
            
            # Cache device info for efficient writing
            self.policy_devices = []
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset:device_offset+DEVICE_HEADER_SIZE])
                
                device_name = device_header['device_type']
                shape = device_header['shape']  # [chunk_length, action_dim]
                frame_size = device_header['frame_size']
                
                self.policy_devices.append({
                    'device_name': device_name,
                    'shape': shape,
                    'frame_size': frame_size,
                    'data_offset': None  # Will be calculated in write method
                })
                        
        except Exception as e:
            logger.error(f"PolicyConnector: Failed to connect to policy SHM: {e}")
            raise

    def _write_action_chunk_dict(self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: int=-1) -> None:
        """Write dict-based action chunking to policy SHM."""
        if not self.policy_shm:
            return
        try:
            buf = self.policy_shm.buf
            timestamp_ns = time.time_ns()
            
            # Update manager header to indicate data is available
            device_count = len(self.policy_devices)
            manager_header = pack_policy_header(device_count, timestamp_ns)  # Use timestamp to indicate data availability
            buf[:POLICY_HEADER_SIZE] = manager_header
            
            # Calculate frame sizes for get_data_offset
            frame_sizes = [device['frame_size'] for device in self.policy_devices]
            
            # Write data for each device
            for i, device_info in enumerate(self.policy_devices):
                device_name = device_info['device_name']
                expected_shape = device_info['shape']  # [chunk_length, action_dim]
                
                # Get action chunk for this device
                if device_name not in action_chunk_dict:
                    logger.warning(f"No action chunk for device: {device_name}")
                    continue
                
                action_chunk = action_chunk_dict[device_name]
                
                # Ensure correct shape
                if len(action_chunk.shape) == 1:
                    action_chunk = action_chunk.reshape(1, -1)
                
                # Verify shape matches expected
                if action_chunk.shape != tuple(expected_shape):
                    logger.error(f"Shape mismatch for {device_name}: got {action_chunk.shape}, expected {expected_shape}")
                    continue
                
                # Calculate data offset for this device
                data_offset = get_policy_data_offset(i, device_count, frame_sizes)
                
                # Write frame header with observation timestamp
                frame_header = pack_frame_header(observation_timestamp)
                buf[data_offset:data_offset+FRAME_HEADER_SIZE] = frame_header
                
                # Write chunk data
                data_start = data_offset + FRAME_HEADER_SIZE
                chunk_bytes = action_chunk.astype(np.float64).tobytes()
                data_end = data_start + len(chunk_bytes)
                buf[data_start:data_end] = chunk_bytes
            
            logger.info(f"Wrote action chunks to policy SHM: devices={list(action_chunk_dict.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to write action chunk dict: {e}")
            traceback.print_exc()

    def _assemble_observation_dict(self) -> Optional[Dict[str, np.ndarray]]:
        """Assemble observation dictionary using atomic read operation."""
        try:
            # Use the new atomic read function
            all_data = self.read_all_device_data()
            if all_data is None:
                return None, None
            
            # Extract just the data arrays (not timestamps)
            obs_dict = {}
            all_obs_time = []
            for device_name, (timestamp, data) in all_data.items():
                obs_dict[device_name] = data
                if self.master_device is None:
                    all_obs_time.append(timestamp)

            # Compute the timestamp for this frame of observation
            if self.master_device is None:
                obs_time = int(np.array(all_obs_time).mean().item()) # use averaging time of all devices if the `master_device` is invalid
            else:
                obs_time = all_data[self.master_device][0]
            return obs_dict, obs_time
            
        except Exception as e:
            logger.error(f"Error assembling observation dict: {e}")
            return None, None


    def run(self) -> None:
        if not self.connect():
            logger.error("PolicyConnector failed to connect to summary SHM")
            return

        # Connect to policy SHM for outputting action chunking
        self._connect_policy_shm()

        logger.info(f"PolicyConnector running with {self.fps} Hz inference rate. Press Ctrl+C to stop...")
        self.running = True


        # if self.replay_data is None:
            # load replay data
        import h5py
        # episode_path = "/home/zixiying/Desktop/wz/SuperInference/calibrate3/episode_0000.hdf5"
        # episode_path = "/home/zixiying/Desktop/wz/SuperInference/calibrate/leftgripper_translation_up_left_front.hdf5"
        # left up right | up-down-rot left-right-rot dir-no-change-rot
        episode_path = "/home/zixiying/Desktop/wz/SuperInference/data/pick_task_v5/episode_0000.hdf5"
        replay_actions = None

        step_count = 0
        # Use policy chunk_length to match policy SHM shape
        delta = self.chunk_length
        try:
            while self.running:
                start = time.time()
                obs_dict, obs_time = self._assemble_observation_dict()
                # Keep the latest valid observation to allow recalculation even if current obs missing
                if obs_dict is not None:
                    self._last_obs_dict = obs_dict
                    self._last_obs_time = obs_time
                
                # Recompute actions if first time or calibration updated (use cached obs if needed)
                if replay_actions is None or (self.calib_gui is not None and self.calib_gui.is_dirty()):
                    # Pick the observation to use
                    obs_for_calc = obs_dict if obs_dict is not None else self._last_obs_dict
                    if obs_for_calc is None:
                        logger.warning("No observation available to recompute trajectory during calibration change.")
                    else:
                        # Sync current GUI matrix to globals
                        if self.calib_gui is not None:
                            new_matrix = self.calib_gui.get_current_matrix()
                            logger.info(f"Applying new calibration matrix from GUI:\n{new_matrix}")
                            set_vive_tcp_T(new_matrix)
                            self.calib_gui.mark_clean()
                            logger.info("Calibration GUI marked as clean")
                        # Clear previous visualization chunk cache and viewer nodes
                        self.chunk_trajectories.clear()
                        self.chunk_counter = 0
                        try:
                            rr.log("trajectory/world/chunk_trajectories", rr.Clear(recursive=True))
                        except Exception:
                            pass
                        logger.info("Recomputing trajectory with updated calibration...")
                        replay_actions = extract_traj_relative(episode_path, obs_for_calc)
                        step_count = 0
                        logger.info(f"Recomputed absolute actions from replay data. Trajectory length: {len(replay_actions)}")
                        # Immediate full-trajectory preview for instant feedback
                        try:
                            self._visualize_full_trajectory('RizonRobot_1', replay_actions)
                        except Exception as e:
                            logger.error(f"Failed to visualize full trajectory preview: {e}")
                    
                # If we still don't have actions, continue loop
                if replay_actions is None:
                    # Keep loop responsive even without actions
                    time.sleep(0.005)
                    continue

                # Calibration mode: only preview full trajectory, do not send actions
                if self.enable_calibration:
                    time.sleep(0.01)
                    continue

                # Replay mode: send fixed-size chunks to policy SHM
                if step_count + delta > len(replay_actions):
                    logger.info("Trajectory replay finished. Looping to start.")
                    step_count = 0

                action_chunk = replay_actions[step_count:step_count+delta]
                if action_chunk.shape[0] != delta:
                    # Safety: if insufficient samples, loop back
                    step_count = 0
                    action_chunk = replay_actions[step_count:step_count+delta]

                action_chunk_dict = {
                    'RizonRobot_1': action_chunk,
                }
                logger.info(f"Sending action chunk. Step: {step_count} -> {step_count+delta}")

                # Visualize predicted trajectories if enabled
                if self.enable_visualization:
                    for device_name in action_chunk_dict.keys():
                        self._visualize_predicted_trajectory(device_name, action_chunk_dict.copy(), obs_time)
                self._write_action_chunk_dict(action_chunk_dict, obs_time)
                step_count += delta
                elapsed = time.time() - start
                sleep_time = max(0.0, self.update_interval - elapsed)
                logger.info(f"Update interval: {self.update_interval} seconds, step count: {step_count}")
                # # TODO: use precise_time_looping
                if sleep_time > 0:
                    if self.calib_gui is not None:
                        time.sleep(0.01)  # Keep GUI responsive
                    else:
                        time.sleep(sleep_time)
                        
        except KeyboardInterrupt:
            logger.info("Stopping PolicyConnector...")
        finally:
            self.stop()
            if self.policy_shm:
                try:
                    self.policy_shm.close()
                except Exception:
                    pass


def load_policy_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
        
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy Connector - run a policy and write action chunking to policy SHM")
    parser.add_argument("--config", type=str, required=False, default="policy/configs/diffusion_policy_obs-rel-trans-clean.yaml",
                        help="YAML path for policy connector config")
    parser.add_argument("--transform_json", type=str, required=False, default="data/tcp_to_vive_transform.json",
                        help="Path to JSON file containing the TCP to Vive transformation matrix")

    args = parser.parse_args()
    
    # Initialize transformation matrices from JSON file
    initialize_transform_from_json(args.transform_json)

    cfg_yaml = load_policy_yaml(args.config)
    policy_cfg = cfg_yaml.get('policy', {})

    policy_class = policy_cfg.get('class', 'TinyMLPPolicy')
    policy_params = policy_cfg.get('params', {})
    fps = policy_cfg.get('fps', 50.0)
    chunk_length = policy_cfg.get('chunk_length', 10)

    obs_cfg = policy_cfg.get('obs', {})
    obs_devices = obs_cfg.get('devices')

    controls_cfg = policy_cfg.get('controls', [])

    master_device = policy_cfg.get('master_device', None)
    enable_visualization = policy_cfg.get('enable_visualization', True)
    max_trajectory_points = policy_cfg.get('max_trajectory_points', 200)

    connector = DataReplayerRelative(
        summary_shm_name=cfg_yaml.get('summary_shm', 'device_summary_data'),
        policy_class=policy_class,
        policy_params=policy_params,
        obs_devices=obs_devices,
        controls=controls_cfg,
        fps=fps,
        chunk_length=chunk_length,
        master_device=master_device,
        enable_visualization=enable_visualization,
        max_trajectory_points=max_trajectory_points,
        enable_calibration=False
    )

    connector.run()



if __name__ == "__main__":
    main() 