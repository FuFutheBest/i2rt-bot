#!/usr/bin/env python3
"""
Interactive 6DOF Transform Visualizer with GUI Controls using Rerun

This tool provides a GUI-based interface to visualize 4x4 rigid body transformations
by adjusting x, y, z position and roll, pitch, yaw rotation parameters using sliders.

Features:
- Real-time 3D coordinate frame visualization
- GUI sliders for parameter adjustment with arcball rotation control
- 4x4 transformation matrix display
- Uses scipy.spatial.transform.Rotation for quaternion operations
- Uses rerun for 3D visualization
- Tkinter GUI for parameter controls
- Support for setting initial transformation matrix

Usage:
    python scripts/analysis/transform_visualizer_gui.py [--initial-matrix matrix_file.json]

Author: Han Xue
"""

import numpy as np
import rerun as rr
import threading
import tkinter as tk
from tkinter import ttk
from typing import Tuple, Optional, Any
import scipy.spatial.transform as st
from transforms3d.euler import mat2euler
from loguru import logger
import argparse
import json


class TransformVisualizerGUI:
    """Interactive 6DOF transform visualizer with GUI controls"""
    
    def __init__(self, initial_matrix: Optional[np.ndarray] = None):
        """Initialize the visualizer
        
        Args:
            initial_matrix: Optional 4x4 transformation matrix to start with
        """
        # Initialize rerun
        rr.init("transform_visualizer_gui", spawn=True)
        
        # Initialize from provided matrix or identity
        if initial_matrix is not None:
            current_matrix = initial_matrix.copy()
        else:
            current_matrix = np.eye(4)
        
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
        try:
            euler_angles = mat2euler(current_matrix[:3, :3], axes='sxyz')
            self.roll: float = float(euler_angles[0])
            self.pitch: float = float(euler_angles[1])
            self.yaw: float = float(euler_angles[2])
        except Exception:
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0

        # Parameter ranges
        self.position_range = (-0.3, 0.3)
        self.rotation_range_deg = (-180.0, 180.0)

        # Control flags
        self.running = True
        self.gui_root: Optional[tk.Tk] = None
        self._dirty: bool = True
        self._lock = threading.Lock()

        # Tk variables
        self._x_var = None
        self._y_var = None
        self._z_var = None
        self._roll_var = None
        self._pitch_var = None
        self._yaw_var = None
        
        # Arcball state
        self._arc_canvas: Optional[tk.Canvas] = None
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
        
        # Setup GUI
        self._setup_gui()
        
        logger.info("Transform Visualizer GUI initialized")
        
    def _build_matrix(self) -> np.ndarray:
        """Build 4x4 transformation matrix from current params using quaternion (no gimbal lock)."""
        with self._lock:
            R = st.Rotation.from_quat(self.quat).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array([self.x, self.y, self.z])
        return T

    def get_current_matrix(self) -> np.ndarray:
        """Get current transformation matrix"""
        return self._build_matrix()

    def mark_clean(self) -> None:
        """Mark the current state as clean"""
        with self._lock:
            self._dirty = False

    def is_dirty(self) -> bool:
        """Check if the current state is dirty"""
        with self._lock:
            return self._dirty
    
    def create_coordinate_frame(self, transform_matrix: np.ndarray, 
                               frame_name: str = "transformed_frame", 
                               axis_length: float = 0.3,
                               colors: Optional[list] = None) -> None:
        """
        Create and log a coordinate frame visualization in rerun
        
        Args:
            transform_matrix: 4x4 transformation matrix
            frame_name: Name for the coordinate frame
            axis_length: Length of the coordinate axes
            colors: Optional custom colors for [x, y, z] axes
        """
        if colors is None:
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red, Green, Blue
        
        # Extract position and rotation from transform matrix
        position = transform_matrix[:3, 3]
        rotation_matrix = transform_matrix[:3, :3]
        
        # Log the coordinate frame transformation
        rr.log(
            f"{frame_name}/origin",
            rr.Transform3D(
                translation=position,
                mat3x3=rotation_matrix
            )
        )
        
        # Create coordinate axes as lines
        origin = position
        
        # X-axis
        x_axis_end = origin + rotation_matrix[:, 0] * axis_length
        rr.log(
            f"{frame_name}/x_axis",
            rr.LineStrips3D(
                strips=[[origin, x_axis_end]],
                colors=[colors[0]]
            )
        )
        
        # Y-axis
        y_axis_end = origin + rotation_matrix[:, 1] * axis_length
        rr.log(
            f"{frame_name}/y_axis", 
            rr.LineStrips3D(
                strips=[[origin, y_axis_end]],
                colors=[colors[1]]
            )
        )
        
        # Z-axis
        z_axis_end = origin + rotation_matrix[:, 2] * axis_length
        rr.log(
            f"{frame_name}/z_axis",
            rr.LineStrips3D(
                strips=[[origin, z_axis_end]], 
                colors=[colors[2]]
            )
        )
    
    def log_transform_info(self, transform_matrix: np.ndarray) -> None:
        """
        Log transformation parameters and matrix to rerun
        
        Args:
            transform_matrix: Current 4x4 transformation matrix
        """
        # Log current parameters as scalars for plotting
        rr.log("parameters/position/x", rr.TextLog(self.x))
        rr.log("parameters/position/y", rr.TextLog(self.y))
        rr.log("parameters/position/z", rr.TextLog(self.z))
        rr.log("parameters/rotation/roll_deg", rr.TextLog(np.degrees(self.roll)))
        rr.log("parameters/rotation/pitch_deg", rr.TextLog(np.degrees(self.pitch)))
        rr.log("parameters/rotation/yaw_deg", rr.TextLog(np.degrees(self.yaw)))
        
        # Log transformation matrix as text
        matrix_text = f"""
## Current 4x4 Transformation Matrix:
```
[[{transform_matrix[0, 0]:8.4f},{transform_matrix[0, 1]:8.4f},{transform_matrix[0, 2]:8.4f},{transform_matrix[0, 3]:8.4f}],
[{transform_matrix[1, 0]:8.4f},{transform_matrix[1, 1]:8.4f},{transform_matrix[1, 2]:8.4f},{transform_matrix[1, 3]:8.4f}],
[{transform_matrix[2, 0]:8.4f},{transform_matrix[2, 1]:8.4f},{transform_matrix[2, 2]:8.4f},{transform_matrix[2, 3]:8.4f}],
[{transform_matrix[3, 0]:8.4f},{transform_matrix[3, 1]:8.4f},{transform_matrix[3, 2]:8.4f},{transform_matrix[3, 3]:8.4f}]]
```

## Parameters:
- **Position**: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}
- **Rotation (degrees)**: roll={np.degrees(self.roll):.1f}°, pitch={np.degrees(self.pitch):.1f}°, yaw={np.degrees(self.yaw):.1f}°
- **Rotation (radians)**: roll={self.roll:.3f}, pitch={self.pitch:.3f}, yaw={self.yaw:.3f}

## Quaternion (x, y, z, w):
{self.quat}
"""
        
        rr.log("info/transform_matrix", rr.TextDocument(matrix_text, media_type=rr.MediaType.MARKDOWN))
    
    
    def create_reference_grid(self) -> None:
        """Create a reference grid for scale"""
        grid_size = 2.0
        grid_step = 0.2
        grid_lines = []
        
        # X-direction lines
        for y in np.arange(-grid_size, grid_size + grid_step, grid_step):
            grid_lines.append([[-grid_size, y, 0], [grid_size, y, 0]])
        
        # Y-direction lines  
        for x in np.arange(-grid_size, grid_size + grid_step, grid_step):
            grid_lines.append([[x, -grid_size, 0], [x, grid_size, 0]])
            
        rr.log(
            "reference/grid",
            rr.LineStrips3D(
                strips=grid_lines,
                colors=[[100, 100, 100]]  # Gray
            )
        )
    
    def _setup_gui(self) -> None:
        """Setup the GUI control panel"""
        self.gui_root = tk.Tk()
        self.gui_root.title("Transform Visualizer (x,y,z,r,p,y)")
        self.gui_root.geometry("520x520")

        frame = ttk.Frame(self.gui_root, padding="10")
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

        # Rotation value labels and input fields
        self._roll_val_var = tk.StringVar(value=f"{np.degrees(self.roll):.1f} °")
        self._pitch_val_var = tk.StringVar(value=f"{np.degrees(self.pitch):.1f} °")
        self._yaw_val_var = tk.StringVar(value=f"{np.degrees(self.yaw):.1f} °")
        
        # Display labels
        vals = ttk.Frame(frame)
        vals.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(4,0))
        ttk.Label(vals, text="roll (x):").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(vals, textvariable=self._roll_val_var, width=10).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(vals, text="pitch (y):").grid(row=0, column=2, sticky=tk.W)
        ttk.Label(vals, textvariable=self._pitch_val_var, width=10).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(vals, text="yaw (z):").grid(row=0, column=4, sticky=tk.W)
        ttk.Label(vals, textvariable=self._yaw_val_var, width=10).grid(row=0, column=5, sticky=tk.W)
        row += 1
        
        # Input fields for roll, pitch, yaw (in degrees)
        ttk.Label(frame, text="Rotation (degrees)", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,0))
        row += 1
        
        # Roll input
        self._roll_var = tk.StringVar(value=f"{np.degrees(self.roll):.2f}")
        ttk.Label(frame, text="roll (x):").grid(row=row, column=0, sticky=tk.W)
        roll_entry = ttk.Entry(frame, textvariable=self._roll_var, width=15)
        roll_entry.grid(row=row, column=1, sticky=tk.W, padx=(5,0))
        roll_entry.bind("<Return>", self._on_euler_change)
        roll_entry.bind("<FocusOut>", self._on_euler_change)
        row += 1
        
        # Pitch input
        self._pitch_var = tk.StringVar(value=f"{np.degrees(self.pitch):.2f}")
        ttk.Label(frame, text="pitch (y):").grid(row=row, column=0, sticky=tk.W)
        pitch_entry = ttk.Entry(frame, textvariable=self._pitch_var, width=15)
        pitch_entry.grid(row=row, column=1, sticky=tk.W, padx=(5,0))
        pitch_entry.bind("<Return>", self._on_euler_change)
        pitch_entry.bind("<FocusOut>", self._on_euler_change)
        row += 1
        
        # Yaw input
        self._yaw_var = tk.StringVar(value=f"{np.degrees(self.yaw):.2f}")
        ttk.Label(frame, text="yaw (z):").grid(row=row, column=0, sticky=tk.W)
        yaw_entry = ttk.Entry(frame, textvariable=self._yaw_var, width=15)
        yaw_entry.grid(row=row, column=1, sticky=tk.W, padx=(5,0))
        yaw_entry.bind("<Return>", self._on_euler_change)
        yaw_entry.bind("<FocusOut>", self._on_euler_change)
        row += 1

        # Buttons
        btns = ttk.Frame(frame)
        btns.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btns, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Copy Matrix", command=self._copy_matrix).pack(side=tk.LEFT, padx=5)

        # Initialize on launch
        self._on_change()

        self.gui_root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_change(self, *_args: Any) -> None:
        """Callback when any parameter changes (position only)"""
        try:
            with self._lock:
                self.x = float(self._x_var.get())
                self.y = float(self._y_var.get())
                self.z = float(self._z_var.get())
                self._dirty = True

            # Update on-screen value labels
            self._update_value_labels()

            # Log current matrix for visibility
            T = self.get_current_matrix()
            logger.info(f"GUI Updated transform matrix:\n{T}")
            logger.debug(f"GUI parameters: x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, "
                        f"roll={np.degrees(self.roll):.2f}°, pitch={np.degrees(self.pitch):.2f}°, yaw={np.degrees(self.yaw):.2f}°")
            
            # Update visualization
            self.update_visualization()
        except Exception as e:
            logger.error(f"TransformVisualizerGUI on_change error: {e}")
    
    def _on_euler_change(self, *_args: Any) -> None:
        """Callback when euler angle input changes"""
        try:
            # Read values from input fields (in degrees)
            roll_deg = float(self._roll_var.get())
            pitch_deg = float(self._pitch_var.get())
            yaw_deg = float(self._yaw_var.get())
            
            # Convert to radians
            roll_rad = np.radians(roll_deg)
            pitch_rad = np.radians(pitch_deg)
            yaw_rad = np.radians(yaw_deg)
            
            # Update euler angles and convert to quaternion
            with self._lock:
                # Convert euler angles to quaternion (sxyz extrinsic)
                # scipy uses 'XYZ' (uppercase) for extrinsic rotations
                # 'sxyz' means static/extrinsic xyz rotation order
                rot = st.Rotation.from_euler('XYZ', [roll_rad, pitch_rad, yaw_rad], degrees=False)
                self.quat = rot.as_quat()  # [x, y, z, w]
                # Keep the user-input euler angles (don't recalculate from quaternion to preserve user input)
                self.roll = roll_rad
                self.pitch = pitch_rad
                self.yaw = yaw_rad
                self._dirty = True
            
            # Update value labels (but don't update input fields since user just set them)
            # Also skip recalculating euler from quaternion to preserve user input
            self._update_value_labels(update_input_fields=False, recalculate_euler=False)
            
            # Log current matrix for visibility
            T = self.get_current_matrix()
            logger.info(f"GUI Updated transform matrix from euler input:\n{T}")
            logger.debug(f"GUI euler parameters: roll={roll_deg:.2f}°, pitch={pitch_deg:.2f}°, yaw={yaw_deg:.2f}°")
            
            # Update visualization
            self.update_visualization()
        except ValueError as e:
            logger.warning(f"Invalid euler angle input: {e}. Please enter numeric values.")
            # Restore previous values
            self._roll_var.set(f"{np.degrees(self.roll):.2f}")
            self._pitch_var.set(f"{np.degrees(self.pitch):.2f}")
            self._yaw_var.set(f"{np.degrees(self.yaw):.2f}")
        except Exception as e:
            logger.error(f"TransformVisualizerGUI euler change error: {e}")

    def _update_value_labels(self, update_input_fields: bool = True, recalculate_euler: bool = True) -> None:
        """Update the value labels in the GUI
        
        Args:
            update_input_fields: If False, skip updating input fields (used when input comes from user)
            recalculate_euler: If False, don't recalculate euler angles from quaternion (preserve user input)
        """
        try:
            if self._x_val_var is not None:
                self._x_val_var.set(f"{self.x:.3f} m")
            if self._y_val_var is not None:
                self._y_val_var.set(f"{self.y:.3f} m")
            if self._z_val_var is not None:
                self._z_val_var.set(f"{self.z:.3f} m")
            # Update rpy from current quaternion for display (unless preserving user input)
            if recalculate_euler:
                try:
                    Rm = st.Rotation.from_quat(self.quat).as_matrix()
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
            # Update input fields to match current values (only if difference is significant)
            # This avoids interfering with user input while allowing sync from arcball
            if update_input_fields:
                if self._roll_var is not None:
                    try:
                        current_roll_deg = float(self._roll_var.get())
                        new_roll_deg = np.degrees(self.roll)
                        # Only update if difference is significant (more than 0.1 degree)
                        if abs(current_roll_deg - new_roll_deg) > 0.1:
                            self._roll_var.set(f"{new_roll_deg:.2f}")
                    except (ValueError, AttributeError):
                        # User might be typing, ignore
                        pass
                if self._pitch_var is not None:
                    try:
                        current_pitch_deg = float(self._pitch_var.get())
                        new_pitch_deg = np.degrees(self.pitch)
                        if abs(current_pitch_deg - new_pitch_deg) > 0.1:
                            self._pitch_var.set(f"{new_pitch_deg:.2f}")
                    except (ValueError, AttributeError):
                        pass
                if self._yaw_var is not None:
                    try:
                        current_yaw_deg = float(self._yaw_var.get())
                        new_yaw_deg = np.degrees(self.yaw)
                        if abs(current_yaw_deg - new_yaw_deg) > 0.1:
                            self._yaw_var.set(f"{new_yaw_deg:.2f}")
                    except (ValueError, AttributeError):
                        pass
        except (ValueError, AttributeError):
            # Ignore errors when updating input fields (user might be typing)
            pass
        except Exception as e:
            logger.error(f"TransformVisualizerGUI value label update error: {e}")

    # ---------- Arcball helpers ----------
    def _draw_arcball(self) -> None:
        """Draw the arcball circle"""
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
        """Map 2D canvas coordinates to 3D sphere coordinates"""
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

    def _on_arc_down(self, event: Any) -> None:
        """Handle arcball mouse down event"""
        try:
            self._dragging = True
            self._v0 = self._map_to_sphere(float(event.x), float(event.y))
        except Exception as e:
            logger.error(f"Arcball down error: {e}")

    def _on_arc_move(self, event: Any) -> None:
        """Handle arcball mouse move event"""
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
            # Update euler angles from quaternion and sync input fields
            self._update_value_labels()
            # Sync input fields with current euler values
            if self._roll_var is not None:
                self._roll_var.set(f"{np.degrees(self.roll):.2f}")
            if self._pitch_var is not None:
                self._pitch_var.set(f"{np.degrees(self.pitch):.2f}")
            if self._yaw_var is not None:
                self._yaw_var.set(f"{np.degrees(self.yaw):.2f}")
            self.update_visualization()
            self._v0 = v1
        except Exception as e:
            logger.error(f"Arcball move error: {e}")

    def _on_arc_up(self, _event: Any) -> None:
        """Handle arcball mouse up event"""
        self._dragging = False
        self._v0 = None
    
    def _reset(self) -> None:
        """Reset all parameters to zero"""
        try:
            self._x_var.set(0.0)
            self._y_var.set(0.0)
            self._z_var.set(0.0)
            if self._roll_var is not None:
                self._roll_var.set("0.00")
            if self._pitch_var is not None:
                self._pitch_var.set("0.00")
            if self._yaw_var is not None:
                self._yaw_var.set("0.00")
            with self._lock:
                self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
                self.roll = 0.0
                self.pitch = 0.0
                self.yaw = 0.0
                self._dirty = True
            self._on_change()
            # Redraw arcball for cleanliness
            try:
                self._draw_arcball()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"TransformVisualizerGUI reset error: {e}")

    def _copy_matrix(self) -> None:
        """Copy current transformation matrix to clipboard"""
        try:
            T = self.get_current_matrix()
            matrix_str = "np.array([\n" + \
                "\n".join(["    [" + ", ".join([f"{T[i, j]:8.4f}" for j in range(4)]) + "]" + ("," if i < 3 else "") for i in range(4)]) + \
                "\n])"
            if self.gui_root is not None:
                self.gui_root.clipboard_clear()
                self.gui_root.clipboard_append(matrix_str)
                logger.info("Transformation matrix copied to clipboard")
        except Exception as e:
            logger.error(f"TransformVisualizerGUI copy error: {e}")

    def _on_close(self) -> None:
        """Handle GUI window closing"""
        try:
            if self.gui_root is not None:
                self.gui_root.destroy()
        except Exception:
            pass
    
    def update_visualization(self) -> None:
        """Update the 3D visualization with current parameters"""
        # Compute transformation matrix using the new method
        transform_matrix = self.get_current_matrix()
        
        # Create world coordinate frame (reference) - darker colors
        world_transform = np.eye(4)
        self.create_coordinate_frame(
            world_transform, 
            "world_frame", 
            axis_length=0.5,
            colors=[[150, 75, 75], [75, 150, 75], [75, 75, 150]]  # Darker RGB
        )
        
        # Create transformed coordinate frame - bright colors
        self.create_coordinate_frame(
            transform_matrix, 
            "transformed_frame", 
            axis_length=0.4,
            colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]]  # Bright RGB
        )
        
        # Log transformation information
        self.log_transform_info(transform_matrix)
        
        # Add reference grid
        self.create_reference_grid()
        
        # Add trajectory trace (optional - shows path of origin)
        self.log_trajectory_point()
    
    def log_trajectory_point(self) -> None:
        """Log current position as part of a trajectory"""
        current_pos = np.array([self.x, self.y, self.z])
        
        # Log as a single point for trajectory visualization
        rr.log(
            "trajectory/current_position",
            rr.Points3D(
                positions=[current_pos],
                colors=[[255, 255, 0]],  # Yellow
                radii=[0.02]
            )
        )
    
    def create_reference_grid(self) -> None:
        """Create a reference grid for scale"""
        grid_size = 2.0
        grid_step = 0.2
        grid_lines = []
        
        # X-direction lines
        for y in np.arange(-grid_size, grid_size + grid_step, grid_step):
            grid_lines.append([[-grid_size, y, 0], [grid_size, y, 0]])
        
        # Y-direction lines  
        for x in np.arange(-grid_size, grid_size + grid_step, grid_step):
            grid_lines.append([[x, -grid_size, 0], [x, grid_size, 0]])
            
        rr.log(
            "reference/grid",
            rr.LineStrips3D(
                strips=grid_lines,
                colors=[[80, 80, 80]]  # Dark gray
            )
        )
    
    
    def setup_initial_view(self) -> None:
        """Setup initial rerun view and reference objects"""
        # Set up the 3D view
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
        
        # Log instructions
        instructions = """
# 6DOF Transform Visualizer (GUI Version)

## Rotation Convention:
- **Arcball Control**: Use the circular control to adjust rotation intuitively
- **Quaternion-based**: Uses quaternions internally to avoid gimbal lock
- **Display**: Shows roll, pitch, yaw angles for reference

## Coordinate Frames:
- **Dark Frame**: World reference (origin)
- **Bright Frame**: Transformed frame based on your parameters
- **Yellow Point**: Current position trajectory

## GUI Controls:
Use the sliders and arcball in the GUI window to adjust:
- Position: X, Y, Z coordinates (sliders)
- Rotation: Use arcball for intuitive 3D rotation control, or directly input roll, pitch, yaw values in degrees

## Matrix Output:
The 4x4 transformation matrix is displayed in real-time.
Use the "Copy Matrix" button to copy the current matrix to clipboard.
"""
        rr.log("info/instructions", rr.TextDocument(instructions, media_type=rr.MediaType.MARKDOWN))
        
        # Initial visualization
        self.update_visualization()
    
    
    def run(self) -> None:
        """Run the interactive visualizer"""
        logger.info("Starting Transform Visualizer GUI...")
        
        # Setup initial rerun view
        self.setup_initial_view()
        
        # Start GUI main loop
        try:
            self.gui_root.mainloop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.running = False
        
        logger.info("Transform Visualizer GUI stopped")


def load_matrix_from_file(file_path: str) -> np.ndarray:
    """Load a 4x4 transformation matrix from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            # Direct list format: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            matrix = np.array(data, dtype=float)
        elif isinstance(data, dict):
            # Dictionary format with 'matrix' key
            if 'matrix' in data:
                matrix = np.array(data['matrix'], dtype=float)
            elif 'transform' in data:
                matrix = np.array(data['transform'], dtype=float)
            elif 'transformation_matrix' in data:
                matrix = np.array(data['transformation_matrix'], dtype=float)
            else:
                # Try to find any key that contains a 4x4 matrix
                for key, value in data.items():
                    if isinstance(value, list) and len(value) == 4:
                        if all(isinstance(row, list) and len(row) == 4 for row in value):
                            matrix = np.array(value, dtype=float)
                            break
                else:
                    raise ValueError("No valid 4x4 matrix found in JSON data")
        else:
            raise ValueError("Invalid JSON format: expected list or dict")
        
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got {matrix.shape}")
        
        logger.info(f"Successfully loaded 4x4 matrix from {file_path}")
        return matrix
        
    except Exception as e:
        logger.error(f"Failed to load matrix from {file_path}: {e}")
        raise


def main():
    """Main function to run the transform visualizer"""
    parser = argparse.ArgumentParser(description="Interactive 6DOF Transform Visualizer with GUI Controls")
    parser.add_argument("--initial-matrix", type=str, help="Path to initial 4x4 transformation matrix JSON file")
    
    args = parser.parse_args()
    
    initial_matrix = None
    if args.initial_matrix:
        try:
            initial_matrix = load_matrix_from_file(args.initial_matrix)
            logger.info(f"Loaded initial matrix from {args.initial_matrix}")
        except Exception as e:
            logger.error(f"Failed to load initial matrix: {e}")
            return
    
    try:
        visualizer = TransformVisualizerGUI(initial_matrix=initial_matrix)
        visualizer.run()
    except Exception as e:
        logger.error(f"Error running visualizer: {e}")
        raise


if __name__ == "__main__":
    main()