#!/usr/bin/env python3
"""
Teleoperator device for robot control with GUI sliders.

Author: Jun Lv, Zixi Ying
"""

import bisect
import copy
import time
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Mapping, Tuple, Optional
from pathlib import Path
import transforms3d as t3d
from devices.base import BaseDevice
from devices.config import StateFeedbackConfig
from utils.shm_utils import (
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    get_dtype,
    connect_to_device_shm,
    unpack_device_header,
    unpack_buffer_header,
    unpack_frame_header,
)
from utils.logger_config import logger
from utils.lk_gripper_presets import (
    DEFAULT_LK_GRIPPER_TYPE,
    calculate_lk_gripper_max_width,
    get_lk_gripper_preset,
)
from incubator.r1.rizon_state_layout import RizonDualArmStateLayout
from scipy.spatial.transform import Rotation as R
from utils.time_control import precise_loop_timing
from utils.transform import compute_deviation, compute_deviation_3, matrix4x4_to_pose_6d, pose_6d_to_4x4matrix, pose_6d_to_pose_7d, pose_7d_to_4x4matrix

# Try to import exoskeleton module, handle import error gracefully
try:
    from exoskeleton import AngleEncoder
    EXOSKELETON_AVAILABLE = True
except ImportError:
    logger.warning("exoskeleton module not available. Airexoskeleton classes will not work.")
    EXOSKELETON_AVAILABLE = False
    AngleEncoder = None

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from utils.vr_util import UnityMes, TeleopMode, composed_cmd_interpreter, RealWorldTransforms
except ImportError:
    logger.warning("vr_util module not available. Quest3Teleoperator will not work.")
    UnityMes = None
    TeleopMode = None
    composed_cmd_interpreter = None
    RealWorldTransforms = None

class JointSlider(BaseDevice):
    """
    Joint slider teleoperator device for controlling robot joint positions.
    Provides GUI sliders for real-time joint control.
    """
    # This is all wirtten by Cursor, not reviewed

    def __init__(self, device_class_id=0, *, state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 joint_ranges=None):
        """
        Initialize the joint slider teleoperator.

        Args:
            device_class_id: Unique identifier for this device instance
            fps: Update frequency in Hz
            num_joints: Number of joints to control
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            joint_ranges: List of (min, max) ranges for each joint (default: [-pi, pi] for all)
        """
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        num_joints = self.data_shape[0]

        # Override device name
        self.device_name = "JointSlider"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        # Joint control attributes
        self.num_joints = num_joints

        # Set default joint ranges if not provided
        if joint_ranges is None:
            self.joint_ranges = [(-np.pi, np.pi)] * num_joints  # Default: -pi to pi
        else:
            self.joint_ranges = joint_ranges

        # Current joint positions (initialize to center of ranges)
        self.current_joints = []
        for min_val, max_val in self.joint_ranges:
            self.current_joints.append((min_val + max_val) / 2)

        # GUI elements
        self.root = None
        self.sliders = []
        self.labels = []
        self.value_labels = []

        # Threading
        self.running = False
        self.gui_thread = None
        self.lock = threading.Lock()

        logger.info(f"Joint Slider teleoperator initialized")
        logger.info(f"Number of joints: {num_joints}")
        logger.info(f"Joint ranges: {self.joint_ranges}")
        logger.info(f"Current positions: {[f'{pos:.3f}' for pos in self.current_joints]}")
        logger.info("")

    def _generate_joint_array(self) -> np.ndarray:
        """
        Generate current joint position array.

        Returns:
            numpy.ndarray: Current joint positions
        """
        with self.lock:
            numpy_dtype = get_dtype(self.data_dtype)
            return np.array(self.current_joints, dtype=numpy_dtype)

    def _update_joint(self, joint_idx: int, value: float) -> None:
        """
        Update joint position.

        Args:
            joint_idx: Joint index (0-based)
            value: New joint position
        """
        with self.lock:
            if 0 <= joint_idx < self.num_joints:
                min_val, max_val = self.joint_ranges[joint_idx]
                self.current_joints[joint_idx] = np.clip(value, min_val, max_val)
                logger.debug(f"Joint {joint_idx}: {self.current_joints[joint_idx]:.3f}")

    def _create_gui(self) -> None:
        """Create the GUI with sliders for each joint."""
        self.root = tk.Tk()
        self.root.title(f"Joint Slider Control - Device {self.device_class_id}")
        self.root.geometry("400x600")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Joint Position Control", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Create sliders for each joint
        for i in range(self.num_joints):
            min_val, max_val = self.joint_ranges[i]
            current_val = self.current_joints[i]

            # Joint label
            joint_label = ttk.Label(main_frame, text=f"Joint {i}:")
            joint_label.grid(row=i+1, column=0, sticky=tk.W, pady=5)

            # Slider
            slider = ttk.Scale(
                main_frame,
                from_=min_val,
                to=max_val,
                value=current_val,
                orient=tk.HORIZONTAL,
                length=200,
                command=lambda val, idx=i: self._on_slider_change(idx, val)
            )
            slider.grid(row=i+1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)

            # Value label
            value_label = ttk.Label(main_frame, text=f"{current_val:.3f}")
            value_label.grid(row=i+1, column=2, sticky=tk.W, padx=(5, 0), pady=5)

            self.sliders.append(slider)
            self.labels.append(joint_label)
            self.value_labels.append(value_label)

        # Reset button
        reset_button = ttk.Button(main_frame, text="Reset All", command=self._reset_all)
        reset_button.grid(row=self.num_joints+1, column=0, columnspan=3, pady=20)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=self.num_joints+2, column=0, columnspan=3, pady=10)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)

    def _on_slider_change(self, joint_idx: int, value: str) -> None:
        """Handle slider value change."""
        try:
            value = float(value)
            self._update_joint(joint_idx, value)

            # Update value label
            if joint_idx < len(self.value_labels):
                self.value_labels[joint_idx].config(text=f"{value:.3f}")

        except ValueError:
            logger.warning(f"Invalid slider value: {value}")

    def _reset_all(self) -> None:
        """Reset all joints to center of their ranges."""
        with self.lock:
            for i in range(self.num_joints):
                min_val, max_val = self.joint_ranges[i]
                center_val = (min_val + max_val) / 2
                self.current_joints[i] = center_val

                # Update slider and label
                if i < len(self.sliders):
                    self.sliders[i].set(center_val)
                if i < len(self.value_labels):
                    self.value_labels[i].config(text=f"{center_val:.3f}")

            logger.info("All joints reset to center")

    def _on_closing(self) -> None:
        """Handle window closing."""
        logger.info("GUI window closing...")
        self.running = False
        if self.root:
            self.root.quit()

    def _gui_loop(self) -> None:
        """Main GUI loop."""
        try:
            self._create_gui()
            logger.info("GUI created successfully")

            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Running", foreground="green")

            # Start GUI main loop
            self.root.mainloop()

        except Exception as e:
            logger.error(f"Error in GUI loop: {e}")
            self.running = False

    def start_server(self) -> None:
        """Start the joint slider teleoperator server."""
        if self.running:
            logger.info(f"Joint Slider teleoperator {self.device_class_id} is already running")
            return

        logger.info(f"Starting Joint Slider teleoperator {self.device_class_id} server...")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Start GUI thread
        self.gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self.gui_thread.start()

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main data writing loop
        while self.running:
            try:
                # Get current joint positions and write to SHM
                joint_array = self._generate_joint_array()
                timestamp_ns = time.time_ns()
                self._write_array_to_shm_with_timestamp(joint_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()

            except Exception as e:
                logger.error(f"Error in data writing: {e}")
                break

    def stop_server(self) -> None:
        """Stop the joint slider teleoperator server."""
        if not self.running:
            return

        logger.info(f"Stopping Joint Slider teleoperator {self.device_class_id} server...")
        self.running = False

        # Close GUI
        if self.root:
            try:
                self.root.quit()
            except:
                pass

        # Wait for GUI thread to finish
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=2.0)

        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "num_joints": self.num_joints,
            "joint_ranges": self.joint_ranges,
            "current_joints": self.current_joints.copy()
        })
        return info


class DualArmJointSlider(BaseDevice):
    """
    Dual-arm joint slider teleoperator device for controlling two robot arms.
    Provides GUI sliders for real-time joint control of both arms.
    """

    def __init__(self, device_class_id=0, *, state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 left_joint_ranges=None, right_joint_ranges=None):
        """
        Initialize the dual-arm joint slider teleoperator.

        Args:
            device_class_id: Unique identifier for this device instance
            fps: Update frequency in Hz
            left_arm_joints: Number of joints in left arm
            right_arm_joints: Number of joints in right arm
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            left_joint_ranges: List of (min, max) ranges for left arm joints (default: [-pi, pi] for all)
            right_joint_ranges: List of (min, max) ranges for right arm joints (default: [-pi, pi] for all)
        """
        # Initialize base device with joint position data for both arms
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        total_joints = self.data_shape[0]
        if total_joints % 2:
            raise ValueError("DualArmJointSlider state_feedback shape must have an even width")
        left_arm_joints = right_arm_joints = total_joints // 2

        # Override device name
        self.device_name = "DualArmJointSlider"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        # Joint control attributes
        self.left_arm_joints = left_arm_joints
        self.right_arm_joints = right_arm_joints
        self.total_joints = total_joints

        # Set default joint ranges if not provided
        if left_joint_ranges is None:
            self.left_joint_ranges = [(-np.pi, np.pi)] * left_arm_joints  # Default: -pi to pi
        else:
            self.left_joint_ranges = left_joint_ranges

        if right_joint_ranges is None:
            self.right_joint_ranges = [(-np.pi, np.pi)] * right_arm_joints  # Default: -pi to pi
        else:
            self.right_joint_ranges = right_joint_ranges

        # Current joint positions (initialize to center of ranges)
        self.left_current_joints = []
        for min_val, max_val in self.left_joint_ranges:
            self.left_current_joints.append((min_val + max_val) / 2)

        self.right_current_joints = []
        for min_val, max_val in self.right_joint_ranges:
            self.right_current_joints.append((min_val + max_val) / 2)

        # GUI elements
        self.root = None
        self.left_sliders = []
        self.right_sliders = []
        self.left_labels = []
        self.right_labels = []
        self.left_value_labels = []
        self.right_value_labels = []

        # Threading
        self.running = False
        self.gui_thread = None
        self.lock = threading.Lock()

        logger.info(f"Dual-Arm Joint Slider teleoperator initialized")
        logger.info(f"Left arm joints: {left_arm_joints}")
        logger.info(f"Right arm joints: {right_arm_joints}")
        logger.info(f"Total joints: {total_joints}")
        logger.info(f"Left joint ranges: {self.left_joint_ranges}")
        logger.info(f"Right joint ranges: {self.right_joint_ranges}")
        logger.info(f"Left current positions: {[f'{pos:.3f}' for pos in self.left_current_joints]}")
        logger.info(f"Right current positions: {[f'{pos:.3f}' for pos in self.right_current_joints]}")
        logger.info("")

    def _generate_joint_array(self) -> np.ndarray:
        """
        Generate current joint position array for both arms.

        Returns:
            numpy.ndarray: Current joint positions [left_arm_joints, right_arm_joints]
        """
        with self.lock:
            numpy_dtype = get_dtype(self.data_dtype)
            # Combine left and right arm joint positions
            all_joints = self.left_current_joints + self.right_current_joints
            return np.array(all_joints, dtype=numpy_dtype)

    def _update_left_joint(self, joint_idx: int, value: float) -> None:
        """
        Update left arm joint position.

        Args:
            joint_idx: Joint index (0-based)
            value: New joint position
        """
        if 0 <= joint_idx < self.left_arm_joints:
            min_val, max_val = self.left_joint_ranges[joint_idx]
            with self.lock:
                self.left_current_joints[joint_idx] = np.clip(value, min_val, max_val)

    def _update_right_joint(self, joint_idx: int, value: float) -> None:
        """
        Update right arm joint position.

        Args:
            joint_idx: Joint index (0-based)
            value: New joint position
        """
        if 0 <= joint_idx < self.right_arm_joints:
            min_val, max_val = self.right_joint_ranges[joint_idx]
            with self.lock:
                self.right_current_joints[joint_idx] = np.clip(value, min_val, max_val)

    def _create_gui(self) -> None:
        """Create the dual-arm GUI with sliders for both arms."""
        self.root = tk.Tk()
        self.root.title(f"Dual-Arm Joint Slider - Device {self.device_class_id}")
        self.root.geometry("800x600")

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(3, weight=1)

        # Left Arm Section
        left_frame = ttk.LabelFrame(main_frame, text="Left Arm", padding="5")
        left_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Right Arm Section
        right_frame = ttk.LabelFrame(main_frame, text="Right Arm", padding="5")
        right_frame.grid(row=0, column=2, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))

        # Create sliders for left arm
        for i in range(self.left_arm_joints):
            # Joint label
            label = ttk.Label(left_frame, text=f"Joint {i+1}")
            label.grid(row=i, column=0, sticky=tk.W, padx=(0, 5))
            self.left_labels.append(label)

            # Slider
            min_val, max_val = self.left_joint_ranges[i]
            slider = ttk.Scale(left_frame, from_=min_val, to=max_val,
                             value=self.left_current_joints[i],
                             orient=tk.HORIZONTAL, length=200,
                             command=lambda val, idx=i: self._on_left_slider_change(idx, val))
            slider.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
            self.left_sliders.append(slider)

            # Value label
            value_label = ttk.Label(left_frame, text=f"{self.left_current_joints[i]:.3f}")
            value_label.grid(row=i, column=2, sticky=tk.W)
            self.left_value_labels.append(value_label)

        # Create sliders for right arm
        for i in range(self.right_arm_joints):
            # Joint label
            label = ttk.Label(right_frame, text=f"Joint {i+1}")
            label.grid(row=i, column=0, sticky=tk.W, padx=(0, 5))
            self.right_labels.append(label)

            # Slider
            min_val, max_val = self.right_joint_ranges[i]
            slider = ttk.Scale(right_frame, from_=min_val, to=max_val,
                             value=self.right_current_joints[i],
                             orient=tk.HORIZONTAL, length=200,
                             command=lambda val, idx=i: self._on_right_slider_change(idx, val))
            slider.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
            self.right_sliders.append(slider)

            # Value label
            value_label = ttk.Label(right_frame, text=f"{self.right_current_joints[i]:.3f}")
            value_label.grid(row=i, column=2, sticky=tk.W)
            self.right_value_labels.append(value_label)

        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=(20, 0))

        # Reset button (single button like single arm)
        reset_all_btn = ttk.Button(button_frame, text="Reset All", command=self._reset_all)
        reset_all_btn.grid(row=0, column=0)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        logger.info("Dual-arm GUI created successfully")

    def _on_left_slider_change(self, joint_idx: int, value: str) -> None:
        """Callback for left arm slider changes."""
        value = float(value)
        self._update_left_joint(joint_idx, value)

        # Update value label
        if 0 <= joint_idx < len(self.left_value_labels):
            self.left_value_labels[joint_idx].config(text=f"{value:.3f}")

    def _on_right_slider_change(self, joint_idx: int, value: str) -> None:
        """Callback for right arm slider changes."""
        value = float(value)
        self._update_right_joint(joint_idx, value)

        # Update value label
        if 0 <= joint_idx < len(self.right_value_labels):
            self.right_value_labels[joint_idx].config(text=f"{value:.3f}")

    def _reset_all(self) -> None:
        """Reset all joints to center of their ranges."""
        with self.lock:
            # Reset left arm joints
            for i in range(self.left_arm_joints):
                min_val, max_val = self.left_joint_ranges[i]
                center_val = (min_val + max_val) / 2
                self.left_current_joints[i] = center_val

                # Update slider and label
                if i < len(self.left_sliders):
                    self.left_sliders[i].set(center_val)
                if i < len(self.left_value_labels):
                    self.left_value_labels[i].config(text=f"{center_val:.3f}")

            # Reset right arm joints
            for i in range(self.right_arm_joints):
                min_val, max_val = self.right_joint_ranges[i]
                center_val = (min_val + max_val) / 2
                self.right_current_joints[i] = center_val

                # Update slider and label
                if i < len(self.right_sliders):
                    self.right_sliders[i].set(center_val)
                if i < len(self.right_value_labels):
                    self.right_value_labels[i].config(text=f"{center_val:.3f}")

            logger.info("All joints reset to center")

    def _on_closing(self) -> None:
        """Handle GUI window close event."""
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()

    def _gui_loop(self) -> None:
        """Run the GUI main loop in a separate thread."""
        try:
            self._create_gui()
            logger.info("Dual-arm GUI created successfully")

            # Start GUI main loop
            self.root.mainloop()

        except Exception as e:
            logger.error(f"Error in GUI loop: {e}")
            self.running = False

    def start_server(self) -> None:
        """Start the dual-arm joint slider server."""
        if self.running:
            logger.info(f"Dual-Arm Joint Slider teleoperator {self.device_class_id} is already running")
            return

        logger.info(f"Starting Dual-Arm Joint Slider teleoperator {self.device_class_id} server...")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Start GUI in separate thread
        self.gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self.gui_thread.start()

        logger.info("Dual-arm GUI started in separate thread")
        logger.info("Use the sliders to control both robot arms")
        logger.info("Press Ctrl+C to stop...")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main data loop
        while self.running:
            try:
                # Generate current joint array
                joint_array = self._generate_joint_array()
                timestamp_ns = time.time_ns()
                self._write_array_to_shm_with_timestamp(joint_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()

            except Exception as e:
                logger.error(f"Error in dual-arm joint slider server: {e}")
                break

    def stop_server(self) -> None:
        """Stop the dual-arm joint slider server."""
        self.running = False

        # Close GUI
        if self.root:
            try:
                self.root.quit()
            except:
                pass

        # Wait for GUI thread to finish
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=2)

        # Clean up shared memory
        self._cleanup_shared_memory()

        logger.info("Dual-Arm Joint Slider server stopped")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            'device_name': self.device_name,
            'device_class_id': self.device_class_id,
            'fps': self.fps,
            'data_shape': self.data_shape,
            'data_dtype': self.data_dtype,
            'buffer_size': self.buffer_size,
            'hardware_latency_ms': self.hardware_latency_ms,
            'left_arm_joints': self.left_arm_joints,
            'right_arm_joints': self.right_arm_joints,
            'total_joints': self.total_joints,
            'left_joint_ranges': self.left_joint_ranges,
            'right_joint_ranges': self.right_joint_ranges,
            'current_left_joints': self.left_current_joints,
            'current_right_joints': self.right_current_joints
        }


class Airexoskeleton(BaseDevice):
    """
    Air exoskeleton device for reading joint angle data from physical exoskeleton.
    Reads joint angles and publishes to shared memory.
    """

    def __init__(self, device_class_id=0, *, state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 port="/dev/ttyUSB0", baudrate=1000000, encoder_ids=None,
                 ctrl_type = "joint",arm_type="left",scalar_first_quat_ctrl = False,
                 utilize_button=False, button_key='space',
                 airexo_type="ver1"):
        """
        Initialize the air exoskeleton device.

        Args:
            device_class_id: Unique identifier for this device instance
            fps: Update frequency in Hz (default: 100.0 to match encoder refresh rate)
            num_joints: Number of joints to read (default: 7)
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            port: Serial port for encoder communication (default: /dev/ttyUSB0)
            baudrate: Baudrate for serial communication (default: 1000000)
            encoder_ids: List of encoder IDs (default: [1,2,3,4,5,6,7])
            ctrl_type: Control type ("joint", "eef_delta", "eef_abs")
            arm_type: Arm type ("left" or "right")
            scalar_first_quat_ctrl: Whether to use scalar first quaternion control
            utilize_gripper: Whether to enable gripper support
            utilize_button: Whether to enable button control
            button_key: Key for button control
            airexo_type: Version of airexo ("ver1" or "ver2")
        """
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        utilize_gripper = any(
            primitive.name == "gripper"
            for primitive in self.state_feedback_config.format.primitives
        )
        num_joints = self.data_shape[0] - int(utilize_gripper)

        # Override device name
        self.device_name = "Airexoskeleton"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        # Exoskeleton attributes
        self.num_joints = num_joints
        self.port = port
        self.baudrate = baudrate
        self.encoder_ids = encoder_ids if encoder_ids else list(range(1, num_joints + 1))

        # End-effector control attributes
        self.ctrl_type = ctrl_type
        assert self.ctrl_type in ["joint", "eef_delta", "eef_abs"]
        self.arm_type = arm_type  # "left" or "right"

        # Store airexo type
        self.airexo_type = airexo_type

        # Button control support
        self.utilize_button = utilize_button
        if self.utilize_button:
            assert self.ctrl_type == "eef_delta", "Button control requires ctrl_type == 'eef_delta'"
            from utils.button import Button
            self.button = Button(button_key)

        self.state = True  # True: running, False: paused
        self.last_state = True
        self.detach_time = None  # Time when detach occurred
        self.resume_cooldown = 1.0  # Minimum time (seconds) between detach and resume
        # Pose storage for smooth transition
        self.detach_pos = None
        self.detach_quat = None
        self.resume_pos = None
        self.resume_quat = None

        # Gripper support
        self.utilize_gripper = utilize_gripper
        if self.utilize_gripper:
            assert self.data_shape[0] == 8, f"Gripper enabled requires data_shape[0] == 8, got {self.data_shape[0]}"
        else:
            assert self.data_shape[0] == 7, f"No gripper requires data_shape[0] == 7, got {self.data_shape[0]}"

        # Initialize gripper data storage
        self.gripper_data = 0.0
        # Initialize joint angle processor
        from utils.exoskeleton_utils import create_processor_for_arm
        self.angle_processor = create_processor_for_arm(
            arm_type, num_joints, self.data_dtype, airexo_type
        )
        self.scalar_first_quat_ctrl = scalar_first_quat_ctrl

        # Initialize kinematics if EEF is enabled
        if self.ctrl_type in ["eef_delta", "eef_abs"]:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from utils.exoskeleton_utils import ExoskeletonKinematics
                self.kinematics = ExoskeletonKinematics(arm_type, airexo_type)
                logger.info(f"Exoskeleton kinematics model initialized for EEF output with airexo_type: {airexo_type}.")
            except ImportError as e:
                logger.warning(f"Could not import kinematics module for EEF output: {e}")
                self.ctrl_type = "joint"

        # Current joint angles (initialize to zeros)
        self.current_angles = [0.0] * num_joints
        self.home_pos = None
        self.home_rot = None
        self.home_j = None

        # Encoder instance
        self.encoder = None
        self.encoder_connected = False

        # Threading
        self.running = False
        self.lock = threading.Lock()

        logger.info(f"Air Exoskeleton device initialized")
        logger.info(f"Device ID: {device_class_id}")
        logger.info(f"Number of joints: {num_joints}")
        logger.info(f"Port: {port}")
        logger.info(f"Baudrate: {baudrate}")
        logger.info(f"Encoder IDs: {self.encoder_ids}")

    def _test_port(self) -> bool:
        """Test if the serial port is accessible."""
        try:
            import serial
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                logger.info(f"Port {self.port} is accessible")
                return True
        except Exception as e:
            logger.error(f"Port {self.port} test failed: {e}")
            return False

    def _connect_encoder(self) -> bool:
        """Connect to the angle encoder."""
        max_wait_time = 3  # Increase wait time to 3 seconds
        max_retries = 2    # Add retry mechanism
        if not EXOSKELETON_AVAILABLE:
            logger.error("exoskeleton module not available")
            return False

        # Test port accessibility first
        if not self._test_port():
            logger.error(f"Cannot access port {self.port}")
            return False

        for retry in range(max_retries):
            try:
                if retry > 0:
                    logger.info(f"Retry {retry + 1}/{max_retries} connecting to encoder on {self.port}")
                    time.sleep(1)  # Wait between retries

                self.encoder = AngleEncoder(
                    ids=self.encoder_ids,
                    port=self.port,
                    baudrate=self.baudrate
                )

                # Test connection by reading once
                test_result = self.encoder.get_info()
                if test_result is not None:
                    state = test_result.get('state', None)
                    logger.info(f"Initial encoder state: {state}, full result: {test_result}")

                    if state == 1:  # OK status
                        self.encoder_connected = True
                        logger.info(f"Encoder connected successfully on {self.port}")
                        logger.info(f"Initial reading: {test_result}")
                        # Use angle processor to process home joint angles
                        self.home_j = self.angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                        if self.arm_type == "left":
                            self.home_pos, self.home_rot = self.kinematics.forward_kinematics_left(self.home_j.tolist())
                        elif self.arm_type == "right":
                            self.home_pos, self.home_rot = self.kinematics.forward_kinematics_right(self.home_j.tolist())
                        return True
                    else:
                        logger.info(f"Waiting for encoder to become ready (current state: {state})...")
                        t_s = time.time()
                        while time.time() < t_s + max_wait_time:
                            test_result = self.encoder.get_info()
                            if test_result is not None and isinstance(test_result, dict):
                                state = test_result.get('state', None)
                                if state == 1:
                                    self.encoder_connected = True
                                    logger.info(f"Encoder connected successfully on {self.port}")
                                    logger.info(f"Final reading: {test_result}")
                                    # Use angle processor to process home joint angles
                                    self.home_j = self.angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                                    if self.arm_type == "left":
                                        self.home_pos, self.home_rot = self.kinematics.forward_kinematics_left(
                                            self.home_j.tolist())
                                    elif self.arm_type == "right":
                                        self.home_pos, self.home_rot = self.kinematics.forward_kinematics_right(
                                            self.home_j.tolist())
                                    return True
                            time.sleep(0.2)  # Slightly longer delay

                        logger.warning(f"Encoder not ready after {max_wait_time}s wait")
                        logger.warning(f"Final state: {state} (expected 1 for OK)")
                        logger.warning(f"Port: {self.port}, Baudrate: {self.baudrate}, IDs: {self.encoder_ids}")
                        if test_result:
                            logger.warning(f"Final test result: {test_result}")
                else:
                    logger.error("Failed to get initial reading from encoder")

            except Exception as e:
                logger.error(f"Error connecting to encoder (attempt {retry + 1}): {e}")
                if retry == max_retries - 1:  # Last retry
                    return False

        return False

    def _disconnect_encoder(self) -> None:
        """Disconnect from the angle encoder."""
        if self.encoder:
            try:
                self.encoder.stop_streaming()
                self.encoder = None
                self.encoder_connected = False
                logger.info("Encoder disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting encoder: {e}")

    def _read_joint_angles(self) -> np.ndarray:
        """
        Read current joint angles from encoder.

        Returns:
            numpy.ndarray: Current joint angles
        """
        if not self.encoder_connected or not self.encoder:
            return self.angle_processor.get_zero_array()

        try:
            # Read angles from encoder - returns dict with 'state' and 'joint_pose'
            encoder_info = self.encoder.get_info()

            if encoder_info is not None and isinstance(encoder_info, dict):
                state = encoder_info.get('state', None)
                joint_pose = encoder_info.get('joint_pose', None)

                # Check if state is OK (assuming 1 means OK based on dual arm implementation)
                if state == 1 and joint_pose is not None and len(joint_pose) >= self.num_joints:
                    # Process joint angles (excluding gripper data if present)
                    if self.utilize_gripper and len(joint_pose) >= 8:
                        # Extract joint angles (first 7) and gripper data (last 1)
                        joint_data = joint_pose[:7]
                        gripper_data = joint_pose[7]
                        processed_angles = self.angle_processor.process_raw_angles(joint_data)
                        # Process gripper data using process_gripper function
                        from utils.exoskeleton_utils import process_gripper
                        self.gripper_data = process_gripper(gripper_data, self.arm_type, self.airexo_type)
                        logger.debug(f"Processed gripper data: raw={gripper_data}, processed={self.gripper_data}")
                    else:
                        # Process all data as joint angles
                        processed_angles = self.angle_processor.process_raw_angles(joint_pose)

                    # Combine data based on gripper usage (following DualAirexoskeleton pattern)
                    if not self.utilize_gripper:
                        result = processed_angles
                    else:
                        # Add gripper data if enabled
                        assert hasattr(self, 'gripper_data'), "Gripper data not found"
                        result = np.concatenate([processed_angles, [self.gripper_data]])

                    with self.lock:
                        self.current_angles = result.tolist()

                    return result.copy()
                else:
                    # Invalid state or data, set to zero
                    zero_array = self.angle_processor.get_zero_array()
                    with self.lock:
                        self.current_angles = zero_array.tolist()
                    if state != 1:
                        logger.error(f"Encoder state error: {state}")
                    return zero_array
            else:
                logger.warning("Invalid response format from encoder (expected dict with 'state' and 'joint_pose')")
                return self.angle_processor.get_zero_array()

        except Exception as e:
            logger.error(f"Error reading joint angles: {e}")
            return self.angle_processor.get_zero_array()

    def _compute_end_effector_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose

        Args:
            joint_angles: Joint angle array (may include gripper data)

        Returns:
            End-effector pose [x, y, z, qx, qy, qz, qw] + [gripper_width] if gripper enabled
        """
        if not self.ctrl_type in ["eef_abs","eef_delta"] or not hasattr(self, 'kinematics'):
            if self.utilize_gripper:
                return np.zeros(8, dtype=get_dtype(self.data_dtype))  # 7 elements for pos + quat + 1 for gripper
            else:
                return np.zeros(7, dtype=get_dtype(self.data_dtype))  # 7 elements for pos + quat

        try:
            # Extract joint angles (first 7 elements) for kinematics computation
            # joint_angles may include gripper data as the last element
            if self.utilize_gripper and len(joint_angles) > 7:
                joint_angles_for_kinematics = joint_angles[:7]
            else:
                joint_angles_for_kinematics = joint_angles

            # Use kinematics to compute end-effector pose
            if self.arm_type == "left":
                pos, rot = self.kinematics.forward_kinematics_left(joint_angles_for_kinematics.tolist())
            elif self.arm_type == "right":
                pos, rot = self.kinematics.forward_kinematics_right(joint_angles_for_kinematics.tolist())
            else:
                raise ValueError(f"Invalid arm_type: {self.arm_type}")

            # Convert rotation matrix to quaternion
            quaternion = R.from_matrix(rot).as_quat() #scaler_last
            if self.scalar_first_quat_ctrl:
                quaternion = np.roll(quaternion, 1)

            if self.ctrl_type == "eef_delta":
                # Handle button state transitions
                if self.utilize_button:
                    # Check state transitions
                    if self.last_state and not self.state:  # running -> paused
                        if self.detach_pos is not None:
                            self.home_rot = (R.from_quat(self.resume_quat) * R.from_quat(self.detach_quat).inv() * R.from_matrix(self.home_rot)).as_matrix()
                            if self.arm_type == "left":
                                self.home_pos = self.home_pos + self.resume_pos - self.detach_pos
                            else:
                                self.home_pos = self.home_pos + self.resume_pos - self.detach_pos

                        self.detach_pos = pos.copy()
                        self.detach_quat = quaternion.copy()
                        self.detach_time = time.time()  # Record detach time

                        self.resume_pos = None
                        self.resume_quat = None
                        logger.info(f"Recorded detach pose: pos={self.detach_pos}, quat={self.detach_quat}")
                    elif not self.last_state and self.state:  # paused -> running
                        detach_dt = time.time() - self.detach_time
                        # Check cooldown time
                        if self.detach_time is None or detach_dt >= self.resume_cooldown:
                            self.resume_pos = pos.copy()
                            self.resume_quat = quaternion.copy()
                            logger.info(f"Recorded resume pose: pos={self.resume_pos}, quat={self.resume_quat}")
                        else:
                            remaining_time = self.resume_cooldown - detach_dt
                            logger.debug(f"Resume cooldown active, {remaining_time:.2f}s remaining")
                    # Use compute_deviation_3 after resume if we have all poses
                    if (self.detach_pos is not None and self.resume_pos is not None):
                        home_rot_quat = R.from_matrix(self.home_rot).as_quat()
                        _ , quaternion = compute_deviation_3(
                            None, home_rot_quat,
                            None, self.detach_quat,
                            None, self.resume_quat,
                            None, quaternion
                        )
                        if self.arm_type == "left":
                            pos = self.kinematics.wb_R_left.apply(pos - self.home_pos + self.detach_pos - self.resume_pos)
                        else:
                            pos = self.kinematics.wb_R_right.apply(pos - self.home_pos + self.detach_pos - self.resume_pos)
                    else:
                        # Fall back to original compute_deviation
                        home_rot_quat = R.from_matrix(self.home_rot).as_quat()
                        _ , quaternion = compute_deviation(None, home_rot_quat, None, quaternion, first_inv=True)
                        pos = self.kinematics.wb_R_left.apply(pos - self.home_pos) if self.arm_type == "left" else self.kinematics.wb_R_right.apply(pos - self.home_pos)
                else:
                    # Normal computation without button
                    home_rot_quat = R.from_matrix(self.home_rot).as_quat()
                    _ , quaternion = compute_deviation(None, home_rot_quat, None, quaternion, first_inv=True)
                    pos = self.kinematics.wb_R_left.apply(pos - self.home_pos) if self.arm_type == "left" else self.kinematics.wb_R_right.apply(pos - self.home_pos)

            # Combine position and quaternion
            eef_pose = np.concatenate([pos, quaternion])

            # Add gripper data if enabled (following DualAirexoskeleton pattern)
            if not self.utilize_gripper:
                # Return pose without gripper data
                return eef_pose.astype(get_dtype(self.data_dtype))
            else:
                # Add gripper data if enabled
                assert hasattr(self, 'gripper_data'), "Gripper data not found"
                logger.debug(f"Using self.gripper_data: {self.gripper_data}")
                eef_pose = np.concatenate([eef_pose, [self.gripper_data]])
                return eef_pose.astype(get_dtype(self.data_dtype))

        except Exception as e:
            logger.error(f"Error computing end-effector pose: {e}")
            return np.zeros(7, dtype=get_dtype(self.data_dtype))

    def start_server(self) -> None:
        """Start the air exoskeleton device server."""
        if self.running:
            logger.info(f"Air Exoskeleton device {self.device_class_id} is already running")
            return

        logger.info(f"Starting Air Exoskeleton device {self.device_class_id} server...")
        self.running = True

        # Start button listener if enabled
        if self.utilize_button:
            self.button.start_listener()
            logger.info(f"Button listener started for key: {self.button.key}")

        # Connect to encoder
        if not self._connect_encoder():
            raise RuntimeError("Failed to connect to exoskeleton encoder.")

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")



        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main data reading loop
        while self.running:
            try:
                self.last_state = self.state
                # Check button press and update state
                if self.utilize_button:
                    if self.button.is_pressed():
                        logger.info(f"Button pressed: {'PAUSED' if not self.state else 'RUNNING'}")
                        self.state = not self.state
                # Read current joint angles
                angle_array = self._read_joint_angles() # range from -pi to pi
                # print(f"angle_array: {angle_array}")
                timestamp_ns = time.time_ns()
                if not self.utilize_button or (self.utilize_button and (self.state or self.last_state)):
                    # Write end-effector pose to SHM if EEF control is enabled
                    if self.ctrl_type in ["eef_delta", "eef_abs"]:
                        eef_pose = self._compute_end_effector_pose(angle_array)
                        logger.debug(f"Writing EEF pose to SHM: {eef_pose}, length: {len(eef_pose)}")
                        self._write_array_to_shm_with_timestamp(eef_pose, timestamp_ns)
                        # print(f"eef_pose: {eef_pose}")
                    else:
                        # Write joint angles to SHM (original behavior)
                        self._write_array_to_shm_with_timestamp(angle_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()

            except Exception as e:
                logger.error(f"Error in data reading: {e}")
                break

    def stop_server(self) -> None:
        """Stop the air exoskeleton device server."""
        if not self.running:
            return

        logger.info(f"Stopping Air Exoskeleton device {self.device_class_id} server...")
        self.running = False

        # Stop button listener if enabled
        if self.utilize_button:
            self.button.stop_listener()
            logger.info("Button listener stopped")

        # Disconnect encoder
        self._disconnect_encoder()

        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str,Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "num_joints": self.num_joints,
            "port": self.port,
            "baudrate": self.baudrate,
            "encoder_ids": self.encoder_ids,
            "current_angles": self.current_angles.copy(),
            "encoder_connected": self.encoder_connected
        })
        return info


class DualAirexoskeleton(BaseDevice):
    """
    Dual air exoskeleton device for reading joint angle data from two physical exoskeletons.
    Reads joint angles from both arms and publishes to shared memory.
    """

    def __init__(self, device_class_id=0, *, state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 left_port="/dev/ttyUSB0", right_port="/dev/ttyUSB1",
                 baudrate=1000000, left_encoder_ids=None, right_encoder_ids=None,
                 ctrl_type = "joint",scalar_first_quat_ctrl = False,
                 utilize_button=False, button_key='space',
                 airexo_type="ver1"):
        """
        Initialize the dual air exoskeleton device.

        Args:
            device_class_id: Unique identifier for this device instance
            fps: Update frequency in Hz (default: 100.0 to match encoder refresh rate)
            left_arm_joints: Number of joints in left arm (default: 7)
            right_arm_joints: Number of joints in right arm (default: 7)
            data_dtype: Data type for joint data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            left_port: Serial port for left arm encoder (default: /dev/ttyUSB0)
            right_port: Serial port for right arm encoder (default: /dev/ttyUSB1)
            baudrate: Baudrate for serial communication (default: 1000000)
            left_encoder_ids: List of encoder IDs for left arm (default: [1,2,3,4,5,6,7])
            right_encoder_ids: List of encoder IDs for right arm (default: [1,2,3,4,5,6,7])
            ctrl_type: Control type ("joint", "eef_delta", "eef_abs")
            scalar_first_quat_ctrl: Whether to use scalar first quaternion control
            utilize_gripper: Whether to enable gripper support
            utilize_button: Whether to enable button control
            button_key: Key for button control
            airexo_type: Version of airexo ("ver1" or "ver2")
        """
        # Initialize base device with joint angle data for both arms
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        gripper_count = sum(
            primitive.name == "gripper"
            for primitive in self.state_feedback_config.format.primitives
        )
        total_joints = self.data_shape[0] - gripper_count
        if total_joints % 2:
            raise ValueError("DualAirexoskeleton state_feedback must contain two equal arms")
        left_arm_joints = right_arm_joints = total_joints // 2
        utilize_gripper = gripper_count > 0

        # Override device name
        self.device_name = "DualAirexoskeleton"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        # Exoskeleton attributes
        self.left_arm_joints = left_arm_joints
        self.right_arm_joints = right_arm_joints
        self.total_joints = total_joints
        self.left_port = left_port
        self.right_port = right_port
        self.baudrate = baudrate

        # Encoder IDs
        self.left_encoder_ids = left_encoder_ids if left_encoder_ids else list(range(1, left_arm_joints + 1))
        self.right_encoder_ids = right_encoder_ids if right_encoder_ids else list(range(1, right_arm_joints + 1))

        # End-effector control attributes
        self.ctrl_type = ctrl_type
        assert self.ctrl_type in ["joint", "eef_delta", "eef_abs"]

        # Gripper support
        self.utilize_gripper = utilize_gripper
        if self.utilize_gripper:
            assert self.data_shape[0] == 16, f"Gripper enabled requires data_shape[0] == 16, got {self.data_shape[0]}"
        else:
            assert self.data_shape[0] == 14, f"No gripper requires data_shape[0] == 14, got {self.data_shape[0]}"

        # Store airexo type
        self.airexo_type = airexo_type

        # Initialize joint angle processors for each arm separately
        from utils.exoskeleton_utils import create_processor_for_arm
        self.left_angle_processor = create_processor_for_arm(
            "left", left_arm_joints, self.data_dtype, airexo_type
        )
        self.right_angle_processor = create_processor_for_arm(
            "right", right_arm_joints, self.data_dtype, airexo_type
        )
        self.scalar_first_quat_ctrl = scalar_first_quat_ctrl

        # Initialize kinematics if EEF is enabled
        if self.ctrl_type in ["eef_delta", "eef_abs"]:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from utils.exoskeleton_utils import ExoskeletonKinematics
                self.kinematics = ExoskeletonKinematics("all", airexo_type)
                logger.info(f"Dual arm exoskeleton kinematics model initialized for EEF output with airexo_type: {airexo_type}.")
            except ImportError as e:
                logger.warning(f"Could not import kinematics module for dual EEF output: {e}")
                self.ctrl_type = "joint"

        # Current joint angles (initialize to zeros)
        self.left_current_angles = [0.0] * left_arm_joints
        self.right_current_angles = [0.0] * right_arm_joints

        # Gripper data storage
        self.left_gripper_data = 0.0
        self.right_gripper_data = 0.0
        self.home_pos_left = None
        self.home_rot_left = None
        self.home_j_left = None
        self.home_pos_right = None
        self.home_rot_right = None
        self.home_j_right = None

        # Encoder instances
        self.left_encoder = None
        self.right_encoder = None
        self.left_encoder_connected = False
        self.right_encoder_connected = False

        # Button control configuration
        self.utilize_button = utilize_button
        self.button_key = button_key

        # State management (always present, independent of utilize_button)
        self.state = True  # True = running, False = paused
        self.last_state = True
        self.detach_time = None  # Time when detach occurred
        self.resume_cooldown = 1.0  # Minimum time (seconds) between detach and resume

        # Pose storage for smooth transitions (always present)
        self.left_detach_pos = None
        self.left_detach_quat = None
        self.left_resume_pos = None
        self.left_resume_quat = None

        self.right_detach_pos = None
        self.right_detach_quat = None
        self.right_resume_pos = None
        self.right_resume_quat = None

        # Initialize button only if enabled
        if self.utilize_button:
            # Assert that ctrl_type must be eef_delta for button control
            assert self.ctrl_type == "eef_delta", f"Button control requires ctrl_type='eef_delta', got '{self.ctrl_type}'"

            # Import and initialize button
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.button import Button
            self.button = Button(self.button_key)

            logger.info(f"Button control enabled with key: {self.button_key}")

        # Threading
        self.running = False
        self.lock = threading.Lock()
        self.bias = np.array([126.03, 251.19, 323.26, 281.4, 314.38, 114.15, 61.52, 352.52, 152.57, 72.15, 352.4, 277.20, 358.7, 272.8])

        logger.info(f"Dual Air Exoskeleton device initialized")
        logger.info(f"Device ID: {device_class_id}")
        logger.info(f"Left arm joints: {left_arm_joints}")
        logger.info(f"Right arm joints: {right_arm_joints}")
        logger.info(f"Total joints: {total_joints}")
        logger.info(f"Left port: {left_port}")
        logger.info(f"Right port: {right_port}")
        logger.info(f"Baudrate: {baudrate}")
        logger.info(f"Left encoder IDs: {self.left_encoder_ids}")
        logger.info(f"Right encoder IDs: {self.right_encoder_ids}")

    def _connect_encoders(self) -> Tuple[bool,bool]:
        """Connect to both angle encoders."""
        max_wait_time = 3  # Increased wait time for encoder initialization
        if not EXOSKELETON_AVAILABLE:
            logger.error("exoskeleton module not available")
            return False, False

        left_connected = False
        right_connected = False

        # Connect left encoder
        try:
            self.left_encoder = AngleEncoder(
                ids=self.left_encoder_ids,
                port=self.left_port,
                baudrate=self.baudrate
            )
            # Test connection by reading once
            test_result = self.left_encoder.get_info()
            if test_result is not None and isinstance(test_result, dict):
                state = test_result.get('state', None)
                if state == 1:
                    self.left_encoder_connected = True
                    left_connected = True
                    logger.info(f"Left encoder connected successfully on {self.left_port}")
                    logger.info(f"Left initial reading: {test_result}")
                    # Use left arm angle processor for home position
                    self.home_j_left = self.left_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                    self.home_pos_left, self.home_rot_left = self.kinematics.forward_kinematics_left(
                        self.home_j_left.tolist())

                else:
                    t_s = time.time()
                    while time.time() < t_s + max_wait_time:
                        test_result = self.left_encoder.get_info()
                        if test_result is not None and isinstance(test_result, dict):
                            state = test_result.get('state', None)
                            if state == 1:
                                left_connected = True
                                self.left_encoder_connected = True
                                logger.info(f"Left encoder connected successfully on {self.left_port}")
                                logger.info(f"Left initial reading: {test_result}")
                                # Use left arm angle processor for home position
                                self.home_j_left = self.left_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                                self.home_pos_left, self.home_rot_left = self.kinematics.forward_kinematics_left(self.home_j_left.tolist())

                                break
                        time.sleep(0.1)  # Small delay to avoid busy waiting
                    if not self.left_encoder_connected:
                        logger.warning(f"Left encoder not ready, state: {state} (expected 1 for OK)")
            else:
                logger.error("Failed to get initial reading from left encoder")

        except Exception as e:
            logger.error(f"Error connecting to left encoder: {e}")

        # Connect right encoder
        try:
            self.right_encoder = AngleEncoder(
                ids=self.right_encoder_ids,
                port=self.right_port,
                baudrate=self.baudrate
            )
            # Test connection by reading once
            test_result = self.right_encoder.get_info()
            if test_result is not None and isinstance(test_result, dict):
                state = test_result.get('state', None)
                if state == 1:
                    self.right_encoder_connected = True
                    right_connected = True
                    logger.info(f"Right encoder connected successfully on {self.right_port}")
                    logger.info(f"Right initial reading: {test_result}")
                    # Use right arm angle processor for home position
                    self.home_j_right = self.right_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                    self.home_pos_right, self.home_rot_right = self.kinematics.forward_kinematics_right(
                        self.home_j_right.tolist())
                else:
                    t_s = time.time()
                    while time.time() < t_s + max_wait_time:
                        test_result = self.right_encoder.get_info()
                        if test_result is not None and isinstance(test_result, dict):
                            state = test_result.get('state', None)
                            if state == 1:
                                right_connected = True
                                self.right_encoder_connected = True
                                logger.info(f"Right encoder connected successfully on {self.right_port}")
                                logger.info(f"Right initial reading: {test_result}")
                                # Use right arm angle processor for home position
                                self.home_j_right = self.right_angle_processor.process_raw_angles_for_home(test_result["joint_pose"])
                                self.home_pos_right, self.home_rot_right = self.kinematics.forward_kinematics_right(
                                    self.home_j_right.tolist())
                                break
                        time.sleep(0.1)  # Small delay to avoid busy waiting
                    if not self.right_encoder_connected:
                        logger.warning(f"Right encoder not ready, state: {state} (expected 1 for OK)")
            else:
                logger.error("Failed to get initial reading from right encoder")

        except Exception as e:
            logger.error(f"Error connecting to right encoder: {e}")

        return left_connected, right_connected

    def _disconnect_encoders(self) -> None:
        """Disconnect from both angle encoders."""
        # Disconnect left encoder
        if self.left_encoder:
            try:
                self.left_encoder.stop_streaming()
                self.left_encoder = None
                self.left_encoder_connected = False
                logger.info("Left encoder disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting left encoder: {e}")

        # Disconnect right encoder
        if self.right_encoder:
            try:
                self.right_encoder.stop_streaming()
                self.right_encoder = None
                self.right_encoder_connected = False
                logger.info("Right encoder disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting right encoder: {e}")

    def _read_joint_angles(self) -> None:
        """
        Read current joint angles from both encoders.

        Returns:
            numpy.ndarray: Current joint angles [left_arm_joints, right_arm_joints]
        """
        left_angles = np.zeros(self.left_arm_joints, dtype=get_dtype(self.data_dtype))
        right_angles = np.zeros(self.right_arm_joints, dtype=get_dtype(self.data_dtype))

        # Read left encoder
        if self.left_encoder_connected and self.left_encoder:
            try:
                encoder_info = self.left_encoder.get_info()
                if encoder_info is not None and isinstance(encoder_info, dict):
                    state = encoder_info.get('state', None)
                    joint_pose = encoder_info.get('joint_pose', None)

                    # Check if state is OK (assuming 1 means OK based on the example)
                    if state == 1 and joint_pose is not None and len(joint_pose) >= self.left_arm_joints:
                        # Process left arm joint angles (excluding gripper data if present)
                        if self.utilize_gripper and len(joint_pose) >= 8:
                            # Extract joint angles (first 7) and gripper data (last 1)
                            left_joint_data = joint_pose[:7]
                            raw_left_gripper_data = joint_pose[7]
                            left_processed = self.left_angle_processor.process_raw_angles(left_joint_data)
                            # Process gripper data using process_gripper function
                            from utils.exoskeleton_utils import process_gripper
                            self.left_gripper_data = process_gripper(raw_left_gripper_data, "left", self.airexo_type)
                        else:
                            # Process all data as joint angles
                            left_processed = self.left_angle_processor.process_raw_angles(joint_pose)

                        with self.lock:
                            self.left_current_angles = left_processed.tolist()
                            left_angles[:] = left_processed
                    else:
                        # Invalid state or data, set to zero
                        with self.lock:
                            zero_array = np.zeros(self.left_arm_joints, dtype=get_dtype(self.data_dtype))
                            self.left_current_angles = zero_array.tolist()
                            left_angles.fill(0.0)
                        if state != 1:
                            logger.error(f"Left encoder state error: {state}")
            except Exception as e:
                logger.error(f"Error reading left encoder: {e}")

        # Read right encoder
        if self.right_encoder_connected and self.right_encoder:
            try:
                encoder_info = self.right_encoder.get_info()
                if encoder_info is not None and isinstance(encoder_info, dict):
                    state = encoder_info.get('state', None)
                    joint_pose = encoder_info.get('joint_pose', None)

                    # Check if state is OK (assuming 1 means OK based on the example)
                    if state == 1 and joint_pose is not None and len(joint_pose) >= self.right_arm_joints:
                        # Process right arm joint angles (excluding gripper data if present)
                        if self.utilize_gripper and len(joint_pose) >= 8:
                            # Extract joint angles (first 7) and gripper data (last 1)
                            right_joint_data = joint_pose[:7]
                            raw_right_gripper_data = joint_pose[7]
                            right_processed = self.right_angle_processor.process_raw_angles(right_joint_data)
                            # Process gripper data using process_gripper function
                            from utils.exoskeleton_utils import process_gripper
                            self.right_gripper_data = process_gripper(raw_right_gripper_data, "right", self.airexo_type)
                        else:
                            # Process all data as joint angles
                            right_processed = self.right_angle_processor.process_raw_angles(joint_pose)

                        with self.lock:
                            self.right_current_angles = right_processed.tolist()
                            right_angles[:] = right_processed
                    else:
                        # Invalid state or data, set to zero
                        with self.lock:
                            zero_array = np.zeros(self.right_arm_joints, dtype=get_dtype(self.data_dtype))
                            self.right_current_angles = zero_array.tolist()
                            right_angles.fill(0.0)
                        if state != 1:
                            logger.error(f"Right encoder state error: {state}")
            except Exception as e:
                logger.error(f"Error reading right encoder: {e}")

        # Combine both arms' data
        if not self.utilize_gripper:
            all_angles = np.concatenate([left_angles, right_angles])
        else:
            # Add gripper data if enabled
            assert hasattr(self, 'left_gripper_data') and hasattr(self, 'right_gripper_data'), "Gripper data not found"
            all_angles = np.concatenate([left_angles, [self.left_gripper_data], right_angles, [self.right_gripper_data]])

        return all_angles

    def _compute_dual_end_effector_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute dual-arm end-effector poses

        Args:
            joint_angles: Dual-arm joint angle array [left_arm_joints + right_arm_joints]

        Returns:
            Dual-arm end-effector poses [left_x, left_y, left_z, left_qx, left_qy, left_qz, left_qw,
                                        right_x, right_y, right_z, right_qx, right_qy, right_qz, right_qw]
        """
        if not self.ctrl_type in ["eef_abs","eef_delta"] or not hasattr(self, 'kinematics'):
            if self.utilize_gripper:
                return np.zeros(16, dtype=get_dtype(self.data_dtype))  # 7 elements per arm + 2 gripper
            else:
                return np.zeros(14, dtype=get_dtype(self.data_dtype))  # 7 elements per arm

        try:
            # Separate left and right arm joint angles
            left_angles = joint_angles[:self.left_arm_joints]
            right_angles = joint_angles[self.left_arm_joints:] if not self.utilize_gripper else joint_angles[self.left_arm_joints+1:self.left_arm_joints+self.right_arm_joints+1]

            # Compute left arm end-effector pose
            left_pos, left_rot = self.kinematics.forward_kinematics_left(left_angles.tolist())
            left_quaternion = R.from_matrix(left_rot).as_quat() #scaler_last
            if self.scalar_first_quat_ctrl:
                left_quaternion = np.roll(left_quaternion,1)
            if self.ctrl_type == "eef_delta":
                # Handle button state transitions for left arm
                if self.utilize_button:
                    # Check state transitions for left arm
                    if self.last_state and not self.state:  # running -> paused
                        if self.left_detach_pos is not None:
                            # Update home pose using resume and detach poses
                            self.home_rot_left = (R.from_quat(self.left_resume_quat) * R.from_quat(self.left_detach_quat).inv() * R.from_matrix(self.home_rot_left)).as_matrix()
                            self.home_pos_left = self.home_pos_left + self.left_resume_pos - self.left_detach_pos

                        self.left_detach_pos = left_pos.copy()
                        self.left_detach_quat = left_quaternion.copy()
                        self.detach_time = time.time()  # Record detach time

                        self.left_resume_pos = None
                        self.left_resume_quat = None
                        logger.info(f"Recorded left arm detach pose: pos={self.left_detach_pos}, quat={self.left_detach_quat}")
                    elif not self.last_state and self.state:  # paused -> running
                        detach_dt = time.time() - self.detach_time
                        # Check cooldown time
                        if self.detach_time is None or detach_dt >= self.resume_cooldown:
                            self.left_resume_pos = left_pos.copy()
                            self.left_resume_quat = left_quaternion.copy()
                            logger.info(f"Recorded left arm resume pose: pos={self.left_resume_pos}, quat={self.left_resume_quat}")
                        else:
                            remaining_time = self.resume_cooldown - detach_dt
                            logger.debug(f"Resume cooldown active, {remaining_time:.2f}s remaining")

                    # Use compute_deviation_3 after resume if we have all poses for left arm
                    if (self.left_detach_pos is not None and self.left_resume_pos is not None):
                        home_rot_left_quat = R.from_matrix(self.home_rot_left).as_quat()
                        _ , left_quaternion = compute_deviation_3(
                            None, home_rot_left_quat,
                            None, self.left_detach_quat,
                            None, self.left_resume_quat,
                            None, left_quaternion
                        )
                        left_pos = self.kinematics.wb_R_left.apply(left_pos - self.home_pos_left + self.left_detach_pos - self.left_resume_pos)

                    else:
                        # Fall back to original compute_deviation
                        home_rot_left_quat = R.from_matrix(self.home_rot_left).as_quat()
                        _ , left_quaternion = compute_deviation(None, home_rot_left_quat, None, left_quaternion, first_inv=True)
                        left_pos = self.kinematics.wb_R_left.apply(left_pos - self.home_pos_left)
                else:
                    # Normal computation without button
                    home_rot_left_quat = R.from_matrix(self.home_rot_left).as_quat()
                    _ , left_quaternion = compute_deviation(None, home_rot_left_quat, None, left_quaternion, first_inv=True)
                    left_pos = self.kinematics.wb_R_left.apply(left_pos - self.home_pos_left)
            left_pose = np.concatenate([left_pos, left_quaternion])

            # Compute right arm end-effector pose
            right_pos, right_rot = self.kinematics.forward_kinematics_right(right_angles.tolist())
            right_quaternion = R.from_matrix(right_rot).as_quat() #scaler_last
            if self.scalar_first_quat_ctrl:
                right_quaternion = np.roll(right_quaternion,1)
            if self.ctrl_type == "eef_delta":
                # Handle button state transitions for right arm
                if self.utilize_button:
                    # Check state transitions for right arm
                    if self.last_state and not self.state:  # running -> paused
                        if self.right_detach_pos is not None:
                            # Update home pose using resume and detach poses
                            self.home_rot_right = (R.from_quat(self.right_resume_quat) * R.from_quat(self.right_detach_quat).inv() * R.from_matrix(self.home_rot_right)).as_matrix()
                            self.home_pos_right = self.home_pos_right + self.right_resume_pos - self.right_detach_pos

                        self.right_detach_pos = right_pos.copy()
                        self.right_detach_quat = right_quaternion.copy()
                        self.right_resume_pos = None
                        self.right_resume_quat = None
                        # detach_time already set by left arm processing
                        logger.info(f"Recorded right arm detach pose: pos={self.right_detach_pos}, quat={self.right_detach_quat}")
                    elif not self.last_state and self.state:  # paused -> running
                        # Check cooldown time (same as left arm)
                        detach_dt = time.time() - self.detach_time
                        if self.detach_time is None or detach_dt >= self.resume_cooldown:
                            self.right_resume_pos = right_pos.copy()
                            self.right_resume_quat = right_quaternion.copy()
                            logger.info(f"Recorded right arm resume pose: pos={self.right_resume_pos}, quat={self.right_resume_quat}")
                        else:
                            remaining_time = self.resume_cooldown - detach_dt
                            logger.debug(f"Right arm resume cooldown active, {remaining_time:.2f}s remaining")

                    # Use compute_deviation_3 after resume if we have all poses for right arm
                    if (self.right_detach_pos is not None and
                        self.right_resume_pos is not None):
                        home_rot_right_quat = R.from_matrix(self.home_rot_right).as_quat()
                        _ , right_quaternion = compute_deviation_3(
                            None, home_rot_right_quat,
                            None, self.right_detach_quat,
                            None, self.right_resume_quat,
                            None, right_quaternion
                        )
                        right_pos = self.kinematics.wb_R_right.apply(right_pos - self.home_pos_right + self.right_detach_pos - self.right_resume_pos)

                    else:
                        # Fall back to original compute_deviation
                        home_rot_right_quat = R.from_matrix(self.home_rot_right).as_quat()
                        _ , right_quaternion = compute_deviation(None, home_rot_right_quat, None, right_quaternion, first_inv=True)
                        right_pos = self.kinematics.wb_R_right.apply(right_pos - self.home_pos_right)
                else:
                    # Normal computation without button
                    home_rot_right_quat = R.from_matrix(self.home_rot_right).as_quat()
                    _ , right_quaternion = compute_deviation(None, home_rot_right_quat, None, right_quaternion, first_inv=True)
                    right_pos = self.kinematics.wb_R_right.apply(right_pos - self.home_pos_right)
            right_pose = np.concatenate([right_pos, right_quaternion])

            # Combine dual-arm poses
            if not self.utilize_gripper:
                # Combine dual-arm poses [left_pose(7) + right_pose(7)]
                dual_pose = np.concatenate([left_pose, right_pose])
            else:
                # Add gripper data if enabled
                assert hasattr(self, 'left_gripper_data') and hasattr(self, 'right_gripper_data'), "Gripper data not found"
                dual_pose = np.concatenate([left_pose, [self.left_gripper_data], right_pose, [self.right_gripper_data]])

            return dual_pose.astype(get_dtype(self.data_dtype))

        except Exception as e:
            logger.error(f"Error computing dual end-effector pose: {e}")
            return np.zeros(14, dtype=get_dtype(self.data_dtype))


    def start_server(self):
        """Start the dual air exoskeleton device server."""
        if self.running:
            logger.info(f"Dual Air Exoskeleton device {self.device_class_id} is already running")
            return

        logger.info(f"Starting Dual Air Exoskeleton device {self.device_class_id} server...")
        self.running = True

        # Start button listener if enabled
        if self.utilize_button:
            self.button.start_listener()
            logger.info(f"Button listener started for key: {self.button.key}")

        # Connect to encoders
        left_connected, right_connected = self._connect_encoders()
        if left_connected and right_connected:
            logger.info("Both encoders connected successfully.")
        else:
            if not left_connected and right_connected:
                logger.error("Failed to connect to left encoder.")
            elif not right_connected and left_connected:
                logger.error("Failed to connect to right encoder.")
            else:
                logger.error("Failed to connect to any encoder.")
            raise RuntimeError("Failed to connect to all exoskeleton encoders.")

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")



        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main data reading loop
        while self.running:
            try:
                self.last_state = self.state
                # Check button press and update state
                if self.utilize_button:
                    if self.button.is_pressed():
                        logger.info(f"Button pressed: {'PAUSED' if not self.state else 'RUNNING'}")
                        self.state = not self.state
                    else:
                        self.state = self.last_state

                # Read current joint angles
                angle_array = self._read_joint_angles()

                timestamp_ns = time.time_ns()

                if not self.utilize_button or (self.utilize_button and (self.state or self.last_state)):
                    # Write dual end-effector pose to SHM if EEF control is enabled
                    if self.ctrl_type in ["eef_abs","eef_delta"]:
                        dual_eef_pose = self._compute_dual_end_effector_pose(angle_array)
                        self._write_array_to_shm_with_timestamp(dual_eef_pose, timestamp_ns)
                    else:
                        # Write joint angles to SHM (original behavior)
                        self._write_array_to_shm_with_timestamp(angle_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()

            except Exception as e:
                logger.error(f"Error in data reading: {e}")
                break

    def stop_server(self) -> None:
        """Stop the dual air exoskeleton device server."""
        if not self.running:
            return

        logger.info(f"Stopping Dual Air Exoskeleton device {self.device_class_id} server...")
        self.running = False

        # Stop button listener if enabled
        if self.utilize_button:
            self.button.stop_listener()
            logger.info("Button listener stopped")

        # Disconnect encoders
        self._disconnect_encoders()

        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str,Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "left_arm_joints": self.left_arm_joints,
            "right_arm_joints": self.right_arm_joints,
            "total_joints": self.total_joints,
            "left_port": self.left_port,
            "right_port": self.right_port,
            "baudrate": self.baudrate,
            "left_encoder_ids": self.left_encoder_ids,
            "right_encoder_ids": self.right_encoder_ids,
            "left_current_angles": self.left_current_angles.copy(),
            "right_current_angles": self.right_current_angles.copy(),
            "left_encoder_connected": self.left_encoder_connected,
            "right_encoder_connected": self.right_encoder_connected
        })
        return info


class KeyboardTeleoperator(BaseDevice):
    """
    Keyboard-based dual-arm teleoperator for delta EEF control.

    Uses keyboard input to generate delta end-effector poses for dual-arm robot control.
    Outputs position deltas and gripper states to shared memory.

    Key Mapping:
        Left Arm:  W/S = +/-Y, A/D = -/+X, Q/E = -/+Z, F = gripper toggle
        Right Arm: I/K = +/-Y, J/L = -/+X, U/O = -/+Z, H = gripper toggle

    """

    def __init__(self, device_class_id: int = 0, *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 delta_step: float = 0.01,
                 robot_device_name: str = "RizonRobot",
                 robot_device_class_id: int = 1,
                 max_gripper_width: float = 0.085,
                 signal_files: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize the keyboard teleoperator with absolute EEF control.

        Args:
            device_class_id: Unique identifier for this device instance
            fps: Update frequency in Hz
            data_dtype: Data type for output data
            buffer_size: Number of frames to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds
            delta_step: Position delta per frame when key is held (meters)
            robot_device_name: Name of robot device for SHM connection (for initial pose)
            robot_device_class_id: Device ID of robot for SHM connection
            max_gripper_width: Maximum gripper width in meters
        """
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        if self.data_shape != (16,):
            raise ValueError("KeyboardTeleoperator state_feedback format must have shape [16]")

        # Override device name
        self.device_name = "KeyboardTeleoperator"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        # Extract signal file patterns from config if provided
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._signal_files = signal_files
        self._policy_runner_reset_signal_path = signal_files.get('policy_runner_reset_command')
        self._teleoperator_reset_signal_path = signal_files.get('teleoperator_reset_command')
        self._reset_signal_path = signal_files.get('reset_signal')

        # This producer always owns and publishes its selector flag stream.
        self._human_in_control = False  # False=policy, True=human
        self._control_flag_shm = None
        self._control_flag_shm_name = f"{self.device_name}_{self.device_class_id}_control_flag"
        self._control_flag_buffer_size = 100  # Buffer size for control flag SHM

        # Policy runner reset signal (from PolicyConnector when episode ends)
        self._policy_runner_reset_signal = Path(self._policy_runner_reset_signal_path)
        self._last_policy_runner_reset_mtime = 0.0

        # Teleoperator reset signal - triggered by PolicyConnector when new episode starts
        # This ensures new episode always starts with policy control
        self._teleoperator_reset_signal = Path(self._teleoperator_reset_signal_path)
        self._last_teleoperator_reset_mtime = 0.0

        # Control parameters
        self.delta_step = delta_step
        self.max_gripper_width = max_gripper_width

        # Robot SHM connection for initial pose
        self.robot_device_name = robot_device_name
        self.robot_device_class_id = robot_device_class_id
        self.robot_state_shm = None

        # Absolute pose state: [left_x, left_y, left_z, left_qx, left_qy, left_qz, left_qw,
        #                       left_gripper, right_x, right_y, right_z, right_qx, right_qy, right_qz, right_qw,
        #                       right_gripper]
        # Initialize with identity quaternions - positions will be set from robot state
        self.current_pose = np.zeros(16, dtype=get_dtype(self.data_dtype))
        self.current_pose[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat for left
        self.current_pose[10:14] = [0.0, 0.0, 0.0, 1.0]  # identity quat for right
        self.command_pose = np.zeros(16, dtype=get_dtype(self.data_dtype))
        self.command_pose[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat for left
        self.command_pose[10:14] = [0.0, 0.0, 0.0, 1.0]  # identity quat for right
        self._pose_initialized = False  # Flag to track if pose has been initialized from robot

        # Gripper states (0.0 = closed, max_gripper_width = open)
        self.left_gripper = max_gripper_width
        self.right_gripper = max_gripper_width

        # Keyboard listener
        self.keyboard_listener = None
        self.keyboard_module = None  # Store reference to keyboard module for use in callbacks
        self.running = False

        logger.info(f"Keyboard Teleoperator initialized (absolute EEF mode)")
        logger.info(f"Device ID: {device_class_id}")
        logger.info(f"Delta step: {delta_step} m/frame")
        logger.info(f"Output shape: (16,) - [left_pos(3), left_quat(4), left_gripper, right_pos(3), right_quat(4), right_gripper]")
        logger.info(f"Max gripper width: {max_gripper_width} m")
        logger.info(f"Control flag SHM: {self._control_flag_shm_name} (shape: [1])")
        logger.info("")
        logger.info("Key Mapping:")
        logger.info("  Left Arm:  W/S=+/-Y, A/D=-/+X, Q/E=-/+Z, F=gripper")
        logger.info("  Right Arm: I/K=+/-Y, J/L=-/+X, U/O=-/+Z, H=gripper")
        logger.info("  SPACE: Toggle between policy mode and human control mode")

    def _on_key_press(self, key) -> None:
        """Handle key press events - update absolute pose state."""
        try:
            # Check for SPACE key (shared control mode toggle)
            if self.keyboard_module is not None and key == self.keyboard_module.Key.space:
                self._human_in_control = not self._human_in_control
                self._create_reset_signal_file()
                logger.info(f"Control mode switched: {'HUMAN' if self._human_in_control else 'POLICY'}")
                return

            # Get character for regular keys
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
            if k is None:
                return

            # Update left arm position based on key
            if k == 'd': self.command_pose[0] = self.current_pose[0] + self.delta_step   # +X
            if k == 'a': self.command_pose[0] = self.current_pose[0] - self.delta_step   # -X
            if k == 'w': self.command_pose[1] = self.current_pose[1] + self.delta_step   # +Y
            if k == 's': self.command_pose[1] = self.current_pose[1] - self.delta_step   # -Y
            if k == 'e': self.command_pose[2] = self.current_pose[2] + self.delta_step   # +Z
            if k == 'q': self.command_pose[2] = self.current_pose[2] - self.delta_step   # -Z

            # Update right arm position based on key
            if k == 'l': self.command_pose[8] = self.current_pose[8] + self.delta_step   # +X
            if k == 'j': self.command_pose[8] = self.current_pose[8] - self.delta_step   # -X
            if k == 'i': self.command_pose[9] = self.current_pose[9] + self.delta_step   # +Y
            if k == 'k': self.command_pose[9] = self.current_pose[9] - self.delta_step   # -Y
            if k == 'o': self.command_pose[10] = self.current_pose[10] + self.delta_step   # +Z
            if k == 'u': self.command_pose[10] = self.current_pose[10] - self.delta_step   # -Z

            # Handle gripper toggle with debouncing
            if k == 'f':
                self.left_gripper = self.max_gripper_width - self.left_gripper
                self.command_pose[7] = self.left_gripper
                logger.info(f"Left gripper: {'OPEN' if self.left_gripper > self.max_gripper_width / 2 else 'CLOSED'}")
            elif k == 'h':
                self.right_gripper = self.max_gripper_width - self.right_gripper
                self.command_pose[15] = self.right_gripper
                logger.info(f"Right gripper: {'OPEN' if self.right_gripper > self.max_gripper_width / 2 else 'CLOSED'}")

        except AttributeError:
            # Special keys (shift, ctrl, etc.) - ignore
            pass

    def _on_key_release(self, key) -> None:
        """Handle key release events."""
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
            if k is None:
                return

        except AttributeError:
            pass

    def _reorder_quat_wxyz_to_xyzw(self, quat_wxyz: np.ndarray) -> np.ndarray:
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    def _create_control_flag_shm(self) -> None:
        """Create this producer's control flag SHM (0.0=policy, 1.0=human)."""

        try:
            from utils.shm_utils import (
                pack_device_header,
                pack_buffer_header,
                DEVICE_HEADER_SIZE,
                BUFFER_HEADER_SIZE,
                FRAME_HEADER_SIZE,
            )
            import struct

            # Calculate buffer layout for control flag SHM
            control_flag_shape = (1,)  # Single float64 value
            numpy_dtype = get_dtype(self.data_dtype)
            bytes_per_element = np.dtype(numpy_dtype).itemsize
            data_size = np.prod(control_flag_shape) * bytes_per_element
            frame_size = FRAME_HEADER_SIZE + data_size
            total_shm_size = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + self._control_flag_buffer_size * frame_size

            # Create SHM using multiprocessing
            import multiprocessing.shared_memory as shm

            # Check if SHM already exists, if so, clean it up first
            try:
                existing_shm = shm.SharedMemory(name=self._control_flag_shm_name, create=False)
                existing_shm.close()
                existing_shm.unlink()
                logger.info(f"Removed existing control flag SHM: {self._control_flag_shm_name}")
            except FileNotFoundError:
                # SHM doesn't exist, which is expected
                pass
            except Exception as e:
                # Ignore other errors when checking for existing SHM
                logger.debug(f"Could not check existing SHM: {e}")

            # Create new SHM
            self._control_flag_shm = shm.SharedMemory(
                name=self._control_flag_shm_name,
                create=True,
                size=total_shm_size
            )
            logger.info(f"Control flag SHM created: {self._control_flag_shm_name}")
            # Initialize device header
            device_header = pack_device_header(
                self.device_name,
                self.device_class_id,
                self.fps,
                self.data_dtype,
                control_flag_shape,
                frame_size,
                self.hardware_latency_ms
            )
            self._control_flag_shm.buf[:DEVICE_HEADER_SIZE] = device_header

            # Initialize buffer header
            buffer_header = pack_buffer_header(self._control_flag_buffer_size, 0, 0)
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

            # Initialize write index
            self._control_flag_write_index = 0

            logger.info(f"Created control flag SHM: {self._control_flag_shm_name} ({total_shm_size} bytes)")

        except Exception as e:
            logger.error(f"Failed to create control flag SHM: {e}")
            self._control_flag_shm = None

    def _write_control_flag(self, flag_value: float, timestamp_ns: int) -> None:
        """
        Write control flag value to shared memory.

        Args:
            flag_value: 0.0 for policy mode, 1.0 for human mode
            timestamp_ns: timestamp in nanoseconds (should match the teleoperator array timestamp)
        """
        if self._control_flag_shm is None:
            logger.info(f"Control flag SHM not connected")
            return

        try:
            import struct
            from utils.shm_utils import pack_frame_header, FRAME_HEADER_SIZE

            # Get frame size from device header
            device_header_data = bytes(self._control_flag_shm.buf[:DEVICE_HEADER_SIZE])
            from utils.shm_utils import unpack_device_header
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info['frame_size']

            # Calculate frame offset
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + self._control_flag_write_index * frame_size

            # Pack frame header with provided timestamp
            frame_header = pack_frame_header(timestamp_ns)
            self._control_flag_shm.buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = frame_header

            # Write flag value
            data_start = frame_offset + FRAME_HEADER_SIZE
            flag_bytes = struct.pack('<d', flag_value)  # float64, little endian
            self._control_flag_shm.buf[data_start:data_start + 8] = flag_bytes

            # Update buffer header
            from utils.shm_utils import pack_buffer_header
            current_frames_count = min(self._control_flag_write_index + 1, self._control_flag_buffer_size)
            buffer_header = pack_buffer_header(self._control_flag_buffer_size, current_frames_count, self._control_flag_write_index)
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

            # Update write index (circular buffer)
            self._control_flag_write_index = (self._control_flag_write_index + 1) % self._control_flag_buffer_size

        except Exception as e:
            logger.warning(f"Failed to write control flag: {e}")

    def _clear_control_flag_shm(self) -> None:
        """Clear all frames in the control flag SHM buffer to initial state."""
        if self._control_flag_shm is None:
            return

        try:
            from utils.shm_utils import pack_frame_header, pack_buffer_header, FRAME_HEADER_SIZE

            # Get frame size from device header
            device_header_data = bytes(self._control_flag_shm.buf[:DEVICE_HEADER_SIZE])
            from utils.shm_utils import unpack_device_header
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info['frame_size']

            # Clear all frames in the buffer
            for i in range(self._control_flag_buffer_size):
                frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * frame_size

                # Set frame timestamp to 0
                zero_timestamp_header = pack_frame_header(0)
                self._control_flag_shm.buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = zero_timestamp_header

                # Clear frame data
                data_start = frame_offset + FRAME_HEADER_SIZE
                data_end = frame_offset + frame_size
                self._control_flag_shm.buf[data_start:data_end] = b'\x00' * (frame_size - FRAME_HEADER_SIZE)

            # Reset buffer header
            buffer_header = pack_buffer_header(self._control_flag_buffer_size, 0, 0)
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

            # Reset write index
            self._control_flag_write_index = 0

            logger.debug("Cleared control flag SHM buffer")

        except Exception as e:
            logger.warning(f"Failed to clear control flag SHM: {e}")

    def _cleanup_control_flag_shm(self) -> None:
        """Clean up control flag shared memory."""
        if self._control_flag_shm:
            try:
                self._control_flag_shm.close()
                if hasattr(self._control_flag_shm, 'unlink'):
                    self._control_flag_shm.unlink()
                logger.info(f"Cleaned up control flag SHM: {self._control_flag_shm_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup control flag SHM: {e}")
            finally:
                self._control_flag_shm = None

    def _create_reset_signal_file(self) -> None:
        """Create reset signal file to notify policy runner and policy connector of control mode switch."""
        try:
            from pathlib import Path
            # Use configured reset signal path if available
            reset_signal_path = self._reset_signal_path
            assert reset_signal_path is not None, "reset_signal_path is required in signal_files"
            reset_signal_file = Path(reset_signal_path)
            reset_signal_file.touch()
            logger.debug(f"Created reset signal file: {reset_signal_file}")
        except Exception as e:
            logger.warning(f"Failed to create reset signal file: {e}")

    def _check_policy_runner_reset_signal(self) -> bool:
        """
        Check if policy runner reset signal file has been updated.

        Returns:
            True if reset signal detected, False otherwise
        """
        try:
            if self._policy_runner_reset_signal.exists():
                current_mtime = self._policy_runner_reset_signal.stat().st_mtime
                if current_mtime > self._last_policy_runner_reset_mtime:
                    self._last_policy_runner_reset_mtime = current_mtime
                    return True
            return False
        except Exception as e:
            logger.debug(f"Failed to check policy runner reset signal: {e}")
            return False

    def _check_teleoperator_reset_signal(self) -> bool:
        """
        Check if teleoperator reset signal file has been updated.
        This signal is triggered by PolicyConnector when a new episode starts,
        forcing the teleoperator to switch to policy control mode.

        Returns:
            True if reset signal detected, False otherwise
        """
        try:
            if self._teleoperator_reset_signal.exists():
                current_mtime = self._teleoperator_reset_signal.stat().st_mtime
                if current_mtime > self._last_teleoperator_reset_mtime:
                    self._last_teleoperator_reset_mtime = current_mtime
                    return True
            return False
        except Exception as e:
            logger.debug(f"Failed to check teleoperator reset signal: {e}")
            return False

    def _generate_absolute_array(self) -> np.ndarray:
        """
        Generate current absolute pose array for both arms.
        Converts quaternions from wxyz (Flexiv robot format) to xyzw (expected format).

        Returns:
            numpy.ndarray: 16-element array [left_pos(3), left_quat(4), left_gripper,
                                             right_pos(3), right_quat(4), right_gripper]
        """
        # Return copy with quaternion format conversion (wxyz -> xyzw)
        output = self.command_pose.copy()
        # Convert left quat from wxyz to xyzw
        output[3:7] = self._reorder_quat_wxyz_to_xyzw(output[3:7])
        # Convert right quat from wxyz to xyzw
        output[11:15] = self._reorder_quat_wxyz_to_xyzw(output[11:15])
        return output

    def _start_keyboard_listener(self) -> None:
        """Start the keyboard listener using pynput."""
        try:
            from pynput import keyboard as pynput_keyboard

            # Store reference to keyboard module for use in callbacks
            self.keyboard_module = pynput_keyboard

            self.keyboard_listener = pynput_keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.keyboard_listener.start()
            logger.info("Keyboard listener started (using pynput)")

        except ImportError:
            logger.error("pynput module not installed. Please install: pip install pynput")
            raise RuntimeError("pynput module required for KeyboardTeleoperator")
        except Exception as e:
            logger.error(f"Error starting keyboard listener: {e}")
            raise

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
                logger.info("Keyboard listener stopped")
            except Exception as e:
                logger.error(f"Error stopping keyboard listener: {e}")

    def _connect_robot_state_shm(self) -> None:
        """Connect to robot state shared memory to read current TCP poses for initialization."""
        if self.robot_state_shm is not None:
            return
        try:
            self.robot_state_shm = connect_to_device_shm(
                device_name=self.robot_device_name,
                device_class_id=self.robot_device_class_id,
                read_only=True,
            )
            logger.info(f"Connected to robot state SHM: {self.robot_device_name}_{self.robot_device_class_id}")
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning(f"Failed to connect robot state SHM: {exc}, using default initial poses")

    def _read_robot_poses(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read current TCP poses from robot state SHM.

        Returns:
            Tuple of (left_tcp, right_tcp) arrays, or (None, None) if unavailable
        """
        if self.robot_state_shm is None:
            return None, None

        buf = self.robot_state_shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )

        if buffer_header["current_frames_count"] == 0:
            return None, None

        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size

        frame_header = unpack_frame_header(
            buf[frame_offset : frame_offset + FRAME_HEADER_SIZE]
        )

        if frame_header["timestamp_ns"] == 0:
            return None, None

        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        data_end = data_start + data_size
        dtype = get_dtype(device_header["data_dtype"], np.uint8)
        data_array = np.frombuffer(buf[data_start:data_end], dtype=dtype)
        shape = device_header["shape"]
        if len(shape) > 0:
            data_array = data_array.reshape(shape)
        flat_data = data_array.reshape(-1)

        if flat_data.shape[0] >= RizonDualArmStateLayout.BASE_SIZE:
            left_tcp = flat_data[RizonDualArmStateLayout.LEFT_MEASURED_TCP].copy()
            right_tcp = flat_data[RizonDualArmStateLayout.RIGHT_MEASURED_TCP].copy()
            return left_tcp, right_tcp
        if flat_data.shape[0] >= RizonDualArmStateLayout.LEFT_MEASURED_TCP.stop:
            left_tcp = flat_data[RizonDualArmStateLayout.LEFT_MEASURED_TCP].copy()
            return left_tcp, None

        logger.warning(
            f"Robot state shape too small for TCP pose: {flat_data.shape[0]}"
        )
        return None, None

    def start_server(self) -> None:
        """Start the keyboard teleoperator server with absolute EEF control."""
        if self.running:
            logger.info(f"Keyboard Teleoperator {self.device_class_id} is already running")
            return

        logger.info(f"Starting Keyboard Teleoperator {self.device_class_id} server...")
        self.running = True

        # Start keyboard listener
        self._start_keyboard_listener()

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            self._stop_keyboard_listener()
            raise RuntimeError("Failed to create shared memory")

        self._create_control_flag_shm()

        # Wait for robot state SHM to be connected
        logger.disable(__name__)
        logger.disable("utils.shm_utils")
        while self.robot_state_shm is None:
            try:
                self._connect_robot_state_shm()
            except:
                time.sleep(0.1)
        logger.enable(__name__)
        logger.enable("utils.shm_utils")
        logger.info(f"Robot state SHM connected: {self.robot_state_shm}")

        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")
        logger.info("")
        logger.info("Use keyboard to control robot (absolute EEF mode):")
        logger.info("  Left arm: WASD (XY), QE (Z), F (gripper)")
        logger.info("  Right arm: IJKL (XY), UO (Z), H (gripper)")
        logger.info("Press Ctrl+C to stop...")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main data generation loop
        while self.running:
            try:
                # Check for policy runner reset signal (triggered when episode ends)
                if self._check_policy_runner_reset_signal():
                    self._human_in_control = False
                    # Clear control flag SHM completely and reset to initial state (0.0 = policy mode)
                    self._clear_control_flag_shm()
                    self._write_control_flag(0.0, time.time_ns())
                    logger.info("Policy runner reset: cleared control flag SHM and switched to policy control mode")

                # Check for teleoperator reset signal (triggered by PolicyConnector when new episode starts)
                # This ensures new episode always starts with policy control
                if self._check_teleoperator_reset_signal():
                    self._human_in_control = False
                    # Clear control flag SHM completely and reset to policy mode
                    self._clear_control_flag_shm()
                    self._write_control_flag(0.0, time.time_ns())
                    logger.info("Teleoperator reset signal: switched to policy control mode for new episode")
                    # Delete the signal file after processing to prevent continuous reset
                    try:
                        if self._teleoperator_reset_signal.exists():
                            self._teleoperator_reset_signal.unlink()
                            logger.debug("Deleted teleoperator reset signal file after processing")
                    except Exception as e:
                        logger.warning(f"Failed to delete teleoperator reset signal file: {e}")

                # Try to update poses from robot state (non-blocking)
                left_tcp, right_tcp = self._read_robot_poses()
                if left_tcp is not None and len(left_tcp) >= 7:
                    if not self._pose_initialized or not self._human_in_control:
                        # logger.info(f"Initialized left arm pose from robot: [{left_tcp[0]:.3f}, {left_tcp[1]:.3f}, {left_tcp[2]:.3f}]")
                        self.command_pose[0:7] = left_tcp[:7]
                    self.current_pose[0:7] = left_tcp[:7]
                if right_tcp is not None and len(right_tcp) >= 7:
                    if not self._pose_initialized or not self._human_in_control:
                        # logger.info(f"Initialized right arm pose from robot: [{right_tcp[0]:.3f}, {right_tcp[1]:.3f}, {right_tcp[2]:.3f}]")
                        self.command_pose[8:15] = right_tcp[:7]
                    self.current_pose[8:15] = right_tcp[:7]
                    self._pose_initialized = True
                # Only write to SHM after poses are initialized
                if self._pose_initialized:
                    # Generate absolute pose array and write to SHM
                    absolute_pose = self._generate_absolute_array()
                    # logger.info(f"Writing absolute pose to SHM: {absolute_pose}, length: {len(absolute_pose)}")
                    timestamp_ns = time.time_ns()
                    self._write_array_to_shm_with_timestamp(absolute_pose, timestamp_ns)
                    # Use the same timestamp for control flag to ensure synchronization.
                    self._write_control_flag(1.0 if self._human_in_control else 0.0, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()

            except Exception as e:
                logger.error(f"Error in keyboard teleoperator: {e}")
                break

    def stop_server(self) -> None:
        """Stop the keyboard teleoperator server."""
        if not self.running:
            return

        logger.info(f"Stopping Keyboard Teleoperator {self.device_class_id} server...")
        self.running = False

        # Stop keyboard listener
        self._stop_keyboard_listener()

        # Close robot state SHM connection
        if self.robot_state_shm is not None:
            try:
                self.robot_state_shm.close()
            except Exception as exc:
                logger.warning(f"Failed to close robot state SHM: {exc}")
            self.robot_state_shm = None

        self._cleanup_control_flag_shm()

        # Cleanup shared memory
        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update({
            "control_mode": "eef_abs",
            "human_in_control": self._human_in_control,
            "control_flag_shm": self._control_flag_shm_name,
            "delta_step": self.delta_step,
            "max_gripper_width": self.max_gripper_width,
            "left_gripper": self.left_gripper,
            "right_gripper": self.right_gripper,
            "current_left_pose": self.current_pose[0:7].tolist(),
            "current_right_pose": self.current_pose[8:15].tolist(),
        })
        return info


class Quest3Teleoperator(BaseDevice):
    """
    Quest3 Teleoperator for delta EEF control.
    """

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        host_ip: str = "192.168.2.187",
        port: int = 8082,
        teleop_source: str = "vr",
        teleop_mode: str = "left_arm_6DOF",
        relative_translation_scale: float = 1.0,
        max_gripper_width: float = 0.1,
        min_gripper_width: float = 0.01,
        grasp_force: float = 100.0,
        calibration_path: str = "./data/calibration/ywy",
        robot_device_name: str = "RizonRobot",
        robot_device_class_id: int = 1,
    ) -> None:
        """Initialize the Quest3 teleoperator."""

        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        if self.data_shape != (16,):
            raise ValueError(
                "Quest3Teleoperator state_feedback format must have shape [16]"
            )

        # Override device name
        self.device_name = "Quest3Teleoperator"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        self.debug = False
        self.host_ip = host_ip
        self.port = port

        assert calibration_path is not None, "calibration_path is required for Quest3Teleoperator"
        self.transforms = RealWorldTransforms(calibration_path=calibration_path)
        self.teleop_source = teleop_source
        self.teleop_mode = TeleopMode[teleop_mode]
        self.relative_translation_scale = relative_translation_scale
        self.max_gripper_width = max_gripper_width
        self.min_gripper_width = min_gripper_width
        self.grasp_force = grasp_force
        self.robot_device_name = robot_device_name
        self.robot_device_class_id = robot_device_class_id

        self.left_tracking_state = False
        self.right_tracking_state = False
        self.left_start_real_tcp: Optional[np.ndarray] = None
        self.left_start_unity_tcp: Optional[np.ndarray] = None
        self.right_start_real_tcp: Optional[np.ndarray] = None
        self.right_start_unity_tcp: Optional[np.ndarray] = None

        # self._latest_command: Optional[Dict[str, object]] = None
        # self._command_lock = threading.Lock()
        self._latest_message: Optional[UnityMes] = None
        self._message_lock = threading.Lock()
        self.robot_state_shm = None

        self.output_array = None
        self.app = FastAPI()
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._uvicorn_thread: Optional[threading.Thread] = None
        self._setup_routes()

        logger.info("Quest3 Teleoperator initialized")
        logger.info(f"Device ID: {self.device_class_id}")
        logger.info(f"FPS: {self.fps}")
        logger.info(f"Teleop source: {self.teleop_source}")
        logger.info(f"Teleop mode: {self.teleop_mode.name}")
        logger.info(f"Host: {self.host_ip}:{self.port}")

    def _setup_routes(self) -> None:
        @self.app.post("/unity")
        async def unity(mes: UnityMes) -> Dict[str, str]:
            self._set_latest_message(mes)
            return {"status": "ok"}

        @self.app.get("/")
        async def root() -> Dict[str, str]:
            return {"status": "ok"}

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
            logger.error("Validation Error:")
            logger.error(f"Errors: {exc.errors()}")
            logger.error(f"Request Body: {await request.body()}")

            return JSONResponse(
                status_code=422,
                content={
                    "detail": exc.errors(),
                    "body": await request.json(),
                },
            )

    def _set_latest_message(self, mes: UnityMes) -> None:
        with self._message_lock:
            self._latest_message = mes

    def _get_latest_message(self) -> Optional[UnityMes]:
        with self._message_lock:
            if self._latest_message is None:
                return None
            return copy.deepcopy(self._latest_message)

    def _connect_robot_state_shm(self) -> None:
        if self.robot_state_shm is not None:
            return
        try:
            self.robot_state_shm = connect_to_device_shm(
                device_name=self.robot_device_name,
                device_class_id=self.robot_device_class_id,
                read_only=True,
            )
        except FileNotFoundError:
            logger.warning(
                f"Robot state SHM not found for {self.robot_device_name}_{self.robot_device_class_id}"
            )
        except Exception as exc:
            logger.warning(f"Failed to connect robot state SHM: {exc}")

    def _read_latest_robot_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self._connect_robot_state_shm()
        if self.robot_state_shm is None:
            return None, None

        buf = self.robot_state_shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )

        if buffer_header["current_frames_count"] == 0:
            return None, None

        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size

        frame_header = unpack_frame_header(
            buf[frame_offset : frame_offset + FRAME_HEADER_SIZE]
        )

        if frame_header["timestamp_ns"] == 0:
            return None, None

        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        data_end = data_start + data_size
        dtype = get_dtype(device_header["data_dtype"], np.uint8)
        data_array = np.frombuffer(buf[data_start:data_end], dtype=dtype)
        shape = device_header["shape"]
        if len(shape) > 0:
            data_array = data_array.reshape(shape)
        flat_data = data_array.reshape(-1)

        if flat_data.shape[0] >= RizonDualArmStateLayout.BASE_SIZE:
            left_tcp = flat_data[RizonDualArmStateLayout.LEFT_MEASURED_TCP].copy()
            right_tcp = flat_data[RizonDualArmStateLayout.RIGHT_MEASURED_TCP].copy()
            return left_tcp, right_tcp
        if flat_data.shape[0] >= RizonDualArmStateLayout.LEFT_MEASURED_TCP.stop:
            left_tcp = flat_data[RizonDualArmStateLayout.LEFT_MEASURED_TCP].copy()
            return left_tcp, None

        logger.warning(
            f"Robot state shape too small for TCP pose: {flat_data.shape[0]}"
        )
        return None, None

    def _set_start_tcp(
        self, robot_tcp: np.ndarray, unity_tcp: np.ndarray, is_left: bool
    ) -> None:
        if is_left:
            self.left_start_real_tcp = robot_tcp
            self.left_start_unity_tcp = unity_tcp
            self.left_tracking_state = True
            logger.info(
                f"left start real tcp set to {robot_tcp}, left start unity tcp set to {unity_tcp}"
            )
        else:
            self.right_start_real_tcp = robot_tcp
            self.right_start_unity_tcp = unity_tcp
            self.right_tracking_state = True
            logger.info(
                f"right start real tcp set to {robot_tcp}, right start unity tcp set to {unity_tcp}"
            )

    def _clear_start_tcp(self, is_left: bool) -> None:
        if is_left:
            self.left_start_real_tcp = None
            self.left_start_unity_tcp = None
        else:
            self.right_start_real_tcp = None
            self.right_start_unity_tcp = None

    def _calc_relative_target(self, pos_from_unity: np.ndarray, is_left: bool) -> np.ndarray:
        if is_left:
            start_real_tcp = self.left_start_real_tcp
            start_unity_tcp = self.left_start_unity_tcp
        else:
            start_real_tcp = self.right_start_real_tcp
            start_unity_tcp = self.right_start_unity_tcp

        if start_real_tcp is None or start_unity_tcp is None:
            raise RuntimeError("Start TCP is not set before tracking.")

        target = np.zeros(7)
        target[:3] = (
            self.relative_translation_scale * (pos_from_unity[:3] - start_unity_tcp[:3])
            + start_real_tcp[:3]
        )
        target_rot_mat = (
            t3d.quaternions.quat2mat(pos_from_unity[3:])
            @ np.linalg.inv(t3d.quaternions.quat2mat(start_unity_tcp[3:]))
            @ t3d.quaternions.quat2mat(start_real_tcp[3:])
        )
        target[3:] = t3d.quaternions.mat2quat(target_rot_mat).tolist()
        return target

    def _reorder_quat_wxyz_to_xyzw(self, quat_wxyz: np.ndarray) -> np.ndarray:
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    def _build_output_array(
        self,
        left_target: Optional[np.ndarray],
        right_target: Optional[np.ndarray],
        left_gripper_width_target: float,
        right_gripper_width_target: float,
    ) -> np.ndarray:
        output = np.zeros(self.data_shape[0], dtype=get_dtype(self.data_dtype))

        if left_target is None:
            while left_target is None:
                try:
                    left_target = self._read_latest_robot_state()[0]
                except:
                    time.sleep(0.01)
        if right_target is None:
            while right_target is None:
                try:
                    right_target = self._read_latest_robot_state()[1]
                except:
                    time.sleep(0.01)

        output[0:3] = left_target[:3]
        output[3:7] = self._reorder_quat_wxyz_to_xyzw(left_target[3:])
        output[7] = left_gripper_width_target
        output[8:11] = right_target[:3]
        output[11:15] = self._reorder_quat_wxyz_to_xyzw(right_target[3:])
        output[15] = right_gripper_width_target
        return output

    def _process_message(self, mes: UnityMes) -> None:
        mes.leftHand.cmd = 12 if mes.leftHand.buttonState[4] else 10
        mes.rightHand.cmd = 12 if mes.rightHand.buttonState[4] else 10

        if str(mes.leftHand.cmd) not in composed_cmd_interpreter[self.teleop_source]:
            raise ValueError(f"Unsupported command {mes.leftHand.cmd} for {self.teleop_source}")
        if str(mes.rightHand.cmd) not in composed_cmd_interpreter[self.teleop_source]:
            raise ValueError(f"Unsupported command {mes.rightHand.cmd} for {self.teleop_source}")

        interpreted_left_cmd = composed_cmd_interpreter[self.teleop_source][str(mes.leftHand.cmd)]
        interpreted_right_cmd = composed_cmd_interpreter[self.teleop_source][str(mes.rightHand.cmd)]

        valid_gripper_interval = self.max_gripper_width - self.min_gripper_width
        left_gripper_width_target = (
            self.max_gripper_width - mes.leftHand.triggerState * valid_gripper_interval
            if self.left_tracking_state
            else self.max_gripper_width
        )
        right_gripper_width_target = (
            self.max_gripper_width - mes.rightHand.triggerState * valid_gripper_interval
            if self.right_tracking_state
            else self.max_gripper_width
        )

        r_pos_from_unity = self.transforms.unity2robot_frame(
            np.array(mes.rightHand.wristPos + mes.rightHand.wristQuat), False
        )
        l_pos_from_unity = self.transforms.unity2robot_frame(
            np.array(mes.leftHand.wristPos + mes.leftHand.wristQuat), True
        )

        if interpreted_left_cmd.tracking and not self.left_tracking_state:
            logger.info("left start tracking")
            left_tcp, _ = self._read_latest_robot_state()
            if left_tcp is None:
                logger.warning("Left robot TCP unavailable; skipping tracking start")
            else:
                self._set_start_tcp(left_tcp, l_pos_from_unity, is_left=True)
        elif not interpreted_left_cmd.tracking and self.left_tracking_state:
            logger.info("left stop tracking")
            self.left_tracking_state = False
            self._clear_start_tcp(is_left=True)

        if interpreted_right_cmd.tracking and not self.right_tracking_state:
            logger.info("right start tracking")
            _, right_tcp = self._read_latest_robot_state()
            if right_tcp is None:
                logger.warning("Right robot TCP unavailable; skipping tracking start")
            else:
                self._set_start_tcp(right_tcp, r_pos_from_unity, is_left=False)
        elif not interpreted_right_cmd.tracking and self.right_tracking_state:
            logger.info("right stop tracking")
            self.right_tracking_state = False
            self._clear_start_tcp(is_left=False)

        if self.robot_state_shm is None:
            return

        left_target = None
        if self.left_tracking_state:
            left_target_7d_in_robot = self._calc_relative_target(l_pos_from_unity, is_left=True)
            left_target_6d_in_world = matrix4x4_to_pose_6d(
                self.transforms.left_robot_base_to_world_transform
                @ pose_7d_to_4x4matrix(left_target_7d_in_robot)
            )
            if self.teleop_mode == TeleopMode.left_arm_3D_translation:
                left_target_6d_in_world[3] = np.clip(left_target_6d_in_world[3], -np.pi, -np.pi)
                left_target_6d_in_world[4] = np.clip(left_target_6d_in_world[4], 0, 0)
                left_target_6d_in_world[5] = np.clip(left_target_6d_in_world[5], np.pi, np.pi)

                left_target = pose_6d_to_pose_7d(
                    matrix4x4_to_pose_6d(
                        self.transforms.world_to_left_robot_base_transform
                        @ pose_6d_to_4x4matrix(left_target_6d_in_world)
                    )
                )
            elif self.teleop_mode == TeleopMode.left_arm_3D_translation_Y_rotation:
                left_target_6d_in_world[3] = np.clip(left_target_6d_in_world[3], -np.pi, -np.pi)
                left_target_6d_in_world[5] = np.clip(left_target_6d_in_world[5], np.pi, np.pi)

                left_target = pose_6d_to_pose_7d(
                    matrix4x4_to_pose_6d(
                        self.transforms.world_to_left_robot_base_transform
                        @ pose_6d_to_4x4matrix(left_target_6d_in_world)
                    )
                )
            elif self.teleop_mode in (TeleopMode.left_arm_6DOF, TeleopMode.dual_arm_6DOF):
                left_target = left_target_7d_in_robot
            elif self.teleop_mode == TeleopMode.dual_arm_3D_translation:
                left_target_6d_in_world[3] = np.clip(left_target_6d_in_world[3], -np.pi / 2, -np.pi / 2)
                left_target_6d_in_world[4] = np.clip(left_target_6d_in_world[4], 0, 0)
                left_target_6d_in_world[5] = np.clip(left_target_6d_in_world[5], np.pi, np.pi)

                left_target = pose_6d_to_pose_7d(
                    matrix4x4_to_pose_6d(
                        self.transforms.world_to_left_robot_base_transform
                        @ pose_6d_to_4x4matrix(left_target_6d_in_world)
                    )
                )
            else:
                raise ValueError(f"Unsupported teleoperation mode: {self.teleop_mode}")

        right_target = None
        if self.right_tracking_state:
            right_target_7d_in_robot = self._calc_relative_target(r_pos_from_unity, is_left=False)
            right_target_6d_in_world = matrix4x4_to_pose_6d(
                self.transforms.right_robot_base_to_world_transform
                @ pose_7d_to_4x4matrix(right_target_7d_in_robot)
            )
            if self.teleop_mode in (
                TeleopMode.left_arm_3D_translation,
                TeleopMode.left_arm_3D_translation_Y_rotation,
                TeleopMode.left_arm_6DOF,
            ):
                right_target = right_target_7d_in_robot
            elif self.teleop_mode in (TeleopMode.dual_arm_3D_translation, TeleopMode.dual_arm_6DOF):
                right_target_6d_in_world[3] = np.clip(right_target_6d_in_world[3], np.pi / 2, np.pi / 2)
                right_target_6d_in_world[4] = np.clip(right_target_6d_in_world[4], 0, 0)
                right_target_6d_in_world[5] = np.clip(right_target_6d_in_world[5], np.pi, np.pi)

                right_target = pose_6d_to_pose_7d(
                    matrix4x4_to_pose_6d(
                        self.transforms.world_to_right_robot_base_transform
                        @ pose_6d_to_4x4matrix(right_target_6d_in_world)
                    )
                )
            else:
                raise ValueError(f"Unsupported teleoperation mode: {self.teleop_mode}")

        self.output_array = self._build_output_array(
            left_target=left_target,
            right_target=right_target,
            left_gripper_width_target=left_gripper_width_target,
            right_gripper_width_target=right_gripper_width_target,
        )

    def start_server(self) -> None:
        """Start the Quest3 teleoperator server."""
        if self.running:
            logger.info(f"Quest3 Teleoperator {self.device_class_id} is already running")
            return

        logger.info(f"Starting Quest3 Teleoperator {self.device_class_id} server...")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")
        logger.info("Waiting for Quest3 messages...")

        # Start FastAPI server in background thread
        config = uvicorn.Config(self.app, host=self.host_ip, port=self.port, log_level="error")
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
        self._uvicorn_thread.start()

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Wait for robot state SHM to be connected
        logger.disable(__name__)
        logger.disable("utils.shm_utils")
        while self.robot_state_shm is None:
            try:
                self._connect_robot_state_shm()
            except:
                time.sleep(0.1)
        logger.enable(__name__)
        logger.enable("utils.shm_utils")
        logger.info(f"Robot state SHM connected: {self.robot_state_shm}")

        while self.running:
            try:
                mes = self._get_latest_message()
                if mes is not None:
                    self._process_message(mes)
                timestamp_ns = time.time_ns()
                if self.output_array is not None:
                    self._write_array_to_shm_with_timestamp(self.output_array, timestamp_ns)
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in Quest3 teleoperator: {e}")
                break

    def stop_server(self) -> None:
        """Stop the Quest3 teleoperator server."""
        if not self.running:
            return

        logger.info(f"Stopping Quest3 Teleoperator {self.device_class_id} server...")
        self.running = False

        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._uvicorn_thread is not None:
            self._uvicorn_thread.join(timeout=2.0)
            self._uvicorn_thread = None
        if self.robot_state_shm is not None:
            try:
                self.robot_state_shm.close()
            except Exception as exc:
                logger.warning(f"Failed to close robot state SHM: {exc}")
            self.robot_state_shm = None

        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = super().get_device_info()
        info.update(
            {
                "host": self.host_ip,
                "port": self.port,
                "teleop_source": self.teleop_source,
                "teleop_mode": self.teleop_mode.name,
                "relative_translation_scale": self.relative_translation_scale,
                "max_gripper_width": self.max_gripper_width,
                "min_gripper_width": self.min_gripper_width,
                "grasp_force": self.grasp_force,
                "robot_device_name": self.robot_device_name,
                "robot_device_class_id": self.robot_device_class_id,
                "left_tracking_state": self.left_tracking_state,
                "right_tracking_state": self.right_tracking_state,
                # "latest_command": self.get_latest_command(),
            }
        )
        return info

class _VuerFlexivSide:
    """Per-arm clutch / tracking bookkeeping for Quest3VuerTeleoperator.

    Mirrors R1's ``FlexivSide`` but does NOT hold a robot handle: the TCP
    anchor is read from the robot state SHM and execution happens inside
    ``RizonRobot`` via the 16D eef_abs control SHM.
    """

    def __init__(self, label: str, gripper_open_value: float):
        self.label = label
        self.T_ctrl_ref: Optional[np.ndarray] = None   # controller pose at clutch
        self.T_tcp_ref: Optional[np.ndarray] = None     # robot TCP at clutch
        self.T_tcp_cmd: Optional[np.ndarray] = None      # last commanded TCP (4x4)
        self.last_gripper: float = gripper_open_value


class Quest3VuerTeleoperator(BaseDevice):
    """Quest 3 (Vuer / WebXR) dual-arm teleoperator for Flexiv Rizon arms.

    This conforms R1_teleoperation's monolithic Flexiv following loop to
    SuperInference's SHM architecture:

    * Captures Quest controller / head poses + buttons through the vendored
      ``QuestVrWebRTCViewer`` (Vuer browser stack, WebRTC video).
    * Ports R1's clutch / absolute-or-relative orientation following /
      workspace clamp / rate-limit logic to produce a 16D absolute
      end-effector target ``[L_xyz, L_quat_xyzw, L_grip, R_xyz, R_quat_xyzw,
      R_grip]`` written to ``<name>_<id>_data`` (consumed by ``RizonRobot``
      in ``eef_abs`` mode).
    * Publishes a secondary head SHM ``<name>_<id>_head`` carrying the head
      4x4 pose + left controller buttons for ``YamHeadRobot``.

    The TCP anchor used by the clutch is read from the robot state SHM
    (``robot_device_name``/``robot_device_class_id``), so ``RizonRobot`` itself is
    not modified.

    NOTE (phase 1): external-force ("collision wrench") arm stop is
    intentionally NOT implemented here; add it later either in this device
    (reading the wrench from the robot state SHM) or inside ``RizonRobot``.
    The gripper force-limit (motor iq) gate from R1 is likewise deferred to
    the gripper backend inside ``RizonRobot``.
    """

    # --- coordinate frames (ported verbatim from R1) ---
    R_QUEST_TO_FLEXIV = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    R_NATURAL_FLEXIV = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    STATE_WAIT = "WAIT"
    STATE_FULL = "FULL"

    # Head SHM layout (float64):
    #   [0:16]  head 4x4 matrix (row-major flatten)
    #   [16]    head_visible (1.0 / 0.0)
    #   [17:25] left controller buttons (up to 8)
    #   [25]    left_controller_visible (1.0 / 0.0)
    HEAD_DATA_DIM = 26

    def __init__(
        self,
        device_class_id: int = 0,
        device_name: Optional[str] = None,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        # --- robot state SHM (clutch TCP anchor) ---
        robot_device_name: str = "RizonRobot",
        robot_device_class_id: int = 1,
        # --- Quest / Vuer viewer ---
        host: str = "0.0.0.0",
        port: int = 8012,
        webrtc_src: str = "",
        cert_file: str = "certs/cert.pem",
        key_file: str = "certs/key.pem",
        xr_frame_rate: int = 120,
        world_locked: bool = False,
        video_height: float = 2.5,
        video_y: float = 1.0,
        video_z: float = -2.5,
        skip_patch: bool = False,
        allow_cache: bool = False,
        vuer_viewer_path: Optional[str] = None,
        # --- Flexiv following params (R1 defaults) ---
        flexiv_orientation_mode: str = "relative",
        flexiv_pos_scale: float = 1.0,
        flexiv_max_step_m: float = 0.005,
        flexiv_max_step_rad: float = 0.05,
        flexiv_workspace_radius: float = 0.30,
        deadman: bool = True,
        enable_left: bool = True,
        enable_right: bool = True,
        gripper_open_value: float = 1.0,
        # RIGHT-B one-shot homing: file written for RizonRobot's
        # _check_and_execute_home_command (joint-space go-home). MUST match the
        # robot's signal_files.robot_home_command_pattern (runtime/default.yaml).
        home_command_pattern: str = "/tmp/superinference_robot_{device_name}_{device_class_id}_home_command",
        home_preset_index: Optional[int] = None,
        # RIGHT-B also homes the YAM head arm (back to yam_initial_q + WAIT).
        # Set yam_device_name to "" / None to skip the head arm.
        yam_device_name: Optional[str] = "YamHeadRobot",
        yam_device_class_id: int = 0,
        # Controller-channel diagnostics (~1 Hz HAND_MOVE / visibility logging).
        # Off by default; enable via device config `diag: true` when debugging
        # Quest connectivity.
        diag: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            device_name=device_name or "Quest3VuerTeleoperator",
            state_feedback=state_feedback,
        )

        # Robot state SHM (read-only TCP anchor source).
        self.robot_device_name = robot_device_name
        self.robot_device_class_id = robot_device_class_id
        self.robot_state_shm = None

        # Quest viewer config.
        self.host_ip = host
        self.port = int(port)
        self.webrtc_src = webrtc_src
        self.cert_file = cert_file
        self.key_file = key_file
        self.xr_frame_rate = int(xr_frame_rate)
        self.world_locked = bool(world_locked)
        self.video_height = float(video_height)
        self.video_y = float(video_y)
        self.video_z = float(video_z)
        self.skip_patch = bool(skip_patch)
        self.allow_cache = bool(allow_cache)
        self.vuer_viewer_path = vuer_viewer_path
        self._viewer = None

        # Flexiv following params.
        self.flexiv_orientation_mode = flexiv_orientation_mode
        self.flexiv_pos_scale = float(flexiv_pos_scale)
        self.flexiv_max_step_m = float(flexiv_max_step_m)
        self.flexiv_max_step_rad = float(flexiv_max_step_rad)
        self.flexiv_workspace_radius = float(flexiv_workspace_radius)
        self.deadman = bool(deadman)
        self.enable_left = bool(enable_left)
        self.enable_right = bool(enable_right)
        self.gripper_open_value = float(gripper_open_value)

        # RIGHT-B one-shot homing.
        self._home_command_pattern = str(home_command_pattern)
        self._home_preset_index = home_preset_index
        self.yam_device_name = yam_device_name
        self.yam_device_class_id = yam_device_class_id

        # State machine + per-side state.
        self._flx_state = self.STATE_WAIT
        self._left = _VuerFlexivSide("left", self.gripper_open_value)
        self._right = _VuerFlexivSide("right", self.gripper_open_value)

        # Button edge bookkeeping.
        self._ra_prev = False
        self._rb_prev = False
        self._lgrip_prev = False
        self._rgrip_prev = False

        # Controller-channel diagnostics (throttled ~1 Hz in _step); gated by
        # the `diag` config flag.
        self._diag_enabled = bool(diag)
        self._diag_last_t = 0.0
        self._diag_last_hand_move = 0

        # Secondary head SHM.
        self._head_shm = None
        self._head_shm_name = f"{self.device_name}_{self.device_class_id}_head"
        self._head_buffer_size = self.buffer_size
        self._head_write_index = 0

        self.running = False

        logger.info(
            f"Quest3VuerTeleoperator initialized: data SHM={self.shared_memory_name}, "
            f"head SHM={self._head_shm_name}, robot anchor="
            f"{self.robot_device_name}_{self.robot_device_class_id}, "
            f"orientation_mode={self.flexiv_orientation_mode}, deadman={self.deadman}"
        )

    # ------------------------------------------------------------------
    # Math helpers (ported from R1)
    # ------------------------------------------------------------------
    @staticmethod
    def _orthonormalize(R3: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(R3)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1.0
            Rn = U @ Vt
        return Rn

    @staticmethod
    def _is_identity_matrix(M: np.ndarray, tol: float = 1e-9) -> bool:
        if np.allclose(M, 0.0, atol=tol):
            return True
        return np.allclose(M, np.eye(4), atol=tol)

    def _slerp_step(self, R_from: np.ndarray, R_to: np.ndarray, max_step_rad: float) -> np.ndarray:
        R_delta = R_to @ R_from.T
        ang = np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
        if ang <= max_step_rad or ang < 1e-9:
            return R_to.copy()
        rotvec = R.from_matrix(R_delta).as_rotvec()
        rotvec_clipped = rotvec * (max_step_rad / ang)
        R_step = R.from_rotvec(rotvec_clipped).as_matrix()
        return R_step @ R_from

    @staticmethod
    def _tcp7_wxyz_to_T(tcp7: np.ndarray) -> np.ndarray:
        """Robot-state TCP [x,y,z, qw,qx,qy,qz] -> 4x4."""
        T = np.eye(4)
        T[:3, 3] = tcp7[:3]
        q_wxyz = tcp7[3:7]
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
        return T

    @staticmethod
    def _T_to_pos_quat_xyzw(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos = T[:3, 3].copy()
        quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
        return pos, quat_xyzw

    @staticmethod
    def _btn(btns, idx: int, thresh: float = 0.5) -> bool:
        return idx < len(btns) and float(btns[idx]) > thresh

    @staticmethod
    def _trigger(btns) -> float:
        if len(btns) < 1:
            return 0.0
        return float(np.clip(btns[0], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Robot state SHM (TCP anchor)
    # ------------------------------------------------------------------
    def _connect_robot_state_shm(self) -> None:
        if self.robot_state_shm is not None:
            return
        try:
            self.robot_state_shm = connect_to_device_shm(
                device_name=self.robot_device_name,
                device_class_id=self.robot_device_class_id,
                read_only=True,
            )
        except FileNotFoundError:
            logger.warning(
                f"Robot state SHM not found for {self.robot_device_name}_{self.robot_device_class_id}"
            )
        except Exception as exc:
            logger.warning(f"Failed to connect robot state SHM: {exc}")

    def _read_latest_robot_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return commanded TCP anchors, falling back to measured legacy slots."""
        self._connect_robot_state_shm()
        if self.robot_state_shm is None:
            return None, None

        buf = self.robot_state_shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )
        if buffer_header["current_frames_count"] == 0:
            return None, None

        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size

        frame_header = unpack_frame_header(buf[frame_offset : frame_offset + FRAME_HEADER_SIZE])
        if frame_header["timestamp_ns"] == 0:
            return None, None

        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        dtype = get_dtype(device_header["data_dtype"], np.uint8)
        data_array = np.frombuffer(buf[data_start : data_start + data_size], dtype=dtype)
        flat_data = data_array.reshape(-1)

        if flat_data.shape[0] >= RizonDualArmStateLayout.ARM_DETAIL_SIZE:
            return (
                flat_data[RizonDualArmStateLayout.LEFT_COMMAND_TCP].copy(),
                flat_data[RizonDualArmStateLayout.RIGHT_COMMAND_TCP].copy(),
            )
        if flat_data.shape[0] >= RizonDualArmStateLayout.BASE_SIZE:
            return (
                flat_data[RizonDualArmStateLayout.LEFT_MEASURED_TCP].copy(),
                flat_data[RizonDualArmStateLayout.RIGHT_MEASURED_TCP].copy(),
            )
        if flat_data.shape[0] >= RizonDualArmStateLayout.LEFT_MEASURED_TCP.stop:
            return flat_data[RizonDualArmStateLayout.LEFT_MEASURED_TCP].copy(), None
        return None, None

    # ------------------------------------------------------------------
    # Quest viewer
    # ------------------------------------------------------------------
    def _make_viewer(self):
        """Instantiate the vendored QuestVrWebRTCViewer.

        Dependencies (Vuer / aiortc / the vendored quest_vuer module) are
        assumed present; ``vuer_viewer_path`` may point at the directory
        containing ``test_vuer_webrtc_vr.py`` if it is not importable.
        """
        import sys as _sys

        if self.vuer_viewer_path:
            p = str(Path(self.vuer_viewer_path).resolve())
            if p not in _sys.path:
                _sys.path.insert(0, p)

        QuestVrWebRTCViewer = None
        patch_vuer = None
        install_nocache_patch = None
        try:
            from third_party.quest_vuer.test_vuer_webrtc_vr import (  # type: ignore
                QuestVrWebRTCViewer,
                patch_vuer,
                install_nocache_patch,
            )
        except Exception:
            from test_vuer_webrtc_vr import (  # type: ignore
                QuestVrWebRTCViewer,
            )
            try:
                from vuer_controller_patch import patch_vuer  # type: ignore
                from vuer_nocache import install_nocache_patch  # type: ignore
            except Exception:
                patch_vuer = None
                install_nocache_patch = None

        from types import SimpleNamespace

        args = SimpleNamespace(
            host=self.host_ip,
            port=self.port,
            webrtc_src=self.webrtc_src,
            cert_file=self.cert_file,
            key_file=self.key_file,
            height=self.video_height,
            y=self.video_y,
            z=self.video_z,
            world_locked=self.world_locked,
            xr_frame_rate=self.xr_frame_rate,
        )

        if not self.skip_patch and patch_vuer is not None:
            try:
                patch_vuer()
            except Exception as exc:
                logger.warning(f"patch_vuer() failed (continuing): {exc}")
        if not self.allow_cache and install_nocache_patch is not None:
            try:
                install_nocache_patch()
            except Exception as exc:
                logger.warning(f"install_nocache_patch() failed (continuing): {exc}")

        return QuestVrWebRTCViewer(args)

    # ------------------------------------------------------------------
    # Head SHM (secondary output for YamHeadRobot)
    # ------------------------------------------------------------------
    def _create_head_shm(self) -> None:
        try:
            from utils.shm_utils import pack_device_header, pack_buffer_header
            import multiprocessing.shared_memory as shm

            head_shape = (self.HEAD_DATA_DIM,)
            numpy_dtype = get_dtype(self.data_dtype)
            bytes_per_element = np.dtype(numpy_dtype).itemsize
            data_size = int(np.prod(head_shape)) * bytes_per_element
            frame_size = FRAME_HEADER_SIZE + data_size
            total_shm_size = (
                DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + self._head_buffer_size * frame_size
            )

            try:
                existing = shm.SharedMemory(name=self._head_shm_name, create=False)
                existing.close()
                existing.unlink()
                logger.info(f"Removed existing head SHM: {self._head_shm_name}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.debug(f"Could not check existing head SHM: {exc}")

            self._head_shm = shm.SharedMemory(
                name=self._head_shm_name, create=True, size=total_shm_size
            )
            device_header = pack_device_header(
                self.device_name,
                self.device_class_id,
                self.fps,
                self.data_dtype,
                head_shape,
                frame_size,
                self.hardware_latency_ms,
            )
            self._head_shm.buf[:DEVICE_HEADER_SIZE] = device_header
            buffer_header = pack_buffer_header(self._head_buffer_size, 0, 0)
            self._head_shm.buf[
                DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            ] = buffer_header
            self._head_write_index = 0
            logger.info(
                f"Created head SHM: {self._head_shm_name} ({total_shm_size} bytes, "
                f"shape={head_shape})"
            )
        except Exception as exc:
            logger.error(f"Failed to create head SHM: {exc}")
            self._head_shm = None

    def _write_head_data(self, head_array: np.ndarray, timestamp_ns: int) -> None:
        if self._head_shm is None:
            return
        try:
            from utils.shm_utils import pack_frame_header, pack_buffer_header

            device_info = unpack_device_header(bytes(self._head_shm.buf[:DEVICE_HEADER_SIZE]))
            frame_size = device_info["frame_size"]
            numpy_dtype = get_dtype(self.data_dtype)
            payload = np.ascontiguousarray(head_array, dtype=numpy_dtype)

            frame_offset = (
                DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + self._head_write_index * frame_size
            )
            frame_header = pack_frame_header(timestamp_ns)
            self._head_shm.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = frame_header

            data_start = frame_offset + FRAME_HEADER_SIZE
            data_bytes = payload.tobytes()
            self._head_shm.buf[data_start : data_start + len(data_bytes)] = data_bytes

            current_frames_count = min(self._head_write_index + 1, self._head_buffer_size)
            buffer_header = pack_buffer_header(
                self._head_buffer_size, current_frames_count, self._head_write_index
            )
            self._head_shm.buf[
                DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            ] = buffer_header
            self._head_write_index = (self._head_write_index + 1) % self._head_buffer_size
        except Exception as exc:
            logger.error(f"Failed to write head data: {exc}")

    def _cleanup_head_shm(self) -> None:
        if self._head_shm is None:
            return
        try:
            self._head_shm.close()
            self._head_shm.unlink()
        except Exception as exc:
            logger.warning(f"Failed to cleanup head SHM: {exc}")
        self._head_shm = None

    # ------------------------------------------------------------------
    # Core following logic (ported from R1 _command_flexiv_side)
    # ------------------------------------------------------------------
    def _command_flexiv_side(self, side: _VuerFlexivSide, ctrl_mat: np.ndarray) -> np.ndarray:
        """Update ``side.T_tcp_cmd`` from a fresh controller pose. Returns it."""
        R_ctrl_now = self._orthonormalize(ctrl_mat[:3, :3])
        p_ctrl_now = ctrl_mat[:3, 3]
        R_ctrl_ref = side.T_ctrl_ref[:3, :3]
        p_ctrl_ref = side.T_ctrl_ref[:3, 3]

        if self.flexiv_orientation_mode == "absolute":
            R_tcp_target = (
                self.R_QUEST_TO_FLEXIV
                @ R_ctrl_now
                @ self.R_QUEST_TO_FLEXIV.T
                @ self.R_NATURAL_FLEXIV
            )
        else:
            R_delta_q = R_ctrl_now @ R_ctrl_ref.T
            R_delta_f = self.R_QUEST_TO_FLEXIV @ R_delta_q @ self.R_QUEST_TO_FLEXIV.T
            R_tcp_target = R_delta_f @ side.T_tcp_ref[:3, :3]

        p_delta_q = p_ctrl_now - p_ctrl_ref
        p_delta_f = self.R_QUEST_TO_FLEXIV @ p_delta_q
        p_tcp_target = side.T_tcp_ref[:3, 3] + self.flexiv_pos_scale * p_delta_f

        # Spherical workspace clamp around this side's engage TCP.
        dp = p_tcp_target - side.T_tcp_ref[:3, 3]
        r = float(np.linalg.norm(dp))
        if r > self.flexiv_workspace_radius:
            p_tcp_target = side.T_tcp_ref[:3, 3] + dp * (self.flexiv_workspace_radius / r)

        # Rate limit relative to last commanded TCP.
        p_step = p_tcp_target - side.T_tcp_cmd[:3, 3]
        step_norm = float(np.linalg.norm(p_step))
        if step_norm > self.flexiv_max_step_m:
            p_step *= self.flexiv_max_step_m / step_norm
        p_new = side.T_tcp_cmd[:3, 3] + p_step
        R_new = self._slerp_step(side.T_tcp_cmd[:3, :3], R_tcp_target, self.flexiv_max_step_rad)

        T_cmd = np.eye(4)
        T_cmd[:3, 3] = p_new
        T_cmd[:3, :3] = R_new
        side.T_tcp_cmd = T_cmd
        return T_cmd

    def _hold_at_robot_tcp(self, side: _VuerFlexivSide, tcp7: Optional[np.ndarray]) -> None:
        """Pin a side's command to the live robot TCP (used in WAIT)."""
        if tcp7 is not None:
            side.T_tcp_cmd = self._tcp7_wxyz_to_T(tcp7)

    def _request_robot_home(self) -> None:
        """Drop the home-command file so RizonRobot does a joint-space go-home.

        Reuses RizonRobot._check_and_execute_home_command: the robot reads the
        preset index from the file (empty -> its default preset), clears the
        control SHM and moves both arms to the configured home joints.
        """
        try:
            path = Path(
                self._home_command_pattern.format(
                    device_name=self.robot_device_name,
                    device_class_id=self.robot_device_class_id,
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            content = "" if self._home_preset_index is None else str(int(self._home_preset_index))
            path.write_text(content)
            logger.info(f"[home] RIGHT B -> requested go-home via {path}")
        except Exception as exc:
            logger.error(f"[home] failed to write home command file: {exc}")

        # Also home the YAM head arm (-> yam_initial_q + WAIT), if configured.
        if self.yam_device_name:
            try:
                yam_path = Path(
                    self._home_command_pattern.format(
                        device_name=self.yam_device_name,
                        device_class_id=self.yam_device_class_id,
                    )
                )
                yam_path.parent.mkdir(parents=True, exist_ok=True)
                yam_path.write_text("")
                logger.info(f"[home] RIGHT B -> requested YAM go-home via {yam_path}")
            except Exception as exc:
                logger.error(f"[home] failed to write YAM home command file: {exc}")

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------
    def start_server(self) -> None:
        if self.running:
            logger.info(f"Quest3VuerTeleoperator {self.device_class_id} already running")
            return

        logger.info(f"Starting Quest3VuerTeleoperator {self.device_class_id} server...")
        self.running = True

        # Main 16D EEF SHM + secondary head SHM.
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        self._create_head_shm()

        # Start Vuer viewer (browser stack + WebRTC video plane).
        self._viewer = self._make_viewer()
        self._viewer.start()
        logger.info(f"Vuer viewer started on https://{self.host_ip}:{self.port}")

        # Wait for robot state SHM so the clutch has a valid TCP anchor.
        logger.disable(__name__)
        logger.disable("utils.shm_utils")
        while self.running and self.robot_state_shm is None:
            try:
                self._connect_robot_state_shm()
            except Exception:
                pass
            if self.robot_state_shm is None:
                time.sleep(0.1)
        logger.enable(__name__)
        logger.enable("utils.shm_utils")
        logger.info(f"Robot state SHM connected: {self.robot_state_shm}")

        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        prev_left_mat = None
        prev_right_mat = None

        while self.running:
            try:
                output = self._step(prev_left_mat, prev_right_mat)
                if output is not None:
                    out_array, head_array, prev_left_mat, prev_right_mat = output
                    timestamp_ns = time.time_ns()
                    self._write_array_to_shm_with_timestamp(out_array, timestamp_ns)
                    self._write_head_data(head_array, timestamp_ns)
                wait_for_next_iteration()
            except Exception as exc:
                logger.error(f"Error in Quest3VuerTeleoperator loop: {exc}")
                break

    def _step(self, prev_left_mat, prev_right_mat):
        """One control tick. Returns (out16, head26, prev_left, prev_right)."""
        viewer = self._viewer

        head_mat = viewer.head_matrix
        head_ok = not self._is_identity_matrix(head_mat)
        left_visible = viewer.left_controller_visible
        right_visible = viewer.right_controller_visible
        l_btns = viewer.left_buttons
        r_btns = viewer.right_buttons
        left_mat = viewer.left_controller_matrix
        right_mat = viewer.right_controller_matrix
        left_pose_ok = left_visible and not self._is_identity_matrix(left_mat)
        right_pose_ok = right_visible and not self._is_identity_matrix(right_mat)

        # Button edges.
        ra = self._btn(r_btns, 4)
        ra_edge = ra and not self._ra_prev
        self._ra_prev = ra

        # RIGHT B (gamepad index 5): one-shot "return to home".
        rb = self._btn(r_btns, 5)
        rb_edge = rb and not self._rb_prev
        self._rb_prev = rb

        lgrip_raw = self._btn(l_btns, 1)
        rgrip_raw = self._btn(r_btns, 1)
        lgrip_active = lgrip_raw if self.deadman else True
        rgrip_active = rgrip_raw if self.deadman else True
        lgrip_edge_on = lgrip_active and not self._lgrip_prev
        rgrip_edge_on = rgrip_active and not self._rgrip_prev
        self._lgrip_prev = lgrip_active
        self._rgrip_prev = rgrip_active

        ltrig_val = self._trigger(l_btns)
        rtrig_val = self._trigger(r_btns)

        # --- controller-channel diagnostics (~1 Hz) ---------------------------
        # If HAND_MOVE events stop arriving (Quest not in an immersive session,
        # Vuer websocket dropped, hand-tracking instead of controllers, stale
        # cached page), buttons/visibility freeze and engage silently no-ops.
        # This log makes that observable: events/s, visibility, button counts.
        # Gated by the `diag` config flag (off by default).
        now_diag = time.time()
        if self._diag_enabled and now_diag - self._diag_last_t >= 1.0:
            hm = getattr(viewer, "hand_move_count", -1)
            hm_rate = hm - self._diag_last_hand_move if hm >= 0 else -1
            self._diag_last_hand_move = hm
            self._diag_last_t = now_diag
            if hm_rate == 0:
                logger.warning(
                    "[diag] NO HAND_MOVE events in last 1s — Quest not streaming "
                    "controllers (enter VR / wake controllers / reload :%d). "
                    "L_vis=%s R_vis=%s head_ok=%s"
                    % (self.port, left_visible, right_visible, head_ok)
                )
            else:
                logger.info(
                    "[diag] HAND_MOVE=%s/s | L_vis=%s(btn%d) R_vis=%s(btn%d) "
                    "head_ok=%s RA=%.0f Lgrip=%.2f Rgrip=%.2f state=%s"
                    % (
                        hm_rate, left_visible, len(l_btns), right_visible,
                        len(r_btns), head_ok, 1.0 if ra else 0.0,
                        lgrip_raw, rgrip_raw, self._flx_state,
                    )
                )

        left_tcp7, right_tcp7 = self._read_latest_robot_state()

        # RIGHT B: disengage to WAIT and ask RizonRobot to go to home joints.
        # Going to WAIT first stops teleop tracking so the held targets re-pin to
        # the robot TCP (which becomes the home pose once homing finishes).
        if rb_edge:
            self._flx_state = self.STATE_WAIT
            for s, tcp in ((self._right, right_tcp7), (self._left, left_tcp7)):
                s.T_ctrl_ref = s.T_tcp_ref = None
                self._hold_at_robot_tcp(s, tcp)
            self._request_robot_home()

        # RIGHT A toggles BOTH arms WAIT <-> FULL.
        if ra_edge:
            if self._flx_state == self.STATE_WAIT:
                missing = []
                if self.enable_right and not right_pose_ok:
                    missing.append("RIGHT")
                if self.enable_left and not left_pose_ok:
                    missing.append("LEFT")
                if missing:
                    logger.warning(
                        f"[state] RIGHT A pressed but controller(s) not visible: "
                        f"{','.join(missing)} — wake them and retry"
                    )
                else:
                    self._flx_state = self.STATE_FULL
                    self._left.T_ctrl_ref = self._left.T_tcp_ref = None
                    self._right.T_ctrl_ref = self._right.T_tcp_ref = None
                    logger.info(f"[state] FLEXIV -> FULL (mode={self.flexiv_orientation_mode})")
            else:
                self._flx_state = self.STATE_WAIT
                for s, tcp in ((self._right, right_tcp7), (self._left, left_tcp7)):
                    s.T_ctrl_ref = s.T_tcp_ref = None
                    self._hold_at_robot_tcp(s, tcp)
                logger.info("[state] FLEXIV -> WAIT (RIGHT A disengage)")

        # Per-side clutch: capture refs on GRIP rising edge.
        if (
            self.enable_right
            and rgrip_edge_on
            and self._flx_state == self.STATE_FULL
            and right_pose_ok
            and right_tcp7 is not None
        ):
            self._right.T_ctrl_ref = right_mat.copy()
            self._right.T_ctrl_ref[:3, :3] = self._orthonormalize(self._right.T_ctrl_ref[:3, :3])
            self._right.T_tcp_ref = self._tcp7_wxyz_to_T(right_tcp7)
            self._right.T_tcp_cmd = self._right.T_tcp_ref.copy()
            logger.info("[clutch] FLEXIV-right: refs captured")
        if (
            self.enable_left
            and lgrip_edge_on
            and self._flx_state == self.STATE_FULL
            and left_pose_ok
            and left_tcp7 is not None
        ):
            self._left.T_ctrl_ref = left_mat.copy()
            self._left.T_ctrl_ref[:3, :3] = self._orthonormalize(self._left.T_ctrl_ref[:3, :3])
            self._left.T_tcp_ref = self._tcp7_wxyz_to_T(left_tcp7)
            self._left.T_tcp_cmd = self._left.T_tcp_ref.copy()
            logger.info("[clutch] FLEXIV-left: refs captured")

        # Per-side tracking flags.
        right_tracking = (
            self.enable_right
            and self._flx_state == self.STATE_FULL
            and self._right.T_ctrl_ref is not None
            and rgrip_active
            and right_pose_ok
        )
        left_tracking = (
            self.enable_left
            and self._flx_state == self.STATE_FULL
            and self._left.T_ctrl_ref is not None
            and lgrip_active
            and left_pose_ok
        )

        # Update each side's commanded TCP. Whenever a side is NOT actively
        # tracking (WAIT, or FULL with the clutch released / controller lost),
        # keep its commanded TCP pinned to the LIVE robot pose. This prevents a
        # stale pre-home target from lingering after a RIGHT-B go-home: once the
        # robot's published state catches up to the home pose, the command follows
        # it instead of snapping the arm back to its historical pose on re-engage.
        if right_tracking:
            self._command_flexiv_side(self._right, right_mat)
        else:
            self._hold_at_robot_tcp(self._right, right_tcp7)
        if left_tracking:
            self._command_flexiv_side(self._left, left_mat)
        else:
            self._hold_at_robot_tcp(self._left, left_tcp7)

        # Grippers: press=close, release=open (target = 1 - trigger) while tracking.
        if right_tracking:
            self._right.last_gripper = 1.0 - rtrig_val
        if left_tracking:
            self._left.last_gripper = 1.0 - ltrig_val

        # If a side still has no command (no robot state yet), skip this tick.
        if self._left.T_tcp_cmd is None or self._right.T_tcp_cmd is None:
            return None

        # Build the 16D eef_abs output: [L_xyz, L_quat_xyzw, L_grip, R_...].
        out = np.zeros(self.data_shape[0], dtype=get_dtype(self.data_dtype))
        l_pos, l_quat = self._T_to_pos_quat_xyzw(self._left.T_tcp_cmd)
        r_pos, r_quat = self._T_to_pos_quat_xyzw(self._right.T_tcp_cmd)
        out[0:3] = l_pos
        out[3:7] = l_quat
        out[7] = self._left.last_gripper
        out[8:11] = r_pos
        out[11:15] = r_quat
        out[15] = self._right.last_gripper

        # Build the head SHM payload for YamHeadRobot.
        head = np.zeros(self.HEAD_DATA_DIM, dtype=get_dtype(self.data_dtype))
        head[0:16] = np.asarray(head_mat, dtype=float).reshape(-1)
        head[16] = 1.0 if head_ok else 0.0
        n_btn = min(len(l_btns), 8)
        if n_btn > 0:
            head[17 : 17 + n_btn] = np.asarray(l_btns[:n_btn], dtype=float)
        head[25] = 1.0 if left_visible else 0.0

        return out, head, prev_left_mat, prev_right_mat

    def stop_server(self) -> None:
        if not self.running:
            return
        logger.info(f"Stopping Quest3VuerTeleoperator {self.device_class_id} server...")
        self.running = False

        if self.robot_state_shm is not None:
            try:
                self.robot_state_shm.close()
            except Exception as exc:
                logger.warning(f"Failed to close robot state SHM: {exc}")
            self.robot_state_shm = None

        self._cleanup_head_shm()
        self._cleanup_shared_memory()
        logger.info("Quest3VuerTeleoperator stopped")

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update(
            {
                "host": self.host_ip,
                "port": self.port,
                "webrtc_src": self.webrtc_src,
                "robot_device_name": self.robot_device_name,
                "robot_device_class_id": self.robot_device_class_id,
                "flexiv_orientation_mode": self.flexiv_orientation_mode,
                "deadman": self.deadman,
                "state": self._flx_state,
                "head_shm": self._head_shm_name,
            }
        )
        return info


class ReplayTeleoperator(BaseDevice):
    """Replay a recorded R1 NEDF3 episode back onto the live robot stack.

    Publishes its own 16-wide state SHM (``ReplayTeleoperator_0_data``), plus a
    7-wide head EE-pose SHM (``ReplayTeleoperator_0_head_cmd``) that
    ``YamHeadRobot`` (``head_pose_replay: true``) IKs onto the head arm. So the
    existing servo / IK / force-gating / safety logic moves the real robot; no
    teleop hardware involved.

    Recorded source topics (written by ``incubator/r1/async_data_saver_NEDF3_r1.py``):
        nmx/hal/arm/left/tcp/cmd   -> out[0:7]   (pose7 xyzw, base frame, m)
        nmx/hal/hand/left/cmd      -> out[7]     (width_m -> normalized 0..1)
        nmx/hal/arm/right/tcp/cmd  -> out[8:15]
        nmx/hal/hand/right/cmd     -> out[15]
        nmx/hal/head/cmd           -> head_cmd pose7 (xyzw, YAM base frame)
    """

    HEAD_CMD_DIM = 7

    def __init__(
        self,
        device_class_id: int = 0,
        device_name: Optional[str] = None,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        # --- replay source / pacing ---
        episode: str = "saved_data/episode_0000",
        speed: float = 1.0,
        loop: bool = False,
        warmup_s: float = 2.0,
        # Gripper width(m) -> normalized. Defaults come from the shared LK preset.
        gripper_min_width: float = 0.0,
        gripper_max_width: Optional[float] = None,
        gripper_type: str = DEFAULT_LK_GRIPPER_TYPE,
        gripper_arm_length_m: Optional[float] = None,
        gripper_apex_angle_max_deg: Optional[float] = None,
        # also stream the head EE pose for YamHeadRobot(head_pose_replay)
        enable_head: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            device_name=device_name or "ReplayTeleoperator",
            state_feedback=state_feedback,
        )
        self.episode = episode
        self.speed = max(1e-3, float(speed))
        self.loop = bool(loop)
        self.warmup_s = float(warmup_s)
        self.gripper_min_width = float(gripper_min_width)
        gripper_preset = get_lk_gripper_preset(gripper_type)
        arm_length_m = (
            float(gripper_arm_length_m)
            if gripper_arm_length_m is not None
            else float(gripper_preset["arm_length_m"])
        )
        apex_angle_max_deg = (
            float(gripper_apex_angle_max_deg)
            if gripper_apex_angle_max_deg is not None
            else float(gripper_preset["apex_angle_max_deg"])
        )
        self.gripper_max_width = (
            float(gripper_max_width)
            if gripper_max_width is not None
            else calculate_lk_gripper_max_width(
                arm_length_m,
                apex_angle_max_deg,
                self.gripper_min_width,
            )
        )
        if self.gripper_max_width <= self.gripper_min_width:
            raise ValueError(
                "gripper_max_width must be greater than gripper_min_width"
            )
        self.enable_head = bool(enable_head)

        # Secondary head EE-pose SHM (mirrors Quest3VuerTeleoperator's head SHM
        # plumbing, but carries a 7-wide pose7 target instead of the 26-wide
        # headset frame).
        self._head_shm = None
        self._head_shm_name = f"{self.device_name}_{self.device_class_id}_head_cmd"
        self._head_buffer_size = self.buffer_size
        self._head_write_index = 0

        # Filled in start_server() by _load_episode().
        self._left_pose: list = []
        self._right_pose: list = []
        self._left_grip: list = []
        self._right_grip: list = []
        self._head_pose: list = []
        self._master: list = []

        self.running = False
        logger.info(
            f"ReplayTeleoperator initialized: data SHM={self.shared_memory_name}, "
            f"head SHM={self._head_shm_name}, episode={self.episode}, "
            f"speed={self.speed}, loop={self.loop}, enable_head={self.enable_head}"
        )

    # ------------------------------------------------------------------
    # Episode loading (nmx_nedf_api)
    # ------------------------------------------------------------------
    def _load_episode(self) -> None:
        from nmx_nedf_api import NEDFFactory, NEDFReaderConfig

        ep = Path(self.episode).expanduser()
        meta = ep / "metadata.json" if ep.is_dir() else ep
        if not meta.exists():
            raise FileNotFoundError(f"Episode metadata not found: {meta}")
        reader = NEDFFactory.get_reader(NEDFReaderConfig(metadata_file_path=meta))

        def pose_series(topic: str) -> list:
            out = []
            try:
                for msg in reader.iter_decoded_messages(topic):
                    d = msg.decoded_message
                    p = d.pose.pos
                    q = d.pose.quat  # stored w,x,y,z
                    out.append(
                        (
                            int(msg.message.log_time),
                            np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64),
                        )
                    )
            except Exception as exc:
                logger.warning(f"[replay] no data for {topic}: {exc}")
                return []
            out.sort(key=lambda x: x[0])
            return out

        def grip_series(topic: str) -> list:
            out = []
            try:
                for msg in reader.iter_decoded_messages(topic):
                    d = msg.decoded_message
                    for j in d.joints:
                        out.append((int(msg.message.log_time), float(j.pos)))
                        break
            except Exception as exc:
                logger.warning(f"[replay] no data for {topic}: {exc}")
                return []
            out.sort(key=lambda x: x[0])
            return out

        self._left_pose = pose_series("nmx/hal/arm/left/tcp/cmd")
        self._right_pose = pose_series("nmx/hal/arm/right/tcp/cmd")
        self._left_grip = grip_series("nmx/hal/hand/left/cmd")
        self._right_grip = grip_series("nmx/hal/hand/right/cmd")
        self._head_pose = pose_series("nmx/hal/head/cmd")

        if not self._left_pose or not self._right_pose:
            raise RuntimeError(
                "[replay] episode missing arm tcp/cmd for one or both arms "
                f"(left={len(self._left_pose)}, right={len(self._right_pose)})"
            )
        # Master clock = the denser arm-pose stream.
        self._master = (
            self._left_pose
            if len(self._left_pose) >= len(self._right_pose)
            else self._right_pose
        )
        dur = (self._master[-1][0] - self._master[0][0]) / 1e9
        logger.info(
            f"[replay] loaded {self.episode}: master={len(self._master)} frames "
            f"({dur:.1f}s), L/R pose={len(self._left_pose)}/{len(self._right_pose)}, "
            f"L/R grip={len(self._left_grip)}/{len(self._right_grip)}, "
            f"head={len(self._head_pose)}"
        )

    @staticmethod
    def _nearest(series: list, ts: int):
        if not series:
            return None
        keys = [s[0] for s in series]
        i = bisect.bisect_left(keys, ts)
        if i == 0:
            return series[0][1]
        if i >= len(series):
            return series[-1][1]
        before, after = series[i - 1], series[i]
        return before[1] if (ts - before[0]) <= (after[0] - ts) else after[1]

    def _width_to_normalized(self, width_m: float) -> float:
        span = self.gripper_max_width - self.gripper_min_width
        if span <= 0:
            return 0.0
        return float(np.clip((width_m - self.gripper_min_width) / span, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Head EE-pose SHM (7-wide), same low-level layout as a device frame
    # ------------------------------------------------------------------
    def _create_head_cmd_shm(self) -> None:
        try:
            from utils.shm_utils import pack_device_header, pack_buffer_header
            import multiprocessing.shared_memory as shm

            head_shape = (self.HEAD_CMD_DIM,)
            numpy_dtype = get_dtype(self.data_dtype)
            bytes_per_element = np.dtype(numpy_dtype).itemsize
            data_size = int(np.prod(head_shape)) * bytes_per_element
            frame_size = FRAME_HEADER_SIZE + data_size
            total_shm_size = (
                DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + self._head_buffer_size * frame_size
            )
            try:
                existing = shm.SharedMemory(name=self._head_shm_name, create=False)
                existing.close()
                existing.unlink()
                logger.info(f"Removed existing head cmd SHM: {self._head_shm_name}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.debug(f"Could not check existing head cmd SHM: {exc}")

            self._head_shm = shm.SharedMemory(
                name=self._head_shm_name, create=True, size=total_shm_size
            )
            device_header = pack_device_header(
                self.device_name,
                self.device_class_id,
                self.fps,
                self.data_dtype,
                head_shape,
                frame_size,
                self.hardware_latency_ms,
            )
            self._head_shm.buf[:DEVICE_HEADER_SIZE] = device_header
            buffer_header = pack_buffer_header(self._head_buffer_size, 0, 0)
            self._head_shm.buf[
                DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            ] = buffer_header
            self._head_write_index = 0
            logger.info(
                f"Created head cmd SHM: {self._head_shm_name} ({total_shm_size} bytes, "
                f"shape={head_shape})"
            )
        except Exception as exc:
            logger.error(f"Failed to create head cmd SHM: {exc}")
            self._head_shm = None

    def _write_head_cmd(self, pose7: np.ndarray, timestamp_ns: int) -> None:
        if self._head_shm is None:
            return
        try:
            from utils.shm_utils import pack_frame_header, pack_buffer_header

            device_info = unpack_device_header(bytes(self._head_shm.buf[:DEVICE_HEADER_SIZE]))
            frame_size = device_info["frame_size"]
            numpy_dtype = get_dtype(self.data_dtype)
            payload = np.ascontiguousarray(pose7, dtype=numpy_dtype)

            frame_offset = (
                DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + self._head_write_index * frame_size
            )
            self._head_shm.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = (
                pack_frame_header(timestamp_ns)
            )
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_bytes = payload.tobytes()
            self._head_shm.buf[data_start : data_start + len(data_bytes)] = data_bytes

            current_frames_count = min(self._head_write_index + 1, self._head_buffer_size)
            self._head_shm.buf[
                DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            ] = pack_buffer_header(
                self._head_buffer_size, current_frames_count, self._head_write_index
            )
            self._head_write_index = (self._head_write_index + 1) % self._head_buffer_size
        except Exception as exc:
            logger.error(f"Failed to write head cmd: {exc}")

    def _cleanup_head_cmd_shm(self) -> None:
        if self._head_shm is None:
            return
        try:
            self._head_shm.close()
            self._head_shm.unlink()
        except Exception as exc:
            logger.warning(f"Failed to cleanup head cmd SHM: {exc}")
        self._head_shm = None

    # ------------------------------------------------------------------
    # Frame emission
    # ------------------------------------------------------------------
    def _emit_at(self, ts: int) -> None:
        out = np.zeros(self.data_shape[0], dtype=get_dtype(self.data_dtype))
        lp = self._nearest(self._left_pose, ts)
        rp = self._nearest(self._right_pose, ts)
        if lp is not None:
            out[0:7] = lp
        if rp is not None:
            out[8:15] = rp
        lg = self._nearest(self._left_grip, ts)
        rg = self._nearest(self._right_grip, ts)
        out[7] = self._width_to_normalized(lg) if lg is not None else 1.0
        out[15] = self._width_to_normalized(rg) if rg is not None else 1.0

        now_ns = time.time_ns()
        self._write_array_to_shm_with_timestamp(out, now_ns)

        if self.enable_head and self._head_shm is not None:
            hp = self._nearest(self._head_pose, ts)
            if hp is not None:
                self._write_head_cmd(np.asarray(hp, dtype=np.float64), now_ns)

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------
    def start_server(self) -> None:
        if self.running:
            logger.info(f"ReplayTeleoperator {self.device_class_id} already running")
            return
        logger.info(f"Starting ReplayTeleoperator {self.device_class_id} server...")
        self.running = True

        # Create the control SHM(s) FIRST so the robot's connect-with-retry
        # window finds them quickly, then load the (slower) episode.
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")
        if self.enable_head:
            self._create_head_cmd_shm()

        self._load_episode()

        ts0 = self._master[0][0]
        ts_end = self._master[-1][0]

        # Warm-up: hold the first recorded pose so the arm can servo to it
        # smoothly (Flexiv eef_abs rejects >0.5 m single-step jumps).
        if self.warmup_s > 0:
            logger.info(f"[replay] warm-up {self.warmup_s:.1f}s holding first frame...")
            warm_deadline = time.monotonic() + self.warmup_s
            while self.running and time.monotonic() < warm_deadline:
                self._emit_at(ts0)
                time.sleep(self.update_interval)

        logger.info(
            f"[replay] playback start (speed x{self.speed:.2f}, "
            f"{'loop' if self.loop else 'once'})"
        )
        while self.running:
            play_t0_wall = time.monotonic()
            for i in range(len(self._master)):
                if not self.running:
                    break
                ts = self._master[i][0]
                self._emit_at(ts)
                # Pace to recorded time (scaled by speed).
                target_elapsed = (ts - ts0) / 1e9 / self.speed
                sleep_for = target_elapsed - (time.monotonic() - play_t0_wall)
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 1.0))
            if not self.loop:
                break
            logger.info("[replay] loop -> restart")

        # Finished: hold the last frame so the arm stays put (does not go limp).
        logger.info("[replay] playback done — holding last frame")
        while self.running:
            self._emit_at(ts_end)
            time.sleep(self.update_interval)

    def stop_server(self) -> None:
        if not self.running:
            return
        logger.info(f"Stopping ReplayTeleoperator {self.device_class_id} server...")
        self.running = False
        self._cleanup_head_cmd_shm()
        self._cleanup_shared_memory()
        logger.info("ReplayTeleoperator stopped")

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update(
            {
                "episode": self.episode,
                "speed": self.speed,
                "loop": self.loop,
                "head_shm": self._head_shm_name,
            }
        )
        return info


def main():
    """Main function to run the joint slider teleoperator."""
    import argparse

    parser = argparse.ArgumentParser(description="Joint Slider Teleoperator for Robot Control")
    parser.add_argument("--device-class-id", "-d", type=int, default=0,
                        help="Device ID (default: 0)")
    parser.add_argument("--fps", "-f", type=float, default=10.0,
                        help="Update frequency in Hz (default: 10.0)")
    parser.add_argument("--num-joints", "-n", type=int, default=7,
                        help="Number of joints to control (default: 7)")
    parser.add_argument("--dtype", "-t", default="float64",
                        choices=['float32', 'float64'],
                        help="Data type (default: float64)")
    parser.add_argument("--buffer-size", "-b", type=int, default=100,
                        help="Buffer size in frames (default: 100)")

    args = parser.parse_args()

    state_feedback = {
        "fps": args.fps,
        "buffer_size": args.buffer_size,
        "hardware_latency_ms": 0.0,
        "format": {
            "shape": [args.num_joints],
            "dtype": args.dtype,
            "type": [f"j1-{args.num_joints}"],
        },
    }

    # Create teleoperator
    teleop = JointSlider(
        device_class_id=args.device_class_id,
        state_feedback=state_feedback,
    )

    logger.info("Joint Slider Teleoperator")
    logger.info("========================")
    logger.info(f"Device ID: {args.device_class_id}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Number of joints: {args.num_joints}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Buffer size: {args.buffer_size} frames")
    logger.info("")

    try:
        logger.info("Teleoperator is running. Use sliders to control joint positions...")
        logger.info(f"Device info: {teleop.get_device_info()}")
        teleop.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        teleop.stop_server()


if __name__ == "__main__":
    main()
