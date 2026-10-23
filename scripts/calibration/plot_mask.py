#!/usr/bin/env python3
"""
Trajectory Data Visualizer

This script visualizes trajectory data from episodes.zarr.zip files by displaying
two camera video streams using OpenCV. It supports synchronized playback of
left and right camera feeds.

Features:
- Load trajectory data from zarr format
- Display two camera streams in separate OpenCV windows
- Synchronized playback with keyboard controls
- Support for different camera naming conventions

Usage:
    python scripts/calibration/plot_mask.py --trajectory_path robot_data/episode_0000.hdf5
    python plot_mask.py --trajectory_path episodes.zarr.zip
    python plot_mask.py --trajectory_path episodes.zarr.zip --episode_idx 0
    python plot_mask.py --trajectory_path episodes.zarr.zip --episode_idx 0 --camera_left img_left --camera_right img_right

Author: Assistant
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import h5py
from tqdm import tqdm
import time

# Get the project root directory (parent of scripts directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from utils.logger_config import logger


class TrajectoryVisualizer:
    """Visualizer for trajectory data with dual camera display and mask drawing."""
    
    def __init__(self, trajectory_path: Path, episode_idx: int = 0, 
                 camera_left: str = "OpenCVCameraDevice_1", camera_right: str = "OpenCVCameraDevice_6",
                 playback_speed: float = 1.0, window_size: tuple = (640, 480), enable_mask_drawing: bool = True):
        """
        Initialize the trajectory visualizer.
        
        Args:
            trajectory_path: Path to the HDF5 file
            episode_idx: Index of the episode to visualize (not used for single file)
            camera_left: Key for left camera data
            camera_right: Key for right camera data
            playback_speed: Playback speed multiplier (1.0 = normal speed)
            window_size: Tuple of (width, height) for window size
            enable_mask_drawing: Enable mask drawing functionality
        """
        self.trajectory_path = Path(trajectory_path)
        self.episode_idx = episode_idx
        self.camera_left = camera_left
        self.camera_right = camera_right
        self.playback_speed = playback_speed
        self.window_size = window_size
        self.enable_mask_drawing = enable_mask_drawing
        
        # Mask drawing variables - separate for left and right cameras
        self.mask_drawing_mode = "ellipse"  # "ellipse" or "polygon"
        self.current_camera = "left"  # Which camera is being edited
        self.ellipse_created = {"left": False, "right": False}  # Track if ellipse has been created
        
        # Left camera mask data
        self.left_ellipse_center = None
        self.left_ellipse_axes = None
        self.left_ellipse_angle = 0
        self.left_polygon_points = []
        self.left_drawing_polygon = False
        self.left_mask = None
        self.left_mask_overlay = None
        self.left_dragging_ellipse = False
        self.left_resizing_ellipse = False
        self.left_ellipse_initial_axes = None
        
        # Separate masks for left camera
        self.left_ellipse_mask = None
        self.left_polygon_mask = None
        self.left_ellipse_mask_overlay = None
        self.left_polygon_mask_overlay = None
        
        # Right camera mask data
        self.right_ellipse_center = None
        self.right_ellipse_axes = None
        self.right_ellipse_angle = 0
        self.right_polygon_points = []
        self.right_drawing_polygon = False
        self.right_mask = None
        self.right_mask_overlay = None
        self.right_dragging_ellipse = False
        self.right_resizing_ellipse = False
        self.right_ellipse_initial_axes = None
        
        # Separate masks for right camera
        self.right_ellipse_mask = None
        self.right_polygon_mask = None
        self.right_ellipse_mask_overlay = None
        self.right_polygon_mask_overlay = None
        
        # Data storage
        self.left_images: List[np.ndarray] = []
        self.right_images: List[np.ndarray] = []
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False  # Start in paused mode for frame-by-frame viewing
        self.frame_delay = 1  # Minimal delay for frame-by-frame mode
        
        # OpenCV window names
        self.left_window = "Left Camera"
        self.right_window = "Right Camera"
        
        logger.info(f"Initializing trajectory visualizer")
        logger.info(f"Trajectory path: {self.trajectory_path}")
        logger.info(f"Episode index: {self.episode_idx}")
        logger.info(f"Left camera key: {self.camera_left}")
        logger.info(f"Right camera key: {self.camera_right}")
        logger.info(f"Playback speed: {self.playback_speed}x")
    
    def load_trajectory_data(self) -> bool:
        """
        Load trajectory data from HDF5 file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.trajectory_path.exists():
                logger.error(f"Trajectory file not found: {self.trajectory_path}")
                return False
            
            logger.info(f"Loading trajectory data from {self.trajectory_path}")
            
            # Load HDF5 data
            with h5py.File(self.trajectory_path, 'r') as h5_file:
                # List available data keys
                data_keys = list(h5_file.keys())
                logger.info(f"Available data keys: {data_keys}")
                
                # Check if camera data exists
                if self.camera_left not in h5_file:
                    logger.error(f"Left camera data '{self.camera_left}' not found. Available keys: {data_keys}")
                    return False
                
                if self.camera_right not in h5_file:
                    logger.error(f"Right camera data '{self.camera_right}' not found. Available keys: {data_keys}")
                    return False
                
                # Load image data
                logger.info("Loading left camera images...")
                left_camera_data = h5_file[self.camera_left]
                self.left_images = self._load_image_sequence_hdf5(left_camera_data)
                
                logger.info("Loading right camera images...")
                right_camera_data = h5_file[self.camera_right]
                self.right_images = self._load_image_sequence_hdf5(right_camera_data)
                
                if not self.left_images or not self.right_images:
                    logger.error("Failed to load camera images")
                    return False
                
                # Ensure both sequences have the same length
                min_length = min(len(self.left_images), len(self.right_images))
                self.left_images = self.left_images[:min_length]
                self.right_images = self.right_images[:min_length]
                
                self.total_frames = min_length
                
                logger.info(f"Loaded {self.total_frames} frames for both cameras")
                logger.info(f"Left camera image shape: {self.left_images[0].shape}")
                logger.info(f"Right camera image shape: {self.right_images[0].shape}")
                
                return True
            
        except Exception as e:
            logger.error(f"Error loading trajectory data: {e}")
            return False
    
    def _load_image_sequence_hdf5(self, image_data) -> List[np.ndarray]:
        """
        Load image sequence from HDF5 data.
        
        Args:
            image_data: HDF5 dataset containing image data
            
        Returns:
            List of numpy arrays representing images
        """
        images = []
        
        try:
            # Get the shape of the image data
            shape = image_data.shape
            logger.info(f"Image data shape: {shape}")
            
            # Load all images
            for i in tqdm(range(shape[0]), desc="Loading images"):
                img = image_data[i]
                
                # Ensure image is in correct format
                if len(img.shape) == 3:
                    # Convert to uint8 if necessary
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    images.append(img)
                else:
                    logger.warning(f"Unexpected image shape at frame {i}: {img.shape}")
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"Error loading image sequence: {e}")
            return []
    
    def setup_windows(self):
        """Setup OpenCV windows for display."""
        # Create windows with fixed size
        cv2.namedWindow(self.left_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.right_window, cv2.WINDOW_NORMAL)
        
        # Set window sizes (width, height)
        cv2.resizeWindow(self.left_window, self.window_size[0], self.window_size[1])
        cv2.resizeWindow(self.right_window, self.window_size[0], self.window_size[1])
        
        # Position windows side by side
        cv2.moveWindow(self.left_window, 100, 100)
        cv2.moveWindow(self.right_window, 800, 100)
        
        # Set up mouse callbacks for mask drawing
        if self.enable_mask_drawing:
            cv2.setMouseCallback(self.left_window, self._mouse_callback_left)
            cv2.setMouseCallback(self.right_window, self._mouse_callback_right)
        
        logger.info(f"OpenCV windows created with size {self.window_size[0]}x{self.window_size[1]}")
        if self.enable_mask_drawing:
            logger.info("Mask drawing enabled - use mouse to draw masks")
    
    def display_frame(self, frame_idx: int):
        """
        Display a single frame from both cameras.
        
        Args:
            frame_idx: Index of the frame to display
        """
        if frame_idx >= self.total_frames:
            return
        
        # Get images
        left_img = self.left_images[frame_idx]
        right_img = self.right_images[frame_idx]
        
        # Resize images to fit window size
        left_img_resized = cv2.resize(left_img, self.window_size)
        right_img_resized = cv2.resize(right_img, self.window_size)
        
        # Add frame information overlay
        left_img_with_info = self._add_frame_info(left_img_resized, frame_idx, "Left")
        right_img_with_info = self._add_frame_info(right_img_resized, frame_idx, "Right")
        
        # Add mask overlay if drawing
        if self.enable_mask_drawing:
            if self.left_mask is not None:
                left_img_with_info = self._apply_mask_overlay(left_img_with_info, "left")
            if self.right_mask is not None:
                right_img_with_info = self._apply_mask_overlay(right_img_with_info, "right")
        
        # Display images
        cv2.imshow(self.left_window, left_img_with_info)
        cv2.imshow(self.right_window, right_img_with_info)
    
    def _add_frame_info(self, img: np.ndarray, frame_idx: int, camera_name: str) -> np.ndarray:
        """
        Add frame information overlay to image.
        
        Args:
            img: Input image
            frame_idx: Current frame index
            camera_name: Name of the camera
            
        Returns:
            Image with overlay information
        """
        img_copy = img.copy()
        
        # Add text overlay
        progress = (frame_idx / max(1, self.total_frames - 1)) * 100
        text_lines = [
            f"{camera_name} Camera",
            f"Frame: {frame_idx}/{self.total_frames-1} ({progress:.1f}%)",
            f"Time: {frame_idx/30:.2f}s" if self.total_frames > 0 else "Time: 0.00s",
            f"Status: {'Playing' if self.is_playing else 'Paused'}"
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(img_copy, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25
        
        return img_copy
    
    def handle_keyboard_input(self) -> bool:
        """
        Handle keyboard input for frame-by-frame control.
        
        Returns:
            True to continue, False to exit
        """
        key = cv2.waitKey(self.frame_delay) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            logger.info("Exit requested by user")
            return False
        elif key == ord(' '):  # Space bar - pause/play
            self.is_playing = not self.is_playing
            logger.info(f"Playback {'paused' if not self.is_playing else 'resumed'}")
        elif key == ord('r'):  # 'r' - restart
            self.current_frame = 0
            logger.info("Restarted to frame 0")
        elif key == ord('f'):  # 'f' - fast forward
            self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
            logger.info(f"Fast forward to frame {self.current_frame}")
        elif key == ord('b'):  # 'b' - rewind
            self.current_frame = max(self.current_frame - 10, 0)
            logger.info(f"Rewind to frame {self.current_frame}")
        elif key == ord('s') or key == ord('d'):  # 's' or 'd' - step forward
            self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            logger.info(f"Step forward to frame {self.current_frame}")
        elif key == ord('a') or key == ord('w'):  # 'a' or 'w' - step backward
            self.current_frame = max(self.current_frame - 1, 0)
            logger.info(f"Step backward to frame {self.current_frame}")
        elif key == ord('h'):  # 'h' - help
            self._print_help()
        elif key == ord('j'):  # 'j' - jump to specific frame
            self._jump_to_frame()
        elif key == ord('g'):  # 'g' - go to beginning
            self.current_frame = 0
            logger.info("Jumped to beginning")
        elif key == ord('e'):  # 'e' - go to end
            self.current_frame = self.total_frames - 1
            logger.info(f"Jumped to end (frame {self.current_frame})")
        elif key == ord('m'):  # 'm' - toggle mask drawing mode
            if self.enable_mask_drawing:
                self.mask_drawing_mode = "polygon" if self.mask_drawing_mode == "ellipse" else "ellipse"
                logger.info(f"Mask drawing mode: {self.mask_drawing_mode}")
        elif key == ord('c'):  # 'c' - clear current camera mask
            if self.enable_mask_drawing:
                if self.current_camera == "left":
                    self.left_ellipse_center = None
                    self.left_ellipse_axes = None
                    self.left_polygon_points = []
                    self.left_drawing_polygon = False
                    self.left_dragging_ellipse = False
                    self.left_resizing_ellipse = False
                    self.left_ellipse_initial_axes = None
                    self.left_mask = None
                    self.left_mask_overlay = None
                    self.left_ellipse_mask = None
                    self.left_polygon_mask = None
                    self.left_ellipse_mask_overlay = None
                    self.left_polygon_mask_overlay = None
                    self.ellipse_created["left"] = False
                    logger.info("Left camera mask cleared")
                else:
                    self.right_ellipse_center = None
                    self.right_ellipse_axes = None
                    self.right_polygon_points = []
                    self.right_drawing_polygon = False
                    self.right_dragging_ellipse = False
                    self.right_resizing_ellipse = False
                    self.right_ellipse_initial_axes = None
                    self.right_mask = None
                    self.right_mask_overlay = None
                    self.right_ellipse_mask = None
                    self.right_polygon_mask = None
                    self.right_ellipse_mask_overlay = None
                    self.right_polygon_mask_overlay = None
                    self.ellipse_created["right"] = False
                    logger.info("Right camera mask cleared")
        elif key == ord('x'):  # 'x' - save mask
            if self.enable_mask_drawing:
                self._save_mask()
        elif key == ord('z'):  # 'z' - save mask to specified directory
            if self.enable_mask_drawing:
                self._save_mask_to_directory()
        elif key == ord('t'):  # 't' - toggle current camera
            if self.enable_mask_drawing:
                self.current_camera = "right" if self.current_camera == "left" else "left"
                logger.info(f"Current camera: {self.current_camera}")
        elif key == ord('p'):  # 'p' - adjust ellipse parameters
            if self.enable_mask_drawing:
                self._adjust_ellipse_parameters()
        
        return True
    
    def _adjust_ellipse_parameters(self):
        """Adjust ellipse parameters interactively."""
        if self.current_camera == "left":
            if not self.ellipse_created["left"]:
                logger.warning("No left ellipse to adjust. Create one first.")
                return
            
            center = self.left_ellipse_center
            axes = self.left_ellipse_axes
            angle = self.left_ellipse_angle
        else:
            if not self.ellipse_created["right"]:
                logger.warning("No right ellipse to adjust. Create one first.")
                return
            
            center = self.right_ellipse_center
            axes = self.right_ellipse_axes
            angle = self.right_ellipse_angle
        
        if center is None or axes is None:
            logger.warning("No ellipse to adjust.")
            return
        
        print(f"\n=== Adjusting {self.current_camera} ellipse parameters ===")
        print(f"Current center: ({center[0]}, {center[1]})")
        print(f"Current axes: ({axes[0]}, {axes[1]})")
        print(f"Current angle: {angle}")
        print("\nEnter new values (press Enter to keep current value):")
        
        try:
            # Get new center
            new_x = input(f"Center X [{center[0]}]: ").strip()
            if new_x:
                center = (int(new_x), center[1])
            
            new_y = input(f"Center Y [{center[1]}]: ").strip()
            if new_y:
                center = (center[0], int(new_y))
            
            # Get new axes
            new_a = input(f"Axis A (width) [{axes[0]}]: ").strip()
            if new_a:
                axes = (int(new_a), axes[1])
            
            new_b = input(f"Axis B (height) [{axes[1]}]: ").strip()
            if new_b:
                axes = (axes[0], int(new_b))
            
            # Get new angle
            new_angle = input(f"Angle [{angle}]: ").strip()
            if new_angle:
                angle = float(new_angle)
            
            # Update ellipse
            if self.current_camera == "left":
                self.left_ellipse_center = center
                self.left_ellipse_axes = axes
                self.left_ellipse_angle = angle
            else:
                self.right_ellipse_center = center
                self.right_ellipse_axes = axes
                self.right_ellipse_angle = angle
            
            # Update mask
            self._update_mask(self.current_camera)
            
            print(f"Updated {self.current_camera} ellipse:")
            print(f"  Center: {center}")
            print(f"  Axes: {axes}")
            print(f"  Angle: {angle}")
            
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
        except KeyboardInterrupt:
            logger.info("Parameter adjustment cancelled")
    
    def _mouse_callback_left(self, event, x, y, flags, param):
        """Mouse callback for left camera window."""
        if self.enable_mask_drawing:
            self.current_camera = "left"
            self._mouse_callback(event, x, y, flags, param, "left")
    
    def _mouse_callback_right(self, event, x, y, flags, param):
        """Mouse callback for right camera window."""
        if self.enable_mask_drawing:
            self.current_camera = "right"
            self._mouse_callback(event, x, y, flags, param, "right")
    
    def _mouse_callback(self, event, x, y, flags, param, camera_side):
        """Handle mouse events for mask drawing."""
        if not self.enable_mask_drawing:
            return
        
        if self.mask_drawing_mode == "ellipse":
            self._handle_ellipse_drawing(event, x, y, flags, camera_side)
        elif self.mask_drawing_mode == "polygon":
            self._handle_polygon_drawing(event, x, y, flags, camera_side)
    
    def _handle_ellipse_drawing(self, event, x, y, flags, camera_side):
        """Handle ellipse drawing, dragging, and resizing."""
        if camera_side == "left":
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.ellipse_created["left"]:
                    # First time - create ellipse
                    self.left_ellipse_center = (x, y)
                    self.left_ellipse_axes = (50, 30)  # Default size
                    self.ellipse_created["left"] = True
                    self.left_dragging_ellipse = False
                    self.left_resizing_ellipse = False
                    logger.info("Created left ellipse (first time)")
                elif self.left_ellipse_center is not None:
                    # Ellipse already exists - check interaction type
                    if self._is_point_in_ellipse(x, y, self.left_ellipse_center, self.left_ellipse_axes):
                        # Check if clicking near the edge for resizing
                        if self._is_point_near_ellipse_edge(x, y, self.left_ellipse_center, self.left_ellipse_axes):
                            self.left_resizing_ellipse = True
                            self.left_ellipse_initial_axes = self.left_ellipse_axes
                            logger.info("Started resizing left ellipse")
                        else:
                            # Clicking inside ellipse - drag it
                            self.left_dragging_ellipse = True
                            logger.info("Started dragging left ellipse")
                    else:
                        # Clicking outside ellipse - ignore (don't create new one)
                        logger.info("Clicking outside ellipse - ignored (ellipse already exists)")
                self._update_mask("left")
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.left_dragging_ellipse and self.left_ellipse_center is not None:
                    # Move ellipse center while dragging
                    self.left_ellipse_center = (x, y)
                    self._update_mask("left")
                elif self.left_resizing_ellipse and self.left_ellipse_center is not None and self.left_ellipse_initial_axes is not None:
                    # Resize ellipse based on distance from center
                    dx = abs(x - self.left_ellipse_center[0])
                    dy = abs(y - self.left_ellipse_center[1])
                    self.left_ellipse_axes = (max(10, dx), max(10, dy))
                    self._update_mask("left")
                    
            elif event == cv2.EVENT_LBUTTONUP:
                if self.left_dragging_ellipse:
                    logger.info("Finished dragging left ellipse")
                    self.left_dragging_ellipse = False
                elif self.left_resizing_ellipse:
                    logger.info("Finished resizing left ellipse")
                    self.left_resizing_ellipse = False
                    self.left_ellipse_initial_axes = None
                    
        else:  # right camera
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.ellipse_created["right"]:
                    # First time - create ellipse
                    self.right_ellipse_center = (x, y)
                    self.right_ellipse_axes = (50, 30)  # Default size
                    self.ellipse_created["right"] = True
                    self.right_dragging_ellipse = False
                    self.right_resizing_ellipse = False
                    logger.info("Created right ellipse (first time)")
                elif self.right_ellipse_center is not None:
                    # Ellipse already exists - check interaction type
                    if self._is_point_in_ellipse(x, y, self.right_ellipse_center, self.right_ellipse_axes):
                        # Check if clicking near the edge for resizing
                        if self._is_point_near_ellipse_edge(x, y, self.right_ellipse_center, self.right_ellipse_axes):
                            self.right_resizing_ellipse = True
                            self.right_ellipse_initial_axes = self.right_ellipse_axes
                            logger.info("Started resizing right ellipse")
                        else:
                            # Clicking inside ellipse - drag it
                            self.right_dragging_ellipse = True
                            logger.info("Started dragging right ellipse")
                    else:
                        # Clicking outside ellipse - ignore (don't create new one)
                        logger.info("Clicking outside ellipse - ignored (ellipse already exists)")
                self._update_mask("right")
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.right_dragging_ellipse and self.right_ellipse_center is not None:
                    # Move ellipse center while dragging
                    self.right_ellipse_center = (x, y)
                    self._update_mask("right")
                elif self.right_resizing_ellipse and self.right_ellipse_center is not None and self.right_ellipse_initial_axes is not None:
                    # Resize ellipse based on distance from center
                    dx = abs(x - self.right_ellipse_center[0])
                    dy = abs(y - self.right_ellipse_center[1])
                    self.right_ellipse_axes = (max(10, dx), max(10, dy))
                    self._update_mask("right")
                    
            elif event == cv2.EVENT_LBUTTONUP:
                if self.right_dragging_ellipse:
                    logger.info("Finished dragging right ellipse")
                    self.right_dragging_ellipse = False
                elif self.right_resizing_ellipse:
                    logger.info("Finished resizing right ellipse")
                    self.right_resizing_ellipse = False
                    self.right_ellipse_initial_axes = None
    
    def _handle_polygon_drawing(self, event, x, y, flags, camera_side):
        """Handle polygon drawing."""
        if camera_side == "left":
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.left_drawing_polygon:
                    self.left_polygon_points = []
                    self.left_drawing_polygon = True
                self.left_polygon_points.append((x, y))
                self._update_mask("left")
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click to finish polygon
                if self.left_drawing_polygon and len(self.left_polygon_points) >= 3:
                    # Auto-close polygon by connecting to first point
                    self.left_polygon_points.append(self.left_polygon_points[0])
                    self.left_drawing_polygon = False
                    self._update_mask("left")
            elif event == cv2.EVENT_MBUTTONDOWN:  # Middle click to close polygon
                if self.left_drawing_polygon and len(self.left_polygon_points) >= 3:
                    # Close polygon by connecting to first point
                    self.left_polygon_points.append(self.left_polygon_points[0])
                    self.left_drawing_polygon = False
                    self._update_mask("left")
        else:  # right camera
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.right_drawing_polygon:
                    self.right_polygon_points = []
                    self.right_drawing_polygon = True
                self.right_polygon_points.append((x, y))
                self._update_mask("right")
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click to finish polygon
                if self.right_drawing_polygon and len(self.right_polygon_points) >= 3:
                    # Auto-close polygon by connecting to first point
                    self.right_polygon_points.append(self.right_polygon_points[0])
                    self.right_drawing_polygon = False
                    self._update_mask("right")
            elif event == cv2.EVENT_MBUTTONDOWN:  # Middle click to close polygon
                if self.right_drawing_polygon and len(self.right_polygon_points) >= 3:
                    # Close polygon by connecting to first point
                    self.right_polygon_points.append(self.right_polygon_points[0])
                    self.right_drawing_polygon = False
                    self._update_mask("right")
    
    def _update_mask(self, camera_side):
        """Update the mask for specified camera based on drawing state."""
        if not self.enable_mask_drawing:
            return
        
        if camera_side == "left":
            # Update ellipse mask
            self._update_ellipse_mask("left")
            # Update polygon mask
            self._update_polygon_mask("left")
            # Combine masks
            self._combine_masks("left")
        else:  # right camera
            # Update ellipse mask
            self._update_ellipse_mask("right")
            # Update polygon mask
            self._update_polygon_mask("right")
            # Combine masks
            self._combine_masks("right")
    
    def _update_ellipse_mask(self, camera_side):
        """Update ellipse mask for specified camera."""
        if camera_side == "left":
            if self.left_ellipse_center is not None and self.left_ellipse_axes is not None:
                # Create ellipse mask
                mask = np.zeros(self.window_size[::-1], dtype=np.uint8)
                cv2.ellipse(mask, self.left_ellipse_center, self.left_ellipse_axes, self.left_ellipse_angle, 0, 360, 1, -1)
                self.left_ellipse_mask = mask
                self.left_ellipse_mask_overlay = self._create_mask_overlay(mask)
            else:
                self.left_ellipse_mask = None
                self.left_ellipse_mask_overlay = None
        else:  # right camera
            if self.right_ellipse_center is not None and self.right_ellipse_axes is not None:
                # Create ellipse mask
                mask = np.zeros(self.window_size[::-1], dtype=np.uint8)
                cv2.ellipse(mask, self.right_ellipse_center, self.right_ellipse_axes, self.right_ellipse_angle, 0, 360, 1, -1)
                self.right_ellipse_mask = mask
                self.right_ellipse_mask_overlay = self._create_mask_overlay(mask)
            else:
                self.right_ellipse_mask = None
                self.right_ellipse_mask_overlay = None
    
    def _update_polygon_mask(self, camera_side):
        """Update polygon mask for specified camera."""
        if camera_side == "left":
            if len(self.left_polygon_points) >= 3:
                # Create polygon mask
                mask = np.zeros(self.window_size[::-1], dtype=np.uint8)
                polygon_array = np.array(self.left_polygon_points, dtype=np.int32)
                # Fill polygon area with 1 (mask out)
                cv2.fillPoly(mask, [polygon_array], 1)
                self.left_polygon_mask = mask
                self.left_polygon_mask_overlay = self._create_mask_overlay(mask)
            else:
                self.left_polygon_mask = None
                self.left_polygon_mask_overlay = None
        else:  # right camera
            if len(self.right_polygon_points) >= 3:
                # Create polygon mask
                mask = np.zeros(self.window_size[::-1], dtype=np.uint8)
                polygon_array = np.array(self.right_polygon_points, dtype=np.int32)
                # Fill polygon area with 1 (mask out)
                cv2.fillPoly(mask, [polygon_array], 1)
                self.right_polygon_mask = mask
                self.right_polygon_mask_overlay = self._create_mask_overlay(mask)
            else:
                self.right_polygon_mask = None
                self.right_polygon_mask_overlay = None
    
    def _combine_masks(self, camera_side):
        """Combine ellipse and polygon masks."""
        if camera_side == "left":
            # Start with ellipse mask (outer boundary)
            if self.left_ellipse_mask is not None:
                combined_mask = self.left_ellipse_mask.copy()
            else:
                combined_mask = np.zeros(self.window_size[::-1], dtype=np.uint8)
            
            # Subtract polygon mask (inner holes)
            if self.left_polygon_mask is not None:
                combined_mask = combined_mask & (~self.left_polygon_mask)
            
            self.left_mask = combined_mask
            self.left_mask_overlay = self._create_mask_overlay(combined_mask)
        else:  # right camera
            # Start with ellipse mask (outer boundary)
            if self.right_ellipse_mask is not None:
                combined_mask = self.right_ellipse_mask.copy()
            else:
                combined_mask = np.zeros(self.window_size[::-1], dtype=np.uint8)
            
            # Subtract polygon mask (inner holes)
            if self.right_polygon_mask is not None:
                combined_mask = combined_mask & (~self.right_polygon_mask)
            
            self.right_mask = combined_mask
            self.right_mask_overlay = self._create_mask_overlay(combined_mask)
    
    def _create_mask_overlay(self, mask):
        """Create a visual overlay for the mask with different colors for different mask types."""
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[mask == 1] = [0, 255, 0]  # Green for ellipse areas (visible)
        overlay[mask == 2] = [0, 255, 255]  # Yellow for polygon outlines
        overlay[mask == 0] = [255, 0, 0]  # Red for masked areas
        return overlay
    
    def _apply_mask_overlay(self, img, camera_side):
        """Apply mask overlay to image."""
        if camera_side == "left" and self.left_mask_overlay is None:
            return img
        elif camera_side == "right" and self.right_mask_overlay is None:
            return img
        
        # Get the appropriate overlay
        overlay = self.left_mask_overlay if camera_side == "left" else self.right_mask_overlay
        
        # Resize overlay to match image size
        overlay_resized = cv2.resize(overlay, (img.shape[1], img.shape[0]))
        
        # Blend overlay with image
        alpha = 0.3  # Transparency
        result = cv2.addWeighted(img, 1 - alpha, overlay_resized, alpha, 0)
        
        # Draw current drawing state
        if camera_side == "left":
            if self.mask_drawing_mode == "ellipse" and self.left_ellipse_center is not None:
                cv2.ellipse(result, self.left_ellipse_center, self.left_ellipse_axes, self.left_ellipse_angle, 0, 360, (0, 255, 255), 2)
            
            if self.mask_drawing_mode == "polygon" and len(self.left_polygon_points) > 0:
                for i, point in enumerate(self.left_polygon_points):
                    cv2.circle(result, point, 3, (0, 255, 255), -1)
                    if i > 0:
                        cv2.line(result, self.left_polygon_points[i-1], point, (0, 255, 255), 2)
        else:  # right camera
            if self.mask_drawing_mode == "ellipse" and self.right_ellipse_center is not None:
                cv2.ellipse(result, self.right_ellipse_center, self.right_ellipse_axes, self.right_ellipse_angle, 0, 360, (0, 255, 255), 2)
            
            if self.mask_drawing_mode == "polygon" and len(self.right_polygon_points) > 0:
                for i, point in enumerate(self.right_polygon_points):
                    cv2.circle(result, point, 3, (0, 255, 255), -1)
                    if i > 0:
                        cv2.line(result, self.right_polygon_points[i-1], point, (0, 255, 255), 2)
        
        return result
    
    def _is_point_in_ellipse(self, x, y, center, axes):
        """Check if a point is inside an ellipse."""
        if center is None or axes is None:
            return False
        
        # Calculate normalized coordinates
        dx = (x - center[0]) / axes[0]
        dy = (y - center[1]) / axes[1]
        
        # Check if point is inside ellipse (ellipse equation: (x/a)² + (y/b)² <= 1)
        return (dx * dx + dy * dy) <= 1.0
    
    def _is_point_near_ellipse_edge(self, x, y, center, axes, threshold=0.1):
        """Check if a point is near the edge of an ellipse for resizing."""
        if center is None or axes is None:
            return False
        
        # Calculate normalized coordinates
        dx = (x - center[0]) / axes[0]
        dy = (y - center[1]) / axes[1]
        
        # Calculate distance from ellipse edge
        distance_from_edge = abs((dx * dx + dy * dy) - 1.0)
        
        # Check if point is near the edge (within threshold)
        return distance_from_edge <= threshold
    
    def _save_mask(self, output_dir: Optional[Path] = None):
        """Save masks for both cameras to PNG files."""
        if (self.left_mask is None and self.right_mask is None and 
            self.left_ellipse_mask is None and self.right_ellipse_mask is None and
            self.left_polygon_mask is None and self.right_polygon_mask is None):
            logger.warning("No masks to save")
            return
        
        # Use provided output directory or default
        if output_dir is None:
            output_dir = Path("masks")
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save left camera masks
        if self.left_mask is not None:
            # Combined mask
            left_filepath = output_dir / "left_mask.png"
            cv2.imwrite(str(left_filepath), self.left_mask * 255)
            logger.info(f"Left combined mask saved to {left_filepath}")
        
        if self.left_ellipse_mask is not None:
            # Ellipse mask
            left_ellipse_filepath = output_dir / "ellipse_left_mask.png"
            cv2.imwrite(str(left_ellipse_filepath), self.left_ellipse_mask * 255)
            logger.info(f"Left ellipse mask saved to {left_ellipse_filepath}")
        
        if self.left_polygon_mask is not None:
            # Polygon mask
            left_polygon_filepath = output_dir / "polygon_left_mask.png"
            cv2.imwrite(str(left_polygon_filepath), self.left_polygon_mask * 255)
            logger.info(f"Left polygon mask saved to {left_polygon_filepath}")
        
        # Save right camera masks
        if self.right_mask is not None:
            # Combined mask
            right_filepath = output_dir / "right_mask.png"
            cv2.imwrite(str(right_filepath), self.right_mask * 255)
            logger.info(f"Right combined mask saved to {right_filepath}")
        
        if self.right_ellipse_mask is not None:
            # Ellipse mask
            right_ellipse_filepath = output_dir / "ellipse_right_mask.png"
            cv2.imwrite(str(right_ellipse_filepath), self.right_ellipse_mask * 255)
            logger.info(f"Right ellipse mask saved to {right_ellipse_filepath}")
        
        if self.right_polygon_mask is not None:
            # Polygon mask
            right_polygon_filepath = output_dir / "polygon_right_mask.png"
            cv2.imwrite(str(right_polygon_filepath), self.right_polygon_mask * 255)
            logger.info(f"Right polygon mask saved to {right_polygon_filepath}")
    
    def _save_mask_to_directory(self):
        """Save masks to a user-specified directory."""
        try:
            output_dir = input("Enter output directory path (press Enter for default 'masks'): ").strip()
            if not output_dir:
                output_dir = "masks"
            
            output_path = Path(output_dir)
            self._save_mask(output_path)
            logger.info(f"Masks saved to {output_path}")
            
        except KeyboardInterrupt:
            logger.info("Save cancelled")
        except Exception as e:
            logger.error(f"Error saving masks: {e}")
    
    def _jump_to_frame(self):
        """Jump to a specific frame number."""
        try:
            frame_input = input(f"Enter frame number (0-{self.total_frames-1}): ")
            frame_num = int(frame_input)
            if 0 <= frame_num < self.total_frames:
                self.current_frame = frame_num
                logger.info(f"Jumped to frame {self.current_frame}")
            else:
                logger.warning(f"Frame number {frame_num} out of range (0-{self.total_frames-1})")
        except ValueError:
            logger.warning("Invalid frame number")
        except KeyboardInterrupt:
            logger.info("Jump cancelled")
    
    def _print_help(self):
        """Print keyboard controls help."""
        help_text = """
Frame-by-Frame Controls:
  q, ESC    - Quit
  SPACE     - Pause/Resume playback
  r         - Restart to frame 0
  f         - Fast forward (10 frames)
  b         - Rewind (10 frames)
  
Frame Navigation:
  s, d      - Step forward (1 frame)
  a, w      - Step backward (1 frame)
  g         - Go to beginning (frame 0)
  e         - Go to end (last frame)
  j         - Jump to specific frame number

Mask Drawing Controls:
  m         - Toggle mask drawing mode (ellipse/polygon)
  t         - Toggle current camera (left/right)
  c         - Clear current camera mask
  p         - Adjust ellipse parameters (center, axes, angle)
  x         - Save all masks to PNG files (default directory)
  z         - Save all masks to specified directory
  
Mouse Controls:
  Ellipse Mode:
    - First click: Create ellipse (only once per camera)
    - Click inside existing ellipse: Drag the ellipse to new position
    - Click near edge of existing ellipse: Resize the ellipse
    - Drag while resizing: Adjust ellipse size
  Polygon Mode:
    - Left click to add points
    - Right click to finish polygon (auto-close to first point)
    - Middle click to close polygon (auto-close to first point)
    - Polygon areas are masked out (holes in the ellipse)
  
  h         - Show this help
        """
        print(help_text)
    
    def run(self):
        """Run the trajectory visualizer."""
        # Load data
        if not self.load_trajectory_data():
            logger.error("Failed to load trajectory data")
            return False
        
        # Setup windows
        self.setup_windows()
        
        # Print help
        self._print_help()
        
        logger.info("Starting trajectory visualization")
        logger.info("Press 'h' for help, 'q' to quit")
        
        try:
            while True:
                # Display current frame
                self.display_frame(self.current_frame)
                
                # Handle keyboard input
                if not self.handle_keyboard_input():
                    break
                
                # Advance frame if playing
                if self.is_playing:
                    self.current_frame = (self.current_frame + 1) % self.total_frames
                
                # Small delay for smooth playback
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            logger.info("Visualization completed")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize trajectory data from HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_mask.py --trajectory_path saved_data/episode_0000.hdf5
  python plot_mask.py --trajectory_path saved_data/episode_0000.hdf5 --window_width 800 --window_height 600
  python plot_mask.py --trajectory_path saved_data/episode_0000.hdf5 --playback_speed 0.5
  python plot_mask.py --trajectory_path saved_data/episode_0000.hdf5 --camera_left OpenCVCameraDevice_1 --camera_right OpenCVCameraDevice_6
        """
    )
    
    parser.add_argument(
        "--trajectory_path",
        type=Path,
        required=True,
        help="Path to the HDF5 file"
    )
    
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Index of the episode to visualize (default: 0)"
    )
    
    parser.add_argument(
        "--camera_left",
        type=str,
        default="OpenCVCameraDevice_1",
        help="Key for left camera data (default: OpenCVCameraDevice_1)"
    )
    
    parser.add_argument(
        "--camera_right",
        type=str,
        default="OpenCVCameraDevice_6",
        help="Key for right camera data (default: OpenCVCameraDevice_6)"
    )
    
    parser.add_argument(
        "--playback_speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )
    
    parser.add_argument(
        "--window_width",
        type=int,
        default=640,
        help="Window width in pixels (default: 640)"
    )
    
    parser.add_argument(
        "--window_height",
        type=int,
        default=480,
        help="Window height in pixels (default: 480)"
    )
    
    parser.add_argument(
        "--no-mask-drawing",
        action="store_true",
        help="Disable mask drawing functionality"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.trajectory_path.exists():
        logger.error(f"Trajectory file not found: {args.trajectory_path}")
        sys.exit(1)
    
    if args.playback_speed <= 0:
        logger.error("Playback speed must be positive")
        sys.exit(1)
    
    # Create and run visualizer
    visualizer = TrajectoryVisualizer(
        trajectory_path=args.trajectory_path,
        episode_idx=args.episode_idx,
        camera_left=args.camera_left,
        camera_right=args.camera_right,
        playback_speed=args.playback_speed,
        window_size=(args.window_width, args.window_height),
        enable_mask_drawing=not args.no_mask_drawing
    )
    
    success = visualizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
