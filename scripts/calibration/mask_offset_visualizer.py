#!/usr/bin/env python3
"""
Mask Offset Visualizer

This script visualizes UMI and robot masked images with offset compensation.
It displays three windows:
1. UMI image masked with UMI mask
2. Robot image masked with robot mask  
3. Overlay of both images after offset compensation

Features:
- Load video data from HDF5 files
- Load mask files for UMI and robot
- Load offset data from analysis results
- Interactive frame navigation
- Real-time visualization with offset compensation

Usage:
    python scripts/calibration/mask_offset_visualizer.py   --umi_video saved_data/episode_0000.hdf5   --robot_video robot_data/episode_0001.hdf5   --umi_mask masks/umi/mapped/left_mask.png   --robot_mask masks/robot/mapped/left_mask.png   --offset_file mask_offset/left/mask_offset_analysis_20250929_151954.json 


Author: Assistant
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import h5py
import json
from tqdm import tqdm
import time

# Get the project root directory (parent of scripts directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from utils.logger_config import logger


class MaskOffsetVisualizer:
    """Visualizer for UMI and robot masked images with offset compensation."""
    
    def __init__(self, umi_video_path: Path, robot_video_path: Path, 
                 umi_mask_path: Path, robot_mask_path: Path, 
                 offset_file_path: Path, umi_camera_key: str = "OpenCVCameraDevice_1",
                 robot_camera_key: str = "OpenCVCameraDevice_1",
                 umi_start_frame: int = 0, robot_start_frame: int = 0):
        """
        Initialize the mask offset visualizer.
        
        Args:
            umi_video_path: Path to UMI video HDF5 file
            robot_video_path: Path to robot video HDF5 file
            umi_mask_path: Path to UMI mask file
            robot_mask_path: Path to robot mask file
            offset_file_path: Path to offset analysis JSON file
            umi_camera_key: Camera key for UMI video
            robot_camera_key: Camera key for robot video
            umi_start_frame: Starting frame index for UMI video
            robot_start_frame: Starting frame index for robot video
        """
        self.umi_video_path = Path(umi_video_path)
        self.robot_video_path = Path(robot_video_path)
        self.umi_mask_path = Path(umi_mask_path)
        self.robot_mask_path = Path(robot_mask_path)
        self.offset_file_path = Path(offset_file_path)
        self.umi_camera_key = umi_camera_key
        self.robot_camera_key = robot_camera_key
        self.umi_start_frame = umi_start_frame
        self.robot_start_frame = robot_start_frame
        
        # Data containers
        self.umi_video_data = None
        self.robot_video_data = None
        self.umi_mask = None
        self.robot_mask = None
        self.offset_data = None
        
        # Video properties
        self.num_frames = 0
        self.current_frame = 0
        self.video_height = 0
        self.video_width = 0
        
        # Playback control
        self.is_playing = False
        self.playback_direction = 1  # 1 for forward, -1 for backward
        
        # Offset compensation
        self.offset_x = 0
        self.offset_y = 0
        
        logger.info(f"Initializing mask offset visualizer")
        logger.info(f"UMI video: {self.umi_video_path}")
        logger.info(f"Robot video: {self.robot_video_path}")
        logger.info(f"UMI mask: {self.umi_mask_path}")
        logger.info(f"Robot mask: {self.robot_mask_path}")
        logger.info(f"Offset file: {self.offset_file_path}")
    
    def load_video_data(self) -> bool:
        """
        Load video data from HDF5 files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load UMI video
            if not self.umi_video_path.exists():
                logger.error(f"UMI video file not found: {self.umi_video_path}")
                return False
            
            logger.info(f"Loading UMI video from {self.umi_video_path}")
            with h5py.File(self.umi_video_path, 'r') as h5_file:
                if self.umi_camera_key not in h5_file:
                    available_keys = list(h5_file.keys())
                    logger.error(f"UMI camera data '{self.umi_camera_key}' not found. Available keys: {available_keys}")
                    return False
                
                self.umi_video_data = h5_file[self.umi_camera_key][:]
                logger.info(f"UMI video loaded: {self.umi_video_data.shape}")
            
            # Load robot video
            if not self.robot_video_path.exists():
                logger.error(f"Robot video file not found: {self.robot_video_path}")
                return False
            
            logger.info(f"Loading robot video from {self.robot_video_path}")
            with h5py.File(self.robot_video_path, 'r') as h5_file:
                if self.robot_camera_key not in h5_file:
                    available_keys = list(h5_file.keys())
                    logger.error(f"Robot camera data '{self.robot_camera_key}' not found. Available keys: {available_keys}")
                    return False
                
                self.robot_video_data = h5_file[self.robot_camera_key][:]
                logger.info(f"Robot video loaded: {self.robot_video_data.shape}")
            
            # Handle different video lengths and starting frames
            umi_frames = self.umi_video_data.shape[0]
            robot_frames = self.robot_video_data.shape[0]
            
            # Apply starting frame offsets
            if self.umi_start_frame > 0:
                if self.umi_start_frame >= umi_frames:
                    logger.error(f"UMI start frame {self.umi_start_frame} >= total frames {umi_frames}")
                    return False
                self.umi_video_data = self.umi_video_data[self.umi_start_frame:]
                umi_frames = self.umi_video_data.shape[0]
                logger.info(f"Applied UMI start frame offset: {self.umi_start_frame}, remaining frames: {umi_frames}")
            
            if self.robot_start_frame > 0:
                if self.robot_start_frame >= robot_frames:
                    logger.error(f"Robot start frame {self.robot_start_frame} >= total frames {robot_frames}")
                    return False
                self.robot_video_data = self.robot_video_data[self.robot_start_frame:]
                robot_frames = self.robot_video_data.shape[0]
                logger.info(f"Applied robot start frame offset: {self.robot_start_frame}, remaining frames: {robot_frames}")
            
            if umi_frames != robot_frames:
                logger.warning(f"Video frame count mismatch: UMI {umi_frames} frames vs Robot {robot_frames} frames")
                logger.info(f"Using minimum frame count: {min(umi_frames, robot_frames)}")
                
                # Truncate both videos to the minimum length
                min_frames = min(umi_frames, robot_frames)
                self.umi_video_data = self.umi_video_data[:min_frames]
                self.robot_video_data = self.robot_video_data[:min_frames]
            
            # Validate other dimensions
            if self.umi_video_data.shape[1:] != self.robot_video_data.shape[1:]:
                logger.error(f"Video dimension mismatch: UMI {self.umi_video_data.shape[1:]} vs Robot {self.robot_video_data.shape[1:]}")
                return False
            
            self.num_frames = self.umi_video_data.shape[0]
            self.video_height = self.umi_video_data.shape[1]
            self.video_width = self.umi_video_data.shape[2]
            
            logger.info(f"Video properties: {self.num_frames} frames, {self.video_width}x{self.video_height}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading video data: {e}")
            return False
    
    def load_masks(self) -> bool:
        """
        Load mask files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load UMI mask
            if not self.umi_mask_path.exists():
                logger.error(f"UMI mask file not found: {self.umi_mask_path}")
                return False
            
            logger.info(f"Loading UMI mask from {self.umi_mask_path}")
            self.umi_mask = cv2.imread(str(self.umi_mask_path), cv2.IMREAD_GRAYSCALE)
            if self.umi_mask is None:
                logger.error(f"Failed to load UMI mask from {self.umi_mask_path}")
                return False
            
            # Load robot mask
            if not self.robot_mask_path.exists():
                logger.error(f"Robot mask file not found: {self.robot_mask_path}")
                return False
            
            logger.info(f"Loading robot mask from {self.robot_mask_path}")
            self.robot_mask = cv2.imread(str(self.robot_mask_path), cv2.IMREAD_GRAYSCALE)
            if self.robot_mask is None:
                logger.error(f"Failed to load robot mask from {self.robot_mask_path}")
                return False
            
            # Normalize masks to 0-1 range
            if self.umi_mask.max() > 1:
                self.umi_mask = self.umi_mask / 255.0
            if self.robot_mask.max() > 1:
                self.robot_mask = self.robot_mask / 255.0
            
            # Ensure masks match video dimensions
            if self.umi_mask.shape != (self.video_height, self.video_width):
                logger.warning(f"UMI mask shape {self.umi_mask.shape} doesn't match video dimensions {self.video_height}x{self.video_width}")
                self.umi_mask = cv2.resize(self.umi_mask, (self.video_width, self.video_height), interpolation=cv2.INTER_NEAREST)
            
            if self.robot_mask.shape != (self.video_height, self.video_width):
                logger.warning(f"Robot mask shape {self.robot_mask.shape} doesn't match video dimensions {self.video_height}x{self.video_width}")
                self.robot_mask = cv2.resize(self.robot_mask, (self.video_width, self.video_height), interpolation=cv2.INTER_NEAREST)
            
            logger.info(f"UMI mask shape: {self.umi_mask.shape}")
            logger.info(f"Robot mask shape: {self.robot_mask.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading masks: {e}")
            return False
    
    def load_offset_data(self) -> bool:
        """
        Load offset data from JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.offset_file_path.exists():
                logger.error(f"Offset file not found: {self.offset_file_path}")
                return False
            
            logger.info(f"Loading offset data from {self.offset_file_path}")
            
            with open(self.offset_file_path, 'r') as f:
                self.offset_data = json.load(f)
            
            # Extract offset values
            if 'offset_results' in self.offset_data and 'centroid' in self.offset_data['offset_results']:
                centroid_data = self.offset_data['offset_results']['centroid']
                self.offset_x = centroid_data.get('offset_x', 0)
                self.offset_y = centroid_data.get('offset_y', 0)
                
                logger.info(f"Offset loaded: X={self.offset_x:.2f}, Y={self.offset_y:.2f}")
                return True
            else:
                logger.error("Invalid offset data format")
                return False
                
        except Exception as e:
            logger.error(f"Error loading offset data: {e}")
            return False
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to image.
        
        Args:
            image: Input image
            mask: Mask to apply
            
        Returns:
            Masked image
        """
        if len(image.shape) == 3:
            # Color image
            masked = image.copy()
            for c in range(image.shape[2]):
                masked[:, :, c] = image[:, :, c] * mask
        else:
            # Grayscale image
            masked = image * mask
        
        return masked.astype(np.uint8)
    
    def apply_offset_compensation(self, image: np.ndarray, offset_x: float, offset_y: float) -> np.ndarray:
        """
        Apply offset compensation to image.
        
        Args:
            image: Input image
            offset_x: X offset in pixels
            offset_y: Y offset in pixels
            
        Returns:
            Offset compensated image
        """
        if offset_x == 0 and offset_y == 0:
            return image
        
        # Create transformation matrix
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        
        # Apply transformation
        offset_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        return offset_image
    
    def create_overlay(self, umi_masked: np.ndarray, robot_masked: np.ndarray, 
                      offset_x: float, offset_y: float) -> np.ndarray:
        """
        Create overlay of UMI and robot images with offset compensation.
        
        Args:
            umi_masked: UMI masked image
            robot_masked: Robot masked image
            offset_x: X offset for compensation
            offset_y: Y offset for compensation
            
        Returns:
            Overlay image
        """
        # Apply offset compensation to robot image
        robot_offset = self.apply_offset_compensation(robot_masked, offset_x, offset_y)
        
        # Create overlay with different colors
        overlay = np.zeros_like(umi_masked)
        
        # UMI in green channel
        if len(umi_masked.shape) == 3:
            overlay[:, :, 1] = umi_masked[:, :, 1]  # Green channel
        else:
            overlay[:, :, 1] = umi_masked
        
        # Robot in red channel (after offset)
        if len(robot_offset.shape) == 3:
            overlay[:, :, 2] = robot_offset[:, :, 2]  # Red channel
        else:
            overlay[:, :, 2] = robot_offset
        
        # Blend overlapping areas
        overlap_mask = (overlay[:, :, 1] > 0) & (overlay[:, :, 2] > 0)
        if np.any(overlap_mask):
            # Yellow for overlapping areas
            overlay[overlap_mask, 0] = 0
            overlay[overlap_mask, 1] = 255
            overlay[overlap_mask, 2] = 255
        
        return overlay
    
    def get_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get processed frames for visualization.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of (umi_masked, robot_masked, overlay)
        """
        if frame_idx >= self.num_frames:
            frame_idx = self.num_frames - 1
        
        # Get original frames
        umi_frame = self.umi_video_data[frame_idx]
        robot_frame = self.robot_video_data[frame_idx]
        
        # Ensure frames are in correct format
        if umi_frame.dtype != np.uint8:
            if umi_frame.max() <= 1.0:
                umi_frame = (umi_frame * 255).astype(np.uint8)
            else:
                umi_frame = umi_frame.astype(np.uint8)
        
        if robot_frame.dtype != np.uint8:
            if robot_frame.max() <= 1.0:
                robot_frame = (robot_frame * 255).astype(np.uint8)
            else:
                robot_frame = robot_frame.astype(np.uint8)
        
        # Apply masks
        umi_masked = self.apply_mask(umi_frame, self.umi_mask)
        robot_masked = self.apply_mask(robot_frame, self.robot_mask)
        
        # Create overlay with offset compensation
        overlay = self.create_overlay(umi_masked, robot_masked, self.offset_x, self.offset_y)
        
        return umi_masked, robot_masked, overlay
    
    def add_frame_info(self, image: np.ndarray, frame_idx: int, title: str) -> np.ndarray:
        """
        Add frame information to image.
        
        Args:
            image: Input image
            frame_idx: Current frame index
            title: Window title
            
        Returns:
            Image with frame information
        """
        info_image = image.copy()
        
        # Add frame number
        frame_text = f"Frame: {frame_idx}/{self.num_frames-1}"
        cv2.putText(info_image, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add title
        cv2.putText(info_image, title, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add offset information for overlay window
        if "Overlay" in title:
            offset_text = f"Offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}"
            cv2.putText(info_image, offset_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return info_image
    
    def visualize(self) -> bool:
        """
        Start interactive visualization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting interactive visualization")
            logger.info("Controls:")
            logger.info("  - Arrow keys: Navigate frames")
            logger.info("  - 's': Start/stop forward playback")
            logger.info("  - 'a': Start/stop backward playback")
            logger.info("  - 'q': Quit")
            logger.info("  - 'r': Reset to frame 0")
            logger.info("  - 'p': Save current frame")
            logger.info("  - 'h': Show help")
            
            # Create windows
            cv2.namedWindow("UMI Masked", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Robot Masked", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Overlay with Offset", cv2.WINDOW_NORMAL)
            
            # Resize windows
            cv2.resizeWindow("UMI Masked", 600, 400)
            cv2.resizeWindow("Robot Masked", 600, 400)
            cv2.resizeWindow("Overlay with Offset", 600, 400)
            
            while True:
                # Get current frames
                umi_masked, robot_masked, overlay = self.get_frame(self.current_frame)
                
                # Add frame information
                umi_display = self.add_frame_info(umi_masked, self.current_frame, "UMI Masked")
                robot_display = self.add_frame_info(robot_masked, self.current_frame, "Robot Masked")
                overlay_display = self.add_frame_info(overlay, self.current_frame, "Overlay with Offset")
                
                # Add playback status to overlay
                if self.is_playing:
                    status_text = f"Playing {'Forward' if self.playback_direction > 0 else 'Backward'}"
                    cv2.putText(overlay_display, status_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Display images
                cv2.imshow("UMI Masked", umi_display)
                cv2.imshow("Robot Masked", robot_display)
                cv2.imshow("Overlay with Offset", overlay_display)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF  # Increased wait time for smoother playback
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.current_frame = 0
                    self.is_playing = False
                    logger.info("Reset to frame 0")
                elif key == ord('s'):
                    # Start/stop forward playback
                    if self.is_playing and self.playback_direction > 0:
                        self.is_playing = False
                        logger.info("Stopped forward playback")
                    else:
                        self.is_playing = True
                        self.playback_direction = 1
                        logger.info("Started forward playback")
                elif key == ord('a'):
                    # Start/stop backward playback
                    if self.is_playing and self.playback_direction < 0:
                        self.is_playing = False
                        logger.info("Stopped backward playback")
                    else:
                        self.is_playing = True
                        self.playback_direction = -1
                        logger.info("Started backward playback")
                elif key == ord('p'):
                    self.save_current_frame(umi_masked, robot_masked, overlay)
                elif key == ord('h'):
                    self.show_help()
                elif key == 83:  # Right arrow
                    self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
                    self.is_playing = False
                elif key == 81:  # Left arrow
                    self.current_frame = max(self.current_frame - 1, 0)
                    self.is_playing = False
                elif key == 84:  # Up arrow
                    self.current_frame = min(self.current_frame + 10, self.num_frames - 1)
                    self.is_playing = False
                elif key == 82:  # Down arrow
                    self.current_frame = max(self.current_frame - 10, 0)
                    self.is_playing = False
                
                # Handle automatic playback
                if self.is_playing:
                    # Update frame based on playback direction
                    if self.playback_direction > 0:
                        # Forward playback
                        self.current_frame += 1
                        if self.current_frame >= self.num_frames:
                            self.current_frame = self.num_frames - 1
                            self.is_playing = False
                            logger.info("Reached end of video, stopped playback")
                    else:
                        # Backward playback
                        self.current_frame -= 1
                        if self.current_frame < 0:
                            self.current_frame = 0
                            self.is_playing = False
                            logger.info("Reached beginning of video, stopped playback")
            
            cv2.destroyAllWindows()
            logger.info("Visualization ended")
            return True
            
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            return False
    
    def save_current_frame(self, umi_masked: np.ndarray, robot_masked: np.ndarray, overlay: np.ndarray):
        """Save current frame images."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path("mask_offset_visualization")
            output_dir.mkdir(exist_ok=True)
            
            # Save individual images
            umi_path = output_dir / f"umi_masked_frame_{self.current_frame:04d}_{timestamp}.png"
            robot_path = output_dir / f"robot_masked_frame_{self.current_frame:04d}_{timestamp}.png"
            overlay_path = output_dir / f"overlay_frame_{self.current_frame:04d}_{timestamp}.png"
            
            cv2.imwrite(str(umi_path), umi_masked)
            cv2.imwrite(str(robot_path), robot_masked)
            cv2.imwrite(str(overlay_path), overlay)
            
            logger.info(f"Saved frame {self.current_frame} to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
    
    def show_help(self):
        """Show help information."""
        help_text = """
Mask Offset Visualizer Help
==========================

Controls:
  Arrow Keys: Navigate frames
    Left/Right: Previous/Next frame
    Up/Down: Jump 10 frames
  's': Start/stop forward playback
  'a': Start/stop backward playback
  'q': Quit visualization
  'r': Reset to frame 0
  'p': Save current frame
  'h': Show this help

Windows:
  1. UMI Masked: UMI image with UMI mask applied
  2. Robot Masked: Robot image with robot mask applied
  3. Overlay with Offset: Both images overlaid with offset compensation

Colors in Overlay:
  Green: UMI masked areas
  Red: Robot masked areas (after offset)
  Yellow: Overlapping areas

Playback:
  - Press 's' to start forward playback, press again to stop
  - Press 'a' to start backward playback, press again to stop
  - Playback automatically stops at video boundaries
  - Arrow keys will stop any active playback
        """
        print(help_text)
    
    def run(self) -> bool:
        """
        Run the complete visualization pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting mask offset visualization pipeline")
        
        # Load video data
        if not self.load_video_data():
            logger.error("Failed to load video data")
            return False
        
        # Load masks
        if not self.load_masks():
            logger.error("Failed to load masks")
            return False
        
        # Load offset data
        if not self.load_offset_data():
            logger.error("Failed to load offset data")
            return False
        
        # Start visualization
        if not self.visualize():
            logger.error("Failed to start visualization")
            return False
        
        logger.info("Mask offset visualization completed successfully")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize UMI and robot masked images with offset compensation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_offset_visualizer.py --umi_video saved_data/episode_0000.hdf5 --robot_video robot_data/episode_0000.hdf5 --umi_mask masks/umi/left_mask.png --robot_mask masks/robot/right_mask.png --offset_file mask_offset/mask_offset_analysis_20250929_143350.json
        """
    )
    
    parser.add_argument(
        "--umi_video",
        type=Path,
        required=True,
        help="Path to UMI video HDF5 file"
    )
    
    parser.add_argument(
        "--robot_video",
        type=Path,
        required=True,
        help="Path to robot video HDF5 file"
    )
    
    parser.add_argument(
        "--umi_mask",
        type=Path,
        required=True,
        help="Path to UMI mask file"
    )
    
    parser.add_argument(
        "--robot_mask",
        type=Path,
        required=True,
        help="Path to robot mask file"
    )
    
    parser.add_argument(
        "--offset_file",
        type=Path,
        required=True,
        help="Path to offset analysis JSON file"
    )
    
    parser.add_argument(
        "--umi_camera_key",
        type=str,
        default="OpenCVCameraDevice_1",
        help="Camera key for UMI video (default: OpenCVCameraDevice_1)"
    )
    
    parser.add_argument(
        "--robot_camera_key",
        type=str,
        default="OpenCVCameraDevice_1",
        help="Camera key for robot video (default: OpenCVCameraDevice_1)"
    )
    
    parser.add_argument(
        "--umi_start_frame",
        type=int,
        default=0,
        help="Starting frame index for UMI video (default: 0)"
    )
    
    parser.add_argument(
        "--robot_start_frame",
        type=int,
        default=0,
        help="Starting frame index for robot video (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.umi_video.exists():
        logger.error(f"UMI video file not found: {args.umi_video}")
        sys.exit(1)
    
    if not args.robot_video.exists():
        logger.error(f"Robot video file not found: {args.robot_video}")
        sys.exit(1)
    
    if not args.umi_mask.exists():
        logger.error(f"UMI mask file not found: {args.umi_mask}")
        sys.exit(1)
    
    if not args.robot_mask.exists():
        logger.error(f"Robot mask file not found: {args.robot_mask}")
        sys.exit(1)
    
    if not args.offset_file.exists():
        logger.error(f"Offset file not found: {args.offset_file}")
        sys.exit(1)
    
    # Create and run visualizer
    visualizer = MaskOffsetVisualizer(
        umi_video_path=args.umi_video,
        robot_video_path=args.robot_video,
        umi_mask_path=args.umi_mask,
        robot_mask_path=args.robot_mask,
        offset_file_path=args.offset_file,
        umi_camera_key=args.umi_camera_key,
        robot_camera_key=args.robot_camera_key,
        umi_start_frame=args.umi_start_frame,
        robot_start_frame=args.robot_start_frame
    )
    
    success = visualizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
