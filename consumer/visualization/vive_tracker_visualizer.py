#!/usr/bin/env python3
"""
Pose Visualizer - 3D visualization of 6D pose data from Vive Trackers.

This module provides a 3D visualizer that reads 6D pose data from shared memory (launch with device starter)
and displays it in real-time using matplotlib 3D plotting.

Author: Han Xue, Jun Lv
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
from utils.logger_config import logger
from utils.time_control import precise_loop_timing
from utils.rerun_visualization import OpenXRTrackerVisualizer
from consumer.core.base import BaseConsumer


class PoseVisualizer(BaseConsumer):
    """
    3D pose visualizer for Vive Tracker data.
    
    This class reads 6D pose data from shared memory and provides
    real-time 3D visualization of tracker positions and orientations.
    
    Supports multiple data shapes:
    - [7]: Single tracker (legacy format, converted to [1,7])
    - [1,7]: Single tracker 
    - [2,7]: Dual tracker
    - [N,7]: N trackers
    
    Each tracker gets its own trajectory visualization and unique naming.
    """
    
    def __init__(self, summary_shm_name: str = "device_summary_data", 
                 max_trajectory_points: int = 100, update_interval_ms: int = 50) -> None:
        """
        Initialize the pose visualizer.
        
        Args:
            summary_shm_name: Name of the summary shared memory
            max_trajectory_points: Maximum number of trajectory points to display
            update_interval_ms: Update interval in milliseconds
        """
        super().__init__(summary_shm_name)
        
        self.max_trajectory_points: int = max_trajectory_points
        self.update_interval_ms: int = update_interval_ms
        
        # Trajectory storage for each tracker
        self.trajectories: Dict[str, deque] = {}
        self.timestamps: Dict[str, deque] = {}
        
        # Thread for visualization loop
        self.visualization_thread: Optional[threading.Thread] = None
        
        # Initialize rerun visualizer using the new utility class
        self.visualizer = OpenXRTrackerVisualizer(
            app_name="pose_visualizer",
            spawn=True
        )

    
    def _update_visualization(self) -> None:
        """Update the visualization with current pose data."""
        try:
            # Read data from all Vive Tracker devices
            all_data = self.read_all_device_data()
            logger.debug(f"All data: {all_data}")
            if all_data is None:
                return
            
            # Collect tracker data for batch visualization
            tracker_data: Dict[str, Dict[str, Any]] = {}
            current_time_ns = time.time_ns()
            
            for device_key, data in all_data.items():
                logger.debug(f"Device key: {device_key}, data: {data}")
                if (device_key in self.devices and 
                    self.devices[device_key]['type'] == 'ViveTrackerDevice' and
                    data is not None):
                    
                    timestamp_ns, pose_data = data
                    
                    # Handle different pose data shapes: [7], [1,7], [2,7], etc.
                    pose_array = np.array(pose_data)
                    
                    # Reshape to ensure consistent 2D format: (num_trackers, 7)
                    if pose_array.ndim == 1 and len(pose_array) == 7:
                        # Single tracker case: [7] -> [1, 7]
                        pose_array = pose_array.reshape(1, 7)
                    elif pose_array.ndim == 2 and pose_array.shape[1] == 7:
                        # Multi-tracker case: already in [N, 7] format
                        pass
                    else:
                        logger.warning(f"Unsupported pose data shape: {pose_array.shape} for device {device_key}")
                        continue
                    
                    # Process each tracker in the data
                    num_trackers = pose_array.shape[0]
                    for tracker_idx in range(num_trackers):
                        tracker_pose = pose_array[tracker_idx]
                        x, y, z = tracker_pose[0:3]
                        qx, qy, qz, qw = tracker_pose[3:7]
                        
                        # Check if pose is valid (not all zeros)
                        if np.any(tracker_pose[:3] != 0):
                            # Create unique trajectory key for each tracker
                            trajectory_key = f"{device_key}_tracker_{tracker_idx}"
                            
                            # Initialize trajectory storage for new tracker
                            if trajectory_key not in self.trajectories:
                                self.trajectories[trajectory_key] = deque(maxlen=self.max_trajectory_points)
                                self.timestamps[trajectory_key] = deque(maxlen=self.max_trajectory_points)
                                logger.info(f"Initialized trajectory storage for {trajectory_key}")
                            
                            # Update trajectory
                            self.trajectories[trajectory_key].append([x, y, z])
                            self.timestamps[trajectory_key].append(timestamp_ns)

                            # Prepare data for batch visualization with descriptive names
                            tracker_name = f"device_{device_key}_tracker_{tracker_idx}"
                            tracker_data[tracker_name] = {
                                'position': [x, y, z],
                                'quaternion': [qx, qy, qz, qw]  # Already in [x,y,z,w] format
                            }
            
            # Visualize all trackers at once using the new visualizer
            if tracker_data:
                self.visualizer.log_multiple_trackers(tracker_data, current_time_ns)
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
    
    def _visualization_loop(self) -> None:
        """Main visualization loop that runs in a separate thread."""
        logger.info(f"Visualization loop started with {self.update_interval_ms}ms interval")
        
        # Create precise timing function
        update_interval_sec = self.update_interval_ms / 1000.0
        wait_for_next_iteration = precise_loop_timing(update_interval_sec)
        
        while self.running:
            try:
                self._update_visualization()
                # Wait for next iteration using precise timing
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in visualization loop: {e}")
                break
        
        logger.info("Visualization loop stopped")
    
    def start_visualization(self) -> None:
        """Start the 3D visualization."""
        if self.running:
            logger.info("Visualizer is already running")
            return
        
        # Connect to shared memory
        if not self.connect():
            logger.error("Failed to connect to shared memory")
            return
        
        self.running = True
        logger.info("Starting 3D pose visualization...")
        
        # Start the visualization loop in a separate thread
        self.visualization_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.visualization_thread.start()

    
    def stop_visualization(self) -> None:
        """Stop the 3D visualization."""
        if not self.running:
            return
        
        logger.info("Stopping 3D pose visualization...")
        self.running = False

        # Wait for the visualization thread to finish
        if hasattr(self, 'visualization_thread') and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)  # Wait up to 2 seconds
            if self.visualization_thread.is_alive():
                logger.warning("Visualization thread did not stop gracefully")

        # Disconnect from shared memory
        self.disconnect()
    
    def get_trajectory_data(self, device_key: str, tracker_idx: Optional[int] = None) -> Union[Tuple[List[List[float]], List[int]], Dict[str, Tuple[List[List[float]], List[int]]]]:
        """
        Get trajectory data for a specific device or tracker.
        
        Args:
            device_key: Device identifier
            tracker_idx: Optional tracker index. If None, returns all trackers for the device
            
        Returns:
            If tracker_idx is specified: tuple (positions, timestamps) lists
            If tracker_idx is None: dict mapping tracker keys to (positions, timestamps) tuples
        """
        if tracker_idx is not None:
            # Return data for specific tracker
            trajectory_key = f"{device_key}_tracker_{tracker_idx}"
            if trajectory_key in self.trajectories:
                return list(self.trajectories[trajectory_key]), list(self.timestamps[trajectory_key])
            return [], []
        else:
            # Return data for all trackers of this device
            result = {}
            for traj_key in self.trajectories:
                if traj_key.startswith(f"{device_key}_tracker_"):
                    result[traj_key] = (list(self.trajectories[traj_key]), list(self.timestamps[traj_key]))
            return result
    
    def clear_trajectories(self) -> None:
        """Clear all trajectory data."""
        for trajectory_key in self.trajectories:
            self.trajectories[trajectory_key].clear()
            self.timestamps[trajectory_key].clear()
    
    def save_trajectory_data(self, filename: str) -> None:
        """
        Save trajectory data to file.
        
        Args:
            filename: Output filename
        """
        import json
        
        data = {}
        for trajectory_key in self.trajectories:
            positions = list(self.trajectories[trajectory_key])
            timestamps = list(self.timestamps[trajectory_key])
            data[trajectory_key] = {
                'positions': positions,
                'timestamps': timestamps
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Trajectory data saved to {filename}")
    
    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        self.stop_visualization()


def main() -> None:
    """Main function to run the pose visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Pose Visualizer for Vive Trackers")
    parser.add_argument("--summary-shm", "-s", type=str, default="device_summary_data",
                        help="Summary shared memory name (default: device_summary_data)")
    parser.add_argument("--max-points", "-m", type=int, default=100,
                        help="Maximum trajectory points to display (default: 100)")
    parser.add_argument("--update-interval", "-u", type=int, default=50,
                        help="Update interval in milliseconds (default: 50)")
    parser.add_argument("--save-trajectory", "-t", type=str, default=None,
                        help="Save trajectory data to file (optional)")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = PoseVisualizer(
        summary_shm_name=args.summary_shm,
        max_trajectory_points=args.max_points,
        update_interval_ms=args.update_interval
    )
    
    logger.info("3D Pose Visualizer for Vive Trackers")
    logger.info("=====================================")
    logger.info(f"Summary SHM: {args.summary_shm}")
    logger.info(f"Max trajectory points: {args.max_points}")
    logger.info(f"Update interval: {args.update_interval} ms")
    
    try:
        logger.info("Starting visualization. Press Ctrl+C to stop...")
        visualizer.start_visualization()
        
        # Keep the main thread alive while visualization is running
        while visualizer.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        visualizer.stop_visualization()
        
        # Save trajectory data if requested
        if args.save_trajectory:
            visualizer.save_trajectory_data(args.save_trajectory)


if __name__ == "__main__":
    main() 