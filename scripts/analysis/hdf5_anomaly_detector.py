"""
HDF5 Episode Anomaly Detector

This script detects various anomalies in HDF5 episode data including:
- Data frequency anomalies
- Data length anomalies  
- Timestamp discontinuity
- Image quality issues (black images)
- Zero data detection
- Trajectory jumps
- Change point detection

All detection functions are implemented as methods in a single EpisodeAnomalyDetector class,
making it easy to add new detection functions.

usage:
    python scripts/analysis/hdf5_anomaly_detector.py 
    python scripts/analysis/hdf5_anomaly_detector.py --config configs/anomaly_config/default.yaml

Author: Junxie Xu
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import h5py
import numpy as np
import yaml
from tqdm import tqdm

# Import logger from utils
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.logger_config import logger
from utils.hdf5_utils import load_episode_with_timestamps


# ============================================================================
# Episode Anomaly Detector Class
# ============================================================================

class EpisodeAnomalyDetector:
    """
    Detects anomalies in a single HDF5 episode.
    
    All data is stored as instance attributes for easy access by detection methods.
    New detection methods can be added easily as class methods.
    """
    
    def __init__(
        self, 
        episode_idx: int,
        episode_path: Path,
        h5_file: h5py.File,
        feat: Dict[str, str],
        config: Dict[str, Any]
    ):
        """
        Initialize detector with episode data.
        
        Args:
            episode_idx: Episode index
            episode_path: Path to episode file
            h5_file: Open HDF5 file handle
            feat: Feature mapping configuration
            config: Detection configuration
        """
        self.episode_idx = episode_idx
        self.episode_path = episode_path
        self.feat = feat
        self.config = config
        
        # Load all data from HDF5
        data_dict, timestamp_dict = load_episode_with_timestamps(h5_file, feat)
        
        # Extract specific data streams based on feat mapping
        self.pose_data = None
        self.pose_timestamps = None
        if 'pose_unified' in feat and feat['pose_unified'] in data_dict:
            pose_key = feat['pose_unified']
            self.pose_data = data_dict[pose_key]
            self.pose_timestamps = timestamp_dict[pose_key]
        
        self.img_left_data = None
        self.img_left_timestamps = None
        if 'img_left' in feat and feat['img_left'] in data_dict:
            img_left_key = feat['img_left']
            self.img_left_data = data_dict[img_left_key]
            self.img_left_timestamps = timestamp_dict[img_left_key]
        
        self.img_right_data = None
        self.img_right_timestamps = None
        if 'img_right' in feat and feat['img_right'] in data_dict:
            img_right_key = feat['img_right']
            self.img_right_data = data_dict[img_right_key]
            self.img_right_timestamps = timestamp_dict[img_right_key]
        
        self.grip_left_data = None
        self.grip_left_timestamps = None
        if 'grip_left' in feat and feat['grip_left'] in data_dict:
            grip_left_key = feat['grip_left']
            self.grip_left_data = data_dict[grip_left_key]
            self.grip_left_timestamps = timestamp_dict[grip_left_key]
        
        self.grip_right_data = None
        self.grip_right_timestamps = None
        if 'grip_right' in feat and feat['grip_right'] in data_dict:
            grip_right_key = feat['grip_right']
            self.grip_right_data = data_dict[grip_right_key]
            self.grip_right_timestamps = timestamp_dict[grip_right_key]
        
        # Storage for detected anomalies
        self.anomalies = []
    
    def _add_anomaly(self, severity: str, message: str, details: Dict[str, Any] = None):
        """
        Add an anomaly to the list.
        
        Args:
            severity: 'error', 'warning', or 'info'
            message: Description of the anomaly
            details: Additional details dictionary
        """
        anomaly = {
            'severity': severity,
            'message': message,
            'episode_idx': self.episode_idx,
            'episode_path': str(self.episode_path)
        }
        if details:
            anomaly['details'] = details
        self.anomalies.append(anomaly)
    
    # ========================================================================
    # Detection Functions
    # ========================================================================
    
    def detect_frequency_anomalies(self):
        """
        Detect if data frequencies are within expected ranges.
        
        Config format:
            expected_fps:
                pose: [80, 120]
                image: [20, 30]
                grip: [40, 60]
            fps_tolerance: 0.2
        """
        if 'expected_fps' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'expected_fps' not found in config. "
                "Skipping frequency anomaly detection."
            )
            return
        
        expected_fps = self.config['expected_fps']
        if 'fps_tolerance' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'fps_tolerance' not found in config. "
                "Skipping frequency anomaly detection."
            )
            return
        tolerance = float(self.config['fps_tolerance'])
        
        # Check pose frequency
        if self.pose_data is not None and len(self.pose_timestamps) > 1:
            if 'pose' not in expected_fps:
                logger.warning(
                    f"Episode {self.episode_idx}: 'pose' not found in expected_fps config. "
                    "Skipping pose frequency check."
                )
            else:
                duration = self.pose_timestamps[-1] - self.pose_timestamps[0]
                if duration > 0:
                    actual_fps = (len(self.pose_timestamps) - 1) / duration
                    expected_range = expected_fps['pose']
                    min_fps = expected_range[0] * (1 - tolerance)
                    max_fps = expected_range[1] * (1 + tolerance)
                    
                    if actual_fps < min_fps or actual_fps > max_fps:
                        self._add_anomaly(
                            'error',
                            f"Pose data frequency {actual_fps:.1f}Hz outside expected range {expected_range[0]}-{expected_range[1]}Hz",
                            {'actual_fps': actual_fps, 'expected_range': expected_range}
                        )
                        logger.error(
                            f"Episode {self.episode_idx}: Pose freq {actual_fps:.1f}Hz, "
                            f"expected {expected_range[0]}-{expected_range[1]}Hz"
                        )
        
        # Check image frequencies
        for img_name, img_data, img_timestamps in [
            ('img_left', self.img_left_data, self.img_left_timestamps),
            ('img_right', self.img_right_data, self.img_right_timestamps)
        ]:
            if img_data is not None and len(img_timestamps) > 1:
                if 'image' not in expected_fps:
                    logger.warning(
                        f"Episode {self.episode_idx}: 'image' not found in expected_fps config. "
                        f"Skipping {img_name} frequency check."
                    )
                else:
                    duration = img_timestamps[-1] - img_timestamps[0]
                    if duration > 0:
                        actual_fps = (len(img_timestamps) - 1) / duration
                        expected_range = expected_fps['image']
                        min_fps = expected_range[0] * (1 - tolerance)
                        max_fps = expected_range[1] * (1 + tolerance)
                        
                        if actual_fps < min_fps or actual_fps > max_fps:
                            self._add_anomaly(
                                'error',
                                f"{img_name} frequency {actual_fps:.1f}Hz outside expected range {expected_range[0]}-{expected_range[1]}Hz",
                                {'device': img_name, 'actual_fps': actual_fps, 'expected_range': expected_range}
                            )
                            logger.error(
                                f"Episode {self.episode_idx}: {img_name} freq {actual_fps:.1f}Hz, "
                                f"expected {expected_range[0]}-{expected_range[1]}Hz"
                            )
        
        # Check gripper frequencies
        for grip_name, grip_data, grip_timestamps in [
            ('grip_left', self.grip_left_data, self.grip_left_timestamps),
            ('grip_right', self.grip_right_data, self.grip_right_timestamps)
        ]:
            if grip_data is not None and len(grip_timestamps) > 1:
                if 'grip' not in expected_fps:
                    logger.warning(
                        f"Episode {self.episode_idx}: 'grip' not found in expected_fps config. "
                        f"Skipping {grip_name} frequency check."
                    )
                else:
                    duration = grip_timestamps[-1] - grip_timestamps[0]
                    if duration > 0:
                        actual_fps = (len(grip_timestamps) - 1) / duration
                        expected_range = expected_fps['grip']
                        min_fps = expected_range[0] * (1 - tolerance)
                        max_fps = expected_range[1] * (1 + tolerance)
                        
                        if actual_fps < min_fps or actual_fps > max_fps:
                            self._add_anomaly(
                                'error',
                                f"{grip_name} frequency {actual_fps:.1f}Hz outside expected range {expected_range[0]}-{expected_range[1]}Hz",
                                {'device': grip_name, 'actual_fps': actual_fps, 'expected_range': expected_range}
                            )
                            logger.error(
                                f"Episode {self.episode_idx}: {grip_name} freq {actual_fps:.1f}Hz, "
                                f"expected {expected_range[0]}-{expected_range[1]}Hz"
                            )
    
    def detect_length_anomalies(self):
        """
        Detect if any data stream has significantly fewer samples than others.
        Only compares within same category: img vs img, grip vs grip.
        
        Config: min_length_ratio
        """
        if 'min_length_ratio' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'min_length_ratio' not found in config. "
                "Skipping length anomaly detection."
            )
            return
        
        min_ratio = float(self.config['min_length_ratio'])
        
        # Compare image lengths (left vs right)
        img_lengths = {}
        if self.img_left_data is not None:
            img_lengths['img_left'] = len(self.img_left_data)
        if self.img_right_data is not None:
            img_lengths['img_right'] = len(self.img_right_data)
        
        # Check for completely missing cameras
        expected_cameras = ['img_left', 'img_right']
        for camera_name in expected_cameras:
            camera_data = getattr(self, f"{camera_name}_data", None)
            if camera_name not in img_lengths:  # Camera has no data
                self._add_anomaly(
                    'error',
                    f"Device '{camera_name}' is completely missing (no data found)",
                    {
                        'device': camera_name,
                        'anomaly_type': 'missing_device_data',
                        'category': 'image'
                    }
                )
                logger.error(
                    f"Episode {self.episode_idx}: {camera_name} has no data"
                )
        
        if len(img_lengths) > 1:
            max_img_length = max(img_lengths.values())
            for device_name, length in img_lengths.items():
                ratio = length / max_img_length if max_img_length > 0 else 0
                if ratio < min_ratio:
                    self._add_anomaly(
                        'error',
                        f"Device '{device_name}' has significantly less data than other images ({length} vs {max_img_length}, ratio {ratio:.2%})",
                        {'device': device_name, 'length': length, 'max_length': max_img_length, 'ratio': ratio, 'category': 'image'}
                    )
                    logger.error(
                        f"Episode {self.episode_idx}: {device_name} has only {length} frames "
                        f"vs max image {max_img_length} ({ratio:.2%})"
                    )
        
        # Compare gripper lengths (left vs right)
        grip_lengths = {}
        if self.grip_left_data is not None:
            grip_lengths['grip_left'] = len(self.grip_left_data)
        if self.grip_right_data is not None:
            grip_lengths['grip_right'] = len(self.grip_right_data)
        
        if len(grip_lengths) > 1:
            max_grip_length = max(grip_lengths.values())
            for device_name, length in grip_lengths.items():
                ratio = length / max_grip_length if max_grip_length > 0 else 0
                if ratio < min_ratio:
                    self._add_anomaly(
                        'error',
                        f"Device '{device_name}' has significantly less data than other grippers ({length} vs {max_grip_length}, ratio {ratio:.2%})",
                        {'device': device_name, 'length': length, 'max_length': max_grip_length, 'ratio': ratio, 'category': 'gripper'}
                    )
                    logger.error(
                        f"Episode {self.episode_idx}: {device_name} has only {length} frames "
                        f"vs max gripper {max_grip_length} ({ratio:.2%})"
                    )
    
    def detect_timestamp_discontinuity(self):
        """
        Detect discontinuities in timestamps (large gaps or non-monotonic).
        
        Config: max_timestamp_gap (seconds)
        """
        if 'max_timestamp_gap' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'max_timestamp_gap' not found in config. "
                "Skipping timestamp discontinuity detection."
            )
            return
        
        max_gap = float(self.config['max_timestamp_gap'])
        
        for device_name, timestamps in [
            ('pose', self.pose_timestamps),
            ('img_left', self.img_left_timestamps),
            ('img_right', self.img_right_timestamps),
            ('grip_left', self.grip_left_timestamps),
            ('grip_right', self.grip_right_timestamps)
        ]:
            # First check: if any device has < 70 timestamps, recording is too short
            if timestamps is not None and len(timestamps) < 70:
                self._add_anomaly(
                    'error',
                    "Recording time too short",
                    {'anomaly_type': 'short_recording'}
                )
                logger.error(
                    f"Episode {self.episode_idx}: Recording time too short"
                )
                return  # End detection for this episode
            
            # Second check: skip devices with insufficient data for discontinuity analysis
            if timestamps is None or len(timestamps) < 2:
                continue
            
            # Check for non-monotonic timestamps
            time_diffs = np.diff(timestamps)
            if np.any(time_diffs <= 0):
                num_violations = np.sum(time_diffs <= 0)
                self._add_anomaly(
                    'error',
                    f"Device '{device_name}' has non-monotonic timestamps",
                    {'device': device_name, 'num_violations': int(num_violations)}
                )
                logger.error(
                    f"Episode {self.episode_idx}: {device_name} has {num_violations} "
                    f"non-monotonic timestamp pairs"
                )
            
            # Check for large gaps
            max_gap_found = np.max(time_diffs)
            if max_gap_found > max_gap:
                gap_indices = np.where(time_diffs > max_gap)[0]
                self._add_anomaly(
                    'error',
                    f"Device '{device_name}' has large timestamp gaps",
                    {
                        'device': device_name,
                        'max_gap': f"{max_gap_found:.3f}s",
                        'num_gaps': len(gap_indices),
                        'gap_indices': gap_indices.tolist()[:10]
                    }
                )
                logger.error(
                    f"Episode {self.episode_idx}: {device_name} has {len(gap_indices)} "
                    f"gaps > {max_gap}s (max: {max_gap_found:.3f}s)"
                )
    
    def detect_black_images(self):
        """
        Detect if images are too dark (possibly all-black or camera failure).
        
        Config: min_brightness, image_check_ratio
        """
        if 'min_brightness' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'min_brightness' not found in config. "
                "Skipping black image detection."
            )
            return
        
        if 'image_check_ratio' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'image_check_ratio' not found in config. "
                "Skipping black image detection."
            )
            return
        
        min_brightness = float(self.config['min_brightness'])
        check_ratio = float(self.config['image_check_ratio'])
        
        for img_name, img_data in [
            ('img_left', self.img_left_data),
            ('img_right', self.img_right_data)
        ]:
            if img_data is None or len(img_data) == 0:
                # Report missing image data as an error
                self._add_anomaly(
                    'error',
                    f"Device '{img_name}' has no image data (camera failure or not connected)",
                    {
                        'device': img_name,
                        'data_status': 'missing' if img_data is None else 'empty',
                        'anomaly_type': 'missing_image_data'
                    }
                )
                logger.error(
                    f"Episode {self.episode_idx}: {img_name} has no image data"
                )
                continue
            
            # Sample images to check
            num_check = max(1, int(len(img_data) * check_ratio))
            check_indices = np.linspace(0, len(img_data) - 1, num_check, dtype=int)
            
            num_dark = 0
            for idx in check_indices:
                img = img_data[idx]
                
                # Convert to grayscale if needed
                if img.ndim == 3 and img.shape[-1] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img
                
                mean_brightness = np.mean(gray)
                
                if mean_brightness < min_brightness:
                    num_dark += 1
            
            if num_dark > 0:
                self._add_anomaly(
                    'error',
                    f"Device '{img_name}' has dark images (possible camera failure)",
                    {
                        'device': img_name,
                        'num_dark': num_dark,
                        'num_checked': num_check,
                        'total_images': len(img_data)
                    }
                )
                logger.error(
                    f"Episode {self.episode_idx}: {img_name} has {num_dark}/{num_check} dark images"
                )
    
    def detect_zero_data(self):
        """
        Detect if data is mostly zeros or frozen at a constant value (device failure or not connected).
        
        For pose data (trajectory): Check if xyz and quaternion are all zeros or frozen -> error
        For gripper data: Check if data is all zeros or frozen -> warning (might be single-arm data)
        
        Config: 
            zero_data_threshold: ratio of zeros to consider as error
            static_data_std_threshold: maximum standard deviation to consider data as frozen
        """
        if 'zero_data_threshold' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'zero_data_threshold' not found in config. "
                "Skipping zero/static data detection."
            )
            return
        
        if 'static_data_std_threshold' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'static_data_std_threshold' not found in config. "
                "Skipping zero/static data detection."
            )
            return
        
        zero_threshold = float(self.config['zero_data_threshold'])
        static_std_threshold = float(self.config['static_data_std_threshold'])
        
        # Check pose data (trajectory) 
        if self.pose_data is not None and len(self.pose_data) > 0:
            # Handle bimanual data (T, 2, 7)
            if self.pose_data.ndim == 3 and self.pose_data.shape[1] == 2:
                # Bimanual: check each arm separately
                for arm_idx in range(2):
                    arm_name = 'left' if arm_idx == 0 else 'right'
                    arm_data = self.pose_data[:, arm_idx, :]  # (T, 7)
                    
                    # Extract xyz (first 3 dims) and quaternion (last 4 dims)
                    xyz_data = arm_data[:, :3]
                    quat_data = arm_data[:, 3:7]
                    
                    # Check xyz
                    xyz_zero_ratio = np.sum(np.abs(xyz_data) < 1e-9) / xyz_data.size
                    xyz_std = np.std(xyz_data, axis=0)
                    
                    if xyz_zero_ratio > zero_threshold:
                        self._add_anomaly(
                            'error',
                            f"Pose {arm_name} arm XYZ data is {xyz_zero_ratio:.2%} zeros (device failure or not connected)",
                            {'device': f'pose_{arm_name}', 'component': 'xyz', 'zero_ratio': xyz_zero_ratio, 'anomaly_type': 'zero_data'}
                        )
                        logger.error(
                            f"Episode {self.episode_idx}: Pose {arm_name} arm XYZ has {xyz_zero_ratio:.2%} zero values"
                        )
                    elif np.all(xyz_std < static_std_threshold):
                        mean_vals = np.mean(xyz_data, axis=0)
                        self._add_anomaly(
                            'error',
                            f"Pose {arm_name} arm XYZ is frozen at constant value [{mean_vals[0]:.6f}, {mean_vals[1]:.6f}, {mean_vals[2]:.6f}] (std: {xyz_std})",
                            {'device': f'pose_{arm_name}', 'component': 'xyz', 'frozen_values': mean_vals.tolist(), 'std': xyz_std.tolist(), 'anomaly_type': 'static_data'}
                        )
                        logger.error(
                            f"Episode {self.episode_idx}: Pose {arm_name} arm XYZ frozen at [{mean_vals[0]:.6f}, {mean_vals[1]:.6f}, {mean_vals[2]:.6f}]"
                        )
                    
                    # Check quaternion
                    quat_zero_ratio = np.sum(np.abs(quat_data) < 1e-9) / quat_data.size
                    quat_std = np.std(quat_data, axis=0)
                    
                    if quat_zero_ratio > zero_threshold:
                        self._add_anomaly(
                            'error',
                            f"Pose {arm_name} arm quaternion data is {quat_zero_ratio:.2%} zeros (device failure or not connected)",
                            {'device': f'pose_{arm_name}', 'component': 'quaternion', 'zero_ratio': quat_zero_ratio, 'anomaly_type': 'zero_data'}
                        )
                        logger.error(
                            f"Episode {self.episode_idx}: Pose {arm_name} arm quaternion has {quat_zero_ratio:.2%} zero values"
                        )
                    elif np.all(quat_std < static_std_threshold):
                        mean_vals = np.mean(quat_data, axis=0)
                        self._add_anomaly(
                            'error',
                            f"Pose {arm_name} arm quaternion is frozen at constant value [{mean_vals[0]:.6f}, {mean_vals[1]:.6f}, {mean_vals[2]:.6f}, {mean_vals[3]:.6f}] (std: {quat_std})",
                            {'device': f'pose_{arm_name}', 'component': 'quaternion', 'frozen_values': mean_vals.tolist(), 'std': quat_std.tolist(), 'anomaly_type': 'static_data'}
                        )
                        logger.error(
                            f"Episode {self.episode_idx}: Pose {arm_name} arm quaternion frozen at [{mean_vals[0]:.6f}, {mean_vals[1]:.6f}, {mean_vals[2]:.6f}, {mean_vals[3]:.6f}]"
                        )
            else:
                logger.warning(
                    f"Episode {self.episode_idx}: Unexpected pose data shape {self.pose_data.shape}, "
                    f"expected (T, 2, 7) for bimanual data"
                )
        
        frozen_grippers = []  # Collect frozen grippers to check if both are frozen
        
        for grip_name, grip_data in [
            ('grip_left', self.grip_left_data),
            ('grip_right', self.grip_right_data)
        ]:
            if grip_data is not None and len(grip_data) > 0:
                # Check for zero data
                zero_ratio = np.sum(np.abs(grip_data) < 1e-9) / grip_data.size
                if zero_ratio > zero_threshold:
                    self._add_anomaly(
                        'warning',
                        f"{grip_name} data is {zero_ratio:.2%} zeros (possibly single-arm data collection)",
                        {'device': grip_name, 'zero_ratio': zero_ratio, 'anomaly_type': 'zero_data'}
                    )
                    logger.warning(
                        f"Episode {self.episode_idx}: {grip_name} has {zero_ratio:.2%} zero values (possibly single-arm data)"
                    )
                    continue
                
                # Check for frozen data
                std = np.std(grip_data)
                if std < static_std_threshold:
                    mean_val = np.mean(grip_data)
                    frozen_grippers.append({
                        'name': grip_name,
                        'mean_val': float(mean_val),
                        'std': float(std)
                    })
        
        # Report frozen grippers - only report error if both frozen
        if len(frozen_grippers) == 2:
            # Both grippers frozen - error
            for grip_info in frozen_grippers:
                self._add_anomaly(
                    'error',
                    "Both grippers frozen",
                    {'anomaly_type': 'both_grippers_frozen'}
                )
            logger.error(
                f"Episode {self.episode_idx}: Both grippers frozen"
            )
        # Note: Single gripper frozen is not reported (normal for single-arm operation)
    
    def detect_trajectory_jumps(self):
        """
        Detect trajectory anomalies including position jumps, velocity extremes, and quaternion jumps (orientation changes).
        
        Config: 
            max_position_jump: Maximum allowed position jump in meters
            max_velocity: Maximum allowed velocity in m/s
            max_quaternion_jump: Maximum allowed quaternion value jump threshold
        """
        if self.pose_data is None or len(self.pose_data) < 2:
            return
        
        if 'max_position_jump' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'max_position_jump' not found in config. "
                "Skipping trajectory jump detection."
            )
            return
        
        if 'max_velocity' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'max_velocity' not found in config. "
                "Skipping trajectory jump detection."
            )
            return
        
        if 'max_quaternion_jump' not in self.config:
            logger.error(
                f"Episode {self.episode_idx}: 'max_quaternion_jump' not found in config. "
                "Skipping quaternion jump detection."
            )
            return

        max_pos_jump = float(self.config['max_position_jump'])
        max_velocity = float(self.config['max_velocity'])
        max_quat_jump = float(self.config['max_quaternion_jump'])
        
        # Handle bimanual data (T, 2, 7)
        if self.pose_data.ndim == 3 and self.pose_data.shape[1] == 2:
            # Bimanual: check each arm separately
            for arm_idx, arm_name in [(0, 'left'), (1, 'right')]:
                arm_data = self.pose_data[:, arm_idx, :]  # (T, 7)
                positions = arm_data[:, :3]  # Extract xyz positions (T, 3)
                
                # Check position jumps
                position_diffs = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
                jump_indices = np.where(position_diffs > max_pos_jump)[0]
                
                if len(jump_indices) > 0:
                    max_jump = np.max(position_diffs[jump_indices])
                    self._add_anomaly(
                        'error',
                        f"{arm_name.capitalize()} arm has large position jumps",
                        {
                            'arm': arm_name,
                            'num_jumps': len(jump_indices),
                            'max_jump': f"{max_jump:.4f}m",
                            'jump_indices': jump_indices.tolist()[:10]
                        }
                    )
                    logger.error(
                        f"Episode {self.episode_idx}: {arm_name} arm has {len(jump_indices)} "
                        f"position jumps > {max_pos_jump}m (max: {max_jump:.4f}m)"
                    )
                
                # Check velocities
                if len(self.pose_timestamps) > 1:
                    time_diffs = np.diff(self.pose_timestamps)
                    time_diffs = np.maximum(time_diffs, 1e-6)  # Avoid division by zero
                    
                    # position_diffs is already scalar distances (1D array)
                    # velocities is also 1D array (scalar speeds in m/s)
                    velocities = position_diffs / time_diffs
                    # Use absolute value since velocities are already scalar magnitudes
                    velocity_magnitudes = np.abs(velocities)
                    
                    extreme_vel_indices = np.where(velocity_magnitudes > max_velocity)[0]
                    
                    if len(extreme_vel_indices) > 0:
                        max_vel = np.max(velocity_magnitudes[extreme_vel_indices])
                        self._add_anomaly(
                            'error',
                            f"{arm_name.capitalize()} arm has extreme velocities",
                            {
                                'arm': arm_name,
                                'num_extreme': len(extreme_vel_indices),
                                'max_velocity': f"{max_vel:.4f}m/s",
                                'indices': extreme_vel_indices.tolist()[:10]
                            }
                        )
                        logger.error(
                            f"Episode {self.episode_idx}: {arm_name} arm has {len(extreme_vel_indices)} "
                            f"extreme velocities > {max_velocity}m/s (max: {max_vel:.4f}m/s)"
                        )
                
                # Check quaternion component jumps (detect sudden value changes in individual components)
                quaternions = arm_data[:, 3:7]  # Extract quaternions (T, 4) - (x, y, z, w) format
                
                if len(quaternions) > 1:
                    quat_diffs = np.abs(quaternions[1:] - quaternions[:-1])  # (T-1, 4)
                    quat_jump_threshold = max_quat_jump
                    component_jumps = quat_diffs > quat_jump_threshold  # (T-1, 4) boolean array
                    frame_has_jump = np.any(component_jumps, axis=1)  # (T-1,) boolean array
                    quat_jump_indices = np.where(frame_has_jump)[0]
                    if len(quat_jump_indices) > 0:
                        max_component_jump = np.max(quat_diffs[quat_jump_indices])
                        total_component_jumps = np.sum(component_jumps[quat_jump_indices])
                        
                        self._add_anomaly(
                            'error',
                            f"{arm_name.capitalize()} arm has quaternion component jumps",
                            {
                                'arm': arm_name,
                                'num_frames_with_jumps': len(quat_jump_indices),
                                'total_component_jumps': int(total_component_jumps),
                                'max_component_jump': f"{max_component_jump:.4f}",
                                'threshold': quat_jump_threshold,
                                'jump_frame_indices': quat_jump_indices.tolist()[:10],
                                'anomaly_type': 'quaternion_component_jump'
                            }
                        )
                        logger.error(
                            f"Episode {self.episode_idx}: {arm_name} arm has {len(quat_jump_indices)} frames "
                            f"with quaternion component jumps > {quat_jump_threshold} "
                            f"(max component jump: {max_component_jump:.4f}, total: {total_component_jumps})"
                        )
                
    def run_all_detections(self) -> List[Dict[str, Any]]:
        """
        Run all detection functions and return list of anomalies.
        
        Returns:
            List of anomaly dictionaries
        """
        # Run all detection methods
        self.detect_frequency_anomalies()
        self.detect_length_anomalies()
        self.detect_timestamp_discontinuity()
        self.detect_black_images()
        self.detect_zero_data()
        self.detect_trajectory_jumps()

        
        return self.anomalies


# ============================================================================
# Dataset Processing Orchestrator
# ============================================================================

def extract_episode_files_from_config(config: Dict[str, Any]) -> Tuple[List[Path], List[Dict], List[str]]:
    """
    Extract episode file paths from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (file_paths, feat_configs, task_names)
    """
    all_files = []
    all_feats = []
    all_tasks = []
    
    data_config = config.get('data', {})
    
    for dataset_dir, dataset_info in data_config.items():
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            logger.warning(f"Dataset directory does not exist: {dataset_dir}")
            continue
        
        # Find all episode files
        files = sorted(dataset_path.glob("episode_*.hdf5"), 
                      key=lambda p: int(p.stem.split("_")[1]))
        
        # Temporarily commented out to test algorithm functionality
        # Filter out invalid episodes
        # invalid_ids = dataset_info.get('invalid_id', [])
        # valid_files = [f for f in files if int(f.stem.split("_")[1]) not in invalid_ids]
        valid_files = files  # Process all files including invalid ones
        
        # Get feature mapping and task
        feat = dataset_info.get('feat', {})
        task = dataset_info.get('task', '')
        
        all_files.extend(valid_files)
        all_feats.extend([feat] * len(valid_files))
        all_tasks.extend([task] * len(valid_files))
        
        logger.info(f"Found {len(valid_files)} valid episodes in {dataset_dir}")
    
    return all_files, all_feats, all_tasks


def process_dataset(config_path: str):
    """
    Process entire dataset and detect anomalies.
    
    Args:
        config_path: Path to anomaly detection configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Extract anomaly_detect configuration
    if 'anomaly_detect' not in full_config:
        raise ValueError("Configuration must have 'anomaly_detect' section")
    
    config = full_config['anomaly_detect']
    
    logger.info("="*80)
    logger.info("Starting HDF5 Episode Anomaly Detection")
    logger.info("="*80)
    
    # Load dataset configuration
    data_config = config.get('data', {})
    if 'dataset_config' in data_config:
        # Load from external config file
        dataset_config_path = Path(data_config['dataset_config'])
        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        logger.info(f"Loaded dataset config from {dataset_config_path}")
    else:
        raise ValueError("Configuration must specify 'dataset_config'")
    
    # Extract episode files
    files, feats, tasks = extract_episode_files_from_config(dataset_config)
    
    if len(files) == 0:
        logger.error("No episode files found!")
        return
    
    logger.info(f"Found {len(files)} episodes to process")
    
    # Process each episode
    all_anomalies = []
    episodes_with_errors = []
    episodes_with_warnings = []
    
    stats = {
        'total_episodes': 0,
        'episodes_with_errors': 0,
        'episodes_with_warnings': 0,
        'total_errors': 0,
        'total_warnings': 0,
    }
    
    for ep_idx, (ep_path, feat, task) in enumerate(
        tqdm(zip(files, feats, tasks), total=len(files), desc="Detecting anomalies")
    ):
        stats['total_episodes'] += 1  # Count episode before processing
        
        try:
            with h5py.File(ep_path, 'r') as h5:
                # Load episode data (needed for both detection and visualization)
                data_dict, timestamp_dict = load_episode_with_timestamps(h5, feat)
                
                # Create detector for this episode
                detector = EpisodeAnomalyDetector(ep_idx, ep_path, h5, feat, config)
                
                # Run all detections
                anomalies = detector.run_all_detections()
                
                if anomalies:
                    all_anomalies.extend(anomalies)
                    
                    # Count errors and warnings
                    has_error = any(a['severity'] == 'error' for a in anomalies)
                    has_warning = any(a['severity'] == 'warning' for a in anomalies)
                    
                    if has_error:
                        stats['episodes_with_errors'] += 1
                        episodes_with_errors.append(ep_idx)
                    
                    if has_warning:
                        stats['episodes_with_warnings'] += 1
                        episodes_with_warnings.append(ep_idx)
                    
                    stats['total_errors'] += sum(1 for a in anomalies if a['severity'] == 'error')
                    stats['total_warnings'] += sum(1 for a in anomalies if a['severity'] == 'warning')
        
        except Exception as e:
            logger.error(f"Error processing episode {ep_idx} ({ep_path}): {e}")
            import traceback
            traceback.print_exc()
            
            # Record the exception as an error anomaly
            all_anomalies.append({
                'severity': 'error',
                'message': f"Exception during processing: {str(e)}",
                'episode_idx': ep_idx,
                'episode_path': str(ep_path),
                'details': {'exception_type': type(e).__name__}
            })
            
            # Update statistics
            stats['episodes_with_errors'] += 1
            stats['total_errors'] += 1
            episodes_with_errors.append(ep_idx)
    
    # Generate summary report
    logger.info("="*80)
    logger.info("ANOMALY DETECTION RESULTS")
    logger.info("="*80)
    logger.info(f"Total episodes processed: {stats['total_episodes']}")
    logger.info(f"Episodes with errors: {stats['episodes_with_errors']}")
    logger.info(f"Episodes with warnings: {stats['episodes_with_warnings']}")
    logger.info(f"Total errors: {stats['total_errors']}")
    logger.info(f"Total warnings: {stats['total_warnings']}")
    
    # Display error episodes
    if episodes_with_errors:
        logger.error(f"\n❌ Episodes with ERRORS:")
        logger.error(f"  {sorted(episodes_with_errors)}")
    else:
        logger.info("\n✅ No episodes with errors!")
    
    # Display warning episodes
    if episodes_with_warnings:
        logger.warning(f"\n⚠️  Episodes with WARNINGS:")
        logger.warning(f"  {sorted(episodes_with_warnings)}")
    else:
        logger.info("✅ No episodes with warnings!")
    
    # Save detailed report
    if config.get('output', {}).get('save_report', True):
        output_config = config.get('output', {})
        report_dir = Path(output_config.get('report_dir', 'logs/anomaly_reports'))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"anomaly_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'config_path': str(config_path),
            'statistics': stats,
            'episodes_with_errors': sorted(episodes_with_errors),
            'episodes_with_warnings': sorted(episodes_with_warnings),
            'detailed_anomalies': all_anomalies
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Detailed report saved to: {report_path}")
    
    logger.info("="*80)
    logger.info("Anomaly detection completed")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for anomaly detection script."""
    parser = argparse.ArgumentParser(
        description="Detect anomalies in HDF5 episode data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python hdf5_anomaly_detector.py

  # Use custom config
  python hdf5_anomaly_detector.py --config configs/anomaly_config/custom.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/anomaly_config/default.yaml',
        help='Path to anomaly detection configuration file'
    )
    
    args = parser.parse_args()
    
    # Process dataset
    process_dataset(args.config)


if __name__ == "__main__":
    main()
