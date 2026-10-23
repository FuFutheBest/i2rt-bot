"""
HDF5 Utilities

This module provides shared utility functions for HDF5 episode data processing.
These functions are used by both hdf5_anomaly_detector.py and hdf5_episode_visualizer.py.

Author: Junxie Xu
"""

from typing import Dict, Tuple
import h5py
import numpy as np
from utils.logger_config import logger


def load_episode_with_timestamps(h5_file: h5py.File, feat: Dict[str, str]) -> Tuple[Dict, Dict]:
    """
    Load episode data and timestamps from HDF5 file (async format only).
    
    This function extracts data from HDF5 files organized in async format where
    each device is stored as a group containing 'data' and 'timestamps' datasets.
    
    Args:
        h5_file: Open HDF5 file handle
        feat: Feature configuration dictionary mapping logical names to device keys.
              Example: {'pose_unified': 'ViveTrackerDevice_4', 
                       'img_left': 'CameraDevice_0', 
                       'grip_left': 'RotaryEncoderDevice_0'}
        
    Returns:
        Tuple of (data_dict, timestamp_dict) where:
        - data_dict: Dictionary mapping device_key -> numpy array of data
        - timestamp_dict: Dictionary mapping device_key -> numpy array of timestamps (in seconds)
        
    Example:
        >>> with h5py.File('episode_0000.hdf5', 'r') as h5:
        >>>     feat = {'pose_unified': 'ViveTrackerDevice_4', 'img_left': 'CameraDevice_0'}
        >>>     data_dict, timestamp_dict = load_episode_with_timestamps(h5, feat)
        >>>     pose_data = data_dict['ViveTrackerDevice_4']  # (T, 2, 7) array
        >>>     pose_timestamps = timestamp_dict['ViveTrackerDevice_4']  # (T,) array
    """
    data_dict = {}
    timestamp_dict = {}
    
    # Load async format: groups containing 'data' and 'timestamps'
    for device_key in h5_file.keys():
        if device_key in ['freq', 'timestamp', 'start_time_ns']:
            continue
        
        if isinstance(h5_file[device_key], h5py.Group):
            device_group = h5_file[device_key]
            if 'data' in device_group and 'timestamps' in device_group:
                data_dict[device_key] = device_group['data'][:]
                timestamp_dict[device_key] = device_group['timestamps'][:]
    
    # Also load grip data if specified in feat
    for grip_key in ["grip_left", "grip_right"]:
        if grip_key in feat:
            device_key = feat[grip_key]
            if device_key in h5_file and isinstance(h5_file[device_key], h5py.Group):
                if 'data' in h5_file[device_key] and 'timestamps' in h5_file[device_key]:
                    data_dict[device_key] = h5_file[device_key]['data'][:]
                    timestamp_dict[device_key] = h5_file[device_key]['timestamps'][:]
    
    return data_dict, timestamp_dict




