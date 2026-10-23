#!/usr/bin/env python3
"""
Script to inspect the structure and content of HDF5 episode data files.
This helps understand the data format before creating visualization tools.
"""

import h5py
import numpy as np
from typing import Dict, Any, List
import argparse
import os
from utils.logger_config import logger


def print_hdf5_structure(name: str, obj: h5py.Group) -> None:
    """Recursively print HDF5 group/dataset structure."""
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        logger.info(f"{indent}{name}: {obj.shape} {obj.dtype}")
        # Show some sample data for small arrays
        if obj.size < 20 and obj.dtype.kind in ['i', 'f']:
            logger.info(f"{indent}  Sample: {obj[...]}")
        elif obj.dtype.kind == 'S':  # String data
            logger.info(f"{indent}  Sample: {obj[...]}")
    else:
        logger.info(f"{indent}{name}/ (Group)")


def analyze_episode_data(file_path: str) -> Dict[str, Any]:
    """Analyze episode data and return summary information."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {}
    
    logger.info(f"Analyzing episode data: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            logger.info("=== HDF5 File Structure ===")
            f.visititems(print_hdf5_structure)
            
            # Extract key information
            info = {}
            
            # Check for common keys
            common_keys = ['observations', 'actions', 'rewards', 'dones', 'infos']
            for key in common_keys:
                if key in f:
                    info[key] = {
                        'shape': f[key].shape if hasattr(f[key], 'shape') else 'Group',
                        'dtype': f[key].dtype if hasattr(f[key], 'dtype') else 'Group'
                    }
            
            # Check for image data
            def find_images(name: str, obj: h5py.Dataset) -> None:
                if isinstance(obj, h5py.Dataset):
                    # Look for image-like data (3D or 4D arrays with reasonable dimensions)
                    if len(obj.shape) >= 3:
                        if obj.shape[-1] == 3 or obj.shape[-1] == 1:  # RGB or grayscale
                            info.setdefault('images', {})[name] = {
                                'shape': obj.shape,
                                'dtype': obj.dtype,
                                'min_val': np.min(obj[0]) if obj.size > 0 else None,
                                'max_val': np.max(obj[0]) if obj.size > 0 else None
                            }
            
            f.visititems(find_images)
            
            # Check for trajectory data
            def find_trajectories(name: str, obj: h5py.Dataset) -> None:
                if isinstance(obj, h5py.Dataset):
                    # Look for trajectory-like data (position, orientation, etc.)
                    if 'pos' in name.lower() or 'position' in name.lower():
                        info.setdefault('trajectories', {})[name] = {
                            'shape': obj.shape,
                            'dtype': obj.dtype
                        }
                    elif 'quat' in name.lower() or 'orientation' in name.lower():
                        info.setdefault('trajectories', {})[name] = {
                            'shape': obj.shape,
                            'dtype': obj.dtype
                        }
                    elif 'action' in name.lower():
                        info.setdefault('actions_detail', {})[name] = {
                            'shape': obj.shape,
                            'dtype': obj.dtype
                        }
            
            f.visititems(find_trajectories)
            
            logger.info("\n=== Data Summary ===")
            for key, value in info.items():
                logger.info(f"{key}: {value}")
            
            return info
            
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 episode data structure")
    parser.add_argument("file_path", help="Path to HDF5 episode file")
    args = parser.parse_args()
    
    analyze_episode_data(args.file_path)


if __name__ == "__main__":
    main() 