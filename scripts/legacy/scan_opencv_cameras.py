#!/usr/bin/env python3
"""
OpenCV Camera Scanner - Quick scan for available cameras.

Author: Jun Lv
"""

import cv2
import argparse
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_config import logger


def scan_cameras(max_cameras: int = 20, verbose: bool = False) -> List[Dict]:
    """
    Quick scan for available OpenCV cameras.
    
    Args:
        max_cameras: Maximum number of camera IDs to check
        verbose: Whether to show scanning progress
        
    Returns:
        List of available camera information dictionaries
    """
    available_cameras = []
    
    logger.info(f"Scanning cameras (IDs 0-{max_cameras-1})...")
    if verbose:
        logger.info("-" * 40)
    
    for camera_id in range(max_cameras):
        if verbose:
            logger.info(f"Camera {camera_id}... ", end="", flush=True)
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get basic info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        'id': camera_id,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'frame_shape': frame.shape
                    }
                    
                    available_cameras.append(camera_info)
                    
                    if verbose:
                        logger.info(f"✓ {width}x{height} @ {fps:.1f}fps")
                    else:
                        logger.info(f"  Camera {camera_id}: {width}x{height} @ {fps:.1f}fps")
                else:
                    if verbose:
                        logger.warning("No frame")
            else:
                if verbose:
                    logger.warning("Not available")
            
            cap.release()
            
        except Exception as e:
            if verbose:
                logger.error(f"Error: {e}")
            try:
                cap.release()
            except:
                pass
    
    return available_cameras


def print_camera_summary(cameras: List[Dict[str, Any]]) -> None:
    """Print a summary of all found cameras."""
    if not cameras:
        logger.warning("No cameras found!")
        return
    
    logger.info(f"\nFound {len(cameras)} camera(s):")
    logger.info("=" * 50)
    
    for cam in cameras:
        logger.info(f"Camera {cam['id']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
        logger.info(f"Frame shape: {cam['frame_shape']}")
        logger.info("")

def main() -> None:
    """Main function to run the camera scanner."""
    parser = argparse.ArgumentParser(description="Quick scan for OpenCV cameras")
    parser.add_argument("--max-cameras", "-m", type=int, default=10,
                        help="Maximum camera IDs to check (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show scanning progress")
    
    args = parser.parse_args()
    
    logger.info("OpenCV Camera Scanner")
    logger.info("====================")
    
    # Scan cameras
    cameras = scan_cameras(
        max_cameras=args.max_cameras,
        verbose=args.verbose
    )
    
    # Print results
    print_camera_summary(cameras)
    
    logger.info(f"Found {len(cameras)} camera(s)")


if __name__ == "__main__":
    main() 