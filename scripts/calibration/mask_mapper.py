#!/usr/bin/env python3
"""
Mask Mapper and Coverage Checker

This script maps masks from 640x480 resolution back to original image dimensions
and visualizes the coverage to ensure proper alignment with the original images.

Features:
- Load masks from 640x480 resolution
- Map masks to original image dimensions
- Visualize coverage on original images
- Save mapped masks in original resolution
- Check mask coverage quality

Usage:
    python scripts/calibration/mask_mapper.py --mask_path masks/robot/left_mask.png --original_image_path robot_data/episode_0001.hdf5 --camera_key OpenCVCameraDevice_1 --output_dir masks/robot/
    python mask_mapper.py --mask_path masks/mask_left_combined_frame_0000_20250101_120000.png --original_image_path saved_data/episode_0000.hdf5 --camera_key OpenCVCameraDevice_1
    python mask_mapper.py --mask_path masks/mask_left_combined_frame_0000_20250101_120000.png --original_image_path saved_data/episode_0000.hdf5 --camera_key OpenCVCameraDevice_1 --frame_idx 0

Author: Assistant
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
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


class MaskMapper:
    """Mapper for scaling masks from 640x480 to original image dimensions."""
    
    def __init__(self, mask_path: Path, original_image_path: Path, camera_key: str, 
                 frame_idx: int = 0, output_dir: Optional[Path] = None):
        """
        Initialize the mask mapper.
        
        Args:
            mask_path: Path to the 640x480 mask file
            original_image_path: Path to the original HDF5 file with images
            camera_key: Key for camera data in HDF5 file
            frame_idx: Frame index to use for original image
            output_dir: Directory to save mapped masks (default: masks/mapped)
        """
        self.mask_path = Path(mask_path)
        self.original_image_path = Path(original_image_path)
        self.camera_key = camera_key
        self.frame_idx = frame_idx
        self.output_dir = output_dir or Path("masks/mapped")
        
        # Original image dimensions
        self.original_height = None
        self.original_width = None
        self.original_image = None
        
        # Mask data
        self.mask_640x480 = None
        self.mapped_mask = None
        
        # Coverage analysis
        self.coverage_stats = {}
        
        logger.info(f"Initializing mask mapper")
        logger.info(f"Mask path: {self.mask_path}")
        logger.info(f"Original image path: {self.original_image_path}")
        logger.info(f"Camera key: {self.camera_key}")
        logger.info(f"Frame index: {self.frame_idx}")
    
    def load_original_image(self) -> bool:
        """
        Load original image from HDF5 file to get dimensions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.original_image_path.exists():
                logger.error(f"Original image file not found: {self.original_image_path}")
                return False
            
            logger.info(f"Loading original image from {self.original_image_path}")
            
            with h5py.File(self.original_image_path, 'r') as h5_file:
                # Check if camera data exists
                if self.camera_key not in h5_file:
                    available_keys = list(h5_file.keys())
                    logger.error(f"Camera data '{self.camera_key}' not found. Available keys: {available_keys}")
                    return False
                
                # Load the specific frame
                camera_data = h5_file[self.camera_key]
                if self.frame_idx >= camera_data.shape[0]:
                    logger.error(f"Frame index {self.frame_idx} out of range. Available frames: {camera_data.shape[0]}")
                    return False
                
                # Load image
                img = camera_data[self.frame_idx]
                
                # Ensure image is in correct format
                if len(img.shape) == 3:
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    self.original_image = img
                    self.original_height, self.original_width = img.shape[:2]
                    
                    logger.info(f"Original image dimensions: {self.original_width}x{self.original_height}")
                    logger.info(f"Original image shape: {img.shape}")
                    
                    return True
                else:
                    logger.error(f"Unexpected image shape: {img.shape}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error loading original image: {e}")
            return False
    
    def load_mask(self) -> bool:
        """
        Load the 640x480 mask.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.mask_path.exists():
                logger.error(f"Mask file not found: {self.mask_path}")
                return False
            
            logger.info(f"Loading mask from {self.mask_path}")
            
            # Load mask
            mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.error(f"Failed to load mask from {self.mask_path}")
                return False
            
            # Check if mask is 640x480
            if mask.shape != (480, 640):
                logger.warning(f"Mask shape {mask.shape} is not 640x480. Expected (480, 640)")
                # Still proceed but warn user
            
            # Normalize mask to 0-1 range
            if mask.max() > 1:
                mask = mask / 255.0
            
            self.mask_640x480 = mask.astype(np.uint8)
            
            logger.info(f"Loaded mask with shape: {self.mask_640x480.shape}")
            logger.info(f"Mask value range: {self.mask_640x480.min()} - {self.mask_640x480.max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading mask: {e}")
            return False
    
    def map_mask_to_original(self) -> bool:
        """
        Map the 640x480 mask to original image dimensions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.mask_640x480 is None:
                logger.error("No mask loaded")
                return False
            
            if self.original_height is None or self.original_width is None:
                logger.error("Original image dimensions not available")
                return False
            
            logger.info(f"Mapping mask from 640x480 to {self.original_width}x{self.original_height}")
            
            # Resize mask to original dimensions
            self.mapped_mask = cv2.resize(
                self.mask_640x480, 
                (self.original_width, self.original_height), 
                interpolation=cv2.INTER_NEAREST  # Use nearest neighbor to preserve binary nature
            )
            
            logger.info(f"Mapped mask shape: {self.mapped_mask.shape}")
            logger.info(f"Mapped mask value range: {self.mapped_mask.min()} - {self.mapped_mask.max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error mapping mask: {e}")
            return False
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """
        Analyze mask coverage on the original image.
        
        Returns:
            Dictionary with coverage statistics
        """
        if self.mapped_mask is None or self.original_image is None:
            logger.error("Mapped mask or original image not available")
            return {}
        
        try:
            # Calculate coverage statistics
            total_pixels = self.mapped_mask.size
            masked_pixels = np.sum(self.mapped_mask > 0)
            coverage_percentage = (masked_pixels / total_pixels) * 100
            
            # Calculate mask area in original coordinates
            mask_area = np.sum(self.mapped_mask)
            
            # Calculate bounding box of mask
            coords = np.where(self.mapped_mask > 0)
            if len(coords[0]) > 0:
                min_y, max_y = coords[0].min(), coords[0].max()
                min_x, max_x = coords[1].min(), coords[1].max()
                bbox_width = max_x - min_x + 1
                bbox_height = max_y - min_y + 1
                bbox_area = bbox_width * bbox_height
            else:
                min_x = min_y = max_x = max_y = 0
                bbox_width = bbox_height = bbox_area = 0
            
            self.coverage_stats = {
                'total_pixels': total_pixels,
                'masked_pixels': masked_pixels,
                'coverage_percentage': coverage_percentage,
                'mask_area': mask_area,
                'bbox': {
                    'min_x': min_x,
                    'min_y': min_y,
                    'max_x': max_x,
                    'max_y': max_y,
                    'width': bbox_width,
                    'height': bbox_height,
                    'area': bbox_area
                },
                'original_dimensions': (self.original_width, self.original_height),
                'mask_dimensions': self.mapped_mask.shape
            }
            
            logger.info("Coverage analysis completed:")
            logger.info(f"  Total pixels: {total_pixels}")
            logger.info(f"  Masked pixels: {masked_pixels}")
            logger.info(f"  Coverage: {coverage_percentage:.2f}%")
            logger.info(f"  Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
            logger.info(f"  Bounding box size: {bbox_width}x{bbox_height}")
            
            return self.coverage_stats
            
        except Exception as e:
            logger.error(f"Error analyzing coverage: {e}")
            return {}
    
    def visualize_coverage(self, show_overlay: bool = True, save_visualization: bool = True) -> bool:
        """
        Visualize mask coverage on original image.
        
        Args:
            show_overlay: Whether to show overlay visualization
            save_visualization: Whether to save visualization images
            
        Returns:
            True if successful, False otherwise
        """
        if self.mapped_mask is None or self.original_image is None:
            logger.error("Mapped mask or original image not available")
            return False
        
        try:
            # Create overlay visualization
            if show_overlay:
                # Create colored overlay
                overlay = np.zeros_like(self.original_image)
                overlay[self.mapped_mask > 0] = [0, 255, 0]  # Green for masked areas
                
                # Blend with original image
                alpha = 0.3
                blended = cv2.addWeighted(self.original_image, 1 - alpha, overlay, alpha, 0)
                
                # Add coverage information text
                coverage_text = f"Coverage: {self.coverage_stats.get('coverage_percentage', 0):.2f}%"
                cv2.putText(blended, coverage_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show visualization
                cv2.namedWindow("Mask Coverage Visualization", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Mask Coverage Visualization", 800, 600)
                cv2.imshow("Mask Coverage Visualization", blended)
                
                logger.info("Press any key to close visualization window")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Save visualization if requested
            if save_visualization:
                self._save_visualization()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing coverage: {e}")
            return False
    
    def _save_visualization(self):
        """Save visualization images."""
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save mapped mask
            mapped_mask_filename = f"mapped_mask_frame_{self.frame_idx:04d}_{timestamp}.png"
            mapped_mask_path = self.output_dir / mapped_mask_filename
            cv2.imwrite(str(mapped_mask_path), self.mapped_mask * 255)
            logger.info(f"Mapped mask saved to {mapped_mask_path}")
            
            # Save binary mapped mask
            binary_mask_filename = f"mapped_mask_binary_frame_{self.frame_idx:04d}_{timestamp}.png"
            binary_mask_path = self.output_dir / binary_mask_filename
            cv2.imwrite(str(binary_mask_path), self.mapped_mask)
            logger.info(f"Binary mapped mask saved to {binary_mask_path}")
            
            # Save overlay visualization
            overlay = np.zeros_like(self.original_image)
            overlay[self.mapped_mask > 0] = [0, 255, 0]  # Green for masked areas
            alpha = 0.3
            blended = cv2.addWeighted(self.original_image, 1 - alpha, overlay, alpha, 0)
            
            # Add coverage information
            coverage_text = f"Coverage: {self.coverage_stats.get('coverage_percentage', 0):.2f}%"
            cv2.putText(blended, coverage_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            overlay_filename = f"coverage_visualization_frame_{self.frame_idx:04d}_{timestamp}.png"
            overlay_path = self.output_dir / overlay_filename
            cv2.imwrite(str(overlay_path), blended)
            logger.info(f"Coverage visualization saved to {overlay_path}")
            
            # Save original image for reference
            original_filename = f"original_image_frame_{self.frame_idx:04d}_{timestamp}.png"
            original_path = self.output_dir / original_filename
            cv2.imwrite(str(original_path), self.original_image)
            logger.info(f"Original image saved to {original_path}")
            
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
    
    def run(self) -> bool:
        """
        Run the complete mask mapping and analysis pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting mask mapping and coverage analysis")
        
        # Load original image
        if not self.load_original_image():
            logger.error("Failed to load original image")
            return False
        
        # Load mask
        if not self.load_mask():
            logger.error("Failed to load mask")
            return False
        
        # Map mask to original dimensions
        if not self.map_mask_to_original():
            logger.error("Failed to map mask to original dimensions")
            return False
        
        # Analyze coverage
        coverage_stats = self.analyze_coverage()
        if not coverage_stats:
            logger.error("Failed to analyze coverage")
            return False
        
        # Visualize coverage
        if not self.visualize_coverage(show_overlay=True, save_visualization=True):
            logger.error("Failed to visualize coverage")
            return False
        
        logger.info("Mask mapping and coverage analysis completed successfully")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Map masks from 640x480 to original image dimensions and check coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_mapper.py --mask_path masks/mask_left_combined_frame_0000_20250101_120000.png --original_image_path saved_data/episode_0000.hdf5 --camera_key OpenCVCameraDevice_1
  python mask_mapper.py --mask_path masks/mask_left_combined_frame_0000_20250101_120000.png --original_image_path saved_data/episode_0000.hdf5 --camera_key OpenCVCameraDevice_1 --frame_idx 0 --output_dir masks/mapped
        """
    )
    
    parser.add_argument(
        "--mask_path",
        type=Path,
        required=True,
        help="Path to the 640x480 mask file"
    )
    
    parser.add_argument(
        "--original_image_path",
        type=Path,
        required=True,
        help="Path to the original HDF5 file with images"
    )
    
    parser.add_argument(
        "--camera_key",
        type=str,
        required=True,
        help="Key for camera data in HDF5 file"
    )
    
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index to use for original image (default: 0)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save mapped masks (default: masks/mapped)"
    )
    
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Skip interactive visualization"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.mask_path.exists():
        logger.error(f"Mask file not found: {args.mask_path}")
        sys.exit(1)
    
    if not args.original_image_path.exists():
        logger.error(f"Original image file not found: {args.original_image_path}")
        sys.exit(1)
    
    # Create and run mapper
    mapper = MaskMapper(
        mask_path=args.mask_path,
        original_image_path=args.original_image_path,
        camera_key=args.camera_key,
        frame_idx=args.frame_idx,
        output_dir=args.output_dir
    )
    
    success = mapper.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()