#!/usr/bin/env python3
"""
Mask Offset Analyzer

This script compares masks from two different data collection methods (UMI data collection 
vs robot hand deployment) to calculate the 2D camera offset from robot camera data to UMI data.

The calculated offset represents how much the robot camera position needs to be adjusted
to align with the UMI camera position.

Features:
- Load and compare two mask images from different collection methods
- Calculate 2D offset from robot to UMI using various alignment algorithms (centroid, template matching, feature matching)
- Visualize the offset and alignment results
- Save offset data and analysis results
- Support for different mask formats and resolutions

Usage:
    python scripts/calibration/mask_offset_analyzer.py --umi_mask_path masks/umi/mapped/right_mask.png --robot_mask_path masks/robot/mapped/right_mask.png --output_dir mask_offset/right
    python mask_offset_analyzer.py --umi_mask_path masks/umi_left_mask.png --robot_mask_path masks/robot_left_mask.png
    python mask_offset_analyzer.py --umi_mask_path masks/umi_left_mask.png --robot_mask_path masks/robot_left_mask.png --method template_matching
    python mask_offset_analyzer.py --umi_mask_path masks/umi_left_mask.png --robot_mask_path masks/robot_left_mask.png --output_dir offset_analysis

Author: Assistant
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import json
import time
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error

# Get the project root directory (parent of scripts directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from utils.logger_config import logger


class MaskOffsetAnalyzer:
    """Analyzer for calculating 2D offset between masks from different collection methods."""
    
    def __init__(self, umi_mask_path: Path, robot_mask_path: Path, 
                 output_dir: Optional[Path] = None, method: str = "centroid"):
        """
        Initialize the mask offset analyzer.
        
        Args:
            umi_mask_path: Path to the UMI data collection mask
            robot_mask_path: Path to the robot hand deployment mask
            output_dir: Directory to save analysis results (default: offset_analysis)
            method: Alignment method ("centroid", "template_matching", "feature_matching", "all")
        """
        self.umi_mask_path = Path(umi_mask_path)
        self.robot_mask_path = Path(robot_mask_path)
        self.output_dir = output_dir or Path("offset_analysis")
        self.method = method
        
        # Mask data
        self.umi_mask = None
        self.robot_mask = None
        self.umi_mask_original = None
        self.robot_mask_original = None
        
        # Analysis results
        self.offset_results = {}
        self.alignment_quality = {}
        
        # Supported methods
        self.supported_methods = ["centroid", "template_matching", "feature_matching", "contour_matching"]
        
        logger.info(f"Initializing mask offset analyzer")
        logger.info(f"UMI mask path: {self.umi_mask_path}")
        logger.info(f"Robot mask path: {self.robot_mask_path}")
        logger.info(f"Analysis method: {self.method}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_masks(self) -> bool:
        """
        Load both mask images.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load UMI mask
            if not self.umi_mask_path.exists():
                logger.error(f"UMI mask file not found: {self.umi_mask_path}")
                return False
            
            self.umi_mask_original = cv2.imread(str(self.umi_mask_path), cv2.IMREAD_GRAYSCALE)
            if self.umi_mask_original is None:
                logger.error(f"Failed to load UMI mask from {self.umi_mask_path}")
                return False
            
            # Load robot mask
            if not self.robot_mask_path.exists():
                logger.error(f"Robot mask file not found: {self.robot_mask_path}")
                return False
            
            self.robot_mask_original = cv2.imread(str(self.robot_mask_path), cv2.IMREAD_GRAYSCALE)
            if self.robot_mask_original is None:
                logger.error(f"Failed to load robot mask from {self.robot_mask_path}")
                return False
            
            # Normalize masks to 0-1 range
            self.umi_mask = (self.umi_mask_original > 127).astype(np.uint8)
            self.robot_mask = (self.robot_mask_original > 127).astype(np.uint8)
            
            logger.info(f"UMI mask shape: {self.umi_mask.shape}")
            logger.info(f"Robot mask shape: {self.robot_mask.shape}")
            logger.info(f"UMI mask coverage: {np.sum(self.umi_mask) / self.umi_mask.size * 100:.2f}%")
            logger.info(f"Robot mask coverage: {np.sum(self.robot_mask) / self.robot_mask.size * 100:.2f}%")
            
            # Ensure both masks have the same dimensions
            if self.umi_mask.shape != self.robot_mask.shape:
                logger.warning(f"Mask dimensions don't match. UMI: {self.umi_mask.shape}, Robot: {self.robot_mask.shape}")
                # Resize robot mask to match UMI mask
                self.robot_mask = cv2.resize(self.robot_mask, 
                                           (self.umi_mask.shape[1], self.umi_mask.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                self.robot_mask_original = cv2.resize(self.robot_mask_original, 
                                                    (self.umi_mask.shape[1], self.umi_mask.shape[0]), 
                                                    interpolation=cv2.INTER_NEAREST)
                logger.info(f"Resized robot mask to match UMI mask: {self.robot_mask.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading masks: {e}")
            return False
    
    def calculate_centroid_offset(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calculate offset using centroid-based alignment.
        
        Returns:
            Tuple of (offset_x, offset_y, quality_metrics)
        """
        try:
            # Calculate centroids
            umi_moments = cv2.moments(self.umi_mask)
            robot_moments = cv2.moments(self.robot_mask)
            
            if umi_moments['m00'] == 0 or robot_moments['m00'] == 0:
                logger.error("One or both masks have zero area")
                return 0.0, 0.0, {'error': 'zero_area'}
            
            umi_centroid = (umi_moments['m10'] / umi_moments['m00'], 
                           umi_moments['m01'] / umi_moments['m00'])
            robot_centroid = (robot_moments['m10'] / robot_moments['m00'], 
                             robot_moments['m01'] / robot_moments['m00'])
            
            # Calculate offset (from robot to UMI)
            offset_x = umi_centroid[0] - robot_centroid[0]
            offset_y = umi_centroid[1] - robot_centroid[1]
            
            # Calculate quality metrics
            umi_area = np.sum(self.umi_mask)
            robot_area = np.sum(self.robot_mask)
            area_ratio = min(umi_area, robot_area) / max(umi_area, robot_area)
            
            # Calculate overlap after alignment
            shifted_robot = self._shift_mask(self.robot_mask, offset_x, offset_y)
            intersection = np.sum(self.umi_mask & shifted_robot)
            union = np.sum(self.umi_mask | shifted_robot)
            iou = intersection / union if union > 0 else 0
            
            quality_metrics = {
                'umi_centroid': umi_centroid,
                'robot_centroid': robot_centroid,
                'area_ratio': area_ratio,
                'iou_after_alignment': iou,
                'umi_area': int(umi_area),
                'robot_area': int(robot_area)
            }
            
            logger.info(f"Centroid offset: ({offset_x:.2f}, {offset_y:.2f}) pixels")
            logger.info(f"Area ratio: {area_ratio:.3f}, IoU after alignment: {iou:.3f}")
            
            return offset_x, offset_y, quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating centroid offset: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    def calculate_template_matching_offset(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calculate offset using template matching.
        
        Returns:
            Tuple of (offset_x, offset_y, quality_metrics)
        """
        try:
            # Use the smaller mask as template
            if np.sum(self.umi_mask) < np.sum(self.robot_mask):
                template = self.umi_mask
                image = self.robot_mask
                template_is_umi = True
            else:
                template = self.robot_mask
                image = self.umi_mask
                template_is_umi = False
            
            # Perform template matching
            result = cv2.matchTemplate(image.astype(np.float32), template.astype(np.float32), 
                                     cv2.TM_CCOEFF_NORMED)
            
            # Find best match location
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Calculate template center
            template_center_x = template.shape[1] // 2
            template_center_y = template.shape[0] // 2
            
            # Calculate match center
            match_center_x = max_loc[0] + template_center_x
            match_center_y = max_loc[1] + template_center_y
            
            # Calculate image center
            image_center_x = image.shape[1] // 2
            image_center_y = image.shape[0] // 2
            
            # Calculate offset (from robot to UMI)
            if template_is_umi:
                # UMI is template, robot is image
                offset_x = image_center_x - match_center_x
                offset_y = image_center_y - match_center_y
            else:
                # Robot is template, UMI is image
                offset_x = match_center_x - image_center_x
                offset_y = match_center_y - image_center_y
            
            quality_metrics = {
                'match_confidence': float(max_val),
                'match_location': max_loc,
                'template_is_umi': template_is_umi,
                'template_size': template.shape,
                'image_size': image.shape
            }
            
            logger.info(f"Template matching offset: ({offset_x:.2f}, {offset_y:.2f}) pixels")
            logger.info(f"Match confidence: {max_val:.3f}")
            
            return offset_x, offset_y, quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating template matching offset: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    def calculate_contour_matching_offset(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calculate offset using contour-based alignment.
        
        Returns:
            Tuple of (offset_x, offset_y, quality_metrics)
        """
        try:
            # Find contours
            umi_contours, _ = cv2.findContours(self.umi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            robot_contours, _ = cv2.findContours(self.robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not umi_contours or not robot_contours:
                logger.error("No contours found in one or both masks")
                return 0.0, 0.0, {'error': 'no_contours'}
            
            # Get largest contours
            umi_contour = max(umi_contours, key=cv2.contourArea)
            robot_contour = max(robot_contours, key=cv2.contourArea)
            
            # Calculate contour moments and centroids
            umi_M = cv2.moments(umi_contour)
            robot_M = cv2.moments(robot_contour)
            
            if umi_M['m00'] == 0 or robot_M['m00'] == 0:
                logger.error("Contour has zero area")
                return 0.0, 0.0, {'error': 'zero_contour_area'}
            
            umi_centroid = (umi_M['m10'] / umi_M['m00'], umi_M['m01'] / umi_M['m00'])
            robot_centroid = (robot_M['m10'] / robot_M['m00'], robot_M['m01'] / robot_M['m00'])
            
            # Calculate offset (from robot to UMI)
            offset_x = umi_centroid[0] - robot_centroid[0]
            offset_y = umi_centroid[1] - robot_centroid[1]
            
            # Calculate shape similarity using Hu moments
            umi_hu = cv2.HuMoments(umi_M).flatten()
            robot_hu = cv2.HuMoments(robot_M).flatten()
            
            # Calculate contour match using matchShapes
            shape_match = cv2.matchShapes(umi_contour, robot_contour, cv2.CONTOURS_MATCH_I1, 0)
            
            quality_metrics = {
                'umi_contour_area': float(cv2.contourArea(umi_contour)),
                'robot_contour_area': float(cv2.contourArea(robot_contour)),
                'shape_match_score': float(shape_match),
                'umi_centroid': umi_centroid,
                'robot_centroid': robot_centroid,
                'hu_moments_similarity': float(np.mean(np.abs(umi_hu - robot_hu)))
            }
            
            logger.info(f"Contour matching offset: ({offset_x:.2f}, {offset_y:.2f}) pixels")
            logger.info(f"Shape match score: {shape_match:.3f} (lower is better)")
            
            return offset_x, offset_y, quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating contour matching offset: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    def calculate_feature_matching_offset(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calculate offset using feature-based matching (ORB features).
        
        Returns:
            Tuple of (offset_x, offset_y, quality_metrics)
        """
        try:
            # Convert masks to 8-bit images for feature detection
            umi_img = (self.umi_mask * 255).astype(np.uint8)
            robot_img = (self.robot_mask * 255).astype(np.uint8)
            
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=1000)
            
            # Detect keypoints and descriptors
            umi_kp, umi_desc = orb.detectAndCompute(umi_img, None)
            robot_kp, robot_desc = orb.detectAndCompute(robot_img, None)
            
            if umi_desc is None or robot_desc is None or len(umi_desc) < 4 or len(robot_desc) < 4:
                logger.warning("Insufficient features detected for matching")
                return 0.0, 0.0, {'error': 'insufficient_features'}
            
            # Match features using BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(umi_desc, robot_desc)
            
            if len(matches) < 4:
                logger.warning("Insufficient matches for reliable offset calculation")
                return 0.0, 0.0, {'error': 'insufficient_matches'}
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Extract matched points
            umi_pts = np.float32([umi_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            robot_pts = np.float32([robot_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Calculate average offset from matched points (from robot to UMI)
            offsets = umi_pts.reshape(-1, 2) - robot_pts.reshape(-1, 2)
            
            # Use RANSAC to find robust offset estimate
            if len(offsets) >= 10:
                # Use top matches for robust estimation
                top_matches = min(20, len(offsets))
                robust_offsets = offsets[:top_matches]
                
                # Remove outliers using median absolute deviation
                median_offset = np.median(robust_offsets, axis=0)
                mad = np.median(np.abs(robust_offsets - median_offset), axis=0)
                threshold = 2.0 * mad
                
                inliers = np.all(np.abs(robust_offsets - median_offset) <= threshold, axis=1)
                if np.sum(inliers) > 0:
                    final_offset = np.mean(robust_offsets[inliers], axis=0)
                else:
                    final_offset = median_offset
            else:
                final_offset = np.mean(offsets, axis=0)
            
            offset_x, offset_y = final_offset
            
            quality_metrics = {
                'num_features_umi': len(umi_kp),
                'num_features_robot': len(robot_kp),
                'num_matches': len(matches),
                'avg_match_distance': float(np.mean([m.distance for m in matches])),
                'offset_std': [float(np.std(offsets[:, 0])), float(np.std(offsets[:, 1]))],
                'robust_inliers': int(np.sum(inliers)) if len(offsets) >= 10 else len(offsets)
            }
            
            logger.info(f"Feature matching offset: ({offset_x:.2f}, {offset_y:.2f}) pixels")
            logger.info(f"Found {len(matches)} matches from {len(umi_kp)} and {len(robot_kp)} features")
            
            return offset_x, offset_y, quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating feature matching offset: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    def _shift_mask(self, mask: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
        """
        Shift a mask by given offset.
        
        Args:
            mask: Input mask
            shift_x: Horizontal shift in pixels
            shift_y: Vertical shift in pixels
            
        Returns:
            Shifted mask
        """
        rows, cols = mask.shape
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
        return shifted
    
    def analyze_offset(self) -> bool:
        """
        Perform offset analysis using specified method(s).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            methods_to_run = self.supported_methods if self.method == "all" else [self.method]
            
            for method in methods_to_run:
                if method not in self.supported_methods:
                    logger.warning(f"Unsupported method: {method}")
                    continue
                
                logger.info(f"Running {method} analysis...")
                
                if method == "centroid":
                    offset_x, offset_y, quality = self.calculate_centroid_offset()
                elif method == "template_matching":
                    offset_x, offset_y, quality = self.calculate_template_matching_offset()
                elif method == "feature_matching":
                    offset_x, offset_y, quality = self.calculate_feature_matching_offset()
                elif method == "contour_matching":
                    offset_x, offset_y, quality = self.calculate_contour_matching_offset()
                
                self.offset_results[method] = {
                    'offset_x': float(offset_x),
                    'offset_y': float(offset_y),
                    'offset_magnitude': float(np.sqrt(offset_x**2 + offset_y**2)),
                    'offset_angle_degrees': float(np.degrees(np.arctan2(offset_y, offset_x)))
                }
                
                self.alignment_quality[method] = quality
            
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing offset: {e}")
            return False
    
    def visualize_results(self, show_visualization: bool = True, save_visualization: bool = True) -> bool:
        """
        Visualize offset analysis results.
        
        Args:
            show_visualization: Whether to show interactive visualization
            save_visualization: Whether to save visualization images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory
            if save_visualization:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            for method, result in self.offset_results.items():
                offset_x = result['offset_x']
                offset_y = result['offset_y']
                
                # Create visualization
                vis_img = self._create_offset_visualization(method, offset_x, offset_y)
                
                if show_visualization:
                    cv2.namedWindow(f"Offset Analysis - {method}", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(f"Offset Analysis - {method}", 1600, 900)
                    cv2.imshow(f"Offset Analysis - {method}", vis_img)
                
                if save_visualization:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"offset_analysis_{method}_{timestamp}.png"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), vis_img)
                    logger.info(f"Visualization saved: {filepath}")
            
            if show_visualization:
                logger.info("Press any key to close visualization windows")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
            return False
    
    def _create_offset_visualization(self, method: str, offset_x: float, offset_y: float) -> np.ndarray:
        """
        Create offset visualization image.
        
        Args:
            method: Analysis method name
            offset_x: X offset in pixels
            offset_y: Y offset in pixels
            
        Returns:
            Visualization image
        """
        # Create base visualization
        height, width = self.umi_mask.shape
        vis_width = width * 4  # Four panels side by side
        vis_height = height + 150  # Extra space for text
        
        vis_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Panel 1: UMI mask (green)
        umi_colored = np.zeros((height, width, 3), dtype=np.uint8)
        umi_colored[self.umi_mask > 0] = [0, 255, 0]  # Green
        vis_img[80:80+height, 0:width] = umi_colored
        
        # Panel 2: Robot mask (red)
        robot_colored = np.zeros((height, width, 3), dtype=np.uint8)
        robot_colored[self.robot_mask > 0] = [0, 0, 255]  # Red
        vis_img[80:80+height, width:2*width] = robot_colored
        
        # Panel 3: Direct overlay (before alignment)
        direct_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        direct_overlay[self.umi_mask > 0] = [0, 255, 0]  # UMI in green
        direct_overlay[self.robot_mask > 0] = [0, 0, 255]  # Robot in red
        
        # Show direct overlap in yellow
        direct_overlap = (self.umi_mask > 0) & (self.robot_mask > 0)
        direct_overlay[direct_overlap] = [0, 255, 255]  # Direct overlap in yellow
        
        vis_img[80:80+height, 2*width:3*width] = direct_overlay
        
        # Panel 4: Aligned overlay (after offset correction)
        aligned_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        aligned_overlay[self.umi_mask > 0] = [0, 255, 0]  # UMI in green
        
        # Shift robot mask and overlay
        shifted_robot = self._shift_mask(self.robot_mask, offset_x, offset_y)
        aligned_overlay[shifted_robot > 0] = [0, 0, 255]  # Robot in red
        
        # Show aligned overlap in white
        aligned_overlap = (self.umi_mask > 0) & (shifted_robot > 0)
        aligned_overlay[aligned_overlap] = [255, 255, 255]  # Aligned overlap in white
        
        vis_img[80:80+height, 3*width:4*width] = aligned_overlay
        
        # Add labels with larger font
        font_scale = 1.2
        font_thickness = 3
        cv2.putText(vis_img, "UMI Mask", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        cv2.putText(vis_img, "Robot Mask", (width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
        cv2.putText(vis_img, "Direct Overlay", (2*width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(vis_img, "Aligned Overlay", (3*width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Add color legend
        legend_y = 25
        cv2.putText(vis_img, "Green=UMI", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_img, "Red=Robot", (width + 10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis_img, "Yellow=Direct Overlap", (2*width + 10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis_img, "White=Aligned Overlap", (3*width + 10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add offset information with larger font
        info_y1 = height + 100
        info_y2 = height + 130
        font_scale_info = 1.0
        font_thickness_info = 2
        
        cv2.putText(vis_img, f"Method: {method.upper()}", (10, info_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, (255, 255, 0), font_thickness_info)
        cv2.putText(vis_img, f"Offset: ({offset_x:.2f}, {offset_y:.2f}) pixels", (400, info_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, (255, 255, 0), font_thickness_info)
        
        magnitude = np.sqrt(offset_x**2 + offset_y**2)
        angle = np.degrees(np.arctan2(offset_y, offset_x))
        cv2.putText(vis_img, f"Magnitude: {magnitude:.2f}px", (10, info_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, (255, 255, 0), font_thickness_info)
        cv2.putText(vis_img, f"Direction: {angle:.1f} degrees", (400, info_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, (255, 255, 0), font_thickness_info)
        
        # Draw arrow showing offset direction on the aligned overlay panel
        center_x = 3*width + width//2
        center_y = 80 + height//2
        arrow_scale = min(3.0, max(1.0, magnitude / 10))  # Scale arrow based on offset magnitude
        arrow_end_x = int(center_x + offset_x * arrow_scale)
        arrow_end_y = int(center_y + offset_y * arrow_scale)
        
        # Draw arrow with thicker line
        cv2.arrowedLine(vis_img, (center_x, center_y), (arrow_end_x, arrow_end_y), (0, 255, 255), 4, tipLength=0.3)
        
        # Add arrow label
        arrow_label_x = center_x + 20
        arrow_label_y = center_y - 20
        cv2.putText(vis_img, "Robot->UMI", (arrow_label_x, arrow_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return vis_img
    
    def save_results(self) -> bool:
        """
        Save analysis results to JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare results data
            results_data = {
                'analysis_info': {
                    'umi_mask_path': str(self.umi_mask_path),
                    'robot_mask_path': str(self.robot_mask_path),
                    'analysis_method': self.method,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'mask_dimensions': self.umi_mask.shape
                },
                'offset_results': self.offset_results,
                'alignment_quality': self.alignment_quality
            }
            
            # Save to JSON file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"mask_offset_analysis_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis results saved to {filepath}")
            
            # Also save a summary text file
            summary_filename = f"mask_offset_summary_{timestamp}.txt"
            summary_filepath = self.output_dir / summary_filename
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("Mask Offset Analysis Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"UMI Mask: {self.umi_mask_path}\n")
                f.write(f"Robot Mask: {self.robot_mask_path}\n")
                f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for method, result in self.offset_results.items():
                    f.write(f"{method.upper()} METHOD RESULTS:\n")
                    f.write(f"  Offset X: {result['offset_x']:.3f} pixels\n")
                    f.write(f"  Offset Y: {result['offset_y']:.3f} pixels\n")
                    f.write(f"  Magnitude: {result['offset_magnitude']:.3f} pixels\n")
                    f.write(f"  Angle: {result['offset_angle_degrees']:.1f} degrees\n\n")
            
            logger.info(f"Summary saved to {summary_filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def run(self) -> bool:
        """
        Run the complete mask offset analysis pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting mask offset analysis")
        
        # Load masks
        if not self.load_masks():
            logger.error("Failed to load masks")
            return False
        
        # Analyze offset
        if not self.analyze_offset():
            logger.error("Failed to analyze offset")
            return False
        
        # Visualize results
        if not self.visualize_results(show_visualization=True, save_visualization=True):
            logger.error("Failed to visualize results")
            return False
        
        # Save results
        if not self.save_results():
            logger.error("Failed to save results")
            return False
        
        # Print summary
        self._print_summary()
        
        logger.info("Mask offset analysis completed successfully")
        return True
    
    def _print_summary(self):
        """Print analysis summary to console."""
        print("\n" + "=" * 60)
        print("MASK OFFSET ANALYSIS SUMMARY")
        print("=" * 60)
        
        for method, result in self.offset_results.items():
            print(f"\n{method.upper()} METHOD:")
            print(f"  Offset: ({result['offset_x']:.2f}, {result['offset_y']:.2f}) pixels")
            print(f"  Magnitude: {result['offset_magnitude']:.2f} pixels")
            print(f"  Direction: {result['offset_angle_degrees']:.1f} degrees")
            
            if method in self.alignment_quality:
                quality = self.alignment_quality[method]
                if 'iou_after_alignment' in quality:
                    print(f"  IoU after alignment: {quality['iou_after_alignment']:.3f}")
                if 'match_confidence' in quality:
                    print(f"  Match confidence: {quality['match_confidence']:.3f}")
                if 'num_matches' in quality:
                    print(f"  Feature matches: {quality['num_matches']}")
        
        print("\n" + "=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze 2D offset between masks from different data collection methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_offset_analyzer.py --umi_mask_path masks/umi_left_mask.png --robot_mask_path masks/robot_left_mask.png
  python mask_offset_analyzer.py --umi_mask_path masks/umi_left_mask.png --robot_mask_path masks/robot_left_mask.png --method template_matching
  python mask_offset_analyzer.py --umi_mask_path masks/umi_left_mask.png --robot_mask_path masks/robot_left_mask.png --method all --output_dir offset_analysis
        """
    )
    
    parser.add_argument(
        "--umi_mask_path",
        type=Path,
        required=True,
        help="Path to the UMI data collection mask image"
    )
    
    parser.add_argument(
        "--robot_mask_path",
        type=Path,
        required=True,
        help="Path to the robot hand deployment mask image"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="centroid",
        choices=["centroid", "template_matching", "feature_matching", "contour_matching", "all"],
        help="Offset calculation method (default: centroid)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save analysis results (default: offset_analysis)"
    )
    
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Skip interactive visualization"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.umi_mask_path.exists():
        logger.error(f"UMI mask file not found: {args.umi_mask_path}")
        sys.exit(1)
    
    if not args.robot_mask_path.exists():
        logger.error(f"Robot mask file not found: {args.robot_mask_path}")
        sys.exit(1)
    
    # Create and run analyzer
    analyzer = MaskOffsetAnalyzer(
        umi_mask_path=args.umi_mask_path,
        robot_mask_path=args.robot_mask_path,
        output_dir=args.output_dir,
        method=args.method
    )
    
    success = analyzer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
