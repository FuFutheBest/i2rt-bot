#!/usr/bin/env python3
"""
HDF5 Data Visualizer and Zarr Converter

This script supports both:
1. Legacy format: Two separate ViveTrackerDevice with [7] shape each
2. New unified format: Single ViveTrackerDevice with [2,7] shape

The unified format uses:
- ViveTrackerDevice_4 with tracker_roles=['left_foot', 'right_foot'] 
- Data shape: [T, 2, 7] where [:, 0, :] = left_arm, [:, 1, :] = right_arm

Features:
- Complete HDF5 to zarr conversion with ReplayBuffer
- Camera image processing and compression (center crop + resize to 224x224)
- Vive Tracker pose anomaly detection (detects zero poses and jumps > 1cm)
- Comprehensive logging of problematic episodes
- Optional trajectory visualization with rerun
- Support for both legacy and unified tracker formats
- Optional image logging to reduce memory pressure
- Multi-threaded image processing for performance
- Support for mask application during image processing
- Configuration file support for flexible dataset specification


Usage examples:
- Basic conversion: python hdf5_to_zarr_data_converter.py config.yaml output.zarr.zip --enable-zarr-conversion
- With visualization: python hdf5_to_zarr_data_converter.py config.yaml output.zarr.zip --visualize --enable-zarr-conversion
- Without anomaly detection: python hdf5_to_zarr_data_converter.py config.yaml output.zarr.zip --no-anomaly-detection --enable-zarr-conversion
- Without image logging: python hdf5_to_zarr_data_converter.py config.yaml output.zarr.zip --no-image-logging --enable-zarr-conversion

Author: Han Xue
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%

import cv2
import h5py
import numpy as np
import yaml
import zarr
from tqdm import tqdm
import scipy.spatial.transform as st
from concurrent.futures import ThreadPoolExecutor
import rerun as rr
from utils.rerun_visualization import (
    RerunVisualizer, visualize_trajectory_with_colors, 
    visualize_trajectory_with_rotation, gripper_width_to_color, log_text_summary,
    visualize_action_time_series
)
from utils.logger_config import logger

# Import zarr conversion dependencies
from utils.replay_buffer import ReplayBuffer
from utils.imagecodecs_numcodecs import register_codecs, JpegXl

# Register codecs for image compression
register_codecs()


# ---------- helpers ----------------------------------------------------------
def load_offset_from_json(json_path: str) -> tuple:
    """
    Load offset values from JSON calibration file.
    Supports two JSON formats:
    1. Old format: {"offset": {"x": ..., "y": ...}}
    2. New format: {"offset_results": {"centroid": {"offset_x": ..., "offset_y": ...}}}

    Args:
        json_path: Path to the JSON file containing offset information

    Returns:
        tuple: (offset_x, offset_y) offset values in pixels
    """
    import json
    try:
        if not os.path.exists(json_path):
            logger.warning(f"JSON file does not exist: {json_path}")
            return 0.0, 0.0

        with open(json_path, 'r') as f:
            data = json.load(f)

        logger.debug(f"Successfully loaded JSON from: {os.path.basename(json_path)}")
        logger.debug(f"JSON keys: {list(data.keys())}")

        # Try new format first: offset_results.centroid.offset_x/offset_y
        if 'offset_results' in data and 'centroid' in data['offset_results']:
            offset_x = data['offset_results']['centroid']['offset_x']
            offset_y = data['offset_results']['centroid']['offset_y']
            logger.debug(f"Using new format (offset_results.centroid)")
            logger.debug(f"Extracted offset: x={offset_x}, y={offset_y}")
            return offset_x, offset_y

        # Try old format: offset.x/y
        elif 'offset' in data:
            offset_x = data['offset']['x']
            offset_y = data['offset']['y']
            logger.debug(f"Using old format (offset.x/y)")
            logger.debug(f"Extracted offset: x={offset_x}, y={offset_y}")
            return offset_x, offset_y

        else:
            logger.warning(f"No recognized offset format found in JSON.")
            logger.warning(f"Available keys: {list(data.keys())}")
            return 0.0, 0.0

    except Exception as e:
        logger.error(f"Error loading offset from {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0


def apply_offset_to_image(img: np.ndarray, offset_x: float, offset_y: float) -> np.ndarray:
    """
    Apply spatial offset to image using affine transformation.

    Args:
        img: (H, W, C) uint8 image
        offset_x: Horizontal offset in pixels
        offset_y: Vertical offset in pixels

    Returns:
        Transformed image (H, W, C) uint8
    """
    H, W = img.shape[:2]

    # Create translation matrix
    translation_matrix = np.float32([
        [1, 0, offset_x],
        [0, 1, offset_y]
    ])

    # Apply affine transformation
    transformed_img = cv2.warpAffine(img, translation_matrix, (W, H),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
    return transformed_img


def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    quat: (..., 4)  [x, y, z, w]
    return (..., 3) axis-angle (OpenCV convention, norm = angle)
    """
    rot = st.Rotation.from_quat(quat[..., [0, 1, 2, 3]])  # scipy: [x,y,z,w]
    return rot.as_rotvec()


def normalize_gripper_width(gripper_data: np.ndarray, normalization_factor: float) -> np.ndarray:
    """
    Normalize gripper width data using a pre-computed normalization factor.

    Args:
        gripper_data: (T, 1) gripper width data
        normalization_factor: the factor used for normalization (dataset_max / target_max)

    Returns:
        normalized_data: (T, 1) normalized gripper data
    """
    if normalization_factor == 0 or normalization_factor == 1.0:
        return gripper_data.copy()

    # Normalize: divide all values by normalization factor
    normalized_data = gripper_data / normalization_factor

    return normalized_data


def compute_dataset_gripper_normalization_factors(files: List[Path], feats: List[dict], 
                                                  target_max: float = 0.085) -> Tuple[float, float]:
    """
    Compute normalization factors for gripper width data across the entire dataset.

    Args:
        files: List of episode file paths
        feats: List of feature configurations
        target_max: target maximum value for normalization (default: 0.085)

    Returns:
        Tuple of (left_normalization_factor, right_normalization_factor)
    """
    left_max = 0.0
    right_max = 0.0
    
    logger.info("Computing global gripper width normalization factors...")
    
    for ep_path, feat in tqdm(zip(files, feats), total=len(files), desc="Scanning gripper data"):
        try:
            with h5py.File(ep_path, "r") as h5:
                left_grip_key = feat['grip_left']
                right_grip_key = feat['grip_right']
                
                grip_left = h5[left_grip_key][()] / 1000.0  # Convert to meters
                grip_right = h5[right_grip_key][()] / 1000.0
                
                left_max = max(left_max, np.max(grip_left))
                right_max = max(right_max, np.max(grip_right))
        except Exception as e:
            logger.warning(f"Error reading gripper data from {ep_path}: {e}")
            continue
    
    # Calculate normalization factors
    left_norm_factor = left_max / target_max if left_max > 0 else 1.0
    right_norm_factor = right_max / target_max if right_max > 0 else 1.0
    
    logger.info(f"Global gripper width statistics:")
    logger.info(f"  Left gripper max: {left_max:.4f}m, normalization factor: {left_norm_factor:.4f}")
    logger.info(f"  Right gripper max: {right_max:.4f}m, normalization factor: {right_norm_factor:.4f}")
    
    return left_norm_factor, right_norm_factor


def replace_zero_rows_with_last_nonzero(arr):
    """
    arr: (N, D) np.ndarray
    将每一行全为0的行替换为最近一次非零行（前面没有非零则保持为零）
    """
    arr = arr.copy()
    last_nonzero = None
    for i in range(len(arr)):
        if np.all(arr[i] == 0):
            if last_nonzero is not None:
                arr[i] = last_nonzero
        else:
            last_nonzero = arr[i].copy()
    return arr


def apply_mask_to_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image, setting masked regions to black (0).

    Args:
        img: (H, W, C) uint8 image
        mask: (H, W) uint8 mask where 255 means keep, 0 means mask out

    Returns:
        Masked image (H, W, C) uint8
    """
    if mask.ndim == 2:
        # Convert 2D mask to 3D for broadcasting
        mask = mask[:, :, np.newaxis]

    # Apply mask: keep pixels where mask > 0, set to 0 where mask == 0
    masked_img = img * (mask > 0).astype(np.uint8)
    return masked_img


def center_crop_square_resize_with_mask(img: np.ndarray, mask: np.ndarray = None, tgt: int = 224) -> np.ndarray:
    """
    img: (H, W, C)  uint8
    mask: (H, W) uint8 mask (optional)
    1) Apply mask if provided
    2) center-crop to square based on the shorter edge
    3) resize → tgt×tgt
    """
    # Apply mask first if provided
    if mask is not None:
        img = apply_mask_to_image(img, mask)

    H, W, C = img.shape

    # Find the shorter edge and use it as the crop size
    shorter_edge = min(H, W)

    # Calculate crop offsets for both dimensions to center the crop
    h_off = (H - shorter_edge) // 2
    w_off = (W - shorter_edge) // 2

    # Center crop to square
    img = img[h_off:h_off + shorter_edge, w_off:w_off + shorter_edge]

    # 使用OpenCV高质量resize
    img = cv2.resize(img, (tgt, tgt), interpolation=cv2.INTER_AREA)
    return img


def detect_pose_anomalies(pose_data: np.ndarray, position_threshold: float = 0.04,
                         tracker_name: str = "tracker") -> dict:
    """
    Detect anomalies in pose data including zero poses and large jumps.
    
    Args:
        pose_data: (T, 7) array with [x, y, z, qx, qy, qz, qw] format
        position_threshold: threshold in meters for detecting jumps (default: 1cm)
        tracker_name: name for logging purposes
    
    Returns:
        dict with anomaly information: {
            'zero_frames': list of frame indices with all-zero poses,
            'jump_frames': list of (frame_idx, distance) tuples for large jumps,
            'has_anomalies': bool indicating if any anomalies were found
        }
    """
    anomalies = {
        'zero_frames': [],
        'jump_frames': [],
        'has_anomalies': False
    }
    
    if len(pose_data) == 0:
        return anomalies
    
    # Check for all-zero poses (before any processing)
    positions = pose_data[:, :3]  # Extract position data
    
    for i, pos in enumerate(positions):
        if np.allclose(pos, 0.0, atol=1e-6):
            anomalies['zero_frames'].append(i)
    
    # Check for large position jumps between consecutive frames
    if len(positions) > 1:
        position_diffs = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        
        for i, diff in enumerate(position_diffs):
            if diff > position_threshold:
                anomalies['jump_frames'].append((i + 1, diff))  # i+1 because we compare with next frame
    
    # Set has_anomalies flag
    anomalies['has_anomalies'] = len(anomalies['zero_frames']) > 0 or len(anomalies['jump_frames']) > 0
    
    return anomalies


def center_crop_square_resize(img: np.ndarray, tgt: int = 224) -> np.ndarray:
    """
    img: (H, W, C)  uint8
    1) center-crop to square based on the shorter edge
    2) resize → tgt×tgt
    """
    H, W, C = img.shape

    # Find the shorter edge and use it as the crop size
    shorter_edge = min(H, W)

    # Calculate crop offsets for both dimensions to center the crop
    h_off = (H - shorter_edge) // 2
    w_off = (W - shorter_edge) // 2

    # Center crop to square
    img = img[h_off:h_off + shorter_edge, w_off:w_off + shorter_edge]

    # 使用OpenCV高质量resize
    img = cv2.resize(img, (tgt, tgt), interpolation=cv2.INTER_AREA)
    return img



compression_level = 99
# tcp_vive_T = np.array([[0.0000, -0.9397, 0.3420, -0.0317],
#                        [1.0000, 0.0000, 0.0000, 0.0000],
#                        [0.0000, 0.3420, 0.9397, -0.272],
#                        [0.0000, 0.0000, 0.0000, 1.0000]])
tcp_vive_T = np.array([[1.0, 0.0, 0.0, -0.0317],
                       [0.0000, 1.0000, 0.0000, 0.0000],
                       [0.0000, 0.0, 1.0, -0.272],
                       [0.0000, 0.0000, 0.0000, 1.0000]])
vive_tcp_T = np.linalg.inv(tcp_vive_T)

hdf5_info = {}

def extract_episode_files(hdf5_info):
    all_valid_files = []
    all_valid_lines = []
    all_valid_feats = []
    all_valid_masks = []
    for key in hdf5_info:
        hdf5_dir = Path(key)
        files = sorted(hdf5_dir.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
        info = hdf5_info[key]
        if info["lines"] is not None:
            with open(os.path.join(os.path.dirname(__file__), info["lines"]), 'r') as f:
                lines = f.readlines()
            lines = [[int(x) for x in line.strip().split(' ')] for line in lines]
        else:
            lines = [None for _ in range(len(files))]
        
        # Support both 'valid_id' and 'invalid_id' formats
        if "invalid_id" in info:
            invalid_id = info["invalid_id"]
            valid_index = [idx for idx, f in enumerate(files) if int(f.stem.split("_")[1]) not in invalid_id]
        elif "valid_id" in info:
            valid_id = info["valid_id"]
            valid_index = [idx for idx, f in enumerate(files) if int(f.stem.split("_")[1]) in valid_id]
        else:
            # If neither specified, use all files
            valid_index = list(range(len(files)))
        
        valid_files = [files[idx] for idx in valid_index]
        valid_lines = [lines[idx] for idx in valid_index]
        valid_feats = [info["feat"] for _ in range(len(valid_files))]
        # Extract mask information if available
        valid_masks = [info.get("masks", {}) for _ in range(len(valid_files))]
        all_valid_files.extend(valid_files)
        all_valid_lines.extend(valid_lines)
        all_valid_feats.extend(valid_feats)
        all_valid_masks.extend(valid_masks)
    return all_valid_files, all_valid_lines, all_valid_feats, all_valid_masks


# ---------- main converter ---------------------------------------------------
class Converter:
    def __init__(self, out_zip: Path, enable_visualization: bool = False, enable_anomaly_detection: bool = True,
                 enable_image_logging: bool = True, enable_zarr_conversion: bool = False):
        self.enable_visualization = enable_visualization
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_image_logging = enable_image_logging
        self.enable_zarr_conversion = enable_zarr_conversion
        self.visualizer = None
        self.out_zip = Path(out_zip)
        
        # Initialize ReplayBuffer for zarr conversion
        if self.enable_zarr_conversion:
            self.replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
        else:
            self.replay_buffer = None
        
        # Anomaly tracking
        self.problematic_episodes = []
        self.anomaly_stats = {
            'total_episodes': 0,
            'episodes_with_anomalies': 0,
            'total_zero_frames': 0,
            'total_jump_frames': 0,
            'max_jump_distance': 0.0
        }
        
        # Mask storage
        self.masks = {}
        
        self.image_resize_resolution: Tuple[int, int] = (400, 300)
        self.per_episode_time_span_ns: int = 10_000_000_000
        self.frame_interval_ns: int = 10_000_000

        # Load offset calibration from JSON files
        # Script is in: SuperInference/scripts/
        # Need to go up 1 level to get to SuperInference root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        superinference_root = os.path.dirname(script_dir)

        # Use relative paths from SuperInference root
        left_offset_path = os.path.join(superinference_root, "calibration/mask_offset/left/umi2center_left.json")
        right_offset_path = os.path.join(superinference_root, "calibration/mask_offset/right/umi2center_right.json")

        # Debug: log paths
        logger.info("=" * 80)
        logger.info("Loading Offset Configuration")
        logger.info("=" * 80)
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"SuperInference root: {superinference_root}")
        logger.info(f"Left offset path: {left_offset_path}")
        logger.info(f"Left offset exists: {os.path.exists(left_offset_path)}")
        logger.info(f"Right offset path: {right_offset_path}")
        logger.info(f"Right offset exists: {os.path.exists(right_offset_path)}")

        left_offset_x_raw, left_offset_y_raw = load_offset_from_json(left_offset_path)
        right_offset_x_raw, right_offset_y_raw = load_offset_from_json(right_offset_path)

        # The offset in JSON represents "robot to umi" displacement
        # To align robot image to umi position, we need to apply the negative offset
        self.left_offset_x = -left_offset_x_raw
        self.left_offset_y = -left_offset_y_raw
        self.right_offset_x = -right_offset_x_raw
        self.right_offset_y = -right_offset_y_raw

        logger.info(
            f"Raw offset (robot->umi): left=({left_offset_x_raw:.2f}, {left_offset_y_raw:.2f}), right=({right_offset_x_raw:.2f}, {right_offset_y_raw:.2f})")
        logger.info(
            f"Applied compensation: left=({self.left_offset_x:.2f}, {self.left_offset_y:.2f}), right=({self.right_offset_x:.2f}, {self.right_offset_y:.2f})")
        logger.info("=" * 80)

        if self.enable_visualization:
            self.visualizer = RerunVisualizer(app_name="UMI_Trajectory_Visualization", spawn=True)
            if self.visualizer.is_initialized:
                self.visualizer.setup_3d_world("world", coordinate_system="y_up")
                logger.info("Rerun visualization initialized")

    def _load_masks(self, mask_info: dict) -> dict:
        """
        Load mask images from configuration.

        Args:
            mask_info: Dictionary containing mask paths

        Returns:
            Dictionary with loaded mask arrays
        """
        masks = {}

        if not mask_info:
            return masks

        for mask_name, mask_path in mask_info.items():
            try:
                # Support both absolute and relative paths
                if os.path.isabs(mask_path):
                    full_mask_path = mask_path
                else:
                    # Construct full path relative to script directory
                    full_mask_path = os.path.join(os.path.dirname(__file__), mask_path)

                if os.path.exists(full_mask_path):
                    mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks[mask_name] = mask
                        logger.info(f"Loaded mask: {mask_name} from {full_mask_path} (shape: {mask.shape})")
                    else:
                        logger.warning(f"Could not load mask {full_mask_path}")
                else:
                    logger.warning(f"Mask file not found: {full_mask_path}")

            except Exception as e:
                logger.error(f"Error loading mask {mask_name}: {e}")

        return masks

    def run(self):
        # 1. collect all episode_XXXX.hdf5
        files, lines, feats, masks_info = extract_episode_files(hdf5_info)

        # Load masks if available
        if masks_info and len(masks_info) > 0:
            self.masks = self._load_masks(masks_info[0])  # Assuming all episodes use same masks

        # 2. First pass: Perform anomaly detection on all episodes
        logger.info("First pass: Performing anomaly detection on all episodes...")
        anomalous_episode_indices = set()
        
        if self.enable_anomaly_detection:
            for ep_idx, (ep_path, feat) in enumerate(tqdm(zip(files, feats), total=len(files), desc="Anomaly detection")):
                try:
                    with h5py.File(ep_path, "r") as h5:
                        # Extract pose data
                        if 'pose_unified' in feat:
                            unified_pose_key = feat['pose_unified']
                            unified_pose = h5[unified_pose_key][()]
                            left_pose = unified_pose[:, 0, :]
                            right_pose = unified_pose[:, 1, :]
                        else:
                            left_pose_key = feat['pose_left']
                            right_pose_key = feat['pose_right']
                            left_pose = h5[left_pose_key][()]
                            right_pose = h5[right_pose_key][()]
                        
                        # Detect anomalies
                        left_anomalies = detect_pose_anomalies(left_pose, tracker_name="left_tracker")
                        right_anomalies = detect_pose_anomalies(right_pose, tracker_name="right_tracker")
                        
                        if left_anomalies['has_anomalies'] or right_anomalies['has_anomalies']:
                            anomalous_episode_indices.add(ep_idx)
                            episode_info = {
                                'episode_idx': ep_idx,
                                'episode_path': str(ep_path),
                                'anomalies': {'left': left_anomalies, 'right': right_anomalies}
                            }
                            self.problematic_episodes.append(episode_info)
                            
                            # Update statistics
                            self.anomaly_stats['total_zero_frames'] += len(left_anomalies['zero_frames']) + len(right_anomalies['zero_frames'])
                            self.anomaly_stats['total_jump_frames'] += len(left_anomalies['jump_frames']) + len(right_anomalies['jump_frames'])
                            
                            for _, distance in left_anomalies['jump_frames'] + right_anomalies['jump_frames']:
                                self.anomaly_stats['max_jump_distance'] = max(self.anomaly_stats['max_jump_distance'], distance)
                        
                        self.anomaly_stats['total_episodes'] += 1
                        
                except Exception as e:
                    logger.warning(f"Error during anomaly detection for {ep_path}: {e}")
                    continue
            
            logger.info(f"Anomaly detection complete: {len(anomalous_episode_indices)} anomalous episodes found")
        
        # 3. Filter out anomalous episodes for normalization
        clean_files = [f for idx, f in enumerate(files) if idx not in anomalous_episode_indices]
        clean_feats = [f for idx, f in enumerate(feats) if idx not in anomalous_episode_indices]
        
        logger.info(f"Computing normalization factors using {len(clean_files)} clean episodes (excluding {len(anomalous_episode_indices)} anomalous episodes)")
        
        # 4. Compute global gripper width normalization factors using only clean episodes
        left_norm_factor, right_norm_factor = compute_dataset_gripper_normalization_factors(clean_files, clean_feats)

        episode_ends = []
        frame_cursor = 0
        # 5. process each episode
        T_records = {}
        
        # Initialize visualization data storage
        all_left_positions = []
        all_right_positions = []
        all_left_positions_new = []
        all_right_positions_new = []
        all_left_quaternions = []
        all_right_quaternions = []
        all_left_quaternions_new = []
        all_right_quaternions_new = []
        all_left_grippers = []
        all_right_grippers = []
        episode_boundaries = []
        
        for ep_idx, (ep_path, line, feat) in enumerate(tqdm(zip(files, lines, feats), total=len(files), desc="Processing episodes")):
            with h5py.File(ep_path, "r") as h5:
                try:
                    episode_data = dict()
                    # Extract unified pose data or fallback to separate devices
                    left_grip_key = feat['grip_left']
                    right_grip_key = feat['grip_right']
                    grip_left = h5[left_grip_key][()] / 1000.0  # (T, 1) float32
                    grip_right = h5[right_grip_key][()] / 1000.0  # (T, 1) float32

                    # Normalize using global factors computed from entire dataset
                    grip_left = normalize_gripper_width(grip_left, left_norm_factor)
                    grip_right = normalize_gripper_width(grip_right, right_norm_factor)

                    # Check if we have unified pose data or separate devices
                    if 'pose_unified' in feat:
                        # New unified format: [T, 2, 7] where [:, 0, :] is left and [:, 1, :] is right
                        unified_pose_key = feat['pose_unified']
                        unified_pose = h5[unified_pose_key][()]  # (T, 2, 7) [px, py, pz, qx, qy, qz, qw]

                        if ep_idx == 0:  # Log format info for first episode
                            logger.info(f"Using unified pose format: {unified_pose.shape} from {unified_pose_key}")

                        # Extract left and right poses from unified data
                        # Convention: tracker_roles = ['left_foot', 'right_foot'] maps to [left_arm, right_arm]
                        left_pose = unified_pose[:, 0, :]  # (T, 7) - first tracker (left_foot -> left_arm)
                        right_pose = unified_pose[:, 1, :]  # (T, 7) - second tracker (right_foot -> right_arm)
                    else:
                        # Fallback to separate devices for backward compatibility
                        left_pose_key = feat['pose_left']
                        right_pose_key = feat['pose_right']
                        left_pose = h5[left_pose_key][()]  # (T, 7) [px, py, pz, qx, qy, qz, qw]
                        right_pose = h5[right_pose_key][()]  # (T, 7) [px, py, pz, qx, qy, qz, qw]

                        if ep_idx == 0:  # Log format info for first episode
                            logger.info(f"Using separate pose devices: {left_pose.shape} from {left_pose_key}, {right_pose.shape} from {right_pose_key}")

                    # Check if this episode has anomalies (already detected in first pass)
                    has_episode_anomalies = ep_idx in anomalous_episode_indices

                    # Process left pose
                    left_pose = replace_zero_rows_with_last_nonzero(left_pose)
                    pos_left = left_pose[:, :3]
                    quat_left = left_pose[:, 3:]
                    axis_angle_left = quaternion_to_axis_angle(quat_left)
                    pose_mat_left = pose_to_mat(np.concatenate([pos_left, axis_angle_left], axis=-1))
                    pose_left = mat_to_pose(pose_mat_left)
                    pos_left = pose_left[:, :3]
                    axis_angle_left = pose_left[:, 3:]

                    # Process right pose
                    right_pose = replace_zero_rows_with_last_nonzero(right_pose)
                    pos_right = right_pose[:, :3]
                    quat_right = right_pose[:, 3:]
                    axis_angle_right = quaternion_to_axis_angle(quat_right)
                    pose_mat_right = pose_to_mat(np.concatenate([pos_right, axis_angle_right], axis=-1))
                    pose_right = mat_to_pose(pose_mat_right)
                    pos_right = pose_right[:, :3]
                    axis_angle_right = pose_right[:, 3:]

                    # Apply transformations to create new trajectories
                    pose_left_mat_new = pose_mat_left
                    pose_left_new = mat_to_pose(pose_left_mat_new)
                    pos_left_new = pose_left_new[:, :3]
                    axis_angle_left_new = pose_left_new[:, 3:]

                    pose_right_mat_new = pose_mat_right
                    pose_right_new = mat_to_pose(pose_right_mat_new)
                    pos_right_new = pose_right_new[:, :3]
                    axis_angle_right_new = pose_right_new[:, 3:]

                    # Convert axis-angle to quaternions for visualization
                    quat_left = st.Rotation.from_rotvec(axis_angle_left).as_quat()
                    quat_right = st.Rotation.from_rotvec(axis_angle_right).as_quat()
                    quat_left_new = st.Rotation.from_rotvec(axis_angle_left_new).as_quat()
                    quat_right_new = st.Rotation.from_rotvec(axis_angle_right_new).as_quat()

                    # Calculate timestamps based on available data sources
                    if 'pose_unified' in feat:
                        all_ts = [left_pose.shape[0], grip_left.shape[0], grip_right.shape[0],
                                  h5[feat['img_left']].shape[0], h5[feat['img_right']].shape[0]]
                    else:
                        all_ts = [left_pose.shape[0], right_pose.shape[0], grip_left.shape[0], grip_right.shape[0],
                                  h5[feat['img_left']].shape[0], h5[feat['img_right']].shape[0]]

                    T = min(all_ts)
                    maxT = max(all_ts)
                    if np.abs(maxT - T) >= 3:
                        print(f"Warning: max/min length mismatch: {T} vs {maxT} in {ep_path}")
                    if line is not None:
                        start, end = line
                        end = min(end, T)
                    else:
                        start, end = 0, T
                    T_records[str(ep_path)] = [start, end]
                    start_pose_left = np.concatenate([pos_left[0], axis_angle_left[0]]).astype(np.float64)
                    end_pose_left = np.concatenate([pos_left[-1], axis_angle_left[-1]]).astype(np.float64)

                    start_pose_right = np.concatenate([pos_right[0], axis_angle_right[0]]).astype(np.float64)
                    end_pose_right = np.concatenate([pos_right[-1], axis_angle_right[-1]]).astype(np.float64)
                    # append

                    episode_data['robot0_eef_pos'] = pos_left[start:end]
                    episode_data['robot0_eef_rot_axis_angle'] = axis_angle_left[start:end]
                    episode_data['robot0_gripper_width'] = grip_left[start:end]
                    episode_data['robot0_demo_start_pose'] = np.tile(start_pose_left, (end - start, 1))
                    episode_data['robot0_demo_end_pose'] = np.tile(end_pose_left, (end - start, 1))
                    episode_data['robot1_eef_pos'] = pos_right[start:end]
                    episode_data['robot1_eef_rot_axis_angle'] = axis_angle_right[start:end]
                    episode_data['robot1_gripper_width'] = grip_right[start:end]
                    episode_data['robot1_demo_start_pose'] = np.tile(start_pose_right, (end - start, 1))
                    episode_data['robot1_demo_end_pose'] = np.tile(end_pose_right, (end - start, 1))
                    
                    # Add episode data to replay buffer if zarr conversion is enabled
                    # Skip episodes with anomalies if anomaly detection is enabled
                    if self.enable_zarr_conversion and self.replay_buffer is not None:
                        if self.enable_anomaly_detection and has_episode_anomalies:
                            logger.info(f"Skipping episode {ep_idx} in zarr conversion due to anomalies")
                        else:
                            self.replay_buffer.add_episode(data=episode_data, compressors=None)

                    frame_cursor += (end - start)
                    episode_ends.append(frame_cursor)

                    # Collect visualization data if enabled
                    if self.enable_visualization and self.visualizer and self.visualizer.is_initialized:
                        # Store positions, quaternions and gripper states for this episode
                        all_left_positions.append(pos_left[start:end])
                        all_right_positions.append(pos_right[start:end])
                        all_left_positions_new.append(pos_left_new[start:end])
                        all_right_positions_new.append(pos_right_new[start:end])
                        all_left_quaternions.append(quat_left[start:end])
                        all_right_quaternions.append(quat_right[start:end])
                        all_left_quaternions_new.append(quat_left_new[start:end])
                        all_right_quaternions_new.append(quat_right_new[start:end])
                        all_left_grippers.append(grip_left[start:end])
                        all_right_grippers.append(grip_right[start:end])
                        episode_boundaries.append(len(all_left_positions[-1]))

                        # Only load images if image logging is enabled
                        left_images = None
                        right_images = None
                        if self.enable_image_logging:
                            left_images = self._load_episode_image_sequence(
                                h5_file=h5,
                                dataset_key=feat.get('img_left'),
                                start=start,
                                end=end,
                                episode_path=ep_path
                            )
                            right_images = self._load_episode_image_sequence(
                                h5_file=h5,
                                dataset_key=feat.get('img_right'),
                                start=start,
                                end=end,
                                episode_path=ep_path
                            )

                        # Visualize current episode trajectory (both original and transformed)
                        self._visualize_episode_trajectory(
                            ep_idx, ep_path,
                            pos_left[start:end], pos_right[start:end],
                            pos_left_new[start:end], pos_right_new[start:end],
                            quat_left[start:end], quat_right[start:end],
                            quat_left_new[start:end], quat_right_new[start:end],
                            grip_left[start:end], grip_right[start:end]
                        )

                        # Only load and log images if image logging is enabled
                        if self.enable_image_logging:
                            # Debug: Log image loading results
                            if left_images is not None:
                                logger.info(f"Loaded {len(left_images)} left images for episode {ep_idx}")
                            else:
                                logger.warning(f"No left images loaded for episode {ep_idx}")

                            if right_images is not None:
                                logger.info(f"Loaded {len(right_images)} right images for episode {ep_idx}")
                            else:
                                logger.warning(f"No right images loaded for episode {ep_idx}")

                            camera_images: Dict[str, List[np.ndarray]] = {}
                            left_key = feat.get('img_left')
                            if left_key and left_images is not None:
                                camera_images[left_key] = left_images
                            right_key = feat.get('img_right')
                            if right_key and right_images is not None:
                                camera_images[right_key] = right_images

                            if camera_images:
                                self._log_episode_images(
                                    ep_idx=ep_idx,
                                    episode_name=ep_path.stem,
                                    camera_images=camera_images
                                )
                        else:
                            logger.debug(f"Skipping image logging for episode {ep_idx} (disabled)")
                except Exception as e:
                    logger.error(f"Error processing {ep_path}: {e}")

        # Visualize complete dataset if enabled
        if self.enable_visualization and self.visualizer and self.visualizer.is_initialized:
            self._visualize_complete_dataset(
                all_left_positions,
                all_right_positions,
                all_left_positions_new,
                all_right_positions_new,
                all_left_quaternions,
                all_right_quaternions,
                all_left_quaternions_new,
                all_right_quaternions_new,
                all_left_grippers,
                all_right_grippers,
                episode_boundaries
            )

        # Output anomaly detection results
        if self.enable_anomaly_detection:
            self._output_anomaly_results()

        # Process camera images and save to zarr if enabled
        if self.enable_zarr_conversion and self.replay_buffer is not None:
            # Get indices of problematic episodes to skip
            problematic_indices = set([ep['episode_idx'] for ep in self.problematic_episodes]) if self.enable_anomaly_detection else set()
            self._convert_to_zarr(files, feats, T_records, problematic_indices)
            self._save_replay_buffer()


    
    def _visualize_episode_trajectory(self, ep_idx: int, ep_path: Path, 
                                    pos_left: np.ndarray, pos_right: np.ndarray,
                                    pos_left_new: np.ndarray, pos_right_new: np.ndarray,
                                    quat_left: np.ndarray, quat_right: np.ndarray,
                                    quat_left_new: np.ndarray, quat_right_new: np.ndarray,
                                    grip_left: np.ndarray, grip_right: np.ndarray) -> None:
        """Visualize trajectory for a single episode (original and transformed) with rotation."""
        try:
            # Set time context for this episode to enable per-episode playback
            rr.set_time("episode", sequence=ep_idx)
            
            # Generate colors based on gripper states
            left_colors = [gripper_width_to_color(g[0] if len(g) > 0 else g) for g in grip_left]
            right_colors = [gripper_width_to_color(g[0] if len(g) > 0 else g) for g in grip_right]
            
            # Generate fixed colors for transformed trajectories
            left_colors_new = [[100, 150, 255]] * len(pos_left_new)  # Light blue for transformed left
            right_colors_new = [[255, 150, 100]] * len(pos_right_new)  # Light orange for transformed right
            
            episode_name = ep_path.stem
            
            # Visualize original left arm trajectory with rotation (gripper-based colors)
            visualize_trajectory_with_rotation(
                f"world/episodes/{episode_name}/left_arm_original",
                pos_left, quat_left, left_colors, 
                arrow_scale=0.02, rotation_scale=0.03, show_every_n=8
            )
            
            # Visualize transformed left arm trajectory with rotation (fixed color)
            visualize_trajectory_with_rotation(
                f"world/episodes/{episode_name}/left_arm_transformed", 
                pos_left_new, quat_left_new, left_colors_new, 
                arrow_scale=0.02, rotation_scale=0.03, show_every_n=8
            )
            
            # Visualize original right arm trajectory with rotation (gripper-based colors)
            visualize_trajectory_with_rotation(
                f"world/episodes/{episode_name}/right_arm_original", 
                pos_right, quat_right, right_colors, 
                arrow_scale=0.02, rotation_scale=0.03, show_every_n=8
            )
            
            # Visualize transformed right arm trajectory with rotation (fixed color)
            visualize_trajectory_with_rotation(
                f"world/episodes/{episode_name}/right_arm_transformed", 
                pos_right_new, quat_right_new, right_colors_new, 
                arrow_scale=0.02, rotation_scale=0.03, show_every_n=8
            )

            # Log episode info
            episode_info = f"Episode {ep_idx}: {episode_name}\n"
            episode_info += f"Frames: {len(pos_left)}\n"
            episode_info += f"Left gripper range: {grip_left.min():.3f} - {grip_left.max():.3f}\n"
            episode_info += f"Right gripper range: {grip_right.min():.3f} - {grip_right.max():.3f}\n"
            episode_info += f"Left position change: {np.linalg.norm(pos_left_new - pos_left, axis=1).mean():.3f}m avg\n"
            episode_info += f"Right position change: {np.linalg.norm(pos_right_new - pos_right, axis=1).mean():.3f}m avg"

            log_text_summary(f"world/episodes/{episode_name}/info", episode_info)

            # Log time-series data for this episode to enable step-by-step playback in rerun
            time_series_base_path = f"time_series/episodes/{episode_name}"

            def _log_episode_time_series(
                positions: np.ndarray,
                grippers: np.ndarray,
                arm_name: str
            ) -> None:
                if len(positions) == 0 or len(grippers) == 0:
                    return

                episode_offset_ns = int(ep_idx * 10_000_000_000)
                frame_interval_ns = 10_000_000  # 10ms per frame surrogate for now

                visualize_action_time_series(
                    base_path=time_series_base_path,
                    device_name="dataset",
                    arm_name=arm_name,
                    positions=positions,
                    grippers=grippers,
                    timestamp_ns=episode_offset_ns,
                    step_offset_ns=frame_interval_ns,
                    time_timeline="episode_time",
                    use_sequence_time=False,
                    constant_sequence_timelines={"episode": ep_idx, "frame": 0},
                    per_step_sequence_timelines={
                        "episode_frame": lambda step_idx: step_idx
                    },
                    axis_ranges={
                        "pos_x": (-0.1, 0.1),
                        "pos_y": (-0.1, 0.1),
                        "pos_z": (-0.1, 0.1),
                        "gripper": (-0.01, 0.09)
                    }
                )

            _log_episode_time_series(pos_left, grip_left, "left_arm")
            _log_episode_time_series(pos_right, grip_right, "right_arm")
            
        except Exception as e:
            logger.error(f"Error visualizing episode {ep_idx}: {e}")

    def _visualize_complete_dataset(self, all_left_positions: List[np.ndarray], 
                                  all_right_positions: List[np.ndarray],
                                  all_left_positions_new: List[np.ndarray],
                                  all_right_positions_new: List[np.ndarray],
                                  all_left_quaternions: List[np.ndarray],
                                  all_right_quaternions: List[np.ndarray],
                                  all_left_quaternions_new: List[np.ndarray],
                                  all_right_quaternions_new: List[np.ndarray],
                                  all_left_grippers: List[np.ndarray],
                                  all_right_grippers: List[np.ndarray],
                                  episode_boundaries: List[int]) -> None:
        """Visualize complete dataset overview (original and transformed) with rotation."""
        try:
            # Set time context for overview
            rr.set_time("overview", timestamp=0)
            
            # Concatenate all episodes
            all_left_pos = np.concatenate(all_left_positions, axis=0)
            all_right_pos = np.concatenate(all_right_positions, axis=0)
            all_left_pos_new = np.concatenate(all_left_positions_new, axis=0)
            all_right_pos_new = np.concatenate(all_right_positions_new, axis=0)
            all_left_quat = np.concatenate(all_left_quaternions, axis=0)
            all_right_quat = np.concatenate(all_right_quaternions, axis=0)
            all_left_quat_new = np.concatenate(all_left_quaternions_new, axis=0)
            all_right_quat_new = np.concatenate(all_right_quaternions_new, axis=0)
            all_left_grip = np.concatenate(all_left_grippers, axis=0)
            all_right_grip = np.concatenate(all_right_grippers, axis=0)
            
            # Generate colors for complete dataset
            left_colors = [gripper_width_to_color(g[0] if len(g) > 0 else g) for g in all_left_grip]
            right_colors = [gripper_width_to_color(g[0] if len(g) > 0 else g) for g in all_right_grip]
            
            # Generate fixed colors for transformed trajectories
            left_colors_new = [[80, 120, 200]] * len(all_left_pos_new)  # Darker blue for transformed left
            right_colors_new = [[200, 120, 80]] * len(all_right_pos_new)  # Darker orange for transformed right
            
            # Visualize original complete trajectories with rotation
            visualize_trajectory_with_rotation(
                "world/complete_dataset/left_arm_original_all",
                all_left_pos, all_left_quat, left_colors, 
                arrow_scale=0.015, rotation_scale=0.02, show_every_n=20,
                episode_boundaries=episode_boundaries
            )
            
            visualize_trajectory_with_rotation(
                "world/complete_dataset/right_arm_original_all",
                all_right_pos, all_right_quat, right_colors, 
                arrow_scale=0.015, rotation_scale=0.02, show_every_n=20,
                episode_boundaries=episode_boundaries
            )
            
            # Visualize transformed complete trajectories with rotation
            visualize_trajectory_with_rotation(
                "world/complete_dataset/left_arm_transformed_all",
                all_left_pos_new, all_left_quat_new, left_colors_new, 
                arrow_scale=0.015, rotation_scale=0.02, show_every_n=20,
                episode_boundaries=episode_boundaries
            )
            
            visualize_trajectory_with_rotation(
                "world/complete_dataset/right_arm_transformed_all",
                all_right_pos_new, all_right_quat_new, right_colors_new, 
                arrow_scale=0.015, rotation_scale=0.02, show_every_n=20,
                episode_boundaries=episode_boundaries
            )
            
            # Log dataset statistics
            dataset_info = f"Complete Dataset Statistics:\n"
            dataset_info += f"Transform: vive_tcp_T transformation applied\n"
            dataset_info += f"Total episodes: {len(all_left_positions)}\n"
            dataset_info += f"Total frames: {len(all_left_pos)}\n\n"
            
            dataset_info += f"ORIGINAL TRAJECTORIES:\n"
            dataset_info += f"Left arm position range:\n"
            dataset_info += f"  X: {all_left_pos[:, 0].min():.3f} - {all_left_pos[:, 0].max():.3f}\n"
            dataset_info += f"  Y: {all_left_pos[:, 1].min():.3f} - {all_left_pos[:, 1].max():.3f}\n"
            dataset_info += f"  Z: {all_left_pos[:, 2].min():.3f} - {all_left_pos[:, 2].max():.3f}\n"
            dataset_info += f"Right arm position range:\n"
            dataset_info += f"  X: {all_right_pos[:, 0].min():.3f} - {all_right_pos[:, 0].max():.3f}\n"
            dataset_info += f"  Y: {all_right_pos[:, 1].min():.3f} - {all_right_pos[:, 1].max():.3f}\n"
            dataset_info += f"  Z: {all_right_pos[:, 2].min():.3f} - {all_right_pos[:, 2].max():.3f}\n"
            
            dataset_info += f"\nTRANSFORMED TRAJECTORIES:\n"
            dataset_info += f"Left arm position range:\n"
            dataset_info += f"  X: {all_left_pos_new[:, 0].min():.3f} - {all_left_pos_new[:, 0].max():.3f}\n"
            dataset_info += f"  Y: {all_left_pos_new[:, 1].min():.3f} - {all_left_pos_new[:, 1].max():.3f}\n"
            dataset_info += f"  Z: {all_left_pos_new[:, 2].min():.3f} - {all_left_pos_new[:, 2].max():.3f}\n"
            dataset_info += f"Right arm position range:\n"
            dataset_info += f"  X: {all_right_pos_new[:, 0].min():.3f} - {all_right_pos_new[:, 0].max():.3f}\n"
            dataset_info += f"  Y: {all_right_pos_new[:, 1].min():.3f} - {all_right_pos_new[:, 1].max():.3f}\n"
            dataset_info += f"  Z: {all_right_pos_new[:, 2].min():.3f} - {all_right_pos_new[:, 2].max():.3f}\n"
            
            dataset_info += f"\nGRIPPER DATA:\n"
            dataset_info += f"Left gripper range: {all_left_grip.min():.3f} - {all_left_grip.max():.3f}\n"
            dataset_info += f"Right gripper range: {all_right_grip.min():.3f} - {all_right_grip.max():.3f}\n"
            
            dataset_info += f"\nTRANSFORMATION STATISTICS:\n"
            dataset_info += f"Left arm avg position change: {np.linalg.norm(all_left_pos_new - all_left_pos, axis=1).mean():.3f}m\n"
            dataset_info += f"Right arm avg position change: {np.linalg.norm(all_right_pos_new - all_right_pos, axis=1).mean():.3f}m"
            
            log_text_summary("world/complete_dataset/statistics", dataset_info)
            
            logger.info("Complete dataset visualization finished")
            
        except Exception as e:
            logger.error(f"Error visualizing complete dataset: {e}")
    
    
    
    def _load_episode_image_sequence(self, h5_file, dataset_key: Optional[str], start: int, end: int, episode_path: Path) -> Optional[List[np.ndarray]]:
        """
        Load and resize image sequence from HDF5 file for a specific episode.

        Args:
            h5_file: Open HDF5 file handle
            dataset_key: Key for the image dataset in HDF5
            start: Start frame index
            end: End frame index
            episode_path: Path to episode file for logging

        Returns:
            List of resized images or None if dataset_key is None or loading fails
        """
        if dataset_key is None:
            return None

        try:
            if dataset_key not in h5_file:
                logger.warning(f"Image dataset '{dataset_key}' not found in {episode_path}")
                return None

            # Load image data
            image_data = h5_file[dataset_key][start:end]  # (T, H, W, C)

            if len(image_data) == 0:
                logger.warning(f"No image data found for {dataset_key} in range [{start}:{end}]")
                return None

            # Resize images to reduce log volume
            resized_images = []
            for i, img in enumerate(image_data):
                try:
                    # Ensure image is in correct format (H, W, C)
                    if len(img.shape) == 3:
                        # Ensure image is uint8 format for rerun
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)

                        # Resize using OpenCV
                        resized_img = cv2.resize(
                            img,
                            self.image_resize_resolution,
                            interpolation=cv2.INTER_AREA
                        )

                        # Ensure the resized image is in the correct format (H, W, C)
                        if len(resized_img.shape) == 3 and resized_img.shape[2] in [1, 3, 4]:
                            resized_images.append(resized_img)
                        else:
                            logger.warning(f"Unexpected resized image shape {resized_img.shape} at frame {i}")
                            continue
                    else:
                        logger.warning(f"Unexpected image shape {img.shape} at frame {i}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to resize image at frame {i}: {e}")
                    continue

            logger.debug(f"Loaded {len(resized_images)} resized images from {dataset_key}")
            return resized_images

        except Exception as e:
            logger.error(f"Failed to load image sequence from {dataset_key}: {e}")
            return None

    def _log_episode_images(self, ep_idx: int, episode_name: str, camera_images: Dict[str, List[np.ndarray]]) -> None:
        """
        Log camera images for an episode with proper time synchronization.

        Args:
            ep_idx: Episode index
            episode_name: Name of the episode
            camera_images: Dictionary mapping camera keys to image lists
        """
        try:
            # Calculate episode time offset (same as used in trajectory visualization)
            episode_offset_ns = int(ep_idx * self.per_episode_time_span_ns)

            for camera_key, images in camera_images.items():
                if not images:
                    continue

                logger.info(f"Logging {len(images)} images for camera {camera_key} in episode {ep_idx}")

                for frame_idx, img in enumerate(images):
                    # Use the same time context as trajectory visualization
                    frame_time_ns = episode_offset_ns + frame_idx * self.frame_interval_ns

                    # Set multiple time contexts for better synchronization
                    rr.set_time("episode", sequence=ep_idx)
                    rr.set_time("episode_time", timestamp=1e-9 * frame_time_ns)
                    rr.set_time("episode_frame", sequence=frame_idx)
                    rr.set_time("step_time", timestamp=1e-9 * frame_time_ns)

                    # Log the image with a simpler path structure
                    image_path = f"images/{camera_key}"
                    rr.log(image_path, rr.Image(img))

                    # Also log to a more specific path for episode organization
                    episode_image_path = f"episodes/{episode_name}/images/{camera_key}"
                    rr.log(episode_image_path, rr.Image(img))

        except Exception as e:
            logger.error(f"Failed to log episode images for {episode_name}: {e}")

    def _output_anomaly_results(self) -> None:
        """Output comprehensive anomaly detection results."""
        try:
            # Update final statistics
            self.anomaly_stats['episodes_with_anomalies'] = len(self.problematic_episodes)
            
            logger.info("=" * 80)
            logger.info("VIVE TRACKER POSE ANOMALY DETECTION RESULTS")
            logger.info("=" * 80)
            
            # Overall statistics
            logger.info(f"Total episodes processed: {self.anomaly_stats['total_episodes']}")
            logger.info(f"Episodes with anomalies: {self.anomaly_stats['episodes_with_anomalies']}")
            logger.info(f"Anomaly rate: {self.anomaly_stats['episodes_with_anomalies']/max(1, self.anomaly_stats['total_episodes'])*100:.1f}%")
            logger.info(f"Total zero frames detected: {self.anomaly_stats['total_zero_frames']}")
            logger.info(f"Total jump frames detected: {self.anomaly_stats['total_jump_frames']}")
            logger.info(f"Maximum jump distance: {self.anomaly_stats['max_jump_distance']:.4f}m")
            
            if self.problematic_episodes:
                logger.info("\nPROBLEMATIC EPISODES:")
                logger.info("-" * 60)
                
                problematic_indices = []
                for episode_info in self.problematic_episodes:
                    ep_idx = episode_info['episode_idx']
                    ep_path = episode_info['episode_path']
                    anomalies = episode_info['anomalies']
                    
                    problematic_indices.append(ep_idx)
                    
                    logger.info(f"Episode {ep_idx}: {Path(ep_path).name}")
                    
                    # Left tracker anomalies
                    left_anom = anomalies['left']
                    if left_anom['has_anomalies']:
                        logger.info(f"  Left tracker:")
                        if left_anom['zero_frames']:
                            logger.info(f"    - Zero frames: {len(left_anom['zero_frames'])} (indices: {left_anom['zero_frames'][:10]}{'...' if len(left_anom['zero_frames']) > 10 else ''})")
                        if left_anom['jump_frames']:
                            logger.info(f"    - Jump frames: {len(left_anom['jump_frames'])}")
                            for frame_idx, distance in left_anom['jump_frames'][:5]:  # Show first 5 jumps
                                logger.info(f"      Frame {frame_idx}: {distance:.4f}m jump")
                            if len(left_anom['jump_frames']) > 5:
                                logger.info(f"      ... and {len(left_anom['jump_frames']) - 5} more jumps")
                    
                    # Right tracker anomalies
                    right_anom = anomalies['right']
                    if right_anom['has_anomalies']:
                        logger.info(f"  Right tracker:")
                        if right_anom['zero_frames']:
                            logger.info(f"    - Zero frames: {len(right_anom['zero_frames'])} (indices: {right_anom['zero_frames'][:10]}{'...' if len(right_anom['zero_frames']) > 10 else ''})")
                        if right_anom['jump_frames']:
                            logger.info(f"    - Jump frames: {len(right_anom['jump_frames'])}")
                            for frame_idx, distance in right_anom['jump_frames'][:5]:  # Show first 5 jumps
                                logger.info(f"      Frame {frame_idx}: {distance:.4f}m jump")
                            if len(right_anom['jump_frames']) > 5:
                                logger.info(f"      ... and {len(right_anom['jump_frames']) - 5} more jumps")
                    
                    logger.info("")
                
                logger.info("SUMMARY OF PROBLEMATIC EPISODE INDICES:")
                logger.info(f"Episodes with anomalies: {sorted(problematic_indices)}")
                
            else:
                logger.info("\n✅ No pose anomalies detected in any episodes!")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error outputting anomaly results: {e}")

    def _convert_to_zarr(self, files: List[Path], feats: List[dict], T_records: dict, problematic_indices: set = None) -> None:
        """Process and save camera images to replay buffer, skipping problematic episodes.
        
        Args:
            files: List of episode file paths
            feats: List of feature configurations
            T_records: Dictionary of time records for each episode
            problematic_indices: Set of episode indices to skip (default: None)
        """
        try:
            if problematic_indices is None:
                problematic_indices = set()
                
            logger.info(f"Processing camera images (skipping {len(problematic_indices)} problematic episodes)...")

            # Setup image compression and datasets
            n_cameras = 2
            img_compressor = JpegXl(level=compression_level, numthreads=1)

            for cam_id in range(n_cameras):
                name = f'camera{cam_id}_rgb'
                _ = self.replay_buffer.data.require_dataset(
                    name=name,
                    shape=(self.replay_buffer['robot0_eef_pos'].shape[0],) + (224, 224) + (3,),
                    chunks=(1,) + (224, 224) + (3,),
                    compressor=img_compressor,
                    dtype=np.uint8
                )

            frame_cursor = 0

            # Process each episode's images
            for ep_idx, (ep_path, feat) in enumerate(tqdm(zip(files, feats), total=len(files), desc="Processing camera images")):
                # Skip problematic episodes
                if ep_idx in problematic_indices:
                    logger.info(f"Skipping problematic episode {ep_idx} during image processing")
                    continue
                    
                with h5py.File(ep_path, "r") as h5:
                    start, end = T_records[str(ep_path)]
                    left_img_key = feat['img_left']
                    right_img_key = feat['img_right']

                    img_left = h5[left_img_key][start:end]  # for camera0_rgb
                    img_right = h5[right_img_key][start:end]  # for camera1_rgb

                    # Resize images to match mask dimensions (400x300) if needed
                    # Check if images need resizing (e.g., from 1600x1200 to 400x300)
                    if len(img_left) > 0:
                        original_shape = img_left[0].shape
                        target_height, target_width = 300, 400  # Match mask dimensions
                        
                        if original_shape[:2] != (target_height, target_width):
                            logger.info(f"Resizing images from {original_shape} to ({target_height}, {target_width}, 3)")
                            
                            # Resize left images
                            img_left_resized = []
                            for img in img_left:
                                img_resized = cv2.resize(img, (target_width, target_height), 
                                                       interpolation=cv2.INTER_AREA)
                                img_left_resized.append(img_resized)
                            img_left = np.array(img_left_resized)
                            
                            # Resize right images
                            img_right_resized = []
                            for img in img_right:
                                img_resized = cv2.resize(img, (target_width, target_height),
                                                       interpolation=cv2.INTER_AREA)
                                img_right_resized.append(img_resized)
                            img_right = np.array(img_right_resized)

                    # Apply offset compensation to all frames
                    img_left_offset = []
                    for img in img_left:
                        img_compensated = apply_offset_to_image(img, self.left_offset_x, self.left_offset_y)
                        img_left_offset.append(img_compensated)

                    img_right_offset = []
                    for img in img_right:
                        img_compensated = apply_offset_to_image(img, self.right_offset_x, self.right_offset_y)
                        img_right_offset.append(img_compensated)

                    # Convert lists back to arrays
                    img_left = np.array(img_left_offset)
                    img_right = np.array(img_right_offset)

                    # Get masks for left and right cameras
                    left_mask = self.masks.get('mask_left', None)
                    right_mask = self.masks.get('mask_right', None)

                    # Process images in parallel with masks
                    with ThreadPoolExecutor() as executor:
                        # Create partial functions with masks
                        process_left = lambda img: center_crop_square_resize_with_mask(img, left_mask)
                        process_right = lambda img: center_crop_square_resize_with_mask(img, right_mask)

                        img_left = list(executor.map(process_left, img_left))
                        img_right = list(executor.map(process_right, img_right))

                    img_left = np.stack(img_left)
                    img_right = np.stack(img_right)

                    # Save to replay buffer
                    img_array_left = self.replay_buffer.data['camera0_rgb']
                    for idx, img_left_i in enumerate(img_left):
                        img_array_left[frame_cursor + idx] = img_left_i

                    img_array_right = self.replay_buffer.data['camera1_rgb']
                    for idx, img_right_i in enumerate(img_right):
                        img_array_right[frame_cursor + idx] = img_right_i

                    frame_cursor += (end - start)

            logger.info("Camera image processing completed")

        except Exception as e:
            logger.error(f"Error processing camera images: {e}")

    def _save_replay_buffer(self) -> None:
        """Save replay buffer to zarr zip file."""
        try:
            logger.info(f"Saving ReplayBuffer to {self.out_zip}")
            with zarr.ZipStore(self.out_zip, mode='w') as zip_store:
                self.replay_buffer.save_to_store(store=zip_store)
            logger.info(f"Finished saving to {self.out_zip}")
        except Exception as e:
            logger.error(f"Error saving replay buffer: {e}")

# ---------- entry ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UMI HDF5 episodes to zarr format with optional trajectory visualization and anomaly detection")
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "out_zarr_zip",
        type=Path,
        nargs="?",
        default=Path("episodes.zarr.zip"),
        help="Output .zarr.zip (default: episodes.zarr.zip)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable rerun trajectory visualization"
    )
    parser.add_argument(
        "--no-anomaly-detection",
        action="store_true", 
        help="Disable vive tracker pose anomaly detection"
    )
    parser.add_argument(
        "--no-image-logging",
        action="store_true",
        help="Disable image logging to reduce memory pressure"
    )
    parser.add_argument(
        "--enable-zarr-conversion",
        action="store_true",
        help="Enable HDF5 to zarr conversion with ReplayBuffer"
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Merge config into hdf5_info
    for k, v in config_data.items():
        if k not in hdf5_info:
            hdf5_info[k] = v
    
    # Create output directory if needed
    data_dir = os.path.dirname(args.out_zarr_zip)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    Converter(
        out_zip=args.out_zarr_zip,
        enable_visualization=args.visualize,
        enable_anomaly_detection=not args.no_anomaly_detection,
        enable_image_logging=not args.no_image_logging,
        enable_zarr_conversion=args.enable_zarr_conversion,
    ).run()