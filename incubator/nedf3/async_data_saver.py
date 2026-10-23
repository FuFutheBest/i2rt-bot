#!/usr/bin/env python3
"""
Asynchronous Data Saver Consumer - Save device manager data to NEDF3 (UMI) format.

Uses NEDF SDK NEDF3Writer: all samples are written via ``write_message(topic, proto, ...)``
Camera frames use HAL ``RGBD`` topics in ``HAL_CAMERA_TOPICS`` (no ``/camera/intrinsic`` or ``/camera/extrinsic`` MCAP topics).

I2RT file-based robot commands are not used in this module.

Author:Yichen Xie
"""

import sys
import time
import argparse
import os
import json
import uuid
import numpy as np
import cv2
import yaml
from typing import Dict, List, Optional, Any, Tuple, Set
import queue
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from pynput import keyboard as pynput_keyboard

# NEDF SDK imports
from nmx_nedf_api import (
    NEDFFactory,
    NEDFWriterConfig,
    NEDFMetadata,
    NEDFTopicsConfig,
    NEDFType
)
from nmx_msg.Image_pb2 import Image, ImageDataType, RGBD

# Ensure project root is on PYTHONPATH when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger_config import logger
from utils.episode_time_ledger import record_episode_timing
from utils.time_control import precise_loop_timing
from incubator.control_panel.episode_anomaly_detector import (
    EpisodeAnalysisContext,
    EpisodeAnomalyDetector,
)
from utils.anomaly_jump_segments import (
    JumpSegmentExportConfig,
    cleanup_jump_segments_before_today,
    export_anomaly_jump_segments,
    quarantine_episode,
)
from utils.anomaly_status import build_anomaly_status, publish_anomaly_status
from utils.image_transform import crop_pad_to_aspect_and_size
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_device_shm,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
    unpack_frame_header,
)
from utils.gripper_angle_calibration import (
    GripperAngleCalibration,
    process_gripper_angle_with_date_check,
    require_today_calibrations_for_both_arms,
)
from incubator.nedf3.windows_DM3.collection_session import (
    build_extra_device_entries,
    get_metadata_fields_from_session,
    normalize_task_description,
    parse_extra_device_identifier,
)

# import debugpy; debugpy.connect(("0.0.0.0", 5678))


# Wrist camera indices in windows_collect_webcam: camera_0=right, camera_1=left.
_WRIST_CAMERA_INDICES: Tuple[int, ...] = (0, 1)

# Per-camera entries in metadata.extra (camera_0, camera_1, camera_2).
_METADATA_EXTRA_CAMERA_COUNT: int = 3
DEFAULT_EXTRA_DEVICE_TYPES: Tuple[str, ...] = (
    "robopocket_2.0_pro",
    "robopocket_2.0_pro",
    "robopocket_2.0_pro",
)
DEFAULT_EXTRA_HARDWARE_STRINGS: Tuple[str, ...] = ("310", "310", "310")
WRITE_QUEUE_FULL_WARN_INTERVAL_SEC = 1.0
MAINTENANCE_REQUEST_NAME = ".auto_upload_maintenance_request.json"
MAINTENANCE_DONE_NAME = ".auto_upload_maintenance_done.json"
MAINTENANCE_REQUEST_TIMEOUT_SEC = 300.0
MAINTENANCE_POLL_INTERVAL_SEC = 0.2
DISCARD_SHARING_RETRY_DELAYS_SEC: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.8, 1.6)
DEFAULT_TRACKER_STATUS_PATH = Path(PROJECT_ROOT) / "logs" / "tracker_status.json"

# Canonical lab_pro NEDF3 metadata (19 topics); see lab_pro_canonical_metadata_example.json.
NEDF3_RECORD_TOPIC_LIST: Tuple[str, ...] = (
    "nmx/nedf/metadata",
    "nmx/hal/camera/camera_0/rgbd",
    "nmx/hal/umi/right/pose",
    "nmx/hal/umi/right/hand_state",
    "nmx/hal/umi/right/raw_hand_state",
    "nmx/hal/umi/right/slam",
    "nmx/hal/umi/right/extra",
    "nmx/hal/camera/camera_1/rgbd",
    "nmx/hal/umi/left/pose",
    "nmx/hal/umi/left/hand_state",
    "nmx/hal/umi/left/raw_hand_state",
    "nmx/hal/umi/left/slam",
    "nmx/hal/umi/left/extra",
    "nmx/hal/camera/camera_2/rgbd",
    "nmx/hal/umi/ego/pose",
    "nmx/hal/umi/ego/hand_state",
    "nmx/hal/umi/ego/raw_hand_state",
    "nmx/hal/umi/ego/slam",
    "nmx/hal/umi/ego/extra",
)

NEDF3_EXTRA_KEYS: Tuple[str, ...] = (
    "nmx/hal/umi/right/raw_hand_state",
    "nmx/hal/umi/right/slam",
    "nmx/hal/umi/right/extra",
    "nmx/hal/umi/left/raw_hand_state",
    "nmx/hal/umi/left/slam",
    "nmx/hal/umi/left/extra",
    "nmx/hal/umi/ego/raw_hand_state",
    "nmx/hal/umi/ego/slam",
    "nmx/hal/umi/ego/extra",
)

# ViveTracker row index → UMI side name (matches device row layout: left, right, ego).
_VIVE_ROW_TO_UMI_SIDE: Tuple[str, str, str] = ("left", "right", "ego")

HAL_CAMERA_TOPICS: Tuple[str, str, str] = (
    "nmx/hal/camera/camera_0/rgbd",
    "nmx/hal/camera/camera_1/rgbd",
    "nmx/hal/camera/camera_2/rgbd",
)

# Metadata ``camera_info`` keys are ``camera_0``..``camera_2``; values are fixed role names
# for device suffix N in ``WebCameraDevice_N`` / ``OpenCVCameraDevice_N`` / ``HikCameraDevice_N``.
NEDF3_CAMERA_ROLE_LABELS: Tuple[str, str, str] = (
    "right_wrist",
    "left_wrist",
    "head",
)

# Stage 1 ``extra.processed`` null placeholders (canonical lab_pro / nmx_processor).
_PROCESSED_EXTRA_FIELDS: Tuple[str, ...] = (
    "aligned_timestamp_ns",
    "alignment_mode",
    "downsample_kept",
    "aligned_pose",
    "aligned_hand_state",
    "aligned_raw_hand_state",
    "aligned_slam",
    "aligned_pose_tcp",
    "aligned_corrected_roi_marker_angle",
    "aligned_corrected_gripper_width",
    "aligned_arkit_diagnostics",
)

def _umi_hal_topic(side: str, kind: str) -> str:
    return f"nmx/hal/umi/{side}/{kind}"


def _pose7_to_umi_pose(row: np.ndarray) -> Any:
    """Build ``UmiPose`` from Vive ``[x,y,z,qx,qy,qz,qw]`` (nmx_msg Pose3 + UmiPose.pose)."""
    from nmx_msg.Quaternion_pb2 import Quaternion
    from nmx_msg.Umi_pb2 import UmiPose
    from nmx_msg.Vec_pb2 import Vec3

    x, y, z, qx, qy, qz, qw = np.asarray(row, dtype=np.float64).reshape(7)
    umi_pose = UmiPose()
    umi_pose.pose.pos.CopyFrom(
        Vec3(x=float(x), y=float(y), z=float(z))
    )
    umi_pose.pose.quat.CopyFrom(
        Quaternion(w=float(qw), x=float(qx), y=float(qy), z=float(qz))
    )
    return umi_pose


def _empty_umi_state() -> Any:
    """Empty ``UmiState`` (no joints) for unavailable hand/raw streams."""
    from nmx_msg.Umi_pb2 import UmiState

    return UmiState()


def _processed_gripper_angle_to_raw_hand_state(processed_angle_deg: float) -> Any:
    """Build ``UmiState`` with calibrated ``gripper.angle`` (0 = closed, increases with opening)."""
    from nmx_msg.Umi_pb2 import UmiState

    state = UmiState()
    joint = state.joints.add()
    joint.name = "gripper.angle"
    joint.pos = float(processed_angle_deg)
    return state


def _empty_slam_frame_exceptions() -> Any:
    """Empty SLAM payload: ``UmiSlamFrameExceptions`` with ``exceptions`` length 0."""
    try:
        from nmx_msg.UmiSlam_pb2 import UmiSlamFrameExceptions
    except ModuleNotFoundError:
        consumer_dir = os.path.dirname(os.path.abspath(__file__))
        if consumer_dir not in sys.path:
            sys.path.insert(0, consumer_dir)
        from _vendor.UmiSlam_pb2 import UmiSlamFrameExceptions

    return UmiSlamFrameExceptions()


def _empty_processed_extra_payload() -> Dict[str, Any]:
    return {key: None for key in _PROCESSED_EXTRA_FIELDS}


def _empty_extra_metadata_message() -> Any:
    """``extra`` HAL topic payload with Stage 1 ``processed`` null placeholders."""
    from nmx_msg.Metadata_pb2 import Metadata

    meta_msg = Metadata()
    meta_msg.version = "nedf3"
    payload = {"processed": _empty_processed_extra_payload()}
    meta_msg.metadata = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return meta_msg


def _build_encoded_image_message(
    data: bytes, rows: int, cols: int, channels: int, encoded_format: str
) -> Image:
    image = Image()
    image.data = data
    image.type = ImageDataType.ENCODED
    image.encoded_format = encoded_format
    image.rows = int(rows)
    image.cols = int(cols)
    image.channels = int(channels)
    return image


def _empty_depth_image_message() -> Image:
    """Canonical empty/invalid depth: zero dims, empty encoded_format, no payload."""
    return _build_encoded_image_message(b"", 0, 0, 0, "")


def _build_rgbd_from_jpeg(
    jpg_bytes: bytes, rows: int, cols: int, channels: int
) -> RGBD:
    """Wrap JPEG RGB bytes in ``RGBD`` with empty depth (no depth sensor on collect path)."""
    rgbd = RGBD()
    rgbd.rgb.CopyFrom(
        _build_encoded_image_message(jpg_bytes, rows, cols, channels, "jpeg")
    )
    rgbd.depth.CopyFrom(_empty_depth_image_message())
    if hasattr(rgbd, "depth_scale"):
        rgbd.depth_scale = 1.0
    return rgbd


try:
    from .data_saver import DataSaverConsumer as SyncDataSaverConsumer
    from .data_saver import wait_enter_blocking, enter_pressed_nonblocking
except ImportError:
    from data_saver import DataSaverConsumer as SyncDataSaverConsumer
    from data_saver import wait_enter_blocking, enter_pressed_nonblocking


def _normalize_metadata_extra_list(
    values: Optional[List[str]],
    default: Tuple[str, ...],
    field_name: str,
) -> List[str]:
    """Return exactly ``_METADATA_EXTRA_CAMERA_COUNT`` strings for metadata.extra."""
    source = list(default) if values is None else [str(v).strip() for v in values]
    if len(source) != _METADATA_EXTRA_CAMERA_COUNT:
        raise ValueError(
            f"{field_name} must have {_METADATA_EXTRA_CAMERA_COUNT} entries "
            f"(camera_0..camera_2), got {len(source)}"
        )
    if any(not item for item in source):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    return source


def parse_metadata_extra_csv(raw: str, field_name: str) -> List[str]:
    """Parse comma-separated metadata.extra list for three cameras."""
    parts = [part.strip() for part in raw.split(",")]
    return _normalize_metadata_extra_list(parts, DEFAULT_EXTRA_DEVICE_TYPES, field_name)


class DataSaverConsumer(SyncDataSaverConsumer):
    """
    Data saver consumer for storing device manager data to NEDF3 (UMI) format.

    This consumer saves data using NEDF SDK:
    - Camera devices: canonical ids ``camera_0``..``camera_2``; JPEG in HAL ``RGBD`` messages (no intrinsic/extrinsic MCAP topics)
    - Vive: rows of 7D → ``nmx/hal/umi/{left,right,ego}/pose``; per-frame empty ``hand_state`` /
      ``slam`` / ``extra`` when no hardware source (no fake zeros); empty ``raw_hand_state`` only
      for sides without a ``RotaryEncoderDevice``
    - RotaryEncoder: first/second configured encoder → left/right ``raw_hand_state`` (calibrated ``gripper.angle``);
      ``hand_state`` stays empty (Vive companion placeholders only)
    - Metadata: canonical 19-topic ``record_topic_list`` and ``extra_keys`` (lab_pro layout)

    Features:
    - Async writing for high-frequency data collection
    - Support for image resizing
    - NEDF3 metadata generation (no I2RT sidecar commands)
    """

    def __init__(self, summary_shm_name: str = "device_summary_data",
                 output_dir: str = "saved_data", save_interval: float = 0.04,
                 start_idx: int = 0, image_size: Optional[Tuple[int, int]] = None,
                 wrist_crop_aspect_hw: Optional[Tuple[int, int]] = (4, 3),
                 wrist_output_size: Optional[Tuple[int, int]] = (480, 640),
                 queue_size: int = 5000,
                 robot_type: str = "UMI", task_name: str = "data_collection",
                 end_effector_vendor_model: str = "",
                 extra_device_types: Optional[List[str]] = None,
                 extra_hardware_strings: Optional[List[str]] = None,
                 codebase_version: str = "",
                 extra_device_zero: str = "",
                 dm3_user_id: str = "",
                 dm3_user_name: str = "",
                 task_description: str = "",
                 enable_anomaly_detection: bool = True,
                 anomaly_config_path: Optional[str] = None) -> None:
        """
        Initialize the data saver consumer.

        Args:
            summary_shm_name: Name of the summary shared memory
            output_dir: Directory to save data files
            save_interval: Interval between saves in seconds
            start_idx: Starting episode index
            image_size: Target image size (width, height) for resizing all cameras
            wrist_crop_aspect_hw: Height:width aspect for left/right wrist cameras.
                Default (4, 3) is portrait 4:3 (height > width). None disables crop/pad.
            wrist_output_size: Output (width, height) for wrist cameras after crop.
                Default 480x640 matches head pixel count (640x480 h×w), portrait 4:3.
            queue_size: Size of async write queue (default: 1000)
            robot_type: Value for NEDF ``robot_vendor_model`` (e.g. UMI or Vendor_Model)
            task_name: Task name for NEDF metadata
            end_effector_vendor_model: NEDF3 ``end_effector_vendor_model`` (VendorName_ModelName per spec)
            extra_device_types: ``metadata.extra.device_type`` per camera (camera_0..2)
            extra_hardware_strings: ``metadata.extra.hardware_string`` per camera (camera_0..2)
            codebase_version: ``metadata.extra.codebase_version`` (empty by default)
            extra_device_zero: ``metadata.extra.device[0].identifier`` from GUI
            dm3_user_id: ``metadata.extra.user_id`` from DM3 login
            dm3_user_name: ``metadata.extra.user_name`` from DM3 login
            task_description: NEDF ``task_description`` from selected DM3 task
            enable_anomaly_detection: Whether to enable episode anomaly detection
            anomaly_config_path: Path to anomaly detection config file
        """
        super().__init__(summary_shm_name, output_dir, save_interval, start_idx, image_size)
        self.wrist_crop_aspect_hw = wrist_crop_aspect_hw
        self.wrist_output_size = wrist_output_size
        self.extra_device_types = _normalize_metadata_extra_list(
            extra_device_types, DEFAULT_EXTRA_DEVICE_TYPES, "device_type"
        )
        self.extra_hardware_strings = _normalize_metadata_extra_list(
            extra_hardware_strings, DEFAULT_EXTRA_HARDWARE_STRINGS, "hardware_string"
        )
        self.codebase_version = codebase_version
        self.extra_device_zero = extra_device_zero.strip()
        self.dm3_user_id = dm3_user_id.strip()
        self.dm3_user_name = dm3_user_name.strip()
        self.task_description = normalize_task_description(task_description)
        self._maintenance_request_path = Path(self.output_dir).expanduser().resolve().parent / MAINTENANCE_REQUEST_NAME
        self._maintenance_done_path = Path(self.output_dir).expanduser().resolve().parent / MAINTENANCE_DONE_NAME

        # Async writing infrastructure
        self.write_queue = queue.Queue(maxsize=queue_size)
        self.write_thread = None
        self.write_thread_running = False

        # NEDF3 writer management
        self.nedf_writer = None
        self.nedf_writer_context = None  # Store the context manager
        self.topics_config: Optional[NEDFTopicsConfig] = None
        self.current_episode_dir = None
        self.episode_active = False
        self.device_data_counts: Dict[str, int] = {}

        # NEDF metadata configuration
        self.robot_type = robot_type
        self.task_name = task_name
        self.end_effector_vendor_model = end_effector_vendor_model
        self.umi_to_tcp_transform: List[float] = []

        # Camera ID mapping (device_key -> camera_id)
        self.camera_id_mapping: Dict[str, str] = {}
        # Camera topic mapping (camera_id -> image topic)
        self.camera_image_topic_mapping: Dict[str, str] = {}

        # Episode anomaly detection
        self.enable_anomaly_detection = enable_anomaly_detection
        self.anomaly_config_path = anomaly_config_path
        self.anomaly_config_loaded_path = ""
        self.anomaly_detector = None
        self.anomaly_stats = {
            'total_episodes_checked': 0,
            'total_anomalies': 0,
            'episodes_with_errors': 0,
            'last_check_time': 0
        }
        self._episode_analysis_data: Dict[str, List[np.ndarray]] = {}
        self._episode_analysis_timestamps_ns: Dict[str, List[int]] = {}
        self._episode_image_black_flags: Dict[str, List[bool]] = {}
        self._episode_analysis_context_valid = True
        self._episode_tracker_status_start: Dict[str, Any] = {}

        self.episode_running = False
        self._discard_keyboard_pressed = False
        self._gripper_angle_calibrations: Dict[str, GripperAngleCalibration] = {}
        self._gripper_calibration_date: str = date.today().isoformat()
        self._rotary_encoder_umi_sides: Set[str] = set()
        self._direct_device_shms: Dict[str, Any] = {}
        self._jump_segment_cleanup_date = ""
        logger.info("Direct device SHM saver is enabled for all device streams")

        # Keyboard listener for manual episode discard (pressing 'd')
        self.keyboard_listener = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load anomaly detection configuration
        if self.enable_anomaly_detection:
            self._load_anomaly_config()
            if self.enable_anomaly_detection:
                self._cleanup_old_anomaly_jump_segments()

    def _build_nedf_metadata_extra(self) -> Dict[str, Any]:
        """Build metadata.extra payload written into each episode metadata.json."""
        return {
            "device_type": list(self.extra_device_types),
            "hardware_string": list(self.extra_hardware_strings),
            "codebase_version": self.codebase_version,
            "device": build_extra_device_entries(self.extra_device_zero),
            "user_id": self.dm3_user_id,
            "user_name": self.dm3_user_name,
        }

    def _load_anomaly_config(self) -> None:
        """Load anomaly detection configuration from file."""
        try:
            if self.anomaly_config_path and os.path.exists(self.anomaly_config_path):
                config_path = self.anomaly_config_path
            else:
                # Try default config path
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                config_path = os.path.join(project_root, "configs/anomaly_config/default.yaml")

                if not os.path.exists(config_path):
                    logger.error("No anomaly detection config file found, disabling anomaly detection")
                    self.enable_anomaly_detection = False
                    return

            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                self.anomaly_config = full_config.get('anomaly_detection', {})

            if not self.anomaly_config:
                logger.warning("No 'anomaly_detection' section found in config file, disabling anomaly detection")
                self.enable_anomaly_detection = False
                return

            # Initialize anomaly detector
            self.anomaly_detector = EpisodeAnomalyDetector(self.anomaly_config)
            self.anomaly_config_loaded_path = str(config_path)
            logger.info(f"Loaded anomaly detection config from: {config_path}")

        except Exception as e:
            logger.error(f"Error loading anomaly detection config: {e}")
            logger.warning("Disabling anomaly detection due to config error")
            self.enable_anomaly_detection = False

    def _reset_episode_analysis_context(self) -> None:
        self._episode_analysis_data = {}
        self._episode_analysis_timestamps_ns = {}
        self._episode_image_black_flags = {}
        self._episode_analysis_context_valid = True

    def _append_episode_analysis_sample(
        self,
        device_key: str,
        timestamp_ns: int,
        data: Optional[np.ndarray] = None,
        *,
        image_black: Optional[bool] = None,
    ) -> None:
        self._episode_analysis_timestamps_ns.setdefault(device_key, []).append(
            int(timestamp_ns)
        )
        if data is not None:
            self._episode_analysis_data.setdefault(device_key, []).append(data)
        if image_black is not None:
            self._episode_image_black_flags.setdefault(device_key, []).append(
                bool(image_black)
            )

    def _build_episode_analysis_context(
        self,
        feat_mapping: Dict[str, str],
    ) -> Optional[EpisodeAnalysisContext]:
        if not self._episode_analysis_context_valid:
            logger.warning(
                "In-memory anomaly context was invalidated; falling back to NEDF3 reader"
            )
            return None
        expected_device_keys = set(feat_mapping.values())
        missing = sorted(
            device_key
            for device_key in expected_device_keys
            if not self._episode_analysis_timestamps_ns.get(device_key)
        )
        if missing:
            logger.warning(
                "In-memory anomaly context is incomplete; falling back to NEDF3 "
                f"reader for devices={missing}"
            )
            return None

        all_timestamps = [
            timestamp
            for timestamps in self._episode_analysis_timestamps_ns.values()
            for timestamp in timestamps
        ]
        if not all_timestamps:
            return None
        min_timestamp_ns = min(all_timestamps)

        data_dict: Dict[str, Any] = {}
        timestamp_dict: Dict[str, np.ndarray] = {}
        image_black_flags: Dict[str, np.ndarray] = {}
        for device_key, timestamps in self._episode_analysis_timestamps_ns.items():
            timestamp_values = np.asarray(timestamps, dtype=np.int64)
            timestamp_dict[device_key] = (
                timestamp_values - min_timestamp_ns
            ).astype(np.float64) / 1e9
            flags = self._episode_image_black_flags.get(device_key)
            if flags is not None:
                image_black_flags[device_key] = np.asarray(flags, dtype=bool)
                data_dict[device_key] = np.empty(len(flags), dtype=np.uint8)
                continue
            samples = self._episode_analysis_data.get(device_key, [])
            if samples:
                data_dict[device_key] = np.asarray(samples, dtype=np.float32)

        logger.info(
            "[anomaly_memory_context] "
            + " ".join(
                f"{device_key}={len(timestamps)}"
                for device_key, timestamps in sorted(
                    self._episode_analysis_timestamps_ns.items()
                )
            )
        )
        return EpisodeAnalysisContext(
            data_dict=data_dict,
            timestamp_dict=timestamp_dict,
            image_black_flags=image_black_flags,
        )

    def _episode_analysis_time_range_ns(self) -> Optional[Tuple[int, int]]:
        if not self._episode_analysis_context_valid:
            return None
        timestamps = [
            timestamp
            for device_timestamps in self._episode_analysis_timestamps_ns.values()
            for timestamp in device_timestamps
        ]
        if not timestamps:
            return None
        return min(timestamps), max(timestamps)

    def _detect_episode_anomalies(
        self,
        episode_dir: str,
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, str]]:
        """
        Detect anomalies in the saved episode.

        Args:
            episode_dir: Path to the episode directory

        Returns:
            Tuple of (has_errors, anomalies, feature mapping).
        """
        if not self.enable_anomaly_detection or not self.anomaly_detector:
            return False, [], {}

        try:
            episode_path = Path(episode_dir)
            episode_display_name = episode_path.name.removesuffix("_temp")

            # Create feature mapping from device information
            feat_mapping = {}
            camera_position = 0
            encoder_position = 0
            for device_key, device_info in self.devices.items():
                device_type = device_info['type']

                # Map device types to feature keys
                if device_type in ['OpenCVCameraDevice', 'HikCameraDevice', 'WebCameraDevice']:
                    # Dataset roles follow camera order, not global logical IDs.
                    cam_idx = camera_position
                    camera_position += 1
                    if cam_idx == 0:
                        feat_mapping["img_right"] = device_key
                    elif cam_idx == 1:
                        feat_mapping["img_left"] = device_key
                    elif cam_idx == 2:
                        feat_mapping["img_ego"] = device_key
                elif device_type in ['ViveTrackerDevice']:
                    feat_mapping['pose_unified'] = device_key
                elif device_type in ['RotaryEncoderDevice']:
                    dev_id = encoder_position
                    encoder_position += 1
                    if dev_id == 0:
                        feat_mapping['grip_left'] = device_key
                    elif dev_id == 1:
                        feat_mapping['grip_right'] = device_key

            try:
                analysis_context = self._build_episode_analysis_context(feat_mapping)
            except Exception as context_error:
                logger.warning(
                    "Could not build in-memory anomaly context; falling back to "
                    f"NEDF3 reader: {context_error}"
                )
                analysis_context = None
            # Run anomaly detection. The in-memory context mirrors successful
            # writer calls; the detector falls back to NEDF3 reads when absent.
            anomalies, has_errors = self.anomaly_detector.detect_episode_anomalies(
                episode_path,
                feat_mapping,
                analysis_context=analysis_context,
            )
            tracker_episode_status = self._tracker_episode_status_summary(
                self._episode_tracker_status_start,
                self._read_tracker_status_snapshot(),
            )

            # Update statistics
            self.anomaly_stats['total_episodes_checked'] += 1
            self.anomaly_stats['total_anomalies'] += len(anomalies)
            self.anomaly_stats['last_check_time'] = time.time()

            if has_errors:
                self.anomaly_stats['episodes_with_errors'] += 1

            anomaly_status = build_anomaly_status(
                enabled=self.enable_anomaly_detection,
                episode_dir=episode_path,
                anomalies=anomalies,
                has_errors=has_errors,
                discarded=has_errors,
                stats=self.anomaly_stats,
                config_path=self.anomaly_config_loaded_path or self.anomaly_config_path,
                feat_mapping=feat_mapping,
                tracker_status=tracker_episode_status,
            )
            try:
                publish_anomaly_status(anomaly_status)
            except Exception as publish_error:
                logger.warning(f"Failed to publish anomaly status: {publish_error}")

            # Log anomalies
            if anomalies:
                logger.warning(
                    f"🔍 Episode {episode_display_name}: Found {len(anomalies)} anomalies"
                )

                errors = [a for a in anomalies if a['severity'] == 'error']
                warnings = [a for a in anomalies if a['severity'] == 'warning']

                if errors:
                    logger.error(f"  ❌ {len(errors)} errors detected:")
                    for error in errors:
                        logger.error(f"    - {error['message']}")

                if warnings:
                    logger.warning(f"  ⚠️  {len(warnings)} warnings detected:")
                    for warning in warnings:
                        logger.warning(f"    - {warning['message']}")
                if feat_mapping.get("pose_unified") and tracker_episode_status.get("available"):
                    untracked = int(tracker_episode_status.get("untracked_delta") or 0)
                    role_delta = tracker_episode_status.get("untracked_roles_delta") or {}
                    if untracked:
                        logger.warning(
                            "  Vive tracker untracked during episode: "
                            f"{untracked} frames, roles={role_delta}"
                        )
                    else:
                        logger.warning(
                            "  Vive tracker untracked during episode: 0 "
                            "(pose warnings are from written tracker samples)"
                        )
            else:
                logger.info(f"✅ Episode {episode_display_name}: No anomalies detected")

            return has_errors, anomalies, feat_mapping

        except Exception as e:
            logger.error(f"Error detecting anomalies in episode {episode_dir}: {e}")
            error_status = build_anomaly_status(
                enabled=self.enable_anomaly_detection,
                episode_dir=Path(episode_dir),
                anomalies=[],
                has_errors=True,
                discarded=True,
                stats=self.anomaly_stats,
                config_path=self.anomaly_config_loaded_path or self.anomaly_config_path,
                tracker_status=self._tracker_episode_status_summary(
                    self._episode_tracker_status_start,
                    self._read_tracker_status_snapshot(),
                ),
                error=str(e),
            )
            try:
                publish_anomaly_status(error_status)
            except Exception as publish_error:
                logger.warning(f"Failed to publish anomaly status: {publish_error}")
            import traceback
            traceback.print_exc()
            return True, [], {}

    def _anomaly_segment_output_root(self) -> Path:
        segment_config = self.anomaly_config.get("jump_segment_export", {})
        if not isinstance(segment_config, dict):
            segment_config = {}
        configured_root = segment_config.get("output_root")
        if configured_root:
            configured_path = Path(str(configured_root)).expanduser()
            if configured_path.is_absolute():
                return configured_path
            return Path(PROJECT_ROOT) / configured_path
        return Path(PROJECT_ROOT) / "logs" / "anomaly_jump_segments"

    def _anomaly_segment_export_config(self) -> JumpSegmentExportConfig:
        segment_config = self.anomaly_config.get("jump_segment_export", {})
        if not isinstance(segment_config, dict):
            segment_config = {}
        return JumpSegmentExportConfig(
            output_root=self._anomaly_segment_output_root(),
            pre_seconds=float(segment_config.get("pre_seconds", 2.0)),
            post_seconds=float(segment_config.get("post_seconds", 2.0)),
            max_camera_frames=int(segment_config.get("max_camera_frames", 7)),
            max_image_width=int(segment_config.get("max_image_width", 320)),
            max_segments_per_episode=int(segment_config.get("max_segments_per_episode", 20)),
        )

    def _cleanup_old_anomaly_jump_segments(self) -> None:
        """Run cleanup once per local date, including after a midnight rollover."""
        today = date.today()
        today_text = today.isoformat()
        if self._jump_segment_cleanup_date == today_text:
            return
        self._jump_segment_cleanup_date = today_text
        try:
            result = cleanup_jump_segments_before_today(
                self._anomaly_segment_output_root(),
                today=today,
            )
        except Exception as exc:
            logger.warning(f"Anomaly jump clip cleanup skipped: {exc}")
            return
        if result.removed_episode_count or result.failed_episode_count:
            removed_mb = result.removed_bytes / (1024 * 1024)
            logger.info(
                "Anomaly jump clip cleanup: "
                f"removed_episodes={result.removed_episode_count}, "
                f"removed_mb={removed_mb:.2f}, "
                f"failed={result.failed_episode_count}"
            )

    def _export_anomaly_jump_segments(
        self,
        episode_dir: str,
        anomalies: List[Dict[str, Any]],
        feat_mapping: Dict[str, str],
    ) -> bool:
        segment_config = self.anomaly_config.get("jump_segment_export", {})
        if isinstance(segment_config, dict) and not bool(segment_config.get("enabled", True)):
            logger.info("Anomaly jump segment export disabled by config")
            return True
        try:
            cached_data = None
            cached_timestamps = None
            if self.anomaly_detector is not None:
                loaded_data, loaded_timestamps = self.anomaly_detector.last_loaded_data()
                if loaded_data and loaded_timestamps:
                    cached_data = loaded_data
                    cached_timestamps = loaded_timestamps
            created_dirs = export_anomaly_jump_segments(
                Path(episode_dir),
                anomalies=anomalies,
                feat_mapping=feat_mapping,
                anomaly_config=self.anomaly_config,
                export_config=self._anomaly_segment_export_config(),
                data_dict=cached_data,
                timestamp_dict=cached_timestamps,
            )
            if created_dirs:
                logger.info(
                    f"Saved {len(created_dirs)} anomaly jump segment(s) before discarding episode"
                )
            return True
        except Exception as exc:
            logger.error(f"Failed to export anomaly jump segments for {episode_dir}: {exc}")
            return False

    def _quarantine_failed_anomaly_export(self, episode_dir: str, reason: str) -> bool:
        segment_config = self.anomaly_config.get("jump_segment_export", {})
        fallback_enabled = True
        if isinstance(segment_config, dict):
            fallback_enabled = bool(segment_config.get("quarantine_on_export_failure", True))
        if not fallback_enabled:
            return False
        try:
            quarantine_episode(
                Path(episode_dir),
                self._anomaly_segment_output_root(),
                reason=reason,
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to quarantine anomaly episode {episode_dir}: {exc}")
            return False

    def _setup_keyboard_listener(self) -> None:
        """Setup keyboard listener for manual episode discard using 'd' key."""
        if os.environ.get("SUPERINFERENCE_ENTER_CONTROL_FILE"):
            logger.info("Keyboard discard listener disabled for file-controlled automation")
            return
        try:

            def on_press(key: Any) -> None:
                try:
                    if hasattr(key, 'char') and key.char == 'd' and self.episode_running:
                        self._discard_keyboard_pressed = True
                        logger.info("'D' pressed - discarding current episode")
                except AttributeError:
                    pass

            self.keyboard_listener = pynput_keyboard.Listener(on_press=on_press)
            self.keyboard_listener.start()
            logger.info("Keyboard listener started. Press 'D' to discard current episode")
        except Exception as e:
            logger.error(f"Failed to setup keyboard listener: {e}")
            self.keyboard_listener = None

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
                logger.info("Keyboard listener stopped")
            except Exception as e:
                logger.warning(f"Failed to stop keyboard listener: {e}")

    def _has_rotary_encoder_devices(self) -> bool:
        return any(
            device_info.get("type") == "RotaryEncoderDevice"
            for device_info in self.devices.values()
        )

    def _umi_sides_with_rotary_encoders(self) -> Set[str]:
        return set(self._rotary_encoder_sides().values())

    def _rotary_encoder_sides(self) -> Dict[str, str]:
        keys = [key for key, info in self.devices.items() if info.get("type") == "RotaryEncoderDevice"]
        return dict(zip(keys, ("left", "right")))

    def _load_required_gripper_calibrations(self) -> Dict[str, GripperAngleCalibration]:
        if not self._has_rotary_encoder_devices():
            return {}
        self._gripper_calibration_date = date.today().isoformat()
        return require_today_calibrations_for_both_arms(PROJECT_ROOT)

    def open_episode_streams(self, episode_dir: str) -> None:
        """
        Open NEDF3 writer for the current episode.

        Args:
            episode_dir: Directory path for this episode
        """
        try:
            self._gripper_angle_calibrations = self._load_required_gripper_calibrations()
            self._rotary_encoder_umi_sides = self._umi_sides_with_rotary_encoders()
            os.makedirs(episode_dir, exist_ok=True)
            self.current_episode_dir = episode_dir
            self.topics_config = NEDFTopicsConfig()

            # Build camera_info: keys camera_0..camera_2, values role labels; map device_key -> canonical id
            from utils.recording_device_roles import nedf_camera_entries

            camera_entries = nedf_camera_entries(
                self.devices, {"OpenCVCameraDevice", "HikCameraDevice", "WebCameraDevice"}
            )
            camera_info: Dict[str, str] = {}
            for cam_idx, device_key, device_info in camera_entries:
                canonical_id = f"camera_{cam_idx}"
                if cam_idx < len(NEDF3_CAMERA_ROLE_LABELS):
                    role_label = NEDF3_CAMERA_ROLE_LABELS[cam_idx]
                else:
                    role_label = str(device_info.get("name", canonical_id))
                camera_info[canonical_id] = role_label
                self.camera_id_mapping[device_key] = canonical_id
                self.camera_image_topic_mapping[canonical_id] = (
                    self._resolve_camera_image_topic(canonical_id)
                )
                logger.debug(
                    f"Registered camera: {device_key} -> {canonical_id} ({role_label}), "
                    f"image_topic={self.camera_image_topic_mapping[canonical_id]}"
                )

            main_camera = (
                "camera_0"
                if "camera_0" in camera_info
                else (next(iter(camera_info)) if camera_info else "CAMID_1")
            )
            n_cam = len(camera_info)
            arm_layout = "dual_arm" if n_cam >= 2 else "single_arm"

            record_topic_list = list(NEDF3_RECORD_TOPIC_LIST)

            metadata = NEDFMetadata(
                hardware_config_name="",
                task_name=self.task_name,
                task_description=self.task_description,
                robot_vendor_model=self.robot_type,
                data_format="NEDF3",
                mcap_file_name="data.mcap",
                mcap_file_type="CHUNK_FILE",
                collection_mode="robopocket_pro",
                end_effector="gripper",
                end_effector_vendor_model=self.end_effector_vendor_model,
                robot_type=arm_layout,
                arm_base_to_world_transform=[],
                umi_to_tcp_transform=self.umi_to_tcp_transform,
                camera_info=camera_info,
                main_camera=main_camera,
                extra_keys=list(NEDF3_EXTRA_KEYS),
                extra=self._build_nedf_metadata_extra(),
                record_topic_list=record_topic_list,
            )

            metadata_path = os.path.join(episode_dir, "metadata.json")
            writer_config = NEDFWriterConfig(
                metadata_file_path=Path(metadata_path),
                metadata=metadata,
                topics_config=self.topics_config,
                max_file_size_mb=100,
                mcap_chunk_size_kb=256,
            )

            self.nedf_writer_context = NEDFFactory.get_writer(NEDFType.NEDF3, writer_config)
            self.nedf_writer = self.nedf_writer_context.__enter__()
            self._reset_episode_analysis_context()

            for device_key in self.devices:
                self.device_data_counts[device_key] = 0

            self.episode_active = True
            logger.info(f"Opened NEDF3 writer in {episode_dir}")
            logger.info(f"Cameras: {list(camera_info.keys())}")
            logger.info(f"record_topic_list ({len(record_topic_list)}): {record_topic_list}")

        except Exception as e:
            logger.error(f"Error opening NEDF3 writer: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close_episode_streams(self) -> None:
        """Close NEDF3 writer."""
        try:
            if self.nedf_writer_context is not None:
                self.nedf_writer_context.__exit__(None, None, None)
                self.nedf_writer = None
                self.nedf_writer_context = None
                self.topics_config = None

                for device_key, count in self.device_data_counts.items():
                    logger.debug(f"Wrote {count} frames for {device_key}")

                logger.info("Closed NEDF3 writer")

            self.device_data_counts.clear()
            self.camera_id_mapping.clear()
            self.camera_image_topic_mapping.clear()
            self.episode_active = False

        except Exception as e:
            logger.error(f"Error closing NEDF3 writer: {e}")
            import traceback
            traceback.print_exc()

    def _write_canonical_umi_companion_topics(
        self, side: str, timestamp_ns: int
    ) -> None:
        """Write empty/null companion UMI topics for one side (no fake numeric placeholders)."""
        writer = self.nedf_writer
        if writer is None:
            return
        writer.write_message(
            _umi_hal_topic(side, "hand_state"),
            _empty_umi_state(),
            timestamp_ns,
            timestamp_ns,
        )
        if side not in self._rotary_encoder_umi_sides:
            writer.write_message(
                _umi_hal_topic(side, "raw_hand_state"),
                _empty_umi_state(),
                timestamp_ns,
                timestamp_ns,
            )
        writer.write_message(
            _umi_hal_topic(side, "slam"),
            _empty_slam_frame_exceptions(),
            timestamp_ns,
            timestamp_ns,
        )
        writer.write_message(
            _umi_hal_topic(side, "extra"),
            _empty_extra_metadata_message(),
            timestamp_ns,
            timestamp_ns,
        )

    def _write_canonical_umi_pose_bundle(
        self, side: str, pose_row: np.ndarray, timestamp_ns: int
    ) -> None:
        """Write real ``pose`` plus empty companion topics for this timestamp."""
        writer = self.nedf_writer
        if writer is None:
            return
        writer.write_message(
            _umi_hal_topic(side, "pose"),
            _pose7_to_umi_pose(pose_row),
            timestamp_ns,
            timestamp_ns,
        )
        self._write_canonical_umi_companion_topics(side, timestamp_ns)

    def _transform_camera_image(self, data_array: np.ndarray, camera_id: str) -> np.ndarray:
        """Apply wrist crop/pad and optional global resize before encoding."""
        original_shape = data_array.shape

        if self._is_wrist_camera_id(camera_id) and self.wrist_crop_aspect_hw is not None:
            aspect_h, aspect_w = self.wrist_crop_aspect_hw
            target_width: Optional[int] = None
            target_height: Optional[int] = None
            if self.wrist_output_size is not None:
                target_width, target_height = self.wrist_output_size
            data_array = crop_pad_to_aspect_and_size(
                data_array,
                aspect_h=aspect_h,
                aspect_w=aspect_w,
                target_width=target_width,
                target_height=target_height,
            )

        if self.image_size is not None:
            target_width, target_height = self.image_size
            data_array = cv2.resize(
                data_array,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )

        if data_array.shape[:2] != original_shape[:2]:
            logger.debug(
                f"Camera {camera_id}: transformed image from {original_shape[:2]} "
                f"to {data_array.shape[:2]}"
            )
        return data_array

    def _async_write_worker(self) -> None:
        """
        Background worker thread that continuously consumes data from write queue.

        This thread runs independently, writing data to NEDF3 as it becomes available.
        """
        logger.info("Async write worker thread started")

        while self.write_thread_running:
            write_task = None
            try:
                # Block for up to 0.1s waiting for data
                write_task = self.write_queue.get(timeout=0.1)

                # Extract task data
                device_key = write_task['device_key']
                device_type = write_task['device_type']
                timestamp_ns = write_task['timestamp_ns']
                data_array = write_task['data_array']

                # Check if episode is active and writer is open
                if not self.episode_active or self.nedf_writer is None or self.topics_config is None:
                    logger.warning(f"Episode not active or writer not open for {device_key}, dropping data")
                    self.write_queue.task_done()
                    continue

                # Process based on device type
                if device_type in ['OpenCVCameraDevice', 'HikCameraDevice', 'WebCameraDevice']:
                    # Handle camera data
                    camera_id = self.camera_id_mapping.get(device_key)
                    if camera_id is None:
                        logger.warning(f"Camera ID not found for {device_key}")
                        self.write_queue.task_done()
                        continue

                    data_array = self._transform_camera_image(data_array, camera_id)

                    # Convert RGB to BGR for OpenCV encoding
                    # Images from devices are stored in RGB format, but cv2.imencode expects BGR
                    if len(data_array.shape) == 3 and data_array.shape[2] == 3:
                        data_array = cv2.cvtColor(data_array, cv2.COLOR_RGB2BGR)

                    # Encode image as JPG
                    _, jpg_bytes = cv2.imencode('.jpg', data_array, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    rows = int(data_array.shape[0])
                    cols = int(data_array.shape[1])
                    channels = int(data_array.shape[2]) if len(data_array.shape) == 3 else 1
                    rgbd_msg = _build_rgbd_from_jpeg(
                        jpg_bytes.tobytes(), rows, cols, channels
                    )

                    rgbd_topic = self.camera_image_topic_mapping.get(camera_id)
                    if not rgbd_topic:
                        rgbd_topic = self._resolve_camera_image_topic(camera_id)
                    self.nedf_writer.write_message(
                        rgbd_topic, rgbd_msg, timestamp_ns, timestamp_ns
                    )
                    try:
                        image_black = bool(
                            int(jpg_bytes.size) < 1000 or np.max(data_array) < 5
                        )
                        self._append_episode_analysis_sample(
                            device_key,
                            timestamp_ns,
                            image_black=image_black,
                        )
                    except Exception as exc:
                        self._episode_analysis_context_valid = False
                        logger.warning(
                            f"Could not cache camera anomaly sample for {device_key}: {exc}"
                        )

                elif device_type == "ViveTrackerDevice":
                    # (N, 7) rows: [0]→left, [1]→right, [2]→ego; pose + empty companion topics.
                    flat = np.asarray(data_array, dtype=np.float64).reshape(-1)
                    n = flat.size // 7
                    if n <= 0:
                        logger.warning(
                            f"ViveTrackerDevice {device_key}: need at least 7 floats, got {flat.size}"
                        )
                        self.write_queue.task_done()
                        continue
                    if flat.size % 7 != 0:
                        logger.debug(
                            f"ViveTrackerDevice {device_key}: length {flat.size} not a multiple of 7, truncating"
                        )
                    n_write = min(n, len(_VIVE_ROW_TO_UMI_SIDE))
                    mat = flat[: n_write * 7].reshape(n_write, 7)
                    for i in range(n_write):
                        side = _VIVE_ROW_TO_UMI_SIDE[i]
                        self._write_canonical_umi_pose_bundle(
                            side, mat[i], timestamp_ns
                        )
                    try:
                        self._append_episode_analysis_sample(
                            device_key,
                            timestamp_ns,
                            np.asarray(mat[:n_write], dtype=np.float32).copy(),
                        )
                    except Exception as exc:
                        self._episode_analysis_context_valid = False
                        logger.warning(
                            f"Could not cache pose anomaly sample for {device_key}: {exc}"
                        )

                elif device_type == "RotaryEncoderDevice":
                    device_info = write_task.get("device_info") or self.devices.get(
                        device_key, {}
                    )
                    side = self._rotary_encoder_sides().get(device_key)
                    if side is None:
                        logger.warning(
                            f"RotaryEncoderDevice {device_key}: no left/right role in configured encoder order; skipping"
                        )
                        self.write_queue.task_done()
                        continue

                    angle_vals = np.asarray(data_array, dtype=np.float64).reshape(-1)
                    if angle_vals.size < 1:
                        logger.warning(
                            f"RotaryEncoderDevice {device_key}: empty angle sample, skipping"
                        )
                        self.write_queue.task_done()
                        continue

                    angle_deg = float(angle_vals[0])
                    calibration = self._gripper_angle_calibrations.get(side)
                    if calibration is None:
                        raise RuntimeError(
                            f"Missing today's angle calibration for {side} gripper; "
                            "run GUI calibration before collecting."
                        )
                    processed_angle_deg = process_gripper_angle_with_date_check(
                        angle_deg,
                        calibration,
                        self._gripper_calibration_date,
                    )
                    self.nedf_writer.write_message(
                        _umi_hal_topic(side, "raw_hand_state"),
                        _processed_gripper_angle_to_raw_hand_state(processed_angle_deg),
                        timestamp_ns,
                        timestamp_ns,
                    )
                    try:
                        self._append_episode_analysis_sample(
                            device_key,
                            timestamp_ns,
                            np.asarray([processed_angle_deg], dtype=np.float32),
                        )
                    except Exception as exc:
                        self._episode_analysis_context_valid = False
                        logger.warning(
                            f"Could not cache gripper anomaly sample for {device_key}: {exc}"
                        )

                else:
                    logger.warning(f"Unknown device type: {device_type} for {device_key}")
                    # Still mark task as done to avoid queue blocking
                    self.write_queue.task_done()
                    continue

                # Update counter (initialize if not exists)
                if device_key not in self.device_data_counts:
                    self.device_data_counts[device_key] = 0
                self.device_data_counts[device_key] += 1

                # Log progress periodically
                if self.device_data_counts[device_key] % 100 == 0:
                    logger.debug(f"{device_key}: wrote {self.device_data_counts[device_key]} frames")

                self.write_queue.task_done()

            except queue.Empty:
                # No data available, continue waiting
                continue
            except Exception as e:
                if write_task is not None:
                    self.write_queue.task_done()
                logger.error(f"Error in async write worker: {e}")
                import traceback
                traceback.print_exc()

        logger.info("Async write worker thread stopped")

    @staticmethod
    def _camera_index_from_device_key(device_key: str) -> Optional[int]:
        """Parse trailing integer from e.g. ``WebCameraDevice_0`` or ``camera_0`` -> ``0``."""
        try:
            return int(str(device_key).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return None

    def _read_tracker_status_snapshot(self) -> Dict[str, Any]:
        path = Path(os.environ.get("SI_TRACKER_STATUS_PATH") or DEFAULT_TRACKER_STATUS_PATH)
        if not path.is_file():
            return {"available": False, "status_path": str(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"available": False, "status_path": str(path), "error": str(exc)}
        if not isinstance(payload, dict):
            return {"available": False, "status_path": str(path), "error": "status payload is not an object"}
        payload = dict(payload)
        payload["available"] = True
        payload["status_path"] = str(path)
        return payload

    @staticmethod
    def _tracker_int(payload: Dict[str, Any], key: str) -> int:
        try:
            return int(payload.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _tracker_role_counts(payload: Dict[str, Any]) -> Dict[str, int]:
        raw = payload.get("untracked_roles_total")
        if not isinstance(raw, dict):
            raw = payload.get("skipped_untracked_roles_total")
        if not isinstance(raw, dict):
            return {}
        counts: Dict[str, int] = {}
        for role, value in raw.items():
            try:
                counts[str(role)] = int(value or 0)
            except (TypeError, ValueError):
                counts[str(role)] = 0
        return counts

    def _tracker_episode_status_summary(
        self,
        start_status: Dict[str, Any],
        end_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_total = self._tracker_int(start_status, "untracked_total")
        if not start_total:
            start_total = self._tracker_int(start_status, "skipped_untracked_total")
        end_total = self._tracker_int(end_status, "untracked_total")
        if not end_total:
            end_total = self._tracker_int(end_status, "skipped_untracked_total")
        role_start = self._tracker_role_counts(start_status)
        role_end = self._tracker_role_counts(end_status)
        role_delta: Dict[str, int] = {}
        for role in sorted(set(role_start) | set(role_end)):
            delta = max(0, role_end.get(role, 0) - role_start.get(role, 0))
            if delta:
                role_delta[role] = delta
        return {
            "available": bool(end_status.get("available")),
            "status_path": end_status.get("status_path", ""),
            "start_updated_at": start_status.get("updated_at", ""),
            "end_updated_at": end_status.get("updated_at", ""),
            "untracked_start": start_total,
            "untracked_end": end_total,
            "untracked_delta": max(0, end_total - start_total),
            "untracked_occurred": end_total > start_total,
            "untracked_roles_delta": role_delta,
            "last_untracked_roles": end_status.get(
                "last_untracked_roles",
                end_status.get("last_missing_roles", []),
            ),
            "last_untracked_at": end_status.get("last_untracked_at", ""),
            "pose_quality_reason": end_status.get(
                "pose_quality_reason",
                end_status.get("pose_hz_low_reason", ""),
            ),
            "frame_loop_hz": end_status.get("frame_loop_hz"),
            "write_hz": end_status.get("write_hz"),
            "focused_write_hz": end_status.get("focused_write_hz"),
        }

    @classmethod
    def _is_wrist_camera_id(cls, camera_id: str) -> bool:
        """Return True for canonical wrist cameras (camera_0, camera_1)."""
        cam_idx = cls._camera_index_from_device_key(camera_id)
        return cam_idx is not None and cam_idx in _WRIST_CAMERA_INDICES

    @staticmethod
    def _resolve_camera_image_topic(camera_id: str) -> str:
        """Map ``camera_N`` to ``nmx/hal/camera/camera_N/rgbd`` (see ``HAL_CAMERA_TOPICS``)."""
        try:
            camera_idx = int(str(camera_id).rsplit("_", 1)[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Invalid camera_id {camera_id!r}; expected form 'camera_<index>'"
            ) from exc
        if 0 <= camera_idx < len(HAL_CAMERA_TOPICS):
            return HAL_CAMERA_TOPICS[camera_idx]
        raise ValueError(
            f"No HAL image topic for {camera_id!r}; "
            f"supported indices 0..{len(HAL_CAMERA_TOPICS) - 1}"
        )

    def start_async_writer(self) -> None:
        """Start the background async writer thread."""
        if not self.write_thread_running:
            self.write_thread_running = True
            self.write_thread = threading.Thread(target=self._async_write_worker, daemon=True)
            self.write_thread.start()
            logger.info("Started async write thread")

    def stop_async_writer(self) -> None:
        """Stop the background async writer thread and close file streams."""
        if self.write_thread_running:
            logger.info("Stopping async write thread...")
            self.write_thread_running = False

            if self.write_thread:
                self.write_thread.join(timeout=5.0)
                if self.write_thread.is_alive():
                    logger.warning("Async write thread did not stop gracefully")
                else:
                    logger.info("Async write thread stopped")

            # Process any remaining items in queue
            remaining = 0
            while not self.write_queue.empty():
                try:
                    self.write_queue.get_nowait()
                    remaining += 1
                except queue.Empty:
                    break

            if remaining > 0:
                logger.warning(f"Discarded {remaining} items from write queue")

            # Close any open file streams
            if self.episode_active:
                self.close_episode_streams()

    def collect_data(self) -> None:
        """
        Collect data from all devices in a non-blocking manner.

        Each device is polled independently:
        - High-frequency devices (e.g., cameras) update frequently
        - Low-frequency devices (e.g., robot states) update less often
        - Only new data (based on timestamp changes) is collected
        - Data is queued for async NEDF3 writing
        """
        current_time = time.time()

        for device_key in self.devices:
            device_info = self.devices[device_key]
            data = self._read_direct_device_data(device_key, device_info)

            if data is not None:
                timestamp_ns, data_array = data

                # Only process if this is new data (timestamp changed)
                if timestamp_ns != self.devices[device_key]['last_timestamp']:

                    # Queue data for the background writer thread.
                    if self.write_thread_running:
                        try:
                            # Try non-blocking put first
                            write_task = {
                                'device_key': device_key,
                                'device_type': device_info['type'],
                                'device_name': device_info['name'],
                                'timestamp_ns': timestamp_ns,
                                'data_array': data_array.copy(),  # Copy to avoid race conditions
                                'device_info': device_info,
                                'collect_time': current_time
                            }
                            self.write_queue.put_nowait(write_task)
                        except queue.Full:
                            # Never block the 5ms collection loop; drop and warn with rate limiting.
                            now = time.time()
                            device_info = self.devices[device_key]
                            dropped = int(device_info.get('dropped_write_queue_full', 0)) + 1
                            device_info['dropped_write_queue_full'] = dropped
                            last_warn = float(device_info.get('last_queue_full_warn_time', 0.0))
                            if now - last_warn >= WRITE_QUEUE_FULL_WARN_INTERVAL_SEC:
                                device_info['last_queue_full_warn_time'] = now
                                logger.warning(f"Write queue full, dropping data for {device_key} "
                                               f"(dropped={dropped}, queue size: "
                                               f"{self.write_queue.qsize()}/{self.write_queue.maxsize})")

                    # Update tracking
                    self.devices[device_key]['last_timestamp'] = timestamp_ns
                    self.devices[device_key]['last_update_time'] = current_time

        # Each device is read directly from its own SHM ring buffer and deduplicated
        # by source timestamp, avoiding summary SHM lock contention during recording.

    def _read_direct_device_data(
        self,
        device_key: str,
        device_info: Dict[str, Any],
    ) -> Optional[Tuple[int, np.ndarray]]:
        """Read the latest frame directly from a device SHM ring buffer."""
        channel = self._direct_device_shms.get(device_key)
        if channel is None:
            try:
                channel = connect_to_device_shm(
                    device_info["type"],
                    int(device_info["id"]),
                    read_only=True,
                )
                self._direct_device_shms[device_key] = channel
                logger.info(f"Direct saver connected to {device_key}_data")
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Required device SHM is not available for {device_key}_data"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to connect required device SHM for {device_key}_data: {exc}"
                ) from exc

        try:
            buf = channel.buf
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            buffer_header = unpack_buffer_header(
                buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
            if int(buffer_header["current_frames_count"]) <= 0:
                return None

            buffer_size = int(buffer_header["buffer_size"])
            write_index = int(buffer_header["write_index"])
            frame_size = int(device_header["frame_size"])
            frame_index = (write_index - 1) % buffer_size
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_index * frame_size

            frame_header = unpack_frame_header(
                buf[frame_offset:frame_offset + FRAME_HEADER_SIZE]
            )
            timestamp_ns = int(frame_header["timestamp_ns"])
            if timestamp_ns == 0:
                return None

            dtype = get_dtype(device_header.get("data_dtype", "float64"), np.float64)
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = frame_offset + frame_size
            data_array = np.frombuffer(buf[data_start:data_end], dtype=dtype).copy()
            shape = tuple(device_header.get("shape") or device_info.get("shape") or ())
            if shape:
                data_array = data_array.reshape(shape)
            return timestamp_ns, data_array

        except (FileNotFoundError, BufferError, ValueError, OSError) as exc:
            try:
                channel.close()
            except Exception:
                pass
            self._direct_device_shms.pop(device_key, None)
            raise RuntimeError(f"Direct device SHM became unavailable for {device_key}_data") from exc
        except Exception as exc:
            raise RuntimeError(f"Direct saver failed reading {device_key}_data: {exc}") from exc

    def _record_episode_timing(self, episode_dir: str, status: str) -> None:
        """Persist timing outside the episode without affecting finalization."""
        episode_name = Path(episode_dir).name.removesuffix("_temp")
        try:
            record = record_episode_timing(
                episode_dir,
                output_dir=self.output_dir,
                task_name=self.task_name,
                status=status,
                timestamp_range_ns=self._episode_analysis_time_range_ns(),
            )
            duration = record.get("duration_seconds")
            duration_text = "unreadable" if duration is None else f"{duration:.6f}"
            logger.info(
                f"[episode_timing] status={status} episode={episode_name} "
                f"duration_seconds={duration_text}"
            )
        except Exception as exc:
            logger.warning(
                f"Failed to record episode timing for {episode_name} ({status}): {exc}"
            )
            logger.info(
                f"[episode_timing] status={status} episode={episode_name} "
                "duration_seconds=unreadable"
            )

    def _finalize_nedf3_episode(self, episode_dir: str) -> bool:
        """
        Validate a temporary NEDF3 episode before publishing its final name.

        Args:
            episode_dir: Directory containing NEDF3 files

        Returns:
            True if finalization found or hit an error for this episode.
        """
        if self.enable_anomaly_detection:
            self._cleanup_old_anomaly_jump_segments()
        try:
            episode_name = os.path.basename(episode_dir)
            final_episode_name = (
                episode_name[:-5] if episode_name.endswith("_temp") else episode_name
            )
            final_episode_dir = os.path.join(self.output_dir, final_episode_name)

            if not os.path.exists(episode_dir):
                logger.warning(f"Temporary NEDF3 episode no longer exists: {episode_dir}")
                return False
            needs_publish = os.path.abspath(episode_dir) != os.path.abspath(
                final_episode_dir
            )
            if needs_publish and os.path.exists(final_episode_dir):
                raise FileExistsError(
                    f"Final episode directory already exists: {final_episode_dir}"
                )

            # Keep rejected data outside the uploader-visible finalized namespace.
            has_errors, anomalies, feat_mapping = self._detect_episode_anomalies(
                episode_dir
            )
            if has_errors:
                self._record_episode_timing(episode_dir, "anomaly_error")
                logger.error(
                    f"Discarding temporary episode due to anomaly errors: {episode_dir}"
                )
                exported = self._export_anomaly_jump_segments(
                    episode_dir,
                    anomalies,
                    feat_mapping,
                )
                if not exported and self._quarantine_failed_anomaly_export(
                    episode_dir,
                    "Failed to export anomaly jump segments before discard",
                ):
                    return True
                if not self._discard_nedf3_episode(episode_dir):
                    logger.error(
                        "Rejected episode remains in upload-excluded temporary "
                        f"directory for later cleanup: {episode_dir}"
                    )
                return True

            if needs_publish:
                os.rename(episode_dir, final_episode_dir)

            logger.info(f"Finalized NEDF3 episode: {final_episode_dir}")
            self._record_episode_timing(final_episode_dir, "accepted")

            self.episode_counts += 1
            logger.info(
                f"[episode_status] saved_count={self.episode_counts} "
                f"last_episode={final_episode_name}"
            )
            return False

        except Exception as e:
            logger.error(f"Error finalizing NEDF3 episode: {e}")
            import traceback
            traceback.print_exc()
            return True

    def _discard_nedf3_episode(self, episode_dir: str) -> bool:
        """
        Discard NEDF3 episode, retrying transient Windows sharing violations.
        """
        import shutil

        if not os.path.exists(episode_dir):
            return True

        try:
            for attempt in range(len(DISCARD_SHARING_RETRY_DELAYS_SEC) + 1):
                try:
                    shutil.rmtree(episode_dir)
                except PermissionError as exc:
                    winerror = getattr(exc, "winerror", None)
                    if winerror not in (32, 33) or attempt >= len(
                        DISCARD_SHARING_RETRY_DELAYS_SEC
                    ):
                        raise

                    delay = DISCARD_SHARING_RETRY_DELAYS_SEC[attempt]
                    logger.warning(
                        "Episode file is still in use; retrying discard "
                        f"in {delay:.1f}s ({attempt + 1}/"
                        f"{len(DISCARD_SHARING_RETRY_DELAYS_SEC)}): {episode_dir}"
                    )
                    time.sleep(delay)
                else:
                    logger.info(f"Discarded NEDF3 episode: {episode_dir}")
                    return True

                if not os.path.exists(episode_dir):
                    logger.info(f"Discarded NEDF3 episode: {episode_dir}")
                    return True

            return False
        except Exception as e:
            logger.error(f"Error discarding NEDF3 episode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _wait_for_upload_maintenance(self, episode_name: str) -> None:
        """Ask the upload console to do one full scan before the next episode can start."""
        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "episode_name": episode_name,
            "output_dir": str(Path(self.output_dir).expanduser().resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
        }

        try:
            self._maintenance_request_path.parent.mkdir(parents=True, exist_ok=True)
            if self._maintenance_done_path.exists():
                self._maintenance_done_path.unlink()
            temp_path = self._maintenance_request_path.with_suffix(
                self._maintenance_request_path.suffix + ".tmp"
            )
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(self._maintenance_request_path)
        except Exception as exc:
            logger.warning(f"Could not request upload maintenance scan: {exc}")
            return

        logger.info(
            "Waiting for upload maintenance full scan before next episode "
            f"(request_id={request_id}, timeout={MAINTENANCE_REQUEST_TIMEOUT_SEC:.0f}s)"
        )
        deadline = time.monotonic() + MAINTENANCE_REQUEST_TIMEOUT_SEC
        while self.running and time.monotonic() < deadline:
            try:
                if self._maintenance_done_path.exists():
                    done = json.loads(self._maintenance_done_path.read_text(encoding="utf-8"))
                    if done.get("request_id") == request_id:
                        error = done.get("error")
                        if error:
                            logger.warning(f"Upload maintenance full scan reported error: {error}")
                        elapsed = done.get("elapsed_sec")
                        episode_count = done.get("episode_count")
                        if isinstance(elapsed, (int, float)):
                            scan_mode = str(done.get("scan_mode") or "unknown")
                            scan_elapsed = float(done.get("scan_elapsed_sec") or 0.0)
                            enqueue_elapsed = float(done.get("enqueue_elapsed_sec") or 0.0)
                            logger.info(
                                "Upload maintenance full scan completed "
                                f"in {elapsed:.2f}s (episodes={episode_count}, "
                                f"mode={scan_mode}, scan={scan_elapsed:.3f}s, "
                                f"enqueue={enqueue_elapsed:.3f}s)"
                            )
                        else:
                            logger.info("Upload maintenance full scan completed")
                        self._cleanup_upload_maintenance_files()
                        return
            except Exception as exc:
                logger.warning(f"Error while waiting for upload maintenance scan: {exc}")
            time.sleep(MAINTENANCE_POLL_INTERVAL_SEC)

        logger.warning(
            "Upload maintenance full scan did not complete before timeout; "
            "allowing next episode to avoid blocking collection"
        )
        self._cleanup_upload_maintenance_files()

    def _cleanup_upload_maintenance_files(self) -> None:
        for path in (self._maintenance_request_path, self._maintenance_done_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                logger.debug(f"Failed to remove upload maintenance signal {path}: {exc}")


    def start_saving(self) -> None:
        """Start the data saving process."""
        if not self.summary_shm:
            logger.error("Not connected to summary SHM")
            return

        self.running = True


        self.start_async_writer()
        logger.info("Async write mode enabled - data will be written continuously to disk")

        # Start keyboard listener for manual episode discard
        self._setup_keyboard_listener()

        logger.info("Start data acquisition. Ctrl+C to exit.")
        logger.info("+" * 40)

        episode_discarded = False
        try:
            while self.running:
                logger.info("Press Enter to start data acquisition.")
                wait_enter_blocking()
                self.episode_running = True
                logger.info("Data acquisition started.")
                logger.opt(colors=True).info("<green>{}</green>", "🟢" * 20)
                time_episode_start = time.time()
                self._episode_tracker_status_start = self._read_tracker_status_snapshot()

                episode_name = f"episode_{uuid.uuid4()}"
                episode_dir = os.path.join(self.output_dir, f"{episode_name}_temp")
                try:
                    self.open_episode_streams(episode_dir)
                except RuntimeError as exc:
                    logger.error(f"Cannot start episode: {exc}")
                    self.episode_running = False
                    if os.path.isdir(episode_dir):
                        self._discard_nedf3_episode(episode_dir)
                    continue

                # Poll shared memory frequently; the async writer owns disk I/O.
                wait_for_next_collect = precise_loop_timing(0.005)

                while True:
                    # Collect data from devices - non-blocking, independent per device
                    self.collect_data()

                    # Wait for next collection interval
                    wait_for_next_collect()

                    # Check for episode end via keyboard
                    episode_ended = enter_pressed_nonblocking()
                    episode_discarded = False

                    # check for episode discard signal
                    if self._discard_keyboard_pressed:
                        episode_discarded = True
                        self._discard_keyboard_pressed = False

                    episode_ended = episode_ended or episode_discarded

                    if episode_discarded:
                        logger.info("Episode discarded via signal")

                    if episode_ended:
                        finish_time = time.time()
                        logger.info("Data acquisition stopped by user.")
                        break

                logger.info("Waiting for async writer to finish...")
                self.write_queue.join()  # Wait for all queued data to be written
                logger.info("All data written to NEDF3")

                self.close_episode_streams()

                if episode_discarded:
                    episode_had_error = False
                    logger.info("Episode was discarded, skipping finalization")
                    self._record_episode_timing(episode_dir, "manual_discard")
                    self._discard_nedf3_episode(episode_dir)
                else:
                    logger.info("Finalizing NEDF3 episode...")
                    episode_had_error = self._finalize_nedf3_episode(episode_dir)

                completion_message = (
                    f"Episode completed in {finish_time - time_episode_start:.2f}s"
                )
                if episode_had_error:
                    logger.error(completion_message)
                else:
                    logger.info(completion_message)
                self._wait_for_upload_maintenance(episode_name)

        except KeyboardInterrupt:
            logger.info("Data saving interrupted by user")

        finally:
            # Stop keyboard listener
            self._stop_keyboard_listener()
            self.stop()

    def stop(self) -> None:
        """Stop the data saver consumer and async writer."""
        if not self.running:
            return

        logger.info("Stopping data saver consumer...")

        # Stop async writer first
        self.stop_async_writer()

        # Call parent stop method
        super().stop()

        for device_key, channel in list(self._direct_device_shms.items()):
            try:
                channel.close()
            except Exception as exc:
                logger.debug(f"Error closing direct SHM for {device_key}: {exc}")
        self._direct_device_shms.clear()

        logger.info("Data saver consumer stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of the data saver consumer."""
        status = {
            'connected': self.summary_shm is not None,
            'running': self.running,
            'devices': len(self.devices),
            'output_dir': self.output_dir,
            'save_interval': self.save_interval,
            'frame_timestamp': self.frame_timestamp,
            'write_thread_running': self.write_thread_running,
            'queue_size': self.write_queue.qsize(),
            'anomaly_detection_enabled': self.enable_anomaly_detection,
        }

        # Add buffer information
        buffer_info = {}
        for device_key, buffer_data in self.data_buffer.items():
            buffer_info[device_key] = len(buffer_data)
        status['buffer_samples'] = buffer_info

        # Add anomaly detection statistics
        if self.enable_anomaly_detection:
            status['anomaly_stats'] = self.anomaly_stats.copy()

        return status


def main() -> None:
    """Main function to run the data saver consumer."""
    parser = argparse.ArgumentParser(description="Data Saver Consumer - Save Device Data to NEDF3 (UMI) Format")
    parser.add_argument("--summary-shm", "-s", default="device_summary_data",
                        help="Summary shared memory name (default: device_summary_data)")
    parser.add_argument("--output-dir", "-o", default="saved_data",
                        help="Output directory for saved data (default: saved_data)")
    parser.add_argument("--interval", "-i", type=float, default=0.001,
                        help="Save interval in seconds (default: 0.001)")
    parser.add_argument("--start-idx", "-st", type=int, default=0,
                        help="Start index of the episode")
    parser.add_argument("--image-size", "-is", type=str, default=None,
                        help="Target image size for all cameras as 'width,height' (e.g., '400,300'). "
                             "Applied after wrist crop when enabled.")
    parser.add_argument(
        "--wrist-crop-aspect",
        type=str,
        default="4,3",
        help="Height:width aspect for left/right wrist crop (default: 4,3 = portrait, "
             "height > width). Use 'none' to disable wrist crop/pad.",
    )
    parser.add_argument(
        "--wrist-output-size",
        type=str,
        default="480,640",
        help="Wrist output size as 'width,height' after crop (default: 480,640 = 640×480 "
             "portrait 4:3, same pixels as head). Use 'none' to keep cropped resolution.",
    )
    parser.add_argument(
        "--disable-wrist-crop",
        action="store_true",
        help="Disable center crop/pad for left/right wrist cameras.",
    )
    parser.add_argument("--queue-size", "-q", type=int, default=5000,
                        help="Async write queue size (default: 5000)")
    parser.add_argument("--robot-type", "-r", type=str, default="UMI",
                        help="NEDF metadata robot_vendor_model string (default: UMI)")
    parser.add_argument("--task-name", "-t", type=str, default="data_collection",
                        help="Task name for NEDF metadata (default: data_collection)")
    parser.add_argument(
        "--end-effector-vendor-model",
        "-e",
        type=str,
        default="",
        help="NEDF3 metadata end_effector_vendor_model (VendorName_ModelName; default empty)",
    )
    parser.add_argument("--enable-anomaly-detection", action="store_true", default=True,
                        help="Enable episode anomaly detection (default: True)")
    parser.add_argument("--disable-anomaly-detection", action="store_true",
                        help="Disable episode anomaly detection")
    parser.add_argument("--anomaly-config", type=str, default=None,
                        help="Path to anomaly detection config file")
    parser.add_argument(
        "--extra-device-type",
        type=str,
        default=",".join(DEFAULT_EXTRA_DEVICE_TYPES),
        help="metadata.extra device_type for camera_0,camera_1,camera_2 "
             "(default: robopocket_2.0_pro × 3)",
    )
    parser.add_argument(
        "--extra-hardware-string",
        type=str,
        default=",".join(DEFAULT_EXTRA_HARDWARE_STRINGS),
        help="metadata.extra hardware_string for camera_0,camera_1,camera_2 "
             "(default: 310,310,310)",
    )
    parser.add_argument(
        "--extra-device-zero",
        type=str,
        default="",
        help="metadata.extra device[0].identifier from data collection GUI (e.g. 07)",
    )
    parser.add_argument(
        "--dm3-user-id",
        type=str,
        default="",
        help="metadata.extra user_id from DM3 login",
    )
    parser.add_argument(
        "--dm3-user-name",
        type=str,
        default="",
        help="metadata.extra user_name from DM3 login",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="",
        help="NEDF task_description from selected DM3 task",
    )
    parser.add_argument(
        "--dm3-session-file",
        type=str,
        default=None,
        help="Optional path to dm3_collection_session.json; CLI args override file values",
    )
    parser.add_argument("--status", action="store_true",
                        help="Show consumer status and exit")

    args = parser.parse_args()

    # Determine anomaly detection setting
    enable_anomaly_detection = args.enable_anomaly_detection and not args.disable_anomaly_detection

    # Parse image size
    image_size = None
    if args.image_size:
        try:
            width, height = map(int, args.image_size.split(','))
            image_size = (width, height)
            logger.info(f"Target image size: {image_size} (width, height)")
        except ValueError:
            logger.error(f"Invalid image size format: {args.image_size}. Expected 'width,height' (e.g., '400,300')")
            return

    wrist_crop_aspect_hw: Optional[Tuple[int, int]] = (4, 3)
    if args.disable_wrist_crop or args.wrist_crop_aspect.lower() == "none":
        wrist_crop_aspect_hw = None
    else:
        try:
            aspect_h, aspect_w = map(int, args.wrist_crop_aspect.split(","))
            wrist_crop_aspect_hw = (aspect_h, aspect_w)
        except ValueError:
            logger.error(
                f"Invalid wrist crop aspect: {args.wrist_crop_aspect}. "
                "Expected 'height,width' (e.g., '4,3') or 'none'."
            )
            return

    wrist_output_size: Optional[Tuple[int, int]] = (480, 640)
    if args.wrist_output_size.lower() == "none":
        wrist_output_size = None
    else:
        try:
            wrist_width, wrist_height = map(int, args.wrist_output_size.split(","))
            wrist_output_size = (wrist_width, wrist_height)
        except ValueError:
            logger.error(
                f"Invalid wrist output size: {args.wrist_output_size}. "
                "Expected 'width,height' (e.g., '480,640') or 'none'."
            )
            return

    try:
        extra_device_types = parse_metadata_extra_csv(args.extra_device_type, "device_type")
        extra_hardware_strings = parse_metadata_extra_csv(
            args.extra_hardware_string, "hardware_string"
        )
    except ValueError as exc:
        logger.error(str(exc))
        return

    dm3_user_id = args.dm3_user_id.strip()
    dm3_user_name = args.dm3_user_name.strip()
    task_description = normalize_task_description(args.task_description)
    extra_device_zero = args.extra_device_zero.strip()
    if args.dm3_session_file:
        try:
            session_fields = get_metadata_fields_from_session(
                extra_device_zero=extra_device_zero or None,
            )
            if not dm3_user_id:
                dm3_user_id = str(session_fields.get("user_id") or "")
            if not dm3_user_name:
                dm3_user_name = str(session_fields.get("user_name") or "")
            if not task_description:
                task_description = str(session_fields.get("task_description") or "")
            if not extra_device_zero:
                extra_device_zero = parse_extra_device_identifier(
                    session_fields.get("device")
                )
        except Exception as exc:
            logger.warning(f"Failed to load DM3 session file: {exc}")

    # Create data saver consumer
    consumer = DataSaverConsumer(
        summary_shm_name=args.summary_shm,
        output_dir=args.output_dir,
        save_interval=args.interval,
        start_idx=args.start_idx,
        image_size=image_size,
        wrist_crop_aspect_hw=wrist_crop_aspect_hw,
        wrist_output_size=wrist_output_size,
        queue_size=args.queue_size,
        robot_type=args.robot_type,
        task_name=args.task_name,
        end_effector_vendor_model=args.end_effector_vendor_model,
        extra_device_types=extra_device_types,
        extra_hardware_strings=extra_hardware_strings,
        codebase_version="",
        extra_device_zero=extra_device_zero,
        dm3_user_id=dm3_user_id,
        dm3_user_name=dm3_user_name,
        task_description=task_description,
        enable_anomaly_detection=enable_anomaly_detection,
        anomaly_config_path=args.anomaly_config,
    )

    if args.status:
        # Show status
        status = consumer.get_status()
        logger.info("Data Saver Consumer Status:")
        logger.info(f"Connected: {status['connected']}")
        logger.info(f"Running: {status['running']}")
        logger.info(f"Devices: {status['devices']}")
        logger.info(f"Output directory: {status['output_dir']}")
        logger.info(f"Save interval: {status['save_interval']}s")
        logger.info(f"Write thread running: {status['write_thread_running']}")
        logger.info(f"Write queue size: {status['queue_size']}")
        logger.info(f"Frame timestamp: {status['frame_timestamp']}")
        logger.info(f"Anomaly detection: {status['anomaly_detection_enabled']}")

        if status['buffer_samples']:
            logger.info("\nBuffer samples:")
            for device_key, samples in status['buffer_samples'].items():
                logger.info(f"  {device_key}: {samples} samples")

        if status.get('anomaly_stats'):
            anomaly_stats = status['anomaly_stats']
            logger.info(f"\nAnomaly detection statistics:")
            logger.info(f"  Episodes checked: {anomaly_stats['total_episodes_checked']}")
            logger.info(f"  Total anomalies: {anomaly_stats['total_anomalies']}")
            logger.info(f"  Episodes with errors: {anomaly_stats['episodes_with_errors']}")
            if anomaly_stats['last_check_time'] > 0:
                import datetime
                last_time = datetime.datetime.fromtimestamp(anomaly_stats['last_check_time'])
                logger.info(f"  Last check: {last_time}")

        return

    logger.info("Data Saver Consumer - Save Device Data to NEDF3 (UMI) Format")
    logger.info("========================================================")
    logger.info(f"Summary SHM: {args.summary_shm}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Save interval: {args.interval}s")
    logger.info(f"Robot type: {args.robot_type}")
    logger.info(f"Task name: {args.task_name}")
    if args.end_effector_vendor_model:
        logger.info(f"End-effector vendor/model: {args.end_effector_vendor_model}")
    logger.info(f"Write mode: Async (NEDF3 streaming)")
    logger.info(f"Queue size: {args.queue_size}")
    logger.info(f"Anomaly detection: {enable_anomaly_detection}")
    if args.anomaly_config:
        logger.info(f"Anomaly config: {args.anomaly_config}")
    if wrist_crop_aspect_hw:
        wrist_out = (
            f"{wrist_output_size[0]}x{wrist_output_size[1]}"
            if wrist_output_size
            else "native cropped resolution"
        )
        logger.info(
            f"Wrist crop/pad: aspect {wrist_crop_aspect_hw[0]}:{wrist_crop_aspect_hw[1]} "
            f"(h:w) -> {wrist_out}"
        )
    else:
        logger.info("Wrist crop/pad: disabled")
    if image_size:
        logger.info(f"Global image resize: {image_size[0]}x{image_size[1]}")
    logger.info(f"metadata.extra device_type: {extra_device_types}")
    logger.info(f"metadata.extra hardware_string: {extra_hardware_strings}")
    logger.info(f"metadata.extra device[0].identifier: {extra_device_zero or '-'}")
    logger.info(f"metadata.extra user_id: {dm3_user_id or '-'}")
    logger.info(f"metadata.extra user_name: {dm3_user_name or '-'}")
    logger.info(f"task_description: {task_description or '-'}")
    logger.info("")

    # Try to connect
    if not consumer.connect():
        logger.error("Failed to connect to summary SHM. Make sure the Device Manager is running.")
        return

    try:
        consumer.start_saving()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()
