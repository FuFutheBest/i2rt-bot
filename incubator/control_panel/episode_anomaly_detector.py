#!/usr/bin/env python3
"""
Episode anomaly detector for NEDF3 data.

This module reads saved NEDF3 data and performs comprehensive anomaly checks.

Author: Assistant
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

# NEDF SDK imports are optional so pure contract tests remain hardware-free.
try:
    from dm3_core.nedf import NEDFFactory, NEDFReaderConfig
except ImportError:
    NEDFFactory = None
    NEDFReaderConfig = None

from incubator.control_panel.nedf3_io import (
    extract_gripper_width,
    extract_pose_wxyz,
    extract_rgb_jpeg,
)
from incubator.control_panel.recording_cameras import resolve_recorded_camera_ids
from incubator.control_panel.recording_contract import (
    NEDF3_ARM_TOPICS,
    NEDF3_DATA_FORMAT,
    NEDF3_HAND_TOPICS,
    normalize_data_format,
)

from utils.episode_anomaly_algorithms import EpisodeAnomalyAlgorithms
from utils.logger_config import logger


class EpisodeAnomalyDetector(EpisodeAnomalyAlgorithms):
    """
    Anomaly detector for saved NEDF3 episodes.

    This detector analyzes complete episode data after it has been saved
    and detects various types of anomalies including:
    - Frequency anomalies
    - Data quality issues
    - Timestamp discontinuities
    - Device-specific anomalies
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize episode anomaly detector.

        Args:
            config: Anomaly detection configuration from YAML file
        """
        self.config = config
        logger.info("Episode anomaly detector initialized")

    def detect_episode_anomalies(
        self, episode_dir: Path, feat_mapping: Dict[str, str]
    ) -> Tuple[List[Dict], bool]:
        """
        Detect anomalies in a saved NEDF3 episode.

        Args:
            episode_dir: Path to an NEDF3 episode directory
            feat_mapping: Mapping of feature names to device keys

        Returns:
            Tuple of (anomaly_list, has_errors) where:
            - anomaly_list: List of detected anomalies
            - has_errors: True if any critical errors were found
        """
        anomalies = []

        try:
            # Load episode data
            data_dict, timestamp_dict = self._load_episode_data(
                episode_dir, feat_mapping
            )

            if not data_dict or not timestamp_dict:
                anomalies.append(
                    {
                        "severity": "error",
                        "type": "no_data",
                        "message": f"No data found in episode {episode_dir.name}",
                        "episode_path": str(episode_dir),
                    }
                )
                return anomalies, True

            # Run all detection methods
            anomalies.extend(
                self._detect_missing_devices(data_dict, feat_mapping, episode_dir)
            )
            anomalies.extend(
                self._detect_frequency_anomalies(data_dict, timestamp_dict, episode_dir)
            )
            anomalies.extend(
                self._detect_length_anomalies(data_dict, timestamp_dict, episode_dir)
            )
            anomalies.extend(
                self._detect_image_anomalies(
                    data_dict, timestamp_dict, feat_mapping, episode_dir
                )
            )
            anomalies.extend(
                self._detect_pose_anomalies(
                    data_dict, timestamp_dict, feat_mapping, episode_dir
                )
            )
            anomalies.extend(
                self._detect_grip_anomalies(
                    data_dict, timestamp_dict, feat_mapping, episode_dir
                )
            )

        except Exception as e:
            logger.error(f"Error detecting anomalies in {episode_dir}: {e}")
            anomalies.append(
                {
                    "severity": "error",
                    "type": "detection_error",
                    "message": f"Error during anomaly detection: {str(e)}",
                    "episode_path": str(episode_dir),
                }
            )

        # Check if any errors were found
        has_errors = any(a["severity"] == "error" for a in anomalies)

        return anomalies, has_errors

    @staticmethod
    def _timestamp_ns(message: Any) -> int:
        record = getattr(message, "message", message)
        for name in ("log_time", "publish_time", "timestamp_ns", "timestamp"):
            value = getattr(record, name, None)
            if value is not None:
                return int(value)
        raise ValueError("NEDF message has no timestamp")

    @staticmethod
    def _iter_lowdim(reader: Any, topic: str):
        legacy = getattr(reader, "get_lowdim", None)
        if callable(legacy):
            yield from legacy(topic)
            return
        yield from reader.iter_decoded_messages(topic)

    @staticmethod
    def _iter_camera(reader: Any, camera_id: str, data_format: str):
        normalize_data_format(data_format)
        topic = f"nmx/hal/camera/{camera_id}/rgbd"
        yield from reader.iter_decoded_messages(topic)

    @staticmethod
    def _camera_payload(message: Any, data_format: str) -> bytes:
        normalize_data_format(data_format)
        return extract_rgb_jpeg(message)

    def _load_episode_data(
        self, episode_dir: Path, feat_mapping: Dict[str, str]
    ) -> Tuple[Dict, Dict]:
        """Load anomaly inputs from a canonical NEDF3 episode."""
        data_dict = {}
        timestamp_dict = {}

        # Get metadata file path
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"metadata.json not found: {metadata_path}")
            return data_dict, timestamp_dict

        try:
            if NEDFFactory is None or NEDFReaderConfig is None:
                raise RuntimeError("dm3-core is required to read rollout episodes")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data_format = normalize_data_format(metadata.get("data_format"))
            reader_config = NEDFReaderConfig(metadata_file_path=str(metadata_path))
            reader = NEDFFactory.get_reader(reader_config)

            # Read camera images (store JPG data directly for faster processing)
            camera_info = metadata.get("camera_info")
            requested_cameras = {
                "left": feat_mapping.get("img_left"),
                "right": feat_mapping.get("img_right"),
            }
            resolved_cameras = resolve_recorded_camera_ids(
                camera_info,
                requested_cameras,
                data_format=data_format,
            )
            for img_key, role in (("img_left", "left"), ("img_right", "right")):
                device_key = feat_mapping.get(img_key)
                camera_id = resolved_cameras.get(role)
                if not device_key or not camera_id:
                    continue
                jpg_data_list = []
                img_timestamps_list = []
                for msg in self._iter_camera(reader, camera_id, data_format):
                    payload = self._camera_payload(msg, data_format)
                    jpg_data_list.append(np.frombuffer(payload, dtype=np.uint8))
                    img_timestamps_list.append(self._timestamp_ns(msg) / 1e9)
                if jpg_data_list:
                    data_dict[device_key] = jpg_data_list
                    timestamp_dict[device_key] = np.asarray(img_timestamps_list)

            # Read pose data
            if "pose_unified" in feat_mapping:
                pose_device_key = feat_mapping["pose_unified"]
                pose_data_list = []
                pose_timestamps_list = []
                if data_format == NEDF3_DATA_FORMAT:
                    by_side: Dict[str, Dict[int, np.ndarray]] = {}
                    for side in ("left", "right"):
                        by_side[side] = {
                            self._timestamp_ns(msg): extract_pose_wxyz(msg)
                            for msg in reader.iter_decoded_messages(
                                NEDF3_ARM_TOPICS[side]["tcp/state"]
                            )
                        }
                    for timestamp_ns in sorted(
                        set(by_side["left"]) & set(by_side["right"])
                    ):
                        pose_data_list.append(
                            np.stack(
                                [
                                    by_side["left"][timestamp_ns],
                                    by_side["right"][timestamp_ns],
                                ]
                            )
                        )
                        pose_timestamps_list.append(timestamp_ns / 1e9)
                else:
                    pose_topic = f"/lowdim/{pose_device_key.lower()}"
                    for msg in self._iter_lowdim(reader, pose_topic):
                        log_ts_ns = self._timestamp_ns(msg)
                        lowdim_data = getattr(msg, "decoded_message", msg)
                        pose_array = np.asarray(lowdim_data.data, dtype=np.float32)
                        if len(pose_array) == 14:
                            pose_data_list.append(pose_array.reshape(2, 7))
                            pose_timestamps_list.append(log_ts_ns / 1e9)
                        elif len(pose_array) == 7:
                            pose_reshaped = np.zeros((2, 7), dtype=np.float32)
                            pose_reshaped[0, :] = pose_array
                            pose_data_list.append(pose_reshaped)
                            pose_timestamps_list.append(log_ts_ns / 1e9)
                            logger.error(
                                "Single-arm pose data detected, padding right arm with zeros"
                            )
                if pose_data_list:
                    data_dict[pose_device_key] = np.array(pose_data_list)
                    timestamp_dict[pose_device_key] = np.array(pose_timestamps_list)

            # Rizon dual-arm rollouts persist each arm on its own 7D topic.
            for pose_key, side in (("pose_left", "left"), ("pose_right", "right")):
                if pose_key not in feat_mapping:
                    continue
                pose_device_key = feat_mapping[pose_key]
                pose_topic = (
                    NEDF3_ARM_TOPICS[side]["tcp/state"]
                    if data_format == NEDF3_DATA_FORMAT
                    else f"/lowdim/{pose_device_key.lower()}"
                )
                pose_data_list = []
                pose_timestamps_list = []
                iterator = (
                    reader.iter_decoded_messages(pose_topic)
                    if data_format == NEDF3_DATA_FORMAT
                    else self._iter_lowdim(reader, pose_topic)
                )
                for msg in iterator:
                    pose_array = (
                        extract_pose_wxyz(msg)
                        if data_format == NEDF3_DATA_FORMAT
                        else np.asarray(
                            getattr(msg, "decoded_message", msg).data,
                            dtype=np.float32,
                        )
                    )
                    if pose_array.shape != (7,):
                        logger.warning(
                            f"Ignoring invalid {pose_key} sample from {pose_topic}: "
                            f"expected 7 values, got {pose_array.shape}"
                        )
                        continue
                    pose_data_list.append(pose_array)
                    pose_timestamps_list.append(self._timestamp_ns(msg) / 1e9)
                if pose_data_list:
                    data_dict[pose_device_key] = np.asarray(pose_data_list)
                    timestamp_dict[pose_device_key] = np.asarray(pose_timestamps_list)

            # Read gripper data
            for grip_key, side in (("grip_left", "left"), ("grip_right", "right")):
                grip_device_key = feat_mapping.get(grip_key)
                if not grip_device_key:
                    continue
                grip_topic = (
                    NEDF3_HAND_TOPICS[side]["state"]
                    if data_format == NEDF3_DATA_FORMAT
                    else f"/lowdim/{grip_device_key.lower()}"
                )
                iterator = (
                    reader.iter_decoded_messages(grip_topic)
                    if data_format == NEDF3_DATA_FORMAT
                    else self._iter_lowdim(reader, grip_topic)
                )
                grip_data_list = []
                grip_timestamps_list = []
                for msg in iterator:
                    if data_format == NEDF3_DATA_FORMAT:
                        grip_value = extract_gripper_width(msg)
                        if grip_value is None:
                            continue
                    else:
                        lowdim_data = getattr(msg, "decoded_message", msg)
                        grip_value = float(lowdim_data.data[0])
                    grip_data_list.append([grip_value])
                    grip_timestamps_list.append(self._timestamp_ns(msg) / 1e9)
                if grip_data_list:
                    data_dict[grip_device_key] = np.asarray(
                        grip_data_list, dtype=np.float32
                    )
                    timestamp_dict[grip_device_key] = np.asarray(grip_timestamps_list)

        except Exception as e:
            logger.error(f"Error loading episode data from {episode_dir}: {e}")
            return {}, {}

        # Convert to relative timestamps
        if timestamp_dict:
            min_timestamp = min(ts[0] for ts in timestamp_dict.values() if len(ts) > 0)
            for device_key in timestamp_dict:
                timestamp_dict[device_key] = timestamp_dict[device_key] - min_timestamp

        return data_dict, timestamp_dict
