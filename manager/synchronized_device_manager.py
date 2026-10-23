#!/usr/bin/env python3
"""
Synchronized Device Manager - Ensures all device data has very similar timestamps.

This manager reads the latest frame from a master device, then finds frames from
other devices with timestamps closest to the master device's timestamp, considering
hardware latency compensation.

Author: Jun Lv
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from devices import DEVICE_CLASSES
from devices.contracts import DeviceKind, parse_device_kind
from .base_device_manager import BaseDeviceManager
from .utils.data_interpolation import (
    LowDimInterpolationError,
    extrapolate_low_dim_data,
    interpolate_low_dim_data,
)
from utils.data_format import DataFormat, DataFormatError, parse_data_format
from utils.device_config import state_format_mapping
from utils.logger_config import logger
from utils.shm_utils import get_dtype
from utils.synchronization import (
    BINARY_MATCH,
    CAMERA_ALIGN,
    SynchronizationModeError,
    normalize_sync_mode,
)


class SynchronizedDeviceManager(BaseDeviceManager):
    """
    Synchronized Device Manager for time-synchronized device data aggregation.

    This manager ensures that all device data has very similar timestamps by:
    1. Reading the latest frame from a master device
    2. Finding frames from other devices with timestamps closest to the master's timestamp
    3. Compensating for hardware latency differences
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: Optional[str] = None,
        buffer_size: int = 100
    ) -> None:
        """
        Initialize the Synchronized Device Manager.

        Args:
            config: Configuration dictionary (for Hydra mode)
            config_path: Path to configuration file (for traditional mode)
            buffer_size: Buffer size for summary SHM
        """
        super().__init__(config=config, config_path=config_path, buffer_size=buffer_size)

        # Get the master device configuration-list index from config.
        self._get_master_device_class_id()
        self.sync_mode = self._get_sync_mode()
        if self.sync_mode == CAMERA_ALIGN:
            self._validate_camera_interpolation_contract()

        # Track master device timestamp for update checking
        self.master_last_timestamp = None

    def _get_sync_mode(self) -> str:
        """Resolve the configured synchronization mode with a binary-match default."""
        device_manager_config = self.config.get("device_manager", {})
        try:
            return normalize_sync_mode(
                device_manager_config.get("sync_mode", BINARY_MATCH)
            )
        except SynchronizationModeError as exc:
            raise ValueError(f"Invalid device_manager.sync_mode: {exc}") from exc

    def _get_master_device_class_id(self) -> None:
        """Get and validate the master device configuration-list index."""
        self.master_device_class_id = self.config.get("device_manager", {}).get(
            "master_device_class_id"
        )

        if self.master_device_class_id is None:
            raise ValueError("master_device_class_id not found in configuration file")

        # Disabled entries retain their IDs; never index the filtered list by ID.
        if not any(
            device["device_class_id"] == self.master_device_class_id
            for device in self.devices
        ):
            raise ValueError(
                f"Master device_class_id {self.master_device_class_id} is not enabled"
            )

        # Get master device by index
        master_device = next(d for d in self.devices if d["device_class_id"] == self.master_device_class_id)

        # Print complete master device information
        logger.info("=" * 50)
        logger.info("MASTER DEVICE CONFIGURATION")
        logger.info("=" * 50)
        logger.info(
            "Master Device Configuration Index: %s", self.master_device_class_id
        )
        logger.info(f"Device ID: {master_device['device_class_id']}")
        logger.info(f"Device Class: {master_device['device_class']}")
        logger.info("Device Config:")
        for key, value in master_device['config'].items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)

    @staticmethod
    def _is_camera_device(device: Dict[str, Any]) -> bool:
        """Identify image streams from the shared capability registry."""
        raw_kind = device.get("device_kind")
        if raw_kind is not None:
            return parse_device_kind(raw_kind) is DeviceKind.CAMERA
        try:
            return DEVICE_CLASSES.kind(device["device_class"]) is DeviceKind.CAMERA
        except KeyError:
            return False

    @staticmethod
    def _device_data_format(device: Dict[str, Any]) -> Optional[DataFormat]:
        """Return one cached low-dimensional format contract, when configured."""
        if SynchronizedDeviceManager._is_camera_device(device):
            return None
        cached_format = device.get("data_format_contract")
        if isinstance(cached_format, DataFormat):
            return cached_format

        raw_format = state_format_mapping(device["config"])
        if raw_format is None:
            return None
        try:
            data_format = parse_data_format(raw_format)
        except DataFormatError as exc:
            raise ValueError(
                "Invalid state_feedback.format for "
                f"{device['device_class']}_{device['device_class_id']}: {exc}"
            ) from exc
        device["data_format_contract"] = data_format
        return data_format

    def _validate_camera_interpolation_contract(self) -> None:
        """Validate the low-dimensional layout required by camera-master matching."""
        master_device = next(d for d in self.devices if d["device_class_id"] == self.master_device_class_id)
        if not self._is_camera_device(master_device):
            raise ValueError(
                "camera_align sync_mode requires master_device_class_id to select "
                "a camera device"
            )
        for device_index, device in enumerate(self.devices):
            if device["device_class_id"] == self.master_device_class_id:
                continue

            data_format = self._device_data_format(device)
            if data_format is None:
                if self._is_camera_device(device):
                    continue
                raise ValueError(
                    "camera_align sync_mode requires low-dimensional device "
                    f"{device['device_class']}_{device['device_class_id']} to declare "
                    "state_feedback.format"
                )

            actual_shape = tuple(device.get("data_shape") or ())
            if actual_shape and actual_shape != data_format.shape:
                raise ValueError(
                    f"{device['device_class']}_{device['device_class_id']} SHM shape "
                    f"{actual_shape} does not match data_format.shape {data_format.shape}"
                )
            data_dtype = get_dtype(data_format.dtype, np.float64)
            if not np.issubdtype(np.dtype(data_dtype), np.floating):
                raise ValueError(
                    "camera_align sync_mode requires low-dimensional device "
                    f"{device['device_class']}_{device['device_class_id']} to use a floating dtype"
                )

    def _calculate_adjusted_timestamp(self, master_timestamp_ns: int, device: Dict[str, Any]) -> int:
        """
        Calculate adjusted timestamp for a device considering hardware latency.

        Args:
            master_timestamp_ns: Master device timestamp in nanoseconds
            device: Device dictionary

        Returns:
            int: Adjusted timestamp in nanoseconds
        """
        # Get master device by index
        master_device = next(d for d in self.devices if d["device_class_id"] == self.master_device_class_id)

        # Get hardware latency for the device
        hardware_latency_ms = self._get_device_hardware_latency(device)
        master_latency_ms = self._get_device_hardware_latency(master_device)

        # Calculate latency difference
        latency_diff_ms = hardware_latency_ms - master_latency_ms
        latency_diff_ns = int(latency_diff_ms * 1_000_000)  # Convert ms to ns

        # Adjust timestamp
        adjusted_timestamp = master_timestamp_ns + latency_diff_ns

        logger.debug(f"Device {device['device_class_id']}: latency={hardware_latency_ms}ms, "
                         f"adjusted_timestamp={adjusted_timestamp}")

        return adjusted_timestamp

    def _find_synchronized_frames(
        self,
        master_timestamp_ns: int,
        master_frame: Optional[tuple] = None,
    ) -> List[Optional[tuple]]:
        """
        Find synchronized frames for all devices based on master timestamp.

        Args:
            master_timestamp_ns: Master device timestamp in nanoseconds

        Returns:
            List of (timestamp_ns, data_array) tuples for each device, or None if failed
        """
        device_frames = []

        for device_index, device in enumerate(self.devices):
            # IDs remain stable when disabled devices are filtered out.
            if device["device_class_id"] == self.master_device_class_id:
                # Reuse the exact frame that supplied the target timestamp.
                frame_data = master_frame or self._read_device_frame(device)
                device_frames.append(frame_data)
                continue

            # For other devices, find frame with closest timestamp
            adjusted_timestamp = self._calculate_adjusted_timestamp(master_timestamp_ns, device)

            # Try to connect to device if not connected
            if not self._connect_to_device(device):
                logger.warning(f"Cannot connect to device {device['device_class_id']}")
                device_frames.append(None)
                continue

            # Search for frame with closest timestamp
            search_result = self._search_device_buffer_for_timestamp(device, adjusted_timestamp)

            if search_result is None:
                logger.warning(f"Cannot find synchronized frame for device {device['device_class_id']}")
                device_frames.append(None)
                continue

            timestamp_ns, data_array, frame_index = search_result
            logger.debug(f"Device {device['device_class_id']}: found synchronized frame at index {frame_index}, "
                            f"timestamp_diff={abs(timestamp_ns - adjusted_timestamp)/1_000_000:.2f}ms")
            device_frames.append((timestamp_ns, data_array))

        return device_frames

    def _search_device_buffer_bracket(
        self,
        device: Dict[str, Any],
        target_timestamp_ns: int,
    ) -> Optional[Tuple[tuple, tuple]]:
        """Return the adjacent frames that bracket a target timestamp.

        The device ring is interpreted in chronological order. A target outside
        the retained range returns ``None``; bounded forward extrapolation is
        handled separately by the camera synchronization path.
        """
        buffer_info = self._get_device_buffer_info(device)
        if buffer_info is None:
            return None

        buffer_size, current_frames_count, write_index, _, _ = buffer_info
        if current_frames_count == 0:
            return None
        if current_frames_count < buffer_size:
            start_idx, end_idx = 0, current_frames_count - 1
        else:
            start_idx, end_idx = write_index, write_index + buffer_size - 1

        left, right = start_idx, end_idx
        previous: Optional[tuple] = None
        following: Optional[tuple] = None
        while left <= right:
            mid = (left + right) // 2
            actual_index = mid % buffer_size
            frame_data = self._read_device_frame(device, actual_index)
            if frame_data is None:
                return None

            timestamp_ns, data_array = frame_data
            frame = (timestamp_ns, data_array, actual_index)
            if timestamp_ns == target_timestamp_ns:
                return frame, frame
            if timestamp_ns < target_timestamp_ns:
                previous = frame
                left = mid + 1
            else:
                following = frame
                right = mid - 1

        if previous is None or following is None:
            return None
        return previous, following

    @staticmethod
    def _interpolate_bracketed_frame(
        bracket: Tuple[tuple, tuple],
        target_timestamp_ns: int,
        data_format: DataFormat,
    ) -> Optional[tuple]:
        """Interpolate a low-dimensional frame at an in-range target time."""
        start, end = bracket
        start_timestamp_ns, start_data, _ = start
        end_timestamp_ns, end_data, _ = end
        if start_timestamp_ns == end_timestamp_ns:
            return int(target_timestamp_ns), np.array(start_data, copy=True)
        if end_timestamp_ns < start_timestamp_ns:
            return None

        ratio = (target_timestamp_ns - start_timestamp_ns) / (
            end_timestamp_ns - start_timestamp_ns
        )
        if not 0.0 <= ratio <= 1.0:
            return None
        try:
            interpolated = interpolate_low_dim_data(
                start_data,
                end_data,
                ratio,
                data_format,
            )
        except LowDimInterpolationError as exc:
            logger.warning("Cannot interpolate low-dimensional frame: %s", exc)
            return None
        return int(target_timestamp_ns), interpolated

    def _extrapolate_latest_robot_frame(
        self, device: Dict[str, Any], target_timestamp_ns: int, data_format: DataFormat
    ) -> Optional[tuple]:
        """Project the latest two robot samples by at most 1.5 configured periods."""
        kind = device.get("device_kind")
        if kind is None:
            kind = DEVICE_CLASSES.kind(device["device_class"])
        if parse_device_kind(kind) is not DeviceKind.LOW_DIM_RW:
            return None
        try:
            fps = float(device["config"]["state_feedback"]["fps"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(fps) or fps <= 0:
            return None
        info = self._get_device_buffer_info(device)
        if info is None:
            return None
        size, count, write_index, _, _ = info
        if count < 2 or size < 2:
            return None
        end_index = count - 1 if count < size else (write_index - 1) % size
        start = self._read_device_frame(device, (end_index - 1) % size)
        end = self._read_device_frame(device, end_index)
        if start is None or end is None:
            return None
        start_ts, start_data = start
        end_ts, end_data = end
        horizon_ns = target_timestamp_ns - end_ts
        if end_ts <= start_ts or not 0 < horizon_ns <= 1.5e9 / fps:
            return None
        ratio = 1.0 + horizon_ns / (end_ts - start_ts)
        try:
            projected = extrapolate_low_dim_data(start_data, end_data, ratio, data_format)
        except LowDimInterpolationError:
            return None
        return int(target_timestamp_ns), projected

    def _find_camera_synchronized_frames(
        self,
        master_timestamp_ns: int,
        master_frame: tuple,
    ) -> Optional[List[Optional[tuple]]]:
        """Align low-dimensional streams to one camera frame.

        Other camera/image streams retain closest-frame matching. Every
        low-dimensional stream declares a format that selects linear or SLERP
        interpolation for each ordered segment.
        """
        device_frames: List[Optional[tuple]] = []
        for device_index, device in enumerate(self.devices):
            if device["device_class_id"] == self.master_device_class_id:
                device_frames.append(master_frame)
                continue

            if not self._connect_to_device(device):
                logger.warning("Cannot connect to device %s", device["device_class_id"])
                return None

            adjusted_timestamp = self._calculate_adjusted_timestamp(
                master_timestamp_ns, device
            )
            data_format = self._device_data_format(device)
            if data_format is None:
                search_result = self._search_device_buffer_for_timestamp(
                    device, adjusted_timestamp
                )
                if search_result is None:
                    logger.warning(
                        "Cannot find image frame synchronized to camera for device %s",
                        device["device_class_id"],
                    )
                    return None
                timestamp_ns, data_array, _ = search_result
                device_frames.append((timestamp_ns, data_array))
                continue

            bracket = self._search_device_buffer_bracket(
                device, adjusted_timestamp
            )
            if bracket is None:
                interpolated_frame = self._extrapolate_latest_robot_frame(
                    device, adjusted_timestamp, data_format
                )
            else:
                interpolated_frame = self._interpolate_bracketed_frame(
                    bracket,
                    adjusted_timestamp,
                    data_format,
                )
            if interpolated_frame is None:
                return None
            device_frames.append(interpolated_frame)

        return device_frames

    def _search_device_buffer_for_timestamp(self, device: Dict[str, Any], target_timestamp_ns: int) -> Optional[tuple]:
        """
        Search device buffer for frame with timestamp closest to target using binary search.

        Args:
            device: Device dictionary
            target_timestamp_ns: Target timestamp in nanoseconds

        Returns:
            tuple: (timestamp_ns, data_array, frame_index) or None
        """
        buffer_info = self._get_device_buffer_info(device)
        if buffer_info is None:
            return None

        buffer_size, current_frames_count, write_index, frame_size, shape = buffer_info

        if current_frames_count == 0:
            return None

        try:
            # Binary search on buffer indices (frames are already time-ordered)
            result = self._binary_search_buffer_indices(
                device, target_timestamp_ns, buffer_info
            )

            if result is None:
                return None

            # Return the result directly (timestamp_ns, data_array, frame_index)
            return result

        except Exception as e:
            logger.error(f"Error searching buffer for timestamp {target_timestamp_ns} in device {device['device_class_id']}: {e}")
            return None

    def _binary_search_buffer_indices(self, device: Dict[str, Any], target_timestamp_ns: int, buffer_info: tuple) -> Optional[tuple]:
        """
        Binary search on buffer indices to find the frame with closest timestamp.
        Handles circular buffer time ordering correctly.

        Args:
            device: Device dictionary
            target_timestamp_ns: Target timestamp in nanoseconds
            buffer_info: Buffer information tuple

        Returns:
            tuple: (timestamp_ns, data_array, frame_index) or None if not found
        """
        buffer_size, current_frames_count, write_index, _, _ = buffer_info

        if current_frames_count == 0:
            return None

        if current_frames_count < buffer_size:
            return self._binary_search_range(device, target_timestamp_ns, 0, current_frames_count - 1, buffer_info)

        return self._binary_search_range(device, target_timestamp_ns, write_index, write_index + buffer_size - 1, buffer_info)

    def _binary_search_range(self, device: Dict[str, Any], target_timestamp_ns: int, start_idx: int, end_idx: int, buffer_info: tuple) -> Optional[tuple]:
        """
        Binary search in a range of buffer indices (time-ordered).
        Handles circular buffer by converting virtual indices to actual buffer indices.

        Args:
            device: Device dictionary
            target_timestamp_ns: Target timestamp in nanoseconds
            start_idx: Start index (inclusive, can be virtual for circular buffer)
            end_idx: End index (inclusive, can be virtual for circular buffer)
            buffer_info: Buffer information tuple

        Returns:
            tuple: (timestamp_ns, data_array, frame_index) or None if not found
        """
        if start_idx > end_idx:
            return None

        buffer_size = buffer_info[0]  # Get buffer_size

        left, right = start_idx, end_idx
        closest_result = None
        closest_diff = float('inf')

        while left <= right:
            mid = (left + right) // 2

            # Convert virtual index to actual buffer index
            actual_idx = mid % buffer_size

            # Read timestamp at actual index
            frame_data = self._read_device_frame(device, actual_idx)
            if frame_data is None:
                # Skip invalid frame, continue search
                right = mid - 1
                continue

            timestamp_ns, data_array = frame_data
            current_diff = abs(timestamp_ns - target_timestamp_ns)

            # Update closest if current is better
            if current_diff < closest_diff:
                closest_diff = current_diff
                closest_result = (timestamp_ns, data_array, actual_idx)

            # If we found exact match, return it
            if timestamp_ns == target_timestamp_ns:
                return timestamp_ns, data_array, actual_idx

            # Binary search logic
            if timestamp_ns < target_timestamp_ns:
                left = mid + 1
            else:
                right = mid - 1

        return closest_result

    def _apply_hardware_latency_compensation(self, device: Dict[str, Any], timestamp_ns: int) -> int:
        """
        Apply hardware latency compensation to timestamp.

        Args:
            device: Device dictionary
            timestamp_ns: Original timestamp in nanoseconds

        Returns:
            int: Compensated timestamp in nanoseconds
        """
        hardware_latency_ms = self._get_device_hardware_latency(device)
        hardware_latency_ns = int(hardware_latency_ms * 1_000_000)  # Convert ms to ns
        compensated_timestamp_ns = timestamp_ns - hardware_latency_ns

        logger.debug(f"Device {device['device_class_id']}: original_timestamp={timestamp_ns}, "
                    f"hardware_latency={hardware_latency_ms}ms, "
                    f"compensated_timestamp={compensated_timestamp_ns}")

        return compensated_timestamp_ns

    def write_frames_to_summary_shm(self, device_frames: List[Optional[tuple]]) -> None:
        """
        Write device frames to summary SHM buffer with hardware latency compensation.

        This method overrides the base implementation to apply hardware latency compensation
        to timestamps before writing them to the summary SHM.

        Args:
            device_frames: List of (timestamp_ns, data_array) tuples for each device, or None if failed
        """
        # Apply hardware latency compensation to each device frame
        compensated_frames = []
        for i, (device, frame_data) in enumerate(zip(self.devices, device_frames)):
            if frame_data is None:
                compensated_frames.append(None)
                continue

            timestamp_ns, data_array = frame_data
            compensated_timestamp_ns = self._apply_hardware_latency_compensation(device, timestamp_ns)
            compensated_frames.append((compensated_timestamp_ns, data_array))

        # Call parent method with compensated frames
        super().write_frames_to_summary_shm(compensated_frames)

    def _read_all_device_frames(self) -> Optional[List[Optional[tuple]]]:
        """
        Read synchronized frames from all devices.

        Returns:
            List of (timestamp_ns, data_array) tuples for each device, or None if master device hasn't updated
        """
        # Get master device by index
        master_device = next(d for d in self.devices if d["device_class_id"] == self.master_device_class_id)

        # Read master device frame first
        master_frame = self._read_device_frame(master_device)
        if master_frame is None:
            logger.error("Cannot read master device frame")
            return None

        master_timestamp_ns, _ = master_frame

        # Check if master device has new data
        if self.master_last_timestamp is not None and master_timestamp_ns == self.master_last_timestamp:
            return None  # Master device hasn't updated, skip update

        # Camera-master mode uses interpolation for explicitly typed low-dimensional
        # streams. The legacy/default path retains closest-frame binary matching.
        if getattr(self, "sync_mode", BINARY_MATCH) == CAMERA_ALIGN:
            synchronized_frames = self._find_camera_synchronized_frames(
                master_timestamp_ns,
                master_frame,
            )
            # A newer robot sample may arrive after this camera frame. Only
            # consume the frame on success so the next poll can retry it.
            if synchronized_frames is not None:
                self.master_last_timestamp = master_timestamp_ns
        else:
            self.master_last_timestamp = master_timestamp_ns
            synchronized_frames = self._find_synchronized_frames(
                master_timestamp_ns,
                master_frame,
            )

        return synchronized_frames


def main() -> None:
    """Main entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Synchronized Device Manager with buffer support")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--buffer-size", "-b", type=int, default=100,
                        help="Buffer size for summary SHM (default: 100)")

    args = parser.parse_args()

    manager = SynchronizedDeviceManager(
        config_path=args.config,
        buffer_size=args.buffer_size
    )

    manager.start()


if __name__ == "__main__":
    main()
