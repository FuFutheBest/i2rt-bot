#!/usr/bin/env python3
"""Shared-memory runtime IO for PolicyConnector.

Author: Jun Lv, Han Xue, Zheng Wang
"""

from __future__ import annotations

import struct
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from utils.logger_config import logger
from utils.shm_utils import (
    ACTION_INDEX_FORMAT,
    ACTION_INDEX_SIZE,
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    POLICY_HEADER_SIZE,
    connect_to_policy_shm,
    get_device_info_offset,
    get_dtype,
    get_policy_action_index_offset,
    unpack_buffer_header,
    unpack_device_header,
    unpack_frame_header,
    unpack_policy_header,
)


class PolicyRuntimeIO:
    """Own policy/summary SHM connection, metadata, and read helpers."""

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def find_value_by_timestamp_with_history_too_old_detection(
        self, target_timestamp_ns: int, tolerance_ms: float = 50.0
    ) -> Optional[Union[np.ndarray, str]]:
        """Find the closest value prediction and detect when the requested timestamp is too old."""
        try:
            if (
                not self.value_shm.created
                or self.value_shm.shm is None
                or self.value_shm.shape is None
            ):
                logger.warning("PolicyConnector: Value SHM not created or invalid")
                return None

            buf = self.value_shm.shm.buf
            buffer_header = unpack_buffer_header(
                buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
            buffer_size = buffer_header["buffer_size"]
            current_frames_count = buffer_header["current_frames_count"]
            write_index = buffer_header["write_index"]
            if current_frames_count == 0:
                logger.warning("PolicyConnector: Value SHM is empty")
                return None

            tolerance_ns = int(tolerance_ms * 1_000_000)

            if current_frames_count < buffer_size:
                start_idx = 0
                end_idx = current_frames_count - 1
            else:
                start_idx = write_index
                end_idx = write_index + buffer_size - 1

            def _value_frame_timestamp(frame_idx: int) -> int:
                frame_offset = (
                    DEVICE_HEADER_SIZE
                    + BUFFER_HEADER_SIZE
                    + frame_idx * self.value_shm.frame_size
                )
                return unpack_frame_header(
                    buf[frame_offset : frame_offset + FRAME_HEADER_SIZE]
                )["timestamp_ns"]

            best_frame_index = None
            best_timestamp_diff = float("inf")
            left, right = start_idx, end_idx
            while left <= right:
                mid = (left + right) // 2
                actual_frame_idx = mid % buffer_size
                frame_timestamp_ns = _value_frame_timestamp(actual_frame_idx)
                timestamp_diff = abs(frame_timestamp_ns - target_timestamp_ns)
                if timestamp_diff < best_timestamp_diff:
                    best_timestamp_diff = timestamp_diff
                    best_frame_index = actual_frame_idx
                if frame_timestamp_ns < target_timestamp_ns:
                    left = mid + 1
                elif frame_timestamp_ns > target_timestamp_ns:
                    right = mid - 1
                else:
                    best_timestamp_diff = 0
                    break

            if best_frame_index is not None and best_timestamp_diff <= tolerance_ns:
                frame_offset = (
                    DEVICE_HEADER_SIZE
                    + BUFFER_HEADER_SIZE
                    + best_frame_index * self.value_shm.frame_size
                )
                data_start = frame_offset + FRAME_HEADER_SIZE
                data_end = data_start + self.value_shm.frame_data_size
                value = np.frombuffer(
                    buf[data_start:data_end],
                    dtype=get_dtype(self.value_shm.dtype_str, np.float64),
                ).copy()
                return value.reshape(self.value_shm.shape)

            logger.warning(
                "No value prediction found within tolerance. "
                f"best_diff={best_timestamp_diff / 1_000_000:.2f}ms, tolerance={tolerance_ms}ms"
            )
            return None

        except Exception as e:
            logger.error(
                f"Error finding value by timestamp with no-history detection: {e}"
            )
            return None

    def _reconnect_all_shms(self) -> None:
        """Reconnect to all SHMs (summary SHM and policy SHM) after reset signal."""
        max_retries = 50
        retry_delay = 0.5  # Wait 0.5 seconds between retries

        # Store old connections in case reconnection fails (store references before closing)
        old_summary_shm = None
        old_policy_shm = None

        # Do not probe ``.buf`` here: a readonly DataChannel creates a
        # memoryview for that property, which would keep the old mapping open
        # while reconnect closes it below.
        if hasattr(self, "summary_shm") and self.summary_shm is not None:
            old_summary_shm = self.summary_shm

        # Closing an already-invalid channel is handled by the existing close
        # guard below, so no buffer access is needed before reconnecting.
        if self.policy_shm is not None:
            old_policy_shm = self.policy_shm

        # Try to reconnect summary SHM with retries
        summary_reconnected = False
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"PolicyConnector: Reconnecting summary SHM... (attempt {attempt + 1}/{max_retries})"
                )
                # Close existing summary SHM connection only if we have a valid old one
                # Use 'is not None' to avoid triggering __len__ or __bool__ on potentially released objects
                if old_summary_shm is not None and self.summary_shm == old_summary_shm:
                    try:
                        old_summary_shm.close()
                    except Exception:
                        pass
                    self.summary_shm = None
                    old_summary_shm = None  # Clear reference after closing

                # Reconnect to summary SHM
                if self.connect():
                    summary_reconnected = True
                    logger.info(
                        "PolicyConnector: Successfully reconnected to summary SHM"
                    )
                    break
                else:
                    logger.warning(
                        f"PolicyConnector: Failed to reconnect to summary SHM (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                logger.warning(
                    f"PolicyConnector: Error reconnecting summary SHM (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        if not summary_reconnected:
            logger.error(
                "PolicyConnector: Failed to reconnect to summary SHM after all retries"
            )
            # Don't try to restore old connection - it's likely already invalid
            # Will retry in next cycle when reset signal is detected again

        # Try to reconnect policy SHM with retries
        policy_reconnected = False
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"PolicyConnector: Reconnecting policy SHM... (attempt {attempt + 1}/{max_retries})"
                )
                # Close existing policy SHM connection only if we have a valid old one
                # Use 'is not None' to avoid triggering __len__ or __bool__ on potentially released objects
                if old_policy_shm is not None and self.policy_shm == old_policy_shm:
                    try:
                        old_policy_shm.close()
                    except Exception:
                        pass
                    self.policy_shm = None
                    old_policy_shm = None  # Clear reference after closing

                # Reconnect to policy SHM
                self._connect_policy_shm()
                policy_reconnected = True
                logger.info("PolicyConnector: Successfully reconnected to policy SHM")
                break
            except FileNotFoundError:
                logger.warning(
                    f"PolicyConnector: Policy SHM not found (attempt {attempt + 1}/{max_retries}), waiting..."
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                logger.warning(
                    f"PolicyConnector: Error reconnecting policy SHM (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        if not policy_reconnected:
            logger.error(
                "PolicyConnector: Failed to reconnect to policy SHM after all retries"
            )
            # Don't try to restore old connection - it's likely already invalid
            # Will retry in next cycle when reset signal is detected again

        if summary_reconnected and policy_reconnected:
            logger.info(
                "PolicyConnector: Successfully reconnected all SHMs after reset"
            )
        else:
            logger.warning(
                "PolicyConnector: Partial reconnection - some SHMs may need retry in next cycle"
            )

    def _merge_action_chunks(
        self, action_chunk_dict: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Merge multi-device action chunks into a single array for recording.
        Chunks are concatenated along the action dimension in sorted device order.
        """
        if not action_chunk_dict:
            return None
        try:
            device_names = sorted(action_chunk_dict.keys())
            chunks = [action_chunk_dict[name] for name in device_names]
            base_len = chunks[0].shape[0]
            if any(c.shape[0] != base_len for c in chunks):
                logger.warning(
                    "Action chunk merge skipped due to mismatched chunk lengths"
                )
                return None
            merged = np.concatenate(chunks, axis=1)
            return merged
        except Exception as exc:
            logger.warning(f"Failed to merge action chunks: {exc}")
            return None

    def _connect_policy_shm(self) -> None:
        """Connect to policy SHM created by ActionExecutor."""
        try:
            start_time = time.time()
            timeout_sec = max(10.0, 60.0)
            while True:
                try:
                    # Connect to existing policy SHM using unified interface (read-write for policy connector)
                    self.policy_shm = connect_to_policy_shm(read_only=False)
                    break
                except Exception as exc:
                    if time.time() - start_time < timeout_sec:
                        logger.info(
                            f"PolicyConnector: Waiting for policy SHM 'policy_data'... retry in {self.retry_connect_secs}s"
                        )
                        time.sleep(self.retry_connect_secs)
                        continue
                    raise exc

            # Read manager header to get device count and cache device info
            buf = self.policy_shm.buf
            manager_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
            device_count = manager_header["device_count"]

            # Cache device info for efficient writing
            self.policy_devices = []
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(
                    buf[device_offset : device_offset + DEVICE_HEADER_SIZE]
                )

                device_name = device_header["device_type"]
                shape = device_header["shape"]  # [chunk_length, action_dim]
                frame_size = device_header["frame_size"]

                self.policy_devices.append(
                    {
                        "device_name": device_name,
                        "shape": shape,
                        "frame_size": frame_size,
                        "data_offset": None,
                    }
                )

            for device_info in self.policy_devices:
                logger.debug(
                    f"Device {device_info['device_name']}: expected_shape={device_info['shape']}"
                )

        except Exception as e:
            logger.error(f"PolicyConnector: Failed to connect to policy SHM: {e}")
            raise

    def _read_policy_action_index(self) -> Optional[int]:
        """Read action_index stored at the tail of policy SHM."""
        if not self.policy_shm or not self.policy_devices:
            return None
        try:
            device_count = len(self.policy_devices)
            frame_sizes = [device["frame_size"] for device in self.policy_devices]
            offset = get_policy_action_index_offset(device_count, frame_sizes)
            return struct.unpack(
                ACTION_INDEX_FORMAT,
                self.policy_shm.buf[offset : offset + ACTION_INDEX_SIZE],
            )[0]
        except Exception as exc:
            logger.warning(
                f"PolicyConnector: Failed to read action_index from policy SHM: {exc}"
            )
            return None

    def _get_prediction_action_index(
        self,
        iteration_count: int,
        read_action_index: Optional[Callable[[], Optional[int]]] = None,
    ) -> Optional[int]:
        """Return RTC action progress while keeping the first prediction stateless."""
        if not getattr(self.policy, "enable_rtc", False):
            return None
        reader = read_action_index or self._read_policy_action_index
        action_index_value = reader()
        action_index = (
            max(0, int(action_index_value)) if action_index_value is not None else None
        )
        return None if iteration_count == 0 else action_index

    def _build_action_configs(self) -> List[Dict[str, Any]]:
        action_configs: List[Dict[str, Any]] = []
        for cfg in self.controls_config:
            shm_name = cfg["shm_name"]
            device_name = cfg["device_key"]
            shape = cfg.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                # policy_data transports each control target as one flattened
                # float64 row; the executor restores the target SHM shape.
                action_dim = int(np.prod(shape))
            else:
                raise ValueError(
                    f"Invalid shape for control device {device_name}: {shape}"
                )
            action_configs.append(
                {
                    "device_name": device_name,
                    "action_dim": action_dim,
                    "shm_name": shm_name,
                }
            )
        return action_configs

    def _get_action_chunk_shape(self) -> tuple:
        """
        Compute merged action chunk shape based on action_configs.
        Returns (chunk_length, total_action_dim).
        """
        total_dim = 0
        for cfg in self.action_configs:
            total_dim += int(cfg.get("action_dim", 0))
        if total_dim <= 0:
            total_dim = 1
        return (self.chunk_length, total_dim)
