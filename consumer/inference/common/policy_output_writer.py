#!/usr/bin/env python3
"""
Policy output writer for action chunk SHM writes.

Author: OpenAI Assistant
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.logger_config import logger
from utils.shm_utils import (
    FRAME_HEADER_SIZE,
    POLICY_HEADER_SIZE,
    get_policy_data_offset,
    pack_frame_header,
    pack_policy_header,
)


PolicyShmGetter = Callable[[], Any]
PolicyDevicesGetter = Callable[[], List[Dict[str, Any]]]
ReconnectCallback = Callable[[], None]


class PolicyOutputWriter:
    """Write dict-based action chunks to the policy SHM contract."""

    def __init__(
        self,
        *,
        policy_shm_getter: PolicyShmGetter,
        policy_devices_getter: PolicyDevicesGetter,
        reconnect_policy_shm: ReconnectCallback,
        timestamp_logger: Any,
    ) -> None:
        self.policy_shm_getter = policy_shm_getter
        self.policy_devices_getter = policy_devices_getter
        self.reconnect_policy_shm = reconnect_policy_shm
        self.timestamp_logger = timestamp_logger

    def write_action_chunk_dict(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        observation_timestamp: int = -1,
    ) -> Optional[int]:
        """Write action chunks and return the policy write timestamp."""
        policy_shm = self.policy_shm_getter()
        policy_devices = self.policy_devices_getter()
        if not policy_shm:
            logger.warning("PolicyConnector: policy_shm is None, cannot write action chunks")
            return None

        try:
            buf = policy_shm.buf
            timestamp_ns = time.time_ns()

            device_count = len(policy_devices)
            manager_header = pack_policy_header(device_count, timestamp_ns)
            buf[:POLICY_HEADER_SIZE] = manager_header

            frame_sizes = [device["frame_size"] for device in policy_devices]

            for i, device_info in enumerate(policy_devices):
                device_name = device_info["device_name"]
                expected_shape = device_info["shape"]

                if device_name not in action_chunk_dict:
                    logger.warning(f"No action chunk for device: {device_name}")
                    continue

                action_chunk = action_chunk_dict[device_name]
                if len(action_chunk.shape) == 1:
                    action_chunk = action_chunk.reshape(1, -1)

                if action_chunk.shape != tuple(expected_shape):
                    logger.error(
                        f"Shape mismatch for {device_name}: got {action_chunk.shape}, expected {expected_shape}"
                    )
                    continue

                data_offset = get_policy_data_offset(i, device_count, frame_sizes)

                frame_header = pack_frame_header(observation_timestamp)
                buf[data_offset:data_offset + FRAME_HEADER_SIZE] = frame_header

                data_start = data_offset + FRAME_HEADER_SIZE
                chunk_bytes = action_chunk.astype(np.float64).tobytes()
                data_end = data_start + len(chunk_bytes)
                buf[data_start:data_end] = chunk_bytes
                self.timestamp_logger.log_event(
                    "action_chunk_written",
                    device_name=device_name,
                    observation_timestamp_ns=observation_timestamp,
                    policy_write_timestamp_ns=timestamp_ns,
                    data_offset=data_offset,
                    chunk_shape=list(action_chunk.shape),
                    first_action=action_chunk[0] if action_chunk.shape[0] > 0 else None,
                )

            logger.debug(f"Wrote action chunks to policy SHM: devices={list(action_chunk_dict.keys())}")
            return timestamp_ns

        except (OSError, ValueError, AttributeError) as exc:
            logger.warning(f"Failed to write action chunk dict (SHM may have been recreated): {exc}")
            logger.info("PolicyConnector: Attempting to reconnect to policy SHM...")
            try:
                self.reconnect_policy_shm()
                logger.info("PolicyConnector: Successfully reconnected to policy SHM")
            except Exception as reconnect_error:
                logger.error(f"PolicyConnector: Failed to reconnect to policy SHM: {reconnect_error}")
            return None
        except Exception as exc:
            logger.error(f"Failed to write action chunk dict: {exc}")
            return None
