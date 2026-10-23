#!/usr/bin/env python3
"""
Policy history helper mixins.

Author: OpenAI Assistant
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger_config import logger
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_shared_memory,
    unpack_buffer_header,
    unpack_device_header,
    unpack_frame_header,
)


class PolicyHistoryTimingMixin:
    """Mixin containing a cohesive subset of policy-history behavior."""

    def clear_state(self) -> None:
        """Clear pending policy-history timing state."""
        self.owner._pending_policy_history_spec = None
        self.owner._policy_history_arrival_wait_start_ns = None

    def attach_action_chunk_timing(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        first_action_sent_timestamp_ns: int,
    ) -> None:
        """Record the action chunk timeline used by arrival-anchored history."""
        owner = self.owner
        history_spec = owner._pending_policy_history_spec
        if history_spec is None:
            return
        if str(history_spec.get("anchor_mode", "now")) != "arrival":
            return
        chunk_lengths = [
            int(chunk.shape[0])
            for chunk in action_chunk_dict.values()
            if (
                isinstance(chunk, np.ndarray)
                and chunk.ndim >= 1
                and chunk.shape[0] > 0
            )
        ]
        if not chunk_lengths:
            return

        chunk_length = max(chunk_lengths)
        first_action_sent_timestamp_ns = int(first_action_sent_timestamp_ns)
        last_action_sent_timestamp_ns = first_action_sent_timestamp_ns + (
            (chunk_length - 1) * owner.action_step_interval_ns
        )
        history_spec["action_chunk_first_action_sent_timestamp_ns"] = first_action_sent_timestamp_ns
        history_spec["action_chunk_last_action_sent_timestamp_ns"] = int(last_action_sent_timestamp_ns)
        history_spec["action_chunk_step_interval_ns"] = int(owner.action_step_interval_ns)
        history_spec["action_chunk_length"] = int(chunk_length)
        history_spec["action_chunk_device_names"] = list(action_chunk_dict.keys())

    def control_shm_name_for_device(self, device_name: str) -> Optional[str]:
        """Return configured control SHM name for a device."""
        for cfg in self.owner.controls_config:
            if cfg.get("device_key") == device_name:
                return str(cfg["shm_name"])
        return None

    def get_control_shm(self, shm_name: str) -> Optional[Any]:
        """Connect to and cache a control SHM used for history timing."""
        cached = self.owner._policy_history_control_shms.get(shm_name)
        if cached is not None:
            return cached
        try:
            control_shm = connect_to_shared_memory(shm_name, read_only=True)
        except Exception as exc:
            logger.warning(
                f"PolicyConnector: Cannot read control SHM '{shm_name}' for "
                f"arrival history timing ({exc})"
            )
            return None
        self.owner._policy_history_control_shms[shm_name] = control_shm
        return control_shm

    def read_control_action_timestamps_after(
        self,
        device_name: str,
        after_timestamp_ns: int,
    ) -> List[int]:
        """Read control frame timestamps after a given timestamp."""
        shm_name = self.control_shm_name_for_device(device_name)
        if shm_name is None:
            logger.warning(
                f"PolicyConnector: No control SHM configured for {device_name}; "
                "cannot verify arrival history dispatch timing"
            )
            return []

        control_shm = self.get_control_shm(shm_name)
        if control_shm is None:
            return []

        try:
            buf = control_shm.buf
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            buffer_header = unpack_buffer_header(
                buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
            frame_size = int(device_header["frame_size"])
            buffer_size = int(buffer_header["buffer_size"])

            timestamps: List[int] = []
            for frame_index in range(buffer_size):
                frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_index * frame_size
                frame_header = unpack_frame_header(buf[frame_offset:frame_offset + FRAME_HEADER_SIZE])
                timestamp_ns = int(frame_header["timestamp_ns"])
                if timestamp_ns > int(after_timestamp_ns):
                    timestamps.append(timestamp_ns)
            return sorted(timestamps)
        except Exception as exc:
            logger.warning(
                f"PolicyConnector: Failed reading control SHM timestamps for "
                f"{device_name} ({exc})"
            )
            return []

    def resolve_last_dispatch_timestamp(
        self,
        history_spec: Dict[str, Any],
    ) -> Optional[int]:
        """Resolve the last dispatched timestamp for an arrival-anchored action chunk."""
        first_action_sent_timestamp_ns = history_spec.get(
            "action_chunk_first_action_sent_timestamp_ns"
        )
        chunk_length = int(history_spec.get("action_chunk_length", 0))
        device_names = [
            str(name) for name in history_spec.get("action_chunk_device_names", [])
        ]
        if (
            first_action_sent_timestamp_ns is None
            or chunk_length <= 0
            or not device_names
        ):
            return None

        last_dispatch_timestamps: List[int] = []
        dispatch_counts: Dict[str, int] = {}
        for device_name in device_names:
            timestamps = self.read_control_action_timestamps_after(
                device_name=device_name,
                after_timestamp_ns=int(first_action_sent_timestamp_ns),
            )
            dispatch_counts[device_name] = len(timestamps)
            if len(timestamps) < chunk_length:
                history_spec["action_chunk_dispatch_counts"] = dispatch_counts
                return None
            last_dispatch_timestamps.append(int(timestamps[chunk_length - 1]))

        last_dispatch_timestamp_ns = max(last_dispatch_timestamps)
        history_spec["action_chunk_dispatch_counts"] = dispatch_counts
        history_spec["action_chunk_last_action_dispatched_timestamp_ns"] = int(
            last_dispatch_timestamp_ns
        )
        return int(last_dispatch_timestamp_ns)
