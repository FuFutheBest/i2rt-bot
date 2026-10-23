"""Contract tests for shared-memory packing, sizes, and offsets.

Author: OpenAI Assistant
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import shm_utils


@pytest.mark.unit
def test_device_header_pack_unpack_round_trip() -> None:
    """Device headers should preserve the public SHM metadata contract."""
    packed = shm_utils.pack_device_header(
        device_type="BaseDevice",
        device_class_id=2,
        fps=30.0,
        data_dtype="float64",
        shape=(3, 4),
        frame_size=shm_utils.FRAME_HEADER_SIZE + 3 * 4 * np.dtype(np.float64).itemsize,
        hardware_latency_ms=12.5,
    )

    unpacked = shm_utils.unpack_device_header(packed)

    assert len(packed) == shm_utils.DEVICE_HEADER_SIZE
    assert unpacked["device_type"] == "BaseDevice"
    assert unpacked["device_class_id"] == 2
    assert unpacked["fps"] == pytest.approx(30.0)
    assert unpacked["data_dtype"] == "float64"
    assert unpacked["shape"] == (3, 4)
    assert unpacked["hardware_latency_ms"] == pytest.approx(12.5)


@pytest.mark.unit
def test_manager_policy_and_frame_headers_round_trip() -> None:
    """Manager, policy, and frame headers should round-trip timestamp fields."""
    manager_header = shm_utils.pack_manager_header(
        device_count=2,
        buffer_size=4,
        current_frames_count=3,
        write_index=1,
        timestamp=123456789,
    )
    policy_header = shm_utils.pack_policy_header(device_count=2, timestamp=987654321)
    frame_header = shm_utils.pack_frame_header(timestamp_ns=555)
    summary_frame_header = shm_utils.pack_summary_frame_header(frame_timestamp_ns=777)

    assert shm_utils.unpack_manager_header(manager_header) == {
        "device_count": 2,
        "buffer_size": 4,
        "current_frames_count": 3,
        "write_index": 1,
        "update_timestamp": 123456789,
    }
    assert shm_utils.unpack_policy_header(policy_header)["update_timestamp"] == 987654321
    assert shm_utils.unpack_frame_header(frame_header)["timestamp_ns"] == 555
    assert shm_utils.unpack_summary_frame_header(summary_frame_header)["frame_timestamp_ns"] == 777


@pytest.mark.unit
def test_size_calculations_match_documented_layouts() -> None:
    """Size helpers should match the documented binary layouts."""
    frame_data_size = 24
    frame_sizes = [
        shm_utils.FRAME_HEADER_SIZE + 24,
        shm_utils.FRAME_HEADER_SIZE + 8,
    ]

    assert shm_utils.calculate_device_shm_size(3, frame_data_size) == (
        shm_utils.DEVICE_HEADER_SIZE
        + shm_utils.BUFFER_HEADER_SIZE
        + 3 * (shm_utils.FRAME_HEADER_SIZE + frame_data_size)
    )
    assert shm_utils.calculate_summary_shm_size(2, frame_sizes, 5) == (
        shm_utils.MANAGER_HEADER_SIZE
        + 2 * shm_utils.DEVICE_HEADER_SIZE
        + 5 * (shm_utils.SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes))
    )
    assert shm_utils.calculate_policy_shm_size(2, frame_sizes) == (
        shm_utils.MANAGER_HEADER_SIZE
        + 2 * shm_utils.DEVICE_HEADER_SIZE
        + sum(frame_sizes)
        + shm_utils.ACTION_INDEX_SIZE
    )


@pytest.mark.unit
def test_offset_helpers_are_monotonic_and_layout_consistent() -> None:
    """Offset helpers should point to deterministic non-overlapping regions."""
    frame_sizes = [16, 24, 32]
    device_count = len(frame_sizes)
    buffer_size = 4

    assert shm_utils.get_device_info_offset(0) == shm_utils.MANAGER_HEADER_SIZE
    assert shm_utils.get_device_info_offset(1) == (
        shm_utils.MANAGER_HEADER_SIZE + shm_utils.DEVICE_HEADER_SIZE
    )
    assert shm_utils.get_data_offset(2, device_count, frame_sizes) == (
        shm_utils.MANAGER_HEADER_SIZE + device_count * shm_utils.DEVICE_HEADER_SIZE + 40
    )
    assert shm_utils.get_policy_action_index_offset(device_count, frame_sizes) == (
        shm_utils.MANAGER_HEADER_SIZE + device_count * shm_utils.DEVICE_HEADER_SIZE + sum(frame_sizes)
    )

    frame_1_start = shm_utils.get_summary_frame_start_offset(1, device_count, frame_sizes)
    frame_1_device_2 = shm_utils.get_summary_frame_offset(2, 1, device_count, frame_sizes, buffer_size)
    assert frame_1_device_2 == frame_1_start + shm_utils.SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes[:2])


@pytest.mark.unit
def test_shape_validation_rejects_too_many_dimensions() -> None:
    """Shape limits are part of the public SHM contract."""
    with pytest.raises(ValueError):
        shm_utils.pack_device_header(
            device_type="TooWide",
            device_class_id=0,
            fps=1.0,
            data_dtype="float32",
            shape=(1, 2, 3, 4, 5),
            frame_size=8,
        )
