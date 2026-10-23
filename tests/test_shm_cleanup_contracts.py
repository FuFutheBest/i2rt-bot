"""Shared-memory cleanup regression tests."""

from multiprocessing import shared_memory
from types import SimpleNamespace
from uuid import uuid4

import pytest

from consumer.inference.common.policy_runtime_io import PolicyRuntimeIO
from devices.robots.rizon.home import RizonHomeMixin
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    DataChannel,
    connect_to_shared_memory,
    create_shared_memory,
    pack_buffer_header,
    pack_device_header,
    unpack_buffer_header,
)


def test_data_channel_releases_owned_buffer_before_close() -> None:
    raw = shared_memory.SharedMemory(
        name=f"superinference_test_{uuid4().hex}", create=True, size=64
    )
    channel = DataChannel(raw, read_only=True)
    try:
        channel.close()
    finally:
        raw.unlink()


def test_data_channel_close_releases_cached_readonly_view() -> None:
    raw = shared_memory.SharedMemory(
        name=f"superinference_test_{uuid4().hex}", create=True, size=64
    )
    channel = DataChannel(raw, read_only=True)
    view = channel.buf
    try:
        assert channel.buf is view
        channel.close()
    finally:
        raw.unlink()


def test_policy_runtime_reconnect_releases_held_readonly_summary_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_summary = shared_memory.SharedMemory(
        name=f"superinference_summary_{uuid4().hex}", create=True, size=64
    )
    raw_policy = shared_memory.SharedMemory(
        name=f"superinference_policy_{uuid4().hex}", create=True, size=64
    )
    raw_replacement_summary = shared_memory.SharedMemory(
        name=f"superinference_summary_{uuid4().hex}", create=True, size=64
    )
    raw_replacement_policy = shared_memory.SharedMemory(
        name=f"superinference_policy_{uuid4().hex}", create=True, size=64
    )
    old_summary = DataChannel(raw_summary, read_only=True)
    old_policy = DataChannel(raw_policy)
    replacement_summary = DataChannel(raw_replacement_summary, read_only=True)
    replacement_policy = DataChannel(raw_replacement_policy)
    held_view = old_summary.buf
    owner = SimpleNamespace(summary_shm=old_summary, policy_shm=old_policy)

    def connect() -> bool:
        owner.summary_shm = replacement_summary
        return True

    def connect_policy_shm() -> None:
        owner.policy_shm = replacement_policy

    owner.connect = connect
    monkeypatch.setattr(
        PolicyRuntimeIO,
        "_connect_policy_shm",
        lambda _runtime_io: connect_policy_shm(),
    )
    try:
        PolicyRuntimeIO(owner)._reconnect_all_shms()

        assert owner.summary_shm is replacement_summary
        assert owner.policy_shm is replacement_policy
        with pytest.raises(ValueError):
            len(held_view)
    finally:
        replacement_summary.close()
        replacement_policy.close()
        raw_summary.unlink()
        raw_policy.unlink()
        raw_replacement_summary.unlink()
        raw_replacement_policy.unlink()


def test_rizon_home_clears_control_shm_through_temporary_writer() -> None:
    name = f"superinference_control_test_{uuid4().hex}"
    buffer_size = 2
    frame_size = 16
    owner = create_shared_memory(
        name,
        DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + buffer_size * frame_size,
    )
    owner.buf[:DEVICE_HEADER_SIZE] = pack_device_header(
        "RizonRobot", 2, 30.0, "float64", (1,), frame_size
    )
    owner.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = (
        pack_buffer_header(buffer_size, 2, 1)
    )
    reader = connect_to_shared_memory(name, read_only=True)
    robot = SimpleNamespace(
        control_shared_memory={name: reader},
        _owned_control_shm_names={name},
        use_cartesian_motion_planner=False,
    )
    try:
        RizonHomeMixin._clear_control_shm(robot)
        header = unpack_buffer_header(
            bytes(
                owner.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
        )
        assert header["current_frames_count"] == 0
        assert header["write_index"] == 0
        assert robot.control_shared_memory[name] is reader
        assert reader.is_read_only
    finally:
        reader.close()
        owner.close()
        owner.unlink()
