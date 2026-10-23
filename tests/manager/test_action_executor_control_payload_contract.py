"""Payload-contract tests for action-executor control SHM writes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from consumer.inference.common.policy_runtime_io import PolicyRuntimeIO
from manager.base_policy_runner import BasePolicyRunner
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    pack_buffer_header,
    pack_device_header,
    unpack_buffer_header,
    unpack_device_header,
)


def _target(*, shape: tuple[int, ...] = (2, 3), dtype: str = "float32") -> dict[str, Any]:
    return {
        "device_key": "Robot_0",
        "shm_name": "Robot_0_control",
        "shape": shape,
        "dtype": dtype,
    }


def _headers(
    *,
    shape: tuple[int, ...] = (2, 3),
    dtype: str = "float32",
    frame_size: int | None = None,
    buffer_size: int = 2,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    payload_size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual_frame_size = (
        FRAME_HEADER_SIZE + payload_size if frame_size is None else int(frame_size)
    )
    device_header = unpack_device_header(
        pack_device_header(
            "Robot",
            0,
            30.0,
            dtype,
            shape,
            actual_frame_size,
        )
    )
    buffer_header = unpack_buffer_header(pack_buffer_header(buffer_size, 0, 0))
    total_size = (
        DEVICE_HEADER_SIZE
        + BUFFER_HEADER_SIZE
        + buffer_size * actual_frame_size
    )
    return device_header, buffer_header, total_size


def test_control_shm_contract_accepts_matching_shape_dtype_and_layout() -> None:
    target = _target()
    device_header, buffer_header, total_size = _headers()

    contract = BasePolicyRunner._validate_control_shm_contract(
        target,
        device_header,
        buffer_header,
        buffer_length=total_size,
    )

    assert contract["action_dim"] == 6
    assert contract["target_shape"] == (2, 3)
    assert contract["target_dtype"] == np.dtype("float32")
    assert contract["payload_size"] == 6 * np.dtype("float32").itemsize


@pytest.mark.parametrize(
    ("target_shape", "target_dtype", "header_shape", "header_dtype", "message"),
    [
        ((2, 3), "float32", (6,), "float32", "shape does not match"),
        ((2, 3), "float32", (2, 3), "float64", "dtype does not match"),
    ],
)
def test_control_shm_contract_rejects_shape_or_dtype_mismatch(
    target_shape: tuple[int, ...],
    target_dtype: str,
    header_shape: tuple[int, ...],
    header_dtype: str,
    message: str,
) -> None:
    target = _target(shape=target_shape, dtype=target_dtype)
    device_header, buffer_header, total_size = _headers(
        shape=header_shape,
        dtype=header_dtype,
    )

    with pytest.raises(ValueError, match=message):
        BasePolicyRunner._validate_control_shm_contract(
            target,
            device_header,
            buffer_header,
            buffer_length=total_size,
        )


def test_control_shm_contract_rejects_frame_size_that_disagrees_with_payload() -> None:
    target = _target()
    device_header, buffer_header, total_size = _headers(frame_size=64)

    with pytest.raises(ValueError, match="frame_size does not match"):
        BasePolicyRunner._validate_control_shm_contract(
            target,
            device_header,
            buffer_header,
            buffer_length=total_size,
        )


def test_connect_robot_control_rejects_mismatched_header_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    device_header, buffer_header, total_size = _headers(shape=(6,))
    buffer = bytearray(total_size)
    buffer[:DEVICE_HEADER_SIZE] = pack_device_header(
        "Robot",
        0,
        30.0,
        "float32",
        (6,),
        device_header["frame_size"],
    )
    buffer[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = (
        pack_buffer_header(
            buffer_header["buffer_size"],
            buffer_header["current_frames_count"],
            buffer_header["write_index"],
        )
    )

    class _Channel:
        def __init__(self) -> None:
            self.buf = buffer
            self.closed = False

        def close(self) -> None:
            self.closed = True

    channel = _Channel()
    monkeypatch.setattr(
        "manager.base_policy_runner.connect_to_shared_memory",
        lambda *_args, **_kwargs: channel,
    )
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.retry_connect_secs = 0.0
    runner.connect_timeout_secs = 0.0

    assert runner._connect_robot_control(target) is None
    assert channel.closed is True


def test_policy_action_config_flattens_multidimensional_target_shape() -> None:
    runtime_io = PolicyRuntimeIO(SimpleNamespace(controls_config=[_target()]))

    assert runtime_io._build_action_configs() == [
        {
            "device_name": "Robot_0",
            "action_dim": 6,
            "shm_name": "Robot_0_control",
        }
    ]


class _EventLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log_event(self, event_type: str, **fields: Any) -> None:
        self.records.append({"event_type": event_type, **fields})


class _ActionLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log_action(self, **fields: Any) -> None:
        self.records.append(fields)


def _robot_state() -> tuple[dict[str, Any], bytearray]:
    target = _target()
    device_header, buffer_header, total_size = _headers()
    contract = BasePolicyRunner._validate_control_shm_contract(
        target,
        device_header,
        buffer_header,
        buffer_length=total_size,
    )
    buffer = bytearray(total_size)
    buffer[:DEVICE_HEADER_SIZE] = pack_device_header(
        "Robot",
        0,
        30.0,
        "float32",
        (2, 3),
        contract["frame_size"],
    )
    buffer[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = (
        pack_buffer_header(2, 0, 0)
    )
    return (
        {
            "device_info": target,
            "shm": SimpleNamespace(buf=buffer),
            "shm_name": target["shm_name"],
            "buffer_size": 2,
            **contract,
        },
        buffer,
    )


def test_action_executor_casts_flat_policy_row_to_target_dtype_and_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_state, buffer = _robot_state()
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.timestamp_logger = _EventLogger()
    runner.action_state_xyz_logger = _ActionLogger()
    event_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("manager.base_policy_runner.time.time_ns", lambda: event_ns)

    result = runner._write_action_to_robot(
        robot_state,
        np.arange(6, dtype=np.float64),
    )

    assert result == event_ns
    data_start = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + FRAME_HEADER_SIZE
    data_end = data_start + robot_state["payload_size"]
    written = np.frombuffer(buffer[data_start:data_end], dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(written, np.arange(6, dtype=np.float32).reshape(2, 3))
    assert runner.timestamp_logger.records[0]["action"].dtype == np.dtype("float32")
    assert runner.action_state_xyz_logger.records[0]["action"].shape == (2, 3)
    assert unpack_buffer_header(
        buffer[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
    ) == {"buffer_size": 2, "current_frames_count": 1, "write_index": 1, "padding": 0}


def test_action_executor_rejects_bad_policy_row_without_mutating_control_shm() -> None:
    robot_state, buffer = _robot_state()
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.timestamp_logger = _EventLogger()
    runner.action_state_xyz_logger = _ActionLogger()
    before = bytes(buffer)

    result = runner._write_action_to_robot(
        robot_state,
        np.arange(5, dtype=np.float64),
    )

    assert result is None
    assert bytes(buffer) == before
    assert not runner.timestamp_logger.records
    assert not runner.action_state_xyz_logger.records
