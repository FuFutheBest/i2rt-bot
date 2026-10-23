"""Tests for explicit runtime data contracts.

Author: OpenAI Assistant
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.contracts import (
    ActionChunk,
    ActionTargetSpec,
    ControlCommand,
    DeviceFrame,
    DeviceSpec,
    ObservationBatch,
    SummaryFrame,
    make_device_key,
    normalize_shape,
    validate_action_chunk_against_target,
    validate_control_command_against_target,
)
from utils.errors import PolicyRuntimeError, SHMContractError


@pytest.mark.unit
def test_device_spec_normalizes_and_exposes_device_key() -> None:
    """DeviceSpec should normalize metadata used in SHM headers."""
    spec = DeviceSpec(
        name="BaseDevice",
        device_class_id=1,
        fps=30,
        dtype="float64",
        shape=(2, 3),
        frame_size=56,
    )

    assert spec.device_key == "BaseDevice_1"
    assert spec.fps == 30.0
    assert spec.shape == (2, 3)


@pytest.mark.unit
def test_device_frame_and_summary_frame_validate_keys() -> None:
    """SummaryFrame mapping keys should match embedded DeviceFrame keys."""
    frame = DeviceFrame(
        device_key="BaseDevice_0",
        timestamp_ns=100,
        data=np.array([1.0], dtype=np.float64),
    )
    summary = SummaryFrame(timestamp_ns=120, frames={"BaseDevice_0": frame})

    assert summary.frames["BaseDevice_0"].is_stale(reference_timestamp_ns=200, max_age_ns=50)
    with pytest.raises(SHMContractError):
        SummaryFrame(timestamp_ns=120, frames={"Other_0": frame})


@pytest.mark.unit
def test_observation_batch_requires_numpy_values() -> None:
    """ObservationBatch should reject non-array observations."""
    ObservationBatch(
        timestamp_ns=1,
        observations={"Robot_0": np.array([1.0], dtype=np.float64)},
    )

    with pytest.raises(SHMContractError):
        ObservationBatch(timestamp_ns=1, observations={"Robot_0": [1.0]})


@pytest.mark.unit
def test_action_chunk_validation_against_target() -> None:
    """Policy output shape mismatches should fail before SHM writes."""
    target = ActionTargetSpec(
        device_name="RizonRobot_1",
        shm_name="RizonRobot_1_control",
        shape=(8,),
        dtype="float64",
    )
    chunk = ActionChunk(
        target="RizonRobot_1_control",
        timestamp_ns=10,
        action_index=0,
        data=np.zeros((4, 8), dtype=np.float64),
    )

    validate_action_chunk_against_target(chunk, target)

    invalid_chunk = ActionChunk(
        target="RizonRobot_1_control",
        timestamp_ns=10,
        action_index=0,
        data=np.zeros((4, 7), dtype=np.float64),
    )
    with pytest.raises(PolicyRuntimeError):
        validate_action_chunk_against_target(invalid_chunk, target)


@pytest.mark.unit
def test_control_command_validation_against_target() -> None:
    """Per-step control commands should match the target command shape exactly."""
    target = ActionTargetSpec(
        device_name="RizonRobot_1",
        shm_name="RizonRobot_1_control",
        shape=(8,),
    )
    command = ControlCommand(
        target="RizonRobot_1_control",
        timestamp_ns=10,
        data=np.zeros((8,), dtype=np.float64),
    )

    validate_control_command_against_target(command, target)

    with pytest.raises(SHMContractError):
        validate_control_command_against_target(
            ControlCommand(target="RizonRobot_1_control", timestamp_ns=10, data=np.zeros((1, 8))),
            target,
        )


@pytest.mark.unit
def test_contract_helpers_reject_invalid_values() -> None:
    """Helper validators should fail on invalid public contract values."""
    assert make_device_key("Camera", 2) == "Camera_2"
    assert normalize_shape([1, 2]) == (1, 2)

    with pytest.raises(SHMContractError):
        make_device_key("", 0)
    with pytest.raises(SHMContractError):
        normalize_shape([1, 0])
    with pytest.raises(SHMContractError):
        DeviceSpec(name="Bad", device_class_id=0, fps=0, dtype="float64", shape=(1,), frame_size=1)
    with pytest.raises(SHMContractError):
        ActionTargetSpec(device_name="Robot_0", shm_name="Robot_0_control", shape=(1,), dtype="bad")
