#!/usr/bin/env python3
"""
Typed data contracts shared across SuperInference runtime boundaries.

Author: OpenAI Assistant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from utils.dtypes import SUPPORTED_DTYPE_NAMES
from utils.errors import PolicyRuntimeError, SHMContractError
QUATERNION_FORMAT = "xyzw"


def make_device_key(device_name: str, device_class_id: int) -> str:
    """Return the canonical device key used across SHM and config contracts."""
    if not device_name:
        raise SHMContractError("device_name must be non-empty")
    if int(device_class_id) < 0:
        raise SHMContractError("device_class_id must be non-negative", context=device_name)
    return f"{device_name}_{int(device_class_id)}"


def normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    """Normalize a shape-like sequence into a tuple of positive integers."""
    normalized = tuple(int(dim) for dim in shape)
    if not normalized:
        raise SHMContractError("shape must not be empty")
    if any(dim <= 0 for dim in normalized):
        raise SHMContractError(f"shape dimensions must be positive: {normalized}")
    return normalized


def validate_dtype_name(dtype: str) -> str:
    """Validate and normalize a supported SHM dtype name."""
    dtype_name = str(dtype)
    if dtype_name not in SUPPORTED_DTYPE_NAMES:
        raise SHMContractError(f"unsupported dtype '{dtype_name}'")
    return dtype_name


def validate_timestamp_ns(timestamp_ns: int, *, field_name: str = "timestamp_ns") -> int:
    """Validate timestamp fields that cross process boundaries."""
    timestamp = int(timestamp_ns)
    if timestamp < 0:
        raise SHMContractError(f"{field_name} must be non-negative")
    return timestamp


@dataclass(frozen=True)
class DeviceSpec:
    """Static metadata for one device SHM stream."""

    name: str
    device_class_id: int
    fps: float
    dtype: str
    shape: Tuple[int, ...]
    frame_size: int
    hardware_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_class_id", int(self.device_class_id))
        if self.device_class_id < 0:
            raise SHMContractError("device_class_id must be non-negative", context=self.name)
        if not self.name:
            raise SHMContractError("name must be non-empty")
        if float(self.fps) <= 0.0:
            raise SHMContractError("fps must be positive", context=self.device_key)
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "dtype", validate_dtype_name(self.dtype))
        object.__setattr__(self, "shape", normalize_shape(self.shape))
        object.__setattr__(self, "frame_size", int(self.frame_size))
        if self.frame_size <= 0:
            raise SHMContractError("frame_size must be positive", context=self.device_key)
        object.__setattr__(self, "hardware_latency_ms", float(self.hardware_latency_ms))

    @property
    def device_key(self) -> str:
        """Return the canonical key, for example RizonRobot_1."""
        return make_device_key(self.name, self.device_class_id)


@dataclass(frozen=True)
class DeviceFrame:
    """One timestamped frame emitted by a device."""

    device_key: str
    timestamp_ns: int
    data: np.ndarray
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.device_key:
            raise SHMContractError("device_key must be non-empty")
        object.__setattr__(self, "timestamp_ns", validate_timestamp_ns(self.timestamp_ns))
        if not isinstance(self.data, np.ndarray):
            raise SHMContractError("DeviceFrame.data must be a numpy array", context=self.device_key)
        object.__setattr__(self, "valid", bool(self.valid))

    def is_stale(self, reference_timestamp_ns: int, max_age_ns: int) -> bool:
        """Return True when this frame is older than the allowed age."""
        reference = validate_timestamp_ns(reference_timestamp_ns, field_name="reference_timestamp_ns")
        max_age = int(max_age_ns)
        if max_age < 0:
            raise SHMContractError("max_age_ns must be non-negative", context=self.device_key)
        return reference - self.timestamp_ns > max_age


@dataclass(frozen=True)
class SummaryFrame:
    """Synchronized or latest-frame summary emitted by the device manager."""

    timestamp_ns: int
    frames: Mapping[str, DeviceFrame]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", validate_timestamp_ns(self.timestamp_ns))
        if not self.frames:
            raise SHMContractError("SummaryFrame.frames must not be empty")
        for key, frame in self.frames.items():
            if key != frame.device_key:
                raise SHMContractError(
                    f"frame key mismatch: mapping key '{key}' != frame key '{frame.device_key}'"
                )


@dataclass(frozen=True)
class ObservationBatch:
    """Policy-ready observations built from a summary frame."""

    timestamp_ns: int
    observations: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", validate_timestamp_ns(self.timestamp_ns))
        if not self.observations:
            raise SHMContractError("ObservationBatch.observations must not be empty")
        for key, value in self.observations.items():
            if not key:
                raise SHMContractError("observation key must be non-empty")
            if not isinstance(value, np.ndarray):
                raise SHMContractError(f"observation '{key}' must be a numpy array")


@dataclass(frozen=True)
class ActionTargetSpec:
    """Static metadata for one policy output/control target."""

    device_name: str
    shm_name: str
    shape: Tuple[int, ...]
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if not self.device_name:
            raise SHMContractError("device_name must be non-empty")
        if not self.shm_name:
            raise SHMContractError("shm_name must be non-empty", context=self.device_name)
        object.__setattr__(self, "shape", normalize_shape(self.shape))
        object.__setattr__(self, "dtype", validate_dtype_name(self.dtype))

    @property
    def action_dim(self) -> int:
        """Return flattened per-step action dimension."""
        return int(np.prod(self.shape))


@dataclass(frozen=True)
class ActionChunk:
    """Chunked policy output for one control target."""

    target: str
    timestamp_ns: int
    action_index: int
    data: np.ndarray

    def __post_init__(self) -> None:
        if not self.target:
            raise PolicyRuntimeError("ActionChunk.target must be non-empty")
        object.__setattr__(self, "timestamp_ns", validate_timestamp_ns(self.timestamp_ns))
        object.__setattr__(self, "action_index", int(self.action_index))
        if self.action_index < 0:
            raise PolicyRuntimeError("action_index must be non-negative", context=self.target)
        if not isinstance(self.data, np.ndarray):
            raise PolicyRuntimeError("ActionChunk.data must be a numpy array", context=self.target)
        if self.data.ndim != 2:
            raise PolicyRuntimeError(
                f"ActionChunk.data must be 2D (chunk_length, action_dim), got shape {self.data.shape}",
                context=self.target,
            )

    @property
    def chunk_length(self) -> int:
        """Return number of future action steps."""
        return int(self.data.shape[0])

    @property
    def action_dim(self) -> int:
        """Return flattened per-step action dimension."""
        return int(self.data.shape[1])


@dataclass(frozen=True)
class ControlCommand:
    """One per-step command destined for a robot control SHM."""

    target: str
    timestamp_ns: int
    data: np.ndarray

    def __post_init__(self) -> None:
        if not self.target:
            raise SHMContractError("ControlCommand.target must be non-empty")
        object.__setattr__(self, "timestamp_ns", validate_timestamp_ns(self.timestamp_ns))
        if not isinstance(self.data, np.ndarray):
            raise SHMContractError("ControlCommand.data must be a numpy array", context=self.target)


def validate_action_chunk_against_target(action_chunk: ActionChunk, target_spec: ActionTargetSpec) -> None:
    """Validate a policy action chunk before writing it to policy SHM."""
    if action_chunk.target != target_spec.device_name and action_chunk.target != target_spec.shm_name:
        raise PolicyRuntimeError(
            f"action target '{action_chunk.target}' does not match '{target_spec.device_name}' "
            f"or '{target_spec.shm_name}'"
        )
    if action_chunk.data.shape[1:] != target_spec.shape:
        raise PolicyRuntimeError(
            f"action chunk shape {action_chunk.data.shape[1:]} does not match target shape {target_spec.shape}",
            context=action_chunk.target,
        )


def validate_control_command_against_target(command: ControlCommand, target_spec: ActionTargetSpec) -> None:
    """Validate one per-step control command before writing it to a control SHM."""
    if command.target != target_spec.device_name and command.target != target_spec.shm_name:
        raise SHMContractError(
            f"control target '{command.target}' does not match '{target_spec.device_name}' "
            f"or '{target_spec.shm_name}'"
        )
    if command.data.shape != target_spec.shape:
        raise SHMContractError(
            f"control command shape {command.data.shape} does not match target shape {target_spec.shape}",
            context=command.target,
        )
