"""Stable action and observation contracts for the RoboTwin bridge.

The bridge deliberately keeps its public wire formats small.  SuperInference
owns action scheduling; RoboTwin owns simulation state and rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

JOINT14_FIELDS = (
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_gripper",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_gripper",
)

EEF16_FIELDS = (
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "right_gripper",
)

ACTION_FIELDS = {"joint14": JOINT14_FIELDS, "eef16": EEF16_FIELDS}


def action_schema_dim(action_schema: str) -> int:
    """Return the fixed vector dimension for a configured action schema."""
    try:
        return len(ACTION_FIELDS[str(action_schema)])
    except KeyError as exc:
        supported = ", ".join(sorted(ACTION_FIELDS))
        raise ValueError(f"Unsupported RoboTwin action schema {action_schema!r}; use {supported}") from exc


def validate_action(action: Any, action_schema: str) -> np.ndarray:
    """Return one finite float64 action vector with the configured layout."""
    expected_dim = action_schema_dim(action_schema)
    vector = np.asarray(action, dtype=np.float64).reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(
            f"RoboTwin {action_schema} action must have {expected_dim} values, got {vector.size}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"RoboTwin {action_schema} action contains non-finite values")
    return vector.copy()


def validate_float64_control_dtype(dtype: Any) -> str:
    """Normalize and enforce the control dtype required by BasePolicyRunner."""
    normalized = np.dtype(dtype)
    if normalized != np.dtype(np.float64):
        raise ValueError(
            "RoboTwin control_dtype must be float64 because BasePolicyRunner writes "
            "the existing control SHM frame without dtype conversion"
        )
    return normalized.name


@dataclass(frozen=True)
class ControlSetpoint:
    """A target accepted from the control SHM, without execution-complete semantics."""

    sequence: int
    control_write_timestamp_ns: int
    action_schema: str
    action: np.ndarray


@dataclass(frozen=True)
class CameraStreamSpec:
    """One virtual camera stream published by the single RoboTwin process."""

    camera_name: str
    device_class_id: int
    data_shape: tuple[int, int, int]
    fps: float
    buffer_size: int

    @classmethod
    def from_config(cls, config: dict[str, Any], *, default_fps: float, default_buffer_size: int) -> "CameraStreamSpec":
        camera_name = str(config["camera_name"])
        data_shape = tuple(int(value) for value in config.get("data_shape", (480, 640, 3)))
        if len(data_shape) != 3 or data_shape[-1] != 3 or min(data_shape) <= 0:
            raise ValueError(
                f"camera stream {camera_name!r} needs an HxWx3 data_shape, got {data_shape!r}"
            )
        fps = float(config.get("fps", default_fps))
        if fps <= 0:
            raise ValueError(f"camera stream {camera_name!r} fps must be positive")
        return cls(
            camera_name=camera_name,
            device_class_id=int(config["device_class_id"]),
            data_shape=data_shape,
            fps=fps,
            buffer_size=int(config.get("buffer_size", default_buffer_size)),
        )


def parse_camera_streams(
    streams: Iterable[dict[str, Any]] | None,
    *,
    default_fps: float,
    default_buffer_size: int,
) -> tuple[CameraStreamSpec, ...]:
    """Parse camera stream config and reject duplicate SHM identities."""
    parsed = tuple(
        CameraStreamSpec.from_config(stream, default_fps=default_fps, default_buffer_size=default_buffer_size)
        for stream in (streams or ())
    )
    names = [spec.camera_name for spec in parsed]
    ids = [spec.device_class_id for spec in parsed]
    if len(names) != len(set(names)):
        raise ValueError("RoboTwin camera stream names must be unique")
    if len(ids) != len(set(ids)):
        raise ValueError("RoboTwin camera stream device IDs must be unique")
    return parsed
