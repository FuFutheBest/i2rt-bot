"""Primitive-aware interpolation for low-dimensional device streams."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from utils.data_format import DataFormat, SLERP_INTERPOLATION


class LowDimInterpolationError(ValueError):
    """Raised when a low-dimensional frame cannot be safely interpolated."""


def _slerp_quaternion(
    start: np.ndarray,
    end: np.ndarray,
    ratio: float,
    quaternion_order: str,
) -> np.ndarray:
    """Interpolate one quaternion segment and preserve its declared order."""
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise LowDimInterpolationError("quaternion endpoints must be finite")
    if np.linalg.norm(start) < 1e-8 or np.linalg.norm(end) < 1e-8:
        raise LowDimInterpolationError("quaternion endpoints must be non-zero")

    if quaternion_order == "wxyz":
        start_xyzw = start[[1, 2, 3, 0]]
        end_xyzw = end[[1, 2, 3, 0]]
    elif quaternion_order == "xyzw":
        start_xyzw = start
        end_xyzw = end
    else:
        raise LowDimInterpolationError(
            f"unsupported quaternion order '{quaternion_order}'"
        )

    try:
        rotations = Rotation.from_quat([start_xyzw, end_xyzw])
        if ratio > 1.0:
            delta = (rotations[0].inv() * rotations[1]).as_rotvec()
            result_xyzw = (rotations[1] * Rotation.from_rotvec(delta * (ratio - 1.0))).as_quat()
        else:
            result_xyzw = Slerp([0.0, 1.0], rotations)([ratio]).as_quat()[0]
    except ValueError as exc:
        raise LowDimInterpolationError(f"cannot slerp quaternion: {exc}") from exc

    if quaternion_order == "wxyz":
        return result_xyzw[[3, 0, 1, 2]]
    return result_xyzw


def interpolate_low_dim_data(
    start_data: Any,
    end_data: Any,
    ratio: float,
    data_format: DataFormat,
) -> np.ndarray:
    """Interpolate a frame using its ordered primitive contract.

    Linear primitives are interpolated element-wise. Quaternion primitives use
    shortest-path SLERP and remain normalized in their declared component order.
    """
    if not 0.0 <= ratio <= 1.0:
        raise LowDimInterpolationError("interpolation ratio must be within [0, 1]")
    return _sample_low_dim_data(start_data, end_data, ratio, data_format)


def extrapolate_low_dim_data(
    start_data: Any,
    end_data: Any,
    ratio: float,
    data_format: DataFormat,
) -> np.ndarray:
    """Project linear positions and shortest-path angular velocity forward.

    The caller bounds the time horizon. Explicit velocity primitives retain
    their latest value, consistent with constant-velocity projection.
    """
    if not np.isfinite(ratio) or ratio < 1.0:
        raise LowDimInterpolationError("extrapolation ratio must be finite and >= 1")
    if not np.all(np.isfinite(start_data)) or not np.all(np.isfinite(end_data)):
        raise LowDimInterpolationError("extrapolation endpoints must be finite")
    result = _sample_low_dim_data(start_data, end_data, ratio, data_format)
    if not np.all(np.isfinite(result)):
        raise LowDimInterpolationError("extrapolated frame must be finite")
    return result


def _sample_low_dim_data(
    start_data: Any,
    end_data: Any,
    ratio: float,
    data_format: DataFormat,
) -> np.ndarray:
    start_array = np.asarray(start_data)
    end_array = np.asarray(end_data)
    if start_array.shape != end_array.shape:
        raise LowDimInterpolationError("interpolation endpoints must have the same shape")
    if start_array.size != data_format.dimension:
        raise LowDimInterpolationError(
            "frame dimension does not match its data_format.shape"
        )
    if not np.issubdtype(start_array.dtype, np.floating) or not np.issubdtype(
        end_array.dtype, np.floating
    ):
        raise LowDimInterpolationError(
            "low-dimensional interpolation requires floating-point frame data"
        )

    output_dtype = np.result_type(start_array.dtype, end_array.dtype)
    start_flat = start_array.astype(np.float64, copy=False).reshape(-1)
    end_flat = end_array.astype(np.float64, copy=False).reshape(-1)
    result = start_flat + (end_flat - start_flat) * float(ratio)

    offset = 0
    for primitive in data_format.primitives:
        stop = offset + primitive.width
        if primitive.interpolation == SLERP_INTERPOLATION:
            result[offset:stop] = _slerp_quaternion(
                start_flat[offset:stop],
                end_flat[offset:stop],
                float(ratio),
                primitive.quaternion_order or "",
            )
        elif ratio > 1.0 and (primitive.name.startswith("dq") or primitive.name == "twist"):
            result[offset:stop] = end_flat[offset:stop]
        offset = stop

    return result.reshape(start_array.shape).astype(output_dtype, copy=False)
