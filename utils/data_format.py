"""Contracts for low-dimensional stream interpolation metadata."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from utils.contracts import normalize_shape, validate_dtype_name
from utils.errors import SHMContractError


LINEAR_INTERPOLATION = "linear"
SLERP_INTERPOLATION = "slerp"


class DataFormatError(ValueError):
    """Raised when a low-dimensional data-format contract is invalid."""


@dataclass(frozen=True)
class DataPrimitive:
    """One ordered low-dimensional segment and its interpolation rule."""

    name: str
    width: int
    interpolation: str
    quaternion_order: Optional[str] = None


@dataclass(frozen=True)
class DataFormat:
    """Validated shape and ordered primitive layout for a low-dimensional stream."""

    shape: tuple[int, ...]
    primitives: tuple[DataPrimitive, ...]
    dtype: Optional[str] = None

    @property
    def dimension(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class ImageFormat:
    """Validated image layout published by a camera device."""

    shape: tuple[int, ...]
    type: str
    dtype: str


_JOINT_PRIMITIVE = re.compile(r"^(?:j|dq)([1-9][0-9]*)-([1-9][0-9]*)$")
_LINEAR_PRIMITIVE = re.compile(r"^linear:([1-9][0-9]*)$")


def _parse_primitive(value: Any) -> DataPrimitive:
    if not isinstance(value, str) or not value.strip():
        raise DataFormatError("data_format.type entries must be non-empty strings")

    name = value.strip().casefold()
    fixed_primitives = {
        "xyz": DataPrimitive("xyz", 3, LINEAR_INTERPOLATION),
        "scalar": DataPrimitive("scalar", 1, LINEAR_INTERPOLATION),
        "gripper": DataPrimitive("gripper", 1, LINEAR_INTERPOLATION),
        "motor_angle": DataPrimitive("motor_angle", 1, LINEAR_INTERPOLATION),
        "wrench": DataPrimitive("wrench", 6, LINEAR_INTERPOLATION),
        "twist": DataPrimitive("twist", 6, LINEAR_INTERPOLATION),
        "qwqxqyqz": DataPrimitive(
            "qwqxqyqz", 4, SLERP_INTERPOLATION, quaternion_order="wxyz"
        ),
        "qxqyqzqw": DataPrimitive(
            "qxqyqzqw", 4, SLERP_INTERPOLATION, quaternion_order="xyzw"
        ),
    }
    if name in fixed_primitives:
        return fixed_primitives[name]

    joint_match = _JOINT_PRIMITIVE.fullmatch(name)
    if joint_match is not None:
        first, last = (int(group) for group in joint_match.groups())
        if last < first:
            raise DataFormatError(
                f"joint primitive '{value}' must have an increasing range"
            )
        return DataPrimitive(name, last - first + 1, LINEAR_INTERPOLATION)

    linear_match = _LINEAR_PRIMITIVE.fullmatch(name)
    if linear_match is not None:
        return DataPrimitive(name, int(linear_match.group(1)), LINEAR_INTERPOLATION)

    raise DataFormatError(
        "unsupported data_format primitive "
        f"'{value}'; supported primitives are xyz, qwqxqyqz, qxqyqzqw, "
        "scalar, gripper, motor_angle, wrench, twist, jN-M, dqN-M, and linear:N"
    )


def parse_data_format(
    value: Any,
    *,
    require_dtype: bool = False,
    context: str = "data_format",
) -> DataFormat:
    """Validate and normalize one low-dimensional stream format mapping."""
    if not isinstance(value, Mapping):
        raise DataFormatError(f"{context} must be a mapping with shape, type, and dtype")

    unknown = set(value) - {"shape", "type", "dtype"}
    if unknown:
        raise DataFormatError(
            f"{context} contains unknown fields: {', '.join(sorted(unknown))}"
        )

    if "shape" not in value:
        raise DataFormatError(f"{context}.shape is required")
    if "type" not in value:
        raise DataFormatError(f"{context}.type is required")
    if require_dtype and "dtype" not in value:
        raise DataFormatError(f"{context}.dtype is required")

    raw_shape = value["shape"]
    if isinstance(raw_shape, (str, bytes)) or not isinstance(raw_shape, Sequence):
        raise DataFormatError(f"{context}.shape must be a sequence")
    try:
        shape = normalize_shape(raw_shape)
    except SHMContractError as exc:
        raise DataFormatError(str(exc)) from exc

    raw_types = value["type"]
    if isinstance(raw_types, (str, bytes)) or not isinstance(raw_types, Sequence):
        raise DataFormatError(f"{context}.type must be a sequence")
    primitives = tuple(_parse_primitive(item) for item in raw_types)
    if not primitives:
        raise DataFormatError(f"{context}.type must contain at least one primitive")

    declared_dimension = math.prod(shape)
    primitive_dimension = sum(primitive.width for primitive in primitives)
    if primitive_dimension != declared_dimension:
        raise DataFormatError(
            f"{context} primitive width "
            f"{primitive_dimension} does not match shape dimension {declared_dimension}"
        )

    dtype: Optional[str] = None
    if "dtype" in value:
        try:
            dtype = validate_dtype_name(str(value["dtype"]))
        except SHMContractError as exc:
            raise DataFormatError(f"{context}.dtype: {exc}") from exc

    return DataFormat(shape=shape, primitives=primitives, dtype=dtype)


def parse_image_format(value: Any, *, context: str = "state_feedback.format") -> ImageFormat:
    """Validate the common RGB camera stream contract."""
    if not isinstance(value, Mapping):
        raise DataFormatError(f"{context} must be a mapping with shape, type, and dtype")

    unknown = set(value) - {"shape", "type", "dtype"}
    if unknown:
        raise DataFormatError(
            f"{context} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    missing = {"shape", "type", "dtype"} - set(value)
    if missing:
        raise DataFormatError(
            f"{context} requires fields: shape, dtype, type; missing "
            + ", ".join(sorted(missing))
        )

    raw_shape = value["shape"]
    if isinstance(raw_shape, (str, bytes)) or not isinstance(raw_shape, Sequence):
        raise DataFormatError(f"{context}.shape must be a sequence")
    try:
        shape = normalize_shape(raw_shape)
    except SHMContractError as exc:
        raise DataFormatError(str(exc)) from exc
    if len(shape) != 3 or shape[2] != 3:
        raise DataFormatError(
            f"{context}.shape must be [height, width, 3] for an RGB camera"
        )

    image_type = value["type"]
    if not isinstance(image_type, str) or image_type.strip().casefold() != "rgb":
        raise DataFormatError(f"{context}.type must be rgb for a camera")
    try:
        dtype = validate_dtype_name(str(value["dtype"]))
    except SHMContractError as exc:
        raise DataFormatError(f"{context}.dtype: {exc}") from exc

    return ImageFormat(shape=shape, type="rgb", dtype=dtype)
