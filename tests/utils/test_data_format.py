"""Contracts for structured low-dimensional device data formats."""

from __future__ import annotations

import pytest

from utils.data_format import (
    DataFormatError,
    parse_data_format,
    parse_image_format,
)


def test_data_format_counts_ordered_primitive_dimensions() -> None:
    data_format = parse_data_format(
        {
            "shape": [14],
            "type": ["j1-7", "xyz", "qwqxqyqz"],
            "dtype": "float64",
        }
    )

    assert data_format.shape == (14,)
    assert data_format.dtype == "float64"
    assert [primitive.width for primitive in data_format.primitives] == [7, 3, 4]
    assert data_format.primitives[-1].quaternion_order == "wxyz"


def test_data_format_rejects_primitive_dimension_mismatch() -> None:
    with pytest.raises(DataFormatError, match="does not match"):
        parse_data_format(
            {
                "shape": [7],
                "type": ["xyz", "qwqxqyqz", "scalar"],
            }
        )


def test_data_format_requires_dtype_for_grouped_stream_contracts() -> None:
    with pytest.raises(DataFormatError, match="dtype is required"):
        parse_data_format(
            {"shape": [7], "type": ["j1-7"]},
            require_dtype=True,
            context="state_feedback.format",
        )


def test_data_format_accepts_robot_state_primitives() -> None:
    data_format = parse_data_format(
        {
            "shape": [13],
            "type": ["motor_angle", "wrench", "twist"],
            "dtype": "float64",
        }
    )

    assert data_format.dimension == 13


def test_image_format_requires_rgb_and_preserves_shape() -> None:
    image_format = parse_image_format(
        {
            "shape": [480, 640, 3],
            "type": "rgb",
            "dtype": "uint8",
        }
    )

    assert image_format.shape == (480, 640, 3)
    assert image_format.type == "rgb"
    assert image_format.dtype == "uint8"


def test_image_format_rejects_non_rgb_type() -> None:
    with pytest.raises(DataFormatError, match="must be rgb"):
        parse_image_format(
            {
                "shape": [480, 640, 3],
                "type": "bgr",
                "dtype": "uint8",
            }
        )
