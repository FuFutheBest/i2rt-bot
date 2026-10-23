"""ROI marker angle to physical gripper width (packaged calibration JSON).

Mirrors ``nmx_data.nmx_processor.gripper.width_geometry`` so SuperInference
converters can map detector line angles (deg) to width (m) without installing
the full ``nmx-data`` wheel.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_CALIBRATION_ID = "lab_pro_visual_angle_width_20260528_detector_space_v1"
_WIDTH_DEFINITION = "effective_fixed_minz_to_moving_minz"
_ARM_SIDES = frozenset({"left", "right"})


@dataclass(frozen=True, slots=True)
class VisualWidthEstimate:
    opening_ratio: float
    base_width_m: float
    base_width_norm: float
    corrected_width_norm: float
    width_m: float


@dataclass(frozen=True, slots=True)
class VisualWidthCalibration:
    calibration_id: str
    theta_open_max_deg: float
    fixed_xz_mm: tuple[float, float]
    moving_close_xz_mm: tuple[float, float]
    pivot_center_xz_mm: tuple[float, float]
    rotation_sign: int
    close_distance_offset_mm: float
    close_normalized_open_width_mm: float
    physical_width_min_mm: float
    physical_width_max_mm: float
    side_close_open_deg: dict[str, tuple[float, float]]
    affine_width_norm_by_side: dict[str, tuple[float, float]]


class VisualWidthCalibrationError(ValueError):
    pass


def _as_point(value: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise VisualWidthCalibrationError(f"{field_name} must be a length-2 list")
    x_coord = float(value[0])
    z_coord = float(value[1])
    if not math.isfinite(x_coord) or not math.isfinite(z_coord):
        raise VisualWidthCalibrationError(f"{field_name} must contain finite values")
    return x_coord, z_coord


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VisualWidthCalibrationError(f"{field_name} must be finite") from exc
    if not math.isfinite(number):
        raise VisualWidthCalibrationError(f"{field_name} must be finite")
    return number


def _line_delta_deg(angle_deg: float, reference_deg: float) -> float:
    return ((float(angle_deg) - float(reference_deg) + 90.0) % 180.0) - 90.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


def _validate_calibration(data: dict[str, Any]) -> VisualWidthCalibration:
    calibration_id = str(data.get("calibration_id") or "")
    if calibration_id != _DEFAULT_CALIBRATION_ID:
        raise VisualWidthCalibrationError(
            f"unsupported calibration_id: {calibration_id!r}"
        )
    if data.get("output_width_unit") != "m":
        raise VisualWidthCalibrationError("output_width_unit must be 'm'")
    if data.get("width_definition") != _WIDTH_DEFINITION:
        raise VisualWidthCalibrationError(
            f"width_definition must be {_WIDTH_DEFINITION!r}"
        )

    cad = data.get("cad_minz")
    if not isinstance(cad, dict):
        raise VisualWidthCalibrationError("cad_minz must be an object")
    theta_open_max_deg = _finite_float(
        data.get("theta_open_max_deg"), field_name="theta_open_max_deg"
    )
    rotation_sign = int(cad.get("rotation_sign"))
    if rotation_sign not in {-1, 1}:
        raise VisualWidthCalibrationError("rotation_sign must be -1 or 1")

    endpoints = data.get("roi_angle_endpoints_deg_mod180")
    if not isinstance(endpoints, dict):
        raise VisualWidthCalibrationError("roi_angle_endpoints_deg_mod180 is required")
    side_close_open_deg: dict[str, tuple[float, float]] = {}
    for side in sorted(_ARM_SIDES):
        side_data = endpoints.get(side)
        if not isinstance(side_data, dict):
            raise VisualWidthCalibrationError(f"missing ROI endpoints for {side}")
        close_deg = _finite_float(side_data.get("close"), field_name=f"{side}.close")
        open_deg = _finite_float(side_data.get("open"), field_name=f"{side}.open")
        side_close_open_deg[side] = (close_deg % 180.0, open_deg % 180.0)

    affine_data = data.get("affine_width_norm_correction")
    if not isinstance(affine_data, dict):
        raise VisualWidthCalibrationError("affine_width_norm_correction is required")
    sides_data = affine_data.get("sides")
    if not isinstance(sides_data, dict):
        raise VisualWidthCalibrationError(
            "affine_width_norm_correction.sides is required"
        )
    affine_width_norm_by_side: dict[str, tuple[float, float]] = {}
    for side in sorted(_ARM_SIDES):
        side_data = sides_data.get(side)
        if not isinstance(side_data, dict):
            raise VisualWidthCalibrationError(
                f"missing affine width correction for {side}"
            )
        slope = _finite_float(side_data.get("slope"), field_name=f"{side}.slope")
        intercept = _finite_float(
            side_data.get("intercept"), field_name=f"{side}.intercept"
        )
        if slope <= 0.0:
            raise VisualWidthCalibrationError(f"{side}.slope must be positive")
        affine_width_norm_by_side[side] = (slope, intercept)

    return VisualWidthCalibration(
        calibration_id=calibration_id,
        theta_open_max_deg=theta_open_max_deg,
        fixed_xz_mm=_as_point(cad.get("fixed_xz_mm"), field_name="fixed_xz_mm"),
        moving_close_xz_mm=_as_point(
            cad.get("moving_close_xz_mm"), field_name="moving_close_xz_mm"
        ),
        pivot_center_xz_mm=_as_point(
            cad.get("pivot_center_xz_mm"), field_name="pivot_center_xz_mm"
        ),
        rotation_sign=rotation_sign,
        close_distance_offset_mm=_finite_float(
            cad.get("close_distance_offset_mm"), field_name="close_distance_offset_mm"
        ),
        close_normalized_open_width_mm=_finite_float(
            cad.get("close_normalized_open_width_mm"),
            field_name="close_normalized_open_width_mm",
        ),
        physical_width_min_mm=_finite_float(
            cad.get("physical_width_min_mm"), field_name="physical_width_min_mm"
        ),
        physical_width_max_mm=_finite_float(
            cad.get("physical_width_max_mm"), field_name="physical_width_max_mm"
        ),
        side_close_open_deg=side_close_open_deg,
        affine_width_norm_by_side=affine_width_norm_by_side,
    )


def _load_calibration_json(calibration_path: Path) -> VisualWidthCalibration:
    with calibration_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise VisualWidthCalibrationError("visual width calibration must be an object")
    return _validate_calibration(data)


@lru_cache(maxsize=4)
def _cached_calibration(calibration_path_str: str) -> VisualWidthCalibration:
    return _load_calibration_json(Path(calibration_path_str))


def load_visual_width_calibration(calibration_path: Path) -> VisualWidthCalibration:
    """Load and validate calibration from ``visual_angle_to_width.json``."""
    path = calibration_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"gripper width calibration not found: {path}")
    return _cached_calibration(str(path))


def _rotate_point(
    point: tuple[float, float],
    center: tuple[float, float],
    angle_rad: float,
) -> tuple[float, float]:
    px, pz = point
    cx, cz = center
    dx = px - cx
    dz = pz - cz
    cos_v = math.cos(angle_rad)
    sin_v = math.sin(angle_rad)
    return cx + cos_v * dx - sin_v * dz, cz + sin_v * dx + cos_v * dz


def estimate_visual_width_from_angle_deg(
    side: str,
    angle_deg: float,
    calibration: VisualWidthCalibration,
) -> VisualWidthEstimate:
    """Map ROI marker line angle (deg) to physical gripper width (m)."""
    if side not in _ARM_SIDES:
        raise ValueError(f"unsupported gripper side: {side!r}")
    if not math.isfinite(float(angle_deg)):
        raise ValueError("angle_deg must be finite")

    close_deg, open_deg = calibration.side_close_open_deg[side]
    denominator = _line_delta_deg(open_deg, close_deg)
    ratio = _clamp(
        _line_delta_deg(float(angle_deg) % 180.0, close_deg) / denominator, 0.0, 1.0
    )

    cad_angle_rad = math.radians(ratio * calibration.theta_open_max_deg)
    moving_xz = _rotate_point(
        calibration.moving_close_xz_mm,
        calibration.pivot_center_xz_mm,
        calibration.rotation_sign * cad_angle_rad,
    )
    raw_distance_mm = math.dist(moving_xz, calibration.fixed_xz_mm)
    base_width_m = max(
        0.0, raw_distance_mm - calibration.close_distance_offset_mm
    ) / 1000.0

    base_max_width_m = calibration.close_normalized_open_width_mm / 1000.0
    physical_min_width_m = calibration.physical_width_min_mm / 1000.0
    physical_max_width_m = calibration.physical_width_max_mm / 1000.0
    base_width_norm = _clamp(base_width_m / base_max_width_m, 0.0, 1.0)

    slope, intercept = calibration.affine_width_norm_by_side[side]
    corrected_width_norm = _clamp(
        slope * base_width_norm + intercept, 0.0, 1.0
    )
    width_m = physical_min_width_m + corrected_width_norm * (
        physical_max_width_m - physical_min_width_m
    )
    return VisualWidthEstimate(
        opening_ratio=ratio,
        base_width_m=base_width_m,
        base_width_norm=base_width_norm,
        corrected_width_norm=corrected_width_norm,
        width_m=width_m,
    )

