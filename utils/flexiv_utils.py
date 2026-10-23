#!/usr/bin/env python3
"""
Shared Flexiv RDK versioning and geometry helpers.

Author: Zixi Ying
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.logger_config import logger

SUPPORTED_RDK_VERSIONS: Tuple[Tuple[int, int, int], ...] = (
    (1, 7, 0),
    (1, 8, 0),
    (1, 9, 0),
    (1, 9, 1),
    (2, 0, 0),
)


@dataclass(frozen=True)
class MethodCandidate:
    """One callable candidate for cross-version symbol probing."""

    name: str
    min_version: Optional[Tuple[int, int, int]] = None
    max_version: Optional[Tuple[int, int, int]] = None


def parse_version(version_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse a semantic-ish RDK version string into a 3-tuple."""
    parts = version_str.split(".")
    parsed: list[int] = []
    for part in parts:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parsed.append(int(digits))
        if len(parsed) >= 3:
            break
    if len(parsed) < 2:
        return None
    while len(parsed) < 3:
        parsed.append(0)
    return parsed[0], parsed[1], parsed[2]


def format_version(version: Tuple[int, int, int]) -> str:
    """Format a normalized RDK version tuple."""
    return f"{version[0]}.{version[1]}.{version[2]}"


def normalize_supported_rdk_version(
    version: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int]]:
    """Normalize patch-level variants to supported API profiles."""
    major, minor, patch = version
    if (major, minor) == (1, 7):
        return (1, 7, 0)
    if (major, minor) == (1, 8):
        return (1, 8, 0)
    if (major, minor, patch) == (1, 9, 0):
        return (1, 9, 0)
    if (major, minor, patch) == (1, 9, 1):
        return (1, 9, 1)
    if (major, minor) == (2, 0):
        return (2, 0, 0)
    return None


def require_supported_rdk_version(version: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return a supported API profile or raise a descriptive error."""
    normalized = normalize_supported_rdk_version(version)
    if normalized is not None:
        return normalized
    supported = ", ".join(format_version(v) for v in SUPPORTED_RDK_VERSIONS)
    raise RuntimeError(
        f"Unsupported RDK version {format_version(version)}. Supported versions: {supported}"
    )


def call_versioned_method(
    obj: Any,
    version: Tuple[int, int, int],
    candidates: Sequence[MethodCandidate],
    *args: Any,
) -> Any:
    """Call the first matching method candidate by version and symbol existence."""
    for candidate in candidates:
        if candidate.min_version is not None and version < candidate.min_version:
            continue
        if candidate.max_version is not None and version > candidate.max_version:
            continue
        fn = getattr(obj, candidate.name, None)
        if callable(fn):
            return fn(*args)
    candidate_names = ", ".join(candidate.name for candidate in candidates)
    raise RuntimeError(
        f"No compatible method found for version={format_version(version)}. "
        f"Candidates: {candidate_names}"
    )


def detect_python_rdk_version(sdk: Any) -> Tuple[int, int, int]:
    """Detect the official flexivrdk Python package version."""
    version_candidates: list[str] = []
    for attr_name in ("__version__", "VERSION", "version"):
        attr_value = getattr(sdk, attr_name, None)
        if isinstance(attr_value, str) and attr_value.strip():
            version_candidates.append(attr_value.strip())
    try:
        pkg_version = metadata.version("flexivrdk")
        if pkg_version:
            version_candidates.append(pkg_version)
    except metadata.PackageNotFoundError:
        pass
    for candidate in version_candidates:
        parsed = parse_version(candidate)
        if parsed is not None:
            normalized = require_supported_rdk_version(parsed)
            logger.info("Detected flexivrdk version: {}", format_version(normalized))
            return normalized
    raise RuntimeError(
        "Cannot determine flexivrdk version from module attributes or package metadata."
    )


def detect_pybind_rdk_version(sdk: Any) -> Optional[Tuple[int, int, int]]:
    """Detect the RDK API version compiled into flexiv_rdk_ext, if exported."""
    for attr_name in ("__rdk_version__", "RDK_VERSION", "rdk_version"):
        attr_value = getattr(sdk, attr_name, None)
        if isinstance(attr_value, str) and attr_value.strip():
            parsed = parse_version(attr_value.strip())
            if parsed is not None:
                return require_supported_rdk_version(parsed)
    return None


def infer_flange_frame_from_urdf(urdf_path: str | Path) -> str:
    """Infer end-effector frame from URDF, preferring exact 'flange'."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    frame_names = [str(frame.name) for frame in model.frames]
    if "flange" in frame_names:
        return "flange"
    flange_matches = [name for name in frame_names if "flange" in name.lower()]
    if len(flange_matches) == 1:
        return flange_matches[0]
    if len(flange_matches) > 1:
        raise ValueError(f"Multiple URDF frames contain 'flange': {flange_matches}")
    raise ValueError(f"No URDF frame named or containing 'flange' found in {urdf_path}")


def flexiv_pose_wxyz_to_matrix(pose: Sequence[float]) -> List[List[float]]:
    """Convert Flexiv pose [x,y,z,qw,qx,qy,qz] to a row-major 4x4 transform."""
    pose_arr = np.asarray(pose, dtype=float).reshape(-1)
    if int(pose_arr.size) != 7:
        raise ValueError(f"Flexiv pose must have 7 values, got {int(pose_arr.size)}")
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = R.from_quat(
        [pose_arr[4], pose_arr[5], pose_arr[6], pose_arr[3]]
    ).as_matrix()
    matrix[:3, 3] = pose_arr[:3]
    return matrix.tolist()


def flexiv_pose_wxyz_to_xyzw_list(pose: Sequence[float]) -> List[float]:
    """Convert Flexiv pose ``[x, y, z, qw, qx, qy, qz]`` to ``[x, y, z, qx, qy, qz, qw]`` (scipy-style)."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    if int(p.size) != 7:
        raise ValueError(f"Flexiv pose must have 7 values, got {int(p.size)}")
    return [float(p[0]), float(p[1]), float(p[2]), float(p[4]), float(p[5]), float(p[6]), float(p[3])]


def flexiv_pose_xyzw_to_wxyz_list(pose: Sequence[float]) -> List[float]:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` to Flexiv pose ``[x, y, z, qw, qx, qy, qz]``."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    if int(p.size) != 7:
        raise ValueError(f"Pose must have 7 values, got {int(p.size)}")
    return [float(p[0]), float(p[1]), float(p[2]), float(p[6]), float(p[3]), float(p[4]), float(p[5])]
