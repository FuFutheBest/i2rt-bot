"""Utility helpers for loading calibration transforms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.logger_config import logger


def resolve_repository_calibration_path(
    json_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve a repository-owned calibration path independently of process CWD."""
    path = Path(json_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return (root / path).resolve()


def load_transform_from_json(json_path: str) -> np.ndarray:
    """Load a 4x4 homogeneous transform from a JSON file.

    Args:
        json_path: Path pointing to a JSON file containing a 4x4 matrix.

    Returns:
        The loaded transform as a float64 numpy array.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file content cannot be parsed into a 4x4 matrix.
    """

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Transform JSON file not found: {json_path}")

    try:
        data = json.loads(path.read_text())
        transform = np.array(data, dtype=np.float64)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format in {json_path}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise ValueError(f"Error loading transform matrix from {json_path}: {exc}") from exc

    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {json_path}, got shape {transform.shape}")

    logger.info(f"Loaded transform matrix from {json_path}")
    return transform


def load_tcp_umi_transforms(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper returning both TCP→UMI and UMI→TCP transforms."""

    tcp_umi = load_transform_from_json(json_path)
    umi_tcp = np.linalg.inv(tcp_umi)
    return tcp_umi, umi_tcp
