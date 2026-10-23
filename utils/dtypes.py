"""Side-effect-free dtype definitions shared by config and SHM contracts."""

from __future__ import annotations

import numpy as np


DTYPE_MAP = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
}

SUPPORTED_DTYPE_NAMES = frozenset(DTYPE_MAP)


__all__ = ["DTYPE_MAP", "SUPPORTED_DTYPE_NAMES"]
