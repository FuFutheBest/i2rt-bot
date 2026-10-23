#!/usr/bin/env python3
"""
Cartesian XYZ safety bounding box for end-effector commands.

Provides a `SafetyBounds` helper that clamps an absolute TCP position
(in the robot base frame) to a user-defined axis-aligned cuboid. Used
by `RizonRobot._process_eef_command` to enforce a workspace limit at
inference time. Quaternion and gripper components are not touched.

Author: Zixi Ying
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.logger_config import logger


_SUPPORTED_FRAMES = {"robot_base"}


@dataclass
class SafetyBounds:
    """Axis-aligned Cartesian bounding box for TCP position clamping."""

    enabled: bool
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    frame: str = "robot_base"

    # Internal: rate-limited logging state.
    _last_log_time: float = 0.0
    _log_min_interval_s: float = 1.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "SafetyBounds":
        """Build a SafetyBounds from a raw config dict.

        Accepts None or `enabled: false` to produce a pass-through instance.
        Raises ValueError on malformed configuration.
        """
        if cfg is None:
            return cls(
                enabled=False,
                min_xyz=np.zeros(3, dtype=float),
                max_xyz=np.zeros(3, dtype=float),
            )

        if not isinstance(cfg, dict):
            raise ValueError(
                f"safety_bounds config must be a mapping, got {type(cfg).__name__}"
            )

        enabled = bool(cfg.get("enabled", True))
        frame = str(cfg.get("frame", "robot_base"))
        if frame not in _SUPPORTED_FRAMES:
            raise ValueError(
                f"safety_bounds.frame={frame!r} not supported; "
                f"expected one of {sorted(_SUPPORTED_FRAMES)}."
            )

        if not enabled:
            # Allow placeholder bounds when disabled (avoid forcing users to fill values).
            min_xyz = np.asarray(cfg.get("min_xyz", [0.0, 0.0, 0.0]), dtype=float)
            max_xyz = np.asarray(cfg.get("max_xyz", [0.0, 0.0, 0.0]), dtype=float)
            return cls(enabled=False, min_xyz=min_xyz, max_xyz=max_xyz, frame=frame)

        if "min_xyz" not in cfg or "max_xyz" not in cfg:
            raise ValueError(
                "safety_bounds requires both min_xyz and max_xyz when enabled."
            )

        min_xyz = np.asarray(cfg["min_xyz"], dtype=float).reshape(-1)
        max_xyz = np.asarray(cfg["max_xyz"], dtype=float).reshape(-1)
        if min_xyz.shape != (3,) or max_xyz.shape != (3,):
            raise ValueError(
                f"safety_bounds min_xyz/max_xyz must be length-3 vectors; "
                f"got shapes {min_xyz.shape} and {max_xyz.shape}."
            )
        if not np.all(max_xyz > min_xyz):
            raise ValueError(
                f"safety_bounds requires max_xyz > min_xyz element-wise; "
                f"got min={min_xyz.tolist()}, max={max_xyz.tolist()}."
            )

        instance = cls(
            enabled=True,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            frame=frame,
        )
        logger.info(
            f"SafetyBounds enabled (frame={frame}): "
            f"min={min_xyz.tolist()}, max={max_xyz.tolist()}"
        )
        return instance

    def clip_position(
        self, pos: np.ndarray, *, arm_name: str = ""
    ) -> Tuple[np.ndarray, bool]:
        """Clamp a TCP position to the bounding box.

        Returns the (possibly clipped) position and a flag indicating whether
        any axis was actually clamped. When disabled, returns the input
        position unchanged with `was_clipped=False`.
        """
        pos_arr = np.asarray(pos, dtype=float).reshape(-1)
        if pos_arr.shape != (3,):
            raise ValueError(
                f"SafetyBounds.clip_position expects shape (3,), got {pos_arr.shape}."
            )
        if not self.enabled:
            return pos_arr, False

        clipped = np.clip(pos_arr, self.min_xyz, self.max_xyz)
        was_clipped = bool(np.any(np.abs(clipped - pos_arr) > 1e-9))
        if was_clipped:
            self._maybe_log_clip(pos_arr, clipped, arm_name=arm_name)
        return clipped, was_clipped

    def _maybe_log_clip(
        self,
        original: np.ndarray,
        clipped: np.ndarray,
        *,
        arm_name: str,
    ) -> None:
        """Emit a rate-limited warning when clipping happens."""
        now = time.monotonic()
        if (now - self._last_log_time) < self._log_min_interval_s:
            return
        self._last_log_time = now
        diff = clipped - original
        arm_label = f"[{arm_name}] " if arm_name else ""
        logger.warning(
            f"{arm_label}safety clip applied: "
            f"requested={original.tolist()} -> clipped={clipped.tolist()} "
            f"(delta={diff.tolist()}, bounds min={self.min_xyz.tolist()} "
            f"max={self.max_xyz.tolist()})"
        )
