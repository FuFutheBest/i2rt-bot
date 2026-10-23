#!/usr/bin/env python3
"""
nmxrdk adapter layer for the X1 robot device integration.

This module centralizes access to the X1 (TJ) robot SDK ``nmxrdk`` / ``nmxrt``
behind thin, typed handles so that the rest of SuperInference never touches the
raw vendor objects directly. It mirrors the design of ``flexiv_adapter.py`` but
targets the nmxrdk dual-arm + head + gripper API used by the existing
the incubator X1 teleoperation scripts.

Quaternion convention note: nmxrdk uses (w, x, y, z) for TCP poses, while scipy
uses (x, y, z, w). All handles in this module expose poses in nmxrdk-native
``[x, y, z, qw, qx, qy, qz]`` (wxyz) order; callers convert to/from scipy xyzw
at the boundary.

Author: Zixi Ying
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from utils.logger_config import logger


def _import_nmx_sdk() -> Tuple[Any, Any, Any]:
    """Import nmxrdk / nmxrt and the RobotSide enum.

    Kept lazy (called at construction time, not import time) so this module can
    still be imported on a host without the SDK (e.g. for syntax checks); the
    SDK is only present inside the robot_test docker container.
    """
    try:
        import nmxrdk  # type: ignore[import-not-found]
        import nmxrt  # type: ignore[import-not-found]
        from nmxrdk.common.robot_side import RobotSide  # type: ignore[import-not-found]

        return nmxrdk, nmxrt, RobotSide
    except ImportError as exc:
        raise ImportError(
            "nmxrdk/nmxrt is not importable. The X1 robot SDK is only available "
            "inside the robot_test docker container."
        ) from exc


class NmxArmHandle:
    """Single-arm wrapper decoupling call sites from the raw nmxrdk arm object."""

    def __init__(self, arm: Any, rt: Any, label: str) -> None:
        self._arm = arm
        self._rt = rt
        self._label = label

    @property
    def raw(self) -> Any:
        return self._arm

    def _ensure_ok(self, result: Any, action: str) -> Any:
        """Raise if the nmxrdk call returned an ``nmxrt.Error``."""
        if isinstance(result, self._rt.Error):
            raise RuntimeError(f"[{self._label}] {action} failed: {result.message}")
        return result

    def enable(self) -> Any:
        return self._arm.enable()

    def get_tcp_pose7(self) -> np.ndarray:
        """Return current TCP pose ``[x, y, z, qw, qx, qy, qz]`` (wxyz)."""
        state = self._ensure_ok(self._arm.get_tcp_state(), "get_tcp_state")
        p = state.pose
        return np.array(
            [p.pos.x, p.pos.y, p.pos.z, p.quat.w, p.quat.x, p.quat.y, p.quat.z],
            dtype=np.float64,
        )

    def get_joint_pos(self) -> np.ndarray:
        """Return current joint positions (rad)."""
        state = self._ensure_ok(self._arm.get_joint_state(), "get_joint_state")
        return np.array([j.pos for j in state.joints], dtype=np.float64)

    def move_line(
        self,
        pose7: Sequence[float],
        velocity: float,
        blocking: bool = False,
    ) -> Any:
        """Cartesian linear move to ``pose7`` ([x,y,z,qw,qx,qy,qz], wxyz)."""
        target = [float(v) for v in np.asarray(pose7, dtype=float).reshape(-1)]
        if len(target) != 7:
            raise ValueError(f"move_line pose7 must have 7 values, got {len(target)}")
        return self._arm.move_line(target, float(velocity), bool(blocking))

    def move_joint(
        self,
        positions: Sequence[float],
        velocity_ratio: float,
        acceleration_ratio: float,
        blocking: bool = True,
        timeout_ms: int = 15000,
    ) -> Any:
        """Joint-space move (used for homing). ``velocity/acceleration_ratio`` are %."""
        target = [float(v) for v in np.asarray(positions, dtype=float).reshape(-1)]
        vel = [float(velocity_ratio)] * len(target)
        acc = [float(acceleration_ratio)] * len(target)
        return self._arm.move_joint(target, vel, acc, bool(blocking), int(timeout_ms), {})


class NmxGripperHandle:
    """Gripper wrapper decoupling call sites from the raw nmxrdk gripper object."""

    def __init__(self, gripper: Any, rt: Any, label: str) -> None:
        self._gripper = gripper
        self._rt = rt
        self._label = label

    @property
    def raw(self) -> Any:
        return self._gripper

    def set_width(self, width: float) -> Any:
        """Command gripper opening width in meters."""
        return self._gripper.set_width(float(width))


class NmxHeadHandle:
    """2-DoF head wrapper (joint[0]=pitch, joint[1]=yaw)."""

    def __init__(self, head: Any, rt: Any) -> None:
        self._head = head
        self._rt = rt

    @property
    def raw(self) -> Any:
        return self._head

    def _ensure_ok(self, result: Any, action: str) -> Any:
        if isinstance(result, self._rt.Error):
            raise RuntimeError(f"[head] {action} failed: {result.message}")
        return result

    def enable(self) -> Any:
        return self._head.enable()

    def get_joint_pos(self) -> np.ndarray:
        state = self._ensure_ok(self._head.get_joint_state(), "get_joint_state")
        return np.array([j.pos for j in state.joints], dtype=np.float64)

    def move_joint(
        self,
        positions: Sequence[float],
        velocity: Sequence[float],
        acceleration: Sequence[float],
        blocking: bool = False,
        timeout_ms: int = 1000,
    ) -> Any:
        target = [float(v) for v in np.asarray(positions, dtype=float).reshape(-1)]
        vel = [float(v) for v in velocity]
        acc = [float(v) for v in acceleration]
        return self._head.move_joint(target, vel, acc, bool(blocking), int(timeout_ms), {})

    def stop(self) -> Any:
        return self._head.stop()

    def disable(self) -> Any:
        return self._head.disable()


class NmxCameraHandle:
    """Camera wrapper around an nmxrdk camera; returns decoded BGR frames."""

    def __init__(self, camera: Any, rt: Any, cam_id: str) -> None:
        self._camera = camera
        self._rt = rt
        self._cam_id = cam_id

    @property
    def raw(self) -> Any:
        return self._camera

    def get_bgr(self) -> Optional[np.ndarray]:
        """Return the latest decoded color frame as HxWx3 BGR uint8, or None.

        nmxrdk's ``get_rgbd(decode=True)`` returns ``(rgb_frame, depth_frame)``
        where ``rgb_frame.get("data")`` is the decoded image (BGR, OpenCV
        convention). Any SDK error / missing frame yields None so callers can
        degrade gracefully instead of crashing.
        """
        try:
            result = self._camera.get_rgbd(decode=True)
        except Exception:
            return None
        if isinstance(result, self._rt.Error):
            return None
        rgb_frame = result[0] if isinstance(result, (tuple, list)) else result
        if isinstance(rgb_frame, self._rt.Error) or rgb_frame is None:
            return None
        data = rgb_frame.get("data") if hasattr(rgb_frame, "get") else None
        if data is None:
            return None
        return np.asarray(data)


class NmxAdapter:
    """Thin adapter around the nmxrdk RobotClient for the X1 dual-arm robot.

    Examples:
        adapter = NmxAdapter("1e783be2dd870ef5")
        left = adapter.arm("left")
        pose = left.get_tcp_pose7()
        left.move_line(pose, velocity=0.1)
    """

    def __init__(self, robot_id: str) -> None:
        self._sdk, self._rt, self._RobotSide = _import_nmx_sdk()
        self._robot_id = str(robot_id)
        self._client = self._sdk.RobotClient(self._robot_id)
        logger.info(f"NmxAdapter connected to RobotClient(robot_id={self._robot_id})")

    @property
    def raw(self) -> Any:
        return self._client

    def _side_enum(self, side: str) -> Any:
        normalized = str(side).strip().lower()
        if normalized == "left":
            return self._RobotSide.LEFT
        if normalized == "right":
            return self._RobotSide.RIGHT
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def arm(self, side: str) -> NmxArmHandle:
        return NmxArmHandle(self._client.arm(self._side_enum(side)), self._rt, f"{side} arm")

    def gripper(self, side: str) -> NmxGripperHandle:
        return NmxGripperHandle(
            self._client.gripper(self._side_enum(side)), self._rt, f"{side} gripper"
        )

    def head(self) -> NmxHeadHandle:
        return NmxHeadHandle(self._client.head(), self._rt)

    def camera(self, cam_id: str) -> NmxCameraHandle:
        """Return a camera handle. Raises if the camera id is unavailable."""
        return NmxCameraHandle(self._client.camera(cam_id), self._rt, cam_id)

    def is_error(self, result: Any) -> bool:
        """Return True if ``result`` is an ``nmxrt.Error``."""
        return isinstance(result, self._rt.Error)
