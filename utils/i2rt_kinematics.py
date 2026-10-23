"""Current I2RT YAM FK/IK adapter used by randomized-home sampling."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def create_i2rt_kinematics(*, site_name: str = "grasp_site") -> Tuple[Any, str]:
    """Create the vendor-supported six-joint YAM kinematics implementation.

    The pinned I2RT submodule owns the MJCF composition and model location.  Do
    not hard-code a model path here: older post-train code pointed at a removed
    ``kinematics_mj`` module and an obsolete YAM XML location.
    """
    try:
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    except ImportError as exc:
        raise RuntimeError(
            "I2RT randomized home requires the installed i2rt kinematics dependencies"
        ) from exc

    model_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
    return Kinematics(model_path, site_name), "i2rt.robots.kinematics.Kinematics"


def arm_joint_limits(kinematics: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read the six arm limits from the active vendor MuJoCo model."""
    try:
        ranges = np.asarray(kinematics._configuration.model.jnt_range, dtype=np.float64)
    except AttributeError as exc:
        raise RuntimeError("I2RT kinematics does not expose MuJoCo joint limits") from exc
    if ranges.shape[0] < 6 or ranges.shape[1] != 2:
        raise RuntimeError("I2RT kinematics model has invalid arm joint limits")
    lower = ranges[:6, 0]
    upper = ranges[:6, 1]
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("I2RT kinematics model has non-finite arm joint limits")
    return lower, upper


__all__ = ["arm_joint_limits", "create_i2rt_kinematics"]
