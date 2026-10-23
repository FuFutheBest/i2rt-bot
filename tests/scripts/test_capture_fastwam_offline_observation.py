"""Trusted-SHM identity checks for the SuperOffline capture command."""

from __future__ import annotations

import pytest

from scripts.analysis.capture_fastwam_offline_observation import (
    _validate_trusted_capture_shm_names,
)


def test_trusted_capture_accepts_only_device_owned_shm_names() -> None:
    _validate_trusted_capture_shm_names(
        left_device="WebCameraOpenPi_1",
        right_device="WebCameraOpenPi_3",
        robot_device="RizonRobot_2",
        left_shm="WebCameraOpenPi_1_data",
        right_shm="WebCameraOpenPi_3_data",
        robot_shm="RizonRobot_2_data",
    )

    with pytest.raises(ValueError, match="--left-shm"):
        _validate_trusted_capture_shm_names(
            left_device="WebCameraOpenPi_1",
            right_device="WebCameraOpenPi_3",
            robot_device="RizonRobot_2",
            left_shm="other_camera_data",
            right_shm="WebCameraOpenPi_3_data",
            robot_shm="RizonRobot_2_data",
        )
