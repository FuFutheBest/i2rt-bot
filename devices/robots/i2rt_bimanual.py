"""DeviceStarter-owned I2RT bimanual teleoperation device."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from devices.config import StateFeedbackConfig
from devices.contracts import DeviceKind
from devices.robots.i2rt_teleop.config import (
    REPO_ROOT,
    load_teleoperation_config,
)
from devices.robots.i2rt_teleop.coordinator import run_teleoperation


LOGGER = logging.getLogger(__name__)


class I2RTBimanualTeleop:
    """Own the complete leader/follower rig and publish both robot streams."""

    device_kind = DeviceKind.LOW_DIM_R

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        device_id: int | None = None,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        follower_state_feedback: StateFeedbackConfig | Mapping[str, Any],
        teleoperation_config_path: str | Path,
        device_name: str = "I2RTRobot",
        follower_device_class_id: int = 1,
        follower_device_id: int | None = None,
    ) -> None:
        if device_id is not None:
            device_class_id = device_id
        if follower_device_id is not None:
            follower_device_class_id = follower_device_id
        if int(device_class_id) == int(follower_device_class_id):
            raise ValueError("leader and follower device class IDs must be different")
        self.device_class_id = int(device_class_id)
        self.follower_device_class_id = int(follower_device_class_id)
        self.device_name = str(device_name).strip()
        if not self.device_name:
            raise ValueError("device_name must be non-empty")

        path = Path(teleoperation_config_path).expanduser()
        self.teleoperation_config_path = (
            path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
        )
        self.teleoperation_config = load_teleoperation_config(
            self.teleoperation_config_path
        )
        self.state_feedback = self._state_contract(state_feedback, "leader")
        self.follower_state_feedback = self._state_contract(
            follower_state_feedback, "follower"
        )

    @staticmethod
    def _state_contract(
        value: StateFeedbackConfig | Mapping[str, Any], role: str
    ) -> StateFeedbackConfig:
        contract = StateFeedbackConfig.require(
            value, device_kind=DeviceKind.LOW_DIM_R
        )
        primitive_names = tuple(
            primitive.name for primitive in contract.format.primitives
        )
        if (
            contract.format.shape != (14,)
            or not np.issubdtype(np.dtype(contract.format.dtype), np.floating)
            or primitive_names != ("j1-7", "j1-7")
        ):
            raise ValueError(
                f"{role} state feedback must be float (14,) with [j1-7, j1-7]"
            )
        return contract

    def start_server(self) -> None:
        """Run the coordinated hardware stack until DeviceStarter stops it."""
        LOGGER.info(
            "Starting DeviceStarter-owned I2RT bimanual teleoperation: %s_%s/%s",
            self.device_name,
            self.device_class_id,
            self.follower_device_class_id,
        )
        run_teleoperation(
            self.teleoperation_config_path,
            self.teleoperation_config,
            device_name=self.device_name,
            leader_device_id=self.device_class_id,
            follower_device_id=self.follower_device_class_id,
            leader_state_feedback=self.state_feedback,
            follower_state_feedback=self.follower_state_feedback,
        )


__all__ = ["I2RTBimanualTeleop"]
