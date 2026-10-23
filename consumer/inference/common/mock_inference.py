"""Checkpoint-free policy used by standalone and Control Panel smoke tests."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


_LEFT_ARM_STATE = slice(14, 22)
_RIGHT_ARM_STATE = slice(36, 44)


class ControlPanelNoMotionPolicy:
    """Hold both live TCP poses while moving both grippers in lockstep."""

    block = True
    enable_rtc = False
    obs_horizon = 1

    def __init__(
        self,
        *,
        robot_device: str = "RizonRobot_2",
        gripper_min: float = 0.0,
        gripper_max: float = 0.08,
        gripper_travel_seconds: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
        **_unused: Any,
    ) -> None:
        self.robot_device = str(robot_device)
        self.gripper_min = float(gripper_min)
        self.gripper_max = float(gripper_max)
        if not self.gripper_max > self.gripper_min:
            raise ValueError("gripper_max must be greater than gripper_min")
        self.gripper_travel_seconds = max(0.5, float(gripper_travel_seconds))
        self._clock = clock
        self._last_update: Optional[float] = None
        self._gripper_target: Optional[float] = None
        self._gripper_direction = 1.0

    def load(self) -> None:
        return None

    def reset(self) -> None:
        self._last_update = None
        self._gripper_target = None
        self._gripper_direction = 1.0

    @staticmethod
    def _latest(value: Any) -> np.ndarray:
        if isinstance(value, list):
            value = value[-1]
        array = np.asarray(value, dtype=np.float64)
        while array.ndim > 1:
            array = array[-1]
        return array

    def _robot_state(self, observation: Mapping[str, Any]) -> np.ndarray:
        value = observation.get(self.robot_device)
        if value is None:
            value = next(
                (
                    candidate
                    for name, candidate in observation.items()
                    if str(name).casefold().startswith(("rizon", "robot"))
                ),
                None,
            )
        if value is None:
            raise ValueError("Rizon state is missing from the mock policy observation")
        state = self._latest(value)
        if state.size < 44 or not np.all(np.isfinite(state[:44])):
            raise ValueError("Mock policy requires a finite 44D Rizon state")
        return state

    def _next_gripper_target(self, state: np.ndarray) -> float:
        now = float(self._clock())
        if self._gripper_target is None:
            measured = float(np.mean([state[21], state[43]]))
            self._gripper_target = float(
                np.clip(measured, self.gripper_min, self.gripper_max)
            )
            self._last_update = now
            return self._gripper_target
        elapsed = max(0.0, min(0.5, now - float(self._last_update or now)))
        self._last_update = now
        travel = self.gripper_max - self.gripper_min
        target = self._gripper_target + (
            self._gripper_direction * travel * elapsed / self.gripper_travel_seconds
        )
        while target > self.gripper_max or target < self.gripper_min:
            if target > self.gripper_max:
                target = self.gripper_max - (target - self.gripper_max)
                self._gripper_direction = -1.0
            elif target < self.gripper_min:
                target = self.gripper_min + (self.gripper_min - target)
                self._gripper_direction = 1.0
        self._gripper_target = float(target)
        return self._gripper_target

    @staticmethod
    def _arm_hold_action(arm_state: np.ndarray, gripper: float) -> np.ndarray:
        # Rizon state uses [xyz, qw, qx, qy, qz, grip], while eef_abs control
        # uses [xyz, qx, qy, qz, qw, grip].
        return np.concatenate(
            (
                arm_state[:3],
                arm_state[4:7],
                arm_state[3:4],
                np.asarray([gripper], dtype=np.float64),
            )
        )

    def predict(
        self,
        observation: Mapping[str, Any],
        action_configs: Sequence[Mapping[str, Any]],
        chunk_length: int = 50,
        action_index: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        del action_index
        state = self._robot_state(observation)
        gripper = self._next_gripper_target(state)
        hold = np.concatenate(
            (
                self._arm_hold_action(state[_LEFT_ARM_STATE], gripper),
                self._arm_hold_action(state[_RIGHT_ARM_STATE], gripper),
            )
        )
        if hold.shape != (16,):
            raise RuntimeError(f"Unexpected mock action shape: {hold.shape}")
        if len(action_configs) != 1:
            raise ValueError("Mock inference requires exactly one action target")
        result: dict[str, np.ndarray] = {}
        for action_config in action_configs:
            action_dim = int(action_config.get("action_dim", 0))
            device_name = str(action_config.get("device_name", ""))
            if action_dim != 16 or not device_name:
                raise ValueError(
                    "Mock inference only supports one 16D dual-arm action target"
                )
            result[device_name] = np.repeat(
                hold.reshape(1, 16), max(1, int(chunk_length)), axis=0
            )
        return result


__all__ = ["ControlPanelNoMotionPolicy"]
