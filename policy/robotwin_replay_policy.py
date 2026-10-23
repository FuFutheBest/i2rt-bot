"""A deterministic joint14 policy used to validate the RoboTwin async stack."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from integrations.robotwin.contracts import action_schema_dim, validate_action
from policy.base import BasePolicy, ObservationType


class RoboTwinReplayPolicy(BasePolicy):
    """Replay qpos targets or hold measured qpos when no replay file is supplied.

    This is intentionally a debugging baseline, not a learned policy.  It lets
    the full async policy -> action executor -> SAPIEN drive pipeline run before
    downloading a checkpoint.
    """

    def __init__(
        self,
        *,
        robot_device: str = "RoboTwinSimRobot_0",
        action_schema: str = "joint14",
        actions_path: Optional[str] = None,
        advance_per_prediction: int = 1,
        inference_delay_ms: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.robot_device = str(robot_device)
        self.action_schema = str(action_schema)
        self.action_dim = action_schema_dim(self.action_schema)
        self.actions_path = Path(actions_path).expanduser() if actions_path else None
        self.advance_per_prediction = max(1, int(advance_per_prediction))
        self.inference_delay_ms = max(0.0, float(inference_delay_ms))
        self._actions: Optional[np.ndarray] = None
        self._index = 0

    def load(self) -> None:
        if self.actions_path is not None:
            if not self.actions_path.is_file():
                raise FileNotFoundError(f"RoboTwin replay actions not found: {self.actions_path}")
            loaded = np.load(self.actions_path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if "actions" not in loaded:
                    raise ValueError("RoboTwin replay .npz must contain an 'actions' array")
                actions = loaded["actions"]
                loaded.close()
            else:
                actions = loaded
            actions = np.asarray(actions, dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != self.action_dim or actions.shape[0] == 0:
                raise ValueError(
                    f"RoboTwin replay actions must have shape [T, {self.action_dim}], got {actions.shape}"
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError("RoboTwin replay actions contain non-finite values")
            self._actions = actions.copy()
        self.initialized = True

    def reset(self) -> None:
        self._index = 0

    def predict(
        self,
        observation: ObservationType,
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        **_: Any,
    ) -> Dict[str, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("RoboTwinReplayPolicy.load() must be called before predict()")
        if self.inference_delay_ms:
            time.sleep(self.inference_delay_ms / 1000.0)
        device_name = self._resolve_control_device(action_configs)
        if self._actions is None:
            current = self._extract_current_joint14(observation)
            chunk = np.repeat(current[None, :], int(chunk_length), axis=0)
        else:
            chunk = self._chunk_from_replay(int(chunk_length))
        return {device_name: chunk.astype(np.float64, copy=False)}

    def _resolve_control_device(self, action_configs: List[Dict[str, Any]]) -> str:
        matching = [config for config in action_configs if config.get("action_dim") == self.action_dim]
        if len(matching) != 1:
            raise ValueError(
                f"RoboTwin replay expects exactly one control target with dimension {self.action_dim}, "
                f"got {len(matching)}"
            )
        return str(matching[0]["device_name"])

    def _extract_current_joint14(self, observation: ObservationType) -> np.ndarray:
        if not isinstance(observation, dict) or self.robot_device not in observation:
            raise ValueError(f"RoboTwin observation does not contain robot device {self.robot_device!r}")
        return validate_action(observation[self.robot_device], self.action_schema)

    def _chunk_from_replay(self, chunk_length: int) -> np.ndarray:
        assert self._actions is not None
        indices = np.minimum(
            np.arange(chunk_length, dtype=np.int64) + self._index,
            self._actions.shape[0] - 1,
        )
        chunk = self._actions[indices].copy()
        self._index = min(self._index + self.advance_per_prediction, self._actions.shape[0] - 1)
        return chunk
