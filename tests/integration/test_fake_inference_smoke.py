"""Mocked inference smoke test for the core policy/action contract.

Author: OpenAI Assistant
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from manager.utils.action_manager import BaseChunkManager
from policy.base import BasePolicy, ObservationType


class FakePolicy(BasePolicy):
    """Small deterministic policy for hardware-free integration tests."""

    def predict(
        self,
        observation: ObservationType,
        action_configs: List[Dict],
        chunk_length: int = 1,
    ) -> Dict[str, np.ndarray]:
        obs_vector = self._to_vector(observation)
        base_value = float(obs_vector.sum()) if obs_vector.size else 0.0
        outputs: Dict[str, np.ndarray] = {}
        for action_config in action_configs:
            target = action_config["device_name"]
            action_dim = int(action_config["action_dim"])
            chunk = np.full((chunk_length, action_dim), base_value, dtype=np.float64)
            outputs[target] = chunk
        return outputs


@pytest.mark.integration
def test_fake_policy_to_chunk_manager_smoke() -> None:
    """A fake observation should become dispatchable action steps without hardware."""
    policy = FakePolicy()
    policy.load()
    chunk_manager = BaseChunkManager()
    observation = {
        "Camera_0": np.ones((2, 2), dtype=np.float64),
        "Robot_0": np.array([0.5, 0.5], dtype=np.float64),
    }
    action_configs = [{"device_name": "Robot_0", "action_dim": 8}]

    action_chunk = policy.predict(observation, action_configs, chunk_length=3)
    chunk_manager.put(action_chunk, observation_timestamp=123)

    assert policy.initialized is True
    assert action_chunk["Robot_0"].shape == (3, 8)
    for _ in range(3):
        action_step = chunk_manager.get()
        assert action_step is not None
        assert action_step["Robot_0"].shape == (8,)
    assert chunk_manager.get() is None
