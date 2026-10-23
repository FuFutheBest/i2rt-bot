"""Pytest configuration and shared fixtures for SuperInference tests.

Author: OpenAI Assistant
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root path."""
    return REPO_ROOT


@pytest.fixture
def minimal_device_config() -> dict:
    """Return a minimal config with one fake BaseDevice entry."""
    return {
        "devices": {
            "devices": [
                {
                    "class": "BaseDevice",
                    "device_class_id": 0,
                    "state_feedback": {
                        "fps": 10.0,
                        "buffer_size": 3,
                        "hardware_latency_ms": 0.0,
                        "format": {
                            "shape": [2],
                            "dtype": "float64",
                            "type": ["linear:2"],
                        },
                    },
                }
            ]
        }
    }


@pytest.fixture
def single_arm_action_chunk() -> np.ndarray:
    """Return a simple single-arm action chunk: xyz + quaternion + gripper."""
    return np.array(
        [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5],
            [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0, 0.6],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def dual_arm_action_chunk() -> np.ndarray:
    """Return a dual-arm action chunk with left/right single-arm layouts."""
    left = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5]], dtype=np.float64)
    right = np.array([[0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0, 0.7]], dtype=np.float64)
    return np.concatenate([left, right], axis=1)
