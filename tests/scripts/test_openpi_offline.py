"""Hardware-free contracts for the OpenPi SuperOffline evaluator helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analysis import evaluate_openpi_offline as evaluator_module
from scripts.analysis.evaluate_openpi_offline import (
    SUPER_OFFLINE_PI_CONFIG_NAME,
    _action_step_interval_ns,
    _load_cached_policy,
    _predict_like_live,
    _selected_devices,
)


def test_superoffline_openpi_uses_default_training_config() -> None:
    assert SUPER_OFFLINE_PI_CONFIG_NAME == "default"


class _Backend:
    def __init__(self) -> None:
        self.noise: np.ndarray | None = None

    def infer(self, _inputs: object, **kwargs: object) -> dict[str, np.ndarray]:
        noise = np.asarray(kwargs["noise"], dtype=np.float32)
        self.noise = noise.copy()
        return {"actions": noise}


class _Policy:
    def __init__(self, *, model_action_dim: int = 16) -> None:
        self._policy = _Backend()
        self.pi_config = SimpleNamespace(
            model=SimpleNamespace(action_horizon=3, action_dim=model_action_dim)
        )
        self.calls: list[object] = []

    def predict(
        self,
        observation: object,
        *,
        action_configs: list[dict[str, str]],
        chunk_length: int,
        action_index: object,
    ) -> dict[str, np.ndarray]:
        self.calls.append((observation, action_configs, chunk_length, action_index))
        output = self._policy.infer({"unused": True})
        return {
            action_configs[0]["device_name"]: output["actions"][
                :chunk_length, :16
            ]
        }


def test_openpi_predict_uses_seeded_noise_and_restores_live_backend() -> None:
    policy = _Policy()
    original_backend = policy._policy

    action = _predict_like_live(
        policy,
        {"WebCameraOpenPi_1": np.zeros((2, 2, 3), dtype=np.uint8)},
        robot_device="RizonRobot_2",
        chunk_length=2,
        seed=17,
    )

    expected = np.random.default_rng(17).standard_normal((3, 16), dtype=np.float32)
    assert policy._policy is original_backend
    assert policy.calls[0][3] is None
    assert action.shape == (2, 16)
    assert np.array_equal(action, expected[:2].astype(np.float64))


def test_openpi_predict_uses_native_model_width_before_16d_output_transform() -> None:
    policy = _Policy(model_action_dim=32)

    action = _predict_like_live(
        policy,
        {"WebCameraOpenPi_1": np.zeros((2, 2, 3), dtype=np.uint8)},
        robot_device="RizonRobot_2",
        chunk_length=2,
        seed=23,
    )

    expected = np.random.default_rng(23).standard_normal((3, 32), dtype=np.float32)
    assert policy._policy.noise is not None
    assert policy._policy.noise.shape == (3, 32)
    assert action.shape == (2, 16)
    assert np.array_equal(action, expected[:2, :16].astype(np.float64))


def test_openpi_offline_uses_configured_camera_order() -> None:
    camera_devices, robot_device = _selected_devices(
        {
            "obs": {
                "devices": [
                    "WebCameraOpenPi_3",
                    "WebCameraOpenPi_1",
                    "RizonRobot_2",
                ]
            }
        }
    )

    assert camera_devices == ["WebCameraOpenPi_3", "WebCameraOpenPi_1"]
    assert robot_device == "RizonRobot_2"


def test_openpi_action_timebase_uses_explicit_control_rate() -> None:
    assert _action_step_interval_ns({"control_fps": 30.0}) == 33_333_333


def test_openpi_batch_policy_is_loaded_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    policy = object()

    def build(*_args: object, **_kwargs: object) -> object:
        calls.append(object())
        return policy

    monkeypatch.setattr(evaluator_module, "_CACHED_POLICY", None)
    monkeypatch.setattr(evaluator_module, "_build_policy", build)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    first = _load_cached_policy({}, checkpoint=checkpoint, prompt="task", device="cuda")
    second = _load_cached_policy({}, checkpoint=checkpoint, prompt="task", device="cuda")

    assert first is second is policy
    assert len(calls) == 1
