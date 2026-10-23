"""Tests for latency-based automatic action dropping.

Author: Cursor Agent
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from manager.utils import load_chunk_manager
from manager.utils.action_manager import BaseChunkManager


def _chunk(length: int = 12) -> dict[str, np.ndarray]:
    """Create a deterministic single-device action chunk."""
    return {"robot": np.arange(length, dtype=np.float64).reshape(length, 1)}


def test_fixed_drop_first_actions_preserves_existing_behavior() -> None:
    """A fixed integer should still apply only when replacing a chunk."""
    manager = BaseChunkManager(drop_first_actions=3)

    manager.put(_chunk(), observation_timestamp=100)
    np.testing.assert_array_equal(manager.get()["robot"], np.array([0.0]))

    manager.put(_chunk(), observation_timestamp=200)
    np.testing.assert_array_equal(manager.get()["robot"], np.array([3.0]))
    assert manager.get_original_current_step() == 3


def test_auto_drop_uses_current_chunk_end_to_end_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fast chunk must not inherit a preceding chunk's slower P95 latency."""
    current_times: Iterator[int] = iter([1_300_000_000, 2_129_000_000])
    monkeypatch.setattr(
        "manager.utils.action_manager.time.time_ns",
        lambda: next(current_times),
    )
    manager = BaseChunkManager(drop_first_actions="auto", execution_fps=30.0)

    manager.put(_chunk(), observation_timestamp=1_000_000_000)
    np.testing.assert_array_equal(manager.get()["robot"], np.array([0.0]))

    manager.put(_chunk(), observation_timestamp=2_000_000_000)
    np.testing.assert_allclose(manager.get()["robot"], np.array([3.87]))
    assert manager.get_original_current_offset() == pytest.approx(3.87)
    assert manager.get_original_current_step() == 4
    assert manager._inference_latency_tracker.p95() > 0.25


def test_auto_drop_fractionally_resamples_eef_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fractional offsets should linearly interpolate vectors and SLERP rotations."""
    current_times: Iterator[int] = iter([1_000_000_000, 2_050_000_000])
    monkeypatch.setattr(
        "manager.utils.action_manager.time.time_ns",
        lambda: next(current_times),
    )
    manager = BaseChunkManager(drop_first_actions="auto", execution_fps=10.0)

    chunk = np.zeros((3, 16), dtype=np.float64)
    chunk[:, 0] = [0.0, 10.0, 20.0]
    chunk[:, 3:7] = Rotation.from_euler(
        "z",
        [[0.0], [90.0], [180.0]],
        degrees=True,
    ).as_quat()
    chunk[:, 7] = [0.0, 1.0, 2.0]
    chunk[:, 8] = [20.0, 10.0, 0.0]
    chunk[:, 11:15] = Rotation.from_euler(
        "x",
        [[0.0], [-90.0], [-180.0]],
        degrees=True,
    ).as_quat()
    chunk[:, 15] = [2.0, 1.0, 0.0]

    manager.put({"robot": chunk}, observation_timestamp=1_000_000_000)
    manager.put({"robot": chunk}, observation_timestamp=2_000_000_000)

    first_action = manager.get()
    assert first_action is not None
    np.testing.assert_allclose(
        first_action["robot"][[0, 7, 8, 15]],
        [5.0, 0.5, 15.0, 1.5],
    )
    expected_left_quaternion = Rotation.from_euler("z", 45.0, degrees=True).as_quat()
    expected_right_quaternion = Rotation.from_euler("x", -45.0, degrees=True).as_quat()
    left_similarity = abs(
        float(np.dot(first_action["robot"][3:7], expected_left_quaternion))
    )
    right_similarity = abs(
        float(np.dot(first_action["robot"][11:15], expected_right_quaternion))
    )
    assert left_similarity == pytest.approx(1.0)
    assert right_similarity == pytest.approx(1.0)
    assert manager.get_original_current_offset() == pytest.approx(0.5)
    assert manager.get_original_current_step() == 1

    second_action = manager.get()
    assert second_action is not None
    np.testing.assert_allclose(
        second_action["robot"][[0, 7, 8, 15]],
        [15.0, 1.5, 5.0, 0.5],
    )
    assert manager.get() is None


def test_auto_drop_uses_p95_only_when_timestamp_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical P95 remains an explicit fallback for timestamp-less chunks."""
    monkeypatch.setattr(
        "manager.utils.action_manager.time.time_ns",
        lambda: 1_100_000_000,
    )
    manager = BaseChunkManager(drop_first_actions="auto", execution_fps=20.0)

    manager.put(_chunk(), observation_timestamp=1_000_000_000)
    manager.put(_chunk(), observation_timestamp=None)

    np.testing.assert_array_equal(manager.get()["robot"], np.array([2.0]))
    assert manager.get_original_current_step() == 2


def test_auto_drop_does_not_guess_without_latency_data() -> None:
    """Automatic mode should keep zero offset when no latency can be measured."""
    manager = BaseChunkManager(drop_first_actions="auto", execution_fps=30.0)

    manager.put(_chunk(), observation_timestamp=None)
    manager.put(_chunk(), observation_timestamp=None)

    np.testing.assert_array_equal(manager.get()["robot"], np.array([0.0]))
    assert manager.get_original_current_step() == 0


@pytest.mark.parametrize(
    "manager_type",
    [
        "base_manager",
        "smooth_chunk_manager",
    ],
)
def test_loader_injects_execution_fps_for_compatible_managers(
    manager_type: str,
) -> None:
    """Every compatible manager should receive executor FPS for automatic mode."""
    manager_config = {
        "type": manager_type,
        "drop_first_actions": "auto",
        "interpolate_auto_drop": True,
        "quaternion_format": "xyzw",
        "auto_drop_gripper_mode": "linear",
    }
    if manager_type == "smooth_chunk_manager":
        manager_config["smoothing"] = {"method": "savgol"}
    manager = load_chunk_manager(manager_config, execution_fps=25.0)

    assert manager.drop_first_actions == "auto"
    assert manager.execution_fps == 25.0
    assert manager.interpolate_auto_drop is True
    assert manager.quaternion_format == "xyzw"
    assert manager.auto_drop_gripper_mode == "linear"


def test_drop_first_actions_rejects_unknown_string() -> None:
    """Only the explicit 'auto' string is accepted."""
    with pytest.raises(ValueError, match="non-negative integer or 'auto'"):
        BaseChunkManager(drop_first_actions="dynamic")


def test_base_auto_drop_config_uses_base_manager() -> None:
    """The reusable auto-drop profile must retain the base manager contract."""
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "action_executor"
        / "base_auto_drop.yaml"
    )
    with config_path.open(encoding="utf-8") as config_file:
        manager_config = yaml.safe_load(config_file)["action_executor"]["chunk_manager"]

    manager = load_chunk_manager(manager_config, execution_fps=30.0)
    assert type(manager) is BaseChunkManager
    assert manager.drop_first_actions == "auto"
