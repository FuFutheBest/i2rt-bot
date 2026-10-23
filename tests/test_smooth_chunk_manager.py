"""Tests for the configurable two-stage smooth chunk manager.

Author: Cursor Agent
"""

import numpy as np
import pytest

from manager.utils.smooth_chunk_manager import SmoothChunkManager


def _vector_chunk(length: int = 5) -> dict[str, np.ndarray]:
    values = np.arange(length, dtype=np.float64)
    return {"robot": (values * values).reshape(-1, 1)}


def _pose_chunk(length: int = 5) -> dict[str, np.ndarray]:
    chunk = np.zeros((length, 8), dtype=np.float64)
    chunk[:, 0] = np.arange(length, dtype=np.float64)
    angles = np.linspace(0.0, np.pi / 2.0, length)
    chunk[:, 5] = np.sin(angles / 2.0)
    chunk[:, 6] = np.cos(angles / 2.0)
    chunk[:, 7] = np.linspace(0.0, 1.0, length)
    return {"robot": chunk}


def test_requires_at_least_one_processing_stage() -> None:
    with pytest.raises(ValueError, match="At least one"):
        SmoothChunkManager()


@pytest.mark.parametrize(
    ("smoothing", "upsampling", "message"),
    [
        ({"method": "median"}, None, "smoothing.method"),
        (None, {"method": "nearest"}, "upsampling.method"),
    ],
)
def test_rejects_unknown_methods(
    smoothing: dict[str, object] | None,
    upsampling: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SmoothChunkManager(smoothing=smoothing, upsampling=upsampling)


def test_savgol_stage_preserves_policy_chunk_length() -> None:
    manager = SmoothChunkManager(
        smoothing={"method": "savgol", "upsample_ratio": 2, "window": 5, "polyorder": 2}
    )
    manager.put(_vector_chunk(length=7))

    assert manager.current_action_chunk_dict is not None
    assert manager.current_action_chunk_dict["robot"].shape == (7, 1)


def test_moving_average_stage_preserves_length_when_fps_match() -> None:
    manager = SmoothChunkManager(
        smoothing={
            "method": "moving_average",
            "upsample_ratio": 3,
            "window": 5,
            "source_fps": 30.0,
            "command_fps": 30.0,
        }
    )
    manager.put(_vector_chunk(length=7))

    assert manager.current_action_chunk_dict is not None
    assert manager.current_action_chunk_dict["robot"].shape == (7, 1)


def test_moving_average_stage_resamples_to_command_fps() -> None:
    manager = SmoothChunkManager(
        smoothing={
            "method": "moving_average",
            "upsample_ratio": 2,
            "window": 3,
            "source_fps": 10.0,
            "command_fps": 20.0,
        }
    )
    manager.put(_vector_chunk(length=4))

    assert manager.current_action_chunk_dict is not None
    assert manager.current_action_chunk_dict["robot"].shape == (7, 1)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_execution_upsampling_expands_timeline(method: str) -> None:
    manager = SmoothChunkManager(upsampling={"method": method, "ratio": 3})
    manager.put(_vector_chunk(length=4))

    assert manager.current_action_chunk_dict is not None
    result = manager.current_action_chunk_dict["robot"]
    assert result.shape == (10, 1)
    np.testing.assert_allclose(result[[0, -1], 0], np.array([0.0, 9.0]))


def test_execution_upsampling_tracks_original_action_timeline() -> None:
    manager = SmoothChunkManager(
        upsampling={"method": "linear", "ratio": 2},
        execution_fps=10.0,
    )
    manager.put(_vector_chunk(length=4))

    np.testing.assert_allclose(
        manager.get_current_original_action_offsets(),
        np.arange(7, dtype=np.float64) / 2.0,
    )
    assert manager.get_original_action_step_interval_ns() == 200_000_000

    assert manager.get() is not None
    assert manager.get_original_current_offset() == pytest.approx(0.0)
    assert manager.get_original_current_step() == 0
    assert manager.get() is not None
    assert manager.get_original_current_offset() == pytest.approx(0.5)
    assert manager.get_original_current_step() == 1


def test_auto_alignment_accounts_for_upsampling_boundary_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_times = iter([1_000_000_000, 2_000_000_000])
    monkeypatch.setattr(
        "manager.utils.action_manager.time.time_ns",
        lambda: next(current_times),
    )
    manager = SmoothChunkManager(
        upsampling={"method": "linear", "ratio": 2},
        drop_first_actions="auto",
        execution_fps=10.0,
    )
    chunk = _vector_chunk(length=6)

    manager.put(chunk, observation_timestamp=900_000_000)
    assert manager.get() is not None
    manager.put(chunk, observation_timestamp=1_800_000_000)

    original_offsets = manager.get_current_original_action_offsets()
    np.testing.assert_allclose(
        original_offsets,
        np.arange(1.0, 5.0 + 0.5, 0.5),
    )
    assert manager.current_original_start_offset == pytest.approx(1.0)
    assert manager.current_original_start_index == 1
    assert manager.get_original_action_step_interval_ns() == 200_000_000


def test_smoothing_then_execution_upsampling_runs_in_order() -> None:
    manager = SmoothChunkManager(
        smoothing={"method": "savgol", "upsample_ratio": 2, "window": 5},
        upsampling={"method": "linear", "ratio": 2},
    )
    manager.put(_vector_chunk(length=5))

    assert manager.current_action_chunk_dict is not None
    assert manager.current_action_chunk_dict["robot"].shape == (9, 1)


def test_quaternion_channels_remain_normalized() -> None:
    manager = SmoothChunkManager(
        smoothing={"method": "savgol", "upsample_ratio": 2, "window": 5},
        upsampling={"method": "linear", "ratio": 2},
    )
    manager.put(_pose_chunk(length=5))

    assert manager.current_action_chunk_dict is not None
    quaternions = manager.current_action_chunk_dict["robot"][:, 3:7]
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6)


def test_boundary_context_uses_dispatched_actions() -> None:
    manager = SmoothChunkManager(
        smoothing={
            "method": "savgol",
            "upsample_ratio": 2,
            "window": 5,
            "boundary_context_steps": 2,
            "boundary_blend_steps": 2,
        }
    )
    manager.put(_vector_chunk(length=5))
    assert manager.get() is not None
    assert manager.get() is not None
    assert manager.get() is not None

    context = manager._get_dispatched_context()
    assert manager.current_action_chunk_dict is not None
    np.testing.assert_array_equal(
        context["robot"],
        manager.current_action_chunk_dict["robot"][1:3],
    )


def test_drop_and_skip_are_applied_after_processing() -> None:
    manager = SmoothChunkManager(
        upsampling={"method": "linear", "ratio": 2},
        drop_first_actions=2,
        skip_first_chunk=True,
    )
    manager.put(_vector_chunk(length=4), observation_timestamp=1)
    assert manager.get() is None

    manager.put(_vector_chunk(length=4), observation_timestamp=2)
    first_action = manager.get()
    assert first_action is not None
    np.testing.assert_allclose(first_action["robot"], np.array([1.0]))
    assert manager.get_original_current_step() == 1


def test_dual_arm_quaternion_layout_is_supported() -> None:
    chunk = np.zeros((3, 16), dtype=np.float64)
    chunk[:, 6] = 1.0
    chunk[:, 14] = 1.0
    manager = SmoothChunkManager(upsampling={"method": "linear", "ratio": 2})
    manager.put({"robot": chunk})

    assert manager.current_action_chunk_dict is not None
    result = manager.current_action_chunk_dict["robot"]
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(result[:, 11:15], axis=1), 1.0)


def test_aggressive_bringup_settings_enable_unmistakable_smoothing() -> None:
    manager = SmoothChunkManager(
        drop_first_actions=18,
        smoothing={
            "method": "savgol",
            "window": 47,
            "polyorder": 1,
            "mode": "interp",
            "upsample_ratio": 1,
            "boundary_context_steps": 18,
            "boundary_blend_steps": 30,
            "smooth_gripper": False,
        },
        upsampling={"method": None},
    )

    chunk = np.zeros((48, 16), dtype=np.float64)
    chunk[:, 0] = np.tile([1.0, -1.0], 24)
    chunk[:, 6] = 1.0
    chunk[:, 14] = 1.0
    chunk[:, 7] = np.tile([0.0, 1.0], 24)
    chunk[:, 15] = np.tile([1.0, 0.0], 24)
    manager.put({"robot": chunk})

    assert manager.current_action_chunk_dict is not None
    smoothed = manager.current_action_chunk_dict["robot"]
    input_total_variation = np.abs(np.diff(chunk[:, 0])).sum()
    output_total_variation = np.abs(np.diff(smoothed[:, 0])).sum()
    assert output_total_variation < 0.05 * input_total_variation
    np.testing.assert_array_equal(smoothed[:, [7, 15]], chunk[:, [7, 15]])

    for _ in range(30):
        assert manager.get() is not None
    replacement = chunk.copy()
    replacement[:, 0] *= -1.0
    replacement[:, 7] = chunk[::-1, 7]
    replacement[:, 15] = chunk[::-1, 15]
    manager.put({"robot": replacement}, observation_timestamp=1)

    assert manager.current_action_chunk_dict is not None
    replacement_smoothed = manager.current_action_chunk_dict["robot"]
    assert replacement_smoothed.shape == (30, 16)
    replacement_input = replacement[18:]
    replacement_input_variation = np.abs(np.diff(replacement_input[:, 0])).sum()
    replacement_output_variation = np.abs(np.diff(replacement_smoothed[:, 0])).sum()
    assert replacement_output_variation < 0.05 * replacement_input_variation
    np.testing.assert_array_equal(
        replacement_smoothed[:, [7, 15]], replacement_input[:, [7, 15]]
    )
