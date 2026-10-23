"""Focused observation-history contracts used by visual-memory policies."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from consumer.inference.common.observation_builder import ObservationBuilder


def _frame(timestamp_ns: int, value: int) -> dict[str, Any]:
    camera = np.array([value], dtype=np.int64)
    robot = np.array([value + 100], dtype=np.int64)
    return {
        "frame_timestamp_ns": timestamp_ns,
        "data": {
            "Camera_0": (timestamp_ns, camera),
            "Robot_0": (timestamp_ns, robot),
        },
    }


def _history_builder(frame_lookup: dict[int, dict[str, Any]]) -> ObservationBuilder:
    policy = SimpleNamespace(
        obs_horizon=4,
        obs_time_interval=1e-9,
        condition_history_stride=1,
    )
    owner = SimpleNamespace(
        policy=policy,
        summary_shm_lock=nullcontext(),
        find_frame_by_timestamp=lambda timestamp_ns, tolerance_ms: frame_lookup.get(
            timestamp_ns
        ),
    )
    return ObservationBuilder(
        read_all_device_data=lambda **_kwargs: None,
        master_device="Robot_0",
        owner=owner,
    )


@pytest.mark.unit
def test_observation_history_is_oldest_to_latest_with_source_timestamps() -> None:
    builder = _history_builder(
        {
            397: _frame(397, 1),
            398: _frame(398, 2),
            399: _frame(399, 3),
        }
    )
    latest = {
        "Camera_0": np.array([4], dtype=np.int64),
        "Robot_0": np.array([104], dtype=np.int64),
    }

    history = builder._collect_observation_history(latest, 400)

    assert history is not None
    assert [int(frame[0]) for frame in history["Camera_0"]] == [1, 2, 3, 4]
    assert history["_history_frame_timestamps_ns"].tolist() == [397, 398, 399, 400]
    assert [bool(value) for value in history["Camera_0_is_pad"]] == [
        False,
        False,
        False,
        False,
    ]


@pytest.mark.unit
def test_episode_reset_pads_history_with_anchor_source_timestamp() -> None:
    builder = _history_builder({})
    builder.reset_history()
    latest = {
        "Camera_0": np.array([7], dtype=np.int64),
        "Robot_0": np.array([107], dtype=np.int64),
    }

    history = builder._collect_observation_history(latest, 500)

    assert history is not None
    assert [int(frame[0]) for frame in history["Camera_0"]] == [7, 7, 7, 7]
    assert history["_history_frame_timestamps_ns"].tolist() == [500, 500, 500, 500]
    assert [bool(value) for value in history["Camera_0_is_pad"]] == [
        True,
        True,
        True,
        False,
    ]
