#!/usr/bin/env python3
"""Tests for ActionExecutor-owned logging of original and aligned action chunks.

Author: SuperInference Contributors
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from manager.base_policy_runner import BasePolicyRunner
from manager.utils.action_manager import BaseChunkManager
from manager.utils.smooth_chunk_manager import SmoothChunkManager


class _RecordingXYZLogger:
    """Capture logger calls without starting the asynchronous file writer."""

    def __init__(self) -> None:
        self.meta_records: list[dict[str, Any]] = []
        self.action_records: list[dict[str, Any]] = []

    def log_meta(self, event_type: str, **fields: Any) -> None:
        self.meta_records.append({"event_type": event_type, **fields})

    def log_action(
        self,
        *,
        device_name: str,
        action: np.ndarray,
        timestamp_ns: int,
        event_type: str = "action_xyz",
        **fields: Any,
    ) -> None:
        self.action_records.append(
            {
                "event_type": event_type,
                "device_name": device_name,
                "action": np.asarray(action).copy(),
                "timestamp_ns": int(timestamp_ns),
                **fields,
            }
        )


def _make_runner(
    chunk_manager: BaseChunkManager,
    xyz_logger: _RecordingXYZLogger,
    *,
    execution_clock: str = "wall_time",
) -> BasePolicyRunner:
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.chunk_manager = chunk_manager
    runner.action_state_xyz_logger = xyz_logger
    runner.action_step_interval_ns = 100_000_000
    runner.execution_clock = execution_clock
    runner._last_action_dispatch_wall_timestamp_ns = None
    runner._estimated_wall_action_step_interval_ns = runner.action_step_interval_ns
    runner._action_chunk_log_index = 0
    runner._pending_action_chunk_log = None
    return runner


def test_action_executor_logs_exact_fractionally_aligned_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BaseChunkManager(
        drop_first_actions="auto",
        execution_fps=10.0,
        interpolate_auto_drop=True,
    )
    first_chunk = {"robot": np.zeros((4, 8), dtype=np.float64)}
    manager.put(first_chunk, observation_timestamp=1_000_000_000)

    replacement = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.01],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.02],
            [2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.03],
            [3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.04],
        ],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        "manager.utils.action_manager.time.time_ns",
        lambda: 2_000_000_000,
    )
    manager.put(
        {"robot": replacement},
        observation_timestamp=1_850_000_000,
    )
    expected_aligned = manager.get_current_action_chunk()["robot"]
    assert manager.current_original_start_offset == pytest.approx(1.5)
    assert expected_aligned.shape == (2, 8)

    xyz_logger = _RecordingXYZLogger()
    runner = _make_runner(manager, xyz_logger)
    runner._queue_action_chunk_log(
        original_action_chunk_dict={"robot": replacement},
        observation_timestamp_ns=1_850_000_000,
        policy_write_timestamp_ns=2_010_000_000,
    )

    manager.current_action_chunk_dict["robot"][:] = -1.0
    runner._log_pending_action_chunk(switch_timestamp_ns=2_020_000_000)

    assert [record["event_type"] for record in xyz_logger.meta_records] == [
        "chunk_switch",
        "pred_chunk_start",
        "pred_chunk_end",
    ]
    all_actions = [
        record
        for record in xyz_logger.action_records
        if record["event_type"] == "all_actions"
    ]
    aligned_actions = [
        record
        for record in xyz_logger.action_records
        if record["event_type"] == "aligned_actions"
    ]
    assert len(all_actions) == 4
    np.testing.assert_allclose(
        np.stack([record["action"] for record in all_actions]),
        replacement,
    )
    assert [record["timestamp_ns"] for record in all_actions] == [
        1_870_000_000,
        1_970_000_000,
        2_070_000_000,
        2_170_000_000,
    ]
    assert {
        record["pred_chunk_timestamp_source"] for record in all_actions
    } == {"action_executor_offset_timeline"}
    assert all(record["chunk_stage"] == "policy_original" for record in all_actions)

    assert len(aligned_actions) == 2
    np.testing.assert_allclose(
        np.stack([record["action"] for record in aligned_actions]),
        expected_aligned,
    )
    assert [record["timestamp_ns"] for record in aligned_actions] == [
        2_020_000_000,
        2_120_000_000,
    ]
    assert {
        record["pred_chunk_timestamp_source"]
        for record in aligned_actions
    } == {"action_executor_dispatch", "action_executor_extrapolated"}
    for record in aligned_actions:
        assert record["pred_chunk_index"] == 1
        assert record["chunk_length"] == 2
        assert record["original_chunk_length"] == 4
        assert record["original_start_offset"] == pytest.approx(1.5)
        assert record["chunk_stage"] == "action_executor_aligned"
    chunk_start = xyz_logger.meta_records[1]
    chunk_end = xyz_logger.meta_records[2]
    assert chunk_start["timestamp_ns"] == all_actions[0]["timestamp_ns"]
    assert chunk_end["timestamp_ns"] == all_actions[-1]["timestamp_ns"]
    assert runner._pending_action_chunk_log is None


def test_action_executor_marks_robotwin_joint14_chunks() -> None:
    manager = BaseChunkManager(execution_fps=10.0)
    joint_chunk = np.arange(42, dtype=np.float64).reshape(3, 14)
    manager.put({"RoboTwinSimRobot_0": joint_chunk}, observation_timestamp=1)
    xyz_logger = _RecordingXYZLogger()
    runner = _make_runner(manager, xyz_logger)

    runner._queue_action_chunk_log(
        original_action_chunk_dict={"RoboTwinSimRobot_0": joint_chunk},
        observation_timestamp_ns=1,
        policy_write_timestamp_ns=2,
    )
    runner._log_pending_action_chunk(switch_timestamp_ns=3)

    chunk_records = [
        record
        for record in xyz_logger.action_records
        if record["event_type"] in {"all_actions", "aligned_actions"}
    ]
    assert chunk_records
    assert {record["vector_layout"] for record in chunk_records} == {"joint14"}


def test_action_executor_does_not_log_empty_processed_chunk() -> None:
    manager = BaseChunkManager(drop_first_actions=4)
    manager.put(
        {"robot": np.zeros((4, 8), dtype=np.float64)},
        observation_timestamp=1,
    )
    manager.put(
        {"robot": np.ones((4, 8), dtype=np.float64)},
        observation_timestamp=2,
    )
    xyz_logger = _RecordingXYZLogger()
    runner = _make_runner(manager, xyz_logger)

    runner._queue_action_chunk_log(
        original_action_chunk_dict={"robot": np.ones((4, 8), dtype=np.float64)},
        observation_timestamp_ns=2,
        policy_write_timestamp_ns=3,
    )
    runner._log_pending_action_chunk(switch_timestamp_ns=4)

    assert runner._pending_action_chunk_log is None
    assert runner._action_chunk_log_index == 0
    assert xyz_logger.meta_records == []
    assert xyz_logger.action_records == []


def test_action_executor_uses_distinct_original_and_execution_intervals() -> None:
    manager = SmoothChunkManager(
        upsampling={"method": "linear", "ratio": 2},
        execution_fps=10.0,
    )
    original_chunk = {
        "robot": np.arange(4, dtype=np.float64).reshape(-1, 1)
    }
    manager.put(original_chunk, observation_timestamp=1)

    xyz_logger = _RecordingXYZLogger()
    runner = _make_runner(manager, xyz_logger)
    runner._queue_action_chunk_log(
        original_action_chunk_dict=original_chunk,
        observation_timestamp_ns=1,
        policy_write_timestamp_ns=2,
    )
    runner._log_pending_action_chunk(switch_timestamp_ns=1_000_000_000)

    all_actions = [
        record
        for record in xyz_logger.action_records
        if record["event_type"] == "all_actions"
    ]
    aligned_actions = [
        record
        for record in xyz_logger.action_records
        if record["event_type"] == "aligned_actions"
    ]

    assert [record["timestamp_ns"] for record in all_actions] == [
        1_000_000_000,
        1_200_000_000,
        1_400_000_000,
        1_600_000_000,
    ]
    assert [record["timestamp_ns"] for record in aligned_actions] == [
        1_000_000_000,
        1_100_000_000,
        1_200_000_000,
        1_300_000_000,
        1_400_000_000,
        1_500_000_000,
        1_600_000_000,
    ]
    np.testing.assert_allclose(
        [record["original_action_offset"] for record in aligned_actions],
        np.arange(7, dtype=np.float64) / 2.0,
    )
    assert all_actions[-1]["timestamp_ns"] == aligned_actions[-1]["timestamp_ns"]
    assert all(
        record["original_step_interval_ns"] == 200_000_000
        for record in all_actions
    )
    assert all(
        record["aligned_step_interval_ns"] == 100_000_000
        for record in aligned_actions
    )


def test_action_executor_marks_sim_time_chunks_as_estimated_wall_preview() -> None:
    manager = BaseChunkManager(execution_fps=10.0)
    chunk = {"robot": np.arange(24, dtype=np.float64).reshape(3, 8)}
    manager.put(chunk, observation_timestamp=1)

    xyz_logger = _RecordingXYZLogger()
    runner = _make_runner(manager, xyz_logger, execution_clock="sim_time")
    runner._queue_action_chunk_log(
        original_action_chunk_dict=chunk,
        observation_timestamp_ns=1,
        policy_write_timestamp_ns=2,
    )
    runner._log_pending_action_chunk(switch_timestamp_ns=1_000_000_000)

    chunk_records = [
        record
        for record in xyz_logger.action_records
        if record["event_type"] in {"all_actions", "aligned_actions"}
    ]

    assert chunk_records
    assert [
        record["timestamp_ns"]
        for record in chunk_records
        if record["event_type"] == "aligned_actions"
    ] == [1_000_000_000, 1_100_000_000, 1_200_000_000]
    assert {record["pred_chunk_timestamp_source"] for record in chunk_records} == {
        "estimated_wall_dispatch_time"
    }
    assert {record["timestamp_semantics"] for record in chunk_records} == {
        "estimated_wall_dispatch_time"
    }
    assert [
        record["sim_preview_offset_ns"]
        for record in chunk_records
        if record["event_type"] == "aligned_actions"
    ] == [0, 100_000_000, 200_000_000]
    assert [
        record["wall_preview_offset_ns"]
        for record in chunk_records
        if record["event_type"] == "aligned_actions"
    ] == [0, 100_000_000, 200_000_000]


def test_action_executor_sim_time_preview_uses_observed_wall_interval() -> None:
    manager = BaseChunkManager(execution_fps=10.0)
    chunk = {"robot": np.arange(24, dtype=np.float64).reshape(3, 8)}
    manager.put(chunk, observation_timestamp=1)

    xyz_logger = _RecordingXYZLogger()
    runner = _make_runner(manager, xyz_logger, execution_clock="sim_time")
    runner._update_wall_dispatch_interval_estimate(1_000_000_000)
    runner._update_wall_dispatch_interval_estimate(1_300_000_000)
    runner._queue_action_chunk_log(
        original_action_chunk_dict=chunk,
        observation_timestamp_ns=1,
        policy_write_timestamp_ns=2,
    )
    runner._log_pending_action_chunk(switch_timestamp_ns=2_000_000_000)

    aligned_actions = [
        record
        for record in xyz_logger.action_records
        if record["event_type"] == "aligned_actions"
    ]

    assert [record["timestamp_ns"] for record in aligned_actions] == [
        2_000_000_000,
        2_140_000_000,
        2_280_000_000,
    ]
    assert {record["estimated_wall_action_step_interval_ns"] for record in aligned_actions} == {
        140_000_000
    }


def test_visualizer_shows_original_and_aligned_action_series() -> None:
    html_path = (
        Path(__file__).resolve().parents[1]
        / "visualizers"
        / "action_state_xyz_visualizer.html"
    )
    html = html_path.read_text(encoding="utf-8")

    assert "all actions xyzw (original full chunk)" in html
    assert "aligned actions xyzw (post alignment)" in html
    assert "aligned_actions" in html
    assert "joint_positions" in html
    assert "joint_13" in html
    assert "isLegacyRoboTwinJoint14" in html
    assert "original_step_interval_ms" in html
    assert "aligned_step_interval_ms" in html
    assert "pred abs xyzw (per chunk)" not in html
    assert "pred_abs_xyz" not in html
