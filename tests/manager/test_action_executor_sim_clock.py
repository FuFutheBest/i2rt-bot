from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from devices.config import ShmEndpointConfig
from manager.base_policy_runner import BasePolicyRunner


def _runner_with_sim_clock() -> BasePolicyRunner:
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.sim_clock_shm = SimpleNamespace(buf=bytearray(8))
    runner.action_step_interval_ns = 100_000_000
    runner._next_sim_dispatch_time_ns = None
    return runner


def _write_sim_clock(runner: BasePolicyRunner, sim_time_ns: int) -> None:
    runner.sim_clock_shm.buf[:8] = int(sim_time_ns).to_bytes(8, byteorder="little", signed=False)


def test_sim_clock_name_is_derived_from_first_control_endpoint() -> None:
    first_endpoint = ShmEndpointConfig(
        device="RoboTwinSimRobot", device_class_id=3, port="command"
    )
    second_endpoint = ShmEndpointConfig(
        device="OtherRobot", device_class_id=9, port="command"
    )

    assert BasePolicyRunner._derive_sim_clock_shm_name(
        [{"endpoint": first_endpoint}, {"endpoint": second_endpoint}]
    ) == "RoboTwinSimRobot_3_sim_clock"


def test_sim_time_requires_a_structured_control_endpoint() -> None:
    with pytest.raises(ValueError, match="structured"):
        BasePolicyRunner._derive_sim_clock_shm_name([])


def test_sim_time_runner_derives_clock_from_first_control_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("manager.base_policy_runner.signal.signal", lambda *_args: None)
    monkeypatch.setenv("SI_ACTION_STATE_XYZ_LOG_ENABLED", "0")
    monkeypatch.setenv("SI_TIMESTAMP_LOG_ENABLED", "0")
    runner = BasePolicyRunner(
        config={"policy": {"params": {}}},
        robot_control_shms=[
            {
                "endpoint": {
                    "device": "RoboTwinSimRobot",
                    "device_class_id": 3,
                    "port": "command",
                },
                "shape": [14],
            },
            {
                "endpoint": {
                    "device": "OtherRobot",
                    "device_class_id": 9,
                    "port": "command",
                },
                "shape": [7],
            },
        ],
        execution_clock="sim_time",
        signal_files={"policy_runner_reset_command": str(tmp_path / "reset")},
    )
    try:
        assert runner.sim_clock_shm_name == "RoboTwinSimRobot_3_sim_clock"
    finally:
        runner.stop()


def test_sim_clock_shm_is_not_a_runner_configuration_parameter() -> None:
    assert "sim_clock_shm" not in inspect.signature(BasePolicyRunner).parameters


def test_sim_clock_dispatch_advances_on_sim_time_deadlines() -> None:
    runner = _runner_with_sim_clock()

    _write_sim_clock(runner, 1_000_000_000)
    assert runner._sim_dispatch_due() is True

    _write_sim_clock(runner, 1_050_000_000)
    assert runner._sim_dispatch_due() is False

    _write_sim_clock(runner, 1_100_000_000)
    assert runner._sim_dispatch_due() is True

    assert runner._sim_dispatch_due() is False

    _write_sim_clock(runner, 1_200_000_000)
    assert runner._sim_dispatch_due() is True


def test_sim_clock_dispatch_reports_all_deadlines_crossed_by_one_poll() -> None:
    runner = _runner_with_sim_clock()

    _write_sim_clock(runner, 0)
    assert runner._sim_dispatch_due() is True


    _write_sim_clock(runner, 350_000_000)
    assert runner._sim_dispatch_due() is True
    assert runner._sim_dispatch_steps_due == 3
    assert runner._sim_dispatch_skipped_deadlines == 2
    assert runner._sim_dispatch_deadline_ns == 300_000_000
    assert runner._sim_dispatch_lateness_ns == 50_000_000
    assert runner._sim_dispatch_due() is False
    assert runner._next_sim_dispatch_time_ns == 400_000_000


class _FakeChunkManager:
    def __init__(self, *, drop_first_actions: str | int = 0) -> None:
        self.drop_first_actions = drop_first_actions
        self.actions = [
            {"robot": np.asarray([index], dtype=np.float64)}
            for index in range(5)
        ]

    def is_empty(self) -> bool:
        return not self.actions

    def get(self) -> dict[str, np.ndarray] | None:
        if not self.actions:
            return None
        return self.actions.pop(0)


def test_sim_time_catch_up_consumes_old_chunk_to_latest_row() -> None:
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.chunk_manager = _FakeChunkManager()
    runner.chunk_length = 5

    action, consumed, requested, aligned = runner._consume_due_actions(
        3,
        chunk_replaced=False,
    )

    assert action is not None
    np.testing.assert_array_equal(action["robot"], np.asarray([2.0]))
    assert consumed == 3
    assert requested == 3
    assert aligned is False


def test_sim_time_replacement_chunk_is_consumed_once_after_auto_alignment() -> None:
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.chunk_manager = _FakeChunkManager(drop_first_actions="auto")
    runner.chunk_length = 5

    action, consumed, requested, aligned = runner._consume_due_actions(
        3,
        chunk_replaced=True,
    )

    assert action is not None
    np.testing.assert_array_equal(action["robot"], np.asarray([0.0]))
    assert consumed == 1
    assert requested == 1
    assert aligned is True


class _RecordingMetaLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log_meta(self, event_type: str, **fields: Any) -> None:
        self.records.append({"event_type": event_type, **fields})


def test_no_action_dispatch_metrics_keep_dispatch_count_and_reason() -> None:
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.action_state_xyz_logger = _RecordingMetaLogger()
    runner.execution_fps = 15.0
    runner._action_dispatch_count = 2
    runner._first_action_dispatch_wall_timestamp_ns = 1_000_000_000
    runner._last_action_dispatch_wall_timestamp_ns = 1_100_000_000
    runner._first_action_dispatch_sim_timestamp_ns = 0
    runner._last_action_dispatch_sim_timestamp_ns = 100_000_000
    runner._deadline_due_count = 4
    runner._deadline_miss_count = 1

    runner._log_no_action_dispatch_metrics(
        1_200_000_000,
        {"sim_dispatch_steps_due": 2},
        "chunk_exhausted",
    )

    record = runner.action_state_xyz_logger.records[0]
    assert record["event_type"] == "action_dispatch_metrics"
    assert record["dispatch_count"] == 2
    assert record["dispatch_emitted"] is False
    assert record["dispatch_skip_reason"] == "chunk_exhausted"
    assert record["sim_dispatch_steps_due"] == 2
    assert record["deadline_miss_count"] == 1
