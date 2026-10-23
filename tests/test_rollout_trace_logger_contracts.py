#!/usr/bin/env python3
"""Focused contract tests for rollout trace JSONL loggers.

Author: SuperInference Contributors
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from consumer.inference.common.rollout_controller import RolloutController
from manager.base_policy_runner import BasePolicyRunner
from utils.action_state_xyz_logger import (
    ActionStateXYZLogger,
    _action_arm_records,
    _state_arm_records,
)
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    pack_buffer_header,
)
from utils.timestamp_event_logger import TimestampEventLogger


def test_rollout_saver_resolution_is_package_root_relative(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)

    saver_script = Path(RolloutController._saver_script())

    assert saver_script.is_absolute()
    assert saver_script.is_file()
    assert saver_script.name == "async_infer_data_saver_NEDF3.py"
    assert saver_script.parent.name == "io"
    assert saver_script.parent.parent.name == "consumer"


def test_rollout_saver_uses_unified_python312_nedf_sdk() -> None:
    saver = Path(RolloutController._saver_script()).read_text(encoding="utf-8")
    r1_saver = Path("incubator/r1/async_data_saver_NEDF3_r1.py").read_text(
        encoding="utf-8"
    )

    assert "nmx_nedf_api" not in saver
    assert "from dm3_core.nedf import" in r1_saver
    assert "from nmx_nedf_api import" not in r1_saver


def test_action_state_base_record_exposes_timestamp_aliases() -> None:
    logger = ActionStateXYZLogger("contract_test", enabled=False)

    record = logger._base_record("sample", 1_700_000_000_123_456_789)

    assert record["timestamp_ns"] == record["event_ns"]
    assert record["timestamp_time_utc"] == record["event_time_utc"]
    assert record["source"] == "contract_test"


def test_timestamp_event_jsonl_exposes_timestamp_aliases(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    event_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("utils.timestamp_event_logger.time.time_ns", lambda: event_ns)
    logger = TimestampEventLogger("contract_test", log_dir=str(tmp_path))

    logger.log_event("sample", observation_timestamp_ns=123)

    assert logger.path is not None
    records = [
        json.loads(line)
        for line in logger.path.read_text(encoding="utf-8").splitlines()
    ]
    sample = records[-1]
    assert sample["event_ns"] == sample["timestamp_ns"] == event_ns
    assert sample["event_time_utc"] == sample["timestamp_time_utc"]
    assert sample["observation_timestamp_ns"] == 123


def test_dual_arm_action_and_state_vector_layouts() -> None:
    action = np.arange(16, dtype=np.float64)
    state = np.arange(44, dtype=np.float64)

    action_records = list(_action_arm_records(action))
    state_records = list(_state_arm_records(state))

    assert [(record["arm"], record["xyz"]) for record in action_records] == [
        ("left", [0.0, 1.0, 2.0]),
        ("right", [8.0, 9.0, 10.0]),
    ]
    assert [record["gripper_width"] for record in action_records] == [7.0, 15.0]
    assert [(record["arm"], record["xyz"]) for record in state_records] == [
        ("left", [14.0, 15.0, 16.0]),
        ("right", [36.0, 37.0, 38.0]),
    ]
    assert [record["gripper_width"] for record in state_records] == [21.0, 43.0]


def test_xyz_width_aliases_are_exact_and_omitted_without_width_channel() -> None:
    eight_dimensional = list(_action_arm_records(np.arange(8, dtype=np.float64)))[0]
    seven_dimensional = list(_action_arm_records(np.arange(7, dtype=np.float64)))[0]

    assert (
        eight_dimensional["w"]
        == eight_dimensional["width"]
        == eight_dimensional["gripper_width"]
        == 7.0
    )
    assert {"w", "width", "gripper_width"}.isdisjoint(seven_dimensional)


def test_joint14_records_keep_joint_values_out_of_xyzw_trace(tmp_path: Path) -> None:
    logger = ActionStateXYZLogger("joint14_contract", log_dir=str(tmp_path))
    values = np.arange(14, dtype=np.float64)

    logger.log_action(
        device_name="RoboTwinSimRobot_0",
        action=values,
        timestamp_ns=100,
        event_type="all_actions",
        vector_layout="joint14",
    )
    logger.log_state(
        device_name="RoboTwinSimRobot_0",
        state=values,
        timestamp_ns=101,
        event_type="state_xyz",
        vector_layout="joint14",
    )
    assert logger.path is not None
    path = logger.path
    logger.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    action = next(record for record in records if record["event_type"] == "all_actions")
    state = next(record for record in records if record["event_type"] == "state_xyz")
    for record, kind in ((action, "action"), (state, "state")):
        assert record["vector_layout"] == "joint14"
        assert record["vector_kind"] == kind
        assert record["joint_positions"] == list(values)
        assert len(record["joint_names"]) == 14
        assert {"x", "y", "z", "w"}.isdisjoint(record)


def test_joint14_logger_skips_invalid_vector_without_raising(tmp_path: Path) -> None:
    logger = ActionStateXYZLogger("joint14_invalid", log_dir=str(tmp_path))

    logger.log_action(
        device_name="RoboTwinSimRobot_0",
        action=np.full(14, np.nan),
        timestamp_ns=100,
        vector_layout="joint14",
    )
    assert logger.path is not None
    path = logger.path
    logger.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event_type"] for record in records] == ["logger_started"]


def test_action_executor_dispatch_exposes_executed_aliases(
    monkeypatch: Any,
) -> None:
    class _SharedMemory:
        def __init__(self, size: int) -> None:
            self.buf = bytearray(size)

    class _EventLogger:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def log_event(self, event_type: str, **fields: Any) -> None:
            self.records.append({"event_type": event_type, **fields})

    class _XYZLogger:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def log_action(self, **fields: Any) -> None:
            self.records.append(fields)

    action = np.arange(8, dtype=np.float64)
    frame_size = FRAME_HEADER_SIZE + action.nbytes
    shm = _SharedMemory(DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_size)
    shm.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = (
        pack_buffer_header(1, 0, 0)
    )
    timestamp_logger = _EventLogger()
    xyz_logger = _XYZLogger()
    runner = BasePolicyRunner.__new__(BasePolicyRunner)
    runner.timestamp_logger = timestamp_logger
    runner.action_state_xyz_logger = xyz_logger
    event_ns = 1_700_000_000_123_456_789
    monkeypatch.setattr("manager.base_policy_runner.time.time_ns", lambda: event_ns)

    result = runner._write_action_to_robot(
        {
            "shm": shm,
            "shm_name": "robot_control",
            "frame_size": frame_size,
            "action_dim": 8,
            "target_shape": (8,),
            "target_dtype": np.dtype("float64"),
            "payload_size": action.nbytes,
            "device_info": {"device_key": "robot"},
        },
        action,
    )

    assert result == event_ns
    for record in (timestamp_logger.records[0], xyz_logger.records[0]):
        assert record["dispatch_timestamp_ns"] == event_ns
        assert record["executed_action_timestamp_ns"] == event_ns
        assert record["action_stage"] == "executed"
    assert timestamp_logger.records[0]["action_timestamp_ns"] == event_ns
