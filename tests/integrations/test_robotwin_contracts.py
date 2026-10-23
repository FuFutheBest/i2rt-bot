"""Hardware-free contracts for the RoboTwin asynchronous bridge."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from integrations.robotwin.contracts import EEF16_FIELDS, JOINT14_FIELDS, validate_action
from integrations.robotwin.setpoint import AsyncEefResolver, LatestSetpoint
from integrations.robotwin.trace import RoboTwinTrace


def test_action_layouts_are_fixed_and_validate_exact_dimensions() -> None:
    assert len(JOINT14_FIELDS) == 14
    assert len(EEF16_FIELDS) == 16
    np.testing.assert_array_equal(validate_action(np.zeros(14), "joint14"), np.zeros(14))
    np.testing.assert_array_equal(validate_action(np.zeros(16), "eef16"), np.zeros(16))
    with pytest.raises(ValueError, match="must have 14"):
        validate_action(np.zeros(13), "joint14")


def test_latest_setpoint_replaces_target_without_waiting_for_arrival() -> None:
    setpoints = LatestSetpoint("joint14")
    first = setpoints.accept(np.zeros(14), 100)
    second = setpoints.accept(np.ones(14), 101)

    assert first is not None
    assert second is not None
    assert setpoints.accept(np.full(14, 2.0), 101) is None
    assert setpoints.latest is second
    np.testing.assert_allclose(second.action, np.ones(14))


def test_eef_resolution_stays_off_the_physics_callsite() -> None:
    def resolver(target_eef16: np.ndarray, current_joint14: np.ndarray) -> np.ndarray:
        assert target_eef16.shape == (16,)
        return current_joint14 + 0.25

    setpoint = LatestSetpoint("eef16").accept(np.zeros(16), 100)
    assert setpoint is not None
    worker = AsyncEefResolver(resolver)
    try:
        worker.submit(setpoint, np.zeros(14))
        result = None
        deadline = time.monotonic() + 1.0
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(0.001)
        assert result is not None
        sequence, joint_target = result
        assert sequence == setpoint.sequence
        np.testing.assert_allclose(joint_target, np.full(14, 0.25))
    finally:
        worker.close()


def test_eef_resolution_coalesces_intermediate_setpoints() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[int] = []

    def resolver(target_eef16: np.ndarray, _current_joint14: np.ndarray) -> np.ndarray:
        value = int(target_eef16[0])
        calls.append(value)
        if value == 1:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        return np.full(14, value, dtype=np.float64)

    source = LatestSetpoint("eef16")
    first = source.accept(np.full(16, 1.0), 100)
    second = source.accept(np.full(16, 2.0), 101)
    third = source.accept(np.full(16, 3.0), 102)
    assert first is not None and second is not None and third is not None
    worker = AsyncEefResolver(resolver)
    try:
        worker.submit(first, np.zeros(14))
        assert first_started.wait(timeout=1.0)
        worker.submit(second, np.zeros(14))
        worker.submit(third, np.zeros(14))
        release_first.set()

        result = None
        deadline = time.monotonic() + 1.0
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(0.001)
        assert result is not None
        sequence, joint_target = result
        assert sequence == third.sequence
        np.testing.assert_allclose(joint_target, np.full(14, 3.0))
        assert calls == [1, 3]
    finally:
        worker.close()


def test_trace_keeps_full_joint_vectors_and_the_control_join_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SI_ACTION_STATE_XYZ_RUN_ID", "robotwin-contract")
    trace = RoboTwinTrace("robotwin_contract", log_dir=str(tmp_path))
    setpoint = LatestSetpoint("joint14").accept(np.arange(14, dtype=np.float64), 123)
    assert setpoint is not None
    trace.control_accepted(setpoint, timestamp_ns=200, sim_step=1, sim_time_ns=4_000_000)
    trace.action_applied(
        setpoint,
        target_joint14=setpoint.action,
        timestamp_ns=300,
        sim_step=2,
        sim_time_ns=8_000_000,
    )
    trace.measured_state(
        measured_qpos=np.arange(14, dtype=np.float64),
        measured_qvel=np.zeros(14),
        target_qpos=setpoint.action,
        last_control_write_timestamp_ns=123,
        last_sim_apply_timestamp_ns=300,
        timestamp_ns=400,
        sim_step=3,
        sim_time_ns=12_000_000,
    )
    trace_path = Path(trace.path or "")
    trace.close()

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    applied = next(record for record in records if record["event_type"] == "sim_action_applied")
    measured = next(record for record in records if record["event_type"] == "sim_measured_state")
    assert applied["control_write_timestamp_ns"] == 123
    assert applied["executed_action_timestamp_ns"] == 300
    assert applied["target_qpos"] == list(np.arange(14, dtype=float))
    assert measured["measured_qpos"] == list(np.arange(14, dtype=float))
    assert measured["sim_step"] == 3
