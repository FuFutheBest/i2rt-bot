from __future__ import annotations

import os
from pathlib import Path

import pytest

from incubator.control_panel.inference_control import (
    InferenceControlConflictError,
    InferenceControlConsumer,
    InferenceControlCoordinator,
)


def _home_target() -> dict:
    return {
        "schema_version": 1,
        "source": "home_pose_candidates",
        "config_name": "fixture",
        "config_hash": "a" * 64,
        "arms": {
            "left": {
                "candidate_id": "left-01",
                "joints": [0.1] * 7,
                "collision_checked": False,
            },
            "right": {
                "candidate_id": "right-01",
                "joints": [-0.1] * 7,
                "collision_checked": False,
            },
        },
    }


def test_reset_is_pending_until_inference_consumer_applies_it(
    tmp_path: Path,
) -> None:
    coordinator = InferenceControlCoordinator(tmp_path / "runs")
    control_dir = coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    target = _home_target()

    submitted = coordinator.submit_reset(
        idempotency_key="reset-0001",
        home_target=target,
    )
    assert submitted["status"] == "pending"
    assert coordinator.state()["pending_command"]["command_id"] == submitted["command_id"]

    received = []
    consumer = InferenceControlConsumer(
        control_dir,
        str(coordinator.run_id),
        owner_pid=os.getpid(),
    )
    assert consumer.process_commands(
        received.append,
        state_provider=lambda: {"inference_paused": False},
    ) == 1
    assert received == [
        {
            "schema_version": 1,
            "run_id": coordinator.run_id,
            "command_id": submitted["command_id"],
            "idempotency_key": "reset-0001",
            "action": "reset",
            "created_at": received[0]["created_at"],
            "home_target": target,
        }
    ]

    state = coordinator.state()
    assert state["pending_command"] is None
    assert state["last_command"]["status"] == "applied"
    assert state["paused"] is False
    assert not list((control_dir / "commands").glob("*.json"))


def test_reset_failure_is_acked_and_does_not_stay_pending(tmp_path: Path) -> None:
    coordinator = InferenceControlCoordinator(tmp_path / "runs")
    control_dir = coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    coordinator.submit_reset(idempotency_key="reset-0002")
    consumer = InferenceControlConsumer(
        control_dir,
        str(coordinator.run_id),
        owner_pid=os.getpid(),
    )

    def fail(_command: dict) -> None:
        raise RuntimeError("home failed")

    assert consumer.process_commands(
        fail,
        state_provider=lambda: {"inference_paused": True},
    ) == 1
    state = coordinator.state()
    assert state["pending_command"] is None
    assert state["last_command"]["status"] == "failed"
    assert state["last_command"]["error"]["code"] == "handler_error"
    assert state["paused"] is True


def test_only_one_reset_can_be_pending(tmp_path: Path) -> None:
    coordinator = InferenceControlCoordinator(tmp_path / "runs")
    coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    coordinator.submit_reset(idempotency_key="reset-0003")

    with pytest.raises(InferenceControlConflictError, match="already pending"):
        coordinator.submit_reset(idempotency_key="reset-0004")


def test_pause_and_resume_publish_authoritative_paused_state(tmp_path: Path) -> None:
    coordinator = InferenceControlCoordinator(tmp_path / "runs")
    control_dir = coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    paused = False
    consumer = InferenceControlConsumer(
        control_dir,
        str(coordinator.run_id),
        owner_pid=os.getpid(),
    )

    pause = coordinator.submit_pause(idempotency_key="pause-0001")

    def apply(command: dict) -> None:
        nonlocal paused
        paused = command["action"] == "pause"

    assert consumer.process_commands(
        apply,
        state_provider=lambda: {"inference_paused": paused},
    ) == 1
    state = coordinator.state()
    assert state["last_command"]["command_id"] == pause["command_id"]
    assert state["last_command"]["action"] == "pause"
    assert state["paused"] is True

    resume = coordinator.submit_resume(idempotency_key="resume-0001")
    assert consumer.process_commands(
        apply,
        state_provider=lambda: {"inference_paused": paused},
    ) == 1
    state = coordinator.state()
    assert state["last_command"]["command_id"] == resume["command_id"]
    assert state["last_command"]["action"] == "resume"
    assert state["paused"] is False


def test_reset_completion_can_publish_resume_state_after_home_signal(
    tmp_path: Path,
) -> None:
    coordinator = InferenceControlCoordinator(tmp_path / "runs")
    control_dir = coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    consumer = InferenceControlConsumer(
        control_dir,
        str(coordinator.run_id),
        owner_pid=os.getpid(),
    )
    submitted = coordinator.submit_reset(idempotency_key="reset-complete-0001")
    received = []
    assert consumer.process_commands(
        received.append,
        state_provider=lambda: {"inference_paused": True},
    ) == 1
    assert coordinator.state()["paused"] is True
    assert coordinator.state()["reset_pending"] is False

    consumer.update_applied_state(
        submitted["command_id"],
        inference_paused=False,
    )
    assert coordinator.state()["paused"] is False
    assert coordinator.state()["reset_pending"] is False


def test_reset_ack_can_remain_pending_until_home_completes(tmp_path: Path) -> None:
    coordinator = InferenceControlCoordinator(tmp_path / "runs")
    control_dir = coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    consumer = InferenceControlConsumer(
        control_dir,
        str(coordinator.run_id),
        owner_pid=os.getpid(),
    )
    submitted = coordinator.submit_reset(idempotency_key="reset-home-pending-0001")
    assert consumer.process_commands(
        lambda _command: None,
        state_provider=lambda: {
            "inference_paused": True,
            "reset_pending": True,
        },
    ) == 1
    state = coordinator.state()
    assert state["reset_pending"] is True
    assert state["last_command"]["command_id"] == submitted["command_id"]

    consumer.update_applied_state(
        submitted["command_id"],
        inference_paused=False,
        reset_pending=False,
    )
    assert coordinator.state()["reset_pending"] is False


def test_idempotency_key_cannot_be_reused_for_another_action(tmp_path: Path) -> None:
    coordinator = InferenceControlCoordinator(tmp_path / "runs")
    coordinator.prepare_run()
    coordinator.bind_owner(os.getpid())
    coordinator.submit_pause(idempotency_key="same-key")

    with pytest.raises(InferenceControlConflictError, match="another inference"):
        coordinator.submit_resume(idempotency_key="same-key")
