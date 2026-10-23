"""Hardware-free fault injection for authoritative UI state synchronization."""

from __future__ import annotations

import copy
import threading
from typing import Any

import pytest

from incubator.control_panel import state_sync as state_sync_module
from incubator.control_panel.state_sync import StateSyncBroker, StateTransitionConflict


def _device(
    *,
    status: str = "stopped",
    running: bool = False,
    ready: bool = False,
    desired: str = "stopped",
    connected: bool = False,
    fresh: bool = False,
    fail_closed: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "key": "RizonRobot:2",
        "class": "RizonRobot",
        "id": 2,
        "risk": "motion",
        "status": status,
        "running": running,
        "ready": ready,
        "desired_state": desired,
        "pid": 123 if running else None,
        "restart_count": 0,
        "last_error": reason,
        "shm": {
            "connected": connected,
            "fresh": fresh,
            "error": None,
        },
        "safety": {
            "source": "fixture",
            "known": not fail_closed,
            "operational": True if not fail_closed else None,
            "fault": False if not fail_closed else None,
            "estop": False if not fail_closed else None,
            "fail_closed": fail_closed,
            "reason": reason,
        },
    }


def _snapshot(device: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "revision": 1,
        "selected_config": "cfg",
        "active_config": "cfg" if device and device["running"] else None,
        "devices": [device] if device is not None else [],
        "workflow": {
            "runtime": {
                "status": "stopped",
                "desired_state": "stopped",
                "active": False,
                "ready": False,
                "error": None,
            },
            "inference": {
                "status": "stopped",
                "desired_state": "stopped",
                "running": False,
                "error": None,
            },
        },
        "recording": {"status": "stopped", "active": False},
        "grippers": {"status": "idle", "sides": {}},
    }


def _pending_home_snapshot(
    device: dict[str, Any],
    *,
    action: str = "reset",
    created_at: str = "2026-07-13T00:00:00+00:00",
) -> dict[str, Any]:
    snapshot = _snapshot(device)
    command = {
        "command_id": "home-command",
        "action": action,
        "status": "pending",
        "created_at": created_at,
    }
    if action == "reset":
        snapshot["workflow"]["inference"]["reset"] = {
            "pending_command": command,
        }
    else:
        snapshot["workflow"]["recording"] = {
            "pending_command": command,
        }
    return snapshot


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_start_ack_is_not_success_when_process_dies() -> None:
    raw = _snapshot(_device(status="starting", running=True, desired="running"))
    broker = StateSyncBroker(lambda: copy.deepcopy(raw))
    command = broker.begin(
        "device:RizonRobot:2",
        "start",
        "running",
        timeout_ms=5_000,
        token="process-failure-token",
    )
    broker.acknowledge(command["token"])

    raw["devices"][0].update(
        {
            "status": "failed",
            "running": False,
            "ready": False,
            "last_error": "worker exited with code 9",
        }
    )
    observed = broker.sample_now()

    transition = observed["state_sync"]["transitions"]["device:RizonRobot:2"]
    assert transition["status"] == "failed"
    assert "code 9" in transition["reason"]


def test_expected_rizon_initialization_does_not_lock_parallel_starts() -> None:
    raw = _snapshot(
        _device(
            status="starting",
            running=True,
            ready=False,
            desired="running",
            connected=False,
            fresh=False,
            fail_closed=True,
            reason="Waiting for the first Rizon health heartbeat",
        )
    )
    raw["devices"][0]["safety"]["initializing"] = True
    broker = StateSyncBroker(lambda: copy.deepcopy(raw))
    broker.begin(
        "device:RizonRobot:2",
        "start",
        "running",
        timeout_ms=5_000,
        token="expected-initialization-token",
    )

    observed = broker.sample_now()

    device = observed["devices"][0]
    transition = observed["state_sync"]["transitions"]["device:RizonRobot:2"]
    assert device["safety"]["fail_closed"] is True
    assert device["safety"]["initializing"] is True
    assert observed["safety"]["fail_closed"] is False
    assert transition["status"] == "pending"


@pytest.mark.parametrize(
    ("status", "connected", "reason"),
    [
        ("orphaned", True, "owned process disappeared"),
        ("degraded", False, "device SHM disconnected"),
    ],
)
def test_external_kill_and_shm_disconnect_fail_closed(
    status: str, connected: bool, reason: str
) -> None:
    raw = _snapshot(
        _device(
            status=status,
            running=False,
            desired="running",
            connected=connected,
            fresh=False,
            fail_closed=True,
            reason=reason,
        )
    )
    broker = StateSyncBroker(lambda: copy.deepcopy(raw))

    observed = broker.sample_now()

    device = observed["devices"][0]
    assert device["observed"]["ready"] is False
    assert device["safety"]["fail_closed"] is True
    assert observed["safety"]["fail_closed"] is True


def test_external_estop_prevents_running_transition_completion() -> None:
    raw = _snapshot(
        _device(
            status="degraded",
            running=True,
            ready=False,
            desired="running",
            connected=True,
            fresh=True,
            fail_closed=True,
            reason="Emergency stop is active",
        )
    )
    raw["devices"][0]["safety"]["estop"] = True
    broker = StateSyncBroker(lambda: copy.deepcopy(raw))
    broker.begin(
        "device:RizonRobot:2",
        "start",
        "running",
        timeout_ms=5_000,
        token="external-estop-token",
    )

    observed = broker.sample_now()

    assert observed["safety"]["estop"] == {
        "active": True,
        "known": True,
        "source": "device_health_aggregate",
    }
    assert (
        observed["state_sync"]["transitions"]["device:RizonRobot:2"]["status"]
        == "failed"
    )


def test_stuck_stop_times_out_without_claiming_stopped() -> None:
    clock = _Clock()
    raw = _snapshot(
        _device(
            status="stopping",
            running=True,
            desired="stopped",
            connected=True,
            fresh=True,
        )
    )
    broker = StateSyncBroker(
        lambda: copy.deepcopy(raw),
        monotonic=clock,
        utc_now=lambda: "2026-07-13T00:00:00+00:00",
        epoch_factory=lambda: "epoch",
    )
    broker.begin(
        "device:RizonRobot:2",
        "stop",
        "stopped",
        timeout_ms=100,
        token="stuck-stop-token",
    )
    broker.sample_now()
    clock.advance(0.2)

    observed = broker.current()

    transition = observed["state_sync"]["transitions"]["device:RizonRobot:2"]
    assert transition["status"] == "timed_out"
    assert observed["devices"][0]["observed"]["running"] is True


def test_failed_state_can_recover_on_a_new_observation() -> None:
    raw = _snapshot(
        _device(
            status="degraded",
            running=True,
            desired="running",
            connected=True,
            fresh=False,
            fail_closed=True,
            reason="temporary heartbeat loss",
        )
    )
    broker = StateSyncBroker(lambda: copy.deepcopy(raw))
    first = broker.sample_now()
    first_revision = first["state_sync"]["observed_revision"]

    raw["devices"][0] = _device(
        status="running",
        running=True,
        ready=True,
        desired="running",
        connected=True,
        fresh=True,
    )
    recovered = broker.sample_now()

    assert recovered["devices"][0]["ready"] is True
    assert recovered["safety"]["fail_closed"] is False
    assert recovered["state_sync"]["observed_revision"] > first_revision


@pytest.mark.parametrize("action", ["reset", "episode_accept", "episode_discard"])
def test_pending_home_transition_suppresses_only_transient_heartbeat_failure(
    action: str,
) -> None:
    device = _device(
        status="degraded",
        running=True,
        ready=False,
        desired="running",
        connected=True,
        fresh=False,
        fail_closed=True,
        reason="Rizon health heartbeat is stale",
    )
    raw = _pending_home_snapshot(device, action=action)
    broker = StateSyncBroker(
        lambda: copy.deepcopy(raw),
        utc_now=lambda: "2026-07-13T00:00:20+00:00",
    )

    observed = broker.sample_now()

    result = observed["devices"][0]
    assert result["expected_transition"]["kind"] == "home_reset"
    assert result["expected_transition"]["action"] == action
    assert result["safety"]["fail_closed"] is False
    assert result["safety"]["transient_fail_closed_suppressed"] is True
    assert result["safety"]["raw_fail_closed"] is True
    assert result["safety"]["raw_reason"] == "Rizon health heartbeat is stale"
    assert result["last_error"] == "Rizon health heartbeat is stale"
    assert observed["safety"]["fail_closed"] is False


@pytest.mark.parametrize(
    ("safety_patch", "running", "connected"),
    [
        ({"fault": True}, True, True),
        ({"estop": True}, True, True),
        ({"operational": False}, True, True),
        ({"reduced": True}, True, True),
        ({}, False, True),
        ({}, True, False),
    ],
)
def test_pending_home_transition_never_masks_explicit_or_disconnected_failures(
    safety_patch: dict[str, Any],
    running: bool,
    connected: bool,
) -> None:
    device = _device(
        status="degraded",
        running=running,
        ready=False,
        desired="running",
        connected=connected,
        fresh=False,
        fail_closed=True,
        reason="Rizon health heartbeat is stale",
    )
    device["safety"].update(safety_patch)
    raw = _pending_home_snapshot(device)
    broker = StateSyncBroker(
        lambda: copy.deepcopy(raw),
        utc_now=lambda: "2026-07-13T00:00:20+00:00",
    )

    observed = broker.sample_now()

    result = observed["devices"][0]
    assert "expected_transition" not in result
    assert result["safety"]["fail_closed"] is True
    assert observed["safety"]["fail_closed"] is True


def test_expired_home_transition_keeps_heartbeat_failure_fail_closed() -> None:
    device = _device(
        status="degraded",
        running=True,
        ready=False,
        desired="running",
        connected=True,
        fresh=False,
        fail_closed=True,
        reason="Rizon health heartbeat is stale",
    )
    raw = _pending_home_snapshot(
        device,
        created_at="2026-07-12T23:58:00+00:00",
    )
    broker = StateSyncBroker(
        lambda: copy.deepcopy(raw),
        utc_now=lambda: "2026-07-13T00:00:00+00:00",
    )

    observed = broker.sample_now()

    assert "expected_transition" not in observed["devices"][0]
    assert observed["safety"]["fail_closed"] is True


def test_stale_cached_snapshot_is_marked_non_authoritative() -> None:
    clock = _Clock()
    broker = StateSyncBroker(
        lambda: _snapshot(),
        monotonic=clock,
        utc_now=lambda: "2026-07-13T00:00:00+00:00",
        epoch_factory=lambda: "epoch",
    )
    broker.sample_now()
    clock.advance(10.0)

    stale = broker.current()

    assert stale["state_sync"]["stale"] is True
    assert stale["state_sync"]["authoritative"] is False


def test_multiple_cold_tabs_share_one_probe() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()

    def source() -> dict[str, Any]:
        nonlocal calls
        with lock:
            calls += 1
        entered.set()
        assert release.wait(2.0)
        return _snapshot()

    broker = StateSyncBroker(source)
    results: list[dict[str, Any]] = []
    threads = [
        threading.Thread(target=lambda: results.append(broker.sample_if_due()))
        for _ in range(8)
    ]
    threads[0].start()
    assert entered.wait(1.0)
    for thread in threads[1:]:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(2.0)
        assert not thread.is_alive()

    assert calls == 1
    assert len(results) == 8


def test_other_tab_cannot_overlap_resource_transition() -> None:
    broker = StateSyncBroker(lambda: _snapshot())
    broker.begin(
        "workflow:runtime",
        "start",
        "running",
        timeout_ms=5_000,
        token="first-tab-token",
    )

    with pytest.raises(StateTransitionConflict):
        broker.begin(
            "workflow:runtime",
            "start",
            "running",
            timeout_ms=5_000,
            token="second-tab-token",
        )


def test_same_tab_retry_is_idempotent() -> None:
    broker = StateSyncBroker(lambda: _snapshot())
    first = broker.begin(
        "workflow:runtime",
        "start",
        "running",
        timeout_ms=5_000,
        token="same-tab-token",
    )
    duplicate = broker.begin(
        "workflow:runtime",
        "start",
        "running",
        timeout_ms=5_000,
        token="same-tab-token",
    )

    assert first["duplicate"] is False
    assert duplicate["duplicate"] is True


def test_safety_stop_supersedes_pending_start_with_a_new_token() -> None:
    broker = StateSyncBroker(lambda: _snapshot())
    broker.begin(
        "device:RizonRobot:2",
        "start",
        "running",
        timeout_ms=5_000,
        token="pending-start-token",
    )

    stop = broker.begin(
        "device:RizonRobot:2",
        "stop",
        "stopped",
        timeout_ms=5_000,
        token="new-safety-stop-token",
    )

    assert stop["duplicate"] is False
    assert stop["action"] == "stop"
    assert stop["desired"] == "stopped"


def test_source_recovery_broadcasts_even_when_payload_is_identical() -> None:
    calls = 0

    def source() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("temporary probe failure")
        return _snapshot()

    broker = StateSyncBroker(source)
    initial = broker.sample_now()
    initial_event = initial["state_sync"]["event_revision"]
    initial_observed = initial["state_sync"]["observed_revision"]
    failed = broker.sample_now()
    assert failed["state_sync"]["authoritative"] is False

    recovered = broker.sample_now()

    assert recovered["state_sync"]["authoritative"] is True
    assert recovered["state_sync"]["event_revision"] > initial_event
    assert recovered["state_sync"]["observed_revision"] > initial_observed


def test_live_token_history_overflow_never_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state_sync_module, "MAX_TOKEN_HISTORY", 2)
    broker = StateSyncBroker(lambda: _snapshot())

    for index in range(3):
        broker.begin(
            f"device:Camera:{index}",
            "start",
            "running",
            timeout_ms=5_000,
            token=f"history-live-token-{index}",
        )

    assert len(broker._token_history) == 3
