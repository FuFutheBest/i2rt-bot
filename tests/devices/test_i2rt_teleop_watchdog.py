"""Deterministic tests for the worker-local I2RT watchdog."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from portal import packlib

from devices.robots.i2rt_teleop import coordinator, worker
from devices.robots.i2rt_teleop.config import ControlConfig, TeleoperationConfig
from devices.robots.i2rt_teleop.coordinator import (
    BimanualCoordinator,
    RpcEndpointClient,
)
from devices.robots.i2rt_teleop.runtime import (
    ACTIVE,
    ALIGNING,
    FAULT,
    SAFE_HOLD,
    WorkerControlRuntime,
)


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1e9)


class FakeEndpoint:
    def __init__(
        self, *, role: str = "follower", follow_gripper: bool = True
    ) -> None:
        self.role = role
        self.follow_gripper = follow_gripper
        self.state = np.arange(7, dtype=np.float64)
        self.buttons = np.zeros(2)
        self.commands: list[np.ndarray] = []
        self.safe_states: list[np.ndarray] = []
        self.position_holds: list[np.ndarray] = []
        self.feedback_updates: list[tuple[bool, np.ndarray, np.ndarray]] = []
        self.read_error: Exception | None = None

    def get_info(self):
        if self.read_error is not None:
            raise self.read_error
        if self.role == "leader":
            return self.state.copy(), self.buttons.copy()
        return self.state.copy()

    def command_joint_pos(self, command) -> None:
        value = np.asarray(command, dtype=np.float64).copy()
        self.commands.append(value)
        if self.role == "follower":
            self.state[:6] = value[:6]
            if self.follow_gripper:
                self.state[-1] = value[-1]

    def enter_safe_state(self, state) -> None:
        value = np.asarray(state, dtype=np.float64).copy()
        self.safe_states.append(value)

    def enter_position_hold(self, state) -> None:
        value = np.asarray(state, dtype=np.float64).copy()
        self.position_holds.append(value)

    def set_feedback_enabled(self, enabled, kp, kd) -> None:
        self.feedback_updates.append(
            (enabled, np.asarray(kp).copy(), np.asarray(kd).copy())
        )


class BlockingCommandEndpoint(FakeEndpoint):
    """Fake endpoint whose command call can emulate a stuck hardware API."""

    def __init__(self) -> None:
        super().__init__()
        self.command_started = threading.Event()
        self.release_command = threading.Event()

    def command_joint_pos(self, command) -> None:
        self.command_started.set()
        self.release_command.wait(timeout=2.0)
        super().command_joint_pos(command)


class BlockingStartupEndpoint(FakeEndpoint):
    """Fake endpoint whose first feedback read waits for explicit release."""

    def __init__(self) -> None:
        super().__init__()
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def get_info(self):
        self.read_started.set()
        self.release_read.wait(timeout=2.0)
        return super().get_info()


def _runtime(
    role: str = "follower",
    *,
    leader_feedback: bool = True,
) -> tuple[WorkerControlRuntime, FakeEndpoint, FakeClock]:
    endpoint = FakeEndpoint(role=role)
    clock = FakeClock()
    control = replace(
        ControlConfig(),
        command_frame_max_age_s=0.05,
        command_lease_timeout_s=0.10,
        feedback_timeout_s=0.05,
        following_error_fault_threshold=10.0,
        leader_feedback=leader_feedback,
    )
    runtime = WorkerControlRuntime(
        endpoint,
        role=role,
        side="left",
        control=control,
        clock_ns=clock,
    )
    runtime.run_once()
    return runtime, endpoint, clock


def _frame(clock: FakeClock, sequence: int, phase: str, size: int = 7):
    return {
        "session_id": "session",
        "sequence": sequence,
        "issued_at_ns": clock(),
        "phase": phase,
        "joint_position": np.linspace(0.0, 0.6, size),
    }


def _portal_roundtrip(value):
    wire = b"".join(bytes(part) for part in packlib.pack(value))
    return packlib.unpack(wire)


class _RpcFuture:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def result(self, timeout: float):
        raise self.error


class _PortalClient:
    def __init__(self, _address: str, error: Exception) -> None:
        self.error = error
        self.close_timeout = None

    def __getattr__(self, _method: str):
        return lambda *_args: _RpcFuture(self.error)

    def close(self, timeout: float) -> None:
        self.close_timeout = timeout


def test_runtime_starts_by_holding_observed_follower_position() -> None:
    runtime, endpoint, _ = _runtime()

    assert runtime.get_status()["phase"] == SAFE_HOLD
    np.testing.assert_allclose(endpoint.safe_states, [np.arange(7)])


def test_runtime_start_can_be_cancelled_while_hardware_read_blocks() -> None:
    endpoint = BlockingStartupEndpoint()
    runtime = WorkerControlRuntime(
        endpoint,
        role="follower",
        side="left",
        control=ControlConfig(),
    )
    cancel_event = threading.Event()
    result: list[bool] = []
    start_thread = threading.Thread(
        target=lambda: result.append(
            runtime.start(timeout_s=1.0, cancel_event=cancel_event)
        )
    )
    start_thread.start()
    assert endpoint.read_started.wait(timeout=1.0)

    cancel_event.set()
    runtime.request_stop()
    start_thread.join(timeout=0.2)

    assert not start_thread.is_alive()
    assert result == [False]
    endpoint.release_read.set()
    assert runtime.close()


def test_worker_signal_during_endpoint_construction_stops_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[int, Any] = {}
    endpoint_closed = False

    class Endpoint:
        def close(self) -> None:
            nonlocal endpoint_closed
            endpoint_closed = True

    endpoint = Endpoint()
    config = SimpleNamespace(
        control=ControlConfig(),
        arms={
            "left": SimpleNamespace(
                leader=SimpleNamespace(can_channel="can0", rpc_port=1234)
            )
        },
    )

    monkeypatch.setattr(worker, "load_teleoperation_config", lambda _path: config)
    monkeypatch.setattr(
        worker.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    def create_endpoint(_channel: str) -> Endpoint:
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return endpoint

    monkeypatch.setattr(worker, "create_leader_endpoint", create_endpoint)
    monkeypatch.setattr(
        worker,
        "WorkerControlRuntime",
        lambda *_args, **_kwargs: pytest.fail("runtime must not be constructed"),
    )

    worker.serve_worker(Path("teleop.yaml"), "left", "leader")

    assert endpoint_closed


def test_worker_signal_while_serving_preserves_safe_close_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[int, Any] = {}
    events: list[str] = []

    class Endpoint:
        def close(self) -> None:
            events.append("endpoint.close")

    class Runtime:
        stop_requests = 0

        def start(self, _timeout_s: float, *, cancel_event) -> bool:
            assert not cancel_event.is_set()
            return True

        def request_stop(self) -> None:
            self.stop_requests += 1

        def close(self) -> bool:
            events.append("runtime.close")
            return True

        def __getattr__(self, _name: str):
            return lambda *_args, **_kwargs: None

    class Server:
        def __init__(self, _port: int) -> None:
            self.running = False

        def bind(self, _name: str, _method) -> None:
            pass

        def start(self, *, block: bool) -> None:
            assert block is False
            self.running = True
            events.append("server.start")
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        def close(self) -> None:
            assert self.running
            self.running = False
            events.append("server.close")

    runtime = Runtime()
    config = SimpleNamespace(
        control=ControlConfig(),
        arms={
            "left": SimpleNamespace(
                leader=SimpleNamespace(can_channel="can0", rpc_port=1234)
            )
        },
    )
    monkeypatch.setattr(worker, "load_teleoperation_config", lambda _path: config)
    monkeypatch.setattr(worker, "create_leader_endpoint", lambda _channel: Endpoint())
    monkeypatch.setattr(worker, "WorkerControlRuntime", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(worker.portal, "Server", Server)
    monkeypatch.setattr(
        worker.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    worker.serve_worker(Path("teleop.yaml"), "left", "leader")

    assert runtime.stop_requests == 0
    assert events == [
        "server.start",
        "server.close",
        "runtime.close",
        "endpoint.close",
    ]


def test_runtime_applies_only_current_ordered_session_frames() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.begin_control_session("session")
    runtime.submit_control_frame(_frame(clock, 0, ALIGNING))
    runtime.run_once()

    np.testing.assert_allclose(endpoint.commands[-1], np.linspace(0.0, 0.6, 7))
    assert runtime.get_status()["last_applied_sequence"] == 0

    with pytest.raises(ValueError, match="sequence must increase"):
        runtime.submit_control_frame(_frame(clock, 0, ALIGNING))
    stale = _frame(clock, 1, ALIGNING)
    clock.advance(0.06)
    with pytest.raises(ValueError, match="stale"):
        runtime.submit_control_frame(stale)


def test_watchdog_rpc_contract_accepts_portal_integer_arrays() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.begin_control_session("session")

    transported_frame = _portal_roundtrip(_frame(clock, 0, ALIGNING))
    runtime.submit_control_frame(transported_frame)
    runtime.run_once()
    runtime.run_once()

    transported_sample = _portal_roundtrip(runtime.get_sample())
    state, buttons, sampled_at_ns, last_applied = BimanualCoordinator._sample(
        transported_sample, "follower"
    )

    np.testing.assert_allclose(endpoint.commands[-1], np.linspace(0.0, 0.6, 7))
    np.testing.assert_allclose(state, endpoint.state)
    np.testing.assert_allclose(buttons, [0.0])
    assert sampled_at_ns == clock()
    assert last_applied == 0


def test_rpc_timeout_names_endpoint_method_and_deadline(monkeypatch) -> None:
    monkeypatch.setattr(
        coordinator.portal,
        "Client",
        lambda address: _PortalClient(address, TimeoutError()),
    )
    client = RpcEndpointClient(1234, 0.1, "left follower")

    with pytest.raises(
        RuntimeError,
        match=r"left follower get_sample RPC timed out after 0\.100s",
    ):
        client.get_sample()


def test_rpc_disconnect_with_empty_message_names_exception(monkeypatch) -> None:
    class Disconnected(Exception):
        pass

    monkeypatch.setattr(
        coordinator.portal,
        "Client",
        lambda address: _PortalClient(address, Disconnected()),
    )
    client = RpcEndpointClient(1237, 0.1, "right leader")

    with pytest.raises(
        RuntimeError,
        match="right leader ping RPC failed: Disconnected",
    ):
        client.ping()


def test_rpc_client_closes_portal_transport(monkeypatch) -> None:
    transport = _PortalClient("unused", TimeoutError())
    monkeypatch.setattr(coordinator.portal, "Client", lambda _address: transport)
    client = RpcEndpointClient(1236, 0.1, "left leader")

    client.close()

    assert transport.close_timeout == 0.1


def test_can_preflight_rejects_non_can_interface(tmp_path) -> None:
    from devices.robots.i2rt_teleop.config import (
        ArmConfig,
        EndpointConfig,
        FollowerConfig,
        GripperConfig,
    )

    channels = {
        "left_leader": "can_leader_l",
        "left_follower": "can_follower_l",
        "right_leader": "can_leader_r",
        "right_follower": "can_follower_r",
    }
    for channel in channels.values():
        type_path = tmp_path / channel / "type"
        type_path.parent.mkdir()
        type_path.write_text("280\n", encoding="ascii")
    (tmp_path / channels["right_leader"] / "type").write_text(
        "772\n", encoding="ascii"
    )
    arms = {
        side: ArmConfig(
            leader=EndpointConfig(
                channels[f"{side}_leader"], 1236 if side == "left" else 1237
            ),
            follower=FollowerConfig(
                channels[f"{side}_follower"],
                1234 if side == "left" else 1235,
                GripperConfig(backend="none"),
            ),
        )
        for side in ("left", "right")
    }
    config = TeleoperationConfig(control=ControlConfig(), arms=arms)

    with pytest.raises(
        RuntimeError,
        match=r"right/leader channel 'can_leader_r' is interface type 772",
    ):
        coordinator._validate_can_interfaces(config, sys_class_net=tmp_path)


def test_startup_failure_stops_workers_before_closing_rpc_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    events: list[str] = []

    class FakeProcesses:
        def __init__(self, _config_path, _config) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    class FakeClient:
        def __init__(self, label: str) -> None:
            self.label = label

        def close(self) -> None:
            events.append(f"close:{self.label}")

    leaders = {side: FakeClient(f"{side}/leader") for side in ("left", "right")}
    followers = {
        side: FakeClient(f"{side}/follower") for side in ("left", "right")
    }
    monkeypatch.setattr(coordinator, "WorkerProcesses", FakeProcesses)
    monkeypatch.setattr(coordinator, "_validate_can_interfaces", lambda _config: None)
    monkeypatch.setattr(
        coordinator,
        "_connect_clients",
        lambda _config: (leaders, followers),
    )

    def fail_startup(*_args) -> None:
        events.append("wait")
        raise TimeoutError("left/leader did not initialize")

    monkeypatch.setattr(coordinator, "_wait_for_workers", fail_startup)
    config = TeleoperationConfig(control=ControlConfig(), arms={})

    with pytest.raises(TimeoutError, match="left/leader"):
        coordinator.run_teleoperation(tmp_path / "teleop.yaml", config)

    assert events[:3] == ["start", "wait", "stop"]
    assert events[3:] == [
        "close:left/follower",
        "close:right/follower",
        "close:left/leader",
        "close:right/leader",
    ]


def test_command_lease_expiry_latches_fault_and_holds_current_position() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.begin_control_session("session")
    runtime.submit_control_frame(_frame(clock, 0, ALIGNING))
    runtime.run_once()
    safe_count = len(endpoint.safe_states)

    clock.advance(0.11)
    runtime.run_once()

    status = runtime.get_status()
    assert status["phase"] == FAULT
    assert status["fault_reason"] == "command lease expired"
    assert len(endpoint.safe_states) == safe_count + 1
    with pytest.raises(RuntimeError, match="faulted"):
        runtime.begin_control_session("new-session")


def test_feedback_timeout_latches_fault() -> None:
    runtime, endpoint, clock = _runtime()
    endpoint.read_error = RuntimeError("CAN timeout")
    clock.advance(0.06)

    runtime.run_once()

    assert runtime.get_status()["phase"] == FAULT
    assert "feedback read failed" in runtime.get_status()["fault_reason"]


def test_follower_watchdog_ignores_external_gripper_tracking_error() -> None:
    endpoint = FakeEndpoint(follow_gripper=False)
    clock = FakeClock()
    control = replace(
        ControlConfig(),
        following_error_fault_threshold=0.1,
        following_error_fault_samples=1,
    )
    runtime = WorkerControlRuntime(
        endpoint,
        role="follower",
        side="left",
        control=control,
        clock_ns=clock,
    )
    runtime.run_once()
    runtime.begin_control_session("session")
    command = endpoint.state.copy()
    command[-1] = 0.0

    runtime.submit_control_frame(
        {
            "session_id": "session",
            "sequence": 0,
            "issued_at_ns": clock(),
            "phase": ALIGNING,
            "joint_position": command,
        }
    )
    runtime.run_once()

    assert endpoint.state[-1] == 6.0
    assert runtime.get_status()["phase"] == ALIGNING
    assert runtime.get_status()["fault_reason"] is None


def test_follower_watchdog_reports_side_joint_and_phase() -> None:
    runtime, endpoint, clock = _runtime()
    runtime.control = replace(
        runtime.control,
        following_error_fault_threshold=0.1,
        following_error_fault_samples=1,
    )
    runtime.begin_control_session("session")
    command = endpoint.state.copy()
    command[3] += 0.2
    runtime.submit_control_frame(
        {
            "session_id": "session",
            "sequence": 0,
            "issued_at_ns": clock(),
            "phase": ALIGNING,
            "joint_position": command,
        }
    )

    runtime.run_once()

    assert runtime.get_status()["fault_reason"] == (
        "left follower following error 0.200 rad at joint 4 "
        "exceeds 0.100 rad during ALIGNING"
    )


def test_leader_holds_during_alignment_then_enables_active_feedback() -> None:
    runtime, endpoint, clock = _runtime(role="leader")
    runtime.begin_control_session("session")
    alignment_frame = _frame(clock, 0, ALIGNING, size=6)
    runtime.submit_control_frame(alignment_frame)
    runtime.run_once()
    assert len(endpoint.position_holds) == 1
    np.testing.assert_allclose(
        endpoint.position_holds[0], alignment_frame["joint_position"]
    )
    assert endpoint.feedback_updates == []

    runtime.set_phase("session", ACTIVE)
    runtime.submit_control_frame(_frame(clock, 1, ACTIVE, size=6))
    runtime.run_once()

    assert len(endpoint.feedback_updates) == 1
    assert endpoint.feedback_updates[0][0] is True


def test_active_transition_releases_hold_when_leader_feedback_is_disabled() -> None:
    runtime, endpoint, clock = _runtime(role="leader", leader_feedback=False)
    runtime.begin_control_session("session")
    runtime.submit_control_frame(_frame(clock, 0, ALIGNING, size=6))
    runtime.run_once()

    runtime.set_phase("session", ACTIVE)
    runtime.submit_control_frame(_frame(clock, 1, ACTIVE, size=6))
    runtime.run_once()

    assert len(endpoint.feedback_updates) == 1
    assert endpoint.feedback_updates[0][0] is False


def test_leader_watchdog_fault_releases_alignment_hold() -> None:
    runtime, endpoint, clock = _runtime(role="leader")
    runtime.begin_control_session("session")
    runtime.submit_control_frame(_frame(clock, 0, ALIGNING, size=6))
    runtime.run_once()
    safe_count = len(endpoint.safe_states)

    clock.advance(0.11)
    runtime.run_once()

    assert runtime.get_status()["phase"] == FAULT
    assert len(endpoint.safe_states) == safe_count + 1


def test_threaded_watchdog_remains_responsive_when_hardware_command_blocks() -> None:
    endpoint = BlockingCommandEndpoint()
    control = replace(
        ControlConfig(),
        command_frame_max_age_s=0.20,
        command_lease_timeout_s=0.20,
        feedback_timeout_s=0.05,
        watchdog_check_interval_s=0.005,
    )
    runtime = WorkerControlRuntime(
        endpoint,
        role="follower",
        side="left",
        control=control,
    )
    runtime.start(timeout_s=1.0)
    try:
        runtime.begin_control_session("session")
        runtime.submit_control_frame(
            {
                "session_id": "session",
                "sequence": 0,
                "issued_at_ns": time.monotonic_ns(),
                "phase": ALIGNING,
                "joint_position": np.linspace(0.0, 0.6, 7),
            }
        )
        assert endpoint.command_started.wait(timeout=1.0)

        started = time.monotonic()
        status = runtime.get_status()
        rpc_elapsed = time.monotonic() - started
        assert rpc_elapsed < 0.05
        assert status["io_inflight"] is True

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if runtime.get_status()["phase"] == FAULT:
                break
            time.sleep(0.005)
        status = runtime.get_status()
        assert status["phase"] == FAULT
        assert "hardware I/O stalled" in status["fault_reason"]
    finally:
        endpoint.release_command.set()
        assert runtime.close()
