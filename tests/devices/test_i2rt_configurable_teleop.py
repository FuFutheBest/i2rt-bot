"""Hardware-free tests for configurable I2RT bimanual teleoperation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from devices.robots.i2rt_teleop.config import (
    ArmConfig,
    ControlConfig,
    EndpointConfig,
    FollowerConfig,
    GripperConfig,
    TeleoperationConfig,
    load_teleoperation_config,
)
from devices.robots.i2rt_teleop.coordinator import (
    BimanualCoordinator,
    I2RTStatePublisher,
)
from devices.robots.i2rt_teleop.endpoints import (
    CompositeFollowerEndpoint,
    LeaderEndpoint,
)
from devices.robots.i2rt_bimanual import I2RTBimanualTeleop
from devices.robots.i2rt_teleop.grippers import LKTechGripper


class FakeArm:
    def __init__(self, *, integrated: bool) -> None:
        self.integrated = integrated
        self.commands: list[np.ndarray] = []
        self.gains: list[tuple[np.ndarray, np.ndarray]] = []
        self.closed = False

    def get_observations(self):
        result = {"joint_pos": np.arange(6, dtype=np.float64)}
        if self.integrated:
            result["gripper_pos"] = np.array([0.75])
        return result

    def command_joint_pos(self, command):
        self.commands.append(np.asarray(command, dtype=np.float64).copy())

    def update_kp_kd(self, kp, kd):
        self.gains.append((np.asarray(kp).copy(), np.asarray(kd).copy()))

    def close(self):
        self.closed = True


class FakeLeaderArm(FakeArm):
    def __init__(self) -> None:
        super().__init__(integrated=False)
        self.default_kp = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
        self.default_kd = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
        encoder = SimpleNamespace(position=0.25, io_inputs=[1, 0])
        self.motor_chain = SimpleNamespace(
            get_same_bus_device_states=lambda: [encoder]
        )

    def get_robot_info(self):
        return {"kp": self.default_kp, "kd": self.default_kd}


class FakeGripper:
    def __init__(self, state: float = 0.25) -> None:
        self.state = state
        self.commands: list[float] = []
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def read_normalized(self):
        return self.state

    def command_normalized(self, value):
        self.commands.append(float(value))
        return True

    def close(self):
        self.closed = True


class FakeLKDriver:
    max_width_m = 0.09

    def __init__(self) -> None:
        self.enabled = False
        self.disabled = False
        self.enable_count = 0
        self.disable_count = 0
        self.moves: list[tuple[float, float, object]] = []
        self.width = 0.034
        self.state_reads = 0
        self.move_event = threading.Event()

    def Enable(self, _label):
        self.enabled = True
        self.disabled = False
        self.enable_count += 1

    def Disable(self):
        self.enabled = False
        self.disabled = True
        self.disable_count += 1

    def Move(self, width, velocity, force_limit):
        self.moves.append((float(width), float(velocity), force_limit))
        self.width = float(width)
        self.move_event.set()
        return True

    def states(self):
        self.state_reads += 1
        return SimpleNamespace(width=self.width)


class IntermittentLKDriver(FakeLKDriver):
    def __init__(
        self, *, feedback_failures: int = 0, command_failures: int = 0
    ) -> None:
        super().__init__()
        self.feedback_failures = feedback_failures
        self.command_failures = command_failures

    def Move(self, width, velocity, force_limit):
        if self.command_failures > 0:
            self.command_failures -= 1
            raise RuntimeError("simulated command timeout")
        return super().Move(width, velocity, force_limit)

    def states(self):
        self.state_reads += 1
        if self.feedback_failures > 0:
            self.feedback_failures -= 1
            raise RuntimeError("simulated serial timeout")
        return SimpleNamespace(width=self.width)


class BlockingFeedbackLKDriver(FakeLKDriver):
    def __init__(self) -> None:
        super().__init__()
        self.feedback_started = threading.Event()
        self.release_feedback = threading.Event()

    def states(self):
        self.feedback_started.set()
        if not self.release_feedback.wait(timeout=1.0):
            raise RuntimeError("test did not release feedback read")
        return super().states()


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for background gripper I/O")
        time.sleep(0.005)


class FakeRpcClient:
    def __init__(self, samples, *, follow_gripper: bool = True) -> None:
        self.samples = list(samples)
        self.follow_gripper = follow_gripper
        self.commands: list[np.ndarray] = []
        self.sessions: list[tuple[str, str]] = []
        first = self.samples[0]
        self.role = (
            "leader"
            if isinstance(first, (tuple, list)) and len(first) == 2
            else "follower"
        )
        self.phase = "SAFE_HOLD"
        self.session_id = None
        self.last_applied_sequence = -1
        self._follower_state = (
            np.asarray(first, dtype=np.float64).copy()
            if self.role == "follower"
            else None
        )

    def get_sample(self):
        if len(self.samples) > 1:
            raw = self.samples.pop(0)
        else:
            raw = self.samples[0]
        if self.role == "leader":
            state, buttons = raw
        else:
            state = self._follower_state
            buttons = np.zeros(1)
        return {
            "state": np.asarray(state, dtype=np.float64).copy(),
            "buttons": np.asarray(buttons, dtype=np.float64).copy(),
            "sampled_at_ns": time.monotonic_ns(),
            "phase": self.phase,
            "fault_reason": None,
            "last_applied_sequence": self.last_applied_sequence,
        }

    def start_get_sample(self):
        return self.get_sample()

    def finish_get_sample(self, result):
        return result

    def begin_control_session(self, session_id):
        self.session_id = session_id
        self.phase = "ALIGNING"
        self.last_applied_sequence = -1
        self.sessions.append(("begin", session_id))

    def submit_control_frame(self, frame):
        assert frame["session_id"] == self.session_id
        assert frame["phase"] == self.phase
        command = np.asarray(frame["joint_position"], dtype=np.float64).copy()
        self.commands.append(command)
        self.last_applied_sequence = frame["sequence"]
        if self.role == "follower":
            self._follower_state[:6] = command[:6]
            if self.follow_gripper:
                self._follower_state[-1] = command[-1]

    def set_phase(self, session_id, phase):
        assert session_id == self.session_id
        self.phase = phase
        self.sessions.append(("phase", phase))

    def end_control_session(self, session_id):
        assert session_id == self.session_id
        self.sessions.append(("end", session_id))
        self.session_id = None
        self.phase = "SAFE_HOLD"


class FakeStatePublisher:
    def __init__(self) -> None:
        self.samples = []

    def publish(self, leader_states, follower_states) -> None:
        self.samples.append((leader_states, follower_states))


def _write_config(tmp_path: Path, *, left_backend: str = "lktech") -> Path:
    calibration = tmp_path / "left.json"
    calibration.write_text(
        json.dumps(
            {
                "min_angle_deg": -1000.0,
                "max_angle_deg": 20.0,
                "gear_ratio": 10.0,
            }
        ),
        encoding="utf-8",
    )
    document = {
        "version": 1,
        "control": {"frequency_hz": 100.0},
        "arms": {
            "left": {
                "leader": {"can_channel": "can_leader_l", "rpc_port": 1236},
                "follower": {
                    "can_channel": "can_follower_l",
                    "rpc_port": 1234,
                    "gripper": {
                        "backend": left_backend,
                        "port": "/dev/ttyUSB-left",
                        "calibration_file": str(calibration),
                        "startup_width_ratio": 1.0,
                    },
                },
            },
            "right": {
                "leader": {"can_channel": "can_leader_r", "rpc_port": 1237},
                "follower": {
                    "can_channel": "can_follower_r",
                    "rpc_port": 1235,
                    "gripper": {"backend": "i2rt_can", "type": "linear_4310"},
                },
            },
        },
    }
    path = tmp_path / "teleop.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_config_supports_mixed_gripper_backends(tmp_path: Path) -> None:
    config = load_teleoperation_config(_write_config(tmp_path))

    assert config.arms["left"].follower.gripper.backend == "lktech"
    assert config.arms["left"].follower.gripper.calibration_file == (
        tmp_path / "left.json"
    )
    assert config.arms["left"].follower.gripper.startup_width_ratio == 1.0
    assert config.arms["right"].follower.gripper.backend == "i2rt_can"


def test_repository_config_uses_lktech_grippers(repo_root: Path) -> None:
    config = load_teleoperation_config(
        repo_root / "configs" / "i2rt_bimanual_teleop.yaml"
    )

    assert config.arms["left"].follower.gripper.backend == "lktech"
    assert config.arms["right"].follower.gripper.backend == "lktech"
    assert config.arms["left"].follower.gripper.startup_width_ratio == 1.0
    assert config.arms["right"].follower.gripper.startup_width_ratio == 1.0
    assert config.arms["left"].follower.gripper.invert is False
    assert config.arms["right"].follower.gripper.invert is False
    assert config.arms["left"].follower.rpc_port == 1234
    assert config.arms["right"].leader.rpc_port == 1237
    assert config.episode_signal_pattern == "/tmp/superinference_episode_{signal}"


def test_composite_device_delegates_lifecycle_to_teleoperation_runtime(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    state_feedback = {
        "fps": 100.0,
        "buffer_size": 4,
        "hardware_latency_ms": 0.0,
        "format": {
            "shape": [14],
            "dtype": "float64",
            "type": ["j1-7", "j1-7"],
        },
    }
    calls = []
    monkeypatch.setattr(
        "devices.robots.i2rt_bimanual.run_teleoperation",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    device = I2RTBimanualTeleop(
        device_id=0,
        device_name="I2RTRobot",
        follower_device_id=1,
        state_feedback=state_feedback,
        follower_state_feedback=state_feedback,
        teleoperation_config_path=repo_root
        / "configs"
        / "i2rt_bimanual_teleop.yaml",
    )

    device.start_server()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == repo_root / "configs" / "i2rt_bimanual_teleop.yaml"
    assert kwargs["device_name"] == "I2RTRobot"
    assert kwargs["leader_device_id"] == 0
    assert kwargs["follower_device_id"] == 1
    assert kwargs["leader_state_feedback"].format.shape == (14,)
    assert kwargs["follower_state_feedback"].format.shape == (14,)


def test_config_rejects_duplicate_rpc_ports(tmp_path: Path) -> None:
    path = _write_config(tmp_path, left_backend="none")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["arms"]["right"]["follower"]["rpc_port"] = 1234
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="rpc_port values must be unique"):
        load_teleoperation_config(path)


def test_config_rejects_lktech_without_calibration(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["arms"]["left"]["follower"]["gripper"]["calibration_file"] = str(
        tmp_path / "missing.json"
    )
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        load_teleoperation_config(path)


def test_integrated_follower_keeps_seven_value_can_command() -> None:
    arm = FakeArm(integrated=True)
    endpoint = CompositeFollowerEndpoint(
        arm,
        GripperConfig(backend="i2rt_can"),
    )
    command = np.linspace(0.0, 0.6, 7)

    np.testing.assert_allclose(endpoint.get_info(), [0, 1, 2, 3, 4, 5, 0.75])
    endpoint.command_joint_pos(command)
    np.testing.assert_allclose(arm.commands[-1], command)
    endpoint.update_kp_kd(np.ones(7), np.full(7, 2.0))

    assert arm.gains[-1][0].shape == (7,)


def test_leader_endpoint_maps_handle_and_applies_gain_update_immediately() -> None:
    arm = FakeLeaderArm()
    endpoint = LeaderEndpoint(arm)

    state, buttons = endpoint.get_info()
    np.testing.assert_allclose(state, [0, 1, 2, 3, 4, 5, 0.75])
    np.testing.assert_allclose(buttons, [1, 0])

    endpoint.update_kp_kd(np.ones(6), np.zeros(6))
    np.testing.assert_allclose(arm.commands[-1], np.arange(6))

    hold_target = np.linspace(0.1, 0.6, 6)
    endpoint.enter_position_hold(hold_target)
    np.testing.assert_allclose(arm.gains[-1][0], arm.default_kp)
    np.testing.assert_allclose(arm.gains[-1][1], arm.default_kd)
    np.testing.assert_allclose(arm.commands[-1], hold_target)


def test_external_follower_splits_arm_and_gripper_command() -> None:
    arm = FakeArm(integrated=False)
    gripper = FakeGripper(state=0.3)
    endpoint = CompositeFollowerEndpoint(
        arm,
        GripperConfig(backend="lktech", port="/dev/fake"),
        gripper,
    )
    command = np.linspace(0.0, 0.6, 7)

    endpoint.start()
    endpoint.command_joint_pos(command)
    np.testing.assert_allclose(arm.commands[-1], command[:6])
    endpoint.update_kp_kd(np.ones(6), np.full(6, 2.0))

    np.testing.assert_allclose(endpoint.get_info(), [0, 1, 2, 3, 4, 5, 0.3])
    assert gripper.commands == [pytest.approx(command[-1])]
    assert arm.gains[-1][0].shape == (6,)

    endpoint.close()
    assert gripper.closed and arm.closed


def test_lktech_adapter_maps_normalized_value_and_rate_limits() -> None:
    driver = FakeLKDriver()
    config = GripperConfig(
        backend="lktech",
        port="/dev/fake-rate-limit",
        min_width_m=0.01,
        max_width_m=0.09,
        scale_factor=0.8,
        command_offset=0.1,
        command_hz=100.0,
        feedback_hz=100.0,
        deadband=0.002,
        command_velocity=0.4,
    )
    adapter = LKTechGripper(config, label="test", driver=driver)

    adapter.start()
    assert adapter.command_normalized(0.5)
    _wait_until(lambda: len(driver.moves) == 1)
    assert driver.moves[-1] == pytest.approx((0.034, 0.4, None))
    assert adapter.command_normalized(0.8)
    assert not adapter.command_normalized(0.8)
    _wait_until(lambda: len(driver.moves) == 2)
    _wait_until(lambda: adapter.read_normalized() == pytest.approx(0.8))
    assert len(driver.moves) == 2

    adapter.close()
    assert driver.disabled
    assert adapter.health == "stopped"


def test_lktech_adapter_moves_to_configured_startup_width() -> None:
    driver = FakeLKDriver()
    config = GripperConfig(
        backend="lktech",
        port="/dev/fake-startup",
        min_width_m=0.01,
        max_width_m=0.09,
        startup_width_ratio=1.0,
        command_velocity=0.4,
    )
    adapter = LKTechGripper(config, label="test", driver=driver)

    adapter.start()

    _wait_until(lambda: len(driver.moves) == 1)
    assert len(driver.moves) == 1
    assert driver.moves[0][0] == pytest.approx(0.09)
    assert driver.moves[0][1] == pytest.approx(0.4)
    assert driver.moves[0][2] is None
    adapter.close()


def test_lktech_adapter_recovers_without_replaying_stale_command() -> None:
    driver = IntermittentLKDriver(feedback_failures=1)
    config = GripperConfig(
        backend="lktech",
        port="/dev/fake-transient-timeout",
        min_width_m=0.0,
        max_width_m=0.09,
        startup_width_ratio=0.8,
        command_hz=100.0,
        feedback_hz=100.0,
    )
    adapter = LKTechGripper(
        config,
        label="test",
        driver=driver,
        reconnect_interval_s=0.01,
    )
    adapter.start()

    _wait_until(lambda: driver.enable_count >= 2)
    _wait_until(lambda: adapter.health == "healthy")
    assert adapter.read_normalized() == pytest.approx(0.8)
    assert len(driver.moves) == 1
    assert not adapter.command_normalized(0.8)
    time.sleep(0.03)
    assert len(driver.moves) == 1

    assert adapter.command_normalized(0.7)
    _wait_until(lambda: len(driver.moves) == 2)

    adapter.close()


def test_persistent_lktech_failure_does_not_interrupt_follower_arm() -> None:
    driver = IntermittentLKDriver(feedback_failures=1000)
    config = GripperConfig(
        backend="lktech",
        port="/dev/fake-persistent-timeout",
        min_width_m=0.0,
        max_width_m=0.09,
        startup_width_ratio=0.6,
        feedback_hz=100.0,
    )
    adapter = LKTechGripper(
        config,
        label="test",
        driver=driver,
        reconnect_interval_s=0.01,
    )
    arm = FakeArm(integrated=False)
    endpoint = CompositeFollowerEndpoint(
        arm,
        config,
        adapter,
    )
    endpoint.start()

    _wait_until(lambda: adapter.health == "recovering")
    command = np.linspace(0.0, 0.6, 7)
    endpoint.command_joint_pos(command)
    state = endpoint.get_info()

    np.testing.assert_allclose(arm.commands[-1], command[:6])
    np.testing.assert_allclose(state[:6], np.arange(6))
    assert state[-1] == pytest.approx(0.6)
    assert adapter.last_error is not None

    endpoint.close()
    assert arm.closed
    assert adapter.health == "stopped"


def test_lktech_command_failure_isolated_from_arm_commands() -> None:
    driver = IntermittentLKDriver(command_failures=1)
    config = GripperConfig(
        backend="lktech",
        port="/dev/fake-command-timeout",
        min_width_m=0.0,
        max_width_m=0.09,
        command_hz=100.0,
        feedback_hz=100.0,
    )
    adapter = LKTechGripper(
        config,
        label="test",
        driver=driver,
        reconnect_interval_s=0.01,
    )
    arm = FakeArm(integrated=False)
    endpoint = CompositeFollowerEndpoint(arm, config, adapter)
    endpoint.start()

    first = np.linspace(0.0, 0.5, 7)
    endpoint.command_joint_pos(first)
    _wait_until(lambda: adapter.health == "recovering")
    second = np.linspace(0.1, 0.7, 7)
    endpoint.command_joint_pos(second)

    np.testing.assert_allclose(arm.commands, [first[:6], second[:6]])
    endpoint.close()


def test_blocked_lktech_feedback_does_not_stall_follower_arm() -> None:
    driver = BlockingFeedbackLKDriver()
    config = GripperConfig(
        backend="lktech",
        port="/dev/fake-blocked-feedback",
        min_width_m=0.0,
        max_width_m=0.09,
        startup_width_ratio=0.4,
    )
    adapter = LKTechGripper(config, label="test", driver=driver)
    arm = FakeArm(integrated=False)
    endpoint = CompositeFollowerEndpoint(arm, config, adapter)
    endpoint.start()

    assert driver.feedback_started.wait(timeout=1.0)
    started_at = time.monotonic()
    state = endpoint.get_info()
    command = np.linspace(0.1, 0.7, 7)
    endpoint.command_joint_pos(command)
    elapsed = time.monotonic() - started_at

    np.testing.assert_allclose(state[:6], np.arange(6))
    np.testing.assert_allclose(arm.commands[-1], command[:6])
    assert elapsed < 0.1

    driver.release_feedback.set()
    endpoint.close()


def test_lktech_adapter_builds_sdk_driver_without_opening_serial(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "gripper.json"
    calibration.write_text(
        json.dumps(
            {
                "min_angle_deg": -2757.35,
                "max_angle_deg": -1715.87,
                "gear_ratio": 10.0,
            }
        ),
        encoding="utf-8",
    )
    adapter = LKTechGripper(
        GripperConfig(
            backend="lktech",
            port="/dev/does-not-open",
            calibration_file=calibration,
            gripper_type="soft",
        ),
        label="left",
    )

    driver = adapter._build_driver()

    assert driver._motor is None
    assert driver._base_config.port == "/dev/does-not-open"
    assert driver._base_config.min_angle_deg == pytest.approx(-2757.35)
    assert driver._base_config.max_angle_deg == pytest.approx(-1715.87)
    assert driver._base_config.gear_ratio == pytest.approx(10.0)


def _coordinator_config() -> TeleoperationConfig:
    arms = {
        side: ArmConfig(
            leader=EndpointConfig(f"can_leader_{side[0]}", 2000 + index),
            follower=FollowerConfig(
                f"can_follower_{side[0]}",
                3000 + index,
                GripperConfig(backend="none"),
            ),
        )
        for index, side in enumerate(("left", "right"))
    }
    return TeleoperationConfig(
        control=replace(
            ControlConfig(),
            alignment_duration_s=0.0,
            alignment_max_velocity_rad_s=1.0e9,
            alignment_settle_timeout_s=0.1,
            alignment_stable_samples=1,
            button_debounce_samples=1,
            bilateral_kp=0.2,
            bilateral_kd=0.1,
        ),
        arms=arms,
    )


def test_coordinator_button_edges_activate_and_deactivate() -> None:
    state = np.arange(7, dtype=np.float64)
    followers = {
        side: FakeRpcClient([np.zeros(7)]) for side in ("left", "right")
    }
    leaders = {
        "left": FakeRpcClient(
            [
                (state, np.array([1.0])),
                (state, np.array([1.0])),
                (state, np.array([1.0])),
                (state, np.array([0.0])),
                (state, np.array([1.0])),
            ]
        ),
        "right": FakeRpcClient(
            [
                (state, np.array([1.0])),
                (state, np.array([1.0])),
                (state, np.array([1.0])),
                (state, np.array([0.0])),
                (state, np.array([0.0])),
            ]
        ),
    }
    coordinator = BimanualCoordinator(
        _coordinator_config(), leaders, followers, sleep=lambda _seconds: None
    )

    coordinator.step()
    assert coordinator.active
    for side in ("left", "right"):
        np.testing.assert_allclose(followers[side].commands[-1], state)
        np.testing.assert_allclose(leaders[side].commands[0], state[:6])
        assert leaders[side].commands[-1].shape == (6,)
        assert followers[side].phase == "ACTIVE"
        assert leaders[side].phase == "ACTIVE"

    coordinator.step()
    assert coordinator.active
    coordinator.step()
    assert not coordinator.active
    for side in ("left", "right"):
        assert followers[side].phase == "SAFE_HOLD"
        assert leaders[side].phase == "SAFE_HOLD"


def test_coordinator_alignment_does_not_wait_for_external_gripper() -> None:
    leader_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    followers = {
        side: FakeRpcClient([np.zeros(7)], follow_gripper=False)
        for side in ("left", "right")
    }
    leaders = {
        side: FakeRpcClient([(leader_state, np.array([1.0]))])
        for side in ("left", "right")
    }
    coordinator = BimanualCoordinator(
        _coordinator_config(), leaders, followers, sleep=lambda _seconds: None
    )

    coordinator.step()

    assert coordinator.active
    for follower in followers.values():
        np.testing.assert_allclose(follower._follower_state[:6], leader_state[:6])
        assert follower._follower_state[-1] == 0.0


def test_coordinator_maps_only_right_wrist_rotation() -> None:
    config = _coordinator_config()
    right = config.arms["right"]
    config = replace(
        config,
        arms={
            **config.arms,
            "right": replace(
                right,
                follower=replace(
                    right.follower,
                    joint_directions=(1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0),
                ),
            ),
        },
    )
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    followers = {
        side: FakeRpcClient([np.zeros(7)]) for side in ("left", "right")
    }
    leaders = {
        side: FakeRpcClient([(state, np.zeros(1))]) for side in ("left", "right")
    }
    coordinator = BimanualCoordinator(config, leaders, followers)

    np.testing.assert_allclose(coordinator._map_follower_state("left", state), state)
    np.testing.assert_allclose(
        coordinator._map_follower_state("right", state),
        [0.1, 0.2, 0.3, 0.4, 0.5, -0.6, 0.7],
    )


def test_coordinator_scales_alignment_duration_to_joint_distance() -> None:
    config = _coordinator_config()
    config = replace(
        config,
        control=replace(
            config.control,
            frequency_hz=10.0,
            alignment_duration_s=1.0,
            alignment_max_velocity_rad_s=0.5,
        ),
    )
    follower_state = np.zeros(7)
    leader_state = np.zeros(7)
    leader_state[0] = 1.4
    followers = {
        side: FakeRpcClient([follower_state.copy()])
        for side in ("left", "right")
    }
    leaders = {
        side: FakeRpcClient([(leader_state.copy(), np.zeros(1))])
        for side in ("left", "right")
    }
    elapsed = [0.0]
    coordinator = BimanualCoordinator(
        config,
        leaders,
        followers,
        clock=lambda: elapsed[0],
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    )
    coordinator._begin_session()

    coordinator._align_followers(
        {side: leader_state.copy() for side in ("left", "right")},
        {side: follower_state.copy() for side in ("left", "right")},
    )

    assert elapsed[0] == pytest.approx(4.2)


def test_coordinator_rejects_activation_when_leader_does_not_hold_target() -> None:
    frozen = np.zeros(7)
    displaced = np.full(7, 0.5)
    followers = {
        side: FakeRpcClient([np.zeros(7)]) for side in ("left", "right")
    }
    leaders = {
        side: FakeRpcClient(
            [
                (frozen, np.array([1.0])),
                (frozen, np.array([1.0])),
                (displaced, np.array([1.0])),
            ]
        )
        for side in ("left", "right")
    }
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    coordinator = BimanualCoordinator(
        _coordinator_config(), leaders, followers, clock=clock, sleep=sleep
    )

    with pytest.raises(TimeoutError, match="leaders and followers did not settle"):
        coordinator.step()

    assert not coordinator.active
    assert all(client.phase == "SAFE_HOLD" for client in leaders.values())
    assert all(client.phase == "SAFE_HOLD" for client in followers.values())


def test_coordinator_single_press_cancels_alignment_after_chord_release() -> None:
    state = np.zeros(7)
    followers = {
        side: FakeRpcClient([state.copy()]) for side in ("left", "right")
    }
    leaders = {
        "left": FakeRpcClient(
            [(state, [1]), (state, [0]), (state, [1])]
        ),
        "right": FakeRpcClient(
            [(state, [1]), (state, [0]), (state, [0])]
        ),
    }
    coordinator = BimanualCoordinator(
        _coordinator_config(), leaders, followers, sleep=lambda _seconds: None
    )

    coordinator.step()

    assert not coordinator.active
    assert all(client.phase == "SAFE_HOLD" for client in leaders.values())
    assert all(client.phase == "SAFE_HOLD" for client in followers.values())


def test_coordinator_does_not_enter_from_alternating_button_presses() -> None:
    state = np.zeros(7)
    followers = {
        side: FakeRpcClient([state]) for side in ("left", "right")
    }
    leaders = {
        "left": FakeRpcClient([(state, [1]), (state, [0]), (state, [0])]),
        "right": FakeRpcClient([(state, [0]), (state, [1]), (state, [0])]),
    }
    coordinator = BimanualCoordinator(
        _coordinator_config(), leaders, followers, sleep=lambda _seconds: None
    )

    coordinator.step()
    coordinator.step()
    coordinator.step()

    assert not coordinator.active
    assert all(not client.sessions for client in followers.values())


def test_coordinator_ends_all_workers_when_one_command_fails() -> None:
    state = np.zeros(7)
    followers = {
        side: FakeRpcClient([state]) for side in ("left", "right")
    }
    leaders = {
        side: FakeRpcClient([(state, [1])]) for side in ("left", "right")
    }
    coordinator = BimanualCoordinator(
        _coordinator_config(), leaders, followers, sleep=lambda _seconds: None
    )
    coordinator.step()
    assert coordinator.active

    def fail_command(_frame):
        raise RuntimeError("simulated RPC failure")

    followers["left"].submit_control_frame = fail_command
    with pytest.raises(RuntimeError, match="simulated RPC failure"):
        coordinator.step()

    assert not coordinator.active
    assert all(client.phase == "SAFE_HOLD" for client in followers.values())
    assert all(client.phase == "SAFE_HOLD" for client in leaders.values())


def test_coordinator_second_buttons_emit_episode_edges() -> None:
    state = np.arange(7, dtype=np.float64)
    followers = {
        side: FakeRpcClient([np.zeros(7)]) for side in ("left", "right")
    }
    leaders = {
        "left": FakeRpcClient(
            [
                (state, [0, 0]),
                (state, [0, 0]),
                (state, [0, 1]),
                (state, [0, 1]),
                (state, [0, 0]),
            ]
        ),
        "right": FakeRpcClient(
            [
                (state, [0, 1]),
                (state, [0, 1]),
                (state, [0, 1]),
                (state, [0, 0]),
                (state, [0, 1]),
            ]
        ),
    }
    signals = []
    publisher = FakeStatePublisher()
    coordinator = BimanualCoordinator(
        _coordinator_config(),
        leaders,
        followers,
        state_publisher=publisher,
        episode_signal=signals.append,
        sleep=lambda _seconds: None,
    )

    for _ in range(5):
        coordinator.step()

    assert signals == ["toggle", "discard", "toggle"]
    assert len(publisher.samples) == 5


def test_coordinator_starts_all_state_rpcs_before_waiting_for_results() -> None:
    started: list[str] = []
    state = np.arange(7, dtype=np.float64)

    class DeferredRpcClient(FakeRpcClient):
        def __init__(self, label, samples) -> None:
            super().__init__(samples)
            self.label = label

        def start_get_sample(self):
            started.append(self.label)
            return super().start_get_sample()

        def finish_get_sample(self, result):
            assert len(started) == 4
            return result

    followers = {
        side: DeferredRpcClient(f"{side}.follower", [state.copy()])
        for side in ("left", "right")
    }
    leaders = {
        side: DeferredRpcClient(
            f"{side}.leader", [(state.copy(), np.zeros(1))]
        )
        for side in ("left", "right")
    }
    coordinator = BimanualCoordinator(_coordinator_config(), leaders, followers)

    coordinator.step()

    assert started == [
        "left.leader",
        "left.follower",
        "right.leader",
        "right.follower",
    ]


def test_i2rt_state_publisher_uses_leader_follower_shm_layout() -> None:
    from consumer.io.async_data_saver_i2rt_NEDF3 import DeviceShmReader

    device_name = f"I2RTStatePublisherTest_{time.time_ns()}"
    publisher = I2RTStatePublisher(
        device_name=device_name,
        frequency_hz=50.0,
        buffer_size=4,
    )
    leader_reader = DeviceShmReader(f"{device_name}_0_data")
    follower_reader = DeviceShmReader(f"{device_name}_1_data")
    try:
        leader_reader.connect()
        follower_reader.connect()
        publisher.publish(
            {"left": np.arange(7), "right": np.arange(7) + 10},
            {"left": np.arange(7) + 20, "right": np.arange(7) + 30},
        )

        np.testing.assert_allclose(
            leader_reader.read_latest().data,
            np.concatenate((np.arange(7), np.arange(7) + 10)),
        )
        np.testing.assert_allclose(
            follower_reader.read_latest().data,
            np.concatenate((np.arange(7) + 20, np.arange(7) + 30)),
        )
    finally:
        leader_reader.close()
        follower_reader.close()
        publisher.close()
