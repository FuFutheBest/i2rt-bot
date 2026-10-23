"""Process supervision and control loop for bimanual I2RT teleoperation."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import portal

from devices.base import BaseDevice
from devices.config import StateFeedbackConfig
from utils.logger_config import logger as app_logger

from .config import REPO_ROOT, SIDES, TeleoperationConfig
from .endpoints import ARM_DOF
from .runtime import ACTIVE, ALIGNING, FAULT, rpc_integer


LOGGER = logging.getLogger(__name__)
CAN_INTERFACE_TYPE = "280"


def _validate_can_interfaces(
    config: TeleoperationConfig,
    *,
    sys_class_net: Path = Path("/sys/class/net"),
) -> None:
    """Reject missing or incorrectly named interfaces before touching hardware."""

    problems: list[str] = []
    for side in SIDES:
        arm_config = config.arms[side]
        for role, channel in (
            ("leader", arm_config.leader.can_channel),
            ("follower", arm_config.follower.can_channel),
        ):
            type_path = sys_class_net / channel / "type"
            try:
                interface_type = type_path.read_text(encoding="ascii").strip()
            except FileNotFoundError:
                problems.append(f"{side}/{role} channel {channel!r} does not exist")
                continue
            except OSError as exc:
                problems.append(
                    f"{side}/{role} channel {channel!r} cannot be inspected: {exc}"
                )
                continue
            if interface_type != CAN_INTERFACE_TYPE:
                problems.append(
                    f"{side}/{role} channel {channel!r} is interface type "
                    f"{interface_type}, expected CAN type {CAN_INTERFACE_TYPE}"
                )
    if problems:
        raise RuntimeError("I2RT CAN preflight failed: " + "; ".join(problems))


class I2RTStatePublisher:
    """Publish coordinator samples using the existing I2RT SHM contract."""

    def __init__(
        self,
        *,
        device_name: str = "I2RTRobot",
        leader_device_id: int = 0,
        follower_device_id: int = 1,
        frequency_hz: float = 100.0,
        buffer_size: int = 5000,
        leader_state_feedback: (
            StateFeedbackConfig | Mapping[str, Any] | None
        ) = None,
        follower_state_feedback: (
            StateFeedbackConfig | Mapping[str, Any] | None
        ) = None,
    ) -> None:
        default_state_feedback = {
            "fps": frequency_hz,
            "buffer_size": buffer_size,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [14],
                "dtype": "float64",
                "type": ["j1-7", "j1-7"],
            },
        }
        leader_contract = leader_state_feedback or default_state_feedback
        follower_contract = follower_state_feedback or default_state_feedback
        self._devices = {
            "leader": BaseDevice(
                device_class_id=leader_device_id,
                device_name=device_name,
                state_feedback=leader_contract,
            ),
            "follower": BaseDevice(
                device_class_id=follower_device_id,
                device_name=device_name,
                state_feedback=follower_contract,
            ),
        }
        for role, device in self._devices.items():
            if device.data_shape != (14,) or not np.issubdtype(
                np.dtype(device.data_dtype), np.floating
            ):
                raise ValueError(
                    f"{role} state feedback must be a floating-point (14,) stream"
                )
        created: list[BaseDevice] = []
        try:
            for device in self._devices.values():
                device._create_shared_memory()
                created.append(device)
        except Exception:
            for device in created:
                device._cleanup_shared_memory()
            raise

    def publish(
        self,
        leader_states: Mapping[str, np.ndarray],
        follower_states: Mapping[str, np.ndarray],
    ) -> None:
        timestamp_ns = time.time_ns()
        for role, states in (
            ("leader", leader_states),
            ("follower", follower_states),
        ):
            sample = np.concatenate((states["left"], states["right"])).astype(
                np.float64, copy=False
            )
            if sample.shape != (14,):
                raise ValueError(f"{role} SHM sample must have shape (14,)")
            if not self._devices[role]._write_array_to_shm_with_timestamp(
                sample, timestamp_ns
            ):
                raise RuntimeError(f"failed to publish {role} SHM sample")

    def close(self) -> None:
        for device in self._devices.values():
            device._cleanup_shared_memory()


class EpisodeSignalWriter:
    """Atomically emit recorder commands without blocking teleoperation."""

    def __init__(
        self,
        pattern: str = "/tmp/superinference_episode_{signal}",
    ) -> None:
        if "{signal}" not in pattern:
            raise ValueError("episode signal pattern must contain {signal}")
        self.pattern = pattern

    def emit(self, signal: str) -> None:
        if signal not in {"toggle", "discard"}:
            raise ValueError(f"unsupported episode signal: {signal}")
        destination = Path(self.pattern.format(signal=signal))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(f"{time.time_ns()}\n", encoding="ascii")
            temporary.replace(destination)
        except OSError as exc:
            LOGGER.warning("Could not write episode %s signal: %s", signal, exc)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class RpcEndpointClient:
    """Client for the RPC contract exposed by each worker."""

    def __init__(
        self, port: int, timeout_s: float, label: str | None = None
    ) -> None:
        self.port = port
        self.timeout_s = timeout_s
        self.label = label or f"worker on port {port}"
        self._client = portal.Client(f"127.0.0.1:{port}")

    def _start_call(self, method: str, *args: Any) -> Any:
        try:
            return getattr(self._client, method)(*args)
        except TimeoutError as exc:
            raise RuntimeError(
                f"{self.label} {method} RPC timed out after {self.timeout_s:.3f}s"
            ) from exc
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            raise RuntimeError(
                f"{self.label} {method} RPC failed: {detail}"
            ) from exc

    def _finish_call(self, method: str, future: Any) -> Any:
        try:
            return future.result(timeout=self.timeout_s)
        except TimeoutError as exc:
            raise RuntimeError(
                f"{self.label} {method} RPC timed out after {self.timeout_s:.3f}s"
            ) from exc
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            raise RuntimeError(
                f"{self.label} {method} RPC failed: {detail}"
            ) from exc

    def _call(self, method: str, *args: Any) -> Any:
        return self._finish_call(method, self._start_call(method, *args))

    def ping(self) -> Mapping[str, str]:
        return self._call("ping")

    def get_sample(self):
        return self._call("get_sample")

    def start_get_sample(self):
        return self._start_call("get_sample")

    def finish_get_sample(self, future: Any):
        return self._finish_call("get_sample", future)

    def get_status(self):
        return self._call("get_status")

    def begin_control_session(self, session_id: str):
        return self._call("begin_control_session", session_id)

    def set_phase(self, session_id: str, phase: str):
        return self._call("set_phase", session_id, phase)

    def submit_control_frame(self, frame: Mapping[str, Any]):
        return self._call("submit_control_frame", frame)

    def end_control_session(self, session_id: str):
        return self._call("end_control_session", session_id)

    def reset_fault(self):
        return self._call("reset_fault")

    def close(self) -> None:
        self._client.close(timeout=self.timeout_s)


class WorkerProcesses:
    """Start and stop the four side/role worker processes."""

    def __init__(self, config_path: Path, config: TeleoperationConfig) -> None:
        self.config_path = config_path.resolve()
        self.config = config
        self.processes: dict[tuple[str, str], subprocess.Popen] = {}

    def start(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        for role in ("follower", "leader"):
            for side in SIDES:
                command = [
                    sys.executable,
                    "-m",
                    "devices.robots.i2rt_teleop.worker",
                    "--config",
                    str(self.config_path),
                    "--side",
                    side,
                    "--role",
                    role,
                ]
                LOGGER.info("Starting %s %s worker", side, role)
                self.processes[(side, role)] = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    start_new_session=True,
                )

    def ensure_running(self) -> None:
        failed = [
            (side, role, process.returncode)
            for (side, role), process in self.processes.items()
            if process.poll() is not None
        ]
        if failed:
            detail = ", ".join(
                f"{side}/{role} exited {returncode}"
                for side, role, returncode in failed
            )
            raise RuntimeError(f"Teleoperation worker failure: {detail}")

    def stop(self) -> None:
        if self.processes:
            LOGGER.info("Stopping I2RT teleoperation workers")
        for process in reversed(tuple(self.processes.values())):
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 8.0
        for process in reversed(tuple(self.processes.values())):
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self.processes.clear()


class _DebouncedButton:
    def __init__(self, samples: int) -> None:
        self.samples = samples
        self.stable = False
        self._candidate = False
        self._candidate_count = 0

    def update(self, raw: bool) -> tuple[bool, bool]:
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1
        rising = False
        if self._candidate_count >= self.samples and self.stable != raw:
            self.stable = raw
            rising = raw
        return self.stable, rising


class _AlignmentCancelled(Exception):
    """Operator cancelled the in-progress alignment without faulting the rig."""


class BimanualCoordinator:
    """Coordinate one atomic bimanual session through watchdog-protected RPC."""

    def __init__(
        self,
        config: TeleoperationConfig,
        leaders: Mapping[str, RpcEndpointClient],
        followers: Mapping[str, RpcEndpointClient],
        *,
        health_check: Callable[[], None] | None = None,
        state_publisher: I2RTStatePublisher | None = None,
        episode_signal: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.leaders = leaders
        self.followers = followers
        self.health_check = health_check or (lambda: None)
        self.state_publisher = state_publisher
        self.episode_signal = episode_signal
        self.clock = clock
        self.sleep = sleep
        self.active = False
        self._session_id: str | None = None
        self._sequence = -1
        self._control_buttons = {
            side: _DebouncedButton(config.control.button_debounce_samples)
            for side in SIDES
        }
        self._transition_armed = True
        self._chord_press_times: dict[str, float] = {}
        self._episode_button_latched = {side: False for side in SIDES}

    @staticmethod
    def _sample(result: Any, role: str) -> tuple[np.ndarray, np.ndarray, int, int]:
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{role} RPC must return a sample mapping")
        state = np.asarray(result.get("state"), dtype=np.float64).reshape(-1)
        if state.size != 7 or not np.all(np.isfinite(state)):
            raise RuntimeError(f"{role} RPC returned an invalid seven-value state")
        buttons = np.asarray(
            result.get("buttons", np.zeros(1)), dtype=np.float64
        ).reshape(-1)
        if buttons.size == 0 or not np.all(np.isfinite(buttons)):
            raise RuntimeError(f"{role} RPC returned an invalid button state")
        try:
            sampled_at_ns = rpc_integer(
                result.get("sampled_at_ns"), "sampled_at_ns", minimum=1
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{role} RPC returned an invalid sample timestamp"
            ) from exc

        if result.get("phase") == FAULT:
            raise RuntimeError(
                f"{role} worker watchdog fault: {result.get('fault_reason')}"
            )
        try:
            last_applied = rpc_integer(
                result.get("last_applied_sequence", -1),
                "last_applied_sequence",
                minimum=-1,
            )
        except ValueError as exc:
            raise RuntimeError(f"{role} RPC returned an invalid sequence") from exc
        return state, buttons, sampled_at_ns, last_applied

    def _read_states(
        self,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[tuple[str, str], int],
    ]:
        leader_states: dict[str, np.ndarray] = {}
        follower_states: dict[str, np.ndarray] = {}
        leader_buttons: dict[str, np.ndarray] = {}
        applied_sequences: dict[tuple[str, str], int] = {}
        max_age_ns = int(self.config.control.feedback_timeout_s * 1e9)
        endpoints = tuple(
            (side, role, clients[side])
            for side in SIDES
            for role, clients in (
                ("leader", self.leaders),
                ("follower", self.followers),
            )
        )
        pending = {
            (side, role): client.start_get_sample()
            for side, role, client in endpoints
        }
        results = {
            (side, role): client.finish_get_sample(pending[(side, role)])
            for side, role, client in endpoints
        }

        for side in SIDES:
            leader_state, buttons, sampled_at_ns, applied = self._sample(
                results[(side, "leader")], "leader"
            )
            if time.monotonic_ns() - sampled_at_ns > max_age_ns:
                raise RuntimeError(f"{side} leader feedback is stale")
            leader_states[side] = leader_state
            leader_buttons[side] = buttons
            applied_sequences[(side, "leader")] = applied

            follower_state, _, sampled_at_ns, applied = self._sample(
                results[(side, "follower")], "follower"
            )
            if time.monotonic_ns() - sampled_at_ns > max_age_ns:
                raise RuntimeError(f"{side} follower feedback is stale")
            follower_states[side] = follower_state
            applied_sequences[(side, "follower")] = applied
        return leader_states, follower_states, leader_buttons, applied_sequences

    def _alignment_cancel_requested(
        self, leader_buttons: Mapping[str, np.ndarray]
    ) -> bool:
        rising = []
        for side in SIDES:
            raw = bool(leader_buttons[side][0] > 0.5)
            _, pressed = self._control_buttons[side].update(raw)
            rising.append(pressed)
        return any(rising)

    def _handle_episode_buttons(
        self, leader_buttons: Mapping[str, np.ndarray]
    ) -> None:
        button_down = {
            side: bool(buttons.size > 1 and buttons[1] > 0.5)
            for side, buttons in leader_buttons.items()
        }
        pressed = {
            side: button_down[side] and not self._episode_button_latched[side]
            for side in SIDES
        }
        self._episode_button_latched.update(button_down)
        if self.episode_signal is None:
            return
        if pressed["left"]:
            self.episode_signal("discard")
            LOGGER.info("Requested active episode discard from left second button")
        elif pressed["right"]:
            self.episode_signal("toggle")
            LOGGER.info("Requested episode start/finish from right second button")

    def _control_button_action(
        self, leader_buttons: Mapping[str, np.ndarray]
    ) -> str | None:
        now = self.clock()
        down: dict[str, bool] = {}
        rising: dict[str, bool] = {}
        for side in SIDES:
            raw = bool(leader_buttons[side][0] > 0.5)
            down[side], rising[side] = self._control_buttons[side].update(raw)

        if not any(down.values()):
            self._transition_armed = True
            self._chord_press_times.clear()
            return None
        if not self._transition_armed:
            return None
        if self.active and any(rising.values()):
            self._transition_armed = False
            return "deactivate"

        for side in SIDES:
            if rising[side]:
                self._chord_press_times[side] = now
        if all(down.values()) and set(self._chord_press_times) == set(SIDES):
            spread = abs(
                self._chord_press_times["left"]
                - self._chord_press_times["right"]
            )
            self._transition_armed = False
            if spread <= self.config.control.enter_chord_window_s:
                return "activate"
            self._chord_press_times.clear()
            return None
        if self._chord_press_times and now - min(
            self._chord_press_times.values()
        ) > self.config.control.enter_chord_window_s:
            self._transition_armed = False
            self._chord_press_times.clear()
        return None

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _map_follower_state(self, side: str, state: np.ndarray) -> np.ndarray:
        directions = np.asarray(
            self.config.arms[side].follower.joint_directions,
            dtype=np.float64,
        )
        return np.asarray(state, dtype=np.float64) * directions

    def _submit_commands(
        self,
        leader_targets: Mapping[str, np.ndarray],
        follower_targets: Mapping[str, np.ndarray],
        phase: str,
    ) -> int:
        if self._session_id is None:
            raise RuntimeError("no control session is active")
        sequence = self._next_sequence()
        issued_at_ns = time.monotonic_ns()
        for side in SIDES:
            self.followers[side].submit_control_frame(
                {
                    "session_id": self._session_id,
                    "sequence": sequence,
                    "issued_at_ns": issued_at_ns,
                    "phase": phase,
                    "joint_position": follower_targets[side],
                }
            )
            self.leaders[side].submit_control_frame(
                {
                    "session_id": self._session_id,
                    "sequence": sequence,
                    "issued_at_ns": issued_at_ns,
                    "phase": phase,
                    "joint_position": leader_targets[side][:6],
                }
            )
        return sequence

    def _begin_session(self) -> None:
        self._session_id = uuid.uuid4().hex
        self._sequence = -1
        begun: list[RpcEndpointClient] = []
        try:
            for clients in (self.followers, self.leaders):
                for side in SIDES:
                    clients[side].begin_control_session(self._session_id)
                    begun.append(clients[side])
        except Exception:
            for client in begun:
                try:
                    client.end_control_session(self._session_id)
                except Exception:
                    pass
            self._session_id = None
            raise

    def _align_followers(
        self,
        leader_targets: Mapping[str, np.ndarray],
        follower_starts: Mapping[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        follower_targets = {
            side: self._map_follower_state(side, leader_targets[side])
            for side in SIDES
        }
        max_distance = max(
            float(
                np.max(
                    np.abs(
                        follower_targets[side][:ARM_DOF]
                        - follower_starts[side][:ARM_DOF]
                    )
                )
            )
            for side in SIDES
        )
        # Cubic smoothstep reaches 1.5 times its average velocity at midpoint.
        speed_limited_duration = (
            1.5
            * max_distance
            / self.config.control.alignment_max_velocity_rad_s
        )
        duration = max(
            self.config.control.alignment_duration_s,
            speed_limited_duration,
        )
        steps = max(1, int(round(duration * self.config.control.frequency_hz)))
        interval = duration / steps if duration > 0.0 else 0.0
        LOGGER.info(
            "Aligning followers over %.2f seconds (max distance %.3f rad, "
            "velocity limit %.3f rad/s) while leaders hold position",
            duration,
            max_distance,
            self.config.control.alignment_max_velocity_rad_s,
        )
        for step in range(1, steps + 1):
            _, _, leader_buttons, _ = self._read_states()
            if self._alignment_cancel_requested(leader_buttons):
                raise _AlignmentCancelled("top button pressed during alignment")
            linear_alpha = step / steps
            alpha = linear_alpha * linear_alpha * (3.0 - 2.0 * linear_alpha)
            targets = {
                side: follower_starts[side] * (1.0 - alpha)
                + follower_targets[side] * alpha
                for side in SIDES
            }
            self._submit_commands(leader_targets, targets, ALIGNING)
            self.health_check()
            if interval > 0.0:
                self.sleep(interval)

        deadline = self.clock() + self.config.control.alignment_settle_timeout_s
        stable_samples = 0
        while True:
            (
                leader_states,
                follower_states,
                leader_buttons,
                applied,
            ) = self._read_states()
            if self._alignment_cancel_requested(leader_buttons):
                raise _AlignmentCancelled("top button pressed during alignment")
            all_applied = all(value >= 0 for value in applied.values())
            leader_errors = {
                side: float(
                    np.max(
                        np.abs(
                            leader_states[side][:ARM_DOF]
                            - leader_targets[side][:ARM_DOF]
                        )
                    )
                )
                for side in SIDES
            }
            follower_errors = {
                side: float(
                    np.max(
                        np.abs(
                            follower_states[side][:ARM_DOF]
                            - follower_targets[side][:ARM_DOF]
                        )
                    )
                )
                for side in SIDES
            }
            aligned = all(
                max(leader_errors[side], follower_errors[side])
                <= self.config.control.alignment_tolerance
                for side in SIDES
            )
            if all_applied and aligned:
                stable_samples += 1
                if stable_samples >= self.config.control.alignment_stable_samples:
                    return leader_states, follower_states
            else:
                stable_samples = 0
            if self.clock() >= deadline:
                detail = ", ".join(
                    f"{side} leader error={leader_errors[side]:.4f} rad, "
                    f"follower error={follower_errors[side]:.4f} rad"
                    for side in SIDES
                )
                raise TimeoutError(
                    "leaders and followers did not settle at the frozen targets "
                    f"({detail}, tolerance="
                    f"{self.config.control.alignment_tolerance:.4f} rad)"
                )
            self._submit_commands(leader_targets, follower_targets, ALIGNING)
            self.sleep(1.0 / self.config.control.frequency_hz)

    def _activate(
        self,
        leader_states: Mapping[str, np.ndarray],
        follower_states: Mapping[str, np.ndarray],
    ) -> None:
        frozen_targets = {side: leader_states[side].copy() for side in SIDES}
        self._begin_session()
        try:
            settled_leaders, settled_followers = self._align_followers(
                frozen_targets, follower_states
            )
            assert self._session_id is not None
            for clients in (self.followers, self.leaders):
                for side in SIDES:
                    clients[side].set_phase(self._session_id, ACTIVE)
            self.active = True
            self._submit_commands(
                {
                    side: self._map_follower_state(side, settled_followers[side])
                    for side in SIDES
                },
                {
                    side: self._map_follower_state(side, settled_leaders[side])
                    for side in SIDES
                },
                ACTIVE,
            )
        except _AlignmentCancelled as exc:
            failures = self._abort_session()
            LOGGER.info("Teleoperation disabled: %s", exc)
            if failures:
                raise RuntimeError(
                    "one or more workers failed to enter safe hold"
                ) from failures[0]
            return
        except Exception:
            self._abort_session()
            raise
        LOGGER.info("Teleoperation enabled")

    def _abort_session(self) -> list[Exception]:
        session_id = self._session_id
        self.active = False
        self._session_id = None
        if session_id is None:
            return []
        failures: list[Exception] = []
        for clients in (self.followers, self.leaders):
            for side in SIDES:
                try:
                    clients[side].end_control_session(session_id)
                except Exception as exc:
                    failures.append(exc)
                    LOGGER.warning(
                        "Could not end %s worker session for %s: %s",
                        side,
                        type(clients[side]).__name__,
                        exc,
                    )
        return failures

    def _deactivate(self) -> None:
        failures = self._abort_session()
        LOGGER.info("Teleoperation disabled")
        if failures:
            raise RuntimeError(
                "one or more workers failed to enter safe hold"
            ) from failures[0]

    def step(self) -> None:
        try:
            self.health_check()
            leader_states, follower_states, leader_buttons, _ = self._read_states()
            if self.state_publisher is not None:
                try:
                    self.state_publisher.publish(leader_states, follower_states)
                except Exception as exc:
                    LOGGER.warning("Could not publish I2RT state SHM: %s", exc)
            self._handle_episode_buttons(leader_buttons)
            action = self._control_button_action(leader_buttons)
            if action == "activate":
                self._activate(leader_states, follower_states)
                return
            elif action == "deactivate":
                self._deactivate()
                return

            if self.active:
                self._submit_commands(
                    {
                        side: self._map_follower_state(side, follower_states[side])
                        for side in SIDES
                    },
                    {
                        side: self._map_follower_state(side, leader_states[side])
                        for side in SIDES
                    },
                    ACTIVE,
                )
        except Exception:
            if self._session_id is not None:
                self._abort_session()
            raise

    def run(self) -> None:
        interval = 1.0 / self.config.control.frequency_hz
        LOGGER.info(
            "Teleoperation ready; press both top buttons to enable, one to disable"
        )
        while True:
            started_at = self.clock()
            self.step()
            remaining = interval - (self.clock() - started_at)
            if remaining > 0.0:
                self.sleep(remaining)

    def close(self) -> None:
        if self._session_id is not None:
            self._abort_session()


def _connect_clients(
    config: TeleoperationConfig,
) -> tuple[dict[str, RpcEndpointClient], dict[str, RpcEndpointClient]]:
    timeout = config.control.cycle_rpc_timeout_s
    leaders = {
        side: RpcEndpointClient(
            config.arms[side].leader.rpc_port, timeout, f"{side} leader"
        )
        for side in SIDES
    }
    followers = {
        side: RpcEndpointClient(
            config.arms[side].follower.rpc_port, timeout, f"{side} follower"
        )
        for side in SIDES
    }
    return leaders, followers


def _wait_for_workers(
    config: TeleoperationConfig,
    processes: WorkerProcesses,
    leaders: Mapping[str, RpcEndpointClient],
    followers: Mapping[str, RpcEndpointClient],
) -> None:
    deadline = time.monotonic() + config.control.startup_timeout_s
    pending = {
        (side, role): client
        for side in SIDES
        for role, client in (
            ("leader", leaders[side]),
            ("follower", followers[side]),
        )
    }
    while pending:
        processes.ensure_running()
        for key, client in tuple(pending.items()):
            try:
                client.ping()
            except Exception:
                continue
            pending.pop(key)
            LOGGER.info("I2RT worker ready: %s/%s", *key)
        if not pending:
            return
        if time.monotonic() >= deadline:
            names = ", ".join(f"{side}/{role}" for side, role in pending)
            raise TimeoutError(f"Timed out waiting for teleoperation workers: {names}")
        time.sleep(0.1)


def run_teleoperation(
    config_path: Path,
    config: TeleoperationConfig,
    *,
    device_name: str = "I2RTRobot",
    leader_device_id: int = 0,
    follower_device_id: int = 1,
    leader_state_feedback: (
        StateFeedbackConfig | Mapping[str, Any] | None
    ) = None,
    follower_state_feedback: (
        StateFeedbackConfig | Mapping[str, Any] | None
    ) = None,
) -> None:
    """Launch workers and run until interrupted or a component fails."""

    processes = WorkerProcesses(config_path, config)
    coordinator = None
    state_publisher = None
    stage = "starting worker processes"
    leaders: dict[str, RpcEndpointClient] = {}
    followers: dict[str, RpcEndpointClient] = {}
    try:
        stage = "validating CAN interfaces"
        _validate_can_interfaces(config)
        app_logger.info("I2RT teleoperation startup: starting four hardware workers")
        stage = "starting worker processes"
        processes.start()
        stage = "connecting to worker RPC endpoints"
        leaders, followers = _connect_clients(config)
        stage = "waiting for worker hardware initialization"
        _wait_for_workers(config, processes, leaders, followers)
        app_logger.info("I2RT teleoperation startup: all hardware workers are ready")
        stage = "creating robot state streams"
        state_publisher = I2RTStatePublisher(
            device_name=device_name,
            leader_device_id=leader_device_id,
            follower_device_id=follower_device_id,
            frequency_hz=config.control.frequency_hz,
            leader_state_feedback=leader_state_feedback,
            follower_state_feedback=follower_state_feedback,
        )
        stage = "running coordinated teleoperation"
        signal_writer = EpisodeSignalWriter(config.episode_signal_pattern)
        coordinator = BimanualCoordinator(
            config,
            leaders,
            followers,
            health_check=processes.ensure_running,
            state_publisher=state_publisher,
            episode_signal=signal_writer.emit,
        )
        coordinator.run()
    except KeyboardInterrupt:
        LOGGER.info("Teleoperation interrupted")
    except Exception as exc:
        app_logger.exception(
            f"I2RT teleoperation failed while {stage}: {type(exc).__name__}: {exc}"
        )
        raise
    finally:
        if coordinator is not None:
            coordinator.close()
        if state_publisher is not None:
            state_publisher.close()

        # Stop the RPC servers before closing their clients. Portal client close
        # can otherwise wait indefinitely when startup failed partway through.
        processes.stop()
        for clients in (followers, leaders):
            for side, client in clients.items():
                try:
                    client.close()
                except Exception as exc:
                    LOGGER.warning(
                        "Could not close %s RPC client during shutdown: %s",
                        side,
                        exc,
                    )
