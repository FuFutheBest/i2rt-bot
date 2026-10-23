from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from devices.robots.robotwin import _LoopTiming, RoboTwinSimRobot
from utils.shm_utils import DEVICE_HEADER_SIZE, unpack_device_header


class _RuntimeStub:
    def __init__(self) -> None:
        self.targets: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.frames = {
            "head_camera": np.zeros((2, 3, 3), dtype=np.uint8),
            "left_camera": np.ones((2, 3, 3), dtype=np.uint8),
            "right_camera": np.full((2, 3, 3), 2, dtype=np.uint8),
        }

    def apply_joint14_target(
        self, target: np.ndarray, velocity: np.ndarray | None = None
    ) -> None:
        self.targets.append(np.asarray(target, dtype=np.float64).copy())
        if velocity is None:
            velocity = np.zeros_like(target)
        self.velocities.append(np.asarray(velocity, dtype=np.float64).copy())

    def should_render_gui(self, sim_step: int) -> bool:
        return sim_step > 0

    def capture_rgb(self, *, render_gui: bool = False) -> dict[str, np.ndarray]:
        return self.frames

    def render_gui_if_due(self, sim_step: int) -> bool:
        return sim_step > 0


class _TraceStub:
    def __init__(self) -> None:
        self.applied: list[dict[str, object]] = []
        self.meta: list[dict[str, object]] = []
        self.logger = self

    def action_applied(self, setpoint: object, **kwargs: object) -> None:
        self.applied.append({"setpoint": setpoint, **kwargs})

    def measured_state(self, **kwargs: object) -> None:
        self.meta.append({"event_type": "sim_measured_state", **kwargs})

    def log_meta(self, event_type: str, **kwargs: object) -> None:
        self.meta.append({"event_type": event_type, **kwargs})


def _robot_for_interpolation() -> tuple[RoboTwinSimRobot, _RuntimeStub, _TraceStub]:
    robot = RoboTwinSimRobot.__new__(RoboTwinSimRobot)
    robot._sim_step = 0
    robot._physics_interval_s = 0.1
    robot._control_interval_steps = 5
    robot._control_interval_s = 0.5
    robot._drive_joint14 = np.zeros(14, dtype=np.float64)
    robot._target_joint14 = np.zeros(14, dtype=np.float64)
    robot._active_segment = None
    robot._last_control_write_timestamp_ns = None
    robot._last_sim_apply_timestamp_ns = None
    robot._trace_timing_interval_s = 1.0
    robot._last_timing_trace_monotonic = 0.0
    robot._realtime_factor_warn_threshold = 0.95
    robot._last_realtime_factor_monotonic = None
    robot._last_realtime_factor_sim_time_ns = None
    robot._last_realtime_factor = None
    robot._realtime_factor_degraded = False
    robot.control_fps = 10.0
    robot._control_accept_count = 0
    robot._first_control_accept_wall_timestamp_ns = None
    robot._last_control_accept_wall_timestamp_ns = None
    robot._first_control_accept_sim_timestamp_ns = None
    robot._last_control_accept_sim_timestamp_ns = None
    robot._last_camera_publish_monotonic_by_name = {}
    robot._runtime_lock = threading.RLock()
    robot._stopped = True
    robot._runtime = _RuntimeStub()
    robot._trace = _TraceStub()
    return robot, robot._runtime, robot._trace


def test_robotwin_sim_clock_name_is_derived_from_its_device_identity() -> None:
    assert (
        RoboTwinSimRobot._derive_sim_clock_shm_name("RoboTwinSimRobot", 7)
        == "RoboTwinSimRobot_7_sim_clock"
    )


def test_robotwin_interpolates_control_target_at_physics_steps() -> None:
    robot, runtime, trace = _robot_for_interpolation()
    setpoint = SimpleNamespace(control_write_timestamp_ns=123)

    robot._queue_joint_target(setpoint, np.ones(14, dtype=np.float64))

    for step in range(5):
        robot._sim_step = step
        robot._apply_interpolated_target()

    np.testing.assert_allclose(
        [target[0] for target in runtime.targets], [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    np.testing.assert_allclose(
        [velocity[0] for velocity in runtime.velocities], [2.0] * 5
    )
    np.testing.assert_allclose(robot._drive_joint14, np.ones(14, dtype=np.float64))
    assert robot._active_segment is None
    assert len(trace.applied) == 1
    assert trace.applied[0]["setpoint"] is setpoint


def test_robotwin_new_target_interpolates_from_current_drive_target() -> None:
    robot, runtime, _trace = _robot_for_interpolation()
    robot._sim_step = 10
    robot._drive_joint14 = np.full(14, 0.4, dtype=np.float64)
    setpoint = SimpleNamespace(control_write_timestamp_ns=456)

    robot._queue_joint_target(setpoint, np.ones(14, dtype=np.float64))
    robot._apply_interpolated_target()

    np.testing.assert_allclose(runtime.targets[-1], np.full(14, 0.52, dtype=np.float64))


def test_robotwin_finishes_due_segment_before_queueing_next_target() -> None:
    robot, runtime, trace = _robot_for_interpolation()
    first = SimpleNamespace(control_write_timestamp_ns=1)
    second = SimpleNamespace(control_write_timestamp_ns=2)

    robot._queue_joint_target(first, np.ones(14, dtype=np.float64))
    robot._sim_step = 4
    robot._queue_joint_target(second, np.full(14, 2.0, dtype=np.float64))

    np.testing.assert_allclose(runtime.targets[-1], np.ones(14, dtype=np.float64))
    assert robot._active_segment is not None
    np.testing.assert_allclose(
        robot._active_segment.start_joint14, np.ones(14, dtype=np.float64)
    )
    assert [event["event_type"] for event in trace.meta] == []


def test_robotwin_finishes_quantized_control_segment_without_superseding() -> None:
    robot, runtime, trace = _robot_for_interpolation()
    robot._physics_interval_s = 1.0 / 250.0
    robot._control_interval_s = 1.0 / 15.0
    first = SimpleNamespace(control_write_timestamp_ns=1)
    second = SimpleNamespace(control_write_timestamp_ns=2)

    robot._queue_joint_target(first, np.ones(14, dtype=np.float64))
    robot._sim_step = 15
    robot._queue_joint_target(second, np.full(14, 2.0, dtype=np.float64))

    np.testing.assert_allclose(runtime.targets[-1], np.ones(14, dtype=np.float64))
    assert robot._active_segment is not None
    np.testing.assert_allclose(
        robot._active_segment.start_joint14, np.ones(14, dtype=np.float64)
    )
    assert [event["event_type"] for event in trace.meta] == []


def test_robotwin_control_header_uses_control_fps() -> None:
    robot, _runtime, _trace = _robot_for_interpolation()
    robot.device_name = "RoboTwinSimRobot"
    robot.device_class_id = 0
    robot.control_fps = 17.0
    robot.control_dtype = "float64"
    robot.control_shape = (14,)
    robot.control_frame_size = 120
    robot.control_buffer_size = 3
    robot.default_control_shm_name = "RoboTwinSimRobot_0_control"
    robot.control_shared_memory = {
        robot.default_control_shm_name: SimpleNamespace(buf=bytearray(1024))
    }

    robot._initialize_control_buffer_header()

    header = unpack_device_header(
        bytes(
            robot.control_shared_memory[robot.default_control_shm_name].buf[
                :DEVICE_HEADER_SIZE
            ]
        )
    )
    assert header["fps"] == 17.0


def test_robotwin_state_snapshot_publishes_measured_feedback() -> None:
    robot, _runtime, trace = _robot_for_interpolation()
    robot._drive_joint14 = np.full(14, 0.25, dtype=np.float64)
    robot._measured_joint14 = np.full(14, -0.5, dtype=np.float64)
    robot._measured_qvel14 = np.zeros(14, dtype=np.float64)
    writes: list[np.ndarray] = []

    def write_state(state: np.ndarray, _timestamp_ns: int) -> None:
        writes.append(state.copy())

    robot._write_array_to_shm_with_timestamp = write_state

    assert robot._publish_state_snapshot(force=True) is True

    np.testing.assert_allclose(writes[-1], np.full(14, -0.5, dtype=np.float64))
    np.testing.assert_allclose(
        trace.meta[-1]["measured_qpos"], np.full(14, -0.5, dtype=np.float64)
    )
    np.testing.assert_allclose(
        trace.meta[-1]["target_qpos"], np.full(14, 0.25, dtype=np.float64)
    )


def test_robotwin_loop_timing_trace_can_be_disabled() -> None:
    robot, _runtime, trace = _robot_for_interpolation()
    robot._trace_timing_interval_s = 0.0

    robot._maybe_trace_loop_timing(_LoopTiming(loop_started_ns=1, loop_total_ns=2))

    assert trace.meta == []


def test_robotwin_loop_timing_trace_records_sections_after_interval() -> None:
    robot, _runtime, trace = _robot_for_interpolation()
    timing = _LoopTiming(
        loop_started_ns=1,
        accept_control_ns=2,
        poll_eef_ns=3,
        apply_target_ns=4,
        scene_step_ns=5,
        gui_render_ns=6,
        read_state_ns=7,
        publish_state_ns=8,
        publish_camera_ns=9,
        runtime_lock_wait_ns=10,
        sleep_requested_ns=11,
        sleep_call_requested_ns=12,
        sleep_ns=10,
        spin_wait_ns=13,
        sleep_overshoot_ns=14,
        loop_total_ns=11,
        state_published=True,
        camera_published=True,
        gui_rendered=True,
        overrun_ns=12,
        catch_up_steps=2,
        catch_up_reset=True,
    )

    with patch("devices.robots.robotwin.time.monotonic", return_value=2.0):
        robot._maybe_trace_loop_timing(timing)

    assert trace.meta == [
        {
            "event_type": "sim_loop_timing",
            "accept_control_ns": 2,
            "poll_eef_ns": 3,
            "apply_target_ns": 4,
            "scene_step_ns": 5,
            "gui_render_ns": 6,
            "read_state_ns": 7,
            "publish_state_ns": 8,
            "publish_camera_ns": 9,
            "runtime_lock_wait_ns": 10,
            "sleep_requested_ns": 11,
            "sleep_call_requested_ns": 12,
            "sleep_ns": 10,
            "spin_wait_ns": 13,
            "sleep_overshoot_ns": 14,
            "loop_total_ns": 11,
            "state_published": True,
            "camera_published": True,
            "gui_rendered": True,
            "overrun_ns": 12,
            "catch_up_steps": 2,
            "catch_up_reset": True,
            "sim_step": 0,
            "sim_time_ns": 0,
            "realtime_factor": None,
            "realtime_factor_warn_threshold": 0.95,
            "control_rate_basis": "sim_control_accepted",
            "control_accept_count": 0,
            "configured_control_fps": 10.0,
            "effective_control_hz_wall": None,
            "effective_control_hz_sim": None,
        }
    ]


def test_robotwin_realtime_factor_is_computed_from_sim_and_wall_deltas() -> None:
    robot, _runtime, trace = _robot_for_interpolation()
    robot._sim_step = 10
    with patch("devices.robots.robotwin.time.monotonic", return_value=1.0):
        robot._maybe_trace_loop_timing(_LoopTiming(loop_started_ns=1))

    robot._sim_step = 20
    with patch("devices.robots.robotwin.time.monotonic", return_value=2.0):
        robot._maybe_trace_loop_timing(_LoopTiming(loop_started_ns=2))

    assert trace.meta[-1]["realtime_factor"] == 1.0


def test_robotwin_camera_snapshot_sync_writes_due_streams() -> None:
    robot, _runtime, _trace = _robot_for_interpolation()
    spec = SimpleNamespace(
        camera_name="head_camera",
        data_shape=(2, 3, 3),
        fps=30.0,
    )
    writes: list[tuple[np.ndarray, int]] = []

    def write_frame(frame: np.ndarray, timestamp_ns: int) -> None:
        writes.append((frame.copy(), timestamp_ns))

    robot._camera_specs = (spec,)
    robot._camera_devices = {
        "head_camera": SimpleNamespace(_write_array_to_shm_with_timestamp=write_frame)
    }
    robot._last_camera_publish_monotonic_by_name = {"head_camera": 0.0}

    published, capture_ns, write_ns, wait_s = robot._publish_camera_snapshot_sync(force=True)

    assert published is True
    assert capture_ns >= 0
    assert write_ns >= 0
    assert wait_s >= 0.0

    assert len(writes) == 1
    np.testing.assert_array_equal(writes[0][0], np.zeros((2, 3, 3), dtype=np.uint8))
    assert writes[0][1] > 0


def test_robotwin_camera_snapshot_publishes_synchronized_stream_set() -> None:
    robot, runtime, _trace = _robot_for_interpolation()
    specs = (
        SimpleNamespace(camera_name="head_camera", data_shape=(2, 3, 3), fps=6.0),
        SimpleNamespace(camera_name="left_camera", data_shape=(2, 3, 3), fps=6.0),
        SimpleNamespace(camera_name="right_camera", data_shape=(2, 3, 3), fps=6.0),
    )
    writes: dict[str, list[tuple[np.ndarray, int]]] = {spec.camera_name: [] for spec in specs}

    def writer(name: str):
        def write_frame(frame: np.ndarray, timestamp_ns: int) -> None:
            writes[name].append((frame.copy(), timestamp_ns))

        return write_frame

    robot._camera_specs = specs
    robot._camera_devices = {
        spec.camera_name: SimpleNamespace(
            _write_array_to_shm_with_timestamp=writer(spec.camera_name)
        )
        for spec in specs
    }
    robot._last_camera_publish_monotonic_by_name = {
        "head_camera": 1.0,
        "left_camera": 1.001,
        "right_camera": 1.002,
    }

    with patch("devices.robots.robotwin.time.monotonic", return_value=1.2):
        published, _capture_ns, _write_ns, wait_s = robot._publish_camera_snapshot_sync(force=False)

    assert published is True
    assert wait_s > 0.0
    assert len(runtime.targets) == 0
    assert {name: len(name_writes) for name, name_writes in writes.items()} == {
        "head_camera": 1,
        "left_camera": 1,
        "right_camera": 1,
    }
    timestamps = {name_writes[0][1] for name_writes in writes.values()}
    assert len(timestamps) == 1


def test_robotwin_wait_until_tick_allows_bounded_catchup() -> None:
    robot, _runtime, _trace = _robot_for_interpolation()
    robot._max_catch_up_steps = 2
    timing = _LoopTiming(loop_started_ns=1)

    with patch("devices.robots.robotwin.time.monotonic", side_effect=[10.0, 10.1]):
        next_tick, catch_up_steps = robot._wait_until_next_physics_tick(9.9, timing, 1)

    assert next_tick == 10.1
    assert catch_up_steps == 0
    assert timing.catch_up_reset is True
    assert timing.overrun_ns > 0
