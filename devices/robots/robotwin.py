"""RoboTwin/SAPIEN bridge for SuperInference asynchronous policy evaluation."""

from __future__ import annotations

import subprocess
import sys
import time
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from devices.base import BaseDevice
from devices.config import StateFeedbackConfig
from devices.contracts import DeviceKind
from devices.robots.base import BaseRobot
from integrations.robotwin.contracts import (
    CameraStreamSpec,
    action_schema_dim,
    parse_camera_streams,
    validate_action,
)
from integrations.robotwin.runtime import RoboTwinDirectRuntime
from integrations.robotwin.setpoint import (
    AsyncEefResolver,
    LatestSetpoint,
    load_eef_resolver,
)
from integrations.robotwin.trace import RoboTwinTrace
from utils.logger_config import logger
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    create_shared_memory,
    pack_buffer_header,
    pack_device_header,
)


@dataclass
class _JointTargetSegment:
    setpoint: Any
    start_joint14: np.ndarray
    target_joint14: np.ndarray
    velocity_joint14: np.ndarray
    start_sim_time_s: float
    duration_s: float
    trace_logged: bool = False


@dataclass
class _LoopTiming:
    loop_started_ns: int
    accept_control_ns: int = 0
    poll_eef_ns: int = 0
    apply_target_ns: int = 0
    scene_step_ns: int = 0
    gui_render_ns: int = 0
    read_state_ns: int = 0
    publish_state_ns: int = 0
    publish_camera_ns: int = 0
    runtime_lock_wait_ns: int = 0
    sleep_requested_ns: int = 0
    sleep_call_requested_ns: int = 0
    sleep_ns: int = 0
    spin_wait_ns: int = 0
    sleep_overshoot_ns: int = 0
    loop_total_ns: int = 0
    state_published: bool = False
    camera_published: bool = False
    gui_rendered: bool = False
    overrun_ns: int = 0
    catch_up_steps: int = 0
    catch_up_reset: bool = False


class RoboTwinCamera(BaseDevice):
    """Descriptor class for a camera stream owned by :class:`RoboTwinSimRobot`.

    These streams are deliberately not independently startable: a single
    SAPIEN process must own the task, cameras, GUI, and physics scene.
    """

    device_kind = DeviceKind.CAMERA

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        managed_by: Optional[str] = None,
    ) -> None:
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        self.managed_by = managed_by

    def start_server(self) -> None:
        owner = self.managed_by or "RoboTwinSimRobot"
        raise RuntimeError(
            f"{self.device_name}_{self.device_class_id} is a virtual stream owned by {owner}; "
            "start the RoboTwin bridge instead"
        )


class RoboTwinSimRobot(BaseRobot):
    """One process that owns RoboTwin simulation, GUI, state, and camera SHMs.

    The bridge bypasses :meth:`BaseRobot.start_server` because SAPIEN physics,
    state publishing, and camera rendering must be scheduled explicitly from one
    owner process.  Control commands are read from SHM by the physics loop,
    converted into time-bounded joint target segments, and applied at
    ``physics_fps``.
    """

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: Mapping[str, Any],
        control: Mapping[str, Any],
        physics_fps: float = 250.0,
        robotwin_root: str = "third_party/RoboTwin",
        task_name: str = "beat_block_hammer",
        task_config: str = "demo_clean",
        seed: int = 0,
        enable_gui: bool = False,
        gui_render_fps: float = 30.0,
        renderer_shader_dir: str = "default",
        viewer_shader_dir: str = "minimal",
        viewer_resolution: tuple[int, int] | list[int] = (640, 480),
        renderer_max_num_materials: int = 128,
        renderer_max_num_textures: int = 512,
        joint_stiffness: float | None = None,
        joint_damping: float | None = None,
        gripper_stiffness: float | None = None,
        gripper_damping: float | None = None,
        camera_streams: Optional[list[dict[str, Any]]] = None,
        eef_resolver: Optional[str] = None,
        trace_log_dir: str = "logs/action_state_xyz",
        trace_visualizer: Optional[dict[str, Any]] = None,
        trace_timing_interval_s: float = 1.0,
        realtime_factor_warn_threshold: float = 0.95,
        physics_sleep_spin_margin_s: float = 0.002,
        max_catch_up_steps: int = 4,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            control=control,
        )

        self.control_type = self.control_config.type
        if self.control_type == "joints":
            self.action_schema = "joint14"
        elif self.control_type == "eef_abs":
            self.action_schema = "eef16"
        else:
            raise ValueError(
                "RoboTwinSimRobot supports only control.type=joints or eef_abs"
            )
        expected_control_dim = action_schema_dim(self.action_schema)
        if self.control_shape != (expected_control_dim,):
            raise ValueError(
                f"RoboTwin {self.action_schema} requires control.format.shape "
                f"[{expected_control_dim}], got {self.control_shape}"
            )
        if self.data_shape != (14,):
            raise ValueError(
                "RoboTwin state_feedback.format.shape must be [14] measured joints"
            )
        state_primitives = tuple(
            primitive.name for primitive in self.state_feedback_config.format.primitives
        )
        control_primitives = tuple(
            primitive.name for primitive in self.control_config.format.primitives
        )
        if state_primitives != ("j1-7", "j1-7"):
            raise ValueError(
                "RoboTwin state_feedback.format.type must be [j1-7, j1-7]"
            )
        expected_control_primitives = (
            ("j1-7", "j1-7")
            if self.action_schema == "joint14"
            else (
                "xyz",
                "qxqyqzqw",
                "gripper",
                "xyz",
                "qxqyqzqw",
                "gripper",
            )
        )
        if control_primitives != expected_control_primitives:
            raise ValueError(
                f"RoboTwin control.type={self.control_type} conflicts with "
                f"control.format.type={list(control_primitives)}"
            )
        if float(physics_fps) <= 0:
            raise ValueError("RoboTwin physics_fps must be positive")
        if float(physics_fps) < self.fps:
            raise ValueError(
                "RoboTwin physics_fps must be at least the robot state fps"
            )
        if float(physics_fps) < self.command_fps:
            raise ValueError(
                "RoboTwin physics_fps must be at least the control_fps"
            )
        if not 0 < float(realtime_factor_warn_threshold) <= 1:
            raise ValueError(
                "RoboTwin realtime_factor_warn_threshold must be in (0, 1]"
            )

        if np.dtype(self.control_dtype) != np.dtype(np.float64):
            raise ValueError("RoboTwin control.format.dtype must be float64")
        if np.dtype(self.data_dtype) != np.dtype(np.float64):
            raise ValueError("RoboTwin state data_dtype must be float64")
        self.physics_fps = float(physics_fps)
        self._physics_interval_s = 1.0 / self.physics_fps
        self.control_fps = self.command_fps
        self._control_interval_s = 1.0 / self.control_fps
        self._physics_sleep_spin_margin_s = max(
            0.0, float(physics_sleep_spin_margin_s)
        )
        self._max_catch_up_steps = max(0, int(max_catch_up_steps))
        self.robotwin_root = str(robotwin_root)
        self.task_name = str(task_name)
        self.task_config = str(task_config)
        self.seed = int(seed)
        self.enable_gui = bool(enable_gui)
        self.gui_render_fps = float(gui_render_fps)
        self.renderer_shader_dir = str(renderer_shader_dir).strip()
        if not self.renderer_shader_dir:
            raise ValueError("RoboTwin renderer_shader_dir must not be empty")
        self.viewer_shader_dir = str(viewer_shader_dir).strip()
        if not self.viewer_shader_dir:
            raise ValueError("RoboTwin viewer_shader_dir must not be empty")
        self.viewer_resolution = tuple(int(value) for value in viewer_resolution)
        if len(self.viewer_resolution) != 2 or min(self.viewer_resolution) <= 0:
            raise ValueError(
                "RoboTwin viewer_resolution must contain two positive values"
            )
        self.renderer_max_num_materials = int(renderer_max_num_materials)
        self.renderer_max_num_textures = int(renderer_max_num_textures)
        if self.renderer_max_num_materials <= 0 or self.renderer_max_num_textures <= 0:
            raise ValueError(
                "RoboTwin renderer material and texture capacities must be positive"
            )
        self._camera_specs: tuple[CameraStreamSpec, ...] = parse_camera_streams(
            camera_streams,
            default_fps=self.fps,
            default_buffer_size=self.state_feedback_config.buffer_size,
        )
        for spec in self._camera_specs:
            if spec.fps <= 0:
                raise ValueError(
                    f"RoboTwin camera {spec.camera_name!r} fps must be positive"
                )
            if spec.fps > self.physics_fps:
                raise ValueError(
                    f"RoboTwin camera {spec.camera_name!r} fps must not exceed physics_fps"
                )
        self._camera_devices: dict[str, BaseDevice] = {}

        self._setpoints = LatestSetpoint(self.action_schema)
        self._eef_resolver: Optional[AsyncEefResolver] = None
        if self.action_schema == "eef16":
            if not eef_resolver:
                raise ValueError(
                    "eef16 requires eef_resolver='package.module:callable'; the resolver must return joint14"
                )
            self._eef_resolver = AsyncEefResolver(load_eef_resolver(eef_resolver))
        elif eef_resolver:
            raise ValueError("eef_resolver is only valid with action_schema='eef16'")
        self._pending_eef_setpoints: dict[int, Any] = {}

        self._runtime = RoboTwinDirectRuntime(
            robotwin_root=self.robotwin_root,
            task_name=self.task_name,
            task_config=self.task_config,
            seed=self.seed,
            physics_fps=self.physics_fps,
            enable_gui=self.enable_gui,
            gui_render_fps=self.gui_render_fps,
            renderer_shader_dir=self.renderer_shader_dir,
            viewer_shader_dir=self.viewer_shader_dir,
            viewer_resolution=self.viewer_resolution,
            renderer_max_num_materials=self.renderer_max_num_materials,
            renderer_max_num_textures=self.renderer_max_num_textures,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            gripper_stiffness=gripper_stiffness,
            gripper_damping=gripper_damping,
        )
        trace_dir = str(Path(trace_log_dir).expanduser().resolve())
        self._trace = RoboTwinTrace(
            f"{self.device_name}_{self.device_class_id}_bridge", log_dir=trace_dir
        )
        self._trace_visualizer = dict(trace_visualizer or {})
        self._trace_timing_interval_s = max(0.0, float(trace_timing_interval_s))
        self._realtime_factor_warn_threshold = float(realtime_factor_warn_threshold)
        self._last_timing_trace_monotonic = 0.0
        self._last_realtime_factor_monotonic: Optional[float] = None
        self._last_realtime_factor_sim_time_ns: Optional[int] = None
        self._last_realtime_factor: Optional[float] = None
        self._realtime_factor_degraded = False
        self._sim_step = 0
        self._target_joint14 = np.zeros(14, dtype=np.float64)
        self._drive_joint14 = np.zeros(14, dtype=np.float64)
        self._measured_joint14 = np.zeros(14, dtype=np.float64)
        self._measured_qvel14 = np.zeros(14, dtype=np.float64)
        self._active_segment: Optional[_JointTargetSegment] = None
        self._last_control_write_timestamp_ns: Optional[int] = None
        self._last_sim_apply_timestamp_ns: Optional[int] = None
        self._control_accept_count = 0
        self._first_control_accept_wall_timestamp_ns: Optional[int] = None
        self._last_control_accept_wall_timestamp_ns: Optional[int] = None
        self._first_control_accept_sim_timestamp_ns: Optional[int] = None
        self._last_control_accept_sim_timestamp_ns: Optional[int] = None
        self._last_state_publish_monotonic = 0.0
        self._last_camera_publish_monotonic_by_name = {
            spec.camera_name: 0.0 for spec in self._camera_specs
        }
        self._runtime_lock = threading.RLock()
        self._camera_stop_event = threading.Event()
        self._camera_thread: Optional[threading.Thread] = None
        self._sim_clock_shm_name = self._derive_sim_clock_shm_name(
            self.device_name, self.device_class_id
        )
        self._sim_clock_shm: Optional[Any] = None
        self._stopped = False

    @staticmethod
    def _derive_sim_clock_shm_name(device_name: str, device_class_id: int) -> str:
        """Return the bridge-owned sim-time channel for one robot instance."""
        return f"{device_name}_{device_class_id}_sim_clock"

    def _request_shutdown(self, signum: int, _frame: Any) -> None:
        """Leave the physics loop and let ``start_server`` run its cleanup."""
        logger.info(
            f"RoboTwin bridge received signal {signum}; requesting graceful shutdown"
        )
        self.running = False

    def _install_shutdown_signal_handlers(self) -> dict[int, Any]:
        """Use bridge-local handlers instead of an inherited starter handler."""
        if threading.current_thread() is not threading.main_thread():
            return {}
        previous: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, self._request_shutdown)
        return previous

    @staticmethod
    def _restore_shutdown_signal_handlers(previous: dict[int, Any]) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def get_robot_state(self) -> np.ndarray:
        """Return measured joint feedback, not the current drive target."""
        return self._measured_joint14.copy()

    @staticmethod
    def _effective_frequency_hz(
        count: int,
        first_timestamp_ns: Optional[int],
        last_timestamp_ns: Optional[int],
    ) -> Optional[float]:
        if count < 2 or first_timestamp_ns is None or last_timestamp_ns is None:
            return None
        elapsed_ns = int(last_timestamp_ns) - int(first_timestamp_ns)
        if elapsed_ns <= 0:
            return None
        return float(count - 1) * 1_000_000_000.0 / float(elapsed_ns)

    def _record_control_acceptance_metrics(
        self,
        timestamp_ns: int,
        sim_time_ns: int,
    ) -> dict[str, Any]:
        """Measure the cadence of new high-level targets accepted by the bridge."""
        self._control_accept_count = getattr(self, "_control_accept_count", 0) + 1
        if getattr(self, "_first_control_accept_wall_timestamp_ns", None) is None:
            self._first_control_accept_wall_timestamp_ns = int(timestamp_ns)
        self._last_control_accept_wall_timestamp_ns = int(timestamp_ns)
        if getattr(self, "_first_control_accept_sim_timestamp_ns", None) is None:
            self._first_control_accept_sim_timestamp_ns = int(sim_time_ns)
        self._last_control_accept_sim_timestamp_ns = int(sim_time_ns)
        count = int(self._control_accept_count)
        return {
            "control_rate_basis": "sim_control_accepted",
            "control_accept_count": count,
            "configured_control_fps": float(self.control_fps),
            "effective_control_hz_wall": self._effective_frequency_hz(
                count,
                self._first_control_accept_wall_timestamp_ns,
                self._last_control_accept_wall_timestamp_ns,
            ),
            "effective_control_hz_sim": self._effective_frequency_hz(
                count,
                self._first_control_accept_sim_timestamp_ns,
                self._last_control_accept_sim_timestamp_ns,
            ),
        }

    def _control_metrics_snapshot(self) -> dict[str, Any]:
        count = int(getattr(self, "_control_accept_count", 0))
        return {
            "control_rate_basis": "sim_control_accepted",
            "control_accept_count": count,
            "configured_control_fps": float(getattr(self, "control_fps", 0.0)),
            "effective_control_hz_wall": self._effective_frequency_hz(
                count,
                getattr(self, "_first_control_accept_wall_timestamp_ns", None),
                getattr(self, "_last_control_accept_wall_timestamp_ns", None),
            ),
            "effective_control_hz_sim": self._effective_frequency_hz(
                count,
                getattr(self, "_first_control_accept_sim_timestamp_ns", None),
                getattr(self, "_last_control_accept_sim_timestamp_ns", None),
            ),
        }

    def _update_realtime_factor(
        self,
        now_monotonic: float,
        sim_time_ns: int,
    ) -> Optional[float]:
        """Return sim-time progress divided by wall-time progress."""
        previous_wall = getattr(self, "_last_realtime_factor_monotonic", None)
        previous_sim = getattr(self, "_last_realtime_factor_sim_time_ns", None)
        self._last_realtime_factor_monotonic = float(now_monotonic)
        self._last_realtime_factor_sim_time_ns = int(sim_time_ns)
        if previous_wall is None or previous_sim is None:
            return None

        wall_delta_s = float(now_monotonic) - float(previous_wall)
        sim_delta_s = (int(sim_time_ns) - int(previous_sim)) / 1_000_000_000.0
        if wall_delta_s <= 0 or sim_delta_s < 0:
            return None
        realtime_factor = sim_delta_s / wall_delta_s
        self._last_realtime_factor = float(realtime_factor)
        threshold = float(getattr(self, "_realtime_factor_warn_threshold", 0.95))
        was_degraded = bool(getattr(self, "_realtime_factor_degraded", False))
        upper_threshold = max(1.0, 2.0 - threshold)
        is_degraded = realtime_factor < threshold or realtime_factor > upper_threshold
        if is_degraded and not was_degraded:
            logger.warning(
                "RoboTwin realtime factor left the allowed range: "
                f"sim/wall={realtime_factor:.3f}, "
                f"allowed=[{threshold:.3f}, {upper_threshold:.3f}]"
            )
        elif was_degraded and not is_degraded:
            logger.info(
                "RoboTwin realtime factor recovered: "
                f"sim/wall={realtime_factor:.3f}, "
                f"allowed=[{threshold:.3f}, {upper_threshold:.3f}]"
            )
        self._realtime_factor_degraded = is_degraded
        return float(realtime_factor)

    def accept_control_frame(
        self, action: Any, *, control_write_timestamp_ns: int
    ) -> bool:
        """Record a newer control target; returns before any physical motion finishes."""
        setpoint = self._setpoints.accept(action, control_write_timestamp_ns)
        if setpoint is None:
            return False
        now_ns = time.time_ns()
        sim_time_ns = self._sim_time_ns()
        self._trace.control_accepted(
            setpoint,
            timestamp_ns=now_ns,
            sim_step=self._sim_step,
            sim_time_ns=sim_time_ns,
            control_metrics=self._record_control_acceptance_metrics(
                now_ns,
                sim_time_ns,
            ),
        )
        if self.action_schema == "joint14":
            self._queue_joint_target(setpoint, setpoint.action)
        else:
            assert self._eef_resolver is not None
            self._pending_eef_setpoints = {setpoint.sequence: setpoint}
            self._eef_resolver.submit(setpoint, self._measured_joint14)
        return True

    def start_server(self) -> None:
        """Create all SHMs, start the direct-drive task, and run the physics loop."""
        if self.running:
            logger.info("RoboTwin bridge is already running")
            return
        self._stopped = False
        self.running = True
        previous_signal_handlers = self._install_shutdown_signal_handlers()
        try:
            self._create_shared_memory()
            self._create_camera_streams()
            self._create_sim_clock_shared_memory()
            # Resolve and create the configured source topology.  The bridge
            # still owns its canonical command channel, while external source
            # endpoints are attached read-only by BaseRobot.
            self._prepare_control_loop_resources()
            self._runtime.start()
            with self._runtime_lock:
                self._measured_joint14, self._measured_qvel14 = (
                    self._runtime.read_joint14_state()
                )
            self._target_joint14 = self._measured_joint14.copy()
            self._drive_joint14 = self._measured_joint14.copy()
            self._publish_state_snapshot(force=True)
            self._publish_camera_snapshot_sync(force=True)
            self._start_camera_publisher()
            self._trace.logger.log_meta(
                "sim_bridge_started",
                action_schema=self.action_schema,
                task_name=self.task_name,
                task_config=self.task_config,
                physics_fps=self.physics_fps,
                state_fps=self.fps,
                control_fps=self.control_fps,
                trace_timing_interval_s=self._trace_timing_interval_s,
                realtime_factor_warn_threshold=self._realtime_factor_warn_threshold,
                realtime_factor_definition="sim_time_delta_over_wall_monotonic_delta",
                sim_clock_shm_name=self._sim_clock_shm_name,
                camera_fps_by_stream={
                    spec.camera_name: spec.fps for spec in self._camera_specs
                },
                control_shm_name=self.default_control_shm_name,
                camera_streams=[spec.camera_name for spec in self._camera_specs],
            )
            self._run_physics_loop()
        except KeyboardInterrupt:
            logger.info("RoboTwin bridge interrupted")
        except Exception:
            logger.exception("RoboTwin bridge failed")
            raise
        finally:
            self.stop_server()
            self._restore_shutdown_signal_handlers(previous_signal_handlers)

    def stop_server(self) -> None:
        """Stop physics first, then close SAPIEN and all bridge-owned SHMs."""
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        trace_path = self._trace.path
        try:
            self._stop_camera_publisher()
            if self._eef_resolver is not None:
                self._eef_resolver.close()
            self._runtime.close()
        finally:
            for camera in self._camera_devices.values():
                camera._cleanup_shared_memory()
            self._camera_devices.clear()
            self._cleanup_shared_memory()
            self._cleanup_control_shared_memory()
            if self._sim_clock_shm is not None:
                try:
                    self._sim_clock_shm.close()
                    self._sim_clock_shm.unlink()
                except FileNotFoundError:
                    pass
                finally:
                    self._sim_clock_shm = None
            self._trace.logger.log_meta(
                "sim_bridge_stopped",
                sim_step=self._sim_step,
                sim_time_ns=self._sim_time_ns(),
                realtime_factor=self._last_realtime_factor,
                realtime_factor_warn_threshold=self._realtime_factor_warn_threshold,
                **self._control_metrics_snapshot(),
            )
            self._trace.close()
        self._maybe_generate_trace_visualizer(trace_path)

    def _run_physics_loop(self) -> None:
        next_tick = time.monotonic()
        catch_up_steps = 0
        while self.running:
            timing = _LoopTiming(loop_started_ns=time.perf_counter_ns())
            timing.catch_up_steps = catch_up_steps

            # Pull the newest high-level action written by ActionExecutor.
            section_started_ns = time.perf_counter_ns()
            self._process_source_selection_transition()
            self._accept_latest_control_from_shm()
            timing.accept_control_ns = time.perf_counter_ns() - section_started_ns

            # Convert any completed async EEF resolver result into a joint target.
            section_started_ns = time.perf_counter_ns()
            self._poll_eef_resolution()
            timing.poll_eef_ns = time.perf_counter_ns() - section_started_ns

            # Apply this physics step's interpolated drive target to SAPIEN.
            section_started_ns = time.perf_counter_ns()
            self._apply_interpolated_target()
            timing.apply_target_ns = time.perf_counter_ns() - section_started_ns

            section_started_ns = time.perf_counter_ns()
            lock_started_ns = section_started_ns
            with self._runtime_lock:
                timing.runtime_lock_wait_ns += time.perf_counter_ns() - lock_started_ns
                self._runtime.step()
            timing.scene_step_ns = time.perf_counter_ns() - section_started_ns
            self._sim_step += 1
            self._write_sim_clock()

            section_started_ns = time.perf_counter_ns()
            lock_started_ns = section_started_ns
            with self._runtime_lock:
                timing.runtime_lock_wait_ns += time.perf_counter_ns() - lock_started_ns
                timing.gui_rendered = self._runtime.render_gui_if_due(self._sim_step)
            timing.gui_render_ns = time.perf_counter_ns() - section_started_ns

            section_started_ns = time.perf_counter_ns()
            lock_started_ns = section_started_ns
            with self._runtime_lock:
                timing.runtime_lock_wait_ns += time.perf_counter_ns() - lock_started_ns
                self._measured_joint14, self._measured_qvel14 = (
                    self._runtime.read_joint14_state()
                )
            timing.read_state_ns = time.perf_counter_ns() - section_started_ns

            section_started_ns = time.perf_counter_ns()
            timing.state_published = self._publish_state_snapshot(force=False)
            timing.publish_state_ns = time.perf_counter_ns() - section_started_ns

            next_tick += self._physics_interval_s
            next_tick, catch_up_steps = self._wait_until_next_physics_tick(
                next_tick, timing, catch_up_steps
            )
            timing.loop_total_ns = time.perf_counter_ns() - timing.loop_started_ns
            self._maybe_trace_loop_timing(timing)

    def _wait_until_next_physics_tick(
        self, next_tick: float, timing: _LoopTiming, catch_up_steps: int
    ) -> tuple[float, int]:
        remaining = next_tick - time.monotonic()
        timing.sleep_requested_ns = int(round(max(0.0, remaining) * 1_000_000_000))
        if remaining <= 0:
            timing.overrun_ns = int(round(abs(remaining) * 1_000_000_000))
            catch_up_steps += 1
            if self._max_catch_up_steps and catch_up_steps >= self._max_catch_up_steps:
                # Bounded catch-up prevents a long render/driver stall from causing
                # an unbounded burst of physics steps.
                next_tick = time.monotonic()
                timing.catch_up_reset = True
                catch_up_steps = 0
            return next_tick, catch_up_steps

        spin_margin_s = min(
            self._physics_sleep_spin_margin_s,
            max(0.0, 0.5 * self._physics_interval_s),
        )
        sleep_for_s = max(0.0, remaining - spin_margin_s)
        if sleep_for_s > 0:
            timing.sleep_call_requested_ns = int(round(sleep_for_s * 1_000_000_000))
            sleep_started_ns = time.perf_counter_ns()
            time.sleep(sleep_for_s)
            timing.sleep_ns = time.perf_counter_ns() - sleep_started_ns

        spin_started_ns = time.perf_counter_ns()
        while True:
            remaining = next_tick - time.monotonic()
            if remaining <= 0:
                break
            if remaining > 0.001:
                time.sleep(0)
        timing.spin_wait_ns = time.perf_counter_ns() - spin_started_ns
        timing.sleep_overshoot_ns = int(
            round(max(0.0, time.monotonic() - next_tick) * 1_000_000_000)
        )
        return next_tick, 0

    def _accept_latest_control_from_shm(self) -> None:
        source_name = (
            self._selected_source_name(self._selector_value_for_control_tick())
            or self.default_control_shm_name
        )
        timestamp_ns, action = self.read_latest_control_command(
            source_name,
            self._minimum_control_timestamp_ns(),
        )
        if timestamp_ns is None or action is None:
            return
        self.accept_control_frame(action, control_write_timestamp_ns=int(timestamp_ns))

    def _initialize_control_buffer_header(self) -> None:
        device_header = pack_device_header(
            f"{self.device_name}_control",
            self.device_class_id,
            self.control_fps,
            self.control_dtype,
            self.control_shape,
            self.control_frame_size,
        )
        buffer_header = pack_buffer_header(self.control_buffer_size, 0, 0)
        control_buffer = self.control_shared_memory[self.default_control_shm_name].buf
        control_buffer[:DEVICE_HEADER_SIZE] = device_header
        control_buffer[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

    def _create_sim_clock_shared_memory(self) -> None:
        self._sim_clock_shm = create_shared_memory(self._sim_clock_shm_name, 8)
        self._write_sim_clock()

    def _write_sim_clock(self) -> None:
        if self._sim_clock_shm is None:
            return
        self._sim_clock_shm.buf[:8] = int(self._sim_time_ns()).to_bytes(
            8, byteorder="little", signed=False
        )

    def _poll_eef_resolution(self) -> None:
        if self._eef_resolver is None:
            return
        try:
            resolved = self._eef_resolver.poll()
        except Exception as exc:
            self._trace.logger.log_meta(
                "sim_eef_resolve_failed",
                error=str(exc),
                sim_step=self._sim_step,
                sim_time_ns=self._sim_time_ns(),
            )
            return
        if resolved is None:
            return
        sequence, joint_target = resolved
        setpoint = self._pending_eef_setpoints.pop(sequence, None)
        if setpoint is None:
            return
        self._queue_joint_target(setpoint, joint_target)

    def _queue_joint_target(self, setpoint: Any, target_joint14: Any) -> None:
        self._finish_active_segment_if_due_next_step()
        if self._active_segment is not None:
            self._trace.logger.log_meta(
                "sim_control_superseded",
                control_write_timestamp_ns=self._active_segment.setpoint.control_write_timestamp_ns,
                superseded_by_control_write_timestamp_ns=setpoint.control_write_timestamp_ns,
                sim_step=self._sim_step,
                sim_time_ns=self._sim_time_ns(),
            )
        self._target_joint14 = validate_action(target_joint14, "joint14")
        self._active_segment = _JointTargetSegment(
            setpoint=setpoint,
            start_joint14=self._drive_joint14.copy(),
            target_joint14=self._target_joint14.copy(),
            velocity_joint14=(self._target_joint14 - self._drive_joint14)
            / self._control_interval_s,
            start_sim_time_s=self._sim_time_s_float(),
            duration_s=self._control_interval_s,
        )

    def _finish_active_segment_if_due_next_step(self) -> None:
        segment = self._active_segment
        if segment is None:
            return
        next_sim_time_s = self._sim_time_s_float() + self._physics_interval_s
        completion_slack_s = min(
            2.0 * self._physics_interval_s,
            0.25 * self._control_interval_s,
        )
        if next_sim_time_s - segment.start_sim_time_s + completion_slack_s < segment.duration_s:
            return
        self._drive_joint14 = segment.target_joint14.copy()
        with self._runtime_lock:
            self._runtime.apply_joint14_target(
                self._drive_joint14, segment.velocity_joint14
            )
        if not segment.trace_logged:
            segment.trace_logged = True
            applied_at_ns = time.time_ns()
            self._last_control_write_timestamp_ns = segment.setpoint.control_write_timestamp_ns
            self._last_sim_apply_timestamp_ns = applied_at_ns
            self._trace.action_applied(
                segment.setpoint,
                target_joint14=segment.target_joint14,
                timestamp_ns=applied_at_ns,
                sim_step=self._sim_step,
                sim_time_ns=self._sim_time_ns(),
            )
        self._active_segment = None

    def _apply_interpolated_target(self) -> None:
        segment = self._active_segment
        if segment is None:
            return

        next_sim_time_s = self._sim_time_s_float() + self._physics_interval_s
        elapsed_s = max(0.0, next_sim_time_s - segment.start_sim_time_s)
        alpha = min(1.0, elapsed_s / segment.duration_s)
        self._drive_joint14 = segment.start_joint14 + alpha * (
            segment.target_joint14 - segment.start_joint14
        )
        with self._runtime_lock:
            self._runtime.apply_joint14_target(
                self._drive_joint14, segment.velocity_joint14
            )

        if not segment.trace_logged:
            segment.trace_logged = True
            applied_at_ns = time.time_ns()
            self._last_control_write_timestamp_ns = segment.setpoint.control_write_timestamp_ns
            self._last_sim_apply_timestamp_ns = applied_at_ns
            self._trace.action_applied(
                segment.setpoint,
                target_joint14=segment.target_joint14,
                timestamp_ns=applied_at_ns,
                sim_step=self._sim_step,
                sim_time_ns=self._sim_time_ns(),
            )

        if alpha >= 1.0:
            self._active_segment = None

    def _publish_state_snapshot(self, *, force: bool) -> bool:
        now_monotonic = time.monotonic()
        if (
            not force
            and now_monotonic - self._last_state_publish_monotonic < self.update_interval
        ):
            return False
        self._last_state_publish_monotonic = now_monotonic
        timestamp_ns = time.time_ns()
        self._write_array_to_shm_with_timestamp(self._measured_joint14, timestamp_ns)
        self._trace.measured_state(
            measured_qpos=self._measured_joint14,
            measured_qvel=self._measured_qvel14,
            target_qpos=self._drive_joint14,
            last_control_write_timestamp_ns=self._last_control_write_timestamp_ns,
            last_sim_apply_timestamp_ns=self._last_sim_apply_timestamp_ns,
            timestamp_ns=timestamp_ns,
            sim_step=self._sim_step,
            sim_time_ns=self._sim_time_ns(),
        )
        return True

    def _start_camera_publisher(self) -> None:
        if not self._camera_specs or self._camera_thread is not None:
            return
        self._camera_stop_event.clear()
        self._camera_thread = threading.Thread(
            target=self._run_camera_publisher,
            name=f"{self.device_name}_{self.device_class_id}_camera_publisher",
            daemon=True,
        )
        self._camera_thread.start()

    def _stop_camera_publisher(self) -> None:
        self._camera_stop_event.set()
        thread = self._camera_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("RoboTwin camera publisher did not stop within 5s")
        self._camera_thread = None

    def _run_camera_publisher(self) -> None:
        while self.running and not self._camera_stop_event.is_set():
            loop_started_ns = time.perf_counter_ns()
            try:
                published, capture_ns, write_ns, wait_after_capture_s = (
                    self._publish_camera_snapshot_sync(force=False)
                )
            except Exception:
                logger.exception("RoboTwin camera publisher failed")
                self.running = False
                return

            loop_total_ns = time.perf_counter_ns() - loop_started_ns
            if published and self._trace_timing_interval_s > 0:
                self._trace.logger.log_meta(
                    "sim_camera_timing",
                    publish_camera_ns=loop_total_ns,
                    capture_camera_ns=capture_ns,
                    write_camera_shm_ns=write_ns,
                    wait_after_capture_ns=int(
                        round(wait_after_capture_s * 1_000_000_000)
                    ),
                    sim_step=self._sim_step,
                    sim_time_ns=self._sim_time_ns(),
                )

            wait_s = self._next_camera_wait_s()
            self._camera_stop_event.wait(wait_s)

    def _next_camera_wait_s(self) -> float:
        if not self._camera_specs:
            return 0.1
        now_monotonic = time.monotonic()
        waits = []
        for spec in self._camera_specs:
            last_publish = self._last_camera_publish_monotonic_by_name.get(
                spec.camera_name, 0.0
            )
            waits.append(max(0.0, last_publish + 1.0 / spec.fps - now_monotonic))
        return min(waits) if waits else 0.1

    def _publish_camera_snapshot_sync(self, *, force: bool) -> tuple[bool, int, int, float]:
        if not self._camera_specs:
            return False, 0, 0, 0.1
        now_monotonic = time.monotonic()
        due_specs = [
            spec
            for spec in self._camera_specs
            if force
            or now_monotonic
            - self._last_camera_publish_monotonic_by_name.get(spec.camera_name, 0.0)
            >= 1.0 / spec.fps
        ]
        if not due_specs:
            return False, 0, 0, self._next_camera_wait_s()

        capture_monotonic = now_monotonic
        capture_started_ns = time.perf_counter_ns()
        frames = self._runtime.capture_rgb(render_gui=False)
        capture_ns = time.perf_counter_ns() - capture_started_ns
        timestamp_ns = time.time_ns()
        write_started_ns = time.perf_counter_ns()
        # RoboTwin renders all three ACT cameras for one scene snapshot. Publish
        # the full synchronized camera set once per camera tick instead of
        # letting tiny per-stream due-time skew trigger repeated full renders.
        for spec in self._camera_specs:
            frame = np.asarray(frames[spec.camera_name], dtype=np.uint8)
            if frame.shape != spec.data_shape:
                raise ValueError(
                    f"RoboTwin camera {spec.camera_name!r} returned {frame.shape}; "
                    f"configured SHM shape is {spec.data_shape}"
                )
            self._camera_devices[spec.camera_name]._write_array_to_shm_with_timestamp(
                frame, timestamp_ns
            )
            self._last_camera_publish_monotonic_by_name[spec.camera_name] = capture_monotonic
        write_ns = time.perf_counter_ns() - write_started_ns
        return True, capture_ns, write_ns, self._next_camera_wait_s()

    def _maybe_trace_loop_timing(self, timing: _LoopTiming) -> None:
        if self._trace_timing_interval_s <= 0:
            return
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_timing_trace_monotonic < self._trace_timing_interval_s:
            return
        self._last_timing_trace_monotonic = now_monotonic
        sim_time_ns = self._sim_time_ns()
        realtime_factor = self._update_realtime_factor(now_monotonic, sim_time_ns)
        self._trace.logger.log_meta(
            "sim_loop_timing",
            accept_control_ns=timing.accept_control_ns,
            poll_eef_ns=timing.poll_eef_ns,
            apply_target_ns=timing.apply_target_ns,
            scene_step_ns=timing.scene_step_ns,
            gui_render_ns=timing.gui_render_ns,
            read_state_ns=timing.read_state_ns,
            publish_state_ns=timing.publish_state_ns,
            publish_camera_ns=timing.publish_camera_ns,
            runtime_lock_wait_ns=timing.runtime_lock_wait_ns,
            sleep_requested_ns=timing.sleep_requested_ns,
            sleep_call_requested_ns=timing.sleep_call_requested_ns,
            sleep_ns=timing.sleep_ns,
            spin_wait_ns=timing.spin_wait_ns,
            sleep_overshoot_ns=timing.sleep_overshoot_ns,
            loop_total_ns=timing.loop_total_ns,
            state_published=timing.state_published,
            camera_published=timing.camera_published,
            gui_rendered=timing.gui_rendered,
            overrun_ns=timing.overrun_ns,
            catch_up_steps=timing.catch_up_steps,
            catch_up_reset=timing.catch_up_reset,
            sim_step=self._sim_step,
            sim_time_ns=sim_time_ns,
            realtime_factor=realtime_factor,
            realtime_factor_warn_threshold=float(
                getattr(self, "_realtime_factor_warn_threshold", 0.95)
            ),
            **self._control_metrics_snapshot(),
        )

    def _create_camera_streams(self) -> None:
        for spec in self._camera_specs:
            # Camera streams use the image-format contract (``type: rgb``),
            # so construct the camera descriptor with the CAMERA device kind.
            # A plain BaseDevice would parse the same mapping as a low-dimensional
            # stream and reject the RGB format.
            camera = RoboTwinCamera(
                device_class_id=spec.device_class_id,
                state_feedback={
                    "fps": spec.fps,
                    "buffer_size": spec.buffer_size,
                    "hardware_latency_ms": 0.0,
                    "format": {
                        "shape": list(spec.data_shape),
                        "dtype": "uint8",
                        "type": "rgb",
                    },
                },
            )
            camera._create_shared_memory()
            self._camera_devices[spec.camera_name] = camera

    def _sim_time_ns(self) -> int:
        return int(round(self._sim_step * self._physics_interval_s * 1_000_000_000))

    def _sim_time_s_float(self) -> float:
        return self._sim_step * self._physics_interval_s

    def _maybe_generate_trace_visualizer(self, trace_path: Optional[str]) -> None:
        if not self._trace_visualizer.get("enabled", False) or not trace_path:
            return
        script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "visualization"
            / "robotwin_trace_visualizer.py"
        )
        if not script.is_file():
            logger.warning(
                f"RoboTwin trace visualizer is enabled but script is missing: {script}"
            )
            return
        command = [sys.executable, str(script), "--trace", trace_path]
        output_path = self._trace_visualizer.get("output")
        if output_path:
            command.extend(("--output", str(output_path)))
        if self._trace_visualizer.get("open_browser", False):
            command.append("--open")
        try:
            subprocess.run(command, check=False, timeout=30)
        except Exception as exc:
            logger.warning(f"Failed to generate RoboTwin trace visualizer: {exc}")
