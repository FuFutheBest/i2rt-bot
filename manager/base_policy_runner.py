#!/usr/bin/env python3
"""
Policy Runner - reads action chunking from policy and dispatches to robot control SHMs.

Author: Jun Lv, Zheng Wang
"""

import time
import struct
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import signal
from collections import deque
from pathlib import Path

from utils.shm_utils import (
    pack_device_header, pack_buffer_header, pack_frame_header,
    unpack_device_header, unpack_buffer_header, unpack_frame_header,
    DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    pack_policy_header, POLICY_HEADER_SIZE, unpack_policy_header,
    get_device_info_offset, get_policy_data_offset,
    get_policy_action_index_offset, create_policy_shm, connect_to_shared_memory,
    ACTION_INDEX_SIZE, ACTION_INDEX_FORMAT
)
from utils.logger_config import logger
from utils.policy_control_config import parse_control_targets
from utils.time_control import precise_loop_timing
from utils.config_parser import ensure_config_dict, load_config
from utils.action_state_xyz_logger import ActionStateXYZLogger
from utils.timestamp_event_logger import TimestampEventLogger
from manager.utils import load_chunk_manager


class _DisabledActionVisualizer:
    """No-op visualizer used until an operator explicitly enables plotting."""

    enable = False

    def add_action(self, action_dict: Dict[str, np.ndarray]) -> None:
        del action_dict

    def update(self) -> None:
        """Preserve the visualizer's deprecated no-op compatibility method."""
        return None

    def close(self) -> None:
        return None


class BasePolicyRunner:
    """
    Policy Runner for reading action chunking and dispatching to robot control SHMs.

    This class supports device-specific action delays to synchronize execution across
    devices with different latencies. Actions are queued per device, and devices with
    larger delays have their actions held longer before dispatch.

    Args:
        config: Configuration dictionary (alternative to config_path)
        config_path: Path to configuration YAML file
        robot_control_shms: List of robot control SHM configurations
        execution_fps: Execution frequency in Hz
        chunk_length: Length of action chunks
        chunk_manager: Chunk manager configuration
        enable_action_visualizer: Enable action visualization
        action_visualizer_config: Action visualizer configuration
        device_delays: Dict mapping device names to delay frames. Devices with higher
                      latency should have smaller delays, while devices with lower
                      latency should have larger delays to ensure synchronized execution.
                      Example: {'RizonRobot_1': 0, 'RizonRobot_2': 10}
                      This means RizonRobot_2 actions will be delayed by 10 frames.
    """

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            *,
            config_path: str = "config.yaml",
            robot_control_shms: Optional[List[Dict[str, Any]]] = None,
            execution_fps: float = 100.0,
            chunk_length: int = 10,
            signal_files: Optional[Dict[str, str]] = None,
            chunk_manager: Optional[Dict[str, Any]] = None,
            execution_clock: str = "wall_time",
            sim_clock_poll_interval_s: float = 0.001,
            enable_action_visualizer: bool = False,
            action_visualizer_config: Optional[Dict[str, Any]] = None,
            device_delays: Optional[Dict[str, int]] = None,
            device_dimension_delays: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        self.config_path = config_path
        self.robot_control_shms = [
            target.runtime_mapping()
            for target in parse_control_targets(
                robot_control_shms,
                field="action_executor.robot_control_shms",
            )
        ]
        self.execution_fps = execution_fps
        self.chunk_length = chunk_length

        self.running = False
        self.policy_shm = None
        self.policy_device_count: int = 0
        self.policy_frame_sizes: List[int] = []
        self.robot_states = []

        # Initialize execution state
        self.current_step = 0
        self.current_action_chunk_dict = None  # Cache current chunk dict
        self.last_read_timestamp = 0  # Track last read timestamp
        self.last_reset_timestamp = 0  # Track when last reset occurred (nanoseconds)
        self.chunk_manager = load_chunk_manager(chunk_manager or {}, execution_fps=self.execution_fps)
        self.action_step_interval_ns = int(round(1e9 / self.execution_fps))
        self.execution_clock = self._validate_execution_clock(execution_clock)
        self.sim_clock_shm_name = (
            self._derive_sim_clock_shm_name(self.robot_control_shms)
            if self.execution_clock == "sim_time"
            else None
        )
        self.sim_clock_poll_interval_s = max(0.0001, float(sim_clock_poll_interval_s))
        self.sim_clock_shm = None
        self._next_sim_dispatch_time_ns: Optional[int] = None
        # Sim-time scheduling diagnostics.  ``_sim_dispatch_due`` keeps its
        # boolean API for callers/tests, while these fields preserve how many
        # action deadlines were crossed by one poll.
        self._sim_dispatch_steps_due = 0
        self._sim_dispatch_skipped_deadlines = 0
        self._sim_dispatch_deadline_ns: Optional[int] = None
        self._sim_dispatch_lateness_ns = 0
        self._sim_dispatch_sim_time_ns: Optional[int] = None
        self._deadline_due_count = 0
        self._deadline_miss_count = 0
        self._action_dispatch_count = 0
        self._first_action_dispatch_wall_timestamp_ns: Optional[int] = None
        self._first_action_dispatch_sim_timestamp_ns: Optional[int] = None
        self._last_action_dispatch_sim_timestamp_ns: Optional[int] = None
        self._last_action_dispatch_wall_timestamp_ns: Optional[int] = None
        self._estimated_wall_action_step_interval_ns = self.action_step_interval_ns
        self._action_chunk_log_index = 0
        self._pending_action_chunk_log: Optional[Dict[str, Any]] = None
        logger.info(
            f"ActionExecutor chunk_manager: type={type(self.chunk_manager).__name__}, "
            f"config={chunk_manager or {}}, execution_clock={self.execution_clock}"
        )
        self.timestamp_logger = TimestampEventLogger("action_executor")
        self.action_state_xyz_logger = ActionStateXYZLogger("action_executor")
        # Retry settings for connecting to robot control SHMs (to avoid race with robot startup)
        self.retry_connect_secs: float = 0.5
        self.connect_timeout_secs: float = 60.0

        # Device delay management: device_name -> delay_frames (for backward compatibility)
        self.device_delays = device_delays or {}
        # Action queues for each device: device_name -> deque of actions
        self.device_action_queues: Dict[str, deque] = {}
        # Fine-grained delay config: device_name -> list of (indices, delay_frames)
        # Example: {'RizonRobot_1': [([7, 8, 9], 10), ([10, 11], 5)]}
        self.device_dimension_delays: Dict[str, List[Tuple[List[int], int]]] = {}
        # Store first action for each device to repeat during delay period
        self.device_first_actions: Dict[str, Optional[np.ndarray]] = {}

        # Signal files config
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._policy_runner_reset_command_path = signal_files.get('policy_runner_reset_command')
        if self._policy_runner_reset_command_path is None:
            raise ValueError("policy_runner_reset_command_path is required in signal_files")

        # Parse dimension delays from config format
        if device_dimension_delays:
            for device_name, delay_configs in device_dimension_delays.items():
                parsed_delays = []
                for delay_config in delay_configs:
                    indices = delay_config.get('indices', [])
                    delay = delay_config.get('delay', 0)
                    parsed_delays.append((indices, delay))
                self.device_dimension_delays[device_name] = parsed_delays

        # Keep plotting dependencies out of the core manager import path.  Most
        # runners do not enable visualization, and the actual visualizer pulls
        # in optional matplotlib/OpenCV/Plotly dependencies.
        self.enable_action_visualizer = enable_action_visualizer
        action_viz_cfg = action_visualizer_config or {}
        # Parse time_range if provided (can be None, or tuple/list [start, end])
        time_range = action_viz_cfg.get('time_range', None)
        if time_range is not None:
            if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                time_range = tuple(time_range)
            else:
                logger.warning(f"Invalid time_range format: {time_range}, expected [start, end] or None. Ignoring.")
                time_range = None

        if enable_action_visualizer:
            # Import only for the explicitly requested operator-facing feature.
            from visualizers.action_visualizer import ActionVisualizer

            self.action_visualizer = ActionVisualizer(
                max_history=action_viz_cfg.get('max_history', 500),
                update_interval_ms=action_viz_cfg.get('update_interval_ms', 50),
                figsize=tuple(action_viz_cfg.get('figsize', [12, 8])),
                window_size=action_viz_cfg.get('window_size', 100),
                enable=True,
                output_dir=action_viz_cfg.get('output_dir', './action_visualizations'),
                html_filename=action_viz_cfg.get('html_filename', 'action_visualization.html'),
                time_range=time_range,
            )
        else:
            self.action_visualizer = _DisabledActionVisualizer()
        # Visualization updates are now handled automatically in background thread
        # No need for manual update counter

        if config is None:
            self.config = load_config(self.config_path)
        else:
            self.config = ensure_config_dict(config)
        self.enable_rtc: bool = self._get_enable_rtc_from_config(self.config)

        # Initialize device action queues based on device delays
        self._initialize_device_queues()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if self.device_delays:
            logger.info(f"Device delays configured: {self.device_delays}")
        logger.info(f"RTC enabled: {self.enable_rtc}")

    @staticmethod
    def _validate_execution_clock(execution_clock: str) -> str:
        normalized = str(execution_clock).strip().lower()
        if normalized not in {"wall_time", "sim_time"}:
            raise ValueError("execution_clock must be 'wall_time' or 'sim_time'")
        return normalized

    @staticmethod
    def _derive_sim_clock_shm_name(
            robot_control_shms: List[Dict[str, Any]],
    ) -> str:
        """Derive the bridge-owned sim-time channel from the first command target."""
        if not robot_control_shms:
            raise ValueError(
                "execution_clock='sim_time' requires at least one structured "
                "action_executor.robot_control_shms command endpoint"
            )
        endpoint = robot_control_shms[0]["endpoint"]
        return f"{endpoint.device}_{endpoint.device_class_id}_sim_clock"

    def _get_enable_rtc_from_config(self, config: Dict[str, Any]) -> bool:
        """Extract enable_rtc from policy params."""
        policy_config = config.get("policy", {})
        params_config = policy_config.get("params", {})
        return bool(params_config.get("enable_rtc", False))

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        self.stop()

    def _initialize_device_queues(self) -> None:
        """
        Initialize action queues for each device based on configured delays.

        For devices with dimension-level delays, we create separate queues for each
        dimension group. For devices with only whole-action delays, we create a single queue.
        """
        # Get canonical device keys from structured control targets.
        for robot_cfg in self.robot_control_shms:
            device_name = robot_cfg["device_key"]

            # Initialize first action storage
            self.device_first_actions[device_name] = None

            # Check if this device has dimension-level delays
            if device_name in self.device_dimension_delays:
                # Create a dict of queues: dimension_group_id -> deque
                # We'll use a special structure: dict with 'dimension_queues' key
                dimension_queues = {}
                for idx, (indices, delay) in enumerate(self.device_dimension_delays[device_name]):
                    dimension_queues[idx] = deque()
                # Also create a queue for dimensions not specified (default delay)
                dimension_queues['default'] = deque()
                self.device_action_queues[device_name] = dimension_queues
                logger.info(
                    f"Initialized dimension-level queues for {device_name}: {len(dimension_queues) - 1} groups + default")
            else:
                # Initialize single queue for whole action
                self.device_action_queues[device_name] = deque()

        logger.info(f"Initialized action queues for devices: {list(self.device_action_queues.keys())}")

    def start(self) -> bool:
        """Start the action executor."""
        if not self.connect():
            logger.error("Failed to connect, cannot start executor")
            return False

        try:
            return self.run_connected()
        finally:
            self.stop()

    def run_connected(self) -> bool:
        """Run the paced execution loop after control SHMs are connected."""
        logger.info("Starting ActionExecutor")
        self.running = True
        wait_for_next_execution = precise_loop_timing(1.0 / float(self.execution_fps))

        try:
            while self.running:
                if self.execution_clock == "sim_time":
                    dispatch_ready = self._sim_dispatch_due()
                    self._update_execution(
                        dispatch_ready=dispatch_ready,
                        dispatch_steps=getattr(self, "_sim_dispatch_steps_due", 0),
                    )
                    time.sleep(self.sim_clock_poll_interval_s)
                else:
                    self._update_execution(dispatch_ready=True)
                    wait_for_next_execution()
        except KeyboardInterrupt:
            pass

        return True

    def stop(self) -> None:
        """Stop the action executor."""
        self.running = False

        # Close action visualizer
        if self.enable_action_visualizer:
            try:
                self.action_visualizer.close()
            except Exception as e:
                logger.warning(f"Error closing action visualizer: {e}")

        try:
            self.action_state_xyz_logger.close()
        except Exception as e:
            logger.warning(f"Error closing action/state xyz logger: {e}")

        # Some optional chunk managers keep a state SHM or other external
        # resources open while stitching successive predictions.
        if hasattr(self.chunk_manager, "close"):
            try:
                self.chunk_manager.close()
            except Exception as e:
                logger.warning(f"Error closing chunk manager resources: {e}")

        # Close robot control SHMs
        for robot_state in self.robot_states:
            try:
                shm = robot_state.get('shm')
                if shm is not None:
                    shm.close()
            except Exception:
                pass
        if self.sim_clock_shm is not None:
            try:
                self.sim_clock_shm.close()
            except Exception:
                pass
            self.sim_clock_shm = None

        # Close policy SHM
        if self.policy_shm is not None:
            try:
                self.policy_shm.close()
            except Exception:
                pass

        logger.info("ActionExecutor stopped")

    def connect(self) -> bool:
        """Connect to robot control SHMs and create policy SHM."""
        try:
            # Connect to robot control SHMs
            for robot_cfg in self.robot_control_shms:
                robot_state = self._connect_robot_control(robot_cfg)
                if robot_state is None:
                    logger.error(
                        "Failed to connect to robot control SHM: %s",
                        robot_cfg["shm_name"],
                    )
                    return False
                self.robot_states.append(robot_state)

            # Create policy SHM
            self._create_policy_shm()
            self._connect_execution_clock()

            # Reset chunk manager state to ensure new chunks are recognized after restart
            # This is important when PolicyRunner restarts while PolicyConnector is still running
            if hasattr(self.chunk_manager, 'last_read_time'):
                self.chunk_manager.last_read_time = -1  # Reset to -1 to ensure first read is always considered new
                logger.info("ActionExecutor: Reset chunk manager state after restart")

            logger.info("ActionExecutor connected successfully")
            return True

        except Exception as e:
            logger.error(f"ActionExecutor failed to connect: {e}")
            return False

    def _connect_execution_clock(self) -> None:
        if self.execution_clock != "sim_time":
            return
        if not self.sim_clock_shm_name:
            raise RuntimeError(
                "sim-time clock was not derived from "
                "action_executor.robot_control_shms"
            )

        start_time = time.time()
        while True:
            try:
                self.sim_clock_shm = connect_to_shared_memory(
                    self.sim_clock_shm_name, read_only=True, log_missing=False
                )
                logger.info(
                    f"ActionExecutor connected to sim-time clock SHM: {self.sim_clock_shm_name}"
                )
                return
            except Exception as exc:
                if time.time() - start_time < self.connect_timeout_secs:
                    logger.info(
                        f"ActionExecutor: Waiting for sim clock SHM '{self.sim_clock_shm_name}'... "
                        f"retry in {self.retry_connect_secs}s"
                    )
                    time.sleep(self.retry_connect_secs)
                    continue
                raise RuntimeError(
                    f"Failed to connect to sim clock SHM '{self.sim_clock_shm_name}'"
                ) from exc

    def _read_sim_clock_ns(self) -> Optional[int]:
        if self.sim_clock_shm is None:
            return None
        try:
            return int.from_bytes(
                self.sim_clock_shm.buf[:8], byteorder="little", signed=False
            )
        except Exception as exc:
            logger.error(f"Failed to read sim clock SHM: {exc}")
            return None

    def _sim_dispatch_due(self) -> bool:
        """Advance sim-time deadlines and report whether one is ready.

        A single polling wake-up may cross several action periods.  The
        boolean return remains compatible with the old executor, while the
        diagnostic fields tell ``_update_execution`` how far the chunk manager
        must advance to stay on the current simulation-time index.
        """
        self._sim_dispatch_steps_due = 0
        self._sim_dispatch_skipped_deadlines = 0
        self._sim_dispatch_deadline_ns = None
        self._sim_dispatch_lateness_ns = 0
        self._sim_dispatch_sim_time_ns = None
        sim_time_ns = self._read_sim_clock_ns()
        if sim_time_ns is None:
            return False
        sim_time_ns = int(sim_time_ns)
        self._sim_dispatch_sim_time_ns = sim_time_ns
        if self._next_sim_dispatch_time_ns is None:
            # Preserve the historical behavior: the first available chunk is
            # dispatched immediately, then regular sim-time deadlines begin.
            self._next_sim_dispatch_time_ns = sim_time_ns + self.action_step_interval_ns
            self._sim_dispatch_steps_due = 1
            self._sim_dispatch_deadline_ns = sim_time_ns
            self._deadline_due_count = getattr(self, "_deadline_due_count", 0) + 1
            return True

        next_deadline_ns = int(self._next_sim_dispatch_time_ns)
        if sim_time_ns < next_deadline_ns:
            return False

        # Compute the number of crossed deadlines arithmetically instead of
        # looping once per deadline when a simulator stalls for a long time.
        due_steps = ((sim_time_ns - next_deadline_ns) // self.action_step_interval_ns) + 1
        latest_deadline_ns = next_deadline_ns + (due_steps - 1) * self.action_step_interval_ns
        self._next_sim_dispatch_time_ns = latest_deadline_ns + self.action_step_interval_ns
        self._sim_dispatch_steps_due = int(due_steps)
        self._sim_dispatch_skipped_deadlines = max(0, int(due_steps) - 1)
        self._sim_dispatch_deadline_ns = int(latest_deadline_ns)
        self._sim_dispatch_lateness_ns = max(0, sim_time_ns - latest_deadline_ns)
        self._deadline_due_count = getattr(self, "_deadline_due_count", 0) + int(due_steps)
        self._deadline_miss_count = getattr(self, "_deadline_miss_count", 0) + self._sim_dispatch_skipped_deadlines
        return True

    def _connect_robot_control(self, robot_cfg: Dict) -> Optional[Dict]:
        """Connect to a single robot control SHM using unified interface (read-write for policy runner)."""
        control_shm_name = robot_cfg["shm_name"]

        start_time = time.time()
        while True:
            robot_shm = None
            try:
                # Use unified SHM interface for read-write connection (policy runner needs to write)
                robot_shm = connect_to_shared_memory(
                    control_shm_name, read_only=False, log_missing=False
                )
                buf = robot_shm.buf
                device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
                buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
                payload_contract = self._validate_control_shm_contract(
                    robot_cfg,
                    device_header,
                    buffer_header,
                    buffer_length=len(buf),
                )

                robot_state = {
                    'device_info': robot_cfg,
                    'shm': robot_shm,
                    'shm_name': control_shm_name,
                    **payload_contract,
                    'buffer_size': buffer_header['buffer_size'],
                }

                logger.info(f"Connected to robot control SHM: {control_shm_name} (READ-WRITE)")
                return robot_state
            except ValueError as exc:
                if robot_shm is not None:
                    robot_shm.close()
                logger.error(
                    "ActionExecutor: Control SHM '%s' violates its configured payload contract: %s",
                    control_shm_name,
                    exc,
                )
                return None
            except Exception as e:
                if robot_shm is not None:
                    robot_shm.close()
                if time.time() - start_time < self.connect_timeout_secs:
                    logger.info(
                        f"ActionExecutor: Waiting for control SHM '{control_shm_name}'... retry in {self.retry_connect_secs}s"
                    )
                    time.sleep(self.retry_connect_secs)
                    continue
                logger.error(f"Failed to connect to robot control SHM after retries: {e}")
                return None

    @staticmethod
    def _validate_control_shm_contract(
        robot_cfg: Dict[str, Any],
        device_header: Dict[str, Any],
        buffer_header: Dict[str, Any],
        *,
        buffer_length: int,
    ) -> Dict[str, Any]:
        """Validate one configured command target against its owned SHM header.

        ``policy_data`` is deliberately a float64 transport channel. This
        boundary is where a policy row is converted into the dtype and shape
        that the command SHM owner declared, so the destination header must
        agree exactly with the structured target configuration.
        """
        target_shape = tuple(robot_cfg["shape"])
        target_dtype_name = str(robot_cfg["dtype"])
        target_dtype = np.dtype(target_dtype_name)
        header_shape = tuple(device_header["shape"])
        header_dtype_name = str(device_header["data_dtype"])

        if header_shape != target_shape:
            raise ValueError(
                "control SHM shape does not match configured target: "
                f"header={header_shape}, config={target_shape}"
            )
        if header_dtype_name != target_dtype_name:
            raise ValueError(
                "control SHM dtype does not match configured target: "
                f"header={header_dtype_name!r}, config={target_dtype_name!r}"
            )

        payload_size = int(np.prod(target_shape)) * target_dtype.itemsize
        frame_size = int(device_header["frame_size"])
        expected_frame_size = FRAME_HEADER_SIZE + payload_size
        if frame_size != expected_frame_size:
            raise ValueError(
                "control SHM frame_size does not match configured payload: "
                f"header={frame_size}, expected={expected_frame_size}"
            )

        buffer_size = int(buffer_header["buffer_size"])
        write_index = int(buffer_header["write_index"])
        current_frames_count = int(buffer_header["current_frames_count"])
        if buffer_size <= 0:
            raise ValueError("control SHM buffer_size must be positive")
        if not 0 <= write_index < buffer_size:
            raise ValueError(
                "control SHM write_index is outside the declared buffer: "
                f"write_index={write_index}, buffer_size={buffer_size}"
            )
        if not 0 <= current_frames_count <= buffer_size:
            raise ValueError(
                "control SHM current_frames_count is outside the declared buffer: "
                f"current_frames_count={current_frames_count}, buffer_size={buffer_size}"
            )
        required_buffer_length = (
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + buffer_size * frame_size
        )
        if int(buffer_length) < required_buffer_length:
            raise ValueError(
                "control SHM is smaller than its declared header layout: "
                f"size={buffer_length}, required={required_buffer_length}"
            )

        return {
            "action_dim": int(np.prod(target_shape)),
            "frame_size": frame_size,
            "target_shape": target_shape,
            "target_dtype": target_dtype,
            "payload_size": payload_size,
        }

    @staticmethod
    def _coerce_action_for_control_target(
        action: np.ndarray,
        robot_state: Dict[str, Any],
    ) -> np.ndarray:
        """Convert one float64 policy row to the target control payload."""
        target_shape = tuple(robot_state["target_shape"])
        target_dtype = np.dtype(robot_state["target_dtype"])
        action_dim = int(robot_state["action_dim"])
        raw_action = np.asarray(action)

        # Policy chunks are always [time, flattened_action_dim]. Accept the
        # native target shape too for direct callers, but reject arbitrary
        # equal-size layouts instead of silently reshaping them.
        if raw_action.shape == target_shape:
            shaped_action = raw_action
        elif raw_action.ndim == 1 and raw_action.shape == (action_dim,):
            shaped_action = raw_action.reshape(target_shape)
        else:
            raise ValueError(
                "policy action shape does not match configured control target: "
                f"action={raw_action.shape}, target={target_shape}, "
                f"transport_row=({action_dim},)"
            )
        if np.iscomplexobj(shaped_action):
            raise ValueError("policy action must not contain complex values")

        try:
            target_action = np.ascontiguousarray(shaped_action, dtype=target_dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"policy action cannot be converted to control dtype {target_dtype.name}"
            ) from exc
        if target_action.shape != target_shape:
            raise ValueError(
                "converted policy action shape does not match configured control target: "
                f"action={target_action.shape}, target={target_shape}"
            )
        if target_action.nbytes != int(robot_state["payload_size"]):
            raise ValueError(
                "converted policy action byte size does not match control payload: "
                f"action={target_action.nbytes}, expected={robot_state['payload_size']}"
            )
        return target_action

    def _create_policy_shm(self) -> None:
        """Create or connect to policy SHM for PolicyConnector to write action chunking using unified interface."""
        try:
            # Calculate total action dimension and frame sizes for each device
            device_count = len(self.robot_states)
            frame_sizes = []

            for robot_state in self.robot_states:
                action_dim = robot_state['action_dim']
                # Each device frame: [FRAME_HEADER] + [chunk_data]
                # policy_data is a float64 transport channel. The target SHM
                # dtype conversion happens only at the execution boundary.
                chunk_data_size = self.chunk_length * action_dim * 8
                frame_size = FRAME_HEADER_SIZE + chunk_data_size
                frame_sizes.append(frame_size)

            self.policy_device_count = device_count
            self.policy_frame_sizes = frame_sizes

            # Use unified SHM creation interface
            self.policy_shm = create_policy_shm(device_count, frame_sizes)

            logger.info(
                f"Policy header: {POLICY_HEADER_SIZE} bytes, Device headers: {device_count * DEVICE_HEADER_SIZE} bytes, Data: {sum(frame_sizes)} bytes, action_index: {ACTION_INDEX_SIZE} bytes (tail)")

            # Initialize header
            self._initialize_policy_header()
            # Initialize action_index slot
            self._write_action_index(0)

        except Exception as e:
            logger.error(f"Failed to create/connect policy SHM: {e}")
            raise

    def _initialize_policy_header(self) -> None:
        """Initialize policy SHM header with manager and device headers."""
        device_count = len(self.robot_states)

        # Pack manager header (timestamp will be updated when data is written)
        manager_header = pack_policy_header(device_count, 0)  # 0 timestamp initially

        # Write manager header to SHM
        self.policy_shm.buf[:POLICY_HEADER_SIZE] = manager_header

        # Pack and write device headers for each robot control
        for i, robot_state in enumerate(self.robot_states):
            robot_cfg = robot_state['device_info']
            device_name = robot_cfg["device_key"]

            action_dim = robot_state['action_dim']
            chunk_data_size = self.chunk_length * action_dim * 8
            frame_size = FRAME_HEADER_SIZE + chunk_data_size

            # Use device_name as device_type for policy SHM
            device_header = pack_device_header(
                device_type=device_name,
                device_class_id=0,  # Not relevant for policy SHM
                fps=self.execution_fps,
                data_dtype="float64",
                shape=[self.chunk_length, action_dim],  # chunk shape
                frame_size=frame_size,
                hardware_latency_ms=0.0
            )

            # Write device header to SHM
            device_offset = get_device_info_offset(i)
            self.policy_shm.buf[device_offset:device_offset + DEVICE_HEADER_SIZE] = device_header

    def _write_action_index(self, action_index: int) -> None:
        if not self.policy_shm or not self.policy_frame_sizes:
            return
        try:
            offset = get_policy_action_index_offset(self.policy_device_count, self.policy_frame_sizes)
            action_index_bytes = struct.pack(ACTION_INDEX_FORMAT, int(action_index))
            self.policy_shm.buf[offset:offset + ACTION_INDEX_SIZE] = action_index_bytes
        except Exception as exc:
            logger.error(f"Failed to write action_index to policy SHM: {exc}")

    def _consume_due_actions(
        self,
        dispatch_steps: int,
        *,
        chunk_replaced: bool,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], int, int, bool]:
        """Consume due manager rows and return only the newest row.

        ``BaseChunkManager(auto)`` has already aligned a replacement chunk to
        the current wall-clock latency.  Consuming all sim-time catch-up rows
        again would double-count that latency, so a replacement takes one
        current row.  An existing chunk, however, must advance once per missed
        sim-time action period or its internal action index falls behind the
        simulator.
        """
        if dispatch_steps <= 0 or self.chunk_manager.is_empty():
            return None, 0, 0, False

        manager_drop_mode = getattr(self.chunk_manager, "drop_first_actions", None)
        replacement_already_aligned = chunk_replaced and manager_drop_mode == "auto"
        requested_steps = 1 if replacement_already_aligned else max(1, int(dispatch_steps))

        # A stalled simulator can cross more deadlines than a configured chunk
        # contains.  The manager will return None once exhausted; this bound
        # prevents a long stall from turning into an unbounded catch-up loop.
        requested_steps = min(requested_steps, max(1, int(self.chunk_length)))
        latest_action: Optional[Dict[str, np.ndarray]] = None
        consumed_steps = 0
        for _ in range(requested_steps):
            action = self.chunk_manager.get()
            if action is None:
                break
            latest_action = action
            consumed_steps += 1

        return (
            latest_action,
            consumed_steps,
            requested_steps,
            replacement_already_aligned,
        )

    def _dispatch_metric_frequency(
        self,
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

    def _record_action_dispatch_metrics(
        self,
        dispatch_timestamp_ns: int,
        dispatch_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Record executor-side action cadence and deadline diagnostics."""
        self._action_dispatch_count = getattr(self, "_action_dispatch_count", 0) + 1
        if getattr(self, "_first_action_dispatch_wall_timestamp_ns", None) is None:
            self._first_action_dispatch_wall_timestamp_ns = int(dispatch_timestamp_ns)
        self._last_action_dispatch_wall_timestamp_ns = int(dispatch_timestamp_ns)
        sim_timestamp_ns = None
        if dispatch_context is not None:
            raw_sim_timestamp = dispatch_context.get("sim_time_ns")
            if raw_sim_timestamp is not None:
                sim_timestamp_ns = int(raw_sim_timestamp)
        if sim_timestamp_ns is None and self.execution_clock == "sim_time":
            sim_timestamp_ns = self._read_sim_clock_ns()
        if (
            sim_timestamp_ns is not None
            and getattr(self, "_first_action_dispatch_sim_timestamp_ns", None) is None
        ):
            self._first_action_dispatch_sim_timestamp_ns = int(sim_timestamp_ns)
        if sim_timestamp_ns is not None:
            self._last_action_dispatch_sim_timestamp_ns = int(sim_timestamp_ns)

        count = int(self._action_dispatch_count)
        fields: Dict[str, Any] = {
            "dispatch_count": count,
            "dispatch_emitted": True,
            "dispatch_skip_reason": None,
            "configured_control_fps": float(self.execution_fps),
            "effective_action_dispatch_hz_wall": self._dispatch_metric_frequency(
                count,
                getattr(self, "_first_action_dispatch_wall_timestamp_ns", None),
                int(dispatch_timestamp_ns),
            ),
            "effective_action_dispatch_hz_sim": self._dispatch_metric_frequency(
                count,
                getattr(self, "_first_action_dispatch_sim_timestamp_ns", None),
                getattr(self, "_last_action_dispatch_sim_timestamp_ns", None),
            ),
            "deadline_due_count": int(getattr(self, "_deadline_due_count", 0)),
            "deadline_miss_count": int(getattr(self, "_deadline_miss_count", 0)),
            "deadline_miss_rate": (
                float(getattr(self, "_deadline_miss_count", 0))
                / float(getattr(self, "_deadline_due_count", 0))
                if getattr(self, "_deadline_due_count", 0) > 0
                else None
            ),
        }
        if dispatch_context:
            fields.update(dispatch_context)
        return fields

    def _build_dispatch_context(
        self,
        dispatch_steps: int,
        *,
        chunk_replaced: bool,
        replacement_already_aligned: bool,
        requested_steps: int,
        consumed_steps: int,
        action_index: Optional[int],
        action_original_offset: Optional[float],
    ) -> Dict[str, Any]:
        """Build the common timing context for emitted and skipped dispatches."""
        return {
            "sim_time_ns": getattr(self, "_sim_dispatch_sim_time_ns", None),
            "sim_dispatch_steps_due": int(dispatch_steps),
            "sim_dispatch_skipped_deadlines": int(
                getattr(self, "_sim_dispatch_skipped_deadlines", 0)
            ),
            "sim_dispatch_deadline_ns": getattr(
                self, "_sim_dispatch_deadline_ns", None
            ),
            "sim_dispatch_lateness_ns": int(
                getattr(self, "_sim_dispatch_lateness_ns", 0)
            ),
            "chunk_replaced": bool(chunk_replaced),
            "chunk_replacement_wall_aligned": bool(replacement_already_aligned),
            "chunk_steps_requested": int(requested_steps),
            "chunk_steps_consumed": int(consumed_steps),
            "chunk_steps_skipped": max(
                0, int(requested_steps) - int(consumed_steps)
            ),
            "action_original_index": action_index,
            "action_original_offset": action_original_offset,
            "chunk_manager_current_step": int(
                getattr(self.chunk_manager, "current_step", -1)
            ),
        }

    def _log_action_dispatch_metrics(
        self,
        dispatch_timestamp_ns: int,
        fields: Dict[str, Any],
    ) -> None:
        # Keep the additional diagnostics on the non-blocking trace path.  The
        # existing action write already has a synchronous timestamp event, so
        # duplicating that I/O here would perturb the control loop.
        self.action_state_xyz_logger.log_meta(
            "action_dispatch_metrics",
            timestamp_ns=int(dispatch_timestamp_ns),
            **fields,
        )

    def _log_no_action_dispatch_metrics(
        self,
        dispatch_timestamp_ns: int,
        dispatch_context: Dict[str, Any],
        reason: str,
    ) -> None:
        """Record a due cycle that did not produce a robot SHM write."""
        count = int(getattr(self, "_action_dispatch_count", 0))
        fields: Dict[str, Any] = {
            "dispatch_count": count,
            "dispatch_emitted": False,
            "dispatch_skip_reason": reason,
            "configured_control_fps": float(self.execution_fps),
            "effective_action_dispatch_hz_wall": self._dispatch_metric_frequency(
                count,
                getattr(self, "_first_action_dispatch_wall_timestamp_ns", None),
                getattr(self, "_last_action_dispatch_wall_timestamp_ns", None),
            ),
            "effective_action_dispatch_hz_sim": self._dispatch_metric_frequency(
                count,
                getattr(self, "_first_action_dispatch_sim_timestamp_ns", None),
                getattr(self, "_last_action_dispatch_sim_timestamp_ns", None),
            ),
            "deadline_due_count": int(getattr(self, "_deadline_due_count", 0)),
            "deadline_miss_count": int(getattr(self, "_deadline_miss_count", 0)),
            "deadline_miss_rate": (
                float(getattr(self, "_deadline_miss_count", 0))
                / float(getattr(self, "_deadline_due_count", 0))
                if getattr(self, "_deadline_due_count", 0) > 0
                else None
            ),
        }
        fields.update(dispatch_context)
        self.action_state_xyz_logger.log_meta(
            "action_dispatch_metrics",
            timestamp_ns=int(dispatch_timestamp_ns),
            **fields,
        )

    def _update_execution(
        self,
        *,
        dispatch_ready: bool = True,
        dispatch_steps: Optional[int] = None,
    ) -> None:
        """
        Update execution loop - read action chunk, manage delays, and dispatch actions.

        This method handles the complete execution pipeline:
        1. Read new action chunks from policy SHM
        2. Get next action from chunk manager
        3. Enqueue actions to per-device delay queues
        4. Dequeue actions that have reached required delay
        5. Dispatch ready actions to robot control SHMs
        """
        try:
            # Check for reset command (from robot reset)
            self._check_and_handle_reset_command()

            if dispatch_steps is None:
                dispatch_steps = 1 if dispatch_ready else 0
            dispatch_steps = max(0, int(dispatch_steps))
            chunk_replaced = False

            # Check if we need to read a new action chunk
            chunk_result = self._read_latest_policy_frame()

            # If we got a chunk, check if it's new based on timestamp
            if chunk_result is not None:
                latest_chunk_dict, timestamp_ns, policy_write_timestamp_ns = chunk_result

                # IMPORTANT: After reset, only accept chunks with observation timestamp
                # AFTER the reset timestamp. This ensures the policy has completed at least
                # one fresh inference using observations from the new episode.
                if self.last_reset_timestamp > 0 and timestamp_ns <= self.last_reset_timestamp:
                    # Skip this chunk - it was inferred from old observations before reset
                    logger.debug(f"Skipping stale action chunk (obs_ts={timestamp_ns} <= reset_ts={self.last_reset_timestamp})")
                elif self.chunk_manager.is_empty() or self.chunk_manager.is_new_chunk(timestamp_ns):
                    # Accept chunk: either first read after restart, or new chunk timestamp
                    self.chunk_manager.put(latest_chunk_dict, timestamp_ns)
                    chunk_replaced = True
                    self._queue_action_chunk_log(
                        original_action_chunk_dict=latest_chunk_dict,
                        observation_timestamp_ns=int(timestamp_ns),
                        policy_write_timestamp_ns=int(policy_write_timestamp_ns),
                    )
            # Advance the action index to the current sim-time deadline.  When
            # a replacement chunk is auto-aligned in wall time, consume only
            # its current row; otherwise consume every crossed deadline and
            # dispatch the newest row.  Every due cycle gets a diagnostic,
            # including cycles that cannot produce a robot SHM write.
            if dispatch_ready and dispatch_steps > 0:
                if self.chunk_manager.is_empty():
                    self._log_no_action_dispatch_metrics(
                        time.time_ns(),
                        self._build_dispatch_context(
                            dispatch_steps,
                            chunk_replaced=chunk_replaced,
                            replacement_already_aligned=False,
                            requested_steps=0,
                            consumed_steps=0,
                            action_index=None,
                            action_original_offset=None,
                        ),
                        "no_action_chunk",
                    )
                else:
                    (
                        action_dict,
                        consumed_steps,
                        requested_steps,
                        replacement_already_aligned,
                    ) = self._consume_due_actions(
                        dispatch_steps,
                        chunk_replaced=chunk_replaced,
                    )
                    if action_dict is None:
                        self._log_no_action_dispatch_metrics(
                            time.time_ns(),
                            self._build_dispatch_context(
                                dispatch_steps,
                                chunk_replaced=chunk_replaced,
                                replacement_already_aligned=replacement_already_aligned,
                                requested_steps=requested_steps,
                                consumed_steps=consumed_steps,
                                action_index=None,
                                action_original_offset=None,
                            ),
                            "chunk_exhausted",
                        )
                    else:
                        action_index = None
                        action_original_offset = None
                        if self.enable_rtc and not hasattr(self.chunk_manager, "get_original_current_step"):
                            raise RuntimeError(
                                "RTC enabled but chunk_manager lacks "
                                "get_original_current_step; cannot write action_index."
                            )
                        if hasattr(self.chunk_manager, "get_original_current_step"):
                            action_index = self.chunk_manager.get_original_current_step()
                            if action_index is not None:
                                self._write_action_index(action_index)
                        if hasattr(self.chunk_manager, "get_original_current_offset"):
                            action_original_offset = self.chunk_manager.get_original_current_offset()
                        # Visualize action if enabled
                        if self.enable_action_visualizer:
                            self.action_visualizer.add_action(action_dict)

                        # === Device Delay Management ===
                        # Enqueue actions to per-device queues for delay management
                        for device_name, action in action_dict.items():
                            if device_name in self.device_action_queues:
                                self._enqueue_action_with_delays(device_name, action)

                        # Dequeue actions from queues that have reached required delay depth
                        ready_actions = {}
                        for device_name in self.device_action_queues.keys():
                            ready_action = self._dequeue_action_with_delays(device_name)
                            if ready_action is not None:
                                ready_actions[device_name] = ready_action

                        dispatch_context = self._build_dispatch_context(
                            dispatch_steps,
                            chunk_replaced=chunk_replaced,
                            replacement_already_aligned=replacement_already_aligned,
                            requested_steps=requested_steps,
                            consumed_steps=consumed_steps,
                            action_index=action_index,
                            action_original_offset=action_original_offset,
                        )

                        if ready_actions:
                            # logger.info(f"Dispatching {len(ready_actions)} actions to robots")
                            # logger.info(f"Ready actions: {ready_actions}")
                            switch_timestamp_ns = self._dispatch_action_to_robots(
                                ready_actions,
                                dispatch_context=dispatch_context,
                            )
                            if switch_timestamp_ns is not None:
                                self._log_pending_action_chunk(int(switch_timestamp_ns))
                            else:
                                self._log_no_action_dispatch_metrics(
                                    time.time_ns(),
                                    dispatch_context,
                                    "no_robot_write",
                                )
                        else:
                            self._log_no_action_dispatch_metrics(
                                time.time_ns(),
                                dispatch_context,
                                "device_delay_not_ready",
                            )

        except Exception as e:
            logger.error(f"Error in execution update: {e}")

    def _queue_action_chunk_log(
        self,
        original_action_chunk_dict: Dict[str, np.ndarray],
        observation_timestamp_ns: int,
        policy_write_timestamp_ns: int,
    ) -> None:
        """Snapshot both the original and post-manager chunks until dispatch."""
        aligned_chunk_dict = self.chunk_manager.get_current_action_chunk()
        if not aligned_chunk_dict:
            self._pending_action_chunk_log = None
            return

        aligned_original_action_offsets = (
            self.chunk_manager.get_current_original_action_offsets()
        )
        self._action_chunk_log_index += 1
        self._pending_action_chunk_log = {
            "pred_chunk_index": int(self._action_chunk_log_index),
            "original_action_chunk_dict": {
                device_name: np.asarray(chunk).copy()
                for device_name, chunk in original_action_chunk_dict.items()
            },
            "aligned_action_chunk_dict": aligned_chunk_dict,
            "aligned_original_action_offsets": (
                aligned_original_action_offsets.copy()
            ),
            "observation_timestamp_ns": int(observation_timestamp_ns),
            "policy_write_timestamp_ns": int(policy_write_timestamp_ns),
            "original_step_interval_ns": int(
                self.chunk_manager.get_original_action_step_interval_ns()
            ),
            "original_start_index": int(
                getattr(self.chunk_manager, "current_original_start_index", 0)
            ),
            "original_start_offset": float(
                getattr(self.chunk_manager, "current_original_start_offset", 0.0)
            ),
        }

    @staticmethod
    def _normalize_action_chunk_for_log(
        raw_chunk: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Normalize one action chunk to a non-empty time-by-action matrix."""
        chunk = np.asarray(raw_chunk)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            return None
        return chunk

    @staticmethod
    def _action_visualizer_layout(
        device_name: str,
        action: np.ndarray,
    ) -> Dict[str, str]:
        """Mark RoboTwin's fixed joint target instead of pretending it is xyz."""
        if (
            str(device_name).lower().startswith("robotwinsimrobot")
            and np.asarray(action).size == 14
        ):
            return {"vector_layout": "joint14"}
        return {}

    def _chunk_preview_timing_fields(
        self,
        *,
        switch_timestamp_ns: int,
        preview_offset_ns: int,
        timestamp_source: str,
    ) -> Dict[str, Any]:
        """Return timestamp fields for a predicted chunk sample.

        In sim-time dispatch mode, future chunk samples are estimated wall-clock
        execution points. The estimate uses the observed action dispatch wall
        interval when available, while preserving the nominal sim-time offset.
        """
        if self.execution_clock == "sim_time":
            nominal_interval_ns = max(1, int(self.action_step_interval_ns))
            estimated_wall_interval_ns = int(
                getattr(
                    self,
                    "_estimated_wall_action_step_interval_ns",
                    self.action_step_interval_ns,
                )
            )
            wall_preview_offset_ns = int(
                round(float(preview_offset_ns) * estimated_wall_interval_ns / nominal_interval_ns)
            )
            return {
                "timestamp_ns": int(switch_timestamp_ns) + wall_preview_offset_ns,
                "pred_chunk_timestamp_source": "estimated_wall_dispatch_time",
                "timestamp_semantics": "estimated_wall_dispatch_time",
                "sim_preview_offset_ns": int(preview_offset_ns),
                "sim_preview_offset_s": float(preview_offset_ns) / 1e9,
                "wall_preview_offset_ns": wall_preview_offset_ns,
                "wall_preview_offset_s": float(wall_preview_offset_ns) / 1e9,
                "estimated_wall_action_step_interval_ns": estimated_wall_interval_ns,
                "preview_original_timestamp_source": timestamp_source,
            }
        return {
            "timestamp_ns": int(switch_timestamp_ns) + int(preview_offset_ns),
            "pred_chunk_timestamp_source": timestamp_source,
        }

    def _update_wall_dispatch_interval_estimate(self, dispatch_timestamp_ns: int) -> None:
        if self.execution_clock != "sim_time":
            return
        previous_timestamp_ns = self._last_action_dispatch_wall_timestamp_ns
        self._last_action_dispatch_wall_timestamp_ns = int(dispatch_timestamp_ns)
        if previous_timestamp_ns is None:
            return
        observed_interval_ns = int(dispatch_timestamp_ns) - int(previous_timestamp_ns)
        if observed_interval_ns <= 0:
            return
        max_reasonable_interval_ns = max(self.action_step_interval_ns * 20, 1_000_000_000)
        if observed_interval_ns > max_reasonable_interval_ns:
            return
        previous_estimate_ns = int(self._estimated_wall_action_step_interval_ns)
        self._estimated_wall_action_step_interval_ns = int(
            round(0.8 * previous_estimate_ns + 0.2 * observed_interval_ns)
        )

    def _log_pending_action_chunk(self, switch_timestamp_ns: int) -> None:
        """Log original and aligned trajectories on one offset-aware timeline."""
        pending = self._pending_action_chunk_log
        if pending is None:
            return
        self._pending_action_chunk_log = None

        pred_chunk_index = int(pending["pred_chunk_index"])
        observation_timestamp_ns = int(pending["observation_timestamp_ns"])
        policy_write_timestamp_ns = int(pending["policy_write_timestamp_ns"])
        original_start_index = int(pending["original_start_index"])
        original_start_offset = float(pending["original_start_offset"])
        original_step_interval_ns = int(pending["original_step_interval_ns"])
        aligned_original_action_offsets = np.asarray(
            pending["aligned_original_action_offsets"],
            dtype=np.float64,
        )
        original_start_offset_ns = -int(round(original_start_offset * original_step_interval_ns))
        common_fields = {
            "pred_chunk_index": pred_chunk_index,
            "policy_write_timestamp_ns": policy_write_timestamp_ns,
            "observation_timestamp_ns": observation_timestamp_ns,
            "original_start_index": original_start_index,
            "original_start_offset": original_start_offset,
            "original_step_interval_ns": original_step_interval_ns,
            "aligned_step_interval_ns": self.action_step_interval_ns,
        }
        self.action_state_xyz_logger.log_meta(
            "chunk_switch",
            timestamp_ns=int(switch_timestamp_ns),
            device_name="action_executor",
            chunk_stage="action_executor_aligned",
            **common_fields,
        )

        original_action_chunk_dict = pending["original_action_chunk_dict"]
        aligned_action_chunk_dict = pending["aligned_action_chunk_dict"]
        original_chunk_lengths = {
            device_name: int(chunk.shape[0])
            for device_name, chunk in original_action_chunk_dict.items()
            if chunk.ndim > 0
        }
        for device_name, raw_chunk in original_action_chunk_dict.items():
            original_chunk = self._normalize_action_chunk_for_log(raw_chunk)
            if original_chunk is None:
                continue

            original_chunk_length = int(original_chunk.shape[0])
            aligned_chunk = self._normalize_action_chunk_for_log(
                aligned_action_chunk_dict.get(device_name, np.empty((0, 0)))
            )
            aligned_chunk_length = (
                0 if aligned_chunk is None else int(aligned_chunk.shape[0])
            )
            original_fields = {
                **common_fields,
                "chunk_stage": "policy_original",
                "chunk_length": original_chunk_length,
                "original_chunk_length": original_chunk_length,
                "aligned_chunk_length": aligned_chunk_length,
            }
            for event_type, step_index in (
                ("pred_chunk_start", 0),
                ("pred_chunk_end", original_chunk_length - 1),
            ):
                self.action_state_xyz_logger.log_meta(
                    event_type,
                    **self._chunk_preview_timing_fields(
                        switch_timestamp_ns=int(switch_timestamp_ns),
                        preview_offset_ns=(
                            original_start_offset_ns
                            + step_index * original_step_interval_ns
                        ),
                        timestamp_source="action_executor_offset_timeline",
                    ),
                    device_name=device_name,
                    action_step_index=step_index,
                    **original_fields,
                )

            for step_index, action in enumerate(original_chunk):
                self.action_state_xyz_logger.log_action(
                    device_name=device_name,
                    action=action,
                    **self._chunk_preview_timing_fields(
                        switch_timestamp_ns=int(switch_timestamp_ns),
                        preview_offset_ns=(
                            original_start_offset_ns
                            + step_index * original_step_interval_ns
                        ),
                        timestamp_source="action_executor_offset_timeline",
                    ),
                    event_type="all_actions",
                    action_step_index=step_index,
                    **self._action_visualizer_layout(device_name, action),
                    **original_fields,
                )

        for device_name, raw_chunk in aligned_action_chunk_dict.items():
            aligned_chunk = self._normalize_action_chunk_for_log(raw_chunk)
            if aligned_chunk is None:
                continue

            aligned_chunk_length = int(aligned_chunk.shape[0])
            original_chunk_length = int(
                original_chunk_lengths.get(device_name, aligned_chunk_length)
            )
            aligned_fields = {
                **common_fields,
                "chunk_stage": "action_executor_aligned",
                "chunk_length": aligned_chunk_length,
                "original_chunk_length": original_chunk_length,
                "aligned_chunk_length": aligned_chunk_length,
            }
            for step_index, action in enumerate(aligned_chunk):
                if step_index < aligned_original_action_offsets.shape[0]:
                    original_action_offset = float(
                        aligned_original_action_offsets[step_index]
                    )
                else:
                    original_action_offset = original_start_offset + step_index
                timestamp_source = (
                    "action_executor_dispatch"
                    if step_index == 0
                    else "action_executor_extrapolated"
                )
                self.action_state_xyz_logger.log_action(
                    device_name=device_name,
                    action=action,
                    **self._chunk_preview_timing_fields(
                        switch_timestamp_ns=int(switch_timestamp_ns),
                        preview_offset_ns=step_index * self.action_step_interval_ns,
                        timestamp_source=timestamp_source,
                    ),
                    event_type="aligned_actions",
                    action_step_index=step_index,
                    original_action_offset=original_action_offset,
                    **self._action_visualizer_layout(device_name, action),
                    **aligned_fields,
                )

    def _check_and_handle_reset_command(self) -> None:
        """Check for reset command file and clear action cache and SHM if present."""
        try:
            reset_command_file = Path(self._policy_runner_reset_command_path)

            if reset_command_file.exists():
                try:
                    # Record reset timestamp - only accept action chunks with observation
                    # timestamps AFTER this point to ensure fresh inference in new episode
                    self.last_reset_timestamp = time.time_ns()
                    logger.info(f"Reset timestamp set to {self.last_reset_timestamp}")

                    # Clear action cache in chunk manager
                    if hasattr(self.chunk_manager, 'reset'):
                        self.chunk_manager.reset()
                    else:
                        # Fallback: manually clear if reset method doesn't exist
                        self.chunk_manager.current_action_chunk_dict = None
                        self.chunk_manager.current_step = 0
                        if hasattr(self.chunk_manager, 'last_read_time'):
                            self.chunk_manager.last_read_time = 0
                    self._next_sim_dispatch_time_ns = None
                    self._pending_action_chunk_log = None
                    self._last_action_dispatch_wall_timestamp_ns = None
                    self._estimated_wall_action_step_interval_ns = self.action_step_interval_ns
                    if hasattr(self, "_pending_chunk_switch_observation_timestamp_ns"):
                        self._pending_chunk_switch_observation_timestamp_ns = None
                    logger.debug(f"Before clearing all device action queues, the chunk manager is empty: {self.chunk_manager.is_empty()}")

                    self._sim_dispatch_steps_due = 0
                    self._sim_dispatch_skipped_deadlines = 0
                    self._sim_dispatch_deadline_ns = None
                    self._sim_dispatch_lateness_ns = 0
                    self._sim_dispatch_sim_time_ns = None
                    self._deadline_due_count = 0
                    self._deadline_miss_count = 0
                    self._action_dispatch_count = 0
                    self._first_action_dispatch_wall_timestamp_ns = None
                    self._first_action_dispatch_sim_timestamp_ns = None
                    self._last_action_dispatch_sim_timestamp_ns = None

                    # Clear device action queues (delay management queues)
                    for device_name in self.device_action_queues.keys():
                        queue_structure = self.device_action_queues[device_name]
                        if isinstance(queue_structure, dict):
                            # Dimension-level delays: clear all dimension queues
                            for group_id in queue_structure.keys():
                                queue_structure[group_id].clear()
                        else:
                            # Whole-action delay: clear the single queue
                            queue_structure.clear()
                    logger.info(f"Cleared all device action queues for {len(self.device_action_queues)} devices")

                    # Clear device first actions cache
                    for device_name in self.device_first_actions.keys():
                        self.device_first_actions[device_name] = None
                    logger.info("Cleared all device first actions cache")

                    # Clear policy SHM and robot control SHMs
                    if self._clear_all_shm() is False:
                        logger.error(
                            "PolicyRunner reset remains pending because SHM cleanup failed"
                        )
                        return

                    # Reset action_index in policy SHM to 0 for RTC
                    # This ensures the policy doesn't use stale action_index from previous episode
                    self._write_action_index(0)
                    logger.info("Reset action_index to 0 for RTC")

                    reset_command_file.unlink()
                    logger.debug("PolicyRunner: Action cache, queues, and all SHM buffers cleared due to robot reset")
                except Exception as e:
                    logger.error(f"Error processing reset command file: {e}")
        except Exception as e:
            logger.debug(f"Error checking reset command: {e}")

    def _clear_all_shm(self) -> bool:
        """Clear all SHM buffers: policy SHM and robot control SHMs."""
        success = True
        try:
            # Clear policy SHM
            if self.policy_shm is not None:
                try:
                    buf = self.policy_shm.buf

                    # Read policy header
                    policy_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
                    device_count = policy_header['device_count']

                    # Reset policy header (timestamp=0)
                    reset_policy_header = pack_policy_header(device_count, 0)
                    buf[:POLICY_HEADER_SIZE] = reset_policy_header

                    # Calculate frame sizes
                    frame_sizes = []
                    for i in range(device_count):
                        device_offset = get_device_info_offset(i)
                        device_header = unpack_device_header(buf[device_offset:device_offset+DEVICE_HEADER_SIZE])
                        frame_sizes.append(device_header['frame_size'])

                    # Clear all device frames
                    for i in range(device_count):
                        data_offset = get_policy_data_offset(i, device_count, frame_sizes)
                        frame_size = frame_sizes[i]

                        # Set frame timestamp to 0
                        zero_timestamp_header = pack_frame_header(0)
                        buf[data_offset:data_offset+FRAME_HEADER_SIZE] = zero_timestamp_header

                        # Clear frame data
                        data_start = data_offset + FRAME_HEADER_SIZE
                        data_end = data_offset + frame_size
                        buf[data_start:data_end] = b'\x00' * (frame_size - FRAME_HEADER_SIZE)

                    logger.info("Cleared policy SHM buffer")
                except Exception as e:
                    success = False
                    logger.error(f"Failed to clear policy SHM: {e}")

            # Clear robot control SHMs
            for robot_state in self.robot_states:
                try:
                    robot_shm = robot_state['shm']
                    buf = robot_shm.buf

                    # Read buffer header
                    buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE])
                    buffer_size = buffer_header['buffer_size']
                    frame_size = robot_state['frame_size']

                    # Clear all frames
                    for i in range(buffer_size):
                        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * frame_size
                        # Set frame timestamp to 0
                        zero_timestamp_header = pack_frame_header(0)
                        buf[frame_offset:frame_offset+FRAME_HEADER_SIZE] = zero_timestamp_header
                        # Clear frame data
                        data_start = frame_offset + FRAME_HEADER_SIZE
                        data_end = frame_offset + frame_size
                        buf[data_start:data_end] = b'\x00' * (frame_size - FRAME_HEADER_SIZE)

                    # Reset buffer header
                    reset_buffer_header = pack_buffer_header(buffer_size, 0, 0)
                    buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE] = reset_buffer_header

                    shm_name = robot_state["shm_name"]
                    logger.info(f"Cleared robot control SHM buffer: {shm_name}")
                except Exception as e:
                    success = False
                    logger.error(f"Failed to clear robot control SHM: {e}")
        except Exception as e:
            success = False
            logger.error(f"Error clearing SHM buffers: {e}")
        return success

    def _enqueue_action_with_delays(self, device_name: str, action: np.ndarray) -> None:
        """
        Enqueue action with dimension-level delay support.

        If device has dimension-level delays configured, split the action and enqueue
        each dimension group separately. Otherwise, enqueue the whole action.

        Also stores the first action received for this device to use during delay period.
        """
        # Store first action if not already stored
        if self.device_first_actions[device_name] is None:
            self.device_first_actions[device_name] = action.copy()
            logger.info(f"Stored first action for {device_name} to use during delay period")

        queue_structure = self.device_action_queues[device_name]

        # Check if this device uses dimension-level delays
        if isinstance(queue_structure, dict):
            # Dimension-level delays
            dimension_delays = self.device_dimension_delays[device_name]

            # Track which dimensions have been assigned to groups
            assigned_dims = set()

            # Enqueue each dimension group
            for group_id, (indices, delay) in enumerate(dimension_delays):
                # Extract values for these dimensions
                dim_values = action[indices].copy()
                queue_structure[group_id].append(dim_values)
                assigned_dims.update(indices)
                logger.debug(f"Enqueued {device_name} dims {indices} to group {group_id}, "
                             f"queue_len={len(queue_structure[group_id])}, delay={delay}")

            # Enqueue remaining dimensions to default queue
            all_dims = set(range(len(action)))
            default_dims = sorted(list(all_dims - assigned_dims))
            if default_dims:
                default_values = action[default_dims].copy()
                queue_structure['default'].append(default_values)
                logger.debug(f"Enqueued {device_name} dims {default_dims} to default queue, "
                             f"queue_len={len(queue_structure['default'])}")
        else:
            # Whole-action delay (backward compatibility)
            queue_structure.append(action.copy())
            delay_frames = self.device_delays.get(device_name, 0)
            logger.debug(
                f"Enqueued action for {device_name}, queue length: {len(queue_structure)}, delay: {delay_frames}")

    def _dequeue_action_with_delays(self, device_name: str) -> Optional[np.ndarray]:
        """
        Dequeue action with dimension-level delay support.

        If device has dimension-level delays configured, dequeue from each dimension
        queue and reconstruct the full action. Otherwise, dequeue the whole action.

        During the delay period (when queues are not ready), dimensions that need delay
        will use values from the first action received, while other dimensions use current values.
        """
        queue_structure = self.device_action_queues[device_name]
        first_action = self.device_first_actions.get(device_name)

        # Check if this device uses dimension-level delays
        if isinstance(queue_structure, dict):
            # Dimension-level delays
            dimension_delays = self.device_dimension_delays[device_name]

            # Check which queues are ready and which are not
            queues_ready = {}
            for group_id, (indices, delay) in enumerate(dimension_delays):
                queues_ready[group_id] = len(queue_structure[group_id]) > delay
                if not queues_ready[group_id]:
                    logger.debug(f"Queue not ready for {device_name} group {group_id} (dims {indices}): "
                                 f"current={len(queue_structure[group_id])}, required={delay + 1}")

            # Check default queue (no delay)
            default_delay = self.device_delays.get(device_name, 0)
            default_ready = len(queue_structure['default']) > default_delay
            if not default_ready:
                logger.debug(f"Default queue not ready for {device_name}: "
                             f"current={len(queue_structure['default'])}, required={default_delay + 1}")

            # If no queues are ready at all, return None (no action to dispatch yet)
            if not any(queues_ready.values()) and not default_ready:
                return None

            # Reconstruct action: use first action as base, then fill in ready dimensions
            # Determine total dimensions
            if first_action is not None:
                total_dims = len(first_action)
            else:
                # Fallback: calculate from queue structure
                total_dims = 0
                for group_id, (indices, _) in enumerate(dimension_delays):
                    total_dims = max(total_dims, max(indices) + 1)
                if queue_structure['default']:
                    assigned_dims = set()
                    for indices, _ in dimension_delays:
                        assigned_dims.update(indices)
                    default_count = len(queue_structure['default'][0])
                    total_dims = max(total_dims, len(assigned_dims) + default_count)

            # Start with first action as base (for dimensions that are still in delay period)
            if first_action is not None:
                reconstructed_action = first_action.copy()
            else:
                reconstructed_action = np.zeros(total_dims, dtype=np.float64)

            assigned_dims = set()

            # Fill in dimension groups (only if queue is ready)
            for group_id, (indices, delay) in enumerate(dimension_delays):
                if queues_ready[group_id]:
                    # Queue is ready, use dequeued value
                    dim_values = queue_structure[group_id].popleft()
                    reconstructed_action[indices] = dim_values
                    assigned_dims.update(indices)
                    logger.debug(f"Dequeued {device_name} group {group_id} (dims {indices}), "
                                 f"remaining={len(queue_structure[group_id])}")
                else:
                    # Queue not ready, keep first action values (already in reconstructed_action)
                    assigned_dims.update(indices)
                    logger.debug(
                        f"Using first action for {device_name} group {group_id} (dims {indices}) - queue not ready")

            # Fill in default dimensions (only if queue is ready)
            all_dims = set(range(total_dims))
            default_dims = sorted(list(all_dims - assigned_dims))
            if default_dims:
                if default_ready:
                    # Queue is ready, use dequeued value
                    default_values = queue_structure['default'].popleft()
                    reconstructed_action[default_dims] = default_values
                    logger.debug(f"Dequeued {device_name} default (dims {default_dims}), "
                                 f"remaining={len(queue_structure['default'])}")
                else:
                    # Queue not ready, keep first action values
                    logger.debug(
                        f"Using first action for {device_name} default (dims {default_dims}) - queue not ready")

            return reconstructed_action
        else:
            # Whole-action delay (backward compatibility)
            delay_frames = self.device_delays.get(device_name, 0)

            if len(queue_structure) > delay_frames:
                ready_action = queue_structure.popleft()
                logger.debug(
                    f"Dequeued action for {device_name} (delay={delay_frames}, remaining in queue={len(queue_structure)})")
                return ready_action
            else:
                # During delay period, use first action if available
                if first_action is not None and len(queue_structure) > 0:
                    logger.debug(
                        f"Using first action for {device_name} during delay period (current: {len(queue_structure)}, required: {delay_frames + 1})")
                    return first_action.copy()
                else:
                    logger.debug(
                        f"Waiting for {device_name} queue to fill (current: {len(queue_structure)}, required: {delay_frames + 1})")
                    return None

    def _read_latest_policy_frame(
        self,
    ) -> Optional[Tuple[Dict[str, np.ndarray], int, int]]:
        """Read latest action chunk dict from policy SHM using standard manager/device structure."""
        try:
            if not self.policy_shm:
                return None

            buf = self.policy_shm.buf

            # Read manager header
            manager_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
            device_count = manager_header['device_count']
            update_timestamp = manager_header['update_timestamp']

            # If timestamp is 0, no data has been written yet
            if update_timestamp == 0:
                return None

            # Calculate frame sizes for get_data_offset
            frame_sizes = []
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset:device_offset + DEVICE_HEADER_SIZE])
                frame_sizes.append(device_header['frame_size'])

            # Read device headers and data
            action_chunk_dict = {}
            latest_timestamp = 0

            for i in range(device_count):
                # Read device header
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset:device_offset + DEVICE_HEADER_SIZE])

                device_name = device_header['device_type']
                shape = device_header['shape']  # [chunk_length, action_dim]
                frame_size = device_header['frame_size']

                # Read device data
                data_offset = get_policy_data_offset(i, device_count, frame_sizes)

                # Read frame header
                frame_header = unpack_frame_header(buf[data_offset:data_offset+FRAME_HEADER_SIZE])
                timestamp_ns = frame_header['timestamp_ns']

                # If timestamp is 0, no data for this device
                if timestamp_ns == 0:
                    continue

                # Read chunk data
                data_start = data_offset + FRAME_HEADER_SIZE
                data_end = data_start + (frame_size - FRAME_HEADER_SIZE)

                # Extract chunk data: buf is raw buffer, slicing works directly
                sliced_buf = buf[data_start:data_end]
                chunk_data = np.frombuffer(bytes(sliced_buf), dtype=np.float64)
                action_chunk = chunk_data.reshape(shape).copy()

                action_chunk_dict[device_name] = action_chunk
                latest_timestamp = max(latest_timestamp, timestamp_ns)

            if not action_chunk_dict:
                return None

            return action_chunk_dict, latest_timestamp, int(update_timestamp)

        except Exception as e:
            logger.error(f"Failed to read policy frame: {e}")
            return None

    def _dispatch_action_to_robots(
        self,
        action_dict: Dict[str, np.ndarray],
        *,
        dispatch_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        Dispatch action dict to robot control SHMs.

        This function simply writes the provided actions to their corresponding robot control SHMs.
        Delay management is handled in _update_execution().
        """
        if not self.robot_states or not action_dict:
            return None

        first_write_timestamp_ns: Optional[int] = None
        for robot_state in self.robot_states:
            # Map robot config to device name
            robot_cfg = robot_state['device_info']
            device_name = robot_cfg["device_key"]

            # Find corresponding action in the dict
            if device_name in action_dict:
                robot_action = action_dict[device_name]
                write_timestamp_ns = self._write_action_to_robot(
                    robot_state,
                    robot_action,
                    dispatch_context=dispatch_context,
                )
                if first_write_timestamp_ns is None and write_timestamp_ns is not None:
                    first_write_timestamp_ns = int(write_timestamp_ns)
                logger.debug(
                    f"Dispatched action to {device_name}: {robot_action[:3] if len(robot_action) > 3 else robot_action}")
            else:
                logger.debug(f"No action found for device {device_name} in this dispatch cycle")
        if first_write_timestamp_ns is not None:
            self._update_wall_dispatch_interval_estimate(first_write_timestamp_ns)
            metrics = self._record_action_dispatch_metrics(
                first_write_timestamp_ns,
                dispatch_context,
            )
            self._log_action_dispatch_metrics(first_write_timestamp_ns, metrics)
        return first_write_timestamp_ns

    def _write_action_to_robot(
        self,
        robot_state: Dict[str, Any],
        action: np.ndarray,
        *,
        dispatch_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Write action to robot control SHM."""
        try:
            target_action = self._coerce_action_for_control_target(action, robot_state)
            robot_shm = robot_state['shm']
            buf = robot_shm.buf

            # Read current buffer state
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            write_index = buffer_header['write_index']
            buffer_size = buffer_header['buffer_size']
            frame_size = robot_state['frame_size']
            payload_size = int(robot_state["payload_size"])
            if buffer_size <= 0 or not 0 <= write_index < buffer_size:
                raise ValueError(
                    "control SHM buffer header is invalid at write time: "
                    f"write_index={write_index}, buffer_size={buffer_size}"
                )
            if frame_size != FRAME_HEADER_SIZE + payload_size:
                raise ValueError(
                    "control SHM frame size changed after connection: "
                    f"frame_size={frame_size}, payload_size={payload_size}"
                )

            # Calculate frame offset and write frame header
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + write_index * frame_size
            current_time_ns = time.time_ns()

            # ``target_action`` has already been shaped and cast against the
            # destination SHM contract; no truncation or zero-padding is safe.
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + payload_size
            action_bytes = target_action.tobytes(order="C")
            if len(action_bytes) != payload_size:
                raise ValueError(
                    "converted policy action byte size changed before control write: "
                    f"action={len(action_bytes)}, expected={payload_size}"
                )

            frame_header = pack_frame_header(current_time_ns)
            buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = frame_header
            buf[data_start:data_end] = action_bytes

            # Update buffer header
            new_write_index = (write_index + 1) % buffer_size
            new_current_frames_count = min(buffer_header['current_frames_count'] + 1, buffer_size)
            new_buffer_header = pack_buffer_header(buffer_size, new_current_frames_count, new_write_index)
            buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = new_buffer_header
            robot_cfg = robot_state.get('device_info', {})
            shm_name = robot_state["shm_name"]
            device_name = robot_cfg["device_key"]
            timing_fields = dict(dispatch_context or {})
            self.timestamp_logger.log_event(
                "action_dispatched",
                device_name=device_name,
                shm_name=shm_name,
                action_timestamp_ns=current_time_ns,
                dispatch_timestamp_ns=current_time_ns,
                executed_action_timestamp_ns=current_time_ns,
                action_stage="executed",
                write_index=write_index,
                next_write_index=new_write_index,
                buffer_size=buffer_size,
                current_frames_count=new_current_frames_count,
                frame_offset=frame_offset,
                action=target_action,
                **timing_fields,
            )
            self.action_state_xyz_logger.log_action(
                device_name=device_name,
                action=target_action,
                timestamp_ns=current_time_ns,
                shm_name=shm_name,
                write_index=write_index,
                next_write_index=new_write_index,
                buffer_size=buffer_size,
                frame_offset=frame_offset,
                dispatch_timestamp_ns=current_time_ns,
                executed_action_timestamp_ns=current_time_ns,
                action_stage="executed",
                **timing_fields,
                **self._action_visualizer_layout(device_name, target_action),
            )
            return int(current_time_ns)

        except Exception as e:
            logger.error(f"Failed to write action to robot: {e}")
            return None
