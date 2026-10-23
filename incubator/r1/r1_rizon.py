#!/usr/bin/env python3
"""
Robot Device - Base robot class and PyBullet-based simulated robot.

Author: Jun Lv,Zixi Ying
"""

import time
import threading
import subprocess
import ctypes
import json
import math
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import NoneType
import numpy as np
import multiprocessing.shared_memory as shm
from typing import Optional, Dict, Any, Union, List, Literal, Mapping, Sequence, Tuple
import rerun as rr
from devices.base import BaseDevice
from devices.config import ShmEndpointConfig, StateFeedbackConfig
from devices.contracts import DeviceKind
from devices.robots.config import DUAL_SOURCE_SELECTOR_ROLES, RobotControlConfig
from devices.robots.rizon import RizonRobot as StructuredRizonRobot
from devices.flexiv_adapter import FlexivAdapter
from utils.shm_utils import (
    pack_device_header, pack_buffer_header, pack_frame_header,
    calculate_device_shm_size, DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
    get_control_frame_offset, get_control_frame_offsets,
    get_latest_control_frame_index, get_latest_control_frame_offset,
    get_dtype, create_control_shm, connect_to_control_shm, connect_to_device_shm,
    connect_to_shared_memory, unpack_device_header, unpack_buffer_header
)
from utils.logger_config import logger
from utils.time_control import precise_loop_timing
from utils.rizon_traj_planner_utils import (
    build_planner_start_kwargs,
    planner_utilized,
    traj_smooth_settings,
)
from utils.rizon_traj_run_recording import RizonTrajRunRecorder
from utils.lk_gripper_presets import (
    DEFAULT_LK_GRIPPER_TYPE,
    calculate_lk_gripper_max_width,
    get_lk_gripper_preset,
)
from incubator.r1.rizon_state_layout import (
    RizonDualArmStateLayout,
    validate_rizon_state_shape,
)
from utils.safety_bounds import SafetyBounds
from utils.transform import compute_deviation
from scipy.spatial.transform import Rotation as R
from utils.transform import scalar_transform

Gripper = None
LKGripperConfig = None
LK_GRIPPER_IMPORT_ERROR = None
REPO_ROOT = Path(__file__).resolve().parents[1]
__all__ = ["RizonRobotLKForceGated", "YamHeadRobot"]


def track_i2rt_inflight_call(source: str):
    """Decorate I2RT methods whose blocking duration should be watchdog-visible."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self._track_inflight_call(source):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


class BaseRobot(BaseDevice):
    """
    Base robot class that extends BaseDevice with control capabilities.
    Provides dual SHM buffers: one for state data (inherited) and one for control commands.
    """

    device_kind = DeviceKind.LOW_DIM_RW

    def __init__(self, device_class_id=0, *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 control: RobotControlConfig | Mapping[str, Any],
                 **kwargs):
        """
        Initialize the base robot.

        Args:
            device_class_id: Unique identifier for this robot instance
            data_shape: Shape of robot state data (e.g., joint positions, velocities)
            control_shape: Shape of control command data (e.g., target joint positions)
            fps: Control frequency in Hz
            data_dtype: Data type for state data (string or numpy dtype)
            control_dtype: Data type for control data (string or numpy dtype)
            buffer_size: Number of state frames to store in buffer
            control_buffer_size: Number of control commands to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
            source_selection: Logical command/teleoperation source endpoints are
                resolved by ``RobotControlConfig``; raw SHM names are not accepted.
            **kwargs: Additional arguments passed to BaseDevice
        """
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)
        control_config = RobotControlConfig.require(control)
        # Robot command executer fps
        self.command_fps = control_config.fps

        # Legacy async execution mode: when enabled, control commands are executed in a
        # dedicated command thread at ``command_fps`` (decoupled from the state loop), and
        # RizonRobot's Cartesian motion planner / trajectory recording are allowed to run.
        # When disabled (default), commands are executed synchronously inside the main state
        # loop (single-threaded), which avoids concurrent hardware (e.g. gripper serial) access.
        self.legacy_async_mode = control_config.execution_mode == "legacy_async"
        self._command_thread: Optional[threading.Thread] = None
        self._command_stop = threading.Event()

        # Robot-specific attributes
        self.control_config = control_config
        self.control_shape = control_config.format.shape
        self.control_dtype = control_config.format.dtype
        self.control_buffer_size = control_config.buffer_size

        # Control buffer management
        self.control_shared_memory = None
        self._owned_control_shm_names: set[str] = set()
        self._owned_control_channels: dict[str, Any] = {}
        self._external_control_shm_names: set[str] = set()

        # Control flag SHM (created by the configured selector owner).
        self._control_flag_shm = None
        self._control_flag_shm_name = None

        # Calculate control buffer layout
        self._calculate_control_buffer_layout()
        self._configure_source_selection()

    def _resolve_source_endpoint_name(self, endpoint: ShmEndpointConfig) -> str:
        if not isinstance(endpoint, ShmEndpointConfig):
            raise TypeError("source endpoint must be a validated ShmEndpointConfig")
        return endpoint.resolve_shm_name(
            owner_device=self.device_name,
            owner_device_class_id=self.device_class_id,
        )

    def _endpoint_is_owned_by_self(self, endpoint: ShmEndpointConfig) -> bool:
        if not isinstance(endpoint, ShmEndpointConfig):
            raise TypeError("source endpoint must be a validated ShmEndpointConfig")
        device = endpoint.device
        device_class_id = endpoint.device_class_id
        if device == "self":
            device = self.device_name
            if device_class_id is None:
                device_class_id = self.device_class_id
        return device == self.device_name and int(device_class_id) == self.device_class_id

    def _configure_source_selection(self) -> None:
        """Derive canonical SHM names from the typed source-selection contract."""
        selection = self.control_config.source_selection
        self.source_selection = selection
        self.source_selection_mode = str(selection.mode).casefold()
        if self.source_selection_mode == "single_source":
            source_items = {"source": selection.source}
        elif self.source_selection_mode == "dual_source":
            source_items = dict(selection.sources or {})
        else:  # Defensive: RobotControlConfig validates this already.
            raise ValueError(f"unsupported source_selection.mode: {selection.mode!r}")

        self._source_endpoints: dict[str, ShmEndpointConfig] = {}
        self._source_ports: dict[str, str] = {}
        self._source_names: dict[str, str] = {}
        self._source_name_to_role: dict[str, str] = {}
        for role, endpoint in source_items.items():
            if endpoint is None:
                continue
            name = self._resolve_source_endpoint_name(endpoint)
            self._source_endpoints[role] = endpoint
            self._source_ports[role] = endpoint.port
            self._source_names[role] = name
            self._source_name_to_role.setdefault(name, role)

        self._source_selector_endpoint = None
        self._source_selector_mapping: dict[str, str] = {}
        self._source_selector_owned_by_self = False
        if self.source_selection_mode == "dual_source":
            selector = selection.dual_source_selector
            if selector is None:
                raise ValueError("dual_source requires dual_source_selector")
            self._source_selector_endpoint = selector.endpoint
            self._control_flag_shm_name = self._resolve_source_endpoint_name(
                selector.endpoint
            )
            self._source_selector_owned_by_self = self._endpoint_is_owned_by_self(
                selector.endpoint
            )
            self._source_selector_mapping = dict(DUAL_SOURCE_SELECTOR_ROLES)

        self._owned_control_shm_names = {self.default_control_shm_name}
        self.control_shm_names = []
        for name in self._source_names.values():
            if name not in self.control_shm_names:
                self.control_shm_names.append(name)
        if self.default_control_shm_name not in self.control_shm_names:
            self.control_shm_names.append(self.default_control_shm_name)
        self._external_control_shm_names = {
            name
            for name in self.control_shm_names
            if name not in self._owned_control_shm_names
        }

    def _is_dual_source_mode(self) -> bool:
        return self.source_selection_mode == "dual_source"

    def _source_name_for_role(self, role: str = "source") -> Optional[str]:
        return self._source_names.get(role)

    def _source_port_for_role(self, role: str = "source") -> Optional[str]:
        return self._source_ports.get(role)

    def _read_source_selector_value(self) -> float:
        """Read the selector, defaulting safely to the policy source."""
        if not self._is_dual_source_mode():
            return 0.0
        if self._control_flag_shm is None and not self._source_selector_owned_by_self:
            self._connect_control_flag_shm()
        return self._read_control_flag()

    def _selected_source_role(self, selector_value: Optional[float] = None) -> str:
        if not self._is_dual_source_mode():
            return "source"
        value = self._read_source_selector_value() if selector_value is None else selector_value
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            numeric_value = 0.0
        if numeric_value == 0.0:
            selector_key = "0"
        elif numeric_value == 1.0:
            selector_key = "1"
        else:
            # Selector values are a strict binary protocol. Invalid values fail
            # closed to policy instead of being rounded into teleoperation.
            selector_key = "0"
        role = self._source_selector_mapping.get(selector_key)
        if role in self._source_names:
            return role
        return self._source_selector_mapping.get("0", "policy")

    def _selected_source_name(self, selector_value: Optional[float] = None) -> Optional[str]:
        return self._source_name_for_role(self._selected_source_role(selector_value))

    def _calculate_control_buffer_layout(self) -> None:
        """Calculate the layout of the control command buffer."""
        # Calculate control data size per command
        numpy_control_dtype = get_dtype(self.control_dtype)
        bytes_per_element = np.dtype(numpy_control_dtype).itemsize
        control_data_size = np.prod(self.control_shape) * bytes_per_element

        # Control frame header: timestamp (8 bytes)
        self.control_frame_header_size = FRAME_HEADER_SIZE
        self.control_frame_data_size = control_data_size
        self.control_frame_size = self.control_frame_header_size + self.control_frame_data_size

        # Pre-calculate control frame offsets
        self.control_frame_offsets = get_control_frame_offsets(
            self.control_buffer_size, self.control_frame_size
        )

        # Total control SHM size: device_header + buffer_header + control_frames
        self.total_control_shm_size = calculate_device_shm_size(self.control_buffer_size, self.control_frame_data_size)

        # Control shared memory name
        self.default_control_shm_name = f"{self.device_name}_{self.device_class_id}_control"

    def _create_control_shared_memory(self) -> None:
        """Create shared memory for control commands using unified interface."""
        # Phase 1: Create writable SHM for initialization
        control_shm = create_control_shm(
            device_name=self.device_name,
            device_class_id=self.device_class_id,
            buffer_size=self.control_buffer_size,
            frame_data_size=self.control_frame_data_size,
            readonly_after_creation=False  # Initially writable for initialization
        )

        # Store the robot-owned command channel alongside any external sources.
        self._owned_control_shm_names.add(self.default_control_shm_name)
        if self.control_shm_names is None:
            self.control_shm_names = [self.default_control_shm_name]
        elif self.default_control_shm_name not in self.control_shm_names:
            self.control_shm_names.append(self.default_control_shm_name)
        if self.control_shared_memory is None:
            self.control_shared_memory = {}
        self.control_shared_memory.update({self.default_control_shm_name: control_shm})
        self._owned_control_channels[self.default_control_shm_name] = control_shm

        logger.info(f"Created control shared memory: {self.default_control_shm_name} ({self.total_control_shm_size:,} bytes)")

        # Phase 2: Initialize control buffer header
        self._initialize_control_buffer_header()
        logger.info(f"Initialized control buffer header")

        # Phase 3: Convert to read-only after initialization
        from utils.shm_utils import make_readonly
        readonly_control_shm = make_readonly(control_shm)
        self.control_shared_memory[self.default_control_shm_name] = readonly_control_shm

        logger.info(f"Control SHM converted to READ-ONLY after initialization")

    def _connect_to_existing_control_shm(self, shm_name: Optional[str] = None) -> bool:
        """
        Connect to an existing control shared memory in READ-ONLY mode using unified interface.

        Args:
            shm_name: Name of the existing control SHM (if None, uses default name)

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try to connect to the SHM directly
            # Let the unified interface handle the connection logic
            self.control_shared_memory[shm_name] = connect_to_shared_memory(shm_name, read_only=True)
            logger.info(f"Connected to control SHM: {shm_name} (READ-ONLY)")

            # Validate the connected SHM
            if not self._validate_control_shm(shm_name):
                logger.warning(f"Control SHM validation failed for: {shm_name}")
                self.control_shared_memory[shm_name].close()
                self.control_shared_memory[shm_name] = None
                return False

            logger.info(f"Successfully connected to control source: {shm_name} (READ-ONLY)")
            return True

        except FileNotFoundError:
            logger.warning(f"Control shared memory not found: {shm_name}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to control shared memory {shm_name}: {e}")
            return False

    def _connect_control_flag_shm(self) -> None:
        """
        Connect to the configured external source-selector SHM.

        This process is a read-only consumer. It closes its local handle during
        teardown but never unlinks the selector owned by another device.
        """
        if self._source_selector_owned_by_self or not self._control_flag_shm_name:
            return
        try:
            from utils.shm_utils import connect_to_shared_memory

            self._control_flag_shm = connect_to_shared_memory(
                self._control_flag_shm_name,
                read_only=True
            )
            logger.info(f"Connected to control flag SHM: {self._control_flag_shm_name} (READ-ONLY)")

        except FileNotFoundError:
            logger.warning(f"Control flag SHM not found: {self._control_flag_shm_name}")
            self._control_flag_shm = None
        except Exception as e:
            logger.warning(f"Failed to connect to control flag SHM: {e}")
            self._control_flag_shm = None

    def _validate_control_shm(self, shm_name: Optional[str] = None) -> bool:
        """
        Validate the connected control shared memory.

        Returns:
            bool: True if validation successful, False otherwise
        """
        try:
            from utils.shm_utils import unpack_device_header, unpack_buffer_header

            names = [shm_name] if shm_name is not None else self.control_shm_names
            for name in names:
                if name is None or self.control_shared_memory.get(name) is None:
                    return False
                # Read and validate device header
                device_header_data = self.control_shared_memory[name].buf[:DEVICE_HEADER_SIZE]
                device_info = unpack_device_header(device_header_data)

                # Check control shape
                shm_shape = device_info['shape']
                if shm_shape != self.control_shape:
                    logger.error(f"Control shape mismatch: expected {self.control_shape}, got {shm_shape}")
                    return False

                # Check control dtype
                shm_dtype_str = device_info['data_dtype']  # dtype is at index 3
                if shm_dtype_str != self.control_dtype:
                    logger.error(f"Control dtype mismatch: expected {self.control_dtype}, got {shm_dtype_str}")
                    return False

                # Read and validate buffer header
                buffer_header_data = self.control_shared_memory[name].buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE]
                buffer_info = unpack_buffer_header(buffer_header_data)

                # Check buffer size
                shm_buffer_size = buffer_info['buffer_size']
                if shm_buffer_size != self.control_buffer_size:
                    logger.error(f"Control buffer size mismatch: expected {self.control_buffer_size}, got {shm_buffer_size}")
                    return False

                # Recalculate frame offsets based on connected SHM
                shm_frame_size = device_info['frame_size']  # frame_size is at index 5
                self.control_frame_size = shm_frame_size
                self.control_frame_data_size = shm_frame_size - FRAME_HEADER_SIZE

                self.control_frame_offsets = get_control_frame_offsets(
                    self.control_buffer_size, self.control_frame_size
                )

                logger.info(f"Control SHM validation successful")
                logger.info(f"  - Shape: {shm_shape}")
                logger.info(f"  - Dtype: {shm_dtype_str}")
                logger.info(f"  - Buffer size: {shm_buffer_size}")
                logger.info(f"  - Current count: {buffer_info['current_frames_count']}")
                logger.info(f"  - Write index: {buffer_info['write_index']}")

            return True

        except Exception as e:
            logger.error(f"Error validating control SHM: {e}")
            return False

    def _get_control_buffer_info(self,shm_name: str) -> Optional[tuple]:
        """
        Get current control buffer information from SHM.

        Returns:
            tuple: (current_frames_count, write_index) or (None, None) if error
        """
        try:
            from utils.shm_utils import unpack_buffer_header

            # Read buffer header from SHM
            buffer_header_data = self.control_shared_memory[shm_name].buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE]
            buffer_info = unpack_buffer_header(buffer_header_data)

            current_frames_count = buffer_info['current_frames_count']
            write_index = buffer_info['write_index']

            return current_frames_count, write_index

        except Exception as e:
            logger.error(f"Error reading control buffer info: {e}")
            return None, None

    def _initialize_control_buffer_header(self) -> None:
        """Initialize the control buffer header with metadata."""
        # Pack device header for control buffer
        device_header = pack_device_header(
            f"{self.device_name}_control", self.device_class_id, self.fps,
            self.control_dtype, self.control_shape, self.control_frame_size
        )

        # Pack buffer header (initialize with zeros)
        buffer_header = pack_buffer_header(self.control_buffer_size, 0, 0)

        # Write headers to control SHM
        self.control_shared_memory[self.default_control_shm_name].buf[:DEVICE_HEADER_SIZE] = device_header
        self.control_shared_memory[self.default_control_shm_name].buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE] = buffer_header

    def read_latest_control_command(self, shm_name: str, min_timestamp_ns: int = 0) -> Optional[np.ndarray]:
        """
        Read the latest control command from the control buffer.

        Args:
            shm_name: Name of the shared memory to read from
            min_timestamp_ns: Minimum valid timestamp (in nanoseconds). If the action's
                           timestamp is less than this value, it is considered stale and
                           (None, None) is returned. Default is 0 (no filtering).

        Returns:
            tuple: (timestamp_ns, control_array) or (None, None) if no command available
                   or if the action is older than min_timestamp_ns
        """
        if not self.control_shared_memory[shm_name]:
            return None, None

        # Get current buffer info from SHM
        current_frames_count, write_index = self._get_control_buffer_info(shm_name)
        if current_frames_count is None or current_frames_count == 0:
            return None, None

        try:
            # Calculate frame offset for latest command
            latest_index = get_latest_control_frame_index(write_index, self.control_buffer_size)
            frame_offset = self.control_frame_offsets[latest_index]

            # Read frame header
            frame_header = pack_frame_header(0)  # Placeholder
            frame_header_data = self.control_shared_memory[shm_name].buf[frame_offset:frame_offset+FRAME_HEADER_SIZE]
            timestamp_ns = int.from_bytes(frame_header_data, byteorder='little')

            # Check if action is stale (written before the minimum valid timestamp)
            if min_timestamp_ns > 0 and timestamp_ns > 0 and timestamp_ns < min_timestamp_ns:
                return None, None

            # Read control data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + self.control_frame_data_size
            numpy_control_dtype = get_dtype(self.control_dtype)
            # Extract control data: DataChannel.buf returns raw buffer, slicing works directly
            sliced_buf = self.control_shared_memory[shm_name].buf[data_start:data_end]
            control_array = np.frombuffer(sliced_buf, dtype=numpy_control_dtype)
            # Reshape if needed
            if len(self.control_shape) > 0:
                control_array = control_array.reshape(self.control_shape)

            return timestamp_ns, control_array

        except Exception as e:
            logger.error(f"Error reading control command: {e}")
            return None, None

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state. To be implemented by subclasses.

        Returns:
            numpy.ndarray: Current robot state
        """
        raise NotImplementedError("Subclasses must implement get_robot_state()")

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command. To be implemented by subclasses.

        Args:
            control_array: Control command to execute
        """
        raise NotImplementedError("Subclasses must implement execute_control_command()")

    def _cleanup_control_shared_memory(self) -> None:
        """Clean up control shared memory resources."""
        if self.control_shared_memory is None:
            return
        for shm_name, control_shm in list(self.control_shared_memory.items()):
            if control_shm is not None:
                try:
                    control_shm.close()
                except Exception as exc:
                    logger.warning(f"Failed to close control SHM {shm_name}: {exc}")

                if shm_name in self._owned_control_shm_names:
                    owner_channel = self._owned_control_channels.get(shm_name)
                    try:
                        if owner_channel is not None and owner_channel is not control_shm:
                            owner_channel.close()
                        if owner_channel is not None:
                            owner_channel.unlink()
                        else:
                            control_shm.unlink()
                        logger.info(f"Cleaned up owned control shared memory: {shm_name}")
                    except (FileNotFoundError, PermissionError):
                        pass  # Already cleaned up or this handle is read-only.
                    except Exception as e:
                        logger.error(f"Error unlinking owned control shared memory: {e}")
            self.control_shared_memory[shm_name] = None
        self._owned_control_channels.clear()

    def _check_and_execute_home_command(self) -> None:
        """
        Check for external home command file and execute if present.

        Override in subclasses to implement robot-specific homing behavior.
        Base implementation does nothing.
        """
        pass

    def _clear_control_shm(self) -> None:
        """
        Clear all data in robot control SHM buffers.

        Override in subclasses to implement robot-specific SHM clearing.
        Base implementation does nothing.
        """
        pass

    def _notify_policy_runner_reset(self) -> None:
        """
        Notify policy runner to clear action cache and SHM when robot is resetting.

        Override in subclasses to implement notification mechanism.
        Base implementation does nothing.
        """
        pass

    def _on_motion_planner_loop_tick(self) -> None:
        """Hook invoked once per main control-loop iteration; override in subclasses."""
        pass

    def _on_idle_control_tick(self) -> None:
        """Hook invoked when no fresh control frame is available."""
        pass

    def _read_control_dict_for_execute(self) -> Dict[str, Optional[np.ndarray]]:
        """Read latest control frames from all control SHMs (same filters as main loop)."""
        control_dict: Dict[str, Optional[np.ndarray]] = {}
        if self.control_shm_names is None:
            return control_dict
        policy_start_ts = getattr(self, "_policy_control_start_timestamp", 0)
        grace_period_ns = getattr(self, "_policy_control_grace_period_ns", int(0.5 * 1e9))
        last_home_command_timestamp = getattr(self, "_last_home_command_timestamp", 0)
        min_timestamp_ns = policy_start_ts + grace_period_ns if policy_start_ts > 0 else 0
        min_timestamp_ns = max(min_timestamp_ns, last_home_command_timestamp)
        for shm_name in self.control_shm_names:
            _timestamp_ns, control_array = self.read_latest_control_command(
                shm_name, min_timestamp_ns
            )
            control_dict[shm_name] = control_array
        return control_dict

    def _poll_and_execute_control_once(self) -> None:
        """Run one control tick and delegate idle behavior to subclasses."""
        # While a go-home is running, stay idle: do not execute teleop frames and
        # do not re-stream the (stale, pre-home) hold target — either would fight
        # the homing motion. The homing routine drives the arm and re-pins the
        # hold target to the home pose itself.
        if getattr(self, "_home_command_active", False):
            return
        control_dict = self._read_control_dict_for_execute()
        if any(value is not None for value in control_dict.values()):
            self.execute_control_command(control_dict)
        else:
            self._on_idle_control_tick()

    def _update_source_selection_transition(self, *, context: str) -> None:
        """Apply source-switch side effects using configured source roles."""
        if not self._is_dual_source_mode():
            return

        current_selector_value = self._read_source_selector_value()
        if not hasattr(self, "_last_control_flag"):
            self._last_control_flag = current_selector_value
            return

        previous_role = self._selected_source_role(self._last_control_flag)
        current_role = self._selected_source_role(current_selector_value)
        if previous_role == current_role:
            self._last_control_flag = current_selector_value
            return

        if previous_role == "teleoperation" and current_role == "policy":
            self._clear_control_shm()
            self._policy_control_start_timestamp = time.time_ns()
            logger.info(
                f"Control source switch ({context}): teleoperation -> policy; "
                f"cleared owned command SHM, timestamp={self._policy_control_start_timestamp}"
            )

        if hasattr(self, "_toggle_control_mode") and getattr(self, "is_follower_role", False):
            self._toggle_control_mode()

        self._last_control_flag = current_selector_value

    def _command_loop_worker(self) -> None:
        """Dedicated thread: single-step at command_fps."""
        wait_for_next_command = precise_loop_timing(1.0 / float(self.command_fps))
        while self.running and not self._command_stop.is_set():
            try:
                self._update_source_selection_transition(context="command thread")
                self._poll_and_execute_control_once()
            except Exception as exc:
                logger.error(f"Error in command loop for {self.device_name}_{self.device_class_id}: {exc}")
            wait_for_next_command()

    def _start_command_loop_thread(self) -> None:
        if self._command_thread is not None and self._command_thread.is_alive():
            return
        self._command_stop.clear()
        self._command_thread = threading.Thread(
            target=self._command_loop_worker,
            name=f"{self.device_name}_{self.device_class_id}_command",
            daemon=True,
        )
        self._command_thread.start()

    def _stop_command_loop_thread(self) -> None:
        self._command_stop.set()
        if self._command_thread is not None:
            self._command_thread.join(timeout=2.0)
            self._command_thread = None

    def _prepare_control_loop_resources(self) -> None:
        """Create this robot's command channel and attach configured sources.

        Every robot owns its default ``command`` endpoint even when the active
        source is an external state or command producer.  External endpoints
        are opened read-only and are never unlinked by this process.
        """
        if self.running:
            logger.info(f"Robot {self.device_name}_{self.device_class_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_class_id} server...")
        self.running = True
        try:
            self._create_shared_memory()
            self.control_shared_memory = {
                shm_name: None for shm_name in self.control_shm_names
            }
            self._create_control_shared_memory()

            for shm_name in self.control_shm_names:
                if shm_name not in self._external_control_shm_names:
                    continue
                connected = self._connect_to_existing_control_shm(shm_name)
                if not connected:
                    # Peer device processes may create their endpoint shortly
                    # after this robot starts.
                    for _ in range(20):
                        time.sleep(0.2)
                        if self._connect_to_existing_control_shm(shm_name):
                            connected = True
                            break
                if not connected:
                    logger.warning(
                        f"Failed to connect configured source SHM after retries: {shm_name}"
                    )

            if self._is_dual_source_mode() and not self._source_selector_owned_by_self:
                self._connect_control_flag_shm()

            if (
                self.shared_memory is None
                or self.control_shared_memory.get(self.default_control_shm_name) is None
            ):
                raise RuntimeError("Failed to create robot-owned shared memories")
        except Exception:
            self.running = False
            raise

    def start_server(self) -> None:
        """Start the robot server and run control loop."""
        self._prepare_control_loop_resources()
        if not self.running:
            return

        logger.info(f"Robot {self.device_name}_{self.device_class_id} server started successfully")

        if self.legacy_async_mode:
            self._start_command_loop_thread()
            logger.info(
                "Started precise command thread at {:.3f} Hz (state loop {:.3f} Hz)",
                self.command_fps,
                self.fps,
            )

        # Main control loop
        try:
            assert self.fps >= self.command_fps
            iteration = 1
            command_iteration = int(self.fps // self.command_fps)
            while self.running:
                start_time = time.time()

                # Check for external home command (from GUI reset)
                # Only check periodically to avoid performance impact
                if hasattr(self, '_last_home_check_time'):
                    if time.time() - self._last_home_check_time > 0.1:  # Check every 100ms
                        self._check_and_execute_home_command()
                        self._last_home_check_time = time.time()
                else:
                    self._last_home_check_time = time.time()
                    self._check_and_execute_home_command()

                if hasattr(self, '_check_and_execute_teleop_start_command'):
                    if hasattr(self, '_last_teleop_start_check_time'):
                        if time.time() - self._last_teleop_start_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_start_command()
                            self._last_teleop_start_check_time = time.time()
                    else:
                        self._last_teleop_start_check_time = time.time()
                        self._check_and_execute_teleop_start_command()

                if hasattr(self, '_check_and_execute_teleop_discard_command'):
                    if hasattr(self, '_last_teleop_discard_check_time'):
                        if time.time() - self._last_teleop_discard_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_discard_command()
                            self._last_teleop_discard_check_time = time.time()
                    else:
                        self._last_teleop_discard_check_time = time.time()
                        self._check_and_execute_teleop_discard_command()
                # In legacy async mode, control execution runs in the dedicated command
                # thread; the main loop only publishes state. Otherwise, execute commands
                # synchronously here (single-threaded), throttled to command_fps.
                if not self.legacy_async_mode:
                    # Update source-switch timestamps before reading command frames.
                    self._update_source_selection_transition(context="main loop")
                    if iteration % command_iteration == 0:
                        self._poll_and_execute_control_once()
                    iteration = iteration % command_iteration + 1
                # Get current robot state
                state_array = self.get_robot_state()
                # Write state to shared memory
                if state_array is not None:
                    self._write_array_to_shm_with_timestamp(state_array, time.time_ns())

                if self.legacy_async_mode:
                    self._on_motion_planner_loop_tick()

                # Maintain control frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info(f"Stopping {self.device_name}_{self.device_class_id} server...")
        except Exception as e:
            logger.error(f"Error in robot control loop: {e}")
        finally:
            self.stop_server()

    def stop_server(self) -> None:
        """Stop the robot server and clean up resources."""
        logger.info(f"=== Stopping {self.device_name}_{self.device_class_id} server... in BaseRobot self.running={self.running} ===")
        if not self.running:
            return

        self.running = False
        if self.legacy_async_mode:
            self._stop_command_loop_thread()
        self._cleanup_shared_memory()
        self._cleanup_control_shared_memory()

        # Close connection to control flag SHM (owned by teleoperator, don't unlink)
        if self._control_flag_shm is not None:
            try:
                self._control_flag_shm.close()
                logger.info(f"Closed control flag SHM connection: {self._control_flag_shm_name}")
            except Exception as e:
                logger.warning(f"Failed to close control flag SHM connection: {e}")
            finally:
                self._control_flag_shm = None

        logger.info(f"Stopped {self.device_name}_{self.device_class_id} server")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop_server()


class SimRobot(BaseRobot):
    """
    Simulated robot using PyBullet for physics simulation.
    Controls a robotic arm and provides joint state feedback.
    """

    def __init__(self, device_class_id=0, *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 control: RobotControlConfig | Mapping[str, Any],
                 urdf_path=None, joint_indices=None, enable_gui=False):
        """
        Initialize the simulated robot.

        Args:
            device_class_id: Unique identifier for this robot instance
            urdf_path: Path to robot URDF file (if None, uses default KUKA iiwa)
            joint_indices: List of joint indices to control (if None, uses all joints)
            data_shape: Shape of robot state data (joint positions, velocities, etc.)
            control_shape: Shape of control command data (target joint positions)
            fps: Control frequency in Hz
            data_dtype: Data type for state data
            control_dtype: Data type for control data
            buffer_size: Number of state frames to store in buffer
            control_buffer_size: Number of control commands to store in buffer
            enable_gui: Enable PyBullet GUI for visualization
            source_selection: Logical endpoint supplying the active control frame
            enable_eef_control: Enable end-effector control
            **kwargs: Additional arguments passed to BaseRobot
        """
        # Initialize base robot
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback,
                         control=control)

        # Robot-specific attributes
        self.device_name = "SimRobot"
        self.urdf_path = urdf_path
        self.joint_indices = joint_indices
        self.enable_gui = enable_gui

        # End-effector control attributes
        self.ctrl_type = self.control_config.type
        assert self.ctrl_type in ["joints", "eef_rel", "eef_abs"]

        # IK method selection (only PyBullet supported now)
        self.ik_method = "pybullet"

        # Recalculate SHM names with correct device_name
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"
        self.default_control_shm_name = f"{self.device_name}_{self.device_class_id}_control"

        # PyBullet simulation
        self.physics_client = None
        self.robot_id = None
        self.num_joints = 0
        self.joint_names = []
        self.joint_limits = []

        # Initialize PyBullet
        self._initialize_pybullet()

    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation."""
        try:
            import pybullet as p
            import pybullet_data

            # Connect to PyBullet
            if self.enable_gui:
                self.physics_client = p.connect(p.GUI)  # Use GUI for visualization
                logger.info("PyBullet GUI enabled for visualization")
            else:
                self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless simulation
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Set physics parameters
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1.0 / self.fps)

            # Load robot URDF
            if self.urdf_path is None:
                # Use default KUKA iiwa robot
                self.urdf_path = "kuka_iiwa/model.urdf"

            self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0])
            logger.info(f"Loaded robot URDF: {self.urdf_path}")

            # Get joint information
            self.num_joints = p.getNumJoints(self.robot_id)
            logger.info(f"Robot has {self.num_joints} joints")

            # Get joint names and limits
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                self.joint_names.append(joint_info[1].decode('utf-8'))

                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))

            # Set joint indices to control (default: all revolute joints)
            if self.joint_indices is None:
                self.joint_indices = []
                for i in range(self.num_joints):
                    joint_info = p.getJointInfo(self.robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.joint_indices.append(i)

            logger.info(f"Controlling joints: {self.joint_indices}")
            logger.info(f"Joint names: {[self.joint_names[i] for i in self.joint_indices]}")

            # Set initial joint positions to zero
            initial_positions = [0.0] * len(self.joint_indices)
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0)



            logger.info("PyBullet simulation initialized successfully")
            link_state = p.getLinkState(self.robot_id, 6, computeForwardKinematics=True)
            self.init_pose = np.concatenate([np.array(link_state[4]), np.array(link_state[5])]) # in world frame

        except Exception as e:
            logger.error(f"Error initializing PyBullet: {e}")
            raise

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state (joint positions and velocities).

        Returns:
            numpy.ndarray: Current robot state [joint_positions, joint_velocities]
        """
        try:
            import pybullet as p

            if self.robot_id is None:
                return None

            # Get joint states
            joint_states = p.getJointStates(self.robot_id, self.joint_indices)

            # Extract positions and velocities
            positions = [state[0] for state in joint_states]
            velocities = [state[1] for state in joint_states]

            # Combine into state array
            numpy_data_dtype = get_dtype(self.data_dtype)
            state_array = np.array(positions + velocities, dtype=numpy_data_dtype)

            # Reshape if needed
            if len(self.data_shape) > 0:
                state_array = state_array.reshape(self.data_shape)

            return state_array

        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return None

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command (set target joint positions).

        Args:
            control_array: Target joint positions
        """
        try:
            import pybullet as p

            if self.robot_id is None:
                return

            selected_name = self._selected_source_name()
            control_array = (
                control_dict.get(selected_name)
                if selected_name is not None
                else None
            )
            if control_array is None:
                p.stepSimulation()
                return
            # Ensure control array has correct shape
            if len(control_array.shape) > 0:
                control_data = control_array.flatten()
            else:
                control_data = [control_array]

            # If end-effector control is enabled, use inverse kinematics
            if self.ctrl_type == "eef_abs":
                target_positions = self._compute_ik_iterative_single_arm(control_data, abs=True)
            elif self.ctrl_type == "eef_rel":
                target_positions = self._compute_ik_iterative_single_arm(control_data, abs=False)
            else:
                target_positions = control_data

            # Limit the number of positions to the number of controlled joints
            num_controlled_joints = len(self.joint_indices)
            if len(target_positions) > num_controlled_joints:
                target_positions = target_positions[:num_controlled_joints]
            elif len(target_positions) < num_controlled_joints:
                # Pad with current positions if not enough targets provided
                current_states = p.getJointStates(self.robot_id, self.joint_indices)
                current_positions = [state[0] for state in current_states]
                target_positions.extend(current_positions[len(target_positions):])

            # Apply joint limits
            for i, (joint_idx, target_pos) in enumerate(zip(self.joint_indices, target_positions)):
                lower_limit, upper_limit = self.joint_limits[joint_idx]
                target_positions[i] = np.clip(target_pos, lower_limit, upper_limit)

            # Reset joint states
            for joint_idx, target_pos in zip(self.joint_indices, target_positions):
                p.resetJointState(self.robot_id, joint_idx, target_pos, 0.0)

            # Step simulation
            p.stepSimulation()

        except Exception as e:
            logger.error(f"Error executing control command: {e}")

    def _compute_ik_iterative_single_arm(self, control_data,abs=True):
        """
        Compute inverse kinematics iteratively using PyBullet's calculateInverseKinematics for single arm.

        Args:
            control_data: End-effector pose [pos, orn] where pos is [x, y, z] and orn is [x, y, z, w] (quaternion)

        Returns:
            target_positions: Joint positions for the arm
        """
        try:
            import pybullet as p

            # Assume control_data contains [pos(3), orn(4)] = 7 elements minimum
            if len(control_data) < 7:  # 3+4 = 7 minimum
                logger.warning("Insufficient control data for end-effector control")
                return control_data

            # Extract target pose
            target_pos = control_data[0:3]
            target_orn = control_data[3:7]
            target_pos,target_orn = compute_deviation([0,0,0],[0,0,0,0],target_pos, target_orn) if abs else compute_deviation(self.init_pose[0:3],self.init_pose[3:7],target_pos, target_orn)

            # Get end-effector link index (assume last link)
            end_effector_link = self.num_joints - 1

            # Iterative IK parameters - relaxed tolerances for better convergence
            max_iterations = 20  # Reduced outer iteration count
            position_tolerance = 1e-3  # Relaxed position tolerance to 1mm
            orientation_tolerance = 1e-2  # Relaxed orientation tolerance

            # Get current joint positions as initial guess
            current_joint_states = p.getJointStates(self.robot_id, self.joint_indices)
            q0 = [state[0] for state in current_joint_states]

            # Iterative IK with PyBullet
            for iteration in range(max_iterations):
                # Calculate IK with PyBullet
                joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    end_effector_link,
                    target_pos,
                    target_orn,
                    lowerLimits=[self.joint_limits[i][0] for i in self.joint_indices],
                    upperLimits=[self.joint_limits[i][1] for i in self.joint_indices],
                    jointRanges=[self.joint_limits[i][1] - self.joint_limits[i][0] for i in self.joint_indices],
                    restPoses=q0,
                    maxNumIterations=10,
                    residualThreshold=1e-5
                )

                # Convert to list and apply joint limits
                joint_positions = list(joint_positions[:len(self.joint_indices)])
                for i, joint_idx in enumerate(self.joint_indices):
                    if joint_idx < len(self.joint_limits):
                        lower_limit, upper_limit = self.joint_limits[joint_idx]
                        joint_positions[i] = np.clip(joint_positions[i], lower_limit, upper_limit)

                # Check error
                link_state = p.getLinkState(self.robot_id, end_effector_link, computeForwardKinematics=True)
                current_pos = link_state[4]
                current_orn = link_state[5]

                pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                dot_product = np.clip(np.dot(current_orn, target_orn), -1.0, 1.0)
                orn_error = 2 * np.sqrt(max(0.0, 1 - dot_product ** 2))

                # Check convergence - using relaxed tolerances
                if pos_error < position_tolerance:  # Focus mainly on position convergence
                    break

            # Print final orientation error
            if joint_positions is not None:
                # Calculate final error after IK iterations
                final_link_state = p.getLinkState(self.robot_id, end_effector_link, computeForwardKinematics=True)
                final_current_pos = final_link_state[4]
                final_current_orn = final_link_state[5]

                final_pos_error = np.linalg.norm(np.array(final_current_pos) - np.array(target_pos))
                final_dot_product = np.clip(np.dot(final_current_orn, target_orn), -1.0, 1.0)
                final_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_dot_product ** 2))
                final_orn_error_deg = np.degrees(final_orn_error_rad)

                # Take only the controlled joints
                controlled_positions = joint_positions[:len(self.joint_indices)]
                return list(controlled_positions)
            else:
                logger.warning("IK computation failed, using original control data")
                return control_data

        except Exception as e:
            logger.error(f"Error in iterative IK computation: {e}")
            return control_data

    def stop_server(self) -> None:
        """Stop the robot server and clean up PyBullet resources."""
        if self.physics_client is not None:
            import pybullet as p
            p.disconnect(self.physics_client)
            self.physics_client = None
            self.robot_id = None
            logger.info("PyBullet simulation disconnected")

        super().stop_server()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop_server()


class DualArmSimRobot(BaseRobot):
    """
    Dual-arm simulated robot using PyBullet for physics simulation.
    Controls two robotic arms and provides joint state feedback for both arms.
    """

    def __init__(self, device_class_id=0, *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 control: RobotControlConfig | Mapping[str, Any],
                 urdf_path=None, joint_indices=None, enable_gui=False,
                 left_arm_position=[0.2, 0, 1.2],
                 right_arm_position=[-0.2, 0, 1.2]):
        """
        Initialize the dual-arm simulated robot.

        Args:
            device_class_id: Unique identifier for this robot instance
            urdf_path: Path to robot URDF file (if None, uses default Flexiv Rizon4)
            joint_indices: List of joint indices to control (if None, uses all joints)
            data_shape: Shape of robot state data (joint positions, velocities for both arms)
            control_shape: Shape of control command data (target joint positions for both arms)
            fps: Control frequency in Hz
            data_dtype: Data type for state data
            control_dtype: Data type for control data
            buffer_size: Number of state frames to store in buffer
            control_buffer_size: Number of control commands to store in buffer
            enable_gui: Enable PyBullet GUI for visualization
            source_selection: Logical endpoint supplying the active control frame
            left_arm_position: Position of left arm base [x, y, z]
            right_arm_position: Position of right arm base [x, y, z]
            **kwargs: Additional arguments passed to BaseRobot
        """
        # Initialize base robot
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback,
                         control=control)

        # Robot-specific attributes
        self.device_name = "DualArmSimRobot"
        self.urdf_path = urdf_path
        self.joint_indices = joint_indices
        self.enable_gui = enable_gui
        self.left_arm_position = left_arm_position
        self.right_arm_position = right_arm_position

        # End-effector control attributes
        self.ctrl_type = self.control_config.type
        assert self.ctrl_type in ["joints", "eef_rel", "eef_abs"]

        # IK method selection (only PyBullet supported now)
        self.ik_method = "pybullet"

        # RTB robot models removed - using PyBullet IK only

        # Recalculate SHM names with correct device_name
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"
        self.default_control_shm_name = f"{self.device_name}_{self.device_class_id}_control"

        # PyBullet simulation
        self.physics_client = None
        self.left_robot_id = None
        self.right_robot_id = None
        self.left_num_joints = 0
        self.right_num_joints = 0
        self.left_joint_names = []
        self.right_joint_names = []
        self.left_joint_limits = []
        self.right_joint_limits = []
        self.left_joint_indices = []
        self.right_joint_indices = []

        # Initialize PyBullet
        self._initialize_pybullet()
        # time.sleep(3)

    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation for dual arms."""
        try:
            import pybullet as p
            import pybullet_data

            # Connect to PyBullet
            if self.enable_gui:
                self.physics_client = p.connect(p.GUI)  # Use GUI for visualization
                logger.info("PyBullet GUI enabled for dual-arm visualization")
            else:
                self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless simulation
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Set physics parameters
            p.setGravity(0, 0, 0)
            p.setTimeStep(1.0 / self.fps)

            # Load robot URDF for both arms
            if self.urdf_path is None:
                # Use default Flexiv Rizon4 robot
                self.urdf_path = "./assests/robot/rizon/flexiv_Rizon4_kinematics.urdf"

            # Load left arm
            self.left_orientation = p.getQuaternionFromEuler([np.pi/2, -np.pi/2, 0])
            self.left_robot_id = p.loadURDF(self.urdf_path, self.left_arm_position, self.left_orientation)
            logger.info(f"Loaded left arm URDF: {self.urdf_path} at {self.left_arm_position}")

            # Load right arm
            self.right_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
            self.right_robot_id = p.loadURDF(self.urdf_path, self.right_arm_position, self.right_orientation)
            logger.info(f"Loaded right arm URDF: {self.urdf_path} at {self.right_arm_position}")

            # Get joint information for left arm
            self.left_num_joints = p.getNumJoints(self.left_robot_id)
            logger.info(f"Left arm has {self.left_num_joints} joints")

            # Get joint information for right arm
            self.right_num_joints = p.getNumJoints(self.right_robot_id)
            logger.info(f"Right arm has {self.right_num_joints} joints")

            # Get joint names and limits for left arm
            for i in range(self.left_num_joints):
                joint_info = p.getJointInfo(self.left_robot_id, i)
                self.left_joint_names.append(joint_info[1].decode('utf-8'))

                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.left_joint_limits.append((lower_limit, upper_limit))

            # Get joint names and limits for right arm
            for i in range(self.right_num_joints):
                joint_info = p.getJointInfo(self.right_robot_id, i)
                self.right_joint_names.append(joint_info[1].decode('utf-8'))

                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.right_joint_limits.append((lower_limit, upper_limit))

            # Set joint indices to control (default: all revolute joints)
            if self.joint_indices is None:
                # Left arm joint indices
                self.left_joint_indices = []
                for i in range(self.left_num_joints):
                    joint_info = p.getJointInfo(self.left_robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.left_joint_indices.append(i)

                # Right arm joint indices
                self.right_joint_indices = []
                for i in range(self.right_num_joints):
                    joint_info = p.getJointInfo(self.right_robot_id, i)
                    if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                        self.right_joint_indices.append(i)
            else:
                # Use provided joint indices (assuming they're split between arms)
                mid_point = len(self.joint_indices) // 2
                self.left_joint_indices = self.joint_indices[:mid_point]
                self.right_joint_indices = self.joint_indices[mid_point:]

            logger.info(f"Controlling left arm joints: {self.left_joint_indices}")
            logger.info(f"Left arm joint names: {[self.left_joint_names[i] for i in self.left_joint_indices]}")
            logger.info(f"Controlling right arm joints: {self.right_joint_indices}")
            logger.info(f"Right arm joint names: {[self.right_joint_names[i] for i in self.right_joint_indices]}")

            # Set initial joint positions to zero for both arms
            left_initial_positions = [1] * len(self.left_joint_indices)
            right_initial_positions = [1] * len(self.right_joint_indices)

            for i, joint_idx in enumerate(self.left_joint_indices):
                p.resetJointState(self.left_robot_id, joint_idx, left_initial_positions[i], 0.0)

            for i, joint_idx in enumerate(self.right_joint_indices):
                p.resetJointState(self.right_robot_id, joint_idx,  right_initial_positions[i], 0.0)

            start_time = time.time()

            while time.time() < start_time+1:
                left_link_state = p.getLinkState(self.left_robot_id, 6, computeForwardKinematics=True)
                self.init_pose_left = np.concatenate(
                    [np.array(left_link_state[4]), np.array(left_link_state[5])])  # in world frame

                right_link_state = p.getLinkState(self.right_robot_id, 6, computeForwardKinematics=True)
                self.init_pose_right = np.concatenate(
                    [np.array(right_link_state[4]), np.array(right_link_state[5])])  # in world frame

            left_link_state = p.getLinkState(self.left_robot_id, 6, computeForwardKinematics=True)
            self.init_pose_left = np.concatenate([np.array(left_link_state[4]), np.array(left_link_state[5])]) # in world frame

            right_link_state = p.getLinkState(self.right_robot_id, 6, computeForwardKinematics=True)
            self.init_pose_right = np.concatenate([np.array(right_link_state[4]), np.array(right_link_state[5])]) # in world frame

            logger.info("Dual-arm PyBullet simulation initialized successfully")


        except Exception as e:
            logger.error(f"Error initializing dual-arm PyBullet: {e}")
            raise

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state (joint positions and velocities for both arms).

        Returns:
            numpy.ndarray: Current robot state [left_joint_positions, left_joint_velocities,
                          right_joint_positions, right_joint_velocities]
        """
        try:
            import pybullet as p

            if self.left_robot_id is None or self.right_robot_id is None:
                return None

            # Get joint states for left arm
            left_joint_states = p.getJointStates(self.left_robot_id, self.left_joint_indices)

            # Get joint states for right arm
            right_joint_states = p.getJointStates(self.right_robot_id, self.right_joint_indices)

            # Extract positions and velocities for left arm
            left_positions = [state[0] for state in left_joint_states]
            left_velocities = [state[1] for state in left_joint_states]

            # Extract positions and velocities for right arm
            right_positions = [state[0] for state in right_joint_states]
            right_velocities = [state[1] for state in right_joint_states]

            # Combine into state array: [left_pos, left_vel, right_pos, right_vel]
            numpy_data_dtype = get_dtype(self.data_dtype)
            state_array = np.array(left_positions + left_velocities + right_positions + right_velocities,
                                 dtype=numpy_data_dtype)

            # Reshape if needed
            if len(self.data_shape) > 0:
                state_array = state_array.reshape(self.data_shape)

            return state_array

        except Exception as e:
            logger.error(f"Error getting dual-arm robot state: {e}")
            return None

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command (set target joint positions for both arms).

        Args:
            control_array: Target joint positions for both arms
        """
        try:
            import pybullet as p

            if self.left_robot_id is None or self.right_robot_id is None:
                return

            selected_name = self._selected_source_name()
            control_array = (
                control_dict.get(selected_name)
                if selected_name is not None
                else None
            )
            if control_array is None:
                p.stepSimulation()
                return

            # Ensure control array has correct shape
            if len(control_array.shape) > 0:
                control_data = control_array.flatten()
            else:
                control_data = [control_array]

            # If end-effector control is enabled, use inverse kinematics
            if self.ctrl_type == "eef_abs":
                target_positions = self._compute_ik_iterative(control_data, abs=True)
            elif self.ctrl_type == "eef_rel":
                target_positions = self._compute_ik_iterative(control_data, abs=False)
            else:
                target_positions = control_data

            # Split into left and right arm targets
            left_joint_count = len(self.left_joint_indices)
            left_target_positions = target_positions[:left_joint_count]
            right_target_positions = target_positions[left_joint_count:]

            # Reset joint states for left arm
            for joint_idx, target_pos in zip(self.left_joint_indices, left_target_positions):
                p.resetJointState(self.left_robot_id, joint_idx, target_pos, 0.0)

            # Reset joint states for right arm
            for joint_idx, target_pos in zip(self.right_joint_indices, right_target_positions):
                p.resetJointState(self.right_robot_id, joint_idx, target_pos, 0.0)

            # Step simulation
            p.stepSimulation()

        except Exception as e:
            logger.error(f"Error executing dual-arm control command: {e}")

    def _compute_ik_iterative(self, control_data,abs):
        """
        Compute inverse kinematics iteratively using PyBullet's calculateInverseKinematics.

        Args:
            control_data: End-effector poses for both arms [left_pos, left_orn, right_pos, right_orn]
                         where pos is [x, y, z] and orn is [x, y, z, w] (quaternion)

        Returns:
            target_positions: Joint positions for both arms
        """
        try:
            import pybullet as p
            # Assume control_data contains [left_pos(3), left_orn(4), right_pos(3), right_orn(4)]
            if len(control_data) < 14:  # 3+4+3+4 = 14 minimum
                logger.warning("Insufficient control data for end-effector control")
                return control_data

            left_target_pos = control_data[0:3]
            left_target_orn = control_data[3:7]
            left_target_pos, left_target_orn = compute_deviation(self.left_arm_position, self.left_orientation,
                                                                 left_target_pos, left_target_orn)  if abs else compute_deviation(self.init_pose_left[0:3],self.init_pose_left[3:7],left_target_pos, left_target_orn)

            right_target_pos = control_data[7:10]
            right_target_orn = control_data[10:14]
            right_target_pos, right_target_orn = compute_deviation(self.right_arm_position, self.right_orientation,
                                                                 right_target_pos, right_target_orn) if abs else compute_deviation(self.init_pose_right[0:3],self.init_pose_right[3:7],right_target_pos, right_target_orn)
            left_end_effector_link = 6
            right_end_effector_link = 6

            # Iterative IK parameters
            max_iterations = 20
            position_tolerance = 1e-3
            orientation_tolerance = 1e-2

            # Get current joint positions as initial guess
            left_current_joint_states = p.getJointStates(self.left_robot_id, self.left_joint_indices)
            left_q0 = [state[0] for state in left_current_joint_states]
            right_current_joint_states = p.getJointStates(self.right_robot_id, self.right_joint_indices)
            right_q0 = [state[0] for state in right_current_joint_states]

            # Compute IK for left arm with iteration
            for iteration in range(max_iterations):
                left_joint_positions = p.calculateInverseKinematics(
                    self.left_robot_id,
                    left_end_effector_link,
                    left_target_pos,
                    left_target_orn,
                    lowerLimits=[self.left_joint_limits[i][0] for i in self.left_joint_indices],
                    upperLimits=[self.left_joint_limits[i][1] for i in self.left_joint_indices],
                    jointRanges=[self.left_joint_limits[i][1] - self.left_joint_limits[i][0] for i in self.left_joint_indices],
                    restPoses=left_q0,  # Use current state as rest pose
                    maxNumIterations=10,  # Increase internal iterations per step
                    residualThreshold=1e-5  # Relaxed residual threshold
                )

                # Convert to list and apply joint limits
                left_joint_positions = list(left_joint_positions[:len(self.left_joint_indices)])
                for i, joint_idx in enumerate(self.left_joint_indices):
                    if joint_idx < len(self.left_joint_limits):
                        lower_limit, upper_limit = self.left_joint_limits[joint_idx]
                        left_joint_positions[i] = np.clip(left_joint_positions[i], lower_limit, upper_limit)

                # Check error
                left_link_state = p.getLinkState(self.left_robot_id, left_end_effector_link, computeForwardKinematics=True)
                left_current_pos = left_link_state[4]
                left_current_orn = left_link_state[5]

                left_pos_error = np.linalg.norm(np.array(left_current_pos) - np.array(left_target_pos))
                left_dot_product = np.clip(np.dot(left_current_orn, left_target_orn), -1.0, 1.0)
                left_orn_error = 2 * np.sqrt(max(0.0, 1 - left_dot_product ** 2))

                # Update initial guess
                left_q0 = left_joint_positions[:len(self.left_joint_indices)]

                # Check convergence
                if left_pos_error < position_tolerance:
                    break

            # Compute IK for right arm with iteration
            for iteration in range(max_iterations):
                right_joint_positions = p.calculateInverseKinematics(
                    self.right_robot_id,
                    right_end_effector_link,
                    right_target_pos,
                    right_target_orn,
                    lowerLimits=[self.right_joint_limits[i][0] for i in self.right_joint_indices],
                    upperLimits=[self.right_joint_limits[i][1] for i in self.right_joint_indices],
                    jointRanges=[self.right_joint_limits[i][1] - self.right_joint_limits[i][0] for i in self.right_joint_indices],
                    restPoses=right_q0,
                    maxNumIterations=10,
                    residualThreshold=1e-5
                )

                # Convert to list and apply joint limits
                right_joint_positions = list(right_joint_positions[:len(self.right_joint_indices)])
                for i, joint_idx in enumerate(self.right_joint_indices):
                    if joint_idx < len(self.right_joint_limits):
                        lower_limit, upper_limit = self.right_joint_limits[joint_idx]
                        right_joint_positions[i] = np.clip(right_joint_positions[i], lower_limit, upper_limit)

                # Check error
                right_link_state = p.getLinkState(self.right_robot_id, right_end_effector_link, computeForwardKinematics=True)
                right_current_pos = right_link_state[4]
                right_current_orn = right_link_state[5]
                right_pos_error = np.linalg.norm(np.array(right_current_pos) - np.array(right_target_pos))
                right_dot_product = np.clip(np.dot(right_current_orn, right_target_orn), -1.0, 1.0)
                right_orn_error = 2 * np.sqrt(max(0.0, 1 - right_dot_product ** 2))

                # Update initial guess
                right_q0 = right_joint_positions[:len(self.right_joint_indices)]

                # Check convergence
                if right_pos_error < position_tolerance:
                    logger.debug(f"Right arm IK converged at iteration {iteration+1}, pos_error: {right_pos_error:.6f}")
                    break

            # Print final orientation errors for both arms
            if left_joint_positions is not None and right_joint_positions is not None:
                # Calculate final errors for left arm
                final_left_link_state = p.getLinkState(self.left_robot_id, left_end_effector_link, computeForwardKinematics=True)
                final_left_current_orn = final_left_link_state[5]
                final_left_dot_product = np.clip(np.dot(final_left_current_orn, left_target_orn), -1.0, 1.0)
                final_left_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_left_dot_product ** 2))
                final_left_orn_error_deg = np.degrees(final_left_orn_error_rad)

                # Calculate final errors for right arm
                final_right_link_state = p.getLinkState(self.right_robot_id, right_end_effector_link, computeForwardKinematics=True)
                final_right_current_orn = final_right_link_state[5]
                final_right_dot_product = np.clip(np.dot(final_right_current_orn, right_target_orn), -1.0, 1.0)
                final_right_orn_error_rad = 2 * np.sqrt(max(0.0, 1 - final_right_dot_product ** 2))
                final_right_orn_error_deg = np.degrees(final_right_orn_error_rad)
                # Take only the controlled joints
                left_controlled = left_joint_positions[:len(self.left_joint_indices)]
                right_controlled = right_joint_positions[:len(self.right_joint_indices)]
                return list(left_controlled) + list(right_controlled)
            else:
                logger.warning("IK computation failed, using original control data")
                return control_data

        except Exception as e:
            logger.error(f"Error in iterative IK computation: {e}")
            return control_data



    def stop_server(self) -> None:
        """Stop the dual-arm robot server and clean up resources."""
        self.running = False

        # Clean up PyBullet
        if self.physics_client is not None:
            try:
                import pybullet as p
                p.disconnect()
                logger.info("Disconnected from PyBullet")
            except:
                pass
            self.physics_client = None

        # Clean up shared memory
        self._cleanup_shared_memory()
        self._cleanup_control_shared_memory()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.stop_server()


def sim_main() -> None:
    """Main function for testing SimRobot."""
    import argparse

    parser = argparse.ArgumentParser(description='Simulated Robot Device')
    parser.add_argument('--device_class_id', type=int, default=0, help='Device ID')
    parser.add_argument('--urdf_path', type=str, default=None, help='Path to robot URDF file')
    parser.add_argument('--fps', type=float, default=100.0, help='Control frequency (Hz)')
    parser.add_argument('--buffer_size', type=int, default=100, help='State buffer size')
    parser.add_argument('--control_buffer_size', type=int, default=50, help='Control buffer size')
    parser.add_argument('--enable_gui', action='store_true', help='Enable PyBullet GUI for visualization')
    parser.add_argument('--control_source_device', type=str, default=None,
                       help='Owning device class for an external control source')
    parser.add_argument('--control_source_device_class_id', type=int, default=0,
                       help='Device id for the external control source')
    parser.add_argument('--control_source_port', choices=('state', 'command'),
                       default='command', help='Logical port for the control source')

    args = parser.parse_args()

    # Create and start simulated robot
    robot = SimRobot(
        device_class_id=args.device_class_id,
        state_feedback={
            "fps": args.fps,
            "buffer_size": args.buffer_size,
            "hardware_latency_ms": 0.0,
            "format": {
                "shape": [14],
                "dtype": "float64",
                "type": ["j1-7", "dq1-7"],
            },
        },
        control={
            "type": "joints",
            **(
                {
                    "source_selection": {
                        "mode": "single_source",
                        "source": {
                            "device": args.control_source_device,
                            "device_class_id": args.control_source_device_class_id,
                            "port": args.control_source_port,
                        },
                    }
                }
                if args.control_source_device
                else {}
            ),
            "buffer_size": args.control_buffer_size,
            "format": {
                "shape": [7],
                "dtype": "float64",
                "type": ["j1-7"],
            },
        },
        urdf_path=args.urdf_path,
        enable_gui=args.enable_gui,
    )

    try:
        robot.start_server()
    except KeyboardInterrupt:
        logger.info("Stopping robot...")
    finally:
        robot.stop_server()


class RizonRobot(BaseRobot):
    """
    Rizon Robot Device - Flexiv Rizon robot interface for SuperInference.

    Inherits from BaseRobot and provides real-time robot state monitoring and joint position control.
    """

    def __init__(self, device_class_id=0, robot_sn="Rizon4s-123456", *,
                 state_feedback: StateFeedbackConfig | Mapping[str, Any],
                 control: RobotControlConfig | Mapping[str, Any], home=None, max_velocity=None,
                 tool_name=None, max_acceleration=None, utilize_gripper=False,
                 gripper_backend=None, gripper_name="", gripper=None, inference_gripper_calibration_path=None,
                 gripper_type=DEFAULT_LK_GRIPPER_TYPE,
                 gripper_port="/dev/ttyUSB0", gripper_motor_id=1, gripper_baudrate=115200,
                 gripper_speed_dps=180.0, gripper_home_width=None,
                 gripper_home_ratio=None,
                 gripper_timeout=0.1, gripper_wait_timeout=15.0,
                 gripper_auto_calibrate=False,
                 gripper_calibration_files=None,
                 gripper_min_torque_iq=None,
                 gripper_max_torque_iq=None,
                 gripper_calibration_timeout=15.0, gripper_calibration_settle_time=0.24,
                 gripper_min_encoder=13363, gripper_max_encoder=32819,
                 gripper_min_angle_deg=None, gripper_max_angle_deg=None,
                 gripper_arm_length_m=None,
                 gripper_apex_angle_max_deg=None,
                 gripper_gear_ratio=1.0,
                 gripper_min_vel=0.0, gripper_max_vel=0.1,
                 gripper_min_width=0.0, gripper_max_width=1.0,
                 gripper_force_limit=100.0,
                 gripper_value_mode=None,
                 gripper_input_ratio=False,
                 gripper_scale_factor=1.0,
                 gripper_arm_length=0.0,
                 gripper_apex_angle=90.0,
                 gripper_calibration_dir=None,
                 gripper_command_offset=0.0,
                 delta_eef_motion_scaler=None, impedance_parameter=None, using_tcp_pose=False,
                 max_linear_vel=None, max_linear_acc=None, max_angular_vel=None, max_angular_acc=None,
                 signal_files: Optional[Dict[str, str]] = None,
                 safety_bounds: Optional[Dict[str, Any]] = None,
                 max_contact_wrench: Optional[List[float]] = None,
                 traj_run_recording: Optional[Dict[str, Any]] = None,
                 cartesian_planner: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize Rizon Robot Device.

        Args:
            device_class_id: Unique device identifier
            robot_sn: Robot serial number (e.g., "Rizon4s-123456") or list for dual arm
            data_shape: Shape of robot state data (22 for single arm, 44 for dual arm)
            control_shape: Shape of control command data (7 for single arm, 14 for dual arm, +1 per arm if gripper enabled)
            fps: Update frequency in Hz
            data_dtype: Data type for robot states
            control_dtype: Data type for control commands
            buffer_size: Size of state data buffer
            control_buffer_size: Size of control command buffer
            hardware_latency_ms: Hardware latency compensation
            source_selection: Logical control source endpoint configuration
            control_type: Control type - "joints", "eef_abs", or "eef_rel"
            home: Home configuration dict with 'joints' and/or 'pose' keys
            max_velocity: Maximum joint velocities [rad/s]
            max_acceleration: Maximum joint accelerations [rad/s²]
            utilize_gripper: Enable gripper control, or a 2-item list/tuple of per-arm flags for dual arm
                robots (requires control_shape: [8] single, [16] dual when any arm uses gripper)
            gripper_backend: Optional gripper backend. Defaults to "flexiv" for legacy flat gripper
                config and "lktech" for nested gripper config.
            gripper_name: Optional display name for the LKTech gripper backend
            gripper: Optional nested gripper config dict. When provided, values here override the
                legacy flat gripper_* constructor arguments.
            gripper_port: LK motor serial port, or a 2-item list/tuple for dual arm
            gripper_motor_id: LK motor ID, or a 2-item list/tuple for dual arm
            gripper_baudrate: LK motor baudrate, or a 2-item list/tuple for dual arm
            gripper_speed_dps: Default LK motor speed limit in deg/s, or a 2-item list/tuple
            gripper_home_width: Desired home gripper width in meters, or a 2-item list/tuple
            gripper_home_ratio: Optional home opening ratio in [0, 1]. Only used by
                normalized gripper mode; width_m mode should set home_width directly.
            gripper_value_mode: External gripper value semantics, "normalized" or "width_m".
                In "normalized" mode, repo-facing values follow 0=fully closed, 1=fully open.
            gripper_scale_factor: Per-command scaling factor in [0, 1] applied only to real-time
                policy gripper actions before sending them to the SDK.
            gripper_command_offset: Temporary per-command offset subtracted from real-time
                policy gripper actions before sending them to the SDK.
            delta_eef_motion_scaler: Dict with 'trans' and 'rot' scaling factors for eef_delta mode
            impedance_parameter: Cartesian impedance parameters [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z] for EEF control mode
            max_linear_vel: Maximum linear velocity for EEF control [m/s]
            max_linear_acc: Maximum linear acceleration for EEF control [m/s²]
            max_angular_vel: Maximum angular velocity for EEF control [rad/s]
            max_angular_acc: Maximum angular acceleration for EEF control [rad/s²]
            safety_bounds: Optional dict describing an axis-aligned Cartesian
                bounding box in the robot base frame. Keys: enabled (bool),
                frame (str, only "robot_base" supported), min_xyz (list of 3),
                max_xyz (list of 3). Applied during EEF inference.
            max_contact_wrench: Optional 6-vector [Fx, Fy, Fz, Mx, My, Mz] in
                N / N*m. When the measured external contact wrench at the TCP
                exceeds this limit during EEF motion, the robot enters
                IN_REDUCED_STATE (auto slow/halt) and recovers automatically
                once contact is released. Only applied for EEF control modes.
            command_fps: Command execution frequency in Hz (only used in legacy async mode).
            legacy_async_mode: Master switch for the legacy behavior. When True, control
                commands run in a dedicated command thread at ``command_fps`` and the
                Cartesian motion planner / trajectory recording features below are enabled
                (subject to their own config). When False (default), commands execute
                synchronously in the single-threaded main loop and the planner / recorder
                are fully disabled regardless of their config.
            traj_run_recording: Optional dict for trajectory JSONL streams and optional
                auto-merge on shutdown. Only honored when ``legacy_async_mode`` is True.
            cartesian_planner: Dual-arm Cartesian planner settings (``utilized: true`` to run
                the C++ planner). Only honored when ``legacy_async_mode`` is True.
            **kwargs: Additional arguments passed to BaseRobot
        """
        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback,
                         control=control)
        # Extract signal file patterns from config if provided
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._signal_files = signal_files
        self._robot_home_command_pattern = signal_files.get('robot_home_command_pattern')
        self._policy_runner_reset_command = signal_files.get('policy_runner_reset_command')
        self._reset_signal = signal_files.get('reset_signal')

        # Robot configuration
        self.robot_sn = robot_sn
        self.control_type = self.control_config.type
        assert self.control_type in ["joints", "eef_abs", "eef_rel"], f"Invalid control.type: {self.control_type}"
        self.default_control_shm_name = f"{self.device_name}_{self.device_class_id}_control"
        self.is_dual_arm = isinstance(robot_sn, list) and self.control_shape[0] >= 14
        # Policy control timestamp - used to filter stale actions after control mode switch
        self._policy_control_start_timestamp = 0
        # Grace period (in nanoseconds) to wait for fresh policy actions after control switch
        # This prevents robot from executing old actions while waiting for new policy actions
        self._policy_control_grace_period_ns = int(0.5 * 1e9)  # 0.5 seconds

        gripper_config = dict(gripper) if isinstance(gripper, dict) else {}
        resolved_gripper_backend = gripper_backend
        if gripper_config:
            resolved_gripper_backend = gripper_config.get("backend", resolved_gripper_backend or "lktech")
            utilize_gripper = gripper_config.get("enabled", utilize_gripper)
            gripper_name = gripper_config.get("name", gripper_name)
            gripper_type = gripper_config.get("gripper_type", gripper_config.get("type", gripper_type))
            tool_name = gripper_config.get("tool_name", tool_name)
            gripper_port = gripper_config.get("port", gripper_port)
            gripper_motor_id = gripper_config.get("motor_id", gripper_motor_id)
            gripper_baudrate = gripper_config.get("baudrate", gripper_baudrate)
            gripper_speed_dps = gripper_config.get("speed_dps", gripper_speed_dps)
            gripper_home_width = gripper_config.get("home_width", gripper_home_width)
            gripper_home_ratio = gripper_config.get("home_ratio", gripper_home_ratio)
            gripper_timeout = gripper_config.get("timeout", gripper_timeout)
            gripper_wait_timeout = gripper_config.get("wait_timeout", gripper_wait_timeout)
            gripper_auto_calibrate = gripper_config.get("auto_calibrate", gripper_auto_calibrate)
            gripper_calibration_files = gripper_config.get("calibration_files", gripper_calibration_files)
            gripper_min_torque_iq = gripper_config.get("min_torque_iq", gripper_min_torque_iq)
            gripper_max_torque_iq = gripper_config.get("max_torque_iq", gripper_max_torque_iq)
            gripper_calibration_timeout = gripper_config.get("calibration_timeout", gripper_calibration_timeout)
            gripper_calibration_settle_time = gripper_config.get("calibration_settle_time", gripper_calibration_settle_time)
            gripper_min_encoder = gripper_config.get("min_encoder", gripper_min_encoder)
            gripper_max_encoder = gripper_config.get("max_encoder", gripper_max_encoder)
            gripper_min_angle_deg = gripper_config.get("min_angle_deg", gripper_min_angle_deg)
            gripper_max_angle_deg = gripper_config.get("max_angle_deg", gripper_max_angle_deg)
            gripper_arm_length_m = gripper_config.get("arm_length_m", gripper_arm_length_m)
            gripper_apex_angle_max_deg = gripper_config.get("apex_angle_max_deg", gripper_apex_angle_max_deg)
            gripper_gear_ratio = gripper_config.get("gear_ratio", gripper_gear_ratio)
            gripper_min_vel = gripper_config.get("min_vel", gripper_min_vel)
            gripper_max_vel = gripper_config.get("max_vel", gripper_max_vel)
            gripper_min_width = gripper_config.get("min_width", gripper_min_width)
            gripper_max_width = gripper_config.get("max_width", gripper_max_width)
            gripper_force_limit = gripper_config.get("force_limit", gripper_force_limit)
            gripper_value_mode = gripper_config.get("value_mode", gripper_value_mode)
            gripper_input_ratio = gripper_config.get("input_ratio", gripper_input_ratio)
            gripper_scale_factor = gripper_config.get("scale_factor", gripper_scale_factor)
            gripper_arm_length = gripper_config.get("arm_length", gripper_arm_length)
            gripper_apex_angle = gripper_config.get("apex_angle", gripper_apex_angle)
            gripper_calibration_dir = gripper_config.get("calibration_dir", gripper_calibration_dir)
            gripper_command_offset = gripper_config.get("command_offset", gripper_command_offset)
        if resolved_gripper_backend is None:
            resolved_gripper_backend = "lktech" if gripper_config else "flexiv"
        resolved_gripper_backend = str(resolved_gripper_backend).strip().lower()
        if resolved_gripper_backend not in {"flexiv", "lktech"}:
            raise ValueError(f"Unsupported gripper backend: {resolved_gripper_backend}")
        _LK_WIDTH_FALLBACK = 0.15
        if resolved_gripper_backend == "lktech":
            lk_default_width = calculate_lk_gripper_max_width(
                gripper_arm_length, gripper_apex_angle, _LK_WIDTH_FALLBACK
            )
        else:
            lk_default_width = _LK_WIDTH_FALLBACK
        if gripper_home_width is None:
            gripper_home_width = lk_default_width if resolved_gripper_backend == "lktech" else 0.085
        if gripper_max_width is None:
            gripper_max_width = lk_default_width if resolved_gripper_backend == "lktech" else 0.085
        if gripper_value_mode is None:
            gripper_value_mode = "width_m"
        elif str(gripper_value_mode).strip().lower() == "normalized":
            logger.warning(
                "gripper_value_mode='normalized' is deprecated; migrate the "
                "control stream and configuration to 'width_m'"
            )
        gripper_type = str(gripper_type or DEFAULT_LK_GRIPPER_TYPE).strip().lower()
        if resolved_gripper_backend == "lktech":
            gripper_preset = get_lk_gripper_preset(gripper_type)
            if gripper_min_torque_iq is None:
                gripper_min_torque_iq = gripper_preset["min_torque_iq"]
            if gripper_max_torque_iq is None:
                gripper_max_torque_iq = gripper_preset["max_torque_iq"]
            if gripper_arm_length_m is None:
                gripper_arm_length_m = gripper_preset["arm_length_m"]
            if gripper_apex_angle_max_deg is None:
                gripper_apex_angle_max_deg = gripper_preset["apex_angle_max_deg"]

        # Gripper configuration
        self.utilize_gripper_config = utilize_gripper
        self.utilize_gripper = False
        self.gripper_backend = resolved_gripper_backend
        self.gripper_name = gripper_name
        self.gripper_type = gripper_type
        self.tool_name = tool_name if tool_name is not None else self.gripper_name
        self.gripper_port_config = gripper_port
        self.gripper_motor_id_config = gripper_motor_id
        self.gripper_baudrate_config = gripper_baudrate
        self.gripper_speed_config = gripper_speed_dps
        self.gripper_home_width_config = gripper_home_width
        self.gripper_home_ratio_config = gripper_home_ratio
        self.gripper_timeout_config = gripper_timeout
        self.gripper_wait_timeout_config = gripper_wait_timeout
        self.gripper_auto_calibrate_config = gripper_auto_calibrate
        self.gripper_calibration_files_config = gripper_calibration_files
        self.gripper_min_torque_iq_config = gripper_min_torque_iq
        self.gripper_max_torque_iq_config = gripper_max_torque_iq
        self.gripper_calibration_timeout_config = gripper_calibration_timeout
        self.gripper_calibration_settle_time_config = gripper_calibration_settle_time
        self.gripper_min_encoder_config = gripper_min_encoder
        self.gripper_max_encoder_config = gripper_max_encoder
        self.gripper_min_angle_deg_config = gripper_min_angle_deg
        self.gripper_max_angle_deg_config = gripper_max_angle_deg
        self.gripper_arm_length_config = gripper_arm_length_m
        self.gripper_apex_angle_max_config = gripper_apex_angle_max_deg
        self.gripper_gear_ratio_config = gripper_gear_ratio
        self.gripper_min_vel_config = gripper_min_vel
        self.gripper_max_vel_config = gripper_max_vel
        self.gripper_min_width_config = gripper_min_width
        self.gripper_max_width_config = gripper_max_width
        self.gripper_force_limit_config = gripper_force_limit
        self.gripper_value_mode = str(gripper_value_mode).strip().lower()
        # When True, the REAL-TIME gripper command stream (e.g. a Quest trigger) is a
        # 0-1 width ratio that must be converted to physical meters (n * max_width)
        # before the width_m pipeline. Reset/home/policy paths are unaffected.
        self.gripper_input_ratio = bool(gripper_input_ratio)
        self.gripper_scale_factor_config = gripper_scale_factor
        # Isosceles-triangle leg length L (meters) for the cosine-law width model.
        self.gripper_arm_length_config = gripper_arm_length if gripper_arm_length else gripper_arm_length_m
        # Fully-open gripper apex angle theta_max (degrees), set manually, for the
        # cosine-law width model: max_width = 2 * L * sin(theta_max / 2).
        self.gripper_apex_angle_config = gripper_apex_angle
        # SDK calibration dir for the bundled LKTech wrapper. When set, the
        # gripper loads/saves multi_turn_angle calibration here (R1 parity);
        # this is what enables full-target moves at span*velocity speed instead
        # of the slow encoder-mode err/100 crawl.
        self.gripper_calibration_dir = gripper_calibration_dir
        self.gripper_command_offset_config = gripper_command_offset
        self.using_tcp_pose = using_tcp_pose
        self.state_layout_size = validate_rizon_state_shape(
            self.data_shape,
            is_dual_arm=self.is_dual_arm,
            using_tcp_pose=self.using_tcp_pose,
        )

        # Gripper calibration data
        self.inference_gripper_calibration_path = inference_gripper_calibration_path
        self.gripper_calibration_data = None
        if self.inference_gripper_calibration_path is not None:
            self._load_gripper_calibration_data()
        # Delta EEF motion scaler configuration
        self.delta_eef_motion_scaler = delta_eef_motion_scaler
        if self.delta_eef_motion_scaler is not None:
            logger.info(f"Delta EEF motion scaler configured: trans={self.delta_eef_motion_scaler.get('trans', 1.0)}, rot={self.delta_eef_motion_scaler.get('rot', 1.0)}")

        # Cartesian impedance configuration
        self.impedance_parameter = impedance_parameter
        if self.control_type in ["eef_abs", "eef_rel"]:
            if self.impedance_parameter is not None:
                logger.info(f"Cartesian impedance configured: {self.impedance_parameter}")
            else:
                # Default values: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
                self.impedance_parameter = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]
                logger.info(f"Using default cartesian : {self.impedance_parameter}")
        else:
            # Joint control mode - impedance parameters not used
            self.impedance_parameter = None
            logger.info("Joint control mode - Impedance parameters not applicable")

        # Motion limits configuration for EEF control
        if self.control_type in ["eef_abs", "eef_rel"]:
            # Linear motion limits
            self.max_linear_vel = max_linear_vel if max_linear_vel is not None else 1.0
            self.max_linear_acc = max_linear_acc if max_linear_acc is not None else 2.0

            # Angular motion limits
            self.max_angular_vel = max_angular_vel if max_angular_vel is not None else 2.0
            self.max_angular_acc = max_angular_acc if max_angular_acc is not None else 5.0

            logger.info(f"Motion limits configured - Linear: vel={self.max_linear_vel} m/s, acc={self.max_linear_acc} m/s²")
            logger.info(f"Motion limits configured - Angular: vel={self.max_angular_vel} rad/s, acc={self.max_angular_acc} rad/s²")
        else:
            # Joint control mode - motion limits not used
            self.max_linear_vel = None
            self.max_linear_acc = None
            self.max_angular_vel = None
            self.max_angular_acc = None
            logger.info("Joint control mode - Motion limits not applicable")

        # Safety bounding box for EEF position commands (robot base frame).
        self.safety_bounds = SafetyBounds.from_config(safety_bounds)
        # Per-arm rate limiter state for "safety clip" warnings.
        self._safety_clip_last_log_time: Dict[str, float] = {}

        # Max external contact wrench (force/torque) at TCP. Only honoured in
        # EEF control modes; ignored when control_type=="joints".
        self.max_contact_wrench: Optional[List[float]] = None
        if max_contact_wrench is not None:
            if self.control_type not in ["eef_abs", "eef_rel"]:
                logger.warning(
                    "max_contact_wrench is ignored when control_type is "
                    f"{self.control_type!r} (only EEF modes apply the limit)."
                )
            else:
                wrench_vec = [float(v) for v in max_contact_wrench]
                if len(wrench_vec) != 6:
                    raise ValueError(
                        f"max_contact_wrench must be length 6, got {len(wrench_vec)}."
                    )
                if not all(v > 0.0 for v in wrench_vec):
                    raise ValueError(
                        f"max_contact_wrench values must be positive, got {wrench_vec}."
                    )
                self.max_contact_wrench = wrench_vec
                logger.info(
                    f"Max contact wrench configured: {self.max_contact_wrench} "
                    "(N for first 3, N*m for last 3)"
                )
        # Edge-detect state for IN_REDUCED_STATE wrench-limit events.
        self._last_operational_status: Any = None

        self.gripper_active_by_arm = self._normalize_gripper_active_config(
            self.utilize_gripper_config,
            "utilize_gripper",
        )
        self.utilize_gripper = any(self.gripper_active_by_arm)

        self.gripper_ports = self._normalize_gripper_config_value(self.gripper_port_config, "gripper_port")
        self.gripper_motor_ids = self._normalize_gripper_config_value(self.gripper_motor_id_config, "gripper_motor_id")
        self.gripper_baudrates = self._normalize_gripper_config_value(self.gripper_baudrate_config, "gripper_baudrate")
        self.gripper_speeds = self._normalize_gripper_config_value(self.gripper_speed_config, "gripper_speed_dps")
        self.gripper_home_widths = self._normalize_gripper_config_value(self.gripper_home_width_config, "gripper_home_width")
        self.gripper_home_ratios = self._normalize_gripper_config_value(self.gripper_home_ratio_config, "gripper_home_ratio")
        self.gripper_timeouts = self._normalize_gripper_config_value(self.gripper_timeout_config, "gripper_timeout")
        self.gripper_wait_timeouts = self._normalize_gripper_config_value(self.gripper_wait_timeout_config, "gripper_wait_timeout")
        self.gripper_auto_calibrates = self._normalize_gripper_config_value(self.gripper_auto_calibrate_config, "gripper_auto_calibrate")
        self.gripper_calibration_files = self._normalize_gripper_config_value(self.gripper_calibration_files_config, "gripper_calibration_files")
        self.gripper_min_torque_iqs = self._normalize_gripper_config_value(self.gripper_min_torque_iq_config, "gripper_min_torque_iq")
        self.gripper_max_torque_iqs = self._normalize_gripper_config_value(self.gripper_max_torque_iq_config, "gripper_max_torque_iq")
        self.gripper_calibration_timeouts = self._normalize_gripper_config_value(self.gripper_calibration_timeout_config, "gripper_calibration_timeout")
        self.gripper_calibration_settle_times = self._normalize_gripper_config_value(self.gripper_calibration_settle_time_config, "gripper_calibration_settle_time")
        self.gripper_min_encoders = self._normalize_gripper_config_value(self.gripper_min_encoder_config, "gripper_min_encoder")
        self.gripper_max_encoders = self._normalize_gripper_config_value(self.gripper_max_encoder_config, "gripper_max_encoder")
        self.gripper_min_angle_degs = self._normalize_gripper_config_value(self.gripper_min_angle_deg_config, "gripper_min_angle_deg")
        self.gripper_max_angle_degs = self._normalize_gripper_config_value(self.gripper_max_angle_deg_config, "gripper_max_angle_deg")
        self.gripper_arm_lengths = self._normalize_gripper_config_value(self.gripper_arm_length_config, "gripper_arm_length_m")
        self.gripper_apex_angle_max_degs = self._normalize_gripper_config_value(self.gripper_apex_angle_max_config, "gripper_apex_angle_max_deg")
        self.gripper_gear_ratios = self._normalize_gripper_config_value(self.gripper_gear_ratio_config, "gripper_gear_ratio")
        self.gripper_min_velocities = self._normalize_gripper_config_value(self.gripper_min_vel_config, "gripper_min_vel")
        self.gripper_max_velocities = self._normalize_gripper_config_value(self.gripper_max_vel_config, "gripper_max_vel")
        self.gripper_min_widths = self._normalize_gripper_config_value(self.gripper_min_width_config, "gripper_min_width")
        self.gripper_max_widths = self._normalize_gripper_config_value(self.gripper_max_width_config, "gripper_max_width")
        self.gripper_force_limits = self._normalize_gripper_config_value(self.gripper_force_limit_config, "gripper_force_limit")
        self.gripper_scale_factors = self._normalize_gripper_config_value(self.gripper_scale_factor_config, "gripper_scale_factor")
        self.gripper_apex_angles = self._normalize_gripper_config_value(self.gripper_apex_angle_config, "gripper_apex_angle")
        self.gripper_command_offsets = self._normalize_gripper_config_value(self.gripper_command_offset_config, "gripper_command_offset")
        self._apply_lk_gripper_calibration_files()
        if self.gripper_value_mode not in {"normalized", "width_m"}:
            raise ValueError(
                f"Invalid gripper_value_mode: {self.gripper_value_mode}. "
                "Expected one of: normalized, width_m"
            )
        for index, ratio in enumerate(self.gripper_home_ratios):
            if ratio is None:
                continue
            if not 0.0 <= float(ratio) <= 1.0:
                raise ValueError(
                    f"gripper_home_ratio must be within [0, 1] for index {index}, got {ratio}"
                )
        for index, scale_factor in enumerate(self.gripper_scale_factors):
            scale_value = float(scale_factor)
            if not 0.0 <= scale_value <= 1.0:
                raise ValueError(
                    f"gripper_scale_factor must be within [0, 1] for index {index}, got {scale_factor}"
                )
            self.gripper_scale_factors[index] = scale_value
        for index, command_offset in enumerate(self.gripper_command_offsets):
            self.gripper_command_offsets[index] = float(command_offset)
        self.last_gripper_value = [self._get_home_gripper_value(i) for i in range(len(self.gripper_home_widths))]
        # Last COMMANDED gripper width (m) per side, so the recorder can publish a
        # distinct hand/*/cmd (commanded) signal separate from hand/*/state (the
        # measured opening width). Seeded with the home width; updated on every
        # accepted Move in _move_gripper_value.
        self._last_gripper_cmd_width_m = [
            self._get_home_gripper_width(i) for i in range(len(self.gripper_home_widths))
        ]

        # LKTech command throttle state (R1-parity). The control loop calls the
        # gripper backend every cycle; re-streaming the same multi-turn position
        # at 100 Hz makes the firmware re-plan its trapezoidal profile each time
        # so the motor never reaches cruise speed (slow/laggy gripper). R1 only
        # re-sent on a >deadband trigger change capped at ~20 Hz, which let each
        # command run to completion. Mirror that here.
        self._gripper_cmd_last_target: Dict[int, float] = {}
        self._gripper_cmd_last_time: Dict[int, float] = {}

        # Last Cartesian setpoint sent to each Flexiv (keyed by id(robot)).
        # In NRT_CARTESIAN_MOTION_FORCE the arm only stays stiff while it keeps
        # being fed a target; once the teleop stream goes idle (e.g. Quest not
        # connected) the arm relaxes into a compliant "zero-gravity" feel. We
        # re-stream this hold pose on idle ticks so the arm holds its pose even
        # without an active teleoperator (see _poll_and_execute_control_once).
        self._hold_pose_by_robot: Dict[int, List[float]] = {}

        # Validate control_shape based on gripper usage
        if self.utilize_gripper:
            if self.is_dual_arm:
                assert self.control_shape[0] == 16, f"Dual arm with gripper requires control.format.shape[0] == 16, got {self.control_shape[0]}"
            else:
                assert self.control_shape[0] == 8, f"Single arm with gripper requires control.format.shape[0] == 8, got {self.control_shape[0]}"
            if self.gripper_backend == "lktech":
                for index, active in enumerate(self.gripper_active_by_arm):
                    if not active:
                        continue
                    port = self.gripper_ports[index]
                    assert port is not None and str(port) != "", (
                        f"gripper_port must be specified for enabled gripper index {index}"
                    )
            else:
                assert self.gripper_name, "gripper_name must be specified when using flexiv gripper backend"
        else:
            # Grippers disabled. The teleoperator may still publish a control
            # array that includes gripper slots (16 for dual, 8 for single);
            # those slots are simply ignored downstream. Accept either the
            # arm-only width or the gripper-inclusive width.
            if self.is_dual_arm:
                assert self.control_shape[0] in (14, 16), (
                    f"Dual arm without gripper requires control.format.shape[0] in (14, 16), got {self.control_shape[0]}"
                )
            else:
                assert self.control_shape[0] in (7, 8), (
                    f"Single arm without gripper requires control.format.shape[0] in (7, 8), got {self.control_shape[0]}"
                )

        # Robot interfaces
        if self.is_dual_arm:
            self.robot_left = None
            self.robot_right = None
            self.left_sn = robot_sn[0] if isinstance(robot_sn, list) else robot_sn
            self.right_sn = robot_sn[1] if isinstance(robot_sn, list) and len(robot_sn) > 1 else robot_sn
            logger.info(f"Initializing dual-arm Rizon robots: {self.left_sn}, {self.right_sn}")
        else:
            self.robot = None
            logger.info(f"Initializing single-arm Rizon robot: {robot_sn}")

        # Gripper interfaces
        self.gripper = None
        self.gripper_left = None
        self.gripper_right = None
        self.tool = None
        self.tool_left = None
        self.tool_right = None

        # Set default velocity and acceleration limits
        num_joints = 14 if self.is_dual_arm else 7
        if max_velocity is None:
            self.max_velocity = [1.0] * num_joints  # Conservative velocity limits [rad/s]
        else:
            self.max_velocity = max_velocity

        if max_acceleration is None:
            self.max_acceleration = [2.0] * num_joints  # Conservative acceleration limits [rad/s²]
        else:
            self.max_acceleration = max_acceleration

        # Robot status
        self.is_operational = False
        self.is_enabled = False
        self.current_mode = None

        # Parse home configuration
        self.home_joints = None
        self.home_joints_left = None
        self.home_joints_right = None
        self.home_presets = None
        self.home_preset_index = None
        self.home_pose = None
        self.home_pose_left = None
        self.home_pose_right = None

        self.current_point_time = 0.0
        self._planner_waypoint_index = 0
        self.real_time = 0.0
        self.num_clock_ticks = 0
        self.delta_time = 1.0 / float(self.command_fps)

        # Process home configuration
        if home is not None and isinstance(home, dict):
            self._configure_home(home)

        # Cartesian motion planner and trajectory recording are legacy-async-only features.
        # Both are force-disabled unless legacy_async_mode is enabled, even if their config
        # blocks are present.
        self._cartesian_planner_cfg: Dict[str, Any] = dict(cartesian_planner or {})
        self.use_cartesian_motion_planner = (
            self.legacy_async_mode and planner_utilized(self._cartesian_planner_cfg)
        )

        # Flexiv SDK adapter (keeps direct flexivrdk dependency out of this module)
        self.flexiv_adapter = (
            FlexivAdapter(backend="pybind") if self.use_cartesian_motion_planner else FlexivAdapter()
        )

        # Optional sparse / samples JSONL + merge (``utils/rizon_traj_run_recording``).
        self._planner_trace_perf_origin: Optional[float] = None
        self._traj_rec: Optional[RizonTrajRunRecorder] = (
            RizonTrajRunRecorder.maybe_create(
                traj_run_recording,
                self._cartesian_planner_cfg,
                self.control_type,
            )
            if self.legacy_async_mode
            else None
        )
        # Throttled Cartesian planner diagnostics (queue / executed / trajectory counts).
        self._planner_diag_interval_s: float = 0.5
        self._planner_diag_last_perf: float = 0.0
        self._planner_diag_prev_queue_left: Optional[int] = None
        self._planner_diag_prev_queue_right: Optional[int] = None

    def _reset_planner_command_timeline(self) -> None:
        """Reset planner relative time axis (first waypoint must use time_s=0)."""
        self.current_point_time = 0.0
        self._planner_waypoint_index = 0
        self._planner_add_origin = None

    def _next_planner_time_s(self) -> float:
        """Next strictly spaced planner timestamp: index * (1/command_fps)."""
        time_s = float(self._planner_waypoint_index) * self.delta_time
        self._planner_waypoint_index += 1
        self.current_point_time = float(self._planner_waypoint_index) * self.delta_time
        return time_s

    def _on_motion_planner_loop_tick(self) -> None:
        """Log dual-arm Cartesian planner status periodically for backlog diagnosis."""
        if not (
            self.use_cartesian_motion_planner
            and self.is_dual_arm
            and self.robot_left is not None
            and self.robot_right is not None
        ):
            return
        now = time.perf_counter()
        if now - self._planner_diag_last_perf < self._planner_diag_interval_s:
            return
        self._planner_diag_last_perf = now
        try:
            sl = self.robot_left.position_planner_status()
            sr = self.robot_right.position_planner_status()
        except Exception as exc:
            logger.warning("Planner diagnostic: position_planner_status failed: {}", exc)
            return

        def _stat(obj: Any, key: str, default: float = float("nan")) -> float:
            try:
                return float(obj[key])
            except Exception:
                return default

        ql = int(_stat(sl, "queue_size"))
        qr = int(_stat(sr, "queue_size"))
        el = int(_stat(sl, "executed_count"))
        er = int(_stat(sr, "executed_count"))
        cl = int(_stat(sl, "consumed_count", float(el)))
        cr = int(_stat(sr, "consumed_count", float(er)))
        tl = int(_stat(sl, "trajectory_count"))
        tr = int(_stat(sr, "trajectory_count"))
        growth_note = ""
        if self._planner_diag_prev_queue_left is not None and self._planner_diag_prev_queue_right is not None:
            if ql > self._planner_diag_prev_queue_left or qr > self._planner_diag_prev_queue_right:
                growth_note = " | queue_size grew vs last sample"
        self._planner_diag_prev_queue_left = ql
        self._planner_diag_prev_queue_right = qr
        logger.info(
            "Planner diag: L queue={} consumed={} exec={} traj={} | R queue={} consumed={} exec={} traj={}{}",
            ql,
            cl,
            el,
            tl,
            qr,
            cr,
            er,
            tr,
            growth_note,
        )

    def _maybe_write_traj_run_record_json(self) -> None:
        """On shutdown, merge JSONL fragments into one ``traj_run_record`` JSON when configured."""
        if self._traj_rec is None:
            return
        self._traj_rec.merge_json_if_configured(
            is_dual_arm=self.is_dual_arm,
            tool_left=self.tool_left,
            tool_right=self.tool_right,
            tool_single=self.tool,
        )

    def _close_traj_jsonl_sinks(self) -> None:
        if self._traj_rec is not None:
            self._traj_rec.close_sinks()

    def _maybe_log_sparse_joint_vector(self, joints: List[float]) -> None:
        if self._traj_rec is None or self._traj_rec.sparse_sink is None:
            return
        self._traj_rec.append_sparse_joint(joints)

    def _maybe_log_sparse_tcp_flexiv(
        self,
        flexiv_pose: Union[List[float], np.ndarray],
        *,
        t_perf: float,
        arm: Optional[str] = None,
    ) -> None:
        if self._traj_rec is None or self._traj_rec.sparse_sink is None:
            return
        self._traj_rec.append_sparse_tcp(flexiv_pose, t_perf=t_perf, arm=arm)

    def _write_dense_planner_trace_at_shutdown(self) -> None:
        if self._traj_rec is None:
            return
        origin = self._planner_trace_perf_origin
        self._traj_rec.write_dense_dual_planner_trace(
            use_cartesian_motion_planner=self.use_cartesian_motion_planner,
            is_dual_arm=self.is_dual_arm,
            control_type=self.control_type,
            robot_left=self.robot_left,
            robot_right=self.robot_right,
            planner_perf_origin=float(origin) if origin is not None else None,
        )

    def _poll_traj_sample_rows(self) -> List[Dict[str, Any]]:
        """One samples tick for trajectory JSONL (dual or single arm)."""
        rows: List[Dict[str, Any]] = []
        t0 = float(time.perf_counter())
        if self.is_dual_arm and self.robot_left and self.robot_right:
            sl = self.robot_left.states()
            sr = self.robot_right.states()
            rows.append(
                {
                    "record_stream": "recorded_samples",
                    "t_perf": t0,
                    "arm": "left",
                    "joints": [float(x) for x in np.asarray(sl.q, dtype=np.float64).reshape(-1).tolist()],
                    "tcp_pose": [float(x) for x in np.asarray(sl.tcp_pose, dtype=np.float64).reshape(-1).tolist()],
                }
            )
            rows.append(
                {
                    "record_stream": "recorded_samples",
                    "t_perf": t0 + 1e-9,
                    "arm": "right",
                    "joints": [float(x) for x in np.asarray(sr.q, dtype=np.float64).reshape(-1).tolist()],
                    "tcp_pose": [float(x) for x in np.asarray(sr.tcp_pose, dtype=np.float64).reshape(-1).tolist()],
                }
            )
        elif self.robot:
            st = self.robot.states()
            rows.append(
                {
                    "record_stream": "recorded_samples",
                    "t_perf": t0,
                    "joints": [float(x) for x in np.asarray(st.q, dtype=np.float64).reshape(-1).tolist()],
                    "tcp_pose": [float(x) for x in np.asarray(st.tcp_pose, dtype=np.float64).reshape(-1).tolist()],
                }
            )
        return rows

    def _stop_traj_recording_threads(self, join_timeout_s: float = 2.0) -> None:
        if self._traj_rec is not None:
            self._traj_rec.stop_samples_thread(join_timeout_s)

    def _start_traj_recording_threads_if_configured(self) -> None:
        if self._traj_rec is None:
            return
        self._traj_rec.start_samples_thread_if_configured(self._poll_traj_sample_rows)

    def _planner_home_joint_seeds(self) -> Optional[Tuple[List[float], List[float]]]:
        """Home preset joints as 7-DoF lists per arm, or None if invalid."""
        if self.home_joints_left is None or self.home_joints_right is None:
            logger.error(
                "Dual-arm Cartesian planner requires home.joints_left / joints_right (from presets)"
            )
            return None
        left = np.asarray(self.home_joints_left, dtype=np.float64).reshape(-1).tolist()
        right = np.asarray(self.home_joints_right, dtype=np.float64).reshape(-1).tolist()
        if len(left) != 7 or len(right) != 7:
            logger.error(
                "home joints for Cartesian planner must have length 7 per arm; "
                f"got left={len(left)} right={len(right)}"
            )
            return None
        return left, right

    def _start_dual_cartesian_planners_if_configured(self) -> bool:
        """Start C++ Cartesian planners for dual-arm EEF when enabled; set dense-trace time origin."""
        if self.control_type not in ("eef_abs", "eef_rel"):
            self._planner_trace_perf_origin = None
            return True
        if not self.use_cartesian_motion_planner:
            self._planner_trace_perf_origin = None
            logger.info("Dual-arm EEF: Cartesian motion planner disabled; using send_cartesian_motion_force.")
            return True
        seeds = self._planner_home_joint_seeds()
        if seeds is None:
            return False
        try:
            cp = self._cartesian_planner_cfg
            self._planner_trace_perf_origin = float(time.perf_counter())
            for arm, seed, tool, robot in (
                ("left", seeds[0], self.tool_left, self.robot_left),
                ("right", seeds[1], self.tool_right, self.robot_right),
            ):
                robot.start_cartesian_position_planner(
                    **build_planner_start_kwargs(cp, arm=arm, seed_positions=seed, tool=tool)
                )
            smooth = traj_smooth_settings(cp)
            if smooth["enabled"]:
                control_dt = float(cp.get("control_dt", 0.001))
                logger.info(
                    "Cartesian planner: traj_smooth post-process on Ruckig segments "
                    "(upsample={}, filter_window={}, command_hz={:.0f})",
                    smooth["upsample_ratio"],
                    smooth["filter_window"],
                    1.0 / control_dt,
                )
            logger.info("Dual-arm Cartesian position planner started (dense trace origin set).")
            return True
        except Exception as exc:
            logger.error(f"Failed to start dual Cartesian planners: {exc}")
            return False

    def _stop_dual_cartesian_planners_if_configured(self) -> None:
        """Stop C++ Cartesian planner workers before robot Disable/shutdown."""
        dual_eef = self.control_type in ["eef_abs", "eef_rel"]
        if not dual_eef or not self.use_cartesian_motion_planner:
            return
        if not self.is_dual_arm or self.robot_left is None or self.robot_right is None:
            return
        for arm_name, robot in (("left", self.robot_left), ("right", self.robot_right)):
            try:
                robot.stop_position_planner()
                logger.info("{} arm Cartesian position planner stopped", arm_name)
            except Exception as exc:
                logger.warning("Failed to stop {} arm Cartesian planner cleanly: {}", arm_name, exc)

    def _normalize_gripper_config_value(self, value, field_name: str) -> List[Any]:
        """Normalize single/dual gripper config values into a list."""
        if self.is_dual_arm:
            if isinstance(value, (list, tuple)):
                if len(value) != 2:
                    raise ValueError(f"{field_name} must have 2 values for dual-arm robots, got {len(value)}")
                return list(value)
            return [value, value]

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError(f"{field_name} cannot be empty")
            return [value[0]]
        return [value]

    def _normalize_gripper_active_config(self, value, field_name: str) -> List[bool]:
        """Normalize configured gripper availability into per-arm booleans."""
        normalized: List[bool] = []
        for item in self._normalize_gripper_config_value(value, field_name):
            # Treat YAML/environment boolean strings explicitly.  Python's
            # ``bool("false")`` is true and could unexpectedly energize an arm.
            if isinstance(item, str):
                normalized.append(item.strip().casefold() in {"1", "true", "yes", "on"})
                continue
            if item is None:
                normalized.append(False)
                continue
            if isinstance(item, bool):
                normalized.append(item)
                continue
            try:
                numeric = float(item)
            except (TypeError, ValueError, OverflowError):
                normalized.append(False)
                continue
            normalized.append(math.isfinite(numeric) and numeric != 0)
        return normalized

    def _load_lk_gripper_calibration_file(self, index: int) -> Dict[str, Any]:
        """Load LK gripper calibration JSON for an arm index."""
        path_value = self.gripper_calibration_files[index]
        if path_value is None or str(path_value).strip() == "":
            return {}

        calibration_path = Path(str(path_value)).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = REPO_ROOT / calibration_path
        if not calibration_path.is_file():
            raise FileNotFoundError(f"LK gripper calibration file not found: {calibration_path}")

        with calibration_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        logger.info(f"Loaded LK gripper calibration[{index}] from {calibration_path}")
        return data

    @staticmethod
    def _calibration_value(data: Dict[str, Any], key: str, fallback: Any = None) -> Any:
        return data[key] if key in data else fallback

    def _apply_lk_gripper_calibration_files(self) -> None:
        """Apply LK calibration JSON metadata to runtime conversion parameters."""
        if self.gripper_backend != "lktech":
            return

        self._lk_gripper_calibrations = []
        for index in range(len(self.gripper_ports)):
            calibration = self._load_lk_gripper_calibration_file(index)
            self._lk_gripper_calibrations.append(calibration)
            if not calibration:
                continue

            if calibration.get("gripper_type"):
                self.gripper_type = str(calibration["gripper_type"]).strip().lower()
            for key, values in [
                ("min_torque_iq", self.gripper_min_torque_iqs),
                ("max_torque_iq", self.gripper_max_torque_iqs),
                ("min_encoder", self.gripper_min_encoders),
                ("max_encoder", self.gripper_max_encoders),
                ("min_angle_deg", self.gripper_min_angle_degs),
                ("max_angle_deg", self.gripper_max_angle_degs),
                ("arm_length_m", self.gripper_arm_lengths),
                ("apex_angle_max_deg", self.gripper_apex_angle_max_degs),
                ("gear_ratio", self.gripper_gear_ratios),
            ]:
                if key in calibration:
                    values[index] = calibration[key]

    @staticmethod
    def _get_lk_gripper_classes():
        """Import LK gripper classes lazily so non-LK configs do not require pyserial."""
        global Gripper, LKGripperConfig, LK_GRIPPER_IMPORT_ERROR
        if Gripper is not None and LKGripperConfig is not None:
            return LKGripperConfig, Gripper
        try:
            from third_party.lktech_motor_sdk.example.lktech_gripper import (
                Gripper as ImportedGripper,
                LKGripperConfig as ImportedLKGripperConfig,
            )
        except ImportError as exc:
            LK_GRIPPER_IMPORT_ERROR = exc
            return None, None
        Gripper = ImportedGripper
        LKGripperConfig = ImportedLKGripperConfig
        LK_GRIPPER_IMPORT_ERROR = None
        return LKGripperConfig, Gripper

    def _get_lk_gripper_arm_length(self, index: int) -> float:
        """Return cosine-law arm length for the LK SDK width model."""
        configured = self.gripper_arm_lengths[index]
        if configured is not None:
            return float(configured)
        apex_deg = float(self.gripper_apex_angle_max_degs[index])
        preset = get_lk_gripper_preset(self.gripper_type)
        if preset.get("arm_length_m") is None:
            raise ValueError(f"gripper arm_length_m is required for index {index}")
        return float(preset["arm_length_m"])

    def _switch_tool_if_needed(self, robot, tool, tool_name: str, label: str) -> None:
        """Switch the robot tool if a tool name is configured."""
        if tool is None or not tool_name:
            return

        if robot is not None:
            try:
                logger.info(f"Switching {label} mode to [IDLE] before tool switch")
                robot.switch_mode(self.flexiv_adapter.mode_idle)
                time.sleep(0.2)
            except Exception as exc:
                logger.warning(f"Failed to switch {label} to IDLE before tool switch: {exc}")

        logger.info(f"Switching {label} tool to [{tool_name}]")
        tool.switch(tool_name)

    def _move_flexiv_gripper_home(self, gripper, index: int, label: str) -> bool:
        """Move a Flexiv SDK gripper to the configured physical home width."""
        if gripper is None:
            return False
        home_width = self._get_home_gripper_width(index)
        logger.info(f"Resetting {label} flexiv gripper to home width ({home_width:.4f} m)")
        gripper.move(home_width, float(self.gripper_max_velocities[index]), float(self.gripper_force_limits[index]))
        self.last_gripper_value[index] = self._target_to_measured_gripper_width(home_width)
        return True

    def _read_flexiv_gripper_value(self, gripper, index: int, label: str) -> float:
        """Read a Flexiv SDK gripper using legacy width semantics."""
        if gripper is None:
            return float(self.last_gripper_value[index])
        try:
            raw_width = gripper.states().width
            gripper_value = self._target_to_measured_gripper_width(raw_width)
            self.last_gripper_value[index] = gripper_value
            return gripper_value
        except Exception as e:
            logger.warning(f"Failed to read {label} flexiv gripper value: {e}")
            return float(self.last_gripper_value[index])

    def _move_gripper_home_by_backend(self, gripper, index: int, label: str, wait: bool = False) -> bool:
        """Move a gripper home using the active backend."""
        if self.gripper_backend == "lktech":
            return self._move_gripper_home(gripper, index, label, wait=wait)
        return self._move_flexiv_gripper_home(gripper, index, label)

    def _read_gripper_value_by_backend(self, gripper, index: int, label: str) -> float:
        """Read a gripper value using the active backend."""
        if self.gripper_backend == "lktech":
            return self._read_gripper_value(gripper, index, label)
        return self._read_flexiv_gripper_value(gripper, index, label)

    # R1-parity throttle for streamed LKTech commands.
    GRIPPER_CMD_DEADBAND = 0.02      # normalized target units; ignore tinier changes
    GRIPPER_CMD_DEADBAND_M = 0.002   # width_m target units (meters); ~2 mm
    GRIPPER_CMD_UPDATE_HZ = 20.0     # max re-send rate per gripper

    def _lktech_cmd_throttled(self, gripper_value: float, index: int) -> bool:
        """Return True if this LKTech command should be skipped this cycle.

        Mirrors R1's send gate: only re-issue a position command when the
        normalized target moved by more than a deadband AND a minimum interval
        elapsed. Skipping redundant re-sends lets each multi-turn move run to
        completion at full speed instead of being re-planned every cycle.
        """
        try:
            if self.gripper_value_mode == "width_m":
                target = float(self._measured_to_target_gripper_width(float(gripper_value), index=index))
                deadband = self.GRIPPER_CMD_DEADBAND_M
            else:
                target = float(self._policy_gripper_value_to_normalized_position(float(gripper_value), index))
                deadband = self.GRIPPER_CMD_DEADBAND
        except Exception:
            return False  # can't evaluate target -> don't throttle
        now = time.monotonic()
        last_target = self._gripper_cmd_last_target.get(index)
        last_time = self._gripper_cmd_last_time.get(index, 0.0)
        min_dt = 1.0 / max(1e-3, self.GRIPPER_CMD_UPDATE_HZ)
        changed = last_target is None or abs(target - last_target) > deadband
        if changed and (now - last_time) >= min_dt:
            self._gripper_cmd_last_target[index] = target
            self._gripper_cmd_last_time[index] = now
            return False  # send it
        return True  # throttle

    def _move_gripper_value_by_backend(self, gripper, gripper_value: float, index: int, label: str) -> bool:
        """Command a gripper value using the active backend."""
        if self.gripper_backend == "lktech":
            if gripper is not None and self._lktech_cmd_throttled(gripper_value, index):
                return True
            return self._move_gripper_value(gripper, gripper_value, index, label)

        if gripper is None:
            logger.warning(f"{label} gripper is not available")
            return False
        try:
            target_width = self._measured_to_target_gripper_width(float(gripper_value), index=index)
            gripper.move(
                target_width,
                float(self.gripper_max_velocities[index]),
                float(self.gripper_force_limits[index]),
            )
            self.last_gripper_value[index] = self._target_to_measured_gripper_width(target_width)
            return True
        except Exception as e:
            logger.error(f"Failed to move {label} flexiv gripper to value {gripper_value}: {e}")
            return False

    def _width_to_normalized_gripper_position(self, width: float, index: int) -> float:
        """Convert a physical gripper width in meters to repo/LK-facing normalized position."""
        min_width = float(self.gripper_min_widths[index])
        max_width = float(self.gripper_max_widths[index])
        if max_width <= min_width:
            raise ValueError(
                f"Invalid gripper width range for index {index}: min={min_width}, max={max_width}"
            )

        clipped_width = float(np.clip(width, min_width, max_width))
        open_ratio = (clipped_width - min_width) / (max_width - min_width)
        return float(np.clip(open_ratio, 0.0, 1.0))

    def _normalized_gripper_position_to_width(self, normalized_position: float, index: int) -> float:
        """Convert normalized position back to a physical gripper width in meters."""
        min_width = float(self.gripper_min_widths[index])
        max_width = float(self.gripper_max_widths[index])
        if max_width <= min_width:
            raise ValueError(
                f"Invalid gripper width range for index {index}: min={min_width}, max={max_width}"
            )

        clipped_position = float(np.clip(normalized_position, 0.0, 1.0))
        return float(min_width + clipped_position * (max_width - min_width))

    def _width_to_wrapper_normalized(self, width: float, index: int) -> float:
        """Convert a physical width (m) to the LK wrapper's normalized open ratio n.

        In ``width_m`` mode the wrapper maps n to width via the cosine-law model
        (w = 2 * L * sin(n * theta_max / 2)), so values that feed the wrapper's
        normalized n (e.g. ``home_position``) must use that inverse, NOT the linear
        config min/max-width map (which would only match at the endpoints).
        """
        if self.gripper_value_mode == "width_m":
            L = float(self.gripper_arm_lengths[index])
            theta_max = float(np.radians(float(self.gripper_apex_angles[index])))
            if L > 0.0 and theta_max > 0.0:
                w_max = 2.0 * L * float(np.sin(theta_max / 2.0))
                w = float(np.clip(width, 0.0, w_max))
                ratio = float(np.clip(w / (2.0 * L), 0.0, 1.0))
                theta = 2.0 * float(np.arcsin(ratio))
                return float(np.clip(theta / theta_max, 0.0, 1.0))
        return self._width_to_normalized_gripper_position(width, index)

    @staticmethod
    def _clip_normalized_gripper_position(normalized_position: float) -> float:
        return float(np.clip(normalized_position, 0.0, 1.0))

    def _gripper_width_and_radian(self, gripper, gripper_value: float, index: int) -> Tuple[float, float]:
        """Derive physical width (m) and motor radian from a repo-facing gripper value.

        ``gripper_value`` is the value already read for the state array, so this
        adds NO extra serial traffic.

        - width(m): in ``width_m`` mode the value already IS the physical opening
          width from the cosine-law model (w = 2 * L * sin(theta / 2)); in
          ``normalized`` mode it is mapped via ``gripper_min/max_widths``.
        - radian: from the gripper's multi-turn-angle calibration. The SDK defines
          ``normalized = 1 - (angle - min)/(max - min)`` (see Gripper._read_position),
          so ``angle = min + (1 - normalized) * (max - min)``. Falls back to 0.0 when
          the gripper has no angle calibration (e.g. encoder-only or no gripper).
        """
        if self.gripper_value_mode == "width_m":
            try:
                width_m = float(gripper_value)
            except Exception:
                width_m = 0.0
            normalized = 0.0
            if gripper is not None:
                try:
                    normalized = float(gripper._normalized_from_width(width_m))
                except Exception:
                    normalized = 0.0
        else:
            try:
                normalized = self._clip_normalized_gripper_position(float(gripper_value))
            except Exception:
                normalized = 0.0
            width_m = None
            # LK backend: the repo-facing normalized value is a LINEAR width ratio
            # (n = actual_width / max_width, see _target_width_to_policy_gripper_value),
            # so recover the ACTUAL physical width with the SAME linear scale used on
            # read-back: width = n * max_width. This exactly reproduces the gripper's
            # measured states().width. (The cosine-law model only maps width<->motor
            # angle INSIDE the wrapper and must not be applied to the robot-level n.)
            if gripper is not None and getattr(self, "gripper_backend", None) == "lktech":
                try:
                    width_m = float(normalized * self._lk_gripper_max_width(gripper, index))
                except Exception:
                    width_m = None
            if width_m is None:
                try:
                    width_m = float(self._normalized_gripper_position_to_width(normalized, index))
                except Exception:
                    width_m = 0.0

        radian = 0.0
        if gripper is not None:
            try:
                min_a, max_a, mode = gripper._get_calibrated_range()
                if str(mode) == "multi_turn_angle":
                    angle_deg = float(min_a) + (1.0 - normalized) * (float(max_a) - float(min_a))
                    radian = float(np.radians(angle_deg))
            except Exception:
                radian = 0.0
        return width_m, radian

    def _get_home_gripper_width(self, index: int) -> float:
        """Return the physical home width in meters."""
        if self.gripper_value_mode == "width_m":
            return float(self.gripper_home_widths[index])

        home_ratio = self.gripper_home_ratios[index]
        if home_ratio is None:
            return float(self.gripper_home_widths[index])

        min_width = float(self.gripper_min_widths[index])
        max_width = float(self.gripper_max_widths[index])
        if max_width <= min_width:
            raise ValueError(
                f"Invalid gripper width range for index {index}: min={min_width}, max={max_width}"
            )

        clipped_ratio = float(np.clip(float(home_ratio), 0.0, 1.0))
        return float(min_width + clipped_ratio * (max_width - min_width))

    def _policy_gripper_value_to_normalized_position(self, gripper_value: float, index: int) -> float:
        """Convert repo-facing gripper values into normalized gripper targets."""
        if self.gripper_value_mode == "normalized":
            return self._clip_normalized_gripper_position(gripper_value)

        return float(gripper_value)

    def _normalized_position_to_policy_gripper_value(self, normalized_position: float, index: int) -> float:
        """Convert normalized states back to repo-facing gripper values."""
        if self.gripper_value_mode == "normalized":
            clipped_position = self._clip_normalized_gripper_position(normalized_position)
            return clipped_position

        return float(normalized_position)

    def _lk_gripper_max_width(self, gripper, index: int) -> float:
        """Maximum physical opening width (m) reported by the LK wrapper."""
        try:
            max_width = float(gripper.params().max_width)
            if max_width > 0.0:
                return max_width
        except Exception:
            pass
        return float(self.gripper_max_widths[index])

    def _policy_gripper_value_to_target_width(self, gripper, gripper_value: float, index: int) -> float:
        """Convert a repo-facing gripper value into a physical target width (m).

        The LK wrapper now consumes physical width directly (cosine-law model), so
        we no longer pre-convert to a normalized position. In ``normalized`` mode the
        value is scaled by the gripper's max physical width for backward safety.
        """
        if self.gripper_value_mode == "normalized":
            n = self._clip_normalized_gripper_position(float(gripper_value))
            return float(n * self._lk_gripper_max_width(gripper, index))
        return float(self._measured_to_target_gripper_width(float(gripper_value), index=index))

    def _target_width_to_policy_gripper_value(self, gripper, target_width: float, index: int) -> float:
        """Convert a physical width (m) from the LK wrapper into a repo-facing value."""
        if self.gripper_value_mode == "normalized":
            max_width = self._lk_gripper_max_width(gripper, index)
            if max_width <= 0.0:
                return 0.0
            return float(np.clip(float(target_width) / max_width, 0.0, 1.0))
        return float(self._target_to_measured_gripper_width(float(target_width)))

    def _get_home_gripper_value(self, index: int) -> float:
        """Return the home value in the same semantics used by policy/control."""
        home_width = self._get_home_gripper_width(index)
        if self.gripper_value_mode == "normalized":
            return self._width_to_normalized_gripper_position(home_width, index)
        return home_width

    def _lk_cosine_max_width(self, index: int) -> float:
        """Max physical opening width (m) from the cosine-law model using config L/apex.

        Mirrors the wrapper's max_width = 2 * L * sin(apex / 2) but needs no live
        gripper handle, so it can run inside the realtime scaling path. Falls back to
        the configured gripper_max_width when L/apex are unusable.
        """
        return calculate_lk_gripper_max_width(
            self.gripper_arm_lengths[index],
            self.gripper_apex_angles[index],
            float(self.gripper_max_widths[index]),
        )

    def _scale_realtime_gripper_command(self, gripper_value: float, index: int) -> float:
        """Scale real-time policy gripper commands without affecting reset/calibration flows."""
        # Teleop trigger arrives as a 0-1 width ratio; convert to physical meters
        # (n * cosine max_width) so the width_m pipeline and meters recording stay
        # consistent. Only applies to the realtime stream, never to home/reset/policy.
        if self.gripper_input_ratio and self.gripper_value_mode == "width_m":
            n = self._clip_normalized_gripper_position(float(gripper_value))
            if n < 0.8:
                n = self._clip_normalized_gripper_position(n * float(self.gripper_scale_factors[index]))
            max_width = self._lk_cosine_max_width(index)
            return float(np.clip(n * max_width, 0.0, max_width))

        # NOTE: hardcode here
        if float(gripper_value) < 0.8:
            scaled_value = float(gripper_value) * float(self.gripper_scale_factors[index])
        else:
            scaled_value = float(gripper_value)
        # TODO(remove hack): temporary gripper close bias for 2026-06-26 inference tests.
        scaled_value = max(0.0, scaled_value - float(self.gripper_command_offsets[index]))
        if self.gripper_value_mode == "normalized":
            return self._clip_normalized_gripper_position(scaled_value)

        return float(scaled_value)

    def _build_gripper_runtime_config(self, index: int, label: str):
        """Build runtime config for the bundled LK gripper wrapper."""
        gripper_config_cls, _ = self._get_lk_gripper_classes()
        if gripper_config_cls is None:
            raise ImportError(
                "LKTech gripper support requires the bundled third_party/lktech_motor_sdk "
                "and its Python dependencies, including pyserial."
            ) from LK_GRIPPER_IMPORT_ERROR
        calibration = (
            self._lk_gripper_calibrations[index]
            if hasattr(self, "_lk_gripper_calibrations") and index < len(self._lk_gripper_calibrations)
            else self._load_lk_gripper_calibration_file(index)
        )
        if calibration:
            cal_type = calibration.get("gripper_type")
            if cal_type and str(cal_type).strip().lower() != self.gripper_type:
                logger.warning(
                    f"LK gripper calibration[{index}] type {cal_type!r} differs from "
                    f"configured gripper_type {self.gripper_type!r}; using calibration file values"
                )
        home_width = self._get_home_gripper_width(index)
        return gripper_config_cls(
            port=str(self.gripper_ports[index]),
            motor_id=int(self.gripper_motor_ids[index]),
            baudrate=int(self.gripper_baudrates[index]),
            speed_dps=float(self.gripper_speeds[index]),
            home_position=self._width_to_wrapper_normalized(home_width, index),
            read_timeout=float(self.gripper_timeouts[index]),
            wait_timeout=float(self.gripper_wait_timeouts[index]),
            auto_calibrate=bool(self.gripper_auto_calibrates[index]),
            min_torque_iq=int(self._calibration_value(calibration, "min_torque_iq", self.gripper_min_torque_iqs[index])),
            max_torque_iq=int(self._calibration_value(calibration, "max_torque_iq", self.gripper_max_torque_iqs[index])),
            calibration_timeout=float(self.gripper_calibration_timeouts[index]),
            calibration_settle_time=float(self.gripper_calibration_settle_times[index]),
            min_encoder=(
                int(self._calibration_value(calibration, "min_encoder", self.gripper_min_encoders[index]))
                if self._calibration_value(calibration, "min_encoder", self.gripper_min_encoders[index]) is not None
                else None
            ),
            max_encoder=(
                int(self._calibration_value(calibration, "max_encoder", self.gripper_max_encoders[index]))
                if self._calibration_value(calibration, "max_encoder", self.gripper_max_encoders[index]) is not None
                else None
            ),
            min_angle_deg=(
                float(self._calibration_value(calibration, "min_angle_deg", self.gripper_min_angle_degs[index]))
                if self._calibration_value(calibration, "min_angle_deg", self.gripper_min_angle_degs[index]) is not None
                else None
            ),
            calibration_dir=self.gripper_calibration_dir,
            max_angle_deg=(
                float(self._calibration_value(calibration, "max_angle_deg", self.gripper_max_angle_degs[index]))
                if self._calibration_value(calibration, "max_angle_deg", self.gripper_max_angle_degs[index]) is not None
                else None
            ),
            arm_length_m=float(self._calibration_value(calibration, "arm_length_m", self._get_lk_gripper_arm_length(index))),
            apex_angle_max_deg=float(self._calibration_value(calibration, "apex_angle_max_deg", self.gripper_apex_angle_max_degs[index])),
            gear_ratio=float(self._calibration_value(calibration, "gear_ratio", self.gripper_gear_ratios[index])),
            label=label,
        )

    def _create_lk_gripper(self, robot, index: int, label: str):
        """Build an LK gripper instance from normalized config."""
        _, gripper_cls = self._get_lk_gripper_classes()
        if gripper_cls is None:
            raise ImportError(
                "LKTech gripper support requires the bundled third_party/lktech_motor_sdk "
                "and its Python dependencies, including pyserial."
            ) from LK_GRIPPER_IMPORT_ERROR
        config = self._build_gripper_runtime_config(index, label)
        return gripper_cls(
            robot,
            config=config,
            min_vel=float(self.gripper_min_velocities[index]),
            max_vel=float(self.gripper_max_velocities[index]),
        )

    def _move_gripper_value(self, gripper, gripper_value: float, index: int, label: str, wait: bool = False) -> bool:
        """Command the LK gripper using repo-facing gripper value semantics."""
        if gripper is None:
            logger.warning(f"{label} gripper is not available")
            return False

        try:
            if self.gripper_value_mode == "normalized":
                target_position = self._policy_gripper_value_to_normalized_position(float(gripper_value), index)
                target_width = self._normalized_gripper_position_to_width(target_position, index)
                self.last_gripper_value[index] = self._normalized_position_to_policy_gripper_value(target_position, index)
            else:
                target_width = self._policy_gripper_value_to_target_width(gripper, float(gripper_value), index)
                self.last_gripper_value[index] = float(gripper_value)
                # Record the commanded physical width (m) so the recorder can publish
                # hand/*/cmd distinct from the measured hand/*/state width.
                try:
                    self._last_gripper_cmd_width_m[index] = float(target_width)
                except Exception:
                    pass
            velocity = float(self.gripper_max_velocities[index])
            force_limit = float(self.gripper_force_limits[index])
            ok = gripper.Move(target_width, velocity, force_limit)
            if wait and hasattr(gripper, "motor"):
                return bool(gripper.motor.wait_for_stop(timeout=gripper.wait_timeout))
            return bool(ok)
        except Exception as e:
            logger.error(f"Failed to move {label} gripper to value {gripper_value}: {e}")
            return False

    def _read_gripper_value(self, gripper, index: int, label: str) -> float:
        """Read the current repo-facing gripper value."""
        if gripper is None:
            return float(self.last_gripper_value[index])

        try:
            raw_width = float(gripper.states().width)
            if self.gripper_value_mode == "normalized":
                raw_position = self._width_to_normalized_gripper_position(raw_width, index)
                gripper_value = self._normalized_position_to_policy_gripper_value(raw_position, index)
            else:
                gripper_value = self._target_width_to_policy_gripper_value(gripper, raw_width, index)

            self.last_gripper_value[index] = gripper_value
            return gripper_value
        except Exception as e:
            logger.warning(f"Failed to read {label} gripper value: {e}")
            return float(self.last_gripper_value[index])

    def _move_gripper_home(self, gripper, index: int, label: str, wait: bool = False) -> bool:
        """Move a gripper to its configured home target."""
        home_value = self._get_home_gripper_value(index)
        home_width = self._get_home_gripper_width(index)
        ratio = self.gripper_home_ratios[index]
        ratio_text = "None" if ratio is None else f"{float(ratio):.4f}"
        logger.info(
            f"Resetting {label} gripper to home value "
            f"({home_value:.4f}, mode={self.gripper_value_mode}, width_m={home_width:.4f}, ratio={ratio_text})"
        )
        return self._move_gripper_value(gripper, home_value, index, label, wait=wait)

    def _shutdown_grippers(self) -> None:
        """Power off and disconnect any initialized grippers."""
        gripper_attrs = ("gripper", "gripper_left", "gripper_right")
        for gripper_attr in gripper_attrs:
            gripper = getattr(self, gripper_attr, None)
            if gripper is None:
                continue

            disable_fn = getattr(gripper, "Disable", None)
            if disable_fn is None:
                disable_fn = getattr(gripper, "disable", None)
            if not callable(disable_fn):
                logger.warning(f"Gripper {gripper_attr} has no disable method")
                continue

            try:
                disable_fn()
            except Exception as e:
                logger.warning(f"Failed during {gripper_attr} shutdown: {e}")
            finally:
                setattr(self, gripper_attr, None)

    def _load_gripper_calibration_data(self) -> None:
        """Load gripper calibration data from JSON file."""
        import json
        try:
            with open(self.inference_gripper_calibration_path, 'r') as f:
                calibration_json = json.load(f)

            data_points = calibration_json.get("data_points", [])
            if not data_points:
                logger.warning("No calibration data points found in file")
                return

            # Sort data points by target_width for interpolation
            sorted_points = sorted(data_points, key=lambda x: x["target_width"])

            self.gripper_calibration_data = {
                "target_widths": np.array([p["target_width"] for p in sorted_points]),
                "measured_widths": np.array([p["measured_width"] for p in sorted_points])
            }

            logger.info(f"Loaded gripper calibration data from {self.inference_gripper_calibration_path}")
            logger.info(f"  Target width range: [{self.gripper_calibration_data['target_widths'][0]:.4f}, {self.gripper_calibration_data['target_widths'][-1]:.4f}]")
            logger.info(f"  Measured width range: [{self.gripper_calibration_data['measured_widths'][0]:.4f}, {self.gripper_calibration_data['measured_widths'][-1]:.4f}]")

        except Exception as e:
            logger.error(f"Failed to load gripper calibration data: {e}")
            self.gripper_calibration_data = None

    def _target_to_measured_gripper_width(self, target_width: float) -> float:
        """
        Convert API target width to measured width using calibration data.
        Used in get_robot_state to convert read values.

        Args:
            target_width: Width value read from API

        Returns:
            Corresponding measured width
        """
        if self.gripper_calibration_data is None:
            return target_width

        target_widths = self.gripper_calibration_data["target_widths"]
        measured_widths = self.gripper_calibration_data["measured_widths"]

        # Clip to calibration range
        target_min, target_max = target_widths[0], target_widths[-1]
        clipped_target = np.clip(target_width, target_min, target_max)

        # Find the two nearest points for interpolation
        idx = np.searchsorted(target_widths, clipped_target)

        if idx == 0:
            return measured_widths[0]
        elif idx >= len(target_widths):
            return measured_widths[-1]
        else:
            # Linear interpolation between two nearest points
            t0, t1 = target_widths[idx - 1], target_widths[idx]
            m0, m1 = measured_widths[idx - 1], measured_widths[idx]

            # Interpolation factor
            alpha = (clipped_target - t0) / (t1 - t0) if t1 != t0 else 0.0
            return m0 + alpha * (m1 - m0)

    def _measured_to_target_gripper_width(self, measured_width: float, index: int = 0) -> float:
        """
        Convert desired measured width to API target width using calibration data.
        Used in _execute_eef_command to convert command values.

        Args:
            measured_width: Desired measured width

        Returns:
            Corresponding target width to send to API
        """
        if self.gripper_calibration_data is None:
            return float(measured_width)

        target_widths = self.gripper_calibration_data["target_widths"]
        measured_widths = self.gripper_calibration_data["measured_widths"]

        # Clip to calibration range (measured widths)
        measured_min, measured_max = measured_widths[0], measured_widths[-1]
        clipped_measured = np.clip(measured_width, measured_min, measured_max)

        # Find the two nearest points for interpolation (search in measured_widths)
        idx = np.searchsorted(measured_widths, clipped_measured)

        if idx == 0:
            return float(target_widths[0])
        elif idx >= len(measured_widths):
            return float(target_widths[-1])
        else:
            # Linear interpolation between two nearest points
            m0, m1 = measured_widths[idx - 1], measured_widths[idx]
            t0, t1 = target_widths[idx - 1], target_widths[idx]

            # Interpolation factor
            alpha = (clipped_measured - m0) / (m1 - m0) if m1 != m0 else 0.0
            return float(t0 + alpha * (t1 - t0))

    def _configure_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure home configuration from YAML dict."""
        try:
            if self.is_dual_arm:
                self._configure_dual_arm_home(home_cfg)
            else:
                self._configure_single_arm_home(home_cfg)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error(f"Failed to configure home parameters: {exc}")
            raise

    def _configure_dual_arm_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure dual-arm home joints and poses."""
        presets = home_cfg.get("presets")
        preset_index = home_cfg.get("preset_index")

        if presets is not None:
            if not isinstance(presets, list) or len(presets) == 0:
                raise ValueError("home.presets for dual-arm configuration must be a non-empty list")
            self.home_presets = presets

            if preset_index is None:
                preset_index = 0
            if not isinstance(preset_index, int):
                raise TypeError("home.preset_index must be an integer")
            if not (0 <= preset_index < len(presets)):
                raise IndexError(f"home.preset_index {preset_index} is out of bounds for {len(presets)} presets")

            self.home_preset_index = preset_index
            selected_preset = presets[preset_index]

            if not isinstance(selected_preset, dict):
                raise TypeError(f"Preset at index {preset_index} must be a mapping")

            joints_left = selected_preset.get("joints_left")
            joints_right = selected_preset.get("joints_right")
            pose_left = selected_preset.get("pose_left")
            pose_right = selected_preset.get("pose_right")

            if joints_left is not None:
                self.home_joints_left = joints_left
                logger.info(f"Configured home joints for left arm (preset {preset_index}): {self.home_joints_left}")
            if joints_right is not None:
                self.home_joints_right = joints_right
                logger.info(f"Configured home joints for right arm (preset {preset_index}): {self.home_joints_right}")
            if pose_left is not None:
                self.home_pose_left = pose_left
                logger.info(f"Configured home pose for left arm (preset {preset_index}): {self.home_pose_left}")
            if pose_right is not None:
                self.home_pose_right = pose_right
                logger.info(f"Configured home pose for right arm (preset {preset_index}): {self.home_pose_right}")
        else:
            if "preset_index" in home_cfg:
                raise KeyError("home.preset_index specified without defining home.presets")

            if "joints_left" in home_cfg:
                self.home_joints_left = home_cfg["joints_left"]
                logger.info(f"Configured home joints for left arm: {self.home_joints_left}")
            if "joints_right" in home_cfg:
                self.home_joints_right = home_cfg["joints_right"]
                logger.info(f"Configured home joints for right arm: {self.home_joints_right}")
            if "pose_left" in home_cfg:
                self.home_pose_left = home_cfg["pose_left"]
                logger.info(f"Configured home pose for left arm: {self.home_pose_left}")
            if "pose_right" in home_cfg:
                self.home_pose_right = home_cfg["pose_right"]
                logger.info(f"Configured home pose for right arm: {self.home_pose_right}")

    def _configure_single_arm_home(self, home_cfg: Dict[str, Any]) -> None:
        """Configure single-arm home joints and pose."""
        presets = home_cfg.get("presets")
        preset_index = home_cfg.get("preset_index")

        if presets is not None:
            if not isinstance(presets, list) or len(presets) == 0:
                raise ValueError("home.presets must be a non-empty list")

            self.home_presets = presets

            if preset_index is None:
                preset_index = 0
            if not isinstance(preset_index, int):
                raise TypeError("home.preset_index must be an integer")
            if not (0 <= preset_index < len(presets)):
                raise IndexError(f"home.preset_index {preset_index} is out of bounds for {len(presets)} presets")

            self.home_preset_index = preset_index
            selected_preset = presets[preset_index]

            if not isinstance(selected_preset, dict):
                raise TypeError(f"Preset at index {preset_index} must be a mapping")

            if "joints" in selected_preset:
                self.home_joints = selected_preset["joints"]
                logger.info(f"Configured home joints (preset {preset_index}): {self.home_joints}")
            if "pose" in selected_preset:
                self.home_pose = selected_preset["pose"]
                logger.info(f"Configured home pose (preset {preset_index}): {self.home_pose}")
        else:
            if "preset_index" in home_cfg:
                raise KeyError("home.preset_index specified without defining home.presets")

            if "joints" in home_cfg:
                self.home_joints = home_cfg["joints"]
                logger.info(f"Configured home joints: {self.home_joints}")
            if "pose" in home_cfg:
                self.home_pose = home_cfg["pose"]
                logger.info(f"Configured home pose: {self.home_pose}")

    def _connect_robot(self) -> bool:
        """Connect to Rizon robot(s)."""
        try:
            if self.is_dual_arm:
                return self._connect_dual_arm_robots()
            else:
                return self._connect_single_robot()
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            return False

    def _connect_single_robot(self) -> bool:
        """Connect to single Rizon robot."""
        try:
            logger.info(f"Connecting to Rizon robot: {self.robot_sn}")

            # Create robot interface
            logger.info("Creating robot interface...")
            self.robot = self.flexiv_adapter.create_robot(self.robot_sn)
            logger.info("Robot interface created successfully")

            # Check for faults and enable
            if not self._check_and_enable_robot(self.robot, "Robot"):
                return False

            self.is_enabled = True

            # Initialize gripper if enabled
            if self.utilize_gripper:
                if not self._initialize_gripper():
                    logger.error("Failed to initialize gripper")
                    return False

            self.is_operational = True
            logger.info("Robot connected and operational")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to single robot: {e}")
            return False

    def _connect_dual_arm_robots(self) -> bool:
        """Connect to dual arm Rizon robots."""
        try:
            logger.info(f"Connecting to dual-arm Rizon robots: {self.left_sn}, {self.right_sn}")

            # Create left robot interface
            logger.info("Creating left robot interface...")
            self.robot_left = self.flexiv_adapter.create_robot(self.left_sn)
            logger.info("Left robot interface created successfully")

            # Create right robot interface
            logger.info("Creating right robot interface...")
            self.robot_right = self.flexiv_adapter.create_robot(self.right_sn)
            logger.info("Right robot interface created successfully")

            # Check for faults and enable both robots
            if not self._check_and_enable_robot(self.robot_left, "Left robot"):
                return False
            if not self._check_and_enable_robot(self.robot_right, "Right robot"):
                return False

            self.is_enabled = True

            # Initialize grippers if enabled
            if self.utilize_gripper:
                if not self._initialize_dual_grippers():
                    logger.error("Failed to initialize grippers")
                    return False

            self.is_operational = True
            logger.info("Dual-arm robots connected and operational")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to dual-arm robots: {e}")
            return False

    def _check_and_enable_robot(self, robot, name: str) -> bool:
        """Check for faults and enable a robot."""
        try:
            # Check for faults
            logger.info(f"Checking {name} status...")
            if robot.fault():
                logger.warning(f"{name} has faults, attempting to clear...")
                if not robot.clear_fault():
                    logger.error(f"Failed to clear {name} faults")
                    return False
                logger.info(f"{name} faults cleared successfully")
            else:
                logger.info(f"{name} status: No faults detected")

            # Enable robot
            logger.info(f"Enabling {name}...")
            robot.enable()

            # Wait for operational status
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            logger.info(f"Waiting for {name} to become operational...")

            while not robot.is_operational():
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.error(f"Timeout waiting for {name} to become operational after {timeout} seconds")
                    return False

                # Show progress every 5 seconds
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    remaining = timeout - elapsed
                    logger.info(f"Still waiting for {name}... {remaining:.1f}s remaining")

                time.sleep(1.0)

            logger.info(f"{name} is now operational")
            return True

        except Exception as e:
            logger.error(f"Failed to enable {name}: {e}")
            return False

    def _initialize_gripper(self) -> bool:
        """Initialize gripper for single arm robot."""
        try:
            if self.gripper_backend == "flexiv":
                logger.info(f"Initializing flexiv gripper [{self.gripper_name}]")
                self.gripper = self.flexiv_adapter.create_gripper(self.robot)
                self.tool = self.flexiv_adapter.create_tool(self.robot)
                logger.info(f"Enabling gripper [{self.gripper_name}]")
                self.gripper.enable(self.gripper_name)

                logger.info("Gripper parameters:")
                logger.info(f"  name: {self.gripper.params().name}")
                logger.info(f"  min_width: {round(self.gripper.params().min_width, 3)}")
                logger.info(f"  max_width: {round(self.gripper.params().max_width, 3)}")
                logger.info(f"  min_force: {round(self.gripper.params().min_force, 3)}")
                logger.info(f"  max_force: {round(self.gripper.params().max_force, 3)}")
                logger.info(f"  min_vel: {round(self.gripper.params().min_vel, 3)}")
                logger.info(f"  max_vel: {round(self.gripper.params().max_vel, 3)}")

                self._switch_tool_if_needed(self.robot, self.tool, self.tool_name, "robot")
                self._move_flexiv_gripper_home(self.gripper, 0, "single")
                logger.info("Gripper initialized successfully")
                return True

            logger.info("Initializing LK gripper for single arm")

            self.gripper = self._create_lk_gripper(self.robot, 0, self.gripper_name or "gripper")
            self.tool = self.flexiv_adapter.create_tool(self.robot)

            logger.info(
                f"LK gripper config: port={self.gripper_ports[0]}, id={self.gripper_motor_ids[0]}, "
                f"speed_limit={float(self.gripper_speeds[0]):.2f} deg/s, "
                f"home_width={self._get_home_gripper_width(0):.4f} m, "
                f"home_ratio={self.gripper_home_ratios[0]}"
            )
            self.gripper.Enable(self.gripper_name or "gripper")
            if not self.gripper.Init():
                self._shutdown_grippers()
                return False

            # Print gripper parameters
            logger.info("Gripper parameters:")
            logger.info(f"  name: {self.gripper.params().name}")
            logger.info(f"  min_width: {round(self.gripper.params().min_width, 3)}")
            logger.info(f"  max_width: {round(self.gripper.params().max_width, 3)}")
            logger.info(f"  min_force: {round(self.gripper.params().min_force, 3)}")
            logger.info(f"  max_force: {round(self.gripper.params().max_force, 3)}")
            logger.info(f"  min_vel: {round(self.gripper.params().min_vel, 3)}")
            logger.info(f"  max_vel: {round(self.gripper.params().max_vel, 3)}")

            self._switch_tool_if_needed(self.robot, self.tool, self.tool_name, "robot")
            if not self._move_gripper_home_by_backend(self.gripper, 0, "single", wait=True):
                self._shutdown_grippers()
                return False
            logger.info("Gripper initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize gripper: {e}")
            self._shutdown_grippers()
            return False

    def _initialize_dual_grippers(self) -> bool:
        """Initialize grippers for dual arm robots."""
        try:
            if self.gripper_backend == "flexiv":
                logger.info(f"Initializing flexiv grippers [{self.gripper_name}] for dual arms")
                self.gripper_left = self.flexiv_adapter.create_gripper(self.robot_left)
                self.gripper_right = self.flexiv_adapter.create_gripper(self.robot_right)
                self.tool_left = self.flexiv_adapter.create_tool(self.robot_left)
                self.tool_right = self.flexiv_adapter.create_tool(self.robot_right)

                logger.info(f"Enabling left gripper [{self.gripper_name}]")
                self.gripper_left.enable(self.gripper_name)
                logger.info(f"Enabling right gripper [{self.gripper_name}]")
                self.gripper_right.enable(self.gripper_name)

                logger.info("Left gripper parameters:")
                logger.info(f"  name: {self.gripper_left.params().name}")
                logger.info(f"  min_width: {round(self.gripper_left.params().min_width, 3)}")
                logger.info(f"  max_width: {round(self.gripper_left.params().max_width, 3)}")

                self._switch_tool_if_needed(self.robot_left, self.tool_left, self.tool_name, "left robot")
                self._switch_tool_if_needed(self.robot_right, self.tool_right, self.tool_name, "right robot")
                self._move_flexiv_gripper_home(self.gripper_left, 0, "left")
                self._move_flexiv_gripper_home(self.gripper_right, 1, "right")

                logger.info("Dual grippers initialized successfully")
                return True

            logger.info("Initializing LK grippers for dual arms")
            arm_specs = [
                ("left", 0, self.robot_left, "gripper_left", "tool_left"),
                ("right", 1, self.robot_right, "gripper_right", "tool_right"),
            ]
            for arm_label, index, robot, gripper_attr, tool_attr in arm_specs:
                if not self.gripper_active_by_arm[index]:
                    logger.info(f"{arm_label.capitalize()} gripper is not configured")
                    setattr(self, gripper_attr, None)
                    setattr(self, tool_attr, None)
                    continue

                # Use "gripper-left"/"gripper-right" so the SDK calibration
                # filename ({label}_id{motor_id}_calibration.json) matches the
                # angle-based files R1 already saved under calibration_dir. This
                # loads multi_turn_angle calibration (fast) instead of the slow
                # encoder default.
                gripper_label = f"gripper-{arm_label}"
                gripper = self._create_lk_gripper(robot, index, gripper_label)
                tool = self.flexiv_adapter.create_tool(robot)
                setattr(self, gripper_attr, gripper)
                setattr(self, tool_attr, tool)

                logger.info(
                    f"{arm_label.capitalize()} LK gripper config: "
                    f"port={self.gripper_ports[index]}, id={self.gripper_motor_ids[index]}, "
                    f"speed_limit={float(self.gripper_speeds[index]):.2f} deg/s, "
                    f"home_width={self._get_home_gripper_width(index):.4f} m, "
                    f"home_ratio={self.gripper_home_ratios[index]}"
                )

                gripper.Enable(gripper_label)
                if not gripper.Init():
                    self._shutdown_grippers()
                    return False

                logger.info(f"{arm_label.capitalize()} gripper parameters:")
                logger.info(f"  name: {gripper.params().name}")
                logger.info(f"  min_width: {round(gripper.params().min_width, 3)}")
                logger.info(f"  max_width: {round(gripper.params().max_width, 3)}")

                self._switch_tool_if_needed(robot, tool, self.tool_name, f"{arm_label} robot")
                if not self._move_gripper_home_by_backend(gripper, index, arm_label, wait=True):
                    self._shutdown_grippers()
                    return False

            logger.info("Dual grippers initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize dual grippers: {e}")
            self._shutdown_grippers()
            return False

    def _switch_to_joint_control(self, is_nrt: bool = True) -> bool:
        """Switch robot(s) to joint position control mode (NRT or RT)."""
        try:
            target_mode = (
                self.flexiv_adapter.mode_joint_position
                if is_nrt
                else self.flexiv_adapter.mode_rt_joint_position
            )
            mode_name = "NRT_JOINT_POSITION" if is_nrt else "RT_JOINT_POSITION"
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False
                # Switch both robots to joint control
                self.robot_left.switch_mode(target_mode)
                self.robot_right.switch_mode(target_mode)
                logger.info(f"Switched dual-arm robots to {mode_name} mode")
            else:
                if not self.robot:
                    return False
                # Switch single robot to joint control
                self.robot.switch_mode(target_mode)
                logger.info(f"Switched robot to {mode_name} mode")

            # Warning: Cartesian impedance parameters are not used in joint control mode
            logger.warning("Joint control mode active - Impedance parameters are not applied")

            self.current_mode = mode_name
            return True

        except Exception as e:
            logger.error(f"Failed to switch to joint control mode: {e}")
            return False

    def _switch_to_eef_control(self, is_nrt: bool = True) -> bool:
        """Switch robot(s) to end-effector control mode (NRT or RT)."""
        try:
            target_mode = (
                self.flexiv_adapter.mode_cartesian_motion_force
                if is_nrt
                else self.flexiv_adapter.mode_rt_cartesian_motion_force
            )
            mode_name = "NRT_CARTESIAN_MOTION_FORCE" if is_nrt else "RT_CARTESIAN_MOTION_FORCE"
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False
                # Switch both robots to Cartesian control
                self.robot_left.switch_mode(target_mode)
                self.robot_right.switch_mode(target_mode)
                self.robot_left.set_force_control_axis([False, False, False, False, False, False])
                self.robot_right.set_force_control_axis([False, False, False, False, False, False])
                self.robot_left.set_cartesian_impedance(self.impedance_parameter)
                self.robot_right.set_cartesian_impedance(self.impedance_parameter)
                if self.max_contact_wrench is not None:
                    self.robot_left.set_max_contact_wrench(self.max_contact_wrench)
                    self.robot_right.set_max_contact_wrench(self.max_contact_wrench)
                    logger.info(
                        f"Max contact wrench applied to dual-arm: {self.max_contact_wrench}"
                    )
                logger.info(f"Switched dual-arm robots to {mode_name} mode")
            else:
                if not self.robot:
                    return False
                # Switch single robot to Cartesian control
                self.robot.switch_mode(target_mode)
                self.robot.set_force_control_axis([False, False, False, False, False, False])
                self.robot.set_cartesian_impedance(self.impedance_parameter)
                if self.max_contact_wrench is not None:
                    self.robot.set_max_contact_wrench(self.max_contact_wrench)
                    logger.info(
                        f"Max contact wrench applied: {self.max_contact_wrench}"
                    )
                logger.info(f"Switched robot to {mode_name} mode")

            self.current_mode = mode_name
            return True

        except Exception as e:
            logger.error(f"Failed to switch to end-effector control mode: {e}")
            return False

    def switch_to(self, target: str, is_nrt: bool = True) -> bool:
        """
        Public switch interface for runtime control mode switching.

        Args:
            target: "joint" or "eef".
            is_nrt: True for NRT modes, False for RT modes.
        """
        normalized_target = target.strip().lower()
        if normalized_target == "joint":
            return self._switch_to_joint_control(is_nrt=is_nrt)
        if normalized_target == "eef":
            return self._switch_to_eef_control(is_nrt=is_nrt)
        logger.error(f"Unsupported switch target: {target}")
        return False

    def _is_joint_mode_active(self) -> bool:
        """Return True when robot is in either NRT or RT joint position mode."""
        return self.current_mode in {"NRT_JOINT_POSITION", "RT_JOINT_POSITION"}

    def _is_eef_mode_active(self) -> bool:
        """Return True when robot is in either NRT or RT Cartesian motion-force mode."""
        return self.current_mode in {"NRT_CARTESIAN_MOTION_FORCE", "RT_CARTESIAN_MOTION_FORCE"}

    @staticmethod
    def _flexiv_pose_to_xyzw_pose(pose: Union[List[float], np.ndarray]) -> List[float]:
        """Convert Flexiv pose [x,y,z,qw,qx,qy,qz] to [x,y,z,qx,qy,qz,qw]."""
        p = np.asarray(pose, dtype=float).reshape(-1)
        return [float(p[0]), float(p[1]), float(p[2]), float(p[4]), float(p[5]), float(p[6]), float(p[3])]

    @staticmethod
    def _xyzw_pose_to_flexiv_pose(pose: Union[List[float], np.ndarray]) -> List[float]:
        """Convert [x,y,z,qx,qy,qz,qw] to Flexiv pose [x,y,z,qw,qx,qy,qz]."""
        p = np.asarray(pose, dtype=float).reshape(-1)
        return [float(p[0]), float(p[1]), float(p[2]), float(p[6]), float(p[3]), float(p[4]), float(p[5])]

    def _send_joint_position_command(
        self,
        robot: Any,
        positions: Union[List[float], np.ndarray],
        max_vel: List[float],
        max_acc: List[float],
        is_nrt: bool = True,
    ) -> None:
        """Send joint command through NRT send API or RT stream API."""
        positions_list = np.asarray(positions, dtype=float).reshape(-1).tolist()
        dof = len(positions_list)
        if is_nrt:
            robot.send_joint_position(
                positions=positions_list,
                velocities=[0.0] * dof,
                max_vel=max_vel,
                max_acc=max_acc,
            )
            return
        robot.stream_joint_position(
            positions=positions_list,
            velocities=[0.0] * dof,
            accelerations=[0.0] * dof,
        )

    def _send_cartesian_motion_command(
        self,
        robot: Any,
        pose: Union[List[float], np.ndarray],
        is_nrt: bool = True
    ) -> None:
        """Send Cartesian command through NRT send API or RT stream API."""
        pose_list = np.asarray(pose, dtype=float).reshape(-1).tolist()
        if is_nrt:
            robot.send_cartesian_motion_force(
                pose_list,
                max_linear_vel=self.max_linear_vel,
                max_linear_acc=self.max_linear_acc,
                max_angular_vel=self.max_angular_vel,
                max_angular_acc=self.max_angular_acc,
            )
        else:
            robot.stream_cartesian_motion_force(
                pose=pose_list,
                wrench=[0.0] * 6,
                velocity=[0.0] * 6,
                acceleration=[0.0] * 6,
            )
        # Cache every successfully sent target, regardless of the Flexiv API path.
        # Idle ticks re-send this exact command target rather than substituting the
        # measured TCP pose, which can differ under Cartesian impedance control.
        self._hold_pose_by_robot[id(robot)] = pose_list

    def _resend_hold_targets(self) -> None:
        """Re-stream the last Cartesian target(s) to keep the arm(s) held.

        Called on idle command ticks (no fresh teleop frame). No-op for joint
        control, policy/dual-control rollout, or before any target was sent.
        """
        if self.control_type not in ("eef_abs", "eef_rel"):
            return
        if self._is_dual_source_mode():
            return
        if not self._hold_pose_by_robot:
            return
        try:
            if self.is_dual_arm:
                robots = (
                    self.robot_left,
                    self.robot_right,
                )
            else:
                robots = (self.robot,)
            for robot in robots:
                if robot is None:
                    continue
                pose_list = self._hold_pose_by_robot.get(id(robot))
                if pose_list is None:
                    continue
                robot.send_cartesian_motion_force(
                    pose_list,
                    max_linear_vel=self.max_linear_vel,
                    max_linear_acc=self.max_linear_acc,
                    max_angular_vel=self.max_angular_vel,
                    max_angular_acc=self.max_angular_acc,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Hold-target re-stream skipped: {exc}")

    def _on_idle_control_tick(self) -> None:
        """Keep the latest Rizon Cartesian target active on idle ticks."""
        self._resend_hold_targets()

    def _set_home_pose(self) -> bool:
        """Set home pose from config or current pose if not configured."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return False

                # If home poses not configured, use current poses
                if self.home_pose_left is None or self.home_pose_right is None:
                    logger.info("Home poses not configured, using current poses")

                    # Check if home_joints are configured - if so, move there first and record poses
                    if self.home_joints_left is not None or self.home_joints_right is not None:
                        logger.info("Moving to configured home joint positions")
                        try:
                            # Ensure joint control mode before sending joint positions
                            return_to_eef = self.control_type in ["eef_abs", "eef_rel"]
                            if not self._is_joint_mode_active():
                                if not self._switch_to_joint_control():
                                    raise RuntimeError("Failed to switch to joint position control before homing")
                            # Move left arm to home joint positions if configured
                            if self.home_joints_left is not None:
                                logger.info(f"Moving left arm to home joints: {self.home_joints_left}")
                                self._send_joint_position_command(
                                    self.robot_left,
                                    self.home_joints_left,
                                    self.max_velocity[:7],
                                    self.max_acceleration[:7],
                                )

                            # Move right arm to home joint positions if configured
                            if self.home_joints_right is not None:
                                logger.info(f"Moving right arm to home joints: {self.home_joints_right}")
                                self._send_joint_position_command(
                                    self.robot_right,
                                    self.home_joints_right,
                                    self.max_velocity[7:14],
                                    self.max_acceleration[7:14],
                                )

                            # Wait for motion to complete
                            max_wait_time = 10.0
                            start_time = time.time()
                            while time.time() - start_time < max_wait_time:
                                left_reached = right_reached = False
                                if self.home_joints_left is not None and np.allclose(self.robot_left.states().q, self.home_joints_left, atol=1e-3):
                                    left_reached = True
                                if self.home_joints_right is not None and np.allclose(self.robot_right.states().q, self.home_joints_right, atol=1e-3):
                                    right_reached = True
                                if left_reached and right_reached:
                                    break
                                time.sleep(0.1)
                            logger.info("Dual-arm robots moved to home joint positions")

                            # Reset gripper widths to home position (same as initialization)
                            if self.utilize_gripper and self.gripper_left:
                                self._move_gripper_home_by_backend(self.gripper_left, 0, "left", wait=True)
                            if self.utilize_gripper and self.gripper_right:
                                self._move_gripper_home_by_backend(self.gripper_right, 1, "right", wait=True)

                            # Now get the poses at these joint configurations and set as home poses
                            left_states = self.robot_left.states()
                            right_states = self.robot_right.states()
                            left_tcp = left_states.tcp_pose.copy()
                            right_tcp = right_states.tcp_pose.copy()

                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            if self.home_pose_left is None:
                                self.home_pose_left = self._flexiv_pose_to_xyzw_pose(left_tcp)
                                logger.info(f"Recorded home pose for left arm from home joints: pos={self.home_pose_left[:3]}, quat={self.home_pose_left[3:]}")

                            if self.home_pose_right is None:
                                self.home_pose_right = self._flexiv_pose_to_xyzw_pose(right_tcp)
                                logger.info(f"Recorded home pose for right arm from home joints: pos={self.home_pose_right[:3]}, quat={self.home_pose_right[3:]}")

                            # Switch back to EEF control if needed
                            if return_to_eef and not self._is_eef_mode_active():
                                if not self._switch_to_eef_control():
                                    logger.warning("Failed to switch back to EEF control after homing")

                        except Exception as e:
                            logger.error(f"Failed to move to home joints: {e}")
                            # Fall back to using current poses
                            logger.info("Falling back to using current poses as home poses")
                            left_states = self.robot_left.states()
                            right_states = self.robot_right.states()
                            left_tcp = left_states.tcp_pose.copy()
                            right_tcp = right_states.tcp_pose.copy()

                            if self.home_pose_left is None:
                                self.home_pose_left = self._flexiv_pose_to_xyzw_pose(left_tcp)
                            if self.home_pose_right is None:
                                self.home_pose_right = self._flexiv_pose_to_xyzw_pose(right_tcp)

                    else:
                        logger.info(f"Current left joints: {self.robot_left.states().q}")
                        logger.info(f"Current right joints: {self.robot_right.states().q}")

                    # If still no home poses, get current poses
                    if self.home_pose_left is None or self.home_pose_right is None:
                        # Get current poses for both arms
                        left_states = self.robot_left.states()
                        right_states = self.robot_right.states()

                        # Set home poses (TCP pose: [x, y, z, qw, qx, qy, qz] -> [x, y, z, qx, qy, qz, qw])
                        left_tcp = left_states.tcp_pose.copy()
                        right_tcp = right_states.tcp_pose.copy()

                        if self.home_pose_left is None:
                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            self.home_pose_left = self._flexiv_pose_to_xyzw_pose(left_tcp)

                        if self.home_pose_right is None:
                            # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                            self.home_pose_right = self._flexiv_pose_to_xyzw_pose(right_tcp)

                logger.info(f"Using dual-arm home poses:")
                logger.info(f"  Left: pos={self.home_pose_left[:3]}, quat={self.home_pose_left[3:]}")
                logger.info(f"  Right: pos={self.home_pose_right[:3]}, quat={self.home_pose_right[3:]}")
            else:
                if not self.robot:
                    return False

                # Check if home_joints is configured - if so, move there first and record pose
                if self.home_joints is not None:
                    logger.info(f"Moving to configured home joints: {self.home_joints}")
                    try:
                        # Ensure joint control mode before sending joint positions
                        return_to_eef = self.control_type in ["eef_abs", "eef_rel"]
                        if not self._is_joint_mode_active():
                            if not self._switch_to_joint_control():
                                raise RuntimeError("Failed to switch to joint position control before homing")

                        # Move to home joint positions
                        self._send_joint_position_command(
                            self.robot,
                            self.home_joints,
                            self.max_velocity,
                            self.max_acceleration,
                        )
                        # Wait for motion to complete
                        max_wait_time = 10.0
                        start_time = time.time()
                        while time.time() - start_time < max_wait_time:
                            reached = False
                            if self.home_joints is not None and np.allclose(self.robot.states().q, self.home_joints, atol=1e-3):
                                reached = True
                            if reached:
                                break
                            time.sleep(0.1)
                        logger.info("Robot moved to home joint positions")

                        # Reset gripper width to home position (same as initialization)
                        if self.utilize_gripper and self.gripper:
                            self._move_gripper_home_by_backend(self.gripper, 0, "single", wait=True)

                        # Now get the pose at this joint configuration and set as home_pose
                        states = self.robot.states()
                        tcp = states.tcp_pose.copy()

                        # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                        self.home_pose = self._flexiv_pose_to_xyzw_pose(tcp)
                        logger.info(f"Recorded home pose from home joints: pos={self.home_pose[:3]}, quat={self.home_pose[3:]}")

                        # Switch back to EEF control if needed
                        if return_to_eef and not self._is_eef_mode_active():
                            if not self._switch_to_eef_control():
                                logger.warning("Failed to switch back to EEF control after homing")

                    except Exception as e:
                        logger.error(f"Failed to move to home joints: {e}")
                        # Fall back to using current pose
                        logger.info("Falling back to using current pose as home pose")
                        states = self.robot.states()
                        tcp = states.tcp_pose.copy()
                        self.home_pose = self._flexiv_pose_to_xyzw_pose(tcp)

                else:
                    logger.info(f"Current joints: {self.robot.states().q}")

                # If home pose not configured, use current pose
                if self.home_pose is None:
                    logger.info("Home pose not configured, using current pose")

                    # Get current pose for single arm
                    states = self.robot.states()
                    tcp = states.tcp_pose.copy()

                    # Convert from Flexiv format [x,y,z,qw,qx,qy,qz] to standard [x,y,z,qx,qy,qz,qw]
                    self.home_pose = self._flexiv_pose_to_xyzw_pose(tcp)

                logger.info(f"Using home pose: pos={self.home_pose[:3]}, quat={self.home_pose[3:]}")

            return True

        except Exception as e:
            logger.error(f"Failed to set home pose: {e}")
            return False

    def _check_and_execute_home_command(self) -> None:
        """Check for external home command file and execute if present."""
        try:
            import os
            from pathlib import Path

            # Check for home command file (use pattern from config if available)
            home_command_pattern = self._robot_home_command_pattern
            assert home_command_pattern is not None, "home_command_pattern is required in signal_files"
            home_command_file = Path(home_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id))

            if home_command_file.exists():
                try:
                    # Read home index from file
                    with open(home_command_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            home_idx = int(content)
                        else:
                            home_idx = self.home_preset_index if self.home_preset_index is not None else 0

                    # Delete the command file
                    home_command_file.unlink()

                    # Execute home command
                    logger.info(f"Received external home command with index {home_idx}")

                    # Block the (concurrent) command thread from executing teleop
                    # frames while we home: otherwise it fights the home motion by
                    # re-streaming the stale pre-home Cartesian target.
                    self._home_command_active = True

                    # IMPORTANT: Clear robot control SHM buffers FIRST to prevent any pending actions
                    # from being executed during reset
                    self._clear_control_shm()

                    # Move robot to home position BEFORE notifying policy runner
                    # This ensures that when policy runner resets, the camera images and robot
                    # states already reflect the robot at home position, not mid-movement
                    if self._go_home_with_index(home_idx):
                        logger.info("Successfully moved robot to home position")
                    else:
                        logger.error("Failed to move robot to home position")

                    # Discard every control frame written WHILE homing was running.
                    # The teleoperator sits in WAIT and keeps re-streaming the
                    # robot's published TCP target; during the (blocking) homing
                    # move that target is still the PRE-home pose, so those frames
                    # carry fresh timestamps but stale (pre-home) poses. Without a
                    # cutoff the arm consumes the latest stale frame right after
                    # homing and springs back. Clear the buffer and stamp a cutoff
                    # (+grace) so only NEW frames the teleop writes once it sees the
                    # home pose are accepted -> the arm holds at home.
                    home_end_timestamp_ns = time.time_ns()
                    self._clear_control_shm()
                    self._last_home_command_timestamp = max(
                        getattr(self, "_last_home_command_timestamp", 0),
                        home_end_timestamp_ns,
                    )
                    self._policy_control_start_timestamp = max(
                        getattr(self, "_policy_control_start_timestamp", 0),
                        home_end_timestamp_ns,
                    )

                    # NOW notify policy runner to reset after robot is at home
                    # This ensures the reset timestamp is set after robot reaches home,
                    # so new observations will show robot at home position
                    self._notify_policy_runner_reset()
                    logger.info("Successfully executed external home command")
                except Exception as e:
                    logger.error(f"Error processing home command file: {e}")
                    # Try to delete the file even if there was an error
                    try:
                        home_command_file.unlink()
                    except:
                        pass
                finally:
                    # Re-enable the command thread now that homing is finished.
                    self._home_command_active = False
        except Exception as e:
            logger.debug(f"Error checking home command: {e}")

    def _notify_policy_runner_reset(self) -> None:
        """Notify policy runner and policy connector to reset when robot is resetting."""
        try:
            from pathlib import Path

            # Create reset command file for policy runner (ActionExecutor)
            reset_command_file = Path(self._policy_runner_reset_command)
            reset_command_file.parent.mkdir(parents=True, exist_ok=True)
            reset_command_file.touch()

            # Also create reset signal file for policy connector
            # This notifies the policy connector to reset its RTC state
            reset_signal_file = Path(self._reset_signal)
            reset_signal_file.touch()

            logger.info("Notified policy runner and policy connector to reset due to robot reset")
        except Exception as e:
            logger.warning(f"Failed to notify policy runner/connector of reset: {e}")

    def _clear_control_shm(self) -> None:
        """Clear all data in robot control SHM buffers."""
        try:
            from utils.shm_utils import (
                unpack_buffer_header, pack_buffer_header, pack_frame_header,
                unpack_device_header,
                DEVICE_HEADER_SIZE, BUFFER_HEADER_SIZE, FRAME_HEADER_SIZE,
            )

            for shm_name in self._owned_control_shm_names:
                control_shm = self.control_shared_memory.get(shm_name)
                if control_shm is None:
                    continue
                try:
                    # The source contract marks these channels as owned by this
                    # robot. Reconnect only this local producer channel writable
                    # when the normal handle has been converted to read-only.
                    try:
                        buf = control_shm.buf
                        test_byte = buf[0]
                        buf[0] = test_byte
                    except (TypeError, PermissionError, AttributeError):
                        control_shm.close()
                        control_shm = connect_to_shared_memory(shm_name, read_only=False)
                        self.control_shared_memory[shm_name] = control_shm
                        self._owned_control_channels[shm_name] = control_shm
                        buf = control_shm.buf

                    # Read buffer header
                    buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE])
                    buffer_size = buffer_header['buffer_size']

                    # Read device header to get frame_size
                    device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
                    frame_size = device_header['frame_size']

                    # Clear all frames by setting timestamps to 0
                    for i in range(buffer_size):
                        frame_offset = get_control_frame_offset(i, frame_size)
                        # Set frame timestamp to 0 (invalid timestamp)
                        zero_timestamp_header = pack_frame_header(0)
                        buf[frame_offset:frame_offset+FRAME_HEADER_SIZE] = zero_timestamp_header
                        # Clear frame data (set to zeros)
                        data_start = frame_offset + FRAME_HEADER_SIZE
                        data_end = frame_offset + frame_size
                        buf[data_start:data_end] = b'\x00' * (frame_size - FRAME_HEADER_SIZE)

                    # Reset buffer header (write_index=0, current_frames_count=0)
                    reset_buffer_header = pack_buffer_header(buffer_size, 0, 0)
                    buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE] = reset_buffer_header

                    logger.info(f"Cleared control SHM buffer: {shm_name}")
                except Exception as e:
                    logger.error(f"Failed to clear control SHM {shm_name}: {e}")
            if self.use_cartesian_motion_planner:
                self._reset_planner_command_timeline()
        except Exception as e:
            logger.error(f"Error clearing control SHM: {e}")

    def _go_home_with_index(self, home_idx: int) -> bool:
        """Move robot(s) to home pose with specified preset index."""
        try:
            # Update home preset index if needed
            if self.home_presets is not None and 0 <= home_idx < len(self.home_presets):
                self.home_preset_index = home_idx
                selected_preset = self.home_presets[home_idx]

                # Update home joints/poses from preset
                # Priority: if joints are configured, use joints (will record pose from joints)
                # Otherwise, use pose directly
                if self.is_dual_arm:
                    if "joints_left" in selected_preset:
                        self.home_joints_left = selected_preset["joints_left"]
                        # Clear pose_left to force _set_home_pose to use joints and record pose
                        self.home_pose_left = None
                    if "joints_right" in selected_preset:
                        self.home_joints_right = selected_preset["joints_right"]
                        # Clear pose_right to force _set_home_pose to use joints and record pose
                        self.home_pose_right = None
                    # Only set pose if joints are not available
                    if "pose_left" in selected_preset and self.home_joints_left is None:
                        self.home_pose_left = selected_preset["pose_left"]
                    if "pose_right" in selected_preset and self.home_joints_right is None:
                        self.home_pose_right = selected_preset["pose_right"]
                else:
                    if "joints" in selected_preset:
                        self.home_joints = selected_preset["joints"]
                        # Clear pose to force _set_home_pose to use joints and record pose
                        self.home_pose = None
                    # Only set pose if joints are not available
                    if "pose" in selected_preset and self.home_joints is None:
                        self.home_pose = selected_preset["pose"]

            # Set home pose from config or current pose (same as initialization)
            # This ensures home poses are properly set, especially if using joints
            # If joints are configured, this will move to joints first and record poses
            if not self._set_home_pose():
                logger.error("Failed to set home pose before going home")
                return False

            # Call the original _go_home method
            return self._go_home()
        except Exception as e:
            logger.error(f"Failed to go home with index {home_idx}: {e}")
            return False

    def _go_home(self) -> bool:
        """Move robot(s) to home pose."""
        try:
            if self.control_type == "joints":
                logger.info("Joint control mode - skipping go_home")
                return True

            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right or not self.home_pose_left or not self.home_pose_right:
                    logger.error("Dual-arm robots or home poses not initialized")
                    return False

                # Move both arms to home poses
                # Convert back to Flexiv format [x,y,z,qw,qx,qy,qz]
                left_target = self._xyzw_pose_to_flexiv_pose(self.home_pose_left)
                right_target = self._xyzw_pose_to_flexiv_pose(self.home_pose_right)

                logger.info("Moving dual-arm robots to home poses...")
                self._send_cartesian_motion_command(self.robot_left, left_target)
                self._send_cartesian_motion_command(self.robot_right, right_target)

                # Reset gripper widths to home position (same as initialization)
                if self.utilize_gripper and self.gripper_left:
                    self._move_gripper_home_by_backend(self.gripper_left, 0, "left", wait=False)
                if self.utilize_gripper and self.gripper_right:
                    self._move_gripper_home_by_backend(self.gripper_right, 1, "right", wait=False)

                # Wait for motion to complete
                time.sleep(3.0)
                logger.info("Dual-arm robots moved to home poses")
            else:
                if not self.robot or not self.home_pose:
                    logger.error("Robot or home pose not initialized")
                    return False

                # Move single arm to home pose
                # Convert back to Flexiv format [x,y,z,qw,qx,qy,qz]
                target = self._xyzw_pose_to_flexiv_pose(self.home_pose)

                logger.info("Moving robot to home pose...")
                self._send_cartesian_motion_command(self.robot, target)

                # Reset gripper width to home position (same as initialization)
                if self.utilize_gripper and self.gripper:
                    self._move_gripper_home_by_backend(self.gripper, 0, "single", wait=False)

                # Wait for motion to complete
                time.sleep(3.0)
                logger.info("Robot moved to home pose")

            return True

        except Exception as e:
            logger.error(f"Failed to go home: {e}")
            return False

    def _read_ext_wrench(self, states) -> np.ndarray:
        """Flexiv external TCP wrench ``[Fx,Fy,Fz,Wx,Wy,Wz]`` (force N, torque Nm).

        Read from ``RobotStates.ext_wrench_in_tcp`` (TCP frame). Returns zeros if
        the RDK build does not expose the field so SHM publishing never breaks.
        """
        try:
            w = getattr(states, "ext_wrench_in_tcp", None)
            if w is None:
                return np.zeros(6, dtype=self.data_dtype)
            arr = np.asarray(w, dtype=self.data_dtype).reshape(-1)
            if arr.shape[0] >= 6:
                return arr[:6].astype(self.data_dtype)
            out = np.zeros(6, dtype=self.data_dtype)
            out[: arr.shape[0]] = arr
            return out
        except Exception:
            return np.zeros(6, dtype=self.data_dtype)

    def _read_tcp_vel(self, states) -> np.ndarray:
        """Flexiv measured TCP velocity ``[vx,vy,vz,wx,wy,wz]`` w.r.t. world frame
        (linear m/s, angular rad/s).

        Read from ``RobotStates.tcp_vel``. Returns zeros if the RDK build does not
        expose the field so SHM publishing never breaks.
        """
        try:
            v = getattr(states, "tcp_vel", None)
            if v is None:
                return np.zeros(6, dtype=self.data_dtype)
            arr = np.asarray(v, dtype=self.data_dtype).reshape(-1)
            if arr.shape[0] >= 6:
                return arr[:6].astype(self.data_dtype)
            out = np.zeros(6, dtype=self.data_dtype)
            out[: arr.shape[0]] = arr
            return out
        except Exception:
            return np.zeros(6, dtype=self.data_dtype)

    def get_robot_state(self) -> np.ndarray:
        """Return current robot state without changing the legacy base layout.

        For dual-arm state frames, the 44-value base block always contains
        measured TCP poses. Frames of at least 74 values additionally expose
        commanded TCP poses and external wrench data through
        :class:`RizonDualArmStateLayout`.
        """
        try:
            if not self.is_operational:
                return np.zeros(self.data_shape, dtype=self.data_dtype)

            # Surface contact-wrench safety events via edge detection on the
            # Flexiv operational_status enum.
            self._check_operational_status_change()

            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return np.zeros(self.data_shape, dtype=self.data_dtype)

                # Get states from both robots
                left_states = self.robot_left.states()
                right_states = self.robot_right.states()

                # Combine joint positions and velocities for both arms
                left_positions = np.array(left_states.q, dtype=self.data_dtype)
                left_velocities = np.array(left_states.dq, dtype=self.data_dtype)
                left_eef_pose = np.array(
                    left_states.tcp_pose,
                    dtype=self.data_dtype,
                )
                if self.utilize_gripper:
                    left_gripper_width_value = self._read_gripper_value_by_backend(self.gripper_left, 0, "left")
                    left_gripper_width = np.array([left_gripper_width_value], dtype=self.data_dtype)
                else:
                    left_gripper_width = np.array([0], dtype=self.data_dtype)
                right_positions = np.array(right_states.q, dtype=self.data_dtype)
                right_velocities = np.array(right_states.dq, dtype=self.data_dtype)
                right_eef_pose = np.array(
                    right_states.tcp_pose,
                    dtype=self.data_dtype,
                )
                if self.utilize_gripper:
                    right_gripper_width_value = self._read_gripper_value_by_backend(self.gripper_right, 1, "right")
                    right_gripper_width = np.array([right_gripper_width_value], dtype=self.data_dtype)
                else:
                    right_gripper_width = np.array([0], dtype=self.data_dtype)
                # Combine into single array: [left_pos, left_vel, right_pos, right_vel]
                combined_states = np.concatenate([left_positions, left_velocities, left_eef_pose, left_gripper_width,
                                                right_positions, right_velocities, right_eef_pose, right_gripper_width])

                # Collection tail (when data_shape >= 48): per-side measured
                # gripper width(m) + motor radian -> umi/*/hand_state, raw_hand_state.
                want = int(np.prod(self.data_shape)) if len(self.data_shape) else combined_states.shape[0]
                if want >= RizonDualArmStateLayout.GRIPPER_STATE_SIZE:
                    if self.utilize_gripper:
                        l_w, l_r = self._gripper_width_and_radian(self.gripper_left, float(left_gripper_width[0]), 0)
                        r_w, r_r = self._gripper_width_and_radian(self.gripper_right, float(right_gripper_width[0]), 1)
                    else:
                        l_w, l_r, r_w, r_r = 0.0, 0.0, 0.0, 0.0
                    combined_states = np.concatenate([
                        combined_states,
                        np.array([l_w, l_r, r_w, r_r], dtype=self.data_dtype),
                    ])

                # R1 tail (when data_shape >= 74): commanded TCP pose + external
                # wrench per arm. Keeping commands in the extension preserves the
                # measured-TCP semantics of the public 44-value base block.
                if want >= RizonDualArmStateLayout.ARM_DETAIL_SIZE:
                    left_cmd = self._hold_pose_by_robot.get(id(self.robot_left))
                    right_cmd = self._hold_pose_by_robot.get(id(self.robot_right))
                    left_cmd_tcp = np.array(
                        left_cmd if left_cmd is not None else left_states.tcp_pose,
                        dtype=self.data_dtype,
                    )
                    right_cmd_tcp = np.array(
                        right_cmd if right_cmd is not None else right_states.tcp_pose,
                        dtype=self.data_dtype,
                    )
                    left_wrench = self._read_ext_wrench(left_states)
                    right_wrench = self._read_ext_wrench(right_states)
                    combined_states = np.concatenate([
                        combined_states,
                        left_cmd_tcp, left_wrench,
                        right_cmd_tcp, right_wrench,
                    ])

                # R1 tail (+12, when data_shape >= 86): measured TCP velocity per arm
                # -> arm/*/tcp/vel. Layout [L_tcp_vel(6), R_tcp_vel(6)], each
                # [vx,vy,vz,wx,wy,wz] (world frame). Joint vel (dq) already in q/dq block.
                if want >= RizonDualArmStateLayout.TCP_VELOCITY_SIZE:
                    left_tcp_vel = self._read_tcp_vel(left_states)
                    right_tcp_vel = self._read_tcp_vel(right_states)
                    combined_states = np.concatenate([
                        combined_states,
                        left_tcp_vel, right_tcp_vel,
                    ])

                # R1 tail (+2, when data_shape >= 88): commanded gripper width(m) per
                # side -> hand/*/cmd (distinct from measured hand/*/state above).
                #   [L_grip_cmd_width_m, R_grip_cmd_width_m]
                if want >= RizonDualArmStateLayout.COMMAND_GRIPPER_SIZE:
                    if self.utilize_gripper:
                        l_cmd_w = float(self._last_gripper_cmd_width_m[0])
                        r_cmd_w = float(self._last_gripper_cmd_width_m[1])
                    else:
                        l_cmd_w, r_cmd_w = 0.0, 0.0
                    combined_states = np.concatenate([
                        combined_states,
                        np.array([l_cmd_w, r_cmd_w], dtype=self.data_dtype),
                    ])
            else:
                if not self.robot:
                    return np.zeros(self.data_shape, dtype=self.data_dtype)

                states = self.robot.states()
                if self.using_tcp_pose:
                    ### Using tcp pose
                    # Combine tcp pose and gripper width
                    joint_positions = np.array(states.q, dtype=self.data_dtype)
                    joint_velocities = np.array(states.dq, dtype=self.data_dtype)
                    eef_pose = np.array(
                        states.tcp_pose,
                        dtype=self.data_dtype,
                    )
                    if self.utilize_gripper:
                        gripper_width_value = self._read_gripper_value_by_backend(self.gripper, 0, "single")
                        gripper_width = np.array([gripper_width_value], dtype=self.data_dtype)
                    else:
                        gripper_width = np.array([0], dtype=self.data_dtype)

                    combined_states = np.concatenate([joint_positions,joint_velocities,eef_pose,gripper_width])
                else:
                    ### Using joint
                    # Combine joint positions and velocities
                    joint_positions = np.array(states.q, dtype=self.data_dtype)
                    joint_velocities = np.array(states.dq, dtype=self.data_dtype)

                    # Combine into single array: [pos1, pos2, ..., pos7, vel1, vel2, ..., vel7]
                    combined_states = np.concatenate([joint_positions, joint_velocities])

            return self._reshape_robot_state(combined_states)

        except Exception as e:
            logger.error(f"Failed to get robot states: {e}")
            return np.zeros(self.data_shape, dtype=self.data_dtype)

    def _reshape_robot_state(self, state: np.ndarray) -> np.ndarray:
        """Require an exact state-schema match and reshape for shared memory."""
        flat_state = np.asarray(state, dtype=self.data_dtype).reshape(-1)
        if flat_state.size != self.state_layout_size:
            raise RuntimeError(
                f"Rizon state schema produced {flat_state.size} values, expected "
                f"{self.state_layout_size} for data_shape {self.data_shape}"
            )
        return flat_state.reshape(self.data_shape)

    def _read_control_flag(self) -> float:
        """
        Read the configured dual-source selector SHM.

        This method reads from a pre-connected selector owned by another device.
        The local process never unlinks that external shared memory.

        Returns:
            0.0 if policy is in control, 1.0 if human is in control
            Returns 0.0 (policy) if SHM doesn't exist or read fails
        """
        # Use pre-connected SHM instead of connecting every iteration
        control_shm = self._control_flag_shm

        if control_shm is None:
            return 0.0  # Not connected yet or connection failed

        try:
            from utils.shm_utils import (
                DEVICE_HEADER_SIZE,
                BUFFER_HEADER_SIZE,
                FRAME_HEADER_SIZE,
                unpack_device_header,
                unpack_buffer_header,
                get_dtype,
            )

            # Read device header to get shape and dtype
            device_header_data = bytes(control_shm.buf[:DEVICE_HEADER_SIZE])
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info['frame_size']
            frame_data_size = frame_size - FRAME_HEADER_SIZE

            # Read buffer header to get write position
            buffer_header_data = bytes(control_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            buffer_info = unpack_buffer_header(buffer_header_data)
            buffer_size = buffer_info['buffer_size']
            write_index = buffer_info['write_index']

            if buffer_info['current_frames_count'] == 0:
                # No data yet - return default
                logger.info(f"No data yet - return default")
                return 0.0

            # Calculate frame offset for latest command
            latest_index = get_latest_control_frame_index(write_index, buffer_size)
            frame_offset = get_latest_control_frame_offset(write_index, buffer_size, frame_size)

            # Read frame header (timestamp)
            frame_header_data = control_shm.buf[frame_offset:frame_offset + FRAME_HEADER_SIZE]
            timestamp_ns = int.from_bytes(bytes(frame_header_data), byteorder='little')

            if timestamp_ns == 0:
                # No valid data - return default
                return 0.0

            # Read control flag value
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + frame_data_size
            numpy_dtype = get_dtype(device_info['data_dtype'])
            sliced_buf = control_shm.buf[data_start:data_end]
            # Use bytes() to create a copy, avoiding view references to SHM
            flag_value = np.frombuffer(bytes(sliced_buf), dtype=numpy_dtype)[0]

            return float(flag_value)

        except Exception as e:
            logger.debug(f"Failed to read control flag: {e}")
            return 0.0  # Default to policy mode on error

    def get_control_array(
        self,
        control_dict: Dict[str, np.ndarray],
        selector_value: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Return only the source selected by the typed control contract."""
        selected_name = self._selected_source_name(selector_value)
        if selected_name is None:
            return None
        return control_dict.get(selected_name)

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """Execute control command based on control type."""
        try:
            selector_value = (
                self._read_source_selector_value()
                if self._is_dual_source_mode()
                else None
            )
            selected_role = self._selected_source_role(selector_value)
            control_array = self.get_control_array(control_dict, selector_value)

            if control_array is None:
                return
            logger.debug(f"Rizon using {selected_role} control source")
            if not self.is_operational:
                return

            if self.control_type == "joints":
                self._execute_joint_command(control_array)
            elif self.control_type in ["eef_abs", "eef_rel"]:
                self._execute_eef_command(control_array)
            else:
                logger.error(f"Unknown control type: {self.control_type}")

        except Exception as e:
            logger.error(f"Failed to execute control command: {e}")

    def _execute_joint_command(self, control_array: np.ndarray, is_nrt: bool = True) -> None:
        """Execute joint position command in NRT/RT mode."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return

                # Ensure we have 14 joint positions for dual arm
                if len(control_array) != 14:
                    logger.warning(f"Expected 14 joint positions for dual arm, got {len(control_array)}")
                    return

                # Split into left and right arm commands
                left_pos = control_array[:7].tolist()
                right_pos = control_array[7:14].tolist()

                self._send_joint_position_command(
                    self.robot_left, left_pos, self.max_velocity[:7], self.max_acceleration[:7], is_nrt=is_nrt
                )
                self._send_joint_position_command(
                    self.robot_right, right_pos, self.max_velocity[7:14], self.max_acceleration[7:14], is_nrt=is_nrt
                )
                self._maybe_log_sparse_joint_vector(left_pos + right_pos)
            else:
                if not self.robot:
                    return

                # Ensure we have 7 joint positions
                if len(control_array) != 7:
                    logger.warning(f"Expected 7 joint positions, got {len(control_array)}")
                    return

                # Convert to list
                target_pos = control_array.tolist()
                # Send command
                self._send_joint_position_command(
                    self.robot, target_pos, self.max_velocity, self.max_acceleration, is_nrt=is_nrt
                )
                self._maybe_log_sparse_joint_vector(target_pos)

        except Exception as e:
            logger.error(f"Failed to execute joint command: {e}")

    def _execute_eef_command(self, control_array: np.ndarray, is_nrt: bool = True) -> None:
        """Execute end-effector command in NRT/RT mode."""
        try:
            if self.is_dual_arm:
                if not self.robot_left or not self.robot_right:
                    return

                # Split into left and right arm commands (pose + gripper + pose + gripper format)
                if self.utilize_gripper:
                    if len(control_array) != 16:
                        logger.warning(f"Expected 16 elements for dual arm EEF control, got {len(control_array)}")
                        return
                    left_command = control_array[:7]
                    left_gripper_width = control_array[7]
                    right_command = control_array[8:15]
                    right_gripper_width = control_array[15]
                else:
                    # Gripper disabled: accept either the native 14-wide command
                    # OR a 16-wide gripper teleop stream (Quest still sends the 2
                    # gripper slots). Ignore the gripper slots so arms keep moving
                    # even with the gripper turned off for bring-up/testing.
                    if len(control_array) == 16:
                        left_command = control_array[:7]
                        right_command = control_array[8:15]
                    elif len(control_array) == 14:
                        left_command = control_array[:7]
                        right_command = control_array[7:14]
                    else:
                        logger.warning(f"Expected 14 or 16 elements for dual arm EEF control (gripper off), got {len(control_array)}")
                        return
                left_target = self._process_eef_command(left_command, self.home_pose_left, "left")
                if left_target is None:
                    return
                right_target = self._process_eef_command(right_command, self.home_pose_right, "right")
                if right_target is None:
                    return

                if self.use_cartesian_motion_planner and left_target is not None and right_target is not None:
                    if self.current_point_time == 0.0:
                        self._planner_add_origin = time.perf_counter()
                    t_add_0 = time.perf_counter()
                    self.robot_left.add_cartesian_position_plan(self.current_point_time, left_target)
                    t_add_mid = time.perf_counter()
                    self.robot_right.add_cartesian_position_plan(self.current_point_time, right_target)
                    t_add_1 = time.perf_counter()
                    dt_left_ms = (t_add_mid - t_add_0) * 1000.0
                    dt_right_ms = (t_add_1 - t_add_mid) * 1000.0
                    dt_total_s = t_add_1 - t_add_0
                    new_tt = time.perf_counter()
                    if self.num_clock_ticks % 10 == 0:
                        logger.info(f"period: {new_tt - self.real_time}, dt_total_s:{dt_total_s})")
                    self.real_time = new_tt
                    self.num_clock_ticks += 1
                    self._maybe_log_sparse_tcp_flexiv(left_target, t_perf=self._planner_add_origin + self.current_point_time, arm="left")
                    self._maybe_log_sparse_tcp_flexiv(right_target, t_perf=self._planner_add_origin + self.current_point_time + 1e-9, arm="right")
                    self._next_planner_time_s()
                    if dt_total_s > self.delta_time:
                        plan_t = float(self.current_point_time)
                        logger.warning(
                            "add_cartesian_position_plan slow: left={:.2f}ms right={:.2f}ms total={:.2f}ms "
                            "(>{:.2f}ms = 1x command period) plan_time_s={:.4f}",
                            dt_left_ms,
                            dt_right_ms,
                            dt_total_s * 1000.0,
                            self.delta_time * 1000.0,
                            plan_t,
                        )
                else:
                    t_sparse = time.perf_counter()
                    if left_target is not None:
                        self._maybe_log_sparse_tcp_flexiv(left_target, t_perf=t_sparse, arm="left")
                        self._send_cartesian_motion_command(self.robot_left, left_target, is_nrt=is_nrt)
                    if right_target is not None:
                        self._maybe_log_sparse_tcp_flexiv(right_target, t_perf=t_sparse + 1e-9, arm="right")
                        self._send_cartesian_motion_command(self.robot_right, right_target, is_nrt=is_nrt)

                # Control grippers if enabled.
                if self.utilize_gripper and self.gripper_left:
                    scaled_left_gripper_value = self._scale_realtime_gripper_command(
                        float(left_gripper_width), 0
                    )
                    self._move_gripper_value_by_backend(self.gripper_left, scaled_left_gripper_value, 0, "left")
                if self.utilize_gripper and self.gripper_right:
                    scaled_right_gripper_value = self._scale_realtime_gripper_command(
                        float(right_gripper_width), 1
                    )
                    self._move_gripper_value_by_backend(self.gripper_right, scaled_right_gripper_value, 1, "right")
            else:
                if not self.robot:
                    return

                # Determine expected array size based on gripper usage
                expected_size = 8 if self.utilize_gripper else 7
                if len(control_array) != expected_size:
                    logger.warning(f"Expected {expected_size} elements for EEF control, got {len(control_array)}")
                    return

                # Split command and gripper width
                if self.utilize_gripper:
                    eef_command = control_array[:7]
                    gripper_width = control_array[7]
                    gripper_width_value = float(gripper_width)
                    logger.debug(f"Single arm gripper control: gripper_width={gripper_width}, control_array={control_array}")
                else:
                    eef_command = control_array

                # Process single arm command
                target = self._process_eef_command(eef_command, self.home_pose, "single")

                if target is not None:
                    self._maybe_log_sparse_tcp_flexiv(target, t_perf=time.perf_counter())
                    self._send_cartesian_motion_command(self.robot, target, is_nrt=is_nrt)

                    # Control gripper if enabled
                    if self.utilize_gripper and self.gripper:
                        scaled_gripper_width_value = self._scale_realtime_gripper_command(
                            gripper_width_value, 0
                        )
                        self._move_gripper_value_by_backend(self.gripper, scaled_gripper_width_value, 0, "single")
                    elif self.utilize_gripper and not self.gripper:
                        logger.warning("Gripper enabled but gripper object not available")

        except Exception as e:
            logger.error(f"Failed to execute EEF command: {e}")

    def _process_eef_command(self, command: np.ndarray, home_pose, arm_name: str):
        """Process end-effector command based on control type."""
        try:
            if home_pose is None:
                logger.error(f"Home pose not set for {arm_name} arm")
                return None

            # Extract position and quaternion from command [x, y, z, qx, qy, qz, qw]
            cmd_pos = command[:3]
            cmd_quat = command[3:7]
            if self.control_type == "eef_abs":
                # Absolute positioning
                target_pos = cmd_pos
                target_quat = cmd_quat
                threshold = 0.50
                if arm_name == "left":
                    crt_pos = self.robot_left.states().tcp_pose[:3]
                    # exit if np.linalg.norm(crt_left_pos - target_pos) > 0.01 or np.linalg.norm(crt_left_quat - target_quat) > 0.01
                elif arm_name == "right":
                    crt_pos = self.robot_right.states().tcp_pose[:3]
                else:
                    crt_pos = self.robot.states().tcp_pose[:3]
                delta_move = np.linalg.norm(crt_pos - target_pos)
                if delta_move > threshold:
                    logger.error(f"{arm_name} arm's move is too large {delta_move}")
                    return None

            elif self.control_type == "eef_rel":
                if self.delta_eef_motion_scaler is not None:
                    cmd_pos, cmd_quat = scalar_transform(
                        cmd_pos.copy(), cmd_quat.copy(),
                        self.delta_eef_motion_scaler['trans'],
                        self.delta_eef_motion_scaler['rot']
                    )
                cmd_rot = R.from_quat(cmd_quat)
                # Current home position and orientation
                # TODO: distinguish left robot and right robot
                crt_pose = self.robot_left.states().tcp_pose
                crt_pos = crt_pose[:3]
                crt_quat = crt_pose[3:7]
                crt_rot = R.from_quat(crt_quat)
                    # exit if np.linalg.norm(crt_left_pos - target_pos) > 0.01 or np.linalg.norm(crt_left_quat - target_quat) > 0.01
                target_pos = crt_pos + crt_rot.apply(cmd_pos.copy())
                target_rot = crt_rot * cmd_rot
                target_quat = target_rot.as_quat()

            else:
                logger.error(f"Unknown EEF control type: {self.control_type}")
                return None

            # Apply axis-aligned XYZ safety bounding box on the resolved
            # absolute target position (robot base frame). Quaternion and
            # gripper components are not affected.
            target_pos_arr = np.asarray(target_pos, dtype=float).reshape(-1)
            target_pos_arr, was_clipped = self.safety_bounds.clip_position(
                target_pos_arr, arm_name=arm_name
            )
            if was_clipped:
                self._record_safety_clip_event(arm_name)
            target_pos = target_pos_arr

            return self._xyzw_pose_to_flexiv_pose(
                [target_pos[0], target_pos[1], target_pos[2], target_quat[0], target_quat[1], target_quat[2], target_quat[3]]
            )

        except Exception as e:
            logger.error(f"Failed to process EEF command for {arm_name}: {e}")
            return None

    def _record_safety_clip_event(self, arm_name: str) -> None:
        """Hook for downstream metrics on safety clip events (rate-limited)."""
        now = time.monotonic()
        last = self._safety_clip_last_log_time.get(arm_name, 0.0)
        if now - last < 1.0:
            return
        self._safety_clip_last_log_time[arm_name] = now
        logger.warning(
            f"safety clip event recorded for arm={arm_name or 'single'} "
            f"(bounds enabled, see prior 'safety clip applied' message for details)"
        )

    def _check_operational_status_change(self) -> None:
        """Edge-detect transitions in operational_status and log them.

        Used primarily to surface wrench-limit safety events (transitions
        in/out of IN_REDUCED_STATE caused by SetMaxContactWrench). Failures
        are silently swallowed so robot polling never breaks because of
        diagnostic logging.
        """
        try:
            if self.is_dual_arm:
                handle = self.robot_left or self.robot_right
            else:
                handle = self.robot
            if handle is None:
                return
            current = handle.operational_status()
            if current == self._last_operational_status:
                return
            previous = self._last_operational_status
            self._last_operational_status = current
            current_name = getattr(current, "name", str(current))
            previous_name = (
                getattr(previous, "name", str(previous))
                if previous is not None
                else "UNKNOWN"
            )
            if str(current_name) == "IN_REDUCED_STATE":
                logger.warning(
                    f"contact wrench limit triggered: operational_status "
                    f"{previous_name} -> {current_name} "
                    f"(max_contact_wrench={self.max_contact_wrench})"
                )
            elif str(previous_name) == "IN_REDUCED_STATE":
                logger.info(
                    f"contact wrench cleared: operational_status "
                    f"{previous_name} -> {current_name}"
                )
            else:
                logger.debug(
                    f"operational_status changed: {previous_name} -> {current_name}"
                )
        except Exception as exc:
            logger.debug(f"operational_status polling skipped: {exc}")

    def start_server(self) -> bool:
        """Start the Rizon robot server."""
        try:
            # Connect to robot
            if not self._connect_robot():
                logger.error("Failed to connect to robot")
                return False

            # Switch to appropriate control mode
            if self.control_type == "joints":
                if not self._switch_to_joint_control():
                    logger.error("Failed to switch to joint control mode")
                    return False
            elif self.control_type in ["eef_abs", "eef_rel"]:
                if not self._switch_to_eef_control():
                    logger.error("Failed to switch to end-effector control mode")
                    return False

                # Set home pose for EEF control (from config or current pose)
                if not self._set_home_pose():
                    logger.error("Failed to set home pose")
                    return False

                # Always go to home position for EEF control. Even with the
                # "no-move" preset (home == current pose) this is REQUIRED: it
                # sends the first Cartesian setpoint so the robot actively HOLDS
                # in NRT_CARTESIAN_MOTION_FORCE. Skipping it leaves the arm in
                # Cartesian mode with no setpoint -> compliant / "zero-gravity"
                # float until a teleop command arrives.
                logger.info("Moving to home position for EEF control...")
                if not self._go_home():
                    logger.error("Failed to go to home position")
                    return False

            # Start base robot server
            if not self._start_dual_cartesian_planners_if_configured():
                return False
            self._start_traj_recording_threads_if_configured()
            super().start_server()
            logger.info(f"Rizon robot device {self.device_class_id} started with {self.control_type} control")
            return True

        except Exception as e:
            logger.error(f"Failed to start Rizon robot device: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the Rizon robot server."""
        try:
            self._stop_traj_recording_threads()
            self._write_dense_planner_trace_at_shutdown()
            self._stop_dual_cartesian_planners_if_configured()
            if self.utilize_gripper:
                self._shutdown_grippers()

            # Disable robot(s) if connected
            if self.is_dual_arm:
                if self.robot_left and self.is_enabled:
                    try:
                        self.robot_left.Disable()
                        logger.info("Left robot disabled")
                    except:
                        pass
                if self.robot_right and self.is_enabled:
                    try:
                        self.robot_right.Disable()
                        logger.info("Right robot disabled")
                    except:
                        pass
            else:
                if self.robot and self.is_enabled:
                    try:
                        self.robot.Disable()
                        logger.info("Robot disabled")
                    except:
                        pass

            # Stop base robot server first so the control loop no longer issues
            # gripper commands, then power off the LK grippers. This must run on
            # every normal shutdown (not only on init failure) or the grippers
            # stay powered after the program exits.
            super().stop_server()

            if self.utilize_gripper:
                try:
                    self._shutdown_grippers()
                    logger.info("LK grippers powered off (motor_off + disconnect)")
                except Exception as exc:
                    logger.warning(f"Failed to power off LK grippers: {exc}")

            self._close_traj_jsonl_sinks()
            self._maybe_write_traj_run_record_json()
            logger.info(f"Rizon robot device {self.device_class_id} stopped")

        except Exception as e:
            logger.error(f"Error stopping Rizon robot device: {e}")
            # Best-effort gripper power-off even if the shutdown path above failed.
            if getattr(self, "utilize_gripper", False):
                try:
                    self._shutdown_grippers()
                    logger.info("LK grippers powered off after stop_server error")
                except Exception as exc:
                    logger.warning(f"Failed to power off LK grippers (error path): {exc}")

    def get_status(self) -> dict:
        """Get device status information."""
        status = super().get_status()
        status.update({
            'robot_sn': self.robot_sn,
            'control_type': self.control_type,
            'is_dual_arm': self.is_dual_arm,
            'is_operational': self.is_operational,
            'is_enabled': self.is_enabled,
            'current_mode': self.current_mode,
            'gripper_backend': self.gripper_backend if self.utilize_gripper else None,
            'gripper_value_mode': self.gripper_value_mode if self.utilize_gripper else None,
            'gripper_ports': self.gripper_ports if self.utilize_gripper else None,
            'gripper_motor_ids': self.gripper_motor_ids if self.utilize_gripper else None,
            'gripper_home_ratios': self.gripper_home_ratios if self.utilize_gripper else None,
            'gripper_home_widths': self.gripper_home_widths if self.utilize_gripper else None,
            'gripper_scale_factors': self.gripper_scale_factors if self.utilize_gripper else None,
            'control_shm_connected': all(shm is not None for shm in self.control_shared_memory.values()) if self.control_shared_memory else False,
            'home_pose_set': (self.home_pose is not None) if not self.is_dual_arm else
                           (self.home_pose_left is not None and self.home_pose_right is not None),
            'impedance_parameter': self.impedance_parameter if self.control_type in ["eef_abs", "eef_rel"] else "Not applicable (joint mode)",
            'motion_limits': {
                'max_linear_vel': self.max_linear_vel,
                'max_linear_acc': self.max_linear_acc,
                'max_angular_vel': self.max_angular_vel,
                'max_angular_acc': self.max_angular_acc
            } if self.control_type in ["eef_abs", "eef_rel"] else "Not applicable (joint mode)",
            'legacy_async_mode': self.legacy_async_mode,
            'use_cartesian_motion_planner': self.use_cartesian_motion_planner,
        })
        return status


class I2RTRobot(BaseRobot):
    """
    Inherits from BaseRobot and provides I2RT robot interface for SuperInference.
    """
    def __init__(
        self,
        device_class_id=0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        control: RobotControlConfig | Mapping[str, Any],
        signal_files: Optional[Dict[str, str]] = None,
        # I2RT-specific arguments
        gripper_type: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = "yam_teaching_handle",
        mode: Literal["follower", "leader", "leader_server", "follower_server", "visualizer_local", "visualizer_remote"] = "follower",
        server_host: str = "localhost",
        server_port: int = 11333,
        can_channel: str = "can0",
        bilateral_kp: float = 0.0,
        bilateral_kd: float = 0.0,
        home_joint_pos: Optional[Union[np.ndarray, List[float]]] = None,
        left_server_host: Optional[str] = None,
        right_server_host: Optional[str] = None,
        left_server_port: Optional[int] = None,
        right_server_port: Optional[int] = None,
        left_can_channel: Optional[str] = None,
        right_can_channel: Optional[str] = None,
        server_startup_wait_s: float = 3.0,
        server_launch_python: str = "python",
        minimum_gello_script: Optional[str] = None,
        disconnection_recovery: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            control=control,
        )

        self.gripper_type = gripper_type
        self.mode = mode
        self.server_host = server_host
        self.server_port = server_port
        self.can_channel = can_channel
        self.bilateral_kp = bilateral_kp
        self.bilateral_kd = bilateral_kd

        self.home_joint_pos = None if home_joint_pos is None else np.asarray(home_joint_pos, dtype=np.float64).flatten()
        self.is_leader_role = "leader" in self.mode
        self.is_follower_role = not self.is_leader_role

        self.left_server_host = left_server_host or server_host
        self.right_server_host = right_server_host or server_host
        self.left_server_port = left_server_port if left_server_port is not None else server_port
        self.right_server_port = right_server_port if right_server_port is not None else server_port + 1
        self.left_can_channel = left_can_channel or can_channel
        self.right_can_channel = right_can_channel or can_channel
        self.server_startup_wait_s = server_startup_wait_s
        self.server_launch_python = server_launch_python
        if minimum_gello_script is None:
            from pathlib import Path
            repo_root = Path(__file__).resolve().parents[1]
            self.minimum_gello_script = str(repo_root / "third_party" / "i2rt" / "scripts" / "minimum_gello.py")
        else:
            self.minimum_gello_script = minimum_gello_script

        self._left_client = None
        self._right_client = None
        self._server_processes: List[subprocess.Popen] = []
        self._last_button_left = 0.0
        self._last_button_right = 0.0
        self._leader_flag_write_index = 0
        self._leader_flag_buffer_size = max(100, int(self.control_buffer_size))
        self._leader_flag_owner = False
        self._last_button_pressed = False
        self._control_mode = "follower_autonomous"
        self._last_control_flag = 0.0 if self._is_dual_source_mode() else 1.0
        self._load_default_bilateral_gains()

        # Disconnection recovery configuration
        self._disconnection_recovery_config = disconnection_recovery or {}
        logger.info(f"Disconnection recovery config: {self._disconnection_recovery_config}")
        self._disconnection_recovery_enabled = self._disconnection_recovery_config.get('enabled', False)
        # self._pose_diff_threshold = self._disconnection_recovery_config.get('pose_diff_threshold', 0.3)
        self._stationary_time_threshold = self._disconnection_recovery_config.get('stationary_time_threshold', 2.0)
        self._trusted_state_check_interval = self._disconnection_recovery_config.get('trusted_state_check_interval', 0.01)
        self._stuck_call_check_interval = self._disconnection_recovery_config.get('stuck_call_check_interval', 0.5)
        self._stuck_call_timeout_s = float(self._disconnection_recovery_config.get("stuck_call_timeout_s", 3.0))
        # self._motion_threshold = float(self._disconnection_recovery_config.get("motion_threshold", 0.01))
        self._home_command_active: bool = False
        self._home_command_grace_period_ns: int = int(
            float(self._disconnection_recovery_config.get("home_command_grace_period_s", 0.5)) * 1e9
        )

        # Disconnection monitoring state
        self._last_self_pose: Optional[np.ndarray] = None
        self._self_stationary_start_time: Optional[float] = None
        self._trusted_state_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._recovery_active: bool = False
        self._monitor_state_lock = threading.Lock()
        self._monitored_call_sources = (
            "get_robot_state", "execute_control_command",

            "get_info_left_leader_get_robot_state", "get_info_right_leader_get_robot_state", "get_info_left_follower_get_robot_state",
            "get_info_right_follower_get_robot_state", "get_info_left_leader_go_home", "get_info_right_leader_go_home", "get_info_left_follower_go_home",
            "get_info_right_follower_go_home", "get_info_left_leader_go_rest", "get_info_right_leader_go_rest", "get_info_left_follower_go_rest",
            "get_info_right_follower_go_rest", "get_info_left_leader_trusted_state", "get_info_right_leader_trusted_state", "get_info_left_follower_trusted_state", "get_info_right_follower_trusted_state",

            "command_joint_pos_left_get_robot_state_leader", "command_joint_pos_right_get_robot_state_leader",
            "command_joint_pos_left_smooth_joint_move", "command_joint_pos_right_smooth_joint_move",

            "command_joint_pos_left_execute_control_follower", "command_joint_pos_right_execute_control_follower",
            "command_joint_pos_left_execute_control_leader", "command_joint_pos_right_execute_control_leader",

            "command_joint_pos_left_check_and_execute_teleop_start_leader", "command_joint_pos_right_check_and_execute_teleop_start_leader",
            "command_joint_pos_left_check_and_execute_teleop_start_follower", "command_joint_pos_right_check_and_execute_teleop_start_follower",

            "command_joint_pos_left_start_server_leader", "command_joint_pos_right_start_server_leader",
            "command_joint_pos_left_start_server_follower", "command_joint_pos_right_start_server_follower",

        )
        self._monitored_call_log_interval_s = float(
            self._disconnection_recovery_config.get("monitored_call_log_interval_s", 5.0)
        )
        self._monitored_call_states: Dict[str, Dict[str, Any]] = {
            source: self._new_monitored_call_state()
            for source in self._monitored_call_sources
        }
        self._last_motion_monotonic: float = time.monotonic()

        self._monitor_disconnection_evt = threading.Event()
        self._main_control_loop_thread: Optional[threading.Thread] = None
        self._main_control_loop_generation: int = 0
        self._main_control_loop_lock = threading.Lock()

        # Episode signal tracking for button-controlled data collection
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._episode_signal_pattern = signal_files.get('episode_signal_pattern')
        self._robot_home_command_pattern = signal_files.get('robot_home_command_pattern')
        self._robot_teleop_start_command_pattern = signal_files.get('robot_teleop_start_command_pattern')
        self._robot_teleop_discard_command_pattern = signal_files.get('robot_teleop_discard_command_pattern')
        self._policy_runner_reset_command = signal_files.get('policy_runner_reset_command')
        self._reset_signal = signal_files.get('reset_signal')
        self._robot_reconnect_signal = signal_files.get('robot_reconnect')
        self._robot_halt_command_pattern = signal_files.get('robot_halt_command_pattern')
        self._robot_halt_reply_pattern = signal_files.get('robot_halt_reply_pattern')
        self._is_recording_episode = False

        # self._last_get_info_state_left = None
        # self._last_get_info_state_right = None
        self._last_trusted_get_info_state_left = None
        self._last_trusted_get_info_state_right = None
        self._trusted_state_lock = threading.Lock()
        self.state_outlier_threshold = self._disconnection_recovery_config.get('state_outlier_threshold', 0.2)
        self._robot_halt_command_active = False

        self._ensure_rpc_servers_running()
        self._connect_rpc_clients()
        logger.info(
            f"Initialized I2RTRobot_{self.device_class_id} role="
            f"{'leader' if self.is_leader_role else 'follower'} "
            f"left={self.left_server_host}:{self.left_server_port} "
            f"right={self.right_server_host}:{self.right_server_port}"
        )

    def _connect_rpc_clients(self) -> None:
        """Connect to left/right I2RT RPC servers."""
        try:
            import importlib

            portal_module = importlib.import_module("portal")
            self._left_client = portal_module.Client(f"{self.left_server_host}:{self.left_server_port}")
            self._right_client = portal_module.Client(f"{self.right_server_host}:{self.right_server_port}")
            # minimum_gello leader/follower server exposes get_info()
            self._left_client.get_info().result(timeout=2.0)
            self._right_client.get_info().result(timeout=2.0)
        except Exception as e:
            logger.error(f"Failed to connect I2RT RPC clients: {e}")
            self._left_client = None
            self._right_client = None

    def _load_default_bilateral_gains(self) -> None:
        """Load default KP/KD from bilateral_control.py (with local fallback)."""
        try:
            import importlib.util
            from pathlib import Path

            repo_root = Path(__file__).resolve().parents[1]
            bilateral_path = repo_root / "third_party" / "i2rt" / "examples" / "bilateral_control.py"
            spec = importlib.util.spec_from_file_location("i2rt_bilateral_control_defaults", bilateral_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                leader_default_kp = np.asarray(module.LEADER_DEFAULT_KP, dtype=np.float64)
                leader_default_kd = np.asarray(module.LEADER_DEFAULT_KD, dtype=np.float64)
                follower_default_kp = np.asarray(module.FOLLOWER_DEFAULT_KP, dtype=np.float64)
                follower_default_kd = np.asarray(module.FOLLOWER_DEFAULT_KD, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Failed to import bilateral defaults, using fallback: {e}")

        self._leader_default_kp = leader_default_kp
        self._leader_default_kd = leader_default_kd
        self._leader_zero_kp = np.zeros_like(leader_default_kp)
        self._leader_zero_kd = np.zeros_like(leader_default_kd)
        self._follower_default_kp = follower_default_kp
        self._follower_default_kd = follower_default_kd
        self._follower_zero_kp = np.zeros_like(follower_default_kp)
        self._follower_zero_kd = np.zeros_like(follower_default_kd)

    def _apply_bilateral_gains_for_mode(self) -> None:
        """Apply KP/KD according to current toggled control mode."""
        if self._left_client is None or self._right_client is None:
            return
        try:
            if self.is_leader_role:
                if self._control_mode == "leader_to_follower":
                    kp = self._leader_zero_kp
                    kd = self._leader_zero_kd
                else:
                    kp = self._leader_default_kp * self.bilateral_kp
                    kd = self._leader_default_kd * self.bilateral_kd
            else:
                kp = self._follower_default_kp * self.bilateral_kp
                kd = self._follower_default_kd * self.bilateral_kd

            self._left_client.update_kp_kd(kp, kd)
            self._right_client.update_kp_kd(kp, kd)


        except Exception as e:
            logger.warning(f"Failed to update KP/KD gains in mode {self._control_mode}: {e}")

    def _handle_mode_transition(self, button_pressed: bool) -> None:
        """Handle button press for mode transition or episode control."""
        if button_pressed and not self._last_button_pressed:

            if not self._is_dual_source_mode():
                # Data collection mode: toggle episode recording on button press
                self._toggle_episode_recording()
            else:
                # Dual control mode: toggle between leader and autonomous mode
                self._toggle_control_mode()
        self._last_button_pressed = button_pressed

    def _toggle_episode_recording(self) -> None:
        """Toggle episode recording state and write appropriate signal file."""
        if not self._is_recording_episode:
            # Start episode
            self._is_recording_episode = True
            self._write_episode_signal("start")
            logger.info("Episode started via button press")
        else:
            # Stop episode
            self._is_recording_episode = False
            self._write_episode_signal("end")
            logger.info("Episode stopped via button press")

    def _write_episode_signal(self, signal_type: str) -> None:
        """Write episode signal file for data saver to detect."""
        signal_path = self._episode_signal_pattern.format(signal=signal_type)
        try:
            with open(signal_path, 'w') as f:
                f.write(f"{time.time():.6f}")
            logger.info(f"Wrote episode {signal_type} signal to {signal_path}")
        except Exception as e:
            logger.error(f"Failed to write episode signal {signal_type}: {e}")

    def _toggle_control_mode(self) -> None:
        """Toggle between leader_to_follower and follower_autonomous modes."""
        self._control_mode = (
            "follower_autonomous"
            if self._control_mode == "leader_to_follower"
            else "leader_to_follower"
        )
        self._apply_bilateral_gains_for_mode()
        logger.info(f"I2RT control mode toggled to: {self._control_mode}")

    def _rpc_server_mode(self) -> str:
        return "leader_server" if self.is_leader_role else "follower_server"

    def _rpc_endpoints_ready(self) -> bool:
        """Check whether both RPC endpoints are reachable and serving."""
        try:
            import importlib

            portal_module = importlib.import_module("portal")
            left_client = portal_module.Client(f"{self.left_server_host}:{self.left_server_port}")
            right_client = portal_module.Client(f"{self.right_server_host}:{self.right_server_port}")
            left_client.get_info().result(timeout=1.0)
            right_client.get_info().result(timeout=1.0)
            return True
        except Exception:
            return False

    def _launch_rpc_server_process(self, can_channel: str, server_port: int) -> Optional[subprocess.Popen]:
        """Launch a minimum_gello RPC server process for one arm.

        Uses start_new_session=True so the server process does not receive
        SIGINT from the terminal when the user presses Ctrl+C.  This ensures
        the servers stay alive long enough for _go_rest() to send the final
        joint commands during shutdown.
        """
        cmd = [
            self.server_launch_python,
            self.minimum_gello_script,
            "--can_channel", can_channel,
            "--gripper", self.gripper_type,
            "--mode", self._rpc_server_mode(),
            "--server_port", str(server_port),
        ]
        logger.info(f"Launching I2RT RPC server: {' '.join(cmd)}")
        try:
            return subprocess.Popen(cmd, start_new_session=True)
        except Exception as e:
            logger.error(f"Failed to launch minimum_gello process ({can_channel}, {server_port}): {e}")
            return None

    def _ensure_rpc_servers_running(self) -> None:
        """
        Ensure minimum_gello server processes are running before client init.
        If endpoints are already up, skip launching.
        """
        if len(self._server_processes) == 2:
            return

        left_proc = self._launch_rpc_server_process(self.left_can_channel, self.left_server_port)
        right_proc = self._launch_rpc_server_process(self.right_can_channel, self.right_server_port)
        self._server_processes = [proc for proc in [left_proc, right_proc] if proc is not None]
        if len(self._server_processes) != 2:
            logger.error("Failed to launch both minimum_gello RPC server processes")
            return

        logger.info(f"Waiting {self.server_startup_wait_s:.1f}s for minimum_gello RPC servers")
        time.sleep(self.server_startup_wait_s)
        for _ in range(10):
            if self._rpc_endpoints_ready():
                logger.info("minimum_gello RPC servers are ready")
                return
            time.sleep(0.2)
        logger.error("minimum_gello RPC servers are not reachable after launch")

    def _fit_state_shape(self, state_array: np.ndarray) -> np.ndarray:
        """Pad/crop state to configured shape for robust SHM writes."""
        expected_size = int(np.prod(self.data_shape))
        flattened = state_array.astype(get_dtype(self.data_dtype), copy=False).reshape(-1)
        if flattened.size == expected_size:
            return flattened.reshape(self.data_shape)
        shaped = np.zeros(expected_size, dtype=get_dtype(self.data_dtype))
        copy_size = min(expected_size, flattened.size)
        shaped[:copy_size] = flattened[:copy_size]
        return shaped.reshape(self.data_shape)

    def _leader_get_info(self, client, source: str) -> tuple[np.ndarray, np.ndarray]:
        """Read leader info from RPC and normalize return format."""
        with self._track_inflight_call(source):
            result = client.get_info().result()

        if isinstance(result, (list, tuple)) and len(result) == 2:
            joint_pos = np.asarray(result[0], dtype=np.float64).reshape(-1)
            button_state = np.asarray(result[1], dtype=np.float64).reshape(-1)
            return joint_pos, button_state
        joint_pos = np.asarray(result, dtype=np.float64).reshape(-1)
        return joint_pos, np.zeros(1, dtype=np.float64)

    def _command_joint_pos(self, client, joint_pos: np.ndarray, source: str) -> None:
        """Send joint command while exposing blocking time to the stuck-call watchdog."""
        with self._track_inflight_call(source):
            if "smooth_joint_move" in source and self.is_follower_role:
                pass
            client.command_joint_pos(joint_pos)

    def _follower_get_info(self, client, source: str) -> np.ndarray:
        """Read follower joint+gripper state from RPC."""
        with self._track_inflight_call(source):
            result = client.get_info().result()

        return np.asarray(result, dtype=np.float64).reshape(-1)

    def _smooth_joint_move(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        left_current: np.ndarray,
        right_current: np.ndarray,
        steps: int = 100,
        dt: float = 0.03,
    ) -> None:
        """Linearly interpolate both arms from current to target joint positions."""
        left_target = np.asarray(left_target, dtype=np.float64).reshape(-1)
        right_target = np.asarray(right_target, dtype=np.float64).reshape(-1)
        left_current = np.asarray(left_current, dtype=np.float64).reshape(-1)
        right_current = np.asarray(right_current, dtype=np.float64).reshape(-1)
        if steps <= 0:
            self._command_joint_pos(self._left_client, left_target, "command_joint_pos_left_smooth_joint_move")
            self._command_joint_pos(self._right_client, right_target, "command_joint_pos_right_smooth_joint_move")
            return
        for i in range(steps):
            alpha = i / steps
            left_interp = left_current * (1.0 - alpha) + left_target * alpha
            right_interp = right_current * (1.0 - alpha) + right_target * alpha
            self._command_joint_pos(self._left_client, left_interp, "command_joint_pos_left_smooth_joint_move")
            self._command_joint_pos(self._right_client, right_interp, "command_joint_pos_right_smooth_joint_move")
            time.sleep(dt)

    def _create_leader_control_flag_shm(self) -> None:
        """Create a leader-owned control flag SHM for shared-control takeover."""
        if (
            not self.is_leader_role
            or not self._is_dual_source_mode()
            or not self._source_selector_owned_by_self
        ):
            return
        if self._control_flag_shm is not None:
            return
        if self._control_flag_shm_name is None:
            logger.error("Leader source-selection config has no self-owned control flag endpoint")
            return
        try:
            control_flag_shape = (1,)
            numpy_dtype = get_dtype(self.data_dtype)
            bytes_per_element = np.dtype(numpy_dtype).itemsize
            data_size = np.prod(control_flag_shape) * bytes_per_element
            frame_size = FRAME_HEADER_SIZE + data_size
            total_size = calculate_device_shm_size(self._leader_flag_buffer_size, data_size)

            try:
                existing = shm.SharedMemory(name=self._control_flag_shm_name, create=False)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"Failed to remove old leader control flag SHM: {e}")

            self._control_flag_shm = shm.SharedMemory(
                name=self._control_flag_shm_name,
                create=True,
                size=total_size,
            )
            self._leader_flag_owner = True
            self._leader_flag_write_index = 0

            device_header = pack_device_header(
                self.device_name,
                self.device_class_id,
                self.fps,
                self.data_dtype,
                control_flag_shape,
                frame_size,
                self.hardware_latency_ms,
            )
            self._control_flag_shm.buf[:DEVICE_HEADER_SIZE] = device_header
            buffer_header = pack_buffer_header(self._leader_flag_buffer_size, 0, 0)
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header
            logger.info(f"Created leader control flag SHM: {self._control_flag_shm_name}")
        except Exception as e:
            logger.error(f"Failed to create leader control flag SHM: {e}")
            self._control_flag_shm = None
            self._leader_flag_owner = False

    def _write_leader_control_flag(self, flag_value: float, timestamp_ns: int) -> None:
        """Write leader button ownership flag to SHM."""
        if (
            not self.is_leader_role
            or not self._is_dual_source_mode()
            or not self._source_selector_owned_by_self
        ):
            return
        if self._control_flag_shm is None:
            self._create_leader_control_flag_shm()
        if self._control_flag_shm is None:
            return
        try:
            device_header_data = bytes(self._control_flag_shm.buf[:DEVICE_HEADER_SIZE])
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info["frame_size"]
            frame_offset = get_control_frame_offset(self._leader_flag_write_index, frame_size)
            frame_header = pack_frame_header(timestamp_ns)
            self._control_flag_shm.buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = frame_header

            flag_bytes = np.asarray([flag_value], dtype=get_dtype(self.data_dtype)).tobytes()
            data_start = frame_offset + FRAME_HEADER_SIZE
            self._control_flag_shm.buf[data_start:data_start + len(flag_bytes)] = flag_bytes

            current_frames_count = min(self._leader_flag_write_index + 1, self._leader_flag_buffer_size)
            buffer_header = pack_buffer_header(
                self._leader_flag_buffer_size,
                current_frames_count,
                self._leader_flag_write_index,
            )
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header
            self._leader_flag_write_index = (self._leader_flag_write_index + 1) % self._leader_flag_buffer_size
        except Exception as e:
            logger.warning(f"Failed to write leader control flag: {e}")

    def _calculate_pose_difference(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """
        Calculate L2 norm difference between two joint poses.

        Args:
            pose1: First joint pose array
            pose2: Second joint pose array

        Returns:
            L2 norm of the difference
        """
        try:
            # Handle potential shape mismatches (14D with gripper vs 12D without)
            min_len = min(len(pose1), len(pose2))
            if min_len == 0:
                return float('inf')

            # Compare only joint positions (exclude gripper if present)
            # For I2RT: compare first 6 joints of each arm (12 total)
            compare_indices = list(range(6)) + list(range(7, 13))  # Skip gripper dims at [6] and [13]

            p1_joints = pose1[compare_indices] if len(pose1) >= 14 else pose1[:12]
            p2_joints = pose2[compare_indices] if len(pose2) >= 14 else pose2[:12]

            return float(np.linalg.norm(p1_joints - p2_joints))
        except Exception as e:
            logger.warning(f"Failed to calculate pose difference: {e}")
            return 0.0


    def _reconnect_rpc_clients(self) -> bool:
        """
        Attempt to reconnect RPC clients to minimum_gello servers.

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info("Attempting to reconnect I2RT RPC clients...")

        try:
            # Close existing clients
            if self._left_client is not None:
                try:
                    del self._left_client
                except Exception:
                    pass
            if self._right_client is not None:
                try:
                    del self._right_client
                except Exception:
                    pass

            self._left_client = None
            self._right_client = None

            # Wait brief delay for cleanup
            time.sleep(0.5)

            # Attempt reconnection with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    import importlib
                    portal_module = importlib.import_module("portal")

                    self._left_client = portal_module.Client(
                        f"{self.left_server_host}:{self.left_server_port}"
                    )
                    self._right_client = portal_module.Client(
                        f"{self.right_server_host}:{self.right_server_port}"
                    )

                    # Test connections
                    if self._left_client is not None:
                        self._left_client.get_info().result(timeout=2.0)
                    else:
                        logger.warning(f"Left client is None")
                    if self._right_client is not None:
                        self._right_client.get_info().result(timeout=2.0)
                    else:
                        logger.warning(f"Right client is None")
                    logger.info(f"Successfully reconnected RPC clients on attempt {attempt + 1}")
                    return True

                except Exception as e:
                    logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                    else:
                        self._left_client = None
                        self._right_client = None

            return False

        except Exception as e:
            logger.error(f"Failed to reconnect RPC clients: {e}")
            return False

    def _read_state_from_the_other_shm(self) -> Optional[np.ndarray]:
        """
        Read an explicitly configured state source from the control contract.

        Returns:
            Other robot state array or None if unavailable
        """
        try:
            selector_value = (
                self._read_source_selector_value()
                if self._is_dual_source_mode()
                else None
            )
            selected_role = self._selected_source_role(selector_value)
            other_shm_name = self._selected_source_name(selector_value)
            if (
                other_shm_name is None
                or self._source_port_for_role(selected_role) != "state"
            ):
                logger.debug("Selected source is not a configured state endpoint")
                return None

            # Use the same logic as reading control commands
            _timestamp_ns, other_state = self.read_latest_control_command(other_shm_name, min_timestamp_ns=0)

            if other_state is None:
                logger.info(f"Other robot state is None from SHM: {other_shm_name}")
                return None

            return other_state

        except Exception as e:
            logger.error(f"Failed to read other robot state from SHM: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None


    def _new_monitored_call_state(self) -> Dict[str, Any]:
        now = time.monotonic()
        return {
            "inflight_count": 0,
            "enter_monotonic": None,
            "last_exit_monotonic": now,
            "last_enter_log_monotonic": 0.0,
            "last_exit_log_monotonic": 0.0,
            "stuck_reported": False,
        }

    @contextmanager
    def _track_inflight_call(self, source: str):
        """Track a potentially blocking call for stuck-call watchdog reporting."""
        self._mark_monitored_call_enter(source)
        try:
            yield
        finally:
            self._mark_monitored_call_exit(source)

    def _mark_monitored_call_enter(self, source: str) -> None:
        """Mark a monitored call as entered."""
        now = time.monotonic()
        log_count = None
        with self._monitor_state_lock:
            state = self._monitored_call_states.setdefault(source, self._new_monitored_call_state())
            if state["inflight_count"] <= 0:
                state["enter_monotonic"] = now
            state["inflight_count"] += 1
            if now - state["last_enter_log_monotonic"] >= self._monitored_call_log_interval_s:
                state["last_enter_log_monotonic"] = now
                log_count = state["inflight_count"]
        # if log_count is not None:
        #     logger.info(
        #         f"I2RT monitored_call_enter source={source} "
        #         f"inflight_count={log_count} role={'leader' if self.is_leader_role else 'follower'}"
        #     )

    def _mark_monitored_call_exit(self, source: str) -> None:
        """Mark a monitored call as exited."""
        now = time.monotonic()
        log_count = None
        log_duration = 0.0
        with self._monitor_state_lock:
            state = self._monitored_call_states.setdefault(source, self._new_monitored_call_state())
            enter_time = state["enter_monotonic"]
            state["inflight_count"] = max(0, state["inflight_count"] - 1)
            if enter_time is not None:
                log_duration = now - enter_time
            if state["inflight_count"] == 0:
                state["enter_monotonic"] = None
                state["last_exit_monotonic"] = now
                state["stuck_reported"] = False
            if now - state["last_exit_log_monotonic"] >= self._monitored_call_log_interval_s:
                state["last_exit_log_monotonic"] = now
                log_count = state["inflight_count"]

    def _mark_state_call_enter(self) -> None:
        """Compatibility wrapper for older get_robot_state() instrumentation."""
        self._mark_monitored_call_enter("get_robot_state")

    def _mark_state_call_exit(self) -> None:
        """Compatibility wrapper for older get_robot_state() instrumentation."""
        self._mark_monitored_call_exit("get_robot_state")

    def _mark_rpc_get_info_enter(self) -> None:
        """Compatibility wrapper for older leader get_info() instrumentation."""
        self._mark_monitored_call_enter("left_leader_get_info_rpc")

    def _mark_rpc_get_info_exit(self) -> None:
        """Compatibility wrapper for older leader get_info() instrumentation."""
        self._mark_monitored_call_exit("left_leader_get_info_rpc")


    def _get_stuck_call_status(self) -> tuple[bool, str, float, float]:
        """Return current stuck-call status and durations."""
        now = time.monotonic()
        with self._monitor_state_lock:
            monitored_calls = [
                (
                    source,
                    int(state["inflight_count"]),
                    state["enter_monotonic"],
                    bool(state["stuck_reported"]),
                )
                for source, state in self._monitored_call_states.items()
            ]
            # last_motion_time = self._last_motion_monotonic

        stuck_source = ""
        inflight_duration = 0.0
        stuck_sources = []
        newly_stuck_calls = []
        for source, inflight_count, enter_time, stuck_reported in monitored_calls:
            if inflight_count <= 0 or enter_time is None:
                continue
            current_duration = now - enter_time
            inflight_duration = max(inflight_duration, current_duration)
            if current_duration >= self._stuck_call_timeout_s:
                stuck_sources.append(f"{source}({current_duration:.2f}s)")
                if not stuck_reported:
                    newly_stuck_calls.append((source, inflight_count, current_duration))

        if stuck_sources:
            stuck_source = ", ".join(stuck_sources)
        if newly_stuck_calls:
            with self._monitor_state_lock:
                for source, _inflight_count, _current_duration in newly_stuck_calls:
                    state = self._monitored_call_states.get(source)
                    if state is not None:
                        state["stuck_reported"] = True
            for source, inflight_count, current_duration in newly_stuck_calls:
                logger.warning(
                    f"I2RT monitored_call_stuck source={source} "
                    f"inflight_count={inflight_count} duration={current_duration:.3f}s "
                    f"timeout={self._stuck_call_timeout_s:.3f}s "
                    f"role={'leader' if self.is_leader_role else 'follower'}"
                )

        # stationary_duration = max(0.0, now - last_motion_time)
        # return bool(stuck_source), stuck_source, inflight_duration, stationary_duration
        return bool(stuck_source), stuck_source, inflight_duration

    def _reset_disconnection_monitor_state(self) -> None:
        """Reset monitoring state to avoid stale disconnection signals."""
        self._last_self_pose = None
        self._self_stationary_start_time = None
        now = time.monotonic()
        with self._monitor_state_lock:
            for state in self._monitored_call_states.values():
                state["inflight_count"] = 0
                state["enter_monotonic"] = None
                state["last_exit_monotonic"] = now
                state["last_enter_log_monotonic"] = 0.0
                state["last_exit_log_monotonic"] = 0.0
                state["stuck_reported"] = False
            self._last_motion_monotonic = now

    def _run_main_control_loop(self, loop_generation: int) -> None:
        """Run one generation of the main control loop."""
        logger.info(
            f"[I2RT main-loop g{loop_generation}] started "
            f"role={'leader' if self.is_leader_role else 'follower'}"
        )
        try:
            assert self.fps >= self.command_fps
            iteration = 1
            command_iteration = int(self.fps // self.command_fps)
            while self.running and loop_generation == self._main_control_loop_generation:
                start_time = time.time()

                if hasattr(self, '_check_and_consume_and_reply_robot_halt_command'):
                    if hasattr(self, '_last_robot_halt_check_time'):
                        if time.time() - self._last_robot_halt_check_time > 0.1:  # Check every 100ms
                            self._check_and_consume_and_reply_robot_halt_command()
                            self._last_robot_halt_check_time = time.time()
                    else:
                        self._last_robot_halt_check_time = time.time()
                        self._check_and_consume_and_reply_robot_halt_command()

                # Check for external home command (from GUI reset)
                # Only check periodically to avoid performance impact
                if hasattr(self, '_last_home_check_time'):
                    if time.time() - self._last_home_check_time > 0.1:  # Check every 100ms
                        self._check_and_execute_home_command()
                        self._last_home_check_time = time.time()
                else:
                    self._last_home_check_time = time.time()
                    self._check_and_execute_home_command()

                if hasattr(self, '_check_and_execute_teleop_start_command'):
                    if hasattr(self, '_last_teleop_start_check_time'):
                        if time.time() - self._last_teleop_start_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_start_command()
                            self._last_teleop_start_check_time = time.time()
                    else:
                        self._last_teleop_start_check_time = time.time()
                        self._check_and_execute_teleop_start_command()

                if hasattr(self, '_check_and_execute_teleop_discard_command'):
                    if hasattr(self, '_last_teleop_discard_check_time'):
                        if time.time() - self._last_teleop_discard_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_discard_command()
                            self._last_teleop_discard_check_time = time.time()
                    else:
                        self._last_teleop_discard_check_time = time.time()
                        self._check_and_execute_teleop_discard_command()
                self._update_source_selection_transition(context="I2RT main loop")

                control_dict = {}
                for shm_name in self.control_shm_names:
                    # Read latest control command
                    # Pass minimum timestamp to filter stale actions after control mode switch
                    # Add grace period to allow fresh policy actions to arrive
                    policy_start_ts = getattr(self, '_policy_control_start_timestamp', 0)
                    grace_period_ns = max(
                        getattr(self, '_policy_control_grace_period_ns', int(0.5 * 1e9)),
                        self._home_command_grace_period_ns,
                    )
                    last_home_command_timestamp = getattr(self, '_last_home_command_timestamp', 0)
                    min_timestamp_ns = policy_start_ts + grace_period_ns if policy_start_ts > 0 else 0
                    min_timestamp_ns = max(min_timestamp_ns, last_home_command_timestamp)
                    timestamp_ns, control_array = self.read_latest_control_command(shm_name, min_timestamp_ns)
                    control_dict[shm_name] = control_array

                if any([value is not None for value in control_dict.values()]): # Any non-None values in control_dict
                    if iteration % command_iteration == 0:
                        # Skip command execution while recovery is in progress.
                        if not self._recovery_active and not self._home_command_active and not self._robot_halt_command_active:
                            self.execute_control_command(control_dict)
                iteration = iteration % command_iteration + 1
                # Get current robot state
                state_array = self.get_robot_state()
                # Write state to shared memory
                if state_array is not None:
                    self._write_array_to_shm_with_timestamp(state_array, time.time_ns())

                # Maintain control frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info(f"Stopping {self.device_name}_{self.device_class_id} server...")
        except Exception as e:
            logger.error(f"Error in robot control loop: {e}")
        finally:
            is_latest_generation = (loop_generation == self._main_control_loop_generation)
            logger.info(
                f"[I2RT main-loop g{loop_generation}] exiting "
                f"is_latest={is_latest_generation} running={self.running}"
            )


    @track_i2rt_inflight_call("get_robot_state")
    def get_robot_state(self) -> np.ndarray:
        try:
            if self._left_client is None or self._right_client is None:
                return np.zeros(self.data_shape, dtype=get_dtype(self.data_dtype))

            timestamp_ns = time.time_ns()
            if self.is_leader_role:
                left_pos, btn_left = self._leader_get_info(self._left_client, "get_info_left_leader_get_robot_state")
                right_pos, btn_right = self._leader_get_info(self._right_client, "get_info_right_leader_get_robot_state")
                state = np.concatenate([left_pos, right_pos], axis=0)
                button_left_pressed = btn_left[0] > 0.5 and self._last_button_left < 0.5
                button_right_pressed = btn_right[0] > 0.5 and self._last_button_right < 0.5
                button_pressed = button_left_pressed or button_right_pressed
                self._handle_mode_transition(button_pressed)
                self._last_button_left = btn_left[0]
                self._last_button_right = btn_right[0]
                if button_pressed:
                    self._command_joint_pos(self._left_client, left_pos[:6], "command_joint_pos_left_get_robot_state_leader")
                    self._command_joint_pos(self._right_client, right_pos[:6], "command_joint_pos_right_get_robot_state_leader")
                # Control flag stores mode state: 1.0 means leader controls follower.
                self._write_leader_control_flag(1.0 if self._control_mode == "leader_to_follower" else 0.0, timestamp_ns)
                return self._fit_state_shape(state)

            left_pos = self._follower_get_info(self._left_client, "get_info_left_follower_get_robot_state")
            right_pos = self._follower_get_info(self._right_client, "get_info_right_follower_get_robot_state")
            state = np.concatenate([left_pos, right_pos], axis=0)
            return self._fit_state_shape(state)
        except Exception as e:
            logger.error(f"Failed to get I2RT robot state: {e}")
            return np.zeros(self.data_shape, dtype=get_dtype(self.data_dtype))

    def _check_and_consume_and_reply_robot_halt_command(self) -> None:
        """Check if the robot halt command is present and consume it."""
        from pathlib import Path
        robot_halt_command_file = Path(self._robot_halt_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id))
        if robot_halt_command_file.exists():
            logger.info(f"ROBOT HALT COMMAND FILE FOUND FOR {self.device_name}_{self.device_class_id}, SETTING ROBOT HALT COMMAND ACTIVE")
            self._robot_halt_command_active = True
            robot_halt_command_file.unlink()
            logger.info(f"Consumed robot halt command file for {self.device_name}_{self.device_class_id}")
            # reply
            robot_halt_reply_file = Path(self._robot_halt_reply_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id))
            robot_halt_reply_file.parent.mkdir(parents=True, exist_ok=True)
            robot_halt_reply_file.touch()
            logger.info(f"Replied robot halt command for {self.device_name}_{self.device_class_id}, file: {robot_halt_reply_file}")
        else:
            pass

    def _update_trusted_get_info_state(self, state_left: np.ndarray, state_right: np.ndarray) -> bool:
        def _format_float_list(values: np.ndarray) -> str:
            return "[" + ", ".join(f"{float(value):.3f}" for value in values.tolist()) + "]"

        all_none = (self._last_trusted_get_info_state_left is None and self._last_trusted_get_info_state_right is None )
        all_not_none = (self._last_trusted_get_info_state_left is not None and self._last_trusted_get_info_state_right is not None)
        if not (all_none or all_not_none):
            logger.info("both last_get_info_state_left and last_trusted_get_info_state_left and last_get_info_state_right and last_trusted_get_info_state_right must be None or not None")
            return False

        if all_none:
            with self._trusted_state_lock:
                self._last_trusted_get_info_state_left = state_left
                self._last_trusted_get_info_state_right = state_right
                return True
        else:

            left_state_diff = np.abs(state_left - self._last_trusted_get_info_state_left)
            right_state_diff = np.abs(state_right - self._last_trusted_get_info_state_right)

            # leave out the gripper state
            if len(left_state_diff) == 7:
                left_state_diff = left_state_diff[:6]
            if len(right_state_diff) == 7:
                right_state_diff = right_state_diff[:6]

            left_is_trusted = np.all(left_state_diff < self.state_outlier_threshold) # every joint and gripper is within the outlier threshold
            right_is_trusted = np.all(right_state_diff < self.state_outlier_threshold) # every joint and gripper is within the outlier threshold

            # left judgement
            if left_is_trusted:
                with self._trusted_state_lock:
                    self._last_trusted_get_info_state_left = state_left
            else:
                logger.warning(f"state_outlying! role: {'leader' if self.is_leader_role else 'follower'}, state_left: {_format_float_list(state_left)}, trusted_state_left: {_format_float_list(self._last_trusted_get_info_state_left)}, outlier_index: {np.where(left_state_diff > self.state_outlier_threshold)[0].tolist()}")

            # right judgement
            if right_is_trusted:
                with self._trusted_state_lock:
                    self._last_trusted_get_info_state_right = state_right
            else:
                logger.warning(f"state_outlying! role: {'leader' if self.is_leader_role else 'follower'}, state_right: {_format_float_list(state_right)}, trusted_state_right: {_format_float_list(self._last_trusted_get_info_state_right)}, outlier_index: {np.where(right_state_diff > self.state_outlier_threshold)[0].tolist()}")

            return left_is_trusted and right_is_trusted


    def _monitor_trusted_state(self) -> None:
        """
        Background thread that keeps robot trusted state updated.
        """
        logger.info("Trusted state monitoring thread started")

        while self._monitor_disconnection_evt.is_set():
            try:

                if self._left_client is None or self._right_client is None:
                    logger.error(f"Trusted state monitor sees invalid clients: left={self._left_client is None}, right={self._right_client is None}")
                    time.sleep(self._trusted_state_check_interval)
                    continue

                if self.is_leader_role:
                    left_pos, btn_left = self._leader_get_info(self._left_client, "get_info_left_leader_trusted_state")
                    right_pos, btn_right = self._leader_get_info(self._right_client, "get_info_right_leader_trusted_state")
                else:
                    left_pos = self._follower_get_info(self._left_client, "get_info_left_follower_trusted_state")
                    right_pos = self._follower_get_info(self._right_client, "get_info_right_follower_trusted_state")

                if not self._robot_halt_command_active:
                    self._update_trusted_get_info_state(left_pos, right_pos)
                else:
                    logger.warning(f"robot halt command is active, skipping trusted state update for {self.device_name}_{self.device_class_id}")

                time.sleep(self._trusted_state_check_interval)

            except Exception as e:
                logger.error(f"Error in disconnection monitoring: {e}")
                time.sleep(self._trusted_state_check_interval)

        logger.info("Disconnection monitoring thread stopped")

    def _monitor_stuck_calls(self) -> None:
        """Watchdog that triggers recovery when state/RPC calls are stuck."""
        logger.info("Disconnection stuck-call watchdog thread started")
        while self._monitor_disconnection_evt.is_set():
            try:
                if not self._disconnection_recovery_enabled:
                    time.sleep(self._stuck_call_check_interval)
                    continue

                if self._recovery_active:
                    time.sleep(self._stuck_call_check_interval)
                    continue

                # is_stuck, stuck_source, inflight_duration, stationary_duration = self._get_stuck_call_status()
                is_stuck, stuck_source, inflight_duration = self._get_stuck_call_status()

                if not is_stuck:
                    time.sleep(self._stuck_call_check_interval)
                    continue

                logger.error(
                    f"Disconnection confirmed by stuck-call watchdog: source={stuck_source}, "
                    f"inflight_duration={inflight_duration:.2f}s, "
                    # f"stationary_duration={stationary_duration:.2f}s, "
                    f"role={'leader' if self.is_leader_role else 'follower'}, mode={self._control_mode}"
                )
                self._execute_disconnection_recovery()
                self._reset_disconnection_monitor_state()
            except Exception as e:
                logger.error(f"Error in stuck-call watchdog: {e}")
            time.sleep(self._stuck_call_check_interval)
        logger.info("Disconnection stuck-call watchdog thread stopped")

    def _execute_disconnection_recovery(self) -> None:
        """Execute recovery sequence when disconnection is detected."""
        if self._recovery_active:
            logger.warning("Recovery already in progress, skipping duplicate recovery")
            return

        self._recovery_active = True
        logger.warning("="*60)
        logger.warning("EXECUTING DISCONNECTION RECOVERY")
        logger.warning("="*60)

        try:
            # Step 1: Discard current episode and notify all watchers
            logger.info("Step 1/5: Discarding current episode and informing Leader Robot...")
            # self._write_episode_signal("discard")
            self._signal_connector_robot_reconnect()

            # Step 2: Reconnect RPC clients
            logger.info("Step 2/5: Reconnecting RPC clients...")
            reconnect_success = self._reconnect_rpc_clients()
            if not reconnect_success:
                logger.error("Failed to reconnect RPC clients, recovery incomplete")
                return

            # The Robot should not clear shared memory during disconnection recovery, we should let the server_thread to handle this

            # Step 3: Reset control mode
            logger.info("Step 3/5: Resetting control mode...")
            if self._is_dual_source_mode():
                self._control_mode = "follower_autonomous"
            else:
                self._control_mode = "leader_to_follower"
            self._apply_bilateral_gains_for_mode()

            # Step 4: Go home position
            logger.info("Step 4/5: Moving to home position...")
            home_success = self._go_home()
            if not home_success:
                logger.error("Failed to move to home position")
            else:
                logger.info("Successfully moved to home position")

            # Step 5: Restart main control loop to bypass stuck execute_control_command.
            logger.info("Step 5/5: Restarting main control loop thread...")
            self._restart_main_control_loop(reason="disconnection_recovery")

            # Notify policy runner to reset
            self._notify_policy_runner_reset()

            logger.warning("="*60)
            logger.warning("DISCONNECTION RECOVERY COMPLETED")
            logger.warning("="*60)

        except Exception as e:
            logger.error(f"Exception during disconnection recovery: {e}")
        finally:
            self._recovery_active = False
            self._reset_disconnection_monitor_state()

    def _signal_connector_robot_reconnect(self) -> None:
        """Notify policy connector to treat current episode as discard."""
        if not self._robot_reconnect_signal:
            logger.warning("robot_reconnect signal file path is missing, skip notifying connector")
            return
        try:
            from pathlib import Path

            reconnect_signal_file = Path(self._robot_reconnect_signal)
            reconnect_signal_file.parent.mkdir(parents=True, exist_ok=True)
            reconnect_signal_file.touch()
            logger.info(
                f"Notified connector with robot reconnect signal: {reconnect_signal_file}"
            )
        except Exception as e:
            logger.warning(f"Failed to notify connector robot reconnect signal: {e}")

    @track_i2rt_inflight_call("execute_control_command")
    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        try:
            if self._left_client is None or self._right_client is None:
                return
            selector_value = (
                self._read_source_selector_value()
                if self._is_dual_source_mode()
                else None
            )
            if self.is_follower_role:
                selected_array = self._get_control_array(control_dict, selector_value)
                if selected_array is None:
                    return
                control_array = np.asarray(selected_array, dtype=np.float64).reshape(-1)

                if control_array.size != 14:
                    logger.warning(f"I2RT follower expects 14D command, got {control_array.size}")
                    return

                self._command_joint_pos(self._left_client, control_array[:7], "command_joint_pos_left_execute_control_follower")
                self._command_joint_pos(self._right_client, control_array[7:14], "command_joint_pos_right_execute_control_follower")
                return

            follower_state = self._get_control_array(control_dict, selector_value)
            if follower_state is None:
                return

            follow_array = np.asarray(follower_state, dtype=np.float64).reshape(-1)
            if follow_array.size < 14:
                logger.warning(f"I2RT leader follow expects at least 14D source, got {follow_array.size}")
                return
            # Leader arm accepts only 6-DoF command per arm, so gripper dimension is dropped.
            self._command_joint_pos(self._left_client, follow_array[:6], "command_joint_pos_left_execute_control_leader")
            self._command_joint_pos(self._right_client, follow_array[7:13], "command_joint_pos_right_execute_control_leader")

        except Exception as e:
            logger.error(f"Failed to execute I2RT control command: {e}")

    def _cleanup_control_shared_memory(self) -> None:
        """Cleanup SHM resources without unlinking externally-owned SHMs."""
        super()._cleanup_control_shared_memory()

        if self._leader_flag_owner and self._control_flag_shm is not None:
            try:
                self._control_flag_shm.close()
                self._control_flag_shm.unlink()
                logger.info(f"Cleaned up leader control flag SHM: {self._control_flag_shm_name}")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Failed to cleanup leader control flag SHM: {e}")
            finally:
                self._control_flag_shm = None
                self._leader_flag_owner = False

    def _check_and_execute_home_command(self) -> None:
        """Check for external home/teleop command files and execute if present."""
        try:
            from pathlib import Path
            home_command_file = Path(self._robot_home_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id))
            has_home_command = home_command_file.exists()
            if not has_home_command:
                return
            logger.info(f"Received home command for {self.device_name}_{self.device_class_id}")
            try:
                home_command_file.unlink()
            except Exception:
                pass

            logger.info(f"unlinking home command file for {self.device_name}_{self.device_class_id}")
            self._home_command_active = True
            self._clear_control_shm()
            self._control_mode = "follower_autonomous"
            self._apply_bilateral_gains_for_mode()
            success = self._go_home()
            home_end_timestamp_ns = time.time_ns()
            self._clear_control_shm()
            self._last_home_command_timestamp = max(
                getattr(self, '_last_home_command_timestamp', 0),
                home_end_timestamp_ns,
            )
            self._policy_control_start_timestamp = max(
                getattr(self, '_policy_control_start_timestamp', 0),
                home_end_timestamp_ns,
            )
            if success:
                self._notify_policy_runner_reset()
                logger.info(
                    "I2RT home command completed successfully; "
                    f"control commands before {home_end_timestamp_ns} are ignored, "
                    f"grace={self._home_command_grace_period_ns / 1e9:.3f}s"
                )
            else:
                logger.error("I2RT home command failed")

            logger.info(f"FINISH CHECK_AND_EXECUTE_HOME_COMMAND, HOME COMMAND ACTIVE FOR {self.device_name}_{self.device_class_id} IS SET TO FALSE")
            self._robot_halt_command_active = False

        except Exception as e:
            logger.debug(f"I2RT home command check failed: {e}")
        finally:
            self._home_command_active = False


    def _check_and_execute_teleop_start_command(self) -> None:
        """Consume teleop-start command file and switch to teleoperation mode."""
        from pathlib import Path
        teleop_start_command_file = Path(self._robot_teleop_start_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id))
        has_teleop_start_command = teleop_start_command_file.exists()
        if not has_teleop_start_command:
            return
        try:
            try:
                teleop_start_command_file.unlink()
            except Exception:
                pass

            self._control_mode = "leader_to_follower"
            self._apply_bilateral_gains_for_mode()
            curr_state = self.get_robot_state()
            if curr_state is not None:
                if self.is_leader_role:
                    self._command_joint_pos(self._left_client, curr_state[:6], "command_joint_pos_left_check_and_execute_teleop_start_leader")
                    self._command_joint_pos(self._right_client, curr_state[7:13], "command_joint_pos_right_check_and_execute_teleop_start_leader")
                else:
                    self._command_joint_pos(self._left_client, curr_state[:7], "command_joint_pos_left_check_and_execute_teleop_start_follower")
                    self._command_joint_pos(self._right_client, curr_state[7:14], "command_joint_pos_right_check_and_execute_teleop_start_follower")
                logger.info(f"Set joint pos after bilateral gains: {curr_state}")
            else:
                logger.error(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")
                raise RuntimeError(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")

            logger.info(f"Received teleop-start command for {self.device_name}_{self.device_class_id}")
        except Exception as e:
            logger.warning(f"Failed to process teleop-start command: {e}")

    def _check_and_execute_teleop_discard_command(self) -> None:
        """Consume teleop-discard command file and switch to autonomous mode."""
        from pathlib import Path
        teleop_discard_command_file = Path(self._robot_teleop_discard_command_pattern.format(device_name=self.device_name, device_class_id=self.device_class_id))
        has_teleop_discard_command = teleop_discard_command_file.exists()
        if not has_teleop_discard_command:
            return
        logger.info(f"Received teleop-discard command for {self.device_name}_{self.device_class_id}")
        try:
            self._is_recording_episode = not self._is_recording_episode
            teleop_discard_command_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to process teleop-discard command: {e}")

    def _clear_control_shm(self) -> None:
        """Clear robot-owned control SHM to drop stale commands."""
        try:
            if self.control_shared_memory is None or self.default_control_shm_name not in self.control_shared_memory:
                return
            control_shm = self.control_shared_memory.get(self.default_control_shm_name)
            if control_shm is None:
                return

            try:
                control_shm.buf[0] = control_shm.buf[0]
                writable_shm = control_shm
            except (PermissionError, AttributeError, TypeError, ValueError):
                control_shm.close()
                writable_shm = connect_to_shared_memory(self.default_control_shm_name, read_only=False)
                self.control_shared_memory[self.default_control_shm_name] = writable_shm
                self._owned_control_channels[self.default_control_shm_name] = writable_shm

            buf = writable_shm.buf
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            frame_size = device_header["frame_size"]
            buffer_size = buffer_header["buffer_size"]
            zero_ts = pack_frame_header(0)
            for i in range(buffer_size):
                frame_offset = get_control_frame_offset(i, frame_size)
                buf[frame_offset:frame_offset + FRAME_HEADER_SIZE] = zero_ts
                data_start = frame_offset + FRAME_HEADER_SIZE
                buf[data_start:frame_offset + frame_size] = b"\x00" * (frame_size - FRAME_HEADER_SIZE)
            reset_header = pack_buffer_header(buffer_size, 0, 0)
            buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = reset_header
            logger.info(f"Cleared I2RT control SHM: {self.default_control_shm_name}")
        except Exception as e:
            logger.error(f"Failed to clear I2RT control SHM: {e}")

    def _go_home(self) -> bool:
        """Move both I2RT arms to configured home joint positions."""
        logger.info(f"======== GOING HOME!!! for {self.device_name}_{self.device_class_id} =========")
        if self.home_joint_pos is None:
            logger.warning("home_joint_pos is not configured, skipping go-home")
            return True
        if self._left_client is None or self._right_client is None:
            logger.error("I2RT clients are not operational for go-home")
            return False
        try:
            home = np.asarray(self.home_joint_pos, dtype=np.float64).reshape(-1)
            if self.is_follower_role:
                if home.size != 14:
                    logger.error(f"Follower go-home expects 14 values, got {home.size}")
                    return False
                # left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_home")
                # right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_home")

                if self._last_trusted_get_info_state_left is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    left_current = self._last_trusted_get_info_state_left
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_home")

                if self._last_trusted_get_info_state_right is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    right_current = self._last_trusted_get_info_state_right
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_home")

                left_target = home[:7]
                right_target = home[7:14]
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current,
                    right_current=right_current,
                )
            else:
                if home.size == 14:
                    left_target = home[:6]
                    right_target = home[7:13]
                elif home.size == 12:
                    left_target = home[:6]
                    right_target = home[6:12]
                else:
                    logger.error(f"Leader go-home expects 12 or 14 values, got {home.size}")
                    return False

                # left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_home")
                # right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_home")

                if self._last_trusted_get_info_state_left is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    left_current = self._last_trusted_get_info_state_left
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_home")

                if self._last_trusted_get_info_state_right is not None:
                    logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state")
                    right_current = self._last_trusted_get_info_state_right
                else:
                    logger.warning(f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state")
                    right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_home")

                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current[:6],
                    right_current=right_current[:6],
                )

            logger.info(f"role: {'leader' if self.is_leader_role else 'follower'}, left_current: {left_current.tolist()}, right_current: {right_current.tolist()}")
            self._last_button_left = 0.0
            self._last_button_right = 0.0
            self._last_control_flag = 0.0 if self._is_dual_source_mode() else 1.0

            with self._trusted_state_lock:
                self._last_trusted_get_info_state_left = None
                self._last_trusted_get_info_state_right = None
                logger.info(f"reset trusted state for {self.device_name}_{self.device_class_id}")

            return True
        except Exception as e:
            logger.error(f"Failed to execute I2RT go-home: {e}")
            return False

    def _go_rest(self) -> bool:
        """Move both I2RT arms to configured rest joint positions."""
        logger.info(f"GOING REST!!! for {self.device_name}_{self.device_class_id}")
        if self._left_client is None or self._right_client is None:
            logger.error("I2RT clients are not operational for go-rest")
            return False
        try:
            rest_joint_pos = np.zeros_like(self.home_joint_pos)
            rest_joint_pos[6] = 0.99
            rest_joint_pos[13] = 0.99
            if self.is_follower_role:
                left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_rest")
                right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_rest")
                left_target = rest_joint_pos[:7]
                right_target = rest_joint_pos[7:14]
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current,
                    right_current=right_current,
                )
            else:
                left_target = rest_joint_pos[:6]
                right_target = rest_joint_pos[7:13]
                left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_rest")
                right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_rest")
                self._left_client.update_kp_kd(self._leader_default_kp, self._leader_default_kd)
                self._right_client.update_kp_kd(self._leader_default_kp, self._leader_default_kd)
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current[:6],
                    right_current=right_current[:6],
                )
            return True
        except Exception as e:
            logger.error(f"Failed to execute I2RT go-rest: {e}")
            return False

    def _read_control_flag(self) -> float:
        if self.is_leader_role:
            return 1.0 if self._control_mode == "leader_to_follower" else 0.0

        try:
            device_header_data = bytes(self._control_flag_shm.buf[:DEVICE_HEADER_SIZE])
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info["frame_size"]
            frame_data_size = frame_size - FRAME_HEADER_SIZE

            buffer_header_data = bytes(
                self._control_flag_shm.buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
            buffer_info = unpack_buffer_header(buffer_header_data)
            if buffer_info["current_frames_count"] == 0:
                return 0.0
            latest_index = get_latest_control_frame_index(
                buffer_info["write_index"], buffer_info["buffer_size"]
            )
            frame_offset = get_latest_control_frame_offset(
                buffer_info["write_index"], buffer_info["buffer_size"], frame_size
            )
            timestamp_data = self._control_flag_shm.buf[frame_offset:frame_offset + FRAME_HEADER_SIZE]
            timestamp_ns = int.from_bytes(bytes(timestamp_data), byteorder="little")
            if timestamp_ns == 0:
                return 0.0
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + frame_data_size
            numpy_dtype = get_dtype(device_info["data_dtype"])
            value = np.frombuffer(bytes(self._control_flag_shm.buf[data_start:data_end]), dtype=numpy_dtype)[0]
            return float(value)
        except Exception as e:
            logger.debug(f"Failed to read I2RT control flag: {e}")
            return 0.0

    def _get_control_array(
        self,
        control_dict: Dict[str, np.ndarray],
        selector_value: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Return the exact source endpoint selected by the control contract."""
        selected_role = self._selected_source_role(selector_value)
        selected_name = self._selected_source_name(selector_value)
        if selected_name is None:
            return None
        if self.is_leader_role and self._source_port_for_role(selected_role) != "state":
            return None
        return control_dict.get(selected_name)

    def _notify_policy_runner_reset(self) -> None:
        """Notify policy runner and policy connector to reset when robot is resetting."""
        try:
            from pathlib import Path

            # Create reset command file for policy runner (ActionExecutor)
            reset_command_file = Path(self._policy_runner_reset_command)
            reset_command_file.parent.mkdir(parents=True, exist_ok=True)
            reset_command_file.touch()

            # Also create reset signal file for policy connector
            # This notifies the policy connector to reset its RTC state
            reset_signal_file = Path(self._reset_signal)
            reset_signal_file.touch()

            logger.info("Notified policy runner and policy connector to reset due to robot reset")
        except Exception as e:
            logger.warning(f"Failed to notify policy runner/connector of reset: {e}")

    def _get_status(self) -> dict:
        return {
            "mode": self.mode,
            "is_leader_role": self.is_leader_role,
            "control_shm_names": self.control_shm_names,
            "source_selector_shm_name": self._control_flag_shm_name,
            "left_endpoint": f"{self.left_server_host}:{self.left_server_port}",
            "right_endpoint": f"{self.right_server_host}:{self.right_server_port}",
        }

    def get_status(self) -> dict:
        status = super().get_status()
        status.update(self._get_status())
        return status

    def _prepare_main_control_loop_resources(self) -> None:
        """Use the shared source-selection resource setup for I2RT."""
        self._prepare_control_loop_resources()
        if self.running:
            logger.info(f"Robot {self.device_name}_{self.device_class_id} server started successfully")

    def _force_kill_thread(self, thread: threading.Thread, reason: str) -> bool:
        """Force-stop a thread by injecting SystemExit (unsafe fallback)."""
        if not thread.is_alive():
            return True

        thread_id = thread.ident
        if thread_id is None:
            logger.error(f"Failed to force-kill thread ({reason}): thread ident is None")
            return False

        logger.warning(
            f"Force-killing thread ({reason}) using PyThreadState_SetAsyncExc. "
            f"thread={thread.name}, ident={thread_id}"
        )
        try:
            set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
            set_async_exc.argtypes = [ctypes.c_ulong, ctypes.py_object]
            set_async_exc.restype = ctypes.c_int
            affected = set_async_exc(ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit))
            if affected == 0:
                logger.error(f"Force-kill failed: no thread state found for ident={thread_id}")
                return False
            if affected > 1:
                set_async_exc(ctypes.c_ulong(thread_id), None)
                logger.error(
                    f"Force-kill aborted: multiple thread states affected ({affected}) "
                    f"for ident={thread_id}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Force-kill thread failed ({reason}): {e}")
            return False

    def _restart_main_control_loop(self, reason: str = "unspecified") -> None:
        """Start a new main control loop generation and retire old one."""
        if not self.running:
            logger.warning("Skip restarting main control loop because server is not running")
            return

        with self._main_control_loop_lock:
            old_thread = self._main_control_loop_thread
            self._main_control_loop_generation += 1
            generation = self._main_control_loop_generation
            logger.info(
                f"[I2RT main-loop] restart requested reason={reason}, "
                f"new_generation={generation}"
            )
            new_thread = threading.Thread(
                target=self._run_main_control_loop,
                args=(generation,),
                daemon=True,
                name=f"I2RTRobot_{self.device_class_id}_main_control_loop_g{generation}",
            )
            self._main_control_loop_thread = new_thread
            new_thread.start()
            logger.info(
                f"[I2RT main-loop] new thread started: name={new_thread.name}, ident={new_thread.ident}"
            )

        if old_thread is not None and old_thread.is_alive():
            logger.warning(
                "Previous main control loop thread is still alive after restart; "
                "attempting to stop it explicitly"
            )
            old_thread.join(timeout=0.2)
            if old_thread.is_alive():
                killed = self._force_kill_thread(old_thread, reason=f"restart:{reason}")
                old_thread.join(timeout=0.5)
                if old_thread.is_alive():
                    logger.error(
                        f"[I2RT main-loop] old thread still alive after force-kill "
                        f"name={old_thread.name}, ident={old_thread.ident}, killed={killed}"
                    )
                else:
                    logger.info(
                        f"[I2RT main-loop] old thread stopped after force-kill "
                        f"name={old_thread.name}, ident={old_thread.ident}, killed={killed}"
                    )
            else:
                logger.info(
                    f"[I2RT main-loop] old thread exited during graceful wait "
                    f"name={old_thread.name}, ident={old_thread.ident}"
                )

    def start_server(self) -> None:
        try:
            self._i2rt_stopped = False
            logger.info(f" === starting start_server for {self.device_name}_{self.device_class_id} ===")
            logger.info(f"{self.device_name}_{self.device_class_id} Going to home position...")
            if not self._go_home():
                logger.error(f"Failed to go to home position for {self.device_name}_{self.device_class_id}")
                raise RuntimeError(f"Failed to go to home position for {self.device_name}_{self.device_class_id}")
            self._apply_bilateral_gains_for_mode()
            curr_state = self.get_robot_state()
            if curr_state is not None:
                if self.is_leader_role:
                    self._command_joint_pos(self._left_client, curr_state[:6], "command_joint_pos_left_start_server_leader")
                    self._command_joint_pos(self._right_client, curr_state[7:13], "command_joint_pos_right_start_server_leader")
                else:
                    self._command_joint_pos(self._left_client, curr_state[:7], "command_joint_pos_left_start_server_follower")
                    self._command_joint_pos(self._right_client, curr_state[7:14], "command_joint_pos_right_start_server_follower")
                logger.info(f"Set joint pos after bilateral gains: {curr_state}")
            else:
                logger.error(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")
                raise RuntimeError(f"Failed to get robot state for {self.device_name}_{self.device_class_id}")

            # Start disconnection recovery monitoring thread (follower only)
            logger.info(f"self.is_follower_role:{self.is_follower_role} and self._disconnection_recovery_enabled:{self._disconnection_recovery_enabled} ===")
            self._reset_disconnection_monitor_state()
            self._monitor_disconnection_evt.set()
            logger.info("=== Starting disconnection recovery monitoring thread... ===")
            self._trusted_state_thread = threading.Thread(
                target=self._monitor_trusted_state,
                daemon=True,
                name=f"I2RTRobot_{self.device_class_id}_disconnection_monitor"
            )
            self._trusted_state_thread.start()
            self._watchdog_thread = threading.Thread(
                target=self._monitor_stuck_calls,
                daemon=True,
                name=f"I2RTRobot_{self.device_class_id}_stuck_watchdog"
            )
            self._watchdog_thread.start()
            logger.info("Disconnection recovery monitoring enabled")

            self._prepare_main_control_loop_resources()
            self._restart_main_control_loop(reason="start_server")

            while self.running:
                try:
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break

        except Exception as e:
            logger.error(f"Failed to start {self.device_name}_{self.device_class_id} server: {e}")
            raise e

        finally:
            # Keep shutdown semantics aligned with BaseRobot.start_server:
            # always run stop_server() when start_server exits (including Ctrl+C
            # or external shutdown that flips self.running to False).
            try:
                # IMPORTANT: We should run super().stop_server() to ensure the shared memory is cleaned up before setting the robot home in self.stop_server()
                super().stop_server()
                self.stop_server()
            except Exception as e:
                logger.error(f"Error while stopping {self.device_name}_{self.device_class_id} in start_server finally: {e}")
            logger.info(f"=== Exiting start_server for {self.device_name}_{self.device_class_id} ===")


    def stop_server(self) -> None:
        """Stop robot and terminate locally launched minimum_gello servers.

        Order: _go_rest (needs live RPC servers) -> super cleanup -> terminate servers.
        """
        logger.info(f"=== Stopping {self.device_name}_{self.device_class_id} server... in I2RTRobot self.running={self.running} ===")
        if hasattr(self, '_i2rt_stopped') and self._i2rt_stopped:
            return
        self._monitor_disconnection_evt.clear()
        # Stop disconnection recovery monitoring thread
        if self._trusted_state_thread is not None and self._trusted_state_thread.is_alive():
            logger.info("Stopping disconnection recovery monitoring thread...")
            self._trusted_state_thread.join(timeout=5.0)
            if self._trusted_state_thread.is_alive():
                logger.warning("Disconnection monitoring thread did not stop gracefully")
            self._trusted_state_thread = None
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            logger.info("Stopping disconnection stuck-call watchdog thread...")
            self._watchdog_thread.join(timeout=5.0)
            if self._watchdog_thread.is_alive():
                logger.warning("Disconnection stuck-call watchdog thread did not stop gracefully")
            self._watchdog_thread = None

        self._i2rt_stopped = True

        # stopping main control loop manually
        with self._main_control_loop_lock:
            self._main_control_loop_generation += 1
            control_loop_thread = self._main_control_loop_thread
            self._main_control_loop_thread = None
        if (
            control_loop_thread is not None
            and control_loop_thread.is_alive()
            and control_loop_thread is not threading.current_thread()
        ):
            logger.info("Stopping main control loop thread...")
            control_loop_thread.join(timeout=1.0)
            if control_loop_thread.is_alive():
                logger.warning("Main control loop thread did not stop gracefully")

        try:
            if not self._go_rest():
                logger.error(f"Failed to go to rest position for {self.device_name}_{self.device_class_id}")
        except Exception as e:
            logger.error(f"Exception during go-rest for {self.device_name}_{self.device_class_id}: {e}")
        super().stop_server()
        for proc in self._server_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception as e:
                logger.warning(f"Failed to terminate minimum_gello process: {e}")
        self._server_processes = []

def real_main() -> None:
    """Main function for RizonRobot gripper width calibration."""
    import argparse
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(description='Rizon Robot Gripper Calibration')
    parser.add_argument('--device_class_id', type=int, default=0, help='Device ID')
    parser.add_argument('--robot_sn', type=str, default="Rizon4-062931", help='Robot serial number')
    parser.add_argument('--gripper_name', type=str, default="gripper", help='Display name for the gripper')
    parser.add_argument('--tool_name', type=str, default=None, help='Optional Flexiv tool name to switch before calibration')
    parser.add_argument('--gripper_port', type=str, default="/dev/ttyUSB0", help='LK gripper serial port')
    parser.add_argument('--gripper_motor_id', type=int, default=1, help='LK gripper motor ID')
    parser.add_argument('--gripper_baudrate', type=int, default=115200, help='LK gripper baudrate')
    parser.add_argument('--gripper_speed_dps', type=float, default=180.0, help='LK gripper speed limit in deg/s')
    parser.add_argument('--gripper_home_width', type=float, default=0.085, help='Home gripper width in meters')
    parser.add_argument('--gripper_home_ratio', type=float, default=None, help='Optional home opening ratio in [0, 1], overriding gripper_home_width')
    parser.add_argument('--gripper_auto_calibrate', action='store_true', help='Enable auto calibration during init')
    parser.add_argument('--gripper_min_encoder', type=int, default=None, help='Optional minimum encoder value')
    parser.add_argument('--gripper_max_encoder', type=int, default=None, help='Optional maximum encoder value')
    parser.add_argument('--output_path', type=str, default="inference_gripper_width_calibration.json", help='Output JSON file path')

    args = parser.parse_args()

    robot = StructuredRizonRobot(
        device_class_id=args.device_class_id,
        robot_sn=args.robot_sn,
        state_feedback={
            "fps": 100.0,
            "format": {
                "shape": [22],
                "dtype": "float64",
                "type": ["j1-7", "dq1-7", "xyz", "qwqxqyqz", "gripper"],
            },
        },
        control={
            "type": "eef_abs",
            "format": {
                "shape": [8],
                "dtype": "float64",
                "type": ["xyz", "qxqyqzqw", "gripper"],
            },
        },
        gripper={
            "backend": "lktech",
            "name": args.gripper_name,
            "tool_name": args.tool_name or "",
            "port": args.gripper_port,
            "motor_id": args.gripper_motor_id,
            "baudrate": args.gripper_baudrate,
            "speed_dps": args.gripper_speed_dps,
            "home_width": args.gripper_home_width,
            "home_ratio": args.gripper_home_ratio,
            "auto_calibrate": args.gripper_auto_calibrate,
            "min_encoder": args.gripper_min_encoder,
            "max_encoder": args.gripper_max_encoder,
            "value_mode": "width_m",
            "command_value_mode": "width_m",
        },
        signal_files={
            "robot_home_command_pattern": "/tmp/{device_name}_{device_class_id}_home_command",
            "policy_runner_reset_command": "/tmp/policy_runner_reset_command",
            "reset_signal": "/tmp/policy_connector_reset_signal",
        },
    )
    assert not robot.is_dual_arm, "Dual arm robots are not supported for gripper calibration"

    calibration_data = []

    try:
        # Connect to robot and initialize gripper
        if not robot._connect_robot():
            logger.error("Failed to connect to robot")
            return

        logger.info("=" * 50)
        logger.info("Gripper Width Calibration Tool")
        logger.info("=" * 50)
        logger.info("Commands:")
        logger.info("  - Enter a width value (in meters, e.g., 0.05) to move gripper")
        logger.info("  - After gripper moves, enter the measured width")
        logger.info("  - Enter 'q' or 'exit' to quit and save results")
        logger.info("=" * 50)

        while True:
            # Get target width from user
            user_input = input("\nEnter target gripper width (m) or 'q' to quit: ").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                logger.info("Exiting calibration...")
                break

            try:
                target_width = float(user_input)
            except ValueError:
                logger.warning(f"Invalid input: {user_input}. Please enter a number.")
                continue

            # Validate width range
            if target_width < 0 or target_width > 0.1:
                logger.warning(f"Width {target_width} out of range [0, 0.1]. Please try again.")
                continue

            # Move gripper to target width
            logger.info(f"Moving gripper to width: {target_width:.4f} m")
            try:
                robot._move_gripper_value_by_backend(robot.gripper, target_width, 0, "single")
                time.sleep(3.0)  # Wait for gripper to reach position

                # Read current gripper state
                current_width = robot._read_gripper_value_by_backend(robot.gripper, 0, "single")
                logger.info(f"Gripper reported width: {current_width:.4f} m")
            except Exception as e:
                logger.error(f"Failed to move gripper: {e}")
                continue

            # Get measured width from user
            measured_input = input("Enter measured width (m) or 'skip' to skip this point: ").strip()

            if measured_input.lower() == 'skip':
                logger.info("Skipping this data point.")
                continue

            try:
                measured_width = float(measured_input)
            except ValueError:
                logger.warning(f"Invalid measured width: {measured_input}. Skipping this point.")
                continue

            # Record calibration data
            data_point = {
                "target_width": target_width,
                "reported_width": current_width,
                "measured_width": measured_width,
            }
            calibration_data.append(data_point)
            logger.info(f"Recorded: target={target_width:.4f}, reported={current_width:.4f}, measured={measured_width:.4f}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during calibration: {e}")
    finally:
        # Save calibration data to JSON file
        if calibration_data:
            result = {
                "gripper_name": args.gripper_name,
                "calibration_date": datetime.now().isoformat(),
                "data_points": calibration_data
            }

            try:
                with open(args.output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Calibration data saved to: {args.output_path}")
                logger.info(f"Total data points: {len(calibration_data)}")
            except Exception as e:
                logger.error(f"Failed to save calibration data: {e}")

            # Plot calibration curve
            try:
                import matplotlib.pyplot as plt

                target_widths = [d["target_width"] for d in calibration_data]
                measured_widths = [d["measured_width"] for d in calibration_data]

                # Sort data by target width for line plot
                sorted_indices = np.argsort(target_widths)
                target_sorted = np.array(target_widths)[sorted_indices]
                measured_sorted = np.array(measured_widths)[sorted_indices]

                plt.figure(figsize=(10, 6))

                # Plot data points
                plt.scatter(target_widths, measured_widths, c='blue', s=80, label='Data Points', zorder=3)

                # Plot connecting line
                plt.plot(target_sorted, measured_sorted, 'b-', alpha=0.5, linewidth=1.5, label='Measured')

                # Plot ideal line (y = x)
                min_val = min(min(target_widths), min(measured_widths))
                max_val = max(max(target_widths), max(measured_widths))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal (y=x)')

                # Linear fit
                if len(calibration_data) >= 2:
                    coeffs = np.polyfit(target_widths, measured_widths, 1)
                    fit_line = np.poly1d(coeffs)
                    x_fit = np.linspace(min(target_widths), max(target_widths), 100)
                    plt.plot(x_fit, fit_line(x_fit), 'g-', linewidth=2,
                            label=f'Linear Fit: y={coeffs[0]:.4f}x + {coeffs[1]:.4f}')

                plt.xlabel('Target Width (m)', fontsize=12)
                plt.ylabel('Measured Width (m)', fontsize=12)
                plt.title(f'Inference Gripper Calibration: {args.gripper_name}', fontsize=14)
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')

                # Save plot
                plot_path = args.output_path.replace('.json', '_plot.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Calibration plot saved to: {plot_path}")

                # Show plot
                plt.show()

            except Exception as e:
                logger.error(f"Failed to plot calibration data: {e}")
        else:
            logger.info("No calibration data to save.")

        # Cleanup
        try:
            robot.stop_server()
        except:
            pass

        logger.info("Calibration complete.")


class _YamPoseEMA:
    """First-order EMA low-pass on a 4x4 rigid transform (ported from R1).

    Position filtered linearly; rotation filtered along the SO(3) geodesic.
    ``tau <= 0`` disables it. Used to scrub Quest head jitter before IK.
    """

    def __init__(self, tau: float):
        self.tau = float(tau)
        self._T: Optional[np.ndarray] = None
        self._t_last: Optional[float] = None

    def reset(self) -> None:
        self._T = None
        self._t_last = None

    def update(self, T_now: np.ndarray, t_now: Optional[float] = None) -> np.ndarray:
        if self.tau <= 0.0:
            return T_now
        if t_now is None:
            t_now = time.monotonic()
        if self._T is None:
            self._T = T_now.copy()
            self._t_last = t_now
            return T_now.copy()
        dt = max(t_now - self._t_last, 1e-6)
        self._t_last = t_now
        alpha = dt / (dt + self.tau)
        p_new = (1.0 - alpha) * self._T[:3, 3] + alpha * T_now[:3, 3]
        R_prev = self._T[:3, :3]
        R_now = T_now[:3, :3]
        R_delta = R_now @ R_prev.T
        rotvec = R.from_matrix(R_delta).as_rotvec()
        R_new = R.from_rotvec(rotvec * alpha).as_matrix() @ R_prev
        T_smoothed = np.eye(4)
        T_smoothed[:3, :3] = R_new
        T_smoothed[:3, 3] = p_new
        self._T = T_smoothed
        return T_smoothed.copy()


class _YamPublisher:
    """High-rate, slew-limited republisher for YAM joint targets (ported from R1).

    The control loop writes a coarse target via :meth:`set_target`; a daemon
    thread walks the published target toward it under a per-joint velocity
    (and optional acceleration) cap, calling ``yam_robot.command_joint_pos``
    at a fixed rate. Decouples the i2rt PD from main-loop jitter.
    """

    def __init__(self, yam_robot, initial_q: np.ndarray, *,
                 hz: float = 143.0, vmax_rad_per_s: float = 2.5,
                 amax_rad_per_s2: float = 50.0):
        self._robot = yam_robot
        self._dt = 1.0 / float(hz)
        self._vmax = float(vmax_rad_per_s)
        self._step_max = self._vmax * self._dt
        self._amax = float(amax_rad_per_s2)
        self._lock = threading.Lock()
        self._target = np.asarray(initial_q, dtype=float).copy()
        self._published = self._target.copy()
        self._velocity = np.zeros_like(self._published)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="YamHeadPublisher", daemon=True)
        self._err_logged = False

    def set_target(self, q: np.ndarray) -> None:
        with self._lock:
            self._target = np.asarray(q, dtype=float).copy()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                target = self._target
                published = self._published
                velocity = self._velocity

            if self._amax > 0.0:
                err = target - published
                abs_err = np.abs(err)
                v_brake = np.sqrt(2.0 * self._amax * abs_err)
                v_des = np.sign(err) * np.minimum(self._vmax, v_brake)
                dv = self._amax * self._dt
                v_new = np.clip(v_des, velocity - dv, velocity + dv)
                v_new = np.clip(v_new, -self._vmax, self._vmax)
                new_published = published + v_new * self._dt
            else:
                delta = np.clip(target - published, -self._step_max, self._step_max)
                new_published = published + delta
                v_new = delta / self._dt

            try:
                self._robot.command_joint_pos(new_published)
            except Exception as exc:
                if not self._err_logged:
                    logger.error(f"[yam-publisher] command_joint_pos failed: {exc!r}")
                    self._err_logged = True
                break
            with self._lock:
                self._published = new_published
                self._velocity = v_new
            next_tick += self._dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()


class YamHeadRobot(BaseDevice):
    """Head-driven YAM 6-DOF arm (Quest head pose -> IK -> i2rt joints).

    Conforms R1_teleoperation's YAM head-arm loop to SuperInference: reads the
    head pose + left controller buttons from ``Quest3VuerTeleoperator``'s
    secondary head SHM (``<teleop>_<id>_head``), runs differential IK against
    i2rt's MuJoCo model, slews the joint target through a background publisher
    onto the real arm, and publishes the commanded joint state to its own
    ``<name>_<id>_data`` SHM.

    Self-collision checking (and the associated mink/extra-XML dependency) is
    intentionally dropped per project decision; IK / FK still use the MuJoCo
    model that i2rt ships with the robot (``robot.xml_path``), so no XML file
    needs to be maintained here.

    Engage: LEFT controller A button (head SHM button index 4) toggles
    WAIT <-> FULL, exactly like R1.
    """

    device_kind = DeviceKind.LOW_DIM_R

    # Head -> YAM position mapping (operator behind YAM, same direction as YAM +X).
    R_QUEST_TO_YAM = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    # Head -> YAM orientation mapping. Uses the SAME basis as R_QUEST_TO_YAM
    # (position) so head yaw maps to camera yaw in the natural left=left sense.
    # NOTE: orientation is applied as a conjugation
    # (R_ORI @ R_delta_q @ R_ORI.T); conjugation by a proper rotation cannot
    # flip the sense of exactly one axis. Relative to the previous variant
    # (R_QUEST_TO_YAM @ diag(1,-1,-1)) this flips BOTH yaw and roll while
    # keeping pitch. We pick this one because yaw (left/right) must follow the
    # head; roll is essentially unused in head teleop.
    R_QUEST_TO_YAM_ORI = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    # "Natural" EE orientation for YAM under orientation_mode == "absolute".
    R_NATURAL_YAM = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    STATE_WAIT = "WAIT"
    STATE_FULL = "FULL"

    LEFT_A_BUTTON_IDX = 4  # within head SHM left-buttons block [17:25]

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: Mapping[str, Any],
        device_name: Optional[str] = None,
        # --- head SHM source (Quest3VuerTeleoperator) ---
        teleop_device_name: str = "Quest3VuerTeleoperator",
        teleop_device_class_id: int = 0,
        # --- replay mode ---
        # When True, the head is driven from a recorded EE-pose stream instead of
        # the live Quest head pose: read a 7-wide pose7 [x,y,z,qx,qy,qz,qw] target
        # from "{teleop_device_name}_{teleop_device_class_id}_head_cmd" (written by
        # ReplayTeleoperator), run IK -> joint target, and follow it through the
        # same slew-limited publisher. No head 4x4 / clutch state machine.
        head_pose_replay: bool = False,
        # --- i2rt / YAM hardware ---
        yam_channel: str = "can0",
        yam_arm: str = "yam",
        yam_gripper: str = "no_gripper",
        yam_sim: bool = False,
        yam_ee_site: str = "grasp_site",
        yam_initial_q: str = "0,2.4,0.95,1.15,0,-0.12",
        ramp_duration_s: float = 8.0,
        home_to_zero_on_exit: bool = True,
        # After ramping home, whether to drop into gravity-comp idle (compliant,
        # hand-movable). If gravity comp can't support the arm at the home pose
        # it will SAG/fall. Default False -> keep holding the home pose stiffly
        # (PD position control) until the process exits / power off.
        gravity_comp_idle_on_exit: bool = False,
        # --- following params (R1 defaults) ---
        yam_orientation_mode: str = "relative",
        yam_pos_scale: float = 1.0,
        yam_workspace_radius: float = 0.35,
        yam_max_step: float = 0.03,
        yam_ik_max_iters: int = 30,
        yam_ik_soft_pos_tol: float = 0.02,
        yam_ik_soft_ori_tol: float = 0.10,
        yam_head_filter_tau: float = 0.06,
        # Flip head ROLL (rotation about the viewing axis) only. The head->YAM
        # orientation map is a conjugation, which cannot flip the sense of a
        # single axis; if yaw/pitch are correct but roll is mirrored, enable
        # this. ``yam_head_fwd_axis`` selects the head-local forward axis
        # (WebXR head frame: forward = -z => index 2). Default off so other
        # users / configs are unaffected.
        yam_flip_head_roll: bool = False,
        yam_head_fwd_axis: int = 2,
        # --- background publisher ---
        use_publisher: bool = True,
        yam_publisher_hz: float = 143.0,
        yam_publisher_vmax: Optional[float] = None,
        yam_publisher_amax: float = 50.0,
        # External "go-home" command file (RIGHT-B from the teleoperator): when
        # this file appears the head arm slews back to yam_initial_q and drops to
        # WAIT. Pattern MUST match the teleoperator's home_command_pattern.
        home_command_pattern: str = "/tmp/superinference_robot_{device_name}_{device_class_id}_home_command",
    ) -> None:
        state_config = StateFeedbackConfig.require(
            state_feedback,
            device_kind=DeviceKind.LOW_DIM_R,
        )
        primitive_names = tuple(
            primitive.name for primitive in state_config.format.primitives
        )
        expected_primitives = (
            "j1-6",
            "xyz",
            "qxqyqzqw",
            "xyz",
            "qxqyqzqw",
        )
        if primitive_names != expected_primitives:
            raise ValueError(
                "YamHeadRobot state_feedback.format.type must be "
                "[j1-6, xyz, qxqyqzqw, xyz, qxqyqzqw]"
            )
        super().__init__(
            device_class_id=device_class_id,
            device_name=device_name or "YamHeadRobot",
            state_feedback=state_config,
        )

        self.teleop_device_name = teleop_device_name
        self.teleop_device_class_id = teleop_device_class_id
        self._head_shm_name = f"{teleop_device_name}_{teleop_device_class_id}_head"
        self._head_shm = None
        # Replay: recorded EE-pose target channel (pose7), IK'd onto the arm.
        self.head_pose_replay = bool(head_pose_replay)
        self._head_pose_shm_name = f"{teleop_device_name}_{teleop_device_class_id}_head_cmd"
        self._head_pose_shm = None

        self.yam_channel = yam_channel
        self.yam_arm = yam_arm
        self.yam_gripper = yam_gripper
        self.yam_sim = bool(yam_sim)
        self.yam_ee_site = yam_ee_site
        self.yam_initial_q = yam_initial_q
        self.ramp_duration_s = float(ramp_duration_s)
        self.home_to_zero_on_exit = bool(home_to_zero_on_exit)
        self.gravity_comp_idle_on_exit = bool(gravity_comp_idle_on_exit)

        self.yam_orientation_mode = yam_orientation_mode
        self.yam_pos_scale = float(yam_pos_scale)
        self.yam_workspace_radius = float(yam_workspace_radius)
        self.yam_max_step = float(yam_max_step)
        self.yam_ik_max_iters = int(yam_ik_max_iters)
        self.yam_ik_soft_pos_tol = float(yam_ik_soft_pos_tol)
        self.yam_ik_soft_ori_tol = float(yam_ik_soft_ori_tol)
        self.yam_head_filter_tau = float(yam_head_filter_tau)
        self.yam_flip_head_roll = bool(yam_flip_head_roll)
        self.yam_head_fwd_axis = int(yam_head_fwd_axis)

        self.use_publisher = bool(use_publisher)
        self.yam_publisher_hz = float(yam_publisher_hz)
        self.yam_publisher_vmax = yam_publisher_vmax
        self.yam_publisher_amax = float(yam_publisher_amax)

        # External go-home command file (RIGHT-B).
        self._home_command_pattern = str(home_command_pattern)
        self._last_home_check_time = 0.0

        # Runtime handles.
        self._yam_robot = None
        self._yam_kin = None
        self._yam_model = None  # mujoco MjModel (for qpos mapping only)
        self._yam_n_arm = state_config.format.primitives[0].width
        self._yam_gripper_index = None
        self._publisher: Optional[_YamPublisher] = None
        self._head_filter = _YamPoseEMA(self.yam_head_filter_tau)

        # State machine.
        self._yam_state = self.STATE_WAIT
        self._T_head_ref: Optional[np.ndarray] = None
        self._T_ee_ref: Optional[np.ndarray] = None
        self._q_target: Optional[np.ndarray] = None
        self._ya_prev = False
        self._last_safe_log = 0.0

        self.running = False
        logger.info(
            f"YamHeadRobot initialized: state SHM={self.shared_memory_name}, "
            f"head SHM={self._head_shm_name}, channel={self.yam_channel}, "
            f"orientation_mode={self.yam_orientation_mode}, ee_site={self.yam_ee_site}"
        )

    # ------------------------------------------------------------------
    # Math helpers (ported from R1)
    # ------------------------------------------------------------------
    @staticmethod
    def _orthonormalize(R3: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(R3)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1.0
            Rn = U @ Vt
        return Rn

    @staticmethod
    def _is_identity_matrix(M: np.ndarray, tol: float = 1e-9) -> bool:
        if np.allclose(M, 0.0, atol=tol):
            return True
        return np.allclose(M, np.eye(4), atol=tol)

    def _parse_initial_q(self) -> Optional[np.ndarray]:
        spec = self.yam_initial_q
        if spec is None or spec.strip().lower() in ("none", "skip", ""):
            return None
        tokens = [t for t in spec.replace(";", ",").split(",") if t.strip()]
        vals = [float(t) for t in tokens]
        if len(vals) != self._yam_n_arm:
            logger.warning(
                f"yam_initial_q has {len(vals)} values but YAM has {self._yam_n_arm} "
                f"arm joints; ignoring ramp"
            )
            return None
        return np.array(vals, dtype=float)

    def _robot_q_to_model_qpos(self, robot_q: np.ndarray) -> np.ndarray:
        """Map robot joints to MuJoCo qpos (denormalize sliders + equality constraints)."""
        import mujoco

        model = self._yam_model
        nq = model.nq
        qpos = np.zeros(nq)
        n = min(len(robot_q), nq)
        qpos[:n] = robot_q[:n]
        for j in range(model.njnt):
            adr = model.jnt_qposadr[j]
            if adr >= n:
                continue
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = model.jnt_range[j]
                qpos[adr] = lo + qpos[adr] * (hi - lo)
        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = model.jnt_qposadr[model.eq_obj1id[i]]
            adr2 = model.jnt_qposadr[model.eq_obj2id[i]]
            coef = model.eq_data[i, :5]
            qpos[adr2] = np.polyval(coef[::-1], qpos[adr1])
        return qpos

    def _yam_ee_pose7(self, q_full: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """FK the YAM end-effector to ``[x, y, z, qx, qy, qz, qw]`` (xyzw quat).

        Used only in data-collection mode (when ``data_shape`` is large enough to
        carry the EE pose alongside joints) so the NEDF3 saver can populate
        ``umi/ego/pose`` — joint angles alone cannot fill that Cartesian topic.
        """
        try:
            if q_full is None:
                q_full = self._yam_robot.get_joint_pos()
            T = self._yam_kin.fk(self._robot_q_to_model_qpos(np.asarray(q_full)))
            pos = np.asarray(T[:3, 3], dtype=np.float64)
            quat_xyzw = R.from_matrix(np.asarray(T[:3, :3])).as_quat()  # scipy -> [x,y,z,w]
            return np.concatenate([pos, np.asarray(quat_xyzw, dtype=np.float64)])
        except Exception as exc:
            logger.debug(f"[yam] EE FK for SHM publish failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # i2rt / YAM setup + ramps
    # ------------------------------------------------------------------
    def _setup_yam(self) -> None:
        import mujoco
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType

        arm_type = ArmType.from_string_name(self.yam_arm)
        gripper_type = GripperType.from_string_name(self.yam_gripper)
        logger.info(
            f"[yam] connecting arm={arm_type} gripper={gripper_type} "
            f"channel={self.yam_channel} sim={self.yam_sim}"
        )
        robot = get_yam_robot(
            channel=self.yam_channel,
            arm_type=arm_type,
            gripper_type=gripper_type,
            sim=self.yam_sim,
        )
        info = robot.get_robot_info()
        gripper_index = info.get("gripper_index")
        n_arm = gripper_index if gripper_index is not None else robot.num_dofs()

        xml_path = robot.xml_path
        logger.info(f"[yam] MuJoCo XML: {xml_path}")
        kin = Kinematics(xml_path, self.yam_ee_site)
        model = mujoco.MjModel.from_xml_path(xml_path)

        self._yam_robot = robot
        self._yam_kin = kin
        self._yam_model = model
        self._yam_n_arm = n_arm
        self._yam_gripper_index = gripper_index

        home_arm_q = self._parse_initial_q()
        if home_arm_q is not None:
            self._ramp_to_q(home_arm_q)

        q_init = robot.get_joint_pos().copy()
        self._q_target = q_init.copy()
        T_ee_now = kin.fk(self._robot_q_to_model_qpos(q_init))
        logger.info(f"[yam] ready  EE pos: {T_ee_now[:3, 3]}")

    def _ramp_to_q(self, target_arm_q: np.ndarray, hz: float = 143.0) -> None:
        robot = self._yam_robot
        if hasattr(robot, "disable_gravity_comp"):
            robot.disable_gravity_comp()
        q_start_full = robot.get_joint_pos().copy()
        q_goal_full = q_start_full.copy()
        q_goal_full[: self._yam_n_arm] = np.asarray(target_arm_q, dtype=float)
        logger.info(
            f"[yam] ramp -> arm={q_goal_full[: self._yam_n_arm].tolist()} "
            f"in {self.ramp_duration_s:.1f}s @ {hz:.0f}Hz"
        )
        dt = 1.0 / float(hz)
        steps = max(1, int(round(self.ramp_duration_s * hz)))
        next_tick = time.monotonic()
        for i in range(1, steps + 1):
            if not self.running and i > 1:
                break
            alpha = i / steps
            q = (1.0 - alpha) * q_start_full + alpha * q_goal_full
            robot.command_joint_pos(q)
            next_tick += dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    # ------------------------------------------------------------------
    # Head SHM reading
    # ------------------------------------------------------------------
    def _connect_head_shm(self) -> None:
        if self._head_shm is not None:
            return
        try:
            self._head_shm = connect_to_shared_memory(
                self._head_shm_name, read_only=True, log_missing=False
            )
        except FileNotFoundError:
            logger.warning(f"Head SHM not found: {self._head_shm_name}")
        except Exception as exc:
            logger.warning(f"Failed to connect head SHM: {exc}")

    def _read_head_data(self) -> Optional[np.ndarray]:
        self._connect_head_shm()
        if self._head_shm is None:
            return None
        buf = self._head_shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )
        if buffer_header["current_frames_count"] == 0:
            return None
        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        dtype = get_dtype(device_header["data_dtype"], np.float64)
        arr = np.frombuffer(buf[data_start : data_start + data_size], dtype=dtype).reshape(-1)
        return arr.copy()

    def _connect_head_pose_shm(self) -> None:
        if self._head_pose_shm is not None:
            return
        try:
            self._head_pose_shm = connect_to_shared_memory(
                self._head_pose_shm_name, read_only=True, log_missing=False
            )
        except FileNotFoundError:
            logger.warning(f"Head pose SHM not found: {self._head_pose_shm_name}")
        except Exception as exc:
            logger.warning(f"Failed to connect head pose SHM: {exc}")

    def _read_head_pose_data(self) -> Optional[np.ndarray]:
        self._connect_head_pose_shm()
        if self._head_pose_shm is None:
            return None
        buf = self._head_pose_shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )
        if buffer_header["current_frames_count"] == 0:
            return None
        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        dtype = get_dtype(device_header["data_dtype"], np.float64)
        arr = np.frombuffer(buf[data_start : data_start + data_size], dtype=dtype).reshape(-1)
        return arr.copy()

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------
    def start_server(self) -> None:
        if self.running:
            logger.info(f"YamHeadRobot {self.device_class_id} already running")
            return
        logger.info(f"Starting YamHeadRobot {self.device_class_id} server...")
        self.running = True

        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        self._setup_yam()

        # Wait for the command channel (the teleoperator / replay producer must
        # be up): the live 26-wide head SHM, or the 7-wide replay pose SHM.
        logger.disable("utils.shm_utils")
        if self.head_pose_replay:
            while self.running and self._head_pose_shm is None:
                self._connect_head_pose_shm()
                if self._head_pose_shm is None:
                    time.sleep(0.1)
            logger.enable("utils.shm_utils")
            logger.info(f"Head pose (replay) SHM connected: {self._head_pose_shm_name}")
        else:
            while self.running and self._head_shm is None:
                self._connect_head_shm()
                if self._head_shm is None:
                    time.sleep(0.1)
            logger.enable("utils.shm_utils")
            logger.info(f"Head SHM connected: {self._head_shm_name}")

        # Background slew-limited publisher.
        if self.use_publisher:
            vmax = self.yam_publisher_vmax
            if vmax is None:
                vmax = self.yam_max_step * self.fps
            self._publisher = _YamPublisher(
                self._yam_robot,
                self._q_target,
                hz=self.yam_publisher_hz,
                vmax_rad_per_s=vmax,
                amax_rad_per_s2=self.yam_publisher_amax,
            )
            self._publisher.start()
            logger.info(
                f"[yam-publisher] running @ {self.yam_publisher_hz:.0f} Hz, "
                f"vmax={vmax:.2f} rad/s, amax={self.yam_publisher_amax:.0f} rad/s^2"
            )

        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        while self.running:
            try:
                if self.head_pose_replay:
                    self._step_replay()
                else:
                    # Poll the RIGHT-B go-home command file (~10 Hz).
                    if time.time() - self._last_home_check_time > 0.1:
                        self._check_and_execute_home_command()
                        self._last_home_check_time = time.time()
                    self._step()
                timestamp_ns = time.time_ns()
                if self._q_target is not None:
                    dtype = get_dtype(self.data_dtype)
                    want = int(np.prod(self.data_shape)) if len(self.data_shape) else self._yam_n_arm
                    if want >= self._yam_n_arm + 7:
                        # Collection mode: publish joints + EE pose
                        # ([q..., x, y, z, qx, qy, qz, qw]) so NEDF3 can fill both
                        # umi/ego/hand_state (joints) and umi/ego/pose (Cartesian).
                        # POSE = CONTROL TARGET: FK of the commanded joint target
                        # (self._q_target), not measured FK -> cleaner/less noisy
                        # signal (matches the Flexiv "store commanded TCP" change).
                        try:
                            q_full = self._yam_robot.get_joint_pos()
                        except Exception:
                            q_full = None
                        if q_full is not None:
                            q_state = np.asarray(q_full[: self._yam_n_arm], dtype=dtype)
                        else:
                            q_state = np.asarray(self._q_target[: self._yam_n_arm], dtype=dtype)
                        ee = self._yam_ee_pose7(self._q_target)
                        if ee is None:
                            ee = np.zeros(7, dtype=dtype)
                        state = np.concatenate([q_state, ee.astype(dtype)])
                        # R1 schema extension: also publish the MEASURED-FK EE pose
                        # (FK of the measured joints) so the saver can fill head
                        # tcp/state SEPARATELY from tcp/cmd (commanded FK above).
                        # Appended only when data_shape asks (>= n_arm + 14, i.e.
                        # [20]); the [13] collection layout stays untouched.
                        if want >= self._yam_n_arm + 14:
                            ee_meas = (
                                self._yam_ee_pose7(q_full) if q_full is not None else None
                            )
                            if ee_meas is None:
                                ee_meas = ee
                            state = np.concatenate([state, np.asarray(ee_meas, dtype=dtype)])
                    else:
                        state = np.asarray(self._q_target[: self._yam_n_arm], dtype=dtype)
                    self._write_array_to_shm_with_timestamp(state, timestamp_ns)
                wait_for_next_iteration()
            except Exception as exc:
                logger.error(f"Error in YamHeadRobot loop: {exc}")
                break

    def _set_yam_target(self, q_target: np.ndarray) -> None:
        if self._publisher is not None:
            self._publisher.set_target(q_target)
        else:
            self._yam_robot.command_joint_pos(q_target)

    def _check_and_execute_home_command(self) -> None:
        """If the go-home command file exists, slew to yam_initial_q and WAIT.

        Triggered by the teleoperator's RIGHT-B. Uses the same slew-limited
        publisher as normal control, so the head arm moves there smoothly.
        """
        try:
            from pathlib import Path

            home_file = Path(
                self._home_command_pattern.format(
                    device_name=self.device_name, device_class_id=self.device_class_id
                )
            )
            if not home_file.exists():
                return
            try:
                home_file.unlink()
            except Exception:
                pass

            init_q = self._parse_initial_q()
            if self._q_target is not None:
                q_home = np.asarray(self._q_target, dtype=float).copy()
            else:
                q_home = self._yam_robot.get_joint_pos().copy()
            if init_q is not None:
                q_home[: self._yam_n_arm] = init_q[: self._yam_n_arm]

            # Drop to WAIT (stop following the head) and slew to the home q.
            self._yam_state = self.STATE_WAIT
            self._T_head_ref = None
            self._T_ee_ref = None
            self._head_filter.reset()
            self._q_target = q_home
            self._set_yam_target(self._q_target)
            logger.info(f"[home] YAM go-home -> yam_initial_q, state=WAIT ({home_file.name})")
        except Exception as exc:
            logger.debug(f"YAM home command check failed: {exc}")

    def _step(self) -> None:
        now = time.monotonic()
        head = self._read_head_data()
        if head is None or head.shape[0] < 26:
            return

        head_mat = head[0:16].reshape(4, 4)
        head_ok = head[16] > 0.5 and not self._is_identity_matrix(head_mat)
        left_visible = head[25] > 0.5
        left_buttons = head[17:25]
        ya = (
            self.LEFT_A_BUTTON_IDX < len(left_buttons)
            and float(left_buttons[self.LEFT_A_BUTTON_IDX]) > 0.5
        )
        ya_edge = ya and not self._ya_prev
        self._ya_prev = ya

        # LEFT A toggles WAIT <-> FULL.
        if ya_edge:
            if self._yam_state == self.STATE_WAIT:
                if not head_ok:
                    logger.warning("[state] LEFT A pressed but no head pose yet — Enter VR first")
                elif not left_visible:
                    logger.warning("[state] LEFT A pressed but LEFT controller not visible")
                else:
                    self._T_head_ref = head_mat.copy()
                    self._T_head_ref[:3, :3] = self._orthonormalize(self._T_head_ref[:3, :3])
                    self._head_filter.reset()
                    # Seed the engage reference from the last COMMANDED joint target,
                    # not the measured pose. Under PD control the arm settles a little
                    # below the command by the gravity-induced steady-state droop
                    # (measured ~= command - e); seeding from the measured pose would
                    # snap the command down by e on every engage, and combined with the
                    # disengage path below this ratchets the hold pose lower each cycle.
                    if self._q_target is None:
                        self._q_target = self._yam_robot.get_joint_pos().copy()
                    q_ref = np.asarray(self._q_target, dtype=float).copy()
                    self._T_ee_ref = self._yam_kin.fk(self._robot_q_to_model_qpos(q_ref))
                    self._q_target = q_ref
                    self._set_yam_target(self._q_target)
                    self._yam_state = self.STATE_FULL
                    logger.info(
                        f"[state] YAM -> FULL (mode={self.yam_orientation_mode}; "
                        f"head ref pos {self._T_head_ref[:3, 3]})"
                    )
            else:
                self._yam_state = self.STATE_WAIT
                self._T_head_ref = None
                self._T_ee_ref = None
                self._head_filter.reset()
                # Keep holding the last COMMANDED target instead of re-reading the
                # measured (slightly drooped) joints. Snapping the command to the
                # measured pose here would lower the hold pose by the PD steady-state
                # droop on every lock, so the arm would sag a bit each time.
                if self._q_target is None:
                    self._q_target = self._yam_robot.get_joint_pos().copy()
                self._set_yam_target(self._q_target)
                logger.info("[state] YAM -> WAIT (LEFT A toggle disengage)")

        if self._yam_state != self.STATE_FULL or self._T_head_ref is None:
            return

        # Smoothed head pose -> target EE pose.
        head_for_yam = self._head_filter.update(head_mat, t_now=now)
        R_head_now = self._orthonormalize(head_for_yam[:3, :3])
        p_head_now = head_for_yam[:3, 3]
        R_head_ref = self._T_head_ref[:3, :3]
        p_head_ref = self._T_head_ref[:3, 3]

        if self.yam_orientation_mode == "absolute":
            R_ee_cmd = (
                self.R_QUEST_TO_YAM_ORI @ R_head_now @ self.R_QUEST_TO_YAM_ORI.T @ self.R_NATURAL_YAM
            )
        else:
            R_delta_q = R_head_now @ R_head_ref.T
            if self.yam_flip_head_roll:
                # Conjugation can't flip a single axis, so flip ROLL here:
                # express the delta in the reference head-local frame, negate the
                # rotation-vector component about the head forward axis, map back.
                R_local = R_head_ref.T @ R_head_now
                rv = R.from_matrix(R_local).as_rotvec()
                rv[self.yam_head_fwd_axis] *= -1.0
                R_local = R.from_rotvec(rv).as_matrix()
                R_delta_q = R_head_ref @ R_local @ R_head_ref.T
            R_delta_y = self.R_QUEST_TO_YAM_ORI @ R_delta_q @ self.R_QUEST_TO_YAM_ORI.T
            R_ee_cmd = R_delta_y @ self._T_ee_ref[:3, :3]

        p_delta_q = p_head_now - p_head_ref
        p_delta_y = self.R_QUEST_TO_YAM @ p_delta_q
        p_cmd = self._T_ee_ref[:3, 3] + self.yam_pos_scale * p_delta_y

        delta_p = p_cmd - self._T_ee_ref[:3, 3]
        r = float(np.linalg.norm(delta_p))
        if r > self.yam_workspace_radius:
            p_cmd = self._T_ee_ref[:3, 3] + delta_p * (self.yam_workspace_radius / r)

        T_yam_target = np.eye(4)
        T_yam_target[:3, :3] = R_ee_cmd
        T_yam_target[:3, 3] = p_cmd

        # Warm-start IK from last commanded joints (no CAN read in the loop).
        q_cur = self._q_target
        init_q = self._robot_q_to_model_qpos(q_cur)
        success, q_ik = self._yam_kin.ik(
            T_yam_target,
            self.yam_ee_site,
            init_q=init_q,
            pos_threshold=1e-3,
            ori_threshold=1e-2,
            max_iters=self.yam_ik_max_iters,
        )

        # Best-effort: accept a partial solution within soft tolerances.
        if not success and self.yam_ik_soft_pos_tol > 0.0 and q_ik is not None:
            achieved = self._yam_kin.fk(q_ik)
            pos_err = float(np.linalg.norm(achieved[:3, 3] - T_yam_target[:3, 3]))
            R_err = achieved[:3, :3] @ T_yam_target[:3, :3].T
            ori_err = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
            if pos_err <= self.yam_ik_soft_pos_tol and ori_err <= self.yam_ik_soft_ori_tol:
                success = True

        if not success:
            if now - self._last_safe_log > 1.0:
                logger.warning("[yam-ik] not converged & outside soft tol — holding")
                self._last_safe_log = now
            return

        # Per-loop joint step clamp; preserve the gripper slot.
        q_candidate = q_cur.copy()
        q_candidate[: self._yam_n_arm] = q_ik[: self._yam_n_arm]
        delta_q = q_candidate[: self._yam_n_arm] - self._q_target[: self._yam_n_arm]
        delta_q = np.clip(delta_q, -self.yam_max_step, self.yam_max_step)
        q_step = self._q_target.copy()
        q_step[: self._yam_n_arm] = self._q_target[: self._yam_n_arm] + delta_q
        if self._yam_gripper_index is not None and len(q_step) > self._yam_gripper_index:
            q_step[self._yam_gripper_index] = q_cur[self._yam_gripper_index]

        self._q_target = q_step
        self._set_yam_target(self._q_target)

    def _yam_servo_to_T(self, T_yam_target: np.ndarray, now: float) -> None:
        """IK an absolute EE target -> clamped joint step -> publisher.

        Same IK / soft-tolerance / per-loop step-clamp logic as the head path,
        warm-started from the last commanded joints. Used by replay.
        """
        q_cur = self._q_target
        if q_cur is None:
            return
        init_q = self._robot_q_to_model_qpos(q_cur)
        success, q_ik = self._yam_kin.ik(
            T_yam_target,
            self.yam_ee_site,
            init_q=init_q,
            pos_threshold=1e-3,
            ori_threshold=1e-2,
            max_iters=self.yam_ik_max_iters,
        )
        if not success and self.yam_ik_soft_pos_tol > 0.0 and q_ik is not None:
            achieved = self._yam_kin.fk(q_ik)
            pos_err = float(np.linalg.norm(achieved[:3, 3] - T_yam_target[:3, 3]))
            R_err = achieved[:3, :3] @ T_yam_target[:3, :3].T
            ori_err = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
            if pos_err <= self.yam_ik_soft_pos_tol and ori_err <= self.yam_ik_soft_ori_tol:
                success = True
        if not success:
            if now - self._last_safe_log > 1.0:
                logger.warning("[yam-ik] (replay) not converged & outside soft tol — holding")
                self._last_safe_log = now
            return

        q_candidate = q_cur.copy()
        q_candidate[: self._yam_n_arm] = q_ik[: self._yam_n_arm]
        delta_q = q_candidate[: self._yam_n_arm] - self._q_target[: self._yam_n_arm]
        delta_q = np.clip(delta_q, -self.yam_max_step, self.yam_max_step)
        q_step = self._q_target.copy()
        q_step[: self._yam_n_arm] = self._q_target[: self._yam_n_arm] + delta_q
        if self._yam_gripper_index is not None and len(q_step) > self._yam_gripper_index:
            q_step[self._yam_gripper_index] = q_cur[self._yam_gripper_index]
        self._q_target = q_step
        self._set_yam_target(self._q_target)

    def _step_replay(self) -> None:
        """Replay: follow a recorded EE-pose7 target (same frame as FK output)."""
        now = time.monotonic()
        pose7 = self._read_head_pose_data()
        if pose7 is None or pose7.shape[0] < 7:
            return
        pos = np.asarray(pose7[0:3], dtype=np.float64)
        quat_xyzw = np.asarray(pose7[3:7], dtype=np.float64)
        if float(np.linalg.norm(quat_xyzw)) < 1e-6:
            return  # no valid target yet
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
        T_target = np.eye(4)
        T_target[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T_target[:3, 3] = pos
        self._yam_servo_to_T(T_target, now)

    def stop_server(self) -> None:
        if not self.running:
            return
        logger.info(f"Stopping YamHeadRobot {self.device_class_id} server...")
        self.running = False

        if self._publisher is not None:
            try:
                self._publisher.stop()
            except Exception as exc:
                logger.warning(f"[yam-publisher] stop failed: {exc}")
            self._publisher = None

        # Ramp the arm home to all-zero joints, then gravity-comp idle.
        if self._yam_robot is not None and self.home_to_zero_on_exit:
            try:
                self.running = True  # allow the ramp loop to run during shutdown
                self._ramp_to_q(np.zeros(self._yam_n_arm))
                if self.gravity_comp_idle_on_exit and hasattr(self._yam_robot, "enter_gravity_comp_idle"):
                    self._yam_robot.enter_gravity_comp_idle()
                    logger.info("[yam-home] gravity-comp idle (arm can be moved by hand)")
                else:
                    # Do NOT enter gravity-comp idle: the ramp's final
                    # command_joint_pos already holds the zero pose under PD
                    # position control, so the arm stays rigid (does NOT sag)
                    # until the process exits / power off.
                    logger.info("[yam-home] holding zero pose (stiff) until shutdown")
            except Exception as exc:
                logger.warning(f"[yam-home] ramp failed: {exc}")
            finally:
                self.running = False

        # Stop the underlying i2rt control loop. MotorChainRobot runs its
        # gravity-comp/PD server on a NON-daemon thread; without close() that
        # thread (and its CAN interface) keeps running, blocking interpreter
        # exit so the process hangs at shutdown (Ctrl+C appears unresponsive and
        # multiprocessing's atexit join() never returns). close() sets the stop
        # event, joins the thread, and releases the CAN bus.
        if self._yam_robot is not None:
            try:
                if hasattr(self._yam_robot, "close"):
                    self._yam_robot.close()
                    logger.info("[yam] i2rt control loop stopped (MotorChainRobot closed)")
            except Exception as exc:
                logger.warning(f"[yam] failed to close i2rt robot: {exc}")
            self._yam_robot = None

        if self._head_shm is not None:
            try:
                self._head_shm.close()
            except Exception as exc:
                logger.warning(f"Failed to close head SHM: {exc}")
            self._head_shm = None

        if self._head_pose_shm is not None:
            try:
                self._head_pose_shm.close()
            except Exception as exc:
                logger.warning(f"Failed to close head pose SHM: {exc}")
            self._head_pose_shm = None

        self._cleanup_shared_memory()
        logger.info("YamHeadRobot stopped")

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update(
            {
                "head_shm": self._head_shm_name,
                "yam_channel": self.yam_channel,
                "yam_orientation_mode": self.yam_orientation_mode,
                "yam_ee_site": self.yam_ee_site,
                "state": self._yam_state,
            }
        )
        return info


class RizonRobotLKForceGated(StructuredRizonRobot):
    """RizonRobot + R1-style soft force gate on the LKTech gripper.

    The bundled LKTech SDK ignores ``Move()``'s ``force_limit`` argument
    (see R1_teleoperation), so this subclass re-implements R1's soft cap WITHOUT
    modifying ``RizonRobot``: when a command would close the gripper FURTHER and
    the motor's q-axis current ``iq`` already indicates a hard clamp in the
    closing direction, the command is dropped and the gripper holds position.
    Opening commands are always allowed so the operator can recover from a clamp.

    LKTech sign convention (CompliantGripper defaults max_torque_iq=+160 /
    min_torque_iq=-90)::

        iq > 0  ->  motor pushing in CLOSING direction
        iq < 0  ->  motor pushing in OPENING direction

    so we gate on SIGNED iq (``iq >= threshold``), not ``abs(iq)``. Normalized
    gripper value convention matches R1's ``1 - trigger`` (smaller value = more
    closed), which ``RizonRobot`` passes straight through in ``normalized`` mode,
    so "closing further" == "new value < last commanded value".

    Extra config keys (in addition to all ``RizonRobot`` keys):
      gripper_iq_force_limit: scalar or 2-list. Signed iq cap in the closing
                              direction. ``<= 0`` disables the gate (default 80,
                              matching R1's --gripper-force-limit).
      gripper_force_backoff: scalar or 2-list. Step (in command-value units, i.e.
                             METERS in ``width_m`` mode) to OPEN the gripper by
                             whenever ``iq`` is over the cap, instead of merely
                             holding. Sent once per control cycle so the grip is
                             actively eased open until ``iq`` falls below the cap,
                             regulating the clamp force toward the threshold.
                             ``<= 0`` keeps the legacy "hold position" behavior.
      gripper_force_release: scalar or 2-list. Width margin (command-value units,
                             METERS in ``width_m`` mode) the operator must OPEN
                             past the latched force floor before normal closing is
                             re-enabled. Larger = stickier latch, less chatter.
                             Only meaningful when ``gripper_force_backoff > 0``.
      gripper_force_print_period: throttle (s) for the "blocking"/"backoff" log.

    Latch behavior (when ``gripper_force_backoff > 0``): the first over-force
    closing command latches a "force floor" width. While latched, any tighter
    command is clamped to the floor (the grip never re-closes harder), and the
    floor is eased open step-by-step until ``iq`` falls below the cap. The latch
    only releases once the operator opens past ``floor + gripper_force_release``,
    so a steady squeeze settles at the floor instead of chattering.
    """

    def __init__(
        self,
        device_class_id: int,
        robot_sn: str | Sequence[str],
        state_feedback: Mapping[str, Any],
        control: Mapping[str, Any],
        *,
        gripper: Optional[Mapping[str, Any]] = None,
        motion_limits: Optional[Mapping[str, Any]] = None,
        home: Optional[Mapping[str, Any]] = None,
        network_interface_whitelist: Optional[Sequence[str]] = None,
        signal_files: Optional[Dict[str, str]] = None,
        traj_run_recording: Optional[Dict[str, Any]] = None,
        cartesian_planner: Optional[Dict[str, Any]] = None,
        gripper_iq_force_limit: Union[float, List[float]] = 80.0,
        gripper_force_backoff: Union[float, List[float]] = 0.0,
        gripper_force_release: Union[float, List[float]] = 0.005,
        gripper_force_print_period: float = 1.0,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            robot_sn=robot_sn,
            state_feedback=state_feedback,
            control=control,
            gripper=gripper,
            motion_limits=motion_limits,
            home=home,
            network_interface_whitelist=network_interface_whitelist,
            signal_files=signal_files,
            traj_run_recording=traj_run_recording,
            cartesian_planner=cartesian_planner,
        )
        self.gripper_iq_force_limits = self._normalize_force_gate_value(
            gripper_iq_force_limit, "gripper_iq_force_limit"
        )
        self.gripper_force_backoffs = self._normalize_force_gate_value(
            gripper_force_backoff, "gripper_force_backoff"
        )
        self.gripper_force_releases = self._normalize_force_gate_value(
            gripper_force_release, "gripper_force_release"
        )
        self.gripper_force_print_period = float(gripper_force_print_period)
        # Per-index bookkeeping: last commanded value + last block-log time.
        self._iq_last_value: Dict[int, float] = {}
        self._iq_last_block_log: Dict[int, float] = {}
        # Latched "force floor" width per index (None/absent = not latched).
        self._force_floor: Dict[int, float] = {}

    def _normalize_force_gate_value(
        self,
        value: Union[float, List[float]],
        field: str,
    ) -> List[float]:
        arm_count = len(self.gripper_config.arms)
        values = (
            list(value)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            else [value] * arm_count
        )
        if len(values) != arm_count:
            raise ValueError(f"{field} must contain one value per arm")
        return [float(item) for item in values]

    def _read_gripper_iq(self, gripper, index: int, label: str) -> Optional[float]:
        """Return signed motor iq, or None (logged, throttled) if unreadable."""
        try:
            return float(gripper.motor.read_status2().iq)
        except Exception as exc:  # noqa: BLE001
            now = time.time()
            if now - self._iq_last_block_log.get(index, 0.0) > self.gripper_force_print_period:
                logger.warning(
                    f"[gripper-{label}-force] iq read failed ({exc!r}); skipping force gate"
                )
                self._iq_last_block_log[index] = now
            return None

    def _force_log(self, index: int, message: str) -> None:
        """Throttled warning for the force gate."""
        now = time.time()
        if now - self._iq_last_block_log.get(index, 0.0) > self.gripper_force_print_period:
            logger.warning(message)
            self._iq_last_block_log[index] = now

    def _move_gripper_value(self, gripper, gripper_value: float, index: int, label: str, wait: bool = False) -> bool:
        # Only gate the LKTech backend; other backends keep default behavior.
        if self.gripper_config.backend != "lktech" or gripper is None:
            return super()._move_gripper_value(gripper, gripper_value, index, label, wait)

        try:
            threshold = float(self.gripper_iq_force_limits[index])
        except (IndexError, TypeError, ValueError):
            threshold = 0.0
        try:
            backoff = float(self.gripper_force_backoffs[index])
        except (IndexError, TypeError, ValueError):
            backoff = 0.0
        try:
            release = float(self.gripper_force_releases[index])
        except (IndexError, TypeError, ValueError):
            release = 0.0

        value = float(gripper_value)
        last = self._iq_last_value.get(index)
        floor = self._force_floor.get(index)
        max_open = (
            1.0
            if self.gripper_config.command_value_mode == "normalized"
            else float(self.gripper_config.arms[index].max_width)
        )

        # --- Latched force-floor state machine (only when back-off is enabled) ---
        if floor is not None:
            # Release the latch once the operator deliberately opens past the
            # floor (plus hysteresis), then fall through to normal handling.
            if value >= floor + release:
                self._force_floor.pop(index, None)
                self._force_log(index, f"[gripper-{label}-force] released latch (floor was {floor:.3f})")
                floor = None
            else:
                iq_signed = self._read_gripper_iq(gripper, index, label)
                if iq_signed is not None and iq_signed >= threshold and floor < max_open:
                    # Still over the cap: ease the floor further open.
                    new_floor = min(floor + backoff, max_open)
                    self._force_log(
                        index,
                        f"[gripper-{label}-force] iq={iq_signed:+.0f} >= {threshold:.0f} "
                        f"— easing floor {floor:.3f} -> {new_floor:.3f}",
                    )
                    ok = super()._move_gripper_value(gripper, new_floor, index, label, wait)
                    if ok:
                        self._force_floor[index] = new_floor
                        self._iq_last_value[index] = new_floor
                # Clamp any tighter command to the floor: hold there, never close
                # harder. Re-send only if we are not already holding the floor.
                elif last is None or abs(last - floor) > 1e-6:
                    ok = super()._move_gripper_value(gripper, floor, index, label, wait)
                    if ok:
                        self._iq_last_value[index] = floor
                # Keep re-entering next cycle so we keep watching iq while latched.
                return True

        # --- Not latched: gate on a strictly-closing command ---------------------
        # "Close further" == value decreasing (smaller value = more closed).
        closing = threshold > 0.0 and last is not None and value < last
        if closing:
            iq_signed = self._read_gripper_iq(gripper, index, label)
            if iq_signed is not None and iq_signed >= threshold:
                held = last if last is not None else value
                if backoff > 0.0:
                    # Back off one step AND latch a force floor so a steady squeeze
                    # settles instead of re-crushing on the next cycle.
                    target_open = min(held + backoff, max_open)
                    self._force_log(
                        index,
                        f"[gripper-{label}-force] iq={iq_signed:+.0f} >= {threshold:.0f} "
                        f"— backing off {held:.3f} -> {target_open:.3f} (latched)",
                    )
                    ok = super()._move_gripper_value(gripper, target_open, index, label, wait)
                    if ok:
                        self._force_floor[index] = target_open
                        self._iq_last_value[index] = target_open
                    return ok

                # Legacy behavior (back-off disabled): hold position, no latch.
                self._force_log(
                    index,
                    f"[gripper-{label}-force] iq={iq_signed:+.0f} >= {threshold:.0f} "
                    f"— blocking further close (hold at {held:.2f})",
                )
                return False

        ok = super()._move_gripper_value(gripper, gripper_value, index, label, wait)
        if ok:
            self._iq_last_value[index] = value
        return ok


if __name__ == "__main__":
    real_main() # calibrate the gripper width
    # sim_main()
