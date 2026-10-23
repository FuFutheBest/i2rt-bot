#!/usr/bin/env python3
"""
Robot Device - Base robot class and PyBullet-based simulated robot.

Author: Jun Lv,Zixi Ying
"""

import threading
import time
from functools import wraps

import numpy as np
from typing import Any, Dict, Mapping, Optional

from devices.base import BaseDevice
from devices.config import ShmEndpointConfig, StateFeedbackConfig
from devices.contracts import DeviceKind
from devices.robots.config import DUAL_SOURCE_SELECTOR_ROLES, RobotControlConfig
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    calculate_device_shm_size,
    connect_to_shared_memory,
    create_control_shm,
    get_control_frame_offsets,
    get_dtype,
    get_latest_control_frame_index,
    pack_buffer_header,
    pack_device_header,
    unpack_device_header,
)
from utils.dtypes import SUPPORTED_DTYPE_NAMES
from utils.logger_config import logger
from utils.time_control import precise_loop_timing


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
    source_selector_owner_supported = False

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        control: RobotControlConfig | Mapping[str, Any],
    ) -> None:
        """Initialize state and command channels from structured contracts."""
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
        )
        self.control_config = RobotControlConfig.require(control)

        control_format = self.control_config.format
        # Robot command executer fps
        self.command_fps = self.control_config.fps

        # Legacy async execution mode: when enabled, control commands are executed in a
        # dedicated command thread at ``command_fps`` (decoupled from the state loop), and
        # RizonRobot's Cartesian motion planner / trajectory recording are allowed to run.
        # When disabled (default), commands are executed synchronously inside the main state
        # loop (single-threaded), which avoids concurrent hardware (e.g. gripper serial) access.
        self.legacy_async_mode = self.control_config.execution_mode == "legacy_async"
        self._command_thread: Optional[threading.Thread] = None
        self._command_stop = threading.Event()

        self.control_shape = control_format.shape
        self.control_dtype = control_format.dtype
        self.control_buffer_size = self.control_config.buffer_size

        # Control buffer management
        self.control_shared_memory = None
        self._owned_control_channels = {}
        self._control_shm_layouts = {}
        self._external_source_reconnect_at: dict[str, float] = {}

        # Source-selection selector SHM.  Consumers always attach read-only;
        # a producer-specific subclass (for example I2RT) may own and write it.
        self._control_flag_shm = None
        self._control_flag_shm_name = None
        self._source_selector_fps: float | None = None
        self._source_selector_reconnect_at = 0.0
        self._control_tick_selector_value: float | None = None

        # Calculate control buffer layout
        self._calculate_control_buffer_layout()
        self._configure_source_selection()

    def _resolve_source_endpoint_name(self, endpoint: ShmEndpointConfig) -> str:
        """Resolve a source endpoint to its canonical shared-memory name."""
        if not isinstance(endpoint, ShmEndpointConfig):
            raise TypeError("source endpoint must be a validated ShmEndpointConfig")
        return endpoint.resolve_shm_name(
            owner_device=self.device_name,
            owner_device_class_id=self.device_class_id,
        )

    def _endpoint_is_owned_by_self(self, endpoint: ShmEndpointConfig) -> bool:
        """Return whether this process owns the endpoint's device instance."""
        if not isinstance(endpoint, ShmEndpointConfig):
            raise TypeError("source endpoint must be a validated ShmEndpointConfig")
        device = endpoint.device
        device_class_id = endpoint.device_class_id
        if device == "self":
            device = self.device_name
            if device_class_id is None:
                device_class_id = self.device_class_id
        return device == self.device_name and int(device_class_id) == self.device_class_id

    def _can_own_source_selector(self) -> bool:
        """Whether this device instance publishes its own selector endpoint."""
        return self.source_selector_owner_supported

    def _configure_source_selection(self) -> None:
        """Resolve configured source endpoints and establish ownership metadata."""
        selection = self.control_config.source_selection
        mode = selection.mode
        if mode == "single_source":
            source_items = {"source": selection.source}
            selector = None
        elif mode == "dual_source":
            source_items = selection.sources
            selector = selection.dual_source_selector
        else:
            raise ValueError(f"unsupported source_selection.mode: {mode!r}")
        if source_items is None or any(endpoint is None for endpoint in source_items.values()):
            raise ValueError("source_selection must define every required source endpoint")
        if mode == "dual_source" and selector is None:
            raise ValueError("dual_source requires dual_source_selector")

        self.source_selection = selection
        self.source_selection_mode = mode
        self._source_endpoints = {}
        self._source_ports = {}
        self._source_names = {}
        self._source_name_to_role = {}

        for role, endpoint in source_items.items():
            name = self._resolve_source_endpoint_name(endpoint)
            port = endpoint.port
            self._source_endpoints[role] = endpoint
            self._source_ports[role] = port
            self._source_names[role] = name
            self._source_name_to_role.setdefault(name, role)

        if (
            mode == "dual_source"
            and self._source_names["policy"] == self._source_names["teleoperation"]
        ):
            raise ValueError(
                "dual_source policy and teleoperation endpoints must resolve to "
                "different shared-memory channels"
            )

        self._source_selector_endpoint = None
        self._source_selector_mapping = {}
        self._source_selector_owned_by_self = False
        if mode == "dual_source":
            self._source_selector_endpoint = selector.endpoint
            self._control_flag_shm_name = self._resolve_source_endpoint_name(
                self._source_selector_endpoint
            )
            self._source_selector_owned_by_self = self._endpoint_is_owned_by_self(
                self._source_selector_endpoint
            )
            if (
                self._source_selector_owned_by_self
                and not self._can_own_source_selector()
            ):
                raise ValueError(
                    "dual_source_selector.endpoint may use device: self only for "
                    "a selector-producing robot"
                )
            self._source_selector_mapping = dict(DUAL_SOURCE_SELECTOR_ROLES)

        # The robot always owns its command channel.  Configured endpoints may
        # point at that channel or at external state/command producers.
        self._owned_control_shm_names = {self.default_control_shm_name}
        self._external_control_shm_names = set()
        names = []
        for name in self._source_names.values():
            if name not in names:
                names.append(name)
            if name != self.default_control_shm_name:
                self._external_control_shm_names.add(name)
        if self.default_control_shm_name not in names:
            names.append(self.default_control_shm_name)
        self.control_shm_names = names

    def _is_dual_source_mode(self) -> bool:
        return self.source_selection_mode == "dual_source"

    def _source_name_for_role(self, role: str = "source") -> Optional[str]:
        return self._source_names.get(role)

    def _source_port_for_role(self, role: str = "source") -> Optional[str]:
        return self._source_ports.get(role)

    def _read_source_selector_value(self) -> float:
        """Read the latest selector value, defaulting to policy (zero)."""
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
            key = "0"
        elif numeric_value == 1.0:
            key = "1"
        else:
            # Selector values are a strict binary protocol.  Invalid values fail
            # closed to policy instead of being rounded into teleoperation.
            key = "0"
        role = self._source_selector_mapping.get(key)
        if role in self._source_names:
            return role
        # A missing/stale selector must fail closed to the policy source.
        return self._source_selector_mapping.get("0", "policy")

    def _selected_source_name(self, selector_value: Optional[float] = None) -> Optional[str]:
        return self._source_name_for_role(self._selected_source_role(selector_value))

    def _selected_source_port(self, selector_value: Optional[float] = None) -> Optional[str]:
        return self._source_port_for_role(self._selected_source_role(selector_value))

    def _selector_value_for_control_tick(self) -> Optional[float]:
        """Return the selector value captured before the active control tick."""
        if not self._is_dual_source_mode():
            return None
        return self._control_tick_selector_value

    def _process_source_selection_transition(self) -> None:
        """Apply policy/teleoperation transition effects once per control tick."""
        if not self._is_dual_source_mode():
            return

        selector_value = self._read_source_selector_value()
        self._control_tick_selector_value = selector_value
        current_role = self._selected_source_role(selector_value)
        previous_value = getattr(self, "_last_control_flag", 0.0)
        previous_role = getattr(
            self,
            "_last_source_selection_role",
            self._selected_source_role(previous_value),
        )

        if previous_role == "teleoperation" and current_role == "policy":
            self._clear_control_shm()
            self._policy_control_start_timestamp = time.time_ns()
            logger.info(
                "Source selection switched to policy: cleared control SHM, timestamp=%s",
                self._policy_control_start_timestamp,
            )
        if (
            hasattr(self, "_toggle_control_mode")
            and getattr(self, "is_follower_role", False)
            and current_role != previous_role
        ):
            self._toggle_control_mode()

        self._last_control_flag = selector_value
        self._last_source_selection_role = current_role

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

        # Store in control shared memory dict (don't overwrite user-specified control_shm_names)
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
        logger.info("Initialized control buffer header")

        # Phase 3: Convert to read-only after initialization
        from utils.shm_utils import make_readonly
        readonly_control_shm = make_readonly(control_shm)
        self.control_shared_memory[self.default_control_shm_name] = readonly_control_shm
        self._control_shm_layouts[self.default_control_shm_name] = {
            "shape": tuple(self.control_shape),
            "dtype": self.control_dtype,
            "fps": self.fps,
            "buffer_size": self.control_buffer_size,
            "frame_size": self.control_frame_size,
            "frame_data_size": self.control_frame_data_size,
            "frame_offsets": list(self.control_frame_offsets),
        }

        logger.info("Control SHM converted to READ-ONLY after initialization")

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
            logger.debug(f"Control shared memory not found: {shm_name}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to control shared memory {shm_name}: {e}")
            return False

    def _ensure_external_control_source(self, shm_name: str, *, force: bool = False) -> bool:
        """Attach one configured external source, retrying at a bounded rate."""
        if shm_name not in self._external_control_shm_names:
            return False
        if self.control_shared_memory is None:
            return False
        if self.control_shared_memory.get(shm_name) is not None:
            return True

        now = time.monotonic()
        if not force and now < self._external_source_reconnect_at.get(shm_name, 0.0):
            return False
        self._external_source_reconnect_at[shm_name] = now + 0.5
        connected = self._connect_to_existing_control_shm(shm_name)
        if connected:
            self._external_source_reconnect_at.pop(shm_name, None)
        return connected

    def _disconnect_external_control_source(self, shm_name: str) -> None:
        """Drop a stale external handle so its producer can be reattached."""
        if (
            shm_name not in self._external_control_shm_names
            or self.control_shared_memory is None
        ):
            return
        channel = self.control_shared_memory.get(shm_name)
        if channel is not None:
            try:
                channel.close()
            except Exception as exc:
                logger.debug("Failed to close stale source SHM %s: %s", shm_name, exc)
        self.control_shared_memory[shm_name] = None
        self._control_shm_layouts.pop(shm_name, None)
        self._external_source_reconnect_at[shm_name] = time.monotonic() + 0.5

    def _connect_control_flag_shm(self) -> None:
        """Connect to the configured selector endpoint as a read-only consumer."""
        if not self._control_flag_shm_name:
            return
        if self._source_selector_owned_by_self:
            return
        now = time.monotonic()
        if now < self._source_selector_reconnect_at:
            return
        self._source_selector_reconnect_at = now + 0.5
        try:
            from utils.shm_utils import connect_to_shared_memory

            channel = connect_to_shared_memory(
                self._control_flag_shm_name,
                read_only=True
            )
            if not self._validate_source_selector_shm(channel):
                channel.close()
                logger.warning(
                    "Source selector SHM validation failed for: %s",
                    self._control_flag_shm_name,
                )
                self._control_flag_shm = None
                return
            self._control_flag_shm = channel
            self._source_selector_reconnect_at = 0.0
            logger.info(f"Connected to control flag SHM: {self._control_flag_shm_name} (READ-ONLY)")

        except FileNotFoundError:
            logger.debug(f"Control flag SHM not found: {self._control_flag_shm_name}")
            self._control_flag_shm = None
        except Exception as e:
            logger.warning(f"Failed to connect to control flag SHM: {e}")
            self._control_flag_shm = None

    def _validate_source_selector_shm(self, channel: Any) -> bool:
        """Validate the one-value floating-point source selector channel."""
        try:
            device_info = unpack_device_header(channel.buf[:DEVICE_HEADER_SIZE])
            if tuple(device_info["shape"]) != (1,):
                logger.error(
                    "Source selector shape must be (1,), got %s",
                    tuple(device_info["shape"]),
                )
                return False
            dtype_name = device_info["data_dtype"]
            if dtype_name not in {"float32", "float64"}:
                logger.error(
                    "Source selector dtype must be float32 or float64, got %s",
                    dtype_name,
                )
                return False
            expected_frame_size = FRAME_HEADER_SIZE + np.dtype(get_dtype(dtype_name)).itemsize
            if int(device_info["frame_size"]) != expected_frame_size:
                logger.error(
                    "Source selector frame size mismatch: expected %s, got %s",
                    expected_frame_size,
                    device_info["frame_size"],
                )
                return False
            buffer_info = self._unpack_buffer_header_from_channel(channel)
            buffer_size = int(buffer_info["buffer_size"])
            write_index = int(buffer_info["write_index"])
            if buffer_size <= 0 or not 0 <= write_index < buffer_size:
                logger.error("Invalid source selector buffer metadata")
                return False
            fps = float(device_info["fps"])
            if not np.isfinite(fps) or fps <= 0.0:
                logger.error("Invalid source selector FPS: %s", device_info["fps"])
                return False
            self._source_selector_fps = fps
            return True
        except Exception as exc:
            logger.error("Error validating source selector SHM: %s", exc)
            return False

    def _validate_control_shm(self, shm_name: Optional[str] = None) -> bool:
        """
        Validate the connected control shared memory.

        Returns:
            bool: True if validation successful, False otherwise
        """
        try:
            from utils.shm_utils import unpack_device_header, unpack_buffer_header

            names = [shm_name] if shm_name is not None else self.control_shm_names
            for shm_name in names:
                if shm_name is None or self.control_shared_memory.get(shm_name) is None:
                    return False
                # Read and validate device header
                device_header_data = self.control_shared_memory[shm_name].buf[:DEVICE_HEADER_SIZE]
                device_info = unpack_device_header(device_header_data)

                shm_shape = tuple(device_info["shape"])
                shm_dtype_str = device_info["data_dtype"]
                if not shm_shape or not shm_dtype_str:
                    logger.error(f"Invalid source SHM metadata: {shm_name}")
                    return False

                # A source endpoint carries the same command payload consumed
                # by this robot.  Reject a connected stream whose layout does
                # not match the declared control contract instead of silently
                # reshaping or truncating it.
                if shm_shape != tuple(self.control_shape):
                    logger.error(
                        "Source SHM shape mismatch for %s: expected %s, got %s",
                        shm_name,
                        tuple(self.control_shape),
                        shm_shape,
                    )
                    return False
                if shm_dtype_str not in SUPPORTED_DTYPE_NAMES:
                    logger.error("Invalid source SHM dtype for %s: %s", shm_name, shm_dtype_str)
                    return False
                source_fps = float(device_info["fps"])
                if not np.isfinite(source_fps) or source_fps <= 0.0:
                    logger.error("Invalid source SHM FPS for %s: %s", shm_name, device_info["fps"])
                    return False
                source_dtype = np.dtype(get_dtype(shm_dtype_str))
                expected_dtype = np.dtype(get_dtype(self.control_dtype))
                if source_dtype != expected_dtype:
                    logger.error(
                        "Source SHM dtype mismatch for %s: expected %s, got %s",
                        shm_name,
                        expected_dtype,
                        source_dtype,
                    )
                    return False

                # Read and validate buffer header
                buffer_header_data = self.control_shared_memory[shm_name].buf[DEVICE_HEADER_SIZE:DEVICE_HEADER_SIZE+BUFFER_HEADER_SIZE]
                buffer_info = unpack_buffer_header(buffer_header_data)

                shm_buffer_size = buffer_info['buffer_size']
                if shm_buffer_size <= 0:
                    logger.error(f"Invalid source SHM buffer size: {shm_name}")
                    return False

                # Recalculate frame offsets based on connected SHM
                shm_frame_size = device_info['frame_size']  # frame_size is at index 5
                if shm_frame_size <= FRAME_HEADER_SIZE:
                    logger.error(f"Invalid source SHM frame size: {shm_name}")
                    return False
                self._control_shm_layouts[shm_name] = {
                    "shape": shm_shape,
                    "dtype": shm_dtype_str,
                    "fps": source_fps,
                    "buffer_size": shm_buffer_size,
                    "frame_size": shm_frame_size,
                    "frame_data_size": shm_frame_size - FRAME_HEADER_SIZE,
                    "frame_offsets": get_control_frame_offsets(
                        shm_buffer_size, shm_frame_size
                    ),
                }

                logger.info("Control SHM validation successful")
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

    @staticmethod
    def _max_frame_age_ns(fps: float) -> int:
        """Allow three publish periods, with a 500 ms lower bound."""
        return int(max(0.5, 3.0 / fps) * 1e9)

    def _frame_is_fresh(self, timestamp_ns: int, fps: float) -> bool:
        """Return whether one source frame is current enough to execute."""
        if timestamp_ns <= 0:
            return False
        age_ns = time.time_ns() - timestamp_ns
        return 0 <= age_ns <= self._max_frame_age_ns(fps)

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
        if self.control_shared_memory is None:
            return None, None
        if self.control_shared_memory.get(shm_name) is None:
            self._ensure_external_control_source(shm_name)
        if not self.control_shared_memory.get(shm_name):
            return None, None

        layout = self._control_shm_layouts.get(shm_name)
        if layout is None and not self._validate_control_shm(shm_name):
            return None, None
        layout = self._control_shm_layouts.get(shm_name)
        if layout is None:
            return None, None

        # Get current buffer info from SHM
        current_frames_count, write_index = self._get_control_buffer_info(shm_name)
        if current_frames_count is None:
            self._disconnect_external_control_source(shm_name)
            return None, None
        if current_frames_count == 0:
            return None, None

        try:
            # Calculate frame offset for latest command
            buffer_size = layout["buffer_size"]
            latest_index = get_latest_control_frame_index(write_index, buffer_size)
            frame_offset = layout["frame_offsets"][latest_index]

            frame_header_data = self.control_shared_memory[shm_name].buf[frame_offset:frame_offset+FRAME_HEADER_SIZE]
            timestamp_ns = int.from_bytes(frame_header_data, byteorder='little')

            # Check if action is stale (written before the minimum valid timestamp)
            if min_timestamp_ns > 0 and timestamp_ns > 0 and timestamp_ns < min_timestamp_ns:
                return None, None
            if not self._frame_is_fresh(timestamp_ns, float(layout["fps"])):
                logger.debug("Ignoring stale control frame from %s", shm_name)
                self._disconnect_external_control_source(shm_name)
                return None, None

            # Read control data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + layout["frame_data_size"]
            numpy_control_dtype = get_dtype(layout["dtype"])
            # Extract control data: DataChannel.buf returns raw buffer, slicing works directly
            sliced_buf = self.control_shared_memory[shm_name].buf[data_start:data_end]
            control_array = np.frombuffer(bytes(sliced_buf), dtype=numpy_control_dtype)
            # Reshape if needed
            shape = tuple(layout["shape"])
            if len(shape) > 0:
                expected_count = int(np.prod(shape))
                if control_array.size < expected_count:
                    logger.error(
                        "Source SHM payload is smaller than its declared shape: %s",
                        shm_name,
                    )
                    return None, None
                control_array = control_array[:expected_count].reshape(shape)

            return timestamp_ns, control_array

        except Exception as e:
            logger.error(f"Error reading control command: {e}")
            self._disconnect_external_control_source(shm_name)
            return None, None

    def _read_latest_channel(
        self, channel: Any, min_timestamp_ns: int = 0
    ) -> tuple[Optional[int], Optional[np.ndarray]]:
        """Read a latest frame from a channel not present in control_shm_names."""
        if channel is None:
            return None, None
        try:
            device_info = unpack_device_header(channel.buf[:DEVICE_HEADER_SIZE])
            buffer_info = self._unpack_buffer_header_from_channel(channel)
            if buffer_info["current_frames_count"] == 0:
                return None, None
            frame_size = int(device_info["frame_size"])
            buffer_size = int(buffer_info["buffer_size"])
            frame_offsets = get_control_frame_offsets(buffer_size, frame_size)
            latest_index = get_latest_control_frame_index(
                buffer_info["write_index"], buffer_size
            )
            frame_offset = frame_offsets[latest_index]
            timestamp_ns = int.from_bytes(
                bytes(channel.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE]),
                byteorder="little",
            )
            if min_timestamp_ns > 0 and timestamp_ns > 0 and timestamp_ns < min_timestamp_ns:
                return None, None
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_size = frame_size - FRAME_HEADER_SIZE
            payload = np.frombuffer(
                bytes(channel.buf[data_start : data_start + data_size]),
                dtype=get_dtype(device_info["data_dtype"]),
            )
            shape = tuple(device_info["shape"])
            if shape and payload.size >= int(np.prod(shape)):
                payload = payload[: int(np.prod(shape))].reshape(shape)
            return timestamp_ns, payload
        except Exception as exc:
            logger.debug("Failed to read source selector channel: %s", exc)
            return None, None

    @staticmethod
    def _unpack_buffer_header_from_channel(channel: Any) -> dict[str, int]:
        from utils.shm_utils import unpack_buffer_header

        return unpack_buffer_header(
            channel.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state. To be implemented by subclasses.

        Returns:
            numpy.ndarray: Current robot state
        """
        raise NotImplementedError("Subclasses must implement get_robot_state()")

    def get_device_info(self) -> Dict[str, Any]:
        """Report source topology health alongside the standard device metadata."""
        info = super().get_device_info()
        active_role = self._selected_source_role(
            getattr(self, "_last_control_flag", 0.0)
        )
        active_source = self._source_name_for_role(active_role)
        active_source_ready = bool(
            self.control_shared_memory
            and active_source is not None
            and self.control_shared_memory.get(active_source) is not None
        )
        unavailable_external_sources = sorted(
            source
            for source in self._external_control_shm_names
            if self.control_shared_memory is None
            or self.control_shared_memory.get(source) is None
        )
        info["control_source_status"] = {
            "mode": self.source_selection_mode,
            "active_role": active_role,
            "active_source": active_source,
            "active_source_ready": active_source_ready,
            "unavailable_external_sources": unavailable_external_sources,
        }
        return info

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
        for shm_name, channel in list(self.control_shared_memory.items()):
            if channel:
                try:
                    channel.close()
                except Exception as exc:
                    logger.warning(f"Failed to close control SHM {shm_name}: {exc}")
                if shm_name in self._owned_control_shm_names:
                    owner_channel = self._owned_control_channels.get(shm_name)
                    try:
                        if owner_channel is not None and owner_channel is not channel:
                            owner_channel.close()
                        if owner_channel is not None:
                            owner_channel.unlink()
                        else:
                            channel.unlink()
                        logger.info(f"Cleaned up owned control shared memory: {shm_name}")
                    except (FileNotFoundError, PermissionError):
                        pass  # Already cleaned up or this handle is read-only.
                    except Exception as e:
                        logger.error(f"Error unlinking owned control shared memory: {e}")
            self.control_shared_memory[shm_name] = None
        self._owned_control_channels.clear()

        # Selector channels are separate from payload channels.  Consumers
        # close their local read-only handle but never unlink the producer's
        # shared memory; an owner-specific subclass performs the unlink.
        if self._control_flag_shm is not None and not self._source_selector_owned_by_self:
            try:
                self._control_flag_shm.close()
            except Exception as exc:
                logger.warning("Failed to close selector SHM: %s", exc)
            finally:
                self._control_flag_shm = None

    def _read_control_flag(self) -> float:
        """Read the latest source-selector value from its read-only channel."""
        if self._control_flag_shm is None:
            return 0.0
        timestamp_ns, value = self._read_latest_channel(self._control_flag_shm)
        selector_fps = self._source_selector_fps or self.fps
        if (
            value is None
            or timestamp_ns is None
            or not self._frame_is_fresh(timestamp_ns, selector_fps)
        ):
            if not self._source_selector_owned_by_self:
                try:
                    self._control_flag_shm.close()
                except Exception as exc:
                    logger.debug("Failed to close stale selector SHM: %s", exc)
                self._control_flag_shm = None
                self._source_selector_reconnect_at = time.monotonic() + 0.5
            return 0.0
        try:
            return float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return 0.0

    def _prepare_control_loop_resources(self) -> None:
        """Create owned channels and connect configured external sources."""
        if self.control_shared_memory is None:
            self.control_shared_memory = {
                name: None for name in self.control_shm_names
            }
        else:
            for name in self.control_shm_names:
                self.control_shared_memory.setdefault(name, None)
        for shm_name in self._external_control_shm_names:
            self._ensure_external_control_source(shm_name, force=True)
        # Every writable robot owns this channel, regardless of selected source.
        if self.control_shared_memory.get(self.default_control_shm_name) is None:
            self._create_control_shared_memory()
        if self._is_dual_source_mode() and not self._source_selector_owned_by_self:
            self._connect_control_flag_shm()

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

    def _read_control_dict_for_execute(self) -> Dict[str, Optional[np.ndarray]]:
        """Read latest control frames from all control SHMs (same filters as main loop)."""
        control_dict: Dict[str, Optional[np.ndarray]] = {}
        if self.control_shm_names is None:
            return control_dict
        min_timestamp_ns = self._minimum_control_timestamp_ns()
        for shm_name in self.control_shm_names:
            _timestamp_ns, control_array = self.read_latest_control_command(
                shm_name, min_timestamp_ns
            )
            control_dict[shm_name] = control_array
        return control_dict

    def _minimum_control_timestamp_ns(self) -> int:
        """Return the earliest command timestamp allowed after a policy transition."""
        policy_start_ts = getattr(self, "_policy_control_start_timestamp", 0)
        grace_period_ns = getattr(self, "_policy_control_grace_period_ns", int(0.5 * 1e9))
        last_home_command_timestamp = getattr(self, "_last_home_command_timestamp", 0)
        min_timestamp_ns = policy_start_ts + grace_period_ns if policy_start_ts > 0 else 0
        return max(min_timestamp_ns, last_home_command_timestamp)

    def _poll_and_execute_control_once(self) -> None:
        """One control tick: read latest control SHM and execute if present."""
        control_dict = self._read_control_dict_for_execute()
        if any(value is not None for value in control_dict.values()):
            self.execute_control_command(control_dict)

    def _command_loop_worker(self) -> None:
        """Dedicated thread: single-step at command_fps."""
        wait_for_next_command = precise_loop_timing(1.0 / float(self.command_fps))
        while self.running and not self._command_stop.is_set():
            try:
                self._process_source_selection_transition()
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

    def start_server(self) -> None:
        """Start the robot server and run control loop."""
        if self.running:
            logger.info(f"Robot {self.device_name}_{self.device_class_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_class_id} server...")
        self.running = True

        # Create state shared memory
        self._create_shared_memory()  # State buffer
        self._prepare_control_loop_resources()

        if not self.shared_memory or not any(self.control_shared_memory.values()):
            self.running = False
            raise RuntimeError("Failed to create/connect to shared memories")

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
                    self._process_source_selection_transition()
                    control_dict = {}
                    min_timestamp_ns = self._minimum_control_timestamp_ns()
                    for shm_name in self.control_shm_names:
                        # Read latest control command
                        # Pass minimum timestamp to filter stale actions after control mode switch
                        # Add grace period to allow fresh policy actions to arrive
                        timestamp_ns, control_array = self.read_latest_control_command(shm_name, min_timestamp_ns)
                        control_dict[shm_name] = control_array

                    if any([value is not None for value in control_dict.values()]): # Any non-None values in control_dict
                        if iteration % command_iteration == 0:
                        # Execute control command
                            self.execute_control_command(control_dict)
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
        if not getattr(self, "running", False):
            return

        logger.info(
            "=== Stopping %s_%s server in BaseRobot ===",
            self.device_name,
            self.device_class_id,
        )
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
