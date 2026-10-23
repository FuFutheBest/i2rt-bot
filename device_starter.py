#!/usr/bin/env python3
"""
Device Starter - Starts device processes and the Device Manager.

This module provides centralized management for running multiple devices simultaneously
using separate processes for isolation and reliability, plus the new Device Manager
for data aggregation.

Author: Jun Lv
"""

import os
import sys
import time
import signal
import multiprocessing as mp
from typing import Dict, List, Any, Optional
from pathlib import Path

from omegaconf import DictConfig

# Add devices to path
sys.path.append(os.path.dirname(__file__))
from devices import DEVICE_CLASSES
from devices.cameras.usb_link import (
    UsbPortRequirementError,
    normalize_usb_port_requirement,
    usb_speed_mbps,
    validate_usb_port_requirement,
)
from manager import MANAGER_CLASSES
from utils.device_readiness import (
    DEFAULT_DEVICE_READY_POLL_INTERVAL_S,
    DEFAULT_DEVICE_READY_TIMEOUT_S,
    DeviceReadinessError,
    wait_for_device_streams,
)
from utils.shm_utils import get_dtype
from utils.config_parser import (
    DEFAULT_HYDRA_CONFIG_NAME,
    ensure_config_dict,
    load_config,
    normalize_hydra_config,
    parse_device_configs_with_fields,
)
from utils.config_topology import ConfigTopologyError, validate_config_topology
from utils.cpu_affinity import CpuAffinitySettings
from utils.device_starter_guard import (
    DeviceStarterGuardError,
    DeviceStarterProcessGuard,
    check_gripper_websocket_preflight,
)
from utils.inference_options import as_bool
from utils.logger_config import logger
from incubator.control_panel.process_cleanup import (
    ProcessCleanupError,
    cleanup_before_startup,
    force_cleanup_requested,
)

project_root = os.path.dirname(os.path.abspath(__file__))
path_to_add = os.path.join(project_root, "build/flexiv_rdk_ext")
sys.path.insert(0, path_to_add)


def _install_child_signal_handlers() -> None:
    """Replace inherited supervisor handlers with a one-shot interrupt."""
    interrupted = False

    def request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


class DeviceStarter:
    """
    Device starter that handles multiple device processes and the Device Manager.

    Features:
    - Load configuration from YAML file
    - Start/stop multiple devices in separate processes
    - Start/stop the Device Manager for data aggregation
    - Monitor process health and restart on failure
    - Graceful shutdown handling
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: Optional[str] = None,
        enable_visualizers: bool = False,
        enable_device_manager: bool = True,
    ) -> None:
        """
        Initialize the device starter.

        Args:
            config_path: Path to configuration YAML file
            enable_visualizers: Whether to start visualizers for each device
            enable_device_manager: Whether to start the Device Manager for data aggregation
        """
        if config is None:
            resolved_path = config_path or "config.yaml"
            logger.info(f"Loading configuration from {resolved_path}")
            self.config_path = resolved_path
            self.config = load_config(resolved_path)
        else:
            # When an explicit config dictionary is provided, prefer using it directly
            # and avoid falling back to a default file path to prevent accidental overrides.
            self.config_path = config_path
            self.config = ensure_config_dict(config)
        self.devices: List[Dict[str, Any]] = []
        self.running = False
        self._shutdown_started = False
        self._shutdown_complete = False
        self._child_failure_detected = False
        self.enable_visualizers = as_bool(enable_visualizers)
        self.enable_device_manager = as_bool(enable_device_manager)
        self.device_manager_process = None
        self.policy_runner_process = None

        # Get global default for camera overlay from runtime config
        runtime_cfg = self.config.get("runtime", {})
        self.camera_overlay_enabled_default = runtime_cfg.get(
            "camera_overlay_enabled", False
        )
        self.cpu_affinity = CpuAffinitySettings.from_runtime(runtime_cfg)

        # Load configuration
        logger.info("DeviceStarter initialized")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _parse_device_configs(self) -> List[Dict[str, Any]]:
        """Parse device configurations and create device dictionaries."""
        # Use generic parser with additional fields for device starter
        additional_fields = {
            "process": None,
            "visualizer_process": None,
            "restart_count": 0,
            "last_restart": None,
        }
        return parse_device_configs_with_fields(self.config, additional_fields)

    def _preflight_config_topology(self) -> None:
        """Reject unresolved endpoint references before any device process starts."""
        validate_config_topology(
            self.config,
            validate_all_enabled_devices=True,
        )

    def _preflight_web_camera_usb_requirements(self) -> None:
        """Reject a disallowed camera link before starting any device process."""
        for device_info in self.devices:
            if device_info["device_class"] != "WebCameraOpenPi":
                continue

            device_config = device_info["config"]
            requirement = normalize_usb_port_requirement(
                device_config.get("usb_port_requirement", "any")
            )
            if requirement == "any":
                continue

            capture_id = device_config.get(
                "capture_id", device_info["device_class_id"]
            )
            speed_mbps = usb_speed_mbps(capture_id)
            validate_usb_port_requirement(
                requirement,
                speed_mbps,
                camera_id=int(device_info["device_class_id"]),
                capture_id=capture_id,
            )
            logger.info(
                "WebCameraOpenPi %s USB preflight passed: requirement=%s "
                "negotiated_speed=%s Mbps",
                device_info["device_class_id"],
                requirement,
                speed_mbps,
            )

    @staticmethod
    def _uses_i2rt_random_home_sync(device_info: Dict[str, Any]) -> bool:
        """Return whether this device belongs to the synchronized I2RT home pair."""
        return (
            device_info.get("device_class") == "I2RTRobot"
            and as_bool(device_info.get("config", {}).get("random_init_pose_sync", False))
        )

    def _prepare_i2rt_random_home_startup(self) -> None:
        """Publish one fresh home request before the synchronized pair starts.

        The request is deliberately written after generic process cleanup and
        before either device is forked.  The follower publishes the actual
        target only after it reaches Home, and the leader accepts only that
        target's matching ID.
        """
        synchronized = [
            device_info
            for device_info in self.devices
            if self._uses_i2rt_random_home_sync(device_info)
        ]
        if not synchronized:
            return

        leaders = [
            device_info
            for device_info in synchronized
            if "leader" in str(device_info["config"].get("mode", ""))
        ]
        followers = [
            device_info
            for device_info in synchronized
            if "leader" not in str(device_info["config"].get("mode", ""))
        ]
        if len(synchronized) != 2 or len(leaders) != 1 or len(followers) != 1:
            raise ValueError(
                "I2RT randomized-home sync supports exactly one leader/follower pair"
            )

        runtime = self.config.get("runtime", {})
        signal_files = runtime.get("signal_files", {}) if isinstance(runtime, dict) else {}
        request_path = signal_files.get("i2rt_random_home_request")
        target_path = signal_files.get("i2rt_random_home_target")
        result_pattern = signal_files.get("i2rt_random_home_result_pattern")
        if not isinstance(request_path, str) or not request_path.strip() or not isinstance(
            target_path, str
        ) or not target_path.strip() or not isinstance(result_pattern, str) or not result_pattern.strip():
            raise ValueError(
                "I2RT randomized-home sync requires runtime.signal_files "
                "i2rt_random_home_request, i2rt_random_home_target, and "
                "i2rt_random_home_result_pattern"
            )

        from utils.i2rt_random_home import (
            RandomHomeSyncError,
            publish_random_home_request,
            random_home_result_path,
        )

        try:
            leader_result_path = random_home_result_path(
                result_pattern,
                source_device_id=int(leaders[0]["device_class_id"]),
            )
            follower_result_path = random_home_result_path(
                result_pattern,
                source_device_id=int(followers[0]["device_class_id"]),
            )
        except (RandomHomeSyncError, TypeError, ValueError) as exc:
            raise ValueError(
                "I2RT randomized-home result pattern is invalid"
            ) from exc
        if leader_result_path == follower_result_path:
            raise ValueError(
                "I2RT randomized-home result pattern must produce separate "
                "leader and follower paths"
            )

        request_id = publish_random_home_request(request_path, reason="startup")
        logger.info(
            "Published fresh I2RT randomized-home startup request "
            f"{request_id} for follower {followers[0]['device_class_id']} and "
            f"leader {leaders[0]['device_class_id']}"
        )

    @staticmethod
    def _expand_device_config_sections(
        device_config: Dict[str, Any], valid_params: List[str]
    ) -> Dict[str, Any]:
        """Merge optional device sub-config sections into constructor kwargs."""
        config = device_config.copy()
        if "tracking" in valid_params:
            return config

        tracking_config = config.pop("tracking", None)

        if tracking_config is None:
            return config
        if not isinstance(tracking_config, dict):
            logger.warning(
                f"Ignoring non-dict tracking config for device {config.get('device_class_id')}"
            )
            return config

        merged_config = tracking_config.copy()
        merged_config.update(config)
        return merged_config

    def _start_generic_device_process(
        self, device_info: Dict[str, Any]
    ) -> Optional[mp.Process]:
        """Start a device process using dynamic import based on device class."""
        try:
            # Get device class from mapping
            device_class = DEVICE_CLASSES.get(device_info["device_class"])
            if not device_class:
                logger.error(f"Unknown device class: {device_info['device_class']}")
                return None

            def run_device() -> None:
                device = None
                _install_child_signal_handlers()
                try:
                    self.cpu_affinity.apply(
                        "device",
                        device_class=device_info["device_class"],
                        device_class_id=int(device_info["config"]["device_class_id"]),
                    )
                    # Get device class from devices module
                    from devices import DEVICE_CLASSES

                    device_class = DEVICE_CLASSES.get(device_info["device_class"])
                    if not device_class:
                        raise ValueError(
                            f"Unknown device class: {device_info['device_class']}"
                        )

                    # Get device class constructor parameters
                    import inspect
                    from utils.device_config import parse_device_config

                    sig = inspect.signature(device_class.__init__)
                    valid_params = list(sig.parameters.keys())

                    if "state_feedback" not in valid_params:
                        raise TypeError(
                            f"{device_info['device_class']} must declare the required "
                            "state_feedback constructor parameter"
                        )
                    if (
                        DEVICE_CLASSES.kind(device_info["device_class"]).writable
                        and "control" not in valid_params
                    ):
                        raise TypeError(
                            f"{device_info['device_class']} must declare the required "
                            "control constructor parameter"
                        )

                    # Create device with all config parameters
                    parsed_config = device_info.get("parsed_config")
                    if parsed_config is None:
                        parsed_config = parse_device_config(
                            device_info["config"],
                            device_kind=DEVICE_CLASSES.kind(
                                device_info["device_class"]
                            ),
                        )
                    device_config = self._expand_device_config_sections(
                        parsed_config.constructor_kwargs(),
                        valid_params,
                    )

                    # Filter config to only include valid constructor parameters
                    filtered_config = {
                        k: v for k, v in device_config.items() if k in valid_params
                    }

                    # Create device instance with filtered config parameters
                    device = device_class(**filtered_config)

                    logger.info(
                        f"Starting {device_info['device_class']} {device_info['config']['device_class_id']}"
                    )
                    device.start_server()

                except KeyboardInterrupt:
                    logger.info(
                        f"Stopping {device_info['device_class']} "
                        f"{device_info['device_class_id']} after shutdown request"
                    )
                    if device is not None:
                        close_device = getattr(device, "close", None)
                        if not callable(close_device):
                            close_device = getattr(device, "stop_server", None)
                        if callable(close_device):
                            try:
                                close_device()
                            except Exception as exc:
                                logger.warning(
                                    f"Failed to close {device_info['device_class']} "
                                    f"{device_info['device_class_id']}: {exc}"
                                )
                except Exception as e:
                    logger.error(
                        f"{device_info['device_class']} {device_info['config']['device_class_id']} error: {e}"
                    )
                    raise

            process = mp.Process(
                target=run_device,
                name=f"{device_info['device_class']}-{device_info['config']['device_class_id']}",
            )
            return process

        except Exception as e:
            logger.error(
                f"Failed to create device process for {device_info['device_class']}: {e}"
            )
            return None

    def _start_generic_visualizer_process(
        self, device_info: Dict[str, Any]
    ) -> Optional[mp.Process]:
        """Start a visualizer process using dynamic import based on device class."""
        try:
            # Capture the global default value for camera overlay (needed for multiprocessing)
            camera_overlay_default = self.camera_overlay_enabled_default

            def run_visualizer() -> None:
                _install_child_signal_handlers()
                try:
                    # Visualization is optional; keep plotting dependencies out
                    # of the starter's import path when it is disabled.
                    import matplotlib
                    from visualizers import VISUALIZER_CLASSES

                    self.cpu_affinity.apply(
                        "visualizer",
                        device_class=device_info["device_class"],
                        device_class_id=int(device_info["config"]["device_class_id"]),
                    )
                    # Set matplotlib backend to avoid GUI conflicts in multiprocessing
                    matplotlib.use("TkAgg")

                    # Get visualizer class from mapping
                    visualizer_class = VISUALIZER_CLASSES.get(
                        device_info["device_class"]
                    )
                    if not visualizer_class:
                        logger.error(
                            f"Unknown visualizer type: {device_info['device_class']}"
                        )
                        return

                    # Get device name directly from device class
                    device_name = device_info["device_class"]

                    shared_memory_name = (
                        f"{device_name}_{device_info['device_class_id']}_data"
                    )
                    from utils.device_config import state_feedback_mapping

                    state_feedback = state_feedback_mapping(device_info["config"])
                    update_interval = int(1000 / float(state_feedback["fps"]))
                    raw_format = state_feedback.get("format")
                    format_dtype = (
                        raw_format.get("dtype")
                        if isinstance(raw_format, dict)
                        else None
                    )
                    data_dtype = get_dtype(
                        format_dtype
                        or state_feedback.get("data_dtype", "float32")
                    )

                    # Get enable_overlay from device config, fallback to global default from runtime config
                    # This allows per-device override while having a clear global default
                    enable_overlay = device_info["config"].get(
                        "enable_overlay", camera_overlay_default
                    )

                    # Create visualizer
                    visualizer = visualizer_class(
                        shared_memory_name=shared_memory_name,
                        data_dtype=data_dtype,
                        smoothing_window=5,
                        device_info=device_info,
                        enable_overlay=enable_overlay,
                    )

                    # Try to connect with retries
                    max_retries = 10
                    retry_delay = 0.5
                    for attempt in range(max_retries):
                        if visualizer.connect():
                            break
                        logger.info(
                            f"{device_info['device_class'].capitalize()} visualizer {device_info['device_class_id']}: Waiting for shared memory (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"{device_info['device_class'].capitalize()} visualizer {device_info['device_class_id']}: Failed to connect after {max_retries} attempts"
                        )
                        return

                    # Start visualization
                    visualizer.start_visualization(update_interval=update_interval)

                except KeyboardInterrupt:
                    logger.info(
                        f"Stopping {device_info['device_class']} visualizer "
                        f"for device {device_info['device_class_id']}"
                    )
                except Exception as e:
                    logger.error(
                        f"{device_info['device_class'].capitalize()} visualizer error: {e}"
                    )

            process = mp.Process(
                target=run_visualizer,
                name=f"{device_info['device_class']}_visualizer_{device_info['device_class_id']}",
            )
            process.start()
            return process

        except Exception as e:
            logger.error(
                f"Failed to create visualizer process for {device_info['device_class']}: {e}"
            )
            return None

    def _start_device_manager_process(self) -> Optional[mp.Process]:
        """Start the Device Manager process for data aggregation."""
        try:

            def run_device_manager() -> None:
                _install_child_signal_handlers()
                try:
                    self.cpu_affinity.apply("device_manager")
                    # Get device manager configuration
                    device_manager_config = self._normalized_device_manager_config()
                    manager_type = device_manager_config.get("type", "base")

                    # Get manager class from mapping
                    manager_class = MANAGER_CLASSES.get(manager_type)
                    if not manager_class:
                        raise ValueError(f"Unknown device manager type: {manager_type}")

                    manager_params = device_manager_config.get("params", {}) or {}
                    manager_params = ensure_config_dict(manager_params)

                    # For Hydra configs, pass the config object; for traditional configs, pass config_path
                    if (
                        hasattr(self, "config_path")
                        and self.config_path
                        and not manager_params.get("config_path")
                    ):
                        # Traditional config mode - pass config_path
                        manager_params.setdefault("config_path", self.config_path)
                    else:
                        # Hydra config mode - pass config object
                        manager_params.setdefault("config", self.config)

                    manager = manager_class(**manager_params)
                    logger.info(
                        f"Device Manager ({manager_type}) started for data aggregation"
                    )
                    manager.start()

                except KeyboardInterrupt:
                    logger.info("Stopping Device Manager after shutdown request")
                except Exception as e:
                    logger.error(f"Device Manager error: {e}")
                    raise

            process = mp.Process(target=run_device_manager, name="DeviceManager")
            return process

        except Exception as e:
            logger.error(f"Failed to create Device Manager process: {e}")
            return None

    def _normalized_device_manager_config(self) -> Dict[str, Any]:
        """Normalize legacy synchronized-manager settings before construction.

        ``SynchronizedDeviceManager`` reads ``master_device_class_id`` from the
        top-level config mapping, while constructor kwargs are forwarded from
        ``params``. Move the supported ``params`` form to that single source of
        truth and do not pass it as an unsupported constructor kwarg.
        """
        device_manager_config = ensure_config_dict(
            self.config.get("device_manager", {})
        )
        manager_params = ensure_config_dict(device_manager_config.get("params", {}))
        if device_manager_config.get("type") == "synchronized":
            params_master_device_class_id = manager_params.pop(
                "master_device_class_id", None
            )
            top_level_master_device_class_id = device_manager_config.get(
                "master_device_class_id"
            )
            if params_master_device_class_id is not None:
                if top_level_master_device_class_id is not None and int(
                    top_level_master_device_class_id
                ) != int(params_master_device_class_id):
                    raise ValueError(
                        "device_manager.master_device_class_id conflicts with "
                        "device_manager.params.master_device_class_id"
                    )
                if top_level_master_device_class_id is None:
                    device_manager_config[
                        "master_device_class_id"
                    ] = params_master_device_class_id
        device_manager_config["params"] = manager_params
        self.config["device_manager"] = device_manager_config
        return device_manager_config

    def _start_policy_runner_process(self) -> Optional[mp.Process]:
        """Start the Policy Runner process for action dispatching."""
        try:

            def run_policy_runner() -> None:
                _install_child_signal_handlers()
                try:
                    self.cpu_affinity.apply("policy_runner")
                    # Get policy runner configuration
                    policy_runner_config = self.config.get("action_executor", {})
                    if not policy_runner_config:
                        logger.warning(
                            "No action_executor configuration found, skipping PolicyRunner"
                        )
                        return

                    runner_type = policy_runner_config.get("type", "base_policy_runner")

                    # Get manager class from mapping
                    manager_class = MANAGER_CLASSES.get(runner_type)
                    if not manager_class:
                        raise ValueError(f"Unknown policy runner type: {runner_type}")

                    # Create policy runner instance - extract only known parameters
                    valid_params = {
                        "config_path",
                        "robot_control_shms",
                        "execution_fps",
                        "execution_clock",
                        "sim_clock_poll_interval_s",
                        "chunk_length",
                        "chunk_manager",
                        "config",
                        "enable_action_visualizer",
                        "action_visualizer_config",
                        "device_delays",
                        "device_dimension_delays",
                        "signal_files",
                    }

                    runner_config = {}
                    for key, value in policy_runner_config.items():
                        if key in valid_params:
                            runner_config[key] = value

                    # For Hydra configs, pass the config object; for traditional configs, pass config_path
                    if (
                        hasattr(self, "config_path")
                        and self.config_path
                        and not runner_config.get("config_path")
                    ):
                        # Traditional config mode - pass config_path
                        runner_config.setdefault("config_path", self.config_path)
                    else:
                        # Hydra config mode - pass config object
                        runner_config.setdefault("config", self.config)

                    logger.debug(
                        f"Policy runner config keys: {list(runner_config.keys())}"
                    )
                    runner = manager_class(**runner_config)
                    logger.info(
                        f"Policy Runner ({runner_type}) started for action dispatching"
                    )
                    runner.start()

                except KeyboardInterrupt:
                    logger.info("Stopping Policy Runner after shutdown request")
                except Exception as e:
                    logger.error(f"Policy Runner error: {e}")
                    raise

            process = mp.Process(target=run_policy_runner, name="PolicyRunner")
            return process

        except Exception as e:
            logger.error(f"Failed to create Policy Runner process: {e}")
            return None

    def _start_device_process(self, device_info: Dict[str, Any]) -> bool:
        """Start a single device process and optionally its visualizer."""
        try:
            # Start device process using dynamic import
            process = self._start_generic_device_process(device_info)
            if not process:
                return False

            process.start()
            device_info["process"] = process
            device_info["last_restart"] = time.time()

            logger.info(
                f"Started {device_info['device_class']} device {device_info['device_class_id']} (PID: {process.pid})"
            )

            # A composite device may publish valid streams without having a
            # dedicated visualizer implementation.
            device_visualizer_enabled = as_bool(
                device_info["config"].get("enable_visualizer", True)
            )
            if self.enable_visualizers and device_visualizer_enabled:
                try:
                    # Wait a bit for device to initialize
                    time.sleep(1.0)

                    visualizer_process = self._start_generic_visualizer_process(
                        device_info
                    )
                    if visualizer_process:
                        device_info["visualizer_process"] = visualizer_process
                        logger.info(
                            f"Started {device_info['device_class']} visualizer for device {device_info['device_class_id']} (PID: {visualizer_process.pid})"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to start visualizer for {device_info['device_class']} device {device_info['device_class_id']}: {e}"
                    )

            return True

        except Exception as e:
            logger.error(
                f"Failed to start {device_info['device_class']} device {device_info['device_class_id']}: {e}"
            )
            return False

    def _record_unrecoverable_child_failure(self, label: str, process: Any) -> None:
        """Remember a child failure so later cleanup cannot turn it into success."""
        self._child_failure_detected = True
        exit_code = getattr(process, "exitcode", None)
        detail = "" if exit_code is None else f" (exit code {exit_code})"
        logger.error(
            f"{label} exited unexpectedly{detail}; stopping the incomplete DeviceStarter stack"
        )

    def _shutdown_exit_code(self) -> int:
        """Return the process status after cleanup without masking child failures."""
        return 1 if getattr(self, "_child_failure_detected", False) else 0

    def _monitor_devices(self) -> bool:
        """Monitor device processes and report an unrecoverable child failure."""
        global_settings = self.config.get("global_settings", {})
        restart_on_failure = global_settings.get("restart_on_failure", True)
        max_restart_attempts = global_settings.get("max_restart_attempts", 3)
        restart_delay = global_settings.get("restart_delay", 2.0)

        logger.info("Starting device monitoring...")

        try:
            while self.running:
                for device_info in self.devices:
                    if device_info["process"] and not device_info["process"].is_alive():
                        logger.warning(
                            f"Device {device_info['device_class']}:{device_info['device_class_id']} process died"
                        )

                        synchronized_random_home = self._uses_i2rt_random_home_sync(
                            device_info
                        )
                        if synchronized_random_home:
                            logger.error(
                                "Refusing to restart one I2RT randomized-home device "
                                "independently because it would desynchronize the pair"
                            )
                            self._record_unrecoverable_child_failure(
                                f"I2RT randomized-home device "
                                f"{device_info['device_class']}:{device_info['device_class_id']} process",
                                device_info["process"],
                            )
                            self.running = False
                            return False
                        if (
                            restart_on_failure
                            and device_info["restart_count"] < max_restart_attempts
                        ):
                            logger.info(
                                f"Restarting device {device_info['device_class']}:{device_info['device_class_id']} "
                                f"(attempt {device_info['restart_count'] + 1}/{max_restart_attempts})"
                            )

                            time.sleep(restart_delay)
                            device_info["restart_count"] += 1
                            self._start_device_process(device_info)
                        else:
                            self._record_unrecoverable_child_failure(
                                f"Device {device_info['device_class']}:{device_info['device_class_id']} process",
                                device_info["process"],
                            )
                            self.running = False
                            return False

                    visualizer_process = device_info.get("visualizer_process")
                    if visualizer_process and not visualizer_process.is_alive():
                        self._record_unrecoverable_child_failure(
                            f"Visualizer for {device_info['device_class']}:{device_info['device_class_id']}",
                            visualizer_process,
                        )
                        self.running = False
                        return False

                policy_runner_process = getattr(self, "policy_runner_process", None)
                if policy_runner_process and not policy_runner_process.is_alive():
                    self._record_unrecoverable_child_failure(
                        "Policy Runner process", policy_runner_process
                    )
                    self.running = False
                    return False

                device_manager_process = getattr(self, "device_manager_process", None)
                if device_manager_process and not device_manager_process.is_alive():
                    self._record_unrecoverable_child_failure(
                        "Device Manager process", device_manager_process
                    )
                    self.running = False
                    return False

                time.sleep(1.0)  # Check every second
        except KeyboardInterrupt:
            logger.info("Device monitoring interrupted")
            self.running = False
        return True

    def _wait_for_device_streams(self) -> bool:
        """Wait for live frames instead of assuming a fixed initialization delay."""
        global_settings = self.config.get("global_settings", {})
        has_managed_devices = any(
            device_info["config"].get("managed_by") for device_info in self.devices
        )
        timeout_s = DEFAULT_DEVICE_READY_TIMEOUT_S
        poll_interval_s = DEFAULT_DEVICE_READY_POLL_INTERVAL_S
        if has_managed_devices:
            timeout_s = float(
                global_settings.get("managed_device_ready_timeout", timeout_s)
            )
            poll_interval_s = float(
                global_settings.get(
                    "managed_device_ready_poll_interval", poll_interval_s
                )
            )
        try:
            wait_for_device_streams(
                self.devices,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
        except DeviceReadinessError as exc:
            logger.error(f"Device startup readiness failed: {exc}")
            return False
        return True

    def start_all(self) -> bool:
        """Start all enabled devices and the Device Manager."""
        if self.running:
            logger.warning("Device starter is already running")
            return False

        logger.info("Starting Device Starter")
        try:
            self._preflight_config_topology()
        except ConfigTopologyError as exc:
            logger.error("Cannot start Device Starter: invalid device topology: %s", exc)
            return False
        self.devices = self._parse_device_configs()

        if not self.devices:
            logger.warning("No enabled devices found in configuration")
            return False

        try:
            self._preflight_web_camera_usb_requirements()
        except (UsbPortRequirementError, ValueError) as exc:
            logger.error(f"Cannot start Device Starter: {exc}")
            return False

        try:
            if force_cleanup_requested():
                cleanup_report = cleanup_before_startup("device_starter", force=True)
            else:
                cleanup_report = cleanup_before_startup("device_starter")
            logger.info(f"Startup process cleanup: {cleanup_report.summary()}")
        except ProcessCleanupError as exc:
            logger.error(f"Cannot start Device Starter: {exc}")
            return False

        try:
            self._prepare_i2rt_random_home_startup()
        except (OSError, ValueError) as exc:
            logger.error(f"Cannot start Device Starter: {exc}")
            return False

        # Mark startup active before forking so a signal during hardware
        # initialization still cleans up any children that were started.
        self._shutdown_started = False
        self._shutdown_complete = False
        self._child_failure_detected = False
        self.running = True

        # Start all devices first
        started_count = 0
        for device_info in self.devices:
            managed_by = device_info["config"].get("managed_by")
            if managed_by:
                logger.info(
                    f"Skipping virtual device {device_info['device_class']}_{device_info['device_class_id']}; "
                    f"it is owned by {managed_by}"
                )
                continue
            if self._start_device_process(device_info):
                started_count += 1

        if started_count == 0:
            logger.error("Failed to start any devices")
            self.running = False
            return False

        # Start Policy Runner early so chunk-control SHM exists while robots finish homing.
        action_executor_config = self.config.get("action_executor", {})
        if action_executor_config and action_executor_config.get("type") not in [
            None,
            "null",
            "disabled",
        ]:
            logger.info("Starting Policy Runner for action dispatching...")
            self.policy_runner_process = self._start_policy_runner_process()
            if self.policy_runner_process:
                self.policy_runner_process.start()
                logger.info(
                    f"Started Policy Runner (PID: {self.policy_runner_process.pid})"
                )
            else:
                logger.error("Failed to start Policy Runner")

        if not self._wait_for_device_streams():
            self.stop_all()
            return False

        # Start Device Manager after devices are ready
        if self.enable_device_manager:
            logger.info("Starting Device Manager for data aggregation...")
            self.device_manager_process = self._start_device_manager_process()
            if self.device_manager_process:
                self.device_manager_process.start()
                logger.info(
                    f"Started Device Manager (PID: {self.device_manager_process.pid})"
                )
            else:
                logger.error("Failed to start Device Manager")

        # Write robot home index to shared file for PolicyConnector
        self._write_robot_home_index()

        logger.info(f"Successfully started {started_count}/{len(self.devices)} devices")
        return True

    def stop_all(self) -> None:
        """Stop all running devices, visualizers, and the Device Manager.

        Direct children use a one-shot interrupt handler. Request their graceful
        shutdown explicitly, wait for cleanup, then use SIGTERM/SIGKILL only for
        survivors.
        """
        if getattr(self, "_shutdown_started", False):
            return
        has_managed_processes = bool(
            self.devices or self.device_manager_process or self.policy_runner_process
        )
        if not self.running and not has_managed_processes:
            return

        self._shutdown_started = True
        logger.info("Stopping all devices, visualizers, and Device Manager...")
        self.running = False

        # Create reset signal file to notify PolicyConnector to reconnect SHMs
        self._create_reset_signal()

        # Signal every physical device process. This also covers child processes
        # that are outside the terminal's foreground process group.
        managed_owner_names = {
            str(device_info["config"]["managed_by"])
            for device_info in self.devices
            if device_info["config"].get("managed_by")
        }
        for device_info in self.devices:
            owner_name = f"{device_info['device_class']}_{device_info['device_class_id']}"
            process = device_info.get("process")
            if process is None:
                continue
            try:
                if process.is_alive() and process.pid is not None:
                    logger.info(
                        f"Requesting graceful shutdown of device {owner_name}"
                    )
                    os.kill(process.pid, signal.SIGINT)
            except (AssertionError, OSError, ValueError) as exc:
                logger.warning(
                    f"Could not request graceful shutdown of {owner_name}: {exc}"
                )

        # The bridge can spend up to 30 seconds producing its optional trace
        # page after physics stops. Keep this longer budget scoped to managed
        # stream owners; ordinary hardware devices retain the historical wait.
        global_settings = self.config.get("global_settings", {})
        graceful_timeout = float(global_settings.get("device_shutdown_timeout", 10.0))
        managed_owner_timeout = max(
            graceful_timeout,
            float(global_settings.get("managed_device_owner_shutdown_timeout", 45.0)),
        )
        logger.info(
            f"Waiting up to {graceful_timeout:.1f}s for device processes to shut down gracefully "
            f"({managed_owner_timeout:.1f}s for managed-stream owners)..."
        )
        for device_info in self.devices:
            if device_info.get("process"):
                try:
                    owner_name = (
                        f"{device_info['device_class']}_{device_info['device_class_id']}"
                    )
                    timeout = (
                        managed_owner_timeout
                        if owner_name in managed_owner_names
                        else graceful_timeout
                    )
                    device_info["process"].join(timeout=timeout)
                except (AssertionError, ValueError):
                    pass

        # Phase 2: SIGTERM any survivors, then SIGKILL as last resort
        for device_info in self.devices:
            if device_info.get("process"):
                try:
                    if device_info["process"].is_alive():
                        logger.info(
                            f"Sending SIGTERM to {device_info['device_class']} device {device_info['device_class_id']}"
                        )
                        device_info["process"].terminate()
                except (AssertionError, ValueError):
                    pass

            if device_info.get("visualizer_process"):
                try:
                    if device_info["visualizer_process"].is_alive():
                        logger.info(
                            f"Stopping {device_info['device_class']} visualizer for device {device_info['device_class_id']}"
                        )
                        device_info["visualizer_process"].terminate()
                except (AssertionError, ValueError):
                    pass

        for device_info in self.devices:
            if device_info.get("process"):
                try:
                    device_info["process"].join(timeout=5.0)
                    if device_info["process"].is_alive():
                        logger.warning(
                            f"Force killing {device_info['device_class']} device {device_info['device_class_id']}"
                        )
                        device_info["process"].kill()
                except (AssertionError, ValueError):
                    pass

            if device_info.get("visualizer_process"):
                try:
                    device_info["visualizer_process"].join(timeout=5.0)
                    if device_info["visualizer_process"].is_alive():
                        logger.warning(
                            f"Force killing {device_info['device_class']} visualizer for device {device_info['device_class_id']}"
                        )
                        device_info["visualizer_process"].kill()
                except (AssertionError, ValueError):
                    pass

        # Stop Device Manager
        if self.device_manager_process:
            try:
                if self.device_manager_process.is_alive():
                    logger.info("Stopping Device Manager")
                    self.device_manager_process.terminate()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass

            try:
                self.device_manager_process.join(timeout=5.0)
                if self.device_manager_process.is_alive():
                    logger.warning("Force killing Device Manager")
                    self.device_manager_process.kill()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass

        # Stop policy runner
        if self.policy_runner_process:
            try:
                if self.policy_runner_process.is_alive():
                    logger.info("Stopping Policy Runner")
                    self.policy_runner_process.terminate()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass

            try:
                self.policy_runner_process.join(timeout=5.0)
                if self.policy_runner_process.is_alive():
                    logger.warning("Force killing Policy Runner")
                    self.policy_runner_process.kill()
            except (AssertionError, ValueError):
                # Process may have already terminated
                pass

        logger.info("All devices, visualizers, and Device Manager stopped")
        self._shutdown_complete = True

    def _create_reset_signal(self) -> None:
        """Create reset signal file to notify PolicyConnector to reconnect SHMs."""
        try:
            signal_files = self.config.get("runtime", {}).get("signal_files", {})
            reset_signal_path = signal_files.get("reset_signal")
            assert reset_signal_path is not None, (
                "reset_signal_path is required in signal_files"
            )
            reset_signal_file = Path(reset_signal_path)
            reset_signal_file.parent.mkdir(parents=True, exist_ok=True)
            reset_signal_file.touch()  # Update modification time
            logger.info(
                f"DeviceStarter: Created reset signal file at {reset_signal_path}"
            )
        except Exception as e:
            logger.warning(f"DeviceStarter: Failed to create reset signal file: {e}")

    def _write_robot_home_index(self) -> None:
        """Write robot home index to shared file for PolicyConnector to read."""
        try:
            robot_cfg = self.config.get("robot", {})
            home_index = robot_cfg.get("home", 0) if isinstance(robot_cfg, dict) else 0

            signal_files = self.config.get("runtime", {}).get("signal_files", {})
            home_index_path = signal_files.get("robot_home_index")
            assert home_index_path is not None, (
                "home_index_path is required in signal_files"
            )
            home_index_file = Path(home_index_path)
            home_index_file.parent.mkdir(parents=True, exist_ok=True)
            home_index_file.write_text(str(home_index))
            logger.info(
                f"DeviceStarter: Wrote robot home index {home_index} to {home_index_path}"
            )
        except Exception as e:
            logger.warning(f"DeviceStarter: Failed to write robot home index: {e}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        if getattr(self, "_shutdown_started", False):
            return
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(self._shutdown_exit_code())

    def get_status(self) -> Dict[str, Any]:
        """Get status of all managed devices, visualizers, and Device Manager."""
        # Parse device configs to get total count
        all_devices = self._parse_device_configs()

        status = {
            "running": self.running,
            "total_devices": len(all_devices),
            "enable_visualizers": self.enable_visualizers,
            "enable_device_manager": self.enable_device_manager,
            "device_manager_running": False,
            "device_manager_pid": None,
            "devices": [],
        }

        # Check Device Manager status
        if self.device_manager_process:
            try:
                status["device_manager_running"] = (
                    self.device_manager_process.is_alive()
                )
                status["device_manager_pid"] = self.device_manager_process.pid
            except (AssertionError, ValueError):
                status["device_manager_running"] = False
                status["device_manager_pid"] = None

        # Show configured devices
        for device_info in all_devices:
            device_status = {
                "type": device_info["device_class"],
                "id": device_info["device_class_id"],
                "configured": True,
                "running": False,
                "pid": None,
                "visualizer_running": False,
                "visualizer_pid": None,
                "restart_count": 0,
            }

            # If starter is running, check actual status
            if self.running and self.devices:
                # Find matching device in running devices
                for running_device in self.devices:
                    if (
                        running_device["device_class"] == device_info["device_class"]
                        and running_device["device_class_id"] == device_info["device_class_id"]
                    ):
                        device_status.update(
                            {
                                "running": running_device["process"].is_alive()
                                if running_device["process"]
                                else False,
                                "pid": running_device["process"].pid
                                if running_device["process"]
                                else None,
                                "visualizer_running": running_device[
                                    "visualizer_process"
                                ].is_alive()
                                if running_device["visualizer_process"]
                                else None,
                                "visualizer_pid": running_device[
                                    "visualizer_process"
                                ].pid
                                if running_device["visualizer_process"]
                                else None,
                                "restart_count": running_device["restart_count"],
                            }
                        )
                        break

            status["devices"].append(device_status)

        return status


def _flatten_config(cfg: DictConfig) -> Dict[str, Any]:
    """Convert DictConfig to plain dict for DeviceStarter."""
    return normalize_hydra_config(cfg)


def main() -> None:
    """Main function to run the device starter using Hydra."""
    import hydra
    from omegaconf import DictConfig
    import signal
    import sys

    # Pre-process custom flags before Hydra parses arguments
    # Support -v / --visualize flag for enabling visualizers
    if "-v" in sys.argv or "--visualize" in sys.argv:
        # Remove the flag from argv
        if "-v" in sys.argv:
            sys.argv.remove("-v")
        if "--visualize" in sys.argv:
            sys.argv.remove("--visualize")
        # Add Hydra override
        sys.argv.append("runtime.enable_visualizers=True")

    @hydra.main(
        config_path="configs",
        config_name=DEFAULT_HYDRA_CONFIG_NAME,
        version_base="1.3",
    )
    def hydra_main(cfg: DictConfig) -> None:
        merged_cfg = _flatten_config(cfg)
        runtime_cfg = merged_cfg.get("runtime", {})

        try:
            process_guard = DeviceStarterProcessGuard(merged_cfg, argv=sys.argv)
            process_guard.acquire()
            try:
                endpoints = check_gripper_websocket_preflight(merged_cfg)
                for endpoint in endpoints:
                    logger.info(
                        "LK gripper preflight passed: %s, exclusive_client=%s",
                        endpoint.uri,
                        endpoint.exclusive_client,
                    )

                starter = DeviceStarter(
                    config=merged_cfg,
                    config_path=None,
                    enable_visualizers=as_bool(
                        runtime_cfg.get("enable_visualizers", False)
                    ),
                    enable_device_manager=as_bool(
                        runtime_cfg.get("enable_device_manager", True)
                    ),
                )

                def _shutdown(signum: int, frame: Any) -> None:
                    if getattr(starter, "_shutdown_started", False):
                        return
                    logger.info(f"Received signal {signum}, stopping DeviceStarter")
                    starter.stop_all()
                    sys.exit(starter._shutdown_exit_code())

                signal.signal(signal.SIGINT, _shutdown)
                signal.signal(signal.SIGTERM, _shutdown)

                starter.cpu_affinity.apply("device_starter_main")

                try:
                    if not starter.start_all():
                        raise SystemExit(1)
                    if not starter._monitor_devices():
                        raise SystemExit(1)
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                finally:
                    starter.stop_all()
            finally:
                process_guard.release()
        except DeviceStarterGuardError as exc:
            logger.error(f"DeviceStarter startup blocked: {exc}")
            raise SystemExit(2) from exc

    hydra_main()


if __name__ == "__main__":
    main()
