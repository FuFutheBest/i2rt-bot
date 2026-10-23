#!/usr/bin/env python3
"""
Device Starter (Windows Compatible)

This module provides centralized management for running multiple devices simultaneously
using separate processes. It is optimized for Windows 'spawn' method.

Author: Junjie Xu,Assistant
"""

import os
import sys
import time
import signal
import multiprocessing as mp
from typing import Dict, List, Any, Optional
import inspect
import logging

# -----------------------------------------------------------------------------
# CONFIG & LOGGING SETUP (Placeholder imports to make code runnable)
# -----------------------------------------------------------------------------
try:
    # Add devices to path
    sys.path.append(os.path.dirname(__file__))
    from devices import DEVICE_CLASSES
    from manager import MANAGER_CLASSES
    from utils.device_readiness import (
        DEFAULT_DEVICE_READY_TIMEOUT_S,
        DeviceReadinessError,
        wait_for_device_streams,
    )
    from utils.shm_utils import get_dtype
    from utils.config_parser import ensure_config_dict, parse_device_configs_with_fields, load_config
    from utils.logger_config import logger
except ImportError:
    # Fallback for demonstration if utils are missing
    logging.basicConfig(level=logging.INFO, format='[%(processName)s] %(message)s')
    logger = logging.getLogger("DeviceStarter")

# =============================================================================
# WORKER FUNCTIONS (Must be at module level for Windows Pickling)
# =============================================================================

def _worker_device_run(device_info: Dict[str, Any]) -> None:
    """Worker function to run a specific device."""
    device = None
    try:
        # Re-import inside process to ensure clean state (Windows requirement)
        from devices import DEVICE_CLASSES
        device_class = DEVICE_CLASSES.get(device_info['device_class'])
        if not device_class:
            raise ValueError(f"Unknown device class: {device_info['device_class']}")

        from utils.device_config import parse_device_config

        # Use the validated structured contract produced by the shared parser.
        parsed_config = device_info.get("parsed_config")
        if parsed_config is None:
            parsed_config = parse_device_config(
                device_info["config"],
                device_kind=DEVICE_CLASSES.kind(device_info["device_class"]),
            )
        device_config = parsed_config.constructor_kwargs()

        sig = inspect.signature(device_class.__init__)
        valid_params = list(sig.parameters.keys())
        if "state_feedback" not in valid_params:
            raise TypeError(
                f"{device_info['device_class']} must declare state_feedback"
            )
        if (
            DEVICE_CLASSES.kind(device_info["device_class"]).writable
            and "control" not in valid_params
        ):
            raise TypeError(
                f"{device_info['device_class']} must declare control"
            )
        tracking_config = device_config.pop("tracking", None)
        if tracking_config is not None:
            if "tracking" in valid_params:
                device_config["tracking"] = tracking_config
            elif isinstance(tracking_config, dict):
                merged_config = tracking_config.copy()
                merged_config.update(device_config)
                device_config = merged_config
        filtered_config = {k: v for k, v in device_config.items() if k in valid_params}

        # Initialize Device
        device = device_class(**filtered_config)
        logger.info(f"Starting {device_info['device_class']} {device_info['config'].get('device_class_id', '')}")
        device.start_server()

    finally:
        if device:
            logger.info(f"Cleaning up {device_info['device_class']}...")
            try:
                if hasattr(device, 'close'):
                    device.close()
                elif hasattr(device, 'stop'):
                    device.stop()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
        logger.info(f"{device_info['device_class']} process exited.")

def _worker_visualizer_run(device_info: Dict[str, Any]) -> None:
    """Worker function to run a visualizer."""
    try:
        # Plotting packages are optional and only needed in the visualizer process.
        import matplotlib

        matplotlib.use('TkAgg')
        from visualizers import VISUALIZER_CLASSES
        visualizer_class = VISUALIZER_CLASSES.get(device_info['device_class'])
        if not visualizer_class:
            return

        device_name = device_info['device_class']
        shared_memory_name = f"{device_name}_{device_info['device_class_id']}_data"
        from utils.device_config import state_feedback_mapping

        state_feedback = state_feedback_mapping(device_info["config"])
        update_interval = int(1000 / float(state_feedback["fps"]))
        raw_format = state_feedback.get("format")
        format_dtype = raw_format.get("dtype") if isinstance(raw_format, dict) else None
        data_dtype = get_dtype(format_dtype or state_feedback.get("data_dtype", "float32"))

        # Get enable_overlay from device config, fallback to global default from runtime config
        # This allows per-device override while having a clear global default
        camera_overlay_default = device_info['config'].get('_camera_overlay_default', False)
        enable_overlay = device_info['config'].get('enable_overlay', camera_overlay_default)

        visualizer = visualizer_class(
            shared_memory_name=shared_memory_name,
            data_dtype=data_dtype,
            smoothing_window=5,
            device_info=device_info,
            enable_overlay=enable_overlay,
        )

        # Retry logic for connection
        for attempt in range(10):
            if visualizer.connect():
                break
            time.sleep(0.5)
        else:
            logger.error(f"Visualizer {device_info['device_class_id']} failed to connect")
            return

        # Start Visualization
        visualizer.start_visualization(update_interval=update_interval)

    except Exception as e:
        logger.error(f"Visualizer error: {e}")

def _worker_device_manager_run(config: Dict[str, Any], config_path: Optional[str]) -> None:
    """Worker function for Device Manager."""
    try:
        from utils.config_parser import ensure_config_dict
        from manager import MANAGER_CLASSES
        device_manager_config = config.get('device_manager', {})
        manager_type = device_manager_config.get('type', 'base')
        manager_class = MANAGER_CLASSES.get(manager_type)

        manager_params = device_manager_config.get('params', {}) or {}
        manager_params = ensure_config_dict(manager_params)

        if config_path and not manager_params.get('config_path'):
            manager_params.setdefault('config_path', config_path)
        else:
            manager_params.setdefault('config', config)

        manager = manager_class(**manager_params)
        logger.info(f"Device Manager ({manager_type}) started")

        manager.start()

    except Exception as e:
        logger.error(f"Device Manager error: {e}")

def _worker_policy_runner_run(config: Dict[str, Any], config_path: Optional[str]) -> None:
    """Worker function for Policy Runner."""
    try:
        policy_runner_config = config.get('action_executor', {})
        runner_type = policy_runner_config.get('type', 'base_policy_runner')

        from manager import MANAGER_CLASSES
        manager_class = MANAGER_CLASSES.get(runner_type)

        valid_params = {
            'config_path', 'robot_control_shms',
            'execution_fps', 'chunk_length', 'chunk_manager', 'config'
        }
        runner_config = {k: v for k, v in policy_runner_config.items() if k in valid_params}

        if config_path and not runner_config.get('config_path'):
            runner_config.setdefault('config_path', config_path)
        else:
            runner_config.setdefault('config', config)

        runner = manager_class(**runner_config)
        logger.info(f"Policy Runner started")

        runner.start()

    except Exception as e:
        logger.error(f"Policy Runner error: {e}")

# =============================================================================
# MAIN CLASS
# =============================================================================

class DeviceStarter:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: Optional[str] = None,
        enable_visualizers: bool = False,
        enable_device_manager: bool = True,
    ) -> None:
        # Ensure imports are available (important for Windows multiprocessing)
        try:
            from utils.config_parser import ensure_config_dict, load_config
        except ImportError:
            logger.error("Failed to import config_parser utilities")
            raise

        if config is None:
            resolved_path = config_path or "config.yaml"
            self.config_path = resolved_path
            self.config = load_config(resolved_path)
        else:
            self.config_path = config_path
            self.config = ensure_config_dict(config)

        self.devices: List[Dict[str, Any]] = []
        self.running = False
        self.enable_visualizers = enable_visualizers
        self.enable_device_manager = enable_device_manager
        self.device_manager_process = None
        self.policy_runner_process = None

        # Get global default for camera overlay from runtime config
        runtime_cfg = self.config.get("runtime", {})
        self.camera_overlay_enabled_default = runtime_cfg.get("camera_overlay_enabled", False)

        logger.info("DeviceStarter initialized")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _parse_device_configs(self) -> List[Dict[str, Any]]:
        # Ensure import is available
        from utils.config_parser import parse_device_configs_with_fields

        additional_fields = {
            'process': None,
            'visualizer_process': None,
            'restart_count': 0,
            'last_restart': None
        }
        return parse_device_configs_with_fields(self.config, additional_fields)

    def _start_generic_device_process(self, device_info: Dict[str, Any]) -> Optional[mp.Process]:
        try:
            process = mp.Process(
                target=_worker_device_run,
                args=(device_info,),
                name=f"{device_info['device_class']}-{device_info['config']['device_class_id']}"
            )
            return process
        except Exception as e:
            logger.error(f"Failed to create device process: {e}")
            return None

    def _start_generic_visualizer_process(self, device_info: Dict[str, Any]) -> Optional[mp.Process]:
        try:
            # Add global default to device_info if not already set, so _worker_visualizer_run can use it
            # Create a copy to avoid modifying the original
            device_info_copy = device_info.copy()
            device_info_copy['config'] = device_info['config'].copy()
            if 'enable_overlay' not in device_info_copy['config']:
                device_info_copy['config']['_camera_overlay_default'] = self.camera_overlay_enabled_default

            process = mp.Process(
                target=_worker_visualizer_run,
                args=(device_info_copy,),
                name=f"{device_info['device_class']}_vis"
            )
            process.start()
            return process
        except Exception as e:
            logger.error(f"Failed to create visualizer process: {e}")
            return None

    def _start_device_manager_process(self) -> Optional[mp.Process]:
        try:
            process = mp.Process(
                target=_worker_device_manager_run,
                args=(self.config, self.config_path),
                name="DeviceManager"
            )
            return process
        except Exception as e:
            logger.error(f"Failed to create Device Manager process: {e}")
            return None

    def _start_policy_runner_process(self) -> Optional[mp.Process]:
        try:
            process = mp.Process(
                target=_worker_policy_runner_run,
                args=(self.config, self.config_path),
                name="PolicyRunner"
            )
            return process
        except Exception as e:
            logger.error(f"Failed to create Policy Runner process: {e}")
            return None

    def _start_device_process(self, device_info: Dict[str, Any]) -> bool:
        process = self._start_generic_device_process(device_info)
        if not process: return False

        process.start()
        device_info['process'] = process
        device_info['last_restart'] = time.time()

        if self.enable_visualizers:
            time.sleep(1.0)
            vis_process = self._start_generic_visualizer_process(device_info)
            if vis_process:
                device_info['visualizer_process'] = vis_process

        return True

    def _monitor_devices(self) -> None:
        logger.info("Starting device monitoring...")
        try:
            while self.running:
                for device_info in self.devices:
                    proc = device_info['process']
                    if proc and not proc.is_alive():
                        exitcode = proc.exitcode
                        if exitcode !=0:
                            logger.error(f"Device {device_info['device_class']} died with exit code {exitcode}")
                            logger.error("stopping the system")
                            self.stop_all()
                            sys.exit(1)

                time.sleep(1.0)
        except KeyboardInterrupt:
            # Main thread caught Ctrl+C during sleep
            self.stop_all()

    def start_all(self) -> None:
        if self.running: return
        self.devices = self._parse_device_configs()

        # Start Devices
        for device_info in self.devices:
            self._start_device_process(device_info)

        try:
            wait_for_device_streams(
                self.devices,
                timeout_s=DEFAULT_DEVICE_READY_TIMEOUT_S,
            )
        except DeviceReadinessError as exc:
            logger.error(f"Device startup readiness failed: {exc}")
            self.running = True
            self.stop_all()
            return

        # Start Managers
        if self.enable_device_manager:
            self.device_manager_process = self._start_device_manager_process()
            if self.device_manager_process: self.device_manager_process.start()

        action_cfg = self.config.get('action_executor', {})
        if action_cfg and action_cfg.get('type') not in ['disabled', None]:
            self.policy_runner_process = self._start_policy_runner_process()
            if self.policy_runner_process: self.policy_runner_process.start()

        self.running = True
        logger.info("System Started.")

    def stop_all(self) -> None:
        """Stop all processes."""
        if not self.running: return

        logger.info("Stopping system...")
        self.running = False

        # List of all processes to wait for
        all_procs = []
        for d in self.devices:
            if d['process']: all_procs.append(d['process'])
            if d['visualizer_process']: all_procs.append(d['visualizer_process'])
        if self.device_manager_process: all_procs.append(self.device_manager_process)
        if self.policy_runner_process: all_procs.append(self.policy_runner_process)

        # Wait for processes to finish
        wait_timeout = 5.0
        end_time = time.time() + wait_timeout

        for proc in all_procs:
            if proc.is_alive():
                remain = max(0, end_time - time.time())
                proc.join(timeout=remain)

        # Force kill if still alive
        for proc in all_procs:
            if proc.is_alive():
                logger.warning(f"Process {proc.name} stuck. Force killing...")
                try:
                    proc.terminate()
                    proc.join(timeout=1)
                except:
                    pass

        logger.info("System Stopped.")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}")
        self.stop_all()
        sys.exit(0)

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def _flatten_config(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True)

def main() -> None:
    import hydra
    from omegaconf import DictConfig

    # Arg parsing logic... (same as original)
    if '-v' in sys.argv or '--visualize' in sys.argv:
        if '-v' in sys.argv: sys.argv.remove('-v')
        if '--visualize' in sys.argv: sys.argv.remove('--visualize')
        sys.argv.append('runtime.enable_visualizers=True')

    @hydra.main(config_path="configs", config_name="config", version_base="1.3")
    def hydra_main(cfg: DictConfig) -> None:
        merged_cfg = _flatten_config(cfg)
        runtime_cfg = merged_cfg.get("runtime", {})

        # Fix nested configs logic... (same as original)
        if "devices" in merged_cfg and isinstance(merged_cfg["devices"], dict):
            merged_cfg["devices"] = merged_cfg["devices"].get("devices", [])

        starter = DeviceStarter(
            config=merged_cfg,
            enable_visualizers=runtime_cfg.get("enable_visualizers", False),
            enable_device_manager=runtime_cfg.get("enable_device_manager", True),
        )

        try:
            starter.start_all()
            starter._monitor_devices()
        except KeyboardInterrupt:
            # Double insurance for Ctrl+C
            starter.stop_all()
        finally:
            starter.stop_all()

    hydra_main()

if __name__ == "__main__":
    mp.freeze_support()
    main()
