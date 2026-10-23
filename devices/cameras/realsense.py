#!/usr/bin/env python3
"""Camera device implementation split from the legacy camera module.

Author: Jun Lv, SuperInference
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None

import cv2
import numpy as np
import psutil

from devices.cameras.base import BaseCameraDevice
from devices.config import StateFeedbackConfig
from devices.cameras.realsense_sdk import REALSENSE_CAMERA_AVAILABLE, rs
from utils.logger_config import logger
from utils.time_control import precise_loop_timing

total_cores = psutil.cpu_count()
# Assign the last 8 cores to the server.
num_cores_to_bind = 8
cores_to_bind = set(range(total_cores - num_cores_to_bind, total_cores))
os.sched_setaffinity(0, cores_to_bind)

def _get_realsense_reset_paths() -> Tuple[str, str]:
    """Build process-group-scoped lock and done files for reset synchronization."""
    session_id = str(os.getppid())
    lock_path = f"/tmp/superinference_rs_reset_{session_id}.lock"
    done_path = f"/tmp/superinference_rs_reset_{session_id}.done"
    return lock_path, done_path


def _run_realsense_hardware_reset(wait_seconds: float = 3.0) -> None:
    """Execute RealSense hardware reset for all currently connected devices."""
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        logger.warning("No RealSense devices detected during one-time reset")
        return

    logger.info(f"Running one-time RealSense hardware reset for {len(devices)} device(s)")
    for device in devices:
        try:
            serial = device.get_info(rs.camera_info.serial_number)
        except Exception:
            serial = "unknown"
        device.hardware_reset()
        logger.info(f"RealSense hardware reset triggered for serial={serial}")

    time.sleep(wait_seconds)
    logger.info(f"RealSense reset stabilization wait complete ({wait_seconds:.1f}s)")


def _synchronize_realsense_reset(
    capture_id: int,
    wait_seconds: float = 3.0,
    wait_timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.1,
) -> None:
    """
    Synchronize reset across camera processes.

    Rules:
    - Only physical capture_id == 0 performs the hardware reset.
    - Other camera processes block until reset is completed.
    """
    lock_path, done_path = _get_realsense_reset_paths()

    if capture_id == 0:
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if os.path.exists(done_path):
                logger.info("RealSense reset already completed by device 0")
                return
            try:
                _run_realsense_hardware_reset(wait_seconds=wait_seconds)
            finally:
                with open(done_path, "w", encoding="utf-8") as done_file:
                    done_file.write(str(time.time_ns()))
    else:
        wait_start = time.monotonic()
        logger.info(
            f"Camera {capture_id} waiting for device 0 to finish RealSense reset"
        )
        while not os.path.exists(done_path):
            if (time.monotonic() - wait_start) > wait_timeout_seconds:
                raise RuntimeError(
                    f"Timeout waiting for RealSense reset done signal from device 0 "
                    f"(waited {wait_timeout_seconds:.1f}s)"
                )
            time.sleep(poll_interval_seconds)
        logger.info(f"Camera {capture_id} detected RealSense reset completion")


class OpenCVCameraDevice(BaseCameraDevice):
    """
    Camera device class that keeps OpenCV-compatible naming while using RealSense backend.
    Supports continuous memory buffer for multiple frames and hardware camera parameters.
    """

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        capture_id: int = 0,
        realsense_serial: Optional[str] = None,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        hue: Optional[float] = None,
        gamma: Optional[float] = None,
        sharpness: Optional[float] = None
    ) -> None:
        """
        Initialize the camera device.

        Args:
            device_class_id: Logical camera ID used for shared memory naming
            data_shape: Shape of camera frames (height, width) for grayscale or (height, width, channels) for color
            fps: Target frames per second for capture
            data_dtype: Data type for camera frames (string or numpy dtype, default: np.uint8)
            buffer_size: Number of frames to store in buffer (default: 1)
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
            capture_id: Physical camera enumeration index
            realsense_serial: RealSense serial number (takes precedence over capture_id)
            brightness: Camera brightness setting (if None, not set)
            contrast: Camera contrast setting (if None, not set)
            saturation: Camera saturation setting (if None, not set)
            hue: Camera hue setting (if None, not set)
            gamma: Camera gamma setting (if None, not set)
            sharpness: Camera sharpness setting (if None, not set)

        Raises:
            RuntimeError: If camera initialization fails
        """
        if not REALSENSE_CAMERA_AVAILABLE:
            raise RuntimeError("RealSense camera modules are not available. Please install pyrealsense2.")

        super().__init__(device_class_id=device_class_id, state_feedback=state_feedback)

        # Override device name for OpenCV camera
        self.device_name = "OpenCVCameraDevice"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        # Camera specific attributes
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.pipeline_profile: Optional[rs.pipeline_profile] = None
        self.color_sensor: Optional[rs.sensor] = None
        self.realsense_serial = realsense_serial
        self.capture_id = int(capture_id)
        self.is_color = len(self.data_shape) == 3 and self.data_shape[2] == 3
        self._resize_warned = False  # Flag to prevent repeated resize warnings

        # Camera hardware parameters (None means not set)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.sharpness = sharpness

        # Reset is executed by device 0 once; other camera processes wait.
        _synchronize_realsense_reset(capture_id=self.capture_id)

        # Initialize camera immediately
        if not self._initialize_camera():
            raise RuntimeError(f"Failed to initialize camera {device_class_id}")

    def _initialize_camera(self) -> bool:
        """Initialize RealSense pipeline and set hardware parameters."""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Set basic camera properties
            fps = max(1, int(round(self.fps)))
            if self.is_color:
                height, width, _ = self.data_shape
            else:
                height, width = self.data_shape

            if self.realsense_serial:
                self.config.enable_device(self.realsense_serial)
            else:
                available_devices = list(rs.context().query_devices())
                if self.capture_id < 0 or self.capture_id >= len(available_devices):
                    raise RuntimeError(
                        f"Requested capture_id={self.capture_id}, but only {len(available_devices)} "
                        "RealSense device(s) are connected"
                    )
                auto_selected_serial = available_devices[self.capture_id].get_info(rs.camera_info.serial_number)
                self.config.enable_device(auto_selected_serial)
            logger.info(f"Enabled device {self.realsense_serial or auto_selected_serial}")

            stream_format = rs.format.rgb8 if self.is_color else rs.format.rgb8
            self.config.enable_stream(rs.stream.color, width, height, stream_format, fps)
            self.pipeline_profile = self.pipeline.start(self.config)
            device = self.pipeline_profile.get_device()
            self.color_sensor = self._get_color_sensor(device)
            # Set hardware camera parameters if specified
            self._set_camera_parameters()
            # Warmup and test capture
            for _ in range(3):
                self.pipeline.wait_for_frames()
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Cannot capture frame from camera")
            frame = np.asanyarray(color_frame.get_data())

            logger.info(
                "Camera %s initialized: frame=%s target=%s serial=%s",
                self.device_class_id,
                frame.shape,
                self.data_shape,
            )
            self._log_camera_parameters()
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self._cleanup_camera()
            return False

    @staticmethod
    def _safe_get_info(device: rs.device, info_key: rs.camera_info) -> Optional[str]:
        """Get device information safely."""
        try:
            return device.get_info(info_key)
        except Exception:
            return None

    @staticmethod
    def _get_color_sensor(device: rs.device) -> Optional[rs.sensor]:
        """Get color sensor from RealSense device."""
        try:
            sensors = list(device.query_sensors())
        except Exception:
            sensors = []

        for sensor in sensors:
            try:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    return sensor
            except Exception:
                continue
        return sensors[0] if sensors else None

    def _set_camera_parameters(self) -> None:
        """Set hardware camera parameters using RealSense sensor options."""
        if not self.color_sensor:
            return

        option_map = {
            "brightness": rs.option.brightness,
            "contrast": rs.option.contrast,
            "saturation": rs.option.saturation,
            "hue": rs.option.hue,
            "gamma": rs.option.gamma,
            "sharpness": rs.option.sharpness,
        }
        for attr, option in option_map.items():
            value = getattr(self, attr)
            if value is None:
                continue
            try:
                if not self.color_sensor.supports(option):
                    logger.warning(f"Camera {self.device_class_id} does not support {attr}")
                    continue
                self.color_sensor.set_option(option, float(value))
            except Exception as exc:
                logger.warning(f"Failed to set {attr}={value}: {exc}")

    def _log_camera_parameters(self) -> None:
        """Log the current camera parameters."""
        params = []
        option_map = {
            "brightness": rs.option.brightness,
            "contrast": rs.option.contrast,
            "saturation": rs.option.saturation,
            "hue": rs.option.hue,
            "gamma": rs.option.gamma,
            "sharpness": rs.option.sharpness,
        }
        if self.color_sensor:
            for name, option in option_map.items():
                if self.color_sensor.supports(option):
                    try:
                        params.append(f"{name}={self.color_sensor.get_option(option):.2f}")
                    except Exception:
                        params.append(f"{name}=unavailable")
                else:
                    params.append(f"{name}=unsupported")

        if params:
            logger.info(f"Camera {self.device_class_id} hardware parameters: {', '.join(params)}")
        else:
            logger.info(f"Camera {self.device_class_id}: No hardware parameters set")

    def _capture_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Capture a frame from the camera and convert to the required format.

        Returns:
            tuple: (timestamp_ns, numpy.ndarray) or (None, None) if capture failed
        """
        if not self.pipeline:
            return None, None

        try:
            # Get timestamp immediately when starting to capture
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            timestamp_ns = time.time_ns()
            if not color_frame:
                logger.warning("Failed to capture frame")
                return None, None
            try:
                timestamp_ns = int(color_frame.get_timestamp() * 1_000_000)
            except Exception:
                pass
            frame = np.asanyarray(color_frame.get_data())

            # Convert frame to required format
            if self.is_color:
                # Resize if necessary
                target_height, target_width, _ = self.data_shape
                if frame.shape[:2] != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(f"Resizing frame from {frame.shape[:2]} to ({target_height}, {target_width}). "
                                      f"This adds computational overhead. Consider setting camera resolution to match target shape.")
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))

                # Ensure 3 channels
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:  # RGBA -> RGB
                    frame = frame[:, :, :3]

            else:
                # For grayscale images
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Resize if necessary
                target_height, target_width = self.data_shape
                if frame.shape != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(f"Resizing frame from {frame.shape} to ({target_height}, {target_width}). "
                                      f"This adds computational overhead. Consider setting camera resolution to match target shape.")
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))

            # Convert to required data type
            frame = frame.astype(self.data_dtype)

            return timestamp_ns, frame

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None, None

    def _cleanup_camera(self) -> None:
        """Clean up camera resources."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as exc:
                logger.warning(f"Failed to stop camera pipeline: {exc}")
            self.pipeline = None
            self.pipeline_profile = None
            self.config = None
            self.color_sensor = None
            logger.info("Camera pipeline released")

    def start_server(self) -> None:
        """Start the camera device server and run frame capture loop."""
        if self.running:
            logger.info(f"Camera {self.device_class_id} is already running")
            return

        logger.info(f"Starting camera server {self.device_class_id}...")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(f"Camera server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main frame capture loop
        while self.running:
            try:
                timestamp_ns, frame_array = self._capture_frame()
                if frame_array is not None:
                    self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                break

    def stop_server(self) -> None:
        """Stop the camera device server."""
        if not self.running:
            return

        logger.info(f"Stopping camera server {self.device_class_id}...")
        self.running = False
        self._cleanup_shared_memory()
        logger.info("Camera server stopped")

    def close(self) -> None:
        """Close the camera device and release all resources."""
        self.stop_server()
        self._cleanup_camera()
        logger.info(f"Camera {self.device_class_id} closed")

    def __del__(self) -> None:
        """Destructor to ensure camera and shared memory resources are released."""
        try:
            self._cleanup_camera()
            # Also cleanup shared memory from parent class
            if hasattr(self, 'shared_memory') and self.shared_memory:
                self._cleanup_shared_memory()
        except Exception:
            pass  # Ignore errors during cleanup in destructor

    def get_device_info(self) -> Dict[str, Any]:
        """Get camera device information."""
        info = super().get_device_info()
        info.update({
            "camera_id": self.device_class_id,  # camera_id is same as device_class_id
            "is_color": self.is_color,
            "camera_opened": self.pipeline is not None,
            "requested_realsense_serial": self.realsense_serial,
            "hardware_parameters": {
                "brightness": self.brightness,
                "contrast": self.contrast,
                "saturation": self.saturation,
                "hue": self.hue,
                "gamma": self.gamma,
                "sharpness": self.sharpness
            }
        })
        return info


def main() -> None:
    """Main function to run the OpenCV camera device server."""
    import argparse

    parser = argparse.ArgumentParser(description="OpenCV Camera Device Server")
    parser.add_argument("--device-class-id", "-d", type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--capture-id", type=int, default=0,
                        help="Physical RealSense enumeration index")
    parser.add_argument("--shape", "-s", type=str, default="240,320",
                        help="Frame shape as comma-separated values (default: 240,320)")
    parser.add_argument("--fps", "-f", type=float, default=15.0,
                        help="Frames per second (default: 15.0)")
    parser.add_argument("--dtype", "-t", default="uint8",
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                                'float32', 'float64'],
                        help="Data type (default: uint8)")
    parser.add_argument("--buffer-size", "-b", type=int, default=1,
                        help="Buffer size in frames (default: 1)")
    parser.add_argument("--realsense-serial", "-r", type=str, default=None,
                        help="RealSense serial number (if not specified, uses capture_id index)")
    parser.add_argument("--color", "-c", action="store_true",
                        help="Enable color capture (3 channels)")
    parser.add_argument("--brightness", "-B", type=float, default=None,
                        help="Camera brightness setting (if not specified, not set)")
    parser.add_argument("--contrast", "-C", type=float, default=None,
                        help="Camera contrast setting (if not specified, not set)")
    parser.add_argument("--saturation", "-S", type=float, default=None,
                        help="Camera saturation setting (if not specified, not set)")
    parser.add_argument("--hue", "-H", type=float, default=None,
                        help="Camera hue setting (if not specified, not set)")
    parser.add_argument("--gamma", "-g", type=float, default=None,
                        help="Camera gamma setting (if not specified, not set)")
    parser.add_argument("--sharpness", "-X", type=float, default=None,
                        help="Camera sharpness setting (if not specified, not set)")

    args = parser.parse_args()

    # Parse shape string to tuple
    try:
        data_shape = tuple(int(x.strip()) for x in args.shape.split(','))
        if args.color and len(data_shape) == 2:
            data_shape = data_shape + (3,)  # Add color channel
        elif not args.color and len(data_shape) == 3:
            data_shape = data_shape[:2]  # Remove color channel
    except ValueError:
        logger.error("Invalid shape format. Use comma-separated integers (e.g., '240,320' or '240,320,3')")
        return

    # Import common utilities
    from utils.shm_utils import get_dtype

    data_dtype = get_dtype(args.dtype)

    # Create camera device
    camera = OpenCVCameraDevice(
        device_class_id=args.device_class_id,
        capture_id=args.capture_id,
        state_feedback={
            "fps": args.fps,
            "buffer_size": args.buffer_size,
            "hardware_latency_ms": 0.0,
            "format": {"shape": list(data_shape), "dtype": args.dtype, "type": "rgb"},
        },
        realsense_serial=args.realsense_serial,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        gamma=args.gamma,
        sharpness=args.sharpness
    )

    logger.info("OpenCV Camera Device Server")
    logger.info("===========================")
    logger.info(f"Camera ID: {args.device_class_id}")
    logger.info(f"Frame shape: {data_shape}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Buffer size: {args.buffer_size} frames")
    logger.info(f"Color: {args.color}")
    logger.info(f"RealSense serial: {args.realsense_serial if args.realsense_serial else 'Auto by capture_id'}")
    logger.info("Camera Hardware Parameters:")
    logger.info(f"  Brightness: {args.brightness if args.brightness is not None else 'Not set'}")
    logger.info(f"  Contrast: {args.contrast if args.contrast is not None else 'Not set'}")
    logger.info(f"  Saturation: {args.saturation if args.saturation is not None else 'Not set'}")
    logger.info(f"  Hue: {args.hue if args.hue is not None else 'Not set'}")
    logger.info(f"  Gamma: {args.gamma if args.gamma is not None else 'Not set'}")
    logger.info(f"  Sharpness: {args.sharpness if args.sharpness is not None else 'Not set'}")
    logger.info("")

    try:
        logger.info("Camera server is running. Press Ctrl+C to stop...")
        logger.info(f"Camera info: {camera.get_device_info()}")
        camera.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
