#!/usr/bin/env python3
"""Camera device implementation split from the legacy camera module.

Author: Jun Lv, SuperInference
"""

import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import psutil

from devices.cameras.base import BaseCameraDevice
from devices.config import StateFeedbackConfig
from utils.logger_config import logger
from utils.time_control import precise_loop_timing

total_cores = psutil.cpu_count()
# Assign the last 8 cores to the server.
num_cores_to_bind = 8
cores_to_bind = set(range(total_cores - num_cores_to_bind, total_cores))
os.sched_setaffinity(0, cores_to_bind)

class WebCameraDevice(BaseCameraDevice):
    """
    Web camera device class that captures video from a USB/built-in webcam via
    OpenCV's VideoCapture backend and writes frames to shared memory.

    Mirrors the lifecycle of OpenCVCameraDevice but uses cv2.VideoCapture
    instead of the RealSense pipeline.
    """

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        capture_id: Optional[Union[int, str]] = None,
        backend: Optional[int] = None,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        hue: Optional[float] = None,
        gamma: Optional[float] = None,
        sharpness: Optional[float] = None,
    ) -> None:
        """
        Args:
            device_class_id: Logical camera ID used for shared memory naming.
            state_feedback: Required RGB stream shape, dtype, rate, and buffering.
            capture_id: OS index or device path passed to cv2.VideoCapture.
                Required: physical capture never falls back to the logical ID.
            backend: Optional OpenCV backend (e.g. cv2.CAP_V4L2, cv2.CAP_DSHOW).
            brightness/contrast/saturation/hue/gamma/sharpness: Optional
                hardware parameters; None means leave at the camera default.
        """
        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
        )

        self.device_name = "WebCameraDevice"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        self.is_color = len(self.data_shape) == 3 and self.data_shape[2] == 3
        self._resize_warned = False

        from utils.device_identity import require_capture_id

        self.capture_id = require_capture_id(capture_id)
        self.backend = backend
        self.cap: Optional[cv2.VideoCapture] = None

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.sharpness = sharpness

        if not self._initialize_camera():
            raise RuntimeError(f"Failed to initialize web camera {device_class_id}")

    def _initialize_camera(self) -> bool:
        """Open cv2.VideoCapture and apply requested settings."""
        try:
            if self.backend is not None:
                self.cap = cv2.VideoCapture(self.capture_id, self.backend)
            else:
                self.cap = cv2.VideoCapture(self.capture_id)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(
                    f"cv2.VideoCapture could not open index {self.capture_id}"
                )

            if self.is_color:
                height, width, _ = self.data_shape
            else:
                height, width = self.data_shape

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            self.cap.set(cv2.CAP_PROP_FPS, float(max(1.0, self.fps)))

            self._set_camera_parameters()

            for _ in range(3):
                self.cap.read()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Cannot capture warm-up frame from web camera")

            logger.info(
                "WebCamera %s initialized: capture_id=%s frame=%s target=%s",
                self.device_class_id,
                self.capture_id,
                frame.shape,
                self.data_shape,
            )
            self._log_camera_parameters()
            return True

        except Exception as e:
            logger.error(f"Web camera initialization failed: {e}")
            self._cleanup_camera()
            return False

    _PROP_MAP = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "hue": cv2.CAP_PROP_HUE,
        "gamma": cv2.CAP_PROP_GAMMA,
        "sharpness": cv2.CAP_PROP_SHARPNESS,
    }

    def _set_camera_parameters(self) -> None:
        if not self.cap:
            return
        for attr, prop in self._PROP_MAP.items():
            value = getattr(self, attr)
            if value is None:
                continue
            try:
                if not self.cap.set(prop, float(value)):
                    logger.warning(
                        f"WebCamera {self.device_class_id} did not accept {attr}={value}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to set {attr}={value}: {exc}")

    def _log_camera_parameters(self) -> None:
        if not self.cap:
            return
        params = []
        for name, prop in self._PROP_MAP.items():
            try:
                params.append(f"{name}={self.cap.get(prop):.2f}")
            except Exception:
                params.append(f"{name}=unavailable")
        if params:
            logger.info(
                f"WebCamera {self.device_class_id} hardware parameters: {', '.join(params)}"
            )

    def _capture_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Read a frame from VideoCapture and convert to the required format."""
        if not self.cap:
            return None, None
        try:
            ok, frame = self.cap.read()
            timestamp_ns = time.time_ns()
            if not ok or frame is None:
                logger.warning("Failed to capture frame from web camera")
                return None, None

            if self.is_color:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                target_height, target_width, _ = self.data_shape
                if frame.shape[:2] != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(
                            f"Resizing frame from {frame.shape[:2]} to "
                            f"({target_height}, {target_width})."
                        )
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))
            else:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                target_height, target_width = self.data_shape
                if frame.shape != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(
                            f"Resizing frame from {frame.shape} to "
                            f"({target_height}, {target_width})."
                        )
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))

            frame = frame.astype(self.data_dtype)
            return timestamp_ns, frame

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None, None

    def _cleanup_camera(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                logger.warning(f"Failed to release web camera: {exc}")
            self.cap = None
            logger.info("Web camera released")

    def start_server(self) -> None:
        if self.running:
            logger.info(f"Web camera {self.device_class_id} is already running")
            return

        logger.info(f"Starting Web camera server {self.device_class_id}...")
        self.running = True

        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(
            f"Web camera server started. Shared memory: {self.shared_memory_name}"
        )
        logger.info(
            f"Buffer configuration: {self.buffer_size} frames, "
            f"{self.frame_size:,} bytes per frame"
        )

        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        while self.running:
            try:
                timestamp_ns, frame_array = self._capture_frame()
                if frame_array is not None:
                    self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                break

    def stop_server(self) -> None:
        if not self.running:
            return
        logger.info(f"Stopping Web camera server {self.device_class_id}...")
        self.running = False
        self._cleanup_shared_memory()
        logger.info("Web camera server stopped")

    def close(self) -> None:
        self.stop_server()
        self._cleanup_camera()
        logger.info(f"Web camera {self.device_class_id} closed")

    def __del__(self) -> None:
        try:
            self._cleanup_camera()
            if hasattr(self, "shared_memory") and self.shared_memory:
                self._cleanup_shared_memory()
        except Exception:
            pass

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update({
            "camera_id": self.device_class_id,
            "is_color": self.is_color,
            "camera_opened": self.cap is not None and self.cap.isOpened() if self.cap else False,
            "capture_id": self.capture_id,
            "backend": self.backend,
            "hardware_parameters": {
                "brightness": self.brightness,
                "contrast": self.contrast,
                "saturation": self.saturation,
                "hue": self.hue,
                "gamma": self.gamma,
                "sharpness": self.sharpness,
            },
        })
        return info
