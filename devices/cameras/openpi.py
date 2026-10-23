#!/usr/bin/env python3
"""
OpenPi-dedicated web camera device.

This is a copy/extension of ``WebCameraDevice`` created specifically for the
openpi (pi0) real-robot inference path, so the shared ``devices/camera.py`` stays
untouched. It adds the per-camera options that the fisheye UMI inference config
already declares but that the base ``WebCameraDevice`` silently drops (because
``device_starter.py`` filters config keys by the constructor signature):

  - ``rotation_degrees``: rotate each frame to the training "gripper-at-bottom"
    orientation (0/90/180/270, clockwise), applied before the final resize.
  - ``capture_fourcc``: force the capture codec (e.g. "MJPG").
  - ``capture_buffer_size``: cap the driver frame buffer (e.g. 1) to cut latency.
  - ``camera_controls``: a dict of UVC hardware controls (exposure/gain/white
    balance/...), applied generically via cv2 ``CAP_PROP_*``.
  - ``usb_port_requirement``: ``"usb_2.0"``, ``"usb_3.0"``, or ``"any"``;
    validates the negotiated USB link before the camera server starts. This
    checks link speed, not the physical receptacle type.

NOTE on ``state_feedback.format.shape``: it is the FINAL (post-rotation) frame
shape written to SHM. For 90/270 rotation the raw capture width/height are the
transpose of this shape; the requested capture resolution is swapped
accordingly, and the final resize always enforces this shape.

Author: SuperInference
"""

import time
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from devices.config import StateFeedbackConfig
from .webcam import WebCameraDevice
from .usb_link import (
    CaptureId,
    UsbPortRequirementError,
    normalize_usb_port_requirement,
    usb_speed_mbps,
    validate_usb_port_requirement,
)
from utils.logger_config import logger

# Map config control names -> cv2 CAP_PROP_* attribute names. Unsupported names
# are skipped with a warning (resolved defensively via getattr).
_CONTROL_PROP_MAP: Dict[str, str] = {
    "brightness": "CAP_PROP_BRIGHTNESS",
    "contrast": "CAP_PROP_CONTRAST",
    "saturation": "CAP_PROP_SATURATION",
    "hue": "CAP_PROP_HUE",
    "gamma": "CAP_PROP_GAMMA",
    "gain": "CAP_PROP_GAIN",
    "sharpness": "CAP_PROP_SHARPNESS",
    "backlight": "CAP_PROP_BACKLIGHT",
    "exposure": "CAP_PROP_EXPOSURE",
    "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
    "auto_white_balance": "CAP_PROP_AUTO_WB",
    "white_balance_temperature": "CAP_PROP_WB_TEMPERATURE",
    "power_line_frequency": "CAP_PROP_POWER_LINE_FREQUENCY",
    "pan": "CAP_PROP_PAN",
    "tilt": "CAP_PROP_TILT",
    "focus": "CAP_PROP_FOCUS",
    "autofocus": "CAP_PROP_AUTOFOCUS",
}

_ROTATE_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class WebCameraOpenPi(WebCameraDevice):
    """OpenPI camera adapter with capture, image, and USB-link controls."""

    def __init__(
        self,
        device_class_id: int = 0,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        capture_id: Optional[CaptureId] = None,
        backend: Optional[int] = None,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        hue: Optional[float] = None,
        gamma: Optional[float] = None,
        sharpness: Optional[float] = None,
        rotation_degrees: int = 0,
        camera_controls: Optional[Dict[str, Any]] = None,
        capture_fourcc: Optional[str] = None,
        capture_buffer_size: Optional[int] = None,
        usb_port_requirement: str = "any",
    ) -> None:
        # These must be set before super().__init__, which calls _initialize_camera().
        self.rotation_degrees = self._validate_rotation(int(rotation_degrees))
        self.camera_controls = dict(camera_controls or {})
        self.capture_fourcc = capture_fourcc
        self.capture_buffer_size = capture_buffer_size
        self.usb_port_requirement = self._normalize_usb_port_requirement(
            usb_port_requirement
        )
        self.negotiated_usb_speed_mbps: Optional[float] = None

        super().__init__(
            device_class_id=device_class_id,
            state_feedback=state_feedback,
            capture_id=capture_id,
            backend=backend,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            gamma=gamma,
            sharpness=sharpness,
        )

        # Distinct device/SHM name so this class coexists with WebCameraDevice and
        # is addressed consistently by the manager/connector (which derive the SHM
        # name from the config 'class' string).
        self.device_name = "WebCameraOpenPi"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

    @staticmethod
    def _validate_rotation(rotation_degrees: int) -> int:
        valid_angles = {0, 90, 180, 270}
        if rotation_degrees not in valid_angles:
            raise ValueError(
                f"Unsupported rotation_degrees={rotation_degrees}. Supported: {sorted(valid_angles)}"
            )
        return rotation_degrees

    @staticmethod
    def _normalize_usb_port_requirement(requirement: str) -> str:
        return normalize_usb_port_requirement(requirement)

    def _check_usb_port_requirement(self) -> None:
        """Fail closed when the negotiated USB speed does not match config."""
        if self.usb_port_requirement == "any":
            self.negotiated_usb_speed_mbps = None
            return
        self.negotiated_usb_speed_mbps = usb_speed_mbps(self.capture_id)
        validate_usb_port_requirement(
            self.usb_port_requirement,
            self.negotiated_usb_speed_mbps,
            camera_id=self.device_class_id,
            capture_id=self.capture_id,
        )

    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        if self.rotation_degrees == 0:
            return image
        try:
            return cv2.rotate(image, _ROTATE_MAP[self.rotation_degrees])
        except Exception as exc:
            logger.error(f"Failed to rotate frame by {self.rotation_degrees} degrees: {exc}")
            return image

    def _apply_camera_controls(self) -> None:
        """Apply the generic UVC controls dict via cv2 CAP_PROP_*."""
        if not self.cap or not self.camera_controls:
            return
        for name, value in self.camera_controls.items():
            if value is None:
                continue
            prop_name = _CONTROL_PROP_MAP.get(name)
            prop = getattr(cv2, prop_name, None) if prop_name else None
            if prop is None:
                logger.warning(f"WebCameraOpenPi {self.device_class_id}: unsupported control '{name}', skipped")
                continue
            try:
                if not self.cap.set(prop, float(value)):
                    logger.warning(f"WebCameraOpenPi {self.device_class_id} did not accept {name}={value}")
            except Exception as exc:
                logger.warning(f"Failed to set {name}={value}: {exc}")

    def _initialize_camera(self) -> bool:
        """Open VideoCapture, forcing fourcc/buffer/controls and rotation-aware resolution."""
        try:
            # Reject a disallowed negotiated link before touching the camera.
            self._check_usb_port_requirement()

            if self.backend is not None:
                self.cap = cv2.VideoCapture(self.capture_id, self.backend)
            else:
                self.cap = cv2.VideoCapture(self.capture_id)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture could not open index {self.capture_id}")

            # FOURCC must be set before resolution (MJPG enables higher res/fps on many UVC cams).
            if self.capture_fourcc:
                fourcc = cv2.VideoWriter_fourcc(*str(self.capture_fourcc))
                self.cap.set(cv2.CAP_PROP_FOURCC, float(fourcc))
            if self.capture_buffer_size is not None:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.capture_buffer_size))

            # data_shape is the post-rotation target; request the pre-rotation
            # capture resolution (transposed for 90/270) as a hint to the driver.
            if self.is_color:
                out_h, out_w, _ = self.data_shape
            else:
                out_h, out_w = self.data_shape
            if self.rotation_degrees in (90, 270):
                cap_w, cap_h = out_h, out_w
            else:
                cap_w, cap_h = out_w, out_h

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cap_w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cap_h))
            self.cap.set(cv2.CAP_PROP_FPS, float(max(1.0, self.fps)))

            self._set_camera_parameters()   # base brightness/contrast/... (top-level args)
            self._apply_camera_controls()    # extra UVC controls dict

            for _ in range(3):
                self.cap.read()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Cannot capture warm-up frame from web camera")

            logger.info(
                "WebCameraOpenPi %s initialized: capture_id=%s frame=%s "
                "rotation=%s fourcc=%s buffer=%s target=%s",
                self.device_class_id,
                self.capture_id,
                frame.shape,
                self.rotation_degrees,
                self.capture_fourcc,
                self.capture_buffer_size,
                self.data_shape,
            )
            logger.info(
                "WebCameraOpenPi %s USB requirement=%s negotiated_speed=%s Mbps",
                self.device_class_id,
                self.usb_port_requirement,
                self.negotiated_usb_speed_mbps,
            )
            self._log_camera_parameters()
            return True

        except UsbPortRequirementError:
            self._cleanup_camera()
            raise
        except Exception as e:
            logger.error(f"WebCameraOpenPi initialization failed: {e}")
            self._cleanup_camera()
            return False

    def _capture_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Read a frame, convert to RGB, rotate, then resize to data_shape."""
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
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Rotate to the training orientation before enforcing the target shape.
                frame = self._apply_rotation(frame)

                target_height, target_width, _ = self.data_shape
                if frame.shape[:2] != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(
                            f"Resizing frame from {frame.shape[:2]} to ({target_height}, {target_width})."
                        )
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))
            else:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = self._apply_rotation(frame)
                target_height, target_width = self.data_shape
                if frame.shape != (target_height, target_width):
                    if not self._resize_warned:
                        logger.warning(
                            f"Resizing frame from {frame.shape} to ({target_height}, {target_width})."
                        )
                        self._resize_warned = True
                    frame = cv2.resize(frame, (target_width, target_height))

            frame = frame.astype(self.data_dtype)
            return timestamp_ns, frame

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None, None

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update({
            "rotation_degrees": self.rotation_degrees,
            "capture_fourcc": self.capture_fourcc,
            "capture_buffer_size": self.capture_buffer_size,
            "camera_controls": self.camera_controls,
            "usb_port_requirement": self.usb_port_requirement,
            "usb_speed_mbps": self.negotiated_usb_speed_mbps,
        })
        return info
