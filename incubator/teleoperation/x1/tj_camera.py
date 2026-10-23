#!/usr/bin/env python3
"""
X1 (TJ / nmxrdk) camera device for SuperInference data collection.

Each instance owns one nmxrdk camera (head or wrist) and publishes the decoded
BGR frame to its per-device SHM (``TjCameraDevice_<id>_data``) so that:

  * the device_manager can aggregate it into the summary SHM for the data saver
    (NEDF3 episode recording), and
  * the WebRTC FPV server (``tj_camera_webrtc.py``) can READ the head frame from
    SHM instead of grabbing the camera itself (single grabber per camera, no
    double-grab contention).

Cameras that are not physically installed yet (e.g. the wrist cameras, which are
meant to mirror R1's layout) are handled gracefully: a warning is logged once
and black frames are published so the rest of the pipeline keeps running.

Frames are stored in OpenCV-native BGR order (same convention as
``OpenCVCameraDevice``); colour conversion is done downstream (WebRTC / saver).

Author: Zixi Ying
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from devices.base import BaseDevice
from devices.config import StateFeedbackConfig
from devices.contracts import DeviceKind
from utils.logger_config import logger
from utils.time_control import precise_loop_timing


class TjCameraDevice(BaseDevice):
    """Single nmxrdk camera -> per-device SHM publisher."""

    device_kind = DeviceKind.CAMERA

    def __init__(
        self,
        device_class_id: int = 0,
        device_name: Optional[str] = None,
        robot_id: str = "1e783be2dd870ef5",
        cam_id: str = "CAMID_0",
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        role: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            device_name=device_name or "TjCameraDevice",
            state_feedback=state_feedback,
        )
        self.device_name = device_name or "TjCameraDevice"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        if len(self.data_shape) != 3 or self.data_shape[2] != 3:
            raise ValueError(f"TjCameraDevice data_shape must be (H, W, 3), got {self.data_shape}")
        self._height = int(self.data_shape[0])
        self._width = int(self.data_shape[1])

        self.robot_id = str(robot_id)
        self.cam_id = str(cam_id)
        self.role = role

        self._adapter: Optional[Any] = None
        self._camera: Optional[Any] = None
        self._available = False
        self._warned_unavailable = False

        self._connect_camera()
        logger.info(
            f"TjCameraDevice initialized: cam_id={self.cam_id}, role={self.role or '-'}, "
            f"available={self._available}, shape={self.data_shape}, SHM={self.shared_memory_name}"
        )

    def _connect_camera(self) -> None:
        """Connect to the nmxrdk camera; degrade to black frames if absent.

        The frame probe runs in a daemon thread with a short timeout so an
        unresponsive camera service (e.g. wrist CAMID_1/2 slots that have no
        physical camera plugged in) can never hang the launcher forever:
        ``nmxrdk.get_rgbd(decode=True)`` blocks waiting for a publisher, so a
        plain synchronous probe would deadlock the device __init__.
        """
        import threading

        try:
            from incubator.teleoperation.x1.nmx_adapter import NmxAdapter

            self._adapter = NmxAdapter(self.robot_id)
            self._camera = self._adapter.camera(self.cam_id)

            probe_result: list = [None]

            def _probe() -> None:
                try:
                    probe_result[0] = self._camera.get_bgr()
                except Exception:
                    probe_result[0] = None

            probe_timeout_s = 2.0
            t = threading.Thread(target=_probe, name=f"tj-cam-probe-{self.cam_id}", daemon=True)
            t.start()
            t.join(timeout=probe_timeout_s)

            if t.is_alive():
                # Probe still running -> camera service is up but no publisher
                # (no physical camera). Don't wait, let runtime emit black frames.
                self._available = False
                logger.warning(
                    f"TjCameraDevice cam_id={self.cam_id} probe timed out after "
                    f"{probe_timeout_s:.1f}s; publishing black frames "
                    f"(camera may not be installed yet)."
                )
            else:
                self._available = probe_result[0] is not None
                if not self._available:
                    logger.warning(
                        f"TjCameraDevice cam_id={self.cam_id} connected but returned no frame; "
                        f"publishing black frames (camera may not be installed yet)."
                    )
        except Exception as exc:
            self._available = False
            logger.warning(
                f"TjCameraDevice cam_id={self.cam_id} unavailable ({exc}); "
                f"publishing black frames (camera may not be installed yet)."
            )

    def _black_frame(self) -> np.ndarray:
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def _capture_frame(self) -> Tuple[int, np.ndarray]:
        """Return (timestamp_ns, HxWx3 BGR uint8). Falls back to black frames."""
        timestamp_ns = time.time_ns()
        if not self._available or self._camera is None:
            return timestamp_ns, self._black_frame()

        frame = self._camera.get_bgr()
        if frame is None:
            if not self._warned_unavailable:
                logger.warning(
                    f"TjCameraDevice cam_id={self.cam_id} stopped returning frames; "
                    f"publishing black frames."
                )
                self._warned_unavailable = True
            return timestamp_ns, self._black_frame()

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            frame = cv2.resize(frame, (self._width, self._height))
        return timestamp_ns, np.ascontiguousarray(frame, dtype=np.uint8)

    def start_server(self) -> None:
        if self.running:
            logger.info(f"TjCameraDevice {self.device_class_id} already running")
            return

        logger.info(f"Starting TjCameraDevice {self.device_class_id} (cam_id={self.cam_id})...")
        self.running = True

        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        while self.running:
            try:
                timestamp_ns, frame_array = self._capture_frame()
                self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)
                wait_for_next_iteration()
            except Exception as exc:
                logger.error(f"Error in TjCameraDevice loop: {exc}")
                break

    def stop_server(self) -> None:
        if not self.running:
            return
        logger.info(f"Stopping TjCameraDevice {self.device_class_id}...")
        self.running = False
        self._cleanup_shared_memory()

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update({
            "cam_id": self.cam_id,
            "role": self.role,
            "available": self._available,
        })
        return info
