"""ZED stereo camera adapter.

The ZED SDK is imported only when the device server starts.  This keeps the
registry and configuration checks usable on CI runners that do not have the
vendor SDK installed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from devices.cameras.base import BaseCameraDevice
from devices.config import StateFeedbackConfig
from utils.logger_config import logger
from utils.time_control import precise_loop_timing


class ZedStereoDevice(BaseCameraDevice):
    """Publish a side-by-side RGB frame from a ZED stereo camera to SHM."""

    _SUPPORTED_DTYPES = {
        "uint8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "float32",
        "float64",
    }
    ZED_RESOLUTIONS = {
        "HD720": "HD720",
        "HD1080": "HD1080",
        "HD1200": "HD1200",
        "VGA": "VGA",
    }

    def __init__(
        self,
        device_class_id: int = 0,
        device_name: Optional[str] = None,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        zed_resolution: str = "HD720",
        camera_fps: int = 60,
        disable_self_calib: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs  # Keep compatibility with config-driven device construction.
        super().__init__(
            device_class_id=device_class_id,
            device_name=device_name or "ZedStereoDevice",
            state_feedback=state_feedback,
        )
        normalized_dtype = np.dtype(self.data_dtype)
        if normalized_dtype.name not in self._SUPPORTED_DTYPES:
            supported = ", ".join(sorted(self._SUPPORTED_DTYPES))
            raise ValueError(
                f"Unsupported ZED state_feedback.format.dtype {normalized_dtype.name!r}; "
                f"available: {supported}"
            )
        if len(self.data_shape) != 3 or self.data_shape[2] != 3:
            raise ValueError(
                f"ZedStereoDevice expects data_shape (H, W*2, 3); got {self.data_shape}"
            )
        if self.data_shape[1] <= 0 or self.data_shape[1] % 2:
            raise ValueError(
                f"ZedStereoDevice expects an even side-by-side width; got {self.data_shape[1]}"
            )
        if zed_resolution not in self.ZED_RESOLUTIONS:
            available = ", ".join(self.ZED_RESOLUTIONS)
            raise ValueError(f"Unsupported ZED resolution {zed_resolution!r}; available: {available}")

        self.out_height = self.data_shape[0]
        self.out_width = self.data_shape[1]
        self.zed_resolution = zed_resolution
        self.camera_fps = int(camera_fps)
        self.disable_self_calib = bool(disable_self_calib)

        self._sl: Any = None
        self._zed: Any = None
        self._runtime_params: Any = None
        self._left_mat: Any = None
        self._right_mat: Any = None
        self._stereo_rgb = np.empty(
            (self.out_height, self.out_width, 3),
            dtype=normalized_dtype,
        )
        self.running = False
        logger.info(
            f"ZedStereoDevice initialized: SHM={self.shared_memory_name}, "
            f"resolution={self.zed_resolution}@{self.camera_fps}fps, "
            f"output={self.out_width}x{self.out_height} (side-by-side stereo)"
        )

    def _open_camera(self) -> None:
        """Open the vendor camera and allocate SDK frame buffers."""
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, self.zed_resolution)
        init_params.camera_fps = self.camera_fps
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_verbose = 0
        init_params.camera_disable_self_calib = self.disable_self_calib

        started_at = time.time()
        error = self._zed.open(init_params)
        if error != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {error!r}")
        logger.info(
            f"[zed] Camera.open() done in {time.time() - started_at:.2f}s "
            f"(resolution={self.zed_resolution}, fps={self.camera_fps})"
        )
        self._runtime_params = sl.RuntimeParameters()
        self._left_mat = sl.Mat()
        self._right_mat = sl.Mat()

    @staticmethod
    def _copy_eye(source: np.ndarray, destination: np.ndarray) -> None:
        """Convert an SDK eye image to RGB and copy it into ``destination``."""
        import cv2

        if source.ndim == 2:
            converted = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3 and source.shape[2] == 4:
            converted = cv2.cvtColor(source, cv2.COLOR_BGRA2RGB)
        elif source.ndim == 3 and source.shape[2] == 3:
            converted = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported ZED image shape: {source.shape}")

        if converted.shape[:2] != destination.shape[:2]:
            converted = cv2.resize(converted, (destination.shape[1], destination.shape[0]))
        destination[...] = converted

    def _capture_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        if self._sl is None or self._zed is None:
            raise RuntimeError("ZED camera is not open")
        if self._zed.grab(self._runtime_params) != self._sl.ERROR_CODE.SUCCESS:
            return None, None
        self._zed.retrieve_image(self._left_mat, self._sl.VIEW.LEFT)
        self._zed.retrieve_image(self._right_mat, self._sl.VIEW.RIGHT)
        half_width = self.out_width // 2
        self._copy_eye(self._left_mat.numpy(), self._stereo_rgb[:, :half_width])
        self._copy_eye(self._right_mat.numpy(), self._stereo_rgb[:, half_width:])
        return time.time_ns(), self._stereo_rgb

    def start_server(self) -> None:
        if self.running:
            logger.info(f"ZedStereoDevice {self.device_class_id} already running")
            return
        logger.info(f"Starting ZedStereoDevice {self.device_class_id} server...")
        self.running = True
        try:
            self._create_shared_memory()
            if not self.shared_memory:
                raise RuntimeError("Failed to create shared memory")
            self._open_camera()
            logger.info(f"ZED server started. Shared memory: {self.shared_memory_name}")

            wait_for_next_iteration = precise_loop_timing(self.update_interval)
            while self.running:
                timestamp_ns, frame_array = self._capture_frame()
                if frame_array is not None and timestamp_ns is not None:
                    self._write_array_to_shm_with_timestamp(frame_array, timestamp_ns)
                wait_for_next_iteration()
        except Exception:
            self.running = False
            self._close_camera()
            self._cleanup_shared_memory()
            raise

    def _close_camera(self) -> None:
        if self._zed is not None:
            try:
                self._zed.close()
            except Exception as exc:
                logger.warning(f"Failed to close ZED camera: {exc}")
        self._zed = None
        self._runtime_params = None
        self._left_mat = None
        self._right_mat = None
        self._sl = None

    def stop_server(self) -> None:
        if not self.running and self._zed is None and self.shared_memory is None:
            return
        logger.info(f"Stopping ZedStereoDevice {self.device_class_id} server...")
        self.running = False
        self._close_camera()
        self._cleanup_shared_memory()
        logger.info("ZED server stopped")

    def close(self) -> None:
        self.stop_server()

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update(
            {
                "zed_resolution": self.zed_resolution,
                "camera_fps": self.camera_fps,
                "output_size": (self.out_width, self.out_height),
            }
        )
        return info


__all__ = ["ZedStereoDevice"]
