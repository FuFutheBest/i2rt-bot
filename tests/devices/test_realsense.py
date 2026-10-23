"""Hardware-free tests for the RealSense shared-memory adapter."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from devices.realsense import RealSenseCameraDevice


class _ColorFrame:
    def get_timestamp(self) -> float:
        return 1234.5

    def get_data(self) -> np.ndarray:
        return np.zeros((2, 3, 3), dtype=np.uint8)


class _Pipeline:
    def wait_for_frames(self) -> SimpleNamespace:
        return SimpleNamespace(get_color_frame=lambda: _ColorFrame())


def test_capture_frame_uses_host_epoch_timestamp() -> None:
    camera = RealSenseCameraDevice.__new__(RealSenseCameraDevice)
    camera.pipeline = _Pipeline()
    camera.is_color = True
    camera.data_shape = (2, 3, 3)
    camera.data_dtype = np.dtype("uint8")
    camera._shape_mismatch_warned = False

    before_ns = time.time_ns()
    timestamp_ns, frame = camera._capture_frame()
    after_ns = time.time_ns()

    assert before_ns <= timestamp_ns <= after_ns
    assert frame.shape == (2, 3, 3)


def test_capture_frame_rejects_shape_mismatch_without_resizing() -> None:
    camera = RealSenseCameraDevice.__new__(RealSenseCameraDevice)
    camera.pipeline = _Pipeline()
    camera.is_color = True
    camera.data_shape = (4, 5, 3)
    camera.data_dtype = np.dtype("uint8")
    camera._shape_mismatch_warned = False

    timestamp_ns, frame = camera._capture_frame()

    assert timestamp_ns is None
    assert frame is None
    assert camera._shape_mismatch_warned


def test_capture_loop_relies_on_hardware_frame_pacing() -> None:
    camera = RealSenseCameraDevice.__new__(RealSenseCameraDevice)
    camera.device_class_id = 0
    camera.running = False
    camera.shared_memory = object()
    camera.shared_memory_name = "test"
    camera.buffer_size = 1
    camera.frame_size = 1
    camera._init_show_reference_image = False
    captured: list[tuple[np.ndarray, int]] = []

    camera._create_shared_memory = lambda: None

    def capture_once() -> tuple[int, np.ndarray]:
        camera.running = False
        return 123, np.zeros((2, 3, 3), dtype=np.uint8)

    camera._capture_frame = capture_once
    camera._write_array_to_shm_with_timestamp = (
        lambda frame, timestamp: captured.append((frame, timestamp))
    )
    camera._display_reference_overlay = lambda _frame: None
    camera._display_debug_frame = lambda _frame: None

    camera.start_server()

    assert len(captured) == 1
    assert captured[0][1] == 123
