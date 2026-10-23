"""Unit tests for the read-only camera SHM preview helper."""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np

# The deployment environments include loguru, but the repository's minimal
# system Python used by CI/smoke tests may not. Keep these SHM-only tests
# independent from logging installation details.
try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    class _NoOpLogger:
        def __getattr__(self, name: str) -> Callable[..., None]:
            return lambda *args, **kwargs: None

    sys.modules["loguru"] = types.SimpleNamespace(logger=_NoOpLogger())

from incubator.control_panel.camera_preview import CameraPreviewReader, read_camera_preview
from incubator.control_panel.shm_protocol import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    calculate_device_shm_size,
    pack_buffer_header,
    pack_device_header,
    pack_frame_header,
)


NOW_NS = 1_000_000_000_000


class FakeBuffer:
    """Buffer-protocol-free fake so reads can be observed and mutated."""

    def __init__(
        self,
        data: bytearray,
        *,
        trigger: Optional[slice] = None,
        mutate_once: Optional[Callable[[bytearray], None]] = None,
    ) -> None:
        self.data = data
        self.trigger = trigger
        self.mutate_once = mutate_once
        self.mutated = False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Any) -> Any:
        value = self.data[key]
        if (
            isinstance(key, slice)
            and self.trigger is not None
            and key.start == self.trigger.start
            and key.stop == self.trigger.stop
            and not self.mutated
        ):
            # bytearray slicing copies before the simulated producer advances.
            self.mutated = True
            assert self.mutate_once is not None
            self.mutate_once(self.data)
        return value


class FakeChannel:
    def __init__(self, buffer: FakeBuffer, *, read_only: bool = True) -> None:
        self.buf = buffer
        self.is_read_only = read_only
        self.closed = False
        self.unlink_calls = 0

    def close(self) -> None:
        self.closed = True

    def unlink(self) -> None:
        self.unlink_calls += 1
        raise AssertionError("A preview reader must never unlink device SHM")


class FakeConnector:
    def __init__(self, channels: Iterable[Any]) -> None:
        self.channels = list(channels)
        self.calls = []

    def __call__(
        self, device_class: str, device_class_id: int, *, read_only: bool
    ) -> FakeChannel:
        self.calls.append((device_class, device_class_id, read_only))
        item = self.channels[min(len(self.calls) - 1, len(self.channels) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


class MutableClock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


def build_segment(
    *,
    device_class: str = "WebCameraOpenPi",
    device_class_id: int = 2,
    shape: tuple = (8, 10, 3),
    dtype: str = "uint8",
    buffer_size: int = 1,
    frames: Optional[list] = None,
    write_index: int = 0,
    current_frames_count: Optional[int] = None,
) -> bytearray:
    numpy_dtype = np.dtype(dtype)
    frame_data_size = int(np.prod(shape)) * numpy_dtype.itemsize
    frame_size = FRAME_HEADER_SIZE + frame_data_size
    data = bytearray(calculate_device_shm_size(buffer_size, frame_data_size))
    data[:DEVICE_HEADER_SIZE] = pack_device_header(
        device_class,
        device_class_id,
        30.0,
        dtype,
        shape,
        frame_size,
        12.5,
    )
    frames = frames or []
    if current_frames_count is None:
        current_frames_count = min(len(frames), buffer_size)
    header_start = DEVICE_HEADER_SIZE
    data[header_start : header_start + BUFFER_HEADER_SIZE] = pack_buffer_header(
        buffer_size, current_frames_count, write_index
    )
    for index, (timestamp_ns, array) in enumerate(frames):
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + index * frame_size
        data[frame_offset : frame_offset + FRAME_HEADER_SIZE] = pack_frame_header(
            timestamp_ns
        )
        payload = np.asarray(array, dtype=numpy_dtype).reshape(shape).tobytes()
        payload_start = frame_offset + FRAME_HEADER_SIZE
        data[payload_start : payload_start + len(payload)] = payload
    return data


def channel_for(data: bytearray, **buffer_options: Any) -> FakeChannel:
    return FakeChannel(FakeBuffer(data, **buffer_options))


class CameraPreviewReaderTest(unittest.TestCase):
    def test_reads_rgb_frame_as_jpeg_with_metadata(self) -> None:
        rgb_red = np.zeros((8, 10, 3), dtype=np.uint8)
        rgb_red[:, :, 0] = 255
        channel = channel_for(
            build_segment(frames=[(NOW_NS - 10_000_000, rgb_red)])
        )
        connector = FakeConnector([channel])
        reader = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=connector,
            clock_ns=lambda: NOW_NS,
            jpeg_quality=95,
        )

        result = reader.read()

        self.assertTrue(result.ok)
        self.assertTrue(result.jpeg_bytes.startswith(b"\xff\xd8"))
        self.assertEqual(result.metadata["status"], "ok")
        self.assertEqual(result.metadata["timestamp_ns"], NOW_NS - 10_000_000)
        self.assertEqual(result.metadata["age_ms"], 10.0)
        self.assertEqual(result.metadata["shape"], (8, 10, 3))
        self.assertEqual(result.metadata["dtype"], "uint8")
        self.assertTrue(result.metadata["fresh"])
        self.assertEqual(connector.calls, [("WebCameraOpenPi", 2, True)])

        decoded = cv2.imdecode(
            np.frombuffer(result.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded.shape, (8, 10, 3))
        # cv2.imdecode returns BGR; a published RGB red frame must remain red.
        self.assertGreater(int(decoded[4, 5, 2]), 220)
        self.assertLess(int(decoded[4, 5, 0]), 30)

        reader.close()
        self.assertTrue(channel.closed)
        self.assertEqual(channel.unlink_calls, 0)

    def test_empty_buffer_returns_status_without_reopening_or_encoding(self) -> None:
        channel = channel_for(build_segment(current_frames_count=0))
        connector = FakeConnector([channel])
        reader = CameraPreviewReader(
            "WebCameraOpenPi", 2, connector=connector, clock_ns=lambda: NOW_NS
        )

        result = reader.read()

        self.assertIsNone(result.jpeg_bytes)
        self.assertEqual(result.metadata["status"], "empty")
        self.assertEqual(result.metadata["shape"], (8, 10, 3))
        self.assertTrue(reader.connected)
        self.assertEqual(len(connector.calls), 1)

    def test_rejects_non_camera_shape(self) -> None:
        vector = np.zeros((16,), dtype=np.uint8)
        channel = channel_for(
            build_segment(shape=(16,), frames=[(NOW_NS, vector)])
        )
        result = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        ).read()

        self.assertEqual(result.metadata["status"], "invalid_shape")
        self.assertEqual(result.metadata["shape"], (16,))
        self.assertIsNone(result.jpeg_bytes)

    def test_rejects_non_uint8_dtype(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.float32)
        channel = channel_for(
            build_segment(
                shape=image.shape, dtype="float32", frames=[(NOW_NS, image)]
            )
        )
        result = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        ).read()

        self.assertEqual(result.metadata["status"], "unsupported_dtype")
        self.assertEqual(result.metadata["dtype"], "float32")
        self.assertIsNone(result.jpeg_bytes)

    def test_retries_when_timestamp_changes_during_payload_copy(self) -> None:
        shape = (4, 5, 3)
        first = np.zeros(shape, dtype=np.uint8)
        second = np.full(shape, 200, dtype=np.uint8)
        data = build_segment(
            shape=shape, frames=[(NOW_NS - 20_000_000, first)]
        )
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
        payload_start = frame_offset + FRAME_HEADER_SIZE
        payload_end = payload_start + second.nbytes

        def publish_second_frame(raw: bytearray) -> None:
            raw[frame_offset : frame_offset + FRAME_HEADER_SIZE] = pack_frame_header(
                NOW_NS - 10_000_000
            )
            raw[payload_start:payload_end] = second.tobytes()

        channel = channel_for(
            data,
            trigger=slice(payload_start, payload_end),
            mutate_once=publish_second_frame,
        )
        result = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
            max_read_retries=3,
        ).read()

        self.assertEqual(result.metadata["status"], "ok")
        self.assertEqual(result.metadata["timestamp_ns"], NOW_NS - 10_000_000)
        self.assertEqual(result.metadata["read_attempts"], 2)

    def test_retries_when_write_index_changes_during_payload_copy(self) -> None:
        shape = (4, 5, 3)
        first = np.zeros(shape, dtype=np.uint8)
        second = np.full(shape, 180, dtype=np.uint8)
        data = build_segment(
            shape=shape,
            buffer_size=2,
            frames=[(NOW_NS - 20_000_000, first)],
            write_index=1,
            current_frames_count=1,
        )
        frame_size = FRAME_HEADER_SIZE + first.nbytes
        first_payload_start = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + FRAME_HEADER_SIZE
        first_payload_end = first_payload_start + first.nbytes

        def advance_ring(raw: bytearray) -> None:
            second_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_size
            raw[second_offset : second_offset + FRAME_HEADER_SIZE] = pack_frame_header(
                NOW_NS - 5_000_000
            )
            second_payload_start = second_offset + FRAME_HEADER_SIZE
            raw[
                second_payload_start : second_payload_start + second.nbytes
            ] = second.tobytes()
            raw[
                DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            ] = pack_buffer_header(2, 2, 0)

        channel = channel_for(
            data,
            trigger=slice(first_payload_start, first_payload_end),
            mutate_once=advance_ring,
        )
        result = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
            max_read_retries=3,
        ).read()

        self.assertEqual(result.metadata["status"], "ok")
        self.assertEqual(result.metadata["timestamp_ns"], NOW_NS - 5_000_000)
        self.assertEqual(result.metadata["read_attempts"], 2)
        self.assertEqual(result.metadata["write_index"], 0)

    def test_cached_stale_handle_is_reconnected_to_recreated_shm(self) -> None:
        shape = (4, 5, 3)
        image = np.full(shape, 50, dtype=np.uint8)
        first_channel = channel_for(
            build_segment(shape=shape, frames=[(NOW_NS, image)])
        )
        second_channel = channel_for(
            build_segment(
                shape=shape, frames=[(NOW_NS + 4_000_000_000, image)]
            )
        )
        connector = FakeConnector([first_channel, second_channel])
        clock = MutableClock(NOW_NS)
        reader = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=connector,
            clock_ns=clock,
            stale_after_s=1.0,
        )
        self.assertEqual(reader.read().metadata["status"], "ok")

        clock.now_ns = NOW_NS + 5_000_000_000
        result = reader.read()

        self.assertEqual(result.metadata["status"], "ok")
        self.assertTrue(result.metadata["reconnected"])
        self.assertTrue(first_channel.closed)
        self.assertFalse(second_channel.closed)
        self.assertEqual(len(connector.calls), 2)
        self.assertEqual(first_channel.unlink_calls, 0)

    def test_new_stale_handle_reports_last_jpeg_then_closes(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        channel = channel_for(
            build_segment(frames=[(NOW_NS - 3_000_000_000, image)], shape=image.shape)
        )
        reader = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
            stale_after_s=1.0,
        )

        result = reader.read()

        self.assertEqual(result.metadata["status"], "stale")
        self.assertFalse(result.metadata["fresh"])
        self.assertIsNotNone(result.jpeg_bytes)
        self.assertTrue(channel.closed)
        self.assertFalse(reader.connected)
        self.assertEqual(channel.unlink_calls, 0)

    def test_missing_segment_is_a_structured_error(self) -> None:
        connector = FakeConnector([FileNotFoundError("gone")])
        result = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=connector,
            clock_ns=lambda: NOW_NS,
        ).read()

        self.assertEqual(result.metadata["status"], "not_found")
        self.assertEqual(result.metadata["shm_name"], "WebCameraOpenPi_2_data")
        self.assertIsNone(result.jpeg_bytes)

    def test_one_shot_api_closes_but_never_unlinks(self) -> None:
        image = np.zeros((4, 5), dtype=np.uint8)
        channel = channel_for(
            build_segment(shape=image.shape, frames=[(NOW_NS, image)])
        )
        result = read_camera_preview(
            "WebCameraOpenPi",
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        )

        self.assertTrue(result.ok)
        self.assertTrue(channel.closed)
        self.assertEqual(channel.unlink_calls, 0)

    def test_corrupt_ring_size_is_rejected_after_bounded_reconnect(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        data = build_segment(shape=image.shape, frames=[(NOW_NS, image)])
        data[
            DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
        ] = pack_buffer_header(500, 1, 0)
        first = channel_for(bytearray(data))
        second = channel_for(bytearray(data))
        connector = FakeConnector([first, second])
        result = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=connector,
            clock_ns=lambda: NOW_NS,
        ).read()

        self.assertEqual(result.metadata["status"], "invalid_header")
        self.assertEqual(len(connector.calls), 2)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(first.unlink_calls + second.unlink_calls, 0)

    def test_writable_connector_result_is_refused(self) -> None:
        channel = FakeChannel(FakeBuffer(bytearray(128)), read_only=False)
        result = CameraPreviewReader(
            "WebCameraOpenPi",
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        ).read()

        self.assertEqual(result.metadata["status"], "connection_error")
        self.assertTrue(channel.closed)
        self.assertEqual(channel.unlink_calls, 0)


if __name__ == "__main__":
    unittest.main()
