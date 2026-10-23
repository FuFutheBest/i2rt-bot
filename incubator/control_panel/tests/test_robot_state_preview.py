"""Unit tests for the read-only RizonRobot SHM state helper."""

from __future__ import annotations

import sys
import types
import unittest
from typing import Any, Callable, Iterable, Optional

import numpy as np

# Keep these SHM-only tests independent from the deployment logging package.
try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    class _NoOpLogger:
        def __getattr__(self, name: str) -> Callable[..., None]:
            return lambda *args, **kwargs: None

    sys.modules["loguru"] = types.SimpleNamespace(logger=_NoOpLogger())

from incubator.control_panel.robot_state_preview import (
    RobotStatePreviewReader,
    YamStatePreviewReader,
    read_robot_state_preview,
)
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
    """Buffer-protocol-free fake whose slices can trigger producer mutation."""

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
        raise AssertionError("Robot preview readers must never unlink device SHM")


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
    device_class: str = "RizonRobot",
    device_class_id: int = 2,
    shape: tuple = (44,),
    dtype: str = "float64",
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
        120.0,
        dtype,
        shape,
        frame_size,
        10.0,
    )
    frames = frames or []
    if current_frames_count is None:
        current_frames_count = min(len(frames), buffer_size)
    data[
        DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
    ] = pack_buffer_header(buffer_size, current_frames_count, write_index)
    for index, (timestamp_ns, array) in enumerate(frames):
        frame_offset = (
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + index * frame_size
        )
        data[frame_offset : frame_offset + FRAME_HEADER_SIZE] = pack_frame_header(
            timestamp_ns
        )
        payload = np.asarray(array, dtype=numpy_dtype).reshape(shape).tobytes()
        payload_start = frame_offset + FRAME_HEADER_SIZE
        data[payload_start : payload_start + len(payload)] = payload
    return data


def channel_for(data: bytearray, **buffer_options: Any) -> FakeChannel:
    return FakeChannel(FakeBuffer(data, **buffer_options))


def dual_state(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    state = np.linspace(-2.0, 2.0, 44, dtype=np.float64)
    state[0:7] = left
    state[22:29] = right
    return state


class RobotStatePreviewReaderTest(unittest.TestCase):
    def test_reads_yam_20d_joint_and_measured_tcp_state(self) -> None:
        state = np.zeros(20, dtype=np.float64)
        state[:6] = [0.1, 2.1, 0.7, 0.8, -0.1, 0.03]
        state[6:13] = [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0]
        state[13:20] = [0.2, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]
        channel = channel_for(
            build_segment(
                device_class='YamHeadRobot',
                device_class_id=0,
                shape=(20,),
                frames=[(NOW_NS, state)],
            )
        )
        connector = FakeConnector([channel])

        with YamStatePreviewReader(
            0, connector=connector, clock_ns=lambda: NOW_NS
        ) as reader:
            result = reader.read()

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata['schema'], 'yam_joint_commanded_measured_20')
        self.assertEqual(result.metadata['joints_per_arm'], 6)
        np.testing.assert_allclose(result.joint_positions['yam'], state[:6])
        np.testing.assert_allclose(result.tcp_poses['yam'], state[13:20])
        self.assertEqual(connector.calls, [('YamHeadRobot', 0, True)])

    def test_reads_dual_44d_joint_slices_and_preserves_arm_order(self) -> None:
        left = np.linspace(-0.6, 0.6, 7)
        right = np.linspace(0.7, 1.3, 7)
        left_tcp = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        right_tcp = np.array([-0.1, -0.2, 0.4, 0.0, 0.0, 1.0, 0.0])
        left_gripper = 0.031
        right_gripper = 0.074
        state = dual_state(left, right)
        state[14:21] = left_tcp
        state[21] = left_gripper
        state[36:43] = right_tcp
        state[43] = right_gripper
        channel = channel_for(
            build_segment(
                frames=[(NOW_NS - 10_000_000, state)]
            )
        )
        connector = FakeConnector([channel])
        reader = RobotStatePreviewReader(
            2, connector=connector, clock_ns=lambda: NOW_NS
        )

        result = reader.read()

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "ok")
        np.testing.assert_allclose(result.joint_positions["left"], left)
        np.testing.assert_allclose(result.joint_positions["right"], right)
        np.testing.assert_allclose(result.tcp_poses["left"], left_tcp)
        np.testing.assert_allclose(result.tcp_poses["right"], right_tcp)
        self.assertAlmostEqual(result.gripper_positions["left"], left_gripper)
        self.assertAlmostEqual(result.gripper_positions["right"], right_gripper)
        self.assertEqual(result.metadata["timestamp_ns"], NOW_NS - 10_000_000)
        self.assertEqual(result.metadata["age_ms"], 10.0)
        self.assertEqual(result.metadata["schema"], "dual_tcp_gripper_44")
        self.assertEqual(result.metadata["arm_order"], ("left", "right"))
        self.assertEqual(result.metadata["tcp_arm_order"], ("left", "right"))
        self.assertEqual(
            result.metadata["gripper_arm_order"], ("left", "right")
        )
        self.assertEqual(result.metadata["joints_per_arm"], 7)
        self.assertEqual(connector.calls, [("RizonRobot", 2, True)])
        self.assertIsInstance(result.to_dict()["joint_positions"]["left"], list)
        self.assertIsInstance(result.to_dict()["tcp_poses"]["left"], list)
        self.assertAlmostEqual(
            result.to_dict()["gripper_positions"]["left"], left_gripper
        )
        self.assertAlmostEqual(
            result.to_dict()["gripper_positions"]["right"], right_gripper
        )

        reader.close()
        self.assertTrue(channel.closed)
        self.assertEqual(channel.unlink_calls, 0)

    def test_reads_supported_single_arm_schemas(self) -> None:
        positions = np.linspace(-1.2, 1.2, 7)
        for shape, schema_name in (
            ((14,), "single_joint_14"),
            ((22,), "single_tcp_gripper_22"),
        ):
            with self.subTest(shape=shape):
                state = np.linspace(3.0, 4.0, shape[0])
                state[:7] = positions
                tcp_pose = np.array(
                    [0.4, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0]
                )
                if shape == (22,):
                    state[14:21] = tcp_pose
                    state[21] = 0.057
                channel = channel_for(
                    build_segment(shape=shape, frames=[(NOW_NS, state)])
                )
                with RobotStatePreviewReader(
                    2,
                    connector=FakeConnector([channel]),
                    clock_ns=lambda: NOW_NS,
                ) as reader:
                    result = reader.read()

                self.assertTrue(result.ok)
                self.assertEqual(tuple(result.joint_positions), ("single",))
                np.testing.assert_allclose(
                    result.joint_positions["single"], positions
                )
                self.assertEqual(result.metadata["schema"], schema_name)
                if shape == (22,):
                    np.testing.assert_allclose(
                        result.tcp_poses["single"], tcp_pose
                    )
                    self.assertEqual(
                        result.metadata["gripper_arm_order"], ("single",)
                    )
                    self.assertAlmostEqual(
                        result.gripper_positions["single"], 0.057
                    )
                else:
                    self.assertEqual(result.tcp_poses, {})
                    self.assertEqual(result.gripper_positions, {})
                    self.assertEqual(result.metadata["gripper_arm_order"], ())
                self.assertTrue(channel.closed)
                self.assertEqual(channel.unlink_calls, 0)

    def test_empty_ring_reports_connected_without_joint_data(self) -> None:
        channel = channel_for(build_segment(current_frames_count=0))
        reader = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        )

        result = reader.read()

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.joint_positions, {})
        self.assertEqual(result.metadata["schema"], "dual_tcp_gripper_44")
        self.assertTrue(reader.connected)
        reader.close()

    def test_rejects_unknown_rizon_payload_shape(self) -> None:
        state = np.zeros(43, dtype=np.float64)
        channel = channel_for(
            build_segment(shape=(43,), frames=[(NOW_NS, state)])
        )
        reader = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        )

        result = reader.read()

        self.assertEqual(result.status, "invalid_shape")
        self.assertEqual(result.metadata["shape"], (43,))
        self.assertEqual(result.joint_positions, {})
        reader.close()
        self.assertEqual(channel.unlink_calls, 0)

    def test_rejects_non_floating_state_dtype(self) -> None:
        state = np.zeros(44, dtype=np.int32)
        channel = channel_for(
            build_segment(
                shape=(44,), dtype="int32", frames=[(NOW_NS, state)]
            )
        )
        reader = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        )

        result = reader.read()

        self.assertEqual(result.status, "unsupported_dtype")
        self.assertEqual(result.metadata["dtype"], "int32")
        reader.close()

    def test_retries_when_ring_advances_during_payload_copy(self) -> None:
        first_left = np.full(7, 0.1)
        first_right = np.full(7, 0.2)
        second_left = np.full(7, 1.1)
        second_right = np.full(7, 1.2)
        first = dual_state(first_left, first_right)
        second = dual_state(second_left, second_right)
        data = build_segment(
            buffer_size=2,
            frames=[(NOW_NS - 20_000_000, first)],
            write_index=1,
            current_frames_count=1,
        )
        frame_size = FRAME_HEADER_SIZE + first.nbytes
        first_payload_start = (
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + FRAME_HEADER_SIZE
        )
        first_payload_end = first_payload_start + first.nbytes

        def publish_second(raw: bytearray) -> None:
            second_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_size
            raw[
                second_offset : second_offset + FRAME_HEADER_SIZE
            ] = pack_frame_header(NOW_NS - 5_000_000)
            payload_start = second_offset + FRAME_HEADER_SIZE
            raw[payload_start : payload_start + second.nbytes] = second.tobytes()
            raw[
                DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            ] = pack_buffer_header(2, 2, 0)

        channel = channel_for(
            data,
            trigger=slice(first_payload_start, first_payload_end),
            mutate_once=publish_second,
        )
        result = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
            max_read_retries=3,
        ).read()

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.metadata["read_attempts"], 2)
        self.assertEqual(result.metadata["write_index"], 0)
        np.testing.assert_allclose(
            result.joint_positions["left"], second_left
        )
        np.testing.assert_allclose(
            result.joint_positions["right"], second_right
        )

    def test_cached_stale_handle_reconnects_to_recreated_segment(self) -> None:
        positions = np.zeros(7)
        first_channel = channel_for(
            build_segment(
                frames=[(NOW_NS, dual_state(positions, positions))]
            )
        )
        second_left = np.full(7, 0.75)
        second_channel = channel_for(
            build_segment(
                frames=[
                    (
                        NOW_NS + 4_990_000_000,
                        dual_state(second_left, positions),
                    )
                ]
            )
        )
        connector = FakeConnector([first_channel, second_channel])
        clock = MutableClock(NOW_NS)
        reader = RobotStatePreviewReader(
            2,
            connector=connector,
            clock_ns=clock,
            stale_after_s=1.0,
        )
        self.assertEqual(reader.read().status, "ok")

        clock.now_ns = NOW_NS + 5_000_000_000
        result = reader.read()

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.metadata["reconnected"])
        np.testing.assert_allclose(
            result.joint_positions["left"], second_left
        )
        self.assertTrue(first_channel.closed)
        self.assertFalse(second_channel.closed)
        self.assertEqual(len(connector.calls), 2)
        self.assertEqual(first_channel.unlink_calls, 0)
        reader.close()

    def test_initial_stale_frame_is_returned_then_handle_is_closed(self) -> None:
        left = np.full(7, -0.4)
        channel = channel_for(
            build_segment(
                frames=[
                    (NOW_NS - 3_000_000_000, dual_state(left, np.zeros(7)))
                ]
            )
        )
        reader = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
            stale_after_s=1.0,
        )

        result = reader.read()

        self.assertEqual(result.status, "stale")
        self.assertFalse(result.metadata["fresh"])
        np.testing.assert_allclose(result.joint_positions["left"], left)
        self.assertTrue(channel.closed)
        self.assertFalse(reader.connected)
        self.assertEqual(channel.unlink_calls, 0)

    def test_non_finite_joint_data_is_rejected_with_frame_metadata(self) -> None:
        left = np.zeros(7)
        left[3] = np.nan
        channel = channel_for(
            build_segment(frames=[(NOW_NS, dual_state(left, np.zeros(7)))])
        )
        reader = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        )

        result = reader.read()

        self.assertEqual(result.status, "invalid_data")
        self.assertEqual(result.metadata["timestamp_ns"], NOW_NS)
        self.assertEqual(result.joint_positions, {})
        reader.close()

    def test_invalid_tcp_pose_is_rejected(self) -> None:
        for bad_value, quaternion in (
            (np.inf, np.array([0.0, 0.0, 0.0, 1.0])),
            (0.1, np.zeros(4)),
        ):
            with self.subTest(bad_value=bad_value, quaternion=quaternion):
                state = dual_state(np.zeros(7), np.zeros(7))
                state[14:17] = [bad_value, 0.2, 0.3]
                state[17:21] = quaternion
                channel = channel_for(
                    build_segment(frames=[(NOW_NS, state)])
                )
                result = RobotStatePreviewReader(
                    2,
                    connector=FakeConnector([channel]),
                    clock_ns=lambda: NOW_NS,
                ).read()

                self.assertEqual(result.status, "invalid_data")
                self.assertEqual(result.joint_positions, {})
                self.assertEqual(result.tcp_poses, {})

    def test_non_finite_gripper_position_is_rejected(self) -> None:
        state = dual_state(np.zeros(7), np.zeros(7))
        state[14:21] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        state[36:43] = [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]
        state[21] = np.nan
        channel = channel_for(build_segment(frames=[(NOW_NS, state)]))

        result = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        ).read()

        self.assertEqual(result.status, "invalid_data")
        self.assertEqual(result.joint_positions, {})
        self.assertEqual(result.tcp_poses, {})
        self.assertEqual(result.gripper_positions, {})

    def test_missing_segment_is_structured_error(self) -> None:
        connector = FakeConnector([FileNotFoundError("gone")])
        result = RobotStatePreviewReader(
            2, connector=connector, clock_ns=lambda: NOW_NS
        ).read()

        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.metadata["shm_name"], "RizonRobot_2_data")
        self.assertEqual(result.joint_positions, {})

    def test_one_shot_api_closes_but_never_unlinks(self) -> None:
        state = dual_state(np.zeros(7), np.zeros(7))
        channel = channel_for(build_segment(frames=[(NOW_NS, state)]))
        result = read_robot_state_preview(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        )

        self.assertTrue(result.ok)
        self.assertTrue(channel.closed)
        self.assertEqual(channel.unlink_calls, 0)

    def test_corrupt_ring_size_is_rejected_after_bounded_reconnect(self) -> None:
        state = dual_state(np.zeros(7), np.zeros(7))
        data = build_segment(frames=[(NOW_NS, state)])
        data[
            DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
        ] = pack_buffer_header(500, 1, 0)
        first = channel_for(bytearray(data))
        second = channel_for(bytearray(data))
        connector = FakeConnector([first, second])

        result = RobotStatePreviewReader(
            2, connector=connector, clock_ns=lambda: NOW_NS
        ).read()

        self.assertEqual(result.status, "invalid_header")
        self.assertEqual(len(connector.calls), 2)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(first.unlink_calls + second.unlink_calls, 0)

    def test_writable_connector_result_is_refused(self) -> None:
        channel = FakeChannel(FakeBuffer(bytearray(128)), read_only=False)
        result = RobotStatePreviewReader(
            2,
            connector=FakeConnector([channel]),
            clock_ns=lambda: NOW_NS,
        ).read()

        self.assertEqual(result.status, "connection_error")
        self.assertTrue(channel.closed)
        self.assertEqual(channel.unlink_calls, 0)


if __name__ == "__main__":
    unittest.main()
