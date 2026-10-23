"""Contracts for synchronized device-manager master selection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manager.utils.data_interpolation import interpolate_low_dim_data
from manager.synchronized_device_manager import SynchronizedDeviceManager
from utils.data_format import parse_data_format
from utils.synchronization import BINARY_MATCH, CAMERA_ALIGN


def _camera_config() -> dict:
    return {
        "state_feedback": {
            "format": {
                "shape": [2, 2, 3],
                "type": "rgb",
                "dtype": "uint8",
            }
        }
    }


@pytest.mark.parametrize("disabled_head", [False, True])
def test_master_device_selection_preserves_id_after_disabled_filtering(disabled_head) -> None:
    """Global device class IDs follow the complete configuration list."""
    manager = object.__new__(SynchronizedDeviceManager)
    manager.master_device_class_id = 3
    manager.devices = [
        {"device_class_id": 0, "name": "left"},
        {"device_class_id": 1, "name": "right"},
        {"device_class_id": 2, "name": "head"},
        {"device_class_id": 3, "name": "robot"},
    ]
    if disabled_head:
        manager.devices.pop(2)
    reads: list[str] = []
    searches: list[str] = []

    def read(device: dict) -> tuple[str, str]:
        reads.append(device["name"])
        return ("master", device["name"])

    def search(device: dict, _timestamp: int) -> tuple[str, str, int]:
        searches.append(device["name"])
        return (123, device["name"], 0)

    manager._read_device_frame = read
    manager._calculate_adjusted_timestamp = lambda timestamp, _device: timestamp
    manager._connect_to_device = lambda _device: True
    manager._search_device_buffer_for_timestamp = search

    frames = manager._find_synchronized_frames(123)

    assert reads == ["robot"]
    assert searches == (["left", "right"] if disabled_head else ["left", "right", "head"])
    expected_frames = [
        (123, "left"),
        (123, "right"),
        (123, "head"),
        ("master", "robot"),
    ]
    if disabled_head:
        expected_frames.pop(2)
    assert frames == expected_frames


def test_low_dim_interpolation_uses_linear_and_wxyz_slerp() -> None:
    data_format = parse_data_format(
        {
            "shape": [7],
            "type": ["xyz", "qwqxqyqz"],
        }
    )
    start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    end_xyzw = Rotation.from_euler("z", 90.0, degrees=True).as_quat()
    end = np.array([2.0, 4.0, 6.0, *end_xyzw[[3, 0, 1, 2]]])

    result = interpolate_low_dim_data(start, end, 0.5, data_format)

    np.testing.assert_allclose(result[:3], [1.0, 2.0, 3.0])
    expected_xyzw = Rotation.from_euler("z", 45.0, degrees=True).as_quat()
    expected_wxyz = expected_xyzw[[3, 0, 1, 2]]
    assert abs(float(np.dot(result[3:], expected_wxyz))) == pytest.approx(1.0)
    assert np.linalg.norm(result[3:]) == pytest.approx(1.0)


def test_synchronization_mode_defaults_to_binary_match() -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    manager.config = {"device_manager": {}}

    assert manager._get_sync_mode() == BINARY_MATCH


@pytest.mark.parametrize("mode", ["camera_align", "camera", "camera_sync"])
def test_camera_alignment_names_resolve_to_canonical_mode(mode: str) -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    manager.config = {"device_manager": {"sync_mode": mode}}

    assert manager._get_sync_mode() == CAMERA_ALIGN


def test_camera_alignment_rejects_removed_mode_name() -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    manager.config = {"device_manager": {"sync_mode": "camera_interpolate"}}

    with pytest.raises(ValueError, match="sync_mode must be 'binary_match' or 'camera_align'"):
        manager._get_sync_mode()


def test_camera_sync_rejects_untyped_low_dim_device() -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    manager.master_device_class_id = 0
    manager.devices = [
        {
            "device_class": "WebCameraOpenPi",
            "device_class_id": 0,
            "config": _camera_config(),
        },
        {
            "device_class": "RizonRobot",
            "device_class_id": 2,
            "config": {"state_feedback": {"fps": 120.0}},
            "data_shape": (44,),
        },
    ]

    with pytest.raises(ValueError, match="state_feedback.format"):
        manager._validate_camera_interpolation_contract()


def test_camera_sync_uses_low_dim_interpolation_and_keeps_other_camera_nearest() -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    manager.master_device_class_id = 0
    manager.devices = [
        {
            "device_class": "WebCameraOpenPi",
            "device_class_id": 0,
            "config": _camera_config(),
        },
        {
            "device_class": "WebCameraOpenPi",
            "device_class_id": 1,
            "config": _camera_config(),
        },
        {
            "device_class": "RizonRobot",
            "device_class_id": 2,
            "config": {
                "state_feedback": {
                    "format": {
                        "shape": [7],
                        "type": ["xyz", "qwqxqyqz"],
                        "dtype": "float64",
                    }
                }
            },
        },
    ]
    manager._connect_to_device = lambda _device: True
    manager._calculate_adjusted_timestamp = lambda timestamp, _device: timestamp
    manager._search_device_buffer_for_timestamp = (
        lambda _device, _timestamp: (6, np.array([6], dtype=np.uint8), 0)
    )
    manager._search_device_buffer_bracket = lambda _device, _timestamp: (
        (0, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 0),
        (10, np.array([2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 1.0]), 1),
    )
    master_frame = (5, np.array([5], dtype=np.uint8))

    frames = manager._find_camera_synchronized_frames(5, master_frame)

    assert frames is not None
    assert frames[0] is master_frame
    assert frames[1][0] == 6
    assert frames[2][0] == 5
    np.testing.assert_allclose(frames[2][1][:3], [1.0, 2.0, 3.0])
    assert np.linalg.norm(frames[2][1][3:]) == pytest.approx(1.0)


def test_camera_sync_bracket_search_handles_a_wrapped_ring() -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    device = {"device_class_id": 2}
    frames = {
        0: (40, np.array([40.0])),
        1: (50, np.array([50.0])),
        2: (20, np.array([20.0])),
        3: (30, np.array([30.0])),
    }
    manager._get_device_buffer_info = lambda _device: (4, 4, 2, 16, (1,))
    manager._read_device_frame = lambda _device, index: frames[index]

    bracket = manager._search_device_buffer_bracket(device, 35)

    assert bracket is not None
    assert [frame[0] for frame in bracket] == [30, 40]
    assert manager._search_device_buffer_bracket(device, 10) is None
    assert manager._search_device_buffer_bracket(device, 60) is None


@pytest.mark.parametrize("next_camera_timestamp", [5, 6])
def test_camera_sync_retries_until_robot_bracket_arrives(next_camera_timestamp: int) -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    manager.master_device_class_id = 0
    manager.master_last_timestamp = None
    manager.sync_mode = CAMERA_ALIGN
    camera = {"device_class": "WebCameraOpenPi", "device_class_id": 0, "config": _camera_config()}
    robot = {
        "device_class": "RizonRobot",
        "device_class_id": 2,
        "config": {
            "state_feedback": {
                "format": {"shape": [1], "type": ["linear:1"], "dtype": "float64"}
            }
        },
    }
    manager.devices = [camera, robot]
    camera_frame = (5, np.zeros((2, 2, 3), dtype=np.uint8))
    robot_frames = [(0, np.array([0.0]))]
    manager._read_device_frame = lambda device, index=None: (
        camera_frame if device is camera else robot_frames[index]
    )
    manager._get_device_buffer_info = lambda _: (4, len(robot_frames), len(robot_frames), 16, (1,))
    manager._connect_to_device = lambda _: True
    manager._calculate_adjusted_timestamp = lambda timestamp, _: timestamp

    # Missing the following robot sample must neither extrapolate nor consume
    # the camera frame, including when several polls fail in succession.
    assert manager._read_all_device_frames() is None
    assert manager._read_all_device_frames() is None
    assert manager.master_last_timestamp is None

    # Retry the same frame, or prefer a newer camera frame if one arrived.
    camera_frame = (next_camera_timestamp, camera_frame[1])
    robot_frames.append((10, np.array([10.0])))
    frames = manager._read_all_device_frames()
    assert frames is not None
    assert frames[0] is camera_frame
    assert frames[1][0] == next_camera_timestamp
    np.testing.assert_allclose(frames[1][1], [float(next_camera_timestamp)])
    assert manager.master_last_timestamp == next_camera_timestamp
    assert manager._read_all_device_frames() is None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("horizon_ns", [1, 12_500_000, 12_500_001])
def test_camera_sync_forward_projection_is_bounded_and_does_not_consume_failures(
    wrapped: bool, horizon_ns: int
) -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    manager.master_device_class_id = 0
    manager.master_last_timestamp = None
    manager.sync_mode = CAMERA_ALIGN
    camera = {"device_class": "WebCameraOpenPi", "device_class_id": 0, "config": _camera_config()}
    robot = {
        "device_class": "RizonRobot", "device_class_id": 2,
        "config": {"state_feedback": {
            "fps": 120,
            "format": {"shape": [1], "type": ["scalar"], "dtype": "float64"},
        }},
    }
    manager.devices = [camera, robot]
    chronological = [(10_000_000, np.array([1.])), (20_000_000, np.array([2.]))]
    frames = chronological[::-1] if wrapped else chronological
    target = 20_000_000 + horizon_ns
    camera_frame = (target, np.zeros((2, 2, 3), dtype=np.uint8))
    manager._read_device_frame = lambda device, index=None: (
        camera_frame if device is camera else frames[index]
    )
    manager._get_device_buffer_info = lambda _: (
        (2, 2, 1, 16, (1,)) if wrapped else (4, 2, 2, 16, (1,))
    )
    manager._connect_to_device = lambda _: True
    manager._calculate_adjusted_timestamp = lambda timestamp, _: timestamp

    result = manager._read_all_device_frames()
    if horizon_ns > 12_500_000:
        assert result is None
        assert manager.master_last_timestamp is None
        # Later robot data makes this same camera frame eligible for interpolation.
        frames[:] = [(20_000_000, np.array([2.])), (40_000_000, np.array([4.]))]
        manager._get_device_buffer_info = lambda _: (2, 2, 0, 16, (1,))
        result = manager._read_all_device_frames()
    assert result is not None
    assert result[1][0] == target
    np.testing.assert_allclose(result[1][1], [target / 10_000_000])
    assert manager.master_last_timestamp == target
    assert manager._read_all_device_frames() is None


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf"), "invalid"])
def test_forward_projection_requires_valid_configured_rate(fps) -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    device = {"device_class": "RizonRobot", "config": {"state_feedback": {"fps": fps}}}
    data_format = parse_data_format({"shape": [1], "type": ["scalar"]})
    assert manager._extrapolate_latest_robot_frame(device, 30, data_format) is None


@pytest.mark.parametrize("timestamps,target", [([10, 10], 11), ([20, 10], 21), ([10, 20], 5)])
def test_forward_projection_rejects_invalid_time_order(timestamps, target) -> None:
    manager = object.__new__(SynchronizedDeviceManager)
    device = {"device_class": "RizonRobot", "config": {"state_feedback": {"fps": 120}}}
    manager._get_device_buffer_info = lambda _: (2, 2, 0, 16, (1,))
    manager._read_device_frame = lambda _, index: (timestamps[index], np.array([float(index)]))
    data_format = parse_data_format({"shape": [1], "type": ["scalar"]})
    assert manager._extrapolate_latest_robot_frame(device, target, data_format) is None


@pytest.mark.parametrize("order", ["qwqxqyqz", "qxqyqzqw"])
@pytest.mark.parametrize("sign", [1, -1])
def test_forward_projection_preserves_angular_velocity_and_quaternion_order(order, sign) -> None:
    from manager.utils.data_interpolation import extrapolate_low_dim_data

    data_format = parse_data_format({"shape": [8], "type": ["xyz", order, "dq1-1"]})
    start_q = Rotation.from_euler("xyz", [15, 20, 30], degrees=True)
    delta = Rotation.from_euler("z", 10, degrees=True)
    end_q = start_q * delta
    indices = [3, 0, 1, 2] if order == "qwqxqyqz" else [0, 1, 2, 3]
    start = np.array([0., 0., 0., *start_q.as_quat()[indices], 1.], dtype=np.float32)
    end = np.array([1., 2., 3., *(sign * end_q.as_quat()[indices]), 2.], dtype=np.float32)
    result = extrapolate_low_dim_data(start, end, 2.5, data_format)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:3], [2.5, 5., 7.5])
    expected = (start_q * Rotation.from_euler("z", 25, degrees=True)).as_quat()[indices]
    assert abs(np.dot(result[3:7], expected)) == pytest.approx(1.)
    assert np.linalg.norm(result[3:7]) == pytest.approx(1.)
    assert result[7] == 2.


def test_extrapolation_rejects_nonfinite_and_interpolation_stays_in_range() -> None:
    from manager.utils.data_interpolation import LowDimInterpolationError, extrapolate_low_dim_data

    data_format = parse_data_format({"shape": [1], "type": ["scalar"]})
    for ratio in [0.5, float("nan"), float("inf")]:
        with pytest.raises(LowDimInterpolationError):
            extrapolate_low_dim_data([0.], [1.], ratio, data_format)
    with pytest.raises(LowDimInterpolationError):
        extrapolate_low_dim_data([0.], [float("nan")], 1.5, data_format)
    with pytest.raises(LowDimInterpolationError):
        interpolate_low_dim_data([0.], [1.], 1.5, data_format)
