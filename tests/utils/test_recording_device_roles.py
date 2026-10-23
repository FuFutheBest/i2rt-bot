import pytest

from utils.recording_device_roles import nedf_camera_entries


@pytest.mark.parametrize("camera_class,expected", [
    ("WebCameraDevice", [(0, 1), (1, 2), (2, 4)]),
    ("WebCameraOpenPi", [(0, 2), (1, 1), (2, 4)]),
])
def test_camera_roles_ignore_non_camera_ids_and_capture_nodes(camera_class, expected):
    devices = {
        "ZedStereoDevice_0": {"type": "ZedStereoDevice"},
        f"{camera_class}_1": {"type": camera_class, "capture_id": 99},
        f"{camera_class}_2": {"type": camera_class, "capture_id": 4},
        "RizonRobot_3": {"type": "RizonRobot"},
        f"{camera_class}_4": {"type": camera_class, "capture_id": "/dev/video80"},
    }
    entries = nedf_camera_entries(devices, {camera_class})
    assert [(index, key) for index, key, _ in entries] == [
        (index, f"{camera_class}_{logical_id}") for index, logical_id in expected
    ]


def test_ambiguous_mixed_camera_order_is_rejected():
    with pytest.raises(ValueError, match="explicit role mapping"):
        nedf_camera_entries({
            "WebCameraOpenPi_0": {"type": "WebCameraOpenPi"},
            "WebCameraDevice_1": {"type": "WebCameraDevice"},
        }, {"WebCameraOpenPi", "WebCameraDevice"})
