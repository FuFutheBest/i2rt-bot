"""Hardware-free USB negotiated-link tests for V4L2 cameras."""

from __future__ import annotations

from pathlib import Path

import pytest

from devices.cameras.usb_link import (
    UsbPortRequirementError,
    normalize_usb_port_requirement,
    usb_speed_mbps,
    validate_usb_port_requirement,
    video_device_name,
)


@pytest.mark.parametrize("value", ["USB_2.0", " usb_3.0 ", "ANY"])
def test_normalize_usb_port_requirement(value: str) -> None:
    assert normalize_usb_port_requirement(value) == value.strip().casefold()


@pytest.mark.parametrize("value", ["usb_1.1", "", None])
def test_normalize_usb_port_requirement_rejects_unknown_value(value: object) -> None:
    with pytest.raises(ValueError, match="usb_port_requirement"):
        normalize_usb_port_requirement(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("requirement", "speed"),
    [
        ("usb_2.0", 480.0),
        ("usb_3.0", 5000.0),
        ("usb_3.0", 10000.0),
        ("any", None),
    ],
)
def test_validate_usb_port_requirement_accepts_matching_speed(
    requirement: str,
    speed: float | None,
) -> None:
    validate_usb_port_requirement(
        requirement,
        speed,
        camera_id=7,
        capture_id="/dev/video7",
    )


@pytest.mark.parametrize(
    ("requirement", "speed"),
    [("usb_2.0", 5000.0), ("usb_3.0", 480.0), ("usb_2.0", None)],
)
def test_validate_usb_port_requirement_rejects_mismatched_speed(
    requirement: str,
    speed: float | None,
) -> None:
    with pytest.raises(UsbPortRequirementError, match=requirement):
        validate_usb_port_requirement(
            requirement,
            speed,
            camera_id=7,
            capture_id="/dev/video7",
        )


def test_video_device_name_supports_index_and_stable_path(tmp_path: Path) -> None:
    assert video_device_name(3) == "video3"
    assert video_device_name("003") == "video3"
    assert video_device_name("/dev/video4") == "video4"

    stable_path = tmp_path / "by-path-camera"
    stable_path.symlink_to("/dev/video8")
    assert video_device_name(str(stable_path)) == "video8"


def test_usb_speed_mbps_walks_to_nearest_usb_speed(tmp_path: Path) -> None:
    sysfs_root = tmp_path / "sys" / "class" / "video4linux"
    video_dir = sysfs_root / "video7"
    usb_device = tmp_path / "sys" / "devices" / "usb1" / "1-7"
    usb_interface = usb_device / "1-7:1.0"
    video_dir.mkdir(parents=True)
    usb_interface.mkdir(parents=True)
    (video_dir / "device").symlink_to(usb_interface)
    (usb_device / "speed").write_text("5000\n", encoding="utf-8")

    assert usb_speed_mbps(7, sysfs_root=sysfs_root) == 5000.0


def test_usb_speed_mbps_returns_none_for_missing_or_invalid_speed(tmp_path: Path) -> None:
    sysfs_root = tmp_path / "sys" / "class" / "video4linux"
    video_dir = sysfs_root / "video9"
    usb_interface = tmp_path / "sys" / "devices" / "usb1" / "1-9" / "1-9:1.0"
    video_dir.mkdir(parents=True)
    usb_interface.mkdir(parents=True)
    (video_dir / "device").symlink_to(usb_interface)
    (usb_interface.parent / "speed").write_text("not-a-speed\n", encoding="utf-8")

    assert usb_speed_mbps(9, sysfs_root=sysfs_root) is None
    assert usb_speed_mbps(10, sysfs_root=sysfs_root) is None
