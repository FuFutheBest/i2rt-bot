"""Hardware-free contracts for OpenPI camera USB link requirements."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip(
    "cv2",
    reason="OpenPI USB camera adapter tests require OpenCV imports to run",
)

openpi = importlib.import_module("devices.cameras.openpi")


def _camera(requirement: str) -> openpi.WebCameraOpenPi:
    camera = openpi.WebCameraOpenPi.__new__(openpi.WebCameraOpenPi)
    camera.device_class_id = 7
    camera.capture_id = "/dev/video7"
    camera.usb_port_requirement = requirement
    camera.negotiated_usb_speed_mbps = None
    return camera


@pytest.mark.parametrize(
    ("requirement", "speed"),
    [
        ("usb_2.0", 480.0),
        ("usb_3.0", 5000.0),
        ("usb_3.0", 10000.0),
        ("any", None),
    ],
)
def test_usb_port_requirement_accepts_matching_speed(
    monkeypatch: pytest.MonkeyPatch,
    requirement: str,
    speed: float | None,
) -> None:
    monkeypatch.setattr(openpi, "usb_speed_mbps", lambda _index: speed)

    camera = _camera(requirement)
    camera._check_usb_port_requirement()

    assert camera.negotiated_usb_speed_mbps == speed


def test_usb_port_requirement_any_skips_sysfs_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_lookup(_index: object) -> None:
        raise AssertionError("any should not query sysfs")

    monkeypatch.setattr(openpi, "usb_speed_mbps", unexpected_lookup)

    camera = _camera("any")
    camera._check_usb_port_requirement()

    assert camera.negotiated_usb_speed_mbps is None


def test_usb_port_requirement_rejects_before_opening_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _camera("usb_2.0")
    camera.cap = None
    monkeypatch.setattr(openpi, "usb_speed_mbps", lambda _index: 5000.0)
    monkeypatch.setattr(
        openpi.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: pytest.fail("capture should not open"),
    )

    with pytest.raises(openpi.UsbPortRequirementError, match="usb_2.0"):
        camera._initialize_camera()


@pytest.mark.parametrize(
    ("requirement", "speed"),
    [("usb_2.0", 5000.0), ("usb_3.0", 480.0), ("usb_2.0", None)],
)
def test_usb_port_requirement_rejects_mismatched_speed(
    monkeypatch: pytest.MonkeyPatch,
    requirement: str,
    speed: float | None,
) -> None:
    monkeypatch.setattr(openpi, "usb_speed_mbps", lambda _index: speed)

    with pytest.raises(openpi.UsbPortRequirementError, match=requirement):
        _camera(requirement)._check_usb_port_requirement()


@pytest.mark.parametrize("value", ["USB_2.0", " usb_3.0 ", "ANY"])
def test_usb_port_requirement_is_normalized(value: str) -> None:
    normalized = openpi.WebCameraOpenPi._normalize_usb_port_requirement(value)
    assert normalized == value.strip().casefold()


def test_usb_port_requirement_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="usb_port_requirement"):
        openpi.WebCameraOpenPi._normalize_usb_port_requirement("usb_1.1")
