"""USB link-speed checks shared by V4L2 camera adapters and launch preflight."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union


CaptureId = Union[int, str]
USB_PORT_REQUIREMENTS = frozenset({"any", "usb_2.0", "usb_3.0"})
USB2_HIGH_SPEED_MBPS = 480.0
USB3_MIN_SPEED_MBPS = 5000.0


class UsbPortRequirementError(RuntimeError):
    """Raised when a camera is connected at a disallowed USB link speed."""


def video_device_name(capture_id: CaptureId) -> Optional[str]:
    """Resolve an OpenCV index/path to a ``videoN`` sysfs node name."""
    if isinstance(capture_id, bool):
        return None
    if isinstance(capture_id, int):
        return f"video{capture_id}" if capture_id >= 0 else None

    text = str(capture_id).strip()
    if text.isdecimal():
        return f"video{int(text)}"
    try:
        resolved = Path(text).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    name = resolved.name
    if name.startswith("video") and name[5:].isdecimal():
        return name
    return None


def usb_speed_mbps(
    video_index: CaptureId,
    *,
    sysfs_root: Path = Path("/sys/class/video4linux"),
) -> Optional[float]:
    """Return the negotiated USB link speed for a V4L2 video node.

    The value is read from the nearest USB device ``speed`` file in sysfs. A
    missing value means that the node is not USB-backed or is unavailable.
    ``video_index`` may be an integer, ``/dev/videoN``, or a stable V4L2 path.
    """
    video_name = video_device_name(video_index)
    if video_name is None:
        return None

    device_link = sysfs_root / video_name / "device"
    try:
        device_node = device_link.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None

    for parent in (device_node, *device_node.parents):
        speed_file = parent / "speed"
        try:
            if speed_file.is_file():
                speed = float(speed_file.read_text(encoding="utf-8").strip())
                return speed if math.isfinite(speed) else None
        except (OSError, ValueError, OverflowError):
            continue
    return None


def normalize_usb_port_requirement(requirement: str) -> str:
    """Validate and normalize a configured negotiated-link requirement."""
    if not isinstance(requirement, str):
        raise ValueError(
            f"Unsupported usb_port_requirement={requirement!r}; "
            f"expected one of: {', '.join(sorted(USB_PORT_REQUIREMENTS))}"
        )
    normalized = requirement.strip().casefold()
    if normalized not in USB_PORT_REQUIREMENTS:
        raise ValueError(
            f"Unsupported usb_port_requirement={requirement!r}; "
            f"expected one of: {', '.join(sorted(USB_PORT_REQUIREMENTS))}"
        )
    return normalized


def validate_usb_port_requirement(
    requirement: str,
    speed_mbps: Optional[float],
    *,
    camera_id: int,
    capture_id: CaptureId,
) -> None:
    """Raise when a configured negotiated-link requirement is not met."""
    normalized = normalize_usb_port_requirement(requirement)
    if normalized == "any":
        return
    if speed_mbps is None:
        raise UsbPortRequirementError(
            f"Camera {camera_id} requires {normalized}, but USB link speed could "
            f"not be determined for capture_id={capture_id!r}"
        )

    if normalized == "usb_2.0":
        accepted = speed_mbps == USB2_HIGH_SPEED_MBPS
    else:
        accepted = speed_mbps >= USB3_MIN_SPEED_MBPS
    if not accepted:
        raise UsbPortRequirementError(
            f"Camera {camera_id} requires {normalized}, but negotiated USB speed "
            f"is {speed_mbps:g} Mbps (capture_id={capture_id!r})"
        )
