"""NEDF camera roles are independent of global SHM IDs and USB enumeration."""

from collections.abc import Mapping, Set
from typing import Any


def nedf_camera_entries(
    devices: Mapping[str, Mapping[str, Any]], camera_types: Set[str]
) -> list[tuple[int, str, Mapping[str, Any]]]:
    """Map configured camera order to right=0, left=1, head=2.

    OpenPI presets list left then right; the R1/Windows collection presets
    list right then left. Non-recorded devices never consume camera positions.
    """
    cameras = [(key, info) for key, info in devices.items() if info["type"] in camera_types]
    if len(cameras) > 3:
        raise ValueError("NEDF supports at most three recorded camera roles")
    openpi = [info["type"] == "WebCameraOpenPi" for _, info in cameras]
    if any(openpi) and not all(openpi):
        raise ValueError("Mixed OpenPI/collection camera order needs explicit role mapping")
    order = (1, 0, 2) if any(openpi) else (0, 1, 2)
    return sorted((order[index], key, info) for index, (key, info) in enumerate(cameras))
