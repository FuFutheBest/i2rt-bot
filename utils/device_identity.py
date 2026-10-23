"""Configuration-owned logical identities, independent of capture hardware."""

from collections.abc import Mapping, Sequence
from typing import Any


def require_capture_id(value: Any) -> int | str:
    """Accept explicit numeric indices or nonempty paths, including env strings."""
    if isinstance(value, str):
        value = value.strip()
        if value.isdecimal():
            return int(value)
        if value:
            return value
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError("capture_id is required: use a non-negative index or device path")


def require_device_class_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("device_class_id must be a non-negative integer")
    return value


def validate_device_class_ids(devices: Sequence[Mapping[str, Any]]) -> None:
    """Validate the complete list before disabled devices are filtered out."""
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise ValueError("devices must be a list of device configurations")
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping):
            raise ValueError(f"devices[{index}] must be a mapping")
        if "device_id" in device:
            raise ValueError("device_id was removed; use device_class_id")
        value = require_device_class_id(device.get("device_class_id"))
        if value != index:
            raise ValueError(
                f"devices[{index}].device_class_id must be {index}, got {value}; "
                "IDs must follow the configured list order from zero"
            )
