"""Typed construction contracts shared by every device."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from devices.contracts import DeviceKind, parse_device_kind
from utils.data_format import (
    DataFormat,
    ImageFormat,
    parse_data_format,
    parse_image_format,
)


StateFormat = DataFormat | ImageFormat


SHM_ENDPOINT_PORTS = ("state", "command", "control_flag")
_SHM_PORT_SUFFIXES = {
    "state": "data",
    "command": "control",
    "control_flag": "control_flag",
}


def _parse_non_negative_int(value: Any, *, field: str) -> int:
    """Parse a concrete device id without accepting booleans or fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if number != value or number < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return number


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _reject_unknown(
    value: Mapping[str, Any], *, allowed: set[str], field: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"{field} contains unknown fields: {', '.join(sorted(map(str, unknown)))}"
        )


@dataclass(frozen=True)
class ShmEndpointConfig:
    """A logical shared-memory endpoint owned by one device instance.

    ``device: self`` refers to the device owning the surrounding contract.
    Other device classes must provide an explicit ``device_class_id``.  The logical
    port is translated to the canonical SHM suffix only when the endpoint is
    resolved against an owner.
    """

    device: str
    device_class_id: int | None
    port: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.device, str)
            and self.device.strip().casefold() == "self"
            and self.device_class_id is not None
        ):
            raise ValueError(
                "endpoint.device_class_id must be omitted when endpoint.device is 'self'"
            )

    @classmethod
    def from_mapping(
        cls,
        value: "ShmEndpointConfig | Mapping[str, Any]",
        *,
        field: str = "endpoint",
        allowed_ports: Sequence[str] = SHM_ENDPOINT_PORTS,
    ) -> "ShmEndpointConfig":
        if isinstance(value, cls):
            raw: dict[str, Any] = {
                "device": value.device,
                "port": value.port,
            }
            if value.device_class_id is not None:
                raw["device_class_id"] = value.device_class_id
            return cls.from_mapping(
                raw,
                field=field,
                allowed_ports=allowed_ports,
            )
        mapping = _require_mapping(value, field=field)
        _reject_unknown(mapping, allowed={"device", "device_class_id", "port"}, field=field)

        raw_device = mapping.get("device")
        if not isinstance(raw_device, str) or not raw_device.strip():
            raise ValueError(f"{field}.device must be a non-empty string")
        device = raw_device.strip()

        raw_port = mapping.get("port")
        if not isinstance(raw_port, str):
            raise ValueError(f"{field}.port must be a string")
        port = raw_port.strip().casefold()
        allowed = tuple(allowed_ports)
        if port not in allowed:
            choices = ", ".join(allowed)
            raise ValueError(f"{field}.port must be one of: {choices}")

        has_device_class_id = "device_class_id" in mapping and mapping["device_class_id"] is not None
        device_class_id = (
            _parse_non_negative_int(mapping["device_class_id"], field=f"{field}.device_class_id")
            if has_device_class_id
            else None
        )
        if device == "self" and device_class_id is not None:
            raise ValueError(
                f"{field}.device_class_id must be omitted when {field}.device is 'self'"
            )
        if device != "self" and device_class_id is None:
            raise ValueError(
                f"{field}.device_class_id is required when {field}.device is not 'self'"
            )

        return cls(device=device, device_class_id=device_class_id, port=port)

    def resolve_shm_name(self, *, owner_device: str, owner_device_class_id: int) -> str:
        """Resolve this endpoint to its canonical shared-memory name."""
        if not isinstance(owner_device, str) or not owner_device.strip():
            raise ValueError("owner_device must be a non-empty string")
        owner_id = _parse_non_negative_int(owner_device_class_id, field="owner_device_class_id")
        if self.device == "self":
            device = owner_device.strip()
            device_class_id = owner_id
        else:
            device = self.device
            if self.device_class_id is None:  # Defensive: construction validates this.
                raise ValueError("external endpoint is missing device_class_id")
            device_class_id = self.device_class_id
        return f"{device}_{device_class_id}_{_SHM_PORT_SUFFIXES[self.port]}"

    def resolve(self, *, owner_device: str, owner_device_class_id: int) -> str:
        """Short alias for :meth:`resolve_shm_name`."""
        return self.resolve_shm_name(
            owner_device=owner_device,
            owner_device_class_id=owner_device_class_id,
        )


def _positive_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return number


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number != value or number <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return number


@dataclass(frozen=True)
class StateFeedbackConfig:
    """Validated state stream required by every device constructor."""

    device_kind: DeviceKind
    format: StateFormat
    fps: float
    buffer_size: int
    hardware_latency_ms: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        device_kind: DeviceKind | str,
    ) -> "StateFeedbackConfig":
        if not isinstance(value, Mapping):
            raise ValueError("state_feedback must be a mapping")
        unknown = set(value) - {
            "format",
            "fps",
            "buffer_size",
            "hardware_latency_ms",
        }
        if unknown:
            raise ValueError(
                "state_feedback contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        required = {"format", "fps", "buffer_size", "hardware_latency_ms"}
        missing = required - set(value)
        if missing:
            raise ValueError(
                "state_feedback requires fields: fps, buffer_size, "
                "hardware_latency_ms, format; missing "
                + ", ".join(sorted(missing))
            )

        kind = parse_device_kind(device_kind)
        raw_format = value["format"]
        if kind is DeviceKind.CAMERA:
            stream_format: StateFormat = parse_image_format(raw_format)
        else:
            stream_format = parse_data_format(
                raw_format,
                require_dtype=True,
                context="state_feedback.format",
            )

        try:
            hardware_latency_ms = float(value["hardware_latency_ms"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "state_feedback.hardware_latency_ms must be a number"
            ) from exc
        if not math.isfinite(hardware_latency_ms) or hardware_latency_ms < 0.0:
            raise ValueError(
                "state_feedback.hardware_latency_ms must be non-negative and finite"
            )

        return cls(
            device_kind=kind,
            format=stream_format,
            fps=_positive_float(value["fps"], field="state_feedback.fps"),
            buffer_size=_positive_int(
                value["buffer_size"], field="state_feedback.buffer_size"
            ),
            hardware_latency_ms=hardware_latency_ms,
        )

    @classmethod
    def require(
        cls,
        value: "StateFeedbackConfig | Mapping[str, Any]",
        *,
        device_kind: DeviceKind | str,
    ) -> "StateFeedbackConfig":
        """Accept a parsed contract or parse a direct-construction mapping once."""
        kind = parse_device_kind(device_kind)
        if isinstance(value, cls):
            if value.device_kind is not kind:
                raise ValueError(
                    "state_feedback was parsed for "
                    f"{value.device_kind.value}, not {kind.value}"
                )
            return value
        return cls.from_mapping(value, device_kind=kind)


__all__ = [
    "SHM_ENDPOINT_PORTS",
    "ShmEndpointConfig",
    "StateFeedbackConfig",
    "StateFormat",
]
