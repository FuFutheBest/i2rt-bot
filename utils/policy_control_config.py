"""Structured policy control-target configuration.

Policy and action-executor configuration names logical command endpoints rather
than copying raw shared-memory names.  Runtime code resolves the canonical name
only after this contract has been validated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from devices.config import ShmEndpointConfig
from utils.contracts import make_device_key, normalize_shape, validate_dtype_name


@dataclass(frozen=True)
class ControlTargetConfig:
    """One policy-writable robot command endpoint and its payload contract."""

    endpoint: ShmEndpointConfig
    shape: tuple[int, ...]
    dtype: str = "float64"

    def __post_init__(self) -> None:
        endpoint = ShmEndpointConfig.from_mapping(
            self.endpoint,
            field="policy control endpoint",
            allowed_ports=("command",),
        )
        if endpoint.device == "self":
            raise ValueError("policy control endpoint.device cannot be 'self'")
        if endpoint.device_class_id is None:  # Defensive: ShmEndpointConfig validates this.
            raise ValueError("policy control endpoint.device_class_id is required")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "shape", normalize_shape(self.shape))
        object.__setattr__(self, "dtype", validate_dtype_name(self.dtype))

    @classmethod
    def from_mapping(
        cls,
        value: "ControlTargetConfig | Mapping[str, Any]",
        *,
        field: str = "policy.controls[]",
    ) -> "ControlTargetConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be a mapping")
        if "shm" in value:
            raise ValueError(f"{field}.shm was removed; use {field}.endpoint")
        unknown = set(value) - {"endpoint", "shape", "dtype"}
        if unknown:
            raise ValueError(
                f"{field} contains unknown fields: {', '.join(sorted(map(str, unknown)))}"
            )
        if "endpoint" not in value:
            raise ValueError(f"{field}.endpoint is required")
        if "shape" not in value:
            raise ValueError(f"{field}.shape is required")
        shape = value["shape"]
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise ValueError(f"{field}.shape must be a sequence")
        endpoint = ShmEndpointConfig.from_mapping(
            value["endpoint"],
            field=f"{field}.endpoint",
            allowed_ports=("command",),
        )
        return cls(
            endpoint=endpoint,
            shape=tuple(shape),
            dtype=str(value.get("dtype", "float64")),
        )

    @property
    def device_key(self) -> str:
        """Canonical policy action key, for example ``RizonRobot_1``."""
        assert self.endpoint.device_class_id is not None
        return make_device_key(self.endpoint.device, self.endpoint.device_class_id)

    @property
    def shm_name(self) -> str:
        """Canonical command SHM name derived from the structured endpoint."""
        assert self.endpoint.device_class_id is not None
        return self.endpoint.resolve_shm_name(
            owner_device=self.endpoint.device,
            owner_device_class_id=self.endpoint.device_class_id,
        )

    def runtime_mapping(self) -> dict[str, Any]:
        """Return normalized fields consumed by policy and action-executor code."""
        return {
            "endpoint": self.endpoint,
            "device_key": self.device_key,
            "shm_name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    def config_mapping(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe representation without a raw SHM name."""
        assert self.endpoint.device_class_id is not None
        return {
            "endpoint": {
                "device": self.endpoint.device,
                "device_class_id": self.endpoint.device_class_id,
                "port": self.endpoint.port,
            },
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


def parse_control_targets(
    value: Any,
    *,
    field: str,
) -> list[ControlTargetConfig]:
    """Parse a non-null policy/action-executor control-target sequence."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return [
        ControlTargetConfig.from_mapping(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
