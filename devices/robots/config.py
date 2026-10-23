"""Structured runtime configuration for robot state and control streams.

The control contract deliberately keeps shared-memory topology separate from
the command layout.  A source endpoint is described by the owning device,
instance id, and logical port; the concrete shared-memory name is derived only
when a device instance is available.  This prevents raw SHM names from being
copied between configurations and makes source arbitration explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from devices.config import ShmEndpointConfig
from utils.data_format import DataFormat, parse_data_format


CONTROL_TYPES = ("joints", "eef_abs", "eef_rel")
SOURCE_SELECTION_MODES = ("single_source", "dual_source")
SOURCE_PORTS = ("state", "command")
SOURCE_ROLES = ("policy", "teleoperation")
DUAL_SOURCE_SELECTOR_ROLES = {"0": "policy", "1": "teleoperation"}
_SELF_DEVICE = "self"


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], *, allowed: set[str], field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"{field} contains unknown fields: {', '.join(sorted(map(str, unknown)))}"
        )


@dataclass(frozen=True)
class DualSourceSelectorConfig:
    """Flag endpoint for the fixed binary dual-source protocol."""

    endpoint: ShmEndpointConfig

    @classmethod
    def from_mapping(
        cls,
        value: "DualSourceSelectorConfig | Mapping[str, Any]",
        *,
        field: str = "control.source_selection.dual_source_selector",
    ) -> "DualSourceSelectorConfig":
        if isinstance(value, cls):
            return cls.from_mapping({"endpoint": value.endpoint}, field=field)
        mapping = _require_mapping(value, field=field)
        _reject_unknown(mapping, allowed={"endpoint"}, field=field)
        if "endpoint" not in mapping:
            raise ValueError(f"{field}.endpoint is required")

        endpoint = ShmEndpointConfig.from_mapping(
            mapping["endpoint"],
            field=f"{field}.endpoint",
            allowed_ports=("control_flag",),
        )
        return cls(endpoint=endpoint)


@dataclass(frozen=True)
class SourceSelectionConfig:
    """Validated single- or dual-source command arbitration contract."""

    mode: str
    source: ShmEndpointConfig | None = None
    sources: dict[str, ShmEndpointConfig] | None = None
    dual_source_selector: DualSourceSelectorConfig | None = None

    @classmethod
    def default(cls) -> "SourceSelectionConfig":
        """Return the implicit self-owned command source."""
        return cls(
            mode="single_source",
            source=ShmEndpointConfig(device=_SELF_DEVICE, device_class_id=None, port="command"),
        )

    @classmethod
    def from_mapping(
        cls,
        value: "SourceSelectionConfig | Mapping[str, Any] | None",
        *,
        field: str = "control.source_selection",
    ) -> "SourceSelectionConfig":
        if value is None:
            return cls.default()
        if isinstance(value, cls):
            # Re-parse the public dataclass form as well: callers may build it
            # directly rather than through ``from_mapping``.
            raw: dict[str, Any] = {"mode": value.mode}
            if value.mode == "single_source":
                raw["source"] = value.source
            else:
                raw["sources"] = value.sources
                raw["dual_source_selector"] = value.dual_source_selector
            return cls.from_mapping(raw, field=field)
        mapping = _require_mapping(value, field=field)
        _reject_unknown(
            mapping,
            allowed={"mode", "source", "sources", "dual_source_selector"},
            field=field,
        )
        if "mode" not in mapping:
            raise ValueError(f"{field}.mode is required when source_selection is present")
        raw_mode = mapping["mode"]
        if not isinstance(raw_mode, str):
            raise ValueError(f"{field}.mode must be a string")
        mode = raw_mode.strip().casefold()
        if mode not in SOURCE_SELECTION_MODES:
            choices = ", ".join(SOURCE_SELECTION_MODES)
            raise ValueError(f"{field}.mode must be one of: {choices}")

        if mode == "single_source":
            if "source" not in mapping:
                raise ValueError(f"{field}.source is required for single_source")
            if "sources" in mapping or "dual_source_selector" in mapping:
                raise ValueError(
                    f"{field} single_source accepts only mode and source"
                )
            source = ShmEndpointConfig.from_mapping(
                mapping["source"],
                field=f"{field}.source",
                allowed_ports=SOURCE_PORTS,
            )
            return cls(mode=mode, source=source)

        if "sources" not in mapping:
            raise ValueError(f"{field}.sources is required for dual_source")
        if "dual_source_selector" not in mapping:
            raise ValueError(
                f"{field}.dual_source_selector is required for dual_source"
            )
        if "source" in mapping:
            raise ValueError(f"{field} dual_source accepts sources, not source")

        raw_sources = _require_mapping(mapping["sources"], field=f"{field}.sources")
        if set(raw_sources) != set(SOURCE_ROLES):
            raise ValueError(
                f"{field}.sources must contain exactly: {', '.join(SOURCE_ROLES)}"
            )
        sources = {
            role: ShmEndpointConfig.from_mapping(
                raw_sources[role],
                field=f"{field}.sources.{role}",
                allowed_ports=SOURCE_PORTS,
            )
            for role in SOURCE_ROLES
        }
        selector = DualSourceSelectorConfig.from_mapping(
            mapping["dual_source_selector"],
            field=f"{field}.dual_source_selector",
        )
        return cls(mode=mode, sources=sources, dual_source_selector=selector)

    def resolve_source(
        self,
        role: str | None = None,
        *,
        owner_device: str,
        owner_device_class_id: int,
    ) -> str:
        """Resolve one configured source to its canonical SHM name."""
        if self.mode == "single_source":
            if self.source is None:
                raise ValueError("single_source configuration has no source")
            endpoint = self.source
        else:
            if role not in SOURCE_ROLES:
                raise ValueError(
                    "dual_source resolution requires role 'policy' or 'teleoperation'"
                )
            if self.sources is None or role not in self.sources:
                raise ValueError(f"dual_source configuration has no {role} source")
            endpoint = self.sources[role]
        return endpoint.resolve_shm_name(
            owner_device=owner_device,
            owner_device_class_id=owner_device_class_id,
        )


def _positive_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
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
class RobotControlConfig:
    """Validated command stream layout, sources, and execution settings."""

    type: str
    format: DataFormat
    execution_mode: str
    fps: float
    buffer_size: int
    source_selection: SourceSelectionConfig
    cartesian_impedance: tuple[float, ...] | None

    @classmethod
    def require(
        cls, value: "RobotControlConfig | Mapping[str, Any]"
    ) -> "RobotControlConfig":
        """Accept a parsed contract or parse a direct-construction mapping once."""
        if isinstance(value, cls):
            return cls(
                type=value.type,
                format=value.format,
                execution_mode=value.execution_mode,
                fps=value.fps,
                buffer_size=value.buffer_size,
                source_selection=SourceSelectionConfig.from_mapping(
                    value.source_selection
                ),
                cartesian_impedance=value.cartesian_impedance,
            )
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RobotControlConfig":
        if not isinstance(value, Mapping):
            raise ValueError("control must be a mapping")
        _reject_unknown(
            value,
            allowed={
                "type",
                "format",
                "execution",
                "buffer_size",
                "source_selection",
                "cartesian_impedance",
            },
            field="control",
        )
        if "type" not in value:
            raise ValueError("control.type is required")
        if "format" not in value:
            raise ValueError("control.format is required")

        execution = value.get("execution", {})
        if not isinstance(execution, Mapping):
            raise ValueError("control.execution must be a mapping")
        unknown_execution = set(execution) - {"mode", "fps"}
        if unknown_execution:
            raise ValueError(
                "control.execution contains unknown fields: "
                + ", ".join(sorted(unknown_execution))
            )
        execution_mode = str(execution.get("mode", "synchronous")).casefold()
        if execution_mode not in {"synchronous", "legacy_async"}:
            raise ValueError(
                "control.execution.mode must be synchronous or legacy_async"
            )

        cartesian_impedance = value.get("cartesian_impedance")
        impedance_values: tuple[float, ...] | None = None
        if cartesian_impedance is not None:
            if not isinstance(cartesian_impedance, Mapping):
                raise ValueError("control.cartesian_impedance must be a mapping")
            unknown_impedance = set(cartesian_impedance) - {
                "translation_n_m",
                "rotation_nm_rad",
            }
            if unknown_impedance:
                raise ValueError(
                    "control.cartesian_impedance contains unknown fields: "
                    + ", ".join(sorted(unknown_impedance))
                )
            translation = tuple(
                float(item)
                for item in cartesian_impedance.get("translation_n_m", ())
            )
            rotation = tuple(
                float(item)
                for item in cartesian_impedance.get("rotation_nm_rad", ())
            )
            if len(translation) != 3 or len(rotation) != 3:
                raise ValueError(
                    "control.cartesian_impedance translation and rotation must "
                    "each contain 3 values"
                )
            if any(item <= 0.0 for item in (*translation, *rotation)):
                raise ValueError("control.cartesian_impedance values must be positive")
            impedance_values = (*translation, *rotation)

        control_type = value["type"]
        if not isinstance(control_type, str) or control_type not in CONTROL_TYPES:
            raise ValueError(
                "control.type must be one of: " + ", ".join(CONTROL_TYPES)
            )

        return cls(
            type=control_type,
            format=parse_data_format(
                value["format"], require_dtype=True, context="control.format"
            ),
            execution_mode=execution_mode,
            fps=_positive_float(
                execution.get("fps", 100.0), field="control.execution.fps"
            ),
            buffer_size=_positive_int(
                value.get("buffer_size", 50), field="control.buffer_size"
            ),
            source_selection=SourceSelectionConfig.from_mapping(
                value.get("source_selection")
            ),
            cartesian_impedance=impedance_values,
        )


__all__ = [
    "CONTROL_TYPES",
    "SOURCE_SELECTION_MODES",
    "SOURCE_PORTS",
    "SOURCE_ROLES",
    "ShmEndpointConfig",
    "DualSourceSelectorConfig",
    "SourceSelectionConfig",
    "RobotControlConfig",
]
