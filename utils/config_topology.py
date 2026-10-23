"""Static cross-device validation for structured shared-memory endpoints.

The endpoint contract intentionally names device instances instead of concrete
shared-memory strings.  This module verifies that those references can be
realized by the enabled device graph before any process, hardware adapter, or
shared-memory channel is started.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from devices import DEVICE_CLASSES
from devices.config import ShmEndpointConfig
from devices.robots.config import SourceSelectionConfig
from utils.device_config import (
    DeviceConfigError,
    ParsedDeviceConfig,
    parse_device_config,
)
from utils.errors import SHMContractError
from utils.inference_options import as_bool
from utils.policy_control_config import ControlTargetConfig, parse_control_targets


class ConfigTopologyError(ValueError):
    """Raised when enabled devices cannot satisfy an endpoint topology."""


DeviceKey = tuple[str, int]


@dataclass
class _DeviceRecord:
    """One enabled device declaration and its lazily parsed stream contract."""

    index: int
    key: DeviceKey
    raw: Mapping[str, Any]
    parsed: ParsedDeviceConfig | None = None
    parse_error_recorded: bool = False

    @property
    def label(self) -> str:
        return f"{self.key[0]}_{self.key[1]}"


def _device_enabled(device: Mapping[str, Any]) -> bool:
    """Match the launcher's device-level enabled semantics."""
    value = device.get("enabled", True)
    if isinstance(value, (list, tuple)):
        return any(as_bool(item) for item in value)
    return as_bool(value)


def _device_list(config: Mapping[str, Any]) -> Sequence[Any] | None:
    """Read devices from either composed or Hydra-grouped config shape."""
    devices = config.get("devices", [])
    if isinstance(devices, Mapping):
        devices = devices.get("devices", [])
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        return None
    return devices


def _device_key_from_mapping(device: Mapping[str, Any]) -> DeviceKey | None:
    """Return an exact configured device key without case-normalizing names."""
    device_class = device.get("class")
    if not isinstance(device_class, str) or not device_class:
        return None
    raw_id = device.get("device_class_id")
    if isinstance(raw_id, bool):
        return None
    try:
        device_class_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    if device_class_id < 0:
        return None
    return device_class, device_class_id


def _format_device_key(key: DeviceKey) -> str:
    return f"{key[0]}_{key[1]}"


def _resolve_endpoint_key(
    endpoint: ShmEndpointConfig, owner: _DeviceRecord
) -> DeviceKey:
    """Resolve ``self`` only against the enclosing config record."""
    if endpoint.device == "self":
        return owner.key
    assert endpoint.device_class_id is not None
    return endpoint.device, endpoint.device_class_id


def _parse_record(
    record: _DeviceRecord,
    *,
    errors: list[str],
) -> ParsedDeviceConfig | None:
    """Parse a stream contract once, recording a contextual static error."""
    if record.parsed is not None:
        return record.parsed
    try:
        device_kind = DEVICE_CLASSES.kind(record.key[0])
        record.parsed = parse_device_config(record.raw, device_kind=device_kind)
    except (DeviceConfigError, KeyError, TypeError, ValueError) as exc:
        if not record.parse_error_recorded:
            errors.append(f"devices[{record.index}] ({record.label}): {exc}")
            record.parse_error_recorded = True
    return record.parsed


def _target_record(
    key: DeviceKey,
    *,
    context: str,
    records: Mapping[DeviceKey, _DeviceRecord],
    disabled_keys: set[DeviceKey],
    errors: list[str],
) -> _DeviceRecord | None:
    """Look up one endpoint owner and distinguish disabled from missing."""
    target = records.get(key)
    if target is not None:
        return target
    label = _format_device_key(key)
    if key in disabled_keys:
        errors.append(f"{context} refers to disabled device {label}")
    else:
        errors.append(
            f"{context} must refer to an enabled configured device; {label} is missing"
        )
    return None


def _same_payload(
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    actual_shape: tuple[int, ...],
    actual_dtype: str,
) -> bool:
    return expected_shape == actual_shape and expected_dtype == actual_dtype


def _payload_error(
    *,
    context: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    actual_shape: tuple[int, ...],
    actual_dtype: str,
    stream: str,
    producer: _DeviceRecord,
    errors: list[str],
) -> None:
    errors.append(
        f"{context} {stream} payload from {producer.label} must match "
        f"shape={expected_shape}, dtype={expected_dtype}; got "
        f"shape={actual_shape}, dtype={actual_dtype}"
    )


def _validate_source_endpoint(
    endpoint: ShmEndpointConfig,
    *,
    owner: _DeviceRecord,
    consumer: ParsedDeviceConfig,
    context: str,
    records: Mapping[DeviceKey, _DeviceRecord],
    disabled_keys: set[DeviceKey],
    errors: list[str],
) -> None:
    """Check one robot command source against the consumer control stream."""
    target_key = _resolve_endpoint_key(endpoint, owner)
    target = _target_record(
        target_key,
        context=context,
        records=records,
        disabled_keys=disabled_keys,
        errors=errors,
    )
    if target is None:
        return
    target_contract = _parse_record(target, errors=errors)
    if target_contract is None:
        return

    expected_shape = consumer.control.format.shape
    expected_dtype = consumer.control.format.dtype
    assert expected_dtype is not None

    if endpoint.port == "state":
        if not target_contract.device_kind.readable:
            errors.append(f"{context} state endpoint {target.label} is not readable")
            return
        source_format = target_contract.state_feedback.format
        source_dtype = source_format.dtype
        assert source_dtype is not None
        if not _same_payload(
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            actual_shape=source_format.shape,
            actual_dtype=source_dtype,
        ):
            _payload_error(
                context=context,
                expected_shape=expected_shape,
                expected_dtype=expected_dtype,
                actual_shape=source_format.shape,
                actual_dtype=source_dtype,
                stream="state",
                producer=target,
                errors=errors,
            )
        return

    if endpoint.port == "command":
        if not target_contract.device_kind.writable or target_contract.control is None:
            errors.append(
                f"{context} command endpoint {target.label} must refer to a "
                "writable device with control"
            )
            return
        source_format = target_contract.control.format
        source_dtype = source_format.dtype
        assert source_dtype is not None
        if not _same_payload(
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            actual_shape=source_format.shape,
            actual_dtype=source_dtype,
        ):
            _payload_error(
                context=context,
                expected_shape=expected_shape,
                expected_dtype=expected_dtype,
                actual_shape=source_format.shape,
                actual_dtype=source_dtype,
                stream="command",
                producer=target,
                errors=errors,
            )
        return

    errors.append(f"{context} has unsupported source port {endpoint.port!r}")


def _produces_control_flag(
    record: _DeviceRecord,
    *,
    errors: list[str],
) -> bool:
    """Return known selector-producer capability without importing hardware code."""
    device_class = record.key[0]
    if device_class == "KeyboardTeleoperator":
        return True
    if device_class != "I2RTRobot":
        return False

    # I2RT publishes its own flag only from a leader configured for dual-source
    # control with the selector endpoint owned by that same instance.
    mode = str(record.raw.get("mode", "")).strip().casefold()
    if "leader" not in mode:
        return False
    parsed = _parse_record(record, errors=errors)
    if parsed is None or parsed.control is None:
        return False
    selection = parsed.control.source_selection
    if selection.mode != "dual_source" or selection.dual_source_selector is None:
        return False
    return (
        _resolve_endpoint_key(selection.dual_source_selector.endpoint, record)
        == record.key
    )


def _validate_selector_endpoint(
    endpoint: ShmEndpointConfig,
    *,
    owner: _DeviceRecord,
    context: str,
    records: Mapping[DeviceKey, _DeviceRecord],
    disabled_keys: set[DeviceKey],
    errors: list[str],
) -> None:
    """Check that the fixed binary selector has a configured producer."""
    target_key = _resolve_endpoint_key(endpoint, owner)
    target = _target_record(
        target_key,
        context=context,
        records=records,
        disabled_keys=disabled_keys,
        errors=errors,
    )
    if target is None:
        return
    if not _produces_control_flag(target, errors=errors):
        errors.append(
            f"{context} endpoint {target.label} does not produce control_flag"
        )


def _validate_source_selection(
    record: _DeviceRecord,
    *,
    consumer: ParsedDeviceConfig,
    records: Mapping[DeviceKey, _DeviceRecord],
    disabled_keys: set[DeviceKey],
    errors: list[str],
) -> None:
    """Validate all source and selector endpoints for one writable device."""
    control = consumer.control
    if control is None:
        return
    selection: SourceSelectionConfig = control.source_selection
    base = f"devices[{record.index}].control.source_selection"
    if selection.mode == "single_source":
        assert selection.source is not None
        _validate_source_endpoint(
            selection.source,
            owner=record,
            consumer=consumer,
            context=f"{base}.source",
            records=records,
            disabled_keys=disabled_keys,
            errors=errors,
        )
        return

    assert selection.sources is not None
    assert selection.dual_source_selector is not None
    for role, endpoint in selection.sources.items():
        _validate_source_endpoint(
            endpoint,
            owner=record,
            consumer=consumer,
            context=f"{base}.sources.{role}",
            records=records,
            disabled_keys=disabled_keys,
            errors=errors,
        )
    _validate_selector_endpoint(
        selection.dual_source_selector.endpoint,
        owner=record,
        context=f"{base}.dual_source_selector",
        records=records,
        disabled_keys=disabled_keys,
        errors=errors,
    )


def _validate_control_target(
    target: ControlTargetConfig,
    *,
    context: str,
    records: Mapping[DeviceKey, _DeviceRecord],
    disabled_keys: set[DeviceKey],
    errors: list[str],
) -> None:
    """Check one policy/action command target against its robot control stream."""
    key = (target.endpoint.device, target.endpoint.device_class_id)
    assert key[1] is not None
    target_record = _target_record(
        (key[0], key[1]),
        context=context,
        records=records,
        disabled_keys=disabled_keys,
        errors=errors,
    )
    if target_record is None:
        return
    target_contract = _parse_record(target_record, errors=errors)
    if target_contract is None:
        return
    if not target_contract.device_kind.writable or target_contract.control is None:
        errors.append(
            f"{context} command endpoint {target_record.label} must refer to a "
            "writable device with control"
        )
        return
    control_format = target_contract.control.format
    control_dtype = control_format.dtype
    assert control_dtype is not None
    if not _same_payload(
        expected_shape=target.shape,
        expected_dtype=target.dtype,
        actual_shape=control_format.shape,
        actual_dtype=control_dtype,
    ):
        _payload_error(
            context=context,
            expected_shape=target.shape,
            expected_dtype=target.dtype,
            actual_shape=control_format.shape,
            actual_dtype=control_dtype,
            stream="command",
            producer=target_record,
            errors=errors,
        )


def _validate_control_target_list(
    value: Any,
    *,
    field: str,
    records: Mapping[DeviceKey, _DeviceRecord],
    disabled_keys: set[DeviceKey],
    errors: list[str],
) -> None:
    try:
        targets = parse_control_targets(value, field=field)
    except (TypeError, ValueError, SHMContractError) as exc:
        errors.append(str(exc))
        return
    for index, target in enumerate(targets):
        _validate_control_target(
            target,
            context=f"{field}[{index}]",
            records=records,
            disabled_keys=disabled_keys,
            errors=errors,
        )


def topology_validation_errors(
    config: Mapping[str, Any],
    *,
    validate_all_enabled_devices: bool = False,
) -> list[str]:
    """Return static endpoint-graph errors for a composed runtime config.

    The default validates only declarations that participate in structured
    topology.  This keeps read-only Control Panel inspection useful for its
    historical preview-only device records.  Process launchers and the full
    schema use ``validate_all_enabled_devices=True`` so every enabled device
    also satisfies the typed construction contract before startup.
    """
    if not isinstance(config, Mapping):
        return ["config must be a mapping"]
    devices = _device_list(config)
    if devices is None:
        return ["devices must be a sequence"]

    errors: list[str] = []
    records: dict[DeviceKey, _DeviceRecord] = {}
    disabled_keys: set[DeviceKey] = set()
    for index, raw_device in enumerate(devices):
        if not isinstance(raw_device, Mapping):
            if validate_all_enabled_devices:
                errors.append(f"devices[{index}] must be a mapping")
            continue
        key = _device_key_from_mapping(raw_device)
        if key is None:
            if validate_all_enabled_devices:
                errors.append(
                    f"devices[{index}] requires a non-empty class and non-negative device_class_id"
                )
            continue
        if not _device_enabled(raw_device):
            disabled_keys.add(key)
            continue
        if key in records:
            errors.append(f"duplicate enabled device key: {_format_device_key(key)}")
            continue
        records[key] = _DeviceRecord(index=index, key=key, raw=raw_device)

    for record in records.values():
        raw_control = record.raw.get("control")
        has_explicit_source_selection = (
            isinstance(raw_control, Mapping) and "source_selection" in raw_control
        )
        if not validate_all_enabled_devices and not has_explicit_source_selection:
            continue
        parsed = _parse_record(record, errors=errors)
        if parsed is not None and parsed.control is not None:
            _validate_source_selection(
                record,
                consumer=parsed,
                records=records,
                disabled_keys=disabled_keys,
                errors=errors,
            )

    policy = config.get("policy")
    if isinstance(policy, Mapping) and "controls" in policy:
        _validate_control_target_list(
            policy.get("controls"),
            field="policy.controls",
            records=records,
            disabled_keys=disabled_keys,
            errors=errors,
        )

    action_executor = config.get("action_executor")
    if isinstance(action_executor, Mapping) and "robot_control_shms" in action_executor:
        _validate_control_target_list(
            action_executor.get("robot_control_shms"),
            field="action_executor.robot_control_shms",
            records=records,
            disabled_keys=disabled_keys,
            errors=errors,
        )
    return errors


def validate_config_topology(
    config: Mapping[str, Any],
    *,
    validate_all_enabled_devices: bool = False,
) -> None:
    """Raise one error when the enabled structured endpoint graph is invalid."""
    errors = topology_validation_errors(
        config,
        validate_all_enabled_devices=validate_all_enabled_devices,
    )
    if errors:
        raise ConfigTopologyError("; ".join(errors))


__all__ = [
    "ConfigTopologyError",
    "topology_validation_errors",
    "validate_config_topology",
]
