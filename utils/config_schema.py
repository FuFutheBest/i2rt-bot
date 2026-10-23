#!/usr/bin/env python3
"""Lightweight YAML configuration schema checks.

Author: OpenAI Assistant
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Container, Iterable, Mapping, Optional, Sequence

import yaml

from devices import DEVICE_CLASSES
from devices.robots.config import SourceSelectionConfig
from utils.contracts import normalize_shape, validate_dtype_name
from utils.device_config import DeviceConfigError, parse_device_config
from utils.errors import ConfigError, SHMContractError
from utils.logger_config import logger
from utils.policy_control_config import ControlTargetConfig, parse_control_targets
from utils.config_topology import ConfigTopologyError, validate_config_topology
from utils.synchronization import SynchronizationModeError, normalize_sync_mode


@dataclass(frozen=True)
class ConfigSchemaIssue:
    """Schema validation issue for a config file."""

    path: Path
    message: str

    def format(self, root: Path) -> str:
        """Format an issue relative to the repository root."""
        return f"{self.path.relative_to(root)}: {self.message}"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary, returning an empty dict for empty files."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def validate_top_level_config(path: Path, data: dict[str, Any]) -> list[ConfigSchemaIssue]:
    """Validate shared structure expected from top-level runtime configs."""
    issues: list[ConfigSchemaIssue] = []

    defaults = data.get("defaults")
    if defaults is not None and not isinstance(defaults, list):
        issues.append(ConfigSchemaIssue(path, "defaults must be a list when present"))

    if path.name == "config.yaml":
        required_groups = {"devices", "device_manager", "action_executor", "consumer", "runtime"}
        configured_groups = set()
        for item in defaults or []:
            if not isinstance(item, dict) or not item:
                continue
            group_name = str(next(iter(item))).lstrip("/").split("@", maxsplit=1)[0]
            configured_groups.add(
                "devices" if group_name == "devices/infer" else group_name
            )
        missing_groups = sorted(required_groups - configured_groups)
        if missing_groups:
            issues.append(ConfigSchemaIssue(path, f"missing default groups: {missing_groups}"))

    policy = data.get("policy")
    if isinstance(policy, dict):
        policy_class = policy.get("class")
        if "class" in policy and (not isinstance(policy_class, str) or not policy_class):
            issues.append(ConfigSchemaIssue(path, "policy.class must be a non-empty string"))
        params = policy.get("params")
        if params is not None and not isinstance(params, dict):
            issues.append(ConfigSchemaIssue(path, "policy.params must be a mapping when present"))

    hydra = data.get("hydra")
    if hydra is not None and not isinstance(hydra, dict):
        issues.append(ConfigSchemaIssue(path, "hydra must be a mapping when present"))

    return issues


def validate_top_level_configs(paths: Iterable[Path]) -> list[ConfigSchemaIssue]:
    """Validate all top-level runtime config files."""
    issues: list[ConfigSchemaIssue] = []
    for path in sorted(paths):
        issues.extend(validate_top_level_config(path, load_yaml_config(path)))
    return issues


REQUIRED_RUNTIME_SIGNAL_FILES = (
    "start_recording",
    "stop_recording",
    "reset_signal",
    "robot_home_index",
    "episode_result",
    "policy_runner_reset_command",
    "teleoperator_reset_command",
    "rollout_ready",
    "robot_reconnect",
    "robot_halt_command_pattern",
    "robot_halt_reply_pattern",
)

_USB_PORT_REQUIREMENTS = {"any", "usb_2.0", "usb_3.0"}


def _fail(message: str, *, context: Optional[str] = None) -> None:
    """Log and raise a configuration error."""
    full_message = f"{context}: {message}" if context else message
    logger.error(full_message)
    raise ConfigError(message, context=context)


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    """Return value as a mapping or raise ConfigError."""
    if not isinstance(value, Mapping):
        _fail("expected a mapping", context=context)
    return value


def _require_sequence(value: Any, *, context: str) -> Sequence[Any]:
    """Return value as a sequence or raise ConfigError."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("expected a sequence", context=context)
    return value


def _validate_registry_value(
    value: Optional[str],
    registry: Optional[Container[str]],
    *,
    context: str,
    allow_none: bool = False,
) -> None:
    """Validate a config string against an optional registry."""
    if value is None:
        if allow_none:
            return
        _fail("required value is missing", context=context)
    if registry is not None and value not in registry:
        _fail(f"unknown registry value '{value}'", context=context)


def _normalize_shape_config(shape: Sequence[Any], *, context: str) -> tuple[int, ...]:
    """Normalize a config shape and raise ConfigError on contract violations."""
    try:
        return normalize_shape(shape)
    except SHMContractError as exc:
        _fail(str(exc), context=context)


def _validate_dtype_config(dtype: str, *, context: str) -> str:
    """Validate a config dtype and raise ConfigError on contract violations."""
    try:
        return validate_dtype_name(dtype)
    except SHMContractError as exc:
        _fail(str(exc), context=context)


def validate_action_target_config(control: Mapping[str, Any], *, context: str = "policy.controls[]") -> None:
    """Validate one structured policy command target."""
    try:
        ControlTargetConfig.from_mapping(control, field=context)
    except (TypeError, ValueError, SHMContractError) as exc:
        _fail(str(exc), context=context)


def validate_control_source_selection_config(
    control: Mapping[str, Any],
    *,
    context: str = "devices[].control",
) -> None:
    """Validate the structured source topology of one robot control block.

    ``RobotControlConfig`` performs the same validation while constructing a
    device.  Keeping this small schema-level entry point lets static config
    checks report source-selection errors before any device class is imported.
    """
    if not isinstance(control, Mapping):
        _fail("control must be a mapping", context=context)
    try:
        SourceSelectionConfig.from_mapping(control.get("source_selection"))
    except (TypeError, ValueError) as exc:
        _fail(str(exc), context=f"{context}.source_selection")


def validate_device_config(
    device: Mapping[str, Any],
    *,
    device_registry: Optional[Container[str]] = None,
    context: str = "devices[]",
) -> None:
    """Validate one device config without importing hardware modules."""
    device_class = device.get("class")
    _validate_registry_value(
        str(device_class) if device_class is not None else None,
        device_registry,
        context=f"{context}.class",
    )
    if "device_class_id" not in device:
        _fail("device requires 'device_class_id'", context=context)
    try:
        device_class_id = int(device["device_class_id"])
    except (TypeError, ValueError):
        _fail("device_class_id must be an integer", context=context)
    if device_class_id < 0:
        _fail("device_class_id must be non-negative", context=context)

    try:
        parse_device_config(
            device,
            device_kind=DEVICE_CLASSES.kind(str(device_class)),
        )
    except (DeviceConfigError, KeyError, ValueError) as exc:
        _fail(str(exc), context=context)

    raw_control = device.get("control")
    if isinstance(raw_control, Mapping):
        validate_control_source_selection_config(
            raw_control,
            context=f"{context}.control",
        )

    if "usb_port_requirement" in device:
        requirement = device["usb_port_requirement"]
        if (
            not isinstance(requirement, str)
            or requirement.strip().casefold() not in _USB_PORT_REQUIREMENTS
        ):
            _fail(
                "usb_port_requirement must be one of: any, usb_2.0, usb_3.0",
                context=context,
            )


def validate_policy_config(
    policy: Mapping[str, Any],
    *,
    policy_registry: Optional[Container[str]] = None,
    context: str = "policy",
) -> None:
    """Validate the policy block used by inference consumers."""
    policy_class = policy.get("class")
    _validate_registry_value(
        str(policy_class) if policy_class is not None else None,
        policy_registry,
        context=f"{context}.class",
    )
    if "chunk_length" in policy and int(policy["chunk_length"]) <= 0:
        _fail("chunk_length must be positive", context=context)
    if "fps" in policy and float(policy["fps"]) <= 0.0:
        _fail("fps must be positive", context=context)

    controls = policy.get("controls", [])
    controls_sequence = _require_sequence(controls, context=f"{context}.controls")
    if not controls_sequence:
        _fail("policy.controls must not be empty", context=context)
    for index, control in enumerate(controls_sequence):
        validate_action_target_config(
            _require_mapping(control, context=f"{context}.controls[{index}]"),
            context=f"{context}.controls[{index}]",
        )

    obs = policy.get("obs", {})
    if obs:
        obs_mapping = _require_mapping(obs, context=f"{context}.obs")
        devices = obs_mapping.get("devices", [])
        _require_sequence(devices, context=f"{context}.obs.devices")


def validate_runtime_config(runtime: Mapping[str, Any], *, context: str = "runtime") -> None:
    """Validate runtime coordination fields required by policy and action processes."""
    signal_files = runtime.get("signal_files")
    if signal_files is None:
        _fail("runtime requires 'signal_files'", context=context)
    signal_mapping = _require_mapping(signal_files, context=f"{context}.signal_files")
    missing = [key for key in REQUIRED_RUNTIME_SIGNAL_FILES if not signal_mapping.get(key)]
    if missing:
        _fail(f"missing signal_files keys: {missing}", context=context)


def validate_device_manager_config(
    device_manager: Mapping[str, Any],
    *,
    manager_registry: Optional[Container[str]] = None,
    context: str = "device_manager",
) -> None:
    """Validate the device manager config block."""
    manager_type = device_manager.get("type")
    _validate_registry_value(
        str(manager_type) if manager_type is not None else None,
        manager_registry,
        context=f"{context}.type",
    )
    params = device_manager.get("params", {})
    params_mapping = _require_mapping(params, context=f"{context}.params")
    if "buffer_size" in params_mapping and int(params_mapping["buffer_size"]) <= 0:
        _fail("buffer_size must be positive", context=f"{context}.params")
    if manager_type == "synchronized":
        try:
            normalize_sync_mode(device_manager.get("sync_mode", "binary_match"))
        except SynchronizationModeError as exc:
            _fail(str(exc), context=f"{context}.sync_mode")
        params_master_device_class_id = params_mapping.get("master_device_class_id")
        top_level_master_device_class_id = device_manager.get("master_device_class_id")
        if params_master_device_class_id is None and top_level_master_device_class_id is None:
            _fail(
                "synchronized manager requires master_device_class_id or params.master_device_class_id",
                context=context,
            )
        try:
            params_master_index = (
                None if params_master_device_class_id is None else int(params_master_device_class_id)
            )
            top_level_master_index = (
                None if top_level_master_device_class_id is None else int(top_level_master_device_class_id)
            )
        except (TypeError, ValueError):
            _fail("master_device_class_id must be an integer", context=context)
        if (
            params_master_index is not None
            and top_level_master_index is not None
            and params_master_index != top_level_master_index
        ):
            _fail(
                "master_device_class_id conflicts with params.master_device_class_id",
                context=context,
            )


def validate_action_executor_config(
    action_executor: Mapping[str, Any],
    *,
    manager_registry: Optional[Container[str]] = None,
    chunk_manager_registry: Optional[Container[str]] = None,
    context: str = "action_executor",
) -> None:
    """Validate action executor and chunk manager references."""
    executor_type = action_executor.get("type")
    _validate_registry_value(
        str(executor_type) if executor_type is not None else None,
        manager_registry,
        context=f"{context}.type",
        allow_none=True,
    )
    if "sim_clock_shm" in action_executor:
        _fail(
            "sim_clock_shm was removed; execution_clock='sim_time' derives the "
            "bridge clock from the first robot_control_shms command endpoint",
            context=f"{context}.sim_clock_shm",
        )
    if "robot_control_shms" in action_executor:
        try:
            parse_control_targets(
                action_executor["robot_control_shms"],
                field=f"{context}.robot_control_shms",
            )
        except (TypeError, ValueError, SHMContractError) as exc:
            _fail(str(exc), context=f"{context}.robot_control_shms")
    chunk_manager = action_executor.get("chunk_manager")
    if chunk_manager is None:
        return
    chunk_mapping = _require_mapping(chunk_manager, context=f"{context}.chunk_manager")
    manager_type = chunk_mapping.get("type")
    _validate_registry_value(
        str(manager_type) if manager_type is not None else None,
        chunk_manager_registry,
        context=f"{context}.chunk_manager.type",
        allow_none=True,
    )


def _iter_device_configs(config: Mapping[str, Any]) -> Sequence[Any]:
    """Return device configs from either Hydra-style nested or legacy flat shape."""
    devices_section = config.get("devices", [])
    if isinstance(devices_section, Mapping):
        devices = devices_section.get("devices", [])
    else:
        devices = devices_section
    return _require_sequence(devices, context="devices")


def validate_superinference_config(
    config: Mapping[str, Any],
    *,
    device_registry: Optional[Container[str]] = None,
    policy_registry: Optional[Container[str]] = None,
    manager_registry: Optional[Container[str]] = None,
    chunk_manager_registry: Optional[Container[str]] = None,
) -> None:
    """Validate a composed SuperInference config without starting processes."""
    config_mapping = _require_mapping(config, context="config")

    from utils.device_identity import validate_device_class_ids

    try:
        validate_device_class_ids(_iter_device_configs(config_mapping))
    except ValueError as exc:
        _fail(str(exc), context="devices")
    for index, device in enumerate(_iter_device_configs(config_mapping)):
        validate_device_config(
            _require_mapping(device, context=f"devices[{index}]"),
            device_registry=device_registry,
            context=f"devices[{index}]",
        )

    try:
        validate_config_topology(
            config_mapping,
            validate_all_enabled_devices=True,
        )
    except ConfigTopologyError as exc:
        _fail(str(exc), context="device topology")

    if "device_manager" in config_mapping:
        validate_device_manager_config(
            _require_mapping(config_mapping["device_manager"], context="device_manager"),
            manager_registry=manager_registry,
        )
    if "action_executor" in config_mapping:
        validate_action_executor_config(
            _require_mapping(config_mapping["action_executor"], context="action_executor"),
            manager_registry=manager_registry,
            chunk_manager_registry=chunk_manager_registry,
        )
    if "policy" in config_mapping:
        validate_policy_config(
            _require_mapping(config_mapping["policy"], context="policy"),
            policy_registry=policy_registry,
        )
    if "runtime" in config_mapping:
        validate_runtime_config(_require_mapping(config_mapping["runtime"], context="runtime"))
