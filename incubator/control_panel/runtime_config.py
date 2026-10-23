#!/usr/bin/env python3
"""Hydra configuration discovery and composition for runtime control tools.

The command-line device starter and the web control server must resolve a
configuration in exactly the same way.  Keeping that logic here avoids subtle
differences in interpolation and config-group normalization.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from utils.hydra_config import compose_hydra_application_config, normalize_hydra_config
from utils.inference_options import as_bool


from .recording import (
    normalize_recording_camera_mapping,
    validate_recording_camera_mapping,
)
from .gripper_contract import rizon_gripper_enabled_sides

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"

# ``inspect_runtime_config`` is exposed by the web API.  Keep its response
# deliberately narrow: runtime config files commonly contain credentials,
# host addresses, calibration paths, and other values that are only needed by
# the process that launches a device.  These patterns are used only to accept
# simple display values; arbitrary nested config values never cross the API.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_DTYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_SAFE_SHM_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_TOOL_ASSET_PATH_RE = re.compile(r"^[A-Za-z0-9_./@+-]{1,512}$")
_POLICY_INFER_CONFIG_NAMESPACES = ("openpi", "fastwam", "lingbotva")
_TOOL_ELEMENT_TYPES = frozenset(
    {
        "tool",
        "robot_tool",
        "end_tool",
        "end_effector",
        "end_effector_tool",
        "eef",
        "eef_tool",
        "tcp_tool",
    }
)
_TOOL_VISUAL_MODELS = {
    "umi3": {
        "kind": "urdf",
        "url": "/static/assets/tools/umi3_camera/soft/umi3_gripper.urdf",
        "default_variant": "soft",
        "variants": {
            "soft": {
                "kind": "urdf",
                "url": "/static/assets/tools/umi3_camera/soft/umi3_gripper.urdf",
            },
            "hard": {
                "kind": "urdf",
                "url": "/static/assets/tools/umi3_camera/hard/umi3_gripper.urdf",
            },
        },
        "joint_name": "gripper_joint",
        "joint_lower_rad": 0.0,
        "joint_upper_rad": math.pi / 2.0,
        "closed_joint_rad": 0.0,
    }
}
_YAM_VISUAL_MODELS = {
    "yam": {
        "url": "/static/assets/yam/yam_no_gripper.urdf",
        "joint_count": 6,
    }
}
_YAM_BASE_POSITION_WORLD = (0.0, 0.0, 0.2)
_UMI3_SOFT_GRIPPER_ARTICULATION = {
    "arm_length_m": 0.06543,
    "max_width_m": 0.0925319933860716,
    "apex_angle_max_deg": 90.0,
}
_TOOL_VISUAL_CONFIG_KEYS = frozenset({"model", "articulation", "articulations"})
_TOOL_VISUAL_ARTICULATION_KEYS = frozenset(
    {"value_mode", "arm_length_m", "max_width_m", "apex_angle_max_deg"}
)


class RuntimeConfigError(ValueError):
    """Raised when a requested runtime configuration cannot be composed."""


# The control panel owns device lifecycle management and the model launchers
# registered here. Keep this table explicit: metadata in a YAML file may
# describe a workflow, but cannot register an unimplemented launcher.
CONTROL_PANEL_SUPPORTED_WORKFLOW_KINDS = frozenset({"device", "openpi", "fastwam"})
CONTROL_PANEL_SUPPORTED_POLICY_PREFIXES = ("OpenPi",)
CONTROL_PANEL_SUPPORTED_POLICY_CLASSES = frozenset({"FastWAMPolicy"})
_SOFTWARE_DEVICE_CLASSES = frozenset(
    {
        "BaseDevice",
        "MockDevice",
        "SimulatedDevice",
        "ControlPanelNoMotionPolicy",
    }
)


def normalize_config_name(config_name: str) -> str:
    """Return a safe native Hydra launch-config name without a YAML suffix.

    Names are relative to ``configs/``. Root files use ``foo`` and namespaced
    launch files use their full path, for example
    ``policy_infer/openpi/foo`` or ``policy_infer/fastwam/foo``.
    Device/config-manager directories are Hydra config groups and are
    intentionally rejected as launch targets.
    """
    if not isinstance(config_name, str) or not config_name.strip():
        raise RuntimeConfigError("config name is required")

    name = config_name.strip()
    if name.endswith(".yaml"):
        name = name[:-5]
    elif name.endswith(".yml"):
        name = name[:-4]

    parts = Path(name).parts
    if (
        not name
        or name in {".", ".."}
        or any(part in {"", ".", ".."} for part in parts)
        or not _is_native_launch_config_path(parts)
        or any(_SAFE_IDENTIFIER_RE.fullmatch(part) is None for part in parts)
    ):
        raise RuntimeConfigError(
            "config name must be a launch config relative to configs/ "
            "(for example policy_infer/openpi/my_config)"
        )
    return name


def _is_native_launch_config_path(parts: tuple[str, ...]) -> bool:
    """Return whether path components describe a supported launch config."""
    if len(parts) == 1:
        return True
    if len(parts) == 2:
        return parts[0] == "control_panel"
    return (
        len(parts) == 3
        and parts[0] == "policy_infer"
        and parts[1] in _POLICY_INFER_CONFIG_NAMESPACES
    )


def list_runtime_configs(config_dir: Optional[Path] = None) -> List[str]:
    """List launch configs using the same names Hydra accepts natively."""
    root = (config_dir or DEFAULT_CONFIG_DIR).resolve()
    if not root.is_dir():
        raise RuntimeConfigError(f"config directory does not exist: {root}")

    names = {
        path.stem
        for pattern in ("*.yaml", "*.yml")
        for path in root.glob(pattern)
        if path.is_file()
    }
    control_panel_root = root / "control_panel"
    if control_panel_root.is_dir():
        names.update(
            path.relative_to(root).with_suffix("").as_posix()
            for pattern in ("*.yaml", "*.yml")
            for path in control_panel_root.glob(pattern)
            if path.is_file()
        )
    policy_infer_root = root / "policy_infer"
    for namespace in _POLICY_INFER_CONFIG_NAMESPACES:
        namespace_root = policy_infer_root / namespace
        if namespace_root.is_dir():
            names.update(
                path.relative_to(root).with_suffix("").as_posix()
                for pattern in ("*.yaml", "*.yml")
                for path in namespace_root.glob(pattern)
                if path.is_file()
            )
    return sorted(names, key=str.casefold)


def normalize_runtime_config(config: Any) -> Dict[str, Any]:
    """Return the same normalized application mapping used by device_starter."""
    try:
        return normalize_hydra_config(config)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(str(exc)) from exc


def validate_runtime_config(config: Mapping[str, Any]) -> List[str]:
    """Return human-readable validation errors without touching hardware."""
    from devices.robots.config import RobotControlConfig
    from utils.device_config import DeviceConfigError, reject_removed_device_fields
    from utils.config_topology import topology_validation_errors
    from utils.policy_control_config import parse_control_targets
    errors: List[str] = []
    devices = config.get("devices", [])
    if not isinstance(devices, list):
        return ["devices must resolve to a list"]

    seen: set[Tuple[str, str]] = set()
    for index, device in enumerate(devices):
        prefix = f"devices[{index}]"
        if not isinstance(device, Mapping):
            errors.append(f"{prefix} must be a mapping")
            continue
        if "class" not in device:
            errors.append(f"{prefix}.class is required")
        if "device_class_id" not in device:
            errors.append(f"{prefix}.device_class_id is required")
        if "class" not in device or "device_class_id" not in device:
            continue
        key = (str(device["class"]), str(device["device_class_id"]))
        if key in seen:
            errors.append(f"duplicate device key: {key[0]}:{key[1]}")
        seen.add(key)

        raw_control = device.get("control")
        try:
            reject_removed_device_fields(device, field=prefix)
        except DeviceConfigError as exc:
            errors.append(str(exc))
        if raw_control is not None:
            try:
                RobotControlConfig.require(raw_control)
            except (TypeError, ValueError) as exc:
                errors.append(f"{prefix}.control: {exc}")

    policy = config.get("policy")
    if isinstance(policy, Mapping) and "controls" in policy:
        try:
            parse_control_targets(policy.get("controls"), field="policy.controls")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    action_executor = config.get("action_executor")
    if isinstance(action_executor, Mapping):
        if "sim_clock_shm" in action_executor:
            errors.append(
                "action_executor.sim_clock_shm was removed; "
                "execution_clock='sim_time' derives it from the command endpoint"
            )
        if "robot_control_shms" in action_executor:
            try:
                parse_control_targets(
                    action_executor.get("robot_control_shms"),
                    field="action_executor.robot_control_shms",
                )
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
    errors.extend(topology_validation_errors(config))
    errors.extend(validate_recording_camera_mapping(config))
    return errors


def _metadata_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = config.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _configured_policy_class(config: Mapping[str, Any]) -> Optional[str]:
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        return None
    value = policy.get("class")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _policy_class(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> Optional[str]:
    explicit = metadata.get("policy_class")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return _configured_policy_class(config)


def _workflow_kind(policy_class: Optional[str], metadata: Mapping[str, Any]) -> str:
    explicit = metadata.get("workflow_kind")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if policy_class is None:
        return "device"
    normalized = policy_class.casefold()
    if normalized == "controlpanelnomotionpolicy":
        return "device"
    if normalized.startswith("openpi"):
        return "openpi"
    if normalized == "fastwampolicy":
        return "fastwam"
    if normalized.startswith("diffusion"):
        return "diffusion"
    if normalized == "lingbotvapolicy":
        return "lingbotva"
    if normalized == "nmxvlapolicy":
        return "nmx_vla"
    return "unknown"


def _hardware_required(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    explicit = metadata.get("hardware_required")
    if isinstance(explicit, bool):
        return explicit
    policy = config.get("policy")
    policy_class = metadata.get("policy_class")
    if not policy_class and isinstance(policy, Mapping):
        policy_class = policy.get("class")
    if policy_class and str(policy_class) not in _SOFTWARE_DEVICE_CLASSES:
        return True
    devices = config.get("devices", [])
    if not isinstance(devices, list):
        return True
    for device in devices:
        if not isinstance(device, Mapping) or not _device_is_enabled(device):
            continue
        device_class = device.get("class")
        if str(device_class) not in _SOFTWARE_DEVICE_CLASSES:
            return True
    return False


def runtime_config_metadata(
    config_name: str,
    config: Optional[Mapping[str, Any]] = None,
    *,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return stable, safe metadata used by config pickers and diagnostics.

    Existing YAML files do not need a mass migration: fields are derived from
    the composed config and can be overridden by a small ``metadata`` mapping.
    """
    normalize_config_name(config_name)
    normalized_errors = list(errors or [])
    metadata = _metadata_mapping(config or {})
    configured_policy_class = _configured_policy_class(config or {})
    policy_class = _policy_class(config or {}, metadata)
    workflow_kind = _workflow_kind(policy_class, metadata)
    configured_workflow_kind = _workflow_kind(configured_policy_class, {})
    hardware_required = _hardware_required(config or {}, metadata)
    status_value = metadata.get("status")
    status = status_value.strip() if isinstance(status_value, str) else ""
    if not status:
        status = "invalid" if normalized_errors else "available"
    # Metadata may disable an entry, but it cannot register a new workflow or
    # relabel a composed policy. Registration belongs in this code-level
    # capability table so a future or custom policy cannot become selectable
    # by editing YAML alone.
    metadata_matches_composed_policy = (
        (
            configured_policy_class is None
            and policy_class is None
            and workflow_kind == configured_workflow_kind == "device"
        )
        or (
            configured_policy_class is not None
            and policy_class == configured_policy_class
            and workflow_kind == configured_workflow_kind
        )
    )
    registry_supported = (
        metadata_matches_composed_policy
        and configured_workflow_kind in CONTROL_PANEL_SUPPORTED_WORKFLOW_KINDS
        and (
            configured_workflow_kind == "device"
            or bool(
                configured_policy_class
                and (
                    configured_policy_class.startswith(
                        CONTROL_PANEL_SUPPORTED_POLICY_PREFIXES
                    )
                    or configured_policy_class
                    in CONTROL_PANEL_SUPPORTED_POLICY_CLASSES
                )
            )
        )
    )
    supported_override = metadata.get("control_panel_supported")
    if isinstance(supported_override, bool):
        control_panel_supported = bool(supported_override and registry_supported)
    else:
        control_panel_supported = registry_supported

    support_reason = metadata.get("support_reason")
    if not isinstance(support_reason, str) or not support_reason.strip():
        if control_panel_supported:
            support_reason = "supported by control panel"
        elif not metadata_matches_composed_policy:
            support_reason = (
                "metadata does not match the composed policy/workflow: "
                f"{workflow_kind}/{policy_class} vs "
                f"{configured_workflow_kind}/{configured_policy_class}"
            )
        elif policy_class:
            support_reason = f"workflow/policy is not supported: {workflow_kind}/{policy_class}"
        else:
            support_reason = f"workflow is not supported: {workflow_kind}"

    result: Dict[str, Any] = {
        "workflow_kind": workflow_kind,
        "policy_class": policy_class,
        "hardware_required": hardware_required,
        "status": status,
        "control_panel_supported": control_panel_supported,
        "support_reason": support_reason,
    }
    for key in ("label", "description", "supported_by"):
        value = metadata.get(key)
        if isinstance(value, (str, list, tuple)):
            result[key] = list(value) if isinstance(value, tuple) else value
    return result


def compose_runtime_config(
    config_name: str, config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Compose and normalize a native Hydra launch configuration."""
    name = normalize_config_name(config_name)
    root = (config_dir or DEFAULT_CONFIG_DIR).resolve()
    source = root / f"{name}.yaml"
    alternate_source = root / f"{name}.yml"
    compose_name = name
    if not source.is_file() and not alternate_source.is_file():
        if len(Path(name).parts) == 1:
            policy_infer_root = root / "policy_infer"
            for namespace in _POLICY_INFER_CONFIG_NAMESPACES:
                namespace_root = policy_infer_root / namespace
                namespaced_source = namespace_root / f"{name}.yaml"
                namespaced_alternate = namespace_root / f"{name}.yml"
                if namespaced_source.is_file() or namespaced_alternate.is_file():
                    compose_name = f"policy_infer/{namespace}/{name}"
                    break
            else:
                raise RuntimeConfigError(f"unknown config: {name}")
        else:
            raise RuntimeConfigError(f"unknown config: {name}")

    try:
        return compose_hydra_application_config(
            compose_name,
            config_dir=root,
            job_name="device_control",
        )
    except RuntimeConfigError:
        raise
    except Exception as exc:
        raise RuntimeConfigError(f"failed to compose {name}: {exc}") from exc


def _device_is_enabled(device: Mapping[str, Any]) -> bool:
    """Interpret the small set of YAML boolean spellings used by device configs."""
    value = device.get("enabled", True)
    if isinstance(value, (list, tuple)):
        return any(as_bool(item) for item in value)
    return as_bool(value)


def _safe_identifier(value: Any) -> Any:
    """Return a JSON-safe identifier without exposing arbitrary config text."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and _SAFE_IDENTIFIER_RE.fullmatch(value):
        return value
    return None


def _safe_shape(value: Any) -> Optional[List[int]]:
    """Keep only finite, non-negative integer dimensions for the UI."""
    if not isinstance(value, (list, tuple)) or not value or len(value) > 8:
        return None
    shape: List[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            return None
        if dimension < 0 or dimension > 1_000_000:
            return None
        shape.append(dimension)
    return shape


def _safe_number(value: Any) -> Optional[float | int]:
    """Return a finite numeric display value, preserving integer FPS values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _side_value(value: Any, index: int, default: Any = None) -> Any:
    """Read a scalar or side-indexed config value without expanding it."""
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else default
    return default if value is None else value


def _safe_motor_id(value: Any) -> Optional[int]:
    """Return a bounded integer motor ID without accepting boolean aliases."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    motor_id = int(numeric)
    return motor_id if 0 <= motor_id <= 255 else None


def extract_gripper_slots(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return the configured gripper identity contract for preflight.

    This remains config-only. Calibration contents and serial reachability are
    supplied by GripperControlManager.state; the panel can therefore show
    the expected port/motor/file alongside the observed artifact without
    exposing the full device config.
    """
    slots: Dict[str, Dict[str, Any]] = {
        "left": {"side": "left", "configured": False},
        "right": {"side": "right", "configured": False},
    }
    raw_devices = config.get("devices")
    if isinstance(raw_devices, Mapping):
        raw_devices = raw_devices.get("devices")
    if not isinstance(raw_devices, list):
        return slots
    for device in raw_devices:
        if not isinstance(device, Mapping) or not _device_is_enabled(device):
            continue
        if str(device.get("class") or "").casefold() != "rizonrobot":
            continue
        gripper = device.get("gripper")
        if not isinstance(gripper, Mapping):
            flat_backend = str(device.get("gripper_backend") or "").strip().casefold()
            if flat_backend not in {"lktech", ""}:
                continue
            if flat_backend == "" and "gripper_server" not in device:
                continue
            gripper = {
                "backend": "lktech",
                "enabled": device.get("utilize_gripper", False),
                "port": device.get("gripper_port"),
                "motor_id": device.get("gripper_motor_id"),
                "calibration_files": device.get("gripper_calibration_files"),
            }
        backend = str(gripper.get("backend", "lktech")).strip().casefold()
        if backend != "lktech":
            continue
        enabled_sides = rizon_gripper_enabled_sides(device, gripper)
        for index, side in enumerate(("left", "right")[:len(enabled_sides)]):
            if not enabled_sides[index]:
                continue
            raw_file = _side_value(gripper.get("calibration_files"), index)
            file_name = None
            if isinstance(raw_file, str) and raw_file.strip():
                file_name = Path(raw_file).name
            raw_port = _side_value(gripper.get("port"), index)
            port = raw_port.strip() if isinstance(raw_port, str) else None
            motor_id = _safe_motor_id(
                _side_value(gripper.get("motor_id"), index)
            )
            slots[side] = {
                "side": side,
                "configured": True,
                "port": port,
                "motor_id": motor_id,
                "motor_id_valid": motor_id is not None,
                "calibration_file": file_name,
                "backend": backend,
            }
    return slots


def _safe_integer(value: Any, *, minimum: int = 0, maximum: int = 1_000_000) -> Optional[int]:
    """Return a bounded integer for an execution-profile display field."""
    number = _safe_number(value)
    if number is None or int(number) != number:
        return None
    integer = int(number)
    if integer < minimum or integer > maximum:
        return None
    return integer


def _safe_delay_value(value: Any) -> Any:
    """Keep only bounded delay scalars while hiding arbitrary config values."""
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _safe_identifier(key)
            safe_value = _safe_number(item)
            if isinstance(safe_key, str) and safe_value is not None:
                result[safe_key] = safe_value
            if len(result) >= 32:
                break
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value[:32]:
            result.append(_safe_number(item))
        return result
    return None


def _profile_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Unwrap a Hydra config-group section without exposing its raw contents."""
    section = config.get(name)
    if not isinstance(section, Mapping):
        return {}
    if "type" not in section and isinstance(section.get(name), Mapping):
        nested = section.get(name)
        return nested if isinstance(nested, Mapping) else {}
    return section


def _safe_chunk_manager(value: Any) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Return an allowlisted ActionManager type and tuning values."""
    if not isinstance(value, Mapping):
        return _safe_identifier(value), {}
    manager_type = _safe_identifier(value.get("type"))
    options: Dict[str, Any] = {}
    numeric_fields = {
        "search_tolerance": (0, 4096),
        "pos_weight": (0.0, 100.0),
        "rot_weight": (0.0, 100.0),
        "ee_weight": (0.0, 100.0),
        "discount_factor": (0.0, 1.0),
        "direction_consistency_range": (0, 4096),
        "skip_first_n_steps": (0, 4096),
    }
    for key, (minimum, maximum) in numeric_fields.items():
        if key not in value:
            continue
        raw = _safe_number(value.get(key))
        if raw is None or float(raw) < minimum or float(raw) > maximum:
            continue
        options[key] = int(raw) if isinstance(minimum, int) and int(raw) == raw else raw
    drop_first = value.get("drop_first_actions")
    if isinstance(drop_first, bool):
        options["drop_first_actions"] = drop_first
    elif isinstance(drop_first, str) and _SAFE_IDENTIFIER_RE.fullmatch(drop_first):
        options["drop_first_actions"] = drop_first
    return manager_type, options


def extract_execution_profile(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return allowlisted policy/runtime knobs useful in the Control Panel.

    This is deliberately not a resolved-config dump. Paths, credentials, signal
    files, and arbitrary manager parameters stay on the process side; operators
    only need rates, chunking, manager kind, and capability flags when diagnosing
    a run.
    """
    policy = config.get("policy")
    policy = policy if isinstance(policy, Mapping) else {}
    params = policy.get("params")
    params = params if isinstance(params, Mapping) else {}
    action = _profile_section(config, "action_executor")
    manager = _profile_section(config, "device_manager")
    runtime = _profile_section(config, "runtime")
    # OpenPI historically calls this ``ckpt_dir`` while FastWAM owns a
    # ``ckpt_path`` (which may point at either a checkpoint directory or a
    # concrete weights file). The panel only needs to know whether an
    # operator supplied a value, so keep both names in the summary.
    checkpoint = params.get("ckpt_dir")
    if checkpoint is None:
        checkpoint = params.get("ckpt_path")
    checkpoint_configured = isinstance(checkpoint, str) and bool(checkpoint.strip())
    invalid_fields: list[str] = []

    def safe_number(mapping: Mapping[str, Any], key: str, path_name: str) -> Any:
        raw = mapping.get(key)
        value = _safe_number(raw)
        if raw is not None and value is None:
            invalid_fields.append(path_name)
        return value

    def safe_integer(
        mapping: Mapping[str, Any],
        key: str,
        path_name: str,
        *,
        minimum: int = 1,
    ) -> Any:
        raw = mapping.get(key)
        value = _safe_integer(raw, minimum=minimum)
        if raw is not None and value is None:
            invalid_fields.append(path_name)
        return value

    chunk_manager_value = action.get("chunk_manager")
    chunk_manager_type, chunk_manager_options = _safe_chunk_manager(
        chunk_manager_value
    )
    if isinstance(chunk_manager_value, Mapping):
        numeric_bounds = {
            "search_tolerance": (0, 4096),
            "pos_weight": (0.0, 100.0),
            "rot_weight": (0.0, 100.0),
            "ee_weight": (0.0, 100.0),
            "discount_factor": (0.0, 1.0),
            "direction_consistency_range": (0, 4096),
            "skip_first_n_steps": (0, 4096),
        }
        for key, (minimum, maximum) in numeric_bounds.items():
            if key not in chunk_manager_value:
                continue
            raw = _safe_number(chunk_manager_value.get(key))
            if (
                raw is None
                or float(raw) < minimum
                or float(raw) > maximum
                or (
                    isinstance(minimum, int)
                    and int(raw) != raw
                )
            ):
                invalid_fields.append(f"action_executor.chunk_manager.{key}")
        drop_first = chunk_manager_value.get("drop_first_actions")
        if drop_first is not None and not (
            isinstance(drop_first, bool)
            or (
                isinstance(drop_first, str)
                and _SAFE_IDENTIFIER_RE.fullmatch(drop_first)
            )
        ):
            invalid_fields.append("action_executor.chunk_manager.drop_first_actions")

    manager_target_key = (
        "target_fps" if manager.get("target_fps") is not None else "fps"
    )
    manager_target_path = f"device_manager.{manager_target_key}"
    profile = {
        "policy": {
            "class": _safe_identifier(policy.get("class")),
            "fps": safe_number(policy, "fps", "policy.fps"),
            "control_fps": safe_number(
                policy, "control_fps", "policy.control_fps"
            ),
            "value_prediction_query_fps": safe_number(
                policy,
                "value_prediction_query_fps",
                "policy.value_prediction_query_fps",
            ),
            "chunk_length": safe_integer(
                policy, "chunk_length", "policy.chunk_length", minimum=1
            ),
            "obs_horizon": safe_integer(
                policy, "obs_horizon", "policy.obs_horizon", minimum=1
            ),
            "execution_fps": safe_number(
                params, "execution_fps", "policy.params.execution_fps"
            ),
            "checkpoint_configured": checkpoint_configured,
        },
        "action_executor": {
            "type": _safe_identifier(action.get("type")),
            "execution_fps": safe_number(
                action, "execution_fps", "action_executor.execution_fps"
            ),
            "chunk_length": safe_integer(
                action,
                "chunk_length",
                "action_executor.chunk_length",
                minimum=1,
            ),
            "chunk_manager": chunk_manager_type,
            "chunk_manager_options": chunk_manager_options,
            "device_delays": _safe_delay_value(action.get("device_delays")),
            "device_dimension_delays": _safe_delay_value(
                action.get("device_dimension_delays")
            ),
        },
        "device_manager": {
            "type": _safe_identifier(manager.get("type")),
            "target_fps": safe_number(
                manager,
                manager_target_key,
                manager_target_path,
            ),
            "master_device_class_id": _safe_identifier(manager.get("master_device_class_id")),
        },
        "runtime": {
            "enable_device_manager": as_bool(
                runtime.get("enable_device_manager", True)
            ),
            "enable_visualizers": as_bool(
                runtime.get("enable_visualizers", False)
            ),
        },
    }
    profile["invalid_fields"] = list(dict.fromkeys(invalid_fields))[:32]
    return profile


def _safe_joint_vector(value: Any) -> Optional[List[float]]:
    """Return exactly seven finite joint angles for the browser viewer."""
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        return None
    joints: List[float] = []
    for joint in value:
        if isinstance(joint, bool) or not isinstance(joint, (int, float)):
            return None
        numeric = float(joint)
        if not math.isfinite(numeric):
            return None
        joints.append(numeric)
    return joints


def _safe_float_vector(value: Any, length: int) -> Optional[List[float]]:
    """Return an exact-length vector containing only finite real numbers."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result: List[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        numeric = float(item)
        if not math.isfinite(numeric):
            return None
        result.append(numeric)
    return result


def _normalized_quaternion_wxyz(value: Any) -> Optional[List[float]]:
    """Sanitize and normalize a quaternion in Flexiv's ``wxyz`` order."""
    quaternion: Optional[List[float]] = None
    if isinstance(value, Mapping):
        keys = ("w", "x", "y", "z")
        if all(key in value for key in keys):
            quaternion = _safe_float_vector([value[key] for key in keys], 4)
    else:
        quaternion = _safe_float_vector(value, 4)
    if quaternion is None:
        return None
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return [component / norm for component in quaternion]


def _rpy_to_quaternion_wxyz(value: Any) -> Optional[List[float]]:
    """Convert finite roll/pitch/yaw radians to a normalized quaternion."""
    rpy = _safe_float_vector(value, 3)
    if rpy is None:
        return None
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _normalized_quaternion_wxyz(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _parse_tool_transform(value: Any) -> Optional[Dict[str, Any]]:
    """Parse an explicitly configured flange-to-tool transform.

    Flexiv Cartesian poses use ``[x, y, z, qw, qx, qy, qz]``.  Mapping
    inputs may instead provide ``translation``/``position``/``xyz`` plus an
    explicitly ordered ``quaternion_wxyz``, a named quaternion mapping, or
    roll/pitch/yaw radians.  The public representation always uses normalized
    ``wxyz`` so the browser never has to infer an order.
    """
    translation: Optional[List[float]] = None
    quaternion: Optional[List[float]] = None

    pose = _safe_float_vector(value, 7)
    if pose is not None:
        translation = pose[:3]
        quaternion = _normalized_quaternion_wxyz(pose[3:])
    elif isinstance(value, Mapping):
        tcp_location = value.get("tcp_location")
        if tcp_location is not None:
            pose = _safe_float_vector(tcp_location, 7)
            if pose is not None:
                translation = pose[:3]
                quaternion = _normalized_quaternion_wxyz(pose[3:])

        nested_pose = value.get("pose")
        if nested_pose is not None:
            pose = _safe_float_vector(nested_pose, 7)
            if pose is not None:
                translation = pose[:3]
                quaternion = _normalized_quaternion_wxyz(pose[3:])

        if translation is None:
            for key in ("translation", "position", "xyz", "tcp_position"):
                if key in value:
                    translation = _safe_float_vector(value.get(key), 3)
                    break

        if quaternion is None:
            if "quaternion_wxyz" in value:
                quaternion = _normalized_quaternion_wxyz(value.get("quaternion_wxyz"))
            else:
                for key in ("quaternion", "orientation"):
                    if key not in value:
                        continue
                    raw_quaternion = value.get(key)
                    if isinstance(raw_quaternion, Mapping):
                        quaternion = _normalized_quaternion_wxyz(raw_quaternion)
                    elif value.get("quaternion_order", "wxyz") == "wxyz":
                        quaternion = _normalized_quaternion_wxyz(raw_quaternion)
                    break
            if quaternion is None and "tcp_orientation" in value:
                quaternion = _normalized_quaternion_wxyz(value.get("tcp_orientation"))
            if quaternion is None and all(
                key in value for key in ("qw", "qx", "qy", "qz")
            ):
                quaternion = _normalized_quaternion_wxyz(
                    [value["qw"], value["qx"], value["qy"], value["qz"]]
                )
            if quaternion is None and "rpy" in value:
                quaternion = _rpy_to_quaternion_wxyz(value.get("rpy"))

    if translation is None or quaternion is None:
        return None
    return {
        "frame": "flange",
        "translation": translation,
        "quaternion": quaternion,
        "quaternion_order": "wxyz",
    }


def _normalized_tool_element_type(element: Mapping[str, Any]) -> Optional[str]:
    """Return one explicitly declared tool/eef element type.

    Element names are never used as a type heuristic.  This is important for
    configs that also list active robots and cameras next to the active tool.
    """
    for key in ("type", "element_type", "kind", "category", "role", "class"):
        value = element.get(key)
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
        if normalized in _TOOL_ELEMENT_TYPES:
            return normalized
    return None


def _tool_element_is_active(element: Mapping[str, Any]) -> bool:
    """Accept the two field spellings used by Elements config exports."""
    for key in ("activate", "active"):
        if key not in element:
            continue
        value = element.get(key)
        if value is True:
            return True
        if isinstance(value, str):
            return value.strip().casefold() in {"true", "1", "yes", "on"}
        return value == 1 and not isinstance(value, bool)
    return False


def _tool_element_name(
    element: Mapping[str, Any], mapping_key: Optional[str]
) -> Optional[str]:
    for key in ("name", "tool_name", "id"):
        value = element.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if _SAFE_IDENTIFIER_RE.fullmatch(stripped):
                return stripped
            return None
    if mapping_key and _SAFE_IDENTIFIER_RE.fullmatch(mapping_key):
        return mapping_key
    return None


def _iter_element_records(
    value: Any, source: str
) -> List[Tuple[Mapping[str, Any], Optional[str], str]]:
    """Flatten list and name-keyed mapping forms without mutating the config."""
    records: List[Tuple[Mapping[str, Any], Optional[str], str]] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                records.append((item, None, f"{source}[{index}]"))
        return records
    if not isinstance(value, Mapping):
        return records

    # A single inline element is also accepted for defensive compatibility.
    if any(
        key in value
        for key in ("type", "element_type", "kind", "category", "role", "class")
    ):
        records.append((value, None, source))
        return records
    for key, item in value.items():
        if isinstance(item, Mapping):
            records.append((item, str(key), f"{source}.{key}"))
    return records


def _element_applies_to_side(element: Mapping[str, Any], side: str) -> Tuple[bool, int]:
    raw = None
    for key in ("side", "arm_side", "robot_side", "sides"):
        if key in element:
            raw = element.get(key)
            break
    if raw is None:
        return True, 0
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    normalized = {
        str(value).strip().casefold()
        for value in values
        if isinstance(value, (str, int))
    }
    if normalized & {"both", "all", "dual", "*"}:
        return True, 0
    return side in normalized, 1


def _element_applies_to_device(
    element: Mapping[str, Any], device: Mapping[str, Any]
) -> bool:
    for key in ("device_class_id", "robot_device_class_id"):
        if key in element and str(element.get(key)) != str(device.get("device_class_id")):
            return False
    return True


def _active_tool_element(
    config: Mapping[str, Any], device: Mapping[str, Any], side: str
) -> Tuple[
    Optional[str], Optional[str], Optional[Mapping[str, Any]], int, Optional[str]
]:
    """Resolve an active Elements tool with deterministic ambiguity handling."""
    gripper = device.get("gripper")
    containers: List[Tuple[int, str, Any]] = []
    if isinstance(gripper, Mapping) and "elements" in gripper:
        containers.append((0, "gripper.elements", gripper.get("elements")))
    if "elements" in device:
        containers.append((1, "device.elements", device.get("elements")))
    if "elements" in config:
        containers.append((2, "elements", config.get("elements")))

    candidates: List[Tuple[int, int, str, str, Mapping[str, Any]]] = []
    invalid_sources: List[str] = []
    for priority, source, collection in containers:
        for element, mapping_key, element_source in _iter_element_records(
            collection, source
        ):
            if not _tool_element_is_active(element):
                continue
            if _normalized_tool_element_type(element) is None:
                continue
            if not _element_applies_to_device(element, device):
                continue
            applies, side_specificity = _element_applies_to_side(element, side)
            if not applies:
                continue
            name = _tool_element_name(element, mapping_key)
            if name is None:
                invalid_sources.append(element_source)
                continue
            candidates.append(
                (priority, -side_specificity, name.casefold(), element_source, element)
            )

    candidates.sort(key=lambda item: item[:4])
    if not candidates:
        warning = None
        if invalid_sources:
            warning = "active tool element has no safe name: " + ", ".join(
                sorted(invalid_sources, key=str.casefold)
            )
        return None, None, None, 0, warning

    selected = candidates[0]
    selected_name = _tool_element_name(selected[4], None)
    if selected_name is None:
        # A name-keyed mapping can supply the name even when the element does
        # not repeat it. Recover it from the already validated sort key/source.
        selected_name = next(
            (
                _tool_element_name(element, mapping_key)
                for _, source, collection in containers
                for element, mapping_key, element_source in _iter_element_records(
                    collection, source
                )
                if element is selected[4] and element_source == selected[3]
            ),
            None,
        )
    warning = None
    if len(candidates) > 1:
        names = [_tool_element_name(item[4], None) or item[2] for item in candidates]
        warning = (
            f"multiple active tool elements for {side}; selected "
            f"{selected_name} deterministically from {selected[3]} "
            f"among {', '.join(names)}"
        )
    return selected_name, selected[3], selected[4], len(candidates), warning


def _safe_tool_asset_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    raw = value.strip().replace("\\", "/")
    if (
        not raw
        or not _SAFE_TOOL_ASSET_PATH_RE.fullmatch(raw)
        or ":" in raw
        or "?" in raw
        or "#" in raw
        or any(part in {"", ".."} for part in raw.lstrip("/").split("/"))
    ):
        return None
    if raw.startswith("/static/"):
        normalized = raw
    elif raw.startswith("static/"):
        normalized = f"/{raw}"
    elif raw.startswith("assets/"):
        normalized = f"/static/{raw}"
    else:
        return None
    if not normalized.casefold().endswith((".obj", ".urdf", ".mtl")):
        return None
    return normalized


def _safe_tool_asset_scale(value: Any) -> List[float]:
    scale = _safe_float_vector(value, 3)
    if scale is None or any(component <= 0 or component > 1000 for component in scale):
        return [1.0, 1.0, 1.0]
    return scale


def _tool_asset_from_element(
    element: Optional[Mapping[str, Any]], source: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if element is None or source is None:
        return None, None

    raw: Any = None
    asset_source: Optional[str] = None
    for key in ("visual_asset", "tool_asset", "asset", "visual", "mesh", "urdf"):
        if key in element:
            raw = element.get(key)
            asset_source = f"{source}.{key}"
            break
    if raw is None:
        direct = {
            key: element.get(key)
            for key in (
                "url",
                "mesh_url",
                "model_url",
                "urdf_url",
                "material_url",
                "scale",
            )
            if key in element
        }
        if direct:
            raw = direct
            asset_source = source
    if raw is None or asset_source is None:
        return None, None

    if isinstance(raw, str):
        raw = {"url": raw}
    if not isinstance(raw, Mapping):
        return None, f"{asset_source} is not a valid tool asset descriptor"
    url_value = None
    for key in ("url", "path", "src", "mesh_url", "model_url", "urdf_url", "filename"):
        if key in raw:
            url_value = raw.get(key)
            break
    url = _safe_tool_asset_url(url_value)
    if url is None or not url.casefold().endswith((".obj", ".urdf")):
        return (
            None,
            f"{asset_source} must reference a same-origin .obj or .urdf under /static",
        )
    material_url = _safe_tool_asset_url(raw.get("material_url", raw.get("mtl_url")))
    if material_url is not None and not material_url.casefold().endswith(".mtl"):
        material_url = None
    kind = "urdf" if url.casefold().endswith(".urdf") else "obj"
    return {
        "kind": kind,
        "url": url,
        "material_url": material_url,
        "scale": _safe_tool_asset_scale(raw.get("scale", [1.0, 1.0, 1.0])),
        "source": asset_source,
    }, None


def _parse_visual_mount_transform(value: Any) -> Optional[Dict[str, Any]]:
    """Parse one explicit flange-to-visual-base transform without aliases."""
    if not isinstance(value, Mapping) or value.get("frame") != "flange":
        return None
    if set(value) != {"frame", "translation", "quaternion_wxyz"}:
        return None
    translation = _safe_float_vector(value.get("translation"), 3)
    quaternion = _normalized_quaternion_wxyz(value.get("quaternion_wxyz"))
    if translation is None or quaternion is None:
        return None
    return {
        "frame": "flange",
        "translation": translation,
        "quaternion": quaternion,
        "quaternion_order": "wxyz",
    }


def _strict_positive_float(
    value: Any, *, maximum: float
) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0 or numeric > maximum:
        return None
    return numeric


def _tool_visual_articulation(
    visual: Mapping[str, Any], side: str, model: Mapping[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    source = "gripper.visual.articulation"
    if "articulation" in visual and "articulations" in visual:
        return None, source, "tool-visual-articulation-fields-are-invalid"
    raw = visual.get("articulation")
    side_values = visual.get("articulations")
    if side_values is not None:
        if not isinstance(side_values, Mapping) or not set(side_values).issubset(
            {"left", "right"}
        ):
            return None, "gripper.visual.articulations", (
                "tool-visual-articulation-fields-are-invalid"
            )
        if side in side_values:
            raw = side_values.get(side)
            source = f"gripper.visual.articulations.{side}"
    if not isinstance(raw, Mapping):
        return None, source, "tool-visual-articulation-is-required"
    if set(raw) != _TOOL_VISUAL_ARTICULATION_KEYS:
        return None, source, "tool-visual-articulation-fields-are-invalid"

    raw_mode = raw.get("value_mode")
    value_mode = raw_mode.strip().casefold() if isinstance(raw_mode, str) else ""
    if value_mode not in {"width_m", "normalized"}:
        return None, source, "tool-visual-value-mode-is-invalid"
    arm_length_m = _strict_positive_float(raw.get("arm_length_m"), maximum=1.0)
    max_width_m = _strict_positive_float(raw.get("max_width_m"), maximum=1.0)
    apex_angle_max_deg = _strict_positive_float(
        raw.get("apex_angle_max_deg"), maximum=180.0
    )
    if None in (arm_length_m, max_width_m, apex_angle_max_deg):
        return None, source, "tool-visual-articulation-values-are-invalid"

    expected_width = 2.0 * arm_length_m * math.sin(
        math.radians(apex_angle_max_deg) / 2.0
    )
    if not math.isclose(max_width_m, expected_width, rel_tol=1e-6, abs_tol=1e-9):
        return None, source, "tool-visual-calibration-geometry-is-inconsistent"
    return {
        "joint_name": model["joint_name"],
        "joint_lower_rad": model["joint_lower_rad"],
        "joint_upper_rad": model["joint_upper_rad"],
        "closed_joint_rad": model["closed_joint_rad"],
        "value_mode": value_mode,
        "arm_length_m": arm_length_m,
        "max_width_m": max_width_m,
        "apex_angle_max_deg": apex_angle_max_deg,
    }, source, None


def _tool_visual_descriptor(
    device: Mapping[str, Any], side: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Resolve an allowlisted visual without accepting config-provided URLs."""
    gripper = device.get("gripper")
    if not isinstance(gripper, Mapping):
        return False, None
    if "visual" not in gripper:
        # Every ``soft`` gripper in this repository is the LKTech UMI3 used by
        # the dual-arm configs. Older configs predate viewer-only metadata, so
        # keep them visually compatible without mutating their Hydra files or
        # inventing a controller TCP transform.
        gripper_type = gripper.get("gripper_type")
        normalized_type = (
            gripper_type.strip().casefold()
            if isinstance(gripper_type, str)
            else ""
        )
        side_index = 0 if side == "left" else 1
        enabled = gripper.get("enabled", True)
        if isinstance(enabled, (list, tuple)):
            side_enabled = (
                bool(enabled[side_index]) if side_index < len(enabled) else False
            )
        else:
            side_enabled = bool(enabled)
        if normalized_type != "soft" or not side_enabled:
            return False, None

        model = _TOOL_VISUAL_MODELS["umi3"]
        value_mode = gripper.get("value_mode")
        normalized_mode = (
            value_mode.strip().casefold()
            if isinstance(value_mode, str)
            else "width_m"
        )
        if normalized_mode not in {"width_m", "normalized"}:
            normalized_mode = "width_m"
        return True, {
            "model": "umi3",
            "status": "available",
            "kind": model["kind"],
            "url": model["url"],
            "default_variant": model["default_variant"],
            "variants": model["variants"],
            "mirror_axis": "y" if side == "left" else None,
            "mount_transform": {
                "frame": "flange",
                "translation": [0.0, 0.0, 0.0],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "quaternion_order": "wxyz",
            },
            "articulation": {
                "joint_name": model["joint_name"],
                "joint_lower_rad": model["joint_lower_rad"],
                "joint_upper_rad": model["joint_upper_rad"],
                "closed_joint_rad": model["closed_joint_rad"],
                "value_mode": normalized_mode,
                **_UMI3_SOFT_GRIPPER_ARTICULATION,
            },
            "source": "gripper.gripper_type",
            "mount_source": "builtin.umi3-soft.flange-identity",
            "articulation_source": "builtin.umi3-soft.articulation",
            "reason": None,
        }
    raw = gripper.get("visual")
    base = {
        "model": None,
        "status": "invalid-config",
        "kind": None,
        "url": None,
        "mount_transform": None,
        "articulation": None,
        "source": "gripper.visual",
        "mount_source": f"gripper.visual_mount_transforms.{side}",
        "articulation_source": None,
        "reason": "tool-visual-config-is-invalid",
    }
    if not isinstance(raw, Mapping) or not set(raw).issubset(_TOOL_VISUAL_CONFIG_KEYS):
        return True, base
    raw_model = raw.get("model")
    model_name = raw_model.strip().casefold() if isinstance(raw_model, str) else ""
    model = _TOOL_VISUAL_MODELS.get(model_name)
    if model is None:
        base["reason"] = "tool-visual-model-is-not-allowlisted"
        return True, base

    base.update(
        {
            "model": model_name,
            "kind": model["kind"],
            "url": model["url"],
            "default_variant": model["default_variant"],
            "variants": model["variants"],
            "mirror_axis": "y" if side == "left" else None,
        }
    )
    mounts = gripper.get("visual_mount_transforms")
    mount = (
        _parse_visual_mount_transform(mounts.get(side))
        if (
            isinstance(mounts, Mapping)
            and set(mounts).issubset({"left", "right"})
            and side in mounts
        )
        else None
    )
    if mount is None:
        base["reason"] = "tool-visual-mount-transform-is-invalid"
        return True, base

    articulation, articulation_source, articulation_error = _tool_visual_articulation(
        raw, side, model
    )
    base["articulation_source"] = articulation_source
    if articulation is None:
        base["reason"] = articulation_error
        return True, base

    base.update(
        {
            "status": "available",
            "mount_transform": mount,
            "articulation": articulation,
            "reason": None,
        }
    )
    return True, base


def _tool_selection(device: Mapping[str, Any]) -> Tuple[Any, Optional[str], bool]:
    """Return the authoritative explicit tool-name field, including null/blank."""
    gripper = device.get("gripper")
    gripper_mapping = gripper if isinstance(gripper, Mapping) else None

    if gripper_mapping is not None and "tool_name" in gripper_mapping:
        return gripper_mapping.get("tool_name"), "gripper.tool_name", True
    if "tool_name" in device:
        return device.get("tool_name"), "tool_name", True
    # ``gripper.name`` names the actuator driver, not necessarily the Flexiv
    # controller tool.  Treating it as a tool silently visualizes the wrong
    # end effector, so missing tool_name proceeds to Elements fallback.
    return None, None, False


def _tool_transform_candidate(
    device: Mapping[str, Any],
    side: str,
    selected_name: Optional[str],
    selected_element: Optional[Mapping[str, Any]] = None,
    selected_element_source: Optional[str] = None,
) -> Tuple[Any, Optional[str], bool]:
    """Return an explicit transform candidate without consulting RDK/hardware."""
    if selected_element is not None and selected_element_source is not None:
        for key in ("tool_transforms", "tool_poses", "tcp_locations"):
            collection = selected_element.get(key)
            if isinstance(collection, Mapping) and side in collection:
                return (
                    collection[side],
                    f"{selected_element_source}.{key}.{side}",
                    True,
                )
        for key in (
            f"tool_transform_{side}",
            f"tool_pose_{side}",
            f"tcp_location_{side}",
        ):
            if key in selected_element:
                return (
                    selected_element.get(key),
                    f"{selected_element_source}.{key}",
                    True,
                )
        for key in (
            "tool_transform",
            "tool_pose",
            "transform",
            "pose",
            "tcp_location",
            "tcp",
        ):
            if key in selected_element:
                return (
                    selected_element.get(key),
                    f"{selected_element_source}.{key}",
                    True,
                )

    gripper = device.get("gripper")
    containers: List[Tuple[str, Mapping[str, Any]]] = []
    if isinstance(gripper, Mapping):
        containers.append(("gripper", gripper))
    containers.append(("device", device))

    for prefix, container in containers:
        for key in ("tool_transforms", "tool_poses"):
            collection = container.get(key)
            if isinstance(collection, Mapping) and side in collection:
                return collection[side], f"{prefix}.{key}.{side}", True
        for key in (f"tool_transform_{side}", f"tool_pose_{side}"):
            if key in container:
                return container.get(key), f"{prefix}.{key}", True

    for prefix, container in containers:
        for key in ("tool_transform", "tool_pose"):
            if key in container:
                return container.get(key), f"{prefix}.{key}", True

    if selected_name:
        for prefix, container in containers:
            definitions = container.get("tool_definitions")
            if isinstance(definitions, Mapping) and selected_name in definitions:
                definition = definitions[selected_name]
                if isinstance(definition, Mapping):
                    for key in ("transform", "pose", "tool_transform", "tool_pose"):
                        if key in definition:
                            return (
                                definition.get(key),
                                f"{prefix}.tool_definitions.{selected_name}.{key}",
                                True,
                            )
                return (
                    definition,
                    f"{prefix}.tool_definitions.{selected_name}",
                    True,
                )
    return None, None, False


def _tool_sides(
    device: Mapping[str, Any], resolved: Mapping[str, Any]
) -> Tuple[str, ...]:
    """Map one Rizon config device to viewer sides without touching hardware."""
    robot_sn = device.get("robot_sn")
    home = _selected_home_mapping(device.get("home"))
    dual = isinstance(robot_sn, (list, tuple)) and len(robot_sn) >= 2
    dual = dual or (
        isinstance(home, Mapping) and ("joints_left" in home or "joints_right" in home)
    )
    if dual:
        return ("left", "right")
    explicit = _single_robot_side(device)
    if explicit:
        return (explicit,)
    # Match the aggregate state endpoint's established single-arm convention.
    return ("right",) if resolved["right"]["status"] == "not-configured" else ("left",)


def extract_robot_tools(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return sanitized per-side tool selection and static flange transforms.

    This is intentionally config-only. Explicit non-empty ``tool_name`` wins.
    Null, blank, or missing tool names fall back to the active tool/eef entry
    exported in ``elements``. No robot connection or SDK import is performed.
    """
    resolved: Dict[str, Dict[str, Any]] = {
        side: {
            "name": None,
            "resolved_tool_name": None,
            "status": "not-configured",
            "transform": None,
            "asset": None,
            "source": None,
            "resolved_tool_source": None,
            "transform_source": None,
            "resolution": "none",
            "resolution_warning": None,
            "active_element_count": 0,
            "reason": "rizon-tool-not-configured",
        }
        for side in ("left", "right")
    }
    devices = config.get("devices", [])
    if not isinstance(devices, list):
        return resolved

    for device in devices:
        if not isinstance(device, Mapping) or not _device_is_enabled(device):
            continue
        if str(device.get("class", "")).casefold() != "rizonrobot":
            continue

        raw_name, name_source, _explicit_field_present = _tool_selection(device)
        explicit_name: Optional[str] = None
        invalid_name = False
        if raw_name is None:
            pass
        elif isinstance(raw_name, str):
            stripped = raw_name.strip()
            if stripped and _SAFE_IDENTIFIER_RE.fullmatch(stripped):
                explicit_name = stripped
            elif stripped:
                invalid_name = True
        else:
            invalid_name = True

        for side in _tool_sides(device, resolved):
            resolved_name = explicit_name
            element: Optional[Mapping[str, Any]] = None
            element_source: Optional[str] = None
            active_element_count = 0
            resolution_warning: Optional[str] = None
            resolution = "explicit-tool-name" if resolved_name else "none"
            resolved_source = name_source if resolved_name else None
            if not resolved_name and not invalid_name:
                (
                    fallback_name,
                    element_source,
                    element,
                    active_element_count,
                    resolution_warning,
                ) = _active_tool_element(config, device, side)
                if fallback_name:
                    resolved_name = fallback_name
                    resolved_source = element_source
                    resolution = "elements-active-tool"

            candidate, transform_source, transform_present = _tool_transform_candidate(
                device,
                side,
                resolved_name,
                selected_element=element,
                selected_element_source=element_source,
            )
            transform = _parse_tool_transform(candidate) if transform_present else None
            asset, asset_warning = _tool_asset_from_element(element, element_source)
            if asset_warning:
                resolution_warning = (
                    f"{resolution_warning}; {asset_warning}"
                    if resolution_warning
                    else asset_warning
                )
            if transform is not None:
                status = "available"
                reason = None
            elif transform_present:
                status = "invalid-transform"
                reason = "configured-tool-transform-is-invalid"
            elif invalid_name:
                status = "invalid-config"
                reason = "configured-tool-name-is-invalid"
            elif resolved_name:
                status = "name-known-pose-unknown"
                reason = "tool-transform-not-present-in-config"
            else:
                status = "not-configured"
                reason = "rizon-tool-not-configured"
            tool_payload = {
                "name": resolved_name,
                "resolved_tool_name": resolved_name,
                "status": status,
                "transform": transform,
                "asset": asset,
                "source": transform_source or resolved_source,
                "resolved_tool_source": resolved_source,
                "transform_source": transform_source,
                "resolution": resolution,
                "resolution_warning": resolution_warning,
                "active_element_count": active_element_count,
                "reason": reason,
            }
            visual_declared, visual = _tool_visual_descriptor(device, side)
            if visual_declared:
                tool_payload["visual"] = visual
            resolved[side] = tool_payload
    return resolved


def _selected_home_mapping(home: Any) -> Optional[Mapping[str, Any]]:
    """Resolve the same direct/preset home structure used by RizonRobot."""
    if not isinstance(home, Mapping):
        return None
    presets = home.get("presets")
    if presets is None:
        return home
    if not isinstance(presets, list) or not presets:
        return None
    preset_index = home.get("preset_index", 0)
    if (
        isinstance(preset_index, bool)
        or not isinstance(preset_index, int)
        or not 0 <= preset_index < len(presets)
    ):
        return None
    selected = presets[preset_index]
    return selected if isinstance(selected, Mapping) else None


def _single_robot_side(device: Mapping[str, Any]) -> Optional[str]:
    """Use only explicit config structure to place a single-arm home pose."""
    for key in ("arm_side", "robot_side", "side"):
        value = device.get(key)
        if isinstance(value, str) and value.strip().lower() in {"left", "right"}:
            return value.strip().lower()
    return None


def extract_robot_home(config: Mapping[str, Any]) -> Dict[str, Optional[List[float]]]:
    """Extract a redacted dual-arm home pose from enabled Rizon config devices.

    Dual-arm homes are mapped only through their explicit ``joints_left`` and
    ``joints_right`` fields. A legacy single-arm ``joints`` vector is exposed
    only when that device declares an explicit side, avoiding a left/right
    guess in the control plane.
    """
    resolved: Dict[str, Optional[List[float]]] = {
        "joints_left": None,
        "joints_right": None,
    }
    devices = config.get("devices", [])
    if not isinstance(devices, list):
        return resolved

    for device in devices:
        if not isinstance(device, Mapping) or not _device_is_enabled(device):
            continue
        if str(device.get("class", "")).casefold() != "rizonrobot":
            continue
        selected = _selected_home_mapping(device.get("home"))
        if selected is None:
            continue

        for side in ("left", "right"):
            key = f"joints_{side}"
            joints = _safe_joint_vector(selected.get(key))
            if joints is not None and resolved[key] is None:
                resolved[key] = joints

        single_joints = _safe_joint_vector(selected.get("joints"))
        single_side = _single_robot_side(device)
        if single_joints is not None and single_side is not None:
            key = f"joints_{single_side}"
            if resolved[key] is None:
                resolved[key] = single_joints

    return resolved


def _safe_dtype(value: Any) -> Optional[str]:
    if isinstance(value, str) and _SAFE_DTYPE_RE.fullmatch(value):
        return value
    return None


def _safe_shm_name(value: Any) -> Optional[str]:
    if isinstance(value, str) and _SAFE_SHM_NAME_RE.fullmatch(value):
        return value
    return None


def _configured_device_key(device: Mapping[str, Any]) -> Optional[str]:
    device_class = device.get("class")
    device_class_id = device.get("device_class_id")
    if not isinstance(device_class, str) or device_class_id is None:
        return None
    return f"{device_class}:{device_class_id}"


def _device_reference_aliases(device_key: str) -> tuple[str, str]:
    device_class, device_class_id = device_key.rsplit(":", 1)
    return device_key, f"{device_class}_{device_class_id}"


def _policy_contract_device_keys(
    config: Mapping[str, Any], keys_by_alias: Mapping[str, str]
) -> set[str]:
    """Return config devices explicitly consumed or controlled by the policy."""
    from utils.policy_control_config import ControlTargetConfig
    required: set[str] = set()
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        return required

    obs = policy.get("obs")
    raw_observed = obs.get("devices") if isinstance(obs, Mapping) else None
    if isinstance(raw_observed, list):
        for value in raw_observed:
            if isinstance(value, str):
                key = keys_by_alias.get(value.strip())
                if key:
                    required.add(key)

    controls = policy.get("controls")
    if isinstance(controls, list):
        for index, control in enumerate(controls):
            if not isinstance(control, Mapping):
                continue
            try:
                target = ControlTargetConfig.from_mapping(
                    control, field=f"policy.controls[{index}]"
                )
            except (TypeError, ValueError):
                continue
            device_key = keys_by_alias.get(target.device_key)
            if device_key:
                required.add(device_key)
    return required


def extract_camera_slots(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return stable head/left/right camera roles for the control panel.

    Runtime bindings remain keyed by the logical config device.  The physical
    V4L2 camera selected by the user is only a session override for that key.
    A slot is required exactly when the policy observation contract consumes
    it; a recording-only head camera therefore remains optional for inference.
    """
    raw_devices = config.get("devices")
    devices = (
        [
            device
            for device in raw_devices
            if isinstance(device, Mapping) and _device_is_enabled(device)
        ]
        if isinstance(raw_devices, list)
        else []
    )

    camera_devices: list[Mapping[str, Any]] = []
    keys_by_alias: Dict[str, str] = {}
    devices_by_key: Dict[str, Mapping[str, Any]] = {}
    for device in devices:
        key = _configured_device_key(device)
        if key is None:
            continue
        devices_by_key[key] = device
        for alias in _device_reference_aliases(key):
            keys_by_alias[alias] = key
        class_name = str(device.get("class") or "").casefold()
        shape = _safe_shape(device.get("shape", device.get("data_shape")))
        if (
            "camera" in class_name
            or "iphone" in class_name
            or (shape is not None and len(shape) == 3 and shape[-1] in (1, 3, 4))
        ):
            camera_devices.append(device)

    def canonical_reference(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        return keys_by_alias.get(value.strip())

    policy = config.get("policy")
    obs = policy.get("obs") if isinstance(policy, Mapping) else None
    raw_observed = obs.get("devices") if isinstance(obs, Mapping) else None
    camera_keys = {
        key
        for device in camera_devices
        if (key := _configured_device_key(device)) is not None
    }
    observed_camera_keys: list[str] = []
    if isinstance(raw_observed, list):
        for value in raw_observed:
            key = canonical_reference(value)
            if key and key not in observed_camera_keys and key in camera_keys:
                observed_camera_keys.append(key)
    observed_set = set(observed_camera_keys)

    try:
        recording_mapping = normalize_recording_camera_mapping(config)
    except ValueError:
        recording_mapping = {
            "head_camera": None,
            "left_camera": None,
            "right_camera": None,
        }

    assigned: Dict[str, Optional[str]] = {
        role: canonical_reference(recording_mapping.get(f"{role}_camera"))
        for role in ("head", "left", "right")
    }
    sources: Dict[str, str] = {
        role: "recording" if assigned[role] else ""
        for role in ("head", "left", "right")
    }

    camera_config = config.get("camera")
    if isinstance(camera_config, Mapping):
        for role in ("head", "left", "right"):
            if assigned[role] is not None:
                continue
            configured_id = camera_config.get(f"{role}_capture_id")
            matches = [
                _configured_device_key(device)
                for device in camera_devices
                if configured_id is not None and str(device.get("capture_id")) == str(configured_id)
            ]
            if len(matches) == 1:
                assigned[role] = matches[0]
                sources[role] = "camera"

    used = {key for key in assigned.values() if key}
    remaining_observed = [key for key in observed_camera_keys if key not in used]
    if assigned["head"] is None and len(remaining_observed) >= 3:
        assigned["head"] = remaining_observed.pop(0)
        sources["head"] = "policy"
    for role in ("left", "right"):
        if assigned[role] is None and remaining_observed:
            assigned[role] = remaining_observed.pop(0)
            sources[role] = "policy"

    slots: Dict[str, Dict[str, Any]] = {}
    for role in ("head", "left", "right"):
        key = assigned[role]
        required = bool(key and key in observed_set)
        slots[role] = {
            "role": role,
            "device_key": key,
            "configured": bool(key and key in devices_by_key),
            "required": required,
            "inference_required": required,
            "recording_configured": bool(
                canonical_reference(recording_mapping.get(f"{role}_camera"))
            ),
            "source": sources[role] or "unconfigured",
        }
    return slots


def extract_robot_slots(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return the Rizon and YAM head-arm control slots.

    ``YamHeadRobot`` is an active motion dependency of every config that
    declares it. It may track the Rizon TCP midpoint without appearing in the
    OpenPI observation list, so policy-contract inspection alone is not enough
    to decide whether runtime startup may omit it.
    """
    raw_devices = config.get("devices")
    devices = (
        [
            device
            for device in raw_devices
            if isinstance(device, Mapping) and _device_is_enabled(device)
        ]
        if isinstance(raw_devices, list)
        else []
    )
    keys_by_alias: Dict[str, str] = {}
    devices_by_class: Dict[str, list[str]] = {}
    for device in devices:
        key = _configured_device_key(device)
        if key is None:
            continue
        for alias in _device_reference_aliases(key):
            keys_by_alias[alias] = key
        class_name = str(device.get("class") or "").casefold()
        devices_by_class.setdefault(class_name, []).append(key)

    contract_keys = _policy_contract_device_keys(config, keys_by_alias)

    def slot(role: str, class_name: str, *, required_by_default: bool) -> Dict[str, Any]:
        candidates = devices_by_class.get(class_name.casefold(), [])
        key = candidates[0] if candidates else None
        policy_required = bool(key and key in contract_keys)
        required = bool(key and (required_by_default or policy_required))
        return {
            "role": role,
            "device_key": key,
            "configured": bool(key),
            "required": required,
            "inference_required": required,
            "source": (
                "policy"
                if policy_required
                else "runtime-default"
                if key and required_by_default
                else "config"
                if key
                else "unconfigured"
            ),
        }

    return {
        "rizon": slot("rizon", "RizonRobot", required_by_default=True),
        "i2rt_head": slot(
            "i2rt_head", "YamHeadRobot", required_by_default=True
        ),
    }


def _safe_yam_initial_joints(value: Any, joint_count: int) -> Optional[List[float]]:
    if isinstance(value, str):
        try:
            value = [
                float(token.strip())
                for token in value.replace(";", ",").split(",")
                if token.strip()
            ]
        except ValueError:
            return None
    return _safe_float_vector(value, joint_count)


def extract_yam_visualization(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the browser-safe YAM model and startup-pose contract."""

    unavailable = {
        "configured": False,
        "device_key": None,
        "model": None,
        "status": "not-configured",
        "urdf_url": None,
        "base_position_world": list(_YAM_BASE_POSITION_WORLD),
        "joint_count": 6,
        "joint_names": [f"joint{index}" for index in range(1, 7)],
        "initial_joints": None,
    }
    raw_devices = config.get("devices")
    if not isinstance(raw_devices, list):
        return unavailable

    for device in raw_devices:
        if not isinstance(device, Mapping) or not _device_is_enabled(device):
            continue
        if str(device.get("class") or "").casefold() != "yamheadrobot":
            continue

        model = str(device.get("yam_arm") or "yam").strip().casefold()
        spec = _YAM_VISUAL_MODELS.get(model)
        if spec is None:
            return {
                **unavailable,
                "configured": True,
                "device_key": _configured_device_key(device),
                "model": model if _SAFE_IDENTIFIER_RE.fullmatch(model) else None,
                "status": "unsupported-model",
            }

        joint_count = int(spec["joint_count"])
        initial_joints = _safe_yam_initial_joints(
            device.get("yam_initial_q"), joint_count
        )
        return {
            "configured": True,
            "device_key": _configured_device_key(device),
            "model": model,
            "status": "available" if initial_joints is not None else "invalid-initpos",
            "urdf_url": spec["url"],
            "base_position_world": list(_YAM_BASE_POSITION_WORLD),
            "joint_count": joint_count,
            "joint_names": [
                f"joint{index}" for index in range(1, joint_count + 1)
            ],
            "initial_joints": initial_joints,
        }

    return unavailable


def _device_summary(device: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the intentionally small public representation of one device."""
    device_class = _safe_identifier(device.get("class"))
    device_class_id = _safe_identifier(device.get("device_class_id"))
    class_text = str(device_class) if device_class is not None else "unknown"
    id_text = str(device_class_id) if device_class_id is not None else "unknown"
    class_name = class_text.lower()

    state_feedback = device.get("state_feedback")
    state_feedback = state_feedback if isinstance(state_feedback, Mapping) else device
    state_format = state_feedback.get("format")
    state_format = state_format if isinstance(state_format, Mapping) else {}
    shape = _safe_shape(
        state_format.get("shape", device.get("shape", device.get("data_shape")))
    )
    dtype = _safe_dtype(
        state_format.get("dtype", device.get("dtype", device.get("data_dtype")))
    )
    fps = _safe_number(state_feedback.get("fps"))
    shm_name = _safe_shm_name(
        device.get(
            "shm_name",
            device.get("shared_memory_name", device.get("data_shm_name")),
        )
    )
    preview = (
        "camera" in class_name
        or "iphone" in class_name
        or (shape is not None and len(shape) == 3 and shape[-1] in (1, 3, 4))
    )

    # Keep these keys stable for the browser while ensuring every value comes
    # from an explicit, validated scalar above.  In particular, do not add a
    # catch-all ``config`` field here.
    return {
        "key": f"{class_text}:{id_text}",
        "class": device_class,
        "device_class": device_class,
        "type": device_class,
        "id": device_class_id,
        "device_class_id": device_class_id,
        "enabled": _device_is_enabled(device),
        "capabilities": {
            "preview": preview,
            "motion": "robot" in class_name,
        },
        "fps": fps,
        "shape": shape,
        "dtype": dtype,
        "shm_name": shm_name,
    }


def summarize_runtime_config(
    config_name: str, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the safe UI-facing summary for an already composed config."""
    name = normalize_config_name(config_name)
    errors = validate_runtime_config(config)
    devices = config.get("devices", [])
    resolved_devices: List[Dict[str, Any]] = []
    disabled_count = 0
    if isinstance(devices, list):
        for device in devices:
            if not isinstance(device, Mapping):
                continue
            if not _device_is_enabled(device):
                disabled_count += 1
                continue
            resolved_devices.append(_device_summary(device))
    metadata = runtime_config_metadata(name, config, errors=errors)
    return {
        "name": name,
        "valid": not errors,
        "errors": errors,
        "metadata": metadata,
        **metadata,
        "device_count": len(resolved_devices),
        "disabled_count": disabled_count,
        "devices": resolved_devices,
        "camera_slots": extract_camera_slots(config),
        "robot_slots": extract_robot_slots(config),
        "yam_visualization": extract_yam_visualization(config),
        "home": extract_robot_home(config),
        "configured_tools": extract_robot_tools(config),
        "gripper_slots": extract_gripper_slots(config),
        "execution_profile": extract_execution_profile(config),
    }


def inspect_runtime_config(
    config_name: str, config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Compose a config and return only its safe UI-facing device summary.

    ``compose_runtime_config`` remains the private/full-fidelity path used by
    process startup.  This function is used by the web API, so it must never
    return the resolved config or arbitrary per-device fields.
    """
    name = normalize_config_name(config_name)
    config = compose_runtime_config(name, config_dir=config_dir)
    return summarize_runtime_config(name, config)


__all__ = [
    "DEFAULT_CONFIG_DIR",
    "RuntimeConfigError",
    "compose_runtime_config",
    "extract_robot_home",
    "extract_robot_tools",
    "extract_camera_slots",
    "extract_execution_profile",
    "extract_gripper_slots",
    "extract_robot_slots",
    "extract_yam_visualization",
    "inspect_runtime_config",
    "list_runtime_configs",
    "normalize_config_name",
    "normalize_runtime_config",
    "runtime_config_metadata",
    "summarize_runtime_config",
    "validate_runtime_config",
]
