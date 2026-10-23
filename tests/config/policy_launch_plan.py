#!/usr/bin/env python3
"""Check policy/device launch contracts without starting a process or hardware.

The single-config mode is useful for a new or failing preset. The
--all-supported mode checks native launch configs for CI and
configuration-regression coverage. This helper intentionally does *not* import
device_starter, device classes, or policy runners.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

# This module intentionally lives with configuration contract tests, but remains
# directly executable from any working directory for CI/debugging workflows.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.errors import SHMContractError
from utils.policy_control_config import ControlTargetConfig

DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
_CONFIG_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:@+-]{1,160}$")
_COMPOSE_LOCK = threading.RLock()
_POLICY_INFER_NAMESPACES = frozenset(
    {"diffusion", "fastwam", "lingbotva", "nmx_vla", "openpi", "robotwin"}
)


class LaunchPlanError(ValueError):
    """Raised when a config cannot be safely resolved into a launch plan."""


def flatten_hydra_config(config: Any) -> dict[str, Any]:
    """Load the shared resolver lazily while keeping a patchable test seam."""
    from utils.config_parser import normalize_hydra_config as _normalize_hydra_config

    return _normalize_hydra_config(config)


def _config_source(root: Path, config_name: str) -> Path | None:
    """Return the in-tree YAML source for a normalized Hydra config name."""
    for suffix in (".yaml", ".yml"):
        candidate = root / f"{config_name}{suffix}"
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.is_relative_to(root):
            return resolved
    return None


def resolve_config_name(
    config_name: str, config_dir: Path | None = None
) -> tuple[str, Path]:
    """Resolve a config request to a native Hydra name and source YAML path.

    Namespaced configs such as ``policy_infer/openpi/example`` and
    ``policy_infer/fastwam/example`` are preserved as-is, which is important
    because Hydra treats them differently from the historical bare ``example``
    shorthand.
    Bare names remain supported only as an unambiguous convenience fallback to
    ``configs/policy_infer/openpi/<name>.yaml``.
    """
    if not isinstance(config_name, str) or not config_name.strip():
        raise LaunchPlanError("--config-name is required")

    requested = config_name.strip()
    for suffix in (".yaml", ".yml"):
        if requested.endswith(suffix):
            requested = requested[: -len(suffix)]
            break

    parts = requested.split("/")
    if (
        not requested
        or requested.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or not _is_native_launch_config_path(parts)
        or any(_CONFIG_COMPONENT_RE.fullmatch(part) is None for part in parts)
    ):
        raise LaunchPlanError(
            "config name must be a relative Hydra name such as "
            "policy_infer/openpi/my_config"
        )

    root = (config_dir or DEFAULT_CONFIG_DIR).resolve()
    if not root.is_dir():
        raise LaunchPlanError(f"config directory does not exist: {root}")

    source = _config_source(root, requested)
    if source is not None:
        return requested, source

    if len(parts) == 1:
        native_name = f"policy_infer/openpi/{requested}"
        source = _config_source(root, native_name)
        if source is not None:
            return native_name, source

    raise LaunchPlanError(f"unknown Hydra config: {requested}")


def _is_native_launch_config_path(parts: list[str]) -> bool:
    """Return whether path components describe a supported launch config."""
    if len(parts) == 1:
        return True
    if len(parts) == 2:
        return parts[0] == "control_panel"
    return (
        len(parts) == 3
        and parts[0] == "policy_infer"
        and parts[1] in _POLICY_INFER_NAMESPACES
    )


def compose_launch_config(
    config_name: str, config_dir: Path | None = None
) -> tuple[str, Path, dict[str, Any]]:
    """Compose a launch config without running Hydra jobs or touching hardware."""
    native_name, source = resolve_config_name(config_name, config_dir)
    root = (config_dir or DEFAULT_CONFIG_DIR).resolve()

    try:
        # Keep direct --help/path-validation usable in a minimal interpreter;
        # Hydra is needed only when composing an actual config.
        from hydra import compose, initialize_config_dir
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        with _COMPOSE_LOCK:
            hydra_state = HydraConfig.instance()
            previous_hydra_config = hydra_state.cfg
            try:
                with initialize_config_dir(
                    config_dir=str(root),
                    version_base="1.3",
                    job_name="policy_launch_plan",
                ):
                    composed = compose(
                        config_name=native_name, return_hydra_config=True
                    )
                    # Some device configs contain ${hydra:runtime.*}.  Supplying
                    # this transient context mirrors @hydra.main while keeping
                    # the operation strictly in-memory.
                    hydra_state.set_config(composed)
                    application_keys = [
                        str(key) for key in composed.keys() if key != "hydra"
                    ]
                    application_config = OmegaConf.masked_copy(
                        composed, application_keys
                    )
                    resolved = flatten_hydra_config(application_config)
            finally:
                hydra_state.cfg = previous_hydra_config
    except LaunchPlanError:
        raise
    except Exception as exc:
        raise LaunchPlanError(f"failed to compose {native_name}: {exc}") from exc

    if not isinstance(resolved, dict):
        raise LaunchPlanError(
            f"composed config must be a mapping, got {type(resolved).__name__}"
        )

    config = dict(resolved)
    devices = config.get("devices")
    if isinstance(devices, Mapping):
        config["devices"] = devices.get("devices", [])
    for section_name in ("device_manager", "action_executor"):
        section = config.get(section_name)
        if isinstance(section, Mapping) and "type" not in section:
            nested = section.get(section_name)
            if isinstance(nested, Mapping):
                config[section_name] = dict(nested)
    return native_name, source, config


def _enabled(device: Mapping[str, Any]) -> bool:
    value = device.get("enabled", True)
    if isinstance(value, str):
        return value.strip().casefold() not in {"false", "0", "no", "off"}
    return value is not False


def _normalize_optional_conda_env(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise LaunchPlanError(f"{field} must be a non-empty Conda environment name")
    return value.strip()


def _safe_identifier(value: Any) -> str | int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and _SAFE_IDENTIFIER_RE.fullmatch(value):
        return value
    return None


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return value


def _safe_shape(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) > 8:
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        result.append(item)
    return result


def _safe_controls(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    controls: list[dict[str, Any]] = []
    for index, control in enumerate(value):
        if not isinstance(control, Mapping):
            raise LaunchPlanError(f"{field}[{index}] must be a mapping")
        try:
            controls.append(
                ControlTargetConfig.from_mapping(
                    control, field=f"{field}[{index}]"
                ).config_mapping()
            )
        except (TypeError, ValueError, SHMContractError) as exc:
            raise LaunchPlanError(str(exc)) from exc
    return controls


def _policy_runner(policy_class: str | None) -> str | None:
    if not policy_class:
        return None
    prefixes = (
        ("OpenPi", "consumer.inference.openpi_policy_infer"),
        ("Diffusion", "consumer.inference.diffusion_policy_infer"),
        ("LingBotVA", "consumer.inference.lingbot_va_infer"),
        ("NMXVLA", "consumer.inference.nmx_vla_infer"),
        ("RoboTwin", "consumer.inference.robotwin_policy_infer"),
    )
    for prefix, module in prefixes:
        if policy_class.startswith(prefix):
            return module
    if policy_class == "FastWAMPolicy":
        return "consumer.inference.fastwam_policy_infer"
    return None


def _device_contract(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw_devices = config.get("devices")
    if not isinstance(raw_devices, list):
        return [], 0

    devices: list[dict[str, Any]] = []
    disabled_count = 0
    for raw_device in raw_devices:
        if not isinstance(raw_device, Mapping):
            continue
        if not _enabled(raw_device):
            disabled_count += 1
            continue
        device_class = _safe_identifier(raw_device.get("class"))
        device_class_id = _safe_identifier(raw_device.get("device_class_id"))
        state_feedback = raw_device.get("state_feedback")
        state_feedback = (
            state_feedback if isinstance(state_feedback, Mapping) else raw_device
        )
        state_format = state_feedback.get("format")
        state_format = state_format if isinstance(state_format, Mapping) else raw_device
        control = raw_device.get("control")
        control = control if isinstance(control, Mapping) else {}
        control_format = control.get("format")
        control_format = control_format if isinstance(control_format, Mapping) else {}
        devices.append(
            {
                "key": (
                    f"{device_class}:{device_class_id}"
                    if device_class is not None and device_class_id is not None
                    else None
                ),
                "class": device_class,
                "device_class_id": device_class_id,
                "fps": _safe_number(state_feedback.get("fps")),
                "data_shape": _safe_shape(
                    state_format.get("shape", raw_device.get("data_shape"))
                ),
                "data_dtype": _safe_identifier(
                    state_format.get("dtype", raw_device.get("data_dtype"))
                ),
                "control_shape": _safe_shape(control_format.get("shape")),
                "control_dtype": _safe_identifier(control_format.get("dtype")),
                "gripper_configured": isinstance(raw_device.get("gripper"), Mapping),
            }
        )
    return devices, disabled_count


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    return section if isinstance(section, Mapping) else {}


def _command(conda_env: str | None, arguments: Sequence[str]) -> dict[str, Any]:
    """Render a compatibility command without inventing a command-prefix choice."""
    if conda_env is not None:
        argv = ["conda", "run", "-n", conda_env, "python", *arguments]
        return {
            "argv": argv,
            "shell": shlex.join(argv),
            "execution_prefix_resolution_required": False,
        }
    argv = ["python", *arguments]
    return {
        "argv": argv,
        "shell": shlex.join(argv),
        "execution_prefix_resolution_required": True,
    }


def build_launch_plan(
    config_name: str,
    *,
    config_dir: Path | None = None,
    device_env: str | None = None,
    policy_env: str | None = None,
) -> dict[str, Any]:
    """Build a reviewable plan without launching a process or opening hardware."""
    normalized_device_env = _normalize_optional_conda_env(device_env, field="device_env")
    normalized_policy_env = _normalize_optional_conda_env(policy_env, field="policy_env")

    native_name, source, config = compose_launch_config(config_name, config_dir)
    policy = _section(config, "policy")
    policy_params = policy.get("params")
    policy_params = policy_params if isinstance(policy_params, Mapping) else {}
    policy_class_value = _safe_identifier(policy.get("class"))
    policy_class = policy_class_value if isinstance(policy_class_value, str) else None
    runner_module = _policy_runner(policy_class)
    is_fastwam = policy_class == "FastWAMPolicy"
    controls = _safe_controls(policy.get("controls"), field="policy.controls")
    devices, disabled_count = _device_contract(config)
    action_executor = _section(config, "action_executor")
    action_controls = _safe_controls(
        action_executor.get("robot_control_shms"),
        field="action_executor.robot_control_shms",
    )
    chunk_manager = action_executor.get("chunk_manager")
    chunk_manager_type = (
        _safe_identifier(chunk_manager.get("type"))
        if isinstance(chunk_manager, Mapping)
        else None
    )
    device_manager = _section(config, "device_manager")

    try:
        source_display = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_display = str(source)

    contract_errors: list[str] = []
    if policy_class is None:
        contract_errors.append(
            "policy.class is missing or invalid; no consumer command was generated"
        )
    elif runner_module is None:
        contract_errors.append(
            f"no registered consumer runner for policy class {policy_class!r}"
        )
    if not devices:
        contract_errors.append("no enabled devices were resolved")
    if not controls:
        contract_errors.append(
            "policy.controls is empty; ActionExecutor has no declared control channel"
        )

    commands: dict[str, Any] = {
        "working_directory": str(REPO_ROOT),
        "device_starter": _command(
            normalized_device_env,
            ["device_starter.py", "--config-name", native_name],
        ),
        "server": None,
        "consumer": None,
    }
    if is_fastwam:
        # FastWAM owns model memory in a separate process. Keep this command
        # explicit in the read-only plan so direct launches cannot accidentally
        # start a client without its websocket server.
        commands["server"] = _command(
            normalized_policy_env,
            ["consumer/fastwam_server.py", "--config-name", native_name],
        )
    if runner_module is not None:
        commands["consumer"] = _command(
            normalized_policy_env,
            ["-m", runner_module, "--config-name", native_name],
        )

    return {
        "mode": "read-only-launch-plan",
        "started_processes": 0,
        "environment_resolution_required": (
            normalized_device_env is None or normalized_policy_env is None
        ),
        "config": {
            "requested_name": config_name,
            "hydra_config_name": native_name,
            "source_path": source_display,
        },
        "policy": {
            "class": policy_class,
            "runner_module": runner_module,
            "fps": _safe_number(policy.get("fps")),
            "control_fps": _safe_number(policy.get("control_fps")),
            "chunk_length": _safe_number(policy.get("chunk_length")),
            "obs_horizon": _safe_number(policy_params.get("obs_horizon")),
            "num_robots": _safe_number(policy_params.get("num_robots")),
            "pi_config_name": _safe_identifier(policy_params.get("pi_config_name")),
            "checkpoint_configured": any(
                isinstance(policy_params.get(key), str)
                and policy_params[key].strip()
                for key in ("ckpt_dir", "ckpt_path")
            ),
            "server_required": is_fastwam,
            "observation_devices": [
                value
                for value in (
                    _safe_identifier(item)
                    for item in _section(policy, "obs").get("devices", [])
                )
                if value is not None
            ],
            "controls": controls,
        },
        "devices": {"enabled": devices, "disabled_count": disabled_count},
        "action_executor": {
            "type": _safe_identifier(action_executor.get("type")),
            "execution_fps": _safe_number(action_executor.get("execution_fps")),
            "chunk_length": _safe_number(action_executor.get("chunk_length")),
            "chunk_manager": chunk_manager_type,
            "control_channels": action_controls or controls,
        },
        "device_manager": {
            "type": _safe_identifier(device_manager.get("type")),
            "master_device_class_id": _safe_identifier(
                device_manager.get("master_device_class_id")
            ),
            "target_fps": _safe_number(
                device_manager.get("target_fps", device_manager.get("fps"))
            ),
        },
        "commands": commands,
        "contract": {
            "valid": not contract_errors,
            "errors": contract_errors,
        },
        # Keep the legacy key stable for callers that render optional warnings.
        "warnings": [],
        "start_order": [
            "Resolve the planner, device, and policy command prefixes before launch.",
            "Review this plan and complete the required calibration/preflight checks.",
            "Run the resolved device entrypoint in one terminal and wait for device readiness.",
            "For FastWAM, start the model server and wait for /healthz readiness before the policy client.",
            "Run the resolved policy entrypoint in a second terminal only after the device side is ready.",
        ],
    }


def discover_supported_config_names(config_dir: Path | None = None) -> list[str]:
    """Return native OpenPI/FastWAM launch presets for automated checking.

    Hydra config groups and historical experiment presets are intentionally
    excluded. Generated configs are user artifacts, not repository contracts,
    so they are also omitted from this batch scope.
    """
    root = (config_dir or DEFAULT_CONFIG_DIR).resolve()
    if not root.is_dir():
        raise LaunchPlanError(f"config directory does not exist: {root}")

    candidates = [
        path
        for suffix in ("*.yaml", "*.yml")
        for path in root.glob(suffix)
        if path.is_file()
    ]
    policy_infer_root = root / "policy_infer"
    for namespace in ("openpi", "fastwam"):
        namespace_root = policy_infer_root / namespace
        if namespace_root.is_dir():
            candidates.extend(
                path
                for suffix in ("*.yaml", "*.yml")
                for path in namespace_root.rglob(suffix)
                if path.is_file()
            )

    names = {
        path.relative_to(root).with_suffix("").as_posix()
        for path in candidates
        if not path.stem.startswith("generated_")
        and not (
            path.relative_to(root).parts[:2] == ("policy_infer", "openpi")
            and len(path.relative_to(root).parts) != 3
        )
        and not (
            path.relative_to(root).parts[:2] == ("policy_infer", "fastwam")
            and len(path.relative_to(root).parts) != 3
        )
        and "# @package _global_" in path.read_text(encoding="utf-8")
        and re.search(r"(?m)^\s*policy:\s*", path.read_text(encoding="utf-8"))
    }
    if not names:
        raise LaunchPlanError(
            f"no native OpenPI/FastWAM launch configs found under {policy_infer_root}"
        )
    return sorted(names, key=str.casefold)


def build_supported_launch_contracts(
    *,
    config_dir: Path | None = None,
    device_env: str | None = None,
    policy_env: str | None = None,
) -> dict[str, Any]:
    """Compose every curated launch preset and return a CI-friendly report."""
    names = discover_supported_config_names(config_dir)
    entries: list[dict[str, Any]] = []
    for name in names:
        try:
            plan = build_launch_plan(
                name,
                config_dir=config_dir,
                device_env=device_env,
                policy_env=policy_env,
            )
        except LaunchPlanError as exc:
            entries.append(
                {
                    "config_name": name,
                    "status": "failed",
                    "errors": [str(exc)],
                    "plan": None,
                }
            )
            continue

        errors = list(plan.get("contract", {}).get("errors", []))
        entries.append(
            {
                "config_name": name,
                "status": "passed" if not errors else "failed",
                "errors": errors,
                "plan": plan,
            }
        )

    failed = sum(entry["status"] == "failed" for entry in entries)
    return {
        "mode": "read-only-supported-launch-contracts",
        "scope": "configs/policy_infer/openpi + configs/policy_infer/fastwam "
        "(native launch configs)",
        "started_processes": 0,
        "environment_resolution_required": device_env is None or policy_env is None,
        "summary": {
            "total": len(entries),
            "passed": len(entries) - failed,
            "failed": failed,
        },
        "configs": entries,
    }


def _render_operator_command(
    command: Mapping[str, Any],
    *,
    interpreter_placeholder: str,
) -> str:
    if command.get("execution_prefix_resolution_required") is not True:
        return str(command["shell"])
    argv = command.get("argv")
    if isinstance(argv, list) and argv[:1] == ["python"]:
        return interpreter_placeholder + " " + shlex.join(str(item) for item in argv[1:])
    return str(command["shell"])


def render_text(plan: Mapping[str, Any]) -> str:
    """Render the compact operator-facing form of a launch plan."""
    config = plan["config"]
    policy = plan["policy"]
    devices = plan["devices"]
    action = plan["action_executor"]
    commands = plan["commands"]
    lines = [
        "Read-only policy launch contract",
        "No process was started and no hardware was opened.",
        "",
        f"Hydra config: {config['hydra_config_name']} ({config['source_path']})",
        f"Policy: {policy['class'] or 'unconfigured'} -> {policy['runner_module'] or 'no registered runner'}",
        f"Contract: {'PASS' if plan.get('contract', {}).get('valid') else 'FAIL'}",
        (
            "Policy rates: "
            f"inference={policy['fps']!r}, control={policy['control_fps']!r}, "
            f"chunk_length={policy['chunk_length']!r}"
        ),
        f"Enabled devices: {len(devices['enabled'])}; disabled: {devices['disabled_count']}",
        (
            "ActionExecutor: "
            f"type={action['type']!r}, execution_fps={action['execution_fps']!r}, "
            f"chunk_manager={action['chunk_manager']!r}"
        ),
        "",
        "Resolve the device command prefix, then start devices from the repository root:",
        "  " + _render_operator_command(
            commands["device_starter"],
            interpreter_placeholder="<device-prefix>",
        ),
    ]
    server = commands.get("server")
    if server is not None:
        lines.extend(
            [
                "For FastWAM, start the model server and wait for its /healthz response:",
                "  "
                + _render_operator_command(
                    server,
                    interpreter_placeholder="<fastwam-server-prefix>",
                ),
            ]
        )
    consumer = commands["consumer"]
    if consumer is not None:
        lines.extend(
            [
                "After device readiness, resolve the policy command prefix and start it in a second terminal:",
                "  " + _render_operator_command(
                    consumer,
                    interpreter_placeholder="<policy-prefix>",
                ),
            ]
        )
    else:
        lines.append(
            "No consumer command was generated; resolve the policy runner warning first."
        )
    errors = plan.get("contract", {}).get("errors", [])
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in errors)
    return "\n".join(lines)


def render_supported_text(report: Mapping[str, Any]) -> str:
    """Render a compact batch report for humans and CI logs."""
    summary = report["summary"]
    lines = [
        "Read-only supported launch contract check",
        "No process was started and no hardware was opened.",
        f"Scope: {report['scope']}",
        (
            f"Summary: {summary['passed']} passed, {summary['failed']} failed, "
            f"{summary['total']} total"
        ),
    ]
    for entry in report["configs"]:
        plan = entry.get("plan") or {}
        policy = plan.get("policy") or {}
        lines.append(
            f"{'PASS' if entry['status'] == 'passed' else 'FAIL'} "
            f"{entry['config_name']}: "
            f"{policy.get('class') or 'unconfigured'} -> "
            f"{policy.get('runner_module') or 'no registered runner'}"
        )
        lines.extend(f"  - {error}" for error in entry.get("errors", []))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--config-name",
        help=(
            "Native Hydra name, e.g. "
            "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
        ),
    )
    selection.add_argument(
        "--all-supported",
        action="store_true",
        help="Check every native OpenPI launch preset and exit non-zero on failure",
    )
    parser.add_argument(
        "--device-env",
        default=None,
        help="Optional Conda environment for device_starter.py; omit for an interpreter-neutral plan",
    )
    parser.add_argument(
        "--policy-env",
        default=None,
        help="Optional Conda environment for the policy consumer; omit for an interpreter-neutral plan",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the full plan as JSON"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.all_supported:
            report = build_supported_launch_contracts(
                device_env=args.device_env,
                policy_env=args.policy_env,
            )
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(render_supported_text(report))
            return 0 if report["summary"]["failed"] == 0 else 1

        plan = build_launch_plan(
            args.config_name,
            device_env=args.device_env,
            policy_env=args.policy_env,
        )
    except LaunchPlanError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(plan))
    return 0 if plan["contract"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
