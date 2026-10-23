"""Safe, structured diagnostics for the Control Panel.

The diagnostics layer consumes already-sanitized config and lifecycle snapshots.
It never opens a device, attaches to shared memory, or mutates a configuration.
That makes the same report useful in the browser, tests, and a future standalone
preflight command.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


PREFLIGHT_SCHEMA_VERSION = 1
RUNTIME_METRICS_SCHEMA_VERSION = 1
RUNTIME_TUNING_SCHEMA_VERSION = 2
_TERMINAL_DEVICE_FAILURES = frozenset(
    {"failed", "conflict", "orphaned", "stop_timeout", "external"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_list(value: Any, *, limit: int = 16) -> list[str]:
    if isinstance(value, str):
        return [value[:512]] if value else []
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text[:512])
        if len(result) >= limit:
            break
    return result


def _safe_number(value: Any) -> Optional[float | int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(float(value)) else None
    except (OverflowError, ValueError):
        return None


def _device_index(devices: Any) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(devices, (list, tuple)):
        return {}
    result: Dict[str, Mapping[str, Any]] = {}
    for item in devices:
        if not isinstance(item, Mapping):
            continue
        key = item.get("key")
        if key is None:
            key = (
                f"{item.get('class', item.get('device_class', 'unknown'))}:"
                f"{item.get('id', item.get('device_class_id', 'unknown'))}"
            )
        result[str(key)] = item
    return result


def _required_slots(summary: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    raw = summary.get(key)
    if isinstance(raw, Mapping):
        values: Iterable[Any] = raw.values()
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = ()
    return [
        item
        for item in values
        if isinstance(item, Mapping)
        and bool(item.get("required", item.get("inference_required", False)))
    ]


def _check(
    check_id: str,
    label: str,
    status: str,
    *,
    evidence: Optional[Mapping[str, Any]] = None,
    remediation: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = (
        status
        if status in {"ok", "warning", "blocking", "unknown", "not_required"}
        else "unknown"
    )
    severity = {
        "blocking": "blocking",
        "warning": "warning",
        "unknown": "warning",
        "ok": "info",
        "not_required": "info",
    }[normalized]
    result: Dict[str, Any] = {
        "id": check_id,
        "label": label,
        "status": normalized,
        "severity": severity,
    }
    if evidence:
        result["evidence"] = dict(evidence)
    if remediation:
        result["remediation"] = str(remediation)[:512]
    return result


def _merged_metadata(summary: Mapping[str, Any]) -> Dict[str, Any]:
    nested = _as_mapping(summary.get("metadata"))
    metadata = dict(nested)
    metadata.update(
        {
            key: value
            for key, value in summary.items()
            if key != "metadata"
            and key
            in {
                "workflow_kind",
                "policy_class",
                "hardware_required",
                "status",
                "control_panel_supported",
                "support_reason",
            }
        }
    )
    return metadata


def _positive_number(value: Any) -> Optional[float | int]:
    number = _safe_number(value)
    return number if number is not None and float(number) > 0 else None


def _safe_integer(
    value: Any, *, minimum: int = 1, maximum: int = 4096
) -> Optional[int]:
    number = _safe_number(value)
    if number is None or int(number) != number:
        return None
    integer = int(number)
    return integer if minimum <= integer <= maximum else None


def _is_readiness_blocker(value: str) -> bool:
    text = str(value or "").strip().casefold()
    return any(
        marker.casefold() in text
        for marker in (
            "设备未就绪",
            "devices not ready",
            "runtime dependency lost",
            "devices are not ready",
            "device is not ready",
            "请先正确启动运行时",
            "请先正确启动 devicemanager",
            "device manager and actionexecutor",
            "runtime is not ready",
            "inference control is not ready",
            "start runtime",
        )
    )


def _workflow_hard_blockers(
    runtime: Mapping[str, Any],
    inference: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    readiness: list[str] = []
    explicit = runtime.get("preparation_blockers")
    if isinstance(explicit, (list, tuple)):
        hard.extend(_text_list(explicit))
    for raw in _text_list(runtime.get("blockers")) + _text_list(
        inference.get("blockers")
    ):
        if _is_readiness_blocker(raw):
            readiness.append(raw)
        elif raw not in hard:
            hard.append(raw)
    for raw in _text_list(runtime.get("error")) + _text_list(inference.get("error")):
        if raw not in hard:
            hard.append(raw)
    return list(dict.fromkeys(hard)), list(dict.fromkeys(readiness))


def _detection_state(
    key: Optional[str],
    kind: str,
    discovery: Mapping[str, Any],
    devices: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    live = devices.get(str(key)) if key else None
    configured = _as_mapping(discovery.get("configured_detection")).get(str(key))
    configured = configured if isinstance(configured, Mapping) else None
    if configured is not None and "detected" in configured:
        detected = configured.get("detected")
        if isinstance(detected, (list, tuple)):
            return (
                "present" if any(item is True for item in detected) else "missing",
                live,
                configured,
            )
        if isinstance(detected, bool):
            return ("present" if detected else "missing"), live, configured
    if live is not None:
        status = str(live.get("status") or "").casefold()
        if status in _TERMINAL_DEVICE_FAILURES:
            return "missing", live, configured
        return "present", live, configured
    items = discovery.get("cameras" if kind == "camera" else "robots")
    if kind == "robot" and isinstance(items, (list, tuple)) and key:
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if str(key) in {
                str(item.get("id")),
                str(item.get("rdk_sn")),
                str(item.get("serial_number")),
            }:
                return "present", live, configured
    if isinstance(items, (list, tuple)):
        # A completed passive scan is authoritative about absence. Keep the
        # unknown state only for callers that did not provide a scan at all.
        return "missing", live, configured
    return "unknown", live, configured


def _hardware_slot_check(
    summary: Mapping[str, Any],
    *,
    discovery: Optional[Mapping[str, Any]],
    devices: Mapping[str, Mapping[str, Any]],
    slot_key: str,
    label: str,
    kind: str,
    hardware_required: bool,
) -> Dict[str, Any]:
    required = _required_slots(summary, slot_key)
    if not required:
        return _check(slot_key, label, "not_required", evidence={"count": 0})
    discovery_map = _as_mapping(discovery)
    missing_config: list[str] = []
    missing_physical: list[str] = []
    unknown: list[str] = []
    unhealthy: list[str] = []
    not_ready: list[str] = []
    observed: list[Dict[str, Any]] = []
    for slot in required:
        role = str(slot.get("role") or kind)
        key = slot.get("device_key")
        if not slot.get("configured") or not key:
            missing_config.append(role)
            observed.append({"role": role, "state": "unconfigured"})
            continue
        state, live, configured = _detection_state(
            str(key), kind, discovery_map, devices
        )
        status = str((live or {}).get("status") or "").casefold()
        if state == "missing":
            missing_physical.append(role)
        elif state == "unknown":
            unknown.append(role)
        if status in _TERMINAL_DEVICE_FAILURES:
            unhealthy.append(f"{role}:{status}")
        elif live is not None and live.get("ready") is not True:
            not_ready.append(role)
        observed.append(
            {
                "role": role,
                "device_key": str(key),
                "state": state,
                "live_status": live.get("status") if live else None,
                "configured_detected": (
                    configured.get("detected") if configured else None
                ),
            }
        )
    evidence = {
        "count": len(required),
        "missing_config": missing_config,
        "missing_physical": missing_physical,
        "unknown": unknown,
        "unhealthy": unhealthy,
        "not_ready": not_ready,
        "observed": observed,
    }
    if missing_config or missing_physical or unhealthy:
        status = "blocking"
    elif unknown and hardware_required:
        status = "unknown"
    elif not_ready:
        status = "warning"
    else:
        status = "ok"
    remediation = None
    if status in {"blocking", "unknown"}:
        remediation = "确认硬件 ID/绑定并重新扫描；未确认物理身份前不要启动。"
    elif status == "warning":
        remediation = "设备尚未就绪；启动运行时后会按配置自动准备必需设备。"
    return _check(
        slot_key,
        label,
        status,
        evidence=evidence,
        remediation=remediation,
    )


def _execution_profile_check(
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    profile = _as_mapping(summary.get("execution_profile"))
    policy = _as_mapping(profile.get("policy"))
    executor = _as_mapping(profile.get("action_executor"))
    manager = _as_mapping(profile.get("device_manager"))
    values = {
        "policy_fps": _positive_number(policy.get("fps")),
        "control_fps": _positive_number(policy.get("control_fps")),
        "chunk_length": _safe_integer(policy.get("chunk_length")),
        "action_executor_fps": _positive_number(executor.get("execution_fps")),
        "device_manager_fps": _positive_number(manager.get("target_fps")),
    }
    issues: list[str] = _text_list(profile.get("invalid_fields"))
    policy_class = metadata.get("policy_class")
    if policy_class and str(policy_class).casefold() not in {
        "none",
        "null",
        "disabled",
    }:
        for field, raw_key in (
            ("policy_fps", "fps"),
            ("control_fps", "control_fps"),
        ):
            if policy.get(raw_key) is not None and values[field] is None:
                issues.append(f"{field} must be a finite positive number")
        if policy.get("chunk_length") is not None and values["chunk_length"] is None:
            issues.append("chunk_length must be a positive integer")
    for raw, source, field in (
        (executor, "execution_fps", "action_executor_fps"),
        (manager, "target_fps", "device_manager_fps"),
    ):
        kind = str(raw.get("type") or "").casefold()
        if kind not in {"", "none", "null", "disabled"} and raw.get(source) is not None:
            if values[field] is None:
                issues.append(f"{field} must be a finite positive number")
    if issues:
        return _check(
            "execution_profile",
            "执行参数",
            "blocking",
            evidence={"values": values, "issues": issues},
            remediation="修复 FPS、chunk length 或 manager 参数后再启动。",
        )
    if not any(value is not None for value in values.values()):
        return _check(
            "execution_profile",
            "执行参数",
            "not_required",
            evidence={"values": values},
        )
    return _check("execution_profile", "执行参数", "ok", evidence={"values": values})


def _gripper_check(
    summary: Mapping[str, Any],
    grippers: Optional[Mapping[str, Any]],
    *,
    hardware_required: bool,
) -> Dict[str, Any]:
    expected = {
        str(side): item
        for side, item in _as_mapping(summary.get("gripper_slots")).items()
        if isinstance(item, Mapping) and item.get("configured") is True
    }
    if not expected:
        return _check(
            "gripper_calibration",
            "夹爪标定",
            "not_required",
            evidence={"sides": []},
        )
    if grippers is None:
        return _check(
            "gripper_calibration",
            "夹爪标定",
            "unknown" if hardware_required else "warning",
            evidence={"expected_sides": sorted(expected)},
            remediation="读取夹爪状态并确认 calibration artifact。",
        )
    side_map = _as_mapping(grippers.get("sides"))
    missing: list[str] = []
    invalid_config_identity: list[str] = []
    identity_mismatch: list[str] = []
    missing_ports: list[str] = []
    invalid_artifacts: list[str] = []
    unverified: list[str] = []
    errors: list[str] = []
    observed: Dict[str, Any] = {}
    for side, contract in expected.items():
        expected_port = contract.get("port")
        expected_motor = contract.get("motor_id")
        invalid_identity_fields: list[str] = []
        if not isinstance(expected_port, str) or not expected_port.strip():
            invalid_identity_fields.append("port")
        if (
            contract.get("motor_id_valid") is False
            or isinstance(expected_motor, bool)
            or not isinstance(expected_motor, int)
            or not 0 <= expected_motor <= 255
        ):
            invalid_identity_fields.append("motor_id")
        if invalid_identity_fields:
            invalid_config_identity.append(
                f"{side}:{','.join(invalid_identity_fields)}"
            )
        item = _as_mapping(side_map.get(side))
        if not item:
            missing.append(side)
            continue
        actual_port = item.get("port")
        actual_motor = item.get("motor_id")
        mismatches: list[str] = []
        if expected_port and actual_port and str(expected_port) != str(actual_port):
            mismatches.append("port")
        if expected_motor is not None and actual_motor is not None:
            if (
                isinstance(actual_motor, bool)
                or not isinstance(actual_motor, int)
                or actual_motor != expected_motor
            ):
                mismatches.append("motor_id")
        if mismatches:
            identity_mismatch.append(f"{side}:{','.join(mismatches)}")
        if not actual_port or item.get("port_present") is False:
            missing_ports.append(side)
        artifact = _as_mapping(item.get("calibration_artifact"))
        artifact_status = str(artifact.get("status") or "").casefold()
        artifact_present = artifact.get("present")
        artifact_valid = artifact.get("valid")
        artifact_identity = str(artifact.get("identity_status") or "").casefold()
        artifact_missing = not artifact or artifact_present is False
        artifact_safe = (
            artifact_status == "valid"
            and artifact_valid is True
            and artifact_identity == "verified"
            and item.get("calibrated") is True
        )
        artifact_invalid = not artifact_safe or artifact_missing
        if artifact_status == "unverified":
            unverified.append(side)
        if artifact_invalid:
            invalid_artifacts.append(f"{side}:{artifact_status or 'invalid'}")
        if str(item.get("status") or "").casefold() in _TERMINAL_DEVICE_FAILURES:
            errors.append(f"{side}:{item.get('status')}")
        observed[side] = {
            "port": actual_port,
            "motor_id": actual_motor,
            "port_present": item.get("port_present"),
            "artifact": {
                "status": artifact.get("status"),
                "identity_status": artifact.get("identity_status"),
                "name": artifact.get("name"),
            },
        }
    if grippers.get("last_error"):
        errors.append(str(grippers.get("last_error"))[:256])
    evidence = {
        "expected_sides": sorted(expected),
        "missing": missing,
        "invalid_config_identity": invalid_config_identity,
        "identity_mismatch": identity_mismatch,
        "missing_ports": missing_ports,
        "invalid_artifacts": invalid_artifacts,
        # Keep a concise alias for older client-side diagnostics consumers.
        "uncalibrated": list(invalid_artifacts),
        "unverified": unverified,
        "errors": errors,
        "observed": observed,
    }
    if (
        missing
        or invalid_config_identity
        or identity_mismatch
        or missing_ports
        or invalid_artifacts
        or errors
    ):
        status = "blocking"
    else:
        status = "ok"
    return _check(
        "gripper_calibration",
        "夹爪标定",
        status,
        evidence=evidence,
        remediation=(
            "先确认夹爪 ID/串口并执行一键标定；artifact 必须包含匹配的身份字段。"
            if status == "blocking"
            else None
        ),
    )


def _workflow_stage_requirements(
    workflow: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Separate runtime-preparation blockers from inference prerequisites."""
    if workflow is None:
        return {
            "known": False,
            "runtime": {
                "hard": [],
                "readiness": ["workflow state unavailable"],
                "active": False,
                "ready": False,
            },
            "inference": {
                "hard": [],
                "readiness": ["workflow state unavailable"],
                "active": False,
                "ready": False,
            },
        }

    workflow_map = _as_mapping(workflow)
    runtime = _as_mapping(workflow_map.get("runtime"))
    inference = _as_mapping(workflow_map.get("inference"))
    if not runtime and not inference:
        return {
            "known": False,
            "runtime": {
                "hard": [],
                "readiness": ["workflow state unavailable"],
                "active": False,
                "ready": False,
            },
            "inference": {
                "hard": [],
                "readiness": ["workflow state unavailable"],
                "active": False,
                "ready": False,
            },
        }
    active_statuses = {"starting", "running", "stopping"}

    def collect(section: Mapping[str, Any]) -> Dict[str, Any]:
        hard: list[str] = []
        readiness: list[str] = []
        for raw in _text_list(section.get("preparation_blockers")) + _text_list(
            section.get("blockers")
        ):
            if _is_readiness_blocker(raw):
                readiness.append(raw)
            elif raw not in hard:
                hard.append(raw)
        error = section.get("error")
        if error:
            hard.append(str(error)[:512])
        status = str(section.get("status") or "").casefold()
        if status in {"failed", "error", "conflict", "orphaned"} and not error:
            hard.append(f"workflow stage status is {status}")
        return {
            "hard": list(dict.fromkeys(hard)),
            "readiness": list(dict.fromkeys(readiness)),
            "active": bool(
                section.get("active")
                or section.get("running")
                or section.get("status") in active_statuses
            ),
            "ready": section.get("ready") is True,
        }

    runtime_requirements = collect(runtime)
    inference_requirements = collect(inference)
    if not runtime_requirements["ready"]:
        runtime_requirements["readiness"].append(
            "runtime 尚未 ready"
        )
        inference_requirements["readiness"].append(
            "请先正确启动 DeviceManager 与 ActionExecutor"
        )
    if runtime_requirements["active"] or inference_requirements["active"]:
        runtime_requirements["hard"].append(
            "当前 runtime/inference 仍在运行，请先停止"
        )
    if inference_requirements["active"]:
        inference_requirements["hard"].append(
            "Inference 已在运行，请先停止"
        )
    for requirements in (runtime_requirements, inference_requirements):
        requirements["hard"] = list(dict.fromkeys(requirements["hard"]))
        requirements["readiness"] = list(
            dict.fromkeys(requirements["readiness"])
        )
    return {
        "known": True,
        "runtime": runtime_requirements,
        "inference": inference_requirements,
    }


def build_preflight_report(
    config_summary: Optional[Mapping[str, Any]],
    *,
    discovery: Optional[Mapping[str, Any]] = None,
    grippers: Optional[Mapping[str, Any]] = None,
    workflow: Optional[Mapping[str, Any]] = None,
    devices: Optional[Sequence[Mapping[str, Any]]] = None,
    discovery_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a fail-closed, read-only startup checklist."""
    summary = _as_mapping(config_summary)
    metadata = _merged_metadata(summary)
    config_name = str(summary.get("name") or "")
    hardware_required = metadata.get("hardware_required") is True
    checks: list[Dict[str, Any]] = []

    if not summary:
        checks.append(
            _check(
                "config",
                "运行配置",
                "blocking",
                remediation="先选择一个可用的运行配置。",
            )
        )
    else:
        errors = _text_list(summary.get("errors"))
        if errors or summary.get("valid") is False:
            checks.append(
                _check(
                    "config",
                    "运行配置",
                    "blocking",
                    evidence={"errors": errors or ["configuration is invalid"]},
                    remediation="修复配置校验错误后重新选择。",
                )
            )
        elif metadata.get("control_panel_supported") is False:
            checks.append(
                _check(
                    "config",
                    "运行配置",
                    "blocking",
                    evidence={
                        "workflow_kind": metadata.get("workflow_kind"),
                        "policy_class": metadata.get("policy_class"),
                    },
                    remediation="选择面板已集成的 workflow，或先注册对应 launcher。",
                )
            )
        else:
            checks.append(
                _check(
                    "config",
                    "运行配置",
                    "ok",
                    evidence={
                        "name": config_name,
                        "workflow_kind": metadata.get("workflow_kind"),
                        "policy_class": metadata.get("policy_class"),
                    },
                )
            )

    checks.append(_execution_profile_check(summary, metadata))
    devices_index = _device_index(devices)
    checks.append(
        _hardware_slot_check(
            summary,
            discovery=discovery,
            devices=devices_index,
            slot_key="camera_slots",
            label="必需相机",
            kind="camera",
            hardware_required=hardware_required,
        )
    )
    checks.append(
        _hardware_slot_check(
            summary,
            discovery=discovery,
            devices=devices_index,
            slot_key="robot_slots",
            label="必需机械臂",
            kind="robot",
            hardware_required=hardware_required,
        )
    )
    checks.append(
        _gripper_check(summary, grippers, hardware_required=hardware_required)
    )

    if workflow is None:
        checks.append(
            _check(
                "workflow_idle",
                "启动前空闲",
                "unknown",
                remediation="读取最新运行时状态后再启动。",
            )
        )
    else:
        workflow_map = _as_mapping(workflow)
        runtime = _as_mapping(workflow_map.get("runtime"))
        inference = _as_mapping(workflow_map.get("inference"))
        active = bool(
            runtime.get("active")
            or inference.get("running")
            or runtime.get("status") in {"starting", "running", "stopping"}
            or inference.get("status") in {"starting", "running", "stopping"}
        )
        hard, readiness = _workflow_hard_blockers(runtime, inference)
        if active:
            checks.append(
                _check(
                    "workflow_idle",
                    "启动前空闲",
                    "blocking",
                    evidence={
                        "runtime_status": runtime.get("status"),
                        "inference_status": inference.get("status"),
                        "blockers": hard,
                    },
                    remediation="先停止当前 runtime/inference，再执行配置或标定。",
                )
            )
        elif hard:
            checks.append(
                _check(
                    "workflow_idle",
                    "启动前空闲",
                    "blocking",
                    evidence={"blockers": hard, "readiness": readiness},
                    remediation="按照 blocker 提示完成 checkpoint、环境或进程检查。",
                )
            )
        elif readiness:
            checks.append(
                _check(
                    "workflow_idle",
                    "启动前空闲",
                    "warning",
                    evidence={"readiness": readiness},
                    remediation="设备尚未就绪；点击启动运行时后会自动准备必需设备。",
                )
            )
        else:
            checks.append(_check("workflow_idle", "启动前空闲", "ok"))

    if discovery_error:
        checks.append(
            _check(
                "hardware_discovery",
                "硬件发现",
                "blocking" if hardware_required else "unknown",
                evidence={"error": str(discovery_error)[:512]},
                remediation="重新扫描硬件；未确认身份前不要启动运动设备。",
            )
        )
    elif discovery is None:
        checks.append(
            _check(
                "hardware_discovery",
                "硬件发现",
                "unknown" if hardware_required else "not_required",
                remediation=(
                    "执行一次硬件扫描以确认物理身份。" if hardware_required else None
                ),
            )
        )
    else:
        warnings = _text_list(discovery.get("warnings"))
        checks.append(
            _check(
                "hardware_discovery",
                "硬件发现",
                "warning" if warnings else "ok",
                evidence={
                    "camera_count": len(discovery.get("cameras") or []),
                    "robot_count": len(discovery.get("robots") or []),
                    "warnings": warnings,
                    "revision": discovery.get("revision"),
                },
                remediation="核对发现结果与机器 profile。" if warnings else None,
            )
        )

    statuses = [str(item.get("status")) for item in checks]
    blocking_count = statuses.count("blocking")
    warning_count = statuses.count("warning")
    unknown_count = statuses.count("unknown")
    if blocking_count:
        overall = "blocked"
    elif unknown_count and hardware_required:
        overall = "unknown"
    elif warning_count or unknown_count:
        overall = "warning"
    else:
        overall = "ready"

    # Config/hardware checks are common to both stages. The aggregate workflow
    # row intentionally stays conservative, while stage results distinguish a
    # runtime that can prepare devices from inference that still needs a model.
    advisory_check_ids = {"camera_slots", "robot_slots", "hardware_discovery"}
    common_checks = [
        item
        for item in checks
        if item.get("id") not in {"workflow_idle", *advisory_check_ids}
    ]
    common_blocking = any(
        item.get("status") == "blocking" for item in common_checks
    )
    common_unknown = any(item.get("status") == "unknown" for item in common_checks)
    common_can_start = not common_blocking and not (
        common_unknown and hardware_required
    )
    stage_requirements = _workflow_stage_requirements(workflow)
    advisory_checks = [
        {
            "id": item.get("id"),
            "status": item.get("status"),
            "label": item.get("label"),
        }
        for item in checks
        if item.get("id") in advisory_check_ids
        and item.get("status") in {"blocking", "unknown", "warning"}
    ]

    def stage_result(stage: str) -> Dict[str, Any]:
        requirements = _as_mapping(stage_requirements.get(stage))
        hard = _text_list(requirements.get("hard"))
        readiness = _text_list(requirements.get("readiness"))
        known = stage_requirements.get("known") is True
        ready = (
            requirements.get("ready") is True
            if stage == "runtime"
            else _as_mapping(stage_requirements.get("runtime")).get("ready") is True
        )
        can_start = bool(
            common_can_start
            and known
            and not hard
            and (stage != "inference" or ready)
        )
        if not known or hard or (stage == "inference" and not ready):
            status = "blocked"
        elif readiness or not ready:
            status = "warning"
        else:
            status = "ready"
        return {
            "status": status,
            "can_start": can_start,
            "blockers": hard,
            "readiness": readiness,
            "advisories": list(advisory_checks),
        }

    stages = {
        "runtime": stage_result("runtime"),
        "inference": stage_result("inference"),
    }
    # Preserve the original aggregate alias for existing clients: it remains
    # conservative and includes physical-discovery blockers. New callers that
    # need lifecycle-stage semantics should use the explicit stage aliases.
    can_start = blocking_count == 0 and not (
        unknown_count and hardware_required
    )
    next_actions: list[str] = []
    for item in checks:
        if item.get("status") in {"blocking", "unknown", "warning"} and item.get(
            "remediation"
        ):
            action = str(item["remediation"])
            if action not in next_actions:
                next_actions.append(action)
    for stage in stages.values():
        for action in _text_list(stage.get("blockers")) + _text_list(
            stage.get("readiness")
        ):
            if action not in next_actions:
                next_actions.append(action)
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "config_name": config_name or None,
        "overall": overall,
        "can_start": can_start,
        "can_start_runtime": stages["runtime"]["can_start"],
        "can_start_inference": stages["inference"]["can_start"],
        "stages": stages,
        "summary": {
            "total": len(checks),
            "blocking": blocking_count,
            "warning": warning_count,
            "unknown": unknown_count,
        },
        "checks": checks,
        "next_actions": next_actions[:8],
    }


def _metric_mapping(*values: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in values:
        if isinstance(value, Mapping):
            merged.update(value)
    return merged


def _metric_number(mapping: Mapping[str, Any], *keys: str) -> Optional[float | int]:
    for key in keys:
        value = _safe_number(mapping.get(key))
        if value is not None:
            return value
    return None


def build_runtime_metrics(
    state: Mapping[str, Any],
    config_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Project lifecycle state into a compact, allowlisted telemetry contract."""
    snapshot = _as_mapping(state)
    summary = _as_mapping(config_summary)
    profile = _as_mapping(summary.get("execution_profile"))
    if not profile:
        profile = _as_mapping(
            _as_mapping(snapshot.get("effective_config")).get("execution_profile")
        )
    policy_profile = _as_mapping(profile.get("policy"))
    executor_profile = _as_mapping(profile.get("action_executor"))
    manager_profile = _as_mapping(profile.get("device_manager"))
    workflow = _as_mapping(snapshot.get("workflow"))
    runtime = _as_mapping(workflow.get("runtime"))
    inference = _as_mapping(workflow.get("inference"))
    runtime_executor = _as_mapping(runtime.get("action_executor"))
    runtime_manager = _as_mapping(runtime.get("device_manager"))
    policy_telemetry = _metric_mapping(
        inference.get("metrics"),
        inference.get("telemetry"),
        snapshot.get("policy_metrics"),
    )
    executor_telemetry = _metric_mapping(
        runtime_executor.get("metrics"),
        runtime_executor.get("telemetry"),
        snapshot.get("action_executor_metrics"),
    )
    manager_telemetry = _metric_mapping(
        runtime_manager.get("metrics"),
        runtime_manager.get("telemetry"),
        snapshot.get("device_manager_metrics"),
    )
    checkpoint = _as_mapping(snapshot.get("checkpoint"))
    if not checkpoint:
        checkpoint = _as_mapping(inference.get("checkpoint"))
    checkpoint_valid = checkpoint.get("valid") is True
    checkpoint_exists = checkpoint.get("exists") is True
    profile_checkpoint_configured = policy_profile.get("checkpoint_configured") is True

    devices_payload: list[Dict[str, Any]] = []
    raw_devices = snapshot.get("devices")
    if isinstance(raw_devices, (list, tuple)):
        for raw in raw_devices:
            if not isinstance(raw, Mapping):
                continue
            shm = _as_mapping(raw.get("shm"))
            shm_fps = (
                _safe_number(shm.get("fps")) if shm.get("connected") is True else None
            )
            devices_payload.append(
                {
                    "key": raw.get("key"),
                    "class": raw.get("class", raw.get("device_class")),
                    "id": raw.get("id", raw.get("device_class_id")),
                    "status": raw.get("status"),
                    "configured_fps": _safe_number(raw.get("fps")),
                    # SHM fps is producer-reported, not a local loop measure.
                    "reported_fps": shm_fps,
                    "shm_fps": shm_fps,
                    "actual_fps": None,
                    "measurement_kind": (
                        "producer_reported" if shm_fps is not None else None
                    ),
                    "frame_age_ms": _safe_number(shm.get("age_ms")),
                    "frames": shm.get("frames"),
                    "connected": shm.get("connected") is True,
                    "fresh": shm.get("fresh") is True,
                    "shape": shm.get("shape", raw.get("shape")),
                    "dtype": shm.get("dtype", raw.get("dtype")),
                }
            )

    executor_type = executor_profile.get("type")
    raw_chunk_manager = executor_profile.get("chunk_manager")
    profile_chunk_manager_options = _as_mapping(
        executor_profile.get("chunk_manager_options")
    )
    if isinstance(raw_chunk_manager, Mapping):
        chunk_manager = raw_chunk_manager.get("type")
        profile_chunk_manager_options = _metric_mapping(
            raw_chunk_manager,
            profile_chunk_manager_options,
        )
    else:
        chunk_manager = raw_chunk_manager
    chunk_manager = str(chunk_manager) if chunk_manager else None
    return {
        "schema_version": RUNTIME_METRICS_SCHEMA_VERSION,
        "sampled_at": _utc_now(),
        "selected_config": snapshot.get("selected_config"),
        "workflow_kind": workflow.get("kind"),
        "policy": {
            "class": policy_profile.get("class"),
            "status": inference.get("status"),
            "running": inference.get("running") is True,
            "pid": inference.get("pid"),
            "configured_fps": _safe_number(policy_profile.get("fps")),
            "control_fps": _safe_number(policy_profile.get("control_fps")),
            "chunk_length": policy_profile.get("chunk_length"),
            "checkpoint_configured": bool(
                profile_checkpoint_configured
                or checkpoint_valid
                or checkpoint_exists
                or checkpoint.get("override_active") is True
            ),
            "checkpoint_valid": checkpoint_valid,
            "checkpoint_override_active": checkpoint.get("override_active") is True,
            "actual_fps": _metric_number(policy_telemetry, "actual_fps", "loop_fps"),
            "inference_latency_ms": _metric_number(
                policy_telemetry, "inference_latency_ms", "latency_ms"
            ),
            "queue_depth": _metric_number(policy_telemetry, "queue_depth"),
        },
        "action_executor": {
            "status": runtime.get("status"),
            "running": runtime_executor.get("running") is True,
            "ready": runtime_executor.get("ready") is True,
            "pid": runtime_executor.get("pid"),
            "executor_type": executor_type,
            # Keep manager_type for old clients, but make it the chunk manager.
            "manager_type": chunk_manager,
            "chunk_manager": chunk_manager,
            "chunk_manager_options": dict(profile_chunk_manager_options),
            "configured_fps": _safe_number(executor_profile.get("execution_fps")),
            "chunk_length": executor_profile.get("chunk_length"),
            "queue_depth": _metric_number(
                executor_telemetry, "queue_depth", "queue_size"
            ),
            "chunks_received": _metric_number(executor_telemetry, "chunks_received"),
            "chunks_dropped": _metric_number(executor_telemetry, "chunks_dropped"),
            "dispatch_latency_ms": _metric_number(
                executor_telemetry, "dispatch_latency_ms", "latency_ms"
            ),
            "actual_fps": _metric_number(executor_telemetry, "actual_fps", "loop_fps"),
        },
        "device_manager": {
            "status": runtime.get("status"),
            "running": runtime_manager.get("running") is True,
            "ready": runtime_manager.get("ready") is True,
            "pid": runtime_manager.get("pid"),
            "type": manager_profile.get("type"),
            "target_fps": _safe_number(manager_profile.get("target_fps")),
            "actual_fps": _metric_number(manager_telemetry, "actual_fps", "loop_fps"),
        },
        "devices": devices_payload,
        "availability": {
            "measured": [
                "device_shm_fps",
                "device_shm_age",
                "device_shm_frames",
            ],
            "exported_when_reported": [
                "policy_actual_fps",
                "policy_inference_latency_ms",
                "action_executor_queue_depth",
                "action_executor_dispatch_latency_ms",
                "device_manager_actual_loop_fps",
            ],
            "not_measured": [
                "device_actual_consumer_fps",
            ],
        },
    }


def _tuning_control(
    key: str,
    label: str,
    value: Any,
    *,
    minimum: float | int,
    maximum: float | int,
    step: float | int,
    unit: str,
    apply_mode: str,
    section_enabled: bool,
) -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "value": value,
        "min": minimum,
        "max": maximum,
        "step": step,
        "unit": unit,
        "apply_mode": apply_mode,
        "section_enabled": section_enabled,
        "bounds_source": "ui_safe_default",
        "writable": section_enabled,
    }


def build_runtime_tuning_schema(
    config_summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Describe allowlisted, restart-aware knobs without arbitrary YAML paths."""
    summary = _as_mapping(config_summary)
    profile = _as_mapping(summary.get("execution_profile"))
    policy = _as_mapping(profile.get("policy"))
    policy_enabled = bool(policy.get("class"))
    controls = [
        _tuning_control(
            "policy.fps",
            "Policy inference FPS",
            _safe_number(policy.get("fps")),
            minimum=0.1,
            maximum=30.0,
            step=0.1,
            unit="Hz",
            apply_mode="inference_restart",
            section_enabled=policy_enabled,
        ),
        _tuning_control(
            "policy.control_fps",
            "Policy control FPS",
            _safe_number(policy.get("control_fps")),
            minimum=1.0,
            maximum=120.0,
            step=1.0,
            unit="Hz",
            apply_mode="runtime_and_inference_restart",
            section_enabled=policy_enabled,
        ),
        _tuning_control(
            "policy.chunk_length",
            "Action chunk length",
            _safe_integer(policy.get("chunk_length")),
            minimum=1,
            maximum=512,
            step=1,
            unit="steps",
            apply_mode="inference_restart",
            section_enabled=policy_enabled,
        ),
    ]
    return {
        "schema_version": RUNTIME_TUNING_SCHEMA_VERSION,
        "config_name": summary.get("name"),
        "capabilities": {
            "read_only": False,
            "apply_supported": True,
            "reason": (
                "调参可保存到当前 config，也可生成隔离 overlay；"
                "修改将在对应运行时重启后生效。"
            ),
        },
        "controls": controls,
    }


__all__ = [
    "PREFLIGHT_SCHEMA_VERSION",
    "RUNTIME_METRICS_SCHEMA_VERSION",
    "RUNTIME_TUNING_SCHEMA_VERSION",
    "build_preflight_report",
    "build_runtime_metrics",
    "build_runtime_tuning_schema",
]
