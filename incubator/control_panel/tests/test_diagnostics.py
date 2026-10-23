"""Contract tests for Control Panel diagnostics and preflight projections."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pytest

from incubator.control_panel.diagnostics import (
    PREFLIGHT_SCHEMA_VERSION,
    RUNTIME_METRICS_SCHEMA_VERSION,
    RUNTIME_TUNING_SCHEMA_VERSION,
    build_preflight_report,
    build_runtime_metrics,
    build_runtime_tuning_schema,
)


def _summary(
    *,
    camera_slots: Mapping[str, Any] | None = None,
    robot_slots: Mapping[str, Any] | None = None,
    gripper_slots: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": "fixture",
        "valid": True,
        "errors": [],
        "metadata": {
            "workflow_kind": "openpi",
            "policy_class": "OpenPiUmiPolicy",
            "hardware_required": True,
            "control_panel_supported": True,
        },
        "control_panel_supported": True,
        "hardware_required": True,
        "camera_slots": dict(camera_slots or {}),
        "robot_slots": dict(robot_slots or {}),
        "gripper_slots": dict(gripper_slots or {}),
        "execution_profile": dict(
            profile
            or {
                "policy": {
                    "class": "OpenPiUmiPolicy",
                    "fps": 4.0,
                    "control_fps": 30.0,
                    "chunk_length": 48,
                },
                "action_executor": {
                    "type": "ActionExecutor",
                    "execution_fps": 30.0,
                    "chunk_length": 48,
                    "chunk_manager": "base_manager",
                },
                "device_manager": {"type": "DeviceManager"},
            }
        ),
    }


def _stopped_workflow(*, blockers: list[str] | None = None) -> dict[str, Any]:
    return {
        "runtime": {
            "status": "stopped",
            "active": False,
            "ready": False,
            "blockers": list(blockers or []),
            "device_manager": {"running": False, "ready": False},
            "action_executor": {"running": False, "ready": False},
        },
        "inference": {
            "status": "stopped",
            "running": False,
            "blockers": [],
        },
    }


def _clean_preflight_inputs() -> dict[str, Any]:
    return {
        "grippers": {"sides": {}},
        "discovery": {"warnings": [], "cameras": [], "robots": []},
        "workflow": _stopped_workflow(),
        "devices": [],
    }


def _check(report: Mapping[str, Any], check_id: str) -> Mapping[str, Any]:
    return next(item for item in report["checks"] if item["id"] == check_id)


def test_preflight_does_not_block_before_devices_have_started() -> None:
    """Stopped devices are expected before startup, not an idle-state fault."""

    inputs = _clean_preflight_inputs()
    inputs["workflow"] = _stopped_workflow(blockers=["devices not ready: RizonRobot:0"])
    report = build_preflight_report(_summary(), **inputs)

    assert report["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert _check(report, "workflow_idle")["status"] == "warning"
    assert report["can_start"] is True


def test_runtime_not_ready_inference_dependency_is_a_readiness_warning() -> None:
    inputs = _clean_preflight_inputs()
    workflow = _stopped_workflow()
    workflow["inference"]["blockers"] = [
        "请先正确启动 DeviceManager 与 ActionExecutor"
    ]
    report = build_preflight_report(_summary(), **{**inputs, "workflow": workflow})

    assert _check(report, "workflow_idle")["status"] == "warning"
    assert report["can_start_runtime"] is True


def test_preflight_separates_runtime_and_inference_readiness() -> None:
    inputs = _clean_preflight_inputs()
    workflow = _stopped_workflow()
    workflow["inference"]["blockers"] = [
        "请先正确启动 DeviceManager 与 ActionExecutor",
        "Checkpoint does not exist: /missing",
    ]
    report = build_preflight_report(_summary(), **{**inputs, "workflow": workflow})

    assert report["can_start_runtime"] is True
    assert report["can_start_inference"] is False
    assert report["stages"]["runtime"]["status"] == "warning"
    assert report["stages"]["inference"]["status"] == "blocked"
    assert "Checkpoint does not exist: /missing" in report["stages"]["inference"]["blockers"]


def test_preflight_allows_inference_when_runtime_is_ready() -> None:
    inputs = _clean_preflight_inputs()
    workflow = _stopped_workflow()
    workflow["runtime"].update(
        {
            "status": "running",
            "active": True,
            "ready": True,
        }
    )
    workflow["inference"]["blockers"] = []
    report = build_preflight_report(_summary(), **{**inputs, "workflow": workflow})

    assert report["can_start_runtime"] is False
    assert report["can_start_inference"] is True
    assert report["stages"]["runtime"]["status"] == "blocked"
    assert report["stages"]["inference"]["status"] == "ready"
    assert "当前 runtime/inference 仍在运行，请先停止" not in report["stages"]["inference"]["blockers"]


@pytest.mark.parametrize(
    ("devices", "expected_status", "can_start"),
    [
        ([], "blocking", False),
        (
            [{"key": "RizonRobot:0", "status": "stopped", "ready": False}],
            "warning",
            True,
        ),
        (
            [{"key": "RizonRobot:0", "status": "degraded", "ready": False}],
            "warning",
            True,
        ),
        (
            [{"key": "RizonRobot:0", "status": "running", "ready": True}],
            "ok",
            True,
        ),
    ],
)
def test_preflight_requires_required_hardware_to_be_live_and_ready(
    devices: list[dict[str, Any]], expected_status: str, can_start: bool
) -> None:
    robot_slots = {
        "rizon": {
            "role": "rizon",
            "device_key": "RizonRobot:0",
            "configured": True,
            "required": True,
        }
    }
    inputs = _clean_preflight_inputs()
    inputs["devices"] = devices
    report = build_preflight_report(_summary(robot_slots=robot_slots), **inputs)

    robot_check = _check(report, "robot_slots")
    assert robot_check["status"] == expected_status
    assert report["can_start"] is can_start


def test_preflight_detects_missing_physical_camera_binding() -> None:
    camera_slots = {
        "left": {
            "role": "left",
            "device_key": "WebCameraOpenPi:0",
            "configured": True,
            "required": True,
        }
    }
    inputs = _clean_preflight_inputs()
    inputs["discovery"] = {
        "warnings": [],
        "cameras": [],
        "robots": [],
        "configured_detection": {
            "WebCameraOpenPi:0": {
                "kind": "camera",
                "configured_device_class_id": 0,
                "detected": False,
            }
        },
    }
    report = build_preflight_report(_summary(camera_slots=camera_slots), **inputs)

    assert _check(report, "camera_slots")["status"] == "blocking"
    assert report["can_start"] is False


def test_preflight_blocks_missing_gripper_calibration_artifact() -> None:
    inputs = _clean_preflight_inputs()
    inputs["grippers"] = {
        "sides": {
            "left": {
                "port": "/dev/ttyUSB0",
                "motor_id": 1,
                "port_present": True,
                "calibrated": True,
                "calibration_artifact": {
                    "present": False,
                    "status": "missing",
                },
            }
        }
    }
    report = build_preflight_report(
        _summary(
            gripper_slots={
                "left": {
                    "side": "left",
                    "configured": True,
                    "port": "/dev/ttyUSB0",
                    "motor_id": 1,
                }
            }
        ),
        **inputs,
    )

    gripper_check = _check(report, "gripper_calibration")
    assert gripper_check["status"] == "blocking"
    assert "left:missing" in gripper_check["evidence"]["invalid_artifacts"]


def test_preflight_blocks_unverified_gripper_artifact() -> None:
    inputs = _clean_preflight_inputs()
    inputs["grippers"] = {
        "sides": {
            "left": {
                "port": "/dev/ttyUSB0",
                "motor_id": 1,
                "port_present": True,
                "calibrated": False,
                "calibration_artifact": {
                    "present": True,
                    "valid": False,
                    "status": "unverified",
                    "identity_status": "unverified",
                },
            }
        }
    }
    report = build_preflight_report(
        _summary(
            gripper_slots={
                "left": {
                    "side": "left",
                    "configured": True,
                    "port": "/dev/ttyUSB0",
                    "motor_id": 1,
                    "motor_id_valid": True,
                }
            }
        ),
        **inputs,
    )

    gripper_check = _check(report, "gripper_calibration")
    assert gripper_check["status"] == "blocking"
    assert "left:unverified" in gripper_check["evidence"]["invalid_artifacts"]


def test_preflight_blocks_invalid_configured_gripper_motor_id() -> None:
    inputs = _clean_preflight_inputs()
    inputs["grippers"] = {
        "sides": {
            "left": {
                "port": "/dev/ttyUSB0",
                "motor_id": 1,
                "port_present": True,
                "calibrated": True,
                "calibration_artifact": {
                    "present": True,
                    "valid": True,
                    "status": "valid",
                    "identity_status": "verified",
                },
            }
        }
    }
    report = build_preflight_report(
        _summary(
            gripper_slots={
                "left": {
                    "side": "left",
                    "configured": True,
                    "port": "/dev/ttyUSB0",
                    "motor_id": None,
                    "motor_id_valid": False,
                }
            }
        ),
        **inputs,
    )

    gripper_check = _check(report, "gripper_calibration")
    assert gripper_check["status"] == "blocking"
    assert gripper_check["evidence"]["invalid_config_identity"] == [
        "left:motor_id"
    ]


def test_preflight_rejects_invalid_values_from_redacted_runtime_summary() -> None:
    summary = _summary(
        profile={
            "invalid_fields": ["policy.fps"],
            "policy": {
                "class": "OpenPiUmiPolicy",
                "fps": None,
                "control_fps": 30.0,
                "chunk_length": 48,
            },
            "action_executor": {},
        }
    )
    report = build_preflight_report(
        summary,
        **_clean_preflight_inputs(),
    )
    check = _check(report, "execution_profile")
    assert check["status"] == "blocking"
    assert check["evidence"]["issues"] == ["policy.fps"]


def test_preflight_rejects_failed_workflow_stage_without_error_text() -> None:
    workflow = _stopped_workflow()
    workflow["runtime"]["status"] = "failed"
    workflow["runtime"]["error"] = None
    report = build_preflight_report(
        _summary(),
        **{**_clean_preflight_inputs(), "workflow": workflow},
    )
    assert report["stages"]["runtime"]["status"] == "blocked"
    assert report["stages"]["runtime"]["can_start"] is False


@pytest.mark.parametrize("field", ["fps", "control_fps"])
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_preflight_rejects_non_finite_execution_profile_values(
    field: str,
    bad_value: float,
) -> None:
    profile = {
        "policy": {
            "class": "OpenPiUmiPolicy",
            "fps": 4.0,
            "control_fps": 30.0,
            "chunk_length": 48,
        },
        "action_executor": {
            "type": "ActionExecutor",
            "execution_fps": 30.0,
            "chunk_length": 48,
            "chunk_manager": "base_manager",
        },
    }
    profile["policy"][field] = bad_value
    inputs = _clean_preflight_inputs()
    report = build_preflight_report(_summary(profile=profile), **inputs)

    assert _check(report, "execution_profile")["status"] == "blocking"


def test_runtime_metrics_projects_manager_and_device_fields() -> None:
    summary = _summary(
        profile={
            "policy": {
                "class": "OpenPiUmiPolicy",
                "fps": 4.0,
                "control_fps": 30.0,
                "chunk_length": 48,
                "checkpoint_configured": True,
            },
            "action_executor": {
                "type": "ActionExecutor",
                "execution_fps": 30.0,
                "chunk_length": 48,
                "chunk_manager": "auto_manager_v2",
            },
            "device_manager": {"type": "DeviceManager"},
        }
    )
    state = {
        "selected_config": "fixture",
        "workflow": {
            "kind": "openpi",
            "runtime": {
                "status": "running",
                "device_manager": {"running": True, "ready": True, "pid": 11},
                "action_executor": {"running": True, "ready": True, "pid": 12},
            },
            "inference": {
                "status": "running",
                "running": True,
                "pid": 13,
            },
        },
        "checkpoint": {
            "valid": True,
            "exists": True,
            "override_active": False,
        },
        "devices": [
            {
                "key": "WebCameraOpenPi:0",
                "class": "WebCameraOpenPi",
                "id": 0,
                "status": "running",
                "fps": 30.0,
                "shm": {
                    "connected": True,
                    "fresh": True,
                    "fps": 29.5,
                    "age_ms": 8.0,
                    "frames": 120,
                    "shape": [480, 640, 3],
                    "dtype": "uint8",
                },
            }
        ],
    }

    metrics = build_runtime_metrics(state, summary)

    assert metrics["schema_version"] == RUNTIME_METRICS_SCHEMA_VERSION
    assert metrics["policy"]["configured_fps"] == 4.0
    assert metrics["policy"]["control_fps"] == 30.0
    assert metrics["policy"]["chunk_length"] == 48
    assert metrics["policy"]["checkpoint_configured"] is True
    assert metrics["action_executor"]["manager_type"] == "auto_manager_v2"
    assert metrics["action_executor"]["configured_fps"] == 30.0
    assert metrics["action_executor"]["running"] is True
    assert metrics["device_manager"]["running"] is True
    assert metrics["devices"][0]["reported_fps"] == 29.5
    assert metrics["devices"][0]["shm_fps"] == 29.5
    assert metrics["devices"][0]["actual_fps"] is None
    assert metrics["devices"][0]["measurement_kind"] == "producer_reported"
    assert metrics["devices"][0]["frame_age_ms"] == 8.0
    assert metrics["devices"][0]["fresh"] is True


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_runtime_metrics_never_exports_non_finite_policy_numbers(
    bad_value: float,
) -> None:
    profile = {"policy": {"fps": bad_value, "control_fps": 30.0, "chunk_length": 48}}
    metrics = build_runtime_metrics(
        {"workflow": {}, "devices": []},
        {"execution_profile": profile},
    )

    assert metrics["policy"]["configured_fps"] is None
    assert metrics["policy"]["configured_fps"] is not bad_value


def test_runtime_tuning_schema_exposes_restart_scope_and_effective_values() -> None:
    summary = _summary()

    schema = build_runtime_tuning_schema(summary)
    controls = {item["key"]: item for item in schema["controls"]}

    assert RUNTIME_TUNING_SCHEMA_VERSION == 2
    assert schema["schema_version"] == 2
    assert schema["config_name"] == "fixture"
    assert schema["capabilities"]["read_only"] is False
    assert schema["capabilities"]["apply_supported"] is True
    assert "overlay" in schema["capabilities"]["reason"]
    assert controls["policy.fps"]["value"] == 4.0
    assert controls["policy.fps"]["apply_mode"] == "inference_restart"
    assert controls["policy.control_fps"]["value"] == 30.0
    assert controls["policy.chunk_length"]["value"] == 48
    assert "action_executor.execution_fps" not in controls
    assert "device_manager.target_fps" not in controls
    assert all(item["writable"] is True for item in controls.values())
