"""Hardware-free contracts for allowlisted articulated tool visuals."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest
from omegaconf import OmegaConf

sys.modules.setdefault("devices", types.SimpleNamespace(DEVICE_CLASSES={}))
import manager  # noqa: E402, F401
sys.modules.setdefault("visualizers", types.SimpleNamespace(VISUALIZER_CLASSES={}))

from incubator.control_panel.runtime_config import extract_robot_tools  # noqa: E402


UMI3_URL = "/static/assets/tools/umi3_camera/soft/umi3_gripper.urdf"
UMI3_VARIANTS = {
    "soft": {
        "kind": "urdf",
        "url": "/static/assets/tools/umi3_camera/soft/umi3_gripper.urdf",
    },
    "hard": {
        "kind": "urdf",
        "url": "/static/assets/tools/umi3_camera/hard/umi3_gripper.urdf",
    },
}
MAX_WIDTH_M = 0.0925319933860716


def _identity_mount() -> Dict[str, Any]:
    return {
        "frame": "flange",
        "translation": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _visual() -> Dict[str, Any]:
    return {
        "model": "umi3",
        "articulation": {
            "value_mode": "width_m",
            "arm_length_m": 0.06543,
            "max_width_m": MAX_WIDTH_M,
            "apex_angle_max_deg": 90.0,
        },
    }


def _dual_device(
    visual: Any = None, mounts: Any = None, *, tool_name: Any = ""
) -> Dict[str, Any]:
    gripper: Dict[str, Any] = {
        "name": "gripper",
        "tool_name": tool_name,
        "visual": _visual() if visual is None else visual,
        "visual_mount_transforms": {
            "left": _identity_mount(),
            "right": _identity_mount(),
        }
        if mounts is None
        else mounts,
    }
    return {
        "class": "RizonRobot",
        "device_class_id": 2,
        "robot_sn": ["Rizon4-012345", "Rizon4R-012346"],
        "data_shape": [44],
        "control_shape": [16],
        "gripper": gripper,
    }


def test_umi3_visual_is_allowlisted_per_side_without_fabricating_tool_pose() -> None:
    tools = extract_robot_tools({"devices": [_dual_device()]})

    for side in ("left", "right"):
        tool = tools[side]
        assert tool["name"] is None
        assert tool["status"] == "not-configured"
        assert tool["transform"] is None
        assert tool["reason"] == "rizon-tool-not-configured"
        visual = tool["visual"]
        assert visual == {
            "model": "umi3",
            "status": "available",
            "kind": "urdf",
            "url": UMI3_URL,
            "default_variant": "soft",
            "variants": UMI3_VARIANTS,
            "mirror_axis": "y" if side == "left" else None,
            "mount_transform": {
                "frame": "flange",
                "translation": [0.0, 0.0, 0.0],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "quaternion_order": "wxyz",
            },
            "articulation": {
                "joint_name": "gripper_joint",
                "joint_lower_rad": 0.0,
                "joint_upper_rad": math.pi / 2.0,
                "closed_joint_rad": 0.0,
                "value_mode": "width_m",
                "arm_length_m": 0.06543,
                "max_width_m": MAX_WIDTH_M,
                "apex_angle_max_deg": 90.0,
            },
            "source": "gripper.visual",
            "mount_source": f"gripper.visual_mount_transforms.{side}",
            "articulation_source": "gripper.visual.articulation",
            "reason": None,
        }


@pytest.mark.parametrize(
    ("visual", "reason"),
    [
        ({"model": "unknown", "articulation": _visual()["articulation"]},
         "tool-visual-model-is-not-allowlisted"),
        ({**_visual(), "url": "https://attacker.invalid/tool.urdf"},
         "tool-visual-config-is-invalid"),
        ({**_visual(), "model": "../umi3"},
         "tool-visual-model-is-not-allowlisted"),
    ],
)
def test_model_and_url_are_fail_closed_to_the_fixed_allowlist(
    visual: Dict[str, Any], reason: str
) -> None:
    payload = extract_robot_tools({"devices": [_dual_device(visual=visual)]})
    serialized = json.dumps(payload)
    for side in ("left", "right"):
        assert payload[side]["visual"]["status"] == "invalid-config"
        assert payload[side]["visual"]["url"] is None
        assert payload[side]["visual"]["reason"] == reason
    assert "attacker.invalid" not in serialized


def test_mounts_are_independent_and_require_exact_flange_schema() -> None:
    mounts = {
        "left": {
            "frame": "flange",
            "translation": [0.01, -0.02, 0.03],
            "quaternion_wxyz": [2.0, 0.0, 0.0, 0.0],
        },
        "right": {
            "frame": "tcp",
            "translation": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }
    tools = extract_robot_tools({"devices": [_dual_device(mounts=mounts)]})
    assert tools["left"]["visual"]["status"] == "available"
    assert tools["left"]["visual"]["mount_transform"]["translation"] == [
        0.01,
        -0.02,
        0.03,
    ]
    assert tools["right"]["visual"]["status"] == "invalid-config"
    assert tools["right"]["visual"]["mount_transform"] is None
    assert (
        tools["right"]["visual"]["reason"]
        == "tool-visual-mount-transform-is-invalid"
    )


@pytest.mark.parametrize(
    ("articulation", "reason"),
    [
        ({}, "tool-visual-articulation-fields-are-invalid"),
        ({
            "value_mode": "width_m",
            "arm_length_m": 0.06543,
            "max_width_m": MAX_WIDTH_M,
            "apex_angle_max_deg": 90.0,
            "url": "/static/attacker.urdf",
        }, "tool-visual-articulation-fields-are-invalid"),
        ({
            "value_mode": "raw_encoder",
            "arm_length_m": 0.06543,
            "max_width_m": MAX_WIDTH_M,
            "apex_angle_max_deg": 90.0,
        }, "tool-visual-value-mode-is-invalid"),
        ({
            "value_mode": "width_m",
            "arm_length_m": True,
            "max_width_m": MAX_WIDTH_M,
            "apex_angle_max_deg": 90.0,
        }, "tool-visual-articulation-values-are-invalid"),
        ({
            "value_mode": "width_m",
            "arm_length_m": 0.06543,
            "max_width_m": 0.5,
            "apex_angle_max_deg": 90.0,
        }, "tool-visual-calibration-geometry-is-inconsistent"),
    ],
)
def test_articulation_is_finite_ranged_and_geometry_consistent(
    articulation: Dict[str, Any], reason: str
) -> None:
    visual = {"model": "umi3", "articulation": articulation}
    tools = extract_robot_tools({"devices": [_dual_device(visual=visual)]})
    for side in ("left", "right"):
        assert tools[side]["visual"]["status"] == "invalid-config"
        assert tools[side]["visual"]["articulation"] is None
        assert tools[side]["visual"]["reason"] == reason


def test_side_specific_articulation_is_supported_without_cross_side_aliasing() -> None:
    left = _visual()["articulation"]
    right = {**left, "value_mode": "normalized"}
    visual = {"model": "umi3", "articulations": {"left": left, "right": right}}
    tools = extract_robot_tools({"devices": [_dual_device(visual=visual)]})
    assert tools["left"]["visual"]["articulation"]["value_mode"] == "width_m"
    assert tools["right"]["visual"]["articulation"]["value_mode"] == "normalized"
    assert tools["left"]["visual"]["articulation_source"].endswith(".left")
    assert tools["right"]["visual"]["articulation_source"].endswith(".right")


def test_shared_and_side_articulation_are_not_ambiguously_merged() -> None:
    visual = {
        "model": "umi3",
        "articulation": _visual()["articulation"],
        "articulations": {"left": _visual()["articulation"]},
    }
    tools = extract_robot_tools({"devices": [_dual_device(visual=visual)]})
    for side in ("left", "right"):
        assert tools[side]["visual"]["status"] == "invalid-config"
        assert (
            tools[side]["visual"]["reason"]
            == "tool-visual-articulation-fields-are-invalid"
        )


def test_staging_umi3_device_config_exports_the_fixed_visual_descriptor() -> None:
    path = (
            Path(__file__).resolve().parents[3]
        / "configs"
        / "devices"
        / "infer"
        / "infer_rizon_fisheye_umi3_dual_openpi.yaml"
    )
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert isinstance(config, dict)
    tools = extract_robot_tools(config)
    for side in ("left", "right"):
        assert tools[side]["status"] == "not-configured"
        assert tools[side]["transform"] is None
        assert tools[side]["visual"]["status"] == "available"
        assert tools[side]["visual"]["url"] == UMI3_URL


def test_legacy_soft_gripper_config_gets_builtin_umi3_visual_without_yaml_changes() -> None:
    path = (
            Path(__file__).resolve().parents[3]
        / "configs"
        / "devices"
        / "infer"
        / "infer_rizon_fisheye_umi3_dual_openpi_yam_head.yaml"
    )
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert isinstance(config, dict)
    rizon = next(
        device
        for device in config["devices"]
        if device.get("class") == "RizonRobot"
    )
    assert "visual" not in rizon["gripper"]

    tools = extract_robot_tools(config)
    for side in ("left", "right"):
        assert tools[side]["status"] == "not-configured"
        assert tools[side]["transform"] is None
        assert tools[side]["visual"]["status"] == "available"
        assert tools[side]["visual"]["url"] == UMI3_URL
        assert tools[side]["visual"]["mount_transform"]["frame"] == "flange"
        assert tools[side]["visual"]["source"] == "gripper.gripper_type"


def test_builtin_soft_gripper_visual_respects_per_side_enabled_state() -> None:
    device = _dual_device()
    device["gripper"].pop("visual")
    device["gripper"].pop("visual_mount_transforms")
    device["gripper"].update(
        {
            "gripper_type": "soft",
            "enabled": [True, False],
            "value_mode": "width_m",
        }
    )

    tools = extract_robot_tools({"devices": [device]})
    assert tools["left"]["visual"]["status"] == "available"
    assert "visual" not in tools["right"]
