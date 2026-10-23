"""Hardware-free coverage for active Elements tool resolution."""

from __future__ import annotations

import sys
import types
from typing import Any, Dict

sys.modules.setdefault("devices", types.SimpleNamespace(DEVICE_CLASSES={}))
import manager  # noqa: E402, F401
sys.modules.setdefault("visualizers", types.SimpleNamespace(VISUALIZER_CLASSES={}))

from incubator.control_panel.runtime_config import extract_robot_tools  # noqa: E402


def _dual_device(
    *,
    tool_name: Any = "",
    elements: Any = None,
    tool_transform: Any = None,
) -> Dict[str, Any]:
    gripper: Dict[str, Any] = {"name": "gripper", "tool_name": tool_name}
    if elements is not None:
        gripper["elements"] = elements
    if tool_transform is not None:
        gripper["tool_transform"] = tool_transform
    return {
        "class": "RizonRobot",
        "device_class_id": 2,
        "robot_sn": ["Rizon4-012345", "Rizon4R-012346"],
        "data_shape": [44],
        "control_shape": [16],
        "gripper": gripper,
    }


def _tool_element(name: str, *, side: str | None = None) -> Dict[str, Any]:
    element: Dict[str, Any] = {
        "type": "end_effector",
        "activate": True,
        "name": name,
        "tcp_location": [0.01, -0.02, 0.31, 1.0, 0.0, 0.0, 0.0],
    }
    if side is not None:
        element["side"] = side
    return element


def test_explicit_non_empty_tool_name_wins_over_active_elements() -> None:
    tools = extract_robot_tools(
        {
            "devices": [
                _dual_device(
                    tool_name="explicit_tcp",
                    elements=[_tool_element("elements_tcp")],
                    tool_transform=[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                )
            ]
        }
    )
    for side in ("left", "right"):
        assert tools[side]["resolved_tool_name"] == "explicit_tcp"
        assert tools[side]["resolved_tool_source"] == "gripper.tool_name"
        assert tools[side]["resolution"] == "explicit-tool-name"
        assert tools[side]["active_element_count"] == 0


def test_blank_tool_name_falls_back_to_side_specific_active_tool_elements() -> None:
    tools = extract_robot_tools(
        {
            "devices": [
                _dual_device(
                    elements=[
                        {"type": "robot", "activate": True, "name": "do-not-use"},
                        {"type": "camera", "activate": True, "name": "also-not-a-tool"},
                        _tool_element("right_tcp", side="right"),
                        _tool_element("left_tcp", side="left"),
                    ]
                )
            ]
        }
    )
    assert tools["left"]["resolved_tool_name"] == "left_tcp"
    assert tools["right"]["resolved_tool_name"] == "right_tcp"
    for side in ("left", "right"):
        assert tools[side]["status"] == "available"
        assert tools[side]["resolution"] == "elements-active-tool"
        assert tools[side]["resolved_tool_source"].startswith("gripper.elements[")
        assert tools[side]["active_element_count"] == 1


def test_top_level_name_keyed_elements_and_active_alias_are_supported() -> None:
    tools = extract_robot_tools(
        {
            "elements": {
                "umi3_tcp": {
                    "class": "Tool",
                    "active": True,
                    "tcp": {
                        "tcp_position": [0.0, 0.0, 0.25],
                        "tcp_orientation": [1.0, 0.0, 0.0, 0.0],
                    },
                }
            },
            "devices": [_dual_device(tool_name=None)],
        }
    )
    assert tools["left"]["resolved_tool_name"] == "umi3_tcp"
    assert tools["left"]["resolved_tool_source"] == "elements.umi3_tcp"
    assert tools["left"]["transform"]["translation"] == [0.0, 0.0, 0.25]


def test_multiple_active_tools_choose_deterministically_and_report_warning() -> None:
    tools = extract_robot_tools(
        {
            "devices": [
                _dual_device(
                    elements=[
                        _tool_element("z_tool"),
                        _tool_element("a_tool"),
                    ]
                )
            ]
        }
    )
    for side in ("left", "right"):
        assert tools[side]["resolved_tool_name"] == "a_tool"
        assert tools[side]["active_element_count"] == 2
        assert "multiple active tool elements" in tools[side]["resolution_warning"]
        assert "selected a_tool deterministically" in tools[side]["resolution_warning"]


def test_zero_active_tool_elements_is_none_without_error_or_wrong_type_guess() -> None:
    tools = extract_robot_tools(
        {
            "elements": [
                {"type": "camera", "activate": True, "name": "wrist"},
                {"type": "robot", "activate": True, "name": "rizon"},
                {"type": "tool", "activate": False, "name": "inactive_tool"},
            ],
            "devices": [_dual_device()],
        }
    )
    for side in ("left", "right"):
        assert tools[side]["resolved_tool_name"] is None
        assert tools[side]["resolved_tool_source"] is None
        assert tools[side]["resolution"] == "none"
        assert tools[side]["status"] == "not-configured"
        assert tools[side]["transform"] is None


def test_active_tool_asset_is_sanitized_for_same_origin_viewer_loading() -> None:
    element = _tool_element("umi3_tcp")
    element["visual_asset"] = {
        "url": "/static/assets/tools/umi3_tcp.obj",
        "material_url": "/static/assets/tools/umi3_tcp.mtl",
        "scale": [1.0, 1.0, 1.0],
    }
    tools = extract_robot_tools({"devices": [_dual_device(elements=[element])]})
    assert tools["left"]["asset"] == {
        "kind": "obj",
        "url": "/static/assets/tools/umi3_tcp.obj",
        "material_url": "/static/assets/tools/umi3_tcp.mtl",
        "scale": [1.0, 1.0, 1.0],
        "source": "gripper.elements[0].visual_asset",
    }


def test_external_tool_asset_url_is_not_exposed_but_tool_resolution_survives() -> None:
    element = _tool_element("umi3_tcp")
    element["visual_asset"] = {"url": "https://example.invalid/tool.obj"}
    tools = extract_robot_tools({"devices": [_dual_device(elements=[element])]})
    assert tools["left"]["resolved_tool_name"] == "umi3_tcp"
    assert tools["left"]["status"] == "available"
    assert tools["left"]["asset"] is None
    assert "same-origin .obj or .urdf" in tools["left"]["resolution_warning"]
