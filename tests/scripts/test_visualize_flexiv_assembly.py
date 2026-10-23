from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.visualization import visualize_flexiv_assembly as viewer
from scripts.visualization.visualize_flexiv_assembly import (
    _html,
    _page_payload,
    _variant,
)


def test_variant_from_supported_serials() -> None:
    assert _variant("Rizon 4-063256") == "Rizon4"
    assert _variant("Rizon 4R-062088") == "Rizon4R"


def test_direct_script_execution_adds_repository_root_to_module_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts/visualization/visualize_flexiv_assembly.py"
    code = (
        "import runpy, sys; "
        f"root = {str(repo_root)!r}; "
        "sys.path = [item for item in sys.path if item != root]; "
        f"runpy.run_path({str(script)!r}, run_name='viewer_import_test'); "
        "import devices"
    )

    subprocess.run([sys.executable, "-c", code], check=True, cwd=repo_root.parent)


def test_port_conflict_fails_before_reading_robot_state(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        monkeypatch.setattr(viewer, "cleanup_before_startup", lambda _role: SimpleNamespace(candidates=[]))
        monkeypatch.setattr(viewer, "_page_payload", lambda args: pytest.fail("RDK state read before port bind"))

        with pytest.raises(OSError, match="Address already in use"):
            viewer.main(["left", "right", "--gripper-type", "umi3_camera", "--port", str(port)])


def test_offline_payload_contains_frames_and_models() -> None:
    args = argparse.Namespace(
        left_robot_sn="Rizon 4-063256",
        right_robot_sn="Rizon 4R-062088",
        gripper_type="umi3_camera",
        tool_transform=None,
        offline=True,
    )
    payload = _page_payload(args)
    assert [arm["variant"] for arm in payload["arms"]] == ["Rizon4", "Rizon4R"]
    assert all(len(arm["flange_to_tool"]) == 7 for arm in payload["arms"])
    assert all(arm["urdf"].startswith("/assets/robot/rizon/") for arm in payload["arms"])
    assert payload["gripper_urdf"].endswith("/soft/umi3_gripper.urdf")
    assert payload["gripper_type"] == "umi3_camera"

    html = _html(payload)
    assert 'querySelector("child")?.getAttribute("link")' in html
    assert "meshes/${match[1]}" in html
    assert "meshes/${spec.variant}/${match[1]}" not in html
    assert 'mesh.getAttribute("scale")||"1 1 1"' in html
    assert "object.scale.set(...scale)" in html
    assert "verified-umi3-lk-gripper" in html
    assert "await buildGripper(flange,index)" in html
    assert "BoxGeometry" not in html
    assert "index===0?-.72:.72" in html
    assert "spec.side.toUpperCase()" in html
    assert "rootLink=[...links].find" in html
    assert "await addRobotVisuals(rootLink,arm,index)" in html
    assert "camera.up.set(0,0,1)" in html
    assert "grid.rotation.x=Math.PI/2" in html
    assert "controls.minDistance=0.5" in html
    assert "controls.maxDistance=6" in html
    assert "controls.enableZoom=false" in html
    assert 'canvas.addEventListener("wheel"' in html
    assert "Math.max(-120,Math.min(120,event.deltaY*unit))" in html
    assert "wheelLocked=false" in html
    assert "wheelUnlock=setTimeout" in html
    assert "Math.abs(normalized)/120,1)*.30" in html
    assert "Math.exp(step)" in html
    assert "controls.enablePan=true" in html
    assert "controls.screenSpacePanning=true" in html
    assert "controls.listenToKeyEvents(window)" in html
