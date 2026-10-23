#!/usr/bin/env python3
"""Show two Flexiv Rizon arms, flange/tool frames, and an LK gripper.

The viewer is deliberately independent of the control panel.  It reads the
current joint state and Flexiv ``tcp_location`` when ``flexivrdk`` is
available, then serves a small local Three.js page.  ``--offline`` (or a
missing SDK) uses a nominal pose so the model can still be inspected on a
development machine.

Example::

    python scripts/visualization/visualize_flexiv_assembly.py \
        "Rizon 4-063256" "Rizon 4R-062088" --gripper-type umi3_camera

The viewer uses the verified UMI3 soft-finger CAD assets bundled with the
control panel. The tool frame remains the exact RDK-reported flange-to-tool
transform whenever hardware is available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from incubator.control_panel.process_cleanup import (  # noqa: E402
    ProcessCleanupError,
    cleanup_before_startup,
    force_cleanup_requested,
)
NOMINAL_JOINTS = [0.0, -0.42, 0.0, 1.05, 0.0, 0.6, 0.0]
IDENTITY_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display Flexiv Rizon assembly geometry and flange/tool frames."
    )
    parser.add_argument("left_robot_sn", help="Flexiv serial number for the left arm")
    parser.add_argument("right_robot_sn", help="Flexiv serial number for the right arm")
    parser.add_argument(
        "gripper_type_positional",
        nargs="?",
        choices=("umi3_camera",),
        help="Optional positional spelling of --gripper-type.",
    )
    parser.add_argument(
        "--gripper-type",
        "--gripper_type",
        default=None,
        choices=("umi3_camera",),
        help="Tool model to display (currently only umi3_camera).",
    )
    parser.add_argument(
        "--tool-transform",
        nargs=7,
        type=float,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Override flange-to-tool pose [m, quaternion wxyz]. Otherwise use RDK tcp_location.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not connect to RDK; use nominal joints and identity tool transforms.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Serve the page without opening a browser.")
    parser.add_argument("--port", type=int, default=0, help="Local HTTP port (0 chooses a free port).")
    parser.add_argument(
        "--force-process-cleanup",
        action="store_true",
        help="Force-kill related launcher processes after graceful cleanup fails.",
    )
    return parser


def _variant(serial: str) -> str:
    normalized = str(serial).casefold().replace(" ", "")
    if "4r" in normalized:
        return "Rizon4R"
    if "4" in normalized:
        return "Rizon4"
    raise ValueError(
        f"Cannot infer a supported Rizon model from serial {serial!r}; "
        "use a Rizon 4 or Rizon 4R serial."
    )


def _finite_pose(value: Any) -> list[float] | None:
    try:
        pose = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(pose) != 7 or not all(math.isfinite(item) for item in pose):
        return None
    norm = math.sqrt(sum(item * item for item in pose[3:]))
    if norm <= 1e-12:
        return None
    pose[3:] = [item / norm for item in pose[3:]]
    return pose


def _read_rdk_state(serial: str) -> tuple[list[float], list[float] | None, str | None]:
    """Read state without enabling or commanding the robot."""
    try:
        from devices.flexiv_adapter import FlexivAdapter

        adapter = FlexivAdapter(backend="python")
        robot = adapter.create_robot(serial)
        state = robot.states()
        joints = [float(item) for item in state.q]
        if len(joints) != 7 or not all(math.isfinite(item) for item in joints):
            raise RuntimeError("RDK returned an invalid seven-joint state")
        tool = adapter.create_tool(robot)
        raw = getattr(tool.params(), "tcp_location", None)
        return joints, _finite_pose(raw), None
    except Exception as exc:  # Hardware/SDK availability is intentionally optional.
        return list(NOMINAL_JOINTS), None, str(exc)


def _page_payload(args: argparse.Namespace) -> dict[str, Any]:
    gripper_type = (
        args.gripper_type
        or getattr(args, "gripper_type_positional", None)
        or "umi3_camera"
    )
    arms: list[dict[str, Any]] = []
    for side, serial in (("left", args.left_robot_sn), ("right", args.right_robot_sn)):
        variant = _variant(serial)
        warning = None
        joints = list(NOMINAL_JOINTS)
        rdk_tool = None
        if not args.offline:
            joints, rdk_tool, warning = _read_rdk_state(serial)
        tool_pose = _finite_pose(args.tool_transform) if args.tool_transform else rdk_tool
        if tool_pose is None:
            tool_pose = list(IDENTITY_POSE)
            warning = warning or "RDK tcp_location unavailable; using identity flange-to-tool transform"
        arms.append(
            {
                "side": side,
                "serial": serial,
                "variant": variant,
                "urdf": f"/assets/robot/rizon/flexiv_{variant}_kinematics.urdf",
                "joints": joints,
                "flange_to_tool": tool_pose,
                "warning": warning,
            }
        )
    return {
        "gripper_type": gripper_type,
        "gripper_urdf": "/assets/tools/umi3_camera/soft/umi3_gripper.urdf",
        "arms": arms,
    }


def _html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flexiv assembly viewer</title><style>
html,body{margin:0;height:100%;overflow:hidden;background:#101419;color:#e7edf2;font:14px system-ui,sans-serif}
#scene{width:100vw;height:100vh;display:block}#hud{position:fixed;left:16px;top:12px;max-width:560px;padding:12px 14px;background:#182027e8;border:1px solid #3c4c56;border-radius:6px;line-height:1.45}
.warning{color:#ffc266}.legend{color:#9ed8ff}
</style></head><body><canvas id="scene"></canvas><div id="hud"></div>
<script type="module">
import * as THREE from "/vendor/three.module.min.js";
import {OrbitControls} from "/vendor/OrbitControls.js";
import {OBJLoader} from "/vendor/OBJLoader.js";
const DATA=__PAYLOAD__;
const canvas=document.querySelector("#scene"), renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(innerWidth,innerHeight);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x101419);
const camera=new THREE.PerspectiveCamera(42,innerWidth/innerHeight,.01,20); camera.up.set(0,0,1);camera.position.set(2.1,1.8,2.2);
const controls=new OrbitControls(camera,canvas);controls.target.set(.35,0,.45);controls.minDistance=0.5;controls.maxDistance=6;controls.enableZoom=false;controls.enablePan=true;controls.screenSpacePanning=true;controls.panSpeed=.8;controls.keyPanSpeed=18;controls.listenToKeyEvents(window);controls.enableDamping=true;controls.dampingFactor=.08;controls.update();
let wheelLocked=false,wheelUnlock=0;canvas.addEventListener("wheel",event=>{event.preventDefault();clearTimeout(wheelUnlock);wheelUnlock=setTimeout(()=>{wheelLocked=false},150);if(wheelLocked)return;wheelLocked=true;const unit=event.deltaMode===1?16:event.deltaMode===2?100:1,normalized=Math.max(-120,Math.min(120,event.deltaY*unit)),step=Math.sign(normalized)*Math.min(Math.abs(normalized)/120,1)*.30,offset=camera.position.clone().sub(controls.target),distance=Math.max(controls.minDistance,Math.min(controls.maxDistance,offset.length()*Math.exp(step)));camera.position.copy(controls.target).add(offset.setLength(distance));controls.update();},{passive:false});
scene.add(new THREE.HemisphereLight(0xddeeff,0x202830,2.2)); const key=new THREE.DirectionalLight(0xffffff,2.0); key.position.set(2,2,3); scene.add(key);
const grid=new THREE.GridHelper(2.8,28,0x46606c,0x26343b);grid.rotation.x=Math.PI/2;scene.add(grid);scene.add(new THREE.AxesHelper(.35));
const loader=new OBJLoader(), hud=document.querySelector("#hud"), rows=[];
function rpy(r){const q=new THREE.Quaternion(); q.setFromEuler(new THREE.Euler(r[0],r[1],r[2],"XYZ")); return q;}
function pose(v){const g=new THREE.Group(); g.position.set(v[0],v[1],v[2]); g.quaternion.set(v[4],v[5],v[6],v[3]); return g;}
function axis(size=.12){return new THREE.AxesHelper(size);}
function label(text,color){
  const c=document.createElement("canvas");c.width=768;c.height=112;const x=c.getContext("2d");x.fillStyle="#101820e8";x.fillRect(0,0,c.width,c.height);x.strokeStyle=color;x.lineWidth=6;x.strokeRect(3,3,c.width-6,c.height-6);x.fillStyle="#fff";x.font="bold 38px system-ui";x.textAlign="center";x.textBaseline="middle";x.fillText(text,c.width/2,c.height/2);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));s.scale.set(.72,.105,1);s.renderOrder=10;return s;
}
async function stl(url){
  const b=await(await fetch(url)).arrayBuffer(),v=new DataView(b),n=v.getUint32(80,true),p=new Float32Array(n*9),norm=new Float32Array(n*9);let o=84;
  for(let i=0;i<n;i++){const nx=v.getFloat32(o,true),ny=v.getFloat32(o+4,true),nz=v.getFloat32(o+8,true);o+=12;for(let q=0;q<3;q++){const k=i*9+q*3;p[k]=v.getFloat32(o,true);p[k+1]=v.getFloat32(o+4,true);p[k+2]=v.getFloat32(o+8,true);norm[k]=nx;norm[k+1]=ny;norm[k+2]=nz;o+=12;}o+=2;}
  const g=new THREE.BufferGeometry();g.setAttribute("position",new THREE.BufferAttribute(p,3));g.setAttribute("normal",new THREE.BufferAttribute(norm,3));return g;
}
async function buildGripper(flange,index){
  const xml=await(await fetch(DATA.gripper_urdf)).text(),doc=new DOMParser().parseFromString(xml,"application/xml"),root=new THREE.Group();root.name="verified-umi3-lk-gripper";flange.add(root);
  const links=[...doc.querySelectorAll("link")],groups=new Map();for(const link of links){const g=new THREE.Group();groups.set(link.getAttribute("name"),g);}
  root.add(groups.get("base_link"));const joint=doc.querySelector("joint"),child=groups.get(joint.querySelector("child").getAttribute("link")),parent=groups.get(joint.querySelector("parent").getAttribute("link")),jo=joint.querySelector("origin"),jxyz=(jo?.getAttribute("xyz")||"0 0 0").split(/\\s+/).map(Number),jr=(jo?.getAttribute("rpy")||"0 0 0").split(/\\s+/).map(Number);child.position.set(...jxyz);child.quaternion.copy(rpy(jr));parent.add(child);
  for(const link of links){const visual=link.querySelector(":scope > visual"),mesh=visual?.querySelector("mesh");if(!mesh)continue;const filename=mesh.getAttribute("filename"),scale=(mesh.getAttribute("scale")||"1 1 1").split(/\\s+/).map(Number),geometry=await stl(`/assets/tools/umi3_camera/soft/${filename}`),material=new THREE.MeshStandardMaterial({color:link.getAttribute("name")==="base_link"?0x343b42:0x16a3ad,roughness:.68}),object=new THREE.Mesh(geometry,material);object.scale.set(...scale);groups.get(link.getAttribute("name")).add(object);}
  if(index===0)root.scale.y=-1;
}
async function addRobotVisuals(link,group,index){
  if(!link)return;for(const visual of link.querySelectorAll(":scope > visual")){const mesh=visual.querySelector("mesh"),file=mesh?.getAttribute("filename");if(!file)continue;const match=file.match(/meshes\\/(.+)$/);if(!match)continue;const object=await new Promise((resolve,reject)=>loader.load(`/assets/robot/rizon/meshes/${match[1]}`,resolve,undefined,reject)),vo=visual.querySelector("origin"),vxyz=(vo?.getAttribute("xyz")||"0 0 0").split(/\\s+/).map(Number),vr=(vo?.getAttribute("rpy")||"0 0 0").split(/\\s+/).map(Number),scale=(mesh.getAttribute("scale")||"1 1 1").split(/\\s+/).map(Number);object.position.set(...vxyz);object.quaternion.copy(rpy(vr));object.scale.set(...scale);object.traverse(n=>{if(n.isMesh)n.material=new THREE.MeshStandardMaterial({color:index===0?0x2aa6b0:0xe09a4c,roughness:.7});});group.add(object);}
}
async function buildArm(spec,index){
  const arm=new THREE.Group();arm.name=`arm-${spec.side}`;arm.position.set(0,index===0?-.72:.72,0);scene.add(arm);const tag=label(`${spec.side.toUpperCase()} · ${spec.serial}`,index===0?"#2aa6b0":"#e09a4c");tag.position.set(0,0,.24);arm.add(tag);
  const xml=await (await fetch(spec.urdf)).text(), doc=new DOMParser().parseFromString(xml,"application/xml");
  const joints=[...doc.querySelectorAll("joint")];
  const links=new Map(), linkGroups=new Map(); [...doc.querySelectorAll("link")].forEach(l=>links.set(l.getAttribute("name"),l));
  const childNames=new Set(joints.map(j=>j.querySelector("child")?.getAttribute("link"))),rootLink=[...links].find(([name])=>!childNames.has(name))?.[1];await addRobotVisuals(rootLink,arm,index);
  let parent=arm, flange=arm, qi=0;
  for(const j of joints){
    const origin=j.querySelector("origin"), xyz=(origin?.getAttribute("xyz")||"0 0 0").split(/\\s+/).map(Number), rr=(origin?.getAttribute("rpy")||"0 0 0").split(/\\s+/).map(Number);
    const jointGroup=new THREE.Group(); jointGroup.position.set(...xyz); jointGroup.quaternion.copy(rpy(rr)); parent.add(jointGroup);
    const type=j.getAttribute("type"), axisValue=(j.querySelector("axis")?.getAttribute("xyz")||"0 0 1").split(/\\s+/).map(Number), linkName=j.querySelector("child")?.getAttribute("link"), link=links.get(linkName);
    const linkGroup=new THREE.Group(); jointGroup.add(linkGroup); linkGroups.set(linkName,linkGroup);
    if(type!=="fixed"){const q=spec.joints[qi++]||0; const a=new THREE.Vector3(...axisValue).normalize(); linkGroup.setRotationFromAxisAngle(a,q);}
    await addRobotVisuals(link,linkGroup,index);
    parent=linkGroup; if(linkName==="flange") flange=linkGroup;
  }
  flange.add(axis(.16));
  const tool=pose(spec.flange_to_tool); tool.name="rdk-tool-frame"; flange.add(tool); tool.add(axis(.13));
  await buildGripper(flange,index);
  rows.push(`<div><b>${spec.side}</b> ${spec.serial} · ${spec.variant}</div><div class="legend">cyan/orange robot · RGB axes: flange and RDK tool frames</div>`); if(spec.warning)rows.push(`<div class="warning">${spec.warning}</div>`);
}
for(let i=0;i<DATA.arms.length;i++) await buildArm(DATA.arms[i],i);
rows.push(`<div class="legend">LK gripper: verified UMI3 soft-finger CAD visual</div>`); hud.innerHTML=rows.join("");
window.__flexivViewer={scene,camera,controls};
addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
function loop(){controls.update();renderer.render(scene,camera);requestAnimationFrame(loop)} loop();
</script></body></html>""".replace("__PAYLOAD__", data)


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, root: Path, temp_root: Path, **kwargs: Any) -> None:
        self._root = root
        self._temp_root = temp_root
        super().__init__(*args, directory=str(temp_root), **kwargs)

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].lstrip("/")
        if clean.startswith("assets/robot/rizon/"):
            return str(self._root / clean)
        if clean.startswith("vendor/"):
            return str(self._root / "incubator" / "control_panel" / "static" / clean)
        if clean.startswith("assets/tools/"):
            relative = clean.removeprefix("assets/tools/")
            return str(self._root / "incubator" / "control_panel" / "static" / "assets" / "tools" / relative)
        return str(self._temp_root / clean)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        force_cleanup = args.force_process_cleanup or force_cleanup_requested()
        if force_cleanup:
            cleanup_report = cleanup_before_startup("visualize_assembly", force=True)
        else:
            cleanup_report = cleanup_before_startup("visualize_assembly")
    except ProcessCleanupError as exc:
        print(f"visualize_flexiv_assembly: ERROR: {exc}", file=sys.stderr)
        return 2
    if cleanup_report.candidates:
        print(f"visualize_flexiv_assembly: process cleanup: {cleanup_report.summary()}")
    with tempfile.TemporaryDirectory(prefix="flexiv-assembly-") as directory:
        temp_root = Path(directory)

        def handler(*hargs: Any, **kwargs: Any) -> _Handler:
            return _Handler(*hargs, root=REPO_ROOT, temp_root=temp_root, **kwargs)

        server = ThreadingHTTPServer(("127.0.0.1", int(args.port)), handler)
        try:
            payload = _page_payload(args)
            (temp_root / "index.html").write_text(_html(payload), encoding="utf-8")
            url = f"http://127.0.0.1:{server.server_port}/index.html"
            print(f"Flexiv assembly viewer: {url}")
            if not args.no_browser:
                threading.Timer(0.15, webbrowser.open, args=(url,)).start()
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nViewer stopped.")
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
