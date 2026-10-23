#!/usr/bin/env python3
"""Detect the R1 USB cameras + LKTech gripper serial ports and emit the matching
config values (capture_id for each WebCameraDevice, by-path port for each
gripper), then optionally diff them against an existing device config.

R1-SPECIFIC: this script targets the R1 rig's hardware (4K U3 head cam, openMV
wrist cams, CH340 gripper adapters) and defaults to the R1 device configs. It is
not a general device-detection tool.

WHY: this host re-enumerates /dev/videoN and /dev/ttyUSBN on every replug/reboot,
so the hard-coded capture_id / by-path ports in
``configs/devices/quest3_collect_r1.yaml`` drift constantly. Run this before a
session to get the correct values (or to confirm the config still matches).

Classification (by USB device name / vendor):
  * "4K U3 Camera"   -> HEAD       (camera_2)
  * "openMV"         -> WRIST      (camera_0 / camera_1)
  * "ZED"            -> excluded   (handled by ZedStereoDevice, not a WebCamera)
  * CH340 (1a86:7523) serial -> GRIPPER (left/right)

LEFT/RIGHT cannot be inferred from enumeration. The script assigns them in a
stable order and marks them "VERIFY": confirm by preview (cameras) / by jogging
one hand (grippers) and swap if mirrored.

Usage
-----
    # just print detected hardware + a ready-to-paste YAML snippet
    python scripts/check_device_ports_r1.py

    # also diff against a config (default: configs/devices/quest3_collect_r1.yaml)
    python scripts/check_device_ports_r1.py --check
    python scripts/check_device_ports_r1.py --check --config configs/devices/quest3_replay_r1.yaml

    # try to connect every device listed in the config (3 web cameras, ZED, 2
    # LKTech grippers, 2 Flexiv arms), grab a frame from the left/right wrist cams
    # so you can tell them apart, and report whatever fails to connect
    python scripts/check_device_ports_r1.py --connect
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "devices"
    / "quest3_collect_r1.yaml"
)

# Make repo modules (devices.flexiv_adapter) and the bundled LKTech SDK importable
# for the --connect path, regardless of where the script is launched from.
for _p in (REPO_ROOT, REPO_ROOT / "third_party" / "lktech_motor_sdk"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Where --connect drops the captured camera previews (left/right wrist + head).
PREVIEW_DIR = REPO_ROOT / "logs" / "device_check"

# WebCameraDevice device_class_id -> human role. Matches the R1 config convention:
# camera_0 = inhand_right, camera_1 = inhand_left, camera_2 = head.
CAMERA_ROLE_BY_ID = {0: "right_wrist", 1: "left_wrist", 2: "head"}

DEFAULT_GRIPPER_BAUDRATE = 115200

# USB IDs of the gripper serial adapter. Default = CH340 (the LKTech rig's
# adapter); override with --gripper-vendor / --gripper-product for other chips.
DEFAULT_GRIPPER_VENDOR = "1a86"
DEFAULT_GRIPPER_PRODUCT = "7523"


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        return ""


def detect_camera_by_paths() -> Dict[int, str]:
    """Map /dev/videoN -> its stable ``/dev/v4l/by-path`` capture symlink.

    The config's ``capture_id`` is one of these by-path strings (stable across
    replug/reboot), so we resolve each link's realpath back to a video node and pick
    the canonical ``usb-0:`` variant over the duplicate ``usbv2/usbv3`` symlinks that
    point at the same node.
    """
    by_node: Dict[int, List[str]] = {}
    for link in glob.glob("/dev/v4l/by-path/*"):
        try:
            target = os.path.realpath(link)
        except OSError:
            continue
        m = re.search(r"/dev/video(\d+)$", target)
        if not m:
            continue
        by_node.setdefault(int(m.group(1)), []).append(link)

    canonical: Dict[int, str] = {}
    for node, links in by_node.items():
        # usb-0 first, then usbv2/usbv3 duplicates; index0 link before index1.
        canonical[node] = sorted(links, key=lambda p: ("usbv" in p, p))[0]
    return canonical


def detect_cameras() -> List[Dict[str, object]]:
    """Parse ``v4l2-ctl --list-devices`` into [{name, bus, capture_id, by_path, nodes}]."""
    out = _run(["v4l2-ctl", "--list-devices"])
    if not out.strip():
        print("[warn] `v4l2-ctl --list-devices` returned nothing "
              "(install v4l-utils, or no cameras).", file=sys.stderr)
        return []

    node_by_path = detect_camera_by_paths()
    cameras: List[Dict[str, object]] = []
    cur_name: Optional[str] = None
    cur_bus: Optional[str] = None
    cur_nodes: List[int] = []

    def _flush() -> None:
        if cur_name is None:
            return
        if cur_nodes:
            capture_node = min(cur_nodes)  # lowest /dev/videoN = capture node
            cameras.append({
                "name": cur_name,
                "bus": cur_bus or "",
                "capture_id": capture_node,
                "by_path": node_by_path.get(capture_node, ""),
                "nodes": sorted(cur_nodes),
            })

    for line in out.splitlines():
        if not line.strip():
            continue
        if not line.startswith(("\t", " ")):
            # Header line: "<Name> (usb-<bus>):"
            _flush()
            m = re.match(r"^(.*?)\s*\(([^)]*)\):\s*$", line)
            if m:
                cur_name, cur_bus = m.group(1).strip(), m.group(2).strip()
            else:
                cur_name, cur_bus = line.strip().rstrip(":"), ""
            cur_nodes = []
        else:
            m = re.search(r"/dev/video(\d+)", line)
            if m:
                cur_nodes.append(int(m.group(1)))
    _flush()
    return cameras


def classify_camera(name: str) -> str:
    low = name.lower()
    if "zed" in low:
        return "zed"
    if "4k u3" in low or "4k_u3" in low:
        return "head"
    if "openmv" in low:
        return "wrist"
    return "unknown"


# --------------------------------------------------------------------------- #
# Gripper serial ports
# --------------------------------------------------------------------------- #
def _usb_ids_for_tty(tty: str) -> Tuple[str, str]:
    """Return (idVendor, idProduct) for /dev/ttyUSBn via udevadm, else ('','')."""
    out = _run(["udevadm", "info", "-q", "property", "-n", tty])
    vid = pid = ""
    for line in out.splitlines():
        if line.startswith("ID_VENDOR_ID="):
            vid = line.split("=", 1)[1].strip()
        elif line.startswith("ID_MODEL_ID="):
            pid = line.split("=", 1)[1].strip()
    return vid, pid


def detect_gripper_ports(vendor: str = DEFAULT_GRIPPER_VENDOR,
                         product: str = DEFAULT_GRIPPER_PRODUCT) -> List[Dict[str, str]]:
    """Find the gripper serial adapters as stable by-path symlinks.

    Keeps adapters whose USB IDs match ``vendor`` (and ``product`` when both the
    requested and discovered product IDs are known). Returns
    [{by_path, tty, vendor, product}], sorted by by-path so the order is
    deterministic run-to-run. Prefers the canonical ``usb-0:`` variant over the
    duplicate ``usbv2-0:`` symlink that points at the same tty.
    """
    links = glob.glob("/dev/serial/by-path/*")
    # Drop the duplicate usbv2 variant when a usb-0 variant exists for the tty.
    by_tty: Dict[str, List[str]] = {}
    for link in links:
        try:
            tty = os.path.realpath(link)
        except OSError:
            continue
        by_tty.setdefault(tty, []).append(link)

    results: List[Dict[str, str]] = []
    for tty, candidate_links in by_tty.items():
        canonical = sorted(
            candidate_links,
            key=lambda p: ("usbv2" in p, p),  # usb-0 first, then usbv2
        )[0]
        vid, pid = _usb_ids_for_tty(tty)
        # Filter by USB IDs when udevadm gave us values; if it failed, keep all.
        if vid and vid != vendor:
            continue
        if product and pid and pid != product:
            continue
        results.append({
            "by_path": canonical,
            "tty": tty,
            "vendor": vid,
            "product": pid,
        })

    results.sort(key=lambda d: d["by_path"])
    return results


# --------------------------------------------------------------------------- #
# Connection test (--connect): read the config and try to talk to each device
# --------------------------------------------------------------------------- #
def load_device_config(config_path: Path) -> List[Dict[str, Any]]:
    """Return the ``devices`` list from a YAML device config (empty on failure)."""
    try:
        import yaml
    except ImportError:
        print("[warn] PyYAML not available; cannot run --connect.", file=sys.stderr)
        return []
    if not config_path.exists():
        print(f"[warn] config not found: {config_path}", file=sys.stderr)
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    devices = cfg.get("devices", []) if isinstance(cfg, dict) else []
    return [d for d in devices if isinstance(d, dict)]


def extract_web_cameras(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pull WebCameraDevice specs, tagged with their role (left/right wrist, head)."""
    cams: List[Dict[str, Any]] = []
    for d in devices:
        if d.get("class") != "WebCameraDevice":
            continue
        device_class_id = int(d.get("device_class_id", -1))
        cams.append({
            "device_class_id": device_class_id,
            "role": CAMERA_ROLE_BY_ID.get(device_class_id, f"camera_{device_class_id}"),
            "capture_id": d.get("capture_id"),
            "backend": d.get("backend"),
            "fourcc": d.get("fourcc"),
            "data_shape": d.get("data_shape") or [480, 640, 3],
            "fps": float(d.get("fps", 30.0)),
        })
    cams.sort(key=lambda c: c["device_class_id"])
    return cams


def extract_zed(devices: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for d in devices:
        if d.get("class") == "ZedStereoDevice":
            return {
                "device_class_id": int(d.get("device_class_id", 0)),
                "zed_resolution": d.get("zed_resolution", "HD720"),
                "camera_fps": int(d.get("camera_fps", 60)),
                "disable_self_calib": bool(d.get("disable_self_calib", True)),
            }
    return None


def extract_grippers(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pull LKTech gripper (port, motor_id, baudrate) pairs, ordered left then right."""
    labels = ["left", "right"]
    for d in devices:
        gripper = d.get("gripper")
        if not isinstance(gripper, dict):
            continue
        if str(gripper.get("backend", "")).lower() != "lktech":
            continue
        ports = gripper.get("port") or []
        motor_ids = gripper.get("motor_id") or []
        baud = gripper.get("baudrate", DEFAULT_GRIPPER_BAUDRATE)
        results: List[Dict[str, Any]] = []
        for i, port in enumerate(ports):
            results.append({
                "label": labels[i] if i < len(labels) else f"gripper_{i}",
                "port": str(port),
                "motor_id": int(motor_ids[i]) if i < len(motor_ids) else 1,
                "baudrate": int(baud),
            })
        return results
    return []


def extract_arms(devices: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Pull the dual Flexiv arm serials, ordered left then right (config convention)."""
    labels = ["left", "right"]
    for d in devices:
        cls = str(d.get("class", ""))
        if not cls.startswith("RizonRobot"):
            continue
        sns = d.get("robot_sn")
        if isinstance(sns, str):
            sns = [sns]
        if not isinstance(sns, list):
            continue
        return [
            {"label": labels[i] if i < len(labels) else f"arm_{i}", "robot_sn": str(sn)}
            for i, sn in enumerate(sns)
        ]
    return []


def open_web_camera(spec: Dict[str, Any]) -> "Any":
    """Open a cv2.VideoCapture for a WebCameraDevice spec (fourcc/res applied)."""
    import cv2

    capture_id = spec["capture_id"]
    backend = spec.get("backend")
    cap = (
        cv2.VideoCapture(capture_id, int(backend))
        if backend is not None
        else cv2.VideoCapture(capture_id)
    )
    if not cap or not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {capture_id}")

    data_shape = spec.get("data_shape") or [480, 640, 3]
    height, width = int(data_shape[0]), int(data_shape[1])
    fourcc = spec.get("fourcc")
    if fourcc:
        code = cv2.VideoWriter_fourcc(*str(fourcc)[:4])
        cap.set(cv2.CAP_PROP_FOURCC, float(code))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(max(1.0, spec.get("fps", 30.0))))
    return cap


def test_web_camera(spec: Dict[str, Any]) -> Tuple[bool, str, Optional["Any"]]:
    """Open a web camera and grab one frame. Returns (ok, message, frame_bgr)."""
    cap = None
    try:
        cap = open_web_camera(spec)
        for _ in range(5):  # warm-up: first frames are often blank on UVC cams
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "opened but warm-up frame read failed", None
        return True, f"connected, frame {tuple(frame.shape)}", frame
    except Exception as exc:  # noqa: BLE001 - report any failure verbatim
        return False, str(exc), None
    finally:
        if cap is not None:
            cap.release()


def test_zed(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """Open the ZED and grab one frame (no depth), then close. Returns (ok, message)."""
    try:
        import pyzed.sl as sl
    except Exception as exc:  # noqa: BLE001
        return False, f"pyzed not importable: {exc}"

    resolutions = {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD1200": sl.RESOLUTION.HD1200,
        "VGA": sl.RESOLUTION.VGA,
    }
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = resolutions.get(spec.get("zed_resolution", "HD720"), sl.RESOLUTION.HD720)
    init.camera_fps = int(spec.get("camera_fps", 60))
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    init.sdk_verbose = 0
    init.camera_disable_self_calib = bool(spec.get("disable_self_calib", True))
    try:
        err = zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            return False, f"ZED open failed: {err!r}"
        if zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            return False, "ZED opened but grab() failed"
        return True, f"connected ({spec.get('zed_resolution', 'HD720')}@{init.camera_fps})"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        try:
            zed.close()
        except Exception:  # noqa: BLE001
            pass


def test_gripper(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """Open the LKTech serial port and ping the motor. Returns (ok, message)."""
    try:
        from lktech_motor_sdk import LKMotor
    except Exception as exc:  # noqa: BLE001
        return False, f"lktech_motor_sdk import failed: {exc}"

    motor = LKMotor(
        port=spec["port"],
        motor_id=int(spec.get("motor_id", 1)),
        baudrate=int(spec.get("baudrate", DEFAULT_GRIPPER_BAUDRATE)),
        timeout=0.2,
    )
    try:
        motor.connect()
    except Exception as exc:  # noqa: BLE001
        return False, f"serial open failed: {exc}"
    try:
        if motor.ping():
            return True, "serial open + motor responds"
        return False, "serial open but motor not responding (status read timed out)"
    finally:
        try:
            motor.disconnect()
        except Exception:  # noqa: BLE001
            pass


def test_flexiv_arm(adapter: "Any", robot_sn: str) -> Tuple[bool, str]:
    """Connect to one Flexiv arm and read its joint states. Returns (ok, message)."""
    try:
        robot = adapter.create_robot(robot_sn)
        states = robot.states()
        q = [round(float(v), 3) for v in list(getattr(states, "q", []))[:3]]
        return True, f"connected (q[:3]={q})"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _label_frame(frame_bgr: "Any", text: str) -> "Any":
    """Burn a readable banner onto the top of a frame (for the wrist comparison)."""
    import cv2

    img = frame_bgr.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    return img


def save_wrist_previews(frames_by_role: Dict[str, "Any"], show: bool) -> List[Path]:
    """Save per-camera previews + a labeled left|right wrist comparison image.

    Returns the list of written file paths. When ``show`` is true and a display is
    available, also pops up the comparison window (close it / press a key to continue).
    """
    import cv2
    import numpy as np

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for role, frame in frames_by_role.items():
        if frame is None:
            continue
        path = PREVIEW_DIR / f"{role}.png"
        cv2.imwrite(str(path), frame)
        written.append(path)

    left = frames_by_role.get("left_wrist")
    right = frames_by_role.get("right_wrist")
    compare: Optional["Any"] = None
    if left is not None and right is not None:
        height = min(left.shape[0], right.shape[0])
        left_r = cv2.resize(left, (int(left.shape[1] * height / left.shape[0]), height))
        right_r = cv2.resize(right, (int(right.shape[1] * height / right.shape[0]), height))
        left_lbl = _label_frame(left_r, "LEFT wrist (camera_1)")
        right_lbl = _label_frame(right_r, "RIGHT wrist (camera_0)")
        compare = np.hstack([left_lbl, right_lbl])
        compare_path = PREVIEW_DIR / "wrist_compare.png"
        cv2.imwrite(str(compare_path), compare)
        written.append(compare_path)

    if show and os.environ.get("DISPLAY"):
        try:
            window = compare if compare is not None else (left if left is not None else right)
            if window is not None:
                cv2.imshow("wrist L|R preview (press any key to continue)", window)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not show preview window: {exc}", file=sys.stderr)

    return written


def run_connection_tests(config_path: Path, show_preview: bool) -> bool:
    """Try to connect every device in the config; print results. Returns all-ok."""
    devices = load_device_config(config_path)
    if not devices:
        print("[error] no devices loaded; cannot run connection tests.", file=sys.stderr)
        return False

    cameras = extract_web_cameras(devices)
    zed = extract_zed(devices)
    grippers = extract_grippers(devices)
    arms = extract_arms(devices)

    print("=" * 70)
    print(f"CONNECTION TEST vs {config_path}")
    print("=" * 70)

    failures: List[str] = []
    frames_by_role: Dict[str, "Any"] = {}

    # ---- Web cameras (2 wrist + 1 head): open + grab one frame ----
    print("\n-- Web cameras --")
    for cam in cameras:
        ok, msg, frame = test_web_camera(cam)
        role = str(cam["role"])
        frames_by_role[role] = frame
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {role:<11} (camera_{cam['device_class_id']}): {msg}")
        print(f"             capture_id: {cam['capture_id']}")
        if not ok:
            failures.append(f"camera {role} (camera_{cam['device_class_id']})")

    # ---- ZED stereo ----
    print("\n-- ZED stereo --")
    if zed is None:
        print("  [SKIP] no ZedStereoDevice in config")
    else:
        ok, msg = test_zed(zed)
        print(f"  [{'OK  ' if ok else 'FAIL'}] ZED (device_{zed['device_class_id']}): {msg}")
        if not ok:
            failures.append("ZED stereo camera")

    # ---- LKTech grippers ----
    print("\n-- LKTech grippers --")
    if not grippers:
        print("  [SKIP] no LKTech gripper in config")
    for g in grippers:
        ok, msg = test_gripper(g)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {g['label']:<5} gripper (motor_id={g['motor_id']}): {msg}")
        print(f"             port: {g['port']}")
        if not ok:
            failures.append(f"{g['label']} gripper")

    # ---- Flexiv arms ----
    print("\n-- Flexiv arms --")
    if not arms:
        print("  [SKIP] no Rizon robot in config")
    else:
        try:
            from devices.flexiv_adapter import FlexivAdapter
            adapter = FlexivAdapter(backend="python")
        except Exception as exc:  # noqa: BLE001
            adapter = None
            print(f"  [FAIL] could not init Flexiv backend: {exc}")
            failures.extend(f"{a['label']} arm ({a['robot_sn']})" for a in arms)
        if adapter is not None:
            for a in arms:
                ok, msg = test_flexiv_arm(adapter, a["robot_sn"])
                print(f"  [{'OK  ' if ok else 'FAIL'}] {a['label']:<5} arm ({a['robot_sn']}): {msg}")
                if not ok:
                    failures.append(f"{a['label']} arm ({a['robot_sn']})")

    # ---- Wrist previews so the operator can tell L/R apart ----
    print("\n-- Wrist camera previews (to distinguish left/right) --")
    written = save_wrist_previews(frames_by_role, show_preview)
    if written:
        for path in written:
            print(f"  saved: {path}")
        if not (frames_by_role.get("left_wrist") is not None
                and frames_by_role.get("right_wrist") is not None):
            print("  [warn] one or both wrist cameras failed; comparison image incomplete.")
    else:
        print("  [warn] no camera frames captured; nothing to preview.")

    # ---- Summary ----
    print("\n" + "=" * 70)
    if failures:
        print(f"CONNECTION SUMMARY: {len(failures)} device(s) FAILED to connect:")
        for item in failures:
            print(f"  - {item}")
    else:
        print("CONNECTION SUMMARY: all devices connected successfully.")
    print("=" * 70)
    return not failures


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_detection(cameras: List[Dict[str, object]],
                    grippers: List[Dict[str, str]]) -> None:
    print("=" * 70)
    print("DETECTED CAMERAS (v4l2-ctl --list-devices)")
    print("=" * 70)
    if not cameras:
        print("  (none)")
    for cam in cameras:
        role = classify_camera(str(cam["name"]))
        print(f"  [{role:^7}] /dev/video{cam['capture_id']:<3} "
              f"nodes={cam['nodes']}  {cam['name']}  ({cam['bus']})")
        by_path = str(cam["by_path"]) or "<no by-path symlink found>"
        print(f"            by-path: {by_path}")

    print()
    print("=" * 70)
    print("DETECTED GRIPPER SERIAL (CH340 via /dev/serial/by-path)")
    print("=" * 70)
    if not grippers:
        print("  (none)")
    for g in grippers:
        ids = f"{g['vendor']}:{g['product']}" if g["vendor"] else "ids=?"
        print(f"  {g['by_path']} -> {g['tty']}  ({ids})")


def build_snippet(cameras: List[Dict[str, object]],
                  grippers: List[Dict[str, str]]) -> str:
    heads = [c for c in cameras if classify_camera(str(c["name"])) == "head"]
    wrists = [c for c in cameras if classify_camera(str(c["name"])) == "wrist"]
    wrists.sort(key=lambda c: c["capture_id"])

    def _cap(cam: Dict[str, object]) -> str:
        # The config uses the stable by-path string for capture_id; fall back to
        # the /dev/videoN node only when no by-path symlink exists.
        return str(cam["by_path"]) or f"/dev/video{cam['capture_id']}"

    lines: List[str] = []
    lines.append("# ---- paste into configs/devices/quest3_collect_r1.yaml ----")
    lines.append("# camera_0 = inhand_right, camera_1 = inhand_left (VERIFY L/R by preview)")
    if len(wrists) >= 1:
        lines.append(f'#   camera_0 (right wrist): capture_id: "{_cap(wrists[0])}"')
    else:
        lines.append("#   camera_0 (right wrist): capture_id: <NO WRIST CAM FOUND>")
    if len(wrists) >= 2:
        lines.append(f'#   camera_1 (left  wrist): capture_id: "{_cap(wrists[1])}"')
    else:
        lines.append("#   camera_1 (left  wrist): capture_id: <NO 2nd WRIST CAM FOUND>")
    if heads:
        lines.append(f'#   camera_2 (head):        capture_id: "{_cap(heads[0])}"')
    else:
        lines.append("#   camera_2 (head):        capture_id: <NO 4K U3 CAM FOUND>")

    lines.append("")
    lines.append("# gripper.port (left first, right second -- VERIFY L/R by jogging a hand):")
    if grippers:
        lines.append("      port:")
        labels = ["left ", "right"]
        for i, g in enumerate(grippers[:2]):
            label = labels[i] if i < len(labels) else f"#{i}"
            lines.append(f'        - "{g["by_path"]}"   # {label} ({g["tty"]})')
        if len(grippers) > 2:
            lines.append(f"# NOTE: found {len(grippers)} CH340 adapters (expected 2)")
    else:
        lines.append("      port: []   # NO CH340 GRIPPER ADAPTERS FOUND")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optional diff against an existing config
# --------------------------------------------------------------------------- #
def check_against_config(config_path: Path,
                         cameras: List[Dict[str, object]],
                         grippers: List[Dict[str, str]]) -> bool:
    """Return True if the config matches detected hardware (ignoring L/R)."""
    try:
        import yaml
    except ImportError:
        print("[warn] PyYAML not available; skipping --check.", file=sys.stderr)
        return True
    if not config_path.exists():
        print(f"[warn] config not found: {config_path}", file=sys.stderr)
        return True

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    devices = cfg.get("devices", []) if isinstance(cfg, dict) else []
    cfg_caps = sorted(
        str(d["capture_id"])
        for d in devices
        if isinstance(d, dict) and d.get("class") == "WebCameraDevice"
        and d.get("capture_id") is not None
    )
    cfg_ports: List[str] = []
    for d in devices:
        if isinstance(d, dict) and "gripper" in d and isinstance(d["gripper"], dict):
            ports = d["gripper"].get("port") or []
            cfg_ports = [str(p) for p in ports]

    det_caps = sorted(
        str(c["by_path"]) or f"/dev/video{c['capture_id']}"
        for c in cameras
        if classify_camera(str(c["name"])) in ("head", "wrist")
    )
    det_ports = sorted(g["by_path"] for g in grippers)

    print()
    print("=" * 70)
    print(f"DIFF vs {config_path}")
    print("=" * 70)
    ok = True

    print(f"  cameras: config capture_id set = {cfg_caps}")
    print(f"           detected (head+wrist)     = {det_caps}")
    if cfg_caps != det_caps:
        ok = False
        print("  [MISMATCH] camera capture_id set differs -> update the config")
    else:
        print("  [OK] camera capture_id set matches (L/R still to verify)")

    print(f"  grippers: config ports   = {sorted(cfg_ports)}")
    print(f"            detected ports = {det_ports}")
    if sorted(cfg_ports) != det_ports:
        ok = False
        print("  [MISMATCH] gripper by-path ports differ -> update the config")
    else:
        print("  [OK] gripper by-path ports match (L/R still to verify)")

    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="Diff detected hardware against --config and exit non-zero on mismatch.")
    p.add_argument("--connect", action="store_true",
                   help="Try to connect every device in --config (web cameras, ZED, "
                        "grippers, arms), preview the wrist cams, and report failures.")
    p.add_argument("--no-show", action="store_true",
                   help="With --connect, save wrist previews but do not open a display window.")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                   help=f"Device config to diff/connect against (default: {DEFAULT_CONFIG})")
    p.add_argument("--gripper-vendor", default=DEFAULT_GRIPPER_VENDOR,
                   help="USB idVendor of the gripper serial adapter "
                        f"(default: {DEFAULT_GRIPPER_VENDOR} = CH340)")
    p.add_argument("--gripper-product", default=DEFAULT_GRIPPER_PRODUCT,
                   help="USB idProduct of the gripper serial adapter "
                        f"(default: {DEFAULT_GRIPPER_PRODUCT} = CH340)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --connect is a self-contained mode: talk to each device in the config and
    # preview the wrist cams; skip the enumeration/snippet output to keep it focused.
    if args.connect:
        ok = run_connection_tests(Path(args.config), show_preview=not args.no_show)
        sys.exit(0 if ok else 1)

    cameras = detect_cameras()
    grippers = detect_gripper_ports(args.gripper_vendor, args.gripper_product)

    print_detection(cameras, grippers)
    print()
    print(build_snippet(cameras, grippers))

    if args.check:
        ok = check_against_config(Path(args.config), cameras, grippers)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
