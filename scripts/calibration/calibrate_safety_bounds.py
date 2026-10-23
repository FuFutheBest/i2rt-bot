#!/usr/bin/env python3
"""
Calibrate an end-effector safety bounding box by manually dragging the
Flexiv Rizon robot in compliant mode.

The script connects to the robot, switches into a soft Cartesian impedance
mode (force-control axes disabled), continuously echoes the current TCP
pose back to the controller so the arm tracks human hand motion, and
records the running min/max of the TCP XYZ position while the operator
drags the end-effector through the desired workspace.

Workflow:
  1. Connect & Enable the robot.
  2. Switch to NRT_CARTESIAN_MOTION_FORCE with soft stiffness + damping.
  3. Operator presses Enter to start sampling.
  4. Operator drags the robot to all corners of the desired workspace.
  5. Operator presses Enter to stop sampling.
  6. Script prints the calibrated bounds (with optional inward margin) as
     a YAML snippet ready to paste under the RizonRobot device config.
     Optionally writes the snippet back into a config file in place.
  7. Restores a stiff hold mode before exiting (to prevent the arm from
     drifting under gravity once compliance is removed).

Reference: iphone_calibration/core/flexiv_robot.py FlexivRobot.enable_compliance().

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Make sibling utils importable when running this script standalone.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.logger_config import logger  # noqa: E402

try:
    import flexivrdk
except ImportError as exc:  # pragma: no cover - runtime dependency
    logger.error(f"flexivrdk import failed: {exc}")
    logger.error("Install Flexiv RDK Python bindings before running this script.")
    raise


DEFAULT_COMPLIANT_STIFFNESS: List[float] = [1000.0, 1000.0, 1000.0, 8.0, 8.0, 8.0]
DEFAULT_HOLD_STIFFNESS: List[float] = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]
DEFAULT_DAMPING: List[float] = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
CONTROL_HZ = 100.0
SAMPLE_HZ = 20.0
STATUS_PRINT_PERIOD_S = 0.5


@dataclass
class BoundsAccumulator:
    """Running min/max accumulator for a 3D point cloud."""

    min_xyz: Optional[np.ndarray] = None
    max_xyz: Optional[np.ndarray] = None
    samples: int = 0

    def update(self, pos: np.ndarray) -> None:
        if self.min_xyz is None:
            self.min_xyz = pos.copy()
            self.max_xyz = pos.copy()
        else:
            self.min_xyz = np.minimum(self.min_xyz, pos)
            self.max_xyz = np.maximum(self.max_xyz, pos)
        self.samples += 1

    def reset(self) -> None:
        self.min_xyz = None
        self.max_xyz = None
        self.samples = 0

    def has_data(self) -> bool:
        return self.min_xyz is not None and self.samples > 0


def _local_ipv4_addresses() -> List[str]:
    """Return IPv4 addresses on this host (used as Flexiv NIC whitelist)."""
    ip_cmd = shutil.which("ip")
    if not ip_cmd:
        return []
    try:
        result = subprocess.run(
            [ip_cmd, "-4", "-o", "addr", "show", "scope", "global"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    addresses: List[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        ip = parts[parts.index("inet") + 1].split("/")[0]
        if not ip.startswith("127.") and ip not in addresses:
            addresses.append(ip)
    return addresses


class CompliantTeacher:
    """Lightweight wrapper around a single Flexiv Rizon for compliant teaching."""

    def __init__(
        self,
        robot_sn: str,
        compliant_stiffness: Sequence[float] = DEFAULT_COMPLIANT_STIFFNESS,
        hold_stiffness: Sequence[float] = DEFAULT_HOLD_STIFFNESS,
        damping: Sequence[float] = DEFAULT_DAMPING,
    ) -> None:
        self.robot_sn = robot_sn
        self.compliant_stiffness = list(compliant_stiffness)
        self.hold_stiffness = list(hold_stiffness)
        self.damping = list(damping)

        self._robot: Optional[flexivrdk.Robot] = None
        self._mode_enum = flexivrdk.Mode
        self._control_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_tcp: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()

    def connect(self, enable_timeout_s: float = 20.0) -> None:
        whitelist = _local_ipv4_addresses()
        logger.info(
            f"Connecting to {self.robot_sn} (NIC whitelist={whitelist or 'none'})..."
        )
        robot = flexivrdk.Robot(self.robot_sn, whitelist, False, False)

        if robot.fault():
            logger.warning("Robot reports fault state; attempting ClearFault()")
            robot.ClearFault()

        if not robot.operational():
            robot.Enable()
            deadline = time.time() + enable_timeout_s
            while time.time() < deadline:
                if robot.operational():
                    break
                time.sleep(0.1)
            if not robot.operational():
                raise RuntimeError("Robot failed to reach operational state in time.")

        robot.SwitchMode(self._mode_enum.NRT_CARTESIAN_MOTION_FORCE)
        time.sleep(0.1)
        self._robot = robot
        logger.info("Robot connected and switched to NRT_CARTESIAN_MOTION_FORCE.")

    def enable_compliance(self) -> None:
        if self._robot is None:
            raise RuntimeError("Robot not connected.")
        self._robot.SetForceControlAxis([False] * 6)
        self._robot.SetCartesianImpedance(self.compliant_stiffness, self.damping)
        logger.info(
            f"Compliant mode active: stiffness={self.compliant_stiffness} "
            f"damping={self.damping}"
        )

    def restore_hold(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.SetForceControlAxis([False] * 6)
            self._robot.SetCartesianImpedance(self.hold_stiffness, self.damping)
            logger.info(f"Restored hold stiffness: {self.hold_stiffness}")
        except Exception as exc:
            logger.warning(f"Failed to restore hold stiffness: {exc}")

    def start_control_loop(self) -> None:
        if self._robot is None:
            raise RuntimeError("Robot not connected.")
        self._stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop, name="CompliantControlLoop", daemon=True
        )
        self._control_thread.start()

    def stop_control_loop(self) -> None:
        self._stop_event.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None

    def stop(self) -> None:
        self.stop_control_loop()
        if self._robot is not None:
            try:
                self._robot.Stop()
            except Exception:
                pass

    def get_current_tcp(self) -> Optional[np.ndarray]:
        """Snapshot of the latest TCP pose (length-7 vector) or None."""
        with self._latest_lock:
            return None if self._latest_tcp is None else self._latest_tcp.copy()

    def _control_loop(self) -> None:
        period = 1.0 / CONTROL_HZ
        assert self._robot is not None
        while not self._stop_event.is_set():
            start = time.time()
            try:
                states = self._robot.states()
                tcp_pose = np.asarray(states.tcp_pose, dtype=float).reshape(-1)
                if tcp_pose.size >= 7:
                    with self._latest_lock:
                        self._latest_tcp = tcp_pose[:7]
                    self._robot.SendCartesianMotionForce(
                        tcp_pose[:7].tolist(),
                        [0.0] * 6,
                        [0.0] * 6,
                    )
            except Exception as exc:
                logger.debug(f"Control loop iteration error: {exc}")
            elapsed = time.time() - start
            time.sleep(max(0.0, period - elapsed))


def _sample_loop(
    teacher: CompliantTeacher,
    accumulator: BoundsAccumulator,
    sample_active: threading.Event,
    stop_event: threading.Event,
) -> None:
    """Background sampler; appends TCP xyz to accumulator while active."""
    period = 1.0 / SAMPLE_HZ
    while not stop_event.is_set():
        start = time.time()
        if sample_active.is_set():
            tcp = teacher.get_current_tcp()
            if tcp is not None:
                accumulator.update(tcp[:3])
        elapsed = time.time() - start
        time.sleep(max(0.0, period - elapsed))


def _status_loop(
    teacher: CompliantTeacher,
    accumulator: BoundsAccumulator,
    sample_active: threading.Event,
    stop_event: threading.Event,
) -> None:
    """Background printer; emits a status line every STATUS_PRINT_PERIOD_S."""
    while not stop_event.is_set():
        tcp = teacher.get_current_tcp()
        if tcp is not None:
            pos = tcp[:3]
            state = "SAMPLING" if sample_active.is_set() else "idle    "
            if accumulator.has_data():
                bbox_str = (
                    f" bbox min={np.round(accumulator.min_xyz, 4).tolist()} "
                    f"max={np.round(accumulator.max_xyz, 4).tolist()} "
                    f"N={accumulator.samples}"
                )
            else:
                bbox_str = ""
            sys.stdout.write(
                f"\r[{state}] TCP xyz={np.round(pos, 4).tolist()}{bbox_str}    "
            )
            sys.stdout.flush()
        time.sleep(STATUS_PRINT_PERIOD_S)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _apply_margin(
    min_xyz: np.ndarray, max_xyz: np.ndarray, margin_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink the bbox inward by `margin_m` per side. Returns (min, max)."""
    if margin_m <= 0.0:
        return min_xyz.copy(), max_xyz.copy()
    shrunk_min = min_xyz + margin_m
    shrunk_max = max_xyz - margin_m
    if np.any(shrunk_max <= shrunk_min):
        logger.warning(
            f"Margin {margin_m} m too aggressive vs bbox; falling back to raw bbox."
        )
        return min_xyz.copy(), max_xyz.copy()
    return shrunk_min, shrunk_max


def _format_yaml_snippet(
    min_xyz: np.ndarray, max_xyz: np.ndarray, frame: str = "robot_base"
) -> str:
    """Format a paste-ready YAML snippet for the safety_bounds block."""
    def _fmt(v: np.ndarray) -> str:
        return "[" + ", ".join(f"{x:.4f}" for x in v.tolist()) + "]"

    return (
        "safety_bounds:\n"
        "  enabled: true\n"
        f'  frame: "{frame}"\n'
        f"  min_xyz: {_fmt(min_xyz)}\n"
        f"  max_xyz: {_fmt(max_xyz)}\n"
    )


def _write_bounds_to_yaml(
    output_path: Path,
    device_class_id: Optional[int],
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    frame: str = "robot_base",
) -> None:
    """Patch the `safety_bounds:` block in the target YAML file in place.

    Prefers ruamel.yaml (preserves comments/formatting). Falls back to PyYAML
    (will lose comments) and writes a *.bak alongside the original.
    """
    if not output_path.exists():
        raise FileNotFoundError(f"Output file does not exist: {output_path}")

    backup_path = output_path.with_suffix(output_path.suffix + ".bak")
    shutil.copy2(output_path, backup_path)
    logger.info(f"Backup created at {backup_path}")

    try:
        from ruamel.yaml import YAML  # type: ignore[import-not-found]
        from ruamel.yaml.comments import CommentedMap, CommentedSeq  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "ruamel.yaml not installed; falling back to PyYAML "
            "(comments in the original file will be lost)."
        )
        _write_bounds_pyyaml(output_path, device_class_id, min_xyz, max_xyz, frame)
        return

    yaml_loader = YAML()
    yaml_loader.preserve_quotes = True
    yaml_loader.indent(mapping=2, sequence=4, offset=2)

    with output_path.open("r") as f:
        data = yaml_loader.load(f)

    devices = data.get("devices") if isinstance(data, dict) else None
    if not isinstance(devices, list):
        raise RuntimeError(
            f"YAML file {output_path} does not contain a top-level `devices:` list."
        )

    target = _select_robot_entry(devices, device_class_id)
    if target is None:
        raise RuntimeError(
            f"No RizonRobot device entry found in {output_path} "
            f"(device_class_id filter={device_class_id})."
        )

    bounds_map = CommentedMap()
    bounds_map["enabled"] = True
    bounds_map["frame"] = frame
    min_seq = CommentedSeq([float(round(v, 4)) for v in min_xyz.tolist()])
    max_seq = CommentedSeq([float(round(v, 4)) for v in max_xyz.tolist()])
    min_seq.fa.set_flow_style()
    max_seq.fa.set_flow_style()
    bounds_map["min_xyz"] = min_seq
    bounds_map["max_xyz"] = max_seq
    target["safety_bounds"] = bounds_map

    with output_path.open("w") as f:
        yaml_loader.dump(data, f)
    logger.info(f"Wrote safety_bounds into {output_path}")


def _write_bounds_pyyaml(
    output_path: Path,
    device_class_id: Optional[int],
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    frame: str,
) -> None:
    """PyYAML fallback when ruamel.yaml is unavailable."""
    import yaml

    with output_path.open("r") as f:
        data = yaml.safe_load(f)

    devices = data.get("devices") if isinstance(data, dict) else None
    if not isinstance(devices, list):
        raise RuntimeError(
            f"YAML file {output_path} does not contain a top-level `devices:` list."
        )

    target = _select_robot_entry(devices, device_class_id)
    if target is None:
        raise RuntimeError(
            f"No RizonRobot device entry found in {output_path} "
            f"(device_class_id filter={device_class_id})."
        )

    target["safety_bounds"] = {
        "enabled": True,
        "frame": frame,
        "min_xyz": [float(round(v, 4)) for v in min_xyz.tolist()],
        "max_xyz": [float(round(v, 4)) for v in max_xyz.tolist()],
    }

    with output_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    logger.info(
        f"Wrote safety_bounds into {output_path} (PyYAML; comments not preserved)."
    )


def _select_robot_entry(devices: list, device_class_id: Optional[int]):
    """Return the matching RizonRobot entry or None."""
    candidates = [d for d in devices if isinstance(d, dict) and d.get("class") == "RizonRobot"]
    if not candidates:
        return None
    if device_class_id is not None:
        for entry in candidates:
            if entry.get("device_class_id") == device_class_id:
                return entry
        return None
    if len(candidates) > 1:
        logger.warning(
            f"Multiple RizonRobot entries found ({len(candidates)}); "
            f"defaulting to the first one. Pass --device-class-id to disambiguate."
        )
    return candidates[0]


def _parse_vec(value: str, expected_len: int) -> List[float]:
    parts = [v.strip() for v in value.split(",")]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated floats, got {len(parts)}."
        )
    return [float(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--robot-sn", required=True, help="Robot serial number (e.g. Rizon4-063114).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to a device YAML to patch in place with the calibrated bounds. "
            "If omitted, the snippet is only printed to stdout."
        ),
    )
    parser.add_argument(
        "--device-class-id",
        type=int,
        default=None,
        help="device_class_id of the RizonRobot entry to patch (when multiple in the YAML).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Inward shrink (meters) applied to bbox before writing/printing.",
    )
    parser.add_argument(
        "--stiffness",
        type=lambda s: _parse_vec(s, 6),
        default=None,
        help="6-vector Cartesian stiffness for compliant mode (default 1000,1000,1000,8,8,8).",
    )
    parser.add_argument(
        "--damping",
        type=lambda s: _parse_vec(s, 6),
        default=None,
        help="6-vector Cartesian damping (default 0.6,0.6,0.6,0.6,0.6,0.6).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Force snippet-print only, even if --output is given.",
    )
    args = parser.parse_args()

    compliant_stiffness = args.stiffness if args.stiffness is not None else DEFAULT_COMPLIANT_STIFFNESS
    damping = args.damping if args.damping is not None else DEFAULT_DAMPING

    teacher = CompliantTeacher(
        robot_sn=args.robot_sn,
        compliant_stiffness=compliant_stiffness,
        damping=damping,
    )
    accumulator = BoundsAccumulator()
    sample_active = threading.Event()
    stop_event = threading.Event()

    try:
        teacher.connect()
        teacher.enable_compliance()
        teacher.start_control_loop()
        # Give the control loop a moment to populate the first TCP snapshot.
        time.sleep(0.2)

        sampler_thread = threading.Thread(
            target=_sample_loop,
            args=(teacher, accumulator, sample_active, stop_event),
            daemon=True,
        )
        status_thread = threading.Thread(
            target=_status_loop,
            args=(teacher, accumulator, sample_active, stop_event),
            daemon=True,
        )
        sampler_thread.start()
        status_thread.start()

        logger.info("Ready. The robot is now compliant; drag it to explore the workspace.")
        print("\n>>> Press ENTER to START sampling the bounding box...", flush=True)
        input()
        sample_active.set()
        logger.info("Sampling started. Move the robot through the safe workspace.")
        print(">>> Press ENTER again to STOP sampling.", flush=True)
        input()
        sample_active.clear()
        logger.info("Sampling stopped.")
        # Brief pause so the status line stops updating before we print results.
        time.sleep(0.2)

        if not accumulator.has_data():
            logger.error("No samples collected; aborting without output.")
            return

        raw_min = accumulator.min_xyz
        raw_max = accumulator.max_xyz
        adj_min, adj_max = _apply_margin(raw_min, raw_max, args.margin)

        logger.info(
            f"Raw bbox    : min={np.round(raw_min, 4).tolist()} "
            f"max={np.round(raw_max, 4).tolist()} "
            f"(N={accumulator.samples})"
        )
        logger.info(
            f"With margin : min={np.round(adj_min, 4).tolist()} "
            f"max={np.round(adj_max, 4).tolist()} (margin={args.margin} m)"
        )

        snippet = _format_yaml_snippet(adj_min, adj_max)
        print("\n----- Paste this under the RizonRobot device entry -----")
        print(snippet)
        print("--------------------------------------------------------")

        if args.output and not args.print_only:
            _write_bounds_to_yaml(
                Path(args.output),
                device_class_id=args.device_class_id,
                min_xyz=adj_min,
                max_xyz=adj_max,
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    finally:
        stop_event.set()
        teacher.restore_hold()
        teacher.stop()


if __name__ == "__main__":
    main()
