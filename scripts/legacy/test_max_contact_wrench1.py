#!/usr/bin/env python3
"""
Verify Flexiv `SetMaxContactWrench` behavior by commanding a slow
horizontal sinusoidal trajectory in NRT_CARTESIAN_MOTION_FORCE mode.

The robot uses an explicit Cartesian impedance (rigid tracking by default);
the only safety mechanism under test is the contact-wrench limit. When the
operator blocks the path, the measured external wrench at the TCP should
saturate near the configured limit and the robot should automatically
enter `IN_REDUCED_STATE` (motion slows/halts). Releasing the obstruction
should cause `operational_status` to leave the reduced state and the
trajectory to resume.

Usage:
    python scripts/legacy/test_max_contact_wrench1.py \
        --robot-sn Rizon4-063114 \
        --axis x --amplitude 0.10 --speed 0.05 --duration 60 \
        --max-wrench 20,20,20,5,5,5
 python scripts/legacy/test_max_contact_wrench1.py     --robot-sn Rizon4R-062088     --max-wrench 5,5,5,3,3,3     --amplitude 0.08 --speed 0.04 --duration 300 --axis z
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

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


DEFAULT_MAX_WRENCH: List[float] = [20.0, 20.0, 20.0, 5.0, 5.0, 5.0]
DEFAULT_IMPEDANCE: List[float] = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]
CONTROL_HZ = 50.0
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _local_ipv4_addresses() -> List[str]:
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
    out: List[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        ip = parts[parts.index("inet") + 1].split("/")[0]
        if not ip.startswith("127.") and ip not in out:
            out.append(ip)
    return out


def _connect(robot_sn: str, enable_timeout_s: float = 20.0) -> flexivrdk.Robot:
    whitelist = _local_ipv4_addresses()
    logger.info(f"Connecting to {robot_sn} (NIC whitelist={whitelist or 'none'})...")
    robot = flexivrdk.Robot(robot_sn, whitelist, False, False)
    if robot.fault():
        logger.warning("Robot reports fault state; calling ClearFault()")
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
    robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
    time.sleep(0.1)
    logger.info("Robot connected and in NRT_CARTESIAN_MOTION_FORCE mode.")
    return robot


def _parse_vec(value: str, expected_len: int) -> List[float]:
    parts = [v.strip() for v in value.split(",")]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated floats, got {len(parts)}."
        )
    return [float(p) for p in parts]


def _status_name(status: object) -> str:
    return getattr(status, "name", str(status))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Flexiv SetMaxContactWrench: drive a slow horizontal "
            "sinusoid and verify the robot auto-halts on contact."
        )
    )
    parser.add_argument("--robot-sn", required=True, help="Robot serial number.")
    parser.add_argument("--axis", choices=list(AXIS_INDEX.keys()), default="x",
                        help="Translation axis (robot base frame). Default: x.")
    parser.add_argument("--amplitude", type=float, default=0.10,
                        help="One-sided amplitude in meters (default 0.10).")
    parser.add_argument("--speed", type=float, default=0.05,
                        help="Approximate target speed in m/s (default 0.05).")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Total run duration in seconds (default 60).")
    parser.add_argument(
        "--max-wrench",
        type=lambda s: _parse_vec(s, 6),
        default=None,
        help=(
            "6-vector [Fx,Fy,Fz,Mx,My,Mz] in N / N*m (default 20,20,20,5,5,5). "
            "Robot enters IN_REDUCED_STATE when measured contact exceeds this."
        ),
    )
    parser.add_argument(
        "--impedance",
        type=lambda s: _parse_vec(s, 6),
        default=DEFAULT_IMPEDANCE,
        help=(
            "Cartesian impedance [Kx,Ky,Kz,Krx,Kry,Krz]. "
            "Default: 10000,10000,10000,1500,1500,1500."
        ),
    )
    parser.add_argument(
        "--max-linear-vel", type=float, default=0.15,
        help="Flexiv SendCartesianMotionForce max_linear_vel (m/s).",
    )
    parser.add_argument(
        "--max-linear-acc", type=float, default=0.5,
        help="Flexiv SendCartesianMotionForce max_linear_acc (m/s^2).",
    )
    args = parser.parse_args()

    if args.amplitude <= 0.0 or args.speed <= 0.0 or args.duration <= 0.0:
        parser.error("amplitude/speed/duration must be positive.")

    max_wrench = args.max_wrench if args.max_wrench is not None else DEFAULT_MAX_WRENCH
    if not all(v > 0 for v in max_wrench):
        parser.error("All max-wrench values must be positive.")
    if not all(v > 0 for v in args.impedance):
        parser.error("All impedance values must be positive.")

    robot: Optional[flexivrdk.Robot] = None
    try:
        robot = _connect(args.robot_sn)
        robot.SetForceControlAxis([False] * 6)
        robot.SetCartesianImpedance(list(args.impedance))
        logger.info(f"SetCartesianImpedance applied: {args.impedance}")
        robot.SetMaxContactWrench(list(max_wrench))
        logger.info(
            f"SetMaxContactWrench applied: {max_wrench} "
            "(N for first 3, N*m for last 3)"
        )

        states = robot.states()
        start_tcp = np.asarray(states.tcp_pose, dtype=float).reshape(-1)
        if start_tcp.size < 7:
            raise RuntimeError(f"Unexpected tcp_pose shape: {start_tcp.shape}")
        center = start_tcp[:7].copy()
        axis_idx = AXIS_INDEX[args.axis]
        omega = args.speed / args.amplitude  # rad/s of the sinusoid
        period_s = 1.0 / CONTROL_HZ

        logger.info(
            f"Center TCP={np.round(center[:3], 4).tolist()} "
            f"axis={args.axis} amplitude={args.amplitude} m speed~{args.speed} m/s "
            f"duration={args.duration} s"
        )
        logger.info(
            "Block the path with your hand to verify auto-stop on contact. "
            "Ctrl+C to abort."
        )

        last_status_t = 0.0
        last_status: Optional[object] = None
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed > args.duration:
                break

            target = center.copy()
            target[axis_idx] = center[axis_idx] + args.amplitude * np.sin(omega * elapsed)

            robot.SendCartesianMotionForce(
                target[:7].tolist(),
                [0.0] * 6,
                [0.0] * 6,
                args.max_linear_vel,
                1.0,
                args.max_linear_acc,
                5.0,
            )

            now = time.time()
            states = robot.states()

            current_status = robot.operational_status()
            if last_status is None or _status_name(current_status) != _status_name(last_status):
                from_name = _status_name(last_status) if last_status is not None else "UNKNOWN"
                to_name = _status_name(current_status)
                if to_name == "IN_REDUCED_STATE":
                    logger.warning(
                        f"contact wrench limit triggered: {from_name} -> {to_name}"
                    )
                elif from_name == "IN_REDUCED_STATE":
                    logger.info(f"contact wrench cleared: {from_name} -> {to_name}")
                else:
                    logger.debug(f"operational_status: {from_name} -> {to_name}")
                last_status = current_status

            if now - last_status_t > 0.5:
                last_status_t = now
                meas = np.asarray(states.tcp_pose, dtype=float).reshape(-1)[:3]
                cmd_pos = target[:3]
                err = cmd_pos - meas
                ext_wrench = np.asarray(states.ext_wrench_in_tcp, dtype=float).reshape(-1)
                if ext_wrench.size < 6:
                    ext_wrench = np.zeros(6)
                force_mag = float(np.linalg.norm(ext_wrench[:3]))
                torque_mag = float(np.linalg.norm(ext_wrench[3:6]))
                sys.stdout.write(
                    "\r"
                    f"t={elapsed:5.1f}s  "
                    f"cmd={np.round(cmd_pos, 4).tolist()}  "
                    f"meas={np.round(meas, 4).tolist()}  "
                    f"err={np.round(err, 4).tolist()}  "
                    f"wrench=[F={np.round(ext_wrench[:3], 2).tolist()}, "
                    f"M={np.round(ext_wrench[3:6], 2).tolist()}]  "
                    f"|F|={force_mag:5.2f}N |M|={torque_mag:4.2f}Nm  "
                    f"status={_status_name(current_status)}      "
                )
                sys.stdout.flush()

            sleep_left = period_s - (time.time() - now)
            if sleep_left > 0:
                time.sleep(sleep_left)

        sys.stdout.write("\n")
        sys.stdout.flush()
        logger.info("Test complete; returning to start pose.")
        robot.SendCartesianMotionForce(
            center[:7].tolist(), [0.0] * 6, [0.0] * 6,
            args.max_linear_vel, 1.0, args.max_linear_acc, 5.0,
        )
        time.sleep(1.0)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        logger.warning("Interrupted by user.")
    finally:
        if robot is not None:
            try:
                robot.Stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
