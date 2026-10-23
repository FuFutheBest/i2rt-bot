#!/usr/bin/env python3
"""
Reset the Flexiv Rizon TCP force/torque sensor offset to zero via the
``ZeroFTSensor`` primitive.

Useful when ``ext_wrench_in_tcp`` shows a non-trivial residual force (e.g.
``|F| ~ 15-20 N``) even though nothing is in contact with the robot. The
residual usually comes from an inaccurate tool mass / center of mass
configuration in Flexiv Elements; this primitive samples the current
"static" wrench and re-defines it as the new zero point.

Workflow:
  1. Power on the robot, clear any fault, and enable it.
  2. Move the arm to a quiet, mid-range pose (avoid extreme reach).
  3. Make sure the end-effector is NOT touching anything.
  4. Run this script. It prints the F/T reading before and after zeroing.

Caveat: the zero offset is only correct for the current pose / tool
weight. If the tool mass / CoM in Elements is wrong, the bias will
re-appear at other poses. A permanent fix requires configuring the tool
correctly in Flexiv Elements (mass + CoM with < 100g error) and setting
it as the active tool.

Usage:
    python scripts/calibration/zero_ft_sensor.py --robot-sn Rizon4-063114
    python scripts/calibration/zero_ft_sensor.py --robot-sn Rizon4-063114 \
        --data-collect-time 2.0 --settle 1.5 --probe-seconds 5


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
    return robot


def _read_ext_wrench(robot: flexivrdk.Robot) -> np.ndarray:
    states = robot.states()
    arr = np.asarray(states.ext_wrench_in_tcp, dtype=float).reshape(-1)
    if arr.size < 6:
        arr = np.concatenate([arr, np.zeros(6 - arr.size)])
    return arr[:6]


def _fmt_wrench(w: np.ndarray) -> str:
    f = w[:3]
    m = w[3:]
    f_mag = float(np.linalg.norm(f))
    m_mag = float(np.linalg.norm(m))
    return (
        f"F=[{f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f}] N "
        f"M=[{m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f}] Nm "
        f"|F|={f_mag:.2f}N |M|={m_mag:.2f}Nm"
    )


def _probe(robot: flexivrdk.Robot, seconds: float) -> np.ndarray:
    """Sample ext_wrench_in_tcp for ``seconds`` and return the mean."""
    if seconds <= 0.0:
        return _read_ext_wrench(robot)
    samples: List[np.ndarray] = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        samples.append(_read_ext_wrench(robot))
        time.sleep(0.05)
    if not samples:
        return _read_ext_wrench(robot)
    return np.mean(np.stack(samples), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Flexiv's ZeroFTSensor primitive to reset the TCP wrench "
            "offset. The robot must be static and not in contact."
        )
    )
    parser.add_argument("--robot-sn", required=True, help="Robot serial number.")
    parser.add_argument(
        "--settle", type=float, default=1.0,
        help="Seconds to wait before sampling (lets the arm fully settle).",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="Max seconds to wait for the primitive to terminate.",
    )
    parser.add_argument(
        "--data-collect-time", type=float, default=None,
        help=(
            "Optional override for the primitive's dataCollectTime "
            "parameter (seconds)."
        ),
    )
    parser.add_argument(
        "--probe-seconds", type=float, default=3.0,
        help=(
            "Idle wrench probe duration before and after zeroing, used "
            "to confirm that the residual force dropped."
        ),
    )
    args = parser.parse_args()

    if args.settle < 0.0:
        parser.error("--settle must be >= 0.")
    if args.timeout <= 0.0:
        parser.error("--timeout must be > 0.")
    if args.probe_seconds < 0.0:
        parser.error("--probe-seconds must be >= 0.")

    robot: Optional[flexivrdk.Robot] = None
    try:
        robot = _connect(args.robot_sn)

        logger.info("Probing idle wrench BEFORE zeroing...")
        before = _probe(robot, args.probe_seconds)
        logger.info(f"BEFORE: mean {_fmt_wrench(before)}")

        logger.warning(
            "About to run ZeroFTSensor. Make sure NOTHING is in contact "
            f"with the robot. Sampling in {args.settle:.1f}s..."
        )
        robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        time.sleep(args.settle)

        params: dict = {}
        if args.data_collect_time is not None:
            params["dataCollectTime"] = float(args.data_collect_time)
        robot.ExecutePrimitive("ZeroFTSensor", params)

        deadline = time.time() + args.timeout
        ok = False
        while time.time() < deadline:
            try:
                states = robot.primitive_states()
                terminated = None
                if hasattr(states, "get"):
                    terminated = states.get("terminated")
                if terminated is None:
                    try:
                        terminated = states["terminated"]
                    except Exception:
                        terminated = None
                if bool(terminated):
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.1)

        if not ok:
            logger.error(
                f"ZeroFTSensor primitive did NOT terminate within {args.timeout}s. "
                "The sensor offset may not have been updated."
            )
            sys.exit(1)

        logger.info("ZeroFTSensor finished. Probing idle wrench AFTER zeroing...")
        after = _probe(robot, args.probe_seconds)
        logger.info(f"AFTER : mean {_fmt_wrench(after)}")

        f_drop = float(np.linalg.norm(before[:3]) - np.linalg.norm(after[:3]))
        if np.abs(after[:3]).max() < 2.0:
            logger.info(
                f"OK: residual force below 2N (drop ~{f_drop:.2f}N). "
                "ext_wrench_in_tcp now reflects true external contact."
            )
        else:
            logger.warning(
                f"Residual force is still {np.linalg.norm(after[:3]):.2f}N "
                f"(drop ~{f_drop:.2f}N). Consider configuring tool mass + CoM "
                "in Flexiv Elements; ZeroFTSensor alone cannot fully fix a "
                "wrong tool model."
            )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    finally:
        if robot is not None:
            try:
                robot.Stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
