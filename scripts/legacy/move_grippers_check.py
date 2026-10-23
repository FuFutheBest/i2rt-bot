#!/usr/bin/env python3
"""
Sequentially connect the LKTech soft grippers and give each a gentle open/close
motion so a human can visually confirm which physical hand responds on which
serial port.

Ports are visited in the order given by ``--ports`` (default: /dev/ttyUSB0 then
/dev/ttyUSB1). For each port the matching per-side calibration JSON supplies the
angle range / gear ratio / torque limits, so no re-calibration sweep is needed.

Usage
-----
    python scripts/legacy/move_grippers_check.py
    python scripts/legacy/move_grippers_check.py --ports /dev/ttyUSB0 /dev/ttyUSB1

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logger_config import logger  # noqa: E402

# Default port -> calibration JSON mapping (index 0 = left, matches the device
# config's gripper.port / gripper.calibration_files [left, right] convention).
DEFAULT_PORT_TO_JSON: Dict[str, str] = {
    "/dev/ttyUSB0": "calibration/lktech_gripper/LKTech-Gripper-1_2_id1_calibration.json",
    "/dev/ttyUSB1": "calibration/lktech_gripper/LKTech-Gripper-2_2_id1_calibration.json",
}


def _load_calibration(json_rel_path: str) -> Dict[str, Any]:
    path = (REPO_ROOT / json_rel_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _move_and_wait(gripper: Any, width_m: float, velocity: float, settle_s: float) -> None:
    logger.info(f"  -> Move to {width_m*1000:.1f} mm (vel={velocity})")
    gripper.Move(width_m, velocity)
    time.sleep(settle_s)
    state = gripper.states()
    logger.info(f"     state: width={state.width*1000:.1f} mm, is_moving={state.is_moving}")


def exercise_gripper(
    port: str,
    calib: Dict[str, Any],
    *,
    velocity: float,
    settle_s: float,
    open_width_m: float,
    close_width_m: float,
) -> bool:
    from third_party.lktech_motor_sdk.example.lktech_gripper import Gripper

    label = str(calib.get("label", port))
    motor_id = int(calib.get("motor_id", 1))
    max_width = float(calib.get("max_width", 0.0925))

    # Clamp the requested motion to the calibrated physical range for safety.
    open_w = min(open_width_m, max_width)
    close_w = max(close_width_m, 0.0)

    logger.info(f"=== {label} on {port} (id={motor_id}, max_width={max_width*1000:.1f} mm) ===")
    gripper = Gripper(
        robot=None,
        port=port,
        motor_id=motor_id,
        baudrate=115200,
        speed_dps=180.0,
        auto_calibrate=False,
        min_torque_iq=int(calib.get("min_torque_iq", -40)),
        max_torque_iq=int(calib.get("max_torque_iq", 190)),
        min_angle_deg=calib.get("min_angle_deg"),
        max_angle_deg=calib.get("max_angle_deg"),
        arm_length_m=float(calib.get("arm_length_m", 0.06543)),
        apex_angle_max_deg=float(calib.get("apex_angle_max_deg", 90.0)),
        gear_ratio=float(calib.get("gear_ratio", 1.0)),
    )

    try:
        gripper.Enable(label)
        state = gripper.states()
        logger.info(f"  initial width={state.width*1000:.1f} mm")

        # Gentle open, then close, then settle to a mid position.
        _move_and_wait(gripper, open_w, velocity, settle_s)
        _move_and_wait(gripper, close_w, velocity, settle_s)
        _move_and_wait(gripper, (open_w + close_w) / 2.0, velocity, settle_s)
        logger.info(f"  {label} motion done.")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"  {label} on {port} failed: {exc}")
        return False
    finally:
        try:
            gripper.Disable()
        except Exception:  # noqa: BLE001
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequentially exercise the LKTech grippers.")
    parser.add_argument(
        "--ports",
        nargs="+",
        default=["/dev/ttyUSB0", "/dev/ttyUSB1"],
        help="Serial ports to visit, in order (default: /dev/ttyUSB0 /dev/ttyUSB1).",
    )
    parser.add_argument("--velocity", type=float, default=0.3, help="Move velocity (0-1 range, gentle).")
    parser.add_argument("--settle", type=float, default=1.5, help="Seconds to wait after each move.")
    parser.add_argument("--open-width", type=float, default=0.06, help="Open target width in meters.")
    parser.add_argument("--close-width", type=float, default=0.02, help="Close target width in meters.")
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    failures = 0
    for port in cli.ports:
        json_rel = DEFAULT_PORT_TO_JSON.get(port)
        if json_rel is None:
            logger.error(f"No calibration JSON mapping for port {port}; skipping.")
            failures += 1
            continue
        calib = _load_calibration(json_rel)
        ok = exercise_gripper(
            port,
            calib,
            velocity=cli.velocity,
            settle_s=cli.settle,
            open_width_m=cli.open_width,
            close_width_m=cli.close_width,
        )
        if not ok:
            failures += 1
        time.sleep(0.5)

    if failures:
        logger.error(f"Finished with {failures} failure(s).")
        return 1
    logger.info("All grippers exercised successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
