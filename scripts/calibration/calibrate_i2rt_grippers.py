#!/usr/bin/env python3
"""Calibrate LKTech grippers declared by the I2RT teleoperation config."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_DIR))

import calibrate_lk_gripper as calibration
from devices.robots.i2rt_teleop.config import SIDES, load_teleoperation_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate LKTech grippers from an I2RT teleoperation config."
    )
    parser.add_argument("--config", default="configs/i2rt_bimanual_teleop.yaml")
    parser.add_argument("--side", choices=("both", *SIDES), default="both")
    parser.add_argument("--speed-dps", type=float, default=180.0)
    parser.add_argument("--calibration-cycles", type=int, default=1)
    return parser


def _args(cli: argparse.Namespace, gripper_type: str) -> Namespace:
    return Namespace(
        gripper_type=gripper_type, min_torque_iq=None, max_torque_iq=None,
        arm_length_m=None, apex_angle_max_deg=None, max_width=None,
        gear_ratio=None, baudrate=115200, speed_dps=cli.speed_dps,
        read_timeout=0.1, wait_timeout=15.0, calibration_timeout=15.0,
        calibration_settle_time=0.24, calibration_cycles=cli.calibration_cycles,
        calibration_dir=None, name="I2RT-LKTech",
    )


def _write_result(
    path: Path, *, side: str, port: str, motor_id: int,
    result: tuple[float, float, float], settings: dict[str, Any],
) -> None:
    existing: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    min_angle, max_angle, gear_ratio = result
    existing.update({
        "label": existing.get("label") or f"I2RT-{side}",
        "motor_id": motor_id, "port": port, "saved_at_epoch": time.time(),
        "min_angle_deg": float(min_angle), "max_angle_deg": float(max_angle),
        "gripper_type": settings["gripper_type"],
        "min_torque_iq": settings["min_torque_iq"],
        "max_torque_iq": settings["max_torque_iq"],
        "arm_length_m": settings["arm_length_m"],
        "apex_angle_max_deg": settings["apex_angle_max_deg"],
        "max_width": settings["max_width"], "gear_ratio": float(gear_ratio),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    cli = build_parser().parse_args()
    config_path = Path(cli.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_teleoperation_config(config_path)
    selected = SIDES if cli.side == "both" else (cli.side,)
    tasks: list[tuple[str, str, int, Path, str]] = []
    for side in selected:
        gripper = config.arms[side].follower.gripper
        if gripper.backend != "lktech" or not gripper.port or gripper.calibration_file is None:
            raise RuntimeError(f"{side} gripper is not configured as LKTech with a calibration file")
        tasks.append((side, gripper.port, gripper.motor_id, gripper.calibration_file, gripper.gripper_type))

    for side, port, motor_id, path, gripper_type in tasks:
        print(f"[{side}] calibrating {port} (motor {motor_id}); writing {path}", flush=True)
        args = _args(cli, gripper_type)
        result = calibration.calibrate_gripper(args, port, motor_id, side)
        if result is None:
            return 1
        settings = calibration._resolve_calibration_settings(args)
        _write_result(path, side=side, port=port, motor_id=motor_id, result=result, settings=settings)
        print(f"[{side}] calibration written: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
