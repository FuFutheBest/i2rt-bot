#!/usr/bin/env python3
"""Interactively move an LK gripper to normalized widths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LKTECH_SDK_ROOT = REPO_ROOT / "third_party" / "lktech_motor_sdk"
if str(LKTECH_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(LKTECH_SDK_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to an LK gripper, then repeatedly read an open ratio "
            "from stdin and move the gripper there. Ratio 0=closed, 1=open."
        )
    )
    parser.add_argument("port", nargs="?", default="/dev/ttyUSB0", help="Serial port")
    parser.add_argument("min_encoder", nargs="?", type=int, default=35400, help="Closed/min encoder")
    parser.add_argument("max_encoder", nargs="?", type=int, default=55300, help="Open/max encoder")
    parser.add_argument("--id", type=int, default=1, help="Motor ID")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--speed",
        "--speed-dps",
        dest="speed_dps",
        type=float,
        default=180.0,
        help="Motion speed limit in deg/s",
    )
    parser.add_argument("--read-timeout", type=float, default=0.1, help="Serial read timeout in seconds")
    parser.add_argument("--wait-timeout", type=float, default=15.0, help="Motion wait timeout in seconds")
    return parser


def parse_width(text: str) -> float:
    width = float(text)
    if width < 0.0 or width > 1.0:
        raise ValueError("open ratio must be in [0, 1] (0=closed, 1=open)")
    return width


def main() -> int:
    args = build_parser().parse_args()

    from third_party.lktech_motor_sdk.example.lktech_gripper import Gripper

    gripper = Gripper(
        robot=None,
        port=args.port,
        motor_id=args.id,
        baudrate=args.baudrate,
        speed_dps=args.speed_dps,
        read_timeout=args.read_timeout,
        wait_timeout=args.wait_timeout,
        auto_calibrate=False,
        min_encoder=args.min_encoder,
        max_encoder=args.max_encoder,
    )

    print(
        f"Connecting to LK gripper on {args.port} "
        f"(id={args.id}, encoder={args.min_encoder}..{args.max_encoder}, "
        f"speed={args.speed_dps:.1f} dps)"
    )
    try:
        gripper.Enable("interactive-lk-gripper")
        print("Input open ratio in [0, 1] (0=closed, 1=open). Type q/quit/exit to stop.")
        while True:
            try:
                raw = input("width> ").strip()
            except EOFError:
                print()
                break

            if raw.lower() in {"q", "quit", "exit"}:
                break
            if not raw:
                continue

            try:
                width = parse_width(raw)
            except ValueError as exc:
                print(f"Invalid input: {exc}")
                continue

            ok = gripper._move_to(width, wait=True)
            state = gripper.states()
            print(f"target={width:.4f}, reached={ok}, current={state.width:.4f}")
    except KeyboardInterrupt:
        print()
        return 130
    finally:
        try:
            gripper.Disable()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
