#!/usr/bin/env python3
"""
Auto-calibrate the LKTech dual-arm grippers and write results into the correct
per-side calibration JSON, using the device config as the single source of truth.

Why this exists
---------------
`scripts/calibration/calibrate_lk_gripper.py` numbers its results purely by ``--port`` order
(``1/2``, ``2/2``) and never writes ``min/max_angle_deg`` itself, so mapping the
output back to the left/right JSON by hand is error-prone -- especially after a
USB re-enumeration flips ``/dev/ttyUSB*`` between the two hands.

This tool removes the guesswork:

* It reads the ``RizonRobot`` device's ``gripper`` block from the device config.
* ``gripper.port`` is treated as ``[left, right]`` and ``gripper.calibration_files``
  as ``[left_json, right_json]`` (same convention as ``devices/robot.py``, where
  ``robot_sn[0]`` is the left arm).
* It calibrates ``port[i]`` and writes that result into ``calibration_files[i]``,
  so ``port -> hand -> JSON`` is guaranteed consistent by construction.
* Each JSON's ``port`` field is refreshed to the port actually used, and a span
  fingerprint check flags when a freshly measured range diverges from the stored
  one (absolute angle drifts by full turns across power cycles; span should not).

The left/right <-> physical-port assignment still comes from the config (or the
``--left-port`` / ``--right-port`` overrides); no tool can infer which hand a
serial port belongs to without a human observing which gripper moves.

Usage
-----
    # Calibrate both hands using the config's gripper.port order, write JSONs
    python scripts/calibration/auto_calibrate_grippers.py

    # USB re-enumerated: override the physical ports for this run
    python scripts/calibration/auto_calibrate_grippers.py --left-port /dev/ttyUSB0 --right-port /dev/ttyUSB1

    # Only one hand, or preview without writing
    python scripts/calibration/auto_calibrate_grippers.py --side left
    python scripts/calibration/auto_calibrate_grippers.py --dry-run

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the single-gripper calibration + settings resolution logic.
import calibrate_lk_gripper as clg  # noqa: E402

DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "devices"
    / "infer"
    / "infer_rizon_fisheye_umi3_dual_openpi.yaml"
)
# gripper.port / gripper.calibration_files order convention: index 0 = left arm.
SIDE_NAMES = ("left", "right")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the dual-arm LKTech grippers and write min/max angle, "
            "gear_ratio and port into the config-referenced per-side JSON files."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Device config YAML with a RizonRobot gripper block (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
        help="Which hand(s) to calibrate. Default: both.",
    )
    parser.add_argument(
        "--left-port",
        default=None,
        help="Override the left-hand serial port (else taken from config gripper.port[0]).",
    )
    parser.add_argument(
        "--right-port",
        default=None,
        help="Override the right-hand serial port (else taken from config gripper.port[1]).",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate.")
    parser.add_argument(
        "--speed-dps",
        type=float,
        default=180.0,
        help="Calibration sweep speed limit in deg/s (kept gentle; not the runtime speed).",
    )
    parser.add_argument("--read-timeout", type=float, default=0.1, help="Serial read timeout (s).")
    parser.add_argument("--wait-timeout", type=float, default=15.0, help="Motion wait timeout (s).")
    parser.add_argument("--calibration-timeout", type=float, default=15.0, help="Per-direction calibration timeout (s).")
    parser.add_argument("--calibration-settle-time", type=float, default=0.24, help="Settle time after each sweep (s).")
    parser.add_argument(
        "--calibration-cycles",
        type=int,
        default=1,
        help="Number of complete open/close calibration cycles per gripper.",
    )
    parser.add_argument("--name", default="LKTech-Gripper", help="Display name used in logs.")
    parser.add_argument(
        "--span-tolerance",
        type=float,
        default=0.05,
        help="Relative angle-span change vs. the stored JSON above which a warning is printed.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Calibrate both hands at once. Default is sequential (safer while jaws sweep).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run calibration and print results but do NOT write any JSON file.",
    )
    return parser


def _find_gripper_block(config_path: Path) -> Dict[str, Any]:
    """Return the RizonRobot device's ``gripper`` dict from a device config YAML."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Device config not found: {config_path}")
    doc = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    devices = doc.get("devices")
    if not isinstance(devices, list):
        raise ValueError(f"No 'devices' list found in {config_path}")
    for dev in devices:
        if not isinstance(dev, dict):
            continue
        if str(dev.get("class")) == "RizonRobot" and isinstance(dev.get("gripper"), dict):
            return dev["gripper"]
    raise ValueError(f"No RizonRobot device with a 'gripper' block found in {config_path}")


def _as_pair(value: Any, field: str) -> List[Any]:
    """Coerce a config value into a 2-item [left, right] list."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value, value]
    if len(items) != 2:
        raise ValueError(f"gripper.{field} must have 2 entries [left, right], got: {value!r}")
    return items


def _build_calib_args(cli: Namespace, gripper_type: str) -> Namespace:
    """Assemble the Namespace expected by calibrate_lk_gripper.calibrate_gripper()."""
    return Namespace(
        gripper_type=gripper_type,
        min_torque_iq=None,  # -> resolved from the gripper-type preset
        max_torque_iq=None,
        arm_length_m=None,
        apex_angle_max_deg=None,
        max_width=None,
        gear_ratio=None,  # -> suggested from measured span / apex angle
        baudrate=cli.baudrate,
        speed_dps=cli.speed_dps,
        read_timeout=cli.read_timeout,
        wait_timeout=cli.wait_timeout,
        calibration_timeout=cli.calibration_timeout,
        calibration_settle_time=cli.calibration_settle_time,
        calibration_cycles=cli.calibration_cycles,
        # We persist JSON ourselves (per-side, config-driven), so keep the SDK
        # from writing its own index-named files.
        calibration_dir=None,
        name=cli.name,
    )


def _write_calibration_json(
    json_path: Path,
    *,
    side: str,
    port: str,
    motor_id: int,
    min_angle_deg: float,
    max_angle_deg: float,
    gear_ratio: float,
    settings: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    """Merge fresh calibration into an existing JSON (or create one) and write it."""
    existing: Dict[str, Any] = {}
    if json_path.is_file():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    data = dict(existing)
    data.setdefault("label", existing.get("label") or f"{side}")
    data["motor_id"] = int(motor_id)
    data["port"] = port
    data["saved_at_epoch"] = time.time()
    data.setdefault("min_encoder", existing.get("min_encoder"))
    data.setdefault("max_encoder", existing.get("max_encoder"))
    data["min_angle_deg"] = float(min_angle_deg)
    data["max_angle_deg"] = float(max_angle_deg)
    data["gripper_type"] = settings["gripper_type"]
    data["min_torque_iq"] = settings["min_torque_iq"]
    data["max_torque_iq"] = settings["max_torque_iq"]
    data["arm_length_m"] = settings["arm_length_m"]
    data["apex_angle_max_deg"] = settings["apex_angle_max_deg"]
    data["max_width"] = settings["max_width"]
    data["gear_ratio"] = float(gear_ratio)

    if not dry_run:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return data


def _old_span(json_path: Path) -> Optional[float]:
    if not json_path.is_file():
        return None
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
        lo, hi = d.get("min_angle_deg"), d.get("max_angle_deg")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return abs(float(hi) - float(lo))
    except json.JSONDecodeError:
        pass
    return None


def main() -> int:
    cli = build_parser().parse_args()
    config_path = Path(cli.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    gripper = _find_gripper_block(config_path)
    ports = [str(p) for p in _as_pair(gripper["port"], "port")]
    calib_files = [str(p) for p in _as_pair(gripper["calibration_files"], "calibration_files")]
    motor_ids = [int(m) for m in _as_pair(gripper.get("motor_id", [1, 1]), "motor_id")]
    gripper_type = str(gripper.get("gripper_type", clg.DEFAULT_LK_GRIPPER_TYPE))

    # Per-hand port overrides (handy after a USB re-enumeration).
    if cli.left_port:
        ports[0] = cli.left_port
    if cli.right_port:
        ports[1] = cli.right_port

    if ports[0] == ports[1]:
        print(
            f"ERROR: left and right resolve to the same port {ports[0]!r}; "
            "check the config or pass --left-port/--right-port.",
            file=sys.stderr,
        )
        return 2

    sides = SIDE_NAMES if cli.side == "both" else (cli.side,)

    print(f"Config: {config_path}")
    for i, side in enumerate(SIDE_NAMES):
        marker = " (selected)" if side in sides else ""
        json_path = (REPO_ROOT / calib_files[i]).resolve()
        print(f"  {side:>5}: port={ports[i]}  id={motor_ids[i]}  json={json_path.name}{marker}")
    if cli.dry_run:
        print("  [dry-run] no JSON will be written.")

    calib_args = _build_calib_args(cli, gripper_type)
    settings = clg._resolve_calibration_settings(calib_args)

    tasks = [(side, ports[SIDE_NAMES.index(side)], motor_ids[SIDE_NAMES.index(side)]) for side in sides]

    results: Dict[str, Tuple[float, float, float]] = {}
    failures = 0
    if cli.parallel and len(tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            fut = {
                ex.submit(clg.calibrate_gripper, calib_args, port, mid, side): side
                for side, port, mid in tasks
            }
            for f in as_completed(fut):
                side = fut[f]
                try:
                    r = f.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[{side}] calibration crashed: {exc}", file=sys.stderr)
                    r = None
                if r is None:
                    failures += 1
                else:
                    results[side] = r
    else:
        for side, port, mid in tasks:
            r = clg.calibrate_gripper(calib_args, port, mid, side)
            if r is None:
                failures += 1
            else:
                results[side] = r

    if failures:
        print(f"\nCalibration finished with {failures} failure(s); no JSON written.", file=sys.stderr)
        return 1

    # Write results into the config-referenced per-side JSON.
    print("\n=== Results (written to config-referenced JSON) ===")
    print(f"{'side':>5} | {'port':<13} | {'min_deg':>10} | {'max_deg':>10} | {'span':>8} | json")
    for side, port, mid in tasks:
        idx = SIDE_NAMES.index(side)
        min_a, max_a, gear = results[side]
        json_path = (REPO_ROOT / calib_files[idx]).resolve()
        prev_span = _old_span(json_path)
        new_span = abs(max_a - min_a)
        _write_calibration_json(
            json_path,
            side=side,
            port=port,
            motor_id=mid,
            min_angle_deg=min_a,
            max_angle_deg=max_a,
            gear_ratio=gear,
            settings=settings,
            dry_run=cli.dry_run,
        )
        print(
            f"{side:>5} | {port:<13} | {min_a:>10.2f} | {max_a:>10.2f} | {new_span:>8.2f} | {json_path.name}"
        )
        if prev_span is not None and prev_span > 1e-6:
            rel = abs(new_span - prev_span) / prev_span
            if rel > cli.span_tolerance:
                print(
                    f"      WARNING: {side} angle-span changed {rel*100:.1f}% "
                    f"(was {prev_span:.2f}, now {new_span:.2f}). "
                    "A large change usually means wrong port<->hand mapping or a mechanical issue.",
                    file=sys.stderr,
                )

    # Remind the user if the ports used differ from what the config stores, since
    # the runtime still reads gripper.port from the config (not from these JSONs).
    config_ports = [str(p) for p in _as_pair(gripper["port"], "port")]
    if [ports[0], ports[1]] != config_ports:
        print(
            "\nNOTE: the ports used differ from config gripper.port "
            f"({config_ports} -> [{ports[0]}, {ports[1]}]). "
            "Update gripper.port in the device config to keep [left, right] correct, "
            "then restart the device end.",
            file=sys.stderr,
        )

    if cli.dry_run:
        print("\n[dry-run] done (nothing written).")
    else:
        print("\nAll calibrations written. Restart the device end (terminal A) to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
