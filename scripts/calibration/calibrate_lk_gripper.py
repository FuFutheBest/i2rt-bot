#!/usr/bin/env python3
"""

Connect to an LK gripper, run position calibration, and print the result.

    python scripts/calibration/calibrate_lk_gripper.py --port /dev/ttyUSB0 --id 1
    python scripts/calibration/calibrate_lk_gripper.py --port /dev/ttyUSB0 /dev/ttyUSB1 --id 1
    python scripts/calibration/calibrate_lk_gripper.py --port /dev/ttyUSB0 /dev/ttyUSB1 --id 1 2
    python scripts/calibration/calibrate_lk_gripper.py --port /dev/ttyUSB1 /dev/ttyUSB0 --id 1 --gripper-type soft


"""


from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LKTECH_SDK_ROOT = REPO_ROOT / "third_party" / "lktech_motor_sdk"
if str(LKTECH_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(LKTECH_SDK_ROOT))

from utils.lk_gripper_presets import DEFAULT_LK_GRIPPER_TYPE, get_lk_gripper_preset

FIXED_GEAR_RATIO: float = 10.0
NO_LOAD_CURRENT_SAMPLE_COUNT: int = 10
NO_LOAD_CURRENT_SAMPLE_INTERVAL_S: float = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect to an LK gripper, calibrate position range, and print min/max angle values."
    )
    parser.add_argument(
        "--port",
        nargs="+",
        default=["/dev/ttyUSB0"],
        help="Serial port(s), e.g. /dev/ttyUSB0 /dev/ttyUSB1",
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        default=[1],
        help="Motor ID(s). Pass one ID for all ports, or one ID per port.",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--name", default="LKTech-Gripper", help="Display name for logs")
    parser.add_argument(
        "--gripper-type",
        default=DEFAULT_LK_GRIPPER_TYPE,
        help="LKTech gripper type preset. Currently available: soft.",
    )
    parser.add_argument("--speed-dps", type=float, default=180.0, help="Speed limit in deg/s")
    parser.add_argument("--read-timeout", type=float, default=0.1, help="Serial read timeout in seconds")
    parser.add_argument("--wait-timeout", type=float, default=15.0, help="Motion wait timeout in seconds")
    parser.add_argument(
        "--arm-length-m",
        type=float,
        default=None,
        help=(
            "Cosine-law gripper arm length in meters. Defaults to the selected "
            "--gripper-type preset."
        ),
    )
    parser.add_argument(
        "--apex-angle-max-deg",
        type=float,
        default=None,
        help=(
            "Fully-open jaw apex angle in degrees for the SDK width model. "
            "Defaults to the selected --gripper-type preset."
        ),
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=None,
        help="Optional physical maximum gripper opening in meters for metadata/output.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Optional directory where the SDK should save per-gripper calibration JSON files.",
    )
    parser.add_argument(
        "--min-torque-iq",
        type=int,
        default=None,
        help="Closing contact IQ threshold used during calibration. Defaults to gripper type preset.",
    )
    parser.add_argument(
        "--max-torque-iq",
        type=int,
        default=None,
        help="Opening contact IQ threshold used during calibration. Defaults to gripper type preset.",
    )
    parser.add_argument(
        "--calibration-timeout",
        type=float,
        default=15.0,
        help="Per-direction calibration timeout in seconds",
    )
    parser.add_argument(
        "--calibration-settle-time",
        type=float,
        default=0.24,
        help="Settle time after each calibration sweep in seconds",
    )
    parser.add_argument(
        "--calibration-cycles",
        type=int,
        default=1,
        help="Number of complete open/close calibration cycles",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Calibrate ports one by one instead of in parallel.",
    )
    return parser


def _infer_max_width(arm_length_m: float, apex_angle_max_deg: float) -> float:
    return float(2.0 * float(arm_length_m) * math.sin(math.radians(float(apex_angle_max_deg)) / 2.0))


def _infer_arm_length_m(max_width: float, apex_angle_max_deg: float) -> float:
    apex_rad = math.radians(float(apex_angle_max_deg))
    denom = 2.0 * math.sin(apex_rad / 2.0)
    if denom <= 0.0:
        raise ValueError("--apex-angle-max-deg must be positive")
    return float(max_width) / denom


def _print_no_load_current(gripper: Any, label: str) -> None:
    """Print averaged idle current before any calibration motion is commanded."""
    bus_current_samples: list[float] = []
    iq_samples: list[int] = []
    motor = gripper.motor

    try:
        for sample_index in range(NO_LOAD_CURRENT_SAMPLE_COUNT):
            bus_current_samples.append(float(motor.read_status1().current))
            iq_samples.append(int(motor.read_status2().iq))
            if sample_index + 1 < NO_LOAD_CURRENT_SAMPLE_COUNT:
                time.sleep(NO_LOAD_CURRENT_SAMPLE_INTERVAL_S)
    except Exception as exc:
        print(f"[{label}] 读取空载电流失败： {exc}", file=sys.stderr)
        return

    mean_bus_current_a = sum(bus_current_samples) / len(bus_current_samples)
    mean_iq = sum(iq_samples) / len(iq_samples)
    mean_abs_iq = sum(abs(iq) for iq in iq_samples) / len(iq_samples)
    print(
        f"[{label}] 空载电流：母线={mean_bus_current_a:.3f} A, "
        f"IQ 均值={mean_iq:+.1f} LSB, IQ 绝对值均值={mean_abs_iq:.1f} LSB, "
        f"IQ 范围=[{min(iq_samples):+d}, {max(iq_samples):+d}] LSB, "
        f"采样数={len(iq_samples)}"
    )


def _run_position_calibration(
    gripper: Any,
    calibration_cycles: int,
    label: str,
) -> tuple[float, float]:
    """Run calibration against SDK versions with or without ``cycle_count``."""
    cycle_count = int(calibration_cycles)
    if cycle_count < 1:
        raise ValueError(f"标定次数必须至少为 1，当前为 {cycle_count}")

    calibrate = gripper._calibrate_encoder_range
    parameters = inspect.signature(calibrate).parameters
    supports_cycle_count = "cycle_count" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if supports_cycle_count:
        return calibrate(cycle_count=cycle_count)

    gripper._require_enabled()
    config = gripper._base_config
    opening_contact_iq = abs(int(config.min_torque_iq))
    closing_contact_iq = abs(int(config.max_torque_iq))
    if opening_contact_iq == 0 or closing_contact_iq == 0:
        raise ValueError("标定接触检测的 IQ 阈值不能为零")

    calibration_speed_abs = max(40.0, min(150.0, abs(float(config.speed_dps)) * 0.6))
    opening_speed_dps = -calibration_speed_abs
    closing_speed_dps = calibration_speed_abs
    min_angle_samples: list[float] = []
    max_angle_samples: list[float] = []

    try:
        for cycle_index in range(cycle_count):
            print(f"[{label}] 第 {cycle_index + 1}/{cycle_count} 轮标定：张开")
            min_boundary_encoder = gripper._calibrate_single_position(
                speed_dps=opening_speed_dps,
                contact_iq_threshold=opening_contact_iq,
                timeout=float(config.calibration_timeout),
                settle_time=float(config.calibration_settle_time),
                position_name="minimum",
            )
            if min_boundary_encoder is None:
                print(
                    f"[{label}] 第 {cycle_index + 1} 轮未到达最小角度边界",
                    file=sys.stderr,
                )
            else:
                try:
                    min_angle_samples.append(float(gripper._get_stable_multi_turn_angle()))
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[{label}] 第 {cycle_index + 1} 轮最小角度采样失败： {exc}",
                        file=sys.stderr,
                    )

            print(f"[{label}] 第 {cycle_index + 1}/{cycle_count} 轮标定：闭合")
            max_boundary_encoder = gripper._calibrate_single_position(
                speed_dps=closing_speed_dps,
                contact_iq_threshold=closing_contact_iq,
                timeout=float(config.calibration_timeout),
                settle_time=float(config.calibration_settle_time),
                position_name="maximum",
            )
            if max_boundary_encoder is None:
                print(
                    f"[{label}] 第 {cycle_index + 1} 轮未到达最大角度边界",
                    file=sys.stderr,
                )
            else:
                try:
                    max_angle_samples.append(float(gripper._get_stable_multi_turn_angle()))
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[{label}] 第 {cycle_index + 1} 轮最大角度采样失败： {exc}",
                        file=sys.stderr,
                    )
    finally:
        try:
            gripper.motor.torque_control(0)
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] 停止标定力矩输出失败： {exc}", file=sys.stderr)

    if not min_angle_samples or not max_angle_samples:
        raise RuntimeError(
            "标定未能采集到两端位置 "
            f"(min={len(min_angle_samples)}, max={len(max_angle_samples)})"
        )

    min_angle_deg = float(statistics.median(min_angle_samples))
    max_angle_deg = float(statistics.median(max_angle_samples))
    if abs(max_angle_deg - min_angle_deg) < 1e-6:
        raise RuntimeError(
            f"标定范围无效： min_angle_deg={min_angle_deg}, max_angle_deg={max_angle_deg}"
        )

    gripper._min_angle_deg = min_angle_deg
    gripper._max_angle_deg = max_angle_deg
    gripper._min_encoder = None
    gripper._max_encoder = None
    config.min_encoder = None
    config.max_encoder = None
    config.min_angle_deg = min_angle_deg
    config.max_angle_deg = max_angle_deg
    gripper._save_calibration_to_file()
    return min_angle_deg, max_angle_deg


def _resolve_calibration_settings(args: argparse.Namespace) -> dict:
    preset = get_lk_gripper_preset(args.gripper_type)
    arm_length_m = (
        float(args.arm_length_m)
        if args.arm_length_m is not None
        else float(preset["arm_length_m"])
    )
    apex_angle_max_deg = (
        float(args.apex_angle_max_deg)
        if args.apex_angle_max_deg is not None
        else float(preset["apex_angle_max_deg"])
    )
    max_width = float(args.max_width) if args.max_width is not None else _infer_max_width(
        arm_length_m,
        apex_angle_max_deg,
    )
    return {
        "gripper_type": preset["gripper_type"],
        "min_torque_iq": int(args.min_torque_iq if args.min_torque_iq is not None else preset["min_torque_iq"]),
        "max_torque_iq": int(args.max_torque_iq if args.max_torque_iq is not None else preset["max_torque_iq"]),
        "arm_length_m": arm_length_m,
        "apex_angle_max_deg": apex_angle_max_deg,
        "max_width": max_width,
    }


def _calibration_file_path(args: argparse.Namespace, label: str, motor_id: int) -> Optional[Path]:
    if not args.calibration_dir:
        return None
    safe_label = label.replace("/", "_")
    return Path(args.calibration_dir).expanduser() / f"{safe_label}_id{int(motor_id)}_calibration.json"


def _update_calibration_metadata(
    args: argparse.Namespace,
    label: str,
    motor_id: int,
    settings: dict,
    gear_ratio: float,
) -> None:
    calibration_path = _calibration_file_path(args, f"{args.name}-{label}" if label else args.name, motor_id)
    if calibration_path is None or not calibration_path.is_file():
        return

    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    data.update(
        {
            "gripper_type": settings["gripper_type"],
            "min_torque_iq": settings["min_torque_iq"],
            "max_torque_iq": settings["max_torque_iq"],
            "arm_length_m": settings["arm_length_m"],
            "apex_angle_max_deg": settings["apex_angle_max_deg"],
            "max_width": settings["max_width"],
            "gear_ratio": gear_ratio,
        }
    )
    calibration_path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def calibrate_gripper(args: argparse.Namespace, port: str, motor_id: int, label: str) -> Optional[Tuple[float, float, float]]:
    from third_party.lktech_motor_sdk.example.lktech_gripper import Gripper

    settings = _resolve_calibration_settings(args)
    gripper = Gripper(
        robot=None,
        port=port,
        motor_id=motor_id,
        baudrate=args.baudrate,
        speed_dps=args.speed_dps,
        read_timeout=args.read_timeout,
        wait_timeout=args.wait_timeout,
        auto_calibrate=False,
        min_torque_iq=settings["min_torque_iq"],
        max_torque_iq=settings["max_torque_iq"],
        calibration_timeout=args.calibration_timeout,
        calibration_settle_time=args.calibration_settle_time,
        calibration_dir=args.calibration_dir,
        arm_length_m=settings["arm_length_m"],
        apex_angle_max_deg=settings["apex_angle_max_deg"],
        gear_ratio=FIXED_GEAR_RATIO,
    )

    try:
        print(f"[{label}] 正在连接 {port} 上的夹爪（电机 ID={motor_id}，波特率={args.baudrate}）")
        print(
            f"[{label}] 夹爪类型={settings['gripper_type']}, "
            f"最小力矩 IQ={settings['min_torque_iq']}, 最大力矩 IQ={settings['max_torque_iq']}, "
            f"指长（米）={settings['arm_length_m']}, 最大张角（度）={settings['apex_angle_max_deg']}"
        )
        gripper.Enable(f"{args.name}-{label}" if label else args.name)
        _print_no_load_current(gripper, label)

        print(f"[{label}] 正在进行位置标定……")
        min_angle_deg, max_angle_deg = _run_position_calibration(
            gripper,
            calibration_cycles=args.calibration_cycles,
            label=label,
        )

        print(f"[{label}] 标定完成。")
        print(f"[{label}] 最小角度（度）={min_angle_deg}")
        print(f"[{label}] 最大角度（度）={max_angle_deg}")
        angle_span_deg = abs(float(max_angle_deg) - float(min_angle_deg))
        _update_calibration_metadata(args, label, motor_id, settings, FIXED_GEAR_RATIO)
        print(f"[{label}] 角度跨度（度）={angle_span_deg}")
        print(f"[{label}] 夹爪类型={settings['gripper_type']}")
        print(f"[{label}] 最小力矩 IQ={settings['min_torque_iq']}")
        print(f"[{label}] 最大力矩 IQ={settings['max_torque_iq']}")
        print(f"[{label}] 指长（米）={settings['arm_length_m']}")
        print(f"[{label}] 最大张角（度）={settings['apex_angle_max_deg']}")
        print(f"[{label}] 最大开口（米）={settings['max_width']}")
        print(f"[{label}] 传动比={FIXED_GEAR_RATIO}")
        return float(min_angle_deg), float(max_angle_deg), FIXED_GEAR_RATIO
    except Exception as exc:
        print(f"[{label}] 标定失败： {exc}", file=sys.stderr)
        return None
    finally:
        try:
            gripper.Disable()
        except Exception:
            pass


def main() -> int:
    args = build_parser().parse_args()

    ports = list(args.port)
    motor_ids = list(args.id)
    if len(motor_ids) == 1:
        motor_ids = motor_ids * len(ports)
    elif len(motor_ids) != len(ports):
        print(
            f"--id expects either one value or one value per port; got {len(motor_ids)} IDs for {len(ports)} ports",
            file=sys.stderr,
        )
        return 2

    tasks = [
        (port, motor_id, f"{index}/{len(ports)}")
        for index, (port, motor_id) in enumerate(zip(ports, motor_ids), start=1)
    ]

    failures = 0
    results: dict[str, Tuple[float, float, float]] = {}
    if args.sequential or len(tasks) == 1:
        for port, motor_id, label in tasks:
            result = calibrate_gripper(args, port, motor_id, label)
            if result is None:
                failures += 1
            else:
                results[label] = result
    else:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {
                executor.submit(calibrate_gripper, args, port, motor_id, label): (port, motor_id, label)
                for port, motor_id, label in tasks
            }
            for future in as_completed(future_to_task):
                port, motor_id, label = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(
                        f"[{label}] Calibration crashed on {port} (id={motor_id}): {exc}",
                        file=sys.stderr,
                    )
                    result = None
                if result is None:
                    failures += 1
                else:
                    results[label] = result

    if failures:
        print(f"Calibration finished with {failures} failure(s).", file=sys.stderr)
        return 1

    ordered_results = [results[label] for _, _, label in tasks]
    settings = _resolve_calibration_settings(args)
    print("Calibration YAML snippet:")
    print("gripper:")
    print(f"  gripper_type: \"{settings['gripper_type']}\"")
    print(f"  min_torque_iq: {settings['min_torque_iq']}")
    print(f"  max_torque_iq: {settings['max_torque_iq']}")
    print(f"  min_angle_deg: {[result[0] for result in ordered_results]}")
    print(f"  max_angle_deg: {[result[1] for result in ordered_results]}")
    print(f"  arm_length_m: {settings['arm_length_m']}")
    print(f"  apex_angle_max_deg: {settings['apex_angle_max_deg']}")
    print(f"  max_width: {settings['max_width']}")
    print(f"  gear_ratio: {[result[2] for result in ordered_results]}")
    print("All calibrations complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
