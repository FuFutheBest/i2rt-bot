#!/usr/bin/env python3
"""
Directly test LK grippers using the same command semantics as policy inference.

Policy sends gripper width in meters at action indices 7 (left) and 15 (right).
RizonRobot applies _scale_realtime_gripper_command() then calls LKGripper.Move().

This script skips device_starter/SHM and talks to the serial grippers directly,
using calibration files and scale_factor from the inference device config.

Usage:
  python scripts/legacy/test_policy_gripper_control.py --arm left --duration-s 5 --hz 20

Author: Jun Lv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LKTECH_SDK_ROOT = REPO_ROOT / "third_party" / "lktech_motor_sdk"
if str(LKTECH_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(LKTECH_SDK_ROOT))

from utils.config_parser import parse_device_configs
from utils.logger_config import logger
from utils.time_control import precise_loop_timing

DEFAULT_DEVICES_YAML = (
    REPO_ROOT / "configs/devices/infer/rizon_dual_eef_abs_fisheye_webcam.yaml"
)


@dataclass(frozen=True)
class GripperRuntimeConfig:
    ports: Tuple[str, str]
    motor_ids: Tuple[int, int]
    speed_dps: Tuple[float, float]
    max_vel: Tuple[float, float]
    home_width: Tuple[float, float]
    scale_factor: Tuple[float, float]
    calibration_files: Tuple[str, str]
    value_mode: str


def _resolve_config_path(path: Path, camera_left_capture_id: int, camera_right_capture_id: int) -> Dict[str, Any]:
    """Load a yaml config file and inject camera ids for ${camera.*} interpolation."""
    import yaml

    text = path.read_text(encoding="utf-8")
    text = text.replace("${camera.left_capture_id}", str(camera_left_capture_id))
    text = text.replace("${camera.right_capture_id}", str(camera_right_capture_id))
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid yaml config: {path}")
    return loaded


def _find_rizon_device_config(
    devices_yaml: Path,
    robot_device: str,
    camera_left_capture_id: int,
    camera_right_capture_id: int,
) -> Dict[str, Any]:
    """Load RizonRobot device config from the devices yaml."""
    target_id: Optional[int] = None
    if "_" in robot_device:
        suffix = robot_device.rsplit("_", 1)[-1]
        if suffix.isdigit():
            target_id = int(suffix)

    cfg_dict = _resolve_config_path(
        devices_yaml,
        camera_left_capture_id=camera_left_capture_id,
        camera_right_capture_id=camera_right_capture_id,
    )
    parsed_devices = parse_device_configs(cfg_dict)

    for entry in parsed_devices:
        if entry.get("device_class") != "RizonRobot":
            continue
        device_class_id = int(entry.get("device_class_id", -1))
        if target_id is None or device_class_id == target_id:
            config = entry.get("config")
            if isinstance(config, dict):
                return config
    raise KeyError(f"RizonRobot device not found for {robot_device!r} in {devices_yaml}")


def _parse_gripper_runtime(rizon_cfg: Dict[str, Any]) -> GripperRuntimeConfig:
    gripper = dict(rizon_cfg.get("gripper") or {})
    ports = tuple(str(port) for port in gripper.get("port", ["/dev/ttyUSB1", "/dev/ttyUSB0"]))
    motor_ids = tuple(int(value) for value in gripper.get("motor_id", [1, 1]))
    speed_dps = tuple(float(value) for value in gripper.get("speed_dps", [1000.0, 1000.0]))
    max_vel = tuple(float(value) for value in gripper.get("max_vel", [0.5, 0.5]))
    home_width = tuple(float(value) for value in gripper.get("home_width", [0.09, 0.09]))
    scale_factor = tuple(float(value) for value in gripper.get("scale_factor", [1.0, 1.0]))
    calibration_files = tuple(
        str((REPO_ROOT / path).resolve()) for path in gripper.get("calibration_files", ["", ""])
    )
    value_mode = str(gripper.get("value_mode", "width_m"))
    return GripperRuntimeConfig(
        ports=ports,
        motor_ids=motor_ids,
        speed_dps=speed_dps,
        max_vel=max_vel,
        home_width=home_width,
        scale_factor=scale_factor,
        calibration_files=calibration_files,
        value_mode=value_mode,
    )


def scale_realtime_gripper_command(gripper_value: float, scale_factor: float) -> float:
    """Mirror devices.robot.RizonRobot._scale_realtime_gripper_command for width_m mode."""
    if float(gripper_value) < 0.8:
        return float(gripper_value) * float(scale_factor)
    return float(gripper_value)


def _load_calibration_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _calibration_value(calibration: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = calibration.get(key, default)
    return default if value is None else value


def _build_direct_gripper(index: int, runtime: GripperRuntimeConfig, label: str):
    from third_party.lktech_motor_sdk.example.lktech_gripper import Gripper, LKGripperConfig

    calibration_path = runtime.calibration_files[index]
    calibration = _load_calibration_dict(calibration_path) if calibration_path else {}
    config = LKGripperConfig(
        port=runtime.ports[index],
        motor_id=runtime.motor_ids[index],
        baudrate=115200,
        speed_dps=runtime.speed_dps[index],
        home_position=1.0,
        read_timeout=0.1,
        wait_timeout=15.0,
        auto_calibrate=False,
        min_torque_iq=int(_calibration_value(calibration, "min_torque_iq", -90)),
        max_torque_iq=int(_calibration_value(calibration, "max_torque_iq", 160)),
        min_encoder=_calibration_value(calibration, "min_encoder"),
        max_encoder=_calibration_value(calibration, "max_encoder"),
        min_angle_deg=_calibration_value(calibration, "min_angle_deg"),
        max_angle_deg=_calibration_value(calibration, "max_angle_deg"),
        arm_length_m=float(_calibration_value(calibration, "arm_length_m", 0.0)),
        apex_angle_max_deg=float(_calibration_value(calibration, "apex_angle_max_deg", 90.0)),
        gear_ratio=float(_calibration_value(calibration, "gear_ratio", 1.0)),
        label=label,
    )
    gripper = Gripper(
        None,
        config=config,
        min_vel=0.0,
        max_vel=runtime.max_vel[index],
    )
    gripper.Enable(label)
    return gripper


def _resolve_arms(arm: str) -> List[Tuple[str, int]]:
    if arm == "left":
        return [("left", 0)]
    if arm == "right":
        return [("right", 1)]
    if arm == "both":
        return [("left", 0), ("right", 1)]
    raise ValueError(f"Unsupported arm selector: {arm!r}")


def _target_range_for_arm(
    args: argparse.Namespace,
    runtime: GripperRuntimeConfig,
    arm_index: int,
) -> Tuple[float, float]:
    """Resolve ramp start/end policy widths for one arm."""
    min_width = float(args.min_width)
    max_width = float(args.max_width) if args.max_width is not None else runtime.home_width[arm_index]
    return min_width, max_width


def _wait_for_motion_stop(
    gripper: Any,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> Tuple[Any, bool]:
    """Wait until the gripper reports it is no longer moving."""
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    last_state = gripper.states()
    while time.monotonic() < deadline:
        last_state = gripper.states()
        if not last_state.is_moving:
            return last_state, True
        time.sleep(poll_interval_s)
    return last_state, False


def run_direct_test(args: argparse.Namespace, runtime: GripperRuntimeConfig) -> int:
    if runtime.value_mode != "width_m":
        raise ValueError(f"Only width_m mode is supported, got {runtime.value_mode!r}")

    arms = _resolve_arms(args.arm)
    hz = max(float(args.hz), 1e-6)
    duration_s = max(float(args.duration_s), 0.0)
    sample_count = max(2, int(round(duration_s * hz)) + 1)
    wait_for_next_iteration = precise_loop_timing(1.0 / hz)
    grippers: List[Tuple[str, int, Optional[Any]]] = []

    try:
        for arm_name, arm_index in arms:
            if args.dry_run:
                grippers.append((arm_name, arm_index, None))
            else:
                logger.info(
                    f"Connecting {arm_name} gripper: port={runtime.ports[arm_index]}, "
                    f"calibration={runtime.calibration_files[arm_index]}"
                )
                grippers.append((arm_name, arm_index, _build_direct_gripper(arm_index, runtime, arm_name)))

        for arm_name, arm_index, gripper in grippers:
            min_width, _ = _target_range_for_arm(args, runtime, arm_index)
            scaled_min = scale_realtime_gripper_command(min_width, runtime.scale_factor[arm_index])
            velocity = runtime.max_vel[arm_index]
            if args.dry_run:
                logger.info(
                    f"[dry-run] pre_close arm={arm_name} "
                    f"policy_value={min_width:.4f} scaled_width_m={scaled_min:.4f}"
                )
                continue

            if gripper is None:
                raise RuntimeError("Gripper is not initialized")
            ok = gripper.Move(scaled_min, velocity)
            state, stopped = _wait_for_motion_stop(gripper, args.preclose_timeout_s)
            logger.info(
                f"pre_close arm={arm_name} cmd={min_width:.4f} scaled={scaled_min:.4f} "
                f"width={state.width:.4f} stopped={stopped} move_ok={ok}"
            )

        logger.info(
            "Sending ramp: arm={arm}, duration_s={duration:.3f}, hz={hz:.3f}, samples={samples}".format(
                arm=args.arm,
                duration=duration_s,
                hz=hz,
                samples=sample_count,
            )
        )

        for sample_idx in range(sample_count):
            alpha = sample_idx / float(sample_count - 1)
            for arm_name, arm_index, gripper in grippers:
                min_width, max_width = _target_range_for_arm(args, runtime, arm_index)
                target = min_width + (max_width - min_width) * alpha
                scaled = scale_realtime_gripper_command(target, runtime.scale_factor[arm_index])
                velocity = runtime.max_vel[arm_index]
                if args.dry_run:
                    logger.info(
                        f"[dry-run] sample={sample_idx}/{sample_count - 1} arm={arm_name} "
                        f"policy_value={target:.4f} scaled_width_m={scaled:.4f}"
                    )
                    continue

                if gripper is None:
                    raise RuntimeError("Gripper is not initialized")
                ok = gripper.Move(scaled, velocity)
                if sample_idx % max(1, int(args.log_every)) == 0 or sample_idx == sample_count - 1:
                    state = gripper.states()
                    logger.info(
                        f"sample={sample_idx}/{sample_count - 1} arm={arm_name} "
                        f"cmd={target:.4f} scaled={scaled:.4f} width={state.width:.4f} "
                        f"is_moving={state.is_moving} move_ok={ok}"
                    )

            if sample_idx < sample_count - 1:
                wait_for_next_iteration()
        return 0
    finally:
        for _, _, gripper in grippers:
            if gripper is None:
                continue
            try:
                gripper.Disable()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly test LK grippers with policy width_m command semantics."
    )
    parser.add_argument(
        "--devices-yaml",
        type=Path,
        default=DEFAULT_DEVICES_YAML,
        help="Devices yaml containing RizonRobot gripper config",
    )
    parser.add_argument("--robot-device", default="RizonRobot_2", help="Robot device name in config")
    parser.add_argument("--camera-left-capture-id", type=int, default=1, help="Resolve ${camera.left_capture_id} in devices yaml")
    parser.add_argument("--camera-right-capture-id", type=int, default=0, help="Resolve ${camera.right_capture_id} in devices yaml")
    parser.add_argument("--arm", choices=("left", "right", "both"), default="left")
    parser.add_argument("--min-width", type=float, default=0.0, help="Ramp start policy width_m")
    parser.add_argument("--max-width", type=float, default=None, help="Ramp end policy width_m; defaults to config home_width")
    parser.add_argument("--duration-s", type=float, default=5.0, help="Ramp duration in seconds")
    parser.add_argument("--hz", type=float, default=20.0, help="Command send frequency")
    parser.add_argument("--preclose-timeout-s", type=float, default=8.0, help="Max time to wait for pre-close")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N samples")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without moving hardware")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rizon_cfg = _find_rizon_device_config(
        devices_yaml=args.devices_yaml,
        robot_device=args.robot_device,
        camera_left_capture_id=args.camera_left_capture_id,
        camera_right_capture_id=args.camera_right_capture_id,
    )
    runtime = _parse_gripper_runtime(rizon_cfg)
    logger.info(
        "Gripper runtime: value_mode={value_mode}, ports={ports}, home_width={home}, scale={scale}".format(
            value_mode=runtime.value_mode,
            ports=runtime.ports,
            home=runtime.home_width,
            scale=runtime.scale_factor,
        )
    )
    return run_direct_test(args, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
