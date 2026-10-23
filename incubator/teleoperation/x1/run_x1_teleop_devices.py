#!/usr/bin/env python3
"""
Standalone launcher for the X1 teleop device pair (no hydra / device_manager).

Brings up ``X1Robot`` and ``X1VuerTeleoperator`` as two separate processes so
the VR -> SHM -> X1 control loop can be tested before the full device_starter
(hydra) path is available in the environment.

Run INSIDE the robot_test container (nmxrdk + vuer available):
    cd /root/SuperInference
    python incubator/teleoperation/x1/run_x1_teleop_devices.py --arm-side left

Then open the Vuer page on the Quest:
    https://<host-LAN-ip>:8012   (accept the self-signed cert) -> Enter VR

SAFETY: this drives the real robot. Start with --arm-side left, keep the
e-stop reachable, and verify axis alignment before using both arms / grippers.

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Make the SuperInference project root importable (this file lives under
# <root>/incubator/teleoperation/x1/).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _run_robot(kwargs: Dict[str, Any]) -> None:
    from devices.robot import X1Robot

    device = X1Robot(**kwargs)
    try:
        device.start_server()
    except KeyboardInterrupt:
        pass
    finally:
        device.stop_server()


def _run_teleop(kwargs: Dict[str, Any]) -> None:
    from incubator.teleoperation.x1.x1_teleoperator import X1VuerTeleoperator

    device = X1VuerTeleoperator(**kwargs)
    try:
        device.start_server()
    except KeyboardInterrupt:
        pass
    finally:
        device.stop_server()


def _run_camera(kwargs: Dict[str, Any]) -> None:
    from incubator.teleoperation.x1.tj_camera import TjCameraDevice

    device = TjCameraDevice(**kwargs)
    try:
        device.start_server()
    except KeyboardInterrupt:
        pass
    finally:
        device.stop_server()


def _run_device_manager(config: Dict[str, Any]) -> None:
    from manager.base_device_manager import BaseDeviceManager

    manager = BaseDeviceManager(config=config)
    try:
        manager.start()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            manager.stop()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--robot-id", default="1e783be2dd870ef5")
    p.add_argument("--arm-side", choices=("left", "right", "both"), default="left")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--no-https", action="store_true",
                   help="serve HTTP (use ngrok) instead of direct HTTPS")
    p.add_argument("--webrtc-src", default="",
                   help="WebRTC video offer endpoint from tj_camera_webrtc.py")
    p.add_argument("--no-grippers", action="store_true",
                   help="omit gripper primitives from the X1 command stream")
    p.add_argument("--no-head", action="store_true",
                   help="omit head primitives from the X1 command stream")
    p.add_argument("--no-home", action="store_true",
                   help="skip joint-space homing on startup")
    p.add_argument("--move-vel", type=float, default=0.3)
    p.add_argument("--workspace-radius", type=float, default=0.25)
    p.add_argument("--smooth", type=float, default=0.4)
    p.add_argument("--lock-orientation", action="store_true")
    p.add_argument("--fixed-frame", action="store_true")
    # ---- data collection ----
    p.add_argument("--collect", action="store_true",
                   help="also start the camera devices + device_manager so the "
                        "summary SHM (device_summary_data) is populated for the "
                        "NEDF3 data saver.")
    p.add_argument("--head-cam-id", default="CAMID_0",
                   help="nmxrdk CAMID for the head/ego camera (real).")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=float, default=30.0)
    return p.parse_args()


def _camera_specs(args: argparse.Namespace):
    """device_class_id -> role/cam_id. Mirrors R1's camera_0/1/2 layout.

    Wrist cameras are not installed yet (placeholder CAMIDs) -> TjCameraDevice
    publishes black frames + a warning, so the pipeline still runs end-to-end.
    """
    return [
        dict(device_class_id=0, cam_id="CAMID_1", role="inhand_right"),
        dict(device_class_id=1, cam_id="CAMID_2", role="inhand_left"),
        dict(device_class_id=2, cam_id=args.head_cam_id, role="head"),
    ]


_X1_PRIMITIVE_WIDTHS = {
    "xyz": 3,
    "qxqyqzqw": 4,
    "gripper": 1,
    "j1-2": 2,
}


def _x1_command_types(*, grippers: bool, head: bool) -> list[str]:
    """Return the shared arm/head layout used by robot and teleoperator."""
    types: list[str] = []
    for _ in range(2):
        types.extend(("xyz", "qxqyqzqw"))
        if grippers:
            types.append("gripper")
    if head:
        types.append("j1-2")
    return types


def _x1_command_format(types: list[str]) -> dict[str, Any]:
    return {
        "shape": [sum(_X1_PRIMITIVE_WIDTHS[name] for name in types)],
        "dtype": "float64",
        "type": list(types),
    }


def _x1_robot_state_feedback() -> dict[str, Any]:
    return {
        "fps": 60.0,
        "buffer_size": 1000,
        "hardware_latency_ms": 0.0,
        "format": {
            "shape": [30],
            "dtype": "float64",
            "type": [
                "j1-7",
                "xyz",
                "qwqxqyqz",
                "j1-7",
                "xyz",
                "qwqxqyqz",
                "j1-2",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    enable_grippers = not args.no_grippers
    enable_head = not args.no_head
    command_types = _x1_command_types(
        grippers=enable_grippers,
        head=enable_head,
    )
    command_format = _x1_command_format(command_types)
    robot_state_feedback = _x1_robot_state_feedback()
    robot_control = {
        "type": "eef_abs",
        "source_selection": {
            "mode": "single_source",
            "source": {
                "device": "X1VuerTeleoperator",
                "device_class_id": 0,
                "port": "state",
            },
        },
        "execution": {"mode": "synchronous", "fps": 15.0},
        "buffer_size": 100,
        "format": command_format,
    }
    teleop_state_feedback = {
        "fps": 50.0,
        "buffer_size": 100,
        "hardware_latency_ms": 0.0,
        "format": command_format,
    }

    robot_kwargs: Dict[str, Any] = dict(
        device_class_id=1,
        robot_id=args.robot_id,
        state_feedback=robot_state_feedback,
        control=robot_control,
        motion_limits={"velocity": {"tcp_linear_m_s": args.move_vel}},
        home_on_start=not args.no_home,
    )
    teleop_kwargs: Dict[str, Any] = dict(
        device_class_id=0,
        state_feedback=teleop_state_feedback,
        robot_device_name="X1Robot",
        robot_device_class_id=1,
        arm_side=args.arm_side,
        port=args.port,
        https=not args.no_https,
        webrtc_src=args.webrtc_src,
        enable_head=enable_head,
        enable_grippers=enable_grippers,
        workspace_radius=args.workspace_radius,
        smooth=args.smooth,
        lock_orientation=args.lock_orientation,
        fixed_frame=args.fixed_frame,
    )

    if args.collect:
        teleop_kwargs["enable_episode_trigger"] = True

    ctx = mp.get_context("spawn")
    p_robot = ctx.Process(target=_run_robot, args=(robot_kwargs,), name="X1Robot")
    p_teleop = ctx.Process(target=_run_teleop, args=(teleop_kwargs,), name="X1VuerTeleoperator")

    # Ordered list of processes (start order); shutdown happens in reverse.
    procs = [p_robot]
    cam_procs = []
    p_dm = None

    if args.collect:
        for spec in _camera_specs(args):
            cam_kwargs = dict(
                device_class_id=spec["device_class_id"],
                robot_id=args.robot_id,
                cam_id=spec["cam_id"],
                role=spec["role"],
                state_feedback={
                    "fps": args.cam_fps,
                    "buffer_size": 100,
                    "hardware_latency_ms": 0.0,
                    "format": {
                        "shape": [args.cam_height, args.cam_width, 3],
                        "dtype": "uint8",
                        "type": "rgb",
                    },
                },
            )
            cp = ctx.Process(
                target=_run_camera, args=(cam_kwargs,),
                name=f"TjCameraDevice_{spec['device_class_id']}",
            )
            cam_procs.append(cp)

    print(f"[launch] starting X1Robot (arm_side={args.arm_side}, "
          f"grippers={enable_grippers}, head={enable_head}) ...")
    p_robot.start()
    # Give the robot a head start so it homes and publishes its state SHM first.
    time.sleep(2.0)

    if args.collect:
        for cp, spec in zip(cam_procs, _camera_specs(args)):
            print(f"[launch] starting TjCameraDevice_{spec['device_class_id']} "
                  f"(cam_id={spec['cam_id']}, role={spec['role']}) ...")
            cp.start()
        procs.extend(cam_procs)

    print(f"[launch] starting X1VuerTeleoperator on "
          f"{'https' if not args.no_https else 'http'}://<host>:{args.port} ...")
    p_teleop.start()
    procs.append(p_teleop)

    if args.collect:
        # Let all device SHMs come up before the manager connects to them.
        time.sleep(3.0)
        dm_config = {
            "summary_shm": "device_summary_data",
            "devices": [
                {"class": "X1Robot", "device_class_id": 1,
                 "state_feedback": robot_state_feedback,
                 "control": robot_control},
                {"class": "X1VuerTeleoperator", "device_class_id": 0,
                 "state_feedback": teleop_state_feedback,
                 "enable_head": enable_head,
                 "enable_grippers": enable_grippers},
                {"class": "TjCameraDevice", "device_class_id": 0,
                 "state_feedback": {"fps": args.cam_fps, "buffer_size": 100,
                                    "hardware_latency_ms": 0.0,
                                    "format": {"shape": [args.cam_height, args.cam_width, 3],
                                               "dtype": "uint8", "type": "rgb"}}},
                {"class": "TjCameraDevice", "device_class_id": 1,
                 "state_feedback": {"fps": args.cam_fps, "buffer_size": 100,
                                    "hardware_latency_ms": 0.0,
                                    "format": {"shape": [args.cam_height, args.cam_width, 3],
                                               "dtype": "uint8", "type": "rgb"}}},
                {"class": "TjCameraDevice", "device_class_id": 2,
                 "state_feedback": {"fps": args.cam_fps, "buffer_size": 100,
                                    "hardware_latency_ms": 0.0,
                                    "format": {"shape": [args.cam_height, args.cam_width, 3],
                                               "dtype": "uint8", "type": "rgb"}}},
            ],
        }
        p_dm = ctx.Process(
            target=_run_device_manager, args=(dm_config,), name="DeviceManager",
        )
        print("[launch] starting DeviceManager (-> device_summary_data) ...")
        p_dm.start()
        procs.append(p_dm)
        print("[launch] data collection ready: summary SHM = device_summary_data")
        print("[launch] FPV: python incubator/teleoperation/x1/tj_camera_webrtc.py "
              "--https --shm-name TjCameraDevice_2_data")

    stop = {"flag": False}

    def _shutdown(signum: int, frame: Any) -> None:
        if stop["flag"]:
            return
        stop["flag"] = True
        print("\n[launch] shutting down ...")
        for proc in reversed(procs):
            if proc.is_alive():
                proc.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[launch] all processes up ({', '.join(p.name for p in procs)}); "
          f"press Ctrl+C to stop.")
    try:
        while True:
            dead = [p for p in procs if not p.is_alive()]
            if dead:
                for p in dead:
                    print(f"[launch] process '{p.name}' exited unexpectedly "
                          f"(exitcode={p.exitcode}); shutting everything down.")
                break
            time.sleep(0.5)
    finally:
        _shutdown(signal.SIGINT, None)
        for proc in reversed(procs):
            try:
                proc.join(timeout=10.0)
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
    print("[launch] done")


if __name__ == "__main__":
    main()
