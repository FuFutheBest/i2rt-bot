#!/usr/bin/env python3
"""VR 遥操作延迟排查探针。

目的：把"控制器/机器人侧延迟"和"网络/输入侧延迟"分开测，定位瓶颈。

三种模式（用 --mode 选）：

  input  —— 只跑 Vuer viewer，测「输入/网络」侧：
            手柄数据到达频率(Hz)、数据新鲜度(age, ms)、是否真的在更新。
            不连机器人，安全。挥动手柄观察数值。

  robot  —— 不开 VR，测「控制/机器人」侧：
            默认安全：反复对【当前位姿】调用 move_line(零运动)，测纯调用/通信往返耗时。
            加 --motion --execute：发小幅正弦 TCP 目标，测调用耗时 + 实际跟踪误差/滞后。

  full   —— 完整遥操作 + 全链路打点：同时报告输入频率/新鲜度、控制环频率、
            move_line 调用耗时、相机取帧耗时。用来在真实遥操作中看各段占比。

在 robot_test 容器里运行（venv）：
    source /root/tvenv/bin/activate
    cd /root/incubator/teleoperation/x1

    # 1) 先单独测网络/输入（挥动手柄）：
    python3 tj_teleop_latency_probe.py --mode input

    # 2) 再单独测机器人/控制（安全零运动调用延迟）：
    python3 tj_teleop_latency_probe.py --mode robot --arm-side left

    # 3) 机器人侧带运动的跟踪测试（小幅正弦，会动臂）：
    python3 tj_teleop_latency_probe.py --mode robot --arm-side left --motion --execute

    # 4) 真实遥操作全链路打点：
    python3 tj_teleop_latency_probe.py --mode full --arm-side both --no-home
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from multiprocessing import Value, shared_memory

import numpy as np


# ============================================================
# 统计累加器
# ============================================================
class Stat:
    """区间统计：count/avg/max/p95（毫秒或任意单位）。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._vals: list[float] = []

    def add(self, v: float):
        self._vals.append(v)

    def summary(self) -> str:
        if not self._vals:
            return "n=0"
        arr = np.array(self._vals)
        p95 = float(np.percentile(arr, 95))
        return (f"n={len(arr)} avg={arr.mean():.1f} "
                f"max={arr.max():.1f} p95={p95:.1f}")

    @property
    def n(self) -> int:
        return len(self._vals)


# ============================================================
# 共享内存（viewer 需要一个图像 shm 才能启动）
# ============================================================
def make_shm(resolution):
    img_shape = (resolution[0], 2 * resolution[1], 3)
    shm = shared_memory.SharedMemory(
        create=True, size=int(np.prod(img_shape)) * np.uint8().itemsize)
    return shm, img_shape


# ============================================================
# mode: input —— 输入/网络侧
# ============================================================
def resolve_certs(args):
    """返回 (cert, key) 路径或 (None, None)。--https 时默认用 certs/ 下的自签证书。"""
    if not args.https:
        return None, None
    here = Path(__file__).resolve().parent
    cert = args.cert or str(here / "certs" / "cert.pem")
    key = args.key or str(here / "certs" / "key.pem")
    if not Path(cert).exists() or not Path(key).exists():
        raise FileNotFoundError(
            f"证书不存在: {cert} / {key}。请先生成自签证书或用 --cert/--key 指定。")
    return cert, key


def build_instrumented_viewer(shm_name, img_shape, port, stream_camera,
                              cert=None, key=None):
    """延迟导入依赖 vuer 的模块，并返回一个带时间戳的 viewer 实例。"""
    # tj_controller_teleop 在 import 时会装好 Q3_DIR 路径并加载 vuer
    from tj_controller_teleop import TJControllerViewer

    class InstrumentedViewer(TJControllerViewer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # 跨进程共享：最近一次 HAND_MOVE 的墙钟时间 + 累计计数
            self.hand_update_t = Value("d", 0.0, lock=True)
            self.hand_update_count = Value("i", 0, lock=True)

        async def on_hand_move(self, event, session, fps=60):
            now = time.time()
            with self.hand_update_t.get_lock():
                self.hand_update_t.value = now
            with self.hand_update_count.get_lock():
                self.hand_update_count.value += 1
            await super().on_hand_move(event, session, fps)

    return InstrumentedViewer(shm_name, img_shape, port=port,
                              stream_camera=stream_camera, cert=cert, key=key)


def run_input(args) -> int:
    resolution = (540, 960)
    shm, img_shape = make_shm(resolution)
    try:
        from tj_controller_teleop import patch_vuer, install_nocache_patch
        patch_vuer()
        install_nocache_patch()

        cert, key = resolve_certs(args)
        viewer = build_instrumented_viewer(shm.name, img_shape, args.port,
                                            stream_camera=False, cert=cert, key=key)
        viewer.start()
        if cert:
            print(f"[input] Vuer HTTPS 直连，port={args.port}。Quest 打开 "
                  f"https://<主机LAN_IP>:{args.port} -> 接受证书 -> Enter VR，然后【挥动手柄】。")
        else:
            print(f"[input] Vuer 起来了，port={args.port}。host 上 `ngrok http {args.port}`，"
                  f"Quest 打开 ngrok https 地址 -> Enter VR，然后【挥动手柄】观察下列数值。")
        print("[input] 解读：update_rate 应接近手柄上报率(~60-90Hz)；"
              "age 应是个位数 ms。若 rate 低/age 大 -> 网络/隧道是瓶颈。")

        last_count = 0
        last_t = time.time()
        last_mat = None
        report_interval = 1.0 / max(0.2, args.report_hz)
        t_end = time.time() + args.duration if args.duration > 0 else None

        while True:
            time.sleep(report_interval)
            now = time.time()
            with viewer.hand_update_count.get_lock():
                cnt = viewer.hand_update_count.value
            with viewer.hand_update_t.get_lock():
                last_update_t = viewer.hand_update_t.value

            dt = now - last_t
            rate = (cnt - last_count) / dt if dt > 0 else 0.0
            age_ms = (now - last_update_t) * 1000.0 if last_update_t > 0 else float("nan")

            # 数据是否真的在变（区分"有事件但矩阵没动"）
            mat = viewer.right_controller_matrix
            if not viewer.right_controller_visible:
                mat = viewer.left_controller_matrix
            changed = "?"
            if last_mat is not None:
                changed = "yes" if not np.allclose(mat, last_mat, atol=1e-6) else "no"
            last_mat = mat.copy()

            vis_l = viewer.left_controller_visible
            vis_r = viewer.right_controller_visible
            print(f"[input] update_rate={rate:5.1f}Hz  age={age_ms:6.1f}ms  "
                  f"matrix_changed={changed}  visible(L/R)={int(vis_l)}/{int(vis_r)}  "
                  f"total_events={cnt}")

            last_count, last_t = cnt, now
            if t_end and now >= t_end:
                break
    finally:
        shm.close()
        shm.unlink()
    return 0


# ============================================================
# mode: robot —— 控制/机器人侧
# ============================================================
def _pose7(arm):
    import nmxrt
    s = arm.get_tcp_state()
    if isinstance(s, nmxrt.Error):
        raise RuntimeError(f"get_tcp_state failed: {s.message}")
    p = s.pose
    return [p.pos.x, p.pos.y, p.pos.z, p.quat.w, p.quat.x, p.quat.y, p.quat.z]


def run_robot(args) -> int:
    import nmxrdk
    import nmxrt
    from nmxrdk.common.robot_side import RobotSide

    side_enum = RobotSide.LEFT if args.arm_side == "left" else RobotSide.RIGHT
    robot = nmxrdk.RobotClient(args.robot_id)
    arm = robot.arm(side_enum)

    base = _pose7(arm)
    print(f"[robot] side={args.arm_side} base TCP={[round(v,4) for v in base]}")
    if args.motion and not args.execute:
        print("[robot] --motion 需要配合 --execute 才会真正发运动。仅做零运动调用延迟测试。")
    motion = args.motion and args.execute
    if motion:
        print(f"[robot] 运动测试：轴={args.axis} 幅值={args.amp}m 频率={args.freq}Hz "
              f"move_vel={args.move_vel} 发送率≈{args.rate}Hz")
    else:
        print(f"[robot] 安全零运动测试：对当前位姿反复 move_line，发送率≈{args.rate}Hz")

    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    call_stat = Stat()       # move_line 调用耗时(ms)
    read_stat = Stat()       # get_tcp_state 读取耗时(ms)
    err_stat = Stat()        # 跟踪误差(mm, 仅 motion)
    period = 1.0 / max(1.0, args.rate)
    report_interval = 1.0 / max(0.2, args.report_hz)

    t0 = time.time()
    last_report = t0
    sent = 0
    t_end = t0 + args.duration if args.duration > 0 else None

    try:
        while True:
            tick = time.time()
            target = list(base)
            if motion:
                target[axis_idx] = base[axis_idx] + args.amp * math.sin(
                    2.0 * math.pi * args.freq * (tick - t0))

            # 计时：move_line 调用本身（非阻塞），反映 SDK/通信往返成本
            tc = time.time()
            res = arm.move_line(list(target), args.move_vel, False)
            call_ms = (time.time() - tc) * 1000.0
            call_stat.add(call_ms)
            sent += 1
            if isinstance(res, nmxrt.Error):
                print(f"[robot] move_line err: {res.message}")

            # 计时：状态读取往返
            tr = time.time()
            meas = _pose7(arm)
            read_stat.add((time.time() - tr) * 1000.0)

            if motion:
                err_mm = abs(meas[axis_idx] - target[axis_idx]) * 1000.0
                err_stat.add(err_mm)

            now = time.time()
            if now - last_report >= report_interval:
                eff_rate = sent / (now - last_report)
                line = (f"[robot] send_rate={eff_rate:5.1f}Hz  "
                        f"move_line_ms[{call_stat.summary()}]  "
                        f"read_ms[{read_stat.summary()}]")
                if motion:
                    line += f"  track_err_mm[{err_stat.summary()}]"
                print(line)
                call_stat.reset(); read_stat.reset(); err_stat.reset()
                sent = 0
                last_report = now

            if t_end and now >= t_end:
                break

            sleep = period - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[robot] stop")
    finally:
        if motion:
            try:
                arm.move_line(list(base), args.move_vel, False)
            except Exception:
                pass

    print("[robot] 解读：move_line_ms 若偏大(几十~上百 ms) -> 控制器/通信是瓶颈，"
          "且会随 send_hz 叠加；track_err 大或滞后 -> 机器人执行端跟不上。")
    return 0


# ============================================================
# mode: full —— 完整遥操作 + 全链路打点
# ============================================================
class TimedArm:
    """包一层，给 move_line / get_tcp_state 计时。"""

    def __init__(self, arm, call_stat: Stat):
        self._arm = arm
        self._call_stat = call_stat

    def __getattr__(self, name):
        return getattr(self._arm, name)

    def move_line(self, *a, **k):
        t = time.time()
        r = self._arm.move_line(*a, **k)
        self._call_stat.add((time.time() - t) * 1000.0)
        return r


def run_full(args) -> int:
    import threading
    import cv2
    import nmxrdk
    import nmxrt
    from nmxrdk.common.robot_side import RobotSide
    from tj_controller_teleop import (
        patch_vuer, install_nocache_patch, ArmController,
        HEAD_CAMERA_ID, RESOLUTION,
    )
    import teleop_home

    patch_vuer()
    install_nocache_patch()

    resolution = RESOLUTION
    shm, img_shape = make_shm(resolution)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)

    cert, key = resolve_certs(args)
    viewer = build_instrumented_viewer(shm.name, img_shape, args.port,
                                        stream_camera=not args.no_camera,
                                        cert=cert, key=key)
    viewer.start()
    if cert:
        print(f"[full] Vuer HTTPS 直连，port={args.port}。Quest 打开 "
              f"https://<主机LAN_IP>:{args.port} 并进入 VR。")
    else:
        print(f"[full] Vuer 起来了，port={args.port}。host 上 `ngrok http {args.port}` 并进入 VR。")

    robot = nmxrdk.RobotClient(args.robot_id)
    cam = robot.camera(HEAD_CAMERA_ID) if not args.no_camera else None

    if not args.no_home:
        teleop_home.go_home(robot, arm_side="both", do_head=True, blocking=True)

    move_stat = Stat()
    cam_stat = Stat()

    arms = []
    if args.arm_side in ("left", "both"):
        arms.append(ArmController(TimedArm(robot.arm(RobotSide.LEFT), move_stat),
                                  "left", "left", args))
    if args.arm_side in ("right", "both"):
        arms.append(ArmController(TimedArm(robot.arm(RobotSide.RIGHT), move_stat),
                                  "right", "right", args))

    cam_stop = threading.Event()

    def camera_loop():
        res = resolution
        while not cam_stop.is_set():
            try:
                tc = time.time()
                rgb_frame, _ = cam.get_rgbd(decode=True)
                cam_stat.add((time.time() - tc) * 1000.0)
                if not isinstance(rgb_frame, nmxrt.Error):
                    rgb = rgb_frame.get("data")
                    if rgb is not None:
                        rgb = cv2.resize(rgb, (res[1], res[0]))
                        if rgb.shape[2] == 3:
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        np.copyto(img_array, np.hstack((rgb, rgb)))
            except Exception:
                pass
            time.sleep(0.01)

    cam_thread = None
    if cam is not None:
        cam_thread = threading.Thread(target=camera_loop, daemon=True)
        cam_thread.start()

    last_count, last_t = 0, time.time()
    last_report = time.time()
    loop_stat = Stat()
    report_interval = 1.0 / max(0.2, args.report_hz)
    dt = args.dt

    print("[full] 解读：input_rate/age=网络侧；loop_rate=控制环；"
          "move_line_ms=控制器调用；cam_ms=视频取帧。哪段大就是哪段的锅。")
    try:
        while True:
            t_loop = time.time()
            for ctrl in arms:
                ctrl.update(viewer)
            loop_stat.add((time.time() - t_loop) * 1000.0)

            now = time.time()
            if now - last_report >= report_interval:
                with viewer.hand_update_count.get_lock():
                    cnt = viewer.hand_update_count.value
                with viewer.hand_update_t.get_lock():
                    last_update_t = viewer.hand_update_t.value
                drep = now - last_report
                in_rate = (cnt - last_count) / drep if drep > 0 else 0.0
                age_ms = (now - last_update_t) * 1000.0 if last_update_t > 0 else float("nan")
                loop_rate = loop_stat.n / drep if drep > 0 else 0.0
                print(f"[full] input_rate={in_rate:5.1f}Hz age={age_ms:6.1f}ms | "
                      f"loop_rate={loop_rate:5.1f}Hz loop_ms[{loop_stat.summary()}] | "
                      f"move_line_ms[{move_stat.summary()}] | cam_ms[{cam_stat.summary()}]")
                loop_stat.reset(); move_stat.reset(); cam_stat.reset()
                last_count, last_report = cnt, now

            sleep = dt - (time.time() - t_loop)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[full] stop")
    finally:
        cam_stop.set()
        if cam_thread is not None:
            cam_thread.join(timeout=1.0)
        shm.close()
        shm.unlink()
    return 0


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("input", "robot", "full"), default="input")
    p.add_argument("--robot-id", default="1e783be2dd870ef5")
    p.add_argument("--arm-side", choices=("left", "right", "both"), default="left")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--https", action="store_true",
                   help="局域网直连 HTTPS（绕过 ngrok）。默认用 certs/ 下自签证书。")
    p.add_argument("--cert", default=None, help="TLS 证书路径(默认 certs/cert.pem)")
    p.add_argument("--key", default=None, help="TLS 私钥路径(默认 certs/key.pem)")
    p.add_argument("--report-hz", type=float, default=2.0, help="打印报告频率")
    p.add_argument("--duration", type=float, default=0.0, help="运行秒数，0=一直跑")

    # robot 模式
    p.add_argument("--rate", type=float, default=30.0, help="robot 模式发送频率(Hz)")
    p.add_argument("--move-vel", type=float, default=0.3, help="move_line 线速度(m/s)")
    p.add_argument("--motion", action="store_true", help="robot 模式发正弦运动(需 --execute)")
    p.add_argument("--execute", action="store_true", help="允许真正发运动")
    p.add_argument("--amp", type=float, default=0.02, help="正弦幅值(m)")
    p.add_argument("--freq", type=float, default=0.25, help="正弦频率(Hz)")
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")

    # full 模式（透传给 ArmController 的遥操作参数）
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-home", action="store_true")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--send-hz", type=float, default=15.0)
    p.add_argument("--smooth", type=float, default=0.4)
    p.add_argument("--pos-scale", type=float, default=1.0)
    p.add_argument("--workspace-radius", type=float, default=0.25)
    p.add_argument("--orientation-mode", choices=("absolute", "relative"), default="relative")
    p.add_argument("--lock-orientation", action="store_true")
    p.add_argument("--fixed-frame", action="store_true")
    p.add_argument("--no-deadman", action="store_true")
    p.add_argument("--engage-button", choices=("a", "b", "trigger"), default="a")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "input":
        return run_input(args)
    if args.mode == "robot":
        return run_robot(args)
    return run_full(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted by user", file=sys.stderr)
        raise SystemExit(130)

