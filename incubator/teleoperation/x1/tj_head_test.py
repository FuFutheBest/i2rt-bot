#!/usr/bin/env python3
"""头部两个舵机上下(pitch)/左右(yaw)运动测试。

关节定义（与 tj_controller_teleop.py / Head_control.py 一致）：
    joint[0] = pitch  上下俯仰
    joint[1] = yaw    左右偏航

本脚本以「当前位置」为基准做相对小步来回扫描，并在每一步后回读舵机实际位置，
用来判断 pitch / yaw 两个轴的硬件是否真的能动、能动多大范围。

在 robot_test 容器里运行：
    cd /root/incubator/teleoperation/x1
    # 干跑：只打印将要发送的目标，不下发
    python3 tj_head_test.py --dry-run

    # 实际测试上下 + 左右（默认每步 10 度，来回各扫一遍）
    python3 tj_head_test.py

    # 只测上下 / 只测左右
    python3 tj_head_test.py --axis pitch
    python3 tj_head_test.py --axis yaw

    # 自定义步幅(度)与停留时间(秒)
    python3 tj_head_test.py --step-deg 15 --settle 1.0
"""

import argparse
import math
import sys
import time

import nmxrdk
import nmxrt


DEFAULT_ROBOT_ID = "1e783be2dd870ef5"

PITCH_IDX = 0
YAW_IDX = 1

# 相对当前位置的安全幅度(rad)。脚本只会在 [current - SAFE, current + SAFE] 内移动。
SAFE_REL = 0.45            # ≈ ±25.8°，相对当前
HEAD_VEL = [1.0, 1.0]
HEAD_ACC = [2.0, 2.0]
MOVE_TIMEOUT_MS = 3000


def ensure_ok(result, action):
    if isinstance(result, nmxrt.Error):
        raise RuntimeError(f"{action} failed: {result.message}")
    return result


def positions(joint_state):
    return [j.pos for j in joint_state.joints]


def fmt(values, prec=4):
    return "[" + ", ".join(f"{v:.{prec}f}" for v in values) + "]"


def parse_args():
    p = argparse.ArgumentParser(description="头部 pitch/yaw 运动测试")
    p.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    p.add_argument("--axis", choices=["pitch", "yaw", "both"], default="both",
                   help="测试哪个轴。默认 both（先 pitch 后 yaw）。")
    p.add_argument("--step-deg", type=float, default=10.0,
                   help="单步幅度(度)，相对当前位置。默认 10。")
    p.add_argument("--settle", type=float, default=0.8,
                   help="每步后等待回读的时间(秒)。默认 0.8。")
    p.add_argument("--non-blocking", action="store_true",
                   help="move_joint 非阻塞下发。")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要发送的目标，不真正下发。")
    return p.parse_args()


def main():
    args = parse_args()
    step = math.radians(args.step_deg)
    if step > SAFE_REL:
        print(f"[warn] step {args.step_deg}deg 超过安全幅度，夹到 {math.degrees(SAFE_REL):.1f}deg")
        step = SAFE_REL

    robot = nmxrdk.RobotClient(args.robot_id)
    head = robot.head()
    blocking = not args.non_blocking

    ensure_ok(head.enable(), "head.enable")
    start = positions(ensure_ok(head.get_joint_state(), "head.get_joint_state"))
    if len(start) < 2:
        raise RuntimeError(f"期望 2 个头部关节，实际 {len(start)}: {start}")

    print(f"robot_id   : {args.robot_id}")
    print(f"起始位置   : pitch={start[PITCH_IDX]:.4f} rad ({math.degrees(start[PITCH_IDX]):.1f}deg), "
          f"yaw={start[YAW_IDX]:.4f} rad ({math.degrees(start[YAW_IDX]):.1f}deg)")
    print(f"单步幅度   : {math.degrees(step):.1f}deg ({step:.4f} rad), 相对当前")
    print(f"模式       : {'DRY-RUN（不下发）' if args.dry_run else 'LIVE'}")
    print("-" * 60)

    def move_to(target, label):
        target = list(target)
        print(f"[{label}] -> 目标 {fmt(target)}  "
              f"(pitch={math.degrees(target[PITCH_IDX]):+.1f}deg, "
              f"yaw={math.degrees(target[YAW_IDX]):+.1f}deg)")
        if args.dry_run:
            return
        try:
            ensure_ok(
                head.move_joint(target, HEAD_VEL, HEAD_ACC, blocking, MOVE_TIMEOUT_MS, {}),
                "head.move_joint")
        except Exception as exc:
            print(f"        !! move_joint 异常: {exc}")
            return
        time.sleep(args.settle)
        fb = positions(ensure_ok(head.get_joint_state(), "head.get_joint_state"))
        err = [fb[i] - target[i] for i in range(2)]
        print(f"        反馈 {fmt(fb)}  误差 {fmt(err)}")
        moved = abs(fb[PITCH_IDX] - start[PITCH_IDX]) + abs(fb[YAW_IDX] - start[YAW_IDX])
        if moved < 0.01:
            print("        ⚠ 舵机几乎没动（<0.6deg），该轴可能未响应/被限位。")

    def sweep(idx, name):
        print(f"\n==== 测试 {name} (joint[{idx}]) ====")
        # 当前 -> +step -> 当前 -> -step -> 当前
        base = list(start)
        seq = [
            (base[idx] + step, f"{name} +{math.degrees(step):.0f}deg"),
            (base[idx],        f"{name} 回中"),
            (base[idx] - step, f"{name} -{math.degrees(step):.0f}deg"),
            (base[idx],        f"{name} 回中"),
        ]
        for val, label in seq:
            tgt = list(start)
            tgt[idx] = val
            move_to(tgt, label)

    try:
        if args.axis in ("pitch", "both"):
            sweep(PITCH_IDX, "pitch(上下)")
        if args.axis in ("yaw", "both"):
            sweep(YAW_IDX, "yaw(左右)")

        print("\n==== 回到起始位置 ====")
        move_to(start, "home")
    finally:
        try:
            head.stop()
        except Exception:
            pass

    print("\n完成。对照每步『反馈』与『目标』即可判断两个轴是否都能动。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

