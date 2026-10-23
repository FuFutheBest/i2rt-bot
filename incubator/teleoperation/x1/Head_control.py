#!/usr/bin/env python3
"""读取并控制头部两个舵机（pitch 上下 / yaw 左右）。

头部关节定义（来自 tj_controller_teleop.py 用法）：
    joint[0] = pitch  上下俯仰
    joint[1] = yaw    左右偏航
    head.move_joint([pitch, yaw], vel, acc, blocking, timeout, opts)

在 robot_test 容器里运行：
    # 只读当前两个舵机位置（不发指令）：
    python3 incubator/teleoperation/x1/Head_control.py

    # 绝对目标（rad），dry-run 先看一眼，确认后再加 --execute：
    python3 incubator/teleoperation/x1/Head_control.py --pitch 0.2 --yaw 0.0
    python3 incubator/teleoperation/x1/Head_control.py --pitch 0.2 --yaw 0.0 --execute

    # 用角度（度）更直观：
    python3 incubator/teleoperation/x1/Head_control.py --pitch-deg 15 --yaw-deg -10 --execute

    # 相对当前位置移动（小步试探硬件范围时很有用）：
    python3 incubator/teleoperation/x1/Head_control.py --pitch 0.05 --relative --execute

    # 回零位（两个舵机都回 0）：
    python3 incubator/teleoperation/x1/Head_control.py --home --execute
"""

import argparse
import math
import sys
import time

import nmxrdk
import nmxrt


DEFAULT_ROBOT_ID = "1e783be2dd870ef5"

# =============================================================================
# >>> 头部舵机安全配置（按需修改） <<<
#
# 限位单位为 rad。这里是「软限位」——脚本会把目标夹到该范围内，防止打到机械限位。
# 注意：SDK 不提供读取硬件真实限位的接口，下面默认值参考 tj_controller_teleop.py
# 的 pitch ±0.6 rad，yaw 暂时给一个保守值。请用 --relative 小步试探后再放宽。
# =============================================================================
PITCH_LIMIT = (-0.60, 0.60)   # rad, joint[0] 上下俯仰安全范围 (≈ ±34°)
YAW_LIMIT = (-0.60, 0.60)     # rad, joint[1] 左右偏航安全范围 (≈ ±34°，保守值，请实测)

HEAD_VEL = [1.0, 1.0]         # 关节速度
HEAD_ACC = [2.0, 2.0]         # 关节加速度
MOVE_TIMEOUT_MS = 2000
# =============================================================================

PITCH_IDX = 0
YAW_IDX = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取/控制头部两个舵机 (pitch/yaw)。")
    parser.add_argument(
        "--robot-id",
        default=DEFAULT_ROBOT_ID,
        help="nmxrdk.RobotClient 使用的 robot id。",
    )

    parser.add_argument("--pitch", type=float, default=None, help="pitch 目标 (rad)。")
    parser.add_argument("--yaw", type=float, default=None, help="yaw 目标 (rad)。")
    parser.add_argument("--pitch-deg", type=float, default=None, help="pitch 目标 (度)。")
    parser.add_argument("--yaw-deg", type=float, default=None, help="yaw 目标 (度)。")

    parser.add_argument(
        "--relative",
        action="store_true",
        help="目标值按相对当前位置叠加（默认是绝对目标）。",
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="两个舵机都回到 0 位（覆盖 --pitch/--yaw）。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正下发 move_joint。不加则只打印当前/目标（dry-run）。",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="非阻塞下发。",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="不对目标做软限位夹取（危险，确认硬件范围后再用）。",
    )
    return parser.parse_args()


def ensure_not_error(result, action: str):
    if isinstance(result, nmxrt.Error):
        raise RuntimeError(f"{action} failed: {result.message}")
    return result


def head_joint_positions(joint_state) -> list[float]:
    return [j.pos for j in joint_state.joints]


def rad2deg(values) -> list[float]:
    return [math.degrees(v) for v in values]


def fmt(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in values) + "]"


def clamp(value: float, limit: tuple[float, float]) -> float:
    lo, hi = limit
    return max(lo, min(hi, value))


def resolve_target(args, current: list[float]) -> list[float] | None:
    """根据命令行参数解析出 [pitch, yaw] 绝对目标；无目标则返回 None。"""
    if args.home:
        return [0.0, 0.0]

    pitch = args.pitch
    if pitch is None and args.pitch_deg is not None:
        pitch = math.radians(args.pitch_deg)
    yaw = args.yaw
    if yaw is None and args.yaw_deg is not None:
        yaw = math.radians(args.yaw_deg)

    if pitch is None and yaw is None:
        return None

    target = list(current)
    if pitch is not None:
        target[PITCH_IDX] = current[PITCH_IDX] + pitch if args.relative else pitch
    if yaw is not None:
        target[YAW_IDX] = current[YAW_IDX] + yaw if args.relative else yaw
    return target


def main() -> int:
    args = parse_args()

    robot = nmxrdk.RobotClient(args.robot_id)
    head = robot.head()
    blocking = not args.non_blocking

    ensure_not_error(head.enable(), "head.enable")
    state = ensure_not_error(head.get_joint_state(), "head.get_joint_state")
    current = head_joint_positions(state)
    if len(current) < 2:
        raise RuntimeError(f"期望 2 个头部关节，实际 {len(current)}: {current}")

    print(f"robot_id: {args.robot_id}")
    print(f"软限位: pitch={PITCH_LIMIT} rad, yaw={YAW_LIMIT} rad")
    print(f"当前舵机(rad): pitch={current[PITCH_IDX]:.4f}, yaw={current[YAW_IDX]:.4f}")
    print(f"当前舵机(度): pitch={math.degrees(current[PITCH_IDX]):.2f}, "
          f"yaw={math.degrees(current[YAW_IDX]):.2f}")

    target = resolve_target(args, current)
    if target is None:
        print("只读模式（未指定 --pitch/--yaw/--home）。如需控制请加目标参数。")
        return 0

    if not args.no_clamp:
        clamped = [clamp(target[PITCH_IDX], PITCH_LIMIT), clamp(target[YAW_IDX], YAW_LIMIT)]
        if clamped != target:
            print(f"目标被软限位夹取: {fmt(target)} -> {fmt(clamped)}")
        target = clamped

    print(f"目标舵机(rad): pitch={target[PITCH_IDX]:.4f}, yaw={target[YAW_IDX]:.4f}")
    print(f"目标舵机(度): pitch={math.degrees(target[PITCH_IDX]):.2f}, "
          f"yaw={math.degrees(target[YAW_IDX]):.2f}")

    if not args.execute:
        print("dry-run only. 加 --execute 才会真正下发。")
        return 0

    print("sending head.move_joint...")
    status = ensure_not_error(
        head.move_joint(target, HEAD_VEL, HEAD_ACC, blocking, MOVE_TIMEOUT_MS, {}),
        "head.move_joint",
    )
    print(f"move_joint status: {status}")

    time.sleep(0.5)
    after = head_joint_positions(
        ensure_not_error(head.get_joint_state(), "head.get_joint_state after move")
    )
    print(f"到位后(rad): pitch={after[PITCH_IDX]:.4f}, yaw={after[YAW_IDX]:.4f}")
    print(f"到位后(度): pitch={math.degrees(after[PITCH_IDX]):.2f}, "
          f"yaw={math.degrees(after[YAW_IDX]):.2f}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

