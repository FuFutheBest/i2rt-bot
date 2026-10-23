#!/usr/bin/env python3
"""TJ 机械臂健康检查：读取当前 TCP / 关节状态，可选做一次安全的小幅相对移动。

用法：
  # 只读当前状态（不发任何指令）
  python3 tj_arm_check.py --side left

  # 读完后让 TCP 沿机器人 +Z 抬高 3cm，再回到原位（验证臂能否执行 move_line）
  python3 tj_arm_check.py --side left --move-z 0.03
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import nmxrdk
import nmxrt
from nmxrdk.common.robot_side import RobotSide

ROBOT_ID = "1e783be2dd870ef5"
MOVE_VEL = 0.05      # m/s，move_line 保守速度
JOINT_VEL_RATIO = 3  # %，move_joint 关节速度比例(慢)
JOINT_ACC_RATIO = 3  # %


def side_of(s):
    return RobotSide.LEFT if s == "left" else RobotSide.RIGHT


def tcp_pose7(arm):
    s = arm.get_tcp_state()
    if isinstance(s, nmxrt.Error):
        raise RuntimeError(f"get_tcp_state failed: {s.message}")
    p = s.pose
    return [p.pos.x, p.pos.y, p.pos.z, p.quat.w, p.quat.x, p.quat.y, p.quat.z]


def joint_pos(arm):
    s = arm.get_joint_state()
    if isinstance(s, nmxrt.Error):
        raise RuntimeError(f"get_joint_state failed: {s.message}")
    return [j.pos for j in s.joints]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--move-z", type=float, default=0.0,
                    help="沿 +Z 的相对移动量(米)，0 表示只读不动")
    ap.add_argument("--center-joints", action="store_true",
                    help="关节空间回中：把每个关节夹到 ±band 安全带内并 move_joint 过去(脱离限位)")
    ap.add_argument("--band", type=float, default=2.2,
                    help="--center-joints 的安全带(rad)，默认 2.2(远离 ±π 限位)")
    ap.add_argument("--joints", default=None,
                    help='显式 7 关节目标(rad)，如 "[0,-1.2,1.0,0,-1.0,0,0]"，优先于 --center-joints')
    ap.add_argument("--blocking", action="store_true", default=True)
    ap.add_argument("--non-blocking", dest="blocking", action="store_false")
    args = ap.parse_args()

    robot = nmxrdk.RobotClient(ROBOT_ID)
    arm = robot.arm(side_of(args.side))

    pose = tcp_pose7(arm)
    joints = joint_pos(arm)
    print(f"[{args.side}] current TCP [x,y,z,qw,qx,qy,qz] = "
          f"{np.round(pose, 4).tolist()}")
    print(f"[{args.side}] current joints(rad)            = "
          f"{np.round(joints, 4).tolist()}")

    # ---- 关节空间回中 / 显式关节目标(脱离关节限位) ----
    if args.joints is not None or args.center_joints:
        if args.joints is not None:
            import ast
            target_j = [float(v) for v in ast.literal_eval(args.joints)]
            if len(target_j) != len(joints):
                raise ValueError(f"--joints 需要 {len(joints)} 个值")
        else:
            target_j = [float(np.clip(j, -args.band, args.band)) for j in joints]
        print(f"[{args.side}] move_joint -> {np.round(target_j, 4).tolist()} "
              f"(vel={JOINT_VEL_RATIO}%, blocking={args.blocking}) ...")
        vel = [JOINT_VEL_RATIO] * len(target_j)
        acc = [JOINT_ACC_RATIO] * len(target_j)
        res = arm.move_joint(target_j, vel, acc, args.blocking, False)
        print(f"[{args.side}] move_joint return(repr) = {res!r}")
        if isinstance(res, nmxrt.Error):
            print(f"[{args.side}] move_joint ERROR: {res.message}")
            return
        time.sleep(1.0)
        print(f"[{args.side}] after joints = {np.round(joint_pos(arm), 4).tolist()}")
        print(f"[{args.side}] after TCP    = {np.round(tcp_pose7(arm), 4).tolist()}")
        return

    if abs(args.move_z) < 1e-9:
        print("只读模式（未指定 --move-z）。如需测试运动加 --move-z 0.03")
        return

    target = list(pose)
    target[2] += args.move_z  # 只改 z，姿态保持不变
    print(f"[{args.side}] sending move_line -> z {pose[2]:.4f} -> {target[2]:.4f} "
          f"(blocking={args.blocking}) ...")
    res = arm.move_line(target, MOVE_VEL, args.blocking)
    print(f"[{args.side}] move_line return(repr) = {res!r}")
    if isinstance(res, nmxrt.Error):
        print(f"[{args.side}] move_line ERROR: {res.message}")
        return

    time.sleep(1.0)
    after = tcp_pose7(arm)
    print(f"[{args.side}] after move TCP z = {after[2]:.4f} "
          f"(期望≈{target[2]:.4f}, 实际位移={after[2]-pose[2]:+.4f})")

    print(f"[{args.side}] returning to original pose ...")
    res = arm.move_line(list(pose), MOVE_VEL, args.blocking)
    print(f"[{args.side}] return move_line return(repr) = {res!r}")
    time.sleep(1.0)
    final = tcp_pose7(arm)
    print(f"[{args.side}] final TCP z = {final[2]:.4f}")


if __name__ == "__main__":
    main()

