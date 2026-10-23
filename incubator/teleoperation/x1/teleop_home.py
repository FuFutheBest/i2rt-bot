#!/usr/bin/env python3
"""遥操作启动时的「初始位置」配置与回零函数。

被 tj_teleop.py / tj_controller_teleop.py 在启动时调用，确保每次启动先把
双臂(Joint 控制)和头部舵机带到统一的初始位姿，再进入遥操作。

直接修改下面的常量即可调整初始位置。
"""

from __future__ import annotations

import time

import nmxrt
from nmxrdk.common.robot_side import RobotSide


# =============================================================================
# >>> 初始位置配置（按需修改） <<<
# =============================================================================
# 双臂初始位置（Joint 控制，单位 rad，7 个关节）
INIT_LEFT_JOINT = [-0.782691, -0.602215, -0.674036, -1.747366, 0.299272, -0.156720, -0.062556]
INIT_RIGHT_JOINT = [0.868282, -0.262868, 0.663391, -1.677535, -0.119538, -0.355314, -0.103250]

# 头部舵机初始位置（rad）：[pitch 上下, yaw 左右]
INIT_HEAD = [0.60, -0.06]

# 回零运动参数
HOME_JOINT_VEL_RATIO = 10.0    # %，move_joint 关节速度比例（保守、平稳）
HOME_JOINT_ACC_RATIO = 10.0    # %
HOME_JOINT_TIMEOUT_MS = 15000  # 单臂回零阻塞超时

HOME_HEAD_VEL = [1.0, 1.0]
HOME_HEAD_ACC = [2.0, 2.0]
HOME_HEAD_TIMEOUT_MS = 5000
# =============================================================================


def _side_enum(side: str):
    return RobotSide.LEFT if side == "left" else RobotSide.RIGHT


def _is_err(result) -> bool:
    return isinstance(result, nmxrt.Error)


def home_arms(robot, sides=("left", "right"), blocking: bool = True) -> list[str]:
    """把指定侧的手臂用 move_joint 带到初始位置。

    返回错误信息列表（空列表表示全部成功）。单侧失败不会中断另一侧。
    """
    errors: list[str] = []
    targets = {"left": INIT_LEFT_JOINT, "right": INIT_RIGHT_JOINT}
    vel = [HOME_JOINT_VEL_RATIO] * 7
    acc = [HOME_JOINT_ACC_RATIO] * 7

    for side in sides:
        target = targets[side]
        try:
            arm = robot.arm(_side_enum(side))
            res = arm.enable()
            if _is_err(res):
                print(f"[home] {side} arm.enable warning: {res.message}")
            print(f"[home] {side} arm -> init joint {[round(v, 3) for v in target]} ...")
            res = arm.move_joint(target, vel, acc, blocking, HOME_JOINT_TIMEOUT_MS, {})
            if _is_err(res):
                raise RuntimeError(res.message)
            time.sleep(0.3)
            st = arm.get_joint_state()
            if not _is_err(st):
                now = [round(j.pos, 3) for j in st.joints]
                print(f"[home] {side} arm reached: {now}")
        except Exception as exc:
            errors.append(f"{side} arm: {exc}")
            print(f"[home] {side} arm ERROR: {exc}")
    return errors


def home_head(robot, blocking: bool = True):
    """把头部两个舵机带到初始位置。返回 head 句柄（已 enable）或 None。"""
    try:
        head = robot.head()
        res = head.enable()
        if _is_err(res):
            print(f"[home] head.enable warning: {res.message}")
        print(f"[home] head -> init [pitch={INIT_HEAD[0]}, yaw={INIT_HEAD[1]}] ...")
        res = head.move_joint(INIT_HEAD, HOME_HEAD_VEL, HOME_HEAD_ACC,
                              blocking, HOME_HEAD_TIMEOUT_MS, {})
        if _is_err(res):
            raise RuntimeError(res.message)
        time.sleep(0.3)
        st = head.get_joint_state()
        if not _is_err(st):
            now = [round(j.pos, 3) for j in st.joints]
            print(f"[home] head reached: {now}")
        return head
    except Exception as exc:
        print(f"[home] head ERROR: {exc}")
        return None


def go_home(robot, arm_side: str = "both", do_head: bool = True,
            blocking: bool = True):
    """启动时回到初始位置。

    arm_side: "left" / "right" / "both"
    do_head:  是否同时把头部带到初始位置
    返回 (errors:list[str], head_handle 或 None)
    """
    sides = ["left", "right"] if arm_side == "both" else [arm_side]
    print("[home] === 回到初始位置 ===")
    errors = home_arms(robot, sides=sides, blocking=blocking)
    head = home_head(robot, blocking=blocking) if do_head else None
    if errors:
        print(f"[home] 完成（有错误）: {errors}")
    else:
        print("[home] 完成。")
    return errors, head

