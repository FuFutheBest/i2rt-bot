#!/usr/bin/env python3
"""Move Tianji arm(s) to an absolute joint or TCP target.

Supports single arm (left/right) or both arms at once.

目标位置直接在脚本顶部的「目标位置配置区」用纯数字向量修改，
终端只需选择模式和侧别，无需再传具体坐标。

Run inside the robot_test container:
    # 双臂，joint 模式（使用脚本里的 JOINT_TARGET_LEFT / JOINT_TARGET_RIGHT）:
    python3 incubator/teleoperation/x1/TJArm_MOVE2position.py --mode joint --side both --execute

    # 双臂，tcp 模式（使用脚本里的 TCP_TARGET_LEFT / TCP_TARGET_RIGHT）:
    python3 incubator/teleoperation/x1/TJArm_MOVE2position.py --mode tcp --side both --execute

    # 单臂:
    python3 incubator/teleoperation/x1/TJArm_MOVE2position.py --mode tcp --side left --execute

    # 先 dry-run 查看当前/目标（不加 --execute）:
    python3 incubator/teleoperation/x1/TJArm_MOVE2position.py --mode tcp --side both

    # 夹爪:
    python3 incubator/teleoperation/x1/TJArm_MOVE2position.py --side left --gripper-width 0.03 --execute

    # (可选) 临时用命令行覆盖脚本里的目标，仍接受纯数字向量:
    python3 incubator/teleoperation/x1/TJArm_MOVE2position.py --mode joint --side left \
        --joint-target "[0,0,0,0,0,0,0]" --execute
"""

import argparse
import ast
import contextlib
import gc
import json
import os
import sys
import time

import nmxrdk
import nmxrt
from nmxrdk.common.robot_side import RobotSide


DEFAULT_ROBOT_ID = "1e783be2dd870ef5"
DEFAULT_JOINT_VEL_RATIO = 3.0
DEFAULT_JOINT_ACC_RATIO = 3.0
DEFAULT_LINEAR_VEL = 0.02


# =============================================================================
# >>> 在这里修改目标位置（纯数字向量，无需在终端传参） <<<
#
# JOINT 模式：7 个关节角，单位 rad，顺序 [j1, j2, j3, j4, j5, j6, j7]
# TCP   模式：位姿 7 维，顺序 [x, y, z, qw, qx, qy, qz]
#             (x/y/z 单位 m，qw/qx/qy/qz 为四元数)
#
# 运行时：
#   --mode joint --side both   ->  使用 JOINT_TARGET_LEFT / JOINT_TARGET_RIGHT
#   --mode tcp   --side both   ->  使用 TCP_TARGET_LEFT   / TCP_TARGET_RIGHT
#   --side left / right        ->  只用对应一侧
# 终端如仍传 --joint-target / --tcp-target 等参数，则覆盖这里的默认值。
# =============================================================================

# ---- JOINT 目标（rad）：[j1, j2, j3, j4, j5, j6, j7] ----
# 零位置（home）
JOINT_TARGET_LEFT = [0, 0, 0, 0, 0, 0, 0]
JOINT_TARGET_RIGHT = [0, 0, 0, 0, 0, 0, 0]

# 服务场景 Joint
# JOINT_TARGET_LEFT = [-0.782691, -0.602215, -0.674036, -1.747366, 0.299272, -0.156720, -0.062556]
# JOINT_TARGET_RIGHT = [0.868282, -0.262868, 0.663391, -1.677535, -0.119538, -0.355314, -0.103250]

# ---- TCP 目标：[x, y, z, qw, qx, qy, qz] ----
# 零位置（home）
TCP_TARGET_LEFT =[0.104804, -0.463161, 0.166985, 0.048313, 0.015163, -0.923104, -0.381201]
TCP_TARGET_RIGHT = [0.041636, 0.453309, 0.310680, 0.325349, 0.943460, 0.047499, -0.042144]

# 服务场景 TCP
# TCP_TARGET_LEFT = [0.158753, -0.455150, 0.166093, 0.052617, 0.078988, -0.920590, -0.378821]
# TCP_TARGET_RIGHT = [0.093408, 0.449833, 0.312363, 0.320999, 0.945590, -0.017528, -0.050125]

# =============================================================================


@contextlib.contextmanager
def suppress_native_output(enabled: bool):
    """Suppress noisy native stdout/stderr logs from nmxrt."""
    if not enabled:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move Tianji arm(s)/gripper to target position.")
    parser.add_argument(
        "--robot-id",
        default=os.getenv("NMXRDK_ROBOT_ID", DEFAULT_ROBOT_ID),
        help="Robot id used by nmxrdk.RobotClient.",
    )
    parser.add_argument(
        "--mode",
        choices=("joint", "tcp"),
        default="joint",
        help="Arm control mode. joint uses move_joint, tcp uses move_line.",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Arm side(s) to control.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send commands. Without this flag, only prints current/target.",
    )
    parser.add_argument(
        "--return-home",
        action="store_true",
        help="Move back to the original arm state after the test move.",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="Send non-blocking arm command.",
    )
    parser.add_argument(
        "--verbose-nmxrt",
        action="store_true",
        help="Show raw NmxRT create/drop logs.",
    )

    joint_group = parser.add_argument_group("joint mode")
    joint_group.add_argument(
        "--joint-target",
        default=None,
        help='Absolute 7-DoF joint target in radians, shared by all selected sides. Example: "[0,0,0,0,0,0,0]".',
    )
    joint_group.add_argument(
        "--joint-target-left",
        default=None,
        help="Absolute 7-DoF joint target in radians for the left arm. Overrides --joint-target.",
    )
    joint_group.add_argument(
        "--joint-target-right",
        default=None,
        help="Absolute 7-DoF joint target in radians for the right arm. Overrides --joint-target.",
    )
    joint_group.add_argument(
        "--joint-vel-ratio",
        type=float,
        default=DEFAULT_JOINT_VEL_RATIO,
        help="Joint velocity ratio in percent. Default: 3.",
    )
    joint_group.add_argument(
        "--joint-acc-ratio",
        type=float,
        default=DEFAULT_JOINT_ACC_RATIO,
        help="Joint acceleration ratio in percent. Default: 3.",
    )

    tcp_group = parser.add_argument_group("tcp mode")
    tcp_target_help = (
        "Absolute TCP target. Accepts [x,y,z,qw,qx,qy,qz] or "
        '{"pose":{"x":...,"y":...,"z":...},"quat":{"w":...,"x":...,"y":...,"z":...}}.'
    )
    tcp_group.add_argument(
        "--tcp-target",
        default=None,
        help=tcp_target_help + " Shared by all selected sides.",
    )
    tcp_group.add_argument(
        "--tcp-target-left",
        default=None,
        help="TCP target for the left arm. Overrides --tcp-target.",
    )
    tcp_group.add_argument(
        "--tcp-target-right",
        default=None,
        help="TCP target for the right arm. Overrides --tcp-target.",
    )
    tcp_group.add_argument(
        "--linear-vel",
        type=float,
        default=DEFAULT_LINEAR_VEL,
        help="TCP linear velocity in m/s.",
    )

    gripper_group = parser.add_argument_group("gripper")
    gripper_group.add_argument(
        "--gripper-width",
        type=float,
        default=None,
        help="Optional gripper opening width in meters. Example: 0.03.",
    )
    gripper_group.add_argument(
        "--gripper-side",
        choices=("same", "left", "right", "both"),
        default="same",
        help="Which gripper to control. Default: same side(s) as arm.",
    )
    gripper_action = gripper_group.add_mutually_exclusive_group()
    gripper_action.add_argument(
        "--gripper-open",
        action="store_true",
        help="Open the selected gripper.",
    )
    gripper_action.add_argument(
        "--gripper-close",
        action="store_true",
        help="Close the selected gripper.",
    )
    return parser.parse_args()


def side_to_robot_side(side: str):
    if side == "left":
        return RobotSide.LEFT
    if side == "right":
        return RobotSide.RIGHT
    raise ValueError(f"Unsupported side: {side}")


def resolve_sides(side: str) -> list[str]:
    if side == "both":
        return ["left", "right"]
    return [side]


def ensure_not_error(result, action: str):
    if isinstance(result, nmxrt.Error):
        raise RuntimeError(f"{action} failed: {result.message}")
    return result


def parse_float_list(value: str, expected_len: int, name: str) -> list[float]:
    value = value.strip()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value

    if isinstance(parsed, (list, tuple)):
        result = [float(item) for item in parsed]
    else:
        try:
            result = [float(item.strip()) for item in str(parsed).split(",")]
        except ValueError as exc:
            raise ValueError(f"{name} must be comma-separated floats: {value}") from exc

    if len(result) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(result)}")
    return result


def parse_tcp_target(value: str, name: str = "--tcp-target") -> list[float]:
    value = value.strip()
    if value.startswith("{"):
        try:
            target = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} JSON parse failed: {exc}") from exc

        try:
            pose = target["pose"]
            quat = target["quat"]
            return [
                float(pose["x"]),
                float(pose["y"]),
                float(pose["z"]),
                float(quat["w"]),
                float(quat["x"]),
                float(quat["y"]),
                float(quat["z"]),
            ]
        except KeyError as exc:
            raise ValueError(f"{name} missing key: {exc}") from exc

    return parse_float_list(value, 7, name)


def joint_positions(joint_state) -> list[float]:
    joints = list(joint_state.joints)
    positions = [joint.pos for joint in joints]
    if len(positions) != 7:
        raise RuntimeError(f"Expected 7 joints, got {len(positions)}: {positions}")
    return positions


def tcp_pose(tcp_state) -> list[float]:
    pose = tcp_state.pose
    return [
        pose.pos.x,
        pose.pos.y,
        pose.pos.z,
        pose.quat.w,
        pose.quat.x,
        pose.quat.y,
        pose.quat.z,
    ]


def move_joint_with_speed(
    arm,
    target: list[float],
    blocking: bool,
    vel_ratio: float,
    acc_ratio: float,
):
    vel = [vel_ratio] * 7
    acc = [acc_ratio] * 7
    follow = False
    return ensure_not_error(
        arm.move_joint(target, vel, acc, blocking, follow),
        "move_joint",
    )


def move_line(arm, target: list[float], linear_vel: float, blocking: bool):
    return ensure_not_error(
        arm.move_line(target, linear_vel, blocking),
        "move_line",
    )


def validate_vector(values, expected_len: int, name: str) -> list[float]:
    result = [float(item) for item in values]
    if len(result) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(result)}")
    return result


JOINT_TARGET_DEFAULTS = {"left": JOINT_TARGET_LEFT, "right": JOINT_TARGET_RIGHT}
TCP_TARGET_DEFAULTS = {"left": TCP_TARGET_LEFT, "right": TCP_TARGET_RIGHT}


def resolve_joint_target(args, side: str):
    per_side = getattr(args, f"joint_target_{side}")
    raw = per_side if per_side is not None else args.joint_target
    if raw is not None:
        return parse_float_list(raw, 7, f"--joint-target-{side}")
    # Fall back to the in-script config block.
    return validate_vector(JOINT_TARGET_DEFAULTS[side], 7, f"JOINT_TARGET_{side.upper()}")


def resolve_tcp_target(args, side: str):
    per_side = getattr(args, f"tcp_target_{side}")
    raw = per_side if per_side is not None else args.tcp_target
    if raw is not None:
        return parse_tcp_target(raw, f"--tcp-target-{side}")
    # Fall back to the in-script config block.
    return validate_vector(TCP_TARGET_DEFAULTS[side], 7, f"TCP_TARGET_{side.upper()}")


def resolve_gripper_sides(arm_sides: list[str], gripper_side: str) -> list[str]:
    if gripper_side == "same":
        return list(arm_sides)
    if gripper_side == "both":
        return ["left", "right"]
    return [gripper_side]


def main() -> int:
    args = parse_args()
    messages: list[str] = []

    def log(msg: str = "") -> None:
        messages.append(msg)

    move_errors: list[str] = []

    sides = resolve_sides(args.side)

    # Arm targets always come from the in-script config block (or CLI overrides),
    # so there is always an arm target to move to.
    has_gripper_target = (
        args.gripper_width is not None or args.gripper_open or args.gripper_close
    )

    with suppress_native_output(not args.verbose_nmxrt):
        robot = nmxrdk.RobotClient(args.robot_id)
        blocking = not args.non_blocking

        messages.append(f"robot_id: {args.robot_id}")
        messages.append(f"sides: {', '.join(sides)}")
        messages.append(f"mode: {args.mode}")

        arms = {side: robot.arm(side_to_robot_side(side)) for side in sides}

        current_by_side: dict[str, list[float]] = {}
        target_by_side: dict[str, list[float]] = {}

        for side in sides:
            arm = arms[side]
            messages.append("")
            messages.append(f"----- {side} arm -----")

            if args.mode == "joint":
                target = resolve_joint_target(args, side)
                state = ensure_not_error(arm.get_joint_state(), f"{side} get_joint_state")
                current = joint_positions(state)
                current_by_side[side] = current
                messages.append(f"current joints(rad): {current}")
                if target is not None:
                    target_by_side[side] = target
                    messages.append(f"target joints(rad):  {target}")
                    messages.append(
                        f"joint vel/acc ratio(%): {args.joint_vel_ratio}/{args.joint_acc_ratio}"
                    )
                else:
                    messages.append("no joint target for this side, skip move.")
            else:
                target = resolve_tcp_target(args, side)
                state = ensure_not_error(arm.get_tcp_state(), f"{side} get_tcp_state")
                current = tcp_pose(state)
                current_by_side[side] = current
                messages.append(f"current tcp [x,y,z,qw,qx,qy,qz]: {current}")
                if target is not None:
                    target_by_side[side] = target
                    messages.append(f"target tcp  [x,y,z,qw,qx,qy,qz]: {target}")
                    messages.append(f"linear_vel(m/s): {args.linear_vel}")
                else:
                    messages.append("no tcp target for this side, skip move.")

        if has_gripper_target:
            gripper_sides = resolve_gripper_sides(sides, args.gripper_side)
            messages.append("")
            messages.append(f"gripper side(s): {', '.join(gripper_sides)}")
            if args.gripper_width is not None:
                messages.append(f"gripper target width(m): {args.gripper_width}")
            elif args.gripper_open:
                messages.append("gripper action: open")
            elif args.gripper_close:
                messages.append("gripper action: close")

        if not args.execute:
            messages.append("")
            messages.append("dry-run only. Add --execute to send commands.")
            for side in sides:
                del arms[side]
            arms.clear()
            del robot
            gc.collect()
            print("\n".join(messages))
            return 0

        for side in sides:
            if side not in target_by_side:
                continue
            arm = arms[side]
            target = target_by_side[side]
            log("")
            try:
                if args.mode == "joint":
                    log(f"[{side}] sending move_joint...")
                    status = move_joint_with_speed(
                        arm,
                        target,
                        blocking=blocking,
                        vel_ratio=args.joint_vel_ratio,
                        acc_ratio=args.joint_acc_ratio,
                    )
                    log(f"[{side}] move_joint status: {status}")
                    time.sleep(0.5)
                    after_state = ensure_not_error(
                        arm.get_joint_state(), f"{side} get_joint_state after move"
                    )
                    log(f"[{side}] after joints(rad): {joint_positions(after_state)}")
                else:
                    log(f"[{side}] sending move_line...")
                    status = move_line(arm, target, linear_vel=args.linear_vel, blocking=blocking)
                    log(f"[{side}] move_line status: {status}")
                    time.sleep(0.5)
                    after_state = ensure_not_error(
                        arm.get_tcp_state(), f"{side} get_tcp_state after move"
                    )
                    log(f"[{side}] after tcp: {tcp_pose(after_state)}")
            except Exception as exc:
                # Do not let one side abort the others (e.g. a blocking-status
                # query timeout on one arm should not block the other arm).
                move_errors.append(f"{side}: {exc}")
                log(f"[{side}] ERROR during move: {exc}")

        if args.return_home:
            for side in sides:
                if side not in target_by_side:
                    continue
                arm = arms[side]
                home = current_by_side[side]
                messages.append("")
                messages.append(f"[{side}] returning to original state...")
                if args.mode == "joint":
                    status = move_joint_with_speed(
                        arm,
                        home,
                        blocking=blocking,
                        vel_ratio=args.joint_vel_ratio,
                        acc_ratio=args.joint_acc_ratio,
                    )
                    messages.append(f"[{side}] return status: {status}")
                    time.sleep(0.5)
                    final_state = ensure_not_error(
                        arm.get_joint_state(), f"{side} get_joint_state final"
                    )
                    messages.append(f"[{side}] final joints(rad): {joint_positions(final_state)}")
                else:
                    status = move_line(arm, home, linear_vel=args.linear_vel, blocking=blocking)
                    messages.append(f"[{side}] return status: {status}")
                    time.sleep(0.5)
                    final_state = ensure_not_error(
                        arm.get_tcp_state(), f"{side} get_tcp_state final"
                    )
                    messages.append(f"[{side}] final tcp: {tcp_pose(final_state)}")

        if has_gripper_target:
            gripper_sides = resolve_gripper_sides(sides, args.gripper_side)
            for g_side in gripper_sides:
                gripper = robot.gripper(side_to_robot_side(g_side))
                messages.append("")
                if args.gripper_width is not None:
                    messages.append(f"[{g_side}] sending gripper.set_width...")
                    status = ensure_not_error(
                        gripper.set_width(args.gripper_width),
                        "gripper.set_width",
                    )
                elif args.gripper_open:
                    messages.append(f"[{g_side}] sending gripper.open...")
                    status = ensure_not_error(gripper.open(), "gripper.open")
                else:
                    messages.append(f"[{g_side}] sending gripper.close...")
                    status = ensure_not_error(gripper.close(), "gripper.close")
                messages.append(f"[{g_side}] gripper status: {status}")

        for side in sides:
            del arms[side]
        arms.clear()
        del robot
        gc.collect()

        if move_errors:
            log("")
            log("=== move errors (other sides still attempted) ===")
            for err in move_errors:
                log(f"  - {err}")

    print("\n".join(messages))

    if move_errors:
        return 1

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

