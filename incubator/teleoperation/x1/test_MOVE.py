#!/usr/bin/env python3
"""Minimal Tianji dual-arm control smoke test.

Run inside the robot_test container:
    python3 incubator/teleoperation/x1/test_MOVE.py --side left
    python3 incubator/teleoperation/x1/test_MOVE.py --mode joint --side left --execute
    python3 incubator/teleoperation/x1/test_MOVE.py --mode tcp --side left --tcp-dz 0.01 --execute
    python3 incubator/teleoperation/x1/test_MOVE.py --side left --gripper-width 0.03 --execute
"""

import argparse
import os
import sys
import time

import nmxrdk
import nmxrt
from nmxrdk.common.robot_side import RobotSide


DEFAULT_ROBOT_ID = "1e783be2dd870ef5"
DEFAULT_DELTA_RAD = 0.02
DEFAULT_JOINT_INDEX = 6  # 0-based index, joint 7 by default.
DEFAULT_LINEAR_VEL = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Tianji arm/gripper via nmxrdk.")
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
        choices=("left", "right"),
        default="left",
        help="Arm side to control.",
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

    joint_group = parser.add_argument_group("joint mode")
    parser.add_argument(
        "--joint-index",
        type=int,
        default=DEFAULT_JOINT_INDEX,
        help="0-based joint index to move. Default: 6, the 7th joint.",
    )
    joint_group.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA_RAD,
        help="Small joint delta in radians.",
    )
    joint_group.add_argument(
        "--joint-target",
        default=None,
        help="Absolute 7-DoF joint target in radians, comma separated.",
    )

    tcp_group = parser.add_argument_group("tcp mode")
    tcp_group.add_argument(
        "--tcp-target",
        default=None,
        help="Absolute TCP pose [x,y,z,qw,qx,qy,qz], comma separated.",
    )
    tcp_group.add_argument("--tcp-dx", type=float, default=0.0, help="TCP x delta in meters.")
    tcp_group.add_argument("--tcp-dy", type=float, default=0.0, help="TCP y delta in meters.")
    tcp_group.add_argument("--tcp-dz", type=float, default=0.01, help="TCP z delta in meters.")
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
        choices=("same", "left", "right"),
        default="same",
        help="Which gripper to control. Default: same side as arm.",
    )
    return parser.parse_args()


def side_to_robot_side(side: str):
    if side == "left":
        return RobotSide.LEFT
    if side == "right":
        return RobotSide.RIGHT
    raise ValueError(f"Unsupported side: {side}")


def ensure_not_error(result, action: str):
    if isinstance(result, nmxrt.Error):
        raise RuntimeError(f"{action} failed: {result.message}")
    return result


def parse_float_list(value: str, expected_len: int, name: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated floats: {value}") from exc
    if len(result) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(result)}")
    return result


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


def move_joint(arm, target: list[float], blocking: bool):
    vel = [0.0] * 7
    acc = [0.0] * 7
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


def control_gripper(robot, arm_side: str, gripper_side: str, width: float):
    side = arm_side if gripper_side == "same" else gripper_side
    gripper = robot.gripper(side_to_robot_side(side))
    return ensure_not_error(gripper.set_width(width), "gripper.set_width")


def main() -> int:
    args = parse_args()
    if not 0 <= args.joint_index < 7:
        raise ValueError("--joint-index must be in [0, 6]")

    robot = nmxrdk.RobotClient(args.robot_id)
    arm = robot.arm(side_to_robot_side(args.side))
    blocking = not args.non_blocking

    print(f"robot_id: {args.robot_id}")
    print(f"side: {args.side}")

    if args.mode == "joint":
        state = ensure_not_error(arm.get_joint_state(), "get_joint_state")
        current = joint_positions(state)
        target = (
            parse_float_list(args.joint_target, 7, "--joint-target")
            if args.joint_target
            else current.copy()
        )
        if args.joint_target is None:
            target[args.joint_index] += args.delta

        print("mode: joint")
        print(f"current joints(rad): {current}")
        print(f"target joints(rad):  {target}")
    else:
        state = ensure_not_error(arm.get_tcp_state(), "get_tcp_state")
        current = tcp_pose(state)
        target = (
            parse_float_list(args.tcp_target, 7, "--tcp-target")
            if args.tcp_target
            else current.copy()
        )
        if args.tcp_target is None:
            target[0] += args.tcp_dx
            target[1] += args.tcp_dy
            target[2] += args.tcp_dz

        print("mode: tcp")
        print(f"current tcp [x,y,z,qw,qx,qy,qz]: {current}")
        print(f"target tcp  [x,y,z,qw,qx,qy,qz]: {target}")
        print(f"linear_vel(m/s): {args.linear_vel}")

    if args.gripper_width is not None:
        actual_gripper_side = args.side if args.gripper_side == "same" else args.gripper_side
        print(f"gripper side: {actual_gripper_side}")
        print(f"gripper target width(m): {args.gripper_width}")

    if not args.execute:
        print("dry-run only. Add --execute to send commands.")
        return 0

    if args.mode == "joint":
        print("sending move_joint...")
        status = move_joint(arm, target, blocking=blocking)
        print(f"move_joint status: {status}")

        time.sleep(0.5)
        after_state = ensure_not_error(arm.get_joint_state(), "get_joint_state after move")
        print(f"after joints(rad):   {joint_positions(after_state)}")

        if args.return_home:
            print("returning to original joints...")
            status = move_joint(arm, current, blocking=blocking)
            print(f"return status: {status}")
            time.sleep(0.5)
            final_state = ensure_not_error(arm.get_joint_state(), "get_joint_state final")
            print(f"final joints(rad):   {joint_positions(final_state)}")
    else:
        print("sending move_line...")
        status = move_line(arm, target, linear_vel=args.linear_vel, blocking=blocking)
        print(f"move_line status: {status}")

        time.sleep(0.5)
        after_state = ensure_not_error(arm.get_tcp_state(), "get_tcp_state after move")
        print(f"after tcp:           {tcp_pose(after_state)}")

        if args.return_home:
            print("returning to original tcp pose...")
            status = move_line(arm, current, linear_vel=args.linear_vel, blocking=blocking)
            print(f"return status: {status}")
            time.sleep(0.5)
            final_state = ensure_not_error(arm.get_tcp_state(), "get_tcp_state final")
            print(f"final tcp:           {tcp_pose(final_state)}")

    if args.gripper_width is not None:
        print("sending gripper.set_width...")
        status = control_gripper(
            robot,
            arm_side=args.side,
            gripper_side=args.gripper_side,
            width=args.gripper_width,
        )
        print(f"gripper status: {status}")

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

