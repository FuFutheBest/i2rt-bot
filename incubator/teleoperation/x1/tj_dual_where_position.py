#!/usr/bin/env python3
"""Print current joint and TCP positions for both Tianji arms.

Run inside the robot_test container:
    python3 incubator/teleoperation/x1/tj_dual_where_position.py

Optional:
    python3 incubator/teleoperation/x1/tj_dual_where_position.py --verbose-nmxrt
"""

import argparse
import contextlib
import gc
import os
import sys

import nmxrdk
import nmxrt
from nmxrdk.common.robot_side import RobotSide


DEFAULT_ROBOT_ID = "1e783be2dd870ef5"


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
    parser = argparse.ArgumentParser(
        description="Print current joint and TCP positions for both Tianji arms.",
    )
    parser.add_argument(
        "--robot-id",
        default=os.getenv("NMXRDK_ROBOT_ID", DEFAULT_ROBOT_ID),
        help="Robot id used by nmxrdk.RobotClient.",
    )
    parser.add_argument(
        "--verbose-nmxrt",
        action="store_true",
        help="Show raw NmxRT create/drop logs.",
    )
    return parser.parse_args()


def ensure_not_error(result, action: str):
    if isinstance(result, nmxrt.Error):
        raise RuntimeError(f"{action} failed: {result.message}")
    return result


def joint_positions(joint_state) -> list[float]:
    joints = list(joint_state.joints)
    return [joint.pos for joint in joints]


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


def format_floats(values: list[float], precision: int = 6) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in values) + "]"


def main() -> int:
    args = parse_args()
    messages: list[str] = []

    with suppress_native_output(not args.verbose_nmxrt):
        robot = nmxrdk.RobotClient(args.robot_id)

        messages.append(f"robot_id: {args.robot_id}")

        for side_name, side_enum in (("left", RobotSide.LEFT), ("right", RobotSide.RIGHT)):
            arm = robot.arm(side_enum)

            joint_state = ensure_not_error(arm.get_joint_state(), f"{side_name} get_joint_state")
            tcp_state = ensure_not_error(arm.get_tcp_state(), f"{side_name} get_tcp_state")

            joints = joint_positions(joint_state)
            tcp = tcp_pose(tcp_state)

            messages.append("")
            messages.append(f"===== {side_name} arm =====")
            messages.append(f"joints(rad):                {format_floats(joints)}")
            messages.append(f"tcp [x,y,z,qw,qx,qy,qz]:    {format_floats(tcp)}")

            del arm

        del robot
        gc.collect()

    print("\n".join(messages))

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

