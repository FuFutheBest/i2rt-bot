"""Fail-closed joint-zero verification for YAM disconnects.

Author: SuperInference Contributors
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class JointZeroConfirmation:
    confirmed: bool
    consecutive_samples: int
    max_error_rad: float
    feedback: Optional[np.ndarray]
    error: Optional[str] = None


def _threads_running_bound_method(
    owner: Any,
    method_name: str,
    threads: Iterable[threading.Thread],
) -> list[threading.Thread]:
    matches: list[threading.Thread] = []
    for thread in threads:
        target = getattr(thread, "_target", None)
        if (
            getattr(target, "__self__", None) is owner
            and getattr(target, "__name__", None) == method_name
        ):
            matches.append(thread)
    return matches


def close_yam_robot_without_can_race(
    robot: Any,
    *,
    join_timeout_s: float = 2.0,
    enumerate_threads: Callable[[], list[threading.Thread]] = threading.enumerate,
) -> None:
    """Stop i2rt control loops before its SDK closes the CAN socket."""
    if not np.isfinite(join_timeout_s) or join_timeout_s <= 0.0:
        raise ValueError("join_timeout_s must be positive and finite")

    motor_chain = getattr(robot, "motor_chain", None)
    if motor_chain is None:
        robot.close()
        return

    stop_event = getattr(robot, "_stop_event", None)
    server_thread = getattr(robot, "_server_thread", None)
    if stop_event is not None and callable(getattr(stop_event, "set", None)):
        stop_event.set()
    if (
        server_thread is not None
        and server_thread is not threading.current_thread()
        and callable(getattr(server_thread, "join", None))
    ):
        server_thread.join(timeout=join_timeout_s)
        if (
            callable(getattr(server_thread, "is_alive", None))
            and server_thread.is_alive()
        ):
            raise RuntimeError("i2rt robot server thread did not stop before CAN close")

    motor_threads = _threads_running_bound_method(
        motor_chain,
        "_set_torques_and_update_state",
        enumerate_threads(),
    )
    motor_was_running = bool(getattr(motor_chain, "running", False))
    if (
        motor_was_running
        and bool(getattr(motor_chain, "start_thread_flag", False))
        and not motor_threads
    ):
        raise RuntimeError("cannot identify the active i2rt DM torque thread")

    motor_chain.running = False
    for thread in motor_threads:
        if thread is threading.current_thread():
            continue
        thread.join(timeout=join_timeout_s)
    alive = [thread.name for thread in motor_threads if thread.is_alive()]
    if alive:
        motor_chain.running = True
        raise RuntimeError(
            "i2rt DM torque thread did not stop before CAN close: "
            + ", ".join(alive)
        )

    robot.close()


def confirm_measured_joint_zero(
    robot: Any,
    goal_full: np.ndarray,
    *,
    arm_joint_count: int,
    tolerance_rad: float,
    timeout_s: float,
    consecutive_samples: int,
    poll_interval_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> JointZeroConfirmation:
    """Command zero hold until measured joints confirm the disconnect posture."""
    if arm_joint_count < 1:
        raise ValueError("arm_joint_count must be at least 1")
    if not np.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
        raise ValueError("tolerance_rad must be positive and finite")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive and finite")
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be at least 1")
    if not np.isfinite(poll_interval_s) or poll_interval_s <= 0.0:
        raise ValueError("poll_interval_s must be positive and finite")

    goal = np.asarray(goal_full, dtype=float).reshape(-1)
    if goal.size < arm_joint_count:
        raise ValueError("goal_full does not contain every arm joint")
    if not np.all(np.isfinite(goal)):
        raise ValueError("goal_full contains non-finite values")

    deadline = monotonic() + timeout_s
    confirmed_samples = 0
    max_error_rad = float("inf")
    last_feedback: Optional[np.ndarray] = None
    last_error: Optional[str] = None

    while monotonic() < deadline:
        try:
            robot.command_joint_pos(goal)
            feedback = np.asarray(robot.get_joint_pos(), dtype=float).reshape(-1)
            if feedback.size < arm_joint_count:
                raise RuntimeError(
                    f"expected {arm_joint_count} joints, got {feedback.size}"
                )
            arm_feedback = feedback[:arm_joint_count]
            if not np.all(np.isfinite(arm_feedback)):
                raise RuntimeError("joint feedback contains non-finite values")
            last_feedback = arm_feedback.copy()
            max_error_rad = float(np.max(np.abs(arm_feedback)))
            last_error = None
            if max_error_rad <= tolerance_rad:
                confirmed_samples += 1
                if confirmed_samples >= consecutive_samples:
                    return JointZeroConfirmation(
                        confirmed=True,
                        consecutive_samples=confirmed_samples,
                        max_error_rad=max_error_rad,
                        feedback=last_feedback,
                    )
            else:
                confirmed_samples = 0
        except Exception as exc:
            confirmed_samples = 0
            last_error = f"{type(exc).__name__}: {exc}"
        sleep(poll_interval_s)

    return JointZeroConfirmation(
        confirmed=False,
        consecutive_samples=confirmed_samples,
        max_error_rad=max_error_rad,
        feedback=last_feedback,
        error=last_error,
    )


__all__ = [
    "JointZeroConfirmation",
    "close_yam_robot_without_can_race",
    "confirm_measured_joint_zero",
]
