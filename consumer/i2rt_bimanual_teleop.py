#!/usr/bin/env python3
"""Run standalone I2RT teleoperation for diagnostics.

Normal data collection owns this runtime through DeviceStarter.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devices.robots.i2rt_teleop.config import SIDES, load_teleoperation_config
from devices.robots.i2rt_teleop.coordinator import run_teleoperation


def _check_runtime_dependencies(config) -> None:
    import portal  # noqa: F401
    from i2rt.robots.get_robot import get_yam_robot  # noqa: F401

    if any(
        config.arms[side].follower.gripper.backend == "lktech" for side in SIDES
    ):
        sdk_root = REPO_ROOT / "third_party" / "lktech_motor_sdk"
        if str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))
        from lktech_motor_sdk import LKMotor  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "i2rt_bimanual_teleop.yaml",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and imports without opening hardware",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = args.config.expanduser().resolve()
    config = load_teleoperation_config(config_path)
    _check_runtime_dependencies(config)

    if args.check:
        for side in SIDES:
            arm = config.arms[side]
            print(
                f"{side}: leader={arm.leader.can_channel}, "
                f"follower={arm.follower.can_channel}, "
                f"gripper={arm.follower.gripper.backend}"
            )
        print("Configuration and runtime imports are valid; no hardware was opened.")
        return

    run_teleoperation(config_path, config)


if __name__ == "__main__":
    main()
