#!/usr/bin/env python3
"""
Replay left-arm LK gripper widths from parquet trajectories.

The default archive contains multiple trajectories. This script only reads the
left gripper command and sends it directly to the left LK gripper, without
replaying arm poses or writing policy shared memory.

Author: Jun Lv
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.legacy.test_policy_gripper_control import (  # noqa: E402
    DEFAULT_DEVICES_YAML,
    GripperRuntimeConfig,
    _build_direct_gripper,
    _find_rizon_device_config,
    _parse_gripper_runtime,
    scale_realtime_gripper_command,
)
from utils.logger_config import logger  # noqa: E402
from utils.time_control import precise_loop_timing  # noqa: E402

DEFAULT_ARCHIVE = Path("/home/noematrix/replay-data/giftbox/giftbox_right_gripper_samples_parquet.tar.gz")
LEFT_ARM_INDEX = 0
LEFT_ACTION_INDEX = 7
LEFT_HAND_STATE_COLUMN = "observation.extra.left.hand_state"


def list_parquet_members(archive_path: Path) -> List[str]:
    """Return parquet member names in stable archive order, preferring normal trajectories."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member.name
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".parquet")
        ]
    normal_members = [name for name in members if "/normal/" in f"/{name}"]
    abnormal_members = [name for name in members if name not in normal_members]
    return normal_members + abnormal_members


def select_member(members: Sequence[str], member: Optional[str], trajectory_index: int) -> str:
    """Select a parquet member by exact/substr name or index."""
    if not members:
        raise ValueError("No parquet trajectories found in archive")

    if member:
        matches = [
            name
            for name in members
            if name == member or Path(name).name == member or member in name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one member match for {member!r}, got {matches}")
        return matches[0]

    if trajectory_index < 0 or trajectory_index >= len(members):
        raise IndexError(f"trajectory-index must be in [0, {len(members) - 1}], got {trajectory_index}")
    return members[trajectory_index]


def read_parquet_from_archive(archive_path: Path, member: str) -> pd.DataFrame:
    """Read one parquet member from a tar.gz archive."""
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Failed to extract {member!r} from {archive_path}")
        return pd.read_parquet(io.BytesIO(extracted.read()))


def extract_left_gripper_widths(df: pd.DataFrame, source: str) -> np.ndarray:
    """Extract left gripper widths in policy width_m semantics."""
    if source in {"auto", "action"} and "action" in df.columns:
        actions = np.stack(df["action"].to_numpy()).astype(np.float64)
        if actions.ndim == 2 and actions.shape[1] > LEFT_ACTION_INDEX:
            return actions[:, LEFT_ACTION_INDEX]
        if source == "action":
            raise ValueError(f"Expected action dim > {LEFT_ACTION_INDEX}, got {actions.shape}")

    if source in {"auto", "hand-state"} and LEFT_HAND_STATE_COLUMN in df.columns:
        return df[LEFT_HAND_STATE_COLUMN].to_numpy(dtype=np.float64)

    raise KeyError(
        "Could not find left gripper data. Expected action[7] or "
        f"{LEFT_HAND_STATE_COLUMN!r}."
    )


def slice_widths(widths: np.ndarray, start_frame: int, max_frames: Optional[int], stride: int) -> np.ndarray:
    """Apply simple frame selection to a width trajectory."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    selected = widths[max(0, start_frame) :: stride]
    if max_frames is not None:
        selected = selected[: max(1, int(max_frames))]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError("No finite left gripper widths selected")
    return selected.astype(np.float64)


def load_runtime(devices_yaml: Path, robot_device: str, camera_left_capture_id: int, camera_right_capture_id: int) -> GripperRuntimeConfig:
    """Load gripper runtime config from the existing device yaml."""
    rizon_cfg = _find_rizon_device_config(
        devices_yaml=devices_yaml,
        robot_device=robot_device,
        camera_left_capture_id=camera_left_capture_id,
        camera_right_capture_id=camera_right_capture_id,
    )
    return _parse_gripper_runtime(rizon_cfg)


def replay_widths(args: argparse.Namespace, runtime: GripperRuntimeConfig, widths: np.ndarray) -> int:
    """Replay a width trajectory to the left gripper."""
    hz = max(float(args.hz), 1e-6)
    wait_for_next_iteration = precise_loop_timing(1.0 / hz)
    velocity = float(runtime.max_vel[LEFT_ARM_INDEX])

    logger.info(
        "Left gripper replay: samples={samples}, hz={hz:.3f}, "
        "min={min_width:.5f}, max={max_width:.5f}, first={first:.5f}, last={last:.5f}".format(
            samples=widths.size,
            hz=hz,
            min_width=float(np.min(widths)),
            max_width=float(np.max(widths)),
            first=float(widths[0]),
            last=float(widths[-1]),
        )
    )

    gripper: Optional[Any] = None
    try:
        if args.dry_run:
            logger.info("[dry-run] Hardware connection skipped")
        else:
            logger.info(
                f"Connecting left gripper: port={runtime.ports[LEFT_ARM_INDEX]}, "
                f"calibration={runtime.calibration_files[LEFT_ARM_INDEX]}"
            )
            gripper = _build_direct_gripper(LEFT_ARM_INDEX, runtime, "left")

        if float(args.start_delay_s) > 0.0:
            logger.info(f"Starting left gripper replay in {args.start_delay_s:.1f}s")
            time.sleep(float(args.start_delay_s))

        for sample_idx, target in enumerate(widths):
            scaled = scale_realtime_gripper_command(float(target), runtime.scale_factor[LEFT_ARM_INDEX])
            if args.dry_run:
                logger.info(
                    f"[dry-run] sample={sample_idx}/{widths.size - 1} "
                    f"target={float(target):.5f} scaled={scaled:.5f}"
                )
            else:
                if gripper is None:
                    raise RuntimeError("Left gripper is not initialized")
                ok = gripper.Move(scaled, velocity)
                if sample_idx % max(1, int(args.log_every)) == 0 or sample_idx == widths.size - 1:
                    state = gripper.states()
                    logger.info(
                        f"sample={sample_idx}/{widths.size - 1} target={float(target):.5f} "
                        f"scaled={scaled:.5f} width={state.width:.5f} move_ok={ok}"
                    )

            if sample_idx < widths.size - 1:
                wait_for_next_iteration()
        return 0
    finally:
        if gripper is not None:
            try:
                gripper.Disable()
            except Exception as exc:
                logger.warning(f"Failed to disable left gripper: {exc}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Replay only the left-arm gripper from parquet samples.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="tar.gz containing parquet trajectories")
    parser.add_argument("--devices-yaml", type=Path, default=DEFAULT_DEVICES_YAML, help="Device yaml with gripper config")
    parser.add_argument("--robot-device", default="RizonRobot_2", help="Robot device name in config")
    parser.add_argument("--camera-left-capture-id", type=int, default=1, help="Resolve ${camera.left_capture_id} in devices yaml")
    parser.add_argument("--camera-right-capture-id", type=int, default=0, help="Resolve ${camera.right_capture_id} in devices yaml")
    parser.add_argument("--member", default=None, help="Parquet member name or substring to replay")
    parser.add_argument("--trajectory-index", type=int, default=0, help="Trajectory index after preferring normal/ members")
    parser.add_argument("--list", action="store_true", help="List trajectories and exit")
    parser.add_argument("--source", choices=("auto", "action", "hand-state"), default="auto", help="Left gripper source")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to replay")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to replay")
    parser.add_argument("--stride", type=int, default=1, help="Replay every Nth frame")
    parser.add_argument("--hz", type=float, default=30.0, help="Command send frequency")
    parser.add_argument("--start-delay-s", type=float, default=3.0, help="Delay before hardware replay starts")
    parser.add_argument("--log-every", type=int, default=30, help="Log every N samples")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without moving hardware")
    return parser


def main() -> int:
    """Replay selected left gripper trajectory."""
    args = build_parser().parse_args()
    members = list_parquet_members(args.archive)
    if args.list:
        for index, name in enumerate(members):
            logger.info(f"[{index}] {name}")
        return 0

    member = select_member(members, args.member, int(args.trajectory_index))
    logger.info(f"Selected trajectory: {member}")
    df = read_parquet_from_archive(args.archive, member)
    widths = slice_widths(
        extract_left_gripper_widths(df, args.source),
        start_frame=int(args.start_frame),
        max_frames=args.max_frames,
        stride=int(args.stride),
    )
    runtime = load_runtime(
        devices_yaml=args.devices_yaml,
        robot_device=str(args.robot_device),
        camera_left_capture_id=int(args.camera_left_capture_id),
        camera_right_capture_id=int(args.camera_right_capture_id),
    )
    if runtime.value_mode != "width_m":
        raise ValueError(f"Only width_m gripper mode is supported, got {runtime.value_mode!r}")
    return replay_widths(args, runtime, widths)


if __name__ == "__main__":
    raise SystemExit(main())
