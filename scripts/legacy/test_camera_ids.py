#!/usr/bin/env python3
"""
Probe every /dev/video* node and report which camera indices are actually
capture-capable (cv2.VideoCapture(index) == /dev/video<index>). Optionally dump a
single preview frame for one chosen index so you can map that id to a physical
wrist camera.

Why this exists: each openMV USB camera exposes two /dev/video nodes and only one
of them can actually capture; index numbering also changes whenever a camera is
re-plugged / moved to another USB port. This helper answers "which /dev/videoN can
capture, and what does a given id see".

It only uses cv2 + sysfs (no SHM / device pipeline), so the cameras must be FREE
when running it (stop device_starter first, otherwise VideoCapture can't open a
busy device).

Usage (repo root, device end stopped):
    # list all available (capture-capable) camera ids, no image written
    ./.venv/bin/python scripts/legacy/test_camera_ids.py

    # additionally dump one preview frame for id 4
    ./.venv/bin/python scripts/legacy/test_camera_ids.py --id 4

Author: Wang Zheng
"""

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow running as `python scripts/legacy/test_camera_ids.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_config import logger


def list_video_indices() -> List[int]:
    """Return sorted /dev/video* indices present on the host."""
    indices: List[int] = []
    for path in glob.glob("/dev/video*"):
        tail = path.replace("/dev/video", "")
        if tail.isdigit():
            indices.append(int(tail))
    return sorted(indices)


def read_model_name(index: int) -> str:
    """Read the human-readable camera name from sysfs (best-effort)."""
    name_path = f"/sys/class/video4linux/video{index}/name"
    try:
        with open(name_path, "r") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def probe_capture(
    index: int,
    fourcc: str = "MJPG",
    warmup: int = 10,
    backend: int = cv2.CAP_V4L2,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Try to open /dev/video<index> with MJPG and grab one frame.

    Forces the V4L2 backend so cv2.VideoCapture(index) maps strictly to
    /dev/video<index>. With the default auto backend cv2 may fall back to FFMPEG,
    which enumerates cameras in its own order and makes the index mismatch the
    /dev/videoN numbering.

    Returns (ok, frame). ok is False for metadata-only nodes or busy/absent
    devices; frame is the last successfully read BGR frame (or None).
    """
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False, None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    ok = False
    frame: Optional[np.ndarray] = None
    for _ in range(max(1, warmup)):  # let auto-exposure / pipeline settle
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False, None
    return True, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="if set, dump ONE preview frame for this camera index; if omitted, only "
             "list the available camera ids (no image is written)",
    )
    parser.add_argument("--out-dir", type=str, default="logs/cam_id_test")
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--warmup", type=int, default=10, help="frames to read before saving")
    parser.add_argument(
        "--auto-backend",
        action="store_true",
        help="use cv2's default (auto) backend instead of forcing V4L2; NOT recommended "
             "because the index may then no longer match /dev/videoN",
    )
    args = parser.parse_args()

    backend = cv2.CAP_ANY if args.auto_backend else cv2.CAP_V4L2

    indices = list_video_indices()
    if not indices:
        logger.error("No /dev/video* nodes found.")
        return

    logger.info(f"Probing video indices: {indices}")
    available: List[int] = []
    for index in indices:
        model = read_model_name(index)
        ok, frame = probe_capture(index, fourcc=args.fourcc, warmup=args.warmup, backend=backend)
        if ok and frame is not None:
            h, w = frame.shape[:2]
            available.append(index)
            logger.info(f"  /dev/video{index}: AVAILABLE  model='{model}'  frame={w}x{h}")
        else:
            logger.info(f"  /dev/video{index}: not available  model='{model}'  (metadata-only / busy / no frame)")

    logger.info("=" * 60)
    logger.info(f"Available (capture-capable) camera ids: {available}")
    logger.info("=" * 60)

    # Only dump an image when the user explicitly asks for a specific id.
    if args.id is None:
        logger.info("No --id given, so no image was written. Re-run with --id <N> to dump one.")
        return

    if args.id not in available:
        logger.error(
            f"--id {args.id} is not an available capture-capable camera "
            f"(available: {available}). No image written."
        )
        return

    model = read_model_name(args.id)
    ok, frame = probe_capture(args.id, fourcc=args.fourcc, warmup=args.warmup, backend=backend)
    if not ok or frame is None:
        logger.error(f"Failed to grab a frame from /dev/video{args.id} on the second read.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    h, w = frame.shape[:2]
    out_path = os.path.join(args.out_dir, f"video{args.id}_{model.replace(' ', '_')}_{w}x{h}.png")
    cv2.imwrite(out_path, frame)
    logger.info(f"Wrote preview for id {args.id}: {out_path}  ({w}x{h}, model='{model}')")


if __name__ == "__main__":
    main()
