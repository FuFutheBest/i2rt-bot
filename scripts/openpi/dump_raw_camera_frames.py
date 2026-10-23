#!/usr/bin/env python3
"""
Dump raw (undistorted/unrotated/unresized) frames straight from cv2.VideoCapture
for the OpenPi wrist cameras, plus 90/180/270 rotation previews.

This is a standalone helper for deciding the correct `rotation_degrees` and crop
that make the deployed wrist frames match the training framing. It only uses cv2
(no SHM / device pipeline), so the cameras must be FREE when running it
(stop device_starter first, otherwise VideoCapture cannot open a busy device).

Author: Wang Zheng
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

# Allow running as `python scripts/openpi/dump_raw_camera_frames.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_config import logger

_ROTATE_FLAGS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    return cv2.rotate(image, _ROTATE_FLAGS[degrees])


def dump_camera(
    capture_id: int,
    label: str,
    out_dir: str,
    fourcc: str = "MJPG",
    warmup: int = 10,
    rotations: Optional[List[int]] = None,
) -> None:
    rotations = rotations if rotations is not None else [0, 90, 180, 270]
    cap = cv2.VideoCapture(capture_id)
    if not cap.isOpened():
        logger.error(f"[{label}] cannot open camera index {capture_id} (is device_starter still running?)")
        return
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    frame = None
    for _ in range(max(1, warmup)):  # let auto-exposure / pipeline settle
        ok, frame = cap.read()
    cap.release()

    if frame is None or not ok:
        logger.error(f"[{label}] failed to read a frame from camera index {capture_id}")
        return

    h, w = frame.shape[:2]
    logger.info(f"[{label}] native frame: index={capture_id} shape=(H={h}, W={w}) dtype={frame.dtype}")

    raw_path = os.path.join(out_dir, f"{label}_id{capture_id}_raw_{w}x{h}.png")
    cv2.imwrite(raw_path, frame)
    logger.info(f"[{label}] wrote {raw_path}")

    for deg in rotations:
        if deg == 0:
            continue
        rot = _rotate(frame, deg)
        rh, rw = rot.shape[:2]
        rot_path = os.path.join(out_dir, f"{label}_id{capture_id}_rot{deg}_{rw}x{rh}.png")
        cv2.imwrite(rot_path, rot)
        logger.info(f"[{label}] wrote {rot_path} (rot {deg} -> H={rh}, W={rw})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-capture-id", type=int, default=2, help="left wrist camera index (config camera.left_capture_id)")
    parser.add_argument("--right-capture-id", type=int, default=0, help="right wrist camera index (config camera.right_capture_id)")
    parser.add_argument("--out-dir", type=str, default="logs/raw_camera_dump")
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--warmup", type=int, default=10, help="frames to read before saving")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cams: Dict[str, int] = {"left": args.left_capture_id, "right": args.right_capture_id}
    for label, dev_id in cams.items():
        dump_camera(dev_id, label, args.out_dir, fourcc=args.fourcc, warmup=args.warmup)

    logger.info(f"Done. Inspect frames under: {args.out_dir}")


if __name__ == "__main__":
    main()
