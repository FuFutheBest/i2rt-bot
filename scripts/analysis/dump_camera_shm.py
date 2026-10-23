#!/usr/bin/env python3
"""
Dump the latest frame of one or more camera shared-memory buffers to PNG files.

Reads the post-rotation / post-resize device output that feeds the policy, so the
saved PNGs can be compared 1:1 with the training wrist frames. With --wrist it
also writes the exact image that enters the model (after _crop_resize_wrist),
WITHOUT loading the policy model, so it is a cheap way to inspect the model input.

Author: SuperInference
"""

import argparse
import os
import sys
from typing import List

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizers.base_visualizer import BaseVisualizer  # noqa: E402
from utils.shm_utils import get_dtype  # noqa: E402
from utils.logger_config import logger  # noqa: E402


def dump_one(shm_name: str, out_path: str, dtype: str, wrist: bool = False) -> bool:
    vis = BaseVisualizer(shared_memory_name=shm_name, data_dtype=get_dtype(dtype))
    if not vis.connect():
        logger.error(f"Could not connect to {shm_name}")
        return False
    ts, frame = vis.read_latest_frame()
    vis.disconnect()
    if frame is None:
        logger.error(f"No frame available in {shm_name}")
        return False
    img = np.asarray(frame)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Always write the raw device SHM frame (rotated + resized to data_shape).
    device_img = img
    if device_img.ndim == 3 and device_img.shape[2] == 3:
        device_img = cv2.cvtColor(device_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, device_img)
    logger.info(f"{shm_name}: device frame shape={img.shape} -> {out_path}")

    # Optionally also write the exact model-input image (post _crop_resize_wrist).
    # Imported lazily so the script never needs to load the policy model itself.
    if wrist:
        from policy.openpi_umi_policy import _as_uint8_hwc, _crop_resize_wrist

        model_input = _crop_resize_wrist(_as_uint8_hwc(img))
        bgr = model_input
        if bgr.ndim == 3 and bgr.shape[2] == 3:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        root, ext = os.path.splitext(out_path)
        model_path = f"{root}_model_input{ext}"
        cv2.imwrite(model_path, bgr)
        logger.info(f"{shm_name}: model input shape={model_input.shape} -> {model_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump camera SHM frames to PNG")
    parser.add_argument("--shm", nargs="+", required=True, help="SHM names, e.g. WebCameraOpenPi_2_data")
    parser.add_argument("--out-dir", default="logs/cam_dump")
    parser.add_argument("--dtype", default="uint8")
    parser.add_argument(
        "--wrist",
        action="store_true",
        help="Also write the exact model-input image after _crop_resize_wrist (no model load).",
    )
    args = parser.parse_args()

    names: List[str] = args.shm
    for name in names:
        out = os.path.join(args.out_dir, f"{name}.png")
        dump_one(name, out, args.dtype, wrist=args.wrist)


if __name__ == "__main__":
    main()
