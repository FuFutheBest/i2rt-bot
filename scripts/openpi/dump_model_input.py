#!/usr/bin/env python3
"""
Dump the wrist images that the OpenPi policy actually feeds to the model, WITHOUT
loading the policy/model (so it never touches the GPU / never OOMs).

Pipeline reproduced here (all on CPU):
  camera SHM frame (RGB, already rotated + resized to 640x480 by WebCameraOpenPi)
    -> _crop_resize_wrist: crop _WRIST_CROP_LRTB then resize to _WRIST_RESIZE_HW (320x240)
This is exactly the per-arm model input produced by
``policy/openpi_umi_policy.py::_crop_resize_wrist`` (openpi then resize-with-pads it
to 224x224 internally, which is not reproduced here).

Use this instead of ``OPENPI_DUMP_OBS=1`` when you only want to inspect the model
input image and cannot afford to load the full model.

Author: SuperInference
"""

import argparse
import os
import sys
import time
from typing import Tuple

import cv2
import numpy as np
from skimage.transform import resize

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from visualizers.base_visualizer import BaseVisualizer  # noqa: E402
from utils.shm_utils import get_dtype  # noqa: E402
from utils.logger_config import logger  # noqa: E402

# Kept in sync with policy/openpi_umi_policy.py. The crop is all-zero for nmx_data
# (no L/R wrist crop); the resize matches the training wrist size (320H x 240W).
_WRIST_CROP_LRTB: Tuple[int, int, int, int] = (0, 0, 0, 0)
_WRIST_RESIZE_HW: Tuple[int, int] = (320, 240)


def _crop_resize_wrist(img: np.ndarray) -> np.ndarray:
    """CPU-only copy of policy._crop_resize_wrist (no torch / no GPU)."""
    height, width = img.shape[:2]
    left, right, top, bottom = _WRIST_CROP_LRTB
    img = img[top : height - bottom, left : width - right]
    out_h, out_w = _WRIST_RESIZE_HW
    img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)
    return img.astype(np.uint8)


def _save_rgb(img: np.ndarray, out_path: str) -> None:
    """SHM stores RGB HWC uint8; cv2.imwrite expects BGR."""
    out = img
    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)


def dump_one(shm_name: str, side: str, out_dir: str, dtype: str, save_device_frame: bool) -> bool:
    vis = BaseVisualizer(shared_memory_name=shm_name, data_dtype=get_dtype(dtype))
    if not vis.connect():
        logger.error(f"Could not connect to {shm_name}")
        return False
    _, frame = vis.read_latest_frame()
    vis.disconnect()
    if frame is None:
        logger.error(f"No frame available in {shm_name}")
        return False

    device_frame = np.asarray(frame)
    model_input = _crop_resize_wrist(device_frame)

    out_model = os.path.join(out_dir, f"{side}_model_input_{model_input.shape[0]}x{model_input.shape[1]}.png")
    _save_rgb(model_input, out_model)
    logger.info(f"{shm_name}: device {device_frame.shape} -> model input {model_input.shape} -> {out_model}")

    if save_device_frame:
        out_dev = os.path.join(out_dir, f"{side}_device_{device_frame.shape[0]}x{device_frame.shape[1]}.png")
        _save_rgb(device_frame, out_dev)
        logger.info(f"{shm_name}: saved device frame -> {out_dev}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump OpenPi model-input wrist images from camera SHM (CPU only, no model load)."
    )
    parser.add_argument("--left-shm", default="WebCameraOpenPi_2_data",
                        help="Left wrist camera SHM name (default: WebCameraOpenPi_2_data)")
    parser.add_argument("--right-shm", default="WebCameraOpenPi_0_data",
                        help="Right wrist camera SHM name (default: WebCameraOpenPi_0_data)")
    parser.add_argument("--out-dir", default="logs/openpi_model_input")
    parser.add_argument("--dtype", default="uint8")
    parser.add_argument("--device-frame", action="store_true",
                        help="Also save the raw 640x480 SHM frame for reference.")
    parser.add_argument("--watch", type=float, default=0.0,
                        help="If > 0, re-dump every N seconds until Ctrl+C (overwrites).")
    args = parser.parse_args()

    cams = [(args.left_shm, "left"), (args.right_shm, "right")]

    def _dump_round() -> None:
        for shm_name, side in cams:
            dump_one(shm_name, side, args.out_dir, args.dtype, args.device_frame)

    if args.watch > 0:
        logger.info(f"Watch mode: re-dumping every {args.watch:.2f}s (Ctrl+C to stop)")
        try:
            while True:
                _dump_round()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            logger.info("Stopped.")
    else:
        _dump_round()


if __name__ == "__main__":
    main()
