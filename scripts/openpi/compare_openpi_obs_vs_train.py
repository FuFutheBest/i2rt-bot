#!/usr/bin/env python3
"""
Side-by-side comparison of the OpenPi wrist images fed at inference time against
the wrist frames stored in the training LeRobot dataset.

Purpose: at deployment, quickly eyeball whether the deployed left/right wrist
images match the training framing (orientation / field-of-view / aspect ratio)
and, crucially, what the Pi0 vision encoder actually sees after
`resize_with_pad(224, 224)`. This is the visual check for guide items A1/A3
(rotation direction + `_convert_obs_to_pi` output never compared to training).

Two image sources are combined per wrist:
  - inference: the PNGs dumped by `policy/openpi_umi_policy.py::_maybe_dump_obs`
    (enable with env `OPENPI_DUMP_OBS=1`); reads `left_img_*.png` / `right_img_*.png`.
  - training : a reference frame from the dataset wrist videos
    (`observation.images.wrist_image_1` -> left, `wrist_image_2` -> right),
    extracted on the fly with ffmpeg, or supplied directly via --train-left/right.

Mapping follows the policy: cam_keys[0] -> left -> wrist_image_1,
cam_keys[1] -> right -> wrist_image_2.

The 224x224 panels replicate openpi's `resize_with_pad` (PIL, bilinear, zero pad)
so the comparison reflects the true model input, not just the raw dump.

Author: Wang Zheng
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

# Allow running as `python scripts/openpi/compare_openpi_obs_vs_train.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_config import logger

# Dataset video keys, in (left, right) order matching the policy.
_LEFT_VIDEO_KEY = "observation.images.wrist_image_1"
_RIGHT_VIDEO_KEY = "observation.images.wrist_image_2"

_MODEL_INPUT_HW = (224, 224)  # openpi ResizeImages target


def resize_with_pad_pil(image: Image.Image, height: int, width: int) -> Image.Image:
    """Replicate openpi_client.image_tools.resize_with_pad for a single PIL image.

    Resizes preserving aspect ratio (bilinear) then zero-pads to (width, height),
    matching exactly what the Pi0 model transform applies to every camera image.
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return zero_image


def _latest_dump(obs_dump_dir: str, name: str, index: Optional[int]) -> str:
    """Locate a dumped PNG `<name>_<idx>.png`; pick `index` or the highest idx."""
    if index is not None:
        path = os.path.join(obs_dump_dir, f"{name}_{index:03d}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dump not found: {path}")
        return path
    candidates = glob.glob(os.path.join(obs_dump_dir, f"{name}_*.png"))
    if not candidates:
        raise FileNotFoundError(
            f"No '{name}_*.png' in {obs_dump_dir}. "
            f"Run inference with env OPENPI_DUMP_OBS=1 first."
        )

    def _idx(p: str) -> int:
        m = re.search(rf"{re.escape(name)}_(\d+)\.png$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(candidates, key=_idx)


def _extract_train_frame(
    dataset_dir: str, video_key: str, episode: int, frame: int, out_dir: str
) -> str:
    """Extract a single frame from a dataset wrist video via ffmpeg."""
    chunk = episode // 1000  # info.json chunks_size = 1000
    video = os.path.join(
        dataset_dir,
        "videos",
        f"chunk-{chunk:03d}",
        video_key,
        f"episode_{episode:06d}.mp4",
    )
    if not os.path.exists(video):
        raise FileNotFoundError(f"Training video not found: {video}")
    out_png = os.path.join(out_dir, f"{video_key.split('.')[-1]}_ep{episode}_f{frame}.png")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video,
        "-vf", f"select=eq(n\\,{frame})",
        "-vframes", "1",
        out_png,
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(out_png):
        raise RuntimeError(f"ffmpeg produced no frame for {video} (frame {frame}).")
    return out_png


def _load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _make_tile(
    img: Image.Image, label: str, tile_w: int, tile_h: int
) -> Image.Image:
    """Render `img` centered inside a tile (aspect preserved) with a top label."""
    bg = (24, 24, 24)
    label_h = 18
    tile = Image.new("RGB", (tile_w, tile_h), bg)
    box_w, box_h = tile_w - 8, tile_h - label_h - 8
    iw, ih = img.size
    scale = min(box_w / iw, box_h / ih)
    disp = img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), Image.NEAREST)
    ox = (tile_w - disp.size[0]) // 2
    oy = label_h + (box_h - disp.size[1]) // 2
    tile.paste(disp, (ox, oy))
    draw = ImageDraw.Draw(tile)
    draw.text((4, 4), f"{label} {iw}x{ih}", fill=(0, 255, 120))
    return tile


def build_comparison(
    train_left: Image.Image,
    infer_left: Image.Image,
    train_right: Image.Image,
    infer_right: Image.Image,
    out_path: str,
) -> None:
    """Build a 2x4 grid: rows = left/right wrist, cols = train | infer | train->224 | infer->224."""
    tile_w, tile_h = 240, 320
    rows: List[List[Tuple[Image.Image, str]]] = [
        [
            (train_left, "L train"),
            (infer_left, "L infer"),
            (resize_with_pad_pil(train_left, *_MODEL_INPUT_HW), "L train->model"),
            (resize_with_pad_pil(infer_left, *_MODEL_INPUT_HW), "L infer->model"),
        ],
        [
            (train_right, "R train"),
            (infer_right, "R infer"),
            (resize_with_pad_pil(train_right, *_MODEL_INPUT_HW), "R train->model"),
            (resize_with_pad_pil(infer_right, *_MODEL_INPUT_HW), "R infer->model"),
        ],
    ]
    n_cols = len(rows[0])
    n_rows = len(rows)
    canvas = Image.new("RGB", (tile_w * n_cols, tile_h * n_rows), (0, 0, 0))
    for r, row in enumerate(rows):
        for c, (img, label) in enumerate(row):
            tile = _make_tile(img, label, tile_w, tile_h)
            canvas.paste(tile, (c * tile_w, r * tile_h))
    canvas.save(out_path)
    logger.info(f"Saved comparison grid -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare deployed OpenPi wrist images vs training dataset frames."
    )
    p.add_argument(
        "--obs-dump", default="logs/openpi_obs_dump",
        help="Dir with inference dumps (left_img_*.png / right_img_*.png).",
    )
    p.add_argument(
        "--dump-index", type=int, default=None,
        help="Specific dump index (000, 001, ...). Default: latest available.",
    )
    p.add_argument(
        "--use-device-frames", action="store_true",
        help="Compare against the device SHM dumps (left_device/right_device, "
             "pre-policy 640x480) instead of the final policy inputs (left_img/right_img).",
    )
    # Training reference: either a dataset to extract from, or explicit PNGs.
    p.add_argument(
        "--train-dataset", default=None,
        help="LeRobot dataset root (contains meta/ and videos/) to extract a reference frame from.",
    )
    p.add_argument("--episode", type=int, default=0, help="Training episode index for the reference frame.")
    p.add_argument("--frame", type=int, default=60, help="Training frame number for the reference frame.")
    p.add_argument("--train-left", default=None, help="Explicit left-wrist training reference PNG (overrides dataset).")
    p.add_argument("--train-right", default=None, help="Explicit right-wrist training reference PNG (overrides dataset).")
    p.add_argument(
        "--out", default=None,
        help="Output comparison PNG. Default: <obs-dump>/compare_train_vs_infer.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    infer_left_name = "left_device" if args.use_device_frames else "left_img"
    infer_right_name = "right_device" if args.use_device_frames else "right_img"
    infer_left = _load(_latest_dump(args.obs_dump, infer_left_name, args.dump_index))
    infer_right = _load(_latest_dump(args.obs_dump, infer_right_name, args.dump_index))
    logger.info(
        f"Inference frames: {infer_left_name}={infer_left.size}, "
        f"{infer_right_name}={infer_right.size}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        if args.train_left and args.train_right:
            train_left_path, train_right_path = args.train_left, args.train_right
        elif args.train_dataset:
            train_left_path = _extract_train_frame(
                args.train_dataset, _LEFT_VIDEO_KEY, args.episode, args.frame, tmp
            )
            train_right_path = _extract_train_frame(
                args.train_dataset, _RIGHT_VIDEO_KEY, args.episode, args.frame, tmp
            )
        else:
            raise SystemExit(
                "Provide either --train-dataset (to extract) or --train-left/--train-right (explicit PNGs)."
            )
        train_left = _load(train_left_path)
        train_right = _load(train_right_path)
        logger.info(f"Training frames: left={train_left.size}, right={train_right.size}")

        out_path = args.out or os.path.join(args.obs_dump, "compare_train_vs_infer.png")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        build_comparison(train_left, infer_left, train_right, infer_right, out_path)


if __name__ == "__main__":
    main()
