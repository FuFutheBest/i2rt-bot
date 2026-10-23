#!/usr/bin/env python3
"""
Preview /dev/video0-3 side by side for quick camera identification.

Author: Jun Lv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logger_config import logger


def _open_capture(index: int, backend: Optional[int]) -> Optional[cv2.VideoCapture]:
    if backend is not None:
        capture = cv2.VideoCapture(index, backend)
    else:
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def _read_frame(capture: cv2.VideoCapture) -> Optional[np.ndarray]:
    for _ in range(3):
        capture.read()
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


def _placeholder_panel(label: str, width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        label,
        (12, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "no frame",
        (12, height // 2 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


def _resize_panel(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid frame shape for resize")
    scale = float(target_height) / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _annotate_panel(frame: np.ndarray, label: str) -> np.ndarray:
    annotated = frame.copy()
    cv2.putText(
        annotated,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def collect_panels(
    indices: List[int],
    *,
    panel_height: int,
    backend: Optional[int],
) -> List[np.ndarray]:
    panels: List[np.ndarray] = []
    panel_width = int(panel_height * 16 / 9)

    for index in indices:
        label = f"video{index}"
        capture = _open_capture(index, backend)
        if capture is None:
            logger.warning(f"{label}: failed to open")
            panels.append(_placeholder_panel(label, panel_width, panel_height))
            continue

        try:
            frame = _read_frame(capture)
            if frame is None:
                logger.warning(f"{label}: opened but no frame")
                panels.append(_placeholder_panel(label, panel_width, panel_height))
                continue
            panel = _resize_panel(frame, panel_height)
            panels.append(_annotate_panel(panel, label))
            logger.info(f"{label}: frame shape={frame.shape}")
        finally:
            capture.release()

    return panels


def build_montage(panels: List[np.ndarray]) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to display")
    return np.hstack(panels)


def preview_devices(
    indices: List[int],
    *,
    duration_s: float,
    panel_height: int,
    backend: Optional[int],
) -> None:
    panels = collect_panels(indices, panel_height=panel_height, backend=backend)
    montage = build_montage(panels)
    window_name = "video preview"
    cv2.imshow(window_name, montage)
    cv2.waitKey(1)
    time.sleep(max(0.0, float(duration_s)))
    cv2.destroyWindow(window_name)


def parse_indices(tokens: List[str]) -> List[int]:
    """Parse indices from CLI tokens like: 0 2, 0,2, or [0,2]."""
    if not tokens:
        return [0, 1, 2, 3]

    joined = " ".join(tokens).strip().strip("[]")
    raw_parts: List[str] = []
    for chunk in joined.replace(";", ",").split(","):
        raw_parts.extend(chunk.split())

    indices: List[int] = []
    for part in raw_parts:
        cleaned = part.strip()
        if cleaned:
            indices.append(int(cleaned))

    if not indices:
        raise ValueError("No valid capture indices provided")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview selected video indices side by side with OpenCV imshow."
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=["0", "1", "2", "3"],
        metavar="N",
        help=(
            "OpenCV capture indices to preview. "
            "Examples: --indices 0 2 | --indices 0,2 | --indices [0,2]"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Display duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=360,
        help="Resize each panel to this height before stitching (default: 360)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "v4l2"],
        default="v4l2",
        help="OpenCV capture backend (default: v4l2)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    backend = cv2.CAP_V4L2 if args.backend == "v4l2" else None
    preview_devices(
        parse_indices(list(args.indices)),
        duration_s=float(args.duration),
        panel_height=int(args.panel_height),
        backend=backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
