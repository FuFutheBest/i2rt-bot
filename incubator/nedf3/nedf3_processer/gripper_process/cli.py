"""CLI entry point for gripper angle tracking.

Usage:
    python -m process_json export <test_video> <json_output> [masks_dir]

The module can also be used programmatically — import from process.* directly
for lab or DM3 integration with zero CLI dependency.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import GripperDetector, DesktopFrameResult
from .plot import save_angle_curve_plot


DEFAULT_MASKS_DIR = "./output_masks/left"
START_FRAME_IDX = 1
END_FRAME_IDX = 140
PAUSE_FRAMES = 300

DEFAULT_CFG_JSON = """{
    "erode_px": 0,
    "min_contour_area": 1.5,
    "roi_padding": 25
}"""


def _build_curve_output_path(video_path: str) -> Path:
    stem = Path(video_path).stem or "angle"
    return Path(f"{stem}_angle_curve.png")


def _read_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps):
        return 30.0
    return fps


def _ensure_parent_dir(path: Path):
    if path.parent != Path():
        os.makedirs(path.parent, exist_ok=True)


def _to_three_points(tracked_pts: Optional[list]) -> Optional[list]:
    if tracked_pts is None or len(tracked_pts) < 3:
        return None
    return [tracked_pts[0], tracked_pts[1], tracked_pts[2]]


def _format_optional(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


# ── Public API ──────────────────────────────────────────────────────────────────

def run_tracker(
    video_path: str,
    masks_dir: str = DEFAULT_MASKS_DIR,
    cfg_json: str = "",
    headless: bool = False,
) -> list:
    """Track gripper angle through a video and save the angle curve.

    Args:
        video_path: path to test MP4.
        masks_dir: directory containing roi_mask_{0,1,2}.png.
        cfg_json: optional JSON config override.
        headless: if True, skip the per-frame pause window.

    Returns:
        List of (time_sec, angle_deg_or_None) samples.
    """
    cfg = cfg_json if cfg_json else DEFAULT_CFG_JSON
    detector = GripperDetector(masks_dir, cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    curve_output = _build_curve_output_path(video_path)
    angle_samples = []

    fps = _read_fps(cap)
    frame_window_len = END_FRAME_IDX - START_FRAME_IDX + 1
    pause_frame_limit = min(PAUSE_FRAMES, frame_window_len)
    print(
        f"Processing frames {START_FRAME_IDX}-{END_FRAME_IDX}, "
        f"pausing after each of the first {pause_frame_limit} frames in that window, "
        f"and exporting angle curve..."
    )

    frame_count = 0
    processed_count = 0

    while frame_count < END_FRAME_IDX:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        frame_count += 1

        if frame_count < START_FRAME_IDX:
            continue

        processed_count += 1
        time_sec = (frame_count - 1) / fps

        mask_viz = np.zeros_like(frame)
        angle_opt = detector.process_and_draw(frame, mask_viz)
        angle_samples.append((time_sec, angle_opt))

        _draw_angle_status(frame, angle_opt)

        if processed_count <= pause_frame_limit and not headless:
            if mask_viz is None or mask_viz.size == 0:
                mask_viz = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            should_continue = _pause_after_frame(
                frame_count, processed_count, frame_window_len, frame, mask_viz
            )
            if not should_continue:
                break

    cap.release()
    cv2.destroyAllWindows()

    save_angle_curve_plot(angle_samples, str(curve_output))

    print(
        f"Finished. Processed {processed_count} frames from window "
        f"{START_FRAME_IDX}-{END_FRAME_IDX}."
    )
    print(f"Angle curve saved to {curve_output}")
    return angle_samples


def export_tracking_json(
    video_path: str,
    json_output_path: str,
    masks_dir: str = DEFAULT_MASKS_DIR,
    cfg_json: str = "",
) -> int:
    """Export per-frame angle data to JSON.

    Returns:
        Number of frames exported.
    """
    cfg = cfg_json if cfg_json else DEFAULT_CFG_JSON
    detector = GripperDetector(masks_dir, cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = _read_fps(cap)
    output_path = Path(json_output_path)
    _ensure_parent_dir(output_path)

    detected_count = 0
    frame_idx = 0
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        frame_idx += 1
        time_sec = (frame_idx - 1) / fps

        result = detector.process_for_export(frame)
        pts = _to_three_points(result.tracked_pts)
        detected = result.angle_deg is not None and pts is not None

        if detected:
            detected_count += 1

        frames.append({
            "frame_idx": frame_idx,
            "time_sec": round(time_sec, 6),
            "angle_deg": None if result.angle_deg is None else round(float(result.angle_deg), 6),
            "detected": bool(detected),
        })

    payload = {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_idx,
        "detected_count": detected_count,
        "frames": frames,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    cap.release()
    print(
        f"Exported {frame_idx} frames to JSON {output_path} "
        f"(detected {detected_count}/{frame_idx})."
    )
    return frame_idx


# ── Internal helpers ─────────────────────────────────────────────────────────────

def _draw_angle_status(frame: np.ndarray, angle_opt: Optional[float]):
    if angle_opt is not None:
        text = f"Angle: {angle_opt:.2f} deg"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "LOST (Check Console)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)


def _pause_after_frame(
    frame_idx: int,
    window_idx: int,
    window_len: int,
    frame: np.ndarray,
    mask_viz: np.ndarray,
) -> bool:
    cv2.imshow("Debug: HSV Mask + ROI", mask_viz)
    cv2.imshow("Tracker Debug View", frame)
    print(
        f"Paused after frame {frame_idx} (window {window_idx}/{window_len}). "
        f"Press any key to continue, or q to quit."
    )
    key = cv2.waitKey(0) & 0xFF
    return key != ord("q") and key != ord("Q")


def _print_usage():
    print("=== Tracker CLI ===")
    print("Usage:")
    print("  python -m process_json export <test_video.mp4> <json_output_path> [masks_dir]")


# ── CLI entry ────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None):
    args = argv or sys.argv[1:]
    if len(args) < 1:
        _print_usage()
        return

    command = args[0]

    if command == "run":
        if len(args) < 2:
            _print_usage()
            return
        video_path = args[1]
        masks_dir = args[2] if len(args) > 2 else DEFAULT_MASKS_DIR
        run_tracker(video_path, masks_dir)

    elif command == "export":
        if len(args) < 3:
            _print_usage()
            return
        video_path = args[1]
        json_output = args[2]
        masks_dir = args[3] if len(args) > 3 else DEFAULT_MASKS_DIR
        export_tracking_json(video_path, json_output, masks_dir)

    else:
        _print_usage()


if __name__ == "__main__":
    main()

