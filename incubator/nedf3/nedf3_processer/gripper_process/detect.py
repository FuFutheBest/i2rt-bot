import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Cfg
from . import util


def _resize_mask_to_frame(mask: np.ndarray, frame_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a single-channel ROI mask to (H, W) of the frame when sizes differ."""
    fh, fw = frame_hw
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    mh, mw = int(mask.shape[0]), int(mask.shape[1])
    if mh == fh and mw == fw:
        return mask
    return cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)


@dataclass
class Detection:
    pts: np.ndarray   # (N, 2) float64
    target_pos: np.ndarray  # (2,) float64
    angle: float


class FramePrep:
    def __init__(self, cfg: Cfg):
        self.lower = np.array([cfg.hsv_lower[0], cfg.hsv_lower[1], cfg.hsv_lower[2]], dtype=np.uint8)
        self.upper = np.array([cfg.hsv_upper[0], cfg.hsv_upper[1], cfg.hsv_upper[2]], dtype=np.uint8)


DEBUG = False


def detect_frame(
    frame: np.ndarray,
    masks: list,
    prep: FramePrep,
    cfg: Cfg,
    predictions: Optional[np.ndarray] = None,
    out_debug_viz: Optional[np.ndarray] = None,
) -> Optional[Detection]:
    """Detect 3 collinear, evenly-spaced markers in a BGR frame.

    Args:
        frame: BGR image (H, W, 3) uint8.
        masks: list of 3 ROI mask images (H, W) uint8; may differ from ``frame``
            resolution (e.g. masks from full-res prep, frames downscaled for NEDF3)
            — they are resized to the frame size before use.
        prep: FramePrep with HSV bounds.
        cfg: detection configuration.
        predictions: optional (3, 2) Kalman predictions (unused, for API compat).
        out_debug_viz: optional BGR image to draw debug overlay onto.

    Returns:
        Detection if a valid triple is found, else None.
    """
    _ = predictions  # kept for API compatibility

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, prep.lower, prep.upper)

    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    scaled_masks: List[np.ndarray] = [
        _resize_mask_to_frame(m, (fh, fw)) for m in masks
    ]
    roi_01 = cv2.bitwise_or(scaled_masks[0], scaled_masks[1])
    total_roi = cv2.bitwise_or(roi_01, scaled_masks[2])
    pure_mask = cv2.bitwise_and(color_mask, total_roi)

    if out_debug_viz is not None and DEBUG:
        out_debug_viz[:] = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        roi_contours, _ = cv2.findContours(total_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_debug_viz, roi_contours, -1, (255, 255, 0), 2)

    mask_for_contour = pure_mask
    if cfg.erode_px > 0:
        k = cfg.erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask_for_contour = cv2.erode(pure_mask, kernel)

    contours, _ = cv2.findContours(mask_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circularity = (4.0 * math.pi * area / (peri * peri)) if peri > 0.0 else 0.0

        if area > 1.0 and DEBUG:
            print(f"Found blob {idx}: area={area:.2f}, circularity={circularity:.2f}")

        if area <= 80.0:
            continue
        if area < cfg.min_contour_area:
            continue

        m = cv2.moments(cnt)
        if m["m00"] <= 0.0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        candidates.append((np.array([cx, cy]), area))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:15]
    n = len(candidates)
    if n < 3:
        return None

    best_det = None
    min_line_err = float("inf")

    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                pts = np.array([candidates[i][0], candidates[j][0], candidates[k][0]])
                pts = pts[pts[:, 0].argsort()]  # sort by x

                if not util.check_spacing(pts):
                    continue
                result = util.fit_line_angle(pts)
                if result is None:
                    continue
                angle, mean_dist = result
                if mean_dist > 3.0:
                    continue

                if mean_dist < min_line_err:
                    min_line_err = mean_dist
                    best_det = Detection(
                        pts=pts.copy(),
                        target_pos=pts[cfg.target_mask_index].copy(),
                        angle=angle,
                    )

    return best_det

