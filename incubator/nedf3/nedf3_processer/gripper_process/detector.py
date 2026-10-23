import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import Cfg
from .detect import FramePrep
from .tracker import Tracker


@dataclass
class DesktopFrameResult:
    angle_deg: Optional[float]
    tracked_pts: Optional[list]  # list of [x, y] pairs


class GripperDetector:
    """High-level gripper angle detector.

    Loads 3 ROI mask PNGs and tracks green markers frame-by-frame
    using Kalman filtering with optical-flow fallback.

    Usage:
        detector = GripperDetector(masks_dir, cfg_json)
        angle = detector.process_and_draw(frame, debug_viz)
        # or for headless export:
        result = detector.process_for_export(frame)
    """

    def __init__(self, masks_dir: str, cfg_json: str = ""):
        self.cfg = Cfg.from_json_or_default(cfg_json)
        self._prep = FramePrep(self.cfg)

        self._masks = []
        for i in range(3):
            path = os.path.join(masks_dir, f"roi_mask_{i}.png")
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                raise FileNotFoundError(f"roi_mask_{i}.png not found in {masks_dir}")
            self._masks.append(mask)

        self._tracker = Tracker()

    def reset(self):
        self._tracker.reset()

    def process_bgra(self, bgra: bytes, width: int, height: int) -> Optional[float]:
        """Process a BGRA frame from iOS camera.

        Args:
            bgra: raw BGRA pixel bytes.
            width: frame width in pixels.
            height: frame height in pixels.

        Returns:
            Angle in degrees, or None if tracking lost.
        """
        expected_len = width * height * 4
        if len(bgra) != expected_len:
            return None

        bgra_arr = np.frombuffer(bgra, dtype=np.uint8).reshape((height, width, 4))
        bgr = cv2.cvtColor(bgra_arr, cv2.COLOR_BGRA2BGR)
        dummy_viz = np.zeros((height, width, 3), dtype=np.uint8)
        return self._tracker.update(bgr, self._masks, self._prep, self.cfg, dummy_viz)

    def process_and_draw(
        self, frame: np.ndarray, out_debug_viz: np.ndarray
    ) -> Optional[float]:
        """Process a BGR frame and draw tracked points on it.

        Args:
            frame: BGR image (modified in-place with overlay).
            out_debug_viz: BGR image for debug visualization (modified in-place).

        Returns:
            Angle in degrees, or None if tracking lost.
        """
        angle = self._tracker.update(frame, self._masks, self._prep, self.cfg, out_debug_viz)

        pts = self._tracker.tracked_pts
        if pts is not None:
            for pt in pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), 2, cv2.LINE_AA)
            if len(pts) >= 3:
                cv2.line(
                    frame,
                    (int(pts[0][0]), int(pts[0][1])),
                    (int(pts[2][0]), int(pts[2][1])),
                    (255, 0, 0), 2, cv2.LINE_AA,
                )

        return angle

    def process_for_export(self, frame: np.ndarray) -> DesktopFrameResult:
        """Process a BGR frame for CSV export (no drawing).

        Args:
            frame: BGR image (not modified).

        Returns:
            DesktopFrameResult with angle and tracked points.
        """
        dummy_viz = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        angle_deg = self._tracker.update(frame, self._masks, self._prep, self.cfg, dummy_viz)

        pts = self._tracker.tracked_pts
        tracked_list = None
        if pts is not None:
            tracked_list = [[float(p[0]), float(p[1])] for p in pts]

        return DesktopFrameResult(angle_deg=angle_deg, tracked_pts=tracked_list)

