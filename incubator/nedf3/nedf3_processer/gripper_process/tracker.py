from typing import Optional

import cv2
import numpy as np

from .config import Cfg
from .detect import Detection, FramePrep, detect_frame
from . import util


class Kalman2D:
    """Simple 1D-per-axis Kalman filter for 2D point tracking."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.px = 25.0
        self.py = 25.0
        self.q = 1.0
        self.r = 4.0

    def predict(self):
        self.px += self.q
        self.py += self.q

    def update(self, mx: float, my: float):
        kx = self.px / (self.px + self.r)
        ky = self.py / (self.py + self.r)
        self.x += kx * (mx - self.x)
        self.y += ky * (my - self.y)
        self.px *= 1.0 - kx
        self.py *= 1.0 - ky


class Tracker:
    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.tracked_pts: Optional[np.ndarray] = None  # (N, 2) float32 for optical flow
        self.kalmans: list = [None, None, None]
        self.dropout_count: int = 0

    def reset(self):
        self.prev_gray = None
        self.tracked_pts = None
        self.kalmans = [None, None, None]
        self.dropout_count = 0

    def update(
        self,
        frame: np.ndarray,
        masks: list,
        prep: FramePrep,
        cfg: Cfg,
        out_debug_viz: np.ndarray,
    ) -> Optional[float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        predictions = self._predict_kalmans()
        detection = detect_frame(frame, masks, prep, cfg, predictions, out_debug_viz)

        if detection is not None:
            result = self._on_detection(detection, cfg)
        else:
            result = self._on_no_detection(gray, masks, cfg)

        self.prev_gray = gray
        return result

    def _predict_kalmans(self) -> Optional[np.ndarray]:
        for k in self.kalmans:
            if k is not None:
                k.predict()
        if any(k is None for k in self.kalmans):
            return None
        preds = np.array([[k.x, k.y] for k in self.kalmans], dtype=np.float64)
        return preds

    def _on_detection(self, det: Detection, cfg: Cfg) -> Optional[float]:
        self.dropout_count = 0
        for i in range(min(3, len(det.pts))):
            pt = det.pts[i]
            if self.kalmans[i] is not None:
                dist = float(np.linalg.norm(pt - np.array([self.kalmans[i].x, self.kalmans[i].y])))
                if dist < cfg.gate_radius:
                    self.kalmans[i].update(float(pt[0]), float(pt[1]))
                else:
                    self.kalmans[i] = Kalman2D(float(pt[0]), float(pt[1]))
            else:
                self.kalmans[i] = Kalman2D(float(pt[0]), float(pt[1]))

        self.tracked_pts = det.pts[:3].astype(np.float32).copy()
        return det.angle

    def _on_no_detection(self, curr_gray: np.ndarray, _masks: list, cfg: Cfg) -> Optional[float]:
        if self.dropout_count >= cfg.max_dropout_frames:
            self._reset_tracking()
            return None

        if self.prev_gray is None or self.tracked_pts is None or len(self.tracked_pts) < 3:
            return None

        prev_pts = self.tracked_pts[:3].reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.01),
            minEigThreshold=1e-4,
        )

        if status is None or not np.all(status[:3] == 1) or next_pts.shape[0] < 3:
            self._reset_tracking()
            return None

        new_pts = next_pts[:3].reshape(3, 2).astype(np.float64)
        if not util.check_spacing(new_pts):
            self._reset_tracking()
            return None

        result = util.fit_line_angle(new_pts)
        if result is None:
            self._reset_tracking()
            return None
        angle, mean_dist = result
        if mean_dist > 5.0:
            self._reset_tracking()
            return None

        self.dropout_count += 1
        self.tracked_pts = next_pts[:3].reshape(3, 2).astype(np.float32).copy()
        return angle

    def _reset_tracking(self):
        self.tracked_pts = None
        self.kalmans = [None, None, None]
        self.dropout_count = 0

