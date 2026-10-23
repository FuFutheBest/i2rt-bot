import math
from typing import Optional

import cv2
import numpy as np


def fit_line_angle(pts: np.ndarray) -> Optional[tuple]:
    """Fit a line through >=3 points using L2 distance.

    Args:
        pts: (N, 2) float64 array of point coordinates.

    Returns:
        (angle_deg, mean_dist_to_line) or None if fitting fails.
    """
    if len(pts) < 3:
        return None

    float_pts = pts.astype(np.float32).reshape(-1, 1, 2)
    line = cv2.fitLine(float_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line[0][0], line[1][0], line[2][0], line[3][0]

    angle = math.degrees(math.atan2(float(vy), float(vx)))

    den = math.sqrt(float(vx) ** 2 + float(vy) ** 2)
    if den <= 0.0:
        return None

    dists = np.abs(vy * (pts[:, 0] - x0) - vx * (pts[:, 1] - y0)) / den
    mean_dist = float(np.mean(dists))

    return (angle, mean_dist)


def check_spacing(pts: np.ndarray) -> bool:
    """Check that 3 points are reasonably evenly spaced.

    The two shortest pair-distances must have ratio < 2.0,
    and the shortest >= 1.0 px.
    """
    assert len(pts) == 3
    d01 = float(np.linalg.norm(pts[0] - pts[1]))
    d12 = float(np.linalg.norm(pts[1] - pts[2]))
    d02 = float(np.linalg.norm(pts[0] - pts[2]))
    dists = sorted([d01, d12, d02])
    s1, s2 = dists[0], dists[1]
    return s1 >= 1.0 and (s2 / s1) < 2.0


def assign_best_3(
    score: np.ndarray, active: list, n_cands: int
) -> list:
    """Greedy-permutation assignment for 3 tracker slots.

    Args:
        score: (3, 64) float array, score[t][c] > 0 means candidate c
               is visible from tracker t.
        active: tracker indices that are not in dropout.
        n_cands: number of candidates (<= 64).

    Returns:
        [Optional[int]; 3] — best candidate index per tracker slot.
    """
    n = n_cands
    best_total = 0.0
    best = [None, None, None]

    if n >= 3 and len(active) == 3:
        t0, t1, t2 = active[0], active[1], active[2]
        for c0 in range(n):
            for c1 in range(c0 + 1, n):
                for c2 in range(c1 + 1, n):
                    for tc0, tc1, tc2 in [
                        (c0, c1, c2), (c0, c2, c1), (c1, c0, c2),
                        (c1, c2, c0), (c2, c0, c1), (c2, c1, c0),
                    ]:
                        s0 = score[t0][tc0]
                        s1 = score[t1][tc1]
                        s2 = score[t2][tc2]
                        if s0 > 0.0 and s1 > 0.0 and s2 > 0.0:
                            total = s0 + s1 + s2
                            if total > best_total:
                                best_total = total
                                best = [None, None, None]
                                best[t0] = tc0
                                best[t1] = tc1
                                best[t2] = tc2
                                if total >= 3.0:
                                    return best

    if best_total == 0.0 and n >= 2 and len(active) >= 2:
        t0, t1 = active[0], active[1]
        for c0 in range(n):
            for c1 in range(c0 + 1, n):
                for tc0, tc1 in [(c0, c1), (c1, c0)]:
                    s0 = score[t0][tc0]
                    s1 = score[t1][tc1]
                    if s0 > 0.0 and s1 > 0.0:
                        total = s0 + s1
                        if total > best_total:
                            best_total = total
                            best = [None, None, None]
                            best[t0] = tc0
                            best[t1] = tc1

    if best_total == 0.0 and n >= 1 and len(active) > 0:
        for t_idx in active:
            for c0 in range(n):
                s = score[t_idx][c0]
                if s > 0.0 and s > best_total:
                    best_total = s
                    best = [None, None, None]
                    best[t_idx] = c0

    return best

