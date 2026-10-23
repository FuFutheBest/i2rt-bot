import os
from typing import Optional

import cv2
import numpy as np


def save_angle_curve_plot(
    samples: list,
    output_path: str,
    title: str = "Angle vs Time",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Save an angle-vs-time curve as a PNG image.

    Args:
        samples: list of (time_sec, angle_deg_or_None) tuples.
        output_path: path to save the PNG.
        title: plot title.
        width: canvas width in pixels.
        height: canvas height in pixels.
    """
    margin_left = 110
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, title, (margin_left, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)

    valid_count = sum(1 for _, a in samples if a is not None)
    subtitle = f"Valid samples: {valid_count}/{len(samples)}"
    cv2.putText(canvas, subtitle, (margin_left, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom),
                  (60, 60, 60), 2, cv2.LINE_AA)

    if len(samples) == 0 or valid_count == 0:
        cv2.putText(canvas, "No valid angle samples were detected.",
                    (margin_left, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(output_path, canvas)
        return

    valid_angles = [a for _, a in samples if a is not None]
    min_angle = min(valid_angles)
    max_angle = max(valid_angles)
    raw_span = abs(max_angle - min_angle)
    padding = 5.0 if raw_span < 1.0 else raw_span * 0.1
    min_angle -= padding
    max_angle += padding

    time_max = max(samples[-1][0], 1e-6)
    angle_span = max(max_angle - min_angle, 1e-6)

    # Y-axis ticks
    for tick_idx in range(7):
        ratio = tick_idx / 6.0
        y = int(plot_top + ratio * (plot_bottom - plot_top))
        label_angle = max_angle - ratio * angle_span

        cv2.line(canvas, (plot_left, y), (plot_right, y),
                 (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{label_angle:.1f}", (25, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    # X-axis ticks
    for tick_idx in range(7):
        ratio = tick_idx / 6.0
        x = int(plot_left + ratio * (plot_right - plot_left))
        label_time = ratio * time_max

        cv2.line(canvas, (x, plot_top), (x, plot_bottom),
                 (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{label_time:.2f}s", (x - 22, plot_bottom + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Angle (deg)", (20, plot_top - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Time (s)", (plot_right - 80, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    # Data points
    x_span = float(plot_right - plot_left)
    y_span = float(plot_bottom - plot_top)
    prev_point: Optional[tuple] = None

    for time_sec, angle_deg in samples:
        if angle_deg is None:
            prev_point = None
            continue

        x_ratio = time_sec / time_max if time_max > 1e-6 else 0.0
        y_ratio = (max_angle - angle_deg) / angle_span
        point = (int(plot_left + x_ratio * x_span),
                 int(plot_top + y_ratio * y_span))

        cv2.circle(canvas, point, 2, (255, 120, 0), -1, cv2.LINE_AA)

        if prev_point is not None:
            cv2.line(canvas, prev_point, point, (255, 120, 0), 2, cv2.LINE_AA)

        prev_point = point

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)

