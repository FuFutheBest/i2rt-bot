#!/usr/bin/env python3
"""
Plot velocity profiles from trajectory run record JSON files.

Reads the same schema as ``scripts/benchmark/waypoint_trajectory_analyzer.py`` (traj_run_record).
Computes numerical derivatives for joint commands (``dense_command_traj``,
``sparse_waypoint_plan.joints``, ``recorded_samples.joints``) and optional TCP
linear velocity from ``sparse_waypoint_plan.tcp_pose`` / ``recorded_samples.tcp_pose``.

Does not require Pinocchio/FK; suitable for large planner dense traces.

By default the plot/stats time window is the **intersection** of
``dense_command_traj``, ``sparse_waypoint_plan`` (joints and/or tcp_pose when
present), and ``recorded_samples`` (joints and/or tcp_pose), matching
``scripts/benchmark/waypoint_trajectory_analyzer.py`` overlay alignment. Leading/trailing
segments where only one stream exists are omitted so figures are less redundant.
Use ``--t-start`` / ``--t-end`` to narrow further; ``--stream-intersection-only``
falls back to intersecting only the requested ``--stream`` series.

Exports ``window_min_speed`` stats: per ``--window-period`` (default 0.03 s), the
fraction of windows whose minimum speed is below threshold
(``frac_windows_min_below_threshold``) — use this to A/B planner vs no_planner.

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from utils.logger_config import logger

TRAJ_RUN_RECORD_KEYS: Tuple[str, ...] = (
    "traj_space",
    "sparse_waypoint_plan",
    "dense_command_traj",
    "recorded_samples",
    "urdf_path",
    "ee_frame",
    "world_frame_transform",
    "tcp_frame_transform",
)

STREAM_DENSE_JOINT = "dense_joint"
STREAM_SPARSE_JOINT = "sparse_joint"
STREAM_SPARSE_TCP = "sparse_tcp"
STREAM_SAMPLE_JOINT = "sample_joint"
STREAM_SAMPLE_TCP = "sample_tcp"
STREAM_CHOICES: Tuple[str, ...] = (
    STREAM_DENSE_JOINT,
    STREAM_SPARSE_JOINT,
    STREAM_SPARSE_TCP,
    STREAM_SAMPLE_JOINT,
    STREAM_SAMPLE_TCP,
)

WaypointSeries = List[List[Any]]
TimedSeries = Tuple[np.ndarray, np.ndarray]  # (t_abs, values (N, D))

_PLOT_COLORS: Dict[str, str] = {
    STREAM_DENSE_JOINT: "#C44E52",
    STREAM_SPARSE_JOINT: "#DD8452",
    STREAM_SPARSE_TCP: "#DD8452",
    STREAM_SAMPLE_JOINT: "#4C72B0",
    STREAM_SAMPLE_TCP: "#4C72B0",
}


def load_traj_run_record(path: Path) -> Dict[str, Any]:
    """Load and validate the trajectory run record JSON."""
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError("Record root must be a JSON object")
    missing = [key for key in TRAJ_RUN_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f"Record missing keys: {missing}")
    return record


def _series_to_arrays(series: WaypointSeries, value_dim: int, name: str) -> TimedSeries:
    """Convert ``[[t, values], ...]`` to sorted numpy arrays."""
    if not isinstance(series, list) or len(series) < 2:
        raise ValueError(f"{name} must be a list with at least two samples")
    times: List[float] = []
    values: List[np.ndarray] = []
    prev_t = -math.inf
    for idx, item in enumerate(series):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name}[{idx}] must be [time, values]")
        t = float(item[0])
        v = np.asarray(item[1], dtype=np.float64).reshape(-1)
        if int(v.size) != value_dim:
            raise ValueError(f"{name}[{idx}] expected length {value_dim}, got {int(v.size)}")
        if t <= prev_t:
            raise ValueError(f"{name} times must be strictly increasing (idx={idx})")
        prev_t = t
        times.append(t)
        values.append(v)
    t_arr = np.asarray(times, dtype=np.float64)
    v_arr = np.stack(values, axis=0)
    order = np.argsort(t_arr)
    return t_arr[order], v_arr[order]


def _try_series(series: Any, value_dim: int, name: str) -> Optional[TimedSeries]:
    """Parse series or return ``None`` when fewer than two samples."""
    if not isinstance(series, list) or len(series) < 2:
        return None
    return _series_to_arrays(series, value_dim, name)


def _finite_difference_velocity(t: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward difference velocity at segment mid-times."""
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("Times must be strictly increasing for velocity")
    velocity = np.diff(values, axis=0) / dt[:, np.newaxis]
    t_mid = 0.5 * (t[1:] + t[:-1])
    return t_mid, velocity


def _clip_to_window(t: np.ndarray, values: np.ndarray, t_lo: float, t_hi: float) -> TimedSeries:
    """Keep samples with ``t in [t_lo, t_hi]`` (inclusive)."""
    mask = (t >= t_lo) & (t <= t_hi)
    return t[mask], values[mask]


def _clip_velocity_to_window(t_mid: np.ndarray, velocity: np.ndarray, t_lo: float, t_hi: float) -> TimedSeries:
    """Keep velocity samples whose mid-time lies in ``[t_lo, t_hi]``."""
    mask = (t_mid >= t_lo) & (t_mid <= t_hi)
    return t_mid[mask], velocity[mask]


def _subsample_for_plot(t: np.ndarray, values: np.ndarray, max_points: int) -> TimedSeries:
    """Uniformly subsample when the series is too long for matplotlib."""
    n = int(t.size)
    if n <= max_points:
        return t, values
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return t[idx], values[idx]


def _load_stream(record: Dict[str, Any], stream: str) -> Optional[TimedSeries]:
    """Load a named velocity source as ``(t, values)``."""
    sparse = record["sparse_waypoint_plan"]
    samples = record["recorded_samples"]
    if stream == STREAM_DENSE_JOINT:
        return _try_series(record.get("dense_command_traj") or [], 7, "dense_command_traj")
    if stream == STREAM_SPARSE_JOINT:
        return _try_series(sparse.get("joints") or [], 7, "sparse_waypoint_plan.joints")
    if stream == STREAM_SPARSE_TCP:
        return _try_series(sparse.get("tcp_pose") or [], 7, "sparse_waypoint_plan.tcp_pose")
    if stream == STREAM_SAMPLE_JOINT:
        return _try_series(samples.get("joints") or [], 7, "recorded_samples.joints")
    if stream == STREAM_SAMPLE_TCP:
        return _try_series(samples.get("tcp_pose") or [], 7, "recorded_samples.tcp_pose")
    raise ValueError(f"Unknown stream: {stream!r}")


def _default_speed_threshold(stream: str) -> float:
    """Near-stop threshold for ``window_min_speed`` (joint: deg/s, TCP: m/s)."""
    if stream in (STREAM_DENSE_JOINT, STREAM_SPARSE_JOINT, STREAM_SAMPLE_JOINT):
        return 1.0
    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        return 0.01
    raise ValueError(f"Unknown stream: {stream!r}")


def _compute_velocity_in_window(
    t: np.ndarray,
    values: np.ndarray,
    stream: str,
    t_lo: float,
    t_hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full-resolution velocity in the analysis window (no subsampling).

    Returns ``(t_mid, values_for_stats)`` where joint rows are deg/s and TCP rows are |v_xyz| m/s.
    """
    t_w, v_w = _clip_to_window(t, values, t_lo, t_hi)
    if t_w.size < 2:
        raise ValueError(f"{stream}: not enough samples in time window [{t_lo}, {t_hi}]")
    t_mid, vel = _finite_difference_velocity(t_w, v_w)
    t_mid, vel = _clip_velocity_to_window(t_mid, vel, t_lo, t_hi)
    if t_mid.size < 2:
        raise ValueError(f"{stream}: not enough velocity samples in time window")

    if stream in (STREAM_DENSE_JOINT, STREAM_SPARSE_JOINT, STREAM_SAMPLE_JOINT):
        return t_mid, np.rad2deg(vel)

    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        speed = np.linalg.norm(vel[:, :3], axis=1, keepdims=True)
        return t_mid, speed

    raise ValueError(f"Unsupported stream for velocity: {stream!r}")


def _speed_norms(values_for_stats: np.ndarray, stream: str) -> np.ndarray:
    """Scalar speed per sample for global and window statistics."""
    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        return values_for_stats.reshape(-1)
    return np.linalg.norm(values_for_stats, axis=1)


def _velocity_plot_series(
    t_mid: np.ndarray,
    values_for_stats: np.ndarray,
    stream: str,
    t_lo: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """Subsample and label velocity for matplotlib."""
    t_plot, v_plot = _subsample_for_plot(t_mid, values_for_stats, max_points)
    t_rel = t_plot - float(t_lo)

    if stream in (STREAM_DENSE_JOINT, STREAM_SPARSE_JOINT, STREAM_SAMPLE_JOINT):
        labels = [f"dq{i} [deg/s]" for i in range(int(v_plot.shape[1]))]
        title = f"{stream} (joint, numerical derivative)"
        return t_rel, v_plot, title, labels

    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        labels = ["|v_xyz| [m/s]"]
        title = f"{stream} (TCP xyz speed)"
        return t_rel, v_plot, title, labels

    raise ValueError(f"Unsupported stream for velocity: {stream!r}")


def _window_min_speed_stats(
    t_mid: np.ndarray,
    speed: np.ndarray,
    t_lo: float,
    t_hi: float,
    window_period_s: float,
    threshold: float,
) -> Dict[str, Any]:
    """
    Per fixed-length window, take min(speed); summarize stop-go tendency.

    ``frac_windows_min_below_threshold`` is the main A/B metric: fraction of windows
  whose minimum speed falls below ``threshold`` (Ruckig rest-to-rest shows up here).
    """
    if window_period_s <= 0.0:
        return {}
    window_mins: List[float] = []
    t_start = float(t_lo)
    while t_start + window_period_s <= float(t_hi) + 1e-12:
        t_end = t_start + window_period_s
        mask = (t_mid >= t_start) & (t_mid < t_end)
        if np.any(mask):
            window_mins.append(float(np.min(speed[mask])))
        t_start = t_end
    if not window_mins:
        return {
            "window_period_s": window_period_s,
            "speed_threshold": threshold,
            "n_windows": 0,
            "window_min_speed_median": None,
            "window_min_speed_p95": None,
            "frac_windows_min_below_threshold": None,
        }
    mins_arr = np.asarray(window_mins, dtype=np.float64)
    return {
        "window_period_s": window_period_s,
        "speed_threshold": threshold,
        "n_windows": int(mins_arr.size),
        "window_min_speed_median": float(np.median(mins_arr)),
        "window_min_speed_p95": float(np.percentile(mins_arr, 95)),
        "frac_windows_min_below_threshold": float(np.mean(mins_arr < threshold)),
    }


def _series_time_span(series: Optional[TimedSeries]) -> Optional[Tuple[float, float]]:
    """Return ``[t_first, t_last]`` when the series has at least two samples."""
    if series is None:
        return None
    t_arr, _ = series
    if int(t_arr.size) < 2:
        return None
    return float(t_arr[0]), float(t_arr[-1])


def _intervals_intersection(intervals: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """``[max(starts), min(ends)]`` or ``None`` when empty or degenerate."""
    if not intervals:
        return None
    t_lo = max(lo for lo, _ in intervals)
    t_hi = min(hi for _, hi in intervals)
    if t_hi <= t_lo + 1e-9:
        return None
    return t_lo, t_hi


def _record_three_trajectory_time_intersection(record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Common time window for dense / sparse / recorded_samples overlays.

    Any sparse or sample sub-series (joints, tcp_pose) with at least two points
    contributes its span; the result is the intersection of all contributing spans.
    """
    intervals: List[Tuple[float, float]] = []

    dense = _try_series(record.get("dense_command_traj") or [], 7, "dense_command_traj")
    span = _series_time_span(dense)
    if span is not None:
        intervals.append(span)

    sparse = record.get("sparse_waypoint_plan")
    if isinstance(sparse, dict):
        for key in ("joints", "tcp_pose"):
            series = _try_series(sparse.get(key) or [], 7, f"sparse_waypoint_plan.{key}")
            span = _series_time_span(series)
            if span is not None:
                intervals.append(span)

    samples = record.get("recorded_samples")
    if isinstance(samples, dict):
        for key in ("joints", "tcp_pose"):
            series = _try_series(samples.get(key) or [], 7, f"recorded_samples.{key}")
            span = _series_time_span(series)
            if span is not None:
                intervals.append(span)

    return _intervals_intersection(intervals)


def _default_time_window(record: Dict[str, Any], streams: Sequence[str]) -> Tuple[float, float]:
    """Intersection of time spans for requested streams that exist."""
    intervals: List[Tuple[float, float]] = []
    for stream in streams:
        series = _load_stream(record, stream)
        if series is None:
            logger.warning("Stream {} not available in record; skipping", stream)
            continue
        span = _series_time_span(series)
        if span is not None:
            intervals.append(span)
    intersection = _intervals_intersection(intervals)
    if intersection is not None:
        return intersection
    if not intervals:
        raise ValueError("No streams available for time window")
    raise ValueError(
        "Plotted streams have no overlapping time span; pass --t-start/--t-end explicitly"
    )


def _resolve_plot_time_window(
    record: Dict[str, Any],
    streams: Sequence[str],
    t_lo: Optional[float],
    t_hi: Optional[float],
    stream_intersection_only: bool,
) -> Tuple[float, float, str, Dict[str, Optional[Tuple[float, float]]]]:
    """
    Choose ``[t_lo, t_hi]`` for velocity plots and stats.

    Returns ``(t_lo, t_hi, mode_label, per_source_spans)``.
    """
    per_source_spans: Dict[str, Optional[Tuple[float, float]]] = {
        "dense_command_traj": _series_time_span(
            _try_series(record.get("dense_command_traj") or [], 7, "dense_command_traj")
        ),
        "sparse_waypoint_plan.joints": _series_time_span(
            _try_series(
                (record.get("sparse_waypoint_plan") or {}).get("joints") or [],
                7,
                "sparse_waypoint_plan.joints",
            )
        ),
        "sparse_waypoint_plan.tcp_pose": _series_time_span(
            _try_series(
                (record.get("sparse_waypoint_plan") or {}).get("tcp_pose") or [],
                7,
                "sparse_waypoint_plan.tcp_pose",
            )
        ),
        "recorded_samples.joints": _series_time_span(
            _try_series(
                (record.get("recorded_samples") or {}).get("joints") or [],
                7,
                "recorded_samples.joints",
            )
        ),
        "recorded_samples.tcp_pose": _series_time_span(
            _try_series(
                (record.get("recorded_samples") or {}).get("tcp_pose") or [],
                7,
                "recorded_samples.tcp_pose",
            )
        ),
    }

    auto_lo: Optional[float] = None
    auto_hi: Optional[float] = None
    mode = "three_trajectory_intersection"
    if stream_intersection_only:
        auto_lo, auto_hi = _default_time_window(record, streams)
        mode = "stream_intersection"
    else:
        overlay = _record_three_trajectory_time_intersection(record)
        if overlay is not None:
            auto_lo, auto_hi = overlay
        else:
            logger.warning(
                "No overlap among dense/sparse/recorded_samples; "
                "falling back to intersection of requested --stream series"
            )
            auto_lo, auto_hi = _default_time_window(record, streams)
            mode = "stream_intersection_fallback"

    assert auto_lo is not None and auto_hi is not None
    resolved_lo = auto_lo if t_lo is None else max(auto_lo, float(t_lo))
    resolved_hi = auto_hi if t_hi is None else min(auto_hi, float(t_hi))
    if resolved_hi <= resolved_lo + 1e-9:
        raise ValueError(
            f"Invalid time window [{resolved_lo}, {resolved_hi}] "
            f"(auto overlay [{auto_lo}, {auto_hi}], mode={mode}); check --t-start/--t-end"
        )
    if t_lo is not None or t_hi is not None:
        mode = f"{mode}+cli_narrow"
    logger.info(
        "Velocity analysis window [{:.3f}, {:.3f}] s (mode={}, duration={:.3f} s)",
        resolved_lo,
        resolved_hi,
        mode,
        resolved_hi - resolved_lo,
    )
    return resolved_lo, resolved_hi, mode, per_source_spans


def _velocity_stats(
    t_mid: np.ndarray,
    values_for_stats: np.ndarray,
    stream: str,
    t_lo: float,
    t_hi: float,
    window_period_s: Optional[float],
    window_speed_threshold: Optional[float],
) -> Dict[str, Any]:
    """Summary statistics for one velocity stream (full resolution, not plot subsample)."""
    norms = _speed_norms(values_for_stats, stream)
    unit = "m/s" if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP) else "deg/s"
    point_threshold = 0.01 if unit == "m/s" else 1.0
    dt = np.diff(t_mid)
    out: Dict[str, Any] = {
        "stream": stream,
        "n": int(t_mid.size),
        "duration_s": float(t_mid[-1] - t_mid[0]) if t_mid.size >= 2 else 0.0,
        "dt_median_s": float(np.median(dt)) if dt.size else 0.0,
        "speed_unit": unit,
        "speed_median": float(np.median(norms)),
        "speed_p95": float(np.percentile(norms, 95)),
        "speed_max": float(np.max(norms)),
        "near_zero_frac": float(np.mean(norms < point_threshold)),
    }
    if window_period_s is not None and window_period_s > 0.0:
        thresh = (
            window_speed_threshold
            if window_speed_threshold is not None
            else _default_speed_threshold(stream)
        )
        out["window_min_speed"] = _window_min_speed_stats(
            t_mid, norms, t_lo, t_hi, window_period_s, float(thresh)
        )
    return out


def plot_velocity_profiles(
    record: Dict[str, Any],
    streams: Sequence[str],
    output_png: Path,
    t_lo: Optional[float],
    t_hi: Optional[float],
    max_points: int,
    segment_period_s: Optional[float],
    window_period_s: Optional[float],
    window_speed_threshold: Optional[float],
    stream_intersection_only: bool = False,
) -> Dict[str, Any]:
    """Plot per-stream velocity panels and return JSON-serializable summary."""
    available: List[str] = []
    panels: List[Tuple[str, np.ndarray, np.ndarray, str, List[str]]] = []
    for stream in streams:
        series = _load_stream(record, stream)
        if series is None:
            continue
        available.append(stream)

    if not available:
        raise ValueError("None of the requested streams are present in the record")

    t_lo, t_hi, time_window_mode, per_source_spans = _resolve_plot_time_window(
        record=record,
        streams=available,
        t_lo=t_lo,
        t_hi=t_hi,
        stream_intersection_only=stream_intersection_only,
    )

    grid_period = segment_period_s if segment_period_s is not None else window_period_s

    stats: List[Dict[str, Any]] = []
    for stream in available:
        t_arr, values = _load_stream(record, stream)
        assert t_arr is not None and values is not None
        t_mid, values_for_stats = _compute_velocity_in_window(t_arr, values, stream, t_lo, t_hi)
        t_rel, vel_plot, title, labels = _velocity_plot_series(
            t_mid, values_for_stats, stream, t_lo, max_points
        )
        panels.append((stream, t_rel, vel_plot, title, labels))
        stats.append(
            _velocity_stats(
                t_mid,
                values_for_stats,
                stream,
                t_lo,
                t_hi,
                window_period_s,
                window_speed_threshold,
            )
        )

    n_panels = sum(int(p[2].shape[1]) for p in panels)
    fig_height = max(4.0, 2.0 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, fig_height), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    span = float(t_hi) - float(t_lo)
    for stream, t_rel, vel_plot, title, labels in panels:
        color = _PLOT_COLORS.get(stream, "#333333")
        for dim in range(int(vel_plot.shape[1])):
            ax = axes[ax_idx]
            ax.plot(t_rel, vel_plot[:, dim], color=color, linewidth=0.85, alpha=0.95, label=title)
            ax.set_ylabel(labels[dim])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            if grid_period is not None and grid_period > 0.0 and span > grid_period:
                n_mark = int(span / grid_period) + 1
                for k in range(n_mark):
                    x = k * grid_period
                    if x <= span:
                        ax.axvline(x, color="gray", linewidth=0.4, alpha=0.35)
            ax_idx += 1

    if span > 1e-12:
        for ax in axes:
            ax.set_xlim(0.0, span)
    axes[-1].set_xlabel("t - t_window_start [s]")
    fig.suptitle(
        f"Velocity profiles (window [{t_lo:.3f}, {t_hi:.3f}] s, traj_space={record['traj_space']})",
        fontsize=13,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    return {
        "time_window": {"t_lo": t_lo, "t_hi": t_hi},
        "time_window_mode": time_window_mode,
        "trajectory_source_spans": {
            key: None if span is None else {"t_lo": span[0], "t_hi": span[1]}
            for key, span in per_source_spans.items()
        },
        "streams_plotted": [p[0] for p in panels],
        "stats": stats,
        "figure": str(output_png),
        "segment_period_s": segment_period_s,
        "window_period_s": window_period_s,
        "window_speed_threshold": window_speed_threshold,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    default_record = Path("logs/traj/trajectory_run_record.json")
    default_png = Path("logs/traj/waypoint_velocity_analysis.png")
    default_json = Path("logs/traj/waypoint_velocity_analysis.json")
    parser = argparse.ArgumentParser(
        description="Plot velocity profiles from trajectory run record JSON",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=default_record,
        help=f"Path to traj_run_record JSON (default: {default_record})",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=default_png,
        help=f"Output figure path (default: {default_png})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_json,
        help=f"Output summary JSON path (default: {default_json})",
    )
    parser.add_argument(
        "--stream",
        action="append",
        dest="streams",
        choices=list(STREAM_CHOICES),
        help=(
            "Velocity source to plot; repeatable. Default: dense_joint only. "
            f"Choices: {', '.join(STREAM_CHOICES)}"
        ),
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=None,
        help=(
            "Window start [s] in record perf_counter time "
            "(default: intersection of dense + sparse + recorded_samples spans)"
        ),
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help=(
            "Window end [s] in record perf_counter time "
            "(default: same three-trajectory intersection as --t-start)"
        ),
    )
    parser.add_argument(
        "--stream-intersection-only",
        action="store_true",
        help=(
            "Use intersection of requested --stream series only "
            "(skip dense/sparse/sample global overlay window)"
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Max points per stream after subsampling for plotting (default: 20000)",
    )
    parser.add_argument(
        "--segment-period",
        type=float,
        default=None,
        help="Optional vertical grid period [s] on plots (e.g. 0.03 for planner waypoint spacing)",
    )
    parser.add_argument(
        "--window-period",
        type=float,
        default=0.03,
        help=(
            "Fixed window length [s] for window_min_speed stats "
            "(default: 0.03; set 0 to disable)"
        ),
    )
    parser.add_argument(
        "--window-speed-threshold",
        type=float,
        default=None,
        help=(
            "Speed threshold for frac_windows_min_below_threshold "
            "(default: 1.0 deg/s joint, 0.01 m/s TCP)"
        ),
    )
    parser.add_argument(
        "--no-window-analysis",
        action="store_true",
        help="Disable window_min_speed block in output JSON",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    streams: List[str] = args.streams if args.streams else [STREAM_DENSE_JOINT]
    record = load_traj_run_record(args.record)
    window_period: Optional[float] = None if args.no_window_analysis else float(args.window_period)
    if window_period is not None and window_period <= 0.0:
        window_period = None

    summary = plot_velocity_profiles(
        record=record,
        streams=streams,
        output_png=args.output_png,
        t_lo=args.t_start,
        t_hi=args.t_end,
        max_points=int(args.max_points),
        segment_period_s=args.segment_period,
        window_period_s=window_period,
        window_speed_threshold=args.window_speed_threshold,
        stream_intersection_only=bool(args.stream_intersection_only),
    )
    payload = {"record_path": str(args.record), **summary}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("Saved figure to {}", args.output_png)
    logger.info("Saved summary to {}", args.output_json)


if __name__ == "__main__":
    main()
#     python scripts/benchmark/waypoint_velocity_analyzer.py \
#   --record /home/zixiying/Downloads/no_planner/no_planner.json \
#   --stream sparse_tcp --stream sample_tcp \
#   --segment-period 0.03 \
#   --output-json logs/traj/waypoint_velocity_analysis_no.json

#     python scripts/benchmark/waypoint_velocity_analyzer.py \
#   --stream dense_joint --stream sample_tcp \
#   --segment-period 0.03
