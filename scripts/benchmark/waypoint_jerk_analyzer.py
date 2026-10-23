#!/usr/bin/env python3
"""
Plot jerk profiles from trajectory run record JSON files.

Reads the same schema as ``scripts/benchmark/waypoint_trajectory_analyzer.py`` (traj_run_record).
Computes numerical third derivatives for joint commands (``dense_command_traj``,
``sparse_waypoint_plan.joints``, ``recorded_samples.joints``) and optional TCP
linear jerk from ``sparse_waypoint_plan.tcp_pose`` / ``recorded_samples.tcp_pose``.

Does not require Pinocchio/FK; suitable for large planner dense traces.

By default the plot/stats time window is the **intersection** of
``dense_command_traj``, ``sparse_waypoint_plan`` (joints and/or tcp_pose when
present), and ``recorded_samples`` (joints and/or tcp_pose), matching
``scripts/benchmark/waypoint_trajectory_analyzer.py`` overlay alignment.

Exports ``window_max_jerk`` stats: per ``--window-period`` (default 0.03 s), the
fraction of windows whose maximum jerk exceeds threshold
(``frac_windows_max_above_threshold``) — useful for spotting harsh acceleration changes.

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


def _finite_difference_derivative(t: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward difference derivative at segment mid-times."""
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("Times must be strictly increasing for differentiation")
    derivative = np.diff(values, axis=0) / dt[:, np.newaxis]
    t_mid = 0.5 * (t[1:] + t[:-1])
    return t_mid, derivative


def _clip_to_window(t: np.ndarray, values: np.ndarray, t_lo: float, t_hi: float) -> TimedSeries:
    """Keep samples with ``t in [t_lo, t_hi]`` (inclusive)."""
    mask = (t >= t_lo) & (t <= t_hi)
    return t[mask], values[mask]


def _clip_mid_to_window(t_mid: np.ndarray, values: np.ndarray, t_lo: float, t_hi: float) -> TimedSeries:
    """Keep derivative samples whose mid-time lies in ``[t_lo, t_hi]``."""
    mask = (t_mid >= t_lo) & (t_mid <= t_hi)
    return t_mid[mask], values[mask]


def _subsample_for_plot(t: np.ndarray, values: np.ndarray, max_points: int) -> TimedSeries:
    """Uniformly subsample when the series is too long for matplotlib."""
    n = int(t.size)
    if n <= max_points:
        return t, values
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return t[idx], values[idx]


def _load_stream(record: Dict[str, Any], stream: str) -> Optional[TimedSeries]:
    """Load a named trajectory source as ``(t, values)``."""
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


def _default_jerk_threshold(stream: str) -> float:
    """High-jerk threshold for ``window_max_jerk`` (joint: deg/s^3, TCP: m/s^3)."""
    if stream in (STREAM_DENSE_JOINT, STREAM_SPARSE_JOINT, STREAM_SAMPLE_JOINT):
        return 5000.0
    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        return 5.0
    raise ValueError(f"Unknown stream: {stream!r}")


def _compute_jerk_in_window(
    t: np.ndarray,
    values: np.ndarray,
    stream: str,
    t_lo: float,
    t_hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full-resolution jerk in the analysis window (no subsampling).

    Joint: numerical d^3q/dt^3 in deg/s^3. TCP: |j_xyz| in m/s^3 from xyz position.
    Requires at least four position samples in the window.
    """
    t_w, v_w = _clip_to_window(t, values, t_lo, t_hi)
    if t_w.size < 4:
        raise ValueError(
            f"{stream}: need at least 4 position samples in time window [{t_lo}, {t_hi}] for jerk"
        )

    t_vel, vel_rad = _finite_difference_derivative(t_w, v_w)
    t_vel, vel_rad = _clip_mid_to_window(t_vel, vel_rad, t_lo, t_hi)
    if t_vel.size < 3:
        raise ValueError(f"{stream}: not enough velocity samples for acceleration")

    if stream in (STREAM_DENSE_JOINT, STREAM_SPARSE_JOINT, STREAM_SAMPLE_JOINT):
        vel = np.rad2deg(vel_rad)
        t_acc, acc = _finite_difference_derivative(t_vel, vel)
        t_acc, acc = _clip_mid_to_window(t_acc, acc, t_lo, t_hi)
        if t_acc.size < 2:
            raise ValueError(f"{stream}: not enough acceleration samples for jerk")
        t_jerk, jerk = _finite_difference_derivative(t_acc, acc)
        t_jerk, jerk = _clip_mid_to_window(t_jerk, jerk, t_lo, t_hi)
        if t_jerk.size < 2:
            raise ValueError(f"{stream}: not enough jerk samples in time window")
        return t_jerk, jerk

    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        t_acc, acc_xyz = _finite_difference_derivative(t_vel, vel_rad[:, :3])
        t_acc, acc_xyz = _clip_mid_to_window(t_acc, acc_xyz, t_lo, t_hi)
        if t_acc.size < 2:
            raise ValueError(f"{stream}: not enough acceleration samples for jerk")
        t_jerk, jerk_xyz = _finite_difference_derivative(t_acc, acc_xyz)
        t_jerk, jerk_xyz = _clip_mid_to_window(t_jerk, jerk_xyz, t_lo, t_hi)
        if t_jerk.size < 2:
            raise ValueError(f"{stream}: not enough jerk samples in time window")
        jerk_mag = np.linalg.norm(jerk_xyz, axis=1, keepdims=True)
        return t_jerk, jerk_mag

    raise ValueError(f"Unsupported stream for jerk: {stream!r}")


def _jerk_norms(values_for_stats: np.ndarray, stream: str) -> np.ndarray:
    """Scalar jerk magnitude per sample for global and window statistics."""
    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        return values_for_stats.reshape(-1)
    return np.linalg.norm(values_for_stats, axis=1)


def _jerk_plot_series(
    t_jerk: np.ndarray,
    values_for_stats: np.ndarray,
    stream: str,
    t_lo: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """Subsample and label jerk for matplotlib."""
    t_plot, j_plot = _subsample_for_plot(t_jerk, values_for_stats, max_points)
    t_rel = t_plot - float(t_lo)

    if stream in (STREAM_DENSE_JOINT, STREAM_SPARSE_JOINT, STREAM_SAMPLE_JOINT):
        labels = [f"dddq{i} [deg/s^3]" for i in range(int(j_plot.shape[1]))]
        title = f"{stream} (joint, 3rd numerical derivative)"
        return t_rel, j_plot, title, labels

    if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP):
        labels = ["|j_xyz| [m/s^3]"]
        title = f"{stream} (TCP xyz jerk magnitude)"
        return t_rel, j_plot, title, labels

    raise ValueError(f"Unsupported stream for jerk: {stream!r}")


def _window_max_jerk_stats(
    t_jerk: np.ndarray,
    jerk: np.ndarray,
    t_lo: float,
    t_hi: float,
    window_period_s: float,
    threshold: float,
) -> Dict[str, Any]:
    """
    Per fixed-length window, take max(jerk); summarize harsh motion segments.

    ``frac_windows_max_above_threshold`` is the main A/B metric: fraction of windows
    whose maximum jerk exceeds ``threshold``.
    """
    if window_period_s <= 0.0:
        return {}
    window_maxs: List[float] = []
    t_start = float(t_lo)
    while t_start + window_period_s <= float(t_hi) + 1e-12:
        t_end = t_start + window_period_s
        mask = (t_jerk >= t_start) & (t_jerk < t_end)
        if np.any(mask):
            window_maxs.append(float(np.max(jerk[mask])))
        t_start = t_end
    if not window_maxs:
        return {
            "window_period_s": window_period_s,
            "jerk_threshold": threshold,
            "n_windows": 0,
            "window_max_jerk_median": None,
            "window_max_jerk_p95": None,
            "frac_windows_max_above_threshold": None,
        }
    maxs_arr = np.asarray(window_maxs, dtype=np.float64)
    return {
        "window_period_s": window_period_s,
        "jerk_threshold": threshold,
        "n_windows": int(maxs_arr.size),
        "window_max_jerk_median": float(np.median(maxs_arr)),
        "window_max_jerk_p95": float(np.percentile(maxs_arr, 95)),
        "frac_windows_max_above_threshold": float(np.mean(maxs_arr > threshold)),
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
    Choose ``[t_lo, t_hi]`` for jerk plots and stats.

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
        "Jerk analysis window [{:.3f}, {:.3f}] s (mode={}, duration={:.3f} s)",
        resolved_lo,
        resolved_hi,
        mode,
        resolved_hi - resolved_lo,
    )
    return resolved_lo, resolved_hi, mode, per_source_spans


def _jerk_stats(
    t_jerk: np.ndarray,
    values_for_stats: np.ndarray,
    stream: str,
    t_lo: float,
    t_hi: float,
    window_period_s: Optional[float],
    window_jerk_threshold: Optional[float],
) -> Dict[str, Any]:
    """Summary statistics for one jerk stream (full resolution, not plot subsample)."""
    norms = _jerk_norms(values_for_stats, stream)
    unit = "m/s^3" if stream in (STREAM_SPARSE_TCP, STREAM_SAMPLE_TCP) else "deg/s^3"
    point_threshold = 0.1 if unit == "m/s^3" else 50.0
    dt = np.diff(t_jerk)
    out: Dict[str, Any] = {
        "stream": stream,
        "n": int(t_jerk.size),
        "duration_s": float(t_jerk[-1] - t_jerk[0]) if t_jerk.size >= 2 else 0.0,
        "dt_median_s": float(np.median(dt)) if dt.size else 0.0,
        "jerk_unit": unit,
        "jerk_median": float(np.median(norms)),
        "jerk_p95": float(np.percentile(norms, 95)),
        "jerk_max": float(np.max(norms)),
        "near_zero_frac": float(np.mean(norms < point_threshold)),
    }
    if window_period_s is not None and window_period_s > 0.0:
        thresh = (
            window_jerk_threshold
            if window_jerk_threshold is not None
            else _default_jerk_threshold(stream)
        )
        out["window_max_jerk"] = _window_max_jerk_stats(
            t_jerk, norms, t_lo, t_hi, window_period_s, float(thresh)
        )
    return out


def plot_jerk_profiles(
    record: Dict[str, Any],
    streams: Sequence[str],
    output_png: Path,
    t_lo: Optional[float],
    t_hi: Optional[float],
    max_points: int,
    segment_period_s: Optional[float],
    window_period_s: Optional[float],
    window_jerk_threshold: Optional[float],
    stream_intersection_only: bool = False,
) -> Dict[str, Any]:
    """Plot per-stream jerk panels and return JSON-serializable summary."""
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
        t_jerk, values_for_stats = _compute_jerk_in_window(t_arr, values, stream, t_lo, t_hi)
        t_rel, jerk_plot, title, labels = _jerk_plot_series(
            t_jerk, values_for_stats, stream, t_lo, max_points
        )
        panels.append((stream, t_rel, jerk_plot, title, labels))
        stats.append(
            _jerk_stats(
                t_jerk,
                values_for_stats,
                stream,
                t_lo,
                t_hi,
                window_period_s,
                window_jerk_threshold,
            )
        )

    n_panels = sum(int(p[2].shape[1]) for p in panels)
    fig_height = max(4.0, 2.0 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, fig_height), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    span = float(t_hi) - float(t_lo)
    for stream, t_rel, jerk_plot, title, labels in panels:
        color = _PLOT_COLORS.get(stream, "#333333")
        for dim in range(int(jerk_plot.shape[1])):
            ax = axes[ax_idx]
            ax.plot(t_rel, jerk_plot[:, dim], color=color, linewidth=0.85, alpha=0.95, label=title)
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
        f"Jerk profiles (window [{t_lo:.3f}, {t_hi:.3f}] s, traj_space={record['traj_space']})",
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
        "window_jerk_threshold": window_jerk_threshold,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    default_record = Path("logs/traj/trajectory_run_record.json")
    default_png = Path("logs/traj/waypoint_jerk_analysis.png")
    default_json = Path("logs/traj/waypoint_jerk_analysis.json")
    parser = argparse.ArgumentParser(
        description="Plot jerk profiles from trajectory run record JSON",
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
            "Trajectory source to plot; repeatable. Default: dense_joint only. "
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
            "Fixed window length [s] for window_max_jerk stats "
            "(default: 0.03; set 0 to disable)"
        ),
    )
    parser.add_argument(
        "--window-jerk-threshold",
        type=float,
        default=None,
        help=(
            "Jerk threshold for frac_windows_max_above_threshold "
            "(default: 5000 deg/s^3 joint, 5.0 m/s^3 TCP)"
        ),
    )
    parser.add_argument(
        "--no-window-analysis",
        action="store_true",
        help="Disable window_max_jerk block in output JSON",
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

    summary = plot_jerk_profiles(
        record=record,
        streams=streams,
        output_png=args.output_png,
        t_lo=args.t_start,
        t_hi=args.t_end,
        max_points=int(args.max_points),
        segment_period_s=args.segment_period,
        window_period_s=window_period,
        window_jerk_threshold=args.window_jerk_threshold,
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
