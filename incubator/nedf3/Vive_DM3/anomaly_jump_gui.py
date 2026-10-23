#!/usr/bin/env python3
"""
Tkinter frame for browsing exported anomaly jump segments.

Author: Yichen Xie
"""

from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasTkAgg = None
    Figure = None

from utils.logger_config import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENT_ROOT = PROJECT_ROOT / "logs" / "anomaly_jump_segments"
ARM_TO_INDEX = {"left": 0, "right": 1, "ego": 2}


class AnomalyJumpFrame:
    """Browse exported anomaly jump clips inside the DM3 GUI."""

    def __init__(
        self,
        parent: ttk.Frame,
        *,
        default_root: Path = DEFAULT_SEGMENT_ROOT,
        log: Callable[..., None],
        show_error: Callable[..., None],
    ) -> None:
        self.parent = parent
        self.log = log
        self.show_error = show_error
        self.root_var = tk.StringVar(value=str(default_root))
        self.status_var = tk.StringVar(value="跳变片段：等待刷新")
        self.segment_tree: Optional[ttk.Treeview] = None
        self.segments: List[Dict[str, Any]] = []
        self.figure: Optional[Any] = None
        self.pose_axis: Optional[Any] = None
        self.canvas: Optional[Any] = None
        self.image_strip: Optional[ttk.Frame] = None
        self.photo_refs: List[tk.PhotoImage] = []
        self.play_button: Optional[ttk.Button] = None
        self.playback_job: Optional[str] = None
        self.playback_started_at = 0.0
        self.playback_start_sec = 0.0
        self.playback_end_sec = 0.0
        self.playback_segment_dir: Optional[Path] = None
        self.playback_metadata: Dict[str, Any] = {}
        self.playback_timestamps = np.array([], dtype=np.float64)
        self.playback_xyz = np.empty((0, 3), dtype=np.float64)
        self.camera_sequences: Dict[str, List[Tuple[float, Path]]] = {}
        self.camera_labels: Dict[str, tk.Label] = {}
        self.camera_caption_vars: Dict[str, tk.StringVar] = {}

        self._build(parent)
        self.refresh_segments()

    def _build(self, parent: ttk.Frame) -> None:
        top = ttk.LabelFrame(parent, text="异常跳变片段", padding=8)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="片段目录：").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.root_var, width=58).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(4, 8),
        )
        ttk.Button(top, text="选择目录", command=self._choose_root).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="刷新", command=self.refresh_segments).pack(side=tk.LEFT, padx=4)
        self.play_button = ttk.Button(
            top,
            text="重新播放",
            command=self._start_playback,
            state=tk.DISABLED,
        )
        self.play_button.pack(side=tk.LEFT, padx=4)
        ttk.Label(top, textvariable=self.status_var, foreground="#374151").pack(
            side=tk.LEFT,
            padx=(12, 0),
        )

        content = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.LabelFrame(content, text="片段列表", padding=8)
        columns = ("episode", "type", "arm", "time", "value")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=18)
        for col, title, width in (
            ("episode", "Episode", 220),
            ("type", "Type", 145),
            ("arm", "Arm", 65),
            ("time", "Time(s)", 80),
            ("value", "Value", 90),
        ):
            tree.heading(col, text=title)
            tree.column(col, width=width, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        tree.bind("<<TreeviewSelect>>", lambda _event: self._render_selected_segment())
        self.segment_tree = tree
        content.add(list_frame, weight=1)

        detail_frame = ttk.Frame(content)
        plot_frame = ttk.LabelFrame(detail_frame, text="跳变 3D 轨迹", padding=8)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self._build_plot(plot_frame)

        image_frame = ttk.LabelFrame(detail_frame, text="相近相机帧", padding=8)
        image_frame.pack(fill=tk.X)
        self.image_strip = ttk.Frame(image_frame)
        self.image_strip.pack(fill=tk.X)
        content.add(detail_frame, weight=3)

    def _build_plot(self, parent: ttk.Frame) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(
                parent,
                text="matplotlib unavailable",
                foreground="#9ca3af",
                anchor=tk.CENTER,
            ).pack(fill=tk.BOTH, expand=True)
            return
        figure = Figure(figsize=(8.0, 4.8), dpi=100)
        figure.patch.set_facecolor("#111827")
        self.pose_axis = figure.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(figure, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(bg="#111827", highlightthickness=1, highlightbackground="#374151")
        widget.pack(fill=tk.BOTH, expand=True)
        self.figure = figure
        self.canvas = canvas
        self._draw_empty_plot("select a segment")

    def _choose_root(self) -> None:
        selected = filedialog.askdirectory(
            title="选择异常跳变片段目录",
            initialdir=self.root_var.get() or str(DEFAULT_SEGMENT_ROOT),
        )
        if selected:
            self.root_var.set(selected)
            self.refresh_segments()

    def refresh_segments(self) -> None:
        self._cancel_playback()
        root = Path(self.root_var.get()).expanduser()
        self.segments = self._scan_segment_metadata(root)
        tree = self.segment_tree
        if tree is not None:
            tree.delete(*tree.get_children())
            for index, item in enumerate(self.segments):
                meta = item["metadata"]
                tree.insert(
                    "",
                    tk.END,
                    iid=str(index),
                    values=(
                        meta.get("episode_name", "-"),
                        meta.get("anomaly_type", "-"),
                        meta.get("arm", "-"),
                        f"{float(meta.get('time_sec', 0.0)):.3f}",
                        f"{float(meta.get('value', 0.0)):.4f}",
                    ),
                )
        self.status_var.set(f"跳变片段：找到 {len(self.segments)} 段")
        self._draw_empty_plot("select a segment")
        self._clear_images()
        if self.play_button is not None:
            self.play_button.configure(state=tk.DISABLED)

    def _scan_segment_metadata(self, root: Path) -> List[Dict[str, Any]]:
        if not root.exists():
            return []
        items: List[Dict[str, Any]] = []
        for metadata_path in root.rglob("segment.json"):
            if "quarantine" in metadata_path.parts:
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to read anomaly segment metadata {metadata_path}: {exc}")
                continue
            items.append({"path": metadata_path.parent, "metadata": metadata})
        items.sort(key=lambda item: item["path"].stat().st_mtime, reverse=True)
        return items

    def _render_selected_segment(self) -> None:
        tree = self.segment_tree
        if tree is None:
            return
        selection = tree.selection()
        if not selection:
            return
        try:
            segment = self.segments[int(selection[0])]
        except (IndexError, ValueError):
            return
        try:
            self._load_playback_segment(Path(segment["path"]), segment["metadata"])
            self._start_playback()
            self.status_var.set(f"跳变片段：{segment['path'].name}")
        except Exception as exc:
            self.show_error(exc, title="跳变片段可视化失败")

    def _load_playback_segment(self, segment_dir: Path, metadata: Dict[str, Any]) -> None:
        self._cancel_playback()
        self.playback_segment_dir = segment_dir
        self.playback_metadata = dict(metadata)
        self.playback_start_sec = float(metadata.get("window_start_sec", 0.0))
        self.playback_end_sec = float(metadata.get("window_end_sec", self.playback_start_sec))
        self._load_pose_playback(segment_dir, metadata)
        self.camera_sequences = self._load_camera_sequences(segment_dir)
        self._build_camera_slots()
        if self.play_button is not None:
            self.play_button.configure(state=tk.NORMAL)

    def _load_pose_playback(self, segment_dir: Path, metadata: Dict[str, Any]) -> None:
        self.playback_timestamps = np.array([], dtype=np.float64)
        self.playback_xyz = np.empty((0, 3), dtype=np.float64)
        pose_path = segment_dir / "pose.npz"
        if not pose_path.exists():
            return

        with np.load(pose_path, allow_pickle=True) as pose_npz:
            timestamps = np.asarray(pose_npz["timestamps"], dtype=np.float64)
            data = np.asarray(pose_npz["data"], dtype=np.float64)

        arm = str(metadata.get("arm") or "left")
        arm_index = ARM_TO_INDEX.get(arm, 0)
        if data.ndim != 3 or data.shape[1] <= arm_index:
            return

        self.playback_timestamps = timestamps
        self.playback_xyz = data[:, arm_index, :3]
        if timestamps.size > 0:
            self.playback_start_sec = float(min(self.playback_start_sec, timestamps[0]))
            self.playback_end_sec = float(max(self.playback_end_sec, timestamps[-1]))

    def _load_camera_sequences(self, segment_dir: Path) -> Dict[str, List[Tuple[float, Path]]]:
        sequences: Dict[str, List[Tuple[float, Path]]] = {}
        for camera_dir in sorted(segment_dir.glob("camera_*")):
            if not camera_dir.is_dir():
                continue
            role = camera_dir.name.replace("camera_", "", 1)
            frames = self._load_camera_frame_metadata(camera_dir)
            if frames:
                sequences[role] = frames
        return sequences

    def _load_camera_frame_metadata(self, camera_dir: Path) -> List[Tuple[float, Path]]:
        metadata_path = camera_dir / "frames.json"
        frames: List[Tuple[float, Path]] = []
        if metadata_path.exists():
            try:
                raw_items = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to read camera frame metadata {metadata_path}: {exc}")
                raw_items = []
            for item in raw_items:
                try:
                    timestamp = float(item.get("timestamp_sec", 0.0))
                    image_path = camera_dir / str(item.get("file") or "")
                except (AttributeError, TypeError, ValueError):
                    continue
                if image_path.is_file():
                    frames.append((timestamp, image_path))
        if not frames:
            for image_path in sorted(camera_dir.glob("*.png")):
                frames.append((0.0, image_path))
        frames.sort(key=lambda item: item[0])
        return frames

    def _draw_empty_plot(self, message: str) -> None:
        if self.pose_axis is None or self.canvas is None:
            return
        self.pose_axis.clear()
        self._style_pose_axis()
        self.pose_axis.text2D(
            0.5,
            0.5,
            message,
            transform=self.pose_axis.transAxes,
            ha="center",
            va="center",
            color="#6b7280",
        )
        self.canvas.draw_idle()

    def _draw_playback_frame(self, current_time: float) -> None:
        if self.pose_axis is None or self.canvas is None:
            return
        self.pose_axis.clear()
        self._style_pose_axis()

        event_time = float(self.playback_metadata.get("time_sec", 0.0))
        arm = str(self.playback_metadata.get("arm") or "left")
        timestamps = self.playback_timestamps
        xyz = self.playback_xyz
        if timestamps.size == 0 or xyz.size == 0:
            self.pose_axis.text2D(0.5, 0.5, "pose.npz not found", transform=self.pose_axis.transAxes)
            self.canvas.draw_idle()
            return

        current_index = int(np.searchsorted(timestamps, current_time, side="right") - 1)
        current_index = min(max(current_index, 0), len(timestamps) - 1)
        event_index = int(np.argmin(np.abs(timestamps - event_time)))

        self.pose_axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color="#64748b",
            linewidth=1,
            alpha=0.45,
            label="full window",
        )
        self.pose_axis.plot(
            xyz[: current_index + 1, 0],
            xyz[: current_index + 1, 1],
            xyz[: current_index + 1, 2],
            color="#60a5fa",
            linewidth=2.4,
            label="playback",
        )
        self.pose_axis.scatter(
            [xyz[current_index, 0]],
            [xyz[current_index, 1]],
            [xyz[current_index, 2]],
            color="#facc15",
            s=34,
            label="current",
        )
        self.pose_axis.scatter(
            [xyz[event_index, 0]],
            [xyz[event_index, 1]],
            [xyz[event_index, 2]],
            color="#f87171",
            s=42,
            label=f"{arm} jump",
        )
        legend = self.pose_axis.legend(loc="upper left", fontsize=8)
        for text in legend.get_texts():
            text.set_color("#e5e7eb")
        self.pose_axis.set_title(f"Pose 3D ({arm}) t={current_time:.2f}s", color="#e5e7eb")
        self._set_equal_pose_limits(xyz)

        self.canvas.draw_idle()

    def _build_camera_slots(self) -> None:
        self._clear_images()
        if self.image_strip is None:
            return
        if not self.camera_sequences:
            ttk.Label(self.image_strip, text="No camera frames").grid(row=0, column=0, sticky=tk.W)
            return

        for column, role in enumerate(sorted(self.camera_sequences.keys())):
            frame = ttk.Frame(self.image_strip)
            frame.grid(row=0, column=column, padx=5, pady=4, sticky=tk.NW)
            ttk.Label(frame, text=f"camera_{role}", font=("Consolas", 8, "bold")).pack()
            image_label = tk.Label(frame, bg="#111827", fg="#9ca3af", text="waiting")
            image_label.pack()
            caption_var = tk.StringVar(value="--")
            ttk.Label(frame, textvariable=caption_var, font=("Consolas", 7)).pack()
            self.camera_labels[role] = image_label
            self.camera_caption_vars[role] = caption_var

    def _update_camera_frames(self, current_time: float) -> None:
        photos: List[tk.PhotoImage] = []
        for role, frames in self.camera_sequences.items():
            label = self.camera_labels.get(role)
            caption_var = self.camera_caption_vars.get(role)
            if label is None or not frames:
                continue
            frame_index = min(
                range(len(frames)),
                key=lambda index: abs(frames[index][0] - current_time),
            )
            timestamp, image_path = frames[frame_index]
            try:
                photo = tk.PhotoImage(file=str(image_path))
            except tk.TclError as exc:
                logger.warning(f"Failed to load anomaly playback image {image_path}: {exc}")
                continue
            label.configure(image=photo, text="")
            label.image = photo
            photos.append(photo)
            if caption_var is not None:
                caption_var.set(f"t={timestamp:.3f}s")
        self.photo_refs = photos

    def _start_playback(self) -> None:
        if self.playback_segment_dir is None:
            return
        self._cancel_playback()
        self.playback_started_at = time.monotonic()
        self._playback_tick()

    def _cancel_playback(self) -> None:
        if self.playback_job is None:
            return
        try:
            self.parent.after_cancel(self.playback_job)
        except Exception:
            pass
        self.playback_job = None

    def _playback_tick(self) -> None:
        duration = max(0.1, self.playback_end_sec - self.playback_start_sec)
        elapsed = time.monotonic() - self.playback_started_at
        current_time = self.playback_start_sec + min(elapsed, duration)
        self._draw_playback_frame(current_time)
        self._update_camera_frames(current_time)

        if elapsed < duration:
            self.playback_job = self.parent.after(120, self._playback_tick)
        else:
            self.playback_job = None

    def _clear_images(self) -> None:
        self.photo_refs = []
        self.camera_labels = {}
        self.camera_caption_vars = {}
        if self.image_strip is None:
            return
        for child in self.image_strip.winfo_children():
            child.destroy()

    def _style_pose_axis(self) -> None:
        if self.pose_axis is None:
            return
        self.pose_axis.set_facecolor("#111827")
        self.pose_axis.set_title("Pose 3D", color="#e5e7eb")
        self.pose_axis.set_xlabel("X (m)", color="#9ca3af")
        self.pose_axis.set_ylabel("Y (m)", color="#9ca3af")
        self.pose_axis.set_zlabel("Z (m)", color="#9ca3af")
        self.pose_axis.tick_params(colors="#9ca3af", labelsize=8)
        self.pose_axis.grid(True, color="#374151", alpha=0.35)

    def _set_equal_pose_limits(self, xyz: np.ndarray) -> None:
        if self.pose_axis is None or xyz.size == 0:
            return
        mins = np.min(xyz, axis=0)
        maxs = np.max(xyz, axis=0)
        mids = (mins + maxs) / 2.0
        span = max(float(np.max(maxs - mins)), 0.1)
        half = span * 0.6
        self.pose_axis.set_xlim(mids[0] - half, mids[0] + half)
        self.pose_axis.set_ylim(mids[1] - half, mids[1] + half)
        self.pose_axis.set_zlim(mids[2] - half, mids[2] + half)
        try:
            self.pose_axis.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

