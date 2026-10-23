#!/usr/bin/env python3
"""
Tkinter frame for active hardware diagnostics in the Vive DM3 GUI.

Author: Yichen Xie
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasTkAgg = None
    Figure = None

from incubator.nedf3.Vive_DM3.controller import ViveDM3Controller
from incubator.nedf3.Vive_DM3.diagnostics import (
    CameraDiagnosticResult,
    EncoderDiagnosticResult,
    _probe_camera_index,
    load_diagnostic_config,
    run_encoder_diagnostics,
)
from incubator.nedf3.Vive_DM3.process_manager import DEFAULT_CONFIG_NAME, scan_config_files, subprocess_env
from utils.logger_config import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_KEYS = (
    ("right_capture_id", "右腕 USB"),
    ("left_capture_id", "左腕 USB"),
    ("ego_capture_id", "头戴 USB"),
)
DIAGNOSTIC_ENCODER_DURATION_SEC = 5.0
DIAGNOSTIC_TRACKER_INTERVAL_SEC = 0.2
DIAGNOSTIC_CAMERA_PREVIEW_WIDTH = 320
DIAGNOSTIC_CAMERA_PREVIEW_HEIGHT = 180
TRACKER_TRAJECTORY_MAX_POINTS = 500
TRACKER_TRAJECTORY_COLORS = (
    "#22c55e",
    "#60a5fa",
    "#f97316",
    "#e879f9",
    "#facc15",
    "#2dd4bf",
    "#f87171",
    "#a78bfa",
)


class TrackerDiagnosticFrame:
    """Manage real-time Vive Tracker diagnostics and 3D trajectory display."""

    def __init__(
        self,
        parent: ttk.Frame,
        *,
        root: tk.Tk,
        controller: ViveDM3Controller,
        config_var: tk.StringVar,
        on_config_changed: Callable[[], None],
        other_diagnostic_running: Callable[[], bool],
        log: Callable[..., None],
        show_warning: Callable[[str, str], None],
        show_error: Callable[..., None],
        on_state_changed: Optional[Callable[[], None]] = None,
    ) -> None:
        self.parent = parent
        self.root = root
        self.controller = controller
        self.config_var = config_var
        self.on_config_changed = on_config_changed
        self.other_diagnostic_running = other_diagnostic_running
        self.log = log
        self.show_warning = show_warning
        self.show_error = show_error
        self.on_state_changed = on_state_changed

        self.status_var = tk.StringVar(value="Tracker：未启动检测")
        self.tracker_process: Optional[subprocess.Popen[str]] = None
        self.tracker_tree: Optional[ttk.Treeview] = None
        self.tracker_trajectories: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self.tracker_role_order: List[str] = []
        self.tracker_figure: Optional[Any] = None
        self.tracker_axis: Optional[Any] = None
        self.tracker_plot_canvas: Optional[Any] = None
        self.buttons: Dict[str, ttk.Button] = {}

        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        top = ttk.LabelFrame(parent, text="Tracker 轨迹自检", padding=8)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="配置：").pack(side=tk.LEFT)
        config_combo = ttk.Combobox(
            top,
            textvariable=self.config_var,
            values=scan_config_files(),
            state="readonly",
            width=28,
        )
        config_combo.pack(side=tk.LEFT, padx=(4, 12))
        config_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_config_changed())
        ttk.Label(top, textvariable=self.status_var, foreground="#374151").pack(side=tk.LEFT)

        self.buttons["stop"] = ttk.Button(
            top,
            text="停止 Tracker 检测",
            style="Stop.TButton",
            command=self.stop_tracker_diagnostic,
            state=tk.DISABLED,
        )
        self.buttons["stop"].pack(side=tk.RIGHT, padx=4)
        self.buttons["start"] = ttk.Button(
            top,
            text="开始 Tracker 实时检测",
            command=self.start_tracker_diagnostic,
        )
        self.buttons["start"].pack(side=tk.RIGHT, padx=4)

        content = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.LabelFrame(content, text="实时状态", padding=8)
        columns = ("role", "valid", "tracked", "position", "quaternion")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        for col, title, width in (
            ("role", "Role", 120),
            ("valid", "Valid", 80),
            ("tracked", "Tracked", 90),
            ("position", "Position xyz", 230),
            ("quaternion", "Quaternion xyzw", 300),
        ):
            tree.heading(col, text=title)
            tree.column(col, width=width, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        self.tracker_tree = tree
        content.add(table_frame, weight=1)

        plot_frame = ttk.LabelFrame(content, text="3D 轨迹", padding=8)
        self._build_tracker_plot(plot_frame)
        content.add(plot_frame, weight=3)
        self.update_gate()

    def update_gate(self) -> None:
        if not self.buttons:
            return
        device = self.controller.processes.device
        saver = self.controller.processes.saver
        busy = (
            device.running
            or device.process is not None
            or device.stopping
            or saver.running
            or saver.process is not None
            or saver.stopping
        )
        tracker_running = self.is_running()
        other_running = self.other_diagnostic_running()
        self.buttons["start"].configure(
            state=tk.DISABLED if busy or other_running or tracker_running else tk.NORMAL
        )
        self.buttons["stop"].configure(state=tk.NORMAL if tracker_running else tk.DISABLED)
        if busy:
            self.status_var.set("Tracker：请先停止设备/保存进程")
        elif other_running:
            self.status_var.set("Tracker：请等待相机/编码器自检完成")

    def is_running(self) -> bool:
        return self.tracker_process is not None and self.tracker_process.poll() is None

    def _notify_state_changed(self) -> None:
        if self.on_state_changed is not None:
            self.on_state_changed()

    def _diagnostics_allowed(self) -> bool:
        device = self.controller.processes.device
        saver = self.controller.processes.saver
        if (
            device.running
            or device.process is not None
            or device.stopping
            or saver.running
            or saver.process is not None
            or saver.stopping
        ):
            self.show_warning("Tracker 自检", "请先停止设备/保存进程后再执行 Tracker 自检。")
            self.update_gate()
            return False
        if self.other_diagnostic_running():
            self.show_warning("Tracker 自检", "相机/编码器自检正在运行，请等待完成后再启动 Tracker。")
            self.update_gate()
            return False
        if self.is_running():
            return False
        return True

    def _diagnostic_config_name(self) -> str:
        return self.config_var.get().strip() or DEFAULT_CONFIG_NAME

    def _build_tracker_plot(self, parent: ttk.Frame) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            ttk.Label(
                parent,
                text="matplotlib unavailable\n3D trajectory disabled",
                foreground="#9ca3af",
                anchor=tk.CENTER,
                justify=tk.CENTER,
            ).pack(fill=tk.BOTH, expand=True)
            return

        figure = Figure(figsize=(7.5, 5.2), dpi=100)
        figure.patch.set_facecolor("#111827")
        axis = figure.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(figure, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#374151",
        )
        widget.pack(fill=tk.BOTH, expand=True)

        self.tracker_figure = figure
        self.tracker_axis = axis
        self.tracker_plot_canvas = canvas
        self._draw_tracker_plot()

    def _reset_tracker_trajectories(self, roles: List[str]) -> None:
        self.tracker_role_order = [str(role) for role in roles]
        self.tracker_trajectories = {
            role: deque(maxlen=TRACKER_TRAJECTORY_MAX_POINTS)
            for role in self.tracker_role_order
        }
        self._draw_tracker_plot()

    def _tracker_color(self, role: str) -> str:
        if role not in self.tracker_role_order:
            self.tracker_role_order.append(role)
        index = self.tracker_role_order.index(role)
        return TRACKER_TRAJECTORY_COLORS[index % len(TRACKER_TRAJECTORY_COLORS)]

    def _parse_tracker_position(self, pose: Any) -> Optional[Tuple[float, float, float]]:
        if not isinstance(pose, list) or len(pose) < 3:
            return None
        try:
            return float(pose[0]), float(pose[1]), float(pose[2])
        except (TypeError, ValueError):
            return None

    def _append_tracker_position(
        self,
        role: str,
        position: Tuple[float, float, float],
        status: Dict[str, Any],
    ) -> None:
        if not status.get("position_valid") or not status.get("position_tracked"):
            return
        if role not in self.tracker_role_order:
            self.tracker_role_order.append(role)
        if role not in self.tracker_trajectories:
            self.tracker_trajectories[role] = deque(maxlen=TRACKER_TRAJECTORY_MAX_POINTS)
        self.tracker_trajectories[role].append(position)

    def _draw_tracker_plot(self) -> None:
        axis = self.tracker_axis
        canvas = self.tracker_plot_canvas
        if axis is None or canvas is None:
            return

        axis.clear()
        axis.set_facecolor("#111827")
        axis.set_title("Tracker 3D trajectory", color="#e5e7eb", fontsize=11, pad=10)
        axis.set_xlabel("X (m)", color="#9ca3af", labelpad=6)
        axis.set_ylabel("Y (m)", color="#9ca3af", labelpad=6)
        axis.set_zlabel("Z (m)", color="#9ca3af", labelpad=6)
        axis.tick_params(colors="#9ca3af", labelsize=8)
        axis.grid(True, color="#374151", alpha=0.35)

        all_points: List[Tuple[float, float, float]] = []
        plotted_roles = 0
        for role in self.tracker_role_order:
            points = list(self.tracker_trajectories.get(role, []))
            if not points:
                continue
            color = self._tracker_color(role)
            xs, ys, zs = zip(*points)
            axis.plot(xs, ys, zs, color=color, linewidth=2.0, label=f"{role} ({len(points)})")
            axis.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=color, s=24)
            all_points.extend(points)
            plotted_roles += 1

        if plotted_roles == 0:
            axis.text2D(
                0.5,
                0.5,
                "waiting for tracked poses",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
                fontsize=10,
            )
            self._set_tracker_plot_limits([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        else:
            legend = axis.legend(
                loc="upper left",
                fontsize=8,
                facecolor="#111827",
                edgecolor="#374151",
            )
            for text in legend.get_texts():
                text.set_color("#e5e7eb")
            self._set_tracker_plot_limits(all_points)

        try:
            axis.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        canvas.draw_idle()

    def _set_tracker_plot_limits(self, points: List[Tuple[float, float, float]]) -> None:
        axis = self.tracker_axis
        if axis is None or not points:
            return
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        zs = [point[2] for point in points]
        x_mid = (min(xs) + max(xs)) / 2.0
        y_mid = (min(ys) + max(ys)) / 2.0
        z_mid = (min(zs) + max(zs)) / 2.0
        span = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
            0.1,
        )
        half_span = span * 0.6
        axis.set_xlim(x_mid - half_span, x_mid + half_span)
        axis.set_ylim(y_mid - half_span, y_mid + half_span)
        axis.set_zlim(z_mid - half_span, z_mid + half_span)

    def start_tracker_diagnostic(self) -> None:
        if not self._diagnostics_allowed():
            return
        config_name = self._diagnostic_config_name()
        if self.tracker_tree is not None:
            self.tracker_tree.delete(*self.tracker_tree.get_children())
        self._reset_tracker_trajectories([])
        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "vive_tracker_status_probe.py"),
            "--config-name",
            config_name,
            "--interval-sec",
            str(DIAGNOSTIC_TRACKER_INTERVAL_SEC),
        ]
        self.status_var.set("Tracker：正在启动 probe...")
        try:
            self.tracker_process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=subprocess_env(),
            )
        except Exception as exc:
            self.status_var.set(f"Tracker：启动失败 {exc}")
            self.show_error(exc, title="Tracker 检测失败")
            self.update_gate()
            self._notify_state_changed()
            return
        threading.Thread(target=self._read_tracker_probe_output, daemon=True).start()
        self.update_gate()
        self._notify_state_changed()

    def _read_tracker_probe_output(self) -> None:
        process = self.tracker_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.root.after(0, lambda payload=line: self._handle_tracker_probe_line(payload))
        exit_code = process.wait()
        self.root.after(0, lambda code=exit_code: self._on_tracker_probe_exit(code))

    def _handle_tracker_probe_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.status_var.set(f"Tracker：{text[-180:]}")
            return
        payload_type = str(payload.get("type") or "")
        if payload_type == "started":
            roles = [str(role) for role in payload.get("roles") or []]
            self._reset_tracker_trajectories(roles)
            self.status_var.set(f"Tracker：probe 已启动 roles={roles}")
            return
        if payload_type == "error":
            stage = str(payload.get("stage") or "runtime")
            message = str(payload.get("message") or "")
            trace = str(payload.get("traceback") or "").strip()
            self.status_var.set(f"Tracker：错误[{stage}] {message}")
            self.log(f"Tracker 自检错误[{stage}]：{message}\n", level="error")
            if trace:
                self.log(trace + "\n", level="error")
            return
        if payload_type != "tracker_status":
            return
        session_state = str(payload.get("session_state") or "")
        statuses = payload.get("statuses")
        if not isinstance(statuses, dict):
            statuses = {}
        self.status_var.set(
            f"Tracker：{session_state} | roles={len(statuses)} | "
            f"updated={time.strftime('%H:%M:%S')}"
        )
        tree = self.tracker_tree
        if tree is not None:
            tree.delete(*tree.get_children())
        for role_path, status in sorted(statuses.items()):
            if not isinstance(status, dict):
                continue
            role = str(status.get("role") or role_path)
            valid = (
                "Y" if status.get("position_valid") and status.get("orientation_valid") else "N"
            )
            tracked = (
                "Y" if status.get("position_tracked") and status.get("orientation_tracked") else "N"
            )
            pose = status.get("pose")
            position_text = "-"
            quat_text = "-"
            position = self._parse_tracker_position(pose)
            if isinstance(pose, list) and len(pose) == 7:
                position_text = f"{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}"
                quat_text = f"{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f}"
            if position is not None:
                self._append_tracker_position(role, position, status)
            if tree is not None:
                tree.insert("", tk.END, values=(role, valid, tracked, position_text, quat_text))
        self._draw_tracker_plot()

    def _on_tracker_probe_exit(self, exit_code: int) -> None:
        if self.tracker_process is not None and self.tracker_process.poll() is not None:
            self.tracker_process = None
        current_status = self.status_var.get()
        if exit_code != 0 and "错误" in current_status:
            self.status_var.set(f"{current_status} | exit={exit_code}")
        else:
            self.status_var.set(f"Tracker：probe 已停止 exit={exit_code}")
        self.update_gate()
        self._notify_state_changed()

    def stop_tracker_diagnostic(self) -> None:
        process = self.tracker_process
        if process is None:
            self.update_gate()
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as exc:
                logger.warning(f"Failed to stop tracker diagnostic probe: {exc}")
        self.tracker_process = None
        self.status_var.set("Tracker：已停止检测")
        self.update_gate()
        self._notify_state_changed()


class DiagnosticFrame:
    """Manage the device diagnostics tab and single-test execution gates."""

    def __init__(
        self,
        parent: ttk.Frame,
        *,
        root: tk.Tk,
        controller: ViveDM3Controller,
        config_var: tk.StringVar,
        camera_vars: Dict[str, tk.StringVar],
        encoder_port_vars: Dict[str, tk.StringVar],
        on_config_changed: Callable[[], None],
        run_background: Callable[..., None],
        log: Callable[..., None],
        show_warning: Callable[[str, str], None],
        show_error: Callable[..., None],
        is_tracker_running: Optional[Callable[[], bool]] = None,
        on_state_changed: Optional[Callable[[], None]] = None,
    ) -> None:
        self.parent = parent
        self.root = root
        self.controller = controller
        self.config_var = config_var
        self.camera_vars = camera_vars
        self.encoder_port_vars = encoder_port_vars
        self.on_config_changed = on_config_changed
        self.run_background = run_background
        self.is_tracker_running = is_tracker_running or (lambda: False)
        self.on_state_changed = on_state_changed
        self.log = log
        self.show_warning = show_warning
        self.show_error = show_error

        self.status_var = tk.StringVar(value="设备自检：等待操作")
        self.camera_status_vars: Dict[str, tk.StringVar] = {}
        self.camera_image_labels: Dict[str, tk.Label] = {}
        self.camera_photos: Dict[str, tk.PhotoImage] = {}
        self.camera_running = False
        self.encoder_running = False
        self.encoder_status_vars: Dict[str, tk.StringVar] = {
            "left": tk.StringVar(value="左编码器：未检测"),
            "right": tk.StringVar(value="右编码器：未检测"),
        }
        self.encoder_samples: Dict[str, List[Tuple[float, float]]] = {
            "left": [],
            "right": [],
        }
        self.encoder_canvas: Optional[tk.Canvas] = None
        self.buttons: Dict[str, ttk.Button] = {}

        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        top = ttk.LabelFrame(parent, text="采集前主动自检", padding=8)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="配置：").pack(side=tk.LEFT)
        config_combo = ttk.Combobox(
            top,
            textvariable=self.config_var,
            values=scan_config_files(),
            state="readonly",
            width=28,
        )
        config_combo.pack(side=tk.LEFT, padx=(4, 12))
        config_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_config_changed())
        ttk.Label(
            top,
            textvariable=self.status_var,
            foreground="#374151",
        ).pack(side=tk.LEFT)
        self.buttons["encoder"] = ttk.Button(
            top,
            text="编码器检测 5s",
            command=self.run_encoder_diagnostic,
        )
        self.buttons["encoder"].pack(side=tk.RIGHT, padx=4)
        self.buttons["camera"] = ttk.Button(
            top,
            text="相机端口检测",
            command=self.run_camera_diagnostic,
        )
        self.buttons["camera"].pack(side=tk.RIGHT, padx=4)

        camera_frame = ttk.LabelFrame(parent, text="相机首帧预览", padding=8)
        camera_frame.pack(fill=tk.X, pady=(0, 8))
        for col, (key, label) in enumerate(CAMERA_KEYS):
            card = ttk.Frame(camera_frame)
            card.grid(row=0, column=col, sticky=tk.NSEW, padx=4)
            camera_frame.columnconfigure(col, weight=1)
            ttk.Label(card, text=label).pack(anchor=tk.W)
            status = tk.StringVar(value="未检测")
            self.camera_status_vars[key] = status
            ttk.Label(card, textvariable=status, foreground="#374151").pack(anchor=tk.W)
            preview = tk.Frame(
                card,
                width=DIAGNOSTIC_CAMERA_PREVIEW_WIDTH,
                height=DIAGNOSTIC_CAMERA_PREVIEW_HEIGHT,
                bg="#111827",
            )
            preview.pack(fill=tk.NONE, expand=False, pady=(4, 0))
            preview.pack_propagate(False)
            image = tk.Label(
                preview,
                text="No frame",
                bg="#111827",
                fg="#9ca3af",
            )
            image.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
            self.camera_image_labels[key] = image

        encoder_frame = ttk.LabelFrame(parent, text="编码器读数曲线", padding=8)
        encoder_frame.pack(fill=tk.X, pady=(0, 8))
        encoder_status = ttk.Frame(encoder_frame)
        encoder_status.pack(fill=tk.X)
        ttk.Label(
            encoder_status,
            textvariable=self.encoder_status_vars["left"],
            foreground="#16a34a",
        ).pack(side=tk.LEFT)
        ttk.Label(
            encoder_status,
            textvariable=self.encoder_status_vars["right"],
            foreground="#2563eb",
        ).pack(side=tk.LEFT, padx=(18, 0))
        self.encoder_canvas = tk.Canvas(
            encoder_frame,
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#374151",
            height=220,
        )
        self.encoder_canvas.pack(fill=tk.X, expand=False, pady=(6, 0))
        self.encoder_canvas.bind("<Configure>", lambda _event: self._draw_encoder_plot())

        self.update_gate()

    def update_gate(self) -> None:
        if not self.buttons:
            return
        device = self.controller.processes.device
        saver = self.controller.processes.saver
        busy = (
            device.running
            or device.process is not None
            or device.stopping
            or saver.running
            or saver.process is not None
            or saver.stopping
        )
        diagnostic_running = self.camera_running or self.encoder_running
        tracker_running = self.is_tracker_running()
        normal_state = tk.DISABLED if busy or diagnostic_running or tracker_running else tk.NORMAL
        for key in ("camera", "encoder"):
            if key in self.buttons:
                self.buttons[key].configure(state=normal_state)
        if busy:
            self.status_var.set("设备自检：请先停止设备/保存进程")
        elif tracker_running:
            self.status_var.set("设备自检：请先停止 Tracker 自检")

    def _diagnostics_allowed(self) -> bool:
        device = self.controller.processes.device
        saver = self.controller.processes.saver
        if (
            device.running
            or device.process is not None
            or device.stopping
            or saver.running
            or saver.process is not None
            or saver.stopping
        ):
            self.show_warning("设备自检", "请先停止设备/保存进程后再执行主动自检。")
            self.update_gate()
            return False
        if self.camera_running or self.encoder_running:
            self.show_warning("设备自检", "已有单项自检正在运行，请等待完成后再执行其它自检。")
            self.update_gate()
            return False
        if self.is_tracker_running():
            self.show_warning("设备自检", "Tracker 自检正在运行，请先停止后再执行其它自检。")
            self.update_gate()
            return False
        return True

    def _diagnostic_config_name(self) -> str:
        return self.config_var.get().strip() or DEFAULT_CONFIG_NAME

    def is_running(self) -> bool:
        return self.camera_running or self.encoder_running

    def _notify_state_changed(self) -> None:
        if self.on_state_changed is not None:
            self.on_state_changed()

    def run_camera_diagnostic(self) -> None:
        if not self._diagnostics_allowed():
            return
        if self.camera_running:
            return
        self.camera_running = True
        camera_indices = {key: var.get().strip() for key, var in self.camera_vars.items()}
        for key, status in self.camera_status_vars.items():
            status.set(f"{camera_indices.get(key, '')}: 检测中...")
            label = self.camera_image_labels.get(key)
            if label is not None:
                label.configure(image="", text="checking...", bg="#111827", fg="#9ca3af")
        self.status_var.set("设备自检：正在检测相机端口")
        self.update_gate()
        self._notify_state_changed()

        def runner() -> None:
            results: Dict[str, CameraDiagnosticResult] = {}
            try:
                for key, label in CAMERA_KEYS:
                    result = _probe_camera_index(label, str(camera_indices.get(key) or "").strip())
                    results[key] = result
                    self.root.after(
                        0,
                        lambda item_key=key, item_result=result: self._render_camera_result(
                            item_key,
                            item_result,
                        ),
                    )
            finally:
                self.root.after(0, lambda final_results=results: self._on_camera_complete(final_results))

        threading.Thread(target=runner, daemon=True).start()

    def _on_camera_complete(self, results: Dict[str, CameraDiagnosticResult]) -> None:
        self.camera_running = False
        ok_count = sum(1 for result in results.values() if result.ok)
        self.status_var.set(f"设备自检：相机检测完成 {ok_count}/{len(results)} OK")
        self.log(f"相机端口检测完成：{ok_count}/{len(results)} OK\n")
        self.update_gate()
        self._notify_state_changed()

    def _render_camera_result(self, key: str, result: CameraDiagnosticResult) -> None:
        status = self.camera_status_vars.get(key)
        if status is not None:
            prefix = "OK" if result.ok else "ERROR"
            status.set(f"{prefix} index={result.capture_id} {result.message}")
        label = self.camera_image_labels.get(key)
        if label is None:
            return
        if result.image_png_b64:
            try:
                photo = tk.PhotoImage(data=result.image_png_b64, format="png")
            except tk.TclError as exc:
                try:
                    preview_dir = PROJECT_ROOT / "logs" / "diagnostic_camera_previews"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview_path = preview_dir / f"{key}.png"
                    preview_path.write_bytes(base64.b64decode(result.image_png_b64))
                    photo = tk.PhotoImage(file=str(preview_path))
                except Exception:
                    label.configure(
                        image="",
                        text=f"preview decode failed: {exc}",
                        bg="#111827",
                        fg="#f87171",
                    )
                    return
            self.camera_photos[key] = photo
            label.configure(image=photo, text="", bg="#111827")
        else:
            label.configure(image="", text=result.message, bg="#111827", fg="#f87171")

    def run_encoder_diagnostic(self) -> None:
        if not self._diagnostics_allowed():
            return
        if self.encoder_running:
            return
        config_name = self._diagnostic_config_name()
        diagnostic_config = load_diagnostic_config(config_name)
        encoder_ports = {key: var.get().strip() for key, var in self.encoder_port_vars.items()}
        encoder_baudrates = diagnostic_config.get("encoder_baudrates") or {}
        encoder_fps = diagnostic_config.get("encoder_fps") or {}
        self.encoder_running = True
        self.encoder_samples = {"left": [], "right": []}
        self._draw_encoder_plot()
        for side in ("left", "right"):
            self.encoder_status_vars[side].set(f"{side}: 检测中 5s...")
        self.status_var.set("设备自检：正在采样编码器 5s")
        self.update_gate()
        self._notify_state_changed()

        def work() -> Dict[str, EncoderDiagnosticResult]:
            return run_encoder_diagnostics(
                encoder_ports,
                encoder_baudrates,
                encoder_fps,
                duration_sec=DIAGNOSTIC_ENCODER_DURATION_SEC,
            )

        def on_success(results: Dict[str, EncoderDiagnosticResult]) -> None:
            self._on_encoder_complete(results)

        def on_finished() -> None:
            self.encoder_running = False
            self.update_gate()
            self._notify_state_changed()

        self.run_background(work, on_success, finally_callback=on_finished)

    def _on_encoder_complete(self, results: Dict[str, EncoderDiagnosticResult]) -> None:
        ok_count = 0
        for side, result in results.items():
            if result.ok:
                ok_count += 1
            self.encoder_samples[side] = list(result.samples)
            prefix = "OK" if result.ok else "ERROR"
            latest = result.samples[-1][1] if result.samples else None
            latest_text = f" calibrated={latest:.2f} deg" if latest is not None else ""
            self.encoder_status_vars[side].set(
                f"{prefix} {side} {result.port} hz={result.hz:.1f}/{result.target_fps:.1f}"
                f" samples={len(result.samples)} errors={result.error_count}{latest_text}"
            )
        self._draw_encoder_plot()
        self.status_var.set(f"设备自检：编码器检测完成 {ok_count}/{len(results)} OK")
        self.log(f"编码器检测完成：{ok_count}/{len(results)} OK\n")

    def _draw_encoder_plot(self) -> None:
        canvas = self.encoder_canvas
        if canvas is None:
            return
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")
        if width < 80 or height < 80:
            return
        plot_left = 46
        plot_right = width - 12
        plot_top = 16
        plot_bottom = height - 26
        mid_gap = 16
        plot_height = max(20, (plot_bottom - plot_top - mid_gap) // 2)
        rects = {
            "left": (plot_left, plot_top, plot_right, plot_top + plot_height),
            "right": (
                plot_left,
                plot_top + plot_height + mid_gap,
                plot_right,
                plot_top + plot_height + mid_gap + plot_height,
            ),
        }
        for side, color in (("left", "#22c55e"), ("right", "#60a5fa")):
            self._draw_encoder_side(
                canvas,
                rects[side],
                self.encoder_samples.get(side, []),
                side,
                color,
            )

    def _draw_encoder_side(
        self,
        canvas: tk.Canvas,
        rect: Tuple[int, int, int, int],
        samples: List[Tuple[float, float]],
        side: str,
        color: str,
    ) -> None:
        x0, y0, x1, y1 = rect
        canvas.create_rectangle(x0, y0, x1, y1, outline="#374151", fill="#0b1220")
        canvas.create_text(
            x0 + 5,
            y0 + 4,
            text=side,
            anchor=tk.NW,
            fill=color,
            font=("Consolas", 8, "bold"),
        )
        if not samples:
            canvas.create_text(
                (x0 + x1) // 2,
                (y0 + y1) // 2,
                text="waiting for samples",
                fill="#6b7280",
                font=("Consolas", 9),
            )
            return
        values = [value for _elapsed, value in samples]
        min_value = min(values)
        max_value = max(values)
        if max_value - min_value < 1e-6:
            min_value -= 1.0
            max_value += 1.0
        else:
            pad = 0.12 * (max_value - min_value)
            min_value -= pad
            max_value += pad
        canvas.create_text(
            x0 - 4,
            y0 + 2,
            text=f"{max_value:.1f}",
            anchor=tk.NE,
            fill="#9ca3af",
            font=("Consolas", 7),
        )
        canvas.create_text(
            x0 - 4,
            y1 - 2,
            text=f"{min_value:.1f}",
            anchor=tk.SE,
            fill="#9ca3af",
            font=("Consolas", 7),
        )
        points: List[float] = []
        x_range = max(DIAGNOSTIC_ENCODER_DURATION_SEC, samples[-1][0])
        y_range = max(1e-6, max_value - min_value)
        for elapsed, value in samples:
            x = x0 + (elapsed / x_range) * max(1, x1 - x0)
            y = y1 - ((value - min_value) / y_range) * max(1, y1 - y0)
            points.extend([x, y])
        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=2)
        else:
            canvas.create_oval(
                points[0] - 2,
                points[1] - 2,
                points[0] + 2,
                points[1] + 2,
                fill=color,
                outline=color,
            )
