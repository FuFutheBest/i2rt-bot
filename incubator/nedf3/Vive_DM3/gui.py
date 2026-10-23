#!/usr/bin/env python3
"""
Combined tkinter GUI for DM3 login, Vive data collection, and upload monitoring.

Author: Yichen Xie
"""

from __future__ import annotations

import json
import queue
import re
import sys
import threading
import tkinter as tk
import time
from collections import deque
from datetime import date
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from incubator.nedf3.Vive_DM3.process_manager import (
    DEFAULT_CONFIG_NAME,
    list_serial_ports,
    scan_config_files,
    strip_ansi_for_display,
)
from incubator.nedf3.Vive_DM3.controller import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EXTRA_DEVICE_TYPE,
    DEFAULT_EXTRA_HARDWARE_STRING,
    ViveDM3Controller,
)
from incubator.nedf3.Vive_DM3.anomaly_jump_gui import AnomalyJumpFrame
from incubator.nedf3.Vive_DM3.diagnostic_gui import (
    CAMERA_KEYS as _CAMERA_KEYS,
    DiagnosticFrame,
    TrackerDiagnosticFrame,
)
from incubator.nedf3.Vive_DM3.episode_stats import (
    DailyEpisodeStats,
    format_duration,
    scan_today_episode_stats,
)
from incubator.nedf3.Vive_DM3.resource_policy import apply_upload_process_policy
from incubator.nedf3.Vive_DM3.upload_gui import UploadConsoleFrame
from utils.gripper_angle_calibration import (
    GripperAngleCalibration,
    load_today_angle_calibration,
    process_gripper_angle_with_date_check,
)
from utils.logger_config import logger
from utils.shm_utils import (
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    MANAGER_HEADER_SIZE,
    connect_to_summary_shm,
    get_dtype,
    get_device_info_offset,
    get_summary_frame_offset,
    unpack_device_header,
    unpack_frame_header,
    unpack_manager_header,
)
from incubator.nedf3.windows_DM3.collection_session import extract_task_description
from incubator.nedf3.windows_DM3.dm3_api import DEFAULT_BASE_URL, DM3TaskExecution


_ERROR_COLOR = "#cc0000"
_WARNING_COLOR = "#d97706"
_SUCCESS_COLOR = "#2563eb"
_RECORDING_COLOR = "#16a34a"
_LOG_DEFAULT_COLOR = "#d4d4d4"
UI_FLUSH_INTERVAL_MS = 200
MAX_PROCESS_LOG_LINES_PER_FLUSH = 120
MAX_CONSOLE_LOG_LINES_PER_FLUSH = 80
ENTER_DEBOUNCE_SEC = 0.3
PENDING_START_ENTER_TTL_SEC = 30.0
GRIPPER_POLL_INTERVAL_SEC = 0.04
GRIPPER_RENDER_INTERVAL_MS = 40
GRIPPER_HISTORY_SEC = 30.0
GRIPPER_MAX_SAMPLES = int(GRIPPER_HISTORY_SEC / GRIPPER_POLL_INTERVAL_SEC) + 50

# Loguru lines: ``| ERROR    |`` / ``| WARNING  |`` (NO_COLOR subprocess output).
_PROCESS_LOG_ERROR_LEVEL_RE = re.compile(r"\|\s*ERROR\s+\|", re.IGNORECASE)
_PROCESS_LOG_WARNING_LEVEL_RE = re.compile(r"\|\s*WARNING\s+\|", re.IGNORECASE)
_DATA_ACQUISITION_STARTED_RE = re.compile(r"Data acquisition started\.|🟢+", re.IGNORECASE)
_EPISODE_COMPLETED_RE = re.compile(r"Episode completed in \d+(?:\.\d+)?s", re.IGNORECASE)

# async_data_saver_NEDF3 anomaly_detection + episode_anomaly_detector messages.
_ANOMALY_LOG_ERROR_RE = re.compile(
    r"❌|errors detected|Discarding finalized episode due to anomaly|"
    r"Error detecting anomalies in",
    re.IGNORECASE,
)
_ANOMALY_LOG_WARNING_RE = re.compile(
    r"⚠️|warnings detected|Found \d+ anomalies|🔍 Episode",
    re.IGNORECASE,
)


def _configure_log_text_tags(text: tk.Text) -> None:
    text.tag_configure("log_info", foreground=_LOG_DEFAULT_COLOR)
    text.tag_configure("log_recording", foreground=_RECORDING_COLOR)
    text.tag_configure("log_success", foreground=_SUCCESS_COLOR)
    text.tag_configure("log_warning", foreground=_WARNING_COLOR)
    text.tag_configure("log_error", foreground=_ERROR_COLOR)


def _classify_process_log_line(line: str) -> str:
    """Map saver/device subprocess log lines to console color tags."""
    plain = strip_ansi_for_display(line)
    if _PROCESS_LOG_ERROR_LEVEL_RE.search(plain) or _ANOMALY_LOG_ERROR_RE.search(plain):
        return "error"
    if _PROCESS_LOG_WARNING_LEVEL_RE.search(plain) or _ANOMALY_LOG_WARNING_RE.search(plain):
        return "warning"
    if _DATA_ACQUISITION_STARTED_RE.search(plain):
        return "recording"
    if _EPISODE_COMPLETED_RE.search(plain):
        return "success"
    return "info"


def _show_colored_message(
    parent: tk.Misc,
    title: str,
    message: str,
    *,
    color: str,
) -> None:
    """Modal dialog with colored title and message text."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(False, False)

    frame = ttk.Frame(dialog, padding=16)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        frame,
        text=title,
        font=("Sans", 11, "bold"),
        fg=color,
        anchor=tk.W,
    ).pack(fill=tk.X)
    tk.Label(
        frame,
        text=message,
        fg=color,
        justify=tk.LEFT,
        wraplength=420,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(8, 0))

    btn_row = ttk.Frame(frame)
    btn_row.pack(fill=tk.X, pady=(16, 0))
    ttk.Button(btn_row, text="确定", command=dialog.destroy).pack(side=tk.RIGHT)
    dialog.bind("<Return>", lambda _event: dialog.destroy())
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.wait_window()


class _GripperSummaryReader:
    """Read the latest rotary encoder samples from the summary SHM."""

    def __init__(self) -> None:
        self._summary_shm: Optional[Any] = None
        self._device_order: List[str] = []
        self._frame_sizes: List[int] = []
        self._device_info: Dict[str, Dict[str, Any]] = {}
        self._side_keys: Dict[str, str] = {}
        self._next_connect_attempt = 0.0
        self.status = "等待 device_summary_data"

    def close(self) -> None:
        if self._summary_shm is not None:
            try:
                self._summary_shm.close()
            except Exception as exc:
                logger.debug(f"Failed to close gripper summary SHM reader: {exc}")
        self._summary_shm = None
        self._device_order.clear()
        self._frame_sizes.clear()
        self._device_info.clear()
        self._side_keys.clear()

    def read_latest(self) -> Tuple[Dict[str, float], str]:
        if not self._ensure_connected():
            return {}, self.status
        try:
            values = self._read_latest_connected()
            if values:
                self.status = "夹爪 SHM 已连接"
            elif self._side_keys:
                self.status = "夹爪 SHM 已连接，等待数据"
            else:
                self.status = "未找到 RotaryEncoderDevice_0/1"
            return values, self.status
        except Exception as exc:
            self.status = f"夹爪 SHM 读取失败：{exc}"
            self.close()
            return {}, self.status

    def _ensure_connected(self) -> bool:
        if self._summary_shm is not None:
            return True
        now = time.monotonic()
        if now < self._next_connect_attempt:
            return False
        self._next_connect_attempt = now + 1.0
        try:
            self._summary_shm = connect_to_summary_shm(read_only=True)
            self._parse_summary_header()
            return True
        except FileNotFoundError:
            self.status = "等待 device_summary_data"
            self.close()
            return False
        except Exception as exc:
            self.status = f"夹爪 SHM 连接失败：{exc}"
            self.close()
            return False

    def _parse_summary_header(self) -> None:
        if self._summary_shm is None:
            return
        buf = self._summary_shm.buf
        manager_header = unpack_manager_header(buf[:MANAGER_HEADER_SIZE])
        device_count = int(manager_header["device_count"])
        self._device_order.clear()
        self._frame_sizes.clear()
        self._device_info.clear()
        self._side_keys.clear()
        for index in range(device_count):
            device_offset = get_device_info_offset(index)
            device_header = unpack_device_header(
                buf[device_offset : device_offset + DEVICE_HEADER_SIZE]
            )
            device_key = f"{device_header['device_type']}_{device_header['device_class_id']}"
            self._device_order.append(device_key)
            self._frame_sizes.append(int(device_header["frame_size"]))
            self._device_info[device_key] = device_header
            if device_header["device_type"] == "RotaryEncoderDevice":
                side = {0: "left", 1: "right"}.get(int(device_header["device_class_id"]))
                if side is not None:
                    self._side_keys[side] = device_key
        if self._side_keys:
            logger.info(f"Gripper monitor found devices: {self._side_keys}")

    def _read_latest_connected(self) -> Dict[str, float]:
        if self._summary_shm is None:
            return {}
        buf = self._summary_shm.buf
        manager_header = unpack_manager_header(buf[:MANAGER_HEADER_SIZE])
        current_frames_count = int(manager_header["current_frames_count"])
        buffer_size = int(manager_header["buffer_size"])
        write_index = int(manager_header["write_index"])
        device_count = int(manager_header["device_count"])
        if current_frames_count <= 0 or buffer_size <= 0:
            return {}
        frame_index = (write_index - 1) % buffer_size
        values: Dict[str, float] = {}
        for side, device_key in self._side_keys.items():
            value = self._read_device_value(
                device_key,
                frame_index,
                device_count,
                buffer_size,
            )
            if value is not None:
                values[side] = value
        return values

    def _read_device_value(
        self,
        device_key: str,
        frame_index: int,
        device_count: int,
        buffer_size: int,
    ) -> Optional[float]:
        if self._summary_shm is None or device_key not in self._device_info:
            return None
        device_index = self._device_order.index(device_key)
        frame_size = self._frame_sizes[device_index]
        frame_offset = get_summary_frame_offset(
            device_index,
            frame_index,
            device_count,
            self._frame_sizes,
            buffer_size,
        )
        buf = self._summary_shm.buf
        frame_header = unpack_frame_header(buf[frame_offset : frame_offset + FRAME_HEADER_SIZE])
        if int(frame_header["timestamp_ns"]) == 0:
            return None
        data_start = frame_offset + FRAME_HEADER_SIZE
        data_end = frame_offset + frame_size
        device_info = self._device_info[device_key]
        dtype = get_dtype(str(device_info["data_dtype"]), np.uint8)
        data = np.frombuffer(buf[data_start:data_end], dtype=dtype)
        if data.size == 0:
            return None
        return float(data.reshape(-1)[0])


class ViveDM3GUI:
    """Desktop GUI combining DM3 workflow and Vive data collection."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Vive DM3 数据采集控制台")
        self.root.geometry("1380x860")
        self.root.minsize(1100, 720)

        self.controller = ViveDM3Controller()
        self.controller.calibration.set_save_complete_callback(
            self._on_calibration_save_complete
        )
        self.current_tasks: List[DM3TaskExecution] = []
        self.selected_task: Optional[DM3TaskExecution] = None
        self.upload_panel: Optional[UploadConsoleFrame] = None
        self.diagnostic_panel: Optional[DiagnosticFrame] = None
        self.tracker_panel: Optional[TrackerDiagnosticFrame] = None
        self.anomaly_jump_panel: Optional[AnomalyJumpFrame] = None
        self.notebook: Optional[ttk.Notebook] = None
        self.dm3_tab: Optional[ttk.Frame] = None
        self.collect_tab: Optional[ttk.Frame] = None
        self.diagnostic_tab: Optional[ttk.Frame] = None
        self.tracker_tab: Optional[ttk.Frame] = None
        self.anomaly_jump_tab: Optional[ttk.Frame] = None
        self.upload_tab: Optional[ttk.Frame] = None

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._log_offsets: Dict[str, int] = {"device": 0, "saver": 0}
        self._upload_maintenance_waiting = False
        self._pending_start_enter_at: Optional[float] = None
        self._last_enter_event_at = 0.0
        self.gripper_status_var = tk.StringVar(value="夹爪曲线：等待设备启动")
        self.gripper_value_vars: Dict[str, tk.StringVar] = {
            "left": tk.StringVar(value="左夹爪: --"),
            "right": tk.StringVar(value="右夹爪: --"),
        }
        self.gripper_canvas: Optional[tk.Canvas] = None
        self._gripper_reader = _GripperSummaryReader()
        self._gripper_samples: Dict[str, Deque[Tuple[float, float]]] = {
            "left": deque(maxlen=GRIPPER_MAX_SAMPLES),
            "right": deque(maxlen=GRIPPER_MAX_SAMPLES),
        }
        self._gripper_latest_values: Dict[str, float] = {}
        self._gripper_status_text = "等待 device_summary_data"
        self._gripper_lock = threading.Lock()
        self._gripper_reader_lock = threading.Lock()
        self._gripper_stop_event = threading.Event()
        self._gripper_thread: Optional[threading.Thread] = None
        self._gripper_monitor_enabled = False
        self._gripper_calibrations: Dict[str, GripperAngleCalibration] = {}
        self._gripper_calibration_date = ""
        self._gripper_calibration_errors: Dict[str, str] = {}
        self._gripper_next_calibration_attempt: Dict[str, float] = {}
        self._recording_overlay: Optional[tk.Toplevel] = None
        self._recording_overlay_visible = False
        self._daily_stats_refreshing = False
        self._daily_stats_refresh_pending = False
        self._daily_stats_revision = 0
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Start.TButton", foreground="white", background="#2d8c3c")
        style.map("Start.TButton", background=[("active", "#3aaf4d")])
        style.configure("Stop.TButton", foreground="white", background="#c42b2b")
        style.map("Stop.TButton", background=[("active", "#e03e3e")])
        style.configure("Accent.TButton", foreground="white", background="#2563eb")
        style.map("Accent.TButton", background=[("active", "#1d4ed8")])
        style.configure("Danger.TButton", foreground="white", background="#b91c1c")
        style.map("Danger.TButton", background=[("active", "#991b1b")])

        self._build_ui()
        self._request_daily_stats_refresh()
        self._refresh_session_display()
        self._reload_config_fields()
        self._refresh_serial_ports()
        self._schedule_ui_flush()
        self._start_gripper_monitor()
        self._schedule_gripper_plot_refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Return>", self._on_global_enter)
        self.root.bind("<KP_Enter>", self._on_global_enter)

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        content = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content, padding=0)
        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        dm3_tab = ttk.Frame(notebook, padding=4)
        collect_tab = ttk.Frame(notebook, padding=4)
        diagnostic_tab = ttk.Frame(notebook, padding=4)
        tracker_tab = ttk.Frame(notebook, padding=4)
        anomaly_jump_tab = ttk.Frame(notebook, padding=4)
        upload_tab = ttk.Frame(notebook, padding=0)
        notebook.add(dm3_tab, text="DM3 登录")
        notebook.add(collect_tab, text="数据采集")
        notebook.add(diagnostic_tab, text="设备自检")
        notebook.add(tracker_tab, text="Tracker 轨迹自检")
        notebook.add(anomaly_jump_tab, text="跳变片段回放")
        notebook.add(upload_tab, text="上传与性能监控")
        self.notebook = notebook
        self.dm3_tab = dm3_tab
        self.collect_tab = collect_tab
        self.diagnostic_tab = diagnostic_tab
        self.tracker_tab = tracker_tab
        self.anomaly_jump_tab = anomaly_jump_tab
        self.upload_tab = upload_tab

        self._build_dm3_tab(dm3_tab)
        self._build_collection_tab(collect_tab)
        self.diagnostic_panel = DiagnosticFrame(
            diagnostic_tab,
            root=self.root,
            controller=self.controller,
            config_var=self.config_var,
            camera_vars=self.camera_vars,
            encoder_port_vars=self.encoder_port_vars,
            on_config_changed=self._on_config_changed,
            run_background=self._run_background,
            log=self._log,
            show_warning=self._show_warning,
            show_error=self._show_error,
            is_tracker_running=lambda: (
                self.tracker_panel.is_running() if self.tracker_panel is not None else False
            ),
            on_state_changed=lambda: (
                self.tracker_panel.update_gate() if self.tracker_panel is not None else None
            ),
        )
        self.tracker_panel = TrackerDiagnosticFrame(
            tracker_tab,
            root=self.root,
            controller=self.controller,
            config_var=self.config_var,
            on_config_changed=self._on_config_changed,
            other_diagnostic_running=lambda: (
                self.diagnostic_panel.is_running() if self.diagnostic_panel is not None else False
            ),
            log=self._log,
            show_warning=self._show_warning,
            show_error=self._show_error,
            on_state_changed=lambda: (
                self.diagnostic_panel.update_gate() if self.diagnostic_panel is not None else None
            ),
        )
        self.anomaly_jump_panel = AnomalyJumpFrame(
            anomaly_jump_tab,
            log=self._log,
            show_error=self._show_error,
        )
        self.upload_panel = UploadConsoleFrame(
            upload_tab,
            controller=self.controller,
            default_data_dir=DEFAULT_DATA_ROOT,
            show_login_controls=False,
        )
        self.upload_panel.pack(fill=tk.BOTH, expand=True)
        content.add(left_panel, weight=3)

        right_panel = ttk.PanedWindow(content, orient=tk.VERTICAL)

        log_frame = ttk.LabelFrame(right_panel, text="控制台日志", padding=6)
        self.console_log = tk.Text(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            width=42,
            state=tk.DISABLED,
        )
        scroll = ttk.Scrollbar(log_frame, command=self.console_log.yview)
        self.console_log.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _configure_log_text_tags(self.console_log)
        right_panel.add(log_frame, weight=1)

        gripper_frame = self._build_gripper_monitor(right_panel)
        right_panel.add(gripper_frame, weight=1)
        content.add(right_panel, weight=1)

    def _build_gripper_monitor(self, parent: ttk.PanedWindow) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="左右夹爪实时曲线", padding=6)
        value_row = ttk.Frame(frame)
        value_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(value_row, textvariable=self.gripper_value_vars["left"], foreground="#16a34a").pack(
            side=tk.LEFT
        )
        ttk.Label(value_row, textvariable=self.gripper_value_vars["right"], foreground="#2563eb").pack(
            side=tk.LEFT, padx=(14, 0)
        )
        ttk.Label(frame, textvariable=self.gripper_status_var, foreground="gray").pack(
            anchor=tk.W, pady=(0, 4)
        )
        self.gripper_canvas = tk.Canvas(
            frame,
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#374151",
            height=260,
        )
        self.gripper_canvas.pack(fill=tk.BOTH, expand=True)
        self.gripper_canvas.bind("<Configure>", lambda _event: self._draw_gripper_plot())
        return frame

    def _build_dm3_tab(self, parent: ttk.Frame) -> None:
        login = ttk.LabelFrame(parent, text="1. DM3 登录", padding=8)
        login.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(login, text="基础地址：").grid(row=0, column=0, sticky=tk.W)
        self.base_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        ttk.Entry(login, textvariable=self.base_url_var, width=36).grid(row=0, column=1, sticky=tk.W, padx=4)
        ttk.Label(login, text="手机号：").grid(row=0, column=2, sticky=tk.W, padx=(8, 0))
        self.phone_var = tk.StringVar()
        ttk.Entry(login, textvariable=self.phone_var, width=16).grid(row=0, column=3, sticky=tk.W, padx=4)
        ttk.Label(login, text="密码：").grid(row=0, column=4, sticky=tk.W)
        self.password_var = tk.StringVar()
        ttk.Entry(login, textvariable=self.password_var, width=16, show="*").grid(row=0, column=5, sticky=tk.W, padx=4)
        self.login_btn = ttk.Button(login, text="登录", command=self._login)
        self.login_btn.grid(row=0, column=6, padx=(8, 4))
        self.logout_btn = ttk.Button(login, text="退出登录", command=self._logout, state=tk.DISABLED)
        self.logout_btn.grid(row=0, column=7, padx=4)
        self.login_status_var = tk.StringVar(value="未登录")
        ttk.Label(login, textvariable=self.login_status_var, foreground="gray").grid(
            row=1, column=0, columnspan=8, sticky=tk.W, pady=(6, 0)
        )

        tasks = ttk.LabelFrame(parent, text="2. 我的任务", padding=8)
        tasks.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        ctrl = ttk.Frame(tasks)
        ctrl.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(ctrl, text="执行日期：").pack(side=tk.LEFT)
        self.execute_day_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.execute_day_var, width=14).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="页码：").pack(side=tk.LEFT, padx=(8, 0))
        self.page_var = tk.StringVar(value="1")
        ttk.Entry(ctrl, textvariable=self.page_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="每页：").pack(side=tk.LEFT, padx=(8, 0))
        self.page_size_var = tk.StringVar(value="10")
        ttk.Entry(ctrl, textvariable=self.page_size_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="获取任务", command=self._fetch_tasks).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Button(ctrl, text="选中当前任务", command=self._select_highlighted_task).pack(side=tk.LEFT, padx=4)

        columns = ("id", "status", "name")
        self.task_tree = ttk.Treeview(tasks, columns=columns, show="headings", height=8)
        for col, title, width in (("id", "执行 ID", 100), ("status", "状态", 100), ("name", "名称", 520)):
            self.task_tree.heading(col, text=title)
            self.task_tree.column(col, width=width, anchor=tk.W)
        self.task_tree.pack(fill=tk.BOTH, expand=True)
        self.selected_task_var = tk.StringVar(value="已选任务：无")
        ttk.Label(tasks, textvariable=self.selected_task_var, foreground="gray").pack(anchor=tk.W, pady=(6, 0))

    def _build_collection_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(top, text="打开 Rerun 可视化", command=self._run_visualizer).pack(side=tk.LEFT)

        pre = ttk.Frame(parent)
        pre.pack(fill=tk.X, pady=(0, 6))
        enc = ttk.LabelFrame(pre, text="串口、夹爪标定与 metadata.extra", padding=8)
        enc.pack(side=tk.LEFT, fill=tk.Y, anchor=tk.N, padx=(0, 8))
        enc_row = ttk.Frame(enc)
        enc_row.pack(fill=tk.X)
        self.encoder_port_vars: Dict[str, tk.StringVar] = {}
        self.encoder_combos: Dict[str, ttk.Combobox] = {}
        for key, label in (("left_port", "左夹爪"), ("right_port", "右夹爪")):
            ttk.Label(enc_row, text=f"{label}：").pack(side=tk.LEFT, padx=(8, 2))
            var = tk.StringVar()
            self.encoder_port_vars[key] = var
            combo = ttk.Combobox(enc_row, textvariable=var, width=10)
            combo.pack(side=tk.LEFT)
            self.encoder_combos[key] = combo
        ttk.Button(enc_row, text="刷新 COM", command=self._refresh_serial_ports).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Button(enc_row, text="保存到 YAML", command=self._save_encoder_ports).pack(side=tk.LEFT, padx=4)

        cal_row = ttk.Frame(enc)
        cal_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(cal_row, text="标定侧：").pack(side=tk.LEFT)
        self.calibration_side_var = tk.StringVar(value="left")
        ttk.Radiobutton(cal_row, text="左", variable=self.calibration_side_var, value="left").pack(side=tk.LEFT)
        ttk.Radiobutton(cal_row, text="右", variable=self.calibration_side_var, value="right").pack(side=tk.LEFT, padx=(4, 12))
        self.calibration_start_btn = ttk.Button(cal_row, text="开始标定", style="Start.TButton", command=self._start_calibration)
        self.calibration_start_btn.pack(side=tk.LEFT, padx=4)
        self.calibration_stop_btn = ttk.Button(
            cal_row, text="取消标定", style="Stop.TButton", command=self._stop_calibration, state=tk.DISABLED
        )
        self.calibration_stop_btn.pack(side=tk.LEFT, padx=4)
        self.calibration_status_var = tk.StringVar(value="标定前请先停止设备进程")
        ttk.Label(enc, textvariable=self.calibration_status_var, foreground="gray", wraplength=360).pack(
            anchor=tk.W, pady=(6, 0)
        )

        camera_meta = ttk.Frame(enc)
        camera_meta.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(camera_meta, text="device_type：").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        self.extra_device_type_var = tk.StringVar(
            value=self.controller.extra_device_type or DEFAULT_EXTRA_DEVICE_TYPE
        )
        ttk.Entry(camera_meta, textvariable=self.extra_device_type_var, width=22).grid(
            row=0, column=1, sticky=tk.W, padx=(4, 12), pady=(0, 4)
        )
        ttk.Label(camera_meta, text="hardware_string：").grid(row=0, column=2, sticky=tk.W, pady=(0, 4))
        self.extra_hardware_string_var = tk.StringVar(
            value=self.controller.extra_hardware_string or DEFAULT_EXTRA_HARDWARE_STRING
        )
        ttk.Entry(camera_meta, textvariable=self.extra_hardware_string_var, width=8).grid(
            row=0, column=3, sticky=tk.W, padx=(4, 0), pady=(0, 4)
        )
        ttk.Label(camera_meta, text="device id：").grid(row=1, column=0, sticky=tk.W)
        self.extra_device_zero_var = tk.StringVar(value=self.controller.extra_device_zero)
        ttk.Entry(camera_meta, textvariable=self.extra_device_zero_var, width=22).grid(
            row=1, column=1, sticky=tk.W, padx=(4, 0)
        )
        ttk.Label(
            camera_meta,
            text="以上字段只需填一个值，会自动复制到 camera_0..2",
            foreground="gray",
            wraplength=340,
        ).grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

        meta = ttk.LabelFrame(pre, text="DM3 元数据 / 输出", padding=8)
        meta.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.dm3_user_var = tk.StringVar(value="未登录")
        self.dm3_task_var = tk.StringVar(value="未选择任务")
        ttk.Label(meta, textvariable=self.dm3_user_var).pack(anchor=tk.W)
        ttk.Label(meta, textvariable=self.dm3_task_var, wraplength=360).pack(anchor=tk.W, pady=(2, 6))

        grid = ttk.Frame(meta)
        grid.pack(fill=tk.X)
        ttk.Label(grid, text="输出目录：").grid(row=0, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value=self.controller.output_dir)
        ttk.Entry(grid, textvariable=self.output_dir_var, width=42).grid(row=0, column=1, sticky=tk.W, padx=4)
        ttk.Button(grid, text="浏览", command=self._browse_output_dir).grid(row=0, column=2, padx=4)
        ttk.Label(grid, text="任务名称：").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self.task_name_var = tk.StringVar(value=self.controller.task_name)
        ttk.Entry(grid, textvariable=self.task_name_var, width=24).grid(row=1, column=1, sticky=tk.W, padx=4, pady=(4, 0))

        device = ttk.LabelFrame(parent, text="设备启动 (windows_device_start.py)", padding=8)
        device.pack(fill=tk.X, pady=(0, 6))
        device_main = ttk.Frame(device)
        device_main.pack(fill=tk.X)
        device_left = ttk.Frame(device_main)
        device_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        drow = ttk.Frame(device_left)
        drow.pack(fill=tk.X)
        ttk.Label(drow, text="配置：").pack(side=tk.LEFT)
        self.config_var = tk.StringVar(value=DEFAULT_CONFIG_NAME)
        self.config_combo = ttk.Combobox(
            drow,
            textvariable=self.config_var,
            values=scan_config_files(),
            state="readonly",
            width=34,
        )
        self.config_combo.pack(side=tk.LEFT, padx=4)
        self.config_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_config_changed())
        self.device_start_btn = ttk.Button(
            drow,
            text="启动设备",
            style="Start.TButton",
            command=self._start_device,
            state=tk.DISABLED,
        )
        self.device_start_btn.pack(side=tk.LEFT, padx=(12, 4))
        self.device_stop_btn = ttk.Button(
            drow, text="停止设备", style="Stop.TButton", command=self._stop_device, state=tk.DISABLED
        )
        self.device_stop_btn.pack(side=tk.LEFT, padx=4)

        cam_row = ttk.Frame(device_left)
        cam_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(cam_row, text="USB 索引：").pack(side=tk.LEFT)
        self.camera_vars: Dict[str, tk.StringVar] = {}
        vcmd = (self.root.register(self._validate_int), "%P")
        for key, label in _CAMERA_KEYS:
            ttk.Label(cam_row, text=label).pack(side=tk.LEFT, padx=(8, 2))
            var = tk.StringVar()
            self.camera_vars[key] = var
            ttk.Entry(
                cam_row,
                textvariable=var,
                width=4,
                validate="key",
                validatecommand=vcmd,
            ).pack(side=tk.LEFT)
        ttk.Button(
            cam_row, text="从配置加载", command=self._load_camera_indices_from_config, width=10
        ).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(
            cam_row, text="保存到 YAML", command=self._save_camera_indices_to_yaml, width=11
        ).pack(side=tk.LEFT, padx=4)

        saver = ttk.LabelFrame(parent, text="数据保存 (async_data_saver_NEDF3.py)", padding=8)
        saver.pack(fill=tk.X, pady=(0, 6))
        srow = ttk.Frame(saver)
        srow.pack(fill=tk.X)
        self.saver_start_btn = ttk.Button(
            srow,
            text="启动保存",
            style="Start.TButton",
            command=self._start_saver,
            state=tk.DISABLED,
        )
        self.saver_start_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.saver_stop_btn = ttk.Button(
            srow, text="停止保存", style="Stop.TButton", command=self._stop_saver, state=tk.DISABLED
        )
        self.saver_stop_btn.pack(side=tk.LEFT, padx=4)
        self.saver_status_var = tk.StringVar(value="保存：停止 | 待机")
        self.saver_enter_action_var = tk.StringVar(value="等待启动保存")
        ttk.Label(srow, textvariable=self.saver_status_var, foreground="#374151").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(srow, text=" | ").pack(side=tk.LEFT)
        self.saver_enter_action_label = ttk.Label(
            srow,
            textvariable=self.saver_enter_action_var,
            foreground="#c42b2b",
        )
        self.saver_enter_action_label.pack(side=tk.LEFT)
        self.saved_count_var = tk.StringVar(value="已保存 episode: 0")
        ttk.Label(srow, textvariable=self.saved_count_var).pack(side=tk.RIGHT)
        stats_row = ttk.Frame(saver)
        stats_row.pack(fill=tk.X, pady=(6, 0))
        self.total_duration_var = tk.StringVar(value="今日总采集时间：统计中...")
        ttk.Label(stats_row, textvariable=self.total_duration_var).pack(side=tk.LEFT)
        self.effective_duration_var = tk.StringVar(value="今日有效数据时间：统计中...")
        ttk.Label(stats_row, textvariable=self.effective_duration_var).pack(
            side=tk.LEFT, padx=(18, 0)
        )

        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        self.device_log = self._make_log_widget(paned, "设备输出")
        paned.add(self.device_log["frame"], weight=1)
        self.saver_log = self._make_log_widget(paned, "保存输出")
        paned.add(self.saver_log["frame"], weight=1)
        self._update_collection_gate()

    def _update_collection_gate(self) -> None:
        device = self.controller.processes.device
        saver = self.controller.processes.saver

        if device.running or device.process is not None or device.stopping:
            self.device_start_btn.configure(state=tk.DISABLED)
        else:
            self.device_start_btn.configure(state=tk.NORMAL)

        if saver.running or saver.process is not None or saver.stopping:
            self.saver_start_btn.configure(state=tk.DISABLED)
        else:
            self.saver_start_btn.configure(state=tk.NORMAL)
        if self.diagnostic_panel is not None:
            self.diagnostic_panel.update_gate()
        if self.tracker_panel is not None:
            self.tracker_panel.update_gate()

    def _make_log_widget(self, parent: ttk.PanedWindow, title: str) -> Dict[str, Any]:
        frame = ttk.LabelFrame(parent, text=title, padding=4)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            state=tk.DISABLED,
        )
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)
        _configure_log_text_tags(text)
        return {"frame": frame, "text": text}

    def _on_config_changed(self) -> None:
        self.controller.config_name = self.config_var.get().strip() or DEFAULT_CONFIG_NAME
        self.controller.reload_config_defaults()
        self._reload_config_fields()

    def _reload_config_fields(self) -> None:
        for key, var in self.camera_vars.items():
            var.set(self.controller.camera_indices.get(key, ""))
        for key, var in self.encoder_port_vars.items():
            var.set(self.controller.encoder_ports.get(key, ""))

    @staticmethod
    def _validate_int(value: str) -> bool:
        return value == "" or value.isdigit()

    def _load_camera_indices_from_config(self) -> None:
        self.controller.config_name = self.config_var.get().strip() or DEFAULT_CONFIG_NAME
        self.controller.reload_config_defaults()
        self._reload_config_fields()

    def _save_camera_indices_to_yaml(self) -> None:
        config_name = self.config_var.get().strip()
        if not config_name:
            self._show_warning("警告", "请选择配置文件。")
            return

        indices: Dict[str, str] = {}
        for key, label in _CAMERA_KEYS:
            value = self.camera_vars[key].get().strip()
            if not value:
                self._show_warning("警告", f"请填写 {label} 索引。")
                return
            indices[key] = value

        if not messagebox.askyesno(
            "确认保存",
            f"将 USB 索引写入 configs/{config_name}.yaml？\n\n"
            f"右腕={indices['right_capture_id']}, "
            f"左腕={indices['left_capture_id']}, "
            f"头戴={indices['ego_capture_id']}",
        ):
            return

        try:
            self.controller.config_name = config_name
            self.controller.camera_indices = indices
            self.controller.save_camera_indices_to_config()
        except (OSError, KeyError, ValueError) as exc:
            self._show_error(exc, title="保存失败")
            return

        messagebox.showinfo("已保存", f"已更新 configs/{config_name}.yaml")
        self._log(f"已保存 USB 索引到 configs/{config_name}.yaml\n")

    def _sync_controller_from_ui(self) -> None:
        self.controller.config_name = self.config_var.get().strip() or DEFAULT_CONFIG_NAME
        output_dir = self.output_dir_var.get().strip()
        output_dir_changed = output_dir != self.controller.output_dir
        self.controller.output_dir = output_dir
        self.controller.task_name = self.task_name_var.get().strip()
        self.controller.extra_device_zero = self.extra_device_zero_var.get().strip()
        self.controller.extra_device_type = self.extra_device_type_var.get().strip()
        self.controller.extra_hardware_string = self.extra_hardware_string_var.get().strip()
        self.controller.camera_indices = {k: v.get().strip() for k, v in self.camera_vars.items()}
        self.controller.encoder_ports = {k: v.get().strip() for k, v in self.encoder_port_vars.items()}
        self._sync_upload_data_dir()
        if output_dir_changed:
            self._request_daily_stats_refresh()

    def _sync_upload_data_dir(self) -> None:
        if self.upload_panel is not None:
            self.upload_panel.set_data_dir(self._upload_parent_dir())

    def _upload_parent_dir(self) -> Path:
        output_dir = self.output_dir_var.get().strip() or self.controller.output_dir
        return Path(output_dir).expanduser().resolve().parent

    def _refresh_session_display(self) -> None:
        state = self.controller.session.public_state()
        if state["logged_in"]:
            self.login_status_var.set(f"已登录：{state['user_name']} ({state['user_id']})")
            self.login_btn.configure(state=tk.DISABLED)
            self.logout_btn.configure(state=tk.NORMAL)
            self.dm3_user_var.set(f"用户：{state['user_name']} ({state['user_id']})")
        else:
            self.login_status_var.set("未登录")
            self.login_btn.configure(state=tk.NORMAL)
            self.logout_btn.configure(state=tk.DISABLED)
            self.dm3_user_var.set("用户：未登录")
        if state["task_selected"]:
            self.selected_task_var.set(f"已选任务：{state['task_execution_id']} / {state['task_name']}")
            desc = state.get("task_description") or ""
            self.dm3_task_var.set(f"任务：{state['task_name']} | {desc}")
        else:
            self.selected_task_var.set("已选任务：无")
            self.dm3_task_var.set("任务：未选择")
        self._update_login_tab_access(bool(state["logged_in"]))

    def _update_login_tab_access(self, logged_in: bool) -> None:
        if self.notebook is None or self.collect_tab is None or self.upload_tab is None:
            return
        state = tk.NORMAL if logged_in else tk.DISABLED
        self.notebook.tab(self.collect_tab, state=state)
        self.notebook.tab(self.upload_tab, state=state)
        if not logged_in and self.dm3_tab is not None:
            self.notebook.select(self.dm3_tab)

    def _login(self) -> None:
        phone = self.phone_var.get().strip()
        password = self.password_var.get()
        base_url = self.base_url_var.get().strip() or DEFAULT_BASE_URL
        if not phone or not password:
            self._show_warning("缺少输入", "请输入手机号和密码。")
            return

        def work() -> Dict[str, Any]:
            return self.controller.session.login(phone, password, base_url)

        def on_success(_result: Dict[str, Any]) -> None:
            self._refresh_session_display()
            if self.upload_panel is not None:
                self.upload_panel.request_auto_start()
            self._log("DM3 登录成功，已启动自动上传\n")

        self._run_background(work, on_success)

    def _logout(self) -> None:
        def work() -> None:
            self.controller.shutdown_upload_manager()
            self.controller.session.logout()

        def on_success(_result: None) -> None:
            if self.upload_panel is not None:
                self.upload_panel.manager = None
                self.upload_panel._auto_start_requested = True
            self._refresh_session_display()
            self._log("已退出 DM3 登录，自动上传已停止\n")

        self._run_background(work, on_success)

    def _fetch_tasks(self) -> None:
        page = int(self.page_var.get().strip() or "1")
        page_size = int(self.page_size_var.get().strip() or "10")
        execute_day = self.execute_day_var.get().strip() or None

        def work() -> List[DM3TaskExecution]:
            return self.controller.session.fetch_tasks(page=page, page_size=page_size, execute_day=execute_day)

        self._run_background(work, self._on_tasks_loaded)

    def _on_tasks_loaded(self, tasks: List[DM3TaskExecution]) -> None:
        self.current_tasks = tasks
        self.task_tree.delete(*self.task_tree.get_children())
        for idx, task in enumerate(tasks):
            self.task_tree.insert("", tk.END, iid=str(idx), values=(task.id or "", task.status or "-", task.name))
        self._log(f"已加载 {len(tasks)} 个任务\n")

    def _select_highlighted_task(self) -> None:
        selection = self.task_tree.selection()
        if not selection:
            self._show_warning("未选择任务", "请先选择一条任务。")
            return
        idx = int(selection[0])

        def work() -> Dict[str, Any]:
            return self.controller.session.select_task_by_index(idx)

        def on_success(_state: Dict[str, Any]) -> None:
            self.selected_task = self.current_tasks[idx]
            self.controller.apply_selected_task_defaults()
            self.output_dir_var.set(self.controller.output_dir)
            self.task_name_var.set(self.controller.task_name)
            self._sync_upload_data_dir()
            self._request_daily_stats_refresh()
            self._refresh_session_display()
            desc = extract_task_description(self.selected_task.raw)
            self._log(f"已选任务：{self.selected_task.id} / {self.selected_task.name}\n")
            if desc:
                self._log(f"任务描述：{desc}\n")

        self._run_background(work, on_success)

    def _refresh_serial_ports(self) -> None:
        ports = [entry["device"] for entry in list_serial_ports()]
        for combo in self.encoder_combos.values():
            combo["values"] = ports

    def _save_encoder_ports(self) -> None:
        try:
            self._sync_controller_from_ui()
            self.controller.save_encoder_ports_to_config()
            self._log("已保存编码器端口到 YAML\n")
        except Exception as exc:
            self._show_error(exc)

    def _start_calibration(self) -> None:
        if self.controller.processes.device.running:
            self._show_warning("标定", "请先停止设备进程以独占串口。")
            return
        self._sync_controller_from_ui()
        side = self.calibration_side_var.get().strip().lower()
        self.calibration_start_btn.configure(state=tk.DISABLED)
        self.calibration_stop_btn.configure(state=tk.DISABLED)
        self.calibration_status_var.set("正在连接编码器...")

        def work() -> Dict[str, Any]:
            return self.controller.calibration.start(
                config_name=self.controller.config_name,
                side=side,
                encoder_ports=self.controller.encoder_ports,
            )

        def on_success(status: Dict[str, Any]) -> None:
            self.calibration_stop_btn.configure(state=tk.NORMAL)
            self.calibration_status_var.set(str(status.get("message", "")))
            self._log(f"开始 {side} 夹爪标定；Enter 开始/停止录制\n")

        def finally_callback() -> None:
            if not self.controller.calibration.status_dict().get("active"):
                self.calibration_start_btn.configure(state=tk.NORMAL)
                self.calibration_stop_btn.configure(state=tk.DISABLED)

        self._run_background(work, on_success, finally_callback=finally_callback)

    def _stop_calibration(self) -> None:
        status = self.controller.calibration.stop()
        self.calibration_start_btn.configure(state=tk.NORMAL)
        self.calibration_stop_btn.configure(state=tk.DISABLED)
        self.calibration_status_var.set(str(status.get("message", "")))

    def _toggle_calibration_recording(self) -> None:
        if not self.controller.calibration.status_dict().get("active"):
            return
        try:
            status = self.controller.calibration.toggle_recording()
            self.calibration_status_var.set(str(status.get("message", "")))
        except RuntimeError as exc:
            self._show_error(exc, title="标定")

    def _on_calibration_save_complete(self, success: bool, message: str) -> None:
        def show() -> None:
            self.calibration_status_var.set(message.replace("\n", " "))
            self.calibration_start_btn.configure(state=tk.NORMAL)
            self.calibration_stop_btn.configure(state=tk.DISABLED)
            if success:
                self._gripper_calibrations.clear()
                self._gripper_calibration_errors.clear()
                self._gripper_next_calibration_attempt.clear()
                messagebox.showinfo("标定完成", message)
                self._log(f"标定完成：{message.replace(chr(10), ' ')}\n")
            else:
                self._show_error_message("标定失败", message)
                self._log(f"标定失败：{message}\n", level="error")

        self.root.after(0, show)

    def _start_device(self) -> None:
        try:
            self._gripper_monitor_enabled = False
            self._release_gripper_summary_reader("等待设备启动")
            self._sync_controller_from_ui()
            self.controller.start_device()
            self._gripper_monitor_enabled = True
            self.device_start_btn.configure(state=tk.DISABLED)
            self.device_stop_btn.configure(state=tk.NORMAL)
            self._log("设备进程已启动\n")
        except Exception as exc:
            self._show_error(exc)

    def _stop_device(self) -> None:
        self._gripper_monitor_enabled = False
        self._release_gripper_summary_reader("设备已停止，等待重新启动")
        self.controller.processes.stop_device_async()
        self.device_stop_btn.configure(state=tk.DISABLED)
        self._update_collection_gate()
        self._log("正在停止设备进程...\n")

    def _start_saver(self) -> None:
        try:
            self._pending_start_enter_at = None
            self._sync_controller_from_ui()
            self.controller.start_saver()
            self.saver_start_btn.configure(state=tk.DISABLED)
            self.saver_stop_btn.configure(state=tk.NORMAL)
            self._log("数据保存进程已启动；Enter 切换 episode\n")
        except Exception as exc:
            self._show_error(exc)

    def _stop_saver(self) -> None:
        self._pending_start_enter_at = None
        self.controller.processes.stop_saver_async()
        self.saver_stop_btn.configure(state=tk.DISABLED)
        self._update_collection_gate()
        self._log("正在停止数据保存进程...\n")

    def _run_visualizer(self) -> None:
        try:
            self.controller.processes.run_visualizer()
            self._log("已启动 Rerun 可视化\n")
        except Exception as exc:
            self._show_error(exc)

    def _on_global_enter(self, _event: tk.Event) -> None:
        if self.controller.calibration.status_dict().get("active"):
            self._pending_start_enter_at = None
            self._toggle_calibration_recording()
            return
        if not self.controller.processes.saver.running:
            return
        now = time.monotonic()
        if now - self._last_enter_event_at < ENTER_DEBOUNCE_SEC:
            return
        self._last_enter_event_at = now
        saver = self.controller.processes.saver
        if not saver.recording and not saver.ready_for_episode:
            if saver.finalizing_episode or self._upload_maintenance_waiting:
                self._pending_start_enter_at = now
                self._log("[control] Start request queued until episode finalization completes.\n")
                return
            self._log("正在保存当前 episode，暂不能开始下一段录制\n")
            return
        self._pending_start_enter_at = None
        try:
            self.controller.processes.send_enter_to_saver()
        except RuntimeError as exc:
            self._log(f"{exc}\n")

    def _dispatch_pending_start_enter(self) -> None:
        pending_at = self._pending_start_enter_at
        if pending_at is None:
            return
        if self.controller.calibration.status_dict().get("active"):
            self._pending_start_enter_at = None
            return
        saver = self.controller.processes.saver
        if not saver.running:
            self._pending_start_enter_at = None
            return
        age = time.monotonic() - pending_at
        if age > PENDING_START_ENTER_TTL_SEC:
            self._pending_start_enter_at = None
            self._log("[control] Queued start request expired.\n")
            return
        if saver.recording:
            self._pending_start_enter_at = None
            return
        if not saver.ready_for_episode:
            return
        self._pending_start_enter_at = None
        try:
            self.controller.processes.send_enter_to_saver()
        except RuntimeError as exc:
            self._log(f"{exc}\n")

    def _browse_output_dir(self) -> None:
        initial = self.output_dir_var.get().strip() or str(DEFAULT_DATA_ROOT)
        chosen = filedialog.askdirectory(initialdir=initial)
        if chosen:
            self.output_dir_var.set(chosen)
            self.controller.output_dir = chosen
            self._sync_upload_data_dir()
            self._request_daily_stats_refresh()

    def _schedule_ui_flush(self) -> None:
        self._flush_console_logs()
        self._poll_process_logs()
        self._update_runtime_status()
        self.root.after(UI_FLUSH_INTERVAL_MS, self._schedule_ui_flush)

    def _request_daily_stats_refresh(self) -> None:
        """Refresh daily duration off the Tk thread, coalescing repeated requests."""
        if self._daily_stats_refreshing:
            self._daily_stats_refresh_pending = True
            return
        stats_root = self._daily_stats_root()
        self._daily_stats_refreshing = True
        self.total_duration_var.set("今日总采集时间：统计中...")
        self.effective_duration_var.set("今日有效数据时间：统计中...")

        def runner() -> None:
            stats = scan_today_episode_stats(stats_root, backfill_missing=True)
            self.root.after(
                0,
                lambda: self._finish_daily_stats_refresh(stats_root, stats),
            )

        threading.Thread(target=runner, name="dm3-daily-duration", daemon=True).start()

    def _finish_daily_stats_refresh(
        self,
        stats_root: str,
        stats: DailyEpisodeStats,
    ) -> None:
        self._daily_stats_refreshing = False
        current_stats_root = self._daily_stats_root()
        if current_stats_root == stats_root:
            total_suffix = f"（{stats.total_episode_count} episodes"
            if stats.anomaly_error_count:
                total_suffix += f"，异常 {stats.anomaly_error_count}"
            if stats.manual_discard_count:
                total_suffix += f"，手动丢弃 {stats.manual_discard_count}"
            total_suffix += "）"
            if stats.unreadable_count:
                total_suffix += f"，{stats.unreadable_count} 个无法读取"
            self.total_duration_var.set(
                "今日总采集时间："
                f"{format_duration(stats.total_duration_seconds)} {total_suffix}"
            )
            self.effective_duration_var.set(
                "今日有效数据时间："
                f"{format_duration(stats.duration_seconds)} "
                f"（{stats.episode_count} episodes）"
            )
        if self._daily_stats_refresh_pending or current_stats_root != stats_root:
            self._daily_stats_refresh_pending = False
            self._request_daily_stats_refresh()

    def _daily_stats_root(self) -> str:
        output_dir = Path(
            self.output_dir_var.get().strip() or self.controller.output_dir
        ).expanduser().resolve()
        default_root = Path(DEFAULT_DATA_ROOT).expanduser().resolve()
        if output_dir == default_root or default_root in output_dir.parents:
            return str(default_root)
        return str(output_dir.parent)

    def _start_gripper_monitor(self) -> None:
        if self._gripper_thread is not None and self._gripper_thread.is_alive():
            return
        self._gripper_stop_event.clear()
        self._gripper_thread = threading.Thread(
            target=self._gripper_poll_loop,
            name="dm3-gripper-shm-monitor",
            daemon=True,
        )
        self._gripper_thread.start()

    def _gripper_poll_loop(self) -> None:
        while not self._gripper_stop_event.is_set():
            if not self._gripper_monitor_enabled:
                self._gripper_stop_event.wait(GRIPPER_POLL_INTERVAL_SEC)
                continue
            with self._gripper_reader_lock:
                raw_values, status = self._gripper_reader.read_latest()
            values: Dict[str, float] = {}
            for side, raw_value in raw_values.items():
                processed = self._process_gripper_sample(side, raw_value)
                if processed is not None:
                    values[side] = processed
            if raw_values and not values:
                error = next(iter(self._gripper_calibration_errors.values()), "")
                status = f"等待今日夹爪标定：{error}" if error else "等待今日夹爪标定"
            now = time.monotonic()
            with self._gripper_lock:
                self._gripper_status_text = status
                for side, value in values.items():
                    self._gripper_latest_values[side] = value
                    self._gripper_samples[side].append((now, value))
            self._gripper_stop_event.wait(GRIPPER_POLL_INTERVAL_SEC)

    def _process_gripper_sample(self, side: str, raw_angle: float) -> Optional[float]:
        calibration = self._load_gripper_calibration(side)
        if calibration is None:
            return None
        expected_date = self._gripper_calibration_date or date.today().isoformat()
        try:
            return process_gripper_angle_with_date_check(
                raw_angle,
                calibration,
                expected_date,
            )
        except (ValueError, RuntimeError) as exc:
            self._gripper_calibration_errors[side] = str(exc)
            return None

    def _load_gripper_calibration(self, side: str) -> Optional[GripperAngleCalibration]:
        today = date.today().isoformat()
        if self._gripper_calibration_date != today:
            self._gripper_calibrations.clear()
            self._gripper_calibration_errors.clear()
            self._gripper_next_calibration_attempt.clear()
            self._gripper_calibration_date = today
        if side in self._gripper_calibrations:
            return self._gripper_calibrations[side]
        now = time.monotonic()
        if now < self._gripper_next_calibration_attempt.get(side, 0.0):
            return None
        try:
            calibration = load_today_angle_calibration(str(PROJECT_ROOT), side, expected_date=today)
        except (FileNotFoundError, ValueError) as exc:
            self._gripper_calibration_errors[side] = str(exc)
            self._gripper_next_calibration_attempt[side] = now + 2.0
            return None
        self._gripper_calibrations[side] = calibration
        self._gripper_calibration_errors.pop(side, None)
        self._gripper_next_calibration_attempt.pop(side, None)
        return calibration

    def _release_gripper_summary_reader(self, status: str = "等待 device_summary_data") -> None:
        with self._gripper_reader_lock:
            self._gripper_reader.close()
        with self._gripper_lock:
            self._gripper_latest_values.clear()
            for samples in self._gripper_samples.values():
                samples.clear()
            self._gripper_status_text = status

    def _schedule_gripper_plot_refresh(self) -> None:
        if self._gripper_stop_event.is_set():
            return
        self._draw_gripper_plot()
        self.root.after(GRIPPER_RENDER_INTERVAL_MS, self._schedule_gripper_plot_refresh)

    def _draw_gripper_plot(self) -> None:
        canvas = self.gripper_canvas
        if canvas is None:
            return
        with self._gripper_lock:
            samples = {
                side: list(values)
                for side, values in self._gripper_samples.items()
            }
            latest = dict(self._gripper_latest_values)
            status = self._gripper_status_text

        self.gripper_status_var.set(f"夹爪曲线：{status}")
        self.gripper_value_vars["left"].set(
            f"左夹爪: {latest['left']:.2f} deg" if "left" in latest else "左夹爪: --"
        )
        self.gripper_value_vars["right"].set(
            f"右夹爪: {latest['right']:.2f} deg" if "right" in latest else "右夹爪: --"
        )

        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")
        if width < 80 or height < 80:
            return

        now = time.monotonic()
        plot_left = 42
        plot_right = width - 10
        plot_top = 18
        plot_bottom = height - 24
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
        colors = {"left": "#22c55e", "right": "#60a5fa"}
        labels = {"left": "Left", "right": "Right"}
        for side in ("left", "right"):
            self._draw_gripper_side_plot(
                canvas,
                rects[side],
                samples[side],
                now,
                labels[side],
                colors[side],
            )

    def _draw_gripper_side_plot(
        self,
        canvas: tk.Canvas,
        rect: Tuple[int, int, int, int],
        samples: List[Tuple[float, float]],
        now: float,
        label: str,
        color: str,
    ) -> None:
        x0, y0, x1, y1 = rect
        canvas.create_rectangle(x0, y0, x1, y1, outline="#374151", fill="#0b1220")
        canvas.create_text(
            x0 + 5,
            y0 + 4,
            text=label,
            anchor=tk.NW,
            fill=color,
            font=("Consolas", 8, "bold"),
        )
        recent = [(t, v) for t, v in samples if now - t <= GRIPPER_HISTORY_SEC]
        if not recent:
            canvas.create_text(
                (x0 + x1) // 2,
                (y0 + y1) // 2,
                text="waiting for data",
                fill="#6b7280",
                font=("Consolas", 9),
            )
            return

        values = [value for _timestamp, value in recent]
        min_value = min(values)
        max_value = max(values)
        if max_value - min_value < 1e-6:
            min_value -= 1.0
            max_value += 1.0
        else:
            pad = (max_value - min_value) * 0.12
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
        canvas.create_text(
            x1,
            y1 + 4,
            text=f"{GRIPPER_HISTORY_SEC:.0f}s",
            anchor=tk.NE,
            fill="#6b7280",
            font=("Consolas", 7),
        )

        points: List[float] = []
        value_range = max_value - min_value
        plot_width = max(1, x1 - x0)
        plot_height = max(1, y1 - y0)
        for timestamp, value in recent:
            age = max(0.0, min(GRIPPER_HISTORY_SEC, now - timestamp))
            x = x1 - (age / GRIPPER_HISTORY_SEC) * plot_width
            y = y1 - ((value - min_value) / value_range) * plot_height
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

    def _poll_process_logs(self) -> None:
        mapping = {
            "device": (self.controller.processes.device, self.device_log["text"]),
            "saver": (self.controller.processes.saver, self.saver_log["text"]),
        }
        for name, (managed, widget) in mapping.items():
            offset = self._log_offsets[name]
            lines, next_offset = managed.read_logs_since(offset, limit=MAX_PROCESS_LOG_LINES_PER_FLUSH)
            self._log_offsets[name] = next_offset
            if not lines:
                continue
            if name == "saver":
                self._handle_saver_upload_maintenance_logs(lines)
            if len(lines) > MAX_PROCESS_LOG_LINES_PER_FLUSH:
                lines = [
                    f"[skipped {len(lines) - MAX_PROCESS_LOG_LINES_PER_FLUSH} older log lines to keep UI responsive]\n",
                    *lines[-MAX_PROCESS_LOG_LINES_PER_FLUSH:],
                ]
            widget.configure(state=tk.NORMAL)
            for line in lines:
                plain = strip_ansi_for_display(line)
                tag = f"log_{_classify_process_log_line(plain)}"
                widget.insert(tk.END, plain, tag)
            widget.see(tk.END)
            widget.configure(state=tk.DISABLED)

    def _handle_saver_upload_maintenance_logs(self, lines: List[str]) -> None:
        for line in lines:
            plain = strip_ansi_for_display(line)
            if "Waiting for upload maintenance full scan before next episode" in plain:
                self._upload_maintenance_waiting = True
                if self.upload_panel is not None:
                    self.upload_panel.handle_maintenance_request_once()
                return
            if "Upload maintenance full scan completed" in plain:
                self._upload_maintenance_waiting = False
            elif "Upload maintenance full scan did not complete before timeout" in plain:
                self._upload_maintenance_waiting = False

    def _update_recording_indicator(self, recording: bool) -> None:
        if recording:
            self._show_recording_overlay()
        else:
            self._hide_recording_overlay()

    def _show_recording_overlay(self) -> None:
        if (
            self._recording_overlay is not None
            and self._recording_overlay.winfo_exists()
        ):
            self._position_recording_overlay()
            return
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.configure(bg="#22c55e")
        self._recording_overlay = overlay
        self._recording_overlay_visible = True

        label = tk.Label(
            overlay,
            text="采集中",
            bg="#22c55e",
            fg="#052e16",
            font=("Sans", 72, "bold"),
            padx=84,
            pady=48,
        )
        label.pack(fill=tk.BOTH, expand=True)
        overlay.bind("<Button-1>", lambda _event: self._position_recording_overlay())
        self._position_recording_overlay()

    def _position_recording_overlay(self) -> None:
        overlay = self._recording_overlay
        if overlay is None or not overlay.winfo_exists():
            return
        self.root.update_idletasks()
        overlay.update_idletasks()
        width = overlay.winfo_reqwidth()
        height = overlay.winfo_reqheight()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = max(1, self.root.winfo_width())
        root_height = max(1, self.root.winfo_height())
        x = root_x + (root_width - width) // 2
        y = root_y + (root_height - height) // 2
        overlay.geometry(f"{width}x{height}+{x}+{y}")

    def _hide_recording_overlay(self) -> None:
        overlay = self._recording_overlay
        self._recording_overlay = None
        self._recording_overlay_visible = False
        if overlay is not None and overlay.winfo_exists():
            overlay.destroy()

    def _update_runtime_status(self) -> None:
        device = self.controller.processes.device
        saver = self.controller.processes.saver
        self._dispatch_pending_start_enter()
        self._update_collection_gate()
        if device.stopping:
            self.device_stop_btn.configure(state=tk.DISABLED)
        elif device.running or device.process is not None:
            self.device_stop_btn.configure(state=tk.NORMAL)
        else:
            self.device_stop_btn.configure(state=tk.DISABLED)
        if saver.stopping:
            self.saver_stop_btn.configure(state=tk.DISABLED)
        elif saver.running or saver.process is not None:
            self.saver_stop_btn.configure(state=tk.NORMAL)
        else:
            self.saver_stop_btn.configure(state=tk.DISABLED)

        if device.running or device.process is not None:
            self._gripper_monitor_enabled = not device.stopping
        else:
            self._gripper_monitor_enabled = False
            self._release_gripper_summary_reader("设备未运行，夹爪曲线暂停")

        recording = "采集中" if saver.recording else "待机"
        self._update_recording_indicator(saver.recording)
        self.saver_status_var.set(f"保存：{'运行' if saver.running else '停止'} | {recording}")
        if not saver.running:
            self.saver_enter_action_var.set("等待启动保存")
            self.saver_enter_action_label.configure(foreground="#c42b2b")
        elif saver.recording:
            self.saver_enter_action_var.set("按 Enter 停止并保存")
            self.saver_enter_action_label.configure(foreground="#c42b2b")
        elif saver.ready_for_episode:
            self._upload_maintenance_waiting = False
            self.saver_enter_action_var.set("允许Enter继续录制")
            self.saver_enter_action_label.configure(foreground="#2d8c3c")
        elif self._upload_maintenance_waiting:
            self.saver_enter_action_var.set("等待上传全盘扫描和UI刷新")
            self.saver_enter_action_label.configure(foreground="#c42b2b")
        else:
            self.saver_enter_action_var.set("等待保存当前episode")
            self.saver_enter_action_label.configure(foreground="#c42b2b")
        self.saved_count_var.set(f"已保存 episode: {saver.saved_episode_count}")
        if saver.episode_stats_revision != self._daily_stats_revision:
            self._daily_stats_revision = saver.episode_stats_revision
            self._request_daily_stats_refresh()

        cal = self.controller.calibration.status_dict()
        if cal.get("active"):
            self.calibration_start_btn.configure(state=tk.DISABLED)
            self.calibration_stop_btn.configure(state=tk.NORMAL)
            if cal.get("message"):
                self.calibration_status_var.set(str(cal["message"]))
        elif not cal.get("saving"):
            self.calibration_start_btn.configure(state=tk.NORMAL)
            self.calibration_stop_btn.configure(state=tk.DISABLED)

    def _run_background(
        self,
        work: Callable[[], Any],
        on_success: Callable[[Any], None],
        *,
        finally_callback: Optional[Callable[[], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        def runner() -> None:
            try:
                result = work()
            except Exception as exc:
                logger.error(f"Vive_DM3 GUI operation failed: {exc}")
                self.root.after(0, lambda error=exc: self._show_error(error, error_callback=error_callback))
            else:
                self.root.after(0, lambda value=result: on_success(value))
            finally:
                if finally_callback is not None:
                    self.root.after(0, finally_callback)

        threading.Thread(target=runner, daemon=True).start()

    def _show_warning(self, title: str, message: str, *, parent: Optional[tk.Misc] = None) -> None:
        self._log(f"警告：{message}\n", level="warning")
        _show_colored_message(parent or self.root, title, message, color=_WARNING_COLOR)

    def _show_error_message(self, title: str, message: str) -> None:
        self._log(f"错误：{message}\n", level="error")
        _show_colored_message(self.root, title, message, color=_ERROR_COLOR)

    def _show_error(
        self,
        exc: Exception,
        *,
        title: str = "错误",
        error_callback: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        message = str(exc)
        self._log(f"错误：{message}\n", level="error")
        if error_callback is not None:
            error_callback(exc)
        _show_colored_message(self.root, title, message, color=_ERROR_COLOR)

    def _log(self, message: str, *, level: str = "info") -> None:
        self.log_queue.put((message, level))

    def _log_json(self, data: Dict[str, Any]) -> None:
        self._log(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def _flush_console_logs(self) -> None:
        lines: List[tuple[str, str]] = []
        while True:
            try:
                lines.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if not lines:
            return
        if len(lines) > MAX_CONSOLE_LOG_LINES_PER_FLUSH:
            lines = [
                (
                    f"[skipped {len(lines) - MAX_CONSOLE_LOG_LINES_PER_FLUSH} older UI log messages to keep UI responsive]\n",
                    "warning",
                ),
                *lines[-MAX_CONSOLE_LOG_LINES_PER_FLUSH:],
            ]
        self.console_log.configure(state=tk.NORMAL)
        for message, level in lines:
            tag = "log_info"
            if level == "warning":
                tag = "log_warning"
            elif level == "error":
                tag = "log_error"
            self.console_log.insert(tk.END, message, tag)
        self.console_log.see(tk.END)
        self.console_log.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        if self.controller.processes.saver.running or self.controller.processes.device.running:
            if not messagebox.askyesno("退出", "采集进程仍在运行，确定退出并停止？"):
                return
        self._hide_recording_overlay()
        if self.tracker_panel is not None:
            self.tracker_panel.stop_tracker_diagnostic()
        self._gripper_stop_event.set()
        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=1.0)
        self._release_gripper_summary_reader("GUI 已关闭")
        try:
            if self.upload_panel is not None:
                self.upload_panel.shutdown(stop_manager=False)
            self.controller.shutdown()
        except Exception as exc:
            logger.warning(f"Shutdown error: {exc}")
        self.root.destroy()


def _disable_windows_console_quick_edit() -> None:
    """Prevent CMD QuickEdit from freezing the GUI when the console is clicked."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)
        if handle in (0, -1):
            return
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        enable_quick_edit = 0x0040
        enable_insert_mode = 0x0020
        new_mode = mode.value & ~enable_quick_edit & ~enable_insert_mode
        kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass


def main() -> None:
    """Launch the combined Vive DM3 GUI."""
    _disable_windows_console_quick_edit()
    policy = apply_upload_process_policy()
    if not policy.affinity:
        raise RuntimeError(f"Upload CPU isolation failed; refusing to start GUI: {policy.errors}")
    if policy.ok:
        logger.info(
            f"Applied upload GUI process policy: pid={policy.pid}, "
            f"affinity={policy.affinity}, process_priority={policy.priority}, "
            f"memory_limit_mb={policy.memory_limit_mb}; "
            f"upload worker threads use SI_UPLOAD_THREAD_PRIORITY "
            f"(legacy SI_UPLOAD_PRIORITY)"
        )
    else:
        logger.warning(
            f"Upload GUI process policy partially applied: pid={policy.pid}, "
            f"affinity={policy.affinity}, process_priority={policy.priority}, "
            f"errors={policy.errors}"
        )
    ViveDM3GUI().run()


if __name__ == "__main__":
    main()
