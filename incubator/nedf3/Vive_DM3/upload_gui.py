#!/usr/bin/env python3
"""
Unified Vive/DM3 episode upload console.

This GUI owns upload controls: automatic protected upload, end-of-day force
upload, failed-item retry, and a minimal queue view. Detailed progress lives
on the performance monitor web page.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from incubator.nedf3.Vive_DM3.resource_policy import apply_upload_process_policy
from incubator.nedf3.Vive_DM3.upload_manager import (
    AutoUploadManager,
    DEFAULT_KEEP_LATEST_COUNT,
    DEFAULT_MULTIPART_THREADS,
    DEFAULT_PROTECT_MINUTES,
    DEFAULT_UPLOAD_WORKERS,
    UPLOAD_BACKEND_OSS,
)
from incubator.nedf3.windows_DM3.dm3_api import DEFAULT_BASE_URL

if TYPE_CHECKING:
    from incubator.nedf3.Vive_DM3.controller import ViveDM3Controller


MAINTENANCE_REQUEST_NAME = ".auto_upload_maintenance_request.json"
MAINTENANCE_DONE_NAME = ".auto_upload_maintenance_done.json"
DEFAULT_PERF_MONITOR_PORT = 18765
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


class UploadConsoleFrame(ttk.Frame):
    """Embeddable upload console used by both the main GUI and upload-only window."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        controller: Optional[Any] = None,
        default_data_dir: Optional[Union[Path, str]] = None,
        show_login_controls: bool = True,
    ) -> None:
        super().__init__(parent, padding=8)

        if controller is None:
            from incubator.nedf3.Vive_DM3.controller import ViveDM3Controller

            controller = ViveDM3Controller()
        self.controller = controller
        self._owns_controller = controller is None
        self._show_login_controls = show_login_controls
        self.manager: Optional[AutoUploadManager] = None
        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._known_tree_items: set[str] = set()
        self._known_failure_items: set[str] = set()
        self._failure_item_paths: Dict[str, str] = {}
        self._failure_item_episode_paths: Dict[str, list[str]] = {}
        self._closed = False
        self._refresh_after_id: Optional[str] = None
        self._default_data_dir = Path(default_data_dir) if default_data_dir is not None else DEFAULT_DATA_ROOT
        self._auto_start_requested = True
        self._upload_progress_bar: Optional[ttk.Progressbar] = None
        self._upload_progress_item: Optional[str] = None
        self._upload_speed_samples: Dict[str, tuple[float, int, float]] = {}
        self._last_snapshot_had_upload = False
        self._progress_refresh_counter = 0
        self._maintenance_lock = threading.Lock()
        self._maintenance_request_id: Optional[str] = None
        self._maintenance_in_flight = False
        self._cache_init_lock = threading.Lock()
        self._cache_init_in_flight = False
        self._daily_curves_in_flight = False

        self._build_ui()
        self._refresh_login_display()
        self._schedule_refresh()

    def _build_ui(self) -> None:
        top = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        top.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(top)
        if self._show_login_controls:
            self._build_login_panel(left)
        else:
            self._build_shared_login_panel(left)
        self._build_upload_controls(left)
        self._build_perf_panel(left)
        top.add(left, weight=0)

        right = ttk.Frame(top)
        right_split = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_split.pack(fill=tk.BOTH, expand=True)
        self._build_episode_table(right_split)
        self._build_failure_tree(right_split)
        self._build_log_panel(right)
        top.add(right, weight=1)

    def _build_login_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="DM3 登录", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(frame, text="地址").grid(row=0, column=0, sticky=tk.W)
        self.base_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        ttk.Entry(frame, textvariable=self.base_url_var, width=34).grid(
            row=0, column=1, columnspan=3, sticky=tk.EW, padx=4
        )
        ttk.Label(frame, text="手机号").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.phone_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.phone_var, width=16).grid(
            row=1, column=1, sticky=tk.W, padx=4, pady=(6, 0)
        )
        ttk.Label(frame, text="密码").grid(row=1, column=2, sticky=tk.W, padx=(8, 0), pady=(6, 0))
        self.password_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.password_var, width=16, show="*").grid(
            row=1, column=3, sticky=tk.W, padx=4, pady=(6, 0)
        )

        row = ttk.Frame(frame)
        row.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=(8, 0))
        self.login_btn = ttk.Button(row, text="登录", command=self._login)
        self.login_btn.pack(side=tk.LEFT)
        self.logout_btn = ttk.Button(row, text="退出", command=self._logout, state=tk.DISABLED)
        self.logout_btn.pack(side=tk.LEFT, padx=6)
        self.login_status_var = tk.StringVar(value="未登录")
        ttk.Label(row, textvariable=self.login_status_var, foreground="#4b5563").pack(side=tk.LEFT, padx=8)

        frame.columnconfigure(1, weight=1)

    def _build_shared_login_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="DM3 状态", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        self.login_status_var = tk.StringVar(value="未登录")
        ttk.Label(frame, textvariable=self.login_status_var, foreground="#4b5563", wraplength=420).pack(
            fill=tk.X
        )

    def _build_upload_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="上传控制", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(frame, text="任务父目录").grid(row=0, column=0, sticky=tk.W)
        self.data_dir_var = tk.StringVar(value=str(self._default_data_dir))
        ttk.Entry(frame, textvariable=self.data_dir_var, width=42, state="readonly").grid(
            row=0, column=1, sticky=tk.EW, padx=4
        )

        ttk.Label(frame, text="resource_type").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.resource_type_var = tk.StringVar(value=self.controller.resource_type)
        ttk.Entry(frame, textvariable=self.resource_type_var, width=18, state="readonly").grid(
            row=1, column=1, sticky=tk.W, padx=4, pady=(6, 0)
        )

        policy = ttk.Frame(frame)
        policy.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
        ttk.Label(policy, text="保护分钟").pack(side=tk.LEFT)
        self.protect_minutes_var = tk.StringVar(value=str(DEFAULT_PROTECT_MINUTES))
        ttk.Spinbox(policy, from_=0, to=1440, textvariable=self.protect_minutes_var, width=6).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(policy, text="保留最新条数").pack(side=tk.LEFT)
        self.keep_latest_var = tk.StringVar(value=str(DEFAULT_KEEP_LATEST_COUNT))
        ttk.Spinbox(policy, from_=0, to=500, textvariable=self.keep_latest_var, width=6).pack(side=tk.LEFT, padx=(4, 12))

        tuning = ttk.Frame(frame)
        tuning.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
        ttk.Label(tuning, text="重试").pack(side=tk.LEFT)
        self.max_retries_var = tk.StringVar(value="5")
        ttk.Spinbox(tuning, from_=0, to=50, textvariable=self.max_retries_var, width=5).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(tuning, text="分片线程").pack(side=tk.LEFT)
        self.multipart_threads_var = tk.StringVar(value=str(DEFAULT_MULTIPART_THREADS))
        ttk.Spinbox(tuning, from_=1, to=8, textvariable=self.multipart_threads_var, width=5).pack(side=tk.LEFT, padx=(4, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))
        self.auto_btn = ttk.Button(actions, text="启动自动上传", style="Accent.TButton", command=self._toggle_auto)
        self.auto_btn.pack(side=tk.LEFT)
        ttk.Button(actions, text="手动全盘扫描", command=lambda: self._enqueue(force=False)).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="确认全部上传", command=self._confirm_upload_all).pack(side=tk.LEFT, padx=6)
        self.daily_curves_btn = ttk.Button(actions, text="生成日报曲线", command=self._generate_daily_curves)
        self.daily_curves_btn.pack(side=tk.LEFT, padx=6)
        self.summary_var = tk.StringVar(value="等待扫描")
        ttk.Label(frame, textvariable=self.summary_var, foreground="#374151").grid(
            row=5, column=0, columnspan=3, sticky=tk.W, pady=(8, 0)
        )
        frame.columnconfigure(1, weight=1)

    def _build_perf_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="性能监控报警", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        self.perf_var = tk.StringVar(value="等待采样")
        ttk.Label(frame, textvariable=self.perf_var, justify=tk.LEFT, wraplength=420).pack(fill=tk.X)
        self.alert_var = tk.StringVar(value="无报警")
        ttk.Label(frame, textvariable=self.alert_var, foreground="#b91c1c", wraplength=420).pack(fill=tk.X, pady=(6, 0))
        monitor_row = ttk.Frame(frame)
        monitor_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(monitor_row, text="打开性能监控页", command=self._open_perf_monitor).pack(side=tk.LEFT)
        ttk.Label(
            monitor_row,
            text=f"http://localhost:{self._perf_monitor_port()}",
            foreground="#6b7280",
        ).pack(side=tk.LEFT, padx=(8, 0))

    def _build_episode_table(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Episode 队列", padding=6)
        if not isinstance(parent, ttk.PanedWindow):
            frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        columns = ("task", "status", "age")
        self.episode_tree = ttk.Treeview(frame, columns=columns, show="tree headings", height=16)
        self.episode_tree.heading("#0", text="episode")
        self.episode_tree.column("#0", width=280, anchor=tk.W)
        settings = {
            "task": ("任务", 140),
            "status": ("状态", 96),
            "age": ("年龄", 88),
        }
        for key, (title, width) in settings.items():
            self.episode_tree.heading(key, text=title)
            self.episode_tree.column(key, width=width, anchor=tk.W)

        yscroll = ttk.Scrollbar(frame, command=self.episode_tree.yview)
        self.episode_tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.episode_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._upload_progress_bar = ttk.Progressbar(
            self.episode_tree,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100,
        )
        self.episode_tree.bind("<Configure>", lambda _event: self._place_upload_progress_bar())
        self.episode_tree.bind("<Motion>", lambda _event: self._place_upload_progress_bar())
        self.episode_tree.bind("<MouseWheel>", lambda _event: self.after_idle(self._place_upload_progress_bar))
        self.episode_tree.bind("<ButtonRelease-1>", lambda _event: self.after_idle(self._place_upload_progress_bar))
        if isinstance(parent, ttk.PanedWindow):
            parent.add(frame, weight=2)

    def _build_failure_tree(self, parent: ttk.PanedWindow) -> None:
        frame = ttk.LabelFrame(parent, text="上传失败 episode", padding=6)
        self.failure_tree = ttk.Treeview(frame, show="tree", height=10)
        self.failure_tree.heading("#0", text="路径")
        self.failure_tree.column("#0", width=620, anchor=tk.W)
        yscroll = ttk.Scrollbar(frame, command=self.failure_tree.yview)
        self.failure_tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.failure_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.failure_menu = tk.Menu(self.failure_tree, tearoff=False)
        self.failure_menu.add_command(label="强制尝试上传该路径", command=self._force_upload_failure_selection)
        self.failure_tree.bind("<Button-3>", self._show_failure_context_menu)
        parent.add(frame, weight=1)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="上传日志", padding=6)
        frame.pack(fill=tk.BOTH, expand=False)
        self.log_text = tk.Text(
            frame,
            height=10,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#111827",
            fg="#e5e7eb",
            state=tk.DISABLED,
        )
        scroll = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _manager(self) -> AutoUploadManager:
        self.controller.resource_type = self.resource_type_var.get().strip()
        manager = self.controller.create_upload_manager(
            Path(self.data_dir_var.get().strip().strip('"')),
            log_callback=lambda message: self.log_queue.put((message, "info")),
        )
        manager.configure(
            resource_type=self.resource_type_var.get().strip(),
            protect_minutes=int(self.protect_minutes_var.get() or str(DEFAULT_PROTECT_MINUTES)),
            keep_latest_count=int(self.keep_latest_var.get() or str(DEFAULT_KEEP_LATEST_COUNT)),
            poll_interval_sec=0,
            max_retries=int(self.max_retries_var.get() or "5"),
            upload_workers=DEFAULT_UPLOAD_WORKERS,
            multipart_threads=int(
                self.multipart_threads_var.get() or str(DEFAULT_MULTIPART_THREADS)
            ),
            upload_backend=UPLOAD_BACKEND_OSS,
        )
        self.manager = manager
        self._start_cache_initialization(manager)
        return manager

    def _start_cache_initialization(self, manager: AutoUploadManager) -> None:
        if manager.episode_cache_initialized():
            return
        with self._cache_init_lock:
            if self._cache_init_in_flight:
                return
            self._cache_init_in_flight = True
        self.log_queue.put(("Initializing upload episode cache with one full scan\n", "info"))

        def runner() -> None:
            error: Optional[str] = None
            episode_count = 0
            try:
                episode_count = len(manager.scan())
            except Exception as exc:
                error = str(exc) or exc.__class__.__name__
            finally:
                with self._cache_init_lock:
                    self._cache_init_in_flight = False
            if error:
                self.log_queue.put((f"Upload cache initialization failed: {error}\n", "error"))
                return
            current_manager = self.manager
            if current_manager is not manager:
                if current_manager is not None and not self._closed:
                    self.after(0, lambda: self._start_cache_initialization(current_manager))
                return
            if not manager.episode_cache_initialized():
                if not self._closed:
                    self.after(0, lambda: self._start_cache_initialization(manager))
                return
            self.log_queue.put((f"Upload cache initialized: episodes={episode_count}\n", "info"))
            if not self._closed:
                self.after(0, lambda: self._refresh_snapshot(force_table=True))

        threading.Thread(target=runner, daemon=True).start()

    def _require_login(self) -> bool:
        if self.controller.session.public_state().get("logged_in"):
            return True
        hint = "请先登录 DM3，再启动上传。"
        if not self._show_login_controls:
            hint = "请先在“DM3 登录”页登录，再启动上传。"
        messagebox.showwarning("未登录", hint, parent=self.winfo_toplevel())
        return False

    def _login(self) -> None:
        phone = self.phone_var.get().strip()
        password = self.password_var.get()
        base_url = self.base_url_var.get().strip() or DEFAULT_BASE_URL
        if not phone or not password:
            messagebox.showwarning("缺少输入", "请输入手机号和密码。", parent=self.winfo_toplevel())
            return

        def work() -> Dict[str, Any]:
            return self.controller.session.login(phone, password, base_url)

        def on_success(_result: Dict[str, Any]) -> None:
            self._refresh_login_display()
            self.request_auto_start()

        self._run_background(work, on_success)

    def _logout(self) -> None:
        def work() -> None:
            self.controller.shutdown_upload_manager()
            self.controller.session.logout()

        def on_success(_result: None) -> None:
            self.manager = None
            self._auto_start_requested = True
            self._refresh_login_display()
            self._update_auto_button_text()

        self._run_background(work, on_success)

    def _refresh_login_display(self) -> None:
        state = self.controller.session.public_state()
        if state["logged_in"]:
            self.login_status_var.set(f"{state['user_name']} ({state['user_id']})")
            if self._show_login_controls:
                self.login_btn.configure(state=tk.DISABLED)
                self.logout_btn.configure(state=tk.NORMAL)
        else:
            self.login_status_var.set("未登录")
            if self._show_login_controls:
                self.login_btn.configure(state=tk.NORMAL)
                self.logout_btn.configure(state=tk.DISABLED)

    def _toggle_auto(self) -> None:
        try:
            if not self._require_login():
                return
            manager = self._manager()
            if manager.upload_progress_snapshot()["auto_enabled"]:
                self._auto_start_requested = False
                manager.stop_auto()
            else:
                self._auto_start_requested = True
                manager.start_auto()
            self._flush_logs()
            self._update_auto_button_text()
        except Exception as exc:
            self._show_error(exc)

    def _update_auto_button_text(self) -> None:
        logged_in = bool(self.controller.session.public_state().get("logged_in"))
        auto_enabled = False
        if self.manager is not None:
            auto_enabled = bool(self.manager.upload_progress_snapshot().get("auto_enabled"))
        if auto_enabled:
            auto_text = "停止自动上传"
        elif self._auto_start_requested and not logged_in:
            auto_text = "自动上传待登录"
        else:
            auto_text = "启动自动上传"
        self.auto_btn.configure(text=auto_text)

    def _enqueue(self, *, force: bool) -> None:
        try:
            if not self._require_login():
                return
            manager = self._manager()
        except Exception as exc:
            self._show_error(exc)
            return

        self.log_queue.put(("开始全盘扫描 episode\n", "info"))

        def work() -> int:
            episodes = manager.scan()
            return manager.enqueue_eligible_episodes(episodes, force=force)

        def on_success(count: int) -> None:
            self.log_queue.put((f"已入队 {count} 条 episode\n", "info"))
            self._refresh_snapshot(force_table=True)

        self._run_background(work, on_success)
        return

    def _confirm_upload_all(self) -> None:
        data_dir = self.data_dir_var.get().strip()
        if not messagebox.askyesno(
            "确认全部上传",
            f"将绕过 30 分钟/最新条数保护，把所有未成功上传的完整 episode 入队。\n{data_dir}",
            parent=self.winfo_toplevel(),
        ):
            return
        self._enqueue(force=True)

    def _generate_daily_curves(self) -> None:
        if self._daily_curves_in_flight:
            return
        data_dir = Path(self.data_dir_var.get().strip().strip('"')).expanduser().resolve()
        output_dir = data_dir / "reports" / "daily_curves"
        self._daily_curves_in_flight = True
        self.daily_curves_btn.configure(state=tk.DISABLED)
        self.log_queue.put((f"开始生成今日采集曲线：{data_dir}\n", "info"))

        def work() -> Dict[str, Any]:
            try:
                from utils.daily_collection_report import (
                    DEFAULT_RESAMPLE_POINTS,
                    default_report_date,
                    generate_daily_reports,
                )

                report_date = default_report_date()
                reports = generate_daily_reports(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    report_date=report_date,
                    resample_points=DEFAULT_RESAMPLE_POINTS,
                )
                return {
                    "ok": True,
                    "report_date": report_date.isoformat(),
                    "report_count": len(reports),
                    "output_dir": str(output_dir),
                }
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def on_success(result: Dict[str, Any]) -> None:
            self._daily_curves_in_flight = False
            self.daily_curves_btn.configure(state=tk.NORMAL)
            if not result.get("ok"):
                self._show_error(RuntimeError(str(result.get("error") or "daily curve generation failed")))
                return
            self.log_queue.put((
                "日报曲线生成完成："
                f"{result.get('report_date')} | {result.get('report_count')} 张 | {result.get('output_dir')}\n",
                "info",
            ))
            self._flush_logs()

        self._run_background(work, on_success)

    def _retry_failed(self) -> None:
        try:
            if not self._require_login():
                return
            count = self._manager().enqueue_failed()
            self.log_queue.put((f"已重新入队 {count} 条失败 episode\n", "info"))
        except Exception as exc:
            self._show_error(exc)

    def set_data_dir(self, data_dir: Union[Path, str]) -> None:
        value = str(data_dir)
        if self.data_dir_var.get().strip() == value:
            return
        self.data_dir_var.set(value)
        if self.manager is not None:
            self._manager()
        self.log_queue.put((f"数据目录已切换：{value}\n", "info"))

    def request_auto_start(self) -> None:
        self._auto_start_requested = True
        self._refresh_snapshot()

    def stop_auto_for_logout(self) -> None:
        try:
            self.controller.shutdown_upload_manager()
        finally:
            self.manager = None
            self._auto_start_requested = True

    def _schedule_refresh(self) -> None:
        if self._closed:
            return
        self._refresh_login_display()
        self._flush_logs()
        self._refresh_snapshot_if_due()
        self._refresh_after_id = self.after(500, self._schedule_refresh)

    def handle_maintenance_request_once(self) -> None:
        if self._closed:
            return
        if self.manager is None:
            try:
                self.manager = self._manager()
            except Exception as exc:
                self.log_queue.put((f"Failed to initialize upload manager for maintenance: {exc}\n", "error"))
                return
        self._handle_maintenance_request_once()

    def _handle_maintenance_request_once(self) -> None:
        manager = self.manager
        if manager is None:
            return
        request_path = manager.data_dir / MAINTENANCE_REQUEST_NAME
        try:
            if not request_path.exists():
                return
            payload = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.log_queue.put((f"Failed to read upload maintenance request: {exc}\n", "error"))
            return
        request_id = str(payload.get("request_id") or "")
        if not request_id:
            return
        with self._maintenance_lock:
            if self._maintenance_in_flight or self._maintenance_request_id == request_id:
                return
            self._maintenance_in_flight = True
            self._maintenance_request_id = request_id
        self.log_queue.put((f"Inter-episode upload maintenance started: {request_id}\n", "info"))
        can_enqueue = bool(self.controller.session.public_state().get("logged_in"))
        threading.Thread(
            target=self._run_maintenance_scan,
            args=(manager, payload, can_enqueue),
            daemon=True,
        ).start()

    def _run_maintenance_scan(
        self,
        manager: AutoUploadManager,
        payload: Dict[str, Any],
        can_enqueue: bool,
    ) -> None:
        request_id = str(payload.get("request_id") or "")
        started = time.perf_counter()
        episode_count = 0
        queued_count = 0
        scan_mode = ""
        target_found = False
        scan_elapsed = 0.0
        enqueue_elapsed = 0.0
        enqueue_phases: Dict[str, float] = {}
        error: Optional[str] = None
        try:
            output_dir = Path(str(payload.get("output_dir") or "")).expanduser().resolve()
            episode_name = str(payload.get("episode_name") or "")
            if not episode_name or Path(episode_name).name != episode_name:
                raise ValueError(f"Invalid maintenance episode name: {episode_name!r}")
            scan_started = time.perf_counter()
            result = manager.refresh_after_episode(output_dir / episode_name)
            scan_elapsed = time.perf_counter() - scan_started
            episodes = result.episodes
            episode_count = len(episodes)
            scan_mode = result.scan_mode
            target_found = result.target_found
            if can_enqueue:
                enqueue_started = time.perf_counter()
                queued_count = manager.enqueue_eligible_episodes(
                    episodes,
                    force=False,
                    timings=enqueue_phases,
                )
                enqueue_elapsed = time.perf_counter() - enqueue_started
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
        elapsed = time.perf_counter() - started
        done_payload: Dict[str, Any] = {
            "request_id": request_id,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_sec": elapsed,
            "episode_count": episode_count,
            "queued_count": queued_count,
            "scan_mode": scan_mode,
            "target_found": target_found,
            "scan_elapsed_sec": scan_elapsed,
            "enqueue_elapsed_sec": enqueue_elapsed,
            "enqueue_phases": enqueue_phases,
        }
        if error:
            done_payload["error"] = error
        try:
            self._write_maintenance_done_file(manager, done_payload)
        except Exception as exc:
            error = f"{error}; done write failed: {exc}" if error else f"done write failed: {exc}"
        with self._maintenance_lock:
            self._maintenance_in_flight = False
        self.after(
            0,
            lambda payload=done_payload, scan_error=error: self._finish_maintenance_scan_ui(
                manager,
                payload,
                scan_error,
            ),
        )

    @staticmethod
    def _write_maintenance_done_file(
        manager: AutoUploadManager,
        done_payload: Dict[str, Any],
    ) -> None:
        done_path = manager.data_dir / MAINTENANCE_DONE_NAME
        tmp_path = done_path.with_suffix(done_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(done_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp_path.replace(done_path)

    def _finish_maintenance_scan_ui(
        self,
        manager: AutoUploadManager,
        done_payload: Dict[str, Any],
        error: Optional[str],
    ) -> None:
        self._refresh_snapshot(force_table=True)
        if error:
            self.log_queue.put((f"Inter-episode upload maintenance failed: {error}\n", "error"))
        else:
            self.log_queue.put((
                "Inter-episode upload maintenance completed and UI refreshed: "
                f"mode={done_payload.get('scan_mode')}, "
                f"target_found={done_payload.get('target_found')}, "
                f"episodes={done_payload.get('episode_count')}, "
                f"queued={done_payload.get('queued_count')}, "
                f"scan={float(done_payload.get('scan_elapsed_sec') or 0.0):.3f}s, "
                f"enqueue={float(done_payload.get('enqueue_elapsed_sec') or 0.0):.3f}s, "
                f"elapsed={float(done_payload.get('elapsed_sec') or 0.0):.2f}s\n",
                "info",
            ))

    def _refresh_snapshot_if_due(self) -> None:
        if self.manager is None:
            self._refresh_snapshot()
            return
        self._update_auto_button_text()
        self._progress_refresh_counter += 1
        if self._last_snapshot_had_upload or self._progress_refresh_counter >= 2:
            self._progress_refresh_counter = 0
            self._refresh_upload_progress()

    def _refresh_snapshot(self, *, force_table: bool = False) -> None:
        if self.manager is None:
            try:
                self.manager = self._manager()
            except Exception:
                return
        collection_busy = self._is_collection_busy()
        with self._cache_init_lock:
            cache_init_in_flight = self._cache_init_in_flight
        background_refresh = not collection_busy and not cache_init_in_flight
        self.manager.set_background_refresh_enabled(background_refresh)
        self.manager.set_auto_enqueue_suspended(not background_refresh)
        snapshot = self.manager.snapshot(background_refresh=background_refresh)
        if self._auto_start_requested and self.controller.session.public_state().get("logged_in") and not snapshot["auto_enabled"]:
            self.manager.start_auto()
            self._flush_logs()
            snapshot = self.manager.snapshot(background_refresh=background_refresh)
        self._last_snapshot_had_upload = bool(snapshot.get("current_episode"))
        self._update_auto_button_text()

        stats = snapshot["stats"]
        self.summary_var.set(
            f"总数 {stats['total']} | 完整 {stats['complete']} | 队列 {stats['queued']} | "
            f"成功 {stats['success']} | 失败 {stats['failed']} | 保护中 {stats['protected']}"
        )

        perf = snapshot["performance"]
        self.perf_var.set(
            f"CPU {perf['cpu_percent']:.0f}% | 内存 {perf['memory_percent']:.0f}% | "
            f"磁盘读 {perf['disk_read_mb_s']:.1f} MB/s 写 {perf['disk_write_mb_s']:.1f} MB/s | "
            f"上传 {perf['net_upload_mb_s']:.1f} MB/s 下载 {perf['net_download_mb_s']:.1f} MB/s"
        )
        alerts = perf.get("alerts") or []
        self.alert_var.set(" | ".join(alerts) if alerts else "无报警")

        if collection_busy and not force_table:
            self._update_upload_progress_bar(self._uploading_row(snapshot))
            return

        live_names = set()
        for row in snapshot.get("queue_episodes", snapshot["episodes"]):
            item_id = row.get("id") or row.get("path") or row["name"]
            live_names.add(item_id)
            values = (
                row.get("task_name", ""),
                self._format_status(row["status"]),
                self._format_age(row["age_seconds"]),
            )
            if item_id in self._known_tree_items:
                self.episode_tree.item(item_id, text=row["name"], values=values)
            else:
                self.episode_tree.insert("", tk.END, iid=item_id, text=row["name"], values=values)
                self._known_tree_items.add(item_id)

        for stale in list(self._known_tree_items - live_names):
            self.episode_tree.delete(stale)
            self._known_tree_items.remove(stale)

        self._refresh_failure_tree(snapshot.get("failed_file_tree") or [])

    def _refresh_upload_progress(self) -> None:
        if self.manager is None:
            return
        progress = self.manager.upload_progress_snapshot()
        self._last_snapshot_had_upload = bool(progress.get("current_episode"))
        perf = progress.get("performance") or {}
        if perf:
            self.perf_var.set(
                f"CPU {perf['cpu_percent']:.0f}% | 内存 {perf['memory_percent']:.0f}% | "
                f"磁盘读 {perf['disk_read_mb_s']:.1f} MB/s 写 {perf['disk_write_mb_s']:.1f} MB/s | "
                f"上传 {perf['net_upload_mb_s']:.1f} MB/s 下载 {perf['net_download_mb_s']:.1f} MB/s"
            )
            alerts = perf.get("alerts") or []
            self.alert_var.set(" | ".join(alerts) if alerts else "无报警")
        current = progress.get("current_episode")
        if not current:
            self._update_upload_progress_bar(None)
            return
        row = {
            "id": current,
            "path": progress.get("episode_path") or current,
            "name": progress.get("episode_name") or Path(current).name,
            "status": "uploading",
            "progress": progress.get("progress") or {},
        }
        item_id = row["id"]
        if item_id in self._known_tree_items:
            values = list(self.episode_tree.item(item_id, "values"))
            if len(values) >= 7:
                values[1] = ""
                values[6] = self._format_progress(
                    row["progress"],
                    speed_mb_s=self._upload_speed_mb_s(item_id, row),
                )
                self.episode_tree.item(item_id, values=tuple(values))
        self._update_upload_progress_bar(row)

    def _uploading_row(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return next(
            (row for row in snapshot.get("queue_episodes", snapshot["episodes"]) if row["status"] == "uploading"),
            None,
        )

    def _refresh_failure_tree(self, failed_roots: list[Dict[str, Any]]) -> None:
        self.failure_tree.delete(*self.failure_tree.get_children())
        self._known_failure_items.clear()
        self._failure_item_paths.clear()
        self._failure_item_episode_paths.clear()

        tasks: Dict[str, Dict[str, Any]] = {}
        for root in failed_roots:
            task_path = str(root.get("task_path") or "")
            task = tasks.setdefault(
                task_path,
                {
                    "task_name": root.get("task_name", ""),
                    "task_path": task_path,
                    "episodes": [],
                },
            )
            task["episodes"].append(root)

        for task_path, task in sorted(tasks.items(), key=lambda item: str(item[1]["task_name"])):
            task_id = self._failure_item_id(task_path or str(task.get("task_name") or "task"))
            episode_paths = [str(root.get("path") or "") for root in task["episodes"]]
            self.failure_tree.insert(
                "",
                tk.END,
                iid=task_id,
                text=f"{task.get('task_name') or task_path} ({len(episode_paths)} failed)",
                open=True,
            )
            self._failure_item_paths[task_id] = task_path
            self._failure_item_episode_paths[task_id] = episode_paths
            self._known_failure_items.add(task_id)
            for root in sorted(task["episodes"], key=lambda item: str(item.get("episode_name", ""))):
                self._insert_failed_episode_node(task_id, root)

    def _insert_failed_episode_node(self, parent_id: str, root: Dict[str, Any]) -> None:
        episode_path = str(root.get("path") or "")
        root_id = self._failure_item_id(episode_path)
        text = str(root.get("episode_name", ""))
        self.failure_tree.insert(
            parent_id,
            tk.END,
            iid=root_id,
            text=text,
            open=True,
        )
        self._failure_item_paths[root_id] = episode_path
        self._failure_item_episode_paths[root_id] = [episode_path]
        self._known_failure_items.add(root_id)

    @staticmethod
    def _failure_item_id(path: str) -> str:
        return "failure:" + path.replace("\\", "/").replace(":", "|")

    def _show_failure_context_menu(self, event: tk.Event) -> None:
        item_id = self.failure_tree.identify_row(event.y)
        if not item_id:
            return
        self.failure_tree.selection_set(item_id)
        self.failure_menu.tk_popup(event.x_root, event.y_root)

    def _force_upload_failure_selection(self) -> None:
        selection = self.failure_tree.selection()
        if not selection:
            return
        episode_paths = self._failure_item_episode_paths.get(selection[0]) or []
        if not episode_paths:
            return
        try:
            if not self._require_login():
                return
            count = self._manager().force_enqueue_paths(Path(path) for path in episode_paths)
            self.log_queue.put((f"已插队强制尝试上传 {count} 条 episode\n", "info"))
            self._refresh_snapshot()
        except Exception as exc:
            self._show_error(exc)

    def _is_collection_busy(self) -> bool:
        processes = getattr(self.controller, "processes", None)
        saver = getattr(processes, "saver", None)
        if saver is None:
            return False
        if not bool(getattr(saver, "running", False)):
            return False
        return bool(getattr(saver, "recording", False)) or not bool(getattr(saver, "ready_for_episode", False))

    def _flush_logs(self) -> None:
        lines: list[tuple[str, str]] = []
        while True:
            try:
                lines.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if not lines:
            return
        self.log_text.configure(state=tk.NORMAL)
        for text, _level in lines:
            self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _append_upload_log(self, message: str, *, level: str = "info") -> None:
        """Append one line to the upload log panel and flush immediately."""
        text = message if message.endswith("\n") else f"{message}\n"
        self.log_queue.put((text, level))
        self._flush_logs()

    def _run_background(
        self,
        work: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        def runner() -> None:
            try:
                result = work()
            except Exception as exc:
                self.after(0, lambda error=exc: self._show_error(error))
            else:
                self.after(0, lambda value=result: on_success(value))

        threading.Thread(target=runner, daemon=True).start()

    def _show_error(self, exc: Exception) -> None:
        message = str(exc)
        self.log_queue.put((f"错误：{message}\n", "error"))
        messagebox.showerror("错误", message, parent=self.winfo_toplevel())

    @staticmethod
    def _perf_monitor_port() -> int:
        raw = os.environ.get("PERF_PORT", str(DEFAULT_PERF_MONITOR_PORT)).strip()
        try:
            return max(1, min(65535, int(raw)))
        except ValueError:
            return DEFAULT_PERF_MONITOR_PORT

    def _open_perf_monitor(self) -> None:
        url = f"http://localhost:{self._perf_monitor_port()}"
        try:
            opened = webbrowser.open(url, new=2)
        except Exception as exc:
            self._show_error(exc)
            return
        if not opened:
            messagebox.showwarning(
                "打开失败",
                f"系统未能打开浏览器，请手动访问 {url}",
                parent=self.winfo_toplevel(),
            )

    def shutdown(self, *, stop_manager: bool = True) -> None:
        self._closed = True
        if self._refresh_after_id is not None:
            try:
                self.after_cancel(self._refresh_after_id)
            except tk.TclError:
                pass
            self._refresh_after_id = None
        if not stop_manager:
            return
        if self._owns_controller:
            self.controller.shutdown()
        elif self.manager is not None:
            self.manager.shutdown()
        self.manager = None

    def _update_upload_progress_bar(self, row: Optional[Dict[str, Any]]) -> None:
        if row is None or self._upload_progress_bar is None:
            self._upload_progress_item = None
            self._upload_speed_samples.clear()
            self._hide_upload_progress_bar()
            return
        item_id = row.get("id") or row.get("path") or row["name"]
        self._upload_progress_item = item_id
        self._upload_progress_bar.configure(value=self._progress_percent(row.get("progress") or {}))
        self._place_upload_progress_bar()

    def _place_upload_progress_bar(self) -> None:
        if self._upload_progress_bar is None or not self._upload_progress_item:
            self._hide_upload_progress_bar()
            return
        try:
            bbox = self.episode_tree.bbox(self._upload_progress_item, "status")
        except tk.TclError:
            bbox = ()
        if not bbox:
            self._hide_upload_progress_bar()
            return
        x, y, width, height = bbox
        self._upload_progress_bar.place(
            x=x + 3,
            y=y + max(1, (height - 12) // 2),
            width=max(20, width - 6),
            height=12,
        )
        self._upload_progress_bar.lift()

    def _hide_upload_progress_bar(self) -> None:
        if self._upload_progress_bar is not None:
            self._upload_progress_bar.place_forget()

    @staticmethod
    def _progress_percent(progress: Dict[str, int]) -> float:
        total = int(progress.get("total_bytes") or 0)
        current = int(progress.get("current_bytes") or progress.get("completed_bytes") or 0)
        if total > 0:
            return max(0.0, min(100.0, 100.0 * current / total))
        files_total = int(progress.get("total_files") or 0)
        files_done = int(progress.get("completed_files") or 0)
        if files_total > 0:
            return max(0.0, min(100.0, 100.0 * files_done / files_total))
        return 0.0

    def _upload_speed_mb_s(self, item_id: str, row: Dict[str, Any]) -> Optional[float]:
        if row.get("status") != "uploading":
            self._upload_speed_samples.pop(item_id, None)
            return None
        progress = row.get("progress") or {}
        current = int(progress.get("current_bytes") or progress.get("completed_bytes") or 0)
        now = time.monotonic()
        sample = self._upload_speed_samples.get(item_id)
        self._upload_speed_samples = {item_id: (now, current, 0.0)} if sample is None else self._upload_speed_samples
        if sample is None:
            return None
        last_time, last_bytes, last_speed = sample
        elapsed = now - last_time
        if elapsed <= 0.2:
            return last_speed
        delta = current - last_bytes
        speed = max(0.0, delta / elapsed / (1024 * 1024))
        if last_speed > 0:
            speed = (0.35 * speed) + (0.65 * last_speed)
        self._upload_speed_samples = {item_id: (now, current, speed)}
        return speed

    @staticmethod
    def _format_status(status: str) -> str:
        labels = {
            "pending": "待入队",
            "protected": "保护中",
            "queued": "队列中",
            "uploading": "上传中",
            "success": "已上传",
            "failed": "失败",
            "incomplete": "未完整",
        }
        return labels.get(status, status)

    @staticmethod
    def _format_age(seconds: float) -> str:
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"{minutes}m"
        return f"{minutes // 60}h{minutes % 60:02d}m"

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes} B"
        if num_bytes < 1024**2:
            return f"{num_bytes / 1024:.1f} KB"
        if num_bytes < 1024**3:
            return f"{num_bytes / 1024**2:.1f} MB"
        return f"{num_bytes / 1024**3:.2f} GB"

    @staticmethod
    def _format_progress(progress: Dict[str, int], *, speed_mb_s: Optional[float] = None) -> str:
        total = int(progress.get("total_bytes") or 0)
        current = int(progress.get("current_bytes") or progress.get("completed_bytes") or 0)
        files_done = int(progress.get("completed_files") or 0)
        files_total = int(progress.get("total_files") or 0)
        if total > 0:
            percent = 100.0 * current / total
            text = (
                f"{percent:.2f}% "
                f"{UploadConsoleFrame._format_bytes(current)}/"
                f"{UploadConsoleFrame._format_bytes(total)} "
                f"{files_done}/{files_total}"
            )
            if speed_mb_s is not None:
                text += f" | {speed_mb_s:.1f} MB/s"
            return text
        if files_total > 0:
            percent = 100.0 * files_done / files_total
            text = f"{percent:.0f}% {files_done}/{files_total}"
            if speed_mb_s is not None:
                text += f" | {speed_mb_s:.1f} MB/s"
            return text
        if speed_mb_s is not None:
            return f"{speed_mb_s:.1f} MB/s"
        return "-"


class UploadConsole:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Vive DM3 上传与性能监控")
        self.root.geometry("1220x780")
        self.root.minsize(1040, 680)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Accent.TButton", foreground="white", background="#2563eb")
        style.map("Accent.TButton", background=[("active", "#1d4ed8")])
        style.configure("Danger.TButton", foreground="white", background="#b91c1c")
        style.map("Danger.TButton", background=[("active", "#991b1b")])

        self.panel = UploadConsoleFrame(self.root)
        self.panel.pack(fill=tk.BOTH, expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self.panel.shutdown()
        self.root.destroy()


def main() -> None:
    policy = apply_upload_process_policy()
    if not policy.affinity:
        raise RuntimeError(f"Upload CPU isolation failed; refusing to start console: {policy.errors}")
    UploadConsole().run()


if __name__ == "__main__":
    main()
