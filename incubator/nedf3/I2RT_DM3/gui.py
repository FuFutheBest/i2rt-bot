"""Three-tab DM3 application for I2RT NEDF3 collection."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Optional

from incubator.nedf3.I2RT_DM3.controller import (
    DEFAULT_DATA_ROOT,
    I2RTDM3Controller,
)
from incubator.nedf3.Vive_DM3.upload_gui import UploadConsoleFrame
from incubator.nedf3.windows_DM3.dm3_api import (
    DEFAULT_BASE_URL,
    DM3TaskExecution,
)
from utils.logger_config import logger


POLL_INTERVAL_MS = 250


class I2RTDM3App:
    """DM3 login, I2RT collection, and upload in one operator window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.controller = I2RTDM3Controller()
        self.current_tasks: list[DM3TaskExecution] = []
        self._log_offsets = {"device": 0, "saver": 0}
        self._ui_messages: "queue.Queue[str]" = queue.Queue()
        self._closing = False

        self.root.title("I2RT DM3 数据采集")
        self.root.geometry("1280x820")
        self.root.minsize(1080, 700)
        self._configure_style()
        self._build_ui()
        self._refresh_session_display()
        self._poll_runtime()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Start.TButton", foreground="white", background="#167d3f")
        style.map("Start.TButton", background=[("active", "#116532")])
        style.configure("Stop.TButton", foreground="white", background="#b42318")
        style.map("Stop.TButton", background=[("active", "#8f1c13")])

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.login_tab = ttk.Frame(self.notebook, padding=10)
        self.collection_tab = ttk.Frame(self.notebook, padding=10)
        self.upload_tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(self.login_tab, text="DM3 登录")
        self.notebook.add(self.collection_tab, text="数据采集")
        self.notebook.add(self.upload_tab, text="上传与性能监控")

        self._build_login_tab()
        self._build_collection_tab()
        self.upload_panel = UploadConsoleFrame(
            self.upload_tab,
            controller=self.controller,
            default_data_dir=DEFAULT_DATA_ROOT,
            show_login_controls=False,
        )
        self.upload_panel.pack(fill=tk.BOTH, expand=True)

    def _build_login_tab(self) -> None:
        login = ttk.LabelFrame(self.login_tab, text="1. DM3 登录", padding=10)
        login.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(login, text="基础地址").grid(row=0, column=0, sticky=tk.W)
        self.base_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        ttk.Entry(login, textvariable=self.base_url_var, width=38).grid(
            row=0, column=1, sticky=tk.EW, padx=6
        )
        ttk.Label(login, text="手机号").grid(row=0, column=2, sticky=tk.W)
        self.phone_var = tk.StringVar()
        ttk.Entry(login, textvariable=self.phone_var, width=18).grid(
            row=0, column=3, padx=6
        )
        ttk.Label(login, text="密码").grid(row=0, column=4, sticky=tk.W)
        self.password_var = tk.StringVar()
        ttk.Entry(login, textvariable=self.password_var, width=18, show="*").grid(
            row=0, column=5, padx=6
        )
        self.login_btn = ttk.Button(login, text="登录", command=self._login)
        self.login_btn.grid(row=0, column=6, padx=(8, 4))
        self.logout_btn = ttk.Button(
            login,
            text="退出登录",
            command=self._logout,
            state=tk.DISABLED,
        )
        self.logout_btn.grid(row=0, column=7, padx=4)
        self.login_status_var = tk.StringVar(value="未登录")
        ttk.Label(login, textvariable=self.login_status_var, foreground="#4b5563").grid(
            row=1,
            column=0,
            columnspan=8,
            sticky=tk.W,
            pady=(8, 0),
        )
        login.columnconfigure(1, weight=1)

        tasks = ttk.LabelFrame(self.login_tab, text="2. 我的任务", padding=10)
        tasks.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(tasks)
        controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(controls, text="执行日期").pack(side=tk.LEFT)
        self.execute_day_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.execute_day_var, width=14).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(controls, text="页码").pack(side=tk.LEFT, padx=(10, 0))
        self.page_var = tk.StringVar(value="1")
        ttk.Entry(controls, textvariable=self.page_var, width=5).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(controls, text="每页").pack(side=tk.LEFT, padx=(10, 0))
        self.page_size_var = tk.StringVar(value="10")
        ttk.Entry(controls, textvariable=self.page_size_var, width=5).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(controls, text="获取任务", command=self._fetch_tasks).pack(
            side=tk.LEFT, padx=(10, 4)
        )
        ttk.Button(
            controls,
            text="选中当前任务",
            command=self._select_task,
        ).pack(side=tk.LEFT, padx=4)

        columns = ("id", "status", "name")
        self.task_tree = ttk.Treeview(tasks, columns=columns, show="headings", height=14)
        for key, title, width in (
            ("id", "执行 ID", 120),
            ("status", "状态", 120),
            ("name", "任务名称", 700),
        ):
            self.task_tree.heading(key, text=title)
            self.task_tree.column(key, width=width, anchor=tk.W)
        scroll = ttk.Scrollbar(tasks, command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.task_tree.pack(fill=tk.BOTH, expand=True)
        self.selected_task_var = tk.StringVar(value="已选任务：无")
        ttk.Label(
            tasks,
            textvariable=self.selected_task_var,
            foreground="#4b5563",
        ).pack(anchor=tk.W, pady=(8, 0))

    def _build_collection_tab(self) -> None:
        metadata = ttk.LabelFrame(
            self.collection_tab,
            text="DM3 任务与输出",
            padding=10,
        )
        metadata.pack(fill=tk.X, pady=(0, 8))
        self.collection_user_var = tk.StringVar(value="用户：未登录")
        self.collection_task_var = tk.StringVar(value="任务：未选择")
        ttk.Label(metadata, textvariable=self.collection_user_var).grid(
            row=0, column=0, columnspan=3, sticky=tk.W
        )
        ttk.Label(
            metadata,
            textvariable=self.collection_task_var,
            wraplength=950,
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(4, 8))
        ttk.Label(metadata, text="输出目录").grid(row=2, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value=self.controller.output_dir)
        ttk.Entry(metadata, textvariable=self.output_dir_var).grid(
            row=2, column=1, sticky=tk.EW, padx=6
        )
        ttk.Button(metadata, text="浏览", command=self._browse_output_dir).grid(
            row=2, column=2
        )
        metadata.columnconfigure(1, weight=1)

        controls = ttk.Frame(self.collection_tab)
        controls.pack(fill=tk.X, pady=(0, 8))
        device = ttk.LabelFrame(controls, text="设备", padding=10)
        device.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        ttk.Label(
            device,
            text=f"配置：{self.controller.device_config_name}",
        ).pack(anchor=tk.W)
        device_actions = ttk.Frame(device)
        device_actions.pack(fill=tk.X, pady=(8, 0))
        self.device_start_btn = ttk.Button(
            device_actions,
            text="启动设备",
            style="Start.TButton",
            command=self._start_device,
        )
        self.device_start_btn.pack(side=tk.LEFT)
        self.device_stop_btn = ttk.Button(
            device_actions,
            text="停止设备",
            style="Stop.TButton",
            command=self._stop_device,
            state=tk.DISABLED,
        )
        self.device_stop_btn.pack(side=tk.LEFT, padx=6)
        self.device_status_var = tk.StringVar(value="设备：停止")
        ttk.Label(device_actions, textvariable=self.device_status_var).pack(
            side=tk.LEFT, padx=10
        )

        saver = ttk.LabelFrame(controls, text="NEDF3 保存", padding=10)
        saver.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        ttk.Label(
            saver,
            text=f"配置：{self.controller.saver_config_path}",
        ).pack(anchor=tk.W)
        saver_actions = ttk.Frame(saver)
        saver_actions.pack(fill=tk.X, pady=(8, 0))
        self.saver_start_btn = ttk.Button(
            saver_actions,
            text="启动保存",
            style="Start.TButton",
            command=self._start_saver,
        )
        self.saver_start_btn.pack(side=tk.LEFT)
        self.saver_stop_btn = ttk.Button(
            saver_actions,
            text="停止保存",
            style="Stop.TButton",
            command=self._stop_saver,
            state=tk.DISABLED,
        )
        self.saver_stop_btn.pack(side=tk.LEFT, padx=6)
        self.toggle_btn = ttk.Button(
            saver_actions,
            text="开始 / 完成 Episode",
            command=lambda: self._send_episode_command("toggle"),
            state=tk.DISABLED,
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=(12, 4))
        self.discard_btn = ttk.Button(
            saver_actions,
            text="丢弃 Episode",
            command=lambda: self._send_episode_command("discard"),
            state=tk.DISABLED,
        )
        self.discard_btn.pack(side=tk.LEFT, padx=4)
        self.saver_status_var = tk.StringVar(value="保存：停止")
        ttk.Label(saver, textvariable=self.saver_status_var).pack(
            anchor=tk.W,
            pady=(8, 0),
        )
        self.saved_count_var = tk.StringVar(value="已保存 Episode：0")
        ttk.Label(saver, textvariable=self.saved_count_var).pack(anchor=tk.W)

        logs = ttk.PanedWindow(self.collection_tab, orient=tk.HORIZONTAL)
        logs.pack(fill=tk.BOTH, expand=True)
        self.device_log = self._create_log_panel(logs, "设备输出")
        self.saver_log = self._create_log_panel(logs, "保存输出")
        logs.add(self.device_log["frame"], weight=1)
        logs.add(self.saver_log["frame"], weight=1)

    @staticmethod
    def _create_log_panel(parent: ttk.PanedWindow, title: str) -> dict[str, Any]:
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            font=("Monospace", 9),
            state=tk.DISABLED,
        )
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)
        return {"frame": frame, "text": text}

    def _login(self) -> None:
        phone = self.phone_var.get().strip()
        password = self.password_var.get()
        base_url = self.base_url_var.get().strip() or DEFAULT_BASE_URL
        if not phone or not password:
            messagebox.showwarning("缺少输入", "请输入手机号和密码。")
            return
        self.login_btn.configure(state=tk.DISABLED)

        def success(_state: dict[str, Any]) -> None:
            self.password_var.set("")
            self._refresh_session_display()
            self.upload_panel.request_auto_start()
            self._log_message("DM3 登录成功")

        self._run_background(
            lambda: self.controller.session.login(phone, password, base_url),
            success,
            finally_callback=self._refresh_session_display,
        )

    def _logout(self) -> None:
        def work() -> None:
            self.controller.shutdown_upload_manager()
            self.controller.session.logout()

        def success(_result: None) -> None:
            self.upload_panel.manager = None
            self._refresh_session_display()
            self._log_message("已退出 DM3")

        self._run_background(work, success)

    def _fetch_tasks(self) -> None:
        try:
            page = max(1, int(self.page_var.get() or "1"))
            page_size = max(1, int(self.page_size_var.get() or "10"))
        except ValueError:
            messagebox.showwarning("输入错误", "页码和每页数量必须是整数。")
            return
        execute_day = self.execute_day_var.get().strip() or None

        def success(tasks: list[DM3TaskExecution]) -> None:
            self.current_tasks = tasks
            self.task_tree.delete(*self.task_tree.get_children())
            for index, task in enumerate(tasks):
                self.task_tree.insert(
                    "",
                    tk.END,
                    iid=str(index),
                    values=(task.id or "", task.status or "-", task.name),
                )
            self._log_message(f"已加载 {len(tasks)} 个任务")

        self._run_background(
            lambda: self.controller.session.fetch_tasks(
                page=page,
                page_size=page_size,
                execute_day=execute_day,
            ),
            success,
        )

    def _select_task(self) -> None:
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("未选择任务", "请先选择一条任务。")
            return
        index = int(selection[0])

        def success(_state: dict[str, Any]) -> None:
            self.controller.apply_selected_task_defaults()
            self.output_dir_var.set(self.controller.output_dir)
            self.upload_panel.set_data_dir(Path(self.controller.output_dir).parent)
            self._refresh_session_display()
            self._log_message(f"已选择任务：{self.controller.task_name}")

        self._run_background(
            lambda: self.controller.session.select_task_by_index(index),
            success,
        )

    def _browse_output_dir(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.output_dir_var.get() or str(DEFAULT_DATA_ROOT)
        )
        if selected:
            self.output_dir_var.set(selected)
            self.controller.output_dir = selected
            self.upload_panel.set_data_dir(Path(selected).parent)

    def _start_device(self) -> None:
        self.device_start_btn.configure(state=tk.DISABLED)
        self._run_background(
            self.controller.start_device,
            lambda _result: self._log_message("设备进程已启动"),
            finally_callback=self._update_controls,
        )

    def _stop_device(self) -> None:
        if self.controller.processes.saver.running:
            messagebox.showwarning(
                "请先停止保存",
                "停止设备前请先停止 NEDF3 保存进程。",
            )
            return
        self.device_stop_btn.configure(state=tk.DISABLED)
        self._run_background(
            self.controller.processes.stop_device,
            lambda _result: self._log_message("设备进程已停止"),
            finally_callback=self._update_controls,
        )

    def _start_saver(self) -> None:
        self.controller.output_dir = self.output_dir_var.get().strip()
        if not self.controller.output_dir:
            messagebox.showwarning("缺少输出目录", "请选择输出目录。")
            return
        self.saver_start_btn.configure(state=tk.DISABLED)
        self._run_background(
            self.controller.start_saver,
            lambda _result: self._log_message("NEDF3 保存进程已启动"),
            finally_callback=self._update_controls,
        )

    def _stop_saver(self) -> None:
        self.saver_stop_btn.configure(state=tk.DISABLED)
        self._run_background(
            self.controller.processes.stop_saver,
            lambda _result: self._log_message("NEDF3 保存进程已停止"),
            finally_callback=self._update_controls,
        )

    def _send_episode_command(self, command: str) -> None:
        try:
            self.controller.processes.send_saver_command(command)
        except Exception as exc:
            self._show_error(exc)

    def _refresh_session_display(self) -> None:
        state = self.controller.session.public_state()
        logged_in = bool(state["logged_in"])
        if logged_in:
            identity = f"{state['user_name']} ({state['user_id']})"
            self.login_status_var.set(f"已登录：{identity}")
            self.collection_user_var.set(f"用户：{identity}")
            self.login_btn.configure(state=tk.DISABLED)
            self.logout_btn.configure(state=tk.NORMAL)
        else:
            self.login_status_var.set("未登录")
            self.collection_user_var.set("用户：未登录")
            self.login_btn.configure(state=tk.NORMAL)
            self.logout_btn.configure(state=tk.DISABLED)
        if state["task_selected"]:
            selected = f"{state['task_execution_id']} / {state['task_name']}"
            self.selected_task_var.set(f"已选任务：{selected}")
            description = state.get("task_description") or ""
            self.collection_task_var.set(
                f"任务：{state['task_name']} | {description}"
            )
        else:
            self.selected_task_var.set("已选任务：无")
            self.collection_task_var.set("任务：未选择")
        tab_state = tk.NORMAL if logged_in else tk.DISABLED
        self.notebook.tab(self.collection_tab, state=tab_state)
        self.notebook.tab(self.upload_tab, state=tab_state)
        if not logged_in:
            self.notebook.select(self.login_tab)
        self._update_controls()

    def _poll_runtime(self) -> None:
        if self._closing:
            return
        self._drain_process_logs()
        self._drain_ui_messages()
        self._update_controls()
        self.root.after(POLL_INTERVAL_MS, self._poll_runtime)

    def _drain_process_logs(self) -> None:
        for name, managed, widget in (
            ("device", self.controller.processes.device, self.device_log["text"]),
            ("saver", self.controller.processes.saver, self.saver_log["text"]),
        ):
            lines, offset = managed.read_logs_since(self._log_offsets[name])
            self._log_offsets[name] = offset
            if not lines:
                continue
            widget.configure(state=tk.NORMAL)
            widget.insert(tk.END, "".join(lines))
            widget.see(tk.END)
            widget.configure(state=tk.DISABLED)

    def _update_controls(self) -> None:
        session = self.controller.session.public_state()
        device = self.controller.processes.device
        saver = self.controller.processes.saver
        device_live = device.running or device.process is not None
        saver_live = saver.running or saver.process is not None
        self.device_start_btn.configure(
            state=tk.DISABLED if device_live or device.stopping else tk.NORMAL
        )
        self.device_stop_btn.configure(
            state=tk.NORMAL if device_live and not device.stopping else tk.DISABLED
        )
        can_start_saver = (
            device.ready_for_episode
            and not saver_live
            and not saver.stopping
            and bool(session["task_selected"])
        )
        self.saver_start_btn.configure(
            state=tk.NORMAL if can_start_saver else tk.DISABLED
        )
        self.saver_stop_btn.configure(
            state=tk.NORMAL if saver_live and not saver.stopping else tk.DISABLED
        )
        self.toggle_btn.configure(state=tk.NORMAL if saver.running else tk.DISABLED)
        self.discard_btn.configure(
            state=tk.NORMAL if saver.running and saver.recording else tk.DISABLED
        )
        if device.stopping:
            device_status = "停止中"
        elif device.running and not device.ready_for_episode:
            device_status = "启动中"
        else:
            device_status = "运行" if device.running else "停止"
        self.device_status_var.set(f"设备：{device_status}")
        if saver.stopping:
            saver_text = "停止中"
        elif saver.recording:
            saver_text = "采集中"
        elif saver.running:
            saver_text = "待机"
        else:
            saver_text = "停止"
        self.saver_status_var.set(f"保存：{saver_text}")
        self.saved_count_var.set(
            f"已保存 Episode：{saver.saved_episode_count} | 最近：{saver.last_episode_name}"
        )

    def _run_background(
        self,
        work: Callable[[], Any],
        on_success: Callable[[Any], None],
        *,
        finally_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        def runner() -> None:
            try:
                result = work()
            except Exception as exc:
                logger.error("I2RT DM3 operation failed: %s", exc)
                if not self._closing:
                    self.root.after(0, lambda error=exc: self._show_error(error))
            else:
                if not self._closing:
                    self.root.after(0, lambda value=result: on_success(value))
            finally:
                if finally_callback is not None and not self._closing:
                    self.root.after(0, finally_callback)

        threading.Thread(target=runner, daemon=True).start()

    def _log_message(self, message: str) -> None:
        self._ui_messages.put(message)

    def _drain_ui_messages(self) -> None:
        while True:
            try:
                message = self._ui_messages.get_nowait()
            except queue.Empty:
                return
            self.saver_log["text"].configure(state=tk.NORMAL)
            self.saver_log["text"].insert(tk.END, f"[GUI] {message}\n")
            self.saver_log["text"].see(tk.END)
            self.saver_log["text"].configure(state=tk.DISABLED)

    def _show_error(self, exc: Exception) -> None:
        messagebox.showerror("错误", str(exc), parent=self.root)

    def _on_close(self) -> None:
        processes = self.controller.processes
        if (processes.device.running or processes.saver.running) and not messagebox.askyesno(
            "退出",
            "采集进程仍在运行。确定停止设备与保存并退出吗？",
            parent=self.root,
        ):
            return
        self._closing = True
        try:
            self.upload_panel.shutdown(stop_manager=False)
            self.controller.shutdown()
        except Exception as exc:
            logger.warning("I2RT DM3 shutdown error: %s", exc)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    root = tk.Tk()
    I2RTDM3App(root).run()


if __name__ == "__main__":
    main()


__all__ = ["I2RTDM3App", "main"]
