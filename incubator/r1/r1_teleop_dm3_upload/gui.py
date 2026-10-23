#!/usr/bin/env python3
"""DM3 云端上传图形界面(Linux 版,移植自 windows_DM3/gui.py)。

流程与 Windows GUI 一致:
    1. DM3 登录(手机号 + 密码)
    2. 拉取 / 选择我的任务
    3. 选择本地数据目录,绑定 upload.json,上传已完成的 <uuid> episode 目录到阿里云 OSS

运行:
    python3 scripts/diagnostics/runtime_profiles.py run \
        --config-name quest3_collect_r1 --role upload -- \
        -m incubator.r1.r1_teleop_dm3_upload.gui

需要带 Tk 的 Python(Debian/Ubuntu: ``sudo apt install python3-tk``),
实际上传还需要 ``oss2``(``pip install oss2``)。
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger_config import logger

from incubator.r1.r1_teleop_dm3_upload.dm3_api import DEFAULT_BASE_URL, DM3ApiClient, DM3TaskExecution
from incubator.r1.r1_teleop_dm3_upload.oss_uploader import (
    UploadProgress,
    UploadResult,
    upload_episode_data_dir,
)
from incubator.r1.r1_teleop_dm3_upload.upload_state import (
    load_upload_state,
    update_upload_state,
    utc_now_iso,
    write_upload_state,
)

# 字体:用带完整中文字形的 Noto CJK,避免中文回退导致发虚。
UI_FONT_FAMILY = "WenQuanYi Micro Hei"          # 界面字体(小字号 hinting 工整)
LOG_FONT_FAMILY = "WenQuanYi Micro Hei Mono"    # 日志等宽字体(含中文)
UI_FONT_SIZE = 11  # 视觉"号数"(11 号)


def _px(pt: int) -> int:
    """把"点号"转成负数像素值。

    Tk 默认会按屏幕物理尺寸做分数缩放(本机约 1.125),正数字号会被换算成
    非整数像素 -> 边缘发虚、粗细不均。负数字号直接以"像素"为单位,绕过缩放,
    渲染为精确整数像素,字最锐利。这里按标准 96 dpi 把点号折算成像素。
    """
    return -int(round(pt * 96 / 72))


UI_FONT_PX = _px(UI_FONT_SIZE)
LOG_FONT = (LOG_FONT_FAMILY, UI_FONT_PX)
DEFAULT_DATA_DIR = PROJECT_ROOT / "saved_data"
DEFAULT_RESOURCE_TYPE = "teleoperation_r1"


def _default_data_dir() -> Path:
    """``saved_data/<YYYYMMDD>`` (today), matching the data-saver GUI's layout.

    Each day's episodes live under a dated folder, so the upload GUI defaults to
    the same dated path the saver writes to.
    """
    return DEFAULT_DATA_DIR / datetime.now().strftime("%Y%m%d")


def _fmt_mb(num_bytes: int) -> str:
    """Format a byte count as a compact MiB string for the progress label."""
    return f"{num_bytes / (1024 * 1024):.1f}"


class DM3UploadGUI:
    """把本地 episode 绑定到 DM3 任务并上传到 OSS 的桌面界面。"""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DM3 数据上传 (Linux)")
        self.root.geometry("980x760")
        self.root.minsize(820, 640)

        self.client = DM3ApiClient()
        self.current_tasks: List[DM3TaskExecution] = []
        self.selected_task: Optional[DM3TaskExecution] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        # Latest upload-progress snapshot, written by the worker thread and read
        # by the Tk thread's periodic flusher (single reference swap is atomic).
        self._upload_progress: Optional[UploadProgress] = None

        self._setup_fonts()
        self._build_ui()
        self._schedule_log_flush()
        self._schedule_progress_flush()

    def _setup_fonts(self) -> None:
        """统一界面字体为清晰的中文字体,并消除分数缩放导致的模糊。"""
        import tkinter.font as tkfont

        # 关键:把 Tk 的分数缩放(本机默认 ~1.125)强制为 1.0,
        # 配合下面的"负数像素字号",字体按精确整数像素渲染,不再发虚。
        try:
            self.root.tk.call("tk", "scaling", 1.0)
        except Exception:
            pass

        # 把 Tk 内置的命名字体全部换成 Noto CJK(像素字号),所有控件随之更新。
        for name in (
            "TkDefaultFont",
            "TkTextFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkIconFont",
            "TkTooltipFont",
        ):
            try:
                f = tkfont.nametofont(name)
                f.configure(family=UI_FONT_FAMILY, size=UI_FONT_PX)
            except Exception:
                pass

        try:
            tkfont.nametofont("TkFixedFont").configure(family=LOG_FONT_FAMILY, size=UI_FONT_PX)
        except Exception:
            pass

        # ttk 控件(Label/Button/Treeview 等)显式指定字体,确保生效。
        row_h = int(round(-UI_FONT_PX * 2.0))
        style = ttk.Style(self.root)
        style.configure(".", font=(UI_FONT_FAMILY, UI_FONT_PX))
        style.configure("Treeview", font=(UI_FONT_FAMILY, UI_FONT_PX), rowheight=row_h)
        style.configure("Treeview.Heading", font=(UI_FONT_FAMILY, UI_FONT_PX, "bold"))
        style.configure("TLabelframe.Label", font=(UI_FONT_FAMILY, UI_FONT_PX, "bold"))

    def run(self) -> None:
        """运行 tkinter 事件循环。"""
        self.root.mainloop()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        self._build_login_section(main)
        self._build_task_section(main)
        self._build_upload_section(main)
        self._build_log_section(main)

    def _build_login_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1. DM3 登录", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(frame, text="服务地址:").grid(row=0, column=0, sticky=tk.W)
        self.base_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        ttk.Entry(frame, textvariable=self.base_url_var, width=38).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(frame, text="手机号:").grid(row=0, column=2, sticky=tk.W, padx=(12, 0))
        self.phone_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.phone_var, width=18).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(frame, text="密码:").grid(row=0, column=4, sticky=tk.W, padx=(12, 0))
        self.password_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.password_var, width=18, show="*").grid(row=0, column=5, sticky=tk.W, padx=4)

        self.login_btn = ttk.Button(frame, text="登录", command=self._login)
        self.login_btn.grid(row=0, column=6, padx=(12, 4))
        self.logout_btn = ttk.Button(frame, text="登出", command=self._logout, state=tk.DISABLED)
        self.logout_btn.grid(row=0, column=7, padx=4)

        self.login_status_var = tk.StringVar(value="未登录")
        ttk.Label(frame, textvariable=self.login_status_var, foreground="gray").grid(
            row=1, column=0, columnspan=8, sticky=tk.W, pady=(6, 0)
        )

    def _build_task_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2. 我的任务", padding=8)
        frame.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        control = ttk.Frame(frame)
        control.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(control, text="执行日期:").pack(side=tk.LEFT)
        self.execute_day_var = tk.StringVar()
        ttk.Entry(control, textvariable=self.execute_day_var, width=14).pack(side=tk.LEFT, padx=4)
        ttk.Label(control, text="页码:").pack(side=tk.LEFT, padx=(12, 0))
        self.page_var = tk.StringVar(value="1")
        ttk.Entry(control, textvariable=self.page_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(control, text="每页数量:").pack(side=tk.LEFT, padx=(12, 0))
        self.page_size_var = tk.StringVar(value="10")
        ttk.Entry(control, textvariable=self.page_size_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(control, text="拉取任务", command=self._fetch_tasks).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(control, text="选中此任务", command=self._select_highlighted_task).pack(
            side=tk.LEFT, padx=4
        )

        columns = ("id", "status", "name")
        self.task_tree = ttk.Treeview(frame, columns=columns, show="headings", height=7)
        self.task_tree.heading("id", text="执行 ID")
        self.task_tree.heading("status", text="状态")
        self.task_tree.heading("name", text="名称")
        self.task_tree.column("id", width=120, anchor=tk.W)
        self.task_tree.column("status", width=120, anchor=tk.W)
        self.task_tree.column("name", width=560, anchor=tk.W)
        self.task_tree.pack(fill=tk.X)

        self.selected_task_var = tk.StringVar(value="已选任务: 无")
        ttk.Label(frame, textvariable=self.selected_task_var, foreground="gray").pack(anchor=tk.W, pady=(6, 0))

    def _build_upload_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3. 本地数据与上传", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row1, text="数据目录:").pack(side=tk.LEFT)
        self.data_dir_var = tk.StringVar(value=str(_default_data_dir()))
        ttk.Entry(row1, textvariable=self.data_dir_var, width=70).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(row1, text="浏览...", command=self._browse_data_dir).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="资源类型:").pack(side=tk.LEFT)
        self.resource_type_var = tk.StringVar(value=DEFAULT_RESOURCE_TYPE)
        ttk.Entry(row2, textvariable=self.resource_type_var, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="绑定 upload.json", command=self._bind_upload_json).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(row2, text="查看 upload.json", command=self._show_upload_json).pack(side=tk.LEFT, padx=4)
        self.upload_btn = ttk.Button(row2, text="上传到 OSS", command=self._upload_to_oss)
        self.upload_btn.pack(side=tk.LEFT, padx=4)
        # When checked, re-upload every file even if it already exists on OSS.
        self.force_upload_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="强制重传(不跳过已传)",
                        variable=self.force_upload_var).pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=(8, 0))
        self.upload_progress_var = tk.DoubleVar(value=0.0)
        self.upload_progressbar = ttk.Progressbar(
            row3, orient=tk.HORIZONTAL, mode="determinate",
            maximum=100.0, variable=self.upload_progress_var,
        )
        self.upload_progressbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.upload_progress_text_var = tk.StringVar(value="尚未开始")
        ttk.Label(row3, textvariable=self.upload_progress_text_var, width=40).pack(
            side=tk.LEFT, padx=(8, 0)
        )

    def _build_log_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="日志", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            frame,
            wrap=tk.WORD,
            font=LOG_FONT,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            height=16,
            state=tk.DISABLED,
        )
        scrollbar = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _login(self) -> None:
        phone = self.phone_var.get().strip()
        password = self.password_var.get()
        base_url = self.base_url_var.get().strip() or DEFAULT_BASE_URL
        if not phone or not password:
            messagebox.showwarning("缺少输入", "请填写手机号和密码。")
            return

        def work() -> str:
            self.client = DM3ApiClient(base_url=base_url)
            self.client.login(phone, password)
            return "登录成功"

        self._run_background(work, self._on_login_success)

    def _on_login_success(self, message: str) -> None:
        self.login_status_var.set(message)
        self.login_btn.configure(state=tk.DISABLED)
        self.logout_btn.configure(state=tk.NORMAL)
        self._log(message)

    def _logout(self) -> None:
        def work() -> str:
            self.client.logout()
            return "已登出"

        self._run_background(work, self._on_logout_success)

    def _on_logout_success(self, message: str) -> None:
        self.current_tasks = []
        self.selected_task = None
        self.task_tree.delete(*self.task_tree.get_children())
        self.selected_task_var.set("已选任务: 无")
        self.login_status_var.set("未登录")
        self.login_btn.configure(state=tk.NORMAL)
        self.logout_btn.configure(state=tk.DISABLED)
        self._log(message)

    def _fetch_tasks(self) -> None:
        page = int(self.page_var.get().strip() or "1")
        page_size = int(self.page_size_var.get().strip() or "10")
        execute_day = self.execute_day_var.get().strip() or None

        def work() -> List[DM3TaskExecution]:
            response = self.client.get_my_task_executions(page=page, page_size=page_size, execute_day=execute_day)
            return self.client.normalize_task_executions(response)

        self._run_background(work, self._on_tasks_loaded)

    def _on_tasks_loaded(self, tasks: List[DM3TaskExecution]) -> None:
        self.current_tasks = tasks
        self.task_tree.delete(*self.task_tree.get_children())
        for idx, task in enumerate(tasks):
            self.task_tree.insert("", tk.END, iid=str(idx), values=(task.id or "", task.status or "-", task.name))
        self._log(f"已加载 {len(tasks)} 个任务")

    def _select_highlighted_task(self) -> None:
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("未选择任务", "请先在列表中选择一行任务。")
            return
        idx = int(selection[0])
        self.selected_task = self.current_tasks[idx]
        self.selected_task_var.set(f"已选任务: {self.selected_task.id} / {self.selected_task.name}")
        self._log(f"已选任务: {self.selected_task.id} / {self.selected_task.name}")

    def _browse_data_dir(self) -> None:
        initial = self.data_dir_var.get().strip() or str(_default_data_dir())
        # The dated default may not exist yet; open the nearest existing parent.
        initial_path = Path(initial)
        while not initial_path.exists() and initial_path != initial_path.parent:
            initial_path = initial_path.parent
        chosen = filedialog.askdirectory(initialdir=str(initial_path), title="选择本地数据目录")
        if chosen:
            self.data_dir_var.set(chosen)

    def _bind_upload_json(self) -> None:
        if self.selected_task is None:
            messagebox.showwarning("未选择任务", "请先选择一个 DM3 任务。")
            return
        data_dir = self._data_dir()
        resource_type = self._resource_type()
        state_path = update_upload_state(
            data_dir,
            status="pending",
            task_execution=self.selected_task.raw,
            note="从 DM3 上传界面绑定",
            resource_type=resource_type,
        )
        self._log(f"已写入 {state_path}")

    def _show_upload_json(self) -> None:
        data_dir = self._data_dir()
        state = load_upload_state(data_dir)
        if not state:
            messagebox.showinfo("upload.json", f"未在 {data_dir} 找到 upload.json")
            return
        self._log_json(state)

    def _upload_to_oss(self) -> None:
        data_dir = self._data_dir()
        resource_type = self._resource_type()
        force = bool(self.force_upload_var.get())
        force_note = "\n\n强制重传: 已在 OSS 上的同名文件也会重新上传。" if force else ""
        if not messagebox.askyesno(
            "确认上传",
            "将上传以下目录中已完成的 <uuid> episode 文件夹:\n"
            f"{data_dir}\n\n"
            "OSS 命名: 每个 <uuid> 目录上传为 <uuid>/...;*_temp 目录会被跳过。\n\n"
            f"资源类型 resource_type={resource_type}"
            f"{force_note}",
        ):
            return

        self.upload_btn.configure(state=tk.DISABLED)
        self._reset_upload_progress()

        def work() -> UploadResult:
            update_upload_state(
                data_dir,
                status="uploading",
                task_execution=self.selected_task.raw if self.selected_task else None,
                note=f"正在上传到 DM3 OSS resource_type={resource_type}",
                resource_type=resource_type,
            )
            sts_token = self.client.fetch_oss_sts_token(resource_type=resource_type)
            try:
                result = upload_episode_data_dir(
                    data_dir,
                    sts_token,
                    progress=self._log,
                    on_progress=self._on_upload_progress,
                    token_provider=lambda: self.client.fetch_oss_sts_token(
                        resource_type=resource_type
                    ),
                    skip_existing=not force,
                )
            except Exception as exc:
                update_upload_state(
                    data_dir,
                    status="failed",
                    note=f"上传失败: {exc}",
                    resource_type=resource_type,
                )
                raise

            state = load_upload_state(data_dir)
            state.update(
                {
                    "status": "success",
                    "resource_type": resource_type,
                    "updated_at": utc_now_iso(),
                    "oss": {
                        "bucket": result.bucket,
                        "endpoint": result.endpoint,
                        "prefix": result.prefix,
                        "file_count": len(result.files),
                        "files": [
                            {
                                "local_path": item.local_path,
                                "object_key": item.object_key,
                                "size_bytes": item.size_bytes,
                            }
                            for item in result.files
                        ],
                    },
                }
            )
            write_upload_state(data_dir, state)
            return result

        self._run_background(work, self._on_upload_success, finally_callback=self._enable_upload_button)

    def _on_upload_success(self, result: UploadResult) -> None:
        self._set_progress_complete(len(result.files))
        if result.files:
            first_key = result.files[0].object_key
            self._log(
                f"上传完成: 共 {len(result.files)} 个文件。"
                f"示例对象: oss://{result.bucket}/{first_key}"
            )
        else:
            self._log(f"上传完成: 0 个文件 -> oss://{result.bucket}/{result.prefix}")
        messagebox.showinfo("上传完成", f"已上传 {len(result.files)} 个文件。")

    def _enable_upload_button(self) -> None:
        self.upload_btn.configure(state=tk.NORMAL)

    # ---- upload progress bar -------------------------------------------------
    def _on_upload_progress(self, snapshot: UploadProgress) -> None:
        """Worker-thread hook: just stash the latest snapshot (atomic reference
        swap); the Tk-thread flusher renders it so we never touch widgets here."""
        self._upload_progress = snapshot

    def _reset_upload_progress(self) -> None:
        self._upload_progress = None
        self.upload_progress_var.set(0.0)
        self.upload_progress_text_var.set("准备上传...")

    def _set_progress_complete(self, file_count: int) -> None:
        self._upload_progress = None
        self.upload_progress_var.set(100.0)
        self.upload_progress_text_var.set(f"完成 · {file_count} 个文件")

    def _schedule_progress_flush(self) -> None:
        self._flush_progress()
        self.root.after(100, self._schedule_progress_flush)

    def _flush_progress(self) -> None:
        snapshot = self._upload_progress
        if snapshot is None:
            return
        total = max(1, snapshot.total_bytes)
        percent = min(100.0, snapshot.uploaded_bytes / total * 100.0)
        self.upload_progress_var.set(percent)
        name = Path(snapshot.current_file).name if snapshot.current_file else ""
        self.upload_progress_text_var.set(
            f"{percent:5.1f}%  ·  {snapshot.uploaded_files}/{snapshot.total_files} 文件"
            f"  ·  {_fmt_mb(snapshot.uploaded_bytes)}/{_fmt_mb(snapshot.total_bytes)} MB"
            + (f"  ·  {name}" if name else "")
        )

    def _data_dir(self) -> Path:
        raw = self.data_dir_var.get().strip().strip('"')
        if not raw:
            raise ValueError("数据目录不能为空。")
        return Path(raw).expanduser().resolve()

    def _resource_type(self) -> str:
        resource_type = self.resource_type_var.get().strip()
        if not resource_type:
            raise ValueError("资源类型不能为空。")
        return resource_type

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
                logger.error(f"DM3 界面操作失败: {exc}")
                self.root.after(0, lambda error=exc: self._show_error(error))
            else:
                self.root.after(0, lambda value=result: on_success(value))
            finally:
                if finally_callback is not None:
                    self.root.after(0, finally_callback)

        threading.Thread(target=runner, daemon=True).start()

    def _show_error(self, exc: Exception) -> None:
        self._log(f"错误: {exc}")
        messagebox.showerror("错误", str(exc))

    def _log(self, message: str) -> None:
        self.log_queue.put(f"{message}\n")

    def _log_json(self, data: Dict[str, Any]) -> None:
        import json

        self._log(json.dumps(data, ensure_ascii=False, indent=2))

    def _schedule_log_flush(self) -> None:
        self._flush_logs()
        self.root.after(50, self._schedule_log_flush)

    def _flush_logs(self) -> None:
        lines: List[str] = []
        while True:
            try:
                lines.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if not lines:
            return
        self.log_text.configure(state=tk.NORMAL)
        for line in lines:
            self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> None:
    """GUI 入口。"""
    DM3UploadGUI().run()


if __name__ == "__main__":
    main()
