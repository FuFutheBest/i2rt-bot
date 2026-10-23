#!/usr/bin/env python3
"""
consumer/data_saver_gui_r1.py - Tkinter GUI for the R1 NEDF3 saver.

A lightweight desktop GUI to drive R1 NEDF3 episode recording:

  * Fill in the NEDF3 metadata fields that change per session:
    ``task_name``, ``task_description``, ``device identifier``, ``user_id``,
    ``user_name``. They are written into each episode's metadata.json.
  * Press <Enter> (or the Start/Stop button) to start and then stop an episode.
  * Browse recorded episodes, replay the recorded video (reuses
    ``scripts/inspect_nedf3_episode.py``), and delete unwanted episode folders.

The actual data acquisition runs in a background thread via
``DataSaverConsumer.start_saving()`` in GUI control mode; this module only
sends start/stop/discard events and reads status.

Author: SuperInference
"""

import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logger_config import logger

from incubator.r1.r1_teleop_dm3_upload.dm3_api import (
    DEFAULT_BASE_URL,
    DM3ApiClient,
    DM3TaskExecution,
)

INSPECT_SCRIPT = REPO_ROOT / "incubator" / "r1" / "inspect_nedf3_episode.py"


class DataSaverGUI:
    """Tkinter GUI bound to a running ``DataSaverConsumer`` (gui_control=True)."""

    def __init__(self, consumer: Any, parent: Optional[Any] = None) -> None:
        self.consumer = consumer
        self.consumer.gui_control = True

        self._recording = False
        self._saver_thread: Optional[threading.Thread] = None
        self._saver_started = False

        # DM3 account / task state.
        self.client = DM3ApiClient()
        self.current_tasks: List[DM3TaskExecution] = []
        self.selected_task: Optional[DM3TaskExecution] = None

        # ``parent`` is None -> standalone window (owns its own Tk root and main
        # loop). Otherwise this panel is embedded into a host window (e.g. the
        # unified r1_control_gui) and only builds a Frame; the host owns the
        # toplevel, mainloop and window-close handling.
        self._embedded = parent is not None
        if self._embedded:
            self._tk = parent.winfo_toplevel()
            self.root = ttk.Frame(parent)
            self.root.pack(fill="both", expand=True)
        else:
            self._tk = tk.Tk()
            self._tk.title("R1 NEDF3 Data Saver")
            self._tk.geometry("1120x640")
            self._tk.minsize(960, 560)
            self.root = self._tk

        self._build_widgets()

        if not self._embedded:
            self._tk.protocol("WM_DELETE_WINDOW", self._on_close)

        # Enter / numpad-Enter toggles start/stop recording. Bound on the
        # toplevel in BOTH modes (a foot pedal that emits Enter can then drive
        # recording hands-free). Events from focused children bubble up to the
        # toplevel, so this catches Enter anywhere in the window; the multi-line
        # description box is exempted inside _on_enter. Rebind (not add="+") so
        # re-creating the embedded panel never stacks duplicate handlers.
        self._tk.bind("<Return>", self._on_enter)
        self._tk.bind("<KP_Enter>", self._on_enter)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # --- DM3 login + assigned tasks --------------------------------
        # Built first so the metadata form (user_id/user_name/task_name/
        # task_description StringVars) exists before any handler fires.
        # Kept full-width on top (the login/tasks rows are naturally wide).
        self._build_dm3_section(self.root, pad)

        # --- Two-column body -------------------------------------------
        # Left: metadata + storage. Right: recording controls + episode
        # browser. This keeps the (tall) recording/episodes section beside the
        # metadata instead of stacked below it, so the whole panel fits without
        # vertical clipping inside the unified control window.
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        # --- Metadata form ---------------------------------------------
        form = ttk.LabelFrame(left, text="Episode Metadata")
        form.pack(fill="x", **pad)

        self.var_task_name = tk.StringVar(value=getattr(self.consumer, "task_name", ""))
        self.var_device = tk.StringVar(value=getattr(self.consumer, "device_identifier", ""))
        self.var_user_id = tk.StringVar(value=getattr(self.consumer, "user_id", ""))
        self.var_user_name = tk.StringVar(value=getattr(self.consumer, "user_name", ""))

        # Task name / User ID / User name come from DM3 (login + selected task)
        # and are shown read-only; only "Device identifier" is hand-entered.
        ttk.Label(form, text="Task name").grid(row=0, column=0, sticky="e", **pad)
        self.ent_task_name = ttk.Entry(
            form, textvariable=self.var_task_name, width=40, state="readonly")
        self.ent_task_name.grid(row=0, column=1, sticky="we", **pad)

        ttk.Label(form, text="Device identifier").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(form, textvariable=self.var_device, width=40).grid(
            row=1, column=1, sticky="we", **pad)

        ttk.Label(form, text="User ID").grid(row=2, column=0, sticky="e", **pad)
        self.ent_user_id = ttk.Entry(
            form, textvariable=self.var_user_id, width=40, state="readonly")
        self.ent_user_id.grid(row=2, column=1, sticky="we", **pad)

        ttk.Label(form, text="User name").grid(row=3, column=0, sticky="e", **pad)
        self.ent_user_name = ttk.Entry(
            form, textvariable=self.var_user_name, width=40, state="readonly")
        self.ent_user_name.grid(row=3, column=1, sticky="we", **pad)

        ttk.Label(form, text="Task description").grid(row=4, column=0, sticky="ne", **pad)
        self.txt_desc = tk.Text(form, width=40, height=3, wrap="word")
        self.txt_desc.insert("1.0", getattr(self.consumer, "task_description", ""))
        # Read-only: filled from the selected DM3 task, never typed by the user.
        self.txt_desc.config(state="disabled")
        self.txt_desc.grid(row=4, column=1, sticky="we", **pad)

        # Entries whose value comes from DM3 and must never be hand-edited; their
        # state is managed as readonly/disabled (never "normal").
        self._readonly_entries = [self.ent_task_name, self.ent_user_id, self.ent_user_name]

        form.columnconfigure(1, weight=1)

        # --- Storage location ------------------------------------------
        storage = ttk.LabelFrame(left, text="Storage")
        storage.pack(fill="x", **pad)

        # Default the save path to <output_dir>/<YYYYMMDD> (e.g. saved_data/20260624)
        # so each day's episodes are grouped under a dated folder.
        self.var_output_dir = tk.StringVar(value=self._default_output_dir())
        ttk.Label(storage, text="Save path").grid(row=0, column=0, sticky="e", **pad)
        self.ent_output_dir = ttk.Entry(storage, textvariable=self.var_output_dir)
        self.ent_output_dir.grid(row=0, column=1, sticky="we", **pad)
        self.btn_browse_dir = ttk.Button(storage, text="Browse...", command=self._browse_output_dir)
        self.btn_browse_dir.grid(row=0, column=2, **pad)
        # Re-list episodes when the path changes (applied to consumer on start).
        self.var_output_dir.trace_add("write", lambda *_: self._refresh_episodes())
        storage.columnconfigure(1, weight=1)

        # --- Recording controls ----------------------------------------
        ctrl = ttk.LabelFrame(right, text="Recording  (press Enter to start / stop)")
        ctrl.pack(fill="x", **pad)

        self.btn_toggle = ttk.Button(ctrl, text="Start (Enter)", command=self._toggle_recording)
        self.btn_toggle.grid(row=0, column=0, **pad)

        self.btn_discard = ttk.Button(ctrl, text="Discard current", command=self._discard_recording,
                                      state="disabled")
        self.btn_discard.grid(row=0, column=1, **pad)

        self.lbl_status = ttk.Label(ctrl, text="\u25cf IDLE", foreground="gray")
        self.lbl_status.grid(row=0, column=2, sticky="w", **pad)

        self.lbl_info = ttk.Label(ctrl, text="")
        self.lbl_info.grid(row=1, column=0, columnspan=3, sticky="w", **pad)

        # --- Episode browser -------------------------------------------
        browser = ttk.LabelFrame(right, text="Recorded Episodes")
        browser.pack(fill="both", expand=True, **pad)

        # Pack the action buttons FIRST with side="bottom" so they always claim
        # space at the bottom of the browser; the list then fills whatever is
        # left. This way, if the panel is short on vertical room, the listbox
        # (which has a scrollbar) gets clipped instead of the buttons.
        btns = ttk.Frame(browser)
        btns.pack(side="bottom", fill="x", padx=8, pady=4)
        ttk.Button(btns, text="Refresh", command=self._refresh_episodes).pack(side="left", padx=4)
        ttk.Button(btns, text="Replay video", command=self._replay_selected).pack(side="left", padx=4)
        ttk.Button(btns, text="Delete", command=self._delete_selected).pack(side="left", padx=4)
        ttk.Button(btns, text="Open folder", command=self._open_output_dir).pack(side="left", padx=4)

        list_frame = ttk.Frame(browser)
        list_frame.pack(side="top", fill="both", expand=True, padx=8, pady=4)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=6)
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self._replay_selected())

        self._episode_dirs: List[Path] = []
        self._refresh_episodes()

    # ------------------------------------------------------------------ #
    # DM3 login + assigned tasks
    # ------------------------------------------------------------------ #
    def _build_dm3_section(self, parent: Any, pad: dict) -> None:
        # Login row.
        login = ttk.LabelFrame(parent, text="DM3 Login")
        login.pack(fill="x", **pad)

        ttk.Label(login, text="Server").grid(row=0, column=0, sticky="e", **pad)
        self.var_base_url = tk.StringVar(value=DEFAULT_BASE_URL)
        ttk.Entry(login, textvariable=self.var_base_url, width=28).grid(
            row=0, column=1, sticky="we", **pad)

        ttk.Label(login, text="Phone").grid(row=0, column=2, sticky="e", **pad)
        self.var_phone = tk.StringVar()
        ttk.Entry(login, textvariable=self.var_phone, width=16).grid(
            row=0, column=3, sticky="we", **pad)

        ttk.Label(login, text="Password").grid(row=0, column=4, sticky="e", **pad)
        self.var_password = tk.StringVar()
        ttk.Entry(login, textvariable=self.var_password, width=16, show="*").grid(
            row=0, column=5, sticky="we", **pad)

        self.btn_login = ttk.Button(login, text="Login", command=self._login)
        self.btn_login.grid(row=0, column=6, **pad)
        self.btn_logout = ttk.Button(login, text="Logout", command=self._logout, state="disabled")
        self.btn_logout.grid(row=0, column=7, **pad)

        self.var_login_status = tk.StringVar(value="Not logged in")
        ttk.Label(login, textvariable=self.var_login_status, foreground="gray").grid(
            row=1, column=0, columnspan=8, sticky="w", **pad)
        login.columnconfigure(1, weight=1)

        # Assigned tasks.
        tasks = ttk.LabelFrame(parent, text="My Tasks")
        tasks.pack(fill="x", **pad)

        ctl = ttk.Frame(tasks)
        ctl.pack(fill="x", padx=8, pady=4)
        ttk.Label(ctl, text="Execute day").pack(side="left")
        self.var_execute_day = tk.StringVar()
        ttk.Entry(ctl, textvariable=self.var_execute_day, width=12).pack(side="left", padx=4)
        ttk.Label(ctl, text="Page").pack(side="left", padx=(8, 0))
        self.var_page = tk.StringVar(value="1")
        ttk.Entry(ctl, textvariable=self.var_page, width=4).pack(side="left", padx=4)
        ttk.Label(ctl, text="Size").pack(side="left", padx=(8, 0))
        self.var_page_size = tk.StringVar(value="20")
        ttk.Entry(ctl, textvariable=self.var_page_size, width=4).pack(side="left", padx=4)
        self.btn_fetch_tasks = ttk.Button(
            ctl, text="Fetch tasks", command=self._fetch_tasks, state="disabled")
        self.btn_fetch_tasks.pack(side="left", padx=(8, 4))
        self.btn_select_task = ttk.Button(
            ctl, text="Use selected task", command=self._select_highlighted_task, state="disabled")
        self.btn_select_task.pack(side="left", padx=4)

        cols = ("id", "status", "name")
        self.task_tree = ttk.Treeview(tasks, columns=cols, show="headings", height=5)
        self.task_tree.heading("id", text="Exec ID")
        self.task_tree.heading("status", text="Status")
        self.task_tree.heading("name", text="Task name")
        self.task_tree.column("id", width=90, anchor="w")
        self.task_tree.column("status", width=110, anchor="w")
        self.task_tree.column("name", width=460, anchor="w")
        self.task_tree.pack(fill="x", padx=8, pady=(0, 4))
        self.task_tree.bind("<Double-Button-1>", lambda e: self._select_highlighted_task())

        self.var_selected_task = tk.StringVar(value="Selected task: none")
        ttk.Label(tasks, textvariable=self.var_selected_task, foreground="gray").pack(
            anchor="w", padx=8, pady=(0, 4))

    def _run_background(self, work: Callable[[], Any], on_success: Callable[[Any], None]) -> None:
        """Run ``work`` off the Tk thread; deliver result/error on the main loop."""
        def runner() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda error=exc: self._on_bg_error(error))
                return
            self.root.after(0, lambda: on_success(result))

        threading.Thread(target=runner, daemon=True).start()

    def _on_bg_error(self, exc: Exception) -> None:
        logger.error(f"GUI background task failed: {exc}")
        messagebox.showerror("Error", str(exc))

    def _login(self) -> None:
        phone = self.var_phone.get().strip()
        password = self.var_password.get()
        base_url = self.var_base_url.get().strip() or DEFAULT_BASE_URL
        if not phone or not password:
            messagebox.showwarning("Missing input", "Please fill in phone and password.")
            return
        self.btn_login.config(state="disabled")

        def work() -> str:
            self.client = DM3ApiClient(base_url=base_url)
            self.client.login(phone, password)
            return "ok"

        self._run_background(work, self._on_login_success)

    def _on_login_success(self, _result: Any) -> None:
        uid = self.client.user_id
        name = self.client.user_name
        # Auto-fill the metadata user fields from the logged-in account.
        if uid:
            self.var_user_id.set(uid)
        if name:
            self.var_user_name.set(name)
        self.var_login_status.set(
            f"Logged in  |  user_id={uid or '?'}  user_name={name or '?'}"
        )
        self.btn_login.config(state="disabled")
        self.btn_logout.config(state="normal")
        self.btn_fetch_tasks.config(state="normal")
        self.btn_select_task.config(state="normal")
        logger.info("GUI: DM3 login succeeded")

    def _logout(self) -> None:
        def work() -> str:
            try:
                self.client.logout()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"DM3 logout error (ignored): {exc}")
            return "ok"

        self._run_background(work, self._on_logout_success)

    def _on_logout_success(self, _result: Any) -> None:
        self.current_tasks = []
        self.selected_task = None
        self.task_tree.delete(*self.task_tree.get_children())
        self.var_selected_task.set("Selected task: none")
        self.var_login_status.set("Not logged in")
        self.btn_login.config(state="normal")
        self.btn_logout.config(state="disabled")
        self.btn_fetch_tasks.config(state="disabled")
        self.btn_select_task.config(state="disabled")

    def _fetch_tasks(self) -> None:
        try:
            page = int(self.var_page.get().strip() or "1")
            page_size = int(self.var_page_size.get().strip() or "20")
        except ValueError:
            messagebox.showwarning("Invalid input", "Page and Size must be integers.")
            return
        execute_day = self.var_execute_day.get().strip() or None

        def work() -> List[DM3TaskExecution]:
            response = self.client.get_my_task_executions(
                page=page, page_size=page_size, execute_day=execute_day)
            return self.client.normalize_task_executions(response)

        self._run_background(work, self._on_tasks_loaded)

    def _on_tasks_loaded(self, tasks: List[DM3TaskExecution]) -> None:
        self.current_tasks = tasks
        self.task_tree.delete(*self.task_tree.get_children())
        for idx, task in enumerate(tasks):
            self.task_tree.insert(
                "", tk.END, iid=str(idx),
                values=(task.id or "", task.status or "-", task.name),
            )
        logger.info(f"GUI: loaded {len(tasks)} DM3 task(s)")
        if not tasks:
            messagebox.showinfo("No tasks", "No assigned tasks were returned.")

    def _select_highlighted_task(self) -> None:
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("No selection", "Please select a task row first.")
            return
        task = self.current_tasks[int(selection[0])]
        self.selected_task = task
        # Auto-fill task name + description into the (read-only) metadata form.
        self.var_task_name.set(task.name)
        self._set_desc_text(task.description)
        self.var_selected_task.set(f"Selected task: {task.id} / {task.name}")
        logger.info(f"GUI: selected task {task.id} / {task.name}")

    def _set_desc_text(self, text: str) -> None:
        """Programmatically replace the read-only description box content."""
        self.txt_desc.config(state="normal")
        self.txt_desc.delete("1.0", "end")
        if text:
            self.txt_desc.insert("1.0", text)
        self.txt_desc.config(state="disabled")

    # ------------------------------------------------------------------ #
    # Recording control
    # ------------------------------------------------------------------ #
    def _on_enter(self, event: Any) -> Optional[str]:
        # Ignore Enter raised from within the multi-line description box.
        if event.widget is self.txt_desc:
            return None
        self._toggle_recording()
        return "break"

    def _apply_metadata(self) -> None:
        out = self.var_output_dir.get().strip()
        if out:
            out_path = str(Path(out).expanduser())
            self.consumer.output_dir = out_path
            os.makedirs(out_path, exist_ok=True)
        self.consumer.set_metadata_fields(
            task_name=self.var_task_name.get().strip(),
            task_description=self.txt_desc.get("1.0", "end").strip(),
            device_identifier=self.var_device.get().strip(),
            user_id=self.var_user_id.get().strip(),
            user_name=self.var_user_name.get().strip(),
        )

    def _toggle_recording(self) -> None:
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        if not self.var_task_name.get().strip():
            messagebox.showwarning("Missing field", "Please fill in the task name first.")
            return
        self._apply_metadata()
        self.consumer.gui_start_episode()
        self._recording = True
        self.btn_toggle.config(text="Stop (Enter)")
        self.btn_discard.config(state="normal")
        self._set_form_state("disabled")
        logger.info("GUI: episode start requested")

    def _stop_recording(self) -> None:
        self.consumer.gui_stop_episode()
        self._finish_recording_ui()
        logger.info("GUI: episode stop requested")
        # Give the saver a moment to finalize, then refresh the list.
        self.root.after(1500, self._refresh_episodes)

    def _discard_recording(self) -> None:
        if not self._recording:
            return
        self.consumer.gui_discard_episode()
        self._finish_recording_ui()
        logger.info("GUI: episode discard requested")

    def _finish_recording_ui(self) -> None:
        self._recording = False
        self.btn_toggle.config(text="Start (Enter)")
        self.btn_discard.config(state="disabled")
        self._set_form_state("normal")

    def _set_form_state(self, state: str) -> None:
        # Toggle entry widgets. DM3-sourced fields are never made editable: they
        # stay "readonly" when the form is active and "disabled" while recording.
        for child in self._iter_entries():
            try:
                if child in self._readonly_entries:
                    child.config(state=("disabled" if state == "disabled" else "readonly"))
                else:
                    child.config(state=state)
            except tk.TclError:
                pass
        # Task description is always read-only (filled from the selected task).
        self.txt_desc.config(state="disabled")
        # The save path must stay fixed while an episode is recording.
        self.btn_browse_dir.config(state=state)

    def _iter_entries(self) -> List[Any]:
        result: List[Any] = []

        def walk(widget: Any) -> None:
            for child in widget.winfo_children():
                if isinstance(child, ttk.Entry):
                    result.append(child)
                walk(child)

        walk(self.root)
        return result

    # ------------------------------------------------------------------ #
    # Episode browser
    # ------------------------------------------------------------------ #
    def _default_output_dir(self) -> str:
        """``<consumer output_dir>/<YYYYMMDD>`` (today), e.g. ``saved_data/20260624``.

        If the consumer's output_dir already ends with a today's-date folder
        (e.g. resumed run), it is reused as-is instead of nesting another date.
        """
        base = Path(getattr(self.consumer, "output_dir", "saved_data"))
        today = datetime.now().strftime("%Y%m%d")
        if base.name == today:
            return str(base)
        return str(base / today)

    def _output_dir(self) -> Path:
        path = self.var_output_dir.get().strip()
        if path:
            return Path(path).expanduser()
        return Path(getattr(self.consumer, "output_dir", "saved_data"))

    def _browse_output_dir(self) -> None:
        if self._recording:
            return
        current = self.var_output_dir.get().strip()
        initial = current if current and os.path.isdir(current) else str(REPO_ROOT)
        chosen = filedialog.askdirectory(title="Select save directory", initialdir=initial)
        if chosen:
            self.var_output_dir.set(chosen)

    def _refresh_episodes(self) -> None:
        self.listbox.delete(0, tk.END)
        self._episode_dirs = []
        out_dir = self._output_dir()
        if not out_dir.exists():
            return
        episodes = []
        for entry in out_dir.iterdir():
            if not entry.is_dir():
                continue
            # Skip in-progress temp dirs; only show finalized episodes.
            if entry.name.endswith("_temp"):
                continue
            if not (entry / "metadata.json").exists():
                continue
            episodes.append(entry)
        episodes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for entry in episodes:
            self._episode_dirs.append(entry)
            self.listbox.insert(tk.END, entry.name)

    def _selected_episode(self) -> Optional[Path]:
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Please select an episode in the list first.")
            return None
        return self._episode_dirs[sel[0]]

    def _replay_selected(self) -> None:
        episode = self._selected_episode()
        if episode is None:
            return
        if not INSPECT_SCRIPT.exists():
            messagebox.showerror(
                "Replay unavailable",
                f"Inspection script not found:\n{INSPECT_SCRIPT}",
            )
            return

        def _run() -> None:
            try:
                subprocess.run(
                    [sys.executable, str(INSPECT_SCRIPT), str(episode)],
                    cwd=str(REPO_ROOT),
                    check=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Replay failed for {episode}: {exc}")

        threading.Thread(target=_run, daemon=True).start()
        logger.info(f"GUI: replaying {episode}")

    def _delete_selected(self) -> None:
        episode = self._selected_episode()
        if episode is None:
            return
        if self._recording:
            messagebox.showwarning("Busy", "Stop the current recording before deleting episodes.")
            return
        if not messagebox.askyesno(
            "Confirm delete",
            f"Delete this episode folder permanently?\n\n{episode}",
        ):
            return
        try:
            shutil.rmtree(episode)
            logger.info(f"GUI: deleted episode {episode}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Delete failed", f"Could not delete:\n{episode}\n\n{exc}")
        self._refresh_episodes()

    def _open_output_dir(self) -> None:
        out_dir = self._output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(out_dir)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out_dir)])
            elif sys.platform.startswith("win"):
                os.startfile(str(out_dir))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open failed", f"Could not open:\n{out_dir}\n\n{exc}")

    # ------------------------------------------------------------------ #
    # Status polling + lifecycle
    # ------------------------------------------------------------------ #
    def _poll_status(self) -> None:
        active = bool(getattr(self.consumer, "episode_active", False))
        if active:
            self.lbl_status.config(text="\u25cf RECORDING", foreground="red")
        elif self._recording:
            self.lbl_status.config(text="\u25cf STARTING...", foreground="orange")
        else:
            self.lbl_status.config(text="\u25cf IDLE", foreground="gray")

        out_dir = self._output_dir()
        n = len(self._episode_dirs)
        last = getattr(self.consumer, "last_finalized_episode_dir", None)
        info = f"Output: {out_dir}   |   episodes: {n}"
        if last:
            info += f"   |   last: {Path(last).name}"
        self.lbl_info.config(text=info)

        self.root.after(500, self._poll_status)

    def _on_close(self) -> None:
        if self._recording:
            if not messagebox.askyesno(
                "Recording in progress",
                "An episode is still recording. Stop and quit?",
            ):
                return
            self.consumer.gui_stop_episode()
        self.consumer.request_shutdown()
        self.root.after(300, self.root.destroy)

    def start_saver_thread(self) -> None:
        """Start the saver loop + status polling (idempotent).

        Split out of ``run()`` so an embedding host (r1_control_gui) can defer
        starting the saver until devices are up and the summary SHM is ready.
        """
        if self._saver_started:
            return
        self._saver_started = True
        # Start the saver loop (blocks on GUI start events) in the background.
        self._saver_thread = threading.Thread(
            target=self.consumer.start_saving, daemon=True
        )
        self._saver_thread.start()
        self._poll_status()

    def shutdown_saver(self) -> None:
        """Stop recording (if any) and ask the consumer to shut down.

        For the embedded host to call on close; does NOT destroy any widgets
        (the host owns the toplevel lifecycle).
        """
        if self._embedded:
            # Remove our Enter bindings from the shared toplevel so they don't
            # fire into this (about-to-be-destroyed) panel.
            for seq in ("<Return>", "<KP_Enter>"):
                try:
                    self._tk.unbind(seq)
                except Exception:  # noqa: BLE001
                    pass
        try:
            if self._recording:
                self.consumer.gui_stop_episode()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.consumer.request_shutdown()
        except Exception:  # noqa: BLE001
            pass

    def run(self) -> None:
        self.start_saver_thread()
        self._tk.mainloop()


def run_gui_app(consumer: Any) -> None:
    """Build and run the Tk GUI bound to ``consumer`` (blocks until window closes)."""
    gui = DataSaverGUI(consumer)
    gui.run()
