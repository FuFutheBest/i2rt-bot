#!/usr/bin/env python3
"""
Data Collection GUI - Unified launcher for device startup and data saving.

Provides a tkinter-based GUI to:
  1. Start/stop `windows_device_start.py` with selectable Hydra config.
  2. Start/stop `incubator/r1/async_data_saver_NEDF3_r1.py` with configurable
     output directory and start index.
  3. Display real-time stdout/stderr of both processes in split panels
     with ANSI color rendering and emoji support.
  4. Press Enter in the Data Saver output panel to control episode
     start/stop, exactly like pressing Enter in a terminal.

Usage:
    python scripts/analysis/data_collection_gui.py

Author: Yahong Wang, Assistant
"""

import os
import re
import sys
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_NAME = "windows_async_webcam"
DEFAULT_OUTPUT_BASE = os.path.join("data")

_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")

_ANSI_COLOR_MAP: Dict[int, str] = {
    30: "#000000", 31: "#cc0000", 32: "#4e9a06", 33: "#c4a000",
    34: "#3465a4", 35: "#75507b", 36: "#06989a", 37: "#d3d7cf",
    90: "#555753", 91: "#ef2929", 92: "#8ae234", 93: "#fce94f",
    94: "#729fcf", 95: "#ad7fa8", 96: "#34e2e2", 97: "#eeeeec",
}


def _scan_config_files() -> List[str]:
    """Return Hydra names for top-level and OpenPI primary configs."""
    if not CONFIGS_DIR.is_dir():
        return [DEFAULT_CONFIG_NAME]

    config_paths = list(CONFIGS_DIR.glob("*.yaml"))
    openpi_dir = CONFIGS_DIR / "openpi"
    if openpi_dir.is_dir():
        config_paths.extend(openpi_dir.rglob("*.yaml"))

    names = sorted(
        path.relative_to(CONFIGS_DIR).with_suffix("").as_posix()
        for path in config_paths
    )
    return names if names else [DEFAULT_CONFIG_NAME]


def _parse_ansi_line(line: str) -> List[Tuple[str, str]]:
    """Parse a line with ANSI color codes into (text, color_hex) segments.

    Returns a list of tuples. color_hex is "" for default foreground.
    """
    segments: List[Tuple[str, str]] = []
    current_color = ""
    pos = 0

    for m in _ANSI_RE.finditer(line):
        if m.start() > pos:
            segments.append((line[pos:m.start()], current_color))

        codes = m.group(1).split(";") if m.group(1) else ["0"]
        for code_str in codes:
            code = int(code_str) if code_str.isdigit() else 0
            if code == 0:
                current_color = ""
            elif code in _ANSI_COLOR_MAP:
                current_color = _ANSI_COLOR_MAP[code]
            elif code == 1:
                pass  # bold — ignore, keep current color

        pos = m.end()

    if pos < len(line):
        segments.append((line[pos:], current_color))

    return segments


class ProcessPanel(ttk.LabelFrame):
    """Reusable widget: scrollable log for one subprocess with ANSI color support.

    - Reader threads push raw lines into a thread-safe deque.
    - A periodic timer (50 ms) drains the deque in batches, parses ANSI colors,
      and does one bulk insert + scroll.
    - Emoji / unicode characters are preserved as-is.
    """

    _FLUSH_INTERVAL_MS = 50
    _MAX_LINES = 8000

    def __init__(self, master: tk.Widget, title: str, **kw) -> None:
        super().__init__(master, text=title, **kw)
        self.process: Optional[subprocess.Popen] = None
        self._reader_threads: List[threading.Thread] = []
        self._title = title

        self._pending: deque = deque()
        self._lock = threading.Lock()
        self._color_tags: Dict[str, bool] = {}

        self._build_log_area()
        self._schedule_flush()

    def _build_log_area(self) -> None:
        log_frame = ttk.Frame(self)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            selectbackground="#264f78",
            state=tk.DISABLED,
            height=18,
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.log_text.tag_configure("stderr", foreground="#f44747")

    def _ensure_color_tag(self, color_hex: str) -> str:
        """Lazily create a tk Text tag for a given hex color."""
        tag_name = f"c_{color_hex.lstrip('#')}"
        if tag_name not in self._color_tags:
            self.log_text.tag_configure(tag_name, foreground=color_hex)
            self._color_tags[tag_name] = True
        return tag_name

    # ---- batched flush ------------------------------------------------------

    def _schedule_flush(self) -> None:
        self._flush_pending()
        self.after(self._FLUSH_INTERVAL_MS, self._schedule_flush)

    def _flush_pending(self) -> None:
        """Drain all queued lines in one shot to minimise tk redraws."""
        with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            self._pending.clear()

        self.log_text.configure(state=tk.NORMAL)
        for raw_line, is_stderr in batch:
            if is_stderr:
                self.log_text.insert(tk.END, raw_line, "stderr")
            else:
                segments = _parse_ansi_line(raw_line)
                for text, color in segments:
                    if color:
                        tag = self._ensure_color_tag(color)
                        self.log_text.insert(tk.END, text, tag)
                    else:
                        self.log_text.insert(tk.END, text)

        total_lines = int(self.log_text.index("end-1c").split(".")[0])
        if total_lines > self._MAX_LINES:
            self.log_text.delete("1.0", f"{total_lines - self._MAX_LINES}.0")

        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ---- log helpers --------------------------------------------------------

    def append_log(self, raw_line: str, is_stderr: bool = False) -> None:
        """Thread-safe: push raw line to deque, flushed by periodic timer."""
        with self._lock:
            self._pending.append((raw_line, is_stderr))

    def clear_log(self) -> None:
        with self._lock:
            self._pending.clear()
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ---- process management -------------------------------------------------

    def stop_process(self) -> None:
        if not self.is_running():
            return
        try:
            if sys.platform == "win32":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

        threading.Thread(target=self._force_kill_after_timeout, daemon=True).start()

    def _force_kill_after_timeout(self, timeout: float = 8.0) -> None:
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _read_stream(self, stream, is_stderr: bool) -> None:
        try:
            for line in iter(stream.readline, ""):
                if line:
                    self.append_log(line, is_stderr)
        except Exception:
            pass

    def _wait_for_exit(self) -> None:
        if self.process is None:
            return
        self.process.wait()
        self.log_text.after(0, self._on_process_exit)

    def _on_process_exit(self) -> None:
        pass


class DataCollectionGUI:
    """Main application window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Data Collection Launcher")
        self.root.geometry("1100x820")
        self.root.minsize(900, 700)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Start.TButton", foreground="white", background="#2d8c3c")
        style.map("Start.TButton", background=[("active", "#3aaf4d")])
        style.configure("Stop.TButton", foreground="white", background="#c42b2b")
        style.map("Stop.TButton", background=[("active", "#e03e3e")])
        style.configure("TLabelframe.Label", font=("Sans", 10))

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main)
        top_frame.pack(fill=tk.X, pady=(0, 6))

        self._build_device_config(top_frame)
        self._build_saver_config(top_frame)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        self.device_panel = ProcessPanel(paned, "Device Starter Output")
        self.device_panel._on_process_exit = self._update_device_buttons
        paned.add(self.device_panel, weight=1)

        self.saver_panel = ProcessPanel(paned, "Data Saver Output")
        self.saver_panel._on_process_exit = self._update_saver_buttons
        paned.add(self.saver_panel, weight=1)

        # Bind Enter key globally on the whole window.
        # When saver is running, pressing Enter anywhere sends '\n' to its stdin,
        # exactly like pressing Enter in a terminal.
        self.root.bind("<Return>", self._on_saver_enter_key)
        self.root.bind("<KP_Enter>", self._on_saver_enter_key)

    def _build_device_config(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1. Device Starter (windows_device_start.py)", padding=8)
        frame.pack(fill=tk.X, pady=(0, 4))

        row = ttk.Frame(frame)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Config:").pack(side=tk.LEFT)
        self.config_var = tk.StringVar(value=DEFAULT_CONFIG_NAME)
        config_names = _scan_config_files()
        self.config_combo = ttk.Combobox(
            row, textvariable=self.config_var, values=config_names,
            state="readonly", width=40,
        )
        self.config_combo.pack(side=tk.LEFT, padx=(4, 12))
        if DEFAULT_CONFIG_NAME in config_names:
            self.config_combo.set(DEFAULT_CONFIG_NAME)

        ttk.Button(row, text="Refresh", command=self._refresh_configs, width=8).pack(side=tk.LEFT, padx=(0, 20))

        self.device_start_btn = ttk.Button(
            row, text="Start", style="Start.TButton", command=self._start_device, width=10
        )
        self.device_start_btn.pack(side=tk.LEFT, padx=4)

        self.device_stop_btn = ttk.Button(
            row, text="Stop", style="Stop.TButton", command=self._stop_device, width=10, state=tk.DISABLED
        )
        self.device_stop_btn.pack(side=tk.LEFT, padx=4)

    def _build_saver_config(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2. Data Saver (NEDF3 R1)", padding=8)
        frame.pack(fill=tk.X)

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(row1, text="Output Dir:").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value="data/my_task")
        self.output_dir_entry = ttk.Entry(row1, textvariable=self.output_dir_var, width=50)
        self.output_dir_entry.pack(side=tk.LEFT, padx=(4, 4))

        ttk.Button(row1, text="Browse...", command=self._browse_output_dir, width=9).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row1, text="Start Idx:").pack(side=tk.LEFT)
        self.start_idx_var = tk.StringVar(value="0")
        vcmd = (self.root.register(self._validate_int), "%P")
        self.start_idx_entry = ttk.Entry(
            row1, textvariable=self.start_idx_var, width=8,
            validate="key", validatecommand=vcmd,
        )
        self.start_idx_entry.pack(side=tk.LEFT, padx=(4, 12))

        self.saver_start_btn = ttk.Button(
            row1, text="Start", style="Start.TButton", command=self._start_saver, width=10
        )
        self.saver_start_btn.pack(side=tk.LEFT, padx=4)

        self.saver_stop_btn = ttk.Button(
            row1, text="Stop", style="Stop.TButton", command=self._stop_saver, width=10, state=tk.DISABLED
        )
        self.saver_stop_btn.pack(side=tk.LEFT, padx=4)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.episode_toggle_btn = ttk.Button(
            row1, text="Enter (Start/Stop Episode)", command=self._send_enter_to_saver, width=26, state=tk.DISABLED
        )
        self.episode_toggle_btn.pack(side=tk.LEFT, padx=4)

        hint = ttk.Label(row1, text="or press Enter in output panel", foreground="gray")
        hint.pack(side=tk.LEFT, padx=(4, 0))

    # ---- actions ------------------------------------------------------------

    def _refresh_configs(self) -> None:
        names = _scan_config_files()
        self.config_combo.configure(values=names)
        if self.config_var.get() not in names and names:
            self.config_combo.set(names[0])

    def _browse_output_dir(self) -> None:
        initial = self.output_dir_var.get() or DEFAULT_OUTPUT_BASE
        chosen = filedialog.askdirectory(initialdir=initial, title="Select Output Directory")
        if chosen:
            self.output_dir_var.set(chosen)

    @staticmethod
    def _validate_int(value: str) -> bool:
        return value == "" or value.isdigit()

    @staticmethod
    def _subprocess_env() -> Dict[str, str]:
        """Build env dict that forces unbuffered output and proper encoding."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Force UTF-8 encoding to handle Unicode characters like ✓ (U+2713)
        env["PYTHONIOENCODING"] = "utf-8"
        # On Windows, also set console code page to UTF-8
        if sys.platform == "win32":
            env["PYTHONUTF8"] = "1"
        return env

    def _start_device(self) -> None:
        config_name = self.config_var.get()
        if not config_name:
            messagebox.showwarning("Warning", "Please select a config file.")
            return

        cmd = [
            sys.executable, "-u",
            str(PROJECT_ROOT / "windows_device_start.py"),
            "--config-name", config_name,
        ]

        if sys.platform != "win32":
            kwargs_popen: Dict = {"preexec_fn": os.setsid}
        else:
            kwargs_popen = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

        self.device_panel.clear_log()

        try:
            self.device_panel.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                bufsize=0,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._subprocess_env(),
                **kwargs_popen,
            )
        except Exception as exc:
            self.device_panel.append_log(f"Failed to start: {exc}\n", is_stderr=True)
            return

        for stream, is_stderr in [
            (self.device_panel.process.stdout, False),
            (self.device_panel.process.stderr, True),
        ]:
            t = threading.Thread(
                target=self.device_panel._read_stream, args=(stream, is_stderr), daemon=True
            )
            t.start()
        threading.Thread(target=self.device_panel._wait_for_exit, daemon=True).start()

        self.device_start_btn.configure(state=tk.DISABLED)
        self.device_stop_btn.configure(state=tk.NORMAL)

    def _stop_device(self) -> None:
        self.device_panel.stop_process()
        self.device_stop_btn.configure(state=tk.DISABLED)

    def _start_saver(self) -> None:
        config_name = self.config_var.get()
        if not config_name:
            messagebox.showwarning("Warning", "Please select a config file.")
            return

        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showwarning("Warning", "Please specify an output directory.")
            return

        start_idx = self.start_idx_var.get().strip() or "0"

        cmd = [
            sys.executable, "-u",
            str(PROJECT_ROOT / "incubator" / "r1" / "async_data_saver_NEDF3_r1.py"),
            "--config", config_name,
            "--output-dir", output_dir,
            "--start-idx", start_idx,
            "--disable-button-control",
        ]

        if sys.platform != "win32":
            kwargs_popen: Dict = {"preexec_fn": os.setsid}
        else:
            kwargs_popen = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

        self.saver_panel.clear_log()

        try:
            # For Windows, we need to ensure the subprocess can properly handle piped stdin
            # Add environment variable to signal that we're running in GUI mode
            env = self._subprocess_env()
            env["GUI_MODE"] = "1"  # Signal to subprocess that it's running in GUI mode
            
            self.saver_panel.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                bufsize=0,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                **kwargs_popen,
            )
        except Exception as exc:
            self.saver_panel.append_log(f"Failed to start: {exc}\n", is_stderr=True)
            return

        for stream, is_stderr in [
            (self.saver_panel.process.stdout, False),
            (self.saver_panel.process.stderr, True),
        ]:
            t = threading.Thread(
                target=self.saver_panel._read_stream, args=(stream, is_stderr), daemon=True
            )
            t.start()
        threading.Thread(target=self.saver_panel._wait_for_exit, daemon=True).start()

        self.saver_start_btn.configure(state=tk.DISABLED)
        self.saver_stop_btn.configure(state=tk.NORMAL)
        self.episode_toggle_btn.configure(state=tk.NORMAL)

    def _stop_saver(self) -> None:
        self.saver_panel.stop_process()
        self.saver_stop_btn.configure(state=tk.DISABLED)
        self.episode_toggle_btn.configure(state=tk.DISABLED)

    def _send_enter_to_saver(self) -> None:
        """Send a newline to the saver subprocess stdin (simulates pressing Enter)."""
        proc = self.saver_panel.process
        if proc is None:
            self.saver_panel.append_log("[GUI DEBUG] Cannot send Enter: process is None\n", is_stderr=True)
            return
        if proc.poll() is not None:
            self.saver_panel.append_log(f"[GUI DEBUG] Cannot send Enter: process has exited with code {proc.poll()}\n", is_stderr=True)
            return
        if proc.stdin is None:
            self.saver_panel.append_log("[GUI DEBUG] Cannot send Enter: stdin is None\n", is_stderr=True)
            return
        
        try:
            self.saver_panel.append_log("[GUI DEBUG] Sending Enter to saver subprocess...\n", is_stderr=False)
            # Send newline character
            proc.stdin.write("\n")
            proc.stdin.flush()
            self.saver_panel.append_log("[GUI DEBUG] Enter sent and flushed successfully\n", is_stderr=False)
        except (OSError, BrokenPipeError) as e:
            self.saver_panel.append_log(f"[GUI DEBUG] Error sending Enter: {e}\n", is_stderr=True)
        except Exception as e:
            self.saver_panel.append_log(f"[GUI DEBUG] Unexpected error sending Enter: {e}\n", is_stderr=True)

    def _on_saver_enter_key(self, event: tk.Event) -> Optional[str]:
        """Handle Enter key press anywhere in the window.

        Skip if focus is on an Entry or Combobox (user is typing text).
        """
        focused = self.root.focus_get()
        if isinstance(focused, (ttk.Entry, tk.Entry, ttk.Combobox)):
            return None
        if self.saver_panel.is_running():
            self._send_enter_to_saver()
        return "break"

    # ---- button state sync --------------------------------------------------

    def _update_device_buttons(self) -> None:
        self.device_start_btn.configure(state=tk.NORMAL)
        self.device_stop_btn.configure(state=tk.DISABLED)

    def _update_saver_buttons(self) -> None:
        self.saver_start_btn.configure(state=tk.NORMAL)
        self.saver_stop_btn.configure(state=tk.DISABLED)
        self.episode_toggle_btn.configure(state=tk.DISABLED)

    # ---- lifecycle ----------------------------------------------------------

    def _on_close(self) -> None:
        if self.device_panel.is_running() or self.saver_panel.is_running():
            if not messagebox.askokcancel(
                "Confirm Exit",
                "There are running processes. Stop them and exit?",
            ):
                return
        if self.device_panel.is_running():
            self.device_panel.stop_process()
        if self.saver_panel.is_running():
            self.saver_panel.stop_process()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = DataCollectionGUI()
    app.run()


if __name__ == "__main__":
    main()
