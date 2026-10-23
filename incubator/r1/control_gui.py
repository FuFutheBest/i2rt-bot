#!/usr/bin/env python3
"""Unified R1 control console.

A single Tkinter window that:
  * Start / End all devices (launches device_starter.py as a subprocess and
    shuts it down gracefully with SIGINT).
  * Embeds the existing NEDF3 data-collection panel (DataSaverGUI).
  * Shows live previews of the three USB cameras (right wrist / left wrist /
    head) read straight from their shared-memory buffers, so you can confirm at
    a glance whether each camera is connected.

Run from the repo root:
    python scripts/r1_control_gui.py
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import numpy as np

try:
    import cv2
except Exception as exc:  # noqa: BLE001
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except Exception:  # noqa: BLE001
    _PIL_OK = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.shm_utils import (  # noqa: E402
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_shared_memory,
    connect_to_summary_shm,
    get_dtype,
    unpack_device_header,
    unpack_buffer_header,
)

from incubator.r1.async_data_saver_NEDF3_r1 import (  # noqa: E402
    R1DataSaverConsumer,
    extract_r1_runtime_metadata,
    load_composed_config,
)
from incubator.r1.data_saver_gui import DataSaverGUI  # noqa: E402


# Camera role -> WebCameraDevice id. Matches configs/devices/quest3_collect_r1.yaml:
#   id0 = right wrist, id1 = left wrist, id2 = head.
# (cam_id, title, (width, height)). Wrist cameras stream 16:9, so their tiles
# use a 16:9 thumbnail to avoid vertical squashing; the head camera stays 4:3.
CAMERA_LAYOUT = [
    (0, "Right wrist (id0)", (240, 135)),  # 16:9
    (1, "Left wrist (id1)", (240, 135)),   # 16:9
    (2, "Head (id2)", (240, 160)),         # 4:3
]

PREVIEW_SIZE = (240, 160)  # default (width, height) of a preview thumbnail
PREVIEW_FPS = 15
DEFAULT_CONFIG_NAME = "quest3_collect_r1"
DEFAULT_RUNTIME = "r1_cpu_24core"
CONFIG_PATH = os.path.join(REPO_ROOT, "configs")


class CameraPreview(ttk.Frame):
    """One camera tile: background SHM reader + Tk image label + status dot."""

    def __init__(self, parent: tk.Misc, cam_id: int, title: str,
                 size=PREVIEW_SIZE, fps: int = PREVIEW_FPS) -> None:
        super().__init__(parent)
        self.cam_id = cam_id
        self.title = title
        self.shm_name = f"WebCameraDevice_{cam_id}_data"
        self.size = size
        self.fps = max(1, fps)

        self._shm = None
        self._last_write_index = -1
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._photo = None  # keep a ref so Tk doesn't GC the image
        self._connected = False
        self._ui_scheduled = False

        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text=title).pack(side="left", padx=2)
        self._status = tk.Label(header, text="\u25cf", fg="gray")  # status dot
        self._status.pack(side="right", padx=2)

        # A fixed-size gray placeholder keeps the tile at exactly ``size`` pixels
        # whether or not a frame is present. (A text-only Label would size in
        # character units -> a giant tile until the first frame arrives.)
        self._placeholder = self._make_placeholder()
        self._img = tk.Label(
            self, bg="#202020", fg="#bbbbbb", text="NO SIGNAL",
            compound="center",
        )
        if self._placeholder is not None:
            self._img.config(image=self._placeholder)
        else:
            # No Pillow: fall back to a char-sized box (approximate).
            self._img.config(width=max(8, size[0] // 8), height=max(4, size[1] // 16))
        self._img.pack(padx=2, pady=2)

    def _make_placeholder(self):
        if not _PIL_OK:
            return None
        try:
            return ImageTk.PhotoImage(
                Image.new("RGB", (self.size[0], self.size[1]), (32, 32, 32))
            )
        except Exception:  # noqa: BLE001
            return None

    # ----- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._last_write_index = -1
        self._thread = threading.Thread(
            target=self._run, name=f"cam{self.cam_id}-shm", daemon=True
        )
        self._thread.start()
        if not self._ui_scheduled:
            self._ui_scheduled = True
            self._schedule_ui()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._latest = None
        self._connected = False
        self._close_shm()

    def _close_shm(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:  # noqa: BLE001
                pass
            self._shm = None

    # ----- SHM reading (background thread) ---------------------------------
    def _connect(self) -> bool:
        if self._shm is not None:
            return True
        try:
            self._shm = connect_to_shared_memory(
                self.shm_name, read_only=True, log_missing=False
            )
            return True
        except FileNotFoundError:
            return False
        except Exception:  # noqa: BLE001
            return False

    def _read_latest_frame(self) -> Optional[np.ndarray]:
        if not self._connect():
            return None
        buf = self._shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE: DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )
        if buffer_header["current_frames_count"] == 0:
            return None
        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        if write_index == self._last_write_index:
            return None
        self._last_write_index = write_index

        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = (
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size
        )
        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        dtype = get_dtype(device_header["data_dtype"], np.uint8)
        arr = np.frombuffer(buf[data_start: data_start + data_size], dtype=dtype)
        shape = device_header["shape"]
        if len(shape) > 0:
            arr = arr.reshape(shape)
        return np.ascontiguousarray(arr)

    def _run(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            try:
                frame = self._read_latest_frame()
            except Exception:  # noqa: BLE001
                # SHM was likely torn down mid-read; drop the handle and retry.
                self._close_shm()
                frame = None
            if frame is not None:
                with self._lock:
                    self._latest = frame
            time.sleep(period)

    # ----- UI update (main thread) -----------------------------------------
    def _schedule_ui(self) -> None:
        self._update_ui()
        if not self._stop.is_set():
            self.after(int(1000 / self.fps), self._schedule_ui)
        else:
            self._ui_scheduled = False
            self._show_no_signal()

    def _update_ui(self) -> None:
        with self._lock:
            frame = None if self._latest is None else self._latest.copy()
        if frame is None:
            if self._connected:
                self._connected = False
                self._status.config(fg="gray")
                self._show_no_signal()
            return

        if not self._connected:
            self._connected = True
            self._status.config(fg="#22cc22")

        img = self._to_photo(frame)
        if img is not None:
            self._photo = img
            self._img.config(image=img, text="")

    def _to_photo(self, frame: np.ndarray):
        try:
            disp = frame
            if disp.ndim == 3 and disp.shape[2] >= 3:
                disp = disp[:, :, :3]
            if cv2 is not None:
                disp = cv2.resize(disp, self.size, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            else:
                rgb = disp[..., ::-1] if disp.ndim == 3 else disp
            if _PIL_OK:
                return ImageTk.PhotoImage(Image.fromarray(rgb))
            return None
        except Exception:  # noqa: BLE001
            return None

    def _show_no_signal(self) -> None:
        self._photo = None
        if self._placeholder is not None:
            self._img.config(image=self._placeholder, text="NO SIGNAL")
        else:
            self._img.config(image="", text="NO SIGNAL")


class R1ControlGUI:
    """Top-level window orchestrating devices, cameras and the saver panel."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("R1 Control Console")
        self.root.geometry("1080x980")
        self.root.minsize(960, 820)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.proc: Optional[subprocess.Popen] = None
        # Process-group id of the device_starter subprocess (== its pid because we
        # start it in a new session). Cached so we can always target the group
        # even after ``self.proc`` is reaped/cleared.
        self._pgid: Optional[int] = None
        self._consumer: Optional[R1DataSaverConsumer] = None
        self._panel: Optional[DataSaverGUI] = None
        self._stop_ready = threading.Event()
        self._cam_previews: List[CameraPreview] = []
        self._shutdown_lock = threading.Lock()
        self._closing = False

        # Last-resort net: if the interpreter exits for ANY reason (crash,
        # uncaught exception, sys.exit) without a clean teardown, make sure the
        # device process group never outlives this GUI.
        atexit.register(self._atexit_cleanup)

        # Own SIGINT/SIGTERM so a terminal Ctrl+C on the GUI still tears down the
        # device group. (Installed on the main thread; re-installed after the
        # consumer is built because its __init__ hijacks these handlers.)
        self._install_signal_handlers()

        self._build_top_bar()
        self._build_camera_row()
        self._build_panel_area()

        if not _PIL_OK:
            self._set_collect_status(
                "Pillow not installed: previews disabled (pip install pillow)",
                "red",
            )
        if cv2 is None:
            self._set_collect_status(
                f"OpenCV import failed: {_CV2_IMPORT_ERROR}", "red"
            )

    # ----- layout ----------------------------------------------------------
    def _build_top_bar(self) -> None:
        pad = {"padx": 6, "pady": 4}
        bar = ttk.LabelFrame(self.root, text="Devices")
        bar.pack(fill="x", **pad)

        ttk.Label(bar, text="config-name").grid(row=0, column=0, sticky="e", **pad)
        self.var_config = tk.StringVar(value=DEFAULT_CONFIG_NAME)
        ttk.Entry(bar, textvariable=self.var_config, width=22).grid(
            row=0, column=1, sticky="w", **pad
        )

        ttk.Label(bar, text="runtime").grid(row=0, column=2, sticky="e", **pad)
        self.var_runtime = tk.StringVar(value=DEFAULT_RUNTIME)
        ttk.Entry(bar, textvariable=self.var_runtime, width=16).grid(
            row=0, column=3, sticky="w", **pad
        )

        self.btn_start = ttk.Button(bar, text="Start", command=self._on_start)
        self.btn_start.grid(row=0, column=4, **pad)
        self.btn_end = ttk.Button(
            bar, text="End", command=self._on_end, state="disabled"
        )
        self.btn_end.grid(row=0, column=5, **pad)

        self.lbl_device_status = tk.Label(bar, text="STOPPED", fg="gray")
        self.lbl_device_status.grid(row=0, column=6, sticky="w", **pad)
        bar.columnconfigure(6, weight=1)

    def _build_camera_row(self) -> None:
        pad = {"padx": 6, "pady": 4}
        cams = ttk.LabelFrame(self.root, text="Camera previews")
        cams.pack(fill="x", **pad)
        for col, (cam_id, title, size) in enumerate(CAMERA_LAYOUT):
            tile = CameraPreview(cams, cam_id, title, size=size)
            tile.grid(row=0, column=col, padx=6, pady=4, sticky="n")
            self._cam_previews.append(tile)
            cams.columnconfigure(col, weight=1)

    def _build_panel_area(self) -> None:
        pad = {"padx": 6, "pady": 4}
        wrap = ttk.LabelFrame(self.root, text="Data collection")
        wrap.pack(fill="both", expand=True, **pad)

        self.lbl_collect_status = tk.Label(
            wrap, text="Idle - press Start to launch devices", fg="gray", anchor="w"
        )
        self.lbl_collect_status.pack(fill="x", padx=6, pady=2)

        self.panel_container = ttk.Frame(wrap)
        self.panel_container.pack(fill="both", expand=True)

    # ----- status helpers --------------------------------------------------
    def _set_device_status(self, text: str, color: str) -> None:
        self.lbl_device_status.config(text=text, fg=color)

    def _set_collect_status(self, text: str, color: str) -> None:
        self.lbl_collect_status.config(text=text, fg=color)

    # ----- start -----------------------------------------------------------
    def _on_start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        cfg = self.var_config.get().strip() or DEFAULT_CONFIG_NAME
        runtime = self.var_runtime.get().strip() or DEFAULT_RUNTIME
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "device_starter.py"),
            "--config-path",
            CONFIG_PATH,
            "--config-name",
            cfg,
            f"runtime={runtime}",
        ]
        try:
            # start_new_session=True makes device_starter the leader of a fresh
            # process group so we can deliver SIGINT to the WHOLE group on stop
            # (mimicking a terminal Ctrl+C). device_starter.stop_all() relies on
            # each device worker having received SIGINT itself to run its own
            # graceful shutdown (YAM/Flexiv go-rest); signalling only the parent
            # skips that and the arms never return home.
            self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True)
        except Exception as exc:  # noqa: BLE001
            self._set_device_status(f"FAILED: {exc}", "red")
            return

        try:
            self._pgid = os.getpgid(self.proc.pid)
        except Exception:  # noqa: BLE001
            self._pgid = self.proc.pid  # new session -> pgid == pid

        self._stop_ready.clear()
        self.btn_start.config(state="disabled")
        self.btn_end.config(state="normal")
        self._set_device_status("STARTING", "orange")
        self._set_collect_status("Waiting for devices to come up...", "orange")

        for cam in self._cam_previews:
            cam.start()

        threading.Thread(target=self._monitor_proc, daemon=True).start()
        threading.Thread(target=self._wait_ready_and_attach, daemon=True).start()

    def _wait_ready_and_attach(self) -> None:
        # Background thread: ONLY wait until the device manager has published the
        # summary SHM. The consumer itself MUST be built on the main thread
        # because its __init__ calls signal.signal(), which Python only allows
        # from the main thread.
        while not self._stop_ready.is_set():
            if self.proc is None or self.proc.poll() is not None:
                return
            try:
                ch = connect_to_summary_shm(read_only=True)
                try:
                    ch.close()
                except Exception:  # noqa: BLE001
                    pass
                break
            except FileNotFoundError:
                time.sleep(0.5)
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        if self._stop_ready.is_set():
            return

        # Give the device manager a moment to populate device slots, then hand
        # off to the main thread for consumer construction + panel attach.
        time.sleep(0.5)
        self.root.after(0, self._build_consumer_and_attach)

    def _build_consumer_and_attach(self) -> None:
        # MAIN THREAD ONLY (scheduled via root.after).
        if self._stop_ready.is_set() or self._consumer is not None:
            return
        self._set_collect_status("Connecting data saver...", "orange")
        try:
            config_name = self.var_config.get().strip() or DEFAULT_CONFIG_NAME
            runtime_name = self.var_runtime.get().strip() or DEFAULT_RUNTIME
            composed_config = load_composed_config(
                config_name,
                overrides=[f"runtime={runtime_name}"],
            )
            runtime_metadata = extract_r1_runtime_metadata(composed_config)
            consumer = R1DataSaverConsumer(
                summary_shm_name="device_summary_data",
                output_dir="saved_data",
                save_interval=0.001,
                async_write=True,
                robot_type="R1-Flexiv-Rizon4",
                task_name="data_collection",
                end_effector_vendor_model="lktech_gripper",
                device_type="teleoperation_r1",
                **runtime_metadata,
                enable_button_control=False,
                signal_files={},
            )
            # GUI owns episode start/stop; DataSaverGUI also sets this, but make
            # it explicit so the saver loop waits for the panel.
            consumer.gui_control = True
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._set_collect_status(f"consumer init failed: {exc}", "red")
            return

        # The consumer's __init__ re-registered SIGINT/SIGTERM to its own
        # handler; restore ours so Ctrl+C still tears down the device group.
        self._install_signal_handlers()

        if not consumer.connect():
            self._set_collect_status(
                "Failed to connect to summary SHM (is the device manager up?)",
                "red",
            )
            return

        self._consumer = consumer
        self._attach_panel()

    def _attach_panel(self) -> None:
        if self._consumer is None or self._stop_ready.is_set():
            return
        for child in self.panel_container.winfo_children():
            child.destroy()
        self._panel = DataSaverGUI(self._consumer, parent=self.panel_container)
        self._panel.start_saver_thread()
        self._set_device_status("RUNNING", "#22cc22")
        self._set_collect_status("Devices connected - ready to record", "#22cc22")

    # ----- monitor ---------------------------------------------------------
    def _monitor_proc(self) -> None:
        proc = self.proc
        if proc is None:
            return
        # Poll (not blocking wait) so reaping stays lock-coordinated with
        # _reap_child() called from the shutdown paths.
        while proc.poll() is None:
            if self._stop_ready.is_set():
                # Shutdown in progress; let the teardown path own reaping.
                return
            time.sleep(0.5)
        if self._stop_ready.is_set():
            return  # expected exit (End / close)
        self.root.after(0, self._on_proc_exited)

    def _on_proc_exited(self) -> None:
        code = self.proc.poll() if self.proc else None
        self._set_device_status(f"EXITED (code {code})", "red")
        self._set_collect_status("Devices stopped unexpectedly", "red")
        self._teardown_collection()
        self.btn_start.config(state="normal")
        self.btn_end.config(state="disabled")

    # ----- end / shutdown --------------------------------------------------
    def _on_end(self) -> None:
        self._stop_ready.set()
        self.btn_end.config(state="disabled")
        self._set_device_status("STOPPING", "orange")
        self._set_collect_status("Shutting down devices (arms returning home)...", "orange")
        self._teardown_collection()
        for cam in self._cam_previews:
            cam.stop()

        def worker() -> None:
            self._shutdown_group(graceful=True)
            try:
                self.root.after(0, self._on_stopped)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _teardown_collection(self) -> None:
        if self._panel is not None:
            try:
                self._panel.shutdown_saver()
            except Exception:  # noqa: BLE001
                pass
        if self._consumer is not None:
            try:
                self._consumer.stop()
            except Exception:  # noqa: BLE001
                pass
        for child in self.panel_container.winfo_children():
            child.destroy()
        self._panel = None
        self._consumer = None

    def _killpg(self, sig: int) -> None:
        """Send ``sig`` to the device process group (fallback: direct child)."""
        pgid = self._pgid
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return  # group already gone
            except Exception:  # noqa: BLE001
                pass
        proc = self.proc
        if proc is not None:
            try:
                proc.send_signal(sig)
            except Exception:  # noqa: BLE001
                pass

    def _reap_child(self) -> None:
        """Reap the device_starter child if it has exited (clears the zombie).

        ``Popen.poll`` is WNOHANG and guarded by Popen's internal waitpid lock,
        so it's safe to call concurrently with the monitor thread. Without this,
        a just-killed child lingers as a zombie and ``killpg(pgid, 0)`` keeps
        reporting the group as 'alive'.
        """
        proc = self.proc
        if proc is not None:
            try:
                proc.poll()
            except Exception:  # noqa: BLE001
                pass

    def _group_alive(self) -> bool:
        """True if any (non-zombie) process in the device group is still alive."""
        pgid = self._pgid
        if pgid is None:
            return False
        self._reap_child()
        try:
            os.killpg(pgid, 0)  # signal 0 == existence check
            return True
        except Exception:  # noqa: BLE001 (ProcessLookupError / PermissionError)
            return False

    def _wait_group_gone(self, timeout: float) -> bool:
        """Poll until the whole process group has exited or ``timeout`` elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._group_alive():
                return True
            time.sleep(0.2)
        return not self._group_alive()

    def _shutdown_group(self, graceful: bool = True) -> bool:
        """Tear down the entire device process group; return True once it's gone.

        Idempotent and safe to call from any thread / exit path. Escalates
        SIGINT (graceful go-rest) -> SIGTERM -> SIGKILL and verifies, by
        checking the *group* (not just the parent), that nothing survives.
        """
        with self._shutdown_lock:
            if not self._group_alive():
                self._pgid = None
                return True
            if graceful:
                # Each device worker runs its own go-rest; device_starter then
                # joins them. Give the arms time to ramp home.
                self._killpg(signal.SIGINT)
                self._wait_group_gone(30)
            if self._group_alive():
                self._killpg(signal.SIGTERM)
                self._wait_group_gone(8)
            if self._group_alive():
                self._killpg(signal.SIGKILL)
                self._wait_group_gone(3)
            gone = not self._group_alive()
            if gone:
                self._pgid = None
            return gone

    def _install_signal_handlers(self) -> None:
        """Route SIGINT/SIGTERM (on the GUI process) to a clean group teardown.

        Must run on the main thread. Re-installed after the consumer is built
        because ``consumer.__init__`` overrides these with its own handler.
        """
        try:
            signal.signal(signal.SIGINT, self._on_signal)
            signal.signal(signal.SIGTERM, self._on_signal)
        except Exception:  # noqa: BLE001 (e.g. not main thread)
            pass

    def _on_signal(self, signum, frame) -> None:  # noqa: ANN001
        self._shutdown_group(graceful=True)
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass

    def _atexit_cleanup(self) -> None:
        """Final safety net at interpreter shutdown: never leave the group alive."""
        if self._group_alive():
            self._killpg(signal.SIGINT)
            self._wait_group_gone(5)
            if self._group_alive():
                self._killpg(signal.SIGKILL)
                self._wait_group_gone(2)

    def _on_stopped(self) -> None:
        self.proc = None
        self.btn_start.config(state="normal")
        self.btn_end.config(state="disabled")
        if self._group_alive():
            self._set_device_status("STOP FAILED (processes survived)", "red")
            self._set_collect_status(
                "Some device processes could not be killed - check manually", "red"
            )
        else:
            self._set_device_status("STOPPED", "gray")
            self._set_collect_status("Idle - press Start to launch devices", "gray")

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_ready.set()
        self._teardown_collection()
        for cam in self._cam_previews:
            cam.stop()

        if not self._group_alive():
            self.root.destroy()
            return

        # Keep the window alive while the arms ramp home; destroy only once the
        # whole device group is confirmed gone so we never orphan it.
        self._set_device_status("STOPPING", "orange")
        self._set_collect_status(
            "Shutting down devices (arms returning home), please wait...", "orange"
        )
        try:
            self.btn_start.config(state="disabled")
            self.btn_end.config(state="disabled")
        except Exception:  # noqa: BLE001
            pass

        def worker() -> None:
            self._shutdown_group(graceful=True)
            try:
                self.root.after(0, self.root.destroy)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=worker, daemon=True).start()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = R1ControlGUI()  # installs its own SIGINT/SIGTERM handlers
    try:
        app.run()
    finally:
        # Covers a mainloop that returns via an uncaught error; atexit is the
        # absolute last resort behind this.
        app._shutdown_group(graceful=True)


if __name__ == "__main__":
    main()
