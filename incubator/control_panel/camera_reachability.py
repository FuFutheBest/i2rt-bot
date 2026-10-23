"""Bounded, process-isolated V4L2 frame reachability probes."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, Iterable, Mapping


_FRAME_PROBE_WORKER = r"""
import sys

import cv2

path = sys.argv[1]
capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
try:
    if not capture.isOpened():
        print("open_failed", flush=True)
        raise SystemExit(2)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    capture.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    )
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        print("frame_failed", flush=True)
        raise SystemExit(3)
    print("ok", flush=True)
finally:
    capture.release()
"""


class CameraFrameProbe:
    """Fetch one frame per camera without risking the control-panel process.

    A broken xHCI device can leave an opener in uninterruptible kernel sleep.
    Each camera is therefore opened in its own process. Timed-out workers are
    killed when possible and retained as a circuit breaker while the kernel
    still owns them, preventing every dropdown open from leaking another
    blocked process.
    """

    def __init__(self, *, timeout_seconds: float = 1.25) -> None:
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self._lock = threading.Lock()
        self._blocked: Dict[str, subprocess.Popen[str]] = {}

    @staticmethod
    def _probe_path(value: str) -> str:
        return str(Path(value).expanduser())

    @staticmethod
    def _start(path: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-c", _FRAME_PROBE_WORKER, path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            close_fds=True,
            start_new_session=True,
        )

    @staticmethod
    def _output(process: subprocess.Popen[str]) -> str:
        try:
            stdout, _ = process.communicate(timeout=0.05)
        except (OSError, subprocess.SubprocessError):
            return ""
        lines = str(stdout or "").strip().splitlines()
        return lines[-1] if lines else ""

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> bool:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - release runtime is Linux
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=0.05)
        except (OSError, subprocess.TimeoutExpired):
            return process.poll() is not None
        return True

    def probe(self, paths: Iterable[str]) -> Mapping[str, Mapping[str, object]]:
        requested = list(dict.fromkeys(self._probe_path(path) for path in paths))
        results: Dict[str, Mapping[str, object]] = {}
        if not requested:
            return results

        with self._lock:
            for path, process in list(self._blocked.items()):
                if process.poll() is None:
                    continue
                self._output(process)
                self._blocked.pop(path, None)

            pending: Dict[str, subprocess.Popen[str]] = {}
            for path in requested:
                blocked = self._blocked.get(path)
                if blocked is not None and blocked.poll() is None:
                    results[path] = {
                        "reachable": False,
                        "status": "probe_still_blocked",
                    }
                    continue
                try:
                    pending[path] = self._start(path)
                except OSError:
                    results[path] = {
                        "reachable": False,
                        "status": "probe_start_failed",
                    }

            deadline = time.monotonic() + self.timeout_seconds
            while pending and time.monotonic() < deadline:
                for path, process in list(pending.items()):
                    if process.poll() is None:
                        continue
                    output = self._output(process)
                    results[path] = {
                        "reachable": process.returncode == 0 and output == "ok",
                        "status": output or f"exit_{process.returncode}",
                    }
                    pending.pop(path, None)
                if pending:
                    time.sleep(0.01)

            for path, process in pending.items():
                reaped = self._terminate(process)
                if not reaped:
                    self._blocked[path] = process
                results[path] = {
                    "reachable": False,
                    "status": "timeout",
                }
        return results


__all__ = ["CameraFrameProbe"]
