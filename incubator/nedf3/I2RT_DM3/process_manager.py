"""Subprocess ownership for the I2RT DM3 collection interface."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Optional, TextIO

import psutil

from utils.logger_config import logger
from utils.runtime_profiles import RuntimeProfileError, select_runtime_python


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEVICE_CONFIG = "i2rt_data_collect_async"
DEFAULT_SAVER_CONFIG = "configs/i2rt_nedf3_saver.yaml"
DEVICE_STOP_TIMEOUT_S = 50.0
SAVER_STOP_TIMEOUT_S = 15.0
VISUALIZER_STOP_TIMEOUT_S = 10.0
_ANSI_RE = re.compile(
    r"(?:\x1b\][^\x07]*(?:\x07|\x1b\\))|(?:\x1b\[[0-?]*[ -/]*[@-~])"
)
_WATCHDOG_FAULT_MARKER = "watchdog fault:"


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
            "LOGURU_COLORIZE": "0",
            "FORCE_COLOR": "0",
        }
    )
    return env


def remove_ansi(text: str) -> str:
    """Remove terminal control sequences while preserving log line spacing."""
    return _ANSI_RE.sub("", text)


def strip_ansi(text: str) -> str:
    return remove_ansi(text).strip()


def scan_saved_episodes(output_dir: str) -> tuple[int, str]:
    root = Path(output_dir)
    if not root.is_dir():
        return 0, "—"
    episodes = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and child.name.startswith("episode_")
        and not child.name.endswith(("_inprogress", "_temp"))
    ]
    if not episodes:
        return 0, "—"
    episodes.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return len(episodes), episodes[0].name


@dataclass
class ManagedProcess:
    name: str
    process: Optional[subprocess.Popen[str]] = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=8000))
    log_start_offset: int = 0
    log_next_offset: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    stopping: bool = False
    recording: bool = False
    finalizing_episode: bool = False
    ready_for_episode: bool = False
    exit_code: Optional[int] = None
    saved_episode_count: int = 0
    last_episode_name: str = "—"
    teleoperation_phase: str = "stopped"
    teleoperation_fault: str = ""

    def append_log(self, line: str) -> None:
        line = remove_ansi(line)
        with self.lock:
            if self.logs.maxlen is not None and len(self.logs) == self.logs.maxlen:
                self.log_start_offset += 1
            self.logs.append(line)
            self.log_next_offset += 1

    def clear_logs(self) -> None:
        with self.lock:
            self.logs.clear()
            self.log_start_offset = 0
            self.log_next_offset = 0

    def read_logs_since(self, offset: int) -> tuple[list[str], int]:
        with self.lock:
            offset = max(offset, self.log_start_offset)
            if offset >= self.log_next_offset:
                return [], self.log_next_offset
            start = max(0, offset - self.log_start_offset)
            return list(islice(self.logs, start, None)), self.log_next_offset

    def status_dict(self) -> dict[str, Any]:
        status = {
            "name": self.name,
            "running": self.running,
            "stopping": self.stopping,
            "recording": self.recording,
            "finalizing_episode": self.finalizing_episode,
            "ready_for_episode": self.ready_for_episode,
            "exit_code": self.exit_code,
            "saved_episode_count": self.saved_episode_count,
            "last_episode_name": self.last_episode_name,
        }
        if self.name == "device":
            status.update(
                teleoperation_phase=self.teleoperation_phase,
                teleoperation_fault=self.teleoperation_fault,
            )
        return status


def kill_process_tree(pid: int, wait_timeout_s: float) -> bool:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    processes = [*reversed(root.children(recursive=True)), root]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=wait_timeout_s)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=wait_timeout_s)
    return not alive


class I2RTProcessManager:
    """Own DeviceStarter and the I2RT NEDF3 saver as process groups."""

    def __init__(
        self,
        *,
        device_config_name: str = DEFAULT_DEVICE_CONFIG,
        require_recording_profile: bool = False,
    ) -> None:
        self.device = ManagedProcess("device")
        self.saver = ManagedProcess("saver")
        self.calibration = ManagedProcess("calibration")
        self.visualizer = ManagedProcess("visualizer")
        self.device_config_name = device_config_name
        self.require_recording_profile = require_recording_profile
        self._saver_profile = None
        self._visualizer_profile = None
        self._launch_lock = threading.Lock()
        self._output_dir = str(REPO_ROOT / "data" / "i2rt_nedf3")

    def start_device(
        self,
        config_name: str = DEFAULT_DEVICE_CONFIG,
        *,
        environment: Optional[dict[str, str]] = None,
    ) -> None:
        with self._launch_lock:
            if (
                self.calibration.running
                or self.calibration.stopping
                or self.calibration.process is not None
            ):
                raise RuntimeError("夹爪标定进程运行时不能启动设备。")
            if self.device.stopping:
                raise RuntimeError("设备进程正在停止，请稍候再启动。")
            if self.device.running:
                raise RuntimeError("设备进程已在运行。")
            command = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "device_starter.py"),
                "--config-name",
                config_name,
            ]
            self.device_config_name = config_name
            if environment:
                self._start_process(
                    self.device,
                    command,
                    stdin=False,
                    environment=environment,
                )
            else:
                self._start_process(self.device, command, stdin=False)

    def stop_device(self) -> None:
        self._stop_process(self.device, timeout_s=DEVICE_STOP_TIMEOUT_S)

    def stop_device_async(self) -> None:
        threading.Thread(target=self.stop_device, daemon=True).start()

    def start_saver(
        self,
        *,
        config_path: str = DEFAULT_SAVER_CONFIG,
        output_dir: str,
        task_name: str,
        task_description: str,
        dm3_user_id: str,
        dm3_user_name: str,
        task_execution_id: str,
    ) -> None:
        if self.saver.stopping:
            raise RuntimeError("保存进程正在停止，请稍候再启动。")
        if self.saver.running:
            raise RuntimeError("数据保存进程已在运行。")
        try:
            selection = select_runtime_python(
                "recording",
                config_name=self.device_config_name,
                fallback=None if self.require_recording_profile else sys.executable,
                repo_root=REPO_ROOT,
                require_verified_profile=True,
                required=True,
            )
        except RuntimeProfileError as exc:
            raise RuntimeError(f"NEDF3 录制环境不可用：{exc}") from exc
        assert selection is not None
        self._saver_profile = selection.profile
        self._output_dir = output_dir
        self._refresh_saved_episode_status()
        command = [
            str(selection.python),
            "-u",
            "-m",
            "consumer.io.async_data_saver_i2rt_NEDF3",
            "--config",
            config_path,
            "--output-dir",
            output_dir,
            "--task-name",
            task_name,
            "--task-description",
            task_description,
            "--dm3-user-id",
            dm3_user_id,
            "--dm3-user-name",
            dm3_user_name,
            "--task-execution-id",
            task_execution_id,
        ]
        self._start_process(self.saver, command, stdin=True)
        self.saver.ready_for_episode = False

    def stop_saver(self) -> None:
        self._stop_process(self.saver, timeout_s=SAVER_STOP_TIMEOUT_S)

    def stop_saver_async(self) -> None:
        threading.Thread(target=self.stop_saver, daemon=True).start()

    def start_calibration(self, config_path: str = "configs/i2rt_bimanual_teleop.yaml") -> None:
        with self._launch_lock:
            if (
                self.calibration.running
                or self.calibration.stopping
                or self.calibration.process is not None
            ):
                raise RuntimeError("夹爪标定进程已在运行。")
            if any(
                managed.running or managed.stopping or managed.process is not None
                for managed in (self.device, self.saver)
            ):
                raise RuntimeError("标定前请停止设备和 NEDF3 保存进程。")
            command = [
                sys.executable,
                "-u",
                str(
                    REPO_ROOT
                    / "scripts"
                    / "calibration"
                    / "calibrate_i2rt_grippers.py"
                ),
                "--config",
                config_path,
                "--side",
                "both",
            ]
            self._start_process(self.calibration, command, stdin=False)

    def stop_calibration(self) -> None:
        self._stop_process(self.calibration, timeout_s=15.0)

    def start_visualizer(self, episode_dir: Path) -> Path:
        """Launch Rerun for one validated episode in an idle system."""
        with self._launch_lock:
            if any(
                managed.running or managed.stopping or managed.process is not None
                for managed in (self.device, self.saver, self.calibration)
            ):
                raise RuntimeError("启动 Rerun 前请停止设备、保存和标定进程。")
            if (
                self.visualizer.running
                or self.visualizer.stopping
                or self.visualizer.process is not None
            ):
                raise RuntimeError("Rerun 正在加载 Episode，请稍候。")

            episode_dir = episode_dir.expanduser().resolve()

            try:
                selection = select_runtime_python(
                    "recording",
                    config_name=self.device_config_name,
                    fallback=None if self.require_recording_profile else sys.executable,
                    repo_root=REPO_ROOT,
                    require_verified_profile=True,
                    required=True,
                )
            except RuntimeProfileError as exc:
                raise RuntimeError(f"NEDF3 可视化环境不可用：{exc}") from exc
            assert selection is not None
            self._visualizer_profile = selection.profile
            command = [
                str(selection.python),
                "-u",
                str(REPO_ROOT / "scripts" / "visualization" / "i2rt_nedf3_rerun.py"),
                str(episode_dir),
            ]
            self._start_process(self.visualizer, command, stdin=False)
            return episode_dir

    def send_saver_command(self, command: str) -> None:
        process = self.saver.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("数据保存进程未运行。")
        if command not in {"toggle", "discard"}:
            raise ValueError(f"unsupported saver command: {command}")
        was_recording = self.saver.recording
        if command == "toggle" and was_recording:
            self.saver.finalizing_episode = True
            self.saver.ready_for_episode = False
        try:
            process.stdin.write("\n" if command == "toggle" else "d\n")
            process.stdin.flush()
        except Exception:
            if command == "toggle" and was_recording:
                self.saver.finalizing_episode = False
                self.saver.ready_for_episode = False
            raise

    def runtime_status(self) -> dict[str, Any]:
        return {
            "device": self.device.status_dict(),
            "saver": self.saver.status_dict(),
            "calibration": self.calibration.status_dict(),
            "visualizer": self.visualizer.status_dict(),
        }

    def refresh_saved_episodes(self, output_dir: Optional[str] = None) -> None:
        if output_dir is not None:
            self._output_dir = output_dir
        self._refresh_saved_episode_status()

    def shutdown(self) -> None:
        if self.saver.process is not None:
            self.stop_saver()
        if self.device.process is not None:
            self.stop_device()
        if self.calibration.process is not None:
            self.stop_calibration()
        if self.visualizer.process is not None:
            self._stop_process(
                self.visualizer,
                timeout_s=VISUALIZER_STOP_TIMEOUT_S,
            )

    def _start_process(
        self,
        managed: ManagedProcess,
        command: list[str],
        *,
        stdin: bool,
        environment: Optional[dict[str, str]] = None,
    ) -> None:
        managed.clear_logs()
        managed.exit_code = None
        managed.recording = False
        managed.finalizing_episode = False
        managed.ready_for_episode = False
        if managed.name == "device":
            managed.teleoperation_phase = "starting"
            managed.teleoperation_fault = ""
        managed.append_log(f"$ {' '.join(command)}\n")
        kwargs: dict[str, Any] = {
            "cwd": str(REPO_ROOT),
            "stdin": subprocess.PIPE if stdin else subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": subprocess_env(),
        }
        profile = {
            "saver": self._saver_profile,
            "visualizer": self._visualizer_profile,
        }.get(managed.name)
        if profile is not None:
            kwargs["env"] = profile.process_environment(
                config_name=self.device_config_name,
                base=kwargs["env"],
            )
        if environment:
            kwargs["env"].update(environment)
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        managed.process = process
        managed.running = True
        threading.Thread(
            target=self._read_output,
            args=(managed, process),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._wait_for_exit,
            args=(managed, process),
            daemon=True,
        ).start()

    def _read_output(
        self,
        managed: ManagedProcess,
        process: subprocess.Popen[str],
    ) -> None:
        stream: Optional[TextIO] = process.stdout
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                managed.append_log(line)
                self._update_runtime_from_log(managed, line)
        finally:
            stream.close()

    def _update_runtime_from_log(self, managed: ManagedProcess, line: str) -> None:
        plain = strip_ansi(line)
        if managed.name == "device":
            if "Teleoperation ready;" in plain:
                managed.teleoperation_phase = "safe_hold"
            elif "Aligning followers" in plain:
                managed.teleoperation_phase = "aligning"
            elif "Teleoperation enabled" in plain:
                managed.teleoperation_phase = "active"
            elif "Teleoperation disabled" in plain:
                managed.teleoperation_phase = "safe_hold"
            elif _WATCHDOG_FAULT_MARKER in plain:
                managed.teleoperation_phase = "fault"
                managed.teleoperation_fault = plain.split(
                    _WATCHDOG_FAULT_MARKER, 1
                )[1].strip()
                managed.ready_for_episode = False
            if (
                "Successfully started" in plain
                and "devices" in plain
                and managed.teleoperation_phase != "fault"
            ):
                managed.ready_for_episode = True
            return
        if managed.name != "saver":
            return
        if plain.startswith("Ready:"):
            managed.recording = False
            managed.finalizing_episode = False
            managed.ready_for_episode = True
        elif plain.startswith("Recording to "):
            managed.recording = True
            managed.finalizing_episode = False
            managed.ready_for_episode = False
        elif plain.startswith("Finalized "):
            managed.recording = False
            managed.finalizing_episode = False
            managed.ready_for_episode = True
            managed.last_episode_name = Path(plain.removeprefix("Finalized ")).name
            self._refresh_saved_episode_status()
        elif plain.startswith("Discarded "):
            managed.recording = False
            managed.finalizing_episode = False
            managed.ready_for_episode = True

    def _wait_for_exit(
        self,
        managed: ManagedProcess,
        process: subprocess.Popen[str],
    ) -> None:
        return_code = process.wait()
        managed.append_log(f"\n[process exited with code {return_code}]\n")
        managed.exit_code = return_code
        managed.running = False
        managed.recording = False
        managed.finalizing_episode = False
        managed.ready_for_episode = False
        if managed.name == "device":
            if return_code == 0:
                managed.teleoperation_phase = "stopped"
            else:
                managed.teleoperation_phase = "fault"
                if not managed.teleoperation_fault:
                    managed.teleoperation_fault = (
                        f"device process exited with code {return_code}"
                    )
        if managed.process is process:
            managed.process = None

    def _stop_process(self, managed: ManagedProcess, *, timeout_s: float) -> None:
        if managed.stopping:
            return
        process = managed.process
        if process is None or process.poll() is not None:
            managed.running = False
            managed.process = None
            return
        managed.stopping = True
        try:
            try:
                if sys.platform == "win32":
                    process.send_signal(
                        getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)
                    )
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except (OSError, ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "%s did not stop within %.0fs; killing its process tree",
                    managed.name,
                    timeout_s,
                )
                kill_process_tree(process.pid, wait_timeout_s=5.0)
                process.wait(timeout=5.0)
        finally:
            managed.running = False
            managed.recording = False
            managed.finalizing_episode = False
            managed.ready_for_episode = False
            if managed.name == "device":
                managed.teleoperation_phase = "stopped"
            if managed.process is process:
                managed.process = None
            managed.stopping = False

    def _refresh_saved_episode_status(self) -> None:
        count, last_name = scan_saved_episodes(self._output_dir)
        self.saver.saved_episode_count = count
        self.saver.last_episode_name = last_name


__all__ = [
    "DEFAULT_DEVICE_CONFIG",
    "DEFAULT_SAVER_CONFIG",
    "I2RTProcessManager",
    "REPO_ROOT",
]
