#!/usr/bin/env python3
"""
Subprocess manager for Vive_DM3 device startup, NEDF3 saver, and visualizer.

Author: Cursor Agent
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Deque, Dict, IO, List, Optional, Tuple

import yaml

from utils.config_parser import ensure_config_dict, parse_device_configs
from utils.logger_config import logger
from incubator.nedf3.Vive_DM3.resource_policy import apply_recording_process_policy
from utils.process_utils import kill_process_tree
from utils.rerun_spawn import enrich_subprocess_env
from utils.shm_utils import cleanup_device_runtime, collect_device_shm_names

# Desktop GUI uses file-backed subprocess logs to avoid stdout pipe backpressure.
COLLECTION_PERF_SAFE: bool = True


def collection_perf_safe_enabled() -> bool:
    """Return True when file-backed subprocess logging is enabled."""
    return COLLECTION_PERF_SAFE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs" / "vive_dm3"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_NAME = "windows_async_webcam"
DEFAULT_OUTPUT_BASE = "data"
DEVICE_PROCESS_STOP_TIMEOUT_S = 40.0
SAVER_PROCESS_STOP_TIMEOUT_S = 12.0

_CAMERA_CAPTURE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("right_capture_id", "右腕 USB"),
    ("left_capture_id", "左腕 USB"),
    ("ego_capture_id", "头戴 USB"),
)

_EPISODE_STATUS_RE = re.compile(
    r"\[episode_status\]\s+saved_count=(\d+)\s+last_episode=(\S+)"
)
_FINALIZED_EPISODE_RE = re.compile(r"Finalized NEDF3 episode:\s*(.+)")
_DISCARDED_EPISODE_RE = re.compile(r"Discarded NEDF3 episode:\s*(.+)")
_EPISODE_TIMING_RE = re.compile(r"\[episode_timing\]\s+status=\S+\s+episode=\S+")
_ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def scan_config_files() -> List[str]:
    if not CONFIGS_DIR.is_dir():
        return [DEFAULT_CONFIG_NAME]
    names = sorted(path.stem for path in CONFIGS_DIR.glob("*.yaml"))
    return names or [DEFAULT_CONFIG_NAME]


def load_config_defaults(config_name: str = DEFAULT_CONFIG_NAME) -> Dict[str, Any]:
    """Load camera and encoder defaults from a top-level Hydra config."""
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    payload: Dict[str, Any] = {
        "config_name": config_name,
        "camera_indices": {key: "" for key, _ in _CAMERA_CAPTURE_KEYS},
        "encoder_ports": {"left_port": "", "right_port": ""},
    }
    if not config_path.is_file():
        return payload

    try:
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return payload

    camera_section = data.get("camera")
    if isinstance(camera_section, dict):
        for key, _ in _CAMERA_CAPTURE_KEYS:
            value = camera_section.get(key)
            if value is not None:
                payload["camera_indices"][key] = str(value)

    encoder_section = data.get("encoder")
    if isinstance(encoder_section, dict):
        for port_key in ("left_port", "right_port"):
            value = encoder_section.get(port_key)
            if value is not None:
                payload["encoder_ports"][port_key] = str(value)

    return payload


def list_serial_ports() -> List[Dict[str, str]]:
    """Return available serial ports for RotaryEncoderDevice selection."""
    try:
        from serial.tools import list_ports

        ports: List[Dict[str, str]] = []
        for entry in list_ports.comports():
            device = str(entry.device or "").strip()
            if not device:
                continue
            description = str(entry.description or "").strip()
            ports.append({"device": device, "description": description})
        ports.sort(key=lambda item: item["device"])
        return ports
    except Exception:
        return []


def save_camera_capture_indices(config_name: str, indices: Dict[str, str]) -> None:
    """Persist capture_id values into configs/<name>.yaml, preserving comments."""
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    text = config_path.read_text(encoding="utf-8")
    for key, _ in _CAMERA_CAPTURE_KEYS:
        value = str(indices.get(key) or "").strip()
        if not value:
            raise ValueError(f"请填写 {key}。")
        pattern = rf"(^(\s*{re.escape(key)}:\s*))\S+"
        replacement = rf"\g<1>{value}"
        new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count == 0:
            raise KeyError(f"配置文件中未找到字段：{key}")
        text = new_text
    config_path.write_text(text, encoding="utf-8")


def save_encoder_ports(config_name: str, ports: Dict[str, str]) -> None:
    """Persist encoder.left_port / encoder.right_port into configs/<name>.yaml."""
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    text = config_path.read_text(encoding="utf-8")
    for key in ("left_port", "right_port"):
        value = str(ports.get(key) or "").strip()
        if not value:
            raise ValueError(f"请填写 encoder.{key}。")
        pattern = rf'(^(\s*{re.escape(key)}:\s*))(?:"[^"]*"|\S+)'
        replacement = rf'\g<1>"{value}"'
        new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count == 0:
            raise KeyError(f"配置文件中未找到字段：{key}")
        text = new_text
    config_path.write_text(text, encoding="utf-8")


def _compose_hydra_config(config_name: str, overrides: List[str]) -> Dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(CONFIGS_DIR),
        version_base="1.3",
    ):
        cfg = compose(config_name=config_name, overrides=overrides)
    config = ensure_config_dict(OmegaConf.to_container(cfg, resolve=True))
    devices_section = config.get("devices")
    if isinstance(devices_section, dict):
        config["devices"] = devices_section.get("devices", [])
    return config


def precleanup_device_runtime(config_name: str, overrides: List[str]) -> None:
    """Kill orphan workers and remove stale SHM before launching device startup."""
    try:
        config = _compose_hydra_config(config_name, overrides)
        shm_names = collect_device_shm_names(parse_device_configs(config))
    except Exception as exc:
        logger.warning(f"Pre-start shared memory cleanup skipped: {exc}")
        return
    cleanup_device_runtime(shm_names)


def subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    env["LOGURU_COLORIZE"] = "0"
    env["FORCE_COLOR"] = "0"
    if sys.platform == "win32":
        env["PYTHONUTF8"] = "1"
    return enrich_subprocess_env(env)


def strip_ansi(text: str) -> str:
    return _ANSI_STRIP_RE.sub("", text).strip()


def strip_ansi_for_display(text: str) -> str:
    """Remove ANSI color codes but preserve line endings and spacing."""
    return _ANSI_STRIP_RE.sub("", text)


def scan_saved_episodes(output_dir: str) -> Tuple[int, str]:
    root = Path(output_dir)
    if not root.is_dir():
        return 0, "—"

    episodes: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("episode_") or name.endswith("_temp"):
            continue
        if (child / "metadata.json").exists() or (child / "data.mcap").exists():
            episodes.append(child)

    if not episodes:
        return 0, "—"

    episodes.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return len(episodes), episodes[0].name


@dataclass
class ManagedProcess:
    name: str
    process: Optional[subprocess.Popen] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=8000))
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
    episode_stats_revision: int = 0
    log_file_path: Optional[Path] = None
    log_file_handle: Optional[IO[str]] = None

    def append_log(self, line: str) -> None:
        with self.lock:
            if self.logs.maxlen is not None and len(self.logs) == self.logs.maxlen:
                self.log_start_offset += 1
            self.logs.append(line)
            self.log_next_offset += 1

    def read_logs_since(self, offset: int, *, limit: Optional[int] = None) -> Tuple[List[str], int]:
        with self.lock:
            start_offset = self.log_start_offset
            next_offset = self.log_next_offset
            if offset < 0:
                offset = 0
            if offset < start_offset:
                offset = start_offset
            if offset >= next_offset:
                return [], next_offset
            if limit is not None and limit > 0:
                offset = max(offset, next_offset - limit)
            start_index = offset - start_offset
            items = list(islice(self.logs, max(0, start_index), None))
            return items, next_offset

    def clear_logs(self) -> None:
        with self.lock:
            self.logs.clear()
            self.log_start_offset = 0
            self.log_next_offset = 0

    def status_dict(self) -> Dict[str, Any]:
        return {
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


class ProcessManager:
    """Manage device, saver, and visualizer subprocesses."""

    def __init__(self) -> None:
        self.device = ManagedProcess("device")
        self.saver = ManagedProcess("saver")
        self.visualizer = ManagedProcess("visualizer")
        self._output_dir = str(PROJECT_ROOT / DEFAULT_OUTPUT_BASE)
        self._last_device_config_name = DEFAULT_CONFIG_NAME
        self._last_device_overrides: List[str] = []

    def set_output_dir(self, output_dir: str) -> None:
        self._output_dir = output_dir

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def runtime_status(self) -> Dict[str, Any]:
        """Cheap status for high-frequency UI polling.

        Do not scan config files, parse yaml, or scan saved episode folders here.
        """
        return {
            "device": self.device.status_dict(),
            "saver": self.saver.status_dict(),
            "visualizer": self.visualizer.status_dict(),
        }

    def status(self) -> Dict[str, Any]:
        if not self.saver.running:
            count, last_name = scan_saved_episodes(self._output_dir)
            self.saver.saved_episode_count = count
            self.saver.last_episode_name = last_name
        return {
            "device": self.device.status_dict(),
            "saver": self.saver.status_dict(),
            "visualizer": self.visualizer.status_dict(),
            "configs": scan_config_files(),
            "default_config": DEFAULT_CONFIG_NAME,
            "config_defaults": load_config_defaults(DEFAULT_CONFIG_NAME),
            "default_output_dir": str(PROJECT_ROOT / DEFAULT_OUTPUT_BASE),
        }

    def start_device(
        self,
        *,
        config_name: str,
        camera_indices: Dict[str, str],
        encoder_ports: Optional[Dict[str, str]] = None,
    ) -> None:
        if self.device.stopping:
            raise RuntimeError("设备进程正在停止，请稍候再启动。")
        if self.device.running:
            raise RuntimeError("设备进程已在运行。")

        overrides: List[str] = []
        for key, _ in _CAMERA_CAPTURE_KEYS:
            value = (camera_indices.get(key) or "").strip()
            if value:
                overrides.append(f"camera.{key}={value}")

        ports = encoder_ports or {}
        for key in ("left_port", "right_port"):
            value = str(ports.get(key) or "").strip()
            if value:
                overrides.append(f'encoder.{key}="{value}"')

        self._last_device_config_name = config_name
        self._last_device_overrides = list(overrides)
        precleanup_device_runtime(config_name, overrides)

        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "windows_device_start.py"),
            "--config-name",
            config_name,
            *overrides,
        ]
        self._start_process(self.device, cmd, stdin_mode=None, resource_role="recording")
        self.device.running = True

    def stop_device(self) -> None:
        self._stop_process(self.device, timeout_s=DEVICE_PROCESS_STOP_TIMEOUT_S)

    def stop_device_async(self) -> None:
        if (
            not self.device.running
            and self.device.process is None
            and not self.device.stopping
        ):
            return
        threading.Thread(target=self.stop_device, daemon=True).start()

    def start_saver(
        self,
        *,
        config_name: str,
        output_dir: str,
        task_name: str,
        extra_device_type: str,
        extra_hardware_string: str,
        extra_device_zero: str,
        dm3_user_id: str,
        dm3_user_name: str,
        task_description: str,
    ) -> None:
        if self.saver.running:
            raise RuntimeError("数据保存进程已在运行。")

        self._output_dir = output_dir
        count, last_name = scan_saved_episodes(output_dir)
        self.saver.saved_episode_count = count
        self.saver.last_episode_name = last_name

        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "consumer" / "async_data_saver_NEDF3.py"),
            "--output-dir",
            output_dir,
            "--task-name",
            task_name,
            "--extra-device-type",
            extra_device_type,
            "--extra-hardware-string",
            extra_hardware_string,
            "--extra-device-zero",
            extra_device_zero,
            "--dm3-user-id",
            dm3_user_id,
            "--dm3-user-name",
            dm3_user_name,
            "--task-description",
            task_description,
        ]
        env = subprocess_env()
        env["GUI_MODE"] = "1"
        self._start_process(self.saver, cmd, stdin_mode="pipe", env=env, resource_role="recording")
        self.saver.running = True
        self.saver.recording = False
        self.saver.finalizing_episode = False
        self.saver.ready_for_episode = False

    def stop_saver(self) -> None:
        self._stop_process(self.saver, timeout_s=SAVER_PROCESS_STOP_TIMEOUT_S)
        self.saver.recording = False
        self.saver.finalizing_episode = False
        self.saver.ready_for_episode = False

    def stop_saver_async(self) -> None:
        if (
            not self.saver.running
            and self.saver.process is None
            and not self.saver.stopping
        ):
            return
        threading.Thread(target=self.stop_saver, daemon=True).start()

    def send_enter_to_saver(self) -> None:
        proc = self.saver.process
        if proc is None or proc.poll() is not None:
            raise RuntimeError("数据保存进程未运行。")
        if proc.stdin is None:
            raise RuntimeError("数据保存进程 stdin 不可用。")
        proc.stdin.write("\n")
        proc.stdin.flush()

    def run_visualizer(self) -> None:
        if self.visualizer.running:
            raise RuntimeError("可视化进程已在运行。")
        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "nedf3_rerun_visualizer.py"),
        ]
        self._start_process(self.visualizer, cmd, stdin_mode=None, resource_role="normal")
        self.visualizer.running = True

    def _start_process(
        self,
        managed: ManagedProcess,
        cmd: List[str],
        *,
        stdin_mode: Optional[str],
        env: Optional[Dict[str, str]] = None,
        resource_role: str = "normal",
    ) -> None:
        managed.clear_logs()
        managed.exit_code = None
        managed.append_log(f"$ {' '.join(cmd)}\n")

        kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid

        popen_kwargs: Dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "bufsize": 0,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "env": env or subprocess_env(),
            **kwargs,
        }
        if stdin_mode == "pipe":
            popen_kwargs["stdin"] = subprocess.PIPE
        else:
            popen_kwargs["stdin"] = subprocess.DEVNULL

        use_file_stdout = collection_perf_safe_enabled()
        if use_file_stdout:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_path = LOGS_DIR / f"{managed.name}.log"
            log_handle = log_path.open("w", encoding="utf-8", buffering=1)
            managed.log_file_path = log_path
            managed.log_file_handle = log_handle
            popen_kwargs["stdout"] = log_handle
            popen_kwargs["stderr"] = subprocess.STDOUT
            managed.process = subprocess.Popen(cmd, **popen_kwargs)
            self._apply_child_resource_policy(managed, resource_role)
            threading.Thread(
                target=self._tail_process_log,
                args=(managed,),
                daemon=True,
            ).start()
        else:
            managed.log_file_path = None
            managed.log_file_handle = None
            managed.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **popen_kwargs,
            )
            self._apply_child_resource_policy(managed, resource_role)
            for stream, is_stderr in (
                (managed.process.stdout, False),
                (managed.process.stderr, True),
            ):
                threading.Thread(
                    target=self._read_stream,
                    args=(managed, stream, is_stderr),
                    daemon=True,
                ).start()

        threading.Thread(
            target=self._wait_for_exit,
            args=(managed,),
            daemon=True,
        ).start()

    def _apply_child_resource_policy(self, managed: ManagedProcess, resource_role: str) -> None:
        proc = managed.process
        if proc is None or resource_role != "recording":
            return
        try:
            result = apply_recording_process_policy(proc.pid)
            managed.append_log(
                "[resource_policy] "
                f"role={result.role} pid={result.pid} affinity={result.affinity} "
                f"priority={result.priority} errors={result.errors}\n"
            )
            if not result.ok:
                raise RuntimeError(
                    "failed to isolate recording process resources: "
                    f"affinity={result.affinity} priority={result.priority} errors={result.errors}"
                )
        except Exception as exc:
            managed.append_log(f"[resource_policy] failed to apply recording policy: {exc}\n")
            self._stop_process(managed, timeout_s=2.0)
            raise

    def _tail_process_log(self, managed: ManagedProcess) -> None:
        """Follow a subprocess log file to avoid stdout PIPE backpressure."""
        log_path = managed.log_file_path
        if log_path is None:
            return
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                while True:
                    proc = managed.process
                    if proc is None:
                        break
                    line = reader.readline()
                    if line:
                        managed.append_log(line)
                        self._update_runtime_from_log(managed, line)
                        continue
                    if proc.poll() is not None:
                        remainder = reader.read()
                        if remainder:
                            for partial in remainder.splitlines(keepends=True):
                                managed.append_log(partial)
                                self._update_runtime_from_log(managed, partial)
                        break
                    time.sleep(0.05)
        finally:
            self._close_process_log_handle(managed)

    @staticmethod
    def _close_process_log_handle(managed: ManagedProcess) -> None:
        handle = managed.log_file_handle
        if handle is None:
            return
        try:
            handle.close()
        except Exception:
            pass
        managed.log_file_handle = None

    def _read_stream(
        self,
        managed: ManagedProcess,
        stream,
        is_stderr: bool,
    ) -> None:
        prefix = "[stderr] " if is_stderr else ""
        try:
            for line in iter(stream.readline, ""):
                managed.append_log(f"{prefix}{line}")
                self._update_runtime_from_log(managed, line)
        finally:
            stream.close()

    def _update_runtime_from_log(self, managed: ManagedProcess, line: str) -> None:
        plain = strip_ansi(line)
        if managed.name == "saver":
            if "Press Enter to start data acquisition." in plain:
                managed.recording = False
                managed.finalizing_episode = False
                managed.ready_for_episode = True
            elif "Data acquisition started." in plain:
                managed.recording = True
                managed.finalizing_episode = False
                managed.ready_for_episode = False
            elif any(
                marker in plain
                for marker in (
                    "Data acquisition stopped by user.",
                    "Episode was discarded, skipping finalization",
                )
            ):
                managed.recording = False
                managed.finalizing_episode = True
                managed.ready_for_episode = False
            elif "Episode completed in" in plain:
                managed.recording = False
                managed.finalizing_episode = True

            match = _EPISODE_STATUS_RE.search(plain)
            if match:
                managed.saved_episode_count = int(match.group(1))
                managed.last_episode_name = match.group(2)
            else:
                finalized = _FINALIZED_EPISODE_RE.search(plain)
                if finalized:
                    managed.last_episode_name = finalized.group(1).strip()
                discarded = _DISCARDED_EPISODE_RE.search(plain)
                if discarded:
                    managed.last_episode_name = discarded.group(1).strip()

            if _EPISODE_TIMING_RE.search(plain):
                managed.episode_stats_revision += 1

            if not collection_perf_safe_enabled() and (
                _EPISODE_STATUS_RE.search(plain)
                or _FINALIZED_EPISODE_RE.search(plain)
                or _DISCARDED_EPISODE_RE.search(plain)
            ):
                count, last_name = scan_saved_episodes(self._output_dir)
                managed.saved_episode_count = count
                if last_name != "—":
                    managed.last_episode_name = last_name

    def _wait_for_exit(self, managed: ManagedProcess) -> None:
        proc = managed.process
        if proc is None:
            return
        proc.wait()
        managed.exit_code = proc.returncode
        managed.running = False
        managed.recording = False
        managed.finalizing_episode = False
        managed.ready_for_episode = False
        managed.append_log(f"\n[process exited with code {proc.returncode}]\n")

    def _stop_process(self, managed: ManagedProcess, *, timeout_s: float = 12.0) -> None:
        if managed.stopping:
            return
        proc = managed.process
        if proc is None or proc.poll() is not None:
            managed.running = False
            managed.finalizing_episode = False
            managed.ready_for_episode = False
            managed.process = None
            managed.stopping = False
            return

        managed.stopping = True
        try:
            try:
                if sys.platform == "win32":
                    proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()

            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"{managed.name} process did not exit within {timeout_s:.0f}s; "
                    "force killing process tree"
                )
                if not kill_process_tree(proc.pid, wait_timeout_s=5.0):
                    logger.error(
                        f"{managed.name} root PID {proc.pid} survived process-tree kill"
                    )
                proc.wait(timeout=3)
        finally:
            self._close_process_log_handle(managed)
            managed.running = False
            managed.process = None
            managed.stopping = False
            if managed.name == "device":
                precleanup_device_runtime(
                    self._last_device_config_name,
                    self._last_device_overrides,
                )
