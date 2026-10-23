#!/usr/bin/env python3
"""Windows process resource policy for keeping upload work away from recording."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import psutil


UPLOAD_AFFINITY_ENV = "SI_UPLOAD_CPU_AFFINITY"
UPLOAD_CORE_COUNT_ENV = "SI_UPLOAD_CORE_COUNT"
GUI_PROCESS_PRIORITY_ENV = "SI_GUI_PROCESS_PRIORITY"
UPLOAD_THREAD_PRIORITY_ENV = "SI_UPLOAD_THREAD_PRIORITY"
# Legacy alias: SI_UPLOAD_PRIORITY now controls upload worker thread priority only.
UPLOAD_PRIORITY_ENV = "SI_UPLOAD_PRIORITY"
UPLOAD_MEMORY_LIMIT_MB_ENV = "SI_UPLOAD_MEMORY_LIMIT_MB"
RECORDING_AFFINITY_ENV = "SI_RECORDING_CPU_AFFINITY"
RECORDING_PRIORITY_ENV = "SI_RECORDING_PRIORITY"
TRACKER_AFFINITY_ENV = "SI_TRACKER_CPU_AFFINITY"
TRACKER_CORE_COUNT_ENV = "SI_TRACKER_CORE_COUNT"
TRACKER_PRIORITY_ENV = "SI_TRACKER_PRIORITY"
TRACKER_ISOLATION_ENV = "SI_TRACKER_ISOLATION"
POLICY_STATUS_PATH_ENV = "SI_RESOURCE_POLICY_STATUS_PATH"

DEFAULT_UPLOAD_CORE_COUNT = 2
DEFAULT_GUI_PROCESS_PRIORITY = "normal"
DEFAULT_UPLOAD_THREAD_PRIORITY = "below_normal"
DEFAULT_UPLOAD_MEMORY_LIMIT_MB = 4096
DEFAULT_RECORDING_PRIORITY = "high"
DEFAULT_TRACKER_CORE_COUNT = 2
DEFAULT_TRACKER_PRIORITY = "high"
SYSTEM_AFFINITY_ENV = "SI_SYSTEM_CPU_AFFINITY"

# Windows thread priorities (SetThreadPriority), not process priority classes.
THREAD_PRIORITY_VALUES = {
    "idle": -15,
    "below_normal": -1,
    "normal": 0,
    "above_normal": 1,
    "high": 2,
}

PRIORITY_CLASSES = {
    "idle": psutil.IDLE_PRIORITY_CLASS if sys.platform == "win32" else None,
    "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else None,
    "normal": psutil.NORMAL_PRIORITY_CLASS if sys.platform == "win32" else None,
    "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else None,
    "high": psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else None,
    "realtime": psutil.REALTIME_PRIORITY_CLASS if sys.platform == "win32" else None,
}


@dataclass(frozen=True)
class ResourcePolicyResult:
    role: str
    pid: int
    affinity: List[int]
    priority: str
    memory_limit_mb: Optional[int]
    errors: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_cpu_list(value: str, available: Iterable[int]) -> List[int]:
    available_set = set(available)
    cpus: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            candidates = range(start, end + step, step)
        else:
            candidates = [int(chunk)]
        for cpu in candidates:
            if cpu in available_set and cpu not in cpus:
                cpus.append(cpu)
    return cpus


def get_system_affinity() -> List[int]:
    """Return the CPU set used as the partition baseline for this Windows session."""
    override = os.environ.get(SYSTEM_AFFINITY_ENV, "").strip()
    logical = list(range(psutil.cpu_count(logical=True) or 1))
    if override:
        try:
            parsed = parse_cpu_list(override, logical)
            if parsed:
                return parsed
        except ValueError:
            pass
    try:
        return psutil.Process(os.getpid()).cpu_affinity()
    except Exception:
        return logical


def initialize_system_affinity() -> List[int]:
    """Capture the pre-partition CPU set so child policies share the same baseline."""
    affinity = get_system_affinity()
    if affinity and not os.environ.get(SYSTEM_AFFINITY_ENV, "").strip():
        os.environ[SYSTEM_AFFINITY_ENV] = ",".join(str(cpu) for cpu in affinity)
    return affinity


def _default_upload_affinity(available: List[int]) -> List[int]:
    if not available:
        return []
    requested = _env_int(UPLOAD_CORE_COUNT_ENV, DEFAULT_UPLOAD_CORE_COUNT)
    count = max(1, min(len(available), requested))
    return available[-count:]


def _upload_affinity_for_partition(available: List[int]) -> List[int]:
    affinity_text = os.environ.get(UPLOAD_AFFINITY_ENV, "").strip()
    if affinity_text:
        try:
            return parse_cpu_list(affinity_text, available)
        except ValueError:
            return []
    return _default_upload_affinity(available)


def _tracker_isolation_enabled() -> bool:
    value = os.environ.get(TRACKER_ISOLATION_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _recording_pool(available: List[int]) -> List[int]:
    upload_affinity = set(_upload_affinity_for_partition(available))
    recording_affinity = [cpu for cpu in available if cpu not in upload_affinity]
    return recording_affinity or list(available)


def _tracker_affinity_for_partition(available: List[int]) -> List[int]:
    if not _tracker_isolation_enabled():
        return []
    affinity_text = os.environ.get(TRACKER_AFFINITY_ENV, "").strip()
    if affinity_text:
        try:
            return parse_cpu_list(affinity_text, _recording_pool(available))
        except ValueError:
            return []
    recording_affinity = _recording_pool(available)
    if not recording_affinity:
        return []
    requested = _env_int(TRACKER_CORE_COUNT_ENV, DEFAULT_TRACKER_CORE_COUNT)
    count = max(0, min(len(recording_affinity), requested))
    if count == 0:
        return []
    return recording_affinity[:count]


def _default_recording_affinity(available: List[int]) -> List[int]:
    tracker_affinity = set(_tracker_affinity_for_partition(available))
    recording_affinity = [cpu for cpu in _recording_pool(available) if cpu not in tracker_affinity]
    return recording_affinity or _recording_pool(available)


def _default_tracker_affinity(available: List[int]) -> List[int]:
    return _tracker_affinity_for_partition(available)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _priority_name(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    return value if value in PRIORITY_CLASSES else default


def _upload_thread_priority_name() -> str:
    explicit = os.environ.get(UPLOAD_THREAD_PRIORITY_ENV, "").strip().lower()
    if explicit and explicit in THREAD_PRIORITY_VALUES:
        return explicit
    legacy = os.environ.get(UPLOAD_PRIORITY_ENV, "").strip().lower()
    if legacy and legacy in THREAD_PRIORITY_VALUES:
        return legacy
    return DEFAULT_UPLOAD_THREAD_PRIORITY


def _set_working_set_limit(memory_limit_mb: int, errors: List[str]) -> None:
    if sys.platform != "win32" or memory_limit_mb <= 0:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessWorkingSetSizeEx.argtypes = (
        wintypes.HANDLE,
        ctypes.c_size_t,
        ctypes.c_size_t,
        wintypes.DWORD,
    )
    kernel32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    kernel32.GetProcessWorkingSetSizeEx.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    current_min = ctypes.c_size_t()
    current_max = ctypes.c_size_t()
    flags = wintypes.DWORD()
    if not kernel32.GetProcessWorkingSetSizeEx(
        handle,
        ctypes.byref(current_min),
        ctypes.byref(current_max),
        ctypes.byref(flags),
    ):
        errors.append(f"read_working_set_failed={ctypes.get_last_error()}")
        return
    bytes_limit = int(memory_limit_mb) * 1024 * 1024
    min_bytes = min(int(current_min.value), bytes_limit)
    if not kernel32.SetProcessWorkingSetSizeEx(handle, min_bytes, bytes_limit, flags.value):
        errors.append(f"working_set_limit_failed={ctypes.get_last_error()}")


def apply_upload_process_policy() -> ResourcePolicyResult:
    """Pin the GUI process to upload cores while keeping UI process priority normal."""
    initialize_system_affinity()
    return _apply_policy(
        role="upload",
        affinity_env=UPLOAD_AFFINITY_ENV,
        priority_env=GUI_PROCESS_PRIORITY_ENV,
        default_priority=DEFAULT_GUI_PROCESS_PRIORITY,
        default_affinity_factory=_default_upload_affinity,
        memory_limit_mb=_env_int(UPLOAD_MEMORY_LIMIT_MB_ENV, DEFAULT_UPLOAD_MEMORY_LIMIT_MB),
    )


def apply_upload_thread_policy() -> str:
    """Lower only the calling thread so upload I/O does not compete with recording."""
    priority = _upload_thread_priority_name()
    if sys.platform != "win32":
        return priority
    thread_value = THREAD_PRIORITY_VALUES.get(priority)
    if thread_value is None:
        return priority
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentThread.restype = wintypes.HANDLE
        kernel32.SetThreadPriority.argtypes = (wintypes.HANDLE, ctypes.c_int)
        kernel32.SetThreadPriority.restype = wintypes.BOOL
        handle = kernel32.GetCurrentThread()
        if not kernel32.SetThreadPriority(handle, thread_value):
            raise OSError(ctypes.get_last_error(), "SetThreadPriority failed")
    except Exception:
        pass
    return priority


def apply_recording_process_policy(pid: int) -> ResourcePolicyResult:
    """Keep recording subprocesses off upload and dedicated tracker cores."""
    initialize_system_affinity()
    return _apply_policy(
        role="recording",
        pid=pid,
        affinity_env=RECORDING_AFFINITY_ENV,
        priority_env=RECORDING_PRIORITY_ENV,
        default_priority=DEFAULT_RECORDING_PRIORITY,
        default_affinity_factory=_default_recording_affinity,
        memory_limit_mb=None,
    )


def apply_tracker_process_policy(pid: Optional[int] = None) -> ResourcePolicyResult:
    """Reserve dedicated CPU cores for the ViveTracker OpenXR polling loop."""
    initialize_system_affinity()
    return _apply_policy(
        role="tracker",
        pid=pid,
        affinity_env=TRACKER_AFFINITY_ENV,
        priority_env=TRACKER_PRIORITY_ENV,
        default_priority=DEFAULT_TRACKER_PRIORITY,
        default_affinity_factory=_default_tracker_affinity,
        memory_limit_mb=None,
    )


def _apply_policy(
    *,
    role: str,
    affinity_env: str,
    priority_env: str,
    default_priority: str,
    default_affinity_factory,
    memory_limit_mb: Optional[int],
    pid: Optional[int] = None,
) -> ResourcePolicyResult:
    process = psutil.Process(os.getpid() if pid is None else pid)
    errors: List[str] = []
    available = get_system_affinity()

    affinity_text = os.environ.get(affinity_env, "").strip()
    try:
        affinity = parse_cpu_list(affinity_text, available) if affinity_text else default_affinity_factory(available)
        if affinity:
            process.cpu_affinity(affinity)
    except Exception as exc:
        affinity = []
        errors.append(f"set_affinity_failed={exc}")

    priority = _priority_name(priority_env, default_priority)
    priority_class = PRIORITY_CLASSES.get(priority)
    if priority_class is not None:
        try:
            process.nice(priority_class)
        except Exception as exc:
            errors.append(f"set_priority_failed={exc}")

    if memory_limit_mb is not None and memory_limit_mb > 0 and pid is None:
        _set_working_set_limit(memory_limit_mb, errors)

    result = ResourcePolicyResult(
        role=role,
        pid=process.pid,
        affinity=affinity,
        priority=priority,
        memory_limit_mb=memory_limit_mb,
        errors=errors,
    )
    write_policy_status(result)
    return result


def expected_partition() -> dict:
    available = get_system_affinity()
    upload_affinity = _upload_affinity_for_partition(available)
    tracker_affinity = _tracker_affinity_for_partition(available)
    recording_affinity = _default_recording_affinity(available)
    return {
        "system_affinity": available,
        "upload_affinity": upload_affinity,
        "tracker_affinity": tracker_affinity,
        "recording_affinity": recording_affinity,
        "upload_core_count": len(upload_affinity),
        "tracker_core_count": len(tracker_affinity),
        "recording_core_count": len(recording_affinity),
        "tracker_isolation_enabled": _tracker_isolation_enabled(),
        "upload_memory_limit_mb": _env_int(UPLOAD_MEMORY_LIMIT_MB_ENV, DEFAULT_UPLOAD_MEMORY_LIMIT_MB),
        "gui_process_priority": _priority_name(GUI_PROCESS_PRIORITY_ENV, DEFAULT_GUI_PROCESS_PRIORITY),
        "upload_thread_priority": _upload_thread_priority_name(),
        "recording_priority": _priority_name(RECORDING_PRIORITY_ENV, DEFAULT_RECORDING_PRIORITY),
        "tracker_priority": _priority_name(TRACKER_PRIORITY_ENV, DEFAULT_TRACKER_PRIORITY),
    }


def write_policy_status(result: ResourcePolicyResult) -> None:
    status_path_text = os.environ.get(POLICY_STATUS_PATH_ENV, "").strip()
    if not status_path_text:
        return
    try:
        from datetime import datetime, timezone

        status_path = Path(status_path_text)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if status_path.is_file():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    existing = payload
            except Exception:
                existing = {}
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        existing["partition"] = expected_partition()
        existing[result.role] = {
            "pid": result.pid,
            "affinity": result.affinity,
            "priority": result.priority,
            "memory_limit_mb": result.memory_limit_mb,
            "ok": result.ok,
            "errors": result.errors,
        }
        tmp = status_path.with_suffix(status_path.suffix + ".tmp")
        tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(status_path)
    except Exception:
        return

