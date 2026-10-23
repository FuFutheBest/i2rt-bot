"""Discover and stop stale SuperInference launcher processes before startup.

The four hardware-facing entry points can be launched independently, so a
previous invocation may still own a serial port, robot connection, camera, or
shared-memory producer.  This module uses ``psutil`` to identify only known
launcher commands owned by the current user.  Matching descendants are
included because multiprocessing workers do not retain the launcher's command
line.

Startup cleanup sends ``SIGTERM`` and waits for a bounded interval.  It never
falls back to ``SIGKILL`` automatically: a surviving motion worker must block a
new hardware start instead of being forcibly terminated without its safety
cleanup path.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import shutil
import subprocess
from typing import Iterable, Sequence

import psutil


PROCESS_ROLES = (
    "rollout_preflight",
    "device_starter",
    "control_panel",
    "visualize_assembly",
)
DEFAULT_STARTUP_ROLES = frozenset(PROCESS_ROLES)
DEFAULT_GRACE_TIMEOUT_S = 10.0
FORCE_CLEANUP_ENV = "SUPERINFERENCE_FORCE_PROCESS_CLEANUP"


@dataclass(frozen=True)
class ProcessCandidate:
    """A process identified as a launcher or descendant of one."""

    pid: int
    ppid: int
    role: str
    command: str
    root_pid: int


@dataclass
class ProcessCleanupReport:
    """Results from one discovery/termination pass."""

    candidates: list[ProcessCandidate] = field(default_factory=list)
    terminated_pids: list[int] = field(default_factory=list)
    survivors: list[ProcessCandidate] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.survivors and not self.errors

    def summary(self) -> str:
        """Return a concise operator-facing summary."""
        if not self.candidates:
            return "no related processes found"
        parts = [
            f"found {len(self.candidates)} related process(es)",
            f"stopped {len(self.terminated_pids)}",
        ]
        if self.terminated_pids:
            parts.append(
                "terminated PIDs=" + ",".join(str(pid) for pid in self.terminated_pids)
            )
        if self.survivors:
            parts.append(f"{len(self.survivors)} survivor(s)")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return ", ".join(parts)


class ProcessCleanupError(RuntimeError):
    """Raised when a related process survived startup cleanup."""

    def __init__(self, report: ProcessCleanupReport) -> None:
        self.report = report
        details = "; ".join(
            f"PID {item.pid} ({item.role}): {item.command}" for item in report.survivors
        )
        errors = "; ".join(
            f"PID {pid}: {message}" for pid, message in sorted(report.errors.items())
        )
        suffix = "; ".join(part for part in (details, errors) if part)
        message = "related process cleanup incomplete"
        if suffix:
            message += f": {suffix}"
        super().__init__(message)


def force_cleanup_requested() -> bool:
    """Return whether the operator explicitly enabled forced cleanup."""
    return os.environ.get(FORCE_CLEANUP_ENV, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _command_text(command: Sequence[str]) -> str:
    return " ".join(str(item) for item in command if str(item))


def classify_command(command: Sequence[str]) -> str | None:
    """Return the known launcher role for a process command line."""
    args = [str(item) for item in command]
    basenames = {Path(item).name.casefold() for item in args}
    if "rollout_device_preflight.py" in basenames:
        return "rollout_preflight"
    if "device_starter.py" in basenames:
        return "device_starter"
    if "visualize_flexiv_assembly.py" in basenames:
        return "visualize_assembly"

    for index, argument in enumerate(args[:-1]):
        if argument != "-m":
            continue
        module = args[index + 1].casefold()
        if module in {"control_panel", "incubator.control_panel"}:
            return "control_panel"
        if module in {"device_starter", "scripts.device_starter"}:
            return "device_starter"

    # Frozen release panels use a dedicated executable instead of ``-m``.
    if any(
        name in {"superinference-control-panel", "control-panel"} for name in basenames
    ):
        return "control_panel"
    return None


def _safe_process_info(
    process: psutil.Process,
) -> tuple[int, int, str, tuple[str, ...]] | None:
    try:
        info = process.as_dict(attrs=("pid", "ppid", "cmdline", "name"))
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    pid = int(info.get("pid") or process.pid)
    ppid = int(info.get("ppid") or 0)
    raw_command = info.get("cmdline") or []
    command = tuple(str(item) for item in raw_command if str(item))
    if not command:
        name = str(info.get("name") or "")
        command = (name,) if name else ()
    return pid, ppid, str(info.get("name") or ""), command


def _systemd_unit_for_pid(pid: int) -> str | None:
    """Return the systemd service unit owning ``pid``, when discoverable."""
    try:
        cgroup_text = Path(f"/proc/{int(pid)}/cgroup").read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None
    for line in cgroup_text.splitlines():
        _hierarchy, _controllers, path = line.partition(":")
        unit = path.rsplit("/", 1)[-1]
        if unit.endswith(".service"):
            return unit
    return None


def _stop_supervised_services(
    candidates: Sequence[ProcessCandidate],
    *,
    timeout_s: float,
) -> None:
    """Prevent systemd from respawning a process being force-cleaned.

    ``systemctl stop --no-block`` only changes the supervisor state; the normal
    signal/kill pass below still owns process termination and its bounded wait.
    This helper is used only by explicit forced cleanup.
    """
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return
    units = {
        unit
        for candidate in candidates
        if (unit := _systemd_unit_for_pid(candidate.pid)) is not None
    }
    if not units:
        return
    command_timeout = max(1.0, min(float(timeout_s) if timeout_s else 1.0, 5.0))
    for unit in sorted(units):
        try:
            subprocess.run(
                [systemctl, "--user", "stop", "--no-block", unit],
                capture_output=True,
                text=True,
                timeout=command_timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # The process pass remains authoritative. If the service respawns,
            # the post-cleanup discovery reports the replacement PID.
            continue


def discover_related_processes(
    roles: Iterable[str] = DEFAULT_STARTUP_ROLES,
    *,
    exclude_pids: Iterable[int] = (),
    same_user: bool = True,
) -> list[ProcessCandidate]:
    """Find launcher processes and all visible descendants.

    Process discovery is intentionally read-only.  The current process and
    processes owned by another user are excluded by default.
    """
    requested = {str(role) for role in roles}
    excluded = {int(pid) for pid in exclude_pids}
    excluded.add(os.getpid())
    current_user: str | None = None
    if same_user:
        try:
            current_user = psutil.Process(os.getpid()).username()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            current_user = None

    records: dict[int, tuple[int, str, tuple[str, ...]]] = {}
    roots: dict[int, str] = {}
    for process in psutil.process_iter(
        attrs=("pid", "ppid", "cmdline", "name", "username")
    ):
        info = _safe_process_info(process)
        if info is None:
            continue
        pid, ppid, _name, command = info
        if pid in excluded:
            continue
        if same_user:
            try:
                username = process.username()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            if current_user is not None and username != current_user:
                continue
        records[pid] = (ppid, _command_text(command), command)
        role = classify_command(command)
        if role in requested:
            roots[pid] = role

    children: dict[int, list[int]] = defaultdict(list)
    for pid, (ppid, _command, _argv) in records.items():
        children[ppid].append(pid)

    candidates: dict[int, ProcessCandidate] = {}
    queue: deque[tuple[int, int, str]] = deque(
        (pid, pid, role) for pid, role in roots.items()
    )
    while queue:
        pid, root_pid, root_role = queue.popleft()
        record = records.get(pid)
        if record is None or pid in candidates:
            continue
        ppid, command, _argv = record
        role = root_role if pid == root_pid else f"child_of_{root_role}"
        candidates[pid] = ProcessCandidate(pid, ppid, role, command, root_pid)
        for child_pid in children.get(pid, []):
            queue.append((child_pid, root_pid, root_role))

    return sorted(candidates.values(), key=lambda item: (item.root_pid, item.pid))


def cleanup_related_processes(
    roles: Iterable[str] = DEFAULT_STARTUP_ROLES,
    *,
    exclude_pids: Iterable[int] = (),
    grace_timeout_s: float = DEFAULT_GRACE_TIMEOUT_S,
    same_user: bool = True,
    force: bool = False,
) -> ProcessCleanupReport:
    """Terminate related processes and report any that remain alive.

    ``force`` is available for an explicit operator/tooling call, but startup
    callers should leave it disabled so a motion process cannot be hard-killed
    before its robot-rest and shared-memory cleanup runs.
    """
    requested_roles = tuple(str(role) for role in roles)
    candidates = discover_related_processes(
        requested_roles,
        exclude_pids=exclude_pids,
        same_user=same_user,
    )
    report = ProcessCleanupReport(candidates=candidates)
    if not candidates:
        return report

    if force:
        _stop_supervised_services(candidates, timeout_s=grace_timeout_s)

    handles: list[psutil.Process] = []
    for candidate in candidates:
        try:
            process = psutil.Process(candidate.pid)
            process.send_signal(signal.SIGTERM)
            handles.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            report.terminated_pids.append(candidate.pid)
        except (psutil.AccessDenied, OSError) as exc:
            report.errors[candidate.pid] = f"{type(exc).__name__}: {exc}"

    timeout = max(0.0, float(grace_timeout_s))
    gone, alive = psutil.wait_procs(handles, timeout=timeout)
    report.terminated_pids.extend(process.pid for process in gone)

    if alive and force:
        force_handles: list[psutil.Process] = []
        for process in alive:
            try:
                process.kill()
                force_handles.append(process)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                report.terminated_pids.append(process.pid)
            except (psutil.AccessDenied, OSError) as exc:
                report.errors[process.pid] = f"{type(exc).__name__}: {exc}"
        forced_gone, alive = psutil.wait_procs(force_handles, timeout=timeout)
        report.terminated_pids.extend(process.pid for process in forced_gone)

    candidate_by_pid = {candidate.pid: candidate for candidate in candidates}
    survivor_by_pid = {
        process.pid: candidate_by_pid[process.pid]
        for process in alive
        if process.pid in candidate_by_pid
    }

    # A supervisor such as systemd may immediately respawn a panel after its
    # original PID exits.  Re-scan after the wait so a replacement PID cannot
    # race the caller's hardware startup.
    for candidate in discover_related_processes(
        requested_roles,
        exclude_pids=exclude_pids,
        same_user=same_user,
    ):
        if candidate.pid not in report.terminated_pids:
            survivor_by_pid[candidate.pid] = candidate
    report.survivors = sorted(
        survivor_by_pid.values(), key=lambda item: (item.root_pid, item.pid)
    )
    report.terminated_pids = sorted(set(report.terminated_pids))
    return report


def cleanup_before_startup(
    role: str,
    *,
    grace_timeout_s: float = DEFAULT_GRACE_TIMEOUT_S,
    force: bool = False,
) -> ProcessCleanupReport:
    """Clean all known launchers before starting ``role``.

    The current process is always excluded, so this is safe to call from the
    entry point itself.  A survivor raises ``ProcessCleanupError`` and callers
    should abort startup to avoid two processes competing for hardware.
    """
    normalized_role = str(role).strip()
    if normalized_role not in PROCESS_ROLES:
        raise ValueError(f"unknown startup process role: {role!r}")
    report = cleanup_related_processes(
        DEFAULT_STARTUP_ROLES,
        grace_timeout_s=grace_timeout_s,
        force=force,
    )
    if report.survivors or report.errors:
        raise ProcessCleanupError(report)
    return report


__all__ = [
    "DEFAULT_GRACE_TIMEOUT_S",
    "DEFAULT_STARTUP_ROLES",
    "FORCE_CLEANUP_ENV",
    "PROCESS_ROLES",
    "ProcessCandidate",
    "ProcessCleanupError",
    "ProcessCleanupReport",
    "classify_command",
    "cleanup_before_startup",
    "cleanup_related_processes",
    "discover_related_processes",
    "force_cleanup_requested",
]
