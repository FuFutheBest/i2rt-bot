from __future__ import annotations

import os
import signal
from typing import Any

import pytest

from incubator.control_panel import process_cleanup


class _FakeProcess:
    def __init__(
        self, pid: int, ppid: int, command: list[str], username: str = "tester"
    ) -> None:
        self.pid = pid
        self._ppid = ppid
        self._command = command
        self._username = username
        self.signals: list[int] = []
        self.killed = False

    def as_dict(self, attrs: Any = None) -> dict[str, Any]:
        del attrs
        return {
            "pid": self.pid,
            "ppid": self._ppid,
            "cmdline": self._command,
            "name": self._command[0] if self._command else "",
        }

    def username(self) -> str:
        return self._username

    def send_signal(self, value: int) -> None:
        self.signals.append(value)

    def kill(self) -> None:
        self.killed = True


def test_classify_known_launchers_and_ignores_similar_test_names() -> None:
    assert (
        process_cleanup.classify_command(["python", "device_starter.py"])
        == "device_starter"
    )
    assert (
        process_cleanup.classify_command(["python", "rollout_device_preflight.py"])
        == "rollout_preflight"
    )
    assert (
        process_cleanup.classify_command(["python", "visualize_flexiv_assembly.py"])
        == "visualize_assembly"
    )
    assert (
        process_cleanup.classify_command(["python", "-m", "control_panel"])
        == "control_panel"
    )
    assert (
        process_cleanup.classify_command(["pytest", "test_device_starter_control.py"])
        is None
    )


def test_discover_related_processes_includes_launcher_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = [
        _FakeProcess(101, 1, ["python", "/repo/device_starter.py"]),
        _FakeProcess(102, 101, ["python", "-c", "spawn_main()"]),
        _FakeProcess(103, 1, ["python", "unrelated.py"]),
        _FakeProcess(999, 1, ["pytest"]),
    ]
    by_pid = {process.pid: process for process in processes}
    monkeypatch.setattr(process_cleanup.os, "getpid", lambda: 999)
    monkeypatch.setattr(
        process_cleanup.psutil, "process_iter", lambda **_: iter(processes)
    )
    monkeypatch.setattr(process_cleanup.psutil, "Process", lambda pid: by_pid[pid])

    discovered = process_cleanup.discover_related_processes({"device_starter"})

    assert [(item.pid, item.role, item.root_pid) for item in discovered] == [
        (101, "device_starter", 101),
        (102, "child_of_device_starter", 101),
    ]


def test_cleanup_reports_survivor_without_force_killing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = process_cleanup.ProcessCandidate(
        pid=101,
        ppid=1,
        role="device_starter",
        command="python device_starter.py",
        root_pid=101,
    )
    fake = _FakeProcess(101, 1, ["python", "device_starter.py"])
    monkeypatch.setattr(
        process_cleanup,
        "discover_related_processes",
        lambda *_, **__: [candidate],
    )
    monkeypatch.setattr(process_cleanup.psutil, "Process", lambda pid: fake)
    monkeypatch.setattr(
        process_cleanup.psutil, "wait_procs", lambda handles, timeout: ([], handles)
    )

    report = process_cleanup.cleanup_related_processes(
        {"device_starter"},
        grace_timeout_s=0.0,
    )

    assert fake.signals
    assert report.survivors == [candidate]
    assert report.terminated_pids == []
    assert fake.killed is False


def test_cleanup_detects_a_supervisor_respawned_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = process_cleanup.ProcessCandidate(101, 1, "control_panel", "old", 101)
    replacement = process_cleanup.ProcessCandidate(202, 1, "control_panel", "new", 202)
    fake = _FakeProcess(101, 1, ["python", "-m", "control_panel"])
    discoveries = iter(([original], [replacement]))
    monkeypatch.setattr(
        process_cleanup,
        "discover_related_processes",
        lambda *_, **__: next(discoveries),
    )
    monkeypatch.setattr(process_cleanup.psutil, "Process", lambda _pid: fake)
    monkeypatch.setattr(
        process_cleanup.psutil, "wait_procs", lambda handles, timeout: (handles, [])
    )

    report = process_cleanup.cleanup_related_processes(
        {"control_panel"},
        grace_timeout_s=0.0,
    )

    assert report.terminated_pids == [101]
    assert report.survivors == [replacement]


def test_cleanup_before_startup_refuses_to_continue_with_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = process_cleanup.ProcessCandidate(101, 1, "control_panel", "panel", 101)
    report = process_cleanup.ProcessCleanupReport(
        candidates=[candidate], survivors=[candidate]
    )
    monkeypatch.setattr(
        process_cleanup, "cleanup_related_processes", lambda *_, **__: report
    )

    with pytest.raises(process_cleanup.ProcessCleanupError, match="PID 101"):
        process_cleanup.cleanup_before_startup("device_starter")


def test_cleanup_never_targets_current_process_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_cleanup, "discover_related_processes", lambda *_, **__: []
    )
    report = process_cleanup.cleanup_before_startup("control_panel")
    assert report.ok
    assert os.getpid() not in report.terminated_pids


def test_force_cleanup_requested_reads_explicit_environment_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(process_cleanup.FORCE_CLEANUP_ENV, "yes")
    assert process_cleanup.force_cleanup_requested() is True

    monkeypatch.setenv(process_cleanup.FORCE_CLEANUP_ENV, "0")
    assert process_cleanup.force_cleanup_requested() is False


def test_force_cleanup_stops_supervised_unit_and_kills_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = process_cleanup.ProcessCandidate(
        pid=101,
        ppid=1,
        role="control_panel",
        command="python -m control_panel",
        root_pid=101,
    )
    fake = _FakeProcess(101, 1, ["python", "-m", "control_panel"])
    discoveries = iter(([candidate], []))
    waits = iter((([], [fake]), ([fake], [])))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        process_cleanup, "discover_related_processes", lambda *_, **__: next(discoveries)
    )
    monkeypatch.setattr(process_cleanup.psutil, "Process", lambda _pid: fake)
    monkeypatch.setattr(
        process_cleanup.psutil,
        "wait_procs",
        lambda handles, timeout: next(waits),
    )
    monkeypatch.setattr(process_cleanup.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr(
        process_cleanup,
        "_systemd_unit_for_pid",
        lambda _pid: "cp-test.service",
    )
    monkeypatch.setattr(
        process_cleanup.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )

    report = process_cleanup.cleanup_related_processes(
        {"control_panel"},
        grace_timeout_s=0.0,
        force=True,
    )

    assert calls == [["/usr/bin/systemctl", "--user", "stop", "--no-block", "cp-test.service"]]
    assert fake.signals == [signal.SIGTERM]
    assert fake.killed is True
    assert report.ok
    assert report.terminated_pids == [101]
