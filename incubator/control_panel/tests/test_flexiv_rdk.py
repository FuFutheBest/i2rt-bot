from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from incubator.control_panel import flexiv_rdk
from incubator.control_panel.flexiv_rdk import (
    RDK_ROOT_ENV,
    RDK_SITE_ENV,
    RDK_VERSION_ENV,
    WORKER_RDK_SITE_KEY,
    WORKER_RDK_VERSION_KEY,
    FlexivRDKCandidate,
    FlexivRDKProbeAttempt,
    FlexivRDKProbeError,
    FlexivRDKResolver,
    FlexivRDKSelection,
    discover_flexiv_rdk_candidates,
    loaded_flexiv_rdk_version,
)
from incubator.control_panel.lifecycle import DeviceLifecycleManager, DeviceStartupError


def _rdk_root(tmp_path: Path) -> Path:
    root = tmp_path / "flexivrdk"
    for version in ("1.7.0", "1.8.0", "1.8.1", "1.9.0", "1.9.1", "1.9.2"):
        (root / f"rdk-{version}/site-packages").mkdir(parents=True)
    return root


def _rizon_config() -> dict[str, Any]:
    return {
        "devices": [
            {
                "class": "RizonRobot",
                "device_class_id": 2,
                "robot_sn": ["Rizon4-012345", "Rizon4R-012346"],
                "data_shape": [44],
                "data_dtype": "float64",
                "fps": 100.0,
            }
        ],
        "global_settings": {},
        "runtime": {"signal_files": {}},
    }


def test_packaged_inventory_is_firmware_versioned_and_newest_first(tmp_path) -> None:
    root = _rdk_root(tmp_path)

    candidates = discover_flexiv_rdk_candidates(
        environment={RDK_ROOT_ENV: str(root)}
    )

    assert [candidate.version for candidate in candidates] == [
        "1.9.2",
        "1.9.1",
        "1.9.0",
        "1.8.1",
        "1.8.0",
        "1.7.0",
    ]
    assert all("rdk-" in str(candidate.site_packages) for candidate in candidates)


def test_explicit_firmware_override_selects_only_that_rdk(tmp_path) -> None:
    root = _rdk_root(tmp_path)

    candidates = discover_flexiv_rdk_candidates(
        explicit_version="1.9.0",
        environment={RDK_ROOT_ENV: str(root)},
    )

    assert [candidate.version for candidate in candidates] == ["1.9.0"]


def _fake_rdk_distribution(site: Path, version: str) -> Path:
    module = site / "flexivrdk/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    dist_info = site / f"flexivrdk-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: flexivrdk\nVersion: {version}\n",
        encoding="utf-8",
    )
    return module


def test_selected_distribution_wins_over_stale_module_version(tmp_path) -> None:
    site = tmp_path / "rdk-1.8.1/site-packages"
    module = _fake_rdk_distribution(site, "1.8.1")
    sdk = SimpleNamespace(__file__=str(module), __version__="1.8.0")

    version = loaded_flexiv_rdk_version(
        sdk,
        environment={RDK_SITE_ENV: str(site), RDK_VERSION_ENV: "1.8.1"},
    )

    assert version == "1.8.1"


def test_selected_distribution_rejects_module_from_another_profile(tmp_path) -> None:
    selected_site = tmp_path / "rdk-1.8.1/site-packages"
    _fake_rdk_distribution(selected_site, "1.8.1")
    wrong_module = _fake_rdk_distribution(
        tmp_path / "rdk-1.8.0/site-packages", "1.8.0"
    )
    sdk = SimpleNamespace(__file__=str(wrong_module), __version__="1.8.0")

    with pytest.raises(flexiv_rdk.FlexivRDKError, match="outside the selected"):
        loaded_flexiv_rdk_version(
            sdk,
            environment={RDK_SITE_ENV: str(selected_site)},
        )


def test_resolver_falls_back_until_robot_firmware_accepts_rdk(tmp_path) -> None:
    root = _rdk_root(tmp_path)
    calls: list[str] = []

    def runner(_command, **kwargs):
        version = kwargs["env"][RDK_VERSION_ENV]
        calls.append(version)
        ok = version == "1.9.0"
        return SimpleNamespace(
            returncode=0 if ok else 3,
            stdout=json.dumps(
                {
                    "ok": ok,
                    "loaded_version": version,
                    "error": None if ok else "robot firmware is incompatible",
                }
            ),
            stderr="",
        )

    resolver = FlexivRDKResolver(
        python=Path("/runtime/python"),
        environment={RDK_ROOT_ENV: str(root)},
        runner=runner,
    )

    first = resolver.select(_rizon_config()["devices"][0])
    second = resolver.select(_rizon_config()["devices"][0])

    assert first.candidate.version == "1.9.0"
    assert [attempt.version for attempt in first.attempts] == [
        "1.9.2",
        "1.9.1",
        "1.9.0",
    ]
    assert second.candidate.version == "1.9.0"
    assert second.cache_hit is True
    assert [attempt.version for attempt in second.attempts] == ["1.9.0"]
    assert calls == ["1.9.2", "1.9.1", "1.9.0", "1.9.0"]


def test_probe_timeout_continues_to_the_next_rdk(tmp_path) -> None:
    root = _rdk_root(tmp_path)

    def runner(command, **kwargs):
        version = kwargs["env"][RDK_VERSION_ENV]
        if version == "1.9.2":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "loaded_version": version}),
            stderr="",
        )

    resolver = FlexivRDKResolver(
        python=Path("/runtime/python"),
        environment={RDK_ROOT_ENV: str(root)},
        runner=runner,
    )

    selection = resolver.select(_rizon_config()["devices"][0])

    assert selection.candidate.version == "1.9.1"
    assert selection.attempts[0].timed_out is True


def test_probe_request_never_enables_or_moves_robot(monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []

    class Robot:
        def __init__(self, serial, whitelist, verbose):
            calls.append(("connect", (serial, whitelist, verbose)))

        def operational_status(self):
            calls.append(("status", None))
            return SimpleNamespace(name="IDLE")

    monkeypatch.setattr(flexiv_rdk.metadata, "version", lambda _name: "1.9.0")
    payload = flexiv_rdk._probe_request(
        {
            "expected_version": "1.9.0",
            "robot_serials": ["Rizon4-012345"],
            "network_interface_whitelist": ["192.168.2.10"],
        },
        sdk=SimpleNamespace(Robot=Robot),
    )

    assert payload["ok"] is True
    assert calls == [
        ("connect", ("Rizon4-012345", ["192.168.2.10"], False)),
        ("status", None),
    ]


class _FakeProcess:
    pid = 1234
    exitcode = None

    def __init__(self):
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=0.0):
        del timeout
        return None

    def terminate(self):
        self.alive = False


class _FakeResolver:
    def __init__(self, selection=None, error=None):
        self.selection = selection
        self.error = error
        self.calls = 0

    def select(self, _config):
        self.calls += 1
        if self.error:
            raise self.error
        return self.selection


def test_lifecycle_injects_selected_rdk_only_into_rizon_worker(monkeypatch, tmp_path) -> None:
    site = tmp_path / "rdk-1.9.0/site-packages"
    site.mkdir(parents=True)
    attempt = FlexivRDKProbeAttempt("1.9.0", True, None, 0.1)
    selection = FlexivRDKSelection(
        FlexivRDKCandidate("1.9.0", site, "packaged-release"), (attempt,)
    )
    resolver = _FakeResolver(selection=selection)
    manager = DeviceLifecycleManager(
        _rizon_config(), flexiv_rdk_resolver=resolver
    )
    monkeypatch.setattr(
        manager,
        "_probe_shared_memory",
        lambda info: {
            "name": "missing",
            "connected": False,
            "fresh": False,
            "last_timestamp_ns": None,
            "age_ms": None,
            "fps": info["config"].get("fps"),
            "shape": info["config"].get("data_shape"),
            "dtype": info["config"].get("data_dtype"),
            "frames": 0,
            "error": None,
        },
    )
    monkeypatch.setattr(
        manager, "_start_generic_device_process", lambda _info: _FakeProcess()
    )

    status = manager.start_device_observed("RizonRobot", 2)

    config = manager.devices[0]["config"]
    assert resolver.calls == 1
    assert config[WORKER_RDK_VERSION_KEY] == "1.9.0"
    assert config[WORKER_RDK_SITE_KEY] == str(site)
    assert status["flexiv_rdk"]["version"] == "1.9.0"
    assert status["flexiv_rdk"]["probe"]["status"] == "selected"
    manager.stop_all()


def test_lifecycle_reports_starting_while_rdk_probe_blocks(
    monkeypatch, tmp_path
) -> None:
    site = tmp_path / "rdk-1.9.0/site-packages"
    site.mkdir(parents=True)
    selection = FlexivRDKSelection(
        FlexivRDKCandidate("1.9.0", site, "packaged-release"),
        (FlexivRDKProbeAttempt("1.9.0", True, None, 0.1),),
    )
    probe_started = threading.Event()
    release_probe = threading.Event()

    class BlockingResolver:
        def select(self, _config):
            probe_started.set()
            assert release_probe.wait(timeout=2.0)
            return selection

    manager = DeviceLifecycleManager(
        _rizon_config(), flexiv_rdk_resolver=BlockingResolver()
    )
    monkeypatch.setattr(
        manager,
        "_probe_shared_memory",
        lambda info: {
            "name": "missing",
            "connected": False,
            "fresh": False,
            "last_timestamp_ns": None,
            "age_ms": None,
            "fps": info["config"].get("fps"),
            "shape": info["config"].get("data_shape"),
            "dtype": info["config"].get("data_dtype"),
            "frames": 0,
            "error": None,
        },
    )
    monkeypatch.setattr(
        manager, "_start_generic_device_process", lambda _info: _FakeProcess()
    )
    result: dict[str, Any] = {}

    def start() -> None:
        try:
            result["status"] = manager.start_device_observed("RizonRobot", 2)
        except BaseException as exc:  # pragma: no cover - asserted below
            result["error"] = exc

    thread = threading.Thread(target=start)
    thread.start()
    try:
        assert probe_started.wait(timeout=1.0)
        probing = manager.get_device_status("RizonRobot", 2)
        assert probing["status"] == "starting"
        assert probing["desired_state"] == "running"
        assert probing["pid"] is None
        assert probing["progress"]["phase"] == "rizon_selecting_rdk"
    finally:
        release_probe.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "error" not in result
    assert result["status"]["status"] == "starting"
    manager.stop_all()


def test_lifecycle_probe_failure_does_not_spawn_or_retry_robot(monkeypatch) -> None:
    attempt = FlexivRDKProbeAttempt(
        "1.9.2", False, "firmware mismatch", 0.1
    )
    resolver = _FakeResolver(
        error=FlexivRDKProbeError("no compatible RDK", [attempt])
    )
    manager = DeviceLifecycleManager(
        _rizon_config(), flexiv_rdk_resolver=resolver
    )
    manager.devices[0]["flexiv_rdk_version"] = "1.9.0"
    manager.devices[0]["flexiv_rdk_site"] = "/stale/rdk"
    manager.devices[0]["config"][WORKER_RDK_VERSION_KEY] = "1.9.0"
    manager.devices[0]["config"][WORKER_RDK_SITE_KEY] = "/stale/rdk"
    monkeypatch.setattr(
        manager,
        "_probe_shared_memory",
        lambda info: {
            "name": "missing",
            "connected": False,
            "fresh": False,
            "last_timestamp_ns": None,
            "age_ms": None,
            "fps": info["config"].get("fps"),
            "shape": info["config"].get("data_shape"),
            "dtype": info["config"].get("data_dtype"),
            "frames": 0,
            "error": None,
        },
    )
    spawned = []
    monkeypatch.setattr(
        manager,
        "_start_device_process",
        lambda *_args, **_kwargs: spawned.append(True),
    )

    with pytest.raises(DeviceStartupError, match="no compatible RDK"):
        manager.start_device_observed("RizonRobot", 2)

    status = manager.get_device_status("RizonRobot", 2)
    assert spawned == []
    assert status["desired_state"] == "stopped"
    assert status["failure_code"] == "flexiv_rdk_probe_failed"
    assert status["flexiv_rdk"]["version"] is None
    assert status["flexiv_rdk"]["probe"]["status"] == "failed"
    assert WORKER_RDK_VERSION_KEY not in manager.devices[0]["config"]
    assert WORKER_RDK_SITE_KEY not in manager.devices[0]["config"]
