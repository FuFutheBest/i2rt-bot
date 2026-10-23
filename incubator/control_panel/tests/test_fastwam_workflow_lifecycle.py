"""Hardware-free lifecycle contracts for the FastWAM control-panel adapter."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import socket
import sys
import threading
from typing import Any, Dict, List

import pytest

from incubator.control_panel.fastwam_workflow import FastWAMWorkflowLifecycle
from incubator.control_panel.runtime_config import RuntimeConfigError, runtime_config_metadata
from incubator.control_panel.server import DeviceController
from policy.fastwam_policy import _format_websocket_uri
from utils.runtime_profiles import RuntimeRegistry


class _FakePopen:
    _next_pid = 81_000

    def __init__(self, command: List[str], environment: Dict[str, str]) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.command = list(command)
        self.environment = dict(environment)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, value: int) -> None:
        self.returncode = -int(value)

    def wait(self, timeout: float = 0.0) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode


def _config(checkpoint: Path, port: int = 19_890) -> Dict[str, Any]:
    return {
        "devices": [],
        "device_manager": {"type": "base"},
        "action_executor": {"type": "base"},
        "policy": {
            "class": "FastWAMPolicy",
            "params": {
                "ckpt_path": str(checkpoint),
                "host": "127.0.0.1",
                "port": port,
                "port_range": [port, port + 2],
            },
            "obs": {"devices": []},
            "controls": [],
            "infer_rollout": {"enabled": False, "required": False},
        },
    }


def _workflow(
    tmp_path: Path,
    *,
    register_server_profile: bool = False,
) -> tuple[FastWAMWorkflowLifecycle, List[_FakePopen]]:
    repo_root = tmp_path / "repo"
    (repo_root / "consumer").mkdir(parents=True)
    (repo_root / "configs").mkdir()
    (repo_root / "consumer" / "fastwam_server.py").touch()
    (repo_root / "consumer" / "fastwam_policy_infer.py").touch()
    (repo_root / "configs" / "fixture.yaml").touch()
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "checkpoints" / "weights").mkdir(parents=True)
    (checkpoint / "config.yaml").write_text("framework: {}\n", encoding="utf-8")
    (checkpoint / "dataset_statistics.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "checkpoints" / "weights" / "step_000001.pt").write_bytes(b"fixture")
    python_target = tmp_path / "python3.10"
    python_target.touch()
    python_path = tmp_path / "venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(python_target)
    if register_server_profile:
        registry = RuntimeRegistry.load(repo_root=repo_root)
        registry.register(
            "fastwam-server",
            python=Path(sys.executable),
            environment={
                "FASTWAM_PROFILE_TEST": "1",
                "SUPERINFERENCE_FASTWAM_SERVER_PORT": "9999",
            },
            imports=["json"],
        )
        registry.verify("fastwam-server")
        registry.bind("fastwam-server", "fastwam-server", config_name="fixture")
        registry.save()
    processes: List[_FakePopen] = []

    def popen(command: List[str], **kwargs: Any) -> _FakePopen:
        process = _FakePopen(command, kwargs.get("env", {}))
        processes.append(process)
        return process

    workflow = FastWAMWorkflowLifecycle(
        "fixture",
        _config(checkpoint),
        repo_root=repo_root,
        device_python=Path(sys.executable),
        inference_python=python_path,
        recording_python=python_path,
        server_run_root=tmp_path / "server-runs",
        inference_control_run_root=tmp_path / "inference-runs",
        server_health_probe=lambda host, port, expected: True,
        recording_profile_checker=lambda path: (True, ""),
        popen_factory=popen,
        shm_identity=lambda name: None,
    )
    # Keep the test independent of whatever listeners happen to exist on the
    # host running the suite; port allocation itself is covered by the class
    # contract and the fake probe here models an available endpoint.
    workflow._port_is_available = lambda host, port: True  # type: ignore[method-assign]
    return workflow, processes


def _mark_server_ready(workflow: FastWAMWorkflowLifecycle) -> None:
    process = workflow.server_process
    ready_file = workflow._server_ready_file
    assert process is not None and ready_file is not None
    _host, port = workflow._server_endpoint()
    ready_file.write_text(
        json.dumps(
            {
                "run_id": workflow._server_run_id,
                "pid": process.pid,
                "config_name": workflow.config_name,
                "port": port,
                "model_loaded": True,
            }
        ),
        encoding="utf-8",
    )


def test_fastwam_metadata_and_default_factory_selection(tmp_path: Path) -> None:
    metadata = runtime_config_metadata(
        "fixture",
        {"devices": [], "policy": {"class": "FastWAMPolicy"}},
    )
    assert metadata["workflow_kind"] == "fastwam"
    assert metadata["control_panel_supported"] is True

    class _Jobs:
        def enqueue(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def cancel_episode(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class _Logs:
        def append(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    controller = object.__new__(DeviceController)
    controller._workflow_factory = None
    controller._inference_test_mode = False
    controller._rollout_enabled = False
    controller._rollout_output_root = None
    controller.recording_python = None
    controller.conversion_jobs = _Jobs()
    controller._recording_finalized_enqueue = None
    controller.logs = _Logs()
    workflow = controller._make_workflow("fixture", _config(tmp_path / "ckpt"))
    assert isinstance(workflow, FastWAMWorkflowLifecycle)

    mismatched = _config(tmp_path / "other-ckpt")
    mismatched["policy"]["class"] = "OpenPiUmiPolicy"
    mismatched["metadata"] = {"workflow_kind": "fastwam"}
    assert not isinstance(
        controller._make_workflow("mismatched", mismatched),
        FastWAMWorkflowLifecycle,
    )


def test_fastwam_client_formats_ipv6_websocket_endpoint() -> None:
    assert _format_websocket_uri("::1", 18090) == "ws://[::1]:18090"
    assert _format_websocket_uri("[::1]", 18090) == "ws://[::1]:18090"
    assert _format_websocket_uri("127.0.0.1", 18090) == "ws://127.0.0.1:18090"


def test_fastwam_server_port_probe_accepts_bracketed_ipv6(tmp_path: Path) -> None:
    workflow, _processes = _workflow(tmp_path)
    probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        probe.bind(("::1", 0))
        port = int(probe.getsockname()[1])
    finally:
        probe.close()

    assert workflow._port_is_available("[::1]", port)


def test_fastwam_workflow_uses_raw_ipv6_literal_for_process_commands(
    tmp_path: Path,
) -> None:
    workflow, processes = _workflow(tmp_path)
    workflow.config["policy"]["params"]["host"] = "[::1]"

    workflow.start_server([])
    server_command = processes[0].command
    assert server_command[server_command.index("--host") + 1] == "::1"

    workflow._server_port = 19_891
    client_command = workflow._inference_command()
    assert client_command[client_command.index("--server-host") + 1] == "::1"


def test_mock_inference_mode_does_not_start_fastwam_server(tmp_path: Path) -> None:
    workflow, processes = _workflow(tmp_path)
    workflow._test_inference_mode = True

    blockers = workflow._server_blockers()

    assert any("mock inference mode" in blocker for blocker in blockers)
    assert workflow.state([])["server"]["can_start"] is False
    assert processes == []


def test_release_fastwam_server_requires_explicit_nmx_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, _processes = _workflow(tmp_path)
    release_root = tmp_path / "release"
    bundled_python = release_root / "bin" / "python-policy"
    bundled_python.parent.mkdir(parents=True)
    bundled_python.touch()
    workflow.server_python = bundled_python.resolve()
    monkeypatch.setenv("SUPERINFERENCE_RELEASE_ROOT", str(release_root))
    monkeypatch.delenv("SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL", raising=False)

    blockers = workflow._server_blockers()

    assert any("separately provisioned NMX model environment" in blocker for blocker in blockers)


def test_server_starts_without_runtime_and_publishes_owned_state(
    tmp_path: Path,
) -> None:
    workflow, processes = _workflow(tmp_path)

    state = workflow.start_server([])

    assert state["kind"] == "fastwam"
    assert state["runtime"]["ready"] is False
    assert state["server"]["status"] == "starting"
    assert state["server"]["progress"]["phase"] == "spawning_process"
    assert workflow.server_python == tmp_path / "venv" / "bin" / "python"
    assert processes[0].command[0] == str(workflow.server_python)
    assert processes[0].command[1].endswith("fastwam_server.py")
    assert "--ckpt-path" in processes[0].command
    assert workflow._server_run_dir is not None
    assert workflow._server_run_dir.stat().st_mode & 0o777 == 0o700

    _mark_server_ready(workflow)
    ready = workflow.state([])
    assert ready["server"]["ready"] is True
    assert ready["server"]["endpoint"] == "127.0.0.1:19890"


def test_server_uses_config_specific_runtime_profile(tmp_path: Path) -> None:
    workflow, processes = _workflow(tmp_path, register_server_profile=True)

    workflow.start_server([])

    assert workflow.server_python == Path(sys.executable)
    assert processes[0].environment["SUPERINFERENCE_RUNTIME_PROFILE"] == "fastwam-server"
    assert processes[0].environment["FASTWAM_PROFILE_TEST"] == "1"
    assert processes[0].environment["SUPERINFERENCE_FASTWAM_SERVER_PORT"] == "19890"


def test_fastwam_client_command_uses_server_endpoint_without_checkpoint_override(
    tmp_path: Path,
) -> None:
    workflow, _processes = _workflow(tmp_path)
    workflow._server_port = 19_891

    command = workflow._inference_command()
    environment: Dict[str, str] = {}
    workflow._apply_inference_checkpoint_override(command, environment)

    assert command[1].endswith("fastwam_policy_infer.py")
    assert command[command.index("--server-host") + 1] == "127.0.0.1"
    assert command[command.index("--server-port") + 1] == "19891"
    assert "SUPERINFERENCE_CHECKPOINT_OVERRIDE" not in environment
    assert workflow.checkpoint_state()["owner_stage"] == "server"
    assert workflow.checkpoint_state()["params_required"] is False


def test_server_preflight_rejects_incomplete_checkpoint_artifacts(
    tmp_path: Path,
) -> None:
    workflow, processes = _workflow(tmp_path)
    (tmp_path / "checkpoint" / "dataset_statistics.json").unlink()

    blockers = workflow._server_blockers()

    assert any("dataset_statistics.json" in blocker for blocker in blockers)
    assert workflow.state([])["server"]["can_start"] is False
    assert processes == []


def test_fastwam_checkpoint_browser_and_override_accept_weight_file(
    tmp_path: Path,
) -> None:
    workflow, _processes = _workflow(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    weight = checkpoint / "checkpoints" / "weights" / "step_000001.pt"

    controller = DeviceController.__new__(DeviceController)
    controller._lock = threading.RLock()
    controller._checkpoint_browse_roots = (tmp_path.resolve(),)
    controller.workflow = workflow
    controller.selected_config = "fixture"
    controller.revision = 0
    controller._checkpoint_generation = 0
    controller._checkpoint_override = None
    controller._effective_config = workflow.config
    controller.state = lambda: {"checkpoint": workflow.checkpoint_state()}
    controller.logs = type(
        "_Logs",
        (),
        {"append": lambda self, *args, **kwargs: None},
    )()

    listing = controller.browse_checkpoints(str(weight))
    entry = next(item for item in listing["entries"] if item["path"] == str(weight))
    assert listing["accepts_files"] is True
    assert entry["kind"] == "file"
    assert entry["selectable"] is True

    root_listing = controller.browse_checkpoints(str(checkpoint))
    config_entry = next(
        item
        for item in root_listing["entries"]
        if item["path"] == str(checkpoint / "config.yaml")
    )
    assert config_entry["selectable"] is False

    selected = controller.set_checkpoint_override(
        str(weight),
        expected_selected_config="fixture",
        expected_revision=0,
        expected_generation=0,
    )
    assert selected["checkpoint"]["effective_path"] == str(weight)
    assert selected["checkpoint"]["override_path"] == str(weight)


def test_fastwam_checkpoint_browser_rejects_non_weight_files(tmp_path: Path) -> None:
    workflow, _processes = _workflow(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    non_weight_file = checkpoint / "config.yaml"

    controller = DeviceController.__new__(DeviceController)
    controller._lock = threading.RLock()
    controller._checkpoint_browse_roots = (tmp_path.resolve(),)
    controller.workflow = workflow

    with pytest.raises(RuntimeConfigError, match=r"\.pt suffix"):
        controller._canonical_checkpoint_path(
            str(non_weight_file),
            require_directory=True,
            allow_file=True,
        )
