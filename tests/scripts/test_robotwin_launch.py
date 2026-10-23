"""Focused behavior tests for the RoboTwin process launcher."""

from __future__ import annotations

import importlib.util
import sys
import venv
from pathlib import Path
from typing import Any

import pytest

from utils.runtime_profiles import REGISTRY_ENV, RuntimeRegistry


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "robotwin" / "launch.py"


@pytest.fixture(autouse=True)
def _isolated_runtime_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "missing-runtime-profiles.json"))


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("robotwin_launch_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_adds_the_repository_root_to_script_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "path",
        [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() != REPOSITORY_ROOT
        ],
    )

    module = _load_module()

    assert module.sys.path[0] == str(REPOSITORY_ROOT)


def test_launcher_reexecs_with_python_no_user_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.setattr(module.sys, "argv", [str(SCRIPT_PATH), "--help"])

    def execvpe(executable: str, argv: list[str], env: dict[str, str]) -> None:
        calls.append((executable, argv, env))
        raise SystemExit(0)

    monkeypatch.setattr(module.os, "execvpe", execvpe)

    with pytest.raises(SystemExit):
        module._ensure_no_user_site_packages()

    assert len(calls) == 1
    assert calls[0][0] == module.sys.executable
    assert calls[0][1] == [module.sys.executable, str(SCRIPT_PATH), "--help"]
    assert calls[0][2]["PYTHONNOUSERSITE"] == "1"


def test_launcher_fails_when_device_starter_exits_and_stops_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_ensure_no_user_site_packages", lambda: None)

    class _Process:
        def __init__(self, exit_code: int | None) -> None:
            self.returncode = exit_code
            self.signals: list[int] = []

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, signal: int) -> None:
            self.signals.append(signal)
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

        def terminate(self) -> None:
            self.returncode = 0

    device = _Process(exit_code=1)
    connector = _Process(exit_code=None)
    started: list[list[str]] = []

    def popen(command: list[str], **_kwargs: object) -> _Process:
        started.append(command)
        return device if len(started) == 1 else connector

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "_wait_for_summary_shm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--skip-preflight",
            "--summary-shm",
            "custom_robotwin_summary",
        ],
    )

    with pytest.raises(RuntimeError, match="DeviceStarter exited unexpectedly"):
        module.main()

    assert "summary_shm=custom_robotwin_summary" in started[0]
    assert started[1][-2:] == ["--summary-shm", "custom_robotwin_summary"]
    assert connector.signals == [module.signal.SIGINT]


def test_launcher_cleanup_ignores_a_second_keyboard_interrupt() -> None:
    module = _load_module()

    class _InterruptingProcess:
        def poll(self) -> int | None:
            raise KeyboardInterrupt

    module._terminate(_InterruptingProcess(), "RoboTwin policy connector")


def test_launcher_returns_preflight_failure_without_starting_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_ensure_no_user_site_packages", lambda: None)
    monkeypatch.setattr(
        module,
        "_resolve_preflight_settings",
        lambda _config_name: {
            "robotwin_root": "third_party/RoboTwin",
            "task": "beat_block_hammer",
            "task_config": "demo_clean",
            "act_source": None,
            "act_ckpt": None,
        },
    )
    monkeypatch.setattr(
        module,
        "_preflight_command",
        lambda _settings, **_kwargs: ["preflight"],
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 1})(),
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "children must not start after preflight failure"
        ),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])

    assert module.main() == 1


def test_preflight_command_uses_the_composed_task_and_act_artifacts() -> None:
    module = _load_module()

    command = module._preflight_command(
        {
            "robotwin_root": "custom/RoboTwin",
            "task": "stack_blocks",
            "task_config": "demo_randomized",
            "act_source": "custom/act/policy",
            "act_ckpt": "custom/act/checkpoint",
        }
    )

    assert command[2:] == [
        "--robotwin-root",
        "custom/RoboTwin",
        "--task",
        "stack_blocks",
        "--task-config",
        "demo_randomized",
        "--act-source",
        "custom/act/policy",
        "--act-ckpt",
        "custom/act/checkpoint",
    ]


def test_preflight_command_adds_xpolicylab_policy_checks() -> None:
    module = _load_module()

    command = module._preflight_command(
        {
            "robotwin_root": "custom/RoboTwin",
            "task": "beat_block_hammer",
            "task_config": "demo_clean",
            "act_source": None,
            "act_ckpt": None,
            "xpolicylab_root": "custom/RoboTwin/XPolicyLab",
            "xpolicylab_policy": "Pi_05",
        }
    )

    assert command[-4:] == [
        "--xpolicylab-root",
        "custom/RoboTwin/XPolicyLab",
        "--xpolicylab-policy",
        "Pi_05",
    ]


def test_xpolicylab_server_settings_can_be_forced_with_ckpt_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    omega_conf = pytest.importorskip(
        "omegaconf",
        reason="RoboTwin config override tests require the optional OmegaConf dependency",
    ).OmegaConf
    monkeypatch.setattr(
        module,
        "_compose_config",
        lambda _config_name: omega_conf.create(
            {
                "robotwin": {
                    "xpolicylab_root": "third_party/RoboTwin/XPolicyLab",
                    "task_name": "beat_block_hammer",
                    "seed": 7,
                    "xpolicylab_server": {
                        "enabled": False,
                        "policy_name": "Pi_05",
                        "host": "127.0.0.1",
                        "port": 6000,
                        "bench_name": "RoboTwin",
                        "ckpt_name": "lerobot_v30",
                        "env_cfg_type": "aloha_agilex",
                        "action_type": "joint",
                        "runtime_env": "uv",
                    },
                }
            }
        ),
    )

    settings = module._resolve_xpolicylab_server_settings(
        "policy_infer/robotwin/config_infer_robotwin_pi05_joint14",
        force_enabled=True,
        ckpt_name_override="custom_ckpt",
    )

    assert settings["enabled"] is True
    assert settings["ckpt_name"] == "custom_ckpt"
    assert settings["task_name"] == "beat_block_hammer"
    assert settings["seed"] == 7


def test_runtime_selection_uses_verified_config_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    registry_path = tmp_path / "runtime_profiles.json"
    registry = RuntimeRegistry.load(repo_root=REPOSITORY_ROOT, path=registry_path)
    registry.register("robotwin", python=Path(sys.executable), imports=["json"])
    registry.verify("robotwin")
    registry.bind("device", "robotwin", config_name="policy_infer/robotwin/config_infer_robotwin_pi05_joint14")
    registry.save()
    monkeypatch.setenv(REGISTRY_ENV, str(registry_path))

    python, profile = module._runtime_selection(
        "device",
        "policy_infer/robotwin/config_infer_robotwin_pi05_joint14",
    )

    assert python == Path(sys.executable)
    assert profile is not None and profile.name == "robotwin"


def test_child_environment_preserves_workflow_owned_run_identity(tmp_path: Path) -> None:
    module = _load_module()
    registry = RuntimeRegistry.load(repo_root=tmp_path, path=tmp_path / "profiles.json")
    registry.register(
        "robotwin",
        python=Path(sys.executable),
        environment={
            "ROBOTWIN_PROFILE_TEST": "1",
            "SI_TIMESTAMP_RUN_ID": "profile-must-not-win",
        },
    )
    profile = registry.profile("robotwin")

    environment = module._child_environment(
        profile,
        config_name="policy_infer/robotwin/config_infer_robotwin_replay_joint14",
        run_id="owned-run-id",
    )

    assert environment["ROBOTWIN_PROFILE_TEST"] == "1"
    assert environment["SI_TIMESTAMP_RUN_ID"] == "owned-run-id"
    assert environment["SI_ACTION_STATE_XYZ_RUN_ID"] == "owned-run-id"
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_pi05_policy_server_profile_supplies_uv_project_path(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "openpi"
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    registry = RuntimeRegistry.load(repo_root=tmp_path, path=tmp_path / "profiles.json")
    registry.register("pi05", python=python, cwd=project, imports=["json"])
    registry.verify("pi05")
    profile = registry.profile("pi05")
    settings = {"policy_name": "Pi_05", "runtime_env": "uv"}

    module._apply_xpolicylab_server_profile(settings, profile)

    assert settings["runtime_env"] == str(project)


def test_fastwam_policy_server_profile_supplies_verified_conda_prefix(
    tmp_path: Path,
) -> None:
    module = _load_module()
    prefix = tmp_path / "fastwam"
    # Relocatable uv CPython needs its base installation, including encodings.
    venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
    python = prefix / "bin/python"
    registry = RuntimeRegistry.load(repo_root=tmp_path, path=tmp_path / "profiles.json")
    registry.register("fastwam", python=python, cwd=tmp_path, imports=["json"])
    registry.verify("fastwam")
    profile = registry.profile("fastwam")
    settings = {"policy_name": "FastWAM", "runtime_env": "fastwam"}

    module._apply_xpolicylab_server_profile(settings, profile)

    assert settings["runtime_env"] == str(prefix)


def test_xpolicylab_server_runtime_requires_profile_or_explicit_override() -> None:
    module = _load_module()

    with pytest.raises(module.RuntimeProfileError, match="not calibrated"):
        module._validate_xpolicylab_server_runtime(
            {"policy_name": "Pi_05", "runtime_env": "not_configured"}
        )


def test_launcher_waits_for_xpolicylab_server_before_starting_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_ensure_no_user_site_packages", lambda: None)
    monkeypatch.setattr(
        module,
        "_resolve_xpolicylab_server_settings",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "xpolicylab_root": "third_party/RoboTwin/XPolicyLab",
            "policy_name": "Pi_05",
            "bench_name": "RoboTwin",
            "task_name": "beat_block_hammer",
            "ckpt_name": "lerobot_v30",
            "env_cfg_type": "aloha_agilex",
            "action_type": "joint",
            "seed": 0,
            "policy_gpu_id": 0,
            "runtime_env": "uv",
            "port": 6000,
            "host": "127.0.0.1",
        },
    )
    monkeypatch.setattr(module, "_xpolicylab_server_command", lambda _settings: ["server"])
    monkeypatch.setattr(module, "_wait_for_summary_shm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_preflight_command",
        lambda _settings, **_kwargs: ["preflight"],
    )
    monkeypatch.setattr(
        module,
        "_resolve_preflight_settings",
        lambda _config_name: {
            "robotwin_root": "third_party/RoboTwin",
            "task": "beat_block_hammer",
            "task_config": "demo_clean",
            "act_source": None,
            "act_ckpt": None,
        },
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})(),
    )

    events: list[str] = []

    class _Process:
        def __init__(self, name: str, exit_code: int | None = None) -> None:
            self.name = name
            self.returncode = exit_code

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, _signal: int) -> None:
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return int(self.returncode or 0)

        def terminate(self) -> None:
            self.returncode = 0

    def popen(command: list[str], **_kwargs: object) -> _Process:
        if command == ["server"]:
            events.append("popen:server")
            return _Process("server")
        if command[1:3] == ["device_starter.py", "--config-name"]:
            events.append("popen:device_starter")
            return _Process("device", exit_code=1)
        events.append("popen:connector")
        return _Process("connector")

    def wait_for_tcp_port(*_args: object, **_kwargs: object) -> None:
        events.append("server-ready")

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "_wait_for_tcp_port", wait_for_tcp_port)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--start-xpolicylab-server"])

    with pytest.raises(RuntimeError, match="DeviceStarter exited unexpectedly"):
        module.main()

    assert events[:2] == ["popen:server", "server-ready"]
    assert events[2] == "popen:device_starter"


def test_xpolicylab_server_command_uses_policy_launcher(tmp_path: Path) -> None:
    module = _load_module()
    launcher = tmp_path / "XPolicyLab" / "policy" / "FastWAM" / "setup_eval_policy_server.sh"
    launcher.parent.mkdir(parents=True)
    launcher.touch()

    command = module._xpolicylab_server_command(
        {
            "xpolicylab_root": str(tmp_path / "XPolicyLab"),
            "policy_name": "FastWAM",
            "bench_name": "RoboTwin",
            "task_name": "beat_block_hammer",
            "ckpt_name": "demo_clean-50",
            "env_cfg_type": "aloha_agilex",
            "action_type": "joint",
            "seed": 3,
            "policy_gpu_id": 0,
            "runtime_env": "fastwam",
            "port": 6010,
            "host": "127.0.0.1",
        }
    )

    assert command == [
        "bash",
        str(launcher),
        "RoboTwin",
        "beat_block_hammer",
        "demo_clean-50",
        "aloha_agilex",
        "joint",
        "3",
        "0",
        "fastwam",
        "6010",
        "127.0.0.1",
    ]
