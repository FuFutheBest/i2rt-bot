from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.runtime_profiles import (
    REGISTRY_ENV,
    RuntimeProfileVerificationError,
    RuntimeRegistry,
    RuntimeRegistryError,
    select_runtime_python,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "runtime_profiles.py"


def _registry(tmp_path: Path) -> RuntimeRegistry:
    return RuntimeRegistry.load(
        repo_root=tmp_path,
        path=tmp_path / ".superinference/local/runtime_profiles.json",
    )


def test_registry_round_trip_preserves_repo_relative_interpreter_entry(
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "runtime/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    registry = _registry(tmp_path)

    registry.register(
        "core",
        python=interpreter,
        imports=["json"],
        environment={"PYTHONNOUSERSITE": "1"},
    )
    registry.verify("core")
    registry.bind("device", "core")
    registry.save()

    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    assert payload["profiles"]["core"]["python"] == {
        "base": "repo",
        "path": "runtime/bin/python",
    }
    assert registry.path.parent.stat().st_mode & 0o777 == 0o700
    assert registry.path.stat().st_mode & 0o777 == 0o600

    loaded = RuntimeRegistry.load(repo_root=tmp_path, path=registry.path)
    profile = loaded.resolve("device", required=True)
    assert profile is not None
    assert profile.python == interpreter
    assert profile.verified is True


def test_config_binding_overrides_default_role_binding(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.register("core", python=Path(sys.executable))
    registry.register("robotwin", python=Path(sys.executable))
    registry.bind("device", "core")
    registry.bind(
        "device", "robotwin", config_name="policy_infer/robotwin/config_infer_robotwin_pi05_joint14.yaml"
    )

    default = registry.resolve("device", required=True)
    configured = registry.resolve(
        "device",
        config_name="policy_infer/robotwin/config_infer_robotwin_pi05_joint14",
        required=True,
    )

    assert default is not None and default.name == "core"
    assert configured is not None and configured.name == "robotwin"


def test_selection_precedence_is_explicit_environment_profile_fallback(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.register("core", python=Path(sys.executable))
    registry.bind("policy", "core")
    registry.save()
    environment = {
        REGISTRY_ENV: str(registry.path),
        "POLICY_PYTHON": "/environment/python",
    }

    explicit = select_runtime_python(
        "policy",
        explicit="/explicit/python",
        environment_names=("POLICY_PYTHON",),
        fallback="/fallback/python",
        repo_root=tmp_path,
        environ=environment,
    )
    configured = select_runtime_python(
        "policy",
        environment_names=("POLICY_PYTHON",),
        fallback="/fallback/python",
        repo_root=tmp_path,
        environ=environment,
    )
    profiled = select_runtime_python(
        "policy",
        environment_names=("MISSING_PYTHON",),
        fallback="/fallback/python",
        repo_root=tmp_path,
        environ=environment,
    )

    assert explicit is not None and explicit.source == "explicit"
    assert configured is not None and configured.source == "environment:POLICY_PYTHON"
    assert profiled is not None and profiled.source == "profile:core"


def test_profile_verification_reports_the_failing_import(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.register(
        "broken",
        python=Path(sys.executable),
        imports=["module_that_does_not_exist_for_runtime_profile_test"],
    )

    with pytest.raises(
        RuntimeProfileVerificationError, match="module_that_does_not_exist"
    ):
        registry.verify("broken")


def test_process_environment_prepends_profile_bin_and_sets_identity(
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "venv/bin/python"
    registry = _registry(tmp_path)
    registry.register(
        "policy",
        python=interpreter,
        environment={"PYTHONNOUSERSITE": "1"},
    )
    profile = registry.profile("policy")

    environment = profile.process_environment(
        {"PATH": "/usr/bin"},
        config_name="policy_infer/openpi/demo.yaml",
    )

    assert environment["PATH"] == f"{interpreter.parent}{os.pathsep}/usr/bin"
    assert environment["SUPERINFERENCE_RUNTIME_PROFILE"] == "policy"
    assert environment["SUPERINFERENCE_CONFIG_NAME"] == "policy_infer/openpi/demo"
    assert environment[REGISTRY_ENV] == str(registry.path)

    repeated = profile.process_environment(
        environment, config_name="policy_infer/openpi/demo"
    )
    assert repeated["PATH"] == environment["PATH"]


def test_malformed_registry_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "runtime_profiles.json"
    path.write_text('{"schema_version": 99}', encoding="utf-8")

    with pytest.raises(
        RuntimeRegistryError, match="unsupported runtime registry schema"
    ):
        RuntimeRegistry.load(repo_root=tmp_path, path=path)


def test_registry_from_another_worktree_is_rejected(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    registry = _registry(first_root)
    registry.register("core", python=Path(sys.executable))
    registry.save()

    with pytest.raises(RuntimeRegistryError, match="different repository root"):
        RuntimeRegistry.load(repo_root=second_root, path=registry.path)


def test_cli_register_bind_resolve_and_run(tmp_path: Path) -> None:
    registry_path = tmp_path / ".superinference/local/runtime_profiles.json"
    common = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(tmp_path),
        "--registry",
        str(registry_path),
    ]
    registered = subprocess.run(
        [
            *common,
            "register",
            "core",
            "--python",
            sys.executable,
            "--import",
            "json",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert registered.returncode == 0, registered.stderr
    assert json.loads(registered.stdout)["verified"] is True

    bound = subprocess.run(
        [*common, "bind", "--role", "planner", "--profile", "core"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bound.returncode == 0, bound.stderr

    resolved = subprocess.run(
        [*common, "resolve", "--role", "planner", "--require-verified"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert Path(resolved.stdout.strip()) == Path(sys.executable)

    executed = subprocess.run(
        [
            *common,
            "run",
            "--role",
            "planner",
            "--",
            "-c",
            (
                "import os; print(os.environ['SUPERINFERENCE_RUNTIME_PROFILE']); "
                "print(os.environ['SUPERINFERENCE_RUNTIME_ROLE'])"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout.splitlines() == ["core", "planner"]

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["profiles"]["core"]["verification"]["fingerprint"] = "stale"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    doctor = subprocess.run(
        [*common, "doctor", "--role", "planner"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert doctor.returncode == 1
    assert doctor.stdout.startswith("FAIL core:")
    assert "environment fingerprint changed" in doctor.stdout


def test_doctor_requires_all_fastwam_runtime_roles(tmp_path: Path) -> None:
    config_name = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam"
    registry = _registry(tmp_path)
    registry.register("core", python=Path(sys.executable), imports=["json"])
    registry.verify("core")
    for role in (
        "launcher",
        "planner",
        "device",
        "policy",
        "fastwam-server",
    ):
        registry.bind(role, "core", config_name=config_name)
    registry.save()
    common = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(tmp_path),
        "--registry",
        str(registry.path),
    ]

    missing = subprocess.run(
        [*common, "doctor", "--config-name", config_name, "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 2
    assert "missing required runtime role bindings: recording" in missing.stderr

    registry.bind("recording", "core", config_name=config_name)
    registry.save()
    complete = subprocess.run(
        [*common, "doctor", "--config-name", config_name, "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert complete.returncode == 0, complete.stderr
    payload = json.loads(complete.stdout)
    assert payload["ok"] is True
    assert payload["required_roles"] == [
        "launcher",
        "planner",
        "device",
        "policy",
        "fastwam-server",
        "recording",
    ]
