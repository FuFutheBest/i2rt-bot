from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = ROOT / ".claude/skills/workstation-inference-bootstrap/scripts"


def _load_preflight() -> Any:
    path = SKILL_SCRIPTS / "inference_preflight.py"
    spec = importlib.util.spec_from_file_location(
        "workstation_inference_preflight", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewed_rdk_selection_is_shared_with_bootstrap() -> None:
    preflight = _load_preflight()
    compatibility = preflight._load_compatibility()

    assert preflight.select_flexiv_rdk("v3.10.3", None, compatibility) == (
        "1.8.0",
        None,
    )
    selected, error = preflight.select_flexiv_rdk("v3.10", "1.9.2", compatibility)

    assert selected is None
    assert "incompatible" in error


def test_bootstrap_defaults_to_a_read_only_command_plan(tmp_path: Path) -> None:
    script = SKILL_SCRIPTS / "bootstrap_openpi_source.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--env-dir",
            str(tmp_path / "openpi-main312"),
            "--nedf-wheelhouse",
            str(tmp_path / "private-wheels"),
            "--robot-software-version",
            "v3.10.3",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Selected reviewed Flexiv RDK: 1.8.0" in result.stdout
    assert (
        "git submodule update --init third_party/openpi third_party/i2rt"
        in result.stdout
    )
    assert "create_source_py312_env.sh" in result.stdout
    assert "flexiv_rdk_1_8_0.txt" in result.stdout
    assert "--device-python" in result.stdout
    assert "runtime_profiles.py" in result.stdout
    assert "register openpi-main312" in result.stdout
    assert "--role policy --profile openpi-main312" in result.stdout
    assert "--checkpoint" not in result.stdout
    assert "Plan only" in result.stdout
    assert not (tmp_path / "openpi-main312").exists()


def test_preflight_accepts_config_owned_checkpoint_paths(tmp_path: Path) -> None:
    preflight = _load_preflight()
    config = tmp_path / "config.yaml"
    config.write_text(
        "policy:\n  params:\n    ckpt_dir: /home/example/checkpoint\n",
        encoding="utf-8",
    )
    report = preflight.Report()

    preflight._check_config(report, ROOT, str(config))

    assert not any(
        record["subject"] == "machine-local paths" for record in report.records
    )
    assert any(record["subject"] == "checkpoint" for record in report.records)


def test_execute_fails_before_install_when_private_wheels_are_missing(
    tmp_path: Path,
) -> None:
    script = SKILL_SCRIPTS / "bootstrap_openpi_source.py"
    wheelhouse = tmp_path / "private-wheels"
    wheelhouse.mkdir()
    env_dir = tmp_path / "openpi-main312"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--env-dir",
            str(env_dir),
            "--nedf-wheelhouse",
            str(wheelhouse),
            "--robot-software-version",
            "v3.10.3",
            "--execute",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "expected exactly one regular wheel matching" in result.stderr
    assert not env_dir.exists()
