#!/usr/bin/env python3
"""Plan or execute the reviewed source-checkout OpenPI environment bootstrap."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import inference_preflight

DEFAULT_CONFIG = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
DEFAULT_RUNTIME_PROFILE = "openpi-main312"
REQUIRED_NEDF_WHEELS = (
    "dm3_core-0.2.18-*.whl",
    "nmx_msg-2.2.0-*.whl",
)


def _absolute(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    return expanded


def _command_text(
    command: Sequence[str], environment: dict[str, str] | None = None
) -> str:
    prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in (environment or {}).items()
    )
    rendered = shlex.join([str(item) for item in command])
    return f"{prefix} {rendered}".strip()


def _build_commands(
    args: argparse.Namespace, rdk_version: str
) -> list[tuple[list[str], dict[str, str]]]:
    repo_root = args.repo_root.expanduser().resolve()
    env_dir = _absolute(args.env_dir, "environment path")
    wheelhouse = _absolute(args.nedf_wheelhouse, "NEDF3 wheelhouse")
    env_python = env_dir / "bin/python"
    release_root = repo_root / "incubator/control_panel/release"
    rdk_requirements = (
        release_root
        / "requirements"
        / f"flexiv_rdk_{rdk_version.replace('.', '_')}.txt"
    )
    if not rdk_requirements.is_file():
        raise ValueError(f"reviewed RDK requirements are missing: {rdk_requirements}")

    commands = [
        (["uv", "python", "install", "3.12"], {}),
        (
            [
                "git",
                "submodule",
                "update",
                "--init",
                *inference_preflight.OPENPI_SUBMODULES,
            ],
            {},
        ),
        (
            [
                str(release_root / "create_source_py312_env.sh"),
                str(env_dir),
                str(wheelhouse),
            ],
            {"PYTHON_BIN": "3.12"},
        ),
        (
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(env_python),
                "--build-constraints",
                str(release_root / "requirements/build_constraints.txt"),
                "-r",
                str(rdk_requirements),
            ],
            {},
        ),
        (
            [
                str(env_python),
                str(Path(__file__).with_name("inference_preflight.py")),
                "--repo-root",
                str(repo_root),
                "--target",
                "flexiv",
                "--robot-software-version",
                args.robot_software_version,
                "--rdk-version",
                rdk_version,
                "--config",
                args.config,
                "--device-python",
                str(env_python),
            ],
            {},
        ),
    ]
    runtime_cli = repo_root / "scripts/diagnostics/runtime_profiles.py"
    register_command = [
        str(env_python),
        str(runtime_cli),
        "--repo-root",
        str(repo_root),
        "register",
        args.runtime_profile_name,
        "--python",
        str(env_python),
        "--cwd",
        str(repo_root),
        "--env",
        "PYTHONNOUSERSITE=1",
    ]
    for module in (
        "hydra",
        "omegaconf",
        "jax",
        "torch",
        "dm3_core.nedf",
        "nmx_msg",
    ):
        register_command.extend(("--import", module))
    commands.append((register_command, {}))
    for role in ("launcher", "planner", "device", "policy", "recording", "converter"):
        commands.append(
            (
                [
                    str(env_python),
                    str(runtime_cli),
                    "--repo-root",
                    str(repo_root),
                    "bind",
                    "--config-name",
                    args.config,
                    "--role",
                    role,
                    "--profile",
                    args.runtime_profile_name,
                ],
                {},
            )
        )
    return commands


def _validate_execution_inputs(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.expanduser().resolve()
    env_dir = _absolute(args.env_dir, "environment path")
    wheelhouse = _absolute(args.nedf_wheelhouse, "NEDF3 wheelhouse")
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        raise ValueError(
            f"source profile requires Linux x86_64; found {system} {machine}"
        )
    if not (repo_root / ".git").exists():
        raise ValueError(f"repository is not a Git checkout: {repo_root}")
    if env_dir.exists() or env_dir.is_symlink():
        raise ValueError(f"refusing to replace existing environment path: {env_dir}")
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise ValueError(f"NEDF3 wheelhouse must be a real directory: {wheelhouse}")
    for pattern in REQUIRED_NEDF_WHEELS:
        matches = [
            path
            for path in wheelhouse.glob(pattern)
            if path.is_file() and not path.is_symlink()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one regular wheel matching {pattern} in {wheelhouse}"
            )
    config_path = inference_preflight._config_path(repo_root, args.config)
    if not config_path.is_file():
        raise ValueError(f"OpenPI config is missing: {config_path}")
    if shutil.which("uv") is None:
        raise ValueError("uv is not on PATH")
    if os.environ.get("SUPERINFERENCE_RELEASE_ROOT"):
        raise ValueError(
            "unset SUPERINFERENCE_RELEASE_ROOT before creating a source environment"
        )
    gpu_ok, gpu_detail = inference_preflight._gpu_summary()
    if not gpu_ok:
        raise ValueError(f"NVIDIA driver preflight failed: {gpu_detail}")


def _run(command: Sequence[str], environment: dict[str, str], cwd: Path) -> None:
    subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env={**os.environ, **environment},
        check=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--nedf-wheelhouse", type=Path, required=True)
    parser.add_argument("--robot-software-version", required=True)
    parser.add_argument("--rdk-version")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--runtime-profile-name",
        default=DEFAULT_RUNTIME_PROFILE,
        help="Logical local runtime profile name written after successful validation.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the plan. Without this flag, only print commands.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compatibility = inference_preflight._load_compatibility()
    selected_rdk, error = inference_preflight.select_flexiv_rdk(
        args.robot_software_version,
        args.rdk_version,
        compatibility,
    )
    if error is not None or selected_rdk is None:
        raise ValueError(error or "no reviewed Flexiv RDK was selected")
    commands = _build_commands(args, selected_rdk)

    print(f"Selected reviewed Flexiv RDK: {selected_rdk}")
    print("Bootstrap plan:")
    for index, (command, environment) in enumerate(commands, start=1):
        print(f"{index}. {_command_text(command, environment)}")
    if not args.execute:
        print(
            "Plan only; rerun with --execute after reviewing paths and private inputs."
        )
        return 0

    _validate_execution_inputs(args)
    repo_root = args.repo_root.expanduser().resolve()
    for command, environment in commands:
        _run(command, environment, repo_root)

    print(
        "Source profile is ready. Launch command (hardware-facing; review bindings first):"
    )
    print(
        _command_text(
            [
                "python3",
                str(repo_root / "scripts/diagnostics/runtime_profiles.py"),
                "run",
                "--config-name",
                args.config,
                "--role",
                "launcher",
                "--",
                "-m",
                "incubator.control_panel",
                "--config-name",
                args.config,
            ]
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
