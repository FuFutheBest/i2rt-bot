#!/usr/bin/env python3
"""Read-only readiness report for a fresh OpenPI workstation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY_PATH = SKILL_ROOT / "references" / "flexiv_rdk_compatibility.json"
OPENPI_SUBMODULES = (
    "third_party/openpi",
    "third_party/i2rt",
    "third_party/lktech_motor_sdk",
)
PROFILE_PROBE_MARKER = "__SUPERINFERENCE_SOURCE_PROFILE_JSON__="


def _load_compatibility() -> dict[str, Any]:
    with COMPATIBILITY_PATH.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid compatibility data: {COMPATIBILITY_PATH}")
    return payload


def _normalized_version(value: str) -> str:
    return value.strip().lower().removeprefix("v")


def _canonical_rdk_version(value: str, compatibility: dict[str, Any]) -> str:
    normalized = _normalized_version(value)
    aliases = compatibility["rdk_aliases"]
    return str(aliases.get(normalized, normalized))


def _robot_software_key(value: str) -> str:
    normalized = _normalized_version(value)
    if normalized.startswith("3e."):
        return "3E." + normalized.removeprefix("3e.")
    return normalized


def _resolved_robot_software_key(value: str, compatibility: dict[str, Any]) -> str:
    requested = _robot_software_key(value)
    matrix = compatibility["robot_software_to_rdk"]
    if requested in matrix:
        return requested
    candidates = [key for key in matrix if requested.startswith(f"{key}.")]
    return max(candidates, key=len) if candidates else requested


def _command_version(command: Sequence[str]) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, f"{command[0]} is not on PATH"
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip().splitlines()
    if result.returncode:
        return False, output[0] if output else f"exit {result.returncode}"
    return True, output[0] if output else executable


def _python_312_available(explicit: Path | None = None) -> tuple[bool, str]:
    executables = ([str(explicit.expanduser())] if explicit is not None else []) + [
        "python3.12",
        "python3",
    ]
    for executable in executables:
        path = shutil.which(executable)
        if explicit is not None and executable == str(explicit.expanduser()):
            path = executable if Path(executable).is_file() else None
        if path is None:
            continue
        result = subprocess.run(
            [
                path,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip()
        if result.returncode == 0 and version == "3.12":
            return True, f"{path} ({version})"
    return False, "python3.12 is required by the source OpenPI profile"


def _gpu_summary() -> tuple[bool, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, "nvidia-smi is not on PATH"
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        return False, detail or f"nvidia-smi exited {result.returncode}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return bool(lines), "; ".join(lines) if lines else "no NVIDIA GPU detected"


def _installed_distribution_version(
    python: Path, distribution: str
) -> tuple[bool, str]:
    executable = python.expanduser()
    if not executable.is_file():
        return False, f"Python executable is missing: {executable}"
    code = f"from importlib import metadata; print(metadata.version({distribution!r}))"
    try:
        result = subprocess.run(
            [str(executable), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        return False, detail or f"exit {result.returncode}"
    return True, result.stdout.strip()


def _config_path(repo_root: Path, config_name: str) -> Path:
    candidate = Path(config_name)
    if candidate.suffix in {".yaml", ".yml"}:
        return candidate if candidate.is_absolute() else repo_root / candidate
    return repo_root / "configs" / f"{config_name}.yaml"


class Report:
    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def add(self, status: str, subject: str, detail: str) -> None:
        self.records.append({"status": status, "subject": subject, "detail": detail})

    @property
    def has_blocker(self) -> bool:
        return any(record["status"] == "BLOCKER" for record in self.records)


def _check_repository(report: Report, repo_root: Path) -> None:
    if not (repo_root / ".git").exists():
        report.add("BLOCKER", "repository", f"not a Git checkout: {repo_root}")
        return
    report.add("PASS", "repository", str(repo_root))

    for relative in OPENPI_SUBMODULES:
        path = repo_root / relative
        if path.is_dir() and any(path.iterdir()):
            report.add("PASS", "submodule", relative)
        else:
            report.add(
                "BLOCKER",
                "submodule",
                f"{relative} is absent; initialize it before OpenPI installation",
            )


def _check_host(report: Report, python: Path | None = None) -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        report.add(
            "BLOCKER", "host", f"requires Linux x86_64; detected {system} {machine}"
        )
    else:
        report.add("PASS", "host", f"{system} {machine}")

    ok, detail = _command_version(("uv", "--version"))
    report.add("PASS" if ok else "BLOCKER", "uv", detail)
    ok, detail = _python_312_available(python)
    report.add("PASS" if ok else "BLOCKER", "python", detail)
    ok, detail = _gpu_summary()
    report.add("PASS" if ok else "BLOCKER", "NVIDIA GPU", detail)
    report.add("INFO", "profile", "OpenPI Python 3.12 source profile")


def _check_config(report: Report, repo_root: Path, config_name: str) -> None:
    config_path = _config_path(repo_root, config_name)
    if not config_path.is_file():
        report.add("BLOCKER", "config", f"missing: {config_path}")
    else:
        report.add("PASS", "config", str(config_path))
        contents = config_path.read_text(encoding="utf-8", errors="replace")
        hardcoded = [
            line.strip()
            for line in contents.splitlines()
            if "python_path:" in line
            if "/home/" in line
            if "${oc.env:" not in line
        ]
        if hardcoded:
            report.add(
                "ACTION",
                "machine-local paths",
                "set an environment-backed Python path or create a machine-local generated config",
            )
        report.add(
            "INFO",
            "checkpoint",
            "resolved from policy.params.ckpt_dir and validated by the installed profile probe",
        )


def _check_flexiv(
    report: Report,
    robot_software: str | None,
    explicit_rdk: str | None,
    compatibility: dict[str, Any],
) -> str | None:
    if robot_software is None:
        report.add(
            "BLOCKER",
            "Flexiv robot software",
            "read the robot software version from the robot/vendor tool; Elements UI version is not sufficient",
        )
        return None

    robot_key = _resolved_robot_software_key(robot_software, compatibility)
    selected, error = select_flexiv_rdk(robot_software, explicit_rdk, compatibility)
    if error is not None:
        report.add("BLOCKER", "Flexiv RDK", error)
        return None
    assert selected is not None
    report.add(
        "PASS",
        "Flexiv RDK",
        f"robot software {robot_key} -> RDK {selected} (project-reviewed)",
    )
    return selected


def select_flexiv_rdk(
    robot_software: str,
    explicit_rdk: str | None,
    compatibility: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Return the reviewed RDK selection without inspecting or opening hardware."""
    robot_key = _resolved_robot_software_key(robot_software, compatibility)
    matrix = compatibility["robot_software_to_rdk"]
    accepted_raw = matrix.get(robot_key)
    if not isinstance(accepted_raw, list):
        return (
            None,
            f"robot software {robot_software} is not in the compatibility matrix",
        )

    accepted = [_canonical_rdk_version(value, compatibility) for value in accepted_raw]
    reviewed = set(compatibility["project_reviewed_rdk_versions"])
    reviewed_candidates = [version for version in accepted if version in reviewed]
    if explicit_rdk is not None:
        selected = _canonical_rdk_version(explicit_rdk, compatibility)
        if selected not in accepted:
            return (
                None,
                f"RDK {selected} is incompatible with robot software {robot_key}; accepted: {', '.join(accepted)}",
            )
    elif reviewed_candidates:
        selected = reviewed_candidates[-1]
    else:
        selected = accepted[-1]

    if selected not in reviewed:
        detail = (
            f"robot software {robot_key} requires {selected}, but the adapter/release only reviews "
            f"{', '.join(sorted(reviewed))}; obtain an approved RDK profile and complete an API/hardware review"
        )
        return None, detail
    return selected, None


def _check_installed_flexiv_rdk(
    report: Report, python: Path, expected: str | None
) -> None:
    if expected is None:
        report.add(
            "INFO",
            "installed Flexiv RDK",
            "skipped because compatibility selection failed",
        )
        return
    ok, detail = _installed_distribution_version(python, "flexivrdk")
    if not ok:
        report.add("BLOCKER", "installed Flexiv RDK", detail)
    elif detail != expected:
        report.add(
            "BLOCKER", "installed Flexiv RDK", f"expected {expected}; found {detail}"
        )
    else:
        report.add("PASS", "installed Flexiv RDK", f"{detail} in {python.expanduser()}")


def _profile_probe(
    python: Path,
    repo_root: Path,
    config_name: str,
    target: str,
) -> tuple[dict[str, Any] | None, str]:
    code = f"""
import importlib.util
import json
import math
from importlib import metadata
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root))
required_modules = [
    "can", "cv2", "dm3_core", "hydra", "i2rt", "jax", "lktech_motor_sdk",
    "mujoco", "nmx_msg", "numpy", "pyarrow", "ruamel.yaml", "serial", "torch",
]
if sys.argv[3] == "flexiv":
    required_modules.append("flexivrdk")

def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

missing_modules = [name for name in required_modules if not module_available(name)]
if missing_modules:
    raise ModuleNotFoundError("missing source profile modules: " + ", ".join(missing_modules))

from incubator.control_panel.runtime_config import compose_runtime_config
from utils.openpi_source import activate_openpi_source

config = compose_runtime_config(sys.argv[2])
policy = config.get("policy", {{}})
params = policy.get("params", {{}})
activate_openpi_source(policy.get("openpi_src"), repo_root=repo_root)
from utils.openpi_checkpoint import validate_openpi_checkpoint_contract

controls = policy.get("controls", [])
action_dim = sum(math.prod(control.get("shape", [])) for control in controls)
checkpoint = params.get("ckpt_dir")
contract = validate_openpi_checkpoint_contract(
    params.get("pi_config_name"),
    checkpoint,
    expected_action_dim=action_dim or None,
)

import can
import cv2
import dm3_core
import hydra
import i2rt
import jax
import jax.numpy as jnp
import lktech_motor_sdk
import mujoco
import nmx_msg
import numpy
import pyarrow
from ruamel.yaml import YAML
import serial
import torch
from dm3_core.mcap import McapFileType, ReaderFactory, WriterFactory
from dm3_core.nedf import NEDFFactory, NEDFReaderBase, NEDFType, NEDFWriterBase
from nmx_msg.Arm_pb2 import ArmEndForce, ArmState, ArmTcpCmd, ArmTcpState
from nmx_msg.Hand_pb2 import HandCmd, HandState
from nmx_msg.Image_pb2 import RGBD
from nmx_msg.Metadata_pb2 import Metadata
from policy.openpi_umi_policy import OpenPiUmiPolicy

required_nedf_api = (
    (NEDFFactory, "get_writer"),
    (NEDFFactory, "get_reader"),
    (NEDFWriterBase, "write_message"),
    (NEDFReaderBase, "iter_decoded_messages"),
    (NEDFReaderBase, "get_raw_topics_timestamps"),
)
assert all(callable(getattr(owner, method, None)) for owner, method in required_nedf_api)
assert ReaderFactory and WriterFactory and McapFileType and hasattr(NEDFType, "NEDF3")
for message_type in (RGBD, ArmTcpCmd, ArmTcpState, ArmState, ArmEndForce, HandCmd, HandState, Metadata):
    assert hasattr(message_type, "SerializeToString")
if sys.argv[3] == "flexiv":
    import flexivrdk
    assert flexivrdk

jax_sum = float(jnp.arange(8, dtype=jnp.float32).sum())
torch_cuda = torch.cuda.is_available()
torch_sum = torch.arange(8, device="cuda").sum().item() if torch_cuda else None
payload = {{
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "openpi": metadata.version("openpi"),
    "flexivrdk": metadata.version("flexivrdk") if sys.argv[3] == "flexiv" else None,
    "dm3_core": metadata.version("dm3-core"),
    "nmx_msg": metadata.version("nmx-msg"),
    "i2rt": metadata.version("i2rt"),
    "mujoco": mujoco.__version__,
    "pyarrow": pyarrow.__version__,
    "jax_backend": jax.default_backend(),
    "jax_sum": jax_sum,
    "torch": torch.__version__,
    "torch_cuda": torch_cuda,
    "torch_sum": torch_sum,
    "checkpoint": contract,
}}
assert can and cv2 and dm3_core and hydra and i2rt and lktech_motor_sdk
assert nmx_msg and OpenPiUmiPolicy and YAML and serial
print({PROFILE_PROBE_MARKER!r} + json.dumps(payload, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [
                str(python.expanduser()),
                "-c",
                code,
                str(repo_root),
                config_name,
                target,
            ],
            cwd=str(repo_root),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode:
        detail = output.splitlines()[-1] if output else f"exit {result.returncode}"
        return None, detail
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(PROFILE_PROBE_MARKER):
            try:
                payload = json.loads(line.removeprefix(PROFILE_PROBE_MARKER))
            except json.JSONDecodeError as exc:
                return None, f"invalid profile probe JSON: {exc}"
            return payload if isinstance(payload, dict) else None, ""
    return None, "profile probe returned no result marker"


def _check_source_profile(
    report: Report,
    python: Path,
    repo_root: Path,
    config_name: str,
    target: str,
) -> None:
    payload, error = _profile_probe(python, repo_root, config_name, target)
    if payload is None:
        report.add("BLOCKER", "source OpenPI profile", error)
        return
    python_version = str(payload.get("python", ""))
    if not python_version.startswith("3.12."):
        report.add(
            "BLOCKER",
            "source OpenPI profile",
            f"expected Python 3.12; found {python_version}",
        )
        return
    if payload.get("jax_backend") != "gpu" or payload.get("torch_cuda") is not True:
        report.add(
            "BLOCKER",
            "source OpenPI GPU",
            f"JAX backend={payload.get('jax_backend')}, torch CUDA={payload.get('torch_cuda')}",
        )
        return
    contract = payload.get("checkpoint", {})
    report.add(
        "PASS",
        "source OpenPI profile",
        f"Python {python_version}, OpenPI {payload.get('openpi')}, RDK {payload.get('flexivrdk')}, "
        f"asset {contract.get('asset_id')}, action_dim {contract.get('action_dim')}",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target", choices=("policy-only", "flexiv"), default="flexiv")
    parser.add_argument("--robot-software-version")
    parser.add_argument("--rdk-version")
    parser.add_argument("--config")
    parser.add_argument("--device-python", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compatibility = _load_compatibility()
    repo_root = args.repo_root.expanduser().resolve()
    report = Report()
    default_config = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"

    _check_repository(report, repo_root)
    _check_host(report, args.device_python)
    _check_config(report, repo_root, args.config or default_config)
    if args.target == "flexiv":
        selected_rdk = _check_flexiv(
            report, args.robot_software_version, args.rdk_version, compatibility
        )
        if args.device_python is not None:
            _check_installed_flexiv_rdk(report, args.device_python, selected_rdk)
    else:
        report.add("INFO", "Flexiv RDK", "not required for a policy-only setup")
    if args.device_python is not None:
        _check_source_profile(
            report,
            args.device_python,
            repo_root,
            args.config or default_config,
            args.target,
        )
    else:
        report.add(
            "ACTION",
            "source OpenPI profile",
            "after installation, rerun with --device-python <absolute-env-python>",
        )

    payload = {"ready": not report.has_blocker, "checks": report.records}
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        for record in report.records:
            print(f"[{record['status']}] {record['subject']}: {record['detail']}")
        print(
            "\nREADY"
            if payload["ready"]
            else "\nNOT READY: resolve BLOCKER entries before installation/launch."
        )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
