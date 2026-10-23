"""Dependency profile ownership contracts.

Author: SuperInference Contributors
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import pytest


pytestmark = [pytest.mark.architecture, pytest.mark.dependency]
ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements"
EXPECTED_PROFILES = {
    "requirements.txt",
    "requirements-all.txt",
    "requirements-dev.txt",
    "requirements-hardware.txt",
    "requirements-policy.txt",
    "requirements-robotwin.txt",
    "requirements-scripts.txt",
}


def _requirements(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or line.startswith("-r "):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        assert match is not None, f"unparseable requirement in {path.name}: {raw_line}"
        names.add(match.group(1).casefold().replace("_", "-"))
    return names


def _top_level_imports(paths: Iterable[Path]) -> set[str]:
    imports: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", maxsplit=1)[0])
    return imports


def test_project_dependency_profiles_are_centralized() -> None:
    assert list(ROOT.glob("requirements*.txt")) == []
    assert {path.name for path in REQUIREMENTS.glob("requirements*.txt")} == (
        EXPECTED_PROFILES
    )


def test_base_profile_remains_hardware_and_vendor_free() -> None:
    base = _requirements(REQUIREMENTS / "requirements.txt")
    forbidden = {
        "dm3-core",
        "flexivrdk",
        "nmx-msg",
        "nmx-nedf-api",
        "opencv-python",
        "pyopenxr",
        "pyrealsense2",
        "pyserial",
        "record3d",
        "torch",
        "torchvision",
    }
    assert base.isdisjoint(forbidden)
    assert {
        "hydra-core",
        "loguru",
        "numpy",
        "omegaconf",
        "psutil",
        "pydantic",
        "pyyaml",
        "rerun-sdk",
        "requests",
        "scipy",
    } <= base


def test_optional_dependencies_are_owned_by_named_profiles() -> None:
    profiles = {
        name: _requirements(REQUIREMENTS / f"requirements-{name}.txt")
        for name in ("dev", "scripts", "hardware", "policy")
    }
    assert {
        "imageio",
        "pre-commit",
        "pytest",
        "pytest-mock",
        "ruff",
        "transforms3d",
    } <= profiles["dev"]
    assert {
        "dash",
        "h5py",
        "imagecodecs",
        "matplotlib",
        "opencv-python",
        "pandas",
        "pyarrow",
        "plotly",
        "scikit-learn",
        "zarr",
    } <= profiles["scripts"]
    assert {
        "fastapi",
        "matplotlib",
        "mujoco",
        "opencv-python",
        "pandas",
        "pyopenxr",
        "pyrealsense2",
        "pyserial",
        "record3d",
        "transforms3d",
        "uvicorn",
    } <= profiles["hardware"]
    assert {
        "dill",
        "einops",
        "msgpack",
        "websocket-client",
        "websockets",
    } <= profiles["policy"]


def test_reorganized_tree_imports_are_owned_by_profiles() -> None:
    profile_modules = {
        "scripts": {
            "dash": "dash",
            "h5py": "h5py",
            "pandas": "pandas",
            "sklearn": "scikit-learn",
            "zarr": "zarr",
        },
        "hardware": {
            "fastapi": "fastapi",
            "flask": "flask",
            "pandas": "pandas",
            "pyrealsense2": "pyrealsense2",
            "serial": "pyserial",
            "transforms3d": "transforms3d",
            "uvicorn": "uvicorn",
        },
        "policy": {
            "dill": "dill",
            "einops": "einops",
            "msgpack": "msgpack",
            "websockets": "websockets",
        },
    }
    import_roots = {
        "scripts": (ROOT / "scripts",),
        "hardware": (ROOT / "devices",),
        "policy": (ROOT / "policy", ROOT / "consumer" / "inference", ROOT / "utils"),
    }
    for profile, module_requirements in profile_modules.items():
        imports = _top_level_imports(
            path
            for directory in import_roots[profile]
            for path in directory.rglob("*.py")
        )
        requirements = _requirements(REQUIREMENTS / f"requirements-{profile}.txt")
        assert module_requirements.keys() <= imports
        assert set(module_requirements.values()) <= requirements


def test_all_profile_includes_each_named_profile_once() -> None:
    includes = [
        line.strip()
        for line in (REQUIREMENTS / "requirements-all.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip().startswith("-r ")
    ]
    assert includes == [
        "-r requirements-dev.txt",
        "-r requirements-scripts.txt",
        "-r requirements-hardware.txt",
        "-r requirements-policy.txt",
    ]


def test_robotwin_profile_preserves_its_separate_vendor_stack() -> None:
    profile = REQUIREMENTS / "requirements-robotwin.txt"
    includes = [
        line.strip()
        for line in profile.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("-r ")
    ]

    assert includes == [
        "-r requirements.txt",
        "-r ../third_party/RoboTwin/scripts/requirements.txt",
    ]
    assert "setuptools" in _requirements(profile)
    assert "-r requirements-robotwin.txt" not in (
        REQUIREMENTS / "requirements-all.txt"
    ).read_text(encoding="utf-8")


def test_source_environment_includes_comment_preserving_yaml_writer() -> None:
    requirements = _requirements(
        ROOT / "incubator/control_panel/release/requirements/main_py312.txt"
    )
    assert "ruamel-yaml" in requirements
    builder = (
        ROOT / "incubator/control_panel/release/create_source_py312_env.sh"
    ).read_text(encoding="utf-8")
    assert "from ruamel.yaml import YAML" in builder
    assert "import serial" in builder


def test_source_recording_installer_requires_approved_wheelhouse() -> None:
    release = ROOT / "incubator/control_panel/release"
    builder = (release / "create_source_py312_env.sh").read_text(encoding="utf-8")
    installer = (release / "install_source_recording_profile.sh").read_text(
        encoding="utf-8"
    )

    assert '--find-links "$wheelhouse"' in builder
    assert "requirements/main_py312.txt" in builder
    assert '--find-links "$wheelhouse"' in installer
    assert "dm3_core-0.2.18-*.whl" in installer
    assert "nmx_msg-2.2.0-*.whl" in installer
    assert "from consumer.io.async_infer_data_saver_NEDF3 import" in installer
    assert "NEDF3 recording profile import probe passed" in installer


def test_vendor_model_stacks_remain_vendor_owned() -> None:
    policy = _requirements(REQUIREMENTS / "requirements-policy.txt")
    assert policy.isdisjoint(
        {
            "accelerate",
            "diffusers",
            "openpi",
            "torch",
            "torchvision",
            "transformers",
        }
    )

    openpi_metadata_path = ROOT / "third_party" / "openpi" / "pyproject.toml"
    lingbot_requirements_path = ROOT / "third_party" / "nmx_vla" / "requirements.txt"
    missing_metadata = [
        path.relative_to(ROOT).as_posix()
        for path in (openpi_metadata_path, lingbot_requirements_path)
        if not path.is_file()
    ]
    if missing_metadata:
        pytest.skip(
            reason="optional vendor submodule metadata is not initialized; run vendor jobs for ownership checks"
        )

    openpi_metadata = openpi_metadata_path.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11"' in openpi_metadata
    assert re.search(r'"openpi-client(?:[<>=].*)?"', openpi_metadata)
    assert re.search(r'"torch(?:[<>=].*)?"', openpi_metadata)

    lingbot_requirements = _requirements(lingbot_requirements_path)
    assert "diffusers" in lingbot_requirements
    assert "torch" not in lingbot_requirements
    assert {"transformers", "websockets"} <= lingbot_requirements
    assert (
        "fastwam"
        in (ROOT / "policy" / "lingbot_va_policy.py")
        .read_text(encoding="utf-8")
        .casefold()
    )
    assert (
        "fastwam"
        in (ROOT / "consumer" / "inference" / "lingbot_va_server.py")
        .read_text(encoding="utf-8")
        .casefold()
    )
