"""Configuration registry contract tests.

Author: OpenAI Assistant
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml


def _registry_keys(path: Path, variable_name: str) -> set[str]:
    """Read string keys from a top-level registry dictionary without importing hardware modules."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    if not isinstance(node.value, ast.Dict):
                        return set()
                    keys: set[str] = set()
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            keys.add(key.value)
                    return keys
    return set()


def _yaml_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.yaml"))


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


KNOWN_UNREGISTERED_POLICY_REFERENCES: set[str] = set()

CANONICAL_ACTION_EXECUTOR_PRESETS = {
    "auto_v2.yaml",
    "base.yaml",
    "base_auto_drop.yaml",
    "default.yaml",
    "infer_i2rt_joint.yaml",
    "robotwin.yaml",
    "smooth.yaml",
}

CANONICAL_DEVICE_MANAGER_PRESETS = {
    "base.yaml",
    "camera_sync.yaml",
    "robotwin.yaml",
    "sync.yaml",
}


@pytest.mark.config
def test_device_config_classes_are_registered(repo_root: Path) -> None:
    """Every device preset should refer to a known device registry key."""
    device_registry = _registry_keys(repo_root / "devices" / "__init__.py", "DEVICE_CLASSES")
    assert device_registry, "DEVICE_CLASSES registry should not be empty"

    unknown: list[str] = []
    for path in sorted((repo_root / "configs" / "devices").rglob("*.yaml")):
        data = _load_yaml(path)
        devices = data.get("devices", [])
        if isinstance(devices, dict):
            devices = devices.get("devices", [])
        for device in devices or []:
            if (
                isinstance(device, dict)
                and device.get("enabled", True) is not False
                and device.get("class") not in device_registry
            ):
                unknown.append(f"{path.name}:{device.get('class')}")

    assert unknown == []


@pytest.mark.config
def test_policy_classes_referenced_by_launch_configs_are_registered(repo_root: Path) -> None:
    """Policy launch configs should refer to known policy registry keys."""
    policy_registry = _registry_keys(repo_root / "policy" / "__init__.py", "POLICY_CLASSES")
    assert policy_registry, "POLICY_CLASSES registry should not be empty"

    unknown: list[str] = []
    for path in sorted((repo_root / "configs" / "policy_infer").rglob("*.yaml")):
        policy = _load_yaml(path).get("policy")
        if isinstance(policy, dict) and policy.get("class") not in (None, *policy_registry):
            reference = f"{path.name}:{policy.get('class')}"
            if reference not in KNOWN_UNREGISTERED_POLICY_REFERENCES:
                unknown.append(reference)

    assert unknown == []


@pytest.mark.config
def test_no_known_unregistered_policy_references_remain(repo_root: Path) -> None:
    """Known policy registry drift should stay empty after config cleanup."""
    policy_registry = _registry_keys(repo_root / "policy" / "__init__.py", "POLICY_CLASSES")
    unresolved: list[str] = []
    for reference in KNOWN_UNREGISTERED_POLICY_REFERENCES:
        config_name, policy_class = reference.split(":", maxsplit=1)
        config = _load_yaml(repo_root / "configs" / config_name)
        policy = config.get("policy", {})
        if isinstance(policy, dict) and policy.get("class") == policy_class:
            if policy_class not in policy_registry:
                unresolved.append(reference)

    assert unresolved == []


@pytest.mark.config
def test_manager_and_chunk_manager_types_are_registered(repo_root: Path) -> None:
    """Manager and chunk-manager types in config files should be known registry keys."""
    manager_registry = _registry_keys(repo_root / "manager" / "__init__.py", "MANAGER_CLASSES")
    chunk_registry = _registry_keys(repo_root / "manager" / "utils" / "__init__.py", "CHUNK_MANAGERS")

    assert manager_registry
    assert chunk_registry

    unknown_managers: list[str] = []
    for path in _yaml_files(repo_root / "configs" / "device_manager"):
        manager = _load_yaml(path).get("device_manager", {})
        manager_type = manager.get("type") if isinstance(manager, dict) else None
        if manager_type not in (None, *manager_registry):
            unknown_managers.append(f"{path.name}:{manager_type}")

    unknown_chunks: list[str] = []
    for path in _yaml_files(repo_root / "configs" / "action_executor"):
        executor = _load_yaml(path).get("action_executor", {})
        if not isinstance(executor, dict):
            continue
        chunk_manager = executor.get("chunk_manager", {})
        if isinstance(chunk_manager, dict):
            manager_type = chunk_manager.get("type")
            if manager_type not in (None, *chunk_registry):
                unknown_chunks.append(f"{path.name}:{manager_type}")

    assert unknown_managers == []
    assert unknown_chunks == []


@pytest.mark.config
def test_action_executor_presets_are_canonical(repo_root: Path) -> None:
    """Action-executor presets stay limited to the shared operating modes."""
    action_executor_root = repo_root / "configs" / "action_executor"
    assert {path.name for path in _yaml_files(action_executor_root)} == (
        CANONICAL_ACTION_EXECUTOR_PRESETS
    )


@pytest.mark.config
def test_device_manager_presets_are_canonical(repo_root: Path) -> None:
    """Device-manager presets stay grouped by behavior, not hardware names."""
    device_manager_root = repo_root / "configs" / "device_manager"
    assert {path.name for path in _yaml_files(device_manager_root)} == (
        CANONICAL_DEVICE_MANAGER_PRESETS
    )


@pytest.mark.config
def test_device_manager_defaults_use_canonical_presets(repo_root: Path) -> None:
    """Every launcher must select a retained behavior-level manager preset."""
    preset_names = {path.removesuffix(".yaml") for path in CANONICAL_DEVICE_MANAGER_PRESETS}
    references: list[str] = []

    for path in sorted((repo_root / "configs").rglob("*.yaml")):
        defaults = _load_yaml(path).get("defaults", [])
        if not isinstance(defaults, list):
            continue
        for default in defaults:
            if not isinstance(default, dict):
                continue
            for group_name, preset_name in default.items():
                if str(group_name).lstrip("/") == "device_manager":
                    if preset_name not in preset_names:
                        references.append(f"{path.relative_to(repo_root)}:{preset_name}")

    assert references == []
