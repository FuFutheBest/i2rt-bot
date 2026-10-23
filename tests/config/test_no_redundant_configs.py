from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_ONLY_KEYS = {"chunk_length", "fps", "left_capture_id", "pi_config_name", "right_capture_id", "task"}
GRIPPER_IDENTITY_KEYS = {
    "calibration_file",
    "calibration_files",
    "motor_id",
    "motor_ids",
    "port",
    "ports",
    "serial",
    "serial_number",
}


def _without_deployment_identity(value: Any, *, in_gripper: bool = False) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            normalized = str(key).casefold()
            child_in_gripper = in_gripper or normalized == "gripper"
            ignored = (
                normalized in DEPLOYMENT_ONLY_KEYS
                or "checkpoint" in normalized
                or "ckpt" in normalized
                or (child_in_gripper and normalized in GRIPPER_IDENTITY_KEYS)
            )
            if not ignored:
                result[str(key)] = _without_deployment_identity(child, in_gripper=child_in_gripper)
        return result
    if isinstance(value, list):
        return [_without_deployment_identity(child, in_gripper=in_gripper) for child in value]
    return value


def test_configs_are_not_duplicated_by_deployment_only_fields() -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        # Workstation overlays contain only deployment values by design and
        # are not repository-owned launch presets.
        if path.is_relative_to(ROOT / "configs" / "local"):
            continue
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        fingerprint = json.dumps(
            _without_deployment_identity(parsed),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        groups[fingerprint].append(path.relative_to(ROOT).as_posix())

    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    assert not duplicates, f"redundant config groups: {duplicates}"
