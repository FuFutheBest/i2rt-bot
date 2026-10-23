"""Control-panel-owned inference test mode and rollout configuration."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from utils.inference_options import (
    MOCK_INFERENCE_ENV,
    MOCK_POLICY_CLASS,
    POLICY_OVERRIDES_ENV,
    as_bool,
    mock_gripper_range,
    mock_inference_config_error,
)

PREFERENCES_FILENAME = "inference-options.json"


def default_preferences_path(
    environment: Optional[Mapping[str, str]] = None,
) -> Path:
    env = os.environ if environment is None else environment
    data_root = Path(
        env.get(
            "SUPERINFERENCE_DATA_ROOT",
            str(Path.home() / ".local/share/superinference-control-panel"),
        )
    ).expanduser()
    return data_root / PREFERENCES_FILENAME


def load_rollout_output_root(path: Path) -> Optional[Path]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    raw_root = payload.get("rollout_output_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        return None
    candidate = Path(raw_root.strip()).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate.resolve(strict=False)


def save_rollout_output_root(path: Path, output_root: Path) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    payload = {
        "schema_version": 1,
        "rollout_output_root": str(Path(output_root).expanduser().resolve()),
    }
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def build_control_panel_recording_config(
    config: Mapping[str, Any],
    *,
    enabled: bool,
    output_root: Optional[Path],
) -> dict[str, Any]:
    """Replace YAML rollout settings with the panel's stable runtime profile."""
    result = copy.deepcopy(dict(config))
    policy = result.get("policy")
    if not isinstance(policy, dict):
        policy = {}
        result["policy"] = policy
    params = policy.get("params")
    task = params.get("task") if isinstance(params, Mapping) else None
    controls = policy.get("controls")
    control_shms = copy.deepcopy(controls) if isinstance(controls, list) else []

    from .runtime_config import extract_camera_slots

    slots = extract_camera_slots(result)

    def recorder_device_key(slot: Mapping[str, Any]) -> Optional[str]:
        key = slot.get("device_key")
        if not isinstance(key, str) or ":" not in key:
            return None
        device_class, device_class_id = key.rsplit(":", 1)
        return f"{device_class}_{device_class_id}"

    camera_mapping = {
        f"{role}_camera": (
            recorder_device_key(slot)
            if isinstance(slot, Mapping) and slot.get("configured")
            else None
        )
        for role, slot in slots.items()
        if role in {"head", "left", "right"}
    }
    for role in ("head", "left", "right"):
        camera_mapping.setdefault(f"{role}_camera", None)

    rollout_enabled = as_bool(enabled)
    policy["infer_rollout"] = {
        "enabled": rollout_enabled,
        "required": rollout_enabled,
        "summary_shm_name": str(result.get("summary_shm", "device_summary_data")),
        "output_dir": str(output_root) if output_root is not None else "",
        "save_interval": 0.001,
        "start_idx": 0,
        "image_size": None,
        "async_write": True,
        "queue_size": 50000,
        "robot_type": "Rizon",
        "data_format": "NEDF3",
        "task_name": str(task or "control_panel_inference_rollout"),
        "ready_timeout": 15.0,
        "auto_start": False,
        "anomaly_config_path": "configs/anomaly_config/rizon_fisheye_umi3.yaml",
        "policy_outputs": {
            "value": {"enabled": False},
            "q_value": {"enabled": False},
            "q_value_timestamps": {"enabled": False},
            "action_chunk": {
                "name": "action_chunk",
                "device_type": "ActionChunk",
                "buffer_size": 200,
                "dtype": "float64",
                "enabled": rollout_enabled,
            },
        },
        "control_shms": control_shms,
        "episode_control": {
            "enabled": rollout_enabled,
            "success_key": "s",
            "failure_key": "f",
            "discard_key": "d",
            "auto_restart": True,
            "home_wait_time": 15.0,
            "halt_timeout": 30.0,
        },
        **camera_mapping,
    }
    return result


__all__ = [
    "MOCK_INFERENCE_ENV",
    "MOCK_POLICY_CLASS",
    "POLICY_OVERRIDES_ENV",
    "as_bool",
    "build_control_panel_recording_config",
    "default_preferences_path",
    "load_rollout_output_root",
    "mock_gripper_range",
    "mock_inference_config_error",
    "save_rollout_output_root",
]
