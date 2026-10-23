#!/usr/bin/env python3
"""Evaluate one dual-arm Diffusion checkpoint from recorded observations.

This action-only diagnostic entrypoint composes the selected native Hydra
configuration, reads one exact ``TimestampedObservationRecorder`` input,
reconstructs that policy's observation history from raw recorder frames, and
invokes the configured live Diffusion policy. It never creates shared memory,
starts a policy connector, or opens a robot-control output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.fastwam_offline_utils import ObservationDataset  # noqa: E402
from utils.superoffline_action_calibration import prepare_action_calibration  # noqa: E402
from utils.superoffline_registry import (  # noqa: E402
    DUAL_ARM_TCP_ACTION_CONTRACT,
    DUAL_ARM_TCP_ACTION_FRAME,
    DUAL_ARM_TCP_ACTION_LAYOUT,
    DUAL_ARM_TCP_ACTION_UNITS,
    get_superoffline_model_contract,
)
from utils.superoffline_observation import (  # noqa: E402
    read_observation_binding,
    validate_binding_policy_devices,
)


DEFAULT_CONFIG_NAME = "policy_infer/diffusion/config_infer_rizon_eef_dual_abs_mode_umi"
SUPPORTED_POLICY_CLASSES = frozenset({"DiffusionPolicy", "DiffusionPolicyIPhoneUMI"})
DUAL_ARM_ACTION_DIM = 16
DUAL_ARM_ACTION_LAYOUT = DUAL_ARM_TCP_ACTION_LAYOUT
_CONFIG_COMPOSE_LOCK = threading.RLock()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping in the selected config")
    return value


def _policy_class(policy_config: Mapping[str, Any]) -> str:
    policy_class = policy_config.get("class")
    if not isinstance(policy_class, str) or policy_class not in SUPPORTED_POLICY_CLASSES:
        raise ValueError(
            "Diffusion offline evaluation supports only "
            f"{sorted(SUPPORTED_POLICY_CLASSES)}, got {policy_class!r}"
        )
    return policy_class


def _config_for_offline_evaluation(
    config_name: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Compose the exact native configuration used by the live consumer."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from utils.runtime_profiles import RuntimeRegistryError, normalize_config_name

    try:
        name = normalize_config_name(config_name)
    except RuntimeRegistryError as exc:
        raise ValueError(f"invalid native config name: {config_name!r}") from exc
    if name is None:
        raise ValueError("native config name is required")
    config_dir = REPO_ROOT / "configs"
    if not any((config_dir / f"{name}{suffix}").is_file() for suffix in (".yaml", ".yml")):
        raise ValueError(f"unknown native config: {name}")
    with _CONFIG_COMPOSE_LOCK:
        hydra_state = HydraConfig.instance()
        previous_hydra_config = hydra_state.cfg
        try:
            with initialize_config_dir(
                config_dir=str(config_dir),
                version_base="1.3",
                job_name="evaluate_diffusion_offline",
            ):
                composed = compose(config_name=name, return_hydra_config=True)
                hydra_state.set_config(composed)
                policy_value = composed.get("policy")
                policy_config = _mapping(
                    OmegaConf.to_container(policy_value, resolve=True), field="policy"
                )
        finally:
            hydra_state.cfg = previous_hydra_config
    _policy_class(policy_config)
    return {"policy": policy_config}, policy_config


def _selected_devices(policy_config: Mapping[str, Any]) -> tuple[list[str], str]:
    observation_config = _mapping(policy_config.get("obs"), field="policy.obs")
    raw_devices = observation_config.get("devices")
    if isinstance(raw_devices, (str, bytes)) or not isinstance(raw_devices, list):
        raise ValueError("policy.obs.devices must be a list")

    devices = [str(value).strip() for value in raw_devices if str(value).strip()]
    camera_devices = [name for name in devices if "camera" in name.lower()]
    robot_devices = [
        name
        for name in devices
        if name.lower().startswith(("rizon", "robot"))
    ]
    if len(camera_devices) != 2 or len(robot_devices) != 1:
        raise ValueError(
            "dual-arm Diffusion offline evaluation requires exactly two camera devices "
            "and one Rizon/robot device, got "
            f"cameras={camera_devices}, robots={robot_devices}"
        )
    return camera_devices, robot_devices[0]


def _control_shape_size(value: Any) -> int | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    size = 1
    for dimension in value:
        if isinstance(dimension, bool):
            return None
        try:
            parsed = int(dimension)
        except (TypeError, ValueError):
            return None
        if parsed < 1:
            return None
        size *= parsed
    return size


def _validate_dual_arm_contract(policy_config: Mapping[str, Any]) -> None:
    """Reject single-arm configs before loading a heavyweight checkpoint."""
    params = _mapping(policy_config.get("params", {}), field="policy.params")
    configured_robot_count = params.get("num_robots")
    if configured_robot_count is not None and int(configured_robot_count) != 2:
        raise ValueError(
            "dual-arm Diffusion offline evaluation requires policy.params.num_robots=2"
        )

    controls = policy_config.get("controls")
    if not isinstance(controls, list):
        raise ValueError("dual-arm Diffusion offline evaluation requires policy.controls")
    control_sizes = [
        _control_shape_size(control.get("shape"))
        for control in controls
        if isinstance(control, Mapping)
    ]
    if DUAL_ARM_ACTION_DIM not in control_sizes:
        raise ValueError(
            "dual-arm Diffusion offline evaluation requires a 16-dimensional "
            f"policy control, got shapes={control_sizes}"
        )


def _optional_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _configured_policy_kwargs(
    params: Mapping[str, Any],
    *,
    checkpoint: Path,
    device: str,
    camera_devices: Sequence[str],
    robot_device: str,
) -> dict[str, Any]:
    """Keep configured preprocessing values while forcing offline-safe display mode."""
    kwargs: dict[str, Any] = {
        "ckpt_dir": str(checkpoint),
        "num_inference_steps": int(params.get("num_inference_steps", 16)),
        "device": device,
        "enable_visualization": False,
        "camera_devices": list(camera_devices),
        "robot_device": robot_device,
    }
    for key, converter in (
        ("obs_horizon", int),
        ("downsample", int),
        ("obs_time_interval", float),
        ("preprocess_resize_height", int),
        ("preprocess_resize_width", int),
        ("center_crop_min_ratio", float),
        ("center_crop_max_ratio", float),
    ):
        value = params.get(key)
        if value is not None:
            kwargs[key] = converter(value)

    tcp_umi_json_path = _optional_path(params.get("tcp_umi_json_path"))
    if tcp_umi_json_path is not None:
        kwargs["tcp_umi_json_path"] = tcp_umi_json_path
    return kwargs


def _load_policy_class(policy_class: str) -> type[Any]:
    """Import only the explicitly selected implementation and its vendor stack."""
    if policy_class == "DiffusionPolicy":
        from policy.diffusion_policy import DiffusionPolicy

        return DiffusionPolicy
    if policy_class == "DiffusionPolicyIPhoneUMI":
        from policy.diffusion_policy_iphone_umi import DiffusionPolicyIPhoneUMI

        return DiffusionPolicyIPhoneUMI
    raise AssertionError(f"unsupported policy class escaped validation: {policy_class!r}")


def _build_policy(
    policy_config: Mapping[str, Any],
    *,
    checkpoint: Path,
    device: str,
    camera_devices: Sequence[str],
    robot_device: str,
) -> Any:
    policy_class = _policy_class(policy_config)
    params = _mapping(policy_config.get("params", {}), field="policy.params")
    kwargs = _configured_policy_kwargs(
        params,
        checkpoint=checkpoint,
        device=device,
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    if policy_class == "DiffusionPolicyIPhoneUMI":
        from utils.inference_options import as_bool

        for key in (
            "left_robot_base_to_world_transform",
            "right_robot_base_to_world_transform",
        ):
            value = _optional_path(params.get(key))
            if value is not None:
                kwargs[key] = value
        for key in ("use_relative_action", "use_inter_gripper"):
            if key in params:
                kwargs[key] = as_bool(params[key])
    return _load_policy_class(policy_class)(**kwargs)


def _set_seed(seed: int) -> None:
    """Seed the random sources used by the live Diffusion implementations."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_like_live(
    policy: Any,
    observation: Mapping[str, Any],
    *,
    robot_device: str,
    chunk_length: int,
) -> np.ndarray:
    """Call the policy's public preprocessing and action-conversion path."""
    result = policy.predict(
        dict(observation),
        action_configs=[{"device_name": robot_device}],
        chunk_length=chunk_length,
    )
    if not isinstance(result, Mapping) or robot_device not in result:
        raise ValueError(
            "Diffusion policy did not return an action for configured robot "
            f"{robot_device!r}"
        )
    action = np.asarray(result[robot_device], dtype=np.float64)
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != DUAL_ARM_ACTION_DIM:
        raise ValueError(
            "Diffusion policy must return canonical dual-arm action [T,16], "
            f"got {action.shape}"
        )
    return np.ascontiguousarray(action)


def _action_step_interval_ns(policy: Any) -> tuple[int | None, str | None]:
    """Expose a policy-owned interval only when the live class provides one."""
    value = getattr(policy, "action_step_interval_ns", None)
    if isinstance(value, bool):
        return None, None
    try:
        interval_ns = int(value)
    except (TypeError, ValueError):
        return None, None
    if interval_ns < 1:
        return None, None
    return interval_ns, "policy.action_step_interval_ns"


def _observation_history_settings(policy: Any) -> tuple[int, int, float]:
    """Read the target policy's live history contract without config defaults."""
    values = {
        "obs_horizon": getattr(policy, "obs_horizon", None),
        "downsample": getattr(policy, "downsample", None),
    }
    parsed: dict[str, int] = {}
    for field, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"Diffusion policy {field} must be a positive integer")
        try:
            parsed[field] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Diffusion policy {field} must be a positive integer"
            ) from exc
        if parsed[field] < 1:
            raise ValueError(f"Diffusion policy {field} must be a positive integer")
    try:
        interval_s = float(getattr(policy, "obs_time_interval", None))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Diffusion policy obs_time_interval must be positive and finite"
        ) from exc
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError(
            "Diffusion policy obs_time_interval must be positive and finite"
        )
    return parsed["obs_horizon"], parsed["downsample"], interval_s


def _resolve_inference_index(dataset: ObservationDataset, requested: int) -> int:
    index = requested
    if index < 0:
        index += len(dataset.inferences)
    if index < 0 or index >= len(dataset.inferences):
        raise IndexError(f"inference-index {requested} is outside the recorded run")
    return index


def _dataset_from_binding(args: argparse.Namespace) -> tuple[ObservationDataset, Any | None]:
    """Apply an executor-owned role binding without accepting user aliases."""
    if args.observation_binding is None:
        return ObservationDataset(args.observation_dir), None
    binding = read_observation_binding(
        args.observation_binding,
        binding_index=int(args.binding_index),
        config_name=str(args.config_name),
        observation_dir=args.observation_dir,
    )
    return ObservationDataset(
        args.observation_dir,
        device_aliases=binding.target_to_source,
    ), binding


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--inference-index", type=int, default=-1)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-binding", type=Path, default=None)
    parser.add_argument("--binding-index", type=int, default=0)
    parser.add_argument("--action-calibration-snapshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    _root_config, policy_config = _config_for_offline_evaluation(args.config_name)
    _policy_class(policy_config)
    _validate_dual_arm_contract(policy_config)
    camera_devices, robot_device = _selected_devices(policy_config)
    params = _mapping(policy_config.get("params", {}), field="policy.params")
    checkpoint = args.checkpoint.expanduser()
    dataset, binding = _dataset_from_binding(args)
    if binding is not None:
        camera_devices, robot_device = validate_binding_policy_devices(
            binding,
            camera_devices=camera_devices,
            robot_device=robot_device,
            config_name=str(args.config_name),
        )
    if binding is not None:
        action_calibration, calibration_overrides = prepare_action_calibration(
            get_superoffline_model_contract(args.config_name).action_calibration,
            params,
            binding_id=binding.action_calibration_id,
            binding_file_digests=binding.action_calibration_file_digests,
            binding_digest=binding.action_calibration_digest,
            snapshot_root=args.action_calibration_snapshot,
            repo_root=REPO_ROOT,
        )
        params = {**params, **calibration_overrides}
        policy_config = {**policy_config, "params": params}
    inference_index = _resolve_inference_index(dataset, args.inference_index)
    chunk_length = int(policy_config.get("chunk_length", 1))
    if chunk_length < 1:
        raise ValueError("policy.chunk_length must be positive")
    _set_seed(int(args.seed))
    policy = _build_policy(
        policy_config,
        checkpoint=checkpoint,
        device=str(args.device),
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    history_frames, history_stride, observation_interval_s = (
        _observation_history_settings(policy)
    )
    observation, history_timestamps_ns, observation_timestamp_ns = (
        dataset.policy_observation_history_at(
            inference_index,
            history_frames=history_frames,
            history_stride=history_stride,
            observation_interval_s=observation_interval_s,
            camera_devices=camera_devices,
            robot_device=robot_device,
        )
    )

    started = time.perf_counter()
    action = _predict_like_live(
        policy,
        observation,
        robot_device=robot_device,
        chunk_length=chunk_length,
    )
    inference_seconds = time.perf_counter() - started
    finite = bool(np.isfinite(action).all())
    action_step_interval_ns, action_step_interval_source = _action_step_interval_ns(
        policy
    )
    np.savez_compressed(
        output_dir / "offline_result.npz",
        action=action,
        observation_timestamp_ns=np.asarray(observation_timestamp_ns, dtype=np.int64),
        history_timestamps_ns=np.asarray(history_timestamps_ns, dtype=np.int64),
    )
    summary = {
        "schema_version": 1,
        "config_name": args.config_name,
        "policy_class": _policy_class(policy_config),
        "checkpoint": str(checkpoint.resolve()),
        "observation_dir": str(args.observation_dir.expanduser().resolve()),
        "inference_index": inference_index,
        "observation_timestamp_ns": observation_timestamp_ns,
        "history_timestamps_ns": history_timestamps_ns,
        "history_frames": history_frames,
        "history_stride": history_stride,
        "observation_interval_ns": int(round(observation_interval_s * 1e9)),
        "camera_devices": camera_devices,
        "robot_device": robot_device,
        "prompt": args.prompt,
        "seed": int(args.seed),
        "device": str(args.device),
        "num_inference_steps": int(params.get("num_inference_steps", 16)),
        "action_shape": list(action.shape),
        "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
        "action_layout": DUAL_ARM_ACTION_LAYOUT,
        "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
        "action_units": DUAL_ARM_TCP_ACTION_UNITS,
        "action_step_interval_ns": action_step_interval_ns,
        "action_step_interval_source": action_step_interval_source,
        "finite": finite,
        "inference_seconds": inference_seconds,
        "artifacts": {
            "npz": "offline_result.npz",
            "canonical_action_key": "action",
            "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
            "action_layout": DUAL_ARM_ACTION_LAYOUT,
            "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
            "action_units": DUAL_ARM_TCP_ACTION_UNITS,
            "action_step_interval_ns": action_step_interval_ns,
        },
    }
    if binding is not None:
        summary["action_calibration_id"] = action_calibration.calibration_id
        summary["action_calibration_digest"] = action_calibration.digest
        summary["observation_binding_digest"] = binding.digest
        summary["source_profile"] = binding.source_profile
        summary["source_provenance_digest"] = binding.source_provenance_digest
        summary["recording_artifact_digest"] = binding.recording_artifact_digest
        summary["artifacts"]["observation_binding_digest"] = binding.digest
        summary["artifacts"]["action_calibration_id"] = action_calibration.calibration_id
        summary["artifacts"]["action_calibration_digest"] = action_calibration.digest
        summary["artifacts"]["source_provenance_digest"] = binding.source_provenance_digest
        summary["artifacts"]["recording_artifact_digest"] = binding.recording_artifact_digest
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not finite:
        raise RuntimeError("Diffusion policy returned non-finite actions")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
