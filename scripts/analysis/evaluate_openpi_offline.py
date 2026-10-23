#!/usr/bin/env python3
"""Evaluate one OpenPi checkpoint from a recorded real-robot observation.

This is deliberately an action-only diagnostic entrypoint.  It reads a
``TimestampedObservationRecorder`` run and calls ``OpenPiUmiPolicy.predict``;
it never starts a policy connector, opens shared memory, or creates a robot
control output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.fastwam_offline_utils import (  # noqa: E402
    ObservationDataset,
    batch_child_argv,
    observation_image_preview,
    parse_inference_indices,
    write_image,
)
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


DEFAULT_CONFIG_NAME = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
SUPER_OFFLINE_PI_CONFIG_NAME = "default"
ACTION_DIM = 16
_CACHED_POLICY: tuple[tuple[str, str, str], Any] | None = None


class _FixedNoiseBackend:
    """Keep a selected OpenPi evaluation deterministic without changing policy code."""

    def __init__(self, backend: Any, noise: np.ndarray) -> None:
        self._backend = backend
        self._noise = noise

    def infer(self, inputs: Any, **kwargs: Any) -> Any:
        return self._backend.infer(inputs, noise=self._noise, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping in the selected config")
    return value


def _config_for_offline_evaluation(config_name: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Compose the same native config used by the live OpenPi consumer."""
    from incubator.control_panel.runtime_config import compose_runtime_config

    root_config = compose_runtime_config(config_name)
    policy_config = _mapping(root_config.get("policy"), field="policy")
    if policy_config.get("class") != "OpenPiUmiPolicy":
        raise ValueError(
            f"{config_name!r} configures {policy_config.get('class')!r}, not OpenPiUmiPolicy"
        )
    return root_config, policy_config


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
            "OpenPi offline evaluation requires exactly two camera devices and "
            f"one Rizon/robot device, got cameras={camera_devices}, robots={robot_devices}"
        )
    return camera_devices, robot_devices[0]


def _as_optional_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _build_policy(
    policy_config: Mapping[str, Any],
    *,
    checkpoint: Path,
    prompt: str,
    device: str,
) -> Any:
    """Load OpenPI with the original SuperOffline default-config asset override."""
    from utils.openpi_checkpoint import (
        discover_openpi_norm_stats_assets,
        validate_openpi_checkpoint_contract,
    )
    from utils.openpi_source import activate_openpi_source

    params = _mapping(policy_config.get("params"), field="policy.params")
    activate_openpi_source(
        _as_optional_path(policy_config.get("openpi_src")), repo_root=REPO_ROOT
    )
    from openpi.policies import policy_config as openpi_policy_config
    from openpi.training import checkpoints as openpi_checkpoints
    from openpi.training import config as openpi_config
    from policy.openpi_umi_policy import OpenPiUmiPolicy

    assets = discover_openpi_norm_stats_assets(checkpoint)
    asset_id = str(assets["asset_id"])

    class _SuperOfflineOpenPiUmiPolicy(OpenPiUmiPolicy):
        def load_openpi_policy(
            self, pi_config_name: str, ckpt_dir: str, device: str = "cuda"
        ) -> Any:
            contract = validate_openpi_checkpoint_contract(
                pi_config_name,
                ckpt_dir,
                norm_stats_asset_id=asset_id,
                expected_action_dim=ACTION_DIM,
                expected_state_dim=ACTION_DIM,
            )
            self.pi_config = openpi_config.get_config(pi_config_name)
            norm_stats = openpi_checkpoints.load_norm_stats(
                Path(ckpt_dir) / "assets",
                asset_id,
            )
            self.superoffline_checkpoint_contract = contract
            return openpi_policy_config.create_trained_policy(
                self.pi_config,
                ckpt_dir,
                norm_stats=norm_stats,
            )

    return _SuperOfflineOpenPiUmiPolicy(
        pi_config_name=SUPER_OFFLINE_PI_CONFIG_NAME,
        ckpt_dir=str(checkpoint),
        task=prompt,
        device=device,
        obs_horizon=int(params.get("obs_horizon", 1)),
        downsample=int(params.get("downsample", 1)),
        obs_time_interval=float(params.get("obs_time_interval", 0.04)),
        execution_fps=(
            None
            if params.get("execution_fps") is None
            else float(params.get("execution_fps"))
        ),
        enable_rtc=bool(params.get("enable_rtc", False)),
        tcp_umi_json_path=_as_optional_path(params.get("tcp_umi_json_path")),
        num_robots=int(params.get("num_robots", 2)),
        chunk_length=int(policy_config.get("chunk_length", 50)),
        mask_state=bool(params.get("mask_state", False)),
        gripper_open_max=float(params.get("gripper_open_max", 0.11)),
        left_gripper_offset=float(params.get("left_gripper_offset", 0.0)),
        right_gripper_offset=float(params.get("right_gripper_offset", 0.0)),
    )


def _load_cached_policy(
    policy_config: Mapping[str, Any],
    *,
    checkpoint: Path,
    prompt: str,
    device: str,
) -> Any:
    """Keep one OpenPi checkpoint resident for the complete observation batch."""
    global _CACHED_POLICY
    signature = (str(checkpoint.resolve()), prompt, device)
    if _CACHED_POLICY is None:
        _CACHED_POLICY = (
            signature,
            _build_policy(
                policy_config,
                checkpoint=checkpoint,
                prompt=prompt,
                device=device,
            ),
        )
    elif _CACHED_POLICY[0] != signature:
        raise RuntimeError("OpenPi batch attempted to change its policy contract")
    return _CACHED_POLICY[1]


def _predict_like_live(
    policy: Any,
    observation: Mapping[str, np.ndarray],
    *,
    robot_device: str,
    chunk_length: int,
    seed: int,
) -> np.ndarray:
    """Use the policy's deployed preprocessing and TCP action conversion."""
    backend = policy._policy
    model_config = getattr(policy.pi_config, "model", None)
    action_horizon = int(getattr(model_config, "action_horizon", chunk_length))
    model_action_dim = int(getattr(model_config, "action_dim", ACTION_DIM))
    if action_horizon < 1 or model_action_dim < 1:
        raise ValueError(
            "OpenPi checkpoint must expose a positive action horizon and model action dimension; "
            f"got horizon={action_horizon}, action_dim={model_action_dim}"
        )
    noise = np.random.default_rng(seed).standard_normal(
        (action_horizon, model_action_dim), dtype=np.float32
    )
    policy._policy = _FixedNoiseBackend(backend, noise)
    try:
        result = policy.predict(
            dict(observation),
            action_configs=[{"device_name": robot_device}],
            chunk_length=chunk_length,
            action_index=None,
        )
    finally:
        policy._policy = backend
    action = np.asarray(result[robot_device], dtype=np.float64)
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"OpenPi policy returned action {action.shape}, expected [T,16]")
    return action


def _resolve_inference_index(dataset: ObservationDataset, requested: int) -> int:
    index = requested
    if index < 0:
        index += len(dataset.inferences)
    if index < 0 or index >= len(dataset.inferences):
        raise IndexError(f"inference-index {requested} is outside the recorded run")
    return index


def _action_step_interval_ns(policy_config: Mapping[str, Any]) -> int:
    """Read the explicit deployed action-executor rate from the native config."""
    try:
        control_fps = float(policy_config["control_fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OpenPi config must define a positive policy.control_fps") from exc
    if not np.isfinite(control_fps) or control_fps <= 0:
        raise ValueError("OpenPi config must define a positive policy.control_fps")
    return int(round(1e9 / control_fps))


def _dataset_from_binding(args: argparse.Namespace) -> tuple[ObservationDataset, Any | None]:
    """Use only an executor-owned binding when source and target keys differ."""
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
    parser.add_argument("--inference-indices", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-binding", type=Path, default=None)
    parser.add_argument("--binding-index", type=int, default=0)
    parser.add_argument("--action-calibration-snapshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    if args.inference_indices is not None:
        indices = parse_inference_indices(args.inference_indices)
        batch_root = args.output_dir.expanduser().resolve()
        batch_root.mkdir(parents=True, exist_ok=False)
        summaries: list[dict[str, Any]] = []
        for inference_index in indices:
            observation_dir = batch_root / "observations" / f"{inference_index:06d}"
            main(
                batch_child_argv(
                    raw_argv,
                    inference_index=inference_index,
                    output_dir=observation_dir,
                )
            )
            summaries.append(
                json.loads((observation_dir / "summary.json").read_text(encoding="utf-8"))
            )
        (batch_root / "batch_summary.json").write_text(
            json.dumps(
                {"inference_indices": indices, "observations": summaries},
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    _root_config, policy_config = _config_for_offline_evaluation(args.config_name)
    params = _mapping(policy_config.get("params"), field="policy.params")
    action_step_interval_ns = _action_step_interval_ns(policy_config)
    # Preserve a symlink in the user input so OpenPi's checkpoint contract can
    # reject it instead of silently evaluating a different canonical target.
    checkpoint = args.checkpoint.expanduser()
    dataset, binding = _dataset_from_binding(args)
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
    camera_devices, robot_device = _selected_devices(policy_config)
    if binding is not None:
        camera_devices, robot_device = validate_binding_policy_devices(
            binding,
            camera_devices=camera_devices,
            robot_device=robot_device,
            config_name=str(args.config_name),
        )
    observation, observation_timestamp_ns = dataset.exact_policy_observation_at(
        inference_index,
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    prompt = str(args.prompt if args.prompt is not None else params.get("task") or "")
    chunk_length = int(policy_config.get("chunk_length", 50))
    policy = _load_cached_policy(
        policy_config,
        checkpoint=checkpoint,
        prompt=prompt,
        device=str(args.device),
    )

    started = time.perf_counter()
    action = _predict_like_live(
        policy,
        observation,
        robot_device=robot_device,
        chunk_length=chunk_length,
        seed=int(args.seed),
    )
    inference_seconds = time.perf_counter() - started
    finite = bool(np.isfinite(action).all())
    np.savez_compressed(
        output_dir / "offline_result.npz",
        action=action,
        observation_timestamp_ns=np.asarray(observation_timestamp_ns, dtype=np.int64),
    )
    write_image(
        output_dir / "input_grid.jpg",
        observation_image_preview(observation, camera_devices),
    )
    summary = {
        "schema_version": 1,
        "config_name": args.config_name,
        "pi_config_name": SUPER_OFFLINE_PI_CONFIG_NAME,
        "norm_stats_asset_id": getattr(
            policy, "superoffline_checkpoint_contract", {}
        ).get("asset_id"),
        "checkpoint": str(checkpoint.resolve()),
        "observation_dir": str(args.observation_dir.expanduser().resolve()),
        "inference_index": inference_index,
        "observation_timestamp_ns": observation_timestamp_ns,
        "camera_devices": camera_devices,
        "robot_device": robot_device,
        "prompt": prompt,
        "seed": int(args.seed),
        "device": str(args.device),
        "action_shape": list(action.shape),
        "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
        "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
        "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
        "action_units": DUAL_ARM_TCP_ACTION_UNITS,
        "action_step_interval_ns": action_step_interval_ns,
        "finite": finite,
        "inference_seconds": inference_seconds,
        "artifacts": {
            "npz": "offline_result.npz",
            "canonical_action_key": "action",
            "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
            "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
            "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
            "action_units": DUAL_ARM_TCP_ACTION_UNITS,
            "action_step_interval_ns": action_step_interval_ns,
            "input_preview": "input_grid.jpg",
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
        raise RuntimeError("OpenPi policy returned non-finite actions")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
