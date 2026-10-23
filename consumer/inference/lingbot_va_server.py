#!/usr/bin/env python3
"""
Socket server for LingBot-VA inference.

The server keeps the heavy LingBot-VA model and KV cache in one process while
SuperInference talks to it through a lightweight websocket policy client.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import http
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames
import yaml
import imageio.v2 as imageio
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_config import logger
from utils.msgpack_numpy import Packer, unpackb

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NMX_VLA_ROOT = _REPO_ROOT / "third_party" / "nmx_vla"
if str(_NMX_VLA_ROOT) not in sys.path:
    sys.path.append(str(_NMX_VLA_ROOT))

from nmx_vla.model.modules.lingbot_va import (  # noqa: E402
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from nmx_vla.model.modules.lingbot_va.loaders import WanVAEStreamingWrapper  # noqa: E402
from nmx_vla.model.modules.lingbot_va.utils import (  # noqa: E402
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
)

VideoLatents = List[torch.Tensor]


def _make_run_timestamp() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_pid{os.getpid()}"


def _parse_dtype(value: Any) -> torch.dtype:
    """Parse dtype from config string or torch dtype."""
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[normalized]


def _dict_to_namespace(data: Dict[str, Any]) -> SimpleNamespace:
    """Convert a dictionary to a SimpleNamespace recursively."""
    converted: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "norm_stat":
            converted[key] = value
        elif isinstance(value, dict):
            converted[key] = _dict_to_namespace(value)
        elif isinstance(value, list):
            converted[key] = [
                _dict_to_namespace(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            converted[key] = value
    return SimpleNamespace(**converted)


def _pad_latent_to_patch_size(
    latent: torch.Tensor,
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    _, _, frames, height, width = latent.shape
    patch_f, patch_h, patch_w = (int(v) for v in patch_size)
    pad_frames = (-int(frames)) % patch_f
    pad_height = (-int(height)) % patch_h
    pad_width = (-int(width)) % patch_w
    if not (pad_frames or pad_height or pad_width):
        return latent
    return F.pad(latent, (0, pad_width, 0, pad_height, 0, pad_frames))


def _video_latents_frame_count(latents: Union[torch.Tensor, VideoLatents]) -> int:
    if isinstance(latents, torch.Tensor):
        return int(latents.shape[2])
    if not latents:
        raise ValueError("video latent list must be non-empty")
    frame_count = int(latents[0].shape[2])
    for latent in latents[1:]:
        if int(latent.shape[2]) != frame_count:
            raise ValueError("All video latent views must share frame count")
    return frame_count


def _step_video_latents(
    scheduler: FlowMatchScheduler,
    model_output: VideoLatents,
    timestep: torch.Tensor | float,
    sample: VideoLatents,
) -> VideoLatents:
    if len(model_output) != len(sample):
        raise ValueError(
            "Video latent prediction views and sample views must have the same length: "
            f"{len(model_output)} != {len(sample)}"
        )
    return [
        scheduler.step(output, timestep, latent)
        for output, latent in zip(model_output, sample, strict=True)
    ]


def _build_prefix_weights(
    *,
    total_steps: int,
    inference_delay: int,
    prefix_attention_horizon: int,
    schedule: str = "exp",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build per-action-step weights for RTC prefix inpainting."""
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    start = max(int(inference_delay), 0)
    end = max(start + int(prefix_attention_horizon), 0)
    start = min(start, total_steps)
    end = min(end, total_steps)
    start = min(start, end)

    compute_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    step = torch.arange(total_steps, device=device, dtype=compute_dtype)
    if schedule == "ones":
        weights = torch.ones(total_steps, device=device, dtype=compute_dtype)
    elif schedule == "zeros":
        weights = (step < start).to(compute_dtype)
    elif schedule in {"linear", "exp"}:
        denom = max(end - start + 1, 1)
        weights = torch.clamp((float(start - 1) - step) / float(denom) + 1.0, 0.0, 1.0)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / (torch.e - 1.0)
    else:
        raise ValueError(f"Invalid RTC prefix schedule: {schedule!r}")

    weights = torch.where(step >= end, torch.zeros_like(weights), weights)
    return weights.to(dtype=dtype)


def action_prefix_weight_tensor(
    action: torch.Tensor,
    *,
    inference_delay: int,
    prefix_attention_horizon: int,
    schedule: str = "exp",
) -> torch.Tensor:
    """Return weights broadcastable to LingBot action tensors [B,C,F,H,1]."""
    if action.ndim != 5:
        raise ValueError(f"Expected action tensor [B,C,F,H,1], got shape {tuple(action.shape)}")
    frame_count = int(action.shape[2])
    steps_per_frame = int(action.shape[3])
    weights = _build_prefix_weights(
        total_steps=frame_count * steps_per_frame,
        inference_delay=inference_delay,
        prefix_attention_horizon=prefix_attention_horizon,
        schedule=schedule,
        device=action.device,
        dtype=action.dtype,
    )
    return weights.reshape(1, 1, frame_count, steps_per_frame, 1)


def _noisy_prefix_at_step(
    *,
    prefix_action: torch.Tensor,
    initial_noise: torch.Tensor,
    scheduler: FlowMatchScheduler,
    next_step_index: int,
) -> torch.Tensor:
    """Project a clean prefix action onto the scheduler's current noise level."""
    next_step_index = int(next_step_index)
    if next_step_index >= len(scheduler.sigmas):
        sigma = torch.zeros((), device=prefix_action.device, dtype=prefix_action.dtype)
    else:
        sigma = scheduler.sigmas[next_step_index].to(
            device=prefix_action.device,
            dtype=prefix_action.dtype,
        )
    return (1.0 - sigma) * prefix_action + sigma * initial_noise


def apply_prefix_inpainting(
    *,
    action: torch.Tensor,
    prefix_action: torch.Tensor,
    initial_noise: torch.Tensor,
    scheduler: FlowMatchScheduler,
    next_step_index: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Blend the current action sample with a noised RTC prefix."""
    if action.shape != prefix_action.shape:
        raise ValueError(
            "RTC action/prefix shapes must match: "
            f"action={tuple(action.shape)} prefix={tuple(prefix_action.shape)}"
        )
    prefix_noisy = _noisy_prefix_at_step(
        prefix_action=prefix_action,
        initial_noise=initial_noise,
        scheduler=scheduler,
        next_step_index=next_step_index,
    )
    return weights * prefix_noisy + (1.0 - weights) * action


def _build_inverse_used_action_channel_ids(
    used_action_channel_ids: list[int],
    action_dim: int,
) -> list[int]:
    fill_idx = len(used_action_channel_ids)
    inverse = [fill_idx] * int(action_dim)
    for source_idx, dest_idx in enumerate(used_action_channel_ids):
        inverse[int(dest_idx)] = source_idx
    return inverse


def _apply_inference_action_dropout(runtime_data: Dict[str, Any]) -> None:
    """Mirror training-time view/action dropout in the runtime action layout."""
    obs_cam_keys = set(runtime_data.get("obs_cam_keys") or [])
    used_ids = list(runtime_data.get("used_action_channel_ids") or [])
    if not obs_cam_keys or not used_ids:
        return

    drop_slices: list[tuple[int, int, str, list[int]]] = []
    for entry in runtime_data.get("per_dataset_head_view_action_dropout") or []:
        if not bool(entry.get("enabled", False)):
            continue
        view_key = entry.get("view_key")
        if not view_key or view_key in obs_cam_keys:
            continue
        action_slice = entry.get("action_channel_slice") or []
        if len(action_slice) != 2:
            continue
        start, end = int(action_slice[0]), int(action_slice[1])
        if start < 0 or end <= start:
            raise ValueError(
                "per_dataset_head_view_action_dropout.action_channel_slice must be "
                f"an increasing [start, end] pair, got [{start}, {end}]"
            )
        if end > len(used_ids):
            raise ValueError(
                "per_dataset_head_view_action_dropout.action_channel_slice is "
                "interpreted in raw/source action-channel space and exceeds "
                f"used_action_channel_ids: end={end}, used_action_channel_ids={len(used_ids)}"
            )
        canonical_channels = list(dict.fromkeys(int(channel) for channel in used_ids[start:end]))
        if not canonical_channels:
            raise ValueError(
                "per_dataset_head_view_action_dropout.action_channel_slice mapped "
                "to no canonical action channels"
            )
        action_dim = int(runtime_data.get("action_dim", max(used_ids) + 1))
        if any(channel < 0 or channel >= action_dim for channel in canonical_channels):
            raise ValueError(
                "per_dataset_head_view_action_dropout.action_channel_slice maps to "
                f"canonical channels outside action_dim={action_dim}: {canonical_channels}"
            )
        drop_slices.append((start, end, str(view_key), canonical_channels))

    if not drop_slices:
        return

    raw_keep_mask = [True] * len(used_ids)
    for start, end, _view_key, _canonical_channels in drop_slices:
        for raw_idx in range(start, end):
            raw_keep_mask[raw_idx] = False

    filtered_used_ids = [channel for keep, channel in zip(raw_keep_mask, used_ids) if keep]
    if len(filtered_used_ids) == len(used_ids):
        return

    runtime_data["used_action_channel_ids"] = filtered_used_ids
    action_dim = int(runtime_data.get("action_dim", max(used_ids) + 1))
    runtime_data["inverse_used_action_channel_ids"] = _build_inverse_used_action_channel_ids(
        filtered_used_ids,
        action_dim,
    )
    logger.warning(
        "Using inference action layout after view/action dropout: "
        f"raw_dim {len(used_ids)} -> {len(filtered_used_ids)}, "
        "dropped="
        f"{[(start, end, view_key, canonical_channels) for start, end, view_key, canonical_channels in drop_slices]}"
    )


def _namespace_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, SimpleNamespace):
        return vars(value)
    return dict(value)


def _normalize_learnable_action_embedding_config(value: Any) -> Dict[str, Any]:
    config = dict(_namespace_to_dict(value))
    normalized = dict(config)
    normalized["enabled"] = bool(config.get("enabled", False))
    normalized["enable_unused_canonical_dims"] = bool(
        config.get("enable_unused_canonical_dims", False)
    )
    for key in (
        "dataset_used_canonical_dims",
        "gripper_canonical_dims",
        "gripper_raw_dims",
    ):
        normalized[key] = [int(channel) for channel in config.get(key) or []]
    return normalized


def _learnable_action_embedding_enabled(value: Any) -> bool:
    config = _namespace_to_dict(value)
    enabled = config.get("enabled")
    if enabled is not None:
        return bool(enabled)
    return any(
        bool(config.get(key))
        for key in (
            "gripper_canonical_dims",
            "gripper_raw_dims",
            "enable_unused_canonical_dims",
        )
    )


def _resolve_zero_action_condition_uses_missing_embedding(
    runtime_data: Dict[str, Any],
) -> bool:
    configured = runtime_data.get("zero_action_condition_uses_missing_embedding")
    if configured is not None:
        return bool(configured)

    dataset_groups = runtime_data.get("dataset_groups") or []
    if dataset_groups:
        for group in dataset_groups:
            group_config = _namespace_to_dict(_namespace_to_dict(group).get("config"))
            learnable_config = group_config.get("learnable_action_embedding_config")
            if learnable_config is None:
                learnable_config = group_config.get("learnable_action_embedding")
            if _learnable_action_embedding_enabled(learnable_config):
                return True
        return False

    return _learnable_action_embedding_enabled(
        runtime_data.get("learnable_action_embedding_config")
    )


def _select_latest_checkpoint(run_dir: Path) -> Path:
    pattern = re.compile(r"checkpoint_step_(\d+)$")
    candidates: list[Tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        transformer_dir = child / "transformer"
        if transformer_dir.is_dir():
            candidates.append((int(match.group(1)), transformer_dir))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_step_*/transformer found in {run_dir}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_checkpoint_paths(
    ckpt_path: str,
    runtime_config_path: Optional[str] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve ckpt input to transformer dir, run dir, and runtime config path."""
    path = Path(ckpt_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LingBot-VA checkpoint path does not exist: {path}")

    transformer_path: Path
    run_dir: Optional[Path] = None
    if path.is_dir() and (path / "config.json").is_file():
        transformer_path = path
        if path.name == "transformer" and path.parent.name.startswith("checkpoint_step_"):
            run_dir = path.parent.parent
    elif path.is_dir() and path.name.startswith("checkpoint_step_"):
        transformer_path = path / "transformer"
        run_dir = path.parent
    elif path.is_dir():
        transformer_path = _select_latest_checkpoint(path)
        run_dir = path
    else:
        raise ValueError(f"ckpt_path must be a directory, got: {path}")

    if not transformer_path.is_dir():
        raise FileNotFoundError(f"Transformer checkpoint directory not found: {transformer_path}")

    if runtime_config_path:
        resolved_runtime = Path(runtime_config_path).expanduser().resolve()
    elif run_dir is not None and (run_dir / "lingbot_runtime_config.json").is_file():
        resolved_runtime = run_dir / "lingbot_runtime_config.json"
    else:
        resolved_runtime = None

    return transformer_path, run_dir, resolved_runtime


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


_MODEL_CONTRACT_KEYS = ("model_structure", "action_condition_mode")


def _load_model_contract_from_artifact(
    runtime_data: Dict[str, Any],
    transformer_path: Path,
) -> Dict[str, Any]:
    """Load model contract fields that inference config must not override."""
    contract = {
        key: runtime_data[key]
        for key in _MODEL_CONTRACT_KEYS
        if runtime_data.get(key) is not None
    }
    transformer_config_path = transformer_path / "config.json"
    transformer_config: Dict[str, Any] = {}
    if transformer_config_path.is_file():
        transformer_config = _load_json(transformer_config_path)

    for key in _MODEL_CONTRACT_KEYS:
        if contract.get(key) is None and transformer_config.get(key) is not None:
            contract[key] = transformer_config[key]

    if contract.get("model_structure") is None and transformer_config_path.is_file():
        contract["model_structure"] = "shared"
    return contract


def _resolve_run_artifact_path(
    artifact_name: str,
    explicit_path: Optional[str],
    run_dir: Optional[Path],
    transformer_path: Path,
) -> Tuple[Optional[Path], list[Path]]:
    """Resolve a checkpoint/run artifact from explicit and conventional paths."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(str(explicit_path)).expanduser())
    if run_dir is not None:
        candidates.append(run_dir / artifact_name)

    checkpoint_dir = transformer_path.parent if transformer_path.name == "transformer" else None
    if checkpoint_dir is not None:
        candidates.append(checkpoint_dir / artifact_name)
        candidates.append(checkpoint_dir.parent / artifact_name)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate, unique_candidates
    return None, unique_candidates


def _resolve_norm_stats_manifest_path(
    runtime_data: Dict[str, Any],
    run_dir: Optional[Path],
    transformer_path: Path,
) -> Tuple[Optional[Path], list[Path]]:
    """Resolve the latest LingBot per-dataset action norm-stats manifest."""
    return _resolve_run_artifact_path(
        "lingbot_action_norm_stats_manifest.json",
        runtime_data.get("norm_stats_manifest_path"),
        run_dir,
        transformer_path,
    )


def _select_norm_stat_from_manifest(
    manifest: Dict[str, Any],
    runtime_data: Dict[str, Any],
    manifest_path: Path,
) -> Dict[str, Any]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(
            "LingBot action norm-stats manifest must contain a non-empty "
            f"`datasets` list: {manifest_path}"
        )

    requested_dataset_id = runtime_data.get("norm_stats_dataset_id")
    requested_dataset_name = runtime_data.get("norm_stats_dataset_name")
    if requested_dataset_id is None and requested_dataset_name is None:
        if len(datasets) != 1:
            raise ValueError(
                "LingBot action norm-stats manifest contains multiple datasets. "
                "Set `policy.params.norm_stats_dataset_id` or "
                "`policy.params.norm_stats_dataset_name`."
            )
        selected = datasets[0]
    else:
        selected = None
        for record in datasets:
            if not isinstance(record, dict):
                continue
            if (
                requested_dataset_id is not None
                and str(record.get("dataset_id")) == str(requested_dataset_id)
            ):
                selected = record
                break
            if (
                requested_dataset_name is not None
                and str(record.get("dataset_name")) == str(requested_dataset_name)
            ):
                selected = record
                break
        if selected is None:
            raise ValueError(
                "No matching dataset in LingBot action norm-stats manifest for "
                f"dataset_id={requested_dataset_id!r}, "
                f"dataset_name={requested_dataset_name!r}: {manifest_path}"
            )

    if not isinstance(selected, dict):
        raise ValueError(
            "LingBot action norm-stats manifest dataset entries must be mappings: "
            f"{manifest_path}"
        )
    q01 = selected.get("q01")
    q99 = selected.get("q99")
    if not isinstance(q01, list) or not isinstance(q99, list) or len(q01) != len(q99) or not q01:
        raise ValueError(
            "Selected LingBot action norm-stats dataset must contain non-empty "
            f"matching q01/q99 lists: {manifest_path}"
        )
    norm_stat = {
        "q01": q01,
        "q99": q99,
    }
    meta = selected.get("_meta")
    if isinstance(meta, dict):
        norm_stat["_meta"] = meta
    return norm_stat


def _resolve_temporal_metadata_path(
    runtime_data: Dict[str, Any],
    run_dir: Optional[Path],
    transformer_path: Path,
) -> Tuple[Optional[Path], list[Path]]:
    """Resolve LingBot temporal metadata emitted by newer nmx_vla training."""
    return _resolve_run_artifact_path(
        "lingbot_temporal_metadata.json",
        runtime_data.get("temporal_metadata_path"),
        run_dir,
        transformer_path,
    )


_TEMPORAL_METADATA_FIELD_KEYS = (
    "original_video_fps",
    "sampled_video_fps",
    "latent_fps",
    "frame_stride",
    "sampled_video_frames_per_latent",
    "original_video_frames_per_latent",
    "actions_per_frame",
    "actions_per_latent_frame",
)


def _has_temporal_metadata_fields(metadata: Dict[str, Any]) -> bool:
    return any(key in metadata for key in _TEMPORAL_METADATA_FIELD_KEYS)


def _append_unique_string(candidates: List[str], value: Any) -> None:
    if value is None:
        return
    candidate = str(value)
    if candidate and candidate not in candidates:
        candidates.append(candidate)


def _temporal_metadata_dataset_candidates(runtime_data: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    _append_unique_string(candidates, runtime_data.get("norm_stats_dataset_name"))

    dataset_groups = runtime_data.get("dataset_groups")
    if isinstance(dataset_groups, list):
        for group in dataset_groups:
            if not isinstance(group, dict):
                continue
            _append_unique_string(candidates, group.get("dataset_name"))
            group_config = group.get("config")
            if isinstance(group_config, dict):
                _append_unique_string(candidates, group_config.get("dataset_name"))

    _append_unique_string(candidates, runtime_data.get("dataset_name"))

    norm_stat = runtime_data.get("norm_stat")
    if isinstance(norm_stat, dict):
        norm_meta = norm_stat.get("_meta")
        if isinstance(norm_meta, dict):
            _append_unique_string(candidates, norm_meta.get("dataset_name"))

    return candidates


def _select_temporal_metadata(
    temporal_metadata: Dict[str, Any],
    runtime_data: Dict[str, Any],
    temporal_metadata_path: Path,
) -> Dict[str, Any]:
    """Select a concrete temporal metadata payload from flat or dataset-keyed JSON."""
    if _has_temporal_metadata_fields(temporal_metadata):
        return temporal_metadata

    dataset_candidates = _temporal_metadata_dataset_candidates(runtime_data)
    for dataset_name in dataset_candidates:
        selected_metadata = temporal_metadata.get(dataset_name)
        if isinstance(selected_metadata, dict) and _has_temporal_metadata_fields(
            selected_metadata
        ):
            logger.info(
                "Selected LingBot temporal metadata dataset "
                f"{dataset_name!r} from: {temporal_metadata_path}"
            )
            return selected_metadata

    if len(temporal_metadata) == 1:
        dataset_name, selected_metadata = next(iter(temporal_metadata.items()))
        if isinstance(selected_metadata, dict) and _has_temporal_metadata_fields(
            selected_metadata
        ):
            logger.info(
                "Selected the only LingBot temporal metadata dataset "
                f"{dataset_name!r} from: {temporal_metadata_path}"
            )
            return selected_metadata

    available_datasets = sorted(str(key) for key in temporal_metadata.keys())
    raise ValueError(
        "LingBot temporal metadata is keyed by dataset name, but no matching "
        "dataset could be selected. Set `policy.params.norm_stats_dataset_name` "
        "or ensure checkpoint `dataset_groups` matches the metadata keys. "
        f"Candidates: {dataset_candidates}; available: {available_datasets}; "
        f"path: {temporal_metadata_path}"
    )


def _load_policy_params_from_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
            cfg = compose(config_name=path.stem)
        policy_cfg = OmegaConf.to_container(
            OmegaConf.select(cfg, "policy"),
            resolve=True,
        )
        policy_cfg = policy_cfg or {}
    except Exception as exc:
        logger.debug(
            "Falling back to raw YAML config load for %s after Hydra compose failed: %s",
            path,
            exc,
        )
        with open(path, "r", encoding="utf-8") as file:
            root_cfg = yaml.safe_load(file)
        policy_cfg = root_cfg.get("policy", {}) if isinstance(root_cfg, dict) else {}
    params = dict(policy_cfg.get("params", {}) or {})
    server_cfg = dict(params.pop("server", {}) or {})
    params.update(server_cfg)
    return params


def build_job_config(args: argparse.Namespace) -> SimpleNamespace:
    """Build server runtime config from SuperInference config and CLI overrides."""
    params: Dict[str, Any] = {}
    if args.config:
        params.update(_load_policy_params_from_config(args.config))

    cli_overrides = {
        "ckpt_path": args.ckpt_path,
        "base_model_path": args.base_model_path,
        "runtime_config_path": args.runtime_config_path,
        "host": args.host,
        "port": args.port,
        "device": args.device,
        "action_condition_mode": args.action_condition_mode,
        "zero_action_condition": getattr(args, "zero_action_condition", None),
        "attn_mode": args.attn_mode,
        "model_structure": args.model_structure,
        "save_root": args.save_root,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            params[key] = value

    ckpt_path = params.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("LingBot-VA server requires `ckpt_path`")

    transformer_path, run_dir, runtime_config_path = _resolve_checkpoint_paths(
        str(ckpt_path),
        params.get("runtime_config_path"),
    )

    artifact_runtime_data: Dict[str, Any] = {}
    if runtime_config_path is not None:
        artifact_runtime_data.update(_load_json(runtime_config_path))
    model_contract = _load_model_contract_from_artifact(
        runtime_data=artifact_runtime_data,
        transformer_path=transformer_path,
    )

    runtime_data = dict(artifact_runtime_data)
    runtime_data.update(params)
    for key, artifact_value in model_contract.items():
        inference_value = runtime_data.get(key)
        if inference_value is not None and str(inference_value) != str(artifact_value):
            logger.warning(
                f"Ignoring inference override for {key}: "
                f"{inference_value!r} -> {artifact_value!r}"
            )
        runtime_data[key] = artifact_value
    runtime_data["transformer_path"] = str(transformer_path)
    runtime_data["run_dir"] = str(run_dir) if run_dir is not None else None
    runtime_data["runtime_config_path"] = (
        str(runtime_config_path) if runtime_config_path is not None else None
    )

    if runtime_data.get("base_model_path"):
        runtime_data["wan22_pretrained_model_name_or_path"] = runtime_data["base_model_path"]
    if not runtime_data.get("wan22_pretrained_model_name_or_path"):
        raise ValueError(
            "`base_model_path` or runtime `wan22_pretrained_model_name_or_path` is required"
        )

    runtime_data["norm_stat"] = None
    candidate, searched_norm_stats_manifest_paths = _resolve_norm_stats_manifest_path(
        runtime_data=runtime_data,
        run_dir=run_dir,
        transformer_path=transformer_path,
    )
    runtime_data["_searched_norm_stats_manifest_paths"] = [
        str(path) for path in searched_norm_stats_manifest_paths
    ]
    if candidate is not None:
        manifest = _load_json(candidate)
        runtime_data["norm_stat"] = _select_norm_stat_from_manifest(
            manifest,
            runtime_data,
            candidate,
        )
        runtime_data["norm_stats_manifest_path"] = str(candidate)
        logger.info(f"Loaded LingBot action norm stats manifest from: {candidate}")

    temporal_candidate, searched_temporal_metadata_paths = _resolve_temporal_metadata_path(
        runtime_data=runtime_data,
        run_dir=run_dir,
        transformer_path=transformer_path,
    )
    runtime_data["_searched_temporal_metadata_paths"] = [
        str(path) for path in searched_temporal_metadata_paths
    ]
    if temporal_candidate is not None:
        temporal_metadata = _load_json(temporal_candidate)
        runtime_data["temporal_metadata_path"] = str(temporal_candidate)
        selected_temporal_metadata = _select_temporal_metadata(
            temporal_metadata,
            runtime_data,
            temporal_candidate,
        )
        for key, value in selected_temporal_metadata.items():
            if key in runtime_data and runtime_data[key] != value:
                logger.warning(
                    "LingBot temporal metadata mismatch for "
                    f"{key}: config={runtime_data[key]!r}, metadata={value!r}. "
                    "Using metadata value."
                )
            runtime_data[key] = value
        logger.info(f"Loaded LingBot temporal metadata from: {temporal_candidate}")
    if "source_video_frame_stride" not in runtime_data and "frame_stride" in runtime_data:
        runtime_data["source_video_frame_stride"] = runtime_data["frame_stride"]
    inference_obs_cam_keys = params.get("obs_cam_keys") or params.get("model_camera_keys")
    if inference_obs_cam_keys:
        runtime_obs_cam_keys = list(runtime_data.get("obs_cam_keys") or [])
        runtime_data["obs_cam_keys"] = list(inference_obs_cam_keys)
        if runtime_obs_cam_keys and runtime_obs_cam_keys != runtime_data["obs_cam_keys"]:
            logger.warning(
                "Using inference obs_cam_keys from config instead of runtime "
                f"checkpoint metadata: {runtime_obs_cam_keys!r} -> "
                f"{runtime_data['obs_cam_keys']!r}"
            )
    _apply_inference_action_dropout(runtime_data)

    defaults: Dict[str, Any] = {
        "host": "0.0.0.0",
        "port": 18080,
        "device": "cuda",
        "local_rank": 0,
        "param_dtype": "bf16",
        "enable_offload": True,
        "offload_vae": None,
        "offload_text_encoder": None,
        "attn_mode": "flashattn",
        "model_structure": "mot",
        "action_condition_mode": "fastwam",
        "mot_config": {
            "action_hidden_dim": 768,
            "action_ffn_dim": None,
            "action_mlp_hidden_dim": 256,
            "init_mode": "video_interp_alpha",
            "alpha_scale": None,
        },
        "patch_size": [1, 2, 2],
        "snr_shift": 5.0,
        "action_snr_shift": 1.0,
        "height": 256,
        "width": 256,
        "obs_cam_keys": [],
        "env_type": "none",
        "action_dim": 30,
        "actions_per_frame": 1,
        "action_per_frame": None,
        "actions_per_latent_frame": None,
        "sampled_video_frames_per_latent": None,
        "sampled_video_fps": None,
        "video_frames_per_latent": None,
        "source_video_frame_stride": 1,
        "frame_chunk_size": 4,
        "attn_window": 30,
        "guidance_scale": 1.0,
        "action_guidance_scale": 1.0,
        "num_inference_steps": 1,
        "action_num_inference_steps": 10,
        "video_exec_step": -1,
        "used_action_channel_ids": [],
        "inverse_used_action_channel_ids": [],
        "action_norm_method": "quantiles",
        "save_root": "/tmp/lingbot_va_superinference",
        "save_debug_tensors": False,
        "save_predicted_video_latents": False,
        "save_history_cache_latents": False,
        "zero_action_condition": False,
        "zero_action_condition_uses_missing_embedding": None,
        "learnable_action_embedding_config": {
            "enabled": False,
            "enable_unused_canonical_dims": False,
            "dataset_used_canonical_dims": [],
            "gripper_canonical_dims": [],
            "gripper_raw_dims": [],
        },
        "decoded_latent_save_dir": None,
        "absolute_action_quaternion_order": "auto",
        "reshape_mode": "resize",
        "center_crop_before_resize": False,
        "crop_before_resize": None,
        "crop_resize_size": None,
        "pad_after_resize": None,
    }
    merged = {**defaults, **runtime_data}
    resize_size = merged.get("crop_resize_size")
    if (
        str(merged.get("reshape_mode", "resize")).lower() in {"resize", "crop_resize"}
        and resize_size is not None
    ):
        if isinstance(resize_size, (list, tuple)) and len(resize_size) == 2:
            resize_height, resize_width = int(resize_size[0]), int(resize_size[1])
        else:
            resize_height = resize_width = int(resize_size)
        if int(merged.get("height", resize_height)) != resize_height or int(
            merged.get("width", resize_width)
        ) != resize_width:
            logger.warning(
                "LingBot-VA overriding runtime height/width with crop_resize_size: "
                f"({merged.get('height')}, {merged.get('width')}) -> "
                f"({resize_height}, {resize_width})"
            )
        merged["height"] = resize_height
        merged["width"] = resize_width
    merged["zero_action_condition"] = bool(merged.get("zero_action_condition", False))
    merged["learnable_action_embedding_config"] = _normalize_learnable_action_embedding_config(
        merged.get("learnable_action_embedding_config")
    )
    merged["zero_action_condition_uses_missing_embedding"] = (
        _resolve_zero_action_condition_uses_missing_embedding(merged)
    )
    merged["param_dtype"] = _parse_dtype(merged["param_dtype"])
    merged["patch_size"] = tuple(merged["patch_size"])
    if merged.get("action_per_frame") is None:
        merged["action_per_frame"] = merged.get("actions_per_frame", 1)
    if merged.get("actions_per_frame") is None:
        merged["actions_per_frame"] = merged.get("action_per_frame", 1)
    if not merged["used_action_channel_ids"]:
        raise ValueError("Runtime config missing used_action_channel_ids")
    if not merged["inverse_used_action_channel_ids"]:
        merged["inverse_used_action_channel_ids"] = _build_inverse_used_action_channel_ids(
            list(merged["used_action_channel_ids"]),
            int(merged["action_dim"]),
        )
    if merged.get("norm_stat") is None:
        searched = merged.get("_searched_norm_stats_manifest_paths", [])
        raise ValueError(
            "LingBot-VA runtime config missing norm_stat. "
            "Place `lingbot_action_norm_stats_manifest.json` in the checkpoint "
            "run dir or set `policy.params.norm_stats_manifest_path`. "
            f"Searched: {searched}"
        )
    if not merged["obs_cam_keys"]:
        model_camera_keys = params.get("model_camera_keys") or params.get("camera_devices") or []
        merged["obs_cam_keys"] = list(model_camera_keys)

    return _dict_to_namespace(merged)


class LingBotVAServer:
    """LingBot-VA inference runtime with KV-cache support."""

    def __init__(self, job_config: SimpleNamespace) -> None:
        self.cache_name = "pos"
        self.job_config = job_config
        self.save_root = str(job_config.save_root)
        self.dtype = job_config.param_dtype
        requested_device = str(getattr(job_config, "device", "cuda"))
        if requested_device.startswith("cuda") and torch.cuda.is_available():
            if ":" in requested_device:
                self.device = torch.device(requested_device)
            else:
                self.device = torch.device(f"cuda:{int(getattr(job_config, 'local_rank', 0))}")
        else:
            self.device = torch.device("cpu")
        self.enable_offload = bool(getattr(job_config, "enable_offload", True))
        offload_vae = getattr(job_config, "offload_vae", None)
        offload_text_encoder = getattr(job_config, "offload_text_encoder", None)
        self.offload_vae = self.enable_offload if offload_vae is None else bool(offload_vae)
        self.offload_text_encoder = (
            self.enable_offload
            if offload_text_encoder is None
            else bool(offload_text_encoder)
        )
        self.save_debug_tensors = bool(getattr(job_config, "save_debug_tensors", False))
        self.save_predicted_video_latents = bool(
            getattr(job_config, "save_predicted_video_latents", False)
        )
        self.save_history_cache_latents = bool(
            getattr(job_config, "save_history_cache_latents", False)
        )
        self.zero_action_condition = bool(
            getattr(job_config, "zero_action_condition", False)
        )
        self.zero_action_condition_uses_missing_embedding = bool(
            getattr(job_config, "zero_action_condition_uses_missing_embedding", False)
        )
        self.learnable_action_embedding_config = _normalize_learnable_action_embedding_config(
            getattr(job_config, "learnable_action_embedding_config", None)
        )
        decoded_save_dir = getattr(job_config, "decoded_latent_save_dir", None)
        if decoded_save_dir is None:
            decoded_save_dir = os.path.join(self.save_root, "decoded_latents")
        decoded_latent_base_dir = Path(str(decoded_save_dir)).expanduser()
        self.decoded_latent_save_dir = decoded_latent_base_dir / _make_run_timestamp()
        self.decoded_latent_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LingBot-VA decoded latent debug save dir: {self.decoded_latent_save_dir}")
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._decoded_predicted_video_index = 0
        self._decoded_history_cache_index = 0
        self._history_prediction_comparison_index = 0
        self._last_predicted_video_np: Optional[np.ndarray] = None
        self.comparison_grid_save_dir = self.decoded_latent_save_dir / "comparison_grid"
        self.comparison_grid_save_dir.mkdir(parents=True, exist_ok=True)
        if self.zero_action_condition:
            logger.info(
                "LingBot-VA zero_action_condition enabled: action history "
                "uses zero values; learned all-missing embedding="
                f"{self.zero_action_condition_uses_missing_embedding}"
            )
        logger.info(
            "LingBot-VA learnable action embedding config: "
            f"{self.learnable_action_embedding_config}"
        )

        self.scheduler = FlowMatchScheduler(
            shift=float(job_config.snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=float(job_config.action_snr_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        base_model_path = str(job_config.wan22_pretrained_model_name_or_path)
        self.vae = load_vae(
            base_model_path,
            torch_dtype=self.dtype,
            torch_device="cpu" if self.offload_vae else self.device,
            component_name="vae",
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tokenizer = load_tokenizer(base_model_path, component_name="tokenizer")
        self.text_encoder = load_text_encoder(
            base_model_path,
            torch_dtype=self.dtype,
            torch_device="cpu" if self.offload_text_encoder else self.device,
            component_name="text_encoder",
        )
        self.transformer = load_transformer(
            str(job_config.transformer_path),
            torch_dtype=self.dtype,
            torch_device=self.device,
            component_name="transformer",
            attn_mode=str(job_config.attn_mode),
            action_condition_mode=str(job_config.action_condition_mode),
            model_structure=str(job_config.model_structure),
            mot_config=_namespace_to_dict(getattr(job_config, "mot_config", None)),
            learnable_action_embedding_config=self.learnable_action_embedding_config,
        )
        self.transformer.eval().requires_grad_(False)
        logger.info(
            "LingBot-VA server loaded "
            f"transformer={job_config.transformer_path} "
            f"base_model={base_model_path} "
            f"mode={job_config.action_condition_mode} "
            f"structure={job_config.model_structure}"
        )
        logger.info(
            "LingBot-VA offload config: "
            f"vae={'cpu' if self.offload_vae else self.device}, "
            f"text_encoder={'cpu' if self.offload_text_encoder else self.device}, "
            f"transformer={self.device}"
        )

        self.env_type = str(getattr(job_config, "env_type", "none"))
        self.streaming_vae_half = None
        self.frame_st_id = 0
        self.init_latent: Optional[Union[torch.Tensor, VideoLatents]] = None
        self.last_known_video_latent: Optional[Union[torch.Tensor, VideoLatents]] = None
        self.prompt_embeds: Optional[torch.Tensor] = None
        self.negative_prompt_embeds: Optional[torch.Tensor] = None
        self.use_cfg = False
        self.action_mask: Optional[torch.Tensor] = None
        self.actions_q01: Optional[torch.Tensor] = None
        self.actions_q99: Optional[torch.Tensor] = None
        self.action_norm_method = str(getattr(job_config, "action_norm_method", "quantiles"))
        self._reset(prompt=getattr(job_config, "task", None))

    def _mark_action_condition_all_missing(self, action_input_dict: Dict[str, Any]) -> None:
        if self.zero_action_condition and self.zero_action_condition_uses_missing_embedding:
            action_input_dict["action_condition_all_missing"] = True

    def _get_t5_prompt_embeds(
        self,
        prompt: Any = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = device or self.device
        dtype = dtype or self.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(text) for text in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        mask = text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(
            text_input_ids.to(text_encoder_device),
            mask.to(text_encoder_device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [embed[:seq_len] for embed, seq_len in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([embed, embed.new_zeros(max_sequence_length - embed.size(0), embed.size(1))])
                for embed in prompt_embeds
            ],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        return prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1).to(device)

    def encode_prompt(
        self,
        prompt: Any,
        negative_prompt: Optional[Any] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = device or self.device
        dtype = dtype or self.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        return ((latents.float() - latents_mean) * latents_std).to(latents)

    def _resolve_video_frames_per_latent(self) -> int:
        explicit = getattr(self.job_config, "video_frames_per_latent", None)
        if explicit is not None:
            return max(int(explicit), 1)
        sampled_explicit = getattr(self.job_config, "sampled_video_frames_per_latent", None)
        if sampled_explicit is not None:
            return max(int(sampled_explicit), 1)

        for owner in (self.vae, getattr(self.vae, "config", None)):
            if owner is None:
                continue
            for attr_name in (
                "temporal_downsample_factor",
                "temporal_compression_ratio",
            ):
                attr_value = getattr(owner, attr_name, None)
                if attr_value is not None:
                    return max(int(attr_value), 1)

        return 4

    def _resolve_actions_per_latent_frame(self) -> int:
        explicit = getattr(self.job_config, "actions_per_latent_frame", None)
        if explicit is not None:
            return max(int(explicit), 1)

        source_action_per_frame = max(
            int(
                getattr(
                    self.job_config,
                    "actions_per_frame",
                    getattr(self.job_config, "action_per_frame", 1),
                )
            ),
            1,
        )
        source_video_frame_stride = max(
            int(getattr(self.job_config, "source_video_frame_stride", 1)),
            1,
        )
        return source_action_per_frame * source_video_frame_stride * self.video_frames_per_latent

    def preprocess_action(self, action: np.ndarray) -> torch.Tensor:
        action_model_input = torch.from_numpy(np.asarray(action, dtype=np.float32))
        if action_model_input.ndim != 3:
            raise ValueError(f"Expected action state shape [C,F,H], got {tuple(action_model_input.shape)}")
        action_model_input_padded = F.pad(
            action_model_input,
            [0, 0, 0, 0, 0, 1],
            mode="constant",
            value=0,
        )
        action_model_input = action_model_input_padded[
            list(self.job_config.inverse_used_action_channel_ids)
        ]

        if self.action_norm_method == "quantiles":
            assert self.actions_q01 is not None and self.actions_q99 is not None
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6
            ) * 2.0 - 1.0
        else:
            raise NotImplementedError(f"Unsupported action_norm_method: {self.action_norm_method}")
        return action_model_input.unsqueeze(0).unsqueeze(-1)

    def _build_zero_action_model_input(
        self,
        frame_count: int,
        reference_tensor: Union[torch.Tensor, VideoLatents],
    ) -> torch.Tensor:
        """Build an all-zero action condition matching the normalized model layout."""
        tensor_ref = (
            reference_tensor[0]
            if isinstance(reference_tensor, list)
            else reference_tensor
        )
        return torch.zeros(
            [
                1,
                int(self.job_config.action_dim),
                int(frame_count),
                self.actions_per_latent_frame,
                1,
            ],
            device=tensor_ref.device,
            dtype=tensor_ref.dtype,
        )

    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.cpu()[0, ..., 0]
        if self.action_norm_method == "quantiles":
            assert self.actions_q01 is not None and self.actions_q99 is not None
            action = (action + 1.0) / 2.0 * (
                self.actions_q99 - self.actions_q01 + 1e-6
            ) + self.actions_q01
        else:
            raise NotImplementedError(f"Unsupported action_norm_method: {self.action_norm_method}")
        action_np = action.detach().cpu().numpy()
        return action_np[list(self.job_config.used_action_channel_ids)]

    def _prepare_rtc_prefix_action(
        self,
        prefix_action: Any,
        rtc_config: Optional[Dict[str, Any]],
        reference_actions: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Normalize an RTC prefix and build its inpainting weights."""
        if prefix_action is None or not rtc_config:
            return None, None

        try:
            prefix_np = np.asarray(prefix_action, dtype=np.float32)
            if prefix_np.ndim == 2:
                expected_steps = (
                    int(self.job_config.frame_chunk_size)
                    * int(self.actions_per_latent_frame)
                )
                if prefix_np.shape[0] != expected_steps:
                    raise ValueError(
                        "2D RTC prefix must have T=frame_chunk_size*actions_per_latent_frame, "
                        f"got T={prefix_np.shape[0]} expected={expected_steps}"
                    )
                prefix_np = prefix_np.reshape(
                    int(self.job_config.frame_chunk_size),
                    int(self.actions_per_latent_frame),
                    prefix_np.shape[1],
                )
                prefix_np = np.moveaxis(prefix_np, -1, 0)
            if prefix_np.ndim != 3:
                raise ValueError(f"Expected RTC prefix shape [C,F,H] or [T,C], got {prefix_np.shape}")

            prefix_tensor = self.preprocess_action(prefix_np).to(
                device=reference_actions.device,
                dtype=reference_actions.dtype,
            )
            if tuple(prefix_tensor.shape) != tuple(reference_actions.shape):
                raise ValueError(
                    "Normalized RTC prefix shape does not match action sample: "
                    f"prefix={tuple(prefix_tensor.shape)} action={tuple(reference_actions.shape)}"
                )

            schedule = str(rtc_config.get("prefix_attention_schedule", "exp"))
            weights = action_prefix_weight_tensor(
                prefix_tensor,
                inference_delay=int(rtc_config.get("inference_delay", 0)),
                prefix_attention_horizon=int(
                    rtc_config.get("prefix_attention_horizon", prefix_tensor.shape[2] * prefix_tensor.shape[3])
                ),
                schedule=schedule,
            )
            logger.info(
                "LingBot-VA RTC prefix enabled: "
                f"prefix_shape={tuple(prefix_tensor.shape)}, "
                f"inference_delay={int(rtc_config.get('inference_delay', 0))}, "
                f"prefix_attention_horizon={int(rtc_config.get('prefix_attention_horizon', 0))}, "
                f"schedule={schedule}, "
                f"weight_nonzero_steps={int(torch.count_nonzero(weights > 0).item())}, "
                f"weight_sum={float(weights.sum().item()):.4f}, "
                f"weight_min={float(weights.min().item()):.4f}, "
                f"weight_max={float(weights.max().item()):.4f}, "
                "weight_distribution="
                f"{[round(float(v), 4) for v in weights.reshape(-1).detach().float().cpu().tolist()]}"
            )
            return prefix_tensor, weights
        except Exception as exc:
            logger.warning(f"LingBot-VA RTC prefix ignored: {exc}")
            return None, None

    def _repeat_input_for_cfg(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        repeated = dict(input_dict)
        if self.use_cfg:
            assert self.prompt_embeds is not None and self.negative_prompt_embeds is not None
            noisy_latents = repeated["noisy_latents"]
            if isinstance(noisy_latents, list):
                repeated["noisy_latents"] = [
                    latent.repeat(2, 1, 1, 1, 1) for latent in noisy_latents
                ]
            else:
                repeated["noisy_latents"] = noisy_latents.repeat(2, 1, 1, 1, 1)
            repeated["text_emb"] = torch.cat(
                [
                    self.prompt_embeds.to(self.dtype).clone(),
                    self.negative_prompt_embeds.to(self.dtype).clone(),
                ],
                dim=0,
            )
            repeated["grid_id"] = repeated["grid_id"][None].repeat(2, 1, 1)
            repeated["timesteps"] = repeated["timesteps"][None].repeat(2, 1)
        else:
            repeated["grid_id"] = repeated["grid_id"][None]
            repeated["timesteps"] = repeated["timesteps"][None]
        return repeated

    @staticmethod
    def _latent_shape_summary(latents: Union[torch.Tensor, VideoLatents]) -> Any:
        if isinstance(latents, list):
            return [tuple(latent.shape) for latent in latents]
        return tuple(latents.shape)

    def _data_seq_to_video_latents(
        self,
        data_seq: torch.Tensor,
        *,
        frame_count: int,
        batch_size: int,
    ) -> VideoLatents:
        patch_f, patch_h, patch_w = (int(v) for v in self.job_config.patch_size)
        patch_elements = patch_f * patch_h * patch_w
        expanded_token_counts = []
        for latent_height, latent_width in self.latent_hw_shape:
            expanded_token_counts.append(
                (frame_count // patch_f)
                * (latent_height // patch_h)
                * (latent_width // patch_w)
                * patch_elements
            )

        view_sequences = torch.split(data_seq, expanded_token_counts, dim=1)
        if len(view_sequences) != len(self.latent_hw_shape):
            raise ValueError(
                "Video latent sequence split count mismatch: "
                f"{len(view_sequences)} != {len(self.latent_hw_shape)}"
            )
        return [
            data_seq_to_patch(
                self.job_config.patch_size,
                view_seq,
                frame_count,
                latent_height,
                latent_width,
                batch_size=batch_size,
            )
            for view_seq, (latent_height, latent_width) in zip(
                view_sequences, self.latent_hw_shape, strict=True
            )
        ]

    def _get_video_mesh_id(
        self,
        latent_model_input: Union[torch.Tensor, VideoLatents],
        frame_st_id: int,
    ) -> torch.Tensor:
        patch_f, patch_h, patch_w = self.job_config.patch_size
        if isinstance(latent_model_input, torch.Tensor):
            return get_mesh_id(
                latent_model_input.shape[-3] // patch_f,
                latent_model_input.shape[-2] // patch_h,
                latent_model_input.shape[-1] // patch_w,
                0,
                1,
                frame_st_id,
            ).to(self.device)

        grid_ids: List[torch.Tensor] = []
        width_offset = 0
        for latent in latent_model_input:
            view_grid_id = get_mesh_id(
                latent.shape[-3] // patch_f,
                latent.shape[-2] // patch_h,
                latent.shape[-1] // patch_w,
                0,
                1,
                frame_st_id,
            ).to(self.device)
            view_grid_id[2] += width_offset
            width_offset += latent.shape[-1] // patch_w
            grid_ids.append(view_grid_id)
        return torch.cat(grid_ids, dim=1)

    def _prepare_latent_input(
        self,
        latent_model_input: Optional[Union[torch.Tensor, VideoLatents]],
        action_model_input: Optional[torch.Tensor],
        latent_t: Any = 0,
        action_t: Any = 0,
        latent_cond: Optional[Union[torch.Tensor, VideoLatents]] = None,
        action_cond: Optional[torch.Tensor] = None,
        frame_st_id: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        input_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        patch_size = self.job_config.patch_size
        if latent_model_input is not None:
            latent_frame_count = _video_latents_frame_count(latent_model_input)
            input_dict["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones(
                    [latent_frame_count],
                    dtype=torch.float32,
                    device=self.device,
                )
                * latent_t,
                "grid_id": self._get_video_mesh_id(latent_model_input, frame_st_id),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                noisy_latents = input_dict["latent_res_lst"]["noisy_latents"]
                if isinstance(noisy_latents, list):
                    cond_list = latent_cond if isinstance(latent_cond, list) else [latent_cond]
                    if len(noisy_latents) != len(cond_list):
                        raise ValueError(
                            "latent_cond view count must match noisy_latents: "
                            f"{len(cond_list)} != {len(noisy_latents)}"
                        )
                    for latent, cond in zip(noisy_latents, cond_list):
                        latent[:, :, 0:1] = cond[:, :, 0:1]
                else:
                    noisy_latents[:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            input_dict["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones(
                    [action_model_input.shape[2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict["action_res_lst"]["timesteps"][0:1] *= 0
            assert self.action_mask is not None
            input_dict["action_res_lst"]["noisy_latents"][:, ~self.action_mask] *= 0
        return input_dict

    def _center_crop_square_resize_video(
        self,
        video: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize/crop/pad each frame according to runtime preprocessing config."""
        _, _, image_height, image_width = video.shape
        reshape_mode = str(getattr(self.job_config, "reshape_mode", "resize")).lower()
        if bool(getattr(self.job_config, "center_crop_before_resize", False)):
            crop_size = min(image_height, image_width)
            top = max((image_height - crop_size) // 2, 0)
            left = max((image_width - crop_size) // 2, 0)
            video = video[:, :, top : top + crop_size, left : left + crop_size]
        elif reshape_mode == "crop_resize":
            crop_before_resize = getattr(self.job_config, "crop_before_resize", None)
            if (
                not isinstance(crop_before_resize, (list, tuple))
                or len(crop_before_resize) != 4
            ):
                raise ValueError(
                    "crop_before_resize must be [left, right, top, bottom] "
                    "when reshape_mode is 'crop_resize', "
                    f"got {crop_before_resize!r}"
                )
            crop_left, crop_right, crop_top, crop_bottom = (
                int(v) for v in crop_before_resize
            )
            if min(crop_left, crop_right, crop_top, crop_bottom) < 0:
                raise ValueError(
                    "crop_before_resize values must be non-negative, "
                    f"got {crop_before_resize!r}"
                )
            crop_height = image_height - crop_top - crop_bottom
            crop_width = image_width - crop_left - crop_right
            if crop_height <= 0 or crop_width <= 0:
                raise ValueError(
                    "crop_before_resize removes the whole image: "
                    f"image_shape={(image_height, image_width)}, "
                    f"crop={crop_before_resize!r}"
                )
            video = video[
                :,
                :,
                crop_top : crop_top + crop_height,
                crop_left : crop_left + crop_width,
            ]

        if reshape_mode == "original":
            resized_video = video
        elif reshape_mode == "pad":
            pad_h = max(0, int(height) - int(video.shape[2]))
            pad_w = max(0, int(width) - int(video.shape[3]))
            resized_video = F.pad(video, (0, pad_w, 0, pad_h), mode="constant", value=0)
        elif reshape_mode in {"resize", "crop_resize"}:
            resize_size = getattr(self.job_config, "crop_resize_size", None)
            if resize_size is not None:
                if isinstance(resize_size, (list, tuple)) and len(resize_size) == 2:
                    resize_height, resize_width = int(resize_size[0]), int(resize_size[1])
                else:
                    resize_height = resize_width = int(resize_size)
            else:
                resize_height, resize_width = int(height), int(width)
            resized_video = F.interpolate(
                video,
                size=(resize_height, resize_width),
                mode="bilinear",
                align_corners=False,
            )
        else:
            raise ValueError(f"Unsupported LingBot-VA reshape_mode: {reshape_mode!r}")

        pad_after_resize = getattr(self.job_config, "pad_after_resize", None)
        if pad_after_resize is None:
            return resized_video
        if not isinstance(pad_after_resize, (list, tuple)) or len(pad_after_resize) != 4:
            raise ValueError(
                "pad_after_resize must be [left, right, top, bottom], "
                f"got {pad_after_resize!r}"
            )
        return F.pad(
            resized_video,
            tuple(int(v) for v in pad_after_resize),
            mode="constant",
            value=0,
        )

    def _view_preprocess_output_shape(self, camera_index: int) -> Tuple[int, int]:
        if self.env_type == "robotwin_tshape":
            if camera_index == 0:
                height_i, width_i = self.height, self.width
            else:
                height_i, width_i = self.height // 2, self.width // 2
        else:
            height_i, width_i = self.height, self.width

        reshape_mode = str(getattr(self.job_config, "reshape_mode", "resize"))
        resize_size = getattr(self.job_config, "crop_resize_size", None)
        if reshape_mode in {"resize", "crop_resize"} and resize_size is not None:
            if isinstance(resize_size, (list, tuple)) and len(resize_size) == 2:
                out_height, out_width = int(resize_size[0]), int(resize_size[1])
            else:
                out_height = out_width = int(resize_size)
        else:
            out_height, out_width = int(height_i), int(width_i)

        pad_after_resize = getattr(self.job_config, "pad_after_resize", None)
        if pad_after_resize is not None:
            if not isinstance(pad_after_resize, (list, tuple)) or len(pad_after_resize) != 4:
                raise ValueError(
                    "pad_after_resize must be [left, right, top, bottom], "
                    f"got {pad_after_resize!r}"
                )
            left, right, top, bottom = (int(v) for v in pad_after_resize)
            out_height += top + bottom
            out_width += left + right
        return int(out_height), int(out_width)

    def _view_padded_latent_hw(self, camera_index: int) -> Tuple[int, int]:
        image_height, image_width = self._view_preprocess_output_shape(camera_index)
        latent_height = image_height // 16
        latent_width = image_width // 16
        _, patch_h, patch_w = (int(v) for v in self.job_config.patch_size)
        if patch_h > 0 and latent_height % patch_h:
            latent_height += (-latent_height) % patch_h
        if patch_w > 0 and latent_width % patch_w:
            latent_width += (-latent_width) % patch_w
        return int(latent_height), int(latent_width)

    def _encode_obs(self, obs: Dict[str, Any]) -> Optional[Union[torch.Tensor, VideoLatents]]:
        images = obs["obs"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None

        videos = []
        for camera_index, camera_key in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == "robotwin_tshape":
                if camera_index == 0:
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            missing = [idx for idx, item in enumerate(images) if camera_key not in item]
            if missing:
                raise KeyError(f"Missing camera key '{camera_key}' in obs indices {missing}")
            history_video = torch.from_numpy(
                np.stack([np.asarray(item[camera_key]) for item in images])
            ).float().permute(3, 0, 1, 2)
            history_video = self._center_crop_square_resize_video(
                history_video,
                height=height_i,
                width=width_i,
            ).unsqueeze(0)
            videos.append(history_video)

        vae_device = next(self.streaming_vae.vae.parameters()).device
        latents_by_view: VideoLatents = []
        for camera_index, history_video in enumerate(videos):
            video_tensor = history_video.to(vae_device).to(self.dtype) / 255.0 * 2.0 - 1.0
            enc_out = self.streaming_vae.encode_chunk(video_tensor, idx=camera_index)
            mu, _ = torch.chunk(enc_out, 2, dim=1)
            latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
            latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
            mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
            video_latent = _pad_latent_to_patch_size(
                mu_norm,
                self.job_config.patch_size,
            )
            latents_by_view.append(video_latent.to(self.device))
        if len(latents_by_view) == 1:
            return latents_by_view[0]
        return latents_by_view

    def _reset(self, prompt: Optional[str] = None) -> None:
        logger.info("LingBot-VA server reset")
        self.use_cfg = (
            float(self.job_config.guidance_scale) > 1.0
            or float(self.job_config.action_guidance_scale) > 1.0
        )
        self.frame_st_id = 0
        self.init_latent = None
        self.last_known_video_latent = None
        self.transformer.clear_cache(self.cache_name)
        if hasattr(self.streaming_vae, "clear_cache"):
            self.streaming_vae.clear_cache()
        else:
            self.streaming_vae.init_cache()

        self.source_action_per_frame = max(
            int(
                getattr(
                    self.job_config,
                    "actions_per_frame",
                    getattr(self.job_config, "action_per_frame", 1),
                )
            ),
            1,
        )
        self.source_video_frame_stride = max(
            int(getattr(self.job_config, "source_video_frame_stride", 1)),
            1,
        )
        self.video_frames_per_latent = self._resolve_video_frames_per_latent()
        self.actions_per_latent_frame = self._resolve_actions_per_latent_frame()
        logger.info(
            "LingBot-VA action timing: "
            f"source_action_per_frame={self.source_action_per_frame}, "
            f"source_video_frame_stride={self.source_video_frame_stride}, "
            f"video_frames_per_latent={self.video_frames_per_latent}, "
            f"actions_per_latent_frame={self.actions_per_latent_frame}"
        )
        self.height = int(self.job_config.height)
        self.width = int(self.job_config.width)

        patch_size = self.job_config.patch_size
        self.latent_hw_shape = [
            self._view_padded_latent_hw(camera_index)
            for camera_index, _ in enumerate(self.job_config.obs_cam_keys)
        ]
        self.latent_height = max(height for height, _ in self.latent_hw_shape)
        self.latent_width = sum(width for _, width in self.latent_hw_shape)
        latent_token_per_chunk = 0
        for latent_height, latent_width in self.latent_hw_shape:
            latent_token_per_chunk += (
                int(self.job_config.frame_chunk_size) * latent_height * latent_width
            ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = (
            int(self.job_config.frame_chunk_size) * self.actions_per_latent_frame
        )
        logger.info(
            "LingBot-VA latent cache shape: "
            f"view_latent_hw={self.latent_hw_shape}, "
            f"latent_token_per_chunk={latent_token_per_chunk}, "
            f"action_token_per_chunk={action_token_per_chunk}"
        )
        self.transformer.create_empty_cache(
            self.cache_name,
            int(self.job_config.attn_window),
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.device,
            dtype=self.dtype,
            batch_size=2 if self.use_cfg else 1,
        )
        self.latent_token_per_frame = max(
            latent_token_per_chunk // max(int(self.job_config.frame_chunk_size), 1),
            1,
        )
        self.action_token_per_frame = self.actions_per_latent_frame

        self.action_mask = torch.zeros([int(self.job_config.action_dim)], device=self.device).bool()
        self.action_mask[list(self.job_config.used_action_channel_ids)] = True

        self.actions_q01 = torch.tensor(
            self.job_config.norm_stat["q01"],
            dtype=torch.float32,
        ).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(
            self.job_config.norm_stat["q99"],
            dtype=torch.float32,
        ).reshape(-1, 1, 1)

        if prompt is None:
            prompt = getattr(self.job_config, "task", None)
        if prompt is None:
            raise ValueError("LingBot-VA prompt/task is required for inference")
        self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=float(self.job_config.guidance_scale) > 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=self.device,
            dtype=self.dtype,
        )
        self.exp_name = (
            f"run_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{time.time_ns() % 1_000_000_000:09d}_pid{os.getpid()}"
        )
        self.exp_save_root = os.path.join(self.save_root, "real", self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_transformer_cache_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary of current transformer KV cache usage."""
        blocks = getattr(self.transformer, "blocks", None)
        if not blocks:
            return {"available": False, "reason": "transformer has no blocks"}

        first_block = blocks[0]
        attn_caches = getattr(first_block, "attn_caches", None)
        if isinstance(attn_caches, dict):
            cache = attn_caches.get(self.cache_name)
            if cache is None:
                return {"available": False, "reason": f"cache {self.cache_name!r} is empty"}

            if "video_len" in cache and "action_len" in cache:
                video_capacity = int(cache["video_k"].shape[1])
                action_capacity = int(cache["action_k"].shape[1])
                video_len = int(cache["video_len"])
                action_len = int(cache["action_len"])
                video_pred_len = int(cache.get("video_pred_len", 0))
                action_pred_len = int(cache.get("action_pred_len", 0))
                return {
                    "available": True,
                    "type": "mot",
                    "video_tokens": video_len,
                    "video_capacity": video_capacity,
                    "video_frames_approx": video_len / float(self.latent_token_per_frame),
                    "video_pred_tokens": video_pred_len,
                    "action_tokens": action_len,
                    "action_capacity": action_capacity,
                    "action_frames_approx": action_len / float(self.action_token_per_frame),
                    "action_pred_tokens": action_pred_len,
                }

        attn1 = getattr(first_block, "attn1", None)
        attn_caches = getattr(attn1, "attn_caches", None)
        if isinstance(attn_caches, dict):
            cache = attn_caches.get(self.cache_name)
            if cache is None:
                return {"available": False, "reason": f"cache {self.cache_name!r} is empty"}
            mask = cache.get("mask")
            is_pred = cache.get("is_pred")
            used_tokens = int(mask.sum().item()) if mask is not None else 0
            pred_tokens = int(is_pred.sum().item()) if is_pred is not None else 0
            capacity = int(cache["k"].shape[1])
            return {
                "available": True,
                "type": "shared",
                "tokens": used_tokens,
                "capacity": capacity,
                "pred_tokens": pred_tokens,
            }

        return {"available": False, "reason": "unknown cache layout"}

    def _log_transformer_cache_summary(self, prefix: str) -> None:
        summary = self._get_transformer_cache_summary()
        if not summary.get("available", False):
            logger.info(
                f"{prefix}: KV cache unavailable ({summary.get('reason', 'unknown')})"
            )
            return

        if summary.get("type") == "mot":
            logger.info(
                f"{prefix}: KV cache "
                f"video={summary['video_tokens']}/{summary['video_capacity']} tokens "
                f"(~{summary['video_frames_approx']:.2f} frames, "
                f"pred={summary['video_pred_tokens']}), "
                f"action={summary['action_tokens']}/{summary['action_capacity']} tokens "
                f"(~{summary['action_frames_approx']:.2f} frames, "
                f"pred={summary['action_pred_tokens']}), "
                f"frame_st_id={self.frame_st_id}"
            )
            return

        logger.info(
            f"{prefix}: KV cache tokens={summary['tokens']}/{summary['capacity']} "
            f"pred={summary['pred_tokens']}, frame_st_id={self.frame_st_id}"
        )

    def _decode_latents_to_video_np(self, latents: torch.Tensor) -> np.ndarray:
        """Decode normalized Wan latents into a numpy video array [F, H, W, C]."""
        original_device = next(self.vae.parameters()).device
        moved_vae = False
        if original_device != self.device:
            self.vae = self.vae.to(self.device).to(self.dtype)
            moved_vae = True

        try:
            vae_device = next(self.vae.parameters()).device
            vae_dtype = next(self.vae.parameters()).dtype
            latents = latents.to(device=vae_device, dtype=vae_dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                1.0
                / torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            denormalized_latents = latents / latents_std + latents_mean
            video = self.vae.decode(denormalized_latents, return_dict=False)[0]
            decoded = self.video_processor.postprocess_video(video, output_type="np")
            if isinstance(decoded, list):
                video_np = np.asarray(decoded[0])
            else:
                video_np = np.asarray(decoded)
                if video_np.ndim == 5:
                    video_np = video_np[0]
            return video_np
        finally:
            if moved_vae and self.offload_vae:
                self.vae = self.vae.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _crop_decoded_video_to_view_shape(
        self,
        video_np: np.ndarray,
        view_index: Optional[int],
    ) -> np.ndarray:
        if view_index is None:
            return video_np
        target_height, target_width = self._view_preprocess_output_shape(view_index)
        if video_np.shape[1] <= target_height and video_np.shape[2] <= target_width:
            return video_np
        return np.ascontiguousarray(
            video_np[
                :,
                : min(video_np.shape[1], target_height),
                : min(video_np.shape[2], target_width),
                :,
            ]
        )

    def _save_decoded_latents(
        self,
        latents: Union[torch.Tensor, VideoLatents],
        category: str,
        frame_st_id: int,
        anchor_latent: Optional[Union[torch.Tensor, VideoLatents]] = None,
        drop_first_decoded_frame: bool = False,
        view_index: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Decode latents and save each decoded frame as a PNG image."""
        if isinstance(latents, list):
            anchor_list = anchor_latent if isinstance(anchor_latent, list) else None
            videos = []
            for view_index, view_latent in enumerate(latents):
                view_anchor = (
                    anchor_list[view_index]
                    if anchor_list is not None and view_index < len(anchor_list)
                    else None
                )
                view_video = self._save_decoded_latents(
                    view_latent,
                    category=f"{category}_view{view_index}",
                    frame_st_id=frame_st_id,
                    anchor_latent=view_anchor,
                    drop_first_decoded_frame=drop_first_decoded_frame,
                    view_index=view_index,
                )
                if view_video is not None:
                    videos.append(view_video)
            if not videos:
                return None
            frame_count = min(video.shape[0] for video in videos)
            videos = [video[:frame_count] for video in videos]
            return np.ascontiguousarray(np.concatenate(videos, axis=2))

        if category == "predicted_video":
            clip_index = self._decoded_predicted_video_index
            self._decoded_predicted_video_index += 1
        else:
            clip_index = self._decoded_history_cache_index
            self._decoded_history_cache_index += 1

        output_dir = (
            self.decoded_latent_save_dir
            / category
            / f"clip_{clip_index:06d}_frame_st_{frame_st_id:06d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            latents_to_decode = latents.detach()
            if anchor_latent is not None:
                # latents_to_decode = torch.cat(
                #     [
                #         anchor_latent.detach().to(
                #             device=latents_to_decode.device,
                #             dtype=latents_to_decode.dtype,
                #         ),
                #         latents_to_decode,
                #     ],
                #     dim=2,
                # )
                pass
            video_np = self._decode_latents_to_video_np(latents_to_decode)
            if video_np.ndim != 4:
                raise ValueError(f"Decoded video must be [F,H,W,C], got {video_np.shape}")
            video_np = self._crop_decoded_video_to_view_shape(video_np, view_index)
            if drop_first_decoded_frame:
                video_np = video_np[1:]
                if video_np.shape[0] == 0:
                    raise ValueError("Decoded video is empty after dropping anchor frame")

            for frame_index, frame in enumerate(video_np):
                frame_array = np.asarray(frame)
                if frame_array.dtype != np.uint8:
                    if frame_array.size > 0 and frame_array.max() <= 1.0:
                        frame_array = frame_array * 255.0
                    frame_array = np.clip(frame_array, 0.0, 255.0).astype(np.uint8)
                imageio.imwrite(
                    str(output_dir / f"frame_{frame_index:03d}.png"),
                    np.ascontiguousarray(frame_array),
                )
            logger.info(
                f"LingBot-VA saved decoded {category} frames to {output_dir} "
                f"(frames={video_np.shape[0]}, latent_shape={tuple(latents.shape)}, "
                f"decoded_latent_shape={tuple(latents_to_decode.shape)})"
            )
            return video_np
        except Exception as exc:
            logger.warning(
                f"LingBot-VA failed to decode/save {category} latents "
                f"shape={tuple(latents.shape)}: {exc}"
            )
            return None

    def _video_np_to_uint8(self, video_np: np.ndarray) -> np.ndarray:
        video_array = np.asarray(video_np)
        if video_array.ndim != 4:
            raise ValueError(f"Expected video [F,H,W,C], got {video_array.shape}")
        if video_array.shape[-1] == 1:
            video_array = np.repeat(video_array, 3, axis=-1)
        elif video_array.shape[-1] > 3:
            video_array = video_array[..., :3]

        if video_array.dtype == np.uint8:
            return np.ascontiguousarray(video_array)

        video_float = video_array.astype(np.float32, copy=False)
        if video_float.size > 0:
            max_value = float(np.nanmax(video_float))
            min_value = float(np.nanmin(video_float))
            if max_value <= 1.0 and min_value >= 0.0:
                video_float = video_float * 255.0
        return np.ascontiguousarray(np.clip(video_float, 0.0, 255.0).astype(np.uint8))

    def _resize_video_np(self, video_np: np.ndarray, height: int, width: int) -> np.ndarray:
        video_uint8 = self._video_np_to_uint8(video_np)
        if video_uint8.shape[1] == height and video_uint8.shape[2] == width:
            return video_uint8

        video_tensor = torch.from_numpy(video_uint8).float().permute(0, 3, 1, 2)
        resized = F.interpolate(
            video_tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return (
            resized.permute(0, 2, 3, 1)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    def _extract_raw_history_video_np(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        images = obs.get("obs", [])
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            return None

        raw_videos = []
        for camera_index, camera_key in enumerate(self.job_config.obs_cam_keys):
            missing = [idx for idx, item in enumerate(images) if camera_key not in item]
            if missing:
                logger.warning(
                    f"LingBot-VA raw history comparison skipped camera {camera_key}: "
                    f"missing frames {missing}"
                )
                continue

            if self.env_type == "robotwin_tshape" and camera_index > 0:
                height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video = torch.from_numpy(
                np.stack([np.asarray(item[camera_key]) for item in images])
            ).float().permute(3, 0, 1, 2)
            history_video = self._center_crop_square_resize_video(
                history_video,
                height=height_i,
                width=width_i,
            )
            raw_videos.append(
                history_video.permute(1, 2, 3, 0)
                .clamp(0.0, 255.0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

        if not raw_videos:
            return None
        return np.ascontiguousarray(np.concatenate(raw_videos, axis=2))

    def _save_history_prediction_comparison(
        self,
        raw_history_video: Optional[np.ndarray],
        reconstructed_history_video: Optional[np.ndarray],
        predicted_video: Optional[np.ndarray],
        frame_st_id: int,
    ) -> None:
        if (
            raw_history_video is None
            or reconstructed_history_video is None
            or predicted_video is None
        ):
            return

        try:
            recon_uint8 = self._video_np_to_uint8(reconstructed_history_video)
            target_height, target_width = int(recon_uint8.shape[1]), int(recon_uint8.shape[2])
            raw_uint8 = self._resize_video_np(raw_history_video, target_height, target_width)
            pred_uint8 = self._resize_video_np(predicted_video, target_height, target_width)

            frame_count = min(
                raw_uint8.shape[0],
                recon_uint8.shape[0],
                pred_uint8.shape[0],
            )
            if frame_count <= 0:
                return

            output_dir = (
                self.comparison_grid_save_dir
                / f"comparison_{self._history_prediction_comparison_index:06d}"
                f"_frame_st_{frame_st_id:06d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self._history_prediction_comparison_index += 1

            for frame_index in range(frame_count):
                rows = [
                    raw_uint8[frame_index],
                    recon_uint8[frame_index],
                    pred_uint8[frame_index],
                ]
                separator = np.zeros((8, rows[0].shape[1], 3), dtype=np.uint8)
                comparison = np.concatenate(
                    [rows[0], separator, rows[1], separator, rows[2]],
                    axis=0,
                )
                output_path = output_dir / f"frame_{frame_index:03d}.png"
                imageio.imwrite(str(output_path), np.ascontiguousarray(comparison))

            logger.info(
                "LingBot-VA saved history/prediction comparison frames to "
                f"{output_dir} (frames={frame_count})"
            )
        except Exception as exc:
            logger.warning(
                "LingBot-VA failed to save history/prediction comparison grid: "
                f"{exc}"
            )

    def _prefill_initial_latent(self, obs: Dict[str, Any]) -> None:
        if self.frame_st_id != 0:
            return
        init_latent = self._encode_obs(obs)
        if init_latent is None:
            raise ValueError("No initial observation latent can be encoded")
        self.init_latent = init_latent
        if isinstance(init_latent, list):
            self.last_known_video_latent = [
                latent[:, :, -1:].detach().clone() for latent in init_latent
            ]
        else:
            self.last_known_video_latent = init_latent[:, :, -1:].detach().clone()
        input_dict = self._prepare_latent_input(
            init_latent,
            None,
            latent_t=0,
            frame_st_id=0,
        )["latent_res_lst"]
        action_latent_frames = _video_latents_frame_count(init_latent)
        if (not self.zero_action_condition) and "state" in obs:
            state = np.asarray(obs["state"], dtype=np.float32)
            if state.ndim != 3:
                raise ValueError(
                    "Initial FastWAM state condition must have shape [C,F,H], "
                    f"got {state.shape}"
                )
            if state.shape[1] < action_latent_frames:
                pad_count = action_latent_frames - state.shape[1]
                pad = np.repeat(state[:, -1:, :], pad_count, axis=1)
                state = np.concatenate([state, pad], axis=1)
            state = state[:, :action_latent_frames, :]
            zero_action = self.preprocess_action(state).to(
                device=self.device,
                dtype=self.dtype,
            )
            logger.info(
                "LingBot-VA FastWAM initial action cache uses provided state: "
                f"state={tuple(state.shape)} normalized={tuple(zero_action.shape)}"
            )
        else:
            zero_action = torch.zeros(
                [
                    1,
                    int(self.job_config.action_dim),
                    action_latent_frames,
                    self.actions_per_latent_frame,
                    1,
                ],
                device=self.device,
                dtype=self.dtype,
            )
        action_input_dict = self._prepare_latent_input(
            None,
            zero_action,
            action_t=0,
            frame_st_id=0,
        )["action_res_lst"]
        self._mark_action_condition_all_missing(action_input_dict)
        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(action_input_dict),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )
        self.frame_st_id += int(action_latent_frames)

    def _infer_video_latents_for_visualization(self) -> Union[torch.Tensor, VideoLatents]:
        """Predict video latents for visualization without mutating fastwam KV cache."""
        start_time = time.time()
        frame_chunk_size = int(self.job_config.frame_chunk_size)
        if len(self.latent_hw_shape) > 1:
            latents: Union[torch.Tensor, VideoLatents] = [
                torch.randn(
                    1,
                    48,
                    frame_chunk_size,
                    latent_height,
                    latent_width,
                    device=self.device,
                    dtype=self.dtype,
                )
                for latent_height, latent_width in self.latent_hw_shape
            ]
        else:
            latents = torch.randn(
                1,
                48,
                frame_chunk_size,
                self.latent_height,
                self.latent_width,
                device=self.device,
                dtype=self.dtype,
            )
        self.scheduler.set_timesteps(int(self.job_config.num_inference_steps))
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        video_step = int(self.job_config.video_exec_step)
        if video_step != -1:
            timesteps = timesteps[:video_step]

        with torch.no_grad():
            for step_index, timestep in enumerate(timesteps):
                last_step = step_index == len(timesteps) - 1
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    latent_t=timestep,
                    frame_st_id=self.frame_st_id,
                )["latent_res_lst"]
                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict),
                    update_cache=0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )
                if not last_step or video_step != -1:
                    batch_size = 2 if self.use_cfg else 1
                    if isinstance(latents, list):
                        video_noise_pred_by_view = self._data_seq_to_video_latents(
                            video_noise_pred,
                            frame_count=frame_chunk_size,
                            batch_size=batch_size,
                        )
                        if float(self.job_config.guidance_scale) > 1.0:
                            video_noise_pred_by_view = [
                                pred[1:]
                                + float(self.job_config.guidance_scale)
                                * (pred[:1] - pred[1:])
                                for pred in video_noise_pred_by_view
                            ]
                        else:
                            video_noise_pred_by_view = [
                                pred[:1] for pred in video_noise_pred_by_view
                            ]
                        latents = _step_video_latents(
                            self.scheduler,
                            video_noise_pred_by_view,
                            timestep,
                            latents,
                        )
                    else:
                        video_noise_pred = data_seq_to_patch(
                            self.job_config.patch_size,
                            video_noise_pred,
                            frame_chunk_size,
                            self.latent_height,
                            self.latent_width,
                            batch_size=batch_size,
                        )
                        if float(self.job_config.guidance_scale) > 1.0:
                            video_noise_pred = video_noise_pred[1:] + float(
                                self.job_config.guidance_scale
                            ) * (video_noise_pred[:1] - video_noise_pred[1:])
                        else:
                            video_noise_pred = video_noise_pred[:1]
                        latents = self.scheduler.step(
                            video_noise_pred,
                            timestep,
                            latents,
                            return_dict=False,
                        )
        end_time = time.time()
        logger.info(f"LingBot-VA fastwam video latent infer time: {end_time - start_time} seconds")
        return latents

    def _maybe_save_predicted_video_latents(
        self,
        predicted_latents: Union[torch.Tensor, VideoLatents],
        frame_st_id: int,
        anchor_latent: Optional[Union[torch.Tensor, VideoLatents]] = None,
    ) -> None:
        """Decode and save predicted video latents when debug saving is enabled."""
        if not self.save_predicted_video_latents:
            return

        logger.info(f"maybe_save")

        predicted_video_np = self._save_decoded_latents(
            predicted_latents,
            category="predicted_video",
            frame_st_id=frame_st_id,
            anchor_latent=anchor_latent,
            drop_first_decoded_frame=anchor_latent is not None,
        )
        if predicted_video_np is not None:
            self._last_predicted_video_np = predicted_video_np

    def _infer_fastwam(self, obs: Dict[str, Any]) -> np.ndarray:
        if not hasattr(self, '_last_infer_fastwam_time'):
            self._last_infer_fastwam_time = time.time()
        else:
            time_diff = time.time() - self._last_infer_fastwam_time
            logger.info(f"LingBot-VA fastwam infer time interval: {time_diff} seconds")
            self._last_infer_fastwam_time = time.time()
        self._prefill_initial_latent(obs)
        self._log_transformer_cache_summary("LingBot-VA fastwam infer uses")
        if self.save_predicted_video_latents:
            predicted_latents = self._infer_video_latents_for_visualization()
            self._maybe_save_predicted_video_latents(
                predicted_latents,
                frame_st_id=self.frame_st_id,
                anchor_latent=self.last_known_video_latent,
            )
        start_time = time.time()
        frame_chunk_size = int(self.job_config.frame_chunk_size)
        actions = torch.randn(
            1,
            int(self.job_config.action_dim),
            frame_chunk_size,
            self.actions_per_latent_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        rtc_config = obs.get("rtc_config")
        rtc_prefix_action, rtc_prefix_weights = self._prepare_rtc_prefix_action(
            obs.get("rtc_prefix_action"),
            rtc_config if isinstance(rtc_config, dict) else None,
            actions,
        )
        rtc_initial_noise = actions.detach().clone() if rtc_prefix_action is not None else None
        self.action_scheduler.set_timesteps(int(self.job_config.action_num_inference_steps))
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        with torch.no_grad():
            for step_index, timestep in enumerate(action_timesteps):
                last_step = step_index == len(action_timesteps) - 1
                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    action_t=timestep,
                    frame_st_id=self.frame_st_id,
                )["action_res_lst"]
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict),
                    update_cache=0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )
                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred,
                        "b (f n) c -> b c f n 1",
                        f=frame_chunk_size,
                    )
                    if float(self.job_config.action_guidance_scale) > 1.0:
                        action_noise_pred = action_noise_pred[1:] + float(
                            self.job_config.action_guidance_scale
                        ) * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(
                        action_noise_pred,
                        timestep,
                        actions,
                        return_dict=False,
                    )
                    if (
                        rtc_prefix_action is not None
                        and rtc_prefix_weights is not None
                        and rtc_initial_noise is not None
                    ):
                        actions = apply_prefix_inpainting(
                            action=actions,
                            prefix_action=rtc_prefix_action,
                            initial_noise=rtc_initial_noise,
                            scheduler=self.action_scheduler,
                            next_step_index=step_index + 1,
                            weights=rtc_prefix_weights,
                        )

        assert self.action_mask is not None
        actions[:, ~self.action_mask] *= 0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        end_time = time.time()
        logger.info(f"LingBot-VA fastwam action infer time: {end_time - start_time} seconds")
        return self.postprocess_action(actions)

    def _infer_idm(self, obs: Dict[str, Any]) -> np.ndarray:
        if not hasattr(self, '_last_infer_idm_time'):
            self._last_infer_idm_time = time.time()
        else:
            time_diff = time.time() - self._last_infer_idm_time
            logger.info(f"LingBot-VA idm infer time interval: {time_diff} seconds")
            self._last_infer_idm_time = time.time()
        
        frame_chunk_size = int(self.job_config.frame_chunk_size)
        if self.frame_st_id == 0:
            self.init_latent = self._encode_obs(obs)
        if self.init_latent is None:
            raise ValueError("Initial latent is unavailable")

        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            int(self.job_config.action_dim),
            frame_chunk_size,
            self.actions_per_latent_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        rtc_config = obs.get("rtc_config")
        rtc_prefix_action, rtc_prefix_weights = self._prepare_rtc_prefix_action(
            obs.get("rtc_prefix_action"),
            rtc_config if isinstance(rtc_config, dict) else None,
            actions,
        )
        rtc_initial_noise = actions.detach().clone() if rtc_prefix_action is not None else None

        self.scheduler.set_timesteps(int(self.job_config.num_inference_steps))
        self.action_scheduler.set_timesteps(int(self.job_config.action_num_inference_steps))
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        video_step = int(self.job_config.video_exec_step)
        if video_step != -1:
            timesteps = timesteps[:video_step]

        with torch.no_grad():
            for step_index, timestep in enumerate(timesteps):
                last_step = step_index == len(timesteps) - 1
                latent_cond = (
                    self.init_latent[:, :, 0:1].to(self.dtype)
                    if self.frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    latent_t=timestep,
                    latent_cond=latent_cond,
                    frame_st_id=self.frame_st_id,
                )["latent_res_lst"]
                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )
                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size,
                        video_noise_pred,
                        frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if float(self.job_config.guidance_scale) > 1.0:
                        video_noise_pred = video_noise_pred[1:] + float(
                            self.job_config.guidance_scale
                        ) * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred, timestep, latents, return_dict=False)
                if latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond

            self._maybe_save_predicted_video_latents(
                latents,
                frame_st_id=self.frame_st_id,
                anchor_latent=self.last_known_video_latent,
            )

            for step_index, timestep in enumerate(action_timesteps):
                last_step = step_index == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [
                            1,
                            int(self.job_config.action_dim),
                            1,
                            self.actions_per_latent_frame,
                            1,
                        ],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    if self.frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    action_t=timestep,
                    action_cond=action_cond,
                    frame_st_id=self.frame_st_id,
                )["action_res_lst"]
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )
                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred,
                        "b (f n) c -> b c f n 1",
                        f=frame_chunk_size,
                    )
                    if float(self.job_config.action_guidance_scale) > 1.0:
                        action_noise_pred = action_noise_pred[1:] + float(
                            self.job_config.action_guidance_scale
                        ) * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(
                        action_noise_pred,
                        timestep,
                        actions,
                        return_dict=False,
                    )
                    if (
                        rtc_prefix_action is not None
                        and rtc_prefix_weights is not None
                        and rtc_initial_noise is not None
                    ):
                        actions = apply_prefix_inpainting(
                            action=actions,
                            prefix_action=rtc_prefix_action,
                            initial_noise=rtc_initial_noise,
                            scheduler=self.action_scheduler,
                            next_step_index=step_index + 1,
                            weights=rtc_prefix_weights,
                        )
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        assert self.action_mask is not None
        actions[:, ~self.action_mask] *= 0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self.postprocess_action(actions)

    def _compute_kv_cache(self, obs: Dict[str, Any]) -> None:
        start_time = time.time()

        self.transformer.clear_pred_cache(self.cache_name)
        obs_frames = obs.get("obs", [])
        obs_frame_count = len(obs_frames) if isinstance(obs_frames, list) else 1
        if obs_frame_count % self.video_frames_per_latent != 0:
            obs_frames = obs_frames[:obs_frame_count - obs_frame_count % self.video_frames_per_latent]
            obs_frame_count = len(obs_frames)
            obs["obs"] = obs_frames
        state_latent_count = obs_frame_count // self.video_frames_per_latent
        latent_model_input = self._encode_obs(obs)
        if latent_model_input is None:
            raise ValueError("No observation latent can be encoded")

        if (
            str(self.job_config.action_condition_mode) != "fastwam"
            and self.frame_st_id == 0
        ):
            if self.init_latent is not None:
                if isinstance(latent_model_input, list) or isinstance(self.init_latent, list):
                    if not isinstance(latent_model_input, list) or not isinstance(self.init_latent, list):
                        raise ValueError("init_latent and latent_model_input must both be lists in multiview mode")
                    latent_model_input = [
                        torch.cat([init, latent], dim=2)
                        for init, latent in zip(self.init_latent, latent_model_input, strict=True)
                    ]
                else:
                    latent_model_input = torch.cat([self.init_latent, latent_model_input], dim=2)
                state_latent_count += 1
            else:
                if isinstance(latent_model_input, list):
                    self.init_latent = [
                        latent[:, :, 0:1].detach().clone()
                        for latent in latent_model_input
                    ]
                else:
                    self.init_latent = latent_model_input[:, :, 0:1].detach().clone()

        if self.zero_action_condition:
            action_model_input = self._build_zero_action_model_input(
                state_latent_count,
                latent_model_input,
            )
        else:
            obs["state"] = obs["state"][:, :state_latent_count]
            latent_ref = (
                latent_model_input[0]
                if isinstance(latent_model_input, list)
                else latent_model_input
            )
            action_model_input = self.preprocess_action(obs["state"]).to(
                device=latent_ref.device,
                dtype=latent_ref.dtype,
            )
        logger.info(
            f"LingBot-VA KV cache update: obs_frames={obs_frame_count}, "
            f"latent={self._latent_shape_summary(latent_model_input)}, "
            f"action={tuple(action_model_input.shape)}, "
            f"zero_action_condition={self.zero_action_condition}, "
            f"frame_st_id={self.frame_st_id}"
        )
        decode_anchor_latent = self.last_known_video_latent
        raw_history_video_np = self._extract_raw_history_video_np(obs)
        reconstructed_history_video_np = None
        if self.save_history_cache_latents:
            reconstructed_history_video_np = self._save_decoded_latents(
                latent_model_input,
                category="history_cache",
                frame_st_id=self.frame_st_id,
                anchor_latent=decode_anchor_latent,
                drop_first_decoded_frame=decode_anchor_latent is not None,
            )
        self._save_history_prediction_comparison(
            raw_history_video_np,
            reconstructed_history_video_np,
            self._last_predicted_video_np,
            frame_st_id=self.frame_st_id,
        )
        self._last_predicted_video_np = None
        input_dict = self._prepare_latent_input(
            latent_model_input,
            action_model_input,
            frame_st_id=self.frame_st_id,
        )
        if "action_res_lst" in input_dict:
            self._mark_action_condition_all_missing(input_dict["action_res_lst"])

        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )
        self.frame_st_id += _video_latents_frame_count(latent_model_input)
        if isinstance(latent_model_input, list):
            self.last_known_video_latent = [
                latent[:, :, -1:].detach().clone()
                for latent in latent_model_input
            ]
        else:
            self.last_known_video_latent = latent_model_input[:, :, -1:].detach().clone()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        end_time = time.time()
        logger.info(f"LingBot-VA KV cache update time: {end_time - start_time} seconds")

    @torch.no_grad()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        reset = bool(obs.get("reset", False))
        prompt = obs.get("prompt", None)
        compute_kv_cache = bool(obs.get("compute_kv_cache", False))

        if reset:
            self._reset(prompt=prompt)
            return {"ok": True}
        if compute_kv_cache:
            self._compute_kv_cache(obs)
            return {"ok": True}

        logger.info(
            f"LingBot-VA infer chunk: mode={self.job_config.action_condition_mode}, "
            f"frame_st_id={self.frame_st_id}"
        )
        if str(self.job_config.action_condition_mode) == "fastwam":
            action = self._infer_fastwam(obs)
        else:
            action = self._infer_idm(obs)
        return {"action": action}


class LingBotVAWebsocketServer:
    """Serve LingBotVAServer over msgpack websocket."""

    def __init__(
        self,
        policy: LingBotVAServer,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = int(port)
        self._metadata = metadata or {}
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lingbot_va_infer",
        )

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            logger.info(f"LingBot-VA websocket server listening on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info(f"LingBot-VA client connected: {websocket.remote_address}")
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total_time: Optional[float] = None

        while True:
            try:
                start_time = time.monotonic()
                recv_start = time.monotonic()
                request_payload = await websocket.recv()
                recv_wait_time = time.monotonic() - recv_start
                request_bytes = (
                    len(request_payload)
                    if isinstance(request_payload, (bytes, bytearray, memoryview))
                    else None
                )
                unpack_start = time.monotonic()
                obs = unpackb(request_payload)
                unpack_time = time.monotonic() - unpack_start
                request_kind = (
                    "reset"
                    if obs.get("reset")
                    else "compute_kv_cache"
                    if obs.get("compute_kv_cache")
                    else "infer"
                )
                loop = asyncio.get_running_loop()
                infer_start = time.monotonic()
                result = await loop.run_in_executor(
                    self._infer_executor,
                    lambda payload=obs: self._policy.infer(payload),
                )
                infer_time = time.monotonic() - infer_start
                result["server_timing"] = {
                    "request_bytes": request_bytes,
                    "recv_wait_ms": recv_wait_time * 1000.0,
                    "unpack_ms": unpack_time * 1000.0,
                    "infer_ms": infer_time * 1000.0,
                }
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                pack_start = time.monotonic()
                response_payload = packer.pack(result)
                pack_time = time.monotonic() - pack_start
                result["server_timing"]["response_bytes"] = len(response_payload)
                result["server_timing"]["response_pack_ms"] = pack_time * 1000.0
                response_payload = packer.pack(result)
                send_start = time.monotonic()
                await websocket.send(response_payload)
                send_time = time.monotonic() - send_start
                prev_total_time = time.monotonic() - start_time
                logger.info(
                    "LingBot-VA websocket timing: "
                    f"kind={request_kind}, request_bytes={request_bytes}, "
                    f"response_bytes={len(response_payload)}, "
                    f"recv_wait_ms={recv_wait_time * 1000.0:.2f}, "
                    f"unpack_ms={unpack_time * 1000.0:.2f}, "
                    f"infer_ms={infer_time * 1000.0:.2f}, "
                    f"response_pack_ms={pack_time * 1000.0:.2f}, "
                    f"send_ms={send_time * 1000.0:.2f}, "
                    f"total_ms={prev_total_time * 1000.0:.2f}"
                )
            except websockets.ConnectionClosed:
                logger.info(f"LingBot-VA client disconnected: {websocket.remote_address}")
                break
            except Exception:
                error_text = traceback.format_exc()
                await websocket.send(error_text)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _health_check(
        self,
        connection: websocket_server.ServerConnection,
        request: websocket_server.Request,
    ) -> Optional[websocket_server.Response]:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LingBot-VA websocket server")
    parser.add_argument("--config", type=str, default=None, help="SuperInference YAML config path")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Run/checkpoint/transformer dir")
    parser.add_argument("--base-model-path", type=str, default=None, help="Base LingBot-VA model path")
    parser.add_argument("--runtime-config-path", type=str, default=None, help="lingbot_runtime_config.json")
    parser.add_argument("--host", type=str, default=None, help="Server bind host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--device", type=str, default=None, help="cuda/cuda:0/cpu")
    parser.add_argument("--action-condition-mode", type=str, default=None, choices=["fastwam", "idm"])
    parser.add_argument(
        "--zero-action-condition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use zero action history and the all-missing action-condition embedding",
    )
    parser.add_argument("--attn-mode", type=str, default=None, help="torch/flashattn/flex")
    parser.add_argument("--model-structure", type=str, default=None, help="shared/mot")
    parser.add_argument("--save-root", type=str, default=None, help="Debug tensor save root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_config = build_job_config(args)
    model = LingBotVAServer(job_config)
    norm_meta = (getattr(job_config, "norm_stat", {}) or {}).get("_meta", {}) or {}
    configured_action_representation = str(
        getattr(job_config, "action_representation", "auto")
    ).lower()
    if configured_action_representation in ("relative", "absolute"):
        action_representation = configured_action_representation
    else:
        action_representation = str(
            norm_meta.get("action_representation", "relative")
        )
    absolute_action_quaternion_order = str(
        getattr(job_config, "absolute_action_quaternion_order", "auto")
    ).lower()
    if absolute_action_quaternion_order not in ("auto", "wxyz", "xyzw"):
        raise ValueError(
            "absolute_action_quaternion_order must be one of "
            f"auto/wxyz/xyzw, got {absolute_action_quaternion_order!r}"
        )
    if action_representation == "absolute" and absolute_action_quaternion_order != "auto":
        action_quaternion_order = absolute_action_quaternion_order
    elif action_representation == "absolute":
        # Absolute-action training slices raw pose channels directly, so the model
        # action quaternion order stays the dataset/source order. The R1 absolute
        # checkpoint source pose order is wxyz even though relative-action metadata
        # advertises ACTION_QUATERNION_ORDER=xyzw.
        action_quaternion_order = str(
            norm_meta.get(
                "quaternion_order",
                getattr(job_config, "quaternion_order", "xyzw"),
            )
        )
    else:
        action_quaternion_order = str(
            norm_meta.get(
                "action_quaternion_order",
                getattr(job_config, "action_quaternion_order", "xyzw"),
            )
        )
    metadata = {
        "policy": "LingBot-VA",
        "action_condition_mode": str(job_config.action_condition_mode),
        "zero_action_condition": bool(
            getattr(job_config, "zero_action_condition", False)
        ),
        "zero_action_condition_uses_missing_embedding": bool(
            getattr(job_config, "zero_action_condition_uses_missing_embedding", False)
        ),
        "learnable_action_embedding_config": _normalize_learnable_action_embedding_config(
            getattr(job_config, "learnable_action_embedding_config", None)
        ),
        "model_structure": str(job_config.model_structure),
        "transformer_path": str(job_config.transformer_path),
        "runtime_config_path": str(job_config.runtime_config_path),
        "temporal_metadata_path": str(getattr(job_config, "temporal_metadata_path", "")),
        "frame_chunk_size": int(job_config.frame_chunk_size),
        "source_action_per_frame": int(model.source_action_per_frame),
        "source_video_frame_stride": int(model.source_video_frame_stride),
        "video_frames_per_latent": int(model.video_frames_per_latent),
        "actions_per_latent_frame": int(model.actions_per_latent_frame),
        "sampled_video_fps": getattr(job_config, "sampled_video_fps", None),
        "relative_pose_frame": str(getattr(job_config, "relative_pose_frame", "local_frame")),
        "action_representation": action_representation,
        "action_quaternion_order": action_quaternion_order,
    }
    server = LingBotVAWebsocketServer(
        model,
        host=str(job_config.host),
        port=int(job_config.port),
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
