#!/usr/bin/env python3
"""Run read-only reverse FastWAM inference and emit temporal action curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.fastwam_offline_utils import (  # noqa: E402
    ObservationDataset,
    batch_child_argv,
    input_tensor_preview,
    parse_inference_indices,
    relative_action_to_absolute,
    tcp_state16_to_umi_state,
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
    binding_policy_devices,
    read_observation_binding,
)


CHANNEL_NAMES = (
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "right_gripper",
)

_CACHED_RUNTIME: tuple[str, Any] | None = None
_LOCAL_TEXT_CACHE = REPO_ROOT / ".superinference" / "local" / "fastwam-text-cache"


def _load_cached_runtime(runtime_class: Any, params: dict[str, Any]) -> Any:
    """Keep one checkpoint resident while a batch of observations is evaluated."""
    global _CACHED_RUNTIME
    signature = json.dumps(params, sort_keys=True, default=str)
    if _CACHED_RUNTIME is None:
        _CACHED_RUNTIME = (signature, runtime_class(params, load_model=True))
    elif _CACHED_RUNTIME[0] != signature:
        raise RuntimeError("FastWAM batch attempted to change its runtime contract")
    return _CACHED_RUNTIME[1]


def _formatted_prompt(prompt: str, prompt_template: str) -> str:
    task = str(prompt).strip()
    if task.startswith("A video recorded from a robot"):
        return task
    return prompt_template.format(task=task)


def _training_text_cache_dir(checkpoint_root: Path) -> Path | None:
    import yaml

    config_path = checkpoint_root / "config.yaml"
    if not config_path.is_file():
        return None
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        configured = payload["datasets"]["vla_data"].get("text_embedding_cache_dir")
    except (KeyError, OSError, TypeError, yaml.YAMLError):
        return None
    if not configured:
        return None
    return Path(str(configured)).expanduser()


def _prompt_cache_directories(checkpoint_root: Path) -> list[Path]:
    directories = [_LOCAL_TEXT_CACHE]
    for variable in ("FASTWAM_LOCAL_TEXT_CACHE_DIR", "FASTWAM_TEXT_CACHE_DIR"):
        value = os.environ.get(variable)
        if value:
            directories.append(Path(value).expanduser())

    training_dir = _training_text_cache_dir(checkpoint_root)
    if training_dir is not None:
        directories.append(training_dir)
        try:
            relative = training_dir.relative_to("/mnt/workspace")
        except ValueError:
            pass
        else:
            directories.extend(
                (Path.home() / "workspace" / relative, Path.home() / relative)
            )

    directories.append(REPO_ROOT / "logs" / "fastwam_comparisons" / "_text_context_cache")
    try:
        siblings = tuple(REPO_ROOT.parent.iterdir())
    except OSError:
        siblings = ()
    for sibling in siblings:
        if sibling.is_dir() and sibling != REPO_ROOT:
            directories.append(
                sibling / "logs" / "fastwam_comparisons" / "_text_context_cache"
            )

    unique: list[Path] = []
    for directory in directories:
        resolved = directory.expanduser()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _load_prompt_payload(path: Path, context_len: int, torch: Any) -> tuple[Any, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid FastWAM prompt cache payload: {path}")
    context = payload.get("context")
    mask = payload.get("context_mask", payload.get("mask"))
    if not torch.is_tensor(context) or not torch.is_tensor(mask):
        raise ValueError(f"FastWAM prompt cache is missing context or mask: {path}")
    if context.ndim == 3 and context.shape[0] == 1:
        context = context[0]
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if context.ndim != 2 or context.shape[0] != int(context_len):
        raise ValueError(f"FastWAM prompt cache has an invalid context shape: {path}")
    if mask.ndim != 1 or mask.shape[0] != context.shape[0]:
        raise ValueError(f"FastWAM prompt cache has an invalid mask shape: {path}")
    context = context.detach().cpu().clone()
    mask = mask.detach().cpu().bool().clone()
    context[~mask] = 0.0
    return context, torch.ones_like(mask)


def _load_precomputed_prompt(
    checkpoint_root: Path,
    formatted_prompt: str,
    context_len: int,
    torch: Any,
) -> tuple[Any, Any, str] | None:
    filename = (
        hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
        + f".t5_len{int(context_len)}.wan22ti2v5b.pt"
    )
    target = _LOCAL_TEXT_CACHE / filename
    for directory in _prompt_cache_directories(checkpoint_root):
        candidate = directory / filename
        if not candidate.is_file():
            continue
        context, mask = _load_prompt_payload(candidate, context_len, torch)
        if candidate != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                shutil.copyfile(candidate, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return context, mask, "local_cache" if candidate == target else "discovered_cache"
    return None


def _store_prompt_cache(
    formatted_prompt: str,
    context_len: int,
    context: Any,
    mask: Any,
    torch: Any,
) -> None:
    filename = (
        hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
        + f".t5_len{int(context_len)}.wan22ti2v5b.pt"
    )
    target = _LOCAL_TEXT_CACHE / filename
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "context": context.detach().cpu(),
                "mask": mask.detach().cpu().bool(),
            },
            temporary,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def temporal_order_audit(action: np.ndarray, state16: np.ndarray) -> dict[str, Any]:
    values = np.asarray(action, dtype=np.float64)
    state = np.asarray(state16, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 16:
        raise ValueError(f"Expected action [T,16] with T >= 2, got {values.shape}")
    if state.size < 16:
        raise ValueError(f"Expected state with at least 16 values, got {state.shape}")

    arms: dict[str, Any] = {}
    for arm, base in (("left", 0), ("right", 8)):
        anchor = state[base : base + 3]
        first_distance = float(np.linalg.norm(values[0, base : base + 3] - anchor))
        last_distance = float(np.linalg.norm(values[-1, base : base + 3] - anchor))
        arms[arm] = {
            "anchor_xyz": anchor.tolist(),
            "runtime_first_distance_m": first_distance,
            "runtime_last_distance_m": last_distance,
            "runtime_first_is_closer": first_distance <= last_distance,
            "recommended_order": (
                "runtime" if first_distance <= last_distance else "time_reversed"
            ),
        }

    recommendations = {
        str(record["recommended_order"]) for record in arms.values()
    }
    recommended = recommendations.pop() if len(recommendations) == 1 else "mixed"
    return {
        "heuristic": (
            "the first executable pose should be closer to the current state "
            "than the final pose"
        ),
        "recommended_execution_order": recommended,
        "runtime_order_looks_chronological": recommended == "runtime",
        "arms": arms,
    }


def _plot_action_channels(
    path: Path,
    runtime_action: np.ndarray,
    reversed_action: np.ndarray,
    *,
    control_fps: float,
) -> None:
    time_axis = np.arange(runtime_action.shape[0], dtype=np.float64) / control_fps
    figure, axes = plt.subplots(4, 4, figsize=(18, 13), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(time_axis, runtime_action[:, index], color="#c2413b", label="runtime")
        axis.plot(
            time_axis,
            reversed_action[:, index],
            color="#147d92",
            linestyle="--",
            label="time reversed",
        )
        axis.set_title(CHANNEL_NAMES[index])
        axis.grid(alpha=0.25)
        if index // 4 == 3:
            axis.set_xlabel("seconds")
    axes.flat[0].legend(loc="best")
    figure.suptitle("Reverse FastWAM action channels", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_absolute_xyz(
    path: Path,
    runtime_action: np.ndarray,
    reversed_action: np.ndarray,
    state16: np.ndarray,
    *,
    control_fps: float,
) -> None:
    time_axis = np.arange(runtime_action.shape[0], dtype=np.float64) / control_fps
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for row, (arm, base) in enumerate((("left", 0), ("right", 8))):
        for column, label in enumerate(("x", "y", "z")):
            axis = axes[row, column]
            axis.plot(
                time_axis,
                runtime_action[:, base + column],
                color="#c2413b",
                label="runtime",
            )
            axis.plot(
                time_axis,
                reversed_action[:, base + column],
                color="#147d92",
                linestyle="--",
                label="time reversed",
            )
            axis.axhline(
                float(state16[base + column]),
                color="#454545",
                linewidth=1,
                alpha=0.7,
            )
            axis.set_title(f"{arm} {label}")
            axis.set_xlabel("seconds")
            axis.set_ylabel("metres")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.suptitle("Absolute TCP trajectory and current-state anchors", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_trajectory_3d(
    path: Path,
    runtime_action: np.ndarray,
    reversed_action: np.ndarray,
    state16: np.ndarray,
) -> None:
    figure = plt.figure(figsize=(14, 7))
    for index, (arm, base) in enumerate((("left", 0), ("right", 8)), start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        axis.plot(
            *runtime_action[:, base : base + 3].T,
            color="#c2413b",
            label="runtime",
        )
        axis.plot(
            *reversed_action[:, base : base + 3].T,
            color="#147d92",
            linestyle="--",
            label="time reversed",
        )
        axis.scatter(
            *state16[base : base + 3],
            color="#111111",
            s=45,
            label="current state",
        )
        axis.scatter(
            *runtime_action[0, base : base + 3],
            color="#c2413b",
            marker="o",
            s=35,
        )
        axis.scatter(
            *runtime_action[-1, base : base + 3],
            color="#c2413b",
            marker="x",
            s=45,
        )
        axis.set_title(f"{arm} TCP")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _runtime_params(args: argparse.Namespace) -> dict[str, Any]:
    from consumer.inference.fastwam_server import _load_superinference_config

    root_cfg = _load_superinference_config(args.config, args.config_name)
    policy_cfg = dict(root_cfg.get("policy", {}) or {})
    params = dict(policy_cfg.get("params", {}) or {})
    if args.checkpoint is not None:
        params["ckpt_path"] = str(args.checkpoint.expanduser().resolve())
    if args.device is not None:
        params["device"] = args.device
    params["num_inference_steps"] = int(args.num_inference_steps)
    video_requested = args.output_mode == "trajectory_video"
    params["video_visualization"] = {
        "enabled": video_requested,
        "save_predictions": False,
        "max_frames": 49,
        "display_fps": 7.5,
    }
    params["warm_up_on_load"] = False
    params["return_normalized_action"] = True
    return params


def _dataset_from_binding(args: argparse.Namespace) -> tuple[ObservationDataset, Any | None]:
    """Use an executor-owned target-to-recorder binding when one is supplied."""
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


def _bound_devices(binding: Any) -> tuple[list[str], str]:
    """Resolve policy-facing device keys in canonical recorder-role order."""
    return binding_policy_devices(binding)


def main(argv: list[str] | None = None) -> None:
    # Keep pure audit helpers importable from the lightweight control/config
    # environment.  Torch and the NMX stack are needed only for model loading.
    import torch

    from consumer.inference.fastwam_server import (
        DEFAULT_PROMPT_TEMPLATE,
        FastWAMRuntime,
        _resolve_checkpoint,
        _write_rgb_mp4,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--config-name",
        default="policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--inference-index", type=int, default=-1)
    parser.add_argument("--inference-indices", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument(
        "--output-mode",
        choices=("trajectory_only", "trajectory_video"),
        default="trajectory_only",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-diagnostic-plots", action="store_true")
    parser.add_argument("--require-chronological", action="store_true")
    parser.add_argument("--observation-binding", type=Path, default=None)
    parser.add_argument("--binding-index", type=int, default=0)
    parser.add_argument("--action-calibration-snapshot", type=Path, default=None)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
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
    if not np.isfinite(args.control_fps) or args.control_fps <= 0:
        raise ValueError("control-fps must be a positive finite number")
    action_step_interval_ns = int(round(1e9 / args.control_fps))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset, binding = _dataset_from_binding(args)
    inference_index = args.inference_index
    if inference_index < 0:
        inference_index += len(dataset.inferences)
    if inference_index < 0 or inference_index >= len(dataset.inferences):
        raise IndexError(
            f"inference-index {args.inference_index} is outside the recorded run"
        )

    params = _runtime_params(args)
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
        params.update(calibration_overrides)
    checkpoint_root, checkpoint_weight = _resolve_checkpoint(str(params["ckpt_path"]))
    prompt = args.prompt or str(params.get("task") or "")
    prompt_template = str(params.get("prompt_template", DEFAULT_PROMPT_TEMPLATE))
    formatted_prompt = _formatted_prompt(prompt, prompt_template)
    context_len = int(params.get("context_len", 128))
    params["cache_prompt_embeddings"] = True
    cached_prompt = _load_precomputed_prompt(
        checkpoint_root,
        formatted_prompt,
        context_len,
        torch,
    )
    if cached_prompt is not None:
        params["load_text_encoder"] = False
    runtime = _load_cached_runtime(FastWAMRuntime, params)
    prompt_conditioning = "text_encoder"
    if cached_prompt is not None:
        context, context_mask, prompt_conditioning = cached_prompt
        runtime._prompt_embedding_cache[formatted_prompt] = (context, context_mask)
        print(
            f"Using precomputed FastWAM T5 context ({prompt_conditioning})",
            flush=True,
        )
    metadata = runtime.metadata()
    if bool(metadata.get("reverse_temporal_chunks", False)):
        raise RuntimeError(
            "Deployment still reverses temporal chunks before execution"
        )

    if binding is None:
        camera_devices = list(dataset.manifest.get("camera_devices", []))
        robot_device = str(dataset.manifest.get("robot_device") or "RizonRobot_2")
    else:
        camera_devices, robot_device = _bound_devices(binding)
    if len(camera_devices) != 2:
        raise ValueError(f"Expected exactly two camera devices, got {camera_devices}")
    history_frames = int(runtime.contract.get("condition_history_frames", 1))
    history_stride = int(runtime.contract.get("condition_history_stride", 1))
    image_history, tcp_state16, history_timestamps, observation_timestamp_ns = (
        dataset.inference_observation_at(
            inference_index,
            history_frames=history_frames,
            history_stride=history_stride,
            camera_devices=camera_devices,
            robot_device=robot_device,
        )
    )
    tcp_umi_json_path = params.get("tcp_umi_json_path")
    umi_state16 = tcp_state16_to_umi_state(
        tcp_state16,
        tcp_umi_json_path=tcp_umi_json_path,
    )

    with torch.no_grad():
        if history_frames == 1:
            input_image = runtime._preprocess_images(image_history[-1])
            input_video = None
        else:
            input_image = None
            input_video = runtime._preprocess_image_history(
                image_history,
                history_frame_timestamps_ns=history_timestamps,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        observation: dict[str, Any] = {
            "prompt": prompt,
            "state": umi_state16,
            "seed": int(args.seed),
            "timestamp_ns": observation_timestamp_ns,
            "history_frame_timestamps_ns": history_timestamps,
            "return_video_prediction": args.output_mode == "trajectory_video",
            "video_prediction_max_frames": 49,
        }
        if history_frames == 1:
            observation["images"] = image_history[-1]
        else:
            observation["image_history"] = image_history
        result = runtime.infer(observation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started

    if cached_prompt is None:
        generated = runtime._prompt_embedding_cache.get(formatted_prompt)
        if generated is not None:
            _store_prompt_cache(
                formatted_prompt,
                context_len,
                generated[0],
                generated[1],
                torch,
            )

    if result.get("status") != "ok":
        raise RuntimeError(f"FastWAM runtime inference failed: {result}")
    result_data = dict(result["data"])
    relative_runtime = np.asarray(result_data["action"], dtype=np.float64)
    normalized_runtime = np.asarray(
        result_data.get("normalized_action", relative_runtime), dtype=np.float64
    )
    absolute_runtime = relative_action_to_absolute(
        relative_runtime,
        tcp_state16,
        relative_pose_frame=str(runtime.contract.get("relative_pose_frame", "local_frame")),
        action_quaternion_order=str(runtime.contract.get("action_quaternion_order", "xyzw")),
        gripper_max_width=float(params.get("gripper_max_width", 0.0925319933860716)),
        normalize_gripper_width=bool(params.get("normalize_gripper_width", False)),
        clip_gripper_width=bool(params.get("clip_gripper_width", True)),
        tcp_umi_json_path=tcp_umi_json_path,
    )
    normalized_reversed = normalized_runtime[::-1].copy()
    relative_reversed = relative_runtime[::-1].copy()
    absolute_reversed = absolute_runtime[::-1].copy()
    audit = temporal_order_audit(absolute_runtime, tcp_state16)
    prediction_path: str | None = None
    video_prediction = result_data.get("video_prediction")
    if isinstance(video_prediction, dict):
        encoded_frames = video_prediction.get("frames")
        if isinstance(encoded_frames, list) and encoded_frames:
            from PIL import Image

            decoded_frames = [
                np.asarray(Image.open(BytesIO(bytes(frame))).convert("RGB"))
                for frame in encoded_frames
            ]
            prediction_path = "prediction.mp4"
            _write_rgb_mp4(
                output_dir / prediction_path,
                decoded_frames,
                fps=7.5,
            )

    np.savez_compressed(
        output_dir / "offline_result.npz",
        normalized_runtime=normalized_runtime,
        normalized_time_reversed=normalized_reversed,
        relative_runtime=relative_runtime,
        relative_time_reversed=relative_reversed,
        absolute_runtime=absolute_runtime,
        absolute_time_reversed=absolute_reversed,
        state=tcp_state16,
        tcp_state=tcp_state16,
        umi_state=umi_state16,
        channel_names=np.asarray(CHANNEL_NAMES),
        history_timestamps_ns=np.asarray(history_timestamps, dtype=np.int64),
        observation_timestamp_ns=np.asarray(
            observation_timestamp_ns, dtype=np.int64
        ),
    )
    preview = input_tensor_preview(input_image, input_video)
    write_image(output_dir / "input_grid.jpg", preview)
    if not args.skip_diagnostic_plots:
        _plot_action_channels(
            output_dir / "action_channels.png",
            relative_runtime,
            relative_reversed,
            control_fps=args.control_fps,
        )
        _plot_absolute_xyz(
            output_dir / "absolute_xyz.png",
            absolute_runtime,
            absolute_reversed,
            tcp_state16,
            control_fps=args.control_fps,
        )
        _plot_trajectory_3d(
            output_dir / "trajectory_3d.png",
            absolute_runtime,
            absolute_reversed,
            tcp_state16,
        )

    summary = {
        "schema_version": 1,
        "config_name": args.config_name,
        "config_path": args.config,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_override": (None if args.checkpoint is None else str(args.checkpoint.expanduser().resolve())),
        "checkpoint_weight": str(checkpoint_weight),
        "observation_dir": str(args.observation_dir.expanduser().resolve()),
        "inference_index": inference_index,
        "observation_timestamp_ns": observation_timestamp_ns,
        "camera_devices": camera_devices,
        "robot_device": robot_device,
        "tcp_umi_json_path": tcp_umi_json_path,
        "prompt": prompt,
        "prompt_conditioning": prompt_conditioning,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "action_shape": list(absolute_runtime.shape),
        "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
        "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
        "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
        "action_units": DUAL_ARM_TCP_ACTION_UNITS,
        "action_step_interval_ns": action_step_interval_ns,
        "finite": bool(np.isfinite(absolute_runtime).all()),
        "inference_seconds": inference_seconds,
        "output_mode": args.output_mode,
        "video_prediction_available": prediction_path is not None,
        "reverse_temporal_chunks": bool(metadata.get("reverse_temporal_chunks")),
        "runtime_metadata": metadata,
        "temporal_order_audit": audit,
        "artifacts": {
            "npz": "offline_result.npz",
            "canonical_action_key": "absolute_runtime",
            "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
            "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
            "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
            "action_units": DUAL_ARM_TCP_ACTION_UNITS,
            "action_step_interval_ns": action_step_interval_ns,
            "input_preview": "input_grid.jpg",
        },
    }
    if not args.skip_diagnostic_plots:
        summary["artifacts"].update(
            {
                "action_channels": "action_channels.png",
                "absolute_xyz": "absolute_xyz.png",
                "trajectory_3d": "trajectory_3d.png",
            }
        )
    if prediction_path is not None:
        summary["artifacts"]["video_prediction"] = prediction_path
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
        summary["artifacts"]["source_profile"] = binding.source_profile
        summary["artifacts"]["source_provenance_digest"] = binding.source_provenance_digest
        summary["artifacts"]["recording_artifact_digest"] = binding.recording_artifact_digest
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if args.require_chronological and not audit["runtime_order_looks_chronological"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
