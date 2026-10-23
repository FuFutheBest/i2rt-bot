#!/usr/bin/env python3
"""Websocket server for native nmx_vla FastWAM inference."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import gc
import hashlib
import http
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.logger_config import logger  # noqa: E402
from utils.inference_profiling import InferenceProfiler  # noqa: E402
from utils.msgpack_numpy import Packer, unpackb  # noqa: E402
from utils import gpu_safety  # noqa: E402
from consumer.inference.fastwam_model_registry import (  # noqa: E402
    validate_fastwam_variant_config,
    validate_fastwam_variant_runtime_contract,
)

_NMX_VLA_ROOT = _REPO_ROOT / "third_party" / "nmx_vla"
if str(_NMX_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_NMX_VLA_ROOT))

DEFAULT_CONFIG_NAME = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
DEFAULT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
READY_FILE_ENV = "SUPERINFERENCE_FASTWAM_READY_FILE"
PROGRESS_FILE_ENV = "SUPERINFERENCE_FASTWAM_PROGRESS_FILE"
RUN_ID_ENV = "SUPERINFERENCE_FASTWAM_RUN_ID"
CONFIG_NAME_ENV = "SUPERINFERENCE_FASTWAM_CONFIG_NAME"
_ACTION_INFER_MODES = {"observed_frame_cache", "one_pass_future_cache"}


def _atomic_write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a small ownership marker without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    descriptor: Optional[int] = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


class _StartupProgressReporter:
    """Write bounded, process-owned startup progress for Control Panel polling."""

    def __init__(
        self,
        path: Optional[Path],
        *,
        run_id: Optional[str],
        config_name: Optional[str],
    ) -> None:
        self.path = path
        self.run_id = str(run_id or "")
        self.config_name = str(config_name or "")
        self.started_at = time.time()
        self.events: list[Dict[str, Any]] = []

    def update(self, phase: str) -> None:
        if self.path is None:
            return
        now = time.time()
        event = {
            "phase": str(phase),
            "updated_at": now,
            "elapsed_ms": max(0, int((now - self.started_at) * 1000.0)),
        }
        self.events.append(event)
        try:
            _atomic_write_private_json(
                self.path,
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "pid": os.getpid(),
                    "config_name": self.config_name,
                    "phase": str(phase),
                    "started_at": self.started_at,
                    "updated_at": now,
                    "events": self.events[-32:],
                },
            )
        except OSError as exc:
            logger.warning(f"Unable to publish FastWAM startup progress: {exc}")


class _LoguruBridgeHandler(logging.Handler):
    """Forward nmx_vla's stdlib logging, including tracebacks, to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(exception=record.exc_info, depth=6).log(
            level,
            f"[{record.name}] {record.getMessage()}",
        )


def _install_nmx_vla_log_bridge() -> None:
    """Make swallowed compile fallback details visible in logs/app.log."""
    nmx_logger = logging.getLogger("nmx_vla")
    if not any(
        isinstance(handler, _LoguruBridgeHandler)
        for handler in nmx_logger.handlers
    ):
        nmx_logger.addHandler(_LoguruBridgeHandler())
    nmx_logger.setLevel(logging.INFO)
    nmx_logger.propagate = False


def _as_uint8_video_frame(frame: Any) -> np.ndarray:
    """Convert a decoded model frame to a contiguous RGB uint8 image."""
    array = np.asarray(frame)
    if array.size == 0:
        raise ValueError("Video prediction frame is empty")
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected a video frame with HWC/CHW image data, got {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.min()) >= -1.0 and float(finite.max()) <= 1.0:
            array = (array + 1.0) * 127.5 if float(finite.min()) < 0.0 else array * 255.0
    return np.ascontiguousarray(np.clip(array, 0.0, 255.0).astype(np.uint8))


def _video_frame_sequence(video: Any) -> list[Any]:
    """Normalize common video tensor layouts into a time-major frame list."""
    if video is None:
        return []
    if hasattr(video, "detach"):
        # ``bfloat16`` tensors cannot be converted to NumPy directly.
        video = video.detach().cpu().float().numpy()
    if isinstance(video, (list, tuple)):
        return list(video)

    array = np.asarray(video)
    if array.size == 0:
        return []
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError(f"Expected a single video batch, got {array.shape}")
        array = array[0]
    if array.ndim == 4:
        if array.shape[-1] in (1, 3, 4):  # [T,H,W,C]
            return list(array)
        if array.shape[1] in (1, 3, 4):  # [T,C,H,W]
            return [np.moveaxis(frame, 0, -1) for frame in array]
        if array.shape[0] in (1, 3, 4):  # [C,T,H,W]
            return [np.moveaxis(array[:, index], 0, -1) for index in range(array.shape[1])]
        raise ValueError(f"Expected a time-major video tensor, got {array.shape}")
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):  # one HWC frame
            return [array]
        if array.shape[0] in (1, 3, 4):  # one CHW frame
            return [np.moveaxis(array, 0, -1)]
        return list(array)  # grayscale [T,H,W]
    if array.ndim == 2:
        return [array]
    raise ValueError(f"Expected video frames, got {array.shape}")


def _encode_video_prediction_frames(
    video: Any,
    *,
    max_frames: int = 16,
    jpeg_quality: int = 85,
) -> list[bytes]:
    """Sample and JPEG-encode model video frames for websocket transport."""
    frames = _video_frame_sequence(video)
    if not frames:
        return []
    limit = len(frames) if int(max_frames) <= 0 else min(len(frames), int(max_frames))
    indices = np.linspace(0, len(frames) - 1, num=limit, dtype=np.int64)
    quality = max(20, min(int(jpeg_quality), 95))
    try:
        from PIL import Image

        encoded: list[bytes] = []
        for index in indices:
            buffer = BytesIO()
            Image.fromarray(_as_uint8_video_frame(frames[int(index)])).save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=False,
            )
            encoded.append(buffer.getvalue())
        return encoded
    except Exception as exc:
        raise RuntimeError(f"Failed to encode FastWAM video prediction frames: {exc}") from exc


def _write_rgb_mp4(path: Path, frames: list[Any], fps: float) -> None:
    """Write browser-compatible H.264 RGB prediction frames to an MP4 artifact."""
    if fps <= 0:
        raise ValueError(f"Video save fps must be positive, got {fps}")
    normalized = [_as_uint8_video_frame(frame) for frame in frames]
    if not normalized:
        raise ValueError("Cannot save an empty FastWAM video prediction")
    height, width = normalized[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in normalized):
        raise ValueError("FastWAM video prediction frames have inconsistent shapes")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to save browser-compatible FastWAM MP4 video")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp.mp4"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(float(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            input=b"".join(frame.tobytes(order="C") for frame in normalized),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(30.0, (len(normalized) / float(fps)) * 5.0),
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"FFmpeg could not encode FastWAM MP4 video: {detail or 'unknown error'}"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg produced no FastWAM MP4 data for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_dtype(value: Any) -> torch.dtype:
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


def _to_plain_config(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _select_plain(cfg: DictConfig, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off", "none", "null"}:
        return False
    return bool(value)


def _normalize_action_infer_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in _ACTION_INFER_MODES:
        raise ValueError(
            "action_infer_mode must be one of "
            f"{sorted(_ACTION_INFER_MODES)}, got {value!r}"
        )
    return mode


def _normalize_bind_host(host: Any) -> str:
    """Strip URL-only IPv6 brackets before binding a socket."""
    normalized = str(host).strip()
    if normalized.startswith('[') and normalized.endswith(']'):
        return normalized[1:-1]
    return normalized


def _resolve_zero_state_condition(
    fastwam_cfg: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> bool:
    for source in (params or {}, fastwam_cfg):
        for key in ("zero_state_condition", "zero_action_condition"):
            if key in source and source[key] is not None:
                return _as_bool(source[key])
    return False


def _load_superinference_config(
    config_path: Optional[str],
    config_name: Optional[str],
) -> Dict[str, Any]:
    if config_path:
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        config_root = (_REPO_ROOT / "configs").resolve()
        if path.is_relative_to(config_root):
            native_name = path.relative_to(config_root).with_suffix("").as_posix()
            with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
                cfg = compose(config_name=native_name)
        else:
            with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
                cfg = compose(config_name=path.stem)
    else:
        name = config_name or DEFAULT_CONFIG_NAME
        config_dir = _REPO_ROOT / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(config_name=name)
    return {
        "summary_shm": _select_plain(cfg, "summary_shm", "device_summary_data"),
        "runtime": _select_plain(cfg, "runtime", {}),
        "policy": _select_plain(cfg, "policy", {}),
        "profiling": _select_plain(cfg, "profiling", {}),
    }


def _build_server_params(root_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    policy_cfg = cast(Dict[str, Any], root_cfg.get("policy", {}))
    params = dict(policy_cfg.get("params", {}) or {})
    server_cfg = dict(params.pop("server", {}) or {})
    params.update(server_cfg)

    for key in (
        "host",
        "port",
        "ckpt_path",
        "device",
        "action_infer_mode",
        "action_video_attention_mode",
    ):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if args.num_inference_steps is not None:
        params["num_inference_steps"] = int(args.num_inference_steps)
    if getattr(args, "rtc_use_vjp", None) is not None:
        params["rtc_use_vjp"] = bool(args.rtc_use_vjp)
    for key in (
        "c4_video_denoise_steps",
        "c4_video_num_inference_steps",
        "c4_reverse_video_timestep",
    ):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    return params


def _checkpoint_completion_marker_path(weight_path: Path) -> Path:
    exact = weight_path.with_name(f"{weight_path.stem}.complete.json")
    if exact.is_file() or not weight_path.stem.endswith("_last"):
        return exact
    return weight_path.with_name(
        f"{weight_path.stem.removesuffix('_last')}.complete.json"
    )


def _select_latest_weight(weights_dir: Path, *, prefer_completed: bool = False) -> Path:
    pattern = re.compile(r"step_(\d+)(?:_last)?\.pt$", flags=re.IGNORECASE)
    candidates: list[tuple[int, int, Path]] = []
    weight_paths = [
        path
        for path in weights_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == '.pt'
    ] if weights_dir.is_dir() else []
    for path in weight_paths:
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), int(path.stem.endswith("_last")), path))
    if not candidates:
        all_pt = sorted(
            weight_paths,
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not all_pt:
            raise FileNotFoundError(f"No .pt checkpoint found in {weights_dir}")
        return all_pt[0]
    if prefer_completed:
        completed = [
            item for item in candidates
            if _checkpoint_completion_marker_path(item[2]).is_file()
        ]
        if completed:
            candidates = completed
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _resolve_checkpoint(
    ckpt_path: str,
    *,
    prefer_completed: bool = False,
) -> Tuple[Path, Path]:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FastWAM checkpoint path does not exist: {path}")
    if path.is_file():
        if path.suffix.casefold() != '.pt':
            raise ValueError(
                f"FastWAM checkpoint file must have a .pt suffix: {path}"
            )
        weight_path = path
        if path.parent.name == "weights" and path.parent.parent.name == "checkpoints":
            ckpt_root = path.parent.parent.parent
        else:
            ckpt_root = path.parent
    else:
        ckpt_root = path
        weights_dir = ckpt_root / "checkpoints" / "weights"
        if weights_dir.is_dir():
            weight_path = _select_latest_weight(
                weights_dir,
                prefer_completed=prefer_completed,
            )
        elif (ckpt_root / "checkpoint.pt").is_file():
            weight_path = ckpt_root / "checkpoint.pt"
        else:
            raise FileNotFoundError(
                f"No FastWAM weight found in {ckpt_root}; expected "
                "checkpoints/weights/*.pt or checkpoint.pt"
            )

    if not (ckpt_root / "config.yaml").is_file():
        raise FileNotFoundError(f"Missing training config.yaml in {ckpt_root}")
    if not (ckpt_root / "dataset_statistics.json").is_file():
        raise FileNotFoundError(f"Missing dataset_statistics.json in {ckpt_root}")
    return ckpt_root, weight_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maybe_validate_checkpoint_completion_marker(
    ckpt_root: Path,
    weight_path: Path,
    params: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _as_bool(params.get("require_checkpoint_completion_marker", False)):
        return None
    marker_path = _checkpoint_completion_marker_path(weight_path)
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"FastWAM checkpoint completion marker is required but missing: {marker_path}"
        )
    marker = _load_json(marker_path)
    if marker.get("status") != "complete":
        raise ValueError(f"FastWAM checkpoint marker is not complete: {marker_path}")
    match = re.fullmatch(r"step_(\d+)(?:_last)?", weight_path.stem)
    if match is not None and int(marker.get("global_step", -1)) != int(match.group(1)):
        raise ValueError("FastWAM checkpoint marker global_step does not match weight filename")
    recorded_weight = marker.get("weights_path")
    if recorded_weight and Path(str(recorded_weight)).name != weight_path.name:
        raise ValueError(
            "FastWAM checkpoint marker points to a different weight file: "
            f"{recorded_weight!r} vs {weight_path.name!r}"
        )
    expected_size = int(marker.get("weights_size_bytes", -1))
    if expected_size != weight_path.stat().st_size:
        raise ValueError("FastWAM checkpoint size does not match completion marker")
    config_sha256 = str(marker.get("config_sha256", "")).strip().lower()
    if config_sha256 and _sha256_file(ckpt_root / "config.yaml") != config_sha256:
        raise ValueError("FastWAM config.yaml SHA-256 does not match completion marker")
    if _as_bool(params.get("verify_checkpoint_sha256", False)):
        expected_sha256 = str(marker.get("weights_sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("FastWAM checkpoint marker has no valid weights_sha256")
        if _sha256_file(weight_path) != expected_sha256:
            raise ValueError("FastWAM checkpoint SHA-256 does not match completion marker")
    return {**marker, "marker_path": str(marker_path)}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _numeric_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


class SingleFieldLinearNormalizer:
    """Numpy mirror of nmx_vla WAMProcessor's single-field normalizer."""

    RANGE_TOL = 1e-4
    OUTPUT_MIN = -1.0
    OUTPUT_MAX = 1.0

    def __init__(self, stats: Dict[str, Any], mode: str) -> None:
        self.mode = str(mode)
        parsed_stats = {key: _numeric_array(value) for key, value in stats.items()}

        if self.mode == "z-score":
            input_mean = parsed_stats["mean"]
            input_std = parsed_stats["std"]
            self.scale = 1.0 / (input_std + 1e-8)
            self.offset = -input_mean / (input_std + 1e-8)
        else:
            if self.mode == "min/max":
                input_min = parsed_stats["min"]
                input_max = parsed_stats["max"]
            elif self.mode == "q01/q99":
                input_min = parsed_stats["q01"]
                input_max = parsed_stats["q99"]
            else:
                lo, hi = map(float, self.mode.split("/"))
                input_min = np.full_like(parsed_stats["min"], lo, dtype=np.float64)
                input_max = np.full_like(parsed_stats["max"], hi, dtype=np.float64)

            input_range = input_max - input_min
            ignore_dim = input_range < self.RANGE_TOL
            safe_range = input_range.copy()
            safe_range[ignore_dim] = self.OUTPUT_MAX - self.OUTPUT_MIN
            self.scale = (self.OUTPUT_MAX - self.OUTPUT_MIN) / safe_range
            self.offset = self.OUTPUT_MIN - self.scale * input_min
            self.offset[ignore_dim] = (
                self.OUTPUT_MAX + self.OUTPUT_MIN
            ) / 2.0 - input_min[ignore_dim]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float64) * self.scale + self.offset, -5.0, 5.0)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.offset) / self.scale


def _select_field_stats(
    dataset_stats: Dict[str, Any],
    field: str,
    key: str,
    *,
    use_stepwise: bool,
) -> Dict[str, Any]:
    prefix = "stepwise_" if use_stepwise and field == "action" else "global_"
    stats = dataset_stats[field][key]
    selected = {
        name.removeprefix(prefix): value
        for name, value in stats.items()
        if name.startswith(prefix)
    }
    if not selected:
        raise KeyError(f"No {prefix} stats found for {field}.{key}")
    return selected


class FastWAMRuntime:
    """Native FastWAM inference runtime."""

    def __init__(
        self,
        params: Dict[str, Any],
        *,
        load_model: bool = True,
        profiler: Optional[InferenceProfiler] = None,
        startup_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.params = dict(params)
        self.model_variant = self.params.get("model_variant")
        if self.model_variant is not None:
            self.model_variant = str(self.model_variant).strip().lower()
            validate_fastwam_variant_config(self.model_variant, self.params)
        self.profiler = profiler
        self._startup_progress = startup_progress
        self.enable_rtc = _as_bool(self.params.get("enable_rtc", False))
        self.rtc_use_vjp = _as_bool(self.params.get("rtc_use_vjp", True))
        self._gpu_device = str(self.params.get("device", "cuda"))
        self._gpu_min_free_mib = self.params.get("gpu_min_free_mib")
        self._gpu_health_check_enabled = bool(
            load_model and gpu_safety.is_cuda_device(self._gpu_device)
        )
        self._initialize_inference_acceleration_settings()
        ckpt_path = self.params.get("ckpt_path")
        if not ckpt_path:
            raise ValueError("FastWAM server requires policy.params.ckpt_path")

        self._report_startup("resolving_checkpoint")
        prefer_completed = _as_bool(
            self.params.get("require_checkpoint_completion_marker", False)
        )
        self.ckpt_root, self.weight_path = _resolve_checkpoint(
            str(ckpt_path),
            prefer_completed=prefer_completed,
        )
        self.checkpoint_completion = _maybe_validate_checkpoint_completion_marker(
            self.ckpt_root,
            self.weight_path,
            self.params,
        )
        self.training_cfg = OmegaConf.load(self.ckpt_root / "config.yaml")
        self.dataset_stats = _load_json(self.ckpt_root / "dataset_statistics.json")
        video_visualization = self.params.get("video_visualization", {})
        self.video_visualization_config: Dict[str, Any]
        if isinstance(video_visualization, bool):
            self.video_visualization_config = {"enabled": video_visualization}
        else:
            self.video_visualization_config = (
                dict(video_visualization)
                if isinstance(video_visualization, Mapping)
                else {}
            )
        self._video_prediction_failure_count = 0
        self._video_prediction_disabled = False
        self._saved_video_prediction_count = 0
        self._last_preprocessed_camera_frames: list[np.ndarray] = []
        self.contract = self._build_contract()
        if self.model_variant:
            validate_fastwam_variant_runtime_contract(
                self.model_variant,
                self.contract,
            )
        self._report_startup("validating_contract")
        self._validate_runtime_alignment()
        self.model = None
        self._nmx_vla_module_path: Optional[str] = None
        self._dump_input_count = 0

        self.action_normalizer = SingleFieldLinearNormalizer(
            _select_field_stats(
                self.dataset_stats,
                "action",
                "default",
                use_stepwise=bool(self.contract["use_stepwise_action_norm"]),
            ),
            str(self.contract["norm_default_mode"]),
        )
        self.state_normalizer = SingleFieldLinearNormalizer(
            _select_field_stats(
                self.dataset_stats,
                "state",
                "default",
                use_stepwise=False,
            ),
            str(self.contract["norm_default_mode"]),
        )
        self._validate_normalizer_contract()

        if load_model:
            self._ensure_gpu_preflight("model startup")
            profiler_context = (
                self.profiler.section("server/model_load")
                if self.profiler is not None
                else nullcontext()
            )
            with profiler_context:
                self._report_startup("building_model")
                self.model = self._load_model()
                self._report_startup("finalizing_model")
                self._configure_inference_acceleration()

    def _report_startup(self, phase: str) -> None:
        callback = getattr(self, "_startup_progress", None)
        if callback is not None:
            callback(phase)

    def _requested_gpu_device(self) -> str:
        return str(getattr(self, "_gpu_device", self.params.get("device", "cuda")))

    def _ensure_gpu_preflight(self, stage: str) -> None:
        if not gpu_safety.is_cuda_device(self._requested_gpu_device()):
            return
        try:
            gpu_safety.GPU_FAULT_LATCH.ensure_healthy()
            result = gpu_safety.gpu_preflight(
                self._requested_gpu_device(),
                min_free_mib=self._gpu_min_free_mib,
                torch_module=torch,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            gpu_safety.GPU_FAULT_LATCH.trip(reason)
            raise RuntimeError(f"GPU preflight failed during {stage}: {reason}") from exc
        if not result.get("ok"):
            reason = str(result.get("error") or "GPU preflight failed")
            gpu_safety.GPU_FAULT_LATCH.trip(reason)
            raise RuntimeError(f"GPU preflight failed during {stage}: {reason}")

    def _ensure_gpu_ready(self) -> None:
        if not gpu_safety.is_cuda_device(self._requested_gpu_device()):
            return
        gpu_safety.GPU_FAULT_LATCH.ensure_healthy()
        if not self._gpu_health_check_enabled:
            return
        result = gpu_safety.gpu_preflight(
            self._requested_gpu_device(),
            check_memory=False,
            torch_module=torch,
        )
        if not result.get("ok"):
            reason = str(result.get("error") or "GPU health check failed")
            gpu_safety.GPU_FAULT_LATCH.trip(reason)
            raise RuntimeError(f"GPU health check failed before inference: {reason}")

    def _record_gpu_failure(self, error: BaseException) -> bool:
        if not gpu_safety.is_cuda_device(self._requested_gpu_device()):
            return False
        reason = gpu_safety.classify_gpu_error(error)
        if reason is None:
            return False
        newly_latched = gpu_safety.GPU_FAULT_LATCH.trip(reason)
        if newly_latched:
            logger.critical(
                f"FastWAM GPU fault latched; restart the server before retrying: {reason}"
            )
        return True

    def gpu_health(self) -> Dict[str, Any]:
        state = gpu_safety.GPU_FAULT_LATCH.snapshot()
        state.update(
            {
                "device": self._requested_gpu_device(),
                "model_loaded": getattr(self, "model", None) is not None,
            }
        )
        state["ok"] = not bool(state["faulted"])
        return state

    def _validate_normalizer_contract(self) -> None:
        action_shape = tuple(self.action_normalizer.scale.shape)
        expected_action_shape = (
            (int(self.contract["action_horizon"]), int(self.contract["action_dim"]))
            if bool(self.contract["use_stepwise_action_norm"])
            else (int(self.contract["action_dim"]),)
        )
        if action_shape != expected_action_shape:
            raise ValueError(
                "FastWAM action statistics shape mismatch: expected "
                f"{expected_action_shape}, got {action_shape}"
            )
        state_shape = tuple(self.state_normalizer.scale.shape)
        expected_state_shape = (int(self.contract["state_dim"]),)
        if state_shape != expected_state_shape:
            raise ValueError(
                "FastWAM state statistics shape mismatch: expected "
                f"{expected_state_shape}, got {state_shape}"
            )

    def _initialize_inference_acceleration_settings(self) -> None:
        """Validate and store inference-only acceleration settings."""
        compile_value = self.params.get("torch_compile", {})
        if compile_value is None:
            compile_value = {}
        if isinstance(compile_value, bool):
            compile_config: Dict[str, Any] = {"enabled": compile_value}
        elif isinstance(compile_value, Mapping):
            compile_config = dict(compile_value)
        else:
            raise TypeError("policy.params.torch_compile must be a boolean or mapping")
        compile_config["enabled"] = _as_bool(compile_config.get("enabled", False))
        self._torch_compile_config = compile_config

        self._prompt_cache_enabled = _as_bool(
            self.params.get("cache_prompt_embeddings", False)
        )
        self._prompt_cache_max_entries = int(
            self.params.get("prompt_cache_max_entries", 8)
        )
        if self._prompt_cache_enabled and self._prompt_cache_max_entries <= 0:
            raise ValueError(
                "prompt_cache_max_entries must be positive when prompt caching is enabled"
            )
        self._prompt_embedding_cache: OrderedDict[
            str,
            Tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()

        self._warm_up_on_load = _as_bool(
            self.params.get("warm_up_on_load", False)
        )
        self._warmup_requests = int(self.params.get("warmup_requests", 1))
        if self._warm_up_on_load and self._warmup_requests <= 0:
            raise ValueError(
                "warmup_requests must be positive when warm_up_on_load is enabled"
            )
        self._warmup_completed_requests = 0

    def _configure_inference_acceleration(self) -> None:
        """Enable requested optimizations and complete startup warmup."""
        if self.model is None:
            return

        if bool(self._torch_compile_config["enabled"]):
            compile_action_denoise = getattr(
                self.model,
                "compile_action_denoise",
                None,
            )
            if not callable(compile_action_denoise):
                raise RuntimeError(
                    "torch_compile.enabled=true requires an nmx_vla FastWAM "
                    "model with compile_action_denoise(); check out the "
                    "fastwam_infer_speed_up submodule branch"
                )
            dynamic_value = self._torch_compile_config.get("dynamic", False)
            dynamic = (
                None
                if dynamic_value is None
                else _as_bool(dynamic_value)
            )
            compile_result = bool(compile_action_denoise(
                mode=str(
                    self._torch_compile_config.get(
                        "mode",
                        "reduce-overhead",
                    )
                ),
                dynamic=dynamic,
                fullgraph=_as_bool(
                    self._torch_compile_config.get("fullgraph", False)
                ),
            ))
            logger.info(
                "FastWAM torch.compile configuration result: "
                f"returned={compile_result} "
                f"active_before_warmup={bool(getattr(self.model, 'action_denoise_compile_enabled', False))} "
                f"config={self._torch_compile_config}"
            )

        if self._warm_up_on_load:
            self._warm_up_model()

        compile_enabled = bool(
            getattr(
                self.model,
                "action_denoise_compile_enabled",
                False,
            )
        )
        if bool(self._torch_compile_config["enabled"]) and not compile_enabled:
            logger.warning(
                "FastWAM torch.compile was requested but is not active; "
                "inference will use the eager action-denoise path"
            )
        logger.info(
            "FastWAM inference acceleration: "
            f"torch_compile_requested={bool(self._torch_compile_config['enabled'])} "
            f"torch_compile_active={compile_enabled} "
            f"prompt_cache_enabled={self._prompt_cache_enabled} "
            f"prompt_cache_entries={len(self._prompt_embedding_cache)} "
            f"warmup_completed={self._warmup_completed_requests}"
        )

    def _resolve_prompt_conditioning(
        self,
        prompt: str,
    ) -> Tuple[
        Optional[str],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Return a prompt or cached T5 conditioning tensors for inference."""
        if not self._prompt_cache_enabled:
            return prompt, None, None
        if self.model is None:
            raise RuntimeError("FastWAM model is not loaded")

        cached = self._prompt_embedding_cache.pop(prompt, None)
        if cached is None:
            with torch.no_grad():
                context, context_mask = self.model.encode_prompt(prompt)
            cached = (context.detach(), context_mask.detach())
            self._prompt_embedding_cache[prompt] = cached
            while (
                len(self._prompt_embedding_cache)
                > self._prompt_cache_max_entries
            ):
                self._prompt_embedding_cache.popitem(last=False)
            logger.info(
                "Cached FastWAM prompt embedding "
                f"({len(self._prompt_embedding_cache)}/"
                f"{self._prompt_cache_max_entries})"
            )
        else:
            self._prompt_embedding_cache[prompt] = cached
        return None, cached[0], cached[1]

    def _infer_model_action(self, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch the checkpoint-selected action inference implementation."""
        if self.model is None:
            raise RuntimeError("FastWAM model is not loaded")

        action_infer_mode = str(
            self.contract.get("action_infer_mode", "observed_frame_cache")
        )
        training_action_infer_mode = str(
            self.contract.get("training_action_infer_mode", action_infer_mode)
        )
        if (
            _as_bool(self.params.get("enforce_checkpoint_infer_mode", False))
            and action_infer_mode != training_action_infer_mode
        ):
            raise ValueError(
                "FastWAM action inference mode mismatch: checkpoint specifies "
                f"{training_action_infer_mode!r}, but runtime requests "
                f"{action_infer_mode!r}"
            )

        if action_infer_mode == "observed_frame_cache":
            infer_method = self.model.infer_action
            # C4 controls belong exclusively to the FasterWAM one-pass path.
            # Do not rely on a normal nmx_vla infer_action implementation
            # accepting unknown kwargs.
            for key in (
                "c4_video_denoise_steps",
                "c4_video_num_inference_steps",
                "c4_reverse_video_timestep",
            ):
                kwargs.pop(key, None)
        else:
            infer_method = getattr(
                self.model,
                "infer_action_one_pass_future_cache",
                None,
            )
            if not callable(infer_method):
                raise RuntimeError(
                    "one_pass_future_cache requires an nmx_vla FastWAM model with "
                    "infer_action_one_pass_future_cache(); update third_party/nmx_vla "
                    "to the pinned FasterWAM-compatible revision"
                )
            kwargs["num_video_frames"] = int(
                self.contract["sampled_video_num_frames"]
            )

        infer_signature = inspect.signature(infer_method)
        accepts_extra_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in infer_signature.parameters.values()
        )
        if action_infer_mode == "one_pass_future_cache" and not accepts_extra_kwargs:
            required_c4_fields = {
                "c4_video_denoise_steps",
                "c4_video_num_inference_steps",
                "c4_reverse_video_timestep",
            }
            unsupported_c4_fields = required_c4_fields.difference(
                infer_signature.parameters
            )
            if unsupported_c4_fields:
                raise RuntimeError(
                    "one_pass_future_cache requires an nmx_vla "
                    "infer_action_one_pass_future_cache implementation that accepts "
                    f"{sorted(unsupported_c4_fields)}"
                )
        if "rtc_use_vjp" in infer_signature.parameters or accepts_extra_kwargs:
            kwargs.setdefault("rtc_use_vjp", getattr(self, "rtc_use_vjp", True))
        return cast(Dict[str, Any], infer_method(**kwargs))

    def _warm_up_model(self) -> None:
        """Run representative requests to compile and stabilize GPU execution."""
        if self.model is None:
            raise RuntimeError("FastWAM model is not loaded")

        prompt = self._format_prompt(self.params.get("task"))
        model_prompt, context, context_mask = self._resolve_prompt_conditioning(
            prompt
        )
        video_h, video_w = (
            int(value) for value in self.contract["video_size"]
        )
        condition_history_frames = int(
            self.contract.get("condition_history_frames", 1)
        )
        if condition_history_frames == 1:
            input_image: Optional[torch.Tensor] = torch.zeros(
                (3, video_h, video_w),
                dtype=torch.float32,
            )
            input_video: Optional[torch.Tensor] = None
        else:
            input_image = None
            input_video = torch.zeros(
                (3, condition_history_frames, video_h, video_w),
                dtype=torch.float32,
            )
        proprio = torch.zeros(
            int(self.contract["state_dim"]),
            dtype=torch.float32,
        )
        action_horizon = int(self.contract["action_horizon"])
        num_inference_steps = int(self.params.get("num_inference_steps", 10))
        seed_value = self.params.get("seed")
        seed = None if seed_value is None else int(seed_value)
        condition_kwargs: Dict[str, Optional[torch.Tensor]] = {
            "input_image": input_image,
        }
        if input_video is not None:
            condition_kwargs["input_video"] = input_video

        logger.info(
            f"Warming up FastWAM inference for {self._warmup_requests} requests"
        )
        for request_index in range(self._warmup_requests):
            start = time.monotonic()
            compile_active_before = bool(
                getattr(self.model, "action_denoise_compile_enabled", False)
            )
            try:
                with torch.no_grad():
                    self._infer_model_action(
                        prompt=model_prompt,
                        context=context,
                        context_mask=context_mask,
                        action_horizon=action_horizon,
                        proprio=proprio,
                        num_inference_steps=num_inference_steps,
                        c4_video_denoise_steps=int(
                            self.params.get("c4_video_denoise_steps", 0)
                        ),
                        c4_video_num_inference_steps=int(
                            self.params.get(
                                "c4_video_num_inference_steps",
                                num_inference_steps,
                            )
                        ),
                        c4_reverse_video_timestep=_as_bool(
                            self.params.get("c4_reverse_video_timestep", False)
                        ),
                        sigma_shift=self.params.get("sigma_shift"),
                        seed=seed,
                        rand_device=str(
                            self.params.get("rand_device", "cpu")
                        ),
                        tiled=_as_bool(self.params.get("tiled", False)),
                        prefix_actions=None,
                        inference_delay=None,
                        prefix_attention_horizon=None,
                        max_guidance_weight=None,
                        profiler=None,
                        **condition_kwargs,
                    )
            except Exception as exc:
                logger.opt(exception=True).error(
                    "FastWAM warmup failed; continuing without additional "
                    f"warmup requests: {type(exc).__name__}: {exc}"
                )
                break
            compile_active_after = bool(
                getattr(self.model, "action_denoise_compile_enabled", False)
            )
            if compile_active_before and not compile_active_after:
                logger.error(
                    "FastWAM compiled action denoiser was disabled during "
                    f"warmup request {request_index + 1}; the preceding nmx_vla "
                    "exception contains the compile failure traceback"
                )
            self._warmup_completed_requests += 1
            logger.info(
                "FastWAM warmup request "
                f"{request_index + 1}/{self._warmup_requests} completed in "
                f"{(time.monotonic() - start) * 1000.0:.2f} ms"
            )

    def _build_contract(self) -> Dict[str, Any]:
        cfg = self.training_cfg
        ds = cfg.datasets.vla_data
        fastwam_cfg = OmegaConf.to_container(cfg.framework.fastwam, resolve=True) or {}
        shape_meta = OmegaConf.to_container(ds.shape_meta, resolve=True)
        processor_cfg = OmegaConf.to_container(ds.get("processor", {}), resolve=True) or {}
        action_transform = OmegaConf.to_container(
            ds.get("action_transform", {}),
            resolve=True,
        ) or {}
        training_enable_video_prediction = _as_bool(
            fastwam_cfg.get("enable_video_prediction", True)
        )

        image_meta = list(shape_meta.get("images", []))
        if len(image_meta) == 0:
            raise ValueError("FastWAM training config has no image shape_meta")

        action_dim = int(processor_cfg.get("action_output_dim", 0))
        if action_dim <= 0:
            action_dim = int(shape_meta["action"][0]["shape"])
        state_dim = int(processor_cfg.get("proprio_output_dim", 0))
        if state_dim <= 0:
            state_dim = int(shape_meta["state"][0]["shape"])

        num_frames = int(ds.get("num_frames", 33))
        if num_frames < 2:
            raise ValueError(
                "FastWAM num_frames must be at least 2, got "
                f"{num_frames}"
            )
        action_video_freq_ratio = int(ds.get("action_video_freq_ratio", 4))
        if action_video_freq_ratio <= 0:
            raise ValueError(
                "FastWAM action_video_freq_ratio must be positive, got "
                f"{action_video_freq_ratio}"
            )
        if (num_frames - 1) % action_video_freq_ratio != 0:
            raise ValueError(
                "FastWAM num_frames-1 must be divisible by action_video_freq_ratio: "
                f"num_frames={num_frames}, ratio={action_video_freq_ratio}"
            )
        video_num_frames = (num_frames - 1) // action_video_freq_ratio + 1
        # Wan's temporal VAE expects one conditioning frame plus transitions
        # in groups of four (T % 4 == 1).
        if (
            _as_bool(self.video_visualization_config.get("enabled", False))
            and training_enable_video_prediction
            and (video_num_frames <= 1 or (video_num_frames - 1) % 4 != 0)
        ):
            raise ValueError(
                "FastWAM video_num_frames must be 1 + a multiple of 4: "
                f"num_frames={num_frames}, action_video_freq_ratio="
                f"{action_video_freq_ratio}, video_num_frames={video_num_frames}"
            )
        action_horizon = int(action_transform.get("chunk_size_steps", num_frames - 1))
        video_size = [int(v) for v in ds.get("video_size", [224, 448])]
        video_dit_config = fastwam_cfg.get("video_dit_config", {})
        if not isinstance(video_dit_config, Mapping):
            video_dit_config = {}
        else:
            video_dit_config = dict(video_dit_config)

        condition_history_frames = int(ds.get("condition_history_frames", 1))
        if condition_history_frames not in (1, 4):
            raise ValueError(
                "FastWAM inference supports checkpoint condition_history_frames=1 "
                f"or 4, got {condition_history_frames}"
            )
        condition_history_stride = int(ds.get("condition_history_stride", 1))
        if condition_history_stride <= 0:
            raise ValueError(
                "FastWAM checkpoint condition_history_stride must be positive, "
                f"got {condition_history_stride}"
            )

        training_mode_value = fastwam_cfg.get("action_infer_mode")
        if training_mode_value is None or not str(training_mode_value).strip():
            training_mode_value = fastwam_cfg.get("default_action_infer_mode")
        if training_mode_value is None or not str(training_mode_value).strip():
            training_mode_value = cfg.get("action_infer_mode")
        if training_mode_value is None or not str(training_mode_value).strip():
            training_mode_value = cfg.get(
                "default_action_infer_mode",
                "observed_frame_cache",
            )
        training_action_infer_mode = _normalize_action_infer_mode(
            training_mode_value
        )
        action_infer_mode = _normalize_action_infer_mode(
            self.params.get("action_infer_mode", training_action_infer_mode)
        )

        training_attention_value = fastwam_cfg.get(
            "action_video_attention_mode"
        )
        if training_attention_value is None or not str(training_attention_value).strip():
            training_attention_value = cfg.get("action_video_attention_mode")
        training_action_video_attention_mode = (
            None
            if training_attention_value is None
            or not str(training_attention_value).strip()
            else str(training_attention_value).strip().lower()
        )
        if training_action_video_attention_mode not in {
            None,
            "first_frame",
            "full_video",
        }:
            raise ValueError(
                "checkpoint action_video_attention_mode must be 'first_frame' or "
                f"'full_video', got {training_action_video_attention_mode!r}"
            )
        configured_attention_mode = self.params.get(
            "action_video_attention_mode",
            training_action_video_attention_mode,
        )
        if configured_attention_mode is None or not str(
            configured_attention_mode
        ).strip():
            configured_attention_mode = "first_frame"
        action_video_attention_mode = str(
            configured_attention_mode
        ).strip().lower()
        if action_video_attention_mode not in {"first_frame", "full_video"}:
            raise ValueError(
                "action_video_attention_mode must be 'first_frame' or "
                f"'full_video', got {action_video_attention_mode!r}"
            )

        if "seperated_timestep" not in video_dit_config:
            video_dit_config["seperated_timestep"] = video_dit_config.get(
                "separated_timestep",
                False,
            )

        return {
            "ckpt_root": str(self.ckpt_root),
            "weight_path": str(self.weight_path),
            "framework": str(cfg.framework.name),
            "dataset_dirs": list(ds.get("dataset_dirs", [])),
            "image_meta": image_meta,
            "video_size": video_size,
            "concat_multi_camera": str(ds.get("concat_multi_camera", "horizontal")),
            "condition_history_frames": condition_history_frames,
            "condition_history_stride": condition_history_stride,
            "num_frames": num_frames,
            "action_video_freq_ratio": action_video_freq_ratio,
            "video_num_frames": video_num_frames,
            "sampled_video_num_frames": video_num_frames,
            "training_enable_video_prediction": training_enable_video_prediction,
            "video_action_conditioned": _as_bool(
                video_dit_config.get("action_conditioned", False)
            ),
            "seperated_timestep": _as_bool(
                video_dit_config.get("seperated_timestep", False)
            ),
            "fuse_vae_embedding_in_latents": _as_bool(
                video_dit_config.get("fuse_vae_embedding_in_latents", False)
            ),
            "video_scheduler": dict(fastwam_cfg.get("video_scheduler", {}) or {}),
            "action_scheduler": dict(fastwam_cfg.get("action_scheduler", {}) or {}),
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "proprio_dim": state_dim,
            "zero_state_condition": _resolve_zero_state_condition(
                cast(Dict[str, Any], fastwam_cfg),
                self.params,
            ),
            "training_zero_state_condition": _resolve_zero_state_condition(
                cast(Dict[str, Any], fastwam_cfg)
            ),
            "use_stepwise_action_norm": _as_bool(
                processor_cfg.get("use_stepwise_action_norm", False)
            ),
            "norm_default_mode": str(processor_cfg.get("norm_default_mode", "min/max")),
            "relative_pose_frame": str(
                action_transform.get("relative_pose_frame", "local_frame")
            ),
            "source_quaternion_order": str(action_transform.get("quaternion_order", "wxyz")),
            "action_quaternion_order": "xyzw",
            "action_layout": ["left", "right"],
            "prompt_template": str(
                self.params.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
            ),
            "training_model_id": str(fastwam_cfg.get("model_id", "")),
            "training_tokenizer_model_id": str(
                fastwam_cfg.get("tokenizer_model_id", "")
            ),
            "training_load_text_encoder": _as_bool(
                fastwam_cfg.get("load_text_encoder", False)
            ),
            "training_redirect_common_files": _as_bool(
                fastwam_cfg.get("redirect_common_files", False)
            ),
            "training_skip_dit_load_from_pretrain": _as_bool(
                fastwam_cfg.get("skip_dit_load_from_pretrain", False)
            ),
            "training_action_infer_mode": training_action_infer_mode,
            "action_infer_mode": action_infer_mode,
            "training_action_video_attention_mode": (
                training_action_video_attention_mode
            ),
            "action_video_attention_mode": action_video_attention_mode,
            "runtime_model_id": str(
                self.params.get("model_id", fastwam_cfg.get("model_id", ""))
            ),
            "runtime_tokenizer_model_id": str(
                self.params.get(
                    "tokenizer_model_id",
                    fastwam_cfg.get("tokenizer_model_id", ""),
                )
            ),
            "runtime_load_text_encoder": _as_bool(
                self.params.get(
                    "load_text_encoder",
                    fastwam_cfg.get("load_text_encoder", False),
                )
            ),
            "runtime_text_encoder_path": (
                None
                if self.params.get("text_encoder_path") is None
                else str(self.params.get("text_encoder_path"))
            ),
            "runtime_tokenizer_path": (
                None
                if self.params.get("tokenizer_path") is None
                else str(self.params.get("tokenizer_path"))
            ),
            "runtime_skip_pretrain_backbone_load": _as_bool(
                self.params.get("skip_pretrain_backbone_load", True)
            ),
            "runtime_crop_input_to_aspect": self.params.get("crop_input_to_aspect"),
        }

    @staticmethod
    def _raw_image_aspect_wh(image_meta: list[Dict[str, Any]]) -> Optional[float]:
        ratios: list[float] = []
        for meta in image_meta:
            raw_shape = meta.get("raw_shape")
            if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) < 3:
                continue
            raw_h = float(raw_shape[-2])
            raw_w = float(raw_shape[-1])
            if raw_h <= 0 or raw_w <= 0:
                continue
            ratios.append(raw_w / raw_h)
        if not ratios:
            return None
        first = ratios[0]
        if any(abs(ratio - first) > 1e-6 for ratio in ratios[1:]):
            raise ValueError(f"FastWAM training image raw_shape aspects differ: {ratios}")
        return first

    def _validate_runtime_alignment(self) -> None:
        requested_history_values = [
            int(value)
            for value in (
                self.params.get("condition_history_frames"),
                self.params.get("obs_horizon"),
            )
            if value is not None
        ]
        if len(set(requested_history_values)) > 1:
            raise ValueError(
                "policy.params.condition_history_frames and obs_horizon must match"
            )
        if requested_history_values and requested_history_values[0] != int(
            self.contract["condition_history_frames"]
        ):
            raise ValueError(
                "FastWAM visual-history mismatch: runtime config requests "
                f"{requested_history_values[0]} frame(s), but checkpoint config "
                "datasets.vla_data.condition_history_frames is "
                f"{self.contract['condition_history_frames']}"
            )

        requested_stride_values = [
            int(value)
            for value in (
                self.params.get("condition_history_stride"),
                self.params.get("downsample"),
            )
            if value is not None
        ]
        if len(set(requested_stride_values)) > 1:
            raise ValueError(
                "policy.params.condition_history_stride and downsample must match"
            )
        if requested_stride_values and requested_stride_values[0] != int(
            self.contract["condition_history_stride"]
        ):
            raise ValueError(
                "FastWAM visual-history stride mismatch: runtime config requests "
                f"stride {requested_stride_values[0]}, but checkpoint config "
                "datasets.vla_data.condition_history_stride is "
                f"{self.contract['condition_history_stride']}"
            )

        action_infer_mode = str(
            self.contract.get("action_infer_mode", "observed_frame_cache")
        )
        training_action_infer_mode = str(
            self.contract.get("training_action_infer_mode", action_infer_mode)
        )
        if (
            _as_bool(self.params.get("enforce_checkpoint_infer_mode", False))
            and action_infer_mode != training_action_infer_mode
        ):
            raise ValueError(
                "FastWAM action inference mode mismatch: checkpoint specifies "
                f"{training_action_infer_mode!r}, but runtime requests "
                f"{action_infer_mode!r}"
            )
        training_attention_mode = self.contract.get(
            "training_action_video_attention_mode"
        )
        runtime_attention_mode = self.contract.get(
            "action_video_attention_mode"
        )
        if (
            training_attention_mode is not None
            and runtime_attention_mode != training_attention_mode
        ):
            raise ValueError(
                "FastWAM action-video attention mismatch: checkpoint was trained "
                f"with {training_attention_mode!r}, but runtime requests "
                f"{runtime_attention_mode!r}"
            )

        explicit_zero_state = next(
            (
                self.params[key]
                for key in ("zero_state_condition", "zero_action_condition")
                if self.params.get(key) is not None
            ),
            None,
        )
        if explicit_zero_state is not None and _as_bool(explicit_zero_state) != bool(
            self.contract.get("training_zero_state_condition", False)
        ):
            raise ValueError(
                "FastWAM zero-state mismatch: checkpoint was trained with "
                "zero_state_condition="
                f"{self.contract.get('training_zero_state_condition')}, but runtime "
                f"requests {_as_bool(explicit_zero_state)}"
            )

        if action_infer_mode == "one_pass_future_cache":
            self._validate_c4_runtime_contract()
        elif (
            int(self.params.get("c4_video_denoise_steps", 0)) != 0
            or self.params.get("c4_video_num_inference_steps") is not None
            or _as_bool(self.params.get("c4_reverse_video_timestep", False))
        ):
            raise ValueError(
                "C4 video scheduler settings are only valid for "
                "action_infer_mode='one_pass_future_cache'"
            )

        visualization_enabled = _as_bool(
            getattr(self, "video_visualization_config", {}).get("enabled", False)
        )
        if visualization_enabled:
            image_count = len(self.contract.get("image_meta", []))
            if image_count != 2:
                raise ValueError(
                    "FastWAM video visualization requires exactly two camera "
                    f"views (left/right), got {image_count}"
                )
            if str(self.contract.get("concat_multi_camera", "horizontal")) not in {
                "horizontal",
                "vertical",
            }:
                raise ValueError(
                    "FastWAM video visualization requires concat_multi_camera to "
                    "be 'horizontal' or 'vertical'"
                )

        if self.params.get("zero_action_condition") is not None:
            logger.warning(
                "`zero_action_condition` is treated as an alias for "
                "`zero_state_condition`; FastWAM zeroes the normalized proprio token."
            )

        expected_aspect = self._raw_image_aspect_wh(list(self.contract["image_meta"]))
        crop_aspect = self.params.get("crop_input_to_aspect")
        if expected_aspect is not None and crop_aspect is not None:
            if not isinstance(crop_aspect, (list, tuple)) or len(crop_aspect) != 2:
                raise ValueError(
                    f"crop_input_to_aspect must be [width, height], got {crop_aspect!r}"
                )
            runtime_aspect = float(crop_aspect[0]) / float(crop_aspect[1])
            if abs(runtime_aspect - expected_aspect) > 1e-3:
                raise ValueError(
                    "FastWAM runtime crop aspect does not match training raw image "
                    f"aspect: crop_input_to_aspect={crop_aspect} "
                    f"training_raw_w_over_h={expected_aspect:.6f}"
                )
        elif expected_aspect is not None and abs(expected_aspect - 1.0) > 1e-3:
            logger.warning(
                "FastWAM training raw image aspect is %.6f but runtime has no "
                "crop_input_to_aspect. Ensure camera frames already match training.",
                expected_aspect,
            )

        text_encoder_path = self.params.get("text_encoder_path")
        tokenizer_path = self.params.get("tokenizer_path")
        if bool(text_encoder_path) != bool(tokenizer_path):
            raise ValueError(
                "Set both text_encoder_path and tokenizer_path, or neither. "
                "FastWAM prompt encoding must use a matched Wan T5 pair."
            )
        if bool(self.contract["runtime_load_text_encoder"]) and text_encoder_path:
            text_path = Path(str(text_encoder_path)).expanduser()
            tok_path = Path(str(tokenizer_path)).expanduser()
            if not text_path.is_file():
                raise FileNotFoundError(f"FastWAM text encoder not found: {text_path}")
            if not tok_path.is_dir():
                raise FileNotFoundError(f"FastWAM tokenizer dir not found: {tok_path}")
            if not (tok_path / "tokenizer.json").is_file():
                raise FileNotFoundError(
                    f"FastWAM tokenizer.json not found under: {tok_path}"
                )
            expected_names = {
                "models_t5_umt5-xxl-enc-bf16.pth",
                "models_t5_umt5-xxl-enc-bf16.safetensors",
            }
            if text_path.name not in expected_names:
                logger.warning(
                    "FastWAM text encoder filename %s is not the Wan T5 default %s",
                    text_path.name,
                    sorted(expected_names),
                )

    def _validate_c4_runtime_contract(self) -> None:
        """Validate FasterWAM's one-pass future-cache scheduler requirements."""
        c4_video_denoise_steps = self.params.get("c4_video_denoise_steps", 0)
        if isinstance(c4_video_denoise_steps, bool):
            raise ValueError(
                "c4_video_denoise_steps must be an integer in "
                "[0, c4_video_num_inference_steps], not a boolean"
            )
        try:
            c4_video_denoise_steps = int(c4_video_denoise_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "c4_video_denoise_steps must be an integer in "
                "[0, c4_video_num_inference_steps]"
            ) from exc

        action_num_inference_steps = int(
            self.params.get("num_inference_steps", 10)
        )
        raw_video_num_inference_steps = self.params.get(
            "c4_video_num_inference_steps",
            action_num_inference_steps,
        )
        if isinstance(raw_video_num_inference_steps, bool):
            raise ValueError(
                "c4_video_num_inference_steps must be a positive integer, not a boolean"
            )
        try:
            c4_video_num_inference_steps = int(raw_video_num_inference_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "c4_video_num_inference_steps must be a positive integer"
            ) from exc
        if c4_video_num_inference_steps <= 0:
            raise ValueError("c4_video_num_inference_steps must be positive")
        if not 0 <= c4_video_denoise_steps <= c4_video_num_inference_steps:
            raise ValueError(
                "c4_video_denoise_steps must be in "
                "[0, c4_video_num_inference_steps], "
                f"got {c4_video_denoise_steps} for num_inference_steps="
                f"{c4_video_num_inference_steps}"
            )
        self.params["c4_video_denoise_steps"] = c4_video_denoise_steps
        self.params["c4_video_num_inference_steps"] = c4_video_num_inference_steps
        self.params["c4_reverse_video_timestep"] = _as_bool(
            self.params.get("c4_reverse_video_timestep", False)
        )

        if self.contract.get("training_action_video_attention_mode") != "full_video":
            raise ValueError(
                "one_pass_future_cache requires checkpoint "
                "framework.fastwam.action_video_attention_mode=full_video"
            )
        if not _as_bool(self.contract.get("training_enable_video_prediction", False)):
            raise ValueError(
                "one_pass_future_cache requires checkpoint "
                "framework.fastwam.enable_video_prediction=true"
            )
        if not _as_bool(self.contract.get("seperated_timestep", False)):
            raise ValueError(
                "one_pass_future_cache requires checkpoint "
                "video_dit_config.seperated_timestep=true"
            )
        if not _as_bool(
            self.contract.get("fuse_vae_embedding_in_latents", False)
        ):
            raise ValueError(
                "one_pass_future_cache requires checkpoint "
                "video_dit_config.fuse_vae_embedding_in_latents=true"
            )
        sampled_video_num_frames = int(
            self.contract.get("sampled_video_num_frames", 0)
        )
        if sampled_video_num_frames % 4 != 1:
            raise ValueError(
                "one_pass_future_cache requires sampled_video_num_frames % 4 == 1, "
                f"got {sampled_video_num_frames}"
            )

        inference_shifts: Dict[str, float] = {}
        for scheduler_name in ("video_scheduler", "action_scheduler"):
            scheduler = cast(
                Dict[str, Any], self.contract.get(scheduler_name, {})
            )
            if int(scheduler.get("num_train_timesteps", -1)) != 1000:
                raise ValueError(
                    "one_pass_future_cache requires "
                    f"{scheduler_name}.num_train_timesteps=1000"
                )
            for shift_name in ("train_shift", "infer_shift"):
                shift = float(scheduler.get(shift_name, -1.0))
                if not any(abs(shift - allowed) <= 1e-8 for allowed in (1.0, 5.0)):
                    raise ValueError(
                        "one_pass_future_cache requires "
                        f"{scheduler_name}.{shift_name} in (1.0, 5.0), got {shift}"
                    )
            train_shift = float(scheduler["train_shift"])
            infer_shift = float(scheduler["infer_shift"])
            if abs(train_shift - infer_shift) > 1e-8:
                raise ValueError(
                    "one_pass_future_cache requires each inference scheduler "
                    "to preserve its training shift, got "
                    f"{scheduler_name}.train_shift={train_shift} and "
                    f"infer_shift={infer_shift}"
                )
            inference_shifts[scheduler_name] = infer_shift

        sigma_shift = self.params.get("sigma_shift")
        if sigma_shift is not None:
            override = float(sigma_shift)
            if not np.isfinite(override):
                raise ValueError("one_pass_future_cache sigma_shift must be finite")
            mismatches = {
                name: shift
                for name, shift in inference_shifts.items()
                if abs(override - shift) > 1e-8
            }
            if mismatches:
                raise ValueError(
                    "one_pass_future_cache sigma_shift overrides both video and "
                    "action schedulers, so it must match every configured "
                    f"infer_shift; got sigma_shift={override}, shifts={inference_shifts}. "
                    "Omit sigma_shift for mixed video/action shifts."
                )

    def metadata(self) -> Dict[str, Any]:
        meta = dict(self.contract)
        model = getattr(self, "model", None)
        model_video_enabled = _as_bool(
            getattr(
                model,
                "enable_video_prediction",
                self.contract.get("training_enable_video_prediction", True),
            )
        )
        video_prediction_supported = bool(
            model_video_enabled
            and model is not None
            and callable(getattr(model, "infer_joint", None))
        )
        meta.update(
            {
                "server": "FastWAM",
                "runtime_backend": "nmx_vla",
                "model_variant": getattr(self, "model_variant", None),
                "checkpoint_root": str(self.ckpt_root),
                "checkpoint_weight": str(self.weight_path),
                "num_inference_steps": int(self.params.get("num_inference_steps", 10)),
                "c4_video_denoise_steps": int(
                    self.params.get("c4_video_denoise_steps", 0)
                ),
                "c4_video_num_inference_steps": int(
                    self.params.get(
                        "c4_video_num_inference_steps",
                        self.params.get("num_inference_steps", 10),
                    )
                ),
                "c4_reverse_video_timestep": _as_bool(
                    self.params.get("c4_reverse_video_timestep", False)
                ),
                "model_loaded": model is not None,
                "enable_rtc": self.enable_rtc,
                "reverse_temporal_chunks": _as_bool(
                    self._build_model_config().get(
                        "reverse_temporal_chunks", False
                    )
                ),
                "rtc_use_vjp": self.rtc_use_vjp,
                "torch_compile_requested": bool(
                    self._torch_compile_config["enabled"]
                ),
                "torch_compile_active": bool(
                    model is not None
                    and getattr(
                        model,
                        "action_denoise_compile_enabled",
                        False,
                    )
                ),
                "prompt_cache_enabled": self._prompt_cache_enabled,
                "prompt_cache_entries": len(
                    self._prompt_embedding_cache
                ),
                "warmup_completed_requests": self._warmup_completed_requests,
                "nmx_vla_module_path": self._nmx_vla_module_path,
                "video_visualization_enabled": _as_bool(
                    getattr(self, "video_visualization_config", {}).get(
                        "enabled", False
                    )
                ),
                "video_prediction_supported": video_prediction_supported,
                "video_prediction_action_conditioned": _as_bool(
                    self.contract.get("video_action_conditioned", False)
                ),
                "checkpoint_completion_marker": (
                    None
                    if self.checkpoint_completion is None
                    else self.checkpoint_completion.get("marker_path")
                ),
                "checkpoint_complete": self.checkpoint_completion is not None,
                "gpu": self.gpu_health(),
            }
        )
        return meta

    def _build_model_config(self) -> Dict[str, Any]:
        fw = OmegaConf.to_container(
            self.training_cfg.framework.fastwam,
            resolve=True,
        )
        model_cfg = dict(fw)
        model_cfg["proprio_dim"] = int(self.contract["proprio_dim"])
        model_cfg["zero_state_condition"] = _resolve_zero_state_condition(
            model_cfg,
            self.params,
        )
        model_cfg["enable_video_prediction"] = _as_bool(
            model_cfg.get("enable_video_prediction", True)
        )
        video_dit_config = dict(model_cfg.get("video_dit_config", {}) or {})
        if "seperated_timestep" not in video_dit_config and "separated_timestep" in video_dit_config:
            video_dit_config["seperated_timestep"] = video_dit_config[
                "separated_timestep"
            ]
        # The nmx_vla constructor retains the historical misspelling.
        video_dit_config.pop("separated_timestep", None)
        model_cfg["video_dit_config"] = video_dit_config
        model_cfg["action_video_attention_mode"] = str(
            self.contract.get("action_video_attention_mode", "first_frame")
        )
        if self.params.get("reverse_temporal_chunks") is not None:
            model_cfg["reverse_temporal_chunks"] = _as_bool(
                self.params["reverse_temporal_chunks"]
            )

        if self.params.get("model_id") is not None:
            model_cfg["model_id"] = str(self.params["model_id"])
        if self.params.get("tokenizer_model_id") is not None:
            model_cfg["tokenizer_model_id"] = str(self.params["tokenizer_model_id"])

        text_encoder_path = self.params.get("text_encoder_path")
        tokenizer_path = self.params.get("tokenizer_path")
        if bool(text_encoder_path) != bool(tokenizer_path):
            raise ValueError(
                "Set both text_encoder_path and tokenizer_path, or neither."
            )
        # The checkpoint's training contract is the default.  A plain Python
        # ``bool("false")`` would silently enable the encoder for configs that
        # came through an environment/Hydra string, so normalize explicitly.
        configured_load_text_encoder = self.params.get(
            "load_text_encoder",
            model_cfg.get("load_text_encoder", False),
        )
        # Keep the requested value in the model contract.  When explicit local
        # paths are supplied, ``_load_model`` disables automatic loading for
        # construction and ``_attach_text_encoder_if_configured`` attaches the
        # matched pair after the checkpoint payload has been released.
        model_cfg["load_text_encoder"] = _as_bool(configured_load_text_encoder)
        skip_pretrain = _as_bool(self.params.get("skip_pretrain_backbone_load", True))
        model_cfg["skip_dit_load_from_pretrain"] = skip_pretrain
        if skip_pretrain:
            model_cfg["action_dit_pretrained_path"] = None
        elif self.params.get("action_dit_pretrained_path") is not None:
            model_cfg["action_dit_pretrained_path"] = str(
                self.params["action_dit_pretrained_path"]
            )
        return model_cfg

    def _load_model(self) -> Any:
        _install_nmx_vla_log_bridge()
        import nmx_vla
        from nmx_vla.model.modules.wam import create_fastwam

        module_file = getattr(nmx_vla, "__file__", None)
        if module_file is None:
            raise RuntimeError("Unable to determine the imported nmx_vla path")
        module_path = Path(module_file).resolve()
        expected_package_root = (_NMX_VLA_ROOT / "nmx_vla").resolve()
        if not module_path.is_relative_to(expected_package_root):
            raise RuntimeError(
                "FastWAM server imported nmx_vla from the wrong checkout: "
                f"{module_path}; expected a module under {expected_package_root}"
            )
        self._nmx_vla_module_path = str(module_path)
        logger.info(f"Using nmx_vla from {module_path}")

        model_base_path = self.params.get("model_base_path")
        if model_base_path:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base_path)

        device = str(self.params.get("device", "cuda"))
        dtype = _parse_dtype(self.params.get("param_dtype", self.params.get("dtype", "bf16")))
        model_cfg = self._build_model_config()
        logger.info(
            "Building FastWAM model: "
            f"device={device} dtype={dtype} "
            f"load_text_encoder={model_cfg.get('load_text_encoder')} "
            f"skip_pretrain_backbone_load={model_cfg.get('skip_dit_load_from_pretrain')} "
            f"zero_state_condition={model_cfg.get('zero_state_condition')} "
            f"action_video_attention_mode={model_cfg.get('action_video_attention_mode')} "
            f"reverse_temporal_chunks={model_cfg.get('reverse_temporal_chunks')}"
        )
        create_kwargs: Dict[str, Any] = {
            "model_id": model_cfg["model_id"],
            "tokenizer_model_id": model_cfg["tokenizer_model_id"],
            "video_dit_config": model_cfg["video_dit_config"],
            "tokenizer_max_len": int(model_cfg.get("tokenizer_max_len", 128)),
            "load_text_encoder": _as_bool(
                model_cfg.get("load_text_encoder", False)
            ) and not bool(
                self.params.get("text_encoder_path")
                and self.params.get("tokenizer_path")
            ),
            "proprio_dim": model_cfg.get("proprio_dim"),
            "zero_state_condition": _as_bool(
                model_cfg.get("zero_state_condition", False)
            ),
            "action_dit_config": model_cfg.get("action_dit_config"),
            "action_dit_pretrained_path": model_cfg.get("action_dit_pretrained_path"),
            "skip_dit_load_from_pretrain": _as_bool(
                model_cfg.get("skip_dit_load_from_pretrain", False)
            ),
            "video_scheduler": model_cfg.get("video_scheduler"),
            "action_scheduler": model_cfg.get("action_scheduler"),
            "loss": model_cfg.get("loss"),
            "mot_checkpoint_mixed_attn": _as_bool(
                model_cfg.get("mot_checkpoint_mixed_attn", True)
            ),
            "enable_video_prediction": _as_bool(
                model_cfg.get("enable_video_prediction", True)
            ),
            "redirect_common_files": _as_bool(
                model_cfg.get("redirect_common_files", False)
            ),
            "model_dtype": dtype,
            "device": device,
        }
        create_signature = inspect.signature(create_fastwam)
        accepts_extra_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in create_signature.parameters.values()
        )
        supports_attention_mode = (
            "action_video_attention_mode" in create_signature.parameters
            or accepts_extra_kwargs
        )
        action_video_attention_mode = str(
            model_cfg.get("action_video_attention_mode", "first_frame")
        )
        if supports_attention_mode:
            create_kwargs["action_video_attention_mode"] = action_video_attention_mode
        elif action_video_attention_mode != "first_frame":
            raise RuntimeError(
                "The selected nmx_vla FastWAM runtime does not support "
                "action_video_attention_mode=full_video; install the pinned "
                "FasterWAM-compatible revision."
            )
        # Older nmx_vla checkouts can still serve the ordinary chronological
        # layout, but they cannot accept the reverse-layout constructor flag.
        # Inspect the installed API before passing the optional keyword so a
        # stale checkout fails with an actionable message only when reverse
        # output was explicitly requested.
        if "reverse_temporal_chunks" in model_cfg:
            reverse_temporal_chunks = _as_bool(model_cfg["reverse_temporal_chunks"])
            supports_reverse_temporal_chunks = (
                "reverse_temporal_chunks" in create_signature.parameters
                or accepts_extra_kwargs
            )
            if supports_reverse_temporal_chunks:
                create_kwargs["reverse_temporal_chunks"] = reverse_temporal_chunks
            elif reverse_temporal_chunks:
                raise RuntimeError(
                    "The selected nmx_vla FastWAM runtime does not support "
                    "reverse_temporal_chunks=true; install the pinned reverse "
                    "FastWAM nmx_vla revision."
                )
            else:
                logger.info(
                    "Ignoring reverse_temporal_chunks=false because the selected "
                    "nmx_vla runtime uses the standard temporal layout."
                )
        model = create_fastwam(**create_kwargs)
        # FasterWAM checkpoints encode the MOT layout.  A warning-only load
        # would leave part of the action head randomly initialized, so reject
        # incompatible checkpoints before the server can accept observations.
        payload = model.load_checkpoint(
            str(self.weight_path),
            fail_on_incompatible_mot=True,
        )
        checkpoint_step = payload.get("step") if isinstance(payload, dict) else None
        # ``load_checkpoint`` returns the complete CPU state dict.  Keeping it
        # alive while constructing UMT5-XXL adds ~11 GiB to the host-RAM peak.
        # The model has already copied those weights, so retain only the scalar
        # metadata needed by the log line before loading the text encoder.
        del payload
        gc.collect()
        self._attach_text_encoder_if_configured(model, model_cfg, dtype, device)
        # RTC differentiates only with respect to the noisy action latent.  Keep
        # model parameters frozen so autograd does not build or execute the
        # enormous, unused parameter-gradient branches during each VJP.
        model.eval().requires_grad_(False)
        logger.info(
            "Loaded FastWAM checkpoint "
            f"{self.weight_path} (step={checkpoint_step})"
        )
        cleanup_error = gpu_safety.safe_cuda_cleanup(torch)
        if cleanup_error:
            logger.warning(f"CUDA cleanup after FastWAM load failed: {cleanup_error}")
        return model

    def _attach_text_encoder_if_configured(
        self,
        model: Any,
        model_cfg: Dict[str, Any],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        # ``model_cfg`` is the effective config after checkpoint defaults and
        # runtime overrides have been resolved.  Do not fall back to the raw
        # CLI mapping here: a checkpoint with ``load_text_encoder=false`` may
        # intentionally provide a model that never needs a local T5 pair.
        if not _as_bool(
            model_cfg.get(
                "load_text_encoder",
                self.params.get("load_text_encoder", False),
            )
        ):
            return
        if getattr(model, "text_encoder", None) is not None and getattr(model, "tokenizer", None) is not None:
            return

        text_encoder_path = self.params.get("text_encoder_path")
        tokenizer_path = self.params.get("tokenizer_path")
        if not text_encoder_path or not tokenizer_path:
            raise ValueError(
                "load_text_encoder=true but text encoder/tokenizer are not loaded. "
                "Set policy.params.text_encoder_path and policy.params.tokenizer_path, "
                "or provide a tokenizer_model_id cache that nmx_vla can resolve."
            )

        from nmx_vla.model.modules.wam.helpers.loader import _load_registered_model
        from nmx_vla.model.modules.wam.wan_video_text_encoder import HuggingfaceTokenizer

        logger.info(
            "Loading FastWAM text encoder/tokenizer from local paths: "
            f"{text_encoder_path}, {tokenizer_path}"
        )
        model.text_encoder = _load_registered_model(
            str(Path(text_encoder_path).expanduser().resolve()),
            "wan_video_text_encoder",
            torch_dtype=dtype,
            device=device,
            low_cpu_mem_usage=True,
        )
        model.tokenizer = HuggingfaceTokenizer(
            name=str(Path(tokenizer_path).expanduser().resolve()),
            seq_len=int(model_cfg.get("tokenizer_max_len", 128)),
            clean="whitespace",
        )

    def _format_prompt(self, prompt: Optional[str]) -> str:
        task = str(prompt if prompt is not None else self.params.get("task", "")).strip()
        if not task:
            raise ValueError("FastWAM prompt/task is required for inference")
        if task.startswith("A video recorded from a robot"):
            return task
        return str(self.contract["prompt_template"]).format(task=task)

    def _resize_chw(
        self,
        tensor: torch.Tensor,
        size_hw: Tuple[int, int],
        *,
        interpolation: str = "bilinear",
    ) -> torch.Tensor:
        try:
            import torchvision.transforms.functional as TF
            from torchvision.transforms import InterpolationMode

            mode = (
                InterpolationMode.BICUBIC
                if interpolation == "bicubic"
                else InterpolationMode.BILINEAR
            )
            return TF.resize(tensor, list(size_hw), interpolation=mode, antialias=True)
        except Exception:
            mode = "bicubic" if interpolation == "bicubic" else "bilinear"
            return torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=size_hw,
                mode=mode,
                align_corners=False,
                antialias=True,
            ).squeeze(0)

    @staticmethod
    def _center_crop_chw(tensor: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        target_h, target_w = size_hw
        _, h, w = tensor.shape
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return tensor[:, top : top + target_h, left : left + target_w]

    def _resize_smallest_side_chw(self, tensor: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        target_h, target_w = size_hw
        _, orig_h, orig_w = tensor.shape
        ratio = max(float(target_h) / float(orig_h), float(target_w) / float(orig_w))
        resize_hw = (
            max(int(round(orig_h * ratio)), target_h),
            max(int(round(orig_w * ratio)), target_w),
        )
        return self._resize_chw(tensor, resize_hw, interpolation="bicubic")

    @staticmethod
    def _image_to_chw_float(image: np.ndarray) -> torch.Tensor:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA image [H,W,C], got {image.shape}")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _as_uint8_rgb_hwc(image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA image [H,W,C], got {array.shape}")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _center_crop_hwc_to_aspect(
        image: np.ndarray,
        aspect_wh: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        target_w_ratio, target_h_ratio = int(aspect_wh[0]), int(aspect_wh[1])
        if target_w_ratio <= 0 or target_h_ratio <= 0:
            raise ValueError(f"crop_input_to_aspect must be positive [W,H], got {aspect_wh}")

        h, w = image.shape[:2]
        target_ratio = float(target_w_ratio) / float(target_h_ratio)
        current_ratio = float(w) / float(h)

        if current_ratio > target_ratio:
            crop_w = max(1, int(round(h * target_ratio)))
            crop_h = h
            left = max((w - crop_w) // 2, 0)
            top = 0
        else:
            crop_w = w
            crop_h = max(1, int(round(w / target_ratio)))
            left = 0
            top = max((h - crop_h) // 2, 0)

        right = min(left + crop_w, w)
        bottom = min(top + crop_h, h)
        cropped = image[top:bottom, left:right]
        box = {
            "top": int(top),
            "left": int(left),
            "bottom": int(bottom),
            "right": int(right),
            "height": int(cropped.shape[0]),
            "width": int(cropped.shape[1]),
        }
        return np.ascontiguousarray(cropped), box

    def _maybe_crop_input_images(
        self,
        raw_images: list[np.ndarray],
    ) -> Tuple[list[np.ndarray], list[Dict[str, int]]]:
        aspect = self.params.get("crop_input_to_aspect")
        if aspect is None:
            return raw_images, [
                {
                    "top": 0,
                    "left": 0,
                    "bottom": int(image.shape[0]),
                    "right": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                }
                for image in raw_images
            ]
        if not isinstance(aspect, (list, tuple)) or len(aspect) != 2:
            raise ValueError(f"crop_input_to_aspect must be [width, height], got {aspect!r}")
        cropped_images: list[np.ndarray] = []
        crop_boxes: list[Dict[str, int]] = []
        for image in raw_images:
            cropped, box = self._center_crop_hwc_to_aspect(
                image,
                (int(aspect[0]), int(aspect[1])),
            )
            cropped_images.append(cropped)
            crop_boxes.append(box)
        return cropped_images, crop_boxes

    @staticmethod
    def _chw_01_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
        image = tensor.detach().cpu().clamp(0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        return np.ascontiguousarray(np.round(image * 255.0).astype(np.uint8))

    @staticmethod
    def _write_rgb_png(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2

            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        except Exception:
            import imageio.v2 as imageio

            imageio.imwrite(str(path), image)

    def _input_dump_dir_and_limit(self) -> Optional[Tuple[Path, int]]:
        env_flag = os.environ.get("FASTWAM_DUMP_INPUTS")
        if env_flag is not None:
            normalized = env_flag.strip().lower()
            enabled = normalized not in {"", "0", "false", "no", "off"}
            out_dir = (
                "logs/fastwam_input_dump"
                if normalized in {"1", "true", "yes", "on"}
                else env_flag
            )
        else:
            enabled = _as_bool(self.params.get("dump_input_images", False))
            out_dir = str(self.params.get("dump_input_images_dir", "logs/fastwam_input_dump"))
        if not enabled:
            return None

        limit = int(
            os.environ.get(
                "FASTWAM_DUMP_INPUTS_N",
                self.params.get("dump_input_images_n", 1),
            )
        )
        if limit <= 0 or self._dump_input_count >= limit:
            return None
        return Path(out_dir).expanduser(), limit

    def _maybe_dump_input_images(
        self,
        raw_images: list[np.ndarray],
        cropped_images: list[np.ndarray],
        resized_cameras: list[torch.Tensor],
        model_input_01: torch.Tensor,
        crop_boxes: list[Dict[str, int]],
    ) -> None:
        settings = self._input_dump_dir_and_limit()
        if settings is None:
            return
        out_dir, _limit = settings
        idx = self._dump_input_count
        image_meta = list(self.contract["image_meta"])

        try:
            for camera_idx, image in enumerate(raw_images):
                key = str(
                    image_meta[camera_idx].get("key", f"camera_{camera_idx}")
                    if camera_idx < len(image_meta)
                    else f"camera_{camera_idx}"
                )
                raw_rgb = self._as_uint8_rgb_hwc(image)
                raw_path = out_dir / f"{idx:03d}_{key}_raw_{raw_rgb.shape[0]}x{raw_rgb.shape[1]}.png"
                self._write_rgb_png(raw_path, raw_rgb)

                cropped_rgb = self._as_uint8_rgb_hwc(cropped_images[camera_idx])
                cropped_path = (
                    out_dir
                    / f"{idx:03d}_{key}_cropped_{cropped_rgb.shape[0]}x{cropped_rgb.shape[1]}.png"
                )
                self._write_rgb_png(cropped_path, cropped_rgb)

                resized_rgb = self._chw_01_to_uint8_rgb(resized_cameras[camera_idx])
                resized_path = (
                    out_dir
                    / f"{idx:03d}_{key}_resized_{resized_rgb.shape[0]}x{resized_rgb.shape[1]}.png"
                )
                self._write_rgb_png(resized_path, resized_rgb)

            model_rgb = self._chw_01_to_uint8_rgb(model_input_01)
            model_path = out_dir / f"{idx:03d}_model_input_{model_rgb.shape[0]}x{model_rgb.shape[1]}.png"
            self._write_rgb_png(model_path, model_rgb)

            meta = {
                "index": idx,
                "raw_shapes_hwc": [list(np.asarray(image).shape) for image in raw_images],
                "cropped_shapes_hwc": [list(np.asarray(image).shape) for image in cropped_images],
                "crop_boxes_ltrb": crop_boxes,
                "crop_input_to_aspect_wh": self.params.get("crop_input_to_aspect"),
                "resized_shapes_chw": [list(tensor.shape) for tensor in resized_cameras],
                "model_input_shape_chw": list(model_input_01.shape),
                "image_meta": image_meta,
                "video_size": list(self.contract["video_size"]),
                "concat_multi_camera": self.contract["concat_multi_camera"],
            }
            meta_path = out_dir / f"{idx:03d}_metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info(f"[FASTWAM_DUMP_INPUTS] wrote model input images to {out_dir}")
            self._dump_input_count += 1
        except Exception as exc:
            logger.warning(f"[FASTWAM_DUMP_INPUTS] failed to dump input images: {exc}")

    def _preprocess_images(self, images: Any) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        image_meta = list(self.contract["image_meta"])
        if len(images) != len(image_meta):
            raise ValueError(
                f"FastWAM expected {len(image_meta)} camera images, got {len(images)}"
            )

        resized_cameras: list[torch.Tensor] = []
        raw_images = [self._as_uint8_rgb_hwc(image) for image in images]
        cropped_images, crop_boxes = self._maybe_crop_input_images(raw_images)
        for image, meta in zip(cropped_images, image_meta):
            tensor = self._image_to_chw_float(image)
            _, target_h, target_w = meta.get("shape", [3, 224, 224])
            tensor = self._resize_chw(tensor, (int(target_h), int(target_w)))
            resized_cameras.append(tensor)

        concat_mode = str(self.contract["concat_multi_camera"])
        if len(resized_cameras) > 1:
            if concat_mode == "horizontal":
                video = torch.cat(resized_cameras, dim=-1)
            elif concat_mode == "vertical":
                video = torch.cat(resized_cameras, dim=-2)
            else:
                raise ValueError(f"Invalid concat_multi_camera: {concat_mode}")
        else:
            video = resized_cameras[0]

        target_size = tuple(int(v) for v in self.contract["video_size"])
        if tuple(video.shape[-2:]) != target_size:
            video = self._resize_smallest_side_chw(video, target_size)
            video = self._center_crop_chw(video, target_size)
        if len(resized_cameras) == 2:
            split_dim = -1 if concat_mode == "horizontal" else -2
            self._last_preprocessed_camera_frames = [
                self._chw_01_to_uint8_rgb(view)
                for view in torch.tensor_split(video, 2, dim=split_dim)
            ]
        else:
            self._last_preprocessed_camera_frames = []
        self._maybe_dump_input_images(raw_images, cropped_images, resized_cameras, video, crop_boxes)
        video = (video - 0.5) / 0.5
        return video.contiguous()

    def _preprocess_image_history(
        self,
        image_history: Any,
        *,
        history_frame_timestamps_ns: Any = None,
    ) -> torch.Tensor:
        """Build the model's [C, T, H, W] visual-memory conditioning tensor."""
        if not isinstance(image_history, (list, tuple)):
            raise ValueError(
                "FastWAM visual-memory checkpoints require image_history as an "
                "oldest-to-latest list of camera-image lists"
            )
        expected_frames = int(self.contract.get("condition_history_frames", 1))
        if len(image_history) != expected_frames:
            raise ValueError(
                f"FastWAM expected {expected_frames} image-history frames, "
                f"got {len(image_history)}"
            )
        if history_frame_timestamps_ns is not None:
            timestamps = np.asarray(history_frame_timestamps_ns).reshape(-1)
            if len(timestamps) != expected_frames:
                raise ValueError(
                    "FastWAM history timestamp count must match "
                    f"condition_history_frames={expected_frames}, got {len(timestamps)}"
                )

        processed_frames = [
            self._preprocess_images(frame_images) for frame_images in image_history
        ]
        return torch.stack(processed_frames, dim=1).contiguous()

    def _preprocess_state(self, state: Any) -> torch.Tensor:
        state_np = np.asarray(state, dtype=np.float64).reshape(-1)
        expected_dim = int(self.contract["state_dim"])
        if state_np.shape[0] != expected_dim:
            raise ValueError(f"Expected FastWAM state dim {expected_dim}, got {state_np.shape[0]}")
        normalized = self.state_normalizer.forward(state_np)
        return torch.from_numpy(normalized.astype(np.float32))

    def _postprocess_action(self, normalized_action: Any) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.ndim != 2:
            raise ValueError(f"Expected normalized FastWAM action [T,D], got {action.shape}")
        expected_shape = (
            int(self.contract["action_horizon"]),
            int(self.contract["action_dim"]),
        )
        if action.shape != expected_shape:
            raise ValueError(
                "FastWAM model action shape mismatch: expected "
                f"{expected_shape}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Normalized FastWAM action contains NaN or infinity")
        max_abs = float(self.params.get("max_abs_normalized_action", 5.0))
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            raise ValueError(f"max_abs_normalized_action must be finite and positive, got {max_abs}")
        largest = float(np.max(np.abs(action), initial=0.0))
        if largest > max_abs:
            raise ValueError(
                "Normalized FastWAM action exceeds the configured safety domain: "
                f"max_abs={largest:.6f} > max_abs_normalized_action={max_abs:.6f}"
            )
        denormalized = self.action_normalizer.backward(action).astype(np.float64)
        if not np.all(np.isfinite(denormalized)):
            raise ValueError("Denormalized FastWAM action contains NaN or infinity")
        return denormalized

    def _prepare_rtc_inputs(
        self,
        obs: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        if not self.enable_rtc:
            return None, None

        prefix_value = obs.get("prefix_action")
        rtc_value = obs.get("rtc_config")
        if prefix_value is None and rtc_value is None:
            return None, None
        if prefix_value is None or not isinstance(rtc_value, dict):
            raise ValueError(
                "FastWAM RTC request must provide both prefix_action and rtc_config"
            )

        required_keys = {
            "inference_delay",
            "prefix_attention_horizon",
            "max_guidance_weight",
        }
        missing_keys = required_keys.difference(rtc_value)
        if missing_keys:
            raise ValueError(
                f"FastWAM rtc_config is missing keys: {sorted(missing_keys)}"
            )

        prefix_action = np.asarray(prefix_value, dtype=np.float64)
        expected_shape = (
            int(self.contract["action_horizon"]),
            int(self.contract["action_dim"]),
        )
        if prefix_action.shape != expected_shape:
            raise ValueError(
                "FastWAM RTC prefix_action shape mismatch: "
                f"got {prefix_action.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(prefix_action)):
            raise ValueError("FastWAM RTC prefix_action contains NaN or infinity")
        normalized_prefix = self.action_normalizer.forward(prefix_action)
        if not np.all(np.isfinite(normalized_prefix)):
            raise ValueError("Normalized FastWAM RTC prefix contains NaN or infinity")
        prefix_tensor = torch.from_numpy(
            np.asarray(normalized_prefix, dtype=np.float32)
        )
        rtc_config: Dict[str, Any] = {
            "inference_delay": int(rtc_value["inference_delay"]),
            "prefix_attention_horizon": int(
                rtc_value["prefix_attention_horizon"]
            ),
            "max_guidance_weight": float(rtc_value["max_guidance_weight"]),
            "rtc_use_vjp": rtc_value.get("rtc_use_vjp", self.rtc_use_vjp),
        }
        if not isinstance(rtc_config["rtc_use_vjp"], bool):
            raise ValueError("FastWAM RTC rtc_use_vjp must be a boolean")
        if int(rtc_config["inference_delay"]) < 0:
            raise ValueError("FastWAM RTC inference_delay must be non-negative")
        if int(rtc_config["prefix_attention_horizon"]) <= 0:
            raise ValueError("FastWAM RTC prefix_attention_horizon must be positive")
        if not np.isfinite(float(rtc_config["max_guidance_weight"])) or float(rtc_config["max_guidance_weight"]) <= 0:
            raise ValueError("FastWAM RTC max_guidance_weight must be finite and positive")
        return prefix_tensor, rtc_config

    def _save_video_prediction_artifact(
        self,
        video_frames: list[Any],
        *,
        model_prompt: Optional[str],
        seed: Optional[int],
        observation_timestamp_ns: Optional[Any],
    ) -> str:
        """Persist one combined-view prediction plus inputs and metadata."""
        if not video_frames:
            raise ValueError("Cannot save an empty FastWAM video prediction")
        config = self.video_visualization_config
        output_root = Path(
            str(config.get("save_dir", "logs/fastwam_video_predictions"))
        ).expanduser()
        if not output_root.is_absolute():
            output_root = _REPO_ROOT / output_root
        save_index = int(getattr(self, "_saved_video_prediction_count", 0)) + 1
        self._saved_video_prediction_count = save_index
        created_timestamp_ns = time.time_ns()
        artifact_dir = (
            output_root / f"prediction_{save_index:06d}_{created_timestamp_ns}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=False)

        save_fps = float(config.get("save_fps", config.get("display_fps", 7.5)))
        _write_rgb_mp4(
            artifact_dir / "prediction_combined.mp4",
            video_frames,
            save_fps,
        )

        if _as_bool(config.get("save_input_frames", True)):
            from PIL import Image

            for index, frame in enumerate(self._last_preprocessed_camera_frames):
                Image.fromarray(_as_uint8_video_frame(frame)).save(
                    artifact_dir / f"model_input_camera_{index}.png"
                )

        metadata = {
            "schema_version": 1,
            "prediction_index": save_index,
            "created_timestamp_ns": created_timestamp_ns,
            "observation_timestamp_ns": (
                None
                if observation_timestamp_ns is None
                else int(observation_timestamp_ns)
            ),
            "frame_count": len(video_frames),
            "fps": save_fps,
            "concat_mode": str(
                self.contract.get("concat_multi_camera", "horizontal")
            ),
            "view_count": len(self.contract.get("image_meta", [])),
            "video_action_conditioned": bool(
                self.contract.get("video_action_conditioned", False)
            ),
            "seed": seed,
            "prompt": model_prompt,
            "video_file": "prediction_combined.mp4",
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            f"Saved FastWAM video prediction #{save_index} to {artifact_dir}"
        )
        return str(artifact_dir)

    def _maybe_predict_video_for_visualization(
        self,
        *,
        requested: bool,
        model_prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        input_image: torch.Tensor,
        proprio: torch.Tensor,
        action_horizon: int,
        num_inference_steps: int,
        seed: Optional[int],
        action: Optional[torch.Tensor] = None,
        max_frames_override: Optional[Any] = None,
        jpeg_quality_override: Optional[Any] = None,
        observation_timestamp_ns: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the optional joint video path and return transport-safe frames."""
        visualization_cfg = getattr(self, "video_visualization_config", {})
        if not requested or not _as_bool(visualization_cfg.get("enabled", False)):
            return None
        if _as_bool(getattr(self, "_video_prediction_disabled", False)):
            return None
        if not _as_bool(
            getattr(self, "contract", {}).get("training_enable_video_prediction", True)
        ):
            logger.warning(
                "FastWAM video visualization is disabled because this checkpoint "
                "was trained with enable_video_prediction=false."
            )
            return None
        model = getattr(self, "model", None)
        if model is None or not callable(getattr(model, "infer_joint", None)):
            logger.warning(
                "FastWAM video visualization requested, but the loaded nmx_vla "
                "model does not expose infer_joint()."
            )
            return None
        if not _as_bool(getattr(model, "enable_video_prediction", True)):
            logger.warning(
                "FastWAM video visualization is unavailable on the loaded "
                "action-only model."
            )
            return None

        if not _as_bool(
            getattr(self, "contract", {}).get("video_action_conditioned", False)
        ) and not _as_bool(getattr(self, "_video_condition_warning_emitted", False)):
            logger.warning(
                "FastWAM video visualization uses an unconditioned world-model "
                "prediction because video_dit_config.action_conditioned=false."
            )
            self._video_condition_warning_emitted = True

        try:
            max_frames = int(
                max_frames_override
                if max_frames_override is not None
                else visualization_cfg.get("max_prediction_frames", 16)
            )
            jpeg_quality = int(
                jpeg_quality_override
                if jpeg_quality_override is not None
                else visualization_cfg.get("jpeg_quality", 85)
            )
            joint_kwargs: Dict[str, Any] = {
                "prompt": model_prompt,
                "context": context,
                "context_mask": context_mask,
                "input_image": input_image,
                "num_video_frames": int(
                    self.contract.get(
                        "video_num_frames",
                        self.contract.get("num_frames", 33),
                    )
                ),
                "action_horizon": action_horizon,
                "proprio": proprio,
                "num_inference_steps": num_inference_steps,
                "sigma_shift": self.params.get("sigma_shift"),
                "seed": seed,
                "rand_device": str(self.params.get("rand_device", "cpu")),
                "tiled": _as_bool(self.params.get("tiled", False)),
                "test_action_with_infer_action": False,
            }
            if action is not None:
                joint_kwargs["action"] = action
            with torch.no_grad():
                video_result = model.infer_joint(**joint_kwargs)
            video = video_result.get("video") if isinstance(video_result, Mapping) else None
            video_frames = _video_frame_sequence(video)
            saved_artifact_dir: Optional[str] = None
            if _as_bool(visualization_cfg.get("save_predictions", False)):
                try:
                    saved_artifact_dir = self._save_video_prediction_artifact(
                        video_frames,
                        model_prompt=model_prompt,
                        seed=seed,
                        observation_timestamp_ns=observation_timestamp_ns,
                    )
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "Failed to save FastWAM video prediction; continuing "
                        f"without an artifact: {type(exc).__name__}: {exc}"
                    )
            encoded_frames = _encode_video_prediction_frames(
                video_frames,
                max_frames=max_frames,
                jpeg_quality=jpeg_quality,
            )
            if not encoded_frames:
                return None
            result: Dict[str, Any] = {
                "frames": encoded_frames,
                "frame_count": len(video_frames),
                "concat_mode": str(self.contract.get("concat_multi_camera", "horizontal")),
                "view_count": len(self.contract.get("image_meta", [])),
            }
            input_frames = getattr(self, "_last_preprocessed_camera_frames", [])
            if len(input_frames) == 2:
                result["input_frames"] = [
                    _encode_video_prediction_frames(
                        [frame],
                        max_frames=1,
                        jpeg_quality=jpeg_quality,
                    )[0]
                    for frame in input_frames
                ]
            if saved_artifact_dir is not None:
                result["saved_artifact_dir"] = saved_artifact_dir
            self._video_prediction_failure_count = 0
            return result
        except Exception as exc:
            self._record_gpu_failure(exc)
            failures = int(getattr(self, "_video_prediction_failure_count", 0)) + 1
            self._video_prediction_failure_count = failures
            max_failures = max(int(visualization_cfg.get("max_failures", 3)), 1)
            if failures >= max_failures:
                self._video_prediction_disabled = True
                logger.error(
                    "FastWAM video visualization disabled after "
                    f"{failures} consecutive failures: {exc}"
                )
            else:
                logger.warning(
                    "FastWAM video visualization prediction failed "
                    f"({failures}/{max_failures}): {exc}"
                )
            return None

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_gpu_ready()
        if _as_bool(obs.get("reset", False)):
            return {"status": "ok", "data": {"reset": True}}
        if self.model is None:
            raise RuntimeError("FastWAM model is not loaded")

        profiler_context = (
            self.profiler.section("server/infer_total")
            if self.profiler is not None
            else nullcontext()
        )
        with profiler_context:
            prompt_context = (
                self.profiler.section("server/preprocess_prompt")
                if self.profiler is not None
                else nullcontext()
            )
            with prompt_context:
                prompt = self._format_prompt(obs.get("prompt"))
            conditioning_context = (
                self.profiler.section(
                    "server/resolve_prompt_conditioning",
                    category="model",
                )
                if self.profiler is not None
                else nullcontext()
            )
            with conditioning_context:
                model_prompt, context, context_mask = (
                    self._resolve_prompt_conditioning(prompt)
                )
            image_context = (
                self.profiler.section("server/preprocess_images")
                if self.profiler is not None
                else nullcontext()
            )
            with image_context:
                condition_history_frames = int(
                    self.contract.get("condition_history_frames", 1)
                )
                if condition_history_frames == 1:
                    input_image: Optional[torch.Tensor] = self._preprocess_images(
                        obs.get("images")
                    )
                    input_video: Optional[torch.Tensor] = None
                else:
                    input_image = None
                    input_video = self._preprocess_image_history(
                        obs.get("image_history"),
                        history_frame_timestamps_ns=obs.get(
                            "history_frame_timestamps_ns"
                        ),
                    )
            state_context = (
                self.profiler.section("server/preprocess_state")
                if self.profiler is not None
                else nullcontext()
            )
            with state_context:
                proprio = self._preprocess_state(obs.get("state"))
            rtc_context = (
                self.profiler.section("server/prepare_rtc_inputs")
                if self.profiler is not None
                else nullcontext()
            )
            with rtc_context:
                prefix_action, rtc_config = self._prepare_rtc_inputs(obs)

            seed_value = obs.get("seed", self.params.get("seed"))
            seed = None if seed_value is None else int(seed_value)
            num_inference_steps = int(self.params.get("num_inference_steps", 10))
            action_horizon = int(self.contract["action_horizon"])

            start = time.monotonic()
            model_context = (
                self.profiler.section("server/model_infer_action", category="model")
                if self.profiler is not None
                else nullcontext()
            )
            with model_context, torch.no_grad():
                condition_kwargs: Dict[str, Optional[torch.Tensor]] = {
                    "input_image": input_image,
                }
                if input_video is not None:
                    condition_kwargs["input_video"] = input_video
                infer_kwargs = dict(
                    prompt=model_prompt,
                    context=context,
                    context_mask=context_mask,
                    action_horizon=action_horizon,
                    proprio=proprio,
                    num_inference_steps=num_inference_steps,
                    c4_video_denoise_steps=int(
                        self.params.get("c4_video_denoise_steps", 0)
                    ),
                    c4_video_num_inference_steps=int(
                        self.params.get(
                            "c4_video_num_inference_steps",
                            num_inference_steps,
                        )
                    ),
                    c4_reverse_video_timestep=_as_bool(
                        self.params.get("c4_reverse_video_timestep", False)
                    ),
                    sigma_shift=self.params.get("sigma_shift"),
                    seed=seed,
                    rand_device=str(self.params.get("rand_device", "cpu")),
                    tiled=_as_bool(self.params.get("tiled", False)),
                    prefix_actions=prefix_action,
                    inference_delay=(
                        None if rtc_config is None else int(rtc_config["inference_delay"])
                    ),
                    prefix_attention_horizon=(
                        None
                        if rtc_config is None
                        else int(rtc_config["prefix_attention_horizon"])
                    ),
                    max_guidance_weight=(
                        None
                        if rtc_config is None
                        else float(rtc_config["max_guidance_weight"])
                    ),
                    **({"rtc_use_vjp": bool(rtc_config["rtc_use_vjp"])} if rtc_config is not None else {}),
                    profiler=self.profiler,
                )
                try:
                    result = self._infer_model_action(
                        **infer_kwargs,
                        **condition_kwargs,
                    )
                except Exception as exc:
                    self._record_gpu_failure(exc)
                    raise
            infer_ms = (time.monotonic() - start) * 1000.0
            postprocess_context = (
                self.profiler.section("server/postprocess_action")
                if self.profiler is not None
                else nullcontext()
            )
            with postprocess_context:
                normalized_action = result["action"].detach().cpu().float().numpy()
                action = self._postprocess_action(normalized_action)

            video_prediction = None
            video_prediction_ms: Optional[float] = None
            if _as_bool(obs.get("return_video_prediction", False)) and input_image is not None:
                video_start = time.monotonic()
                video_prediction = self._maybe_predict_video_for_visualization(
                    requested=True,
                    model_prompt=model_prompt,
                    context=context,
                    context_mask=context_mask,
                    input_image=input_image,
                    proprio=proprio,
                    action=(
                        result.get("action")
                        if isinstance(result.get("action"), torch.Tensor)
                        else None
                    ),
                    action_horizon=action_horizon,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    max_frames_override=obs.get("video_prediction_max_frames"),
                    jpeg_quality_override=obs.get("video_prediction_jpeg_quality"),
                    observation_timestamp_ns=obs.get("timestamp_ns"),
                )
                video_prediction_ms = (time.monotonic() - video_start) * 1000.0
            elif _as_bool(obs.get("return_video_prediction", False)):
                logger.warning(
                    "FastWAM video visualization is unavailable for visual-memory "
                    "conditioning; request actions without return_video_prediction."
                )

        data: Dict[str, Any] = {
            "action": action,
            "rtc_applied": rtc_config is not None,
        }
        if rtc_config is not None:
            data["rtc_config"] = rtc_config
            data["rtc_use_vjp"] = self.rtc_use_vjp
        if isinstance(result, Mapping) and "future_cache_metadata" in result:
            data["future_cache_metadata"] = result["future_cache_metadata"]
        if _as_bool(self.params.get("return_normalized_action", False)):
            data["normalized_action"] = normalized_action
        if video_prediction is not None:
            data["video_prediction"] = video_prediction
        server_timing: Dict[str, float] = {"infer_ms": infer_ms}
        if video_prediction_ms is not None:
            server_timing["video_prediction_ms"] = video_prediction_ms
        return {
            "status": "ok",
            "data": data,
            "server_timing": server_timing,
        }


class FastWAMWebsocketServer:
    def __init__(
        self,
        policy: FastWAMRuntime,
        host: str,
        port: int,
        metadata: Dict[str, Any],
        api_key: Optional[str] = None,
        ready_file: Optional[Path] = None,
        run_id: Optional[str] = None,
        config_name: Optional[str] = None,
        startup_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._policy = policy
        self._host = _normalize_bind_host(host)
        self._port = int(port)
        self._metadata = metadata
        self._api_key = api_key
        self._ready_file = ready_file
        self._startup_progress = startup_progress
        self._health_identity = {
            "run_id": str(run_id or ""),
            "pid": os.getpid(),
            "config_name": str(config_name or ""),
            "host": self._host,
            "port": self._port,
            "model_loaded": bool(metadata.get("model_loaded")),
        }
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fastwam_infer",
        )
        self._request_count = 0
        self._closed = False

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._infer_executor.shutdown(wait=True, cancel_futures=True)

    async def run(self) -> None:
        if self._startup_progress is not None:
            self._startup_progress("binding_server")
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=self._process_request,
        ) as server:
            self._write_ready_file()
            if self._startup_progress is not None:
                self._startup_progress("ready")
            logger.info(f"FastWAM websocket server listening on {self._host}:{self._port}")
            await server.serve_forever()

    def _write_ready_file(self) -> None:
        if self._ready_file is not None:
            _atomic_write_private_json(self._ready_file, self._health_identity)

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                payload = await websocket.recv()
                request_id = self._request_count
                self._request_count += 1
                if self._policy.profiler is not None:
                    self._policy.profiler.step(request_id)
                unpack_context = (
                    self._policy.profiler.section("server/ws_unpack")
                    if self._policy.profiler is not None
                    else nullcontext()
                )
                with unpack_context:
                    obs = unpackb(payload)
                start = time.monotonic()
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._infer_executor,
                    lambda payload=obs: self._policy.infer(payload),
                )
                result.setdefault("server_timing", {})["total_ms"] = (
                    time.monotonic() - start
                ) * 1000.0
                pack_context = (
                    self._policy.profiler.section("server/ws_pack_send")
                    if self._policy.profiler is not None
                    else nullcontext()
                )
                with pack_context:
                    await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                return
            except Exception as exc:
                self._policy._record_gpu_failure(exc)
                error_text = traceback.format_exc()
                logger.error(f"FastWAM server request failed:\n{error_text}")
                await websocket.send(error_text)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                return

    def _process_request(
        self,
        connection: websocket_server.ServerConnection,
        request: websocket_server.Request,
    ) -> Optional[websocket_server.Response]:
        if request.path == "/healthz":
            policy = getattr(self, "_policy", None)
            health_method = getattr(policy, "gpu_health", None)
            if not callable(health_method):
                # Keep the lightweight websocket contract usable by callers
                # that construct the server facade for protocol-only checks.
                response = connection.respond(
                    http.HTTPStatus.OK,
                    json.dumps({"ok": True, **self._health_identity}, separators=(",", ":")),
                )
                del response.headers["Content-Type"]
                response.headers["Content-Type"] = "application/json"
                return response
            health = health_method()
            healthy = bool(health.get("ok") and health.get("model_loaded"))
            response = connection.respond(
                http.HTTPStatus.OK if healthy else http.HTTPStatus.SERVICE_UNAVAILABLE,
                json.dumps({"ok": healthy, **self._health_identity, "gpu": health}, separators=(",", ":")),
            )
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            return response
        if self._api_key:
            got = request.headers.get("Authorization", "")
            expected = f"Api-Key {self._api_key}"
            if got != expected:
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM websocket server")
    parser.add_argument("--config", type=str, default=None, help="SuperInference YAML config path")
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name under configs/",
    )
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--action-infer-mode",
        choices=sorted(_ACTION_INFER_MODES),
        default=None,
    )
    parser.add_argument(
        "--action-video-attention-mode",
        choices=("first_frame", "full_video"),
        default=None,
    )
    parser.add_argument("--c4-video-denoise-steps", type=int, default=None)
    parser.add_argument(
        "--c4-video-num-inference-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--c4-reverse-video-timestep",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rtc-use-vjp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use exact RTC Jacobian VJP (override policy.params.rtc_use_vjp)",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config_name = os.environ.get(CONFIG_NAME_ENV) or args.config_name
    progress = _StartupProgressReporter(
        (
            Path(os.environ[PROGRESS_FILE_ENV]).expanduser().resolve(strict=False)
            if os.environ.get(PROGRESS_FILE_ENV)
            else None
        ),
        run_id=os.environ.get(RUN_ID_ENV),
        config_name=config_name,
    )
    profiler: Optional[InferenceProfiler] = None
    try:
        progress.update("validating_config")
        root_cfg = _load_superinference_config(args.config, config_name)
        params = _build_server_params(root_cfg, args)
        profiler = InferenceProfiler(root_cfg.get("profiling"), component="server")
        runtime = FastWAMRuntime(
            params,
            load_model=not args.validate_only,
            profiler=profiler,
            startup_progress=progress.update,
        )
        metadata = runtime.metadata()
        logger.info(f"FastWAM runtime metadata:\n{json.dumps(metadata, indent=2)}")

        if args.validate_only:
            print(json.dumps(metadata, indent=2))
            return

        host = _normalize_bind_host(params.get("host", "127.0.0.1"))
        port = int(params.get("port", 18090))
        server = FastWAMWebsocketServer(
            policy=runtime,
            host=host,
            port=port,
            metadata=metadata,
            api_key=params.get("api_key"),
            ready_file=(
                Path(os.environ[READY_FILE_ENV]).expanduser().resolve(strict=False)
                if os.environ.get(READY_FILE_ENV)
                else None
            ),
            run_id=os.environ.get(RUN_ID_ENV),
            config_name=config_name,
            startup_progress=progress.update,
        )
        server.serve_forever()
    except Exception:
        progress.update("failed")
        raise
    finally:
        if profiler is not None:
            profiler.finalize()


if __name__ == "__main__":
    main()
