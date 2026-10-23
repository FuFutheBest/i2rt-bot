"""Hardware-free RTC contracts for the FastWAM model server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

pytest.importorskip(
    "torch",
    reason="FastWAM RTC contracts require the optional PyTorch model runtime",
)

import torch
from omegaconf import OmegaConf

from consumer.fastwam_server import FastWAMRuntime


class _IdentityNormalizer:
    def forward(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)


class _AccelerationModel:
    def __init__(self) -> None:
        self.action_denoise_compile_enabled = False
        self.events: list[str] = []
        self.encoded_prompts: list[str] = []
        self.infer_kwargs: list[Dict[str, Any]] = []

    def compile_action_denoise(
        self,
        *,
        mode: str,
        dynamic: Optional[bool],
        fullgraph: bool,
    ) -> bool:
        self.events.append(f"compile:{mode}:{dynamic}:{fullgraph}")
        self.action_denoise_compile_enabled = True
        return True

    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        self.events.append(f"encode:{prompt}")
        self.encoded_prompts.append(prompt)
        context = torch.full((1, 2, 3), float(len(self.encoded_prompts)))
        context_mask = torch.ones((1, 2), dtype=torch.bool)
        return context, context_mask

    def infer_action(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        self.events.append("infer")
        self.infer_kwargs.append(kwargs)
        return {"action": torch.zeros((3, 2))}


class _FutureCacheModel(_AccelerationModel):
    def infer_action_one_pass_future_cache(
        self,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.events.append("infer-one-pass")
        self.infer_kwargs.append(kwargs)
        return {
            "action": torch.zeros((3, 2)),
            "future_cache_metadata": {"mode": "one_pass_future_cache"},
        }


class _StrictObservedFrameModel:
    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] | None = None

    def infer_action(
        self,
        *,
        prompt: Any,
        context: Any,
        context_mask: Any,
        input_image: Any,
        action_horizon: int,
        proprio: Any,
        num_inference_steps: int,
        sigma_shift: Any,
        seed: Any,
        rand_device: str,
        tiled: bool,
        prefix_actions: Any,
        inference_delay: Any,
        prefix_attention_horizon: Any,
        max_guidance_weight: Any,
        profiler: Any,
    ) -> Dict[str, torch.Tensor]:
        self.kwargs = locals().copy()
        self.kwargs.pop("self")
        return {"action": torch.zeros((3, 2))}


def _image_preprocess_runtime(history_frames: int) -> FastWAMRuntime:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {
        "crop_input_to_aspect": None,
        "dump_input_images": False,
    }
    runtime.contract = {
        "condition_history_frames": history_frames,
        "image_meta": [
            {"key": "left", "shape": [3, 2, 2]},
            {"key": "right", "shape": [3, 2, 2]},
        ],
        "concat_multi_camera": "horizontal",
        "video_size": [2, 4],
    }
    runtime._last_preprocessed_camera_frames = []
    runtime._dump_input_count = 0
    return runtime


def _constant_rgb(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def _contract_runtime(
    *,
    history_frames: int,
    history_stride: int,
    action_infer_mode: str,
    attention_mode: str,
) -> FastWAMRuntime:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.ckpt_root = Path("/tmp/fastwam-contract")
    runtime.weight_path = runtime.ckpt_root / "checkpoints/weights/step.pt"
    runtime.params = {
        "condition_history_frames": history_frames,
        "obs_horizon": history_frames,
        "action_infer_mode": action_infer_mode,
        "action_video_attention_mode": attention_mode,
        "enforce_checkpoint_infer_mode": True,
    }
    runtime.video_visualization_config = {"enabled": False}
    runtime.training_cfg = OmegaConf.create(
        {
            "framework": {
                "name": "FastWAM",
                "fastwam": {
                    "default_action_infer_mode": action_infer_mode,
                    "action_video_attention_mode": attention_mode,
                    "enable_video_prediction": True,
                    "video_dit_config": {
                        "seperated_timestep": True,
                        "fuse_vae_embedding_in_latents": True,
                    },
                    "video_scheduler": {
                        "num_train_timesteps": 1000,
                        "train_shift": 5.0,
                        "infer_shift": 5.0,
                    },
                    "action_scheduler": {
                        "num_train_timesteps": 1000,
                        "train_shift": 5.0,
                        "infer_shift": 5.0,
                    },
                },
            },
            "datasets": {
                "vla_data": {
                    "shape_meta": {
                        "images": [
                            {"key": "left", "shape": [3, 2, 2]},
                            {"key": "right", "shape": [3, 2, 2]},
                        ],
                        "action": [{"shape": 2}],
                        "state": [{"shape": 2}],
                    },
                    "processor": {
                        "action_output_dim": 2,
                        "proprio_output_dim": 2,
                    },
                    "action_transform": {"chunk_size_steps": 3},
                    "condition_history_frames": history_frames,
                    "condition_history_stride": history_stride,
                    "num_frames": 49,
                    "action_video_freq_ratio": 4,
                    "video_size": [2, 4],
                    "concat_multi_camera": "horizontal",
                },
            },
        }
    )
    return runtime


def test_server_normalizes_and_forwards_rtc_prefix() -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.enable_rtc = True
    runtime.contract = {"action_horizon": 3, "action_dim": 2}
    runtime.action_normalizer = _IdentityNormalizer()
    prefix = np.arange(6, dtype=np.float64).reshape(3, 2)

    prefix_tensor, rtc_config = runtime._prepare_rtc_inputs(
        {
            "prefix_action": prefix,
            "rtc_config": {
                "inference_delay": 2,
                "prefix_attention_horizon": 1,
                "max_guidance_weight": 10.0,
            },
        }
    )

    assert isinstance(prefix_tensor, torch.Tensor)
    assert prefix_tensor.dtype == torch.float32
    np.testing.assert_allclose(prefix_tensor.numpy(), prefix)
    assert rtc_config == {
        "inference_delay": 2,
        "prefix_attention_horizon": 1,
        "max_guidance_weight": 10.0,
    }


def test_server_contract_reads_memory_and_fasterwam_modes() -> None:
    memory_runtime = _contract_runtime(
        history_frames=4,
        history_stride=4,
        action_infer_mode="observed_frame_cache",
        attention_mode="first_frame",
    )
    faster_runtime = _contract_runtime(
        history_frames=1,
        history_stride=1,
        action_infer_mode="one_pass_future_cache",
        attention_mode="full_video",
    )

    memory_runtime.contract = memory_runtime._build_contract()
    faster_runtime.contract = faster_runtime._build_contract()
    memory_runtime._validate_runtime_alignment()
    faster_runtime._validate_runtime_alignment()

    assert memory_runtime.contract["condition_history_stride"] == 4
    assert faster_runtime.contract["action_infer_mode"] == "one_pass_future_cache"
    assert faster_runtime.contract["sampled_video_num_frames"] == 13


def test_server_configures_compile_prompt_cache_and_warmup() -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {
        "task": "Fold the towel",
        "torch_compile": {
            "enabled": True,
            "mode": "reduce-overhead",
            "dynamic": False,
            "fullgraph": False,
        },
        "cache_prompt_embeddings": True,
        "prompt_cache_max_entries": 2,
        "warm_up_on_load": True,
        "warmup_requests": 2,
        "num_inference_steps": 10,
    }
    runtime.profiler = None
    runtime.contract = {
        "prompt_template": "Instruction: {task}",
        "video_size": [4, 6],
        "state_dim": 2,
        "action_horizon": 3,
    }
    runtime.model = _AccelerationModel()
    runtime._initialize_inference_acceleration_settings()

    runtime._configure_inference_acceleration()

    assert runtime.model.events == [
        "compile:reduce-overhead:False:False",
        "encode:Instruction: Fold the towel",
        "infer",
        "infer",
    ]
    assert runtime._warmup_completed_requests == 2
    assert list(runtime._prompt_embedding_cache) == ["Instruction: Fold the towel"]
    for kwargs in runtime.model.infer_kwargs:
        assert kwargs["prompt"] is None
        assert kwargs["context"] is not None
        assert kwargs["context_mask"] is not None
        assert tuple(kwargs["input_image"].shape) == (3, 4, 6)


def test_server_prompt_cache_uses_lru_eviction() -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {
        "cache_prompt_embeddings": True,
        "prompt_cache_max_entries": 1,
    }
    runtime.model = _AccelerationModel()
    runtime._initialize_inference_acceleration_settings()

    runtime._resolve_prompt_conditioning("task-a")
    runtime._resolve_prompt_conditioning("task-a")
    runtime._resolve_prompt_conditioning("task-b")
    runtime._resolve_prompt_conditioning("task-a")

    assert runtime.model.encoded_prompts == ["task-a", "task-b", "task-a"]
    assert list(runtime._prompt_embedding_cache) == ["task-a"]


def test_server_forwards_disabled_vjp_mode_to_model() -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {"num_inference_steps": 10}
    runtime.profiler = None
    runtime.rtc_use_vjp = False
    runtime.contract = {"action_horizon": 3}
    runtime.model = _AccelerationModel()
    runtime._format_prompt = lambda _prompt: "task"
    runtime._resolve_prompt_conditioning = lambda _prompt: (
        None,
        torch.zeros((1, 1, 2)),
        torch.ones((1, 1), dtype=torch.bool),
    )
    runtime._preprocess_images = lambda _images: torch.zeros((3, 4, 6))
    runtime._preprocess_state = lambda _state: torch.zeros(2)
    runtime._prepare_rtc_inputs = lambda _obs: (
        torch.zeros((3, 2)),
        {
            "inference_delay": 1,
            "prefix_attention_horizon": 2,
            "max_guidance_weight": 2.0,
        },
    )
    runtime._postprocess_action = lambda action: np.asarray(action)

    result = runtime.infer({"images": [], "state": []})

    assert runtime.model.infer_kwargs[-1]["rtc_use_vjp"] is False
    assert runtime.model.infer_kwargs[-1]["max_guidance_weight"] == 2.0
    assert result["data"]["rtc_use_vjp"] is False


def test_server_preprocesses_memory_history_as_channel_time_height_width() -> None:
    runtime = _image_preprocess_runtime(history_frames=4)
    image_history = [
        [_constant_rgb(10 + timestep), _constant_rgb(110 + timestep)]
        for timestep in range(4)
    ]

    input_video = runtime._preprocess_image_history(image_history)

    assert tuple(input_video.shape) == (3, 4, 2, 4)
    for timestep in range(4):
        torch.testing.assert_close(
            input_video[:, timestep, :, :2],
            torch.full((3, 2, 2), (10 + timestep) / 127.5 - 1.0),
        )
        torch.testing.assert_close(
            input_video[:, timestep, :, 2:],
            torch.full((3, 2, 2), (110 + timestep) / 127.5 - 1.0),
        )


def test_observed_frame_inference_drops_c4_only_kwargs() -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {}
    runtime.rtc_use_vjp = False
    runtime.contract = {"action_infer_mode": "observed_frame_cache"}
    runtime.model = _StrictObservedFrameModel()

    runtime._infer_model_action(
        prompt=None,
        context=None,
        context_mask=None,
        input_image=torch.zeros((3, 2, 2)),
        action_horizon=3,
        proprio=torch.zeros(2),
        num_inference_steps=2,
        c4_video_denoise_steps=0,
        c4_video_num_inference_steps=2,
        c4_reverse_video_timestep=False,
        sigma_shift=None,
        seed=None,
        rand_device="cpu",
        tiled=False,
        prefix_actions=None,
        inference_delay=None,
        prefix_attention_horizon=None,
        max_guidance_weight=None,
        profiler=None,
    )

    assert runtime.model.kwargs is not None
    assert "c4_video_denoise_steps" not in runtime.model.kwargs
    assert "rtc_use_vjp" not in runtime.model.kwargs


def test_server_dispatches_fasterwam_to_one_pass_future_cache() -> None:
    runtime = FastWAMRuntime.__new__(FastWAMRuntime)
    runtime.params = {"num_inference_steps": 2}
    runtime.profiler = None
    runtime.rtc_use_vjp = False
    runtime.enable_rtc = False
    runtime.contract = {
        "condition_history_frames": 1,
        "action_infer_mode": "one_pass_future_cache",
        "sampled_video_num_frames": 13,
        "action_horizon": 3,
    }
    runtime.model = _FutureCacheModel()
    runtime._format_prompt = lambda _prompt: "task"
    runtime._resolve_prompt_conditioning = lambda _prompt: (None, None, None)
    runtime._preprocess_images = lambda _images: torch.zeros((3, 2, 2))
    runtime._preprocess_state = lambda _state: torch.zeros(2)
    runtime._prepare_rtc_inputs = lambda _obs: (None, None)
    runtime._postprocess_action = lambda action: np.asarray(action)

    result = runtime.infer({"images": [], "state": []})

    assert runtime.model.events == ["infer-one-pass"]
    assert runtime.model.infer_kwargs[-1]["num_video_frames"] == 13
    assert runtime.model.infer_kwargs[-1]["c4_video_denoise_steps"] == 0
    assert runtime.model.infer_kwargs[-1]["c4_video_num_inference_steps"] == 2
    assert result["data"]["future_cache_metadata"] == {
        "mode": "one_pass_future_cache"
    }
