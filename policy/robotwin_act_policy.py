"""Adapter for RoboTwin's historical standalone ACT joint14 checkpoint."""

from __future__ import annotations

import importlib
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from integrations.robotwin.contracts import validate_action
from policy.base import BasePolicy, ObservationType


class RoboTwinACTPolicy(BasePolicy):
    """Run an official RoboTwin ACT checkpoint without XPolicyLab.

    ``act_source`` points at the directory containing ``ACT/``.  The supplied
    bootstrap script can export that package from the local RoboTwin git history
    without modifying the user's RoboTwin checkout.
    """

    def __init__(
        self,
        *,
        ckpt_dir: str,
        act_source: str = "third_party/robotwin_act/policy",
        robot_device: str = "RoboTwinSimRobot_0",
        camera_devices: Optional[Mapping[str, str]] = None,
        device: str = "cuda:0",
        inference_dtype: str = "float32",
        chunk_size: int = 50,
        model_config: Optional[Mapping[str, Any]] = None,
        image_height: int = 480,
        image_width: int = 640,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        self.act_source = Path(act_source).expanduser().resolve()
        self.robot_device = str(robot_device)
        self.camera_devices = dict(
            camera_devices
            or {
                "head_camera": "RoboTwinCamera_0",
                "left_camera": "RoboTwinCamera_1",
                "right_camera": "RoboTwinCamera_2",
            }
        )
        self.device = str(device)
        self.inference_dtype = str(inference_dtype).strip().lower()
        self.chunk_size = int(chunk_size)
        self.model_config = dict(model_config or {})
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self._torch: Any = None
        self._model: Any = None
        self._model_dtype: Any = None
        self._last_prediction_timing_ns: Dict[str, int] = {}

    def load(self) -> None:
        checkpoint = self.ckpt_dir / "policy_last.ckpt"
        stats = self.ckpt_dir / "dataset_stats.pkl"
        if not checkpoint.is_file() or not stats.is_file():
            raise FileNotFoundError(
                "RoboTwin ACT requires policy_last.ckpt and dataset_stats.pkl in "
                f"{self.ckpt_dir}"
            )
        if not (self.act_source / "ACT").is_dir():
            raise FileNotFoundError(
                f"RoboTwin ACT source is missing {self.act_source / 'ACT'}; "
                "run scripts/robotwin/bootstrap_act_source.py first"
            )
        source_string = str(self.act_source)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)
        try:
            torch = importlib.import_module("torch")
            act_module = importlib.import_module("ACT.act_policy")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the standalone RoboTwin ACT runtime. Install its optional "
                "torch/torchvision dependencies and bootstrap ACT source."
            ) from exc

        args = {
            "action_dim": 14,
            "backbone": "resnet18",
            "camera_names": ["head_cam", "left_cam", "right_cam"],
            "chunk_size": self.chunk_size,
            "ckpt_dir": str(self.ckpt_dir),
            "dec_layers": 7,
            "device": self.device,
            "dilation": False,
            "dim_feedforward": 3200,
            "dropout": 0.1,
            "enc_layers": 4,
            "hidden_dim": 512,
            "kl_weight": 10.0,
            "lr": 0.00001,
            "lr_backbone": 0.00001,
            "masks": False,
            "nheads": 8,
            "num_epochs": 2000,
            "policy_class": "ACT",
            "position_embedding": "sine",
            "pre_norm": False,
            "temporal_agg": False,
            "weight_decay": 0.0001,
        }
        args.update(self.model_config)
        args["device"] = self.device
        if int(args.get("action_dim", 14)) != 14:
            raise ValueError("RoboTwin ACT adapter only supports action_dim=14")
        if str(args.get("backbone", "resnet18")) != "resnet18":
            raise ValueError(
                "RoboTwin ACT adapter only supports the checkpoint's resnet18 backbone"
            )
        if int(args.get("chunk_size", self.chunk_size)) != self.chunk_size:
            raise ValueError(
                "model_config.chunk_size must match RoboTwinACTPolicy.chunk_size"
            )
        if list(args.get("camera_names", [])) != ["head_cam", "left_cam", "right_cam"]:
            raise ValueError(
                "RoboTwin ACT adapter requires camera_names "
                "['head_cam', 'left_cam', 'right_cam'] in that order"
            )
        if not self.device.startswith("cuda"):
            raise ValueError(
                "the historical RoboTwin ACT runtime requires a CUDA device"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RoboTwin ACT requires CUDA, but torch.cuda.is_available() is false"
            )
        try:
            model_dtype = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
            }[self.inference_dtype]
        except KeyError as exc:
            raise ValueError(
                "RoboTwin ACT inference_dtype must be float32 or float16"
            ) from exc
        self._torch = torch
        self._model_dtype = model_dtype
        # ACT's constructor uses this object as the full DETR configuration,
        # not merely as an action-dimension holder. Construct and restore on
        # CPU first so checkpoint deserialization never temporarily duplicates
        # the model on the same GPU used by SAPIEN.
        self._model = self._build_model_on_cpu(act_module, args, torch)
        if model_dtype == torch.float16:
            self._model.policy.half()
        self._model.policy.to(self.device)
        self._model.device = torch.device(self.device)
        self.initialized = True

    @classmethod
    def _build_model_on_cpu(
        cls, act_module: Any, args: dict[str, Any], torch: Any
    ) -> Any:
        """Restore ACT weights on CPU before allocating the inference model on CUDA."""
        cpu_args = dict(args)
        cpu_args["device"] = "cpu"
        original_load = torch.load

        def load_on_cpu(*load_args: Any, **load_kwargs: Any) -> Any:
            load_kwargs.setdefault("map_location", "cpu")
            return original_load(*load_args, **load_kwargs)

        torch.load = load_on_cpu
        try:
            return cls._build_model_without_backbone_download(act_module, cpu_args)
        finally:
            torch.load = original_load

    @staticmethod
    def _build_model_without_backbone_download(
        act_module: Any, args: dict[str, Any]
    ) -> Any:
        """Construct historical ACT without fetching ImageNet weights.

        The ACT checkpoint restores the entire ResNet state. Its old backbone
        factory nevertheless asks torchvision for pretrained weights during
        construction, which can block or fail in an offline deployment.
        """
        torchvision = importlib.import_module("torchvision")
        original_factory = torchvision.models.resnet18

        def local_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
            factory_kwargs["pretrained"] = False
            return original_factory(*factory_args, **factory_kwargs)

        torchvision.models.resnet18 = local_factory
        try:
            return act_module.ACT(args, SimpleNamespace(**args))
        finally:
            torchvision.models.resnet18 = original_factory

    def predict(
        self,
        observation: ObservationType,
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        **_: Any,
    ) -> Dict[str, np.ndarray]:
        if not self.initialized or self._model is None or self._torch is None:
            raise RuntimeError(
                "RoboTwinACTPolicy.load() must be called before predict()"
            )
        if int(chunk_length) > self.chunk_size:
            raise ValueError(
                f"requested action chunk {chunk_length} exceeds ACT checkpoint chunk_size {self.chunk_size}"
            )
        self._last_prediction_timing_ns = {}
        predict_started_ns = time.perf_counter_ns()
        control_device = self._resolve_control_device(action_configs)
        qpos = self._observation_array(observation, self.robot_device)
        qpos = validate_action(qpos, "joint14")
        images = self._prepare_images(observation)
        qpos_normalized = self._model.pre_process(qpos)
        input_preprocess_finished_ns = time.perf_counter_ns()

        cuda_events = self._prediction_cuda_events()
        if cuda_events is not None:
            cuda_events["h2d_start"].record()
        qpos_tensor = self._torch.from_numpy(
            np.asarray(qpos_normalized, dtype=np.float32)
        ).to(
            self._model.device,
            dtype=self._model_dtype,
        )
        qpos_tensor = qpos_tensor.unsqueeze(0)
        image_tensor = (
            self._torch.from_numpy(images)
            .to(self._model.device, dtype=self._model_dtype)
            .unsqueeze(0)
        )
        if cuda_events is not None:
            cuda_events["h2d_end"].record()
        autocast_context = (
            self._torch.autocast(device_type="cuda", dtype=self._model_dtype)
            if self._model_dtype == self._torch.float16
            else nullcontext()
        )
        if cuda_events is not None:
            cuda_events["forward_start"].record()
        with self._torch.no_grad(), autocast_context:
            raw_actions = self._model.policy(qpos_tensor, image_tensor)
        if cuda_events is not None:
            cuda_events["forward_end"].record()
        output_transfer_started_ns = time.perf_counter_ns()
        raw_actions_numpy = raw_actions.detach().cpu().numpy()
        output_transfer_finished_ns = time.perf_counter_ns()
        actions = self._model.post_process(raw_actions_numpy)
        postprocess_finished_ns = time.perf_counter_ns()
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 14:
            raise ValueError(
                f"RoboTwin ACT returned invalid action shape {actions.shape}"
            )
        timing = {
            "input_preprocess_ns": input_preprocess_finished_ns - predict_started_ns,
            "output_sync_d2h_ns": output_transfer_finished_ns
            - output_transfer_started_ns,
            "postprocess_ns": postprocess_finished_ns - output_transfer_finished_ns,
            "policy_predict_ns": postprocess_finished_ns - predict_started_ns,
        }
        timing.update(self._cuda_event_timing_ns(cuda_events))
        self._last_prediction_timing_ns = timing
        return {control_device: actions[0, : int(chunk_length)].copy()}

    @property
    def last_prediction_timing_ns(self) -> Dict[str, int]:
        """Return the most recent successful predict() timing breakdown."""
        return dict(self._last_prediction_timing_ns)

    def _prediction_cuda_events(self) -> Optional[Dict[str, Any]]:
        """Allocate best-effort CUDA timing events without requiring them in tests."""
        cuda = getattr(self._torch, "cuda", None)
        event_factory = getattr(cuda, "Event", None)
        if not callable(event_factory):
            return None
        try:
            return {
                "h2d_start": event_factory(enable_timing=True),
                "h2d_end": event_factory(enable_timing=True),
                "forward_start": event_factory(enable_timing=True),
                "forward_end": event_factory(enable_timing=True),
            }
        except Exception:
            return None

    def _cuda_event_timing_ns(self, events: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Read CUDA stream durations after output transfer has synchronized the stream."""
        if events is None:
            return {}
        cuda = getattr(self._torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        try:
            if callable(synchronize):
                synchronize(self._model.device)
            return {
                "h2d_ns": int(
                    events["h2d_start"].elapsed_time(events["h2d_end"]) * 1_000_000
                ),
                "cuda_forward_ns": int(
                    events["forward_start"].elapsed_time(events["forward_end"])
                    * 1_000_000
                ),
            }
        except Exception:
            return {}

    def _resolve_control_device(self, action_configs: List[Dict[str, Any]]) -> str:
        matching = [
            config
            for config in action_configs
            if int(config.get("action_dim", 0)) == 14
        ]
        if len(matching) != 1:
            raise ValueError(
                f"RoboTwin ACT expects exactly one joint14 control target, got {len(matching)}"
            )
        return str(matching[0]["device_name"])

    def _prepare_images(self, observation: ObservationType) -> np.ndarray:
        images: list[np.ndarray] = []
        for camera_name in ("head_camera", "left_camera", "right_camera"):
            device_name = self.camera_devices.get(camera_name)
            if not device_name:
                raise ValueError(
                    f"RoboTwin ACT camera mapping is missing {camera_name!r}"
                )
            image = self._observation_array(observation, device_name)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"RoboTwin ACT camera {device_name!r} must be HxWx3, got {image.shape}"
                )
            image = self._resize_if_needed(image)
            images.append(np.moveaxis(image, -1, 0).astype(np.float32) / 255.0)
        return np.stack(images, axis=0)

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (self.image_height, self.image_width):
            return image
        try:
            cv2 = importlib.import_module("cv2")
        except Exception as exc:
            raise RuntimeError(
                "RoboTwin ACT received a non-native camera size and requires optional OpenCV for resize"
            ) from exc
        return cv2.resize(
            image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR
        )

    @staticmethod
    def _observation_array(
        observation: ObservationType, device_name: str
    ) -> np.ndarray:
        if not isinstance(observation, dict) or device_name not in observation:
            raise ValueError(
                f"RoboTwin ACT observation does not contain {device_name!r}"
            )
        value = observation[device_name]
        if isinstance(value, (list, tuple)) and value:
            value = value[-1]
        return np.asarray(value)
