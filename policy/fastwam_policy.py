#!/usr/bin/env python3
"""FastWAM websocket policy adapter for SuperInference."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import websockets.sync.client
from scipy.spatial.transform import Rotation

from consumer.inference.fastwam_model_registry import (
    get_fastwam_model_variant,
    validate_fastwam_variant_config,
    validate_fastwam_variant_runtime_contract,
)
from policy.base import BasePolicy
from utils.calibration_utils import load_tcp_umi_transforms
from utils.latency_tracker import LatencyTracker
from utils.logger_config import logger
from utils.msgpack_numpy import Packer, unpackb
from utils.transform import process_quaternion


def _as_bool(value: Any) -> bool:
    """Parse YAML/Hydra booleans without treating ``"false"`` as true."""
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


def _format_websocket_uri(host: str, port: int) -> str:
    """Format a websocket endpoint, including bracketed IPv6 literals."""
    normalized_host = str(host).strip()
    if normalized_host.startswith("[") and normalized_host.endswith("]"):
        endpoint_host = normalized_host
    elif ":" in normalized_host:
        endpoint_host = f"[{normalized_host}]"
    else:
        endpoint_host = normalized_host
    return f"ws://{endpoint_host}:{int(port)}"


class FastWAMPolicy(BasePolicy):
    """Websocket client for the native nmx_vla FastWAM inference server."""

    supports_history_frame_timestamps = True

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18090,
        api_key: Optional[str] = None,
        connect_timeout: float = 300.0,
        task: str = "",
        camera_devices: Optional[List[str]] = None,
        robot_device: Optional[str] = None,
        reset_on_load: bool = True,
        gripper_max_width: float = 0.11,
        normalize_gripper_width: bool = False,
        clip_gripper_width: bool = True,
        print_predicted_gripper_action: bool = False,
        enable_rtc: bool = False,
        execution_fps: Optional[float] = None,
        chunk_length: int = 1,
        rtc_latency_window: int = 100,
        rtc_max_guidance_weight: float = 10.0,
        rtc_guidance_full_steps_override: Optional[int] = None,
        rtc_delay_margin_steps: int = 0,
        rtc_use_vjp: bool = True,
        model_variant: Optional[str] = None,
        condition_history_frames: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        obs_time_interval: float = 1.0 / 30.0,
        downsample: Optional[int] = None,
        condition_history_stride: Optional[int] = None,
        action_infer_mode: str = "observed_frame_cache",
        action_video_attention_mode: str = "first_frame",
        video_visualization: Optional[Dict[str, Any]] = None,
        tcp_umi_json_path: Optional[str] = None,
        profiler: Any = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.host = str(host)
        self.port = int(port)
        self.api_key = api_key
        self.connect_timeout = float(connect_timeout)
        self.task = str(task)
        self.camera_devices = list(camera_devices) if camera_devices else []
        self.robot_device = robot_device
        self.reset_on_load = _as_bool(reset_on_load)
        self.gripper_max_width = float(gripper_max_width)
        self.normalize_gripper_width = _as_bool(normalize_gripper_width)
        self.clip_gripper_width = _as_bool(clip_gripper_width)
        self.print_predicted_gripper_action = _as_bool(print_predicted_gripper_action)
        self.enable_rtc = _as_bool(enable_rtc)
        self.execution_fps = (
            None if execution_fps is None else float(execution_fps)
        )
        self.chunk_length = int(chunk_length)
        if self.chunk_length <= 0:
            raise ValueError("FastWAM chunk_length must be positive")
        self.profiler = profiler
        if isinstance(video_visualization, bool):
            video_visualization = {"enabled": video_visualization}
        self.video_visualization_config: Dict[str, Any] = dict(video_visualization or {})
        self.video_visualization_enabled = _as_bool(
            self.video_visualization_config.get("enabled", False)
        )
        self.video_prediction_save_enabled = _as_bool(
            self.video_visualization_config.get("save_predictions", False)
        )
        self.video_visualization_interval = max(
            int(self.video_visualization_config.get("prediction_interval", 5)),
            1,
        )
        self.video_visualization_disable_rtc = _as_bool(
            self.video_visualization_config.get("disable_rtc_during_prediction", True)
        )
        self._video_visualization_request_count = 0
        self._video_visualization_failure_count = 0
        self._video_visualization_requests_disabled = False
        self._video_visualization_suppressed = False
        self._video_visualizer: Any = None
        if self.enable_rtc and (
            self.execution_fps is None or self.execution_fps <= 0
        ):
            raise ValueError("FastWAM RTC requires a positive execution_fps")
        self._latency_tracker = LatencyTracker(maxlen=int(rtc_latency_window))
        if not isinstance(rtc_use_vjp, bool):
            raise ValueError("rtc_use_vjp must be a boolean (true or false)")
        self.rtc_use_vjp = rtc_use_vjp
        self.rtc_max_guidance_weight = float(rtc_max_guidance_weight)
        if not np.isfinite(self.rtc_max_guidance_weight) or self.rtc_max_guidance_weight <= 0:
            raise ValueError(
                "rtc_max_guidance_weight must be finite and positive, got "
                f"{rtc_max_guidance_weight!r}"
            )
        self.rtc_guidance_full_steps_override = self._parse_optional_nonnegative_int(
            rtc_guidance_full_steps_override,
            "rtc_guidance_full_steps_override",
        )
        self.rtc_delay_margin_steps = self._parse_nonnegative_int(
            rtc_delay_margin_steps,
            "rtc_delay_margin_steps",
        )

        configured_history_values = [
            int(value)
            for value in (condition_history_frames, obs_horizon)
            if value is not None
        ]
        if len(set(configured_history_values)) > 1:
            raise ValueError(
                "condition_history_frames and obs_horizon must match when both are set"
            )
        self.condition_history_frames = (
            configured_history_values[0] if configured_history_values else 1
        )
        if self.condition_history_frames not in (1, 4):
            raise ValueError(
                "FastWAM inference supports condition_history_frames=1 or 4, "
                f"got {self.condition_history_frames}"
            )
        self.obs_horizon = self.condition_history_frames
        self.obs_time_interval = float(obs_time_interval)
        if self.obs_time_interval <= 0.0:
            raise ValueError("obs_time_interval must be positive")
        configured_stride_values = [
            int(value)
            for value in (condition_history_stride, downsample)
            if value is not None
        ]
        if len(set(configured_stride_values)) > 1:
            raise ValueError(
                "condition_history_stride and downsample must match when both are set"
            )
        self._configured_condition_history_stride: Optional[int] = (
            configured_stride_values[0] if configured_stride_values else None
        )
        self._set_condition_history_stride(
            self._configured_condition_history_stride or 1
        )
        self.action_infer_mode = str(action_infer_mode).strip().lower()
        self.action_video_attention_mode = str(
            action_video_attention_mode
        ).strip().lower()
        if self.action_infer_mode not in {
            "observed_frame_cache",
            "one_pass_future_cache",
        }:
            raise ValueError(
                "action_infer_mode must be 'observed_frame_cache' or "
                f"'one_pass_future_cache', got {action_infer_mode!r}"
            )
        if self.action_video_attention_mode not in {"first_frame", "full_video"}:
            raise ValueError(
                "action_video_attention_mode must be 'first_frame' or "
                f"'full_video', got {action_video_attention_mode!r}"
            )
        self.model_variant = (
            None if model_variant is None else str(model_variant).strip().lower()
        )
        self._model_variant_contract = None
        if self.model_variant:
            self._model_variant_contract = validate_fastwam_variant_config(
                self.model_variant,
                {
                    "model_variant": self.model_variant,
                    "condition_history_frames": self.condition_history_frames,
                    "obs_horizon": self.obs_horizon,
                    "action_infer_mode": self.action_infer_mode,
                    "action_video_attention_mode": self.action_video_attention_mode,
                },
            )
        else:
            # Keep direct construction backward compatible while still exposing
            # an inspectable contract to callers that need it.
            self._model_variant_contract = None
        self.min_infer_obs = (
            (self.obs_horizon - 1) * self.condition_history_stride + 1
        )
        self._uri = _format_websocket_uri(self.host, self.port)
        self._packer = Packer()
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: Dict[str, Any] = {}
        self.relative_pose_frame = "local_frame"
        self.action_quaternion_order = "xyzw"
        self.tcp_umi_json_path: Optional[str] = None
        self.tcp_umi_T = np.eye(4, dtype=np.float64)
        self.umi_tcp_T = np.eye(4, dtype=np.float64)
        if tcp_umi_json_path:
            calibration_path = Path(str(tcp_umi_json_path)).expanduser()
            if not calibration_path.is_absolute():
                calibration_path = Path(__file__).resolve().parents[1] / calibration_path
            tcp_umi, umi_tcp = load_tcp_umi_transforms(str(calibration_path))
            self.tcp_umi_T = self._validate_rigid_transform(tcp_umi, "tcp_umi")
            self.umi_tcp_T = self._validate_rigid_transform(umi_tcp, "umi_tcp")
            if not np.allclose(
                self.tcp_umi_T @ self.umi_tcp_T,
                np.eye(4),
                atol=1e-8,
            ):
                raise ValueError("TCP/UMI calibration matrices are not inverses")
            self.tcp_umi_json_path = str(calibration_path.resolve())
            if np.allclose(self.tcp_umi_T, np.eye(4), atol=1e-8):
                logger.warning(
                    "FastWAM TCP/UMI calibration is identity; this is valid only "
                    "when the logged UMI tool frame is aligned with the robot TCP frame"
                )
        self._prediction_context_timestamp_ns: Optional[int] = None
        self._previous_absolute_action: Optional[np.ndarray] = None
        self._last_rtc_config: Optional[Dict[str, Any]] = None
        self._last_inference_diagnostics: Dict[str, Any] = {}

    @staticmethod
    def _parse_nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        try:
            parsed = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from exc
        if not np.isfinite(numeric) or numeric != float(parsed) or parsed < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        return parsed

    @classmethod
    def _parse_optional_nonnegative_int(cls, value: Any, name: str) -> Optional[int]:
        if value is None:
            return None
        return cls._parse_nonnegative_int(value, name)

    def load(self) -> None:
        self._connect_to_server()
        self.relative_pose_frame = str(
            self._server_metadata.get("relative_pose_frame", "local_frame")
        )
        self.action_quaternion_order = str(
            self._server_metadata.get("action_quaternion_order", "xyzw")
        )
        if self.enable_rtc and not _as_bool(self._server_metadata.get("enable_rtc", False)):
            self.close()
            raise RuntimeError(
                "FastWAM policy enables RTC but the connected server does not. "
                "Start consumer/fastwam_server.py with the same config."
            )
        if self.enable_rtc and not _as_bool(
            self._server_metadata.get("rtc_use_vjp", True)
        ):
            logger.warning(
                "FastWAM RTC exact VJP is disabled; using the fast "
                "identity-Jacobian prefix-guidance approximation"
            )
        server_visualization_disabled = self._server_metadata.get(
            "video_visualization_enabled"
        ) is False
        server_prediction_unsupported = self._server_metadata.get(
            "video_prediction_supported"
        ) is False
        if self.video_visualization_enabled and (
            server_visualization_disabled or server_prediction_unsupported
        ):
            self.video_visualization_enabled = False
            logger.warning(
                "FastWAM video visualization was requested, but the server "
                "configuration/checkpoint does not support it; dashboard disabled."
            )
        if self.video_visualization_enabled and not _as_bool(
            self._server_metadata.get("video_prediction_action_conditioned", False)
        ):
            logger.warning(
                "FastWAM dashboard predictions are unconditioned world-model "
                "frames because the server checkpoint is not action-conditioned."
            )
        logger.info(f"FastWAMPolicy connected to {self._uri}: {self._server_metadata}")
        if self.reset_on_load:
            self.reset()
        self._start_video_visualizer()
        self.initialized = True

    def _connect_to_server(self) -> None:
        """Connect and validate metadata on initial load and reconnects."""
        self._ws, self._server_metadata = self._wait_for_server()
        try:
            self._validate_server_contract()
        except Exception:
            self.close()
            raise

    def _set_condition_history_stride(self, stride: int) -> None:
        """Keep the common observation builder's legacy stride alias coherent."""
        resolved_stride = int(stride)
        if resolved_stride <= 0:
            raise ValueError("condition_history_stride must be positive")
        self.condition_history_stride = resolved_stride
        self.downsample = resolved_stride
        self.min_infer_obs = (
            (self.obs_horizon - 1) * self.condition_history_stride + 1
        )

    def _validate_server_contract(self) -> None:
        """Reject policy/checkpoint pairings that would change model semantics."""
        server_history_frames = int(
            self._server_metadata.get("condition_history_frames", 1)
        )
        if server_history_frames != self.condition_history_frames:
            raise RuntimeError(
                "FastWAM visual-history mismatch: policy config requests "
                f"{self.condition_history_frames} frame(s), but the checkpoint/server "
                f"requires {server_history_frames}."
            )

        server_history_stride = int(
            self._server_metadata.get("condition_history_stride", 1)
        )
        if server_history_stride <= 0:
            raise RuntimeError(
                "FastWAM server returned invalid condition_history_stride="
                f"{server_history_stride}"
            )
        if (
            self._configured_condition_history_stride is not None
            and server_history_stride != self._configured_condition_history_stride
        ):
            raise RuntimeError(
                "FastWAM visual-history stride mismatch: policy config requests "
                f"stride {self._configured_condition_history_stride}, but the "
                f"checkpoint/server requires stride {server_history_stride}."
            )
        self._set_condition_history_stride(server_history_stride)

        server_mode = self._server_metadata.get("action_infer_mode")
        if server_mode is not None and str(server_mode).strip().lower() != self.action_infer_mode:
            raise RuntimeError(
                "FastWAM action inference mode mismatch: policy config requests "
                f"{self.action_infer_mode!r}, but the server uses {server_mode!r}"
            )
        server_attention = self._server_metadata.get("action_video_attention_mode")
        if (
            server_attention is not None
            and str(server_attention).strip().lower()
            != self.action_video_attention_mode
        ):
            raise RuntimeError(
                "FastWAM video-attention mode mismatch: policy config requests "
                f"{self.action_video_attention_mode!r}, but the server uses "
                f"{server_attention!r}"
            )

        if self.model_variant:
            get_fastwam_model_variant(self.model_variant)
            validate_fastwam_variant_runtime_contract(
                self.model_variant,
                self._server_metadata,
            )

    def close(self) -> None:
        video_visualizer = getattr(self, "_video_visualizer", None)
        if video_visualizer is not None:
            try:
                video_visualizer.close()
            except Exception as exc:
                logger.debug(f"FastWAM video visualizer close ignored: {exc}")
            finally:
                self._video_visualizer = None
        ws = getattr(self, "_ws", None)
        if ws is None:
            return
        try:
            ws.close()
        except Exception as exc:
            logger.debug(f"FastWAMPolicy websocket close ignored: {exc}")
        finally:
            self._ws = None

    def __del__(self) -> None:
        self.close()

    def reset(self) -> None:
        self._previous_absolute_action = None
        self._last_rtc_config = None
        self._last_inference_diagnostics = {}
        if self._ws is None:
            return
        self._send_request({"reset": True, "prompt": self.task})
        logger.info("FastWAMPolicy reset server state")

    def set_prediction_context(self, timestamp_ns: int) -> None:
        self._prediction_context_timestamp_ns = int(timestamp_ns)

    def _start_video_visualizer(self) -> None:
        if not self.video_visualization_enabled or self._video_visualizer is not None:
            return
        try:
            from visualizers.fastwam_video_prediction import FastWAMVideoPredictionVisualizer

            panel_size = self.video_visualization_config.get("panel_size", [640, 360])
            if not isinstance(panel_size, Sequence) or len(panel_size) != 2:
                panel_size = [640, 360]
            camera_shm_names = [
                name if name.endswith("_data") else f"{name}_data"
                for name in self.camera_devices
            ]
            self._video_visualizer = FastWAMVideoPredictionVisualizer(
                host=str(self.video_visualization_config.get("host", "127.0.0.1")),
                port=int(self.video_visualization_config.get("port", 8766)),
                display_fps=float(self.video_visualization_config.get("display_fps", 8.0)),
                panel_size=(int(panel_size[0]), int(panel_size[1])),
                jpeg_quality=int(self.video_visualization_config.get("jpeg_quality", 85)),
                camera_shm_names=camera_shm_names,
                live_camera_fps=float(
                    self.video_visualization_config.get("live_camera_fps", 30.0)
                ),
                max_prediction_frames=int(
                    self.video_visualization_config.get("max_prediction_frames", 16)
                ),
                auto_open_browser=_as_bool(
                    self.video_visualization_config.get("auto_open_browser", False)
                ),
            )
            if self._video_visualizer.start() is None:
                self._video_visualizer = None
            else:
                logger.warning(
                    "FastWAM video visualization is enabled; each requested frame "
                    "runs the additional joint video prediction path."
                )
        except Exception as exc:
            self._video_visualizer = None
            logger.warning(f"FastWAM video visualizer disabled: {exc}")

    def _wait_for_server(
        self,
    ) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        logger.info(f"Waiting for FastWAM server at {self._uri}...")
        start_time = time.time()
        for key in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(key, None)

        while True:
            if time.time() - start_time > self.connect_timeout:
                raise TimeoutError(
                    f"Failed to connect FastWAM server within {self.connect_timeout}s"
                )
            try:
                headers = (
                    {"Authorization": f"Api-Key {self.api_key}"}
                    if self.api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=20,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except Exception as exc:
                logger.info(f"Still waiting for FastWAM server: {exc}")
                time.sleep(2.0)

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            self._connect_to_server()

        assert self._ws is not None
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"FastWAM server error:\n{response}")
        result = unpackb(response)
        if isinstance(result, dict) and result.get("status") == "error":
            raise RuntimeError(f"FastWAM server error: {result.get('error')}")
        return result

    def _extract_latest_frame(self, data: Any) -> np.ndarray:
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Observation history list is empty")
            return np.asarray(data[-1])
        array = np.asarray(data)
        if (
            self.condition_history_frames > 1
            and array.ndim == 2
            and array.shape[0] == self.condition_history_frames
        ):
            return np.asarray(array[-1])
        if array.ndim >= 4:
            return np.asarray(array[-1])
        return array

    def _find_robot_device(self, observation: Dict[str, Any]) -> Optional[str]:
        if self.robot_device is not None and self.robot_device in observation:
            return self.robot_device
        for key in observation.keys():
            lowered = key.lower()
            if lowered.startswith("robot") or lowered.startswith("rizon"):
                return key
        return None

    def _extract_robot_state(self, observation: Dict[str, Any]) -> np.ndarray:
        robot_key = self._find_robot_device(observation)
        if robot_key is None:
            raise KeyError("Robot observation data not found in observation dictionary")
        robot_state = self._extract_latest_frame(observation[robot_key])
        return np.asarray(robot_state, dtype=np.float64).reshape(-1)

    def _extract_training_state(self, observation: Dict[str, Any]) -> np.ndarray:
        """Return the 16D UMI state expected by FastWAM checkpoints."""
        robot_state = self._extract_robot_state(observation)
        if robot_state.shape[0] < 44:
            raise ValueError(
                "FastWAM dual-arm state requires RizonRobot state dim >= 44, "
                f"got {robot_state.shape[0]}"
            )
        training_arms = []
        for arm_index in (0, 1):
            tcp_pos, tcp_rot, gripper = self._extract_arm_anchor(
                robot_state,
                arm_index=arm_index,
            )
            umi_pose = self._pose_matrix(tcp_pos, tcp_rot) @ self.tcp_umi_T
            umi_quat_xyzw = self._safe_quat_xyzw(
                Rotation.from_matrix(umi_pose[:3, :3]).as_quat()
            )
            umi_quat_wxyz = np.concatenate(
                [umi_quat_xyzw[3:4], umi_quat_xyzw[:3]],
                axis=0,
            )
            training_arms.append(
                np.concatenate(
                    [umi_pose[:3, 3], umi_quat_wxyz, [gripper]],
                    axis=0,
                )
            )
        training_state = np.concatenate(training_arms, axis=0)
        if not np.all(np.isfinite(training_state)):
            raise ValueError("FastWAM UMI state contains NaN or infinity")
        return training_state

    def _resolve_camera_devices(self, observation: Dict[str, Any]) -> List[str]:
        camera_devices = list(self.camera_devices)
        if not camera_devices:
            camera_devices = [key for key in observation.keys() if "camera" in key.lower()]
        if len(camera_devices) == 0:
            raise ValueError("No camera observation found for FastWAM inference")
        return camera_devices

    @staticmethod
    def _coerce_camera_frame(image: Any, camera_name: str) -> np.ndarray:
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError(
                f"FastWAM camera {camera_name} must provide HWC RGB/RGBA data, "
                f"got shape={frame.shape}"
            )
        return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))

    def _collect_images(self, observation: Dict[str, Any]) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        for camera_name in self._resolve_camera_devices(observation):
            if camera_name not in observation:
                raise KeyError(f"Missing camera observation: {camera_name}")
            image = self._extract_latest_frame(observation[camera_name])
            images.append(self._coerce_camera_frame(image, camera_name))
        return images

    def _extract_camera_history(self, data: Any, camera_name: str) -> List[np.ndarray]:
        if isinstance(data, (list, tuple)):
            frames = list(data)
        else:
            array = np.asarray(data)
            frames = list(array) if array.ndim >= 4 else [array]
        if len(frames) != self.condition_history_frames:
            raise ValueError(
                f"FastWAM camera {camera_name} requires exactly "
                f"{self.condition_history_frames} history frames, got {len(frames)}"
            )
        return [self._coerce_camera_frame(frame, camera_name) for frame in frames]

    def _collect_image_history(
        self,
        observation: Dict[str, Any],
    ) -> List[List[np.ndarray]]:
        camera_histories = []
        for camera_name in self._resolve_camera_devices(observation):
            if camera_name not in observation:
                raise KeyError(f"Missing camera observation: {camera_name}")
            camera_histories.append(
                self._extract_camera_history(observation[camera_name], camera_name)
            )
        return [
            [history[frame_index] for history in camera_histories]
            for frame_index in range(self.condition_history_frames)
        ]

    def _build_infer_payload(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": self.task,
            "state": self._extract_training_state(observation),
        }
        if self.condition_history_frames == 1:
            payload["images"] = self._collect_images(observation)
        else:
            payload["image_history"] = self._collect_image_history(observation)
            history_timestamps = observation.get("_history_frame_timestamps_ns")
            if history_timestamps is not None:
                timestamp_values = np.asarray(history_timestamps).reshape(-1)
                if len(timestamp_values) != self.condition_history_frames:
                    raise ValueError(
                        "FastWAM history timestamp count must match "
                        f"condition_history_frames={self.condition_history_frames}, "
                        f"got {len(timestamp_values)}"
                    )
                payload["history_frame_timestamps_ns"] = [
                    int(timestamp_ns) for timestamp_ns in timestamp_values
                ]
        if self._prediction_context_timestamp_ns is not None:
            payload["timestamp_ns"] = self._prediction_context_timestamp_ns
        return payload

    def _should_request_video_prediction(self) -> bool:
        if (
            not self.video_visualization_enabled
            or bool(
                getattr(self, "_video_visualization_requests_disabled", False)
            )
            or self._video_visualization_suppressed
            or (
                self._video_visualizer is None
                and not bool(getattr(self, "video_prediction_save_enabled", False))
            )
        ):
            return False
        self._video_visualization_request_count += 1
        return (
            (self._video_visualization_request_count - 1)
            % self.video_visualization_interval
            == 0
        )

    @staticmethod
    def _decode_video_frame(frame: Any) -> np.ndarray:
        if isinstance(frame, (bytes, bytearray, memoryview)):
            encoded = np.frombuffer(frame, dtype=np.uint8)
            import cv2

            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Could not decode a JPEG video prediction frame")
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        array = np.asarray(frame)
        if array.size == 0:
            raise ValueError("Video prediction frame is empty")
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Invalid video prediction frame shape: {array.shape}")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if np.issubdtype(array.dtype, np.floating):
            finite = array[np.isfinite(array)]
            if finite.size and float(finite.min()) >= -1.0 and float(finite.max()) <= 1.0:
                array = (
                    (array + 1.0) * 127.5
                    if float(finite.min()) < 0.0
                    else array * 255.0
                )
        return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8)[..., :3])

    def _split_video_prediction(
        self,
        video_prediction: Optional[Dict[str, Any]],
    ) -> Optional[List[List[np.ndarray]]]:
        if not video_prediction:
            return None
        raw_frames = video_prediction.get("frames", [])
        if not raw_frames:
            return None
        try:
            frames = [self._decode_video_frame(frame) for frame in raw_frames]
            view_count = int(video_prediction.get("view_count", 2))
            if view_count != 2:
                logger.warning(
                    f"FastWAM dashboard requires two predicted views, got {view_count}"
                )
                return None
            concat_mode = str(video_prediction.get("concat_mode", "horizontal")).lower()
            if concat_mode not in {"horizontal", "vertical"}:
                raise ValueError(f"Invalid prediction concat mode: {concat_mode!r}")
            views: List[List[np.ndarray]] = [[] for _ in range(view_count)]
            for frame in frames:
                chunks = (
                    np.array_split(frame, view_count, axis=1)
                    if concat_mode == "horizontal"
                    else np.array_split(frame, view_count, axis=0)
                )
                for view_index, chunk in enumerate(chunks):
                    views[view_index].append(np.ascontiguousarray(chunk))
            return views
        except Exception as exc:
            logger.warning(f"FastWAM video prediction frame decoding failed: {exc}")
            return None

    def _update_video_visualizer(
        self,
        observation: Dict[str, Any],
        response: Dict[str, Any],
        *,
        requested: bool,
    ) -> None:
        if self._video_visualizer is None:
            return
        try:
            data = response.get("data", response)
            video_prediction = data.get("video_prediction") if isinstance(data, dict) else None
            if requested:
                if video_prediction is None:
                    self._video_visualization_failure_count = int(
                        getattr(self, "_video_visualization_failure_count", 0)
                    ) + 1
                    max_failures = max(
                        int(self.video_visualization_config.get("max_failures", 3)),
                        1,
                    )
                    if self._video_visualization_failure_count >= max_failures:
                        self._video_visualization_requests_disabled = True
                        logger.error(
                            "FastWAM dashboard stopped requesting predictions after "
                            f"{self._video_visualization_failure_count} empty responses."
                        )
                else:
                    self._video_visualization_failure_count = 0
            prediction_views = self._split_video_prediction(video_prediction)
            if prediction_views is None:
                return

            input_frames: List[np.ndarray]
            encoded_inputs = (
                video_prediction.get("input_frames")
                if isinstance(video_prediction, dict)
                else None
            )
            if isinstance(encoded_inputs, (list, tuple)) and len(encoded_inputs) == 2:
                input_frames = [self._decode_video_frame(frame) for frame in encoded_inputs]
            else:
                input_frames = self._collect_images(observation)
            self._video_visualizer.update(
                input_frames,
                prediction_views,
                observation_index=self._video_visualization_request_count,
            )
        except Exception as exc:
            logger.debug(f"FastWAM video visualizer update ignored: {exc}")

    def _convert_arm_absolute_to_relative(
        self,
        action_chunk: np.ndarray,
        robot_state: np.ndarray,
        arm_index: int,
    ) -> np.ndarray:
        base = arm_index * 8
        anchor_pos, anchor_rot, _anchor_gripper = self._extract_arm_anchor(
            robot_state,
            arm_index=arm_index,
        )
        absolute_pos = np.asarray(
            action_chunk[:, base : base + 3],
            dtype=np.float64,
        )
        absolute_quat = self._safe_quat_xyzw(
            action_chunk[:, base + 3 : base + 7]
        )
        absolute_gripper = np.asarray(
            action_chunk[:, base + 7 : base + 8],
            dtype=np.float64,
        )

        anchor_tcp = self._pose_matrix(anchor_pos, anchor_rot)
        target_tcp = self._poses_to_matrices(absolute_pos, absolute_quat)
        anchor_umi = anchor_tcp @ self.tcp_umi_T
        target_umi = target_tcp @ self.tcp_umi_T
        relative_rotation = anchor_umi[:3, :3].T @ target_umi[:, :3, :3]
        if str(self.relative_pose_frame).lower() == "local_frame":
            relative_pose = np.linalg.inv(anchor_umi) @ target_umi
            relative_pos = relative_pose[:, :3, 3]
        elif str(self.relative_pose_frame).lower() == "world_frame":
            relative_pos = target_umi[:, :3, 3] - anchor_umi[None, :3, 3]
        else:
            raise ValueError(
                f"Invalid FastWAM relative_pose_frame: {self.relative_pose_frame!r}"
            )

        relative_quat = self._safe_quat_xyzw(
            Rotation.from_matrix(relative_rotation).as_quat()
        )
        relative_quat = self._action_quat_from_xyzw(relative_quat)
        return np.concatenate(
            [relative_pos, relative_quat, absolute_gripper],
            axis=-1,
        )

    def _convert_absolute_action_to_relative(
        self,
        action_chunk: np.ndarray,
        observation: Dict[str, Any],
    ) -> np.ndarray:
        robot_state = self._extract_robot_state(observation)
        if robot_state.shape[0] < 44:
            raise ValueError(
                "FastWAM dual-arm RTC requires RizonRobot state dim >= 44, "
                f"got {robot_state.shape[0]}"
            )
        relative_arms = [
            self._convert_arm_absolute_to_relative(
                action_chunk,
                robot_state,
                arm_index=0,
            )
        ]
        if action_chunk.shape[1] >= 16:
            relative_arms.append(
                self._convert_arm_absolute_to_relative(
                    action_chunk,
                    robot_state,
                    arm_index=1,
                )
            )
        return np.concatenate(relative_arms, axis=-1).astype(np.float64)

    @staticmethod
    def _empty_relative_prefix(
        action_horizon: int,
        action_dim: int,
        action_quaternion_order: str = "xyzw",
    ) -> np.ndarray:
        prefix = np.zeros((action_horizon, action_dim), dtype=np.float64)
        order = str(action_quaternion_order).lower()
        if order not in {"xyzw", "wxyz"}:
            raise ValueError(f"Unsupported action quaternion order: {order!r}")
        for arm_base in range(0, action_dim, 8):
            quaternion_w_index = arm_base + (6 if order == "xyzw" else 3)
            if quaternion_w_index < action_dim:
                prefix[:, quaternion_w_index] = 1.0
        return prefix

    def _build_rtc_payload(
        self,
        observation: Dict[str, Any],
        action_index: Optional[int],
    ) -> Dict[str, Any]:
        self._last_inference_diagnostics = {
            "rtc_enabled": bool(self.enable_rtc),
            "rtc_applied": False,
            "rtc_fallback_reason": None,
            "action_index": action_index,
        }
        if (
            not self.enable_rtc
            or action_index is None
            or self._previous_absolute_action is None
        ):
            self._last_rtc_config = None
            self._last_inference_diagnostics["rtc_fallback_reason"] = (
                "rtc_disabled"
                if not self.enable_rtc
                else "missing_action_index"
                if action_index is None
                else "missing_previous_action"
            )
            return {}

        model_horizon = int(
            self._server_metadata.get(
                "action_horizon",
                self._previous_absolute_action.shape[0],
            )
        )
        model_action_dim = int(
            self._server_metadata.get(
                "action_dim",
                self._previous_absolute_action.shape[1],
            )
        )
        previous_horizon = min(
            model_horizon,
            int(self._previous_absolute_action.shape[0]),
        )
        expected_previous_shape = (model_horizon, model_action_dim)
        if self._previous_absolute_action.shape != expected_previous_shape:
            raise ValueError(
                "FastWAM RTC previous action shape mismatch: expected "
                f"{expected_previous_shape}, got {self._previous_absolute_action.shape}"
            )
        assert self.execution_fps is not None
        estimated_inference_delay = self._latency_tracker.compute_inference_delay_steps(
            execution_fps=self.execution_fps,
            use_percentile=0.95,
            default_delay_steps=4,
        )
        if self.rtc_guidance_full_steps_override is not None:
            inference_delay = self.rtc_guidance_full_steps_override
            delay_source = "override"
        else:
            inference_delay = estimated_inference_delay + self.rtc_delay_margin_steps
            delay_source = "p95_plus_margin"
        prefix_start = int(action_index) + 1
        source_overlap_steps = previous_horizon - prefix_start
        guidance_horizon = source_overlap_steps - inference_delay
        if guidance_horizon <= 0:
            logger.warning(
                "FastWAM RTC fallback to normal inference: "
                f"action_index={action_index}, inference_delay={inference_delay}, "
                f"source_overlap_steps={source_overlap_steps}, "
                f"remaining_overlap_steps={guidance_horizon} <= 0"
            )
            self._last_rtc_config = None
            self._last_inference_diagnostics.update(
                {
                    "rtc_fallback_reason": "guide_does_not_fit_previous_chunk",
                    "inference_delay_steps": inference_delay,
                    "guidance_horizon_steps": guidance_horizon,
                    "available_source_steps": source_overlap_steps,
                }
            )
            return {}

        rtc_config: Dict[str, Any] = {
            "inference_delay": int(inference_delay),
            "prefix_attention_horizon": int(guidance_horizon),
            "max_guidance_weight": self.rtc_max_guidance_weight,
            "rtc_use_vjp": self.rtc_use_vjp,
        }
        prefix_end = int(self._previous_absolute_action.shape[0])
        absolute_prefix = self._previous_absolute_action[prefix_start:prefix_end]
        relative_prefix = self._convert_absolute_action_to_relative(
            absolute_prefix,
            observation,
        )
        prefix_action = self._empty_relative_prefix(
            action_horizon=model_horizon,
            action_dim=model_action_dim,
            action_quaternion_order=self.action_quaternion_order,
        )
        if relative_prefix.shape[1] != model_action_dim:
            raise ValueError(
                "FastWAM RTC relative prefix dimension mismatch: expected "
                f"{model_action_dim}, got {relative_prefix.shape[1]}"
            )
        copy_steps = min(relative_prefix.shape[0], model_horizon)
        prefix_action[:copy_steps] = relative_prefix[:copy_steps]

        rtc_config["action_index"] = int(action_index)
        self._last_rtc_config = rtc_config
        self._last_inference_diagnostics.update(
            {
                "rtc_applied": True,
                "inference_delay_steps": inference_delay,
                "guidance_horizon_steps": guidance_horizon,
                "weighted_guidance_steps": guidance_horizon,
                "max_guidance_weight": 10.0,
                "guide_source_start": prefix_start,
                "guide_source_end_exclusive": prefix_end,
                "guide_model_start": 0,
                "guide_model_end_exclusive": copy_steps,
                "weighted_command_start_offset": 0,
                "weighted_command_end_exclusive": guidance_horizon,
                "weighted_model_start": 0,
                "weighted_model_end_exclusive": copy_steps,
            }
        )
        logger.info(
            "FastWAM RTC online config: "
            f"action_index={action_index}, "
            f"delay={inference_delay} ({delay_source}, estimated={estimated_inference_delay}, "
            f"margin={self.rtc_delay_margin_steps}, override={self.rtc_guidance_full_steps_override}), "
            f"prefix_horizon={guidance_horizon}, "
            f"guidance={float(rtc_config['max_guidance_weight']):.3f}, "
            f"vjp={self.rtc_use_vjp}, "
            f"latency_p95={self._latency_tracker.p95() * 1000.0:.1f}ms, "
            f"samples={len(self._latency_tracker)}"
        )
        return {
            "prefix_action": prefix_action,
            "rtc_config": rtc_config,
        }

    def _extract_server_action(self, response: Dict[str, Any]) -> np.ndarray:
        data = response.get("data", response)
        if not isinstance(data, dict) or "action" not in data:
            raise KeyError(f"FastWAM response missing action, keys={list(response.keys())}")
        action = np.asarray(data["action"], dtype=np.float64)
        if action.ndim != 2:
            raise ValueError(f"Expected FastWAM action [T,D], got {action.shape}")
        if "action_horizon" in self._server_metadata and "action_dim" in self._server_metadata:
            expected_shape = (
                int(self._server_metadata["action_horizon"]),
                int(self._server_metadata["action_dim"]),
            )
            if action.shape != expected_shape:
                raise ValueError(
                    "FastWAM response action shape mismatch: expected "
                    f"{expected_shape}, got {action.shape}"
                )
        if not np.all(np.isfinite(action)):
            raise ValueError("FastWAM response action contains NaN or infinity")
        return action

    @staticmethod
    def _validate_rigid_transform(transform: Any, name: str) -> np.ndarray:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"{name} transform must be 4x4, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} transform contains NaN or infinity")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"{name} transform has an invalid homogeneous last row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"{name} transform rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError(f"{name} transform rotation determinant must be +1")
        return matrix

    @staticmethod
    def _pose_matrix(position: np.ndarray, rotation: Rotation) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        return matrix

    @staticmethod
    def _poses_to_matrices(
        positions: np.ndarray,
        quaternions_xyzw: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Pose positions must be [T,3], got {positions.shape}")
        if quaternions_xyzw.shape != (positions.shape[0], 4):
            raise ValueError(
                "Pose quaternions must be [T,4] and match positions, got "
                f"{quaternions_xyzw.shape} vs {positions.shape}"
            )
        matrices = np.repeat(
            np.eye(4, dtype=np.float64)[None, :, :],
            positions.shape[0],
            axis=0,
        )
        matrices[:, :3, :3] = Rotation.from_quat(quaternions_xyzw).as_matrix()
        matrices[:, :3, 3] = positions
        return matrices

    @staticmethod
    def _safe_quat_xyzw(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        if quat.shape[-1:] != (4,):
            raise ValueError(f"Quaternion must have last dimension 4, got {quat.shape}")
        if not np.all(np.isfinite(quat)):
            raise ValueError("Quaternion contains NaN or infinity")
        if quat.ndim == 1:
            norm = float(np.linalg.norm(quat))
            if norm < 1e-12:
                raise ValueError("Quaternion norm is zero")
            if quat[3] < 0.0:
                quat = -quat
            return quat / norm

        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        if np.any(norm < 1e-12):
            raise ValueError("Quaternion norm is zero")
        out = quat / norm
        out = np.where(out[..., 3:4] < 0.0, -out, out)
        return out

    def _action_quat_from_xyzw(self, quat: np.ndarray) -> np.ndarray:
        quat = self._safe_quat_xyzw(quat)
        order = str(self.action_quaternion_order).lower()
        if order == "xyzw":
            return quat
        if order == "wxyz":
            return np.concatenate([quat[..., 3:4], quat[..., :3]], axis=-1)
        raise ValueError(f"Unsupported FastWAM action_quaternion_order: {order!r}")

    def _extract_arm_anchor(
        self,
        robot_state: np.ndarray,
        arm_index: int,
    ) -> Tuple[np.ndarray, Rotation, float]:
        if arm_index == 0:
            tcp_pose_w_gripper = robot_state[14:22]
        else:
            tcp_pose_w_gripper = robot_state[36:44]
        pos = np.asarray(tcp_pose_w_gripper[:3], dtype=np.float64)
        quat_xyzw = np.asarray(
            process_quaternion(tcp_pose_w_gripper[3:7], "f2l"),
            dtype=np.float64,
        )
        quat_xyzw = self._safe_quat_xyzw(quat_xyzw)
        gripper = float(tcp_pose_w_gripper[7])
        return pos, Rotation.from_quat(quat_xyzw), gripper

    def _relative_quat_to_xyzw(self, action_chunk: np.ndarray, quat_slice: slice) -> np.ndarray:
        quat = np.asarray(action_chunk[:, quat_slice], dtype=np.float64)
        order = str(self.action_quaternion_order).lower()
        if order == "xyzw":
            return self._safe_quat_xyzw(quat)
        if order == "wxyz":
            return self._safe_quat_xyzw(
                np.asarray([process_quaternion(q, "f2l") for q in quat], dtype=np.float64)
            )
        raise ValueError(f"Unsupported FastWAM action_quaternion_order: {order!r}")

    def _convert_arm_relative_to_absolute(
        self,
        action_chunk: np.ndarray,
        robot_state: np.ndarray,
        arm_index: int,
    ) -> np.ndarray:
        base = arm_index * 8
        anchor_pos, anchor_rot, _anchor_gripper = self._extract_arm_anchor(
            robot_state,
            arm_index=arm_index,
        )
        rel_pos = np.asarray(action_chunk[:, base : base + 3], dtype=np.float64)
        rel_quat = self._relative_quat_to_xyzw(
            action_chunk,
            slice(base + 3, base + 7),
        )
        rel_gripper = np.asarray(action_chunk[:, base + 7 : base + 8], dtype=np.float64)

        anchor_tcp = self._pose_matrix(anchor_pos, anchor_rot)
        anchor_umi = anchor_tcp @ self.tcp_umi_T
        relative_rot = Rotation.from_quat(rel_quat).as_matrix()
        relative_frame = str(self.relative_pose_frame).lower()
        if relative_frame == "local_frame":
            relative_umi = self._poses_to_matrices(rel_pos, rel_quat)
            target_umi = anchor_umi @ relative_umi
        elif relative_frame == "world_frame":
            target_umi = np.repeat(
                np.eye(4, dtype=np.float64)[None, :, :],
                rel_pos.shape[0],
                axis=0,
            )
            target_umi[:, :3, :3] = anchor_umi[:3, :3] @ relative_rot
            target_umi[:, :3, 3] = anchor_umi[None, :3, 3] + rel_pos
        else:
            raise ValueError(f"Invalid FastWAM relative_pose_frame: {self.relative_pose_frame!r}")

        target_tcp = target_umi @ self.umi_tcp_T
        abs_pos = target_tcp[:, :3, 3]
        abs_quat = self._safe_quat_xyzw(
            Rotation.from_matrix(target_tcp[:, :3, :3]).as_quat()
        )
        return np.concatenate([abs_pos, abs_quat, rel_gripper], axis=-1)

    def _postprocess_grippers(self, action_chunk: np.ndarray) -> np.ndarray:
        if self.gripper_max_width <= 0:
            raise ValueError(f"gripper_max_width must be positive, got {self.gripper_max_width}")

        gripper_indices = [7]
        if action_chunk.shape[1] > 15:
            gripper_indices.append(15)
        for idx in gripper_indices:
            if self.clip_gripper_width or self.normalize_gripper_width:
                action_chunk[:, idx] = np.clip(action_chunk[:, idx], 0.0, self.gripper_max_width)
            if self.normalize_gripper_width:
                action_chunk[:, idx] = action_chunk[:, idx] / self.gripper_max_width
        return action_chunk

    def _convert_relative_action_to_absolute(
        self,
        action_chunk: np.ndarray,
        observation: Dict[str, Any],
        target_action_dim: int,
    ) -> np.ndarray:
        robot_state = self._extract_robot_state(observation)
        if robot_state.shape[0] < 44:
            raise ValueError(
                "FastWAM dual-arm action requires RizonRobot state dim >= 44, "
                f"got {robot_state.shape[0]}"
            )

        model_action_dim = int(action_chunk.shape[1])
        if target_action_dim not in {8, 16} or model_action_dim != target_action_dim:
            raise ValueError(
                "FastWAM action dimension must match the configured single- or "
                "dual-arm contract; "
                f"got model={model_action_dim} target={target_action_dim}"
            )

        left_action = self._convert_arm_relative_to_absolute(
            action_chunk,
            robot_state,
            arm_index=0,
        )
        if target_action_dim <= 8:
            return self._postprocess_grippers(left_action).astype(np.float64)

        right_action = self._convert_arm_relative_to_absolute(
            action_chunk,
            robot_state,
            arm_index=1,
        )
        return self._postprocess_grippers(
            np.concatenate([left_action, right_action], axis=-1)
        ).astype(np.float64)

    def _fit_action_dim(self, action_chunk: np.ndarray, target_dim: int) -> np.ndarray:
        if target_dim <= 0 or action_chunk.shape[1] == target_dim:
            return action_chunk
        if action_chunk.shape[1] > target_dim:
            logger.warning(
                f"FastWAMPolicy truncates action dim from {action_chunk.shape[1]} to {target_dim}"
            )
            return action_chunk[:, :target_dim]
        padding = np.zeros(
            (action_chunk.shape[0], target_dim - action_chunk.shape[1]),
            dtype=action_chunk.dtype,
        )
        return np.concatenate([action_chunk, padding], axis=1)

    def _fit_chunk_length(self, action_chunk: np.ndarray, chunk_length: int) -> np.ndarray:
        if chunk_length <= 0 or action_chunk.shape[0] == chunk_length:
            return action_chunk
        if action_chunk.shape[0] > chunk_length:
            return action_chunk[:chunk_length]
        tail = np.repeat(action_chunk[-1:, :], chunk_length - action_chunk.shape[0], axis=0)
        return np.concatenate([action_chunk, tail], axis=0)

    @staticmethod
    def _compute_rtc_overlap_diagnostics(
        previous_action: Optional[np.ndarray],
        new_action: np.ndarray,
        action_index: Optional[int],
        guidance_horizon: Optional[int],
    ) -> Dict[str, Any]:
        """Summarize how a new prediction agrees with the previous chunk.

        The executor replaces a chunk at the current original action index.  The
        old trajectory therefore starts at ``action_index + 1`` while the new
        trajectory starts at index zero.  These compact metrics make the RTC
        overlap and the hard guidance-mask edge inspectable without logging a
        second copy of every action vector.
        """
        diagnostics: Dict[str, Any] = {"rtc_overlap_available": False}
        if previous_action is None or action_index is None:
            return diagnostics
        previous = np.asarray(previous_action, dtype=np.float64)
        current = np.asarray(new_action, dtype=np.float64)
        if previous.ndim != 2 or current.ndim != 2:
            return diagnostics
        old_start = int(action_index) + 1
        overlap_length = min(previous.shape[0] - old_start, current.shape[0])
        if overlap_length <= 0:
            diagnostics["rtc_overlap_old_start_index"] = old_start
            diagnostics["rtc_overlap_length"] = 0
            return diagnostics

        old_overlap = previous[old_start : old_start + overlap_length]
        new_overlap = current[:overlap_length]
        diagnostics.update(
            {
                "rtc_overlap_available": True,
                "rtc_overlap_old_start_index": old_start,
                "rtc_overlap_length": int(overlap_length),
                "rtc_overlap_guided_length": int(
                    min(max(int(guidance_horizon or 0), 0), overlap_length)
                ),
            }
        )

        # FastWAM's dual-arm EEF layout is [xyz, quat, gripper] x 2.  Keep the
        # calculation explicit so quaternion discontinuities do not dominate the
        # position/trend diagnostics used to explain shaking.
        arm_slices = (("left", slice(0, 3)), ("right", slice(8, 11)))
        for arm_name, xyz_slice in arm_slices:
            if (
                old_overlap.shape[1] < xyz_slice.stop
                or new_overlap.shape[1] < xyz_slice.stop
            ):
                continue
            old_xyz = old_overlap[:, xyz_slice]
            new_xyz = new_overlap[:, xyz_slice]
            error_mm = np.linalg.norm(new_xyz - old_xyz, axis=1) * 1000.0
            guided_length = int(diagnostics["rtc_overlap_guided_length"])
            guided_error = error_mm[:guided_length]
            unguided_error = error_mm[guided_length:]
            arm_fields: Dict[str, Any] = {
                f"rtc_overlap_{arm_name}_xyz_mean_error_mm": float(np.mean(error_mm)),
                f"rtc_overlap_{arm_name}_xyz_p95_error_mm": float(
                    np.percentile(error_mm, 95)
                ),
                f"rtc_overlap_{arm_name}_xyz_endpoint_error_mm": float(error_mm[-1]),
                f"rtc_overlap_{arm_name}_xyz_guided_mean_error_mm": (
                    float(np.mean(guided_error)) if guided_error.size else None
                ),
                f"rtc_overlap_{arm_name}_xyz_unguided_mean_error_mm": (
                    float(np.mean(unguided_error)) if unguided_error.size else None
                ),
            }
            if guided_length < overlap_length:
                arm_fields[f"rtc_overlap_{arm_name}_mask_edge_jump_mm"] = float(
                    np.linalg.norm(
                        new_xyz[guided_length] - new_xyz[guided_length - 1]
                    )
                    * 1000.0
                )

            old_velocity = np.diff(old_xyz, axis=0)
            new_velocity = np.diff(new_xyz, axis=0)
            if old_velocity.shape[0] > 0:
                old_norm = np.linalg.norm(old_velocity, axis=1)
                new_norm = np.linalg.norm(new_velocity, axis=1)
                valid = (old_norm > 1e-9) & (new_norm > 1e-9)
                if np.any(valid):
                    cosine = np.sum(old_velocity[valid] * new_velocity[valid], axis=1) / (
                        old_norm[valid] * new_norm[valid]
                    )
                    arm_fields.update(
                        {
                            f"rtc_overlap_{arm_name}_velocity_cosine_mean": float(
                                np.mean(cosine)
                            ),
                            f"rtc_overlap_{arm_name}_velocity_negative_fraction": float(
                                np.mean(cosine < 0.0)
                            ),
                            f"rtc_overlap_{arm_name}_old_speed_mean_mm": float(
                                np.mean(old_norm[valid]) * 1000.0
                            ),
                            f"rtc_overlap_{arm_name}_new_speed_mean_mm": float(
                                np.mean(new_norm[valid]) * 1000.0
                            ),
                        }
                    )
            diagnostics.update(arm_fields)
        return diagnostics

    def _log_predicted_gripper_action(self, action_chunk: np.ndarray) -> None:
        if not self.print_predicted_gripper_action:
            return
        gripper_indices = [7]
        if action_chunk.shape[1] > 15:
            gripper_indices.append(15)
        logger.info(
            "FastWAMPolicy predicted gripper action "
            f"indices={gripper_indices}, shape={action_chunk[:, gripper_indices].shape}:\n"
            f"{np.array2string(action_chunk[:, gripper_indices], precision=6)}"
        )

    def _split_action_chunk_by_config(
        self,
        action_chunk: np.ndarray,
        action_configs: List[Dict[str, Any]],
        chunk_length: int,
    ) -> Dict[str, np.ndarray]:
        if not action_configs:
            raise ValueError("action_configs is empty")
        total_target_dim = sum(int(cfg.get("action_dim", 0)) for cfg in action_configs)
        if total_target_dim <= 0:
            raise ValueError(f"Invalid action_configs action dims: {action_configs}")

        fitted_chunk = self._fit_chunk_length(action_chunk, chunk_length=chunk_length)
        fitted_chunk = self._fit_action_dim(fitted_chunk, target_dim=total_target_dim)
        self._log_predicted_gripper_action(fitted_chunk)

        action_chunk_dict: Dict[str, np.ndarray] = {}
        start_dim = 0
        for cfg in action_configs:
            device_name = str(cfg["device_name"])
            action_dim = int(cfg["action_dim"])
            end_dim = start_dim + action_dim
            action_chunk_dict[device_name] = fitted_chunk[:, start_dim:end_dim].astype(
                np.float64
            )
            start_dim = end_dim
        return action_chunk_dict

    def predict(
        self,
        observation: Dict[str, Any],
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        action_index: Optional[int] = None,
        _track_latency: bool = True,
    ) -> Dict[str, np.ndarray]:
        if len(action_configs) == 0:
            raise ValueError("action_configs is empty")

        payload_context = (
            self.profiler.section("client/build_payload")
            if self.profiler is not None
            else nullcontext()
        )
        with payload_context:
            payload = self._build_infer_payload(observation)
        request_video_prediction = self._should_request_video_prediction()
        if request_video_prediction:
            payload["return_video_prediction"] = True
            payload["video_prediction_max_frames"] = int(
                self.video_visualization_config.get("max_prediction_frames", 16)
            )
            payload["video_prediction_jpeg_quality"] = int(
                self.video_visualization_config.get("jpeg_quality", 85)
            )
        rtc_context = (
            self.profiler.section("client/build_rtc_payload")
            if self.profiler is not None
            else nullcontext()
        )
        with rtc_context:
            if request_video_prediction and self.video_visualization_disable_rtc:
                # The server returns the action only after the extra joint video
                # pass. A normal RTC prefix would underestimate this response
                # delay, so use an unguided action for this visualization sample.
                self._last_rtc_config = None
                self._last_inference_diagnostics = {
                    "rtc_enabled": bool(self.enable_rtc),
                    "rtc_applied": False,
                    "rtc_fallback_reason": "video_prediction_rtc_disabled",
                    "action_index": action_index,
                }
            else:
                payload.update(self._build_rtc_payload(observation, action_index))
        inference_start = time.monotonic()
        rpc_context = (
            self.profiler.section("client/websocket_rpc")
            if self.profiler is not None
            else nullcontext()
        )
        with rpc_context:
            response = self._send_request(payload)
        inference_latency = time.monotonic() - inference_start
        # The optional response contains a second diffusion/VAE pass and JPEG
        # payload. Keep it out of RTC's control-path latency estimator so the
        # visualization path cannot inflate prefix guidance on later requests.
        if self.enable_rtc and _track_latency and not request_video_prediction:
            self._latency_tracker.add(inference_latency)
        server_timing = response.get("server_timing", {})
        if not isinstance(server_timing, dict):
            server_timing = {}
        self._last_inference_diagnostics.update(
            {
                "rpc_ms": inference_latency * 1000.0,
                "latency_p95_ms": self._latency_tracker.p95() * 1000.0,
                "latency_sample_count": len(self._latency_tracker),
                "server_infer_ms": server_timing.get("infer_ms"),
                "server_total_ms": server_timing.get("total_ms"),
                "video_prediction_requested": request_video_prediction,
            }
        )
        postprocess_context = (
            self.profiler.section("client/postprocess_action")
            if self.profiler is not None
            else nullcontext()
        )
        previous_absolute_action = self._previous_absolute_action
        with postprocess_context:
            relative_action = self._extract_server_action(response)
            total_target_dim = sum(int(cfg.get("action_dim", 0)) for cfg in action_configs)
            target_dim = total_target_dim if total_target_dim > 0 else relative_action.shape[1]
            absolute_action = self._convert_relative_action_to_absolute(
                action_chunk=relative_action,
                observation=observation,
                target_action_dim=target_dim,
            )
            self._last_inference_diagnostics.update(
                self._compute_rtc_overlap_diagnostics(
                    previous_action=previous_absolute_action,
                    new_action=absolute_action,
                    action_index=action_index,
                    guidance_horizon=self._last_inference_diagnostics.get(
                        "guidance_horizon_steps"
                    ),
                )
            )
            self._previous_absolute_action = absolute_action.copy()
            action_chunk_dict = self._split_action_chunk_by_config(
                action_chunk=absolute_action,
                action_configs=action_configs,
                chunk_length=chunk_length,
            )
        self._update_video_visualizer(
            observation,
            response,
            requested=request_video_prediction,
        )
        return action_chunk_dict

    def get_last_inference_diagnostics(self) -> Dict[str, Any]:
        """Return JSON-safe timing and RTC alignment details for trace logging."""
        return dict(self._last_inference_diagnostics)

    def warm_up(
        self,
        action_configs: List[Dict[str, Any]],
        num_observations: int = 10,
        sample_observation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Warm up normal and RTC requests without recording cold-start latency."""
        if not self.enable_rtc or sample_observation is None:
            return

        previous_suppressed = self._video_visualization_suppressed
        self._video_visualization_suppressed = True
        try:
            self._warm_up_impl(
                action_configs=action_configs,
                num_observations=num_observations,
                sample_observation=sample_observation,
            )
        finally:
            self._video_visualization_suppressed = previous_suppressed

    def _warm_up_impl(
        self,
        action_configs: List[Dict[str, Any]],
        num_observations: int,
        sample_observation: Dict[str, Any],
    ) -> None:
        """Run the RTC warm-up loop with visualization requests suppressed."""
        self._latency_tracker.reset()
        normal_warmup_count = min(5, int(num_observations))
        logger.info("FastWAM: warming up normal inference")
        for index in range(normal_warmup_count):
            start = time.monotonic()
            self.predict(
                sample_observation,
                action_configs,
                chunk_length=self.chunk_length,
                action_index=None,
                _track_latency=False,
            )
            elapsed = time.monotonic() - start
            if index > 0:
                self._latency_tracker.add(elapsed)
            logger.info(
                "FastWAM normal warm-up %d/%d: %.2f ms%s",
                index + 1,
                normal_warmup_count,
                elapsed * 1000.0,
                " [excluded]" if index == 0 else "",
            )

        rtc_warmup_count = max(0, min(5, int(num_observations) - normal_warmup_count))
        logger.info("FastWAM: warming up RTC inference")
        for index in range(rtc_warmup_count):
            start = time.monotonic()
            self.predict(
                sample_observation,
                action_configs,
                chunk_length=self.chunk_length,
                action_index=0,
                _track_latency=False,
            )
            elapsed = time.monotonic() - start
            if index > 0:
                self._latency_tracker.add(elapsed)
            logger.info(
                "FastWAM RTC warm-up %d/%d: %.2f ms%s",
                index + 1,
                rtc_warmup_count,
                elapsed * 1000.0,
                " [excluded]" if index == 0 else "",
            )

        logger.info(
            "FastWAM warm-up latency: count=%d mean=%.2fms p95=%.2fms max=%.2fms",
            len(self._latency_tracker),
            self._latency_tracker.mean() * 1000.0,
            self._latency_tracker.p95() * 1000.0,
            self._latency_tracker.max() * 1000.0,
        )
        self.reset()
