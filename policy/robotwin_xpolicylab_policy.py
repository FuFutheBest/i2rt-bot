"""RoboTwin policy adapter for XPolicyLab websocket policy servers."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from integrations.robotwin.contracts import validate_action
from policy.base import BasePolicy, ObservationType
from utils.logger_config import logger


_CAMERA_TO_XPOLICYLAB = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def _format_websocket_url(host: str, port: int) -> str:
    normalized_host = str(host).strip()
    if normalized_host.startswith("ws://") or normalized_host.startswith("wss://"):
        return normalized_host
    if normalized_host.startswith("[") and normalized_host.endswith("]"):
        endpoint_host = normalized_host
    elif ":" in normalized_host:
        endpoint_host = f"[{normalized_host}]"
    else:
        endpoint_host = normalized_host
    return f"ws://{endpoint_host}:{int(port)}"


def _normalize_server_action_type(action_type: str) -> str:
    normalized = str(action_type).strip().lower()
    if normalized in {"joint", "qpos", "joint14"}:
        return "joint14"
    if normalized in {"ee", "endpose", "eef16"}:
        return "eef16"
    raise ValueError(
        f"Unsupported RoboTwin XPolicyLab action_type {action_type!r}; use 'joint' or 'ee'"
    )


def _latest_array(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("RoboTwin XPolicyLab observation history is empty")
        return np.asarray(value[-1])
    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[-1] == 14 and array.shape[0] != 14:
        return np.asarray(array[-1])
    if array.ndim >= 4:
        return np.asarray(array[-1])
    return array


class RoboTwinXPolicyLabPolicy(BasePolicy):
    """Call an XPolicyLab RoboTwin policy server from SuperInference.

    This adapter intentionally does not reuse the repository's Rizon/OpenPI or
    Rizon/FastWAM policies.  XPolicyLab policies for RoboTwin consume RoboTwin's
    own websocket observation contract and can return either joint or end-effector
    action chunks.  The SuperInference side keeps the existing asynchronous
    PolicyConnector -> ActionExecutor -> RoboTwin control SHM pipeline.
    """

    def __init__(
        self,
        *,
        policy_name: str,
        xpolicylab_root: str = "third_party/RoboTwin/XPolicyLab",
        host: str = "127.0.0.1",
        port: int = 6000,
        policy_server_url: Optional[str] = None,
        robot_device: str = "RoboTwinSimRobot_0",
        camera_devices: Optional[Mapping[str, str]] = None,
        task_name: str = "beat_block_hammer",
        task_config: str = "demo_clean",
        env_cfg_type: str = "aloha_agilex",
        action_type: str = "joint",
        seed: int = 0,
        instruction: Optional[str] = None,
        frequency: float = 30.0,
        env_idx: int = 0,
        evaluation_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        action_case_id: Optional[str] = None,
        repeat_index: Optional[int] = None,
        prepare_case_on_load: bool = True,
        reset_on_load: bool = True,
        connect_timeout_s: Optional[float] = None,
        handshake_timeout_s: Optional[float] = None,
        request_timeout_s: Optional[float] = None,
        max_connect_attempts: Optional[int] = None,
        connect_retry_delay_s: Optional[float] = None,
        max_connect_seconds: Optional[float] = None,
        ws_ping_interval_s: Optional[float] = 20.0,
        ws_ping_timeout_s: Optional[float] = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy_name = str(policy_name)
        self.xpolicylab_root = Path(xpolicylab_root).expanduser().resolve()
        self.host = str(host)
        self.port = int(port)
        self.policy_server_url = str(policy_server_url) if policy_server_url else None
        self.robot_device = str(robot_device)
        self.camera_devices = dict(
            camera_devices
            or {
                "head_camera": "RoboTwinCamera_0",
                "left_camera": "RoboTwinCamera_1",
                "right_camera": "RoboTwinCamera_2",
            }
        )
        self.task_name = str(task_name)
        self.task_config = str(task_config)
        self.env_cfg_type = str(env_cfg_type)
        self.action_type = str(action_type)
        self.action_schema = _normalize_server_action_type(self.action_type)
        self.seed = int(seed)
        self.instruction = str(instruction) if instruction else self.task_name.replace("_", " ")
        self.frequency = float(frequency)
        self.env_idx = int(env_idx)
        self.evaluation_id = evaluation_id or f"superinference-robotwin-{self.policy_name}"
        self.trial_id = trial_id or f"{self.task_name}-{self.seed}"
        self.action_case_id = action_case_id or f"{self.task_name}-{self.policy_name}-case"
        self.repeat_index = repeat_index
        self.prepare_case_on_load = bool(prepare_case_on_load)
        self.reset_on_load = bool(reset_on_load)
        self._client_kwargs = {
            "connect_timeout_s": connect_timeout_s,
            "handshake_timeout_s": handshake_timeout_s,
            "request_timeout_s": request_timeout_s,
            "max_connect_attempts": max_connect_attempts,
            "connect_retry_delay_s": connect_retry_delay_s,
            "max_connect_seconds": max_connect_seconds,
            "ws_ping_interval_s": ws_ping_interval_s,
            "ws_ping_timeout_s": ws_ping_timeout_s,
        }
        self._client: Any = None
        self._prediction_context_timestamp_ns: Optional[int] = None
        self._last_prediction_timing_ns: Dict[str, int] = {}

    def load(self) -> None:
        if not self.xpolicylab_root.is_dir():
            raise FileNotFoundError(
                f"XPolicyLab checkout not found: {self.xpolicylab_root}. "
                "Run: git -C third_party/RoboTwin submodule update --init --recursive XPolicyLab"
            )
        policy_dir = self.xpolicylab_root / "policy" / self.policy_name
        if not policy_dir.is_dir():
            raise FileNotFoundError(f"XPolicyLab policy directory not found: {policy_dir}")

        self._add_xpolicylab_paths()
        from client_server.ws import WsModelClient

        self._client = WsModelClient(
            url=self.policy_server_url or _format_websocket_url(self.host, self.port),
            evaluation_id=self.evaluation_id,
            trial_id=self.trial_id,
            action_case_id=self.action_case_id,
            repeat_index=self.repeat_index,
            **{key: value for key, value in self._client_kwargs.items() if value is not None},
        )
        if self.prepare_case_on_load:
            self._prepare_case()
        if self.reset_on_load:
            self.reset()
        self.initialized = True

    def close(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.close()
        except Exception as exc:
            logger.debug(f"RoboTwin XPolicyLab websocket close ignored: {exc}")
        finally:
            self._client = None

    def __del__(self) -> None:
        self.close()

    def reset(self) -> None:
        if self._client is None:
            return
        self._client.call(func_name="reset")

    def set_prediction_context(self, timestamp_ns: int) -> None:
        self._prediction_context_timestamp_ns = int(timestamp_ns)

    def predict(
        self,
        observation: ObservationType,
        action_configs: List[Dict[str, Any]],
        chunk_length: int = 1,
        **_: Any,
    ) -> Dict[str, np.ndarray]:
        if not self.initialized or self._client is None:
            raise RuntimeError("RoboTwinXPolicyLabPolicy.load() must be called before predict()")
        if not isinstance(observation, dict):
            raise ValueError("RoboTwin XPolicyLab policy requires a dict observation")

        self._last_prediction_timing_ns = {}
        predict_started_ns = time.perf_counter_ns()
        control_device = self._resolve_control_device(action_configs)
        control_schema = self._control_schema(action_configs)
        xpl_observation = self._build_xpolicylab_observation(observation)
        input_preprocess_finished_ns = time.perf_counter_ns()

        self._client.call(func_name="update_obs", obs=xpl_observation)
        raw_actions = self._client.call(func_name="get_action")
        inference_finished_ns = time.perf_counter_ns()

        action_chunk = self._flatten_action_chunk(
            raw_actions,
            fallback_joint14=self._current_joint14(observation),
            expected_schema=control_schema,
            chunk_length=int(chunk_length),
        )
        postprocess_finished_ns = time.perf_counter_ns()
        self._last_prediction_timing_ns = {
            "input_preprocess_ns": input_preprocess_finished_ns - predict_started_ns,
            "policy_predict_ns": inference_finished_ns - input_preprocess_finished_ns,
            "postprocess_ns": postprocess_finished_ns - inference_finished_ns,
        }
        return {control_device: action_chunk}

    @property
    def last_prediction_timing_ns(self) -> Dict[str, int]:
        return dict(self._last_prediction_timing_ns)

    def _add_xpolicylab_paths(self) -> None:
        root = str(self.xpolicylab_root)
        parent = str(self.xpolicylab_root.parent)
        integrations = str(self.xpolicylab_root / "integrations")
        for path in (root, parent, integrations):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _prepare_case(self) -> None:
        assert self._client is not None
        try:
            self._client.call(
                func_name="prepare_case",
                obs={
                    "task_name": self.task_name,
                    "task_config": self.task_config,
                    "seed": self.seed,
                    "instruction": self.instruction,
                    "action_type": self.action_type,
                    "env_cfg_type": self.env_cfg_type,
                    "policy_name": self.policy_name,
                },
            )
        except NotImplementedError:
            return
        except Exception as exc:
            logger.debug(f"XPolicyLab prepare_case ignored for {self.policy_name}: {exc}")

    def _resolve_control_device(self, action_configs: List[Dict[str, Any]]) -> str:
        expected_dim = 14 if self.action_schema == "joint14" else 16
        matching = [config for config in action_configs if int(config.get("action_dim", 0)) == expected_dim]
        if len(matching) != 1:
            raise ValueError(
                f"RoboTwin XPolicyLab {self.action_type!r} policy expects exactly one "
                f"control target with dimension {expected_dim}, got {len(matching)}"
            )
        return str(matching[0]["device_name"])

    def _control_schema(self, action_configs: List[Dict[str, Any]]) -> str:
        device_name = self._resolve_control_device(action_configs)
        for config in action_configs:
            if str(config.get("device_name")) == device_name:
                dim = int(config.get("action_dim", 0))
                return "joint14" if dim == 14 else "eef16"
        raise RuntimeError("resolved control device disappeared from action_configs")

    def _build_xpolicylab_observation(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        vision = {}
        for robotwin_name, xpl_name in _CAMERA_TO_XPOLICYLAB.items():
            device_name = self.camera_devices.get(robotwin_name)
            if not device_name:
                raise ValueError(f"Missing camera device mapping for {robotwin_name!r}")
            if device_name not in observation:
                raise KeyError(f"RoboTwin observation missing camera device {device_name!r}")
            image = self._extract_camera(observation[device_name], device_name)
            vision[xpl_name] = {"color": image, "shape": tuple(image.shape[:2])}

        payload: Dict[str, Any] = {
            "data_format_version": "v1.0",
            "instruction": self.instruction,
            "instructions": [self.instruction] if self.instruction else [],
            "task_instruction": self.instruction,
            "env_idx": self.env_idx,
            "vision": vision,
            "state": self._joint14_to_xpl_state(self._current_joint14(observation)),
            "additional_info": {"frequency": int(round(self.frequency))},
        }
        if self._prediction_context_timestamp_ns is not None:
            payload["timestamp_ns"] = self._prediction_context_timestamp_ns
        return payload

    def _current_joint14(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self.robot_device not in observation:
            raise KeyError(f"RoboTwin observation missing robot device {self.robot_device!r}")
        return validate_action(_latest_array(observation[self.robot_device]), "joint14")

    @staticmethod
    def _joint14_to_xpl_state(joint14: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "left_arm_joint_state": np.asarray(joint14[:6], dtype=np.float32),
            "left_ee_joint_state": np.asarray([joint14[6]], dtype=np.float32),
            "right_arm_joint_state": np.asarray(joint14[7:13], dtype=np.float32),
            "right_ee_joint_state": np.asarray([joint14[13]], dtype=np.float32),
        }

    @staticmethod
    def _extract_camera(value: Any, device_name: str) -> np.ndarray:
        image = _latest_array(value)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"RoboTwin camera {device_name!r} must be HxWx3, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _flatten_action_chunk(
        self,
        actions: Any,
        *,
        fallback_joint14: np.ndarray,
        expected_schema: str,
        chunk_length: int,
    ) -> np.ndarray:
        rows = [
            self._flatten_one_action(
                action,
                fallback_joint14=fallback_joint14,
                expected_schema=expected_schema,
            )
            for action in self._normalize_action_chunk(actions)
        ]
        if not rows:
            raise RuntimeError(f"XPolicyLab {self.policy_name} returned an empty action chunk")
        chunk = np.stack(rows, axis=0).astype(np.float64, copy=False)
        if chunk.shape[0] >= chunk_length:
            return chunk[:chunk_length].copy()
        padding = np.repeat(chunk[-1][None, :], chunk_length - chunk.shape[0], axis=0)
        return np.concatenate([chunk, padding], axis=0).astype(np.float64, copy=False)

    def _normalize_action_chunk(self, actions: Any) -> List[Any]:
        if actions is None:
            return []
        if isinstance(actions, Mapping) and "actions" in actions:
            return self._normalize_action_chunk(actions["actions"])
        if isinstance(actions, Mapping):
            return [actions]
        if isinstance(actions, np.ndarray):
            if actions.ndim == 0:
                raise ValueError("Scalar XPolicyLab actions are not supported")
            if actions.ndim == 1:
                return [actions]
            return [row for row in actions]
        if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes, bytearray)):
            if not actions:
                return []
            array = np.asarray(actions)
            if array.dtype != object and array.ndim == 1:
                return [array]
            return list(actions)
        raise TypeError(f"Unsupported XPolicyLab action response type: {type(actions)!r}")

    def _flatten_one_action(
        self,
        action: Any,
        *,
        fallback_joint14: np.ndarray,
        expected_schema: str,
    ) -> np.ndarray:
        if not isinstance(action, Mapping):
            return validate_action(action, expected_schema)

        action_schema = _normalize_server_action_type(str(action.get("action_type", self.action_type)))
        if action_schema != expected_schema:
            raise ValueError(
                f"XPolicyLab {self.policy_name} returned {action_schema}, "
                f"but configured RoboTwin control expects {expected_schema}"
            )
        if action_schema == "joint14":
            return self._flatten_joint_action(action, fallback_joint14=fallback_joint14)
        return self._flatten_eef_action(action)

    def _flatten_joint_action(
        self, action: Mapping[str, Any], *, fallback_joint14: np.ndarray
    ) -> np.ndarray:
        left_arm = self._action_array(
            action,
            ("left_arm_joint_state", "left_arm_joint", "left_joint_state", "arm_joint_state", "joint_state"),
            fallback=fallback_joint14[:6],
            name="left arm joint action",
            expected_dim=6,
        )
        right_arm = self._action_array(
            action,
            ("right_arm_joint_state", "right_arm_joint", "right_joint_state"),
            fallback=fallback_joint14[7:13],
            name="right arm joint action",
            expected_dim=6,
        )
        left_gripper = self._action_scalar(
            action,
            ("left_ee_joint_state", "left_gripper", "left_gripper_pos", "ee_joint_state"),
            fallback=float(fallback_joint14[6]),
            name="left gripper action",
        )
        right_gripper = self._action_scalar(
            action,
            ("right_ee_joint_state", "right_gripper", "right_gripper_pos"),
            fallback=float(fallback_joint14[13]),
            name="right gripper action",
        )
        return validate_action(
            np.concatenate(
                [
                    left_arm,
                    np.asarray([left_gripper], dtype=np.float64),
                    right_arm,
                    np.asarray([right_gripper], dtype=np.float64),
                ]
            ),
            "joint14",
        )

    def _flatten_eef_action(self, action: Mapping[str, Any]) -> np.ndarray:
        left_pose = self._action_array(
            action,
            ("left_ee_pose", "left_endpose", "ee_pose"),
            fallback=None,
            name="left ee action",
            expected_dim=7,
        )
        right_pose = self._action_array(
            action,
            ("right_ee_pose", "right_endpose"),
            fallback=None,
            name="right ee action",
            expected_dim=7,
        )
        left_gripper = self._action_scalar(
            action,
            ("left_ee_joint_state", "left_gripper", "left_gripper_pos", "ee_joint_state"),
            fallback=None,
            name="left gripper action",
        )
        right_gripper = self._action_scalar(
            action,
            ("right_ee_joint_state", "right_gripper", "right_gripper_pos"),
            fallback=None,
            name="right gripper action",
        )
        return validate_action(
            np.concatenate(
                [
                    left_pose,
                    np.asarray([left_gripper], dtype=np.float64),
                    right_pose,
                    np.asarray([right_gripper], dtype=np.float64),
                ]
            ),
            "eef16",
        )

    @staticmethod
    def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @classmethod
    def _action_array(
        cls,
        action: Mapping[str, Any],
        keys: tuple[str, ...],
        *,
        fallback: Optional[np.ndarray],
        name: str,
        expected_dim: int,
    ) -> np.ndarray:
        value = cls._first_present(action, keys)
        if value is None:
            if fallback is None:
                raise KeyError(f"Missing {name}; tried keys: {keys}")
            array = np.asarray(fallback, dtype=np.float64).reshape(-1)
        else:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape[0] != expected_dim:
            raise ValueError(f"{name} must have dim {expected_dim}, got {array.shape[0]}")
        return array

    @classmethod
    def _action_scalar(
        cls,
        action: Mapping[str, Any],
        keys: tuple[str, ...],
        *,
        fallback: Optional[float],
        name: str,
    ) -> float:
        value = cls._first_present(action, keys)
        if value is None:
            if fallback is None:
                raise KeyError(f"Missing {name}; tried keys: {keys}")
            return float(fallback)
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape[0] != 1:
            raise ValueError(f"{name} must be scalar, got shape {array.shape}")
        return float(array[0])
