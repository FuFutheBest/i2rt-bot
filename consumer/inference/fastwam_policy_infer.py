#!/usr/bin/env python3
"""FastWAM policy connector entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from consumer.inference.common.policy_connector import PolicyConnector  # noqa: E402
from utils.superoffline_observation import (  # noqa: E402
    capture_device_descriptors_from_config,
)

DEFAULT_CONFIG_NAME = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
_FASTWAM_POLICY_CLASS = "FastWAMPolicy"


class FastWAMPolicyInfer(PolicyConnector):
    """PolicyConnector wrapper for FastWAMPolicy."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def _select_plain(cfg: DictConfig, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_config(config_path: Optional[str], config_name: Optional[str]) -> Dict[str, Any]:
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
        with initialize_config_dir(version_base="1.3", config_dir=str(_REPO_ROOT / "configs")):
            cfg = compose(config_name=config_name or DEFAULT_CONFIG_NAME)
    return {
        "summary_shm": _select_plain(cfg, "summary_shm", "device_summary_data"),
        "runtime": _select_plain(cfg, "runtime", {}),
        "policy": _select_plain(cfg, "policy", {}),
        "profiling": _select_plain(cfg, "profiling", {}),
        "devices": _select_plain(cfg, "devices.devices", []),
    }


def _build_connector_kwargs(
    root_cfg: Dict[str, Any],
    policy_cfg: Dict[str, Any],
    *,
    policy_class: Optional[str] = None,
    policy_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    obs_cfg = policy_cfg.get("obs", {})
    selected_params = (
        dict(policy_params)
        if policy_params is not None
        else dict(policy_cfg.get("params", {}) or {})
    )
    recorder_config = dict(policy_cfg.get("observation_recorder", {}) or {})
    return {
        "summary_shm_name": root_cfg.get("summary_shm", "device_summary_data"),
        "policy_class": policy_class or policy_cfg.get("class", "FastWAMPolicy"),
        "policy_params": selected_params,
        "obs_devices": obs_cfg.get("devices"),
        "controls": policy_cfg.get("controls", []),
        "reset_cfg": policy_cfg.get("reset", {}),
        "halt_and_home_cfg": policy_cfg.get("halt_and_home", {}),
        "fps": policy_cfg.get("fps", 2.0),
        "control_fps": policy_cfg.get("control_fps", 30.0),
        "value_prediction_query_fps": policy_cfg.get(
            "value_prediction_query_fps", policy_cfg.get("fps", 2.0)
        ),
        "history_with_value": policy_cfg.get("history_with_value", False),
        "enable_value_prediction": policy_cfg.get("enable_value_prediction", False),
        "chunk_length": policy_cfg.get("chunk_length", 32),
        "master_device": policy_cfg.get("master_device"),
        "infer_rollout_config": policy_cfg.get("infer_rollout"),
        "policy_history_video_config": policy_cfg.get("history_video"),
        "runtime_config": root_cfg.get("runtime"),
        "profiling_config": root_cfg.get("profiling"),
        "observation_recorder_config": recorder_config,
    }


def _bind_observation_recorder_capture_identity(
    root_cfg: Dict[str, Any],
    recorder_config: Dict[str, Any],
    policy_params: Dict[str, Any],
    *,
    mock_mode: bool,
) -> Dict[str, Any]:
    """Attach the composed device-configuration identity to an enabled recorder."""
    if mock_mode:
        raise ValueError(
            "SuperOffline recording requires a composed real-robot policy, not mock mode"
        )
    source_profile = recorder_config.get("superoffline_reviewed_source_profile")
    if not source_profile:
        raise ValueError("observation_recorder.superoffline_reviewed_source_profile is required")
    result = dict(recorder_config)
    result["capture_device_descriptors"] = capture_device_descriptors_from_config(
        str(source_profile),
        devices=root_cfg.get("devices", []),
        camera_devices=policy_params.get("camera_devices", []),
        robot_device=str(policy_params.get("robot_device", "")),
    )
    return result


def _panel_module(module_name: str) -> Any:
    """Load shared option helpers without coupling the core runner to the UI."""
    from utils import inference_options

    return inference_options if module_name == "inference_options" else None


def _apply_control_panel_policy_overrides(config: Dict[str, Any]) -> None:
    """Apply the panel's narrowly scoped rollout override, if present."""
    options = _panel_module("inference_options")
    raw_payload = os.environ.get(options.POLICY_OVERRIDES_ENV)
    if not raw_payload:
        return
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {options.POLICY_OVERRIDES_ENV} JSON") from exc
    if not isinstance(payload, dict) or set(payload) - {"infer_rollout"}:
        raise ValueError(
            f"{options.POLICY_OVERRIDES_ENV} may only override policy.infer_rollout"
        )
    rollout = payload.get("infer_rollout")
    if not isinstance(rollout, dict):
        raise ValueError("Control-panel infer_rollout override must be an object")
    policy = config.setdefault("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("policy config must be a mapping")
    policy["infer_rollout"] = rollout


def _mock_policy_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build safe no-motion parameters used by panel hardware-free smoke tests."""
    options = _panel_module("inference_options")
    policy = config.get("policy")
    obs = policy.get("obs") if isinstance(policy, dict) else None
    devices = obs.get("devices") if isinstance(obs, dict) else None
    robot_device = next(
        (
            str(device)
            for device in (devices or [])
            if str(device).casefold().startswith(("rizon", "robot"))
        ),
        "RizonRobot_2",
    )
    gripper_min, gripper_max = options.mock_gripper_range(config)
    return {
        "robot_device": robot_device,
        "gripper_min": gripper_min,
        "gripper_max": gripper_max,
        "gripper_travel_seconds": 2.5,
    }


def _apply_server_endpoint_overrides(
    policy_params: Dict[str, Any],
    *,
    server_host: Optional[str],
    server_port: Optional[int],
    legacy_overrides: Sequence[str] = (),
) -> None:
    """Apply panel-selected server endpoint without permitting arbitrary overrides."""
    for override in legacy_overrides:
        key, separator, value = override.partition("=")
        if not separator or key not in {"policy.params.host", "policy.params.port"}:
            raise ValueError(f"Unsupported FastWAM policy override: {override}")
        if key.endswith(".host"):
            server_host = value
        else:
            try:
                server_port = int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid FastWAM server port: {value}") from exc
    if server_host is not None:
        policy_params["host"] = server_host
    if server_port is not None:
        if not 1 <= int(server_port) <= 65535:
            raise ValueError(f"Invalid FastWAM server port: {server_port}")
        policy_params["port"] = int(server_port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastWAM Policy Connector - run client and write action chunks to policy SHM"
    )
    parser.add_argument("--config", type=str, default=None, help="SuperInference YAML config path")
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name under configs/",
    )
    parser.add_argument("--server-host", type=str, default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--record-observations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--observation-output-dir", type=str, default=None)
    parser.add_argument("legacy_policy_overrides", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()

    merged_cfg = _load_config(args.config, args.config_name)
    _apply_control_panel_policy_overrides(merged_cfg)
    policy_cfg = cast(Dict[str, Any], merged_cfg.get("policy", {}))
    options = _panel_module("inference_options")
    mock_mode = options.as_bool(os.environ.get(options.MOCK_INFERENCE_ENV, False))
    if mock_mode:
        mock_error = options.mock_inference_config_error(merged_cfg)
        if mock_error:
            raise ValueError(mock_error)
        from policy import POLICY_CLASSES
        from consumer.inference.common.mock_inference import ControlPanelNoMotionPolicy

        POLICY_CLASSES[options.MOCK_POLICY_CLASS] = ControlPanelNoMotionPolicy
    policy_class = str(policy_cfg.get("class") or _FASTWAM_POLICY_CLASS)
    raw_params = policy_cfg.get("params")
    policy_params = dict(raw_params) if isinstance(raw_params, dict) else {}
    if mock_mode:
        policy_class = options.MOCK_POLICY_CLASS
        policy_params = _mock_policy_parameters(merged_cfg)
    else:
        _apply_server_endpoint_overrides(
            policy_params,
            server_host=args.server_host,
            server_port=args.server_port,
            legacy_overrides=args.legacy_policy_overrides,
        )
    policy_cfg_for_connector = dict(policy_cfg)
    policy_cfg_for_connector["params"] = policy_params
    connector_kwargs = _build_connector_kwargs(
        root_cfg=merged_cfg,
        policy_cfg=policy_cfg_for_connector,
    )
    connector_kwargs["policy_class"] = policy_class
    connector_kwargs["policy_params"] = policy_params
    recorder_config = dict(connector_kwargs.get("observation_recorder_config") or {})
    if args.record_observations is not None:
        recorder_config["enabled"] = bool(args.record_observations)
    if args.observation_output_dir is not None:
        recorder_config["output_dir"] = str(Path(args.observation_output_dir).expanduser())
    if options.as_bool(recorder_config.get("enabled", False)):
        recorder_config = _bind_observation_recorder_capture_identity(
            merged_cfg,
            recorder_config,
            policy_params,
            mock_mode=mock_mode,
        )
    connector_kwargs["observation_recorder_config"] = recorder_config

    config_file_path: Optional[str] = None
    if args.config:
        config_file_path = str(Path(args.config).expanduser().resolve())
    else:
        candidate = _REPO_ROOT / "configs" / f"{args.config_name}.yaml"
        if candidate.exists():
            config_file_path = str(candidate)
    connector_kwargs["config_file_path"] = config_file_path

    connector = FastWAMPolicyInfer(**connector_kwargs)
    connector.run()


if __name__ == "__main__":
    main()
