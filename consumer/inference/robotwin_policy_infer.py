"""PolicyConnector entry point for RoboTwin joint14 and eef16 presets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from consumer.inference.common.policy_connector import PolicyConnector  # noqa: E402

DEFAULT_CONFIG_NAME = "policy_infer/robotwin/config_infer_robotwin_replay_joint14"


class RoboTwinPolicyInfer(PolicyConnector):
    """Thin connector wrapper; policy adapters own RoboTwin-specific preprocessing."""


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
        config_root = (REPO_ROOT / "configs").resolve()
        if path.is_relative_to(config_root):
            with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
                cfg = compose(config_name=path.relative_to(config_root).with_suffix("").as_posix())
        else:
            with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
                cfg = compose(config_name=path.stem)
    else:
        with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
            cfg = compose(config_name=config_name or DEFAULT_CONFIG_NAME)
    return {
        "summary_shm": _select_plain(cfg, "summary_shm", "device_summary_data"),
        "runtime": _select_plain(cfg, "runtime", {}),
        "policy": _select_plain(cfg, "policy", {}),
    }


def _build_connector_kwargs(root_cfg: Dict[str, Any]) -> Dict[str, Any]:
    policy_cfg = dict(root_cfg.get("policy", {}) or {})
    obs_cfg = dict(policy_cfg.get("obs", {}) or {})
    return {
        "summary_shm_name": root_cfg.get("summary_shm", "device_summary_data"),
        "policy_class": policy_cfg.get("class", "RoboTwinReplayPolicy"),
        "policy_params": dict(policy_cfg.get("params", {}) or {}),
        "obs_devices": obs_cfg.get("devices"),
        "controls": policy_cfg.get("controls", []),
        "reset_cfg": policy_cfg.get("reset", {}),
        "halt_and_home_cfg": policy_cfg.get("halt_and_home", {}),
        "fps": policy_cfg.get("fps", 2.0),
        "control_fps": policy_cfg.get("control_fps", 30.0),
        "enable_value_prediction": policy_cfg.get("enable_value_prediction", False),
        "value_prediction_query_fps": policy_cfg.get("value_prediction_query_fps", policy_cfg.get("fps", 2.0)),
        "history_with_value": policy_cfg.get("history_with_value", False),
        "chunk_length": policy_cfg.get("chunk_length", 50),
        "master_device": policy_cfg.get("master_device"),
        "infer_rollout_config": policy_cfg.get("infer_rollout"),
        "runtime_config": root_cfg.get("runtime", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Explicit YAML config path")
    parser.add_argument("--config-name", type=str, default=DEFAULT_CONFIG_NAME, help="Hydra config name")
    parser.add_argument("--summary-shm", type=str, default=None, help="Override the device summary SHM name")
    args = parser.parse_args()

    root_cfg = _load_config(args.config, args.config_name)
    connector_kwargs = _build_connector_kwargs(root_cfg)
    if args.summary_shm:
        connector_kwargs["summary_shm_name"] = args.summary_shm
    if args.config:
        connector_kwargs["config_file_path"] = str(Path(args.config).expanduser().resolve())
    else:
        candidate = REPO_ROOT / "configs" / f"{args.config_name}.yaml"
        connector_kwargs["config_file_path"] = str(candidate) if candidate.is_file() else None
    RoboTwinPolicyInfer(**connector_kwargs).run()


if __name__ == "__main__":
    main()
