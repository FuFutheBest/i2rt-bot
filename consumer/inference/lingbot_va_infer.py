"""
LingBot-VA Infer - runs LingBot-VA policy and writes action chunking to policy SHM.

This module:
1. Reads observations from summary SHM (via PolicyConnector)
2. Calls local LingBot-VA policy to compute action chunking
3. Writes action chunks to policy SHM for ActionExecutor
"""

import argparse
import os
import sys
from collections import deque
from typing import Any, Dict, Optional, cast

import hydra
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consumer.inference.common.policy_connector import PolicyConnector
from utils.config_parser import load_config
from utils.logger_config import logger


class LingBotVAInfer(PolicyConnector):
    """Policy connector wrapper for LingBotVAPolicy."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.obs_queue = deque(maxlen=getattr(self.policy, "min_infer_obs", 1))

    # def _get_visualizer_app_id(self) -> str:
    #     return "lingbot_va_policy_visualizer"

    # def _parse_single_arm_action(
    #     self, action_chunk: np.ndarray
    # ) -> Dict[str, Dict[str, np.ndarray]]:
    #     if action_chunk.shape[1] not in (7, 8):
    #         logger.warning(
    #             f"Expected single-arm action dim 7 or 8, got {action_chunk.shape[1]}"
    #         )
    #         return {}

    #     parsed: Dict[str, Dict[str, np.ndarray]] = {
    #         "single_arm": {
    #             "position": action_chunk[:, 0:3],
    #             "quaternion": action_chunk[:, 3:7],
    #         }
    #     }
    #     if action_chunk.shape[1] == 8:
    #         parsed["single_arm"]["gripper"] = action_chunk[:, 7:8]
    #     return parsed

    # def _parse_dual_arm_action(
    #     self, action_chunk: np.ndarray
    # ) -> Dict[str, Dict[str, np.ndarray]]:
    #     if action_chunk.shape[1] != 16:
    #         logger.warning(
    #             f"Expected dual-arm action dim 16, got {action_chunk.shape[1]}"
    #         )
    #         return {}

    #     return {
    #         "left_arm": {
    #             "position": action_chunk[:, 0:3],
    #             "quaternion": action_chunk[:, 3:7],
    #             "gripper": action_chunk[:, 7:8],
    #         },
    #         "right_arm": {
    #             "position": action_chunk[:, 8:11],
    #             "quaternion": action_chunk[:, 11:15],
    #             "gripper": action_chunk[:, 15:16],
    #         },
    #     }

    # def _parse_action_chunk(
    #     self,
    #     device_name: str,
    #     action_chunk: np.ndarray,
    # ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    #     del device_name
    #     if action_chunk.shape[1] == 16:
    #         return self._parse_dual_arm_action(action_chunk)
    #     return self._parse_single_arm_action(action_chunk)


def load_policy_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def load_policy_config(config_path: str) -> Dict[str, Any]:
    """Load a policy config with Hydra defaults resolved."""
    abs_config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(abs_config_path)
    config_name = os.path.splitext(os.path.basename(abs_config_path))[0]
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return {
        "summary_shm": OmegaConf.select(cfg, "summary_shm", default="device_summary_data"),
        "runtime": _to_plain_config(OmegaConf.select(cfg, "runtime")),
        "policy": _to_plain_config(OmegaConf.select(cfg, "policy")),
    }


def _to_plain_config(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _flatten_config(cfg: DictConfig) -> Dict[str, Any]:
    return cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


def _build_connector_kwargs(
    root_cfg: Dict[str, Any], policy_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    obs_cfg = policy_cfg.get("obs", {})
    task_config_path = policy_cfg.get("task_config_path")
    task_config = load_config(task_config_path) if task_config_path else None

    return {
        "summary_shm_name": root_cfg.get("summary_shm", "device_summary_data"),
        "policy_class": policy_cfg.get("class", "LingBotVAPolicy"),
        "policy_params": policy_cfg.get("params", {}),
        "obs_devices": obs_cfg.get("devices"),
        "controls": policy_cfg.get("controls", []),
        "fps": policy_cfg.get("fps", 10.0),
        "control_fps": policy_cfg.get("control_fps", 10.0),
        "enable_value_prediction": policy_cfg.get("enable_value_prediction", False),
        "chunk_length": policy_cfg.get("chunk_length", 16),
        "master_device": policy_cfg.get("master_device"),
        "infer_rollout_config": policy_cfg.get("infer_rollout"),
        "policy_history_video_config": policy_cfg.get("history_video"),
        "runtime_config": root_cfg.get("runtime"),
    }


def _main_traditional(config_path: str) -> None:
    cfg_yaml = load_policy_config(config_path)
    policy_cfg = cast(Dict[str, Any], cfg_yaml.get("policy", {}))
    connector_kwargs = _build_connector_kwargs(root_cfg=cfg_yaml, policy_cfg=policy_cfg)
    connector_kwargs["config_file_path"] = os.path.abspath(config_path)
    connector = LingBotVAInfer(**connector_kwargs)
    connector.run()


@hydra.main(
    config_path="../../configs",
    config_name="policy_infer/lingbotva/config_infer_iphone_umi_single_lingbotva",
    version_base="1.3",
)
def _main_hydra(cfg: DictConfig) -> None:
    merged_cfg = _flatten_config(cfg=cfg)
    policy_cfg = cast(Dict[str, Any], merged_cfg.get("policy", {}))
    connector_kwargs = _build_connector_kwargs(root_cfg=merged_cfg, policy_cfg=policy_cfg)

    config_file_path: Optional[str] = None
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            config_name = str(HydraConfig.get().job.config_name)
            config_filename = config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
            config_candidate = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                config_filename,
            )
            if os.path.exists(config_candidate):
                config_file_path = config_candidate
    except Exception as exc:
        logger.warning(f"Could not resolve Hydra config file path: {exc}")

    connector_kwargs["config_file_path"] = config_file_path
    connector = LingBotVAInfer(**connector_kwargs)
    connector.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LingBot-VA Policy Connector - run policy and write action chunks to policy SHM"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="YAML path for policy connector config. If omitted, use Hydra config.",
    )
    args, _unknown = parser.parse_known_args()

    if args.config:
        _main_traditional(args.config)
    else:
        _main_hydra()
