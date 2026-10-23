'''
Diffusion Policy Infer - runs a diffusion policy and writes action chunking to policy SHM.

This module:
1. Reads observations from summary SHM (via BaseConsumer)
2. Runs a diffusion policy to compute action chunking
3. Writes action chunking into policy SHM for ActionExecutor to consume

Policy SHM is connect-only, created by ActionExecutor.

Author: Han Xue, Wang Zheng
'''
import sys
sys.path.append("..")  # To allow imports from parent directory
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_config import logger
from utils.config_parser import flatten_hydra_config
from utils.openpi_source import activate_openpi_source
from utils.inference_options import as_bool
from utils.superoffline_observation import capture_device_descriptors_from_config

from omegaconf import DictConfig
import hydra

# import debugpy; debugpy.connect(("0.0.0.0", 5678))


class _OpenPiPolicyInferMixin:
    def _parse_single_arm_action(
        self, action_chunk: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Parse single arm action chunk into position/quaternion/gripper components.
        """
        if action_chunk.shape[1] != 8:
            logger.warning(
                f"Expected action_chunk with 8 dimensions for single arm, got {action_chunk.shape[1]}"
            )
            return {}

        return {
            "single_arm": {
                "position": action_chunk[:, 0:3],
                "quaternion": action_chunk[:, 3:7],  # [x,y,z,w]
                "gripper": action_chunk[:, 7:8],
            }
        }

    def _parse_dual_arm_action(self, action_chunk: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Parse dual arm action chunk into left and right arm data.

        Args:
            action_chunk: Shape (chunk_length, 16) for dual arm
                         [left_pos(3), left_quat(4), left_gripper(1),
                          right_pos(3), right_quat(4), right_gripper(1)]

        Returns:
            Dict with 'left_arm' and 'right_arm' keys, each containing pos, quat, gripper
        """
        if action_chunk.shape[1] != 16:
            logger.warning(f"Expected action_chunk with 16 dimensions, got {action_chunk.shape[1]}")
            return {}

        parsed_actions = {}

        # Left arm: indices 0-7 (pos: 0-2, quat: 3-6, gripper: 7)
        parsed_actions['left_arm'] = {
            'position': action_chunk[:, 0:3],      # [chunk_length, 3]
            'quaternion': action_chunk[:, 3:7],    # [chunk_length, 4] - [x,y,z,w]
            'gripper': action_chunk[:, 7:8]        # [chunk_length, 1]
        }

        # Right arm: indices 8-15 (pos: 8-10, quat: 11-14, gripper: 15)
        parsed_actions['right_arm'] = {
            'position': action_chunk[:, 8:11],     # [chunk_length, 3]
            'quaternion': action_chunk[:, 11:15],  # [chunk_length, 4] - [x,y,z,w]
            'gripper': action_chunk[:, 15:16]      # [chunk_length, 1]
        }

        return parsed_actions

    def _parse_action_chunk(
        self,
        device_name: str,
        action_chunk: np.ndarray
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        if action_chunk.shape[1] == 16:
            return self._parse_dual_arm_action(action_chunk)
        return self._parse_single_arm_action(action_chunk)


def _flatten_config(cfg: DictConfig) -> Dict[str, Any]:
    return flatten_hydra_config(cfg)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _observation_recorder_config(root_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Bind enabled recordings to the composed, ordered OpenPi observations."""
    policy_cfg = root_cfg.get("policy", {})
    recorder = dict(policy_cfg.get("observation_recorder") or {})
    if not as_bool(recorder.get("enabled", False)):
        return recorder
    devices = policy_cfg.get("obs", {}).get("devices", [])
    if not isinstance(devices, list) or len(devices) != 3:
        raise ValueError("OpenPi observation_recorder requires left/right cameras and robot")
    recorder["camera_devices"] = devices[:2]
    recorder["robot_device"] = devices[2]
    recorder["capture_device_descriptors"] = capture_device_descriptors_from_config(
        recorder.get("superoffline_reviewed_source_profile", ""),
        devices=root_cfg.get("devices", []),
        camera_devices=recorder["camera_devices"],
        robot_device=recorder["robot_device"],
    )
    return recorder


def _config_file_path(config_name: str, *, repo_root: Path) -> Optional[str]:
    config_filename = config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
    candidate = repo_root / "configs" / config_filename
    return str(candidate) if candidate.is_file() else None


@hydra.main(
    config_path="../../configs",
    config_name="policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    merged_cfg = _flatten_config(cfg=cfg)

    # For Hydra mode, policy config is nested at top-level in the selected config

    policy_cfg = merged_cfg.get('policy', {})
    recorder_config = _observation_recorder_config(merged_cfg)

    # Select the requested OpenPi fork before PolicyConnector imports the policy
    # registry.  This is required because all forks use the package name ``openpi``.
    repo_root = _repository_root()
    openpi_source = activate_openpi_source(policy_cfg.get('openpi_src'), repo_root=repo_root)
    if openpi_source is not None:
        logger.info(f"OpenPi source activated: {openpi_source}")
    from consumer.inference.common.policy_connector import PolicyConnector

    class OpenPiPolicyInfer(_OpenPiPolicyInferMixin, PolicyConnector):
        def _get_visualizer_app_id(self) -> str:
            return "openpi_policy_visualizer"

    policy_class = policy_cfg.get('class', 'TinyMLPPolicy')
    policy_params = policy_cfg.get('params', {})
    fps = policy_cfg.get('fps', 50.0)
    control_fps = policy_cfg.get('control_fps', 50.0)
    enable_value_prediction = policy_cfg.get('enable_value_prediction', False)
    value_prediction_query_fps = policy_cfg.get('value_prediction_query_fps', fps)
    history_with_value = policy_cfg.get('history_with_value', False)
    chunk_length = policy_cfg.get('chunk_length', 10)

    obs_cfg = policy_cfg.get('obs', {})
    obs_devices = obs_cfg.get('devices')

    controls_cfg = policy_cfg.get('controls', [])
    reset_cfg = policy_cfg.get('reset', {})
    halt_and_home_cfg = policy_cfg.get('halt_and_home', {})

    master_device = policy_cfg.get('master_device', None)

    # Get config file path from Hydra for rollout saving
    # Hydra stores the config-name in HydraConfig
    config_file_path = None
    try:
        from hydra.core.hydra_config import HydraConfig
        if HydraConfig.initialized():
            config_name = HydraConfig.get().job.config_name
            logger.info(f"Hydra config name: {config_name}")
            config_file_path = _config_file_path(config_name, repo_root=repo_root)
            if config_file_path is None:
                logger.warning(
                    f"Config file not found under: {repo_root / 'configs' / config_name}"
                )
            else:
                logger.info(f"Config file found: {config_file_path}")
        else:
            logger.warning("HydraConfig not initialized, cannot get config file path")
    except Exception as e:
        logger.warning(f"Could not get config file path from Hydra: {e}")

    connector = OpenPiPolicyInfer(
        summary_shm_name=merged_cfg.get('summary_shm', 'device_summary_data'),
        policy_class=policy_class,
        policy_params=policy_params,
        obs_devices=obs_devices,
        controls=controls_cfg,
        reset_cfg=reset_cfg,
        halt_and_home_cfg=halt_and_home_cfg,
        fps=fps,
        control_fps=control_fps,
        enable_value_prediction=enable_value_prediction,
        value_prediction_query_fps=value_prediction_query_fps,
        history_with_value=history_with_value,
        chunk_length=chunk_length,
        master_device=master_device,
        infer_rollout_config=policy_cfg.get('infer_rollout'),
        observation_recorder_config=recorder_config,
        config_file_path=config_file_path,
        runtime_config=merged_cfg.get('runtime', {})
    )

    connector.run()


if __name__ == "__main__":
    main()
