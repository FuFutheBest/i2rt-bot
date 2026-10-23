'''
Diffusion Policy Infer - runs a diffusion policy and writes action chunking to policy SHM.

This module:
1. Reads observations from summary SHM (via BaseConsumer)
2. Runs a diffusion policy to compute action chunking
3. Writes action chunking into policy SHM for ActionExecutor to consume

Policy SHM is connect-only, created by ActionExecutor.

Author: Han Xue, Wang Zheng
'''
import argparse
import sys
sys.path.append("..")  # To allow imports from parent directory
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from consumer.inference.common.policy_connector import PolicyConnector
from utils.logger_config import logger
from utils.config_parser import load_config

from typing import cast
from omegaconf import DictConfig, OmegaConf
import hydra


class DiffusionPolicyInfer(PolicyConnector):
    def __init__(
        self,**kwargs) -> None:
        super().__init__(**kwargs)        

    def _get_visualizer_app_id(self) -> str:
        return "diffusion_policy_visualizer"

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

    def _parse_single_arm_action(self, action_chunk: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Parse single arm action chunk into position/quaternion/gripper components.
        """
        if action_chunk.shape[1] != 8:
            logger.warning(f"Expected action_chunk with 8 dimensions for single arm, got {action_chunk.shape[1]}")
            return {}

        return {
            'single_arm': {
                'position': action_chunk[:, 0:3],
                'quaternion': action_chunk[:, 3:7],  # [x,y,z,w]
                'gripper': action_chunk[:, 7:8]
            }
        }

    def _parse_action_chunk(
        self,
        device_name: str,
        action_chunk: np.ndarray
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        if action_chunk.shape[1] == 16:
            return self._parse_dual_arm_action(action_chunk)
        return self._parse_single_arm_action(action_chunk)


def load_policy_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
        
    return data


def _flatten_config(cfg: DictConfig) -> Dict[str, Any]:
    return cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


def _main_traditional(config_path: str) -> None:
    cfg_yaml = load_policy_yaml(config_path)
    policy_cfg = cfg_yaml.get('policy', {})

    policy_class = policy_cfg.get('class', 'TinyMLPPolicy')
    policy_params = policy_cfg.get('params', {})
    fps = policy_cfg.get('fps', 50.0)
    chunk_length = policy_cfg.get('chunk_length', 10)
    latency_steps = policy_cfg.get('latency_steps', 0)

    obs_cfg = policy_cfg.get('obs', {})
    obs_devices = obs_cfg.get('devices')

    controls_cfg = policy_cfg.get('controls', [])

    master_device = policy_cfg.get('master_device', None)
    enable_visualization = policy_cfg.get('enable_visualization', True)
    max_trajectory_points = policy_cfg.get('max_trajectory_points', 200)
    task_config_path = policy_cfg.get('task_config_path')
    task_config = load_config(task_config_path) if task_config_path else None

    connector = DiffusionPolicyInfer(
        summary_shm_name=cfg_yaml.get("summary_shm", "device_summary_data"),
        policy_class=policy_class,
        policy_params=policy_params,
        obs_devices=obs_devices,
        controls=controls_cfg,
        fps=fps,
        chunk_length=chunk_length,
        master_device=master_device,
        enable_visualization=enable_visualization,
        max_trajectory_points=max_trajectory_points,
        task_config=task_config,
        task_config_path=task_config_path,
        latency_steps=latency_steps,
        infer_recorder_config=policy_cfg.get("infer_recorder"),
        manual_confirm=policy_cfg.get("manual_confirm", False),
    )

    connector.run()


@hydra.main(
    config_path="../../configs",
    config_name="policy_infer/diffusion/config_infer_rizon_eef_dual",
    version_base="1.3",
)
def _main_hydra(cfg: DictConfig) -> None:
    merged_cfg = _flatten_config(cfg)

    # For Hydra mode, policy config is nested at top-level in the selected config
    # TODO: use hydra initialize to parse params for class instantiation

    policy_cfg = merged_cfg.get('policy', {})

    policy_class = policy_cfg.get('class', 'TinyMLPPolicy')
    policy_params = policy_cfg.get('params', {})
    fps = policy_cfg.get('fps', 50.0)
    chunk_length = policy_cfg.get('chunk_length', 10)
    robot_latency_steps = policy_cfg.get('robot_latency_steps', 0)

    obs_cfg = policy_cfg.get('obs', {})
    obs_devices = obs_cfg.get('devices')

    controls_cfg = policy_cfg.get('controls', [])

    master_device = policy_cfg.get('master_device', None)
    enable_visualization = policy_cfg.get('enable_visualization', True)
    max_trajectory_points = policy_cfg.get('max_trajectory_points', 200)
    task_config_path = policy_cfg.get('task_config_path')
    task_config = load_config(task_config_path) if task_config_path else None

    connector = DiffusionPolicyInfer(
        summary_shm_name=merged_cfg.get("summary_shm", "device_summary_data"),
        policy_class=policy_class,
        policy_params=policy_params,
        obs_devices=obs_devices,
        controls=controls_cfg,
        fps=fps,
        chunk_length=chunk_length,
        master_device=master_device,
        enable_visualization=enable_visualization,
        max_trajectory_points=max_trajectory_points,
        task_config=task_config,
        task_config_path=task_config_path,
        robot_latency_steps=robot_latency_steps,
        infer_recorder_config=policy_cfg.get("infer_recorder"),
        manual_confirm=policy_cfg.get("manual_confirm", False),
    )

    connector.run()


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(4071)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()

    parser = argparse.ArgumentParser(description="Policy Connector - run a policy and write action chunking to policy SHM")
    parser.add_argument("--config", type=str, required=False,
                        help="YAML path for policy connector config. If omitted, use Hydra config.")
    args, unknown = parser.parse_known_args()

    if args.config:
        _main_traditional(args.config)
    else:
        _main_hydra()
