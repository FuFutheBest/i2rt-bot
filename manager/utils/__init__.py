#!/usr/bin/env python3
"""
Action Manager Package - Contains different types of action manager

This package provides various manager implementations for deciding
the action for execution from time-varying action chunks

Author: Zheng Wang
"""

from collections.abc import Mapping
from typing import Any

from .action_manager import BaseChunkManager
from .auto_manager_v2 import AutoActionManagerV2
from .joint_action_manager import JointActionManager
from .nodelay_manager import NoDelayChunkManager
from .rtc_blend_manager import RtcBlendChunkManager
from .smooth_chunk_manager import SmoothChunkManager

__all__ = [
    "AutoActionManagerV2",
    "BaseChunkManager",
    "JointActionManager",
    "NoDelayChunkManager",
    "RtcBlendChunkManager",
    "SmoothChunkManager",
]

CHUNK_MANAGERS = {
    "base_manager": BaseChunkManager,
    "nodelay_manager": NoDelayChunkManager,
    "auto_manager_v2": AutoActionManagerV2,
    "smooth_chunk_manager": SmoothChunkManager,
    "rtc_blend_manager": RtcBlendChunkManager,
    "joint_action_manager": JointActionManager,
}


def load_chunk_manager(
    manager_config: Mapping[str, Any] | None = None,
    execution_fps: float | None = None,
) -> BaseChunkManager:
    """
    Load chunk manager from configuration dict.

    Args:
        manager_config: Configuration dictionary for the chunk manager
        execution_fps: Execution FPS from action_executor (used for SmartStartChunkManager and AutoActionManager)
    """
    resolved_config = manager_config or {}
    manager_name = resolved_config.get("type", "base_manager")
    assert manager_name in CHUNK_MANAGERS, (
        f"type `{manager_name}` not in {CHUNK_MANAGERS}"
    )
    MANAGER_CLASS = CHUNK_MANAGERS.get(manager_name)
    manager_para = {k: v for k, v in resolved_config.items() if k != "type"}

    # If execution_fps is provided and manager supports it, add it to parameters
    # This allows dt to be calculated from execution_fps instead of being configured directly
    if execution_fps is not None and manager_name in [
        "base_manager",
        "auto_manager_v2",
        "joint_action_manager",
        "smooth_chunk_manager",
        "rtc_blend_manager",
    ]:
        manager_para["execution_fps"] = execution_fps

    chunk_manager = MANAGER_CLASS(**manager_para)
    return chunk_manager
