#!/usr/bin/env python3
"""
Consumer Package - Data consumers for device manager.

This package provides various consumer implementations for processing
device manager data in different ways.

Author: Jun Lv, Han Xue
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BaseConsumer": ("consumer.core.base", "BaseConsumer"),
    "LatencyConsumer": ("consumer.visualization.latency_visualizer", "LatencyConsumer"),
    "DataSaverConsumer": ("consumer.io.data_saver", "DataSaverConsumer"),
    "PoseVisualizer": ("consumer.visualization.vive_tracker_visualizer", "PoseVisualizer"),
    "PolicyConnector": ("consumer.inference.common.policy_connector", "PolicyConnector"),
    "NEDF3Recorder": ("consumer.io.nedf3_recorder", "NEDF3Recorder"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily import consumers so entrypoints can configure dependencies first."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
