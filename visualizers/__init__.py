"""Lazy exports for optional device visualizers."""

from __future__ import annotations

from typing import Any

__all__ = ["BaseVisualizer", "CameraVisualizer", "ActionVisualizer", "VISUALIZER_CLASSES"]


def __getattr__(name: str) -> Any:
    if name == "BaseVisualizer":
        from .base_visualizer import BaseVisualizer

        return BaseVisualizer
    if name == "CameraVisualizer":
        from .camera_visualizer import CameraVisualizer

        return CameraVisualizer
    if name == "ActionVisualizer":
        from .action_visualizer import ActionVisualizer

        return ActionVisualizer
    if name == "VISUALIZER_CLASSES":
        from .base_visualizer import BaseVisualizer
        from .camera_visualizer import CameraVisualizer

        return {
            "BaseDevice": BaseVisualizer,
            "BaseRobot": BaseVisualizer,
            "SimRobot": BaseVisualizer,
            "OpenCVCameraDevice": CameraVisualizer,
            "HikCameraDevice": CameraVisualizer,
            "WebCameraOpenPi": CameraVisualizer,
            "ViveTrackerDevice": BaseVisualizer,
            "WebCameraDevice": CameraVisualizer,
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
