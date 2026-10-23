#!/usr/bin/env python3
"""RealSense SDK import shim for camera adapters.

Author: OpenAI Assistant
"""

from __future__ import annotations

from typing import Any

from utils.logger_config import logger

try:
    import pyrealsense2 as rs

    REALSENSE_CAMERA_AVAILABLE = True
except ImportError as exc:
    logger.warning(f"RealSense camera modules not available: {exc}")
    rs: Any = None
    REALSENSE_CAMERA_AVAILABLE = False
