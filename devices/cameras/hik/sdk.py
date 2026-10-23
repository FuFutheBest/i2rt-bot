#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: F822
"""Lazy Hikvision SDK boundary for the Hik camera adapter.

Author: OpenAI Assistant
"""

from __future__ import annotations

import importlib
from typing import Any

from utils.logger_config import logger

__all__ = [
    "HIK_CAMERA_AVAILABLE",
    "MV_ACCESS_Exclusive",
    "MV_CC_DEVICE_INFO",
    "MV_CC_DEVICE_INFO_LIST",
    "MV_FRAME_OUT",
    "MV_GENTL_CAMERALINK_DEVICE",
    "MV_GENTL_CXP_DEVICE",
    "MV_GENTL_GIGE_DEVICE",
    "MV_GENTL_XOF_DEVICE",
    "MV_GIGE_DEVICE",
    "MV_OK",
    "MV_TRIGGER_MODE_OFF",
    "MV_USB_DEVICE",
    "MVCC_ENUMVALUE",
    "MVCC_FLOATVALUE",
    "MVCC_INTVALUE",
    "MVCC_STRINGVALUE",
    "MvCamera",
    "PixelType_Gvsp_BGR8_Packed",
    "PixelType_Gvsp_BayerBG8",
    "PixelType_Gvsp_BayerGB8",
    "PixelType_Gvsp_BayerGR8",
    "PixelType_Gvsp_BayerRG8",
    "PixelType_Gvsp_Mono8",
    "PixelType_Gvsp_RGB8_Packed",
]

_SDK_LOADED = False


def _load_sdk() -> None:
    """Load Hikvision bindings only when a camera implementation requests them."""
    global _SDK_LOADED
    if _SDK_LOADED:
        return

    try:
        camera_control = importlib.import_module(
            "third_party.MvImport.MvCameraControl_class"
        )
        camera_params = importlib.import_module("third_party.MvImport.CameraParams_header")
        error_constants = importlib.import_module("third_party.MvImport.MvErrorDefine_const")
        camera_constants = importlib.import_module("third_party.MvImport.CameraParams_const")
        pixel_types = importlib.import_module("third_party.MvImport.PixelType_header")
        values: dict[str, Any] = {
            "HIK_CAMERA_AVAILABLE": True,
            "MvCamera": camera_control.MvCamera,
            "MV_OK": error_constants.MV_OK,
        }
        for name in __all__:
            if name in values or name == "HIK_CAMERA_AVAILABLE":
                continue
            source = pixel_types if name.startswith("PixelType_") else camera_params
            if not hasattr(source, name):
                source = camera_constants
            values[name] = getattr(source, name)
    except ImportError as exc:
        logger.warning(f"Hikvision camera modules not available: {exc}")
        values = {name: 0 for name in __all__}
        values.update(
            {
                "HIK_CAMERA_AVAILABLE": False,
                "MvCamera": None,
                "MV_CC_DEVICE_INFO": None,
                "MV_CC_DEVICE_INFO_LIST": None,
                "MV_FRAME_OUT": None,
                "MVCC_ENUMVALUE": None,
                "MVCC_FLOATVALUE": None,
                "MVCC_INTVALUE": None,
                "MVCC_STRINGVALUE": None,
            }
        )

    globals().update(values)
    _SDK_LOADED = True


def __getattr__(name: str) -> Any:
    """Resolve one SDK symbol on first use."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_sdk()
    return globals()[name]


def __dir__() -> list[str]:
    """Include lazy SDK exports in interactive discovery."""
    return sorted((*globals(), *__all__))
