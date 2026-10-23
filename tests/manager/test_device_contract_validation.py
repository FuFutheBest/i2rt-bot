"""DeviceManager validation of registry, config, and SHM stream contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from devices.contracts import DeviceKind
from manager.base_device_manager import BaseDeviceManager
from utils.shm_utils import pack_device_header


def _device(
    *,
    device_class="BaseDevice",
    shape=(7,),
    dtype="float64",
    format_shape=(7,),
) -> dict:
    header = pack_device_header(
        device_class,
        0,
        30.0,
        dtype,
        shape,
        64,
        0.0,
    )
    return {
        "device_class": device_class,
        "device_class_id": 0,
        "device_kind": DeviceKind.LOW_DIM_R,
        "shm": SimpleNamespace(buf=bytearray(header)),
        "config": {
            "state_feedback": {
                "format": {
                    "shape": list(format_shape),
                    "type": [f"linear:{format_shape[0]}"],
                    "dtype": "float64",
                }
            }
        },
    }


def test_manager_accepts_matching_low_dim_format_and_shm_header() -> None:
    device = _device()

    BaseDeviceManager._validate_device_contract(object(), device)

    assert device["data_shape"] == (7,)
    assert device["data_dtype"] == "float64"
    assert device["data_format_contract"].dimension == 7


def test_manager_rejects_config_shape_that_differs_from_shm() -> None:
    device = _device(shape=(7,), format_shape=(8,))

    with pytest.raises(ValueError, match="does not match SHM shape"):
        BaseDeviceManager._validate_device_contract(object(), device)


def test_manager_compares_long_class_names_using_shm_header_width() -> None:
    device = _device(device_class="RizonRobotLKForceGated")

    BaseDeviceManager._validate_device_contract(object(), device)

    assert device["device_header"]["device_type"] == "RizonRobotLKForceGat"


def test_manager_rejects_write_only_devices() -> None:
    device = _device()
    device["device_kind"] = DeviceKind.LOW_DIM_W

    with pytest.raises(ValueError, match="has no state stream"):
        BaseDeviceManager._validate_device_contract(object(), device)
