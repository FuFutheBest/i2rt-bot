"""Hardware-free checks for I2RT vendor paths and gain fallbacks."""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np

from devices.robots.i2rt import I2RTRobot


def test_i2rt_default_server_script_resolves_from_repository_root() -> None:
    constructor = inspect.getsource(I2RTRobot.__init__)

    assert 'REPO_ROOT / "third_party" / "i2rt" / "scripts"' in constructor
    assert "Path(__file__).resolve().parents[1]" not in constructor


def test_i2rt_bilateral_gain_import_has_working_fallback(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args: None)
    monkeypatch.setattr(I2RTRobot, "stop_server", lambda _self: None)
    robot = I2RTRobot.__new__(I2RTRobot)

    robot._load_default_bilateral_gains()

    np.testing.assert_array_equal(
        robot._leader_default_kp,
        [250.0, 400.0, 400.0, 100.0, 25.0, 25.0],
    )
    np.testing.assert_array_equal(
        robot._follower_default_kp,
        [250.0, 400.0, 400.0, 100.0, 25.0, 25.0, 25.0],
    )
    assert robot._leader_zero_kp.shape == (6,)
    assert robot._follower_zero_kp.shape == (7,)
