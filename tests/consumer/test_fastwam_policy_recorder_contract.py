"""FastWAM launch contracts for trusted SuperOffline recorder input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from consumer.inference.fastwam_policy_infer import (
    _bind_observation_recorder_capture_identity,
    _build_connector_kwargs,
    _load_config,
)
from consumer.inference.openpi_policy_infer import _observation_recorder_config
from utils.logger_config import logger
from utils.observation_recorder import TimestampedObservationRecorder
from utils.superoffline_observation import (
    parse_recorder_observation_contract,
    validate_recording_artifacts,
)


_CONFIG_NAME = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"


@pytest.mark.parametrize(
    "family,variant",
    [("openpi", "openpi"), ("openpi", "openpi_rtc")]
    + [("fastwam", name) for name in ("fastwam", "fastwam_rtc", "fastwam_memory", "fasterwam")],
)
def test_native_recorder_writes_finalized_observations(
    family: str, variant: str, tmp_path: Path,
) -> None:
    config = _load_config(
        None, f"policy_infer/{family}/config_infer_fisheye_umi3_dual_{variant}"
    )
    policy = config["policy"]
    settings = dict(policy["observation_recorder"])
    assert settings["enabled"] is True
    assert settings["output_dir"] == f"logs/offline_observations/{variant}"
    assert settings["fps"] == 30.0
    assert settings["queue_size"] == 512
    assert settings["compress"] is False
    if family == "openpi":
        settings = _observation_recorder_config(config)
    else:
        settings = _bind_observation_recorder_capture_identity(
            config, settings, policy["params"], mock_mode=False
        )
    # Exercise the real writer using synthetic inputs in a temporary directory;
    # never put test observations into the operator's recording directories.
    settings["output_dir"] = str(tmp_path / variant)
    names = list(policy["obs"]["devices"])
    recorder = TimestampedObservationRecorder(
        settings, device_names=names, camera_devices=names[:2], robot_device=names[2]
    )
    arrays = {
        names[0]: np.zeros((360, 640, 3), dtype=np.uint8),
        names[1]: np.ones((360, 640, 3), dtype=np.uint8),
        names[2]: np.zeros(44, dtype=np.float64),
    }
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        root = recorder.start(lambda: None)
        try:
            recorder.record_frame(100, arrays)
            recorder.record_inference(100, {key: value[None] for key, value in arrays.items()})
        finally:
            recorder.stop()
    finally:
        logger.remove(sink)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["frame_count"] == manifest["inference_count"] == 1
    assert manifest["write_failures"] == manifest["dropped_frames"] == 0
    assert manifest["stopped_at_ns"] > 0
    validate_recording_artifacts(root, parse_recorder_observation_contract(manifest))
    assert any(f"enabled at {root.resolve()} (30.0 Hz)" in message for message in messages)
    assert any("frames=1, inferences=1" in message for message in messages)


def test_enabled_fastwam_recorder_receives_composed_capture_identity() -> None:
    root_config = _load_config(None, _CONFIG_NAME)
    policy = root_config["policy"]
    recorder = _bind_observation_recorder_capture_identity(
        root_config,
        dict(policy["observation_recorder"]),
        dict(policy["params"]),
        mock_mode=False,
    )

    descriptor = recorder["capture_device_descriptors"]
    assert descriptor["cameras"]["left_wrist_rgb"]["rotation_degrees"] == 0
    devices = root_config["devices"]
    if isinstance(devices, dict):
        devices = devices["devices"]
    robot = next(d for d in devices if d["class"] == "RizonRobot")
    assert descriptor["robot"]["robot_serial_numbers"] == [sn.lower() for sn in robot["robot_sn"]]


def test_disabled_mock_connector_does_not_construct_a_real_capture_identity() -> None:
    root_config = _load_config(None, _CONFIG_NAME)
    policy = dict(root_config["policy"])
    policy["observation_recorder"] = {"enabled": False}
    policy["params"] = {"robot_device": "RizonRobot_2"}

    kwargs = _build_connector_kwargs(root_config, policy)

    assert kwargs["observation_recorder_config"]["enabled"] is False
    assert "capture_device_descriptors" not in kwargs["observation_recorder_config"]


def test_mock_mode_cannot_enable_a_real_superoffline_recorder() -> None:
    root_config = _load_config(None, _CONFIG_NAME)
    policy = root_config["policy"]

    with pytest.raises(ValueError, match="not mock mode"):
        _bind_observation_recorder_capture_identity(
            root_config,
            dict(policy["observation_recorder"]),
            dict(policy["params"]),
            mock_mode=True,
        )
