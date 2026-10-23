"""OpenPi RTC recordings preserve reviewed source roles without loading a model."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from consumer.inference import openpi_policy_infer
from utils.hydra_config import compose_hydra_application_config
from utils.superoffline_observation import (
    ObservationContractError,
    parse_recorder_observation_contract,
    validate_recording_artifacts,
)
from utils.superoffline_preflight import prepare_superoffline_input
from utils.superoffline_registry import get_superoffline_model_contract

_CONFIG = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc"


def _config(repo_root: Path) -> dict:
    return compose_hydra_application_config(_CONFIG, config_dir=repo_root / "configs")


def test_openpi_launch_records_input_accepted_by_superoffline(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from consumer.inference.common import policy_connector

    cfg = _config(repo_root)
    cfg["policy"]["observation_recorder"]["output_dir"] = str(tmp_path)
    cfg["policy"]["infer_rollout"] = {"enabled": False}
    inputs = {
        "WebCameraOpenPi_0": np.zeros((360, 640, 3), dtype=np.uint8),
        "WebCameraOpenPi_1": np.ones((360, 640, 3), dtype=np.uint8),
        "RizonRobot_2": np.zeros(44, dtype=np.float64),
    }
    captured = {}

    def fake_runtime(policy_class, params):
        # Recorder metadata must never become unsupported model kwargs.
        assert policy_class == "OpenPiUmiPolicy"
        assert "camera_devices" not in params
        assert "robot_device" not in params
        return SimpleNamespace(policy=SimpleNamespace())

    def record_without_hardware(connector):
        recorder = connector.observation_recorder
        captured["root"] = recorder.start(lambda: None)
        try:
            recorder.record_frame(
                100, {key: (100, value) for key, value in inputs.items()}
            )
            connector._record_exact_policy_input(
                {key: [value] for key, value in inputs.items()},
                100,
                action_index=0,
                inference_start_timestamp_ns=101,
            )
        finally:
            recorder.stop()

    monkeypatch.setattr(policy_connector, "PolicyRuntime", fake_runtime)
    monkeypatch.setattr(
        policy_connector.PolicyConnector, "run", record_without_hardware
    )
    monkeypatch.setattr(
        openpi_policy_infer, "activate_openpi_source", lambda *a, **k: None
    )
    openpi_policy_infer.main.__wrapped__(OmegaConf.create(cfg))

    root = captured["root"]
    manifest = json.loads((root / "manifest.json").read_text())
    contract = parse_recorder_observation_contract(manifest)
    assert contract.source_profile == "rizon-umi3-fisheye-openpi-rgb.v2"
    assert not contract.legacy_identity
    assert manifest["frame_count"] == manifest["inference_count"] == 1
    validate_recording_artifacts(root, contract)

    for fastwam in ("fastwam", "fastwam_memory"):
        models = [
            get_superoffline_model_contract(
                f"policy_infer/fastwam/config_infer_fisheye_umi3_dual_{fastwam}"
            ),
            get_superoffline_model_contract(
                "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
            ),
        ]
        prepared = prepare_superoffline_input(
            SimpleNamespace(observation_path=str(root), models=models),
            repo_root=repo_root,
        )
        for binding in prepared.bindings:
            assert binding.target_to_source["WebCameraOpenPi_0"] == "WebCameraOpenPi_0"
            assert binding.target_to_source["WebCameraOpenPi_1"] == "WebCameraOpenPi_1"


@pytest.mark.parametrize("mutation", ["duplicate", "rotation"])
def test_openpi_recorder_rejects_changed_capture_identity(
    repo_root: Path, mutation: str
):
    cfg = _config(repo_root)
    if mutation == "duplicate":
        cfg["policy"]["obs"]["devices"][:2] = ["WebCameraOpenPi_0", "WebCameraOpenPi_0"]
    elif mutation == "rotation":
        cfg["devices"][0]["rotation_degrees"] = 90
    with pytest.raises(ObservationContractError):
        openpi_policy_infer._observation_recorder_config(cfg)


def test_recorder_records_other_robot_serials_without_relabeling(repo_root: Path):
    cfg = _config(repo_root)
    cfg["devices"][2]["robot_sn"] = ["Other-Left", "Other-Right"]
    recorder = openpi_policy_infer._observation_recorder_config(cfg)
    assert recorder["capture_device_descriptors"]["robot"]["robot_serial_numbers"] == [
        "other-left", "other-right"
    ]


def test_disabled_openpi_recorder_does_not_require_source_metadata():
    assert openpi_policy_infer._observation_recorder_config({"policy": {}}) == {}


def test_reviewed_profile_accepts_robot_first_with_contiguous_ids(repo_root: Path):
    cfg = _config(repo_root)
    left, right, robot = cfg["devices"]
    cfg["devices"] = [robot, left, right]
    for index, device in enumerate(cfg["devices"]):
        device["device_class_id"] = index
    cfg["policy"]["obs"]["devices"] = [
        "WebCameraOpenPi_1", "WebCameraOpenPi_2", "RizonRobot_0"
    ]
    recorder = openpi_policy_infer._observation_recorder_config(cfg)
    descriptors = recorder["capture_device_descriptors"]
    assert descriptors["robot"]["device_key"] == "RizonRobot_0"
    assert descriptors["robot"]["device_class_id"] == "0"
    assert descriptors["cameras"]["left_wrist_rgb"]["device_class_id"] == "1"
