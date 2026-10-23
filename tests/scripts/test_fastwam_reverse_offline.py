"""Hardware-free contracts for the FastWAM offline audit helpers."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import evaluate_fastwam_reverse_offline as evaluator_module
from scripts.analysis.capture_fastwam_offline_observation import write_observation_run
from scripts.analysis.evaluate_fastwam_reverse_offline import (
    _bound_devices,
    _dataset_from_binding,
    _load_cached_runtime,
    temporal_order_audit,
)
from scripts.analysis.fastwam_offline_utils import (
    ObservationDataset,
    batch_child_argv,
    parse_inference_indices,
)
from utils.superoffline_observation import (
    create_observation_binding,
    reviewed_capture_device_descriptors,
    write_observation_bindings,
)
from utils.superoffline_action_calibration import reviewed_action_calibration
from utils.superoffline_registry import get_superoffline_model_contract


def test_write_observation_run_matches_dataset_reader(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "observation"
    robot = np.arange(44, dtype=np.float64)
    write_observation_run(
        output_dir,
        arrays={
            "WebCameraOpenPi_1": np.full((360, 640, 3), 3, dtype=np.uint8),
            "WebCameraOpenPi_3": np.full((360, 640, 3), 5, dtype=np.uint8),
            "RizonRobot_2": robot,
        },
        device_timestamps_ns={
            "WebCameraOpenPi_1": 100,
            "WebCameraOpenPi_3": 101,
            "RizonRobot_2": 99,
        },
        camera_devices=["WebCameraOpenPi_1", "WebCameraOpenPi_3"],
        robot_device="RizonRobot_2",
        capture_device_descriptors=reviewed_capture_device_descriptors(
            "rizon-umi3-fisheye-openpi-rgb.v1"
        ),
    )

    dataset = ObservationDataset(output_dir)
    images, state, timestamps, observation_timestamp = (
        dataset.inference_observation_at(
            0,
            history_frames=1,
            history_stride=1,
            camera_devices=["WebCameraOpenPi_1", "WebCameraOpenPi_3"],
            robot_device="RizonRobot_2",
        )
    )

    assert [int(image[0, 0, 0]) for image in images[0]] == [3, 5]
    assert np.array_equal(state, np.concatenate([robot[14:22], robot[36:44]]))
    assert timestamps == [101]
    assert observation_timestamp == 101
    assert json.loads((output_dir / "manifest.json").read_text())["source"] == (
        "read_only_device_shm"
    )

    exact_observation, exact_timestamp = dataset.exact_policy_observation_at(
        0,
        camera_devices=["WebCameraOpenPi_1", "WebCameraOpenPi_3"],
        robot_device="RizonRobot_2",
    )

    assert exact_timestamp == 101
    assert int(exact_observation["WebCameraOpenPi_1"][0, 0, 0]) == 3
    assert int(exact_observation["WebCameraOpenPi_3"][0, 0, 0]) == 5
    assert np.array_equal(exact_observation["RizonRobot_2"], robot)


def test_temporal_order_audit_detects_reversed_runtime_order() -> None:
    state = np.zeros(16, dtype=np.float64)
    action = np.zeros((4, 16), dtype=np.float64)
    action[:, 0] = [0.4, 0.3, 0.2, 0.01]
    action[:, 8] = [0.5, 0.3, 0.1, 0.02]

    audit = temporal_order_audit(action, state)

    assert audit["recommended_execution_order"] == "time_reversed"
    assert not audit["runtime_order_looks_chronological"]
    assert audit["arms"]["left"]["runtime_first_distance_m"] == 0.4
    assert audit["arms"]["left"]["runtime_last_distance_m"] == 0.01


def test_temporal_order_audit_reports_mixed_arm_evidence() -> None:
    state = np.zeros(16, dtype=np.float64)
    action = np.zeros((3, 16), dtype=np.float64)
    action[:, 0] = [0.01, 0.1, 0.2]
    action[:, 8] = [0.2, 0.1, 0.01]

    audit = temporal_order_audit(action, state)

    assert audit["recommended_execution_order"] == "mixed"
    assert not audit["runtime_order_looks_chronological"]


def test_batch_cli_helpers_preserve_order_and_replace_single_sample_flags(
    tmp_path: Path,
) -> None:
    assert parse_inference_indices("4, 1,9") == [4, 1, 9]
    child = batch_child_argv(
        [
            "--config-name",
            "config",
            "--inference-indices",
            "4,1,9",
            "--output-dir",
            "old",
        ],
        inference_index=1,
        output_dir=tmp_path / "observations" / "000001",
    )

    assert "--inference-indices" not in child
    assert child[child.index("--inference-index") + 1] == "1"
    assert child[child.index("--output-dir") + 1].endswith("observations/000001")


def test_fastwam_batch_runtime_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Runtime:
        def __init__(self, params: dict[str, object], *, load_model: bool) -> None:
            calls.append(params)
            assert load_model is True

    monkeypatch.setattr(evaluator_module, "_CACHED_RUNTIME", None)
    params = {"ckpt_path": "/checkpoint", "device": "cuda"}

    first = _load_cached_runtime(Runtime, params)
    second = _load_cached_runtime(Runtime, dict(params))

    assert first is second
    assert calls == [params]


def test_executor_binding_remaps_fastwam_policy_keys_to_recorder_keys(
    tmp_path: Path,
) -> None:
    observation_dir = write_observation_run(
        tmp_path / "observation",
        arrays={
            "WebCameraOpenPi_1": np.full((360, 640, 3), 3, dtype=np.uint8),
            "WebCameraOpenPi_3": np.full((360, 640, 3), 5, dtype=np.uint8),
            "RizonRobot_2": np.arange(44, dtype=np.float64),
        },
        device_timestamps_ns={
            "WebCameraOpenPi_1": 100,
            "WebCameraOpenPi_3": 101,
            "RizonRobot_2": 99,
        },
        camera_devices=["WebCameraOpenPi_1", "WebCameraOpenPi_3"],
        robot_device="RizonRobot_2",
        capture_device_descriptors=reviewed_capture_device_descriptors(
            "rizon-umi3-fisheye-openpi-rgb.v1"
        ),
    )
    config_name = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    contract = get_superoffline_model_contract(config_name)
    calibration = reviewed_action_calibration(contract.action_calibration)
    manifest = json.loads((observation_dir / "manifest.json").read_text())
    binding = create_observation_binding(
        manifest=manifest,
        config_name=config_name,
        target_device_roles=contract.target_device_roles,
        allowed_source_profiles=contract.source_profiles,
        action_calibration_id=calibration.calibration_id,
        action_calibration_file_digests=calibration.file_digests,
        action_calibration_digest=calibration.digest,
        recording_root=observation_dir,
    )
    binding_path = tmp_path / "observation_bindings.json"
    write_observation_bindings(binding_path, [binding])

    dataset, resolved_binding = _dataset_from_binding(
        Namespace(
            observation_binding=binding_path,
            binding_index=0,
            config_name=config_name,
            observation_dir=observation_dir,
        )
    )

    assert resolved_binding == binding
    camera_devices, robot_device = _bound_devices(resolved_binding)
    observation, timestamp_ns = dataset.exact_policy_observation_at(
        0,
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    assert camera_devices == ["WebCameraOpenPi_0", "WebCameraOpenPi_1"]
    assert robot_device == "RizonRobot_2"
    assert timestamp_ns == 101
    assert int(observation["WebCameraOpenPi_0"][0, 0, 0]) == 3
    assert int(observation["WebCameraOpenPi_1"][0, 0, 0]) == 5
    assert np.array_equal(observation["RizonRobot_2"], np.arange(44, dtype=np.float64))
