"""Role and binding contracts for read-only SuperOffline recordings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import utils.superoffline_observation as superoffline_observation
from scripts.analysis.capture_fastwam_offline_observation import write_observation_run
from utils.observation_recorder import TimestampedObservationRecorder
from utils.superoffline_action_calibration import reviewed_action_calibration
from utils.superoffline_observation import (
    LEGACY_IDENTITY_SOURCE_PROFILE,
    ObservationContractError,
    create_observation_binding,
    get_source_profile_contract,
    binding_policy_devices,
    parse_recorder_observation_contract,
    read_observation_binding,
    reviewed_capture_device_descriptors,
    validate_recording_artifacts,
    validate_binding_policy_devices,
    write_observation_bindings,
)
from utils.superoffline_registry import get_superoffline_model_contract


def _action_calibration_kwargs(model: object) -> dict[str, object]:
    calibration = reviewed_action_calibration(getattr(model, "action_calibration"))
    return {
        "action_calibration_id": calibration.calibration_id,
        "action_calibration_file_digests": calibration.file_digests,
        "action_calibration_digest": calibration.digest,
    }


def _recording(
    tmp_path: Path,
    *,
    source_profile: str = "rizon-umi3-fisheye-openpi-rgb.v1",
) -> Path:
    profile = get_source_profile_contract(source_profile)
    left = profile.capture_device_keys["left_wrist_rgb"]
    right = profile.capture_device_keys["right_wrist_rgb"]
    robot = profile.capture_device_keys["rizon_dual_arm_state"]
    return write_observation_run(
        tmp_path / "recording",
        arrays={
            left: np.zeros(
                profile.camera_shapes["left_wrist_rgb"], dtype=np.uint8
            ),
            right: np.ones(
                profile.camera_shapes["right_wrist_rgb"], dtype=np.uint8
            ),
            robot: np.zeros(44, dtype=np.float64),
        },
        device_timestamps_ns={
            left: 100,
            right: 101,
            robot: 99,
        },
        camera_devices=[left, right],
        robot_device=robot,
        source_profile=source_profile,
        capture_device_descriptors=reviewed_capture_device_descriptors(source_profile),
    )


def test_manifest_roles_bind_target_keys_without_guessing_source_names(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path)
    manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
    contract = parse_recorder_observation_contract(manifest)
    model = get_superoffline_model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    )

    binding = create_observation_binding(
        manifest=manifest,
        config_name=model.config_name,
        target_device_roles=model.target_device_roles,
        allowed_source_profiles=model.source_profiles,
        **_action_calibration_kwargs(model),
        recording_root=recording,
    )
    assert binding.target_to_source == {
        "WebCameraOpenPi_0": "WebCameraOpenPi_1",
        "WebCameraOpenPi_1": "WebCameraOpenPi_3",
        "RizonRobot_2": "RizonRobot_2",
    }
    validate_recording_artifacts(recording, contract)

    bindings_path = tmp_path / "bindings.json"
    write_observation_bindings(bindings_path, [binding])
    assert read_observation_binding(
        bindings_path,
        binding_index=0,
        config_name=model.config_name,
        observation_dir=recording,
    ) == binding


def test_role_parser_rejects_duplicate_device_and_swapped_shape(tmp_path: Path) -> None:
    recording = _recording(tmp_path, source_profile="rizon-umi3-iphone-rgb.v1")
    manifest_path = recording / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device_roles"]["right_wrist_rgb"]["device_key"] = "IPhoneCameraDevice_0"
    with pytest.raises(ObservationContractError, match="must be unique"):
        parse_recorder_observation_contract(manifest)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device_roles"]["left_wrist_rgb"]["shape"] = [44]
    with pytest.raises(ObservationContractError, match="HWC"):
        parse_recorder_observation_contract(manifest)


def test_source_provenance_rejects_capture_identity_mutation(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_provenance"]["capture_device_descriptors"]["cameras"][
        "left_wrist_rgb"
    ]["rotation_degrees"] = 90

    with pytest.raises(ObservationContractError, match="do not match reviewed source"):
        parse_recorder_observation_contract(manifest)


def test_historical_descriptor_keys_are_read_without_rewriting_manifest(tmp_path: Path) -> None:
    recording = _recording(tmp_path)
    manifest = json.loads((recording / "manifest.json").read_text())
    descriptors = manifest["source_provenance"]["capture_device_descriptors"]
    for descriptor in [*descriptors["cameras"].values(), descriptors["robot"]]:
        descriptor["device_id"] = descriptor.pop("device_class_id")
    before = json.dumps(manifest, sort_keys=True)
    contract = parse_recorder_observation_contract(manifest)
    assert contract.source_profile == "rizon-umi3-fisheye-openpi-rgb.v1"
    assert json.dumps(manifest, sort_keys=True) == before


@pytest.mark.parametrize("cameras", [None, [], 42])
def test_malformed_descriptors_raise_contract_error(cameras) -> None:
    profile = "rizon-umi3-fisheye-openpi-rgb.v2"
    descriptors = reviewed_capture_device_descriptors(profile)
    descriptors["cameras"] = cameras
    with pytest.raises(ObservationContractError):
        superoffline_observation._capture_device_descriptors(
            get_source_profile_contract(profile), descriptors
        )


def test_binding_cannot_retarget_a_registered_model_to_another_source(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path)
    manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
    source_model = get_superoffline_model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    )
    binding = create_observation_binding(
        manifest=manifest,
        config_name=source_model.config_name,
        target_device_roles=source_model.target_device_roles,
        allowed_source_profiles=source_model.source_profiles,
        **_action_calibration_kwargs(source_model),
        recording_root=recording,
    )
    raw = binding.as_dict()
    target_model = get_superoffline_model_contract(
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva"
    )
    raw["config_name"] = target_model.config_name
    raw["target_device_roles"] = dict(target_model.target_device_roles)
    raw["target_to_source"] = {
        target: manifest["device_roles"][role]["device_key"]
        for target, role in target_model.target_device_roles.items()
    }
    raw["digest"] = superoffline_observation._digest(
        {key: value for key, value in raw.items() if key != "digest"}
    )
    bindings_path = tmp_path / "retargeted-bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "schema": "superoffline.observation-bindings.v1",
                "bindings": [raw],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ObservationContractError, match="source profile.*not approved"):
        read_observation_binding(
            bindings_path,
            binding_index=0,
            config_name=target_model.config_name,
            observation_dir=recording,
        )


def test_binding_rejects_recomposed_policy_with_swapped_camera_order(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path)
    manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
    model = get_superoffline_model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    )
    binding = create_observation_binding(
        manifest=manifest,
        config_name=model.config_name,
        target_device_roles=model.target_device_roles,
        allowed_source_profiles=model.source_profiles,
        **_action_calibration_kwargs(model),
        recording_root=recording,
    )
    cameras, robot = binding_policy_devices(binding)
    assert validate_binding_policy_devices(
        binding,
        camera_devices=cameras,
        robot_device=robot,
        config_name=model.config_name,
    ) == (cameras, robot)

    with pytest.raises(ObservationContractError, match="device order"):
        validate_binding_policy_devices(
            binding,
            camera_devices=list(reversed(cameras)),
            robot_device=robot,
            config_name=model.config_name,
        )


def test_binding_rejects_unknown_profile_and_manifest_replacement(tmp_path: Path) -> None:
    recording = _recording(tmp_path, source_profile="rizon-umi3-iphone-rgb.v1")
    manifest_path = recording / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = get_superoffline_model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    )

    with pytest.raises(ObservationContractError, match="not approved"):
        create_observation_binding(
            manifest=manifest,
            config_name=model.config_name,
            target_device_roles=model.target_device_roles,
            allowed_source_profiles=model.source_profiles,
            **_action_calibration_kwargs(model),
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recording = _recording(tmp_path / "approved")
    manifest_path = recording / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = create_observation_binding(
        manifest=manifest,
        config_name=model.config_name,
        target_device_roles=model.target_device_roles,
        allowed_source_profiles=model.source_profiles,
        **_action_calibration_kwargs(model),
        recording_root=recording,
    )
    bindings_path = tmp_path / "bindings.json"
    write_observation_bindings(bindings_path, [binding])
    manifest["stopped_at_ns"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ObservationContractError, match="changed"):
        read_observation_binding(
            bindings_path,
            binding_index=0,
            config_name=model.config_name,
            observation_dir=recording,
        )


def test_legacy_manifest_allows_only_same_name_identity_binding(tmp_path: Path) -> None:
    manifest = {
        "camera_devices": ["WebCameraOpenPi_0", "WebCameraOpenPi_1"],
        "robot_device": "RizonRobot_2",
    }
    contract = parse_recorder_observation_contract(manifest)
    assert contract.legacy_identity is True
    assert contract.source_profile == LEGACY_IDENTITY_SOURCE_PROFILE
    model = get_superoffline_model_contract(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    )
    binding = create_observation_binding(
        manifest=manifest,
        config_name=model.config_name,
        target_device_roles=model.target_device_roles,
        allowed_source_profiles=model.source_profiles,
        **_action_calibration_kwargs(model),
    )
    assert binding.legacy_identity is True

    other_model = get_superoffline_model_contract(
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva"
    )
    with pytest.raises(ObservationContractError, match="strict same-name"):
        create_observation_binding(
            manifest=manifest,
            config_name=other_model.config_name,
            target_device_roles=other_model.target_device_roles,
            allowed_source_profiles=other_model.source_profiles,
            **_action_calibration_kwargs(other_model),
        )


def test_artifact_validation_rejects_ambiguous_index_array_mappings(
    tmp_path: Path,
) -> None:
    recording = _recording(tmp_path)
    manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
    contract = parse_recorder_observation_contract(manifest)

    (recording / "frames.jsonl").write_text(
        json.dumps(
            {
                "kind": "frame",
                "timestamp_ns": 100,
                "file": "frames/frame_000000.npz",
                "device_array_keys": {
                    "WebCameraOpenPi_1": "WebCameraOpenPi_1",
                    "WebCameraOpenPi_3": "WebCameraOpenPi_1",
                    "RizonRobot_2": "RizonRobot_2",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ObservationContractError, match="multiple recorder devices"):
        validate_recording_artifacts(recording, contract)

    (recording / "frames.jsonl").write_text(
        json.dumps(
            {
                "kind": "frame",
                "timestamp_ns": 100,
                "file": "frames/frame_000000.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (recording / "inference.jsonl").write_text(
        json.dumps(
            {
                "kind": "inference",
                "timestamp_ns": 100,
                "file": "inference/inference_000000.npz",
                "keys": [
                    {"name": "WebCameraOpenPi_1", "array_key": "WebCameraOpenPi_1"},
                    {"name": "WebCameraOpenPi_1", "array_key": "WebCameraOpenPi_3"},
                    {"name": "WebCameraOpenPi_3", "array_key": "WebCameraOpenPi_3"},
                    {"name": "RizonRobot_2", "array_key": "RizonRobot_2"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ObservationContractError, match="more than once"):
        validate_recording_artifacts(recording, contract)


def test_recorder_npz_member_keys_are_injective_for_similar_device_names() -> None:
    assert TimestampedObservationRecorder._array_key("device", "cam/a") != (
        TimestampedObservationRecorder._array_key("device", "cam_a")
    )
