"""Hydra-backed contracts for real-robot SuperOffline model selections."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import pytest

from incubator.control_panel.runtime_config import compose_runtime_config
from utils.superoffline_action_calibration import (
    ActionCalibrationError,
    validate_action_calibration_config,
)
from utils.superoffline_observation import (
    ObservationContractError,
    capture_device_descriptors_from_config,
    get_source_profile_contract,
)
from utils.superoffline_registry import SUPER_OFFLINE_MODEL_CONTRACTS


_SIMULATION_NAMESPACES = frozenset({"robotwin", "simulation", "simulator"})


def _device_key(device: Mapping[str, object]) -> str:
    return f"{device['class']}_{device['device_class_id']}"


def _ordered_targets(contract: object) -> list[str]:
    target_device_roles = getattr(contract, "target_device_roles")
    by_role = {role: target for target, role in target_device_roles.items()}
    return [
        by_role["left_wrist_rgb"],
        by_role["right_wrist_rgb"],
        by_role["rizon_dual_arm_state"],
    ]


def _effective_camera_rotation_degrees(device: Mapping[str, object]) -> int:
    device_class = str(device["class"])
    if device_class == "IPhoneCameraDevice":
        return int(device.get("rotation_degree", 0))
    if device_class == "WebCameraOpenPi":
        return int(device.get("rotation_degrees", 0))
    if device_class == "WebCameraDevice":
        # Base WebCameraDevice drops the legacy rotation_degrees YAML key.
        return 0
    raise AssertionError(f"unsupported SuperOffline camera class: {device_class}")


def _normalized_serials(value: object) -> tuple[str, ...]:
    assert isinstance(value, list)
    return tuple("".join(str(serial).split()).lower() for serial in value)


@pytest.mark.config
@pytest.mark.parametrize("config_name", sorted(SUPER_OFFLINE_MODEL_CONTRACTS))
def test_superoffline_contract_targets_match_composed_model_observations(
    config_name: str,
) -> None:
    """Every registered target is a composed device used by its model policy."""
    namespace = config_name.partition("/")[0]
    assert namespace not in _SIMULATION_NAMESPACES

    config = compose_runtime_config(config_name)
    configured_devices = {
        _device_key(device): device
        for device in config["devices"]
        if isinstance(device, Mapping)
    }
    policy = config["policy"]
    assert isinstance(policy, Mapping)
    observation = policy["obs"]
    assert isinstance(observation, Mapping)
    policy_device_keys = list(observation["devices"])

    contract = SUPER_OFFLINE_MODEL_CONTRACTS[config_name]
    expected_devices = _ordered_targets(contract)
    assert policy_device_keys == expected_devices
    assert set(expected_devices) <= set(configured_devices)

    source_profile = get_source_profile_contract(contract.source_profiles[0])
    capture_descriptor = capture_device_descriptors_from_config(
        contract.source_profiles[0],
        devices=config["devices"],
        camera_devices=policy_device_keys[:2],
        robot_device=policy_device_keys[2],
    )
    assert capture_descriptor["cameras"]["left_wrist_rgb"]["format_type"] == "rgb"
    assert capture_descriptor["robot"]["state_format_type"] == [
        "j1-7",
        "dq1-7",
        "xyz",
        "qwqxqyqz",
        "gripper",
        "j1-7",
        "dq1-7",
        "xyz",
        "qwqxqyqz",
        "gripper",
    ]
    for role in ("left_wrist_rgb", "right_wrist_rgb"):
        device_key = source_profile.capture_device_keys[role]
        device = configured_devices[device_key]
        state_feedback = device["state_feedback"]
        assert isinstance(state_feedback, Mapping)
        image_format = state_feedback["format"]
        assert isinstance(image_format, Mapping)
        assert tuple(image_format["shape"]) == source_profile.camera_shapes[role]
        assert (
            _effective_camera_rotation_degrees(device)
            == source_profile.camera_rotation_degrees[role]
        )

    robot = configured_devices[source_profile.capture_device_keys["rizon_dual_arm_state"]]
    assert _normalized_serials(robot["robot_sn"]) == tuple(
        capture_descriptor["robot"]["robot_serial_numbers"]
    )

    params = policy["params"]
    assert isinstance(params, Mapping)
    calibration = validate_action_calibration_config(
        contract.action_calibration,
        params,
    )
    assert calibration.calibration_id == contract.action_calibration.calibration_id

    if "camera_devices" in params:
        assert list(params["camera_devices"]) == expected_devices[:2]
        assert params["robot_device"] == expected_devices[2]


@pytest.mark.config
def test_capture_descriptor_rejects_physical_identity_mutations() -> None:
    config = compose_runtime_config("policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam")
    policy_devices = list(config["policy"]["obs"]["devices"])

    wrong_rotation = deepcopy(config["devices"])
    wrong_rotation[0]["rotation_degrees"] = 90
    with pytest.raises(ObservationContractError, match="do not match reviewed source"):
        capture_device_descriptors_from_config(
            "rizon-umi3-fisheye-openpi-rgb.v1",
            devices=wrong_rotation,
            camera_devices=policy_devices[:2],
            robot_device=policy_devices[2],
        )

    wrong_serial = deepcopy(config["devices"])
    wrong_serial[2]["robot_sn"] = ["Rizon4-000001", "Rizon4R-000002"]
    with pytest.raises(ObservationContractError, match="do not match reviewed source"):
        capture_device_descriptors_from_config(
            "rizon-umi3-fisheye-openpi-rgb.v1",
            devices=wrong_serial,
            camera_devices=policy_devices[:2],
            robot_device=policy_devices[2],
        )

    wrong_state_layout = deepcopy(config["devices"])
    wrong_state_layout[2]["state_feedback"]["format"]["type"] = [
        "dq1-7",
        "j1-7",
        "xyz",
        "qwqxqyqz",
        "gripper",
        "j1-7",
        "dq1-7",
        "xyz",
        "qwqxqyqz",
        "gripper",
    ]
    with pytest.raises(ObservationContractError, match="do not match reviewed source"):
        capture_device_descriptors_from_config(
            "rizon-umi3-fisheye-openpi-rgb.v1",
            devices=wrong_state_layout,
            camera_devices=policy_devices[:2],
            robot_device=policy_devices[2],
        )


@pytest.mark.config
@pytest.mark.parametrize(
    ("config_name", "parameter", "wrong_value"),
    [
        (
            "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
            "gripper_max_width",
            0.11,
        ),
        (
            "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
            "clip_gripper_width",
            False,
        ),
        (
            "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
            "left_gripper_offset",
            0.005,
        ),
    ],
)
def test_action_calibration_rejects_gripper_semantic_mutations(
    config_name: str,
    parameter: str,
    wrong_value: object,
) -> None:
    config = compose_runtime_config(config_name)
    params = deepcopy(config["policy"]["params"])
    params[parameter] = wrong_value

    with pytest.raises(ActionCalibrationError, match=parameter):
        validate_action_calibration_config(
            SUPER_OFFLINE_MODEL_CONTRACTS[config_name].action_calibration,
            params,
        )
