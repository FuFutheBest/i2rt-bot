"""Trusted recorder-role bindings for SuperOffline comparisons.

Offline evaluators must never infer camera identity from a device-name suffix.
The recorder describes the three physical roles it captured, and the executor
derives a target-policy key binding from that immutable manifest contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


RECORDER_OBSERVATION_CONTRACT = "superoffline.recorder-observation-roles.v1"
OBSERVATION_BINDINGS_SCHEMA = "superoffline.observation-bindings.v1"
SOURCE_PROFILE_PROVENANCE_SCHEMA = "superoffline.source-profile-provenance.v1"
CAPTURE_DEVICE_DESCRIPTOR_SCHEMA = "superoffline.capture-device-descriptors.v1"
LEGACY_IDENTITY_SOURCE_PROFILE = "legacy.identity.v1"
RIZON_DUAL_ARM_STATE_LAYOUT = "rizon.dual-arm-state-44d.v1"
CANONICAL_OBSERVATION_ROLES = (
    "left_wrist_rgb",
    "right_wrist_rgb",
    "rizon_dual_arm_state",
)

_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FISHEYE_OPENPI_SOURCE_PROFILE = "rizon-umi3-fisheye-openpi-rgb.v1"
FISHEYE_OPENPI_RTC_SOURCE_PROFILE = "rizon-umi3-fisheye-openpi-ids-2-0-rgb.v1"
FISHEYE_LINGBOTVA_SOURCE_PROFILE = "rizon-umi3-fisheye-lingbotva-rgb.v1"
IPHONE_SOURCE_PROFILE = "rizon-umi3-iphone-rgb.v1"
LEGACY_IPHONE_SOURCE_PROFILE = "rizon-iphone-dual-legacy-rgb.v1"


class ObservationContractError(ValueError):
    """A recorder or model binding is incomplete, ambiguous, or incompatible."""


@dataclass(frozen=True)
class SourceProfileContract:
    """Reviewed physical capture and action-calibration identity for a recorder."""

    source_profile: str
    camera_rig_id: str
    capture_pipeline_id: str
    robot_rig_id: str
    required_calibration_labels: tuple[str, ...]
    camera_shapes: Mapping[str, tuple[int, int, int]]
    camera_rotation_degrees: Mapping[str, int]
    capture_device_keys: Mapping[str, str]
    robot_serial_numbers: tuple[str, str]
    flexible_device_keys: bool = False
    camera_format_dtype: str = "uint8"
    camera_format_type: str = "rgb"
    robot_state_shape: tuple[int, ...] = (44,)
    robot_state_dtype: str = "float64"
    robot_state_format_type: tuple[str, ...] = (
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
    )
    action_frame_id: str = "rizon.base_tcp.v1"
    action_units: str = "position_m_quaternion_xyzw_gripper_width_m"


_CALIBRATION_LABELS = (
    "tcp_to_umi",
    "left_base_to_world",
    "right_base_to_world",
    "left_gripper",
    "right_gripper",
)

_REVIEWED_CALIBRATION_PATHS = {
    "tcp_to_umi": "calibration/tcp_to_umi_transform.json",
    "left_base_to_world": "calibration/left_base_to_world.json",
    "right_base_to_world": "calibration/right_base_to_world.json",
    "left_gripper": (
        "configs/devices/calibration/lktech_gripper/"
        "LKTech-Gripper-1_2_id1_calibration.json"
    ),
    "right_gripper": (
        "configs/devices/calibration/lktech_gripper/"
        "LKTech-Gripper-2_2_id1_calibration.json"
    ),
}

SOURCE_PROFILE_CONTRACTS: Mapping[str, SourceProfileContract] = {
    FISHEYE_OPENPI_SOURCE_PROFILE: SourceProfileContract(
        source_profile=FISHEYE_OPENPI_SOURCE_PROFILE,
        camera_rig_id="rizon-umi3-fisheye-openpi-ids-1-3.v1",
        capture_pipeline_id="webcamera-openpi-mjpg-rgb-360x640-rotation0-controls.v1",
        robot_rig_id="rizon-umi3-dual-063256-062088.v1",
        required_calibration_labels=_CALIBRATION_LABELS,
        camera_shapes={
            "left_wrist_rgb": (360, 640, 3),
            "right_wrist_rgb": (360, 640, 3),
        },
        camera_rotation_degrees={
            "left_wrist_rgb": 0,
            "right_wrist_rgb": 0,
        },
        capture_device_keys={
            "left_wrist_rgb": "WebCameraOpenPi_1",
            "right_wrist_rgb": "WebCameraOpenPi_3",
            "rizon_dual_arm_state": "RizonRobot_2",
        },
        robot_serial_numbers=("Rizon4-063256", "Rizon4R-062088"),
    ),
    FISHEYE_OPENPI_RTC_SOURCE_PROFILE: SourceProfileContract(
        source_profile=FISHEYE_OPENPI_RTC_SOURCE_PROFILE,
        camera_rig_id="rizon-umi3-fisheye-openpi-ids-2-0.v1",
        capture_pipeline_id="webcamera-openpi-mjpg-rgb-360x640-rotation0-controls.v1",
        robot_rig_id="rizon-umi3-dual-063256-062088.v1",
        required_calibration_labels=_CALIBRATION_LABELS,
        camera_shapes={
            "left_wrist_rgb": (360, 640, 3),
            "right_wrist_rgb": (360, 640, 3),
        },
        camera_rotation_degrees={"left_wrist_rgb": 0, "right_wrist_rgb": 0},
        capture_device_keys={
            "left_wrist_rgb": "WebCameraOpenPi_2",
            "right_wrist_rgb": "WebCameraOpenPi_0",
            "rizon_dual_arm_state": "RizonRobot_2",
        },
        robot_serial_numbers=("Rizon4-063256", "Rizon4R-062088"),
    ),
    FISHEYE_LINGBOTVA_SOURCE_PROFILE: SourceProfileContract(
        source_profile=FISHEYE_LINGBOTVA_SOURCE_PROFILE,
        camera_rig_id="rizon-umi3-fisheye-webcamera-ids-1-0.v1",
        capture_pipeline_id="webcamera-device-mjpg-rgb-360x640-native-orientation.v1",
        robot_rig_id="rizon-umi3-dual-063256-062088.v1",
        required_calibration_labels=_CALIBRATION_LABELS,
        camera_shapes={
            "left_wrist_rgb": (360, 640, 3),
            "right_wrist_rgb": (360, 640, 3),
        },
        # WebCameraDevice does not consume rotation_degrees, so recorder arrays
        # retain their native landscape orientation despite legacy YAML fields.
        camera_rotation_degrees={
            "left_wrist_rgb": 0,
            "right_wrist_rgb": 0,
        },
        capture_device_keys={
            "left_wrist_rgb": "WebCameraDevice_1",
            "right_wrist_rgb": "WebCameraDevice_0",
            "rizon_dual_arm_state": "RizonRobot_2",
        },
        robot_serial_numbers=("Rizon4-063256", "Rizon4R-062088"),
    ),
    IPHONE_SOURCE_PROFILE: SourceProfileContract(
        source_profile=IPHONE_SOURCE_PROFILE,
        camera_rig_id="rizon-umi3-iphone-ids-0-1.v1",
        capture_pipeline_id="iphone-rgb-left-640x480-rotation90-right-480x640-rotation0.v1",
        robot_rig_id="rizon-umi3-dual-063256-062088.v1",
        required_calibration_labels=_CALIBRATION_LABELS,
        camera_shapes={
            "left_wrist_rgb": (640, 480, 3),
            "right_wrist_rgb": (480, 640, 3),
        },
        camera_rotation_degrees={
            "left_wrist_rgb": 90,
            "right_wrist_rgb": 0,
        },
        capture_device_keys={
            "left_wrist_rgb": "IPhoneCameraDevice_0",
            "right_wrist_rgb": "IPhoneCameraDevice_1",
            "rizon_dual_arm_state": "RizonRobot_2",
        },
        robot_serial_numbers=("Rizon4-063256", "Rizon4R-062088"),
    ),
    LEGACY_IPHONE_SOURCE_PROFILE: SourceProfileContract(
        source_profile=LEGACY_IPHONE_SOURCE_PROFILE,
        camera_rig_id="rizon-iphone-dual-ids-0-1-native-rgb.v1",
        capture_pipeline_id="iphone-rgb-dual-480x640-no-rotation.v1",
        robot_rig_id="rizon-dual-062931-00044-dahuan.v1",
        required_calibration_labels=_CALIBRATION_LABELS,
        camera_shapes={
            "left_wrist_rgb": (480, 640, 3),
            "right_wrist_rgb": (480, 640, 3),
        },
        camera_rotation_degrees={
            "left_wrist_rgb": 0,
            "right_wrist_rgb": 0,
        },
        capture_device_keys={
            "left_wrist_rgb": "IPhoneCameraDevice_0",
            "right_wrist_rgb": "IPhoneCameraDevice_1",
            "rizon_dual_arm_state": "RizonRobot_2",
        },
        robot_serial_numbers=("Rizon4-062931", "Rizon4-00044"),
    ),
}


def _required_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ObservationContractError(f"{field} must be a non-empty string")
    return text


def _device_class_id(value: Any, *, field: str) -> str:
    if value is None or isinstance(value, bool):
        raise ObservationContractError(f"{field} must be a non-empty device id")
    text = str(value).strip()
    if not text:
        raise ObservationContractError(f"{field} must be a non-empty device id")
    return text


# v1 identifiers remain readable for immutable historical recordings. New
# recordings bind semantic roles to the composed device keys, not USB numbers.
SUPER_OFFLINE_REVIEWED_SOURCE_PROFILES = {
    FISHEYE_OPENPI_SOURCE_PROFILE: "rizon-umi3-fisheye-openpi-rgb.v2",
    FISHEYE_LINGBOTVA_SOURCE_PROFILE: "rizon-umi3-fisheye-lingbotva-rgb.v2",
    IPHONE_SOURCE_PROFILE: "rizon-umi3-iphone-rgb.v2",
}
for _old_profile, _new_profile in SUPER_OFFLINE_REVIEWED_SOURCE_PROFILES.items():
    _old_contract = SOURCE_PROFILE_CONTRACTS[_old_profile]
    _camera_class = _old_contract.capture_device_keys["left_wrist_rgb"].rsplit("_", 1)[0]
    SOURCE_PROFILE_CONTRACTS[_new_profile] = replace(
        _old_contract,
        source_profile=_new_profile,
        camera_rig_id=_new_profile,
        robot_rig_id="rizon-umi3-dual-arm.v2",
        flexible_device_keys=True,
        capture_device_keys={
            "left_wrist_rgb": f"{_camera_class}_0",
            "right_wrist_rgb": f"{_camera_class}_1",
            "rizon_dual_arm_state": "RizonRobot_2",
        },
    )


def _source_profile(value: Any, *, field: str = "source_profile") -> str:
    profile = _required_text(value, field=field)
    if _PROFILE_RE.fullmatch(profile) is None:
        raise ObservationContractError(
            f"{field} must contain only letters, digits, dot, underscore, or dash"
        )
    return profile


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ObservationContractError(f"could not hash recorder artifact {path}") from exc
    return digest.hexdigest()


def get_source_profile_contract(source_profile: str) -> SourceProfileContract:
    """Return the code-owned contract for one reviewed recorder source."""
    profile = _source_profile(source_profile)
    try:
        return SOURCE_PROFILE_CONTRACTS[profile]
    except KeyError as exc:
        raise ObservationContractError(
            f"source_profile {profile!r} is not a registered reviewed source"
        ) from exc


def _device_key_parts(device_key: str, *, field: str) -> tuple[str, str]:
    device_class, separator, device_class_id = _required_text(device_key, field=field).rpartition(
        "_"
    )
    if not separator or not device_class or not device_class_id:
        raise ObservationContractError(
            f"{field} must contain a device class and device id separated by an underscore"
        )
    return device_class, device_class_id


def _normalized_robot_serials(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObservationContractError(f"{field} must be an array of two serial numbers")
    if len(value) != 2:
        raise ObservationContractError(f"{field} must contain exactly two serial numbers")
    return [
        "".join(_required_text(serial, field=field).split()).casefold()
        for serial in value
    ]


def _format_type_sequence(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObservationContractError(f"{field} must be an ordered array of primitives")
    result = [_required_text(item, field=field) for item in value]
    if not result:
        raise ObservationContractError(f"{field} must not be empty")
    return result


def _expected_capture_device_descriptors(
    contract: SourceProfileContract,
) -> dict[str, Any]:
    cameras: dict[str, dict[str, Any]] = {}
    for role in ("left_wrist_rgb", "right_wrist_rgb"):
        device_key = contract.capture_device_keys[role]
        device_class, device_class_id = _device_key_parts(
            device_key,
            field=f"reviewed source profile camera device {role}",
        )
        cameras[role] = {
            "device_key": device_key,
            "class": device_class,
            "device_class_id": device_class_id,
            "format_shape": list(contract.camera_shapes[role]),
            "format_dtype": contract.camera_format_dtype,
            "format_type": contract.camera_format_type,
            "rotation_degrees": int(contract.camera_rotation_degrees[role]),
        }
    robot_key = contract.capture_device_keys["rizon_dual_arm_state"]
    robot_class, robot_id = _device_key_parts(
        robot_key,
        field="reviewed source profile robot device",
    )
    return {
        "schema": CAPTURE_DEVICE_DESCRIPTOR_SCHEMA,
        "cameras": cameras,
        "robot": {
            "device_key": robot_key,
            "class": robot_class,
            "device_class_id": robot_id,
            "robot_serial_numbers": _normalized_robot_serials(
                contract.robot_serial_numbers,
                field="reviewed source profile robot serial numbers",
            ),
            "state_format_shape": list(contract.robot_state_shape),
            "state_format_dtype": contract.robot_state_dtype,
            "state_format_type": list(contract.robot_state_format_type),
        },
    }


def _capture_device_descriptors(
    contract: SourceProfileContract,
    value: Any,
) -> dict[str, Any]:
    """Normalize and prove one capture descriptor matches its source profile."""
    if not isinstance(value, Mapping):
        raise ObservationContractError(
            "capture_device_descriptors must be an object derived from composed devices"
        )
    # Read historical v1 manifests without rewriting their digest-covered bytes.
    import copy
    value = copy.deepcopy(dict(value))
    legacy_ids = False
    raw_legacy_cameras = value.get("cameras")
    descriptors = list(raw_legacy_cameras.values()) if isinstance(raw_legacy_cameras, Mapping) else []
    descriptors.append(value.get("robot"))
    for descriptor in descriptors:
        if isinstance(descriptor, dict) and "device_id" in descriptor:
            if "device_class_id" in descriptor:
                raise ObservationContractError("mixed historical/current device identity fields")
            descriptor["device_class_id"] = descriptor.pop("device_id")
            legacy_ids = True
    if legacy_ids and contract.flexible_device_keys:
        raise ObservationContractError("v2 source profiles require device_class_id")
    if set(value) != {"schema", "cameras", "robot"}:
        raise ObservationContractError(
            "capture_device_descriptors has unsupported or missing fields"
        )
    if value.get("schema") != CAPTURE_DEVICE_DESCRIPTOR_SCHEMA:
        raise ObservationContractError(
            "capture_device_descriptors.schema is not supported"
        )
    expected = _expected_capture_device_descriptors(contract)
    raw_cameras = value.get("cameras")
    if not isinstance(raw_cameras, Mapping) or set(raw_cameras) != {
        "left_wrist_rgb",
        "right_wrist_rgb",
    }:
        raise ObservationContractError(
            "capture_device_descriptors.cameras must describe both wrist roles"
        )
    normalized_cameras: dict[str, dict[str, Any]] = {}
    for role in ("left_wrist_rgb", "right_wrist_rgb"):
        raw_camera = raw_cameras.get(role)
        if not isinstance(raw_camera, Mapping) or set(raw_camera) != {
            "device_key",
            "class",
            "device_class_id",
            "format_shape",
            "format_dtype",
            "format_type",
            "rotation_degrees",
        }:
            raise ObservationContractError(
                f"capture_device_descriptors.cameras.{role} has unsupported or missing fields"
            )
        rotation = raw_camera.get("rotation_degrees")
        if isinstance(rotation, bool):
            raise ObservationContractError(
                f"capture_device_descriptors.cameras.{role}.rotation_degrees must be an integer"
            )
        try:
            parsed_rotation = int(rotation)
        except (TypeError, ValueError) as exc:
            raise ObservationContractError(
                f"capture_device_descriptors.cameras.{role}.rotation_degrees must be an integer"
            ) from exc
        normalized_cameras[role] = {
            "device_key": _required_text(
                raw_camera.get("device_key"),
                field=f"capture_device_descriptors.cameras.{role}.device_key",
            ),
            "class": _required_text(
                raw_camera.get("class"),
                field=f"capture_device_descriptors.cameras.{role}.class",
            ),
            "device_class_id": _device_class_id(
                raw_camera.get("device_class_id"),
                field=f"capture_device_descriptors.cameras.{role}.device_class_id",
            ),
            "format_shape": list(
                _shape(
                    raw_camera.get("format_shape"),
                    field=f"capture_device_descriptors.cameras.{role}.format_shape",
                )
            ),
            "format_dtype": _dtype(
                raw_camera.get("format_dtype"),
                field=f"capture_device_descriptors.cameras.{role}.format_dtype",
            ),
            "format_type": _required_text(
                raw_camera.get("format_type"),
                field=f"capture_device_descriptors.cameras.{role}.format_type",
            ),
            "rotation_degrees": parsed_rotation,
        }
    raw_robot = value.get("robot")
    if not isinstance(raw_robot, Mapping) or set(raw_robot) != {
        "device_key",
        "class",
        "device_class_id",
        "robot_serial_numbers",
        "state_format_shape",
        "state_format_dtype",
        "state_format_type",
    }:
        raise ObservationContractError(
            "capture_device_descriptors.robot has unsupported or missing fields"
        )
    normalized = {
        "schema": CAPTURE_DEVICE_DESCRIPTOR_SCHEMA,
        "cameras": normalized_cameras,
        "robot": {
            "device_key": _required_text(
                raw_robot.get("device_key"),
                field="capture_device_descriptors.robot.device_key",
            ),
            "class": _required_text(
                raw_robot.get("class"),
                field="capture_device_descriptors.robot.class",
            ),
            "device_class_id": _device_class_id(
                raw_robot.get("device_class_id"),
                field="capture_device_descriptors.robot.device_class_id",
            ),
            "robot_serial_numbers": _normalized_robot_serials(
                raw_robot.get("robot_serial_numbers"),
                field="capture_device_descriptors.robot.robot_serial_numbers",
            ),
            "state_format_shape": list(
                _shape(
                    raw_robot.get("state_format_shape"),
                    field="capture_device_descriptors.robot.state_format_shape",
                )
            ),
            "state_format_dtype": _dtype(
                raw_robot.get("state_format_dtype"),
                field="capture_device_descriptors.robot.state_format_dtype",
            ),
            "state_format_type": _format_type_sequence(
                raw_robot.get("state_format_type"),
                field="capture_device_descriptors.robot.state_format_type",
            ),
        },
    }
    # Serials are provenance, not a cross-robot compatibility restriction.
    expected["robot"]["robot_serial_numbers"] = normalized["robot"]["robot_serial_numbers"]
    if contract.flexible_device_keys:
        for actual, template in [
            *((normalized["cameras"][role], expected["cameras"][role])
              for role in ("left_wrist_rgb", "right_wrist_rgb")),
            (normalized["robot"], expected["robot"]),
        ]:
            if actual["device_key"] != f'{actual["class"]}_{actual["device_class_id"]}':
                raise ObservationContractError("capture device key does not match class/id")
            template["device_key"] = actual["device_key"]
            template["device_class_id"] = actual["device_class_id"]
    if normalized != expected:
        raise ObservationContractError(
            f"capture device descriptors do not match reviewed source profile "
            f"{contract.source_profile!r}"
        )
    if legacy_ids:
        for descriptor in [*expected["cameras"].values(), expected["robot"]]:
            descriptor["device_id"] = descriptor.pop("device_class_id")
    return expected


def reviewed_capture_device_descriptors(source_profile: str) -> dict[str, Any]:
    """Return a reviewed descriptor for synthetic recorder fixtures only."""
    return _expected_capture_device_descriptors(get_source_profile_contract(source_profile))


def _camera_rotation_from_device(device: Mapping[str, Any]) -> int:
    device_class = _required_text(device.get("class"), field="capture camera class")
    if device_class == "IPhoneCameraDevice":
        raw_rotation = device.get("rotation_degree", 0)
    elif device_class == "WebCameraOpenPi":
        raw_rotation = device.get("rotation_degrees", 0)
    elif device_class == "WebCameraDevice":
        # The base device ignores the legacy rotation_degrees YAML field.
        raw_rotation = 0
    else:
        raise ObservationContractError(
            f"unsupported SuperOffline capture camera class {device_class!r}"
        )
    if isinstance(raw_rotation, bool):
        raise ObservationContractError("capture camera rotation must be an integer")
    try:
        return int(raw_rotation)
    except (TypeError, ValueError) as exc:
        raise ObservationContractError("capture camera rotation must be an integer") from exc


def capture_device_descriptors_from_config(
    source_profile: str,
    *,
    devices: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    camera_devices: Sequence[str],
    robot_device: str,
) -> dict[str, Any]:
    """Extract the physical capture identity from a composed launch config."""
    validate_capture_device_identity(
        source_profile,
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    if isinstance(devices, Mapping):
        devices = devices.get("devices", ())
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise ObservationContractError("composed devices must be an array")
    configured: dict[str, Mapping[str, Any]] = {}
    for device in devices:
        if not isinstance(device, Mapping):
            raise ObservationContractError("each composed device must be an object")
        device_class = _required_text(device.get("class"), field="composed device class")
        device_class_id = _device_class_id(device.get("device_class_id"), field="composed device id")
        key = f"{device_class}_{device_class_id}"
        if key in configured:
            raise ObservationContractError(f"duplicate composed capture device {key!r}")
        configured[key] = device

    contract = get_source_profile_contract(source_profile)
    cameras: dict[str, dict[str, Any]] = {}
    for role, key in zip(("left_wrist_rgb", "right_wrist_rgb"), camera_devices):
        try:
            device = configured[key]
        except KeyError as exc:
            raise ObservationContractError(
                f"composed devices are missing reviewed capture device {key!r}"
            ) from exc
        state_feedback = device.get("state_feedback")
        if not isinstance(state_feedback, Mapping):
            raise ObservationContractError(f"composed capture device {key!r} lacks state_feedback")
        image_format = state_feedback.get("format")
        if not isinstance(image_format, Mapping):
            raise ObservationContractError(f"composed capture device {key!r} lacks format")
        cameras[role] = {
            "device_key": key,
            "class": _required_text(device.get("class"), field="composed camera class"),
            "device_class_id": _device_class_id(device.get("device_class_id"), field="composed camera id"),
            "format_shape": list(
                _shape(image_format.get("shape"), field=f"composed camera {key} shape")
            ),
            "format_dtype": _dtype(
                image_format.get("dtype"),
                field=f"composed camera {key} dtype",
            ),
            "format_type": _required_text(
                image_format.get("type"),
                field=f"composed camera {key} format type",
            ),
            "rotation_degrees": _camera_rotation_from_device(device),
        }
    robot_key = robot_device
    try:
        robot = configured[robot_key]
    except KeyError as exc:
        raise ObservationContractError(
            f"composed devices are missing reviewed capture robot {robot_key!r}"
        ) from exc
    robot_feedback = robot.get("state_feedback")
    if not isinstance(robot_feedback, Mapping):
        raise ObservationContractError(
            f"composed capture robot {robot_key!r} lacks state_feedback"
        )
    robot_format = robot_feedback.get("format")
    if not isinstance(robot_format, Mapping):
        raise ObservationContractError(
            f"composed capture robot {robot_key!r} lacks format"
        )
    descriptors = {
        "schema": CAPTURE_DEVICE_DESCRIPTOR_SCHEMA,
        "cameras": cameras,
        "robot": {
            "device_key": robot_key,
            "class": _required_text(robot.get("class"), field="composed robot class"),
            "device_class_id": _device_class_id(robot.get("device_class_id"), field="composed robot id"),
            "robot_serial_numbers": _normalized_robot_serials(
                robot.get("robot_sn"),
                field="composed robot robot_sn",
            ),
            "state_format_shape": list(
                _shape(
                    robot_format.get("shape"),
                    field=f"composed robot {robot_key} shape",
                )
            ),
            "state_format_dtype": _dtype(
                robot_format.get("dtype"),
                field=f"composed robot {robot_key} dtype",
            ),
            "state_format_type": _format_type_sequence(
                robot_format.get("type"),
                field=f"composed robot {robot_key} format type",
            ),
        },
    }
    return _capture_device_descriptors(contract, descriptors)


def _calibration_digests(
    contract: SourceProfileContract,
    value: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ObservationContractError("source profile calibration_digests must be an object")
    normalized = {
        _required_text(label, field="source calibration label"): _required_text(
            digest, field="source calibration digest"
        ).lower()
        for label, digest in value.items()
    }
    expected = set(contract.required_calibration_labels)
    if set(normalized) != expected:
        raise ObservationContractError(
            "source profile calibration_digests must contain exactly "
            + ", ".join(contract.required_calibration_labels)
        )
    for label, digest in normalized.items():
        if _SHA256_RE.fullmatch(digest) is None:
            raise ObservationContractError(
                f"source profile calibration digest for {label!r} must be SHA-256"
            )
    return {label: normalized[label] for label in contract.required_calibration_labels}


def source_profile_provenance(
    source_profile: str,
    *,
    calibration_digests: Mapping[str, Any],
    capture_device_descriptors: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a normalized immutable capture provenance record."""
    contract = get_source_profile_contract(source_profile)
    descriptors = _capture_device_descriptors(contract, capture_device_descriptors)
    keys = {
        **{role: value["device_key"] for role, value in descriptors["cameras"].items()},
        "rizon_dual_arm_state": descriptors["robot"]["device_key"],
    }
    return {
        "schema": SOURCE_PROFILE_PROVENANCE_SCHEMA,
        "source_profile": contract.source_profile,
        "camera_rig_id": contract.camera_rig_id,
        "capture_pipeline_id": contract.capture_pipeline_id,
        "capture_device_keys": keys,
        "capture_device_descriptors": descriptors,
        "robot_rig_id": contract.robot_rig_id,
        "robot_state_layout": RIZON_DUAL_ARM_STATE_LAYOUT,
        "action_frame_id": contract.action_frame_id,
        "action_units": contract.action_units,
        "calibration_digests": _calibration_digests(contract, calibration_digests),
    }


def validate_capture_device_identity(
    source_profile: str,
    *,
    camera_devices: Sequence[str],
    robot_device: str,
) -> None:
    """Reject a source-profile label that does not match its capture devices."""
    if isinstance(camera_devices, (str, bytes)) or len(camera_devices) != 2:
        raise ObservationContractError("reviewed capture requires exactly two camera devices")
    actual = {
        "left_wrist_rgb": _required_text(
            camera_devices[0], field="reviewed capture left camera device"
        ),
        "right_wrist_rgb": _required_text(
            camera_devices[1], field="reviewed capture right camera device"
        ),
        "rizon_dual_arm_state": _required_text(
            robot_device, field="reviewed capture robot device"
        ),
    }
    contract = get_source_profile_contract(source_profile)
    expected = dict(contract.capture_device_keys)
    if contract.flexible_device_keys:
        if len(set(actual.values())) != 3:
            raise ObservationContractError("capture roles must use distinct devices")
        for role, key in actual.items():
            cls, sep, index = key.rpartition("_")
            if not sep or not index.isdecimal() or cls != expected[role].rsplit("_", 1)[0]:
                raise ObservationContractError(f"invalid capture device for role {role}: {key}")
        return
    if actual != expected:
        raise ObservationContractError(
            f"capture devices do not match reviewed source profile {source_profile!r}"
        )


def source_profile_provenance_from_files(
    source_profile: str,
    *,
    calibration_files: Mapping[str, str | Path],
    capture_device_descriptors: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the exact local calibration material used by one capture run."""
    contract = get_source_profile_contract(source_profile)
    if not isinstance(calibration_files, Mapping):
        raise ObservationContractError("source profile calibration_files must be an object")
    paths = {
        _required_text(label, field="source calibration label"): Path(value).expanduser()
        for label, value in calibration_files.items()
    }
    expected = set(contract.required_calibration_labels)
    if set(paths) != expected:
        raise ObservationContractError(
            "source profile calibration_files must contain exactly "
            + ", ".join(contract.required_calibration_labels)
        )
    digests: dict[str, str] = {}
    for label in contract.required_calibration_labels:
        path = paths[label]
        if not path.is_file():
            raise ObservationContractError(
                f"source calibration file for {label!r} is not a file: {path}"
            )
        digests[label] = _sha256_file(path)
    return source_profile_provenance(
        contract.source_profile,
        calibration_digests=digests,
        capture_device_descriptors=capture_device_descriptors,
    )


def reviewed_calibration_files(repo_root: Path | str | None = None) -> dict[str, Path]:
    """Resolve the repository-owned calibration bundle used by all profiles."""
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return {label: root / relative for label, relative in _REVIEWED_CALIBRATION_PATHS.items()}


def verify_source_profile_calibrations(
    provenance: Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
) -> None:
    """Ensure a recording used the same reviewed calibration bytes as this run."""
    source_profile = _source_profile(provenance.get("source_profile"))
    normalized = parse_source_profile_provenance(
        provenance,
        source_profile=source_profile,
    )
    expected = source_profile_provenance_from_files(
        source_profile,
        calibration_files=reviewed_calibration_files(repo_root),
        capture_device_descriptors=normalized["capture_device_descriptors"],
    )
    if normalized["calibration_digests"] != expected["calibration_digests"]:
        raise ObservationContractError(
            "recorder calibration digest does not match this reviewed repository bundle"
        )


def parse_source_profile_provenance(
    value: Any,
    *,
    source_profile: str,
) -> dict[str, Any]:
    """Validate a recorder provenance record against its registered source."""
    if not isinstance(value, Mapping):
        raise ObservationContractError("manifest.source_provenance must be an object")
    expected = source_profile_provenance(
        source_profile,
        calibration_digests=value.get("calibration_digests", {}),
        capture_device_descriptors=value.get("capture_device_descriptors"),
    )
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ObservationContractError(
                f"manifest.source_provenance.{field} does not match reviewed source "
                f"profile {source_profile!r}"
            )
    if set(value) != set(expected):
        raise ObservationContractError(
            "manifest.source_provenance has unsupported or missing fields"
        )
    return expected


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return a stable digest for the finalized recorder manifest."""
    if not isinstance(manifest, Mapping):
        raise ObservationContractError("recorder manifest must be an object")
    return _digest({str(key): value for key, value in manifest.items()})


def _shape(value: Any, *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObservationContractError(f"{field} must be an array of positive integers")
    result: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool):
            raise ObservationContractError(f"{field} must be an array of positive integers")
        try:
            parsed = int(dimension)
        except (TypeError, ValueError) as exc:
            raise ObservationContractError(
                f"{field} must be an array of positive integers"
            ) from exc
        if parsed < 1:
            raise ObservationContractError(f"{field} must be an array of positive integers")
        result.append(parsed)
    if not result:
        raise ObservationContractError(f"{field} must not be empty")
    return tuple(result)


def _dtype(value: Any, *, field: str) -> str:
    try:
        return np.dtype(str(value)).name
    except TypeError as exc:
        raise ObservationContractError(f"{field} is not a valid NumPy dtype") from exc


def _role_kind(role: str) -> str:
    if role in {"left_wrist_rgb", "right_wrist_rgb"}:
        return "camera"
    if role == "rizon_dual_arm_state":
        return "robot"
    raise ObservationContractError(f"unsupported recorder role {role!r}")


@dataclass(frozen=True)
class RecorderDeviceRole:
    """One recorder array with an explicit physical role."""

    role: str
    device_key: str
    dtype: str | None
    shape: tuple[int, ...] | None
    state_layout: str | None = None

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"device_key": self.device_key}
        if self.dtype is not None:
            record["dtype"] = self.dtype
        if self.shape is not None:
            record["shape"] = list(self.shape)
        if self.state_layout is not None:
            record["state_layout"] = self.state_layout
        return record


@dataclass(frozen=True)
class RecorderObservationContract:
    """A parsed recorder manifest, including its semantic device roles."""

    source_profile: str
    roles: Mapping[str, RecorderDeviceRole]
    manifest_digest: str
    source_provenance: Mapping[str, Any] | None = None
    source_provenance_digest: str | None = None
    legacy_identity: bool = False


@dataclass(frozen=True)
class ObservationBinding:
    """A trusted model-key to recorder-key binding for one evaluator."""

    config_name: str
    source_profile: str
    source_provenance_digest: str
    recorder_manifest_digest: str
    recording_artifact_digest: str
    action_calibration_id: str
    action_calibration_file_digests: Mapping[str, str]
    action_calibration_digest: str
    target_device_roles: Mapping[str, str]
    target_to_source: Mapping[str, str]
    digest: str
    legacy_identity: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "source_profile": self.source_profile,
            "source_provenance_digest": self.source_provenance_digest,
            "recorder_manifest_digest": self.recorder_manifest_digest,
            "recording_artifact_digest": self.recording_artifact_digest,
            "action_calibration_id": self.action_calibration_id,
            "action_calibration_file_digests": dict(
                self.action_calibration_file_digests
            ),
            "action_calibration_digest": self.action_calibration_digest,
            "target_device_roles": dict(self.target_device_roles),
            "target_to_source": dict(self.target_to_source),
            "legacy_identity": self.legacy_identity,
            "digest": self.digest,
        }


def binding_policy_devices(binding: ObservationBinding) -> tuple[list[str], str]:
    """Return target policy device keys in the binding's canonical role order."""
    targets = _target_device_roles(binding.target_device_roles)
    devices_by_role = {role: target for target, role in targets.items()}
    return (
        [
            devices_by_role["left_wrist_rgb"],
            devices_by_role["right_wrist_rgb"],
        ],
        devices_by_role["rizon_dual_arm_state"],
    )


def validate_binding_policy_devices(
    binding: ObservationBinding,
    *,
    camera_devices: Sequence[str],
    robot_device: str,
    config_name: str,
) -> tuple[list[str], str]:
    """Reject a recomposed evaluator config whose device order differs from binding."""
    expected_cameras, expected_robot = binding_policy_devices(binding)
    if isinstance(camera_devices, (str, bytes)) or len(camera_devices) != 2:
        raise ObservationContractError(
            f"{config_name}.policy must declare exactly two camera devices"
        )
    actual_cameras = [
        _required_text(value, field=f"{config_name}.policy camera device")
        for value in camera_devices
    ]
    actual_robot = _required_text(
        robot_device,
        field=f"{config_name}.policy robot device",
    )
    if actual_cameras != expected_cameras or actual_robot != expected_robot:
        raise ObservationContractError(
            f"{config_name}.policy device order does not match its trusted observation binding"
        )
    return expected_cameras, expected_robot


def device_roles_from_arrays(
    *,
    camera_devices: Sequence[str],
    robot_device: str,
    arrays: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a recorder manifest role section from one raw observation sample."""
    if isinstance(camera_devices, (str, bytes)) or len(camera_devices) != 2:
        raise ObservationContractError("recorder requires exactly two camera devices")
    device_names = [
        _required_text(name, field="camera device") for name in camera_devices
    ]
    robot_name = _required_text(robot_device, field="robot device")
    if len({*device_names, robot_name}) != 3:
        raise ObservationContractError("recorder role device keys must be unique")

    role_to_device = {
        "left_wrist_rgb": device_names[0],
        "right_wrist_rgb": device_names[1],
        "rizon_dual_arm_state": robot_name,
    }
    result: dict[str, dict[str, Any]] = {}
    for role, device_key in role_to_device.items():
        if device_key not in arrays:
            raise ObservationContractError(
                f"recorder observation is missing {role} device {device_key!r}"
            )
        array = np.asarray(arrays[device_key])
        record: dict[str, Any] = {
            "device_key": device_key,
            "dtype": array.dtype.name,
            "shape": list(array.shape),
        }
        if role == "rizon_dual_arm_state":
            record["state_layout"] = RIZON_DUAL_ARM_STATE_LAYOUT
        result[role] = record
    _parse_device_roles(result)
    return result


def _parse_device_roles(value: Any) -> dict[str, RecorderDeviceRole]:
    if not isinstance(value, Mapping):
        raise ObservationContractError("manifest.device_roles must be an object")
    names = {str(role) for role in value}
    expected = set(CANONICAL_OBSERVATION_ROLES)
    if names != expected:
        raise ObservationContractError(
            "manifest.device_roles must contain exactly "
            + ", ".join(CANONICAL_OBSERVATION_ROLES)
        )

    result: dict[str, RecorderDeviceRole] = {}
    device_keys: set[str] = set()
    for role in CANONICAL_OBSERVATION_ROLES:
        raw = value.get(role)
        if not isinstance(raw, Mapping):
            raise ObservationContractError(f"manifest.device_roles.{role} must be an object")
        device_key = _required_text(
            raw.get("device_key"), field=f"manifest.device_roles.{role}.device_key"
        )
        if device_key in device_keys:
            raise ObservationContractError("manifest.device_roles device_key values must be unique")
        device_keys.add(device_key)
        dtype = _dtype(raw.get("dtype"), field=f"manifest.device_roles.{role}.dtype")
        shape = _shape(raw.get("shape"), field=f"manifest.device_roles.{role}.shape")
        kind = _role_kind(role)
        if kind == "camera":
            if dtype != "uint8":
                raise ObservationContractError(
                    f"manifest.device_roles.{role}.dtype must be uint8"
                )
            if len(shape) < 3 or shape[-1] not in {1, 3, 4}:
                raise ObservationContractError(
                    f"manifest.device_roles.{role}.shape must describe an HWC image"
                )
            state_layout = None
        else:
            if not np.issubdtype(np.dtype(dtype), np.floating):
                raise ObservationContractError(
                    f"manifest.device_roles.{role}.dtype must be floating-point"
                )
            if int(np.prod(shape)) < 44:
                raise ObservationContractError(
                    f"manifest.device_roles.{role}.shape must contain at least 44 values"
                )
            state_layout = _required_text(
                raw.get("state_layout"),
                field=f"manifest.device_roles.{role}.state_layout",
            )
            if state_layout != RIZON_DUAL_ARM_STATE_LAYOUT:
                raise ObservationContractError(
                    f"manifest.device_roles.{role}.state_layout must be "
                    f"{RIZON_DUAL_ARM_STATE_LAYOUT!r}"
                )
        result[role] = RecorderDeviceRole(
            role=role,
            device_key=device_key,
            dtype=dtype,
            shape=shape,
            state_layout=state_layout,
        )
    return result


def _legacy_identity_contract(manifest: Mapping[str, Any]) -> RecorderObservationContract:
    raw_cameras = manifest.get("camera_devices")
    if isinstance(raw_cameras, (str, bytes)) or not isinstance(raw_cameras, Sequence):
        raise ObservationContractError(
            "legacy recorder manifests require camera_devices for strict identity binding"
        )
    if len(raw_cameras) != 2:
        raise ObservationContractError(
            "legacy recorder manifests require exactly two camera_devices"
        )
    camera_devices = [
        _required_text(value, field="legacy manifest camera_devices")
        for value in raw_cameras
    ]
    robot_device = _required_text(
        manifest.get("robot_device"), field="legacy manifest robot_device"
    )
    if len({*camera_devices, robot_device}) != 3:
        raise ObservationContractError("legacy recorder device keys must be unique")
    roles = {
        "left_wrist_rgb": RecorderDeviceRole(
            role="left_wrist_rgb",
            device_key=camera_devices[0],
            dtype=None,
            shape=None,
        ),
        "right_wrist_rgb": RecorderDeviceRole(
            role="right_wrist_rgb",
            device_key=camera_devices[1],
            dtype=None,
            shape=None,
        ),
        "rizon_dual_arm_state": RecorderDeviceRole(
            role="rizon_dual_arm_state",
            device_key=robot_device,
            dtype=None,
            shape=None,
        ),
    }
    return RecorderObservationContract(
        source_profile=LEGACY_IDENTITY_SOURCE_PROFILE,
        roles=roles,
        manifest_digest=manifest_digest(manifest),
        legacy_identity=True,
    )


def parse_recorder_observation_contract(
    manifest: Mapping[str, Any],
) -> RecorderObservationContract:
    """Parse a recorder manifest, allowing only strict identity legacy input."""
    if not isinstance(manifest, Mapping):
        raise ObservationContractError("recorder manifest must be an object")
    raw_roles = manifest.get("device_roles")
    if raw_roles is None:
        return _legacy_identity_contract(manifest)
    contract_name = _required_text(
        manifest.get("observation_contract"), field="manifest.observation_contract"
    )
    if contract_name != RECORDER_OBSERVATION_CONTRACT:
        raise ObservationContractError(
            "manifest.observation_contract must be "
            f"{RECORDER_OBSERVATION_CONTRACT!r}"
        )
    source_profile = _source_profile(manifest.get("source_profile"))
    source_provenance = parse_source_profile_provenance(
        manifest.get("source_provenance"),
        source_profile=source_profile,
    )
    raw_cameras = manifest.get("camera_devices")
    if isinstance(raw_cameras, (str, bytes)) or not isinstance(raw_cameras, Sequence):
        raise ObservationContractError("manifest.camera_devices must be an array")
    camera_devices = [
        _required_text(value, field="manifest camera device") for value in raw_cameras
    ]
    robot_device = _required_text(manifest.get("robot_device"), field="manifest.robot_device")
    validate_capture_device_identity(
        source_profile,
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    roles = _parse_device_roles(raw_roles)
    profile_contract = get_source_profile_contract(source_profile)
    for role, expected_shape in profile_contract.camera_shapes.items():
        if roles[role].shape != expected_shape:
            raise ObservationContractError(
                f"manifest.device_roles.{role}.shape must match reviewed source "
                f"profile {source_profile!r}: {list(expected_shape)}"
            )
    actual_role_keys = {role: roles[role].device_key for role in CANONICAL_OBSERVATION_ROLES}
    if actual_role_keys != source_provenance["capture_device_keys"]:
        raise ObservationContractError(
            "manifest.device_roles do not match reviewed source capture devices"
        )
    return RecorderObservationContract(
        source_profile=source_profile,
        roles=roles,
        manifest_digest=manifest_digest(manifest),
        source_provenance=source_provenance,
        source_provenance_digest=_digest(source_provenance),
    )


def _target_device_roles(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ObservationContractError("target device roles must be an object")
    result = {
        _required_text(target, field="target device key"): _required_text(
            role, field="target device role"
        )
        for target, role in value.items()
    }
    if set(result.values()) != set(CANONICAL_OBSERVATION_ROLES) or len(result) != 3:
        raise ObservationContractError(
            "target device roles must map exactly one target key to each canonical role"
        )
    if len(set(result)) != len(result):
        raise ObservationContractError("target device keys must be unique")
    return result


def create_observation_binding(
    *,
    manifest: Mapping[str, Any],
    config_name: str,
    target_device_roles: Mapping[str, str],
    allowed_source_profiles: Sequence[str],
    action_calibration_id: str,
    action_calibration_file_digests: Mapping[str, str],
    action_calibration_digest: str,
    recording_root: Path | str | None = None,
) -> ObservationBinding:
    """Bind one reviewed policy config to a finalized recorder manifest."""
    contract = parse_recorder_observation_contract(manifest)
    targets = _target_device_roles(target_device_roles)
    allowed_profiles = {_source_profile(value) for value in allowed_source_profiles}
    if not allowed_profiles:
        raise ObservationContractError("model config must declare allowed source profiles")
    if not contract.legacy_identity and contract.source_profile not in allowed_profiles:
        raise ObservationContractError(
            f"recorder source_profile {contract.source_profile!r} is not approved for "
            f"config {config_name!r}"
        )

    target_to_source = {
        target: contract.roles[role].device_key for target, role in targets.items()
    }
    if contract.legacy_identity and any(
        target != source for target, source in target_to_source.items()
    ):
        raise ObservationContractError(
            "legacy recorder manifests allow only strict same-name identity bindings"
        )
    if contract.legacy_identity:
        provenance_digest = _digest({"legacy_identity": True})
        artifact_digest = _digest({"legacy_manifest": contract.manifest_digest})
    else:
        if recording_root is None:
            raise ObservationContractError(
                "a reviewed recorder binding requires recording_root for artifact hashing"
            )
        assert contract.source_provenance_digest is not None
        provenance_digest = contract.source_provenance_digest
        artifact_digest = recording_artifact_digest(recording_root)

    calibration_id = _source_profile(
        action_calibration_id, field="action_calibration_id"
    )
    if not isinstance(action_calibration_file_digests, Mapping):
        raise ObservationContractError(
            "action_calibration_file_digests must be an object"
        )
    calibration_file_digests = {
        _required_text(key, field="action calibration file key"): _required_text(
            value, field="action calibration file digest"
        ).lower()
        for key, value in action_calibration_file_digests.items()
    }
    for key, digest in calibration_file_digests.items():
        if _SHA256_RE.fullmatch(digest) is None:
            raise ObservationContractError(
                f"action calibration file digest for {key!r} must be SHA-256"
            )
    calibration_digest = _required_text(
        action_calibration_digest, field="action_calibration_digest"
    ).lower()
    if _SHA256_RE.fullmatch(calibration_digest) is None:
        raise ObservationContractError(
            "action_calibration_digest must be a SHA-256 digest"
        )

    payload = {
        "config_name": _required_text(config_name, field="config_name"),
        "source_profile": contract.source_profile,
        "source_provenance_digest": provenance_digest,
        "recorder_manifest_digest": contract.manifest_digest,
        "recording_artifact_digest": artifact_digest,
        "action_calibration_id": calibration_id,
        "action_calibration_file_digests": calibration_file_digests,
        "action_calibration_digest": calibration_digest,
        "target_device_roles": targets,
        "target_to_source": target_to_source,
        "legacy_identity": contract.legacy_identity,
    }
    return ObservationBinding(**payload, digest=_digest(payload))


def array_keys_from_index_row(
    row: Mapping[str, Any],
    *,
    kind: str,
) -> dict[str, str]:
    """Parse one recorder index mapping without accepting ambiguous aliases.

    The evaluator and preflight both call this function. A repeated inference
    name therefore cannot be first-wins in one component and last-wins in
    another.
    """
    if kind not in {"frame", "inference"}:
        raise ObservationContractError(f"unsupported recorder index kind {kind!r}")
    raw_mapping = row.get("device_array_keys" if kind == "frame" else "keys")
    if raw_mapping is None:
        return {}

    if kind == "frame":
        if not isinstance(raw_mapping, Mapping):
            raise ObservationContractError("frame record device_array_keys must be an object")
        pairs = raw_mapping.items()
    else:
        if isinstance(raw_mapping, (str, bytes)) or not isinstance(raw_mapping, Sequence):
            raise ObservationContractError("inference record keys must be an array")
        pairs = []
        for entry in raw_mapping:
            if not isinstance(entry, Mapping):
                raise ObservationContractError("inference record key entries must be objects")
            pairs.append((entry.get("name"), entry.get("array_key")))

    result: dict[str, str] = {}
    used_array_keys: set[str] = set()
    for raw_name, raw_array_key in pairs:
        name = _required_text(raw_name, field=f"{kind} record device name")
        array_key = _required_text(raw_array_key, field=f"{kind} record array_key")
        if name in result:
            raise ObservationContractError(
                f"{kind} record declares recorder device {name!r} more than once"
            )
        if array_key in used_array_keys:
            raise ObservationContractError(
                f"{kind} record maps multiple recorder devices to array {array_key!r}"
            )
        result[name] = array_key
        used_array_keys.add(array_key)
    return result


def _array_key_for_device(row: Mapping[str, Any], device_key: str, *, kind: str) -> str:
    mapping = array_keys_from_index_row(row, kind=kind)
    if mapping:
        try:
            return mapping[device_key]
        except KeyError as exc:
            raise ObservationContractError(
                f"{kind} record does not declare recorder device {device_key!r}"
            ) from exc
    return device_key


def _matches_role_array(
    array: np.ndarray,
    role: RecorderDeviceRole,
    *,
    kind: str,
) -> bool:
    if role.dtype is None or role.shape is None:
        return True
    if np.dtype(array.dtype).name != role.dtype:
        return False
    if kind == "frame":
        return tuple(array.shape) == role.shape
    # Exact policy inputs may have an extra chronological history dimension.
    return tuple(array.shape[-len(role.shape) :]) == role.shape


def recording_artifact_digest(root: Path | str) -> str:
    """Hash recorder indexes and every indexed NPZ to detect postflight edits."""
    recording_root = Path(root).expanduser().resolve()
    inventory: dict[str, Any] = {"schema": "superoffline.recording-artifacts.v1"}
    for label, kind in (("frames.jsonl", "frame"), ("inference.jsonl", "inference")):
        index_path = recording_root / label
        try:
            index_bytes = index_path.read_bytes()
        except OSError as exc:
            raise ObservationContractError(
                f"could not read recorder {kind} index {index_path}"
            ) from exc
        artifacts: list[dict[str, str]] = []
        for row_index, line in enumerate(index_bytes.splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ObservationContractError(
                    f"could not parse recorder {kind} index {index_path}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ObservationContractError(
                    f"{kind} index row {row_index} is not an object"
                )
            raw_file = row.get("file")
            if not isinstance(raw_file, str) or not raw_file:
                raise ObservationContractError(
                    f"{kind} index row {row_index} has no artifact file"
                )
            artifact = (recording_root / raw_file).resolve()
            if not artifact.is_relative_to(recording_root) or not artifact.is_file():
                raise ObservationContractError(
                    f"{kind} index row {row_index} artifact is missing or outside recorder root"
                )
            artifacts.append(
                {
                    "file": raw_file,
                    "sha256": _sha256_file(artifact),
                }
            )
        inventory[label] = {
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
            "artifacts": artifacts,
        }
    return _digest(inventory)


def validate_recording_artifacts(
    root: Path,
    contract: RecorderObservationContract,
) -> None:
    """Fail before runtime doctor when any recorder artifact breaks its roles."""
    for label, kind in (("frames.jsonl", "frame"), ("inference.jsonl", "inference")):
        index_path = root / label
        try:
            rows = [
                json.loads(line)
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ObservationContractError(
                f"could not read recorder {kind} index {index_path}"
            ) from exc
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ObservationContractError(f"{kind} index row {row_index} is not an object")
            raw_file = row.get("file")
            if not isinstance(raw_file, str) or not raw_file:
                raise ObservationContractError(f"{kind} index row {row_index} has no artifact file")
            artifact = (root / raw_file).resolve()
            if not artifact.is_relative_to(root.resolve()) or not artifact.is_file():
                raise ObservationContractError(
                    f"{kind} index row {row_index} artifact is missing or outside recorder root"
                )
            try:
                with np.load(artifact, allow_pickle=False) as data:
                    role_array_keys = {
                        role.role: _array_key_for_device(
                            row, role.device_key, kind=kind
                        )
                        for role in contract.roles.values()
                    }
                    if len(set(role_array_keys.values())) != len(role_array_keys):
                        raise ObservationContractError(
                            f"{kind} artifact {raw_file} maps multiple recorder roles to one array"
                        )
                    for role in contract.roles.values():
                        array_key = role_array_keys[role.role]
                        if array_key not in data.files:
                            raise ObservationContractError(
                                f"{kind} artifact {raw_file} is missing {role.device_key!r}"
                            )
                        if not _matches_role_array(data[array_key], role, kind=kind):
                            raise ObservationContractError(
                                f"{kind} artifact {raw_file} does not match role {role.role!r} "
                                "dtype/shape contract"
                            )
            except (OSError, ValueError) as exc:
                if isinstance(exc, ObservationContractError):
                    raise
                raise ObservationContractError(
                    f"could not read {kind} artifact {raw_file}"
                ) from exc


def write_observation_bindings(path: Path, bindings: Sequence[ObservationBinding]) -> None:
    """Write the executor-owned binding bundle consumed by evaluator processes."""
    if not bindings:
        raise ObservationContractError("at least one observation binding is required")
    payload = {
        "schema": OBSERVATION_BINDINGS_SCHEMA,
        "bindings": [binding.as_dict() for binding in bindings],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_observation_binding(
    path: Path | str,
    *,
    binding_index: int,
    config_name: str,
    observation_dir: Path | str,
) -> ObservationBinding:
    """Read and verify an executor-generated binding against its recorder.

    A binding digest is an integrity checksum, not an authorization signature.
    Re-resolve the selected config through the registry so a hand-authored
    bundle cannot remap one model to a recorder source or action calibration
    that the config never reviewed.
    """
    binding_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservationContractError(
            f"could not read observation binding bundle: {binding_path}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != OBSERVATION_BINDINGS_SCHEMA:
        raise ObservationContractError("observation binding bundle has an unsupported schema")
    raw_bindings = payload.get("bindings")
    if isinstance(raw_bindings, (str, bytes)) or not isinstance(raw_bindings, Sequence):
        raise ObservationContractError("observation binding bundle must contain bindings")
    if isinstance(binding_index, bool) or binding_index < 0 or binding_index >= len(raw_bindings):
        raise ObservationContractError("observation binding index is invalid")
    raw = raw_bindings[binding_index]
    if not isinstance(raw, Mapping):
        raise ObservationContractError("observation binding must be an object")
    target_roles = _target_device_roles(raw.get("target_device_roles", {}))
    target_to_source = raw.get("target_to_source")
    if not isinstance(target_to_source, Mapping):
        raise ObservationContractError("observation binding target_to_source must be an object")
    normalized_sources = {
        _required_text(target, field="binding target device"): _required_text(
            source, field="binding recorder device"
        )
        for target, source in target_to_source.items()
    }
    if set(normalized_sources) != set(target_roles):
        raise ObservationContractError("observation binding target keys do not match target roles")
    config = _required_text(raw.get("config_name"), field="binding config_name")
    if config != config_name:
        raise ObservationContractError(
            f"observation binding config {config!r} does not match {config_name!r}"
        )
    try:
        # Keep this import local: the registry imports this module at startup.
        from utils.superoffline_registry import get_superoffline_model_contract

        registered_model = get_superoffline_model_contract(config)
    except (ImportError, ValueError) as exc:
        raise ObservationContractError(
            f"observation binding config {config!r} has no registered SuperOffline contract"
        ) from exc
    expected_target_roles = _target_device_roles(registered_model.target_device_roles)
    if target_roles != expected_target_roles:
        raise ObservationContractError(
            "observation binding target device roles do not match the registered model"
        )
    source_profile = _source_profile(raw.get("source_profile"))
    if source_profile not in registered_model.source_profiles:
        raise ObservationContractError(
            f"observation binding source profile {source_profile!r} is not approved for "
            f"config {config!r}"
        )
    source_provenance_hash = _required_text(
        raw.get("source_provenance_digest"),
        field="binding source_provenance_digest",
    )
    manifest_hash = _required_text(
        raw.get("recorder_manifest_digest"), field="binding recorder_manifest_digest"
    )
    artifact_hash = _required_text(
        raw.get("recording_artifact_digest"),
        field="binding recording_artifact_digest",
    )
    calibration_id = _source_profile(
        raw.get("action_calibration_id"), field="binding action_calibration_id"
    )
    raw_calibration_file_digests = raw.get("action_calibration_file_digests")
    if not isinstance(raw_calibration_file_digests, Mapping):
        raise ObservationContractError(
            "binding action_calibration_file_digests must be an object"
        )
    calibration_file_digests = {
        _required_text(key, field="binding action calibration file key"): _required_text(
            value, field="binding action calibration file digest"
        ).lower()
        for key, value in raw_calibration_file_digests.items()
    }
    for key, value in calibration_file_digests.items():
        if _SHA256_RE.fullmatch(value) is None:
            raise ObservationContractError(
                f"binding action calibration file digest for {key!r} must be SHA-256"
            )
    calibration_digest = _required_text(
        raw.get("action_calibration_digest"),
        field="binding action_calibration_digest",
    ).lower()
    if _SHA256_RE.fullmatch(calibration_digest) is None:
        raise ObservationContractError(
            "binding action_calibration_digest must be a SHA-256 digest"
        )
    try:
        from utils.superoffline_action_calibration import reviewed_action_calibration

        expected_action_calibration = reviewed_action_calibration(
            registered_model.action_calibration
        )
    except ValueError as exc:
        raise ObservationContractError(
            f"could not verify registered action calibration for {config!r}"
        ) from exc
    if (
        calibration_id != expected_action_calibration.calibration_id
        or calibration_file_digests
        != dict(expected_action_calibration.file_digests)
        or calibration_digest != expected_action_calibration.digest
    ):
        raise ObservationContractError(
            "observation binding action calibration does not match the registered model"
        )
    legacy_identity = bool(raw.get("legacy_identity", False))
    signed_payload = {
        "config_name": config,
        "source_profile": source_profile,
        "source_provenance_digest": source_provenance_hash,
        "recorder_manifest_digest": manifest_hash,
        "recording_artifact_digest": artifact_hash,
        "action_calibration_id": calibration_id,
        "action_calibration_file_digests": calibration_file_digests,
        "action_calibration_digest": calibration_digest,
        "target_device_roles": target_roles,
        "target_to_source": normalized_sources,
        "legacy_identity": legacy_identity,
    }
    digest = _required_text(raw.get("digest"), field="binding digest")
    if digest != _digest(signed_payload):
        raise ObservationContractError("observation binding digest does not match its contents")

    root = Path(observation_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservationContractError("could not read recorder manifest for observation binding") from exc
    contract = parse_recorder_observation_contract(manifest)
    if contract.manifest_digest != manifest_hash:
        raise ObservationContractError("recorder manifest changed after observation binding was created")
    if contract.legacy_identity:
        raise ObservationContractError(
            "legacy recorder manifests are not accepted by trusted observation bindings"
        )
    if contract.source_provenance_digest != source_provenance_hash:
        raise ObservationContractError(
            "recorder source provenance changed after observation binding was created"
        )
    assert contract.source_provenance is not None
    verify_source_profile_calibrations(contract.source_provenance)
    if recording_artifact_digest(root) != artifact_hash:
        raise ObservationContractError(
            "recorder artifacts changed after observation binding was created"
        )
    expected_sources = {
        target: contract.roles[role].device_key for target, role in target_roles.items()
    }
    if normalized_sources != expected_sources:
        raise ObservationContractError("observation binding does not match recorder role keys")
    if contract.source_profile != source_profile or contract.legacy_identity != legacy_identity:
        raise ObservationContractError("observation binding source contract does not match recorder")
    return ObservationBinding(**signed_payload, digest=digest)


__all__ = [
    "CAPTURE_DEVICE_DESCRIPTOR_SCHEMA",
    "CANONICAL_OBSERVATION_ROLES",
    "FISHEYE_LINGBOTVA_SOURCE_PROFILE",
    "FISHEYE_OPENPI_RTC_SOURCE_PROFILE",
    "FISHEYE_OPENPI_SOURCE_PROFILE",
    "IPHONE_SOURCE_PROFILE",
    "LEGACY_IDENTITY_SOURCE_PROFILE",
    "LEGACY_IPHONE_SOURCE_PROFILE",
    "OBSERVATION_BINDINGS_SCHEMA",
    "ObservationBinding",
    "ObservationContractError",
    "RECORDER_OBSERVATION_CONTRACT",
    "RIZON_DUAL_ARM_STATE_LAYOUT",
    "SOURCE_PROFILE_CONTRACTS",
    "SOURCE_PROFILE_PROVENANCE_SCHEMA",
    "SourceProfileContract",
    "RecorderDeviceRole",
    "RecorderObservationContract",
    "array_keys_from_index_row",
    "binding_policy_devices",
    "capture_device_descriptors_from_config",
    "create_observation_binding",
    "device_roles_from_arrays",
    "get_source_profile_contract",
    "manifest_digest",
    "parse_recorder_observation_contract",
    "parse_source_profile_provenance",
    "read_observation_binding",
    "recording_artifact_digest",
    "source_profile_provenance",
    "source_profile_provenance_from_files",
    "reviewed_calibration_files",
    "reviewed_capture_device_descriptors",
    "validate_capture_device_identity",
    "validate_binding_policy_devices",
    "validate_recording_artifacts",
    "verify_source_profile_calibrations",
    "write_observation_bindings",
]
