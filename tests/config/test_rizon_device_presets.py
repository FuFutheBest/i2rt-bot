"""Rizon device-preset cleanup contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils.hydra_config import compose_hydra_application_config


CANONICAL_RIZON_DEVICE_PRESETS = {
    "infer_rizon_fisheye_umi3_dual_fastwam.yaml",
    "infer_rizon_fisheye_umi3_dual_openpi.yaml",
    "infer_rizon_fisheye_umi3_dual_openpi_recording_head.yaml",
    "infer_rizon_fisheye_umi3_dual_openpi_yam_head.yaml",
    "rizon_dual_eef_abs_fisheye_webcam.yaml",
    "rizon_dual_eef_abs_iphone.yaml",
    "rizon_dual_eef_abs_iphone_umi3.yaml",
    "rizon_dual_eef_abs_opencv.yaml",
    "rizon_single_eef_abs_iphone.yaml",
    "rizon_single_eef_abs_iphone_umi.yaml",
}

RIZON_LAUNCH_PRESETS = {
    "policy_infer/diffusion/config_infer_rizon_eef_dual.yaml": "rizon_dual_eef_abs_opencv",
    "policy_infer/diffusion/config_infer_rizon_eef_dual_abs_mode_umi.yaml": "rizon_dual_eef_abs_opencv",
    "policy_infer/diffusion/config_infer_rizon_eef_single.yaml": "rizon_single_eef_abs_iphone",
    "policy_infer/diffusion/config_infer_iphone_umi_dual.yaml": "rizon_dual_eef_abs_iphone",
    "policy_infer/diffusion/config_infer_iphone_umi_single.yaml": "rizon_single_eef_abs_iphone_umi",
    "policy_infer/nmx_vla/config_infer_iphone_umi_single_nmxvla.yaml": "rizon_single_eef_abs_iphone_umi",
    "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva.yaml": "rizon_dual_eef_abs_fisheye_webcam",
    "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async.yaml": "rizon_dual_eef_abs_fisheye_webcam",
    "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async_fastwam.yaml": "rizon_dual_eef_abs_fisheye_webcam",
    "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async_fastwam_client.yaml": "rizon_dual_eef_abs_fisheye_webcam",
    "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_block_fastwam.yaml": "rizon_dual_eef_abs_fisheye_webcam",
    "policy_infer/lingbotva/config_infer_iphone_umi3_dual_lingbotva.yaml": "rizon_dual_eef_abs_iphone_umi3",
    "policy_infer/lingbotva/config_infer_iphone_umi_single_lingbotva.yaml": "rizon_single_eef_abs_iphone_umi",
    "policy_infer/lingbotva/config_infer_iphone_umi_single_lingbotva_async.yaml": "rizon_single_eef_abs_iphone_umi",
}

RIZON_DEVICE_CONTRACTS = [
    (
        "policy_infer/diffusion/config_infer_rizon_eef_dual",
        ["OpenCVCameraDevice", "OpenCVCameraDevice"],
        1,
        [44],
        [16],
    ),
    (
        "policy_infer/diffusion/config_infer_rizon_eef_single",
        ["IPhoneCameraDevice"],
        1,
        [22],
        [8],
    ),
    (
        "policy_infer/diffusion/config_infer_iphone_umi_dual",
        ["IPhoneCameraDevice", "IPhoneCameraDevice"],
        2,
        [44],
        [16],
    ),
    (
        "policy_infer/diffusion/config_infer_iphone_umi_single",
        ["IPhoneCameraDevice"],
        1,
        [22],
        [8],
    ),
    (
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
        ["WebCameraDevice", "WebCameraDevice"],
        2,
        [44],
        [16],
    ),
    (
        "policy_infer/lingbotva/config_infer_iphone_umi3_dual_lingbotva",
        ["IPhoneCameraDevice", "IPhoneCameraDevice"],
        2,
        [44],
        [16],
    ),
]


def _selected_devices_preset(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    defaults = data.get("defaults", [])
    assert isinstance(defaults, list)
    for default in defaults:
        if not isinstance(default, dict):
            continue
        for group_name, preset_name in default.items():
            normalized_group = str(group_name).lstrip("/").split("@", maxsplit=1)[0]
            if normalized_group in {"devices", "devices/infer"}:
                return str(preset_name)
    raise AssertionError(f"No devices preset in {path}")


@pytest.mark.config
def test_rizon_device_preset_inventory_is_canonical(repo_root: Path) -> None:
    """Remove unused inference leaves while retaining integration-owned variants."""
    device_root = repo_root / "configs" / "devices" / "infer"
    rizon_presets = {
        path.name
        for path in device_root.glob("*.yaml")
        if path.name.startswith("infer_rizon_")
        or path.name in CANONICAL_RIZON_DEVICE_PRESETS
    }

    assert rizon_presets == CANONICAL_RIZON_DEVICE_PRESETS


@pytest.mark.config
def test_rizon_launchers_select_canonical_device_presets(repo_root: Path) -> None:
    """Launch configs should not retain the removed infer_rizon leaf names."""
    config_root = repo_root / "configs"

    actual = {
        relative_path: _selected_devices_preset(config_root / relative_path)
        for relative_path in RIZON_LAUNCH_PRESETS
    }

    assert actual == RIZON_LAUNCH_PRESETS


@pytest.mark.config
@pytest.mark.parametrize(
    ("config_name", "camera_classes", "robot_id", "state_shape", "control_shape"),
    RIZON_DEVICE_CONTRACTS,
)
def test_rizon_device_preset_contracts(
    repo_root: Path,
    config_name: str,
    camera_classes: list[str],
    robot_id: int,
    state_shape: list[int],
    control_shape: list[int],
) -> None:
    """Renaming device groups must not alter their camera or action contracts."""
    config = compose_hydra_application_config(
        config_name,
        config_dir=repo_root / "configs",
        job_name="rizon_device_preset_contract",
    )
    devices = config["devices"]
    robot = next(device for device in devices if device["class"] == "RizonRobot")

    assert [device["class"] for device in devices[: len(camera_classes)]] == camera_classes
    assert robot["device_class_id"] == devices.index(robot)
    assert robot["state_feedback"]["format"]["shape"] == state_shape
    assert robot["control"]["format"]["shape"] == control_shape
