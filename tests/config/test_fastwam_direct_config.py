"""Direct FastWAM configuration contracts that do not touch hardware."""

from __future__ import annotations

from pathlib import Path

import pytest

from incubator.control_panel.runtime_config import compose_runtime_config
from utils.config_schema import load_yaml_config
from utils.runtime_role_contracts import required_runtime_roles

DIRECT_FASTWAM_CONFIGS = (
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
)

STANDALONE_FASTWAM_VARIANTS = (
    "config_infer_fisheye_umi3_dual_fastwam_memory.yaml",
    "config_infer_fisheye_umi3_dual_fasterwam.yaml",
)

CAMERA_DEVICE_PRESETS = (
    "configs/devices/infer/infer_rizon_fisheye_umi3_dual_fastwam.yaml",
    "configs/devices/infer/infer_rizon_fisheye_umi3_dual_openpi.yaml",
    "configs/devices/infer/infer_rizon_fisheye_umi3_dual_openpi_recording_head.yaml",
    "configs/devices/infer/infer_rizon_fisheye_umi3_dual_openpi_yam_head.yaml",
)

FASTWAM_ROLLOUT_OUTPUT_DIRS = {
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam": (
        "data/rollout_data/fastwam"
    ),
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc": (
        "data/rollout_data/fastwam_rtc"
    ),
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory": (
        "data/rollout_data/fastwam_memory"
    ),
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam": (
        "data/rollout_data/fasterwam"
    ),
}


@pytest.mark.config
@pytest.mark.parametrize("config_name", DIRECT_FASTWAM_CONFIGS)
def test_direct_fastwam_configs_do_not_require_recording_head(config_name: str) -> None:
    """Direct DeviceStarter uses the first wrist camera as the sync master."""
    config = compose_runtime_config(config_name)

    assert [device["class"] for device in config["devices"]] == [
        "WebCameraOpenPi",
        "WebCameraOpenPi",
        "RizonRobot",
    ]
    assert [
        device["usb_port_requirement"] for device in config["devices"][:2]
    ] == ["any", "any"]
    assert config["device_manager"]["master_device_class_id"] == 0
    assert config["device_manager"]["sync_mode"] == "camera_align"
    assert config["devices"][2]["state_feedback"]["format"]["shape"] == [44]
    assert "data_shape" not in config["devices"][2]
    assert "data_format" not in config["devices"][2]


def _device_group(config_path: Path) -> str:
    for default in load_yaml_config(config_path)["defaults"]:
        if not isinstance(default, dict):
            continue
        for group_name, preset_name in default.items():
            group = str(group_name).lstrip("/").split("@", maxsplit=1)[0]
            if group in {"devices", "devices/infer"}:
                return str(preset_name)
    raise AssertionError(f"No /devices default in {config_path}")


@pytest.mark.config
@pytest.mark.parametrize("filename", STANDALONE_FASTWAM_VARIANTS)
def test_fastwam_variants_are_standalone_rollout_configs(
    repo_root: Path,
    filename: str,
) -> None:
    config_path = repo_root / "configs/policy_infer/fastwam" / filename
    config = load_yaml_config(config_path)

    assert config["camera"].keys() >= {"left_capture_id", "right_capture_id"}
    assert _device_group(config_path) == "infer_rizon_fisheye_umi3_dual_fastwam"
    assert "config_infer_fisheye_umi3_dual_fastwam" not in config["defaults"]


@pytest.mark.config
@pytest.mark.parametrize("config_name", DIRECT_FASTWAM_CONFIGS)
def test_fastwam_variants_share_git_ignored_local_model_cache(config_name: str) -> None:
    config = compose_runtime_config(config_name)

    assert (
        config["policy"]["params"]["model_base_path"]
        == ".superinference/local/model-cache"
    )
    assert "text_encoder_path" not in config["policy"]["params"]
    assert not str(config["policy"]["params"].get("model_id", "")).startswith("/home/")


@pytest.mark.config
@pytest.mark.parametrize("config_name", DIRECT_FASTWAM_CONFIGS)
def test_fastwam_configs_copy_the_openpi_rollout_contract(config_name: str) -> None:
    template = compose_runtime_config(
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    )["policy"]["infer_rollout"]
    config = compose_runtime_config(config_name)
    rollout = config["policy"]["infer_rollout"]

    assert set(rollout) == set(template)
    assert rollout["enabled"] is False
    assert rollout["output_dir"] == FASTWAM_ROLLOUT_OUTPUT_DIRS[config_name]
    assert rollout["task"] == config["policy"]["params"]["task"]
    for field in set(template) - {"enabled", "output_dir", "task"}:
        assert rollout[field] == template[field]


@pytest.mark.config
@pytest.mark.parametrize("config_name", DIRECT_FASTWAM_CONFIGS)
def test_fastwam_configs_require_recording_runtime_profile(config_name: str) -> None:
    roles = required_runtime_roles(config_name)

    assert roles
    assert "recording" in roles


@pytest.mark.config
@pytest.mark.parametrize("relative_path", CAMERA_DEVICE_PRESETS)
def test_camera_device_presets_temporarily_allow_any_usb_speed(
    repo_root: Path,
    relative_path: str,
) -> None:
    preset = load_yaml_config(repo_root / relative_path)
    requirements = [
        device["usb_port_requirement"]
        for device in preset["devices"]
        if device["class"] == "WebCameraOpenPi"
    ]

    assert requirements
    assert requirements == ["any"] * len(requirements)


@pytest.mark.config
def test_openpi_and_fastwam_keep_distinct_camera_device_presets(
    repo_root: Path,
) -> None:
    openpi = compose_runtime_config(
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    )
    fastwam = compose_runtime_config(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    )

    assert [
        device["usb_port_requirement"] for device in openpi["devices"][:2]
    ] == ["any", "any"]
    assert [
        device["usb_port_requirement"] for device in fastwam["devices"][:2]
    ] == ["any", "any"]
    assert _device_group(
        repo_root / "configs/policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam.yaml"
    ) == "infer_rizon_fisheye_umi3_dual_fastwam"
    assert _device_group(
        repo_root / "configs/policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi.yaml"
    ) == "infer_rizon_fisheye_umi3_dual_openpi"
    assert load_yaml_config(
        repo_root / "configs/devices/infer/infer_rizon_fisheye_umi3_dual_fastwam.yaml"
    )["metadata"]["device_profile"] == "fastwam"
    assert load_yaml_config(
        repo_root / "configs/devices/infer/infer_rizon_fisheye_umi3_dual_openpi.yaml"
    )["metadata"]["device_profile"] == "openpi"


@pytest.mark.config
@pytest.mark.parametrize(
    ("config_name", "variant", "history_frames", "infer_mode"),
    (
        (
            "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
            "fastwam",
            1,
            "observed_frame_cache",
        ),
        (
            "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
            "fastwam-memory",
            4,
            "observed_frame_cache",
        ),
        (
            "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
            "fasterwam",
            1,
            "one_pass_future_cache",
        ),
    ),
)
def test_fastwam_variants_declare_native_runtime_contract(
    config_name: str,
    variant: str,
    history_frames: int,
    infer_mode: str,
) -> None:
    config = compose_runtime_config(config_name)
    params = config["policy"]["params"]

    assert config["metadata"]["model_variant"] == variant
    assert params["model_variant"] == variant
    assert params["condition_history_frames"] == history_frames
    assert params["obs_horizon"] == history_frames
    assert params["action_infer_mode"] == infer_mode
    assert params["enforce_checkpoint_infer_mode"] is True
    assert isinstance(params["ckpt_path"], str) and params["ckpt_path"]
    assert isinstance(params["task"], str) and params["task"]
    assert "${oc.env:" not in params["ckpt_path"]
    assert "${oc.env:" not in params["task"]


@pytest.mark.config
def test_fasterwam_preserves_the_independent_c4_video_schedule() -> None:
    params = compose_runtime_config(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam"
    )["policy"]["params"]

    assert params["num_inference_steps"] == 10
    assert params["c4_video_denoise_steps"] == 0
    assert params["c4_video_num_inference_steps"] == 2
    assert params["tcp_umi_json_path"] == "calibration/tcp_to_umi_transform.json"
