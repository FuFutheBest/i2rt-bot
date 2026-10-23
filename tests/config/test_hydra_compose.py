"""Hydra composition contract tests for representative runtime configs.

Author: OpenAI Assistant
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from incubator.control_panel.runtime_config import (
    DEFAULT_CONFIG_DIR,
    compose_runtime_config,
    extract_execution_profile,
    list_runtime_configs,
)
from utils.config_parser import normalize_hydra_config
from utils.hydra_config import compose_hydra_application_config

REPRESENTATIVE_CONFIGS = [
    "policy_infer/diffusion/config_infer_rizon_eef_dual_abs_mode_umi",
    "policy_infer/diffusion/config_infer_iphone_umi_dual",
    "policy_infer/diffusion/config_infer_iphone_umi_single",
    "windows_async",
]

INFERENCE_CONFIGS_WITH_PROJECT_LOGS = [
    "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
    "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
]


@pytest.mark.config
@pytest.mark.parametrize("config_name", REPRESENTATIVE_CONFIGS)
def test_representative_configs_compose_and_resolve(
    repo_root: Path, config_name: str
) -> None:
    """Representative configs should compose without starting hardware."""
    with initialize_config_dir(
        config_dir=str(repo_root / "configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name=config_name)

    assert "defaults" not in cfg
    assert "runtime" in cfg or "signal_files" in cfg


@pytest.mark.config
def test_default_config_composes(repo_root: Path) -> None:
    """The default config should compose and resolve without missing group references."""
    with initialize_config_dir(
        config_dir=str(repo_root / "configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name="config")
        OmegaConf.resolve(cfg)


@pytest.mark.config
@pytest.mark.parametrize("config_name", INFERENCE_CONFIGS_WITH_PROJECT_LOGS)
def test_inference_hydra_job_logs_stay_under_project_logs(
    repo_root: Path, config_name: str
) -> None:
    with initialize_config_dir(
        config_dir=str(repo_root / "configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name=config_name, return_hydra_config=True)

    job_logging = OmegaConf.to_container(cfg.hydra.job_logging, resolve=False)
    assert isinstance(job_logging, dict)
    assert job_logging["handlers"]["file"]["filename"] == (
        "${hydra.runtime.cwd}/logs/${hydra.job.name}.log"
    )


@pytest.mark.config
def test_promoted_control_panel_bootstraps_compose_from_root(repo_root: Path) -> None:
    """Control Panel must compose the same shared source as device_starter."""
    assert DEFAULT_CONFIG_DIR == repo_root / "configs"
    native_name = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    rtc_name = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc"
    runtime_configs = set(list_runtime_configs())
    assert native_name in runtime_configs
    assert "control_panel/openpi" not in runtime_configs
    assert {
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc",
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
    } <= runtime_configs
    openpi = compose_runtime_config(native_name)
    openpi_rtc = compose_runtime_config(rtc_name)
    assert openpi["policy"]["params"]["enable_rtc"] is False
    assert openpi_rtc["policy"]["params"]["enable_rtc"] is True
    # Deployment checkpoints may differ between normal and RTC configs. Both
    # entrypoints must resolve the same values for the selected source instead.
    direct_rtc = compose_hydra_application_config(rtc_name, config_dir=repo_root / "configs")
    assert openpi_rtc["policy"]["params"] == direct_rtc["policy"]["params"]
    assert extract_execution_profile(openpi)["policy"]["checkpoint_configured"] is True

    fastwam_name = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam"
    fastwam_rtc_name = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc"
    assert (
        compose_runtime_config(fastwam_name)["policy"]["params"]["enable_rtc"] is True
    )
    assert (
        compose_runtime_config(fastwam_rtc_name)["policy"]["params"]["enable_rtc"]
        is True
    )

    fastwam = compose_runtime_config(fastwam_name)
    assert fastwam["policy"]["params"]["crop_input_to_aspect"] == [3, 4]
    assert extract_execution_profile(fastwam)["policy"]["checkpoint_configured"] is True
    assert [device["class"] for device in fastwam["devices"]] == [
        "WebCameraOpenPi",
        "WebCameraOpenPi",
        "RizonRobot",
    ]
    assert fastwam["device_manager"]["master_device_class_id"] == 0
    assert fastwam["device_manager"]["sync_mode"] == "camera_align"
    assert fastwam["devices"][2]["state_feedback"]["format"]["shape"] == [44]
    assert fastwam["policy"]["fps"] == 1.5
    assert compose_runtime_config(fastwam_rtc_name)["policy"]["fps"] == 1.5


@pytest.mark.config
def test_openpi_top_level_configs_are_only_normal_and_rtc(repo_root: Path) -> None:
    assert {
        path.name
        for path in (
            repo_root / "configs" / "policy_infer" / "openpi"
        ).glob("*.yaml")
    } == {
        "config_infer_fisheye_umi3_dual_openpi.yaml",
        "config_infer_fisheye_umi3_dual_openpi_rtc.yaml",
    }


@pytest.mark.config
def test_legacy_config_namespace_is_removed(repo_root: Path) -> None:
    assert not (repo_root / "configs" / "legacy").exists()


@pytest.mark.config
def test_panel_and_device_starter_share_normalized_config_shape(
    repo_root: Path,
) -> None:
    """The panel view must match the mapping passed into DeviceStarter."""
    native_name = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    panel_config = compose_runtime_config(native_name, repo_root / "configs")
    with initialize_config_dir(
        config_dir=str(repo_root / "configs"),
        version_base="1.3",
    ):
        cfg = compose(config_name=native_name, return_hydra_config=True)
        # The starter calls this helper from inside its Hydra entrypoint.
        from hydra.core.hydra_config import HydraConfig

        hydra_state = HydraConfig.instance()
        previous = hydra_state.cfg
        try:
            hydra_state.set_config(cfg)
            starter_config = normalize_hydra_config(cfg)
        finally:
            hydra_state.cfg = previous

    assert panel_config == starter_config


@pytest.mark.config
def test_core_composer_matches_control_panel_resolution(repo_root: Path) -> None:
    """Core contracts and the Control Panel must resolve the same config."""
    config_name = "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam"

    assert compose_hydra_application_config(
        config_name,
        config_dir=repo_root / "configs",
        job_name="core_contract_test",
    ) == compose_runtime_config(config_name, repo_root / "configs")


@pytest.mark.config
def test_core_composer_preserves_an_enclosing_hydra_context(repo_root: Path) -> None:
    """A preflight composition must not consume its caller's Hydra context."""
    with initialize_config_dir(
        config_dir=str(repo_root / "configs"),
        version_base="1.3",
        job_name="outer_hydra_context",
    ):
        outer_config = compose(
            config_name="policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
            return_hydra_config=True,
        )
        hydra_state = HydraConfig.instance()
        previous = hydra_state.cfg
        hydra_state.set_config(outer_config)
        outer_hydra_config = hydra_state.cfg
        outer_global_hydra = GlobalHydra.instance()
        try:
            resolved = compose_hydra_application_config(
                "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
                config_dir=repo_root / "configs",
                job_name="nested_core_contract_test",
            )
            assert resolved["metadata"]["model_variant"] == "fasterwam"
            assert hydra_state.cfg is outer_hydra_config
            assert GlobalHydra.instance() is outer_global_hydra
            # Composition through the caller's original GlobalHydra still works.
            assert compose(config_name="policy_infer/diffusion/config_infer_iphone_umi_single")
        finally:
            hydra_state.cfg = previous
