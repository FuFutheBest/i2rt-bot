"""Device-manager preset cleanup contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.hydra_config import compose_hydra_application_config
from utils.synchronization import BINARY_MATCH, normalize_sync_mode


DEVICE_MANAGER_CONFIGS = [
    ("i2rt_data_collect_async", "base", None, 100, BINARY_MATCH),
    ("config", "synchronized", 0, 100, BINARY_MATCH),
    ("policy_infer/diffusion/config_infer_rizon_eef_dual", "synchronized", 2, 500, BINARY_MATCH),
    ("policy_infer/diffusion/config_infer_rizon_eef_single", "synchronized", 1, 100, BINARY_MATCH),
    ("policy_infer/diffusion/config_infer_iphone_umi_dual", "synchronized", 2, 100000, BINARY_MATCH),
    ("policy_infer/diffusion/config_infer_iphone_umi_single", "synchronized", 1, 100000, BINARY_MATCH),
    (
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
        "synchronized",
        2,
        500,
        BINARY_MATCH,
    ),
    (
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
        "synchronized",
        0,
        500,
        "camera_align",
    ),
    (
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
        "synchronized",
        0,
        500,
        "camera_align",
    ),
    (
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
        "synchronized",
        0,
        500,
        "camera_align",
    ),
    (
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi",
        "synchronized",
        0,
        500,
        "camera_align",
    ),
    (
        "policy_infer/robotwin/config_infer_robotwin_replay_joint14",
        "synchronized",
        1,
        300,
        BINARY_MATCH,
    ),
]


@pytest.mark.config
@pytest.mark.parametrize(
    ("config_name", "manager_type", "master_index", "buffer_size", "sync_mode"),
    DEVICE_MANAGER_CONFIGS,
)
def test_device_manager_presets_preserve_runtime_contract(
    repo_root: Path,
    config_name: str,
    manager_type: str,
    master_index: int | None,
    buffer_size: int,
    sync_mode: str,
) -> None:
    """Hardware-specific settings belong to the composed launch config."""
    config = compose_hydra_application_config(
        config_name,
        config_dir=repo_root / "configs",
        job_name="device_manager_preset_contract",
    )
    manager = config["device_manager"]

    assert manager["type"] == manager_type
    assert manager.get("master_device_class_id") == master_index
    assert manager["params"]["buffer_size"] == buffer_size
    assert normalize_sync_mode(manager.get("sync_mode", BINARY_MATCH)) == sync_mode
    assert "connect_retry" not in manager["params"]
